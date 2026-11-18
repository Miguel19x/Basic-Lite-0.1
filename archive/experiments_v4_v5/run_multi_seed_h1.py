# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import argparse
import json
import math
import random
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import HybridConfig
from data import (
    ProceduralCompositionalDataset,
    apply_blockwise_diffusion_corruption,
    parse_watt_ms_prediction,
)
from model import TernaryRecursiveDiffusionLM, BitLinear
from train import compute_v4_hybrid_loss, update_bitnet_schedule
from generate import generate_block_causal_with_kv_cache
from baseline_ar import DenseCausalTransformerLM


def set_all_seeds(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)


def compute_dataset_trivial_ceilings(cfg: HybridConfig) -> Dict[str, float]:
    """Calcula en frío el techo trivial de plantilla+PAD y el de dígito mayoritario."""
    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=384)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=96)

    digit_majority_by_offset: Dict[int, int] = {}
    for offset in range(32):
        col_digits = [
            tgt[offset]
            for _, tgt, _, _ in train_ds.unique_records
            if offset < len(tgt) and 48 <= tgt[offset] <= 57
        ]
        if col_digits:
            digit_majority_by_offset[offset] = Counter(col_digits).most_common(1)[0][0]

    total_pos, non_digit_pos, maj_digit_hits, total_digits = 0, 0, 0, 0
    for prompt_b, target_b, _, _ in val_ds.unique_records:
        resp_len = cfg.max_seq_len - (1 + len(prompt_b))
        total_pos += resp_len
        for i, b in enumerate(target_b):
            if 48 <= b <= 57:
                total_digits += 1
                if digit_majority_by_offset.get(i) == b:
                    maj_digit_hits += 1
            else:
                non_digit_pos += 1
        # EOS + PAD posteriores
        non_digit_pos += resp_len - len(target_b)

    return {
        "train_unique_tuples": len(train_ds.unique_records),
        "val_unique_tuples": len(val_ds.unique_records),
        "trivial_template_pad_acc_pct": 100.0 * non_digit_pos / max(1, total_pos),
        "trivial_majority_all_acc_pct": 100.0 * (non_digit_pos + maj_digit_hits) / max(1, total_pos),
        "trivial_majority_digit_acc_pct": 100.0 * maj_digit_hits / max(1, total_digits),
    }


def mcnemar_exact_pvalue(hits_baseline: List[bool], hits_variant: List[bool]) -> float:
    """
    Prueba binomial exacta de McNemar (dos colas) sobre pares discordantes:
    b = casos donde baseline falla y variante acierta
    c = casos donde baseline acierta y variante falla
    """
    assert len(hits_baseline) == len(hits_variant)
    b = sum(1 for h0, h1 in zip(hits_baseline, hits_variant) if (not h0) and h1)
    c = sum(1 for h0, h1 in zip(hits_baseline, hits_variant) if h0 and (not h1))
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Suma acumulada binomial exacta 2 * sum_{i=0}^k C(n, i) * 0.5^n
    tail_prob = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail_prob)


@torch.no_grad()
def evaluate_hybrid_records_detailed(
    model: TernaryRecursiveDiffusionLM,
    records: List[Tuple[bytes, bytes, int, int]],
    cfg: HybridConfig,
    r_infer: int,
) -> Dict:
    model.prepare_for_eval()
    n = len(records)
    fmt_ok, watt_em, ms_em, both_em = 0, 0, 0, 0
    dig_correct, dig_total = 0, 0
    both_hits_per_item: List[bool] = []
    watt_hits_per_item: List[bool] = []
    ms_hits_per_item: List[bool] = []

    t0 = time.perf_counter()
    for prompt_b, target_b, true_w, true_ms in records:
        p_str = prompt_b.decode("utf-8")
        t_str = target_b.decode("utf-8")
        gen_text, _, _ = generate_block_causal_with_kv_cache(
            model, p_str, cfg, num_recursions_override=r_infer
        )
        valid_fmt, pred_w, pred_ms = parse_watt_ms_prediction(gen_text, p_str)

        w_ok = (pred_w == true_w)
        m_ok = (pred_ms == true_ms)
        b_ok = (w_ok and m_ok)

        fmt_ok += int(valid_fmt)
        watt_em += int(w_ok)
        ms_em += int(m_ok)
        both_em += int(b_ok)

        watt_hits_per_item.append(w_ok)
        ms_hits_per_item.append(m_ok)
        both_hits_per_item.append(b_ok)

        continuation = gen_text[len(p_str):] if gen_text.startswith(p_str) else ""
        for idx_c, ch in enumerate(t_str):
            if ch.isdigit():
                dig_total += 1
                if idx_c < len(continuation) and continuation[idx_c] == ch:
                    dig_correct += 1

    latency_ms = ((time.perf_counter() - t0) / max(1, n)) * 1000.0
    return {
        "n": n,
        "fmt_pct": 100.0 * fmt_ok / max(1, n),
        "watt_em_pct": 100.0 * watt_em / max(1, n),
        "ms_em_pct": 100.0 * ms_em / max(1, n),
        "both_em_pct": 100.0 * both_em / max(1, n),
        "digit_acc_pct": 100.0 * dig_correct / max(1, dig_total),
        "latency_ms": latency_ms,
        "both_hits": both_hits_per_item,
        "watt_hits": watt_hits_per_item,
        "ms_hits": ms_hits_per_item,
    }


def train_single_hybrid_seed(
    seed: int,
    randomize_r: bool,
    max_steps: int,
    cpu_threads: int,
) -> Tuple[TernaryRecursiveDiffusionLM, HybridConfig, float]:
    cfg = HybridConfig(
        seed=seed,
        cpu_threads=cpu_threads,
        max_steps=max_steps,
        randomize_train_recursions=randomize_r,
    )
    set_all_seeds(seed, cpu_threads)

    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=max_steps * cfg.batch_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=cfg.betas)

    t0 = time.perf_counter()
    step = 0
    for x_0, can_mask, digit_mask in train_loader:
        step += 1
        update_bitnet_schedule(optimizer, step, cfg)
        r_train = random.choice([1, 2, 3]) if cfg.randomize_train_recursions else cfg.num_recursions

        # Pasada 1 (ruido alto t1)
        x_t1, mask_pos1, t1 = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
        sup1 = mask_pos1 if cfg.lazy_vocab_projection else None
        logits1, inter1, halt1, _, carried1 = model(x_t1, t1, num_recursions=r_train, supervised_mask=sup1)
        loss1, _ = compute_v4_hybrid_loss(logits1, inter1, halt1, x_0, mask_pos1, digit_mask, t1, cfg)

        # Pasada 2 encadenada (ruido menor t2, con predicción del modelo y sin filtración de x_0)
        if logits1.ndim == 2:
            preds_sup1 = logits1.detach().argmax(dim=-1)
            preds1 = x_t1.clone()
            preds1[mask_pos1] = preds_sup1
        else:
            preds1 = logits1.detach().argmax(dim=-1)

        frac_keep = torch.empty_like(t1).uniform_(0.35, 0.85)
        keep_cond = (torch.rand_like(x_0, dtype=torch.float32) < frac_keep.unsqueeze(1)) if t1.dim() == 1 else (torch.rand_like(x_0, dtype=torch.float32) < frac_keep)
        keep_mask = keep_cond & mask_pos1
        for b in range(x_0.size(0)):
            if not keep_mask[b].any() and mask_pos1[b].any():
                idx_m = torch.nonzero(mask_pos1[b], as_tuple=False).squeeze(-1)
                keep_mask[b, idx_m[0]] = True

        x_t2 = x_t1.clone()
        revealed_pos = mask_pos1 & ~keep_mask
        x_t2[revealed_pos] = preds1[revealed_pos]
        x_t2[keep_mask] = cfg.mask_token_id
        t2 = (t1 * frac_keep).clamp(min=0.05)
        dt = (t1 - t2).clamp(min=0.01)

        sup2 = keep_mask if cfg.lazy_vocab_projection else None
        logits2, inter2, halt2, _, _ = model(
            x_t2, t2, delta_t=dt, prev_states=carried1, num_recursions=r_train, supervised_mask=sup2
        )
        loss2, metrics2 = compute_v4_hybrid_loss(logits2, inter2, halt2, x_0, keep_mask, digit_mask, t2, cfg)

        total_loss = 0.45 * loss1 + 0.55 * loss2
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if torch.isfinite(grad_norm):
            optimizer.step()

        if step % max(1, max_steps // 4) == 0 or step == max_steps:
            mode_tag = "R_rand{1,2,3}" if randomize_r else f"R_fix={cfg.num_recursions}"
            print(
                f"  [{mode_tag} | Seed {seed}] Paso {step:03d}/{max_steps} | "
                f"Loss: {total_loss.item():.3f} | TrainDigAcc: {100*metrics2['train_digit_acc']:.1f}%",
                flush=True,
            )
        if step >= max_steps:
            break

    train_sec = time.perf_counter() - t0
    return model, cfg, train_sec


def train_and_eval_ar_seed(seed: int, max_steps: int, cpu_threads: int) -> Dict:
    cfg = HybridConfig(seed=seed, cpu_threads=cpu_threads, max_steps=max_steps)
    set_all_seeds(seed, cpu_threads)

    ar_steps = max_steps * 2  # Igualado en pasadas con las 2 pasadas por paso del modelo híbrido
    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=ar_steps * cfg.batch_size)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=96)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = DenseCausalTransformerLM(cfg, n_layers=2).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    step = 0
    for x_0, can_mask, digit_mask in train_loader:
        step += 1
        inp = x_0[:, :-1]
        tgt = x_0[:, 1:]
        sup_mask = can_mask[:, 1:].float()
        dig_m = digit_mask[:, 1:]

        weights = torch.where(dig_m, torch.full_like(sup_mask, cfg.digit_token_weight), sup_mask)
        logits = model(inp)
        loss_tok = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1), reduction="none").view_as(tgt)
        loss = (loss_tok * weights).sum() / weights.sum().clamp(min=1.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if step >= ar_steps:
            break

    def _eval_ar_records(records: List[Tuple[bytes, bytes, int, int]]) -> Dict:
        n = len(records)
        fmt_ok, w_em, m_em, b_em, d_ok, d_tot = 0, 0, 0, 0, 0, 0
        hits: List[bool] = []
        t0 = time.perf_counter()
        for prompt_b, target_b, tw, tms in records:
            p_str = prompt_b.decode("utf-8")
            t_str = target_b.decode("utf-8")
            gen = model.generate_ar(p_str)
            v_fmt, pw, pms = parse_watt_ms_prediction(gen, p_str)
            fmt_ok += int(v_fmt)
            w_em += int(pw == tw)
            m_em += int(pms == tms)
            both_ok = (pw == tw and pms == tms)
            b_em += int(both_ok)
            hits.append(both_ok)

            cont = gen[len(p_str):] if gen.startswith(p_str) else ""
            for idx_c, ch in enumerate(t_str):
                if ch.isdigit():
                    d_tot += 1
                    if idx_c < len(cont) and cont[idx_c] == ch:
                        d_ok += 1
        lat = ((time.perf_counter() - t0) / max(1, n)) * 1000.0
        return {
            "fmt_pct": 100.0 * fmt_ok / n,
            "watt_em_pct": 100.0 * w_em / n,
            "ms_em_pct": 100.0 * m_em / n,
            "both_em_pct": 100.0 * b_em / n,
            "digit_acc_pct": 100.0 * d_ok / max(1, d_tot),
            "latency_ms": lat,
            "both_hits": hits,
        }

    train_mem = _eval_ar_records(train_ds.unique_records[:96])
    val_ood = _eval_ar_records(val_ds.unique_records)
    print(
        f"  [Baseline AR | Seed {seed}] Train Full-EM: {train_mem['both_em_pct']:.1f}% | "
        f"Held-Out Full-EM: {val_ood['both_em_pct']:.1f}% (DigAcc: {val_ood['digit_acc_pct']:.1f}%)",
        flush=True,
    )
    return {"train_mem": train_mem, "val_ood": val_ood}


def summarize_across_seeds(seed_metrics_list: List[Dict], key: str) -> Tuple[float, float, float]:
    """Devuelve (media, desviación_estándar_muestral, sem_ic95) entre semillas."""
    vals = [m[key] for m in seed_metrics_list]
    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    ic95_pooled = 1.96 * (std_v / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean_v, std_v, ic95_pooled


def main():
    parser = argparse.ArgumentParser(description="Protocolo Multi-Semilla H1 y Paridad Baseline AR (TRDM-v4)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2026], help="Lista de semillas")
    parser.add_argument("--steps", type=int, default=800, help="Pasos de entrenamiento por semilla")
    parser.add_argument("--threads", type=int, default=2, help="Hilos de CPU en Linux")
    parser.add_argument("--skip-random-r", action="store_true", help="Omitir ablación con R_train aleatorio")
    parser.add_argument("--skip-ar", action="store_true", help="Omitir baseline autorregresivo")
    parser.add_argument("--out-json", type=str, default="multi_seed_h1_results.json", help="Ruta JSON de salida")
    args = parser.parse_args()

    base_cfg = HybridConfig(cpu_threads=args.threads, max_steps=args.steps)
    ceilings = compute_dataset_trivial_ceilings(base_cfg)

    print("====================================================================================")
    print(" AUDITORÍA PREVIA DEL DATASET Y TECHOS TRIVIALES")
    print("====================================================================================")
    print(f" Universo composicional : {ceilings['train_unique_tuples']} Train (80.0%) | {ceilings['val_unique_tuples']} Held-Out (20.0%)")
    print(f" Techo trivial plantilla: {ceilings['trivial_template_pad_acc_pct']:.2f}% (acertar todo salvo dígitos)")
    print(f" Techo mayoría global   : {ceilings['trivial_majority_all_acc_pct']:.2f}% (plantilla + dígito más frecuente)")
    print(f" Techo mayoría dígitos  : {ceilings['trivial_majority_digit_acc_pct']:.2f}% (solo en bytes '0'..'9')")
    print(f" Semillas configuradas  : {args.seeds} -> {len(args.seeds) * 96} evaluaciones Held-Out por punto")
    print("====================================================================================\n", flush=True)

    regimes_to_run = [("fixed_R2", False)]
    if not args.skip_random_r:
        regimes_to_run.append(("random_R123", True))

    all_results = {"ceilings": ceilings, "seeds": args.seeds, "steps": args.steps, "regimes": {}, "baseline_ar": {}}
    r_infer_grid = (1, 2, 4, 8)

    for regime_name, rand_r in regimes_to_run:
        print(f">>> Ejecutando Régimen Híbrido: {regime_name} (randomize_train_recursions={rand_r}) <<<", flush=True)
        train_mem_per_seed: List[Dict] = []
        ood_per_r: Dict[int, List[Dict]] = {r: [] for r in r_infer_grid}

        for seed in args.seeds:
            model, cfg_s, _ = train_single_hybrid_seed(seed, rand_r, args.steps, args.threads)
            train_ds = ProceduralCompositionalDataset(cfg_s, split="train", num_samples=96)
            val_ds = ProceduralCompositionalDataset(cfg_s, split="val", num_samples=96)

            mem_res = evaluate_hybrid_records_detailed(model, train_ds.unique_records[:96], cfg_s, r_infer=2)
            train_mem_per_seed.append(mem_res)
            print(
                f"  [{regime_name} | Seed {seed}] Memorización Train (96 tuplas, R=2) -> "
                f"Full-EM: {mem_res['both_em_pct']:.1f}% | DigAcc: {mem_res['digit_acc_pct']:.1f}%",
                flush=True,
            )

            for r_inf in r_infer_grid:
                ood_res = evaluate_hybrid_records_detailed(model, val_ds.unique_records, cfg_s, r_infer=r_inf)
                ood_per_r[r_inf].append(ood_res)

        # Resumen estadístico y test de McNemar emparejado contra R_infer = 2 (umbral Bonferroni alpha = 0.0167)
        pooled_r2_hits = [h for m in ood_per_r[2] for h in m["both_hits"]]
        regime_summary = {"train_memorization_R2": {}, "ood_by_R": {}}

        for metric_k in ("fmt_pct", "digit_acc_pct", "watt_em_pct", "ms_em_pct", "both_em_pct"):
            mean_v, std_v, _ = summarize_across_seeds(train_mem_per_seed, metric_k)
            regime_summary["train_memorization_R2"][metric_k] = {"mean": mean_v, "std": std_v}

        for r_inf in r_infer_grid:
            pooled_hits = [h for m in ood_per_r[r_inf] for h in m["both_hits"]]
            total_n = len(pooled_hits)
            pooled_p = sum(pooled_hits) / max(1, total_n)
            pooled_ic95 = 196.0 * math.sqrt(pooled_p * (1.0 - pooled_p) / max(1, total_n))
            p_mcnemar = mcnemar_exact_pvalue(pooled_r2_hits, pooled_hits) if r_inf != 2 else 1.0

            r_entry = {
                "total_ood_cases": total_n,
                "pooled_both_em_ic95": pooled_ic95,
                "mcnemar_p_vs_R2": p_mcnemar,
                "bonferroni_significant": bool(r_inf != 2 and p_mcnemar < (0.05 / 3.0)),
            }
            for metric_k in ("fmt_pct", "digit_acc_pct", "watt_em_pct", "ms_em_pct", "both_em_pct", "latency_ms"):
                mean_v, std_v, _ = summarize_across_seeds(ood_per_r[r_inf], metric_k)
                r_entry[metric_k] = {"mean": mean_v, "std": std_v}
            regime_summary["ood_by_R"][str(r_inf)] = r_entry

        all_results["regimes"][regime_name] = regime_summary
        print("", flush=True)

    if not args.skip_ar:
        print(f">>> Ejecutando Baseline Autorregresivo Denso ({args.steps * 2} pasos por semilla) <<<", flush=True)
        ar_train_list, ar_ood_list = [], []
        for seed in args.seeds:
            ar_out = train_and_eval_ar_seed(seed, args.steps, args.threads)
            ar_train_list.append(ar_out["train_mem"])
            ar_ood_list.append(ar_out["val_ood"])

        ar_summary = {"train_memorization": {}, "ood": {}}
        for metric_k in ("fmt_pct", "digit_acc_pct", "watt_em_pct", "ms_em_pct", "both_em_pct", "latency_ms"):
            m_tr, s_tr, _ = summarize_across_seeds(ar_train_list, metric_k)
            m_od, s_od, _ = summarize_across_seeds(ar_ood_list, metric_k)
            ar_summary["train_memorization"][metric_k] = {"mean": m_tr, "std": s_tr}
            ar_summary["ood"][metric_k] = {"mean": m_od, "std": s_od}
        all_results["baseline_ar"] = ar_summary
        print("", flush=True)

    # Guardar JSON completo sin las listas crudas de booleanos
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Imprimir tabla Markdown lista para README.md
    print("====================================================================================")
    print(" TABLA CONSOLIDADA MULTI-SEMILLA PARA README.md (N = 288 EVALUACIONES OOD POR FILA)")
    print("====================================================================================")
    print("| Modelo / Régimen | $R_{\\text{infer}}$ | Train Full-EM | OOD DigAcc | OOD Watt EM | OOD MS EM | OOD Full-EM ($\\pm$ DE) | $p$ McNemar vs $R=2$ | Latencia |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for regime_name, reg_data in all_results["regimes"].items():
        tr_em = reg_data["train_memorization_R2"]["both_em_pct"]
        tr_str = f"{tr_em['mean']:.1f}% ± {tr_em['std']:.1f}%"
        for r_str, r_data in reg_data["ood_by_R"].items():
            sig_mark = " **" if r_data["bonferroni_significant"] else ""
            p_str = "—" if r_str == "2" else f"{r_data['mcnemar_p_vs_R2']:.4f}{sig_mark}"
            print(
                f"| `{regime_name}` | {r_str} | {tr_str} | "
                f"{r_data['digit_acc_pct']['mean']:.1f}% ± {r_data['digit_acc_pct']['std']:.1f}% | "
                f"{r_data['watt_em_pct']['mean']:.1f}% | {r_data['ms_em_pct']['mean']:.1f}% | "
                f"**{r_data['both_em_pct']['mean']:.1f}% ± {r_data['both_em_pct']['std']:.1f}%** "
                f"(±{r_data['pooled_both_em_ic95']:.1f}% IC95) | {p_str} | "
                f"{r_data['latency_ms']['mean']:.1f} ms |"
            )

    if all_results["baseline_ar"]:
        ar_tr = all_results["baseline_ar"]["train_memorization"]["both_em_pct"]
        ar_od = all_results["baseline_ar"]["ood"]
        print(
            f"| `Baseline AR Denso` | 1 (AR) | {ar_tr['mean']:.1f}% ± {ar_tr['std']:.1f}% | "
            f"{ar_od['digit_acc_pct']['mean']:.1f}% ± {ar_od['digit_acc_pct']['std']:.1f}% | "
            f"{ar_od['watt_em_pct']['mean']:.1f}% | {ar_od['ms_em_pct']['mean']:.1f}% | "
            f"**{ar_od['both_em_pct']['mean']:.1f}% ± {ar_od['both_em_pct']['std']:.1f}%** | — | "
            f"{ar_od['latency_ms']['mean']:.1f} ms |"
        )
    print(f"\n[Exportado] Resultados detallados guardados en: {args.out_json}")


if __name__ == "__main__":
    main()
