# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import HybridConfig
from data import ProceduralCompositionalDataset, parse_watt_ms_prediction


class DenseCausalTransformerLM(nn.Module):
    def __init__(self, cfg: HybridConfig, n_layers: int = 2):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff_expert * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.layers(h, mask=causal_mask, is_causal=True)
        return self.lm_head(self.final_norm(h))

    @torch.no_grad()
    def generate_ar(self, prompt_text: str, max_new_tokens: int = 24) -> str:
        self.eval()
        prompt_bytes = list(prompt_text.encode("utf-8"))
        seq = [self.cfg.bos_token_id] + prompt_bytes
        x = torch.tensor([seq], dtype=torch.long, device=self.cfg.device)

        for _ in range(max_new_tokens):
            if x.size(1) >= self.cfg.max_seq_len:
                break
            logits = self(x)
            next_tok = int(logits[0, -1].argmax().item())
            if next_tok in (self.cfg.eos_token_id, self.cfg.pad_token_id):
                break
            x = torch.cat([x, torch.tensor([[next_tok]], device=x.device)], dim=1)

        out_bytes = bytearray(t for t in x[0, 1:].tolist() if t < 256)
        return out_bytes.decode("utf-8", errors="replace")


def train_and_evaluate_ar_baseline():
    cfg = HybridConfig()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=3200)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=96)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = DenseCausalTransformerLM(cfg, n_layers=2).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Igualamos el número de pasadas hacia delante/atrás: 2 * cfg.max_steps
    ar_steps = cfg.max_steps * 2
    step = 0
    while step < ar_steps:
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

    # Evaluación Exact Match sobre las 96 combinaciones Held-Out
    watt_em, ms_em, both_em = 0, 0, 0
    t0 = time.perf_counter()
    for prompt_b, _, true_w, true_ms in val_ds.unique_records:
        p_str = prompt_b.decode("utf-8")
        gen = model.generate_ar(p_str)
        _, pw, pms = parse_watt_ms_prediction(gen, p_str)
        watt_em += int(pw == true_w)
        ms_em += int(pms == true_ms)
        both_em += int(pw == true_w and pms == true_ms)
    lat_ms = ((time.perf_counter() - t0) / 96.0) * 1000.0

    params = sum(p.numel() for p in model.parameters())
    print(f"=== Baseline AR Denso Igualado en Cómputo ({params:,} params, {ar_steps} pasos) ===")
    print(
        f"Held-Out (96 tuplas) -> Watt EM: {100*watt_em/96:.1f}% | "
        f"MS EM: {100*ms_em/96:.1f}% | Full Tuple EM: {100*both_em/96:.1f}% | Latencia: {lat_ms:.1f} ms"
    )


if __name__ == "__main__":
    train_and_evaluate_ar_baseline()
