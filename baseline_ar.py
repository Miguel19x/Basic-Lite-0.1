import math
import time
import resource
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import HybridConfig
from data import ProceduralCompositionalDataset


class DenseCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_cache = (k.detach(), v.detach())
        total_len = k.size(2)

        # Causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if L > 1:
            mask = torch.triu(torch.full((L, total_len), float("-inf"), device=x.device), diagonal=start_pos + 1)
            scores = scores + mask

        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out), new_cache


class DenseTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = DenseCausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class DenseAutoregressiveLM(nn.Module):
    """
    Modelo autorregresivo denso de referencia (Baseline) calibrado a ~870k parámetros.
    Usa atención causal estándar y generación autorregresiva token por token.
    """
    def __init__(self, vocab_size: int = 260, d_model: int = 192, n_heads: int = 4, n_layers: int = 2, d_ff: int = 768):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DenseTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        h = self.token_emb(x)
        new_caches = []
        for i, layer in enumerate(self.layers):
            c_i = kv_caches[i] if kv_caches is not None else None
            h, updated_c = layer(h, kv_cache=c_i, start_pos=start_pos)
            new_caches.append(updated_c)
        logits = self.lm_head(self.final_norm(h))
        return logits, new_caches


def train_dense_baseline(cfg: HybridConfig, max_steps: int = 300) -> DenseAutoregressiveLM:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=2400)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = DenseAutoregressiveLM().to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Baseline AR Denso] Parámetros totales: {total_params:,} (FP32) | Dispositivo: {cfg.device}")

    t0 = time.perf_counter()
    step = 0
    while step < max_steps:
        for x_0, can_mask in train_loader:
            step += 1
            # Entrenamiento autorregresivo estándar: predecir token t+1 dado 0..t
            input_seq = x_0[:, :-1].to(cfg.device)
            target_seq = x_0[:, 1:].to(cfg.device)

            logits, _ = model(input_seq)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), target_seq.reshape(-1), ignore_index=cfg.pad_token_id)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 50 == 0 or step == 1:
                elapsed = time.perf_counter() - t0
                print(f"Paso {step:03d}/{max_steps} | Loss AR: {loss.item():.4f} | Tiempo: {elapsed:.1f}s")

            if step >= max_steps:
                break

    return model


@torch.no_grad()
def generate_ar(model: DenseAutoregressiveLM, prompt_text: str, cfg: HybridConfig, max_new_tokens: int = 24) -> Tuple[str, float]:
    model.eval()
    prompt_bytes = [cfg.bos_token_id] + list(prompt_text.encode("utf-8"))
    x = torch.tensor([prompt_bytes], dtype=torch.long, device=cfg.device)

    t0 = time.perf_counter()
    logits, kv_caches = model(x, start_pos=0)
    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_tok.item()]

    curr_pos = x.size(1)
    for _ in range(max_new_tokens - 1):
        if next_tok.item() in (cfg.eos_token_id, cfg.pad_token_id):
            break
        logits, kv_caches = model(next_tok, kv_caches=kv_caches, start_pos=curr_pos)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())
        curr_pos += 1

    latency_ms = (time.perf_counter() - t0) * 1000.0

    resp_bytes = []
    for t in generated:
        if t in (cfg.eos_token_id, cfg.pad_token_id):
            break
        if t < 256:
            resp_bytes.append(t)
    return bytes(resp_bytes).decode("utf-8", errors="replace"), latency_ms


def evaluate_baseline_held_out(model: DenseAutoregressiveLM, cfg: HybridConfig):
    from generate import decode_tokens_clean
    held_out_cases = [
        ("nodo: server\nperfil: turbo\nred: dual\ncarga: pico\n", "watt: 87\nms: 16"),
        ("nodo: compute\nperfil: eco\nred: simple\ncarga: pico\n", "watt: 22\nms: 40"),
        ("nodo: router\nperfil: edge\nred: mesh\ncarga: pico\n", "watt: 53\nms: 7"),
        ("nodo: storage\nperfil: batch\nred: triple\ncarga: media\n", "watt: 35\nms: 22"),
    ]

    print("\n--- Evaluación del Baseline AR Denso en Casos Held-Out ---")
    total_latency = 0.0
    for idx, (prompt, target) in enumerate(held_out_cases, 1):
        salida, lat = generate_ar(model, prompt, cfg)
        total_latency += lat
        acierto = (salida.strip() == target.strip())
        icono = "✓ EXACTO" if acierto else "✗ DIF"
        print(f"Caso #{idx} | {icono} | Latencia: {lat:.1f} ms | Salida: {repr(salida.strip())} | Esperado: {repr(target)}")

    avg_lat = total_latency / len(held_out_cases)
    print(f"Latencia promedio por respuesta AR: {avg_lat:.1f} ms")


if __name__ == "__main__":
    cfg = HybridConfig()
    model = train_dense_baseline(cfg, max_steps=300)
    evaluate_baseline_held_out(model, cfg)
    torch.save(model.state_dict(), "dense_ar_baseline.pt")
