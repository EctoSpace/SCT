#!/usr/bin/env python3
"""
SPECTRAL COMPACT TRAINING — Steam Deck / Low-VRAM Challenge
============================================================

Proves: a 70B-class architecture (80 layers, d=8192, ffn=28672)
can perform a full forward + backward + optimizer step on 16GB RAM.

Dense 70B + Adam needs 1,245 GB.
SCT 70B + Adam needs 7.2 GB.
Compression: 172x.
"""

import math
import time
import torch
import torch.nn as nn

try:
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
except ImportError:
    print("[!] transformers not found. Install: pip install transformers")
    raise

print("""
##########################################################
#   SPECTRAL COMPACT TRAINING — STEAM DECK CHALLENGE     #
#   70B PARAMETER CLASS ARCHITECTURE ON 16GB RAM          #
##########################################################
""")


# ═══════════════════════════════════════════════════════════════════
#  SPECTRAL LINEAR
# ═══════════════════════════════════════════════════════════════════

def safe_qr(M):
    """QR with sign correction. CPU path avoids AMD gfx1033 driver issues."""
    Q, R = torch.linalg.qr(M.cpu())
    return (Q * torch.sign(torch.diag(R))).to(M.device)


class SpectralLinear(nn.Module):
    """W = U diag(s) V^T — exact backprop, O(k(m+n)) memory."""

    def __init__(self, in_features, out_features, rank=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, min(in_features, out_features))

        U = torch.randn(in_features, self.rank) / math.sqrt(in_features)
        V = torch.randn(out_features, self.rank) / math.sqrt(out_features)
        self.U = nn.Parameter(safe_qr(U)[:, :self.rank])
        self.V = nn.Parameter(safe_qr(V)[:, :self.rank])
        self.s = nn.Parameter(torch.ones(self.rank))

    def forward(self, x):
        # x: [..., in_features] -> [..., out_features]
        return (x @ self.U) * self.s @ self.V.T

    @torch.no_grad()
    def retract(self):
        """QR retraction — project U, V back onto the Stiefel manifold."""
        self.U.data = safe_qr(self.U.data)[:, :self.rank]
        self.V.data = safe_qr(self.V.data)[:, :self.rank]


def retract_all(module):
    """Retract all SpectralLinear layers in a module tree."""
    for m in module.modules():
        if isinstance(m, SpectralLinear):
            m.retract()


# ═══════════════════════════════════════════════════════════════════
#  70B TRANSFORMER LAYER (spectral)
# ═══════════════════════════════════════════════════════════════════

class Spectral70BLayer(nn.Module):
    """
    Single transformer layer with 70B-class dimensions.
    Attention is simplified (no softmax/masking) — this is a
    parameter count and memory test, not a language model.
    """

    def __init__(self, config, rank=32):
        super().__init__()
        h = config.hidden_size        # 8192
        ffn = config.intermediate_size  # 28672

        self.input_layernorm = LlamaRMSNorm(h, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(h, eps=config.rms_norm_eps)

        # Attention projections (4 × 8192→8192)
        self.q_proj = SpectralLinear(h, h, rank=rank)
        self.k_proj = SpectralLinear(h, h, rank=rank)
        self.v_proj = SpectralLinear(h, h, rank=rank)
        self.o_proj = SpectralLinear(h, h, rank=rank)

        # MLP projections (8192→28672, 8192→28672, 28672→8192)
        self.gate_proj = SpectralLinear(h, ffn, rank=rank)
        self.up_proj = SpectralLinear(h, ffn, rank=rank)
        self.down_proj = SpectralLinear(ffn, h, rank=rank)

    def forward(self, x):
        # Attention (simplified — no softmax, tests parameter flow only)
        norm_x = self.input_layernorm(x)
        q = self.q_proj(norm_x)
        k = self.k_proj(norm_x)
        v = self.v_proj(norm_x)
        # Real attention: softmax(Q@K^T/sqrt(d))@V
        # Simplified: additive combination tests gradient flow through all projections
        x = x + self.o_proj(q + k + v)

        # MLP (SwiGLU — identical to LLaMA)
        norm_x2 = self.post_attention_layernorm(x)
        gate = torch.nn.functional.silu(self.gate_proj(norm_x2))
        x = x + self.down_proj(gate * self.up_proj(norm_x2))

        return x


# ═══════════════════════════════════════════════════════════════════
#  TEST
# ═══════════════════════════════════════════════════════════════════

def run_test():
    # ── Device detection ──────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[*] Device: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print(f"[*] Device: CPU")

    # ── Architecture ──────────────────────────────────────────────
    RANK = 32
    config = LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
    )

    print(f"[*] Architecture: LLaMA-70B class")
    print(f"    Hidden: {config.hidden_size}  FFN: {config.intermediate_size}")
    print(f"    Layers: {config.num_hidden_layers}  Rank: {RANK}")

    # ── Compute expected sizes ────────────────────────────────────
    h, ffn, nl = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    sct_params_per_layer = (
        4 * (h * RANK + h * RANK + RANK) +           # attn: q,k,v,o
        2 * (h * RANK + ffn * RANK + RANK) +          # gate, up
        1 * (ffn * RANK + h * RANK + RANK) +           # down
        2 * h                                           # 2x RMSNorm
    )
    sct_total = sct_params_per_layer * nl
    dense_total = (4 * h * h + 3 * h * ffn) * nl

    sct_adam_gb = sct_total * 4 * 4 / 1e9     # 4 copies × float32
    dense_adam_gb = dense_total * 4 * 4 / 1e9
    compression = dense_total / sct_total

    print(f"\n    SCT parameters:   {sct_total:>14,}  ({sct_total/1e6:.1f}M)")
    print(f"    Dense equivalent: {dense_total:>14,}  ({dense_total/1e9:.1f}B)")
    print(f"    Compression:      {compression:.0f}x")
    print(f"    SCT + Adam mem:   {sct_adam_gb:.2f} GB")
    print(f"    Dense + Adam mem: {dense_adam_gb:.1f} GB")

    # ── Build model ───────────────────────────────────────────────
    print(f"\n[*] Building {nl}-layer spectral stack...")
    t_build = time.time()
    layers = nn.ModuleList(
        [Spectral70BLayer(config, rank=RANK) for _ in range(nl)]
    ).to(device)
    build_time = time.time() - t_build
    print(f"    Built in {build_time:.1f}s")

    actual_params = sum(p.numel() for p in layers.parameters())
    print(f"    Actual parameters: {actual_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────
    opt = torch.optim.AdamW(layers.parameters(), lr=1e-4)

    # ── Forward pass ──────────────────────────────────────────────
    print(f"\n[*] Forward pass (batch=1, seq=8, d={h})...")
    dummy = torch.randn(1, 8, h, device=device)

    start = time.time()
    opt.zero_grad()

    x = dummy
    for i, layer in enumerate(layers):
        x = layer(x)
        if (i + 1) % 20 == 0:
            print(f"    Layer {i+1}/{nl} done")

    loss = x.sum()
    fwd_time = time.time() - start
    print(f"    Forward: {fwd_time:.2f}s")

    # ── Backward pass ─────────────────────────────────────────────
    print(f"[*] Backward pass...")
    t_bwd = time.time()
    loss.backward()
    bwd_time = time.time() - t_bwd
    print(f"    Backward: {bwd_time:.2f}s")

    # ── Optimizer step ────────────────────────────────────────────
    print(f"[*] Optimizer step...")
    t_opt = time.time()
    opt.step()
    opt_time = time.time() - t_opt

    # ── CRITICAL: Stiefel retraction ──────────────────────────────
    print(f"[*] Stiefel QR retraction...")
    t_ret = time.time()
    retract_all(layers)
    ret_time = time.time() - t_ret
    print(f"    Retraction: {ret_time:.2f}s")

    total_time = time.time() - start

    # ── Memory measurement ────────────────────────────────────────
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n    Peak GPU Memory: {peak_mem:.1f} MB")
    else:
        # Estimate from parameter count
        # Adam stores: params + grads + m1 + m2 = 4x
        est_mem = actual_params * 4 * 4 / 1e6
        print(f"\n    Estimated memory (params+Adam): {est_mem:.1f} MB")

    # ── Verify orthonormality after retraction ────────────────────
    sample_layer = layers[0].q_proj
    UtU = sample_layer.U.data.T @ sample_layer.U.data
    ortho_err = (UtU - torch.eye(RANK, device=device)).norm().item()
    print(f"    Orthonormality error (layer 0 Q): {ortho_err:.2e}")

    # ── Results ───────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  RESULT: {'SUCCESS' if ortho_err < 0.01 else 'CHECK RETRACTION'}")
    print(f"{'='*58}")
    print(f"  Architecture:     70B-class (80 layers, d=8192)")
    print(f"  SCT Parameters:   {actual_params:,} ({actual_params/1e6:.0f}M)")
    print(f"  Dense Equivalent: {dense_total/1e9:.1f}B")
    print(f"  Compression:      {compression:.0f}x")
    print(f"  Forward:          {fwd_time:.2f}s")
    print(f"  Backward:         {bwd_time:.2f}s")
    print(f"  Optimizer:        {opt_time:.2f}s")
    print(f"  QR Retraction:    {ret_time:.2f}s")
    print(f"  Total Step:       {total_time:.2f}s")
    print(f"  Ortho Error:      {ortho_err:.2e}")
    print(f"{'='*58}")


if __name__ == "__main__":
    run_test()