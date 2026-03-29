#!/usr/bin/env python3
"""
SCT vs Dense: SmolLM2-135M Fine-Tuning
=======================================

Head-to-head comparison:
  A) Dense fine-tune — standard AdamW, all parameters
  B) SCT fine-tune  — spectral MLP layers, exact backprop, Stiefel retraction

Same model, same data, same seed, same steps, same learning rate.

Usage:
    python sct_vs_dense.py
    python sct_vs_dense.py --energy 0.99 --steps 300
    python sct_vs_dense.py --rank 128          # fixed rank instead of adaptive
"""

import argparse, math, os, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ═══════════════════════════════════════════════════════════════════
#  SPECTRAL LINEAR — W = U diag(s) V^T, exact backprop
# ═══════════════════════════════════════════════════════════════════

def safe_qr(M):
    dev = M.device
    Q, R = torch.linalg.qr(M.cpu() if dev.type == "mps" else M)
    return (Q * torch.sign(torch.diag(R))).to(dev)


class SpectralLinear(nn.Module):
    """Drop-in nn.Linear replacement storing W = U diag(s) V^T."""

    def __init__(self, U, s, V, bias=None):
        super().__init__()
        self.rank = s.shape[0]
        self.in_features = U.shape[0]
        self.out_features = V.shape[0]
        self.U = nn.Parameter(U)       # [in, k]
        self.s = nn.Parameter(s)       # [k]
        self.V = nn.Parameter(V)       # [out, k]
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        y = (x @ self.U) * self.s @ self.V.T
        return y + self.bias if self.bias is not None else y

    @torch.no_grad()
    def retract(self):
        """QR retraction — project U, V back onto the Stiefel manifold."""
        self.U.data = safe_qr(self.U.data)[:, :self.rank]
        self.V.data = safe_qr(self.V.data)[:, :self.rank]

    def param_count(self):
        n = self.U.numel() + self.V.numel() + self.s.numel()
        return n + (self.bias.numel() if self.bias is not None else 0)

    @classmethod
    def from_linear(cls, linear, rank=0, energy_threshold=0.95):
        """
        Convert pretrained nn.Linear → SpectralLinear via truncated SVD.
        rank=0 means adaptive: find minimum rank retaining energy_threshold.
        """
        W = linear.weight.data.float().cpu()
        m, n = W.shape  # [out, in]

        U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)

        if rank <= 0:
            total_energy = (S_full ** 2).sum()
            cumulative = torch.cumsum(S_full ** 2, dim=0) / total_energy
            k = int((cumulative >= energy_threshold).nonzero(as_tuple=True)[0][0].item()) + 1
            k = max(k, 1)
        else:
            k = min(rank, min(m, n))

        energy_retained = float((S_full[:k] ** 2).sum() / (S_full ** 2).sum())

        # Convention: forward is x @ U * s @ V^T
        # SVD gives W = U_full @ diag(S) @ Vh_full
        # So U_ours = Vh[:k]^T, V_ours = U_full[:,:k]
        layer = cls(
            Vh_full[:k, :].T.contiguous(),   # U: [in, k]
            S_full[:k].contiguous(),           # s: [k]
            U_full[:, :k].contiguous(),        # V: [out, k]
            linear.bias.data.float() if linear.bias is not None else None,
        )
        layer._energy = energy_retained
        layer._dense_params = m * n + (m if linear.bias is not None else 0)
        return layer


# ═══════════════════════════════════════════════════════════════════
#  MODEL SURGERY
# ═══════════════════════════════════════════════════════════════════

MLP_LEAF_NAMES = frozenset([
    "gate_proj", "up_proj", "down_proj",   # LLaMA-style
    "fc_1", "fc_2",                         # SmolLM / GPT-NeoX
    "c_fc", "c_proj",                       # GPT-2
])


def replace_mlp_with_spectral(model, rank, energy, device):
    """Swap MLP nn.Linear → SpectralLinear. Returns per-layer stats."""
    layers_info = []
    total_dense = 0
    total_spectral = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf not in MLP_LEAF_NAMES:
            continue

        spec = SpectralLinear.from_linear(module, rank=rank,
                                           energy_threshold=energy).to(device)
        parent_name, child_name = name.rsplit(".", 1)
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, child_name, spec)

        total_dense += spec._dense_params
        total_spectral += spec.param_count()
        layers_info.append({
            "name": name,
            "shape": f"{spec.out_features}x{spec.in_features}",
            "rank": spec.rank,
            "energy": round(spec._energy, 4),
        })

    return layers_info, total_dense, total_spectral


def retract_all(model):
    for m in model.modules():
        if isinstance(m, SpectralLinear):
            m.retract()


# ═══════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════

def format_alpaca(ex):
    if ex.get("input", "").strip():
        return (f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}")
    return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"


def prepare_data(tokenizer, max_seq_len, max_samples):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    texts = [format_alpaca(ex) for ex in ds]
    enc = tokenizer(texts, truncation=True, max_length=max_seq_len,
                    padding="max_length", return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return enc["input_ids"], enc["attention_mask"], labels


# ═══════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════

def train(model, input_ids, attn_mask, labels, args, label="",
          is_spectral=False):
    device = torch.device(args.device)
    model.to(device).train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    warmup = min(20, args.steps // 5)
    def lr_fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(args.steps - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    n, bs = input_ids.shape[0], args.batch_size
    losses, step, t0 = [], 0, time.time()

    for _ in range(999):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            if step >= args.steps:
                break
            idx = perm[i:i+bs]
            xb = input_ids[idx].to(device)
            mb = attn_mask[idx].to(device)
            yb = labels[idx].to(device)

            logits = model(input_ids=xb, attention_mask=mb).logits
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                yb[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(); sched.step()

            if is_spectral:
                retract_all(model)

            losses.append(loss.item())
            step += 1

            if step % args.log_every == 0 or step == 1:
                w = losses[-args.log_every:]
                avg = sum(w) / len(w)
                print(f"  [{label:30s}] step {step:4d} | loss {avg:.4f} | "
                      f"ppl {math.exp(min(avg,20)):.1f} | {time.time()-t0:.1f}s")
        if step >= args.steps:
            break

    final = losses[-20:] if len(losses) >= 20 else losses
    return {
        "label": label,
        "final_loss": round(sum(final)/len(final), 4),
        "best_loss": round(min(losses), 4),
        "final_ppl": round(math.exp(min(sum(final)/len(final), 20)), 1),
        "steps": step,
        "time_sec": round(time.time()-t0, 1),
        "trainable_params": sum(p.numel() for p in trainable),
        "losses": [round(l, 4) for l in losses],
    }


def generate(model, tokenizer, prompt, device, max_new=80):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, temperature=0.7, do_sample=True,
            top_p=0.9, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.2,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--rank", type=int, default=0,
                   help="Fixed rank. 0 = adaptive via --energy")
    p.add_argument("--energy", type=float, default=0.95,
                   help="Spectral energy retention for adaptive rank")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    rank_str = f"rank={args.rank}" if args.rank > 0 else f"adaptive energy≥{args.energy}"

    print()
    print("=" * 70)
    print("  SCT vs DENSE — SmolLM2-135M Fine-Tuning on Alpaca")
    print("=" * 70)
    print(f"  Model:  {args.model}")
    print(f"  SCT:    {rank_str}")
    print(f"  LR:     {args.lr}   Steps: {args.steps}   Device: {args.device}")

    # ── Data ──────────────────────────────────────────────────────
    print("\n  Loading tokenizer + data...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    input_ids, attn_mask, labels = prepare_data(tok, args.max_seq_len, args.max_samples)
    print(f"  {input_ids.shape[0]} samples, seq_len={args.max_seq_len}")

    prompt = "### Instruction:\nExplain what gravity is in simple terms.\n\n### Response:\n"
    results = {}

    # ══════════════════════════════════════════════════════════════
    #  A) DENSE — standard fine-tuning
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'━'*70}")
    print("  A) DENSE FINE-TUNE")
    print(f"{'━'*70}")

    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    results["dense"] = train(model, input_ids, attn_mask, labels, args,
                             "Dense + AdamW")
    dense_gen = generate(model, tok, prompt, args.device)
    del model
    if args.device == "mps":
        torch.mps.empty_cache()

    # ══════════════════════════════════════════════════════════════
    #  B) SCT — spectral compact training
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'━'*70}")
    print(f"  B) SCT FINE-TUNE ({rank_str})")
    print(f"{'━'*70}")

    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Replace MLP layers with spectral factors
    layers_info, dense_mlp_params, spectral_mlp_params = replace_mlp_with_spectral(
        model, rank=args.rank, energy=args.energy, device=args.device,
    )

    # Unfreeze layernorms (tiny, critical for adaptation)
    norm_params = 0
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in ["layernorm", "ln_", "norm", "rmsnorm"]):
            param.requires_grad = True
            norm_params += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ranks = [l["rank"] for l in layers_info]
    energies = [l["energy"] for l in layers_info]

    print(f"  Replaced {len(layers_info)} MLP layers")
    print(f"  Rank range: {min(ranks)} – {max(ranks)} (mean {sum(ranks)/len(ranks):.0f})")
    print(f"  Energy retained: {min(energies):.3f} – {max(energies):.3f} "
          f"(mean {sum(energies)/len(energies):.3f})")
    print(f"  MLP params: {dense_mlp_params:,} dense → {spectral_mlp_params:,} spectral "
          f"({dense_mlp_params/spectral_mlp_params:.1f}x)")
    print(f"  Norm params: {norm_params:,}")
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    results["sct"] = train(model, input_ids, attn_mask, labels, args,
                           f"SCT ({rank_str})", is_spectral=True)
    sct_gen = generate(model, tok, prompt, args.device)

    # ══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════
    rd, rs = results["dense"], results["sct"]

    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"  {'Method':<35s} {'Loss':>8s} {'PPL':>8s} {'Params':>14s} {'Time':>8s}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*14} {'─'*8}")
    print(f"  {rd['label']:<35s} {rd['final_loss']:>8.4f} {rd['final_ppl']:>8.1f} "
          f"{rd['trainable_params']:>14,} {rd['time_sec']:>7.1f}s")
    print(f"  {rs['label']:<35s} {rs['final_loss']:>8.4f} {rs['final_ppl']:>8.1f} "
          f"{rs['trainable_params']:>14,} {rs['time_sec']:>7.1f}s")

    param_ratio = rd['trainable_params'] / max(rs['trainable_params'], 1)
    ppl_ratio = rs['final_ppl'] / max(rd['final_ppl'], 0.1)

    print(f"\n  Trainable param reduction: {param_ratio:.1f}x")
    print(f"  MLP compression: {dense_mlp_params/spectral_mlp_params:.1f}x")
    print(f"  PPL ratio (SCT / Dense): {ppl_ratio:.2f}x")

    if ppl_ratio < 1.5:
        print(f"  Verdict: SCT MATCHES dense quality ({ppl_ratio:.2f}x PPL)")
    elif ppl_ratio < 3.0:
        print(f"  Verdict: SCT CLOSE to dense ({ppl_ratio:.2f}x PPL)")
    else:
        print(f"  Verdict: SCT DEGRADES quality ({ppl_ratio:.2f}x PPL) — increase energy threshold")

    print(f"\n{'='*70}")
    print("  GENERATION")
    print(f"{'='*70}")
    print(f"\n  [Dense]:\n  {dense_gen[:400]}")
    print(f"\n  [SCT]:\n  {sct_gen[:400]}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "sct_vs_dense_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "dense": rd, "sct": rs,
            "mlp_compression": round(dense_mlp_params/spectral_mlp_params, 2),
            "ppl_ratio": round(ppl_ratio, 3),
            "per_layer": layers_info,
            "config": vars(args),
        }, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()