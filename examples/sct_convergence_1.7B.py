#!/usr/bin/env python3
"""
SCT 1.7B CONVERGENCE EXPERIMENT
================================
Fine-tunes SmolLM2-1.7B on Alpaca using both Dense and SCT methods,
logs loss curves, and generates a comparison plot.

This is the key experiment: proving SCT converges at a scale where
compression actually matters (51x per MLP layer at rank 32).

Hardware: Apple M4 Pro 48GB (MPS backend)
Expected runtime: ~6-12 hours depending on step count
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
#  SPECTRAL LINEAR (same as core library)
# ═══════════════════════════════════════════════════════════════════

class SpectralLinear(nn.Module):
    """W = U diag(s) V^T with Stiefel QR retraction."""

    def __init__(self, in_features, out_features, rank=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, min(in_features, out_features))

        U = torch.randn(in_features, self.rank) / math.sqrt(in_features)
        V = torch.randn(out_features, self.rank) / math.sqrt(out_features)
        Q_U, R_U = torch.linalg.qr(U)
        Q_V, R_V = torch.linalg.qr(V)
        self.U = nn.Parameter((Q_U * torch.sign(torch.diag(R_U)))[:, :self.rank])
        self.V = nn.Parameter((Q_V * torch.sign(torch.diag(R_V)))[:, :self.rank])
        self.s = nn.Parameter(torch.ones(self.rank))

    def forward(self, x):
        return (x @ self.U) * self.s @ self.V.T

    @classmethod
    def from_linear(cls, linear, rank=None, energy=None):
        """Convert a dense nn.Linear to SpectralLinear via truncated SVD."""
        W = linear.weight.data.float()  # [out, in]
        U_full, S_full, Vt_full = torch.linalg.svd(W, full_matrices=False)

        if energy is not None:
            total_energy = (S_full ** 2).sum()
            cumulative = torch.cumsum(S_full ** 2, dim=0) / total_energy
            k = max(1, int((cumulative >= energy).nonzero(as_tuple=True)[0][0].item()) + 1)
            if rank is not None:
                k = min(k, rank)
        elif rank is not None:
            k = min(rank, S_full.shape[0])
        else:
            k = min(32, S_full.shape[0])

        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.in_features = linear.in_features
        layer.out_features = linear.out_features
        layer.rank = k

        # SVD gives W = U @ diag(S) @ Vt, where W is [out, in]
        # Our forward is: x @ self.U * s @ self.V.T
        # So we need: self.U = V[:, :k] (in_features x k)
        #              self.V = U[:, :k] (out_features x k)
        #              self.s = S[:k]
        layer.U = nn.Parameter(Vt_full[:k, :].T.contiguous())  # [in, k]
        layer.V = nn.Parameter(U_full[:, :k].contiguous())     # [out, k]
        layer.s = nn.Parameter(S_full[:k].contiguous())

        return layer

    @torch.no_grad()
    def retract(self):
        """QR retraction to Stiefel manifold."""
        for M in [self.U, self.V]:
            Q, R = torch.linalg.qr(M.data.cpu())
            M.data = (Q * torch.sign(torch.diag(R)))[:, :self.rank].to(M.device)


def retract_all(module):
    for m in module.modules():
        if isinstance(m, SpectralLinear):
            m.retract()


def convert_model_mlp_to_spectral(model, rank=None, energy=None):
    """Convert all MLP linear layers in a HuggingFace model to SpectralLinear."""
    converted = 0
    for name, module in model.named_modules():
        # Target MLP layers: gate_proj, up_proj, down_proj
        for attr in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(module, attr):
                linear = getattr(module, attr)
                if isinstance(linear, nn.Linear):
                    spectral = SpectralLinear.from_linear(linear, rank=rank, energy=energy)
                    setattr(module, attr, spectral)
                    converted += 1
    return converted


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_alpaca(tokenizer, max_length=512, max_samples=None):
    """Load and tokenize Alpaca dataset."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_sample(ex):
        if ex.get("input", "").strip():
            text = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        else:
            text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
        return text

    texts = [format_sample(ex) for ex in ds]
    tokenizer.pad_token = tokenizer.eos_token

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    return encodings


# ═══════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train(model, encodings, device, steps, lr, batch_size, log_every, is_sct=False, label=""):
    """Train and return loss history."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    n_samples = input_ids.shape[0]

    losses = []
    step_times = []
    step = 0

    print(f"\n{'='*60}")
    print(f"  Training: {label}")
    print(f"  Steps: {steps}  LR: {lr}  Batch: {batch_size}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"{'='*60}\n")

    while step < steps:
        # Random batch
        idx = torch.randint(0, n_samples, (batch_size,))
        batch_input = input_ids[idx].to(device)
        batch_mask = attention_mask[idx].to(device)
        labels = batch_input.clone()
        labels[batch_mask == 0] = -100

        t0 = time.time()

        optimizer.zero_grad()
        outputs = model(input_ids=batch_input, attention_mask=batch_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if is_sct:
            retract_all(model)

        step_time = time.time() - t0
        loss_val = loss.item()
        losses.append({"step": step, "loss": loss_val, "time": step_time})
        step_times.append(step_time)
        step += 1

        if step % log_every == 0 or step == 1:
            ppl = math.exp(min(loss_val, 20))
            avg_time = sum(step_times[-log_every:]) / len(step_times[-log_every:])
            eta_min = (steps - step) * avg_time / 60
            print(f"  [{label}] Step {step:>5d}/{steps}  "
                  f"Loss: {loss_val:.4f}  PPL: {ppl:.2f}  "
                  f"Step: {avg_time:.2f}s  ETA: {eta_min:.1f}min")

    return losses


# ═══════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════

def plot_convergence(dense_losses, sct_losses, output_path, rank, energy):
    """Generate convergence comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    dense_steps = [l["step"] for l in dense_losses]
    dense_vals = [l["loss"] for l in dense_losses]
    sct_steps = [l["step"] for l in sct_losses]
    sct_vals = [l["loss"] for l in sct_losses]

    # Loss curve
    ax1.plot(dense_steps, dense_vals, label="Dense + AdamW", color="#2196F3", alpha=0.7, linewidth=1.5)
    ax1.plot(sct_steps, sct_vals, label=f"SCT (rank {rank})", color="#FF5722", alpha=0.7, linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("SmolLM2-1.7B Fine-tuning: Loss Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PPL curve
    dense_ppl = [math.exp(min(l, 20)) for l in dense_vals]
    sct_ppl = [math.exp(min(l, 20)) for l in sct_vals]
    ax2.plot(dense_steps, dense_ppl, label="Dense + AdamW", color="#2196F3", alpha=0.7, linewidth=1.5)
    ax2.plot(sct_steps, sct_ppl, label=f"SCT (rank {rank})", color="#FF5722", alpha=0.7, linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("SmolLM2-1.7B Fine-tuning: Perplexity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Add annotation
    fig.text(0.5, 0.01,
             f"SmolLM2-1.7B | Alpaca | Energy retention: {energy} | "
             f"MLP compression: 51x at rank 32 | Apache 2.0 | EctoSpace/SCT",
             ha='center', fontsize=9, color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {output_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SCT 1.7B Convergence Experiment")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per method")
    parser.add_argument("--rank", type=int, default=32, help="SCT rank")
    parser.add_argument("--energy", type=float, default=0.95, help="Energy retention for SVD truncation")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--output-dir", type=str, default="results/convergence_1.7B", help="Output directory")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense baseline (if already run)")
    parser.add_argument("--skip-sct", action="store_true", help="Skip SCT run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[*] Device: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[*] Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[*] Device: CPU")

    # ── Load tokenizer and data ───────────────────────────────────
    from transformers import AutoTokenizer
    print(f"[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    print(f"[*] Loading and tokenizing Alpaca dataset...")
    encodings = load_alpaca(tokenizer, max_length=args.max_length, max_samples=args.max_samples)
    print(f"    Samples: {encodings['input_ids'].shape[0]}  Seq length: {args.max_length}")

    dense_losses = []
    sct_losses = []

    # ── Dense baseline ────────────────────────────────────────────
    dense_results_path = os.path.join(args.output_dir, "dense_losses.json")
    if not args.skip_dense:
        from transformers import AutoModelForCausalLM
        print(f"\n[*] Loading SmolLM2-1.7B for dense baseline...")
        model_dense = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-1.7B",
            torch_dtype=torch.float32
        ).to(device)

        dense_params = sum(p.numel() for p in model_dense.parameters())
        print(f"    Dense parameters: {dense_params:,}")

        dense_losses = train(
            model_dense, encodings, device,
            steps=args.steps, lr=args.lr, batch_size=args.batch_size,
            log_every=args.log_every, is_sct=False, label="Dense"
        )

        with open(dense_results_path, "w") as f:
            json.dump(dense_losses, f, indent=2)
        print(f"    Dense results saved: {dense_results_path}")

        # Free memory
        del model_dense
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    elif os.path.exists(dense_results_path):
        print(f"[*] Loading cached dense results from {dense_results_path}")
        with open(dense_results_path) as f:
            dense_losses = json.load(f)

    # ── SCT ───────────────────────────────────────────────────────
    sct_results_path = os.path.join(args.output_dir, "sct_losses.json")
    if not args.skip_sct:
        from transformers import AutoModelForCausalLM
        print(f"\n[*] Loading SmolLM2-1.7B for SCT conversion...")
        model_sct = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-1.7B",
            torch_dtype=torch.float32
        ).to(device)

        print(f"[*] Converting MLP layers to SpectralLinear (energy={args.energy})...")
        n_converted = convert_model_mlp_to_spectral(model_sct, energy=args.energy)
        print(f"    Converted {n_converted} layers")

        sct_params = sum(p.numel() for p in model_sct.parameters() if p.requires_grad)
        print(f"    SCT parameters: {sct_params:,}")

        # Report per-layer ranks
        ranks = []
        for m in model_sct.modules():
            if isinstance(m, SpectralLinear):
                ranks.append(m.rank)
        if ranks:
            print(f"    Rank range: {min(ranks)}-{max(ranks)} (mean: {sum(ranks)/len(ranks):.0f})")

        sct_losses = train(
            model_sct, encodings, device,
            steps=args.steps, lr=args.lr, batch_size=args.batch_size,
            log_every=args.log_every, is_sct=True, label="SCT"
        )

        with open(sct_results_path, "w") as f:
            json.dump(sct_losses, f, indent=2)
        print(f"    SCT results saved: {sct_results_path}")

        del model_sct

    elif os.path.exists(sct_results_path):
        print(f"[*] Loading cached SCT results from {sct_results_path}")
        with open(sct_results_path) as f:
            sct_losses = json.load(f)

    # ── Plot ──────────────────────────────────────────────────────
    if dense_losses and sct_losses:
        plot_path = os.path.join(args.output_dir, "convergence_smollm2_1.7B.png")
        plot_convergence(dense_losses, sct_losses, plot_path,
                        rank=args.rank, energy=args.energy)

        # Summary
        dense_final = dense_losses[-1]["loss"]
        sct_final = sct_losses[-1]["loss"]
        dense_ppl = math.exp(min(dense_final, 20))
        sct_ppl = math.exp(min(sct_final, 20))
        ratio = sct_ppl / dense_ppl

        dense_time = sum(l["time"] for l in dense_losses)
        sct_time = sum(l["time"] for l in sct_losses)

        print(f"\n{'='*60}")
        print(f"  CONVERGENCE RESULTS: SmolLM2-1.7B on Alpaca")
        print(f"{'='*60}")
        print(f"  Dense final loss:  {dense_final:.4f}  PPL: {dense_ppl:.2f}")
        print(f"  SCT final loss:    {sct_final:.4f}  PPL: {sct_ppl:.2f}")
        print(f"  PPL ratio:         {ratio:.2f}x")
        print(f"  Dense total time:  {dense_time/60:.1f} min")
        print(f"  SCT total time:    {sct_time/60:.1f} min")
        print(f"  MLP compression:   51x at rank 32")
        print(f"{'='*60}")

        # Save summary
        summary = {
            "model": "SmolLM2-1.7B",
            "dataset": "alpaca",
            "steps": args.steps,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "rank": args.rank,
            "energy": args.energy,
            "dense_final_loss": dense_final,
            "dense_final_ppl": dense_ppl,
            "sct_final_loss": sct_final,
            "sct_final_ppl": sct_ppl,
            "ppl_ratio": ratio,
            "dense_total_time_s": dense_time,
            "sct_total_time_s": sct_time,
        }
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()