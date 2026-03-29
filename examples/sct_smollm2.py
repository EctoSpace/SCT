import argparse, math, os, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def safe_qr(M):
    dev = M.device
    Q, R = torch.linalg.qr(M.cpu() if dev.type == "mps" else M)
    return (Q * torch.sign(torch.diag(R))).to(dev)

class SpectralLinear(nn.Module):
    def __init__(self, U, s, V, bias=None):
        super().__init__()
        self.rank = s.shape[0]
        self.in_features = U.shape[0]
        self.out_features = V.shape[0]
        self.U = nn.Parameter(U)
        self.s = nn.Parameter(s)
        self.V = nn.Parameter(V)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        y = (x @ self.U) * self.s @ self.V.T
        return y + self.bias if self.bias is not None else y

    @torch.no_grad()
    def retract(self):
        self.U.data = safe_qr(self.U.data)[:, :self.rank]
        self.V.data = safe_qr(self.V.data)[:, :self.rank]

    def param_count(self):
        n = self.U.numel() + self.V.numel() + self.s.numel()
        return n + (self.bias.numel() if self.bias is not None else 0)

    @classmethod
    def from_linear(cls, linear, rank=0, energy_threshold=0.95):
        W = linear.weight.data.float().cpu()
        m, n = W.shape 

        U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)

        if rank <= 0:
            total_energy = (S_full ** 2).sum()
            cumulative = torch.cumsum(S_full ** 2, dim=0) / total_energy
            k = int((cumulative >= energy_threshold).nonzero(as_tuple=True)[0][0].item()) + 1
            k = max(k, 1)
        else:
            k = min(rank, min(m, n))

        energy_retained = float((S_full[:k] ** 2).sum() / (S_full ** 2).sum())

        U_ours = Vh_full[:k, :].T.contiguous()
        V_ours = U_full[:, :k].contiguous()
        s_ours = S_full[:k].contiguous()
        bias = linear.bias.data.float() if linear.bias is not None else None

        layer = cls(U_ours, s_ours, V_ours, bias)
        layer._energy_retained = energy_retained
        layer._dense_params = m * n + (m if linear.bias is not None else 0)
        return layer

def replace_mlp_with_spectral(model, rank, energy_threshold, device):
    replacements = []
    total_dense = 0
    total_spectral = 0
    total_energy = 0
    n_replaced = 0

    mlp_names = ["gate_proj", "up_proj", "down_proj", "fc_1", "fc_2", "c_fc", "c_proj"]

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf not in mlp_names:
            continue

        spec = SpectralLinear.from_linear(module, rank=rank, energy_threshold=energy_threshold).to(device)
        parent_name, child_name = name.rsplit(".", 1)
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, child_name, spec)

        total_dense += spec._dense_params
        total_spectral += spec.param_count()
        total_energy += spec._energy_retained
        n_replaced += 1

        replacements.append({
            "name": name,
            "shape": f"{spec.in_features}x{spec.out_features}",
            "rank": spec.rank,
            "energy": f"{spec._energy_retained:.3f}",
            "dense": spec._dense_params,
            "spectral": spec.param_count(),
        })

    avg_energy = total_energy / max(n_replaced, 1)
    compression = total_dense / max(total_spectral, 1)

    return replacements, {
        "n_replaced": n_replaced,
        "total_dense_params": total_dense,
        "total_spectral_params": total_spectral,
        "avg_energy_retained": avg_energy,
        "compression": compression,
    }

def retract_all(model):
    for m in model.modules():
        if isinstance(m, SpectralLinear):
            m.retract()

def format_alpaca(ex):
    if ex.get("input", "").strip():
        return (f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}")
    return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"

def prepare_data(tokenizer, max_seq_len=128, max_samples=500):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    texts = [format_alpaca(ex) for ex in ds]
    enc = tokenizer(texts, truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return enc["input_ids"], enc["attention_mask"], labels

def train_loop(model, input_ids, attn_mask, labels, args, label="", is_spectral=False):
    device = torch.device(args.device)
    model.to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    warmup_steps = min(20, args.steps // 5)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(args.steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    n = input_ids.shape[0]
    bs = args.batch_size
    losses = []
    t0 = time.time()
    step = 0

    for epoch in range(200):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            if step >= args.steps:
                break
            idx = perm[i:i+bs]
            xb = input_ids[idx].to(device)
            mb = attn_mask[idx].to(device)
            yb = labels[idx].to(device)

            out = model(input_ids=xb, attention_mask=mb)
            logits = out.logits[:, :-1, :].contiguous()
            targets = yb[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad()
            scheduler.step()

            if is_spectral:
                retract_all(model)

            losses.append(loss.item())
            step += 1

            if step % args.log_every == 0 or step == 1:
                window = losses[-args.log_every:]
                avg = sum(window) / len(window)
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                print(f"  [{label:28s}] step {step:4d} | loss {avg:.4f} | ppl {math.exp(min(avg, 20)):.1f} | lr {lr_now:.2e} | {elapsed:.1f}s")
        if step >= args.steps:
            break

    elapsed = time.time() - t0
    final_window = losses[-20:] if len(losses) >= 20 else losses
    final_avg = sum(final_window) / len(final_window)
    return {
        "label": label,
        "final_loss": round(final_avg, 4),
        "best_loss": round(min(losses), 4),
        "final_ppl": round(math.exp(min(final_avg, 20)), 1),
        "best_ppl": round(math.exp(min(min(losses), 20)), 1),
        "steps": step,
        "time_sec": round(elapsed, 1),
        "trainable_params": sum(p.numel() for p in trainable),
    }

def generate(model, tokenizer, prompt, device, max_new=80):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, temperature=0.7,
            do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--energy", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    print("=" * 70)
    print("  SCT Fine Tuning v2: SmolLM2 on Alpaca")
    print("=" * 70)
    rank_desc = f"rank={args.rank}" if args.rank > 0 else f"adaptive energy>={args.energy}"
    print(f"  Model: {args.model}")
    print(f"  Rank: {rank_desc}  LR: {args.lr}  Device: {args.device}")
    print(f"  Steps: {args.steps}  Batch: {args.batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids, attn_mask, labels = prepare_data(tokenizer, args.max_seq_len, args.max_samples)
    print(f"  Data: {input_ids.shape[0]} samples, seq_len={args.max_seq_len}\n")

    results = []

    print("  1. DENSE FINE TUNING")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total:,}")
    r = train_loop(model, input_ids, attn_mask, labels, args, "Dense+AdamW")
    results.append(r)

    prompt = "### Instruction:\nExplain what gravity is in simple terms.\n\n### Response:\n"
    dense_gen = generate(model, tokenizer, prompt, args.device)
    del model
    if args.device == "mps": torch.mps.empty_cache()

    print(f"\n  2. SCT FINE TUNING {rank_desc}")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    for param in model.parameters():
        param.requires_grad = False

    replaced, stats = replace_mlp_with_spectral(model, rank=args.rank, energy_threshold=args.energy, device=args.device)

    for name, param in model.named_parameters():
        if "layernorm" in name.lower() or "ln" in name.lower() or "norm" in name.lower():
            param.requires_grad = True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())

    print(f"  Replaced {stats['n_replaced']} layers")
    print(f"  Avg energy retained: {stats['avg_energy_retained']:.3f}")
    print(f"  MLP compression: {stats['compression']:.1f}x")
    print(f"  Trainable: {trainable_count:,} / {total_count:,}")

    r = train_loop(model, input_ids, attn_mask, labels, args, "SCT", is_spectral=True)
    results.append(r)

    sct_gen = generate(model, tokenizer, prompt, args.device)

    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['label']:35s} | loss {r['final_loss']:.4f} | ppl {r['final_ppl']:>7.1f} | params {r['trainable_params']:>12,} | {r['time_sec']}s")

    ratio = results[0]['trainable_params'] / max(results[1]['trainable_params'], 1)
    ppl_ratio = results[1]['final_ppl'] / max(results[0]['final_ppl'], 0.1)
    print(f"\n  Trainable param reduction: {ratio:.1f}x")
    print(f"  PPL ratio: {ppl_ratio:.2f}x")
    print(f"  MLP compression: {stats['compression']:.1f}x")

    print(f"\n{'='*70}")
    print("  GENERATION")
    print(f"{'='*70}")
    print(f"\n  [Dense]: {dense_gen[:300]}")
    print(f"\n  [SCT]:   {sct_gen[:300]}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sct_smollm2_results.json")
    payload = {
        "results": results,
        "stats": stats,
        "per_layer": replaced,
        "config": {"model": args.model, "rank": args.rank, "energy": args.energy, "lr": args.lr, "steps": args.steps},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved to {out_path}")

if __name__ == "__main__":
    main()