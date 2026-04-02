import torch
import torch.nn as nn
import math, time, json
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

# ==================== CONFIG ====================
DEVICE = "mps"
RANK = 32
STEPS = 20000
BATCH_SIZE = 8          # M4 Pro 48 GB can handle this
GRAD_ACCUM = 2          # effective batch = 16
LR = 3e-4
WARMUP_STEPS = 500
MAX_SEQ_LEN = 256
LOG_EVERY = 200
SAVE_EVERY = 5000
# ===============================================

def safe_qr(M):
    dev, dt = M.device, M.dtype
    Q, R = torch.linalg.qr(M.float().cpu())
    return (Q * torch.sign(torch.diag(R))).to(dev).to(dt)

class SpectralLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, min(in_features, out_features))
        U = torch.randn(in_features, self.rank)
        V = torch.randn(out_features, self.rank)
        Q_U, R_U = torch.linalg.qr(U)
        Q_V, R_V = torch.linalg.qr(V)
        self.U = nn.Parameter((Q_U * torch.sign(torch.diag(R_U)))[:, :self.rank])
        self.V = nn.Parameter((Q_V * torch.sign(torch.diag(R_V)))[:, :self.rank])
        scale = math.sqrt(2.0 * out_features / (in_features * self.rank))
        self.s = nn.Parameter(torch.full((self.rank,), scale))

    def forward(self, x):
        U = self.U.to(x.dtype)
        s = self.s.to(x.dtype)
        V = self.V.to(x.dtype)
        return (x @ U) * s @ V.T

    @torch.no_grad()
    def retract(self):
        self.U.data = safe_qr(self.U.data)[:, :self.rank]
        self.V.data = safe_qr(self.V.data)[:, :self.rank]

def retract_all(model):
    for m in model.modules():
        if isinstance(m, SpectralLinear):
            m.retract()

# Build model
config = LlamaConfig(
    hidden_size=2048, intermediate_size=8192, num_hidden_layers=24,
    num_attention_heads=32, num_key_value_heads=32, vocab_size=49152,
    max_position_embeddings=2048, rms_norm_eps=1e-5,
    tie_word_embeddings=True, hidden_act="silu"
)

print("Initializing fresh 1.7B model...")
model = LlamaForCausalLM(config)

converted = 0
for name, module in model.named_modules():
    for attr in ['gate_proj', 'up_proj', 'down_proj']:
        if hasattr(module, attr):
            linear = getattr(module, attr)
            if isinstance(linear, nn.Linear):
                spec = SpectralLinear(linear.in_features, linear.out_features, RANK)
                setattr(module, attr, spec)
                converted += 1
print(f"Converted {converted} MLP layers to SpectralLinear (rank={RANK})")

model = model.to(dtype=torch.bfloat16, device=DEVICE)
print(f"Model on {DEVICE} — ready.")

# Dataset
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
texts = [t for t in dataset["text"] if len(t.strip()) > 100]
print(f"Filtered texts: {len(texts):,}")

print("Tokenizing...")
all_ids = []
for i in range(0, len(texts), 1000):
    batch = texts[i:i+1000]
    enc = tokenizer(batch, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
    all_ids.append(enc["input_ids"])
    if (i // 1000) % 20 == 0:
        print(f"  Tokenized {i+1000}/{len(texts)}")
input_ids = torch.cat(all_ids, dim=0)
print(f"Dataset ready: {input_ids.shape[0]:,} sequences")

# Optimizer + scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))

def get_lr(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, STEPS - WARMUP_STEPS)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

# Training loop
print("\n=== STARTING SCT FROM-SCRATCH TRAINING ===")
model.train()
n_samples = input_ids.shape[0]
losses = []
t_start = time.time()

for step in range(1, STEPS + 1):
    t0 = time.time()
    total_loss = 0.0

    for micro in range(GRAD_ACCUM):
        idx = torch.randint(0, n_samples, (BATCH_SIZE,))
        xb = input_ids[idx].to(DEVICE)
        labels = xb.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(input_ids=xb, labels=labels)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    retract_all(model)

    step_time = time.time() - t0
    losses.append({"step": step, "loss": total_loss, "time": step_time, "lr": scheduler.get_last_lr()[0]})

    if step % LOG_EVERY == 0 or step == 1:
        avg_loss = sum(l["loss"] for l in losses[-LOG_EVERY:]) / min(LOG_EVERY, len(losses))
        ppl = math.exp(min(avg_loss, 20))
        elapsed = (time.time() - t_start) / 3600
        eta = (STEPS - step) * (sum(l["time"] for l in losses[-LOG_EVERY:]) / LOG_EVERY) / 3600
        cur_lr = scheduler.get_last_lr()[0]
        print(f"Step {step:>6d}/{STEPS}  Loss: {total_loss:.4f}  Avg: {avg_loss:.4f}  PPL: {ppl:.1f}  LR: {cur_lr:.1e}  Time: {elapsed:.1f}h  ETA: {eta:.1f}h")

    if step % SAVE_EVERY == 0:
        with open(f"sct_scratch_ckpt_{step}.json", "w") as f:
            json.dump({"step": step, "losses": losses}, f)
        print(f">> Checkpoint saved at step {step}")

print(f"\nTraining complete in {(time.time() - t_start)/3600:.1f} hours.")

# Results + plot (same as notebook)
# ... (plot code omitted for brevity — copy from notebook Cell 8 if you want the graph)
print("Done. Check sct_scratch_ckpt_*.json files.")