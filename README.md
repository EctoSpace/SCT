# Spectral Compact Training (SCT)

**A full 70B-architecture training step in 7.2 GB of memory.**

SCT stores every weight matrix as `W = U diag(s) V^T` and never materializes the dense matrix. Gradients flow through the small spectral factors via standard backpropagation with respect to the factored parameterization. After each optimizer step, U and V are retracted to the Stiefel manifold via QR decomposition.

That is the entire method.

```
Dense 70B + Adam:     1,245 GB
SCT 70B (rank 32):      7.2 GB
Compression:             172x
```

> **Patent Pending** — Irish Short-Term Patent Application PTIE20260000000219, filed March 27, 2026.

---

## Results

### 70B Architecture on Consumer Hardware

A full 70B-class transformer (80 layers, d=8192, ffn=28672, SwiGLU activation, matching LLaMA-3-70B layer dimensions) was initialized in spectral form at rank 32 and executed through one complete training step: forward pass, backward pass, Adam optimizer step, and Stiefel QR retraction. Attention is simplified (additive, no softmax/masking) to isolate the memory and gradient flow test from sequence-length concerns.

| Hardware | Peak Memory | Forward | Backward | Optimizer | QR Retraction | Total Step |
|----------|------------|---------|----------|-----------|---------------|------------|
| Apple M4 Pro (48 GB) | 7,907 MB | 0.08s | 0.09s | 0.22s | 3.02s | 3.41s |
| Steam Deck (16 GB) | 7,236 MB | 0.43s | 0.92s | 2.35s | 2.58s | 6.28s |

452M spectral parameters corresponding to a 77.8B-parameter dense architecture at rank 32. Orthonormality error after retraction: < 2e-06 on both platforms.

**What this demonstrates:** The memory footprint of a 70B-architecture training step (forward, backward, optimizer, manifold retraction) fits within 8 GB. This is an architectural validation of the SCT memory claim, not a trained language model.

**What this does not demonstrate:** Convergence to a useful language model at rank 32, or equivalence to a dense 70B model. These are separate questions that depend on rank, dataset, and training duration.

Video demonstrations of the Steam Deck run are in `results/steamDeck/`.

### Fine-Tuning Convergence (SmolLM2-135M on Alpaca)

Head-to-head recovery test: pre-trained SmolLM2-135M weights converted to spectral form at 95% energy retention, fine-tuned for 400 steps on Alpaca. Same model, same data, same seed, same learning rate.

| Method | Final Loss | Final PPL | Trainable Params |
|--------|-----------|-----------|-----------------|
| Dense + AdamW | 0.2356 | 1.3 | 134,515,008 |
| SCT (energy ≥ 0.95) | 0.6480 | 1.9 | 84,333,271 |

SCT recovers from an initial loss spike (9.4 → 0.65) to 1.46x baseline perplexity, confirming gradient integrity through spectral factors with Stiefel retraction.

**Important context:** SmolLM2-135M (hidden dim 576) is *below* the optimal scale for SCT compression. The adaptive rank at 95% energy produces ranks of 412–466, close to the full dimension. This test validates that the math works, not that compression is useful at this scale. Compression becomes significant at 1.7B+ parameters (see rank sweep below).

### Fine-Tuning Rank Sweep (SmolLM2-1.7B on Alpaca)

Rank sweep on SmolLM2-1.7B: dense baseline vs SCT at ranks 32, 64, 128, 256. MLP layers (gate_proj, up_proj, down_proj) converted to SpectralLinear; attention, embeddings, and norms remain dense. All runs: 2000 steps, batch 4, AdamW, A100 40GB. Dense LR: 2e-5. SCT LR: 5e-4.

| Method | Params | MLP Compression | Loss (smoothed) | PPL (smoothed) | GPU Memory | Step Time |
|--------|--------|----------------|-----------------|----------------|------------|-----------|
| Dense | 1,711M | 1.0x | 1.29 | 3.6 | 35.5 GB | 1.17s |
| SCT r=256 | 692M | 5.9x | 4.33 | 75.6 | 21.3 GB | 1.05s |
| **SCT r=128** | **598M** | **11.7x** | **4.18** | **65.6** | **20.0 GB** | **0.74s** |
| SCT r=64 | 551M | 23.5x | 4.34 | 76.7 | 19.3 GB | 0.62s |
| SCT r=32 | 527M | 46.9x | 4.47 | 86.9 | 19.0 GB | 0.56s |

**Memory efficiency confirmed at scale.** GPU usage drops from 35.5 GB (dense) to 19.0 GB (rank 32), a 46% reduction. Training steps run 2.1x faster. Even rank 256 saves 40% of VRAM.

**All ranks converge to the same loss floor (~4.2–4.5).** Rank 256 (5.9x compression) and rank 32 (46.9x) end within 0.3 loss of each other. This means MLP rank is not the bottleneck at 2000 steps. Rank 128 achieves the best PPL (65.6), likely because 5e-4 is near-optimal for its compression level while being too aggressive for rank 256 (which preserves more pretrained structure and needs a gentler LR).

**The ~3 loss gap vs dense points to the shared LR, not MLP capacity.** At rank 32, MLP spectral parameters account for only 18M of 527M total; attention layers are 403M (77% of the model). All components train at 5e-4, which is 25x the dense baseline LR. A per-component LR schedule (dense LR for attention/embeddings, higher LR for SCT factors) is the clear next step.

Colab notebook: [`proof/SCT_RankSweep_1_7B.ipynb`](proof/SCT_RankSweep_1_7B.ipynb) | Reports: [`docs/SCT_RankSweep_Report.pdf`](docs/SCT_RankSweep_Report.pdf)

### Compression Scales with Model Size

Per-MLP-layer training memory (weights + gradients + Adam states) at rank 32:

| Model | Layer (m × n) | Dense + Adam | SCT (k=32) | Compression |
|-------|---------------|-------------|-----------|-------------|
| SmolLM2-135M | 576 × 1536 | 14.2 MB | 1.1 MB | 13x |
| SmolLM2-1.7B | 2048 × 8192 | 268 MB | 5.2 MB | 51x |
| LLaMA-7B | 4096 × 11008 | 721 MB | 7.7 MB | 93x |
| Qwen-27B | 4096 × 17408 | 1,141 MB | 11.0 MB | 104x |
| LLaMA-70B | 8192 × 28672 | 3,758 MB | 18.9 MB | 199x |

The sweet spot is 1.7B+ where rank 32 gives 50x+ compression per layer.

---

## How It Works

### Forward Pass

```
h  = x @ U       # [batch, k]     project into spectral basis
hs = h * s       # [batch, k]     scale by singular values
y  = hs @ V.T    # [batch, out]   reconstruct in output space
```

Three small matmuls. Cost: O(bk(m+n)) instead of O(bmn). The m×n weight matrix is never built.

### Backward Pass

PyTorch autograd computes gradients dL/dU, dL/ds, dL/dV through the same three operations. Gradient shapes are (m×k), (k,), (n×k). No m×n gradient ever exists.

This is the key practical distinction from prior work: LoRA keeps the full dense model in memory and trains small adapters alongside it; GaLore computes dense gradients then projects them into a low-rank subspace; SCT never materializes any dense matrix at any point during training.

**Note on "exact" gradients:** The gradients are exact with respect to the factored parameterization `W = U diag(s) V^T`. They are not identical to the gradients of a full-rank dense model, because the rank-constrained model defines a different loss landscape.

### Stiefel Retraction

After Adam updates U and V, they drift off the orthonormal manifold. QR retraction fixes this:

```python
Q, R = torch.linalg.qr(U_updated)
U = Q * torch.sign(torch.diag(R))  # sign correction for stability
```

Cost: O(mk²) per layer. This is what makes SCT a *training* method, not just compression. The manifold constraint is maintained throughout optimization, not applied once post-hoc.

---

## Related Work

SCT builds on ideas from several lines of research. The individual components (SVD factorization, Stiefel manifold optimization, low-rank training) are well-studied. The specific combination (permanent truncated SVD storage with layer-local sigma updates from residual projection and Riemannian U/V rotation via QR retraction, targeting pre-training from scratch without backpropagation through the full dense model) appears to be novel.

**Low-rank adaptation and fine-tuning.** LoRA [2] trains small low-rank adapter matrices alongside frozen pre-trained weights. The full dense model remains in memory. SCT replaces the dense matrices entirely: the spectral factors *are* the weights. StelLA (arXiv 2510.01938) uses a similar U·S·V^T decomposition with Stiefel constraints, but applies it to LoRA adapters for fine-tuning, not as a standalone training method.

**SVD-based compression.** SVD-LLM [3] and Decomposable-Net [4] perform post-training SVD truncation on already-trained dense models. This destroys learned capacity because the network used its full spectral budget during training. SCT trains natively in low-rank form from the start, so the network learns to distribute capacity within the available spectral budget. This is analogous to why a 7B model trained from scratch outperforms a 70B model naively pruned to 7B.

**Memory-efficient gradient methods.** GaLore [5] projects dense gradients into low-rank subspaces via periodic SVD, reducing optimizer state memory while keeping full-rank weights and gradients. SCT avoids dense gradients entirely by differentiating through the small spectral factors directly.

**Low-rank training.** SVD Training (arXiv 2004.09031) decomposes weights into full SVD form and trains the components with orthogonality regularization, but still uses standard backpropagation for gradient computation. OIALR (arXiv 2401.08505) observes that orthogonal bases stabilize during training, then freezes U/V and trains only the core matrix. Both approaches still rely on backpropagation through the full model.

**Backpropagation-free methods.** NoProp (arXiv 2503.24322), Mono-Forward (arXiv 2501.09238), and FFzero (arXiv 2603.24790) pursue backpropagation elimination through completely different mechanisms (diffusion-inspired denoising, forward-forward variants). None use spectral decomposition as their weight representation.

**Riemannian optimization.** Optimization on the Stiefel manifold [1] and efficient retractions via Cayley transforms [6] are established techniques. SCT applies these specifically to maintain orthonormality of spectral factors during neural network training.

**Low-rank neural network patents.** US Patent Application 20250021826 [7] covers low-rank compression of trained networks. SCT addresses a different problem: training natively in low-rank form.

### References

```
[1] Absil et al. (2008). Optimization Algorithms on Matrix Manifolds. Princeton.
[2] Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
[3] Wang et al. (2024). SVD-LLM: Truncation-aware SVD for LLM Compression. arXiv:2403.07378.
[4] Yaguchi et al. (2021). Decomposable-Net: Scalable Low-Rank Compression. IJCAI.
[5] Zhao et al. (2024). GaLore: Memory-Efficient LLM Training. ICML. arXiv:2403.03507.
[6] Li et al. (2020). Efficient Riemannian Optimization via Cayley Transform. ICLR.
[7] US Patent Application 20250021826 (2025). Low-Rank Compression of Neural Networks.
```

---

## Limitations

**Rank constrains expressivity.** A rank-k factorization can only represent a rank-k weight matrix. If the task requires higher effective rank, the model will underperform a dense equivalent. However, the 1.7B rank sweep shows that all ranks (32–256) converge to the same loss floor, suggesting that at practical training durations, MLP rank may not be the primary bottleneck.

**Convergence gap vs dense.** The 1.7B rank sweep shows a ~3 loss gap between SCT and dense after 2000 steps. The rank sweep evidence suggests this gap is driven by the shared learning rate across all model components (attention layers are 77% of the SCT model's parameters and are trained at 25x the dense baseline LR), not by MLP rank capacity. Per-component LR scheduling is the clear next step.

**QR retraction cost.** At O(mk²) per layer per step, retraction is cheap for small k but becomes a meaningful fraction of step time. The 70B benchmark shows retraction taking ~40-50% of total step time. At 1.7B scale on A100, retraction overhead is negligible (0.56s total step at rank 32).

**Strongest for pre-training.** When converting pre-trained dense weights to spectral form, the network has already learned to use its full spectral budget. Energy-based rank selection (retaining 95%+ of singular value energy) partially mitigates this, but the rank constraint inevitably loses information. The 1.7B experiments use hard rank caps (32–256) rather than energy thresholds.

**Small models benefit less.** Models below ~1.7B parameters (hidden dim < 2048) produce ranks close to the full dimension at practical energy thresholds, offering little compression. SCT compression scales with the ratio of layer dimension to rank.

---

## Installation

```bash
git clone https://github.com/EctoSpace/SCT.git
cd SCT
pip install -e .
```

Or without installation, just clone and run the examples directly.

### Quick Start: 70B Architecture Test

Fits on any machine with 8+ GB RAM:

```bash
pip install torch transformers
python examples/sct_steamdeck.py
```

### Quick Start: Fine-Tuning SmolLM2

```bash
pip install torch transformers datasets
python examples/macbook_m4pro/sct_smollm2.py --energy 0.95 --steps 400
```

### Quick Start: Head-to-Head Dense vs SCT

```bash
pip install torch transformers datasets
python examples/macbook_m4pro/sct_vs_dense.py --energy 0.95 --steps 400
```

### Quick Start: 1.7B Rank Sweep (Colab)

Open [`proof/SCT_RankSweep_1_7B.ipynb`](proof/SCT_RankSweep_1_7B.ipynb) in Google Colab with an A100 GPU. Runs dense baseline + SCT at ranks 32, 64, 128, 256 in ~2.5 hours (~14.5 compute units).

---

## Core Implementation

The entire method is one class:

```python
from spectral_compact_training import SpectralLinear, retract_all

# Drop-in replacement for nn.Linear
layer = SpectralLinear(in_features=4096, out_features=11008, rank=32)
y = layer(x)

# After optimizer.step(), retract to Stiefel manifold
optimizer.step()
retract_all(model)
```

Convert a pre-trained dense layer:

```python
dense_layer = nn.Linear(4096, 11008)
spectral_layer = SpectralLinear.from_linear(dense_layer, rank=32)
```

---

## Repository Structure

```
spectral_compact_training/            Core library
  __init__.py
  spectral_layer.py                   SpectralLinear + retract_all
  mlp_debug.py                        MLP from-scratch training proof
  mlp_proof_results.json              MLP benchmark results

examples/
  sct_70b_flex.py                     70B on M4 Pro (MPS backend)
  sct_smollm2.py                      SmolLM2 fine-tuning on Alpaca
  sct_steamdeck.py                    70B architecture validation (any hardware)
  sct_vs_dense.py                     Head-to-head Dense vs SCT comparison
  sct_convergence_1.7B.pz             Dense vs SCT rank 32 (SmolLM2-1.7B)
  sct_convergence_1_7B.ipynb          Colab: Dense vs SCT rank 32 (SmolLM2-1.7B)
  sct_RankSweep_1_7B.ipynb            Colab: Rank sweep 32/64/128/256

results/
  mac/
    sct_70b_flex_console.txt          70B console output
    sct_70b_memory_results.json       70B M4 Pro benchmark results
    sct_smollm2_results.json          SmolLM2 fine-tuning results
    sct_smollm2_console.txt           SmolLM2 console output
    sct_vs_dense_results.json         Dense vs SCT comparison results
    sct_vs_dense_console.txt          Dense vs SCT console output
  steamDeck/
    SteamDeck-Demo.mp4                Video: Steam Deck running 70B step
    SteamDeck-Konsole.mp4             Video: terminal output
    SteamDeck-Konsole-Output.txt      Raw console log
  a100/
    sct_conv_dense_losses.json        Convergence Dense baseline (1.7B, 2000 steps)
    sct_conv_summary_colab.json       Convergence summary metrics
    sct_rank_sweep_dense_losses.json  Rank Sweep Dense baseline (1.7B, 2000 steps)
    sct_rank_sweep_r32_losses.json    SCT rank 32 loss history
    sct_rank_sweep_64_losses.json     SCT rank 64 loss history
    sct_rank_sweep_r128_losses.json   SCT rank 128 loss history
    sct_rank_sweep_r256_losses.json   SCT rank 256 loss history
    sct_rank_sweep_summary.json           Rank sweep summary metrics

docs/
  SCT_Patent_Application.pdf          Patent specification
  SCT_Convergence_Report.pdf          Convergence experiment report
  SCT_RankSweep_Report.pdf            Rank sweep report
  patent_pending.webp                 Filing confirmation
```

---

## Important Notes

**What SCT is:** A training method that stores and updates weights exclusively in spectral form with gradients through the factored parameterization and Stiefel manifold constraints. The 70B result is architectural validation: a full training step (forward, backward, optimizer, retraction) completes in 7.2 GB of memory on a Steam Deck (16 GB LPDDR5, Zen 2 CPU). The 1.7B rank sweep validates memory efficiency and convergence on a real LLM fine-tuning task.

**What SCT is not:** A finished 70B language model. Training to convergence requires compute time proportional to the dataset, which SCT does not change. SCT changes *how much memory* you need to do that training. The 1.7B experiments show a loss gap vs dense that appears to be caused by learning rate configuration, not the SCT method itself.

**vs LoRA:** LoRA keeps the full dense model in memory and trains small adapter matrices alongside it. SCT replaces the dense matrices entirely. The spectral factors *are* the weights. LoRA is a fine-tuning add-on; SCT is a different representation of the model itself.

**Current status (March 2026):** Memory efficiency and training throughput improvements are empirically validated at 1.7B scale on A100. Closing the convergence gap via per-component learning rates is the active research direction.

---

## Citation

```bibtex
@misc{kohlberger2026sct,
  title={Spectral Compact Training: Memory-Efficient Neural Network Training
         via Truncated SVD Factorization with Stiefel Manifold Retraction},
  author={Kohlberger, Bj{\"o}rn Roman},
  year={2026},
  note={Irish Patent Application PTIE20260000000219}
}
```

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

## Author

**Björn Roman Kohlberger** — [EctoSpace](https://github.com/EctoSpace)