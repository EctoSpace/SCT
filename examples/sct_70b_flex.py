import math
import time
import os
import json
import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def safe_qr(M):
    dev = M.device
    Q, R = torch.linalg.qr(M.cpu() if dev.type == "mps" else M)
    return (Q * torch.sign(torch.diag(R))).to(dev)

class SpectralLinearDummy(nn.Module):
    def __init__(self, in_features, out_features, rank=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        U = torch.randn(in_features, rank) / math.sqrt(in_features)
        V = torch.randn(out_features, rank) / math.sqrt(out_features)
        self.U = nn.Parameter(safe_qr(U)[:, :rank])
        self.V = nn.Parameter(safe_qr(V)[:, :rank])
        self.s = nn.Parameter(torch.ones(rank))

    def forward(self, x):
        return (x @ self.U) * self.s @ self.V.T

    @torch.no_grad()
    def retract(self):
        self.U.data = safe_qr(self.U.data)[:, :self.rank]
        self.V.data = safe_qr(self.V.data)[:, :self.rank]
        
    @torch.no_grad()
    def check_ortho_error(self):
        eye = torch.eye(self.rank, device=self.U.device)
        err_U = torch.norm(self.U.T @ self.U - eye)
        err_V = torch.norm(self.V.T @ self.V - eye)
        return max(err_U.item(), err_V.item())

class DummySpectral70BLayer(nn.Module):
    def __init__(self, config, rank=32):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.q_proj = SpectralLinearDummy(config.hidden_size, config.hidden_size, rank=rank)
        self.k_proj = SpectralLinearDummy(config.hidden_size, config.hidden_size, rank=rank)
        self.v_proj = SpectralLinearDummy(config.hidden_size, config.hidden_size, rank=rank)
        self.o_proj = SpectralLinearDummy(config.hidden_size, config.hidden_size, rank=rank)
        self.gate_proj = SpectralLinearDummy(config.hidden_size, config.intermediate_size, rank=rank)
        self.up_proj = SpectralLinearDummy(config.hidden_size, config.intermediate_size, rank=rank)
        self.down_proj = SpectralLinearDummy(config.intermediate_size, config.hidden_size, rank=rank)

    def forward(self, x):
        norm_x = self.input_layernorm(x)
        q = self.q_proj(norm_x)
        k = self.k_proj(norm_x)
        v = self.v_proj(norm_x)
        
        attn_out = self.o_proj(q + k + v) 
        x = x + attn_out
        
        norm_x2 = self.post_attention_layernorm(x)
        mlp_out = self.down_proj(self.gate_proj(norm_x2) * self.up_proj(norm_x2))
        return x + mlp_out

    def retract_layer(self):
        self.q_proj.retract()
        self.k_proj.retract()
        self.v_proj.retract()
        self.o_proj.retract()
        self.gate_proj.retract()
        self.up_proj.retract()
        self.down_proj.retract()

def run_70b_memory_test():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n=======================================================")
    print(f"  SPECTRAL COMPACT TRAINING  70B ARCHITECTURE TEST")
    print(f"=======================================================")
    print(f"  Device: {device}")
    
    config = LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8
    )
    rank = 32

    dense_params_per_layer = 4 * (config.hidden_size**2) + 3 * (config.hidden_size * config.intermediate_size)
    total_dense_params = dense_params_per_layer * config.num_hidden_layers
    
    sct_attn_params = 4 * (2 * config.hidden_size * rank + rank)
    sct_mlp_params = 3 * ((config.hidden_size + config.intermediate_size) * rank + rank)
    sct_params_per_layer = sct_attn_params + sct_mlp_params
    total_sct_params = sct_params_per_layer * config.num_hidden_layers
    
    compression = total_dense_params / total_sct_params

    print(f"\n  [Architecture]")
    print(f"  Layers: {config.num_hidden_layers} | Hidden: {config.hidden_size} | FFN: {config.intermediate_size}")
    print(f"  Rank: {rank}")
    print(f"  Dense Equivalent:   {total_dense_params / 1e9:.2f} B parameters")
    print(f"  SCT Parameters:     {total_sct_params / 1e6:.2f} M parameters")
    print(f"  Compression Ratio:  {compression:.0f}x")

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n  [Initializing Model]")
    layers = nn.ModuleList([DummySpectral70BLayer(config, rank=rank) for _ in range(config.num_hidden_layers)]).to(device)
    opt = torch.optim.AdamW(layers.parameters(), lr=1e-3)
    
    dummy_input = torch.randn(1, 16, config.hidden_size, device=device)
    
    print(f"  [Executing Full Training Step]")
    
    t0 = time.time()
    opt.zero_grad()
    x = dummy_input
    for layer in layers:
        x = layer(x)
    loss = x.sum()
    t_fwd = time.time() - t0
    
    t0 = time.time()
    loss.backward()
    t_bwd = time.time() - t0
    
    t0 = time.time()
    opt.step()
    t_opt = time.time() - t0
    
    t0 = time.time()
    with torch.no_grad():
        for layer in layers:
            layer.retract_layer()
    t_ret = time.time() - t0
    
    total_time = t_fwd + t_bwd + t_opt + t_ret
    
    max_err = max(layers[0].q_proj.check_ortho_error(), layers[-1].down_proj.check_ortho_error())

    peak_mb = 0
    if device.type == "mps":
        peak_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
    elif device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            peak_mb = process.memory_info().rss / (1024 * 1024)
        else:
            peak_mb = -1 
            
    print(f"\n=======================================================")
    print(f"  RESULTS")
    print(f"=======================================================")
    if peak_mb > 0:
        print(f"  Estimated Memory:   {peak_mb:.1f} MB  ({peak_mb/1024:.2f} GB)")
    else:
        print(f"  Estimated Memory:   [Requires 'pip install psutil' for CPU]")
        
    print(f"  Forward Pass:       {t_fwd:.2f}s")
    print(f"  Backward Pass:      {t_bwd:.2f}s")
    print(f"  Optimizer Step:     {t_opt:.2f}s")
    print(f"  QR Retraction:      {t_ret:.2f}s")
    print(f"  ******************************")
    print(f"  Total Step Time:    {total_time:.2f}s")
    print(f"\n  Orthonormality Err: {max_err:.2e}")
    print(f"=======================================================\n")

    results_data = {
        "model": "LLaMA_3_70B_Class",
        "hardware": str(device),
        "config": {
            "layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "ffn_size": config.intermediate_size,
            "rank": rank
        },
        "parameters": {
            "dense_equivalent": total_dense_params,
            "sct_actual": total_sct_params,
            "compression_ratio": round(compression, 2)
        },
        "performance": {
            "forward_sec": round(t_fwd, 3),
            "backward_sec": round(t_bwd, 3),
            "optimizer_sec": round(t_opt, 3),
            "retraction_sec": round(t_ret, 3),
            "total_step_sec": round(total_time, 3)
        },
        "memory": {
            "peak_allocated_mb": round(peak_mb, 1) if peak_mb > 0 else None,
            "peak_allocated_gb": round(peak_mb / 1024, 2) if peak_mb > 0 else None
        },
        "stability": {
            "max_orthonormality_error": max_err
        }
    }

    output_file = "sct_70b_memory_results.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)
        
    print(f"  Results saved successfully to {output_file}")

if __name__ == "__main__":
    run_70b_memory_test()