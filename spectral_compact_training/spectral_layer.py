"""
Spectral Compact Training — Core Layer
=======================================

Drop-in replacement for nn.Linear.
Stores weights as W = U diag(s) V^T.
Exact backprop through spectral factors.
QR retraction to Stiefel manifold after each optimizer step.

Patent Pending: PTIE20260000000219
"""

import math
import torch
import torch.nn as nn


def safe_qr(M: torch.Tensor) -> torch.Tensor:
    """QR decomposition with sign correction for stable column orientation.
    CPU fallback for MPS/AMD backends that have QR driver issues."""
    Q, R = torch.linalg.qr(M.cpu() if M.device.type in ("mps",) else M)
    return (Q * torch.sign(torch.diag(R))).to(M.device)


class SpectralLinear(nn.Module):
    """
    W = U diag(s) V^T — drop-in nn.Linear replacement.

    Forward:  y = (x @ U) * s @ V.T
    Memory:   O(k(m+n)) instead of O(mn)
    Backward: exact via autograd through 3 small matmuls

    After each optimizer step, call retract() to maintain
    orthonormality of U and V on the Stiefel manifold.

    Args:
        in_features:  input dimension (m)
        out_features: output dimension (n)
        rank:         spectral rank (k), clamped to min(m, n)
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)

        # Initialize U, V as random orthonormal via QR
        U = torch.randn(in_features, self.rank) / math.sqrt(in_features)
        V = torch.randn(out_features, self.rank) / math.sqrt(out_features)
        self.U = nn.Parameter(safe_qr(U)[:, :self.rank])
        self.V = nn.Parameter(safe_qr(V)[:, :self.rank])
        self.s = nn.Parameter(torch.ones(self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: y = (x @ U) * s @ V^T"""
        return (x @ self.U) * self.s @ self.V.T

    @torch.no_grad()
    def retract(self):
        """QR retraction — project U, V back onto the Stiefel manifold.
        Must be called after each optimizer step."""
        self.U.data = safe_qr(self.U.data)[:, :self.rank]
        self.V.data = safe_qr(self.V.data)[:, :self.rank]

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 32) -> "SpectralLinear":
        """Convert a pretrained nn.Linear to SpectralLinear via truncated SVD."""
        W = linear.weight.data.float().cpu()
        m, n = W.shape  # [out_features, in_features]
        k = min(rank, m, n)

        U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)

        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.in_features = n
        layer.out_features = m
        layer.rank = k

        # Map SVD convention to our forward convention:
        # SVD: W = U_full @ diag(S) @ Vh_full
        # Ours: y = x @ U_ours * s @ V_ours.T
        # So: U_ours = Vh[:k]^T, V_ours = U_full[:,:k]
        layer.U = nn.Parameter(Vh_full[:k, :].T.contiguous())
        layer.V = nn.Parameter(U_full[:, :k].contiguous())
        layer.s = nn.Parameter(S_full[:k].contiguous())

        return layer

    def param_count(self) -> int:
        """Total trainable parameters in this layer."""
        return self.U.numel() + self.V.numel() + self.s.numel()

    def compression_ratio(self) -> float:
        """Ratio of dense params to spectral params."""
        dense = self.in_features * self.out_features
        return dense / self.param_count()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, compression={self.compression_ratio():.1f}x")


def retract_all(module: nn.Module):
    """Retract all SpectralLinear layers in a module tree.
    Call this after every optimizer.step()."""
    for m in module.modules():
        if isinstance(m, SpectralLinear):
            m.retract()