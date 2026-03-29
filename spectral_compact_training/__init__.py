"""Spectral Compact Training — W = U diag(s) V^T with Stiefel QR retraction."""

from .spectral_layer import SpectralLinear, safe_qr, retract_all

__version__ = "0.1.0"
__all__ = ["SpectralLinear", "safe_qr", "retract_all"]