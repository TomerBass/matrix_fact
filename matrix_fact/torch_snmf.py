# Authors: Samuel Cahyawijaya
# License: BSD 3 Clause
"""
PyTorch MatrixFact Semi Non-negative Matrix Factorization.

    TorchSNMF(NMF) : Class for semi non-negative matrix factorization with PyTorch

[1] Ding, C., Li, T. and Jordan, M.. Convex and Semi-Nonnegative Matrix Factorizations.
IEEE Trans. on Pattern Analysis and Machine Intelligence 32(1), 45-55.
"""
import torch
from .base import TorchMatrixFactBase

__all__ = ["TorchSNMF"]


class TorchSNMF(TorchMatrixFactBase):
    def __init__(self, data, num_bases=4, **kwargs):
        self.ridge = 1e-2          # optional LS stabilization
        self.eta_div = 1e-1        # nudge size (0 = off)
        self.div_mode = "repel"    # "repel" (more diverse) | "attract" (less) | "none"
        super().__init__(data, num_bases, **kwargs)

    def _update_w(self):
        # --- LS step (argmin ||X - W H||_F^2 w.r.t. W) ---
        # >>> ADDED: ridge and eye for stability
        k = self.H.shape[0]  # atoms
        ridge = getattr(self, "ridge", 0.0)  # small L2, e.g. 1e-4
        I = torch.eye(k, device=self.H.device, dtype=self.H.dtype)

        # >>> CHANGED: stabilized HH
        HH = self.H @ self.H.T + ridge * I
        self.W = self.data @ self.H.T @ torch.linalg.inv(HH)

        # --- Diversity control nudge on W (optional) ---
        # >>> ADDED: knobs
        eta_div = getattr(self, "eta_div", 0.0)          # step size, e.g. 1e-3
        div_mode = getattr(self, "div_mode", "none")     # "repel" | "attract" | "none"

        if eta_div > 0.0 and div_mode != "none":
            # >>> ADDED: gradient of R(W) = ||off(W^T W)||_F^2  ==>  ∇_W R = 4 W off(W^T W)
            G = self.W.T @ self.W
            off = G - torch.diag(torch.diagonal(G))
            grad = 4.0 * self.W @ off

            if div_mode == "repel":      # MORE diverse atoms
                self.W = self.W - eta_div * grad
            elif div_mode == "attract":  # LESS diverse atoms
                self.W = self.W + eta_div * grad

            # >>> ADDED: scaling invariance (keep WH unchanged; preserves H>=0)
            cn = self.W.norm(dim=0, keepdim=True).clamp_min(1e-12)
            self.W = self.W / cn
            self.H = cn.T * self.H

    def _update_h(self):
        def separate_positive(m):
            return (m.abs() + m) / 2.0

        def separate_negative(m):
            return (m.abs() - m) / 2.0

        XW = torch.mm(self.data[:, :].T, self.W)

        WW = torch.mm(self.W.T, self.W)
        WW_pos = separate_positive(WW)
        WW_neg = separate_negative(WW)

        XW_pos = separate_positive(XW)
        H1 = (XW_pos + torch.mm(self.H.T, WW_neg)).T

        XW_neg = separate_negative(XW)
        H2 = (XW_neg + torch.mm(self.H.T, WW_pos)).T + 1e-9

        self.H *= torch.sqrt(H1 / H2)
