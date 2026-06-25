"""
TC-KAN: Time-Conditioned Kolmogorov-Arnold layer.

Reimplementation of the layer and activation from
    Shen et al., "TC-KAN: Time-Conditioned Kolmogorov-Arnold Networks with
    Time-Dependent Activations for Long-Term Time Series Forecasting",
    Sensors 2026, 26, 2538.

The activation is written as a separate, heavily-commented method (`phi_t`) so the
time-conditioning logic is easy to read and audit.

----------------------------------------------------------------------------------
WATCH-NOTES on the activation (read before trusting the paper's equations verbatim)
----------------------------------------------------------------------------------
1) DOUBLE-COUNTED SiLU PATH.
   Eq. (11) defines  phi_t(x) = sum_k c_k(t) psi_k(x~) + W_base * sigma(x)
   Eq. (12) then writes  TCKAN(x) = W (phi_t(x) ⊙ s̄) + b + W_base * sigma(x).
   Taken literally the SiLU base path appears TWICE, and the first copy is also
   multiplied by (W ⊙ s̄), which standard KAN never does. Real KAN sums a spline
   branch and a SiLU branch ONCE each. Here `phi_t` returns ONLY the polynomial
   (spline) part; the SiLU base branch is added exactly once in `forward`.

2) "HAHN" RECURRENCE IS LEGENDRE.
   Eqs. (3)-(5) are the Legendre three-term recurrence evaluated on x~ = tanh(x).
   tanh keeps the argument in [-1, 1], which is exactly where this orthogonal
   system is well-conditioned, so higher orders stay bounded and stable.

3) COEFFICIENT COLLAPSE (Sec. 4.10).
   The paper's own analysis finds that after training delta_c_k(t) becomes
   constant across t (std_t < 1e-8) because ||W2|| -> ~1e-4. The mechanism then
   acts as a learned per-ORDER bias, not a per-TIMESTEP modulation. The warm-start
   init below (Eq. 13: W2 *= 0.1, b2 = 0) is what produces that behaviour and is
   the single most important init choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------
# Hahn / Legendre-like polynomial basis  (Eqs. 3-5)
# ----------------------------------------------------------------------------------
def hahn_basis(x_tilde: torch.Tensor, K: int) -> torch.Tensor:
    """Evaluate {psi_0, ..., psi_K} on a pre-normalised input.

    Recurrence (paper Eqs. 3-5), with x~ = tanh(x) supplied by the caller:
        psi_0 = 1
        psi_1 = x~
        psi_{k+1} = ((2k+1) x~ psi_k - k psi_{k-1}) / (k+1)

    Args:
        x_tilde: tensor of arbitrary shape [...], assumed already in [-1, 1]
                 (i.e. the caller passed tanh(x)).
        K:       maximum polynomial order (default model uses K = 3 -> 4 bases).

    Returns:
        Tensor [..., K+1] stacking psi_0..psi_K along the last axis.
    """
    psi = [torch.ones_like(x_tilde)]          # psi_0
    if K >= 1:
        psi.append(x_tilde)                   # psi_1
    for k in range(1, K):                     # produces psi_2 .. psi_K
        nxt = ((2 * k + 1) * x_tilde * psi[k] - k * psi[k - 1]) / (k + 1)
        psi.append(nxt)
    return torch.stack(psi, dim=-1)           # [..., K+1]


# ----------------------------------------------------------------------------------
# Time-Conditioned KAN layer
# ----------------------------------------------------------------------------------
class TimeConditionedKANLayer(nn.Module):
    r"""One TC-KAN layer.

    Pipeline for input x of shape [B, L, d_in]:

        x~          = tanh(x)                                   # normalise to [-1,1]
        psi_k(x~)   via Hahn/Legendre recurrence                # [B, L, d_in, K+1]
        e(t)        = Embedding(t),   t = 0..L-1                # [L, d_pos]
        Δc_k(t)     = W2 GELU(W1 e(t) + b1) + b2                # [L, K+1]   (Eq. 9)
        c_k(t)      = c_{k,base} + Δc_k(t)                      # [L, d_in, K+1] (Eq.10)
        phi_t(x)    = Σ_k c_k(t) psi_k(x~)   (POLY PART ONLY)   # [B, L, d_in] (Eq.11*)
        out         = W (phi_t ⊙ s̄) + b   +   W_base σ(x)      # (Eq. 12, SiLU added once)

    Coefficient layout follows the "efficient KAN" reading of Eq. (7): the basis
    expansion is applied per INPUT dimension, and the linear map W mixes inputs to
    outputs. The time offset Δc_k(t) has K+1 entries (paper: "output dimension
    K+1"), i.e. it is shared across input dims and varies per timestep and order.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        seq_len: int,
        K: int = 3,
        d_pos: int = 16,
        base_activation: type[nn.Module] = nn.SiLU,
        time_conditioned: bool = True,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.seq_len = seq_len
        self.K = K
        self.d_pos = d_pos
        self.time_conditioned = time_conditioned

        # --- polynomial (spline) branch ------------------------------------------
        # base coefficients c_{k,base}, per input dim and order.
        self.c_base = nn.Parameter(torch.empty(d_in, K + 1))
        # per-input scale s̄ (Eq. 7 reduces a per-edge scale to per-input via mean).
        self.s_bar = nn.Parameter(torch.ones(d_in))
        # mixer W [d_out, d_in] and bias b [d_out].
        self.W = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out))

        # --- SiLU residual branch (added ONCE, see watch-note 1) ------------------
        self.base_activation = base_activation()
        self.W_base = nn.Parameter(torch.empty(d_out, d_in))

        # --- time-conditioning sub-network (Eqs. 8-9) ----------------------------
        if time_conditioned:
            self.pos_embed = nn.Embedding(seq_len, d_pos)          # e(t), Eq. 8
            self.coef_mlp = nn.Sequential(                         # Δc_k(t), Eq. 9
                nn.Linear(d_pos, 2 * d_pos),                       # W1, b1
                nn.GELU(),
                nn.Linear(2 * d_pos, K + 1),                       # W2, b2
            )
        else:
            self.register_module("pos_embed", None)
            self.register_module("coef_mlp", None)

        self.reset_parameters()

    # ------------------------------------------------------------------------------
    def reset_parameters(self) -> None:
        # spline coeffs: small, so the layer starts near a gentle nonlinearity.
        nn.init.normal_(self.c_base, mean=0.0, std=0.1)
        nn.init.normal_(self.W, mean=0.0, std=1.0 / (self.d_in ** 0.5))
        nn.init.kaiming_uniform_(self.W_base, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

        if self.time_conditioned:
            # WARM-START. Eq. (13) prescribes W2 <- 0.1 * W2_init, b2 <- 0. In
            # practice that 0.1x scale does NOT make Δc_k(t) ~ 0, because the random
            # embedding e(t) and first MLP layer still yield O(1) pre-activations, so
            # the residual drift is comparable to the base-coeff noise. To realise the
            # paper's stated intent ("Δc_k(t) ≈ 0 at start -> TC-KAN ≈ standard KAN")
            # we zero-init the offset HEAD exactly. Gradients still flow into W2 via
            # d(Δc)/dW2 = GELU(...), so time-conditioning is learned from step 1; it
            # simply starts from the standard-KAN point. (Set the two lines below to
            # `out_lin.weight.mul_(0.1)` instead if you want the literal Eq. 13.)
            out_lin = self.coef_mlp[-1]
            with torch.no_grad():
                out_lin.weight.zero_()
                out_lin.bias.zero_()

    # ------------------------------------------------------------------------------
    def coefficients(self, device=None) -> torch.Tensor:
        """Return c_k(t) of shape [L, d_in, K+1] (Eq. 10).

        Useful for the activation-collapse analysis of Sec. 4.10: inspect
        c.std(dim=0) to see how much the coefficients actually vary across t.
        """
        if not self.time_conditioned:
            # broadcast the static base coefficients over all L positions.
            return self.c_base.unsqueeze(0).expand(self.seq_len, -1, -1)

        device = device or self.c_base.device
        t = torch.arange(self.seq_len, device=device)
        e_t = self.pos_embed(t)                       # [L, d_pos]
        delta = self.coef_mlp(e_t)                    # [L, K+1]   (Eq. 9)
        # c_base [d_in, K+1] + delta [L, K+1] -> [L, d_in, K+1]  (Eq. 10)
        return self.c_base.unsqueeze(0) + delta.unsqueeze(1)

    # ------------------------------------------------------------------------------
    def phi_t(self, x: torch.Tensor) -> torch.Tensor:
        """Time-conditioned activation -- POLYNOMIAL PART ONLY (Eq. 11 without the
        SiLU term; see watch-note 1). The SiLU residual is added later in forward.

        Args:
            x: [B, L, d_in]
        Returns:
            phi_t(x): [B, L, d_in], the per-input nonlinear response at each t.
        """
        x_tilde = torch.tanh(x)                       # Eq. 4 normalisation
        psi = hahn_basis(x_tilde, self.K)             # [B, L, d_in, K+1]

        c_t = self.coefficients(device=x.device)      # [L, d_in, K+1]
        # contract over the order axis k:
        #   psi [B, L, d_in, K+1]  *  c_t [1, L, d_in, K+1]  -> sum over last dim
        phi = (psi * c_t.unsqueeze(0)).sum(dim=-1)    # [B, L, d_in]
        return phi

    # ------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_in]  ->  out: [B, L, d_out]   (Eq. 12, corrected)."""
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"sequence length {x.size(1)} != layer seq_len {self.seq_len}; "
                "the positional embedding is indexed by absolute position."
            )

        phi = self.phi_t(x)                           # [B, L, d_in]  (poly only)
        phi = phi * self.s_bar                         # ⊙ s̄         (Eq. 12)

        spline_out = F.linear(phi, self.W)             # W (phi ⊙ s̄) -> [B, L, d_out]
        base_out = F.linear(self.base_activation(x), self.W_base)  # SiLU branch, ONCE
        return spline_out + base_out + self.bias       # + b


# ----------------------------------------------------------------------------------
# Minimal self-test / sanity checks
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, d_in, d_out = 4, 96, 64, 64

    layer = TimeConditionedKANLayer(d_in, d_out, seq_len=L, K=3, d_pos=16)
    x = torch.randn(B, L, d_in)

    # 1) shape
    y = layer(x)
    assert y.shape == (B, L, d_out), y.shape
    print(f"forward OK: {tuple(x.shape)} -> {tuple(y.shape)}")

    # 2) warm-start: Δc_k(t) == 0 at init, so c_k(t) == c_base for every t.
    c = layer.coefficients()                          # [L, d_in, K+1]
    drift = (c - layer.c_base.unsqueeze(0)).abs().max().item()
    print(f"warm-start max |Δc_k(t)| at init: {drift:.2e}  (should be 0)")

    # 3) collapse probe: variation of c across timesteps (Sec. 4.10 uses std_t).
    #    0 at init by construction; this is the quantity that stays ~0 after training.
    std_over_time = c.std(dim=0).mean().item()
    print(f"mean std_t[c_k(t)] at init: {std_over_time:.2e}")

    # 4) TC vs. static equivalence at init: a static-coeff layer should match closely.
    static = TimeConditionedKANLayer(d_in, d_out, L, K=3, time_conditioned=False)
    static.load_state_dict(
        {k: v for k, v in layer.state_dict().items() if k in static.state_dict()},
        strict=False,
    )
    with torch.no_grad():
        diff = (layer(x) - static(x)).abs().max().item()
    print(f"max |TC - static| at init: {diff:.2e}  (warm-start => near 0)")

    n_params = sum(p.numel() for p in layer.parameters())
    print(f"params in this single TC-KAN layer: {n_params:,}")