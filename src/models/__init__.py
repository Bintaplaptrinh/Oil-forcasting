from .baseline_lgbm import tune_lgbm, train_lgbm
from .jump_gated_arimax_catboost import (
    JumpGatedConfig,
    add_shock_features,
    run_jump_gated_arimax_catboost,
    run_jump_gated_arimax_catboost_horizon,
    run_multihorizon_jump_gated_arimax_catboost,
)

_HYBRID_EXPORTS = {
    "build_itransformer",
    "train_itransformer",
    "dual_mae_loss",
    "build_gumnet_lite",
    "train_gumnet_lite",
    "tune_itransformer",
    "build_gumnet_ultra",
    "train_gumnet_ultra",
    "tune_gumnet_ultra",
}

__all__ = [
    "tune_lgbm",
    "train_lgbm",
    "JumpGatedConfig",
    "add_shock_features",
    "run_jump_gated_arimax_catboost",
    "run_jump_gated_arimax_catboost_horizon",
    "run_multihorizon_jump_gated_arimax_catboost",
    *_HYBRID_EXPORTS,
]


def __getattr__(name):
    if name in _HYBRID_EXPORTS:
        from . import hybrid_sota

        value = getattr(hybrid_sota, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
