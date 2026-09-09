"""Step 2 -- train, compare against the baseline, and show the leak."""

from __future__ import annotations

import sys
from pathlib import Path

from ..context import LadderConfig, write_result
from .prepare import load_prepared

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import model as model_mod  # noqa: E402


def run(cfg: LadderConfig) -> dict:
    df = load_prepared(cfg)
    spec = model_mod.Spec.from_config(cfg)
    trained = model_mod.train_model(df, spec)
    honest, leaky = model_mod.leakage_demo(df, spec)

    line = (
        f"baseline {trained.baseline.accuracy:.1%} acc / "
        f"{trained.baseline.recall:.0%} recall · "
        f"model {trained.metrics.accuracy:.1%} / {trained.metrics.recall:.1%} / "
        f"AUC {trained.metrics.roc_auc:.2f} · "
        f"leak {leaky.recall:.1%} vs {honest.recall:.1%}"
    )
    return write_result(
        cfg,
        "model",
        "PASS",
        line,
        {
            "baseline": trained.baseline.as_row(),
            "model": trained.metrics.as_row(),
            "leaky": leaky.as_row(),
            "honest": honest.as_row(),
            "numeric": trained.numeric,
            "categorical": trained.categorical,
            "version": trained.version,
        },
    )
