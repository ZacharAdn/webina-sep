"""Step 3 -- run the app headless, press rung 4's button, count what happened.

AppTest executes the whole script, so every tab's body runs even though nothing
is clicked. That is the point: an exception hiding in tab 3 fails here rather
than on stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from ..context import LadderConfig, LadderError, write_result

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rungs import SUBHEADS  # noqa: E402  -- no Streamlit import, no page config

TIMEOUT = 300


def run(cfg: LadderConfig) -> dict:
    app_file = cfg.root / "src" / "app.py"
    if not app_file.exists():
        raise LadderError(f"{app_file} is missing")

    at = AppTest.from_file(str(app_file), default_timeout=TIMEOUT).run()
    if at.exception:
        raise LadderError(f"app raised on load: {at.exception[0].value.splitlines()[0]}")

    rendered = [element.value for element in at.subheader]
    rungs = sum(1 for head in SUBHEADS if head in rendered)
    if rungs != 4:
        raise LadderError(f"{rungs} of 4 rungs rendered: {rendered}")

    write_buttons = [b for b in at.button if b.label.startswith("Score and write")]
    if not write_buttons:
        raise LadderError("rung 4's write button did not render")

    before = _prediction_count(cfg)
    at = write_buttons[0].click().run()
    if at.exception:
        raise LadderError(
            f"app raised on write: {at.exception[0].value.splitlines()[0]}"
        )
    after = _prediction_count(cfg)
    written = max(after - before, 0)

    destination = (
        "Supabase"
        if any("written to predictions" in s.value for s in at.success)
        else "data/predictions.csv"
    )
    line = f"4 rungs · 0 exceptions · {written} rows to {destination}"
    return write_result(
        cfg,
        "app",
        "PASS",
        line,
        {
            "rungs": rungs,
            "exceptions": 0,
            "rows_written": written,
            "destination": destination,
        },
    )


def _prediction_count(cfg: LadderConfig) -> int:
    path = cfg.root / "data" / "predictions.csv"
    if not path.exists():
        return 0
    return max(sum(1 for _ in path.open(encoding="utf-8")) - 1, 0)
