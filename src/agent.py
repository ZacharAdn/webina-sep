"""Rung 3 -- the agent that recommends an action.

The house rule from the script: a rules layer is the baseline of a
recommendation exactly the way a dummy model is the baseline of a prediction.
The language model is only worth its latency if it beats the rules, and on
stage that is an open question, not a claim.

The rules are probability bands, read out of ladder.toml at call time. Editing
a band's floor or its action text and rerunning is the live-fix moment: the
recommendation flips while everyone watches, and no Python was touched.

Bands.recommend always works, offline, with no key.
recommend_llm is the optional second opinion, served by gpt-oss-120b on Groq
over its OpenAI-compatible endpoint; without GROQ_API_KEY it returns None and
the app says so instead of pretending.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Recommendation:
    action: str
    reason: str
    source: str            # "rules" or "groq"
    unknowns: str = ""


@dataclass(frozen=True)
class Band:
    name: str
    minimum: float
    action: str


@dataclass(frozen=True)
class Bands:
    bands: tuple[Band, ...]

    @classmethod
    def load(cls, root: Path | None = None) -> "Bands":
        path = Path(root or REPO_ROOT) / "ladder.toml"
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        return cls(
            tuple(
                Band(key, float(value["min"]), str(value["action"]))
                for key, value in sorted(
                    raw["bands"].items(), key=lambda kv: -float(kv[1]["min"])
                )
            )
        )

    def recommend(self, record: dict,
                  probability_key: str = "probability") -> Recommendation:
        probability = float(record.get(probability_key) or 0.0)
        for band in self.bands:
            if probability >= band.minimum:
                return Recommendation(
                    band.action,
                    f"Risk {probability:.0%} is in the '{band.name}' band, "
                    f"which starts at {band.minimum:.0%}.",
                    "rules",
                )
        last = self.bands[-1]
        return Recommendation(
            last.action, f"Risk {probability:.0%} is below every band.", "rules"
        )


def llm_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def recommend_llm(record: dict, rules: Recommendation,
                  model: str = DEFAULT_MODEL) -> Recommendation | None:
    """Ask gpt-oss-120b for a second opinion. Returns None when it cannot run.

    Groq speaks the OpenAI wire format, so the OpenAI SDK pointed at
    GROQ_BASE_URL is the whole integration. reasoning_effort is a gpt-oss knob
    and json_object mode is what keeps the reply parseable without a retry.
    """
    if not llm_available():
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL
        )
        prompt = (
            "You are advising a retention team. Here is one record:\n"
            f"{json.dumps(record, default=str, ensure_ascii=False)}\n\n"
            f"A rules engine recommends: {rules.action} ({rules.reason})\n\n"
            "Reply as JSON with exactly these keys: action (one concrete step), "
            "reason (one sentence, grounded in the fields above), unknowns "
            "(what this data does not tell you and would change the answer). "
            "Do not invent facts that are not in the record."
        )
        response = client.chat.completions.create(
            model=model,
            max_tokens=500,
            reasoning_effort="low",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json").strip()
        payload = json.loads(text)
        return Recommendation(
            action=str(payload.get("action", "")).strip(),
            reason=str(payload.get("reason", "")).strip(),
            source="groq",
            unknowns=str(payload.get("unknowns", "")).strip(),
        )
    except Exception:  # noqa: BLE001 -- on stage a failure must not stop the demo
        return None
