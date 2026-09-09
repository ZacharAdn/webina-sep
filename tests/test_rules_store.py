from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import agent as agent_mod  # noqa: E402
import rules_store  # noqa: E402


def test_from_row_accepts_jsonb_as_list_or_string():
    row = {"version": 4, "bands": '[{"name":"low","min":0,"action":"x"},{"name":"high","min":0.5,"action":"y"}]', "source": "learner:rules"}
    got = rules_store.from_row(row)
    assert got.version == 4 and got.bands[0].name == "high"


def test_load_active_without_connection_is_toml_version_one():
    got = rules_store.load_active(None, ROOT)
    assert got.version == 1 and got.source == "toml"


def test_agent_bands_carry_the_version_into_the_reason():
    bands = agent_mod.Bands.load(ROOT)
    rec = bands.recommend({"probability": 0.9})
    assert "rules v1" in rec.reason
    assert rec.source == "rules"


def test_agent_bands_from_a_revised_ruleset_flip_the_recommendation():
    revised = rules_store.RuleSet(2, rules_store.normalise([
        {"name": "high", "min": 0.95, "action": "Call"},
        {"name": "medium", "min": 0.35, "action": "List"},
        {"name": "low", "min": 0.0, "action": "No action"},
    ]), "learner:rules")
    before = agent_mod.Bands.load(ROOT).recommend({"probability": 0.9}).action
    after = agent_mod.Bands.from_rules(revised).recommend({"probability": 0.9}).action
    assert before != after and after == "List"
