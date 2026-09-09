# Rung 3 conversation and rung 4 loop view — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rung 3 opens on the customer scored on rung 2, carries a console where the agent states the rules and proposes changes that apply on a click, and rung 4 shows the loop — rules history and the verdicts that drove it.

**Architecture:** Two new Streamlit-free modules carry the logic (`handoff.py` for the rung 2 to rung 3 record, `rules_chat.py` for the console); `app.py` only renders. Proposals from the console pass `rules_store.normalise` — the same gate the learner uses — and publish through `rules_store.publish`, so there is one write path for rules. Rung 4 reads `rules_store.history` and `data.read_feedback`; nothing new is written.

**Tech Stack:** Python 3.14, Streamlit (`st.chat_message`, `st.chat_input`, `st.session_state`), OpenAI SDK against Groq (`openai/gpt-oss-120b`, `reasoning_effort="low"`), Supabase via `st_supabase_connection`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-rung3-chat-rung4-loop-design.md`

## Global Constraints

- Project venv: `./venv_webina-sep/bin/python` for every command (pytest, ladder.py).
- Modules under `src/` are imported bare (`import rules_store`), never as a package; `src/` is on `sys.path` in tests and in `app.py`.
- No Streamlit import in `handoff.py` or `rules_chat.py`.
- The console never writes to Supabase except through `rules_store.publish(conn, bands, source, rationale, evidence)`; source string is `"chat:groq"`.
- Band validation is only ever `rules_store.normalise(list[dict]) -> tuple[Band, ...]`; band names must equal the active set's names.
- Groq client pattern copied from `learner.propose_llm`: `OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")`, `reasoning_effort="low"`. The console does NOT use `response_format=json_object` — its reply is prose with an optional fenced block.
- The console mirrors the user's language (Hebrew in, Hebrew out); all app copy stays English.
- Other agent's files are read-only for this plan except the one-line change in Task 1: `rules_store.py` (history select), `learner.py`, `data.py`, the migration, `tests/test_learner.py`, `tests/test_rules_store.py`.
- Every task ends with `./venv_webina-sep/bin/python -m pytest tests/ -q` green and a commit ending in `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Render check for any `app.py` change: `./venv_webina-sep/bin/python ladder.py app` must print `app: PASS — 4 rungs · 0 exceptions · ...`.

---

## File structure

    src/handoff.py             new   the record that crosses from rung 2 to rung 3 (dict in, dict out)
    src/rules_chat.py          new   opening message, system prompt, reply, block extraction
    src/rules_store.py         edit  history(): add `bands` to the select (one line)
    src/app.py                 edit  scoring_form stash · rung 3 source block · rules_chat_panel · rung 4 sections
    tests/test_handoff.py      new
    tests/test_rules_chat.py   new
    tests/test_rules_store.py  edit  one skip-without-connection test for history bands
    STATUS.md, WorkLog.md      edit  per task

Deviation from spec §5, stated: the spec put the stash helper in `app.py`; it lives in `src/handoff.py` so it is testable with a plain dict, as spec §4 requires.

---

### Task 1: `rules_store.history` returns the bands

**Files:**
- Modify: `src/rules_store.py` (function `history`, the `.select(...)` string)
- Test: `tests/test_rules_store.py` (append one test)

**Interfaces:**
- Consumes: nothing new.
- Produces: `rules_store.history(conn, limit=10) -> list[dict]` where each dict now also has key `"bands"` (a list of `{"name","min","action"}` dicts, or a JSON string PostgREST may return — `rules_store.from_row` already handles both). Task 6 depends on this.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rules_store.py`:

```python
def test_history_rows_carry_the_bands():
    """Rung 4 shows before/after between the last two versions; it needs the bands."""
    import os, tomllib
    from pathlib import Path

    secrets = ROOT / ".streamlit" / "secrets.toml"
    if not secrets.exists():
        pytest.skip("no secrets.toml -- no live rules table to read")
    for key, value in tomllib.loads(secrets.read_text()).items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
    import data as data_mod
    conn = data_mod.get_connection()
    if conn is None:
        pytest.skip("no Supabase connection")

    rows = rules_store.history(conn, limit=2)

    assert rows, "rules v1 was seeded by the migration"
    assert "bands" in rows[0]
    assert rules_store.from_row(rows[0]).bands  # parses either list or JSON string
```

If `tests/test_rules_store.py` does not already define `ROOT` and import `rules_store` at module level, add at its top:

```python
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rules_store  # noqa: E402
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv_webina-sep/bin/python -m pytest tests/test_rules_store.py::test_history_rows_carry_the_bands -v`
Expected: FAIL with `AssertionError` on `assert "bands" in rows[0]` (the select does not include it). If it SKIPS, the secrets file is missing — the test is still correct; continue.

- [ ] **Step 3: Write minimal implementation**

In `src/rules_store.py`, function `history`, change the select string:

```python
            conn.table(RULES_TABLE).select("version,source,rationale,active,created_at,bands")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv_webina-sep/bin/python -m pytest tests/test_rules_store.py -v`
Expected: all PASS (or the new one SKIPPED without secrets).

- [ ] **Step 5: Commit**

```bash
git add src/rules_store.py tests/test_rules_store.py
git commit -m "$(cat <<'MSG'
rules_store: history returns the bands, so rung 4 can diff versions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 2: `handoff.py` — the record that crosses from rung 2 to rung 3

**Files:**
- Create: `src/handoff.py`
- Test: `tests/test_handoff.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `handoff.KEY = "scored"` — the session-state key.
  - `handoff.stash_scored(session: dict, record: dict, probabilities: dict[str, float], typed: dict) -> dict` — stores `{"record", "probability", "typed", "id", "at"}` under `session[KEY]` and returns it. `id` is `"rung2:" + 8 hex chars`, deterministic in `typed`.
  - `handoff.default_record(session: dict, estimator: str, id_column: str) -> dict | None` — the rung-3 record: the stashed record plus `record[id_column] = id` and `record["probability"] = probabilities[estimator]`; `None` when nothing is stashed or the estimator is missing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_handoff.py`:

```python
"""The customer scored on rung 2 is the customer rung 3 opens on."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import handoff  # noqa: E402

TYPED = {"contract": "Month-to-month", "tenure": 1.0, "monthly_charges": 95.0}
RECORD = {**TYPED, "internet_service": "Fiber optic", "total_charges": 95.0}


def test_default_record_is_none_before_rung_2_ran():
    assert handoff.default_record({}, "logreg", "customer_id") is None


def test_stash_then_default_carries_the_record_id_and_the_estimators_probability():
    session: dict = {}
    handoff.stash_scored(session, RECORD, {"logreg": 0.613, "tree": 0.892}, TYPED)

    record = handoff.default_record(session, "logreg", "customer_id")

    assert record["customer_id"].startswith("rung2:")
    assert record["probability"] == 0.613
    assert record["contract"] == "Month-to-month"
    assert record["total_charges"] == 95.0


def test_default_record_is_none_when_the_estimator_was_not_scored():
    session: dict = {}
    handoff.stash_scored(session, RECORD, {"tree": 0.892}, TYPED)

    assert handoff.default_record(session, "logreg", "customer_id") is None


def test_the_id_is_deterministic_in_the_typed_values():
    a, b = {}, {}
    handoff.stash_scored(a, RECORD, {"logreg": 0.6}, TYPED)
    handoff.stash_scored(b, dict(RECORD), {"logreg": 0.7}, dict(TYPED))

    assert a[handoff.KEY]["id"] == b[handoff.KEY]["id"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv_webina-sep/bin/python -m pytest tests/test_handoff.py -v`
Expected: 4 FAIL / ERROR with `ModuleNotFoundError: No module named 'handoff'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/handoff.py`:

```python
"""The record that crosses from rung 2 to rung 3.

Rung 2's form scores a customer who is not in the table. Rung 3 opens on that
same customer, so the demo reads as one story rather than two tabs that happen
to share a model. The state lives in st.session_state; this module only ever
sees a dict, so it can be tested without Streamlit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

KEY = "scored"
ID_PREFIX = "rung2:"


def stash_scored(session: dict, record: dict, probabilities: dict[str, float],
                 typed: dict) -> dict:
    """Remember the customer rung 2 just scored. Returns what was stored.

    The id is a hash of the typed fields, so scoring the same customer twice
    yields the same id -- which is what lets feedback rows attach to it.
    """
    digest = hashlib.sha1(
        json.dumps(typed, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    session[KEY] = {
        "record": dict(record),
        "probability": {k: float(v) for k, v in probabilities.items()},
        "typed": dict(typed),
        "id": ID_PREFIX + digest,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return session[KEY]


def default_record(session: dict, estimator: str, id_column: str) -> dict | None:
    """The rung-3 record built from the stash, or None when rung 2 has not run.

    The recommender reads one probability -- the configured estimator's -- so
    that is the one written into the record; the other family's number is
    display only and stays in the stash.
    """
    scored = session.get(KEY)
    if not scored or estimator not in scored.get("probability", {}):
        return None
    record = dict(scored["record"])
    record[id_column] = scored["id"]
    record["probability"] = float(scored["probability"][estimator])
    return record
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv_webina-sep/bin/python -m pytest tests/test_handoff.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/handoff.py tests/test_handoff.py
git commit -m "$(cat <<'MSG'
handoff: the customer scored on rung 2 becomes rung 3's record

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: `rules_chat.py` — the console's logic, no Streamlit

**Files:**
- Create: `src/rules_chat.py`
- Test: `tests/test_rules_chat.py`

**Interfaces:**
- Consumes: `rules_store.Band`, `rules_store.RuleSet` (fields `version`, `bands`, `as_rows()`, `band_for(p)`), `rules_store.normalise`.
- Produces:
  - `rules_chat.ChatTurn(text: str, bands: tuple[Band, ...] | None = None, rationale: str = "")`
  - `rules_chat.llm_available() -> bool`
  - `rules_chat.opening_message(rules: RuleSet, record: dict, probability_key="probability") -> str`
  - `rules_chat.system_prompt(rules: RuleSet, digest_rows: list[dict], record: dict) -> str`
  - `rules_chat.extract_bands(text: str, current: RuleSet | None = None) -> tuple[tuple[Band, ...] | None, str]`
  - `rules_chat.strip_block(text: str) -> str`
  - `rules_chat.reply(history: list[dict], system: str, current: RuleSet | None = None, client=None, model=DEFAULT_MODEL) -> ChatTurn | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rules_chat.py`:

```python
"""The console: says what the rules are, proposes changes, never writes."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules_chat  # noqa: E402
from rules_store import Band, RuleSet  # noqa: E402

V1 = RuleSet(1, (
    Band("high", 0.60, "Contact this week and make a concrete offer"),
    Band("medium", 0.35, "Add to next month's outreach list"),
    Band("low", 0.00, "No action"),
), "toml", "ladder.toml")

GOOD_BLOCK = (
    "I would lower the high floor.\n\n```json\n"
    '{"bands": [{"name": "high", "min": 0.5, "action": "Contact this week and make a concrete offer"},'
    '{"name": "medium", "min": 0.35, "action": "Add to next month\'s outreach list"},'
    '{"name": "low", "min": 0.0, "action": "No action"}], "rationale": "Two verdicts said high fired too late."}\n```'
)


class StubClient:
    """Looks enough like the OpenAI client for reply() to call it."""

    def __init__(self, content: str):
        self.content = content
        self.chat = self
        self.completions = self
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_opening_message_names_every_band_and_the_customers_band(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    text = rules_chat.opening_message(V1, {"probability": 0.613})

    for band in V1.bands:
        assert band.name in text and band.action in text
    assert "61%" in text and "'high'" in text
    assert "v1" in text


def test_extract_bands_reads_the_last_block_and_validates_it():
    bands, rationale = rules_chat.extract_bands(GOOD_BLOCK, current=V1)

    assert bands is not None
    assert bands[0] == Band("high", 0.5, "Contact this week and make a concrete offer")
    assert rationale.startswith("Two verdicts")


def test_extract_bands_returns_none_when_the_block_fails_normalise():
    bad = GOOD_BLOCK.replace('"name": "low", "min": 0.0', '"name": "low", "min": 0.1')

    assert rules_chat.extract_bands(bad, current=V1) == (None, "")


def test_extract_bands_returns_none_when_band_names_change():
    renamed = GOOD_BLOCK.replace('"name": "medium"', '"name": "mid"')

    assert rules_chat.extract_bands(renamed, current=V1) == (None, "")


def test_extract_bands_returns_none_without_a_block():
    assert rules_chat.extract_bands("The rules look fine to me.") == (None, "")


def test_reply_returns_prose_without_the_block_and_the_parsed_bands():
    client = StubClient(GOOD_BLOCK)
    history = [{"role": "user", "content": "Is 60% too high a floor?"}]

    turn = rules_chat.reply(history, "system text", current=V1, client=client)

    assert turn is not None
    assert turn.text == "I would lower the high floor."
    assert turn.bands is not None and turn.bands[0].min == 0.5
    assert client.calls[0]["messages"][0] == {"role": "system", "content": "system text"}
    assert client.calls[0]["messages"][1] == history[0]
    assert "response_format" not in client.calls[0]


def test_reply_is_none_without_a_key_and_without_a_client(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert rules_chat.reply([{"role": "user", "content": "hi"}], "s") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv_webina-sep/bin/python -m pytest tests/test_rules_chat.py -v`
Expected: all FAIL / ERROR with `ModuleNotFoundError: No module named 'rules_chat'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/rules_chat.py`:

```python
"""Rung 3's console: the agent says what the rules are and advises how to change them.

Approach A from the design: the conversation is free, production is not. When
the agent suggests a change it ends its reply with one fenced JSON block holding
the bands. extract_bands() runs that block through rules_store.normalise -- the
same gate the learner passes -- and the app shows before/after with an Apply
button. Nothing here writes to Supabase, and nothing here imports Streamlit, so
every function is testable with a dict and a stub client.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from rules_store import Band, RuleSet, normalise

DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


@dataclass
class ChatTurn:
    text: str
    bands: tuple[Band, ...] | None = None
    rationale: str = ""


def llm_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def describe_bands(rules: RuleSet) -> str:
    return "\n".join(f"- {b.name}: from {b.min:.0%} -> {b.action}" for b in rules.bands)


def opening_message(rules: RuleSet, record: dict,
                    probability_key: str = "probability") -> str:
    """Built locally, no model call -- so the console is never empty."""
    probability = float(record.get(probability_key) or 0.0)
    band = rules.band_for(probability)
    return (
        f"Rules v{rules.version}, highest floor first:\n{describe_bands(rules)}\n\n"
        f"This customer is at {probability:.0%}, which lands in '{band.name}' "
        f"-> {band.action}.\n\n"
        "Ask me why, or tell me what you would rather see happen, and I will "
        "propose a change to the bands. Nothing changes until you apply it."
    )


def system_prompt(rules: RuleSet, digest_rows: list[dict], record: dict) -> str:
    return (
        "You are the console of a retention agent. The agent turns a churn "
        "probability into an action using probability bands, highest floor first:\n"
        f"{json.dumps(rules.as_rows(), ensure_ascii=False)}\n"
        f"This is rules version {rules.version}.\n\n"
        "The customer under discussion:\n"
        f"{json.dumps(record, default=str, ensure_ascii=False)}\n\n"
        "What people said about earlier recommendations, folded per band:\n"
        f"{json.dumps(digest_rows, default=str, ensure_ascii=False)}\n\n"
        "Answer in the language the user writes in. Be concrete and short. State "
        "the current rules when asked. When you recommend changing them, end your "
        "reply with exactly one fenced json block of the form "
        '{"bands": [{"name": ..., "min": ..., "action": ...}, ...], '
        '"rationale": "..."} keeping the same band names, floors between 0 and 1 '
        "with the lowest at 0. If no change is justified, say so and do not emit "
        "a block. Never invent feedback that is not listed above."
    )


def extract_bands(text: str, current: RuleSet | None = None
                  ) -> tuple[tuple[Band, ...] | None, str]:
    """The last fenced json block in the reply, validated. (None, '') otherwise."""
    blocks = BLOCK.findall(text)
    if not blocks:
        return None, ""
    try:
        payload = json.loads(blocks[-1])
        bands = normalise(payload["bands"])
    except (ValueError, KeyError, TypeError):
        return None, ""
    if current is not None and {b.name for b in bands} != {b.name for b in current.bands}:
        return None, ""
    return bands, str(payload.get("rationale", "")).strip()


def strip_block(text: str) -> str:
    return BLOCK.sub("", text).strip()


def reply(history: list[dict], system: str, current: RuleSet | None = None,
          client=None, model: str = DEFAULT_MODEL) -> ChatTurn | None:
    """One assistant turn. `history` is [{"role", "content"}, ...]. None when it cannot run.

    `client` is injectable for tests; in the app it is built from GROQ_API_KEY
    exactly the way learner.propose_llm builds its own.
    """
    if client is None:
        if not llm_available():
            return None
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=model, max_tokens=900, reasoning_effort="low",
            messages=[{"role": "system", "content": system}, *history],
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001 -- on stage a failure must not stop the demo
        return None
    if not text:
        return None
    bands, rationale = extract_bands(text, current)
    return ChatTurn(strip_block(text), bands, rationale)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv_webina-sep/bin/python -m pytest tests/test_rules_chat.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rules_chat.py tests/test_rules_chat.py
git commit -m "$(cat <<'MSG'
rules_chat: the console's logic -- opening message, prompt, reply, block parsing

Approach A: proposals pass rules_store.normalise, nothing here writes.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: rung 2 stashes the customer; rung 3 opens on it

**Files:**
- Modify: `src/app.py` — imports (after `import insights as insights_mod`), `scoring_form` (the `bands = agent_mod.Bands.load()` line and the estimator loop), `rung_recommend` (lines 365-373: `top = scored.head(25)` through `record = top.iloc[choice].to_dict()`)
- Test: render check only (`ladder.py app`); logic is covered by Task 2.

**Interfaces:**
- Consumes: `handoff.stash_scored`, `handoff.default_record`, `handoff.KEY` (Task 2); `data_mod.get_connection()`; `agent_mod.Bands.load(root=None, conn=None)`.
- Produces: rung 3's `record` dict now sometimes carries `record[spec.id_column] == "rung2:..."`; Task 5 and the existing `feedback_form` consume it unchanged.

- [ ] **Step 1: Add the import**

In `src/app.py`, after the line `import insights as insights_mod  # noqa: E402`, add:

```python
import handoff  # noqa: E402
```

- [ ] **Step 2: Stash from `scoring_form`**

In `scoring_form`, replace:

```python
    bands = agent_mod.Bands.load()
    left, right = st.columns(2)
    for slot, estimator in ((left, "logreg"), (right, "tree")):
        trained = get_model_for(df, spec, estimator)
        probability = model_mod.score_record(trained, record)
        slot.metric(
```

with:

```python
    bands = agent_mod.Bands.load(conn=data_mod.get_connection())
    left, right = st.columns(2)
    probabilities: dict[str, float] = {}
    for slot, estimator in ((left, "logreg"), (right, "tree")):
        trained = get_model_for(df, spec, estimator)
        probability = model_mod.score_record(trained, record)
        probabilities[estimator] = probability
        slot.metric(
```

and after the `st.caption("Two families, one record. ...")` block that follows the loop, add:

```python
    typed = {column: record[column] for column in drivers + numbers}
    handoff.stash_scored(st.session_state, record, probabilities, typed)
    st.caption("This customer is now the one rung 3 opens on.")
```

- [ ] **Step 3: Rung 3 opens on the stashed customer**

In `rung_recommend`, replace:

```python
    top = scored.head(25)
    labels = [
        f"{row[spec.id_column]} · risk {row['probability']:.0%}"
        for _, row in top.iterrows()
    ]
    choice = st.selectbox(
        "Pick a record", range(len(labels)), format_func=lambda i: labels[i]
    )
    record = top.iloc[choice].to_dict()
```

with:

```python
    top = scored.head(25)
    labels = [
        f"{row[spec.id_column]} · risk {row['probability']:.0%}"
        for _, row in top.iterrows()
    ]
    default = handoff.default_record(st.session_state, spec.estimator, spec.id_column)
    if default is not None:
        stash = st.session_state[handoff.KEY]
        st.markdown("**The customer you scored on rung 2**")
        typed = stash["typed"]
        st.dataframe(
            pd.DataFrame({"field": list(typed), "value": [typed[c] for c in typed]}),
            width="stretch", hide_index=True,
        )
        shown = stash["probability"]
        st.caption(
            " · ".join(f"{model_mod.ESTIMATORS.get(k, k)} {v:.1%}" for k, v in shown.items())
            + f" · the recommender reads {model_mod.ESTIMATORS.get(spec.estimator, spec.estimator)}."
        )
        with st.expander("Or pick one of the 25 highest-risk customers in the table"):
            st.caption(
                "Walking real customers one at a time is how you find out whether "
                "the bands are right. It is the way to formulate the rules the agent "
                "will act on, not only to read them."
            )
            use_table = st.checkbox("Use a customer from the table instead", value=False)
            choice = st.selectbox(
                "Pick a record", range(len(labels)),
                format_func=lambda i: labels[i], disabled=not use_table,
            )
        record = top.iloc[choice].to_dict() if use_table else default
    else:
        st.caption(
            "Nothing scored on rung 2 yet -- score a customer there and it opens "
            "here first. Until then, pick one of the 25 highest-risk customers."
        )
        choice = st.selectbox(
            "Pick a record", range(len(labels)), format_func=lambda i: labels[i]
        )
        record = top.iloc[choice].to_dict()
```

- [ ] **Step 4: Render check and full suite**

Run: `./venv_webina-sep/bin/python ladder.py app 2>&1 | tail -1`
Expected: `app: PASS — 4 rungs · 0 exceptions · N rows to Supabase`

Run: `./venv_webina-sep/bin/python -m pytest tests/ -q`
Expected: all PASS (skips allowed).

- [ ] **Step 5: See it once**

Run: `./venv_webina-sep/bin/python -m streamlit run src/app.py --server.port 8502 --server.headless true` (background), open http://localhost:8502, score a customer on rung 2 with tenure 1 and monthly charges 95, open rung 3.
Expected: rung 3 opens with "The customer you scored on rung 2", the six typed fields, two probabilities, and the recommendation for that customer; the expander below offers the table. Stop the server afterwards (`kill $(lsof -t -nP -iTCP:8502 -sTCP:LISTEN)`).

- [ ] **Step 6: Commit**

```bash
git add src/app.py
git commit -m "$(cat <<'MSG'
app: rung 3 opens on the customer scored on rung 2; the table is the second choice

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 5: the console on rung 3

**Files:**
- Modify: `src/app.py` — imports (add `import rules_chat`), new function `rules_chat_panel` placed directly above `def feedback_form`, one call inserted in `rung_recommend` before `feedback_form(conn, record, rules, rules_set, spec)`.
- Test: render check (`ladder.py app`); logic covered by Task 3.

**Interfaces:**
- Consumes: `rules_chat.*` (Task 3); `learner_mod.digest(rules_set, feedback) -> dict[str, BandDigest]` with `.as_row()`; `learner_mod._diff(before: RuleSet, after: tuple[Band, ...]) -> list[str]`; `rules_store.publish(conn, bands, source, rationale, evidence) -> RuleSet`; `rules_set.as_rows()`.
- Produces: `rules_chat_panel(conn, rules_set, feedback: list[dict], record: dict) -> None`. Session keys: `rules_chat_key`, `rules_chat` (list of `{"role","content"}`), `chat_proposal` (`{"bands","rationale","version"}`).

- [ ] **Step 1: Add the import**

After `import learner as learner_mod  # noqa: E402` add:

```python
import rules_chat  # noqa: E402
```

- [ ] **Step 2: Write the panel**

Insert directly above `def feedback_form(`:

```python
def rules_chat_panel(conn, rules_set, feedback: list[dict], record: dict) -> None:
    """Approach A: the agent talks; a button applies.

    The conversation is keyed to the rules version. When the learner or this
    panel publishes a new version the page reruns, the key changes, and the
    conversation starts over on the new rules -- so the console can never be
    arguing about bands that are no longer live.
    """
    st.markdown("##### Talk to the rules")
    key = f"rules_chat_v{rules_set.version}"
    if st.session_state.get("rules_chat_key") != key:
        restarted = "rules_chat_key" in st.session_state
        st.session_state["rules_chat_key"] = key
        st.session_state["rules_chat"] = [
            {"role": "assistant", "content": rules_chat.opening_message(rules_set, record)}
        ]
        st.session_state.pop("chat_proposal", None)
        if restarted:
            st.caption(f"The rules moved to v{rules_set.version}; the conversation starts over on them.")

    for turn in st.session_state["rules_chat"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    if not rules_chat.llm_available():
        st.info(
            "GROQ_API_KEY is not set, so the console can state the rules but not "
            "discuss them. That is the honest state of the demo, not a failure."
        )
        return

    prompt = st.chat_input("Ask about the rules, or say what should change")
    if prompt:
        st.session_state["rules_chat"].append({"role": "user", "content": prompt})
        digest_rows = (
            [d.as_row() for d in learner_mod.digest(rules_set, feedback).values()]
            if feedback else []
        )
        system = rules_chat.system_prompt(rules_set, digest_rows, record)
        with st.spinner("Thinking..."):
            turn = rules_chat.reply(st.session_state["rules_chat"], system, current=rules_set)
        if turn is None:
            st.session_state["rules_chat"].append(
                {"role": "assistant", "content": "_No usable answer came back. The rules stand._"}
            )
        else:
            st.session_state["rules_chat"].append({"role": "assistant", "content": turn.text})
            if turn.bands is not None:
                st.session_state["chat_proposal"] = {
                    "bands": turn.bands, "rationale": turn.rationale,
                    "version": rules_set.version,
                }
        st.rerun()

    proposal = st.session_state.get("chat_proposal")
    if not proposal or proposal["version"] != rules_set.version:
        return
    changes = learner_mod._diff(rules_set, proposal["bands"])
    st.markdown(f"**The console proposes v{rules_set.version + 1}**")
    if proposal["rationale"]:
        st.write(proposal["rationale"])
    if not changes:
        st.info("Identical to the current bands -- nothing to apply.")
        return
    c1, c2 = st.columns(2)
    c1.markdown(f"v{rules_set.version} (now)")
    c1.dataframe(pd.DataFrame(rules_set.as_rows()), width="stretch", hide_index=True)
    c2.markdown(f"v{rules_set.version + 1} (proposed)")
    c2.dataframe(
        pd.DataFrame([b.__dict__ for b in proposal["bands"]]),
        width="stretch", hide_index=True,
    )
    for line in changes:
        st.markdown(f"- {line}")
    if conn is None:
        st.warning("No Supabase connection -- the proposal can be seen but not applied.")
        return
    if st.button(f"Apply as v{rules_set.version + 1}", type="primary", key="chat_apply"):
        published = rules_store.publish(
            conn, proposal["bands"], "chat:groq", proposal["rationale"],
            {"chat": st.session_state["rules_chat"][-6:]},
        )
        st.session_state.pop("chat_proposal", None)
        st.success(
            f"Rules v{published.version} is active. The recommendation above now "
            "comes from it."
        )
        st.rerun()
```

- [ ] **Step 3: Place it**

In `rung_recommend`, immediately before the line `    feedback_form(conn, record, rules, rules_set, spec)`, insert:

```python
    st.divider()
    rules_chat_panel(conn, rules_set, feedback, record)
    st.divider()
```

- [ ] **Step 4: Render check and full suite**

Run: `./venv_webina-sep/bin/python ladder.py app 2>&1 | tail -1`
Expected: `app: PASS — 4 rungs · 0 exceptions · ...` (AppTest has no key in env unless secrets.toml is read — with secrets present, the panel renders its `chat_input`; either way 0 exceptions.)

Run: `./venv_webina-sep/bin/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Try one real exchange**

Start the app as in Task 4 step 5, score a customer on rung 2, open rung 3, type: `Why is this customer 'high'? Would 50% be a better floor?`
Expected: the opening message lists v1's three bands; the reply is prose; if it proposes, a before/after card with **Apply as v2** appears. Do NOT click Apply on the live database unless Zac asks — the live rules are what the webinar audience sees. Stop the server.

- [ ] **Step 6: Commit**

```bash
git add src/app.py
git commit -m "$(cat <<'MSG'
app: the console on rung 3 -- states the rules, proposes changes, applies on a click

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 6: rung 4 tells the loop's story

**Files:**
- Modify: `src/app.py`, `rung_production` — move the "How much of the loop has closed" block below the predictions table, then append two sections.
- Test: render check (`ladder.py app`).

**Interfaces:**
- Consumes: `rules_store.history(conn, limit=10)` with `bands` (Task 1); `rules_store.from_row(row) -> RuleSet`; `learner_mod._diff`; `data_mod.read_feedback(conn, limit=10) -> list[dict]` (oldest first); `data_mod.loop_counts(conn)`.
- Produces: nothing new.

- [ ] **Step 1: Reorder**

In `rung_production`, cut the block that starts at `    st.markdown("**How much of the loop has closed**")` and ends at the closing `)` of its `st.caption(...)`, and paste it after the `else:` branch that renders the predictions table (i.e. after the caption that begins `"Local mode: this is data/predictions.csv`). Order is now: metrics, write button, predictions table, loop counts.

- [ ] **Step 2: Append the two sections**

At the end of `rung_production` (after the loop-counts caption you just moved), add:

```python
    st.markdown("**Rules history -- what the learner and the console have published**")
    versions = rules_store.history(conn)
    if not versions:
        st.info("No Supabase connection, so no history: the rules are ladder.toml's v1.")
    else:
        shown = pd.DataFrame(versions)
        st.dataframe(
            shown[[c for c in ("version", "source", "rationale", "active", "created_at")
                   if c in shown.columns]],
            width="stretch", hide_index=True,
        )
        if len(versions) >= 2 and versions[0].get("bands") is not None:
            newest = rules_store.from_row(versions[0])
            previous = rules_store.from_row(versions[1])
            for line in learner_mod._diff(previous, newest.bands):
                st.markdown(f"- v{previous.version} -> v{newest.version}: {line}")
        st.caption(
            "Every row is a decision someone can audit: who changed the rules, from "
            "what evidence, and when. The model was never touched -- ten verdicts "
            "move a rule, a retrain needs thousands of outcomes."
        )

    st.markdown("**Latest verdicts from people**")
    verdicts = data_mod.read_feedback(conn, limit=10)
    if not verdicts:
        st.info("No feedback yet. Rung 3 collects it under every recommendation.")
    else:
        frame = pd.DataFrame(verdicts[::-1])
        columns = ("created_at", "record_id", "recommended_action", "verdict",
                   "actual_outcome", "better_action", "rules_version")
        st.dataframe(
            frame[[c for c in columns if c in frame.columns]],
            width="stretch", hide_index=True,
        )
        st.caption(
            "This is the input the learner reads: the recommendation, what a person "
            "said about it, and what actually happened."
        )
```

- [ ] **Step 3: Render check and full suite**

Run: `./venv_webina-sep/bin/python ladder.py app 2>&1 | tail -1`
Expected: `app: PASS — 4 rungs · 0 exceptions · ...`

Run: `./venv_webina-sep/bin/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/app.py
git commit -m "$(cat <<'MSG'
app: rung 4 shows the loop -- rules history with diffs, and the latest verdicts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 7: push, live proof, and the documents

**Files:**
- Modify: `STATUS.md` (Hebrew, the coordination table rows from "בתכנון" to "בוצע", plus the live-proof line), `WorkLog.md` (English, one dated bullet per task above).

**Interfaces:** none.

- [ ] **Step 1: Push and let Streamlit redeploy**

```bash
git push origin main
```

Wait ~90 seconds. Then:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://webina-sep.streamlit.app
```
Expected: `200` (a `303` means still waking; retry).

- [ ] **Step 2: Prove it live, with Playwright MCP**

Open https://webina-sep.streamlit.app, tab "2 · Predict", set tenure 1 and monthly charges 95, click "Score this customer"; tab "3 · Recommend".
Expected: "The customer you scored on rung 2" is the first thing on rung 3, the console shows "Rules v1 ..." as its opening message, the feedback form and learner panel follow. Tab "4 · Production": "Rules history" shows v1 (seed, active) and "Latest verdicts" shows the info line or rows. Take a screenshot into the scratchpad for the record.

- [ ] **Step 3: Documents**

Append to `WorkLog.md` under today's date, one bullet per task (`app:`/`model:`/`docs:` tags per `.claude/worklog-config.md`), each naming the test that guarded it. In `STATUS.md`, flip the three rows of the coordination table to `בוצע` and add one line with the live URL and the time of the live proof. Hebrew lines must start with a Hebrew word.

- [ ] **Step 4: Commit and push**

```bash
git add STATUS.md WorkLog.md
git commit -m "$(cat <<'MSG'
docs: rung 3 console and rung 4 loop view live, proven on the deployed app

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
git push origin main
```

---

## Self-review

- Spec §1 (default record): Tasks 2 and 4. §2 approach A (console, validate, apply on click, no key, no conn, version reset): Tasks 3 and 5. §3 (rung 4 order, history with diff, verdicts, captions): Tasks 1 and 6. §4 tests: `test_handoff.py`, `test_rules_chat.py`, `test_rules_store.py::test_history_rows_carry_the_bands`, render checks. §5 files: all covered; `handoff.py` is the one stated deviation.
- Placeholders: none; every code step carries the code.
- Names: `handoff.stash_scored/default_record/KEY`, `rules_chat.opening_message/system_prompt/extract_bands/strip_block/reply/llm_available/ChatTurn`, `rules_chat_panel`, `rules_store.history/from_row/publish`, `learner_mod.digest/_diff` — used identically across Tasks 2-6.
