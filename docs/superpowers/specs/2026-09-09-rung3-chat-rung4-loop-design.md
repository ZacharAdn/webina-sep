# Rung 3 conversation, rung 4 loop view — design

Status: APPROVED 2026-09-09 22:3x — approach A for section 2 (proposes; a button applies).
Owner of `src/app.py`, `src/rules_chat.py`, `STATUS.md`: this agent. The
second agent owns `rules_store.py`, `learner.py`, the migration and their
tests, and follows this work through STATUS.md.

## 0. What exists, what is asked

Built and live (second agent, committed): `predictions.outcome`, a
`feedback` table, a versioned `rules` table with v1 seeded from ladder.toml;
`rules_store.load_active/publish/history`; `learner.digest/propose`;
on rung 3 a feedback form under the recommendation and a learner panel that
proposes and publishes the next rules version; on rung 4 three loop counts.

Asked (Zac, 22:1x):
1. Rung 3 opens on the customer entered on rung 2's "Score a customer who
   is not in the table" form. Only after that, a picker for other customers,
   framed as a way to formulate the rules the agent acts on.
2. A chat on rung 3 in which the agent states the current rules live and
   advises how to flip or change them.
3. Rung 4 must be legible: feedback on what actually happened, saved to
   Supabase, and a second agent that teaches the first to improve the rules.
   (Item 3 of the completion plan; mostly built — rung 4 has to *show* it.)

## 1. Piece one — the rung 2 customer becomes rung 3's default

`scoring_form` (mine, rung 2) already builds the full 19-column record on
submit. It will also stash it:

    st.session_state["scored"] = {
        "record": record,                       # 19 features + derived total
        "probability": {"logreg": p1, "tree": p2},
        "at": iso timestamp,
    }

Rung 3 gets a record source block at the top, before the recommendation:

- Default, when `scored` exists: "The customer you scored on rung 2" — the
  six fields the user typed, the two probabilities, and which band the
  configured estimator's probability lands in. The record used downstream
  is `scored.record` with `probability = probability[spec.estimator]`
  (logreg — the recommender's input) and a synthetic id
  `record[spec.id_column] = "rung2:" + 8-char sha1 of the six typed values`.
  The second agent's `feedback_form` and `latest_prediction_for` take a
  string id and already handle "no logged prediction" with a caption, so
  nothing there changes.
- Below it, collapsed: "Or pick one of the 25 highest-risk customers in the
  table", with the caption: *"Walking real customers one at a time is how
  you find out whether the bands are right. It is the way to formulate the
  rules the agent will act on, not only to read them."* Picking one
  overrides the default for the rest of the page.
- When nothing was scored on rung 2, the picker is open and is the default,
  with a one-line pointer back to rung 2.

## 2. Piece two — the rules chat (approach A chosen; B and C kept for the record)

**A. Advisory chat that proposes; a button applies. (Recommended.)**
The agent answers in prose, grounded in rules v{n}, the feedback digest
and the selected record. When it recommends a change it appends a fenced
JSON block `{"bands": [...], "rationale": "..."}`. The app extracts it,
validates with `rules_store.normalise` — the same gate the learner passes —
and shows a before/after card with **Apply as v{n+1}**, which calls
`rules_store.publish(conn, bands, "chat:groq", rationale, evidence)`.
Same validator, same publish path, same safety as the learner panel; the
conversation is free but production only changes on a click.

**B. Chat that publishes on "apply".** Faster on stage, one message fewer.
A mis-parsed reply rewrites production rules with no card in between, and
it duplicates the learner's job on thinner evidence. Not recommended.

**C. Chat as narration over the learner.** The chat only triggers
`learner.propose()` and explains the result. Cheapest, but it cannot
"advise me how to change them" — that is the ask. Not recommended.

Under A:

- New module `src/rules_chat.py`, no Streamlit import:
  - `opening_message(rules_set, record, recommendation) -> str` — built
    locally, no LLM: the bands as three lines, which band this customer is
    in and why, and an invitation. Works with no key, so the panel is never
    empty.
  - `system_prompt(rules_set, digest, record, recommendation) -> str`.
  - `reply(history, system) -> ChatTurn(text, bands | None, rationale)` —
    OpenAI SDK at GROQ_BASE_URL, `reasoning_effort="low"`, prose mode (no
    json_object: the reply is text with an optional block). Mirrors the
    user's language: Hebrew in, Hebrew out.
  - `extract_bands(text) -> (bands | None, rationale)` — regex for the
    fenced block, `normalise()` on it, None on anything invalid.
- UI in `app.py`, `rules_chat_panel(conn, rules_set, feedback, record,
  recommendation)`: `st.chat_message` history in
  `st.session_state["rules_chat"]`, keyed by rules version; a version change
  resets it with a note. No key: opening message shown, input disabled,
  the same honest line the second-opinion panel uses. LLM failure: warning,
  rules stand. No connection: the card shows, Apply is disabled with the
  learner panel's wording.
- Placement on rung 3: recommendation → chat panel → feedback form →
  learner panel → top-25 table. The chat sits where the user's eye is after
  reading the recommendation.

## 3. Piece three — rung 4 tells the loop's story

Keep the three counts. Reorder and add, top to bottom, in the order the
completion script walks it (plan item 5):

1. Intro and the three metrics: data source, model version, rules version.
2. "Score and write" button and the predictions table (audit trail).
3. **How much of the loop has closed** — the three loop counts (theirs).
4. **Rules history** — `rules_store.history()` as a table: version,
   source, rationale, active, created_at; and when version > 1, a
   before/after of the last two band sets. `history()` selects only
   metadata today; add `bands` to its select (additive, one line, noted
   for the second agent).
5. **Latest verdicts** — last 10 feedback rows: record, verdict, what
   happened, better action, rules version.

Each block gets one caption saying what a stranger can now read that they
could not before.

## 4. Tests, first

`tests/test_rules_chat.py`, offline:
- `extract_bands` pulls a valid block out of surrounding prose and returns
  None for a block that fails `normalise`.
- `opening_message` names all bands of v{n} and the band the record lands
  in, with no key in the environment.
- `history()` rows carry `bands` (needs a connection; skipped without).
Render check stays `python ladder.py app` (AppTest, all four tabs, 0
exceptions). The scored-record handoff is exercised by a test that calls
the stash-and-default helper with a fake session dict.

## 5. Files

    src/rules_chat.py        new
    src/app.py               scoring_form stash · rung 3 source block +
                             chat panel · rung 4 sections 4-5
    src/rules_store.py       history(): + bands in select
    tests/test_rules_chat.py new
    STATUS.md, WorkLog.md    coordination and log

Out of scope: touching the model, the learner's policy, or the migration.
