"""The four rungs, one app.

Rung 1  describe   -- the dashboard, reading records out of Supabase
Rung 2  predict    -- a model, its baseline, and the leakage story
Rung 3  recommend  -- versioned bands, gpt-oss-120b as the second opinion, and
                      the loop: a verdict on each recommendation, and a second
                      agent that rewrites the bands from those verdicts
Rung 4  production -- write the predictions back to a table, deploy, and count
                      how much of the loop has closed

Nothing here is dataset-specific: every number and every column name in the
captions is computed from the table that was actually loaded. The only file
that changes between datasets is ladder.toml.

Run locally:  streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent as agent_mod  # noqa: E402
import data as data_mod  # noqa: E402
import handoff  # noqa: E402
import insights as insights_mod  # noqa: E402
import learner as learner_mod  # noqa: E402
import rules_chat  # noqa: E402
import model as model_mod  # noqa: E402
import rules_store  # noqa: E402
from rungs import RUNGS, SUBHEADS  # noqa: E402

st.set_page_config(page_title="From data to production", page_icon="📶", layout="wide")

# Secrets become environment variables, so the same code reads a key whether it
# was pasted at the rung-3 gate, written into .streamlit/secrets.toml, or typed
# into Streamlit Cloud's secrets box.
try:
    for _name, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_name, _value)
except Exception:  # noqa: BLE001 -- no secrets file is the normal local case
    pass

ACCENT = "#2563eb"
CONTRA = "#dc2626"


@st.cache_resource(show_spinner="Training the model...")
def get_model_for(df: pd.DataFrame, _spec: model_mod.Spec,
                  estimator: str) -> model_mod.TrainedModel:
    """One trained pipeline per estimator, keyed on the estimator's name.

    Streamlit's underscore rule leaves `_spec` out of the cache key. Asking
    this function for two families with the spec alone would therefore have
    handed back the same pipeline twice, with no error and no warning. The
    plain string is what makes the two calls distinct.
    """
    return model_mod.train_model(df, replace(_spec, estimator=estimator))


def get_model(df: pd.DataFrame, spec: model_mod.Spec) -> model_mod.TrainedModel:
    """The family ladder.toml chose. Everything else on the page uses this."""
    return get_model_for(df, spec, spec.estimator)


@st.cache_data(show_spinner="Running the leakage comparison...")
def get_leakage(df: pd.DataFrame, _spec: model_mod.Spec):
    honest, leaky = model_mod.leakage_demo(df, _spec)
    return honest.as_row(), leaky.as_row()


def main() -> None:
    spec = model_mod.Spec.from_toml()
    load = data_mod.load_records()
    df = load.df

    st.title("From a table to something that runs")
    st.caption(
        f"{'Live' if load.source == 'supabase' else 'Local mode'}: {load.detail}"
    )

    tab1, tab2, tab3, tab4 = st.tabs(list(RUNGS))
    with tab1:
        rung_describe(df, spec)
    with tab2:
        rung_predict(df, spec)
    with tab3:
        rung_recommend(df, spec)
    with tab4:
        rung_production(df, spec, load)


# --------------------------------------------------------------------------
# Rung 1
# --------------------------------------------------------------------------
def rung_describe(df: pd.DataFrame, spec: model_mod.Spec) -> None:
    st.subheader(SUBHEADS[0])

    positive_rate = (df[spec.target].astype(str).str.strip() == spec.positive_label).mean()
    missing = insights_mod.missing_report(df, spec.id_column, spec.target)
    numeric, _ = insights_mod.split_columns(
        df, spec.id_column, spec.target, spec.categorical_int_max_unique
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric(f"{spec.positive_label} rate", f"{positive_rate:.1%}")
    c3.metric("Columns with NULLs", len(missing))
    if numeric:
        c4.metric(f"Median {numeric[0]}", f"{df[numeric[0]].median():,.0f}")

    st.markdown("**One: the split is uneven, and that decides the whole evening.**")
    counts = df[spec.target].value_counts().reset_index()
    counts.columns = [spec.target, "count"]
    st.plotly_chart(
        px.bar(
            counts, x=spec.target, y="count", color=spec.target,
            color_discrete_sequence=[ACCENT, CONTRA],
            title=f"{spec.target}: {positive_rate:.1%} are '{spec.positive_label}'",
        ),
        width="stretch",
    )
    st.caption(
        f"Because {1 - positive_rate:.0%} are not '{spec.positive_label}', a model "
        f"that always says no is {1 - positive_rate:.0%} accurate and worth nothing. "
        "That is rung 2's problem, and it starts here."
    )

    st.markdown("**Two: what is missing, and what the blanks were hiding.**")
    if not missing:
        st.write("No column has a NULL. That is rarer than it sounds -- say so out loud.")
    else:
        top = missing[0]
        explanation = insights_mod.explains_missing(df, top["column"])
        sentence = (
            f"{top['nulls']} records have no {top['column']} "
            f"({top['share']:.1%} of the table)."
        )
        if explanation:
            sentence += (
                f" Every one of them has {explanation['column']} = "
                f"{explanation['value']} -- the blank is a fact about those rows, "
                "not a data error. The mistake is to coerce it to zero."
            )
        st.write(sentence)
        st.dataframe(
            df[df[top["column"]].isna()].head(25),
            width="stretch", hide_index=True,
        )
        if len(missing) > 1:
            st.caption(
                "Other columns with NULLs: "
                + ", ".join(f"{m['column']} ({m['nulls']})" for m in missing[1:])
            )

    st.markdown("**Three: who goes with whom.**")
    ranked = insights_mod.discriminative_categoricals(
        df, spec.target, spec.positive_label, k=3
    )
    if not ranked:
        st.write("No categorical column separates the two groups. That is a finding.")
        return
    for row in ranked:
        rates = (
            pd.DataFrame(
                {"value": list(row["rates"]), "rate": list(row["rates"].values())}
            )
            .sort_values("rate", ascending=False)
        )
        st.plotly_chart(
            px.bar(
                rates, x="value", y="rate",
                color_discrete_sequence=[ACCENT],
                title=f"'{spec.positive_label}' rate by {row['column']}",
            ),
            width="stretch",
        )
    first = ranked[0]
    hi = max(first["rates"], key=first["rates"].get)
    lo = min(first["rates"], key=first["rates"].get)
    st.caption(
        f"{first['column']} = {hi} is at {first['rates'][hi]:.0%}; {lo} is at "
        f"{first['rates'][lo]:.0%} -- a spread of {first['spread']:.0%}. That gap is "
        "why rung 3's bands are worth setting at all."
    )


# --------------------------------------------------------------------------
# Rung 2
# --------------------------------------------------------------------------
def rung_predict(df: pd.DataFrame, spec: model_mod.Spec) -> None:
    st.subheader(SUBHEADS[1])

    trained = get_model(df, spec)
    st.dataframe(
        pd.DataFrame([trained.baseline.as_row(), trained.metrics.as_row()]),
        width="stretch", hide_index=True,
    )
    st.caption(
        f"The baseline earns {trained.baseline.accuracy:.0%} accuracy without catching "
        f"a single '{spec.positive_label}'. The model finds {trained.metrics.recall:.0%} "
        f"of them at ROC-AUC {trained.metrics.roc_auc:.2f}."
    )
    with st.expander("Why the number can lie -- the leakage demo"):
        honest, leaky = get_leakage(df, spec)
        st.dataframe(pd.DataFrame([leaky, honest]), width="stretch", hide_index=True)
        st.caption(
            f"Balance the classes before splitting and the same model reports "
            f"{leaky['recall']:.0%} recall; split first and it is {honest['recall']:.0%}. "
            "One line of code apart."
        )

    st.divider()
    scoring_form(df, spec)


FORM_NUMERIC = ("tenure", "monthly_charges")
FORM_CATEGORICAL_MAX = 4


def scoring_form(df: pd.DataFrame, spec: model_mod.Spec) -> None:
    """Score a customer who is not in the table, on both model families.

    Six fields, and which six is a question asked of the data rather than a list
    written here: the strongest categorical separators come from the same
    function rung 1 uses. The other columns are filled from the table and shown,
    because a default nobody can see is a number nobody can argue with.
    """
    numeric, categorical = model_mod.split_columns(df, spec)
    defaults = model_mod.feature_defaults(df, spec)
    drivers = [
        row["column"]
        for row in insights_mod.discriminative_categoricals(
            df, spec.target, spec.positive_label, k=FORM_CATEGORICAL_MAX
        )
        if row["column"] in categorical
    ]
    numbers = [column for column in FORM_NUMERIC if column in numeric]

    st.markdown("**Score a customer who is not in the table.**")
    record = dict(defaults)
    with st.form("score_one"):
        slots = st.columns(2)
        for position, column in enumerate(drivers + numbers):
            slot = slots[position % 2]
            label = column.replace("_", " ")
            if column in drivers:
                values = sorted(df[column].dropna().unique().tolist(), key=str)
                start = values.index(defaults[column]) if defaults[column] in values else 0
                record[column] = slot.selectbox(label, values, index=start)
            else:
                record[column] = slot.number_input(
                    label,
                    min_value=float(df[column].min()),
                    max_value=float(df[column].max()),
                    value=float(defaults[column]),
                )
        submitted = st.form_submit_button("Score this customer", type="primary")

    if not submitted:
        st.caption(
            f"{len(drivers) + len(numbers)} fields are yours; the remaining "
            f"{len(numeric) + len(categorical) - len(drivers) - len(numbers)} come "
            "from the table and are listed with the result."
        )
        return

    derived = []
    if "total_charges" in numeric and {"tenure", "monthly_charges"} <= set(record):
        record["total_charges"] = float(record["tenure"]) * float(
            record["monthly_charges"]
        )
        derived.append("total_charges")

    bands = agent_mod.Bands.load(conn=data_mod.get_connection())
    left, right = st.columns(2)
    probabilities: dict[str, float] = {}
    for slot, estimator in ((left, "logreg"), (right, "tree")):
        trained = get_model_for(df, spec, estimator)
        probability = model_mod.score_record(trained, record)
        probabilities[estimator] = probability
        slot.metric(
            model_mod.ESTIMATORS.get(estimator, estimator),
            f"{probability:.1%}",
            help=f"P({spec.positive_label}) for the record above",
        )
        slot.caption(bands.recommend({"probability": probability}).action)

    typed = {column: record[column] for column in drivers + numbers}
    handoff.stash_scored(st.session_state, record, probabilities, typed)
    st.session_state.pop("table_record", None)
    st.caption("This customer is now the one rung 3 opens on.")

    filled = {
        column: value
        for column, value in record.items()
        if column not in drivers + numbers
    }
    with st.expander(f"{len(filled)} fields filled in from the table"):
        if derived:
            st.caption(
                "total_charges is derived as tenure x monthly charges, not taken "
                "from the table: a median total against a one-month tenure would "
                "be a contradiction to feed the model."
            )
        st.dataframe(
            pd.DataFrame(
                {"column": list(filled), "value": [filled[c] for c in filled]}
            ),
            width="stretch",
            hide_index=True,
        )


# --------------------------------------------------------------------------
# Rung 3
# --------------------------------------------------------------------------
def rung_recommend(df: pd.DataFrame, spec: model_mod.Spec) -> None:
    st.subheader(SUBHEADS[2])

    conn = data_mod.get_connection()
    trained = get_model(df, spec)
    ranked = insights_mod.discriminative_categoricals(
        df, spec.target, spec.positive_label, k=2
    )
    extra = [row["column"] for row in ranked]
    scored = model_mod.score_records(trained, df, spec, extra)
    rules_set = rules_store.load_active(conn)
    bands = agent_mod.Bands.from_rules(rules_set)
    feedback = data_mod.read_feedback(conn)
    st.caption(f"A probability is not a decision. Rules v{rules_set.version} turn it into one.")

    top = scored.head(25)
    labels = [
        f"{row[spec.id_column]} · risk {row['probability']:.0%}"
        for _, row in top.iterrows()
    ]
    default = handoff.default_record(st.session_state, spec.estimator, spec.id_column)
    override = st.session_state.get("table_record")

    if override is not None:
        record = override
        who = f"{record[spec.id_column]}, from the table"
    elif default is not None:
        record = default
        typed = st.session_state[handoff.KEY]["typed"]
        who = " · ".join(
            value if isinstance(value, str) and value not in ("Yes", "No")
            else f"{field.replace('_', ' ')} {value:g}" if not isinstance(value, str)
            else f"{field.replace('_', ' ')} {value}"
            for field, value in typed.items()
        )
    else:
        st.caption("Nothing scored on rung 2 yet -- score a customer there and it opens here first.")
        choice = st.selectbox(
            "Pick a customer from the table", range(len(labels)),
            format_func=lambda i: labels[i],
        )
        record = top.iloc[choice].to_dict()
        who = f"{record[spec.id_column]}, from the table"

    rules = bands.recommend(record)
    st.markdown(f"**{who}**")
    st.success(rules.action)
    st.caption(rules.reason)

    st.divider()
    rules_chat_panel(conn, rules_set, feedback, record, str(record[spec.id_column]))
    feedback_form(conn, record, rules, rules_set, spec)

    if default is not None:
        with st.expander("Try another customer from the table"):
            st.caption(
                "Walking real customers one at a time is how you find out whether "
                "the bands are right. It is the way to formulate the rules the agent "
                "will act on, not only to read them."
            )
            choice = st.selectbox(
                "Pick a customer", range(len(labels)),
                format_func=lambda i: labels[i], key="table_pick",
            )
            c1, c2 = st.columns(2)
            if c1.button("Use this customer", key="use_table"):
                st.session_state["table_record"] = top.iloc[choice].to_dict()
                st.rerun()
            if override is not None and c2.button(
                "Back to the customer from rung 2", key="back_rung2"
            ):
                st.session_state.pop("table_record", None)
                st.rerun()


# --------------------------------------------------------------------------
# The loop -- a verdict on the recommendation, and the agent that learns from it
# --------------------------------------------------------------------------
WRITE_BATCH = 25
OUTCOMES = {"unknown": "don't know yet", "stayed": "stayed", "left": "left"}


def rules_chat_panel(conn, rules_set, feedback: list[dict], record: dict,
                     record_id: str) -> None:
    """Approach A: the agent talks; a button applies.

    The conversation is keyed to the rules version. When the learner or this
    panel publishes a new version the page reruns, the key changes, and the
    conversation starts over on the new rules -- so the console can never be
    arguing about bands that are no longer live.
    """
    st.markdown("##### Talk to the rules")
    key = rules_chat.chat_key(rules_set, record_id)
    if st.session_state.get("rules_chat_key") != key:
        previous = st.session_state.get("rules_chat_key", "")
        restarted = bool(previous) and not previous.startswith(f"rules_chat_v{rules_set.version}_")
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
        if turn is None or turn.error:
            reason = turn.error if turn is not None else "GROQ_API_KEY is not set"
            st.session_state["rules_chat"].append(
                {"role": "assistant",
                 "content": f"_The model could not answer. {reason}. The rules stand._"}
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


def feedback_form(conn, record: dict, rules, rules_set, spec) -> None:
    """One verdict per recommendation. This is the input the whole loop runs on."""
    st.markdown("##### Was this the right call? Tell the system.")
    record_id = str(record[spec.id_column])
    logged = data_mod.latest_prediction_for(conn, record_id)
    with st.form(f"feedback_{record_id}", clear_on_submit=True):
        c1, c2 = st.columns(2)
        verdict = c1.radio("The recommendation was", ["right", "wrong"], horizontal=True)
        outcome = c2.radio(
            "What actually happened", list(OUTCOMES), horizontal=True,
            format_func=OUTCOMES.get,
        )
        with st.expander("I have a better action"):
            better = st.text_input(
                "A better action, in one line (leave empty if the call was right)"
            )
            note = st.text_input("Why? (optional)")
        sent = st.form_submit_button("Send feedback", type="primary")
    if sent:
        row = {
            "prediction_id": (logged or {}).get("id"),
            "record_id": record_id,
            "probability": round(float(record.get("probability") or 0.0), 4),
            "recommended_action": rules.action,
            "verdict": verdict,
            "actual_outcome": outcome,
            "better_action": better.strip() or None,
            "note": note.strip() or None,
            "rules_version": rules_set.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        ok, message = data_mod.write_feedback(conn, row)
        (st.success if ok else st.error)(message)
        if ok and logged:
            st.caption(
                f"Attached to prediction #{logged['id']} from "
                f"{str(logged.get('created_at', ''))[:16]} -- its outcome column is filled now."
            )
    elif logged is None:
        st.caption(
            "No logged decision for this record yet -- the feedback is still saved, "
            "it just has no prediction row to attach an outcome to. Rung 4 writes those."
        )


def learner_panel(conn, rules_set, feedback: list[dict]) -> None:
    """Agent 2: reads the verdicts, proposes a revision, publishes it on request."""
    if not feedback:
        st.caption("No verdicts yet. Send one on rung 3 and the learner has something to read.")
        return

    prefer_llm = learner_mod.llm_available()
    writer = "gpt-oss-120b" if prefer_llm else "the rule-based learner"
    if st.button("Let the learner revise the rules", type="primary"):
        with st.spinner("Reading the verdicts..."):
            proposal = learner_mod.propose(rules_set, feedback, prefer_llm=prefer_llm)
        st.session_state["proposal"] = proposal
    st.caption(f"{len(feedback)} verdicts on rules v{rules_set.version}; {writer} writes the revision.")
    with st.expander("What the learner read, folded per band"):
        digested = learner_mod.digest(rules_set, feedback)
        st.dataframe(
            pd.DataFrame([d.as_row() for d in digested.values()]),
            width="stretch", hide_index=True,
        )

    proposal = st.session_state.get("proposal")
    if proposal is None or proposal.before.version != rules_set.version:
        return

    st.write(proposal.rationale)
    if not proposal.changed:
        st.info("The evidence does not move any band yet. That is a finding, not a failure.")
        return
    for line in proposal.changes:
        st.markdown(f"- {line}")

    if conn is None:
        st.warning("No Supabase connection -- the revision can be seen but not published.")
        return
    if st.button(f"Publish v{proposal.before.version + 1} -- the recommender uses it from now on",
                 type="secondary"):
        published = rules_store.publish(
            conn, proposal.bands, proposal.source, proposal.rationale, proposal.evidence
        )
        st.session_state.pop("proposal", None)
        st.success(
            f"Rules v{published.version} is active. Back on rung 3, the "
            "recommendation now comes from the revised bands."
        )
        st.rerun()


# --------------------------------------------------------------------------
# Rung 4
# --------------------------------------------------------------------------
def rung_production(df: pd.DataFrame, spec: model_mod.Spec,
                    load: data_mod.LoadResult) -> None:
    st.subheader(SUBHEADS[3])
    st.caption(
        "Three things happen here: a job writes the scores to a table, people "
        "send verdicts on the recommendations, and a second agent rewrites the rules."
    )

    conn = data_mod.get_connection()
    trained = get_model(df, spec)
    scored = model_mod.score_records(trained, df, spec, [])
    rules_set = rules_store.load_active(conn)
    bands = agent_mod.Bands.from_rules(rules_set)
    source = "Supabase" if load.source == "supabase" else "the local CSV"

    st.markdown("**1 · Write the scores**")
    st.caption(f"Model {trained.version} · rules v{rules_set.version} · "
               f"{len(scored):,} customers scored from {source}.")
    target = "Supabase" if conn is not None else "the local predictions file"
    if st.button(f"Score and write the {WRITE_BATCH} riskiest to {target}", type="primary"):
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "record_id": str(row[spec.id_column]),
                "probability": round(float(row["probability"]), 4),
                "recommended_action": bands.recommend(row).action,
                "model_version": f"{trained.version} · rules v{rules_set.version}",
                "created_at": now,
            }
            for row in scored.head(WRITE_BATCH).to_dict("records")
        ]
        ok, message = data_mod.write_predictions(conn, rows)
        (st.success if ok else st.error)(message)

    st.divider()
    st.markdown("**2 · Learn from what came back**")
    counts = data_mod.loop_counts(conn)
    st.caption(f"{counts['logged']:,} decisions logged · "
               f"{counts['verdicts']:,} verdicts from people on rung 3.")
    feedback = data_mod.read_feedback(conn)
    learner_panel(conn, rules_set, feedback)

    st.divider()
    st.markdown("**3 · Who changed the rules, and when**")
    versions = rules_store.history(conn, limit=5)
    if not versions:
        st.caption("No Supabase connection, so no history: the rules are ladder.toml's v1.")
    else:
        for row in versions:
            when = str(row.get("created_at", ""))[:16].replace("T", " ")
            st.markdown(f"- v{row['version']} · {row.get('source', '')} · {when} -- "
                        f"{row.get('rationale', '') or 'initial bands from ladder.toml'}")
        if len(versions) >= 2 and versions[0].get("bands") is not None:
            newest = rules_store.from_row(versions[0])
            previous = rules_store.from_row(versions[1])
            for line in learner_mod._diff(previous, newest.bands):
                st.caption(f"v{previous.version} -> v{newest.version}: {line}")

    with st.expander("Audit trail -- the predictions table"):
        recent = data_mod.read_predictions(conn)
        if recent.empty:
            st.info("Nothing written yet -- press the button above.")
        else:
            st.dataframe(recent, width="stretch", hide_index=True)

    with st.expander("Latest verdicts from people"):
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


if __name__ == "__main__":
    main()
