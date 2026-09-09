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
    st.markdown(
        f"**The baseline earns {trained.baseline.accuracy:.0%} accuracy by never "
        f"predicting a single '{spec.positive_label}'** -- recall "
        f"{trained.baseline.recall:.0%}, ROC-AUC {trained.baseline.roc_auc:.2f}. The "
        f"model is worth talking about only because it finds "
        f"{trained.metrics.recall:.0%} of them at ROC-AUC "
        f"{trained.metrics.roc_auc:.2f}. Accuracy alone would have hidden that."
    )

    st.divider()
    st.markdown("**The same model, two splits, two very different numbers.**")
    honest, leaky = get_leakage(df, spec)
    st.dataframe(pd.DataFrame([leaky, honest]), width="stretch", hide_index=True)
    st.caption(
        f"The upper row balanced the classes before splitting, so copies of the same "
        f"record sat on both sides of the wall. It reports catching "
        f"{leaky['recall']:.0%}. The lower row split first and catches "
        f"{honest['recall']:.0%} (ROC-AUC {leaky['roc_auc']} against "
        f"{honest['roc_auc']}). Same model class, same data, one line of code apart. "
        "The first number is the one that gets promised in a meeting."
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

    st.caption(
        "Two families, one record. When they disagree, the gap is the honest "
        "size of the uncertainty -- and neither number is more true than the "
        "other because it is larger."
    )

    typed = {column: record[column] for column in drivers + numbers}
    handoff.stash_scored(st.session_state, record, probabilities, typed)
    st.caption("This customer is now the one rung 3 opens on.")

    filled = {
        column: value
        for column, value in record.items()
        if column not in drivers + numbers
    }
    with st.expander(f"The {len(filled)} fields filled in from the table"):
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

    st.write(
        "A probability is not a decision. The bands below are the baseline of the "
        "recommendation, exactly the way 'nobody is positive' was the baseline of the "
        "prediction. Anything an agent adds has to beat them."
    )
    st.dataframe(
        pd.DataFrame(
            [{"band": b.name, "from": f"{b.minimum:.0%}", "action": b.action}
             for b in bands.bands]
        ),
        width="stretch", hide_index=True,
    )
    st.caption(
        f"Rules v{rules_set.version} · source: {rules_set.source}"
        + (f" · {rules_set.rationale}" if rules_set.source != "toml" else
           " · these three rows started life in ladder.toml; the learner below writes the next version.")
    )

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

    rules = bands.recommend(record)
    left, right = st.columns(2)
    with left:
        st.markdown("##### Agent 1 · the rules layer")
        st.success(rules.action)
        st.caption(rules.reason)
    with right:
        st.markdown("##### gpt-oss-120b on Groq, second opinion")
        if not agent_mod.llm_available():
            st.info(
                "GROQ_API_KEY is not set, so only the rules layer is running. "
                "That is the honest state of the demo, not a failure."
            )
        elif st.button("Ask the agent", type="secondary"):
            with st.spinner("Thinking..."):
                llm = agent_mod.recommend_llm(record, rules, feedback=feedback)
            if llm is None:
                st.warning(
                    "The agent did not return a usable answer. The rules layer stands."
                )
            else:
                st.success(llm.action)
                st.caption(llm.reason)
                if llm.unknowns:
                    st.markdown(f"**What it does not know:** {llm.unknowns}")
                if feedback:
                    st.caption(f"It read the last {min(8, len(feedback))} feedback rows before answering.")

    st.divider()
    rules_chat_panel(conn, rules_set, feedback, record)
    st.divider()
    feedback_form(conn, record, rules, rules_set, spec)

    st.divider()
    learner_panel(conn, rules_set, feedback)

    st.divider()
    st.markdown("**The 25 highest-risk records, with the band that fires on each.**")
    table = top.copy()
    table["recommendation"] = [
        bands.recommend(row).action for row in top.to_dict("records")
    ]
    table["probability"] = table["probability"].map("{:.0%}".format)
    st.dataframe(table, width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# The loop -- a verdict on the recommendation, and the agent that learns from it
# --------------------------------------------------------------------------
OUTCOMES = {"unknown": "don't know yet", "stayed": "stayed", "left": "left"}


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
    """Agent 2. Shows what it read, proposes a revision, and publishes it on request."""
    st.markdown("##### Agent 2 · the one that learns")
    st.write(
        "The recommender never reads its own track record. This agent does: it "
        "folds every verdict onto the band that produced it and rewrites the bands. "
        "It does not touch the model -- ten verdicts are enough to move a rule, a "
        "retrain needs thousands of outcomes."
    )
    if not feedback:
        st.info("No feedback yet. Send one above and this panel wakes up.")
        return

    digested = learner_mod.digest(rules_set, feedback)
    st.dataframe(
        pd.DataFrame([d.as_row() for d in digested.values()]),
        width="stretch", hide_index=True,
    )
    st.caption(f"{len(feedback)} verdicts read, on rules v{rules_set.version}.")

    prefer_llm = st.toggle(
        "Let gpt-oss-120b write the revision (the rule-based learner runs otherwise)",
        value=learner_mod.llm_available(), disabled=not learner_mod.llm_available(),
    )
    if st.button("Let the learner revise the rules", type="primary"):
        with st.spinner("Reading the verdicts..."):
            proposal = learner_mod.propose(rules_set, feedback, prefer_llm=prefer_llm)
        st.session_state["proposal"] = proposal

    proposal = st.session_state.get("proposal")
    if proposal is None or proposal.before.version != rules_set.version:
        return

    st.markdown(f"**Proposal by `{proposal.source}`**")
    st.write(proposal.rationale)
    if not proposal.changed:
        st.info("The evidence does not move any band yet. That is a finding, not a failure.")
        return
    before = pd.DataFrame(proposal.before.as_rows())
    after = pd.DataFrame([b.__dict__ for b in proposal.bands])
    c1, c2 = st.columns(2)
    c1.markdown(f"v{proposal.before.version} (now)")
    c1.dataframe(before, width="stretch", hide_index=True)
    c2.markdown(f"v{proposal.before.version + 1} (proposed)")
    c2.dataframe(after, width="stretch", hide_index=True)
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
            f"Rules v{published.version} is active. Pick the same record above: "
            "the recommendation now comes from the revised bands."
        )
        st.rerun()


# --------------------------------------------------------------------------
# Rung 4
# --------------------------------------------------------------------------
def rung_production(df: pd.DataFrame, spec: model_mod.Spec,
                    load: data_mod.LoadResult) -> None:
    st.subheader(SUBHEADS[3])

    st.markdown(
        "The first production version of almost any model is not a real-time "
        "endpoint. It is a job that writes to a table, and a page that reads the "
        "table. That is what this tab does."
    )

    conn = data_mod.get_connection()
    trained = get_model(df, spec)
    scored = model_mod.score_records(trained, df, spec, [])
    rules_set = rules_store.load_active(conn)
    bands = agent_mod.Bands.from_rules(rules_set)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data source", "Supabase" if load.source == "supabase" else "local CSV")
    c2.metric("Model version", trained.version)
    c3.metric("Rules version", f"v{rules_set.version}")
    c4.metric("Records scored", f"{len(scored):,}")

    n = st.slider(
        "How many of the riskiest records to write", 5, 100, 25, step=5
    )
    target = "Supabase" if conn is not None else "the local predictions file"
    if st.button(f"Score and write to {target}", type="primary"):
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "record_id": str(row[spec.id_column]),
                "probability": round(float(row["probability"]), 4),
                "recommended_action": bands.recommend(row).action,
                "model_version": f"{trained.version} · rules v{rules_set.version}",
                "created_at": now,
            }
            for row in scored.head(n).to_dict("records")
        ]
        ok, message = data_mod.write_predictions(conn, rows)
        (st.success if ok else st.error)(message)

    st.markdown("**How much of the loop has closed**")
    counts = data_mod.loop_counts(conn)
    l1, l2, l3 = st.columns(3)
    l1.metric("Decisions logged", f"{counts['logged']:,}")
    l2.metric("With an outcome", f"{counts['with_outcome']:,}")
    l3.metric("Verdicts from people", f"{counts['verdicts']:,}")
    st.caption(
        "On the day this was built the middle number was zero, and that was the "
        "point: the column exists and waits for reality. Every verdict on rung 3 "
        "fills it, and the learner there rewrites the rules from it."
    )

    st.markdown("**What is in the predictions table right now**")
    recent = data_mod.read_predictions(conn)
    if recent.empty:
        st.info(
            "Nothing to read yet -- press the button above. Without a Supabase "
            "connection the rows go to data/predictions.csv; run "
            "`python ladder.py supabase` to get the table."
        )
    else:
        st.dataframe(recent, width="stretch", hide_index=True)
        st.caption(
            "Local mode: this is data/predictions.csv on the machine running the app. "
            "It disappears with the container and nobody else can read it -- that gap "
            "is exactly what the Supabase table closes."
            if conn is None else
            "This table is the audit trail. Someone who was not in the room can read "
            "what the model said, when, and under which version -- without running "
            "any of this code."
        )


if __name__ == "__main__":
    main()
