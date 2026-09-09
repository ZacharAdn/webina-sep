"""The four rungs, one app.

Rung 1  describe   -- the dashboard, reading records out of Supabase
Rung 2  predict    -- a model, its baseline, and the leakage story
Rung 3  recommend  -- bands from ladder.toml, and gpt-oss-120b as the second opinion
Rung 4  production -- write the predictions back to a table, and deploy

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
import insights as insights_mod  # noqa: E402
import model as model_mod  # noqa: E402
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

    bands = agent_mod.Bands.load()
    left, right = st.columns(2)
    for slot, estimator in ((left, "logreg"), (right, "tree")):
        trained = get_model_for(df, spec, estimator)
        probability = model_mod.score_record(trained, record)
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

    trained = get_model(df, spec)
    ranked = insights_mod.discriminative_categoricals(
        df, spec.target, spec.positive_label, k=2
    )
    extra = [row["column"] for row in ranked]
    scored = model_mod.score_records(trained, df, spec, extra)
    bands = agent_mod.Bands.load()

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
    st.caption("These three rows live in ladder.toml. Change one and rerun.")

    top = scored.head(25)
    labels = [
        f"{row[spec.id_column]} · risk {row['probability']:.0%}"
        for _, row in top.iterrows()
    ]
    choice = st.selectbox(
        "Pick a record", range(len(labels)), format_func=lambda i: labels[i]
    )
    record = top.iloc[choice].to_dict()

    rules = bands.recommend(record)
    left, right = st.columns(2)
    with left:
        st.markdown("##### The rules layer")
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
                llm = agent_mod.recommend_llm(record, rules)
            if llm is None:
                st.warning(
                    "The agent did not return a usable answer. The rules layer stands."
                )
            else:
                st.success(llm.action)
                st.caption(llm.reason)
                if llm.unknowns:
                    st.markdown(f"**What it does not know:** {llm.unknowns}")

    st.divider()
    st.markdown("**The 25 highest-risk records, with the band that fires on each.**")
    table = top.copy()
    table["recommendation"] = [
        bands.recommend(row).action for row in top.to_dict("records")
    ]
    table["probability"] = table["probability"].map("{:.0%}".format)
    st.dataframe(table, width="stretch", hide_index=True)


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
    bands = agent_mod.Bands.load()

    c1, c2, c3 = st.columns(3)
    c1.metric("Data source", "Supabase" if load.source == "supabase" else "local CSV")
    c2.metric("Model version", trained.version)
    c3.metric("Records scored", f"{len(scored):,}")

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
                "model_version": trained.version,
                "created_at": now,
            }
            for row in scored.head(n).to_dict("records")
        ]
        ok, message = data_mod.write_predictions(conn, rows)
        (st.success if ok else st.error)(message)

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
