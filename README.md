# webina-sep

The four-rung ladder — describe, predict, recommend, deploy — on `WA_Fn-UseC_-Telco-Customer-Churn.csv`
(7,043 rows, predicting `Churn` = `Yes`). Generated on
2026-09-09 by the `data-to-production` skill.

**Start with [REPORT.md](REPORT.md).** One table, seven rows, a measured number
on each.

## Run it

```bash
python3 -m venv venv_webina-sep
source venv_webina-sep/bin/activate
pip install -r requirements.txt

python ladder.py all                # one rung at a time, asking at each gate
python ladder.py all --auto --local-only   # unattended: prepare, model, app, report
streamlit run src/app.py
```

Going to production as well — a Supabase project, a public repo, a live URL:

```bash
python ladder.py all                # stops at the browser step with the form
python ladder.py deploy --record https://webina-sep.streamlit.app
python ladder.py all --from verify
```

## Layout

| Path | What it is |
|---|---|
| `ladder.toml` | The only file that differs between datasets |
| `ladder.py` | The runner: eight steps, fixed order |
| `src/` | The app and its four modules |
| `scripts/ladder/` | The steps themselves |
| `data/webina-sep.csv` | Prepared data — the Supabase seed and the offline fallback |
| `supabase/schema.sql` | Generated from the prepared columns |
| `tests/test_ladder.py` | The checks REPORT.md is built from |

Work is tracked in [TODO.md](TODO.md) (the plan) and [WorkLog.md](WorkLog.md)
(what actually happened).
