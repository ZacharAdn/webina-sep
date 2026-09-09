# TODO — webina-sep

Marks: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` cancelled

Ticked items were completed by the `data-to-production` run on 2026-09-09; open
items are what that run could not finish on its own.

## Milestone 0 — Scaffolding
- [x] Project skeleton, `.gitignore`, git repo, first commit
- [x] `ladder.toml` written from the dataset (7,043 rows, target `Churn`)

## Milestone 1 — Data and model
- [ ] Step 1 `prepare` — snake_case, real NULLs, id and target verified
- [ ] Step 2 `model` — baseline, logistic regression, leakage demo

## Milestone 2 — The app
- [ ] Step 3 `app` — four rungs headless, zero exceptions, rung 4 writes

## Milestone 3 — Supabase
- [ ] Step 4 `supabase` — project, schema, import, secrets

## Milestone 4 — Deploy
- [ ] Step 5 `github` — repo created and pushed
- [ ] Step 6 `deploy` — the Streamlit Cloud form, in a browser
- [ ] Step 7 `verify` — live read, write, read-back

## Milestone 5 — Judgement
- [ ] Step 8 `report` — REPORT.md assembled
- [ ] Read REPORT.md and decide whether the bands in `ladder.toml` are right
