# WorkLog and push policy — webina-sep

## Areas
`data` · `model` · `app` · `supabase` · `deploy` · `docs` · `fix`

## Binary guard
`data/raw.csv` is the source file this project was built from. Check its size
before committing: anything over 25 MB belongs in `.gitignore`, not in history,
and the README says where to fetch it instead. `venv_webina-sep/` never enters git.

## Secrets
`.streamlit/secrets.toml` is gitignored and stays that way. Only the Supabase
publishable key (`sb_publishable_…`) is ever written to it. The secret key
belongs nowhere in this repo. The database password lives in
`~/.supabase/webina-sep.dbpass`, mode 600.

## Entry format
One dated bullet per change, newest at the bottom, area tag first:

    ## 2026-09-02
    - model: retrained after the band edit; recall 0.559, unchanged
