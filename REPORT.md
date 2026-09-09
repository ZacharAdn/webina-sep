# webina-sep — ladder report, 2026-09-09 20:50

| Rung | Status | Measured |
|---|---|---|
| 1 prepare | PASS | 7,043 rows · 21 columns · 11 NULLs in total_charges · 26.5% positive |
| 2 model | PASS | baseline 73.5% acc / 0% recall · model 80.6% / 55.9% / AUC 0.84 · leak 92.9% vs 49.5% |
| 3 app | PASS | 4 rungs · 0 exceptions · 25 rows to Supabase |
| 4 supabase | PASS | ref REDACTED-PROJECT-REF · 7,043 rows · 11 NULLs kept |
| 5 github | PASS | ZacharAdn/webina-sep (public) |
| 6 deploy | PENDING | waiting for the browser step |
| 7 verify | SKIPPED |  |

```
Rerun one step:   python ladder.py model
Rerun from step:  python ladder.py all --from supabase
Open the app:     streamlit run src/app.py
```
