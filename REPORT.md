# webina-sep — ladder report, 2026-09-09 22:02

| Rung | Status | Measured |
|---|---|---|
| 1 prepare | PASS | 7,043 rows · 21 columns · 11 NULLs in total_charges · 26.5% positive |
| 2 model | PASS | baseline 73.5% acc / 0% recall · model 79.3% / 52.1% / AUC 0.84 · leak 93.7% vs 48.4% |
| 3 app | PASS | 4 rungs · 0 exceptions · 25 rows to Supabase |
| 4 supabase | PASS | ref REDACTED-PROJECT-REF · 7,043 rows · 11 NULLs kept |
| 5 github | PASS | ZacharAdn/webina-sep (public) |
| 6 deploy | PASS | https://webina-sep.streamlit.app |
| 7 verify | PASS | live read 7,043 · write id 101 at 19:02:04 · read back OK · app 200 |

```
Rerun one step:   python ladder.py model
Rerun from step:  python ladder.py all --from supabase
Open the app:     streamlit run src/app.py
```
