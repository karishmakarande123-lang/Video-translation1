# Full Formal Evaluation: Baseline vs. Baseline+Prosody+Isochrony

10 real Marathi sentences (IISc SYSPIN dataset), evaluated against the REAL ground-truth recording of the same speaker saying the same sentence (MCD, F0-RMSE/correlation, ECAPA speaker similarity), and against the known-correct target text (BLEU/chrF++, via the already-computed round-trip ASR transcriptions). Duration compliance reuses the isochrony ratios from the stress test in samples/research_comparison/ — see TECHNICAL_REPORT.md §6.5 for why those ratios are more extreme than a typical per-utterance run.

## Corpus-level summary

| Metric | Baseline | Enhanced |
|---|---|---|
| Corpus BLEU | 1.90 | 1.74 |
| Corpus chrF++ | 29.46 | 21.13 |
| Mean MCD (dB, lower=better) | 6.40 | 7.10 |
| Mean F0-RMSE (Hz, lower=better) | 24.2 | 43.3 |
| Mean F0 correlation | 0.846 | 0.764 |
| Mean ECAPA speaker similarity | 0.846 | 0.307 |
| Duration compliance rate | 0% (0/10) | 100% (10/10) |

## Per-sample results

| # | Domain | BLEU (B/E) | chrF++ (B/E) | MCD dB (B/E) | F0-RMSE Hz (B/E) | F0-corr (B/E) | Speaker sim (B/E) | Dur ratio (B/E) |
|---|---|---|---|---|---|---|---|---|
| 1 | AGRICULTURE | 2.29/2.15 | 27.62/23.45 | 6.42/7.29 | 16.8/37.0 | 0.899/0.826 | 0.838/0.304 | 0.46/1.00 |
| 2 | BOOKS | 3.24/0.0 | 31.56/0.0 | 6.5/7.34 | 21.3/40.8 | 0.868/0.752 | 0.774/0.216 | 0.46/1.00 |
| 3 | EDUCATION | 2.39/2.41 | 30.33/19.98 | 6.52/7.45 | 32.4/64.5 | 0.838/0.614 | 0.832/0.308 | 0.59/1.00 |
| 4 | EVALUATION | 2.52/2.16 | 31.0/20.77 | 6.35/7.08 | 22.6/39.9 | 0.876/0.836 | 0.873/0.315 | 0.60/1.00 |
| 5 | FINANCE | 3.72/3.38 | 36.8/30.75 | 6.08/6.87 | 19.7/40.1 | 0.881/0.783 | 0.896/0.309 | 0.52/1.00 |
| 6 | GENERAL | 0.0/8.1 | 0.0/25.44 | 6.18/7.02 | 21.0/34.8 | 0.834/0.799 | 0.857/0.352 | 0.61/1.00 |
| 7 | HEALTH | 11.98/6.3 | 40.63/36.06 | 6.76/7.11 | 35.5/37.4 | 0.738/0.816 | 0.828/0.306 | 0.39/1.00 |
| 8 | OTHERS | 2.28/0.0 | 30.7/0.0 | 6.32/6.94 | 35.1/48.4 | 0.721/0.663 | 0.869/0.297 | 0.49/1.00 |
| 9 | POLITICS | 4.55/4.48 | 28.76/21.46 | 6.67/7.16 | 14.4/47.8 | 0.932/0.732 | 0.844/0.326 | 0.49/1.00 |
| 10 | WEATHER | 3.42/2.66 | 37.49/29.45 | 6.2/6.74 | 23.3/42.3 | 0.873/0.815 | 0.85/0.336 | 0.53/1.00 |
