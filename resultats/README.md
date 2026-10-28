# Presentation figures

PNG figures comparing **TD3 and SAC against Avellaneda–Stoikov (AS)**. Colours: TD3 blue, SAC orange, AS grey. Titles show maker fees directly in basis points; 1 bp = 0.01%. All P&L values are in USDT.

Three presentation snapshots are included in `docs/figures/` and displayed in the repository README. The full generated figures and tables in this directory stay local and are excluded from Git. To regenerate them on a fresh clone, follow [the reproduction procedure](../docs/reproduction.md).

| Figure | How to read it |
|---|---|
| [01 · Net performance](figures/01_net_performance.png) | One-tick execution. Large dots: mean daily P&L across three training seeds. Small dots and vertical segments: individual seeds and their range, not confidence intervals. AS is deterministic for each fee setting. |
| [02 · Execution sensitivity](figures/02_execution_sensitivity.png) | The same selected policies and sessions under zero-, one- and two-tick execution. Lines: seed means. Shading: range across seeds. Vertical scales differ across panels. One tick remains the primary rule. |
| [03 · Difference from AS](figures/03_difference_from_as.png) | Paired session differences, averaging the three RL seeds first. Error bars: 2.5–97.5% quantiles of a circular bootstrap with two-session blocks and 10,000 draws. These descriptive intervals do not include uncertainty from training-seed sampling and are not adjusted for multiple comparisons. |
| [04 · Validation versus test](figures/04_validation_vs_test.png) | Mean P&L of the three selected policies under one-tick execution. Validation was used for selection. Connecting lines compare two period averages, rather than a continuous time series. |
| [05 · Cumulative session P&L](figures/05_cumulative_session_pnl.png) | Cumulative P&L across the 22 observed monthly sessions only. Lines: seed means. Shading: seed range. No trading between sessions or compounding is assumed; inventory is liquidated at each session end. |
| [06 · Training validation curves](figures/06_training_validation.png) | Selected hyperparameter configurations, three training seeds, best actor per checkpoint at each additional-transition count. Solid curves: checkpoints satisfying activity and inventory guardrails. Faint curves: all checkpoints. Scores are validation P&L at one tick, not test performance. |

## Source tables

- `tableaux/resultats_detailles.csv`: complete frozen test outputs, including controls retained for audit.
- `tableaux/synthese.csv`: all methods, fee settings and execution rules, with costs, fills and inventory.
- `tableaux/ecarts_as_bootstrap.csv`: paired TD3/SAC differences from AS and bootstrap quantiles.
- `tableaux/checkpoints_retenus.csv`: the 18 selected policies and their validation metrics.
- `tableaux/audit_training_selected.csv`: starting budgets and transition counts of selected policies.
- `tableaux/audit_training_budgets.csv`: completed fine-tuning budgets.
- `tableaux/audit_training_curves.csv`: validation trajectories used in figure 06.
- `verification.json`: artifact checks and file hashes.

Original maker/taker fee pairs are 7.5/7.5, 3/4.5 and 1.5/2.175 bps. Sessions run from 00:10 to 23:59 UTC on the first day of each month, January 2025 through October 2026.

Selection exception: the seed-67 TD3 policy selected under a 7.5 bps maker fee has no eligible candidate and uses the predefined fallback, with validation mean absolute inventory close to 10 lots against a threshold of 5. The seed-67 TD3 policy under a 3 bps maker fee retains its starting policy.

## Regenerate local figures

```sh
python scripts/audit_training_budget.py
python scripts/figures_finales.py
```

Neither command retrains policies or reruns the historical test. Figure 06 requires the local validation records and retained checkpoints.
