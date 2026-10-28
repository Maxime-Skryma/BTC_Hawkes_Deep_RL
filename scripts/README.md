# Commands

See [the reproduction procedure](../docs/reproduction.md) for execution order.

| Command | Purpose |
|---|---|
| `robust_study.py` | Generate baseline calibration, tuning and pretrained inputs |
| `train_checkpoint_study.py` | Synthetic checkpoint training and selection |
| `finetune_checkpoint_study.py` | Historical reference fine-tuning |
| `finetune_grid_study.py` | Hyperparameter grid, seeds 41 and 53 |
| `final_selection.py` | Classical tuning, grid selection, final checkpoint selection |
| `finetune_confirm.py` | Train missing confirmation seeds |
| `final_test.py` | Frozen historical test, three execution rules |
| `audit_training_budget.py` | Transition-budget audit and validation curves |
| `figures_finales.py` | PNG presentation figures and CSV tables |
| `audit_git_payload.py` | Read-only audit of prospective Git contents |

Other scripts support engine benchmarks and tests. Historical reports and generated artifacts are not shipped.
