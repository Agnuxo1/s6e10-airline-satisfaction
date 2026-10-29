# Airline satisfaction — nested ensembles and Darwin-Cage diagnostics

Reproducible project for [Kaggle Playground S6E10](https://www.kaggle.com/competitions/playground-series-s6e10). CatBoost, LightGBM and XGBoost form the predictive control; our Darwin-Cage program search tests structured residual hypotheses. Public prediction libraries are credited separately.

## Results and contribution

| Experiment | Local evidence | Official public score recorded in project log |
|---|---|---|
| E001 CatBoost baseline | OOF AUC0.958327 |0.95798,submission57736176 |
| E014 feature/categorical model | OOF AUC0.960984 | Not assigned a separate official score here |
| B001,14 own GBDTs | nested/LOFO stack AUC0.961244 |0.96083,submission57759629 |
| S001,14 own+124 public members | nested stack AUC0.962066 |0.96166,submission57759904 |
| D001 Darwin residual study | single-model held-out fold0.95858→0.95959 | Diagnostic gain, not official gain |
| D002 Darwin on own blend,3seeds |1/0/1 programs confirmed; applying them lowers AUC about0.0006 | Rejected for final blend |
| D003 Darwin on full stack,3seeds |0/0/0 programs confirmed; calibration-only change−0.000004 | No contribution established on final stack |

The improvement from public members is **not attributed to Darwin-Cage**. D001 remains a positive result: independent program search finds the route interaction later absorbed by the stronger control. D002/D003 do not erase it, and D001 does not establish a final-ensemble gain. Scores above are traceable campaign records; an archived official receipt is needed for independent acceptance verification. See [complete log](experiments/LOG.md) and [configurations](experiments/configs).

## Reproduce

Python3.11+, Kaggle account already eligible for the competition, and the allowed original-data/public-member sources listed in the selected configuration:

```bash
python -m pip install -r requirements.txt
python tools/make_kernel.py experiments/configs/e001.json --cpu
```

This generates a **private** Kaggle kernel under `kernels/`; it does not submit anything. Inspect its data/model/notebook sources before uploading with Kaggle CLI. The source files are notebook bodies; `tools/make_kernel.py` injects `CONFIG`. OOF predictions, labels, public prediction libraries, datasets and submissions are obtained externally and are not redistributed here. Full stack reproduction requires the124-member public library and exact shared folds. CPU/GPU platform differences can affect timings and fitted models.

## Structure and licences

`src/runner_body.py`: base models; `stack_body.py`: nested stack; `darwin_core.py` and `darwin_body.py`: own residual search; `tools/`: packaging and fold-safe diagnostics. External credit: public library **cdeotte124** and dataset `arseniyshutko/binary-aviation-satisfaction-129k`; their source terms still apply. Code developed here: MIT. Follow [research protocol](docs/RESEARCH_PROTOCOL.md). Public-leaderboard selection remains exploratory, not a hidden-test generalisation guarantee.
