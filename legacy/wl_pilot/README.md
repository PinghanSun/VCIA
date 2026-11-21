# Historical WL graph-learning pilot

This directory ports the archived Weisfeiler–Lehman prototype learner on PROTEINS and IMDB-BINARY. The default design uses 30 rounds, seeds 17, 29 and 43, clean and perturbed conditions, and two closed-loop policies, PCOC and Unscreened. It scores clustering quality on inner validation data and logs other ranking policies on the same frozen proposals. Keep these outputs under a separate run root from the outer-split experiments.

## Inputs and environment

Use Python 3.13 for the dependency versions recorded by the original run. From the repository root, install the pins in a separate virtual environment.

```sh
python -m pip install -r legacy/wl_pilot/requirements.txt
```

Supply the original TU ZIP files with the expected directory and record order. The table gives the byte fingerprints recorded in the archived manifests. Dataset archives and generated artifacts can be supplied locally.

| Dataset file | SHA-256 |
| --- | --- |
| `PROTEINS.zip` | `2da8de15284b88edabca2888ce5444d62f364ed41159260977088c4e53d4d848` |
| `IMDB-BINARY.zip` | `b291ec8b26d85c70faa2ba0a2433e1f407ed2ef5d0fc072d36b9a95e49a1bb27` |

## Regenerate the pilot

```sh
python -m legacy.wl_pilot.run_pilot --dataset data/PROTEINS.zip --out runs/wl-proteins
python -m legacy.wl_pilot.run_imdb_pilot --dataset data/IMDB-BINARY.zip --reference-manifest runs/wl-proteins/manifest.json --out runs/wl-imdb
```

The IMDB-BINARY extension copies the design and accepts the original PROTEINS manifest with matching archived source hashes or a newly generated PROTEINS manifest from this port. Each run root contains `manifest.json`, `run_index.json`, split assignments, round logs, finite audit effects, executed state records, and inner-validation predictions. Use new output paths. The recorded runtimes depend on the local environment.

## Summarize, replay and check

Use the generated run roots from the prior step or replace them with complete historical run roots.

```sh
python -m legacy.wl_pilot.summarize_pilot runs/wl-proteins runs/wl-imdb --out runs/wl-tables
python -m legacy.wl_pilot.replay_design runs/wl-proteins --out runs/wl-proteins-design
python -m legacy.wl_pilot.replay_design runs/wl-imdb --out runs/wl-imdb-design
python -m legacy.wl_pilot.calibrate_budget --run-root runs/wl-proteins --out runs/wl-proteins-budget --policies PCOC Unscreened J-Rank-Feasible Drift-First Random-Feasible No-Count No-Age
python -m legacy.wl_pilot.calibrate_budget --run-root runs/wl-imdb --out runs/wl-imdb-budget --policies PCOC Unscreened J-Rank-Feasible Drift-First Random-Feasible No-Count No-Age
python -m legacy.wl_pilot.calibration --json runs/wl-audit-calibration.json
python -m legacy.wl_pilot.verify_artifacts runs/wl-proteins --dataset data/PROTEINS.zip
python -m legacy.wl_pilot.verify_artifacts runs/wl-imdb --dataset data/IMDB-BINARY.zip --reference-manifest runs/wl-proteins/manifest.json
```

| Command | Evidence and outputs |
| --- | --- |
| `summarize_pilot` | Reads the original round logs and exports matched screening outcomes and costs to `matched_screening.csv`. `fixed_state_ranking.csv` quantifies changes to selected masks, verification decisions and counterfactual accepted updates on the screened trajectory snapshots. |
| `replay_design` | Recomputes frozen candidate selection and verification over the original design sensitivity grid. The historical defaults use 512 design draws and 8,192 verification draws. |
| `calibrate_budget` | Changes verification settings with the candidate and observed trajectory frozen. Reports the exact binomial component, Monte Carlo drift and joint pass estimates, and modeled and observed inference costs. The historical grid uses 10,000 batches per context and setting. |
| `calibration` | Calculates verification operating characteristics from the registered binomial model and finite audit targets. |
| `verify_artifacts` | Reconstructs the original recorded splits, schedule, proposals, checks, and executed states from the supplied dataset and run artifacts. Provide the original reference manifest when checking the historical IMDB-BINARY reference digest. Run with Python assertions enabled. |

`provenance.json` records the source fingerprints of the archived donor scripts. The port changes import paths and CLI I/O. The numerical learner, decision rule, verification inputs and original replay settings follow those scripts.
