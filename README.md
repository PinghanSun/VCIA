# VCIA

Research code for **Auditing Update Combinations in Decentralized Federated Graph-Level Clustering**. Version Composition with Independent Auditing (VCIA) selects an exact-quota combination of current and stored operator versions, freezes the complete proposed execution state, and applies its registered audit route before committing.

[中文运行说明](docs/运行说明.md) · [Protocol and outputs](docs/protocol.md) · [Numerical studies](legacy/README.md) · [Archived WL pilot](legacy/wl_pilot/README.md)

## Try the decision walkthrough

Open [the standalone example](examples/admission_walkthrough.html) in a browser after downloading the repository. It shows the two decisions below and lets you expand the input probabilities, individual scores, and all three quota masks. It works offline without Python.

| Scenario | Policy | Proposed current peers | Outcome |
| --- | --- | --- | --- |
| Individual changes cancel when combined | VCIA | None | Holds after screening |
| Individual changes cancel when combined | Unscreened | `[0, 1]` | Commits both after audit |
| Ranking rules choose different masks | VCIA | `[0, 1]` | Commits both after audit |
| Ranking rules choose different masks | Drift-First | `[0, 2]` | Holds after audit |

After installation, run `python -m pcoc.demo` to reproduce the decisions in the terminal. Run `python -m pcoc.demo --html examples/admission_walkthrough.html` from the repository root to regenerate the browser view. The example calls the production decision functions with fixed, hand-authored probabilities. No data download or training is required.

`COMMIT` denotes an accepted composition in this walkthrough. The demo does not write peer state or execute distributed commit. Each batch contains one unique hand-authored probe. Repeated inputs illustrate the calculation and do not provide independent-sampling evidence.

## Recompute the abstract comparisons

Run `python reproduction/recompute.py` with Python 3.11 or newer. The script uses the recorded CPU timings and fixed-snapshot decisions in [reproduction/](reproduction/). It requires no external datasets or Python packages.

- In perturbed single-process CPU runs, cumulative online runtime was **4.4% lower on PROTEINS** and **5.9% lower on IMDB-BINARY** than for Unscreened. Each dataset contributes 180 paired rounds with identical executed states and matching cache misses and probe counts.
- In fixed-snapshot replay under the same IID audit, VCIA yielded **6.6% more operator replacements** than J-Rank (723 vs. 678). Of 324 snapshots, 179 had a design-feasible candidate. The audit passed 130 VCIA proposals and 134 J-Rank proposals.

These are exploratory comparisons. The script recomputes the summaries from the supplied records. The prospective outer WL configurations below are separate studies.

## Install

Use Python 3.11 or newer. The WL pipeline uses CPU and requires no GPU. The Python module and historical policy identifier retain the names `pcoc` and `PCOC` for compatibility.

```sh
python -m venv .venv
# Linux/macOS: . .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## Inspect a study

```sh
python -m pcoc plan --config configs/outer_ranking.json --job 1:6 --list
python -m pcoc matrix --config configs/robustness.json
python -m pytest
```

`plan`, `matrix` without `--output`, and the unit tests use no benchmark data or learner training. `fetch` retrieves TU ZIP files from the upstream host. Existing ZIPs can be copied into `data/` and checked without extraction.

```sh
python -m pcoc fetch --config configs/outer_ranking.json --data-dir data
python -m pcoc check-data --config configs/outer_ranking.json --data-dir data
```

The binary study checks the registered SHA-256 of each archive by default. The transfer suite records the exact SHA-256 of every input. The upstream dataset catalog describes the dataset sources and terms. [TUDataset catalog](https://chrsmrrs.github.io/datasets/docs/datasets/)

## Run a registered job

Use a new output directory for each study or shard. Omit `--job` for the full plan.

```sh
python -m pcoc run --execute --config configs/outer_ranking.json --data-dir data --job 1:6 --output runs/ranking-01
python -m pcoc.analysis summarize --runs runs/ranking-01 --reference Unscreened --output runs/ranking-01-summary.json
```

| Configuration | Question | Planned trajectories |
| --- | --- | ---: |
| `outer_ranking.json` | Six ranking rules on the same unscreened, feasible, positive-update family at each frozen pool | 540 |
| `outer_screening.json` | Individual J veto, with a finite-target census of each available quota mask | 180 |
| `outer_audit.json` | Sampled VCIA, No-Verify, Census-Audit, Census-Direct, and Simultaneous-family | 450 |
| `outer_controls.json` | VCIA, J-Rank, Frozen, and the unprotected quota | 360 |
| `inner_calibration.json` | Inner nominations, full-pool ranking shadows, exact finite-target evidence | 12 |
| `outer_loopback.json` | Authenticated, journaled, single-host multiprocess TCP version admission | 18 |
| `outer_transfer.json` | Fixed WL protocol on five datasets with the declared class counts and a new split seed | 900 |
| `outer_recovery_100.json`, `outer_recovery_300.json` | Optional WL extensions for longer matched clean, stress and recovery schedules | 540 each |
| `robustness.json` | One-factor sensitivity configurations materialized from the inner base | `pcoc matrix` reports totals |

Run `python -m pcoc matrix --config configs/robustness.json --output generated-configs` to materialize the sweep. Keep each sensitivity condition in its own result directory. For sharded runs, put sibling shards under a dedicated parent and pass that parent to the report commands.

## Analyze recorded outputs

```sh
python -m pcoc.analysis screening --runs runs/screening --output runs/screening-report.json
python -m pcoc.analysis calibrate --runs runs/inner --output runs/inner-calibration.json
python -m pcoc.analysis analytic --output runs/binary-power.json
python -m pcoc.recovery --runs runs/screening --gap 0.01 --window 3 --output runs/paired-recovery.json
```

`calibrate` replays frozen inner evidence with 10,000 probe batches by default. Register its operating target and any selected route, tolerance, or budget before outer evaluation. The outer JSON files contain starting settings. Calibrate probe budgets separately for the optional 100/300-round extensions. The recovery gap/window in the example are illustrative values to register before outer analysis.

## Repository layout

```text
configs/       Registered study definitions and the one-factor grid
examples/      Offline walkthrough generated by the decision demo
reproduction/  Records and summary script for the two abstract comparisons
src/pcoc/      TU/WL learner, admission policies, run ledger, reports, and local TCP protocol
legacy/        Generated-response studies and archived exploratory WL pilot
tests/         Small fixtures for policy, analysis, and transaction invariants
docs/          Provenance, scope, and output documentation
.github/       Unit and CLI checks for Windows and Linux
```

The historical 20-round and 60-round studies use generated responses. Their method names are response-profile labels. The WL pilot is a separate graph-learning study. The prospective outer WL configurations have no recorded results. See the protocol notes for the outstanding extensions.

## Citation

See [CITATION.cff](CITATION.cff). The current metadata follows the anonymized manuscript.
