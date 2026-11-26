# Reproduction guide

Run commands from the directory containing `README.md` with the [Python environment](README.md#set-up-the-environment) active. Give each rerun a new output directory. The single-process real-graph runners perform local learning and admission. Any loopback traffic and timing are scoped to the process and host running the command.

## Jetson run environment

For a Jetson rerun, use the [Jetson Linux setup](README.md#jetson-linux) on the board. The real-graph timing runners measure CPU wall-clock time on the executing host. During a paired timing run, keep the board's power mode and software environment fixed across both arms.

After activating the virtual environment, save the environment details beside the new runs:

```sh
mkdir -p runs/jetson
{
  uname -smr
  cat /etc/os-release
  if [ -r /proc/device-tree/model ]; then
    tr -d '\000' < /proc/device-tree/model
    printf '\n'
  fi
  if [ -r /etc/nv_tegra_release ]; then
    head -n 1 /etc/nv_tegra_release
  fi
  if command -v nvpmodel >/dev/null 2>&1; then
    nvpmodel -q
  fi
  python --version
  python -m pip freeze
} > runs/jetson/environment.txt 2>&1
```

Use the dataset and runner commands below on the same board. Report the resulting timing with its matching environment record. See [NVIDIA's Jetson Linux documentation](https://developer.nvidia.com/embedded/jetpack/downloads) for the JetPack and operating system versions supported on each board.

## Inputs

Create a `data/` directory next to README.md. Save the [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/) archives there with their original names and leave them zipped. Runners check the archive hashes before using the graphs.

| File | Expected SHA-256 |
| --- | --- |
| [PROTEINS.zip](https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip) | `2da8de15284b88edabca2888ce5444d62f364ed41159260977088c4e53d4d848` |
| [ENZYMES.zip](https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip) | `13d832eb6ffa084192daf6e5750250028a18437ee692c38d29a10cd60e18aaf4` |
| [IMDB-BINARY.zip](https://www.chrsmrrs.com/graphkerneldatasets/IMDB-BINARY.zip) | `b291ec8b26d85c70faa2ba0a2433e1f407ed2ef5d0fc072d36b9a95e49a1bb27` |
| [NCI1.zip](https://www.chrsmrrs.com/graphkerneldatasets/NCI1.zip) | `10e1458f3bd9224f14e6d7627e74dcfd13e48d376d73935e7bd2900590ef1d82` |

Supply the frozen cohort split JSON separately for MMFD. The graph runners listed here fit their operators from graph inputs. See the figure commands below for the required CSV and JSON exports.

## Paper-to-code map

| Result | Entry point | Evidence and outputs |
| --- | --- | --- |
| Main Table 3, paired screening | `run_core_gap.py timing` and `score_core_gap.py timing` | Paired rounds, identical executed states and cache misses, `timing_summary.csv` |
| Main Table 2 and §4, fixed-snapshot ranking and finite-pool checks | `run_checks.py`, `score_checks.py`, and `tools/summarize_replay.py` | `replay.csv`, per-policy and common-pass replacement totals, finite-pool feasibility |
| Main Table 2, original 60-round closed loop | `run_evidence.py` and `score_evidence.py` | Per-round admission, replacement totals, state fingerprints and held-out NMI/ARI |
| Main Table 1 and supplement Table S21, closed-loop component ablations | `run_closed_ablation.py` | 216 trajectories, `rounds.csv`, sealed predictions and per-split NMI/ARI |
| Supplement Table S22 and Fig. S2, NCI1 audit cost and feasible admission | `run_nci1_audits.py` | Scheduled snapshots, candidate decisions, audit timings and nested probe pools |
| Main §4 and supplement Tables S17–S18, scalar MMFD composition ablation | `run_mmfd_ablation.py` | 36 matched trajectory pairs, frozen graph splits, committed replacements and audit draws |
| Supplement Tables S15–S16 and S19–S20, audit-budget and probe-reuse checks | `run_vcia_budget_32768.py` and `run_vcia_near_threshold.py` | Closed-loop budget comparisons, paired estimates for reused and fresh probes, and near-threshold replay |
| Main Figs. 3–4 and supplement §§C–D, response-model stress tests | `python -m legacy` | Seeded generated probability trajectories, endpoint metrics, sensitivity and recovery |
| Main Figs. 3–4 and supplement Figs. S1–S2 | Scripts in `figure_source/` | Render the evaluation figures from externally supplied CSV or JSON inputs |

`J-Rank` in fixed-snapshot replay files is the historical key for **J-Rank-Feasible**. The main comparison uses the same audit rule and totals prospective replacements over each policy's own passing snapshots. The replay summary also reports totals on snapshots passed by both policies. The response-model studies use their own generated J-Rank policy profile.

## Verify archived results

If you have a separate copy of `VCIA_Verification_Records.zip`, place it beside the source directory. The commands below extract it, verify `SHA256SUMS.txt`, and recalculate the timing, snapshot replay, audit, component ablation, and NCI1 summaries.

```sh
python -m zipfile -e ../VCIA_Verification_Records.zip ..
python tools/verify_records.py --records ../VCIA_Verification_Records --score-predictions
```

The optional `--score-predictions` flag recalculates NMI and ARI from the saved held-out predictions. To summarize the archived ranking comparison, run:

```sh
python tools/summarize_ranking.py --input ../VCIA_Verification_Records/supplement/data/ranking_replay.csv --output runs/ranking-summary.csv
```

## Calibrate the reassignment check

Run `python tools/verification_calibration.py --output-dir runs/calibration` to generate the CSV and JSON for the binomial calibration table in Supplement B. The script uses the Python standard library and checks each of the eight passing-count cutoffs with two binomial CDF calculations. With 8,192 draws, the largest passing count is 53 and the pass probability at a true reassignment probability of 0.008 is 6.4146%.

## Run real-graph experiments

### Smoke tests

The first three commands use PROTEINS. The MMFD check uses PROTEINS, IMDB-BINARY, and the external frozen split file. The final command uses NCI1.

```sh
python experiments/run_evidence.py --smoke --data data --output runs/closed-loop-smoke
python experiments/run_checks.py --smoke --data data --output runs/replay-smoke
python experiments/run_closed_ablation.py --smoke --data data --output runs/closed-ablation-smoke
python experiments/run_mmfd_ablation.py --smoke --data data --splits data/mmfd_splits.json --output runs/mmfd-smoke
python experiments/run_nci1_audits.py --smoke --data data/NCI1.zip --output runs/nci1-smoke
```

### Fixed-snapshot replay and earlier closed-loop cohort

Run the closed-loop study and score its sealed predictions before launching the fixed-snapshot analysis.

```sh
python experiments/run_evidence.py --data data --output runs/closed-loop
python experiments/score_evidence.py --data data --runs runs/closed-loop
python experiments/run_checks.py --data data --closed-loop runs/closed-loop --output runs/replay
python experiments/score_checks.py --data data --closed-loop runs/closed-loop --runs runs/replay
python tools/summarize_replay.py --replay runs/replay/replay.csv
```

The [closed-loop protocol](experiments/evidence_protocol.json) and the [replay protocol](experiments/replay_protocol.json) record the seeds, peer settings, probe budgets, and audit rules. To reproduce the targeted test-split extension:

```sh
python experiments/run_core_gap.py plan --closed-loop runs/closed-loop --output runs/quality-extension
python experiments/run_core_gap.py extend --closed-loop runs/closed-loop --data data --output runs/quality-extension --outer 31504
python experiments/run_core_gap.py extend --closed-loop runs/closed-loop --data data --output runs/quality-extension --outer 31505
python experiments/run_core_gap.py extend --closed-loop runs/closed-loop --data data --output runs/quality-extension --outer 31506
python experiments/score_core_gap.py quality --closed-loop runs/closed-loop --runs runs/quality-extension --data data
```

### Audit-budget and probe reuse

Use the sealed `runs/closed-loop` output from a full `run_evidence.py` run. The budget runner checks the archived states and replays the audits. The near-threshold runner reuses the resulting nomination and probe records.

```sh
python experiments/run_vcia_budget_32768.py --data data --closed-loop runs/closed-loop --output runs/audit-budget
python experiments/run_vcia_near_threshold.py --data data --closed-loop runs/closed-loop --budget-run runs/audit-budget --output runs/near-threshold
```

The budget runner writes `rounds.csv`, `trajectories.csv`, and `probe_pairs.csv` for the composition, budget, and matched-probe comparisons. The second command writes `candidate_audits.csv` with paired replay and conditional probability estimates. For a quick check, add `--smoke` to both commands and use new output directories. Give both commands the full sealed closed-loop input even in smoke mode.

### Paired screening timing

```sh
python experiments/reference/run_pilot.py --dataset data/PROTEINS.zip --out runs/pilots/results_proteins_inner_v1
python experiments/reference/run_imdb_pilot.py --dataset data/IMDB-BINARY.zip --reference-manifest runs/pilots/results_proteins_inner_v1/manifest.json --out runs/pilots/results_imdb_binary_inner_v1
python experiments/run_core_gap.py timing --data data --pilot-runs runs/pilots --output runs/core-checks
python experiments/score_core_gap.py timing --runs runs/core-checks
```

### Expanded component ablations, MMFD, and NCI1

The expanded closed-loop study uses two initializations for each outer split, six outer seeds per dataset, and 60 rounds. The NCI1 study replays fixed nomination schedules, then retimes them with Census, IID 2,048, and IID 8,192 on four nested pools.

```sh
python experiments/run_closed_ablation.py --data data --workers 4 --output runs/closed-ablation
python experiments/run_mmfd_ablation.py --data data --splits data/mmfd_splits.json --output runs/mmfd
python experiments/run_nci1_audits.py --data data/NCI1.zip --output runs/nci1
```

Pass the frozen cohort splits to the MMFD runner with `--splits`. Each runner writes fresh audit decisions and counters. NCI1 also writes individual timing repeats with the executing host recorded in `execution.json`.

### Check Table 1 and the abstract's ablation results

Run this command after the full closed-loop and MMFD jobs finish:

```sh
python tools/summarize_ablations.py --wl runs/closed-ablation --mmfd runs/mmfd
```

The summary reports committed replacements and audit draws for the Table 1 arms, including the VCIA and No-Composition comparison under moderate perturbation in the WL-TFIDF study and across both conditions in the scalar MMFD study.

## Run the seeded response-model experiments

These commands use generated probabilities. Dataset names in the plots identify response profiles. See the [response-model notes](legacy/README.md) for the protocols and output formats.

```sh
python -m legacy main20 --output runs/numerical20
python -m legacy supplement20 --main runs/numerical20 --output runs/numerical-supplement
python -m legacy followup60 --main runs/numerical20 --output runs/numerical60
```

## Render the evaluation figures

Run the renderers from the extracted code directory. Supply the five Fig. 3 CSVs in `data/plots/experiment_3x3/`: `trajectories.csv`, `policy_contrasts.csv`, `mechanism_contrasts.csv`, `gx_interactions.csv`, and `recovery.csv`. Place the Fig. 4 sensitivity CSV and the supplementary plot JSON in `data/plots/`.

```sh
python -m pip install -c experiments/requirements.txt -r figure_source/requirements.txt
python figure_source/draw_experiment_3x3.py --input-dir data/plots/experiment_3x3
python figure_source/draw_sensitivity.py --input data/plots/Sensitivity_Delta_ACC.csv
python figure_source/draw_closed_loop_nci1.py --input data/plots/closed_loop_nci1_plot_data.json
```

The renderers write four PDFs to `runs/figures/`. System fonts and PDF metadata may differ on rerendering.

If you have the original expanded closed-loop and NCI1 experiment workbook, `python figure_source/extract_closed_loop_nci1.py path/to/workbook.xlsx` rechecks the workbook and exports two tables and the plot data to `runs/workbook-export/`. This optional command requires openpyxl. Pass the resulting `runs/workbook-export/closed_loop_nci1_plot_data.json` to the third renderer.
