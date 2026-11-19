# VCIA protocol and outputs

## Study roles and contrasts

- Roles use grouped graph fingerprints and separate role, partition, initialization, proposal, and probe seeds. Preprocessing fits on training graphs. Labels stratify client allocation and score evaluation assignments. The online admission routes read graph-output evidence.
- `outer_screening` holds the design score, ranking key, thresholds, independent audit, and cache policy constant. The common-pool census reports J-veto losses among empirical feasible masks, exactly admissible masks, and their intersection. Closed-loop arms can obtain different future pools through their commits.
- `outer_ranking` uses the unscreened VCIA ordering as the reference (`Unscreened`). All six comparison rules start from the same eligibility definition. In addition to the closed-loop runs, each reference round records six offline decisions on its identical frozen pool and probe draws. The historical WL pilot's offline ranking shadows use its archived screened family.
- No-Verify and Census-Audit preserve the screened VCIA nomination. Census-Direct uses the declared finite audit role for both selection and admissibility. Simultaneous-family uses a family-wise allocation calibrated to the full mask count before nominating its passing member.
- The architecture fits WL features and local prototype banks, aligns centroids by squared matching, and serializes the actual committed maps. Authentication in the loopback adapter assumes a trusted coordinator and a designated auditor on one host. The workload ledger for this adapter counts TCP application frames, plus elapsed time and restart counters. Shared-host measurements cannot establish network performance on separate machines.
- Graph-level WL regimes use fixed immutable perturbations without label or probe queries. Burst availability holds an identity's draw for four rounds by default when enabled by the robustness grid. Proposal topology changes the local committed-neighbor warm start. Pool sizes of 8, 10, 12, and 16 admit 28, 120, 220, and 4,368 masks at their configured quotas.

## Output contract

| Path | Contents |
| --- | --- |
| `manifest.json` | Selected job IDs, exact settings, archive digests, implementation digests, environment and numerical library versions |
| `run_index.json` | Completed or failed job entries by registered job ID |
| `<dataset>/p<partition>_i<init>/<regime>/<arm>/registration.json` | Frozen policy, exact normalized tolerances, data and feature fingerprints, inference format, audit and evaluation role hashes |
| `roles.json` | Zero-based source-archive graph IDs, per-client train IDs, split/partition seeds and allocation attempts |
| `rounds.jsonl` | Per-round nomination, transaction and state hashes, verification, costs, content clocks/changes, evaluation metrics, predictions when enabled, plus offline evidence as configured |
| `checkpoint.npz` | Final ordered canonical operators/claims and both clock arrays |
| `summary.json` / `failure.json` | Completion and endpoint or a recorded failure with traceback |
| `protocol/` | Loopback coordinator journal for the job |

`selected_payload_hash` binds exact transaction context and is policy specific. `nominated_state_hash` supports cross-policy comparison of the actual proposed content. `executed_state_hash` identifies the committed or retained content. `content_clocks` advance only on committed payload changes. Census `verify_draws` is zero with all audited graphs counted under `verify_unique_graphs`. Population reference work and evaluation-label scores stay under separate ledgers. The overall verification timer includes staging, auditing, and finalizing a nomination where applicable. TCP elapsed time measures overlapping work, so do not add it to total round time.

Report final-round NMI first and ARI, matched ACC, and macro-F1 as secondary outcomes. Intervals pair within partition/initialization and average nested initializations at the partition level. Record failed and unstarted jobs separately from successfully evaluated holds. Keep inner calibration and outer reports in separate directories.

The summaries count successful commits, finite-target violations per scheduled round (`r_all`) and per successful commit (`r_commit`), and admissible canonical content changes (`U_adm` and `C_adm`). These violation and admissible-change summaries require an exact reference for every successful commit in the group. `nominated_H` tracks sampled design detectability independently of committed canonical changes.

## Evidence and remaining interfaces

The archive recognizes three types of work: generated numerical responses, the archived WL pilot, and the prospective outer learner. `legacy/provenance.json` identifies the historical generated scripts and profile data. Public TU ZIP files are obtained separately. Generated outcomes, benchmarks, logs, and private author files are excluded from the release.

| Pending item | Required input |
| --- | --- |
| Final confirmation settings | Inner calibration results and author-registered budgets, route, and recovery criteria before outer evaluation |
| Native external baselines | Upstream code revisions and graph-task, supervision, coordinate, communication, and eligibility mappings for each usable baseline |
| GIN and estimated-K transfer | An agreed unsupervised training objective, clustering head, alignment and codec, and recalibration under the new K/output range |
| Adaptive attacks | An agreed attacker access model, query budget, and objective before payloads can be evaluated |
| Multi-host performance | Separate hosts and registered network, compute, and system-fault conditions |

Repository organization follows common ML artifact conventions for installation, configuration, run commands, and documentation. [Benchmarking GNNs](https://github.com/graphdeeplearning/benchmarking-gnns) · [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) · [GitHub README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
