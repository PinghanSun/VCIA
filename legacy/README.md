# Reproducing the generated-response studies

These three studies use a seeded joint-probability generator. Dataset names identify scenario parameters, and method names identify response profiles. The generator uses synthetic objects with synthetic class labels. For the graph-learning implementation and matched admission-policy studies, see the repository README.

## Generate the data

Run from the repository root with Python 3.10 or later. These three commands require NumPy.

```sh
python -m pip install -r legacy/requirements.txt
python -m legacy main20 --output runs/numerical20
python -m legacy supplement20 --main runs/numerical20 --output runs/numerical-supplement
python -m legacy followup60 --main runs/numerical20 --output runs/numerical60
```

Use a new output directory for each command. The follow-up checks the first 20 latent and metric rounds of every clean or stopped trajectory against the main20 output. It also checks that stopped and continuous branches match through their shared exposure in rounds 5–12. These runs require no dataset download or GPU. Allow several hundred megabytes for per-round CSVs and compressed statistics when keeping all three outputs.

| Command | Design | Output |
| --- | --- | --- |
| `main20` | Five scenarios, 27 profiles, four severities, three partitions, five initializations, 20 rounds | 8,100 trajectories and 162,000 points, condition means and descriptive SDs, paired contrasts, latent recovery, probability examples, and analytic cost estimates |
| `supplement20` | 210 sensitivity settings, 200 learning-label settings, and 100 new mechanism controls | 7,650 new trajectories and 153,000 points, 120 separately labeled main conditions reused in summaries, G/X factorial interactions, and analytic resource and topology estimates |
| `followup60` | Five scenarios, 27 profiles, 15 nested outcomes, and seven branches per profile and scenario | 14,175 trajectories and 850,500 points, primary and sensitivity recovery decisions, grouped counts, and matched probability examples |

## Read the outputs

- Every command produces a `manifest.json`, `rounds_<scenario>.csv`, `round_means.csv`, `run_parameters.csv`, and `stats_<scenario>.npz`. Per-round results include latent score, matched ACC, NMI, ARI, macro-F1, and evaluation NLL from the same generated probabilities. The four clustering metrics are multiplied by 100.
- The statistics archives contain `metadata`, `confusion`, `class_nll_sums`, `metrics`, and `latent`. Main metadata columns are profile index, severity index, partition, and initialization. Supplement metadata columns are one-based condition ID, partition, and initialization. Follow-up metadata columns are profile index, branch code, severity index, partition, and initialization. Profile and severity indices are zero-based and follow `profiles.json`. Follow-up branch codes are clean 0, stop 1, and continuous 2.
- The primary endpoint is round 20, with rounds 16–20 as the secondary endpoint. PCOC-minus-comparator intervals average the five paired initializations within each partition and apply a pointwise Student-(t) interval across three partition means. A smaller evaluation NLL indicates lower loss. Condition-level SDs describe the 15 nested outcomes.
- Recovery uses the contemporaneous clean branch and a mean paired loss greater than 1 point in rounds 9–12 to define disruption. The main follow-up threshold is a gap at most 0.5 points for three contiguous post-stop rounds. Latent gaps are absolute and ACC gaps are clipped below at zero. A continuous-attack branch can cross a threshold while exposure remains active. Its post-stop recovery fields stay empty.
- The analytic cost outputs use a two-layer GCN64 proxy with explicit workload and throughput assumptions. The 4 MB per message topology model covers communication only. Fields for actual epochs, GPU seconds, and peak GPU bytes are empty.
- `provenance.json` records SHA-256 fingerprints for the archived author scripts and input configuration used in this port. The historical workbooks, presentation figures, and screenshots of other papers remain in the author archive. The portable code exports the underlying numerical statistics as CSV and NPZ.

## Check a reproduction against the original arrays

A copy of the author's original NPZ data directory can be used for an optional comparison. Use the relevant archive directory for the selected study.

```sh
python -m legacy verify --run runs/numerical20 --reference /path/to/original/data
```

The comparison reads existing files and checks exact trajectory keys and confusion counts. It requires absolute errors below 1e-10 for the latent scores, metrics, and class-level NLL sums.

The separate [historical WL pilot](wl_pilot/README.md) provides the original exploratory learning loop and its archived sensitivity analyses.
