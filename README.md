# Run VCIA

## Set up the environment

Download and extract the repository, then open a terminal in the extracted directory. Use Python 3.12.

```sh
python -m venv .venv
```

Activate the environment using the command for your shell.

| Shell | Command |
| --- | --- |
| Linux or macOS | `source .venv/bin/activate` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |

```sh
python -m pip install -c experiments/requirements.txt ".[test]"
python -m pcoc.demo
python -m pytest -q
```

## Run a quick graph experiment

Download [PROTEINS.zip](https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip) into `data/` and leave it zipped. Run the command below from the repository root. Use a new output directory each time.

```sh
python experiments/run_closed_ablation.py --smoke --data data --output runs/closed-ablation-smoke
```

## Run the paper experiments

Follow the [input and experiment commands](REPRODUCTION.md#inputs) for dataset downloads, closed-loop ablations, snapshot replay, audit-budget checks, timing, and figure rendering.

### Jetson Linux

Run the environment setup on the Jetson device, then follow the [Jetson run instructions](REPRODUCTION.md#jetson-run-environment).
