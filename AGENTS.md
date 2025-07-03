# AGENTS Guide for tlaloc_repo

This repository implements a graph neural network (GNN) surrogate and gradient-based model predictive control (MPC) 
for EPANET water distribution models. The main example network is `CTown.inp`.

## Project Layout

- `README.md` – high level usage instructions.
- `CTown.inp` – EPANET input file for the C‑Town network.
- `scripts/`
  - `data_generation.py` – create randomized simulation scenarios and produce training datasets.
  - `train_gnn.py` – train a graph neural network surrogate on generated data.
  - `mpc_control.py` – run gradient-based MPC using the trained surrogate.
  - `experiments_validation.py` – validate the surrogate, compare baselines and aggregate results.
  - `metrics.py` – report surrogate accuracy, MPC control and runtime metrics.
- `models/` – storage location for trained weights (`gnn_surrogate.pth`).
- `pytorchcheck.py` – quick script verifying that PyTorch Geometric runs on the configured GPU.
- `data/` – ignored by git; used for generated datasets and temporary simulation outputs.
- `plots/` – ignored by git; stores all figures produced during training, validation and MPC runs.
- `logs/` – ignored by git; JSON summaries such as `surrogate_metrics.json` and `mpc_summary.json`.
- `tests/` – pytest suite:
  - `test_clip.py` – pressures and chlorine are clipped to non‑negative values.
  - `test_early_stop.py` – early stopping logic.
  - `test_energy.py` – pump energy calculations never produce NaNs.
  - `test_extreme_events.py` – scenario generation labels extreme events.
  - `test_hydroconv.py` – `HydroConv` layer conserves mass.
  - `test_load_surrogate.py` – loading surrogate checkpoints (key renaming, NaN detection, newest model search, multi‑task norms).
  - `test_mass_balance.py` – mass balance loss computation.
  - `test_mpc_input_check.py` – MPC simulation requires pump control features.
  - `test_nan_check.py` – training aborts when NaNs are present in the input.
  - `test_normalization.py` – statistics computation and normalization behaviour.
  - `test_normalized_negative.py` – training after normalization when values become negative.
  - `test_pump_controls.py` – at least one pump is active each hour in a generated scenario.
  - `test_recurrent_forward.py` – forward pass shapes for recurrent and multi‑task models.
  - `test_scatter.py` – saving scatter plots.
  - `test_validate_surrogate.py` – validating a surrogate with provided results.
  - `test_headloss_loss.py` – pressure‑headloss consistency penalty.
  - `test_metrics.py` – metric helper functions generate expected tables.
  - `test_visualizations.py` – plotting utilities create labelled images.

## Architecture Overview

1. **Data Generation** – `scripts/data_generation.py` executes multiple EPANET simulations with randomized pump states and scaled base demands. It writes node feature matrices, labels for the next hour pressure and chlorine, and the graph `edge_index` to the `data/` directory.
2. **Surrogate Training** – `scripts/train_gnn.py` loads the generated data and trains a configurable GNN encoder. The model supports heterogeneous node and edge types, optional attention and residual connections. NaNs in the features are replaced with zero to avoid invalid losses. Gradients are clipped to keep the training stable. Scatter plots comparing predictions to EPANET are saved under `plots/`.
3. **MPC Controller** – `scripts/mpc_control.py` loads the trained surrogate (`GNNSurrogate`) and repeatedly optimizes pump speeds via gradient descent. The controller can either propagate the network state entirely through the surrogate or periodically synchronize with EPANET for ground truth. Simulation history is written to `data/mpc_history.csv` and a summary JSON file to `logs/`.
4. **Experiment Validation** – `scripts/experiments_validation.py` evaluates the surrogate on prerecorded EPANET scenarios and compares the MPC controller against two baselines. Results are aggregated into CSV files under `data/` and plots under `plots/`. Validation metrics are stored in `logs/surrogate_metrics.json`.
5 **Metric Reporting and Visualization** – `scripts/metrics.py` summarises accuracy, control quality and computational overhead while `scripts/visualizations.py` creates scatter plots, time‑series and convergence curves for reports.

The repository assumes a working Python environment with PyTorch, PyTorch Geometric and `wntr` installed. A small virtual environment with these dependencies is provided in `.venv/`.

## Testing Protocols

Basic unit tests live in the `tests/` directory. Run them with `pytest`. The recommended workflow is:

1. Verify GPU and PyG installation using `python pytorchcheck.py`.
2. Run the unit tests:
   ```bash
   pytest
   ```
3. Generate a small dataset with
   ```bash
   python scripts/data_generation.py --num-scenarios 10 --output-dir data/
   ```
4. Train the surrogate:
    ```bash
    python scripts/train_gnn.py --x-path data/X_train.npy --y-path data/Y_train.npy --edge-index-path data/edge_index.npy --inp-path CTown.inp
    ```
5. Run the experiment suite which includes a surrogate validation step:
   ```bash
   python scripts/experiments_validation.py --model models/gnn_surrogate.pth --inp CTown.inp
   ```
6. Optionally launch MPC control directly using `scripts/mpc_control.py`.

When adding new features or bug fixes, please create unit tests using `pytest` inside a `tests/` directory and run `pytest` before committing.

## Additional Notes for Codex

- NEVER assume that other scripts work exactly as intended. Always check that scripts you may need in fact do what they say they do as there may be errors.
- Negative pressures or `NaN` values for energy are physically unrealistic and should be avoided. Ensure that simulated pressures remain non‑negative and that computed pump energy never becomes `NaN`.
- Paths should be resolved relative to the repository root so scripts work when launched from any location.
- Generated data and results should remain inside the `data/` folder, if they are plots they go to `plots/`.
- Any changes that you do that influence the way the user is supposed to interact with the scripts should be accompanied with a corresponding change to the `README.md` file in which you declare how the new change is supposed to be used. If the change involves the creation of new files those changes should be present in this `AGENTS.md` file.
- Also aim to consistently have the optimal example prompt in the `README.md` file for optimal usage. 