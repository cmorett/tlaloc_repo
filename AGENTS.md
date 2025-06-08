# AGENTS Guide for tlaloc_repo

This repository implements a graph neural network (GNN) surrogate and gradient-based model predictive control (MPC) 
for EPANET water distribution models. The main example network is `CTown.inp`.

## Project Layout

- `README.md` – high level usage instructions.
- `CTown.inp` – EPANET input file for the C‑Town network.
- `scripts/`
  - `data_generation.py` – create randomized simulation scenarios and produce training datasets.
  - `train_gnn.py` – train a two-layer GCN (`SimpleGCN`) on generated data.
  - `mpc_control.py` – run gradient-based MPC using the trained surrogate.
  - `experiments_validation.py` – validate the surrogate, compare baselines and aggregate results.
- `models/` – storage location for trained weights (`gnn_surrogate.pth`).
- `pytorchcheck.py` – quick script verifying that PyTorch Geometric runs on the configured GPU.
- `data/` – ignored by git; used for generated datasets and simulation logs.

## Architecture Overview

1. **Data Generation** – `scripts/data_generation.py` executes multiple EPANET simulations with randomized pump 
    states and scaled base demands. It writes node feature matrices, labels for the next hour pressure and chlorine, 
    and the graph `edge_index` to the `data/` directory.

2. **Surrogate Training** – `scripts/train_gnn.py` loads the generated data, builds `torch_geometric.data.Data` objects 
    and trains a simple two-layer GCN. NaNs in the features are   replaced with zero to avoid invalid losses. Gradients 
    are clipped to keep the training stable.

3. **MPC Controller** – `scripts/mpc_control.py` loads the trained surrogate (`GNNSurrogate`) and repeatedly optimizes 
    pump speeds via gradient descent. The controller can either propagate the network state entirely through the surrogate 
    or periodically synchronize with EPANET for ground truth. Simulation history is written to `data/mpc_history.csv`.

4. **Experiment Validation** – `scripts/experiments_validation.py` evaluates the surrogate on prerecorded EPANET scenarios 
    and compares the MPC controller against two baselines. Results are aggregated into CSV files and simple plots under `data/`.

The repository assumes a working Python environment with PyTorch, PyTorch Geometric and `wntr` installed. A small virtual environment with these dependencies is provided in `.venv/`.

## Testing Protocols

This project does not ship automated unit tests. The recommended workflow is:

1. Verify GPU and PyG installation using `python pytorchcheck.py`.
2. Generate a small dataset with
   ```bash
   python scripts/data_generation.py --num-scenarios 10 --output-dir data/
   ```
3. Train the surrogate:
    ```bash
    python scripts/train_gnn.py --x-path data/X_train.npy --y-path data/Y_train.npy --edge-index-path data/edge_index.npy --inp-path CTown.inp
    ```
4. Run the experiment suite which includes a surrogate validation step:
   ```bash
   python scripts/experiments_validation.py --model models/gnn_surrogate.pth --inp CTown.inp
   ```
5. Optionally launch MPC control directly using `scripts/mpc_control.py`.

When adding new features or bug fixes, please create unit tests using `pytest` inside a `tests/` directory and run `pytest` before committing.

## Additional Notes for Codex

- NEVER assume that other scripts work exactly as intended. Always check that scripts you may need in fact do what they say they do as there may be errors.
- Negative pressures or `NaN` values for energy are physically unrealistic and should be avoided. Ensure that simulated pressures remain non‑negative and that computed pump energy never becomes `NaN`.
- Paths should be resolved relative to the repository root so scripts work when launched from any location.
- Generated data and results should remain inside the `data/` folder.