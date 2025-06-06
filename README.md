# tlaloc_repo

A GNN, MPC-gradient based, optimizer for EPANET water systems

## Training

The repository provides a simple training script `scripts/train_gnn.py` which
expects feature and label data saved in the `data/` directory as NumPy arrays.
Each node feature vector has the layout
``[base_demand, pressure, chlorine, elevation, pump_1, ..., pump_N]`` where the
additional elements represent the speeds of all pumps in the network.  The
helper script `scripts/data_generation.py` generates these arrays as well as the
graph ``edge_index``.  Two dataset formats are
supported:

1. **Dictionary format** – each entry of ``X`` is a dictionary containing the
   graph ``edge_index`` and a ``node_features`` array.
2. **Matrix format** – ``X`` is an array of node feature matrices and a shared
   ``edge_index`` array is stored separately.

The script automatically detects which format is provided and loads the data
accordingly. When using the matrix format, supply the path to the shared
``edge_index`` file via ``--edge-index-path`` (defaults to ``data/edge_index.npy``).

Training performs node-wise regression with mean squared error. Use
``--output-dim`` to specify how many continuous targets are predicted per node.

Example usage:

```bash
python scripts/train_gnn.py --x-path data/X_train.npy --y-path data/Y_train.npy \
    --epochs 100 --batch-size 16 --hidden-dim 32 --output-dim 2 \
    --edge-index-path data/edge_index.npy
```

The trained model weights are saved to `models/gnn_surrogate.pth` by default.

## Running MPC control

Once the surrogate model is trained you can run gradient-based MPC using
`scripts/mpc_control.py`:

```bash
python scripts/mpc_control.py --horizon 6 --iterations 50
```

This executes a 24‑hour closed loop simulation where pump actions are
optimized at each hour.  Results are written to `data/mpc_history.csv`.

**Important:** the surrogate must be trained on datasets that include pump
control inputs (the additional features appended by `scripts/data_generation.py`).
If a model trained with only four features (demand, pressure, chlorine and
elevation) is loaded, `mpc_control.py` will exit early because pump actions would
have no effect on the predictions.

I'll complete the README Later
