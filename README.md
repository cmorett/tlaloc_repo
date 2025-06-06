# tlaloc_repo

A GNN, MPC-gradient based, optimizer for EPANET water systems

## Training

The repository provides a simple training script `scripts/train_gnn.py` which
expects feature and label data saved in the `data/` directory as NumPy arrays.
Two dataset formats are
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

I'll complete the README Later
