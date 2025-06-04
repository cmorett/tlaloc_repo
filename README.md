# tlaloc_repo

A GNN, MPC-gradient based, optimizer for EPANET water systems

## Training

The repository provides a simple training script `scripts/train_gnn.py` which expects
feature and label data saved as NumPy arrays. Each feature entry should be a
Python dictionary containing `edge_index` and `node_features` arrays. Because the
arrays are stored with Python objects, the script loads them using
`allow_pickle=True`.

Example usage:

```bash
python scripts/train_gnn.py --x-path X_train.npy --y-path y_train.npy \
    --epochs 100 --batch-size 16 --hidden-dim 32 --num-classes 2
```

The trained model weights are saved to `model.pt` by default.
