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

Training performs node-wise regression with mean squared error.  The script
automatically checks that the provided features include pump control inputs by
loading the EPANET network (``--inp-path``).  If the dimension does not match
``4 + num_pumps`` an error is raised. Use ``--output-dim`` to specify how many
continuous targets are predicted per node.

Example usage:

```bash
python scripts/train_gnn.py --x-path data/X_train.npy --y-path data/Y_train.npy \
    --x-val-path data/X_val.npy --y-val-path data/Y_val.npy \
    --epochs 100 --batch-size 16 --hidden-dim 32 --num-layers 3 \
    --normalize --early-stop-patience 5 --edge-index-path data/edge_index.npy \
    --inp-path CTown.inp
```

The trained model now supports validation loss tracking, early stopping and
optional feature normalization.  Each run is stored with a unique timestamp to
avoid overwriting previous checkpoints.  Normalization statistics are saved
alongside the weights and automatically applied by the inference scripts.

To achieve good predictive accuracy the surrogate should be trained on a large
collection of EPANET simulations.  The data generation script accepts a
``--num-scenarios`` argument which controls how many randomized 24‑h simulations
are performed:

To achieve good predictive accuracy the surrogate should be trained on a large
collection of EPANET simulations.  The data generation script automatically
splits each run into training, validation and test sets (``X_train.npy``,
``X_val.npy``, ``X_test.npy`` and corresponding ``Y`` arrays).  The
``--output-dir`` option specifies where these arrays are stored.  A typical call
looks like:

```bash
python scripts/data_generation.py --num-scenarios 2000 --output-dir data/ --seed 42
```

To create sequence datasets for the recurrent surrogate specify ``--sequence-length``:

```bash
python scripts/data_generation.py --num-scenarios 200 --sequence-length 24 --output-dir data/
```

The training script automatically detects such sequence files and will use the recurrent
model. Adjust the recurrent hidden size via ``--rnn-hidden-dim`` if desired.

Validate the resulting model with `scripts/experiments_validation.py` before
running the MPC controller.  The validation script executes a 24‑hour
simulation with EPANET feedback applied every hour (``--feedback-interval`` is
``1`` by default) which keeps predictions from the surrogate model from
diverging over long horizons.

## Running MPC control

Once the surrogate model is trained you can run gradient-based MPC using
`scripts/mpc_control.py`:

```bash
python scripts/mpc_control.py --horizon 6 --iterations 50 --feedback-interval 24
```

By default the controller loads the most recent ``.pth`` file found in the
``models`` directory so retraining will automatically use the newest weights.

This executes a 24‑hour closed loop simulation where pump actions are optimized
at each hour.  EPANET is only called every 24 hours (controlled by
``--feedback-interval``) and all intermediate updates rely on the GNN surrogate
running entirely on a CUDA device.  Results are written to
`data/mpc_history.csv`.

**Important:** the surrogate must be trained on datasets that include pump
control inputs (the additional features appended by `scripts/data_generation.py`).
If a model trained with only four features (demand, pressure, chlorine and
elevation) is loaded, `mpc_control.py` will exit early because pump actions would
have no effect on the predictions.
