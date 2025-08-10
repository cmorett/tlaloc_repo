# tlaloc_repo


A GNN, MPC-gradient based, optimizer for EPANET water systems

## Installation

Create a Python virtual environment and install the required packages from
`requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs PyTorch, PyTorch Geometric, `numpy`, `scikit-learn`, `wntr`,
`pandas`, `matplotlib`, `networkx` and `epyt` which are required for the
training and control scripts.

## Quick GPU check

Verify that PyTorch and PyTorch Geometric are installed correctly by running

```bash
python pytorchcheck.py
```
The script moves a small tensor through a GCN layer on the GPU and prints the
output shape. If this command fails, ensure the CUDA drivers and dependencies
are installed before continuing.

## Training

The repository provides a simple training script `scripts/train_gnn.py` which
expects feature and label data saved in the `data/` directory as NumPy arrays.
Each node feature vector has the layout
``[base_demand, pressure, chlorine, elevation, pump_1, ..., pump_N]`` where each
``pump_i`` denotes the fractional pump speed in ``[0, 1]`` rather than a binary
on/off flag. Reservoir nodes use their constant hydraulic head in the
``pressure`` slot so the model is given the correct supply level. The helper
script `scripts/data_generation.py`
generates these arrays as well as the graph ``edge_index``.  Two dataset formats
are
supported:

1. **Dictionary format** – each entry of ``X`` is a dictionary containing the
   graph ``edge_index`` and a ``node_features`` array.
2. **Matrix format** – ``X`` is an array of node feature matrices and a shared
   ``edge_index`` array is stored separately.

All plots generated during training, validation and MPC experiments are
saved under the top-level ``plots/`` directory.  The scripts automatically
create this folder if it does not yet exist. After each training run
``train_gnn.py`` saves two scatter plots comparing model predictions to
EPANET results: ``pred_vs_actual_pressure_<run>.png`` and
``pred_vs_actual_chlorine_<run>.png``. Reservoirs and tanks are excluded from
these plots since their pressures are fixed. ``error_histograms_<run>.png``
contains histograms and box plots of the prediction errors and the CSV
``logs/accuracy_<run>.csv`` records MAE, RMSE, MAPE and maximum error for
pressure and chlorine. Reservoir and tank nodes are excluded from these metrics
so outliers from fixed heads do not skew the results. Finally ``correlation_heatmap_<run>.png``
visualises pairwise correlations between the unnormalised training features.
When normalization is enabled (the default) the test data is scaled using the
training statistics. During evaluation both predictions **and** the
corresponding ground truth labels are transformed back to physical units before
plotting.
When sequence models are used a component-wise loss curve
``loss_components_<run>.png`` is stored alongside ``loss_curve_<run>.png``.
For sequence datasets a time-series example ``time_series_example_<run>.png``
plots predicted and actual pressure and chlorine for one node across all steps.

The script automatically detects which format is provided and loads the data
accordingly. When using the matrix format, supply the path to the shared
``edge_index`` file via ``--edge-index-path`` (defaults to ``data/edge_index.npy``).
Edge attributes describing pipe length, diameter and roughness are stored in
``edge_attr.npy`` by ``scripts/data_generation.py``. ``train_gnn.py`` loads this
file by default via ``--edge-attr-path``. Pump curve coefficients are saved as
``pump_coeffs.npy`` and included in a dedicated pump curve loss during training
(``--pump-loss``) with weight ``--w_pump``.

Training performs node-wise regression and by default optimizes the mean
absolute error (MAE).  Specify ``--loss-fn`` to switch between MAE (``mae``),
mean squared error (``mse``) or Huber loss (``huber``).  The script
automatically checks that the provided features include pump control inputs by
loading the EPANET network (``--inp-path``).  If the dimension does not match
``4 + num_pumps`` an error is raised. Use ``--output-dim`` to specify how many
continuous targets are predicted per node.

When datasets were generated with ``scripts/data_generation.py`` after this
update, each time step also stores pipe flow rates. ``train_gnn.py``
automatically detects such multi-task arrays and switches to a
``MultiTaskGNNSurrogate`` model which optimizes node and edge losses.

The GNN architecture has been refactored to support **heterogeneous graphs**.
Node embeddings are now conditioned on the component type (junction, tank,
pump or valve) while edges differentiate pipes, pumps and valves.  The helper
scripts automatically compute these type indices from the EPANET network and
store them in ``edge_type.npy`` together with numerical edge attributes in
``edge_attr.npy``. Training and MPC control handle these additional attributes
transparently.

Optionally, the same convolution layer can be reused for all message passing
steps by passing ``--share-weights`` to ``train_gnn.py``. This reduces the
number of parameters and can speed up optimisation.

The surrogate now applies temporal self-attention after the LSTM to re-weight
each node's history and updates tank pressures explicitly from predicted
flows. Before predicting a sequence the current tank volumes must be passed to
``model.reset_tank_levels`` which the MPC script handles automatically.

Example usage:

```bash
python scripts/train_gnn.py \
    --x-path data/X_train.npy --y-path data/Y_train.npy \
    --x-val-path data/X_val.npy --y-val-path data/Y_val.npy \
    --edge-index-path data/edge_index.npy --edge-attr-path data/edge_attr.npy \
    --inp-path CTown.inp \
    --epochs 100 --batch-size 32 --hidden-dim 64 --num-layers 4 \
    --workers 8 \
    --dropout 0.1 --residual --early-stop-patience 10 \
    --weight-decay 1e-5
```
If training is interrupted with ``Ctrl+C`` a final checkpoint containing the
model, optimizer, scheduler state and epoch is saved so progress is not lost.
To continue a previous run pass the checkpoint path via ``--resume``.  All
standard arguments still need to be supplied:

```bash
python scripts/train_gnn.py \
    --resume models/gnn_surrogate_20240101.pth \
    --x-path data/X_train.npy --y-path data/Y_train.npy \
    --edge-index-path data/edge_index.npy --edge-attr-path data/edge_attr.npy \
    --inp-path CTown.inp --epochs 100 --batch-size 32
```
Training will continue from the stored epoch and future checkpoints are written
back to the same file so the run can be resumed repeatedly.
Add ``--loss-fn huber`` to the command above to train with Huber loss or
``--loss-fn mse`` to minimize mean squared error instead.
Pass ``--no-progress`` to disable the training progress bars or
``--progress`` to re-enable them if disabled.
Pressure–headloss consistency is now enforced by default with a weight of ``0.25``.
Pass ``--no-pressure-loss`` if this coupling should be disabled.  To remove the
mass balance penalty (still weighted ``1.0`` by default) use ``--no-physics-loss``.
The surrogate clamps predicted pressures and chlorine concentrations to
non-negative values and applies L2 regularization controlled by
``--weight-decay`` (default ``1e-5``) to avoid degenerate solutions.
Even when the physics-based loss is disabled with ``--no-physics-loss``
the script prints the current epoch number so progress remains visible in
the terminal.

For large graphs you can reduce memory usage by training on subgraphs.
Passing ``--cluster-batch-size <N>`` partitions the network into clusters of
approximately ``N`` nodes and iterates over those clusters instead of loading
the full graph each time.  Use ``--neighbor-sampling`` to sample ``N`` target
nodes and their neighbors on-the-fly instead of deterministic clusters.
During inference on very large networks the same clusters can be evaluated
sequentially to keep memory usage low.

Mixed precision training is enabled by default via the ``--amp`` flag which
wraps model evaluations in ``torch.cuda.amp.autocast`` and scales the loss for
improved GPU throughput. Use ``--no-amp`` to run exclusively in full precision.

A physics-informed mass balance penalty is applied by default to encourage
conservation of predicted flows.  Because each pipe appears twice in the graph
(forward and reverse), the loss divides the imbalance by two so that equal and
opposite flows cancel correctly.  A symmetry term further penalises differences
between forward and reverse edges so paired flows remain equal and opposite.
Reservoirs and tanks are excluded from the mass balance calculation while tank
pressures are no longer part of the direct MSE loss.  Disable the physics terms
with ``--no-physics-loss`` if necessary. ``--pressure_loss`` is enabled by
default to enforce pressure–headloss consistency via the Hazen--Williams
equation.  The mass and edge penalties keep a default weight of ``1.0`` while
the headloss term uses ``0.25``. The relative importance can still be tuned via
``--w_mass`` and ``--w_head`` along with the ``--w_edge`` coefficient
controlling the flow loss.

The trained model now supports validation loss tracking and early stopping.
Normalization is applied automatically so the ``--normalize`` flag is optional.
Each run is stored with a unique timestamp to avoid overwriting previous
checkpoints.  Normalization statistics are saved alongside the weights and
automatically applied by the inference scripts.
You may cancel training early with ``Ctrl+C``.  The loop exits gracefully,
keeping the best model on disk and producing the usual plots.
When interrupted, the current epoch finishes before evaluation so scatter plots
are still generated from the test set.
Data is loaded in parallel using multiple worker processes; pass ``--workers``
to adjust the number (default ``5``).

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
python scripts/data_generation.py \
    --num-scenarios 2000 --output-dir data/ --seed 42 \
    --extreme-event-prob 0.2
```
The generation step writes ``edge_index.npy``, ``edge_attr.npy``, ``edge_type.npy`` and
``pump_coeffs.npy`` alongside the feature and label arrays. It utilizes all available CPU cores by default. The value
``2000`` matches the new default of ``--num-scenarios``. Use
``--num-workers`` to override the number of parallel workers if needed.
Pass ``--show-progress`` to display a live progress bar during simulation
when ``tqdm`` is installed.
If a particular random configuration causes EPANET to fail to produce results,
the script now skips it after a few retries so the actual number of generated
scenarios may be slightly smaller than requested.

Set ``--extreme-event-prob`` to inject rare scenarios such as fire flows,
pump failures or source quality changes.  Scenario labels are stored alongside
the sequence arrays when ``--sequence-length`` is greater than one.
Initial tank levels are now drawn from a Gaussian around the values in
``CTown.inp`` so each scenario begins with slightly different volumes.
Chlorine decay is enabled in the example network via a global bulk reaction
coefficient of ``-0.05`` 1/h which EPANET applies during water quality
simulations.
Pipe roughness coefficients are left unchanged; only demand multipliers and
pump schedules vary between scenarios. Pump speeds now follow a continuous
randomization strategy: each pump starts from a value in ``[0.3, 0.9]`` and is
perturbed by small, temporally correlated Gaussian noise each hour. A short
dwell time around the ``0.05`` threshold prevents rapid cycling and a safeguard
ensures at least one pump remains active at any time.

After scenario generation finishes a plot ``dataset_distributions_<timestamp>.png``
is created under ``plots/`` summarising the sampled demand multipliers and pump
speed settings.  Checking this figure helps ensure the dataset spans diverse
operating conditions before proceeding to training.

To create sequence datasets for the recurrent surrogate specify ``--sequence-length``:

```bash
python scripts/data_generation.py \
    --num-scenarios 200 --sequence-length 24 --output-dir data/ \
    --seed 123
```
This will also generate ``edge_index.npy``, ``edge_attr.npy`` and ``edge_type.npy``
along with ``scenario_train.npy`` etc. recording the type of each scenario.
Scenarios that do not contain at least ``sequence_length + 1`` time steps are
skipped with a warning so the actual dataset may be smaller than requested.

The training script automatically detects such sequence files and will use the recurrent
model. Adjust the recurrent hidden size via ``--rnn-hidden-dim`` if desired.

Validate the resulting model with `scripts/experiments_validation.py` before
running the MPC controller.  The validation script executes a 24‑hour
simulation with EPANET feedback applied every hour (``--feedback-interval`` is
``1`` by default) which keeps predictions from the surrogate model from
diverging over long horizons.  It now also reports mean absolute error (MAE)
 and the maximum absolute error for pressure and chlorine.  Predictions and
 ground truth are denormalized and chlorine values are exponentiated so the
 resulting errors are reported in physical units.  Pressures below 5 m are
 clipped to this lower bound during validation so metrics match the training
distribution.  All metrics are written to
``logs/surrogate_metrics.json`` for reproducibility.

If the dimension of ``edge_attr.npy`` does not match the value stored in the
surrogate checkpoint, ``validate_surrogate`` now raises a ``ValueError``.
Regenerate the dataset or retrain the surrogate when this occurs.

Typical validation command:

```bash
python scripts/experiments_validation.py \
    --model models/gnn_surrogate.pth --inp CTown.inp \
    --horizon 6 --iterations 50 --feedback-interval 1 \
    --run-name baseline
```

## Running MPC control

Once the surrogate model is trained you can run gradient-based MPC using
`scripts/mpc_control.py`:

```bash
python scripts/mpc_control.py \
    --horizon 6 --iterations 50 --feedback-interval 24 \
    --Pmin 20.0 --Cmin 0.2 --energy-scale 1e-9 \
    --w_p 100 --w_c 100 --w_e 1.0 --profile
```

Pass ``--profile`` to print the runtime of each MPC optimisation step. The
controller now builds node features on the GPU to minimise Python overhead.
The surrogate model is compiled with TorchScript during loading for faster
inference.  Use ``--no-jit`` to disable this.  ``propagate_with_surrogate`` can
also accept lists of pressure/chlorine dictionaries to evaluate multiple
scenarios in parallel.

Use ``--skip-normalization`` to disable input normalization and feed raw
features into the surrogate for ablation studies. Outputs are always
de-normalized back to physical units.

``mpc_control.py`` exposes weights on pressure, chlorine and energy terms as
``--w_p``, ``--w_c`` and ``--w_e`` respectively.  The default configuration
scales pump energy from Joules to megawatt-hours via ``--energy-scale 1e-9``
and sets ``w_p``/``w_c`` to 100 so constraint violations dominate energy
trade-offs.  Passing ``--use-barrier`` switches to an exponential-style
penalty which grows superlinearly, enabling an alternative strategy where
energy remains in Joules but the violation weights are increased by several
orders of magnitude.

Pump energy usage in the MPC cost function is computed from predicted flows and
head gains using the EPANET power equations. This removes the need for a
dedicated energy output and ties the optimisation to physical principles.

By default the controller loads the most recent ``.pth`` file found in the
``models`` directory so retraining will automatically use the newest weights.

This executes a 24‑hour closed loop simulation where fractional pump speeds are
optimized at each hour.  EPANET is only called every 24 hours (controlled by
``--feedback-interval``) and all intermediate updates rely on the GNN surrogate
running entirely on a CUDA device.  Results are written to
`data/mpc_history.csv`.  A summary listing constraint violations and total
energy consumption (in Joules) is printed at the end of the run and saved to
``logs/mpc_summary.json``.
The controller updates node demands each hour using the diurnal patterns
specified in ``CTown.inp`` so the surrogate remains aligned with its training
distribution.

**Important:** the surrogate must be trained on datasets that include pump
control inputs (the additional features appended by `scripts/data_generation.py`).
If a model trained with only four features (demand, pressure, chlorine and
elevation) is loaded, `mpc_control.py` will exit early because pump actions would
have no effect on the predictions.
`simulate_closed_loop` now performs the same check and raises a `ValueError`
when the loaded model does not include pump controls so that experiment scripts
fail fast instead of silently optimising with zero gradients.
It also verifies that the provided edge attributes match the model's expected
dimension and aborts with an error when they differ.

## Reporting Key Performance Metrics

The module `scripts/metrics.py` provides helper functions to compute the most
common evaluation metrics for surrogate accuracy, MPC control behaviour and
runtime performance.  Each function returns a formatted ``pandas.DataFrame`` for
easy printing or saving.

Example usage:

```python
from metrics import (
    accuracy_metrics,
    control_metrics,
    computational_metrics,
    export_table,
)

# arrays of ground truth and predictions
acc_df = accuracy_metrics(true_p, pred_p, true_c, pred_c)
control_df = control_metrics(min_p, min_c, energy, p_min=20.0, c_min=0.2)
comp_df = computational_metrics(inference_times, optimisation_times)

# export to CSV
export_table(acc_df, "logs/accuracy.csv")
export_table(control_df, "logs/control.csv")
export_table(comp_df, "logs/computational.csv")
```

Tables use clear labels and units so results can be understood at a glance.  The
same metrics can also be exported to Excel or JSON by passing a path ending in
``.xlsx`` or ``.json``.

## Visualization Suite

Plotting utilities are located within the training and experiment scripts
themselves.  They generate prediction scatter plots, MPC time series,
energy–pressure trade-off charts and convergence curves.  All images are
written to the `plots/` directory so they can be included in reports or
presentations easily.
