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

## Reproducibility

All entry scripts expose a `--seed` flag that seeds Python, NumPy and PyTorch
for repeatable runs. Passing `--deterministic` additionally configures
`CUBLAS_WORKSPACE_CONFIG` and enables `torch.use_deterministic_algorithms`, which
may reduce performance but guarantees deterministic CUDA kernels. Each run saves
its arguments, normalization statistics checksum, model hyperparameters and the
current Git commit to `logs/config.yaml` for provenance.

## Training

The repository provides a simple training script `scripts/train_gnn.py` which
expects feature and label data saved in the `data/` directory as NumPy arrays.
Each node feature vector has the layout
``[base_demand, pressure, elevation, pump_1, …]`` where each
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
``train_gnn.py`` saves a scatter plot comparing predicted and actual pressure
(``pred_vs_actual_pressure_<run>.png``). Reservoirs and tanks are excluded from
these plots since their pressures are fixed. ``error_histograms_<run>.png``
contains histograms and box plots of the prediction errors and the CSV
``logs/accuracy_<run>.csv`` records MAE, RMSE, MAPE and maximum error for
pressure. Reservoir and tank nodes are excluded from these metrics so outliers
from fixed heads do not skew the results. Metrics are accumulated using running
statistics so the full prediction arrays are never stored in memory. To limit the
number of predictions retained for plotting, pass ``--eval-sample N`` (default
``1000``) which keeps only the first ``N`` predictions for the scatter and error
plots. Set ``N=0`` to skip these figures entirely. Finally
``correlation_heatmap_<run>.png`` visualises pairwise correlations between the
unnormalised training features. When normalization is enabled (the default) the
test data is scaled using the training statistics. During evaluation both
predictions **and** the corresponding ground truth labels are transformed back to
physical units before plotting.
The training script saves feature and target means and standard deviations on
the model.  ``mpc_control.py`` loads these tensors and checks their shapes so
inference uses the exact same statistics.  This normalization contract prevents
silent scaling mismatches; pass ``--skip-normalization`` to operate entirely on
raw values. Pass ``--per-node-norm`` to compute statistics for each node index
separately which removes large baseline offsets and often reduces pressure MAE.
When sequence models are used a component-wise loss curve
``loss_components_<run>.png`` is stored alongside ``loss_curve_<run>.png`` and
the per-component pressure and flow losses are recorded each epoch in
``training_<run>.log`` as well as TensorBoard summaries.
For sequence datasets a time-series example ``time_series_example_<run>.png``
plots predicted and actual pressure for one node across all steps.

The script automatically detects which format is provided and loads the data
accordingly. When using the matrix format, supply the path to the shared
``edge_index`` file via ``--edge-index-path`` (defaults to ``data/edge_index.npy``).
Edge attributes describing pipe length, diameter and roughness are stored in
``edge_attr.npy`` by ``scripts/data_generation.py``. ``train_gnn.py`` loads this
file by default via ``--edge-attr-path``. Pump curve coefficients are saved as
``pump_coeffs.npy`` and included in a dedicated pump curve loss during training
(``--pump-loss`` or ``--pump_loss``) with weight ``--w_pump``.
Optional physics losses in ``models/loss_utils.py`` further regularise
training. ``compute_mass_balance_loss`` penalises node flow imbalance,
``pressure_headloss_consistency_loss`` enforces Hazen–Williams head losses and
``pump_curve_loss`` discourages infeasible pump operating points. Combine these
terms with data losses to keep predictions physically plausible.

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
    --epochs 100 --batch-size 32 --hidden-dim 128 --num-layers 4 \
    --lstm-hidden 64 --workers 8 --eval-sample 1000 \
    --dropout 0.1 --residual --early-stop-patience 10 \
    --weight-decay 1e-5 --w-press 5.0 --w-flow 3.0 \
    --checkpoint
```

GNN depth and width are controlled via ``--num-layers`` (choose from {4,6,8}) and ``--hidden-dim`` ({128,256}).
Use ``--residual`` to enable skip connections and ``--use-attention`` for graph attention on node updates.
The LSTM hidden size can be set with ``--lstm-hidden`` (64 or 128).
When training larger models that exceed GPU memory, pass ``--checkpoint`` to
enable gradient checkpointing which recomputes intermediate activations during
backpropagation to lower peak memory usage at the cost of additional compute.
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
Pressure–headloss consistency is now enforced by default with a weight of ``1.0``.
Pass ``--no-pressure-loss`` if this coupling should be disabled.  To remove the
mass balance penalty (now weighted ``2.0`` by default) use ``--no-physics-loss``.
The surrogate clamps predicted pressures to non-negative values and applies L2
regularization controlled by
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
equation.  The mass penalty uses a default weight of ``2.0`` while
the headloss term uses ``1.0``. Node pressure and flow terms use
weights ``--w-press`` (default ``5.0``) and ``--w-flow`` (``3.0``).
The relative importance can still be tuned via these flags together with
``--w_mass`` and ``--w_head``.  To keep the physics penalties on a comparable
scale the script estimates baseline magnitudes for the mass, headloss and pump
curve terms during the first pass over the training data. These values are used
to normalise the respective losses before applying the user-specified weights.
The automatically detected scales can be overridden via ``--mass-scale``,
``--head-scale`` and ``--pump-scale`` if manual tuning or logging is desired.
Scales below ``1e-3`` are automatically clamped to prevent excessively large
physics penalties.
Training logs also report the average mass imbalance per batch and the
percentage of edges with inconsistent headloss signs.

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
    --extreme-rate 0.03 --pump-outage-rate 0.1 --local-surge-rate 0.1 \
    --demand-scale-range 0.8 1.2
```
Append `--deterministic` to enforce deterministic CUDA kernels.
The generation step writes ``edge_index.npy``, ``edge_attr.npy``, ``edge_type.npy`` and
``pump_coeffs.npy`` alongside the feature and label arrays. It utilizes all available CPU cores by default. The value
``2000`` matches the new default of ``--num-scenarios``. Use
``--num-workers`` to override the number of parallel workers if needed.
Pass ``--show-progress`` to display a live progress bar during simulation
when ``tqdm`` is installed.
Use ``--fixed-pump-speed`` to hold pumps at a constant relative speed and
skip pump randomization. For example, running

```bash
python scripts/data_generation.py --fixed-pump-speed 1.0 \
    --extreme-rate 0 --pump-outage-rate 0 --local-surge-rate 0
```

generates a clean batch to inspect mean pressures.
After each run the script prints the mean and standard deviation of all
simulated pressures and appends these values along with the key flags to
`pressure_stats.csv` inside the chosen `--output-dir` so repeated runs can be
compared.
If a particular random configuration causes EPANET to fail to produce results,
the script now skips it after a few retries so the actual number of generated
scenarios may be slightly smaller than requested.

Use ``--demand-scale-range MIN MAX`` to adjust the spread of hourly demand
multipliers. The default ``0.8 1.2`` keeps demands centered around their
original values while introducing modest variability. Pass ``--no-demand-scaling``
to disable random demand multipliers altogether (equivalent to ``--demand-scale-range 1 1``).

``--pump-outage-rate`` randomly shuts off one pump for 2–4 hours while
``--local-surge-rate`` applies ±80% demand changes to a small subnetwork for a
similar duration. ``--extreme-rate`` injects a small fraction of stress tests
that scale demands and disable assets to drive pressures near zero. When an
outage occurs a random pipe may also be closed to mimic maintenance. Initial
tank levels are drawn uniformly from the fraction range specified by
``--tank-level-range`` (default ``0 1`` covers the entire feasible range).
Scenario labels are stored alongside the sequence arrays when
``--sequence-length`` is greater than one. Pipe roughness coefficients are
left unchanged; only demand multipliers and pump schedules vary between
scenarios. Pump speeds follow a continuous randomization strategy: each pump
starts from ``[0.3, 0.9]`` and is perturbed by small, temporally correlated
Gaussian noise each hour with a dwell time to avoid rapid cycling.

After scenario generation finishes plots ``dataset_distributions_<timestamp>.png``
and ``pressure_hist_<timestamp>.png`` are created under ``plots/``. The first
summarises the sampled demand multipliers and pump speeds, while the latter
compares the pressure distribution before (normal scenarios) and after the
augmented extremes. A ``manifest.json`` file in the output directory lists the
min/median/max pressure for each scenario and counts how many runs fall below
10 m to verify coverage of extreme low-pressure events.

To create sequence datasets for the recurrent surrogate specify ``--sequence-length``:

```bash
python scripts/data_generation.py \
    --num-scenarios 200 --sequence-length 24 --output-dir data/ \
    --seed 123
```
Use `--deterministic` with `--seed` for bitwise reproducibility.
This will also generate ``edge_index.npy``, ``edge_attr.npy`` and ``edge_type.npy``
along with ``scenario_train.npy`` etc. recording the type of each scenario.
Scenarios that do not contain at least ``sequence_length + 1`` time steps are
skipped with a warning so the actual dataset may be smaller than requested.

The training script automatically detects such sequence files and will use the recurrent
model. Adjust the recurrent hidden size via ``--lstm-hidden`` (alias ``--rnn-hidden-dim``) choosing 64 or 128.

Validate the resulting model with `scripts/experiments_validation.py` before
running the MPC controller.  Both this script and `scripts/mpc_control.py`
now synchronize with EPANET every hour by default (``--feedback-interval 1``).
The validation script executes a 24‑hour simulation with EPANET feedback
applied every hour which keeps predictions from the surrogate model from
diverging over long horizons.  It now also reports mean absolute error (MAE)
and the maximum absolute error for pressure.  Predictions and ground truth are
denormalized so the resulting errors are reported in physical units.  Pressures below 5 m are
 clipped to this lower bound during validation so metrics match the training
distribution.  All metrics are written to
``logs/surrogate_metrics.json`` for reproducibility.

If your surrogate weights were saved separately from their normalization
statistics, provide the path to the ``*.npz`` file via ``--norm-stats`` so the
script can correctly de-normalize predictions.  The validation script aborts if
``y_mean``/``y_std`` are missing to avoid evaluating normalized outputs and it
now cross-checks an MD5 hash stored in the checkpoint metadata against the
contents of the ``*.npz``.  A mismatch triggers an error instructing you to
regenerate or supply matching normalization files.

If the dimension of ``edge_attr.npy`` does not match the value stored in the
surrogate checkpoint, ``validate_surrogate`` now raises a ``ValueError``.
Regenerate the dataset or retrain the surrogate when this occurs.

Typical validation command:

```bash
python scripts/experiments_validation.py \
    --model models/gnn_surrogate.pth --norm-stats models/gnn_surrogate_norm.npz \
    --inp CTown.inp \
    --horizon 6 --iterations 50 \
    --run-name baseline
```

The validation run also exports an animated view of network pressures and pump speeds:

```bash
python scripts/experiments_validation.py \
    --model models/gnn_surrogate.pth --inp CTown.inp \
    --run-name demo
```

This saves `plots/mpc_animation_demo.gif` and an HTML viewer alongside other
figures.

To benchmark architectural variants, run ``scripts/ablation_study.py`` which trains four configurations (baseline, residual, deep and attention) and reports validation pressure MAE along with training time.

```bash
python scripts/ablation_study.py \
    --x-path data/X_train.npy --y-path data/Y_train.npy \
    --edge-index-path data/edge_index.npy --edge-attr-path data/edge_attr.npy \
    --inp-path CTown.inp --epochs 20 --batch-size 32
```

To examine how surrogate errors accumulate without EPANET feedback, enable
roll-out evaluation:

```bash
python scripts/experiments_validation.py \
    --model models/gnn_surrogate.pth --inp CTown.inp \
    --rollout-eval --rollout-steps 24 --run-name baseline
```

This writes per-step RMSE values to `runs/<name>/rollout_rmse.csv` and a plot
to `runs/<name>/rollout_rmse.png`.

## Running MPC control

Once the surrogate model is trained you can run gradient-based MPC using
`scripts/mpc_control.py` (hourly EPANET feedback is the default):

```bash
python scripts/mpc_control.py \
    --horizon 6 --iterations 50 --feedback-interval 24 \
    --Pmin 20.0 --energy-scale 1e-9 \
    --w_p 100 --w_e 1.0 --bias-correction --bias-window 24 --profile
```

Pass ``--profile`` to print the runtime of each MPC optimisation step. The
controller now builds node features on the GPU to minimise Python overhead.
The surrogate model is compiled with TorchScript during loading for faster
inference.  Use ``--no-jit`` to disable this.  ``propagate_with_surrogate`` can
also accept lists of pressure dictionaries to evaluate multiple
scenarios in parallel.

Use ``--skip-normalization`` to disable input normalization and feed raw
features into the surrogate for ablation studies. Outputs are always
de-normalized back to physical units.

Passing ``--bias-correction`` maintains a rolling mean of the last
``--bias-window`` EPANET–surrogate pressure residuals and subtracts it
from future surrogate predictions. Bias estimates reset whenever ground
truth feedback is applied, and the magnitude range of the current bias is
logged each hour.

``mpc_control.py`` exposes weights on pressure and energy terms as
``--w_p`` and ``--w_e`` respectively.  Raw hourly pump energy can reach
``1e8``–``1e9`` Joules, so the default configuration rescales it to
megawatt-hours via ``--energy-scale 1e-9`` before applying the cost.  This keeps
energy magnitudes comparable to pressure penalties and lets ``w_e`` remain near
unity while ``w_p`` defaults to 100 so constraint violations dominate.
Gradients on the control variables are clipped to ``[-gmax, gmax]`` with
``--gmax`` to improve numerical robustness.

Pump energy usage in the MPC cost function is computed from predicted flows and
head gains using the EPANET power equations. This removes the need for a
dedicated energy output and ties the optimisation to physical principles.

By default the controller loads the most recent ``.pth`` file found in the
``models`` directory so retraining will automatically use the newest weights.

This executes a 24‑hour closed loop simulation where fractional pump speeds are
optimized at each hour. In the command above EPANET is only called every
24 hours because ``--feedback-interval 24`` is specified; without this flag the
controller synchronizes every hour. A warning is printed whenever the feedback
interval exceeds one hour to highlight potential drift. Results are written to
`data/mpc_history.csv`.  A summary listing constraint violations and total
energy consumption (in Joules) is printed at the end of the run and saved to
``logs/mpc_summary.json``.
Each simulated hour also produces a network snapshot coloured by pressure with
pump links scaled by their control inputs, saved under
``plots/mpc_network_state_<run>_t<hour>.png``.
The ``--feedback-interval`` flag controls how often EPANET provides ground
truth. ``1`` (default) refreshes the state each hour, ``0`` disables feedback
entirely for a fully surrogate rollout, and larger values propagate the
surrogate for multiple steps before synchronising with EPANET.
The controller updates node demands each hour using the diurnal patterns
specified in ``CTown.inp`` so the surrogate remains aligned with its training
distribution.

**Important:** the surrogate must be trained on datasets that include pump
control inputs (the additional features appended by `scripts/data_generation.py`).
If a model trained with only three features (demand, pressure and
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
    constraint_metrics,
    computational_metrics,
    export_table,
)

# arrays of ground truth and predictions
acc_df = accuracy_metrics(true_p, pred_p)
constraint_df = constraint_metrics(min_p, energy, p_min=20.0)
comp_df = computational_metrics(inference_times, optimisation_times)

# export to CSV
export_table(acc_df, "logs/accuracy.csv")
export_table(constraint_df, "logs/constraints.csv")
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

For demand forecasts the helper `scripts/forecast_uncertainty.py` computes
hourly forecast errors and 95% confidence intervals.  It overlays the mean
forecast and actual demand with shaded error bands and saves the figure to
`figures/forecast_uncertainty.png` by default:

```python
from scripts.forecast_uncertainty import plot_forecast_uncertainty
import pandas as pd

# 48 hourly samples covering two days
timestamps = pd.date_range("2021-01-01", periods=len(actual), freq="H")
plot_forecast_uncertainty(actual, forecast, timestamps)
```

The plot helps visualize how forecast accuracy varies throughout the day.

## Hyperparameter sweep

Use `scripts/sweep_training.py` to evaluate different loss weights and model
architectures.  The script iterates over combinations of `w_press`, `w_mass`,
`w_head`, network depth, hidden dimension and residual connections, training
each configuration and writing the metrics to `data/sweep_results.csv`.

Visualise the resulting pressure errors with:

```bash
python scripts/plot_sweep.py data/sweep_results.csv
```

which saves `plots/pressure_mae_vs_config.png`.
