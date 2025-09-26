import numpy as np
import torch
from torch.utils.data import DataLoader
import wntr

from scripts.feature_utils import (
    SequenceDataset,
    apply_sequence_normalization,
    build_node_type,
    build_edge_type,
)
from models.gnn_surrogate import MultiTaskGNNSurrogate

cluster = {"J304","J306","J87","J84","J86","J219","J220","J60","J59","J57","J62","J65","J55","J118","J58","J243","J242","J241","J250","J249","J248","J246","J247","J236","J237","J244","J245","J66","J67","J53","J64","J54","J73","J71","J77","J72","J74","J68","J92","J76","J61","J69","J70","J85","J56","T5"}

# Load arrays
X_val = np.load('data/X_val.npy')
Y_val = np.load('data/Y_val.npy', allow_pickle=True)
edge_index = np.load('data/edge_index.npy')
edge_attr = np.load('data/edge_attr.npy')
edge_attr_val_seq = np.load('data/edge_attr_val_seq.npy')

# Build node/edge types
wn = wntr.network.WaterNetworkModel('CTown.inp')
node_type = build_node_type(wn)
edge_type = build_edge_type(wn, edge_index)

# Create dataset
val_ds = SequenceDataset(
    X_val,
    Y_val,
    edge_index,
    edge_attr,
    node_type=node_type,
    edge_type=edge_type,
    edge_attr_seq=edge_attr_val_seq,
)

# Load model and norm stats
state = torch.load('models/gnn_surrogate_pumpcorr_fix2.pth', map_location='cpu', weights_only=False)
norm = state['norm_stats']
x_mean = torch.tensor(norm['x_mean'])
x_std = torch.tensor(norm['x_std'])
y_mean = {k: torch.tensor(v) for k, v in norm['y_mean'].items()}
y_std = {k: torch.tensor(v) for k, v in norm['y_std'].items()}
edge_mean = torch.tensor(norm['edge_mean'])
edge_std = torch.tensor(norm['edge_std'])

# Apply normalization matching training settings
static_cols = [2, 3]
skip_edge_attr_cols = list(range(edge_attr.shape[1] - 5, edge_attr.shape[1]))
apply_sequence_normalization(
    val_ds,
    x_mean,
    x_std,
    y_mean,
    y_std,
    edge_mean,
    edge_std,
    per_node=True,
    static_cols=static_cols,
    skip_edge_attr_cols=skip_edge_attr_cols,
)

# Prepare model
num_pumps = len(wn.pump_name_list)
pump_offset = 5 if False else 4  # no chlorine
model = MultiTaskGNNSurrogate(
    in_channels=val_ds.X.shape[-1],
    hidden_channels=256,
    edge_dim=edge_attr.shape[1],
    node_output_dim=1,
    edge_output_dim=1,
    num_layers=6,
    use_attention=False,
    gat_heads=4,
    dropout=0.1,
    residual=True,
    rnn_hidden_dim=64,
    share_weights=False,
    num_node_types=int(np.max(node_type) + 1),
    num_edge_types=int(np.max(edge_type) + 1),
    use_checkpoint=False,
    pressure_feature_idx=1,
    use_pressure_skip=True,
    num_pumps=num_pumps,
    pump_feature_offset=pump_offset,
)
model.load_state_dict(state['model_state_dict'])
model.eval()

# Attach normalization tensors to model
model.x_mean = x_mean
model.x_std = x_std
model.y_mean = y_mean
model.y_std = y_std
model.edge_mean = edge_mean
model.edge_std = edge_std

# Tank metadata
edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
tank_indices = [i for i, name in enumerate(wn.node_name_list) if name in wn.tank_name_list]
model.tank_indices = torch.tensor(tank_indices, dtype=torch.long)
areas = []
for name in wn.tank_name_list:
    node = wn.get_node(name)
    diam = getattr(node, 'diameter', 0.0)
    areas.append(np.pi * (float(diam) ** 2) / 4.0)
model.tank_areas = torch.tensor(areas, dtype=torch.float32)
tank_edges = []
tank_signs = []
edge_index_np = edge_index
for idx in tank_indices:
    src = np.where(edge_index_np[0] == idx)[0]
    tgt = np.where(edge_index_np[1] == idx)[0]
    tank_edges.append(torch.tensor(np.concatenate([src, tgt]), dtype=torch.long))
    signs = np.concatenate([-np.ones(len(src)), np.ones(len(tgt))])
    tank_signs.append(torch.tensor(signs, dtype=torch.float32))
model.tank_edges = tank_edges
model.tank_signs = tank_signs

# Data loader
loader = DataLoader(val_ds, batch_size=4, shuffle=False)
node_names = np.load('data/node_names.npy', allow_pickle=True)
cluster_idx = {i for i, name in enumerate(node_names.tolist()) if name in cluster}

residuals = {i: [] for i in range(len(node_names))}

with torch.no_grad():
    node_type_tensor = torch.tensor(node_type, dtype=torch.long)
    edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
    for batch in loader:
        X_seq, edge_attr_seq, target = batch
        preds = model(
            X_seq,
            edge_index_tensor,
            edge_attr_seq,
            node_type_tensor,
            edge_type_tensor,
        )
        node_pred = preds['node_outputs']
        node_true = target['node_outputs']
        y_mean_node = y_mean['node_outputs']
        y_std_node = y_std['node_outputs']
        node_pred = node_pred * y_std_node + y_mean_node
        node_true = node_true * y_std_node + y_mean_node
        diff = node_pred - node_true
        diff_np = diff.squeeze(-1).numpy()
        for node_idx in residuals:
            residuals[node_idx].extend(diff_np[..., node_idx].ravel())

cluster_vals = []
other_vals = []
for idx, vals in residuals.items():
    if not vals:
        continue
    mean_abs = np.mean(np.abs(vals))
    if idx in cluster_idx:
        cluster_vals.append(mean_abs)
    else:
        other_vals.append(mean_abs)

print('cluster count', len(cluster_vals), 'mean', np.mean(cluster_vals), 'median', np.median(cluster_vals))
print('others mean', np.mean(other_vals), 'median', np.median(other_vals))
