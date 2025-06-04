import torch
import numpy as np
import pickle
import math
import wntr
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

wn = wntr.network.WaterNetworkModel("CTown.inp")

# Load the numpy arrays that data_generation.py wrote
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_val   = np.load('X_val.npy')
Y_val   = np.load('Y_val.npy')
edge_index = np.load('edge_index.npy')  # should be shape [2, num_edges]

# If you pickled test_results_list:
with open('test_results_list.pkl', 'rb') as f:
    test_results_list = pickle.load(f)

edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
X_train_t = torch.tensor(X_train, dtype=torch.float)
Y_train_t = torch.tensor(Y_train, dtype=torch.float)
X_val_t   = torch.tensor(X_val,   dtype=torch.float)
Y_val_t   = torch.tensor(Y_val,   dtype=torch.float)

# Convert data to torch tensors
edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)  # shape [2, num_edges]
# We'll load features in the training loop for each sample, so we won't fix node features tensor now.

# Define GCN model
class GNNSurrogate(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GNNSurrogate, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        # x: [num_nodes, in_dim]
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Instantiate model
model = GNNSurrogate(in_dim=4, hidden_dim=64, out_dim=2)

# Prepare optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Convert training data to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float)  # shape [num_samples, num_nodes, 4]
Y_train_t = torch.tensor(Y_train, dtype=torch.float)  # shape [num_samples, num_nodes, 2]
X_val_t   = torch.tensor(X_val, dtype=torch.float)
Y_val_t   = torch.tensor(Y_val, dtype=torch.float)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    # We will iterate through each training sample (could also batch if needed)
    for i in range(X_train_t.shape[0]):
        x_input = X_train_t[i]        # tensor shape [num_nodes, in_dim]
        y_target = Y_train_t[i]       # tensor shape [num_nodes, out_dim]
        optimizer.zero_grad()
        y_pred = model(x_input, edge_index_tensor)
        loss = loss_fn(y_pred, y_target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= X_train_t.shape[0]
    # Compute validation loss periodically
    if epoch % 10 == 0 or epoch == num_epochs-1:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for j in range(X_val_t.shape[0]):
                x_input = X_val_t[j]
                y_target = Y_val_t[j]
                y_pred = model(x_input, edge_index_tensor)
                val_loss += loss_fn(y_pred, y_target).item()
        val_loss /= X_val_t.shape[0]
        print(f"Epoch {epoch}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "models/gnn_surrogate.pth")

def prepare_features_for_time(sim_results, wn, time_index):
    """
    Given a single sim_results (the WNTR NetworkResults object),
    plus the original WaterNetworkModel `wn`, and a time index i,
    return an array of shape [num_nodes, 4] with
    [base_demand, pressure(i), chlorine(i), elevation] for each node.
    """
    pressures = sim_results.node['pressure'].values      # shape [T, num_nodes]
    quality   = sim_results.node['quality'].values       # shape [T, num_nodes]
    nodes     = wn.node_name_list                        # list of node names, in consistent order
    feat_nodes = []
    for node_idx, node in enumerate(nodes):
        p_t = pressures[time_index, node_idx]
        c_t = quality[time_index, node_idx]
        if node in wn.junction_name_list:
            base_d = wn.get_node(node).demand_timeseries_list[0].base_value
        else:
            base_d = 0.0
        if node in wn.junction_name_list or node in wn.tank_name_list:
            elev = wn.get_node(node).elevation
        else:
            elev = wn.get_node(node).head
        feat_nodes.append([base_d, p_t, c_t, elev])
    return np.array(feat_nodes)  # shape (num_nodes, 4)


rmse_pressure = 0.0
rmse_chlorine = 0.0
count = 0
for sim_results in test_results_list:  # suppose you have some test scenarios
    pressures = sim_results.node['pressure'].values
    quality = sim_results.node['quality'].values
    times = sim_results.node['pressure'].index
    for i in range(len(times)-1):
        # Prepare input features as before
        # ... (similar to data extraction in 2.4)
        x_feat = prepare_features_for_time(sim_results, wn, i)
        x_tensor = torch.tensor(x_feat, dtype=torch.float)
        with torch.no_grad():
            y_pred = model(x_tensor, edge_index_tensor).numpy()
        # Actual next state:
        y_true_pressure = pressures[i+1]
        y_true_quality = quality[i+1]
        # Compute errors
        diff_p = y_pred[:,0] - y_true_pressure  # pressure error for all nodes
        diff_c = y_pred[:,1] - y_true_quality   # chlorine error for all nodes
        rmse_pressure += np.sum(diff_p**2)
        rmse_chlorine += np.sum(diff_c**2)
        count += len(y_true_pressure)
rmse_pressure = math.sqrt(rmse_pressure / count)
rmse_chlorine = math.sqrt(rmse_chlorine / count)
print(f"Surrogate Pressure RMSE: {rmse_pressure:.2f} units, Chlorine RMSE: {rmse_chlorine:.3f} units")
