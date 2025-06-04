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
import argparse
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def load_dataset(x_path: str, y_path: str):
    """Load graph data saved as numpy object arrays.

    Each entry of ``X`` should be a dictionary containing ``edge_index`` and
    ``node_features`` arrays.
    """
    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    data_list = []
    for graph_dict, label in zip(X, y):
        edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
        node_feat = torch.tensor(graph_dict['node_features'], dtype=torch.float)
        data = Data(x=node_feat, edge_index=edge_index, y=torch.tensor(label))
        data_list.append(data)
    return data_list

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out.squeeze(), batch.y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def main(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_list = load_dataset(args.x_path, args.y_path)
    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)

    sample = data_list[0]
    model = SimpleGCN(
        in_channels=sample.num_node_features,
        hidden_channels=args.hidden_dim,
        out_channels=args.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train(model, loader, optimizer, device)
        if (epoch + 1) % args.log_every == 0:
            print(f"Epoch {epoch+1:03d} \t Loss: {loss:.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple GCN model")
    parser.add_argument("--x-path", default="X_train.npy", help="Path to graph feature file")
    parser.add_argument("--y-path", default="y_train.npy", help="Path to label file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--log-every", type=int, default=10, help="Log every n epochs")
    parser.add_argument("--output", default="model.pt", help="Output model file")
    args = parser.parse_args()
    main(args)
