import wntr
import numpy as np
import numpy as _np
import random
import pandas as pd
import pickle as _pkl

# Load the EPANET INP file for C-Town
inp_file = 'CTown.inp'
wn = wntr.network.WaterNetworkModel(inp_file)

# Set simulation duration and time steps
wn.options.time.duration = 24 * 3600          # 24 hours in seconds
wn.options.time.hydraulic_timestep = 3600     # 1 hour hydraulic time step
wn.options.time.quality_timestep = 3600       # 1 hour quality step (matches hydraulic)
wn.options.time.report_timestep = 3600        # reporting results every hour

wn.options.quality.parameter = 'CHEMICAL'     # simulate chemical transport (e.g., chlorine)

# Initialize source quality (reservoirs and tanks)
for reservoir_name in wn.reservoir_name_list:
    wn.get_node(reservoir_name).initial_quality = 1.0   # 1 mg/L initial at reservoir
for tank_name in wn.tank_name_list:
    wn.get_node(tank_name).initial_quality = 1.0        # 1 mg/L initial at tank

N = 10  # number of scenarios
results_list = []

# Save original base‐demands for all junctions
original_demands = {
    jname: wn.get_node(jname).demand_timeseries_list[0].base_value
    for jname in wn.junction_name_list
}

for scenario in range(N):
    # 1) Randomly scale each junction's demand
    for jname in wn.junction_name_list:
        base = original_demands[jname]
        wn.get_node(jname).demand_timeseries_list[0].base_value = base * random.uniform(0.5, 1.5)

    # 2) Turn all pumps ON (use initial_status instead of .status)
    for pn in wn.pump_name_list:
        wn.get_link(pn).initial_status = 1

    # 3) Close ~30% of pumps (set initial_status = 0)
    pumps_to_off = random.sample(wn.pump_name_list, max(1, int(0.3 * len(wn.pump_name_list))))
    for pn in pumps_to_off:
        wn.get_link(pn).initial_status = 0

    # 4) Run the hydraulic + quality simulation
    sim = wntr.sim.EpanetSimulator(wn)
    sim_results = sim.run_sim()
    results_list.append(sim_results)

    # 5) Restore base‐demands and reopen all pumps for next scenario
    for jname in wn.junction_name_list:
        wn.get_node(jname).demand_timeseries_list[0].base_value = original_demands[jname]
    for pn in wn.pump_name_list:
        wn.get_link(pn).initial_status = 1


# Suppose you want 70% of results for training, 15% for validation, 15% for test
num_total = len(results_list)
indices = np.random.permutation(num_total)
n_train = int(0.7 * num_total)
n_val   = int(0.15 * num_total)
train_idx = indices[:n_train]
val_idx   = indices[n_train : n_train + n_val]
test_idx  = indices[n_train + n_val :]

train_results_list = [results_list[i] for i in train_idx]
val_results_list   = [results_list[i] for i in val_idx]
test_results_list  = [results_list[i] for i in test_idx]

# Then your X_train, Y_train come from train_results_list, etc.
# And `test_results_list` now exists.


# =============================================================================
# Now extract features (X) and labels (Y) from each sim_results in results_list
# =============================================================================

X_list = []  # input features for each time‐step sample
Y_list = []  # target labels for each time‐step sample

for sim_results in results_list:
    pressures = sim_results.node['pressure']   # DataFrame: index=(variable, time) × columns=node
    quality   = sim_results.node['quality']    # DataFrame: chlorine concentration over time
    pressure_array = pressures.values   # shape (num_times, num_nodes)
    quality_array  = quality.values     # shape (num_times, num_nodes)
    times = pressures.index            # times (in seconds) for each row

    # Build (X, Y) for t → t+1
    for i in range(len(times) - 1):
        # At time t = times[i], we collect node‐level features; Y is at time t_next.
        feat_nodes = []
        for node in wn.node_name_list:
            idx = pressures.columns.get_loc(node)
            p_t = pressure_array[i, idx]
            c_t = quality_array[i, idx]

            # Demand at node (if it’s a junction): use current base_demand
            if node in wn.junction_name_list:
                base_d = wn.get_node(node).demand_timeseries_list[0].base_value
            else:
                base_d = 0.0

            # Elevation: junction/tank → .elevation; reservoir → .head
            if (node in wn.junction_name_list) or (node in wn.tank_name_list):
                elev = wn.get_node(node).elevation
            else:
                elev = wn.get_node(node).head

            feat_nodes.append([base_d, p_t, c_t, elev])
        X_list.append(np.array(feat_nodes))

        # Now collect Y at time t+1
        out_nodes = []
        for node in wn.node_name_list:
            idx = pressures.columns.get_loc(node)
            p_next = pressure_array[i + 1, idx]
            c_next = quality_array[i + 1, idx]
            out_nodes.append([p_next, c_next])
        Y_list.append(np.array(out_nodes))


for idx, arr in enumerate(X_list):
    print(f"Index {idx}: arr.shape = {arr.shape}, arr.dtype = {arr.dtype}")
    assert arr.dtype in (np.float32, np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 4


for arr in X_list:
    assert isinstance(arr, np.ndarray)
    assert arr.dtype in (np.float32, np.float64)
    # e.g. if you expect (num_nodes, 4):
    assert arr.ndim == 2 and arr.shape[1] == 4


X_all = np.stack(X_list)   # shape: (num_samples, num_nodes, 4)
Y_all = np.stack(Y_list)   # shape: (num_samples, num_nodes, 2)

# Shuffle & split into train/val
num_samples = X_all.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)
train_idx = indices[: int(0.8 * num_samples)]
val_idx   = indices[int(0.8 * num_samples):]

X_train, Y_train = X_all[train_idx], Y_all[train_idx]
X_val,   Y_val   = X_all[val_idx],   Y_all[val_idx]

# Build edge_index for Graph Neural Network
edge_index = []
node_index_map = {name: idx for idx, name in enumerate(wn.node_name_list)}

for link_name, link_obj in wn.links():
    n1 = link_obj.start_node.name
    n2 = link_obj.end_node.name
    i1, i2 = node_index_map[n1], node_index_map[n2]
    edge_index.append([i1, i2])
    edge_index.append([i2, i1])

edge_index = np.array(edge_index).T  # shape (2, 2 * num_links)

# … after you compute X_train, Y_train, X_val, Y_val, edge_index, test_results_list …

# in data_generation.py, after you build X_all (which should be float):
np.save('X_train.npy', X_train.astype(np.float32))   # <-- explicitly float32 or float64
np.save('Y_train.npy', Y_train.astype(np.float32))
np.save('X_val.npy',   X_val.astype(np.float32))
np.save('Y_val.npy',   Y_val.astype(np.float32))
np.save('edge_index.npy', edge_index.astype(np.int64))


# If test_results_list is not a plain array but a list of results, you could pickle it:

with open('test_results_list.pkl', 'wb') as f:
    _pkl.dump(test_results_list, f)
