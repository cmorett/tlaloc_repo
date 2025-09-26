import torch
state = torch.load("models/gnn_surrogate_pumpcorr_fix.pth", map_location="cpu", weights_only=False)
y_mean = state['norm_stats']['y_mean']['node_outputs']
y_std = state['norm_stats']['y_std']['node_outputs']
import numpy as np
node_names = np.load('data/node_names.npy', allow_pickle=True)
idx = node_names.tolist().index('J304')
print('J304 mean/std', y_mean[idx,0], y_std[idx,0])
