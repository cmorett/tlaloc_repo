import numpy as np
X = np.load("data/X_train.npy")
print("shape", X.shape)
node_names = np.load("data/node_names.npy", allow_pickle=True)
idx = node_names.tolist().index("J304")
print("J304 first features", X[0,0,idx,:10])
print("pump columns", X[0,0,idx,10:])
print("feature count", X.shape[-1])
