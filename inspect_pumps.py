import numpy as np
X = np.load("data/X_train.npy")
node_names = np.load("data/node_names.npy", allow_pickle=True)
idx = node_names.tolist().index("J304")
pumps = X[0,0,idx,4:15]
print("pump features", pumps)
