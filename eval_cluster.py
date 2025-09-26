import csv
import numpy as np
from pathlib import Path

cluster = {"J304","J306","J87","J84","J86","J219","J220","J60","J59","J57","J62","J65","J55","J118","J58","J243","J242","J241","J250","J249","J248","J246","J247","J236","J237","J244","J245","J66","J67","J53","J64","J65","J54","J73","J71","J77","J72","J74","J68","J92","J76","J61","J69","J70","J85","J56","T5"}
node_names = np.load('data/node_names.npy', allow_pickle=True)
idx_to_name = {i: name for i, name in enumerate(node_names.tolist())}
cluster_idx = {i for i, name in idx_to_name.items() if name in cluster}
maes = []
with open('logs/eval_sequence_node_errors.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        val = row['mae'].strip()
        if not val:
            continue
        idx = int(row['node_index'])
        mae = float(val)
        maes.append((idx, mae))
cluster_mae = [mae for idx, mae in maes if idx in cluster_idx]
other_mae = [mae for idx, mae in maes if idx not in cluster_idx]
print('cluster count', len(cluster_mae), 'mean', np.mean(cluster_mae), 'median', np.median(cluster_mae))
print('others mean', np.mean(other_mae), 'median', np.median(other_mae))
print('top cluster nodes:')
for idx, mae in sorted(maes, key=lambda x: x[1], reverse=True)[:10]:
    print(idx, idx_to_name[idx], mae)
