import torch
state = torch.load("models/gnn_surrogate_pumpcorr_fix.pth", map_location="cpu", weights_only=False)
weights = state["model_state_dict"]
for name, tensor in weights.items():
    if "pump_gain" in name:
        print(name, tensor.numpy())
