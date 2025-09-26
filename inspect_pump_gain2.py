import torch
state = torch.load("models/gnn_surrogate_pumpcorr_fix2.pth", map_location="cpu", weights_only=False)
model = state["model_state_dict"]
print("pump_gain", model["pump_gain"].numpy())
