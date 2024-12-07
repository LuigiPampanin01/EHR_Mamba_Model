import torch
import numpy as np
from models.mamba_P12 import CustomMambaModel

# Create dummy data
batch_size = 16
num_sensors = 37
num_static_features = 8
num_time_points = 50

# Random data tensor of shape (batch_size, num_time_points, num_sensors)
data = torch.randn(batch_size, num_sensors, num_time_points)

# Random static features tensor of shape (batch_size, num_static_features)
static = torch.randn(batch_size, num_static_features)

# Random time tensor of shape (batch_size, num_time_points)
time = torch.randint(0, 500, (batch_size, num_time_points))

# Random sensor mask tensor of shape (batch_size, num_time_points, num_sensors)
sensor_mask = torch.randint(0, 2, (batch_size, num_sensors, num_time_points))

# Create an instance of the Mamba_P12 class
model = CustomMambaModel(max_seq_length=num_time_points)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"# of trainable parameters: {params}")

output = model(data, static, time, sensor_mask)

# Print the output
print("Output shape:", output.shape)
#print("Output:", output)

predictions = output.squeeze(-1)

print("Predictions shape:", predictions.shape)
#print("Predictions:", predictions)

model.eval()

output = model(data, static, time, sensor_mask)

print(output)