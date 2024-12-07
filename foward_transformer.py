import torch
import numpy as np
from models.regular_transformer import EncoderClassifierRegular

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

# Create an instance of the EncoderClassifierRegular
model = EncoderClassifierRegular(
    device="cpu",
    pooling="mean",
    num_classes=2,
    sensors_count=num_sensors,
    static_count=num_static_features,
    layers=1,
    heads=1,
    dropout=0.2,
    attn_dropout=0.2
)

# Move the model to the appropriate device
model.to("cpu")

# Perform a forward pass
output = model(data, static, time, sensor_mask)

# Print the output
print("Output shape:", output.shape)
print("Output:", output)

predictions = output.squeeze(-1)

print("Predictions shape:", predictions.shape)
print("Predictions:", predictions)