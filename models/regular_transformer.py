import torch
import numpy as np
from torch import nn
from x_transformers import Encoder


def masked_mean_pooling(datatensor, mask):
    """
    Computes the masked mean pooling of the input tensor along the specified dimension.

    This function calculates the average of the values in the input tensor `datatensor` 
    while ignoring the values at positions where the `mask` tensor is zero. It is 
    particularly useful for handling sequences of varying lengths where padding is 
    applied.

    Args:
        datatensor (torch.Tensor): The input tensor of shape (batch_size, feature_dim, lenght_time).
        mask (torch.Tensor): A binary mask tensor of shape (batch_size, sequence_length) 
                             where 1 indicates valid data points and 0 indicates padding.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, feature_dim) containing the masked 
                      mean pooled values for each sequence in the batch.
    """
 
 
    # eliminate all values learned from nonexistant timepoints
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float() # Takes the mask tensor, adds an extra dimension at the end,
    # expands it to match the size of datatensor, and converts it to a floating-point tensor.
    data_summed = torch.sum(datatensor * mask_expanded, dim=1)

    # find out number of existing timepoints
    data_counts = mask_expanded.sum(1)
    data_counts = torch.clamp(data_counts, min=1e-9)  # put on min clamp

    # Calculate average:
    averaged = data_summed / (data_counts)

    return averaged


def masked_max_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    """
    # eliminate all values learned from nonexistant timepoints
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()

    datatensor[mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    maxed = torch.max(datatensor, 1)[0]

    return maxed


class PositionalEncodingTF(nn.Module):
    """
    Based on the SEFT positional encoding implementation
    """

    def __init__(self, d_model, max_len=500):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        P_time = P_time.float()

        # create a timescale of all times from 0-1
        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        # make a tensor to hold the time embeddings
        times = torch.Tensor(P_time.cpu()).unsqueeze(2)

        # scale the timepoints according to the 0-1 scale
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        # Use a 32-D embedding to represent a single time point
        pe = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1
        )  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        return pe


class EncoderClassifierRegular(nn.Module):

    def __init__(
        self,
        device="cpu",
        pooling="mean",
        num_classes=2,
        sensors_count=37,
        static_count=8,
        layers=1,
        heads=1,
        dropout=0.2,
        attn_dropout=0.2,
        **kwargs
    ):
        super().__init__()

        self.pooling = pooling
        self.device = device
        # Number of sensors, eg, 37 for Physionet Albumin, ALT, AST, etc.
        self.sensors_count = sensors_count
        # Number of static variables of the patient, Age, Gender, etc.
        self.static_count = static_count

        # The input dimension of the sensors is 2 times the number of sensors
        # Because we have a binary mask for each sensor
        # 37 sensors, 37 binary masks = 74
        self.sensor_axis_dim_in = 2 * self.sensors_count

        self.sensor_axis_dim = self.sensor_axis_dim_in
        # If the number of sensors is odd, we add one to make it even
        if self.sensor_axis_dim % 2 != 0:
            self.sensor_axis_dim += 1

        #The subsequent layers may require the static_count to be 12
        self.static_out = self.static_count + 4

        self.attn_layers = Encoder(
            dim=self.sensor_axis_dim,
            depth=layers,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=dropout,
        )

        # This embedding is used for the 37 time series of the pysionet
        # This is 74 because we have a binary mask for each sensor
        # Since it's even, the output is the same as the input
        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in, self.sensor_axis_dim)

        # Static is used for the rest of the constant variables, eg. Age.
        self.static_embedding = nn.Linear(self.static_count, self.static_out)
        # 74 + 12 = = 86
        self.nonlinear_merger = nn.Linear(
            self.sensor_axis_dim + self.static_out,
            self.sensor_axis_dim + self.static_out,
        )

        # This is the final layer that will be used to classify the output
        # Dim input = 74 + 12 = (sensors + mask embedding) + (static embedding)  =  86
        self.classifier = nn.Linear(
            self.sensor_axis_dim + self.static_out, num_classes
        )

        self.pos_encoder = PositionalEncodingTF(self.sensor_axis_dim)

    def forward(self, x, static, time, sensor_mask, **kwargs):

        # Note that the input x is a tensor of shape (N, T, F) where:
        # - N is the batch size,
        # - F is the number of features.
        # - T is the number of time points,

        x_time = torch.clone(x)  # Torch.size(N, F, T)
        x_time = torch.permute(x_time, (0, 2, 1))  # this now has shape (N, T, F)
        
        # TODO: Check if this is correct
        mask = (
            torch.count_nonzero(x_time, dim=2)
        ) > 0  # mask for sum of all sensors for each person/at each timepoint
        # mask is a tensor of shape (N, T)
        # where N is the batch size and T is the number of time points
        # mask is a binary tensor that is 1 if there is a non-zero value at that time point

        # Keep in mind that x_time is a tensor of shape (Batch, #of registrations, #of sensors)

        # add indication for missing sensor values
        x_sensor_mask = torch.clone(sensor_mask)  # (N, F, T)
        x_sensor_mask = torch.permute(x_sensor_mask, (0, 2, 1))  # (N, T, F)
        x_time = torch.cat([x_time, x_sensor_mask], axis=2)  # (N, T, 2F) #Binary

        # make sensor embeddings
        x_time = self.sensor_embedding(x_time)  # (N, T, 2F)

        # add positional encodings
        pe = self.pos_encoder(time).to(self.device)  # taken from RAINDROP, (N, T, pe)
        x_time = torch.add(x_time, pe)  # (N, T, F) (N, F)
        # note by sav x_time has still shape (N, T, 2F)

        # run  attention
        x_time = self.attn_layers(x_time, mask=mask)

        if self.pooling == "mean":
            x_time = masked_mean_pooling(x_time, mask)
        elif self.pooling == "median":
            x_time = torch.median(x_time, dim=1)[0]
        elif self.pooling == "sum":
            x_time = torch.sum(x_time, dim=1)  # sum on time
        elif self.pooling == "max":
            x_time = masked_max_pooling(x_time, mask)

        # concatenate poolingated attented tensors
        static = self.static_embedding(static)
        x_merged = torch.cat((x_time, static), axis=1)

        nonlinear_merged = self.nonlinear_merger(x_merged).relu()

        # classify!
        return self.classifier(nonlinear_merged)
