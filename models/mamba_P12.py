from mortality_part_preprocessing import MortalityDataset, PairedDataset, load_pad_separate
from torch.utils.data import DataLoader
import tqdm
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaModel as TransformersMambaModel
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

class TimeEmbedding(nn.Module):
    def __init__(self, d_model, frequencies=None):
        """
        Initialize TimeEmbedding class.

        Args:
            d_model (int): Dimension of the output embedding.
            frequencies (list of int, optional): Frequencies for sinusoidal embeddings (e.g., hourly, daily).
        """
        super(TimeEmbedding, self).__init__()
        self.d_model = d_model
        self.frequencies = frequencies if frequencies else [1, 24]  # Hourly, daily by default
        
        # Learnable linear layer for continuous time embeddings
        self.time_embedding_layer = nn.Linear(1, d_model)
        
        # Combiner for concatenated embeddings (sinusoidal + learnable)
        self.combiner = nn.Linear(d_model + 2 * len(self.frequencies), d_model)

    def forward(self, timestamps, mask=None):
        """
        Forward pass for time embedding.

        Args:
            timestamps (torch.Tensor): Timestamps of shape [batch_size, seq_len].
            mask (torch.Tensor, optional): Binary mask of shape [batch_size, seq_len].
                                            1 indicates valid time steps, 0 indicates missing values.

        Returns:
            torch.Tensor: Combined time embeddings of shape [batch_size, seq_len, d_model].
        """
        # Normalize timestamps to range [0, 1]
        timestamps_normalized = timestamps / (timestamps.max(dim=1, keepdim=True).values + 1e-8)
        
        # Learnable continuous time embeddings
        continuous_embeddings = self.time_embedding_layer(timestamps_normalized.unsqueeze(-1))  # [batch_size, seq_len, d_model]

        # Sinusoidal embeddings with multiple frequencies
        sinusoidal_embeddings = torch.cat([
            torch.sin(2 * torch.pi * timestamps_normalized.unsqueeze(-1) * freq) for freq in self.frequencies
        ] + [
            torch.cos(2 * torch.pi * timestamps_normalized.unsqueeze(-1) * freq) for freq in self.frequencies
        ], dim=-1)  # [batch_size, seq_len, 2 * len(frequencies)]

        # Combine embeddings
        combined_features = torch.cat([continuous_embeddings, sinusoidal_embeddings], dim=-1)  # [batch_size, seq_len, d_model + sinusoidal_dim]
        combined_embeddings = self.combiner(combined_features)  # [batch_size, seq_len, d_model]

        # Apply mask if provided
        if mask is not None:
            combined_embeddings = combined_embeddings * mask.unsqueeze(-1)  # Zero out masked positions

        return combined_embeddings # [batch_size, seq_len, d_model]

class MambaEmbedding(nn.Module):
    def __init__(self, sensor_count, embedding_dim, max_seq_length, static_size = 8):
        super(MambaEmbedding, self).__init__()
        self.sensor_count = sensor_count
        self.static_size = static_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

        self.sensor_axis_dim_in = 2 * self.sensor_count # 2 * 37 = 74

        # Define the sensor embedding layer
        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in , self.sensor_axis_dim_in)

        # Define the static embedding layer

        self.static_out = self.static_size + 4 # 8 + 4 = 12

        # Define the static embedding layer
        self.static_embedding = nn.Linear(static_size, self.static_out)

        # Define the non-linear merger layer
        self.nonlinear_merger = nn.Linear(self.sensor_axis_dim_in + self.static_out + self.sensor_axis_dim_in, self.embedding_dim)

        # Define the time embedding layer
        self.time_embedding = TimeEmbedding(self.sensor_axis_dim_in)

    def forward(self, x, static, times, mask):
        """
        Args:
            data (torch.Tensor): Input tensor of shape (N, F, T)
            static (torch.Tensor): Static features tensor of shape (N, static_size)
            times (torch.Tensor): Time points tensor of shape (N, T)
            mask (torch.Tensor): Mask tensor of shape (N, F, T)

        Returns:
            torch.Tensor: Encoded output tensor
        """

        x_time = torch.clone(x)  # Torch.size(N, F, T)
        x_time = torch.permute(x_time, (0, 2, 1)) # this now has shape (N, T, F)


        mask_handmade = (
            torch.count_nonzero(x_time, dim=2)
        ) > 0 

        x_sensor_mask = torch.clone(mask)  # (N, F, T)
        x_sensor_mask = torch.permute(x_sensor_mask, (0, 2, 1))  # (N, T, F)


        x_time = torch.cat([x_time, x_sensor_mask], axis=2)  # (N, T, 2F) #Binary


        # make sensor embeddings
        x_time = self.sensor_embedding(x_time)


        time_emb = self.time_embedding(times, mask_handmade)

       # add positional encodings
        x_concat = torch.cat((x_time, time_emb), axis=-1) # (N, T, 2F + time_emb_size)

        # make static embeddings
        static = self.static_embedding(static)
        static_expanded = static.unsqueeze(1).repeat(1, x_time.shape[1], 1)
        x_merged = torch.cat((x_concat, static_expanded), axis=-1)

        # Merge the embeddings
        combined = self.nonlinear_merger(x_merged)

        return combined
    

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)
    
class CustomMambaModel(nn.Module):
    def __init__(
        self, 
        max_seq_length, 
        num_classes=2,
        static_size=8, 
        sensor_count=37, 
        embedding_dim=86, 
        d_model=86,
        **kwargs
        ):

        super().__init__()

        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        self.static_size = static_size
        self.sensor_count = sensor_count
        self.embedding_dim = embedding_dim
        self.d_model = d_model
     

        self.mamba_config = MambaConfig(
            hidden_size=d_model,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=128,
            max_position_embeddings=max_seq_length,
            dropout=0.1
        )
        self.embedding = MambaEmbedding(sensor_count, embedding_dim, max_seq_length, static_size)
        self.head = ClassificationHead(input_dim=d_model, num_classes=num_classes)
        self.mamba_model = TransformersMambaModel(self.mamba_config)
    
    def forward(self, x, static, time, sensor_mask, **kwargs):
        embeddings = self.embedding(x, static, time, sensor_mask)

        mamba_output = self.mamba_model(inputs_embeds=embeddings)
        last_hidden_state = mamba_output.last_hidden_state  # Shape: [batch_size, sequence_length, d_model]
        
        # Pool the sequence embeddings (e.g., mean pooling)
        pooled_output = last_hidden_state.mean(dim=1)  # Shape: [batch_size, d_model]

        # Forward pass through the classification head
        logits = self.head(pooled_output)  # Shape: [batch_size, num_classes]

        return logits

