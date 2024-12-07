from mortality_part_preprocessing import MortalityDataset, PairedDataset, load_pad_separate
from torch.utils.data import DataLoader
import tqdm
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaModel
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from models.deep_set_attention import DeepSetAttentionModel

batch_size = 16

train_batch_size = batch_size // 2  # we concatenate 2 batches together

train_collate_fn = PairedDataset.paired_collate_fn_truncate
val_test_collate_fn = MortalityDataset.non_pair_collate_fn_truncate

base_path = './P12data'

base_path_new = f"{base_path}/split_{1}"


train_pair, val_data, test_data = load_pad_separate(
    'physionet2012', base_path_new, 1
)

train_dataloader = DataLoader(train_pair, train_batch_size, shuffle=True, num_workers=16, collate_fn=train_collate_fn, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=True, num_workers=16, collate_fn=val_test_collate_fn, pin_memory=True)
val_dataloader = DataLoader(val_data, batch_size, shuffle=False, num_workers=16, collate_fn=val_test_collate_fn, pin_memory=True)

iterable_inner_dataloader = iter(train_dataloader) # make the train_dataloader iterable
test_batch = next(iterable_inner_dataloader) # iterate on the next object in a tuple
max_seq_length = test_batch[0].shape[2] # shape[2] = T
sensor_count = test_batch[0].shape[1] # shape[1] = F
static_size = test_batch[2].shape[1] # shape[1] = 8


#--------------------------------------------- End of Load data

#--------------------------------------------- Start of Embedding layer definition

class MambaEmbedding(nn.Module):
    def __init__(self, sensor_count, embedding_dim, max_seq_length, static_size = 8):
        super(MambaEmbedding, self).__init__()
        self.sensor_count = sensor_count
        self.static_size = static_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

        #For now the embedding_dim is the same as the sensor_count
        self.embedding_dim = sensor_count

        self.sensor_axis_dim_in = 2 * self.sensor_count # 2 * 37 = 74

        # Define the sensor embedding layer
        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in , self.sensor_axis_dim_in)


        # Define the static embedding layer

        self.static_out = self.static_size + 4 # 8 + 4 = 12

        # Define the static embedding layer
        self.static_embedding = nn.Linear(static_size, self.static_out)

        # Define the non-linear merger layer
        self.nonlinear_merger = nn.Linear(self.sensor_axis_dim_in + self.static_out, self.sensor_axis_dim_in + self.static_out)

    def forward(self, data, static, times, mask):
        """
        Args:
            data (torch.Tensor): Input tensor of shape (N, F, T)
            static (torch.Tensor): Static features tensor of shape (N, static_size)
            times (torch.Tensor): Time points tensor of shape (N, T)
            mask (torch.Tensor): Mask tensor of shape (N, F, T)

        Returns:
            torch.Tensor: Encoded output tensor
        """

        x_time = torch.clone(data)  # Torch.size(N, F, T)
        x_time = torch.permute(x_time, (0, 2, 1)) # this now has shape (N, T, F)

        x_sensor_mask = torch.clone(mask)  # (N, F, T)
        x_sensor_mask = torch.permute(x_sensor_mask, (0, 2, 1))  # (N, T, F)


        x_time = torch.cat([x_time, x_sensor_mask], axis=2)  # (N, T, 2F) #Binary


        # make sensor embeddings
        x_time = self.sensor_embedding(x_time)

        # make static embeddings
        static = self.static_embedding(static)
        static_expanded = static.unsqueeze(1).repeat(1, x_time.shape[1], 1)
        x_merged = torch.cat((x_time, static_expanded), axis=-1)

        # Merge the embeddings
        combined = self.nonlinear_merger(x_merged).relu()

        return combined
    
#--------------------------------------------- End of Embedding layer definition



# Define Mamba Configuration
mamba_config = MambaConfig(
    d_model=86,
    hidden_size=86,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=256,
    max_position_embeddings=max_seq_length,
    dropout=0.1
)

print(f"Max sequence lenght is: {max_seq_length}")

# # Initialize Mamba model
mamba_model = MambaModel(mamba_config)

# Define Classification Head
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

classification_head = ClassificationHead(input_dim=mamba_config.d_model, num_classes=2)

mamba_embedding = MambaEmbedding(sensor_count, embedding_dim=mamba_config.d_model, max_seq_length=max_seq_length, static_size=static_size)


# Training Loop
for epoch in range(100):  # Number of epochs
    # mamba_model.train()
    # classification_head.train()
    
    running_loss = 0.0

    for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
        # Unpack batch
        data, times, static, labels, mask, delta = batch


        #Legend:
        # N = Batch size
        # F = 37 number of series in the time series, Albumin etc etc
        # T =  Vary from batch to batch, number of registrations
        # 8 = number of feature
        print(f"Data has shape: {data.shape}") # (N, F, T)
        print(f"Times has shape: {times.shape}") #(N, T)
        print(f"Static has shape: {static.shape}") #(N, 8)
        print(f"Mask has shape: {mask.shape}") # (N,F,T)

        # Forward pass
        embeddings = mamba_embedding(data, static, times, mask)

        print(f"Embeddings has shape: {embeddings.shape}") # (N, T, F)

        mamba_output = mamba_model(inputs_embeds=embeddings)

        print(f"Mamba output has shape: {mamba_output.last_hidden_state.shape}") # (N, T, F)

        # Extract the last hidden state
        last_hidden_state = mamba_output.last_hidden_state  # Shape: [batch_size, sequence_length, d_model]

        # Pool the sequence embeddings (e.g., mean pooling)
        pooled_output = last_hidden_state.mean(dim=1)  # Shape: [batch_size, d_model]

        # Forward pass through the classification head
        logits = classification_head(pooled_output)  # Shape: [batch_size, num_classes]
        
        # Initialize softmax
        softmax = nn.Softmax(dim=-1)

        # Apply softmax to logits
        softmax_output = softmax(logits)

        print(f"Softmax output is: {softmax_output}")

        break

        
    break

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader):.4f}")
