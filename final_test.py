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


#--------------------------------------------- Load data


# Define Mamba Configuration
# mamba_config = MambaConfig(

# )

# # Initialize Mamba model
# mamba_model = MambaModel(mamba_config)

# Define Classification Head
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# classification_head = ClassificationHead(input_dim=mamba_config.d_model + 37, num_classes=2)


# Training Loop
for epoch in range(100):  # Number of epochs
    # mamba_model.train()
    # classification_head.train()
    
    running_loss = 0.0

    for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
        # Unpack batch
        data, times, static, labels, mask, delta = batch

        print(data.shape)
        print(mask.shape)
        break

        
    break

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader):.4f}")
