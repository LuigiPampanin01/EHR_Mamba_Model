import numpy as np

# Load the .npy file
file_path = '/zhome/21/3/204026/DL-2/EHR_Mamba_model_89/P12data/split_1/test_physionet2012_1.npy'  # Replace with your file path
data = np.load(file_path)

# Display the contents
print(data)