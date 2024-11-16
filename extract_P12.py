import tarfile
import os

# Define the directory containing the tar.gz files
data_dir = 'P12data'

# Iterate over the files and extract them
for i in range(1, 6):
    file_name = f'P12Data_{i}.tar.gz'
    file_path = os.path.join(data_dir, file_name)
    
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
        print(f'Extracted {file_name} to {data_dir}')