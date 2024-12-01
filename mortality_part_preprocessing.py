import json  # Import the JSON module for handling JSON data
import h5py  # Import the h5py module for handling HDF5 files
import os  # Import the os module for interacting with the operating system
import numpy as np  # Import the numpy module for numerical operations
import tqdm  # Import the tqdm module for progress bars
import torch  # Import the torch module for PyTorch operations
from torch.utils.data import Dataset  # Import the Dataset class from PyTorch

def truncate_to_longest_item_in_batch(data, times, mask, delta):
    """
    Truncate the input tensors to the longest valid time points in the batch.
    This function permutes the dimensions of the input tensors, identifies the valid time points
    based on the mask, and truncates the tensors to include only these valid time points. The 
    tensors are then permuted back to their original dimensions.
    Args:
        data (torch.Tensor): The input data tensor of shape (N, F, T), where N is the batch size, (F is 37 because we have 37 sensors)
                             T is the number of time steps, and F is the number of features.
        times (torch.Tensor): The time steps tensor of shape (N, T).
        mask (torch.Tensor): The mask tensor of shape (N, F, T) indicating valid data points.
        delta (torch.Tensor): The delta tensor of shape (N, F, T) representing time differences.
    Returns:
        tuple: A tuple containing the truncated tensors:
            - data (torch.Tensor): The truncated data tensor of shape (N, F, T').
            - times (torch.Tensor): The truncated time steps tensor of shape (N, T').
            - mask (torch.Tensor): The truncated mask tensor of shape (N, F, T').
            - delta (torch.Tensor): The truncated delta tensor of shape (N, F, T').
    """
    # Permute the dimensions of the data tensor
    data = data.permute((0, 2, 1))  # (N, T, F)
    mask = mask.permute((0, 2, 1))
    delta = delta.permute((0, 2, 1))
    
    # Sum the mask along the last dimension and find valid time points
    # Sum the mask along the last dimension (F) to get the column mask
    col_mask = mask.sum(-1) # (N, T)
    
    # Identify the valid time points where any value in the column mask is non-zero, so that we can keep the time points that have at least one valid feature
    valid_time_points = col_mask.any(dim=0)
    
    # Select only the valid time points and permute back to original dimensions
    data = data[:, valid_time_points, :].permute((0, 2, 1))
    times = times[:, valid_time_points]
    mask = mask[:, valid_time_points, :].permute((0, 2, 1))
    delta = delta[:, valid_time_points, :].permute((0, 2, 1))
    
    return data, times, mask, delta

def load_pad_separate(dataset_id, base_path="", split_index=1, save_path="./processed_datasets"):
    """
    Loads, zero pads, and separates data preprocessed by SeFT

    Files structured as dict = [{
                "ts_values": normalized_values[i],
                "ts_indicators": normalized_measurements[i],
                "ts_times": normalized_times[i],
                "static": normalized_static[i], (8 static values)
                "labels": normalized_labels[i]}]
    """

    # Create the save path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # File paths for preprocessed datasets
    pos_path = os.path.join(save_path, f"{dataset_id}_{split_index}_pos.h5")
    neg_path = os.path.join(save_path, f"{dataset_id}_{split_index}_neg.h5")
    val_path = os.path.join(save_path, f"{dataset_id}_{split_index}_val.h5")
    test_path = os.path.join(save_path, f"{dataset_id}_{split_index}_test.h5")

    # Check if the preprocessed files already exist, and load them if they do
    if save_path and all(os.path.exists(p) for p in [pos_path, neg_path, val_path, test_path]):
        print(f"Loading preprocessed datasets from {save_path}")
        mortality_pos = MortalityDataset(hdf5_path=pos_path)
        mortality_neg = MortalityDataset(hdf5_path=neg_path)
        mortality_val = MortalityDataset(hdf5_path=val_path)
        mortality_test = MortalityDataset(hdf5_path=test_path)
    else:
        # If preprocessed files are not available, proceed with preprocessing
        print(f"Preprocessed files not found. Preprocessing the dataset...")
        Ptrain, Pval, Ptest, norm_params = dataset_loader_splitter(
            dataset_id, base_path, split_index
        )

        # Determine max length based on dataset
        if dataset_id == "physionet2012":
            max_len = 215
        else:
            raise ValueError(f"Dataset {dataset_id} not recognised")

        # Preprocess the datasets
        mortality_pos = MortalityDataset(
            Ptrain, max_length=max_len, norm_params=norm_params
        )
        mortality_neg = MortalityDataset(
            Ptrain, max_length=max_len, norm_params=norm_params
        )
        mortality_val = MortalityDataset(Pval, max_length=max_len, norm_params=norm_params)
        mortality_test = MortalityDataset(
            Ptest, max_length=max_len, norm_params=norm_params
        )

        # Separate positive and negative samples for equal class representation in batches
        ytrain = [item.get("labels") for item in Ptrain]
        ytrain = np.array(ytrain)
        nonzeroes = ytrain.nonzero()[0]
        zeroes = np.where(ytrain == 0)[0]

        # Separate the positive and negative datasets for upsampling
        mortality_pos.select_indices(nonzeroes)
        mortality_neg.select_indices(zeroes)

        # Save the preprocessed datasets if save_path is provided
        if save_path:
            print(f"Saving datasets to {save_path}")
            mortality_pos.save_to_hdf5(pos_path)
            mortality_neg.save_to_hdf5(neg_path)
            mortality_val.save_to_hdf5(val_path)
            mortality_test.save_to_hdf5(test_path)

    # Create a paired dataset from the positive and negative datasets
    mortality_pair = PairedDataset(mortality_pos, mortality_neg)

    return mortality_pair, mortality_val, mortality_test

def dataset_loader_splitter(dataset_id, base_path, split_index):
    """Loads and splits data"""

    # Define file paths for train, validation, test, and normalization data
    split_path_train = "/train_" + dataset_id + "_" + str(split_index) + ".npy"
    split_path_val = "/validation_" + dataset_id + "_" + str(split_index) + ".npy"
    split_path_test = "/test_" + dataset_id + "_" + str(split_index) + ".npy"
    split_path_norm = "/normalization_" + dataset_id + "_" + str(split_index) + ".json"

    print("Loading dataset")

    # Load train, validation, and test data
    Ptrain = np.load(base_path + split_path_train, allow_pickle=True)
    Pval = np.load(base_path + split_path_val, allow_pickle=True)
    Ptest = np.load(base_path + split_path_test, allow_pickle=True)
    
    # Load normalization parameters
    try:
        norm_params = json.load(open(base_path + split_path_norm))
    except Exception:
        norm_params = None

    return Ptrain, Pval, Ptest, norm_params

class PairedDataset(Dataset):
    '''A custom dataset class for handling paired positive and negative samples.
    This class is designed to manage two datasets: one containing positive samples 
    (positive samples could be instances where a certain condition or event of interest (e.g., mortality) is present) 
    and the other containing negative samples.
    It provides functionality to either sample negative examples or repeat positive examples based on the provided flag.
    Attributes:
        dataset_pos (Dataset): The dataset containing positive samples.
        dataset_neg (Dataset): The dataset containing negative samples.
        neg_sample (bool): A flag indicating whether to sample negative examples.
    Methods:
        __init__(self, dataset_pos, dataset_neg, neg_sample=False):
            Initializes the PairedDataset with positive and negative datasets and a flag for negative sampling.
        __len__(self):
            Returns the length of the dataset based on the sampling flag.
        _getitem_negative(self, idx):
            Retrieves a paired sample of positive and negative data when negative sampling is enabled.
        _getitem_positive(self, idx):
            Retrieves a paired sample of positive and negative data when negative sampling is disabled.
        __getitem__(self, idx):
            Retrieves a paired sample based on the sampling flag.
        paired_collate_fn(batch):
            A static method that concatenates and shuffles the paired positive and negative batches.
        paired_collate_fn_truncate(batch):
        A static method that uses the paired_collate_fn to get the batch and then truncates it to the longest item.
            The PairedDataset class is designed to handle these two datasets and provide functionality 
            to either sample negative examples or repeat positive examples based on the neg_sample flag. 
            This can be useful for training machine learning models where you want to balance the number
            of positive and negative samples or create specific pairings for training purposes.'''

    def __init__(self, dataset_pos, dataset_neg, neg_sample=False):
        self.dataset_pos = dataset_pos  # Positive samples dataset
        self.dataset_neg = dataset_neg  # Negative samples dataset
        self.neg_sample = neg_sample  # Flag to indicate negative sampling
        if not self.neg_sample:
            self.dataset_pos.repeat_data(3)  # Repeat positive samples data

    def __len__(self):
        if self.neg_sample:
            return len(self.dataset_neg)  # Length of negative samples dataset
        else:
            return len(self.dataset_pos)  # Length of positive samples dataset

    def _getitem_negative(self, idx):
        pos_data = self.dataset_pos[idx % len(self.dataset_pos)]  # Get positive sample # idx % len(self.dataset_pos) is used to ensure that the index is within the range of the positive dataset
        neg_data = self.dataset_neg[idx]  # Get negative sample
        return pos_data, neg_data

    def _getitem_positive(self, idx):
        pos_data = self.dataset_pos[idx]  # Get positive sample
        neg_data = self.dataset_neg[idx % len(self.dataset_neg)]  # Get negative sample # idx % len(self.dataset_neg) is used to ensure that the index is within the range of the negative dataset
        return pos_data, neg_data

    def __getitem__(self, idx):
        return self._getitem_negative(idx) if self.neg_sample else self._getitem_positive(idx)

    @staticmethod
    def paired_collate_fn(batch):
        """
        Custom collate function to concatenate and shuffle the paired positive and negative batches.
        """
        # Unzip the batch into two lists: positive and negative batches
        pos_batch, neg_batch = zip(*batch)  # The batch is a list of tuples, where each tuple contains a positive sample and a negative sample.
        
        #Each sample is itself a tuple containing multiple elements (e.g., data, times, static, labels, mask, delta).
        # Extract individual elements (data, labels, etc.) from both batches
        pos_data, pos_times, pos_static, pos_labels, pos_mask, pos_delta = zip(*pos_batch)
        neg_data, neg_times, neg_static, neg_labels, neg_mask, neg_delta = zip(*neg_batch)

        # Concatenate each element (data, labels, etc.)
        data = torch.stack(pos_data + neg_data)
        times = torch.stack(pos_times + neg_times)
        static = torch.stack(pos_static + neg_static)
        labels = torch.stack(pos_labels + neg_labels)
        mask = torch.stack(pos_mask + neg_mask)
        delta = torch.stack(pos_delta + neg_delta)

        # Create a list of indices for shuffling
        indices = torch.randperm(data.size(0)) # dimensions of data: (N, T, F), so dim=0 is N

        # Shuffle the concatenated tensors based on the random indices
        data = data[indices]
        times = times[indices]
        static = static[indices]
        labels = labels[indices]
        mask = mask[indices]
        delta = delta[indices]

        return data, times, static, labels, mask, delta
    
    @staticmethod # A static method in Python is a method that belongs to a class but does not require an instance of the class to be called
    def paired_collate_fn_truncate(batch):
        # Use the paired_collate_fn to get the batch
        data, times, static, labels, mask, delta = PairedDataset.paired_collate_fn(batch)
        # Truncate the batch to the longest item
        data, times, mask, delta = truncate_to_longest_item_in_batch(data, times, mask, delta)
        return data, times, static, labels, mask, delta

class MortalityDataset(Dataset):
    '''A PyTorch Dataset class for handling mortality prediction data. This dataset can be initialized 
    either from raw observations or from a preprocessed HDF5 file. It supports various operations 
    such as saving/loading to/from HDF5, selecting specific indices, and repeating the data.
    Attributes:
        data_array (torch.Tensor): Tensor containing the time-series data. Shape: (N, T, F)
        sensor_mask_array (torch.Tensor): Tensor indicating the presence of sensor readings. Shape: (N, T, F)
        times_array (torch.Tensor): Tensor containing the time points of the readings. Shape: (N, T)
        static_array (torch.Tensor): Tensor containing static features. Shape: (N, S)
        label_array (torch.Tensor): Tensor containing the labels for each sample. Shape: (N,)
        delta_array (torch.Tensor): Tensor containing the time deltas between readings. Shape: (N, T, F)
        norm_params (dict): Dictionary containing normalization parameters.
    Methods:
        __init__(obs=None, max_length=2881, norm_params=None, hdf5_path=None):
            Initializes the dataset either from raw observations or from an HDF5 file.
        save_to_hdf5(hdf5_path):
            Saves the dataset to an HDF5 file.
        load_from_hdf5(hdf5_path):
            Loads the dataset from an HDF5 file.
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(idx):
            Returns the sample at the given index.
        select_indices(indices):
            Selects specific indices from the dataset.
        repeat_data(n):
            Repeats the data n times.
        preprocess_sensor_readings(max_length, dict_set):
            Static method to preprocess raw sensor readings into tensors.
        non_pair_collate_fn(batch):
            Static method to collate a batch of samples for the validation dataloader.
        non_pair_collate_fn_truncate(batch):
            Static method to collate and truncate a batch of samples to the longest item in the batch.'''

    def __init__(self, obs=None, max_length=2881, norm_params=None, hdf5_path=None):
        """
        Arguments:
            obs: all experimental results, including active sensors, static sensors, and times (as dict)
        """
        if hdf5_path:
            # Load the dataset from an HDF5 file
            self.load_from_hdf5(hdf5_path)
        else:
            # Process the data if raw observations are provided
            self.norm_params = norm_params
            print("Preprocessing dataset")
            (
                self.data_array,
                self.sensor_mask_array,
                self.times_array,
                self.static_array,
                self.label_array,
                self.delta_array,
            ) = MortalityDataset.preprocess_sensor_readings(max_length, obs)
            self.data_array = self.data_array.permute((0, 2, 1))
            self.sensor_mask_array = self.sensor_mask_array.permute((0, 2, 1))
            self.delta_array = self.delta_array.permute((0, 2, 1))
            print("shape of active data = " + str(np.shape(self.data_array)))
            print("shape of time data = " + str(np.shape(self.times_array)))
            print("shape of static data = " + str(np.shape(self.static_array)))

    def save_to_hdf5(self, hdf5_path):
        # Save the dataset to an HDF5 file
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('data_array', data=self.data_array)
            f.create_dataset('sensor_mask_array', data=self.sensor_mask_array)
            f.create_dataset('times_array', data=self.times_array)
            f.create_dataset('static_array', data=self.static_array)
            f.create_dataset('label_array', data=self.label_array)
            f.create_dataset('delta_array', data=self.delta_array)
            # Save norm_params as JSON string (since it's a dict)
            f.attrs['norm_params'] = json.dumps(self.norm_params)

    def load_from_hdf5(self, hdf5_path):
        """
        Loads datasets and attributes from an HDF5 file into the class instance.

        Args:
            hdf5_path (str): The path to the HDF5 file.

        Attributes:
            data_array (torch.Tensor): The data array loaded from the HDF5 file.
            sensor_mask_array (torch.Tensor): The sensor mask array loaded from the HDF5 file.
            times_array (torch.Tensor): The times array loaded from the HDF5 file.
            static_array (torch.Tensor): The static array loaded from the HDF5 file.
            label_array (torch.Tensor): The label array loaded from the HDF5 file.
            delta_array (torch.Tensor): The delta array loaded from the HDF5 file.
            norm_params (dict): The normalization parameters loaded from the HDF5 file attributes.
            (like mean, std, min, max, etc.)
        Prints:
            str: A message indicating the dataset has been loaded and the path of the HDF5 file.
        """
        '''An HDF5 file can be thought of as a container that holds multiple datasets (like data_array, sensor_mask_array, ...) and attributes.'''
        # Load the dataset from an HDF5 file
        with h5py.File(hdf5_path, 'r') as f:
            self.data_array = torch.tensor(f['data_array'][:], dtype=torch.float32)  # [:] is used to load all the data from the dataset
            self.sensor_mask_array = torch.tensor(f['sensor_mask_array'][:], dtype=torch.float32)
            self.times_array = torch.tensor(f['times_array'][:], dtype=torch.float32)
            self.static_array = torch.tensor(f['static_array'][:], dtype=torch.float32)
            self.label_array = torch.tensor(f['label_array'][:], dtype=torch.long)
            self.delta_array = torch.tensor(f['delta_array'][:], dtype=torch.float32)
            self.norm_params = json.loads(f.attrs['norm_params'])
        print(f"Loaded dataset from {hdf5_path}")

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        return (
            self.data_array[idx],
            self.times_array[idx],
            self.static_array[idx],
            self.label_array[idx],
            self.sensor_mask_array[idx],
            self.delta_array[idx],
        )

    def select_indices(self, indices):
        """
        Select specific indices from the dataset and update the corresponding arrays.

        Parameters:
        indices (array-like): The indices to select from the dataset.

        Updates:
        self.data_array: The data array with selected indices.
        self.times_array: The times array with selected indices.
        self.static_array: The static array with selected indices.
        self.label_array: The label array with selected indices.
        self.sensor_mask_array: The sensor mask array with selected indices.
        self.delta_array: The delta array with selected indices.

        Prints:
        The shapes of the updated arrays.
        """
        # Select specific indices from the dataset
        self.data_array = self.data_array[indices]
        self.times_array = self.times_array[indices]
        self.static_array = self.static_array[indices]
        self.label_array = self.label_array[indices]
        self.sensor_mask_array = self.sensor_mask_array[indices]
        self.delta_array = self.delta_array[indices]
        print("shape of active data = " + str(np.shape(self.data_array))) # the dimension is gonna be (len(indices), T, F)
        print("shape of time data = " + str(np.shape(self.times_array)))
        print("shape of static data = " + str(np.shape(self.static_array)))
        print("shape of labels = " + str(np.shape(self.label_array)))

    def repeat_data(self, n):
        """
        Repeat the data arrays n times along the specified dimensions.

        Parameters:
        n (int): The number of times to repeat the data.

        Modifies:
        self.data_array: Repeats along the first dimension.
        self.times_array: Repeats along the first dimension.
        self.static_array: Repeats along the first dimension.
        self.label_array: Repeats along the first dimension.
        self.sensor_mask_array: Repeats along the first dimension.
        self.delta_array: Repeats along the first dimension.
        """
        # Repeat the data n times
        self.data_array = self.data_array.repeat(n, 1, 1) # dimension is gonna be (N*n, T, F)
        self.times_array = self.times_array.repeat(n, 1) # dimension is gonna be (N*n, T)
        self.static_array = self.static_array.repeat(n, 1) # dimension is gonna be (N*n, S)
        self.label_array = self.label_array.repeat(n)  # dimension is gonna be (N*n,)
        self.sensor_mask_array = self.sensor_mask_array.repeat(n, 1, 1) # dimension is gonna be (N*n, T, F)
        self.delta_array = self.delta_array.repeat(n, 1, 1) # dimension is gonna be (N*n, T, F)

    @staticmethod
    def preprocess_sensor_readings(max_length, dict_set):
        '''Preprocess sensor readings from a dataset.

        This function takes a dataset of sensor readings and preprocesses it into arrays suitable for deep learning models.
        It handles missing readings by zero-padding and computes time deltas between readings.

        Args:
            max_length (int): The maximum length of the time series. Time series shorter than this will be zero-padded.
            dict_set (list of dict): A list of dictionaries, each containing the following keys:
                - "ts_times": List of timestamps for the time series.
                - "ts_indicators": List of sensor mask indicators.
                - "ts_values": List of sensor readings.
                - "static": List of static observations.
                - "labels": List of labels for the time series.

        Returns:
            tuple: A tuple containing the following elements:
                - torch.Tensor: Tensor of shape (subjects, times, sensors) containing the sensor readings.
                - torch.Tensor: Tensor of shape (subjects, times, sensors) containing the sensor mask.
                - torch.Tensor: Tensor of shape (subjects, times) containing the timestamps.
                - torch.Tensor: Tensor of shape (subjects, static_features) containing the static observations.
                - torch.Tensor: Tensor of shape (subjects,) containing the labels.
                - torch.Tensor: Tensor of shape (subjects, times, sensors) containing the time deltas.'''

        # Make a list to hold all individuals
        data_list = []
        sensor_mask_list = []
        static_list = []
        times_list = []
        labels_list = []
        delta_list = []

        # For each individual,
        for ind in tqdm.tqdm(dict_set): # ind is a dictionary
            # Get times, obs values, and static obs values
            times = ind.get("ts_times")
            sensor_mask = ind.get("ts_indicators")
            obs = ind.get("ts_values")  # This is readings for the 36 sensors
            stat = ind.get(
                "static"
            )  # This is static readings for the 9 static data types
            label = ind.get("labels")
            label = np.amax(label)

            # Get size of times list
            if len(times) < max_length:
                # Zero pad the time list
                padding_zeros_times = max_length - len(times)
                times = np.pad(
                    times, (0, padding_zeros_times), "constant", constant_values=(0.0)
                )
                # Zero pad the observations list
                padding_zeros_obs = np.full(
                    (padding_zeros_times, obs.shape[1]), 0, dtype=float
                )
                obs = np.append(obs, padding_zeros_obs, axis=0)
                # Zero pad the sensors mask
                padding_zeros_mask = np.full(
                    (padding_zeros_times, obs.shape[1]), 0, dtype=bool
                )
                sensor_mask = np.append(sensor_mask, padding_zeros_mask, axis=0)

            # Create array of time delta since last reading
            delta = get_delta_t(times, obs, sensor_mask)  # (T, F)
            data_list.append(obs)
            sensor_mask_list.append(sensor_mask)
            times_list.append(times)
            static_list.append(stat)
            labels_list.append(label)
            delta_list.append(delta)

        # Stack the lists into arrays
        data_array = np.stack(data_list)
        sensor_mask_array = np.stack(sensor_mask_list)
        time_array = np.stack(times_list)
        static_array = np.stack(static_list)
        label_array = np.stack(labels_list)
        delta_array = np.stack(delta_list)

        return (
            torch.tensor(data_array, dtype=torch.float32),
            torch.tensor(sensor_mask_array, dtype=torch.float32),
            torch.tensor(time_array, dtype=torch.float32),
            torch.tensor(static_array, dtype=torch.float32),
            torch.tensor(label_array, dtype=torch.long),
            torch.tensor(delta_array, dtype=torch.float32),
        )

    @staticmethod
    def non_pair_collate_fn(batch):
        """
        Custom collate function for the validation dataloader.
        This function organizes the batch into (data, times, static, labels, mask, delta).
        """
        data, times, static, labels, mask, delta = zip(*batch)

        data = torch.stack(data).float()
        times = torch.stack(times).float()
        static = torch.stack(static).float()
        labels = torch.stack(labels).long()
        mask = torch.stack(mask).float()
        delta = torch.stack(delta).float()

        return data, times, static, labels, mask, delta
    
    @staticmethod
    def non_pair_collate_fn_truncate(batch):
        # Use the non_pair_collate_fn to get the batch
        data, times, static, labels, mask, delta = MortalityDataset.non_pair_collate_fn(batch)
        # Truncate the batch to the longest item
        data, times, mask, delta = truncate_to_longest_item_in_batch(data, times, mask, delta)
        return data, times, static, labels, mask, delta

def get_delta_t(times, measurements, measurement_indicators):
    '''Computes the time delta array for each feature measurement.

    This function creates an array with the time elapsed since the most recent 
    feature measurement for each feature in the dataset. It is modified from 
    SeFT's GRU-D Implementation.

    Parameters:
    times (np.ndarray): Array of time points corresponding to each measurement.
    measurements (np.ndarray): Array of feature measurements.
    measurement_indicators (np.ndarray): Binary array indicating whether a 
                                         measurement was taken (1) or not (0).

    Returns:
    np.ndarray: Array of the same shape as `measurements` containing the time 
                deltas for each feature measurement.'''
    """
    Modified from SeFT's GRU-D Implementation.

    Creates array with time from most recent feature measurement.
    """
    dt_list = []

    # First observation has dt = 0
    first_dt = np.zeros(measurement_indicators.shape[1:], dtype=np.float32)  # (F,)
    dt_list.append(first_dt)

    last_dt = first_dt.copy()  # Initialize last_dt before the loop
    for i in range(1, measurement_indicators.shape[0]):
        last_dt = np.where(
            measurement_indicators[i - 1],
            np.full_like(last_dt, times[i] - times[i - 1]),
            times[i] - times[i - 1] + last_dt,
        )
        dt_list.append(last_dt)

    dt_array = np.stack(dt_list)
    dt_array = dt_array.astype(np.float32)  # Ensure consistent data type
    dt_array.shape = measurements.shape  # Reshape to match measurements
    dt_array = dt_array * ~(measurement_indicators.astype(bool))

    return dt_array
