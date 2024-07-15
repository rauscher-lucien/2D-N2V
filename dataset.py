import os
import numpy as np
import torch
import tifffile

from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image

from utils import *

class N2VDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        """
        Initializes the dataset with the path to a folder containing TIFF stacks,
        and an optional transform to be applied to each slice.

        Parameters:
        - root_folder_path: Path to the root folder containing TIFF stack files.
        - transform: Optional transform to be applied to each slice.
        """
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.slices = self.preload_and_process_stacks()

    def preload_and_process_stacks(self):
        all_slices = []
        for subdir, _, files in os.walk(self.root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
            for filename in sorted_files:
                full_path = os.path.join(subdir, filename)
                stack = tifffile.imread(full_path)
                self.preloaded_data[full_path] = stack  # Preload data here
                for i in range(stack.shape[0]):
                    all_slices.append((full_path, i))
        return all_slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        file_path, slice_index = self.slices[index]
        slice_data = self.preloaded_data[file_path][slice_index, ...]

        # Ensure data has a channel dimension before applying any transform
        if slice_data.ndim == 2:
            slice_data = slice_data[np.newaxis, ...]  # Add channel dimension (C, H, W)

        # Apply the transform if specified
        if self.transform:
            slice_data = self.transform(slice_data)

        return slice_data


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.slices = self.preload_and_make_slices(root_folder_path)

    def preload_and_make_slices(self, root_folder_path):
        slices = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume  # Preload data here
                num_slices = volume.shape[0]
                for i in range(num_slices):  # Include all slices
                    slices.append((full_path, i))
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        file_path, slice_index = self.slices[index]
        
        # Access preloaded data instead of reading from file
        input_slice = self.preloaded_data[file_path][slice_index]

        if self.transform:
            input_slice = self.transform(input_slice)
        
        # Add extra channel axis at position 0
        input_slice = input_slice[np.newaxis, ...]

        return input_slice