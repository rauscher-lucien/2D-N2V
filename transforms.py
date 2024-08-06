import numpy as np
import copy
import numpy.ma as ma
import os
import sys
from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image
import torch

class Normalize(object):
    """
    Normalize a single-channel image using mean and standard deviation.

    Args:
        mean (float): Mean for the channel.
        std (float): Standard deviation for the channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Normalize a single-channel image with dimensions (1, H, W).

        Args:
            img (numpy.ndarray): Image to be normalized, expected to be in the format (1, H, W).

        Returns:
            numpy.ndarray: Normalized image.
        """
        # Normalize the image by subtracting the mean and dividing by the standard deviation
        normalized_img = (img - self.mean) / self.std
        return normalized_img
    

class NormalizeInference(object):
    """
    Normalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (tuple): Containing input and target images to be normalized.
        
        Returns:
            Tuple: Normalized input and target images.
        """
        input_img = data

        # Normalize input image
        input_normalized = (input_img - self.mean) / self.std

        return input_normalized



class RandomHorizontalFlip:
    """
    Apply random horizontal flipping to a single-channel image.
    
    Args:
        None needed for initialization.
    """

    def __call__(self, img):
        """
        Apply random horizontal flipping to a single-channel image with dimensions (1, H, W).
        
        Args:
            img (numpy.ndarray): The image to potentially flip, expected to be in the format (1, H, W).
        
        Returns:
            numpy.ndarray: Horizontally flipped image, if applied.
        """
        # Ensure that we're working with a single-channel image
        if img.ndim != 3 or img.shape[0] != 1:
            raise ValueError("Input image must have dimensions (1, H, W).")

        # Apply horizontal flipping with a 50% chance
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2)  # Flip along the width axis, which is axis 2 for (1, H, W)
        return img



class RandomCrop(object):
    """
    Randomly crop a single-channel image to a specified size.
    
    Args:
        output_size (tuple): The target output size (height, width).
    """

    def __init__(self, output_size=(64, 64)):
        """
        Initializes the RandomCrop transformer with the desired output size.

        Parameters:
        - output_size (tuple): The target output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, img):
        """
        Apply random cropping to a single-channel image with dimensions (1, H, W).

        Parameters:
        - img (numpy.ndarray): The image to be cropped, expected to be in the format (1, H, W).

        Returns:
        - numpy.ndarray: Randomly cropped image.
        """
        # Ensure that we're working with a single-channel image
        if img.ndim != 3 or img.shape[0] != 1:
            raise ValueError("Input image must have dimensions (1, H, W).")

        _, h, w = img.shape
        new_h, new_w = self.output_size

        if h > new_h and w > new_w:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            cropped_img = img[:, top:top+new_h, left:left+new_w]
        else:
            # If the image is smaller than the crop size, padding is required
            padding_top = (new_h - h) // 2 if new_h > h else 0
            padding_left = (new_w - w) // 2 if new_w > w else 0
            padding_bottom = new_h - h - padding_top if new_h > h else 0
            padding_right = new_w - w - padding_left if new_w > w else 0

            cropped_img = np.pad(img, ((0, 0), (padding_top, padding_bottom), (padding_left, padding_right)),
                                 mode='constant', constant_values=0)  # Can modify padding mode and value if needed

        return cropped_img



    

class N2V_mask_generator_median(object):

    def __init__(self, perc_pixel=0.198, n2v_neighborhood_radius=5):
        self.perc_pixel = perc_pixel
        self.local_sub_patch_radius = n2v_neighborhood_radius  # Radius for local neighborhood

    @staticmethod
    def __get_stratified_coords2D__(coord_gen, box_size, shape):
        box_count_y = int(np.ceil(shape[1] / box_size))
        box_count_x = int(np.ceil(shape[2] / box_size))
        x_coords = []
        y_coords = []
        for i in range(box_count_y):
            for j in range(box_count_x):
                y, x = next(coord_gen)
                y = int(i * box_size + y)
                x = int(j * box_size + x)
                if y < shape[1] and x < shape[2]:
                    y_coords.append(y)
                    x_coords.append(x)
        return np.array(y_coords), np.array(x_coords)

    @staticmethod
    def __rand_float_coords2D__(boxsize):
        while True:
            yield np.random.rand() * boxsize, np.random.rand() * boxsize

    def __call__(self, data):
        shape = data.shape  # Determine the shape of the input data
        assert len(shape) == 3 and shape[0] == 1, "Input data must have shape (1, height, width)"

        self.dims = len(shape) - 1  # Number of spatial dimensions (excluding the channel dimension)

        num_pix = int(np.product(shape[1:]) / 100.0 * self.perc_pixel)
        assert num_pix >= 1, "Number of blind-spot pixels is below one. At least {}% of pixels should be replaced.".format(100.0 / np.product(shape[1:]))

        self.box_size = np.round(np.sqrt(100 / self.perc_pixel)).astype(int)
        self.get_stratified_coords = self.__get_stratified_coords2D__
        self.rand_float = self.__rand_float_coords2D__(self.box_size)

        label = data  # Input data as the label
        input_data = copy.deepcopy(label)
        mask = np.ones(label.shape, dtype=np.float32)  # Initialize mask

        coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size, shape=shape)
        indexing = (np.full(coords[0].shape, 0), coords[0], coords[1])
        indexing_mask = (np.full(coords[0].shape, 0), coords[0], coords[1])

        value_manipulation = self.pm_median()
        input_val = value_manipulation(input_data[0], coords, self.dims)

        input_data[indexing] = input_val
        mask[indexing_mask] = 0

        return {'input': input_data, 'label': label, 'mask': mask}

    def pm_median(self):
        def patch_median(patch, coords, dims):
            patch_wo_center = self.mask_center(ndims=dims)
            vals = []
            for coord in zip(*coords):
                sub_patch, crop_neg, crop_pos = self.get_subpatch(patch, coord, self.local_sub_patch_radius)
                slices = [slice(-n, s - p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
                sub_patch_mask = patch_wo_center[tuple(slices)]
                vals.append(np.median(sub_patch[sub_patch_mask]))
            return vals
        return patch_median

    def mask_center(self, ndims=2):
        size = self.local_sub_patch_radius * 2 + 1
        patch_wo_center = np.ones((size,) * ndims)
        patch_wo_center[self.local_sub_patch_radius, self.local_sub_patch_radius] = 0
        return ma.make_mask(patch_wo_center)

    def get_subpatch(self, patch, coord, local_sub_patch_radius, crop_patch=True):
        crop_neg, crop_pos = 0, 0
        if crop_patch:
            start = np.array(coord) - local_sub_patch_radius
            end = start + local_sub_patch_radius * 2 + 1
            crop_neg = np.minimum(start, 0)
            crop_pos = np.maximum(0, end - patch.shape)
            start -= crop_neg
            end -= crop_pos
        else:
            start = np.maximum(0, np.array(coord) - local_sub_patch_radius)
            end = start + local_sub_patch_radius * 2 + 1
            shift = np.minimum(0, patch.shape - end)
            start += shift
            end += shift

        slices = [slice(s, e) for s, e in zip(start, end)]
        return patch[tuple(slices)], crop_neg, crop_pos










class CropToMultipleOf32Inference(object):
    """
    Crop each slice in a stack of images to ensure their height and width are multiples of 32.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            stack (numpy.ndarray): Stack of images to be cropped, with shape (H, W, Num_Slices).

        Returns:
            numpy.ndarray: Stack of cropped images.
        """

        stack = data[0]
        h, w, num_slices = stack.shape  # Assuming stack is a numpy array with shape (H, W, Num_Slices)

        # Compute new dimensions to be multiples of 32
        new_h = h - (h % 32)
        new_w = w - (w % 32)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        # Crop each slice in the stack
        cropped_stack = np.zeros((new_h, new_w, num_slices), dtype=stack.dtype)
        for i in range(num_slices):
            cropped_stack[:, :, i] = stack[id_y, id_x, i].squeeze()

        return cropped_stack
    

class CropToMultipleOf16(object):
    """
    Crop each slice in a stack of images to ensure their height and width are multiples of 32.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            stack (numpy.ndarray): Stack of images to be cropped, with shape (H, W, Num_Slices).

        Returns:
            numpy.ndarray: Stack of cropped images.
        """

        input_slice = data

        _, h, w = data.shape

        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        input_slice_cropped = input_slice[:, id_y, id_x]

        return input_slice_cropped
    


class CropToMultipleOf16Inference(object):
    """
    Crop each slice in a stack of images to ensure their height and width are multiples of 32.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            stack (numpy.ndarray): Stack of images to be cropped, with shape (H, W, Num_Slices).

        Returns:
            numpy.ndarray: Stack of cropped images.
        """

        input_slice = data

        h, w = data.shape

        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        input_slice_cropped = input_slice[id_y, id_x].squeeze()

        return input_slice_cropped


class ToTensor(object):
    """
    Convert dictionaries containing single-channel images to PyTorch tensors. This class is specifically
    designed to handle dictionaries where each value is a single-channel image formatted as (1, H, W).
    """

    def __call__(self, data):
        """
        Convert a dictionary of single-channel images to PyTorch tensors, maintaining the channel position.

        Args:
            data (dict): The input must be a dictionary where each value is a single-channel image
            in the format (1, H, W).

        Returns:
            dict: Each converted image as a PyTorch tensor in the format (1, H, W).
        """
        def convert_image(img):
            # Check image dimensions and convert to tensor
            if img.ndim != 3 or img.shape[0] != 1:
                raise ValueError("Unsupported image format: each image must be 2D with a single channel (1, H, W).")
            return torch.from_numpy(img.astype(np.float32))

        # Ensure data is a dictionary of images
        if isinstance(data, dict):
            return {key: convert_image(value) for key, value in data.items()}
        else:
            raise TypeError("Input must be a dictionary of single-channel images.")

        return converted_tensors



class ToTensorInference(object):
    def __call__(self, img):
        # Convert a single image
        return torch.from_numpy(img.astype(np.float32))


class ToNumpy(object):

    def __call__(self, data):

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    
    

    
class BackTo01Range(object):
    """
    Normalize a tensor to the range [0, 1] based on its own min and max values.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor: A tensor with any range of values.
        
        Returns:
            A tensor normalized to the range [0, 1].
        """
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Avoid division by zero in case the tensor is constant
        if (max_val - min_val).item() > 0:
            # Normalize the tensor to [0, 1] based on its dynamic range
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
        else:
            # If the tensor is constant, set it to a default value, e.g., 0, or handle as needed
            normalized_tensor = tensor.clone().fill_(0)  # Here, setting all values to 0

        return normalized_tensor


class Denormalize(object):
    """
    Denormalize an image using mean and standard deviation, then convert it to 16-bit format.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        """
        Initialize with mean and standard deviation.
        
        Args:
            mean (float or tuple): Mean for each channel.
            std (float or tuple): Standard deviation for each channel.
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        """
        Denormalize the image and convert it to 16-bit format.
        
        Args:
            img (numpy array): Normalized image.
        
        Returns:
            numpy array: Denormalized 16-bit image.
        """
        # Denormalize the image by reversing the normalization process
        img_denormalized = (img * self.std) + self.mean

        # Scale the image to the range [0, 65535] and convert to 16-bit unsigned integer
        img_16bit = img_denormalized.astype(np.uint16)
        
        return img_16bit
