import os
import json
import pickle as pkl
from .DataLoader import (
    FewSOLDataloader, 
    GooogleClutterDataloader,
    RealClutterDataLoader
)
import yaml
from easydict import EasyDict
import torch


def join_path(*args):
    """
    Join multiple path components into a single path.

    Args:
        *args (str): One or more string arguments representing path components.

    Returns:
        str: The joined path.

    Example:
        >>> join_path("home", "user", "documents", "file.txt")
        'home/user/documents/file.txt'
    """
    return os.path.join(*args)


def make_cache_dir(dataset_root_dir):
    """
    Create a cache directory for storing data loader files.

    Args:
        dataset_root_dir (str): The root directory of the dataset.

    Returns:
        str: The path to the created cache directory.

    Example:
        >>> make_cache_dir("/path/to/dataset")
        '/path/to/dataset/cache'
    """
    # Create the cache directory if it doesn't exist
    cache_dir = join_path(dataset_root_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_fewsol_dataloader(dataset_root_dir, split='real_objects'):
    """
    Load or create a FewSOL dataloader for a specified split.

    Args:
        dataset_root_dir (str): The root directory of the dataset.
        split (str, optional): The dataset split to load ('real_objects' by default).

    Returns:
        FewSOLDataloader: The FewSOL dataloader for the specified split.

    Raises:
        ValueError: If the specified split is not valid.

    Example:
        >>> loader = load_fewsol_dataloader("/path/to/dataset", split='real_objects')
    """
    split_options = ('real_objects', 'synthetic_objects', 'google_clutter', 'real_clutter')  # Use a tuple for valid split options

    if split not in split_options:
        raise ValueError(f'Split invalid!!! Possible values: {split_options}')

    cache_path = make_cache_dir(dataset_root_dir)
    fewsol_cache_path = join_path(cache_path, f'{split}_fewsol_test_dataloader.pkl')

    # If cached dataloader exists, read it
    if os.path.exists(fewsol_cache_path):
        print(f'Reading from cached FewSOL dataloader: {fewsol_cache_path}')
        with open(fewsol_cache_path, 'rb') as f:
            return pkl.load(f)
    else:
        # Create dataloader and cache
        print(f'No cached dataloader present. Creating one for FewSOL @ :{fewsol_cache_path}')
        
        data_loader = None
        if split == "real_objects" or split == "synthetic_objects":
            data_loader = FewSOLDataloader(join_path(dataset_root_dir, split))
        elif split == "google_clutter":
            data_loader = GooogleClutterDataloader(join_path(dataset_root_dir, "google_scenes"))
        elif split == "real_clutter":
            data_loader = RealClutterDataLoader(join_path(dataset_root_dir, "OCID_objects"))

        # Cache dataloader
        with open(fewsol_cache_path, 'wb') as f:
            pkl.dump(data_loader, f)
        return data_loader



def bb_intersection_over_union(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (list): A list containing the coordinates of the first bounding box in the format [x1, y1, x2, y2].
        boxB (list): A list containing the coordinates of the second bounding box in the format [x1, y1, x2, y2].

    Returns:
        float: The IoU value, a number between 0 and 1.

    Raises:
        ValueError: If the input bounding boxes are not well-formed.

    Example:
        boxA = [10, 10, 50, 50]
        boxB = [30, 30, 70, 70]
        iou = bb_intersection_over_union(boxA, boxB)
        print("IoU:", iou)
    """
    try:
        # Ensure the input bounding boxes are well-formed
        if len(boxA) != 4 or len(boxB) != 4:
            raise ValueError("Bounding boxes must have 4 coordinates [x1, y1, x2, y2].")

        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of the intersection rectangle
        interArea = max((xB - xA), 0) * max((yB - yA), 0)

        # Handle the case of no intersection
        if interArea == 0:
            return 0

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # Return the intersection over union value
        return iou

    except ValueError as e:
        raise Exception(f"Error in bb_intersection_over_union: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def get_config(box_threshold, text_threshold):
    config_file = 'config.yml'
    config_absolute_path = os.path.join(os.getcwd(), config_file)
    # Read data from the YAML file
    logging.info(f"Reading config file: {config_absolute_path}")
    with open(config_absolute_path, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    config = EasyDict(**config_dict)
    # for hyperparameter search
    config.BOX_THRESHOLD = box_threshold
    config.TEXT_THRESHOLD = text_threshold
    return config


def scale_bboxes_wrt_H_W(bboxes, H, W):
    for i in range(bboxes.size(0)):
        bboxes[i] = bboxes[i] * torch.Tensor([W, H, W, H])
        bboxes[i][:2] -= bboxes[i][2:] / 2
        bboxes[i][2:] += bboxes[i][:2]
    return bboxes


# Accepts image and bounding data
def crop_obj_using_bbox(image_data, bbox_data):
    """
    Args: 
        - Image (np.ndarray)
        - 1 bbox (np.ndarray)
    """
    assert len(image_data.shape) ==2 or len(image_data.shape) == 3, f"Only accepts images with 2 or 3 dimensions supplied images has {len(image_data.shape)}"
    
    if len(image_data.shape) == 2:
        return image_data[
            bbox_data[1]:bbox_data[1] + bbox_data[3],
            bbox_data[0]:bbox_data[0] + bbox_data[2]
        ]
    else:
        # rgba axis first
        if image_data.shape[0] == 3 or image_data.shape[0] == 4:
            return image_data[:,
                bbox_data[1]:bbox_data[1] + bbox_data[3],
                bbox_data[0]:bbox_data[0] + bbox_data[2]
            ]
        # rbga axis last
        else:
            return image_data[
                bbox_data[1]:bbox_data[1] + bbox_data[3],
                bbox_data[0]:bbox_data[0] + bbox_data[2]
            ]