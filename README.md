# FewSOL-DataLoader
This repo hosts the PyTorch dataloader for FewSOL dataset.<br>
![FewSOL-Dataset](https://raw.githubusercontent.com/IRVLUTD/FewSOL-DataLoader/main/media/fewsol-dataset.png)

# Using package
First install the package using
```cmd
pip install FewSOLDataLoader
```

# Setup
Step-1. Download the FewSOL dataset from https://irvlutd.github.io/FewSOL/#data
- There are four splits of the FewSOL dataset:
     1. `real_objects` : This is a real single object image split. Each object was captured from 9 angles
     2. `real_clutter` : This is a real clutter image split extracted from the [OCID](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/)
     3. `synthetic_objects` : This is a synthetic single object image split made with 3D google objects. Each object was captured from 9 angles
     4. `google_clutter` : This is a synthetic clutter image split made with 3D google objects
         - Note: The `google_clutter` dataloader may take ~60 seconds to instantiate
- Note: The synthetic portion of the dataset is created using [Google 3D Scanned Objects](https://blog.research.google/2022/06/scanned-objects-by-google-research.html?hl=tr&m=1) dataset.

Step-2. Pass the extracted dataset directory path into the dataloader as shown in the following example

## Usage
### Example
```python
import random
from FewSOLDataLoader import load_fewsol_dataloader

 # Define the root directory
ROOT_DIR = os.getcwd()

# Define the dataset root directory using the join_path function
DATASET_ROOT_DIR = os.path.join(ROOT_DIR, 'FewSOL', 'data')
     
data = load_fewsol_dataloader(DATASET_ROOT_DIR, split="real_objects")    

# Generate a random index within the range of the dataloader's length
rand_idx = random.randint(0, len(data) - 1)

# Retrieve data from the dataloader for the random index
image_data, mask_data, bbox_data, label, questionnaire, file_name, poses = data[rand_idx]

# Synthetic objects and Real objects split also has a depth functionality
if s in ['synthetic_objects','real_objects']:
    depth = test.get_depth(rand_idx)
    print("Depth shape:", depth.shape)
```

### Loading Specfic Data in order to speed up the dataloader
```python
# Retrieve data from the dataloader for the random index
# Default loads all data, Data not loaded will be None
image_data, mask_data, bbox_data, label, questionnaire, file_name, poses = data.get_idx(
    rand_idx,
    load_img=False,
    load_mask=True,
    load_bbox=True,
    load_label=False,
    load_que=False,
    load_pose=False,
)
```

### Getting indexs for a specfic class
```python
# Gets the list of indexs for that contains a specific class
class_idxs = data.get_class_idx("bowl")
rand_class_idx = class_idxs[random.randint(0, len(class_idxs) - 1)]
```

### Crop desired object using bbox data
```python
# Functions supports 3D(color images) and 2D(no rgb axis)
from FewSOLDataLoader.helpers import crop_obj_using_bbox
rand_obj_idx = random.randint(0, len(label) - 1)
cropped_img = crop_obj_using_bbox(image_data[0],  bbox_data[0, rand_obj_idx])
```

## Data Formats

- Image Data Shape
    ```
    # n x c x w x h
    # n = Number of total images
    # c = Number of Channels (RGB)
    # w = Width of the Image
    # h = Height of the image
    ```

- Semantic Segmentation Shape
    ```
    # n x m x w x h
    # n = Number of total images
    # m = Total number of objects in the current images
    # w = Width of the Image
    # h = Height of the image
    ```

- Detection Bounds Shape
    ```
    # n x m x r
    # n = Number of total images
    # m = Total number of objects in the current images
    # r = 4 : x, y, width, height
    ```

- Pose Information
    ```
    # n x m x 4 x 4
    # n = Number of total images
    # m = Total number of objects in the current images
    ```

- Label Output/Description Shape
    ```
    # m = Total number of objects in the images
    ```

# Licenses

All files are licensed under the MIT license __except__ for the below two inside `FewSOL-DataLoader/src/FewSOLDataLoader/`
  - `SingleRealPose.py` - licensed under the NVIDIA Source Code License - Non-commercial as found [here](https://nvlabs.github.io/stylegan2/license.html#:~:text=The%20Work%20and%20any%20derivative,research%20or%20evaluation%20purposes%20only).
  - `CocoFormatConverter.py` - licensed under the CC BY 4.0 LEGAL CODE as found [here](https://cocodataset.org/#termsofuse).


# Bibtex
Please cite FewSOL if it helps your research:
```bibtex
@INPROCEEDINGS{padalunkal2023fewsol,
  title={FewSOL: A Dataset for Few-Shot Object Learning in Robotic Environments}, 
  author={P, Jishnu Jaykumar and Chao, Yu-Wei and Xiang, Yu},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  doi={10.1109/ICRA48891.2023.10161143},
  pages={9140-9146},
  year={2023}
}
```
