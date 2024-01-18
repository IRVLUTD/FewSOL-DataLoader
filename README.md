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

## Example Usage
```python
from FewSOLDataLoader import load_fewsol_dataloader

 # Define the root directory
ROOT_DIR = os.getcwd()

# Define the dataset root directory using the join_path function
DATASET_ROOT_DIR = os.path.join(ROOT_DIR, 'FewSOL', 'data')

# Log that the FewSOL dataloader is being loaded
print('Loading FewSOL dataloader')
     
test = load_fewsol_dataloader(DATASET_ROOT_DIR, split="real_objects")    

# Generate a random index within the range of the dataloader's length
idx = random.randint(0, len(test) - 1)

# Retrieve data from the dataloader for the random index
image_data, semantic_data, bounding_data, label, questionnaire, file_name, poses = test[idx]

# Bounds image helper
# Functions supports 3D(color images) and 2D(no rgb axis)
from FewSOLDataLoader.helpers import bound_img 
bounded_img = bound_img(image_data[0],  bounding_data[0, rand_obj_idx])
```

### Getting indexs for a specfic class
```python
from FewSOLDataLoader import load_fewsol_dataloader

 # Define the root directory
ROOT_DIR = os.getcwd()

# Define the dataset root directory using the join_path function
DATASET_ROOT_DIR = os.path.join(ROOT_DIR, 'FewSOL', 'data')

# Log that the FewSOL dataloader is being loaded
print('Loading FewSOL dataloader')
     
test = load_fewsol_dataloader(DATASET_ROOT_DIR, split="real_objects")    

# Gets the list of indexs for that contains a specific class
class_idxs = test.get_class_idx("bowl")

# Selects a random index out of class idxs
rand_class_idx = class_idxs[random.randint(0, len(class_idxs) - 1)]
    
# Retrieve data from the dataloader for the random index
image_data, semantic_data, bounding_data, label, questionnaire, file_name, poses = test[rand_class_idx]

# Bounds image helper
# Functions supports 3D(color images) and 2D(no rgb axis)
from FewSOLDataLoader.helpers import bound_img 
bounded_img = bound_img(image_data[0],  bounding_data[0, rand_obj_idx])
```

### Loading Specfic Data in order to speed up the dataloader
```python
from FewSOLDataLoader import load_fewsol_dataloader

 # Define the root directory
ROOT_DIR = os.getcwd()

# Define the dataset root directory using the join_path function
DATASET_ROOT_DIR = os.path.join(ROOT_DIR, 'FewSOL', 'data')

# Log that the FewSOL dataloader is being loaded
print('Loading FewSOL dataloader')
     
test = load_fewsol_dataloader(DATASET_ROOT_DIR, split="real_objects")    

# Gets the list of indexs for that contains a specific class
idx = random.randint(0, len(test) - 1)
    
# Retrieve data from the dataloader for the random index
# Default loads all data
# Data not loaded will be None
image_data, semantic_data, bounding_data, label, questionnaire, file_name, poses = test.get_idx(
    idx,
    load_img=False,
    load_sem=True,
    load_bounds=True,
    load_label=False,
    load_quest=False,
    load_pose=False,
)

# Bounds image helper
# Functions supports 3D(color images) and 2D(no rgb axis)
from FewSOLDataLoader.helpers import bound_img 
bounded_img = bound_img(image_data[0],  bounding_data[0, rand_obj_idx])
```

## Data Formats

- Image Data Shape
    ```
    # n x q x w x h
    # n = Number of total images
    # q = 3 : Color slots for RGB
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
