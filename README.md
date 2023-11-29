# FewSOL-DataLoader
This repo hosts the PyTorch dataloader for FewSOL dataset

# Using package
First install the package using
```cmd
pip install path/to/root_dir
```

## Split options
- google_clutter
  - This is a synthetic clutter image split made with 3D google objects
- real_objects
  - This is a real single object image split. Each object was captured from 9 angles
- synthetic_objects
  - This is a synthetic single object image split. Each object was captured from 9 angles
- real_clutter
  - This is a real clutter image split extracted from OCID

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
```

## Data Formats

```
# Image Data Shape
# n x m x q x w x h
# n = Number of total images
# q = 3 : Color slots for RGB
# w = Width of the Image
# h = Height of the image

# Semantic Segmentation Shape
# n x m x w x h
# n = Number of total images
# m = Total number of objects in the current images
# w = Width of the Image
# h = Height of the image

# Detection Bounds Shape
# n x m x r
# n = Number of total images
# m = Total number of objects in the current images
# r = 4 : x, y, width, height

# Label Output/Description Shape
# m = Total number of objects in the images

# Pose Information
# n x m x 4 x 4
# n = Number of total images
# m = Total number of objects in the current images
```

# Datasets
Download the FewSOL dataset from https://irvlutd.github.io/FewSOL/

Pass the extracted dataset directory path into the dataloader as shown in the example above

# Using package
First install the package using
```cmd
pip install path/to/root_dir
```

## Split options
- google_clutter
  - This is a synthetic clutter image split made with 3D google objects
- real_objects
  - This is a real single object image split. Each object was captured from 9 angles
- synthetic_objects
  - This is a synthetic single object image split. Each object was captured from 9 angles
- real_clutter
  - This is a real clutter image split extracted from OCID

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
```

## Data Formats

```
# Image Data Shape
# n x m x q x w x h
# n = Number of total images
# q = 3 : Color slots for RGB
# w = Width of the Image
# h = Height of the image

# Semantic Segmentation Shape
# n x m x w x h
# n = Number of total images
# m = Total number of objects in the current images
# w = Width of the Image
# h = Height of the image

# Detection Bounds Shape
# n x m x r
# n = Number of total images
# m = Total number of objects in the current images
# r = 4 : x, y, width, height

# Label Output/Description Shape
# m = Total number of objects in the images

# Pose Information
# n x m x 4 x 4
# n = Number of total images
# m = Total number of objects in the current images
```

# Datasets
Download the FewSOL dataset from https://irvlutd.github.io/FewSOL/

Pass the extracted dataset directory path into the dataloader as shown in the example above


# Lisences

All files __except__ from 
- 'FewSOL-DataLoader/src/FewSOLDataLoader/SingleRealPose.py' 
- 'FewSOL-DataLoader/src/FewSOLDataLoader/CocoFormatConverter.py' 
is licensed under the MIT liscense.

The file 'FewSOL-DataLoader/src/FewSOLDataLoader/SingleRealPose.py' is licensed under the NVIDIA Source Code License - Non-commercial. This can be found here https://nvlabs.github.io/stylegan2/license.html#:~:text=The%20Work%20and%20any%20derivative,research%20or%20evaluation%20purposes%20only.

The file 'FewSOL-DataLoader/src/FewSOLDataLoader/CocoFormatConverter.py' is licensed under the CC BY 4.0 LEGAL CODE. This can be found here https://cocodataset.org/#termsofuse