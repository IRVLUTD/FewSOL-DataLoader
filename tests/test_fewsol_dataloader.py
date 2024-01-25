from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
# Adds working directory to path in order to access all the packages
sys.path.insert(0, os.getcwd())

import random
import numpy as np
import matplotlib.pyplot as plt
from src.FewSOLDataLoader import (
    load_fewsol_dataloader
)
from src.FewSOLDataLoader.helpers import crop_obj_using_bbox 

"""Test FewSOl dataloader."""
def main():
    # Load the FewSOL dataloader for the 'real_objects' and 'synthetic_objects' split
    splits = ['synthetic_objects','real_objects',  'real_clutter', 'google_clutter']
    for s in splits:
        test_split(s)

def test_split(s):
    print(f"Testing {s} split")
    
    # Define the root directory
    ROOT_DIR = os.getcwd()

    # Define the dataset root directory using the join_path function
    DATASET_ROOT_DIR = os.path.join(ROOT_DIR, "..", "OpenVocabDetection",  'OpenVocabX', 'datasets', 'FewSOL', 'data')
    
    # Removes cache file if it exist in order to test most recent changes
    cache_file = os.path.join(DATASET_ROOT_DIR, "cache", f"{s}_fewsol_test_dataloader.pkl")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("Removed Cached File")

    # Log that the FewSOL dataloader is being loaded
    print('Loading FewSOL dataloader')
     
    test = load_fewsol_dataloader(DATASET_ROOT_DIR, split=s)    

    # Generate a random index within the range of the dataloader's length
    idx = random.randint(0, len(test) - 1)
    
    # Gets the list of indexs for that contains a specific class
    class_idxs = test.get_class_idx("bowl")
    rand_class_idx = class_idxs[random.randint(0, len(class_idxs) - 1)]
    

    # Retrieve data from the dataloader for the random index
    image_data, semantic_data, bounding_data, label, questionnaire, file_name, poses = test[idx]
    #image_data, semantic_data, bounding_data, label, questionnaire, file_name, poses = test[rand_class_idx]
    """
    image_data, semantic_data, bounding_data, label, questionnaire, file_name, poses = test.get_idx(
        idx,
        load_img=True,
        load_sem=False,
        load_bounds=False,
        load_label=False,
        load_quest=False,
        load_pose=False,
    )
    """
    
    
    if s in ['synthetic_objects','real_objects']:
        depth = test.get_depth(idx)
        print("Depth shape:", depth.shape)

    
    # Picks a random object index in the image
    rand_obj_idx = random.randint(0, len(label) - 1)
    label = label[rand_obj_idx]
    if questionnaire != None:
        questionnaire = questionnaire[rand_obj_idx]
       
    modified_sem = None
    # Creates the segementation for all the available objects
    # Each object has a different id corresponding to it's index
    if s == "real_clutter" or s == "google_clutter":
        modified_sem = np.zeros((semantic_data.shape[2], semantic_data.shape[3]))
        for i in range(semantic_data.shape[1]):
            modified_sem[semantic_data[0,i] == 1] = i + 1

        

    # Convert data to numpy arrays and adjust data types
    image_data = image_data.numpy().astype(np.int32)
    semantic_data = semantic_data.numpy().astype(np.int32)
    bounding_data = bounding_data.numpy().astype(np.int32)

    # Set a common size for all images (e.g., 200x200 pixels)
    common_size = (200, 200)

    # Create a figure with four subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        
    # Plot the main image on the first subplot
    main_img = np.moveaxis(image_data[0], 0, -1)
    axs[0, 0].imshow(main_img)
    axs[0, 0].set_title('Main')
    axs[0, 0].axis('off')

    # Plot the semantic data on the second subplot
    if not isinstance(modified_sem,np.ndarray):
        axs[0, 1].imshow(semantic_data[0, 0])
    else:
        axs[0, 1].imshow(modified_sem)
        
    axs[0, 1].set_title('Semantic')
    axs[0, 1].axis('off')

    # Extract and plot the bounded image on the third subplot
    bounded_img = np.moveaxis(crop_obj_using_bbox(image_data[0],  bounding_data[0, rand_obj_idx]),0, -1)
    print("Bounds shape:",bounded_img.shape)
    
    axs[1, 0].imshow(bounded_img)
    axs[1, 0].set_title('Bounded Image')
    axs[1, 0].axis('off')

    # Add a textual description as the fourth subplot
    description = f"Label: {label}\nQuestionnaire: {questionnaire}"
    axs[1, 1].text(0.5, 0.5, description, fontsize=12, ha='center', va='center')
    axs[1, 1].set_title('Label + Questionnaire')
    axs[1, 1].axis('off')
    
    if poses != None:
        print(f"{s} POSE: ", poses[0, rand_obj_idx])
    else:
        print(f"{s} POSE: ", poses)
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
        
    # Saves to file for users that are in terminal
    plt.savefig(f"./tests/test_output/{s}_dataloader_test.png") 

if __name__ == "__main__":
    main()