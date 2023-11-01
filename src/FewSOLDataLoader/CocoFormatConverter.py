import numpy as np
import json
import os
import torch


OUTPUT_NAME = "coco_annotation.json"

# Based off https://cocodataset.org/#format-data
# Also based off https://mmdetection.readthedocs.io/en/latest/user_guides/train.html

# Overwrite is if it will recreate even if it already exists
def CocoConverter(dataset, root, overwrite=False):
    folder_name = root.split("/")[-1]
    print(f"Starting to convert to coco {folder_name}")
    
    output_file = os.path.join(root, OUTPUT_NAME)
    
    # If file already exists then return
    if not overwrite and os.path.isfile(output_file):
        return output_file
    
    annotations = []
    images = []
    obj_count = 0
    
    # Creates categories list
    categories = [{
        'id': i,
        'name': x
    } for i, x in enumerate(dataset.classes)]
    
    
    for idx in range(len(dataset)):
        # Gets data from dataset
        img_data, semantic_data, bounding_data, label, questionnaire, filename = dataset[idx]
        
        # Appends image data
        images.append(dict(id=idx, file_name=filename, height=dataset.w, width=dataset.h))

        # appends all objects data
        for i in range(bounding_data.shape[1]):   
            # Gets the index for the semantic data
            semantic_pos = torch.argwhere(semantic_data[0, i] != 0)
            
            # Converts the semantion into the right format
            #poly = [(semantic_pos[i, 0] + 0.5, semantic_pos[i, 1] + 0.5) for i in range(semantic_pos.shape[0])]
            #poly = [int(p) for x in poly for p in x]
            poly = semantic_pos.transpose(1, 0).flatten().tolist()
            
            # Creates a dictionary of all the required values
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=dataset.classes.index(label[i]) ,
                bbox= bounding_data[0,i].tolist(),
                area= int(bounding_data[0, i, 2] * bounding_data[0, i, 3]),
                segmentation=[poly],
                iscrowd=0
            )

            # Appends data
            annotations.append(data_anno)
            # Iterate object
            obj_count += 1
            
        if idx % 100 == 0:
            print(f"{folder_name}: {idx}/{len(dataset)}")
                    
        
        
    print(f"Finished {folder_name}")
            
    # Puts the data into a dictionary
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories
    )
        
        
    # Saves the data into a json file
    with open(output_file, "w") as f:
        json.dump(coco_format_json, f, indent = 4)
    
    return output_file
        
        