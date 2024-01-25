import os
import matplotlib
import torch
import numpy as np
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torch.utils.data import Dataset
import scipy.io as scp
import json
import time
import cv2

from .SingleRealPose import compute_marker_board_center

# Input Shape
# n x m x q x w x h
# n = Number of total images
# q = 3 : Color slots for RGB
# w = Width of the Image
# h = Height of the image

# Semantic Label Output Shape
# n x m x w x h
# n = Number of total images
# m = Total number of objects in the current images
# w = Width of the Image
# h = Height of the image

# Detection Bounds Output Shape
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


LABEL_FILE_NAME = "name.txt"
QUESTIONNAIRE_FILE_NAME = "questionnaire.txt"    
    
    
# This works for both syntetic and real split
# m = 1 : Total number of objects in the current images
class FewSOLDataloader(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        
        self.ANGLE_COUNT = 9
        
        self.classes = []
        
        # has each class name to all its indexes in the dataloader
        self.classes_to_idxs = {}

        # This is made to create an index for each different image
        self.objects = []
        for filename in os.listdir(self.dataset_dir):
            dir = os.path.join(self.dataset_dir, filename)
            if os.path.isdir(dir):
                
                cur_class = None
                with open(os.path.join(dir, LABEL_FILE_NAME), "r") as r:
                    cur_class = r.read().strip()
                    if cur_class not in self.classes:
                        self.classes.append(cur_class)
                        self.classes_to_idxs[cur_class] = []
                
                # Append each individaul objects number to the list
                for i in range(self.ANGLE_COUNT):
                    obj_path = os.path.join(dir, "{:06d}".format(i))
                    
                    # Adds to the class to idx dictonary
                    self.classes_to_idxs[cur_class].append(len(self.objects))
                    # Adds to the object path list
                    self.objects.append(obj_path)

        # Finds the width and height from the first image
        image = read_image(
            self._getColorFromIdx(0),
            ImageReadMode.RGB,
        )

        self.w = image.shape[1]
        self.h = image.shape[2]

        self.transform = transform
    
    # Returns a list of indexs that contains a specific class
    # Returns None on invalid input
    def get_class_idx(self, class_identifier):
        if isinstance(class_identifier, int):
            if class_identifier >= len(self.classes) or class_identifier < 0:
                return None
            
            return self.classes_to_idxs[self.classes[class_identifier]]
        elif isinstance(class_identifier, str):
            if class_identifier not in self.classes_to_idxs:
                return None
            return self.classes_to_idxs[class_identifier]
        else:
            return None
    
    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        return self.get_idx(idx)
        
    def get_idx(self,
                idx,
                load_img:bool=True,
                load_mask:bool=True,
                load_bbox:bool=True,
                load_label:bool=True,
                load_que:bool=True,
                load_pose:bool=True,
                ):         
        label_file, questionnaire_file = self._getInfoFromIdx(idx)
        
        descriptions = None
        label = None
        questionnaire = None
        
        if load_label:
            with open(label_file, "r") as r:
                label = [r.read().strip()]
        if load_que:
            with open(questionnaire_file, "r") as r:
                questionnaire = [r.read().strip()]
            
            
        semantic_data = None 
        bounding_data = None 
        
        img_data = torch.zeros((1, 3, self.w, self.h), dtype=torch.float64) if load_img else None

        if load_bbox:
            # Semantic data is needed to calculate bounding data
            semantic_data = torch.zeros((1, 1, self.w, self.h), dtype=torch.float64)
            bounding_data = torch.zeros((1, 1, 4), dtype=torch.float64)
        elif load_mask:
            semantic_data = torch.zeros((1, 1, self.w, self.h), dtype=torch.float64)
            
        poses = torch.zeros(1, 1, 4, 4) if load_pose else None
        

        color_file_name = self._getColorFromIdx(idx)
        
        if load_img:
            # Loads the color images
            image = read_image(
                color_file_name, ImageReadMode.RGB
            )
            if self.transform:
                image = self.transform(image)
            img_data[0] = image

        if load_bbox or load_mask:
            # Loads segmentation label
            semantic_label = read_image(
                self._getLabelImgFromIdx(idx), ImageReadMode.GRAY
            )
            semantic_data[0] = semantic_label[0]
        
        if load_bbox:
            right, left, bottom, top = calculate_bound_box(semantic_label[0])
            bounding_data[0] = torch.tensor([bottom, right, top - bottom, left - right])
        
        
        # Loads pose information
        # Loads mat file
        if load_pose:
            mat_file = self._getMatFromIdx(idx)
            mat_data = scp.loadmat(mat_file)

            # This handles synthetic data poses
            if "object_poses" in mat_data:
                # Appends the pose
                poses[0] = torch.Tensor(mat_data["object_poses"].squeeze())
            # This handles real object data poses
            elif "joint_position" in mat_data:
                # Creates camera relative object poses
                mat_data = compute_marker_board_center(mat_data)
                
                # Appends the pose
                poses[0] = torch.Tensor(mat_data["center"])  

        return img_data, semantic_data, bounding_data, label, questionnaire, color_file_name, poses
    
    # Outputs (n, h, w) depth torch tensor
    def get_depth(self, idx):
        depth_img_name = self._getDepthImgFromIdx(idx)
        
        image = torch.tensor(cv2.imread(depth_img_name)).permute(2,0,1)
        
        return image[0].unsqueeze(0)
    

    def _getColorFromIdx(self, idx: int):
        return f"{self.objects[idx]}-color.jpg"

    def _getLabelImgFromIdx(self, idx: int):
        return f"{self.objects[idx]}-label-binary.png"
    
    def _getDepthImgFromIdx(self, idx: int):
        return f"{self.objects[idx]}-depth.png"

    def _getMatFromIdx(self, idx: int):
        return f"{self.objects[idx]}-meta.mat"
    
    # Returns class, questionnaire
    def _getInfoFromIdx(self, idx: int):
        folder = self.objects[idx][:-6]
        return  folder + LABEL_FILE_NAME, folder + QUESTIONNAIRE_FILE_NAME
    
    
    
        
    

# m = 10 : Total number of objects in the image
class RealClutterDataLoader(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir =  dataset_dir
        
        self.ocid_ref_dir = os.path.join(self.dataset_dir, "..", "OCID_ref")
        
        self.OBJ_COUNT = 10
        
        self.classes = []
        
        # has each class name to all its indexes in the dataloader
        self.classes_to_idxs = {}
        
        # Instantiated only if OCID_ref path exists
        self.mapping = None

        self.REFERENCE_FILE_NAME = "referring_expressions.txt"

        # This is made to create an index for each different image
        #
        # File name format 
        #   name_seqxx_objyy
        # 
        # We want to combine all the objects in all sequence into one entry
        
        self.sequences = []
        for filename in os.listdir(self.dataset_dir):
            dir = os.path.join(self.dataset_dir, filename)
            if os.path.isdir(dir):
                # Only adds each sequnence
                if dir[:-2] not in self.sequences:
                    self.sequences.append(dir[:-2])
                
                with open(os.path.join(dir, LABEL_FILE_NAME), "r") as r:
                    cur_class = r.read().strip()
                    
                    if cur_class not in self.classes:
                        self.classes.append(cur_class)
                        # Adds to class idx if it doesnt exist
                        self.classes_to_idxs[cur_class] = []
                                            
                    # Adds to the class to idx dictonary
                    self.classes_to_idxs[cur_class].append(self.sequences.index(dir[:-2]))
                    
                    
                        
        
        # Finds the width and height from the first image
        image = read_image(
            self._getFiles(0, 1)[0],
            ImageReadMode.RGB,
        )

        self.w = image.shape[1]
        self.h = image.shape[2]

        self.transform = transform
        
        
        if os.path.isdir(self.ocid_ref_dir):
            self.AddRefDesc()
    
    def CreateRefToObjMapper(self):
        print("Creating OCID Ref to Obj mapper")
        self.ocid_ref_classes = set()
        
        # Adds all classes for ocid_ref into a set
        for filename in os.listdir(self.ocid_ref_dir):
            file = os.path.join(self.ocid_ref_dir, filename)
            # Only loads the relevant json files
            if len(file) > 4 and file[-4:] != "json":
                continue
            # Loads data from JSON file
            with open(os.path.join(self.ocid_ref_dir, file), "r") as r:
                data = json.load(r)
            for cur_data in data.values():
                self.ocid_ref_classes.add(cur_data['class'])
                        
        # Stores the mapping between ref to obj
        # Adding some values manually because of lack of matching
        self.mapping = {
            "racquetball": "ball",
            "softball": "baseball ball",
            "kleenex": "tissue box",
            "chips_can": "pringles can",
            "food_can": "soup can",
            "pudding_box": "jello box",
            "gelatin_box": "jello box",
            "food_box": "cereal box",
            "master_chef_can": "coffee can",
            "nine-hole_peg_test": "wood box",
        }

        
        for ref_class in self.ocid_ref_classes:
            
            ref_words = set(ref_class.split("_"))
            
            closest_class = None
            max_class = 0
            for obj_class in self.classes:
                obj_words = set(obj_class.split(" "))
                
                # Find the amount of similar words
                sim = len(ref_words.intersection(obj_words))
                
                # Only overwrite if its the more similar
                if sim > max_class:
                    max_class = sim
                    closest_class = obj_class
            
            # Stores the mapping
            if ref_class in self.mapping:
                # Do nothin because it was manually added
                continue
            # Stores the closest mapping
            elif closest_class != None:
                self.mapping[ref_class] = closest_class 
            # Prints classes
            else:
                self.mapping[ref_class] = ""
                print("Cant find ref class:", ref_class)

            
                
    # Adds the OCID ref descriptions to the images
    def AddRefDesc(self):
        # Creates a mapping between ref and obj
        self.CreateRefToObjMapper()
        
        print("Parsing OCID ref into folders...")
        
        # Image name is the only linking trait from ocid-ref to FewSOL ocid
        img_to_folder = {}
        image_len = len("result_2018-08-24-14-37-40")
        
        # Creates a dictionary of image name to folder name
        for filename in os.listdir(self.dataset_dir):
            dir = os.path.join(self.dataset_dir, filename)
            if not os.path.isdir(dir):
                continue
             
            for imgname in os.listdir(dir):
                # Remove old questionnaires
                description_path = os.path.join(dir, QUESTIONNAIRE_FILE_NAME)

                if os.path.isfile(description_path):
                    os.remove(description_path)
                        
                # Adds the file name to the dictionary
                if imgname[-3:] == "jpg":
                    label_path = os.path.join(dir, LABEL_FILE_NAME)
                    # OCID Ref uses "_" instead of whitespace that is in "name.txt"
                    with open(label_path, 'r') as f:
                        key_name = imgname[:image_len] + "|"+ f.readline().strip()
                    img_to_folder[key_name] = dir
                    break
        
        # Keeps tracks fo ad                          
        added = set()
        total_done = 0
        total = len(img_to_folder)
        
        matched_ref = set()
        
        for filename in os.listdir(self.ocid_ref_dir):
            file = os.path.join(self.ocid_ref_dir, filename)
            # Only loads the relevant json files
            if len(file) > 4 and file[-4:] != "json":
                continue
            
            # Loads data from JSON file
            with open(os.path.join(self.ocid_ref_dir, file), "r") as r:
                data = json.load(r)
            
            # Loops through the data in the JSON file
            for cur_data in data.values():
                
                # Gets the key for the data location dictionary
                image_name = cur_data["scene_path"].split("/")[-1][:-4]
                key_name = image_name + "|" + self.mapping[cur_data['class']]
                
                if key_name in img_to_folder:
                    
                    # Checks to make sure the data taken matches
                    beginning = "_".join(cur_data["sequence_path"].split("/"))
                    if img_to_folder[key_name].split("/")[-1][:len(beginning)] != beginning:
                        continue
                    
                    # Gets relevant pahts
                    description_path = os.path.join(img_to_folder[key_name], QUESTIONNAIRE_FILE_NAME)
                    label_path = os.path.join(img_to_folder[key_name], LABEL_FILE_NAME)

                    # Makes sure that the labels match
                    with open(label_path, 'r') as f:
                        label = f.readline()
                    if self.mapping[cur_data['class']] != label:       
                        continue
                    
                                        
                    
                    # Counts the total done
                    if not description_path in added:
                        total_done += 1
                    
                    # Writes the description into the file
                    with open(description_path, 'a') as f:
                        f.write(cur_data['sentence']  + "\n")
                    
                    # Adds the questionnaire to the path so that we know it was added
                    added.add(description_path)
                    
                    matched_ref.add(cur_data['class'])
                    
                    
        
        print(f"Added description into {total_done}/{total} of images")
        
        # This was used to see what classes weren't matched
        # print( self.ocid_ref_classes- matched_ref)
    
    # Returns a list of indexs that contains a specific class
    # Returns None on invalid input
    def get_class_idx(self, class_identifier):
        if isinstance(class_identifier, int):
            if class_identifier >= len(self.classes) or class_identifier < 0:
                return None
            
            return self.classes_to_idxs[self.classes[class_identifier]]
        elif isinstance(class_identifier, str):
            if class_identifier not in self.classes_to_idxs:
                return None
            return self.classes_to_idxs[class_identifier]
        else:
            return None
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.get_idx(idx)
        
    def get_idx(self,
                idx,
                load_img:bool=True,
                load_mask:bool=True,
                load_bbox:bool=True,
                load_label:bool=True,
                load_que:bool=True,
                load_pose:bool=True,
                ):     
                      
        descriptions = [] if load_que else None
        label = [] if load_label else None
        
        semantic_data = None 
        bounding_data = None 
        
        img_data = torch.zeros((1, 3, self.w, self.h), dtype=torch.float64) if load_img else None

        if load_bbox:
            # Semantic data is needed to calculate bounding data
            semantic_data = torch.zeros((1, self.OBJ_COUNT, self.w, self.h), dtype=torch.float64)
            bounding_data = torch.zeros((1, self.OBJ_COUNT, 4), dtype=torch.float64)
        elif load_mask:
            semantic_data = torch.zeros((1, self.OBJ_COUNT, self.w, self.h), dtype=torch.float64)

        
        # OCID does not contain poses
        poses = None
        
        # The last object in the sequence has the complete image
        color_file = self._getFiles(idx, self.OBJ_COUNT)[0]
            
        if load_img:
            # Loads the color images
            image = read_image(
                color_file, ImageReadMode.RGB
            )
            if self.transform:
                image = self.transform(image)
        
            img_data[0] = image

        for obj_i in range(0, self.OBJ_COUNT):
            
            # Gets the label and semantic info for the ith object
            _, mat_file, label_file, ques_file = self._getFiles(idx, obj_i + 1)
            
            if load_label:
                # Appends the obj label to list
                with open(label_file, "r") as r:
                    label.append(r.readline().strip())
            
            if load_que:
                # Appends the questionnaire to list if exists 
                if ques_file == None:
                    descriptions.append(None)
                else:  
                    with open(ques_file, "r") as r:
                        descriptions.append(r.readline().strip())
            
            if load_bbox or load_mask:
                # Loads segmentation label from mat file  
                matData = scp.loadmat(mat_file)
                            
                # Creates a 1 and 0 segmentation array
                objectID = matData["object_id"][0,0]
                semantic_data[0, obj_i] = torch.tensor(matData["label"])
                semantic_data[0, obj_i][semantic_data[0, obj_i] != objectID] = 0
                semantic_data[0, obj_i][semantic_data[0, obj_i] == objectID] = 1

                if load_bbox:
                    right, left, bottom, top = calculate_bound_box(semantic_data[0, obj_i])
                    bounding_data[0, obj_i] = torch.tensor([bottom, right, top - bottom, left - right,])
        
        return img_data, semantic_data, bounding_data, label, descriptions, color_file, poses
    
    # This returns the mat and color file
    def _getFiles(self, seq_num, obj_num):
        obj_path = "{:s}{:02d}".format(self.sequences[seq_num], obj_num)
        
        color = None
        mat = None
        label = None
        questonnaire = None
        
        for filename in os.listdir(obj_path):
            if filename[-9:-4] == "color":
                color = os.path.join(obj_path, filename)
            elif filename[-3:] == "mat":
                mat = os.path.join(obj_path, filename)
            elif filename == LABEL_FILE_NAME:
                label = os.path.join(obj_path, filename)
            elif filename == QUESTIONNAIRE_FILE_NAME:
                questonnaire = os.path.join(obj_path, filename)
                
        # Questionnaire file isn't always given
        if color == None or mat == None or label == None:
            raise Exception("_getFiles: color, mat or name file doesn't exist") 

                
        return color, mat, label, questonnaire


# m = variable: Objecst in the image
class GooogleClutterDataloader(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.ANGLE_COUNT = 7
        self.COLOR_PALLETE_FILE = "palette.txt"
        self.MAPPPER_FILE = "syn_google_scenes_data_mapper.json"
        self.FILE_TO_CLASS_MAPPER = "file_class_mapper.json"
        
        self.dataset_dir =  dataset_dir
        
        # has each class name to all its indexes in the dataloader
        self.classes_to_idxs = {}
        
        # This is made to create an index for each different image
        self.objects = []
        
        # Creates mapper
        current_folder = "/".join(__file__.split("/")[:-1])        
        # Loads object to label mapper
        with open(os.path.join(current_folder, self.MAPPPER_FILE), "r") as r:
            self.mapper = json.load(r)
            
        # Holds the list of classes that each folder has
        # This speeds up the class_to_idx speed
        with open(os.path.join(current_folder, self.FILE_TO_CLASS_MAPPER), 'r') as f:
            self.folder_to_classes = json.load(f)
        
        # Creates classes list
        self.classes = [self._removeUnderscore(label) for label in  list(set(self.mapper.values()))]
                

        for foldername in os.listdir(os.path.join(self.dataset_dir, "train")):
            dir = os.path.join(self.dataset_dir, "train", foldername)
            if not os.path.isdir(dir):
                continue
            
            for imagename in os.listdir(dir):
                if imagename[:3] != "rgb":
                    continue
                
                self.objects.append(os.path.join(dir, imagename))
                    
                # Adding to class to idx -------- 
                    
                # If folder is not saved that add the data to the dictionary
                if foldername not in self.folder_to_classes:
                    _, mat_file = self._getDataFromInt(len(self.objects) - 1)
                    # Loads mat file
                    mat_data = scp.loadmat(mat_file)
                    # Gets google object name
                    object_names = [x.strip() for x in mat_data["object_names"]]
                    # Gets accurate label names from syntetic dataset
                    labels = [self._removeUnderscore(self.mapper[x]) for x in object_names]
                    
                    self.folder_to_classes[foldername] = labels
                        
                
                labels = self.folder_to_classes[foldername]
                for l in labels:
                    if l not in self.classes_to_idxs:
                        self.classes_to_idxs[l] = []
                            
                    self.classes_to_idxs[l].append(len(self.objects) - 1)
                        
                # ---------
        
        # Finds the width and height from the first image
        image = read_image(
            self.objects[0],
            ImageReadMode.RGB,
        )

        self.w = image.shape[1]
        self.h = image.shape[2]

        self.transform = transform
        
        # Converts palette file into a numpy array
        
        # Instantitates numpy 
        self.color_palette = np.zeros((0,3))
        
        # Reads from file
        with open(os.path.join(self.dataset_dir, self.COLOR_PALLETE_FILE), "r") as r:
            lines = r.readlines()
            
            # Puts every line into a new index
            for line in lines:
                color_arr = np.zeros((1,3))
                nums = line.strip().split(" ")
                for i in range(3):
                    color_arr[0,i] = int(nums[i])
                
                # Concats line into palette numpy array
                self.color_palette = np.concatenate((self.color_palette,color_arr), axis=0)
        
        
        
            
    
    # Returns a list of indexs that contains a specific class
    # Returns None on invalid input
    def get_class_idx(self, class_identifier):
        if isinstance(class_identifier, int):
            if class_identifier >= len(self.classes) or class_identifier < 0:
                return None
            
            return self.classes_to_idxs[self.classes[class_identifier]]
        elif isinstance(class_identifier, str):
            if class_identifier not in self.classes_to_idxs:
                return None
            return self.classes_to_idxs[class_identifier]
        else:
            return None    
    
    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        return self.get_idx(idx)
        
    def get_idx(self,
                idx,
                load_img:bool=True,
                load_mask:bool=True,
                load_bbox:bool=True,
                load_label:bool=True,
                load_que:bool=True,
                load_pose:bool=True,
                ):
        # Questionnaire doesnt exist
        questionnaire = None

        color_file = self.objects[idx]        
        
        # Gets file locations
        seg_file, mat_file = self._getDataFromInt(idx)
        
        # Loads mat file
        mat_data = scp.loadmat(mat_file)
        # Gets google object name
        object_names = [x.strip() for x in mat_data["object_names"]]
        
        
        label = [self._removeUnderscore(self.mapper[x]) for x in object_names] if load_label else None

        img_data = None
        if load_img:
            img_data = torch.zeros((1, 3, self.w, self.h), dtype=torch.float64)
            # Loads the color image
            image = read_image(
                color_file, ImageReadMode.RGB
            )
            if self.transform:
                image = self.transform(image)
            img_data[0] = image
        
        
        semantic_data = None 
        bounding_data = None 
        poses = None
        
        if load_bbox:
            # Semantic data is needed to calculate bounding data
            semantic_data = torch.zeros((1, len(object_names), self.w, self.h), dtype=torch.float64)
            bounding_data = torch.zeros((1, len(object_names), 4), dtype=torch.float64)
        elif load_mask:
            semantic_data = torch.zeros((1, len(object_names), self.w, self.h), dtype=torch.float64)
            
        poses = torch.zeros((1, len(object_names), 4, 4)) if load_pose else None
        
        for i in range(len(object_names)):
            if load_mask or load_bbox:
                # Loads segmentation label
                semantic_rgb_label = read_image(seg_file, ImageReadMode.RGB)
                # Gets the palette index
                sem_color = self.color_palette[mat_data[object_names[i]][0][0]]
                
                # Conditional in order to get the indexes where the color matches
                color_cond_idx = (semantic_rgb_label[0 ,:, :] == sem_color[0]) & \
                    (semantic_rgb_label[1 ,:, :] == sem_color[1]) & \
                    (semantic_rgb_label[2 ,:, :] == sem_color[2])
                
                # Sets color matching index to 1
                semantic_data[0,i, color_cond_idx] = 1
        
            if load_bbox:
                right, left, bottom, top = calculate_bound_box(semantic_data[0,i])
                bounding_data[0, i] = torch.tensor([bottom, right, top - bottom, left - right])
        
        if load_pose:  
            # Loading poses
            poses[0] = torch.Tensor(mat_data["object_poses"]).permute((2,0,1))

        return img_data, semantic_data, bounding_data, label, questionnaire, color_file, poses

    def _getDataFromInt(self, idx: int):
        num = self.objects[idx][-9:-4]
        seg_loc = os.path.join(os.path.dirname(self.objects[idx]), f"segmentation_{num}.png")
        mat_loc = os.path.join(os.path.dirname(self.objects[idx]), f"meta_{num}.mat")
        
        if not os.path.isfile(seg_loc) or not os.path.isfile(mat_loc):
            raise Exception("_getDataFromInt: mat or segmenation file doesn't exist") 
            
            
        return seg_loc, mat_loc
    
    def _removeUnderscore(self, label):
        return label.replace("_", " ")


# Calculates the bounding box information from semantic label
def calculate_bound_box(semanticLabel):
    top, bottom, right, left = 0,0,0,0 
    
    # Gets positions where the data is not 0
    contains_pos = torch.argwhere(semanticLabel != 0)
    
    # Handles empty case
    if contains_pos.shape[0] == 0:
        return right, left, bottom, top
    
    # min index where element is not zero
    mins = torch.min(contains_pos, axis = 0)
    bottom = mins[0][1]
    right = mins[0][0]
    
    # Max index where element is not zero
    maxs = torch.max(contains_pos, axis = 0)
    top = maxs[0][1]
    left = maxs[0][0]
    
    return right, left, bottom, top
