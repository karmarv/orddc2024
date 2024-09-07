import random
import os, json
import argparse
import datetime

import cv2 as cv
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

# Configuration and path
random.seed(0) 

# Submission specific class identifiers
SVRDD_DAMAGE_CATEGORIES={
        "D00":          {"id": 1, "label_id": 0, "name": "D00", "color": [220, 20, 60] , "submission_superid": 1, "description": "Longitudinal Crack"}, 
        "D10":          {"id": 2, "label_id": 1, "name": "D10", "color": [0, 0, 142]   , "submission_superid": 2, "description": "Transverse Crack"}, 
        "D20":          {"id": 3, "label_id": 2, "name": "D20", "color": [0, 60, 100]  , "submission_superid": 3, "description": "Aligator Crack"}, 
        "D40":          {"id": 4, "label_id": 3, "name": "D40", "color": [0, 80, 100]  , "submission_superid": 4, "description": "Pothole"}
}

# Global Variables
GLOBAL_IMAGE_SEQ_ID=0               # Initialized
GLOBAL_ANNOT_SEQ_ID=0
EXPORT_CSV_DELIMITER=","
GLOBAL_LABEL_HISTGM={}              # Defect code as keys and count as values

class FrameExtractor:
    """
    OpenCV utility to sample frames in a video 
    """
    def __init__(self, videopath):
        self.videopath   = videopath    
        self.cap         = cv.VideoCapture(videopath)
        self.fps         = self.cap.get(cv.CAP_PROP_FPS)
        self.width       = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height      = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        #print("FPS:{:.2f}, (Frames: {}, Duration {:.2f} s), \t Video:{} ".format(self.fps, self.frame_count, self.frame_count/self.fps, videopath))

    # Extract frame given the identifier
    def image_from_frame(self, frame_id):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
        _, img = self.cap.read()
        return img

def write_json_file(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(" "), 
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
        "id"            : image_id,
        "file_name"     : file_name,
        "width"         : image_size[0],
        "height"        : image_size[1],
        "date_captured" : date_captured,
        "license"       : license_id,
        "coco_url"      : coco_url,
        "flickr_url"    : flickr_url
    }
    return image_info

def create_annotation_info(annotation_id, image_id, category_id, area, bounding_box):
    annotation_info = {
        "id"            : annotation_id,
        "image_id"      : image_id,
        "category_id"   : category_id,   # defect label
        "iscrowd"       : 0,
        "area"          : area,          # float
        "bbox"          : bounding_box,  # [x,y,width,height]
        "segmentation"  : []             # [polygon]
    }
    return annotation_info

def copy_image_by_annotations(data_file_desc, data_dir, output_dir, output_split):
    
    global GLOBAL_IMAGE_SEQ_ID, GLOBAL_ANNOT_SEQ_ID
    imgs_dict = dict()
    gts_coco_labels = {"images":[], "annotations":[]}
    # create target directories by split name
    os.makedirs(os.path.join(output_dir, output_split, "images"), exist_ok=True)
    # read YOLO format train.txt files for image location
    data_files = pd.read_csv(data_file_desc, sep=',', header=None)[0].tolist()
    count = 0
    print("Processing {} images from {}".format(len(data_files), data_file_desc))
    # dataset/svrdd/train/{labels/*.txt, images/*.png}
    for src_img_file in tqdm(data_files):
        src_img_file = os.path.join(data_dir, src_img_file.replace('\\','/'))
        src_img_basename = os.path.basename(src_img_file)
        img_name, extension = os.path.splitext(src_img_basename)
        lbl_filepath = os.path.join(os.path.dirname(src_img_file), "..", "labels", "{}.txt".format(img_name))

        image_info = {}
        # Images unique id in gts_coco_labels["images"] dict
        if src_img_basename in imgs_dict.keys():
            image_info = imgs_dict[src_img_basename]
            image_id   = image_info["id"]
        else:
            image_id   = GLOBAL_IMAGE_SEQ_ID = GLOBAL_IMAGE_SEQ_ID + 1
            # COCO Format Image entry if not in dict for image metadata reuse
            frame_extractor = FrameExtractor(src_img_file)
            image_info = create_image_info(image_id, src_img_basename, image_size=[frame_extractor.width, frame_extractor.height])
            imgs_dict[src_img_basename] = image_info
            gts_coco_labels["images"].append(image_info)
            # Copy unique image to class subfolder
            dst_file_path = os.path.join(output_dir, output_split, "images", src_img_basename)
            # Copy only if unavailable to avoid duplicate run
            if not os.path.isfile(dst_file_path):
                copyfile(src_img_file, dst_file_path)
                        
        # Annotations from YOLO bounding box annotations
        if os.path.isfile(lbl_filepath):  # reading csv file 
            # (x, y), w, and h as the centre coordinate, width, and height of the bounding box
            lbl_df = pd.read_csv(lbl_filepath, delimiter=' ', names=["label_id", "x", "y", "w", "h"], header=None)
            lbl_seen = {}
            img_w, img_h = image_info["width"], image_info["height"]
            for index, ann in lbl_df.iterrows():
                lbl_id = ann["label_id"]
                # 6 0.354492 0.649902 0.294922 0.024414
                # 5 0.342773 0.602051 0.267578 0.079102
                xc, yc, w, h = ann["x"]*img_w, ann["y"]+img_h, ann["w"]*img_w, ann["h"]*img_h
                for key, val in SVRDD_DAMAGE_CATEGORIES.items():
                    if lbl_id == val["label_id"]:
                        lbl_seen[key] = ann["label_id"]
                        coco_label_id = val["id"]
                        # scale the bbox as per [Top-Left X, Top-Left Y, Width, Height] format.
                        bb_left, bb_top       = int(xc-(w/2)), int(yc-(h/2))
                        bb_width, bb_height   = int(w), int(h)
                        area      = bb_width * bb_height
                        box       = [bb_left, bb_top, bb_width, bb_height]
                        # Add annotations to the dict
                        ann_id   = GLOBAL_ANNOT_SEQ_ID = GLOBAL_ANNOT_SEQ_ID + 1
                        ann_info  = create_annotation_info(ann_id, image_id, coco_label_id, area, box)
                        gts_coco_labels["annotations"].append(ann_info)

            count = count + 1           
    return gts_coco_labels


"""
Get the image id and annotation sequence id for next dataset
"""
def get_max_ids_annotations(coco):
    max_image_id = 0
    for entry in coco["images"]:
        if int(entry['id'])>max_image_id:
            max_image_id = int(entry['id'])
    max_anns_id = 0
    for entry in coco["annotations"]:
        if int(entry['id'])>max_anns_id:
            max_anns_id = int(entry['id'])
    return max_image_id+1, max_anns_id+1

def get_coco_base_annotations(coco_json):
    with open(coco_json) as annf:
        coco_extend = json.load(annf)
        len_images, len_labels = get_max_ids_annotations(coco_extend)
    return coco_extend, len_images, len_labels

def get_coco_metadata():
    coco_meta = { 
        "images": [], 
        "annotations": [],
        "categories": [
            {
            "id": 1,
            "name": "D00",
            "supercategory": 1
            },
            {
            "id": 2,
            "name": "D10",
            "supercategory": 2
            },
            {
            "id": 3,
            "name": "D20",
            "supercategory": 3
            },
            {
            "id": 4,
            "name": "D40",
            "supercategory": 4
            }
        ]
    }
    return coco_meta


"""
Usage: 
- time python svrdd_cocofy.py --svrdd-home /data/workspace/orddc/data/svrdd/SVRDD_YOLO/  --output-dir ./rdd2022/coco_plus/ --coco-ext ./rdd2022/coco_plus/annotations/rdd2022_annotations.json

(rdd)$ tree -L 2 rdd2022/svrdd/SVRDD_YOLO/
rdd2022/svrdd/SVRDD_YOLO/
├── Chaoyang
│   ├── images
│   └── labels
├── Dongcheng
│   ├── images
│   └── labels
├── Fengtai
│   ├── images
│   └── labels
├── Haidian
│   ├── images
│   └── labels
├── test.txt
├── train.txt
├── val.txt
└── Xicheng
    ├── images
    └── labels

Annotations:
- categories: Stores the class names for the various object types in the dataset. Note that this toy dataset only has one object type.
- images: Stores the dimensions and file names for each image.
- annotations: Stores the image IDs, category IDs, and the bounding box annotations in [Top-Left X, Top-Left Y, Width, Height] format.
"""
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Convert SVRDD annotations to ImageNet dataset')
    parser.add_argument("--svrdd-home",      type=str,  default="/data/workspace/orddc/data/svrdd/SVRDD_YOLO/",   help="SVRDD folder path")
    parser.add_argument("--output-dir",    type=str,  default="./rdd2022/coco_plus/",  help="Output directory for ImageNet like dataset")
    parser.add_argument("--coco-ext",    type=str,  default="./rdd2022/coco_plus/annotations/rdd2022_annotations.json",  help="Extend and append new data to existing coco dataset json file")
    # 
    args=parser.parse_args()

    in_coco_train = args.coco_ext
    # Empty placeholders 
    coco_extend = { "images": [], "annotations": [] }
    train_ann_dict = get_coco_metadata()       
    # Read if the input COCO file exists for extension
    if os.path.isfile(in_coco_train):
        coco_extend, max_id_images, max_id_labels = get_coco_base_annotations(in_coco_train)
        print("Source Categories:{}".format(coco_extend["categories"]))  
        # Increment the ImageID and AnnID for extension
        GLOBAL_IMAGE_SEQ_ID, GLOBAL_ANNOT_SEQ_ID = max_id_images + 100, max_id_labels + 100
        print("Source Max Images: {}, Labels {}".format(GLOBAL_IMAGE_SEQ_ID, GLOBAL_ANNOT_SEQ_ID))
        
        # Add the read training dataset to the coco-extend labels
        train_ann_dict["images"]      = coco_extend["images"]      + train_ann_dict["images"]
        train_ann_dict["annotations"] = coco_extend["annotations"] + train_ann_dict["annotations"]

    # Load annotations per split and add it to base RDD (train+val in rddd2022_annotation.json) set (for one final training)
    input_splits = ["val", "train"]
    output_split = "train"
    img_count = 0
    for idx, split in enumerate(input_splits):
        print("{}.) {}".format(idx, split))
        data_files = os.path.join(args.svrdd_home, "{}.txt".format(split.lower()))
        # dataset/svrdd/train/{labels/*.txt, images/*.png} add only to train set
        coco_data = copy_image_by_annotations(data_files, args.svrdd_home, args.output_dir, output_split)
        # Accumulate project annotations in train
        if len(coco_data["annotations"])>0:
            train_ann_dict["images"]      = train_ann_dict["images"]      + coco_data["images"]
            train_ann_dict["annotations"] = train_ann_dict["annotations"] + coco_data["annotations"]    
            img_count = img_count + len(coco_data["images"])
            # Add categories information before writing to JSON dump
            coco_data["categories"] = train_ann_dict["categories"]
            write_json_file(os.path.join(args.output_dir, 'annotations', 'svrdd_{}.json'.format(split)), coco_data)
        print("Extracted COCO Images: {} in split {}".format(len(coco_data["images"]), split))

    # Write the extended JSON annotations in COCO format
    write_json_file(os.path.join(args.output_dir, 'annotations', 'rddd2022_annotation_svrdd_train_val_ext.json'), train_ann_dict)
    print("Accumulated COCO Images: {}".format(len(train_ann_dict["images"])))
    print("Accumulated Annotations: {}".format(len(train_ann_dict["annotations"])))