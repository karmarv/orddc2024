import random
import os, json
import argparse
import datetime

import pandas as pd
from shutil import copyfile
from tqdm import tqdm

# Configuration and path
random.seed(0) 

# Submission specific class identifiers
SVRDD_DAMAGE_CATEGORIES={
        "D00":          {"id": 0, "name": "D00", "color": [220, 20, 60] , "submission_superid": 1, "description": "Longitudinal Crack"}, 
        "D10":          {"id": 1, "name": "D10", "color": [0, 0, 142]   , "submission_superid": 2, "description": "Transverse Crack"}, 
        "D20":          {"id": 2, "name": "D20", "color": [0, 60, 100]  , "submission_superid": 3, "description": "Aligator Crack"}, 
        "D40":          {"id": 3, "name": "D40", "color": [0, 80, 100]  , "submission_superid": 4, "description": "Pothole"},
        "Manhole":      {"id": 4, "name": "D40", "color": [50, 80, 100] , "submission_superid": 5, "description": "Manhole Cover"},
        "LongPatch":    {"id": 5, "name": "D40", "color": [50, 80, 100] , "submission_superid": 5, "description": "Longitudinal Patch"},
        "TranPatch":    {"id": 6, "name": "D40", "color": [50, 80, 100] , "submission_superid": 5, "description": "Transverse Patch"}
}


def copy_image_by_annotations(data_file_desc, data_dir, output_dir):
    data_files = pd.read_csv(data_file_desc, sep=',', header=None)[0].tolist()
    # create target directories by label name
    for cls_name in SVRDD_DAMAGE_CATEGORIES.keys():
        os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)
    count = 0
    print("Processing {} images from {}".format(len(data_files), data_file_desc))
    # dataset/svrdd/train/{labels/*.txt, images/*.png}
    for src_img_file in tqdm(data_files):
        src_img_file = os.path.join(data_dir, src_img_file.replace('\\','/'))
        img_name, extension = os.path.splitext(os.path.basename(src_img_file))
        lbl_filepath = os.path.join(os.path.dirname(src_img_file), "..", "labels", "{}.txt".format(img_name))
        if os.path.isfile(lbl_filepath):  # reading csv file 
            # (x, y), w, and h as the centre coordinate, width, and height of the bounding box
            lbl_df = pd.read_csv(lbl_filepath, delimiter=' ', names=["label_id", "x", "y", "w", "h"], header=None)
            lbl_seen = {}
            for index, ann in lbl_df.iterrows():
                lbl_id = ann["label_id"]
                for key, val in SVRDD_DAMAGE_CATEGORIES.items():
                    if lbl_id == val["id"]:
                        lbl_seen[key] = ann["label_id"]
            #print(lbl_seen)
            # Copy images to corresponding labels dir
            for lbl in lbl_seen.keys():
                # Copy image to class subfolder
                dst_file_path = os.path.join(output_dir, lbl, os.path.basename(src_img_file))
                # Copy if unavailable to avoid duplicate
                if not os.path.isfile(dst_file_path):
                    copyfile(src_img_file, dst_file_path)
            count = count + 1           
    return count

"""
Usage: 
- time python svrdd_imgnetfy.py --svrdd-home /data/workspace/orddc/data/svrdd/SVRDD_YOLO/  --output-dir ./rdd2022/svrdd/imgnet

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

"""
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Convert SVRDD annotations to ImageNet dataset')
    parser.add_argument("--svrdd-home",      type=str,  default="/data/workspace/orddc/data/svrdd/SVRDD_YOLO/",   help="SVRDD folder path")
    parser.add_argument("--output-dir",    type=str,  default="./rdd2022/svrdd/imgnet",  help="Output directory for ImageNet like dataset")
    args=parser.parse_args()

    # Load annotations per split
    splits = ["train", "val"]
    for idx, split in enumerate(splits):
        print("{}.) {}".format(idx, split))
        data_files = os.path.join(args.svrdd_home, "{}.txt".format(split.lower()))
        # dataset/svrdd/train/{labels/*.txt, images/*.png}
        im_count = copy_image_by_annotations(data_files, args.svrdd_home, os.path.join(args.output_dir, split.lower()))
        print("{} >> {}".format(split, im_count))
