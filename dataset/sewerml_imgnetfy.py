import random
import os, json
import argparse
import datetime

import pandas as pd
from shutil import copyfile
from tqdm import tqdm

# Configuration and path
random.seed(0) 

# WaterLevel,VA,RB,OB, PF,DE,FS,IS, RO,IN,AF,BE, FO,GR,PH,PB, OS,OP,OK,ND, Defect
DAMAGE_CATEGORIES={
        "WaterLevel":   {"id":  1, "name": "D00", "color": [220, 20, 60] , "submission_superid": 10, "description": "SewerML"}, 
        "VA":           {"id":  2, "name": "D10", "color": [0, 0, 142]   , "submission_superid": 10, "description": "SewerML"}, 
        "RB":           {"id":  3, "name": "D20", "color": [0, 60, 100]  , "submission_superid": 10, "description": "SewerML"}, 
        "OB":           {"id":  4, "name": "D40", "color": [0, 80, 100]  , "submission_superid": 10, "description": "SewerML"},
        "PF":           {"id":  5, "name": "D00", "color": [220, 20, 60] , "submission_superid": 10, "description": "SewerML"}, 
        "DE":           {"id":  6, "name": "D10", "color": [0, 0, 142]   , "submission_superid": 10, "description": "SewerML"}, 
        "FS":           {"id":  7, "name": "D20", "color": [0, 60, 100]  , "submission_superid": 10, "description": "SewerML"}, 
        "IS":           {"id":  8, "name": "D40", "color": [0, 80, 100]  , "submission_superid": 10, "description": "SewerML"},
        "RO":           {"id":  9, "name": "D00", "color": [220, 20, 60] , "submission_superid": 10, "description": "SewerML"}, 
        "IN":           {"id": 10, "name": "D10", "color": [0, 0, 142]   , "submission_superid": 10, "description": "SewerML"}, 
        "AF":           {"id": 11, "name": "D20", "color": [0, 60, 100]  , "submission_superid": 10, "description": "SewerML"}, 
        "BE":           {"id": 12, "name": "D40", "color": [0, 80, 100]  , "submission_superid": 10, "description": "SewerML"},
        "FO":           {"id": 13, "name": "D00", "color": [220, 20, 60] , "submission_superid": 10, "description": "SewerML"}, 
        "GR":           {"id": 14, "name": "D10", "color": [0, 0, 142]   , "submission_superid": 10, "description": "SewerML"}, 
        "PH":           {"id": 15, "name": "D20", "color": [0, 60, 100]  , "submission_superid": 10, "description": "SewerML"}, 
        "PB":           {"id": 16, "name": "D40", "color": [0, 80, 100]  , "submission_superid": 10, "description": "SewerML"}, 
        "OS":           {"id": 17, "name": "D00", "color": [220, 20, 60] , "submission_superid": 10, "description": "SewerML"}, 
        "OP":           {"id": 18, "name": "D10", "color": [0, 0, 142]   , "submission_superid": 10, "description": "SewerML"}, 
        "OK":           {"id": 19, "name": "D20", "color": [0, 60, 100]  , "submission_superid": 10, "description": "SewerML"}, 
        "ND":           {"id": 20, "name": "D40", "color": [0, 80, 100]  , "submission_superid": 10, "description": "SewerML"},
        "Defect":       {"id": 21, "name": "D40", "color": [0, 80, 100]  , "submission_superid": 10, "description": "SewerML"}

}


def copy_image_by_annotations(ann_file, data_dir, output_dir):
    # create target directories by label name
    for cls in DAMAGE_CATEGORIES.keys():
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
    # reading csv file 
    ann_df = pd.read_csv(ann_file)
    print(ann_df.head())
    count = 0
    for index, ann in tqdm(ann_df.iterrows()):
        img_file = ann["Filename"]
        lbl_seen = {}
        for col in DAMAGE_CATEGORIES.keys():
            if ann[col] > 0:
                lbl_seen[col] = ann[col]
        # Copy images to corresponding labels dir
        for lbl in lbl_seen.keys():
            # Copy image to class subfolder
            src_file_path = os.path.join(data_dir, img_file)
            dst_file_path = os.path.join(output_dir, lbl, img_file)
            # Copy if unavailable to avoid duplicate
            if not os.path.isfile(dst_file_path):
                copyfile(src_file_path, dst_file_path)
        count = count + 1
    return count

"""
Usage: 
- time python sewerml_imgnetfy.py --sewerml-home /data/workspace/jacobs/data_fy23/sewerml-ext/  --output-dir ./sewerml/imgnet

"""
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Convert SewerML annotations to ImageNet dataset')
    parser.add_argument("--sewerml-home",      type=str,  default="/data/workspace/jacobs/data_fy23/sewerml-ext",   help="SewerML folder path")
    parser.add_argument("--output-dir",    type=str,  default="./sewerml/imgnet",  help="Output directory for ImageNet like dataset")
    args=parser.parse_args()

    # Load annotations per split
    splits = ["Train", "Val"]
    for idx, split in enumerate(splits):
        print("{}.) {}".format(idx, split))
        ann_filename = "SewerML_{}.csv".format(split)      # dataset/sewerml/SewerML_Test.csv
        ann_filepath = os.path.join(args.sewerml_home, ann_filename)
        data_dir = os.path.join(args.sewerml_home, split.lower())
        im_count = copy_image_by_annotations(ann_filepath, data_dir, os.path.join(args.output_dir, split.lower()))
        print("{} >> {}".format(split, im_count))


"""
(rdd) rahul@bdasl-gpu-server:~/workspace/vision/rdd/orddc2024/dataset$ time python sewerml_imgnetfy.py --sewerml-home /data/workspace/jacobs/data_fy23/sewerml-ext/  --output-dir ./sewerml/imgnet
0.) Train
       Filename  WaterLevel  VA  RB  OB  PF  DE  FS  IS  RO  IN  AF  BE  FO  GR  PH  PB  OS  OP  OK  ND  Defect
0  00000001.png          10   1   1   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0       1
1  00000002.png          10   0   1   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0       1
2  00000003.png          10   0   1   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0       1
3  00000004.png          10   0   0   0   0   0   1   0   0   0   1   0   0   0   0   0   0   0   0   0       1
4  00000005.png          10   1   0   0   0   0   1   0   0   0   1   0   0   0   0   0   0   0   1   0       1
1040129it [03:58, 4358.38it/s]
Train >> 1040129
1.) Val
       Filename  WaterLevel  VA  RB  OB  PF  DE  FS  IS  RO  IN  AF  BE  FO  GR  PH  PB  OS  OP  OK  ND  Defect
0  00000048.png           0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0       0
1  00000049.png           0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0       1
2  00000050.png           0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1       0
3  00000051.png           0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0       0
4  00000052.png           0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0       0
130046it [03:28, 622.38it/s]
Val >> 130046

real    7m30.530s
user    2m27.244s
sys     3m31.395s
"""