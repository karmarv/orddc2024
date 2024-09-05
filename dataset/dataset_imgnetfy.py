import random
import os, json
import argparse
import datetime

import numpy as np
from shutil import copyfile

from xml.etree import ElementTree
from xml.dom import minidom

# Configuration and path
random.seed(0) 

# Submission specific class identifiers
RDD_DAMAGE_CATEGORIES={
        "D00": {"id": 1, "name": "D00", "color": [220, 20, 60] , "submission_superid": 1, "description": "Longitudinal Crack"}, 
        "D10": {"id": 2, "name": "D10", "color": [0, 0, 142]   , "submission_superid": 2, "description": "Transverse Crack"}, 
        "D20": {"id": 3, "name": "D20", "color": [0, 60, 100]  , "submission_superid": 3, "description": "Aligator Crack"}, 
        "D40": {"id": 4, "name": "D40", "color": [0, 80, 100]  , "submission_superid": 4, "description": "Pothole"}
}
# Global Variables
EXPORT_CSV_DELIMITER=","
GLOBAL_LABEL_HISTGM={}              # Defect code as keys and count as values


def write_json_file(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

"""
Log the defect codes for histogram
"""
def hist_log_defect_code(code, region):
    global GLOBAL_LABEL_HISTGM
    if code in GLOBAL_LABEL_HISTGM.keys():
        GLOBAL_LABEL_HISTGM[code][region] = GLOBAL_LABEL_HISTGM[code][region] + 1
    else:
        GLOBAL_LABEL_HISTGM[code] = { "China_Drone":0, "China_MotorBike":0, "Czech":0, "India":0, "Japan":0, "Norway":0, "United_States":0 }
        GLOBAL_LABEL_HISTGM[code][region] = GLOBAL_LABEL_HISTGM[code][region] + 1
    return

def write_hist_defects(output_dir, filename='histogram.json'):
    global GLOBAL_LABEL_HISTGM
    # Write the defect histogram to file
    write_json_file(os.path.join(output_dir, 'annotations', filename), GLOBAL_LABEL_HISTGM)
    print("Defect Histogram: {}".format(GLOBAL_LABEL_HISTGM))
    # Empty upon writing
    GLOBAL_LABEL_HISTGM.clear()
    GLOBAL_LABEL_HISTGM = {}

def copy_image_by_annotations(data_dir, output_dir):
    # create target directories by label name
    for cls in RDD_DAMAGE_CATEGORIES.keys():
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
    # Load labeled dataset
    ann_path = os.path.join(data_dir, "train", "annotations", "xmls")
    img_path = os.path.join(data_dir, "train", "images")
    print("Load annotation: {}".format(ann_path))
    # list annotations/xml dir and for each annotation load the data
    ann_file_list = [filename for filename in os.listdir(ann_path) if filename.endswith('.xml')]
    count = 0
    for ann_id, ann_file in enumerate(ann_file_list):
        img_file = ann_file.split(".")[0] + ".jpg"
        if os.path.isfile(os.path.join(ann_path, ann_file)):
            # Read XML tags for image
            infile_xml = open(os.path.join(ann_path, ann_file))
            tree = ElementTree.parse(infile_xml)
            root = tree.getroot()
            # Read xml tags for bounding box annotations
            for obj in root.iter('object'):
                cls_name, xmlbox = obj.find('name').text, obj.find('bndbox')
                if cls_name in RDD_DAMAGE_CATEGORIES.keys(): # if class is interesting
                    # Copy image to class subfolder
                    src_file_path = os.path.join(img_path, img_file)
                    dst_file_path = os.path.join(output_dir, cls_name, img_file)
                    # Copy if unavailable to avoid duplicate
                    if not os.path.isfile(dst_file_path):
                        copyfile(src_file_path, dst_file_path)
                    # Log codes to histogram and write it out at the end
                    hist_log_defect_code(cls_name, os.path.basename(data_dir))
            count = count + 1
    return count

def copy_image_test(data_dir, output_dir):
    # create target directories by label name
    cls_name = "test"
    os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)
    img_file_list = []
    img_path = os.path.join(data_dir, "test", "images")
    if os.path.isdir(img_path):
        img_file_list = [filename for filename in os.listdir(img_path) if filename.endswith('.jpg')]
    count = 0
    for img_id, img_file in enumerate(img_file_list):
        # Copy image to class subfolder
        src_file_path = os.path.join(img_path, img_file)
        dst_file_path = os.path.join(output_dir, cls_name, img_file)
        # Copy if unavailable to avoid duplicate
        if not os.path.isfile(dst_file_path) and os.path.isfile(src_file_path):
            copyfile(src_file_path, dst_file_path)
            count = count + 1
    return count

"""
Usage: 
- time python dataset_imgnetfy.py --rdd-home ./rdd2022/RDD2022_all_countries/  --output-dir ./rdd2022/imgnet
- time python dataset_imgnetfy.py --rdd-home ./rdd2022/RDD2022/ --output-dir ./rdd2022/imgnet

"""
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Convert RDD2022 annotations to COCO dataset')
    parser.add_argument("--rdd-home",      type=str,  default="/data/workspace/orddc/data/RDD2022",   help="RDD2022 folder path")
    parser.add_argument("--output-dir",    type=str,  default="./rdd2022/imgnet",  help="Output directory for COCO dataset")
    args=parser.parse_args()

    countries = ["China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"]
    
    # Load annotations per country 
    for idx, country in enumerate(countries):
        print("{}.) {}".format(idx, country))
        data_dir = os.path.join(args.rdd_home, country)
        im_count = copy_image_by_annotations(data_dir, args.output_dir)
        print("Train >> ", im_count)
        ot_count = copy_image_test(data_dir, args.output_dir)
        print("Test  >> ", ot_count)
