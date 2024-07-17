import random
import os, json
import argparse
import datetime

import numpy as np

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
GLOBAL_IMAGE_SEQ_ID=0               # Initialized
GLOBAL_ANNOT_SEQ_ID=0
EXPORT_CSV_DELIMITER=","
GLOBAL_LABEL_HISTGM={}              # Defect code as keys and count as values


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

def load_country_annotations(data_dir):
    global GLOBAL_IMAGE_SEQ_ID, GLOBAL_ANNOT_SEQ_ID
    imgs_dict = dict()
    gts_coco_labels = {"images":[], "annotations":[]}
    ann_path = os.path.join(data_dir, "train", "annotations", "xmls")
    #img_path = os.path.join(data_dir, "train", "images")
    print("Load annotation: {}".format(ann_path))
    # list annotations/xml dir and for each annotation load the data
    ann_file_list = [filename for filename in os.listdir(ann_path) if filename.endswith('.xml')]
    for ann_id, ann_file in enumerate(ann_file_list):
        img_file = ann_file.split(".")[0] + ".jpg"
        if os.path.isfile(os.path.join(ann_path, ann_file)):
            # Read XML tags for image
            infile_xml = open(os.path.join(ann_path, ann_file))
            tree = ElementTree.parse(infile_xml)
            root = tree.getroot()
            img_height = int(root.find('size').find('height').text)
            img_width  = int(root.find('size').find('width').text)

            # Images unique id in gts_coco_labels["images"] dict
            if img_file in imgs_dict.keys():
                image_info = imgs_dict[img_file]
                image_id   = image_info["id"]
            else:
                image_id   = GLOBAL_IMAGE_SEQ_ID = GLOBAL_IMAGE_SEQ_ID + 1
                # COCO Format Image entry if not in dict for image metadata reuse
                image_info = create_image_info(image_id, img_file, image_size=[img_width, img_height])
                imgs_dict[img_file] = image_info
                gts_coco_labels["images"].append(image_info)

            # Read xml tags for bounding box annotations
            for obj in root.iter('object'):
                cls_name, xmlbox = obj.find('name').text, obj.find('bndbox')
                if cls_name in RDD_DAMAGE_CATEGORIES.keys(): # if class is interesting
                    GLOBAL_ANNOT_SEQ_ID = GLOBAL_ANNOT_SEQ_ID + 1
                    xmin, xmax = float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text)
                    ymin, ymax = float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
                    bb_w, bb_h = abs(xmax-xmin), abs(ymax-ymin)
                    bbox = [int(xmin), int(ymin), int(bb_w), int(bb_h)]       # (x0, y0, w, h)
                    area = int(bb_w * bb_h)
                    cls_id = RDD_DAMAGE_CATEGORIES[cls_name]["id"]
                    ann_info  = create_annotation_info(GLOBAL_ANNOT_SEQ_ID, image_id, cls_id, area, bbox)
                    gts_coco_labels["annotations"].append(ann_info)
                    # Log codes to histogram and write it out at the end
                    hist_log_defect_code(cls_name, os.path.basename(data_dir))
    return gts_coco_labels


def get_coco_metadata(dataset_type="train"):
    coco_meta = {"images":[], "annotations":[]}
    coco_meta["info"] = {
        "description": "RDD2022 Dataset - {}".format(dataset_type),
        "url": "https://github.com/",
        "version": "0.2.0",
        "year": 2024,
        "contributor": "Rahul Vishwakarma",
        "date_created": datetime.datetime.utcnow().isoformat(" "),
    }
    coco_meta["licenses"] = [
        {
            "id": 1,
            "name": "[TODO] Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        }
    ]

    coco_meta["categories"] = [
        {
            "id": class_label['id'],
            "name": class_name,
            "supercategory": class_label['submission_superid'],
        }
        for class_name, class_label in RDD_DAMAGE_CATEGORIES.items()
    ]
    return coco_meta

def get_clean_basename(filename):
    # Cleanup the directory name for special chars
    basename = os.path.basename(filename)
    clean_basename = "".join( x for x in basename if (x.isalnum() or x in "._-"))
    return clean_basename

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Convert RDD2022 annotations to COCO dataset')
    parser.add_argument("--rdd-home",      type=str,  default="/data/workspace/orddc/data/RDD2022",   help="RDD2022 folder path")
    parser.add_argument("--output-dir",    type=str,  default="./rdd2022/coco",  help="Output directory for COCO dataset")
    args=parser.parse_args()

    rdd_path = args.rdd_home
    countries = ["China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"]
    
    # Empty placeholders 
    train_ann_dict = get_coco_metadata("train")

    # Load annotations per country 
    for idx, country in enumerate(countries):
        print("{}.) {}".format(idx, country))
        data_dir = os.path.join(rdd_path, country)
        gts_coco_dict = load_country_annotations(data_dir)
        # Accumulate the images and labels
        train_ann_dict["images"]      = train_ann_dict["images"]      + gts_coco_dict["images"]
        train_ann_dict["annotations"] = train_ann_dict["annotations"] + gts_coco_dict["annotations"]
        print("Images: {}".format(len(train_ann_dict["images"])))
        print("Annots: {}".format(len(train_ann_dict["annotations"])))
        print(".")

    # Write Json data to file
    write_json_file(os.path.join(args.output_dir, 'annotations', 'rdd2022_annotations.json'), train_ann_dict)
    # Write histogram to file
    write_hist_defects(args.output_dir, filename='histogram_coco4_rdd2022.json')