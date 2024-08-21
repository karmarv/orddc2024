import argparse
import os
import torch
import csv
import cv2 

from mmdet.apis import DetInferencer

# Setup
HOME = os.getcwd()
torch.cuda.empty_cache()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def write_list_file(filename, rows, delimiter=','):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+") as my_csv:
        csvw = csv.writer(my_csv, delimiter=delimiter)
        csvw.writerows(rows)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument('model_file', type=str, help='model file name including directory name')
    parser.add_argument('source_path', type=str, help='Path to the directory containing images for inference')
    parser.add_argument('output_csv_file', type=str, help='output CSV file name including directory name')
    args = parser.parse_args()


    # Load the model weights file
    model_path = args.model_file

    # Path to the directory containing images for inference
    source_path = args.source_path

    # Initialize the DetInferencer with device='cpu' or device='cuda:0'
    inferencer = DetInferencer(model='rtmdet_l_8xb32-300e_coco', weights=model_path, device=DEVICE)

    # list files in img directory
    files = os.listdir(source_path)

    for file in files:
        # make sure file is an image
        if file.endswith(('.jpg', '.png', 'jpeg')):
            img_path = os.path.join(source_path, file)
            print(img_path)
            img = cv2.imread(img_path)
            #img = mmcv.imconvert(img, 'bgr', 'rgb')
            result = inferencer(img)
            print(result)
            break

print("Done")
