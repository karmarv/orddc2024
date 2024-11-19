import math
import os, csv
import sys, time
from tqdm import tqdm

import cv2 
import torch

from mmengine.hub import get_config
from mmengine.config import Config, ConfigDict
from mmdet.apis import inference_detector, init_detector


# Setup
HOME = os.getcwd()
torch.cuda.empty_cache()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parse arguments
if len(sys.argv) != 3:
    print("Usage: python inference_script.py ../../dataset/rdd2022/coco/test/overall_6_countries/ ./test.csv")
    sys.exit(1)


def format_result_item(img_file, pred_instances, show_dir=None):
    """
    Read the predictions and format it into a submission string per image file
    """
    FONT_SCALE = 15e-4  # Adjust for larger font size in all images
    THICKNESS_SCALE = 2e-3  # Adjust for larger thickness in all images
    TEXT_Y_OFFSET_SCALE = 1e-2  # Adjust for larger Y-offset of text and bounding box
    height, width = 3, 3
    if show_dir is not None:
        os.makedirs(show_dir, exist_ok=True)
        img = cv2.imread(img_file)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        height, width, _ = img.shape
        
    pred_array = []
    for idx, pred in enumerate(pred_instances):
        scores = pred.scores.tolist()
        bboxes = pred.bboxes.tolist()
        labels = pred.labels.tolist()
        #print(" Score:", scores[0], "\tBbox:", bbox_str, "\tLabel:", dataset_classes[labels[0]])
        x1,y1,x2,y2 = map(int, bboxes[0])
        pred_array.append([scores[0], labels[0], x1, y1, x2, y2])
        if show_dir is not None:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, "{}-{:.2f}".format(dataset_classes[labels[0]], scores[0]),
                        (x1, y1 - int(height * TEXT_Y_OFFSET_SCALE)),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=min(width, height) * FONT_SCALE,
                        thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
                        color=(0,255,0)
                        )
    # Image overlayed results
    if show_dir is not None:
        cv2.imwrite(os.path.join(show_dir, "pred_{}".format(os.path.basename(img_file))), img)

    # Sort and write the predictions
    sorted_pred_array = sorted(pred_array, key=lambda x: x[0], reverse=True)
    pred_string = ""
    for item in sorted_pred_array:
        score, label, x1, y1, x2, y2 = item
        pred_string += "{} {} {} {} {} ".format(int(label)+1, x1, y1, x2, y2)

    return pred_string

def write_list_file(filename, rows, delimiter=','):
    """
    Write the submission result list to a CSV file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+") as my_csv:
        csvw = csv.writer(my_csv, delimiter=delimiter)
        csvw.writerows(rows)

def predict(model, inputs, pred_score_thr):
    """
    Run prediction using mmdetection inference detector for each image in test
    """
    show_dir=None
    #show_dir="./results"
    all_results = []
    results = inference_detector(model, inputs)
    for img_file, result_item in zip(inputs, results):
        pred_instances = result_item.pred_instances[result_item.pred_instances.scores > pred_score_thr]
        pred_string = format_result_item(img_file, pred_instances, show_dir)
        # Result row item format as per https://orddc2024.sekilab.global/submissions/
        all_results.append([os.path.basename(img_file), pred_string])
    return all_results




"""
Command Usage:
- time CUDA_VISIBLE_DEVICES=0 python inference_script.py ../../dataset/rdd2022/coco/test/overall_6_countries/ ./test.csv


Guidelines:
    ## Step (1) - Environment Setup with CUDA 11.8
    - Setup an Ubuntu 20.04 or 22.04 instance with a NVIDIA GPU on AWS EC2(https://aws.amazon.com/ec2/instance-types/g4/)
    - Install CUDA 11.8 on this instance
    ```
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    ```
    - Environment variables check
    ```
    export CUDA_HOME="/usr/local/cuda-11.8"
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    ```

    ### Step (2) - Python environment with conda
    - Install miniconda on linux using instructions from https://docs.anaconda.com/miniconda/
    - Create virtual environment
    ```
    conda env remove -n mm118 -y
    conda create -n mm118 python=3.9 -y
    conda activate mm118
    ```
    - Install required packages
    ```
    # Ensure CUDA 11.8 based packages
    pip install -r requirements.txt
    ```
    - Issue with the mmcv==2.0.1 package installation through requirement.txt file. This prebuilt binary is available only for CUDA 11.8. Need reinstallation as mentioned below.
    ```
    pip uninstall mmcv
    mim install "mmcv>=2.0.0rc4,<2.1.0"
    ```
    - Verify and test torch packages
    ```
    python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME)'
    ```

    ## Step (3) - Submission inference script
    - Ensure that the *.pth file is available in the same path.
    - Perform inference
    ```
    cd infer/
    CUDA_VISIBLE_DEVICES=0 python inference_script.py ../../dataset/rdd2022/coco/test/overall_6_countries/ ./test.csv
    ```
"""
if __name__ == '__main__':
    is_tta = False
    source_path = sys.argv[1]      # Path to the directory containing images for inference
    csv_file_path = sys.argv[2]    # output CSV file name including directory name

    # Configure detection model
    cfg = get_config('mmyolo::rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py', pretrained=False)
    #cfg = get_config('mmyolo::rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco.py', pretrained=False)
    # Setup a checkpoint file to load
    checkpoint = './mmy_rtml_pre_e250.pth'
    #checkpoint = './mmy_rtmm_640w_e250.pth'
    cfg.model.bbox_head.head_module.num_classes=4
    

    if is_tta:
        #
        # TTA - https://mmyolo.readthedocs.io/en/latest/common_usage/tta.html
        #
        cfg.tta_img_scales = [(640, 640), (960, 960), (1280, 1280)]
        cfg.tta_model = dict(
            type='mmdet.DetTTAModel',
            tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300))
        cfg._multiscale_resize_transforms = [
            dict(
                type='Compose',
                transforms=[
                    dict(type='YOLOv5KeepRatioResize', scale=s),
                    dict(
                        type='LetterResize',
                        scale=s,
                        allow_scale_up=False,
                        pad_val=dict(img=114))
                ]) for s in cfg.tta_img_scales
        ]
        cfg.tta_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='TestTimeAug',
                transforms=[
                    cfg._multiscale_resize_transforms,
                    [
                        dict(type='mmdet.RandomFlip', prob=1.),
                        dict(type='mmdet.RandomFlip', prob=0.)
                    ], [dict(type='mmdet.LoadAnnotations', with_bbox=True)],
                    [
                        dict(
                            type='mmdet.PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                    'scale_factor', 'pad_param', 'flip',
                                    'flip_direction'))
                    ]
                ])
        ]
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.' \
            " Can't use tta !"
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` ' \
            "in config. Can't use tta !"
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        test_data_cfg = cfg.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = cfg.tta_pipeline        

    print("{} - Loading checkpoint {} to model".format(time.strftime("%Y%m%d-%H%M%S"), checkpoint))
    # Initialize the DetInferencer
    model = init_detector(cfg, checkpoint, device=DEVICE, cfg_options={})
    # get model class name
    dataset_classes = model.dataset_meta.get('classes')

    # Perform inference
    pred_score_thr = 0.21
    rdd_results = []
    rdd_IN, rdd_JP, rdd_NO, rdd_US = [], [], [], []
    onlyfiles = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
    for img_file in tqdm(onlyfiles):
        img_filepath = os.path.join(source_path, img_file)
        batch_results = predict(model, [img_filepath], pred_score_thr) # TODO batching here

        # Country specific results aggregation
        if "India" in img_file:
            rdd_IN.extend(batch_results)
        elif "Japan" in img_file:
            rdd_JP.extend(batch_results)
        elif "Norway" in img_file:
            rdd_NO.extend(batch_results)
        elif "United_States" in img_file:
            rdd_US.extend(batch_results)
        # all countries results
        rdd_results.extend(batch_results)
    
    print("{} - Results written to {} file".format(time.strftime("%Y%m%d-%H%M%S"), csv_file_path))
    write_list_file(csv_file_path, rdd_results)
    #write_list_file("{}_IN".format(csv_file_path), rdd_IN)
    #write_list_file("{}_JP".format(csv_file_path), rdd_JP)
    #write_list_file("{}_NO".format(csv_file_path), rdd_NO)
    #write_list_file("{}_US".format(csv_file_path), rdd_US)

        

