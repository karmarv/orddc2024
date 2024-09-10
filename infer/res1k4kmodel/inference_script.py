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
    if show_dir is not None:
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    pred_array = []
    for idx, pred in enumerate(pred_instances):
        scores = pred.scores.tolist()
        bboxes = pred.bboxes.tolist()
        labels = pred.labels.tolist()
        #print(" Score:", scores[0], "\tBbox:", bbox_str, "\tLabel:", dataset_classes[labels[0]])
        x1,y1,x2,y2 = map(int, bboxes[0])
        pred_array.append([scores[0], labels[0], x1, y1, x2, y2])
        if show_dir is not None:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.putText(img, "{}-{:.2f}".format(dataset_classes[labels[0]], scores[0]), (x1+2,y1+18), 0, 0.55, (0,255,0), 2)
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
    batch_results = []
    results = inference_detector(model, inputs)
    for img_file, result_item in zip(inputs, results):
        pred_instances = result_item.pred_instances[result_item.pred_instances.scores > pred_score_thr]
        pred_string = format_result_item(img_file, pred_instances, show_dir=None)
        # Result row item format as per https://orddc2024.sekilab.global/submissions/
        batch_results.append([os.path.basename(img_file), pred_string])
    return batch_results




"""
Usage:
- time CUDA_VISIBLE_DEVICES=0 python inference_script.py ../../dataset/rdd2022/coco/test/overall_6_countries/ ./test.csv
"""
if __name__ == '__main__':
    is_tta = False
    source_path = sys.argv[1]      # Path to the directory containing images for inference
    csv_file_path = sys.argv[2]    # output CSV file name including directory name

    # Configure detection model
    cfg = get_config('mmyolo::rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py', pretrained=False)
    cfg.model.bbox_head.head_module.num_classes=4
    
    # Setup a checkpoint file to load
    checkpoint_1k = './mmy_rtml_pre_e250.pth'
    #checkpoint_1k = './mmy_rtml_1kres_e60.pth'
    checkpoint_4k = './mmy_rtml_4kres_e100.pth'
    #checkpoint_4k = './mmy_rtml_pre_e250.pth'

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

    print("{} - Loading checkpoint {} to model".format(time.strftime("%Y%m%d-%H%M%S"), checkpoint_1k))
    # Initialize the DetInferencer
    model_1k = init_detector(cfg, checkpoint_1k, device=DEVICE, cfg_options={})
    model_4k = init_detector(cfg, checkpoint_4k, device=DEVICE, cfg_options={})

    # Perform inference
    pred_score_thr = 0.2
    rdd_results = []
    rdd_IN, rdd_JP, rdd_NO, rdd_US = [], [], [], []
    onlyfiles = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
    count1k, count4k = 0, 0
    for img_file in tqdm(onlyfiles):
        img_filepath = os.path.join(source_path, img_file)

        # Country specific results aggregation
        if "India" in img_file:
            batch_results = predict(model_1k, [img_filepath], pred_score_thr) # No batching here
            count1k = count1k + 1
            rdd_IN.extend(batch_results)
        elif "Japan" in img_file:
            batch_results = predict(model_1k, [img_filepath], pred_score_thr) # No batching here
            count1k = count1k + 1
            rdd_JP.extend(batch_results)
        elif "Norway" in img_file:
            batch_results = predict(model_4k, [img_filepath], pred_score_thr) # No batching here
            count4k = count4k + 1
            rdd_NO.extend(batch_results)
        elif "United_States" in img_file:
            batch_results = predict(model_1k, [img_filepath], pred_score_thr) # No batching here
            count1k = count1k + 1
            rdd_US.extend(batch_results)
        else:
            batch_results = predict(model_1k, [img_filepath], pred_score_thr) # No batching here
            count1k = count1k + 1

        # all countries results            
        rdd_results.extend(batch_results)
    print("{} Images processed through 4k Model, {} Images processed through 1k Model".format(count4k, count1k))
    print("{} - Results written to {} file".format(time.strftime("%Y%m%d-%H%M%S"), csv_file_path))
    write_list_file(csv_file_path, rdd_results)
    # Other countries results
    write_list_file("{}_IN".format(csv_file_path), rdd_IN)
    write_list_file("{}_JP".format(csv_file_path), rdd_JP)
    write_list_file("{}_NO".format(csv_file_path), rdd_NO)
    write_list_file("{}_US".format(csv_file_path), rdd_US)

        

