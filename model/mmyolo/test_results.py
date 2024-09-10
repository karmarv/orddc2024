
import time
import os, csv
from argparse import ArgumentParser
from pathlib import Path

import mmcv
import cv2 

from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path

from mmdet.utils.misc import get_file_list

"""
python test_results.py  ../../dataset/rdd2022/coco/test/images  ./rtmdet_m_rdd2022.py  ./work_dirs/rtmdet_m_rdd2022/epoch_100.pth  --out-dir ./work_dirs/rtmdet_m_rdd2022/rdd_test/  --to-labelme 
python test_results.py  ../../dataset/rdd2022/coco/test/images  ./yv8_m_rdd2022.py  ./work_dirs/yolov8_m_rdd/best_coco_D00_precision_epoch_300.pth  --out-dir ./work_dirs/yolov8_m_rdd/rdd_test/  --to-labelme --tta

python test_results.py  ../../dataset/rdd2022/coco/test/overall_6_countries/  ./rtmdet_l_rdd2022.py  ./work_dirs/rtmdet_l_rdd_stg_plus/epoch_225.pth  --out-dir ./work_dirs/rtmdet_l_rdd_stg_plus/rdd_test/  --to-labelme  --device cuda:0 --score-thr 0.2
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--class-name',
        nargs='+',
        type=str,
        help='Only Save those classes if set')
    parser.add_argument(
        '--to-labelme',
        action='store_true',
        help='Output labelme style label file')
    args = parser.parse_args()
    return args


def write_list_file(filename, rows, delimiter=','):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+") as my_csv:
        csvw = csv.writer(my_csv, delimiter=delimiter)
        csvw.writerows(rows)

def main():
    args = parse_args()

    if args.to_labelme and args.show:
        raise RuntimeError('`--to-labelme` or `--show` only '
                           'can choose one at the same time.')
    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    if args.tta:
        assert 'tta_model' in config, 'Cannot find ``tta_model`` in config.' \
            " Can't use tta !"
        assert 'tta_pipeline' in config, 'Cannot find ``tta_pipeline`` ' \
            "in config. Can't use tta !"
        config.model = ConfigDict(**config.tta_model, module=config.model)
        test_data_cfg = config.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = config.tta_pipeline

    # TODO: TTA mode will error if cfg_options is not set.
    #  This is an mmdet issue and needs to be fixed later.
    # build the model from a config file and a checkpoint file
    model = init_detector(
        config, args.checkpoint, device=args.device, cfg_options={})

    if not args.show:
        path.mkdir_or_exist(args.out_dir)

    # get file list
    files, source_type = get_file_list(args.img)

    # get model class name
    dataset_classes = model.dataset_meta.get('classes')

    # check class name
    if args.class_name is not None:
        for class_name in args.class_name:
            if class_name in dataset_classes:
                continue

            raise RuntimeError(
                'Expected args.class_name to be one of the list, '
                f'but got "{class_name}"')

    rdd_results = []

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        result = inference_detector(model, file)

        img = mmcv.imread(file)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        progress_bar.update()
        # Get candidate predict info with score threshold
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.score_thr]

        if args.to_labelme:
            # Add results to CSV file
            pred_array = []
            for idx, pred in enumerate(pred_instances):
                scores = pred.scores.tolist()
                bboxes = pred.bboxes.tolist()
                labels = pred.labels.tolist()
                #print(" Score:", scores[0], "\tBbox:", bbox_str, "\tLabel:", dataset_classes[labels[0]])
                x1,y1,x2,y2 = map(int, bboxes[0])
                pred_array.append([scores[0], labels[0], x1, y1, x2, y2])
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
                cv2.putText(img, "{}-{:.2f}".format(dataset_classes[labels[0]], scores[0]), (x1+2,y1+18), 0, 0.55, (0,255,0), 2)
            # Sort and write the predictions
            sorted_pred_array = sorted(pred_array, key=lambda x: x[0], reverse=True)
            pred_string = ""
            for item in sorted_pred_array:
                score, label, x1, y1, x2, y2 = item
                pred_string += "{} {} {} {} {} ".format(int(label)+1, x1, y1, x2, y2)
            # Result row item format as per https://orddc2024.sekilab.global/submissions/
            rdd_results.append([os.path.basename(file), pred_string])
            #if len(pred_string)>0 and idx>3:
            #    print(rdd_results)
            #    exit(0)
            if len(pred_string)>0:
                mmcv.imwrite(img, os.path.join(args.out_dir, "pred_{}.jpg".format(os.path.basename(file))))
            continue

    if args.to_labelme:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        print_log('\nLabelme format label files '
                  f'had all been saved in {args.out_dir}')
        write_list_file(os.path.join(args.out_dir, "..", "{}_{}_test.csv".format(timestr, os.path.basename(args.config))), rdd_results)


if __name__ == '__main__':
    main()
