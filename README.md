# orddc2024
Optimized Road Damage Detection Challenge (IEEE Big Data Cup ORDDC'2024)



### Experiments

#### A. RTMDet 

- First Trial




### Environment
- Python 3.8.18 Installation via Miniconda v23.1.0 - https://docs.conda.io/projects/miniconda/en/latest/ 
  ```bash
  conda env remove -n rdd
  conda create -n rdd python=3.9
  conda activate rdd
  pip install -r requirements.txt
  ```
- MMDetection environment
  ```bash
  pip install -U openmim
  mim install mmengine
  mim install "mmcv>=2.0.0"
  mim install mmdet
  ```
  - Verify installation
  ```bash
  cd demo
  mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
  python image_demo.py demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cuda:0
  ```

### Dataset
- Download data from https://github.com/sekilab/RoadDamageDetector#dataset 
  - Needed a [fix](https://github.com/sekilab/RoadDamageDetector//issues/57) to the downloaded Norway dataset: `zip --fixfix RDD2022_Norway.zip --out RDD2022_Norway_fixed.zip`
  ```bash 
    $ tree -L 3 RDD2022/
    RDD2022/
    ├── China_Drone
    │   └── train
    │       ├── annotations
    │       │   └── xmls/*.xml
    │       └── images/*.jpg
    ├── China_MotorBike
    │   ├── test
    │   │   └── images/*.jpg
    │   └── train
    │       ├── annotations
    │       │   └── xmls/*.xml
    │       └── images/*.jpg
    ├── Czech
    │   ├── test
    │   │   └── images/*.jpg
    │   └── train
    │       ├── annotations
    │       │   └── xmls/*.xml
    │       └── images/*.jpg
    ├── India
    │   ├── test
    │   │   └── images/*.jpg
    │   └── train
    │       ├── annotations
    │       │   └── xmls/*.xml
    │       └── images/*.jpg
    ├── Japan
    │   ├── test
    │   │   └── images/*.jpg
    │   └── train
    │       ├── annotations
    │       │   └── xmls/*.xml
    │       └── images/*.jpg
    ├── Norway
    │   ├── test
    │   │   └── images/*.jpg
    │   └── train
    │       ├── annotations
    │       │   └── xmls/*.xml
    │       └── images/*.jpg
    └── United_States
        ├── test
        │   └── images/*.jpg
        └── train
            ├── annotations
            │   └── xmls/*.xml
            └── images/*.jpg
  ```
- Preparation
  - Convert VOC XML to COCO format
  ```bash
  python dataset_cocofy.py
  ```
  - Split 85:15 for train and validation using [cocosplit](https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py)
  ```bash
  python cocosplit.py --multi-class -s 0.8 ./rdd2022/coco/annotations/rdd2022_annotations.json ./rdd2022/coco/annotations/train.json ./rdd2022/coco/annotations/val.json
  Saved 44006 entries in ./rdd2022/coco/annotations/train.json and 11001 in ./rdd2022/coco/annotations/val.json
  100%|██████████████████████████████████████████████████████████████████| 21109/21109 [00:06<00:00, 3043.20it/s]
  100%|██████████████████████████████████████████████████████████████████| 8246/8246 [00:01<00:00, 5901.35it/s]
  Copied 21110 images in ./rdd2022/coco/annotations/train.json and 8247 in ./rdd2022/coco/annotations/val.json
  ```  