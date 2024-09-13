### Dataset Structure
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
## Data Preparation Workflow
- Convert VOC XML to COCO format
  ```bash
  time python dataset_cocofy.py --rdd-home ./rdd2022/RDD2022_all_countries/ 
  ```
- Split 80:20 for train and validation using [cocosplit](https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py)
    ```bash
    # rahul@old_gpu_server
    python cocosplit.py --multi-class -s 0.8 ./rdd2022/coco/annotations/rdd2022_annotations.json ./rdd2022/coco/annotations/train.json ./rdd2022/coco/annotations/val.json
    Saved 44006 entries in ./rdd2022/coco/annotations/train.json and 11001 in ./rdd2022/coco/annotations/val.json
    100%|█████████████████████████████████████| 21109/21109 [00:06<00:00, 3043.20it/s]
    100%|█████████████████████████████████████| 8246/8246 [00:01<00:00, 5901.35it/s]
    Copied 21110 images in ./rdd2022/coco/annotations/train.json and 8247 in ./rdd2022/coco/annotations/val.json
    ```  
    ```bash
    # rahul@new_gpu_server
    python cocosplit.py --multi-class -s 0.8 ./rdd2022/coco/annotations/rdd2022_annotations.json ./rdd2022/coco/annotations/train.json ./rdd2022/coco/annotations/val.json
    Saved 44006 entries in ./rdd2022/coco/annotations/train.json and 11001 in ./rdd2022/coco/annotations/val.json
    100%|█████████████████████████████████████| 21139/21139 [00:07<00:00, 2834.73it/s]
    100%|█████████████████████████████████████| 8226/8226 [00:01<00:00, 5367.25it/s]
    Copied 21140 images in ./rdd2022/coco/annotations/train.json and 8227 in ./rdd2022/coco/annotations/val.json
    ```
- Converter to ImageNet folder structure for pretraining
- Converter to ImageNet folder structure for external SVRDD data pretraining 

### Dataset Visualization
- COCO Data notebook at [./visualize_coco_data.ipynb](./visualize_coco_data.ipynb)
- COCO Plus data notebook at [./visualize_coco_data_plus.ipynb](./visualize_coco_data_plus.ipynb)