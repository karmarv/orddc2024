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
  - Needed a fix to the downloaded Norway dataset: `zip --fixfix RDD2022_Norway.zip --out RDD2022_Norway_fixed.zip`
  ```bash 
    $ tree -L 3 RDD2022/
    RDD2022/
    ├── China_Drone
    │   └── train
    │       ├── annotations
    │       └── images
    ├── China_MotorBike
    │   ├── test
    │   │   └── images
    │   └── train
    │       ├── annotations
    │       └── images
    ├── Czech
    │   ├── test
    │   │   └── images
    │   └── train
    │       ├── annotations
    │       └── images
    ├── India
    │   ├── test
    │   │   └── images
    │   └── train
    │       ├── annotations
    │       └── images
    ├── Japan
    │   ├── test
    │   │   └── images
    │   └── train
    │       ├── annotations
    │       └── images
    ├── Norway
    │   ├── test
    │   │   └── images
    │   └── train
    │       ├── annotations
    │       └── images
    └── United_States
        ├── test
        │   └── images
        └── train
            ├── annotations
            └── images
  ```
- Preparation
  ```bash

  ```