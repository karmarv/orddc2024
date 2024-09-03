
## (A.) MMDetection - RTMDet

#### Run - RTMDet-L
- Multi 4 GPU # 14 hours for b32 200 epochs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_l_rdd2022.py 4
```
- Test inference for submission to leaderboard
```bash
python test_results.py  ../../dataset/rdd2022/coco/test/images  ./rtmdet_l_rdd2022.py  ./work_dirs/rtmdet_l_rdd2022/epoch_200.pth  --out-dir ./work_dirs/rtmdet_l_rdd2022/rdd_test/  --to-labelme  --tta
```

#### Run - RTMDet-L
- Multi 4 GPU # 14 hours for b32 200 epochs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_l_swin_rdd2022.py 4
```

#### Run - RTMDet-M
- Multi 4 GPU # 14 hours for b48 200 epochs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_m_rdd2022.py 4
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_m_rdd2022.py 4 --resume /home/rahul/workspace/vision/orddc2024/model/mmdetection/work_dirs/rtmdet_m_rdd2022/epoch_200.pth
```
- Test inference for submission to leaderboard
```bash
python test_results.py  ../../dataset/rdd2022/coco/test/images  ./rtmdet_m_rdd2022.py  ./work_dirs/rtmdet_m_rdd2022/epoch_200.pth  --out-dir ./work_dirs/rtmdet_m_rdd2022/rdd_test/  --to-labelme  --tta
```


## (A.) MMDetection - CoDETR

- Multi 4 GPU # 14 hours for b32 200 epochs
```bash
pip install fairscale
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh codetr_r50_rdd2022.py 4
```

## (B.) MMYolo

- Installation: https://github.com/open-mmlab/mmyolo/blob/main/docs/en/get_started/installation.md 
  ```bash
    pip install wandb future tensorboard
    # After running wandb login, enter the API Keys obtained above, and the login is successful.
    wandb login 
  ```

### Exp

#### Run - RTMDet-L
```
# Multi 2 GPU # XXX hours for b28 100 epochs
CUDA_VISIBLE_DEVICES=6,7 PORT=29601 ./tools/dist_train.sh rtmdet_l_rdd2022.py 2
```
> 07/19 08:19:17 --> 07/20 01:34:34
```log
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| D00      | 0.167 | 0.329  | 0.147  | 0.087 | 0.137 | 0.222 |
| D10      | 0.187 | 0.374  | 0.165  | 0.092 | 0.156 | 0.3   |
| D20      | 0.182 | 0.357  | 0.165  | 0.054 | 0.077 | 0.198 |
| D40      | 0.102 | 0.237  | 0.072  | 0.089 | 0.102 | 0.104 |
+----------+-------+--------+--------+-------+-------+-------+
```

#### Run - RTMDet-M
```
# Multi 2 GPU # XXX hours for b48 100 epochs
CUDA_VISIBLE_DEVICES=6,7 PORT=29601 ./tools/dist_train.sh rtmdet_m_rdd2022.py 2
```
> 07/20 11:49:30 --> 07/20 21:48:01
```log
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| D00      | 0.182 | 0.347  | 0.168  | 0.098 | 0.147 | 0.243 |
| D10      | 0.22  | 0.424  | 0.196  | 0.101 | 0.169 | 0.36  |
| D20      | 0.202 | 0.384  | 0.189  | 0.28  | 0.079 | 0.221 |
| D40      | 0.112 | 0.253  | 0.081  | 0.098 | 0.112 | 0.12  |
+----------+-------+--------+--------+-------+-------+-------+
```
- TTA evaluation on test data val batch size 4 
```bash
# Add --tta to dist_test.sh in test.py arguments
CUDA_VISIBLE_DEVICES=4,5 PORT=29602 tools/dist_test.sh rtmdet_m_rdd2022.py  ./work_dirs/rtmdet_m_rdd2022/epoch_100.pth 2
```
  ```log
  +----------+-------+--------+--------+-------+-------+-------+
  | category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
  +----------+-------+--------+--------+-------+-------+-------+
  | D00      | 0.148 | 0.314  | 0.118  | 0.084 | 0.125 | 0.192 |
  | D10      | 0.176 | 0.37   | 0.14   | 0.087 | 0.139 | 0.291 |
  | D20      | 0.188 | 0.359  | 0.171  | 0.107 | 0.076 | 0.205 |
  | D40      | 0.106 | 0.243  | 0.075  | 0.097 | 0.105 | 0.117 |
  +----------+-------+--------+--------+-------+-------+-------+
  ```
- RTMDet-M training with pipeline stage2 switch using custom hooks 40 epoch before end
```bash
CUDA_VISIBLE_DEVICES=4,5 PORT=29601 ./tools/dist_train.sh rtmdet_m_rdd2022.py 2
```
```bash
TODO
```

#### Run - RTMDet-S
```
# Multi 2 GPU # XXX hours for b80 150 epochs
CUDA_VISIBLE_DEVICES=6,7 PORT=29601 ./tools/dist_train.sh rtmdet_s_rdd2022.py 2
```
> 1 07/21 06:33:27  --> 07/22 03:41:05
```log
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| D00      | 0.19  | 0.357  | 0.183  | 0.092 | 0.147 | 0.256 |
| D10      | 0.205 | 0.4    | 0.179  | 0.082 | 0.161 | 0.342 |
| D20      | 0.212 | 0.398  | 0.2    | 0.365 | 0.087 | 0.232 |
| D40      | 0.11  | 0.247  | 0.074  | 0.107 | 0.112 | 0.118 |
+----------+-------+--------+--------+-------+-------+-------+
```



## Yolov8

  - Environment variables
  ```
  #export CUDA_HOME="/usr/local/cuda"     # 12.4
  #export CUDA_HOME="/usr/local/cuda-12.1"
  export CUDA_HOME="/usr/local/cuda-11.8"
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```
  - Test
  ```
  python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME)'
  ```
  - Python environment
  ```
  conda env remove -n yolo
  conda create -n yolo python=3.9 -y
  conda activate yolo
  # Ensure CUDA 11.8
  pip install -r requirements.txt
  ```
  ```
  mim install "mmengine>=0.6.0"
  mim install "mmcv>=2.0.0rc4,<2.1.0"
  mim install "mmdet>=3.0.0,<4.0.0"
  ```

#### Run 1 - 300 epochs
```
# Multi 4 GPU # XXX hours for b80 150 epochs
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh yv8_m_rdd2022.py 4

```
- Test submission 
``` 
python test_results.py  ../../dataset/rdd2022/coco/test/images  ./yv8_m_rdd2022.py  ./work_dirs/yolov8_m_rdd/best_coco_D00_precision_epoch_300.pth  --out-dir ./work_dirs/yolov8_m_rdd/rdd_test/  --to-labelme --tta --device cuda:3
```




## MMPretrain
Improve the backbone with better features in pretraining

Documentation: https://mmpretrain.readthedocs.io/en/stable/notes/pretrain_custom_dataset.html

- Config: model/mmdetection/configs/rtmdet/classification/cspnext-m_8xb256-rsb-a1-600e_in1k.py 
  - [model/mmdetection/configs/rtmdet/classification/README.md](mmdetection/configs/rtmdet/classification/README.md)

```bash
cd mmpretrain
bash ./tools/dist_train.sh cspnext-m_8xb256-rsb-a1-600e_in1k.py 4
```