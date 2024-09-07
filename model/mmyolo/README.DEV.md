## Environment 
  - Environment variables
  ```
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

- RTMDet-L training with pipeline stage2 switch using custom hooks 40 epoch before end
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29601 ./tools/dist_train.sh rtmdet_l_rdd2022.py 4
```
```log
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| D00      | 0.216 | 0.395  | 0.206  | 0.109 | 0.171 | 0.284 |
| D10      | 0.249 | 0.464  | 0.232  | 0.131 | 0.196 | 0.395 |
| D20      | 0.242 | 0.441  | 0.228  | 0.315 | 0.112 | 0.262 |
| D40      | 0.137 | 0.299  | 0.1    | 0.144 | 0.128 | 0.154 |
+----------+-------+--------+--------+-------+-------+-------+
09/04 00:44:15 - mmengine - INFO - bbox_mAP_copypaste: 0.211 0.399 0.191 0.175 0.152 0.274
09/04 00:44:17 - mmengine - INFO - Epoch(val) [250][65/65]    coco/D00_precision: 0.2160  coco/D10_precision: 0.2490  coco/D20_precision: 0.2420  coco/D40_precision: 0.1370  coco/bbox_mAP: 0.2110  coco/bbox_mAP_50: 0.3990  coco/bbox_mAP_75: 0.1910  coco/bbox_mAP_s: 0.1750  coco/bbox_mAP_m: 0.1520  coco/bbox_mAP_l: 0.2740  data_time: 0.0063  time: 0.4866
```
- RTMDet-L training with pipeline stage2 switch using custom hooks 40 epoch before end (mmpretrained checkpoint "../mmpretrain/work_dirs/cspnext-l_8xb256-rsb-a1-600e_in1k/epoch_600.pth")
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29601 ./tools/dist_train.sh rtmdet_l_rdd2022.py 4
```
```log
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| D00      | 0.189 | 0.371  | 0.163  | 0.105 | 0.154 | 0.246 |
| D10      | 0.227 | 0.432  | 0.212  | 0.114 | 0.187 | 0.355 |
| D20      | 0.216 | 0.402  | 0.205  | 0.296 | 0.1   | 0.236 |
| D40      | 0.121 | 0.27   | 0.087  | 0.115 | 0.117 | 0.133 |
+----------+-------+--------+--------+-------+-------+-------+
09/06 00:53:34 - mmengine - INFO - bbox_mAP_copypaste: 0.188 0.369 0.167 0.157 0.140 0.242                                                                                                                                                                │·
09/06 00:53:36 - mmengine - INFO - Epoch(val) [250][65/65]    coco/D00_precision: 0.1890  coco/D10_precision: 0.2270  coco/D20_precision: 0.2160  coco/D40_precision: 0.1210  coco/bbox_mAP: 0.1880  coco/bbox_mAP_50: 0.3690  coco/bbox_mAP_75: 0.1670  c│·oco/bbox_mAP_s: 0.1570  coco/bbox_mAP_m: 0.1400  coco/bbox_mAP_l: 0.2420  data_time: 0.0075  time: 0.5126
```
  - Test inference for submission to leaderboard
  ```bash
  python test_results.py  ../../dataset/rdd2022/coco/test/images  ./rtmdet_l_rdd2022.py  ./work_dirs/rtmdet_l_rdd_stg/20240905_071601-pre600/epoch_250.pth  --out-dir ./work_dirs/rtmdet_l_rdd_stg/20240905_071601-pre600/rdd_test/  --to-labelme  --tta
  ```

- RTMDet-L training with pipeline stage2 switch using custom hooks 40 epoch before end and coco_plus dataset
```bash
CUDA_VISIBLE_DEVICES=6,7 PORT=29603 ./tools/dist_train.sh rtmdet_l_rdd2022.py 2
```
```log
TODO
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





## MMPretrain
Improve the backbone with better features in pretraining

Documentation: https://mmpretrain.readthedocs.io/en/stable/notes/pretrain_custom_dataset.html

- Config: model/mmdetection/configs/rtmdet/classification/cspnext-m_8xb256-rsb-a1-600e_in1k.py 
  - [model/mmdetection/configs/rtmdet/classification/README.md](mmdetection/configs/rtmdet/classification/README.md)

```bash
cd mmpretrain
bash ./tools/dist_train.sh cspnext-m_8xb256-rsb-a1-600e_in1k.py 4
```


```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29602  bash ./tools/dist_train.sh cspnext-l_8xb256-rsb-a1-600e_in1k.py 4
```