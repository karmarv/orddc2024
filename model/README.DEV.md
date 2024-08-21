
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

#### Submission inference script
```
cd infer/
python inference_script.py
```

## (B.) MMYolo - RTMDet

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
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.159
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.324
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.137
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.081
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.206
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.260
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.628
07/20 01:34:30 - mmengine - INFO -
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
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.179
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.352
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.159
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.144
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.236
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.272
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.432
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.537
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.642
07/20 21:47:57 - mmengine - INFO -
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
- RTMDet-M training with pipeline stage2 switch using custom hooks 20 epoch before end
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
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.180
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.350
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.159
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.161
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.237
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.640
07/22 03:41:01 - mmengine - INFO -
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| D00      | 0.19  | 0.357  | 0.183  | 0.092 | 0.147 | 0.256 |
| D10      | 0.205 | 0.4    | 0.179  | 0.082 | 0.161 | 0.342 |
| D20      | 0.212 | 0.398  | 0.2    | 0.365 | 0.087 | 0.232 |
| D40      | 0.11  | 0.247  | 0.074  | 0.107 | 0.112 | 0.118 |
+----------+-------+--------+--------+-------+-------+-------+
```