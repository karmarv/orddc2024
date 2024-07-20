


## MMYolo - RTMDet

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
>
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
# Multi 2 GPU # XXX hours for b56 100 epochs
CUDA_VISIBLE_DEVICES=6,7 PORT=29601 ./tools/dist_train.sh rtmdet_m_rdd2022.py 2
```