## Environment
  - Environment variables
  ```
  export CUDA_HOME="/usr/local/cuda"
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
  conda env remove -n mmdet -y
  conda create -n mmdet python=3.10 -y
  conda activate mmdet
  # Ensure CUDA 11.8
  pip install -r requirements.txt
  ```
  ```
  mim install "mmengine>=0.6.0"
  mim install "mmcv>=2.0.0rc4,<2.2.0"
  mim install "mmdet>=3.0.0,<4.0.0"
  ```

- Alternate Python environment through virtualenv
  ```
  python3.10 -m virtualenv ~/dev/.venv_mmdet
  source ~/dev/.venv_mmdet/bin/activate 
  pip install -r requirements.txt
  pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
  # mim install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
  pip install mmdet==3.3.0

  deactivate  # to get out of environment
  ```

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
- Test inference for submission to leaderboard
```bash
#Sep/05/2024
python test_results.py  ../../dataset/rdd2022/coco/test/images  ./rtmdet_l_rdd2022.py  ./work_dirs/rtmdet_l_rdd2022/epoch_200.pth --out-dir ./work_dirs/rtmdet_l_rdd2022/rdd_test/  --to-labelme --device cuda:0
```

#### Run - RTMDet-M
- Multi 4 GPU # 14 hours for b48 200 epochs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_m_rdd2022.py 4
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_m_rdd2022.py 4 --resume /home/rahul/workspace/vision/orddc2024/model/mmdetection/work_dirs/rtmdet_m_rdd2022/epoch_200.pth
```
- Test inference for submission to leaderboard with/out TTA and threshold 0.25
```bash
python test_results.py  ../../dataset/rdd2022/coco/test/images  ./rtmdet_m_rdd2022.py  ./work_dirs/rtmdet_m_rdd2022/epoch_230.pth  --out-dir ./work_dirs/rtmdet_m_rdd2022/rdd_test/  --to-labelme  --tta --score-thr 0.25
```



- RTMDet-M training with pipeline stage2 switch using custom hooks 50 epoch before end (mmpretrained checkpoint "../mmpretrain/work_dirs/cspnext-m_8xb256-rsb-a1-600e_in1k/20240903_094527/epoch_600.pth")
```bash
CUDA_VISIBLE_DEVICES=1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_m_rdd2022.py 3
```

```log

```

- RTMDet-M multiscale training with coco_plus augmented data
```bash
CUDA_VISIBLE_DEVICES=1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_m_rdd2022.py 3
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