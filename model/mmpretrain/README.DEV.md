



## MMPretrain
Improve the backbone with better features in pretraining

Documentation: https://mmpretrain.readthedocs.io/en/stable/notes/pretrain_custom_dataset.html

- Config: model/mmdetection/configs/rtmdet/classification/cspnext-m_8xb256-rsb-a1-600e_in1k.py 
  - [model/mmdetection/configs/rtmdet/classification/README.md](mmdetection/configs/rtmdet/classification/README.md)

- V1
```bash
cd mmpretrain
bash ./tools/dist_train.sh cspnext-m_8xb256-rsb-a1-600e_in1k.py 4
```

- V2
```bash
cd mmpretrain
CUDA_VISIBLE_DEVICES=4,5 PORT=29602  bash ./tools/dist_train.sh cspnext-l_8xb256-rsb-a1-600e_in1k.py 2
```