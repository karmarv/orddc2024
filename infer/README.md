## Step (1) - Environment Setup with CUDA 11.8
- Setup an Ubuntu 20.04 or 22.04 instance with a NVIDIA GPU on AWS EC2(https://aws.amazon.com/ec2/instance-types/g4/)
- Install CUDA 11.8 on this instance
  ```
  wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
  sudo sh cuda_11.8.0_520.61.05_linux.run
  ```
- Environment variables check
  ```
  export CUDA_HOME="/usr/local/cuda-11.8"
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```

### Step (2) - Python environment with conda
- Install miniconda on linux using instructions from https://docs.anaconda.com/miniconda/
- Create virtual environment
  ```
  conda env remove -n mm118 -y
  conda create -n mm118 python=3.9 -y
  conda activate mm118
  ```
- Install required packages
  ```
  # Ensure CUDA 11.8 based packages
  pip install -r requirements.txt
  ```
  - Issue with the mmcv==2.0.1 package installation through requirement.txt file. This prebuilt binary is available only for CUDA 11.8. Need reinstallation as mentioned below.
  ```
  pip uninstall mmcv
  mim install "mmcv>=2.0.0rc4,<2.1.0"
  ```
- Verify and test torch packages
  ```
  python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME)'
  ```

## Step (3) - Submission inference script
- Ensure that the *.pth file is available in the same path.

#### (a.) All countries inference for the large model
```bash
cd ~/orddc2024/infer/allcountries-l
time CUDA_VISIBLE_DEVICES=0 python inference_script.py ../../dataset/rdd2022/coco/test/overall_6_countries/ ./test_l_thr21.csv
```
- execution log
```log
20240912-171008 - Loading checkpoint ./mmy_rtml_pre_e250.pth to model
Loads checkpoint by local backend from path: ./mmy_rtml_pre_e250.pth
100%|█████████████████████████████████████████████████████| 9035/9035 [04:45<00:00, 31.63it/s]
20240912-171456 - Results written to ./test_l_thr21.csv file

real    4m50.406s
user    119m57.387s
sys     0m53.064s
```



#### (b.) All countries inference for the medium model
```bash
cd ~/orddc2024/infer/allcountries-m
time CUDA_VISIBLE_DEVICES=0 python inference_script.py ../../dataset/rdd2022/coco/test/overall_6_countries/ ./test_m_thr29.csv
```
- execution log
```log
20240912-171556 - Loading checkpoint ./mmy_rtmm_640w_e250.pth to model
Loads checkpoint by local backend from path: ./mmy_rtmm_640w_e250.pth
100%|█████████████████████████████████████████████████████| 9035/9035 [04:32<00:00, 33.19it/s]
20240912-172030 - Results written to ./test_m_thr29.csv file

real    4m36.206s
user    113m26.363s
sys     1m30.325s
```