#### Environment
- Setup an Ubuntu 20.04 or 22.04 instance with a NVIDIA GPU on AWS EC2(https://aws.amazon.com/ec2/instance-types/g4/)
- Install CUDA 11.8 on this instance
  ```
  wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
  sudo sh cuda_11.8.0_520.61.05_linux.run
  ```
- Environment variables check
  ```
  #export CUDA_HOME="/usr/local/cuda"
  export CUDA_HOME="/usr/local/cuda-11.8"
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```
- Python environment
  ```
  conda env remove -n infer
  conda create -n infer python=3.9 -y
  conda activate infer
  # Ensure CUDA 11.8
  pip install -r requirements.txt
  ```
  ```
  mim install mmcv==2.1.0 mmdet==3.3.0
  ```
- Test torch packages
  ```
  python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME)'
  ```

#### Submission inference script

- Perform inference (Setup for 16GB target GPU memory)
  ```
  cd infer/
  CUDA_VISIBLE_DEVICES=1 python inference_script.py
  ```



