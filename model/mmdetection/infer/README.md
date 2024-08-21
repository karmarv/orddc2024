### Environment
- Python 3.8.18 Installation via Miniconda v23.1.0 - https://docs.conda.io/projects/miniconda/en/latest/ 
  ```bash
  conda env remove -n infer
  conda create -n infer python=3.9 -y
  conda activate infer
  ```
- MMDetection environment
  - MMCV CUDA12.1 version for PyTorch 2.1.0 - https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip
  ```bash
  pip install -r requirements.txt
  ```
