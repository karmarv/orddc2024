
# ORDDC 2024 Criteria for inference evaluation
1. Instance Specification: The inference time of the submitted model is measured using an AWS EC2 instance of type "g4dn.xlarge".
2. Execution of Inference Script: The submitted inference script will be executed in subprocess module of Python.
3. Evaluation Scope: The inference script is executed for these countries:
     - India
     - Japan
     - Norway
     - United_States
     - Overall_6_countries
4. Consistency Check: To account for the variation in system measurement, the inference process will be repeated three times for each country. The average inference time from these runs will be used for evaluation.
5. Inference Speed Calculation: For each country: Inference Speed = (Inference_Time) / (Number_of_Test_Images)

## (1.) Create AWS EC2 instance of type "g4dn.xlarge"
- Select AMI (OS): Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20240915
- Select Instance: g4dn.xlarge (4 vCPU, 16GB RAM, 100GB SSD)
- Connect to Instance: ssh -i ~/.ssh/MMM-AWS-Instance-Demo-SSH-Key.pem ubuntu@44.242.235.126
- Verify the GPU on this instance
  ```bash
  ubuntu@ip-172-31-42-14:~$ nvidia-smi
  Sun Sep 29 20:31:16 2024
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
  |                                         |                        |               MIG M. |
  |=========================================+========================+======================|
  |   0  Tesla T4                       On  |   00000000:00:1E.0 Off |                    0 |
  | N/A   25C    P8              9W /   70W |       1MiB /  15360MiB |      0%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+

  +-----------------------------------------------------------------------------------------+
  | Processes:                                                                              |
  |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
  |        ID   ID                                                               Usage      |
  |=========================================================================================|
  |  No running processes found                                                             |
  +-----------------------------------------------------------------------------------------+
  ```
- Pull the codebase: `git clone https://github.com/karmarv/orddc2024.git`


---

## (2.) Download test dataset
```bash
mkdir ~/rdd && cd ~/rdd
wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Japan.zip
wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip
wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Czech.zip  
wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_United_States.zip
wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_MotorBike.zip
wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Norway.zip

zip --fixfix RDD2022_Norway.zip --out RDD2022_Norway_v2.zip
rm -rf RDD2022_Norway.zip
```
```bash
unzip "*.zip"
```
#### Prepare the test folder with images
```bash
mkdir -p Overall_6_countries/test

cp -r ./Japan/test/* ./Overall_6_countries/test/
cp -r ./India/test/* ./Overall_6_countries/test/
cp -r ./Czech/test/* ./Overall_6_countries/test/
cp -r ./United_States/test/* ./Overall_6_countries/test/
cp -r ./China_MotorBike/test/* ./Overall_6_countries/test/
cp -r ./Norway/test/* ./Overall_6_countries/test/
```
## (3.) Build the container for evaluation
```bash
DOCKER_BUILDKIT=1  docker build --file Dockerfile --tag vishwakarmarhl/mmdet_orddc2024_infer .
#docker push vishwakarmarhl/mmdet_orddc2024_infer 
```

#### Verify the installation
```bash
docker run -it --rm --gpus all vishwakarmarhl/mmdet_orddc2024_infer nvidia-smi
docker run -it --rm --gpus all vishwakarmarhl/mmdet_orddc2024_infer python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME)'
```
Above command upon execution should return `2.0.1 True /usr/local/cuda`

---

## (4.) Run inference container with test_images data folder mounted from the host

#### (a.) allcountries-m
- India
  ```bash
  export TEST_IMAGES="/home/ubuntu/rdd/India/test"
  docker run -it --rm --gpus all -v ${TEST_IMAGES}:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-m && time python inference_script.py /rdd/images /rdd/test_docker_m_thr29_IN.csv"
  ```
  - Run 1: output log
  ```log
  20240929-165951 - Loading checkpoint ./mmy_rtmm_640w_e250.pth to model
  Loads checkpoint by local backend from path: ./mmy_rtmm_640w_e250.pth
  100%|██████████████████████████████████| 1959/1959 [01:13<00:00, 26.79it/s]
  20240929-170107 - Results written to /rdd/test_docker_m_thr29_IN.csv file
  real    1m20.141s
  user    2m42.364s
  sys     0m10.111s
  ```
  - Run 2
  ```log
  100%| 1959/1959 [01:12<00:00, 27.03it/s]
  20240929-170333 - Results written to /rdd/test_docker_m_thr29_IN2.csv file
  real    1m19.686s
  ```
  - Run 3
  ```log
  100%| 1959/1959 [01:12<00:00, 27.06it/s]
  20240929-170643 - Results written to /rdd/test_docker_m_thr29_IN3.csv file
  real    1m19.571s
  ``` 

- Japan
  ```bash
  export TEST_IMAGES="/home/ubuntu/rdd/Japan/test"
  docker run -it --rm --gpus all -v ${TEST_IMAGES}:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-m && time python inference_script.py /rdd/images /rdd/test_docker_m_thr29_JP.csv"
  ```
  - Run 1: output log
  ```log
  20240929-165053 - Loading checkpoint ./mmy_rtmm_640w_e250.pth to model
  Loads checkpoint by local backend from path: ./mmy_rtmm_640w_e250.pth
  100%|████████████████████████████████████| 2627/2627 [01:30<00:00, 29.09it/s]
  20240929-165225 - Results written to /rdd/test_docker_m_thr29_JP.csv file
  real    1m37.314s
  user    3m8.575s
  sys     0m11.422s
  ```
  - Run 2
  ```log
  100%| 2627/2627 [01:30<00:00, 29.05it/s]
  20240929-165712 - Results written to /rdd/test_docker_m_thr29_JP2.csv file
  real    1m37.448s
  ```
  - Run 3
  ```log
  100%| 2627/2627 [01:30<00:00, 29.00it/s]
  20240929-165910 - Results written to /rdd/test_docker_m_thr29_JP3.csv file
  real    1m37.612s
  ```  

- Norway
  ```bash
  export TEST_IMAGES="/home/ubuntu/rdd/Norway/test"
  docker run -it --rm --gpus all -v ${TEST_IMAGES}:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-m && time python inference_script.py /rdd/images /rdd/test_docker_m_thr29_NO.csv"
  ```
  - Run 1: output log
  ```log
  20240929-163259 - Loading checkpoint ./mmy_rtmm_640w_e250.pth to model
  Loads checkpoint by local backend from path: ./mmy_rtmm_640w_e250.pth
  100%|████████████████████████████████████| 2040/2040 [03:17<00:00, 10.35it/s]
  20240929-163619 - Results written to /rdd/test_docker_m_thr29_NO.csv file
  real    3m24.279s
  user    7m26.822s
  sys     0m34.929s
  ```
  - Run 2
  ```log
  100%| 2040/2040 [03:15<00:00, 10.42it/s]
  20240929-164221 - Results written to /rdd/test_docker_m_thr29_NO2.csv file
  real    3m22.923s
  ```
  - Run 3
  ```log
  100%| 2040/2040 [03:10<00:00, 10.70it/s]
  20240929-164732 - Results written to /rdd/test_docker_m_thr29_NO3.csv file
  real    3m17.636s
  ```
      
- United_States
  ```bash
  export TEST_IMAGES="/home/ubuntu/rdd/United_States/test"
  docker run -it --rm --gpus all -v ${TEST_IMAGES}:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-m && time python inference_script.py /rdd/images /rdd/test_docker_m_thr29_US.csv"
  ```
  - Run 1: output log
  ```log
  20240929-173548 - Loading checkpoint ./mmy_rtmm_640w_e250.pth to model
  Loads checkpoint by local backend from path: ./mmy_rtmm_640w_e250.pth
  100%|███████████████████████████████████| 1200/1200 [00:40<00:00, 29.48it/s]
  20240929-173631 - Results written to /rdd/test_docker_m_thr29_US.csv file
  real    0m47.930s
  user    1m24.896s
  sys     0m3.615s
  ```
  - Run 2:
  ```log
  100%| 1200/1200 [00:40<00:00, 29.60it/s]
  20240929-155611 - Results written to /rdd/test_docker_m_thr29_US2.csv file
  real    0m47.652s
  ```
  - Run 3:  
  ```log
  100%| 1200/1200 [00:40<00:00, 29.58it/s]
  20240929-155717 - Results written to /rdd/test_docker_m_thr29_US3.csv file
  real    0m47.613s
  ```

- Overall_6_countries
  ```bash
  export TEST_IMAGES="/home/ubuntu/rdd/Overall_6_countries/test"
  docker run -it --rm --gpus all -v ${TEST_IMAGES}:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-m && time python inference_script.py /rdd/images /rdd/test_docker_m_thr29_all.csv"
  ```
  - Run 1: output log
  ```log
  20240929-154549 - Loading checkpoint ./mmy_rtmm_640w_e250.pth to model
  Loads checkpoint by local backend from path: ./mmy_rtmm_640w_e250.pth
  100%|█████████████████████████████████████████████████| 9035/9035 [07:10<00:00, 20.98it/s]
  20240929-155302 - Results written to /rdd/test_docker_m_thr29_all.csv file
  real    7m17.705s
  user    15m22.488s
  sys     0m54.530s
  ```
  - Run 2
  ```log
  100%|9035/9035 [07:19<00:00, 20.54it/s]
  20240929-153626 - Results written to /rdd/test_docker_m_thr29_all2.csv file
  real    7m27.937s
  ```
  - Run 3
  ```log
  100%|9035/9035 [07:16<00:00, 20.70it/s]
  20240929-154447 - Results written to /rdd/test_docker_m_thr29_all3.csv file
  real    7m23.920s
  ```
  - GPU usage
  ```bash
  Mon Sep 30 02:54:46 2024
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
  |                                         |                        |               MIG M. |
  |=========================================+========================+======================|
  |   0  Tesla T4                       On  |   00000000:00:1E.0 Off |                    0 |
  | N/A   38C    P0             80W /   70W |     413MiB /  15360MiB |     53%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+

  +-----------------------------------------------------------------------------------------+
  | Processes:                                                                              |
  |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
  |        ID   ID                                                               Usage      |
  |=========================================================================================|
  |    0   N/A  N/A     27709      C   python                                        410MiB |
  +-----------------------------------------------------------------------------------------+
  ```

#### (b.) allcountries-l

- India
  ```bash
  export TEST_IMAGES="/home/ubuntu/rdd/India/test"
  docker run -it --rm --gpus all -v ${TEST_IMAGES}:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-l && time python inference_script.py /rdd/images /rdd/test_docker_l_thr21_IN.csv"
  ```
  - Run 1: output log
  ```log
  20240929-170737 - Loading checkpoint ./mmy_rtml_pre_e250.pth to model
  Loads checkpoint by local backend from path: ./mmy_rtml_pre_e250.pth
  100%|█████████████████████████████████████| 1959/1959 [01:53<00:00, 17.25it/s]
  20240929-170934 - Results written to /rdd/test_docker_l_thr21_IN.csv file
  real    2m1.588s
  user    4m3.981s
  sys     0m11.706s
  ```
  - Run 2: 
  ```log
  100%| 1959/1959 [01:53<00:00, 17.26it/s]
  20240929-171306 - Results written to /rdd/test_docker_l_thr21_IN2.csv file
  real    2m1.582s
  ```
  - Run 3: 
  ```log
  100%| 1959/1959 [01:54<00:00, 17.16it/s]
  20240929-171518 - Results written to /rdd/test_docker_l_thr21_IN3.csv file
  real    2m2.159s
  ```

- Japan
  ```bash
  export TEST_IMAGES="/home/ubuntu/rdd/Japan/test"
  docker run -it --rm --gpus all -v ${TEST_IMAGES}:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-l && time python inference_script.py /rdd/images /rdd/test_docker_l_thr21_JP.csv"
  ```
  - Run 1: output log
  ```log
  20240929-162301 - Loading checkpoint ./mmy_rtml_pre_e250.pth to model
  Loads checkpoint by local backend from path: ./mmy_rtml_pre_e250.pth
  100%|████████████████████████████████| 2627/2627 [02:27<00:00, 17.81it/s]
  20240929-162532 - Results written to /rdd/test_docker_l_thr21_JP.csv file
  real    2m35.466s
  user    4m58.470s
  sys     0m14.438s
  ```
  - Run 2:
  ```log
  100%|2627/2627 [02:26<00:00, 17.98it/s]
  20240929-162838 - Results written to /rdd/test_docker_l_thr21_JP2.csv file
  real    2m33.909s
  ```
  - Run 3:
  ```log
  100%|2627/2627 [02:27<00:00, 17.84it/s]
  20240929-163121 - Results written to /rdd/test_docker_l_thr21_JP3.csv file
  real    2m35.101s  
  ```

- Norway
  ```bash
  export TEST_IMAGES="/home/ubuntu/rdd/Norway/test"
  docker run -it --rm --gpus all -v ${TEST_IMAGES}:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-l && time python inference_script.py /rdd/images /rdd/test_docker_l_thr21_NO.csv"
  ```
  - Run 1: output log
  ```log
  20240929-160805 - Loading checkpoint ./mmy_rtml_pre_e250.pth to model
  Loads checkpoint by local backend from path: ./mmy_rtml_pre_e250.pth
  100%|████████████████████████████████| 2040/2040 [03:49<00:00,  8.90it/s]
  20240929-161158 - Results written to /rdd/test_docker_l_thr21_NO.csv file
  real    3m57.128s
  user    8m23.901s
  sys     0m37.629s  
  ```
  - Run 2: 
  ```log
  100%| 2040/2040 [03:48<00:00,  8.95it/s]
  20240929-161605 - Results written to /rdd/test_docker_l_thr21_NO2.csv file
  real    3m55.821s
  ```
  - Run 3: 
  ```log
  100%| 2040/2040 [03:47<00:00,  8.96it/s]
  20240929-162138 - Results written to /rdd/test_docker_l_thr21_NO3.csv file
  real    3m55.400s
  ```

- United_States
  ```bash
  export TEST_IMAGES="/home/ubuntu/rdd/United_States/test"
  docker run -it --rm --gpus all -v ${TEST_IMAGES}:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-l && time python inference_script.py /rdd/images /rdd/test_docker_l_thr21_US.csv"
  ```
  - Run 1: output log
  ```log
  20240929-155955 - Loading checkpoint ./mmy_rtml_pre_e250.pth to model
  Loads checkpoint by local backend from path: ./mmy_rtml_pre_e250.pth
  100%|████████████████████████████████| 1200/1200 [01:05<00:00, 18.39it/s]
  20240929-160104 - Results written to /rdd/test_docker_l_thr21_US.csv file

  real    1m13.555s
  user    2m13.508s
  sys     0m4.876s
  ```
  - Run 2: 
  ```log
  100%| 1200/1200 [01:05<00:00, 18.35it/s]
  20240929-160343 - Results written to /rdd/test_docker_l_thr21_US2.csv file
  real    1m13.494s
  ```
  - Run 3: 
  ```log
  100%| 1200/1200 [01:05<00:00, 18.27it/s]
  20240929-160524 - Results written to /rdd/test_docker_l_thr21_US3.csv file
  real    1m13.526s
  ```

- Overall_6_countries
  ```bash
  export TEST_IMAGES="/home/ubuntu/rdd/Overall_6_countries/test"
  docker run -it --rm --gpus all -v ${TEST_IMAGES}:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-l && time python inference_script.py /rdd/images /rdd/test_docker_l_thr21_all.csv"
  ```
  - Run 1: output log and 
  ```log
  20240929-142317 - Loading checkpoint ./mmy_rtml_pre_e250.pth to model
  Loads checkpoint by local backend from path: ./mmy_rtml_pre_e250.pth
  100%|████████████████████████████████| 9035/9035 [10:26<00:00, 14.42it/s]
  20240929-143348 - Results written to /rdd/test_docker_l_thr21_all.csv file

  real    10m36.184s
  user    21m51.259s
  sys     1m4.813s
  ```
  - Run 2: 
  ```log
  100%|9035/9035 [10:27<00:00, 14.41it/s]
  20240929-151320 - Results written to /rdd/test_docker_l_thr21_all2.csv file
  real    10m35.500s
  ```
  - Run 3:
  ```log
  100%| 9035/9035 [10:27<00:00, 14.40it/s]
  20240929-172754 - Results written to /rdd/test_docker_l_thr21_all3.csv file
  real    10m35.711s
  ```
  - GPU usage
  ```bash
  Mon Sep 30 02:52:58 2024
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
  |                                         |                        |               MIG M. |
  |=========================================+========================+======================|
  |   0  Tesla T4                       On  |   00000000:00:1E.0 Off |                    0 |
  | N/A   40C    P0             58W /   70W |     561MiB /  15360MiB |     83%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+

  +-----------------------------------------------------------------------------------------+
  | Processes:                                                                              |
  |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
  |        ID   ID                                                               Usage      |
  |=========================================================================================|
  |    0   N/A  N/A     27517      C   python                                        558MiB |
  +-----------------------------------------------------------------------------------------+
  ```
