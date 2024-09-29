


## Self-Verification of inference script on AWS 
#### Setup the AWS Instance
- TODO

---

#### Download test dataset
  ```bash
  wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Japan.zip
  wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip
  wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Czech.zip  
  wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_United_States.zip
  wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_MotorBike.zip
  wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Norway.zip

  zip --fixfix RDD2022_Norway.zip --out RDD2022_Norway_v2.zip
  ```
#### Prepare the test folder with images
    ```bash
    mkdir test_images

    cp ./Japan/test/* ./test_images/
    cp ./India/test/* ./test_images/
    cp ./Czech/test/* ./test_images/
    cp ./United_states/test/* ./test_images/
    cp ./China_MotorBike/test/* ./test_images/
    cp ./Norway/test/* ./test_images/
    ```
#### Build the container and publish it
```bash
DOCKER_BUILDKIT=1  docker build --file Dockerfile --tag vishwakarmarhl/mmdet_orddc2024_infer .
docker push vishwakarmarhl/mmdet_orddc2024_infer 
```

#### Verify the installation
  ```bash
  docker run -it --rm --gpus all vishwakarmarhl/mmdet_orddc2024_infer nvidia-smi
  docker run -it --rm --gpus all vishwakarmarhl/mmdet_orddc2024_infer python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME)'
  ```
#### Run inference container with test_images data folder mounted from the host
- (1.) allcountries-m
```bash
docker run -it --rm --gpus all -v /home/rahul/workspace/data/rdd:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-m && time python inference_script.py /rdd/test_images /rdd/test_docker_m_thr29.csv"
```

- (2.) allcountries-l
```bash
docker run -it --rm --gpus all -v /home/rahul/workspace/data/rdd:/rdd vishwakarmarhl/mmdet_orddc2024_infer /bin/bash -c "cd /infer/allcountries-l && time python inference_script.py /rdd/test_images /rdd/test_docker_l_thr21.csv"
```

---