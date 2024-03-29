English | [简体中文](README_cn.md)

## Introduction
Many users hope package the PaddleOCR service into a docker image, so that it can be quickly released and used in the docker or K8s environment.

This page provides some standardized code to achieve this goal. You can quickly publish the PaddleOCR project into a callable Restful API service through the following steps. (At present, the deployment based on the HubServing mode is implemented first, and author plans to increase the deployment of the PaddleServing mode in the future)

## 1. Prerequisites

You need to install the following basic components first：
a. Docker
b. Graphics driver and CUDA 10.0+（GPU）
c. NVIDIA Container Toolkit（GPU，Docker 19.03+ can skip this）
d. cuDNN 7.6+（GPU）

## 2. Build Image
a. Go to Dockerfile directory（PS: Need to distinguish between CPU and GPU version, the following takes CPU as an example, GPU version needs to replace the keyword）
```py
cd deploy/docker/hubserving/cpu
```
c. Build image
```py
docker build -t paddleocr:cpu .
```

## 3. Start container
a. CPU version
```py
sudo docker run -dp 8868:8868 --name paddle_ocr paddleocr:cpu
```
b. GPU version (base on NVIDIA Container Toolkit)
```py
sudo nvidia-docker run -dp 8868:8868 --name paddle_ocr paddleocr:gpu
```
c. GPU version (Docker 19.03++)
```py
sudo docker run -dp 8868:8868 --gpus all --name paddle_ocr paddleocr:gpu
```
d. Check service status（If you can see the following statement then it means completed：Successfully installed ocr_system && Running on http://0.0.0.0:8868/）
```py
docker logs -f paddle_ocr
```

## 4. Test
a. Calculate the Base64 encoding of the picture to be recognized (For test purpose, you can use a free online tool such as https://freeonlinetools24.com/base64-image/ )
b. Post a service request（sample request in sample_request.txt）

```py
curl -H "Content-Type:application/json" -X POST --data "{\"images\": [\"Input image Base64 encode(need to delete the code 'data:image/jpg;base64,'）\"]}" http://localhost:8868/predict/ocr_system
```
c. Get response（If the call is successful, the following result will be returned）
```py
{"msg":"","results":[[{"confidence":0.8403433561325073,"text":"约定","text_region":[[345,377],[641,390],[634,540],[339,528]]},{"confidence":0.8131805658340454,"text":"最终相遇","text_region":[[356,532],[624,530],[624,596],[356,598]]}]],"status":"0"}
```
