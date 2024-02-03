# `.\PaddleOCR\deploy\pdserving\pipeline_rpc_client.py`

```py
# 导入所需的库和模块
try:
    from paddle_serving_server_gpu.pipeline import PipelineClient
except ImportError:
    from paddle_serving_server.pipeline import PipelineClient
import numpy as np
import requests
import json
import cv2
import base64
import os

# 创建 PipelineClient 对象并连接到指定的服务器地址
client = PipelineClient()
client.connect(['127.0.0.1:18091'])

# 将 OpenCV 图像数据转换为 base64 编码的字符串
def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

# 解析命令行参数
import argparse
parser = argparse.ArgumentParser(description="args for paddleserving")
parser.add_argument("--image_dir", type=str, default="../../doc/imgs/")
args = parser.parse_args()
test_img_dir = args.image_dir

# 遍历指定目录下的所有图像文件
for img_file in os.listdir(test_img_dir):
    # 读取图像文件的二进制数据
    with open(os.path.join(test_img_dir, img_file), 'rb') as file:
        image_data = file.read()
    # 将图像数据转换为 base64 编码的字符串
    image = cv2_to_base64(image_data)

    # 发送图像数据进行预测，并获取结果
    for i in range(1):
        ret = client.predict(feed_dict={"image": image}, fetch=["res"])
        # 打印预测结果
        print(ret)
```