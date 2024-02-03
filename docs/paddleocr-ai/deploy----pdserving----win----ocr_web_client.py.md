# `.\PaddleOCR\deploy\pdserving\win\ocr_web_client.py`

```
# 导入需要的库
import requests
import json
import cv2
import base64
import os, sys
import time

# 将 OpenCV 图像转换为 base64 编码的字符串
def cv2_to_base64(image):
    #data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(image).decode(
        'utf8')  #data.tostring()).decode('utf8')

# 设置请求头信息
headers = {"Content-type": "application/json"}
# 设置请求的 URL
url = "http://127.0.0.1:9292/ocr/prediction"

# 设置测试图片目录
test_img_dir = "../../../doc/imgs/"
# 遍历测试图片目录下的所有图片文件
for idx, img_file in enumerate(os.listdir(test_img_dir)):
    # 读取图片文件内容
    with open(os.path.join(test_img_dir, img_file), 'rb') as file:
        image_data1 = file.read()

    # 将图片文件内容转换为 base64 编码的字符串
    image = cv2_to_base64(image_data1)
    # 发送请求，获取预测结果
    for i in range(1):
        data = {"feed": [{"image": image}], "fetch": ["save_infer_model/scale_0.tmp_1"]}
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        # 打印预测结果
        print(r.json())

# 再次设置测试图片目录
test_img_dir = "../../../doc/imgs/"
# 打印测试图片总数
print("==> total number of test imgs: ", len(os.listdir(test_img_dir)))
```