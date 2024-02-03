# `.\PaddleOCR\deploy\pdserving\pipeline_http_client.py`

```py
# 导入所需的库
import numpy as np
import requests
import json
import base64
import os

# 导入 argparse 库，用于解析命令行参数
import argparse

# 定义一个函数，将字符串转换为布尔值
def str2bool(v):
    return v.lower() in ("true", "t", "1")

# 创建 ArgumentParser 对象，用于解析命令行参数
parser = argparse.ArgumentParser(description="args for paddleserving")
# 添加命令行参数，指定图片目录，默认值为 "../../doc/imgs/"
parser.add_argument("--image_dir", type=str, default="../../doc/imgs/")
# 添加命令行参数，指定是否进行目标检测，默认值为 True
parser.add_argument("--det", type=str2bool, default=True)
# 添加命令行参数，指定是否进行文本识别，默认值为 True
parser.add_argument("--rec", type=str2bool, default=True)
# 解析命令行参数
args = parser.parse_args()

# 定义一个函数，将 OpenCV 图像转换为 base64 编码
def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

# 定义一个函数，检查文件是否为图片文件
def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif'}
    return any([path.lower().endswith(e) for e in img_end])

# 定义 OCR 服务的 URL
url = "http://127.0.0.1:9998/ocr/prediction"
# 获取测试图片目录
test_img_dir = args.image_dir

# 初始化测试图片列表
test_img_list = []
# 判断测试图片目录是文件且是图片文件，则将其添加到测试图片列表中
if os.path.isfile(test_img_dir) and _check_image_file(test_img_dir):
    test_img_list.append(test_img_dir)
# 判断测试图片目录是文件夹，则遍历文件夹中的文件，将图片文件添加到测试图片列表中
elif os.path.isdir(test_img_dir):
    for single_file in os.listdir(test_img_dir):
        file_path = os.path.join(test_img_dir, single_file)
        if os.path.isfile(file_path) and _check_image_file(file_path):
            test_img_list.append(file_path)
# 如果测试图片列表为空，则抛出异常
if len(test_img_list) == 0:
    raise Exception("not found any img file in {}".format(test_img_dir))

# 遍历测试图片列表
for idx, img_file in enumerate(test_img_list):
    # 读取图片文件内容
    with open(img_file, 'rb') as file:
        image_data1 = file.read()
    # 打印文件名
    # 打印包含图片文件名的分隔符
    print('{}{}{}'.format('*' * 10, img_file, '*' * 10))

    # 将 OpenCV 图像数据转换为 base64 编码格式
    image = cv2_to_base64(image_data1)

    # 构建包含图片键值对的数据字典
    data = {"key": ["image"], "value": [image]}
    # 发送 POST 请求到指定 URL，并传递 JSON 格式的数据
    r = requests.post(url=url, data=json.dumps(data))
    # 解析响应结果为 JSON 格式
    result = r.json()
    # 打印错误编号和错误消息
    print("erro_no:{}, err_msg:{}".format(result["err_no"], result["err_msg"]))
    # 检查是否成功
    if result["err_no"] == 0:
        # 获取 OCR 结果
        ocr_result = result["value"][0]
        # 如果不需要检测，则直接打印 OCR 结果
        if not args.det:
            print(ocr_result)
        else:
            try:
                # 遍历 OCR 结果并打印文本和坐标
                for item in eval(ocr_result):
                    # 返回转录和坐标
                    print("{}, {}".format(item[0], item[1]))
            except Exception as e:
                # 打印 OCR 结果和提示信息
                print(ocr_result)
                print("No results")
                continue

    else:
        # 打印错误消息的详细信息路径
        print(
            "For details about error message, see PipelineServingLogs/pipeline.log"
        )
# 打印输出测试图片列表的总数
print("==> total number of test imgs: ", len(test_img_list))
```