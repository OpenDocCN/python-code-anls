# `.\PaddleOCR\deploy\pdserving\ocr_cpp_client.py`

```py
# 导入所需的库和模块
from paddle_serving_client import Client
import sys
import numpy as np
import base64
import os
import cv2
from paddle_serving_app.reader import Sequential, URL2Image, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose
from ocr_reader import OCRReader
import codecs

# 创建客户端对象
client = Client()
# 加载客户端配置，需要加载多个客户端模型
# 需要进一步研究一些细节
client.load_client_config(sys.argv[1:])
# 连接到服务器
client.connect(["127.0.0.1:8181"])

# 导入PaddlePaddle库
import paddle
# 测试图片目录
test_img_dir = "../../doc/imgs/1.jpg"

# 创建OCRReader对象，指定字符字典路径
ocr_reader = OCRReader(char_dict_path="../../ppocr/utils/ppocr_keys_v1.txt")

# 将OpenCV格式的图片转换为base64编码
def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

# 检查图片文件是否符合要求
def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif'}
    return any([path.lower().endswith(e) for e in img_end])

# 初始化测试图片列表
test_img_list = []
# 如果测试图片目录是文件且是图片文件，则将其添加到测试图片列表中
if os.path.isfile(test_img_dir) and _check_image_file(test_img_dir):
    test_img_list.append(test_img_dir)
# 如果测试图片目录是文件夹，则遍历文件夹中的文件
elif os.path.isdir(test_img_dir):
    for single_file in os.listdir(test_img_dir):
        file_path = os.path.join(test_img_dir, single_file)
        # 如果文件是图片文件，则将其添加到测试图片列表中
        if os.path.isfile(file_path) and _check_image_file(file_path):
            test_img_list.append(file_path)
# 如果测试图片列表为空
if len(test_img_list) == 0:
    # 抛出异常，指示在指定的测试图片目录中未找到任何图像文件
    raise Exception("not found any img file in {}".format(test_img_dir))
# 遍历测试图片列表中的每个图片文件
for img_file in test_img_list:
    # 以二进制只读方式打开图片文件
    with open(img_file, 'rb') as file:
        # 读取图片文件的数据
        image_data = file.read()
    # 将图片数据转换为 base64 编码的字符串
    image = cv2_to_base64(image_data)
    # 初始化结果列表
    res_list = []
    # 使用客户端进行预测，传入图片数据，不返回任何结果，启用批处理模式
    fetch_map = client.predict(feed={"x": image}, fetch=[], batch=True)
    # 如果没有返回结果
    if fetch_map is None:
        # 打印提示信息
        print('no results')
    else:
        # 如果返回结果中包含文本信息
        if "text" in fetch_map:
            # 遍历文本信息列表
            for x in fetch_map["text"]:
                # 对文本信息进行编码
                x = codecs.encode(x)
                # 将编码后的文本信息解码为 UTF-8 格式的字符串
                words = base64.b64decode(x).decode('utf-8')
                # 将解码后的文本信息添加到结果列表中
                res_list.append(words)
        else:
            # 尝试对返回结果进行后处理，包括计算得分
            try:
                # 对单批次结果进行后处理，包括计算得分
                one_batch_res = ocr_reader.postprocess(fetch_map, with_score=True)
                # 遍历单批次结果列表
                for res in one_batch_res:
                    # 将结果添加到结果列表中
                    res_list.append(res[0])
            except:
                # 如果出现异常，打印提示信息
                print('no results')
        # 将结果列表转换为字符串，存储在字典中
        res = {"res": str(res_list)}
        # 打印结果字典
        print(res)
```