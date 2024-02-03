# `.\PaddleOCR\deploy\pdserving\win\ocr_web_server.py`

```
# 导入所需的库和模块
from paddle_serving_client import Client
import cv2
import sys
import numpy as np
import os
from paddle_serving_app.reader import Sequential, URL2Image, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose
from paddle_serving_app.reader import DBPostProcess, FilterBoxes, GetRotateCropImage, SortedBoxes
from ocr_reader import OCRReader
try:
    from paddle_serving_server_gpu.web_service import WebService
except ImportError:
    from paddle_serving_server.web_service import WebService
from paddle_serving_app.local_predict import LocalPredictor
import time
import re
import base64

# 定义 OCRService 类，继承自 WebService 类
class OCRService(WebService):
    # 初始化检测模型调试器
    def init_det_debugger(self, det_model_config):
        # 定义检测模型的预处理步骤
        self.det_preprocess = Sequential([
            ResizeByFactor(32, 960), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                (2, 0, 1))
        ])
        # 创建本地预测器对象
        self.det_client = LocalPredictor()
        # 根据命令行参数选择使用 GPU 还是 CPU 加载模型配置
        if sys.argv[1] == 'gpu':
            self.det_client.load_model_config(
                det_model_config, use_gpu=True, gpu_id=0)
        elif sys.argv[1] == 'cpu':
            self.det_client.load_model_config(det_model_config)
        # 初始化 OCRReader 对象，指定字符字典路径
        self.ocr_reader = OCRReader(
            char_dict_path="../../../ppocr/utils/ppocr_keys_v1.txt")
    # 对输入数据进行预处理，包括解码、转换为图像、检测预处理等操作
    def preprocess(self, feed=[], fetch=[]):
        # 解码 base64 编码的图像数据
        data = base64.b64decode(feed[0]["image"].encode('utf8'))
        # 将解码后的数据转换为 numpy 数组
        data = np.fromstring(data, np.uint8)
        # 使用 OpenCV 解码图像数据
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # 获取原始图像的高度和宽度
        ori_h, ori_w, _ = im.shape
        # 对检测图像进行预处理
        det_img = self.det_preprocess(im)
        # 获取预处理后的检测图像的高度和宽度
        _, new_h, new_w = det_img.shape
        # 将检测图像转换为 4 维数组
        det_img = det_img[np.newaxis, :]
        # 复制检测图像数据
        det_img = det_img.copy()
        # 使用检测模型进行预测
        det_out = self.det_client.predict(
            feed={"x": det_img}, fetch=["save_infer_model/scale_0.tmp_1"], batch=True)
        # 创建过滤框对象
        filter_func = FilterBoxes(10, 10)
        # 创建后处理对象
        post_func = DBPostProcess({
            "thresh": 0.3,
            "box_thresh": 0.5,
            "max_candidates": 1000,
            "unclip_ratio": 1.5,
            "min_size": 3
        })
        # 创建排序框对象
        sorted_boxes = SortedBoxes()
        # 计算高度和宽度的比例
        ratio_list = [float(new_h) / ori_h, float(new_w) / ori_w]
        # 对检测框进行后处理
        dt_boxes_list = post_func(det_out["save_infer_model/scale_0.tmp_1"], [ratio_list])
        # 过滤检测框
        dt_boxes = filter_func(dt_boxes_list[0], [ori_h, ori_w])
        # 对检测框进行排序
        dt_boxes = sorted_boxes(dt_boxes)
        # 创建旋转裁剪图像对象
        get_rotate_crop_image = GetRotateCropImage()
        # 初始化图像列表
        img_list = []
        # 初始化最大宽高比
        max_wh_ratio = 0
        # 遍历检测框，获取旋转裁剪后的图像
        for i, dtbox in enumerate(dt_boxes):
            boximg = get_rotate_crop_image(im, dt_boxes[i])
            img_list.append(boximg)
            h, w = boximg.shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        # 如果图像列表为空，则返回空列表
        if len(img_list) == 0:
            return [], []
        # 获取第一个图像的调整后的宽度和高度
        _, w, h = self.ocr_reader.resize_norm_img(img_list[0], max_wh_ratio).shape
        # 初始化图像数组
        imgs = np.zeros((len(img_list), 3, w, h)).astype('float32')
        # 对图像列表中的每个图像进行调整和归一化
        for id, img in enumerate(img_list):
            norm_img = self.ocr_reader.resize_norm_img(img, max_wh_ratio)
            imgs[id] = norm_img
        # 准备输入数据和输出数据
        feed = {"x": imgs.copy()}
        fetch = ["save_infer_model/scale_0.tmp_1"]
        return feed, fetch, True
    # 对OCR识别结果进行后处理，返回结果字典
    def postprocess(self, feed={}, fetch=[], fetch_map=None):
        # 调用OCR读取器的postprocess方法，获取识别结果和得分
        rec_res = self.ocr_reader.postprocess(fetch_map, with_score=True)
        # 初始化结果列表
        res_lst = []
        # 遍历识别结果列表
        for res in rec_res:
            # 将每个识别结果的第一个元素添加到结果列表中
            res_lst.append(res[0])
        # 将结果列表封装成字典
        res = {"res": res_lst}
        # 返回结果字典
        return res
# 创建 OCR 服务对象，指定名称为 "ocr"
ocr_service = OCRService(name="ocr")
# 加载 OCR 模型配置文件
ocr_service.load_model_config("../ppocr_rec_mobile_2.0_serving")
# 准备 OCR 服务，指定工作目录为 "workdir"，端口为 9292
ocr_service.prepare_server(workdir="workdir", port=9292)
# 初始化文本检测模型调试器，指定检测模型配置文件路径
ocr_service.init_det_debugger(det_model_config="../ppocr_det_mobile_2.0_serving")
# 判断命令行参数是否为 'gpu'，如果是则设置 GPU 设备为 "0"，并运行调试服务
if sys.argv[1] == 'gpu':
    ocr_service.set_gpus("0")
    ocr_service.run_debugger_service(gpu=True)
# 如果命令行参数为 'cpu'，则运行调试服务
elif sys.argv[1] == 'cpu':
    ocr_service.run_debugger_service()
# 运行 OCR Web 服务
ocr_service.run_web_service()
```