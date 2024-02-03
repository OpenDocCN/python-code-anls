# `.\PaddleOCR\deploy\pdserving\web_service.py`

```
# 导入所需的模块和库
from paddle_serving_server.web_service import WebService, Op
import logging
import numpy as np
import copy
import cv2
import base64
# 导入自定义的 OCRReader 类
from ocr_reader import OCRReader, DetResizeForTest, ArgsParser
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose
from paddle_serving_app.reader import DBPostProcess, FilterBoxes, GetRotateCropImage, SortedBoxes

# 获取日志记录器
_LOGGER = logging.getLogger()

# 定义一个名为 DetOp 的类，继承自 Op 类
class DetOp(Op):
    # 初始化操作
    def init_op(self):
        # 定义预处理操作序列
        self.det_preprocess = Sequential([
            DetResizeForTest(), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                (2, 0, 1))
        ])
        # 定义过滤框的函数
        self.filter_func = FilterBoxes(10, 10)
        # 定义后处理函数
        self.post_func = DBPostProcess({
            "thresh": 0.3,
            "box_thresh": 0.6,
            "max_candidates": 1000,
            "unclip_ratio": 1.5,
            "min_size": 3
        })
    # 预处理函数，对输入数据进行预处理
    def preprocess(self, input_dicts, data_id, log_id):
        # 从输入字典中获取唯一的输入数据
        (_, input_dict), = input_dicts.items()
        # 将 base64 编码的图像数据解码为二进制数据
        data = base64.b64decode(input_dict["image"].encode('utf8'))
        # 保存原始图像数据
        self.raw_im = data
        # 将二进制数据转换为 numpy 数组
        data = np.fromstring(data, np.uint8)
        # 使用 OpenCV 解码图像数据
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # 获取原始图像的高度、宽度和通道数
        self.ori_h, self.ori_w, _ = im.shape
        # 对检测图像进行预处理
        det_img = self.det_preprocess(im)
        # 获取处理后的检测图像的高度和宽度
        _, self.new_h, self.new_w = det_img.shape
        # 返回处理后的图像数据和标志
        return {"x": det_img[np.newaxis, :].copy()}, False, None, ""

    # 后处理函数，对模型输出进行后处理
    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        # 获取模型输出的检测结果
        det_out = list(fetch_dict.values())[0]
        # 计算高度和宽度的缩放比例
        ratio_list = [
            float(self.new_h) / self.ori_h, float(self.new_w) / self.ori_w
        ]
        # 对检测结果进行后处理
        dt_boxes_list = self.post_func(det_out, [ratio_list])
        # 过滤得到最终的文本框坐标
        dt_boxes = self.filter_func(dt_boxes_list[0], [self.ori_h, self.ori_w])
        # 构建输出字典，包含文本框坐标和原始图像数据
        out_dict = {"dt_boxes": dt_boxes, "image": self.raw_im}
        # 返回输出字典和标志
        return out_dict, None, ""
# 定义一个名为 RecOp 的类，继承自 Op 类
class RecOp(Op):
    # 初始化操作
    def init_op(self):
        # 初始化 OCRReader 对象，指定字符字典路径
        self.ocr_reader = OCRReader(
            char_dict_path="../../ppocr/utils/ppocr_keys_v1.txt")

        # 初始化 GetRotateCropImage 对象
        self.get_rotate_crop_image = GetRotateCropImage()
        # 初始化 SortedBoxes 对象
        self.sorted_boxes = SortedBoxes()

    # 后处理方法，处理输入字典、获取数据、数据 ID 和日志 ID
    def postprocess(self, input_dicts, fetch_data, data_id, log_id):
        # 初始化识别结果列表
        rec_list = []
        # 获取数据列表的长度
        dt_num = len(self.dt_list)
        # 如果 fetch_data 是字典类型
        if isinstance(fetch_data, dict):
            # 如果 fetch_data 长度大于 0
            if len(fetch_data) > 0:
                # 对 fetch_data 进行后处理，包括得分信息
                rec_batch_res = self.ocr_reader.postprocess(
                    fetch_data, with_score=True)
                # 遍历处理后的结果
                for res in rec_batch_res:
                    rec_list.append(res)
        # 如果 fetch_data 是列表类型
        elif isinstance(fetch_data, list):
            # 遍历 fetch_data 中的每个批次
            for one_batch in fetch_data:
                # 对每个批次进行后处理，包括得分信息
                one_batch_res = self.ocr_reader.postprocess(
                    one_batch, with_score=True)
                # 遍历处理后的结果
                for res in one_batch_res:
                    rec_list.append(res)
        # 初始化结果列表
        result_list = []
        # 遍历数据列表的长度
        for i in range(dt_num):
            # 获取文本信息和数据框
            text = rec_list[i]
            dt_box = self.dt_list[i]
            # 如果文本得分大于等于 0.5
            if text[1] >= 0.5:
                # 将文本信息和数据框添加到结果列表中
                result_list.append([text, dt_box.tolist()])
        # 构建结果字典
        res = {"result": str(result_list)}
        # 返回结果字典
        return res, None, ""


# 定义一个名为 OcrService 的类，继承自 WebService 类
class OcrService(WebService):
    # 获取管道响应方法，接收读取操作对象作为参数
    def get_pipeline_response(self, read_op):
        # 初始化检测操作对象，指定名称和输入操作
        det_op = DetOp(name="det", input_ops=[read_op])
        # 初始化识别操作对象，指定名称和输入操作
        rec_op = RecOp(name="rec", input_ops=[det_op])
        # 返回识别操作对象
        return rec_op


# 创建一个名为 uci_service 的 OCR 服务对象
uci_service = OcrService(name="ocr")
# 解析命令行参数
FLAGS = ArgsParser().parse_args()
# 准备管道配置，传入 YAML 字典
uci_service.prepare_pipeline_config(yml_dict=FLAGS.conf_dict)
# 运行 OCR 服务
uci_service.run_service()
```