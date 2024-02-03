# `.\PaddleOCR\deploy\pdserving\web_service_det.py`

```
# 导入所需的模块和库
from paddle_serving_server.web_service import WebService, Op
import logging
import numpy as np
import cv2
import base64
# 导入自定义的 OCRReader 类和相关函数
from ocr_reader import OCRReader, DetResizeForTest, ArgsParser
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose
from paddle_serving_app.reader import DBPostProcess, FilterBoxes, GetRotateCropImage, SortedBoxes

# 获取 logger 对象
_LOGGER = logging.getLogger()

# 定义一个名为 DetOp 的类，继承 Op 类
class DetOp(Op):
    # 初始化操作
    def init_op(self):
        # 定义预处理操作序列
        self.det_preprocess = Sequential([
            DetResizeForTest(), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                (2, 0, 1))
        ])
        # 定义过滤框函数
        self.filter_func = FilterBoxes(10, 10)
        # 定义后处理函数
        self.post_func = DBPostProcess({
            "thresh": 0.3,
            "box_thresh": 0.5,
            "max_candidates": 1000,
            "unclip_ratio": 1.5,
            "min_size": 3
        })
    # 预处理函数，对输入数据进行预处理操作
    def preprocess(self, input_dicts, data_id, log_id):
        # 从输入字典中获取唯一的输入数据
        (_, input_dict), = input_dicts.items()
        # 将 base64 编码的图像数据解码为二进制数据
        data = base64.b64decode(input_dict["image"].encode('utf8'))
        # 保存原始图像数据
        self.raw_im = data
        # 将二进制数据转换为 numpy 数组
        data = np.fromstring(data, np.uint8)
        # 使用 OpenCV 解码图像数据，以彩色图像方式读取
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # 获取原始图像的高度、宽度和通道数
        self.ori_h, self.ori_w, _ = im.shape
        # 对检测图像进行预处理
        det_img = self.det_preprocess(im)
        # 获取处理后检测图像的高度和宽度
        _, self.new_h, self.new_w = det_img.shape
        # 返回处理后的数据字典和其他信息
        return {"x": det_img[np.newaxis, :].copy()}, False, None, ""

    # 后处理函数，对模型输出进行后处理操作
    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        # 获取模型输出的检测结果
        det_out = list(fetch_dict.values())[0]
        # 计算高度和宽度的缩放比例
        ratio_list = [
            float(self.new_h) / self.ori_h, float(self.new_w) / self.ori_w
        ]
        # 对检测结果进行后处理，得到文本框坐标
        dt_boxes_list = self.post_func(det_out, [ratio_list])
        # 过滤得到的文本框坐标，根据原始图像的高度和宽度
        dt_boxes = self.filter_func(dt_boxes_list[0], [self.ori_h, self.ori_w])
        # 构建输出字典，包含处理后的文本框坐标
        out_dict = {"dt_boxes": str(dt_boxes)}

        # 返回输出字典和其他信息
        return out_dict, None, ""
# 定义一个名为OcrService的类，继承自WebService类
class OcrService(WebService):
    # 定义一个方法，用于获取管道的响应，接受一个read_op参数
    def get_pipeline_response(self, read_op):
        # 创建一个名为det_op的操作，名称为"det"，输入操作为read_op
        det_op = DetOp(name="det", input_ops=[read_op])
        # 返回det_op操作
        return det_op

# 创建一个名为uci_service的OcrService对象，名称为"ocr"
uci_service = OcrService(name="ocr")
# 解析命令行参数，将结果存储在FLAGS中
FLAGS = ArgsParser().parse_args()
# 根据传入的yml_dict参数准备管道配置
uci_service.prepare_pipeline_config(yml_dict=FLAGS.conf_dict)
# 运行uci_service服务
uci_service.run_service()
```