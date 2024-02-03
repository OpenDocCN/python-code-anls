# `.\PaddleOCR\deploy\pdserving\web_service_rec.py`

```py
# 导入所需的模块和库
from paddle_serving_server.web_service import WebService, Op
import logging
import numpy as np
import cv2
import base64
# 导入自定义的 OCRReader 类、DetResizeForTest 类和 ArgsParser 类
from ocr_reader import OCRReader, DetResizeForTest, ArgsParser
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose

# 获取 logger 对象
_LOGGER = logging.getLogger()

# 定义 RecOp 类，继承 Op 类
class RecOp(Op):
    # 初始化操作
    def init_op(self):
        # 创建 OCRReader 对象，指定字符字典路径
        self.ocr_reader = OCRReader(
            char_dict_path="../../ppocr/utils/ppocr_keys_v1.txt")
    # 对输入数据进行预处理
    def preprocess(self, input_dicts, data_id, log_id):
        # 从输入字典中获取唯一的输入数据
        (_, input_dict), = input_dicts.items()
        # 将 base64 编码的图像数据解码为原始图像数据
        raw_im = base64.b64decode(input_dict["image"].encode('utf8'))
        # 将原始图像数据转换为 numpy 数组
        data = np.fromstring(raw_im, np.uint8)
        # 使用 OpenCV 解码图像数据
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        feed_list = []
        max_wh_ratio = 0
        ## 多个小批次，feed_data 的类型是列表
        max_batch_size = 6  # len(dt_boxes)

        # 如果 max_batch_size 为 0，则跳过预测阶段
        if max_batch_size == 0:
            return {}, True, None, ""
        boxes_size = max_batch_size
        rem = boxes_size % max_batch_size

        # 获取图像的高度和宽度
        h, w = im.shape[0:2]
        # 计算图像的宽高比
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
        # 调整图像大小并获取新的宽高
        _, w, h = self.ocr_reader.resize_norm_img(im, max_wh_ratio).shape
        norm_img = self.ocr_reader.resize_norm_img(im, max_batch_size)
        norm_img = norm_img[np.newaxis, :]
        feed = {"x": norm_img.copy()}
        feed_list.append(feed)
        return feed_list, False, None, ""

    # 对输出数据进行后处理
    def postprocess(self, input_dicts, fetch_data, data_id, log_id):
        res_list = []
        if isinstance(fetch_data, dict):
            if len(fetch_data) > 0:
                # 对每个批次的结果进行后处理
                rec_batch_res = self.ocr_reader.postprocess(
                    fetch_data, with_score=True)
                for res in rec_batch_res:
                    res_list.append(res[0])
        elif isinstance(fetch_data, list):
            for one_batch in fetch_data:
                # 对每个批次的结果进行后处理
                one_batch_res = self.ocr_reader.postprocess(
                    one_batch, with_score=True)
                for res in one_batch_res:
                    res_list.append(res[0])

        # 将结果列表转换为字符串并返回
        res = {"res": str(res_list)}
        return res, None, ""
# 定义一个名为OcrService的类，继承自WebService类
class OcrService(WebService):
    # 定义一个方法，用于获取管道的响应，接受一个read_op参数
    def get_pipeline_response(self, read_op):
        # 创建一个名为rec_op的RecOp对象，设置名称为"rec"，输入操作为read_op
        rec_op = RecOp(name="rec", input_ops=[read_op])
        # 返回rec_op对象
        return rec_op

# 创建一个名为uci_service的OcrService对象，设置名称为"ocr"
uci_service = OcrService(name="ocr")
# 解析命令行参数，返回一个FLAGS对象
FLAGS = ArgsParser().parse_args()
# 准备管道配置，传入yml_dict参数为FLAGS对象的conf_dict属性
uci_service.prepare_pipeline_config(yml_dict=FLAGS.conf_dict)
# 运行服务
uci_service.run_service()
```