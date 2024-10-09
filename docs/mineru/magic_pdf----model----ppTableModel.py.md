# `.\MinerU\magic_pdf\model\ppTableModel.py`

```
# 导入所需模块和类
from paddleocr.ppstructure.table.predict_table import TableSystem
from paddleocr.ppstructure.utility import init_args
from magic_pdf.libs.Constants import *
import os
from PIL import Image
import numpy as np

# 定义 ppTableModel 类，用于将表格图像转换为 HTML 格式
class ppTableModel(object):
    """
        该类负责使用预训练模型将表格图像转换为 HTML 格式。

        属性:
        - table_sys: 用解析的参数初始化的 TableSystem 实例。

        方法:
        - __init__(config): 使用配置参数初始化模型。
        - img2html(image): 将 PIL 图像或 NumPy 数组转换为 HTML 字符串。
        - parse_args(**kwargs): 解析配置参数。
    """

    # 初始化方法，接收配置字典作为参数
    def __init__(self, config):
        """
        参数:
        - config (dict): 包含 model_dir 和 device 的配置字典。
        """
        # 解析配置参数并获取参数字典
        args = self.parse_args(**config)
        # 初始化 TableSystem 实例
        self.table_sys = TableSystem(args)

    # 将图像转换为 HTML 格式的方法
    def img2html(self, image):
        """
        参数:
        - image (PIL.Image 或 np.ndarray): 要转换的表格图像。

        返回:
        - HTML (str): 表格内容的 HTML 结构字符串。
        """
        # 检查输入是否为 PIL 图像，如果是则转换为 NumPy 数组
        if isinstance(image, Image.Image):
            image = np.array(image)
        # 使用 table_sys 进行预测，获取预测结果
        pred_res, _ = self.table_sys(image)
        # 提取预测结果中的 HTML 部分
        pred_html = pred_res["html"]
        # 处理 HTML 字符串并封装为表格形式
        res = '<td><table  border="1">' + pred_html.replace("<html><body><table>", "").replace("</table></body></html>",
                                                                                               "") + "</table></td>\n"
        # 返回最终的 HTML 结果
        return res

    # 解析配置参数的方法
    def parse_args(self, **kwargs):
        # 初始化参数解析器
        parser = init_args()
        # 从 kwargs 中获取模型目录
        model_dir = kwargs.get("model_dir")
        # 构造各个模型和字典的路径
        table_model_dir = os.path.join(model_dir, TABLE_MASTER_DIR)
        table_char_dict_path = os.path.join(model_dir, TABLE_MASTER_DICT)
        det_model_dir = os.path.join(model_dir, DETECT_MODEL_DIR)
        rec_model_dir = os.path.join(model_dir, REC_MODEL_DIR)
        rec_char_dict_path = os.path.join(model_dir, REC_CHAR_DICT)
        # 获取设备类型，默认为 CPU
        device = kwargs.get("device", "cpu")
        use_gpu = True if device == "cuda" else False
        # 构造配置字典
        config = {
            "use_gpu": use_gpu,
            "table_max_len": kwargs.get("table_max_len", TABLE_MAX_LEN),
            "table_algorithm": TABLE_MASTER,
            "table_model_dir": table_model_dir,
            "table_char_dict_path": table_char_dict_path,
            "det_model_dir": det_model_dir,
            "rec_model_dir": rec_model_dir,
            "rec_char_dict_path": rec_char_dict_path,
        }
        # 设置解析器的默认值
        parser.set_defaults(**config)
        # 解析并返回参数
        return parser.parse_args([])
```