# `.\MinerU\magic_pdf\model\pp_structure_v2.py`

```
# 导入随机数生成模块
import random

# 从 loguru 导入日志记录器
from loguru import logger

# 尝试导入 PPStructure 模块，处理文档结构识别
try:
    from paddleocr import PPStructure
# 如果导入失败，记录错误并退出
except ImportError:
    logger.error('paddleocr not installed, please install by "pip install magic-pdf[lite]"')
    exit(1)

# 将区域坐标转换为边界框格式
def region_to_bbox(region):
    # 获取区域左上角的 x 坐标
    x0 = region[0][0]
    # 获取区域左上角的 y 坐标
    y0 = region[0][1]
    # 获取区域右下角的 x 坐标
    x1 = region[2][0]
    # 获取区域右下角的 y 坐标
    y1 = region[2][1]
    # 返回边界框的坐标列表
    return [x0, y0, x1, y1]

# 自定义 PaddleOCR 模型类
class CustomPaddleModel:
    # 初始化方法，设置 OCR 和日志显示选项
    def __init__(self, ocr: bool = False, show_log: bool = False):
        # 创建 PPStructure 实例，不处理表格
        self.model = PPStructure(table=False, ocr=ocr, show_log=show_log)

    # 重载调用方法，处理输入图像
    def __call__(self, img):
        # 尝试导入 OpenCV 库
        try:
            import cv2
        # 如果导入失败，记录错误并退出
        except ImportError:
            logger.error("opencv-python not installed, please install by pip.")
            exit(1)
        # 将 RGB 图像转换为 BGR 格式以适配 PaddleOCR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 使用模型处理图像，获取结果
        result = self.model(img)
        spans = []  # 用于存储识别的文本跨度

        # 遍历识别结果的每一行
        for line in result:
            # 从结果中移除图像信息
            line.pop("img")
            """
            为 PaddleOCR 输出适配类型编号
            title: 0  # 标题
            text: 1   # 文本
            header: 2 # 被放弃
            footer: 2 # 被放弃
            reference: 1 # 文本或被放弃
            equation: 8 # 行间公式块
            equation: 14 # 行间公式文本
            figure: 3 # 图片
            figure_caption: 4 # 图片描述
            table: 5 # 表格
            table_caption: 6 # 表格描述
            """
            # 根据类型设置类别 ID
            if line["type"] == "title":
                line["category_id"] = 0
            elif line["type"] in ["text", "reference"]:
                line["category_id"] = 1
            elif line["type"] == "figure":
                line["category_id"] = 3
            elif line["type"] == "figure_caption":
                line["category_id"] = 4
            elif line["type"] == "table":
                line["category_id"] = 5
            elif line["type"] == "table_caption":
                line["category_id"] = 6
            elif line["type"] == "equation":
                line["category_id"] = 8
            elif line["type"] in ["header", "footer"]:
                line["category_id"] = 2
            else:
                # 记录未知类型的警告
                logger.warning(f"unknown type: {line['type']}")

            # 兼容不输出分数的 PaddleOCR 版本
            if line.get("score") is None:
                # 随机生成一个 0.5 到 1.0 之间的分数
                line["score"] = 0.5 + random.random() * 0.5

            # 移除结果中的 "res" 键
            res = line.pop("res", None)
            # 如果 "res" 存在且不为空，遍历其内容
            if res is not None and len(res) > 0:
                for span in res:
                    # 创建新的跨度字典，包含类别 ID、边界框、分数和文本
                    new_span = {
                        "category_id": 15,
                        "bbox": region_to_bbox(span["text_region"]),
                        "score": span["confidence"],
                        "text": span["text"],
                    }
                    # 将新的跨度添加到 spans 列表中
                    spans.append(new_span)

        # 如果 spans 列表非空，则将其扩展到结果中
        if len(spans) > 0:
            result.extend(spans)

        # 返回最终的识别结果
        return result
```