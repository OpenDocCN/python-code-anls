# `.\comic-translate\modules\ocr\pororo\main.py`

```py
import cv2  # 导入 OpenCV 库，用于图像处理
from .pororo import Pororo  # 导入 Pororo 库中的 Pororo 类
from .pororo.pororo import SUPPORTED_TASKS  # 导入 Pororo 库中的 SUPPORTED_TASKS
from .utils.image_util import plt_imshow, put_text  # 导入自定义的图像显示和文本放置函数
import warnings  # 导入警告处理模块

warnings.filterwarnings('ignore')  # 忽略警告信息的输出


class PororoOcr:
    def __init__(self, model: str = "brainocr", lang: str = "ko", **kwargs):
        self.model = model  # 初始化 OCR 模型名称
        self.lang = lang  # 初始化 OCR 语言
        self._ocr = Pororo(task="ocr", lang=lang, model=model, **kwargs)  # 初始化 Pororo OCR 实例
        self.img_path = None  # 初始化图像路径为空
        self.ocr_result = {}  # 初始化 OCR 结果为空字典

    def run_ocr(self, img_path: str, debug: bool = False):
        self.img_path = img_path  # 设置图像路径为传入参数
        self.ocr_result = self._ocr(img_path, detail=True)  # 执行 OCR 运行，并获取详细结果

        if self.ocr_result['description']:  # 如果 OCR 结果中包含文本
            ocr_text = self.ocr_result["description"]  # 获取 OCR 识别的文本内容
        else:
            ocr_text = "No text detected."  # 如果未检测到文本，则返回提示信息

        if debug:  # 如果调试模式开启
            self.show_img_with_ocr()  # 显示带有 OCR 结果的图像

        return ocr_text  # 返回 OCR 的文本结果

    @staticmethod
    def get_available_langs():
        return SUPPORTED_TASKS["ocr"].get_available_langs()  # 获取支持的 OCR 语言列表

    @staticmethod
    def get_available_models():
        return SUPPORTED_TASKS["ocr"].get_available_models()  # 获取支持的 OCR 模型列表

    def get_ocr_result(self):
        return self.ocr_result  # 返回最近一次 OCR 的结果

    def get_img_path(self):
        return self.img_path  # 返回当前处理的图像路径

    def show_img(self):
        plt_imshow(img=self.img_path)  # 显示图像的函数

    def show_img_with_ocr(self):
        if isinstance(self.img_path, str):  # 如果图像路径是字符串
            img = cv2.imread(self.img_path)  # 读取图像文件
        else:
            img = self.img_path  # 否则直接使用传入的图像

        roi_img = img.copy()  # 复制一份图像用于绘制 ROI 区域

        for text_result in self.ocr_result['bounding_poly']:
            text = text_result['description']  # 获取文本内容
            tlX = text_result['vertices'][0]['x']  # 左上角 X 坐标
            tlY = text_result['vertices'][0]['y']  # 左上角 Y 坐标
            trX = text_result['vertices'][1]['x']  # 右上角 X 坐标
            trY = text_result['vertices'][1]['y']  # 右上角 Y 坐标
            brX = text_result['vertices'][2]['x']  # 右下角 X 坐标
            brY = text_result['vertices'][2]['y']  # 右下角 Y 坐标
            blX = text_result['vertices'][3]['x']  # 左下角 X 坐标
            blY = text_result['vertices'][3]['y']  # 左下角 Y 坐标

            pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))  # 构建顶点坐标列表

            topLeft = pts[0]  # 左上角点
            topRight = pts[1]  # 右上角点
            bottomRight = pts[2]  # 右下角点
            bottomLeft = pts[3]  # 左下角点

            cv2.line(roi_img, topLeft, topRight, (0, 255, 0), 2)  # 绘制边界框的四条边
            cv2.line(roi_img, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(roi_img, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(roi_img, bottomLeft, topLeft, (0, 255, 0), 2)

            roi_img = put_text(roi_img, text, topLeft[0], topLeft[1] - 20, font_size=15)  # 在图像上放置文本

        plt_imshow(["Original", "ROI"], [img, roi_img], figsize=(16, 10))  # 显示原始图像和带有 ROI 的图像


if __name__ == "__main__":
    ocr = PororoOcr()  # 创建 PororoOcr 的实例
    image_path = input("Enter image path: ")  # 获取用户输入的图像路径
    text = ocr.run_ocr(image_path, debug=True)  # 运行 OCR 并获取文本结果，同时显示调试信息
    print('Result :', text)  # 打印 OCR 的结果文本
```