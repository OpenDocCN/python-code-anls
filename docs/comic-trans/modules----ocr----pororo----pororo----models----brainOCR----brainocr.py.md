# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\brainocr.py`

```py
"""
This code is primarily based on the following:
https://github.com/JaidedAI/EasyOCR/blob/8af936ba1b2f3c230968dc1022d0cd3e9ca1efbb/easyocr/easyocr.py

Basic usage:
>>> from pororo import Pororo
>>> ocr = Pororo(task="ocr", lang="ko")
>>> ocr("IMAGE_FILE")
"""

import ast  # 导入ast模块，用于解析字符串为Python表达式
from logging import getLogger  # 导入getLogger函数，用于获取日志记录器
from typing import List  # 导入List泛型，用于声明列表类型

import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于科学计算
from PIL import Image  # 导入PIL库中的Image模块，用于图像处理

from .detection import get_detector, get_textbox  # 从自定义模块中导入检测相关函数
from .recognition import get_recognizer, get_text  # 从自定义模块中导入识别相关函数
from .utils import (  # 从自定义模块中导入多个实用工具函数
    diff,
    get_image_list,
    get_paragraph,
    group_text_box,
    reformat_input,
)

LOGGER = getLogger(__name__)  # 获取当前模块的日志记录器对象


class Reader(object):
    
    def __init__(
        self,
        lang: str,
        det_model_ckpt_fp: str,
        rec_model_ckpt_fp: str,
        opt_fp: str,
        device: str,
    ) -> None:
        """
        TODO @karter: modify this such that you download the pretrained checkpoint files
        Parameters:
            lang: language code. e.g, "en" or "ko"
            det_model_ckpt_fp: Detection model's checkpoint path e.g., 'craft_mlt_25k.pth'
            rec_model_ckpt_fp: Recognition model's checkpoint path
            opt_fp: option file path
        """
        # Plug options in the dictionary
        opt2val = self.parse_options(opt_fp)  # 调用parse_options方法解析选项文件，返回字典
        opt2val["vocab"] = self.build_vocab(opt2val["character"])  # 构建词汇表并添加到opt2val字典中
        opt2val["vocab_size"] = len(opt2val["vocab"])  # 计算词汇表大小并存储在opt2val字典中
        opt2val["device"] = device  # 将设备信息添加到opt2val字典中
        opt2val["lang"] = lang  # 将语言信息添加到opt2val字典中
        opt2val["det_model_ckpt_fp"] = det_model_ckpt_fp  # 将检测模型的路径添加到opt2val字典中
        opt2val["rec_model_ckpt_fp"] = rec_model_ckpt_fp  # 将识别模型的路径添加到opt2val字典中

        # Get model objects
        self.detector = get_detector(det_model_ckpt_fp, opt2val["device"])  # 使用get_detector函数获取检测器对象
        self.recognizer, self.converter = get_recognizer(opt2val)  # 使用get_recognizer函数获取识别器和转换器对象
        self.opt2val = opt2val  # 将配置字典存储在对象实例中

    @staticmethod
    def parse_options(opt_fp: str) -> dict:
        opt2val = dict()  # 初始化一个空字典opt2val，用于存储选项及其值
        for line in open(opt_fp, "r", encoding="utf8"):  # 打开选项文件进行逐行读取
            line = line.strip()  # 去除行首尾的空白字符
            if ": " in line:  # 判断是否包含": "，表示该行为选项及其值的格式
                opt, val = line.split(": ", 1)  # 使用": "分割行，得到选项和值
                try:
                    opt2val[opt] = ast.literal_eval(val)  # 尝试将值解析为Python表达式，并存储在opt2val字典中
                except:
                    opt2val[opt] = val  # 如果解析失败，则直接将原始字符串值存储在opt2val字典中

        return opt2val  # 返回解析后的选项字典

    @staticmethod
    def build_vocab(character: str) -> List[str]:
        """Returns vocabulary (=list of characters)"""
        vocab = ["[blank]"] + list(character)  # 构建包含空白标记的字符列表作为词汇表
        return vocab  # 返回词汇表列表
    def detect(self, img: np.ndarray, opt2val: dict):
        """
        :return:
            horizontal_list (list): e.g., [[613, 1496, 51, 190], [136, 1544, 134, 508]]
            free_list (list): e.g., []
        """
        # 使用文本检测器获取图像中的文本框列表
        text_box = get_textbox(self.detector, img, opt2val)
        
        # 根据指定参数对文本框进行分组，得到水平方向和自由排列的文本框列表
        horizontal_list, free_list = group_text_box(
            text_box,
            opt2val["slope_ths"],     # 斜率阈值
            opt2val["ycenter_ths"],   # y 中心点阈值
            opt2val["height_ths"],    # 高度阈值
            opt2val["width_ths"],     # 宽度阈值
            opt2val["add_margin"],    # 添加边距
        )

        min_size = opt2val["min_size"]
        if min_size:
            # 根据最小尺寸过滤水平文本框列表
            horizontal_list = [
                i for i in horizontal_list
                if max(i[1] - i[0], i[3] - i[2]) > min_size
            ]
            # 根据最小尺寸过滤自由排列文本框列表
            free_list = [
                i for i in free_list
                if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size
            ]

        # 返回处理后的水平文本框列表和自由排列文本框列表
        return horizontal_list, free_list

    def recognize(
        self,
        img_cv_grey: np.ndarray,
        horizontal_list: list,
        free_list: list,
        opt2val: dict,
    ):
        """
        Read text in the image
        :return:
            result (list): bounding box, text and confident score
                e.g., [([[189, 75], [469, 75], [469, 165], [189, 165]], '愚园路', 0.3754989504814148),
                 ([[86, 80], [134, 80], [134, 128], [86, 128]], '西', 0.40452659130096436),
                 ([[517, 81], [565, 81], [565, 123], [517, 123]], '东', 0.9989598989486694),
                 ([[78, 126], [136, 126], [136, 156], [78, 156]], '315', 0.8125889301300049),
                 ([[514, 126], [574, 126], [574, 156], [514, 156]], '309', 0.4971577227115631),
                 ([[226, 170], [414, 170], [414, 220], [226, 220]], 'Yuyuan Rd.', 0.8261902332305908),
                 ([[79, 173], [125, 173], [125, 213], [79, 213]], 'W', 0.9848111271858215),
                 ([[529, 173], [569, 173], [569, 213], [529, 213]], 'E', 0.8405593633651733)]
             or list of texts (if skip_details is True)
                e.g., ['愚园路', '西', '东', '315', '309', 'Yuyuan Rd.', 'W', 'E']
        """
        imgH = opt2val["imgH"]  # 从 opt2val 字典中获取图片高度
        paragraph = opt2val["paragraph"]  # 从 opt2val 字典中获取是否段落模式的标志
        skip_details = opt2val["skip_details"]  # 从 opt2val 字典中获取是否跳过详细信息的标志

        if (horizontal_list is None) and (free_list is None):
            y_max, x_max = img_cv_grey.shape  # 获取灰度图像的高度和宽度
            ratio = x_max / y_max  # 计算宽高比
            max_width = int(imgH * ratio)  # 计算根据图片高度等比缩放后的最大宽度
            crop_img = cv2.resize(
                img_cv_grey,
                (max_width, imgH),  # 调整图像尺寸为指定的最大宽度和图片高度
                interpolation=Image.LANCZOS,
            )
            image_list = [([[0, 0], [x_max, 0], [x_max, y_max],
                            [0, y_max]], crop_img)]  # 将裁剪后的图像和其边界框添加到图像列表中
        else:
            image_list, max_width = get_image_list(
                horizontal_list,  # 水平文本线列表
                free_list,  # 自由文本列表
                img_cv_grey,  # 灰度图像
                model_height=imgH,  # 模型期望的高度
            )

        result = get_text(image_list, self.recognizer, self.converter, opt2val)  # 通过图像列表获取文本识别结果

        if paragraph:
            result = get_paragraph(result, mode="ltr")  # 如果开启了段落模式，则对识别结果进行段落处理

        if skip_details:  # 如果只需要文本内容而非详细信息
            return [item[1] for item in result]  # 返回结果中的文本内容列表
        else:  # 如果需要完整的输出：边界框、文本内容和置信度分数
            return result  # 返回完整的识别结果
        """
        Detect text in the image and then recognize it.
        :param image: file path or numpy-array or a byte stream object  # 图像参数可以是文件路径、numpy数组或字节流对象
        :param batch_size: number of images to process in parallel  # 并行处理的图像数量
        :param n_workers: number of worker threads for processing  # 处理的工作线程数
        :param skip_details: whether to skip detailed information in detection  # 是否跳过检测中的详细信息
        :param paragraph: whether to detect paragraph structures  # 是否检测段落结构
        :param min_size: minimum size of text regions to detect  # 检测到的文本区域的最小尺寸
        :param contrast_ths: contrast threshold for text detection  # 文本检测的对比度阈值
        :param adjust_contrast: whether to adjust image contrast before detection  # 是否在检测之前调整图像对比度
        :param filter_ths: threshold for filtering text after detection  # 检测后过滤文本的阈值
        :param text_threshold: threshold for text confidence during recognition  # 识别过程中文本置信度的阈值
        :param low_text: threshold for low confidence text regions  # 低置信度文本区域的阈值
        :param link_threshold: threshold for merging text regions  # 合并文本区域的阈值
        :param canvas_size: size of the output image canvas  # 输出图像的画布大小
        :param mag_ratio: magnification ratio for image resizing  # 图像缩放的放大比例
        :param slope_ths: threshold for text line slope estimation  # 文本行斜率估计的阈值
        :param ycenter_ths: threshold for text center y estimation  # 文本中心y坐标估计的阈值
        :param height_ths: threshold for text height estimation  # 文本高度估计的阈值
        :param width_ths: threshold for text width estimation  # 文本宽度估计的阈值
        :param add_margin: whether to add margin to the bounding box of text regions  # 是否为文本区域的边界框添加边距
        :return: recognized text result from the image  # 图像中识别到的文本结果
        """

        # update `opt2val`
        self.opt2val["batch_size"] = batch_size  # 更新批处理大小选项
        self.opt2val["n_workers"] = n_workers  # 更新工作线程数选项
        self.opt2val["skip_details"] = skip_details  # 更新是否跳过详细信息选项
        self.opt2val["paragraph"] = paragraph  # 更新是否检测段落结构选项
        self.opt2val["min_size"] = min_size  # 更新最小文本区域尺寸选项
        self.opt2val["contrast_ths"] = contrast_ths  # 更新对比度阈值选项
        self.opt2val["adjust_contrast"] = adjust_contrast  # 更新是否调整对比度选项
        self.opt2val["filter_ths"] = filter_ths  # 更新过滤文本阈值选项
        self.opt2val["text_threshold"] = text_threshold  # 更新文本置信度阈值选项
        self.opt2val["low_text"] = low_text  # 更新低置信度文本区域阈值选项
        self.opt2val["link_threshold"] = link_threshold  # 更新合并文本区域阈值选项
        self.opt2val["canvas_size"] = canvas_size  # 更新输出画布大小选项
        self.opt2val["mag_ratio"] = mag_ratio  # 更新图像缩放放大比例选项
        self.opt2val["slope_ths"] = slope_ths  # 更新文本行斜率估计阈值选项
        self.opt2val["ycenter_ths"] = ycenter_ths  # 更新文本中心y坐标估计阈值选项
        self.opt2val["height_ths"] = height_ths  # 更新文本高度估计阈值选项
        self.opt2val["width_ths"] = width_ths  # 更新文本宽度估计阈值选项
        self.opt2val["add_margin"] = add_margin  # 更新是否为文本边界框添加边距选项

        img, img_cv_grey = reformat_input(image)  # 对输入图像进行格式转换，返回图像数组和灰度图像数组

        horizontal_list, free_list = self.detect(img, self.opt2val)  # 使用给定参数检测图像中的文本，返回水平文本区域列表和自由文本区域列表
        result = self.recognize(  # 使用给定参数识别图像中的文本
            img_cv_grey,  # 灰度图像输入
            horizontal_list,  # 水平文本区域列表
            free_list,  # 自由文本区域列表
            self.opt2val,  # 参数字典
        )

        return result  # 返回识别结果
```