# `arknights-mower\packaging\paddleocr.py`

```py
# 导入操作系统模块
import os
# 导入系统模块
import sys
# 导入动态导入模块
import importlib

# 获取当前文件所在目录
__dir__ = os.path.dirname(__file__)

# 导入 PaddlePaddle 模块
import paddle

# 将当前文件所在目录添加到系统路径中
sys.path.append(os.path.join(__dir__, ''))

# 导入 OpenCV 模块
import cv2
# 导入日志模块
import logging
# 导入 NumPy 模块
import numpy as np
# 导入路径模块
from pathlib import Path
# 导入 base64 模块
import base64
# 导入字节流模块
from io import BytesIO
# 导入图像处理模块
from PIL import Image

# 动态导入工具模块
tools = importlib.import_module('.', 'tools')
# 动态导入 OCR 模块
ppocr = importlib.import_module('.', 'ppocr')
# 动态导入结构化模块
ppstructure = importlib.import_module('.', 'ppstructure')

# 从工具模块中导入预测系统函数
from tools.infer import predict_system
# 从 OCR 模块中导入日志模块
from ppocr.utils.logging import get_logger

# 获取日志记录器
logger = get_logger()
# 从 OCR 模块中导入工具函数
from ppocr.utils.utility import check_and_read, get_image_file_list
# 从 OCR 模块中导入网络函数
from ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
# 从工具模块中导入预测系统工具函数
from tools.infer.utility import draw_ocr, str2bool, check_gpu
# 从结构化模块中导入工具函数
from ppstructure.utility import init_args, draw_structure_result
# 从结构化模块中导入预测系统函数
from ppstructure.predict_system import StructureSystem, save_structure_res, to_excel

# 定义导出的模块列表
__all__ = [
    'PaddleOCR', 'PPStructure', 'draw_ocr', 'draw_structure_result',
    'save_structure_res', 'download_with_progressbar', 'to_excel'
]

# 支持的检测模型列表
SUPPORT_DET_MODEL = ['DB']
# 版本号
VERSION = '2.6.1.3'
# 支持的识别模型列表
SUPPORT_REC_MODEL = ['CRNN', 'SVTR_LCNet']
# 基本目录
BASE_DIR = os.path.expanduser("~/.paddleocr/")

# 默认 OCR 模型版本
DEFAULT_OCR_MODEL_VERSION = 'PP-OCRv3'
# 支持的 OCR 模型版本列表
SUPPORT_OCR_MODEL_VERSION = ['PP-OCR', 'PP-OCRv2', 'PP-OCRv3']
# 默认结构化模型版本
DEFAULT_STRUCTURE_MODEL_VERSION = 'PP-StructureV2'
# 支持的结构模型版本
SUPPORT_STRUCTURE_MODEL_VERSION = ['PP-Structure', 'PP-StructureV2']
# 模型的 URL 地址
MODEL_URLS = {
    # 结构模型
    'STRUCTURE': {
        # PP-Structure 版本
        'PP-Structure': {
            'table': {
                'en': {
                    'url': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
                }
            }
        },
        # PP-StructureV2 版本
        'PP-StructureV2': {
            'table': {
                'en': {
                    'url': 'https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
                },
                'ch': {
                    'url': 'https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict_ch.txt'
                }
            },
            'layout': {
                'en': {
                    'url': 'https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar',
                    'dict_path': 'ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt'
                },
                'ch': {
                    'url': 'https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar',
                    'dict_path': 'ppocr/utils/dict/layout_dict/layout_cdla_dict.txt'
                }
            }
        }
    }
}

# 解析命令行参数
def parse_args(mMain=True):
    import argparse
    # 初始化参数解析器
    parser = init_args()
    # 是否添加帮助信息
    parser.add_help = mMain
    # 添加语言参数，默认为中文
    parser.add_argument("--lang", type=str, default='ch')
    # 添加检测参数，默认为 True
    parser.add_argument("--det", type=str2bool, default=True)
    # 添加一个名为 "rec" 的命令行参数，类型为布尔型，默认值为 True
    parser.add_argument("--rec", type=str2bool, default=True)
    # 添加一个名为 "type" 的命令行参数，类型为字符串，默认值为 'ocr'
    parser.add_argument("--type", type=str, default='ocr')
    # 添加一个名为 "ocr_version" 的命令行参数，类型为字符串，可选值为 SUPPORT_OCR_MODEL_VERSION 中的值，默认值为 'PP-OCRv3'，并提供帮助信息
    parser.add_argument(
        "--ocr_version",
        type=str,
        choices=SUPPORT_OCR_MODEL_VERSION,
        default='PP-OCRv3',
        help='OCR Model version, the current model support list is as follows: '
        '1. PP-OCRv3 Support Chinese and English detection and recognition model, and direction classifier model'
        '2. PP-OCRv2 Support Chinese detection and recognition model. '
        '3. PP-OCR support Chinese detection, recognition and direction classifier and multilingual recognition model.'
    )
    # 添加一个名为 "structure_version" 的命令行参数，类型为字符串，可选值为 SUPPORT_STRUCTURE_MODEL_VERSION 中的值，默认值为 'PP-StructureV2'，并提供帮助信息
    parser.add_argument(
        "--structure_version",
        type=str,
        choices=SUPPORT_STRUCTURE_MODEL_VERSION,
        default='PP-StructureV2',
        help='Model version, the current model support list is as follows:'
        ' 1. PP-Structure Support en table structure model.'
        ' 2. PP-StructureV2 Support ch and en table structure model.')

    # 遍历所有命令行参数
    for action in parser._actions:
        # 如果命令行参数的目标在指定的列表中
        if action.dest in [
                'rec_char_dict_path', 'table_char_dict_path', 'layout_dict_path'
        ]:
            # 将命令行参数的默认值设为 None
            action.default = None
    # 如果是主程序入口
    if mMain:
        # 解析命令行参数并返回
        return parser.parse_args()
    else:
        # 创建一个空字典用于存储推理参数
        inference_args_dict = {}
        # 遍历所有命令行参数
        for action in parser._actions:
            # 将命令行参数的目标和默认值存入推理参数字典中
            inference_args_dict[action.dest] = action.default
        # 使用推理参数字典创建命名空间并返回
        return argparse.Namespace(**inference_args_dict)
# 解析语言类型，将语言类型映射为统一的分类
def parse_lang(lang):
    # 拉丁语系语言列表
    latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
    ]
    # 阿拉伯语系语言列表
    arabic_lang = ['ar', 'fa', 'ug', 'ur']
    # 西里尔语系语言列表
    cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
    ]
    # 梵文语系语言列表
    devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
        'sa', 'bgc'
    ]
    # 判断语言类型并映射为统一的分类
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"
    # 确保语言类型在模型 URL 中存在
    assert lang in MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION][
        'rec'], 'param lang must in {}, but got {}'.format(
            MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['rec'].keys(), lang)
    # 根据语言类型确定检测语言
    if lang == "ch":
        det_lang = "ch"
    elif lang == 'structure':
        det_lang = 'structure'
    elif lang in ["en", "latin"]:
        det_lang = "en"
    else:
        det_lang = "ml"
    return lang, det_lang


# 获取模型配置信息
def get_model_config(type, version, model_type, lang):
    # 如果是 OCR 类型，则使用默认的 OCR 模型版本
    if type == 'OCR':
        DEFAULT_MODEL_VERSION = DEFAULT_OCR_MODEL_VERSION
    # 如果是 STRUCTURE 类型，则使用默认的 STRUCTURE 模型版本
    elif type == 'STRUCTURE':
        DEFAULT_MODEL_VERSION = DEFAULT_STRUCTURE_MODEL_VERSION
    else:
        raise NotImplementedError
    # 获取模型 URL
    model_urls = MODEL_URLS[type]
    # 如果指定版本不在模型 URL 中，则使用默认的模型版本
    if version not in model_urls:
        version = DEFAULT_MODEL_VERSION
    # 如果指定的模型类型不在给定版本的模型 URL 中
    if model_type not in model_urls[version]:
        # 如果指定的模型类型在默认版本的模型 URL 中
        if model_type in model_urls[DEFAULT_MODEL_VERSION]:
            # 将版本设置为默认版本
            version = DEFAULT_MODEL_VERSION
        else:
            # 记录错误日志并退出程序
            logger.error('{} models is not support, we only support {}'.format(
                model_type, model_urls[DEFAULT_MODEL_VERSION].keys()))
            sys.exit(-1)

    # 如果指定的语言不在给定版本和模型类型的模型 URL 中
    if lang not in model_urls[version][model_type]:
        # 如果指定的语言在默认版本和模型类型的模型 URL 中
        if lang in model_urls[DEFAULT_MODEL_VERSION][model_type]:
            # 将版本设置为默认版本
            version = DEFAULT_MODEL_VERSION
        else:
            # 记录错误日志并退出程序
            logger.error(
                'lang {} is not support, we only support {} for {} models'.
                format(lang, model_urls[DEFAULT_MODEL_VERSION][model_type].keys(
                ), model_type))
            sys.exit(-1)
    # 返回指定版本、模型类型和语言的模型 URL
    return model_urls[version][model_type][lang]
# 将字节内容解码成 NumPy 数组
def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


# 检查图像类型并进行相应处理
def check_img(img):
    # 如果图像是字节类型，则进行解码
    if isinstance(img, bytes):
        img = img_decode(img)
    # 如果图像是字符串类型
    if isinstance(img, str):
        # 如果是网络链接，则下载图像到本地
        if is_link(img):
            download_with_progressbar(img, 'tmp.jpg')
            img = 'tmp.jpg'
        image_file = img
        # 检查并读取图像文件
        img, flag_gif, flag_pdf = check_and_read(image_file)
        # 如果不是 GIF 或 PDF 格式
        if not flag_gif and not flag_pdf:
            with open(image_file, 'rb') as f:
                img_str = f.read()
                img = img_decode(img_str)
            # 如果解码失败
            if img is None:
                try:
                    # 尝试以不同方式加载图像
                    buf = BytesIO()
                    image = BytesIO(img_str)
                    im = Image.open(image)
                    rgb = im.convert('RGB')
                    rgb.save(buf, 'jpeg')
                    buf.seek(0)
                    image_bytes = buf.read()
                    data_base64 = str(base64.b64encode(image_bytes), encoding="utf-8")
                    image_decode = base64.b64decode(data_base64)
                    img_array = np.frombuffer(image_decode, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except:
                    logger.error("error in loading image:{}".format(image_file))
                    return None
        # 如果加载图像失败
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    # 如果图像是 NumPy 数组且为灰度图像，则转换成 BGR 格式
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


# PaddleOCR 类的子类，用于文本识别
class PaddleOCR(predict_system.TextSystem):
    # 结构系统的子类，用于处理结构化数据
    class PPStructure(StructureSystem):
        # 调用方法，处理图像并返回识别结果
        def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
            img = check_img(img)
            res, _ = super().__call__(
                img, return_ocr_result_in_table, img_idx=img_idx)
            return res


# 主函数
def main():
    # 解析命令行参数，设置为主程序
    args = parse_args(mMain=True)
    # 获取图像目录
    image_dir = args.image_dir
    # 如果图像目录是一个链接
    if is_link(image_dir):
        # 通过进度条下载图像到临时文件
        download_with_progressbar(image_dir, 'tmp.jpg')
        # 设置图像文件列表为临时文件
        image_file_list = ['tmp.jpg']
    else:
        # 获取图像目录下的图像文件列表
        image_file_list = get_image_file_list(args.image_dir)
    # 如果图像文件列表为空
    if len(image_file_list) == 0:
        # 记录错误日志，指出在图像目录中找不到图像
        logger.error('no images find in {}'.format(args.image_dir))
        # 返回
        return
    # 如果命令行参数中的类型是 OCR
    if args.type == 'ocr':
        # 使用命令行参数创建 PaddleOCR 引擎
        engine = PaddleOCR(**(args.__dict__))
    # 如果命令行参数中的类型是 structure
    elif args.type == 'structure':
        # 使用命令行参数创建 PPStructure 引擎
        engine = PPStructure(**(args.__dict__))
    # 如果命令行参数中的类型不是以上两种类型
    else:
        # 抛出未实现的错误
        raise NotImplementedError
```