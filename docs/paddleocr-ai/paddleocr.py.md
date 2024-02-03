# `.\PaddleOCR\paddleocr.py`

```py
# 版权声明和许可证信息
# 该代码受 Apache 许可证版本 2.0 保护
# 可以在遵守许可证的情况下使用该文件
# 可以在以下链接获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的
# 没有任何明示或暗示的担保或条件，查看特定语言的许可证以获取权限和限制

# 导入所需的库
import os
import sys
import importlib

# 获取当前文件所在目录
__dir__ = os.path.dirname(__file__)

# 导入 PaddlePaddle 库
import paddle

# 将当前目录添加到系统路径中
sys.path.append(os.path.join(__dir__, ''))

# 导入 OpenCV、日志、NumPy、Pathlib、Base64、BytesIO、PIL 库
import cv2
import logging
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

# 定义函数用于从文件导入模块
def _import_file(module_name, file_path, make_importable=False):
    # 创建模块规范
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    # 根据规范创建模块
    module = importlib.util.module_from_spec(spec)
    # 执行模块
    spec.loader.exec_module(module)
    # 如果需要使模块可导入，则将其添加到系统模块中
    if make_importable:
        sys.modules[module_name] = module
    return module

# 导入自定义工具模块
tools = _import_file(
    'tools', os.path.join(__dir__, 'tools/__init__.py'), make_importable=True)
# 导入 OCR 模块
ppocr = importlib.import_module('ppocr', 'paddleocr')
# 导入结构化识别模块
ppstructure = importlib.import_module('ppstructure', 'paddleocr')
# 导入日志记录器
from ppocr.utils.logging import get_logger
# 导入推理函数
from tools.infer import predict_system
# 导入工具函数
from ppocr.utils.utility import check_and_read, get_image_file_list, alpha_to_color, binarize_img
# 导入网络相关函数
from ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
# 导入推理工具函数
from tools.infer.utility import draw_ocr, str2bool, check_gpu
# 导入结构化识别工具函数
from ppstructure.utility import init_args, draw_structure_result
# 导入结构化识别系统、保存结果、转为 Excel 函数
from ppstructure.predict_system import StructureSystem, save_structure_res, to_excel

# 获取日志记录器
logger = get_logger()
# 导出所有公共接口
__all__ = [
    'PaddleOCR', 'PPStructure', 'draw_ocr', 'draw_structure_result',
    'save_structure_res', 'download_with_progressbar', 'to_excel'

这是一个包含多个字符串的列表，可能是一组函数或模块的名称。
# 支持的检测模型列表，目前只支持'DB'模型
SUPPORT_DET_MODEL = ['DB']
# 版本号为'2.7.0.3'
VERSION = '2.7.0.3'
# 支持的识别模型列表，目前支持'CRNN'和'SVTR_LCNet'模型
SUPPORT_REC_MODEL = ['CRNN', 'SVTR_LCNet']
# 基础目录为用户的.paddleocr目录
BASE_DIR = os.path.expanduser("~/.paddleocr/")
# 默认的OCR模型版本为'PP-OCRv4'
DEFAULT_OCR_MODEL_VERSION = 'PP-OCRv4'
# 支持的OCR模型版本列表包括'PP-OCR', 'PP-OCRv2', 'PP-OCRv3', 'PP-OCRv4'
SUPPORT_OCR_MODEL_VERSION = ['PP-OCR', 'PP-OCRv2', 'PP-OCRv3', 'PP-OCRv4']
# 默认的结构化模型版本为'PP-StructureV2'
DEFAULT_STRUCTURE_MODEL_VERSION = 'PP-StructureV2'
# 支持的结构化模型版本列表包括'PP-Structure', 'PP-StructureV2'
SUPPORT_STRUCTURE_MODEL_VERSION = ['PP-Structure', 'PP-StructureV2']
# 模型下载链接字典为空
MODEL_URLS = {
    },
    # 定义一个包含不同模型和语言的结构信息的字典
    'STRUCTURE': {
        # PP-Structure 模型
        'PP-Structure': {
            # table 类型
            'table': {
                # 英文语言
                'en': {
                    # 英文表格模型的下载链接
                    'url': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar',
                    # 表格结构字典的路径
                    'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
                }
            }
        },
        # PP-StructureV2 模型
        'PP-StructureV2': {
            # table 类型
            'table': {
                # 英文语言
                'en': {
                    # 英文表格模型的下载链接
                    'url': 'https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar',
                    # 表格结构字典的路径
                    'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
                },
                # 中文语言
                'ch': {
                    # 中文表格模型的下载链接
                    'url': 'https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar',
                    # 表格结构字典的路径
                    'dict_path': 'ppocr/utils/dict/table_structure_dict_ch.txt'
                }
            },
            # layout 类型
            'layout': {
                # 英文语言
                'en': {
                    # 英文布局模型的下载链接
                    'url': 'https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar',
                    # 布局字典的路径
                    'dict_path': 'ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt'
                },
                # 中文语言
                'ch': {
                    # 中文布局模型的下载链接
                    'url': 'https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar',
                    # 布局字典的路径
                    'dict_path': 'ppocr/utils/dict/layout_dict/layout_cdla_dict.txt'
                }
            }
        }
    }
# 解析命令行参数，返回命令行参数的命名空间
def parse_args(mMain=True):
    # 导入 argparse 模块
    import argparse
    # 初始化参数解析器
    parser = init_args()
    # 设置是否添加帮助信息
    parser.add_help = mMain
    # 添加命令行参数 "--lang"，类型为字符串，默认值为 'ch'
    parser.add_argument("--lang", type=str, default='ch')
    # 添加命令行参数 "--det"，类型为 str2bool 函数，默认值为 True
    parser.add_argument("--det", type=str2bool, default=True)
    # 添加命令行参数 "--rec"，类型为 str2bool 函数，默认值为 True
    parser.add_argument("--rec", type=str2bool, default=True)
    # 添加命令行参数 "--type"，类型为字符串，默认值为 'ocr'
    parser.add_argument("--type", type=str, default='ocr')
    # 添加命令行参数 "--ocr_version"，类型为字符串，可选值为 SUPPORT_OCR_MODEL_VERSION 列表中的值，默认值为 'PP-OCRv4'
    parser.add_argument(
        "--ocr_version",
        type=str,
        choices=SUPPORT_OCR_MODEL_VERSION,
        default='PP-OCRv4',
        help='OCR Model version, the current model support list is as follows: '
        '1. PP-OCRv4/v3 Support Chinese and English detection and recognition model, and direction classifier model'
        '2. PP-OCRv2 Support Chinese detection and recognition model. '
        '3. PP-OCR support Chinese detection, recognition and direction classifier and multilingual recognition model.'
    )
    # 添加命令行参数 "--structure_version"，类型为字符串，可选值为 SUPPORT_STRUCTURE_MODEL_VERSION 列表中的值，默认值为 'PP-StructureV2'
    parser.add_argument(
        "--structure_version",
        type=str,
        choices=SUPPORT_STRUCTURE_MODEL_VERSION,
        default='PP-StructureV2',
        help='Model version, the current model support list is as follows:'
        ' 1. PP-Structure Support en table structure model.'
        ' 2. PP-StructureV2 Support ch and en table structure model.')

    # 遍历参数解析器中的所有动作
    for action in parser._actions:
        # 如果动作的目标在指定的列表中
        if action.dest in [
                'rec_char_dict_path', 'table_char_dict_path', 'layout_dict_path'
        ]:
            # 将动作的默认值设为 None
            action.default = None
    # 如果是主程序调用，则解析命令行参数并返回
    if mMain:
        return parser.parse_args()
    # 如果不是主程序调用，则返回推断参数字典
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)


# 解析语言参数
def parse_lang(lang):
    # 拉丁语系的语言列表
    latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
    ]
    # 阿拉伯语系的语言列表
    arabic_lang = ['ar', 'fa', 'ug', 'ur']
    # 西里尔语系的语言列表
    cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
    ]
    # 梵文语系的语言列表
    devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
        'sa', 'bgc'
    ]
    # 根据语言类型进行判断，将语言类型转换为通用的语言类型
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"
    # 断言语言类型在模型 URL 中的识别模型中存在
    assert lang in MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION][
        'rec'], 'param lang must in {}, but got {}'.format(
            MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['rec'].keys(), lang)
    # 根据语言类型进一步判断检测语言类型
    if lang == "ch":
        det_lang = "ch"
    elif lang == 'structure':
        det_lang = 'structure'
    elif lang in ["en", "latin"]:
        det_lang = "en"
    else:
        det_lang = "ml"
    # 返回语言类型和检测语言类型
    return lang, det_lang
# 获取模型配置信息
def get_model_config(type, version, model_type, lang):
    # 根据不同的模型类型设置默认模型版本号
    if type == 'OCR':
        DEFAULT_MODEL_VERSION = DEFAULT_OCR_MODEL_VERSION
    elif type == 'STRUCTURE':
        DEFAULT_MODEL_VERSION = DEFAULT_STRUCTURE_MODEL_VERSION
    else:
        raise NotImplementedError

    # 获取指定模型类型的模型下载链接
    model_urls = MODEL_URLS[type]
    # 如果指定版本不在模型下载链接中，则使用默认版本号
    if version not in model_urls:
        version = DEFAULT_MODEL_VERSION
    # 如果指定模型类型不在指定版本的模型下载链接中，则检查是否在默认版本中
    if model_type not in model_urls[version]:
        if model_type in model_urls[DEFAULT_MODEL_VERSION]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error('{} models is not support, we only support {}'.format(
                model_type, model_urls[DEFAULT_MODEL_VERSION].keys()))
            sys.exit(-1)

    # 如果指定语言不在指定版本和模型类型的模型下载链接中，则检查是否在默认版本中
    if lang not in model_urls[version][model_type]:
        if lang in model_urls[DEFAULT_MODEL_VERSION][model_type]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                'lang {} is not support, we only support {} for {} models'.
                format(lang, model_urls[DEFAULT_MODEL_VERSION][model_type].keys(
                ), model_type))
            sys.exit(-1)
    # 返回指定版本、模型类型和语言的模型下载链接
    return model_urls[version][model_type][lang]


# 将图片内容解码为图像
def img_decode(content: bytes):
    # 将字节内容转换为 numpy 数组
    np_arr = np.frombuffer(content, dtype=np.uint8)
    # 使用 OpenCV 解码图像
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


# 检查图像是否为字节类型，如果是则解码为图像
def check_img(img):
    if isinstance(img, bytes):
        img = img_decode(img)
    # 如果输入的图片是字符串类型
    if isinstance(img, str):
        # 如果是网络链接，则下载图片到本地临时文件
        if is_link(img):
            download_with_progressbar(img, 'tmp.jpg')
            img = 'tmp.jpg'
        # 将图片文件路径保存
        image_file = img
        # 检查并读取图片文件，同时获取是否为gif和pdf的标志
        img, flag_gif, flag_pdf = check_and_read(image_file)
        # 如果不是gif和pdf格式
        if not flag_gif and not flag_pdf:
            # 以二进制只读方式打开图片文件
            with open(image_file, 'rb') as f:
                # 读取图片文件内容
                img_str = f.read()
                # 解码图片内容
                img = img_decode(img_str)
            # 如果解码失败
            if img is None:
                try:
                    # 创建一个字节流缓冲区
                    buf = BytesIO()
                    # 将图片内容转换为字节流
                    image = BytesIO(img_str)
                    # 打开图片
                    im = Image.open(image)
                    # 转换为RGB格式
                    rgb = im.convert('RGB')
                    # 将RGB格式保存到缓冲区
                    rgb.save(buf, 'jpeg')
                    buf.seek(0)
                    # 读取缓冲区内容
                    image_bytes = buf.read()
                    # 将图片内容进行base64编码
                    data_base64 = str(base64.b64encode(image_bytes), encoding="utf-8")
                    # 将base64编码后的内容解码
                    image_decode = base64.b64decode(data_base64)
                    # 将解码后的内容转换为numpy数组
                    img_array = np.frombuffer(image_decode, np.uint8)
                    # 使用OpenCV解码图片数组
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except:
                    # 如果出现异常，记录错误信息
                    logger.error("error in loading image:{}".format(image_file))
                    return None
        # 如果图片为空
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    # 如果图片是numpy数组且为灰度图像
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        # 将灰度图像转换为BGR格式
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 返回处理后的图片
    return img
# 定义一个类 PaddleOCR，继承自 predict_system.TextSystem
class PaddleOCR(predict_system.TextSystem):

# 定义一个类 PPStructure，继承自 StructureSystem
class PPStructure(StructureSystem):
    
    # 定义 __call__ 方法，接受图片、是否返回表格形式的 OCR 结果和图片索引作为参数
    def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
        # 检查并处理输入的图片
        img = check_img(img)
        # 调用父类的 __call__ 方法，传入图片、是否返回表格形式的 OCR 结果和图片索引
        res, _ = super().__call__(
            img, return_ocr_result_in_table, img_idx=img_idx)
        # 返回结果
        return res

# 定义主函数 main
def main():
    # 解析命令行参数
    args = parse_args(mMain=True)
    # 获取图片目录
    image_dir = args.image_dir
    # 如果图片目录是一个链接
    if is_link(image_dir):
        # 从链接下载图片到临时文件 tmp.jpg
        download_with_progressbar(image_dir, 'tmp.jpg')
        # 将图片文件列表设置为临时文件 tmp.jpg
        image_file_list = ['tmp.jpg']
    else:
        # 获取图片目录下的所有图片文件列表
        image_file_list = get_image_file_list(args.image_dir)
    # 如果图片文件列表为空
    if len(image_file_list) == 0:
        # 记录错误日志，指示在 args.image_dir 中未找到任何图片
        logger.error('no images find in {}'.format(args.image_dir))
        return
    # 如果命令行参数中的类型为 'ocr'
    if args.type == 'ocr':
        # 创建 PaddleOCR 引擎，传入命令行参数的字典
        engine = PaddleOCR(**(args.__dict__))
    # 如果命令行参数中的类型为 'structure'
    elif args.type == 'structure':
        # 创建 PPStructure 引擎，传入命令行参数的字典
        engine = PPStructure(**(args.__dict__))
    else:
        # 抛出未实现的错误
        raise NotImplementedError
```