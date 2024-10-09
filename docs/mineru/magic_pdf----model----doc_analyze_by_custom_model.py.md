# `.\MinerU\magic_pdf\model\doc_analyze_by_custom_model.py`

```
# 导入时间模块，用于处理时间相关操作
import time

# 导入fitz库，用于处理PDF文件
import fitz
# 导入numpy库，用于处理数组和数值计算
import numpy as np
# 导入loguru库的logger，用于日志记录
from loguru import logger

# 从配置读取器中导入函数，用于获取模型目录、设备及表格识别配置
from magic_pdf.libs.config_reader import get_local_models_dir, get_device, get_table_recog_config
# 从模型列表中导入MODEL，用于模型选择
from magic_pdf.model.model_list import MODEL
# 导入模型配置
import magic_pdf.model as model_config


# 比较两个字典的项是否相等
def dict_compare(d1, d2):
    # 返回两个字典的项是否完全相同
    return d1.items() == d2.items()


# 从字典列表中移除重复字典
def remove_duplicates_dicts(lst):
    # 创建一个空列表以存储唯一字典
    unique_dicts = []
    # 遍历字典列表
    for dict_item in lst:
        # 如果当前字典不在唯一字典列表中
        if not any(
                dict_compare(dict_item, existing_dict) for existing_dict in unique_dicts
        ):
            # 将当前字典添加到唯一字典列表中
            unique_dicts.append(dict_item)
    # 返回唯一字典列表
    return unique_dicts


# 从PDF字节流中加载图像
def load_images_from_pdf(pdf_bytes: bytes, dpi=200) -> list:
    # 尝试导入Pillow库
    try:
        from PIL import Image
    # 如果导入失败，记录错误并退出
    except ImportError:
        logger.error("Pillow not installed, please install by pip.")
        exit(1)

    # 创建一个空列表以存储图像
    images = []
    # 打开PDF字节流
    with fitz.open("pdf", pdf_bytes) as doc:
        # 遍历每一页
        for index in range(0, doc.page_count):
            page = doc[index]
            # 创建一个缩放矩阵
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            # 获取当前页面的像素图
            pm = page.get_pixmap(matrix=mat, alpha=False)

            # 如果缩放后的宽度或高度超过9000，则不再缩放
            if pm.width > 9000 or pm.height > 9000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            # 将像素数据转换为PIL图像
            img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
            # 将PIL图像转换为numpy数组
            img = np.array(img)
            # 创建图像字典，包含图像及其宽高信息
            img_dict = {"img": img, "width": pm.width, "height": pm.height}
            # 将图像字典添加到图像列表中
            images.append(img_dict)
    # 返回图像列表
    return images


# 单例模式类，用于管理模型实例
class ModelSingleton:
    # 存储单例实例
    _instance = None
    # 存储模型字典
    _models = {}

    # 定义单例的创建方法
    def __new__(cls, *args, **kwargs):
        # 如果实例尚不存在
        if cls._instance is None:
            # 创建新的实例
            cls._instance = super().__new__(cls)
        # 返回单例实例
        return cls._instance

    # 获取模型的方法
    def get_model(self, ocr: bool, show_log: bool):
        # 使用OCR和日志显示参数作为键
        key = (ocr, show_log)
        # 如果该键未在模型字典中
        if key not in self._models:
            # 初始化模型并存入字典
            self._models[key] = custom_model_init(ocr=ocr, show_log=show_log)
        # 返回模型
        return self._models[key]


# 自定义模型初始化函数
def custom_model_init(ocr: bool = False, show_log: bool = False):
    # 初始化模型为None
    model = None

    # 根据模型模式选择模型
    if model_config.__model_mode__ == "lite":
        # 记录警告信息，说明Lite模式的输出质量不可靠
        logger.warning("The Lite mode is provided for developers to conduct testing only, and the output quality is "
                       "not guaranteed to be reliable.")
        # 选择Paddle模型
        model = MODEL.Paddle
    elif model_config.__model_mode__ == "full":
        # 选择PEK模型
        model = MODEL.PEK
    # 检查配置是否允许使用内部模型
    if model_config.__use_inside_model__:
        # 记录模型初始化开始时间
        model_init_start = time.time()
        # 判断使用的模型类型是否为 Paddle
        if model == MODEL.Paddle:
            # 从指定模块导入自定义 Paddle 模型类
            from magic_pdf.model.pp_structure_v2 import CustomPaddleModel
            # 实例化自定义 Paddle 模型
            custom_model = CustomPaddleModel(ocr=ocr, show_log=show_log)
        # 判断使用的模型类型是否为 PEK
        elif model == MODEL.PEK:
            # 从指定模块导入自定义 PEK 模型类
            from magic_pdf.model.pdf_extract_kit import CustomPEKModel
            # 从配置文件中获取模型目录和设备信息
            local_models_dir = get_local_models_dir()
            device = get_device()
            # 获取表格识别的配置
            table_config = get_table_recog_config()
            # 准备模型初始化所需的输入参数
            model_input = {"ocr": ocr,
                           "show_log": show_log,
                           "models_dir": local_models_dir,
                           "device": device,
                           "table_config": table_config}
            # 实例化自定义 PEK 模型
            custom_model = CustomPEKModel(**model_input)
        # 如果模型类型不在允许的范围内，记录错误并退出
        else:
            logger.error("Not allow model_name!")
            exit(1)
        # 计算模型初始化所耗费的时间
        model_init_cost = time.time() - model_init_start
        # 记录模型初始化耗时的信息
        logger.info(f"model init cost: {model_init_cost}")
    # 如果不允许使用内部模型，记录错误并退出
    else:
        logger.error("use_inside_model is False, not allow to use inside model")
        exit(1)

    # 返回自定义模型实例
    return custom_model
# 分析 PDF 文档的字节内容，返回每页的分析结果
def doc_analyze(pdf_bytes: bytes, ocr: bool = False, show_log: bool = False,
                start_page_id=0, end_page_id=None):

    # 获取模型管理器的单例实例
    model_manager = ModelSingleton()
    # 根据 OCR 和日志选项获取自定义模型
    custom_model = model_manager.get_model(ocr, show_log)

    # 从 PDF 字节流中加载图像
    images = load_images_from_pdf(pdf_bytes)

    # 设置结束页码，默认为最后一页
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else len(images) - 1

    # 检查结束页码是否超出范围，如果超出则发出警告并调整为最后一页
    if end_page_id > len(images) - 1:
        logger.warning("end_page_id is out of range, use images length")
        end_page_id = len(images) - 1

    # 初始化用于存储每页结果的列表
    model_json = []
    # 记录文档分析开始时间
    doc_analyze_start = time.time()

    # 遍历所有图像及其索引
    for index, img_dict in enumerate(images):
        img = img_dict["img"]  # 获取当前图像
        page_width = img_dict["width"]  # 获取当前图像宽度
        page_height = img_dict["height"]  # 获取当前图像高度
        # 如果当前索引在开始和结束页码范围内，分析图像
        if start_page_id <= index <= end_page_id:
            result = custom_model(img)  # 使用自定义模型进行分析
        else:
            result = []  # 超出范围则返回空结果
        # 创建当前页的信息字典
        page_info = {"page_no": index, "height": page_height, "width": page_width}
        # 创建当前页的结果字典并添加到模型结果列表中
        page_dict = {"layout_dets": result, "page_info": page_info}
        model_json.append(page_dict)
    # 计算文档分析所花费的时间
    doc_analyze_cost = time.time() - doc_analyze_start
    # 记录文档分析所需时间的日志
    logger.info(f"doc analyze cost: {doc_analyze_cost}")

    # 返回包含每页分析结果的列表
    return model_json
```