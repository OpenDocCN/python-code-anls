# `.\MinerU\magic_pdf\model\pdf_extract_kit.py`

```
# 导入 loguru 日志库
from loguru import logger
# 导入操作系统相关模块
import os
# 导入时间模块
import time

# 从自定义库导入常量
from magic_pdf.libs.Constants import *
# 从模型列表中导入原子模型
from magic_pdf.model.model_list import AtomicModel

# 设置环境变量以禁止 albumentations 检查更新
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  
try:
    # 尝试导入计算机视觉库 OpenCV
    import cv2
    # 尝试导入 YAML 解析库
    import yaml
    # 尝试导入命令行参数解析库
    import argparse
    # 尝试导入数值计算库 NumPy
    import numpy as np
    # 尝试导入 PyTorch 库
    import torch
    # 尝试导入 TorchText 库
    import torchtext

    # 如果 TorchText 版本大于等于 0.18.0，则禁用其弃用警告
    if torchtext.__version__ >= "0.18.0":
        torchtext.disable_torchtext_deprecation_warning()
    # 从 PIL 导入图像处理模块
    from PIL import Image
    # 从 torchvision 导入转换模块
    from torchvision import transforms
    # 从 PyTorch 导入数据集和数据加载器
    from torch.utils.data import Dataset, DataLoader
    # 从 ultralytics 导入 YOLO 模型
    from ultralytics import YOLO
    # 从 unimernet 导入配置模块
    from unimernet.common.config import Config
    # 导入 unimernet 任务模块
    import unimernet.tasks as tasks
    # 从 unimernet 导入处理器加载函数
    from unimernet.processors import load_processor

except ImportError as e:
    # 捕获导入错误并记录异常信息
    logger.exception(e)
    # 记录错误信息提示安装所需依赖
    logger.error(
        'Required dependency not installed, please install by \n'
        '"pip install magic-pdf[full] --extra-index-url https://myhloli.github.io/wheels/"')
    # 退出程序，返回状态码 1
    exit(1)

# 从模型中导入 Layoutlmv3 预测器
from magic_pdf.model.pek_sub_modules.layoutlmv3.model_init import Layoutlmv3_Predictor
# 从模型中导入后处理函数
from magic_pdf.model.pek_sub_modules.post_process import get_croped_image, latex_rm_whitespace
# 从模型中导入自定义的 PaddleOCR
from magic_pdf.model.pek_sub_modules.self_modify import ModifiedPaddleOCR
# 从结构模型中导入结构表模型
from magic_pdf.model.pek_sub_modules.structeqtable.StructTableModel import StructTableModel
# 从模型中导入 ppTableModel
from magic_pdf.model.ppTableModel import ppTableModel


# 初始化表格模型的函数
def table_model_init(table_model_type, model_path, max_time, _device_='cpu'):
    # 判断表格模型类型是否为 STRUCT_EQTABLE
    if table_model_type == STRUCT_EQTABLE:
        # 初始化结构表模型
        table_model = StructTableModel(model_path, max_time=max_time, device=_device_)
    else:
        # 配置字典，包括模型目录和设备
        config = {
            "model_dir": model_path,
            "device": _device_
        }
        # 初始化 ppTableModel
        table_model = ppTableModel(config)
    # 返回初始化的表格模型
    return table_model


# 初始化 MFD 模型的函数
def mfd_model_init(weight):
    # 创建 YOLO 模型实例
    mfd_model = YOLO(weight)
    # 返回初始化的 MFD 模型
    return mfd_model


# 初始化 MFR 模型的函数
def mfr_model_init(weight_dir, cfg_path, _device_='cpu'):
    # 创建命名空间存储配置路径和选项
    args = argparse.Namespace(cfg_path=cfg_path, options=None)
    # 加载配置文件
    cfg = Config(args)
    # 设置预训练模型的路径
    cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
    # 设置模型名称
    cfg.config.model.model_config.model_name = weight_dir
    # 设置标记器配置路径
    cfg.config.model.tokenizer_config.path = weight_dir
    # 设置任务
    task = tasks.setup_task(cfg)
    # 构建模型
    model = task.build_model(cfg)
    # 将模型移动到指定设备
    model = model.to(_device_)
    # 加载可视化处理器
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    # 创建数据转换流程
    mfr_transform = transforms.Compose([vis_processor, ])
    # 返回模型和转换流程
    return [model, mfr_transform]


# 初始化 Layout 模型的函数
def layout_model_init(weight, config_file, device):
    # 创建 Layoutlmv3 预测器实例
    model = Layoutlmv3_Predictor(weight, config_file, device)
    # 返回初始化的模型
    return model


# 初始化 OCR 模型的函数
def ocr_model_init(show_log: bool = False, det_db_box_thresh=0.3):
    # 创建 ModifiedPaddleOCR 实例
    model = ModifiedPaddleOCR(show_log=show_log, det_db_box_thresh=det_db_box_thresh)
    # 返回初始化的 OCR 模型
    return model


# 定义 MathDataset 类，继承自 Dataset
class MathDataset(Dataset):
    # 初始化函数，接收图像路径和变换
    def __init__(self, image_paths, transform=None):
        # 存储图像路径
        self.image_paths = image_paths
        # 存储变换
        self.transform = transform

    # 返回数据集的大小
    def __len__(self):
        return len(self.image_paths)
    # 根据索引获取图像
        def __getitem__(self, idx):
            # 如果不是 PIL 图像，则将其转换为 PIL 图像
            if isinstance(self.image_paths[idx], str):
                # 打开指定路径的图像文件
                raw_image = Image.open(self.image_paths[idx])
            else:
                # 如果已经是图像，则直接赋值
                raw_image = self.image_paths[idx]
            # 如果存在转换函数，则应用转换
            if self.transform:
                # 应用转换并返回处理后的图像
                image = self.transform(raw_image)
                return image
# 单例模式的原子模型类
class AtomModelSingleton:
    # 存储类的唯一实例
    _instance = None
    # 存储已初始化的模型
    _models = {}

    # 重写__new__方法以实现单例模式
    def __new__(cls, *args, **kwargs):
        # 检查实例是否已创建
        if cls._instance is None:
            # 创建唯一实例
            cls._instance = super().__new__(cls)
        # 返回唯一实例
        return cls._instance

    # 获取原子模型的方法，使用模型名进行查找
    def get_atom_model(self, atom_model_name: str, **kwargs):
        # 检查模型是否已初始化
        if atom_model_name not in self._models:
            # 初始化模型并存储到字典中
            self._models[atom_model_name] = atom_model_init(model_name=atom_model_name, **kwargs)
        # 返回已初始化的模型
        return self._models[atom_model_name]


# 原子模型初始化函数，根据模型名进行不同的初始化
def atom_model_init(model_name: str, **kwargs):
    # 如果模型名是布局模型，进行相应初始化
    if model_name == AtomicModel.Layout:
        atom_model = layout_model_init(
            kwargs.get("layout_weights"),  # 获取布局权重
            kwargs.get("layout_config_file"),  # 获取布局配置文件
            kwargs.get("device")  # 获取设备信息
        )
    # 如果模型名是MFD模型，进行相应初始化
    elif model_name == AtomicModel.MFD:
        atom_model = mfd_model_init(
            kwargs.get("mfd_weights")  # 获取MFD权重
        )
    # 如果模型名是MFR模型，进行相应初始化
    elif model_name == AtomicModel.MFR:
        atom_model = mfr_model_init(
            kwargs.get("mfr_weight_dir"),  # 获取MFR权重目录
            kwargs.get("mfr_cfg_path"),  # 获取MFR配置路径
            kwargs.get("device")  # 获取设备信息
        )
    # 如果模型名是OCR模型，进行相应初始化
    elif model_name == AtomicModel.OCR:
        atom_model = ocr_model_init(
            kwargs.get("ocr_show_log"),  # 获取是否显示OCR日志的参数
            kwargs.get("det_db_box_thresh")  # 获取检测阈值
        )
    # 如果模型名是表格模型，进行相应初始化
    elif model_name == AtomicModel.Table:
        atom_model = table_model_init(
            kwargs.get("table_model_type"),  # 获取表格模型类型
            kwargs.get("table_model_path"),  # 获取表格模型路径
            kwargs.get("table_max_time"),  # 获取表格最大时间
            kwargs.get("device")  # 获取设备信息
        )
    # 如果模型名不在允许的范围内，记录错误并退出
    else:
        logger.error("model name not allow")  # 记录错误信息
        exit(1)  # 退出程序

    # 返回初始化后的原子模型
    return atom_model


# 自定义PEK模型类
class CustomPEKModel:
```