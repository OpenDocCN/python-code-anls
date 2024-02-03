# `.\PaddleOCR\ppocr\modeling\architectures\__init__.py`

```py
# 版权声明，告知代码版权归属及使用许可
# 根据 Apache License, Version 2.0 许可协议使用此文件
# 获取许可协议的副本链接
# 根据适用法律或书面同意，分发的软件基于“原样”分发，没有任何明示或暗示的担保或条件
# 请查看许可协议以了解特定语言的权限和限制

# 导入所需的模块和类
import copy
import importlib

from paddle.jit import to_static
from paddle.static import InputSpec

# 导入自定义的模型类
from .base_model import BaseModel
from .distillation_model import DistillationModel

# 导出的模块和函数
__all__ = ["build_model", "apply_to_static"]

# 根据配置构建模型
def build_model(config):
    # 深拷贝配置，避免修改原始配置
    config = copy.deepcopy(config)
    # 如果配置中没有指定模型名称，则使用基础模型
    if not "name" in config:
        arch = BaseModel(config)
    else:
        # 获取模型名称并从当前模块中导入对应的模型类
        name = config.pop("name")
        mod = importlib.import_module(__name__)
        arch = getattr(mod, name)(config)
    return arch

# 将模型转换为静态图模式
def apply_to_static(model, config, logger):
    # 如果配置中未设置静态图模式，则直接返回模型
    if config["Global"].get("to_static", False) is not True:
        return model
    # 检查配置中是否包含静态训练所需的图像形状信息
    assert "d2s_train_image_shape" in config["Global"], "d2s_train_image_shape must be assigned for static training mode..."
    # 支持静态训练的模型列表
    supported_list = [
        "DB", "SVTR_LCNet", "TableMaster", "LayoutXLM", "SLANet", "SVTR"
    ]
    # 获取当前模型的算法类型
    if config["Architecture"]["algorithm"] in ["Distillation"]:
        algo = list(config["Architecture"]["Models"].values())[0]["algorithm"]
    else:
        algo = config["Architecture"]["algorithm"]
    # 确保当前算法在支持的静态训练模型列表中
    assert algo in supported_list, f"algorithms that supports static training must in in {supported_list} but got {algo}"

    # 定义输入规格，包括图像形状和数据类型
    specs = [
        InputSpec(
            [None] + config["Global"]["d2s_train_image_shape"], dtype='float32')
    ]
    # 如果算法是 SVTR_LCNet，则添加对应的输入规格到 specs 列表中
    if algo == "SVTR_LCNet":
        specs.append([
            # 输入规格：[None, 最大文本长度]，数据类型为 int64
            InputSpec([None, config["Global"]["max_text_length"]], dtype='int64'), 
            # 输入规格：[None, 最大文本长度]，数据类型为 int64
            InputSpec([None, config["Global"]["max_text_length"]], dtype='int64'),
            # 输入规格：[None]，数据类型为 int64
            InputSpec([None], dtype='int64'), 
            # 输入规格：[None]，数据类型为 float64
            InputSpec([None], dtype='float64')
        ])
    # 如果算法是 TableMaster，则添加对应的输入规格到 specs 列表中
    elif algo == "TableMaster":
        specs.append(
            [
                # 输入规格：[None, 最大文本长度]，数据类型为 int64
                InputSpec([None, config["Global"]["max_text_length"]], dtype='int64'),
                # 输入规格：[None, 最大文本长度, 4]，数据类型为 float32
                InputSpec([None, config["Global"]["max_text_length"], 4], dtype='float32'),
                # 输入规格：[None, 最大文本长度, 1]，数据类型为 float32
                InputSpec([None, config["Global"]["max_text_length"], 1], dtype='float32'),
                # 输入规格：[None, 6]，数据类型为 float32
                InputSpec([None, 6], dtype='float32'),
            ])
    # 如果算法是 LayoutXLM，则重新赋值 specs 列表
    elif algo == "LayoutXLM":
        specs = [[
            # 输入规格：[None, 512]，数据类型为 int64
            InputSpec(shape=[None, 512], dtype="int64"),  # input_ids
            # 输入规格：[None, 512, 4]，数据类型为 int64
            InputSpec(shape=[None, 512, 4], dtype="int64"),  # bbox
            # 输入规格：[None, 512]，数据类型为 int64
            InputSpec(shape=[None, 512], dtype="int64"),  # attention_mask
            # 输入规格：[None, 512]，数据类型为 int64
            InputSpec(shape=[None, 512], dtype="int64"),  # token_type_ids
            # 输入规格：[None, 3, 224, 224]，数据类型为 float32
            InputSpec(shape=[None, 3, 224, 224], dtype="float32"),  # image
            # 输入规格：[None, 512]，数据类型为 int64
            InputSpec(shape=[None, 512], dtype="int64"),  # label
        ]]
    # 如果算法是 SLANet，则添加对应的输入规格到 specs 列表中
    elif algo == "SLANet":
        specs.append([
            # 输入规格：[None, 最大文本长度 + 2]，数据类型为 int64
            InputSpec([None, config["Global"]["max_text_length"] + 2], dtype='int64'),
            # 输入规格：[None, 最大文本长度 + 2, 4]，数据类型为 float32
            InputSpec([None, config["Global"]["max_text_length"] + 2, 4], dtype='float32'),
            # 输入规格：[None, 最大文本长度 + 2, 1]，数据类型为 float32
            InputSpec([None, config["Global"]["max_text_length"] + 2, 1], dtype='float32'),
            # 输入规格：[None, 6]，数据类型为 float64
            InputSpec([None, 6], dtype='float64'),
        ])
    # 如果算法选择为"SVTR"，则执行以下代码块
    elif algo == "SVTR":
        # 将输入规范添加到specs列表中
        specs.append([
            InputSpec(
                [None, config["Global"]["max_text_length"]], dtype='int64'),
            InputSpec(
                [None], dtype='int64')
        ])
    # 将模型转换为静态图模型，使用specs作为输入规范
    model = to_static(model, input_spec=specs)
    # 记录日志，显示成功应用@to_static并使用的规范
    logger.info("Successfully to apply @to_static with specs: {}".format(specs))
    # 返回转换后的模型
    return model
```