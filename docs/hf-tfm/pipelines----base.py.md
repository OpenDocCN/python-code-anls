# `.\pipelines\base.py`

```
# 导入必要的标准库和第三方库
import collections  # 导入collections模块，用于处理集合类型数据
import csv  # 导入csv模块，用于读写CSV文件
import importlib  # 导入importlib模块，用于动态加载模块
import json  # 导入json模块，用于处理JSON格式数据
import os  # 导入os模块，提供了许多与操作系统交互的函数
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import sys  # 导入sys模块，提供了与Python解释器相关的函数和变量
import traceback  # 导入traceback模块，用于提取和格式化异常的回溯信息
import types  # 导入types模块，用于操作Python类型和对象
import warnings  # 导入warnings模块，用于管理警告信息
from abc import ABC, abstractmethod  # 导入ABC和abstractmethod类，用于定义抽象基类和抽象方法
from collections import UserDict  # 从collections模块导入UserDict类，用于创建自定义字典类型
from contextlib import contextmanager  # 导入contextmanager装饰器，用于创建上下文管理器
from os.path import abspath, exists  # 从os.path模块导入abspath和exists函数，用于处理文件路径
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的类和函数

from ..dynamic_module_utils import custom_object_save  # 导入自定义模块，用于定制对象的保存方式
from ..feature_extraction_utils import PreTrainedFeatureExtractor  # 导入特征提取工具类
from ..image_processing_utils import BaseImageProcessor  # 导入图像处理基类
from ..modelcard import ModelCard  # 导入模型卡片类
from ..models.auto.configuration_auto import AutoConfig  # 导入自动配置类
from ..tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器类
from ..utils import (  # 从utils模块导入多个功能函数和类
    ModelOutput,  # 模型输出类
    add_end_docstrings,  # 添加文档字符串的装饰器
    infer_framework,  # 推断框架类型的函数
    is_tf_available,  # 检查是否有TensorFlow环境的函数
    is_torch_available,  # 检查是否有PyTorch环境的函数
    is_torch_cuda_available,  # 检查是否有CUDA支持的函数
    is_torch_npu_available,  # 检查是否有NPU支持的函数
    is_torch_xpu_available,  # 检查是否有XPU支持的函数
    logging,  # 日志记录对象
)

# 定义一个泛型张量类型
GenericTensor = Union[List["GenericTensor"], "torch.Tensor", "tf.Tensor"]

# 如果有TensorFlow环境，导入必要的模块
if is_tf_available():
    import tensorflow as tf  # 导入TensorFlow库

    from ..models.auto.modeling_tf_auto import TFAutoModel  # 导入TensorFlow自动建模类

# 如果有PyTorch环境，导入必要的模块
if is_torch_available():
    import torch  # 导入PyTorch库
    from torch.utils.data import DataLoader, Dataset  # 导入PyTorch的数据加载器和数据集类

    from ..models.auto.modeling_auto import AutoModel  # 导入PyTorch自动建模类

    # 为了向后兼容重新导出
    from .pt_utils import KeyDataset  # 导入PyTorch工具类中的KeyDataset类
else:
    Dataset = None  # 如果没有PyTorch环境，将Dataset设为None
    KeyDataset = None  # 如果没有PyTorch环境，将KeyDataset设为None

# 如果支持类型检查，导入必要的模型类
if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel  # 导入TensorFlow预训练模型类
    from ..modeling_utils import PreTrainedModel  # 导入通用预训练模型类

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)


# 定义一个不进行数据整理的函数
def no_collate_fn(items):
    if len(items) != 1:
        raise ValueError("This collate_fn is meant to be used with batch_size=1")
    return items[0]


# 定义一个用于填充数据的函数
def _pad(items, key, padding_value, padding_side):
    batch_size = len(items)  # 获取批次大小
    # 检查第一个项目的键对应的值是否为 torch.Tensor 类型
    if isinstance(items[0][key], torch.Tensor):
        # 获取张量的形状信息
        shape = items[0][key].shape
        # 获取张量的维度数
        dim = len(shape)
        
        # 如果键是 "pixel_values" 或 "image"，通常表示图像数据，不需要填充
        if key in ["pixel_values", "image"]:
            # 返回所有项目中指定键的张量拼接结果，按第 0 维度拼接
            # B, C, H, W
            return torch.cat([item[key] for item in items], dim=0)
        # 如果维度为 4 且键是 "input_features"，通常表示批处理的梅尔频谱图
        elif dim == 4 and key == "input_features":
            # 返回所有项目中指定键的张量拼接结果，按第 0 维度拼接
            # 这里是批处理的梅尔频谱图
            return torch.cat([item[key] for item in items], dim=0)
        
        # 计算所有项目中指定键的张量的最大长度和最小长度
        max_length = max(item[key].shape[1] for item in items)
        min_length = min(item[key].shape[1] for item in items)
        # 获取数据类型
        dtype = items[0][key].dtype

        # 根据不同维度情况进行张量初始化和填充
        if dim == 2:
            # 如果最大长度等于最小长度，可以直接拼接
            if max_length == min_length:
                # 返回所有项目中指定键的张量拼接结果，按第 0 维度拼接
                # 对于 ImageGPT，可能不提供填充值，但是应该能够一致地填充，因为大小应该匹配
                return torch.cat([item[key] for item in items], dim=0)
            # 创建一个形状为 (batch_size, max_length) 的零张量，使用指定的数据类型和填充值
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        elif dim == 3:
            # 创建一个形状为 (batch_size, max_length, shape[-1]) 的零张量，使用指定的数据类型和填充值
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value
        elif dim == 4:
            # 创建一个形状为 (batch_size, max_length, shape[-2], shape[-1]) 的零张量，使用指定的数据类型和填充值
            tensor = torch.zeros((batch_size, max_length, shape[-2], shape[-1]), dtype=dtype) + padding_value

        # 遍历项目列表，根据填充方向和维度对张量进行填充操作
        for i, item in enumerate(items):
            if dim == 2:
                # 根据填充方向选择对张量的部分进行填充
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0])] = item[key][0].clone()
            elif dim == 3:
                # 根据填充方向选择对张量的部分进行填充
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :] = item[key][0].clone()
            elif dim == 4:
                # 根据填充方向选择对张量的部分进行填充
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :, :] = item[key][0].clone()

        # 返回填充后的张量
        return tensor
    else:
        # 如果第一个项目的键对应的值不是 torch.Tensor 类型，返回所有项目中指定键的列表
        return [item[key] for item in items]
# 根据给定的 tokenizer 和 feature_extractor 对输入进行填充处理的函数
def pad_collate_fn(tokenizer, feature_extractor):
    # tokenizer 的填充位置，默认为 None
    t_padding_side = None
    # feature_extractor 的填充位置，默认为 None
    f_padding_side = None
    
    # 如果既没有 tokenizer 也没有 feature_extractor，则无法进行批处理，抛出 ValueError 异常
    if tokenizer is None and feature_extractor is None:
        raise ValueError("Pipeline without tokenizer or feature_extractor cannot do batching")
    
    # 如果存在 tokenizer
    if tokenizer is not None:
        # 如果 tokenizer 没有 pad_token_id，则抛出 ValueError 异常
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with "
                "`pipe.tokenizer.pad_token_id = model.config.eos_token_id`."
            )
        else:
            # 设置 tokenizer 的填充值为 pad_token_id，并获取填充位置
            t_padding_value = tokenizer.pad_token_id
            t_padding_side = tokenizer.padding_side
    
    # 如果存在 feature_extractor
    if feature_extractor is not None:
        # 获取 feature_extractor 的填充值和填充位置
        f_padding_value = getattr(feature_extractor, "padding_value", None)
        f_padding_side = getattr(feature_extractor, "padding_side", None)
    
    # 如果 tokenizer 和 feature_extractor 的填充位置存在且不一致，则抛出 ValueError 异常
    if t_padding_side is not None and f_padding_side is not None and t_padding_side != f_padding_side:
        raise ValueError(
            f"The feature extractor, and tokenizer don't agree on padding side {t_padding_side} != {f_padding_side}"
        )
    
    # 默认的填充位置为右侧
    padding_side = "right"
    if t_padding_side is not None:
        padding_side = t_padding_side
    if f_padding_side is not None:
        padding_side = f_padding_side
    
    # 内部函数，用于对批量数据进行填充处理
    def inner(items):
        keys = set(items[0].keys())
        # 检查批量数据中的每个元素是否具有相同的键，否则抛出 ValueError 异常
        for item in items:
            if set(item.keys()) != keys:
                raise ValueError(
                    f"The elements of the batch contain different keys. Cannot batch them ({set(item.keys())} !="
                    f" {keys})"
                )
        
        # 初始化填充后的数据字典
        padded = {}
        # 遍历每个键，根据键的类型选择相应的填充值进行填充
        for key in keys:
            if key in {"input_ids"}:
                # 对于 input_ids 类型的键，根据存在的 tokenizer 和 feature_extractor 选择填充值
                if tokenizer is None and feature_extractor is not None:
                    _padding_value = f_padding_value
                else:
                    _padding_value = t_padding_value
            elif key in {"input_values", "pixel_values", "input_features"}:
                # 对于 input_values, pixel_values, input_features 类型的键，使用 feature_extractor 的填充值
                _padding_value = f_padding_value
            elif key in {"p_mask", "special_tokens_mask"}:
                # 对于 p_mask, special_tokens_mask 类型的键，使用填充值 1
                _padding_value = 1
            elif key in {"attention_mask", "token_type_ids"}:
                # 对于 attention_mask, token_type_ids 类型的键，使用填充值 0
                _padding_value = 0
            else:
                # 对于其他类型的键，默认使用填充值 0
                _padding_value = 0
            # 调用 _pad 函数进行填充，并将填充后的结果存入 padded 字典中
            padded[key] = _pad(items, key, _padding_value, padding_side)
        
        return padded
    
    return inner


def infer_framework_load_model(
    model,
    config: AutoConfig,
    model_classes: Optional[Dict[str, Tuple[type]]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
    **model_kwargs,
):
    """
    模型加载函数，根据不同的框架和任务加载模型

    Parameters:
    - model: 加载的模型实例
    - config: 自动配置对象
    - model_classes: 模型类别的字典，可选
    - task: 任务名称，可选
    - framework: 框架名称，可选
    - **model_kwargs: 其他模型相关的参数

    """
    # 检查是否安装了 TensorFlow 和 PyTorch，如果都没有安装则引发运行时错误
    if not is_tf_available() and not is_torch_available():
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )
    # 如果输入的 model 是一个字符串，说明需要从预定义的任务中加载模型
    if isinstance(model, str):
        # 在 model_kwargs 中添加 "_from_pipeline" 键，指定任务名称
        model_kwargs["_from_pipeline"] = task
        # 初始化一个空的类元组
        class_tuple = ()
        # 检查是否可以使用 Torch，并且 framework 可以是 "pt" 或者 None
        look_pt = is_torch_available() and framework in {"pt", None}
        # 检查是否可以使用 TensorFlow，并且 framework 可以是 "tf" 或者 None
        look_tf = is_tf_available() and framework in {"tf", None}

        # 如果提供了 model_classes
        if model_classes:
            # 如果可以使用 Torch，则将 Torch 模型类添加到 class_tuple 中
            if look_pt:
                class_tuple = class_tuple + model_classes.get("pt", (AutoModel,))
            # 如果可以使用 TensorFlow，则将 TensorFlow 模型类添加到 class_tuple 中
            if look_tf:
                class_tuple = class_tuple + model_classes.get("tf", (TFAutoModel,))

        # 如果提供了 config.architectures
        if config.architectures:
            # 初始化一个空列表用于存储模型类
            classes = []
            # 遍历 config.architectures 中的每个架构
            for architecture in config.architectures:
                # 动态导入 transformers 模块
                transformers_module = importlib.import_module("transformers")
                # 如果可以使用 Torch
                if look_pt:
                    # 尝试从 transformers 模块中获取对应的架构类
                    _class = getattr(transformers_module, architecture, None)
                    # 如果成功获取到类，则将其添加到 classes 列表中
                    if _class is not None:
                        classes.append(_class)
                # 如果可以使用 TensorFlow
                if look_tf:
                    # 尝试从 transformers 模块中获取对应的 TensorFlow 架构类
                    _class = getattr(transformers_module, f"TF{architecture}", None)
                    # 如果成功获取到类，则将其添加到 classes 列表中
                    if _class is not None:
                        classes.append(_class)
            # 将 classes 列表中的所有类添加到 class_tuple 中
            class_tuple = class_tuple + tuple(classes)

        # 如果没有找到合适的模型类，则抛出 ValueError 异常
        if len(class_tuple) == 0:
            raise ValueError(f"Pipeline cannot infer suitable model classes from {model}")

        # 初始化一个字典，用于存储所有的 traceback 信息
        all_traceback = {}
        # 遍历 class_tuple 中的每个模型类
        for model_class in class_tuple:
            # 复制 model_kwargs 到 kwargs
            kwargs = model_kwargs.copy()
            # 如果 framework 是 "pt" 并且 model 的结尾是 ".h5"
            if framework == "pt" and model.endswith(".h5"):
                # 在 kwargs 中添加 "from_tf" 键，指示从 TensorFlow 加载模型
                kwargs["from_tf"] = True
                # 输出警告信息，说明正在尝试使用 PyTorch 加载 TensorFlow 模型
                logger.warning(
                    "Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. "
                    "Trying to load the model with PyTorch."
                )
            # 如果 framework 是 "tf" 并且 model 的结尾是 ".bin"
            elif framework == "tf" and model.endswith(".bin"):
                # 在 kwargs 中添加 "from_pt" 键，指示从 PyTorch 加载模型
                kwargs["from_pt"] = True
                # 输出警告信息，说明正在尝试使用 TensorFlow 加载 PyTorch 模型
                logger.warning(
                    "Model might be a PyTorch model (ending with `.bin`) but PyTorch is not available. "
                    "Trying to load the model with Tensorflow."
                )

            try:
                # 使用 model_class.from_pretrained 方法加载指定的模型
                model = model_class.from_pretrained(model, **kwargs)
                # 如果模型有 eval 方法，则调用它将模型设置为评估模式
                if hasattr(model, "eval"):
                    model = model.eval()
                # 在第一次成功加载模型后停止加载尝试
                break
            # 捕获 OSError 或 ValueError 异常
            except (OSError, ValueError):
                # 将捕获到的异常信息存储到 all_traceback 字典中
                all_traceback[model_class.__name__] = traceback.format_exc()
                # 继续尝试加载下一个模型类

        # 如果 model 仍然是一个字符串，说明无法成功加载任何模型类
        if isinstance(model, str):
            # 初始化一个空字符串，用于存储所有的错误信息
            error = ""
            # 遍历 all_traceback 字典中的每个模型类及其对应的 traceback 信息
            for class_name, trace in all_traceback.items():
                # 拼接错误信息字符串
                error += f"while loading with {class_name}, an error is thrown:\n{trace}\n"
            # 抛出 ValueError 异常，说明无法加载指定的模型
            raise ValueError(
                f"Could not load model {model} with any of the following classes: {class_tuple}. See the original errors:\n\n{error}\n"
            )

    # 如果 framework 是 None，则根据模型的类推断 framework
    if framework is None:
        framework = infer_framework(model.__class__)
    # 返回推断得到的 framework 和加载成功的 model
    return framework, model
# 从给定的模型推断出使用的框架（TensorFlow 或 PyTorch），并返回框架和模型的元组。

def infer_framework_from_model(
    model,
    model_classes: Optional[Dict[str, Tuple[type]]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
    **model_kwargs,
):
    """
    从传入的 `model` 推断出要使用的框架（TensorFlow 或 PyTorch）。返回一个元组 (框架, 模型)。

    如果 `model` 已经被实例化，此函数将从模型类中推断框架。否则，如果 `model` 是一个检查点名称，
    此方法将尝试使用 `model_classes` 实例化它。由于我们不希望实例化模型两次，这个模型将被返回以供流水线使用。

    如果两种框架都安装并可用于 `model`，将选择 PyTorch。

    Args:
        model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
            要从中推断框架的模型。如果是 `str`，则是检查点名称。要推断框架的模型。
        model_classes (dictionary `str` to `type`, *optional*):
            框架到类的映射。
        task (`str`):
            定义将返回的流水线的任务。
        model_kwargs:
            传递给模型的 `from_pretrained(..., **model_kwargs)` 函数的额外关键字参数。

    Returns:
        `Tuple`: 框架和模型的元组。
    """
    if isinstance(model, str):
        # 如果 `model` 是字符串，则从预训练模型加载配置。
        config = AutoConfig.from_pretrained(model, _from_pipeline=task, **model_kwargs)
    else:
        # 否则，从已实例化的模型中获取配置。
        config = model.config
    return infer_framework_load_model(
        model, config, model_classes=model_classes, _from_pipeline=task, task=task, framework=framework, **model_kwargs
    )


def get_framework(model, revision: Optional[str] = None):
    """
    选择要使用的框架（TensorFlow 或 PyTorch）。

    Args:
        model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
            如果两种框架都安装，选择与传入的模型对应的框架（模型类或模型名称）。如果未提供特定模型，则默认使用 PyTorch。
    """
    # 发出警告，表示 `get_framework` 已弃用，将在 v5 中移除，建议使用 `infer_framework_from_model`。
    warnings.warn(
        "`get_framework` is deprecated and will be removed in v5, use `infer_framework_from_model` instead.",
        FutureWarning,
    )
    if not is_tf_available() and not is_torch_available():
        # 如果 TensorFlow 2.0 和 PyTorch 都未安装，则引发运行时错误。
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )
    # 检查 model 是否为字符串类型
    if isinstance(model, str):
        # 检查当前环境是否支持 PyTorch 但不支持 TensorFlow
        if is_torch_available() and not is_tf_available():
            # 使用 AutoModel 类从预训练模型中加载指定的模型
            model = AutoModel.from_pretrained(model, revision=revision)
        # 检查当前环境是否支持 TensorFlow 但不支持 PyTorch
        elif is_tf_available() and not is_torch_available():
            # 使用 TFAutoModel 类从预训练模型中加载指定的模型
            model = TFAutoModel.from_pretrained(model, revision=revision)
        else:
            # 尝试使用 AutoModel 类加载预训练模型，如果出现 OSError 则使用 TFAutoModel 类
            try:
                model = AutoModel.from_pretrained(model, revision=revision)
            except OSError:
                model = TFAutoModel.from_pretrained(model, revision=revision)

    # 推断模型所属的深度学习框架（例如 PyTorch、TensorFlow）
    framework = infer_framework(model.__class__)
    # 返回推断得到的框架名称
    return framework
# 定义函数，选择给定任务的默认模型和修订版本
def get_default_model_and_revision(
    targeted_task: Dict, framework: Optional[str], task_options: Optional[Any]
) -> Union[str, Tuple[str, str]]:
    """
    Select a default model to use for a given task. Defaults to pytorch if ambiguous.

    Args:
        targeted_task (`Dict` ):
           Dictionary representing the given task, that should contain default models

        framework (`str`, None)
           "pt", "tf" or None, representing a specific framework if it was specified, or None if we don't know yet.

        task_options (`Any`, None)
           Any further value required by the task to get fully specified, for instance (SRC, TGT) languages for
           translation task.

    Returns

        `str` The model string representing the default model for this pipeline
    """
    # 如果只有 torch 可用而没有 tensorflow，则设定 framework 为 "pt"
    if is_torch_available() and not is_tf_available():
        framework = "pt"
    # 如果只有 tensorflow 可用而没有 torch，则设定 framework 为 "tf"
    elif is_tf_available() and not is_torch_available():
        framework = "tf"

    # 获取目标任务的默认设置
    defaults = targeted_task["default"]
    # 如果有任务选项，则根据选项获取默认模型
    if task_options:
        # 如果任务选项不在默认设置中，则抛出异常
        if task_options not in defaults:
            raise ValueError(f"The task does not provide any default models for options {task_options}")
        # 获取特定任务选项下的默认模型
        default_models = defaults[task_options]["model"]
    # 如果没有任务选项，但默认设置中包含 "model" 键，则获取默认模型
    elif "model" in defaults:
        default_models = targeted_task["default"]["model"]
    else:
        # 如果出现这种情况，通常表示任务默认设置选择有问题
        # XXX 如果更多任务要参数化，需要更新这个错误消息为更通用的形式
        raise ValueError('The task defaults can\'t be correctly selected. You probably meant "translation_XX_to_YY"')

    # 如果 framework 为 None，则默认使用 "pt"
    if framework is None:
        framework = "pt"

    # 返回指定 framework 下的默认模型
    return default_models[framework]


class PipelineException(Exception):
    """
    Raised by a [`Pipeline`] when handling __call__.

    Args:
        task (`str`): The task of the pipeline.
        model (`str`): The model used by the pipeline.
        reason (`str`): The error message to display.
    """

    def __init__(self, task: str, model: str, reason: str):
        # 初始化异常，继承自 Exception 类
        super().__init__(reason)

        # 初始化异常的任务、模型和原因
        self.task = task
        self.model = model


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each [`~pipelines.Pipeline`].
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        # 抽象方法，用于处理管道每个实例的参数，需要在子类中实现
        raise NotImplementedError()


class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing. Supported data formats
    currently includes:

    - JSON
    - CSV
    - stdin/stdout (pipe)

    `PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets columns to
    pipelines keyword arguments through the `dataset_kwarg_1=dataset_column_1` format.
    """

    # 管道支持的数据格式的基类，包括读取和写入支持的数据格式
    # 当前支持的数据格式包括 JSON、CSV、标准输入/输出（管道）
    # `PipelineDataFormat` 还包括一些工具函数，用于处理多列数据，例如从数据集列映射到管道关键字参数的格式 `dataset_kwarg_1=dataset_column_1`
    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """



    # 支持的数据格式
    SUPPORTED_FORMATS = ["json", "csv", "pipe"]

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite: bool = False,
    ):
        # 输出路径
        self.output_path = output_path
        # 输入路径
        self.input_path = input_path
        # 要读取的列，如果没有指定则为空字符串列表
        self.column = column.split(",") if column is not None else [""]
        # 是否多列读取
        self.is_multi_columns = len(self.column) > 1

        # 如果是多列读取，则解析每列的键值对形式
        if self.is_multi_columns:
            self.column = [tuple(c.split("=")) if "=" in c else (c, c) for c in self.column]

        # 如果指定了输出路径且不允许覆盖写入，则检查输出路径是否已存在
        if output_path is not None and not overwrite:
            if exists(abspath(self.output_path)):
                raise OSError(f"{self.output_path} already exists on disk")

        # 如果指定了输入路径，则检查输入路径是否存在
        if input_path is not None:
            if not exists(abspath(self.input_path)):
                raise OSError(f"{self.input_path} doesnt exist on disk")

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: Union[dict, List[dict]]):
        """
        Save the provided data object with the representation for the current [`~pipelines.PipelineDataFormat`].

        Args:
            data (`dict` or list of `dict`): The data to store.
        """
        raise NotImplementedError()

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save the provided data object as a pickle-formatted binary data on the disk.

        Args:
            data (`dict` or list of `dict`): The data to store.

        Returns:
            `str`: Path where the data has been saved.
        """
        # 获取输出路径的文件名（去除扩展名）
        path, _ = os.path.splitext(self.output_path)
        # 构建保存为 pickle 格式的二进制文件路径
        binary_path = os.path.extsep.join((path, "pickle"))

        # 将数据以二进制形式写入到文件中
        with open(binary_path, "wb+") as f_output:
            pickle.dump(data, f_output)

        # 返回保存数据的文件路径
        return binary_path

    @staticmethod
    def from_str(
        format: str,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ) -> "PipelineDataFormat":
        """
        根据 `format` 参数创建相应的 [`~pipelines.PipelineDataFormat`] 子类实例。

        Args:
            format (`str`):
                所需流水线的格式。可接受的值为 `"json"`、`"csv"` 或 `"pipe"`。
            output_path (`str`, *optional*):
                输出数据保存的路径。
            input_path (`str`, *optional*):
                输入数据所在路径。
            column (`str`, *optional*):
                要读取的列。
            overwrite (`bool`, *optional*, 默认为 `False`):
                是否覆盖 `output_path`。

        Returns:
            [`~pipelines.PipelineDataFormat`]: 合适的数据格式对象。
        """
        if format == "json":
            return JsonPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "csv":
            return CsvPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "pipe":
            return PipedPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        else:
            raise KeyError(f"未知的数据格式 {format} (可用的格式为 json/csv/pipe)")
class CsvPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using CSV data format.

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        # 调用父类的构造函数
        super().__init__(output_path, input_path, column, overwrite=overwrite)

    def __iter__(self):
        # 打开输入文件并创建 CSV 字典读取器
        with open(self.input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.is_multi_columns:
                    yield {k: row[c] for k, c in self.column}  # 如果有多列数据，以字典形式返回
                else:
                    yield row[self.column[0]]  # 否则返回指定列

    def save(self, data: List[dict]):
        """
        Save the provided data object with the representation for the current [`~pipelines.PipelineDataFormat`].

        Args:
            data (`List[dict]`): The data to store.
        """
        with open(self.output_path, "w") as f:
            if len(data) > 0:
                writer = csv.DictWriter(f, list(data[0].keys()))
                writer.writeheader()  # 写入头部
                writer.writerows(data)  # 写入数据


class JsonPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using JSON file format.

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)  # 调用父类的构造函数

        with open(input_path, "r") as f:  
            self._entries = json.load(f)  # 读取 JSON 格式的数据并存储在 _entries 变量中

    def __iter__(self):
        for entry in self._entries:
            if self.is_multi_columns:
                yield {k: entry[c] for k, c in self.column}  # 如果有多列数据，以字典形式返回
            else:
                yield entry[self.column[0]]  # 否则返回指定列

    def save(self, data: dict):
        """
        Save the provided data object in a json file.

        Args:
            data (`dict`): The data to store.
        """
        with open(self.output_path, "w") as f:
            json.dump(data, f)  # 将数据存储为 JSON 格式


class PipedPipelineDataFormat(PipelineDataFormat):
    """
    Read data from piped input to the python process. For multi columns data, columns should separated by \t

    If columns are provided, then the output will be a dictionary with {column_x: value_x}
    """
    def __iter__(self):
        """
        Iterate over input lines from stdin.

        Yields:
            - If the line contains tabs (`\t`):
                - If `self.column` is defined, yield a dictionary mapping column names to line values.
                - Otherwise, yield a tuple of line values.
            - If no tabs are present, yield the entire line.
        """
        for line in sys.stdin:
            # Split for multi-columns
            if "\t" in line:
                line = line.split("\t")
                if self.column:
                    # Dictionary to map arguments
                    yield {kwargs: l for (kwargs, _), l in zip(self.column, line)}
                else:
                    yield tuple(line)

            # No dictionary to map arguments
            else:
                yield line

    def save(self, data: dict):
        """
        Print the provided data.

        Args:
            data (`dict`): The data to be printed.
        """
        print(data)

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save binary data to the specified output path.

        Args:
            data (Union[dict, List[dict]]): The binary data to be saved.

        Returns:
            str: The output path where the data was saved.
        
        Raises:
            KeyError: If `self.output_path` is `None`, indicating a missing output path.
        """
        if self.output_path is None:
            raise KeyError(
                "When using piped input on pipeline outputting large object requires an output file path. "
                "Please provide such output path through --output argument."
            )

        return super().save_binary(data)
class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility.
    """

    @abstractmethod
    def transform(self, X):
        # 抽象方法：子类需实现数据转换的逻辑
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        # 抽象方法：子类需实现预测的逻辑
        raise NotImplementedError()


def build_pipeline_init_args(
    has_tokenizer: bool = False,
    has_feature_extractor: bool = False,
    has_image_processor: bool = False,
    supports_binary_output: bool = True,
) -> str:
    docstring = r"""
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow."""
    if has_tokenizer:
        docstring += r"""
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`]."""
    if has_feature_extractor:
        docstring += r"""
        feature_extractor ([`SequenceFeatureExtractor`]):
            The feature extractor that will be used by the pipeline to encode data for the model. This object inherits from
            [`SequenceFeatureExtractor`]."""
    if has_image_processor:
        docstring += r"""
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`]."""
    docstring += r"""
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.
            
            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
    """
    # 如果支持二进制输出，则添加下面的描述
    if supports_binary_output:
        docstring += r"""
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text."""
    # 返回完整的文档字符串
    return docstring
# 使用指定的参数构建初始化 Pipeline 的参数字典
PIPELINE_INIT_ARGS = build_pipeline_init_args(
    has_tokenizer=True, has_feature_extractor=True, has_image_processor=True, supports_binary_output=True
)

# 如果当前环境支持 Torch，则导入相关的工具类和函数
if is_torch_available():
    from transformers.pipelines.pt_utils import (
        PipelineChunkIterator,  # 导入 PipelineChunkIterator 类，用于处理分块迭代
        PipelineDataset,         # 导入 PipelineDataset 类，用于管道的数据集操作
        PipelineIterator,        # 导入 PipelineIterator 类，用于管道的迭代操作
        PipelinePackIterator,    # 导入 PipelinePackIterator 类，用于管道的打包迭代操作
    )

# 根据指定的参数设置构建 Pipeline，并添加相应的文档字符串
@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True, has_feature_extractor=True, has_image_processor=True))
class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Some pipeline, like for instance [`FeatureExtractionPipeline`] (`'feature-extraction'`) output large tensor object
    as nested-lists. In order to avoid dumping such large structure as textual data we provide the `binary_output`
    constructor argument. If set to `True`, the output will be stored in the pickle format.
    """

    default_input_names = None

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        image_processor: Optional[BaseImageProcessor] = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "",
        args_parser: ArgumentHandler = None,
        device: Union[int, "torch.device"] = None,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        binary_output: bool = False,
        **kwargs,
    ):
        """
        Initialize a pipeline object.

        Parameters:
        - model (Union["PreTrainedModel", "TFPreTrainedModel"]): The pretrained model to use.
        - tokenizer (Optional[PreTrainedTokenizer]): Tokenizer used to preprocess the inputs.
        - feature_extractor (Optional[PreTrainedFeatureExtractor]): Feature extractor for inputs.
        - image_processor (Optional[BaseImageProcessor]): Image processor for image inputs.
        - modelcard (Optional[ModelCard]): ModelCard describing the model's attributes.
        - framework (Optional[str]): The framework where the model is implemented (e.g., 'pt' for PyTorch).
        - task (str): The task associated with the pipeline.
        - args_parser (ArgumentHandler): Custom argument handler for parsing pipeline arguments.
        - device (Union[int, "torch.device"]): Device where the model will be run (CPU/GPU).
        - torch_dtype (Optional[Union[str, "torch.dtype"]]): Data type used in PyTorch models.
        - binary_output (bool): Whether to output results in binary (pickle) format.
        - **kwargs: Additional keyword arguments passed to the pipeline.

        Notes:
        - This constructor initializes a pipeline object with the specified parameters.
        - It supports various preprocessing and postprocessing operations for different tasks.
        - The 'binary_output' flag controls whether outputs are stored in binary format.
        """
        super().__init__()
        # Initialize the pipeline object with the provided parameters
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_processor = image_processor
        self.modelcard = modelcard
        self.framework = framework
        self.task = task
        self.args_parser = args_parser
        self.device = device
        self.torch_dtype = torch_dtype
        self.binary_output = binary_output
        self.kwargs = kwargs
    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        """
        Save the pipeline's model and tokenizer.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
            safe_serialization (`str`):
                Whether to save the model using `safetensors` or the traditional way for PyTorch or Tensorflow.
        """
        # 检查保存路径是否为文件，若是则记录错误并返回
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        # 创建保存路径的目录，若目录已存在则不会重复创建
        os.makedirs(save_directory, exist_ok=True)

        # 如果对象具有 `_registered_impl` 属性，则保存自定义流水线信息和代码
        if hasattr(self, "_registered_impl"):
            # 复制已注册的流水线信息
            pipeline_info = self._registered_impl.copy()
            custom_pipelines = {}
            # 遍历流水线信息
            for task, info in pipeline_info.items():
                # 只保留与当前类相关的流水线信息
                if info["impl"] != self.__class__:
                    continue

                info = info.copy()
                module_name = info["impl"].__module__
                last_module = module_name.split(".")[-1]
                # 将类名转换为完整的模块.类名 形式
                info["impl"] = f"{last_module}.{info['impl'].__name__}"
                # 转换为元组，包含每个任务支持的类的名称
                info["pt"] = tuple(c.__name__ for c in info["pt"])
                info["tf"] = tuple(c.__name__ for c in info["tf"])

                custom_pipelines[task] = info
            # 将自定义流水线信息设置到模型配置中
            self.model.config.custom_pipelines = custom_pipelines
            # 保存流水线的自定义代码和对象
            custom_object_save(self, save_directory)

        # 调用模型的保存方法，将模型保存到指定路径
        self.model.save_pretrained(save_directory, safe_serialization=safe_serialization)

        # 如果存在 tokenizer，则也将其保存到指定路径
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

        # 如果存在特征提取器，则也将其保存到指定路径
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory)

        # 如果存在图像处理器，则也将其保存到指定路径
        if self.image_processor is not None:
            self.image_processor.save_pretrained(save_directory)

        # 如果存在模型卡片信息，则也将其保存到指定路径
        if self.modelcard is not None:
            self.modelcard.save_pretrained(save_directory)
    # 定义上下文管理器，允许在用户指定的设备上进行张量分配，与框架无关。
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

        Returns:
            Context manager

        Examples:

        ```python
        # Explicitly ask for tensor allocation on CUDA device :0
        pipe = pipeline(..., device=0)
        with pipe.device_placement():
            # Every framework specific tensor allocation will be done on the request device
            output = pipe(...)
        ```"""
        # 如果当前框架是 TensorFlow
        if self.framework == "tf":
            # 使用 tf.device 确定张量分配在指定的 CPU 或 GPU 设备上
            with tf.device("/CPU:0" if self.device == -1 else f"/device:GPU:{self.device}"):
                # 使用 yield 将控制权交给调用者
                yield
        else:
            # 如果当前框架不是 TensorFlow
            # 检查设备类型是否为 CUDA
            if self.device.type == "cuda":
                # 使用 torch.cuda.device 确定张量分配在指定的 CUDA 设备上
                with torch.cuda.device(self.device):
                    # 使用 yield 将控制权交给调用者
                    yield
            else:
                # 对于其他类型的设备，默认使用 yield 将控制权交给调用者
                yield

    # 确保 PyTorch 张量位于指定设备上
    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be `torch.Tensor`, the rest is ignored):
                The tensors to place on `self.device`.
            Recursive on lists **only**.

        Return:
            `Dict[str, torch.Tensor]`: The same as `inputs` but on the proper device.
        """
        # 调用内部方法 _ensure_tensor_on_device，将输入张量放置在指定的设备上
        return self._ensure_tensor_on_device(inputs, self.device)

    # 内部方法，递归确保张量位于指定设备上
    def _ensure_tensor_on_device(self, inputs, device):
        # 如果输入是 ModelOutput 类型的对象
        if isinstance(inputs, ModelOutput):
            # 递归处理每个项，并确保它们位于指定设备上
            return ModelOutput(
                {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
            )
        # 如果输入是字典类型
        elif isinstance(inputs, dict):
            # 递归处理每个键值对，并确保值位于指定设备上
            return {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
        # 如果输入是 UserDict 类型
        elif isinstance(inputs, UserDict):
            # 递归处理每个键值对，并确保值位于指定设备上
            return UserDict({name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()})
        # 如果输入是列表类型
        elif isinstance(inputs, list):
            # 递归处理列表中的每个元素，并确保它们位于指定设备上
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        # 如果输入是元组类型
        elif isinstance(inputs, tuple):
            # 递归处理元组中的每个元素，并确保它们位于指定设备上
            return tuple([self._ensure_tensor_on_device(item, device) for item in inputs])
        # 如果输入是 PyTorch 的张量类型
        elif isinstance(inputs, torch.Tensor):
            # 如果目标设备是 CPU，并且张量的数据类型是 float16 或 bfloat16，则将其转换为 float 类型
            if device == torch.device("cpu") and inputs.dtype in {torch.float16, torch.bfloat16}:
                inputs = inputs.float()
            # 将张量移到指定设备上，并返回结果
            return inputs.to(device)
        else:
            # 对于其他类型的输入，直接返回原始输入
            return inputs
    def check_model_type(self, supported_models: Union[List[str], dict]):
        """
        检查模型类是否被流水线支持。

        Args:
            supported_models (`List[str]` or `dict`):
                支持的模型列表或包含模型类值的字典。
        """
        if not isinstance(supported_models, list):  # 如果不是列表，则从模型映射创建
            supported_models_names = []
            for _, model_name in supported_models.items():
                # 映射现在可以包含相同配置的模型元组。
                if isinstance(model_name, tuple):
                    supported_models_names.extend(list(model_name))
                else:
                    supported_models_names.append(model_name)
            if hasattr(supported_models, "_model_mapping"):
                for _, model in supported_models._model_mapping._extra_content.items():
                    if isinstance(model_name, tuple):
                        supported_models_names.extend([m.__name__ for m in model])
                    else:
                        supported_models_names.append(model.__name__)
            supported_models = supported_models_names
        if self.model.__class__.__name__ not in supported_models:
            logger.error(
                f"The model '{self.model.__class__.__name__}' is not supported for {self.task}. Supported models are"
                f" {supported_models}."
            )

    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters):
        """
        _sanitize_parameters 将会被调用，传入来自 `__init__` 或 `__call__` 方法的任意多余的命名参数。
        它应该返回三个字典，这些字典包含各种 `preprocess`、`forward` 和 `postprocess` 方法使用的解析参数。
        如果调用者未指定 kwargs，则不要填充字典。这使您可以在函数签名中保留默认值，这更加自然。

        它不应该直接调用，将会在 `__init__` 和 `__call__` 中自动调用，并由这些方法解析最终参数。
        """
        raise NotImplementedError("_sanitize_parameters not implemented")

    @abstractmethod
    def preprocess(self, input_: Any, **preprocess_parameters: Dict) -> Dict[str, GenericTensor]:
        """
        preprocess 将接受流水线特定的 `input_` 并返回一个包含一切 `_forward` 正常运行所需的字典。
        它应该至少包含一个张量，但也可能包含任意其他项目。
        """
        raise NotImplementedError("preprocess not implemented")
    # 定义一个抽象方法，用于模型的前向传播，接收经过 `preprocess` 处理后的输入数据字典，并返回模型输出
    def _forward(self, input_tensors: Dict[str, GenericTensor], **forward_parameters: Dict) -> ModelOutput:
        """
        _forward will receive the prepared dictionary from `preprocess` and run it on the model. This method might
        involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
        and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.

        It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
        code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
        of the code (leading to faster inference).
        """
        raise NotImplementedError("_forward not implemented")

    # 定义一个抽象方法，用于对 `_forward` 方法的输出进行后处理，将模型的原始输出转换成更友好的格式
    @abstractmethod
    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: Dict) -> Any:
        """
        Postprocess will receive the raw outputs of the `_forward` method, generally tensors, and reformat them into
        something more friendly. Generally it will output a list or a dict or results (containing just strings and
        numbers).
        """
        raise NotImplementedError("postprocess not implemented")

    # 返回一个上下文管理器，用于在推理过程中关闭梯度计算，提高推理效率
    def get_inference_context(self):
        return torch.no_grad

    # 定义模型的前向传播方法 `forward`，根据框架类型选择不同的处理逻辑，确保模型和张量在相同的设备上，并禁用训练部分的代码以加快推理速度
    def forward(self, model_inputs, **forward_params):
        with self.device_placement():
            if self.framework == "tf":
                model_inputs["training"] = False
                model_outputs = self._forward(model_inputs, **forward_params)
            elif self.framework == "pt":
                inference_context = self.get_inference_context()
                with inference_context():
                    model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                    model_outputs = self._forward(model_inputs, **forward_params)
                    model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
            else:
                raise ValueError(f"Framework {self.framework} is not supported")
        return model_outputs

    # 获取数据迭代器，用于模型推理过程中的数据加载和预处理
    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        ):
            # 检查输入是否可迭代
            if isinstance(inputs, collections.abc.Sized):
                # 如果可迭代，创建 PipelineDataset 对象
                dataset = PipelineDataset(inputs, self.preprocess, preprocess_params)
            else:
                # 如果不可迭代，并且设置了多个工作进程
                if num_workers > 1:
                    # 发出警告，因为在可迭代数据集中使用多个工作进程可能会导致错误
                    logger.warning(
                        "For iterable dataset using num_workers>1 is likely to result"
                        " in errors since everything is iterable, setting `num_workers=1`"
                        " to guarantee correctness."
                    )
                    # 设置 num_workers 为 1 以确保正确性
                    num_workers = 1
                # 创建 PipelineIterator 对象
                dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
            # 如果环境变量中未设置 TOKENIZERS_PARALLELISM
            if "TOKENIZERS_PARALLELISM" not in os.environ:
                # 输出日志，禁用 tokenizer 的并行处理，因为 DataLoader 已经在多线程处理了
                logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
                # 设置环境变量 TOKENIZERS_PARALLELISM 为 false
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
            # TODO hack by collating feature_extractor and image_processor
            # 根据情况选择特征提取器或图像处理器作为 feature_extractor
            feature_extractor = self.feature_extractor if self.feature_extractor is not None else self.image_processor
            # 根据 batch_size 是否为 1 选择使用 no_collate_fn 或 pad_collate_fn 作为 collate_fn
            collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, feature_extractor)
            # 创建 DataLoader 对象
            dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
            # 使用 dataloader 创建 PipelineIterator 对象
            model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
            # 使用 model_iterator 创建最终 PipelineIterator 对象
            final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
            # 返回最终的迭代器
            return final_iterator
    # 定义一个特殊方法 __call__，使对象可以像函数一样调用，接收输入参数 inputs 和额外的可变参数 args
    def __call__(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):
        # 如果有额外的位置参数 args，则记录警告并忽略这些参数
        if args:
            logger.warning(f"Ignoring args : {args}")

        # 确定并设置 num_workers 参数：如果未提供，则使用对象属性 _num_workers 或默认值 0
        if num_workers is None:
            if self._num_workers is None:
                num_workers = 0
            else:
                num_workers = self._num_workers

        # 确定并设置 batch_size 参数：如果未提供，则使用对象属性 _batch_size 或默认值 1
        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size

        # 根据传入的关键字参数 kwargs，从中提取预处理、前向传播和后处理的参数
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)

        # 将 __init__ 方法的参数与当前 __call__ 方法的参数合并，不影响 __init__ 方法的参数
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        # 增加调用计数器
        self.call_count += 1
        # 如果调用次数超过 10 次，并且使用的框架是 "pt"，并且设备是 CUDA GPU，则发出警告
        if self.call_count > 10 and self.framework == "pt" and self.device.type == "cuda":
            logger.warning_once(
                "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a"
                " dataset",
                UserWarning,
            )

        # 判断输入是否为 Dataset 类型并且 Dataset 类型存在，或者是生成器类型，或者是列表类型
        is_dataset = Dataset is not None and isinstance(inputs, Dataset)
        is_generator = isinstance(inputs, types.GeneratorType)
        is_list = isinstance(inputs, list)

        # 判断输入是否为可迭代对象，包括 Dataset、生成器和列表
        is_iterable = is_dataset or is_generator or is_list

        # 判断是否可以使用迭代器处理输入数据：当前框架为 "pt" 并且输入是 Dataset、生成器或列表
        can_use_iterator = self.framework == "pt" and (is_dataset or is_generator or is_list)

        # 如果输入是列表类型，并且可以使用迭代器处理，则获取迭代器并返回迭代器的结果列表
        if is_list:
            if can_use_iterator:
                final_iterator = self.get_iterator(
                    inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params
                )
                outputs = list(final_iterator)
                return outputs
            else:
                # 否则，使用 run_multi 方法处理列表输入并返回结果
                return self.run_multi(inputs, preprocess_params, forward_params, postprocess_params)
        # 如果可以使用迭代器处理输入，则直接返回迭代器对象
        elif can_use_iterator:
            return self.get_iterator(
                inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params
            )
        # 如果输入是可迭代对象，则使用 iterate 方法处理输入并返回结果
        elif is_iterable:
            return self.iterate(inputs, preprocess_params, forward_params, postprocess_params)
        # 如果框架为 "pt" 并且当前对象是 ChunkPipeline 类的实例，则处理单个输入并返回结果
        elif self.framework == "pt" and isinstance(self, ChunkPipeline):
            return next(
                iter(
                    self.get_iterator(
                        [inputs], num_workers, batch_size, preprocess_params, forward_params, postprocess_params
                    )
                )
            )
        # 否则，使用 run_single 方法处理单个输入并返回结果
        else:
            return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)
    # 使用给定的输入数据列表并行运行模型，对每个输入调用 `run_single` 方法
    def run_multi(self, inputs, preprocess_params, forward_params, postprocess_params):
        # 返回一个列表，包含每个输入数据经过模型处理后的结果
        return [self.run_single(item, preprocess_params, forward_params, postprocess_params) for item in inputs]

    # 对单个输入数据进行预处理、前向推理和后处理，返回处理后的结果
    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        # 对输入数据进行预处理，得到模型所需的输入格式
        model_inputs = self.preprocess(inputs, **preprocess_params)
        # 对预处理后的输入进行模型推理，得到模型输出
        model_outputs = self.forward(model_inputs, **forward_params)
        # 对模型输出进行后处理，得到最终的处理结果
        outputs = self.postprocess(model_outputs, **postprocess_params)
        # 返回处理后的输出结果
        return outputs

    # 迭代给定的输入数据集，对每个输入数据进行模型处理，并通过生成器返回结果
    def iterate(self, inputs, preprocess_params, forward_params, postprocess_params):
        # 这个函数应该重新命名为 `get_iterator`，这是一个临时的简单解决方案。
        for input_ in inputs:
            # 对每个输入数据调用 `run_single` 方法，通过生成器 `yield` 返回处理结果
            yield self.run_single(input_, preprocess_params, forward_params, postprocess_params)
# 定义 ChunkPipeline 类，继承自 Pipeline 类
class ChunkPipeline(Pipeline):
    
    # 重写 run_single 方法，处理单个输入数据
    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        # 存储所有模型的输出结果
        all_outputs = []
        
        # 遍历预处理后的输入数据
        for model_inputs in self.preprocess(inputs, **preprocess_params):
            # 调用模型的前向推理方法，获取模型输出
            model_outputs = self.forward(model_inputs, **forward_params)
            # 将模型输出添加到结果列表中
            all_outputs.append(model_outputs)
        
        # 对所有模型的输出进行后处理，得到最终的输出结果
        outputs = self.postprocess(all_outputs, **postprocess_params)
        return outputs

    # 获取迭代器方法，用于生成数据迭代器
    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        # 如果环境变量中没有设置 TOKENIZERS_PARALLELISM，则设置为 false
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # 如果 num_workers 大于 1，发出警告信息并将其设置为 1，以确保正确性
        if num_workers > 1:
            logger.warning(
                "For ChunkPipeline using num_workers>0 is likely to result in errors since everything is iterable,"
                " setting `num_workers=1` to guarantee correctness."
            )
            num_workers = 1
        
        # 使用输入数据和预处理函数参数创建 PipelineChunkIterator 对象
        dataset = PipelineChunkIterator(inputs, self.preprocess, preprocess_params)
        
        # 根据 batch_size 的不同选择不同的数据整合函数
        feature_extractor = self.feature_extractor if self.feature_extractor is not None else self.image_processor
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, feature_extractor)
        
        # 使用 DataLoader 创建数据加载器对象
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        
        # 使用 PipelinePackIterator 封装数据加载器，创建模型迭代器
        model_iterator = PipelinePackIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        
        # 使用 PipelineIterator 封装模型迭代器，创建最终迭代器对象
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator


# 定义 PipelineRegistry 类，用于管理支持的任务和任务别名
class PipelineRegistry:
    
    # 初始化方法，接受支持的任务字典和任务别名字典作为参数
    def __init__(self, supported_tasks: Dict[str, Any], task_aliases: Dict[str, str]) -> None:
        self.supported_tasks = supported_tasks  # 存储支持的任务字典
        self.task_aliases = task_aliases  # 存储任务别名字典
    
    # 获取所有支持任务名称的方法
    def get_supported_tasks(self) -> List[str]:
        # 获取所有支持任务的名称列表，包括任务字典中的键和任务别名字典中的键
        supported_task = list(self.supported_tasks.keys()) + list(self.task_aliases.keys())
        supported_task.sort()  # 对任务名称列表进行排序
        return supported_task  # 返回排序后的任务名称列表
    # 检查给定的任务是否存在别名，若存在则替换为其真实任务名
    def check_task(self, task: str) -> Tuple[str, Dict, Any]:
        if task in self.task_aliases:
            task = self.task_aliases[task]

        # 检查任务是否在支持的任务列表中，若是则返回任务名、目标任务配置和空的参数
        if task in self.supported_tasks:
            targeted_task = self.supported_tasks[task]
            return task, targeted_task, None

        # 若任务以"translation"开头，进一步解析任务格式，并返回任务名、翻译任务配置以及相关参数
        if task.startswith("translation"):
            tokens = task.split("_")
            if len(tokens) == 4 and tokens[0] == "translation" and tokens[2] == "to":
                targeted_task = self.supported_tasks["translation"]
                task = "translation"
                return task, targeted_task, (tokens[1], tokens[3])
            # 抛出格式错误的异常信息，要求任务名称使用正确的格式'translation_XX_to_YY'
            raise KeyError(f"Invalid translation task {task}, use 'translation_XX_to_YY' format")

        # 抛出未知任务异常信息，显示当前可用任务列表及格式示例
        raise KeyError(
            f"Unknown task {task}, available tasks are {self.get_supported_tasks() + ['translation_XX_to_YY']}"
        )

    # 注册新的任务流水线，并配置相关的模型类、默认模型和类型信息
    def register_pipeline(
        self,
        task: str,
        pipeline_class: type,
        pt_model: Optional[Union[type, Tuple[type]]] = None,
        tf_model: Optional[Union[type, Tuple[type]]] = None,
        default: Optional[Dict] = None,
        type: Optional[str] = None,
    ) -> None:
        # 如果任务已存在于支持的任务列表中，发出警告并覆盖现有的任务流水线配置
        if task in self.supported_tasks:
            logger.warning(f"{task} is already registered. Overwriting pipeline for task {task}...")

        # 如果没有提供 PyTorch 模型，则设为空元组
        if pt_model is None:
            pt_model = ()
        elif not isinstance(pt_model, tuple):
            pt_model = (pt_model,)

        # 如果没有提供 TensorFlow 模型，则设为空元组
        if tf_model is None:
            tf_model = ()
        elif not isinstance(tf_model, tuple):
            tf_model = (tf_model,)

        # 构建任务实现的字典，包括实现类、PyTorch 和 TensorFlow 模型
        task_impl = {"impl": pipeline_class, "pt": pt_model, "tf": tf_model}

        # 如果提供了默认配置，则检查是否包含模型信息，如果只有 'pt' 或 'tf'，则封装为包含模型键的字典
        if default is not None:
            if "model" not in default and ("pt" in default or "tf" in default):
                default = {"model": default}
            task_impl["default"] = default

        # 如果提供了类型信息，则添加到任务实现字典中
        if type is not None:
            task_impl["type"] = type

        # 将任务实现字典注册到支持的任务列表中，并将其绑定到流水线类的注册实现字典中
        self.supported_tasks[task] = task_impl
        pipeline_class._registered_impl = {task: task_impl}

    # 返回当前支持的任务列表及其配置的字典表示形式
    def to_dict(self):
        return self.supported_tasks
```