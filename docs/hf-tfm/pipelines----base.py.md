# `.\transformers\pipelines\base.py`

```
# 设定文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权许可
# 除了遵守许可证，您不得使用此文件
# 您可以获得许可证的副本
# 在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证
# 除非适用法律要求或书面同意，否则软件分发基于“按原样”基础
# 没有任何形式的担保或条件，无论是明示的还是隐含的
# 有关特定语言的权限和限制，请参阅许可证

# 导入所需的模块
import collections
import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
# 导入ABC、abstractmethod类，以及UserDict类
from abc import ABC, abstractmethod
from collections import UserDict
# 导入contextmanager模块中的contextmanager装饰器
from contextlib import contextmanager
# 导入abspath、exists函数
from os.path import abspath, exists
# 导入TYPE_CHECKING，Any、Dict、List、Optional、Tuple、Union等类型
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
# 导入动态模块工具中的custom_object_save函数
from ..dynamic_module_utils import custom_object_save
# 导入特征提取工具中的PreTrainedFeatureExtractor类
from ..feature_extraction_utils import PreTrainedFeatureExtractor
# 导入图像处理工具中的BaseImageProcessor类
from ..image_processing_utils import BaseImageProcessor
# 导入模型卡片对象
from ..modelcard import ModelCard
# 导入自动配置模块中的AutoConfig类
from ..models.auto.configuration_auto import AutoConfig
# 导入标记化工具中的PreTrainedTokenizer类
from ..tokenization_utils import PreTrainedTokenizer
# 导入辅助工具
from ..utils import (
    ModelOutput,
    add_end_docstrings,
    infer_framework,
    is_tf_available,
    is_torch_available,
    is_torch_cuda_available,
    is_torch_xpu_available,
    logging,
)

# 定义GenericTensor类型的Union类型
GenericTensor = Union[List["GenericTensor"], "torch.Tensor", "tf.Tensor"]

# 如果 TensorFlow 可用
if is_tf_available():
    # 导入 TensorFlow 模块
    import tensorflow as tf
    # 导入自动模型配置模块中的TFAutoModel类
    from ..models.auto.modeling_tf_auto import TFAutoModel

# 如果 PyTorch 可用
if is_torch_available():
    # 导入 torch 模块
    import torch
    # 从 torch.utils.data 中导入 DataLoader、Dataset类
    from torch.utils.data import DataLoader, Dataset
    # 从自动模型配置模块中导入AutoModel类
    from ..models.auto.modeling_auto import AutoModel
    # 为向后兼容性重新导出
    from .pt_utils import KeyDataset
else:
    # 如果 PyTorch 不可用，则将 Dataset 和 KeyDataset 设置为 None
    Dataset = None
    KeyDataset = None

# 如果支持类型检查
if TYPE_CHECKING:
    # 从模型化工具中导入TFPreTrainedModel和PreTrainedModel类
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 no_collate_fn 函数
def no_collate_fn(items):
    # 如果 items 的长度不为 1，则引发 ValueError 异常
    if len(items) != 1:
        raise ValueError("This collate_fn is meant to be used with batch_size=1")
    # 返回 items 的第一个元素
    return items[0]

# 定义 _pad 函数
def _pad(items, key, padding_value, padding_side):
    # 获取 items 的长度，即 batch_size
    batch_size = len(items)
    # 检查第一个item的key对应的值是否为torch.Tensor类型
    if isinstance(items[0][key], torch.Tensor):
        # 获取张量的形状
        shape = items[0][key].shape
        # 获取张量的维度
        dim = len(shape)
        # 如果key是"pixel_values"或"image"，则假设为图片，不需要填充
        if key in ["pixel_values", "image"]:
            # 在指定维度上拼接张量列表
            # B, C, H, W
            return torch.cat([item[key] for item in items], dim=0)
        # 如果维度为4且key是"input_features"，则假设为mel频谱，进行拼接
        elif dim == 4 and key == "input_features":
            return torch.cat([item[key] for item in items], dim=0)
        # 获取各个item中key对应张量的最大长度和最小长度
        max_length = max(item[key].shape[1] for item in items)
        min_length = min(item[key].shape[1] for item in items)
        # 获取张量的数据类型
        dtype = items[0][key].dtype

        # 根据张量维度不同进行填充操作
        if dim == 2:
            if max_length == min_length:
                # 对于`ImageGPT`绕过填充，因为它不提供填充值，不过我们可以一致填充，因为大小应该匹配
                return torch.cat([item[key] for item in items], dim=0)
            # 创建全零张量，并加上填充值，形状为(batch_size, max_length)
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        elif dim == 3:
            # 创建全零张量，并加上填充值，形状为(batch_size, max_length, shape[-1])
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value
        elif dim == 4:
            # 创建全零张量，并加上填充值，形状为(batch_size, max_length, shape[-2], shape[-1])
            tensor = torch.zeros((batch_size, max_length, shape[-2], shape[-1]), dtype=dtype)

        # 对于每个item，根据维度和填充位置进行填充操作
        for i, item in enumerate(items):
            if dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0])] = item[key][0].clone()
            elif dim == 3:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :] = item[key][0].clone()
            elif dim == 4:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :, :] = item[key][0].clone()

        return tensor
    else:
        # 如果key对应的值不是torch.Tensor类型，则返回item中key对应的值的列表
        return [item[key] for item in items]
def pad_collate_fn(tokenizer, feature_extractor):
    # 定义一个函数，用于合并批次数据并进行填充
    # Tokenizer
    t_padding_side = None
    # 初始化 tokenizer 的填充位置
    # Feature extractor
    f_padding_side = None
    # 初始化 feature extractor 的填充位置
    if tokenizer is None and feature_extractor is None:
        # 如果 tokenizer 和 feature_extractor 都为空，则抛出数值错误
        raise ValueError("Pipeline without tokenizer or feature_extractor cannot do batching")
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            # 如果 tokenizer 没有填充令牌，则抛出数值错误
            raise ValueError("Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with "
                "`pipe.tokenizer.pad_token_id = model.config.eos_token_id`.")
        else:
            # 获取 tokenizer 的填充值和填充位置
            t_padding_value = tokenizer.pad_token_id
            t_padding_side = tokenizer.padding_side
    if feature_extractor is not None:
        # Feature extractor 可能为图像，不需要填充
        f_padding_value = getattr(feature_extractor, "padding_value", None)
        f_padding_side = getattr(feature_extractor, "padding_side", None)

    if t_padding_side is not None and f_padding_side is not None and t_padding_side != f_padding_side:
        # 如果 tokenizer 和 feature_extractor 的填充位置不一致，则抛出数值错误
        raise ValueError(
            f"The feature extractor, and tokenizer don't agree on padding side {t_padding_side} != {f_padding_side}"
        )
    padding_side = "right"
    if t_padding_side is not None:
        # 如果 tokenizer 有填充位置，则使用 tokenizer 的填充位置
        padding_side = t_padding_side
    if f_padding_side is not None:
        # 如果 feature extractor 有填充位置，则使用 feature extractor 的填充位置
        padding_side = f_padding_side

    def inner(items):
        # 定义内部函数 inner，用于合并 items 并进行填充
        keys = set(items[0].keys())
        for item in items:
            if set(item.keys()) != keys:
                # 如果 items 中的键不一致，则抛出数值错误
                raise ValueError(
                    f"The elements of the batch contain different keys. Cannot batch them ({set(item.keys())} !="
                    f" {keys})"
                )
        padded = {}
        for key in keys:
            # 遍历 items 的键
            if key in {"input_ids"}:
                # 如果键是 input_ids
                if tokenizer is None and feature_extractor is not None:
                    # 如果 tokenizer 为空且 feature extractor 不为空，则使用 feature extractor 的填充值
                    _padding_value = f_padding_value
                else:
                    # 否则使用 tokenizer 的填充值
                    _padding_value = t_padding_value
            elif key in {"input_values", "pixel_values", "input_features"}:
                # 如果键是 input_values、pixel_values 或 input_features，则使用 feature extractor 的填充值
                _padding_value = f_padding_value
            elif key in {"p_mask", "special_tokens_mask"}:
                # 如果键是 p_mask 或 special_tokens_mask，则填充值为 1
                _padding_value = 1
            elif key in {"attention_mask", "token_type_ids"}:
                # 如果键是 attention_mask 或 token_type_ids，则填充值为 0
                _padding_value = 0
            else:
                # 否则填充值为 0
                _padding_value = 0
            # 执行填充操作
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
    # 推断框架并加载模型
    # 该函数将接受一个模型和一些参数，并返回加载后的模型
    # 从传递的 `model` 中选择要使用的框架（TensorFlow 或 PyTorch），返回一个元组 (framework, model)。
    
    # 如果 `model` 已被实例化，这个函数将仅仅从模型类中推测出框架。否则 `model` 实际上是一个检查点名称，这个方法将尝试使用` model_classes` 实例化它。由于我们不想实例化模型两次，所以这个模型将被返回以供管道使用。
    
    # 如果两个框架都已安装并可用于 `model`，将选择 PyTorch。
    
    # 参数:
    # model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
    # 要从中推测框架的模型。如果是 `str`，是一个检查点名称。要从中推测出框架的模型。
    # config ([`AutoConfig`]):
    # 与模型关联的配置，以帮助使用正确的类
    # model_classes (dictionary `str` to `type`, *optional*):
    # 框架到类的映射。
    # task (`str`):
    # 定义将返回哪个管道的任务。
    # model_kwargs:
    # 传递给模型的 `from_pretrained(..., **model_kwargs)` 函数的额外关键字参数的字典。
    
    # 返回:
    # `Tuple`: 一个包含框架和模型的元组。
    """
    # 如果 TensorFlow 2.0 和 PyTorch 都没有安装和可用
    if not is_tf_available() and not is_torch_available():
        # 抛出运行时错误，至少应安装 TensorFlow 2.0 或 PyTorch
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )
    # 检查model是否为字符串类型
    if isinstance(model, str):
        # 检查参数中是否含有"_from_pipeline"，如果找到了就指定task的值
        model_kwargs["_from_pipeline"] = task
        # 定义空元组class_tuple
        class_tuple = ()
        # 检查是否有torch模块，并且framework是否为"pt"或者None
        look_pt = is_torch_available() and framework in {"pt", None}
        # 检查是否有tensorflow模块，并且framework是否为"tf"或者None
        look_tf = is_tf_available() and framework in {"tf", None}
        # 检查是否有model_classes，并根据look_pt和look_tf添加torch和tensorflow的类，以元组形式存储在class_tuple中
        if model_classes:
            if look_pt:
                class_tuple = class_tuple + model_classes.get("pt", (AutoModel,))
            if look_tf:
                class_tuple = class_tuple + model_classes.get("tf", (TFAutoModel,))

        # 如果配置文件中有architectures
        if config.architectures:
            # 定义空列表classes
            classes = []
            # 遍历architectures中的元素
            for architecture in config.architectures:
                # 导入transformers模块
                transformers_module = importlib.import_module("transformers")
                # 如果look_pt为True
                if look_pt:
                    # 根据architecture在transformers_module中获取类
                    _class = getattr(transformers_module, architecture, None)
                    # 如果_class不为空
                    if _class is not None:
                        # 将_class添加到classes列表中
                        classes.append(_class)
                # 如果look_tf为True
                if look_tf:
                    # 根据architecture在transformers_module中获取类名
                    _class = getattr(transformers_module, f"TF{architecture}", None)
                    # 如果_class不为空
                    if _class is not None:
                        # 将_class添加到classes列表中
                        classes.append(_class)
            # 将classes列表转换为元组，与class_tuple合并
            class_tuple = class_tuple + tuple(classes)

        # 如果class_tuple为空
        if len(class_tuple) == 0:
            # 抛出异常，提示无法根据给定的model找到合适的model类
            raise ValueError(f"Pipeline cannot infer suitable model classes from {model}")

        # 定义空字典all_traceback
        all_traceback = {}
        # 遍历class_tuple中的类
        for model_class in class_tuple:
            # 将model_kwargs进行复制
            kwargs = model_kwargs.copy()
            # 如果framework为"pt"且model字符串以".h5"结尾
            if framework == "pt" and model.endswith(".h5"):
                # 设置kwargs的from_tf为True
                kwargs["from_tf"] = True
                # 输出警告信息，提示model可能为一个tensorflow模型但tensorflow模块不可用，尝试使用pytorch加载model
                logger.warning(
                    "Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. "
                    "Trying to load the model with PyTorch."
                )
            # 如果framework为"tf"且model字符串以".bin"结尾
            elif framework == "tf" and model.endswith(".bin"):
                # 设置kwargs的from_pt为True
                kwargs["from_pt"] = True
                # 输出警告信息，提示model可能为一个pytorch模型但pytorch模块不可用，尝试使用tensorflow加载model
                logger.warning(
                    "Model might be a PyTorch model (ending with `.bin`) but PyTorch is not available. "
                    "Trying to load the model with Tensorflow."
                )

            try:
                # 根据model_class和kwargs加载model
                model = model_class.from_pretrained(model, **kwargs)
                # 如果model有eval方法，调用eval方法
                if hasattr(model, "eval"):
                    model = model.eval()
                # 在第一次成功加载后停止加载
                break
            # 捕获到的异常为OSError或ValueError
            except (OSError, ValueError):
                # 将当前model_class的名称和详细的异常信息添加到all_traceback字典中
                all_traceback[model_class.__name__] = traceback.format_exc()
                continue

        # 如果model仍然为字符串类型
        if isinstance(model, str):
            # 定义空字符串error
            error = ""
            # 遍历all_traceback中的元素
            for class_name, trace in all_traceback.items():
                # 将class_name和trace添加到error字符串中
                error += f"while loading with {class_name}, an error is thrown:\n{trace}\n"
            # 抛出异常，提示无法加载model，并给出详细的失败信息
            raise ValueError(
                f"Could not load model {model} with any of the following classes: {class_tuple}. See the original errors:\n\n{error}\n"
            )

    # 如果framework为None
    if framework is None:
        # 根据model类推断framework
        framework = infer_framework(model.__class__)
    # 返回framework和model
    return framework, model
# 从给定的模型推断框架（TensorFlow或PyTorch）的函数
def infer_framework_from_model(
    model,
    model_classes: Optional[Dict[str, Tuple[type]]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
    **model_kwargs,
):
    """
    根据传入的 `model` 选择要使用的框架（TensorFlow或PyTorch）。返回一个元组 (框架, 模型)。

    如果 `model` 已经实例化，此函数将仅从模型类中推断出框架。否则 `model` 实际上是一个检查点名称，
    此方法将尝试使用 `model_classes` 来实例化它。由于我们不想实例化模型两次，因此该模型将被返回以供管道使用。

    如果两种框架都已安装并可用于 `model`，则选择 PyTorch。

    Args:
        model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
            要从中推断框架的模型。如果是 `str`，表示检查点名称。要从中推断框架的模型。
        model_classes (dictionary `str` to `type`, *optional*):
            一个从框架到类的映射。
        task (`str`):
            定义将返回哪个管道的任务。
        model_kwargs:
            传递给模型的 `from_pretrained(..., **model_kwargs)` 函数的附加关键字参数的字典。

    Returns:
        `Tuple`: 一个元组 (框架, 模型)。
    """
    # 如果 `model` 是字符串，则从预训练配置中获取配置
    if isinstance(model, str):
        config = AutoConfig.from_pretrained(model, _from_pipeline=task, **model_kwargs)
    else:
        # 否则，获取模型的配置
        config = model.config
    # 调用 infer_framework_load_model 函数来进一步推断框架并加载模型
    return infer_framework_load_model(
        model, config, model_classes=model_classes, _from_pipeline=task, task=task, framework=framework, **model_kwargs
    )


def get_framework(model, revision: Optional[str] = None):
    """
    选择要使用的框架（TensorFlow或PyTorch）的函数。

    Args:
        model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
            如果两种框架都已安装，则选择与传入的模型对应的框架（模型类或模型名称）。如果未提供特定的模型，则默认使用 PyTorch。
    """
    # 引发未来警告，告知该函数将在 v5 版本中被移除，建议使用 `infer_framework_from_model` 替代
    warnings.warn(
        "`get_framework` is deprecated and will be removed in v5, use `infer_framework_from_model` instead.",
        FutureWarning,
    )
    # 如果既没有 TensorFlow 2.0 也没有 PyTorch 安装，则引发运行时错误
    if not is_tf_available() and not is_torch_available():
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )
    # 检查model是否为字符串类型
    if isinstance(model, str):
        # 如果torch可用而tf不可用，使用AutoModel类从预训练模型加载模型
        if is_torch_available() and not is_tf_available():
            model = AutoModel.from_pretrained(model, revision=revision)
        # 如果tf可用而torch不可用，使用TFAutoModel类从预训练模型加载模型
        elif is_tf_available() and not is_torch_available():
            model = TFAutoModel.from_pretrained(model, revision=revision)
        # 如果都不可用，尝试使用AutoModel类从预训练模型加载模型，如果出现OSError，则使用TFAutoModel类
        else:
            try:
                model = AutoModel.from_pretrained(model, revision=revision)
            except OSError:
                model = TFAutoModel.from_pretrained(model, revision=revision)

    # 推断模型类别
    framework = infer_framework(model.__class__)
    # 返回推断结果
    return framework
def get_default_model_and_revision(
    targeted_task: Dict, framework: Optional[str], task_options: Optional[Any]
) -> Union[str, Tuple[str, str]]:
    """
    Select a default model to use for a given task. Defaults to pytorch if ambiguous.

    Args:
        targeted_task (`Dict`):
           Dictionary representing the given task, that should contain default models

        framework (`str`, None):
           "pt", "tf" or None, representing a specific framework if it was specified, or None if we don't know yet.

        task_options (`Any`, None):
           Any further value required by the task to get fully specified, for instance (SRC, TGT) languages for
           translation task.

    Returns:
        `str` or `Tuple[str, str]`: The model string representing the default model for this pipeline
    """
    # Determine the framework to use if not explicitly specified
    if is_torch_available() and not is_tf_available():
        framework = "pt"
    elif is_tf_available() and not is_torch_available():
        framework = "tf"

    # Extract default models based on task and options
    defaults = targeted_task["default"]
    if task_options:
        if task_options not in defaults:
            raise ValueError(f"The task does not provide any default models for options {task_options}")
        default_models = defaults[task_options]["model"]
    elif "model" in defaults:
        default_models = targeted_task["default"]["model"]
    else:
        # Raise an error if default models cannot be determined
        raise ValueError('The task defaults can\'t be correctly selected. You probably meant "translation_XX_to_YY"')

    # Set the framework to PyTorch if not specified
    if framework is None:
        framework = "pt"

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
        # Initialize the exception with the given reason
        super().__init__(reason)

        # Store task, model, and reason for further reference
        self.task = task
        self.model = model


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each [`~pipelines.Pipeline`].
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        # Abstract method to be implemented by subclasses
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
```  
    # 初始化方法，设置输出路径、输入路径、读取的列以及是否覆盖输出路径
    Args:
        output_path (`str`): 要保存传出数据的路径。
        input_path (`str`): 要查找输入数据的路径。
        column (`str`): 要读取的列。
        overwrite (`bool`, *可选*, 默认为 `False`):
            是否覆盖 `output_path`。

    SUPPORTED_FORMATS = ["json", "csv", "pipe"]

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite: bool = False,
    ):
        # 设置输出路径
        self.output_path = output_path
        # 设置输入路径
        self.input_path = input_path
        # 如果 column 不为 None，则将其按逗号拆分为列表，否则设置为空列表
        self.column = column.split(",") if column is not None else [""]
        # 检查是否为多列
        self.is_multi_columns = len(self.column) > 1

        # 如果是多列，则将每列按等号拆分为元组，否则设置为（列名，列名）的元组
        if self.is_multi_columns:
            self.column = [tuple(c.split("=")) if "=" in c else (c, c) for c in self.column]

        # 如果输出路径不为 None 且不覆盖已存在的文件，则检查输出路径是否已存在
        if output_path is not None and not overwrite:
            if exists(abspath(self.output_path)):
                raise OSError(f"{self.output_path} already exists on disk")

        # 如果输入路径不为 None，则检查输入路径是否存在
        if input_path is not None:
            if not exists(abspath(self.input_path)):
                raise OSError(f"{self.input_path} doesnt exist on disk")

    @abstractmethod
    def __iter__(self):
        # 抽象方法，迭代器方法，子类必须实现
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: Union[dict, List[dict]]):
        """
        保存提供的数据对象的当前 [`~pipelines.PipelineDataFormat`] 表示形式。

        Args:
            data (`dict` or list of `dict`): 要存储的数据。
        """
        raise NotImplementedError()

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        将提供的数据对象保存为 pickle 格式的二进制数据到磁盘上。

        Args:
            data (`dict` or list of `dict`): 要存储的数据。

        Returns:
            `str`: 数据保存的路径。
        """
        # 将输出路径的扩展名改为 .pickle
        path, _ = os.path.splitext(self.output_path)
        binary_path = os.path.extsep.join((path, "pickle"))

        # 使用二进制写入模式打开文件，并将数据以 pickle 格式写入文件中
        with open(binary_path, "wb+") as f_output:
            pickle.dump(data, f_output)

        # 返回保存的二进制数据路径
        return binary_path

    @staticmethod
    def from_str(
        format: str,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    # 定义一个函数，根据所需的格式返回相应的 PipelineDataFormat 子类实例
    def create_data_format(format: str, output_path: str = None, input_path: str = None, column: str = None, overwrite: bool = False) -> "PipelineDataFormat":
        """
        创建一个合适的 [`~pipelines.PipelineDataFormat`] 的子类实例，具体取决于 `format` 参数。

        Args:
            format (`str`):
                所需管道的格式。可接受的值包括 `"json"`, `"csv"` 或 `"pipe"`。
            output_path (`str`, *optional*):
                保存输出数据的位置。
            input_path (`str`, *optional*):
                查找输入数据的位置。
            column (`str`, *optional*):
                要读取的列。
            overwrite (`bool`, *optional*, 默认为 `False`):
                是否覆盖 `output_path`。

        Returns:
            [`~pipelines.PipelineDataFormat`]: 合适的数据格式。
        """
        # 如果格式是 "json"，返回 JsonPipelineDataFormat 实例
        if format == "json":
            return JsonPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        # 如果格式是 "csv"，返回 CsvPipelineDataFormat 实例
        elif format == "csv":
            return CsvPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        # 如果格式是 "pipe"，返回 PipedPipelineDataFormat 实例
        elif format == "pipe":
            return PipedPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        # 如果格式不在可接受的范围内，触发 KeyError 异常
        else:
            raise KeyError(f"Unknown reader {format} (Available reader are json/csv/pipe)")
# 定义 CsvPipelineDataFormat 类，用于处理使用 CSV 数据格式的管道任务
class CsvPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using CSV data format.

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`): Whether or not to overwrite the `output_path`.
    """

    # 初始化方法，接受输出路径、输入路径、读取的列以及是否覆盖输出路径的参数
    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        # 调用父类的初始化方法
        super().__init__(output_path, input_path, column, overwrite=overwrite)

    # 迭代方法，读取输入路径的 CSV 文件并按照列提取数据返回
    def __iter__(self):
        with open(self.input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 如果是多列数据，则返回以列名对应值的字典
                if self.is_multi_columns:
                    yield {k: row[c] for k, c in self.column}
                else:
                    yield row[self.column[0]]

    # 保存方法，将提供的数据对象保存到指定的输出目录
    def save(self, data: List[dict]):
        """
        Save the provided data object with the representation for the current [`~pipelines.PipelineDataFormat`].

        Args:
            data (`List[dict]`): The data to store.
        """
        with open(self.output_path, "w") as f:
            if len(data) > 0:
                writer = csv.DictWriter(f, list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)


# 定义 JsonPipelineDataFormat 类，用于处理使用 JSON 数据格式的管道任务
class JsonPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using JSON file format.

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`): Whether or not to overwrite the `output_path`.
    """

    # 初始化方法，接受输出路径、输入路径、读取的列以及是否覆盖输出路径的参数
    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        # 调用父类的初始化方法
        super().__init__(output_path, input_path, column, overwrite=overwrite)
        
        # 打开输入路径的 JSON 文件，并将其内容加载到 _entries 属性中
        with open(input_path, "r") as f:
            self._entries = json.load(f)

    # 迭代方法，按照列提取 JSON 数据并返回
    def __iter__(self):
        for entry in self._entries:
            # 如果是多列数据，则返回以列名对应值的字典
            if self.is_multi_columns:
                yield {k: entry[c] for k, c in self.column}
            else:
                yield entry[self.column[0]]

    # 保存方法，将提供的数据对象保存为一个 JSON 文件
    def save(self, data: dict):
        """
        Save the provided data object in a json file.

        Args:
            data (`dict`): The data to store.
        """
        with open(self.output_path, "w") as f:
            json.dump(data, f)


# 定义 PipedPipelineDataFormat 类，用于从管道输入读取数据到 Python 进程
    """
    Read data from piped input to the python process. For multi columns data, columns should separated by \t

    If columns are provided, then the output will be a dictionary with {column_x: value_x}
    Args:
        output_path (`str`): Where to save the outgoing data.  # 定义函数参数，用于指定输出数据的保存路径
        input_path (`str`): Where to look for the input data.  # 定义函数参数，用于指定输入数据的查找路径
        column (`str`): The column to read.  # 定义函数参数，用于指定要读取的列
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.  # 定义函数参数，用于指定是否覆盖输出路径，默认为 False

    def __iter__(self):
        for line in sys.stdin:  # 遍历标准输入的每一行数据
            # Split for multi-columns  # 如果每行数据中包含制表符，则进行分割
            if "\t" in line:  
                line = line.split("\t")  # 使用制表符进行分割
                if self.column:  # 如果指定了要读取的列
                    # Dictionary to map arguments  # 创建用于映射参数的字典
                    yield {kwargs: l for (kwargs, _), l in zip(self.column, line)}  # 生成用于映射参数的字典
                else:
                    yield tuple(line)  # 生成元组形式的数据列表

            # No dictionary to map arguments  # 如果数据中不包含制表符
            else:
                yield line  # 直接生成每行数据

    def save(self, data: dict):
        """
        Print the data.

        Args:
            data (`dict`): The data to store.  # 函数参数，用于指定要存储的数据
        """
        print(data)  # 打印数据

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        if self.output_path is None:
            raise KeyError(
                "When using piped input on pipeline outputting large object requires an output file path. "
                "Please provide such output path through --output argument."
            )  # 如果输出路径为None，则抛出KeyError异常

        return super().save_binary(data)  # 返回调用父类方法的结果
# 定义一个名为 _ScikitCompat 的类，用作 Scikit 和 Keras 兼容性的接口层
class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility.
    """

    # 声明一个抽象方法 transform，子类需要实现该方法
    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

    # 声明一个抽象方法 predict，子类需要实现该方法
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

# 定义初始化 Pipeline 的参数
PIPELINE_INIT_ARGS = r"""
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
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
            the associated CUDA device id. You can pass native `torch.device` or a `str` too.
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
"""

# 如果 torch 可用，则导入相关的模块
if is_torch_available():
    from transformers.pipelines.pt_utils import (
        PipelineChunkIterator,
        PipelineDataset,
        PipelineIterator,
        PipelinePackIterator,
    )

# 为 Pipeline 类添加文档字符串，并引用初始化参数
@add_end_docstrings(PIPELINE_INIT_ARGS)
class Pipeline(_ScikitCompat):
    """
    # Pipeline 类是所有管道类的基类。参考此类获取跨不同管道共享的方法。
    # 实现管道操作的基类。管道工作流程被定义为以下操作序列:
    #     输入 -> 分词 -> 模型推理 -> 后处理(任务相关) -> 输出
    # 管道通过 device 参数支持在 CPU 或 GPU 上运行。
    # 某些管道(如 FeatureExtractionPipeline('feature-extraction'))输出大型张量对象作为嵌套列表。
    # 为了避免将如此大的结构转储为文本数据,我们提供了 binary_output 构造函数参数。
    # 如果设置为 True,输出将以 pickle 格式存储。
    class Pipeline:
        default_input_names = None
    
        # 初始化管道,输入以下参数:
        # model: 预训练模型
        # tokenizer: 预训练分词器(可选)
        # feature_extractor: 预训练特征提取器(可选)
        # image_processor: 图像处理器(可选)
        # modelcard: 模型卡片(可选)
        # framework: 框架(可选)
        # task: 任务名称
        # args_parser: 参数处理器
        # device: 运行设备(CPU 或 GPU)
        # torch_dtype: PyTorch 数据类型(可选)
        # binary_output: 是否以二进制输出
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
            ...
    # 保存当前 pipeline 的模型和分词器到指定目录中
    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        """
        Save the pipeline's model and tokenizer.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
            safe_serialization (`str`):
                Whether to save the model using `safetensors` or the traditional way for PyTorch or Tensorflow.
        """
        # 如果 save_directory 是文件则报错并返回
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        # 如果 save_directory 不存在则创建
        os.makedirs(save_directory, exist_ok=True)

        # 如果有注册的实现
        if hasattr(self, "_registered_impl"):
            # 添加信息到配置
            pipeline_info = self._registered_impl.copy()
            custom_pipelines = {}
            # 对于每个任务和信息
            for task, info in pipeline_info.items():
                # 如果实现不是当前类则继续下一次循环
                if info["impl"] != self.__class__:
                    continue

                # 复制信息
                info = info.copy()
                module_name = info["impl"].__module__
                last_module = module_name.split(".")[-1]
                # 将类名改为其名称/全名
                info["impl"] = f"{last_module}.{info['impl'].__name__}"
                info["pt"] = tuple(c.__name__ for c in info["pt"])
                info["tf"] = tuple(c.__name__ for c in info["tf"])

                custom_pipelines[task] = info
            # 设置当前模型配置的自定义流程信息
            self.model.config.custom_pipelines = custom_pipelines
            # 保存 pipeline 的自定义代码
            custom_object_save(self, save_directory)

        # 保存模型到给定目录
        self.model.save_pretrained(save_directory, safe_serialization=safe_serialization)

        # 如果存在分词器则保存到给定目录
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

        # 如果存在特征提取器则保存到给定目录
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory)

        # 如果存在图片处理器则保存到给定目录
        if self.image_processor is not None:
            self.image_processor.save_pretrained(save_directory)

        # 如果存在模型卡片则保存到给定目录
        if self.modelcard is not None:
            self.modelcard.save_pretrained(save_directory)

    # Scikit / Keras 接口到 transformers 的 pipeline，这个方法将转发到 __call__()
    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X)

    # Scikit / Keras 接口到 transformers 的 pipeline，这个方法将转发到 __call__()
    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X)

    @contextmanager
    # 定义一个上下文管理器，允许以与框架无关的方式在用户指定的设备上分配张量
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
        如果框架为 TensorFlow
        if self.framework == "tf":
            # 在指定的设备上分配张量，如果设备为 CPU，则使用"/CPU:0"，否则使用"/device:GPU:{self.device}"
            with tf.device("/CPU:0" if self.device == -1 else f"/device:GPU:{self.device}"):
                # 生成器
                yield
        else:
            如果设备类型为CUDA
            if self.device.type == "cuda":
                # 使用CUDA设备
                with torch.cuda.device(self.device):
                    # 生成器
                    yield
            else:
                yield

    # 确保PyTorch张量位于指定设备上
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
        使用_ensure_tensor_on_device函数，将输入参数和设备传递给它
        return self._ensure_tensor_on_device(inputs, self.device)

    # 将张量放置到指定设备上的私有函数
    def _ensure_tensor_on_device(self, inputs, device):
        如果inputs为ModelOutput类型
        if isinstance(inputs, ModelOutput):
            返回ModelOutput类型的字典，将每个张量转移到指定设备上
            return ModelOutput(
                {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
            )
        # 如果inputs为字典类型
        elif isinstance(inputs, dict):
            返回字典类型的结果，将每个张量转移到指定设备上
            return {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
        # 如果inputs为UserDict类型
        elif isinstance(inputs, UserDict):
            返回UserDict类型的结果，将每个张量转移到指定设备上
            return UserDict({name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()})
        # 如果inputs为列表类型
        elif isinstance(inputs, list):
            返回列表类型的结果，将每个张量转移到指定设备上
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        # 如果inputs为元组类型
        elif isinstance(inputs, tuple):
            返回元组类型的结果，将每个张量转移到指定设备上
            return tuple([self._ensure_tensor_on_device(item, device) for item in inputs])
        # 如果inputs为torch.Tensor类型
        elif isinstance(inputs, torch.Tensor):
            如果设备为CPU并且数据类型为torch.float16或torch.bfloat16
            if device == torch.device("cpu") and inputs.dtype in {torch.float16, torch.bfloat16}:
                将数据类型转换为float
                inputs = inputs.float()
            将张量移动到指定设备上
            return inputs.to(device)
        else:
            返回输入参数本身
            return inputs
    def check_model_type(self, supported_models: Union[List[str], dict]):
        """
        Check if the model class is in supported by the pipeline.

        Args:
            supported_models (`List[str]` or `dict`):
                The list of models supported by the pipeline, or a dictionary with model class values.
        """
        # 如果supported_models不是list类型，从模型映射创建
        if not isinstance(supported_models, list):  # Create from a model mapping
            supported_models_names = []
            # 遍历supported_models的值
            for _, model_name in supported_models.items():
                # 如果model_name是元组类型，则扩展列表
                if isinstance(model_name, tuple):
                    supported_models_names.extend(list(model_name))
                else:
                    supported_models_names.append(model_name)
            # 如果supported_models有_model_mapping属性
            if hasattr(supported_models, "_model_mapping"):
                # 遍历model_mapping的额外内容
                for _, model in supported_models._model_mapping._extra_content.items():
                    # 如果model是元组类型，则扩展列表
                    if isinstance(model_name, tuple):
                        supported_models_names.extend([m.__name__ for m in model])
                    else:
                        supported_models_names.append(model.__name__)
            # 将supported_models_names赋值给supported_models
            supported_models = supported_models_names
        # 如果self.model的类名不在supported_models中
        if self.model.__class__.__name__ not in supported_models:
            # 记录错误日志
            logger.error(
                f"The model '{self.model.__class__.__name__}' is not supported for {self.task}. Supported models are"
                f" {supported_models}."
            )


    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters):
        """
        _sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionnaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionnaries if the caller didn't specify a kwargs. This
        let's you keep defaults in function signatures, which is more "natural".

        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        # 抛出未实现异常
        raise NotImplementedError("_sanitize_parameters not implemented")


    @abstractmethod
    def preprocess(self, input_: Any, **preprocess_parameters: Dict) -> Dict[str, GenericTensor]:
        """
        Preprocess will take the `input_` of a specific pipeline and return a dictionary of everything necessary for
        `_forward` to run properly. It should contain at least one tensor, but might have arbitrary other items.
        """
        # 抛出未实现异常
        raise NotImplementedError("preprocess not implemented")

    @abstractmethod
    # 定义一个方法 _forward，接收输入张量字典和其他参数，返回模型输出
    def _forward(self, input_tensors: Dict[str, GenericTensor], **forward_parameters: Dict) -> ModelOutput:
        """
        _forward will receive the prepared dictionary from `preprocess` and run it on the model. This method might
        involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
        and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.

        It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
        code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
        of the code (leading to faster inference).
        """
        # 抛出未实现异常，子类需要实现具体的逻辑
        raise NotImplementedError("_forward not implemented")

    # 抽象方法，用于后处理模型输出
    @abstractmethod
    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: Dict) -> Any:
        """
        Postprocess will receive the raw outputs of the `_forward` method, generally tensors, and reformat them into
        something more friendly. Generally it will output a list or a dict or results (containing just strings and
        numbers).
        """
        # 抛出未实现异常，子类需要实现具体的逻辑
        raise NotImplementedError("postprocess not implemented")

    # 返回不执行梯度计算的上下文管理器
    def get_inference_context(self):
        return torch.no_grad

    # 对输入数据进行前向推断
    def forward(self, model_inputs, **forward_params):
        with self.device_placement():
            # 根据框架类型选择不同的前向推断方法
            if self.framework == "tf":
                model_inputs["training"] = False
                model_outputs = self._forward(model_inputs, **forward_params)
            elif self.framework == "pt":
                # 获取适用于推理的上下文管理器
                inference_context = self.get_inference_context()
                # 在推理上下文中运行前向方法，并确保模型输入在正确的设备上
                with inference_context():
                    model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                    model_outputs = self._forward(model_inputs, **forward_params)
                    model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
            else:
                raise ValueError(f"Framework {self.framework} is not supported")
        return model_outputs

    # 获取数据集迭代器
    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    # 定义一个方法，接收输入、预处理参数和其他参数，返回一个迭代器
    def __call__(self, inputs, *, preprocess_params={}, postprocess_params={}, num_workers=0, batch_size=1):
        # 如果输入是可计算大小的数据结构，如列表、元组等
        if isinstance(inputs, collections.abc.Sized):
            # 创建一个PipelineDataset对象，使用输入数据、预处理方法和预处理参数
            dataset = PipelineDataset(inputs, self.preprocess, preprocess_params)
        # 如果输入不是可计算大小的数据结构
        else:
            # 如果使用多个工作线程
            if num_workers > 1:
                # 发出警告，通知可能会出现错误，设置num_workers=1确保正确性
                logger.warning(
                    "For iterable dataset using num_workers>1 is likely to result"
                    " in errors since everything is iterable, setting `num_workers=1`"
                    " to guarantee correctness."
                )
                num_workers = 1
            # 创建一个PipelineIterator对象，使用输入数据、预处理方法和预处理参数
            dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        # 如果环境变量中不存在"TOKENIZERS_PARALLELISM"
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            # 输出日志信息，禁用分词并行处理，因为DataLoader已经在多线程中处理数据
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # 声明一个特征提取器，如果特征提取器存在则使用，否则使用图像处理器
        feature_extractor = self.feature_extractor if self.feature_extractor is not None else self.image_processor
        # 根据batch_size是否等于1，选择不同的数据合并方法
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, feature_extractor)
        # 创建一个DataLoader对象，使用上面创建的数据集、工作线程数量、batch大小和数据合并方法
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        # 创建一个PipelineIterator对象，使用上面创建的DataLoader、前向传播方法和其他参数
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        # 创建一个PipelineIterator对象，使用上面创建的模型迭代器、后处理方法和其他参数
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        # 返回最终迭代器
        return final_iterator
    # 这是 __call__ 方法,是该类的主要入口点,用于处理输入并生成输出
    def __call__(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):
        # 如果存在未使用的位置参数,则发出警告
        if args:
            logger.warning(f"Ignoring args : {args}")
    
        # 如果未指定 num_workers,则使用类属性 _num_workers 的值,如果为 None 则设为 0
        if num_workers is None:
            if self._num_workers is None:
                num_workers = 0
            else:
                num_workers = self._num_workers
        # 如果未指定 batch_size,则使用类属性 _batch_size 的值,如果为 None 则设为 1
        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size
    
        # 根据关键字参数 kwargs 获取预处理、前向传播和后处理的参数
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
    
        # 将 __init__ 中的参数和 __call__ 中的参数合并,但不修改 __init__ 中的参数
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}
    
        # 记录调用次数,并在调用超过 10 次且使用 PyTorch 框架且在 GPU 上运行时发出警告
        self.call_count += 1
        if self.call_count > 10 and self.framework == "pt" and self.device.type == "cuda":
            warnings.warn(
                "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a"
                " dataset",
                UserWarning,
            )
    
        # 检查输入是否为 Dataset、生成器或列表
        is_dataset = Dataset is not None and isinstance(inputs, Dataset)
        is_generator = isinstance(inputs, types.GeneratorType)
        is_list = isinstance(inputs, list)
    
        # 检查输入是否为可迭代对象
        is_iterable = is_dataset or is_generator or is_list
    
        # 检查是否可以使用迭代器处理输入
        can_use_iterator = self.framework == "pt" and (is_dataset or is_generator or is_list)
    
        # 如果输入为列表
        if is_list:
            # 如果可以使用迭代器
            if can_use_iterator:
                # 获取迭代器并转换为列表返回
                final_iterator = self.get_iterator(
                    inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params
                )
                outputs = list(final_iterator)
                return outputs
            # 否则使用 run_multi 方法处理
            else:
                return self.run_multi(inputs, preprocess_params, forward_params, postprocess_params)
        # 如果可以使用迭代器
        elif can_use_iterator:
            # 返回迭代器
            return self.get_iterator(
                inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params
            )
        # 如果输入为可迭代对象
        elif is_iterable:
            # 使用 iterate 方法处理
            return self.iterate(inputs, preprocess_params, forward_params, postprocess_params)
        # 如果输入为 PyTorch 且为 ChunkPipeline 类型
        elif self.framework == "pt" and isinstance(self, ChunkPipeline):
            # 使用 get_iterator 方法处理单个输入
            return next(
                iter(
                    self.get_iterator(
                        [inputs], num_workers, batch_size, preprocess_params, forward_params, postprocess_params
                    )
                )
            )
        # 否则使用 run_single 方法处理输入
        else:
            return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)
    # 定义一个方法，用于并行处理多个输入
    def run_multi(self, inputs, preprocess_params, forward_params, postprocess_params):
        # 使用列表推导式，对每个输入调用run_single方法，返回处理结果列表
        return [self.run_single(item, preprocess_params, forward_params, postprocess_params) for item in inputs]

    # 定义一个方法，用于处理单个输入
    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        # 对输入进行预处理，获取模型输入数据
        model_inputs = self.preprocess(inputs, **preprocess_params)
        # 模型前向传播，获取模型输出数据
        model_outputs = self.forward(model_inputs, **forward_params)
        # 对模型输出数据进行后处理，获取最终输出
        outputs = self.postprocess(model_outputs, **postprocess_params)
        # 返回处理后的输出
        return outputs

    # 定义一个方法，用于迭代处理输入数据
    def iterate(self, inputs, preprocess_params, forward_params, postprocess_params):
        # 这个函数应该再次变成`get_iterator`，这是一个临时的简单解决方案
        # 对输入列表进行迭代，依次调用run_single方法，使用yield生成器返回结果
        for input_ in inputs:
            yield self.run_single(input_, preprocess_params, forward_params, postprocess_params)
class ChunkPipeline(Pipeline):
    # 重载父类 Pipeline 的 run_single 方法
    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        all_outputs = []
        # 遍历数据的预处理结果，获取模型输入
        for model_inputs in self.preprocess(inputs, **preprocess_params):
            # 使用模型进行前向推断，获取模型输出
            model_outputs = self.forward(model_inputs, **forward_params)
            # 将模型输出添加到 all_outputs 中
            all_outputs.append(model_outputs)
        # 对所有输出进行后处理，得到最终结果
        outputs = self.postprocess(all_outputs, **postprocess_params)
        return outputs

    # 获取数据迭代器
    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        # 设置 tokenizer 的并行度为 false
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # 如果 num_workers 大于 1，则警告可能导致错误，将 num_workers 设置为 1
        if num_workers > 1:
            logger.warning(
                "For ChunkPipeline using num_workers>0 is likely to result in errors since everything is iterable,"
                " setting `num_workers=1` to guarantee correctness."
            )
            num_workers = 1
        # 创建 PipelineChunkIterator 对象，用于迭代数据
        dataset = PipelineChunkIterator(inputs, self.preprocess, preprocess_params)

        # 合并 feature_extractor 和 image_processor，构成 collate_fn 函数
        feature_extractor = self.feature_extractor if self.feature_extractor is not None else self.image_processor
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, feature_extractor)
        # 创建 DataLoader 对象，用于加载数据
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        # 创建 PipelinePackIterator 对象，进行模型推断
        model_iterator = PipelinePackIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        # 创建 PipelineIterator 对象，用于后处理
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        # 返回最终数据迭代器
        return final_iterator


class PipelineRegistry:
    # 初始化 PipelineRegistry 类
    def __init__(self, supported_tasks: Dict[str, Any], task_aliases: Dict[str, str]) -> None:
        self.supported_tasks = supported_tasks
        self.task_aliases = task_aliases

    # 获取支持的任务列表
    def get_supported_tasks(self) -> List[str]:
        supported_task = list(self.supported_tasks.keys()) + list(self.task_aliases.keys())
        supported_task.sort()
        return supported_task
    # 检查任务是否存在，如果存在，返回任务名称、目标任务和空值
    def check_task(self, task: str) -> Tuple[str, Dict, Any]:
        # 如果任务名在任务别名中，则使用任务别名中对应的任务名
        if task in self.task_aliases:
            task = self.task_aliases[task]
        # 如果任务名在支持的任务列表中
        if task in self.supported_tasks:
            # 获取目标任务
            targeted_task = self.supported_tasks[task]
            # 返回任务名称、目标任务和空值
            return task, targeted_task, None

        # 如果任务以"translation"开头
        if task.startswith("translation"):
            # 根据"_"将任务名称分割成令牌
            tokens = task.split("_")
            # 如果令牌长度为4，第一个令牌为"translation"，第三个令牌为"to"
            if len(tokens) == 4 and tokens[0] == "translation" and tokens[2] == "to":
                # 获取支持的翻译任务
                targeted_task = self.supported_tasks["translation"]
                # 将任务名称设置为"translation"
                task = "translation"
                # 返回任务名称、目标任务和语言对
                return task, targeted_task, (tokens[1], tokens[3])
            # 抛出KeyError，提示无效的翻译任务格式
            raise KeyError(f"Invalid translation task {task}, use 'translation_XX_to_YY' format")

        # 抛出KeyError，提示未知的任务
        raise KeyError(
            f"Unknown task {task}, available tasks are {self.get_supported_tasks() + ['translation_XX_to_YY']}"
        )

    # 注册流水线
    def register_pipeline(
        self,
        task: str,
        pipeline_class: type,
        pt_model: Optional[Union[type, Tuple[type]]] = None,
        tf_model: Optional[Union[type, Tuple[type]]] = None,
        default: Optional[Dict] = None,
        type: Optional[str] = None,
    ) -> None:
        # 如果任务在支持的任务列表中，发出警告，覆盖任务流水线
        if task in self.supported_tasks:
            logger.warning(f"{task} is already registered. Overwriting pipeline for task {task}...")

        # 如果pt_model为None，则设定为空元组
        if pt_model is None:
            pt_model = ()
        # 如果pt_model不是元组，将其转换为仅包含一个元素的元组
        elif not isinstance(pt_model, tuple):
            pt_model = (pt_model,)

        # 如果tf_model为None，则设定为空元组
        if tf_model is None:
            tf_model = ()
        # 如果tf_model不是元组，将其转换为仅包含一个元素的元组
        elif not isinstance(tf_model, tuple):
            tf_model = (tf_model,)

        # 创建任务实现字典
        task_impl = {"impl": pipeline_class, "pt": pt_model, "tf": tf_model}

        # 如果default不为None
        if default is not None:
            # 如果"default"不在default中，但"pt"或"tf"在default中
            if "model" not in default and ("pt" in default or "tf" in default):
                default = {"model": default}
            # 将默认值设置为default
            task_impl["default"] = default

        # 如果type不为None，将type设置为任务实现字典中的type值
        if type is not None:
            task_impl["type"] = type

        # 将任务及其实现添加到支持的任务列表中
        self.supported_tasks[task] = task_impl
        # 设置流水线类的已注册实现
        pipeline_class._registered_impl = {task: task_impl}

    # 转换为字典
    def to_dict(self):
        return self.supported_tasks
```