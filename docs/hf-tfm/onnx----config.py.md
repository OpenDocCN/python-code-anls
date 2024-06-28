# `.\onnx\config.py`

```py
# 版权声明和版权信息
# 版权所有 © 2021 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的条款，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按“原样”分发，
# 不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

import copy  # 导入 copy 模块
import dataclasses  # 导入 dataclasses 模块
import warnings  # 导入 warnings 模块
from abc import ABC, abstractmethod  # 从 abc 模块导入 ABC 抽象基类和 abstractmethod 装饰器
from collections import OrderedDict  # 从 collections 模块导入 OrderedDict 类
from typing import (  # 导入多个类型提示，包括 TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np  # 导入 numpy 库，并使用 np 别名
from packaging import version  # 从 packaging 模块导入 version 模块

from ..utils import TensorType, is_torch_available, is_vision_available, logging  # 从相对路径的 ..utils 模块导入多个函数和类
from .utils import (  # 从相对路径的 .utils 模块导入 ParameterFormat, compute_effective_axis_dimension, compute_serialized_parameters_size 函数
    ParameterFormat,
    compute_effective_axis_dimension,
    compute_serialized_parameters_size,
)

if TYPE_CHECKING:
    from ..configuration_utils import PretrainedConfig  # 如果 TYPE_CHECKING 为真，导入 PretrainedConfig 类
    from ..feature_extraction_utils import FeatureExtractionMixin  # 如果 TYPE_CHECKING 为真，导入 FeatureExtractionMixin 类
    from ..image_processing_utils import ImageProcessingMixin  # 如果 TYPE_CHECKING 为真，导入 ImageProcessingMixin 类
    from ..tokenization_utils_base import PreTrainedTokenizerBase  # 如果 TYPE_CHECKING 为真，导入 PreTrainedTokenizerBase 类

if is_vision_available():
    from PIL import Image  # 如果 is_vision_available() 返回真，导入 PIL 库中的 Image 类

logger = logging.get_logger(__name__)  # 获取当前模块的 logger 对象

DEFAULT_ONNX_OPSET = 11  # 设置默认的 ONNX 操作集版本号为 11

# 外部数据格式大小限制为 2 GB
EXTERNAL_DATA_FORMAT_SIZE_LIMIT = 2 * 1024 * 1024 * 1024


@dataclasses.dataclass
class PatchingSpec:
    """
    数据类，保存补丁规范。

    Args:
        o: 包含要打补丁的操作的模块 / 对象
        name: 要打补丁的操作的名称
        custom_op: 打补丁的自定义操作
        orig_op: 正在被打补丁的原始操作
        op_wrapper: 包装器（可选），包装原始操作和自定义操作。
            对于类或静态方法很有用。

    """

    o: Any
    name: str
    custom_op: Callable
    orig_op: Optional[Callable] = None
    op_wrapper: Optional[Callable] = None


class OnnxConfig(ABC):
    """
    ONNX 可导出模型的基类，描述通过 ONNX 格式导出模型的元数据。
    """

    default_fixed_batch = 2  # 默认固定批次大小为 2
    default_fixed_sequence = 8  # 默认固定序列长度为 8
    default_fixed_num_choices = 4  # 默认固定选择数量为 4
    torch_onnx_minimum_version = version.parse("1.8")  # Torch 的最小 ONNX 版本为 1.8
    # 定义一个类变量，映射不同任务到其标准输出格式的有序字典
    _tasks_to_common_outputs = {
        "causal-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "default": OrderedDict({"last_hidden_state": {0: "batch", 1: "sequence"}}),
        "image-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "image-segmentation": OrderedDict(
            {
                "logits": {0: "batch", 1: "sequence"},
                "pred_boxes": {0: "batch", 1: "sequence"},
                "pred_masks": {0: "batch", 1: "sequence"},
            }
        ),
        "masked-im": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "masked-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "multiple-choice": OrderedDict({"logits": {0: "batch"}}),
        "object-detection": OrderedDict(
            {
                "logits": {0: "batch", 1: "sequence"},
                "pred_boxes": {0: "batch", 1: "sequence"},
            }
        ),
        "question-answering": OrderedDict(
            {
                "start_logits": {0: "batch", 1: "sequence"},
                "end_logits": {0: "batch", 1: "sequence"},
            }
        ),
        "semantic-segmentation": OrderedDict({"logits": {0: "batch", 1: "num_labels", 2: "height", 3: "width"}}),
        "seq2seq-lm": OrderedDict({"logits": {0: "batch", 1: "decoder_sequence"}}),
        "sequence-classification": OrderedDict({"logits": {0: "batch"}}),
        "token-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "vision2seq-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "speech2seq-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
    }

    # 类的构造函数，初始化对象时调用
    def __init__(self, config: "PretrainedConfig", task: str = "default", patching_specs: List[PatchingSpec] = None):
        self._config = config

        # 检查传入的任务是否在支持的任务列表中，如果不在则抛出异常
        if task not in self._tasks_to_common_outputs:
            raise ValueError(
                f"{task} is not a supported task, supported tasks: {self._tasks_to_common_outputs.keys()}"
            )
        self.task = task

        # 初始化对象的属性 _patching_specs，用于记录应用到对象上的补丁规格
        self._patching_specs = []
        # 遍历传入的 patching_specs 列表，如果不为空则逐个处理
        for spec in patching_specs if patching_specs is not None else []:
            final_spec = spec
            # 如果补丁规格中的原始操作为 None，则替换为 spec.o 上的 spec.name 属性的值
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)

    # 类方法，根据模型配置生成一个 OnnxConfig 实例
    @classmethod
    def from_model_config(cls, config: "PretrainedConfig", task: str = "default") -> "OnnxConfig":
        """
        根据模型配置生成一个 OnnxConfig 实例

        Args:
            config: 导出到 ONNX 时使用的模型配置

        Returns:
            该模型的 OnnxConfig 实例
        """
        return cls(config, task=task)

    # 抽象属性，子类必须实现该属性
    @property
    @abstractmethod
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the input tensors to provide to the model

        Returns:
            For each input: its name associated to the axes symbolic name and the axis position within the tensor
        """
        # 返回模型需要的输入张量的轴定义映射
        raise NotImplementedError()

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the output tensors to provide to the model

        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        # 获取当前任务对应的通用输出，并深拷贝返回
        common_outputs = self._tasks_to_common_outputs[self.task]
        return copy.deepcopy(common_outputs)

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting

        Returns:
            Dictionary with the keys (and their corresponding values) to override
        """
        # 如果模型配置对象有"use_cache"属性，则返回该属性为False的字典
        if hasattr(self._config, "use_cache"):
            return {"use_cache": False}

        # 否则返回None，表示无需覆盖任何配置项
        return None

    @property
    def default_batch_size(self) -> int:
        """
        The default batch size to use if no other indication

        Returns:
            Integer > 0
        """
        # 返回默认的批处理大小，避免ONNX对单个样本批处理的假设
        return OnnxConfig.default_fixed_batch

    @property
    def default_sequence_length(self) -> int:
        """
        The default sequence length to use if no other indication

        Returns:
            Integer > 0
        """
        # 返回默认的序列长度
        return OnnxConfig.default_fixed_sequence

    @property
    def default_num_choices(self) -> int:
        """
        The default number of choices to use if no other indication

        Returns:
            Integer > 0
        """
        # 返回默认的选择数量
        return OnnxConfig.default_fixed_num_choices

    @property
    def default_onnx_opset(self) -> int:
        """
        Which onnx opset to use when exporting the model

        Returns:
            Integer ONNX Opset version
        """
        # 返回导出模型时要使用的ONNX opset版本
        return DEFAULT_ONNX_OPSET

    @property
    def atol_for_validation(self) -> float:
        """
        What absolute tolerance value to use during model conversion validation.

        Returns:
            Float absolute tolerance value.
        """
        # 返回在模型转换验证期间使用的绝对容差值
        return 1e-5

    @property
    def is_torch_support_available(self) -> bool:
        """
        The minimum PyTorch version required to export the model.

        Returns:
            `bool`: Whether the installed version of PyTorch is compatible with the model.
        """
        # 检查是否安装了PyTorch，如果是，则检查版本是否达到要求的最小版本
        if is_torch_available():
            from transformers.utils import get_torch_version

            return version.parse(get_torch_version()) >= self.torch_onnx_minimum_version
        else:
            # 如果未安装PyTorch，则返回False
            return False
    def use_external_data_format(num_parameters: int) -> bool:
        """
        Flag indicating if the model requires using external data format

        Args:
            num_parameters: Number of parameters in the model

        Returns:
            True if the serialized parameter size in float32 >= 2Gb, False otherwise
        """

        return (
            compute_serialized_parameters_size(num_parameters, ParameterFormat.Float)
            >= EXTERNAL_DATA_FORMAT_SIZE_LIMIT
        )

    def _generate_dummy_images(
        self, batch_size: int = 2, num_channels: int = 3, image_height: int = 40, image_width: int = 40
    ):
        """
        Generate dummy images as a list of PIL Image objects.

        Args:
            batch_size: Number of images to generate
            num_channels: Number of color channels per image
            image_height: Height of each image
            image_width: Width of each image

        Returns:
            List of PIL Image objects
        """
        images = []
        for _ in range(batch_size):
            data = np.random.rand(image_height, image_width, num_channels) * 255
            images.append(Image.fromarray(data.astype("uint8")).convert("RGB"))
        return images

    def _generate_dummy_audio(
        self, batch_size: int = 2, sampling_rate: int = 22050, time_duration: float = 5.0, frequency: int = 220
    ):
        """
        Generate dummy audio data as a list of numpy arrays representing audio samples.

        Args:
            batch_size: Number of audio samples to generate
            sampling_rate: Sampling rate of the audio samples
            time_duration: Duration of each audio sample in seconds
            frequency: Frequency of the sine wave to generate

        Returns:
            List of numpy arrays representing audio samples
        """
        audio_data = []
        for _ in range(batch_size):
            t = np.linspace(0, time_duration, int(time_duration * sampling_rate), endpoint=False)
            audio_data.append(0.5 * np.sin(2 * np.pi * frequency * t))
        return audio_data

    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin", "ImageProcessingMixin"],
        batch_size: int = -1,
        seq_length: int = -1,
        num_choices: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
        sampling_rate: int = 22050,
        time_duration: float = 5.0,
        frequency: int = 220,
        tokenizer: "PreTrainedTokenizerBase" = None,
    ):
        """
        Generate dummy inputs for the model, such as images, audio, or text tokens.

        Args:
            preprocessor: Preprocessor object for handling different input types
            batch_size: Number of inputs to generate
            seq_length: Length of sequence inputs
            num_choices: Number of choices (for multiple choice scenarios)
            is_pair: Whether the input is a pair
            framework: Framework type for input handling
            num_channels: Number of channels for image inputs
            image_width: Width of image inputs
            image_height: Height of image inputs
            sampling_rate: Sampling rate for audio inputs
            time_duration: Duration of audio inputs
            frequency: Frequency of audio inputs
            tokenizer: Tokenizer object for token-based inputs

        Returns:
            Dummy inputs suitable for the model
        """

    def generate_dummy_inputs_onnxruntime(self, reference_model_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Generate inputs for ONNX Runtime using the reference model inputs.

        Args:
            reference_model_inputs: Mapping of inputs for the model

        Returns:
            Mapping of inputs suitable for the model's forward function in ONNX Runtime
        """
        return reference_model_inputs

    def patch_ops(self):
        """
        Patch operations on the model instance using predefined specifications.
        """
        for spec in self._patching_specs:
            custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
            setattr(spec.o, spec.name, custom_op)
    # 恢复操作函数原始状态的方法
    def restore_ops(self):
        # 遍历保存在 self._patching_specs 中的所有规格
        for spec in self._patching_specs:
            # 如果规格中的操作包装器为 None，则使用原始操作；否则使用操作包装器包装原始操作
            orig_op = spec.orig_op if spec.op_wrapper is None else spec.op_wrapper(spec.orig_op)
            # 将恢复后的操作设置回原始对象的对应属性上
            setattr(spec.o, spec.name, orig_op)

    @classmethod
    def flatten_output_collection_property(cls, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        """
        Flatten any potential nested structure expanding the name of the field with the index of the element within the
        structure.

        Args:
            name: The name of the nested structure
            field: The structure to, potentially, be flattened

        Returns:
            (Dict[str, Any]): Outputs with flattened structure and key mapping this new structure.

        """
        # 导入 itertools 模块中的 chain 函数，用于将多个可迭代对象连接成一个迭代器
        from itertools import chain

        # 返回一个字典，其键为格式化后的字段名（包含结构的名字和元素在结构中的索引），值为从嵌套结构展开后的元素
        return {f"{name}.{idx}": item for idx, item in enumerate(chain.from_iterable(field))}
class OnnxConfigWithPast(OnnxConfig, ABC):
    # 继承自 OnnxConfig 类，并实现 ABC 抽象类
    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        # 调用父类的构造方法初始化对象
        super().__init__(config, task=task, patching_specs=patching_specs)
        # 设置本类特有的 use_past 属性
        self.use_past = use_past

    @classmethod
    def with_past(cls, config: "PretrainedConfig", task: str = "default") -> "OnnxConfigWithPast":
        """
        实例化一个带有 `use_past` 属性设置为 True 的 OnnxConfig 对象

        Args:
            config: 导出到 ONNX 时使用的底层模型配置

        Returns:
            设置了 `.use_past = True` 的 OnnxConfig 对象
        """
        return cls(config, task=task, use_past=True)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 获取父类的 outputs 属性
        common_outputs = super().outputs
        # 如果 use_past 属性为 True，则调用本类方法填充输出
        if self.use_past:
            self.fill_with_past_key_values_(common_outputs, direction="outputs")

        return common_outputs

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        # 如果 _config 对象有 use_cache 属性，则返回字典 {"use_cache": self.use_past}
        if hasattr(self._config, "use_cache"):
            return {"use_cache": self.use_past}

        return None

    @property
    def num_layers(self) -> int:
        """
        从模型配置中获取层数属性。对于不称为 `num_layers` 的模型配置，请覆盖此方法。
        """
        if not hasattr(self._config, "num_layers"):
            # 如果模型配置中找不到层数属性，则引发 AttributeError
            raise AttributeError(
                "could not find the number of layers attribute in the model configuration, override the num_layers"
                " property of the model OnnxConfig to solve this"
            )
        return self._config.num_layers

    @property
    def num_attention_heads(self) -> int:
        """
        从模型配置中获取注意力头数属性。对于不称为 `num_attention_heads` 的模型配置，请覆盖此方法。
        """
        if not hasattr(self._config, "num_attention_heads"):
            # 如果模型配置中找不到注意力头数属性，则引发 AttributeError
            raise AttributeError(
                "could not find the number of attention heads attribute in the model configuration, override the"
                " num_attention_heads property of the model OnnxConfig to solve this"
            )
        return self._config.num_attention_heads

    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ):
        # 此方法用于生成虚拟输入，具体实现由子类完成
        pass
    ) -> Mapping[str, Any]:
        # TODO: should we set seq_length = 1 when self.use_past = True?
        # 调用父类方法生成虚拟输入数据，获取通用的输入字典
        common_inputs = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        if self.use_past:
            # 如果使用过去的状态信息
            if not is_torch_available():
                # 检查是否安装了 PyTorch
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

            # 获取 batch 和 seq_length
            batch, seqlen = common_inputs["input_ids"].shape
            # 计算过去键值对的长度
            past_key_values_length = seqlen + 2
            # 定义张量的形状
            shape = (
                batch,
                self.num_attention_heads,
                past_key_values_length,
                self._config.hidden_size // self.num_attention_heads,
            )

            if "attention_mask" in common_inputs:
                # 如果存在注意力掩码，扩展掩码的长度
                mask_dtype = common_inputs["attention_mask"].dtype
                common_inputs["attention_mask"] = torch.cat(
                    [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)],
                    dim=1,
                )

            # 初始化过去键值对列表
            common_inputs["past_key_values"] = []
            for _ in range(self.num_layers):
                # 为每一层添加零初始化的过去键值对
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))

        # 返回填充后的通用输入字典
        return common_inputs

    def fill_with_past_key_values_(
        self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str, inverted_values_shape: bool = False
    ):
        """
        Fill the input_or_outputs mapping with past_key_values dynamic axes considering.

        Args:
            inputs_or_outputs: The mapping to fill.
            direction: either "inputs" or "outputs", it specifies whether input_or_outputs is the input mapping or the
                output mapping, this is important for axes naming.
            inverted_values_shape:
                If `True`, store values on dynamic axis 1, else on axis 2.

        """
        # 检查方向是否合法
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        # 根据方向设置键的名称前缀
        name = "past_key_values" if direction == "inputs" else "present"
        for i in range(self.num_layers):
            # 设置键值对的动态轴信息
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
            if inverted_values_shape:
                inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch", 1: "past_sequence + sequence"}
            else:
                inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}

    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 将过去键值对扁平化后存入输出字典
        flattened_output[f"{name}.{idx}.key"] = t[0]
        flattened_output[f"{name}.{idx}.value"] = t[1]
    # 定义一个方法用于扁平化输出集合属性
    def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        # 初始化一个空字典用于存储扁平化后的输出
        flattened_output = {}
        # 如果属性名为 "present" 或 "past_key_values"
        if name in ["present", "past_key_values"]:
            # 遍历字段中的每个元素，使用索引和元素调用 _flatten_past_key_values_ 方法
            for idx, t in enumerate(field):
                self._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            # 否则调用父类的同名方法处理字段，并将结果赋给 flattened_output
            flattened_output = super().flatten_output_collection_property(name, field)

        # 返回扁平化后的输出字典
        return flattened_output
class OnnxSeq2SeqConfigWithPast(OnnxConfigWithPast):
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 调用父类方法获取通用输出
        common_outputs = super(OnnxConfigWithPast, self).outputs
        # 对输出的轴进行适当重命名
        for name, axes_names in common_outputs.items():
            # 根据名称中是否包含"encoder"决定序列名称
            sequence_name = "encoder_sequence" if "encoder" in name else "decoder_sequence"
            for axis_idx, name in axes_names.items():
                # 如果轴名称中包含"sequence"，则重命名为对应的序列名称
                if "sequence" in name:
                    axes_names[axis_idx] = sequence_name
                else:
                    # 否则保持原名称不变
                    axes_names[axis_idx] = name
        # 如果使用过去状态信息，则填充通用输出中的键值对
        if self.use_past:
            self.fill_with_past_key_values_(common_outputs, direction="outputs")

        return common_outputs

    @property
    def num_layers(self) -> Tuple[int]:
        try:
            # 尝试获取父类中的层数
            num_layers = super().num_layers
            # 将层数转换为元组形式 (num_layers, num_layers)
            num_layers = (num_layers, num_layers)
        except AttributeError:
            # 如果父类中不存在 num_layers 属性，则根据配置信息获取编码器和解码器层数
            if hasattr(self._config, "encoder_layers") and hasattr(self._config, "decoder_layers"):
                num_layers = (self._config.encoder_layers, self._config.decoder_layers)
            else:
                # 抛出属性错误异常，提示在模型配置中找不到编码器和解码器层数的属性
                raise AttributeError(
                    "could not find the number of encoder and decoder layers attributes in the model configuration,"
                    " override the num_layers property of the model OnnxConfig to solve this"
                )

        return num_layers

    @property
    def num_attention_heads(self) -> Tuple[int]:
        try:
            # 尝试获取父类中的注意力头数
            num_attention_heads = super().num_attention_heads
            # 将注意力头数转换为元组形式 (num_attention_heads, num_attention_heads)
            num_attention_heads = (num_attention_heads, num_attention_heads)
        except AttributeError:
            # 如果父类中不存在 num_attention_heads 属性，则根据配置信息获取编码器和解码器注意力头数
            if hasattr(self._config, "encoder_attention_heads") and hasattr(self._config, "decoder_attention_heads"):
                num_attention_heads = (self._config.encoder_attention_heads, self._config.decoder_attention_heads)
            else:
                # 抛出属性错误异常，提示在模型配置中找不到编码器和解码器注意力头数的属性
                raise AttributeError(
                    "could not find the number of attention heads for the encoder and the decoder attributes in the"
                    " model configuration, override the num_attention_heads property of the model OnnxConfig to solve"
                    " this"
                )
        return num_attention_heads

    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    def fill_with_past_key_values_(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        # 如果方向不是"inputs"或"outputs"，则抛出数值错误异常
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        # 根据方向确定名称
        name = "past_key_values" if direction == "inputs" else "present"

        # 获取编码器和解码器层数
        num_encoder_layers, num_decoder_layers = self.num_layers
        # 计算最小和最大层数差异
        min_num_layers = min(num_encoder_layers, num_decoder_layers)
        max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
        # 确定剩余方向的名称（编码器或解码器）
        remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

        # 设置编码器和解码器的序列名称
        encoder_sequence = "past_encoder_sequence"
        decoder_sequence = "past_decoder_sequence" if direction == "inputs" else "past_decoder_sequence + sequence"

        # 填充每一层的键值对
        for i in range(min_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.key"] = {0: "batch", 2: encoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.value"] = {0: "batch", 2: encoder_sequence}

        # 对于剩余的层，根据剩余方向名称设置相应的轴信息
        for i in range(min_num_layers, max_num_layers):
            if remaining_side_name == "encoder":
                axes_info = {0: "batch", 2: encoder_sequence}
            else:
                axes_info = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.{remaining_side_name}.key"] = axes_info

    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 将 t 中的键值展平到给定的名称和索引中
        flattened_output[f"{name}.{idx}.decoder.key"] = t[0]
        flattened_output[f"{name}.{idx}.decoder.value"] = t[1]
        flattened_output[f"{name}.{idx}.encoder.key"] = t[2]
        flattened_output[f"{name}.{idx}.encoder.value"] = t[3]
```