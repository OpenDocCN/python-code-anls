# `.\transformers\onnx\config.py`

```py
# 版权声明及许可协议信息

# 导入必要的库和模块
import copy  # 导入copy模块，用于复制对象
import dataclasses  # 导入dataclasses模块，用于创建数据类
import warnings  # 导入warnings模块，用于处理警告信息
from abc import ABC, abstractmethod  # 导入ABC和abstractmethod，用于定义抽象基类和抽象方法
from collections import OrderedDict  # 导入OrderedDict，用于创建有序字典
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union  # 引入类型提示

import numpy as np  # 导入numpy库
from packaging import version  # 导入version，用于处理版本信息

from ..utils import TensorType, is_torch_available, is_vision_available, logging  # 导入自定义的模块和库
from .utils import ParameterFormat, compute_effective_axis_dimension, compute_serialized_parameters_size  # 导入自定义的模块和库

# 如果类型检查生效，导入相应的类
if TYPE_CHECKING:
    from ..configuration_utils import PretrainedConfig  # 导入PretrainedConfig类
    from ..feature_extraction_utils import FeatureExtractionMixin  # 导入FeatureExtractionMixin类
    from ..image_processing_utils import ImageProcessingMixin  # 导入ImageProcessingMixin类
    from ..tokenization_utils_base import PreTrainedTokenizerBase  # 导入PreTrainedTokenizerBase类

# 如果视觉库可用，导入PIL库的Image模块
if is_vision_available():
    from PIL import Image  # 导入PIL库的Image模块，用于图像处理

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

DEFAULT_ONNX_OPSET = 11  # 定义默认的ONNX操作集版本号

# 外部数据格式大小限制为2Gb
EXTERNAL_DATA_FORMAT_SIZE_LIMIT = 2 * 1024 * 1024 * 1024  # 定义外部数据格式大小的限制为2Gb

@dataclasses.dataclass
class PatchingSpec:
    """
    数据类，保存补丁规范的信息

    Args:
        o: 包含要打补丁的操作的模块/对象
        name: 要打补丁的操作的名称
        custom_op: 打补丁的自定义操作
        orig_op: 被打补丁的原始操作
        op_wrapper: 包装器（可选），包装原始操作和自定义操作
            对于类方法或静态方法非常有用
    """
    o: Any  # 模块/对象
    name: str  # 操作的名称
    custom_op: Callable  # 自定义操作
    orig_op: Optional[Callable] = None  # 原始操作（可选）
    op_wrapper: Optional[Callable] = None  # 包装器（可选）

class OnnxConfig(ABC):
    """
    ONNX可导出模型的基类，描述通过ONNX格式导出模型的元数据信息。
    """

    default_fixed_batch = 2  # 默认固定批量大小为2
    default_fixed_sequence = 8  # 默认固定序列长度为8
    default_fixed_num_choices = 4  # 默认固定选择个数为4
    torch_onnx_minimum_version = version.parse("1.8")  # Torch ONNX最低版本为1.8
    # 定义任务到通用输出的映射字典
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

    def __init__(self, config: "PretrainedConfig", task: str = "default", patching_specs: List[PatchingSpec] = None):
        # 初始化函数，初始化配置、任务和修补规格列表
        self._config = config

        # 如果任务不在支持的任务列表中，引发值错误
        if task not in self._tasks_to_common_outputs:
            raise ValueError(
                f"{task} is not a supported task, supported tasks: {self._tasks_to_common_outputs.keys()}"
            )
        self.task = task

        # 对于每个传入的修补规格，如果原始操作为None，则替换为getattr(spec.o, spec.name)并添加到列表中
        self._patching_specs = []
        for spec in patching_specs if patching_specs is not None else []:
            final_spec = spec
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)

    @classmethod
    def from_model_config(cls, config: "PretrainedConfig", task: str = "default") -> "OnnxConfig":
        """
        根据模型配置实例化一个特定模型的OnnxConfig

        Args:
            config: 导出到ONNX时要使用的模型配置

        Returns:
            此模型的OnnxConfig
        """
        return cls(config, task=task)

    @property
    @abstractmethod
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the input tensors to provide to the model

        Returns:
            For each input: its name associated to the axes symbolic name and the axis position within the tensor
        """
        # 返回输入张量的轴定义映射，每个输入对应其名称、轴符号名称和张量中的轴位置
        raise NotImplementedError()

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the output tensors to provide to the model

        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        # 返回输出张量的轴定义映射，每个输出对应其名称、轴符号名称和张量中的轴位置
        common_outputs = self._tasks_to_common_outputs[self.task]
        return copy.deepcopy(common_outputs)

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting

        Returns:
            Dictionary with the keys (and their corresponding values) to override
        """
        # 如果模型的配置中存在 "use_cache" 属性，则返回一个字典，用于在导出之前覆盖配置中的键
        if hasattr(self._config, "use_cache"):
            return {"use_cache": False}

        return None

    @property
    def default_batch_size(self) -> int:
        """
        The default batch size to use if no other indication

        Returns:
            Integer > 0
        """
        # 如果没有其他指示，返回默认的批处理大小
        # 使用2来避免ONNX对单个样本批处理进行假设
        return OnnxConfig.default_fixed_batch

    @property
    def default_sequence_length(self) -> int:
        """
        The default sequence length to use if no other indication

        Returns:
            Integer > 0
        """
        # 如果没有其他指示，返回默认的序列长度
        return OnnxConfig.default_fixed_sequence

    @property
    def default_num_choices(self) -> int:
        """
        The default number of choices to use if no other indication

        Returns:
            Integer > 0
        """
        # 如果没有其他指示，返回默认的选择数量
        return OnnxConfig.default_fixed_num_choices

    @property
    def default_onnx_opset(self) -> int:
        """
        Which onnx opset to use when exporting the model

        Returns:
            Integer ONNX Opset version
        """
        # 导出模型时要使用的ONNX Opset版本
        return DEFAULT_ONNX_OPSET

    @property
    def atol_for_validation(self) -> float:
        """
        What absolute tolerance value to use during model conversion validation.

        Returns:
            Float absolute tolerance value.
        """
        # 模型转换验证期间要使用的绝对容差值
        return 1e-5

    @property
    def is_torch_support_available(self) -> bool:
        """
        The minimum PyTorch version required to export the model.

        Returns:
            `bool`: Whether the installed version of PyTorch is compatible with the model.
        """
        # 导出模型所需的最低PyTorch版本
        if is_torch_available():
            from transformers.utils import get_torch_version

            return version.parse(get_torch_version()) >= self.torch_onnx_minimum_version
        else:
            return False

    @staticmethod
    def use_external_data_format(num_parameters: int) -> bool:
        """
        Flag indicating if the model requires using external data format

        Args:
            num_parameters: Number of parameter on the model

        Returns:
            True if model.num_parameters() * size_of(float32) >= 2Gb False otherwise
        """
        # 判断模型是否需要使用外部数据格式
        return (
            compute_serialized_parameters_size(num_parameters, ParameterFormat.Float)
            >= EXTERNAL_DATA_FORMAT_SIZE_LIMIT
        )

    def _generate_dummy_images(
        self, batch_size: int = 2, num_channels: int = 3, image_height: int = 40, image_width: int = 40
    ):
        # 生成虚拟图片数据
        images = []
        for _ in range(batch_size):
            data = np.random.rand(image_height, image_width, num_channels) * 255
            images.append(Image.fromarray(data.astype("uint8")).convert("RGB"))
        return images

    def _generate_dummy_audio(
        self, batch_size: int = 2, sampling_rate: int = 22050, time_duration: float = 5.0, frequency: int = 220
    ):
        # 生成虚拟音频数据
        audio_data = []
        for _ in range(batch_size):
            # 时间变量
            t = np.linspace(0, time_duration, int(time_duration * sampling_rate), endpoint=False)

            # 在 `frequency` Hz 生成纯正弦波
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
    def generate_dummy_inputs_onnxruntime(self, reference_model_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Generate inputs for ONNX Runtime using the reference model inputs. Override this to run inference with seq2seq
        models which have the encoder and decoder exported as separate ONNX files.

        Args:
            reference_model_inputs ([`Mapping[str, Tensor]`):
                Reference inputs for the model.

        Returns:
            `Mapping[str, Tensor]`: The mapping holding the kwargs to provide to the model's forward function
        """
        # 使用参考模型输入为 ONNX Runtime 生成输入。覆盖此方法以在具有编码器和解码器作为单独 ONNX 文件导出的 seq2seq 模型上运行推理。
        return reference_model_inputs

    def patch_ops(self):
        for spec in self._patching_specs:
            custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
            setattr(spec.o, spec.name, custom_op)
    # 用于还原被替换的操作
    def restore_ops(self):
        # 遍历所有的修补规则
        for spec in self._patching_specs:
            # 如果没有操作包装器，使用原始操作
            orig_op = spec.orig_op if spec.op_wrapper is None else spec.op_wrapper(spec.orig_op)
            # 设置对象的属性为原始操作
            setattr(spec.o, spec.name, orig_op)

    @classmethod
    # 扁平化输出集合属性
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
        # 导入 itertools 模块里的 chain 函数
        from itertools import chain
        # 返回扁平化的结构，并使用新结构的键映射
        return {f"{name}.{idx}": item for idx, item in enumerate(chain.from_iterable(field))}
# 这个类是 OnnxConfig 的子类，提供了一些额外的功能和属性，如 use_past 和 fill_with_past_key_values_
class OnnxConfigWithPast(OnnxConfig, ABC):
    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        # 调用父类 OnnxConfig 的初始化方法
        super().__init__(config, task=task, patching_specs=patching_specs)
        # 设置 use_past 属性
        self.use_past = use_past

    @classmethod
    def with_past(cls, config: "PretrainedConfig", task: str = "default") -> "OnnxConfigWithPast":
        """
        创建一个 OnnxConfigWithPast 实例，并将 use_past 属性设置为 True
        
        参数:
            config: 要使用的预训练模型的配置
            task: 任务名称，默认为 "default"
        
        返回:
            OnnxConfigWithPast 实例
        """
        return cls(config, task=task, use_past=True)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 获取父类 OnnxConfig 的 outputs 属性
        common_outputs = super().outputs
        # 如果 use_past 为 True，则调用 fill_with_past_key_values_ 方法来修改 outputs
        if self.use_past:
            self.fill_with_past_key_values_(common_outputs, direction="outputs")
        return common_outputs

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        # 如果配置中有 use_cache 属性，则将其设置为 use_past 的值
        if hasattr(self._config, "use_cache"):
            return {"use_cache": self.use_past}
        return None

    @property
    def num_layers(self) -> int:
        """
        返回模型的层数。如果配置中没有 num_layers 属性，则需要重写此方法。
        """
        if not hasattr(self._config, "num_layers"):
            raise AttributeError(
                "could not find the number of layers attribute in the model configuration, override the num_layers"
                " property of the model OnnxConfig to solve this"
            )
        return self._config.num_layers

    @property
    def num_attention_heads(self) -> int:
        """
        返回模型的注意力头数。如果配置中没有 num_attention_heads 属性，则需要重写此方法。
        """
        if not hasattr(self._config, "num_attention_heads"):
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
        # 这个方法用于生成模型的dummy输入
        pass
    ) -> Mapping[str, Any]:
        # TODO: should we set seq_length = 1 when self.use_past = True?
        # 调用父类的generate_dummy_inputs方法生成通用的输入数据字典
        common_inputs = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 如果使用过去的隐藏状态
        if self.use_past:
            # 检查是否安装了PyTorch
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

            # 获取输入数据的批量大小和序列长度
            batch, seqlen = common_inputs["input_ids"].shape
            # 设置过去键值对长度，比序列长度多两个
            past_key_values_length = seqlen + 2
            # 设置形状，表示过去键值对的形状
            shape = (
                batch,
                self.num_attention_heads,
                past_key_values_length,
                self._config.hidden_size // self.num_attention_heads,
            )

            # 如果输入数据字典包含注意力掩码
            if "attention_mask" in common_inputs:
                # 获取掩码数据类型
                mask_dtype = common_inputs["attention_mask"].dtype
                # 在掩码的末尾添加与过去键值对长度相同的掩码值
                common_inputs["attention_mask"] = torch.cat(
                    [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)],
                    dim=1,
                )

            # 初始化过去键值对列表
            common_inputs["past_key_values"] = []
            # 遍历每一层
            for _ in range(self.num_layers):
                # 将零张量添加到过去键值对列表中
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))

        # 返回通用输入数据字典
        return common_inputs

    # 将输入或输出映射填充为考虑过去键值对的动态轴
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
        # 检查方向是否有效
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        # 设置名称
        name = "past_key_values" if direction == "inputs" else "present"
        # 遍历每一层
        for i in range(self.num_layers):
            # 设置键的轴
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
            # 如果反转了值的形状
            if inverted_values_shape:
                inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch", 1: "past_sequence + sequence"}
            else:
                inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}

    # 将过去键值对展平
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 设置展平后的输出字典中的键和值
        flattened_output[f"{name}.{idx}.key"] = t[0]
        flattened_output[f"{name}.{idx}.value"] = t[1]
    # 定义一个方法，用于将输出集合属性扁平化，返回字段名到值的字典
    def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        # 初始化一个空的扁平化输出字典
        flattened_output = {}
        
        # 如果字段名在["present", "past_key_values"]中
        if name in ["present", "past_key_values"]:
            # 遍历字段中的元素
            for idx, t in enumerate(field):
                # 调用内部方法，将字段中的值扁平化存储到flattened_output中
                self._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            # 如果字段名不在特定列表中，则调用父类的方法进行处理
            flattened_output = super().flatten_output_collection_property(name, field)
    
        # 返回扁平化处理后的输出字典
        return flattened_output
class OnnxSeq2SeqConfigWithPast(OnnxConfigWithPast):
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 调用父类的outputs方法获取通用输出
        common_outputs = super(OnnxConfigWithPast, self).outputs
        # 重命名输出轴，根据名称中是否包含"encoder"或"decoder"来决定序列的名字
        for name, axes_names in common_outputs.items():
            sequence_name = "encoder_sequence" if "encoder" in name else "decoder_sequence"
            # 遍历输出轴，根据名称中是否包含"sequence"来替换为对应的序列名字
            for axis_idx, name in axes_names.items():
                if "sequence" in name:
                    axes_names[axis_idx] = sequence_name
                else:
                    axes_names[axis_idx] = name
        # 如果使用过去的信息，则填充输出轴信息
        if self.use_past:
            self.fill_with_past_key_values_(common_outputs, direction="outputs")

        return common_outputs

    @property
    def num_layers(self) -> Tuple[int]:
        try:
            num_layers = super().num_layers
            num_layers = (num_layers, num_layers)
        except AttributeError:
            # 获取编码器和解码器层数，如果找不到，则抛出异常
            if hasattr(self._config, "encoder_layers") and hasattr(self._config, "decoder_layers"):
                num_layers = (self._config.encoder_layers, self._config.decoder_layers)
            else:
                raise AttributeError(
                    "could not find the number of encoder and decoder layers attributes in the model configuration,"
                    " override the num_layers property of the model OnnxConfig to solve this"
                )

        return num_layers

    @property
    def num_attention_heads(self) -> Tuple[int]:
        try:
            num_attention_heads = super().num_attention_heads
            num_attention_heads = (num_attention_heads, num_attention_heads)
        except AttributeError:
            # 获取编码器和解码器注意力头数，如果找不到，则抛出异常
            if hasattr(self._config, "encoder_attention_heads") and hasattr(self._config, "decoder_attention_heads"):
                num_attention_heads = (self._config.encoder_attention_heads, self._config.decoder_attention_heads)
            else:
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
        # 以下内容省略，方法未完整
    # 将过去的键值填充到输入或输出中
    def fill_with_past_key_values_(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        # 检查方向参数是否正确
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        # 根据方向确定键名
        name = "past_key_values" if direction == "inputs" else "present"

        # 如果模型配置中存在编码器和解码器层数信息，则都考虑进来
        num_encoder_layers, num_decoder_layers = self.num_layers
        min_num_layers = min(num_encoder_layers, num_decoder_layers)
        max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
        remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

        # 设置编码器和解码器序列的键名
        encoder_sequence = "past_encoder_sequence"
        decoder_sequence = "past_decoder_sequence" if direction == "inputs" else "past_decoder_sequence + sequence"

        # 填充过去的键值到输入或输出中
        for i in range(min_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.key"] = {0: "batch", 2: encoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.value"] = {0: "batch", 2: encoder_sequence}

        # 如果编码器和解码器层数不相等，则处理剩余层
        for i in range(min_num_layers, max_num_layers):
            # 确定剩余层对应的序列方向
            if remaining_side_name == "encoder":
                axes_info = {0: "batch", 2: encoder_sequence}
            else:
                axes_info = {0: "batch", 2: decoder_sequence}
            # 将剩余层的键值填充到输入或输出中
            inputs_or_outputs[f"{name}.{i}.{remaining_side_name}.key"] = axes_info

    # 将过去的键值展开到输出中
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 将解码器和编码器的过去键值展开到输出中
        flattened_output[f"{name}.{idx}.decoder.key"] = t[0]
        flattened_output[f"{name}.{idx}.decoder.value"] = t[1]
        flattened_output[f"{name}.{idx}.encoder.key"] = t[2]
        flattened_output[f"{name}.{idx}.encoder.value"] = t[3]
```