# `.\models\layoutlmv3\configuration_layoutlmv3.py`

```py
# 指定编码格式为UTF-8
# 版权声明和许可信息
# 依据 Apache 许可 2.0 版本，本代码受许可保护
# 禁止在未遵守许可的情况下使用本文件
# 可在以下网址获取许可副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不附带任何明示或暗示的保证或条件
# 有关详细信息，请参阅许可证
""" LayoutLMv3 模型配置"""

# 导入所需模块和库
from collections import OrderedDict  # 有序字典，保留了元素的添加顺序
from typing import TYPE_CHECKING, Any, Mapping, Optional  # 类型提示，不会影响实际代码执行

from packaging import version  # 版本信息管理

from ...configuration_utils import PretrainedConfig  # 预训练配置的基类
from ...onnx import OnnxConfig  # ONNX 模型配置
from ...onnx.utils import compute_effective_axis_dimension  # 计算有效轴维度的函数
from ...utils import logging  # 日志记录工具

# 如果只是类型检查，则导入特定模块
if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin  # 数据处理工具
    from ...utils import TensorType  # 张量类型

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置的映射字典
LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/layoutlmv3-base": "https://huggingface.co/microsoft/layoutlmv3-base/resolve/main/config.json",
}

# LayoutLMv3 模型配置类，继承自预训练配置类
class LayoutLMv3Config(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`LayoutLMv3Model`] 的配置。它用于根据指定的参数实例化一个 LayoutLMv3 模型，定义模型架构。
    使用默认值实例化一个配置将会产生类似于 LayoutLMv3 [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。

    示例：

    ```python
    >>> from transformers import LayoutLMv3Config, LayoutLMv3Model

    >>> # 初始化一个 LayoutLMv3 microsoft/layoutlmv3-base 风格的配置
    >>> configuration = LayoutLMv3Config()

    >>> # 使用 microsoft/layoutlmv3-base 风格的配置初始化一个（带有随机权重）模型
    >>> model = LayoutLMv3Model(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    # 模型类型
    model_type = "layoutlmv3"
    # 初始化函数，用于创建一个新的实例
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为50265
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置嵌入数，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化epsilon值，默认为1e-5
        pad_token_id=1,  # 填充标记id，默认为1
        bos_token_id=0,  # 开始标记id，默认为0
        eos_token_id=2,  # 结束标记id，默认为2
        max_2d_position_embeddings=1024,  # 最大二维位置嵌入数，默认为1024
        coordinate_size=128,  # 坐标大小，默认为128
        shape_size=128,  # 形状大小，默认为128
        has_relative_attention_bias=True,  # 是否有相对注意力偏差，默认为True
        rel_pos_bins=32,  # 相对位置箱数，默认为32
        max_rel_pos=128,  # 最大相对位置，默认为128
        rel_2d_pos_bins=64,  # 二维相对位置箱数，默认为64
        max_rel_2d_pos=256,  # 最大二维相对位置，默认为256
        has_spatial_attention_bias=True,  # 是否有空间注意力偏差，默认为True
        text_embed=True,  # 文本嵌入标记，默认为True
        visual_embed=True,  # 视觉嵌入标记，默认为True
        input_size=224,  # 输入大小，默认为224
        num_channels=3,  # 通道数，默认为3
        patch_size=16,  # 补丁大小，默认为16
        classifier_dropout=None,  # 分类器dropout，默认为None
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化函数
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,  # 其他关键字参数
        )
        # 设置实例的最大二维位置嵌入数
        self.max_2d_position_embeddings = max_2d_position_embeddings
        # 设置实例的坐标大小
        self.coordinate_size = coordinate_size
        # 设置实例的形状大小
        self.shape_size = shape_size
        # 设置实例是否有相对注意力偏差
        self.has_relative_attention_bias = has_relative_attention_bias
        # 设置实例的相对位置箱数
        self.rel_pos_bins = rel_pos_bins
        # 设置实例的最大相对位置
        self.max_rel_pos = max_rel_pos
        # 设置实例是否有空间注意力偏差
        self.has_spatial_attention_bias = has_spatial_attention_bias
        # 设置实例的二维相对位置箱数
        self.rel_2d_pos_bins = rel_2d_pos_bins
        # 设置实例的最大二维相对位置
        self.max_rel_2d_pos = max_rel_2d_pos
        # 设置实例的文本嵌入标记
        self.text_embed = text_embed
        # 设置实例的视觉嵌入标记
        self.visual_embed = visual_embed
        # 设置实例的输入大小
        self.input_size = input_size
        # 设置实例的通道数
        self.num_channels = num_channels
        # 设置实例的补丁大小
        self.patch_size = patch_size
        # 设置实例的分类器dropout
        self.classifier_dropout = classifier_dropout
class LayoutLMv3OnnxConfig(OnnxConfig):
    # 继承自OnnxConfig的LayoutLMv3OnnxConfig类

    # 设置torch_onnx_minimum_version属性为1.12
    torch_onnx_minimum_version = version.parse("1.12")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义inputs方法，返回输入的格式规范
        # 根据任务类型返回不同的输入格式
        if self.task in ["question-answering", "sequence-classification"]:
            return OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "sequence"}),
                    ("attention_mask", {0: "batch", 1: "sequence"}),
                    ("bbox", {0: "batch", 1: "sequence"}),
                    ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ]
            )
        else:
            return OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "sequence"}),
                    ("bbox", {0: "batch", 1: "sequence"}),
                    ("attention_mask", {0: "batch", 1: "sequence"}),
                    ("pixel_values", {0: "batch", 1: "num_channels"}),
                ]
            )

    @property
    def atol_for_validation(self) -> float:
        # 定义atol_for_validation属性，设置为1e-5
        return 1e-5

    @property
    def default_onnx_opset(self) -> int:
        # 定义default_onnx_opset属性，设置为12
        return 12

    def generate_dummy_inputs(
        self,
        processor: "ProcessorMixin",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
        """
        为特定框架生成给ONNX导出器的输入

        参数:
            processor ([`ProcessorMixin`]):
                与此模型配置关联的处理器。
            batch_size (`int`, *可选*, 默认为-1):
                导出模型所用的批量大小(-1表示动态轴)。
            seq_length (`int`, *可选*, 默认为-1):
                导出模型所用的序列长度(-1表示动态轴)。
            is_pair (`bool`, *可选*, 默认为 `False`):
                指示输入是否为一对(句子1，句子2)。
            framework (`TensorType`, *可选*, 默认为 `None`):
                处理器将为其生成张量的框架(PyTorch或TensorFlow)。
            num_channels (`int`, *可选*, 默认为 3):
                生成图像的通道数。
            image_width (`int`, *可选*, 默认为 40):
                生成图像的宽度。
            image_height (`int`, *可选*, 默认为 40):
                生成图像的高度。

        返回:
            Mapping[str, Any]: 包含提供给模型前向函数的参数
        """

        # 使用虚拟图像，不应用 OCR
        setattr(processor.image_processor, "apply_ocr", False)

        # 如果动态轴 (-1)，则向前传递固定维度的2个样本，以避免ONNX进行的优化
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )
        # 如果动态轴 (-1)，则向前传递固定维度的8个标记以避免ONNX进行的优化
        token_to_add = processor.tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )
        # 根据计算批量和序列生成虚拟输入
        dummy_text = [[" ".join([processor.tokenizer.unk_token]) * seq_length]] * batch_size

        # 生成虚拟边界框
        dummy_bboxes = [[[48, 84, 73, 128]]] * batch_size

        # 如果动态轴 (-1)，则向前传递固定维度的2个样本以避免ONNX进行的优化
        # batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
        dummy_image = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)

        inputs = dict(
            processor(
                dummy_image,
                text=dummy_text,
                boxes=dummy_bboxes,
                return_tensors=framework,
            )
        )

        return inputs
```