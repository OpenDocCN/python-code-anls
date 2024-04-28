# `.\transformers\models\perceiver\configuration_perceiver.py`

```
# 导入必要的模块和类
from collections import OrderedDict
from typing import Any, Mapping, Optional, Union

# 导入配置的基类
from ...configuration_utils import PretrainedConfig
# 导入特征提取混合类，用于特征提取
from ...feature_extraction_utils import FeatureExtractionMixin
# 导入 ONNX 配置
from ...onnx import OnnxConfig
# 导入 ONNX 实用工具，用于计算有效轴维度
from ...onnx.utils import compute_effective_axis_dimension
# 导入基础分词器基类
from ...tokenization_utils_base import PreTrainedTokenizerBase
# 导入日志记录工具
from ...utils import TensorType, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型的配置文件映射
PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "deepmind/language-perceiver": "https://huggingface.co/deepmind/language-perceiver/resolve/main/config.json",
    # 查看所有 Perceiver 模型 https://huggingface.co/models?filter=perceiver
}

# Perceiver 模型的配置类，继承自预训练配置基类
class PerceiverConfig(PretrainedConfig):
    r"""
    这是用于存储 [`PerceiverModel`] 配置的配置类。它用于根据指定的参数实例化 Perceiver 模型，定义模型架构。使用默认值实例化配置将产生类似于 Perceiver [deepmind/language-perceiver](https://huggingface.co/deepmind/language-perceiver) 架构的配置。

    配置对象继承自 [`PretrainedConfig`] 并可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import PerceiverModel, PerceiverConfig

    >>> # 初始化 deepmind/language-perceiver 风格的配置
    >>> configuration = PerceiverConfig()

    >>> # 从 deepmind/language-perceiver 风格的配置初始化模型
    >>> model = PerceiverModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""
    # 模型类型标识符
    model_type = "perceiver"
    # 初始化函数，设置各种参数的默认数值
    def __init__(
        self,
        num_latents=256,  # 潜在变量的数量，默认为256
        d_latents=1280,  # 潜在变量的维度，默认为1280
        d_model=768,  # 模型的维度，默认为768
        num_blocks=1,  # 模型的块数，默认为1
        num_self_attends_per_block=26,  # 每个块中的自注意力数量，默认为26
        num_self_attention_heads=8,  # 自注意力头部的数量，默认为8
        num_cross_attention_heads=8,  # 交叉注意力头部的数量，默认为8
        qk_channels=None,  # qk通道数，默认为空
        v_channels=None,  # v通道数，默认为空
        cross_attention_shape_for_attention="kv",  # 交叉注意力的形状，默认为"kv"
        self_attention_widening_factor=1,  # 自注意力扩展因子，默认为1
        cross_attention_widening_factor=1,  # 交叉注意力扩展因子，默认为1
        hidden_act="gelu",  # 隐藏激活函数，默认为"gelu"
        attention_probs_dropout_prob=0.1,  # 注意力概率的dropout比例，默认为0.1
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化的epsilon，默认为1e-12
        use_query_residual=True,  # 使用查询残差，默认为True
        vocab_size=262,  # 词汇表大小，默认为262
        max_position_embeddings=2048,  # 最大位置嵌入数，默认为2048
        image_size=56,  # 图像大小，默认为56
        train_size=[368, 496],  # 训练大小，默认为[368, 496]
        num_frames=16,  # 帧数，默认为16
        audio_samples_per_frame=1920,  # 每帧音频采样数，默认为1920
        samples_per_patch=16,  # 每个补丁的采样数，默认为16
        output_shape=[1, 16, 224, 224],  # 输出形状，默认为[1, 16, 224, 224]
        output_num_channels=512,  # 输出通道数，默认为512
        _label_trainable_num_channels=1024,  # 可训练通道数，默认为1024
        **kwargs,  # 接收关键字参数
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 初始化各个参数的值
        self.num_latents = num_latents
        self.d_latents = d_latents
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.cross_attention_shape_for_attention = cross_attention_shape_for_attention
        self.self_attention_widening_factor = self_attention_widening_factor
        self.cross_attention_widening_factor = cross_attention_widening_factor
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_query_residual = use_query_residual
        # masked language modeling attributes
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        # image classification attributes
        self.image_size = image_size
        # flow attributes
        self.train_size = train_size
        # multimodal autoencoding attributes
        self.num_frames = num_frames
        self.audio_samples_per_frame = audio_samples_per_frame
        self.samples_per_patch = samples_per_patch
        self.output_shape = output_shape
        self.output_num_channels = output_num_channels
        self._label_trainable_num_channels = _label_trainable_num_channels
# 定义一个名为 PerceiverOnnxConfig 的类，它继承自 OnnxConfig 类
class PerceiverOnnxConfig(OnnxConfig):
    
    # 定义一个名为 inputs 的属性，返回类型为 Mapping[str, Mapping[int, str]]
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多选，设定动态轴的值为 {0: "batch", 1: "choice", 2: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 如果任务不是多选，设定动态轴的值为 {0: "batch", 1: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含键值为 "inputs" 和 "attention_mask"，值为 dynamic_axis
        return OrderedDict(
            [
                ("inputs", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )

    # 定义一个名为 atol_for_validation 的属性，返回类型为 float
    @property
    def atol_for_validation(self) -> float:
        # 返回一个固定的值 1e-4
        return 1e-4

    # 定义一个名为 generate_dummy_inputs 的方法，返回类型为 Mapping[str, Any]
    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
        batch_size: int = -1,
        seq_length: int = -1,
        num_choices: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
    ) -> Mapping[str, Any]:
        # 从 `transformers.onnx.config.OnnxConfig` 复制并稍微改动/简化

        # 如果 preprocessor 是 PreTrainedTokenizerBase 的实例
        if isinstance(preprocessor, PreTrainedTokenizerBase):
            # 计算有效轴维度，避免 ONNX 所做的优化
            batch_size = compute_effective_axis_dimension(
                batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
            )
            # 计算有效轴维度，避免 ONNX 所做的优化
            token_to_add = preprocessor.num_special_tokens_to_add(is_pair)
            seq_length = compute_effective_axis_dimension(
                seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
            )
            # 生成根据计算出的 batch 和 sequence 的虚拟输入
            dummy_input = [" ".join(["a"]) * seq_length] * batch_size
            # 调用 preprocessor 生成虚拟输入，并转换成指定类型的数据
            inputs = dict(preprocessor(dummy_input, return_tensors=framework))
            # 将键为 "input_ids" 的值改为 "inputs"，并���回
            inputs["inputs"] = inputs.pop("input_ids")
            return inputs
        # 如果 preprocessor 是 FeatureExtractionMixin 的实例，并且其 model_input_names 的第一个值是 "pixel_values"
        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == "pixel_values":
            # 计算有效轴维度，避免 ONNX 所做的优化
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            # 生成根据计算出的 batch、image_height 和 image_width 的虚拟图片输入
            dummy_input = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
            # 调用 preprocessor 生成虚拟图片输入，并转换成指定类型的数据
            inputs = dict(preprocessor(images=dummy_input, return_tensors=framework))
            # 将键为 "pixel_values" 的值改为 "inputs"，并返回
            inputs["inputs"] = inputs.pop("pixel_values")
            return inputs
        # 如果 preprocessor 不是 PreTrainedTokenizerBase 或 FeatureExtractionMixin 的实例，抛出 ValueError
        else:
            raise ValueError(
                "Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor."
            )
``` 
```