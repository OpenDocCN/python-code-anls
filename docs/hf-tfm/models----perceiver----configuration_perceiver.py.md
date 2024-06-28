# `.\models\perceiver\configuration_perceiver.py`

```py
# coding=utf-8
# 声明文件编码格式为 UTF-8

# 导入必要的模块和类
from collections import OrderedDict  # 导入 OrderedDict 类，用于有序字典
from typing import Any, Mapping, Optional, Union  # 导入类型提示相关的类和方法

# 导入配置相关的类和函数
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...feature_extraction_utils import FeatureExtractionMixin  # 导入特征提取混合类
from ...onnx import OnnxConfig  # 导入 ONNX 配置类
from ...onnx.utils import compute_effective_axis_dimension  # 导入计算有效轴维度的方法
from ...tokenization_utils_base import PreTrainedTokenizerBase  # 导入预训练分词器基类
from ...utils import TensorType, logging  # 导入 TensorType 和 logging 工具

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射表，指定不同模型的配置文件链接
PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "deepmind/language-perceiver": "https://huggingface.co/deepmind/language-perceiver/resolve/main/config.json",
    # 可查看所有 Perceiver 模型列表：https://huggingface.co/models?filter=perceiver
}


class PerceiverConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PerceiverModel`]. It is used to instantiate an
    Perceiver model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Perceiver
    [deepmind/language-perceiver](https://huggingface.co/deepmind/language-perceiver) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```
    >>> from transformers import PerceiverModel, PerceiverConfig

    >>> # Initializing a Perceiver deepmind/language-perceiver style configuration
    >>> configuration = PerceiverConfig()

    >>> # Initializing a model from the deepmind/language-perceiver style configuration
    >>> model = PerceiverModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    model_type = "perceiver"
    # 定义一个初始化方法，用于设置模型的各种参数和属性
    def __init__(
        self,
        num_latents=256,  # Latent space dimensionality
        d_latents=1280,  # Dimensionality of latent vectors
        d_model=768,  # Dimensionality of the model
        num_blocks=1,  # Number of transformer blocks
        num_self_attends_per_block=26,  # Number of self-attention layers per block
        num_self_attention_heads=8,  # Number of self-attention heads
        num_cross_attention_heads=8,  # Number of cross-attention heads
        qk_channels=None,  # Query and key projection dimensionality
        v_channels=None,  # Value projection dimensionality
        cross_attention_shape_for_attention="kv",  # Shape for cross-attention computation
        self_attention_widening_factor=1,  # Self-attention widening factor
        cross_attention_widening_factor=1,  # Cross-attention widening factor
        hidden_act="gelu",  # Activation function for hidden layers
        attention_probs_dropout_prob=0.1,  # Dropout probability for attention weights
        initializer_range=0.02,  # Range for weight initialization
        layer_norm_eps=1e-12,  # Epsilon for layer normalization
        use_query_residual=True,  # Flag indicating whether to use query residual connections
        vocab_size=262,  # Size of vocabulary for masked language modeling
        max_position_embeddings=2048,  # Maximum number of positional embeddings
        image_size=56,  # Size of input images for image classification
        train_size=[368, 496],  # Size of training images
        num_frames=16,  # Number of frames in video input
        audio_samples_per_frame=1920,  # Audio samples per video frame
        samples_per_patch=16,  # Number of samples per image patch
        output_shape=[1, 16, 224, 224],  # Output shape of the model
        output_num_channels=512,  # Number of output channels
        _label_trainable_num_channels=1024,  # Number of channels for trainable labels
        **kwargs,  # Additional keyword arguments
    ):
        # 调用父类的初始化方法，传入额外的关键字参数
        super().__init__(**kwargs)
    
        # 初始化模型的各种参数和属性
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
        # 以下是针对不同任务的特定属性
    
        # Masked Language Modeling任务的属性
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
    
        # Image Classification任务的属性
        self.image_size = image_size
    
        # Flow任务的属性
        self.train_size = train_size
    
        # Multimodal Autoencoding任务的属性
        self.num_frames = num_frames
        self.audio_samples_per_frame = audio_samples_per_frame
        self.samples_per_patch = samples_per_patch
    
        # 输出的形状和通道数属性
        self.output_shape = output_shape
        self.output_num_channels = output_num_channels
    
        # 可训练标签的通道数属性
        self._label_trainable_num_channels = _label_trainable_num_channels
class PerceiverOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            # 如果任务为多选题，则定义动态轴的维度
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则定义动态轴的维度
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典，包含输入名称和对应的动态轴
        return OrderedDict(
            [
                ("inputs", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        # 返回用于验证的绝对容差值
        return 1e-4

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
        # 从`transformers.onnx.config.OnnxConfig`中复制并稍作修改和简化

        if isinstance(preprocessor, PreTrainedTokenizerBase):
            # 如果预处理器是预训练的分词器，则根据需要设置动态轴的维度
            batch_size = compute_effective_axis_dimension(
                batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
            )
            # 获取要添加的特殊标记的数量
            token_to_add = preprocessor.num_special_tokens_to_add(is_pair)
            # 根据需要设置动态轴的维度
            seq_length = compute_effective_axis_dimension(
                seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
            )
            # 根据计算的批次大小和序列长度生成虚拟输入
            dummy_input = [" ".join(["a"]) * seq_length] * batch_size
            # 使用预处理器生成输入字典，并将输入名称标准化为`input_ids`
            inputs = dict(preprocessor(dummy_input, return_tensors=framework))
            inputs["inputs"] = inputs.pop("input_ids")
            return inputs
        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == "pixel_values":
            # 如果预处理器是特征提取混合类，并且模型输入名称为`pixel_values`
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            # 根据指定的批次大小和图像尺寸生成虚拟图像数据
            dummy_input = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
            # 使用预处理器生成输入字典，并将输入名称标准化为`pixel_values`
            inputs = dict(preprocessor(images=dummy_input, return_tensors=framework))
            inputs["inputs"] = inputs.pop("pixel_values")
            return inputs
        else:
            # 如果无法为模型生成虚拟输入，则抛出值错误异常
            raise ValueError(
                "Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor."
            )
```