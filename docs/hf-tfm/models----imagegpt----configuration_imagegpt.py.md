# `.\models\imagegpt\configuration_imagegpt.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" OpenAI ImageGPT 配置"""

# 导入所需的模块
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional
# 导入预训练配置
from ...configuration_utils import PretrainedConfig
# 导入 ONNX 配置
from ...onnx import OnnxConfig
# 导入日志记录工具
from ...utils import logging

# 如果是类型检查，则导入 FeatureExtractionMixin 和 TensorType
if TYPE_CHECKING:
    from ... import FeatureExtractionMixin, TensorType

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射
IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/imagegpt-small": "",
    "openai/imagegpt-medium": "",
    "openai/imagegpt-large": "",
}

# ImageGPT 配置类，用于存储 ImageGPTModel 或 TFImageGPTModel 的配置
class ImageGPTConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`ImageGPTModel`] or a [`TFImageGPTModel`]. It is
    used to instantiate a GPT-2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ImageGPT
    [openai/imagegpt-small](https://huggingface.co/openai/imagegpt-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 512):
            # 定义 GPT-2 模型的词汇表大小，表示可以由 `inputs_ids` 传入时表示的不同标记数量
        n_positions (`int`, *optional*, defaults to 32*32):
            # 此模型可能使用的最大序列长度。通常设置为较大的值（例如，512、1024或2048）
        n_embd (`int`, *optional*, defaults to 512):
            # 嵌入和隐藏状态的维度
        n_layer (`int`, *optional*, defaults to 24):
            # Transformer 编码器中的隐藏层数量
        n_head (`int`, *optional*, defaults to 8):
            # Transformer 编码器中每个注意力层的注意力头数
        n_inner (`int`, *optional*, defaults to None):
            # 内部前馈层的维度。`None` 将设置为 4 倍的 n_embd
        activation_function (`str`, *optional*, defaults to `"quick_gelu"`):
            # 激活函数（可以是 src/transformers/activations.py 中定义的激活函数之一）。默认为 "quick_gelu"
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            # 嵌入、编码器和池化器中所有全连接层的 dropout 概率
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            # 嵌入的 dropout 比率
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            # 注意力的 dropout 比率
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            # 层归一化层中使用的 epsilon
        initializer_range (`float`, *optional*, defaults to 0.02):
            # 用于初始化所有权重矩阵的截断正态初始化器的标准差
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            # 通过除以 sqrt(hidden_size) 缩放注意力权重
        use_cache (`bool`, *optional*, defaults to `True`):
            # 模型是否应返回最后的键/值注意力（并非所有模型都使用）
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            # 是否额外按 `1 / layer_idx + 1` 缩放注意力权重
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            # 是否在计算注意力（点积）之前缩放键（K）并在使用混合精度训练时将注意力点积/softmax 上升级到 float()

    Example:

    ```python
    >>> from transformers import ImageGPTConfig, ImageGPTModel

    >>> # Initializing a ImageGPT configuration
    >>> configuration = ImageGPTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ImageGPTModel(configuration)
    # 访问模型配置
    configuration = model.config
    
    
    
    # 定义模型类型为"imagegpt"
    model_type = "imagegpt"
    # 在推断时需要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射表
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=512 + 1,  # 为开始句子标记(sos)添加一个
        n_positions=32 * 32,
        n_embd=512,
        n_layer=24,
        n_head=8,
        n_inner=None,
        activation_function="quick_gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        tie_word_embeddings=False,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):
        # 设置模型参数
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.tie_word_embeddings = tie_word_embeddings
    
        # 调用父类的初始化函数
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
class ImageGPTOnnxConfig(OnnxConfig):
    # 定义一个名为ImageGPTOnnxConfig的类，继承自OnnxConfig类

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义一个名为inputs的属性，返回一个有序字典
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
            ]
        )

    def generate_dummy_inputs(
        self,
        preprocessor: "FeatureExtractionMixin",
        batch_size: int = 1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
        num_channels: int = 3,
        image_width: int = 32,
        image_height: int = 32,
    ) -> Mapping[str, Any]:
        """
        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            preprocessor ([`PreTrainedTokenizerBase`] or [`FeatureExtractionMixin`]):
                The preprocessor associated with this model configuration.
            batch_size (`int`, *optional*, defaults to -1):
                The batch size to export the model for (-1 means dynamic axis).
            num_choices (`int`, *optional*, defaults to -1):
                The number of candidate answers provided for multiple choice task (-1 means dynamic axis).
            seq_length (`int`, *optional*, defaults to -1):
                The sequence length to export the model for (-1 means dynamic axis).
            is_pair (`bool`, *optional*, defaults to `False`):
                Indicate if the input is a pair (sentence 1, sentence 2)
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the tokenizer will generate tensors for.
            num_channels (`int`, *optional*, defaults to 3):
                The number of channels of the generated images.
            image_width (`int`, *optional*, defaults to 40):
                The width of the generated images.
            image_height (`int`, *optional*, defaults to 40):
                The height of the generated images.

        Returns:
            Mapping[str, Tensor] holding the kwargs to provide to the model's forward function
        """

        # 生成用于ONNX导出器的特定框架的输入
        input_image = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
        # 调用preprocessor函数处理生成的图像，返回处理后的结果
        inputs = dict(preprocessor(images=input_image, return_tensors=framework))

        return inputs
```