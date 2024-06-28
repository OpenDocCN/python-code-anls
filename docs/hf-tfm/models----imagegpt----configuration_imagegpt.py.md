# `.\models\imagegpt\configuration_imagegpt.py`

```
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenAI ImageGPT configuration"""

# 导入必要的模块和类
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional

# 导入配置工具和ONNX配置
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
# 导入日志工具
from ...utils import logging

# 如果是类型检查，导入特定模块
if TYPE_CHECKING:
    from ... import FeatureExtractionMixin, TensorType

# 获取logger对象
logger = logging.get_logger(__name__)

# 预训练模型配置映射字典
IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/imagegpt-small": "",
    "openai/imagegpt-medium": "",
    "openai/imagegpt-large": "",
}


class ImageGPTConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`ImageGPTModel`] or a [`TFImageGPTModel`]. It is
    used to instantiate a GPT-2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ImageGPT
    [openai/imagegpt-small](https://huggingface.co/openai/imagegpt-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义 ImageGPT 模型的配置类 ImageGPTConfig，用于设置模型参数
    Args:
        vocab_size (`int`, *optional*, defaults to 512):
            GPT-2 模型的词汇表大小，定义了可以由 `inputs_ids` 表示的不同标记数量。
        n_positions (`int`, *optional*, defaults to 32*32):
            模型可能使用的最大序列长度。通常设置为一个较大的值（例如 512、1024 或 2048）。
        n_embd (`int`, *optional*, defaults to 512):
            嵌入和隐藏状态的维度。
        n_layer (`int`, *optional*, defaults to 24):
            Transformer 编码器中的隐藏层数。
        n_head (`int`, *optional*, defaults to 8):
            Transformer 编码器中每个注意力层的注意头数。
        n_inner (`int`, *optional*, defaults to None):
            内部前馈层的维度。如果为 `None`，将设置为 4 倍的 n_embd。
        activation_function (`str`, *optional*, defaults to `"quick_gelu"`):
            激活函数（可以是 src/transformers/activations.py 中定义的激活函数之一）。默认为 "quick_gelu"。
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            嵌入层、编码器和池化器中所有全连接层的 dropout 概率。
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            嵌入层的 dropout 比率。
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            注意力机制的 dropout 比率。
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            层归一化层使用的 epsilon。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            是否通过除以 sqrt(hidden_size) 缩放注意力权重。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后一次的键/值注意力（不是所有模型都使用）。
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            是否额外按 `1 / layer_idx + 1` 缩放注意力权重。
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            是否在计算注意力（点积）之前缩放键（K）并在训练时将注意力点积/softmax 升级为 float()（用于混合精度）。

    Example:

    ```python
    >>> from transformers import ImageGPTConfig, ImageGPTModel

    >>> # 初始化一个 ImageGPT 配置对象
    >>> configuration = ImageGPTConfig()

    >>> # 使用配置对象初始化一个模型（带有随机权重）
    >>> model = ImageGPTModel(configuration)
    ```
    # 定义模型类型为"imagegpt"
    model_type = "imagegpt"
    # 在推理阶段需要忽略的关键字列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，将模型配置的名称映射到内部属性名称
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    # 初始化方法，设置模型各种配置参数
    def __init__(
        self,
        vocab_size=512 + 1,  # 词汇表大小，默认为512（加一是为了起始标记的token）
        n_positions=32 * 32,  # 最大位置编码，默认为32*32
        n_embd=512,  # 隐藏单元的维度，默认为512
        n_layer=24,  # 隐藏层的数量，默认为24
        n_head=8,  # 注意力头的数量，默认为8
        n_inner=None,  # 内部隐藏层的维度，默认为None
        activation_function="quick_gelu",  # 激活函数，默认为"quick_gelu"
        resid_pdrop=0.1,  # 残差连接中的dropout概率，默认为0.1
        embd_pdrop=0.1,  # 嵌入层的dropout概率，默认为0.1
        attn_pdrop=0.1,  # 注意力层的dropout概率，默认为0.1
        layer_norm_epsilon=1e-5,  # 层归一化中epsilon的值，默认为1e-5
        initializer_range=0.02,  # 初始化范围，默认为0.02
        scale_attn_weights=True,  # 是否缩放注意力权重，默认为True
        use_cache=True,  # 是否使用缓存，默认为True
        tie_word_embeddings=False,  # 是否绑定词嵌入，默认为False
        scale_attn_by_inverse_layer_idx=False,  # 是否通过逆层索引缩放注意力，默认为False
        reorder_and_upcast_attn=False,  # 是否重排序和上升注意力，默认为False
        **kwargs,  # 其他关键字参数
    ):
        # 初始化各个配置参数
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

        # 调用父类初始化方法，传递绑定词嵌入的参数和其他关键字参数
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
# 定义一个继承自OnnxConfig的ImageGPTOnnxConfig类，用于配置基于ONNX的图像生成模型
class ImageGPTOnnxConfig(OnnxConfig):

    # 定义一个属性方法inputs，返回一个有序字典，描述了模型的输入信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
            ]
        )

    # 定义一个生成虚拟输入数据的方法generate_dummy_inputs
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
                与此模型配置相关联的预处理器。
            batch_size (`int`, *optional*, defaults to -1):
                导出模型的批处理大小（-1表示动态轴）。
            num_choices (`int`, *optional*, defaults to -1):
                多选任务提供的候选答案数量（-1表示动态轴）。
            seq_length (`int`, *optional*, defaults to -1):
                导出模型的序列长度（-1表示动态轴）。
            is_pair (`bool`, *optional*, defaults to `False`):
                指示输入是否为一对（句子1，句子2）。
            framework (`TensorType`, *optional*, defaults to `None`):
                预处理器将为其生成张量的框架（PyTorch或TensorFlow）。
            num_channels (`int`, *optional*, defaults to 3):
                生成图像的通道数。
            image_width (`int`, *optional*, defaults to 40):
                生成图像的宽度。
            image_height (`int`, *optional*, defaults to 40):
                生成图像的高度。

        Returns:
            Mapping[str, Tensor]：包含提供给模型前向函数的关键字参数
        """

        # 使用内部方法_generate_dummy_images生成虚拟输入图像数据
        input_image = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
        
        # 调用预处理器preprocessor，传递生成的图像数据input_image，并根据framework返回张量
        inputs = dict(preprocessor(images=input_image, return_tensors=framework))

        # 返回输入参数字典
        return inputs
```