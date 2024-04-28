# `.\models\dinov2\configuration_dinov2.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 Version 2.0 使用本文件，详情请查看 http://www.apache.org/licenses/LICENSE-2.0
# 仅在符合许可证情况下使用本文件
# 请查看许可证获取许可证副本
# 除非适用法律需要或书面同意，否则不得使用此文件
# 根据许可证以“原样”分发文件，没有任何担保或条件，无论是明示或暗示的
# 请查看许可证获取详细的语言、授权权限和限制
""" DINOV2 模型配置"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取日志对象
logger = logging.get_logger(__name__)

# DINOV2 预训练配置文件映射
DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/dinov2-base": "https://huggingface.co/facebook/dinov2-base/resolve/main/config.json",
}

class Dinov2Config(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是 [`Dinov2Model`] 的配置类，用于存储 [`Dinov2Model`] 的配置。根据指定的参数实例化 Dinov2 模型的配置，定义模型架构。
    使用默认值实例化配置将产生与 Dinov2 [google/dinov2-base-patch16-224](https://huggingface.co/google/dinov2-base-patch16-224)
    架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    
    示例：
    
    ```python
    >>> from transformers import Dinov2Config, Dinov2Model

    >>> # 初始化一个 Dinov2 dinov2-base-patch16-224 风格的配置
    >>> configuration = Dinov2Config()

    >>> # 从 dinov2-base-patch16-224 风格的配置初始化一个模型（带有随机权重）
    >>> model = Dinov2Model(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 模型类型为 "dinov2"
    model_type = "dinov2"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        layerscale_value=1.0,
        drop_path_rate=0.0,
        use_swiglu_ffn=False,
        out_features=None,
        out_indices=None,
        apply_layernorm=True,
        reshape_hidden_states=True,
        **kwargs,
        # 调用父类的初始化方法，传入关键字参数
        super().__init__(**kwargs)

        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置MLP(多层感知器)的比例
        self.mlp_ratio = mlp_ratio
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置隐藏层的丢弃率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力权重的丢弃率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 设置图像大小
        self.image_size = image_size
        # 设置patch的大小
        self.patch_size = patch_size
        # 设置通道数量
        self.num_channels = num_channels
        # 设置QKV是否有偏置
        self.qkv_bias = qkv_bias
        # 设置层缩放值
        self.layerscale_value = layerscale_value
        # 设置丢弃路径率
        self.drop_path_rate = drop_path_rate
        # 设置是否使用SwiGLU作为FFN(Feed Forward Network)的激活函数
        self.use_swiglu_ffn = use_swiglu_ffn
        # 设置阶段名称列表，包括"stem"和"stage1"到"stage{num_hidden_layers}"，例如"stage2"、"stage3"等
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, num_hidden_layers + 1)]
        # 获取对齐的输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        # 是否应用层归一化
        self.apply_layernorm = apply_layernorm
        # 是否重塑隐藏状态
        self.reshape_hidden_states = reshape_hidden_states
# 创建一个名为Dinov2OnnxConfig的类，继承自OnnxConfig类
class Dinov2OnnxConfig(OnnxConfig):
    # 设置属性torch_onnx_minimum_version为"1.11"
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义一个名为inputs的属性方法，返回一个有序字典，表示输入的形状
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义一个名为atol_for_validation的属性方法，返回一个浮点数，表示验证的绝对容差
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```