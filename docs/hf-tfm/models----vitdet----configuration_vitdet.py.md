# `.\models\vitdet\configuration_vitdet.py`

```
# 设置编码格式为 UTF-8
# 版权声明：2023 年 HuggingFace 公司保留所有权利
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发软件，不附带任何明示或暗示的担保或条件
# 请查阅许可证了解更多信息

""" VitDet 模型配置"""

# 从相对路径导入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 导入 logging 模块用于日志记录
from ...utils import logging
# 从 backbone_utils 中导入 BackboneConfigMixin 类和 get_aligned_output_features_output_indices 函数

from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# VitDet 预训练模型配置映射，指定了模型名和其配置文件的下载链接
VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/vit-det-base": "https://huggingface.co/facebook/vit-det-base/resolve/main/config.json",
}

# VitDetConfig 类，继承了 BackboneConfigMixin 和 PretrainedConfig
class VitDetConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是存储 [`VitDetModel`] 配置的类。它用于根据指定的参数实例化 VitDet 模型，定义模型架构。
    使用默认配置实例化一个配置对象将会生成类似于 VitDet [google/vitdet-base-patch16-224] 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。详细信息请参阅 [`PretrainedConfig`] 的文档。

    示例：

    ```python
    >>> from transformers import VitDetConfig, VitDetModel

    >>> # 初始化 VitDet 配置
    >>> configuration = VitDetConfig()

    >>> # 使用配置对象实例化一个模型（带有随机权重）
    >>> model = VitDetModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "vitdet"
    model_type = "vitdet"

    # VitDetConfig 的构造函数，定义了模型的各种配置参数
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        pretrain_image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        drop_path_rate=0.0,
        window_block_indices=[],
        residual_block_indices=[],
        use_absolute_position_embeddings=True,
        use_relative_position_embeddings=False,
        window_size=0,
        out_features=None,
        out_indices=None,
        **kwargs,
        ):
        # 调用父类的构造函数并传递所有关键字参数
        super().__init__(**kwargs)

        # 初始化模型的各种超参数
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.dropout_prob = dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.pretrain_image_size = pretrain_image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.window_block_indices = window_block_indices
        self.residual_block_indices = residual_block_indices
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.use_relative_position_embeddings = use_relative_position_embeddings
        self.window_size = window_size

        # 设定模型各阶段的名称，包括初始的"stem"和从"stage1"到"stageN"的隐藏层
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        
        # 调用函数获取对齐的输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
```