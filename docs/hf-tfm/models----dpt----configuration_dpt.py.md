# `.\models\dpt\configuration_dpt.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本(“许可证”)，除非符合许可证的规定，否则您不得使用此文件。您可以在以下获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则依据许可证分发的软件均基于“按原样”分发，没有任何明示或暗示的担保或条件。
# 请查看许可证以获取有关特定语言的权限和限制
"""DPT 模型配置"""

# 导入所需的模块和库
import copy
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING
from ..bit import BitConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置存档映射
DPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Intel/dpt-large": "https://huggingface.co/Intel/dpt-large/resolve/main/config.json",
    # 在 https://huggingface.co/models?filter=dpt 查看所有 DPT 模型
}

# DPT 模型配置类，用于存储 [`DPTModel`] 的配置
class DPTConfig(PretrainedConfig):
    r"""
    这是用于存储 [`DPTModel`] 配置的配置类。根据指定的参数实例化 DPT 模型，定义模型架构。使用默认值实例化配置将产生类似于 DPT [Intel/dpt-large](https://huggingface.co/Intel/dpt-large) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import DPTModel, DPTConfig

    >>> # 初始化 DPT dpt-large 风格的配置
    >>> configuration = DPTConfig()

    >>> # 从 dpt-large 风格的配置初始化模型
    >>> model = DPTModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""
    
    # 模型类型为 "dpt"
    model_type = "dpt"
    def __init__(
        self,
        hidden_size=768,  # 隐藏层的维度，默认为768
        num_hidden_layers=12,  # Transformer 模型中的隐藏层的数量，默认为12
        num_attention_heads=12,  # Transformer 模型中的注意力头的数量，默认为12
        intermediate_size=3072,  # Transformer 模型中间层的维度，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数的类型，默认为"GELU"
        hidden_dropout_prob=0.0,  # 隐藏层的dropout概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力机制的dropout概率，默认为0.0
        initializer_range=0.02,  # 权重初始化的范围，默认为0.02
        layer_norm_eps=1e-12,  # Layer normalization 中 epsilon 的值，默认为1e-12
        image_size=384,  # 输入图像的大小，默认为384
        patch_size=16,  # 图像分块的大小，默认为16
        num_channels=3,  # 输入图像的通道数，默认为3
        is_hybrid=False,  # 是否使用混合模式，默认为False
        qkv_bias=True,  # 是否在注意力机制中使用偏置项，默认为True
        backbone_out_indices=[2, 5, 8, 11],  # 输出层索引，默认为[2, 5, 8, 11]
        readout_type="project",  # 输出层类型，默认为"project"
        reassemble_factors=[4, 2, 1, 0.5],  # 重组因子，默认为[4, 2, 1, 0.5]
        neck_hidden_sizes=[96, 192, 384, 768],  # "neck"部分隐藏层的大小，默认为[96, 192, 384, 768]
        fusion_hidden_size=256,  # 融合层隐藏层的大小，默认为256
        head_in_index=-1,  # 在输入索引中的头部，默认为-1
        use_batch_norm_in_fusion_residual=False,  # 是否在融合残差中使用批标准化，默认为False
        use_bias_in_fusion_residual=None,  # 是否在融合残差中使用偏置项，默认为None
        add_projection=False,  # 是否添加投影，默认为False
        use_auxiliary_head=True,  # 是否使用辅助头部，默认为True
        auxiliary_loss_weight=0.4,  # 辅助损失的权重，默认为0.4
        semantic_loss_ignore_index=255,  # 语义损失中要忽略的索引，默认为255
        semantic_classifier_dropout=0.1,  # 语义分类器的dropout概率，默认为0.1
        backbone_featmap_shape=[1, 1024, 24, 24],  # 主干网络特征图的形状，默认为[1, 1024, 24, 24]
        neck_ignore_stages=[0, 1],  # 要忽略的 "neck" 阶段，默认为[0, 1]
        backbone_config=None,  # 主干网络配置，默认为None
        **kwargs,  # 其他关键字参数
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)  # 深度拷贝当前实例的字典表示

        if output["backbone_config"] is not None:  # 如果主干网络配置不为空
            output["backbone_config"] = self.backbone_config.to_dict()  # 将主干网络配置序列化为字典

        output["model_type"] = self.__class__.model_type  # 添加模型类型到输出字典
        return output  # 返回序列化后的字典
```