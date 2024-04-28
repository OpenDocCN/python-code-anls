# `.\transformers\models\sew_d\configuration_sew_d.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 ASAPP Inc. 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" SEW-D 模型配置"""

# 导入必要的库
import functools
import operator

# 从 Transformers 库中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 Transformers 库中导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# SEW-D 预训练配置文件映射
SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "asapp/sew-d-tiny-100k": "https://huggingface.co/asapp/sew-d-tiny-100k/resolve/main/config.json",
    # 查看所有 SEW-D 模型：https://huggingface.co/models?filter=sew-d
}

# SEW-D 配置类，继承自预训练配置类 PretrainedConfig
class SEWDConfig(PretrainedConfig):
    r"""
    这是用于存储 [`SEWDModel`] 配置的配置类。根据指定的参数实例化 SEW-D 模型，定义模型架构。使用默认值实例化配置将产生类似于 SEW-D
    [asapp/sew-d-tiny-100k](https://huggingface.co/asapp/sew-d-tiny-100k) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import SEWDConfig, SEWDModel

    >>> # 初始化一个 SEW-D asapp/sew-d-tiny-100k 风格的配置
    >>> configuration = SEWDConfig()

    >>> # 从 asapp/sew-d-tiny-100k 风格的配置初始化一个模型（带有随机权重）
    >>> model = SEWDModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""
    
    # 模型类型为 "sew-d"
    model_type = "sew-d"
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        squeeze_factor=2,  # 压缩因子，默认为2
        max_position_embeddings=512,  # 最大位置嵌入数，默认为512
        position_buckets=256,  # 位置桶数，默认为256
        share_att_key=True,  # 是否共享注意力键，默认为True
        relative_attention=True,  # 是否使用相对注意力，默认为True
        pos_att_type=("p2c", "c2p"),  # 位置注意力类型，默认为("p2c", "c2p")
        norm_rel_ebd="layer_norm",  # 相对嵌入规范化方式，默认为"layer_norm"
        hidden_act="gelu_python",  # 隐藏层激活函数，默认为"gelu_python"
        hidden_dropout=0.1,  # 隐藏层丢弃率，默认为0.1
        activation_dropout=0.1,  # 激活函数丢弃率，默认为0.1
        attention_dropout=0.1,  # 注意力丢弃率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影丢弃率，默认为0.0
        final_dropout=0.1,  # 最终丢弃率，默认为0.1
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-7,  # 层规范化参数，默认为1e-7
        feature_layer_norm_eps=1e-5,  # 特征层规范化参数，默认为1e-5
        feat_extract_norm="group",  # 特征提取规范化方式，默认为"group"
        feat_extract_activation="gelu",  # 特征提取激活函数，默认为"gelu"
        conv_dim=(64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512),  # 卷积维度，默认为指定值
        conv_stride=(5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1),  # 卷积步长，默认为指定值
        conv_kernel=(10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1),  # 卷积核大小，默认为指定值
        conv_bias=False,  # 是否使用卷积偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入数，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入分组数，默认为16
        apply_spec_augment=True,  # 是否应用特殊增强，默认为True
        mask_time_prob=0.05,  # 时间掩码概率，默认为0.05
        mask_time_length=10,  # 时间掩码长度，默认为10
        mask_time_min_masks=2,  # 最小时间掩码数，默认为2
        mask_feature_prob=0.0,  # 特征掩码概率，默认为0.0
        mask_feature_length=10,  # 特征掩码长度，默认为10
        mask_feature_min_masks=0,  # 最小特征掩码数，默认为0
        ctc_loss_reduction="mean",  # CTC损失减少方式，默认为"mean"
        ctc_zero_infinity=False,  # CTC零无穷，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影大小，默认为256
        pad_token_id=0,  # 填充标记ID，默认为0
        bos_token_id=1,  # 起始标记ID，默认为1
        eos_token_id=2,  # 结束标记ID，默认为2
        **kwargs,  # 其他参数
    @property
    def inputs_to_logits_ratio(self):
        # 计算输入到logits比率
        return functools.reduce(operator.mul, self.conv_stride, 1)

    @property
    def hidden_dropout(self):
        # 警告：hidden_dropout在模型中未使用，并将在v4.35中作为配置属性移除
        logger.warning_once("hidden_dropout is not used by the model and will be removed as config attribute in v4.35")
        return self._hidden_dropout

    def to_dict(self):
        """
        将此实例序列化为Python字典。
        """
        output = super().to_dict()
        output["hidden_dropout"] = output.pop("_hidden_dropout")
        return output
```