# `.\transformers\models\wav2vec2\configuration_wav2vec2.py`

```
# 设置代码文件的编码格式为 utf-8
# 版权声明
# 基于 Apache 许可证 2.0 进行许可
# 只有在符合许可证的情况下才能使用该文件
# 可以在下面链接获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得分发本软件
# 分发的软件是基于“按原样”基础分发的，没有任何形式的明示或暗示的担保或条件
# 请查看特定语言版本的许可证以了解权限和限制

""" Wav2Vec2 model配置"""

# 导入必要的库和模块
import functools
import operator
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志对象
logger = logging.get_logger(__name__)

# 预训练模型的配置映射
WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/wav2vec2-base-960h": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json",
    # 在 https://huggingface.co/models?filter=wav2vec2 查看所有的 Wav2Vec2 模型
}

# Wav2Vec2模型的配置类，用于存储Wav2Vec2Model的配置
class Wav2Vec2Config(PretrainedConfig):
    r"""
    这是用于存储[`Wav2Vec2Model`]的配置的配置类。它用于根据指定的参数实例化一个Wav2Vec2模型，定义模型的架构。使用默认值实例化配置将产生类似于Wav2Vec2 [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) 架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。更多信息，请阅读[`PretrainedConfig`]的文档。

    示例:

    ```python
    >>> from transformers import Wav2Vec2Config, Wav2Vec2Model

    >>> # 初始化一个Wav2Vec2 facebook/wav2vec2-base-960h风格的配置
    >>> configuration = Wav2Vec2Config()

    >>> # 初始化一个模型（具有随机权重）使用facebook/wav2vec2-base-960h风格的配置
    >>> model = Wav2Vec2Model(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 模型类型为 "wav2vec2"
    model_type = "wav2vec2"
    # 初始化方法，用于设置模型的各种参数
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout=0.1,  # 隐藏层dropout率，默认为0.1
        activation_dropout=0.1,  # 激活函数dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力dropout率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影dropout率，默认为0.0
        feat_quantizer_dropout=0.0,  # 特征量化dropout率，默认为0.0
        final_dropout=0.1,  # 最终dropout率，默认为0.1
        layerdrop=0.1,  # 层dropout率，默认为0.1
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的epsilon，默认为1e-5
        feat_extract_norm="group",  # 特征提取层归一化方式，默认为group
        feat_extract_activation="gelu",  # 特征提取激活函数，默认为GELU
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积维度，默认为(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积步幅，默认为(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积核大小，默认为(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 是否使用卷积偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入分组数量，默认为16
        do_stable_layer_norm=False,  # 是否进行稳定的层归一化，默认为False
        apply_spec_augment=True,  # 是否应用特征增强，默认为True
        mask_time_prob=0.05,  # 时间掩码概率，默认为0.05
        mask_time_length=10,  # 时间掩码长度，默认为10
        mask_time_min_masks=2,  # 时间掩码最小数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码概率，默认为0.0
        mask_feature_length=10,  # 特征掩码长度，默认为10
        mask_feature_min_masks=0,  # 特征掩码最小数量，默认为0
        num_codevectors_per_group=320,  # 每个编码向量组的编码向量数量，默认为320
        num_codevector_groups=2,  # 编码向量组数量，默认为2
        contrastive_logits_temperature=0.1,  # 对比日志温度，默认为0.1
        num_negatives=100,  # 负样本数量，默认为100
        codevector_dim=256,  # 编码向量维度，默认为256
        proj_codevector_dim=256,  # 投影编码向量维度，默认为256
        diversity_loss_weight=0.1,  # 多样性损失权重，默认为0.1
        ctc_loss_reduction="sum",  # CTC损失减少方式，默认为"sum"
        ctc_zero_infinity=False,  # CTC是否将无限置零，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层和，默认为False
        classifier_proj_size=256,  # 分类器投影大小，默认为256
        tdnn_dim=(512, 512, 512, 512, 1500),  # TDNN维度，默认为(512, 512, 512, 512, 1500)
        tdnn_kernel=(5, 3, 3, 1, 1),  # TDNN卷积核大小，默认为(5, 3, 3, 1, 1)
        tdnn_dilation=(1, 2, 3, 1, 1),  # TDNN膨胀率，默认为(1, 2, 3, 1, 1)
        xvector_output_dim=512,  # X向量输出维度，默认为512
        pad_token_id=0,  # 填充标记ID，默认为0
        bos_token_id=1,  # 开始标记ID，默认为1
        eos_token_id=2,  # 结束标记ID，默认为2
        add_adapter=False,  # 是否添加适配器，默认为False
        adapter_kernel_size=3,  # 适配器卷积核大小，默认为3
        adapter_stride=2,  # 适配器步幅，默认为2
        num_adapter_layers=3,  # 适配器层数，默认为3
        output_hidden_size=None,  # 输出层隐藏大小，默认为None
        adapter_attn_dim=None,  # 适配器注意力维度，默认为None
        **kwargs,
    @property
    # 计算输入与日志比率
    def inputs_to_logits_ratio(self):
        # 使用functools.reduce函数将卷积步幅中的所有值相乘，得到输入与日志比率
        return functools.reduce(operator.mul, self.conv_stride, 1)
```