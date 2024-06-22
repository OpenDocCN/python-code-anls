# `.\models\data2vec\configuration_data2vec_audio.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的规定，否则您不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发
# 没有任何种类的明示或暗示的担保或条件
# 请参阅许可证以了解特定语言的权限和限制

""" Data2VecText configuration"""

# 导入需要的库
import math
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# Data2Vec 音频预训练配置归档映射
DATA2VEC_AUDIO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/data2vec-base-960h": "https://huggingface.co/facebook/data2vec-audio-base-960h/resolve/main/config.json",
    # 查看所有 Data2Vec 音频模型 https://huggingface.co/models?filter=data2vec-audio
}

# Data2Vec 音频配置类，用于存储 Data2VecAudioModel 的配置信息
class Data2VecAudioConfig(PretrainedConfig):
    r"""
    这是用于存储 [`Data2VecAudioModel`] 配置的配置类。它用于根据指定的参数实例化 Data2VecAudio 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 Data2VecAudio [facebook/data2vec-audio-base-960h]
    (https://huggingface.co/facebook/data2vec-audio-base-960h) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 中的文档，了解更多信息。

    示例:

    ```python
    >>> from transformers import Data2VecAudioConfig, Data2VecAudioModel

    >>> # 初始化一个 Data2VecAudio facebook/data2vec-audio-base-960h 风格的配置
    >>> configuration = Data2VecAudioConfig()

    >>> # 从 facebook/data2vec-audio-base-960h 风格的配置初始化一个（具有随机权重的）模型
    >>> model = Data2VecAudioModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""
    # 模型类型
    model_type = "data2vec-audio"
    # 定义一个类，初始化各种参数
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层的数量，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout=0.1,  # 隐藏层的dropout比例，默认为0.1
        activation_dropout=0.1,  # 激活层的dropout比例，默认为0.1
        attention_dropout=0.1,  # 注意力层的dropout比例，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影的dropout比例，默认为0.0
        final_dropout=0.1,  # 最终输出层的dropout比例，默认为0.1
        layerdrop=0.1,  # 层删除的比例，默认为0.1
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的epsilon，默认为1e-5
        feat_extract_activation="gelu",  # 特征提取激活函数，默认为gelu
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积维度，默认为(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积步长，默认为(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积核大小，默认为(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 是否使用卷积偏置，默认为False
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入组数，默认为16
        conv_pos_kernel_size=19,  # 卷积位置核大小，默认为19
        num_conv_pos_embeddings=5,  # 卷积位置嵌入数量，默认为5
        mask_time_prob=0.05,  # 时间掩码的概率，默认为0.05
        mask_time_length=10,  # 时间掩码长度，默认为10
        mask_time_min_masks=2,  # 最小时间掩码数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码的概率，默认为0.0
        mask_feature_length=10,  # 特征掩码长度，默认为10
        mask_feature_min_masks=0,  # 最小特征掩码数量，默认为0
        ctc_loss_reduction="sum",  # CTC损失的缩减方式，默认为"sum"
        ctc_zero_infinity=False,  # CTC是否将无穷大设为0，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影大小，默认为256
        tdnn_dim=(512, 512, 512, 512, 1500),  # TDNN维度，默认为(512, 512, 512, 512, 1500)
        tdnn_kernel=(5, 3, 3, 1, 1),  # TDNN核大小，默认为(5, 3, 3, 1, 1)
        tdnn_dilation=(1, 2, 3, 1, 1),  # TDNN膨胀率，默认为(1, 2, 3, 1, 1)
        xvector_output_dim=512,  # x-vector输出维度，默认为512
        pad_token_id=0,  # 填充标记ID，默认为0
        bos_token_id=1,  # 开始标记ID，默认为1
        eos_token_id=2,  # 结束标记ID，默认为2
        add_adapter=False,  # 是否添加适配器，默认为False
        adapter_kernel_size=3,  # 适配器核大小，默认为3
        adapter_stride=2,  # 适配器步长，默认为2
        num_adapter_layers=3,  # 适配器层数，默认为3
        output_hidden_size=None,  # 输出隐藏层大小，默认为None
        **kwargs,  # 其他关键字参数
    @property
    def inputs_to_logits_ratio(self):
        return math.prod(self.conv_stride)  # 返回卷积步长的乘积
```