# `.\models\nystromformer\configuration_nystromformer.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，版权归属于2022年的UW-Madison和The HuggingFace Inc.团队，保留所有权利
#
# 根据Apache许可证版本2.0授权使用此文件；
# 除非遵守许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“原样”的基础分发，
# 没有任何明示或暗示的担保或条件。
# 有关更多信息，请参阅许可证。
""" Nystromformer模型配置"""

# 从configuration_utils导入PretrainedConfig类
# 从utils导入logging模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# 定义Nystromformer预训练模型配置文件映射字典
NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "uw-madison/nystromformer-512": "https://huggingface.co/uw-madison/nystromformer-512/resolve/main/config.json",
    # 查看所有Nystromformer模型的列表，网址为https://huggingface.co/models?filter=nystromformer
}

# 定义NystromformerConfig类，继承自PretrainedConfig类
class NystromformerConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储[`NystromformerModel`]的配置信息。它用于根据指定的参数实例化
    一个Nystromformer模型，定义模型的架构。使用默认参数实例化一个配置对象将生成与Nystromformer
    [uw-madison/nystromformer-512](https://huggingface.co/uw-madison/nystromformer-512)架构类似的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型的输出。有关更多信息，请参阅
    [`PretrainedConfig`]的文档。
    # 定义 NystromformerConfig 类，用于配置 Nystromformer 模型的参数
    class NystromformerConfig:
    
        # 初始化函数，设置 Nystromformer 模型的各种参数
        def __init__(
            self,
            vocab_size: int = 30000,  # Nystromformer 模型的词汇表大小，默认为 30000
            hidden_size: int = 768,  # 编码器层和池化层的维度，默认为 768
            num_hidden_layers: int = 12,  # Transformer 编码器中隐藏层的数量，默认为 12
            num_attention_heads: int = 12,  # 每个注意力层中的注意力头数，默认为 12
            intermediate_size: int = 3072,  # Transformer 编码器中"中间"（即前馈）层的维度，默认为 3072
            hidden_act: str = "gelu",  # 编码器和池化器中的非线性激活函数，默认为 "gelu"
            hidden_dropout_prob: float = 0.1,  # 嵌入层、编码器和池化器中全连接层的 dropout 概率，默认为 0.1
            attention_probs_dropout_prob: float = 0.1,  # 注意力概率的 dropout 比率，默认为 0.1
            max_position_embeddings: int = 512,  # 模型可能使用的最大序列长度，默认为 512
            type_vocab_size: int = 2,  # 调用 NystromformerModel 时传递的 token_type_ids 的词汇表大小，默认为 2
            segment_means_seq_len: int = 64,  # segment-means 中使用的序列长度，默认为 64
            num_landmarks: int = 64,  # Nystrom 近似 softmax 自注意力矩阵时使用的 landmark（或 Nystrom）点数量，默认为 64
            conv_kernel_size: int = 65,  # Nystrom 近似中使用的深度卷积的内核大小，默认为 65
            inv_coeff_init_option: bool = False,  # 是否使用精确系数计算来初始化 Moore-Penrose 矩阵的迭代方法的初始值，默认为 False
            initializer_range: float = 0.02,  # 用于初始化所有权重矩阵的截断正态初始化器的标准差，默认为 0.02
            layer_norm_eps: float = 1e-12,  # 层归一化层使用的 epsilon，默认为 1e-12
        ):
            # 将参数设置为类的属性
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.segment_means_seq_len = segment_means_seq_len
            self.num_landmarks = num_landmarks
            self.conv_kernel_size = conv_kernel_size
            self.inv_coeff_init_option = inv_coeff_init_option
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
    # 设置模型类型为 Nystromformer
    model_type = "nystromformer"
    
    # 定义一个初始化方法，初始化 NystromformerConfig 类的实例
    def __init__(
        self,
        vocab_size=30000,  # 设置词汇表大小，默认为 30000
        hidden_size=768,  # 设置隐藏层大小，默认为 768
        num_hidden_layers=12,  # 设置隐藏层数，默认为 12
        num_attention_heads=12,  # 设置注意力头数，默认为 12
        intermediate_size=3072,  # 设置中间层大小，默认为 3072
        hidden_act="gelu_new",  # 设置隐藏层激活函数，默认为 gelu_new
        hidden_dropout_prob=0.1,  # 设置隐藏层的 dropout 概率，默认为 0.1
        attention_probs_dropout_prob=0.1,  # 设置注意力概率的 dropout 概率，默认为 0.1
        max_position_embeddings=510,  # 设置最大位置编码长度，默认为 510
        type_vocab_size=2,  # 设置类型词汇表大小，默认为 2
        segment_means_seq_len=64,  # 设置段落均值序列长度，默认为 64
        num_landmarks=64,  # 设置地标数，默认为 64
        conv_kernel_size=65,  # 设置卷积核大小，默认为 65
        inv_coeff_init_option=False,  # 设置逆系数初始化选项，默认为 False
        initializer_range=0.02,  # 设置初始化范围，默认为 0.02
        layer_norm_eps=1e-5,  # 设置层归一化的 epsilon，默认为 1e-5
        pad_token_id=1,  # 设置填充标记 ID，默认为 1
        bos_token_id=0,  # 设置起始标记 ID，默认为 0
        eos_token_id=2,  # 设置结束标记 ID，默认为 2
        **kwargs,  # 其他可选参数
    ):
        # 调用父类的初始化方法，传入 pad_token_id, bos_token_id, eos_token_id 和其他参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        # 将传入的参数保存为对象的属性
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.segment_means_seq_len = segment_means_seq_len
        self.num_landmarks = num_landmarks
        self.conv_kernel_size = conv_kernel_size
        self.inv_coeff_init_option = inv_coeff_init_option
        self.layer_norm_eps = layer_norm_eps
```