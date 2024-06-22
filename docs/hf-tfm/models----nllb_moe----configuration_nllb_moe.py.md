# `.\transformers\models\nllb_moe\configuration_nllb_moe.py`

```py
# 设置文件的编码格式为 utf-8
# 版权所有 2023 年，HuggingFace 公司
#
# 根据 Apache 许可证版本 2.0 授权
# 除非符合许可协议，否则不得使用此文件
# 您可以在以下链接处获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据“现况”基础分发软件
# 没有任何形式的保证或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
""" NLLB-MoE 模型配置"""
# 导入所需的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射
NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/nllb-moe-54B": "https://huggingface.co/facebook/nllb-moe-54b/resolve/main/config.json",
}

# NLLB-MoE 配置类，用于存储 NLLB-MoE 模型的配置
class NllbMoeConfig(PretrainedConfig):
    r"""
    这是存储[`NllbMoeModel`]配置的配置类。根据指定的参数来实例化一个 NLLB-MoE 模型，
    定义模型架构。使用默认值实例化配置将产生与 NLLB-MoE [facebook/nllb-moe-54b] 
    (https://huggingface.co/facebook/nllb-moe-54b)架构相似的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读
    [`PretrainedConfig`] 的文档。

    示例：

    ```python
    >>> from transformers import NllbMoeModel, NllbMoeConfig

    >>> # 初始化一个 NLLB-MoE facebook/nllb-moe-54b 风格的配置
    >>> configuration = NllbMoeConfig()

    >>> # 使用 facebook/nllb-moe-54b 风格的配置初始化模型
    >>> model = NllbMoeModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    # 模型类型
    model_type = "nllb-moe"
    # 推断阶段要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 初始化函数，用于初始化模型的参数
    def __init__(
        # 词汇表大小，默认为128112
        self,
        vocab_size=128112,
        # 最大位置编码数，默认为1024
        max_position_embeddings=1024,
        # 编码器层数，默认为12
        encoder_layers=12,
        # 编码器中前馈网络的维度，默认为4096
        encoder_ffn_dim=4096,
        # 编码器注意力头数，默认为16
        encoder_attention_heads=16,
        # 解码器层数，默认为12
        decoder_layers=12,
        # 解码器中前馈网络的维度，默认为4096
        decoder_ffn_dim=4096,
        # 解码器注意力头数，默认为16
        decoder_attention_heads=16,
        # 编码器层的丢弃率，默认为0.05
        encoder_layerdrop=0.05,
        # 解码器层的丢弃率，默认为0.05
        decoder_layerdrop=0.05,
        # 是否使用缓存，默认为True
        use_cache=True,
        # 是否是编码器-解码器模型，默认为True
        is_encoder_decoder=True,
        # 激活函数，默认为"relu"
        activation_function="relu",
        # 模型维度，默认为1024
        d_model=1024,
        # 全连接层的丢弃率，默认为0.1
        dropout=0.1,
        # 注意力机制的丢弃率，默认为0.1
        attention_dropout=0.1,
        # 激活函数的丢弃率，默认为0.0
        activation_dropout=0.0,
        # 初始化标准差，默认为0.02
        init_std=0.02,
        # 解码器起始令牌的ID，默认为2
        decoder_start_token_id=2,
        # 是否缩放嵌入，默认为True
        scale_embedding=True,
        # 路由器偏置，默认为False
        router_bias=False,
        # 路由器数据类型，默认为"float32"
        router_dtype="float32",
        # 是否忽略填充令牌，默认为False
        router_ignore_padding_tokens=False,
        # 专家数量，默认为128
        num_experts=128,
        # 专家容量，默认为64
        expert_capacity=64,
        # 编码器稀疏步数，默认为4
        encoder_sparse_step=4,
        # 解码器稀疏步数，默认为4
        decoder_sparse_step=4,
        # 路由器Z损失系数，默认为0.001
        router_z_loss_coef=0.001,
        # 路由器辅助损失系数，默认为0.001
        router_aux_loss_coef=0.001,
        # 第二专家策略，默认为"all"
        second_expert_policy="all",
        # 在丢弃之前是否归一化路由器概率，默认为False
        normalize_router_prob_before_dropping=False,
        # 是否批次优先的路由，默认为False
        batch_prioritized_routing=False,
        # MOE评估容量令牌比例，默认为1.0
        moe_eval_capacity_token_fraction=1.0,
        # MOE令牌丢弃率，默认为0.2
        moe_token_dropout=0.2,
        # 填充令牌ID，默认为1
        pad_token_id=1,
        # 起始令牌ID，默认为0
        bos_token_id=0,
        # 结束令牌ID，默认为2
        eos_token_id=2,
        # 输出路由器logits，默认为False
        output_router_logits=False,
        # **kwargs，接收其它关键字参数
        **kwargs,
        self.vocab_size = vocab_size  # 词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码数
        self.d_model = d_model  # 模型维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 编码器前馈网络维度
        self.encoder_layers = encoder_layers  # 编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 编码器注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim  # 解码器前馈网络维度
        self.decoder_layers = decoder_layers  # 解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 解码器注意力头数
        self.dropout = dropout  # 通用dropout
        self.attention_dropout = attention_dropout  # 注意力dropout
        self.activation_dropout = activation_dropout  # 激活函数dropout
        self.activation_function = activation_function  # 激活函数类型
        self.init_std = init_std  # 初始化标准差
        self.encoder_layerdrop = encoder_layerdrop  # 编码器层丢弃概率
        self.decoder_layerdrop = decoder_layerdrop  # 解码器层丢弃概率
        self.use_cache = use_cache  # 是否使用缓存
        self.num_hidden_layers = encoder_layers  # 隐藏层总数（与编码器层数相同）
        self.scale_embedding = scale_embedding  # 是否对嵌入进行缩放
        self.router_z_loss_coef = router_z_loss_coef  # 路由器Z损失系数
        self.router_aux_loss_coef = router_aux_loss_coef  # 路由器辅助损失系数
        self.decoder_sparse_step = decoder_sparse_step  # 解码器稀疏步数
        self.encoder_sparse_step = encoder_sparse_step  # 编码器稀疏步数
        self.num_experts = num_experts  # 专家数量
        self.expert_capacity = expert_capacity  # 专家容量
        self.router_bias = router_bias  # 路由器偏置
        if router_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"`router_dtype` must be one of 'float32', 'float16' or 'bfloat16', got {router_dtype}")
        self.router_dtype = router_dtype  # 路由器数据类型

        self.router_ignore_padding_tokens = router_ignore_padding_tokens  # 是否忽略填充标记
        self.batch_prioritized_routing = batch_prioritized_routing  # 是否进行批量优先路由
        self.second_expert_policy = second_expert_policy  # 第二专家策略
        self.normalize_router_prob_before_dropping = normalize_router_prob_before_dropping  # 是否在丢弃前归一化路由器概率
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction  # MOE评估容量令牌分数
        self.moe_token_dropout = moe_token_dropout  # MOE令牌丢弃率
        self.output_router_logits = output_router_logits  # 输出路由器对数概率
        super().__init__(  # 调用父类的初始化方法
            pad_token_id=pad_token_id,  # 填充标记ID
            bos_token_id=bos_token_id,  # 起始标记ID
            eos_token_id=eos_token_id,  # 结束标记ID
            is_encoder_decoder=is_encoder_decoder,  # 是否为编码器-解码器模型
            decoder_start_token_id=decoder_start_token_id,  # 解码器起始标记ID
            **kwargs,  # 其他参数
        )
```