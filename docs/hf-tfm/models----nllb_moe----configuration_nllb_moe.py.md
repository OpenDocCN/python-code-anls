# `.\models\nllb_moe\configuration_nllb_moe.py`

```
"""
NLLB-MoE model configuration
"""
# 导入所需模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 映射预训练配置文件的 URL 到模型名称
NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/nllb-moe-54B": "https://huggingface.co/facebook/nllb-moe-54b/resolve/main/config.json",
}

# 定义 NllbMoeConfig 类，继承自 PretrainedConfig
class NllbMoeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NllbMoeModel`]. It is used to instantiate an
    NLLB-MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the NLLB-MoE
    [facebook/nllb-moe-54b](https://huggingface.co/facebook/nllb-moe-54b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import NllbMoeModel, NllbMoeConfig

    >>> # Initializing a NllbMoe facebook/nllb-moe-54b style configuration
    >>> configuration = NllbMoeConfig()

    >>> # Initializing a model from the facebook/nllb-moe-54b style configuration
    >>> model = NllbMoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型定义为 "nllb-moe"
    model_type = "nllb-moe"
    # 推断阶段要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，用于配置转换
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 定义一个初始化方法，用于初始化 Transformer 架构的模型参数和设置
    def __init__(
        self,
        vocab_size=128112,  # 词汇表大小，默认为128112
        max_position_embeddings=1024,  # 最大位置编码长度，默认为1024
        encoder_layers=12,  # 编码器层数，默认为12层
        encoder_ffn_dim=4096,  # 编码器中 FeedForward 层的维度，默认为4096
        encoder_attention_heads=16,  # 编码器中注意力头的数量，默认为16个
        decoder_layers=12,  # 解码器层数，默认为12层
        decoder_ffn_dim=4096,  # 解码器中 FeedForward 层的维度，默认为4096
        decoder_attention_heads=16,  # 解码器中注意力头的数量，默认为16个
        encoder_layerdrop=0.05,  # 编码器层级丢弃率，默认为0.05
        decoder_layerdrop=0.05,  # 解码器层级丢弃率，默认为0.05
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否是编码-解码架构，默认为True
        activation_function="relu",  # 激活函数，默认为ReLU
        d_model=1024,  # 模型维度，默认为1024
        dropout=0.1,  # 普通Dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力Dropout率，默认为0.1
        activation_dropout=0.0,  # 激活函数Dropout率，默认为0.0
        init_std=0.02,  # 初始化的标准差，默认为0.02
        decoder_start_token_id=2,  # 解码器起始标记ID，默认为2
        scale_embedding=True,  # 是否缩放嵌入，默认为True
        router_bias=False,  # 路由器是否包含偏置项，默认为False
        router_dtype="float32",  # 路由器数据类型，默认为float32
        router_ignore_padding_tokens=False,  # 路由器是否忽略填充标记，默认为False
        num_experts=128,  # 专家数量，默认为128
        expert_capacity=64,  # 每个专家的容量，默认为64
        encoder_sparse_step=4,  # 编码器稀疏步长，默认为4
        decoder_sparse_step=4,  # 解码器稀疏步长，默认为4
        router_z_loss_coef=0.001,  # 路由器Z损失系数，默认为0.001
        router_aux_loss_coef=0.001,  # 路由器辅助损失系数，默认为0.001
        second_expert_policy="all",  # 第二专家策略，默认为"all"
        normalize_router_prob_before_dropping=False,  # 是否在丢弃前归一化路由器概率，默认为False
        batch_prioritized_routing=False,  # 批量优先路由，默认为False
        moe_eval_capacity_token_fraction=1.0,  # MOE评估容量标记分数，默认为1.0
        moe_token_dropout=0.2,  # MOE标记丢弃率，默认为0.2
        pad_token_id=1,  # 填充标记ID，默认为1
        bos_token_id=0,  # 起始标记ID，默认为0
        eos_token_id=2,  # 结束标记ID，默认为2
        output_router_logits=False,  # 是否输出路由器logits，默认为False
        **kwargs,  # 其他关键字参数
        ):
        # 初始化 Transformer 架构的参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码长度
        self.d_model = d_model  # 模型的维度大小
        self.encoder_ffn_dim = encoder_ffn_dim  # 编码器中间层的维度
        self.encoder_layers = encoder_layers  # 编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 编码器注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim  # 解码器中间层的维度
        self.decoder_layers = decoder_layers  # 解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 解码器注意力头数
        self.dropout = dropout  # 普通丢弃率
        self.attention_dropout = attention_dropout  # 注意力机制中的丢弃率
        self.activation_dropout = activation_dropout  # 激活函数中的丢弃率
        self.activation_function = activation_function  # 激活函数类型
        self.init_std = init_std  # 参数初始化的标准差
        self.encoder_layerdrop = encoder_layerdrop  # 编码器层丢弃率
        self.decoder_layerdrop = decoder_layerdrop  # 解码器层丢弃率
        self.use_cache = use_cache  # 是否使用缓存
        self.num_hidden_layers = encoder_layers  # 隐藏层的数量等同于编码器层数
        self.scale_embedding = scale_embedding  # 如果为 True，嵌入的缩放因子为 sqrt(d_model)
        self.router_z_loss_coef = router_z_loss_coef  # 路由器 z 损失的系数
        self.router_aux_loss_coef = router_aux_loss_coef  # 路由器辅助损失的系数
        self.decoder_sparse_step = decoder_sparse_step  # 解码器稀疏步长
        self.encoder_sparse_step = encoder_sparse_step  # 编码器稀疏步长
        self.num_experts = num_experts  # 专家数量
        self.expert_capacity = expert_capacity  # 专家容量
        self.router_bias = router_bias  # 路由器偏置
        if router_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"`router_dtype` must be one of 'float32', 'float16' or 'bfloat16', got {router_dtype}")
        self.router_dtype = router_dtype  # 路由器数据类型，必须是 float32、float16 或 bfloat16 中的一种

        self.router_ignore_padding_tokens = router_ignore_padding_tokens  # 是否忽略填充标记的路由
        self.batch_prioritized_routing = batch_prioritized_routing  # 是否进行批次优先路由
        self.second_expert_policy = second_expert_policy  # 第二专家策略
        self.normalize_router_prob_before_dropping = normalize_router_prob_before_dropping  # 在丢弃前是否对路由器概率进行归一化
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction  # MOE 评估容量的标记分数
        self.moe_token_dropout = moe_token_dropout  # MOE 标记的丢弃率
        self.output_router_logits = output_router_logits  # 输出路由器的对数概率
        super().__init__(
            pad_token_id=pad_token_id,  # 填充标记的 ID
            bos_token_id=bos_token_id,  # 开始标记的 ID
            eos_token_id=eos_token_id,  # 结束标记的 ID
            is_encoder_decoder=is_encoder_decoder,  # 是否为编码-解码模型
            decoder_start_token_id=decoder_start_token_id,  # 解码器开始标记的 ID
            **kwargs,  # 其他参数
        )
```