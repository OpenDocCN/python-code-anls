# `.\models\graphormer\configuration_graphormer.py`

```
# coding=utf-8
# 定义文件编码格式为 UTF-8

# 导入预训练配置类 PretrainedConfig 和日志记录工具 logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 Graphormer 预训练模型配置文件的下载映射表
GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # pcqm4mv1 现在已经不推荐使用
    "graphormer-base": "https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2/resolve/main/config.json",
    # 查看所有 Graphormer 模型的列表链接
    # See all Graphormer models at https://huggingface.co/models?filter=graphormer
}

# GraphormerConfig 类，用于存储 Graphormer 模型的配置信息
class GraphormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~GraphormerModel`]. It is used to instantiate an
    Graphormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Graphormer
    [graphormer-base-pcqm4mv1](https://huggingface.co/graphormer-base-pcqm4mv1) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    # 模型类型为 "graphormer"
    model_type = "graphormer"
    
    # 推断时需要忽略的键列表，在推断时不考虑过去的键值
    keys_to_ignore_at_inference = ["past_key_values"]
    # 初始化函数，用于初始化一个对象实例
    def __init__(
        self,
        num_classes: int = 1,  # 类别数量，默认为1
        num_atoms: int = 512 * 9,  # 原子数量，默认为512 * 9
        num_edges: int = 512 * 3,  # 边的数量，默认为512 * 3
        num_in_degree: int = 512,  # 输入度，默认为512
        num_out_degree: int = 512,  # 输出度，默认为512
        num_spatial: int = 512,  # 空间维度，默认为512
        num_edge_dis: int = 128,  # 边的分布，默认为128
        multi_hop_max_dist: int = 5,  # 多跳的最大距离，默认为5，有时为20
        spatial_pos_max: int = 1024,  # 空间位置的最大值，默认为1024
        edge_type: str = "multi_hop",  # 边的类型，默认为"multi_hop"
        max_nodes: int = 512,  # 最大节点数量，默认为512
        share_input_output_embed: bool = False,  # 是否共享输入输出嵌入，默认为False
        num_hidden_layers: int = 12,  # 隐藏层的数量，默认为12
        embedding_dim: int = 768,  # 嵌入维度，默认为768
        ffn_embedding_dim: int = 768,  # 前馈网络嵌入维度，默认为768
        num_attention_heads: int = 32,  # 注意力头的数量，默认为32
        dropout: float = 0.1,  # dropout概率，默认为0.1
        attention_dropout: float = 0.1,  # 注意力dropout概率，默认为0.1
        activation_dropout: float = 0.1,  # 激活函数dropout概率，默认为0.1
        layerdrop: float = 0.0,  # 层dropout概率，默认为0.0
        encoder_normalize_before: bool = False,  # 编码器层规范化前标志，默认为False
        pre_layernorm: bool = False,  # 层规范化前标志，默认为False
        apply_graphormer_init: bool = False,  # 是否应用Graphormer初始化，默认为False
        activation_fn: str = "gelu",  # 激活函数名称，默认为"gelu"
        embed_scale: float = None,  # 嵌入缩放因子，默认为None
        freeze_embeddings: bool = False,  # 是否冻结嵌入，默认为False
        num_trans_layers_to_freeze: int = 0,  # 要冻结的转换层数量，默认为0
        traceable: bool = False,  # 是否可追踪，默认为False
        q_noise: float = 0.0,  # 量化噪声，默认为0.0
        qn_block_size: int = 8,  # 量化块大小，默认为8
        kdim: int = None,  # 键的维度，默认为None
        vdim: int = None,  # 值的维度，默认为None
        bias: bool = True,  # 是否使用偏置，默认为True
        self_attention: bool = True,  # 是否使用自注意力，默认为True
        pad_token_id=0,  # 填充标记的ID，默认为0
        bos_token_id=1,  # 开始标记的ID，默认为1
        eos_token_id=2,  # 结束标记的ID，默认为2
        **kwargs,  # 其它参数，用于接收未明确定义的关键字参数
        self.num_classes = num_classes
        self.num_atoms = num_atoms
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.num_edges = num_edges
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
        self.max_nodes = max_nodes
        self.num_hidden_layers = num_hidden_layers
        self.embedding_dim = embedding_dim
        self.hidden_size = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.encoder_normalize_before = encoder_normalize_before
        self.pre_layernorm = pre_layernorm
        self.apply_graphormer_init = apply_graphormer_init
        self.activation_fn = activation_fn
        self.embed_scale = embed_scale
        self.freeze_embeddings = freeze_embeddings
        self.num_trans_layers_to_freeze = num_trans_layers_to_freeze
        self.share_input_output_embed = share_input_output_embed
        self.traceable = traceable
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size


        # 初始化模型的各种参数
        self.num_classes = num_classes  # 类别数目
        self.num_atoms = num_atoms  # 原子数
        self.num_in_degree = num_in_degree  # 输入度数
        self.num_out_degree = num_out_degree  # 输出度数
        self.num_edges = num_edges  # 边的数目
        self.num_spatial = num_spatial  # 空间信息数目
        self.num_edge_dis = num_edge_dis  # 边的分布
        self.edge_type = edge_type  # 边的类型
        self.multi_hop_max_dist = multi_hop_max_dist  # 多跳最大距离
        self.spatial_pos_max = spatial_pos_max  # 空间位置的最大值
        self.max_nodes = max_nodes  # 最大节点数
        self.num_hidden_layers = num_hidden_layers  # 隐藏层的数目
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.hidden_size = embedding_dim  # 隐藏层的大小（等于嵌入维度）
        self.ffn_embedding_dim = ffn_embedding_dim  # FeedForward网络的嵌入维度
        self.num_attention_heads = num_attention_heads  # 注意力头的数目
        self.dropout = dropout  # 通用的dropout率
        self.attention_dropout = attention_dropout  # 注意力模块的dropout率
        self.activation_dropout = activation_dropout  # 激活函数的dropout率
        self.layerdrop = layerdrop  # 层的dropout率
        self.encoder_normalize_before = encoder_normalize_before  # 编码器归一化前
        self.pre_layernorm = pre_layernorm  # 层归一化前
        self.apply_graphormer_init = apply_graphormer_init  # 应用Graphormer初始化
        self.activation_fn = activation_fn  # 激活函数
        self.embed_scale = embed_scale  # 嵌入的缩放因子
        self.freeze_embeddings = freeze_embeddings  # 冻结嵌入
        self.num_trans_layers_to_freeze = num_trans_layers_to_freeze  # 要冻结的转换层数
        self.share_input_output_embed = share_input_output_embed  # 共享输入输出嵌入
        self.traceable = traceable  # 可追踪性
        self.q_noise = q_noise  # Q值的噪声
        self.qn_block_size = qn_block_size  # QN块的大小


        # These parameters are here for future extensions
        # atm, the model only supports self attention
        self.kdim = kdim
        self.vdim = vdim
        self.self_attention = self_attention
        self.bias = bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


        # 这些参数用于未来的扩展
        # 目前，模型仅支持自注意力
        self.kdim = kdim  # 键的维度
        self.vdim = vdim  # 值的维度
        self.self_attention = self_attention  # 自注意力
        self.bias = bias  # 偏置

        # 调用父类的初始化方法，设置特殊的token ID和其他关键字参数
        super().__init__(
            pad_token_id=pad_token_id,  # 填充token的ID
            bos_token_id=bos_token_id,  # 开始token的ID
            eos_token_id=eos_token_id,  # 结束token的ID
            **kwargs,  # 其他关键字参数
        )
```