# `.\models\graphormer\configuration_graphormer.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2022 年 Microsoft、clefourrier 和 The HuggingFace Inc. 团队。保留所有权利。
# 根据 Apache 许可证 2.0 版本（“许可证”）授权。
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件均基于“原样”分发，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取有关特定语言的权限和限制。
""" Graphormer 模型配置"""

# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射
GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # pcqm4mv1 现已弃用
    "graphormer-base": "https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2/resolve/main/config.json",
    # 查看所有 Graphormer 模型 https://huggingface.co/models?filter=graphormer
}

# Graphormer 配置类，用于存储 [`~GraphormerModel`] 的配置
class GraphormerConfig(PretrainedConfig):
    r"""
    这是用于存储 [`~GraphormerModel`] 配置的配置类。它用于根据指定的参数实例化一个 Graphormer 模型，
    定义模型架构。使用默认值实例化配置将产生类似于 Graphormer
    [graphormer-base-pcqm4mv1](https://huggingface.co/graphormer-base-pcqm4mv1) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读
    [`PretrainedConfig`] 的文档以获取更多信息。
    """
    
    # 模型类型为 "graphormer"
    model_type = "graphormer"
    # 推断时要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 初始化函数，设置各种参数的默认值
    def __init__(
        self,
        num_classes: int = 1,  # 类别数量，默认为1
        num_atoms: int = 512 * 9,  # 原子数量，默认为512*9
        num_edges: int = 512 * 3,  # 边的数量，默认为512*3
        num_in_degree: int = 512,  # 输入度，默认为512
        num_out_degree: int = 512,  # 输出度，默认为512
        num_spatial: int = 512,  # 空间维度，默认为512
        num_edge_dis: int = 128,  # 边的距离，默认为128
        multi_hop_max_dist: int = 5,  # 多跳最大距离，默认为5，有时为20
        spatial_pos_max: int = 1024,  # 空间位置最大值，默认为1024
        edge_type: str = "multi_hop",  # 边的类型，默认为"multi_hop"
        max_nodes: int = 512,  # 最大节点数，默认为512
        share_input_output_embed: bool = False,  # 是否共享输入输出嵌入，默认为False
        num_hidden_layers: int = 12,  # 隐藏层的数量，默认为12
        embedding_dim: int = 768,  # 嵌入维度，默认为768
        ffn_embedding_dim: int = 768,  # 前馈网络嵌入维度，默认为768
        num_attention_heads: int = 32,  # 注意力头的数量，默认为32
        dropout: float = 0.1,  # 丢弃率，默认为0.1
        attention_dropout: float = 0.1,  # 注意力丢弃率，默认为0.1
        activation_dropout: float = 0.1,  # 激活函数丢弃率，默认为0.1
        layerdrop: float = 0.0,  # 层丢弃率，默认为0.0
        encoder_normalize_before: bool = False,  # 编码器是否在层归一化之前，默认为False
        pre_layernorm: bool = False,  # 是否在层归一化之前应用预归一化，默认为False
        apply_graphormer_init: bool = False,  # 是否应用Graphormer初始化，默认为False
        activation_fn: str = "gelu",  # 激活函数，默认为"gelu"
        embed_scale: float = None,  # 嵌入缩放，默认为None
        freeze_embeddings: bool = False,  # 是否冻结嵌入，默认为False
        num_trans_layers_to_freeze: int = 0,  # 要冻结的转换层数量，默认为0
        traceable: bool = False,  # 是否可追踪，默认为False
        q_noise: float = 0.0,  # Q噪声，默认为0.0
        qn_block_size: int = 8,  # QN块大小，默认为8
        kdim: int = None,  # K维度，默认为None
        vdim: int = None,  # V维度，默认为None
        bias: bool = True,  # 是否使用偏置，默认为True
        self_attention: bool = True,  # 是否自注意力，默认为True
        pad_token_id=0,  # 填充标记ID，默认为0
        bos_token_id=1,  # 开始标记ID，默认为1
        eos_token_id=2,  # 结束标记ID，默认为2
        **kwargs,  # 其他关键字参数
        # 初始化模型参数
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

        # 以下参数用于未来扩展
        # 目前模型仅支持自注意力
        self.kdim = kdim
        self.vdim = vdim
        self.self_attention = self_attention
        self.bias = bias

        # 调用父类的初始化方法，传入特殊标记的参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
```