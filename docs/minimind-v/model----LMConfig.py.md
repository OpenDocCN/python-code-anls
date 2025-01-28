# `.\minimind-v\model\LMConfig.py`

```py
# 从 transformers 库导入 PretrainedConfig 类，用于定义预训练模型的配置
from transformers import PretrainedConfig
# 从 typing 库导入 List 类型注解，用于声明列表类型
from typing import List


# 定义一个名为 LMConfig 的类，继承自 PretrainedConfig 类，表示最小化版的语言模型配置
class LMConfig(PretrainedConfig):
    # 指定该配置对应的模型类型为 "minimind"
    model_type = "minimind"

    # 初始化方法，设置各项超参数及默认值
    def __init__(
            self,
            dim: int = 512,  # 隐藏层的维度，默认为 512
            n_layers: int = 8,  # 模型的层数，默认为 8
            n_heads: int = 16,  # 每层的注意力头数，默认为 16
            n_kv_heads: int = 8,  # KV 注意力头数，默认为 8
            vocab_size: int = 6400,  # 词汇表大小，默认为 6400
            hidden_dim: int = None,  # 隐藏层的维度，默认为 None
            multiple_of: int = 64,  # 模型维度应为该数的倍数，默认为 64
            norm_eps: float = 1e-5,  # 归一化操作中的 epsilon，默认为 1e-5
            max_seq_len: int = 512,  # 最大序列长度，默认为 512
            dropout: float = 0.0,  # dropout 概率，默认为 0.0
            flash_attn: bool = True,  # 是否启用闪电注意力，默认为 True
            image_special_token: str = '<' * 25 + '>' * 25,  # 图像的特殊标记，默认为一个由 '<' 和 '>' 组成的字符串
            image_ids: List = [30] * 25 + [32] * 25,  # 图像的 ID 列表，默认为一个包含 30 和 32 的列表
            ####################################################
            # 以下是 MOE（混合专家）相关配置
            # 当 use_moe 为 False 时，以下配置无效
            ####################################################
            use_moe: bool = False,  # 是否使用 MOE 模型，默认为 False
            num_experts_per_tok=2,  # 每个 token 使用的专家数量，默认为 2
            n_routed_experts=4,  # 总的专家数量，默认为 4
            n_shared_experts: bool = True,  # 是否使用共享专家，默认为 True
            scoring_func='softmax',  # 专家评分函数，默认为 'softmax'
            aux_loss_alpha=0.01,  # 辅助损失的 alpha 参数，默认为 0.01
            seq_aux=True,  # 是否在序列级别上计算辅助损失，默认为 True
            norm_topk_prob=True,  # 是否对 top-k 概率进行标准化，默认为 True
            **kwargs,  # 允许接收其他未列出的额外参数
    ):
        # 设置各项超参数的值
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        ####################################################
        # 以下是 MOE（混合专家）相关配置
        # 当 use_moe 为 False 时，以下配置无效
        ####################################################
        # 设置 MOE 模型相关的超参数
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个 token 选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 是否使用共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为 'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的 alpha 参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化 top-k 概率
        # 调用父类 PretrainedConfig 的初始化方法，传入额外参数 kwargs
        super().__init__(**kwargs)
```