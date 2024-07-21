# `.\pytorch\test\functorch\attn_ft.py`

```py
# 导入 math 模块
import math

# 导入 PyTorch 库
import torch

# 从 functorch.dim 模块导入 cat, dimlists, dims, softmax 函数
from functorch.dim import cat, dimlists, dims, softmax

# 从 torch 库导入 nn 模块
from torch import nn


# 定义一个自定义的线性层类，继承自 nn.Linear
class Linear(nn.Linear):
    # 重写 forward 方法
    def forward(self, input):
        # 调用 dims() 函数获取维度信息并解包为 ci (输入通道索引) 和 co (输出通道索引)
        ci, co = dims()
        
        # 调用 dimlists() 函数获取维度列表
        b = dimlists()
        
        # 计算线性层的输出结果，使用了自定义的索引方式
        result = (input[b, ci] * self.weight[co, ci]).sum(ci) + self.bias[co]
        
        # 返回结果，并按照指定的顺序排序维度
        return result.order(b, co)


# 定义 BertSelfAttention 类，继承自 nn.Module
class BertSelfAttention(nn.Module):
    # 初始化方法
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        position_embedding_type=None,
        max_position_embeddings=None,
        linear=Linear,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 检查隐藏大小是否能被注意力头数整除
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层
        self.query = linear(hidden_size, self.all_head_size)
        self.key = linear(hidden_size, self.all_head_size)
        self.value = linear(hidden_size, self.all_head_size)

        # 设置注意力概率的丢弃率
        self.dropout_prob = attention_probs_dropout_prob
        self.position_embedding_type = position_embedding_type

        # 如果设置了位置嵌入类型，创建距离嵌入的 Embedding 层
        if self.position_embedding_type is not None:
            assert max_position_embeddings is not None
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * max_position_embeddings - 1, self.attention_head_size
            )

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        past_key_value=None,
```