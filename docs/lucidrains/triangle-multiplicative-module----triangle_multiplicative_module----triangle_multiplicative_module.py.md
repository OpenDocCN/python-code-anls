# `.\lucidrains\triangle-multiplicative-module\triangle_multiplicative_module\triangle_multiplicative_module.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 定义辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义类

# 三角形乘法模块类
class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim = None,
        mix = 'ingoing'
    ):
        super().__init__()
        # 断言 mix 参数只能为 'ingoing' 或 'outgoing'
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        # 如果 hidden_dim 不存在，则设为 dim
        hidden_dim = default(hidden_dim, dim)
        # 对输入进行 LayerNorm 归一化
        self.norm = nn.LayerNorm(dim)

        # 左投影层
        self.left_proj = nn.Linear(dim, hidden_dim)
        # 右投影层
        self.right_proj = nn.Linear(dim, hidden_dim)

        # 左门控层
        self.left_gate = nn.Linear(dim, hidden_dim)
        # 右门控层
        self.right_gate = nn.Linear(dim, hidden_dim)
        # 输出门控层
        self.out_gate = nn.Linear(dim, hidden_dim)

        # 初始化所有门控层的权重为 0，偏置为 1
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        # 根据 mix 参数确定 einsum 公式
        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        # 输出层归一化
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        # 输出层线性变换
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        # 断言输入特征图必须是对称的
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        # 如果 mask 存在，则重组 mask 的维度
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        # 对输入进行归一化
        x = self.norm(x)

        # 左投影
        left = self.left_proj(x)
        # 右投影
        right = self.right_proj(x)

        # 如果 mask 存在，则对左右投影进行 mask 处理
        if exists(mask):
            left = left * mask
            right = right * mask

        # 计算左门控值
        left_gate = self.left_gate(x).sigmoid()
        # 计算右门控值
        right_gate = self.right_gate(x).sigmoid()
        # 计算输出门控值
        out_gate = self.out_gate(x).sigmoid()

        # 左投影乘以左门控值
        left = left * left_gate
        # 右投影乘以右门控值
        right = right * right_gate

        # 执行 einsum 运算，根据 mix_einsum_eq 公式计算输出
        out = einsum(self.mix_einsum_eq, left, right)

        # 对输出进行归一化
        out = self.to_out_norm(out)
        # 输出乘以输出门控值
        out = out * out_gate
        # 返回输出结果
        return self.to_out(out)
```