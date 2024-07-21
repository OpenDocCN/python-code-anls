# `.\pytorch\torch\nn\modules\distance.py`

```
# 导入 torch.nn.functional 中的 F 模块，用于函数操作
# 导入 Tensor 类型，用于处理张量数据
import torch.nn.functional as F
from torch import Tensor

# 从当前目录下的 module.py 文件中导入 Module 类
from .module import Module

# 定义公开的类列表，指定可以从当前模块导出的类名称
__all__ = ["PairwiseDistance", "CosineSimilarity"]

# PairwiseDistance 类，继承自 Module 类，用于计算输入向量或矩阵列之间的成对距离
class PairwiseDistance(Module):
    r"""
    计算输入向量之间或输入矩阵列之间的成对距离。

    距离使用 ``p``-norm 计算，添加常量 ``eps`` 以避免除以零，如果 ``p`` 是负数，则有：

    .. math ::
        \mathrm{dist}\left(x, y\right) = \left\Vert x-y + \epsilon e \right\Vert_p,

    其中 :math:`e` 是全为1的向量，``p``-norm 由以下公式给出：

    .. math ::
        \Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}.

    Args:
        p (real, optional): 范数的阶数。可以是负数。默认值: 2
        eps (float, optional): 避免除以零的小值。
            默认值: 1e-6
        keepdim (bool, optional): 决定是否保持向量维度。
            默认值: False
    Shape:
        - Input1: :math:`(N, D)` 或 :math:`(D)`，其中 `N` 是批量维度，`D` 是向量维度
        - Input2: :math:`(N, D)` 或 :math:`(D)`，与 Input1 相同的形状
        - Output: :math:`(N)` 或 :math:`()`，根据输入维度决定。
          如果 :attr:`keepdim` 是 ``True``，则为 :math:`(N, 1)` 或 :math:`(1)`
          基于输入的维度。

    Examples::
        >>> pdist = nn.PairwiseDistance(p=2)
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> output = pdist(input1, input2)
    """

    __constants__ = ["norm", "eps", "keepdim"]
    norm: float
    eps: float
    keepdim: bool

    def __init__(
        self, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False
    ) -> None:
        super().__init__()
        self.norm = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return F.pairwise_distance(x1, x2, self.norm, self.eps, self.keepdim)


# CosineSimilarity 类，继承自 Module 类，用于计算输入向量之间的余弦相似度
class CosineSimilarity(Module):
    r"""返回 :math:`x_1` 和 :math:`x_2` 之间的余弦相似度，沿着 `dim` 维度计算。

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.

    Args:
        dim (int, optional): 计算余弦相似度的维度。默认值: 1
        eps (float, optional): 避免除以零的小值。
            默认值: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)`，其中 `D` 在 `dim` 位置
        - Input2: :math:`(\ast_1, D, \ast_2)`，与 x1 具有相同数量的维度，在维度 `dim` 上与 x1 的大小匹配，
              并在其他维度上与 x1 广播。
        - Output: :math:`(\ast_1, \ast_2)`
    Examples::
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        >>> output = cos(input1, input2)
    """

    __constants__ = ["dim", "eps"]
    dim: int
    eps: float


    # 定义一个类变量 eps，表示余弦相似度计算中的小数阈值
    eps: float



    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:


    # 定义类的初始化方法
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 设置对象的维度属性
        self.dim = dim
        # 设置对象的 eps 属性，用于控制余弦相似度计算中的数值稳定性
        self.eps = eps



    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:


    # 定义类的前向传播方法
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # 调用 PyTorch 的函数计算输入张量 x1 和 x2 的余弦相似度
        return F.cosine_similarity(x1, x2, self.dim, self.eps)
```