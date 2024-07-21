# `.\pytorch\torch\ao\quantization\experimental\linear.py`

```
# mypy: allow-untyped-defs
# 引入 PyTorch 库
import torch
# 引入 NumPy 库
import numpy as np

# 从 Torch 库中导入特定的量化模块和观察器
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT

# 定义一个继承自 WeightedQuantizedModule 的 APoT 量化线性模块类
class LinearAPoT(WeightedQuantizedModule):
    r"""
    A quantized linear module with quantized tensor as inputs and outputs
    to support APoT quantization.
    We adopt the same interface as `torch.nn.Linear`, see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`~torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        alpha: `alpha` qparam of output Quantized Tensor, type: Tensor
        gamma: `gamma` qparam of output Quantized Tensor, type: Tensor
        quantization_levels: `quantization_levels` qparam of output Quantized Tensor, type: Tensor
        level_indices: `level_indices` qparam of output Quantized Tensor, type: Tensor
        weight: APoT quantized tensor from weight2quantize
        weight_transposed: transposed weight tensor, used in linear transformation calculation (y = x * A^T + b)
    """

    # 初始化函数，接受要量化的权重张量 weight2quantize，以及整数参数 b 和 k
    def __init__(self, weight2quantize: torch.Tensor, b: int, k: int):
        # 断言权重张量为二维张量
        assert weight2quantize.dim() == 2
        # 断言 b 能够整除 k
        assert b % k == 0

        # 调用父类的初始化方法
        super().__init__()

        # 将参数 b 和 k 存储在对象中
        self.b = b
        self.k = k
        self.n = self.b // self.k

        # 创建 APoT 观察器对象，并对输入的权重张量进行观察
        observer = APoTObserver(b=self.b, k=self.k)
        observer(weight2quantize)

        # 根据观察结果计算 alpha、gamma、quantization_levels 和 level_indices
        self.alpha, self.gamma, self.quantization_levels, self.level_indices = observer.calculate_qparams(signed=False)

        # 使用 APoT 量化算法对权重张量进行量化
        quantized_weight = quantize_APoT(weight2quantize, self.alpha, self.gamma, self.quantization_levels, self.level_indices)

        # 将量化后的权重数据存储在对象中
        self.weight = quantized_weight.data

        # 计算权重的转置，用于线性变换计算 (y = x * A^T + b)
        self.weight_transposed = torch.transpose(self.weight, 0, 1)

    # 解析 APoT 值的二进制表示，将其分解为 k 大小的块
    def decompose_APoT(self, x):
        r"""
        Decompose binary representation of APoT values into list of k-sized blocks
        Args:
            x (Tensor): binary representation of APoT quantized tensor
        """
        # 去除二进制表示中的 "0b" 前缀
        x = x[2:]

        # 初始化块列表
        blocks = []

        # 循环直到 x 为空
        while x:
            # 将 x 的前 k 个字符作为一个块加入列表
            blocks.append(x[0:self.k])
            # 更新 x，去除已经处理过的部分
            x = x[self.k:]

        # 返回分解后的块列表
        return blocks
    def bitshift_mul(self, weight_val, r):
        r"""
        Compute multiplication of weight_val * r using bitshifting
        method discussed in APoT paper: https://arxiv.org/pdf/1909.13144.pdf
        Args:
            weight_val: list of binary digits representing APoT quantized weight value
            r: int representing uniformly quantized activation value
        """
        product = 0

        idx = len(weight_val) - 1
        place = 0

        while idx >= 0:
            block = weight_val[idx]

            # reverse digits in block
            block = block[::-1]

            curr_block_result = 0

            for ele in block:
                if int(ele):
                    curr_block_result += r << place
                place += 1

            idx -= 1
            product += curr_block_result

        return product


    def matmul(self, decomposed_weight, activation):
        r"""
        Perform matrix multiplication between decomposed_weight and
        activation by calling bitshift_mul function for each value
        Args:
            decomposed_weight (Tensor): APoT quantized weight decomposed into binary
            activation (Tensor): uniformly quantized activation
        """
        rows1 = activation.size(dim=0)
        cols1 = activation.size(dim=1)

        rows2 = decomposed_weight.shape[0]
        cols2 = decomposed_weight.shape[1]

        result = torch.zeros(rows1, cols2)

        # compute matrix multiplication with bitshifts
        for i in range(rows1):
            for j in range(cols2):
                for k in range(rows2):
                    weight_val = decomposed_weight[k][j]
                    r = int(activation[i][k])

                    product = self.bitshift_mul(weight_val, r)

                    result[i][j] += product

        return result

    def forward(self, activation: torch.Tensor) -> torch.FloatTensor:
        r"""
        Multiply APoT quantized weight and uniformly quantized activation (dtype: quint8)
        with bitshifting instead of matrix multiplication.
        Result has dtype torch.float32
        Args:
            activation (Tensor): uniformly quantized activation tensor
        """
        assert activation.dim() == 2

        weight_rows = self.weight_transposed.size()[0]
        weight_cols = self.weight_transposed.size()[1]

        decomposed_weight: np.ndarray = np.empty(shape=(weight_rows, weight_cols), dtype=object)
        for row in range(weight_rows):
            for col in range(weight_cols):
                decomposed_weight[row][col] = self.decompose_APoT(bin(self.weight_transposed[row][col]))

        result = self.matmul(decomposed_weight, activation).type(torch.FloatTensor)

        return result

    @classmethod
    # 定义一个类方法 `from_reference`，用于从参考模型转换，但不支持类型检查的覆盖
    def from_reference(cls,  # type: ignore[override]
                       ref_qlinear,  # 引用 QLinear 模型的参数
                       alpha: torch.Tensor,  # 线性量化中的 alpha 参数，类型为 Torch 张量
                       gamma: torch.Tensor,  # 线性量化中的 gamma 参数，类型为 Torch 张量
                       quantization_levels: torch.Tensor,  # 量化级别的 Torch 张量
                       level_indices: torch.Tensor):  # 级别索引的 Torch 张量
        # 抛出未实现的错误，表明该方法需要在子类中被实现
        raise NotImplementedError
```