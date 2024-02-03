# `bitsandbytes\bitsandbytes\autograd\_functions.py`

```py
# 导入必要的库
from dataclasses import dataclass
from functools import reduce  # Python 3 中需要
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn

import torch

import bitsandbytes.functional as F

# 定义一个计算可迭代对象乘积的函数
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

# 以下是关于 colTuring 和 colAmpere 格式的逆转换，由 Alex Borzunov 贡献
# https://github.com/bigscience-workshop/petals/blob/main/src/petals/utils/linear8bitlt_patch.py

"""
    这个类在层之间汇总异常维度。
    这对于小型模型特别重要，其中异常特征不太系统化，且出现频率较低。
"""
class GlobalOutlierPooler:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.outliers = set()
        self.model_dim = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def add_outliers(self, outlier_idx, feature_dim):
        if self.model_dim is None:
            self.model_dim = feature_dim
        if feature_dim != self.model_dim:
            return  # 对于第二个 FFN 层，我们不编码异常值

        self.outliers.update(outlier_idx.tolist())

    def get_current_outlier_idx(self):
        return torch.Tensor(list(self.outliers)).to(torch.int64)

# 获取逆转换的索引
def get_inverse_transform_indices(
    transform_tile: Callable[[torch.Tensor], torch.Tensor],
    tile_size: Tuple[int, int],
):
    """
    计算一个索引的排列，以反转指定的（分块）矩阵变换

    :param transform_tile: 一个将正向变换应用于形状为 [dim1, dim2] 的张量的函数
    :param tile_size: 更高级别的瓦片维度，例如 Turing 的 (8, 32) 和 Ampere 的 (32, 32)
    :note: 假设 tile_transform 适用于形状为 tile_size 的基于 CPU 的 int8 张量
    :example: 用于图灵布局的 transform_tile 函数（bitsandbytes.functional as F）
    :returns: 索引
    """
    # 获取 tile_size 的两个维度
    d1, d2 = tile_size
    # 断言 d1 * d2 在 0 到 2^64 之间
    assert 0 < d1 * d2 < 2**64
    # 创建一个包含 0 到 d1*d2-1 的整数张量，类型为 int64，然后将其形状变为 d1 行 d2 列
    tile_indices = torch.arange(d1 * d2, dtype=torch.int64).view(d1, d2)
    # 对每个位置在 tile 中编码为一个包含 <= 8 个唯一字节的元组
    permuted_tile_indices = torch.zeros_like(tile_indices)
    for i in range(8):
        # 选择第 i 个字节，应用变换并跟踪每个索引的最终位置
        ith_dim_indices = torch.div(tile_indices, 256**i, rounding_mode="trunc") % 256
        sample_tile_i = (ith_dim_indices - 128).to(torch.int8).contiguous()
        assert torch.all(sample_tile_i.int() + 128 == ith_dim_indices), "int overflow"
        permuted_tile_i = transform_tile(sample_tile_i)
        ith_permuted_indices = permuted_tile_i.to(tile_indices.dtype) + 128
        permuted_tile_indices += ith_permuted_indices * (256**i)
        if d1 * d2 < 256**i:
            break  # 如果所有索引都适合 i 个字节，提前停止
    return permuted_tile_indices
def undo_layout(permuted_tensor: torch.Tensor, tile_indices: torch.LongTensor) -> torch.Tensor:
    """
    Undo a tiled permutation such as turing or ampere layout

    :param permuted_tensor: torch tensor in a permuted layout
    :param tile_indices: reverse transformation indices, from get_inverse_transform_indices
    :return: contiguous row-major tensor
    """
    # 获取 permuted_tensor 和 tile_indices 的形状信息
    (rows, cols), (tile_rows, tile_cols) = permuted_tensor.shape, tile_indices.shape
    # 检查是否能够整除，确保 tensor 包含整数个瓦片
    assert rows % tile_rows == cols % tile_cols == 0, "tensor must contain a whole number of tiles"
    # 将 permuted_tensor 重塑为二维张量，并转置
    tensor = permuted_tensor.reshape(-1, tile_indices.numel()).t()
    # 创建一个与 tensor 相同形状的空张量 outputs
    outputs = torch.empty_like(tensor)  # note: not using .index_copy because it was slower on cuda
    # 根据 tile_indices 将 tensor 中的数据填充到 outputs 中
    outputs[tile_indices.flatten()] = tensor
    # 将 outputs 重塑为指定形状
    outputs = outputs.reshape(tile_rows, tile_cols, cols // tile_cols, rows // tile_rows)
    # 对 outputs 进行维度置换
    outputs = outputs.permute(3, 0, 2, 1)  # (rows // tile_rows, tile_rows), (cols // tile_cols, tile_cols)
    # 返回最终结果并确保是连续的
    return outputs.reshape(rows, cols).contiguous()


class MatMul8bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, out=None, quant_type="vector", precision=None):
        # 如果 precision 为 None，则设置默认值为 [8, 8, 8]
        if precision is None:
            precision = [8, 8, 8]
        # 如果 precision 的第一个元素不为 8，则直接使用 torch.matmul 计算输出
        if precision[0] != 8:
            with torch.no_grad():
                output = torch.matmul(A, B)
        else:
            # 根据 B 的维度确定 dim 的值
            if len(B.shape) == 2:
                dim = 0
            else:
                dim = 1
            # 对 A 和 B 进行向量量化
            qA, SA = F.vectorwise_quant(A, dim=-1, quant_type=quant_type)
            qB, SB = F.vectorwise_quant(B, dim=dim, quant_type=quant_type)
            # 使用量化后的输入进行整数矩阵乘法
            iout = F.igemm(qA, qB)
            # 对整数矩阵乘法的结果进行反量化得到最终输出
            output = F.vectorwise_mm_dequant(iout, SA, SB, A.dtype, quant_type)

        # 如果 A 或 B 需要梯度，则保存它们用于反向传播
        if A.requires_grad or B.requires_grad:
            ctx.save_for_backward(A, B)

        # 保存量化类型和精度信息到上下文中
        ctx.quant_type = quant_type
        ctx.precision = precision

        return output

    @staticmethod
    mm_cublas = MatMul8bit.apply
    bmm_cublas = MatMul8bit.apply
    matmul_cublas = MatMul8bit.apply
# 检查设备是否支持优化的 int8 内核
def supports_igemmlt(device: torch.device) -> bool:
    # 如果设备的 CUDA 计算能力小于 (7, 5)，则返回 False
    if torch.cuda.get_device_capability(device=device) < (7, 5):
        return False
    # 获取设备名称
    device_name = torch.cuda.get_device_name(device=device)
    # NVIDIA 16 系列的设备，不支持 tensor cores，返回 False
    nvidia16_models = ('GTX 1630', 'GTX 1650', 'GTX 1660')  # https://en.wikipedia.org/wiki/GeForce_16_series
    if any(model_name in device_name for model_name in nvidia16_models):
        return False
    # 其他情况返回 True
    return True

# 获取瓦片大小
def _get_tile_size(format):
    # 断言格式为 "col_turing" 或 "col_ampere"
    assert format in (
        "col_turing",
        "col_ampere",
    ), f"please find this assert and manually enter tile size for {format}"
    # 返回瓦片大小
    return (8, 32) if format == "col_turing" else (32, 32)

# 获取瓦片索引
def get_tile_inds(format, device):
    # 转换函数，将输入转换为指定格式
    transform = lambda x: F.transform(x.to(device), from_order="row", to_order=format)[0].to(x.device)
    # 禁用梯度计算
    with torch.no_grad():
        # 返回瓦片索引
        return get_inverse_transform_indices(transform, _get_tile_size(format)).to(device)

# 定义 MatmulLtState 类
@dataclass
class MatmulLtState:
    _tile_indices: Optional[torch.Tensor] = None
    force_no_igemmlt: bool = False
    CB = None
    CxB = None
    SB = None
    SCB = None

    CxBt = None
    SBt = None
    CBt = None

    subB = None

    outlier_pool = None
    has_accumulated_gradients = False
    threshold = 0.0
    idx = None
    is_training = True
    has_fp16_weights = True
    memory_efficient_backward = False
    use_pool = False
    formatB = F.get_special_format_str()

    # 重置梯度
    def reset_grads(self):
        self.CB = None
        self.CxB = None
        self.SB = None
        self.SCB = None

        self.CxBt = None
        self.SBt = None
        self.CBt = None

    # 获取瓦片索引属性
    @property
    def tile_indices(self):
        # 如果瓦片索引为空，调用 get_tile_inds 获取瓦片索引
        if self._tile_indices is None:
            self._tile_indices = get_tile_inds(self.formatB, self.CxB.device)
        return self._tile_indices

# 定义 MatMul8bitLt 类
class MatMul8bitLt(torch.autograd.Function):
    # forward方法与之前相同，但我们为旧版GPU添加了回退选项
    # backward方法大部分与之前相同，但添加了一个额外的条件（参见"elif state.CxB is not None"）

    @staticmethod
    @staticmethod
class MatMul4Bit(torch.autograd.Function):
    # 定义一个自定义的 PyTorch 自动求导函数 MatMul4Bit

    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state: Optional[F.QuantState] = None):
        # 定义 forward 方法，接受输入 A, B, out, bias, quant_state 参数

        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        # 检查输入是否为空，如果是空的则返回对应形状的空张量
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            B_shape = quant_state.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)

        # 1. Dequantize
        # 2. MatmulnN
        # 对输入 B 进行 4 位量化，然后进行矩阵乘法操作
        output = torch.nn.functional.linear(A, F.dequantize_4bit(B, quant_state).to(A.dtype).t(), bias)

        # 3. Save state
        # 保存量化状态和数据类型信息
        ctx.state = quant_state
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

        # 根据是否需要计算梯度，保存输入张量信息
        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (A, B)
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    # 定义反向传播函数，计算梯度
    def backward(ctx, grad_output):
        # 如果上下文为空，则返回相应的梯度
        if ctx.is_empty:
            # 如果偏置为空，则偏置梯度为None，否则为与偏置相同形状的零张量
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            # 返回与A、B相同形状的零张量，以及偏置梯度为None
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None

        # 获取需要计算梯度的标志
        req_gradA, _, _, req_gradBias, _= ctx.needs_input_grad
        # 获取A和B张量
        A, B = ctx.tensors

        # 初始化梯度为None
        grad_A, grad_B, grad_bias = None, None, None

        # 如果需要计算偏置梯度
        if req_gradBias:
            # 首先计算偏置梯度，然后再改变grad_output的数据类型
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # 不被PyTorch支持，需要创建解决方法
        #if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
        # 如果需要计算A的梯度
        if req_gradA: 
            # 计算A的梯度，先将B反量化为浮点数，然后转置，最后与grad_output相乘
            grad_A = torch.matmul(grad_output, F.dequantize_4bit(B, ctx.state).to(grad_output.dtype).t())

        # 返回A的梯度、B的梯度为None、None、偏置梯度以及None
        return grad_A, grad_B, None, grad_bias, None
# 矩阵乘法函数，支持8位整数矩阵乘法
def matmul(
    # 输入矩阵 A
    A: torch.Tensor,
    # 输入矩阵 B
    B: torch.Tensor,
    # 输出矩阵，可选
    out: Optional[torch.Tensor] = None,
    # 状态信息，可选
    state: Optional[MatmulLtState] = None,
    # 阈值，默认为0.0
    threshold=0.0,
    # 偏置，默认为None
    bias=None
):
    # 如果状态信息为空，则创建一个新的状态对象
    state = state or MatmulLtState()
    # 如果阈值大于0.0，则更新状态对象的阈值
    if threshold > 0.0:
        state.threshold = threshold
    # 调用自定义的8位整数矩阵乘法函数
    return MatMul8bitLt.apply(A, B, out, bias, state)


# 4位整数矩阵乘法函数
def matmul_4bit(A: torch.Tensor, B: torch.Tensor, quant_state: F.QuantState, out: Optional[torch.Tensor] = None, bias=None):
    # 断言量化状态不为空
    assert quant_state is not None
    # 如果输入矩阵 A 的元素数量等于最后一个维度大小且不需要梯度
    if A.numel() == A.shape[-1] and A.requires_grad == False:
        # 如果最后一个维度大小不是块大小的倍数
        if A.shape[-1] % quant_state.blocksize != 0:
            # 输出警告信息
            warn(f'Some matrices hidden dimension is not a multiple of {quant_state.blocksize} and efficient inference kernels are not supported for these (slow). Matrix input size found: {A.shape}')
            # 调用自定义的4位整数矩阵乘法函数
            return MatMul4Bit.apply(A, B, out, bias, quant_state)
        else:
            # 使用4位整数矩阵向量乘法
            out = F.gemv_4bit(A, B.t(), out, state=quant_state)
            # 如果有偏置，则加上偏置
            if bias is not None:
                out += bias
            return out
    else:
        # 调用自定义的4位整数矩阵乘法函数
        return MatMul4Bit.apply(A, B, out, bias, quant_state)
```