# `.\pytorch\torch\distributions\utils.py`

```
# 引入允许未类型化定义的标记，用于类型检查
# 从 functools 模块导入 update_wrapper 函数，用于更新函数属性
# 从 numbers 模块导入 Number 类型，用于处理数字类型
# 从 typing 模块导入 Any 和 Dict 类型，用于类型标注

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数式接口
from torch.overrides import is_tensor_like  # 从 torch.overrides 模块导入 is_tensor_like 函数

euler_constant = 0.57721566490153286060  # 定义欧拉常数

__all__ = [  # 定义模块中对外暴露的所有符号列表
    "broadcast_all",
    "logits_to_probs",
    "clamp_probs",
    "probs_to_logits",
    "lazy_property",
    "tril_matrix_to_vec",
    "vec_to_tril_matrix",
]


def broadcast_all(*values):
    r"""
    给定一组值（可能包含数字），返回一个列表，其中每个值都根据以下规则进行广播：
      - `torch.*Tensor` 实例按照 :ref:`_broadcasting-semantics` 进行广播。
      - numbers.Number 实例（标量）被提升为与传递给 `values` 的第一个张量相同大小和类型的张量。
        如果所有值都是标量，则它们被提升为标量张量。

    Args:
        values (list of `numbers.Number`, `torch.*Tensor` 或实现 __torch_function__ 的对象)

    Raises:
        ValueError: 如果任何值不是 `numbers.Number` 实例、`torch.*Tensor` 实例或实现 __torch_function__ 的实例
    """
    if not all(is_tensor_like(v) or isinstance(v, Number) for v in values):
        raise ValueError(
            "Input arguments must all be instances of numbers.Number, "
            "torch.Tensor or objects implementing __torch_function__."
        )
    if not all(is_tensor_like(v) for v in values):
        options: Dict[str, Any] = dict(dtype=torch.get_default_dtype())
        for value in values:
            if isinstance(value, torch.Tensor):
                options = dict(dtype=value.dtype, device=value.device)
                break
        new_values = [
            v if is_tensor_like(v) else torch.tensor(v, **options) for v in values
        ]
        return torch.broadcast_tensors(*new_values)
    return torch.broadcast_tensors(*values)


def _standard_normal(shape, dtype, device):
    """
    生成一个指定形状、数据类型和设备的标准正态分布张量。

    Args:
        shape (tuple): 张量的形状。
        dtype (torch.dtype): 张量的数据类型。
        device (torch.device): 张量的设备。

    Returns:
        torch.Tensor: 指定形状、数据类型和设备的标准正态分布张量。
    """
    if torch._C._get_tracing_state():
        # [JIT WORKAROUND] 缺少对 .normal_() 方法的支持
        return torch.normal(
            torch.zeros(shape, dtype=dtype, device=device),
            torch.ones(shape, dtype=dtype, device=device),
        )
    return torch.empty(shape, dtype=dtype, device=device).normal_()


def _sum_rightmost(value, dim):
    r"""
    对给定张量的右侧 `dim` 个维度进行求和。

    Args:
        value (Tensor): 至少具有 `.dim()` 为 `dim` 的张量。
        dim (int): 要求和的右侧维度数量。

    Returns:
        Tensor: 对指定维度求和后的张量。
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def logits_to_probs(logits, is_binary=False):
    r"""
    将 logits 张量转换为概率张量。对于二元情况，每个值表示对数几率，而对于
    # 如果 is_binary 为真，则对 logits 进行 sigmoid 函数处理，将其转换为概率值
    if is_binary:
        return torch.sigmoid(logits)
    # 如果 is_binary 为假，则对 logits 进行 softmax 函数处理，计算在最后一个维度上的概率分布
    return F.softmax(logits, dim=-1)
# 用于限制概率值在开区间 (0, 1) 内的函数

def clamp_probs(probs):
    """Clamps the probabilities to be in the open interval `(0, 1)`.

    The probabilities would be clamped between `eps` and `1 - eps`,
    and `eps` would be the smallest representable positive number for the input data type.

    Args:
        probs (Tensor): A tensor of probabilities.

    Returns:
        Tensor: The clamped probabilities.

    Examples:
        >>> probs = torch.tensor([0.0, 0.5, 1.0])
        >>> clamp_probs(probs)
        tensor([1.1921e-07, 5.0000e-01, 1.0000e+00])

        >>> probs = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        >>> clamp_probs(probs)
        tensor([2.2204e-16, 5.0000e-01, 1.0000e+00], dtype=torch.float64)

    """
    eps = torch.finfo(probs.dtype).eps  # 获取概率张量数据类型的最小正数
    return probs.clamp(min=eps, max=1 - eps)  # 将概率值限制在 (eps, 1-eps) 区间内


# 将概率张量转换为 logits（对数几率）的函数
def probs_to_logits(probs, is_binary=False):
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)  # 对概率进行限制
    if is_binary:
        return torch.log(ps_clamped) - torch.log1p(-ps_clamped)  # 二元情况下的 logits 转换
    return torch.log(ps_clamped)  # 多维情况下的 logits 转换


# 用于实现延迟加载属性的装饰器类
class lazy_property:
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """

    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)  # type:ignore[arg-type]

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return _lazy_property_and_property(self.wrapped)
        with torch.enable_grad():
            value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


# 用于兼容多种属性表现形式的类
class _lazy_property_and_property(lazy_property, property):
    """We want lazy properties to look like multiple things.

    * property when Sphinx autodoc looks
    * lazy_property when Distribution validate_args looks
    """

    def __init__(self, wrapped):
        property.__init__(self, wrapped)


# 将下三角矩阵或批量矩阵转换为向量的函数
def tril_matrix_to_vec(mat: torch.Tensor, diag: int = 0) -> torch.Tensor:
    r"""
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector
    which comprises of lower triangular elements from the matrix in row order.
    """
    n = mat.shape[-1]  # 获取矩阵的维度
    if not torch._C._get_tracing_state() and (diag < -n or diag >= n):
        raise ValueError(f"diag ({diag}) provided is outside [{-n}, {n-1}].")
    arange = torch.arange(n, device=mat.device)  # 创建设备上的索引范围
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)  # 创建下三角掩码
    vec = mat[..., tril_mask]  # 应用掩码获取向量
    return vec


# 将向量转换为下三角矩阵或批量矩阵的函数
def vec_to_tril_matrix(vec: torch.Tensor, diag: int = 0) -> torch.Tensor:
    r"""
    Convert a vector into a `D x D` matrix or a batch of matrices with
    the vector elements placed in the lower triangular part of the matrix in row order.
    """
    """
    Convert a vector or a batch of vectors into a batched `D x D`
    lower triangular matrix containing elements from the vector in row order.
    """
    # 计算矩阵维度 D，满足方程 D**2 + (1+2*diag)*D - |diag| * (diag+1) - 2*vec.shape[-1] = 0 的正根
    n = (
        -(1 + 2 * diag)
        + ((1 + 2 * diag) ** 2 + 8 * vec.shape[-1] + 4 * abs(diag) * (diag + 1)) ** 0.5
    ) / 2
    # 机器精度
    eps = torch.finfo(vec.dtype).eps
    # 如果不是在跟踪状态下，并且 n 的四舍五入值与 n 的差大于机器精度，则抛出异常
    if not torch._C._get_tracing_state() and (round(n) - n > eps):
        raise ValueError(
            f"The size of last dimension is {vec.shape[-1]} which cannot be expressed as "
            + "the lower triangular part of a square D x D matrix."
        )
    # 若 n 是 torch.Tensor 类型则取其四舍五入整数值，否则直接四舍五入
    n = round(n.item()) if isinstance(n, torch.Tensor) else round(n)
    # 创建一个全零矩阵，形状为 vec 的前 n-1 维加上 (n, n)
    mat = vec.new_zeros(vec.shape[:-1] + torch.Size((n, n)))
    # 创建一个设备相关的整数张量 arange，值为 0 到 n-1
    arange = torch.arange(n, device=vec.device)
    # 创建一个下三角掩码，值为 True 的位置代表下三角部分（包括对角线和 diag 外的下三角）
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    # 将 vec 的值填充到 mat 的下三角位置
    mat[..., tril_mask] = vec
    # 返回结果矩阵 mat
    return mat
```