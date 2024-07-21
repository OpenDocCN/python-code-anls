# `.\pytorch\torch\autograd\_functions\utils.py`

```py
# mypy: allow-untyped-defs
# 引入操作符模块，用于进行 reduce 操作
import operator
# 从 functools 模块中引入 reduce 函数
from functools import reduce


# 如果 tensor 的大小与给定的 size 相同且 check_same_size 为 True，则返回 tensor 自身；
# 否则，返回 tensor 的连续视图，大小为 size
def maybe_view(tensor, size, check_same_size=True):
    if check_same_size and tensor.size() == size:
        return tensor
    return tensor.contiguous().view(size)


# 如果 tensor 的大小与给定的 old_size 相同且 check_same_size 为 True，则返回 tensor 自身；
# 否则，按照可能的方式进行维度扩展或收缩，以使其与 old_size 相匹配
def maybe_unexpand(tensor, old_size, check_same_size=True):
    if check_same_size and tensor.size() == old_size:
        return tensor
    num_unsqueezed = tensor.dim() - len(old_size)
    expanded_dims = [
        dim
        for dim, (expanded, original) in enumerate(
            zip(tensor.size()[num_unsqueezed:], old_size)
        )
        if expanded != original
    ]

    for _ in range(num_unsqueezed):
        tensor = tensor.sum(0, keepdim=False)
    for dim in expanded_dims:
        tensor = tensor.sum(dim, keepdim=True)
    return tensor


# 检查操作是否支持广播，并且检查 ONNX 是否支持该广播方式。
# 如果 dims1 和 dims2 的维度不同，则 broadcast 为 True。
# 我们始终假设 dims1 和 dims2 的组合是可广播的。
# ONNX 支持以下几种广播方式：
#     1) dims2 中只有一个元素，例如 dims2 = [1, 1]
#     2) dims2 是 dims1 的后缀，例如 dims1 = [2, 3, 4]，dims2 = [3, 4]
# 更多细节可以参考：https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
def check_onnx_broadcast(dims1, dims2):
    broadcast = False
    supported = True
    len1 = len(dims1)
    len2 = len(dims2)
    numel1 = reduce(operator.mul, dims1)  # 计算 dims1 的元素总数
    numel2 = reduce(operator.mul, dims2)  # 计算 dims2 的元素总数

    if len1 < len2:
        broadcast = True
        if numel2 != 1:
            supported = False
    elif len1 > len2:
        broadcast = True
        if numel2 != 1 and dims1[len1 - len2 :] != dims2:
            supported = False
    else:
        if dims1 != dims2:
            broadcast = True
            if numel2 != 1:
                supported = False

    if not supported:
        raise ValueError(
            f"Numpy style broadcasting is not supported in ONNX. Input dims are: {dims1}, {dims2}"
        )

    return broadcast
```