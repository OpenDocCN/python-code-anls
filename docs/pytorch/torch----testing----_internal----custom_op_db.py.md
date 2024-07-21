# `.\pytorch\torch\testing\_internal\custom_op_db.py`

```
# mypy: allow-untyped-defs
# 引入 torch 库
import torch
# 引入 functools 库
import functools
# 从 torch.testing 中引入 make_tensor 函数
from torch.testing import make_tensor
# 从 torch.testing._internal.opinfo.core 中引入 OpInfo 和 SampleInput 类
from torch.testing._internal.opinfo.core import (
    OpInfo,
    SampleInput,
)
# 从 torch.testing._internal.common_dtype 中引入 all_types_and 函数
from torch.testing._internal.common_dtype import all_types_and
# 引入 numpy 库并重命名为 np
import numpy as np
# 从 torch.testing._internal.autograd_function_db 中引入一系列函数
from torch.testing._internal.autograd_function_db import (
    sample_inputs_numpy_cube,
    sample_inputs_numpy_mul,
    sample_inputs_numpy_mul_scalar,
    sample_inputs_numpy_sort,
    sample_inputs_numpy_take,
)
# 从 torch 中引入 Tensor 类
from torch import Tensor
# 从 torch.types 中引入 Number 类型
from torch.types import Number
# 引入所有 typing 模块中的类型定义
from typing import *  # noqa: F403

# Note: [custom op db]
#
# 这是一个自定义操作测试用例集合，以 OpInfo 形式编写，
# 使它们可以被 OpInfo-based 测试轻松消费，以验证子系统是否正确支持它们。

# 定义一个函数，将 torch 的 Tensor 对象转换为 numpy 数组
def to_numpy(tensor):
    return tensor.cpu().numpy()

# 定义一个自定义操作，名称为 "_torch_testing::numpy_cube"，不修改参数
@torch.library.custom_op("_torch_testing::numpy_cube", mutates_args=())
# 接受一个 Tensor 类型的参数 x，返回一个包含两个 Tensor 的元组
def numpy_cube(x: Tensor) -> Tuple[Tensor, Tensor]:
    # 将输入的 Tensor 转换为 numpy 数组
    x_np = to_numpy(x)
    # 计算 x_np 的立方和 3*x_np^2
    dx = torch.tensor(3 * x_np ** 2, device=x.device)
    # 返回 x_np 的立方和 dx
    return torch.tensor(x_np ** 3, device=x.device), dx

# 注册 numpy_cube 的假版本，接受 x 参数并返回它的克隆
@numpy_cube.register_fake
def _(x):
    return x.clone(), x.clone()

# 定义 numpy_cube 的上下文设置函数，接受 ctx、inputs 和 output 参数
def numpy_cube_setup_context(ctx, inputs, output):
    x, = inputs
    cube, dx = output
    # 将 x 和 dx 存储到 ctx 中，以备后续反向传播使用
    ctx.save_for_backward(x, dx)

# 定义 numpy_cube 的反向传播函数，接受 ctx、grad_out 和 grad_dx 参数
def numpy_cube_backward(ctx, grad_out, grad_dx):
    x, dx = ctx.saved_tensors
    # 计算关于 x 的梯度
    grad_x = numpy_mul(grad_out, dx) + 6 * numpy_mul(grad_dx, x)
    return grad_x

# 将 numpy_cube 的反向传播函数注册到 numpy_cube 上
numpy_cube.register_autograd(numpy_cube_backward, setup_context=numpy_cube_setup_context)

# 定义一个自定义操作，名称为 "_torch_testing::numpy_mul"，不修改参数
@torch.library.custom_op("_torch_testing::numpy_mul", mutates_args=())
# 接受两个 Tensor 类型的参数 x 和 y，返回一个 Tensor
def numpy_mul(x: Tensor, y: Tensor) -> Tensor:
    # 返回 x 和 y 的 numpy 数组乘积的 Tensor
    return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)

# 注册 numpy_mul 的假版本，接受 x 和 y 参数并返回它们的逐元素乘积
@numpy_mul.register_fake
def _(x, y):
    assert x.device == y.device
    return (x * y).contiguous()

# 定义 numpy_mul 的上下文设置函数，接受 ctx、inputs 和 output 参数
def numpy_mul_setup_context(ctx, inputs, output):
    # 将 inputs 中的所有参数存储到 ctx 中
    ctx.save_for_backward(*inputs)

# 定义 numpy_mul 的反向传播函数，接受 ctx 和 grad_out 参数
def numpy_mul_backward(ctx, grad_out):
    x, y = ctx.saved_tensors
    # 根据需要计算 x 和 y 的梯度
    grad_x = grad_out * y if ctx.needs_input_grad[0] else None
    grad_y = grad_out * x if ctx.needs_input_grad[1] else None
    return grad_x, grad_y

# 将 numpy_mul 的反向传播函数注册到 numpy_mul 上
numpy_mul.register_autograd(numpy_mul_backward, setup_context=numpy_mul_setup_context)

# 定义一个自定义操作，名称为 "_torch_testing::numpy_mul_scalar"，不修改参数
@torch.library.custom_op("_torch_testing::numpy_mul_scalar", mutates_args=())
# 接受一个 Tensor 类型的参数 x 和一个关键字参数 scalar，返回一个 Tensor
def numpy_mul_scalar(x: Tensor, *, scalar: float) -> Tensor:
    # 返回 x 的 numpy 数组乘以 scalar 的 Tensor
    return torch.tensor(to_numpy(x) * scalar, device=x.device)

# 注册 numpy_mul_scalar 的假版本，接受 x 和 scalar 参数并返回它们的乘积
@numpy_mul_scalar.register_fake
def _(x, *, scalar):
    return (x * scalar).contiguous()

# 定义 numpy_mul_scalar 的上下文设置函数，接受 ctx、inputs、keyword_only_inputs 和 output 参数
def numpy_mul_scalar_setup_context(ctx, inputs, keyword_only_inputs, output):
    # 将关键字参数 scalar 存储到 ctx 中
    ctx.scalar = keyword_only_inputs["scalar"]

# 定义 numpy_mul_scalar 的反向传播函数，接受 ctx 和 grad_out 参数
def numpy_mul_scalar_backward(ctx, grad_out):
    # 计算关于 x 的梯度
    grad_x = grad_out * ctx.scalar
    return grad_x

# 将 numpy_mul_scalar 的反向传播函数注册到 numpy_mul_scalar 上
numpy_mul_scalar.register_autograd(numpy_mul_scalar_backward, setup_context=numpy_mul_scalar_setup_context)

# 定义一个自定义操作，名称为 "_torch_testing::numpy_sort"，不修改参数
@torch.library.custom_op("_torch_testing::numpy_sort", mutates_args=())
# 接受一个 Tensor 类型的参数 x 和一个 int 类型的参数 dim，返回一个包含三个 Tensor 的元组
def numpy_sort(x: Tensor, dim: int) -> Tuple[Tensor, Tensor, Tensor]:
    # TODO: Add implementation for numpy_sort custom op
    pass  # Placeholder, actual implementation needed
    # 获取输入张量 x 的设备信息
    device = x.device
    # 将输入张量 x 转换为 NumPy 数组
    x = to_numpy(x)
    # 按照指定维度 dim 对 NumPy 数组 x 进行排序，并返回排序后的索引
    ind = np.argsort(x, axis=dim)
    # 对排序后的索引 ind 再次进行排序，以获得原始顺序的索引
    ind_inv = np.argsort(ind, axis=dim)
    # 根据排序后的索引 ind，在原始数组 x 上取值，得到排序后的结果
    result = np.take_along_axis(x, ind, axis=dim)
    # 返回排序后的结果张量、排序后的索引张量以及原始顺序的索引张量，所有张量的设备与输入张量 x 保持一致
    return (
        torch.tensor(result, device=device),
        torch.tensor(ind, device=device),
        torch.tensor(ind_inv, device=device),
    )
@numpy_sort.register_fake
def _(x, dim):
    return torch.empty_like(x), torch.empty_like(x, dtype=torch.long), torch.empty_like(x, dtype=torch.long)


# 注册一个名为 `numpy_sort` 的假函数，接受参数 `x` 和 `dim`
# 返回三个张量，分别是与 `x` 大小相同的空张量，以及两个与 `x` 相同大小且数据类型为长整型的空张量
def _(x, dim):
    return torch.empty_like(x), torch.empty_like(x, dtype=torch.long), torch.empty_like(x, dtype=torch.long)



def numpy_sort_setup_context(ctx, inputs, output):
    out, ind, ind_inv = output
    ctx.dim = inputs[1]
    ctx.save_for_backward(ind, ind_inv)
    ctx.mark_non_differentiable(ind, ind_inv)


# 设置 `numpy_sort` 的上下文环境
# 将输出的三个张量 `out`, `ind`, `ind_inv` 分别保存到上下文中
# 标记 `ind` 和 `ind_inv` 为不可微分
def numpy_sort_setup_context(ctx, inputs, output):
    out, ind, ind_inv = output
    ctx.dim = inputs[1]
    ctx.save_for_backward(ind, ind_inv)
    ctx.mark_non_differentiable(ind, ind_inv)



def numpy_sort_backward(ctx, grad_out, grad_ind, grad_ind_inv):
    ind, ind_inv = ctx.saved_tensors
    return numpy_take(grad_out, ind_inv, ind, ctx.dim), None


# 实现 `numpy_sort` 的反向传播函数
# 从上下文中读取保存的张量 `ind` 和 `ind_inv`
# 使用 `numpy_take` 函数计算 `grad_out` 关于 `ind_inv` 的梯度，并返回
def numpy_sort_backward(ctx, grad_out, grad_ind, grad_ind_inv):
    ind, ind_inv = ctx.saved_tensors
    return numpy_take(grad_out, ind_inv, ind, ctx.dim), None



numpy_sort.register_autograd(numpy_sort_backward, setup_context=numpy_sort_setup_context)


# 注册 `numpy_sort` 的自动求导函数为 `numpy_sort_backward`，设置上下文环境为 `numpy_sort_setup_context`
numpy_sort.register_autograd(numpy_sort_backward, setup_context=numpy_sort_setup_context)



@torch.library.custom_op("_torch_testing::numpy_take", mutates_args=())
def numpy_take(x: Tensor, ind: Tensor, ind_inv: Tensor, dim: int) -> Tensor:
    device = x.device
    x = to_numpy(x)
    ind = to_numpy(ind)
    return torch.tensor(np.take_along_axis(x, ind, dim), device=device)


# 注册名为 `_torch_testing::numpy_take` 的自定义操作
# 接受四个参数：`x`, `ind`, `ind_inv`, `dim`
# 返回经过处理的张量，首先将 `x`, `ind` 转换为 NumPy 数组，然后使用 `np.take_along_axis` 函数操作后返回
@torch.library.custom_op("_torch_testing::numpy_take", mutates_args=())
def numpy_take(x: Tensor, ind: Tensor, ind_inv: Tensor, dim: int) -> Tensor:
    device = x.device
    x = to_numpy(x)
    ind = to_numpy(ind)
    return torch.tensor(np.take_along_axis(x, ind, dim), device=device)



@numpy_take.register_fake
def _(x, ind, ind_inv, dim):
    assert x.device == ind.device
    assert x.device == ind_inv.device
    assert ind.dtype == torch.long
    assert ind_inv.dtype == torch.long
    return torch.empty_like(x)


# 注册一个假的 `numpy_take` 函数
# 接受四个参数 `x`, `ind`, `ind_inv`, `dim`
# 确保 `x`, `ind`, `ind_inv` 设备相同且数据类型为长整型
# 返回与 `x` 大小相同的空张量
@numpy_take.register_fake
def _(x, ind, ind_inv, dim):
    assert x.device == ind.device
    assert x.device == ind_inv.device
    assert ind.dtype == torch.long
    assert ind_inv.dtype == torch.long
    return torch.empty_like(x)



def numpy_take_setup_context(ctx, inputs, output):
    x, ind, ind_inv, dim = inputs
    ctx.dim = dim
    ctx.save_for_backward(ind, ind_inv)


# 设置 `numpy_take` 的上下文环境
# 将输入的 `ind` 和 `ind_inv` 保存到上下文中
def numpy_take_setup_context(ctx, inputs, output):
    x, ind, ind_inv, dim = inputs
    ctx.dim = dim
    ctx.save_for_backward(ind, ind_inv)



def numpy_take_backward(ctx, grad_out):
    ind, ind_inv = ctx.saved_tensors
    grad_x = numpy_take(grad_out, ind_inv, ind, ctx.dim)
    return grad_x, None, None, None


# 实现 `numpy_take` 的反向传播函数
# 从上下文中读取保存的张量 `ind` 和 `ind_inv`
# 使用 `numpy_take` 函数计算 `grad_out` 关于 `ind_inv` 的梯度，并返回
def numpy_take_backward(ctx, grad_out):
    ind, ind_inv = ctx.saved_tensors
    grad_x = numpy_take(grad_out, ind_inv, ind, ctx.dim)
    return grad_x, None, None, None



numpy_take.register_autograd(numpy_take_backward, setup_context=numpy_take_setup_context)


# 注册 `numpy_take` 的自动求导函数为 `numpy_take_backward`，设置上下文环境为 `numpy_take_setup_context`
numpy_take.register_autograd(numpy_take_backward, setup_context=numpy_take_setup_context)



@torch.library.custom_op("_torch_testing::numpy_nonzero", mutates_args=())
def numpy_nonzero(x: Tensor) -> Tensor:
    x_np = to_numpy(x)
    res = np.stack(np.nonzero(x_np), axis=1)
    if res.shape[0] <= 1:
        raise RuntimeError("not supported")
    return torch.tensor(res, device=x.device)


# 注册名为 `_torch_testing::numpy_nonzero` 的自定义操作
# 接受参数 `x`，返回经过处理的张量
# 首先将 `x` 转换为 NumPy 数组，然后使用 `np.nonzero` 函数找到非零元素的索引
# 将索引堆叠成二维数组并转换为张量返回，如果结果形状的第一维度小于等于1则抛出运行时错误
@torch.library.custom_op("_torch_testing::numpy_nonzero", mutates_args=())
def numpy_nonzero(x: Tensor) -> Tensor:
    x_np = to_numpy(x)
    res = np.stack(np.nonzero(x_np), axis=1)
    if res.shape[0] <= 1:
        raise RuntimeError("not supported")
    return torch.tensor(res, device=x.device)



@numpy_nonzero.register_fake
def _(x):
    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.create_unbacked_symint()
    shape = [i0, x.dim()]
    result = x.new_empty(shape, dtype=torch.long)
    return result


# 注册一个假的 `numpy_nonzero` 函数
# 接受参数 `x`
# 获取当前上下文环境，创建一个未支持的符号整数 `i0`
# 创建一个形状为 `[i0, x.dim()]` 的新空张量，数据类型为长整型，返回该张量
@numpy_nonzero.register_fake
def _(x):
    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.create_unbacked_symint()
    shape = [i0, x.dim()]
    result = x.new_empty(shape, dtype=torch.long)
    return result



def sample_inputs_numpy_nonzero(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shape = 10
    result = make_arg(shape, low=0.9, high=2)
    mask = make_tensor(shape, low=0, high=2, device=device, dtype=torch.long)
    with torch.no_grad():
        result *= mask

    yield SampleInput(result, args=())


# 定义一个生成 `numpy_nonzero` 函数输入样本的生成器函数
# 使用 `functools.partial` 创建 `make_arg` 函数，用于生成张量
# 创建形
# 注册自定义自动求导函数 `numpy_view_copy_backward` 到 `numpy_view_copy`
numpy_view_copy.register_autograd(numpy_view_copy_backward, setup_context=numpy_view_copy_setup_context)

# 定义生成带有 numpy_view_copy 操作的示例输入函数
def sample_inputs_numpy_view_copy(opinfo, device, dtype, requires_grad, **kwargs):
    # 创建一个生成张量的辅助函数 `make_arg`
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 创建一个张量 `result`，作为示例输入
    result = make_arg(2, 3, 4, low=0.9, high=2)
    # 生成一个带有 `numpy_view_copy` 操作的示例输入对象
    yield SampleInput(result, args=([2, 12],))

# 注册自定义操作 `_torch_testing::numpy_cat` 到 `numpy_cat`，不会改变参数
@torch.library.custom_op('_torch_testing::numpy_cat', mutates_args=())
def numpy_cat(xs: Sequence[Tensor], dim: int) -> Tensor:
    # 断言：确保输入的张量序列不为空
    assert len(xs) > 0
    # 断言：确保输入的所有张量在同一设备上
    assert all(x.device == xs[0].device for x in xs)
    # 断言：确保输入的所有张量具有相同的数据类型
    assert all(x.dtype == xs[0].dtype for x in xs)
    # 转换所有输入张量为 NumPy 数组
    np_xs = [to_numpy(x) for x in xs]
    # 使用 NumPy 进行沿指定维度拼接操作
    np_out = np.concatenate(np_xs, axis=dim)
    # 将结果转换为 PyTorch 张量并返回，指定设备为第一个输入张量的设备
    return torch.tensor(np_out, device=xs[0].device)

# 注册 `numpy_cat` 的伪装函数
@numpy_cat.register_fake
def _(xs, dim):
    # 断言：确保输入的张量序列不为空
    assert len(xs) > 0
    # 断言：确保输入的所有张量在同一设备上
    assert all(x.device == xs[0].device for x in xs)
    # 断言：确保输入的所有张量具有相同的数据类型
    assert all(x.dtype == xs[0].dtype for x in xs)
    # 返回 PyTorch 提供的张量拼接函数的结果
    return torch.cat(xs, dim=dim)

# 定义 `numpy_cat` 的上下文设置函数
def numpy_cat_setup_context(ctx, inputs, output):
    # 获取输入中的张量序列 `xs` 和维度 `dim`，并在上下文中存储维度大小列表和维度
    xs, dim = inputs
    ctx.dim_sizes = [x.shape[dim] for x in xs]
    ctx.dim = dim

# 定义 `numpy_cat` 的反向传播函数
def numpy_cat_backward(ctx, grad_out):
    # 从上下文中获取维度大小列表和维度
    dim_sizes = ctx.dim_sizes
    dim = ctx.dim
    # 计算分割点列表
    splits = list(np.cumsum(dim_sizes)[:-1])
    # 使用 `_torch_testing::numpy_split_copy` 自定义操作进行拆分和复制
    grad_xs = torch.ops._torch_testing.numpy_split_copy(grad_out, splits, dim)
    # 返回梯度张量列表和 `None`
    return grad_xs, None

# 注册自定义自动求导函数 `numpy_cat_backward` 到 `numpy_cat`
numpy_cat.register_autograd(numpy_cat_backward, setup_context=numpy_cat_setup_context)

# 定义生成带有 `numpy_cat` 操作的示例输入函数
def sample_inputs_numpy_cat(opinfo, device, dtype, requires_grad, **kwargs):
    # 创建一个生成张量的辅助函数 `make_arg`
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 创建示例输入张量 `r0`, `r1`, `r2`
    r0 = make_arg(2, 3, 4, low=0.9, high=2)
    r1 = make_arg(4, 3, 4, low=0.9, high=2)
    r2 = make_arg(5, 3, 4, low=0.9, high=2)
    # 生成带有 `numpy_cat` 操作的示例输入对象
    yield SampleInput([r0, r1, r2], args=(0,))

# 注册自定义操作 `_torch_testing::numpy_split_copy` 到 `numpy_split_copy`，不会改变参数
@torch.library.custom_op('_torch_testing::numpy_split_copy', mutates_args=())
def numpy_split_copy(x: Tensor, splits: Sequence[int], dim: int) -> List[Tensor]:
    # 将输入张量 `x` 转换为 NumPy 数组 `x_np`
    x_np = to_numpy(x)
    # 使用 NumPy 进行沿指定维度的分割操作
    arrs = np.split(x_np, splits, axis=dim)
    # 将 NumPy 数组列表转换为 PyTorch 张量列表并返回
    return [torch.tensor(arr, device=x.device, dtype=x.dtype) for arr in arrs]

# 注册 `numpy_split_copy` 的伪装函数
@numpy_split_copy.register_fake
def _(x, splits, dim):
    # 返回输入张量 `x` 按照指定维度 `dim` 进行分割后的克隆列表
    return [xi.clone() for xi in torch.tensor_split(x, splits, dim)]

# 定义 `numpy_split_copy` 的上下文设置函数
def numpy_split_copy_setup_context(ctx, inputs, output):
    # 获取输入中的张量 `x`、分割列表 `splits` 和维度 `dim`，并在上下文中存储维度
    _, _, dim = inputs
    ctx.dim = dim

# 定义 `numpy_split_copy` 的反向传播函数
def numpy_split_copy_backward(ctx, grad_out):
    # 从上下文中获取维度 `dim`
    dim = ctx.dim
    # 使用 `_torch_testing::numpy_cat` 自定义操作进行拼接
    result = torch.ops._torch_testing.numpy_cat(grad_out, dim=dim)
    # 返回结果和 `None`
    return result, None, None

# 注册自定义自动求导函数 `numpy_split_copy_backward` 到 `numpy_split_copy`
numpy_split_copy.register_autograd(numpy_split_copy_backward, setup_context=numpy_split_copy_setup_context)

# 定义生成带有 `numpy_split_copy` 操作的示例输入函数
def sample_inputs_numpy_split_copy(opinfo, device, dtype, requires_grad, **kwargs):
    # 创建一个生成张量的辅助函数 `make_arg`
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 创建示例输入张量 `x`
    x = make_arg(2, 9, low=0.9, high=2)
    # 生成带有 `numpy_split_copy` 操作的示例输入对象
    yield SampleInput(x, args=([1, 3, 6], 1))

# 注册自定义操作 `_torch_testing::numpy_split_copy_with_int` 到 `numpy_split_copy_with_int`，不会改变参数
# 使用 numpy 库将 PyTorch 张量转换为 NumPy 数组
def numpy_split_copy_with_int(x: Tensor, splits: Sequence[int], dim: int) -> Tuple[List[Tensor], int]:
    x_np = to_numpy(x)
    # 在指定维度上对 NumPy 数组进行分割
    arrs = np.split(x_np, splits, axis=dim)
    # 将分割后的 NumPy 数组转换回 PyTorch 张量，并指定设备和数据类型
    return [torch.tensor(arr, device=x.device, dtype=x.dtype) for arr in arrs], len(splits)

# 注册一个虚拟的 numpy_split_copy_with_int 函数
@numpy_split_copy_with_int.register_fake
def _(x, splits, dim):
    # 对输入张量进行分割并复制每个子张量
    return [xi.clone() for xi in torch.tensor_split(x, splits, dim)], len(splits)

# 设置 numpy_split_copy_with_int 函数的上下文环境
def numpy_split_copy_with_int_setup_context(ctx, inputs, output):
    # 提取输入中的维度信息并存储在上下文对象中
    _, _, dim = inputs
    ctx.dim = dim

# 对 numpy_split_copy_with_int 函数进行反向传播
def numpy_split_copy_with_int_backward(ctx, grad_out, _):
    # 调用 C++ 实现的 numpy_cat 函数进行梯度连接
    return torch.ops._torch_testing.numpy_cat(grad_out, dim=ctx.dim), None, None

# 注册 numpy_split_copy_with_int 函数的自动求导规则及其上下文设置
numpy_split_copy_with_int.register_autograd(
    numpy_split_copy_with_int_backward,
    setup_context=numpy_split_copy_with_int_setup_context)

# 注册一个名为 numpy_nms 的自定义 Torch 库操作
@torch.library.custom_op("_torch_testing::numpy_nms", mutates_args=())
def numpy_nms(boxes: Tensor, scores: Tensor, iou_threshold: Number) -> Tensor:
    # 检查输入张量的设备是否相同
    assert boxes.device == scores.device
    device = boxes.device

    # 将输入张量转换为 NumPy 数组
    boxes = to_numpy(boxes)
    scores = to_numpy(scores)

    N = boxes.shape[0]
    assert boxes.shape == (N, 4)
    assert scores.shape == (N,)

    # 提取边界框的坐标信息
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算每个边界框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 根据得分对边界框进行降序排序
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算 IoU（交并比）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU 小于等于阈值的边界框
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    # 将保留的边界框索引转换为 Torch 张量并返回
    result = torch.tensor(np.stack(keep), device=device)
    # 为了满足数据相关的条件 :(
    assert result.size(0) >= 2
    return result

# 注册一个虚拟的 numpy_nms 函数
@numpy_nms.register_fake
def _(boxes, scores, iou_threshold):
    # 检查输入张量的设备是否相同
    assert boxes.device == scores.device
    N = boxes.shape[0]
    assert boxes.shape == (N, 4)
    assert scores.shape == (N,)

    # 获取当前上下文环境并创建未支持的符号整数
    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.create_unbacked_symint()
    # 创建一个空的 Torch 张量作为结果
    result = boxes.new_empty([i0], dtype=torch.int64)
    return result

# 生成用于 numpy_nms 函数的样本输入
def sample_inputs_numpy_nms(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    N = 64
    xs = make_arg([N], low=0, high=28)
    dx = make_arg([N], low=0, high=4)
    ys = make_arg([N], low=0, high=28)
    dy = make_arg([N], low=0, high=4)
    # 创建包含边界框信息的输入张量
    boxes = torch.stack([xs, ys, xs + dx, ys + dy], dim=1).requires_grad_(requires_grad)
    # 创建包含得分信息的输入张量
    scores = make_arg([N], low=0, high=1, requires_grad=requires_grad)
    # 调用 make_arg 函数创建一个参数，生成的值在区间 [0, 1] 之间，返回其值作为 iou_threshold
    iou_threshold = make_arg([], low=0, high=1).item()

    # 使用 yield 语句生成一个 SampleInput 对象，传入 boxes 参数作为位置参数，scores 和 iou_threshold 作为关键字参数
    yield SampleInput(boxes, args=(scores, iou_threshold))
custom_op_db = [
    # 定义自定义操作信息对象，包括操作名称、操作函数、样本输入函数、数据类型、是否支持输出等信息
    OpInfo(
        'NumpyCubeCustomOp',  # 操作名称为 'NumpyCubeCustomOp'
        op=numpy_cube._opoverload,  # 使用 numpy_cube 模块的 _opoverload 函数作为操作函数
        sample_inputs_func=sample_inputs_numpy_cube,  # 使用 sample_inputs_numpy_cube 函数生成样本输入数据
        dtypes=all_types_and(torch.bool, torch.half),  # 支持的数据类型包括所有类型和 torch.bool 和 torch.half
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'NumpyMulCustomOp',  # 操作名称为 'NumpyMulCustomOp'
        op=numpy_mul._opoverload,  # 使用 numpy_mul 模块的 _opoverload 函数作为操作函数
        sample_inputs_func=sample_inputs_numpy_mul,  # 使用 sample_inputs_numpy_mul 函数生成样本输入数据
        dtypes=all_types_and(torch.bool, torch.half),  # 支持的数据类型包括所有类型和 torch.bool 和 torch.half
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'NumpyMulScalarCustomOp',  # 操作名称为 'NumpyMulScalarCustomOp'
        op=numpy_mul_scalar._opoverload,  # 使用 numpy_mul_scalar 模块的 _opoverload 函数作为操作函数
        sample_inputs_func=sample_inputs_numpy_mul_scalar,  # 使用 sample_inputs_numpy_mul_scalar 函数生成样本输入数据
        dtypes=all_types_and(torch.bool, torch.half),  # 支持的数据类型包括所有类型和 torch.bool 和 torch.half
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'NumpySortCustomOp',  # 操作名称为 'NumpySortCustomOp'
        op=numpy_sort._opoverload,  # 使用 numpy_sort 模块的 _opoverload 函数作为操作函数
        sample_inputs_func=sample_inputs_numpy_sort,  # 使用 sample_inputs_numpy_sort 函数生成样本输入数据
        dtypes=all_types_and(torch.bool, torch.half),  # 支持的数据类型包括所有类型和 torch.bool 和 torch.half
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'NumpyTakeCustomOp',  # 操作名称为 'NumpyTakeCustomOp'
        op=numpy_take._opoverload,  # 使用 numpy_take 模块的 _opoverload 函数作为操作函数
        sample_inputs_func=sample_inputs_numpy_take,  # 使用 sample_inputs_numpy_take 函数生成样本输入数据
        dtypes=all_types_and(torch.bool, torch.half),  # 支持的数据类型包括所有类型和 torch.bool 和 torch.half
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'NumpyNonzeroCustomOp',  # 操作名称为 'NumpyNonzeroCustomOp'
        op=numpy_nonzero._opoverload,  # 使用 numpy_nonzero 模块的 _opoverload 函数作为操作函数
        sample_inputs_func=sample_inputs_numpy_nonzero,  # 使用 sample_inputs_numpy_nonzero 函数生成样本输入数据
        dtypes=all_types_and(torch.bool, torch.half),  # 支持的数据类型包括所有类型和 torch.bool 和 torch.half
        supports_autograd=False,  # 不支持自动求导
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'NumpyNMSCustomOp',  # 操作名称为 'NumpyNMSCustomOp'
        op=torch.ops._torch_testing.numpy_nms,  # 使用 torch.ops._torch_testing.numpy_nms 函数作为操作函数
        sample_inputs_func=sample_inputs_numpy_nms,  # 使用 sample_inputs_numpy_nms 函数生成样本输入数据
        dtypes=all_types_and(torch.bool, torch.half),  # 支持的数据类型包括所有类型和 torch.bool 和 torch.half
        supports_autograd=False,  # 不支持自动求导
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'NumpyViewCopyCustomOp',  # 操作名称为 'NumpyViewCopyCustomOp'
        op=torch.ops._torch_testing.numpy_view_copy,  # 使用 torch.ops._torch_testing.numpy_view_copy 函数作为操作函数
        sample_inputs_func=sample_inputs_numpy_view_copy,  # 使用 sample_inputs_numpy_view_copy 函数生成样本输入数据
        dtypes=all_types_and(torch.bool, torch.half),  # 支持的数据类型包括所有类型和 torch.bool 和 torch.half
        supports_autograd=True,  # 支持自动求导
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'NumpyCatCustomOp',  # 操作名称为 'NumpyCatCustomOp'
        op=torch.ops._torch_testing.numpy_cat,  # 使用 torch.ops._torch_testing.numpy_cat 函数作为操作函数
        sample_inputs_func=sample_inputs_numpy_cat,  # 使用 sample_inputs_numpy_cat 函数生成样本输入数据
        dtypes=all_types_and(torch.bool, torch.half),  # 支持的数据类型包括所有类型和 torch.bool 和 torch.half
        supports_autograd=True,  # 支持自动求导
        check_batched_grad=False,  # 不检查批处理的梯度
        check_batched_gradgrad=False,  # 不检查批处理的梯度二阶导数
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'NumpySplitCopyCustomOp',  # 操作名称为 'NumpySplitCopyCustomOp'
        op=torch.ops._torch_testing.numpy_split_copy,  # 使用 torch.ops._torch_testing.numpy_split_copy 函数作为操作函数
        sample_inputs_func=sample_inputs_numpy_split_copy,  # 使用 sample_inputs_numpy_split_copy 函数生成样本输入数据
        dtypes=all_types_and(torch.bool, torch.half),  # 支持的数据类型包括所有类型和 torch.bool 和 torch.half
        supports_autograd=True,  # 支持自动求导
        check_batched_grad=False,  # 不检查批处理的梯度
        check_batched_gradgrad=False,  # 不检查批处理的梯度二阶导数
        supports_out=False,  # 不支持输出
    ),
]
    OpInfo(
        'NumpySplitCopyWithIntCustomOp',  # 自定义操作的名称
        op=torch.ops._torch_testing.numpy_split_copy_with_int,  # 操作的实际函数引用
        sample_inputs_func=sample_inputs_numpy_split_copy,  # 提供样本输入的函数引用
        dtypes=all_types_and(torch.bool, torch.half),  # 支持的数据类型，包括所有类型和布尔型和半精度浮点数
        gradcheck_wrapper=lambda op, *args, **kwargs: op(*args, **kwargs)[0],  # 梯度检查的包装函数，返回操作的第一个输出
        supports_autograd=True,  # 是否支持自动求导
        check_batched_grad=False,  # 是否检查批处理梯度
        check_batched_gradgrad=False,  # 是否检查批处理二阶梯度
        supports_out=False,  # 是否支持输出张量
    ),
# ==============================================================
# some mechanical test cases
# ==============================================================

# 创建一个名为 `lib` 的 Torch 库对象，表示用于测试的 `_torch_testing` 库，类型为 "FRAGMENT"
lib = torch.library.Library("_torch_testing", "FRAGMENT")  # noqa: TOR901

# 定义一个假的 Torch 库函数，函数名为 `source0`，参数为 `Tensor x`，返回值为 `Tensor`
lib.define("source0(Tensor x) -> Tensor")

# 使用装饰器注册假的 `_torch_testing::source0` 函数，实际实现为返回输入张量的克隆
@torch.library.register_fake("_torch_testing::source0", lib=lib)
def _(x):
    return x.clone()

# 定义一个假的 Torch 库函数，函数名为 `source1`，参数为 `Tensor x`，返回值为 `Tensor`
lib.define("source1(Tensor x) -> Tensor")

# 定义一个 Python 函数 `source1_fake`，实现为返回输入张量的克隆
def source1_fake(x):
    return x.clone()

# 使用 Torch 的注册方法注册假的 `_torch_testing::source1` 函数，实际实现为 `source1_fake`
torch.library.register_fake("_torch_testing::source1", source1_fake, lib=lib)

# 定义一个假的 Torch 库函数，函数名为 `source2`，参数为 `Tensor x`，返回值为 `Tensor`
lib.define("source2(Tensor x) -> Tensor")

# 使用装饰器注册假的 `_torch_testing::source2` 函数，实际实现为返回输入张量的克隆
@torch.library.register_fake("_torch_testing::source2", lib=lib)
def _(x):
    return x.clone()

# 定义一个假的 Torch 库函数，函数名为 `source3`，参数为 `Tensor x`，返回值为 `Tensor`
lib.define("source3(Tensor x) -> Tensor")

# 定义一个 Python 函数 `source3_fake`，实现为返回输入张量的克隆
def source3_fake(x):
    return x.clone()

# 使用 Torch 的注册方法注册假的 `_torch_testing::source3` 函数，实际实现为 `source3_fake`
torch.library.register_fake("_torch_testing::source3", source3_fake, lib=lib)

# 使用装饰器定义一个自定义操作 `_torch_testing::source4`，该操作不会修改参数
@torch.library.custom_op("_torch_testing::source4", mutates_args=())
def source4(x: Tensor) -> Tensor:
    return x.clone()

# 使用装饰器注册 `_torch_testing::source4` 的假实现，实际实现为返回输入张量的克隆
@source4.register_fake
def _(x):
    return x.clone()

# 使用装饰器定义一个自定义操作 `_torch_testing::source5`，该操作不会修改参数
@torch.library.custom_op("_torch_testing::source5", mutates_args=())
def source5(x: Tensor) -> Tensor:
    return x.clone()

# 定义一个 Python 函数 `source5_fake`，实现为返回输入张量的克隆
def source5_fake(x):
    return x.clone()

# 使用 Python 的方法注册 `source5_fake` 作为 `_torch_testing::source5` 的假实现
source5.register_fake(source5_fake)
```