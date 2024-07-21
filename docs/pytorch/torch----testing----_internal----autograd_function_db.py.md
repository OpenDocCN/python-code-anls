# `.\pytorch\torch\testing\_internal\autograd_function_db.py`

```py
# mypy: ignore-errors

# 导入 PyTorch 库
import torch
# 导入 functools 模块中的 partial 函数
from functools import partial
# 从 torch.testing 模块导入 make_tensor 函数
from torch.testing import make_tensor
# 从 torch.testing._internal.opinfo.core 导入 OpInfo 和 SampleInput 类
from torch.testing._internal.opinfo.core import (
    OpInfo,
    SampleInput,
)
# 从 torch.testing._internal.common_dtype 导入 all_types_and 函数
from torch.testing._internal.common_dtype import all_types_and
# 导入 numpy 库并使用 np 别名
import numpy as np

# Note: [autograd.Function db]
#
# 这是一组 autograd.Function 的测试用例，以 OpInfo 的形式编写，
# 这样可以很容易地被 OpInfo-based 测试消费，以检查一个子系统是否支持 autograd.Function。
#
# Axes:
# - saves {output, input, intermediate, non-tensor}
# - {inputs, output} x {single tensor, tensors, arbitrary objects}
# - Uses {mark_dirty, mark_non_differentiable, once_differentiable}


# 将 PyTorch 张量转换为 NumPy 数组的函数
def to_numpy(tensor):
    return tensor.cpu().numpy()


# 自定义的 NumpyCube 类，继承自 torch.autograd.Function
class NumpyCube(torch.autograd.Function):
    @staticmethod
    def forward(input):
        # 将输入张量转换为 NumPy 数组
        input_np = to_numpy(input)
        # 计算输入的立方和其导数
        dinput = torch.tensor(3 * input_np ** 2, device=input.device)
        return torch.tensor(input_np ** 3, device=input.device), dinput

    @staticmethod
    def setup_context(ctx, inputs, output):
        # 保存反向传播所需的张量
        ctx.save_for_backward(inputs[0], output[1])
        ctx.save_for_forward(inputs[0], output[1])

    @staticmethod
    def backward(ctx, grad_output, grad_saved):
        # 从上下文中获取保存的张量
        input, dinput = ctx.saved_tensors
        # 计算反向传播梯度
        return NumpyMul.apply(grad_output, dinput) + 6 * NumpyMul.apply(grad_saved, input)

    @staticmethod
    def vmap(info, in_dims, input):
        # 对输入进行 vmap 操作
        result = NumpyCube.apply(input)
        return result, (in_dims[0], in_dims[0])

    @staticmethod
    def jvp(ctx, input_tangent):
        # 计算 Jacobian 向量积
        input, dinput = ctx.saved_tensors
        return NumpyMul.apply(input_tangent, dinput), 6 * NumpyMul.apply(input_tangent, input)


# 自定义的 CubeGenVmap 类，继承自 torch.autograd.Function
class CubeGenVmap(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        # 计算输入的立方和导数
        return x ** 3, 3 * x ** 2

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # 保存反向传播所需的张量
        ctx.save_for_backward(inputs[0], outputs[1])
        ctx.save_for_forward(inputs[0], outputs[1])

    @staticmethod
    def backward(ctx, grad_output, grad_saved):
        # 从上下文中获取保存的张量，并计算反向传播梯度
        input, dinput = ctx.saved_tensors
        result = grad_output * dinput + 6 * dinput
        return result

    @staticmethod
    def jvp(ctx, input_tangent):
        # 计算 Jacobian 向量积
        input, dinput = ctx.saved_tensors
        return MulGenVmap.apply(input_tangent, dinput), 6 * NumpyMul.apply(input_tangent, input)


# 生成 NumpyCube 的输入样本函数
def sample_inputs_numpy_cube(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg(1, low=0.8, high=2), args=())


# 自定义的 NumpyCubeNotComposable 类，继承自 torch.autograd.Function
class NumpyCubeNotComposable(torch.autograd.Function):
    @staticmethod
    def forward(input):
        # 将输入张量转换为 NumPy 数组并返回
        input_np = to_numpy(input)
        return torch.tensor(input_np ** 3, device=input.device), input_np

    @staticmethod
    def setup_context(ctx, inputs, output):
        # 保存额外的上下文信息
        _, input_np = output
        ctx.input_np = input_np
        ctx.device = inputs[0].device
    # 定义一个静态方法，用于计算反向传播时的梯度
    @staticmethod
    # 将该方法标记为PyTorch自动求导的可执行函数，确保仅调用一次
    @torch.autograd.function.once_differentiable
    # 定义反向传播函数，接收上下文对象ctx、输出梯度grad_output和保存的梯度grad_saved作为参数
    def backward(ctx, grad_output, grad_saved):
        # 计算输入数据的平方乘以3的结果，并转换为NumPy数组
        result_np = 3 * (ctx.input_np ** 2)
        # 将结果转换为PyTorch张量，设备与上下文对象ctx相同
        return torch.tensor(result_np, device=ctx.device)
class NumpyMul(torch.autograd.Function):
    # 定义静态方法 forward，接收输入 x 和 y，返回它们的乘积
    @staticmethod
    def forward(x, y):
        return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)

    # 定义静态方法 setup_context，保存输入和输出的上下文信息
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)

    # 定义静态方法 backward，接收上下文 ctx 和梯度 grad_output，计算 x 和 y 的梯度
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        gx = None
        if ctx.needs_input_grad[0]:
            gx = NumpyMul.apply(grad_output, y)
        gy = None
        if ctx.needs_input_grad[1]:
            gy = NumpyMul.apply(grad_output, x)
        return gx, gy

    # 定义静态方法 vmap，对输入 x 和 y 进行向量化映射处理
    @staticmethod
    def vmap(info, in_dims, x, y):
        x_bdim, y_bdim = in_dims
        x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
        y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
        result = NumpyMul.apply(x, y)
        result = result.movedim(-1, 0)
        return result, 0

    # 定义静态方法 jvp，接收上下文 ctx、x 和 y 的切线，返回 x 和 y 的切线对应的 Jacobian 向量积
    @staticmethod
    def jvp(ctx, x_tangent, y_tangent):
        x, y = ctx.saved_tensors
        return x_tangent * y + y_tangent * x


def sample_inputs_numpy_mul(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 生成一个输入样本，用于 NumpyMul 操作，包括广播处理
    yield SampleInput(make_arg(4, low=0.9, high=2), args=(make_arg(3, 4, low=0.9, high=2),))


def sample_inputs_numpy_mul_scalar(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 生成一个输入样本，用于 NumpyMul 操作，传入标量参数
    yield SampleInput(make_arg(4, low=0.9, high=2), args=(), kwargs={"scalar": 3.14})


class MulGenVmap(torch.autograd.Function):
    generate_vmap_rule = True

    # 定义静态方法 forward，接收输入 x 和 y，返回它们的乘积
    @staticmethod
    def forward(x, y):
        return x * y

    # 定义静态方法 setup_context，保存输入和输出的上下文信息
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)

    # 定义静态方法 backward，接收上下文 ctx 和梯度 grad_output，计算 x 和 y 的梯度
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        gx = None
        if ctx.needs_input_grad[0]:
            gx = MulGenVmap.apply(grad_output, y)
        gy = None
        if ctx.needs_input_grad[1]:
            gy = MulGenVmap.apply(grad_output, x)
        return gx, gy

    # 定义静态方法 jvp，接收上下文 ctx、x 和 y 的切线，返回 x 和 y 的切线对应的 Jacobian 向量积
    @staticmethod
    def jvp(ctx, x_tangent, y_tangent):
        x, y = ctx.saved_tensors
        return x_tangent * y + y_tangent * x


class NumpyExp_(torch.autograd.Function):
    # 定义静态方法 forward，接收输入 x，计算其指数，并返回结果
    @staticmethod
    def forward(x):
        x_np = to_numpy(x)
        np.exp(x_np, x_np)
        return x

    # 定义静态方法 setup_context，保存输入和输出的上下文信息，并标记输入 x 为脏（dirty）
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.mark_dirty(x)
        ctx.save_for_backward(output)
        ctx.save_for_forward(output)

    # 定义静态方法 backward，接收上下文 ctx 和梯度 grad_output，计算 x 的梯度
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return NumpyMul.apply(grad_output, output)

    # 定义静态方法 vmap，对输入 x 进行向量化映射处理
    @staticmethod
    def vmap(info, in_dims, x):
        NumpyExp_.apply(x)
        return x, in_dims[0]
    def jvp(ctx, x_tangent):
        # 定义一个 Jacobian-Vector Product (JVP) 的函数，接受上下文 ctx 和关于输入 x 的切线 x_tangent 作为参数
        # 从上下文中获取保存的张量 output
        output, = ctx.saved_tensors
        # 将 x_tangent 与 output 相乘，即计算 JVP
        x_tangent.mul_(output)
        # 返回计算得到的 x_tangent，作为 JVP 的结果
        return x_tangent
class NumpySort(torch.autograd.Function):
    @staticmethod
    def forward(x, dim):
        # 保存当前设备信息
        device = x.device
        # 将输入张量转换为 NumPy 数组
        x = to_numpy(x)
        # 按照指定维度对数组进行排序，返回排序后的索引
        ind = np.argsort(x, axis=dim)
        # 对排序后的索引再次排序，以便获取原始顺序到排序后顺序的映射
        ind_inv = np.argsort(ind, axis=dim)
        # 按照排序后的索引重新排列原始数组，得到排序后的结果
        result = np.take_along_axis(x, ind, axis=dim)
        # 将 NumPy 数组转换为 Torch 张量，并返回结果
        return (
            torch.tensor(x, device=device),
            torch.tensor(ind, device=device),
            torch.tensor(ind_inv, device=device),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        # 获取输入和输出
        x, dim = inputs
        _, ind, ind_inv = output
        # 标记不需要求导的张量
        ctx.mark_non_differentiable(ind, ind_inv)
        # 保存反向传播所需的张量
        ctx.save_for_backward(ind, ind_inv)
        # 保存前向传播所需的张量
        ctx.save_for_forward(ind, ind_inv)
        # 保存维度信息
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output, _0, _1):
        # 获取保存的张量
        ind, ind_inv = ctx.saved_tensors
        # 调用自定义的 NumpyTake 函数进行反向传播
        return NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim), None

    @staticmethod
    def vmap(info, in_dims, x, dim):
        # 获取输入张量的批次维度和排序的维度
        x_bdim, _ = in_dims
        # 将输入张量的批次维度移动到第一维度
        x = x.movedim(x_bdim, 0)
        # 包装排序的维度，处理负数索引
        dim = dim if dim >= 0 else dim + x.dim() - 1
        # 调用 NumpySort 的 apply 方法进行排序操作，并返回结果
        return NumpySort.apply(x, dim + 1), (0, 0, 0)

    @staticmethod
    def jvp(ctx, x_tangent, _):
        # 获取保存的张量
        ind, ind_inv = ctx.saved_tensors
        # 调用自定义的 NumpyTake 函数进行 Jacobi 向量积运算
        return NumpyTake.apply(x_tangent, ind, ind_inv, ctx.dim), None, None

class SortGenVmap(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x, dim):
        # 获取当前设备信息
        device = x.device
        # 对输入张量按指定维度进行排序，并返回排序后的索引
        ind = torch.argsort(x, dim=dim)
        # 对排序后的索引再次排序，以获取原始顺序到排序后顺序的映射
        ind_inv = torch.argsort(ind, axis=dim)
        # 按照排序后的索引重新排列原始张量，得到排序后的结果
        result = torch.take_along_dim(x, ind, dim=dim)
        # 返回排序后的结果张量及相关索引
        return result, ind, ind_inv

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # 获取输入和输出
        x, dim = inputs
        _, ind, ind_inv = outputs
        # 标记不需要求导的张量
        ctx.mark_non_differentiable(ind, ind_inv)
        # 保存反向传播所需的张量
        ctx.save_for_backward(ind, ind_inv)
        # 保存前向传播所需的张量
        ctx.save_for_forward(ind, ind_inv)
        # 保存维度信息
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output, _0, _1):
        # 获取保存的张量
        ind, ind_inv = ctx.saved_tensors
        # 调用自定义的 TakeGenVmap 函数进行反向传播
        return TakeGenVmap.apply(grad_output, ind_inv, ind, ctx.dim), None

    @staticmethod
    def jvp(ctx, x_tangent, _):
        # 获取保存的张量
        ind, ind_inv = ctx.saved_tensors
        # 调用自定义的 TakeGenVmap 函数进行 Jacobi 向量积运算
        return TakeGenVmap.apply(x_tangent, ind, ind_inv, ctx.dim), None, None


def sample_inputs_numpy_sort(opinfo, device, dtype, requires_grad, **kwargs):
    # 部分函数的封装
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 生成示例输入并返回
    yield SampleInput(make_arg(3, 5), args=(1,))


def sample_inputs_numpy_take(opinfo, device, dtype, requires_grad, **kwargs):
    # 部分函数的封装
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 生成示例输入并返回
    tensor = make_arg(3, 5)
    dim = 1
    # 调用 NumpySort 的 apply 方法进行排序操作，并返回结果
    _, ind, ind_inv = NumpySort.apply(tensor, 1)
    yield SampleInput(tensor, args=(ind, ind_inv, dim))


class NumpyTake(torch.autograd.Function):
    @staticmethod
    # 定义一个静态方法，用于执行张量 x 在指定维度 dim 上的索引操作
    def forward(x, ind, ind_inv, dim):
        # 获取张量 x 的设备信息
        device = x.device
        # 将张量 x 转换为 numpy 数组
        x = to_numpy(x)
        # 将 ind 转换为 numpy 数组
        ind = to_numpy(ind)
        # 使用 numpy 的 take_along_axis 函数在维度 dim 上取出对应索引的元素，返回一个张量，并指定设备
        return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

    @staticmethod
    # 静态方法：设置计算图的上下文，保存输入张量中的 ind 和 ind_inv，并设置 dim 属性
    def setup_context(ctx, inputs, output):
        x, ind, ind_inv, dim = inputs
        # 保存 ind 和 ind_inv 到计算图的上下文中，用于反向传播
        ctx.save_for_backward(ind, ind_inv)
        # 保存 ind 和 ind_inv 到计算图的上下文中，用于前向传播
        ctx.save_for_forward(ind, ind_inv)
        # 设置 ctx 的 dim 属性为输入的 dim
        ctx.dim = dim

    @staticmethod
    # 静态方法：反向传播函数，根据保存在 ctx 中的 ind 和 ind_inv 进行 NumpyTake 操作的反向传播
    def backward(ctx, grad_output):
        ind, ind_inv = ctx.saved_tensors
        # 调用 NumpyTake 的 apply 方法进行反向传播计算
        result = NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim)
        # 返回反向传播的结果，后面三个 None 是因为输入的不需要梯度传播
        return result, None, None, None

    @staticmethod
    # 静态方法：对一个批次的数据进行映射操作，扩展维度，并执行 NumpyTake 操作
    def vmap(info, in_dims, x, ind, ind_inv, dim):
        x_bdim, ind_bdim, ind_inv_bdim, _ = in_dims

        # 根据输入的 x_bdim 是否为 None，决定如何处理 dim 的包装
        logical_dim = x.dim() if x_bdim is None else x_bdim - 1
        dim = dim if dim >= 0 else dim + logical_dim

        # 定义一个函数，根据输入的 x_bdim 扩展维度
        def expand_bdim(x, x_bdim):
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)

        # 对输入的 x、ind、ind_inv 进行扩展维度操作
        x = expand_bdim(x, x_bdim)
        ind = expand_bdim(ind, ind_bdim)
        ind_inv = expand_bdim(ind_inv, ind_inv_bdim)

        # 调用 NumpyTake 的 apply 方法执行在 dim + 1 维度上的索引操作，并返回结果和一个 0
        return NumpyTake.apply(x, ind, ind_inv, dim + 1), 0

    @staticmethod
    # 静态方法：Jacobian-Vector Product (JVP) 的计算函数，根据保存在 ctx 中的 ind 和 ind_inv 进行 NumpyTake 操作
    def jvp(ctx, x_tangent, ind_tangent, ind_inv_tangent, _):
        # 断言 ind_tangent 和 ind_inv_tangent 必须为 None，因为这里不需要对索引进行梯度传播
        assert ind_tangent is None
        assert ind_inv_tangent is None
        # 获取保存在 ctx 中的 ind 和 ind_inv
        ind, ind_inv = ctx.saved_tensors
        # 调用 NumpyTake 的 apply 方法执行 JVP 计算
        return NumpyTake.apply(x_tangent, ind, ind_inv, ctx.dim)
class TakeGenVmap(torch.autograd.Function):
    # 定义一个标志，表示生成 vmap 规则
    generate_vmap_rule = True

    @staticmethod
    # 前向传播函数，用于取出沿指定维度的元素
    def forward(x, ind, ind_inv, dim):
        return torch.take_along_dim(x, ind, dim)

    @staticmethod
    # 设置上下文的静态方法，用于保存反向传播所需的数据和维度信息
    def setup_context(ctx, inputs, outputs):
        x, ind, ind_inv, dim = inputs
        ctx.save_for_backward(ind, ind_inv)
        ctx.save_for_forward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    # 反向传播的静态方法，计算梯度
    def backward(ctx, grad_output):
        ind, ind_inv = ctx.saved_tensors
        result = TakeGenVmap.apply(grad_output, ind_inv, ind, ctx.dim)
        return result, None, None, None

    @staticmethod
    # 对 JVP（雅可比向量积）的静态方法，用于计算对输入的导数
    def jvp(ctx, x_tangent, ind_tangent, ind_inv_tangent, _):
        ind, ind_inv = ctx.saved_tensors
        return TakeGenVmap.apply(x_tangent, ind, ind_inv, ctx.dim)


class Select(torch.autograd.Function):
    @staticmethod
    # 前向传播函数，根据索引选择张量中的元素
    def forward(x, idx):
        return x[idx]

    @staticmethod
    # 设置上下文的静态方法，用于保存前向传播所需的输入形状和索引
    def setup_context(ctx, inputs, output):
        x, idx = inputs
        ctx.x_shape = x.shape
        ctx.idx = idx

    @staticmethod
    # 反向传播的静态方法，计算梯度
    def backward(ctx, grad_output):
        result = grad_output.new_zeros(ctx.x_shape)
        result[ctx.idx] = grad_output
        return result, None

    @staticmethod
    # 对 VMap 的静态方法，用于在批处理中应用选择操作
    def vmap(info, in_dims, x, idx):
        x_bdim, _ = in_dims
        x = x.movedim(x_bdim, 1)
        return Select.apply(x, idx), 0

    @staticmethod
    # 对 JVP 的静态方法，计算对输入的雅可比向量积
    def jvp(ctx, x_tangent, _):
        return Select.apply(x_tangent, ctx.idx)


class SelectGenVmap(torch.autograd.Function):
    # 定义一个标志，表示生成 vmap 规则
    generate_vmap_rule = True

    @staticmethod
    # 前向传播函数，根据索引选择张量中的元素
    def forward(x, idx):
        return x[idx]

    @staticmethod
    # 设置上下文的静态方法，用于保存前向传播所需的输入形状和索引
    def setup_context(ctx, inputs, outputs):
        x, idx = inputs
        ctx.x_shape = x.shape
        ctx.idx = idx

    @staticmethod
    # 反向传播的静态方法，计算梯度
    def backward(ctx, grad_output):
        result = grad_output.new_zeros(ctx.x_shape)
        result[ctx.idx] = grad_output
        return result, None

    @staticmethod
    # 对 JVP 的静态方法，计算对输入的雅可比向量积
    def jvp(ctx, x_tangent, _):
        return SelectGenVmap.apply(x_tangent, ctx.idx)


def sample_inputs_select(opinfo, device, dtype, requires_grad, **kwargs):
    # 生成选择操作的示例输入
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg(3, 5), args=(2,))


class ScaleGradGenVmap(torch.autograd.Function):
    # 定义一个标志，表示生成 vmap 规则
    generate_vmap_rule = True
    # 缩放因子
    scale = 3.14

    @staticmethod
    # 前向传播函数，克隆输入张量
    def forward(x):
        return x.clone()

    @staticmethod
    # 设置上下文的静态方法，不保存任何信息
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    # 反向传播的静态方法，计算梯度
    def backward(ctx, grad_output):
        return grad_output * ScaleGradGenVmap.scale

    @staticmethod
    # 对 JVP 的静态方法，计算对输入的雅可比向量积
    def jvp(ctx, x_tangent):
        return x_tangent * ScaleGradGenVmap.scale


class ZeroGradientsGenVmap(torch.autograd.Function):
    # 定义一个标志，表示生成 vmap 规则
    generate_vmap_rule = True

    @staticmethod
    # 前向传播函数，克隆输入张量
    def forward(x, y):
        return x.clone(), y.clone()

    @staticmethod
    # 设置上下文的静态方法，不保存任何信息
    def setup_context(ctx, inputs, outputs):
        pass
    # 定义一个静态方法 `backward`，接受三个参数 `ctx`, `gx`, `gy`
    def backward(ctx, gx, gy):
        # 故意返回一个全零张量，而不是使用 `zeros_like` 或 `new_zeros` 函数
        # 也故意不返回 None
        return (
            # 返回一个故意过大的梯度张量，形状为 (3, 4, *gx.shape)，数据类型和设备与 gx 相同
            torch.zeros(3, 4, *gx.shape, dtype=gx.dtype, device=gx.device),
            # 返回一个形状与 gy 相同的全零张量，数据类型和设备与 gy 相同
            torch.zeros(gy.shape, dtype=gy.dtype, device=gy.device),
        )

    @staticmethod
    # 定义一个静态方法 `jvp`，接受三个参数 `ctx`, `gx`, `gy`
    def jvp(ctx, gx, gy):
        # 故意返回一个全零张量，而不是使用 `zeros_like` 或 `new_zeros` 函数
        # 也故意不返回 None
        return (
            # 返回一个形状与 gx 相同的全零张量，数据类型和设备与 gx 相同
            torch.zeros(gx.shape, dtype=gx.dtype, device=gx.device),
            # 返回一个形状与 gy 相同的全零张量，数据类型和设备与 gy 相同
            torch.zeros(gy.shape, dtype=gy.dtype, device=gy.device),
        )
def sample_inputs_forward_default_args(opinfo, device, dtype, requires_grad, **kwargs):
    # 创建一个部分应用函数，生成一个指定设备、数据类型和梯度属性的张量生成器
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 使用生成器创建一个 SampleInput 对象，并返回
    yield SampleInput(make_arg(3, 5))


class ForwardHasDefaultArgs(torch.autograd.Function):
    @staticmethod
    def forward(x, idx=(2,)):
        # 前向传播函数，根据给定的索引返回张量的子集
        return x[idx]

    @staticmethod
    def setup_context(ctx, inputs, output):
        # 设置上下文函数，用于保存输入张量的形状和索引
        x, idx = inputs
        ctx.x_shape = x.shape
        ctx.idx = idx

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播函数，计算输入张量的梯度
        result = grad_output.new_zeros(ctx.x_shape)
        result[ctx.idx] = grad_output
        return result, None

    @staticmethod
    def vmap(info, in_dims, x, idx):
        # 向量映射函数，对输入张量进行维度重排后调用前向传播函数
        x_bdim, _ = in_dims
        x = x.movedim(x_bdim, 1)
        return ForwardHasDefaultArgs.apply(x, idx), 0

    @staticmethod
    def jvp(ctx, x_tangent, _):
        # 雅可比向量积函数，计算对输入张量的切线方向的影响
        return ForwardHasDefaultArgs.apply(x_tangent, ctx.idx)


autograd_function_db = [
    OpInfo(
        'NumpyCubeAutogradFunction',
        op=NumpyCube.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyExpMarkDirtyAutogradFunction',
        op=lambda x: NumpyExp_.apply(x.clone()),
        inplace_variant=NumpyExp_.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyMulAutogradFunction',
        op=NumpyMul.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_numpy_mul,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyCubeNotComposableAutogradFunction',
        op=lambda x: NumpyCubeNotComposable.apply(x)[0],
        supports_forward_ad=False,
        supports_fwgrad_bwgrad=False,
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpySortAutogradFunction',
        op=NumpySort.apply,
        supports_forward_ad=False,
        supports_fwgrad_bwgrad=False,
        sample_inputs_func=sample_inputs_numpy_sort,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        gradcheck_wrapper=lambda y, ind: y,
    ),
    OpInfo(
        'NumpyTakeAutogradFunction',
        op=NumpyTake.apply,
        supports_forward_ad=False,
        supports_fwgrad_bwgrad=False,
        sample_inputs_func=sample_inputs_numpy_take,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'SelectAutogradFunction',  # 函数名为'SelectAutogradFunction'
        op=Select.apply,  # 操作为Select类的apply方法
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_select,  # 使用sample_inputs_select函数生成示例输入
        dtypes=all_types_and(torch.bool, torch.half),  # 数据类型包括所有类型和torch.bool、torch.half
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'CubeGenVmapAutogradFunction',  # 函数名为'CubeGenVmapAutogradFunction'
        op=CubeGenVmap.apply,  # 操作为CubeGenVmap类的apply方法
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_numpy_cube,  # 使用sample_inputs_numpy_cube函数生成示例输入
        dtypes=all_types_and(torch.bool, torch.half),  # 数据类型包括所有类型和torch.bool、torch.half
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'MulGenVmapAutogradFunction',  # 函数名为'MulGenVmapAutogradFunction'
        op=MulGenVmap.apply,  # 操作为MulGenVmap类的apply方法
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_numpy_mul,  # 使用sample_inputs_numpy_mul函数生成示例输入
        dtypes=all_types_and(torch.bool, torch.half),  # 数据类型包括所有类型和torch.bool、torch.half
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'SortGenVmapAutogradFunction',  # 函数名为'SortGenVmapAutogradFunction'
        op=SortGenVmap.apply,  # 操作为SortGenVmap类的apply方法
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_numpy_sort,  # 使用sample_inputs_numpy_sort函数生成示例输入
        dtypes=all_types_and(torch.bool, torch.half),  # 数据类型包括所有类型和torch.bool、torch.half
        supports_out=False,  # 不支持输出
        gradcheck_wrapper=lambda y, ind: y,  # 梯度检查包装器为lambda函数，接受参数y和ind并返回y
    ),
    OpInfo(
        'SelectGenVmapAutogradFunction',  # 函数名为'SelectGenVmapAutogradFunction'
        op=SelectGenVmap.apply,  # 操作为SelectGenVmap类的apply方法
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_select,  # 使用sample_inputs_select函数生成示例输入
        dtypes=all_types_and(torch.bool, torch.half),  # 数据类型包括所有类型和torch.bool、torch.half
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'ScaleGradGenVmapAutogradFunction',  # 函数名为'ScaleGradGenVmapAutogradFunction'
        op=ScaleGradGenVmap.apply,  # 操作为ScaleGradGenVmap类的apply方法
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_numpy_cube,  # 使用sample_inputs_numpy_cube函数生成示例输入
        dtypes=all_types_and(torch.bool, torch.half),  # 数据类型包括所有类型和torch.bool、torch.half
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'ZeroGradientsGenVmapAutogradFunction',  # 函数名为'ZeroGradientsGenVmapAutogradFunction'
        op=ZeroGradientsGenVmap.apply,  # 操作为ZeroGradientsGenVmap类的apply方法
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_numpy_mul,  # 使用sample_inputs_numpy_mul函数生成示例输入
        dtypes=all_types_and(torch.bool, torch.half),  # 数据类型包括所有类型和torch.bool、torch.half
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        'ForwardHasDefaultArgsAutogradFunction',  # 函数名为'ForwardHasDefaultArgsAutogradFunction'
        op=ForwardHasDefaultArgs.apply,  # 操作为ForwardHasDefaultArgs类的apply方法
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_forward_default_args,  # 使用sample_inputs_forward_default_args函数生成示例输入
        dtypes=all_types_and(torch.bool, torch.half),  # 数据类型包括所有类型和torch.bool、torch.half
        supports_out=False,  # 不支持输出
    ),
]



# 这行代码是一个单独的右方括号字符，用于闭合之前的左方括号或在列表、字典等数据结构中使用。
```