# `.\pytorch\torch\_decomp\decompositions_for_jvp.py`

```
# mypy: allow-untyped-defs
# 导入 inspect 模块，用于获取对象信息
import inspect
# 导入类型相关的声明
from typing import Callable, Dict, List, Optional, Tuple

# 导入 PyTorch 库
import torch
# 导入 PyTorch 内部的分解机制
import torch._decomp
# 从 torch 模块中导入 Tensor 类型
from torch import Tensor
# 导入可能移除输出包装器的函数
from torch._prims_common.wrappers import _maybe_remove_out_wrapper

# 获取 PyTorch 中的分解表
decomposition_table = torch._decomp.decomposition_table
# 创建一个空的字典，用于存储为 JVP 注册的分解函数
decomposition_table_for_jvp: Dict[torch._ops.OperatorBase, Callable] = {}
# 获取注册分解的函数
register_decomposition = torch._decomp.register_decomposition
# 获取 PyTorch 操作符函数
aten = torch.ops.aten

# NOTE: [forward-mode AD decompositions mechanism]
#
# 前向自动微分分解机制位于 VariableType 类中，
# 如果任何输入具有前向梯度，并且未实现前向自动微分公式，
# 同时函数实际上是可微的，则运行分解机制。
# 查看 run_jit_decomposition_with_args_for_jvp 函数。
# 目前我们使用 Python 的分解，我们可以 torchscript 化。
#
# 注意，我们将在分解级别构建反向图，这没问题，因为否则我们会出错。
#
# TODO: 我们用于注册分解的机制似乎不仅仅用于 jvp。因此，这里的一个开放问题是
# torch/csrc/jit/runtime/decomposition_registry.cpp 是否用于其他目的。
# 如果是这样，我们可能会意外地进入分解路径（可能产生难以理解的错误），
# 而不是更早地出错并打印出前向自动微分公式未实现。
#
# 解决方案可能是在启用分解时具有显式的白名单控制。
#

def maybe_register_decomposition(op):
    # 定义装饰器函数，用于注册分解操作
    def decorator(f):
        try:
            return register_decomposition(op)(f)
        except Exception:
            return f

    return decorator


# 需要为 jvp 注册特殊分解的函数列表
decomposition_table_for_jvp = {}


def register_decomposition_for_jvp(fn):
    # 注册函数的 jvp 版本的分解
    return register_decomposition(fn, registry=decomposition_table_for_jvp)


def _register_jit_decomposition_for_jvp(decomp, use_python=False):
    # 如果在 jvp 的分解表中找到了指定的分解函数
    if decomp in decomposition_table_for_jvp:
        decomposition_table_used = decomposition_table_for_jvp
    # 否则，如果在一般的分解表中找到了指定的分解函数
    elif decomp in decomposition_table:
        decomposition_table_used = decomposition_table
    else:
        # 抛出运行时错误，指定的分解函数未找到
        raise RuntimeError(f"could not find decomposition for {decomp}")
    # 获取对应分解函数
    decomp_fn = decomposition_table_used[decomp]

    # `out_wrapper` 扩展了分解函数的签名，增加了一个 `out` 参数。
    # 然而，jit 将使用未包装的函数签名，因此我们需要在这里解除包装，以防止错误。
    decomp_fn = _maybe_remove_out_wrapper(decomp_fn)
    if use_python:
        # 如果 use_python 为 True，则需要在 Torch JIT 中忽略 decomp_fn 函数的编译
        decomp_fn = torch.jit.ignore(decomp_fn)
        
        # 获取 decomp_fn 函数的参数签名
        sig = inspect.signature(decomp_fn)

        # 创建一个字符串，包含从函数签名生成的函数包装形式
        # 例如输出:
        # def wrapped_decomp(x: torch.Tensor, y: int, z: int):
        #   return decomp_fn(x, y, z)
        # Thanks copilot!
        def get_function_def(sig):
            # 获取参数定义部分
            param_def = [f"{param_str}" for param_str in sig.parameters.values()]
            # 获取参数使用部分
            param_use = [f"{param_str}" for param_str in sig.parameters.keys()]

            # 构造包装函数的字符串形式
            return f"def wrapped_decomp({', '.join(param_def)}):\n  return decomp_fn({', '.join(param_use)})\n"

        # 根据函数签名获取函数定义字符串
        f_str = get_function_def(sig)

        # 使用 Torch JIT 创建编译单元，并从中获取函数的图形表示
        graph = torch.jit.CompilationUnit(f_str).wrapped_decomp.graph
    else:
        # 如果 use_python 为 False，则对 decomp_fn 函数进行 Torch JIT 的脚本编译
        graph = torch.jit.script(decomp_fn).graph

    # 将获得的函数图形注册为 decomposition 的一部分
    torch.jit._register_decomposition(decomp, graph)
# 对于这些函数的唯一分解是临时的或者是为了 jvp 的目的而进行的修补

# TODO: 这些也应该属于这里吗？
@maybe_register_decomposition(aten.trace.default)
# 如果是的话，注册 aten.trace.default 函数的分解器
def trace(self: Tensor) -> Tensor:
    return torch.sum(torch.diag(self))


@maybe_register_decomposition(aten.log_sigmoid_forward.default)
# 如果是的话，注册 aten.log_sigmoid_forward.default 函数的分解器
def log_sigmoid_forward(self: Tensor) -> Tuple[Tensor, Tensor]:
    min = torch.minimum(self.new_zeros(()), self)
    z = torch.exp(-torch.abs(self))
    if self.is_cuda:
        buffer = self.new_zeros((0,))
    else:
        buffer = z
    return min - torch.log1p(z), buffer


def recompute_mean_var(
    input: Tensor, rstd: Tensor, inner_dim_indices: List[int], keepdim: bool
):
    # 对于大多数标准化分解来说，除了这里，它将与核心版本相同
    # 我们重新计算均值和方差，以便它们通过输入跟踪梯度

    mean = torch.mean(input, dim=inner_dim_indices, keepdim=keepdim)
    var = torch.var(input, dim=inner_dim_indices, unbiased=False, keepdim=keepdim)
    eps = torch.pow(1 / rstd, 2) - var  # 这让我内心非常难过
    eps = eps.detach()
    rstd = 1 / torch.sqrt(var + eps)
    return mean, rstd


@register_decomposition_for_jvp(aten.native_layer_norm_backward)
# 注册 aten.native_layer_norm_backward 函数的 jvp 分解器
def native_layer_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: List[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    output_mask: List[bool],
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    input_shape = input.shape
    input_ndim = input.dim()

    axis = input_ndim - len(normalized_shape)
    inner_dims = input_shape[axis:]
    outer_dims = input_shape[:axis]
    inner_dim_indices = list(range(axis, input_ndim))
    outer_dim_indices = list(range(0, axis))

    N = 1
    for i in inner_dims:
        N *= i
    M = 1
    for i in outer_dims:
        M *= i
    if M <= 0 or N <= 0:
        return (
            input.new_zeros(input_shape),
            input.new_zeros(input_shape[axis:]),
            input.new_zeros(input_shape[axis:]),
        )

    mean_, rstd_ = recompute_mean_var(input, rstd, inner_dim_indices, keepdim=True)

    x_hat = (input - mean_) * rstd_
    if weight is not None:
        grad_x_hat = grad_out * weight
    else:
        grad_x_hat = grad_out
    a = grad_x_hat * N
    b = torch.sum(grad_x_hat, inner_dim_indices, True)
    c1 = torch.mul(grad_x_hat, x_hat)
    c2 = torch.sum(c1, inner_dim_indices, True)
    c3 = torch.mul(x_hat, c2)
    inner = a - b - c3

    if output_mask[0]:
        d_input: Optional[Tensor] = (rstd_ / N) * inner
    else:
        d_input = torch.zeros_like(input)  # 应该是 None，但不适用于 vjp

    if output_mask[1] and weight is not None:
        if len(outer_dim_indices) > 0:
            d_weight: Optional[Tensor] = torch.sum(
                grad_out * x_hat, outer_dim_indices, False
            )
        else:
            d_weight = grad_out * x_hat
    # 如果weight不为None，则创建一个与weight形状相同的全零张量作为梯度
    elif weight is not None:
        d_weight = torch.zeros_like(weight)  # should be None but doesn't work with vjp
    # 如果bias不为None，并且output_mask[2]为True，则计算偏置项的梯度
    elif output_mask[2] and bias is not None:
        # 如果outer_dim_indices列表不为空，则沿着outer_dim_indices指定的维度对grad_out进行求和
        if len(outer_dim_indices) > 0:
            d_bias: Optional[Tensor] = torch.sum(grad_out, outer_dim_indices, False)
        # 否则直接克隆grad_out作为偏置项的梯度
        else:
            d_bias = grad_out.clone()
    # 如果bias不为None且output_mask[2]为False，则创建一个与bias形状相同的全零张量作为偏置项的梯度
    elif bias is not None:
        d_bias = torch.zeros_like(bias)  # should be None but doesn't work with vjp
    # 如果bias为None，则创建一个标量全零张量作为偏置项的梯度
    else:
        d_bias = torch.zeros(())  # should be None but doesn't work with vjp

    # 返回输入的梯度d_input，权重的梯度d_weight和偏置项的梯度d_bias的元组
    return (d_input, d_weight, d_bias)
# 定义一个函数 `prod`，计算列表中所有整数的乘积
def prod(x: List[int]):
    # 初始化结果为1
    r = 1
    # 遍历列表中的每个整数，依次累乘到结果中
    for i in x:
        r *= i
    # 返回最终的乘积结果
    return r


# 为 `aten.native_batch_norm_backward` 函数注册一个 JVP（Jacobian Vector Product）分解
@register_decomposition_for_jvp(aten.native_batch_norm_backward)
def native_batch_norm_backward(
    grad_out: Tensor,  # 梯度输出张量
    input: Tensor,  # 输入张量
    weight: Optional[Tensor],  # 权重张量（可选）
    running_mean: Optional[Tensor],  # 运行时均值张量（可选）
    running_var: Optional[Tensor],  # 运行时方差张量（可选）
    save_mean: Optional[Tensor],  # 保存的均值张量（可选）
    save_invstd: Optional[Tensor],  # 保存的标准差倒数张量（可选）
    train: bool,  # 是否为训练模式的布尔值
    eps: float,  # 一个小的浮点数，用于数值稳定性
    output_mask: List[bool],  # 输出掩码列表，用于条件判断
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:  # 返回值为三元组，包含梯度输入、梯度权重和梯度偏置（可选）

    # 获取输入张量的形状
    input_shape = input.shape
    # 获取输入张量的维度数
    input_rank = input.dim()
    # 断言输入张量的维度数至少为2，否则抛出错误信息
    assert input_rank >= 2, "rank of the input must be at least 2"

    # 设置轴为1
    axis = 1
    # 计算每个特征的数量
    num_features = prod(input_shape) / input_shape[axis]  # type: ignore[arg-type]

    # 初始化均值和标准差倒数
    mean = save_mean
    invstd = save_invstd

    # 如果处于训练模式
    if train:
        # 断言保存的均值和标准差倒数不为空
        assert (
            save_mean is not None and save_invstd is not None
        ), "when train=True, save_mean and save_invstd are required"

        # 定义减少维度的维度列表
        reduciton_dims = [0] + list(range(2, input.dim()))
        # 断言标准差倒数不为空，用于类型提示
        assert invstd is not None  # for typing
        # 重新计算均值和方差的函数调用
        mean, invstd = recompute_mean_var(input, invstd, reduciton_dims, keepdim=False)
    else:
        # 断言运行时均值和方差不为空
        assert running_mean is not None and running_var is not None
        # 将均值设为运行时均值
        mean = running_mean
        # 计算标准差的倒数
        invstd = torch.rsqrt(running_var + eps)

    # 断言标准差倒数和均值不为空
    assert invstd is not None and mean is not None

    # 创建广播掩码，将所有轴初始化为1
    broadcast_mask = [1] * input_rank
    # 将轴设为输入张量的轴数
    broadcast_mask[axis] = input_shape[axis]

    # 初始化减少轴列表
    reduction_axes: List[int] = []
    # 遍历输入张量的每个维度
    for i in range(input_rank):
        # 如果不是轴
        if i != axis:
            # 将其添加到减少轴列表中
            reduction_axes.append(i)

    # 将均值重塑为广播掩码形式
    mean = torch.reshape(mean, broadcast_mask)
    # 计算标准化因子
    norm = 1.0 / num_features
    # 计算梯度输出的和
    grad_output_sum = torch.sum(grad_out, reduction_axes)
    # 计算点积
    dot_p = torch.sum(grad_out * (input - mean), reduction_axes)

    # 将梯度均值重塑为广播掩码形式
    grad_mean = torch.reshape(grad_output_sum * norm, broadcast_mask)
    # 计算投影比例
    proj_scale = torch.reshape(torch.mul(dot_p * norm, invstd * invstd), broadcast_mask)

    # 如果权重为空
    if weight is None:
        # 将梯度比例重塑为广播掩码形式，乘以标准差
        grad_scale = torch.reshape(invstd, broadcast_mask) * 1.0
    else:
        # 将梯度比例重塑为广播掩码形式，乘以标准差和权重
        grad_scale = torch.reshape(invstd * weight, broadcast_mask)

    # 如果处于训练模式
    if train:
        # 计算投影
        proj = (input - mean) * proj_scale
        # 计算梯度输入
        grad_input = ((grad_out - proj) - grad_mean) * grad_scale
    else:
        # 计算梯度输入
        grad_input = grad_out * grad_scale

    # 如果输出掩码的第二位为真
    if output_mask[1]:
        # 计算权重梯度
        grad_weight = dot_p * invstd
    # 否则如果权重不为空
    elif weight is not None:
        # 初始化与权重相同形状的零张量，应该是None但不适用于vjp
        grad_weight = torch.zeros_like(
            weight
        )  # should be None but doesn't work with vjp
    else:
        # 初始化形状为空的零张量，应该是None但不适用于vjp
        grad_weight = torch.zeros(())  # should be None but doesn't work with vjp

    # 如果输出掩码的第三位为真
    if output_mask[2]:
        # 计算偏置梯度
        grad_bias = grad_output_sum
    else:
        # 初始化与梯度输出和相同形状的零张量，应该是None但不适用于vjp
        grad_bias = torch.zeros_like(
            grad_output_sum
        )  # should be None but doesn't work with vjp

    # 返回梯度输入、梯度权重和梯度偏置
    return (grad_input, grad_weight, grad_bias)
    running_var: Optional[Tensor],  # 可选的张量，用于存储运行中的方差
    save_mean: Optional[Tensor],    # 可选的张量，用于保存均值
    save_var: Optional[Tensor],     # 可选的张量，用于保存方差
    update: bool,                   # 布尔值，指示是否更新统计信息
    eps: float,                     # 浮点数，表示用于数值稳定性的小常数
    output_mask: List[bool],        # 布尔值列表，用于指示输出哪些元素
    reserve: Tensor,                # 张量，用于存储额外的缓冲区或保留空间
# 调用原生批量归一化反向传播函数，计算反向传播的梯度
def native_batch_norm_backward(
    grad_out,
    input,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_var,
    update,
    eps,
    output_mask,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    return native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        update,
        eps,
        output_mask,
    )


# 注册用于 JVP（Jacobian-vector product）分解的 JIT 函数，默认使用 Python 实现
_register_jit_decomposition_for_jvp(torch.ops.aten.trace.default, use_python=True)

# 注册用于 JVP 分解的 JIT 函数，对应于 nll_loss_backward.default 操作
_register_jit_decomposition_for_jvp(torch.ops.aten.nll_loss_backward.default)

# 注册用于 JVP 分解的 JIT 函数，对应于 nll_loss2d_backward.default 操作
_register_jit_decomposition_for_jvp(torch.ops.aten.nll_loss2d_backward.default)

# 注册用于 JVP 分解的 JIT 函数，对应于 _log_softmax_backward_data.default 操作
_register_jit_decomposition_for_jvp(torch.ops.aten._log_softmax_backward_data.default)

# 注册用于 JVP 分解的 JIT 函数，对应于 _softmax_backward_data.default 操作
_register_jit_decomposition_for_jvp(torch.ops.aten._softmax_backward_data.default)

# 注册用于 JVP 分解的 JIT 函数，对应于 log_sigmoid_forward.default 操作
_register_jit_decomposition_for_jvp(torch.ops.aten.log_sigmoid_forward.default)

# 注册用于 JVP 分解的 JIT 函数，对应于 native_layer_norm_backward.default 操作
_register_jit_decomposition_for_jvp(torch.ops.aten.native_layer_norm_backward.default)

# 注册用于 JVP 分解的 JIT 函数，对应于 native_batch_norm_backward.default 操作
_register_jit_decomposition_for_jvp(torch.ops.aten.native_batch_norm_backward.default)

# 注册用于 JVP 分解的 JIT 函数，对应于 cudnn_batch_norm_backward.default 操作
_register_jit_decomposition_for_jvp(torch.ops.aten.cudnn_batch_norm_backward.default)

# 注册用于 JVP 分解的 JIT 函数，对应于 batch_norm_backward.default 操作
_register_jit_decomposition_for_jvp(torch.ops.aten.batch_norm_backward.default)

# 注册用于 JVP 分解的 JIT 函数，对应于 miopen_batch_norm_backward.default 操作
_register_jit_decomposition_for_jvp(torch.ops.aten.miopen_batch_norm_backward.default)
```