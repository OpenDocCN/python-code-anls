# `.\pytorch\torch\_inductor\fx_passes\pad_mm.py`

```
# mypy: allow-untyped-defs
# 引入 functools、itertools、operator 和 typing 等模块
import functools
import itertools
import operator
import typing
# 引入 List、Optional、Union 等类型
from typing import List, Optional, Union

# 引入 torch 库
import torch
# 引入 torch._inductor.runtime.runtime_utils 模块
import torch._inductor.runtime.runtime_utils
# 从 torch 模块中引入 Tensor 类型
from torch import Tensor
# 引入 torch._dynamo.utils.counters 模块
from torch._dynamo.utils import counters
# 引入 torch._inductor.utils 模块
from torch._inductor import utils
# 从 torch._subclasses.fake_tensor 模块中引入 FakeTensor 类型
from torch._subclasses.fake_tensor import FakeTensor
# 引入 torch.utils._mode_utils.no_dispatch 函数
from torch.utils._mode_utils import no_dispatch

# 从 ...utils._triton 模块中引入 has_triton 函数
from ...utils._triton import has_triton

# 从 ..pattern_matcher 模块中引入 fwd_only、gen_register_replacement、joint_fwd_bwd、Match 和 ReplaceFn 类型
from ..pattern_matcher import (
    fwd_only,
    gen_register_replacement,
    joint_fwd_bwd,
    Match,
    ReplaceFn,
    SearchFn,
)

# 设置 torch.ops.aten 别名为 aten
aten = torch.ops.aten

# This flag is only used for testing purpose.
# Changing it to True will ignore comparing do_bench times
# between original pattern and padded one.
# _skip_do_bench_times 标志，用于测试目的，如果设置为 True，则忽略对比原始模式和填充模式的 do_bench 时间

_skip_do_bench_times = False


def fetch_fake_tensors(match, kwarg_names) -> List[Tensor]:
    # 从 match 中获取 kwargs 字典
    kwargs = match.kwargs
    # 返回匹配关键字参数名列表中每个参数对应的 FakeTensor
    return [kwargs[name].meta["val"] for name in kwarg_names]


def unwrap_fake_args(*arg_names):
    # decorator 函数，用于解包伪参数
    def decorator(func):
        # wrapper 函数，接受 match 参数，从中提取伪张量并传递给 func 函数
        def wrapper(match):
            fake_tensors = fetch_fake_tensors(match, arg_names)
            return func(*fake_tensors)

        return wrapper

    return decorator


def get_alignment_size(x: Tensor) -> int:
    # 获取张量 x 的对齐大小
    if x.dtype == torch.float16 or x.dtype == torch.half or x.dtype == torch.bfloat16:
        return 8
    elif x.dtype == torch.float32 or x.dtype == torch.float:
        return 4
    else:
        return 0


def check_device(a: Tensor, b: Tensor) -> bool:
    # 检查张量 a 和 b 是否都在 CUDA 设备上
    return a.is_cuda and b.is_cuda


def check_dtype(a: Tensor, b: Tensor) -> bool:
    # 检查张量 a 和 b 是否都是浮点数类型
    return a.is_floating_point() and b.is_floating_point()


def should_pad_common(
    mat1: Tensor, mat2: Tensor, input: Optional[Tensor] = None
) -> bool:
    # 判断是否应该进行填充的通用条件
    # 允许有符号形状或步幅，只要它们有提示。稍后我们将确保只填充非符号维度。
    def valid_shape_and_stride(t: Optional[Tensor]) -> bool:
        if t is None:
            return True

        symbolic_cnt = 0
        for x in t.size():
            if isinstance(x, int):
                continue
            elif utils.is_symbolic(x):
                if not x.node.has_hint():
                    return False
                symbolic_cnt += 1
            else:
                return False
        # 过滤所有维度都是符号的情况
        if symbolic_cnt == len(t.size()):
            return False
        return all(
            isinstance(x, int) or (utils.is_symbolic(x) and x.node.has_hint())
            for x in t.stride()
        )

    return (
        torch._inductor.config.shape_padding
        and check_device(mat1, mat2)
        and check_dtype(mat1, mat2)
        and all(valid_shape_and_stride(t) for t in (mat1, mat2, input))
    )


def get_padded_length(x: Union[int, torch.SymInt], alignment_size) -> int:
    # 如果 x 是符号整数或对齐大小为 0，或者 x 可以被对齐大小整除，则不填充 x
    if isinstance(x, torch.SymInt) or alignment_size == 0 or x % alignment_size == 0:
        return 0
    # 如果 x 等于 1，返回 0，因为对于尺寸为 1 的维度可以被压缩掉
    if x == 1:
        return 0

    # 计算使 x 对齐到 alignment_size 的下一个倍数的值
    return int((x // alignment_size + 1) * alignment_size) - x
# 定义一个函数，用于在指定维度上给张量进行填充
def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
    # 如果需要填充的长度为0，直接返回原始张量
    if padded_length == 0:
        return x
    # 创建一个与原始张量相同形状（除了指定维度）的全零张量作为填充
    pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
    # 在指定维度上连接原始张量和填充张量
    return torch.cat([x, pad], dim=dim)


# 定义一个函数，使用aten.addmm执行矩阵相乘操作
def addmm_pattern(
    input: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float
) -> Tensor:
    return aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


# 判断是否需要对输入矩阵进行填充以优化计算性能
def should_pad_addmm(match: Match) -> bool:
    # 从匹配对象中获取假张量，分别为mat1、mat2和input
    mat1, mat2, input = fetch_fake_tensors(match, ("mat1", "mat2", "input"))
    # 调用两个辅助函数，判断是否需要对输入矩阵进行填充
    return should_pad_common(mat1, mat2, input) and should_pad_bench(
        match, mat1, mat2, torch.ops.aten.addmm, input=input
    )


# 对输入矩阵进行填充，以优化aten.addmm的计算性能
def pad_addmm(
    input: Optional[Tensor],
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    beta=1.0,
    alpha=1.0,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
):
    # 某些情况下维度顺序反转，因此需要按顺序为每个维度指定左右填充
    # 如果mat1未预填充，则对其进行填充操作
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    # 如果mat2未预填充，则对其进行填充操作
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )

    # 对输入进行填充，只有当维度不等于1时才进行填充操作
    if input is not None:
        # 如果n的填充长度不为0，且输入张量为二维且第二维度不为1，则在第1维度上进行填充
        if n_padded_length != 0:
            if input.dim() == 2 and input.shape[1] != 1:
                input = pad_dim(input, n_padded_length, 1)
            # 如果输入张量为一维且第一维度不为1，则在第0维度上进行填充
            elif input.dim() == 1 and input.shape[0] != 1:
                input = pad_dim(input, n_padded_length, 0)
        # 如果m的填充长度不为0，且输入张量为二维且第一维度不为1，则在第0维度上进行填充
        if m_padded_length != 0 and input.dim() == 2 and input.shape[0] != 1:
            input = pad_dim(input, m_padded_length, 0)

    # 调用aten.addmm执行矩阵乘法操作
    res = aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

    # 根据填充长度对结果进行修剪
    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    # 返回结果张量
    return res


# 替换函数，用于执行addmm操作并在需要时进行填充
def addmm_replace(
    input: Optional[Tensor], mat1: Tensor, mat2: Tensor, beta=1.0, alpha=1.0
) -> Tensor:
    # 计算mat1和mat2的填充长度
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    # 调用pad_addmm函数执行矩阵乘法并进行填充
    return pad_addmm(
        input,
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
        beta,
        alpha,
    )


# 判断是否为矩阵乘法计算受限的情况
def is_mm_compute_bound(M: int, K: int, N: int, dtype: torch.dtype) -> bool:
    denominator = M * K + N * K + M * N
    # 如果分母为0，返回False
    if denominator == 0:
        return False
    # 计算算术强度
    arithmetic_intensity = (M * N * K) / denominator

    # 经验表明，在这种情况下，即使在带宽受限的情况下，性能也会受到一些显著影响
    # 检查条件：如果数据类型是 torch.bfloat16，且 K 大于 M 和 N，并且 CUDA 设备的能力小于 (9, 0)，则返回 True
    if (
        dtype is torch.bfloat16
        and K > M
        and K > N
        and torch.cuda.get_device_capability() < (9, 0)
    ):  # doesnt repro on h100s:
        return True

    # 处理 AMD 环境下的异常情况
    try:
        # 计算机性能平衡计算：设备 TFLOPS 的千倍除以 GPU DRAM 带宽的 GBPS
        machine_balance = (
            1000 * utils.get_device_tflops(dtype)
        ) / utils.get_gpu_dram_gbps()
    except Exception:
        # 如果计算机性能平衡计算中出现异常，返回 True
        return True

    # 由于缓存的存在，可能会低估 dram_gbps 的带宽，因此调整机器性能平衡
    # 如果我们将机器平衡估计得太低，可能会错过一些加速效果，
    # 如果估计得太高，可能会增加不必要的编译时间
    # TODO - 在此处微调系数。作为参考，Triton mm 模型假设 80% 的读取在缓存中，并且缓存速度比 dram_gbps 快 4 倍
    machine_balance = machine_balance * 0.5

    # 返回算术强度是否大于机器性能平衡
    return arithmetic_intensity > machine_balance
# 使用 functools 模块的 lru_cache 装饰器，创建一个缓存函数，无限制缓存大小
@functools.lru_cache(None)
# 返回一个本地缓存对象，用于缓存数据
def get_pad_cache():
    return torch._inductor.codecache.LocalCache()


# 获取缓存中指定键对应的值，返回布尔值
def get_cached_should_pad(key: str) -> bool:
    return get_pad_cache().lookup(key)


# 将指定键值对存入缓存
def set_cached_should_pad(key: str, value: bool):
    return get_pad_cache().set_value(key, value=value)


# 获取缓存中基准矩阵乘法的基准时间
def get_cached_base_mm_benchmark_time(key: str) -> float:
    return get_pad_cache().lookup(key)


# 将基准矩阵乘法的基准时间存入缓存
def set_cached_base_mm_benchmark_time(key: str, value: float):
    return get_pad_cache().set_value(key, value=value)


# 根据输入参数创建矩阵乘法的基准时间键
def should_pad_bench_key(
    match,
    mat1: Tensor,
    mat2: Tensor,
    op,
    input: Optional[Tensor] = None,
    is_base_time_key=False,
) -> str:
    # 定义一个函数，生成表示张量形状、步幅和数据类型的键
    def tensor_key(t):
        return (t.shape, t.stride(), t.dtype)

    # 如果 mat1 是 float32 类型，设置 tf32_key 为对应的 TF32 允许标志
    tf32_key = (
        None if mat1.dtype != torch.float32 else torch.backends.cuda.matmul.allow_tf32
    )

    # 格式化生成键的函数，根据是否为基准时间键决定是否包含排除填充信息
    def fmt_pad(name):
        if is_base_time_key:
            return None
        return f"exclude_pad:{should_exclude_padding_time(match, name)}"

    # 构建矩阵乘法键的元组，包括各种可能影响计算的参数和标志
    key = (
        tensor_key(mat1),
        tensor_key(mat2),
        fmt_pad("mat1"),
        fmt_pad("mat2"),
        op,
        input if input is None else tensor_key(input),
        tf32_key,
    )

    # 将键转换为字符串格式
    key = str(key)
    # 如果是基准时间键，添加额外的标识
    if is_base_time_key:
        key = f"base mm time: {key}"
    return key


# 获取非视图定义的节点
def get_non_view_def(node):
    # 如果节点是获取项操作，递归获取其第一个参数的非视图定义
    if node.op == operator.getitem:
        return get_non_view_def(node.args[0])

    # 如果节点是调用函数操作且目标是 torch._ops.OpOverload 类型且是视图操作，则递归获取其第一个输入节点的非视图定义
    if (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and utils.is_view(node.target)
    ):
        return get_non_view_def(node.all_input_nodes[0])

    # 否则返回节点本身
    return node


# 判断是否应该排除填充时间的函数
def should_exclude_padding_time(match, arg_name):
    # 获取参数对应的非真实张量定义节点
    node_def = get_non_view_def(match.kwargs[arg_name])

    # 如果伪造张量不是连续的，则返回 False
    if not fetch_fake_tensors(match, (arg_name,))[0].is_contiguous():
        return False

    # 乐观地假设我们应该能够内存规划所有非输入参数
    return node_def.op != "placeholder"


# 判断是否应该进行填充时间的基准测试
def should_pad_bench(
    match, mat1: Tensor, mat2: Tensor, op, input: Optional[Tensor] = None
) -> bool:
    # 创建一个部分函数，用于执行 GPU 基准测试
    do_bench = functools.partial(
        torch._inductor.runtime.runtime_utils.do_bench_gpu,
        warmup=5,
    )
    m_padded_length = 0
    n_padded_length = 0
    batchsize = 1
    # 如果 k_padded_length 或 m_padded_length 不为零，则进行以下操作：
    # - 为了使用 constant_pad_nd，维度顺序被反转，对于每个维度，我们指定右边和左边的填充量
    pad_arg = [0, k_padded_length, 0, m_padded_length]
    
    # 如果是矩阵乘法操作（is_bmm 为真），需要额外填充两个维度
    if is_bmm:
        pad_arg.extend((0, 0))
    
    # 调用 PyTorch 的 constant_pad_nd 函数对 mat1 进行填充操作，并返回填充后的结果
    return aten.constant_pad_nd(mat1, pad_arg)
    
    # 如果 k_padded_length 和 m_padded_length 都为零，则直接返回原始的 mat1
    return mat1
# 根据指定的参数对矩阵进行填充，返回填充后的矩阵
def pad_mat2(mat2, *, k_padded_length, n_padded_length, is_bmm=False):
    # 如果需要进行填充
    if k_padded_length != 0 or n_padded_length != 0:
        # 定义填充参数列表，维度顺序与 constant_pad_nd 相反，每个维度指定右侧和左侧的填充量
        pad_arg = [0, n_padded_length, 0, k_padded_length]
        # 如果是 bmm 操作，需要额外填充两个维度
        if is_bmm:
            pad_arg.extend((0, 0))
        # 使用 constant_pad_nd 函数进行矩阵填充
        return aten.constant_pad_nd(mat2, pad_arg)
    else:
        # 如果不需要填充，则直接返回原始矩阵
        return mat2


# 对两个矩阵进行矩阵乘法，支持填充操作，返回乘积矩阵
def pad_mm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    # 如果 mat1 没有预填充，则对其进行填充
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    # 如果 mat2 没有预填充，则对其进行填充
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )
    # 使用 aten.mm 函数进行矩阵乘法运算
    res = aten.mm(mat1, mat2)
    # 如果 m 方向有填充，则去除填充的部分
    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    # 如果 n 方向有填充，则去除填充的部分
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    # 返回乘积矩阵结果
    return res


# 对两个矩阵进行替换操作，直接调用 aten.bmm 函数
def mm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    # 获取需要填充的维度长度
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    # 调用 pad_mm 函数进行矩阵乘法操作并返回结果
    return pad_mm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )


# 对两个矩阵进行批量矩阵乘法，直接调用 aten.bmm 函数
def bmm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    return aten.bmm(mat1, mat2)


# 判断是否需要对矩阵进行填充，返回布尔值
def should_pad_bmm(match: Match) -> bool:
    # 从 match 中获取假的张量 mat1 和 mat2
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    # 判断是否需要对这两个矩阵进行共同填充，并且是否需要对其进行基准填充操作
    return should_pad_common(mat1, mat2) and should_pad_bench(
        match, mat1, mat2, torch.ops.aten.bmm
    )


# 对两个矩阵进行批量矩阵乘法，支持填充操作，返回乘积矩阵
def pad_bmm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    # 如果 mat1 没有预填充，则对其进行填充，同时指定 is_bmm=True
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1,
            m_padded_length=m_padded_length,
            k_padded_length=k_padded_length,
            is_bmm=True,
        )
    # 如果 mat2 没有预填充，则对其进行填充，同时指定 is_bmm=True
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2,
            k_padded_length=k_padded_length,
            n_padded_length=n_padded_length,
            is_bmm=True,
        )
    # 使用 aten.bmm 函数进行批量矩阵乘法运算
    res = aten.bmm(mat1, mat2)
    # 如果 m 方向有填充，则去除填充的部分
    if m_padded_length != 0:
        res = res[:, :-m_padded_length, :]
    # 如果 n 方向有填充，则去除填充的部分
    if n_padded_length != 0:
        res = res[:, :, :-n_padded_length]
    # 返回乘积矩阵结果
    return res


# 对两个矩阵进行替换操作，直接调用 aten.bmm 函数
def bmm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    # 获取需要填充的维度长度
    k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))
    m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    # 调用 pad_bmm 函数，并返回其计算结果
    return pad_bmm(
        mat1,                 # 第一个矩阵参数
        mat2,                 # 第二个矩阵参数
        m_padded_length,      # m 方向填充后的长度
        k_padded_length,      # k 方向填充后的长度
        n_padded_length,      # n 方向填充后的长度
    )
@functools.lru_cache(None)
# 使用 functools 提供的 lru_cache 装饰器，实现函数结果的缓存，参数为 None 表示不限制缓存大小

def _pad_mm_init():
    from .joint_graph import patterns
    # 导入模块中的 patterns 对象，用于模式匹配和替换

    if torch.cuda.is_available():
        # 检查是否有可用的 CUDA 设备，以确定设备类型
        device = "cuda"
    else:
        device = "cpu"
        # 如果没有 CUDA 设备可用，则使用 CPU 设备

    # 创建部分初始化的 tensor 对象，为模式匹配的输入参数提供初始值
    dim2a = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)
    dim2b = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)

    dim3a = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)
    dim3b = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)

    dim1a = functools.partial(torch.empty, (4), device=device, requires_grad=True)

    # 临时解决方案，用于处理 https://github.com/pytorch/pytorch/issues/97894 的问题
    # 0.113377 是一个“魔法”值，用于恢复丢失的输入参数关系
    rep = {"beta": 0.213377, "alpha": 0.113377}

    # 遍历模式匹配的对象列表，注册替换函数及其相关参数
    for pattern, replacement, args, workaround, extra_check in [
        (
            typing.cast(SearchFn, mm_pattern),  # 类型转换，模式匹配函数的类型
            typing.cast(ReplaceFn, mm_replace),  # 类型转换，替换函数的类型
            [dim2a(), dim2b()],  # 模式匹配函数的输入参数列表
            {},  # 额外的修复操作，此处为空字典
            should_pad_mm,  # 额外的检查函数，用于确定是否需要进行填充操作
        ),
        (
            typing.cast(SearchFn, bmm_pattern),
            typing.cast(ReplaceFn, bmm_replace),
            [dim3a(), dim3b()],
            {},
            should_pad_bmm,
        ),
        (
            typing.cast(SearchFn, addmm_pattern),
            typing.cast(ReplaceFn, addmm_replace),
            [dim1a(), dim2a(), dim2b()],
            rep,  # 使用 rep 字典作为标量修复的临时解决方案
            should_pad_addmm,
        ),
    ]:
        assert isinstance(workaround, dict)  # 断言确保 workaround 是一个字典类型，mypy 无法准确推断其类型
        name = pattern.__name__  # 获取模式函数的名称

        gen_register_replacement(
            f"{name}_training",
            pattern,
            replacement,
            args,
            joint_fwd_bwd,
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )
        # 生成注册替换函数调用，用于训练阶段

        gen_register_replacement(
            f"{name}_inference",
            pattern,
            replacement,
            args,
            fwd_only,
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )
        # 生成注册替换函数调用，用于推理阶段
```