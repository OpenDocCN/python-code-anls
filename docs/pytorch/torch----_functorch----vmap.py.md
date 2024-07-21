# `.\pytorch\torch\_functorch\vmap.py`

```
# 忽略类型检查错误，对于mypy来说，这个指令告诉它在当前文件中忽略类型错误
# 代码版权声明，此代码版权归Facebook及其关联公司所有
# 在根目录的LICENSE文件中可以找到BSD风格的许可证

# 导入标准库和第三方库
import contextlib  # 上下文管理工具
import functools  # 函数工具
import itertools  # 迭代工具
import os  # 操作系统相关功能
import threading  # 线程支持
from functools import partial  # 偏函数功能
from typing import Any, Callable, List, Optional, Tuple, Union  # 类型提示

import torch  # PyTorch库
from torch import Tensor  # 张量类型

from torch._C._functorch import (  # 导入functorch模块的特定函数和变量
    _add_batch_dim,  # 添加批次维度
    _remove_batch_dim,  # 移除批次维度
    _vmap_decrement_nesting,  # 减少vmap嵌套
    _vmap_increment_nesting,  # 增加vmap嵌套
    is_batchedtensor,  # 判断是否为批次张量
)
from torch.utils._pytree import (  # 导入_pytree模块的特定函数和变量
    _broadcast_to_and_flatten,  # 广播并展平
    tree_flatten,  # 树形展平
    tree_map_,  # 树形映射
    tree_unflatten,  # 树形展开
    TreeSpec,  # 树结构描述
)

# 定义类型别名
in_dims_t = Union[int, Tuple]  # 输入维度类型
out_dims_t = Union[int, Tuple[int, ...]]  # 输出维度类型


# 装饰器函数，用于标记不支持保存张量钩子的函数
def doesnt_support_saved_tensors_hooks(f):
    message = (
        "torch.func transforms don't yet support saved tensor hooks. "
        "Please open an issue with your use case."
    )

    @functools.wraps(f)
    def fn(*args, **kwargs):
        with torch.autograd.graph.disable_saved_tensors_hooks(message):
            return f(*args, **kwargs)

    return fn


# 函数：验证并获取批次大小
def _validate_and_get_batch_size(
    flat_in_dims: List[Optional[int]], flat_args: List
) -> int:
    batch_sizes = [
        arg.size(in_dim)
        for in_dim, arg in zip(flat_in_dims, flat_args)
        if in_dim is not None
    ]
    if len(batch_sizes) == 0:
        raise ValueError("vmap: Expected at least one Tensor to vmap over")
    if batch_sizes and any(size != batch_sizes[0] for size in batch_sizes):
        raise ValueError(
            f"vmap: Expected all tensors to have the same size in the mapped "
            f"dimension, got sizes {batch_sizes} for the mapped dimension"
        )
    return batch_sizes[0]


# 函数：计算批次输出的数量
def _num_outputs(batched_outputs: Union[Tensor, Tuple[Tensor, ...]]) -> int:
    if isinstance(batched_outputs, tuple):
        return len(batched_outputs)
    return 1


# 函数：将值转换为元组
def _as_tuple(
    value: Any, num_elements: int, error_message_lambda: Callable[[], str]
) -> Tuple:
    if not isinstance(value, tuple):
        return (value,) * num_elements
    if len(value) != num_elements:
        raise ValueError(error_message_lambda())
    return value


# 函数：处理批次输入，准备用于vmap函数的参数
def _process_batched_inputs(
    in_dims: in_dims_t, args: Tuple, func: Callable
) -> Tuple[int, List[Any], List[Any], TreeSpec]:
    if not isinstance(in_dims, int) and not isinstance(in_dims, tuple):
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"expected `in_dims` to be int or a (potentially nested) tuple "
            f"matching the structure of inputs, got: {type(in_dims)}."
        )
    # 如果没有传入任何参数，抛出数值错误异常，说明用户可能忘记添加输入，或者尝试对没有输入的函数进行 vmap 操作，后者是不支持的。
    if len(args) == 0:
        raise ValueError(
            f"vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot to add "
            f"inputs, or you are trying to vmap over a function with no inputs. "
            f"The latter is unsupported."
        )

    # 将输入参数进行扁平化处理，并获取参数结构描述
    flat_args, args_spec = tree_flatten(args)
    
    # 根据输入的维度信息对其进行广播并扁平化处理
    flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
    
    # 如果广播处理后的维度信息为 None，则抛出数值错误异常，说明输入的维度与结构不兼容
    if flat_in_dims is None:
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"in_dims is not compatible with the structure of `inputs`. "
            f"in_dims has structure {tree_flatten(in_dims)[1]} but inputs "
            f"has structure {args_spec}."
        )

    # 遍历扁平化后的参数列表和对应的维度信息
    for i, (arg, in_dim) in enumerate(zip(flat_args, flat_in_dims)):
        # 如果维度信息不是整数且不为 None，则抛出数值错误异常，说明维度信息必须是整数或 None
        if not isinstance(in_dim, int) and in_dim is not None:
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but in_dim must be either "
                f"an integer dimension or None."
            )
        # 如果维度信息是整数且对应的参数不是 Tensor 类型，则抛出数值错误异常，说明不能对非 Tensor 类型的参数进行 vmap 操作
        if isinstance(in_dim, int) and not isinstance(arg, Tensor):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but the input is of type "
                f"{type(arg)}. We cannot vmap over non-Tensor arguments, "
                f"please use None as the respective in_dim"
            )
        # 如果维度信息不为 None 且不在有效范围内，则抛出数值错误异常，说明维度信息超出了对应 Tensor 的维度范围
        if in_dim is not None and (in_dim < -arg.dim() or in_dim >= arg.dim()):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for some input, but that input is a Tensor "
                f"of dimensionality {arg.dim()} so expected in_dim to satisfy "
                f"-{arg.dim()} <= in_dim < {arg.dim()}."
            )
        # 如果维度信息不为 None 且为负数，则对其进行调整，使其落在有效范围内
        if in_dim is not None and in_dim < 0:
            flat_in_dims[i] = in_dim % arg.dim()

    # 返回验证后的批量大小、调整后的维度信息、扁平化后的参数列表和参数结构描述
    return (
        _validate_and_get_batch_size(flat_in_dims, flat_args),
        flat_in_dims,
        flat_args,
        args_spec,
    )
# 为参数列表中每个 Tensor 创建 BatchedTensor。
# 返回可能已经批处理的参数以及批处理大小。
def _create_batched_inputs(
    flat_in_dims: List[Any], flat_args: List[Any], vmap_level: int, args_spec
) -> Tuple:
    # 查看注释 [Ignored _remove_batch_dim, _add_batch_dim]
    # 对于每个参数和对应的输入维度，如果维度为 None，则不进行处理；否则，使用 _add_batch_dim 函数添加批处理维度。
    batched_inputs = [
        arg if in_dim is None else _add_batch_dim(arg, in_dim, vmap_level)
        for in_dim, arg in zip(flat_in_dims, flat_args)
    ]
    # 将批处理后的输入重新组装成原始的数据结构
    return tree_unflatten(batched_inputs, args_spec)


# 可能移除与批处理相关的维度（如果存在）。
def _maybe_remove_batch_dim(name, batched_output, vmap_level, batch_size, out_dim):
    if out_dim is None:
        if isinstance(batched_output, torch.Tensor) and is_batchedtensor(
            batched_output
        ):
            # 如果输出维度为 None，但是输出是一个 BatchedTensor，则抛出异常。
            raise ValueError(
                f"vmap({name}, ...): `{name}` can not return a "
                f"BatchedTensor when out_dim is None"
            )
        return batched_output

    # 如果输出维度不为 None
    if not isinstance(batched_output, torch.Tensor):
        # 如果输出不是 Tensor 类型，则抛出异常。
        raise ValueError(
            f"vmap({name}, ...): `{name}` must only return "
            f"Tensors, got type {type(batched_output)}. "
            "Did you mean to set out_dims= to None for output?"
        )

    # 否则，使用 _remove_batch_dim 函数移除批处理相关的维度。
    return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)


# 取消与 `vmap_level` 相关的批处理（以及任何批处理维度）。
def _unwrap_batched(
    batched_outputs: Union[Tensor, Tuple[Tensor, ...]],
    out_dims: out_dims_t,
    vmap_level: int,
    batch_size: int,
    func: Callable,
) -> Tuple:
    # 将批处理后的输出展平，并获取输出的结构描述。
    flat_batched_outputs, output_spec = tree_flatten(batched_outputs)

    # 定义不兼容的错误情况处理函数。
    def incompatible_error():
        raise ValueError(
            f"vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): "
            f"out_dims is not compatible with the structure of `outputs`. "
            f"out_dims has structure {tree_flatten(out_dims)[1]} but outputs "
            f"has structure {output_spec}."
        )

    # 如果输出是单个 Tensor 的情况
    if isinstance(batched_outputs, torch.Tensor):
        # 处理某些特殊情况，这里需要详细处理，参见 test_out_dims_edge_case
        if isinstance(out_dims, int):
            flat_out_dims = [out_dims]
        elif isinstance(out_dims, tuple) and len(out_dims) == 1:
            flat_out_dims = out_dims
        elif out_dims is None:
            flat_out_dims = [out_dims]
        else:
            incompatible_error()
    else:
        # 如果输出不是单个 Tensor，则展平 out_dims，并将其扩展成与输出结构一致的形式。
        flat_out_dims = _broadcast_to_and_flatten(out_dims, output_spec)
        if flat_out_dims is None:
            incompatible_error()

    # 对每个批处理后的输出和对应的输出维度，使用 _maybe_remove_batch_dim 函数进行处理。
    flat_outputs = [
        _maybe_remove_batch_dim(
            _get_name(func), batched_output, vmap_level, batch_size, out_dim
        )
        for batched_output, out_dim in zip(flat_batched_outputs, flat_out_dims)
    ]
    # 将处理后的输出重新组装成原始的数据结构。
    return tree_unflatten(flat_outputs, output_spec)


# 检查 x 是否为整数或 None。
def _check_int_or_none(x, func, out_dims):
    if isinstance(x, int):
        return
    # 如果变量 x 的值为 None，则直接返回，不进行后续操作
    if x is None:
        return
    # 抛出值错误异常，描述 vmap(func_name, ..., out_dims=out_dims) 中的 `out_dims` 参数必须是一个整数、None 或者表示在输出中应出现的 vmapped 维度的整数 Python 集合。
    raise ValueError(
        f"vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must be "
        f"an int, None or a python collection of ints representing where in the outputs the "
        f"vmapped dimension should appear."
    )
# 检查输出维度是否为整数或整数的 Python 树
def _check_out_dims_is_int_or_int_pytree(out_dims: out_dims_t, func: Callable) -> None:
    # 如果 out_dims 是整数，则直接返回，不进行检查
    if isinstance(out_dims, int):
        return
    # 否则，对 out_dims 中的每个元素应用 _check_int_or_none 函数进行检查
    tree_map_(partial(_check_int_or_none, func=func, out_dims=out_dims), out_dims)


# 获取函数的名称，如果函数没有 __name__ 属性，则返回其表示形式
def _get_name(func: Callable):
    if hasattr(func, "__name__"):
        return func.__name__

    # 不是所有的可调用对象都有 __name__ 属性，例如通过 functools.partial 创建的对象或 nn.Module 类的对象就没有 __name__
    return repr(func)


# 是否已经加载了分解
DECOMPOSITIONS_LOADED = False
# 分解操作的线程锁
DECOMPOSITIONS_LOCK = threading.Lock()
# VMAP 分解库，默认为 None
VMAP_DECOMPOSITIONS_LIB = None


# 惰性加载分解，torch.package、Python 3.11 和没有 torch.jit 的环境对分解不兼容时，仅在需要时加载
def lazy_load_decompositions():
    global DECOMPOSITIONS_LOADED
    # 如果已经加载了分解，则直接返回
    if DECOMPOSITIONS_LOADED:
        return

    with DECOMPOSITIONS_LOCK:
        # 双重检查锁，确保在并发环境中仅加载一次分解
        if DECOMPOSITIONS_LOADED:
            return

        # 检查是否需要加载分解的条件：PYTORCH_JIT 环境变量为 "1" 并且 __debug__ 为 True
        if not (os.environ.get("PYTORCH_JIT", "1") == "1" and __debug__):
            DECOMPOSITIONS_LOADED = True
            return

        # 使用替代方式将运算符注册到分解表中
        # _register_jit_decomposition 对某些运算符（例如 addr）无效，因为 torchscript 无法将生成的张量类型进行联合
        # decomp 应该是 OpOverload 类型
        global VMAP_DECOMPOSITIONS_LIB
        VMAP_DECOMPOSITIONS_LIB = torch.library.Library(
            "aten", "IMPL", "FuncTorchBatched"
        )

        from torch._decomp import decomposition_table

        # 注册 Python 分解到 VMAP 分解库中
        def _register_python_decomposition_vmap(decomp):
            if decomp in decomposition_table:
                VMAP_DECOMPOSITIONS_LIB.impl(decomp, decomposition_table[decomp])
            else:
                raise RuntimeError(f"could not find decomposition for {decomp}")

        # 注册一些常见运算符的分解函数
        _register_python_decomposition_vmap(torch.ops.aten.mse_loss_backward.default)
        _register_python_decomposition_vmap(
            torch.ops.aten.smooth_l1_loss_backward.default
        )
        _register_python_decomposition_vmap(torch.ops.aten.huber_loss_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss_forward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_forward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.addr.default)

        DECOMPOSITIONS_LOADED = True


# 执行 vmap 实现的函数，加载分解表，并检查输出维度是否为整数或整数的 Python 树
def vmap_impl(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs):
    lazy_load_decompositions()
    _check_out_dims_is_int_or_int_pytree(out_dims, func)
    # 处理批处理输入，获取批次大小、扁平化的输入维度、扁平化的参数以及参数规范
    batch_size, flat_in_dims, flat_args, args_spec = _process_batched_inputs(
        in_dims, args, func
    )
    # 如果指定了 chunk_size 参数，则进行分块处理
    chunks_flat_args = _get_chunked_inputs(
        flat_args, flat_in_dims, batch_size, chunk_size
    )
    # 使用分块后的参数执行 _chunked_vmap 函数
    return _chunked_vmap(
        func,
        flat_in_dims,
        chunks_flat_args,
        args_spec,
        out_dims,
        randomness,
        **kwargs,
    )

    # 如果未指定 chunk_size 参数
    return _flat_vmap(
        func,
        batch_size,
        flat_in_dims,
        flat_args,
        args_spec,
        out_dims,
        randomness,
        **kwargs,
    )
# 计算每个 chunk 的大小列表
def get_chunk_sizes(total_elems, chunk_size):
    # 计算能够整除的 chunk 的数量
    n_chunks = total_elems // chunk_size
    # 初始化每个 chunk 的大小列表为 chunk_size 的重复
    chunk_sizes = [chunk_size] * n_chunks
    # 计算余下的元素数
    remainder = total_elems % chunk_size
    # 如果有余下的元素，则添加一个余下的 chunk
    if remainder != 0:
        chunk_sizes.append(remainder)
    return chunk_sizes


# 将输入分块，并按需拆分
def _get_chunked_inputs(flat_args, flat_in_dims, batch_size, chunk_size):
    # 默认的分割索引
    split_idxs = (batch_size,)
    # 如果指定了 chunk_size，则根据 batch_size 和 chunk_size 计算分块大小
    if chunk_size is not None:
        chunk_sizes = get_chunk_sizes(batch_size, chunk_size)
        split_idxs = tuple(itertools.accumulate(chunk_sizes))

    # 对每个输入张量进行分块操作，根据输入维度信息
    flat_args_chunks = tuple(
        t.tensor_split(split_idxs, dim=in_dim)
        if in_dim is not None
        else [t] * len(split_idxs)
        for t, in_dim in zip(flat_args, flat_in_dims)
    )

    # 转置 chunk 维度并展平结构，chunks_flat_args 是展平后的输入参数列表
    chunks_flat_args = zip(*flat_args_chunks)
    return chunks_flat_args


# 将分块的输出展平
def _flatten_chunks_output(chunks_output_):
    # chunks_output_ 是分块输出的列表
    flat_chunks_output = []
    arg_spec = None
    for output in chunks_output_:
        # 对每个输出进行展平操作，并获取参数规范
        flat_output, arg_specs = tree_flatten(output)
        flat_chunks_output.append(flat_output)
        # 只保存第一个输出的参数规范
        if arg_spec is None:
            arg_spec = arg_specs

    # 转置 chunk 维度并展平结构，flat_output_chunks 是展平后的输出列表
    flat_output_chunks = list(zip(*flat_chunks_output))
    return flat_output_chunks, arg_spec


# 按照指定的 out_dims 和参数规范拼接分块输出
def _concat_chunked_outputs(out_dims, arg_spec, flat_output_chunks):
    # 根据参数规范展开 out_dims，并进行拼接操作
    flat_out_dims = _broadcast_to_and_flatten(out_dims, arg_spec)
    assert len(flat_out_dims) == len(flat_output_chunks)
    flat_output = []
    for idx, out_dim in enumerate(flat_out_dims):
        # 在指定维度上拼接张量
        flat_output.append(torch.cat(flat_output_chunks[idx], dim=out_dim))
        # 释放张量占用的内存
        flat_output_chunks[idx] = None

    return flat_output


# 对 chunked_input 应用 vmap 并返回拼接的输出结果
def _chunked_vmap(
    func, flat_in_dims, chunks_flat_args, args_spec, out_dims, randomness, **kwargs
):
    chunks_output = []
    # 如果 randomness 是 "same"，则保存当前的随机数状态
    rs = torch.get_rng_state() if randomness == "same" else None
    for flat_args in chunks_flat_args:
        # 验证并获取批处理大小
        batch_size = _validate_and_get_batch_size(flat_in_dims, flat_args)

        # 如果批处理大小为0，则跳过当前循环，不进行任何计算
        # 例如：
        # >>> chunk_size = 1
        # >>> batch_size = 6
        # >>> t = torch.zeros(batch_size, 1)
        # >>> t.tensor_split([1, 2, 3, 4, 5, 6])
        # (tensor([[0.]]), tensor([[0.]]), tensor([[0.]]), tensor([[0.]]),
        #  tensor([[0.]]), tensor([[0.]]), tensor([], size=(0, 1)))
        if batch_size == 0:
            continue

        # 如果提供了随机数种子 `rs`，则设置随机数生成器状态
        if rs is not None:
            torch.set_rng_state(rs)

        # 将批处理函数 `_flat_vmap` 应用于当前批次的参数 `flat_args`，
        # 返回扁平化的结果并将其添加到 `chunks_output` 中
        chunks_output.append(
            _flat_vmap(
                func,
                batch_size,
                flat_in_dims,
                flat_args,
                args_spec,
                out_dims,
                randomness,
                **kwargs,
            )
        )

    # 将 `chunks_output` 中的结果扁平化，并获取参数规范 `arg_spec`
    flat_output_chunks, arg_spec = _flatten_chunks_output(chunks_output)

    # 从 `chunks_output` 中移除对分块输出张量的引用，以便及时释放内存
    del chunks_output

    # 根据输出维度 `out_dims` 和参数规范 `arg_spec`，连接分块输出张量 `flat_output_chunks`
    flat_output = _concat_chunked_outputs(out_dims, arg_spec, flat_output_chunks)

    # 最终，根据参数规范 `arg_spec`，将扁平化的输出 `flat_output` 进行反扁平化
    return tree_unflatten(flat_output, arg_spec)
# Vmap refactored helper functions:
# 检查随机性参数是否合法，只允许 "error", "different", "same" 这三个值
def _check_randomness_arg(randomness):
    if randomness not in ["error", "different", "same"]:
        raise RuntimeError(
            f"Only allowed values for randomness are 'error', 'different', or 'same'. Got {randomness}"
        )


# 创建一个上下文管理器，用于增加 vmap 的嵌套层级
@contextlib.contextmanager
def vmap_increment_nesting(batch_size, randomness):
    try:
        # 调用内部函数 _vmap_increment_nesting 增加 vmap 嵌套层级
        vmap_level = _vmap_increment_nesting(batch_size, randomness)
        # 进入上下文管理器的 yield 语句，返回当前的 vmap 层级
        yield vmap_level
    finally:
        # 在退出上下文管理器后，调用 _vmap_decrement_nesting 减少 vmap 嵌套层级
        _vmap_decrement_nesting()


# `_flat_vmap` 函数不支持保存张量钩子
@doesnt_support_saved_tensors_hooks
def _flat_vmap(
    func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs
):
    # 使用 vmap_increment_nesting 上下文管理器，增加 vmap 嵌套层级
    with vmap_increment_nesting(batch_size, randomness) as vmap_level:
        # 创建批量化输入
        batched_inputs = _create_batched_inputs(
            flat_in_dims, flat_args, vmap_level, args_spec
        )
        # 调用 func 处理批量化输入，获取批量化输出
        batched_outputs = func(*batched_inputs, **kwargs)
        # 对批量化输出进行解封，返回非批量化的输出
        return _unwrap_batched(batched_outputs, out_dims, vmap_level, batch_size, func)


# `restore_vmap` 是一个私有辅助函数，功能类似于 vmap，但有以下不同：
# - 返回值为 (outputs, out_dims) 元组，out_dims 是与 outputs 相同形状的 pytree，
#   包含 Optional[int]，指定如果存在 vmapped 维度，则其在相应输出中的位置。
# - 不对输入的维度或数据进行验证（vmap 期望至少一个张量进行 vmap 操作）。
#   restore_vmap 允许输入中没有任何数据具有 vmap 维度。
# - 不对输出进行验证（vmap 期望仅张量输出）。
#   restore_vmap 允许返回任意类型的输出（不仅限于张量）。
#
# 简而言之，restore_vmap 比 vmap 更通用，并具有稍微不同的 API。这些放宽条件使我们能够在
# vmap 执行中间"暂停"，然后稍后"恢复"（这就是 autograd.Function 中 generate_vmap_rule=True 实现的内容）。
#
# 技术上可以在 vmap 的实现中使用 restore_vmap，但是进行这种重构有些技术上的挑战：
# - vmap 将张量封装代码与错误检查耦合在一起。
# - vmap 的张量解封代码在 C++ 中；我们需要在 Python 中重写部分代码，因为它与 unwrap_batched 重叠。
@doesnt_support_saved_tensors_hooks
def restore_vmap(func, in_dims, batch_size, randomness):
    def inner(*args, **kwargs):
        # 使用 vmap_increment_nesting 上下文管理器，增加 vmap 嵌套层级
        with vmap_increment_nesting(batch_size, randomness) as vmap_level:
            # 包装批量化输入
            batched_inputs = wrap_batched(args, in_dims, vmap_level)
            # 调用 func 处理批量化输入，获取批量化输出
            batched_outputs = func(*batched_inputs, **kwargs)
            # 对批量化输出进行解封，返回非批量化的输出
            return unwrap_batched(batched_outputs, vmap_level)

    return inner


# 将参数 args 包装成批量化输入，使用 bdims 和 level 指定的信息
def wrap_batched(args, bdims, level):
    # 展平参数列表并获取参数规范
    flat_args, spec = tree_flatten(args)
    # 使用 bdims 和 spec 进行广播并展平，返回包含批量化输入的结果
    flat_bdims = _broadcast_to_and_flatten(bdims, spec)
    assert flat_bdims is not None
    # 使用 flat_bdims, flat_args, level 和 spec 创建批量化输入
    result = _create_batched_inputs(flat_bdims, flat_args, level, spec)
    return result


# 解封批量化的 args，使用 level 指定的信息
def unwrap_batched(args, level):
    # 将输入参数args进行扁平化处理，并获取其规范化描述spec
    flat_args, spec = tree_flatten(args)
    
    # 如果扁平化后的参数列表为空，则直接返回原始参数args和空的规范化描述()
    if len(flat_args) == 0:
        return args, ()
    
    # 初始化结果列表，对于每个扁平化后的参数arg：
    # - 如果arg是torch.Tensor类型，则调用torch._C._functorch._unwrap_batched函数解包批处理张量，同时保留层级level信息
    # - 否则，保持arg不变，并设定其对应的bdims为None
    result = [
        torch._C._functorch._unwrap_batched(arg, level)
        if isinstance(arg, torch.Tensor)
        else (arg, None)
        for arg in flat_args
    ]
    
    # 将result中的输出和批处理维度信息bdims分别解压缩为两个独立的元组
    output, bdims = zip(*result)
    
    # 根据规范化描述spec，将解压缩后的输出output和bdims分别重建为原始结构
    return tree_unflatten(output, spec), tree_unflatten(bdims, spec)
```