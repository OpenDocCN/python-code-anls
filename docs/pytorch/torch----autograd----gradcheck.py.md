# `.\pytorch\torch\autograd\gradcheck.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import deprecated

# 导入 PyTorch 相关模块和函数
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors

# 定义公开的接口列表，虽然不打算公开 `get_*_jacobian` 函数，
# 但它们已经暴露在 `__all__` 中，并且我们维护了对它们的向后兼容性
__all__ = [
    "gradcheck",
    "gradgradcheck",
    "GradcheckError",
    "get_numerical_jacobian",
    "get_analytical_jacobian",
    "get_numerical_jacobian_wrt_specific_input",
]

# 自定义异常类，由 `gradcheck` 和 `gradgradcheck` 函数抛出
class GradcheckError(RuntimeError):
    r"""Error raised by :func:`gradcheck` and :func:`gradgradcheck`."""
    pass

# 判断是否稀疏压缩张量的辅助函数
def _is_sparse_compressed_tensor(obj: torch.Tensor):
    return obj.layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }

# 判断是否任意稀疏张量的辅助函数
def _is_sparse_any_tensor(obj: torch.Tensor):
    return _is_sparse_compressed_tensor(obj) or obj.layout is torch.sparse_coo

# 判断是否浮点型或复数型张量的辅助函数
def _is_float_or_complex_tensor(obj):
    return is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())

# 根据输入张量分配雅可比矩阵的存储空间，返回元组形式的张量列表
def _allocate_jacobians_with_inputs(
    input_tensors: Tuple, numel_output
) -> Tuple[torch.Tensor, ...]:
    # 从输入张量创建零填充张量。如果 `numel_output` 不为 None，
    # 对于 `input_tensors` 中的每个张量，返回一个高为 `t.numel`，宽为 `numel_output` 的新零填充张量。
    # 否则，对于每个张量，返回一个尺寸为 `(t.numel,)` 的一维张量。
    # 每个新张量的布局、dtype 和设备与相应输入张量相同。
    out: List[torch.Tensor] = []
    for t in input_tensors:
        if _is_float_or_complex_tensor(t) and t.requires_grad:
            out.append(t.new_zeros((t.numel(), numel_output), layout=torch.strided))
    return tuple(out)

# 根据输出张量分配雅可比矩阵的存储空间，返回元组形式的张量列表
def _allocate_jacobians_with_outputs(
    output_tensors: Tuple, numel_input, dtype=None, device=None
) -> Tuple[torch.Tensor, ...]:
    # 从输出张量创建零填充张量。如果 `numel_input` 不为 None，
    # 对于 `output_tensors` 中的每个张量，返回一个高为 `numel_input`，宽为 `t.numel` 的新零填充张量。
    # 否则，对于每个张量，返回一个尺寸为 `(t.numel,)` 的一维张量。
    # 每个新张量的布局、dtype 和设备由参数指定或与相应输入张量相同。
    out: List[torch.Tensor] = []
    options = {"dtype": dtype, "device": device, "layout": torch.strided}
    for t in output_tensors:
        if _is_float_or_complex_tensor(t):
            out.append(t.new_zeros((numel_input, t.numel()), **options))
    return tuple(out)

# 遍历张量或张量可迭代对象的辅助函数
def _iter_tensors(
    x: Union[torch.Tensor, Iterable[torch.Tensor]], only_requiring_grad: bool = False
) -> Iterable[torch.Tensor]:
    # 返回一个迭代器，遍历输入的张量或张量可迭代对象。
    # 如果 `only_requiring_grad` 为 True，则只返回需要梯度的张量。
    pass  # Placeholder, 实际功能未在代码段中给出
    # 检查变量 x 是否类似张量
    if is_tensor_like(x):
        # 如果 mypy 未能将 `x` 缩小到 torch.Tensor 类型
        # 检查张量是否需要梯度或者不仅仅需要梯度
        if x.requires_grad or not only_requiring_grad:  # type: ignore[union-attr]
            # 生成并返回张量 x
            yield x  # type: ignore[misc]
    # 如果 x 是一个可迭代对象且不是字符串类型
    elif isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        # 对 x 中的每个元素进行迭代
        for elem in x:
            # 递归调用 _iter_tensors 函数，生成并返回元素 elem 中的张量
            yield from _iter_tensors(elem, only_requiring_grad)
# 返回稀疏张量 x 的稠密版本副本，未指定的元素均被零值元素替换
def _densify(x):
    if isinstance(x, (list, tuple)):
        # 如果 x 是列表或元组，则递归地对每个元素调用 _densify，并返回相同类型的对象
        return type(x)(map(_densify, x))
    elif not is_tensor_like(x) or x.layout in {torch.strided, torch._mkldnn}:  # type: ignore[attr-defined] # no attr _mkldnn
        # 如果 x 不像张量或者布局为 torch.strided 或 torch._mkldnn，则直接返回 x
        return x
    elif x.layout is torch.sparse_coo:
        # 如果 x 的布局为 COO 稀疏张量
        device = x.device
        indices_dtype = x._indices().dtype
        # 创建形状与 x 的稀疏维度相同的全为 1 的张量 tmp
        tmp = torch.ones(x.shape[: x.sparse_dim()], dtype=torch.int8, device=device)
        # 根据 tmp 生成非零元素的索引 indices，并转换为指定的 dtype
        indices = tmp.nonzero().t().to(dtype=indices_dtype)
        # 创建值全为零的稠密张量 values
        values = torch.zeros(
            (tmp.numel(), *x.shape[x.sparse_dim() :]), dtype=x.dtype, device=device
        )
        # 调用 x 的 detach() 方法并且 coalesce() 合并非零元素
        x_coalesced = x.detach().coalesce()
        if x_coalesced.numel() > 0:
            # 计算扁平索引 flat_indices
            stride = tmp.stride()
            flat_indices = (
                x_coalesced.indices()
                .mul(
                    torch.tensor(stride, dtype=indices_dtype, device=device).unsqueeze(
                        1
                    )
                )
                .sum(0)
            )
            # 将合并后的值放入 values 中对应的位置
            values[flat_indices] = x_coalesced.values()
        # 创建稀疏 COO 张量并进行进一步的操作
        return (
            torch.sparse_coo_tensor(indices, values, x.shape)
            ._coalesced_(True)
            .requires_grad_(x.requires_grad)
        )
    elif _is_sparse_compressed_tensor(x):
        # 如果 x 是压缩稀疏张量
        blocksize = (
            x.values().shape[1:3]
            if x.layout in {torch.sparse_bsr, torch.sparse_bsc}
            else None
        )
        compressed_indices = (
            x.crow_indices()
            if x.layout in {torch.sparse_csr, torch.sparse_bsr}
            else x.ccol_indices()
        )
        # 为简单起见，使用中间稀疏 COO 张量 r
        r = _densify(x.detach().to_sparse(layout=torch.sparse_coo)).to_sparse(
            layout=x.layout, blocksize=blocksize
        )
        # 检查所有元素是否在 to_sparse 操作后仍然被指定
        dense_numel = r.values().numel() // max(1, r.values().shape[0])
        batch_numel = compressed_indices.numel() // compressed_indices.shape[-1]
        sparse_numel = r.numel() // max(1, dense_numel * batch_numel)
        if sparse_numel != r._nnz():
            # 如果稀疏元素数不等于 _nnz() 返回的数量，抛出断言错误
            raise AssertionError(
                f"{x.layout} densify failed: expected nnz={sparse_numel} but got {r._nnz()}"
            )
        return r.requires_grad_(x.requires_grad)
    elif _is_sparse_any_tensor(x):
        # 如果 x 是任何类型的稀疏张量，则抛出 NotImplementedError
        raise NotImplementedError(x.layout)
    # 如果 x 不属于以上任何类型，直接返回 x
    return x


# (仅用于慢速 gradcheck) 返回一个生成器，每次迭代返回以下元素之一：
#  1) 一个张量：在所有迭代中返回相同的张量。该张量与输入的原始 x_tensor 不同，
#     准备为其支持原地修改。根据输入张量的是否分步、稀疏或密集，返回的张量可能会有所不同。
def _iter_tensor(x_tensor):
    # 如果输入张量 x_tensor 使用了 torch._mkldnn 布局（DNNL 张量布局），需要特殊处理
    elif x_tensor.layout == torch._mkldnn:  # type: ignore[attr-defined]
        # 使用 product 函数生成所有可能的索引组合，遍历张量的每个元素
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
            # 由于没有实现直接的索引操作，需要先将稀疏的 x_tensor 转换为密集张量
            x_tensor_dense = x_tensor.to_dense()
            # 生成器返回：密集张量 x_tensor_dense，当前索引 x_idx，以及平铺的索引 d_idx
            yield x_tensor_dense, x_idx, d_idx
    else:
        # 对于其他情况，直接使用 .data 获取张量数据，避免版本检查问题
        x_tensor = x_tensor.data
        # 继续使用 product 函数生成所有可能的索引组合，遍历张量的每个元素
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
            # 生成器返回：原始张量 x_tensor，当前索引 x_idx，以及平铺的索引 d_idx
            yield x_tensor, x_idx, d_idx
# 定义一个函数 `_get_numerical_jacobian`，用于计算数值雅可比矩阵。
# 返回类型为列表，其中每个元素是包含多个张量的元组。
# 函数签名包括 `fn`（待计算雅可比矩阵的函数）、`inputs`（传递给 `fn` 的输入）、
# `outputs`（可选参数，避免多次调用 `fn`）、`target`（雅可比矩阵相对于其计算的张量，默认为 `inputs`）、
# `eps`（有限差分期间的扰动幅度，默认为 `1e-3`）、`is_forward_ad`（是否对前向自动微分梯度计算数值雅可比矩阵的错误检查标志，默认为 `False`）。
def _get_numerical_jacobian(
    fn, inputs, outputs=None, target=None, eps=1e-3, is_forward_ad=False
) -> List[Tuple[torch.Tensor, ...]]:
    """Compute the numerical Jacobian of `fn(inputs)` with respect to `target`.

    If not specified, targets are the input. Returns M * N Jacobians where N is the
    number of tensors in target that require grad and M is the number of non-integral
    outputs.

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        target: the Tensors wrt whom Jacobians are calculated (default=`inputs`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)
        is_forward_ad: if this numerical jacobian is computed to be checked wrt
                       forward AD gradients (this is used for error checking only)

    Returns:
        A list of M N-tuples of tensors

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    # 初始化一个空列表，用于存储计算得到的雅可比矩阵
    jacobians: List[Tuple[torch.Tensor, ...]] = []
    # 如果未提供预计算的输出，则调用 `fn` 函数获取输出
    if outputs is None:
        outputs = _as_tuple(fn(*_as_tuple(inputs)))
    # 如果不是前向自动微分，并且输出中有任何复数，则抛出异常
    if not is_forward_ad and any(o.is_complex() for o in outputs):
        raise ValueError(
            "Expected output to be non-complex. get_numerical_jacobian no "
            "longer supports functions that return complex outputs."
        )
    # 如果未指定 `target`，则默认为 `inputs`
    if target is None:
        target = inputs
    # 获取需要梯度的目标张量的索引列表
    inp_indices = [
        i for i, a in enumerate(target) if is_tensor_like(a) and a.requires_grad
    ]
    # 遍历 `target` 中的张量及其索引，计算相对于每个输入的数值雅可比矩阵
    for i, (inp, inp_idx) in enumerate(zip(_iter_tensors(target, True), inp_indices)):
        jacobians += [
            get_numerical_jacobian_wrt_specific_input(
                fn,
                inp_idx,
                inputs,
                outputs,
                eps,
                input=inp,
                is_forward_ad=is_forward_ad,
            )
        ]
    # 返回计算得到的雅可比矩阵列表
    return jacobians


# 使用 `deprecated` 装饰器标记函数 `get_numerical_jacobian`，它是 PyTorch 的私有 API，
# 不应暴露给用户使用，并且在将来的版本中会被移除。
@deprecated(
    "`get_numerical_jacobian` was part of PyTorch's private API and not "
    "meant to be exposed. We are deprecating it and it will be removed "
    "in a future version of PyTorch. If you have a specific use for "
    "this or feature request for this to be a stable API, please file "
    "us an issue at https://github.com/pytorch/pytorch/issues/new",
    category=FutureWarning,
)
# 定义函数 `get_numerical_jacobian`，用于计算给定函数 `fn` 和其输入的数值雅可比矩阵。
# 该函数已被标记为已弃用的 API。
# 函数签名包括 `fn`（要计算雅可比矩阵的函数，必须将输入作为元组传递）、
# `inputs`（传递给 `fn` 的输入）、`target`（雅可比矩阵相对于其计算的张量，默认为 `inputs`）、
# `eps`（有限差分期间的扰动幅度，默认为 `1e-3`）、`grad_out`（输出梯度，默认为 `1.0`）。
def get_numerical_jacobian(fn, inputs, target=None, eps=1e-3, grad_out=1.0):
    """Compute the numerical Jacobian for a given fn and its inputs.

    This is a Deprecated API.

    Args:
        fn: the function to compute the Jacobian for (must take inputs as a tuple)
        input: input to `fn`
        target: the Tensors wrt whom Jacobians are calculated (default=`input`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)
    """
    # 检查 grad_out 参数是否为 1.0，这是为了保持向后兼容性
    if (
        grad_out != 1.0
    ):  
        # 如果 grad_out 不为 1.0，则抛出 ValueError 异常
        raise ValueError(
            "Expected grad_out to be 1.0. get_numerical_jacobian no longer "
            "supports values of grad_out != 1.0."
        )

    # 定义一个函数 fn_pack_inps，将输入打包并调用 fn 函数
    def fn_pack_inps(*inps):
        return fn(inps)

    # 调用 _get_numerical_jacobian 函数，计算 fn_pack_inps 对于输入和目标的数值雅可比矩阵
    jacobians = _get_numerical_jacobian(fn_pack_inps, inputs, None, target, eps)

    # 返回每个输出的雅可比矩阵的第一个元素，组成一个元组返回
    return tuple(jacobian_for_each_output[0] for jacobian_for_each_output in jacobians)
# 计算数值梯度的函数，使用有限差分法计算函数 `fn` 在输入 `entry` 处沿向量 `v` 的方向导数。
def _compute_numerical_gradient(fn, entry, v, norm_v, nbhd_checks_fn):
    if _is_sparse_compressed_tensor(entry):
        # 如果输入是稀疏压缩张量，则需要进行特殊处理，因为它们不支持常规的加减操作。
        assert entry.layout == v.layout, (entry.layout, v.layout)
        assert entry._nnz() == v._nnz(), (entry._nnz(), v._nnz(), entry.shape)
        # 由于稀疏张量有限支持，只能对值进行有限差分计算。
        entry = entry.values()
        v = v.values()
        # 使用 detach 避免稀疏张量的反向计算问题。
        entry = entry.detach()

    orig = entry.clone()
    # 计算 fn() 在 entry - v 处的输出
    entry.copy_(orig - v)
    outa = fn()
    # 计算 fn() 在 entry + v 处的输出
    entry.copy_(orig + v)
    outb = fn()
    entry.copy_(orig)

    # 定义计算数值梯度的函数
    def compute(a, b):
        nbhd_checks_fn(a, b)
        # 使用中心差分法来近似计算梯度
        ret = (b - a) / (2 * norm_v)
        return ret.detach().reshape(-1)

    # 返回由 (outa[i] - outb[i]) / (2 * norm_v) 组成的元组
    return tuple(compute(a, b) for (a, b) in zip(outa, outb))


# 计算对特定输入的数值雅可比向量积（JVP）
def _compute_numerical_jvps_wrt_specific_input(
    jvp_fn, delta, input_is_complex, is_forward_ad=False
) -> List[torch.Tensor]:
    # 对实数 delta 进行 JVP 计算
    jvps: List[torch.Tensor] = []
    ds_dx_tup = jvp_fn(delta[0] if isinstance(delta, tuple) else delta)

    if input_is_complex:  # 复数输入时的处理 C -> R
        ds_dy_tup = (
            jvp_fn(delta[1] * 1j) if isinstance(delta, tuple) else jvp_fn(delta * 1j)
        )
        for ds_dx, ds_dy in zip(ds_dx_tup, ds_dy_tup):
            assert not ds_dx.is_complex()
            # 对共轭 Wirtinger 导数进行计算
            conj_w_d = ds_dx + ds_dy * 1j
            jvps.append(conj_w_d)
    else:
        for ds_dx in ds_dx_tup:  # 实数输入时的处理 R -> R 或者 (R -> C 对于前向 AD 情况)
            assert is_forward_ad or not ds_dx.is_complex()
            jvps.append(ds_dx)
    return jvps


# 合并雅可比矩阵的列，根据列索引映射到输出索引，返回完整的雅可比矩阵列表
def _combine_jacobian_cols(
    jacobians_cols: Dict[int, List[torch.Tensor]], outputs, input, numel
) -> Tuple[torch.Tensor, ...]:
    # jacobians_cols 映射列索引 -> 输出索引 -> 雅可比矩阵的单列
    # 返回一个列表，将输出索引映射到完整的雅可比矩阵张量
    jacobians = _allocate_jacobians_with_outputs(
        outputs, numel, dtype=input.dtype if input.dtype.is_complex else None
    )
    for i, jacobian in enumerate(jacobians):
        for k, v in jacobians_cols.items():
            jacobian[k] = v[i]
    return jacobians


# 准备输入数据，可能包括扰动的输入，根据快速模式返回一个张量
def _prepare_input(
    input: torch.Tensor, maybe_perturbed_input: Optional[torch.Tensor], fast_mode=False
) -> torch.Tensor:
    # 如果输入的布局是 torch._mkldnn，则执行以下操作（忽略类型定义中的属性）
    if input.layout == torch._mkldnn:  # type: ignore[attr-defined] # no attr _mkldnn
        # 如果可能被扰动的输入不为空，则将其转换回 mkldnn 格式
        if maybe_perturbed_input is not None:
            return maybe_perturbed_input.to_mkldnn()
        else:
            # 否则直接返回原始输入
            return input
    # 如果输入是稀疏张量的任何形式
    elif _is_sparse_any_tensor(input):
        # 如果启用了快速模式且可能被扰动的输入不为空
        if fast_mode and maybe_perturbed_input is not None:
            # 返回已经是原始张量的“克隆”版本，因此对克隆版本的更改不会反映在输入上
            return maybe_perturbed_input
        else:
            # 否则返回原始输入
            return input
    else:
        # 如果不满足以上条件，则说明不能使用 entry (input.data) 来支持 gradgrad 的工作
        # 因为在 gradgrad 情况下，fn 需要计算对输入的二阶梯度
        return input
# 检查返回的输出在扰动输入时，是否具有相同的数据类型和形状
def _check_outputs_same_dtype_and_shape(output1, output2, eps, idx=None) -> None:
    # 如果提供了索引，将其格式化为字符串以包含在消息中
    on_index = "on index {idx} " if idx is not None else ""
    # 断言输出1和输出2具有相同的形状
    assert output1.shape == output2.shape, (
        f"Expected `func` to return outputs with the same shape"
        f" when inputs are perturbed {on_index}by {eps}, but got:"
        f" shapes {output1.shape} and {output2.shape}."
    )
    # 断言输出1和输出2具有相同的数据类型
    assert output1.dtype == output2.dtype, (
        f"Expected `func` to return outputs with the same dtype"
        f" when inputs are perturbed {on_index}by {eps}, but got:"
        f" dtypes {output1.dtype} and {output2.dtype}."
    )


# 计算相对于特定输入的数值雅可比矩阵
def get_numerical_jacobian_wrt_specific_input(
    fn, input_idx, inputs, outputs, eps, input=None, is_forward_ad=False
) -> Tuple[torch.Tensor, ...]:
    # 返回 N 个雅可比矩阵，其中 N 是输出的数量。使用字典存储雅可比矩阵列，因为稀疏输入的索引不一定是连续的
    jacobian_cols: Dict[int, List[torch.Tensor]] = {}
    # 如果没有提供输入，使用给定索引从输入列表中获取输入
    input = inputs[input_idx] if input is None else input
    # 断言输入需要梯度计算
    assert input.requires_grad
    # 遍历输入张量的每个元素及其索引和偏导数索引
    for x, idx, d_idx in _iter_tensor(input):
        # 使用准备好的输入创建封装函数
        wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, x)
        # 获取要扰动的输入元素
        input_to_perturb = x[idx]
        # 部分函数，用于检查输出的数据类型和形状是否相同
        nbhd_checks_fn = functools.partial(
            _check_outputs_same_dtype_and_shape, idx=idx, eps=eps
        )
        # 获取数值雅可比向量积函数
        jvp_fn = _get_numerical_jvp_fn(
            wrapped_fn, input_to_perturb, eps, nbhd_checks_fn
        )
        # 计算相对于特定输入的数值雅可比向量积
        jacobian_cols[d_idx] = _compute_numerical_jvps_wrt_specific_input(
            jvp_fn, eps, x.is_complex(), is_forward_ad
        )
    # 合并所有雅可比矩阵列，并返回结果
    return _combine_jacobian_cols(jacobian_cols, outputs, input, input.numel())


# 使用前向模式自动微分计算解析雅可比矩阵
def _get_analytical_jacobian_forward_ad(
    fn, inputs, outputs, *, check_grad_dtypes=False, all_u=None
) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    """使用前向模式自动微分计算`fn(inputs)`相对于`target`的解析雅可比矩阵。

    返回 N * M 个雅可比矩阵，其中 N 是需要梯度计算的目标张量数量，M 是非整数输出的数量。
    与此文件中的其他函数不同，该函数要求函数实际使用传递的输入。
    如果函数通过副作用捕获输入而不是使用传递的输入，则计算的值可能会错误（许多 torch.nn 测试会这样做）。
    """
    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        check_grad_dtypes: if True, will check that the gradient dtype are valid
        all_u (optional): if provided, the Jacobian will be right multiplied with this vector

    Returns:
        A tuple of M N-tuples of tensors
    """
    # To avoid early import issues
    fwAD = torch.autograd.forward_ad

    # Filter out inputs that are tensor-like and require gradient
    tensor_inputs = tuple(i for i in inputs if is_tensor_like(i) and i.requires_grad)

    # Check if any tensor input is complex; raise error if so
    if any(i.is_complex() for i in tensor_inputs):
        raise ValueError(
            "Expected inputs to be non-complex for _get_analytical_jacobian_forward_ad."
        )

    # Allocate space for Jacobians based on inputs and outputs
    if all_u:
        jacobians = tuple(
            _allocate_jacobians_with_outputs(outputs, 1) for i in tensor_inputs
        )
    else:
        jacobians = tuple(
            _allocate_jacobians_with_outputs(outputs, i.numel()) for i in tensor_inputs
        )

    # Return the tuple of Jacobians
    return jacobians
# 准备用于扰动的输入数据，根据需要进行修改
def _get_input_to_perturb(input):
    # 如果输入的布局是torch._mkldnn，将其转换为稠密张量，以便进行需要步幅的操作
    if input.layout == torch._mkldnn:  # type: ignore[attr-defined] # no attr _mkldnn
        input_to_perturb = input.to_dense()
    elif _is_sparse_any_tensor(input):
        # 克隆稀疏张量，因为输入可能需要梯度，而copy_方法调用resize_，而.data不允许调用resize_
        input_to_perturb = input.clone()
    else:
        # 否则直接使用input的数据部分
        input_to_perturb = input.data
    return input_to_perturb


# 包装函数fn，使其输入已预先提供
def _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, fast_mode=False):
    def wrapped_fn():
        inp = tuple(
            # 准备输入，如果是张量并且是目标索引，则使用input_to_perturb进行准备，否则为None
            _prepare_input(a, input_to_perturb if i == input_idx else None, fast_mode)
            if is_tensor_like(a)
            else a
            for i, a in enumerate(_as_tuple(inputs))
        )
        return tuple(a.clone() for a in _as_tuple(fn(*inp)))

    return wrapped_fn


# 包装jvp_fn，使某些参数已预先提供
def _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn):
    def jvp_fn(delta):
        # 计算数值梯度
        return _compute_numerical_gradient(
            wrapped_fn, input_to_perturb, delta, eps, nbhd_checks_fn
        )

    return jvp_fn


# 重塑张量或元组u为给定的形状
def _reshape_tensor_or_tuple(u, shape):
    # 如果u是元组
    if isinstance(u, tuple):
        # 如果u[0]不是稀疏张量，则对元组中的每个张量进行形状重塑
        if not _is_sparse_any_tensor(u[0]):
            return (u[0].reshape(shape), u[1].reshape(shape))
    else:
        # 如果u不是稀疏张量，则对张量进行形状重塑
        if not _is_sparse_any_tensor(u):
            return u.reshape(shape)
    return u


# 将张量或元组u乘以标量k
def _mul_tensor_or_tuple(u, k):
    if isinstance(u, tuple):
        return (k * u[0], k * u[1])
    else:
        return k * u


# 获取关于特定输入的数值JVP（Jacobian Vector Product）
def _get_numerical_jvp_wrt_specific_input(
    fn, input_idx, inputs, u, eps, is_forward_ad=False
) -> List[torch.Tensor]:
    # 获取特定索引的输入
    input = inputs[input_idx]
    # 准备用于扰动的输入数据
    input_to_perturb = _get_input_to_perturb(input)
    # 包装函数fn，使其输入已预先提供
    wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, True)
    # 部分函数，检查输出的dtype和shape是否相同
    nbhd_checks_fn = functools.partial(_check_outputs_same_dtype_and_shape, eps=eps)
    # 获取数值JVP函数
    jvp_fn = _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
    # 调整u的形状以匹配input_to_perturb的形状，并乘以eps
    u = _reshape_tensor_or_tuple(u, input_to_perturb.shape)
    u = _mul_tensor_or_tuple(u, eps)
    # 计算关于特定输入的数值JVP
    return _compute_numerical_jvps_wrt_specific_input(
        jvp_fn, u, input.is_complex(), is_forward_ad
    )


# 获取关于输入的数值vJu（Jacobian-Vector Product）
def _get_numerical_vJu(
    fn, inputs, inp_indices, func_out, all_u, all_v, eps, is_forward_ad
):
    # 注意：all_v也可能为None，在这种情况下，此函数仅计算Ju。
    reduced_jacobians: List[List[torch.Tensor]] = []
    for i, (inp_idx, u) in enumerate(zip(inp_indices, all_u)):
        # 遍历输入索引和对应的值
        # 调用函数 `_get_numerical_jvp_wrt_specific_input` 获取关于特定输入的数值雅可比向量对应于给定的输入
        all_Ju = _get_numerical_jvp_wrt_specific_input(
            fn, inp_idx, inputs, u, eps, is_forward_ad
        )
        # 过滤掉非浮点数输出的 Ju
        filtered_Ju = []
        # 将函数输出转换为元组
        func_out = _as_tuple(func_out)
        assert len(all_Ju) == len(func_out)  # 断言确保 all_Ju 和 func_out 的长度相等
        for Ju, output in zip(all_Ju, func_out):
            if _is_float_or_complex_tensor(output):
                filtered_Ju.append(Ju)  # 如果输出是浮点数或复数张量，则添加 Ju 到 filtered_Ju 中
            else:
                # TODO: 处理其他类型的 Ju，暂时未实现
                pass
        if all_v is not None:
            jacobian_scalars: List[torch.Tensor] = []
            # 对于每对 v 和 filtered_Ju，计算其点乘并进行类型提升
            for v, Ju in zip(all_v, filtered_Ju):
                jacobian_scalars.append(_dot_with_type_promotion(v, Ju))
            reduced_jacobians.append(jacobian_scalars)  # 将计算得到的 Jacobian 标量添加到 reduced_jacobians 中
        else:
            reduced_jacobians.append(filtered_Ju)  # 将过滤后的 Ju 添加到 reduced_jacobians 中
    return reduced_jacobians  # 返回最终的 reduced_jacobians 结果
# 检查两个雅可比张量之间的最大差异是否在某个公差 `atol` 内
def _check_jacobians_equal(j1, j2, atol):
    for j1_x, j2_x in zip(j1, j2):
        if j1_x.numel() != 0 and (j1_x - j2_x).abs().max() > atol:
            return False
    return True

# 对内部张量列表的第i个张量进行堆叠和检查，检查其与第i个可微分输入的大小和dtype是否相同
def _stack_and_check_tensors(
    list_of_list_of_tensors, inputs, numel_outputs
) -> Tuple[Tuple[torch.Tensor, ...], bool, bool]:
    out_jacobians = _allocate_jacobians_with_inputs(inputs, numel_outputs)  # 分配与输入相匹配的雅可比矩阵
    diff_input_list = list(_iter_tensors(inputs, True))  # 获取可微分输入的列表
    correct_grad_sizes = True  # 标志位，表示梯度大小是否正确
    correct_grad_types = True  # 标志位，表示梯度类型是否正确
    for i, tensor_list in enumerate(list_of_list_of_tensors):
        inp = diff_input_list[i]
        out_jacobian = out_jacobians[i]
        for j, tensor in enumerate(tensor_list):
            if tensor is not None and tensor.size() != inp.size():
                correct_grad_sizes = False
            elif tensor is not None and tensor.dtype != inp.dtype:
                correct_grad_types = False
            if tensor is None:
                out_jacobian[:, j].zero_()
            else:
                dense = (
                    tensor.to_dense() if not tensor.layout == torch.strided else tensor
                )
                assert out_jacobian[:, j].numel() == dense.numel()
                out_jacobian[:, j] = dense.reshape(-1)
    return out_jacobians, correct_grad_sizes, correct_grad_types

# 非确定性失败消息，用于指出操作依赖非确定性操作的情况下可能失败
FAILED_NONDET_MSG = """\n
NOTE: If your op relies on non-deterministic operations i.e., it is listed here:
https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
this failure might be expected.

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck.
If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `nondet_tol=<tol>` as a keyword argument.
- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test
  to have `gradcheck_nondet_tol=<tol>`.
- is a Module test (e.g., in common_nn.py), then modify the corresponding
  module_test entry to have `gradcheck_nondet_tol=<tol>`
"""

# 检查解析雅可比矩阵的属性，用于快速和慢速模式
# - 对于慢速模式，vjps[i][j] 是相对于第i个输入的第j行雅可比矩阵
# - 对于快速模式，vjps[i][0] 是相对于第i个输入的雅可比矩阵行的线性组合
def _check_analytical_jacobian_attributes(
    inputs, output, nondet_tol, check_grad_dtypes, fast_mode=False, v=None
) -> Tuple[torch.Tensor, ...]:
    diff_input_list = list(_iter_tensors(inputs, True))  # 获取可微分输入的列表

    def vjp_fn(grad_output):
        return torch.autograd.grad(
            output, diff_input_list, grad_output, retain_graph=True, allow_unused=True
        )
    # 如果开启了快速模式，对输出值进行两次计算以检查重入性（非确定性）
    if fast_mode:
        # 使用特定输出值的梯度函数计算解析雅可比矩阵的两个版本
        vjps1 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
        vjps2 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
    else:
        # 使用默认模式计算解析雅可比矩阵的两个版本
        vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
        vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    # 计算输出张量的元素数量，如果使用快速模式，则为1
    output_numel = output.numel() if not fast_mode else 1

    # 将计算得到的两个雅可比矩阵堆叠并检查张量类型和大小
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(
        vjps1, inputs, output_numel
    )
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)

    # 检查两个雅可比矩阵是否相等，用于判断反向传播的重入性
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)

    # 如果类型不匹配且开启了梯度类型检查，则抛出异常
    if not types_ok and check_grad_dtypes:
        raise GradcheckError("Gradient has dtype mismatch")
    
    # 如果大小不正确，则抛出异常
    if not sizes_ok:
        raise GradcheckError("Analytical gradient has incorrect size")
    
    # 如果不满足重入性要求，则抛出异常，并说明非确定性的公差值
    if not reentrant:
        raise GradcheckError(
            "Backward is not reentrant, i.e., running backward with "
            "same input and grad_output multiple times gives different values, "
            "although analytical gradient matches numerical gradient."
            f"The tolerance for nondeterminism was {nondet_tol}." + FAILED_NONDET_MSG
        )

    # 返回第一个计算得到的雅可比矩阵作为结果
    return jacobians1
# 定义函数_get_analytical_vJu_backward_mode，用于计算反向模式下的分析雅可比矩阵
def _get_analytical_vJu_backward_mode(
    inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u
):
    # 初始化减少的雅可比矩阵列表
    reduced_jacobians: List[List[torch.Tensor]] = []
    
    # 遍历输出和相应的敏感向量 v
    for output, v in zip(outputs, all_v):
        # 调用函数_check_analytical_jacobian_attributes，获取所有的分析雅可比矩阵
        all_vJ = _check_analytical_jacobian_attributes(
            inputs, output, nondet_tol, check_grad_dtypes, fast_mode=True, v=v
        )
        # 初始化当前输出对应的雅可比矩阵标量列表
        jacobian_scalars: List[torch.Tensor] = []
        
        # 遍历每个分析雅可比矩阵和相应的 u 向量
        for vJ, u in zip(all_vJ, all_u):
            # 对 vJ 进行转置并去除多余的维度，确保 vJ 是一个二维张量以便重用慢模式下的错误检查逻辑
            vJ = vJ.T.squeeze(0)
            
            # 如果 vJ 是复数类型，则进行复数到实数的转换
            if vJ.is_complex():  # C -> R
                # 将复数张量视为实部和虚部的视图，并计算实部和虚部的点乘
                tv = torch.view_as_real(vJ.resolve_conj())
                tr = tv.select(-1, 0)
                ti = tv.select(-1, 1)
                jacobian_scalars.append(tr.dot(u[0]) + 1j * ti.dot(u[1]))
            else:  # 如果 vJ 是实数类型，则直接计算点乘
                jacobian_scalars.append(vJ.dot(u))
        
        # 将当前输出对应的雅可比矩阵标量列表添加到减少的雅可比矩阵列表中
        reduced_jacobians.append(jacobian_scalars)
    
    # 返回减少的雅可比矩阵列表
    return reduced_jacobians


# 使用@deprecated装饰器标记函数get_analytical_jacobian，说明该函数已被废弃
@deprecated(
    "`get_analytical_jacobian` was part of PyTorch's private API and not "
    "meant to be exposed. We are deprecating it and it will be removed "
    "in a future version of PyTorch. If you have a specific use for "
    "this or feature request for this to be a stable API, please file "
    "us an issue at https://github.com/pytorch/pytorch/issues/new",
    category=FutureWarning,
)
# 定义函数get_analytical_jacobian，计算输入和输出之间的分析雅可比矩阵
def get_analytical_jacobian(inputs, output, nondet_tol=0.0, grad_out=1.0):
    # 检查参数 grad_out 是否为 1.0，因为只为了向后兼容性而保留
    if (
        grad_out != 1.0
    ):  # grad_out 参数仅出于向后兼容性原因保留
        raise ValueError(
            "Expected grad_out to be 1.0. get_analytical_jacobian no longer "
            "supports values of grad_out != 1.0."
        )
    
    # 检查输出是否为复数类型，因为不再支持返回复数输出的函数
    if output.is_complex():
        raise ValueError(
            "Expected output to be non-complex. get_analytical_jacobian no "
            "longer supports functions that return complex outputs."
        )
    
    # 获取输入张量列表，用于梯度计算
    diff_input_list = list(_iter_tensors(inputs, True))

    # 定义用于计算 VJP（向量-雅可比积）的函数
    def vjp_fn(grad_output):
        return torch.autograd.grad(
            output, diff_input_list, grad_output, retain_graph=True, allow_unused=True
        )

    # 两次计算分析雅可比行以检查非确定性（重新进入性）
    vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    # 获取输出张量的元素数量
    output_numel = output.numel()
    
    # 堆叠和检查张量，以及检查张量的类型和尺寸是否符合要求
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(
        vjps1, inputs, output_numel
    )
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)
    
    # 检查两次计算的雅可比矩阵是否相等，以确定非确定性
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)

    # 返回分析雅可比矩阵、重新进入性标志、尺寸是否符合要求标志、类型是否符合要求标志
    return jacobians1, reentrant, sizes_ok, types_ok
# 计算单个输入输出对的慢速模式下的解析雅可比矩阵。
# 不执行关于数据类型、形状和重入性的检查。
def _get_analytical_jacobian(inputs, outputs, input_idx, output_idx):
    # 调用函数检查解析雅可比矩阵的属性
    jacobians = _check_analytical_jacobian_attributes(
        inputs, outputs[output_idx], nondet_tol=float("inf"), check_grad_dtypes=False
    )
    # 返回指定输入索引的雅可比矩阵
    return jacobians[input_idx]


def _compute_analytical_jacobian_rows(
    vjp_fn, sample_output
) -> List[List[Optional[torch.Tensor]]]:
    # 按标准基向量逐行计算雅可比矩阵：vjp_fn(e) = e^T J 是雅可比矩阵的对应行。
    # 注意：此函数假设vjp_fn(v)对于不同的v返回的张量元素数可能不同。
    # 这在后续将行组合成单个张量时进行检查。
    grad_out_base = torch.zeros_like(
        sample_output, memory_format=torch.legacy_contiguous_format
    )
    flat_grad_out = grad_out_base.view(-1)
    # jacobians_rows[i][j] 是第i个输入的第j行雅可比矩阵
    jacobians_rows: List[List[Optional[torch.Tensor]]] = []
    for j in range(flat_grad_out.numel()):
        flat_grad_out.zero_()
        flat_grad_out[j] = 1.0  # 对雅可比矩阵的第j行进行投影
        grad_inputs = vjp_fn(grad_out_base)
        for i, d_x in enumerate(grad_inputs):
            if j == 0:
                jacobians_rows.append([])
            # 将张量克隆到列表中，如果是张量的话；否则插入空值
            jacobians_rows[i] += [
                d_x.clone() if isinstance(d_x, torch.Tensor) else None
            ]
    return jacobians_rows


def _get_analytical_vjps_wrt_specific_output(
    vjp_fn, sample_output, v
) -> List[List[Optional[torch.Tensor]]]:
    # 返回特定输出的解析VJP（向量-雅可比积）的列表
    vjps: List[List[Optional[torch.Tensor]]] = []
    grad_inputs = vjp_fn(v.reshape(sample_output.shape))
    for vjp in grad_inputs:
        # 克隆张量到列表中，如果是张量的话；否则插入空值
        vjps.append([vjp.clone() if isinstance(vjp, torch.Tensor) else None])
    return vjps


def _check_inputs(tupled_inputs) -> bool:
    # 确保至少有一个输入需要保存梯度信息
    any_input_requiring_grad = False
    # 遍历元组化的输入列表，同时获取索引和输入
    for idx, inp in enumerate(tupled_inputs):
        # 检查输入是否类似于张量并且需要梯度
        if is_tensor_like(inp) and inp.requires_grad:
            # 如果输入不是双精度浮点数或复数类型，发出警告
            if not (inp.dtype == torch.float64 or inp.dtype == torch.complex128):
                warnings.warn(
                    f"Input #{idx} requires gradient and "
                    "is not a double precision floating point or complex. "
                    "This check will likely fail if all the inputs are "
                    "not of double precision floating point or complex. "
                )
            # 如果输入是稀疏张量，获取其值
            if inp.is_sparse:
                content = inp._values()
            # 如果输入是压缩的稀疏张量，获取其值
            elif _is_sparse_compressed_tensor(inp):
                content = inp.values()
            # 否则直接使用输入内容
            else:
                content = inp
            # TODO: 为了覆盖更多问题情况，替换 stride = 0 的检查为 "内存中的任何重叠"，
            # 一旦我们有一个适当的函数来检查它。
            # 检查内容的布局是否不是 torch._mkldnn（MKL-DNN加速器）
            if content.layout is not torch._mkldnn:  # type: ignore[attr-defined]
                # 如果有任何维度的步长为0或者大小小于等于1，抛出运行时错误
                if not all(
                    st > 0 or sz <= 1
                    for st, sz in zip(content.stride(), content.size())
                ):
                    raise RuntimeError(
                        f"The {idx}th input has a dimension with stride 0. gradcheck only "
                        "supports inputs that are non-overlapping to be able to "
                        "compute the numerical gradients correctly. You should call "
                        ".contiguous on the input before passing it to gradcheck."
                    )
            # 标记至少有一个需要梯度的输入
            any_input_requiring_grad = True

    # 如果没有任何一个输入需要梯度，抛出值错误
    if not any_input_requiring_grad:
        raise ValueError(
            "gradcheck expects at least one input tensor to require gradient, "
            "but none of the them have requires_grad=True."
        )
    # 返回真，表示成功通过梯度检查
    return True
def _check_outputs(outputs) -> None:
    # 检查是否有稀疏张量输出，对于是 torch.Tensor 类型的输出中的任何稀疏张量，调用 to_dense() 更容易
    if any(_is_sparse_any_tensor(t) for t in outputs if isinstance(t, torch.Tensor)):
        raise ValueError(
            "Sparse output is not supported at gradcheck yet. "
            "Please call to_dense(masked_grad=...) on the output of fn for gradcheck."
        )
    # 检查是否有 MKLDNN 布局的张量输出，MKLDNN 输出在 gradcheck 中不受支持
    if any(t.layout == torch._mkldnn for t in outputs if isinstance(t, torch.Tensor)):  # type: ignore[attr-defined]
        raise ValueError(
            "MKLDNN output is not supported at gradcheck yet. "
            "Please call to_dense(masked_grad=...) on the output of fn for gradcheck."
        )


def _check_no_differentiable_outputs(
    func, inputs, func_out, eps, *, is_forward_ad
) -> bool:
    # 当没有可微分的输出时，预期函数的数值梯度为零
    jacobians_all_inputs_outputs = _get_numerical_jacobian(
        func, inputs, func_out, eps=eps, is_forward_ad=is_forward_ad
    )
    for jacobians_all_outputs_and_fixed_input in jacobians_all_inputs_outputs:
        for jacobian in jacobians_all_outputs_and_fixed_input:
            if torch.ne(jacobian, 0).sum() > 0:
                raise GradcheckError(
                    "Numerical gradient for function expected to be zero"
                )
    return True


def _check_no_differentiable_outputs_fast(
    func, func_out, all_inputs, inputs_indices, all_u, eps, nondet_tol
):
    # 使用数值 JVP（对于特定输入）检查是否存在可微分输出
    for inp_idx, u in zip(inputs_indices, all_u):
        jvps = _get_numerical_jvp_wrt_specific_input(func, inp_idx, all_inputs, u, eps)
        for jvp in jvps:
            if jvp.numel() == 0:
                continue
            if (jvp - torch.zeros_like(jvp)).abs().max() > nondet_tol:
                raise GradcheckError(
                    "Numerical gradient for function expected to be zero"
                )
    return True


FAILED_BATCHED_GRAD_MSG = """
gradcheck or gradgradcheck failed while testing batched gradient computation.
This could have been invoked in a number of ways (via a test that calls
gradcheck/gradgradcheck directly or via an autogenerated test).

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck.
If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `check_batched_grad=False` as a keyword argument.
- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test
  to have `check_batched_grad=False` and/or `check_batched_gradgrad=False`.

If you're modifying an existing operator that supports batched grad computation,
or wish to make a new operator work with batched grad computation, please read
the following.

To compute batched grads (e.g., jacobians, hessians), we vmap over the backward
# 定义一个用于描述测试失败的批量梯度计算消息的函数，用于前向自动微分（AD）的情况
def _get_failed_batched_grad_test_msg(
    output_idx, input_idx, res, exp, is_forward_ad=False
):
    # 返回一条包含输出索引和输入索引的消息，根据是否是前向自动微分选择不同的消息模板
    return f"""
For output {output_idx} and input {input_idx}:

{FAILED_BATCHED_GRAD_MSG_FWD_AD if is_forward_ad else FAILED_BATCHED_GRAD_MSG}

Got:
{res}

Expected:
{exp}
""".strip()


def _test_batched_grad_forward_ad(func, inputs) -> bool:
    # 导入torch.autograd.forward_ad以避免早期导入问题（我们是否需要这样做？）
    fwAD = torch.autograd.forward_ad  
    # 断言输入是一个元组
    assert isinstance(inputs, tuple)
    # 遍历输入的索引和当前输入
    for input_idx, current_input in enumerate(inputs):
        # 如果当前输入不是张量或不需要梯度，则跳过
        if not (is_tensor_like(current_input) and current_input.requires_grad):
            continue

        # 定义一个计算雅可比向量积的函数
        def jvp(tangent: torch.Tensor):
            # 进入自动微分的双重级别上下文
            with fwAD.dual_level():
                # 创建一个双态对象，将当前输入与切向量包装起来
                dual = fwAD.make_dual(current_input.detach(), tangent)
                # 准备带有双态输入的元组
                inputs_with_dual = tuple(
                    dual
                    if idx == input_idx
                    else (inp.detach() if is_tensor_like(inp) else inp)
                    for idx, inp in enumerate(inputs)
                )
                # 对函数应用输入带双态的参数，得到带双态输出的元组
                dual_outputs = _as_tuple(func(*inputs_with_dual))
                ret = []
                # 遍历双态输出元组中的每个双态输出
                for dual_output in dual_outputs:
                    # 如果双态输出为 None，则跳过
                    if dual_output is None:
                        continue
                    # 解包双态输出，获取原始输出和切向输出
                    primal_out, tangent_out = fwAD.unpack_dual(dual_output)
                    # 如果切向输出不为 None，则将其加入结果列表中
                    if tangent_out is not None:
                        ret.append(tangent_out)
                    else:
                        # 否则创建一个形状与原始输出相同的零张量，并加入结果列表中
                        ret.append(
                            torch.zeros(
                                [], dtype=primal_out.dtype, device=primal_out.device
                            ).expand(primal_out.shape)
                        )
                return tuple(ret)

        # 如果当前输入不是浮点数或复数张量，则跳过
        if not _is_float_or_complex_tensor(current_input):
            continue

        # 生成两个与当前输入形状相同的随机切向向量
        tangents = [torch.randn_like(current_input) for _ in range(2)]
        # 计算期望的雅可比向量积结果
        expected = [jvp(t) for t in tangents]
        # 将计算结果按列堆叠成张量
        expected = [torch.stack(shards) for shards in zip(*expected)]

        try:
            # 对 jvp 函数应用 _vmap 函数，计算批量处理的结果
            result = _vmap(jvp)(torch.stack(tangents))
        except RuntimeError as ex:
            # 如果发生运行时错误，则重新抛出异常，提供更好的错误消息
            raise GradcheckError(
                f"While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG_FWD_AD}"
            ) from ex

        # 遍历结果和期望值，检查它们是否接近
        for input_idx, (res, exp) in enumerate(zip(result, expected)):
            if torch.allclose(res, exp):
                continue
            # 如果结果与期望值不接近，则抛出梯度检查错误
            raise GradcheckError(
                _get_failed_batched_grad_test_msg(
                    input_idx, input_idx, res, exp, is_forward_ad=True
                )
            )
    # 如果所有检查通过，则返回 True
    return True
def _test_batched_grad(input, output, output_idx) -> bool:
    # NB: _test_batched_grad compares two autograd.grad invocations with a single
    # vmap(autograd.grad) invocation. It's not exactly a "gradcheck" in the
    # sense that we're not comparing an analytical jacobian with a numeric one,
    # but it is morally similar (we could have computed a full analytic jac
    # via vmap, but that is potentially slow)

    # 将输入张量列表化，以便逐个处理
    diff_input_list = list(_iter_tensors(input, True))
    
    # 创建一个部分应用的 torch.autograd.grad 函数，其中输出为指定输出、输入为处理过的输入张量列表
    # 保持计算图以便多次反向传播，允许未使用的梯度
    grad = functools.partial(
        torch.autograd.grad,
        output,
        diff_input_list,
        retain_graph=True,
        allow_unused=True,
    )

    # 定义一个函数 vjp，计算输入的向量-雅可比积
    def vjp(v):
        results = grad(v)
        # 处理每个梯度结果，如果为 None 则填充为零张量
        results = tuple(
            grad
            if grad is not None
            else torch.zeros([], dtype=inp.dtype, device=inp.device).expand(inp.shape)
            for grad, inp in zip(results, diff_input_list)
        )
        return results

    # 生成与输出张量形状相同的随机梯度输出列表
    grad_outputs = [torch.randn_like(output) for _ in range(2)]

    # 期望的向量-雅可比积列表，通过 vjp 函数计算
    expected = [vjp(gO) for gO in grad_outputs]
    # 对期望结果进行堆叠，以匹配 vmap(vjp) 的输出
    expected = [torch.stack(shards) for shards in zip(*expected)]

    # 屏蔽警告，因为大多数情况下这些警告是预期的
    # 注意：对于 CUDA 测试，此方法不适用：https://github.com/pytorch/pytorch/issues/50209
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="There is a performance drop")
        warnings.filterwarnings("ignore", message="Please use torch.vmap")
        try:
            # 使用 vmap 应用 vjp 函数，处理随机梯度输出列表
            result = vmap(vjp)(torch.stack(grad_outputs))
        except RuntimeError as ex:
            # 在计算批处理梯度时，如果出现错误，则抛出自定义 GradcheckError
            raise GradcheckError(
                f"While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG}"
            ) from ex

    # 遍历结果和期望值，比较它们是否全部接近
    for input_idx, (res, exp) in enumerate(zip(result, expected)):
        if torch.allclose(res, exp):
            continue
        # 如果不接近，则抛出 GradcheckError 异常
        raise GradcheckError(
            _get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp)
        )
    # 若全部比较通过，则返回 True
    return True


def _test_backward_mul_by_grad_output(outputs, inputs, masked) -> bool:
    # Tests that backward is multiplied by grad_output
    # 将输入张量列表化，以便逐个处理
    diff_input_list: List[torch.Tensor] = list(_iter_tensors(inputs, True))
    
    # 如果没有需要梯度的张量，则抛出 GradcheckError 异常
    if not diff_input_list:
        raise GradcheckError("no Tensors requiring grad found in input")
    
    # 计算梯度，输出使用零张量，以保留传统的内存格式
    grads_input = torch.autograd.grad(
        outputs,
        diff_input_list,
        [
            torch.zeros_like(o, memory_format=torch.legacy_contiguous_format)
            for o in outputs
        ],
        allow_unused=True,
    )
    for gi, di in zip(grads_input, diff_input_list):
        # 遍历 grads_input 和 diff_input_list 中的每对元素
        if gi is None:
            # 如果 gi 为 None，则跳过当前循环，继续下一次循环
            continue
        if isinstance(gi, torch.Tensor) and gi.layout != torch.strided:
            # 如果 gi 是 torch.Tensor 类型且布局不是 torch.strided
            if gi.layout != di.layout:
                # 如果 gi 的布局与 di 的布局不同，抛出 GradcheckError 异常
                raise GradcheckError(
                    "grad is incorrect layout ("
                    + str(gi.layout)
                    + " is not "
                    + str(di.layout)
                    + ")"
                )
            if _is_sparse_any_tensor(gi):
                # 如果 gi 是稀疏张量
                sparse_kind = str(gi.layout).replace("torch.", "").replace("_coo", "")
                if gi.sparse_dim() != di.sparse_dim():
                    # 如果 gi 的稀疏维度与 di 的稀疏维度不同，抛出 GradcheckError 异常
                    raise GradcheckError(
                        f"grad is {sparse_kind} tensor, but has incorrect sparse_dim"
                        f" {gi.sparse_dim()}, expected {di.sparse_dim()}"
                    )
                if gi.dense_dim() != di.dense_dim():
                    # 如果 gi 的密集维度与 di 的密集维度不同，抛出 GradcheckError 异常
                    raise GradcheckError(
                        f"grad is {sparse_kind} tensor, but has incorrect dense_dim"
                        f" {gi.dense_dim()}, expected {di.dense_dim()}"
                    )
            # 将稀疏张量 gi 和 di 转换为密集张量
            gi = gi.to_dense()
            di = di.to_dense()
        if masked:
            # 如果处于 masked 状态
            if not torch.allclose(gi, torch.zeros_like(gi)):
                # 如果 gi 不与全零张量 torch.zeros_like(gi) 全部接近，抛出 GradcheckError 异常
                raise GradcheckError("backward not multiplied by grad_output")
        elif not gi.eq(0).all():
            # 如果不处于 masked 状态且 gi 不全为零
            raise GradcheckError("backward not multiplied by grad_output")
        if gi.dtype != di.dtype:
            # 如果 gi 的数据类型与 di 的数据类型不同，抛出 GradcheckError 异常
            raise GradcheckError("grad is incorrect type")
        if gi.device != di.device:
            # 如果 gi 的设备与 di 的设备不同，抛出 GradcheckError 异常
            raise GradcheckError("grad is incorrect device")
        if gi.size() != di.size():
            # 如果 gi 的尺寸与 di 的尺寸不同，抛出 GradcheckError 异常
            raise GradcheckError("grad is incorrect size")
    return True
def _test_undefined_forward_mode(func, outputs, inputs):
    # 导入 Torch 的 forward_ad 模块，用于自动求导中的前向传播
    fwAD = torch.autograd.forward_ad

    # 从输入中获取张量的索引和张量本身
    inp_tensors_idx, inp_tensors = _get_inp_tensors(inputs)
    # 创建向量以用于前向自动求导，包括所有必要的向量和稠密向量
    all_v, all_u, all_u_dense = _make_vectors(inp_tensors, outputs, use_forward_ad=True)

    # 选出所有需要梯度的张量输入
    tensor_inputs = tuple(i for i in inputs if is_tensor_like(i) and i.requires_grad)

    # 使用 forward_ad 模块的双重级别上下文
    with fwAD.dual_level():
        # 存储前向梯度的列表
        fw_grads = []
        # 存储双重输入的列表
        dual_inputs = []
        # 存储张量的索引集合
        tensor_indices = set()

        # 遍历输入，处理每个需要梯度的张量
        for i, inp in enumerate(inputs):
            if is_tensor_like(inp) and inp.requires_grad:
                # 检查是否为 MKLDNN 布局的输入，如果是则抛出错误
                if inp.layout == torch._mkldnn:  # type: ignore[attr-defined]
                    raise ValueError(
                        "MKLDNN inputs are not support for forward AD gradcheck."
                    )
                # 创建对应的双重张量对象，初始梯度设置为零
                inp = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
                # 如果输入是可微视图，则需要从双重张量中显式读取切线
                fw_grads.append(fwAD.unpack_dual(inp)[1])
                tensor_indices.add(i)
            # 添加到双重输入列表中
            dual_inputs.append(inp)

        # 将向量 u 的值复制到对应的前向梯度中
        for i, (fw_grad, u) in enumerate(zip(fw_grads, all_u)):
            fw_grad.copy_(u.view_as(fw_grad))

        # 遍历输入，处理每个需要梯度的张量
        for idx, inp in enumerate(inputs):
            if idx not in tensor_indices:
                continue
            # 获取原始的双重输入对象
            dual_inp_obj = dual_inputs[idx]

            # case 1 (Materialized Zero Tensor Tangent)
            # 使用 inp 的零切线创建双重输入对象
            dual_inputs[idx] = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
            # 调用函数 func，获取原始输出并过滤出浮点数或复数张量
            raw_outputs = _as_tuple(func(*dual_inputs))
            dual_outputs1 = filter(_is_float_or_complex_tensor, raw_outputs)

            # case 2 (Efficient Zero Tensor Tangent since we don't make a dual object and pass a regular tensor)
            # 直接使用 inp 的原始张量创建双重输入对象
            dual_inputs[idx] = inp.detach()
            # 再次调用函数 func，获取原始输出并过滤出浮点数或复数张量
            raw_outputs = _as_tuple(func(*dual_inputs))
            dual_outputs2 = filter(_is_float_or_complex_tensor, raw_outputs)

            # 恢复原始的双重输入对象
            dual_inputs[idx] = dual_inp_obj

            # 比较两种情况下的双重输出值
            for index_o, (d_o1, d_o2) in enumerate(zip(dual_outputs1, dual_outputs2)):
                # 解包双重张量，获取值和切线
                val1, res1 = fwAD.unpack_dual(d_o1)
                val2, res2 = fwAD.unpack_dual(d_o2)

                # 如果任一输出的切线不为空且二者不相等，则抛出异常
                if not (res1 is None or res2 is None):
                    if not torch.allclose(res1, res2):
                        raise GradcheckError(
                            "Mismatch in tangent values for output with index: ",
                            index_o,
                            " when input: ",
                            inp,
                            " has an undefined tangent value. ",
                            " Got: ",
                            res1,
                            " but expected: ",
                            res2,
                        )

    return True


def _test_undefined_backward_mode(func, outputs, inputs) -> bool:
    # 创建用于存储梯度的输入张量列表
    diff_input_list: List[torch.Tensor] = list(_iter_tensors(inputs, True))
    # 如果 diff_input_list 为空列表，则抛出 GradcheckError 异常，表示输入中没有需要梯度的张量
    if not diff_input_list:
        raise GradcheckError("no Tensors requiring grad found in input")

    # 警告函数，用于提示新的未定义梯度支持检查功能已启用，默认情况下可能会破坏现有调用该函数的代码
    def warn_bc_breaking():
        warnings.warn(
            "Backwards compatibility: New undefined gradient support checking "
            "feature is enabled by default, but it may break existing callers "
            "of this function. If this is true for you, you can call this "
            'function with "check_undefined_grad=False" to disable the feature'
        )

    # 检查对 undefined 梯度支持的函数
    def check_undefined_grad_support(output_to_check):
        # 为输出准备与之对应的全零梯度张量
        grads_output = [
            torch.zeros_like(o, memory_format=torch.legacy_contiguous_format)
            for o in output_to_check
        ]
        try:
            # 尝试计算输出相对于输入列表的梯度，并允许梯度输出中存在未使用的情况
            grads_input = torch.autograd.grad(
                output_to_check, diff_input_list, grads_output, allow_unused=True
            )
        except RuntimeError as e:
            # 捕获运行时异常，警告破坏性变更，并抛出 GradcheckError 异常
            warn_bc_breaking()
            raise GradcheckError(
                "Expected backward function to handle undefined output grads. "
                'Please look at "Notes about undefined output gradients" in '
                '"tools/autograd/derivatives.yaml"'
            ) from e

        # 检查每个输入张量的梯度，如果有不为零的情况则抛出 GradcheckError 异常
        for gi, i in zip(grads_input, diff_input_list):
            if (gi is not None) and (not gi.eq(0).all()):
                warn_bc_breaking()
                raise GradcheckError(
                    "Expected all input grads to be undefined or zero when all output grads are undefined "
                    'or zero. Please look at "Notes about undefined output gradients" in '
                    '"tools/autograd/derivatives.yaml"'
                )
        return True

    # 必须确保所有反向函数在所有输出梯度为未定义时能正常工作
    outputs_to_check = [
        [
            torch._C._functions.UndefinedGrad()(o)
            for o in _differentiable_outputs(func(*inputs))
            # 这个检查过滤掉不是 Tensor 实例的 Tensor-like 对象
            if isinstance(o, torch.Tensor)
        ]
    ]

    # 如果有多个输出梯度，则应能够单独取消定义其中一个而不出错
    if len(outputs_to_check[0]) > 1:
        for undef_grad_idx in range(len(outputs)):
            output_to_check = _differentiable_outputs(func(*inputs))
            outputs_to_check.append(
                [
                    torch._C._functions.UndefinedGrad()(o)
                    if idx == undef_grad_idx
                    else o
                    for idx, o in enumerate(output_to_check)
                ]
            )

    # 对所有待检查的输出执行 check_undefined_grad_support 函数，并返回检查结果的布尔值
    return all(check_undefined_grad_support(output) for output in outputs_to_check)
def _as_tuple(x):
    # 如果输入是元组则直接返回，否则将列表转换为元组返回
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def _differentiable_outputs(x):
    # 返回 x 中所有 requires_grad 为 True 的元素组成的元组
    return tuple(o for o in _as_tuple(x) if o.requires_grad)


def _get_notallclose_msg(
    analytical,
    numerical,
    output_idx,
    input_idx,
    complex_indices,
    test_imag=False,
    is_forward_ad=False,
) -> str:
    # 判断输出和输入是否为复数及测试模式
    out_is_complex = (
        (not is_forward_ad) and complex_indices and output_idx in complex_indices
    )
    inp_is_complex = is_forward_ad and complex_indices and input_idx in complex_indices
    part = "imaginary" if test_imag else "real"
    element = "inputs" if is_forward_ad else "outputs"
    # 构建消息前缀，描述 Jacobian 矩阵在数值和解析上的不匹配情况
    prefix = (
        ""
        if not (out_is_complex or inp_is_complex)
        else f"While considering the {part} part of complex {element} only, "
    )
    mode = "computed with forward mode " if is_forward_ad else ""
    # 返回详细的不匹配消息
    return (
        prefix + "Jacobian %smismatch for output %d with respect to input %d,\n"
        "numerical:%s\nanalytical:%s\n"
        % (mode, output_idx, input_idx, numerical, analytical)
    )


def _transpose(matrix_of_tensors):
    # 将输入矩阵转置为元组列表
    return list(zip(*matrix_of_tensors))


def _real_and_imag_output(fn):
    # 返回新函数 real(fn) 和 imag(fn)，这些函数处理复数输出并应用 torch.real 或 torch.imag
    def apply_to_c_outs(fn, fn_to_apply):
        def wrapped_fn(*inputs):
            outs = _as_tuple(fn(*inputs))
            return tuple(fn_to_apply(o) if o.is_complex() else o for o in outs)

        return wrapped_fn

    return apply_to_c_outs(fn, torch.real), apply_to_c_outs(fn, torch.imag)


def _real_and_imag_input(fn, complex_inp_indices, tupled_inputs):
    # 返回新函数，这些函数处理实部输入而非复数输入，并根据指定的复数输入索引计算输出
    def apply_to_c_inps(fn, fn_to_apply):
        def wrapped_fn(*inputs):
            new_inputs = list(inputs)
            for should_be_complex in complex_inp_indices:
                new_inputs[should_be_complex] = fn_to_apply(
                    new_inputs[should_be_complex], tupled_inputs[should_be_complex]
                )
            return _as_tuple(fn(*new_inputs))

        return wrapped_fn

    # 创建处理实部和虚部输入的函数
    real_fn = apply_to_c_inps(fn, lambda inp, orig: inp + orig.imag * 1j)
    imag_fn = apply_to_c_inps(fn, lambda inp, orig: orig.real + inp * 1j)
    return real_fn, imag_fn


def _gradcheck_real_imag(
    gradcheck_fn,
    func,
    func_out,
    tupled_inputs,
    outputs,
    eps,
    rtol,
    atol,
    check_grad_dtypes,
    check_forward_ad,
    check_backward_ad,
    nondet_tol,
):
    # 在实部和虚部上进行梯度检查，返回详细的检查结果
    check_undefined_grad,


注释：


    # 调用一个函数或者变量，检查梯度是否未定义
    check_undefined_grad,


这行代码调用了一个函数或者变量 `check_undefined_grad`，其作用是检查梯度是否未定义。注释说明了这个操作的目的和可能的功能。
    # 查找所有复数输出的索引列表
    complex_out_indices = [i for i, o in enumerate(outputs) if o.is_complex()]
    # 检查是否存在任何复数输出
    has_any_complex_output = any(o.is_complex() for o in _as_tuple(func_out))
    # 如果需要检查反向传播的梯度
    if check_backward_ad:
        # 如果有任何复数输出
        if has_any_complex_output:
            # 获取实部和虚部函数
            real_fn, imag_fn = _real_and_imag_output(func)

            # 计算虚部函数在给定输入上的输出
            imag_func_out = imag_fn(*tupled_inputs)
            # 获取虚部输出的可微分部分
            imag_outputs = _differentiable_outputs(imag_func_out)
            # 进行梯度检查
            gradcheck_fn(
                imag_fn,
                imag_func_out,
                tupled_inputs,
                imag_outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_out_indices,
                test_imag=True,
            )

            # 计算实部函数在给定输入上的输出
            real_func_out = real_fn(*tupled_inputs)
            # 获取实部输出的可微分部分
            real_outputs = _differentiable_outputs(real_func_out)
            # 进行梯度检查
            gradcheck_fn(
                real_fn,
                real_func_out,
                tupled_inputs,
                real_outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_out_indices,
            )
        else:
            # 如果没有复数输出，直接进行梯度检查
            gradcheck_fn(
                func,
                func_out,
                tupled_inputs,
                outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
            )
    # 如果需要进行前向自动微分检查
    if check_forward_ad:
        # 找出复数输入的索引
        complex_inp_indices = [
            i
            for i, inp in enumerate(tupled_inputs)
            if is_tensor_like(inp) and inp.is_complex()
        ]
        
        # 如果存在复数输入
        if complex_inp_indices:
            # 获取处理实部和虚部的函数
            real_fn, imag_fn = _real_and_imag_input(
                func, complex_inp_indices, tupled_inputs
            )
    
            # 提取虚部输入
            imag_inputs = [
                inp.imag if is_tensor_like(inp) and inp.is_complex() else inp
                for inp in tupled_inputs
            ]
            
            # 计算虚部函数的输出
            imag_func_out = imag_fn(*imag_inputs)
            
            # 计算虚部函数输出的可微分部分
            diff_imag_func_out = _differentiable_outputs(imag_func_out)
            
            # 执行梯度检查
            gradcheck_fn(
                imag_fn,
                imag_func_out,
                imag_inputs,
                diff_imag_func_out,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_inp_indices,
                test_imag=True,
                use_forward_ad=True,
            )
    
            # 提取实部输入
            real_inputs = [
                inp.real if is_tensor_like(inp) and inp.is_complex() else inp
                for inp in tupled_inputs
            ]
            
            # 计算实部函数的输出
            real_func_out = real_fn(*real_inputs)
            
            # 计算实部函数输出的可微分部分
            diff_real_func_out = _differentiable_outputs(real_func_out)
            
            # 执行梯度检查
            gradcheck_fn(
                real_fn,
                real_func_out,
                real_inputs,
                diff_real_func_out,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_inp_indices,
                use_forward_ad=True,
            )
            
            # 如果需要检查未定义的梯度
            if check_undefined_grad:
                # 测试虚部函数的未定义梯度模式
                _test_undefined_forward_mode(imag_fn, imag_func_out, imag_inputs)
                # 测试实部函数的未定义梯度模式
                _test_undefined_forward_mode(real_fn, real_func_out, real_inputs)
        
        # 如果没有复数输入
        else:
            # 执行梯度检查
            gradcheck_fn(
                func,
                func_out,
                tupled_inputs,
                outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                use_forward_ad=True,
            )
            
            # 如果需要检查未定义的梯度
            if check_undefined_grad:
                # 测试函数的未定义梯度模式
                _test_undefined_forward_mode(func, outputs, tupled_inputs)
# 定义一个函数 _slow_gradcheck，用于执行梯度检查，通常用于验证反向传播的正确性
def _slow_gradcheck(
    func,                   # 待检查的函数
    func_out,               # 函数的输出结果
    tupled_inputs,          # 函数的输入参数，以元组形式传入
    outputs,                # 期望的输出结果
    eps,                    # 用于数值微分的小偏移量
    rtol,                   # 相对误差容差
    atol,                   # 绝对误差容差
    check_grad_dtypes,      # 是否检查梯度的数据类型
    nondet_tol,             # 非确定性容差
    *,                      # 以下参数为关键字参数
    use_forward_ad=False,   # 是否使用前向自动微分
    complex_indices=None,   # 复数索引（可选）
    test_imag=False,        # 是否测试虚部（可选）
    masked=False,           # 是否使用遮罩（可选）
):
    func_out = _as_tuple(func_out)  # 将函数输出转换为元组形式
    if not outputs:
        return _check_no_differentiable_outputs(
            func, tupled_inputs, func_out, eps=eps, is_forward_ad=use_forward_ad
        )

    # 如果没有期望的输出，执行检查不可微输出的函数
    tupled_inputs_numerical = tupled_inputs if masked else _densify(tupled_inputs)

    # 转置数值雅可比矩阵，用于数值微分
    numerical = _transpose(
        _get_numerical_jacobian(
            func,
            tupled_inputs_numerical,
            func_out,
            eps=eps,
            is_forward_ad=use_forward_ad,
        )
    )

    # 注释: [numerical vs analytical output length]
    # 数值路径返回所有输出的雅可比矩阵数量，即使输出的 requires_grad 为 False。
    # 这种行为对于 _check_no_differentiable_outputs 函数的正常工作是必要的。

    # 仅保留需要计算梯度的数值雅可比矩阵
    numerical = [nj for o, nj in zip(func_out, numerical) if o.requires_grad]

    if use_forward_ad:
        # 使用前向自动微分方法获取解析雅可比矩阵
        analytical_forward = _get_analytical_jacobian_forward_ad(
            func, tupled_inputs, func_out, check_grad_dtypes=check_grad_dtypes
        )

        # 逐元素比较数值雅可比矩阵和解析雅可比矩阵
        for i, n_per_out in enumerate(numerical):
            for j, n in enumerate(n_per_out):
                a = analytical_forward[j][i]
                # 如果不满足类型提升后的近似相等性，则抛出梯度检查错误
                if not _allclose_with_type_promotion(a, n.to(a.device), rtol, atol):
                    raise GradcheckError(
                        _get_notallclose_msg(
                            a, n, i, j, complex_indices, test_imag, is_forward_ad=True
                        )
                    )
    else:
        # 使用传统的后向自动微分方法进行梯度检查
        for i, o in enumerate(outputs):
            analytical = _check_analytical_jacobian_attributes(
                tupled_inputs, o, nondet_tol, check_grad_dtypes
            )

            # 逐元素比较解析雅可比矩阵和数值雅可比矩阵
            for j, (a, n) in enumerate(zip(analytical, numerical[i])):
                # 如果不满足类型提升后的近似相等性，则抛出梯度检查错误
                if not _allclose_with_type_promotion(a, n.to(a.device), rtol, atol):
                    raise GradcheckError(
                        _get_notallclose_msg(a, n, i, j, complex_indices, test_imag)
                    )

    # 如果所有梯度检查通过，则返回 True
    return True


# 执行类型提升后的向量点积运算
def _dot_with_type_promotion(u, v):
    assert u.dim() == 1 and v.dim() == 1
    return (u * v).sum()


# 使用类型提升后的方法检查两个张量是否在指定的相对和绝对误差容差内近似相等
def _allclose_with_type_promotion(a, b, rtol, atol):
    promoted_type = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=promoted_type)
    b = b.to(dtype=promoted_type)
    return torch.allclose(a, b, rtol, atol)


# 将复数数据类型转换为对应的实数数据类型
def _to_real_dtype(dtype):
    if dtype == torch.complex128:
        return torch.float64
    elif dtype == torch.complex64:
        return torch.float32
    else:
        return dtype


# 从给定张量创建一个具有相同元素数量、相同数据类型/设备的随机向量
def _vec_from_tensor(x, generator, downcast_complex=False):
    # 如果 x 是复数且不降级复数，则创建具有实部的复数张量
    # 否则，创建一个具有相同数据类型和设备的随机向量
    # 检查输入张量 x 的布局是否为稀疏 COO 格式
    if x.layout == torch.sparse_coo:
        # 对于稀疏张量，创建一个具有相同索引的随机稀疏向量。确保设置 size，避免推断为更小的尺寸。
        x_values = x._values()
        # 如果 downcast_complex 为 True，则将复数类型转换为实数类型的 dtype
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        # 生成随机数值，大小与 x_values 相同，置于指定的 dtype 和设备上
        values = (
            torch.rand(x_values.numel(), generator=generator)
            .to(dtype=dtype, device=x.device)
            .view(x_values.shape)
        )
        # 归一化稀疏向量的值
        values /= values.norm()
        # 创建稀疏 COO 张量，使用输入张量 x 的索引和生成的值，设备保持一致
        vec = torch.sparse_coo_tensor(x._indices(), values, x.size(), device=x.device)
    # 如果输入张量 x 是压缩稀疏张量
    elif _is_sparse_compressed_tensor(x):
        # 根据输入张量 x 的布局类型确定压缩索引和普通索引
        if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
            compressed_indices, plain_indices = x.crow_indices(), x.col_indices()
        else:
            compressed_indices, plain_indices = x.ccol_indices(), x.row_indices()
        # 获取输入张量 x 的值
        x_values = x.values()
        # 如果 downcast_complex 为 True，则将复数类型转换为实数类型的 dtype
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        # 生成随机数值，大小与 x_values 相同，置于指定的 dtype 和设备上
        values = (
            torch.rand(x_values.numel(), generator=generator)
            .to(dtype=dtype, device=x.device)
            .view(x_values.shape)
        )
        # 归一化稀疏向量的值
        values /= values.norm()
        # 创建压缩稀疏张量，使用压缩索引、普通索引和生成的值，设备保持一致
        vec = torch.sparse_compressed_tensor(
            compressed_indices,
            plain_indices,
            values,
            x.size(),
            layout=x.layout,
            device=x.device,
        )
    else:
        # 如果 x 不是稀疏张量，则生成一个随机张量
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        # 生成随机数值，大小与 x 的元素数相同，置于指定的 dtype 和设备上
        vec = torch.rand(x.numel(), generator=generator).to(
            dtype=dtype, device=x.device
        )
        # 归一化向量的值
        vec /= vec.norm()
    # 返回生成的向量
    return vec
# 从给定的元组输入中获取需要梯度计算的张量及其索引
def _get_inp_tensors(tupled_inputs):
    # 列表推导式，筛选出张量类对象并且需要梯度计算的元素
    inp_idx_tup = [
        (i, t)
        for i, t in enumerate(tupled_inputs)
        if is_tensor_like(t) and t.requires_grad
    ]
    # 返回分别包含索引和张量的列表
    return [tup[0] for tup in inp_idx_tup], [tup[1] for tup in inp_idx_tup]


# 计算调整后的绝对误差容差值
def _adjusted_atol(atol, u, v):
    # 对于慢速 gradcheck，我们逐元素比较 A 和 B，即对于某些 a, b 我们允许：
    # |a - b| < atol + rtol * b。但现在我们比较 q1 = v^T A u 和 q2 = v^T B u，因此我们必须允许
    # |q1 - q2| < v^T E u + rtol * v^T B u，其中 E 是具有每个条目为 atol 的正确大小的矩阵。
    #
    # 我们看到 atol 需要按 v^T M u 缩放（其中 M 是全为1的 M x N 矩阵）：
    # v^T M u = \sum_{i} \sum_{j} u_i * v_j = (\sum_{i} u_i)(\sum_{i} v_i)
    # TODO: 处理 u 是元组的情况，而不仅仅是取第一个元素
    u = u[0] if isinstance(u, tuple) else u
    # 计算 u 中所有元素的和
    sum_u = u.sum()
    # 如果 v 为 None，则设置 sum_v 为 1.0，否则计算 v 中所有元素的和
    sum_v = 1.0 if v is None else v.sum()
    # 返回调整后的绝对误差容差值
    return atol * float(sum_u) * float(sum_v)


# 快速失败但慢速通过的错误消息
FAST_FAIL_SLOW_OK_MSG = """
Fast gradcheck failed but element-wise differences are small. This means that the
test might've passed in slow_mode!

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck:

If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `fast_mode=False` as a keyword argument.
- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test
  to have `gradcheck_fast_mode=False`
- is a Module test (e.g., in common_nn.py), then modify the corresponding
  module_test entry to have `gradcheck_fast_mode=False`
""".strip()


# 在慢速模式下运行并获取误差
def _run_slow_mode_and_get_error(
    func, tupled_inputs, outputs, input_idx, output_idx, rtol, atol, eps, is_forward_ad
):
    # 在慢速模式下计算雅可比矩阵以获得更好的错误消息
    slow_numerical = _get_numerical_jacobian(
        func, tupled_inputs, outputs, eps=eps, is_forward_ad=is_forward_ad
    )[input_idx][output_idx]
    
    if is_forward_ad:
        # 定义新的函数，用于前向自动微分计算
        def new_fn(inp):
            new_inputs = list(tupled_inputs)
            new_inputs[input_idx] = inp
            return _as_tuple(func(*new_inputs))[output_idx]
        
        # 计算前向自动微分的解析雅可比矩阵
        slow_analytical = _get_analytical_jacobian_forward_ad(
            new_fn, (tupled_inputs[input_idx],), (outputs[output_idx],)
        )[0][0]
    else:
        # 计算解析雅可比矩阵
        slow_analytical = _get_analytical_jacobian(
            tupled_inputs, outputs, input_idx, output_idx
        )
    
    # 假设雅可比矩阵非空且具有相同的形状
    # 计算数值和解析雅可比矩阵的最大差值
    slow_max_diff = (slow_numerical - slow_analytical).abs().max()
    
    # 检查数值和解析雅可比矩阵是否在容差范围内全部相等
    slow_allclose = torch.allclose(slow_analytical, slow_numerical, rtol, atol)
    # 构造包含详细信息的字符串消息
    msg = (
        "\nThe above quantities relating the numerical and analytical jacobians are computed \n"
        "in fast mode. See: https://github.com/pytorch/pytorch/issues/53876 for more background \n"
        "about fast mode. Below, we recompute numerical and analytical jacobians in slow mode:\n\n"
        f"Numerical:\n {slow_numerical}\n"
        f"Analytical:\n{slow_analytical}\n\n"
        f"The max per-element difference (slow mode) is: {slow_max_diff}.\n"
    )
    # 如果慢速模式下的数值梯度检查通过
    if slow_allclose:
        # 将额外消息添加到字符串消息中
        msg += FAST_FAIL_SLOW_OK_MSG
    # 返回构造的消息字符串
    return msg
# 如果输入张量是稀疏的，则将其转换为密集张量，并展平成一维数组返回
def _to_flat_dense_if_sparse(tensor):
    if _is_sparse_any_tensor(tensor):  # 检查输入张量是否为稀疏张量
        return tensor.to_dense().reshape(-1)  # 将稀疏张量转换为密集张量并展平成一维数组
    else:
        return tensor  # 如果输入张量不是稀疏的，直接返回原张量


# 为输入张量生成向量表示，并根据需要在 CPU 上执行
def _make_vectors(inp_tensors, outputs, *, use_forward_ad):
    # 使用自己的生成器以避免影响用户的随机数发生器状态
    g_cpu = torch.Generator()

    def _vec_from_tensor_cpu(*args):
        # 默认将所有张量分配到 CPU 上，以便它们与生成器位于相同设备上，即使用户指定了默认设备
        with torch.device("cpu"):
            return _vec_from_tensor(*args)

    all_u = []  # 存储所有的向量 u
    all_u_dense = []  # 存储所有稀疏张量转换为密集后的向量 u
    for inp in inp_tensors:
        ur = _vec_from_tensor_cpu(inp, g_cpu, True)  # 生成实部向量 ur
        ur_dense = _to_flat_dense_if_sparse(ur)  # 将实部向量转换为密集形式
        if inp.is_complex():  # 如果输入张量是复数类型
            ui = _vec_from_tensor_cpu(inp, g_cpu, True)  # 生成虚部向量 ui
            all_u.append((ur, ui))  # 将实部和虚部向量以元组形式存储
            ui_dense = _to_flat_dense_if_sparse(ui)  # 将虚部向量转换为密集形式
            all_u_dense.append((ur_dense, ui_dense))  # 将密集形式的实部和虚部向量以元组形式存储
        else:
            all_u.append(ur)  # 如果输入张量不是复数类型，只存储实部向量
            all_u_dense.append(ur_dense)  # 只存储密集形式的实部向量
    all_v = (
        None
        if use_forward_ad
        else [_vec_from_tensor_cpu(out, g_cpu) for out in outputs]
    )  # 如果使用 forward-mode 自动微分，all_v 为 None；否则，生成所有输出张量的向量表示
    return all_v, all_u, all_u_dense  # 返回所有输出向量 v，所有输入向量 u，以及转换为密集的所有输入向量 u


# 检查解析梯度与数值梯度是否相等
def _check_analytical_numerical_equal(
    all_analytical,
    all_numerical,
    complex_indices,
    tupled_inputs,
    outputs,
    func,
    all_v,
    all_u,
    rtol,
    atol,
    eps,
    test_imag,
    *,
    is_forward_ad=False,
):
    for i, all_numerical_for_input_i in enumerate(all_numerical):
        for j, n in enumerate(all_numerical_for_input_i):
            # 对于 forward-mode 自动微分，该函数预期的张量顺序与生成的转置顺序相反
            if is_forward_ad:
                a = all_analytical[i][j]  # 解析梯度矩阵 a
            else:
                a = all_analytical[j][i]  # 解析梯度矩阵 a
            n = n.to(device=a.device)  # 将数值梯度张量 n 移动到解析梯度张量 a 的设备上
            updated_atol = _adjusted_atol(atol, all_u[i], all_v[j] if all_v else None)  # 调整公差
            if not _allclose_with_type_promotion(a, n.to(a.device), rtol, updated_atol):
                # 如果解析梯度与数值梯度不近似相等，运行慢速模式并获取错误信息
                jacobians_str = _run_slow_mode_and_get_error(
                    func, tupled_inputs, outputs, i, j, rtol, atol, eps, is_forward_ad
                )
                # 抛出梯度检查错误
                raise GradcheckError(
                    _get_notallclose_msg(
                        a, n, j, i, complex_indices, test_imag, is_forward_ad
                    )
                    + jacobians_str
                )


# 快速梯度检查
def _fast_gradcheck(
    func,
    func_out,
    inputs,
    outputs,
    eps,
    rtol,
    atol,
    check_grad_dtypes,
    nondet_tol,
    *,
    use_forward_ad=False,
    complex_indices=None,
    test_imag=False,
    masked=False,
):
    # 参考链接详细说明了此处的问题
    inp_tensors_idx, inp_tensors = _get_inp_tensors(inputs)  # 获取输入张量的索引和张量列表
    # 后向模式计算 v^T * J (VJP)
    # 因为我们通过有限差分方法计算了 J * u (JVP)，所以我们在 VJP * u 和 v * JVP 之间进行了等式检查
    # ----
    # 计算前向模式下的 J * u（Jacobian向量积）
    # 由于我们已经通过有限差分方法计算了 J * v，因此这里不需要 v 用于正确性检查，如下所断言的那样
    all_v, all_u, all_u_dense = _make_vectors(
        inp_tensors, outputs, use_forward_ad=use_forward_ad
    )

    # 如果没有遮罩，将输入、所有 u 和所有 v 密集化处理；否则保持原始输入、所有 u 和所有 v 不变
    inputs_numerical, all_u_numerical, all_v_numerical = (
        (inputs, all_u, all_v) if masked else _densify((inputs, all_u, all_v))
    )

    # 获取数值计算得到的 v * J * u
    numerical_vJu = _get_numerical_vJu(
        func,
        inputs_numerical,
        inp_tensors_idx,
        func_out,
        all_u_numerical,
        all_v_numerical,
        eps,
        is_forward_ad=use_forward_ad,
    )

    # 如果使用前向自动求导：
    if use_forward_ad:
        # 断言所有 v 应为 None
        assert all_v is None
        # 获取通过前向自动求导得到的解析雅可比矩阵
        analytical_vJu = _get_analytical_jacobian_forward_ad(
            func,
            inputs,
            _as_tuple(func_out),
            all_u=all_u,
            check_grad_dtypes=check_grad_dtypes,
        )
    else:
        # 如果没有指定输出，快速检查函数中是否有不可微输出
        _check_no_differentiable_outputs_fast(
            func, func_out, inputs, inp_tensors_idx, all_u, eps, nondet_tol
        )

        # 获取通过后向自动求导得到的解析 v * J * u
        analytical_vJu = _get_analytical_vJu_backward_mode(
            inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u_dense
        )

    # 检查解析和数值计算结果是否一致
    _check_analytical_numerical_equal(
        analytical_vJu,
        numerical_vJu,
        complex_indices,
        inputs,
        outputs,
        func,
        all_v,
        all_u,
        rtol,
        atol,
        eps,
        test_imag,
        is_forward_ad=use_forward_ad,
    )

    # 返回 True，表示通过了梯度检查
    return True
# Note [VarArg of Tensors]
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 'func' accepts a vararg of tensors, which isn't expressible in the type system at the moment.
# If https://mypy.readthedocs.io/en/latest/additional_features.html?highlight=callable#extended-callable-types is accepted,
# the '...' first argument of Callable can be replaced with VarArg(Tensor).
# For now, we permit any input.
def gradcheck(
    func: Callable[..., Union[_TensorOrTensors]],  # See Note [VarArg of Tensors]
    inputs: _TensorOrTensors,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_batched_grad: bool = False,
    check_batched_forward_grad: bool = False,
    check_forward_ad: bool = False,
    check_backward_ad: bool = True,
    fast_mode: bool = False,
    masked: Optional[bool] = None,
) -> bool:  # noqa: D400,D205
    r"""Check gradients computed via small finite differences against analytical
    gradients wrt tensors in :attr:`inputs` that are of floating point or complex type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    For most of the complex functions we consider for optimization purposes, no notion of
    Jacobian exists. Instead, gradcheck verifies if the numerical and analytical values of
    the Wirtinger and Conjugate Wirtinger derivatives are consistent. Because the gradient
    computation is done under the assumption that the overall function has a real-valued
    output, we treat functions with complex output in a special way. For these functions,
    gradcheck is applied to two real-valued functions corresponding to taking the real
    components of the complex outputs for the first, and taking the imaginary components
    of the complex outputs for the second. For more details, check out
    :ref:`complex_autograd-doc`.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. note::
        Gradcheck may fail when evaluated on non-differentiable points
        because the numerically computed gradients via finite differencing may differ
        those computed analytically (not necessarily because either is incorrect).
        For more context, see :ref:`non-differentiable-func-grad`.

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`torch.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.
    Args:
        func (function): 一个接受张量输入并返回张量或张量元组的Python函数
        inputs (tuple of Tensor or Tensor): 函数的输入
        eps (float, optional): 有限差分的扰动大小
        atol (float, optional): 绝对误差容限
        rtol (float, optional): 相对误差容限
        raise_exception (bool, optional): 指示是否在检查失败时抛出异常。异常提供有关失败性质的详细信息，用于调试梯度检查。
        nondet_tol (float, optional): 非确定性的容限。在通过微分器运行相同输入时，结果必须精确匹配（默认为0.0），或在此容限内。
        check_undefined_grad (bool, optional): 如果为True，则检查未定义输出梯度是否被视为零，适用于Tensor输出。
        check_batched_grad (bool, optional): 如果为True，则检查是否可以使用原型vmap支持计算批次梯度。默认为False。
        check_batched_forward_grad (bool, optional): 如果为True，则检查是否可以使用前向AD和原型vmap支持计算批次前向梯度。默认为False。
        check_forward_ad (bool, optional): 如果为True，则检查使用前向模式AD计算的梯度是否与数值梯度匹配。默认为False。
        check_backward_ad (bool, optional): 如果为False，则不执行依赖于后向模式AD的任何检查。默认为True。
        fast_mode (bool, optional): 仅为从R到R函数实现的快速模式gradcheck和gradgradcheck。如果输入和输出都不是复数，则运行更快的gradcheck实现，不再计算整个Jacobian；否则，我们回退到慢速实现。
        masked (bool, optional): 如果为True，则忽略稀疏张量未指定元素的梯度。默认为False。
    Returns:
        ``True``：如果所有差异满足allclose条件
    """
    assert (
        check_forward_ad or check_backward_ad
    ), "Expected at least one of check_forward_ad or check_backward_ad to be True"
    assert not (
        check_batched_grad and not check_backward_ad
    ), "Setting check_batched_grad=True requires check_backward_ad to be True"
    assert not (
        check_batched_forward_grad and not check_forward_ad
    ), "Setting check_batched_forward_grad=True requires check_forward_ad to be True"
    args = locals().copy()  # 复制当前作用域内的局部变量
    args.pop("raise_exception")  # 移除键为"raise_exception"的元素
    if not raise_exception:
        try:
            return _gradcheck_helper(**args)  # 调用_gradcheck_helper函数，传递args中的参数
        except GradcheckError as e:  # 捕获GradcheckError异常
            return False  # 返回False，表示检查失败
    else:
        # 如果不是第一种情况，调用 _gradcheck_helper 函数并传入 args 字典中的所有关键字参数
        return _gradcheck_helper(**args)
# 定义一个辅助函数，用于检查梯度的正确性
def _gradcheck_helper(
    func,  # 待检查梯度的函数
    inputs,  # 输入给函数的参数
    eps,  # 用于数值微分的小增量
    atol,  # 绝对误差容限
    rtol,  # 相对误差容限
    nondet_tol,  # 非确定性梯度的容限
    check_undefined_grad,  # 是否检查未定义梯度
    check_grad_dtypes,  # 是否检查梯度的数据类型
    check_batched_grad,  # 是否检查批处理梯度
    check_batched_forward_grad,  # 是否检查批处理前向梯度
    check_forward_ad,  # 是否检查前向自动微分
    check_backward_ad,  # 是否检查反向自动微分
    fast_mode,  # 是否启用快速模式
    masked,  # 是否使用掩码
):
    # 将输入转换为元组形式
    tupled_inputs = _as_tuple(inputs)
    # 检查输入的有效性
    _check_inputs(tupled_inputs)

    # 调用函数计算输出
    func_out = func(*tupled_inputs)
    # 确定可微分的输出
    outputs = _differentiable_outputs(func_out)
    # 检查输出的有效性
    _check_outputs(outputs)

    # 根据快速模式选择梯度检查函数
    gradcheck_fn = functools.partial(
        _fast_gradcheck if fast_mode else _slow_gradcheck, masked=masked
    )
    # 执行实部和虚部的梯度检查
    _gradcheck_real_imag(
        gradcheck_fn,
        func,
        func_out,
        tupled_inputs,
        outputs,
        eps,
        rtol,
        atol,
        check_grad_dtypes,
        check_forward_ad=check_forward_ad,
        check_backward_ad=check_backward_ad,
        nondet_tol=nondet_tol,
        check_undefined_grad=check_undefined_grad,
    )

    # 如果需要检查批处理前向梯度
    if check_batched_forward_grad:
        # 测试批处理前向自动微分梯度
        _test_batched_grad_forward_ad(func, tupled_inputs)

    # 如果不需要检查反向自动微分，则直接返回True
    if not check_backward_ad:
        return True

    # 遍历所有输出，进行梯度测试
    for i, o in enumerate(outputs):
        # 如果需要检查批处理梯度
        if check_batched_grad:
            # 测试批处理梯度
            _test_batched_grad(tupled_inputs, o, i)

    # 测试反向乘以梯度输出
    _test_backward_mul_by_grad_output(outputs, tupled_inputs, masked)

    # 如果需要检查未定义梯度且需要反向自动微分
    if check_undefined_grad and check_backward_ad:
        # 测试未定义梯度模式
        _test_undefined_backward_mode(func, outputs, tupled_inputs)
    # 返回True表示梯度检查全部通过
    return True


# 检查梯度的梯度检查函数
def gradgradcheck(
    func: Callable[..., _TensorOrTensors],  # 待检查梯度的函数，接受可变数量的张量
    inputs: _TensorOrTensors,  # 传递给函数的张量或张量的元组
    grad_outputs: Optional[_TensorOrTensors] = None,  # 梯度输出张量或张量的元组，默认为None
    *,
    eps: float = 1e-6,  # 数值微分的小增量，默认为1e-6
    atol: float = 1e-5,  # 绝对误差容限，默认为1e-5
    rtol: float = 1e-3,  # 相对误差容限，默认为1e-3
    gen_non_contig_grad_outputs: bool = False,  # 是否生成非连续梯度输出，默认为False
    raise_exception: bool = True,  # 是否抛出异常，默认为True
    nondet_tol: float = 0.0,  # 非确定性梯度的容限，默认为0.0
    check_undefined_grad: bool = True,  # 是否检查未定义梯度，默认为True
    check_grad_dtypes: bool = False,  # 是否检查梯度的数据类型，默认为False
    check_batched_grad: bool = False,  # 是否检查批处理梯度，默认为False
    check_fwd_over_rev: bool = False,  # 是否检查前向对反向梯度，默认为False
    check_rev_over_rev: bool = True,  # 是否检查反向对反向梯度，默认为True
    fast_mode: bool = False,  # 是否启用快速模式，默认为False
    masked: bool = False,  # 是否使用掩码，默认为False
) -> bool:  # 函数返回布尔值，指示梯度检查结果
    r"""Check gradients of gradients computed via small finite differences
    against analytical gradients wrt tensors in :attr:`inputs` and
    :attr:`grad_outputs` that are of floating point or complex type and with
    ``requires_grad=True``.

    This function checks that backpropagating through the gradients computed
    to the given :attr:`grad_outputs` are correct.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` and
        :attr:`grad_outputs` of double precision. This check will likely fail if
        they are of less precision, e.g., ``FloatTensor``.
    """
        .. warning::
           If any checked tensor in :attr:`input` and :attr:`grad_outputs` has
           overlapping memory, i.e., different indices pointing to the same memory
           address (e.g., from :func:`torch.expand`), this check will likely fail
           because the numerical gradients computed by point perturbation at such
           indices will change values at all other indices that share the same
           memory address.
    
        Args:
            func (function): a Python function that takes Tensor inputs and returns
                a Tensor or a tuple of Tensors
            inputs (tuple of Tensor or Tensor): inputs to the function
            grad_outputs (tuple of Tensor or Tensor, optional): The gradients with
                respect to the function's outputs.
            eps (float, optional): perturbation for finite differences
            atol (float, optional): absolute tolerance
            rtol (float, optional): relative tolerance
            gen_non_contig_grad_outputs (bool, optional): if :attr:`grad_outputs` is
                ``None`` and :attr:`gen_non_contig_grad_outputs` is ``True``, the
                randomly generated gradient outputs are made to be noncontiguous
            raise_exception (bool, optional): indicating whether to raise an exception if
                the check fails. The exception gives more information about the
                exact nature of the failure. This is helpful when debugging gradchecks.
            nondet_tol (float, optional): tolerance for non-determinism. When running
                identical inputs through the differentiation, the results must either match
                exactly (default, 0.0) or be within this tolerance. Note that a small amount
                of nondeterminism in the gradient will lead to larger inaccuracies in
                the second derivative.
            check_undefined_grad (bool, optional): if True, check if undefined output grads
                are supported and treated as zeros
            check_batched_grad (bool, optional): if True, check if we can compute
                batched gradients using prototype vmap support. Defaults to False.
            fast_mode (bool, optional): if True, run a faster implementation of gradgradcheck that
                no longer computes the entire jacobian.
            masked (bool, optional): if True, the gradients of unspecified elements of
                sparse tensors are ignored (default, False).
        Returns:
            True if all differences satisfy allclose condition
        """
        # Ensure at least one of the forward-over-reverse or reverse-over-reverse checks is enabled
        assert (
            check_fwd_over_rev or check_rev_over_rev
        ), "Expected at least one of check_fwd_over_rev or check_rev_over_rev to be True"
        
        # Ensure that if checking for undefined gradients, reverse-over-reverse check is enabled
        assert not (
            check_undefined_grad and not check_rev_over_rev
        ), "Setting check_undefined_grad=True requires check_rev_over_rev to be True"
        
        # Ensure that if checking for batched gradients, reverse-over-reverse check is enabled
        assert not (
            check_batched_grad and not check_rev_over_rev
        ), "Setting check_batched_grad=True requires check_rev_over_rev to be True"
        
        # TODO: do we want to test this too?
    # assert not (check_batched_forward_grad and not check_fwd_over_rev), (
    #     "Setting check_batched_forward_grad=True requires check_fwd_over_rev to be True")
    # 如果设置了 check_batched_forward_grad=True，则必须同时设置 check_fwd_over_rev=True，否则抛出异常

    tupled_inputs = _as_tuple(inputs)
    # 将输入参数转换为元组形式，以便处理不同情况的输入

    if grad_outputs is None:
        # 如果未指定 grad_outputs，则创建与输出具有相同形状、类型和设备的随机张量

        outputs = _differentiable_outputs(func(*tupled_inputs))
        # 调用 func 函数计算输出，并确保其可以进行自动微分

        tupled_grad_outputs = tuple(
            torch.testing.make_tensor(
                x.shape,
                dtype=x.dtype if x.is_floating_point() or x.is_complex() else torch.double,
                device=x.device,
                low=-1,
                high=1,
                requires_grad=True,
                noncontiguous=gen_non_contig_grad_outputs,
            )
            for x in outputs
        )
        # 为每个输出张量生成随机梯度张量，并确保这些张量可以进行梯度计算
    else:
        tupled_grad_outputs = _as_tuple(grad_outputs)
        # 否则，将给定的 grad_outputs 转换为元组形式

    num_outputs = len(tupled_grad_outputs)
    # 计算输出张量的数量

    # NB: We need to save the requires_grad information about the inputs here because gradcheck detaches inputs
    #     before running forward mode AD
    # 注意：我们需要在这里保存关于输入的 requires_grad 信息，因为 gradcheck 在运行前向模式自动微分之前会分离输入

    diff_input_args_indices = {
        i for i, x in enumerate(tupled_inputs) if is_tensor_like(x) and x.requires_grad
    }
    # 收集需要进行梯度检查的输入参数索引，这些参数需要是张量且设置了 requires_grad=True

    diff_grad_output_indices = {
        i for i, x in enumerate(tupled_grad_outputs) if x.requires_grad
    }
    # 收集需要进行梯度检查的输出梯度索引，这些输出梯度需要设置了 requires_grad=True

    def new_func(*args):
        # 定义一个新的函数 new_func，用于进行梯度检查

        # Restore the requires_grad information
        input_args = tuple(
            x.requires_grad_() if i in diff_input_args_indices else x
            for i, x in enumerate(args[:-num_outputs])
        )
        # 恢复输入参数的 requires_grad 信息，确保需要梯度计算的参数保持标记为 requires_grad=True

        outputs = _differentiable_outputs(func(*input_args))
        # 使用恢复后的输入参数计算输出，并确保输出可以进行自动微分

        grad_outputs = tuple(
            x.requires_grad_() if i in diff_grad_output_indices else x
            for i, x in enumerate(args[-num_outputs:])
        )
        # 恢复输出梯度的 requires_grad 信息，确保需要梯度计算的输出梯度保持标记为 requires_grad=True

        diff_input_args = tuple(
            x for i, x in enumerate(input_args) if i in diff_input_args_indices
        )
        # 提取需要进行梯度计算的输入参数

        grad_inputs = torch.autograd.grad(
            outputs, diff_input_args, grad_outputs, create_graph=True, allow_unused=True
        )
        # 计算输入参数相对于输出的梯度，允许创建计算图并处理未使用的梯度

        grad_inputs = tuple(g for g in grad_inputs if g is not None)
        # 确保梯度不为空的情况下返回梯度元组

        return grad_inputs
        # 返回计算得到的梯度

    return gradcheck(
        new_func,
        tupled_inputs + tupled_grad_outputs,
        eps=eps,
        atol=atol,
        rtol=rtol,
        raise_exception=raise_exception,
        nondet_tol=nondet_tol,
        check_undefined_grad=check_undefined_grad,
        check_grad_dtypes=check_grad_dtypes,
        check_batched_grad=check_batched_grad,
        fast_mode=fast_mode,
        check_forward_ad=check_fwd_over_rev,
        check_backward_ad=check_rev_over_rev,
        masked=masked,
    )
    # 调用 gradcheck 函数进行梯度检查，传入新定义的梯度计算函数 new_func 和相关参数
```