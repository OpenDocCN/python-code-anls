# `.\pytorch\torch\autograd\functional.py`

```py
# mypy: allow-untyped-defs
# 引入类型定义 List 和 Tuple
from typing import List, Tuple

# 引入 PyTorch 库
import torch
# 引入 PyTorch 内部的 _vmap 函数
from torch._vmap_internals import _vmap
# 从当前目录导入 forward_ad 模块作为 fwAD
from . import forward_ad as fwAD

# 模块导出的符号列表
__all__ = ["vjp", "jvp", "jacobian", "hessian", "hvp", "vhp"]

# 实用函数

# 将输入 x 转换为元组，如果本身已经是元组或列表则直接返回
def _as_tuple_nocheck(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


# 确保 inp 是一个 Tensor 的元组，如果不是则进行转换
# 如果指定了参数名和函数名，则在类型错误时抛出详细的异常信息
def _as_tuple(inp, arg_name=None, fn_name=None):
    # 确保 inp 是一个 Tensor 的元组
    # 返回原始 inp 是否为元组以及转换后的元组版本
    if arg_name is None and fn_name is None:
        return _as_tuple_nocheck(inp)

    is_inp_tuple = True
    if not isinstance(inp, tuple):
        inp = (inp,)
        is_inp_tuple = False

    for i, el in enumerate(inp):
        if not isinstance(el, torch.Tensor):
            if is_inp_tuple:
                raise TypeError(
                    f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the"
                    f" value at index {i} has type {type(el)}."
                )
            else:
                raise TypeError(
                    f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the"
                    f" given {arg_name} has type {type(el)}."
                )

    return is_inp_tuple, inp


# 处理返回结果，解包可能嵌套的 Tensor 元组
# to_unpack 应为单个布尔值或两个布尔值组成的元组
# 用于：
# - 当结果与输入 inp 匹配时反转 _as_tuple 的效果
# - 可选地移除由多次调用 _as_tuple 创建的两个元组的嵌套
def _tuple_postprocess(res, to_unpack):
    if isinstance(to_unpack, tuple):
        assert len(to_unpack) == 2
        # 如果第二个元素为 False，则解包两层元组中的第一层
        if not to_unpack[1]:
            res = tuple(el[0] for el in res)
        # 如果第一个元素为 False，则返回第一层元组中的第一个元素
        if not to_unpack[0]:
            res = res[0]
    else:
        # 如果 to_unpack 为 False，则返回结果中的第一个元素
        if not to_unpack:
            res = res[0]
    return res


# 预处理输入，确保输入的 Tensor 需要梯度
# inputs 是需要预处理的 Tensor 元组
# create_graph 指定是否希望梯度流回输入中的 Tensor
# need_graph 指定我们内部是否需要梯度流回结果中的 Tensor
# 注意，我们总是创建一个新的 Tensor 对象，以便能够区分
# 作为参数给出的输入和用户函数自动捕获的相同 Tensor
# 有关详细信息，请参阅此问题：https://github.com/pytorch/pytorch/issues/32576
def _grad_preprocess(inputs, create_graph, need_graph):
    res = []  # 结果列表初始化为空
    # 遍历输入列表中的每个张量
    for inp in inputs:
        # 如果需要创建计算图，并且当前张量需要梯度
        if create_graph and inp.requires_grad:
            # 如果张量不是稀疏张量，使用 .view_as() 创建一个浅拷贝
            if not inp.is_sparse:
                # 使用 .view_as() 方法创建与当前张量相同形状的新张量，并添加到结果列表中
                res.append(inp.view_as(inp))
            else:
                # 对于稀疏张量，无法使用 .view_as() 方法，因此使用 .clone() 创建副本
                res.append(inp.clone())
        else:
            # 如果不需要创建计算图，或者当前张量不需要梯度，则将张量从计算图中分离，并根据需要重新设置梯度属性
            res.append(inp.detach().requires_grad_(need_graph))
    # 返回结果列表转换为元组
    return tuple(res)
# 对生成的张量进行后处理，以避免在用户未请求时返回带有历史记录的张量。
def _grad_postprocess(inputs, create_graph):
    # 检查第一个输入是否为张量
    if isinstance(inputs[0], torch.Tensor):
        # 如果用户未请求创建计算图，返回输入张量的分离版本
        if not create_graph:
            return tuple(inp.detach() for inp in inputs)
        else:
            return inputs
    else:
        # 如果输入不是张量，则递归应用后处理函数
        return tuple(_grad_postprocess(inp, create_graph) for inp in inputs)


# 验证输入张量 v 的形状是否与其他张量 other 匹配
def _validate_v(v, other, is_other_tuple):
    # 检查 v 和 other 的长度是否一致
    if len(other) != len(v):
        if is_other_tuple:
            # 如果 other 是元组且长度不匹配，引发运行时错误
            raise RuntimeError(
                f"v is a tuple of invalid length: should be {len(other)} but got {len(v)}."
            )
        else:
            # 如果 other 不是元组但长度不匹配，引发运行时错误
            raise RuntimeError("The given v should contain a single Tensor.")

    # 检查每个元素在张量列表中的大小是否匹配
    for idx, (el_v, el_other) in enumerate(zip(v, other)):
        if el_v.size() != el_other.size():
            prepend = ""
            if is_other_tuple:
                prepend = f"Entry {idx} in "
            # 如果大小不匹配，引发运行时错误
            raise RuntimeError(
                f"{prepend}v has invalid size: should be {el_other.size()} but got {el_v.size()}."
            )


# 在严格模式下，用于进行必要的检查以提供详细的错误信息
def _check_requires_grad(inputs, input_type, strict):
    # 如果不是严格模式，直接返回
    if not strict:
        return

    # 检查输入类型是否有效
    if input_type not in ["outputs", "grad_inputs", "jacobian", "hessian"]:
        # 如果输入类型无效，引发运行时错误
        raise RuntimeError("Invalid input_type to _check_requires_grad")
    # 遍历输入列表 `inputs`，同时获取索引和对应的输入 `inp`
    for i, inp in enumerate(inputs):
        # 检查输入是否为 None
        if inp is None:
            # 如果是 None，则说明这种情况只会出现在 `grad_inputs` 上。
            raise RuntimeError(
                # 抛出运行时异常，指出用户提供的函数输出与输入 `i` 无关。
                f"The output of the user-provided function is independent of input {i}."
                " This is not allowed in strict mode."
            )
        # 检查输入是否不需要梯度
        if not inp.requires_grad:
            # 根据输入类型 `input_type` 进行不同的异常处理
            if input_type == "hessian":
                raise RuntimeError(
                    # 如果是求 Hessian 矩阵，指出用户提供的函数对输入 `i` 的 Hessian 矩阵无关。
                    f"The hessian of the user-provided function with respect to input {i}"
                    " is independent of the input. This is not allowed in strict mode."
                    " You should ensure that your function is thrice differentiable and that"
                    " the hessian depends on the inputs."
                )
            elif input_type == "jacobian":
                raise RuntimeError(
                    # 如果是求 Jacobian 矩阵，指出用户提供的函数对输入 `i` 的 Jacobian 矩阵无关。
                    "While computing the hessian, found that the jacobian of the user-provided"
                    f" function with respect to input {i} is independent of the input. This is not"
                    " allowed in strict mode. You should ensure that your function is twice"
                    " differentiable and that the jacobian depends on the inputs (this would be"
                    " violated by a linear function for example)."
                )
            elif input_type == "grad_inputs":
                raise RuntimeError(
                    # 如果是求梯度 `grad_inputs`，指出用户提供的函数对输入 `i` 的梯度无关。
                    f"The gradient with respect to input {i} is independent of the inputs of the"
                    " user-provided function. This is not allowed in strict mode."
                )
            else:
                raise RuntimeError(
                    # 如果以上情况都不是，则说明输出 `i` 的函数不需要梯度。
                    f"Output {i} of the user-provided function does not require gradients."
                    " The outputs must be computed in a differentiable manner from the input"
                    " when running in strict mode."
                )
# 定义一个函数 `_autograd_grad`，用于计算梯度，支持部分输出为 `None`，并不对它们计算梯度。
def _autograd_grad(
    outputs,                 # 输出张量的元组
    inputs,                  # 输入张量的元组
    grad_outputs=None,       # 梯度输出的元组，默认为 None
    create_graph=False,      # 是否创建计算图，默认为 False
    retain_graph=None,       # 是否保留计算图，默认为 None
    is_grads_batched=False,  # 梯度是否批处理，默认为 False
):
    # 版本的 autograd.grad，允许 `outputs` 中的部分为 `None`，并且不对它们计算梯度。
    # 额外的约束是 `inputs` 必须是一个元组
    assert isinstance(outputs, tuple)    # 检查 `outputs` 必须是元组类型
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)  # 如果 `grad_outputs` 为 None，则初始化为与 `outputs` 长度相同的元组
    assert isinstance(grad_outputs, tuple)    # 检查 `grad_outputs` 必须是元组类型
    assert len(outputs) == len(grad_outputs)  # 检查 `outputs` 和 `grad_outputs` 的长度必须相等

    new_outputs: Tuple[torch.Tensor, ...] = tuple()         # 新的输出张量元组
    new_grad_outputs: Tuple[torch.Tensor, ...] = tuple()    # 新的梯度输出张量元组
    for out, grad_out in zip(outputs, grad_outputs):
        if out is not None and out.requires_grad:
            new_outputs += (out,)            # 将需要计算梯度的输出张量添加到新的输出元组中
            new_grad_outputs += (grad_out,)  # 将对应的梯度输出张量添加到新的梯度输出元组中

    if len(new_outputs) == 0:
        # 没有需要计算梯度的输出，不需要调用 autograd 引擎
        return (None,) * len(inputs)   # 返回与输入张量元组长度相同的 None 元组作为结果
    else:
        # 调用 torch.autograd.grad 计算梯度
        return torch.autograd.grad(
            new_outputs,                 # 需要计算梯度的输出张量元组
            inputs,                      # 输入张量元组
            new_grad_outputs,            # 对应的梯度输出张量元组
            allow_unused=True,           # 允许输出中有未使用的梯度
            create_graph=create_graph,   # 是否创建计算图
            retain_graph=retain_graph,   # 是否保留计算图
            is_grads_batched=is_grads_batched,  # 梯度是否批处理
        )


# 定义一个函数 `_fill_in_zeros`，用于检测梯度中的 `None`，根据标志位决定是否用 0 填充或者抛出错误。
def _fill_in_zeros(grads, refs, strict, create_graph, stage):
    # 用于检测梯度中的 `None`，根据标志位，要么用相应尺寸的全零张量替换它们，要么抛出错误。
    # strict 和 create graph 允许我们确定何时适合抛出错误，stage 提供信息，用于给出良好的错误消息。
    if stage not in ["back", "back_trick", "double_back", "double_back_trick"]:
        raise RuntimeError(f"Invalid stage argument '{stage}' to _fill_in_zeros")  # 如果 stage 参数不在预期的范围内，抛出运行时错误

    res: Tuple[torch.Tensor, ...] = tuple()   # 结果张量元组
    # 遍历梯度列表 `grads`，同时获取索引 `i` 和对应的梯度 `grads_i`
    for i, grads_i in enumerate(grads):
        # 如果梯度 `grads_i` 为 None
        if grads_i is None:
            # 如果启用了严格模式 `strict`
            if strict:
                # 根据阶段 `stage` 进行不同的错误抛出
                if stage == "back":
                    # 抛出运行时错误，指出用户提供的函数的输出与输入 `i` 无关
                    raise RuntimeError(
                        "The output of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode."
                    )
                elif stage == "back_trick":
                    # 抛出运行时错误，指出使用双向传播技巧计算正向模式梯度时，梯度与输入 `i` 无关
                    raise RuntimeError(
                        f"The gradient with respect to the input is independent of entry {i}"
                        " in the grad_outputs when using the double backward trick to compute"
                        " forward mode gradients. This is not allowed in strict mode."
                    )
                elif stage == "double_back":
                    # 抛出运行时错误，指出用户提供的函数的雅可比矩阵与输入 `i` 无关
                    raise RuntimeError(
                        "The jacobian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode."
                    )
                else:
                    # 抛出运行时错误，指出用户提供的函数的海森矩阵与输入 `i` 无关
                    raise RuntimeError(
                        "The hessian of the user-provided function is independent of "
                        f"entry {i} in the grad_jacobian. This is not allowed in strict "
                        "mode as it prevents from using the double backward trick to "
                        "replace forward mode AD."
                    )
            
            # 如果没有抛出错误，则将 `grads_i` 初始化为与 `refs[i]` 相同形状的零张量
            grads_i = torch.zeros_like(refs[i])
        
        # 如果 `grads_i` 不为 None
        else:
            # 如果启用了严格模式 `strict` 并且 `create_graph=True` 且 `grads_i` 不要求梯度
            if strict and create_graph and not grads_i.requires_grad:
                # 如果阶段 `stage` 不包含 "double"
                if "double" not in stage:
                    # 抛出运行时错误，指出用户提供的函数的雅可比矩阵与输入 `i` 无关
                    raise RuntimeError(
                        "The jacobian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode when create_graph=True."
                    )
                else:
                    # 抛出运行时错误，指出用户提供的函数的海森矩阵与输入 `i` 无关
                    raise RuntimeError(
                        "The hessian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode when create_graph=True."
                    )

        # 将 `grads_i` 添加到结果元组 `res` 中
        res += (grads_i,)

    # 返回结果元组 `res`
    return res
# Public API

# 计算函数在给定输入点处的雅可比矩阵与向量 v 的点积

def vjp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between a vector ``v`` and the Jacobian of the given function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor. 传入的 Python 函数，接受张量输入并返回张量元组或张量。
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``. 函数 ``func`` 的输入。
        v (tuple of Tensors or Tensor): The vector for which the vector
            Jacobian product is computed.  Must be the same size as the output
            of ``func``. This argument is optional when the output of ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
            计算雅可比向量乘积的向量 v。必须与 ``func`` 的输出具有相同的大小。如果 ``func`` 的输出包含单个元素，则此参数是可选的，并且将设置为包含单个 ``1`` 的张量。
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
            如果为 ``True``，则将以可微分的方式计算输出和结果。请注意，当 ``strict`` 为 ``False`` 时，结果不能需要梯度或与输入断开连接。默认为 ``False``。
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vjp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
            如果为 ``True``，当检测到存在一个输入使得所有输出都与其无关时，将引发错误。如果为 ``False``，则对于该输入返回零张量作为 vjp，这是预期的数学值。默认为 ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)`` 函数 ``func(inputs)`` 的输出

            vjp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs. 与输入具有相同形状的点积结果。

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> vjp(exp_reducer, inputs, v)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782]),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]]))

        >>> vjp(exp_reducer, inputs, v, create_graph=True)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782], grad_fn=<SumBackward1>),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]], grad_fn=<MulBackward0>))

        >>> def adder(x, y):
        ...     return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = torch.ones(2)
        >>> vjp(adder, inputs, v)
        (tensor([2.4225, 2.3340]),
         (tensor([2., 2.]), tensor([3., 3.])))
    """
    # 开启 Torch 的梯度计算上下文管理器，确保梯度被启用
    with torch.enable_grad():
        # 将输入转换为元组形式，检查输入是否为元组，并进行命名验证
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vjp")
        # 预处理输入，为梯度计算做准备，创建计算图（如果需要）
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        # 调用用户提供的函数，计算输出
        outputs = func(*inputs)
        # 将输出转换为元组形式，检查输出是否为元组，并进行命名验证
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "vjp"
        )
        # 检查输出是否需要梯度计算，如果需要则进行严格检查
        _check_requires_grad(outputs, "outputs", strict=strict)

        # 如果 v 不为 None，则进行如下处理
        if v is not None:
            # 将 v 转换为元组形式，检查 v 是否为元组，并进行命名验证
            _, v = _as_tuple(v, "v", "vjp")
            # 预处理 v，为梯度计算做准备，不需要创建计算图
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            # 验证 v 的形状是否与输出匹配，如果不匹配则抛出异常
            _validate_v(v, outputs, is_outputs_tuple)
        else:
            # 如果 v 为 None，则检查输出是否只包含单个元素的 Tensor
            if len(outputs) != 1 or outputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the "
                    "user-provided function returns "
                    "a single Tensor with a single element."
                )

    # 根据 create_graph 参数决定是否开启梯度计算
    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        # 计算梯度并返回结果
        grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_graph)
        # 填充梯度结果的零值，以满足严格模式和创建计算图的需求
        vjp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "back")

    # 后处理输出和 vjp，并返回给用户
    outputs = _grad_postprocess(outputs, create_graph)
    vjp = _grad_postprocess(vjp, create_graph)

    # 将输出和 vjp 结果转换回标准的元组形式并返回
    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        vjp, is_inputs_tuple
    )
# 计算给定函数在输入点处的雅可比矩阵与向量 v 的点积。

def jvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between the Jacobian of the given function at the point given by the inputs and a vector ``v``.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Jacobian
            vector product is computed. Must be the same size as the input of
            ``func``. This argument is optional when the input to ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            jvp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the output.

    Note:
        ``autograd.functional.jvp`` computes the jvp by using the backward of
        the backward (sometimes called the double backwards trick). This is not
        the most performant way of computing the jvp. Please consider using
        :func:`torch.func.jvp` or the
        :ref:`low-level forward-mode AD API <forward-mode-ad>` instead.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4, 4)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> jvp(exp_reducer, inputs, v)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106]),
         tensor([6.3090, 4.6742, 7.9114, 8.2106]))

        >>> jvp(exp_reducer, inputs, v, create_graph=True)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SumBackward1>),
         tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SqueezeBackward1>))

        >>> def adder(x, y):
        ...     return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.ones(2), torch.ones(2))
        >>> jvp(adder, inputs, v)
        (tensor([2.2399, 2.5005]),
         tensor([5., 5.]))

    """
    # 启用 Torch 的梯度计算上下文管理器
    with torch.enable_grad():
        # 将输入转换为元组格式，并检查是否是元组
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jvp")
        # 对输入进行梯度预处理，设置需要创建计算图
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        # 如果 v 不为 None，则进行以下操作
        if v is not None:
            # 将 v 转换为元组格式，并检查是否是元组
            _, v = _as_tuple(v, "v", "jvp")
            # 对 v 进行梯度预处理，设置不需要创建计算图
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            # 验证 v 的有效性，确保与 inputs 匹配
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            # 如果 v 为 None，则检查 inputs 的长度和元素个数是否符合条件
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                # 如果不符合条件，抛出运行时错误
                raise RuntimeError(
                    "The vector v can only be None if the input to "
                    "the user-provided function is a single Tensor "
                    "with a single element."
                )

        # 调用用户提供的函数 func，计算输出
        outputs = func(*inputs)
        # 将输出转换为元组格式，并检查是否是元组
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "jvp"
        )
        # 检查输出是否需要梯度
        _check_requires_grad(outputs, "outputs", strict=strict)

        # 对梯度输出进行初始化，为每个输出创建一个与之形状相同的零张量，并设置需要梯度
        grad_outputs = tuple(
            torch.zeros_like(out, requires_grad=True) for out in outputs
        )

        # 计算输入的梯度，使用自动求导机制，并设置需要创建计算图
        grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
        # 检查输入梯度是否需要梯度
        _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

    # 如果需要创建计算图，则进入以下代码块
    if create_graph:
        with torch.enable_grad():
            # 计算梯度结果，使用自动求导机制，并设置需要创建计算图
            grad_res = _autograd_grad(
                grad_inputs, grad_outputs, v, create_graph=create_graph
            )
            # 根据返回的梯度结果、输出等信息填充零值，以进行反向传播技巧
            jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")
    else:
        # 如果不需要创建计算图，则直接计算梯度结果
        grad_res = _autograd_grad(
            grad_inputs, grad_outputs, v, create_graph=create_graph
        )
        # 根据返回的梯度结果、输出等信息填充零值，以进行反向传播技巧
        jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")

    # 对输出和 jvp 进行梯度后处理，根据需要创建计算图
    outputs = _grad_postprocess(outputs, create_graph)
    jvp = _grad_postprocess(jvp, create_graph)

    # 返回经过处理后的输出元组和 jvp 元组
    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        jvp, is_outputs_tuple
    )
# 当前函数用于构造一组标准基底，适用于输入的多个张量。
# - 创建一个大小为N×N的单位矩阵，其中N=sum(tensor_numels)。
# - 将单位矩阵分割成多个块，每个块的大小由 `tensor_numels` 决定。
# - 每个块对应一个张量，且与该张量具有相同的数据类型和设备。
#
# 例如，对于 tensor_numels = [1, 2, 1]，此函数返回：
# ( tensor([[1],     tensor([[0, 0],      tensor([[0],
#           [0],             [1, 0],              [0],
#           [0],             [0, 1],              [0],
#           [0]])  ,         [0, 0]])  ,          [1]])  )
#
# 前提条件：tensor_numels == tuple(tensor.numel() for tensor in tensors)
# 前提条件：tensors 至少包含一个元素。
#
# 有关此函数背后的上下文，请参阅注释: [使用vmap和grad计算多个张量的雅可比矩阵]。
# 所有的前提条件在 torch.autograd.functional.jacobian 中都有保护。
def _construct_standard_basis_for(
    tensors: Tuple[torch.Tensor, ...], tensor_numels: Tuple[int, ...]
) -> Tuple[torch.Tensor, ...]:
    # 检查输入的张量数量与张量尺寸数量是否相同
    assert len(tensors) == len(tensor_numels)
    # 检查输入的张量数量是否大于0
    assert len(tensors) > 0
    # 计算所有张量元素的总数
    total_numel = sum(tensor_numels)
    # 为每个张量创建一个零填充的张量块，尺寸由 tensor_numels 决定
    chunks = tuple(
        tensor.new_zeros(total_numel, tensor_numel)
        for tensor, tensor_numel in zip(tensors, tensor_numels)
    )
    # 对每个块进行对角线填充，以构造标准基底
    diag_start_idx = 0
    for chunk, numel in zip(chunks, tensor_numels):
        chunk.diagonal(diag_start_idx).fill_(1)
        diag_start_idx += numel
    # 返回构造的标准基底张量组成的元组
    return chunks


# 函数用于执行前向自动求导计算的 Jacobian 矩阵
# 如果 `strict=True`，则抛出运行时错误，因为前向模式不支持严格模式
# 请使用反向模式 (`strategy="reverse-mode"`) 代替
def _jacfwd(func, inputs, strict=False, vectorize=False):
    if strict:
        raise RuntimeError(
            "torch.autograd.functional.jacobian: `strict=True` "
            'and `strategy="forward-mode"` are not supported together (yet). '
            "Please either set `strict=False` or "
            '`strategy="reverse-mode"`.'
        )
    # 将输入转换为元组形式以便处理
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
    # 输出信息列表，用于存储 Jacobian 相关的信息
    output_info = []
    if vectorize:
        # 如果 vectorize 为真，则执行以下计算雅可比矩阵的步骤

        # NOTE: [Computing jacobian with vmap and grad for multiple outputs]
        # 根据多输出使用 vmap 和 grad 计算雅可比矩阵的说明

        # Step 1: Prepare tangents
        # 步骤1：准备切线向量
        input_numels = tuple(input.numel() for input in inputs)
        # 计算输入张量的元素数量，并转为元组保存

        # Step 2: Compute vmap over computation with dual tensors
        # 步骤2：使用双重张量计算中的 vmap
        def jvp(tangents):
            with fwAD.dual_level():
                # 进入双重计算环境
                dual_inputs = tuple(
                    fwAD.make_dual(input, tangent.view_as(input))
                    for input, tangent in zip(inputs, tangents)
                )
                # 创建双重输入，并将切线视为输入的形式
                _is_outputs_tuple, dual_outputs = _as_tuple(
                    func(*dual_inputs), "outputs"
                )
                # 检查输出是否为元组，并获取双重输出
                output_info.append(_is_outputs_tuple)
                jv = []
                primal_outs = []
                for dual_out in dual_outputs:
                    primal, tangent = fwAD.unpack_dual(dual_out)
                    primal_outs.append(primal)
                    if tangent is not None:
                        jv.append(tangent)
                    else:
                        jv.append(torch.zeros_like(primal))
                # 拆分双重输出，获取原始输出和切线
                output_info.append(primal_outs)
                return tuple(jv)

        outputs_before_split = _vmap(jvp)(tangents)
        # 使用 vmap 对 jvp 函数进行批处理

        is_outputs_tuple, outputs = output_info
        # 获取输出信息元组

        # Step 3: for each of the output tangents, split along dim 0
        # 步骤3：对每个输出切线，在维度0上进行分割
        jacobian_input_output = []
        for jac_output_i, output_i in zip(outputs_before_split, outputs):
            jacobian_output_i_output = []
            for jac, input_j in zip(jac_output_i.split(input_numels, dim=0), inputs):
                # 需要转置雅可比矩阵，因为在正向自动微分中，批处理维度代表输入的维度
                jacobian_input_i_output_j = jac.permute(*range(1, jac.ndim), 0).reshape(
                    (*output_i.shape, *input_j.shape)
                )  # noqa: C409
                # 调整形状以匹配输出和输入的维度

                jacobian_output_i_output.append(jacobian_input_i_output_j)
            jacobian_input_output.append(jacobian_output_i_output)
        # 将结果添加到雅可比输入输出列表中

        # Omit [Step 4] because everything is already transposed w/ forward AD
        # 忽略步骤4，因为所有内容都已经使用正向自动微分转置

        return _tuple_postprocess(
            jacobian_input_output, (is_outputs_tuple, is_inputs_tuple)
        )
        # 返回处理后的元组结果
    else:
        # 如果 vectorize 不为真，则抛出未实现错误
        raise NotImplementedError(
            "Computing Jacobian using forward-AD or forward-over-reverse Hessian is"
            "only implemented for `vectorize=True`."
        )
def jacobian(
    func,
    inputs,
    create_graph=False,
    strict=False,
    vectorize=False,
    strategy="reverse-mode",
):
    r"""Compute the Jacobian of a given function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.  # 输入参数 func 是一个接受张量输入并返回张量元组或张量的 Python 函数。
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.  # 输入参数 inputs 是传递给函数 func 的张量元组或张量。
        create_graph (bool, optional): If ``True``, the Jacobian will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.  # 是否创建计算可导的雅可比矩阵。当 strict 为 False 时，结果不会要求梯度，也不会与输入断开连接。
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.  # 是否启用严格模式。当存在某些输入使得所有输出都与其无关时，如果 strict 为 True，会触发错误。如果为 False，针对这些输入返回零张量作为雅可比矩阵，这是数学上的预期值。
        vectorize (bool, optional): This feature is experimental.
            Please consider using :func:`torch.func.jacrev` or
            :func:`torch.func.jacfwd` instead if you are looking for something
            less experimental and more performant.
            When computing the jacobian, usually we invoke
            ``autograd.grad`` once per row of the jacobian. If this flag is
            ``True``, we perform only a single ``autograd.grad`` call with
            ``batched_grad=True`` which uses the vmap prototype feature.
            Though this should lead to performance improvements in many cases,
            because this feature is still experimental, there may be performance
            cliffs. See :func:`torch.autograd.grad`'s ``batched_grad`` parameter for
            more information.  # 此特性为实验性质。如果寻求更少实验性和更高性能的解决方案，请考虑使用 torch.func.jacrev 或 torch.func.jacfwd。如果设置为 True，在计算雅可比矩阵时，通常我们会对每行雅可比矩阵调用一次 autograd.grad。此时只会执行一次 autograd.grad 调用，使用 batched_grad=True，并使用 vmap 原型特性。尽管在许多情况下这会带来性能改进，但因为此特性仍处于实验阶段，可能会有性能峭壁。有关更多信息，请参阅 torch.autograd.grad 的 batched_grad 参数。
        strategy (str, optional): Set to ``"forward-mode"`` or ``"reverse-mode"`` to
            determine whether the Jacobian will be computed with forward or reverse
            mode AD. Currently, ``"forward-mode"`` requires ``vectorized=True``.
            Defaults to ``"reverse-mode"``. If ``func`` has more outputs than
            inputs, ``"forward-mode"`` tends to be more performant. Otherwise,
            prefer to use ``"reverse-mode"``.  # 设置为 "forward-mode" 或 "reverse-mode"，确定使用前向或后向自动求导来计算雅可比矩阵。当前 "forward-mode" 需要 vectorized=True。如果 func 的输出多于输入，"forward-mode" 往往更高效。否则，建议使用 "reverse-mode"。
    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if there is a single
        input and output, this will be a single Tensor containing the
        Jacobian for the linearized inputs and output. If one of the two is
        a tuple, then the Jacobian will be a tuple of Tensors. If both of
        them are tuples, then the Jacobian will be a tuple of tuple of
        Tensors where ``Jacobian[i][j]`` will contain the Jacobian of the
        ``i``\th output and ``j``\th input and will have as size the
        concatenation of the sizes of the corresponding output and the
        corresponding input and will have same dtype and device as the
        corresponding input. If strategy is ``forward-mode``, the dtype will be
        that of the output; otherwise, the input.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = torch.rand(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> jacobian(exp_reducer, inputs)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],
                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]])

        >>> jacobian(exp_reducer, inputs, create_graph=True)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],
                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]], grad_fn=<ViewBackward>)

        >>> def exp_adder(x, y):
        ...     return 2 * x.exp() + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> jacobian(exp_adder, inputs)
        (tensor([[2.8052, 0.0000],
                [0.0000, 3.3963]]),
         tensor([[3., 0.],
                 [0., 3.]]))
    """
    # 检查策略是否为"forward-mode"或"reverse-mode"，否则抛出异常
    assert strategy in ("forward-mode", "reverse-mode"), (
        'Expected strategy to be either "forward-mode" or "reverse-mode". Hint: If your '
        'function has more outputs than inputs, "forward-mode" tends to be more performant. '
        'Otherwise, prefer to use "reverse-mode".'
    )
    # 如果策略为"forward-mode"，且create_graph为True，则抛出NotImplementedError
    if strategy == "forward-mode":
        if create_graph:
            raise NotImplementedError(
                "torch.autograd.functional.jacobian: `create_graph=True` "
                'and `strategy="forward-mode"` are not supported together (yet). '
                "Please either set `create_graph=False` or "
                '`strategy="reverse-mode"`.'
            )
        # 调用_jacfwd函数并返回结果
        return _jacfwd(func, inputs, strict, vectorize)
# 计算给定标量函数的 Hessian 矩阵。
def hessian(
    func,
    inputs,
    create_graph=False,
    strict=False,
    vectorize=False,
    outer_jacobian_strategy="reverse-mode",
):
    r"""Compute the Hessian of a given scalar function.

    Args:
        func (function): 接受张量输入并返回单个元素张量的 Python 函数。
        inputs (tuple of Tensors or Tensor): 函数 ``func`` 的输入。
        create_graph (bool, optional): 如果为 ``True``，将以可微分方式计算 Hessian 矩阵。
            注意，当 ``strict`` 为 ``False`` 时，结果不能要求梯度，也不能与输入断开连接。
            默认为 ``False``。
        strict (bool, optional): 如果为 ``True``，当检测到存在某个输入使得所有输出与之无关时，将引发错误。
            如果为 ``False``，对于这些输入，将返回一个全零的 Hessian 矩阵，这是数学上的期望值。
            默认为 ``False``。
        vectorize (bool, optional): 这个特性是实验性的。
            如果你希望寻找更少实验性、更高效的方法，请考虑使用 :func:`torch.func.hessian`。
            计算 Hessian 矩阵时，通常我们对 Hessian 的每一行调用一次 ``autograd.grad``。
            如果该标志为 ``True``，我们将使用 vmap 原型功能作为后端，以向量化调用 ``autograd.grad``，
            因此只调用一次而不是每行一次。这应该可以在许多用例中提高性能，但由于此功能尚不完整，
            可能会有性能突变。请使用 `torch._C._debug_only_display_vmap_fallback_warnings(True)`
            来显示任何性能警告，并为您的用例提交问题报告。
            默认为 ``False``。
        outer_jacobian_strategy (str, optional): 计算 Hessian 矩阵时，通过计算 Jacobian 矩阵的 Jacobian 矩阵来实现。
            内部 Jacobian 矩阵始终使用反向模式自动求导。设置策略为 ``"forward-mode"`` 或 ``"reverse-mode"``
            决定了外部 Jacobian 矩阵是使用前向还是反向模式自动求导。目前，使用 ``"forward-mode"`` 计算外部
            Jacobian 矩阵需要 ``vectorize=True``。
            默认为 ``"reverse-mode"``。
    """
    Returns:
        Hessian (Tensor or a tuple of tuple of Tensors): if there is a single input,
        this will be a single Tensor containing the Hessian for the input.
        If it is a tuple, then the Hessian will be a tuple of tuples where
        ``Hessian[i][j]`` will contain the Hessian of the ``i``\th input
        and ``j``\th input with size the sum of the size of the ``i``\th input plus
        the size of the ``j``\th input. ``Hessian[i][j]`` will have the same
        dtype and device as the corresponding ``i``\th input.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> hessian(pow_reducer, inputs)
        tensor([[[[5.2265, 0.0000],
                  [0.0000, 0.0000]],
                 [[0.0000, 4.8221],
                  [0.0000, 0.0000]]],
                [[[0.0000, 0.0000],
                  [1.9456, 0.0000]],
                 [[0.0000, 0.0000],
                  [0.0000, 3.2550]]]])

        >>> hessian(pow_reducer, inputs, create_graph=True)
        tensor([[[[5.2265, 0.0000],
                  [0.0000, 0.0000]],
                 [[0.0000, 4.8221],
                  [0.0000, 0.0000]]],
                [[[0.0000, 0.0000],
                  [1.9456, 0.0000]],
                 [[0.0000, 0.0000],
                  [0.0000, 3.2550]]]], grad_fn=<ViewBackward>)


        >>> def pow_adder_reducer(x, y):
        ...     return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> hessian(pow_adder_reducer, inputs)
        ((tensor([[4., 0.],
                  [0., 4.]]),
          tensor([[0., 0.],
                  [0., 0.]])),
         (tensor([[0., 0.],
                  [0., 0.]]),
          tensor([[6., 0.],
                  [0., 6.]])))
    """
    # 将输入参数转换为元组形式，确保统一处理
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hessian")
    # 检查外部雅可比矩阵策略是否合法
    assert outer_jacobian_strategy in (
        "forward-mode",
        "reverse-mode",
    ), 'Expected strategy to be either "forward-mode" or "reverse-mode".'

    def ensure_single_output_function(*inp):
        # 调用用户提供的函数，获取输出
        out = func(*inp)
        # 将输出转换为元组形式，确保统一处理
        is_out_tuple, t_out = _as_tuple(
            out, "outputs of the user-provided function", "hessian"
        )
        # 检查输出是否需要梯度
        _check_requires_grad(t_out, "outputs", strict=strict)

        # 确保函数返回值为单个 Tensor
        if is_out_tuple or not isinstance(out, torch.Tensor):
            raise RuntimeError(
                "The function given to hessian should return a single Tensor"
            )

        # 确保返回的 Tensor 只包含一个元素
        if out.nelement() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to hessian should contain a single element"
            )

        return out.squeeze()
    # 定义一个函数 jac_func，接受任意数量的输入参数 *inp
    def jac_func(*inp):
        # 如果外部雅可比策略为 "forward-mode"
        if outer_jacobian_strategy == "forward-mode":
            # 将输入参数中的每个张量都设置为需要梯度计算（create_graph=True），以防止它们被分离
            inp = tuple(t.requires_grad_(True) for t in inp)
        
        # 调用 jacobian 函数计算输入函数 jac_func 的雅可比矩阵
        jac = jacobian(ensure_single_output_function, inp, create_graph=True)
        
        # 检查计算得到的雅可比矩阵 jac 是否所有输入都需要梯度，如果不是则引发警告或错误
        _check_requires_grad(jac, "jacobian", strict=strict)
        
        # 返回计算得到的雅可比矩阵
        return jac

    # 调用 jacobian 函数计算包含所有输入参数的整体雅可比矩阵 res
    res = jacobian(
        jac_func,                        # 输入函数为 jac_func
        inputs,                          # 输入参数为 inputs
        create_graph=create_graph,       # 是否创建计算图
        strict=strict,                   # 是否严格检查梯度需求
        vectorize=vectorize,             # 是否向量化计算
        strategy=outer_jacobian_strategy # 使用的外部雅可比策略
    )
    
    # 对计算结果 res 进行后处理，并返回结果
    return _tuple_postprocess(res, (is_inputs_tuple, is_inputs_tuple))
def vhp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between vector ``v`` and Hessian of a  given scalar function at a specified point.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector Hessian
            product is computed. Must be the same size as the input of
            ``func``. This argument is optional when ``func``'s input contains
            a single element and (if it is not provided) will be set as a
            Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vhp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            vhp (tuple of Tensors or Tensor): result of the dot product with the
            same shape as the inputs.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> vhp(pow_reducer, inputs, v)
        (tensor(0.5591),
         tensor([[1.0689, 1.2431],
                 [3.0989, 4.4456]]))
        >>> vhp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.5591, grad_fn=<SumBackward0>),
         tensor([[1.0689, 1.2431],
                 [3.0989, 4.4456]], grad_fn=<MulBackward0>))
        >>> def pow_adder_reducer(x, y):
        ...     return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> vhp(pow_adder_reducer, inputs, v)
        (tensor(4.8053),
         (tensor([0., 0.]),
          tensor([6., 6.])))

    """
    # 启用 Torch 梯度计算上下文管理器
    with torch.enable_grad():
        # 将输入参数转换为元组形式，检查和处理可能存在的梯度需求
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vhp")
        # 预处理输入参数的梯度计算相关操作，确保创建计算图并需求梯度
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        # 如果提供了向量 v，则处理其梯度计算相关操作
        if v is not None:
            _, v = _as_tuple(v, "v", "vhp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            # 如果输入不是单个含有一个元素的张量，则抛出运行时错误
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the input to the user-provided function "
                    "is a single Tensor with a single element."
                )

        # 调用用户提供的函数计算输出
        outputs = func(*inputs)
        # 将输出转换为元组形式，检查是否为多输出情况，并处理梯度需求
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "vhp"
        )
        # 检查输出是否需要梯度，并执行严格模式的检查
        _check_requires_grad(outputs, "outputs", strict=strict)

        # 如果输出是元组形式或第一个输出不是 Torch 张量，则抛出运行时错误
        if is_outputs_tuple or not isinstance(outputs[0], torch.Tensor):
            raise RuntimeError(
                "The function given to vhp should return a single Tensor"
            )

        # 如果输出张量不是单个元素，则抛出运行时错误
        if outputs[0].nelement() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to vhp should contain a single element"
            )

        # 计算 Jacobian 矩阵，自动求取梯度
        jac = _autograd_grad(outputs, inputs, create_graph=True)
        # 检查 Jacobian 矩阵是否需要梯度，并执行严格模式的检查
        _check_requires_grad(jac, "jacobian", strict=strict)

    # 根据是否创建计算图来设置 Torch 梯度是否可用
    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        # 使用 Torch 梯度是否可用来计算 vhp
        grad_res = _autograd_grad(jac, inputs, v, create_graph=create_graph)
        # 填充零值来处理梯度结果 grad_res，严格模式、创建计算图、双向反向传播
        vhp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "double_back")

    # 后处理输出结果的梯度计算
    outputs = _grad_postprocess(outputs, create_graph)
    vhp = _grad_postprocess(vhp, create_graph)

    # 后处理元组形式的输出结果并返回
    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        vhp, is_inputs_tuple
    )
# 定义函数 hvp，用于计算标量函数在指定点的 Hessian 矩阵与向量 v 的乘积
def hvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between the scalar function's Hessian and a vector ``v`` at a specified point.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Hessian vector
            product is computed. Must be the same size as the input of
            ``func``. This argument is optional when ``func``'s input contains
            a single element and (if it is not provided) will be set as a
            Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            hvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            hvp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> hvp(pow_reducer, inputs, v)
        (tensor(0.1448),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]]))

        >>> hvp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.1448, grad_fn=<SumBackward0>),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]], grad_fn=<MulBackward0>))


        >>> def pow_adder_reducer(x, y):
        ...     return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> hvp(pow_adder_reducer, inputs, v)
        (tensor(2.3030),
         (tensor([0., 0.]),
          tensor([6., 6.])))

    Note:

        This function is significantly slower than `vhp` due to backward mode AD constraints.
        If your functions is twice continuously differentiable, then hvp = vhp.t(). So if you
        know that your function satisfies this condition, you should use vhp instead that is
        much faster with the current implementation.

    """
    # 启用 Torch 的梯度计算上下文管理器
    with torch.enable_grad():
        # 将输入转换为元组形式，并检查输入是否合法
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hvp")
        # 对输入进行梯度预处理，为反向传播准备
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        # 如果 v 不为空，则进行相应处理
        if v is not None:
            # 将 v 转换为元组形式，并检查合法性
            _, v = _as_tuple(v, "v", "hvp")
            # 对 v 进行梯度预处理，不需要计算图
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            # 验证 v 是否合法
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            # 如果 v 为空，则要求 inputs 必须是单个元素的 Tensor
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the input to the user-provided function "
                    "is a single Tensor with a single element."
                )

        # 调用用户提供的函数 func，获取其输出
        outputs = func(*inputs)
        # 将输出转换为元组形式，并检查其是否需要梯度
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "hvp"
        )
        # 检查输出是否都需要梯度
        _check_requires_grad(outputs, "outputs", strict=strict)

        # 确保 func 返回的是单个 Tensor
        if is_outputs_tuple or not isinstance(outputs[0], torch.Tensor):
            raise RuntimeError(
                "The function given to hvp should return a single Tensor"
            )

        # 确保 func 返回的 Tensor 中只有一个元素
        if outputs[0].nelement() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to hvp should contain a single element"
            )

        # 计算输出关于输入的雅可比矩阵
        jac = _autograd_grad(outputs, inputs, create_graph=True)
        # 检查雅可比矩阵是否需要梯度
        _check_requires_grad(jac, "jacobian", strict=strict)

        # 创建一个与输入形状相同的零张量元组，用于存储梯度与雅可比矩阵的乘积
        grad_jac = tuple(torch.zeros_like(inp, requires_grad=True) for inp in inputs)

        # 对雅可比矩阵再次进行反向传播，得到双重梯度
        double_back = _autograd_grad(jac, inputs, grad_jac, create_graph=True)
        # 检查双重梯度是否需要梯度
        _check_requires_grad(jac, "hessian", strict=strict)

    # 根据 create_graph 设置是否启用梯度计算
    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        # 计算双重梯度与 v 的内积，得到 Hessian-向量积 (HvP)
        grad_res = _autograd_grad(double_back, grad_jac, v, create_graph=create_graph)
        # 将结果填充零，确保维度与 inputs 一致，用于 double back trick
        hvp = _fill_in_zeros(
            grad_res, inputs, strict, create_graph, "double_back_trick"
        )

    # 对输出和 HVP 进行梯度后处理
    outputs = _grad_postprocess(outputs, create_graph)
    hvp = _grad_postprocess(hvp, create_graph)

    # 返回经过处理的输出元组和 HVP 元组
    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        hvp, is_inputs_tuple
    )
```