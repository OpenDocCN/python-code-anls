# `.\pytorch\torch\_custom_op\autograd.py`

```
# mypy: allow-untyped-defs
# 导入PyTorch库
import torch
# 导入PyTorch的_pytree模块，用于支持树形数据结构
import torch.utils._pytree as pytree
# 导入命名元组(namedtuple)功能
from collections import namedtuple
# 导入functools模块，用于高阶函数操作
import functools


# NOTE [CustomOp autograd kernel indirection]
# 我们将`inner`注册为此custom_op的自动求导内核。
# `inner`要么调用用户注册的自动求导公式，
# 要么进入一个`autograd_not_implemented`内核。
#
# 存在此间接性的原因是为了能够交换自动求导内核（PyTorch分发器实际上不允许我们这样做）。
# 默认情况下，我们希望使用`autograd_not_implemented`行为，
# 但用户可能会注册一个实际上是反向公式的东西。
def autograd_kernel_indirection(custom_op):
    # 定义自动求导的回退函数
    autograd_fallback = autograd_not_implemented(custom_op)

    def inner(*args, **kwargs):
        # 如果custom_op具有'autograd'实现，则获取其对应的内核函数并调用
        if custom_op._has_impl('autograd'):
            kernel = custom_op._get_impl('autograd').func
            return kernel(*args, **kwargs)
        
        # 如NOTE中所述 ["backward", "save_for_backward", and "autograd"]，
        # 如果用户给出了"backward"和"save_for_backward"，我们生成"autograd"实现。
        # 如果用户只提供了其中之一，则报告错误。
        if custom_op._has_impl('save_for_backward') or custom_op._has_impl('backward'):
            missing = (
                'save_for_backward' if custom_op._has_impl('backward')
                else 'backward'
            )
            found = 'save_for_backward' if missing == 'backward' else 'backward'
            loc = custom_op._get_impl(found).location
            raise RuntimeError(
                f"We found a '{found}' registration for {custom_op} at "
                f"{loc} but were unable to find a '{missing}' registration. "
                f"To use the CustomOp API to register a backward formula, "
                f"please provide us both a backward function and a "
                f"'save for backward' function via `impl_backward` and "
                f"`impl_save_for_backward` respectively.")
        
        # 否则，调用autograd回退函数
        return autograd_fallback(*args, **kwargs)
    
    return inner


# TODO(#101191): 使用实际的C++自动求导未实现的回退，
# 或者将默认的自动求导回退更改为自动求导未实现的回退。
def autograd_not_implemented(custom_op):
    def kernel(*args, **kwargs):
        # 如果梯度已启用且输入中有任何一个Tensor是requires_grad的，则抛出错误
        if torch.is_grad_enabled() and pytree.tree_any(
            lambda x: isinstance(x, torch.Tensor) and x.requires_grad, (args, kwargs)
        ):
            raise RuntimeError("Autograd has not been implemented for operator")
        
        # 使用torch._C._AutoDispatchBelowAutograd()，返回custom_op的结果
        with torch._C._AutoDispatchBelowAutograd():
            return custom_op(*args, **kwargs)
    
    return kernel


def mark_non_differentiable(ctx, output, output_differentiability):
    # 输出类型限定为以下类型之一：
    # - Tensor
    # - Tensor[]
    # - int, bool, Scalar, float
    # 详见 _check_can_register_backward
    # 检查是否给定了输出差异性标志
    if output_differentiability is not None:
        # 如果输出不是元组，则转换为元组
        if not isinstance(output, tuple):
            tuple_output = (output,)
        else:
            tuple_output = output  # type: ignore[assignment]
        
        # 断言输出差异性标志的长度与输出元组长度相同
        assert len(output_differentiability) == len(tuple_output)
        
        # 存储不可微分张量的列表
        non_differentiable_tensors = []
        
        # 遍历输出差异性标志和输出元组，检查每个输出的可微分性
        for idx, (differentiable, out) in enumerate(zip(output_differentiability, tuple_output)):
            # 如果输出是 PyTorch 的张量
            if isinstance(out, torch.Tensor):
                # 如果标记为不可微分，则将其添加到不可微分张量列表中
                if not differentiable:
                    non_differentiable_tensors.append(out)
                continue
            
            # 如果输出是列表
            if isinstance(out, list):
                # 如果标记为不可微分，则将其中所有元素添加到不可微分张量列表中
                if not differentiable:
                    non_differentiable_tensors.extend(out)
                continue
            
            # 如果输出不是张量也不是列表，并且被标记为可微分，则抛出运行时错误
            if differentiable:
                raise RuntimeError(
                    f"With output_differentiability={output_differentiability}. "
                    f"At idx {idx}, we received an object of type {type(out)} that "
                    f"is not a Tensor, so it cannot have be marked as differentiable in "
                    f"output_differentiability.")
        
        # 如果存在不可微分张量，则在上下文中标记它们为不可微分
        if non_differentiable_tensors:
            ctx.mark_non_differentiable(*non_differentiable_tensors)
# 定义构建自动微分内核的函数，返回一个应用函数
def construct_autograd_kernel(
        schema,
        output_differentiability,
        custom_op,
        op_overload,
        save_for_backward_fn,
        backward_fn):

    # 定义应用函数 apply，接受任意数量的参数
    def apply(*args):
        # 将输入参数 args 扁平化，并获取其结构信息
        flat_args, spec = pytree.tree_flatten(args)
        out_spec = None

        # 定义前向传播函数 forward，接受一个上下文对象 ctx 和扁平化的参数 flat_args
        def forward(ctx, *flat_args):
            # 设置上下文中的梯度材料化为 True
            ctx.set_materialize_grads(True)
            # 根据结构信息将扁平化的参数恢复成树形结构的 args
            args = pytree.tree_unflatten(list(flat_args), spec)
            # 使用自动分发功能执行重载的操作 op_overload
            with torch._C._AutoDispatchBelowAutograd():
                output = op_overload(*args)

            # 使用参数信息为后向传播提供更好的错误消息
            args_info = namedtuple_args(
                schema, pytree.tree_map(type, args))

            # 构建保存后向传播信息所需的输入
            save_for_backward_fn_inputs = namedtuple_args(schema, args)
            to_save = save_for_backward_fn(save_for_backward_fn_inputs, output)

            # 将保存的数据和参数信息保存到上下文中
            save_pytree_for_backward(ctx, (to_save, args_info))
            # 标记输出张量的非可微部分
            mark_non_differentiable(ctx, output, output_differentiability)

            # 使用 nonlocal 关键字更新输出结构信息
            nonlocal out_spec
            flat_output, out_spec = pytree.tree_flatten(output)
            return tuple(flat_output)

        # 定义后向传播函数 backward，接受一个上下文对象 ctx 和扁平化的梯度 grad_output
        def backward(ctx, *flat_grad_output):
            # 断言确保 out_spec 不为空
            assert out_spec is not None
            # 将扁平化的梯度 grad_output 恢复成树形结构 grads
            grads = pytree.tree_unflatten(list(flat_grad_output), out_spec)
            # 解压保存的数据和参数信息
            saved, args_info = unpack_saved(ctx)
            # 内部上下文对象，目前为空对象
            inner_ctx = object()
            # 如果 grads 不是元组，则转换为元组形式
            if not isinstance(grads, tuple):
                grads = (grads,)
            # 调用用户定义的后向传播函数 backward_fn，计算梯度输入字典
            grad_inputs_dict = backward_fn(inner_ctx, saved, *grads)

            # 验证梯度输入字典的格式是否符合 autograd.Function 的要求
            validate_grad_inputs_dict(grad_inputs_dict, custom_op, args_info)
            return grad_inputs_dict_to_flat_tuple(grad_inputs_dict, args_info)

        # 生成自动微分函数类，包含自定义操作的名称、前向和后向传播函数
        generated_cls = gen_autograd_function(
            custom_op._opname + '_customop', forward, backward)

        # 应用自动生成的类对扁平化参数进行前向传播
        flat_output = generated_cls.apply(*flat_args)
        # 断言确保 out_spec 不为空
        assert out_spec is not None
        # 将扁平化的输出恢复成树形结构，并返回
        return pytree.tree_unflatten(list(flat_output), out_spec)

    # 返回 apply 函数作为构建自动微分内核的结果
    return apply


# 生成自动微分函数的工具函数，返回一个带有指定名称、前向和后向传播方法的类
def gen_autograd_function(name, forward, backward):
    generated_cls = type(
        name,
        (torch.autograd.Function,),
        {
            'forward': staticmethod(forward),
            'backward': staticmethod(backward),
        }
    )
    return generated_cls


# 基于给定的 schema 创建具名元组类，用于存储参数信息
@functools.lru_cache
def namedtuple_args_cls(schema):
    attribs = [arg.name for arg in schema.arguments.flat_all]
    name = str(schema.name) + "_args"
    # 创建具名元组类，名称为 schema.name + "_args"，包含指定的属性列表
    tuple_cls = namedtuple(name, attribs)  # type: ignore[misc]
    return tuple_cls


# 根据给定的 schema 和 args 创建具名元组实例，用于存储参数信息
def namedtuple_args(schema, args):
    # 断言确保参数 args 是元组形式
    assert isinstance(args, tuple)
    # 获取指定 schema 的具名元组类
    tuple_cls = namedtuple_args_cls(schema)
    # 使用元组类和 args 创建具名元组实例并返回
    return tuple_cls(*args)


# 验证梯度输入字典的格式是否符合 autograd.Function 的要求
def validate_grad_inputs_dict(grad_inputs_dict, forward_op, args_info):
    # 定义错误处理函数，用于抛出运行时异常并指示发生错误的具体位置
    def error(what):
        # 获取反向操作的实现细节，用于指示错误发生的具体位置
        backward = forward_op._get_impl('backward')
        # 抛出运行时异常，提供错误消息，指示反向操作返回的类型不符合预期
        raise RuntimeError(
            f"In the backward function defined for {forward_op} at "
            f"{backward.location} using the CustomOp API, {what}")

    # 检查梯度输入字典是否为字典类型，如果不是则调用错误处理函数报错
    if not isinstance(grad_inputs_dict, dict):
        error(f"expected the output of the backward function to be a dict but "
              f"got {type(grad_inputs_dict)}")

    # 获取预期的键集合，这些键是期望在梯度输入字典中出现的参数名称
    expected_keys = {arg.name for arg in forward_op._schema.arguments.flat_all
                     if arg.type.is_tensor_like()}
    # 获取实际的键集合，即梯度输入字典中实际存在的参数名称
    actual_keys = grad_inputs_dict.keys()
    # 检查预期的键集合与实际的键集合是否一致，如果不一致则调用错误处理函数报错
    if expected_keys != actual_keys:
        error(f"expected the returned grad_input dict to have keys "
              f"{expected_keys} but got {actual_keys}. The backward "
              f"function must return a gradient (can be None) for each arg "
              f"to the CustomOp that may be a Tensor or Sequence[Tensor]. "
              f"Args declared to be non-Tensor-like types should not appear "
              f"in the grad_input dict")

    # 遍历梯度输入字典中的每个参数名称和其对应的梯度
    for name, grad in grad_inputs_dict.items():
        # 获取参数信息，即该参数的预期类型
        arg_info = getattr(args_info, name)

        # 如果参数信息是列表，则预期梯度也应该是列表或元组，进行相应的类型和长度检查
        if isinstance(arg_info, list):
            # 如果梯度不是列表或元组类型，则调用错误处理函数报错
            if not isinstance(grad, (tuple, list)):
                error(f"for input '{name}' expected the grad_input dict to "
                      f"hold a list of gradients but got object of type "
                      f"{type(grad)}.")
            # 如果梯度的长度与参数信息列表长度不符，则调用错误处理函数报错
            if not len(grad) == len(arg_info):
                error(f"for input '{name}' expected the grad_input dict to "
                      f"hold a list of {len(arg_info)} gradients but got "
                      f"{len(grad)}")
            # 遍历检查每个梯度及其对应的参数信息是否符合预期的类型要求
            for idx, (g, info) in enumerate(zip(grad, arg_info)):
                # 如果梯度为 None，则跳过检查
                if g is None:
                    continue
                # 如果梯度不是 Tensor 类型，则调用错误处理函数报错
                if not isinstance(g, torch.Tensor):
                    error(f"for input '{name}' expected the grad_input dict to "
                          f"hold a list of None or Tensor gradients but got "
                          f"object of {type(g)} at index {idx}")
                # 如果参数信息不是 Tensor 类型的子类，则调用错误处理函数报错
                if not issubclass(info, torch.Tensor):
                    error(f"for input '{name}', got a Tensor as the gradient "
                          f"for the {idx}-th value but expected None because "
                          f"the {idx}-th value was not a Tensor (it was "
                          f"type {arg_info}")
            continue

        # 如果梯度为 None，则跳过检查
        if grad is None:
            continue
        # 如果梯度不是 Tensor 类型，则调用错误处理函数报错
        if not isinstance(grad, torch.Tensor):
            error(f"got object of type {type(grad)} as the gradient for input "
                  f"'{name}', "
                  f"but expected the gradient to be either None or a Tensor")
        # 如果参数信息不是 Tensor 类型的子类，则调用错误处理函数报错
        if not issubclass(arg_info, torch.Tensor):
            error(f"got a Tensor as the gradient for input '{name}' but "
                  f"expected None as the gradient because input '{name}' "
                  f"was not a Tensor (it was type {arg_info}).")
# 将梯度输入字典转换为扁平化的元组。grad_inputs_dict 是包含梯度输入的字典，args_info 是一个具名元组，描述了参数的信息。
def grad_inputs_dict_to_flat_tuple(grad_inputs_dict, args_info):
    # 结果列表，用于存储转换后的结果
    result = []
    # 遍历 args_info 的每个元素
    for name, arg_info in args_info._asdict().items():
        # 如果当前参数名不在 grad_inputs_dict 中，则使用 None 替代，并添加到结果中
        if name not in grad_inputs_dict:
            result.append(pytree.tree_map(lambda x: None, arg_info))
            continue
        # 如果在 grad_inputs_dict 中找到了当前参数名，直接添加到结果中
        result.append(grad_inputs_dict[name])
    # 将结果列表转换为元组，返回
    return tuple(pytree.tree_leaves(result))

# 将 "stuff"（一个 pytree）保存到 ctx 对象中，使用 unpack_saved 函数来解包它。
# autograd.Function 建议用户使用 ctx.save_for_backward 来保存张量（以避免引用循环），非张量数据保存到 ctx 对象中。
def save_pytree_for_backward(ctx, stuff):
    # 将 pytree 结构扁平化，并返回扁平化后的列表和结构说明
    flat_stuff, spec = pytree.tree_flatten(stuff)
    # 记录扁平化后的元素个数
    num_elts = len(flat_stuff)
    # 找出列表中所有张量的索引
    tensor_idxs = [idx for idx, thing in enumerate(flat_stuff)
                   if isinstance(thing, torch.Tensor)]
    # 找出列表中所有非张量数据的索引
    non_tensor_idxs = [idx for idx, thing in enumerate(flat_stuff)
                       if not isinstance(thing, torch.Tensor)]
    # 从扁平化后的列表中提取所有张量
    tensors = [thing for thing in flat_stuff if isinstance(thing, torch.Tensor)]
    # 从扁平化后的列表中提取所有非张量数据
    non_tensors = [thing for thing in flat_stuff if not isinstance(thing, torch.Tensor)]

    # 将结构说明和元素个数保存到 ctx 对象中
    ctx.spec = spec
    ctx.num_elts = num_elts
    # 将张量保存到 ctx 对象中的 saved_tensors 中
    ctx.save_for_backward(*tensors)
    # 记录张量在 flat_stuff 中的索引
    ctx.tensor_idxs = tensor_idxs
    # 保存非张量数据到 ctx 对象中 saved_non_tensors 中
    ctx.saved_non_tensors = non_tensors
    # 记录非张量数据在 flat_stuff 中的索引
    ctx.non_tensor_idxs = non_tensor_idxs


# save_pytree_for_backward 的反操作
def unpack_saved(ctx):
    # 创建一个与 flat_stuff 元素个数相同的 None 列表
    flat_stuff = [None] * ctx.num_elts
    # 将保存在 ctx.saved_tensors 中的张量恢复到 flat_stuff 中对应的位置
    for tensor, idx in zip(ctx.saved_tensors, ctx.tensor_idxs):
        flat_stuff[idx] = tensor
    # 将保存在 ctx.saved_non_tensors 中的非张量数据恢复到 flat_stuff 中对应的位置
    for non_tensor, idx in zip(ctx.saved_non_tensors, ctx.non_tensor_idxs):
        flat_stuff[idx] = non_tensor
    # 使用结构说明 spec 将 flat_stuff 还原成原始的 pytree 结构 stuff
    stuff = pytree.tree_unflatten(flat_stuff, ctx.spec)
    # 返回恢复后的 pytree
    return stuff
```