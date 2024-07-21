# `.\pytorch\torch\_functorch\autograd_function.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型声明和模块

from typing import Any, NamedTuple, Tuple
# 导入类型声明和命名元组模块

import torch
# 导入 PyTorch 库

import torch.utils._pytree as pytree
# 导入 PyTorch 的 _pytree 模块

from torch._C._functorch import (
    _unwrap_for_grad,
    _wrap_for_grad,
    current_level,
    TransformType,
)
# 从 torch._C._functorch 模块导入必要的函数和类

from torch._functorch.apis import vmap
# 导入 functorch 的 vmap API

from torch._functorch.utils import enable_single_level_autograd_function
# 导入启用单层自动求导函数的工具函数

from torch._functorch.vmap import (
    _add_batch_dim,
    _broadcast_to_and_flatten,
    restore_vmap,
    unwrap_batched,
    wrap_batched,
)
# 导入与 vmap 相关的函数和类

from torch._ops import HigherOrderOperator
# 导入 HigherOrderOperator 类

from torch.autograd.forward_ad import _set_fwd_grad_enabled
# 导入用于设置前向自动求导的函数

# autograd.Function 在常规 PyTorch 调度程序之前运行。
# 这是 autocast 和 torch_dispatch（例如 PythonTLSSnapshot）等功能与其协同工作的方式。
# 我们可能有一天会决定更改这一点，但在那之前，
# 我们需要保持 autograd.Function 在这些功能之前运行的假象。
#
# 我们通过创建一个自定义的 HigherOrderOperator 来实现这一点，
# 该操作符只在 functorch 中特殊地分发。
class CustomFunctionHigherOrderOperator(HigherOrderOperator):
    def __init__(self):
        super().__init__("custom_function_call")

    def __call__(self, autograd_function, *args, **kwargs):
        # 当 custom_function_call 通过 functorch 进行分发时，
        # 它应该只调用 autograd.Function。这与 autograd.Function
        # 在 PyTorch 调度程序之前被调用的行为是一致的。
        #
        # 这将在后续引起麻烦，但这是现有问题。有一个不变量是，
        # 由 make_fx 跟踪的函数应在提供相同 Tensor 时具有相同的行为。
        # 然而，make_fx 将 autograd.Function 视为一个复合体
        # （因为 autograd.Function 发生在 Python 调度键之前），
        # 并且只跟踪前向传递。
        if torch._C._are_functorch_transforms_active():
            return super().__call__(autograd_function, *args, **kwargs)
        return autograd_function.apply(*args, **kwargs)


# "custom_function_call"
# 这是与 functorch 转换一起工作的 autograd.Function 的机制。
# 它包装了一个 autograd.Function；与 functorch 转换的交互通过
# PyDispatcher 和 HigherOrderOperator 定义，而不是通过传统的 PyTorch
# 调度程序。
custom_function_call = CustomFunctionHigherOrderOperator()


# custom_function_call 的梯度规则是构造一个新的 _SingleLevelFunction
# （只与 functorch 的单层（level）一起工作的 autograd.Function），它：
# - 解封装输入
# - 重新分发到 custom_function_call
# - 封装输出
# 其反向传递调用原始 autograd.Function 的反向传递。
#
# 为什么我们需要重新分发到 custom_function_call？
# -----------------------------------------------------
# 这与 ATen 操作符如何与 functorch 的梯度变换一致：
# 定义一个自定义函数调用的梯度实现，该函数被应用于转换类型 Grad 和 Jvp
@custom_function_call.py_impl(TransformType.Grad)
@custom_function_call.py_impl(TransformType.Jvp)
def custom_function_call_grad(interpreter, autograd_function, *operands):
    # 生成单层函数对象，并将解释器和自动求导函数传入
    Generated = generate_single_level_function(interpreter, autograd_function)
    # 使用 enable_single_level_autograd_function 上下文管理器，执行生成的单层函数对象的应用
    with enable_single_level_autograd_function():
        # 应用生成的单层函数对象到操作数上
        flat_out = Generated.apply(*operands)
    # 返回应用后的结果
    return flat_out


def generate_single_level_function(interpreter, autograd_function):
    # 获取解释器的层级
    level = interpreter.level()

    # 定义前向传播函数，处理操作数
    def forward(*operands):
        # 使用 tree_map_only 将操作数映射为 torch.Tensor 类型，并进行 unwrap 操作
        unwrapped_operands = pytree.tree_map_only(
            torch.Tensor, lambda x: _unwrap_for_grad(x, level), operands
        )
        # 打开梯度计算，并启用前向梯度计算，同时使用解释器降级
        with torch.enable_grad(), _set_fwd_grad_enabled(True), interpreter.lower():
            # 执行自定义函数调用，将自动求导函数和解包后的操作数传入
            unwrapped_output = custom_function_call(
                autograd_function, *unwrapped_operands
            )

        # 定义包装函数，用于包装输出
        def wrap_fn(output):
            return _wrap_for_grad(output, level)

        # 保持输出的标识性包装并返回
        return wrap_outputs_maintaining_identity(
            unwrapped_output, unwrapped_operands, operands, wrap_fn
        )

    # 设置上下文的函数，将上下文、输入和输出传入自动求导函数的设置上下文函数
    def setup_context(ctx, inputs, output):
        return autograd_function.setup_context(ctx, inputs, output)

    # 只有当转换类型为 Grad 时才会使用反向传播函数
    def backward(ctx, *grads):
        result = autograd_function.backward(ctx, *grads)
        return result

    # 只有当转换类型为 Jvp 时才会使用 Jvp 函数
    def jvp(ctx, *tangents):
        result = autograd_function.jvp(ctx, *tangents)
        return result

    # 动态生成一个子类名称，用于为 Tensor 的 .grad_fn 字段生成类名
    name = f"{autograd_function.__name__}Generated"
    # 使用 type 动态生成一个子类，继承自 torch.autograd.function._SingleLevelFunction
    Generated = type(
        name,
        (torch.autograd.function._SingleLevelFunction,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
            "jvp": staticmethod(jvp),
            "setup_context": staticmethod(setup_context),
        },
    )
    # 返回生成的子类对象
    return Generated
# wrap_outputs_maintaining_identity函数处理来自vmap、backward（vjp）和jvp静态方法的输出。
# 它区分vmap情况和{backward, jvp}情况的方法是通过是否指定out_dims。
#
# NB: 我们不能仅仅使用out_dims=None作为决定因素。这是因为在vmap静态方法中仍然可能出现out_dims=None！
# 在这种情况下，用户的意思是他们的输出没有被vmap覆盖的维度，这是有效的。
NO_OUT_DIMS = "not specified"


# NOTE [mark_dirty object identity check]
# autograd.Function的ctx.mark_dirty期望返回的输入具有与输入相同的对象标识。
# 仅模式下的functorch将极大地简化这一逻辑。
def wrap_outputs_maintaining_identity(
    outputs, unwrapped_inputs, orig_inputs, wrap_fn, out_dims=NO_OUT_DIMS
):
    # 将输入展开为扁平的列表
    flat_unwrapped_inputs = pytree.arg_tree_leaves(*unwrapped_inputs)
    flat_orig_inputs = pytree.arg_tree_leaves(*orig_inputs)

    # 创建从展开输入到原始输入的映射字典
    unwrapped_input_to_orig_input = {
        id(unwrapped): orig
        for unwrapped, orig in zip(flat_unwrapped_inputs, flat_orig_inputs)
    }

    # 将输出展开为扁平的列表，并获取其结构描述
    flat_outputs, spec = pytree.tree_flatten(outputs)
    result = []

    # 检查是否指定了out_dims
    out_dims_specified = out_dims != NO_OUT_DIMS

    if out_dims_specified:
        # 如果指定了out_dims，则将其广播并展开为扁平的列表
        flat_out_dims = _broadcast_to_and_flatten(out_dims, spec)
        # _broadcast_to_and_flatten如果无法广播，则返回None
        if flat_out_dims is None:
            # 如果无法广播，则抛出运行时错误
            raise RuntimeError(
                f"The autograd.Function's vmap staticmethod returned an "
                f"incompatible (output, out_dims) tuple. "
                f"Expected out_dims={out_dims} "
                f"to be compatible with the structure of `output`. "
                f"out_dims has structure {pytree.tree_flatten(out_dims)[1]} "
                f"but output has structure {spec}. "
                f"For more details, please see "
                f"https://pytorch.org/docs/main/notes/extending.func.html"
            )

    # 遍历扁平化的输出列表
    for i, output in enumerate(flat_outputs):
        if not isinstance(output, torch.Tensor):
            # 如果输出不是Tensor，则直接添加到结果中
            result.append(output)
            continue
        if id(output) in unwrapped_input_to_orig_input:
            # 如果输出的对象标识在映射字典中，则使用映射关系替换输出
            result.append(unwrapped_input_to_orig_input[id(output)])
            continue
        if out_dims_specified:
            # 如果指定了out_dims，则使用wrap_fn包装输出和对应的out_dims
            result.append(wrap_fn(output, flat_out_dims[i]))  # type: ignore[possibly-undefined, index]
        else:
            # 否则，只使用wrap_fn包装输出
            result.append(wrap_fn(output))

    # 将结果列表根据结构描述恢复为原始输出结构
    return pytree.tree_unflatten(result, spec)


# NOTE: [functorch vjp and autograd interaction]
# functorch vjp和autograd交互存在一个边缘情况，将最终通过仅模式的functorch修复。
# 简而言之，无法取消包装死亡的GradTensorWrapper，因此我们（框架）需要手动执行。
# 正常的PyTorch操作符会自动处理这一点，以保持一致性。
#
# class MyExp(torch.autograd.Function):
#     @staticmethod
#     def forward(x):
#         return x.exp()
#
#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         y = output
#         ctx.save_for_backward(y)
#
#     @staticmethod
#     def backward(gy):
#         y, = ctx.saved_tensors()
#         return MyMul.apply(gy, y)
#
# x = torch.randn([], requires_grad=True)
# gy = torch.randn([], requires_grad=True)
# _, vjp_fn = vjp(MySin.apply, x)
# result = vjp_fn(gy)
#
# MyMul is an autograd.Function that is not shown here.
# It saves a `y` for backward (since gy requires grad).
#
# in vjp_fn(gy), we get:
# > MyMul.apply(gy, GradTensorWrapper(y, level=dead))
# Because the y that is saved for backward by MyExp is a GradTensorWrapper
# but is now dead since we are outside the vjp context.
#
# PyTorch dispatcher operations, upon seeing a dead GradTensorWrapper,
# will automatically unwrap the GradTensorWrapper when applied.
# But since autograd.Function technically sits above the regular PyTorch
# dispatcher, it doesn't get this treatment. So we manually do
# the unwrapping to be consistent with regular PyTorch dispatcher operations.


class VmapInfo(NamedTuple):
    batch_size: int
    randomness: str
    # NamedTuple defining VmapInfo with batch_size and randomness fields.


def has_overriden_vmap_rule(autograd_function):
    return autograd_function.vmap is not torch.autograd.Function.vmap
    # Check if autograd_function has overridden the vmap staticmethod.


def validate_vmap_returns_tuple_of_two_elements(result):
    base_error_msg = (
        "Expected the vmap staticmethod to have two returns, an output "
        "and out_dims with pytree structure compatible with the output. "
    )
    if not isinstance(result, tuple):
        raise RuntimeError(base_error_msg + f"Got a {type(result)} instead")
    if not len(result) == 2:
        raise RuntimeError(base_error_msg + f"Got {len(result)} returns instead")
    # Validate that the result tuple from vmap staticmethod has exactly two elements.


@custom_function_call.py_impl(TransformType.Vmap)
def custom_function_call_vmap(interpreter, autograd_function, *operands):
    if autograd_function.generate_vmap_rule:
        if has_overriden_vmap_rule(autograd_function):
            # TODO: Update link to stable once that's out
            # https://github.com/pytorch/pytorch/issues/92029
            raise RuntimeError(
                f"You tried to vmap over {autograd_function.__name__}, but "
                f"it has both generate_vmap_rule=True and an overriden vmap "
                f"staticmethod. Please set generate_vmap_rule=False or delete "
                f"the overriden vmap staticmethod to avoid ambiguity. "
                f"For more details, please see "
                f"https://pytorch.org/docs/main/notes/extending.func.html"
            )
        return custom_function_call_vmap_generate_rule(
            interpreter, autograd_function, *operands
        )
    # Implementation of a custom function call handling vmap semantics.
    # 检查是否已经覆盖了 autograd_function 的 vmap 规则
    if not has_overriden_vmap_rule(autograd_function):
        # 如果没有覆盖，则抛出运行时错误，提示用户需要实现 vmap 静态方法或设置 generate_vmap_rule=True
        # 提供详细信息和链接以便用户查阅
        raise RuntimeError(
            f"You tried to vmap over {autograd_function.__name__}, but "
            f"it does not have vmap support. Please override and implement the "
            f"vmap staticmethod or set generate_vmap_rule=True. "
            f"For more details, please see "
            f"https://pytorch.org/docs/main/notes/extending.func.html"
        )

    # 获取当前解释器的级别和信息
    current_level = interpreter.level()
    info = VmapInfo(
        batch_size=interpreter.batch_size(),
        randomness=interpreter.randomness(),
    )

    # 解包 batched tensors 和对应的维度信息
    unwrapped_operands, in_dims = unwrap_batched(operands, current_level)

    # 如果当前级别没有任何张量被批处理，则跳过当前级别
    # 这样做是为了避免用户在他们的 vmap 静态方法中处理这种情况，并与我们的 C++ 批处理规则 API 保持一致
    if pytree.tree_all(lambda dim: dim is None, in_dims):
        with interpreter.lower():
            # 在解释器下降语境中调用自定义函数
            return custom_function_call(autograd_function, *operands)

    # 在解释器下降语境中调用 autograd_function 的 vmap 方法，传入信息和维度信息
    with interpreter.lower():
        result = autograd_function.vmap(info, in_dims, *unwrapped_operands)

    # 验证 vmap 返回的结果是否是包含两个元素的元组
    validate_vmap_returns_tuple_of_two_elements(result)
    unwrapped_output, out_dims = result

    # 定义包装函数，根据输出维度添加批处理维度
    def wrap_fn(output, out_dim):
        return (
            output
            if out_dim is None
            else _add_batch_dim(output, out_dim, current_level)
        )

    # 使用维护对象标识的包装函数，返回包装后的输出
    return wrap_outputs_maintaining_identity(
        unwrapped_output, unwrapped_operands, operands, wrap_fn, out_dims=out_dims
    )
def vmapify_autograd_function(autograd_function, in_dims, batch_size, randomness):
    # 定义初始值，用于后续保存的变量
    init_val = "not populated"
    # 初始化输出维度、输入形状和保存的张量维度（类型为任意）
    out_dims = init_val
    input_shapes: Any = init_val
    saved_tensors_bdims: Any = init_val

    # 定义前向传播函数
    def forward(*operands):
        nonlocal out_dims  # 使用非本地变量out_dims
        # 调用restore_vmap函数恢复输出和输出维度
        outputs, out_dims = restore_vmap(
            autograd_function.forward, in_dims, batch_size, randomness
        )(*operands)
        return outputs
    def setup_context(ctx, inputs, outputs):
        input_shapes_ = None  # 初始化 input_shapes_ 变量为 None
        saved_tensors_bdims_ = None  # 初始化 saved_tensors_bdims_ 变量为 None

        def inner(inputs, outputs):
            # wrapped_ctx.save_for_backward will:
            # - unwrap batchedtensors into (tensor, bdim)
            # - save_for_backward(*unwrapped_tensors)
            # - assign the bdims to wrapped_ctx._pt_saved_tensors_bdims
            # 使用 wrapped_ctx.save_for_backward 方法：
            # - 将批量张量拆分为 (张量, bdim)
            # - 调用 save_for_backward 方法保存拆分后的张量
            # - 将 bdims 赋值给 wrapped_ctx._pt_saved_tensors_bdims
            wrapped_ctx = CtxCustomSave(ctx, current_level())
            autograd_function.setup_context(wrapped_ctx, inputs, outputs)

            # input_shapes are used for reductify later to reduce expanded gradients
            # to the correct shape.
            # See NOTE: [Why can't we rely on autograd to reduce expanded gradients?]
            # for more details
            # input_shapes 用于稍后的 reductify 操作，将扩展梯度减少到正确的形状。
            # 详见 NOTE: [Why can't we rely on autograd to reduce expanded gradients?] 
            nonlocal input_shapes_
            input_shapes_ = tuple(
                inp.shape if isinstance(inp, torch.Tensor) else None for inp in inputs
            )
            nonlocal saved_tensors_bdims_
            saved_tensors_bdims_ = wrapped_ctx._pt_saved_tensors_bdims

        # See NOTE: [Why do we need to run setup_context under a vmap?]
        # 查看 NOTE: [Why do we need to run setup_context under a vmap?] 注释
        restore_vmap(
            inner,
            (in_dims, out_dims),
            batch_size,
            randomness,
        )(inputs, outputs)

        nonlocal input_shapes  # 设置 input_shapes 变量为函数内 nonlocal 变量 input_shapes_
        input_shapes = input_shapes_
        nonlocal saved_tensors_bdims  # 设置 saved_tensors_bdims 变量为函数内 nonlocal 变量 saved_tensors_bdims_
        saved_tensors_bdims = saved_tensors_bdims_

    def jvp(ctx, *tangents):
        assert out_dims != init_val  # 断言确保 out_dims 不等于 init_val
        assert saved_tensors_bdims != init_val  # 断言确保 saved_tensors_bdims 不等于 init_val

        def jvp_no_context(saved_tensors, tangents):
            # 使用 saved_tensors 和 tangents 创建 wrapped_ctx 上下文
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.jvp(wrapped_ctx, *tangents)

        tangent_in_dims = get_tangents_in_dims(in_dims, tangents)
        out_tangents, out_tangents_dims = restore_vmap(
            jvp_no_context,
            (saved_tensors_bdims, tangent_in_dims),
            batch_size,
            randomness,
        )(ctx.saved_tensors, tangents)

        result = reductify(out_tangents, out_tangents_dims, out_dims, batch_size)
        return result

    def backward(ctx, *grad_outputs):
        assert out_dims != init_val  # 断言确保 out_dims 不等于 init_val
        assert input_shapes != init_val  # 断言确保 input_shapes 不等于 init_val
        assert saved_tensors_bdims != init_val  # 断言确保 saved_tensors_bdims 不等于 init_val

        def backward_no_context(inputs):
            saved_tensors, grad_outputs = inputs
            # 使用 saved_tensors 创建 wrapped_ctx 上下文
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.backward(wrapped_ctx, *grad_outputs)

        grad_ins, grad_ins_dims = restore_vmap(
            backward_no_context,
            ((saved_tensors_bdims, out_dims),),
            batch_size,
            randomness,
        )((ctx.saved_tensors, grad_outputs))
        result = reductify(grad_ins, grad_ins_dims, in_dims, batch_size, input_shapes)
        return result

    name = f"Vmapped{autograd_function.__name__}"
    # 根据指定的参数生成一个新的类对象 Generated
    Generated = type(
        name,  # 类的名称
        (torch.autograd.Function,),  # 继承自 torch.autograd.Function
        {
            "forward": staticmethod(forward),  # 定义静态方法 forward
            "backward": staticmethod(backward),  # 定义静态方法 backward
            "jvp": staticmethod(jvp),  # 定义静态方法 jvp
            "setup_context": staticmethod(setup_context),  # 定义静态方法 setup_context
            "generate_vmap_rule": True,  # 添加一个属性 generate_vmap_rule，值为 True
        },
    )
    
    def get_out_dims():
        assert out_dims != init_val  # 断言确保 out_dims 不等于 init_val
        return out_dims  # 返回 out_dims 变量的值
    
    return Generated, get_out_dims  # 返回 Generated 类对象和 get_out_dims 函数
# tangents might be None, so we need to replace
# the corresponding in_dims with None.
def get_tangents_in_dims(input_dims, tangents):
    # Flatten the input_dims using pytree.tree_flatten
    flat_in_dims, spec = pytree.tree_flatten(input_dims)
    # Extract leaves from the tangents using pytree.arg_tree_leaves
    flat_tangents = pytree.arg_tree_leaves(*tangents)
    # Replace in_dims with None where corresponding tangents are None
    result = [
        None if tangent is None else in_dim
        for in_dim, tangent in zip(flat_in_dims, flat_tangents)
    ]
    # Unflatten the result using the original spec
    return pytree.tree_unflatten(result, spec)


# NOTE: [Why do we need to run setup_context under a vmap?]
# Consider the following autograd.Function
#
# class Sum(torch.autograd.Function):
#    @staticmethod
#    def forward(x):
#        return x.sum()
#    @staticmethod
#    def setup_context(ctx, inputs, outputs):
#        ctx.x_shape = inputs[0]
#    @staticmethod
#    def backward(ctx, gy):
#        return gy.expand(ctx.x_shape)
#
# x = torch.randn(B, 4)
# in_dims = 0
# vmap(Sum.apply, in_dims)(x)
#
# Let's assume for a moment that we didn't vmap setup_context in VmappedSum:
#
# class VmappedSum(torch.autograd.Function):
#    @staticmethod
#    def forward(x):
#        return vmap(Sum.forward, in_dims)(x)
#
#    @staticmethod
#    def setup_context(ctx, inputs, outputs):
#        Sum.setup_context(ctx, inputs, outputs)
#
#    @staticmethod
#    def backward(ctx, gy):
#        def backward_no_context(gy):
#            return gy.expand(ctx.x_shape)
#
#        dims = (0,)
#        gx = vmap(backward_no_context, dims)(gy)
#        return gx
#
# We end up saving [B, 4] as x_shape. In the backward, gy has shape [B],
# and we're doing:
#
# def backward_no_context(gy):
#     return gy.expand([B, 4])
#
# gx = vmap(backward_no_context, dims)(gy: "Tensor[B]")
#
# This gives us the wrong result (gx has shape [B, B, 4], but it should
# have shape [4]). Performing vmap over setup_context means the shape
# saved has shape [4] and leads to a correct result shape for gx.


# Wraps a ctx object. Forwards all attr accesses to the underlying object
# except for the attrs in _pt_attrs
class WrappedCtx:
    _pt_reserved_attrs: Tuple[str, ...] = ("_pt_reserved_attrs", "_pt_inner_ctx")

    def __init__(self, ctx):
        # Ensure ctx is not an instance of WrappedCtx to avoid name collisions
        if not isinstance(ctx, WrappedCtx):
            reserved_attrs = type(self)._pt_reserved_attrs
            # Check for reserved attribute names to prevent collision
            for name in reserved_attrs:
                if not hasattr(ctx, name):
                    continue
                # Raise an error if reserved attribute name is found
                raise RuntimeError(
                    f"PyTorch reserves the {reserved_attrs} field on ctx. "
                    "Please name your fields on ctx something else to avoid name "
                    "collision."
                )
        # Initialize with the inner ctx object
        self._pt_inner_ctx = ctx

    def __getattr__(self, name):
        # Forward attribute access to the inner ctx object
        return getattr(self._pt_inner_ctx, name)

    def __setattr__(self, name, value):
        # Allow setting attributes directly except for reserved ones
        if name in type(self)._pt_reserved_attrs:
            self.__dict__[name] = value
            return
        # Forward attribute assignment to the inner ctx object
        return setattr(self._pt_inner_ctx, name, value)


# Wraps ctx to create a new ctx object that overrides saved_tensors.
class CtxWithSavedTensors(WrappedCtx):
    # 定义保留的属性列表，包括 "_pt_new_saved_tensors" 和 WrappedCtx 类的保留属性
    _pt_reserved_attrs = ("_pt_new_saved_tensors", *WrappedCtx._pt_reserved_attrs)
    
    # 初始化方法，接受 ctx 和 new_saved_tensors 两个参数
    def __init__(self, ctx, new_saved_tensors):
        # 调用父类的初始化方法
        super().__init__(ctx)
        # 设置当前对象的 _pt_new_saved_tensors 属性为传入的 new_saved_tensors
        self._pt_new_saved_tensors = new_saved_tensors
    
    # saved_tensors 属性的装饰器方法，返回当前对象的 _pt_new_saved_tensors 属性
    @property
    def saved_tensors(self):
        return self._pt_new_saved_tensors
# 定义一个自定义的上下文保存类，继承自 WrappedCtx
class CtxCustomSave(WrappedCtx):
    # 保留的私有属性列表，包括父类 WrappedCtx 的保留属性
    _pt_reserved_attrs = (
        "_pt_saved_tensors_bdims",
        "_pt_current_level",
        *WrappedCtx._pt_reserved_attrs,
    )

    # 初始化方法，接受一个上下文 ctx 和当前级别 current_level 参数
    def __init__(self, ctx, current_level):
        super().__init__(ctx)  # 调用父类的初始化方法
        self._pt_saved_tensors_bdims = ()  # 初始化保存张量维度信息的属性为空元组
        self._pt_current_level = current_level  # 初始化当前级别属性为传入的参数 current_level

    # 保存反向传播需要的张量信息
    def save_for_backward(self, *tensors):
        # 调用 unwrap_batched 函数，处理传入的张量和当前级别信息，返回未包装的张量和张量维度信息
        unwrapped_tensors, bdims = unwrap_batched(tensors, self._pt_current_level)
        # 调用内部上下文对象的 save_for_backward 方法，保存未包装的张量
        self._pt_inner_ctx.save_for_backward(*unwrapped_tensors)
        self._pt_saved_tensors_bdims = bdims  # 更新保存的张量维度信息

    # 保存前向传播需要的张量信息
    def save_for_forward(self, *tensors):
        # 调用 unwrap_batched 函数，处理传入的张量和当前级别信息，返回未包装的张量和张量维度信息
        unwrapped_tensors, bdims = unwrap_batched(tensors, self._pt_current_level)
        # 调用内部上下文对象的 save_for_forward 方法，保存未包装的张量
        self._pt_inner_ctx.save_for_forward(*unwrapped_tensors)
        self._pt_saved_tensors_bdims = bdims  # 更新保存的张量维度信息


# 对梯度张量进行降维处理的函数
def reductify(
    grad_input,
    grad_input_bdim,
    input_bdim,
    batch_size,
    target_shape_without_bdim_to_reduce_to=None,
):
    if not isinstance(grad_input, tuple):  # 如果 grad_input 不是元组
        grad_input = (grad_input,)  # 将其转换为单元素元组
    if not isinstance(grad_input_bdim, tuple):  # 如果 grad_input_bdim 不是元组
        grad_input_bdim = (grad_input_bdim,)  # 将其转换为单元素元组
    if not isinstance(input_bdim, tuple):  # 如果 input_bdim 不是元组
        input_bdim = (input_bdim,)  # 将其转换为单元素元组

    if target_shape_without_bdim_to_reduce_to is None:
        # 如果目标形状未指定，将其设置为与 grad_input 元素个数相同的 None 组成的元组
        target_shape_without_bdim_to_reduce_to = len(grad_input) * (None,)
    
    # 使用生成器表达式对 grad_input, grad_input_bdim, input_bdim 和目标形状进行降维 leaf 操作
    result = tuple(
        reductify_leaf(gi, gi_bdim, i_bdim, batch_size, maybe_ishape)
        for gi, gi_bdim, i_bdim, maybe_ishape in zip(
            grad_input,
            grad_input_bdim,
            input_bdim,
            target_shape_without_bdim_to_reduce_to,
        )
    )
    return result  # 返回降维处理后的结果元组


# 对单个梯度张量进行 leaf 级别的降维处理的函数
def reductify_leaf(
    grad_input,
    grad_input_bdim,
    input_bdim,
    batch_size,
    target_shape_without_bdim_to_reduce_to=None,
):
    if grad_input is None:  # 如果 grad_input 为 None
        return None  # 直接返回 None

    if grad_input_bdim is None and input_bdim is None:
        return grad_input  # 如果 grad_input_bdim 和 input_bdim 均为 None，则直接返回 grad_input

    # NOTE: [Why can't we rely on autograd to reduce expanded gradients?]
    # 对于反向自动微分，
    # 当给定 grad_input 和 input 时，用户可能返回一个相对于 input 广播形状的 grad_input 是合法的。
    # 在这种情况下，autograd 会自动将 grad_input 降维到 input 的形状。
    #
    # 然而，当 input_bdim 不为 None 时，我们会遇到问题。
    #
    # [例子 1]
    # grad_input: Tensor[3, 4], input: Tensor[B, 4]
    # 我们可以将 grad_input 扩展为 Tensor[B, 3, 4]，但这不是从 [B, 4] 广播的。
    #
    # [例子 2]
    # grad_input: Tensor[3, B, 4], input: Tensor[B, 4]
    # 我们可以将 grad_input 调整为 Tensor[B, 3, 4]，但这也不是从 [B, 4] 广播的。
    #
    # 这意味着我们需要将 grad_input 降维到 input 的形状。这种行为由 `target_shape_without_bdim_to_reduce_to` 标志控制；
    # 如果 input_bdim 不为 None，则进行手动降维操作；否则不进行降维操作。
    assert input_bdim is not None

    # 如果 grad_input_bdim 为 None，则扩展 grad_input 的维度并调整形状以匹配指定的 batch_size
    if grad_input_bdim is None:
        grad_input = grad_input.unsqueeze(input_bdim)  # 在 input_bdim 维度上添加一个维度
        new_shape = list(grad_input.shape)
        new_shape[input_bdim] = batch_size  # 将指定维度的大小设置为 batch_size
        grad_input = grad_input.expand(new_shape)  # 扩展 grad_input 的形状
        grad_input_bdim = input_bdim  # 更新 grad_input_bdim 的值为 input_bdim

    # 如果指定了 target_shape_without_bdim_to_reduce_to，则使用 vmap 函数对 grad_input 进行尺寸调整
    if target_shape_without_bdim_to_reduce_to is not None:
        return vmap(
            torch.Tensor.sum_to_size,
            in_dims=(grad_input_bdim, None),  # 指定输入的维度映射
            out_dims=input_bdim,  # 指定输出的维度
        )(grad_input, target_shape_without_bdim_to_reduce_to)

    # 如果 input_bdim 和 grad_input_bdim 不相等，则通过 movedim 函数调整 grad_input 的维度
    if input_bdim != grad_input_bdim:
        grad_input = grad_input.movedim(grad_input_bdim, input_bdim)

    # 返回调整后的 grad_input
    return grad_input
def autograd_function_forward_rewritten(original_forward, original_setup_context):
    # 定义一个新的前向传播函数，接受上下文 ctx、任意参数 *args 和 **kwargs
    def new_forward(ctx, *args, **kwargs):
        # 调用原始的前向传播函数 original_forward
        output = original_forward(*args, **kwargs)
        # 调用原始的上下文设置函数 original_setup_context，传入 ctx、args 和 output
        original_setup_context(ctx, args, output)
        # 返回前向传播函数的输出
        return output

    # 返回新定义的前向传播函数
    return new_forward


class AutogradFunctionApply(HigherOrderOperator):
    def __init__(self):
        # 调用父类的初始化方法，设置实例的名称为 "autograd_function_apply"
        super().__init__("autograd_function_apply")

    def __call__(self, fwd, bwd, *fwd_args, **fwd_kwargs):
        # 获取参数 fwd_args 中的 "args_tensor_mask" 键对应的值
        args_tensor_mask = fwd_kwargs["args_tensor_mask"]
        # 计算 args_tensor_mask 中值为 True 的个数，表示张量参数的数量
        length_of_tensor_args = sum(args_tensor_mask)
        # 从 fwd_args 中过滤出原始张量参数，lifted freevars 不是 ApplyTemplate.apply 的参数，
        # 因为我们不需要计算它们的梯度。
        new_fwd_args = fwd_args[:length_of_tensor_args]

        # 定义一个内部类 ApplyTemplate，继承自 torch.autograd.Function
        class ApplyTemplate(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                # 声明 nonlocal 变量 saved_values
                nonlocal saved_values
                # 调用 fwd 函数进行前向传播计算，传入 None、*fwd_args，并接收输出和 saved_values
                output, saved_values = fwd(None, *fwd_args)
                # 返回前向传播的输出
                return output

            @staticmethod
            def backward(ctx, *grad):
                # 调用 bwd 函数进行反向传播计算，传入 None、*grad 和 *saved_values，并返回结果
                return bwd(None, *grad, *saved_values)

        # 调用 ApplyTemplate 类的 apply 方法，传入过滤后的新前向传播参数 new_fwd_args
        return ApplyTemplate.apply(*new_fwd_args)


# 创建 AutogradFunctionApply 的实例 autograd_function_apply
autograd_function_apply = AutogradFunctionApply()
```