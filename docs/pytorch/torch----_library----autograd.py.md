# `.\pytorch\torch\_library\autograd.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和类
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

# 从相关模块中导入需要的对象
from .. import _C, _ops, autograd, Tensor

# 导入工具函数
from ..utils import _pytree
from . import utils

# 定义一个协议，规定了 Info 类的属性类型
class InfoProtocol(Protocol):
    _backward_fn: Optional[Callable]  # 可选的回传函数
    _setup_context_fn: Optional[Callable]  # 可选的上下文设置函数

# 使用 dataclasses 装饰器，定义一个数据类 Info，包含两个可选的回调函数属性
@dataclasses.dataclass
class Info:
    _backward_fn: Optional[Callable]  # 可选的回传函数
    _setup_context_fn: Optional[Callable]  # 可选的上下文设置函数

# 定义一个函数 make_autograd_impl，接受一个操作对象和一个 InfoProtocol 对象作为参数，返回一个可调用对象
def make_autograd_impl(op: _ops.OpOverload, info: InfoProtocol) -> Callable:
    # 根据操作的命名空间、操作名和重载名生成一个字符串作为名称
    name: str = f"GeneratedBackwardFor_{op._namespace}_{op._opname}_{op._overloadname}"

    # 判断操作的模式是否只接受关键字参数
    has_kwarg_only_args = utils.has_kwarg_only_args(op._schema)

    # 定义一个嵌套的数据类 Metadata，包含键集和关键字参数的字典
    @dataclass
    class Metadata:
        keyset: _C.DispatchKeySet  # 分发键集对象
        keyword_only_args: Dict[str, Any]  # 字典，包含关键字参数

    # 定义一个不带梯度的前向传播函数，接受任意数量的参数
    def forward_no_grad(*args):
        metadata = args[-1]  # 最后一个参数是元数据对象
        args = args[:-1]  # 去除最后一个元数据参数

        # 进入自动微分下的自动分派环境
        with _C._AutoDispatchBelowAutograd():
            keyset = metadata.keyset  # 获取键集
            kwargs = metadata.keyword_only_args  # 获取关键字参数
            # 使用重派函数执行操作，传递参数和关键字参数
            result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
            return result

    # 定义一个带上下文的前向传播函数，接受任意数量的参数
    def forward(ctx, *args):
        metadata = args[-1]  # 最后一个参数是元数据对象
        args = args[:-1]  # 去除最后一个元数据参数

        # 进入自动微分下的自动分派环境
        with _C._AutoDispatchBelowAutograd():
            keyset = metadata.keyset  # 获取键集
            kwargs = metadata.keyword_only_args  # 获取关键字参数
            # 使用重派函数执行操作，传递参数和关键字参数
            result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
            
            # 如果存在设置上下文的函数
            if info._setup_context_fn:
                # 使用工具函数填充参数的默认值，保证用户可以访问它们
                args, kwargs = utils.fill_defaults(op._schema, args, kwargs)
                
                # 如果操作有关键字参数
                if has_kwarg_only_args:
                    # 调用设置上下文函数，传递上下文、输入参数、关键字参数输入和输出结果
                    info._setup_context_fn(
                        ctx=ctx, inputs=args, keyword_only_inputs=kwargs, output=result
                    )
                else:
                    # 调用设置上下文函数，传递上下文、输入参数和输出结果
                    info._setup_context_fn(ctx=ctx, inputs=args, output=result)
            return result
    # 定义一个函数 `backward`，用于执行反向传播计算梯度
    def backward(ctx, *grads):
        # 如果有注册的反向传播函数 `_backward_fn`
        if info._backward_fn:
            try:
                # 保存当前的输入梯度需求情况
                prev_needs_input_grad = ctx.needs_input_grad
                # 更新上下文中的输入梯度需求，去掉最后一个
                ctx.needs_input_grad = ctx.needs_input_grad[:-1]
                # 调用注册的反向传播函数 `_backward_fn` 进行计算
                result = info._backward_fn(ctx, *grads)
            finally:
                # 恢复之前保存的输入梯度需求情况
                ctx.needs_input_grad = prev_needs_input_grad
            # 如果计算结果是一个元组，则添加一个空值作为返回结果的一部分
            if isinstance(result, tuple):
                return (*result, None)
            # 否则只返回计算结果和一个空值
            return result, None
        # 如果没有注册的反向传播函数，则抛出运行时错误
        raise RuntimeError(
            f"Trying to backward through {op} but no autograd "
            f"formula was registered. "
            f"Please use register_autograd to add one."
        )

    # 使用 type() 函数动态生成一个新的类 Generated，继承自 autograd.Function 类
    Generated = type(
        name,
        (autograd.Function,),
        {
            "forward": staticmethod(forward),    # 设置 forward 方法为静态方法
            "backward": staticmethod(backward),  # 设置 backward 方法为静态方法
        },
    )

    # 获取操作符 op 的 schema
    schema = op._schema
    # 如果 schema 中的参数或返回值中有任何类似于 tensorlist 的类型
    if any(
        utils.is_tensorlist_like_type(a.type)
        for a in (*schema.arguments, *schema.returns)
    ):
        # 将 Generated 类应用 tensorlist 支持的装饰器
        Generated = supports_tensorlist(Generated)

    # 定义一个函数 `autograd_impl`，用于实现自动求导逻辑
    def autograd_impl(keyset, *args, **keyword_only_args):
        # 如果允许梯度计算且参数中有 Tensor 类型的参数需要求梯度
        if _C.is_grad_enabled() and _pytree.tree_any_only(
            Tensor, lambda x: x.requires_grad, args, not_list_of_tensor
        ):
            # 调用 Generated 类的 apply 方法进行前向传播计算
            result = Generated.apply(*args, Metadata(keyset, keyword_only_args))  # type: ignore[attr-defined]
        else:
            # 否则调用无梯度的前向传播函数进行计算
            result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
        # 返回计算结果
        return result

    # 返回自动求导实现函数 autograd_impl
    return autograd_impl
def supports_tensorlist(cls: Any) -> Any:
    """允许给定的 autograd.Function 类支持 List[Tensor] 输入/输出。

    Regular autograd.Function 有一个限制，它只直接支持 Tensor。应用 @supports_tensorlist
    使得 autograd.Function 能够支持 List[Tensor] 的输入和输出。
    """
    # 保存原始的 forward、backward 和 apply 方法
    orig_forward = cls.forward
    orig_backward = cls.backward
    orig_apply = cls.apply

    @dataclass
    class Metadata:
        input_spec: spec_t
        output_spec: Optional[spec_t] = None
        result_is_tuple: Optional[bool] = None

    def new_forward(ctx, *args):
        # 提取最后一个参数作为 metadata
        metadata = args[-1]
        args = args[:-1]
        # 如果 metadata 不是 Metadata 类型，则抛出 NotImplementedError
        if not isinstance(metadata, Metadata):
            raise NotImplementedError(
                "NYI: calling supports_tensorlist autograd.Function.forward directly. "
                "You should probably be calling .apply instead. "
                "Please file an issue if not."
            )
        # 使用 unflatten 将 args 转换成指定的 input_spec 结构
        args = unflatten(list(args), metadata.input_spec)
        # 调用原始的 forward 方法
        result = orig_forward(ctx, *args)
        # 判断结果是否为 tuple，并更新 metadata
        metadata.result_is_tuple = isinstance(result, tuple)
        if not metadata.result_is_tuple:
            result = (result,)
        # 使用 flatten 函数将 result 扁平化，并获取 output_spec
        flat_result, output_spec = flatten(result, not_list_of_tensor)
        metadata.output_spec = output_spec

        # 如果 ctx 已经有 _pt_metadata 属性，则抛出 RuntimeError
        if hasattr(ctx, "_pt_metadata"):
            raise RuntimeError(
                "Please don't set ctx._pt_metadata; PyTorch uses it to store info"
            )
        # 将 metadata 存储在 ctx._pt_metadata 中
        ctx._pt_metadata = metadata

        return tuple(flat_result)
    def new_backward(ctx, *grads):
        # 检查上下文对象是否具有"_pt_metadata"属性，如果没有则抛出异常
        if not hasattr(ctx, "_pt_metadata"):
            raise NotImplementedError(
                "NYI: calling supports_tensorlist autograd.Function.backward directly. "
                "This will automatically get called by PyTorch autograd. "
                "Please file an issue if you need this."
            )

        # 获取上下文对象的元数据
        metadata = ctx._pt_metadata
        # 将梯度参数解压缩为符合输出规范的形式
        grads = unflatten(list(grads), metadata.output_spec)

        # 如果用户的输入是([x, y, z], w)，则needs_input_grad是(bool, bool, bool, bool, bool)
        # 我们需要：
        # 1. 去掉额外的bool（来自额外的metadata输入）
        # 2. 解压缩以获得正确的结构
        prev_needs_input_grad = ctx.needs_input_grad
        try:
            # 更新上下文对象的输入梯度需求，去掉最后一个bool
            ctx.needs_input_grad = unflatten(
                list(ctx.needs_input_grad[:-1]), metadata.input_spec
            )
            # 调用原始的反向传播函数计算梯度输入
            grad_inputs = orig_backward(ctx, *grads)
        finally:
            # 恢复上下文对象的输入梯度需求
            ctx.needs_input_grad = prev_needs_input_grad

        # 如果grad_inputs不是元组，则转换为元组
        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)
        
        # 假设反向传播中的任何None表示Tensor
        # 如果前向传播的参数是[1, 2, 3]，则反向传播应返回None作为梯度
        # 如果前向传播的参数是[tensor, tensor]，则反向传播可能返回[None, None]、[grad, None]、[None, grad]或[grad, grad]
        # 将梯度输入扁平化，并获取其规范
        flat_grad_inputs, grad_inputs_spec = flatten(
            grad_inputs, not_list_of_optional_tensor
        )
        
        # 检查反向传播返回的结果结构与输入规范是否相同
        if grad_inputs_spec != metadata.input_spec:
            raise RuntimeError(
                f"Expected the return from backward to be of the same structure "
                f"as the inputs. Got: {grad_inputs_spec} (return from backward), "
                f"{metadata.input_spec} (inputs)"
            )
        
        # 返回扁平化后的梯度输入，最后加上一个None
        return tuple(flat_grad_inputs + [None])

    def new_apply(*args):
        # 扁平化输入参数args，并获取其输入规范
        flat_args, input_spec = flatten(args, is_leaf=not_list_of_tensor)
        # 创建元数据对象
        metadata = Metadata(input_spec)
        # 调用原始的应用函数并传入扁平化后的参数和元数据，获取结果
        result = orig_apply(*flat_args, metadata)  # type: ignore[misc]
        # 确保元数据的输出规范不为空
        assert metadata.output_spec is not None
        # 将结果按照元数据的输出规范解压缩
        result = unflatten(list(result), metadata.output_spec)
        # 如果元数据的结果不是元组，则确保结果是单个值并返回
        if not metadata.result_is_tuple:
            assert isinstance(result, tuple)
            assert len(result) == 1
            return result[0]
        # 返回解压缩后的结果
        return result

    # 将自定义的新前向、反向和应用函数分配给类的相应方法
    cls.forward = new_forward
    cls.backward = new_backward
    cls.apply = new_apply
    # 返回修改后的类
    return cls
# 检查参数 `tree` 是否不是张量列表
def not_list_of_tensor(tree):
    # 如果 `tree` 是元组，则返回 False（假设元组中的每个元素都是张量）
    if isinstance(tree, tuple):
        return False
    # 如果 `tree` 是列表，则检查列表中是否存在非张量对象，返回 True 或 False
    if isinstance(tree, list):
        return any(not isinstance(l, Tensor) for l in tree)
    # 如果 `tree` 不是元组也不是列表，则默认返回 True
    return True


# 检查参数 `tree` 是否不是可选张量的列表
def not_list_of_optional_tensor(tree):
    # 如果 `tree` 是元组，则返回 False（假设元组中的每个元素都是可选张量）
    if isinstance(tree, tuple):
        return False
    # 如果 `tree` 是列表，则检查列表中是否存在非 None 且不是张量的对象，返回 True 或 False
    if isinstance(tree, list):
        return any(l is not None and not isinstance(l, Tensor) for l in tree)
    # 如果 `tree` 不是元组也不是列表，则默认返回 True
    return True


# 将 `_pytree` 模块中的 `tree_flatten` 函数赋值给变量 `flatten`
flatten = _pytree.tree_flatten
# 将 `_pytree` 模块中的 `tree_unflatten` 函数赋值给变量 `unflatten`
unflatten = _pytree.tree_unflatten
# 将 `_pytree` 模块中的 `TreeSpec` 类型赋值给变量 `spec_t`
spec_t = _pytree.TreeSpec
```