# `.\pytorch\torch\_higher_order_ops\effects.py`

```
# mypy: allow-untyped-defs
# 从 enum 模块导入 Enum 类
from enum import Enum
# 从 typing 模块导入必要的类型
from typing import Any, Dict, Optional, Tuple, Union

# 导入 torch 库
import torch
# 导入 torch.utils._pytree 模块，别名为 pytree
import torch.utils._pytree as pytree
# 从 torch._C 模块导入 DispatchKey 类
from torch._C import DispatchKey
# 从 torch._higher_order_ops.torchbind 模块导入 call_torchbind 函数
from torch._higher_order_ops.torchbind import call_torchbind
# 从 torch._ops 模块导入 HigherOrderOperator 类和 OpOverload 类
from torch._ops import HigherOrderOperator, OpOverload
# 从 torch._subclasses.fake_tensor 模块导入 FakeTensorMode 枚举
from torch._subclasses.fake_tensor import FakeTensorMode
# 从 torch.fx.experimental.proxy_tensor 模块导入相关函数和类
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

# 定义 _EffectType 枚举，包含一个枚举值 ORDERED
class _EffectType(Enum):
    ORDERED = "Ordered"

# 定义 OpType 别名，可以是 HigherOrderOperator 或 OpOverload 类型的对象
OpType = Union[torch._ops.HigherOrderOperator, torch._ops.OpOverload]

# 定义 SIDE_EFFECTS 字典，将操作类型映射到其副作用类型的枚举值
SIDE_EFFECTS: Dict[OpType, _EffectType] = {
    torch.ops.aten._print.default: _EffectType.ORDERED,
    call_torchbind: _EffectType.ORDERED,
}

# 定义函数 _register_effectful_op，用于注册具有副作用的操作类型
def _register_effectful_op(op: OpType, effect: _EffectType):
    assert isinstance(
        op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
    ) and not has_aliasing(op)
    # 如果已经注册了该操作类型并且副作用类型不同，则抛出 RuntimeError
    if op in SIDE_EFFECTS and SIDE_EFFECTS[op] != effect:
        raise RuntimeError(
            f"Already registered effect type {SIDE_EFFECTS[op]} to op {op}, "
            f"trying to register a different effect type {effect}."
        )
    # 将操作类型及其副作用类型注册到 SIDE_EFFECTS 字典中
    SIDE_EFFECTS[op] = effect

# 定义 WithEffects 类，继承自 HigherOrderOperator 类
class WithEffects(HigherOrderOperator):
    """
    with_effects(token, op, args, kwargs) -> (new_token, op_results)

    This HOP helps ensure ordering between side effectful ops like prints or ops
    using torchbind objects. This is needed to ensure a traced graph from
    AOTAutograd is functional so that future optimization passes do not reorder
    these operators. This is done through threading "effect tokens" through the
    graph to enforce data dependence between side effectful ops.

    The tokens are basically dummy values (torch.tensor([])). We create a token
    per "effect type", which are enumerated in the _EffectType enum.
    """

    def __init__(self):
        super().__init__("with_effects")

    def __call__(
        self,
        token,
        op: OpType,
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        # 确保操作类型是 HigherOrderOperator 或 OpOverload 类型
        assert isinstance(op, (torch._ops.HigherOrderOperator, torch._ops.OpOverload))
        # 确保操作没有别名
        assert not has_aliasing(op), "Ops with aliasing is not supported"
        # 确保操作具有副作用
        assert has_effects(op, args, kwargs)
        # 确保 kwargs 是一个字典
        assert isinstance(kwargs, dict)
        # 调用父类的 __call__ 方法，并返回结果
        return super().__call__(token, op, *args, **kwargs)

# 创建 WithEffects 类的实例对象 with_effects
with_effects = WithEffects()

# 定义函数 has_aliasing，判断操作是否具有别名
def has_aliasing(op: OpType):
    # NOT FOR PUBLIC USE
    # 如果操作是 HigherOrderOperator 类型且不在 SIDE_EFFECTS 字典中，则返回 True
    if isinstance(op, torch._ops.HigherOrderOperator):
        return op not in SIDE_EFFECTS

    # 遍历操作的参数和返回值，如果有别名信息，则返回 True
    for arg in op._schema.arguments:
        if arg.alias_info is not None:
            return True
    for arg in op._schema.returns:
        if arg.alias_info is not None:
            return True
    # 否则返回 False
    return False

# 定义函数 has_effects，判断操作是否具有副作用
def has_effects(op, args, kwargs) -> bool:
    # Skip over the profiler's RecordFunction as they should not show up in the graph
    # 定义跳过的操作集合 _skip_ops
    _skip_ops = {torch.ops.profiler._record_function_exit._RecordFunction}
    # 如果操作符 op 在 _skip_ops 中，则返回 False
    if op in _skip_ops:
        return False

    # 返回条件：
    # - op 是 torch._ops.HigherOrderOperator 或 torch._ops.OpOverload 的实例
    # - op 没有别名
    # - 调用 get_effect_key(op, args, kwargs) 的结果不为 None
    return (
        isinstance(op, (torch._ops.HigherOrderOperator, torch._ops.OpOverload))
        and not has_aliasing(op)
        and get_effect_key(op, args, kwargs) is not None
    )
# 根据操作符和参数处理副作用类型的关键字
def get_effect_key(op, args, kwargs) -> Optional[_EffectType]:
    # 如果操作符在已知的副作用字典中，返回其对应的副作用类型
    if op in SIDE_EFFECTS:
        return SIDE_EFFECTS[op]

    # 遍历参数，检查是否有 Torch 脚本对象
    for arg in args:
        if isinstance(arg, torch.ScriptObject):
            # 将操作符及其相关的副作用类型添加到副作用字典，以便下次遇到同一操作符时可以快速获取
            SIDE_EFFECTS[op] = _EffectType.ORDERED
            return _EffectType.ORDERED

    # 如果未找到已知副作用类型，则返回 None
    return None


@with_effects.py_impl(DispatchKey.CompositeExplicitAutograd)
# 使用密集模式处理带有副作用的 Torch 操作
def with_effects_dense(
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    # 执行 Torch 操作
    out = op(*args, **kwargs)
    # 创建一个新的空张量作为 token
    new_token = torch.tensor([])
    # 如果操作的输出是一个元组，则返回新的 token 和元组内容
    if isinstance(out, tuple):
        return (new_token, *out)
    # 否则返回新的 token 和单个输出
    return (new_token, out)


@with_effects.py_impl(FakeTensorMode)
# 使用伪张量模式处理带有副作用的 Torch 操作
def with_effects_fake(
    mode,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    # 进入伪张量模式的上下文
    with mode:
        # 调用密集模式处理带有副作用的 Torch 操作
        result = with_effects_dense(token, op, *args, **kwargs)
        return result


@with_effects.py_impl(ProxyTorchDispatchMode)
# 使用代理 Torch 调度模式处理带有副作用的 Torch 操作
def with_effects_proxy(
    mode,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    # 如果模式未启用追踪，则使用默认的处理方式
    if not mode.enable_tracing:
        return with_effects(token, op, *args, **kwargs)

    # 禁用代理模式的追踪
    with disable_proxy_modes_tracing():
        # 调用密集模式处理带有副作用的 Torch 操作
        out = with_effects(token, op, *args, **kwargs)

    # 解包代理模式的 token 和参数
    proxy_token = mode.tracer.unwrap_proxy(token)
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)

    # 导入判断是否有副作用的函数
    from torch.fx.node import has_side_effect

    # 避免被 DCE（Dead Code Elimination）优化
    has_side_effect(op)

    # 创建代理并跟踪张量树
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        with_effects,
        (proxy_token, op, *proxy_args),
        proxy_kwargs,
    )
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


# 指定默认的 Torch 调度模式为 AutogradCPU
with_effects.fallthrough(DispatchKey.AutogradCPU)
# 指定默认的 Torch 调度模式为 AutogradCUDA
with_effects.fallthrough(DispatchKey.AutogradCUDA)


# 根据操作符和参数获取 Torch 函数的模式
def _get_schema(op, args) -> torch.FunctionSchema:
    # 如果操作符是 Torch 的操作重载对象，则返回其模式
    if isinstance(op, torch._ops.OpOverload):
        return op._schema
    # 如果操作符是 call_torchbind，则返回对应对象的模式
    elif op == call_torchbind:
        return getattr(args[0], args[1]).schema
    else:
        # 抛出运行时异常，说明无法获取操作符的模式
        raise RuntimeError(f"Unable to get schema for op {op}")


# 处理带有副作用的 Torch 操作
def handle_effects(
    allow_token_discovery: bool,
    tokens: Dict[_EffectType, torch.Tensor],
    op: OpType,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """
    处理带有副作用的 Torch 操作的效果
    """
    # 根据是否允许发现令牌来获取一个令牌。在代理模式跟踪期间，我们不能使用 `tokens.get(op, torch.tensor([]))`，
    # 因为如果令牌不存在，这会在代理模式跟踪期间创建一个空张量。但是在代理模式跟踪期间，所有的令牌应该都存在。
    key = get_effect_key(op, args, kwargs)
    assert key is not None
    
    # 如果给定效果类型的令牌不存在，根据是否允许发现令牌来进行断言处理。
    if key not in tokens:
        assert (
            allow_token_discovery
        ), f"Could not find a token for effect {key} which came from the function {op}"
        # 创建一个空张量作为该效果类型的令牌
        tokens[key] = torch.tensor([])
    
    # 获取效用功能化 API 的上下文环境
    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI
    ctx = PythonFunctionalizeAPI()
    
    # 解包令牌及其参数和关键字参数的张量
    unwrapped_token = ctx.unwrap_tensors([token])[0]  # type: ignore[arg-type]
    unwrapped_args = ctx.unwrap_tensors(args)  # type: ignore[arg-type]
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    
    # 使用效用功能化 API 的重调度上下文管理器调用 `with_effects`
    with ctx.redispatch_to_next():
        (new_token, *unwrapped_outs) = with_effects(
            unwrapped_token, op, *unwrapped_args, **unwrapped_kwargs  # type: ignore[arg-type]
        )
    
    # 获取函数操作的返回值的模式信息
    schema = _get_schema(op, unwrapped_args)
    
    # 根据返回值的数量进行断言和处理
    if len(schema.returns) == 0:
        assert unwrapped_outs[0] is None
        unwrapped_outs = None  # type: ignore[assignment]
    elif len(schema.returns) == 1:
        assert len(unwrapped_outs) == 1
        unwrapped_outs = unwrapped_outs[0]
    else:
        assert len(unwrapped_outs) == len(schema.returns)
    
    # 将新创建的令牌添加到令牌映射中，以便后续调用可以使用这个令牌
    wrapped_token = ctx.wrap_tensors(new_token)
    assert isinstance(wrapped_token, torch.Tensor)
    tokens[key] = wrapped_token
    
    # 将处理过的返回值重新包装并返回
    return ctx.wrap_tensors(unwrapped_outs)  # type: ignore[arg-type]
```