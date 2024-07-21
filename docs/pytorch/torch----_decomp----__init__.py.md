# `.\pytorch\torch\_decomp\__init__.py`

```py
# mypy: allow-untyped-defs
# 导入模块，包括inspect、defaultdict、wraps、chain和特定类型的导入
import inspect
from collections import defaultdict
from functools import wraps
from itertools import chain
from typing import Callable, Dict, List, Sequence, Union

# 导入torch及其相关库
import torch
import torch.library
from torch._ops import HigherOrderOperator, OpOverload, OpOverloadPacket
from torch._prims_common import CustomOutParamAnnotation
from torch.utils import _pytree as pytree

# 模块内公开的符号列表
__all__ = [
    "decomposition_table",
    "pre_autograd_decomposition_table",
    "meta_table",
    "register_decomposition",
    "get_decompositions",
    "core_aten_decompositions",
]

# 全局的操作分解表，用于存储不同阶段的分解表
global_decomposition_table: Dict[
    str, Dict[torch._ops.OperatorBase, Callable]
] = defaultdict(dict)

# 获取后向自动求导后的分解表
decomposition_table = global_decomposition_table["post_autograd"]
# 获取前向自动求导前的分解表
pre_autograd_decomposition_table = global_decomposition_table["pre_autograd"]
# 获取元信息分解表
meta_table = global_decomposition_table["meta"]


def _add_op_to_registry(registry, op, fn):
    """
    This is an internal API for adding an op to the decomposition table.

    If op is OpOverload, it will be added to the registry directly.
    If op is OpOverloadPacket, all the valid op_overloads in the packet will be added to the registry.
    """
    overloads: List[Union[torch._ops.OperatorBase]] = []
    if isinstance(op, HigherOrderOperator):
        # 对于HigherOrderOperator，直接将其添加到注册表中
        registry[op] = fn
        return
    elif isinstance(op, OpOverload):
        # 如果op是OpOverload，将其添加到注册表中
        overloads.append(op)
    else:
        assert isinstance(op, OpOverloadPacket)
        # 如果op是OpOverloadPacket，遍历其中的所有有效op_overload，并添加到注册表中
        for ol in op.overloads():
            overloads.append(getattr(op, ol))

    for op_overload in overloads:
        if op_overload in registry:
            # 如果注册表中已经存在相同的op_overload，则引发运行时错误
            raise RuntimeError(f"duplicate registrations for {op_overload}")
        # 检查op_overload是否有对应的调度条目，如果有，则将其添加到注册表中
        if torch._C._dispatch_has_kernel(op_overload.name()):
            registry[op_overload] = fn


def _convert_out_params(f):
    out_annotation = f.__annotations__.get("out")

    # 如果没有out参数注释，则直接返回函数本身
    if not out_annotation:
        return f

    # 检测out参数是否为元组的hack方法
    # 如果是元组，则做特殊处理（略）
    pass  # Placeholder for further implementation
    # 检查 out_annotation 是否具有 "__origin__" 属性，并且其值为 tuple
    if getattr(out_annotation, "__origin__", None) is tuple:
        # 获取函数 f 的签名信息
        sig = inspect.signature(f)
        # 获取返回值注解的字段名列表
        out_names = sig.return_annotation._fields
        # 如果返回值是一个元组，则需要注册一个函数来解包所有的返回值元素，这是 native_functions.yaml 所期望的行为

        @wraps(f)
        def _fn(*args, **kwargs):
            # 提取所有的返回值关键字参数
            out_kwargs = tuple(kwargs.pop(o, None) for o in out_names)
            # 检查是否所有的返回值关键字参数都被设置或者都未设置
            is_none = out_kwargs[0] is None
            assert all((o is None) == is_none for o in out_kwargs)
            # 调用原始函数 f，并传递新的返回值关键字参数
            return f(*args, **kwargs, out=None if is_none else out_kwargs)

        # 创建用于新函数的参数列表，将 out 参数移除并添加新的关键字参数
        out_params = [
            inspect.Parameter(
                o,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=t,
            )
            for o, t in zip(out_names, out_annotation.__args__)
        ]
        # 重新构建函数的签名，将新的关键字参数拼接进入
        params = chain((v for k, v in sig.parameters.items() if k != "out"), out_params)
        _fn.__signature__ = inspect.Signature(
            parameters=params, return_annotation=sig.return_annotation
        )
        # 更新新函数的注解，移除原始函数的 "out" 参数并添加新的关键字参数
        _fn.__annotations__ = {k: v for k, v in f.__annotations__.items() if k != "out"}
        for o in out_params:
            _fn.__annotations__[o.name] = o.annotation

        # 标记新函数被 out_wrapper 包装
        _fn._torch_decompositions_out_wrapper = f._torch_decompositions_out_wrapper

        # 返回新定义的函数 _fn
        return _fn

    # 如果返回值不是元组，可能存在一个命名为 CustomOutParamAnnotation 的特殊张量输出参数，
    # 在这里移除对应的注解，以避免在包装后继续暴露它
    custom_out_param_name = f.__annotations__.pop(CustomOutParamAnnotation, None)
    # 如果存在自定义的输出参数名，则进入条件分支
    if custom_out_param_name:

        # 使用装饰器 wraps 将函数 f 包装到 _fn 中，并处理自定义输出参数
        @wraps(f)
        def _fn(*args, **kwargs):
            # 弹出 kwargs 中的自定义输出参数值
            out_kwarg = kwargs.pop(custom_out_param_name, None)
            # 调用原始函数 f，传入剩余的 args 和 kwargs，并将 out 参数传递给 f
            return f(*args, **kwargs, out=out_kwarg)

        # 创建一个参数对象 out_param，代表自定义输出参数
        out_param = inspect.Parameter(
            custom_out_param_name,
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=out_annotation,
        )

        # 获取原始函数 f 的签名
        sig = inspect.signature(f)
        # 生成一个新的参数序列，去除原始函数的 out 参数，添加新的 out_param 参数
        params = chain(
            (v for k, v in sig.parameters.items() if k != "out"), (out_param,)
        )
        # 替换 _fn 的签名为新生成的签名
        _fn.__signature__ = inspect.Signature(
            parameters=params, return_annotation=sig.return_annotation
        )

        # 更新 _fn 的注解，去除原始函数 f 中的 out 注解，添加新的 out_param 注解
        _fn.__annotations__ = {k: v for k, v in f.__annotations__.items() if k != "out"}
        _fn.__annotations__[out_param.name] = out_param.annotation

        # 返回处理后的 _fn 函数
        return _fn

    # 如果不存在自定义输出参数名，则直接返回原始函数 f
    return f
# 注册一个函数作为Python分解表的分解函数装饰器
def register_decomposition(
    aten_op, registry=None, *, type="post_autograd", unsafe=False
):
    """
    A decorator to register a function as a decomposition to the Python
    decomposition table.  Use it like this::

        @register_decomposition(torch.ops.aten.clamp_min)
        def clamp_min(x):
            return torch.clamp(self, min=min)

    If you are writing a new decomposition, consider contributing it
    directly to PyTorch in torch._decomp.decompositions.

    This API is experimental; we are almost certainly going to extend
    the API when we make decompositions eligible for use in transforms (e.g.,
    autograd) and not just backend tracing, where we then need to know if a
    decomposition can be used to simulate a transform.

    By default, we also will register it to the Meta key of dispatcher,
    and replace the c++ Meta implementation if there is already one.

    unsafe kwarg is for reuse of this function for registering non-function
    things
    """

    assert type in {"post_autograd", "pre_autograd", "meta"}

    def decomposition_decorator(fn: Callable) -> Callable:
        # 保存原始函数的引用
        orig_fn = fn
        # 如果不安全标志为False，转换输出参数
        if not unsafe:
            fn = _convert_out_params(fn)

        # 非局部变量registry的值初始化为全局分解表中指定类型的注册表
        nonlocal registry
        if registry is None:
            registry = global_decomposition_table[type]

        # 将操作注册到注册表中
        def register(op):
            _add_op_to_registry(registry, op, fn)

        # 处理允许同时处理多个aten操作
        pytree.tree_map_(register, aten_op)
        return orig_fn

    return decomposition_decorator


# 获取指定类型的分解函数字典
def get_decompositions(
    aten_ops: Sequence[Union[torch._ops.OperatorBase, OpOverloadPacket]],
    type: str = "post_autograd",
) -> Dict[torch._ops.OperatorBase, Callable]:
    """
    Retrieve a dictionary of decompositions corresponding to the list of
    operator overloads and overload packets passed as input.  Overload
    packets will include all decomposed overloads in the packet.  If there is
    no decomposition for a requested operator, it is silently ignored.

    This API is experimental; we are almost certainly going to give an alternate,
    more recommended formulation, where a user provides the set of operators
    they know how to implement, and we provide decompositions for everything
    not in this set.
    """
    assert type in {"post_autograd", "pre_autograd", "meta"}

    # 获取全局分解表中指定类型的注册表
    registry = global_decomposition_table[type]
    # 将注册表中的分解操作按照重载包进行分组
    packets_to_overloads = defaultdict(list)
    for opo in registry:
        if isinstance(opo, (OpOverload, OpOverloadPacket)):
            packets_to_overloads[opo.overloadpacket].append(opo)
    # 初始化分解函数字典
    decompositions: Dict[torch._ops.OperatorBase, Callable] = {}
    # 遍历给定的操作列表 `aten_ops`
    for op in aten_ops:
        # 检查操作是否属于 `OpOverloadPacket` 类型，并且在 `packets_to_overloads` 字典中
        if isinstance(op, OpOverloadPacket) and op in packets_to_overloads:
            # 对于每个满足条件的操作，遍历其关联的重载操作
            for op_overload in packets_to_overloads[op]:
                # 将操作重载与其在注册表 `registry` 中的对应项存入 `decompositions` 字典
                decompositions[op_overload] = registry[op_overload]
        # 如果操作属于 `torch._ops.OperatorBase` 类型，并且在注册表 `registry` 中
        elif isinstance(op, (torch._ops.OperatorBase)) and op in registry:
            # 将操作与其在注册表 `registry` 中的对应项存入 `decompositions` 字典
            decompositions[op] = registry[op]
    # 返回存储了操作及其对应项的 `decompositions` 字典
    return decompositions
# 给定一个字典，键是torch._ops.OperatorBase类型的操作符，值是与之关联的可调用对象（函数）
# 和一个序列，包含OpOverload或OpOverloadPacket类型的操作符重载或重载包
def remove_decompositions(
    decompositions: Dict[torch._ops.OperatorBase, Callable],
    aten_ops: Sequence[Union[OpOverload, OpOverloadPacket]],
) -> None:
    """
    给定从get_decompositions()获取的分解字典，移除作为输入传递的操作符重载或重载包列表中的操作符。
    如果分解字典不包含要移除的分解，则静默忽略。
    """
    # 遍历输入的操作符列表
    for op in aten_ops:
        # 如果操作符是OpOverloadPacket类型
        if isinstance(op, OpOverloadPacket):
            # 遍历OpOverloadPacket中的所有重载名称
            for overload_name in op.overloads():
                # 获取重载名称对应的操作符对象
                opo = getattr(op, overload_name)
                # 从分解字典中移除该操作符对象
                decompositions.pop(opo, None)
        # 如果操作符是OpOverload类型
        elif isinstance(op, OpOverload):
            # 从分解字典中移除该操作符对象
            decompositions.pop(op, None)


# 填充表格
import torch._decomp.decompositions
import torch._refs


# 查看注释 [Core ATen Ops]
#
# 列表是从torch/_inductor/decomposition.py复制过来的
# 排除会导致prim ops的分解
# 分解的结果是核心的aten操作
def core_aten_decompositions() -> Dict[torch._ops.OperatorBase, Callable]:
    # 引用torch.ops.aten作为aten命名空间
    aten = torch.ops.aten
    # 返回一个空的字典，函数未完整
    )
```