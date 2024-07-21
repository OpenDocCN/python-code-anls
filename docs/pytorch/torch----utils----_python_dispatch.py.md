# `.\pytorch\torch\utils\_python_dispatch.py`

```py
# mypy: allow-untyped-defs
# 引入上下文管理模块
import contextlib

# 引入警告模块
import warnings
# 引入数据类装饰器
from dataclasses import dataclass
# 引入类型提示相关模块
from typing import Any, Dict, List, Optional, Set, Union

# 引入PyTorch库
import torch
# 引入torchgen模块
import torchgen
# 引入torchgen.model模块
import torchgen.model
# 引入PyTorch C++扩展中的特定函数和类
from torch._C import (
    _get_dispatch_stack_at,
    _len_torch_dispatch_stack,
    _pop_torch_dispatch_stack,
    _push_on_torch_dispatch_stack,
    DispatchKey,
)


# TODO: 在暴露之前，我们需要修复 enable_torch_dispatch_mode 的限制和事项：
# - 我们需要一个更好的用户接口来处理 _DisableTorchDispatch ，
#   能够有选择性地禁用某个类的 __torch_dispatch__ 。
# - 它不能与张量构造函数（如 torch.tensor, torch.Tensor）一起使用。
# - 更好的命名（参见 https://github.com/pytorch/pytorch/pull/63496#discussion_r694091694 ）

# 标志变量，表示当前是否处于 torch_dispatch 模式
_is_in_torch_dispatch_mode = False


# 返回当前是否处于 torch_dispatch 模式的状态
def is_in_torch_dispatch_mode() -> bool:
    return _is_in_torch_dispatch_mode


# 定义一个类 TorchDispatchMode
class TorchDispatchMode:
    """
    TorchDispatchMode 类允许你在动态范围内重写所有可以被 __torch_dispatch__
    函数重写的含义，而不需要实际创建张量子类或手动修补 PyTorch API 中的函数。
    一些常见的情况下可以使用该模式：

        * 你想要重写工厂函数或者其他不接受张量作为参数的函数的含义
          （这些函数不能通过张量子类进行重写）。

        * 你想要在不需要将输入包装在张量子类中的情况下重写所有函数的行为；
          例如，如果你只是想要记录中间计算结果。

        * 你想要显式地控制各种张量子类的执行顺序，而不是通过返回 NotImplemented 隐式地实现。

    独立的 TorchDispatchMode 子类是可组合的：
    可以使用 ``with MyMode():`` 将模式推送到堆栈上。
    当你在 __torch_dispatch__ 实现中调用 PyTorch API 中的函数时，
    默认情况下它们将继续转发到模式堆栈上的下一个模式。
    如果你想要递归地调用当前的 __torch_dispatch__ 实现，可以显式调用 ``self.__torch_dispatch__(...)``，
    或者使用上下文管理器 ``__torch_dispatch__(self)`` 来使 PyTorch API 自我引用
    （在这种情况下要小心无限循环！）
    """

    def __init__(self, _dispatch_key=None):
        if _dispatch_key is not None:
            assert isinstance(_dispatch_key, torch._C.DispatchKey)
            self.__dict__["_dispatch_key"] = _dispatch_key

        self.old_dispatch_mode_flag = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 抛出未实现错误，用于子类重写具体的 __torch_dispatch__ 行为
        raise NotImplementedError
    # 进入上下文管理器时调用，设置全局变量 _is_in_torch_dispatch_mode 为 True
    def __enter__(self):
        global _is_in_torch_dispatch_mode
        # 保存旧的 _is_in_torch_dispatch_mode 标志
        self.old_dispatch_mode_flag = _is_in_torch_dispatch_mode
        _is_in_torch_dispatch_mode = True
        # 将当前对象推入模式栈
        _push_mode(self)
        return self  # 返回当前对象自身作为上下文管理器的结果

    # 退出上下文管理器时调用，恢复旧的 dispatch mode 和弹出模式栈中的模式
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 获取可能存在的 _dispatch_key 或者 _mode_key
        mb_dk_or_mode_key = self.__dict__.get("_dispatch_key", None)
        if mb_dk_or_mode_key is None:
            # 如果不存在 _dispatch_key，则查找 _mode_key
            # 目前在预调度的逻辑中完全不使用模式键（mode keys）
            # 可能需要重新审视这一点
            mb_dk_or_mode_key = self.__dict__.get("_mode_key", None)
        global _is_in_torch_dispatch_mode
        # 恢复旧的 _is_in_torch_dispatch_mode 标志
        _is_in_torch_dispatch_mode = self.old_dispatch_mode_flag
        # 从模式栈中弹出模式键（可能是 _dispatch_key 或 _mode_key）
        _pop_mode(mb_dk_or_mode_key)

    @classmethod
    # 类方法，用于推入新的模式（Mode）实例，不建议使用，已经废弃
    def push(cls, *args, **kwargs):
        warnings.warn(
            "`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`"
        )
        # 实例化 Mode 类并返回
        instance = cls(*args, **kwargs)
        return instance
# 获取当前的调度模式，如果调度栈中有任何模式，则返回栈顶的用户模式
def _get_current_dispatch_mode():
    stack_len = _len_torch_dispatch_stack()
    if stack_len > 0:
        return _get_dispatch_stack_at(stack_len - 1)
    return None


# 检测基础设施模式，根据给定的键返回预调度模式或后调度模式
def _detect_infra_mode(key):
    assert key in [torch._C._TorchDispatchModeKey.FUNCTIONAL, torch._C._TorchDispatchModeKey.PROXY]
    from torch._ops import _get_dispatch_mode_pre_dispatch

    pre_dispatch_mode = _get_dispatch_mode_pre_dispatch(
        key
    )
    post_dispatch_mode = torch._C._get_dispatch_mode(
        key
    )

    assert (pre_dispatch_mode is None) or (
        post_dispatch_mode is None
    )

    if pre_dispatch_mode is None:
        return post_dispatch_mode

    return pre_dispatch_mode


# 取消基础设施模式，根据给定的键取消预调度模式或后调度模式
def _unset_infra_mode(key):
    from torch._ops import _get_dispatch_mode_pre_dispatch, unset_mode_pre_dispatch

    pre_dispatch_mode = _get_dispatch_mode_pre_dispatch(key)
    post_dispatch_mode = torch._C._get_dispatch_mode(key)
    if pre_dispatch_mode and post_dispatch_mode:
        raise AssertionError(
            "Can't have active infra mode on both pre and post dispatch mode stack"
        )

    if pre_dispatch_mode:
        mode = unset_mode_pre_dispatch(key)
        return mode
    if post_dispatch_mode:
        return torch._C._unset_dispatch_mode(key)


# 禁用基础设施模式，根据给定的键禁用预调度模式或后调度模式
def _disable_infra_mode(key):
    assert key in (
        torch._C._TorchDispatchModeKey.FUNCTIONAL,
        torch._C._TorchDispatchModeKey.PROXY,
    )
    mode_unset = _unset_infra_mode(key)
    try:
        yield mode_unset
    finally:
        if mode_unset is not None:
            _push_mode(mode_unset)


# 获取当前调度模式栈，返回所有调度模式的列表
def _get_current_dispatch_mode_stack():
    stack_len = _len_torch_dispatch_stack()
    return [_get_dispatch_stack_at(i) for i in range(stack_len)]


# 推送模式到调度模式栈
def _push_mode(mode: TorchDispatchMode):
    k = mode._dispatch_key if hasattr(mode, "_dispatch_key") else None
    assert k is None or k == torch._C.DispatchKey.PreDispatch
    if k is None:
        _push_on_torch_dispatch_stack(mode)
        return

    from torch._ops import _set_mode_pre_dispatch, get_cached_ops

    # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
    # Clear the cache of every op that has been used so far, for this particular key.
    ks = torch._C._functionality_to_backend_keys(k)
    for op in get_cached_ops():
        for key in ks:
            op._uncache_dispatch(key)
    _set_mode_pre_dispatch(mode)


# 从调度模式栈弹出模式
def _pop_mode(k: Optional[Union[DispatchKey, torch._C._TorchDispatchModeKey]] = None):
    if k == torch._C.DispatchKey.PreDispatch:  # type: ignore[attr-defined]
        from torch._ops import _pop_mode_from_pre_dispatch

        return _pop_mode_from_pre_dispatch()

    if k is None or isinstance(k, torch._C._TorchDispatchModeKey):
        return _pop_torch_dispatch_stack(k)


# 临时性地从调度模式栈中弹出模式，支持上下文管理器
@contextlib.contextmanager
def _pop_mode_temporarily(k: Optional[DispatchKey] = None):
    old = _pop_mode(k)
    try:
        yield old
    finally:
        _push_mode(old)
@contextlib.contextmanager
def _disable_current_modes():
    # 导入必要的模块和函数
    from torch._ops import (
        _len_torch_dispatch_stack_pre_dispatch,
        _pop_mode_from_pre_dispatch,
    )
    from torch._subclasses.functional_tensor import FunctionalTensorMode
    from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
    from torch._subclasses.schema_check_mode import SchemaCheckMode

    # 获取当前在预调度栈中的模式数量
    mode_len_pre_dispatch = _len_torch_dispatch_stack_pre_dispatch()
    # 弹出所有预调度栈中的模式，并保存到列表中
    old_pre_dispatch_modes = [
        _pop_mode_from_pre_dispatch() for _ in range(mode_len_pre_dispatch)
    ]

    # 初始化标志位，用于检测是否存在特定类型的模式
    has_proxy_mode_in_pre_dispatch = False
    has_functional_mode_in_pre_dispatch = False
    has_schema_check_mode_in_pre_dispatch = False

    # 遍历预调度模式列表，检查是否包含特定类型的模式
    for i in old_pre_dispatch_modes:
        if isinstance(i, ProxyTorchDispatchMode):
            has_proxy_mode_in_pre_dispatch = True
        if isinstance(i, FunctionalTensorMode):
            has_functional_mode_in_pre_dispatch = True
        if isinstance(i, SchemaCheckMode):
            has_schema_check_mode_in_pre_dispatch = True

    # 获取当前调度栈中的模式数量
    mode_len = _len_torch_dispatch_stack()
    # 弹出当前调度栈中的所有模式，并保存到列表中
    old_modes = [_pop_mode() for _ in range(mode_len)]

    # 遍历当前调度模式列表，检查是否存在与预调度中重复的模式
    for old in old_modes:
        if (
            isinstance(old, FunctionalTensorMode)
            and has_functional_mode_in_pre_dispatch
        ):
            # 如果同时在预调度和当前调度中存在功能张量模式，则引发异常
            raise AssertionError(
                "Can't have FunctionalMode available both in PreDispatch and Python Key"
            )
        if isinstance(old, ProxyTorchDispatchMode) and has_proxy_mode_in_pre_dispatch:
            # 如果同时在预调度和当前调度中存在代理调度模式，则引发异常
            raise AssertionError(
                "Can't have ProxyTorchDispatchMode available both in PreDispatch and Python Key"
            )
        if (
            isinstance(old, SchemaCheckMode)
            and has_schema_check_mode_in_pre_dispatch
        ):
            # 如果同时在预调度和当前调度中存在模式检查模式，则引发异常
            raise AssertionError(
                "Can't have SchemaCheckMode available both in PreDispatch and Python Key"
            )

    # 尝试执行代码块，产生器函数的主体部分
    try:
        # 返回当前禁用的预调度和当前调度模式列表
        yield old_pre_dispatch_modes + old_modes
    finally:
        # 在禁用模式的结束时，逆序恢复当前调度模式
        for mode in reversed(old_modes):
            _push_mode(mode)
        # 在禁用模式的结束时，逆序恢复预调度模式
        for mode in reversed(old_pre_dispatch_modes):
            _push_mode(mode)
    """
    检查给定对象是否为 torch.Tensor 的子类，并且该子类实现了特定的方法。

    Parameters:
    t : object
        待检查的对象。

    Returns:
    bool
        如果对象是 torch.Tensor 的子类，并且实现了 "__tensor_flatten__" 和 "__tensor_unflatten__" 方法，则返回 True；否则返回 False。
    """
    # 检查对象是否是 torch.Tensor 的子类，并且不是直接的 torch.Tensor 类型
    is_subclass = isinstance(t, torch.Tensor) and type(t) != torch.Tensor
    # 检查子类是否实现了 "__tensor_flatten__" 和 "__tensor_unflatten__" 方法
    return (
        is_subclass
        and hasattr(t, "__tensor_flatten__")
        and hasattr(t, "__tensor_unflatten__")
    )
# 给定一个可追踪的、实现了``__torch_dispatch__``方法的包装张量子类``t``，以及一个回调函数``callback``，
# ``transform_subclass``函数将构建一个新的包装张量子类实例。
def transform_subclass(t, callback, outer_size=None, outer_stride=None):
    # 如果未提供外部大小，使用张量``t``的大小作为默认值
    outer_size = outer_size if outer_size is not None else t.size()
    # 如果未提供外部步幅，使用张量``t``的步幅作为默认值
    outer_stride = outer_stride if outer_stride is not None else t.stride()

    # 调用张量``t``的``__tensor_flatten__``方法，获取其属性和上下文信息
    attrs, ctx = t.__tensor_flatten__()
    # 创建一个空字典，用于存储转换后的张量
    transformed_tensors_dict = {}
    # 遍历所有属性
    for attr in attrs:
        # 对每个属性调用回调函数``callback``，将属性名和属性值作为参数，得到转换后的张量
        transformed_tensors_dict[attr] = callback(attr, getattr(t, attr))
    
    # 调用``type(t).__tensor_unflatten__``方法，使用转换后的张量字典、上下文信息、外部大小和步幅来构造新的张量子类实例
    sub = type(t).__tensor_unflatten__(
        transformed_tensors_dict, ctx, outer_size, outer_stride
    )

    # 断言新构造的张量``sub``的形状与期望的外部大小相同
    assert sub.shape == outer_size, (
        f"Expected return value from {type(t)}__tensor_unflatten__() to have "
        f"shape equal to {outer_size}, but got: {sub.shape}"
    )
    # 断言新构造的张量``sub``的步幅与期望的外部步幅相同
    assert sub.stride() == outer_stride, (
        f"Expected return value from {type(t)}__tensor_unflatten__() to have "
        f"stride equal to {outer_stride}, but got: {sub.stride()}"
    )

    # 返回新构造的张量子类实例
    return sub


# 给定一个操作重载``func``、一个来自torchgen的SchemaInfo（关于模式的缓存信息）和操作重载的输入/输出``args``和``outs``，
# 这个函数检查是否``func``是一个视图操作符（通过检查操作模式中的任何输出是否是输入的不可变别名）。
# 如果是，该函数将输出张量的存储与其对应的输入张量别名进行手动关联。
# 它通过不安全地将输出张量的存储字段覆盖为与输入相同的存储来实现这一点。
def _correct_storage_aliasing(func, schema_info, args, outs):
    # 断言``func``是torch._ops.OpOverload的实例
    assert isinstance(func, torch._ops.OpOverload)
    # 断言``args``是一个元组
    assert isinstance(args, tuple)
    # 断言``outs``是一个列表或元组
    assert isinstance(outs, (list, tuple))
    # 将``outs``扁平化为一个列表
    flat_outs = torch.utils._pytree.tree_leaves(outs)
    # 定义一个函数 alias_non_inplace_storage，用于处理非原地存储的情况
    def alias_non_inplace_storage(arg, ret):
        # 这是一个合理的断言：
        # 需要依赖此 API 进行输出别名的子类
        # 应始终返回手动别名的包装张量子类。
        # 理论上，如果一个需要此 API 的子类有时想返回普通张量，
        # 我们可以移除断言并且不执行别名操作，
        # 但更安全的做法是先了解更多关于这种情况的信息。
        
        # 检查输入参数 arg 和返回值 ret 是否是可以追踪的包装类子类
        if is_traceable_wrapper_subclass(arg) or is_traceable_wrapper_subclass(ret):
            # 如果 ret 是列表，则将其赋值给 ret_list；否则将 ret 包装成列表
            ret_list = ret if isinstance(ret, list) else [ret]
            # 遍历 ret_list 中的每一个元素 r
            for r in ret_list:
                # 断言 arg 和 r 的类型相同，如果不同则触发 AssertionError
                assert type(arg) == type(
                    r
                ), f"""Called {str(func)} with input of type {type(arg)}
                   and output type mismatch."""
def return_and_correct_aliasing(func, schema_info, args, outs):
    """Correct aliasing for the return values of a function.

    Args:
        func: Function object whose return values need alias correction.
        schema_info: Schema information for the function's arguments and returns.
        args: List of arguments passed to the function.
        outs: List of output tensors returned by the function.

    Returns:
        None

    This function ensures correct aliasing between arguments and return values
    to maintain consistency and prevent unintended side effects.
    """

    # Need to run under no_dispatch, because we explicitly do **not**
    # want our subclass to intercept the set_() call.
    with torch.utils._mode_utils.no_dispatch():
        # See Note: [Fake Tensor Dispatch Keys]
        # we're borrowing the way it modifies dispatch key TLS.
        meta_in_tls = torch._C._meta_in_tls_dispatch_include()
        torch._C._set_meta_in_tls_dispatch_include(True)
        try:
            # directly calling this overload, and passing ret.shape, because we **explicitly**
            # don't want to reset the sizes on ret, if the storage implies a size change.
            # Why?
            # The purpose of this API is *not* to change the size/strides of our output- we assume it's already correct.
            # We just want to "fix up" the storage aliasing, without modifying or output's metadata.
            # Example: out = inp.expand(inp.shape[0], inp.shape[0])
            #     This requires swapping the storage of out to be the same as inp,
            #     but we do *not* want it to change the sizes/strides that were compute for out.

            if isinstance(ret, list):
                for r in ret:
                    torch.ops.aten.set_.source_Storage_storage_offset(
                        r,
                        arg.untyped_storage(),
                        r.storage_offset(),
                        r.shape,
                        r.stride(),
                    )
            else:
                assert isinstance(ret, torch.Tensor), f"type: {type(ret)}"
                torch.ops.aten.set_.source_Storage_storage_offset(
                    ret,
                    arg.untyped_storage(),
                    ret.storage_offset(),
                    ret.shape,
                    ret.stride(),
                )
        finally:
            torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)

def is_read_only_alias_match(arg, ret):
    """Check if there is a read-only alias match between an argument and a return value.

    Args:
        arg: Argument tensor to check for aliasing.
        ret: Return value tensor to check for aliasing.

    Returns:
        bool: True if there is a read-only alias match, False otherwise.

    This function checks whether there are shared read-only aliases between
    the argument and return value tensors, ensuring no write operation is involved.
    """
    shared_aliases = arg.alias_set & ret.alias_set
    return len(shared_aliases) > 0 and not arg.is_write

num_args = len(func._schema.arguments)
num_returns = len(func._schema.returns)
for arg_idx in range(num_args):
    for return_idx in range(num_returns):
        if is_read_only_alias_match(
            schema_info.args[arg_idx], schema_info.outs[return_idx]
        ):
            alias_non_inplace_storage(args[arg_idx], outs[return_idx])
# 定义一个类 AliasInfo，用于表示别名信息
class AliasInfo:
    alias_set: Set[str]  # 别名集合，存储字符串类型的别名
    is_write: bool  # 标识是否为写操作
    name: Optional[str]  # 可选的名称信息


# 使用 dataclass 装饰器定义类 SchemaInfo，用于存储模式信息
@dataclass
class SchemaInfo:
    args: List[AliasInfo]  # 参数列表，每个元素为 AliasInfo 类型
    outs: List[AliasInfo]  # 输出列表，每个元素为 AliasInfo 类型


# 创建一个空的字典 parsed_schema_map，用于缓存解析后的模式信息
parsed_schema_map: Dict[Any, SchemaInfo] = {}


# 定义函数 get_alias_info，用于获取给定函数的模式信息
# 如果已经缓存了该函数的模式信息，则直接返回缓存的结果
# 对于 ATen 操作，使用 torchgen 解析模式信息；对于非 ATen 操作，使用 torchscript 解析模式信息
def get_alias_info(func) -> SchemaInfo:
    if func in parsed_schema_map:
        return parsed_schema_map[func]
    
    if func.namespace == "aten":
        # 使用 torchgen 解析 ATen 操作的模式信息
        torchgen_schema_str = str(func._schema)
        assert torchgen_schema_str.startswith("aten::")
        
        # 处理 torchgen 不支持的模式信息格式问题
        torchgen_schema_str = torchgen_schema_str[6:]
        import re
        torchgen_schema_str = re.sub(r"=\[[0, ]+\]", "=0", torchgen_schema_str)
        torchgen_schema_str = re.sub(r"=\[[1, ]+\]", "=1", torchgen_schema_str)
        torchgen_schema_str = torchgen_schema_str.replace("=[0, 1]", "=[0,1]")
        
        # 解析 torchgen 的模式信息
        torchgen_schema = torchgen.model.FunctionSchema.parse(torchgen_schema_str)
        
        # 构建参数和输出的 AliasInfo 列表
        arg_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.annotation is None else set(a.annotation.alias_set)
                ),
                is_write=a.annotation is not None and a.annotation.is_write,
                name=a.name,
            )
            for a in torchgen_schema.arguments.flat_all
        ]
        out_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.annotation is None else set(a.annotation.alias_set)
                ),
                is_write=a.annotation is not None and a.annotation.is_write,
                name=a.name,
            )
            for a in torchgen_schema.returns
        ]
    else:
        # 对于非 ATen 操作，使用 torchscript 解析模式信息
        arg_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.alias_info is None else set(a.alias_info.before_set)
                ),
                is_write=a.alias_info is not None and a.alias_info.is_write,
                name=a.name,
            )
            for a in func._schema.arguments
        ]
        out_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.alias_info is None else set(a.alias_info.before_set)
                ),
                is_write=a.alias_info is not None and a.alias_info.is_write,
                name=a.name,
            )
            for a in func._schema.returns
        ]
    # 使用给定的参数和输出模式创建一个SchemaInfo对象
    schema_info = SchemaInfo(args=arg_schemas, outs=out_schemas)
    # 将新创建的SchemaInfo对象与特定函数func关联，并存储在parsed_schema_map字典中
    parsed_schema_map[func] = schema_info
    # 返回刚创建的SchemaInfo对象作为函数的结果
    return schema_info
# 定义一个函数，用于处理 tensor 的别名问题，以确保在 AOTAutograd 中的正确性。
# 该函数适用于包装器张量 ``__torch_dispatch__`` 的子类，这些子类希望与 torch.compile 协同工作。
def return_and_correct_aliasing(func, args, kwargs, out):
    """
    This function should be used by wrapper tensor ``__torch_dispatch__`` subclasses
    that would like to work with torch.compile. It ensures that the subclass
    properly implements the aliasing behavior of every op,
    which is needed for correctness in AOTAutograd.
    This function will handle:

        * When we see a view op, we will alias the storages of any
          input and output tensor subclasses

        * When we see an inplace or out= op, we will directly
          return the corresponding input tensor, instead of returning
          a (potentially) fresh output tensor.
    """

    # 在这里进行缓存是因为 torchgen 解析肯定不快，而且这个函数在 functionalization 过程中每个 op 都会调用一次。
    # 获取函数的别名信息
    schema_info = get_alias_info(func)

    # 内部函数，用于获取写入别名
    def get_write_alias(x):
        # 如果别名集合为空，则返回 None
        if len(x.alias_set) == 0:
            return None
        # 转换为列表，torchscript 允许复杂的别名集合，但我们的 dispatcher ops 只涉及简单别名
        alias_set = list(x.alias_set)
        assert len(alias_set) == 1  # 我们期望别名集合长度为1
        # 如果 x 是写入操作，则返回该别名
        if x.is_write:
            return alias_set[0]
        return None

    # 内部函数，根据输出别名获取参数
    def get_arg_from_alias(output_alias, schema_info, args, kwargs):
        # 标准化函数的新参数和新关键字参数
        new_args, new_kwargs = torch.fx.operator_schemas.normalize_function(
            func, args=args, kwargs=kwargs
        )
        # 获取输出别名在 schema 的输入参数中的索引
        arg_indices = [
            i for i, a in enumerate(schema_info.args) if output_alias in a.alias_set
        ]
        # 对于任何具有输出别名的 dispatcher op，我们期望它映射到 schema 的输入参数的一个别名。
        assert len(arg_indices) == 1  # 我们期望索引列表长度为1
        idx = arg_indices[0]
        arg_info = schema_info.args[idx]
        # 如果参数信息中存在名称并且在新关键字参数中存在，则返回该参数
        if arg_info.name is not None and arg_info.name in new_kwargs:
            return new_kwargs[arg_info.name]
        return new_args[idx]

    # 如果 func 是 view op，则修正任何 outs 的存储，使其指向与输入相同的存储。
    _correct_storage_aliasing(
        func, schema_info, args, (out,) if not isinstance(out, tuple) else out
    )

    # 特别是对于 inplace_view ops，我们将尽力确保包装器子类的元数据设置正确。
    # 如果 torch.Tag.inplace_view 在函数的标签中
    if torch.Tag.inplace_view in func.tags:
        # 使用 no_dispatch() 确保我们在秘密更改包装器的元数据时，
        # 不会将操作分发到其他地方。
        mutated_args = [
            x
            for i, x in enumerate(args)
            # 如果 schema_info.args[i] 的写别名不为 None，则将其添加到 mutated_args 中
            if get_write_alias(schema_info.args[i]) is not None
        ]
        # 假设：我们只有少量严格遵循模式的 inplace_view 操作：
        # 只有一个参数的元数据会发生变化。
        assert len(mutated_args) == 1
        # 此检查存在是因为通常我们确实希望更新任何包装器子类的元数据，
        # 但 FunctionalTensor 是特殊情况：它覆盖所有的 size/stride 调用以通过内部张量传递。
        # 因此，我们实际上不需要更新元数据（尝试这样做会导致错误）。
        from torch._subclasses.functional_tensor import FunctionalTensor

        if not isinstance(mutated_args[0], FunctionalTensor):
            with torch.utils._mode_utils.no_dispatch():
                # 参见注释: [Fake Tensor Dispatch Keys]
                # 我们借用了它修改分发键 TLS 的方式。
                meta_in_tls = torch._C._meta_in_tls_dispatch_include()
                torch._C._set_meta_in_tls_dispatch_include(True)
                try:
                    # 调用 func(*args, **kwargs)
                    func(*args, **kwargs)
                finally:
                    # 恢复原来的 meta_in_tls 值
                    torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)

    # 接下来：如果输出是可变别名（例如 add_()），我们需要确保直接返回输入。
    
    # 简单情况：我们的输出没有可变别名，因此可以直接返回输出
    if not any(get_write_alias(r) is not None for r in schema_info.outs):
        return out

    # 简化假设：我们没有任何返回类型为 "-> (Tensor(a!), Tensor)" 的操作
    if not all(get_write_alias(r) is not None for r in schema_info.outs):
        # 抛出运行时错误，指示不支持的模式
        raise RuntimeError("Unsupported schema: " + str(func._schema))

    # 如果 func._schema.returns 的长度为 1
    if len(func._schema.returns) == 1:
        # 返回从别名中获取的参数
        return get_arg_from_alias(
            get_write_alias(schema_info.outs[0]), schema_info, args, kwargs
        )

    # 在多返回值的情况下，所有的 aten 操作都返回一个元组/列表，因此相应地进行类型转换
    outs_to_return = type(out)(
        [
            (
                get_arg_from_alias(
                    get_write_alias(schema_info.outs[i]), schema_info, args, kwargs
                )
                if get_write_alias(r) is not None
                else o
            )
            for ((i, r), o) in zip(enumerate(schema_info.outs), out)
        ]
    )
    return outs_to_return
```