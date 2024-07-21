# `.\pytorch\torch\_dispatch\python.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类
import itertools  # 导入 itertools 模块，用于生成迭代器的工具函数
import unittest.mock  # 导入 unittest.mock 模块，用于创建和配置模拟对象
from contextlib import contextmanager  # 导入 contextmanager 类，用于定义上下文管理器
from typing import Iterator  # 导入 Iterator 类型，用于指定生成器函数返回的迭代器类型

import torch  # 导入 PyTorch 模块
import torch._C  # 导入 PyTorch C++ 扩展模块
import torch._ops  # 导入 PyTorch 操作符模块
import torch.utils._python_dispatch  # 导入 PyTorch Python 调度工具模块
import torch.utils._pytree as pytree  # 导入 PyTorch 树形数据结构模块

__all__ = ["enable_python_dispatcher", "no_python_dispatcher", "enable_pre_dispatch"]

# 定义变量，用于简化调用禁用和启用 Python 调度器以及预调度器的方法
no_python_dispatcher = torch._C._DisablePythonDispatcher
enable_python_dispatcher = torch._C._EnablePythonDispatcher
enable_pre_dispatch = torch._C._EnablePreDispatch

# 控制是否对函数进行功能化
CROSSREF_FUNCTIONALIZE = False


def all_py_loaded_overloads() -> Iterator[torch._ops.OpOverload]:
    """
    返回一个迭代器，迭代其中所有从 Python 访问过的 torch.ops 函数的 OpOverload 对象。
    注意：此列表可能不是完整的，仅包含实际从 Python 调用过的函数。
    """
    for ns in torch.ops:
        packets = getattr(torch.ops, ns)
        for op_name in packets:
            packet = getattr(packets, op_name)
            for overload in packet:
                yield getattr(packet, overload)


@contextmanager
def suspend_functionalization():
    """
    提供一个上下文管理器，用于临时挂起功能化（functionalization）过程。
    在上下文中禁用功能化，退出上下文时恢复功能化设置。
    """
    f_tls = torch._C._dispatch_tls_is_dispatch_key_included(
        torch._C.DispatchKey.Functionalize
    )
    f_rv = torch._C._functionalization_reapply_views_tls()
    if f_tls:
        torch._disable_functionalization()
    try:
        yield
    finally:
        if f_tls:
            torch._enable_functionalization(reapply_views=f_rv)


def check_tensor_metadata_matches(nv, rv, desc):
    """
    检查两个张量的元数据是否匹配。
    参数:
    - nv: 新张量
    - rv: 参考张量
    - desc: 描述函数，用于生成断言消息
    """
    assert callable(desc)
    assert nv.size() == rv.size(), f"{desc()}: sizes {nv.size()} != {rv.size()}"
    assert nv.dtype == rv.dtype, f"{desc()}: dtype {nv.dtype} != {rv.dtype}"
    same_strides, idx = torch._prims_common.check_significant_strides(
        nv, rv, only_cuda=False
    )
    assert (
        same_strides
    ), f"{desc()}: strides {nv.stride()} != {rv.stride()} (mismatch at index {idx})"


def check_metadata_matches(n, r, desc):
    """
    检查两个对象的元数据是否匹配。
    参数:
    - n: 新对象
    - r: 参考对象
    - desc: 描述函数，用于生成断言消息
    """
    assert callable(desc)
    n_vals, n_spec = pytree.tree_flatten(n)
    r_vals, r_spec = pytree.tree_flatten(r)
    # TODO: test the specs match; empirically  sometimes we have a tuple
    # on one side and a list on the other
    # 断言确保 n_vals 和 r_vals 的长度相等，如果不相等则抛出带有具体长度信息的异常消息
    assert len(n_vals) == len(r_vals), f"{len(n_vals)} != {len(r_vals)}"
    # 使用 zip 函数遍历 n_vals 和 r_vals，同时获取它们的索引 i，以及对应的值 nv 和 rv
    for i, nv, rv in zip(range(len(n_vals)), n_vals, r_vals):
        # 如果 rv 不是 torch.Tensor 对象，则跳过本次循环，继续处理下一个元素
        if not isinstance(rv, torch.Tensor):
            continue
        # 调用 check_tensor_metadata_matches 函数，比较 nv 和 rv 的元数据是否匹配，
        # 并传入一个 lambda 表达式作为描述函数，生成形如 "desc() output i" 的描述信息
        check_tensor_metadata_matches(nv, rv, lambda: f"{desc()} output {i}")
class Lit:
    def __init__(self, s):
        self.s = s
    def __repr__(self):
        return self.s



# 定义了一个简单的类 `Lit`，用于存储字符串并返回其表示形式
class Lit:
    def __init__(self, s):
        self.s = s
    # 返回存储的字符串表示形式
    def __repr__(self):
        return self.s



def _fmt(a: object) -> object:
    if isinstance(a, torch.Tensor):
        return Lit(
            f"torch.empty_strided({tuple(a.size())}, {a.stride()}, dtype={a.dtype})"
        )
    else:
        return a



# 根据输入参数 `a` 的类型，返回不同的对象或值
def _fmt(a: object) -> object:
    # 如果 `a` 是 `torch.Tensor` 类型，则返回一个 `Lit` 对象，表示该张量的描述信息
    if isinstance(a, torch.Tensor):
        return Lit(
            f"torch.empty_strided({tuple(a.size())}, {a.stride()}, dtype={a.dtype})"
        )
    else:
        return a  # 否则直接返回 `a`，不做处理



def make_crossref_functionalize(op, final_key):
    from torch._subclasses.fake_tensor import FakeTensorMode

    # This case is pretty weird, suppress it for now
    if op == torch.ops.aten.lift_fresh.default:
        return final_key

    def handler(*args, **kwargs):
        fake_mode = FakeTensorMode()

        def fakeify_defun(t):
            if isinstance(t, torch.Tensor):
                if torch._is_functional_tensor(t):
                    r = torch._from_functional_tensor(t)
                    # NB: This assumes that the inner tensor sizes/strides match
                    # the outer tensor sizes/strides.  This doesn't necessarily have to
                    # be the case, see discussion at
                    # https://github.com/pytorch/pytorch/pull/87610/files/401ddeda1d769bedc88a12de332c7357b60e51a4#r1007264456
                    assert t.size() == r.size()
                    assert t.stride() == r.stride()
                else:
                    r = t
                # TODO: suppress guards
                return fake_mode.from_tensor(r)
            return t

        def maybe_detach(t):
            if isinstance(t, torch.Tensor):
                return t.detach()
            else:
                return t

        # TODO: This probably does the wrong thing if you're running other
        # substantive modes with the normal op outside here
        with torch.utils._python_dispatch._disable_current_modes(), suspend_functionalization():
            f_args, f_kwargs = pytree.tree_map(fakeify_defun, (args, kwargs))
            orig_f_args, orig_f_kwargs = pytree.tree_map(
                maybe_detach, (f_args, f_kwargs)
            )
            with fake_mode:
                f_r = op(*f_args, **f_kwargs)
        r = op._op_dk(final_key, *args, **kwargs)

        def desc():
            fmt_args = ", ".join(
                itertools.chain(
                    (repr(pytree.tree_map(_fmt, a)) for a in orig_f_args),
                    (
                        f"{k}={pytree.tree_map(_fmt, v)}"
                        for k, v in orig_f_kwargs.items()
                    ),
                )
            )
            return f"{op}({fmt_args})"

        check_metadata_matches(f_r, r, desc)
        return r

    return handler



# 创建一个函数，用于处理给定的操作 `op` 和最终键 `final_key`
def make_crossref_functionalize(op, final_key):
    from torch._subclasses.fake_tensor import FakeTensorMode

    # 特殊情况处理，暂时忽略
    if op == torch.ops.aten.lift_fresh.default:
        return final_key

    def handler(*args, **kwargs):
        fake_mode = FakeTensorMode()

        # 将函数定义为处理张量的假设函数
        def fakeify_defun(t):
            if isinstance(t, torch.Tensor):
                if torch._is_functional_tensor(t):
                    r = torch._from_functional_tensor(t)
                    # 注意：这里假设内部张量的大小/步长与外部张量的大小/步长匹配
                    assert t.size() == r.size()
                    assert t.stride() == r.stride()
                else:
                    r = t
                # TODO: 禁止某些保护机制
                return fake_mode.from_tensor(r)
            return t

        # 可能的话将张量分离出来
        def maybe_detach(t):
            if isinstance(t, torch.Tensor):
                return t.detach()
            else:
                return t

        # 注意：如果在此外部正常操作时运行其他实质性模式，可能会出错
        with torch.utils._python_dispatch._disable_current_modes(), suspend_functionalization():
            f_args, f_kwargs = pytree.tree_map(fakeify_defun, (args, kwargs))
            orig_f_args, orig_f_kwargs = pytree.tree_map(
                maybe_detach, (f_args, f_kwargs)
            )
            with fake_mode:
                f_r = op(*f_args, **f_kwargs)
        r = op._op_dk(final_key, *args, **kwargs)

        # 描述函数
        def desc():
            fmt_args = ", ".join(
                itertools.chain(
                    (repr(pytree.tree_map(_fmt, a)) for a in orig_f_args),
                    (
                        f"{k}={pytree.tree_map(_fmt, v)}"
                        for k, v in orig_f_kwargs.items()
                    ),
                )
            )
            return f"{op}({fmt_args})"

        # 检查元数据匹配情况
        check_metadata_matches(f_r, r, desc)
        return r

    return handler



# 注意：启用此选项会很慢，不要在热循环中使用。仅供调试目的。
@contextmanager
def enable_crossref_functionalize():
    for op in all_py_loaded_overloads():
        op._uncache_dispatch(torch._C.DispatchKey.Functionalize)



# 定义了一个上下文管理器 `enable_crossref_functionalize`，用于处理所有已加载重载的操作
@contextmanager
def enable_crossref_functionalize():
    for op in all_py_loaded_overloads():
        op._uncache_dispatch(torch._C.DispatchKey.Functionalize)



# 注意：启用此选项会很慢，不要在热循环中使用。仅供调试目的。
@contextmanager
def enable_crossref_functionalize():
    for op in all_py_loaded_overloads():
        op._uncache_dispatch(torch._C.DispatchKey.Functionalize)



# 定义了一个上下文管理器 `enable_crossref_functionalize`，用于处理所有已加载重载的操作
@contextmanager
def enable_crossref_functionalize():
    for op in all_py_loaded_overloads():
        op._uncache_dispatch(torch._C.DispatchKey.Functionalize)
    # 尝试执行以下代码块，启用 Python 调度器并使用 unittest.mock.patch 临时修改 torch._dispatch.python.CROSSREF_FUNCTIONALIZE 为 True
    try:
        # 在当前上下文中启用 Python 调度器和修改 torch._dispatch.python.CROSSREF_FUNCTIONALIZE 为 True
        with enable_python_dispatcher(), unittest.mock.patch(
            "torch._dispatch.python.CROSSREF_FUNCTIONALIZE", True
        ):
            # 使用生成器 yield 将控制权交给调用者，在此期间执行测试代码或其他操作
            yield
    finally:
        # 最终块：在退出当前生成器后，清理所有已加载的 Python 重载操作
        for op in all_py_loaded_overloads():
            # 取消缓存与 torch._C.DispatchKey.Functionalize 相关联的操作
            op._uncache_dispatch(torch._C.DispatchKey.Functionalize)
```