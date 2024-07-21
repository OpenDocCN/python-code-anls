# `.\pytorch\torch\_library\fake_impl.py`

```
# 添加类型检查声明，允许未类型化的定义
# 引入上下文管理器和函数工具
# 从 typing 模块导入 Callable 和 Optional 类型
# 从 typing_extensions 模块导入 deprecated 装饰器
import contextlib
import functools
from typing import Callable, Optional
from typing_extensions import deprecated

# 导入 PyTorch 库
import torch
# 从 torch._library.utils 中导入 Kernel 和 RegistrationHandle 类
from torch._library.utils import Kernel, RegistrationHandle


class FakeImplHolder:
    """一个可以注册虚假实现的容器。"""

    def __init__(self, qualname: str):
        # 初始化实例，保存限定名称和两个可选属性
        self.qualname: str = qualname
        self.kernel: Optional[Kernel] = None
        self.lib: Optional[torch.library.Library] = None

    def register(self, func: Callable, source: str) -> RegistrationHandle:
        """注册一个虚假实现。

        返回一个 RegistrationHandle 对象，用于取消注册这个虚假实现。
        """
        # 如果已经有了 kernel，抛出运行时错误
        if self.kernel is not None:
            raise RuntimeError(
                f"register_fake(...): the operator {self.qualname} "
                f"already has an fake impl registered at "
                f"{self.kernel.source}."
            )
        # 如果 Meta DispatchKey 已经有了实现，抛出运行时错误
        if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta"):
            raise RuntimeError(
                f"register_fake(...): the operator {self.qualname} "
                f"already has an DispatchKey::Meta implementation via a "
                f"pre-existing torch.library or TORCH_LIBRARY registration. "
                f"Please either remove that registration or don't call "
                f"register_fake."
            )

        # 如果 CompositeImplicitAutograd DispatchKey 已经有了实现，抛出运行时错误
        if torch._C._dispatch_has_kernel_for_dispatch_key(
            self.qualname, "CompositeImplicitAutograd"
        ):
            raise RuntimeError(
                f"register_fake(...): the operator {self.qualname} "
                f"already has an implementation for this device type via a "
                f"pre-existing registration to "
                f"DispatchKey::CompositeImplicitAutograd."
                f"CompositeImplicitAutograd operators do not need an fake "
                f"impl; "
                f"instead, the operator will decompose into its constituents "
                f"and those "
                f"can have fake impls defined on them."
            )

        # 将传入的 func 和 source 创建为 Kernel 对象，并存储在 self.kernel 中
        self.kernel = Kernel(func, source)

        # 如果 self.lib 为空，根据命名空间创建一个 torch.library.Library 对象
        if self.lib is None:
            ns = self.qualname.split("::")[0]
            self.lib = torch.library.Library(ns, "FRAGMENT")  # noqa: TOR901
        
        # 构建 Meta DispatchKey 的核心函数，并注册到 self.lib 中
        meta_kernel = construct_meta_kernel(self.qualname, self)
        self.lib.impl(self.qualname, meta_kernel, "Meta")

        # 定义一个 deregister_fake_class 函数，用于取消注册虚假实现
        def deregister_fake_class():
            if self.lib:
                self.lib._destroy()
                self.lib = None
            self.kernel = None

        # 返回 RegistrationHandle 对象，包含 deregister_fake_class 函数
        return RegistrationHandle(deregister_fake_class)


def construct_meta_kernel(qualname: str, fake_impl_holder: FakeImplHolder) -> Callable:
    # 断言 fake_impl_holder.kernel 不为 None
    assert fake_impl_holder.kernel is not None

    # 使用 functools.wraps 装饰器，将内部函数的元数据复制到外部函数中
    @functools.wraps(fake_impl_holder.kernel.func)
    # 定义一个函数 meta_kernel，接受任意位置和关键字参数
    def meta_kernel(*args, **kwargs):
        # 断言确保 fake_impl_holder.kernel 不为 None
        assert fake_impl_holder.kernel is not None
        # 获取 fake_impl_holder.kernel 的源代码
        source = fake_impl_holder.kernel.source

        # 定义一个内部函数 error_on_ctx，用于抛出运行时错误
        def error_on_ctx():
            raise RuntimeError(
                # 抛出错误信息，说明尝试调用 get_ctx() 获取 meta 实现的上下文
                f"Attempted to call get_ctx() for the meta implementation "
                f"for {qualname} (implemented at {source})"
                f"You have presumably called get_ctx() because the operator "
                f"has a data-dependent output shape; if so, there is no "
                f"such meta implementation and this error is the correct "
                f"behavior."
            )

        # 使用 error_on_ctx 函数作为上下文管理器，调用 fake_impl_holder.kernel(*args, **kwargs)
        with set_ctx_getter(error_on_ctx):
            # 返回 fake_impl_holder.kernel 的调用结果
            return fake_impl_holder.kernel(*args, **kwargs)

    # 返回 meta_kernel 函数对象
    return meta_kernel
def get_none():
    return None


# 定义一个函数 get_none，返回 None 对象
def get_none():
    return None



global_ctx_getter: Callable = get_none


# 定义一个全局变量 global_ctx_getter，初始值为 get_none 函数
global_ctx_getter: Callable = get_none



@contextlib.contextmanager
def set_ctx_getter(ctx_getter):
    global global_ctx_getter
    prev = global_ctx_getter
    try:
        # 设置全局的 ctx_getter 函数为传入的参数 ctx_getter
        global_ctx_getter = ctx_getter
        # yield 之前的操作作为上下文管理器的一部分
        yield
    finally:
        # 恢复先前的 global_ctx_getter 函数
        global_ctx_getter = prev


# 定义一个上下文管理器函数 set_ctx_getter，用于设置全局的 ctx_getter 函数
@contextlib.contextmanager
def set_ctx_getter(ctx_getter):
    global global_ctx_getter
    # 保存当前的 global_ctx_getter 函数
    prev = global_ctx_getter
    try:
        # 将全局的 ctx_getter 函数设置为传入的 ctx_getter
        global_ctx_getter = ctx_getter
        # yield 语句之前的部分作为上下文管理器的一部分
        yield
    finally:
        # 恢复之前保存的 global_ctx_getter 函数
        global_ctx_getter = prev



class FakeImplCtx:
    """
    Context object for writing fake implementations for custom operators.
    """

    def __init__(self, _fake_mode, _op):
        self._fake_mode = _fake_mode
        self._shape_env = _fake_mode.shape_env
        self._op = _op

    @deprecated(
        "`create_unbacked_symint` is deprecated, please use `new_dynamic_size` instead",
        category=FutureWarning,
    )
    def create_unbacked_symint(self, *, min=2, max=None) -> torch.SymInt:
        # 使用 new_dynamic_size 替代 create_unbacked_symint，返回 torch.SymInt 对象
        return self.new_dynamic_size(min=min, max=max)


# 定义一个 FakeImplCtx 类，用于编写自定义操作的虚假实现的上下文对象
class FakeImplCtx:
    """
    Context object for writing fake implementations for custom operators.
    """

    def __init__(self, _fake_mode, _op):
        # 初始化方法，接受 _fake_mode 和 _op 参数
        self._fake_mode = _fake_mode
        # 从 _fake_mode 中获取 shape_env 属性，并赋值给 _shape_env
        self._shape_env = _fake_mode.shape_env
        # 将 _op 参数赋值给实例的 _op 属性

    @deprecated(
        "`create_unbacked_symint` is deprecated, please use `new_dynamic_size` instead",
        category=FutureWarning,
    )
    def create_unbacked_symint(self, *, min=2, max=None) -> torch.SymInt:
        # 使用新的方法 new_dynamic_size 替代 create_unbacked_symint，返回 torch.SymInt 对象
        return self.new_dynamic_size(min=min, max=max)
```