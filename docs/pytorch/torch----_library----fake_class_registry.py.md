# `.\pytorch\torch\_library\fake_class_registry.py`

```py
# mypy: allow-untyped-defs
# 引入日志模块，用于记录程序运行信息
import logging
# 引入类型相关模块，包括Any、Dict、Optional、Protocol、Tuple
from typing import Any, Dict, Optional, Protocol, Tuple

# 引入PyTorch库
import torch
# 从torch._library.utils中引入parse_namespace函数
from torch._library.utils import parse_namespace

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


# 一个模拟的脚本对象，包含原始脚本对象、类名及真实对象
class FakeScriptObject:
    def __init__(self, wrapped_obj: Any, script_class_name: str, x: torch.ScriptObject):
        # 包装对象，即传入的真实对象
        self.wrapped_obj = wrapped_obj
        # 原始脚本对象的完全限定类名
        self.script_class_name = script_class_name
        # 真实的脚本对象
        self.real_obj = x


# 一个模拟的脚本方法，包含自身假对象、方法名及可选的函数模式
class FakeScriptMethod:
    def __init__(
        self,
        self_fake_obj: FakeScriptObject,
        method_name: str,
        schema: Optional[torch.FunctionSchema],
    ):
        # 自身的假对象，即模拟的脚本对象
        self.self_fake_obj = self_fake_obj
        # 方法名
        self.method_name = method_name
        # Torch函数模式的可选项
        self.schema = schema

    # 调用方法，实际上是调用了torchbind中的call_torchbind函数
    def __call__(self, *args, **kwargs):
        from torch._higher_order_ops.torchbind import call_torchbind

        return call_torchbind(self.self_fake_obj, self.method_name, *args, **kwargs)


# 定义一个协议，具有从真实对象创建的静态方法
class HasStaticMethodFromReal(Protocol):
    @classmethod
    def from_real(cls, real_obj: torch.ScriptObject):
        pass


# 一个模拟的类注册表，用于管理已注册的类
class FakeClassRegistry:
    def __init__(self):
        # 已注册类的字典，键为完全限定类名，值为相应的模拟类对象
        self._registered_class: Dict[str, Any] = {}

    # 判断指定完全限定类名是否已注册
    def has_impl(self, full_qualname: str) -> bool:
        return full_qualname in self._registered_class

    # 获取指定完全限定类名对应的模拟类对象
    def get_impl(self, full_qualname: str) -> Any:
        self._check_registered(full_qualname)
        return self._registered_class[full_qualname]

    # 注册模拟类对象到指定完全限定类名
    def register(self, full_qualname: str, fake_class=None) -> None:
        if self.has_impl(full_qualname):
            # 记录警告信息，说明已覆盖之前的模拟类对象
            log.warning(
                "%s is already registered. Previous fake class is overrided with  %s.",
                full_qualname,
                fake_class,
            )
        self._registered_class[full_qualname] = fake_class

    # 从注册表中取消指定完全限定类名的模拟类对象
    def deregister(self, full_qualname: str) -> Any:
        if not self.has_impl(full_qualname):
            # 记录警告信息，说明要取消注册的类未找到
            log.warning(
                "Cannot deregister %s. Please use register_fake_class to register it first."
                " Or do you dereigster it twice?",
                full_qualname,
            )
        else:
            # 返回并移除指定类名对应的模拟类对象
            return self._registered_class.pop(full_qualname)

    # 清空注册表，移除所有模拟类对象
    def clear(self) -> None:
        self._registered_class.clear()

    # 检查指定完全限定类名是否已注册，如未注册则抛出运行时错误
    def _check_registered(self, full_qualname: str) -> None:
        if full_qualname not in self._registered_class:
            raise RuntimeError(
                f"{full_qualname} is not registered. Please use register_fake_class to register it first."
            )


# 全局的模拟类注册表对象
global_fake_class_registry = FakeClassRegistry()


# TODO: 在编译时添加此检查以验证__obj_flatten__函数的有效性。
def _check_valid_flat_script_obj(flat_x):
    # 如果flat_x不是元组类型，则抛出运行时错误
    if not isinstance(flat_x, tuple):
        raise RuntimeError("Expect flat x to be a tuple.")
    # 遍历 flat_x 中的每一个元素 tp
    for tp in flat_x:
        # 检查 tp 是否不是 tuple 类型，如果不是则抛出运行时错误
        if not isinstance(tp, tuple):
            raise RuntimeError("Expect flat x to be a tuple of tuples.")
        
        # 检查 tp 的长度是否不等于 2 或者第一个元素不是字符串类型，如果是则抛出运行时错误
        if not len(tp) == 2 or not isinstance(tp[0], str):
            raise RuntimeError(
                "Expect element of flat x to be a tuple of two elements with first element being a string"
            )
# 定义函数 to_fake_obj，接受两个参数：fake_mode 和 x，返回一个 FakeScriptObject
def to_fake_obj(fake_mode, x: torch.ScriptObject) -> FakeScriptObject:
    # 导入模块 torch.utils._pytree，并将其重命名为 pytree
    import torch.utils._pytree as pytree
    # 导入模块 torch.utils._python_dispatch 中的函数 _disable_current_modes
    from torch.utils._python_dispatch import _disable_current_modes

    # 使用 _disable_current_modes 函数创建一个上下文管理器，禁用当前的 dispatch 模式
    with _disable_current_modes():
        # 调用 x 对象的 __obj_flatten__() 方法，将返回结果赋值给 flat_x，类型为 ignore[attr-defined]
        flat_x = x.__obj_flatten__()  # type: ignore[attr-defined]

    # 调用 _check_valid_flat_script_obj 函数，检查 flat_x 是否为有效的扁平化脚本对象
    _check_valid_flat_script_obj(flat_x)

    # 使用 pytree 模块的 tree_map_only 函数，将 flat_x 中所有的 torch.Tensor 替换为 fake_mode.from_tensor(t)
    fake_flattened = pytree.tree_map_only(
        torch.Tensor,
        lambda t: fake_mode.from_tensor(t),
        flat_x,
    )

    # 调用 _find_fake_class_for_script_object 函数，根据 x 对象找到相应的 fake 类，并调用其 __obj_unflatten__ 方法
    fake_x = _find_fake_class_for_script_object(x).__obj_unflatten__(fake_flattened)

    # 创建 FakeScriptObject 对象 fake_x_wrapped，包装 fake_x，并保留原始类型名称和 x 对象的引用，类型为 ignore[attr-defined]
    fake_x_wrapped = FakeScriptObject(fake_x, x._type().qualified_name(), x)  # type: ignore[attr-defined]

    # 遍历 x 对象的所有方法名
    for name in x._method_names():  # type: ignore[attr-defined]
        # 获取 x 对象的属性 name
        attr = getattr(fake_x, name, None)
        if attr:
            # 如果 attr 存在且不是 callable，则抛出 RuntimeError 异常
            if not callable(attr):
                raise RuntimeError(f"Expect {name} to be a callable but got {attr}.")

            # 获取 x 对象中的真实方法 real_attr，类型为 ignore[attr-defined]
            real_attr = getattr(x, name)  # type: ignore[attr-defined]

            # 初始化 method_schema 为 None
            method_schema: Optional[torch.FunctionSchema] = None
            # 如果 real_attr 是 torch.ScriptMethod 类型，则获取其 schema
            if isinstance(real_attr, torch.ScriptMethod):
                method_schema = real_attr.schema  # type: ignore[attr-defined]

            # 为 fake_x_wrapped 对象设置属性 name，值为 FakeScriptMethod 对象，传入 fake_x_wrapped、name 和 method_schema
            setattr(
                fake_x_wrapped,
                name,
                FakeScriptMethod(fake_x_wrapped, name, method_schema),
            )
        else:
            # 记录警告日志，表示 fake 对象的方法没有实现
            log.warning("fake object of %s doesn't implement method %s.", x, name)
    # 返回包装后的 fake_x_wrapped 对象
    return fake_x_wrapped
    def inner(fake_class: HasStaticMethodFromReal):
        # 解析命名空间和类名
        ns, name = parse_namespace(qualname)
    
        # 获取 torch::class_ 并检查其是否存在
        torchbind_class = torch._C._get_custom_class_python_wrapper(ns, name)
    
        # 获取 fake_class 中的 _CONVERT_FROM_REAL_NAME 方法
        from_method = getattr(fake_class, _CONVERT_FROM_REAL_NAME, None)
        if not from_method:
            # 如果 fake_class 没有定义 _CONVERT_FROM_REAL_NAME 类方法，抛出运行时错误
            raise RuntimeError(
                f"{fake_class} doesn't define a classmethod {_CONVERT_FROM_REAL_NAME}."
            )
    
        # 确保 _CONVERT_FROM_REAL_NAME 方法是一个类方法
        if not isinstance(fake_class.__dict__[_CONVERT_FROM_REAL_NAME], classmethod):
            raise RuntimeError(
                f"{_CONVERT_FROM_REAL_NAME} method is not a classmethod."
            )
    
        # 将 fake_class 注册到全局的假类注册表中
        global_fake_class_registry.register(_full_qual_class_name(qualname), fake_class)
        return fake_class
    
    if fake_class is None:
        # 如果 fake_class 为 None，则返回 inner 函数本身，用于延迟执行
        return inner
    # 否则立即执行 inner 函数，并传入 fake_class 参数
    return inner(fake_class)
# 从全限定名解析出命名空间和类名，返回一个包含命名空间和类名的元组
def _ns_and_class_name(full_qualname: str) -> Tuple[str, str]:
    # 使用点号分割完全限定名
    splits = full_qualname.split(".")
    # 断言分割后的列表长度为5，即包含了正确的 Torch 类名结构
    assert len(splits) == 5
    # 解构分割后的列表元素，获取 Torch 相关的命名空间和类名
    _torch, torch_ns, classes, ns, class_name = splits
    # 返回命名空间和类名的元组
    return ns, class_name


# 为脚本对象查找对应的虚假类
def _find_fake_class_for_script_object(x: torch.ScriptObject) -> Any:
    # 获取脚本对象的完全限定名
    full_qualname = x._type().qualified_name()  # type: ignore[attr-defined]
    # 解析出命名空间和类名
    ns, class_name = _ns_and_class_name(full_qualname)
    # 查找对应的虚假类
    fake_class = find_fake_class(full_qualname)
    # 如果找不到对应的虚假类，则抛出运行时错误
    if fake_class is None:
        raise RuntimeError(
            f" ScriptObject's {full_qualname} haven't registered a fake class."
            f" Please use register_fake_class({ns}::{class_name}) to annotate a fake class for the script obj."
            f" Specifically, create a python class that implements a fake version for all the methods"
            f" that're used in the program and put annotated class in the program e.g. after loading the library."
            f" The fake methods can be written in the same way as a meta kernel for an operator but need to additionally"
            f" simulate the object's states. Be sure to add a {_CONVERT_FROM_REAL_NAME} classmethod"
            f" to enable creating a fake obj from a real one."
        )
    # 返回找到的虚假类
    return fake_class


# 从真实对象创建对应的虚假对象
def _fake_obj_from_real(fake_mode, x) -> Any:
    # 获取脚本对象对应的虚假类
    fake_class = _find_fake_class_for_script_object(x)

    # 获取虚假类中定义的从真实对象创建虚假对象的方法
    from_real_method = getattr(fake_class, _CONVERT_FROM_REAL_NAME, None)
    # 如果找不到对应方法，则抛出运行时错误
    if not from_real_method:
        raise RuntimeError(
            f"{fake_class} must define a classmethod {_CONVERT_FROM_REAL_NAME}"
            f" that converts the real object to the fake object."
        )

    # 创建一个上下文对象，用于模拟张量状态
    ctx = torch._library.fake_impl.FakeImplCtx(fake_mode, None)
    # 使用上下文对象设置上下文获取器，以确保使用正确的上下文
    with torch._library.fake_impl.set_ctx_getter(lambda: ctx):
        # 调用虚假类中的方法，将真实对象转换为虚假对象并返回
        return fake_class.from_real(x)
```