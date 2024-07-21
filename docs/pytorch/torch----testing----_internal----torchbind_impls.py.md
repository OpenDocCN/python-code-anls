# `.\pytorch\torch\testing\_internal\torchbind_impls.py`

```
# mypy: allow-untyped-defs
# 引入上下文管理模块
import contextlib
# 引入类型提示模块中的 Optional 类型
from typing import Optional

# 引入 PyTorch 模块
import torch


# 全局变量：标记 TorchBind 实现是否已初始化
_TORCHBIND_IMPLS_INITIALIZED = False

# 全局变量：用于测试的 Torch 脚本对象，可为空
_TENSOR_QUEUE_GLOBAL_TEST: Optional[torch.ScriptObject] = None


# 初始化 TorchBind 实现函数
def init_torchbind_implementations():
    global _TORCHBIND_IMPLS_INITIALIZED
    global _TENSOR_QUEUE_GLOBAL_TEST
    # 如果 TorchBind 实现已初始化，则直接返回
    if _TORCHBIND_IMPLS_INITIALIZED:
        return

    # 加载 TorchBind 测试库
    load_torchbind_test_lib()
    # 注册虚假操作符
    register_fake_operators()
    # 注册虚假类
    register_fake_classes()
    # 创建空的 Tensor 队列并赋值给全局变量
    _TENSOR_QUEUE_GLOBAL_TEST = _empty_tensor_queue()
    # 标记 TorchBind 实现已初始化
    _TORCHBIND_IMPLS_INITIALIZED = True


# 创建并返回空的 Tensor 队列的 Torch 脚本对象
def _empty_tensor_queue() -> torch.ScriptObject:
    return torch.classes._TorchScriptTesting._TensorQueue(
        torch.empty(
            0,
        ).fill_(-1)
    )


# 将以下操作放在函数中，因为对应的库可能尚未加载。
# 注册虚假操作符函数
def register_fake_operators():
    # 定义用于接收 foo 和 z 参数的虚假 takes_foo 操作
    @torch.library.register_fake("_TorchScriptTesting::takes_foo_python_meta")
    def fake_takes_foo(foo, z):
        return foo.add_tensor(z)

    # 定义虚假的队列弹出操作
    @torch.library.register_fake("_TorchScriptTesting::queue_pop")
    def fake_queue_pop(tq):
        return tq.pop()

    # 定义虚假的队列推入操作
    @torch.library.register_fake("_TorchScriptTesting::queue_push")
    def fake_queue_push(tq, x):
        return tq.push(x)

    # 定义虚假的队列大小查询操作
    @torch.library.register_fake("_TorchScriptTesting::queue_size")
    def fake_queue_size(tq):
        return tq.size()

    # 定义接收 foo 和 x 参数的返回列表的虚假操作
    def meta_takes_foo_list_return(foo, x):
        a = foo.add_tensor(x)
        b = foo.add_tensor(a)
        c = foo.add_tensor(b)
        return [a, b, c]

    # 定义接收 foo 和 x 参数的返回元组的虚假操作
    def meta_takes_foo_tuple_return(foo, x):
        a = foo.add_tensor(x)
        b = foo.add_tensor(a)
        return (a, b)

    # 将 meta_takes_foo_list_return 函数绑定到 takes_foo_list_return 操作上
    torch.ops._TorchScriptTesting.takes_foo_list_return.default.py_impl(
        torch._C.DispatchKey.Meta
    )(meta_takes_foo_list_return)

    # 将 meta_takes_foo_tuple_return 函数绑定到 takes_foo_tuple_return 操作上
    torch.ops._TorchScriptTesting.takes_foo_tuple_return.default.py_impl(
        torch._C.DispatchKey.Meta
    )(meta_takes_foo_tuple_return)

    # 将 lambda 函数绑定到 takes_foo 操作上
    torch.ops._TorchScriptTesting.takes_foo.default.py_impl(torch._C.DispatchKey.Meta)(
        lambda cc, x: cc.add_tensor(x)
    )


# 注册虚假类函数
def register_fake_classes():
    # 定义虚假的 _Foo 类
    @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
    class FakeFoo:
        # 构造函数，接收 x 和 y 参数
        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y

        # 类方法：将扁平化的 _Foo 对象恢复为 _Foo 类实例
        @classmethod
        def __obj_unflatten__(cls, flattend_foo):
            return cls(**dict(flattend_foo))

        # 添加张量的虚假操作
        def add_tensor(self, z):
            return (self.x + self.y) * z

    # 定义虚假的 _ContainsTensor 类
    @torch._library.register_fake_class("_TorchScriptTesting::_ContainsTensor")
    class FakeContainsTensor:
        # 构造函数，接收 t 参数
        def __init__(self, t: torch.Tensor):
            self.t = t

        # 类方法：将扁平化的 _ContainsTensor 对象恢复为 _ContainsTensor 类实例
        @classmethod
        def __obj_unflatten__(cls, flattend_foo):
            return cls(**dict(flattend_foo))

        # 获取张量的虚假操作
        def get(self):
            return self.t


# 加载 TorchBind 测试库函数
def load_torchbind_test_lib():
    import unittest
    # 从 torch.testing._internal.common_utils 导入必要的函数和变量，包括 find_library_location, IS_FBCODE, IS_MACOS, IS_SANDCASTLE, IS_WINDOWS
    from torch.testing._internal.common_utils import (
        find_library_location,  # 导入 find_library_location 函数
        IS_FBCODE,               # 导入 IS_FBCODE 变量
        IS_MACOS,                # 导入 IS_MACOS 变量
        IS_SANDCASTLE,           # 导入 IS_SANDCASTLE 变量
        IS_WINDOWS,              # 导入 IS_WINDOWS 变量
    )

    # 如果运行在 Sandcastle 环境或者 FBCODE 环境下
    if IS_SANDCASTLE or IS_FBCODE:
        # 载入指定的自定义类注册的测试库
        torch.ops.load_library("//caffe2/test/cpp/jit:test_custom_class_registrations")

    # 如果运行在 macOS 环境下
    elif IS_MACOS:
        # 抛出单元测试跳过异常，因为在测试中使用了非可移植的 load_library 调用
        raise unittest.SkipTest("non-portable load_library call used in test")

    # 如果运行在其他操作系统环境下
    else:
        # 使用 find_library_location 函数查找 libtorchbind_test.so 文件路径
        lib_file_path = find_library_location("libtorchbind_test.so")

        # 如果运行在 Windows 环境下
        if IS_WINDOWS:
            # 使用 find_library_location 函数查找 torchbind_test.dll 文件路径
            lib_file_path = find_library_location("torchbind_test.dll")

        # 载入指定的动态链接库，根据操作系统确定库文件路径
        torch.ops.load_library(str(lib_file_path))
# 定义一个上下文管理器函数，用于临时注册 Python 实现
@contextlib.contextmanager
def _register_py_impl_temporarily(op_overload, key, fn):
    try:
        # 在 op_overload 对象中注册指定 key 的 Python 实现为 fn
        op_overload.py_impl(key)(fn)
        # yield 表达式，暂停执行代码，将控制权交给调用方
        yield
    finally:
        # 在 finally 块中，清除已注册的 key 对应的 Python 内核
        del op_overload.py_kernels[key]
        # 清空 op_overload 对象的调度缓存
        op_overload._dispatch_cache.clear()
```