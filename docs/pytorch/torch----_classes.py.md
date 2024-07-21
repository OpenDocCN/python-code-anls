# `.\pytorch\torch\_classes.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类
import types

import torch._C

# 定义一个自定义模块类型 `_ClassNamespace`，继承自 `types.ModuleType`
class _ClassNamespace(types.ModuleType):
    def __init__(self, name):
        # 调用父类构造函数初始化模块类型，设置模块名称为 "torch.classes" + name
        super().__init__("torch.classes" + name)
        # 记录当前命名空间的名称
        self.name = name

    # 重载 `__getattr__` 方法，用于动态获取属性
    def __getattr__(self, attr):
        # 调用 Torch 库函数获取自定义类的 Python 包装器
        proxy = torch._C._get_custom_class_python_wrapper(self.name, attr)
        # 如果未找到对应的包装器，抛出运行时异常
        if proxy is None:
            raise RuntimeError(f"Class {self.name}.{attr} not registered!")
        # 返回获取到的包装器对象
        return proxy

# 定义一个 `_Classes` 类，也是一个自定义模块类型，继承自 `types.ModuleType`
class _Classes(types.ModuleType):
    # 指定模块的文件名为 "_classes.py"
    __file__ = "_classes.py"

    def __init__(self):
        # 调用父类构造函数初始化模块类型，设置模块名称为 "torch.classes"
        super().__init__("torch.classes")

    # 重载 `__getattr__` 方法，用于动态获取属性
    def __getattr__(self, name):
        # 创建一个 `_ClassNamespace` 实例，用给定名称初始化命名空间
        namespace = _ClassNamespace(name)
        # 将命名空间对象设置为当前模块的属性，属性名为 `name`
        setattr(self, name, namespace)
        # 返回刚创建的命名空间对象
        return namespace

    # 定义 `loaded_libraries` 属性，返回 Torch 操作模块中的 `loaded_libraries`
    @property
    def loaded_libraries(self):
        return torch.ops.loaded_libraries

    # 定义 `load_library` 方法，用于加载指定路径的共享库到当前进程
    def load_library(self, path):
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom classes with the PyTorch JIT runtime. This allows dynamically
        loading custom classes. For this, you should compile your class
        and the static registration code into a shared library object, and then
        call ``torch.classes.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.classes.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Args:
            path (str): A path to a shared library to load.
        """
        # 调用 Torch 操作模块中的 `load_library` 函数，加载指定路径的共享库
        torch.ops.load_library(path)

# 创建 `_Classes` 类的实例对象 `classes`，作为模块 `torch.classes` 的主入口点
classes = _Classes()
```