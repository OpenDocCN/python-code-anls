# `.\pytorch\torch\backends\xnnpack\__init__.py`

```py
# 启用类型提示允许
mypy: allow-untyped-defs

# 导入系统模块
import sys
# 导入 types 模块
import types

# 导入 PyTorch 库
import torch

# 定义一个描述符类 _XNNPACKEnabled
class _XNNPACKEnabled:
    # 获取属性时调用，返回 XNNPACK 是否启用的状态
    def __get__(self, obj, objtype):
        return torch._C._is_xnnpack_enabled()

    # 设置属性时调用，抛出运行时错误，不支持赋值操作
    def __set__(self, obj, val):
        raise RuntimeError("Assignment not supported")

# 定义 XNNPACKEngine 类，继承自 types.ModuleType
class XNNPACKEngine(types.ModuleType):
    # 初始化方法，接受模块对象 m 和模块名称 name
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    # 获取属性时调用，委托给原始模块对象 m 的同名属性
    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)

    # 定义类属性 enabled，使用 _XNNPACKEnabled 描述符
    enabled = _XNNPACKEnabled()

# 使用 sys.modules 替换技巧，将当前模块名称指向 XNNPACKEngine 类的实例
sys.modules[__name__] = XNNPACKEngine(sys.modules[__name__], __name__)
```