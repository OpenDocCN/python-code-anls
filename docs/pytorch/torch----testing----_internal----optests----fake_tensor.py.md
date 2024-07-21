# `.\pytorch\torch\testing\_internal\optests\fake_tensor.py`

```py
# 忽略类型检查错误，针对mypy工具的设置
# 导入torch._subclasses模块，这是一个下划线开头的内部模块，不建议直接使用
import torch._subclasses

# 定义一个函数is_builtin，判断操作op是否属于特定的命名空间
def is_builtin(op):
    # 判断操作op的命名空间是否在('aten', 'prims', 'prim')之中
    return op.namespace in ('aten', 'prims', 'prim')

# 定义一个函数fake_check，模拟检查操作op的有效性
def fake_check(op, args, kwargs):
    # 使用torch._subclasses.CrossRefFakeMode上下文管理器，忽略操作函数是内置函数的情况
    with torch._subclasses.CrossRefFakeMode(ignore_op_fn=is_builtin):
        # 调用操作op，并传入参数args和kwargs
        op(*args, **kwargs)
```