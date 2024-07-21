# `.\pytorch\functorch\dim\batch_tensor.py`

```py
# 导入需要的模块：contextlib 中的 contextmanager，用于创建上下文管理器
# 从 torch._C._functorch 中导入 _vmap_add_layers 和 _vmap_remove_layers 函数
# _enabled 变量用于表示是否启用了层次映射功能，初始设为 False
from contextlib import contextmanager

from torch._C._functorch import _vmap_add_layers, _vmap_remove_layers

# 定义全局变量 _enabled，用于标识是否启用了层次映射功能，默认为 False
_enabled = False

# 定义上下文管理器函数 _enable_layers，接受一个参数 dims
@contextmanager
def _enable_layers(dims):
    global _enabled  # 声明在函数内部使用全局变量 _enabled
    assert not _enabled  # 断言 _enabled 必须为 False，否则抛出异常

    # 将 dims 中非整数的元素按照 _level 和 size 进行排序，存入 input 变量
    input = sorted((d._level, d.size) for d in dims if not isinstance(d, int))
    n = len(input)  # 计算 input 的长度，存入 n 变量

    try:
        _vmap_add_layers(input)  # 调用 _vmap_add_layers 函数，添加层次映射的层次
        _enabled = True  # 将 _enabled 设置为 True，表示层次映射功能已启用
        yield  # 通过 yield 将控制权交给调用者，即此时在 with 语句中的代码块
    finally:
        _enabled = False  # 无论如何，最终将 _enabled 设置为 False
        _vmap_remove_layers(n)  # 调用 _vmap_remove_layers 函数，移除指定数量的层次映射层次
```