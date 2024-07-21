# `.\pytorch\torch\jit\_ir_utils.py`

```py
# 声明一个类型提示，允许未类型化的函数定义在mypy检查中
# 导入必要的模块
from typing import Union

# 导入PyTorch库
import torch

# 插入点类，用于管理图中的插入点
class _InsertPoint:
    def __init__(
        self,
        insert_point_graph: torch._C.Graph,
        insert_point: Union[torch._C.Node, torch._C.Block],
    ):
        # 初始化插入点对象
        self.insert_point = insert_point  # 设置插入点
        self.g = insert_point_graph  # 设置插入点所属的图
        self.guard = None  # 初始化保护属性为None

    # 进入上下文管理器时调用的方法
    def __enter__(self):
        # 保存当前的插入点
        self.prev_insert_point = self.g.insertPoint()
        # 设置新的插入点为当前插入点
        self.g.setInsertPoint(self.insert_point)

    # 退出上下文管理器时调用的方法
    def __exit__(self, *args):
        # 恢复之前保存的插入点
        self.g.setInsertPoint(self.prev_insert_point)


# 插入点保护函数，返回一个_InsertPoint实例
def insert_point_guard(self, insert_point: Union[torch._C.Node, torch._C.Block]):
    return _InsertPoint(self, insert_point)
```