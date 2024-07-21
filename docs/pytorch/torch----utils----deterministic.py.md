# `.\pytorch\torch\utils\deterministic.py`

```
# 设置类型检查时允许未声明的函数和方法
# 导入 sys 模块，用于访问和修改 Python 运行时的参数和函数
# 导入 types 模块，用于操作 Python 中的类型和类对象
import sys
import types

# 导入 PyTorch 模块，用于机器学习和深度学习任务
import torch

# 定义一个名为 _Deterministic 的类，继承自 types.ModuleType 类型
class _Deterministic(types.ModuleType):
    
    # 定义 fill_uninitialized_memory 属性的 getter 方法
    @property
    def fill_uninitialized_memory(self):
        """
        是否在设置 torch.use_deterministic_algorithms() 为 True 时，
        用已知值填充未初始化的内存。
        """
        return torch._C._get_deterministic_fill_uninitialized_memory()

    # 定义 fill_uninitialized_memory 属性的 setter 方法
    @fill_uninitialized_memory.setter
    def fill_uninitialized_memory(self, mode):
        """
        设置当 torch.use_deterministic_algorithms() 为 True 时，
        填充未初始化的内存的模式。
        """
        return torch._C._set_deterministic_fill_uninitialized_memory(mode)

# 将当前模块的 __class__ 属性设置为 _Deterministic 类，
# 使得当前模块具有 _Deterministic 类的行为和属性
sys.modules[__name__].__class__ = _Deterministic
```