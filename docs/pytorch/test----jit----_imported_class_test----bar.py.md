# `.\pytorch\test\jit\_imported_class_test\bar.py`

```py
import torch

# 导入 torch 库，用于使用 PyTorch 功能

# 这个文件包含了脚本类的定义。
# 它们被 test_jit.py 使用来测试 ScriptClass 的导入功能

@torch.jit.script  # 标记下面的类为 Torch 脚本类，用于 JIT 编译
class FooSameName:  # 定义一个名为 FooSameName 的类
    def __init__(self, y):
        self.y = y  # 在构造函数中初始化实例变量 y
```