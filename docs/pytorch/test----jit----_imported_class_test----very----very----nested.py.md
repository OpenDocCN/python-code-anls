# `.\pytorch\test\jit\_imported_class_test\very\very\nested.py`

```py
import torch  # 导入 torch 库

# This file contains definitions of script classes.
# They are used by test_jit.py to test ScriptClass imports
# 该文件定义了脚本类的定义。
# 它们被 test_jit.py 使用以测试 ScriptClass 的导入

@torch.jit.script  # 使用 torch.jit.script 装饰器，声明这是一个 TorchScript 类
class FooUniqueName:  # 定义一个名为 FooUniqueName 的类
    def __init__(self, y):  # 类的初始化方法，接受参数 y
        self.y = y  # 将参数 y 赋值给实例变量 self.y
```