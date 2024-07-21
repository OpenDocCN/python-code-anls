# `.\pytorch\test\jit\_imported_class_test\foo.py`

```
import torch  # 导入 PyTorch 库

from . import bar  # 导入当前目录下的 bar 模块

# 本文件包含脚本类的定义。
# 这些类被 test_jit.py 使用来测试脚本类的导入功能。

@torch.jit.script  # 使用 PyTorch 的脚本装饰器声明下面的类是一个脚本类
class FooSameName:
    def __init__(self, x):
        self.x = x  # 初始化实例属性 x
        self.nested = bar.FooSameName(x)  # 初始化实例属性 nested，调用 bar 模块中的 FooSameName 类
```