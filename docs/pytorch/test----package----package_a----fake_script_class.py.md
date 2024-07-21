# `.\pytorch\test\package\package_a\fake_script_class.py`

```py
# 导入类型提示中的 Any 类型
from typing import Any

# 导入 PyTorch 库
import torch

# 使用 torch.jit.script 装饰器将类声明为 Torch 脚本
@torch.jit.script
class MyScriptClass:
    """Intended to be scripted."""

    # 类的初始化方法，接受参数 x
    def __init__(self, x):
        # 将参数 x 赋值给实例变量 foo
        self.foo = x

    # 修改实例变量 foo 的方法，接受参数 x
    def set_foo(self, x):
        self.foo = x


# 使用 torch.jit.script 装饰器将函数声明为 Torch 脚本
@torch.jit.script
def uses_script_class(x):
    """Intended to be scripted."""
    # 创建 MyScriptClass 的实例 foo，传入参数 x
    foo = MyScriptClass(x)
    # 返回 MyScriptClass 实例的 foo 属性
    return foo.foo


# 定义 IdListFeature 类
class IdListFeature:
    # 类的初始化方法
    def __init__(self):
        # 初始化实例变量 id_list 为一个全为 1 的张量
        self.id_list = torch.ones(1, 1)

    # 返回自身实例的方法，返回类型为 "IdListFeature"
    def returns_self(self) -> "IdListFeature":
        return IdListFeature()


# 定义 UsesIdListFeature 类，继承自 torch.nn.Module
class UsesIdListFeature(torch.nn.Module):
    # 前向传播方法，接受参数 feature，类型为 Any
    def forward(self, feature: Any):
        # 检查 feature 是否为 IdListFeature 的实例
        if isinstance(feature, IdListFeature):
            # 若是 IdListFeature 的实例，则返回其 id_list 属性
            return feature.id_list
        else:
            # 若不是，则原样返回 feature
            return feature
```