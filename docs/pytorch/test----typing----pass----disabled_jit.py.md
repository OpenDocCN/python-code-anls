# `.\pytorch\test\typing\pass\disabled_jit.py`

```py
# 引入必要的枚举(Enum)类
from enum import Enum
# 引入类型相关的模块
from typing import Type, TypeVar
# 引入参数规范(ParamSpec)类
from typing_extensions import assert_never, assert_type, ParamSpec

# 引入 pytest 测试框架
import pytest

# 引入 PyTorch 中的核心模块
from torch import jit, nn, ScriptDict, ScriptFunction, ScriptList

# 定义一个参数规范 P
P = ParamSpec("P")
# 定义一个类型变量 R，支持协变
R = TypeVar("R", covariant=True)

# 定义一个枚举类 Color，包含三种颜色
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

# 将 Color 枚举类转换为 Torch 的脚本枚举
assert_type(jit.script(Color), Type[Color])

# 将字典 {1: 1} 转换为 Torch 的脚本字典(ScriptDict)
assert_type(jit.script({1: 1}), ScriptDict)

# 将列表 [0] 转换为 Torch 的脚本列表(ScriptList)
assert_type(jit.script([0]), ScriptList)

# 创建一个线性模型，并将其转换为 Torch 的脚本模块(ScriptModule)
scripted_module = jit.script(nn.Linear(2, 2))
assert_type(scripted_module, jit.RecursiveScriptModule)

# 创建一个 ReLU 函数，并将其转换为 Torch 的脚本函数(ScriptFunction)
# 注意：由于参数名问题，不能使用 assert_type 进行断言
# 注意：泛型用法仅在 Python 3.9 及以上版本支持
relu: ScriptFunction = jit.script(nn.functional.relu)

# 尝试将 nn.Linear 类的模块转换为脚本时，会抛出 RuntimeError
with pytest.raises(RuntimeError):
    assert_never(jit.script(nn.Linear))
```