# `.\pytorch\torchgen\api\meta.py`

```
# 从 torchgen.model 模块中导入 NativeFunctionsGroup 类
from torchgen.model import NativeFunctionsGroup

# 遵循分派调用约定，但有一些例外：
#   - 不允许可变参数。元函数总是以函数形式编写。
#     可以查看 FunctionSchema.signature()。
#   - 不返回张量；而是返回描述相应张量的 TensorMeta 对象。

# 定义一个函数 name，接受一个 NativeFunctionsGroup 类型的参数 g，返回一个字符串
def name(g: NativeFunctionsGroup) -> str:
    # 使用函数式版本的重载名称
    # 将 g.functional.func.name 转换为字符串，将其中的点号（.）替换为下划线（_）
    return str(g.functional.func.name).replace(".", "_")
```