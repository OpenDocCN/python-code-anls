# `.\AutoGPT\autogpts\autogpt\autogpt\models\command_parameter.py`

```py
# 导入 dataclasses 模块，用于创建数据类
import dataclasses

# 从 autogpt.core.utils.json_schema 模块中导入 JSONSchema 类
from autogpt.core.utils.json_schema import JSONSchema

# 定义一个数据类 CommandParameter，包含两个字段：name 和 spec
@dataclasses.dataclass
class CommandParameter:
    # 参数名
    name: str
    # 参数规范，使用 JSONSchema 类型
    spec: JSONSchema

    # 定义 __repr__ 方法，返回对象的字符串表示
    def __repr__(self):
        # 返回对象的字符串表示，包括参数名、参数类型、参数描述和是否必需
        return "CommandParameter('%s', '%s', '%s', %s)" % (
            self.name,
            self.spec.type,
            self.spec.description,
            self.spec.required,
        )
```