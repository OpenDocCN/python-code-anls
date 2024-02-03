# `.\AutoGPT\autogpts\autogpt\tests\mocks\mock_commands.py`

```py
# 导入命令装饰器和 JSON 模式
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema

# 定义命令所属的类别
COMMAND_CATEGORY = "mock"

# 使用装饰器定义一个命令
@command(
    "function_based_cmd",  # 命令名称
    "Function-based test command",  # 命令描述
    {
        "arg1": JSONSchema(  # 参数1的 JSON 模式定义
            type=JSONSchema.Type.INTEGER,  # 参数类型为整数
            description="arg 1",  # 参数描述
            required=True,  # 参数必需
        ),
        "arg2": JSONSchema(  # 参数2的 JSON 模式定义
            type=JSONSchema.Type.STRING,  # 参数类型为字符串
            description="arg 2",  # 参数描述
            required=True,  # 参数必需
        ),
    },
)
# 定义一个函数式的测试命令，接受一个整数和一个字符串参数，返回一个字符串
def function_based_cmd(arg1: int, arg2: str) -> str:
    """A function-based test command.

    Returns:
        str: the two arguments separated by a dash.
    """
    # 返回参数1和参数2以破折号分隔的字符串
    return f"{arg1} - {arg2}"
```