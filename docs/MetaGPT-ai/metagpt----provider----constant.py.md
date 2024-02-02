# `MetaGPT\metagpt\provider\constant.py`

```py

# 定义了一个通用函数的模式，用于描述执行代码的函数
GENERAL_FUNCTION_SCHEMA = {
    "name": "execute",  # 函数名称为 execute
    "description": "Executes code on the user's machine, **in the users local environment**, and returns the output",  # 描述函数作用
    "parameters": {  # 函数参数
        "type": "object",  # 参数类型为对象
        "properties": {  # 参数属性
            "language": {  # 编程语言
                "type": "string",  # 类型为字符串
                "description": "The programming language (required parameter to the `execute` function)",  # 描述编程语言参数
                "enum": [  # 枚举值，表示可选的编程语言
                    "python",
                    "R",
                    "shell",
                    "applescript",
                    "javascript",
                    "html",
                    "powershell",
                ],
            },
            "code": {"type": "string", "description": "The code to execute (required)"},  # 代码参数，描述执行的代码
        },
        "required": ["language", "code"],  # 必需的参数
    },
}

# 定义了一个通用工具选择的模式，用于描述选择执行代码的工具
GENERAL_TOOL_CHOICE = {"type": "function", "function": {"name": "execute"}}  # 工具类型为函数，函数名称为 execute

```