# `.\agent\Agent_OpenAI\scripts\utils.py`

```
# 导入json模块，用于处理JSON格式数据
import json
# 从scripts.tool模块中导入Tool类，用于处理工具相关的逻辑
from scripts.tool import Tool

# 解析响应对象中的函数参数，并返回解析后的字典
def parse_function_args(response):
    # 从响应对象中获取第一个选择项的消息内容
    message = response.choices[0].message
    # 解析消息中第一个工具调用的函数参数并返回解析后的字典
    return json.loads(message.tool_calls[0].function.arguments)

# 根据响应对象和工具列表获取特定工具对象
def get_tool_from_response(response, tools: list[Tool]):
    # 获取响应中第一个工具调用的函数名
    tool_name = response.choices[0].message.tool_calls[0].function.name
    # 在给定的工具列表中查找并返回与工具名匹配的工具对象
    for t in tools:
        if t.name == tool_name:
            return t
    # 若未找到对应工具，则引发值错误异常
    raise ValueError(f"Tool {tool_name} not found in tools list.")

# 根据响应对象和工具列表运行特定工具，并返回运行结果
def run_tool_from_response(response, tools: list[Tool]):
    # 获取响应对象中对应的工具对象
    tool = get_tool_from_response(response, tools)
    # 解析函数参数，以字典形式获取工具的关键字参数
    tool_kwargs = parse_function_args(response)
    # 使用获取的关键字参数运行工具，并返回运行结果
    return tool.run(**tool_kwargs)
```