# `.\DB-GPT-src\dbgpt\util\json_utils.py`

```py
# 导入必要的模块
import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
import re  # 导入正则表达式模块
from dataclasses import asdict, is_dataclass  # 导入用于数据类操作的函数
from datetime import date, datetime  # 导入日期和时间处理模块

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义默认的响应格式常量
LLM_DEFAULT_RESPONSE_FORMAT = "llm_response_format_1"

# 定义一个函数，用于序列化日期对象为 ISO 格式字符串
def serialize(obj):
    if isinstance(obj, date):
        return obj.isoformat()

# 扩展 JSON 编码器，支持数据类和日期时间对象的序列化
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)  # 如果是数据类对象，则转换为字典
        if isinstance(obj, datetime):
            return obj.isoformat()  # 如果是日期时间对象，则转换为 ISO 格式字符串
        return super().default(obj)

# 从 JSONDecodeError 消息中提取字符位置信息的函数
def extract_char_position(error_message: str) -> int:
    """Extract the character position from the JSONDecodeError message.

    Args:
        error_message (str): The error message from the JSONDecodeError
          exception.

    Returns:
        int: The character position.
    """
    char_pattern = re.compile(r"\(char (\d+)\)")  # 编译正则表达式，匹配字符位置信息
    if match := char_pattern.search(error_message):  # 搜索匹配
        return int(match[1])  # 返回字符位置的整数值
    else:
        raise ValueError("Character position not found in the error message.")  # 如果未找到字符位置信息，则抛出值错误异常

# 在文本中查找并解析所有的 JSON 对象
def find_json_objects(text):
    json_objects = []  # 存储解析后的 JSON 对象列表
    inside_string = False  # 标记是否在字符串内部
    escape_character = False  # 标记是否处于转义状态
    stack = []  # 栈，用于跟踪 JSON 对象的嵌套
    start_index = -1  # 记录 JSON 对象开始的索引位置

    for i, char in enumerate(text):
        # 处理转义字符
        if char == "\\" and not escape_character:
            escape_character = True
            continue

        # 切换 inside_string 标志
        if char == '"' and not escape_character:
            inside_string = not inside_string

        # 处理字符串内的换行和制表符
        if not inside_string and char == "\n":
            continue
        if inside_string and char == "\n":
            char = "\\n"
        if inside_string and char == "\t":
            char = "\\t"

        # 处理开放的大括号和中括号
        if char in "{[" and not inside_string:
            stack.append(char)
            if len(stack) == 1:
                start_index = i

        # 处理闭合的大括号和中括号
        if char in "}]" and not inside_string and stack:
            if (char == "}" and stack[-1] == "{") or (char == "]" and stack[-1] == "["):
                stack.pop()
                if not stack:
                    end_index = i + 1
                    try:
                        json_obj = json.loads(text[start_index:end_index])  # 尝试解析 JSON 对象
                        json_objects.append(json_obj)  # 解析成功则添加到列表中
                    except json.JSONDecodeError:
                        pass

        # 重置转义字符标志
        escape_character = False if escape_character else escape_character

    return json_objects

@staticmethod
def _format_json_str(jstr):
    """Remove newlines outside of quotes, and handle JSON escape sequences.

    ```
    # 这个函数移除 JSON 字符串中非引号包裹的换行符，以避免 json.loads(s) 解析失败。
    # 例如：
    # "{\n\"tool\": \"python\",\n\"query\": \"print('hello')\nprint('world')\"\n}"
    # 将转换为
    # "{\"tool\": \"python\",\"query\": \"print('hello')\nprint('world')\"}"
    # 这个函数也处理引号内部的 JSON 转义序列。
    # 例如：
    # '{"args": "a\na\na\ta"}' 将转换为 '{"args": "a\\na\\na\\ta"}'
    def remove_newlines_outside_quotes(jstr):
        result = []
        inside_quotes = False
        last_char = " "
        for char in jstr:
            if last_char != "\\" and char == '"':
                inside_quotes = not inside_quotes
            last_char = char
            if not inside_quotes and char == "\n":
                continue
            if inside_quotes and char == "\n":
                char = "\\n"
            if inside_quotes and char == "\t":
                char = "\\t"
            result.append(char)
        return "".join(result)
```