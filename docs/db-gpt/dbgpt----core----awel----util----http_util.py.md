# `.\DB-GPT-src\dbgpt\core\awel\util\http_util.py`

```py
"""HTTP utilities."""

# 导入正则表达式模块
import re

# 将多个路径连接成一个路径
def join_paths(*paths):
    """Join multiple paths into one path.

    Delete the spaces and slashes at both ends of each path, and ensure that there is
    only one slash between the paths.
    """
    # 使用列表推导式，处理每个路径：去除开头和结尾的空格和斜杠，但保证至少有一个斜杠
    stripped_paths = [
        re.sub(r"^[/\s]+|[/\s]+$", "", path) for path in paths if path.strip("/")
    ]
    # 将处理后的路径用单个斜杠连接成完整路径
    full_path = "/".join(stripped_paths)
    # 使用正则表达式，将多个连续的斜杠替换为单个斜杠
    return re.sub(r"/{2,}", "/", "/" + full_path)
```