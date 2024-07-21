# `.\pytorch\tools\setup_helpers\cmake_utils.py`

```
"""
This is refactored from cmake.py to avoid circular imports issue with env.py,
which calls get_cmake_cache_variables_from_file
"""

# 引入将来版本的注解支持
from __future__ import annotations

# 引入正则表达式模块和类型相关的模块
import re
from typing import IO, Optional, Union

# 定义 CMakeValue 类型别名，表示可能是布尔值或字符串的可选类型
CMakeValue = Optional[Union[bool, str]]


# 定义函数，将 CMake 格式的值转换为对应的 Python 值
def convert_cmake_value_to_python_value(
    cmake_value: str, cmake_type: str
) -> CMakeValue:
    r"""Convert a CMake value in a string form to a Python value.

    Args:
      cmake_value (string): The CMake value in a string form (e.g., "ON", "OFF", "1").
      cmake_type (string): The CMake type of :attr:`cmake_value`.

    Returns:
      A Python value corresponding to :attr:`cmake_value` with type :attr:`cmake_type`.
    """

    # 将类型转换为大写
    cmake_type = cmake_type.upper()
    up_val = cmake_value.upper()
    if cmake_type == "BOOL":
        # 检查是否为布尔类型，根据 CMake 的布尔值定义进行转换
        return not (
            up_val in ("FALSE", "OFF", "N", "NO", "0", "", "NOTFOUND")
            or up_val.endswith("-NOTFOUND")
        )
    elif cmake_type == "FILEPATH":
        # 如果类型为文件路径，则根据特定规则处理
        if up_val.endswith("-NOTFOUND"):
            return None
        else:
            return cmake_value
    else:
        # 其他类型直接返回原始值
        return cmake_value


# 定义函数，从 CMakeCache.txt 文件中获取变量及其值，存入字典中
def get_cmake_cache_variables_from_file(
    cmake_cache_file: IO[str],
) -> dict[str, CMakeValue]:
    r"""Gets values in CMakeCache.txt into a dictionary.

    Args:
      cmake_cache_file: A CMakeCache.txt file object.
    Returns:
      dict: A ``dict`` containing the value of cached CMake variables.
    """

    # 初始化结果字典
    results = {}
    
    # 逐行处理 CMakeCache.txt 文件
    for i, line in enumerate(cmake_cache_file, 1):
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            # 如果是空行或注释行，则跳过
            continue

        # 使用正则表达式匹配每行的变量名、类型和值
        matched = re.match(
            r'("?)(.+?)\1(?::\s*([a-zA-Z_-][a-zA-Z0-9_-]*)?)?\s*=\s*(.*)', line
        )
        if matched is None:
            # 如果匹配失败，则抛出异常
            raise ValueError(f"Unexpected line {i} in {repr(cmake_cache_file)}: {line}")
        
        # 提取匹配的组件：变量名、类型和值
        _, variable, type_, value = matched.groups()
        
        # 如果类型未指定，则设为空字符串
        if type_ is None:
            type_ = ""
        
        # 如果类型为 INTERNAL 或 STATIC，则跳过（这些是 CMake 的内部变量）
        if type_.upper() in ("INTERNAL", "STATIC"):
            continue
        
        # 将变量名和转换后的值存入结果字典
        results[variable] = convert_cmake_value_to_python_value(value, type_)

    # 返回最终的结果字典
    return results
```