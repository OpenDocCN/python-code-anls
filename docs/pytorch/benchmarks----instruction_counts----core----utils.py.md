# `.\pytorch\benchmarks\instruction_counts\core\utils.py`

```
# 导入必要的模块
import atexit  # 用于注册退出时执行的清理函数
import re  # 正则表达式模块，用于字符串匹配和操作
import shutil  # 文件操作模块，提供了高级的文件和文件夹操作功能
import textwrap  # 文本处理模块，用于格式化文本段落
from typing import List, Optional, Tuple  # 引入类型提示相关的类和函数

from core.api import GroupedBenchmark, TimerArgs  # 导入来自core.api的特定类
from core.types import Definition, FlatIntermediateDefinition, Label  # 导入来自core.types的特定类

from torch.utils.benchmark.utils.common import _make_temp_dir  # 导入torch库中的_make_temp_dir函数

_TEMPDIR: Optional[str] = None  # 声明一个可选的全局变量_TEMPDIR，用于存储临时目录路径


def get_temp_dir() -> str:
    global _TEMPDIR
    # 如果临时目录路径为空，创建临时目录，并注册退出时删除该目录的操作
    if _TEMPDIR is None:
        _TEMPDIR = _make_temp_dir(
            prefix="instruction_count_microbenchmarks", gc_dev_shm=True
        )
        atexit.register(shutil.rmtree, path=_TEMPDIR)
    return _TEMPDIR  # 返回临时目录路径


def _flatten(
    key_prefix: Label, sub_schema: Definition, result: FlatIntermediateDefinition
) -> None:
    # 递归地将嵌套的sub_schema展平，并存储在result中
    for k, value in sub_schema.items():
        if isinstance(k, tuple):
            assert all(isinstance(ki, str) for ki in k)
            key_suffix: Label = k
        elif k is None:
            key_suffix = ()
        else:
            assert isinstance(k, str)
            key_suffix = (k,)
        
        key: Label = key_prefix + key_suffix  # 组合前缀和后缀，形成完整的键
        if isinstance(value, (TimerArgs, GroupedBenchmark)):
            assert key not in result, f"duplicate key: {key}"
            result[key] = value  # 如果值是TimerArgs或GroupedBenchmark类型，则将键值对存入result
        else:
            assert isinstance(value, dict)
            _flatten(key_prefix=key, sub_schema=value, result=result)  # 递归调用自身处理嵌套的字典


def flatten(schema: Definition) -> FlatIntermediateDefinition:
    """See types.py for an explanation of nested vs. flat definitions."""
    result: FlatIntermediateDefinition = {}  # 初始化结果字典
    _flatten(key_prefix=(), sub_schema=schema, result=result)  # 调用_flatten函数处理给定的schema

    # 确保生成的结果字典是有效的扁平定义
    for k, v in result.items():
        assert isinstance(k, tuple)
        assert all(isinstance(ki, str) for ki in k)
        assert isinstance(v, (TimerArgs, GroupedBenchmark))
    return result  # 返回扁平化后的结果字典


def parse_stmts(stmts: str) -> Tuple[str, str]:
    """Helper function for side-by-side Python and C++ stmts.

    For more complex statements, it can be useful to see Python and C++ code
    side by side. To this end, we provide an **extremely restricted** way
    to define Python and C++ code side-by-side. The schema should be mostly
    self explanatory, with the following non-obvious caveats:
      - Width for the left (Python) column MUST be 40 characters.
      - The column separator is " | ", not "|". Whitespace matters.
    """
    stmts = textwrap.dedent(stmts).strip()  # 去除字符串开头的空白并删除末尾的空行
    lines: List[str] = stmts.splitlines(keepends=False)  # 按行分割字符串为列表，保留行尾换行符

    assert len(lines) >= 3, f"Invalid string:\n{stmts}"  # 断言至少有三行，用于验证字符串是否有效

    column_header_pattern = r"^Python\s{35}\| C\+\+(\s*)$"  # 匹配列标题的正则表达式模式
    signature_pattern = r"^: f\((.*)\)( -> (.+))?\s*$"  # 匹配函数签名的正则表达式模式
    separation_pattern = r"^[-]{40} | [-]{40}$"  # 匹配分隔线的正则表达式模式
    code_pattern = r"^(.{40}) \|($| (.*)$)"  # 匹配代码行的正则表达式模式

    column_match = re.search(column_header_pattern, lines[0])  # 在第一行匹配列标题模式
    if column_match is None:
        raise ValueError(
            f"Column header `{lines[0]}` "
            f"does not match pattern `{column_header_pattern}`"
        )
    # 断言确保在第二行中能够找到与分隔模式匹配的内容
    assert re.search(separation_pattern, lines[1])

    # 初始化 Python 代码行和 C++ 代码行的空列表
    py_lines: List[str] = []
    cpp_lines: List[str] = []

    # 遍历从第三行开始的所有输入行
    for l in lines[2:]:
        # 使用正则表达式匹配当前行的代码模式
        l_match = re.search(code_pattern, l)
        # 如果匹配结果为 None，则抛出 ValueError 异常，指明当前行无效
        if l_match is None:
            raise ValueError(f"Invalid line `{l}`")
        
        # 将 Python 代码片段添加到 py_lines 列表中（第一个捕获组）
        py_lines.append(l_match.groups()[0])
        # 将 C++ 代码片段添加到 cpp_lines 列表中（第三个捕获组，如果不存在则为空字符串）
        cpp_lines.append(l_match.groups()[2] or "")

        # 确保 Python 代码和 C++ 代码的组合可以正确地还原当前行内容，以确保正确性
        l_from_stmts = f"{py_lines[-1]:<40} | {cpp_lines[-1]:<40}".rstrip()
        assert l_from_stmts == l.rstrip(), f"Failed to round trip `{l}`"

    # 返回 Python 代码和 C++ 代码的字符串形式，各自以换行符连接
    return "\n".join(py_lines), "\n".join(cpp_lines)
```