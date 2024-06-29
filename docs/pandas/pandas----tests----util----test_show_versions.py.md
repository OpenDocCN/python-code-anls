# `D:\src\scipysrc\pandas\pandas\tests\util\test_show_versions.py`

```
import json  # 导入处理 JSON 数据的模块
import os    # 导入操作系统相关功能的模块
import re    # 导入正则表达式模块

from pandas.util._print_versions import (  # 导入 pandas 的版本信息打印相关函数
    _get_dependency_info,   # 导入获取依赖信息的函数
    _get_sys_info,          # 导入获取系统信息的函数
)

import pandas as pd   # 导入 pandas 库并使用 pd 别名


def test_show_versions(tmpdir):
    # GH39701
    # 设置输出 JSON 文件的路径
    as_json = os.path.join(tmpdir, "test_output.json")

    # 调用 pandas 的版本信息展示函数，输出为 JSON 格式到指定文件
    pd.show_versions(as_json=as_json)

    with open(as_json, encoding="utf-8") as fd:
        # 检查文件输出是否为有效的 JSON 格式，若不是则抛出异常
        result = json.load(fd)

    # 构建预期的输出字典，包括系统信息和依赖信息
    expected = {
        "system": _get_sys_info(),
        "dependencies": _get_dependency_info(),
    }

    # 断言实际输出与预期输出一致
    assert result == expected


def test_show_versions_console_json(capsys):
    # GH39701
    # 调用 pandas 的版本信息展示函数，将输出格式设为 JSON，并捕获输出内容
    pd.show_versions(as_json=True)
    stdout = capsys.readouterr().out

    # 检查捕获的输出是否为有效的 JSON 格式
    result = json.loads(stdout)

    # 构建预期的输出字典，包括系统信息和依赖信息
    expected = {
        "system": _get_sys_info(),
        "dependencies": _get_dependency_info(),
    }

    # 断言实际输出与预期输出一致
    assert result == expected


def test_show_versions_console(capsys):
    # gh-32041
    # 调用 pandas 的版本信息展示函数，输出格式为控制台文本，并捕获输出内容
    pd.show_versions(as_json=False)
    result = capsys.readouterr().out

    # 检查输出内容中是否包含特定的标题
    assert "INSTALLED VERSIONS" in result

    # 检查输出内容中是否包含完整的提交哈希信息
    assert re.search(r"commit\s*:\s[0-9a-f]{40}\n", result)

    # 检查输出内容中是否包含必需依赖项的版本信息
    assert re.search(r"numpy\s*:\s[0-9]+\..*\n", result)

    # 检查输出内容中是否包含可选依赖项的版本信息
    assert re.search(r"pyarrow\s*:\s([0-9]+.*|None)\n", result)


def test_json_output_match(capsys, tmpdir):
    # GH39701
    # 调用 pandas 的版本信息展示函数，将输出格式设为 JSON，并捕获控制台输出内容
    pd.show_versions(as_json=True)
    result_console = capsys.readouterr().out

    # 设置输出 JSON 文件的路径，并调用 pandas 的版本信息展示函数输出到文件
    out_path = os.path.join(tmpdir, "test_json.json")
    pd.show_versions(as_json=out_path)
    
    # 读取输出到文件的 JSON 内容
    with open(out_path, encoding="utf-8") as out_fd:
        result_file = out_fd.read()

    # 断言控制台输出与文件输出内容一致
    assert result_console == result_file
```