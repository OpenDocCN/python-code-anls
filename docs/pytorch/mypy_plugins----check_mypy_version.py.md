# `.\pytorch\mypy_plugins\check_mypy_version.py`

```
# 导入正则表达式模块
import re
# 导入系统模块
import sys
# 从路径操作模块中导入 Path 类
from pathlib import Path
# 从 mypy.plugin 模块中导入 Plugin 类
from mypy.plugin import Plugin


# 定义函数以获取正确的 mypy 版本
def get_correct_mypy_version():
    # 读取指定路径下的文本内容，并通过正则表达式查找匹配的 mypy 版本信息
    (match,) = re.finditer(
        r"mypy==(\d+(?:\.\d+)*)",
        (
            Path(__file__).parent.parent / ".ci" / "docker" / "requirements-ci.txt"
        ).read_text(),
    )
    # 获取匹配到的 mypy 版本号
    (version,) = match.groups()
    return version


# 定义插件函数，接收一个版本号字符串作为参数
def plugin(version: str):
    # 获取正确的 mypy 版本号
    correct_version = get_correct_mypy_version()
    # 如果传入的版本号与正确的 mypy 版本号不匹配
    if version != correct_version:
        # 打印错误消息，提示用户需要切换到正确的 mypy 版本
        print(
            f"""\
You are using mypy version {version}, which is not supported
in the PyTorch repo. Please switch to mypy version {correct_version}.

For example, if you installed mypy via pip, run this:

    pip install mypy=={correct_version}

Or if you installed mypy via conda, run this:

    conda install -c conda-forge mypy={correct_version}
""",
            # 将错误消息输出到标准错误流
            file=sys.stderr,
        )
    # 返回 Plugin 类型对象
    return Plugin
```