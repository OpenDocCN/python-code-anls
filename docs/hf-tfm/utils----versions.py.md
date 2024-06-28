# `.\utils\versions.py`

```py
# 版权声明和保留版权年份，标识代码版权归HuggingFace团队所有
#
# 根据Apache许可证2.0版（“许可证”）授权，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发的软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查阅许可证了解具体的语言和限制
"""
用于处理包版本的实用工具
"""

import importlib.metadata
import operator
import re
import sys
from typing import Optional

from packaging import version

# 操作符映射表，用于比较版本号
ops = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def _compare_versions(op, got_ver, want_ver, requirement, pkg, hint):
    # 如果获取到的版本号或期望的版本号为空，抛出数值错误
    if got_ver is None or want_ver is None:
        raise ValueError(
            f"Unable to compare versions for {requirement}: need={want_ver} found={got_ver}. This is unusual. Consider"
            f" reinstalling {pkg}."
        )
    # 使用操作符比较获取到的版本号和期望的版本号
    if not ops[op](version.parse(got_ver), version.parse(want_ver)):
        # 如果版本比较不符合要求，抛出导入错误
        raise ImportError(
            f"{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.{hint}"
        )


def require_version(requirement: str, hint: Optional[str] = None) -> None:
    """
    执行运行时的依赖版本检查，使用与pip相同的语法。

    安装的模块版本来自*site-packages*目录通过*importlib.metadata*。

    Args:
        requirement (`str`): pip风格的定义，例如 "tokenizers==0.9.4", "tqdm>=4.27", "numpy"
        hint (`str`, *optional*): 如果未满足要求，打印的建议内容

    Example:

    ```
    require_version("pandas>1.1.2")
    require_version("numpy>1.18.5", "this is important to have for whatever reason")
    ```"""

    hint = f"\n{hint}" if hint is not None else ""

    # 非版本化检查
    if re.match(r"^[\w_\-\d]+$", requirement):
        # 分解要求，获取包名、操作符和期望版本号
        pkg, op, want_ver = requirement, None, None
    # 如果要求不是以等号或不等号开始的，尝试查找匹配
    match = re.findall(r"^([^!=<>\s]+)([\s!=<>]{1,2}.+)", requirement)
    # 如果没有找到匹配项，抛出数值错误并显示详细信息
    if not match:
        raise ValueError(
            "requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but"
            f" got {requirement}"
        )
    # 从匹配结果中提取包名和完整的需求范围
    pkg, want_full = match[0]
    # 将完整的需求范围按逗号分隔，得到多个需求范围
    want_range = want_full.split(",")
    # 初始化一个空字典，用于存储需求操作符和版本号
    wanted = {}
    # 遍历每个需求范围
    for w in want_range:
        # 尝试匹配操作符和版本号
        match = re.findall(r"^([\s!=<>]{1,2})(.+)", w)
        # 如果没有找到匹配项，抛出数值错误并显示详细信息
        if not match:
            raise ValueError(
                "requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23,"
                f" but got {requirement}"
            )
        # 从匹配结果中提取操作符和版本号，并添加到 wanted 字典中
        op, want_ver = match[0]
        wanted[op] = want_ver
        # 如果操作符不在 ops 字典中，抛出数值错误并显示详细信息
        if op not in ops:
            raise ValueError(f"{requirement}: need one of {list(ops.keys())}, but got {op}")

    # 特殊情况处理，如果包名是 "python"
    if pkg == "python":
        # 获取当前 Python 解释器的版本号，并逐个比较所需的版本号范围
        got_ver = ".".join([str(x) for x in sys.version_info[:3]])
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
        return

    # 尝试获取已安装包的版本号
    try:
        got_ver = importlib.metadata.version(pkg)
    # 如果找不到包，则抛出包未找到错误，并显示相关提示信息
    except importlib.metadata.PackageNotFoundError:
        raise importlib.metadata.PackageNotFoundError(
            f"The '{requirement}' distribution was not found and is required by this application. {hint}"
        )

    # 如果指定了版本号或版本范围，则检查已安装的包是否满足要求的版本
    if want_ver is not None:
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
# 定义一个函数 require_version_core，它是对 require_version 的包装器，当版本要求未满足时会发出针对核心的提示信息
def require_version_core(requirement):
    # 提示信息，建议尝试更新 transformers 或安装开发环境依赖，特别是在使用 git 主分支时
    hint = "Try: `pip install transformers -U` or `pip install -e '.[dev]'` if you're working with git main"
    # 调用 require_version 函数，并传入版本要求和提示信息
    return require_version(requirement, hint)
```