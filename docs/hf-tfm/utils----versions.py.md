# `.\transformers\utils\versions.py`

```
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证版本 2.0 授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制

"""
与包版本相关的实用工具
"""

# 导入必要的库
import importlib.metadata
import operator
import re
import sys
from typing import Optional

from packaging import version

# 操作符映射表
ops = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}

# 比较版本号的函数
def _compare_versions(op, got_ver, want_ver, requirement, pkg, hint):
    # 如果版本号为空，则抛出异常
    if got_ver is None or want_ver is None:
        raise ValueError(
            f"Unable to compare versions for {requirement}: need={want_ver} found={got_ver}. This is unusual. Consider"
            f" reinstalling {pkg}."
        )
    # 使用操作符比较版本号
    if not ops[op](version.parse(got_ver), version.parse(want_ver)):
        raise ImportError(
            f"{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.{hint}"
        )

# 检查依赖版本的函数
def require_version(requirement: str, hint: Optional[str] = None) -> None:
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.

    The installed module version comes from the *site-packages* dir via *importlib.metadata*.

    Args:
        requirement (`str`): pip style definition, e.g.,  "tokenizers==0.9.4", "tqdm>=4.27", "numpy"
        hint (`str`, *optional*): what suggestion to print in case of requirements not being met

    Example:

    ```python
    require_version("pandas>1.1.2")
    require_version("numpy>1.18.5", "this is important to have for whatever reason")
    ```"""

    # 如果有提示信息，则添加到提示字符串中
    hint = f"\n{hint}" if hint is not None else ""

    # 非版本化检查
    if re.match(r"^[\w_\-\d]+$", requirement):
        pkg, op, want_ver = requirement, None, None
    # 如果不是特殊情况，解析要求的包名和版本范围
    else:
        # 使用正则表达式匹配包名和版本范围
        match = re.findall(r"^([^!=<>\s]+)([\s!=<>]{1,2}.+)", requirement)
        # 如果没有匹配到结果，抛出数值错误
        if not match:
            raise ValueError(
                "requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but"
                f" got {requirement}"
            )
        # 提取包名和完整的版本范围
        pkg, want_full = match[0]
        # 将版本范围拆分成多个要求
        want_range = want_full.split(",")  # there could be multiple requirements
        # 初始化一个空字典用于存储要求的操作符和版本号
        wanted = {}
        # 遍历每个版本要求
        for w in want_range:
            # 使用正则表达式匹配操作符和版本号
            match = re.findall(r"^([\s!=<>]{1,2})(.+)", w)
            # 如果没有匹配到结果，抛出数值错误
            if not match:
                raise ValueError(
                    "requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23,"
                    f" but got {requirement}"
                )
            # 提取操作符和版本号，存入字典
            op, want_ver = match[0]
            wanted[op] = want_ver
            # 如果操作符不在允许的操作符列表中，抛出数值错误
            if op not in ops:
                raise ValueError(f"{requirement}: need one of {list(ops.keys())}, but got {op}")

    # 处理特殊情况，如果包名为"python"
    if pkg == "python":
        # 获取当前 Python 解释器的版本号
        got_ver = ".".join([str(x) for x in sys.version_info[:3]])
        # 比较当前版本号和要求的版本号
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
        return

    # 检查是否安装了任何版本的包
    try:
        # 获取已安装包的版本号
        got_ver = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        # 如果包未找到，抛出包未找到错误
        raise importlib.metadata.PackageNotFoundError(
            f"The '{requirement}' distribution was not found and is required by this application. {hint}"
        )

    # 检查是否安装了正确版本的包，如果提供了版本号或版本范围
    if want_ver is not None:
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
# 定义一个函数，用于包装 require_version 函数，在失败时发出与核心相关的提示
def require_version_core(requirement):
    # 提示信息，指导用户如何解决版本不匹配的问题
    hint = "Try: `pip install transformers -U` or `pip install -e '.[dev]'` if you're working with git main"
    # 调用 require_version 函数，传入要求的版本和提示信息
    return require_version(requirement, hint)
```