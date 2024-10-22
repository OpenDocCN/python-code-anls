# `.\diffusers\utils\versions.py`

```py
# 版权声明，注明该代码由 HuggingFace 团队所有，并保留所有权利
# 
# 根据 Apache 许可证第 2.0 版授权；使用此文件需遵守许可证。
# 可在以下网址获取许可证：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则按“原样”分发本软件，不提供任何形式的担保或条件。
# 请参阅许可证以了解特定语言约束和
# 限制。
"""
用于处理包版本的实用工具
"""

# 导入必要的库和模块
import importlib.metadata  # 用于获取已安装模块的元数据
import operator  # 提供基本操作符的功能
import re  # 正则表达式模块，用于字符串匹配
import sys  # 系统模块，提供与Python解释器交互的功能
from typing import Optional  # 用于类型提示

from packaging import version  # 导入版本管理工具

# 定义一个操作符字典，将字符串操作符映射到相应的函数
ops = {
    "<": operator.lt,  # 小于
    "<=": operator.le,  # 小于或等于
    "==": operator.eq,  # 等于
    "!=": operator.ne,  # 不等于
    ">=": operator.ge,  # 大于或等于
    ">": operator.gt,  # 大于
}

# 定义一个函数用于比较版本号
def _compare_versions(op, got_ver, want_ver, requirement, pkg, hint):
    # 检查获取的版本或想要的版本是否为 None
    if got_ver is None or want_ver is None:
        raise ValueError(
            f"Unable to compare versions for {requirement}: need={want_ver} found={got_ver}. This is unusual. Consider"
            f" reinstalling {pkg}."
        )  # 抛出异常并提示用户
    # 使用操作符字典比较版本
    if not ops[op](version.parse(got_ver), version.parse(want_ver)):
        raise ImportError(
            f"{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.{hint}"
        )  # 抛出导入异常并提供错误提示

# 定义一个函数检查依赖版本
def require_version(requirement: str, hint: Optional[str] = None) -> None:
    """
    运行时检查依赖版本，使用 pip 的相同语法。

    已安装模块版本来自 *site-packages* 目录，使用 *importlib.metadata* 获取。

    参数：
        requirement (`str`): pip 风格的定义，例如 "tokenizers==0.9.4"，"tqdm>=4.27"，"numpy"
        hint (`str`, *可选*): 如果未满足要求时要打印的建议

    示例：

    ```python
    require_version("pandas>1.1.2")
    require_version("numpy>1.18.5", "this is important to have for whatever reason")
    ```py"""

    # 如果提供了 hint，则格式化为字符串
    hint = f"\n{hint}" if hint is not None else ""

    # 进行无版本检查
    if re.match(r"^[\w_\-\d]+$", requirement):
        pkg, op, want_ver = requirement, None, None  # 解析要求，提取包名
    else:  # 如果不满足前面的条件
        # 使用正则表达式查找符合格式的要求，提取包名和完整的需求字符串
        match = re.findall(r"^([^!=<>\s]+)([\s!=<>]{1,2}.+)", requirement)
        # 如果没有匹配到，抛出值错误
        if not match:
            raise ValueError(
                "requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but"
                f" got {requirement}"  # 报告实际收到的格式
            )
        # 解构匹配结果，pkg为包名，want_full为完整需求字符串
        pkg, want_full = match[0]
        # 将完整需求字符串按逗号分割，可能存在多个需求
        want_range = want_full.split(",")  
        wanted = {}  # 初始化一个字典来存储期望的版本和操作符
        for w in want_range:  # 遍历每个需求
            # 使用正则表达式提取操作符和期望版本
            match = re.findall(r"^([\s!=<>]{1,2})(.+)", w)
            # 如果没有匹配到，抛出值错误
            if not match:
                raise ValueError(
                    "requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23,"
                    f" but got {requirement}"  # 报告实际收到的格式
                )
            op, want_ver = match[0]  # 解构匹配结果，op为操作符，want_ver为期望版本
            wanted[op] = want_ver  # 将操作符和期望版本存入字典
            # 如果操作符不在预定义的操作符集合中，抛出值错误
            if op not in ops:
                raise ValueError(f"{requirement}: need one of {list(ops.keys())}, but got {op}")

    # 特殊情况处理
    if pkg == "python":  # 如果包名是python
        # 获取当前Python的版本字符串
        got_ver = ".".join([str(x) for x in sys.version_info[:3]])
        # 遍历期望的版本和操作符进行比较
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
        return  # 完成比较后直接返回

    # 检查是否安装了任何版本
    try:
        # 尝试获取已安装包的版本
        got_ver = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        # 如果未找到包，则抛出包未找到错误
        raise importlib.metadata.PackageNotFoundError(
            f"The '{requirement}' distribution was not found and is required by this application. {hint}"
        )

    # 如果提供了版本号或范围，检查是否安装了正确的版本
    if want_ver is not None:
        for op, want_ver in wanted.items():  # 遍历期望的版本和操作符
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)  # 进行版本比较
# 定义一个函数，检查所需版本并在失败时提供特定提示
def require_version_core(requirement):
    # 提供一个针对核心的提示，说明如何解决版本问题
    hint = "Try: pip install transformers -U or pip install -e '.[dev]' if you're working with git main"
    # 调用 require_version 函数，传入需求和提示信息
    return require_version(requirement, hint)
```