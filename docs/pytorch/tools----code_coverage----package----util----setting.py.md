# `.\pytorch\tools\code_coverage\package\util\setting.py`

```py
```python`
# 引入 __future__ 模块，支持类型注解的类型自引用
from __future__ import annotations

# 引入操作系统模块
import os
# 引入枚举类型模块
from enum import Enum
# 引入类型提示相关模块
from typing import Dict, List, Set


# <project folder>
# 获取当前用户主目录路径
HOME_DIR = os.environ["HOME"]
# 设置工具文件夹路径，使用当前文件的绝对路径解析上两级作为工具文件夹路径
TOOLS_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir
)


# <profile folder>
# 设置配置文件夹路径，位于工具文件夹下的 profile 子文件夹
PROFILE_DIR = os.path.join(TOOLS_FOLDER, "profile")
# 设置存放 JSON 文件的基础路径，位于配置文件夹下的 json 子文件夹
JSON_FOLDER_BASE_DIR = os.path.join(PROFILE_DIR, "json")
# 设置存放合并文件的基础路径，位于配置文件夹下的 merged 子文件夹
MERGED_FOLDER_BASE_DIR = os.path.join(PROFILE_DIR, "merged")
# 设置存放摘要文件的路径，位于配置文件夹下的 summary 子文件夹
SUMMARY_FOLDER_DIR = os.path.join(PROFILE_DIR, "summary")

# <log path>
# 设置日志文件夹路径，位于配置文件夹下的 log 子文件夹
LOG_DIR = os.path.join(PROFILE_DIR, "log")


# test type, DO NOT change the name, it should be consistent with [buck query --output-attribute] result
# 定义测试类型枚举
class TestType(Enum):
    CPP: str = "cxx_test"
    PY: str = "python_test"


# 定义测试类
class Test:
    name: str
    target_pattern: str
    test_set: str  # like __aten__
    test_type: TestType

    # 初始化方法，接受名称、目标模式、测试集和测试类型作为参数
    def __init__(
        self, name: str, target_pattern: str, test_set: str, test_type: TestType
    ) -> None:
        self.name = name
        self.target_pattern = target_pattern
        self.test_set = test_set
        self.test_type = test_type


# 定义测试列表类型
TestList = List[Test]
# 定义测试状态类型，使用字典表示测试名称到状态集合的映射
TestStatusType = Dict[str, Set[str]]


# option
# 定义选项类，包含是否需要构建、运行、合并、导出、总结以及是否需要运行 pytest 的选项
class Option:
    need_build: bool = False
    need_run: bool = False
    need_merge: bool = False
    need_export: bool = False
    need_summary: bool = False
    need_pytest: bool = False


# test platform
# 定义测试平台枚举
class TestPlatform(Enum):
    FBCODE: str = "fbcode"
    OSS: str = "oss"


# compiler type
# 定义编译器类型枚举
class CompilerType(Enum):
    CLANG: str = "clang"
    GCC: str = "gcc"
```