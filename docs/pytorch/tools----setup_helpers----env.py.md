# `.\pytorch\tools\setup_helpers\env.py`

```py
# 从未来模块导入 annotations 特性，使得类型提示中的字符串可以是字符串字面值或者类型名称。
from __future__ import annotations

# 导入标准库中的操作系统、平台、结构化数据和系统相关功能模块。
import os
import platform
import struct
import sys

# 从迭代工具中导入链式迭代器函数。
from itertools import chain

# 导入类型提示中的 cast 函数和 Iterable 泛型。
from typing import cast, Iterable

# 检测当前操作系统是否为 Windows、Darwin 或 Linux。
IS_WINDOWS = platform.system() == "Windows"
IS_DARWIN = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# 检测当前 Python 环境是否为 Conda 环境。
IS_CONDA = (
    "conda" in sys.version
    or "Continuum" in sys.version
    or any(x.startswith("CONDA") for x in os.environ)
)

# 获取 Conda 安装目录。
CONDA_DIR = os.path.join(os.path.dirname(sys.executable), "..")

# 检测系统是否为 64 位。
IS_64BIT = struct.calcsize("P") == 8

# 定义一个常量，表示构建目录为 "build"。
BUILD_DIR = "build"

# 检测指定环境变量名对应的值是否在指定的真值列表中。
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]

# 检测指定环境变量名对应的值是否在指定的假值列表中。
def check_negative_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["OFF", "0", "NO", "FALSE", "N"]

# 收集环境变量列表中所有变量值分割后的路径列表。
def gather_paths(env_vars: Iterable[str]) -> list[str]:
    return list(chain(*(os.getenv(v, "").split(os.pathsep) for v in env_vars)))

# 根据基础路径生成库文件路径列表。
def lib_paths_from_base(base_path: str) -> list[str]:
    return [os.path.join(base_path, s) for s in ["lib/x64", "lib", "lib64"]]

# 如果存在环境变量 "CFLAGS" 且不存在环境变量 "CXXFLAGS"，则将 "CXXFLAGS" 设置为 "CFLAGS" 的值。
if "CFLAGS" in os.environ and "CXXFLAGS" not in os.environ:
    os.environ["CXXFLAGS"] = os.environ["CFLAGS"]

class BuildType:
    """Checks build type. The build type will be given in :attr:`cmake_build_type_env`. If :attr:`cmake_build_type_env`
    is ``None``, then the build type will be inferred from ``CMakeCache.txt``. If ``CMakeCache.txt`` does not exist,
    os.environ['CMAKE_BUILD_TYPE'] will be used.

    Args:
      cmake_build_type_env (str): The value of os.environ['CMAKE_BUILD_TYPE']. If None, the actual build type will be
        inferred.

    """

    def __init__(self, cmake_build_type_env: str | None = None) -> None:
        # 如果提供了 cmake_build_type_env 参数，则直接使用其值作为构建类型。
        if cmake_build_type_env is not None:
            self.build_type_string = cmake_build_type_env
            return

        # 构建 CMakeCache.txt 文件路径。
        cmake_cache_txt = os.path.join(BUILD_DIR, "CMakeCache.txt")

        # 如果 CMakeCache.txt 文件存在，则从中获取指定的构建类型。
        if os.path.isfile(cmake_cache_txt):
            # 使用 .cmake_utils 模块中的函数从文件中获取 CMake 缓存变量。
            from .cmake_utils import get_cmake_cache_variables_from_file

            with open(cmake_cache_txt) as f:
                cmake_cache_vars = get_cmake_cache_variables_from_file(f)
            # 从 CMake 缓存中获取构建类型。
            self.build_type_string = cast(str, cmake_cache_vars["CMAKE_BUILD_TYPE"])
        else:
            # 如果不存在 CMakeCache.txt 文件，则使用环境变量中的 CMAKE_BUILD_TYPE，若没有则默认为 "Release"。
            self.build_type_string = os.environ.get("CMAKE_BUILD_TYPE", "Release")

    # 判断当前的构建类型是否为 Debug。
    def is_debug(self) -> bool:
        "Checks Debug build."
        return self.build_type_string == "Debug"
    # 检查是否为 RelWithDebInfo 构建类型
    def is_rel_with_deb_info(self) -> bool:
        "Checks RelWithDebInfo build."
        return self.build_type_string == "RelWithDebInfo"
    
    # 检查是否为 Release 构建类型
    def is_release(self) -> bool:
        "Checks Release build."
        return self.build_type_string == "Release"
# 如果环境变量中没有设置'CMAKE_BUILD_TYPE'，则根据其他条件设置它的值
if "CMAKE_BUILD_TYPE" not in os.environ:
    # 如果环境中有设置'DEBUG'标志，则设置'CMAKE_BUILD_TYPE'为"Debug"
    if check_env_flag("DEBUG"):
        os.environ["CMAKE_BUILD_TYPE"] = "Debug"
    # 如果环境中有设置'REL_WITH_DEB_INFO'标志，则设置'CMAKE_BUILD_TYPE'为"RelWithDebInfo"
    elif check_env_flag("REL_WITH_DEB_INFO"):
        os.environ["CMAKE_BUILD_TYPE"] = "RelWithDebInfo"
    # 如果以上条件均不满足，则设置'CMAKE_BUILD_TYPE'为"Release"
    else:
        os.environ["CMAKE_BUILD_TYPE"] = "Release"

# 创建BuildType对象的实例
build_type = BuildType()
```