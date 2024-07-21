# `.\pytorch\tools\build_pytorch_libs.py`

```py
from __future__ import annotations
# 导入未来的注解支持，使得可以在类型提示中使用字符串类型作为字典键

import os
# 导入操作系统相关功能模块

import platform
# 导入平台信息模块，用于获取平台相关信息

import shutil
# 导入文件和目录管理模块，用于高级文件操作

from glob import glob
# 从 glob 模块中导入 glob 函数，用于查找文件路径模式的所有匹配文件路径名列表

from setuptools import distutils  # type: ignore[import]
# 从 setuptools 模块中导入 distutils 子模块，用于安装 Python 包的工具集

from .setup_helpers.cmake import CMake, USE_NINJA
# 从当前包中的 setup_helpers 子模块中导入 CMake 类和 USE_NINJA 常量

from .setup_helpers.env import check_negative_env_flag, IS_64BIT, IS_WINDOWS
# 从当前包中的 setup_helpers 子模块中导入 check_negative_env_flag 函数、IS_64BIT 常量和 IS_WINDOWS 常量


def _overlay_windows_vcvars(env: dict[str, str]) -> dict[str, str]:
    # 定义内部函数 _overlay_windows_vcvars，接受一个字典参数 env 并返回一个字典
    vc_arch = "x64" if IS_64BIT else "x86"
    # 根据 IS_64BIT 常量的值设置 vc_arch 变量为 "x64" 或 "x86"

    if platform.machine() == "ARM64":
        # 如果当前平台是 ARM64 架构
        vc_arch = "x64_arm64"

        # 检查当前 Windows 版本是否支持 x64 模拟
        win11_1st_version = (10, 0, 22000)
        current_win_version = tuple(
            int(version_part) for version_part in platform.version().split(".")
        )
        if current_win_version < win11_1st_version:
            # 如果当前 Windows 版本低于 10.0.22000
            vc_arch = "x86_arm64"
            # 设置 vc_arch 变量为 "x86_arm64"
            print(
                "Warning: 32-bit toolchain will be used, but 64-bit linker "
                "is recommended to avoid out-of-memory linker error!"
            )
            print(
                "Warning: Please consider upgrading to Win11, where x64 "
                "emulation is enabled!"
            )

    vc_env: dict[str, str] = distutils._msvccompiler._get_vc_env(vc_arch)
    # 调用 distutils._msvccompiler._get_vc_env 函数获取指定架构的 VC 环境变量字典
    # Keys in `_get_vc_env` are always lowercase.
    # We turn them into uppercase before overlaying vcvars
    # because OS environ keys are always uppercase on Windows.
    # https://stackoverflow.com/a/7797329
    vc_env = {k.upper(): v for k, v in vc_env.items()}
    # 将 vc_env 中的所有键转换为大写，并创建一个新的字典
    for k, v in env.items():
        uk = k.upper()
        # 将 env 字典中的键转换为大写
        if uk not in vc_env:
            vc_env[uk] = v
            # 如果 uk 不在 vc_env 中，则将 uk 和 v 添加到 vc_env 字典中
    return vc_env
    # 返回处理后的 VC 环境变量字典


def _create_build_env() -> dict[str, str]:
    # 定义内部函数 _create_build_env，返回一个字典
    # XXX - our cmake file sometimes looks at the system environment
    # and not cmake flags!
    # you should NEVER add something to this list. It is bad practice to
    # have cmake read the environment
    # 设置 my_env 变量为当前进程环境的拷贝
    my_env = os.environ.copy()

    if (
        "CUDA_HOME" in my_env
    ):  # 如果环境变量中存在 "CUDA_HOME"
        my_env["CUDA_BIN_PATH"] = my_env["CUDA_HOME"]
        # 将 "CUDA_HOME" 的值赋给 "CUDA_BIN_PATH"
    elif IS_WINDOWS:  # 如果当前操作系统是 Windows
        cuda_win = glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*")
        # 查找 CUDA 安装路径下的版本目录列表
        if len(cuda_win) > 0:
            my_env["CUDA_BIN_PATH"] = cuda_win[0]
            # 如果找到 CUDA 目录，则将第一个目录路径赋给 "CUDA_BIN_PATH"

    if IS_WINDOWS and USE_NINJA:
        # 如果当前操作系统是 Windows 且使用了 Ninja 构建系统
        # When using Ninja under Windows, the gcc toolchain will be chosen as
        # default. But it should be set to MSVC as the user's first choice.
        # 但应将其设置为用户首选的 MSVC 工具链。
        my_env = _overlay_windows_vcvars(my_env)
        # 调用 _overlay_windows_vcvars 函数获取 Windows 平台的 VC 环境变量
        my_env.setdefault("CC", "cl")
        # 如果 "CC" 未设置，则默认设置为 "cl"
        my_env.setdefault("CXX", "cl")
        # 如果 "CXX" 未设置，则默认设置为 "cl"
    return my_env
    # 返回处理后的构建环境变量字典


def build_caffe2(
    version: str | None,
    cmake_python_library: str | None,
    build_python: bool,
    rerun_cmake: bool,
    cmake_only: bool,
    cmake: CMake,
) -> None:
    # 定义 build_caffe2 函数，接受多个参数，无返回值
    my_env = _create_build_env()
    # 调用 _create_build_env 函数获取构建环境变量字典
    build_test = not check_negative_env_flag("BUILD_TEST")
    # 设置 build_test 变量为是否未设置 "BUILD_TEST" 环境变量的逆值
    # 调用 cmake 对象的 generate 方法，生成 CMake 构建所需的配置
    cmake.generate(
        version, cmake_python_library, build_python, build_test, my_env, rerun_cmake
    )
    # 如果指定了 cmake_only 标志，直接返回，不进行后续构建步骤
    if cmake_only:
        return
    # 调用 cmake 对象的 build 方法，执行构建操作
    cmake.build(my_env)
    # 如果设置了 build_python 标志，复制生成的 Proto 文件到指定目录
    if build_python:
        # 构建目录中的 Caffe2 Proto 文件目录
        caffe2_proto_dir = os.path.join(cmake.build_dir, "caffe2", "proto")
        # 遍历 Proto 文件目录下的所有 .py 文件
        for proto_file in glob(os.path.join(caffe2_proto_dir, "*.py")):
            # 排除 __init__.py 文件，进行复制操作
            if proto_file != os.path.join(caffe2_proto_dir, "__init__.py"):
                shutil.copy(proto_file, os.path.join("caffe2", "proto"))
```