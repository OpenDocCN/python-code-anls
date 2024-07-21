# `.\pytorch\tools\setup_helpers\cmake.py`

```py
# "Manages CMake."

# 从未来导入的注释，使得 Python 2.x 中的代码与 Python 3.x 兼容
from __future__ import annotations

# 导入多进程处理模块
import multiprocessing
# 导入操作系统接口模块
import os
# 导入平台信息模块
import platform
# 导入系统相关配置信息模块
import sys
import sysconfig
# 导入版本比较模块
from distutils.version import LooseVersion
# 导入子进程相关异常和执行命令函数
from subprocess import CalledProcessError, check_call, check_output
# 导入类型提示模块
from typing import Any, cast

# 从当前包中导入 which 模块
from . import which
# 从当前包中导入 cmake_utils 模块下的 CMakeValue 类和 get_cmake_cache_variables_from_file 函数
from .cmake_utils import CMakeValue, get_cmake_cache_variables_from_file
# 从当前包中导入 env 模块下的 BUILD_DIR, check_negative_env_flag, IS_64BIT, IS_DARWIN, IS_WINDOWS 变量
from .env import BUILD_DIR, check_negative_env_flag, IS_64BIT, IS_DARWIN, IS_WINDOWS

# 定义一个函数，用于递归创建目录
def _mkdir_p(d: str) -> None:
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Failed to create folder {os.path.abspath(d)}: {e.strerror}"
        ) from e

# Ninja
# 如果环境变量中没有设置 USE_NINJA 为 False，并且 ninja 可执行程序存在于系统路径中，则启用 ninja 构建系统
USE_NINJA = not check_negative_env_flag("USE_NINJA") and which("ninja") is not None
# 如果环境变量中设置了 CMAKE_GENERATOR 且其值为 ninja，则强制使用 ninja 构建系统
if "CMAKE_GENERATOR" in os.environ:
    USE_NINJA = os.environ["CMAKE_GENERATOR"].lower() == "ninja"

# 定义一个类 CMake，用于管理 cmake 相关操作
class CMake:
    "Manages cmake."

    # 初始化方法，设置 cmake 命令路径和构建目录
    def __init__(self, build_dir: str = BUILD_DIR) -> None:
        self._cmake_command = CMake._get_cmake_command()  # 获取 cmake 命令路径
        self.build_dir = build_dir  # 设置构建目录路径

    # 属性方法，返回 CMakeCache.txt 文件路径
    @property
    def _cmake_cache_file(self) -> str:
        r"""Returns the path to CMakeCache.txt.

        Returns:
          string: The path to CMakeCache.txt.
        """
        return os.path.join(self.build_dir, "CMakeCache.txt")

    # 静态方法，返回 cmake 命令路径
    @staticmethod
    def _get_cmake_command() -> str:
        "Returns cmake command."

        cmake_command = "cmake"
        # 如果运行环境为 Windows，则直接返回 cmake 命令
        if IS_WINDOWS:
            return cmake_command
        # 获取 cmake3 和 cmake 的版本信息
        cmake3_version = CMake._get_version(which("cmake3"))
        cmake_version = CMake._get_version(which("cmake"))

        _cmake_min_version = LooseVersion("3.18.0")
        # 如果 cmake 和 cmake3 的版本均小于 3.18.0，则抛出运行时异常
        if all(
            ver is None or ver < _cmake_min_version
            for ver in [cmake_version, cmake3_version]
        ):
            raise RuntimeError("no cmake or cmake3 with version >= 3.18.0 found")

        # 根据版本信息确定最终的 cmake 命令路径
        if cmake3_version is None:
            cmake_command = "cmake"
        elif cmake_version is None:
            cmake_command = "cmake3"
        else:
            if cmake3_version >= cmake_version:
                cmake_command = "cmake3"
            else:
                cmake_command = "cmake"
        return cmake_command

    # 静态方法，返回指定命令的版本信息
    @staticmethod
    def _get_version(cmd: str | None) -> Any:
        "Returns cmake version."

        if cmd is None:
            return None
        # 执行指定命令获取输出，并解析其中的版本信息
        for line in check_output([cmd, "--version"]).decode("utf-8").split("\n"):
            if "version" in line:
                return LooseVersion(line.strip().split(" ")[2])
        raise RuntimeError("no version found")
    def run(self, args: list[str], env: dict[str, str]) -> None:
        "Executes cmake with arguments and an environment."

        # 构建完整的命令列表，包括 cmake 命令本身和传入的参数
        command = [self._cmake_command] + args
        # 打印完整的命令字符串，用于调试和日志记录
        print(" ".join(command))
        try:
            # 执行 cmake 命令，并指定工作目录和环境变量
            check_call(command, cwd=self.build_dir, env=env)
        except (CalledProcessError, KeyboardInterrupt) as e:
            # 捕获可能出现的异常，如调用进程错误或用户中断信号
            # 如果出现异常，则输出相关错误信息并手动退出程序
            # 此处退出码为 1，表示异常终止
            sys.exit(1)

    @staticmethod
    def defines(args: list[str], **kwargs: CMakeValue) -> None:
        "Adds definitions to a cmake argument list."
        # 遍历关键字参数 kwargs，将非空值作为定义添加到参数列表 args 中
        for key, value in sorted(kwargs.items()):
            if value is not None:
                args.append(f"-D{key}={value}")

    def get_cmake_cache_variables(self) -> dict[str, CMakeValue]:
        r"""Gets values in CMakeCache.txt into a dictionary.
        Returns:
          dict: A ``dict`` containing the value of cached CMake variables.
        """
        # 使用上下文管理器打开 CMakeCache.txt 文件，并调用函数将其内容解析为字典
        with open(self._cmake_cache_file) as f:
            return get_cmake_cache_variables_from_file(f)

    def generate(
        self,
        version: str | None,
        cmake_python_library: str | None,
        build_python: bool,
        build_test: bool,
        my_env: dict[str, str],
        rerun: bool,
    ) -> None:
        r"""Generates build files using CMake.
        Args:
          version: The version of CMake to use.
          cmake_python_library: The Python library to use with CMake.
          build_python: Whether to build Python bindings.
          build_test: Whether to build test targets.
          my_env: Environment variables to pass to CMake.
          rerun: Whether to force rerunning CMake even if build files exist.
        """
    def build(self, my_env: dict[str, str]) -> None:
        "Runs cmake to build binaries."

        from .env import build_type

        build_args = [
            "--build",
            ".",
            "--target",
            "install",
            "--config",
            build_type.build_type_string,
        ]

        # 根据以下优先级确定并行度:
        # 1) MAX_JOBS 环境变量
        # 2) 如果使用 Ninja 构建系统，则委托决策给它。
        # 3) 否则，使用处理器数量作为后备方案。

        # 允许用户显式设置并行度。如果未设置，将尝试自动确定。
        max_jobs = os.getenv("MAX_JOBS")

        if max_jobs is not None or not USE_NINJA:
            # Ninja 能够自动确定并行度：只有在不使用 Ninja 时才显式指定。

            # 获取机器上可用的处理器数量。如果 CPU 调度亲和性限制更少，则这可能是可用处理器的高估。
            # 未来，我们应该在支持的平台上使用 os.sched_getaffinity(0) 进行检查。
            max_jobs = max_jobs or str(multiprocessing.cpu_count())

            # 当 cmake 3.12 变为最低版本时，将提供 '-j' 选项，此时 build_args += ['-j', max_jobs] 就足够了。
            # 在此之前，我们使用 "--" 将参数传递给底层构建系统。
            build_args += ["--"]
            if IS_WINDOWS and not USE_NINJA:
                # 这里可能在使用 msbuild
                build_args += [f"/p:CL_MPCount={max_jobs}"]
            else:
                build_args += ["-j", max_jobs]
        self.run(build_args, my_env)
```