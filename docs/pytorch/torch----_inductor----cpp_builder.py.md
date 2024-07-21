# `.\pytorch\torch\_inductor\cpp_builder.py`

```
# mypy: allow-untyped-defs
# This CPP JIT builder is designed to support both Windows and Linux OS.
# The design document please check this RFC: https://github.com/pytorch/pytorch/issues/124245

# 引入必要的模块和库
import copy  # 复制对象
import errno  # 提供错误码定义
import functools  # 提供函数工具，如LRU缓存等
import json  # JSON编码和解码
import logging  # 记录日志信息
import os  # 提供操作系统相关的功能
import platform  # 提供平台相关的信息
import re  # 提供正则表达式操作
import shlex  # 用于解析和操作命令行字符串
import shutil  # 提供高级文件操作
import subprocess  # 提供执行外部命令的功能
import sys  # 提供与Python解释器交互的功能
import sysconfig  # 提供Python配置信息
import warnings  # 提供警告控制

from pathlib import Path  # 提供面向对象的文件系统路径操作
from typing import List, Sequence, Tuple, Union  # 引入类型提示相关的功能

import torch  # PyTorch深度学习框架

from torch._inductor import config, exc  # 引入编译相关的配置和异常
from torch._inductor.cpu_vec_isa import invalid_vec_isa, VecISA  # 引入CPU矢量指令集相关功能
from torch._inductor.runtime.runtime_utils import cache_dir  # 引入运行时工具函数中的缓存目录函数

if config.is_fbcode():
    from triton.fb import build_paths  # 如果是在Facebook的环境下，引入相关的构建路径

    from torch._inductor.fb.utils import (
        log_global_cache_errors,  # 引入Facebook环境下的全局缓存错误日志记录函数
        log_global_cache_stats,  # 引入Facebook环境下的全局缓存统计日志记录函数
        log_global_cache_vals,  # 引入Facebook环境下的全局缓存值日志记录函数
        use_global_cache,  # 引入Facebook环境下的全局缓存使用函数
    )
else:
    # 如果不是在Facebook环境下，定义空的函数来占位，不做任何操作
    def log_global_cache_errors(*args, **kwargs):
        pass

    def log_global_cache_stats(*args, **kwargs):
        pass

    def log_global_cache_vals(*args, **kwargs):
        pass

    def use_global_cache() -> bool:
        return False


# Windows需要设置临时目录来存储.obj文件。
_BUILD_TEMP_DIR = "CxxBuild"

# 初始化变量用于编译
_IS_LINUX = sys.platform.startswith("linux")  # 检查当前操作系统是否为Linux
_IS_MACOS = sys.platform.startswith("darwin")  # 检查当前操作系统是否为MacOS
_IS_WINDOWS = sys.platform == "win32"  # 检查当前操作系统是否为Windows


log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


# =============================== toolchain ===============================
@functools.lru_cache(1)
def cpp_compiler_search(search: str) -> str:
    from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT

    for cxx in search:
        try:
            if cxx is None:
                # 对于Linux系统，需要安装gxx包，详情参见https://anaconda.org/conda-forge/gxx/
                if sys.platform != "linux":
                    continue
                # 默认情况下不安装GXX
                if not os.getenv("TORCH_INDUCTOR_INSTALL_GXX"):
                    continue
                from filelock import FileLock

                lock_dir = get_lock_dir()
                # 使用文件锁保证安全并发安装
                lock = FileLock(
                    os.path.join(lock_dir, "g++.lock"), timeout=LOCK_TIMEOUT
                )
                with lock:
                    cxx = install_gcc_via_conda()
            subprocess.check_output([cxx, "--version"])  # 检查编译器版本
            return cxx
        except (subprocess.SubprocessError, FileNotFoundError, ImportError):
            continue
    raise exc.InvalidCxxCompiler  # 抛出无效的C++编译器异常


def install_gcc_via_conda() -> str:
    """在旧系统上，通过此方法快速获取现代编译器"""
    prefix = os.path.join(cache_dir(), "gcc")
    cxx_path = os.path.join(prefix, "bin", "g++")
    # 如果指定的路径 cxx_path 不存在
    if not os.path.exists(cxx_path):
        # 记录信息到日志，表示正在通过 conda 下载 GCC
        log.info("Downloading GCC via conda")
        
        # 获取环境变量 CONDA_EXE 的值，如果未设置则默认为 "conda"
        conda = os.environ.get("CONDA_EXE", "conda")
        
        # 如果未找到 conda 可执行文件，则尝试使用 shutil.which 查找
        if conda is None:
            conda = shutil.which("conda")
        
        # 如果找到 conda 可执行文件
        if conda is not None:
            # 使用 subprocess.check_call 调用 conda 命令来创建环境
            subprocess.check_call(
                [
                    conda,  # conda 可执行文件路径
                    "create",  # 创建环境命令
                    f"--prefix={prefix}",  # 指定环境安装路径
                    "--channel=conda-forge",  # 使用 conda-forge 频道
                    "--quiet",  # 安装过程中不显示详细信息
                    "-y",  # 自动确认安装
                    "python=3.8",  # 安装 Python 3.8
                    "gxx",  # 安装 gxx（GCC 的一部分）
                ],
                stdout=subprocess.PIPE,  # 标准输出重定向到 PIPE
            )
    
    # 返回 cxx_path 变量的值
    return cxx_path
# 返回 C++ 编译器的名称作为字符串
def get_cpp_compiler() -> str:
    # 如果运行在 Windows 平台
    if _IS_WINDOWS:
        # 获取环境变量中的 CXX 变量作为编译器名称，如果不存在则默认为 "cl"
        compiler = os.environ.get("CXX", "cl")
    else:
        # 如果不是运行在 Windows 平台
        if config.is_fbcode():
            # 如果是在 Facebook 的代码环境中，返回构建路径下的 C++ 编译器
            return build_paths.cc()
        # 如果 config.cpp.cxx 是列表或元组类型
        if isinstance(config.cpp.cxx, (list, tuple)):
            # 将 config.cpp.cxx 转换为元组，用作编译器搜索的参数
            search = tuple(config.cpp.cxx)
        else:
            # 否则将 config.cpp.cxx 作为单一元素的元组
            search = (config.cpp.cxx,)
        # 调用 cpp_compiler_search 函数来搜索合适的编译器，并将结果赋给 compiler
        compiler = cpp_compiler_search(search)
    # 返回得到的编译器名称
    return compiler


# 使用 functools.lru_cache 缓存结果，检查给定的 C++ 编译器是否为 Apple Clang
@functools.lru_cache(None)
def _is_apple_clang(cpp_compiler) -> bool:
    # 调用 subprocess 模块执行命令，获取编译器的版本信息字符串
    version_string = subprocess.check_output([cpp_compiler, "--version"]).decode("utf8")
    # 检查版本信息字符串中是否包含 "Apple"，并返回结果
    return "Apple" in version_string.splitlines()[0]


# 检查给定的 C++ 编译器是否为 Clang
def _is_clang(cpp_compiler) -> bool:
    # 如果操作系统是 macOS
    if sys.platform == "darwin":
        # 调用 _is_apple_clang 函数来判断是否为 Apple Clang
        return _is_apple_clang(cpp_compiler)
    # 对编译器名称进行正则匹配，判断是否为 Clang 或 Clang++，返回匹配结果
    return bool(re.search(r"(clang|clang\+\+)", cpp_compiler))


# 检查给定的 C++ 编译器是否为 GCC
def _is_gcc(cpp_compiler) -> bool:
    # 如果操作系统是 macOS，并且编译器是 Apple Clang
    if sys.platform == "darwin" and _is_apple_clang(cpp_compiler):
        # 返回 False，表示不是 GCC
        return False
    # 对编译器名称进行正则匹配，判断是否为 GCC 或 G++，返回匹配结果
    return bool(re.search(r"(gcc|g\+\+)", cpp_compiler))


# 使用 functools.lru_cache 缓存结果，检查当前系统是否使用 GCC 编译器
@functools.lru_cache(None)
def is_gcc() -> bool:
    # 调用 _is_gcc 函数来判断给定 C++ 编译器是否为 GCC
    return _is_gcc(get_cpp_compiler())


# 使用 functools.lru_cache 缓存结果，检查当前系统是否使用 Clang 编译器
@functools.lru_cache(None)
def is_clang() -> bool:
    # 调用 _is_clang 函数来判断给定 C++ 编译器是否为 Clang
    return _is_clang(get_cpp_compiler())


# 使用 functools.lru_cache 缓存结果，检查当前系统是否使用 Apple Clang 编译器
@functools.lru_cache(None)
def is_apple_clang() -> bool:
    # 调用 _is_apple_clang 函数来判断给定 C++ 编译器是否为 Apple Clang
    return _is_apple_clang(get_cpp_compiler())


# 返回编译器的版本信息字符串
def get_compiler_version_info(compiler: str) -> str:
    SUBPROCESS_DECODE_ARGS = ("oem",) if _IS_WINDOWS else ()
    env = os.environ.copy()
    env["LC_ALL"] = "C"  # 不本地化输出
    try:
        # 尝试执行带 "-v" 参数的编译器命令，获取版本信息字符串
        version_string = subprocess.check_output(
            [compiler, "-v"], stderr=subprocess.STDOUT, env=env
        ).decode(*SUBPROCESS_DECODE_ARGS)
    except Exception as e:
        try:
            # 如果上述命令失败，则尝试执行带 "--version" 参数的编译器命令，获取版本信息字符串
            version_string = subprocess.check_output(
                [compiler, "--version"], stderr=subprocess.STDOUT, env=env
            ).decode(*SUBPROCESS_DECODE_ARGS)
        except Exception as e:
            # 如果两者都失败，则返回空字符串
            return ""
    # 将版本信息字符串中的换行符替换为下划线，并返回处理后的字符串
    version_string = version_string.replace("\r", "_")
    version_string = version_string.replace("\n", "_")
    return version_string


# =============================== cpp builder ===============================

# 将源列表中的每个元素深度复制并添加到目标列表中
def _append_list(dest_list: List[str], src_list: List[str]):
    for item in src_list:
        dest_list.append(copy.deepcopy(item))


# 从原始列表中移除重复项，并返回新列表
def _remove_duplication_in_list(orig_list: List[str]) -> List[str]:
    new_list: List[str] = []
    for item in orig_list:
        if item not in new_list:
            new_list.append(item)
    return new_list


# 如果目录不存在则创建
def _create_if_dir_not_exist(path_dir):
    if not os.path.exists(path_dir):
        try:
            # 使用 Path 类创建目录，并支持递归创建及存在时忽略错误
            Path(path_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # 防止竞争条件
            if exc.errno != errno.EEXIST:
                # 如果不是已存在错误，则抛出运行时异常
                raise RuntimeError(
                    f"Fail to create path {path_dir}"
                )


# 移除指定目录及其内容
    # 检查指定路径是否存在
    if os.path.exists(path_dir):
        # 递归地遍历目录结构，从底层向上删除文件和目录
        for root, dirs, files in os.walk(path_dir, topdown=False):
            # 删除当前目录下的所有文件
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            # 删除当前目录下的所有子目录
            for name in dirs:
                dir_path = os.path.join(root, name)
                os.rmdir(dir_path)
        # 删除最顶层的指定目录
        os.rmdir(path_dir)
# 定义一个函数用于执行命令行命令，返回命令执行的状态
def run_command_line(cmd_line, cwd=None):
    # 使用 shlex.split 将命令行字符串解析为命令列表
    cmd = shlex.split(cmd_line)
    try:
        # 使用 subprocess.check_output 执行命令，并捕获命令的输出结果
        status = subprocess.check_output(args=cmd, cwd=cwd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        # 如果命令执行出错，捕获异常，并解析异常的输出信息
        output = e.output.decode("utf-8")
        # 检查输出中是否包含与 OpenMP 相关的问题
        openmp_problem = "'omp.h' file not found" in output or "libomp" in output
        # 如果是在 macOS 平台上发现了 OpenMP 相关问题，提供解决方案说明
        if openmp_problem and sys.platform == "darwin":
            instruction = (
                "\n\nOpenMP support not found. Please try one of the following solutions:\n"
                "(1) Set the `CXX` environment variable to a compiler other than Apple clang++/g++ "
                "that has builtin OpenMP support;\n"
                "(2) install OpenMP via conda: `conda install llvm-openmp`;\n"
                "(3) install libomp via brew: `brew install libomp`;\n"
                "(4) manually setup OpenMP and set the `OMP_PREFIX` environment variable to point to a path"
                " with `include/omp.h` under it."
            )
            # 将解决方案说明添加到输出信息中
            output += instruction
        # 抛出自定义的 CppCompileError 异常，包含命令和输出信息
        raise exc.CppCompileError(cmd, output) from e
    # 返回命令执行的状态结果
    return status


class BuildOptionsBase:
    """
    这是一个基类，用于存储 C++ 构建选项的模板。
    实际上，要构建一个 C++ 共享库，我们只需选择一个编译器并维护适当的参数。
    """

    def __init__(self) -> None:
        # 初始化各种编译选项和状态
        self._compiler = ""
        self._definations: List[str] = []
        self._include_dirs: List[str] = []
        self._cflags: List[str] = []
        self._ldflags: List[str] = []
        self._libraries_dirs: List[str] = []
        self._libraries: List[str] = []
        # 一些参数难以抽象成跨平台的，直接传递
        self._passthough_args: List[str] = []

        self._aot_mode: bool = False
        self._use_absolute_path: bool = False
        self._compile_only: bool = False

    # 移除重复的编译选项
    def _remove_duplicate_options(self):
        self._definations = _remove_duplication_in_list(self._definations)
        self._include_dirs = _remove_duplication_in_list(self._include_dirs)
        self._cflags = _remove_duplication_in_list(self._cflags)
        self._ldflags = _remove_duplication_in_list(self._ldflags)
        self._libraries_dirs = _remove_duplication_in_list(self._libraries_dirs)
        self._libraries = _remove_duplication_in_list(self._libraries)
        self._passthough_args = _remove_duplication_in_list(self._passthough_args)

    # 返回当前使用的编译器
    def get_compiler(self) -> str:
        return self._compiler

    # 返回定义的编译选项列表
    def get_definations(self) -> List[str]:
        return self._definations

    # 返回包含目录列表
    def get_include_dirs(self) -> List[str]:
        return self._include_dirs

    # 返回编译标志列表
    def get_cflags(self) -> List[str]:
        return self._cflags

    # 返回链接标志列表
    def get_ldflags(self) -> List[str]:
        return self._ldflags

    # 返回库目录列表
    def get_libraries_dirs(self) -> List[str]:
        return self._libraries_dirs

    # 返回要链接的库列表
    def get_libraries(self) -> List[str]:
        return self._libraries
    # 返回当前对象的 _passthough_args 属性，即传递参数列表
    def get_passthough_args(self) -> List[str]:
        return self._passthough_args
    
    # 返回当前对象的 _aot_mode 属性，即 Ahead-of-Time 编译模式是否开启的布尔值
    def get_aot_mode(self) -> bool:
        return self._aot_mode
    
    # 返回当前对象的 _use_absolute_path 属性，即是否使用绝对路径的布尔值
    def get_use_absolute_path(self) -> bool:
        return self._use_absolute_path
    
    # 返回当前对象的 _compile_only 属性，即是否只编译而不运行的布尔值
    def get_compile_only(self) -> bool:
        return self._compile_only
def _get_warning_all_cflag(warning_all: bool = True) -> List[str]:
    # 如果不是在 Windows 系统上，则返回一个包含 "-Wall" 的列表（开启所有警告），否则返回空列表
    if not _IS_WINDOWS:
        return ["Wall"] if warning_all else []
    else:
        return []


def _get_cpp_std_cflag(std_num: str = "c++17") -> List[str]:
    # 如果在 Windows 系统上，则返回一个包含指定 C++ 标准的列表，形如 ["std:c++17"]
    if _IS_WINDOWS:
        return [f"std:{std_num}"]
    else:
        # 否则返回一个包含指定 C++ 标准的列表，形如 ["std=c++17"]
        return [f"std={std_num}"]


def _get_linux_cpp_cflags(cpp_compiler) -> List[str]:
    # 如果不是在 Windows 系统上，则设置一些特定的编译标志列表，如禁用未使用变量和未知预处理指令警告
    if not _IS_WINDOWS:
        cflags = ["Wno-unused-variable", "Wno-unknown-pragmas"]
        # 如果使用的是 Clang 编译器，增加一个特定的警告处理选项
        if _is_clang(cpp_compiler):
            cflags.append("Werror=ignored-optimization-argument")
        return cflags
    else:
        # 如果在 Windows 系统上，则返回一个空列表
        return []


def _get_optimization_cflags() -> List[str]:
    # 如果在 Windows 系统上，则返回一个包含优化选项的列表，如 ["O2"]
    if _IS_WINDOWS:
        return ["O2"]
    else:
        # 否则根据配置文件决定返回不同的优化选项列表，包含开启调试编译时的选项，或者关闭调试编译时的选项
        cflags = ["O0", "g"] if config.aot_inductor.debug_compile else ["O3", "DNDEBUG"]
        cflags.append("ffast-math")
        cflags.append("fno-finite-math-only")

        # 根据配置文件设置是否开启不安全数学优化选项
        if not config.cpp.enable_unsafe_math_opt_flag:
            cflags.append("fno-unsafe-math-optimizations")
        # 根据配置文件设置是否开启浮点数契约优化选项
        if not config.cpp.enable_floating_point_contract_flag:
            cflags.append("ffp-contract=off")

        # 如果不是在 macOS 上，则添加特定的处理器架构优化选项
        if sys.platform != "darwin":
            # 对于 PowerPC 64 Little Endian 架构，添加处理器架构为本地的选项
            if platform.machine() == "ppc64le":
                cflags.append("mcpu=native")
            else:
                cflags.append("march=native")

        return cflags


def _get_shared_cflag(compile_only: bool) -> List[str]:
    # 如果在 Windows 系统上，则设置一个特定的共享库标志列表
    if _IS_WINDOWS:
        """
        MSVC `/MD` using python `ucrtbase.dll` lib as runtime.
        https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features?view=msvc-170
        """
        SHARED_FLAG = ["DLL", "MD"]
    else:
        # 如果仅编译，返回一个包含位置无关代码的标志列表
        if compile_only:
            return ["fPIC"]
        # 如果在 macOS 并且使用 Clang 编译器，则返回一个包含共享和位置无关代码标志的列表，以及未定义符号的动态查找标志
        if platform.system() == "Darwin" and "clang" in get_cpp_compiler():
            # This causes undefined symbols to behave the same as linux
            return ["shared", "fPIC", "undefined dynamic_lookup"]
        else:
            # 否则返回一个包含共享和位置无关代码标志的列表
            return ["shared", "fPIC"]

    return SHARED_FLAG


def get_cpp_options(
    cpp_compiler,
    compile_only: bool,
    warning_all: bool = True,
    extra_flags: Sequence[str] = (),
):
    definations: List[str] = []
    include_dirs: List[str] = []
    cflags: List[str] = []
    ldflags: List[str] = []
    libraries_dirs: List[str] = []
    libraries: List[str] = []
    passthough_args: List[str] = []

    # 合并多个函数返回的编译标志列表
    cflags = (
        _get_shared_cflag(compile_only)
        + _get_optimization_cflags()
        + _get_warning_all_cflag(warning_all)
        + _get_cpp_std_cflag()
        + _get_linux_cpp_cflags(cpp_compiler)
    )

    # 将额外的编译标志转为字符串并添加到传递参数列表中
    passthough_args.append(" ".join(extra_flags))

    # 返回所有编译选项的元组
    return (
        definations,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthough_args,
    )


class CppOptions(BuildOptionsBase):
    """
    这是一个用于配置 C++ 编译选项的类，继承自 BuildOptionsBase。
    """
    """
    This class is inherited from BuildOptionsBase, and as cxx build options.
    This option need contains basic cxx build option, which contains:
    1. OS related args.
    2. Toolchains related args.
    3. Cxx standard related args.
    Note:
    1. This Options is good for assist modules build, such as x86_isa_help.
    """

    # 初始化函数，设置编译选项
    def __init__(
        self,
        compile_only: bool,
        warning_all: bool = True,
        extra_flags: Sequence[str] = (),
        use_absolute_path: bool = False,
    ) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 获取 C++ 编译器对象
        self._compiler = get_cpp_compiler()
        # 是否使用绝对路径的标志
        self._use_absolute_path = use_absolute_path
        # 是否仅编译而不链接的标志
        self._compile_only = compile_only

        # 调用函数获取 C++ 编译选项的元组
        (
            definations,          # 宏定义列表
            include_dirs,         # 头文件目录列表
            cflags,               # 编译选项列表
            ldflags,              # 链接选项列表
            libraries_dirs,       # 库文件目录列表
            libraries,            # 需链接的库列表
            passthough_args,      # 需透传的额外参数列表
        ) = get_cpp_options(
            cpp_compiler=self._compiler,
            compile_only=compile_only,
            extra_flags=extra_flags,
            warning_all=warning_all,
        )

        # 将获取的编译选项追加到相应的成员变量列表中
        _append_list(self._definations, definations)
        _append_list(self._include_dirs, include_dirs)
        _append_list(self._cflags, cflags)
        _append_list(self._ldflags, ldflags)
        _append_list(self._libraries_dirs, libraries_dirs)
        _append_list(self._libraries, libraries)
        _append_list(self._passthough_args, passthough_args)
        # 移除重复的编译选项
        self._remove_duplicate_options()
# 返回一个包含与 _GLIBCXX_USE_CXX11_ABI 相关的编译标志的列表，若不在 Windows 平台上则添加相应标志
def _get_glibcxx_abi_build_flags() -> List[str]:
    if not _IS_WINDOWS:
        return ["-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    else:
        return []


# 返回一个包含 "TORCH_INDUCTOR_CPP_WRAPPER" 宏定义的列表
def _get_torch_cpp_wrapper_defination() -> List[str]:
    return ["TORCH_INDUCTOR_CPP_WRAPPER"]


# 返回一个包含 "C10_USING_CUSTOM_GENERATED_MACROS" 宏定义的列表
def _use_custom_generated_macros() -> List[str]:
    return [" C10_USING_CUSTOM_GENERATED_MACROS"]


# 如果不在 Windows 平台且处于 fbcode 环境下，则返回一组 fbcode 内部宏定义的列表
def _use_fb_internal_macros() -> List[str]:
    if not _IS_WINDOWS:
        if config.is_fbcode():
            fb_internal_macros = [
                "C10_USE_GLOG",
                "C10_USE_MINIMAL_GLOG",
                "C10_DISABLE_TENSORIMPL_EXTENSIBILITY",
            ]
            # TODO: this is to avoid FC breakage for fbcode. When using newly
            # generated model.so on an older verion of PyTorch, need to use
            # the v1 version for aoti_torch_create_tensor_from_blob
            create_tensor_from_blob_v1 = "AOTI_USE_CREATE_TENSOR_FROM_BLOB_V1"

            fb_internal_macros.append(create_tensor_from_blob_v1)

            # TODO: remove comments later:
            # Moved to _get_openmp_args
            # openmp_lib = build_paths.openmp_lib()
            # return [f"-Wp,-fopenmp {openmp_lib} {preprocessor_flags}"]
            return fb_internal_macros
        else:
            return []
    else:
        return []


# 根据传入的参数设置标准的系统库路径和编译标志
def _setup_standard_sys_libs(
    cpp_compiler,
    aot_mode: bool,
    use_absolute_path: bool,
):
    from torch._inductor.codecache import _LINKER_SCRIPT

    cflags: List[str] = []
    include_dirs: List[str] = []
    passthough_args: List[str] = []
    if _IS_WINDOWS:
        return cflags, include_dirs, passthough_args

    if config.is_fbcode():
        cflags.append("nostdinc")
        include_dirs.append(build_paths.sleef())
        include_dirs.append(build_paths.cc_include())
        include_dirs.append(build_paths.libgcc())
        include_dirs.append(build_paths.libgcc_arch())
        include_dirs.append(build_paths.libgcc_backward())
        include_dirs.append(build_paths.glibc())
        include_dirs.append(build_paths.linux_kernel())
        include_dirs.append("include")

        if aot_mode and not use_absolute_path:
            linker_script = _LINKER_SCRIPT
        else:
            linker_script = os.path.basename(_LINKER_SCRIPT)

        if _is_clang(cpp_compiler):
            passthough_args.append(" --rtlib=compiler-rt")
            passthough_args.append(" -fuse-ld=lld")
            passthough_args.append(f" -Wl,--script={linker_script}")
            passthough_args.append(" -B" + build_paths.glibc_lib())
            passthough_args.append(" -L" + build_paths.glibc_lib())

    return cflags, include_dirs, passthough_args


# 使用 functools.lru_cache 装饰器，缓存并返回一个路径字符串，指向 cpp_prefix.h 文件
@functools.lru_cache
def _cpp_prefix_path() -> str:
    from torch._inductor.codecache import write  # TODO

    path = Path(Path(__file__).parent).parent / "codegen/cpp_prefix.h"
    # 使用上下文管理器打开指定路径的文件
    with path.open() as f:
        # 读取文件的全部内容
        content = f.read()
        # 调用 write 函数，将文件内容写入目标位置，获取写入结果中的文件名
        _, filename = write(
            content,
            "h",
        )
    # 返回 write 函数调用结果中的文件名作为函数返回值
    return filename
# 获取选择的向量指令集的构建参数
def _get_build_args_of_chosen_isa(vec_isa: VecISA):
    # 初始化宏列表和构建标志列表
    macros = []
    build_flags = []
    
    # 如果选择的向量指令集不是无效的向量指令集
    if vec_isa != invalid_vec_isa:
        # 后续添加对Windows的支持
        # 遍历向量指令集的构建宏，并深拷贝到宏列表中
        for x in vec_isa.build_macro():
            macros.append(copy.deepcopy(x))
        
        # 获取向量指令集的架构标志，并添加到构建标志列表中
        build_flags = [vec_isa.build_arch_flags()]
        
        # 如果配置为fbcode环境
        if config.is_fbcode():
            # 将向量指令集名称转换为大写字符串
            cap = str(vec_isa).upper()
            # 更新宏列表以反映CPU能力
            macros = [
                f"CPU_CAPABILITY={cap}",
                f"CPU_CAPABILITY_{cap}",
                f"HAVE_{cap}_CPU_DEFINITION",
            ]
    
    # 返回宏列表和构建标志列表作为结果
    return macros, build_flags


# 获取与Torch相关的参数
def _get_torch_related_args(include_pytorch: bool, aot_mode: bool):
    # 导入Torch相关的路径信息
    from torch.utils.cpp_extension import _TORCH_PATH, TORCH_LIB_PATH
    
    # 设置包含目录列表
    include_dirs = [
        os.path.join(_TORCH_PATH, "include"),
        os.path.join(_TORCH_PATH, "include", "torch", "csrc", "api", "include"),
        # 一些内部（旧的）Torch头文件没有正确前缀其包含路径，
        # 因此需要额外添加-Itorch/lib/include/TH。
        os.path.join(_TORCH_PATH, "include", "TH"),
        os.path.join(_TORCH_PATH, "include", "THC"),
    ]
    
    # 设置库目录列表
    libraries_dirs = [TORCH_LIB_PATH]
    
    # 初始化库列表
    libraries = []
    
    # 如果不是在Darwin平台且不在fbcode环境中
    if sys.platform != "darwin" and not config.is_fbcode():
        # 添加常见的Torch库
        libraries = ["torch", "torch_cpu"]
        # 如果不是AOT模式，还需要添加torch_python库
        if not aot_mode:
            libraries.append("torch_python")
    
    # 如果是在Windows平台上
    if _IS_WINDOWS:
        # 添加sleef库
        libraries.append("sleef")
    
    # 对于非ABI兼容模式，无条件引入c10以使用TORCH_CHECK（见PyTorch＃108690）
    if not config.abi_compatible:
        libraries.append("c10")
        libraries_dirs.append(TORCH_LIB_PATH)
    
    # 返回包含目录列表、库目录列表和库列表作为结果
    return include_dirs, libraries_dirs, libraries


# 获取与Python相关的包含目录
def _get_python_include_dirs():
    # 获取Python的包含目录路径
    include_dir = Path(sysconfig.get_path("include"))
    
    # 如果在Darwin上，并且Python可执行文件来自框架，可能会返回不存在的路径
    # 在这种情况下，应使用框架中的Headers文件夹
    if not include_dir.exists() and platform.system() == "Darwin":
        std_lib = Path(sysconfig.get_path("stdlib"))
        include_dir = (std_lib.parent.parent / "Headers").absolute()
    
    # 如果在所需路径下找不到Python.h文件，则发出警告
    if not (include_dir / "Python.h").exists():
        warnings.warn(f"Can't find Python.h in {str(include_dir)}")
    
    # 返回包含Python头文件的路径列表
    return [str(include_dir)]


# 获取与Python相关的参数
def _get_python_related_args():
    # 获取Python相关的包含目录
    python_include_dirs = _get_python_include_dirs()
    
    # 获取Python的包含路径，并根据系统类型选择不同的方案
    python_include_path = sysconfig.get_path(
        "include", scheme="nt" if _IS_WINDOWS else "posix_prefix"
    )
    
    # 如果能够获取Python的包含路径，则添加到包含目录列表中
    if python_include_path is not None:
        python_include_dirs.append(python_include_path)
    
    # 初始化Python库路径列表
    python_lib_path = []
    
    # 如果在Windows平台上
    if _IS_WINDOWS:
        # 获取Python的路径，并构建Python库的路径
        python_path = os.path.dirname(sys.executable)
        python_lib_path = [os.path.join(python_path, "libs")]
    else:
        # 获取Python的配置变量LIBDIR，并将其添加到库路径列表中
        python_lib_path = [sysconfig.get_config_var("LIBDIR")]
    
    # 如果配置为fbcode环境，则添加构建路径中的Python信息
    if config.is_fbcode():
        python_include_dirs.append(build_paths.python())
    
    # 返回Python相关的包含目录和库路径列表作为结果
    return python_include_dirs, python_lib_path


# 缓存修饰器函数，用于检查是否安装了Conda的LLVM OpenMP
@functools.lru_cache(None)
def is_conda_llvm_openmp_installed() -> bool:
    # 尝试运行 conda 命令来列出 llvm-openmp 包的信息，输出结果以 JSON 格式
    command = "conda list llvm-openmp --json"
    # 使用 subprocess 模块执行命令，并捕获其输出
    output = subprocess.check_output(command.split()).decode("utf8")
    # 解析 JSON 格式的输出为 Python 对象，然后判断列表是否非空
    return len(json.loads(output)) > 0
    # 如果捕获到任何 subprocess 相关的异常，返回 False
except subprocess.SubprocessError:
    return False
@functools.lru_cache(None)
def homebrew_libomp() -> Tuple[bool, str]:
    try:
        # 检查是否安装了 `brew`
        subprocess.check_output(["which", "brew"])
        # 获取 `libomp` 的安装路径（如果已安装）
        # 这是 `libomp` 可能被安装的路径
        # 参考 https://github.com/Homebrew/brew/issues/10261#issuecomment-756563567 获取详情
        libomp_path = (
            subprocess.check_output(["brew", "--prefix", "libomp"])
            .decode("utf8")
            .strip()
        )
        # 检查 `libomp` 是否已安装
        omp_available = os.path.exists(libomp_path)
        return omp_available, libomp_path
    except subprocess.SubprocessError:
        # 如果出现任何 subprocess 错误，则返回 False 和空字符串
        return False, ""


def _get_openmp_args(cpp_compiler):
    cflags: List[str] = []
    ldflags: List[str] = []
    include_dir_paths: List[str] = []
    lib_dir_paths: List[str] = []
    libs: List[str] = []
    passthough_args: List[str] = []
    if _IS_MACOS:
        # 如果运行在 macOS 上
        # 根据 https://mac.r-project.org/openmp/ 的建议，正确传递 `openmp` 标志到 MacOS 是通过 `-Xclang`
        cflags.append("Xclang")
        cflags.append("fopenmp")

        # 只有苹果的内置编译器（Apple Clang++）需要 openmp
        omp_available = not _is_apple_clang(cpp_compiler)

        # 检查 `OMP_PREFIX` 环境变量
        omp_prefix = os.getenv("OMP_PREFIX")
        if omp_prefix is not None:
            # 构建头文件路径
            header_path = os.path.join(omp_prefix, "include", "omp.h")
            valid_env = os.path.exists(header_path)
            if valid_env:
                include_dir_paths.append(os.path.join(omp_prefix, "include"))
                lib_dir_paths.append(os.path.join(omp_prefix, "lib"))
            else:
                warnings.warn("environment variable `OMP_PREFIX` is invalid.")
            omp_available = omp_available or valid_env

        # 如果 openmp 不可用，将 `omp` 库添加到链接库列表
        if not omp_available:
            libs.append("omp")

        # 优先使用 `conda install llvm-openmp` 提供的 openmp
        conda_prefix = os.getenv("CONDA_PREFIX")
        if not omp_available and conda_prefix is not None:
            omp_available = is_conda_llvm_openmp_installed()
            if omp_available:
                conda_lib_path = os.path.join(conda_prefix, "lib")
                include_dir_paths.append(os.path.join(conda_prefix, "include"))
                lib_dir_paths.append(conda_lib_path)
                # 在 x86 机器上优先使用 Intel OpenMP
                if os.uname().machine == "x86_64" and os.path.exists(
                    os.path.join(conda_lib_path, "libiomp5.dylib")
                ):
                    libs.append("iomp5")

        # 如果仍然没有可用的 openmp，尝试从 `brew install libomp` 获取
        if not omp_available:
            omp_available, libomp_path = homebrew_libomp()
            if omp_available:
                include_dir_paths.append(os.path.join(libomp_path, "include"))
                lib_dir_paths.append(os.path.join(libomp_path, "lib"))

        # 如果仍然没有可用的 openmp，则在编译错误时让编译器尝试，并提供后续的指令和错误信息
    elif _IS_WINDOWS:
        # 如果运行在 Windows 上
        # /openmp, /openmp:llvm
        # Windows 上的 llvm，新的 openmp：https://devblogs.microsoft.com/cppblog/msvc-openmp-update/
        # msvc openmp：https://learn.microsoft.com/zh-cn/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170
        cflags.append("openmp")
        cflags.append("openmp:experimental")  # MSVC CL
    else:
        # 如果不是在fbcode环境下，则执行以下操作
        if config.is_fbcode():
            # 将构建路径中的OpenMP目录路径添加到包含目录路径列表中
            include_dir_paths.append(build_paths.openmp())

            # 获取OpenMP库路径并构建额外的编译选项
            openmp_lib = build_paths.openmp_lib()
            fb_openmp_extra_flags = f"-Wp,-fopenmp {openmp_lib}"
            # 将额外的编译选项添加到传递的参数列表中
            passthough_args.append(fb_openmp_extra_flags)

            # 添加OpenMP库名称到库列表中
            libs.append("omp")
        else:
            # 如果使用的是clang编译器，添加编译标志以解决找不到omp.h的问题
            if _is_clang(cpp_compiler):
                cflags.append("fopenmp")
                libs.append("gomp")
            else:
                # 否则，添加编译标志以启用OpenMP支持
                cflags.append("fopenmp")
                libs.append("gomp")

    # 返回编译标志、链接标志、包含目录路径、库目录路径、库列表和传递参数列表
    return cflags, ldflags, include_dir_paths, lib_dir_paths, libs, passthough_args
# 返回一个包含 USE_MMAP_SELF 宏的列表，如果 use_mmap_weights 为 True
def get_mmap_self_macro(use_mmap_weights: bool) -> List[str]:
    macros = []
    # 如果 use_mmap_weights 为 True，将 USE_MMAP_SELF 宏添加到 macros 列表中
    if use_mmap_weights:
        macros.append(" USE_MMAP_SELF")
    # 返回包含宏的列表
    return macros


# 获取 C++ Torch 相关的编译选项和配置
def get_cpp_torch_options(
    cpp_compiler,
    vec_isa: VecISA,
    include_pytorch: bool,
    aot_mode: bool,
    compile_only: bool,
    use_absolute_path: bool,
    use_mmap_weights: bool,
):
    # 初始化不同类型的编译选项列表
    definations: List[str] = []
    include_dirs: List[str] = []
    cflags: List[str] = []
    ldflags: List[str] = []
    libraries_dirs: List[str] = []
    libraries: List[str] = []
    passthough_args: List[str] = []

    # 获取 Torch C++ 包装器相关的宏定义
    torch_cpp_wrapper_definations = _get_torch_cpp_wrapper_defination()
    # 获取自定义生成的宏定义
    use_custom_generated_macros_definations = _use_custom_generated_macros()

    # 设置系统标准库相关的编译选项
    (
        sys_libs_cflags,
        sys_libs_include_dirs,
        sys_libs_passthough_args,
    ) = _setup_standard_sys_libs(cpp_compiler, aot_mode, use_absolute_path)

    # 获取选择的 ISA 架构相关的宏定义和构建参数
    isa_macros, isa_ps_args_build_flags = _get_build_args_of_chosen_isa(vec_isa)

    # 获取与 Torch 相关的包含目录、库目录和库文件列表
    (
        torch_include_dirs,
        torch_libraries_dirs,
        torch_libraries,
    ) = _get_torch_related_args(include_pytorch=include_pytorch, aot_mode=aot_mode)

    # 获取与 Python 相关的包含目录和库目录
    python_include_dirs, python_libraries_dirs = _get_python_related_args()

    # 获取 OpenMP 相关的编译和链接选项
    (
        omp_cflags,
        omp_ldflags,
        omp_include_dir_paths,
        omp_lib_dir_paths,
        omp_lib,
        omp_passthough_args,
    ) = _get_openmp_args(cpp_compiler)

    # 获取与 GLIBCXX ABI 相关的编译参数
    cxx_abi_passthough_args = _get_glibcxx_abi_build_flags()
    # 获取来自 Facebook 内部的宏定义
    fb_macro_passthough_args = _use_fb_internal_macros()

    # 获取基于 mmap_self 宏的列表
    mmap_self_macros = get_mmap_self_macro(use_mmap_weights)

    # 组装所有的宏定义列表
    definations = (
        torch_cpp_wrapper_definations
        + use_custom_generated_macros_definations
        + isa_macros
        + fb_macro_passthough_args
        + mmap_self_macros
    )
    # 组装所有的包含目录列表
    include_dirs = (
        sys_libs_include_dirs
        + python_include_dirs
        + torch_include_dirs
        + omp_include_dir_paths
    )
    # 组装所有的编译选项列表
    cflags = sys_libs_cflags + omp_cflags
    # 组装所有的链接选项列表
    ldflags = omp_ldflags
    # 组装所有的库目录列表
    libraries_dirs = python_libraries_dirs + torch_libraries_dirs + omp_lib_dir_paths
    # 组装所有的库文件列表
    libraries = torch_libraries + omp_lib
    # 组装所有的透传参数列表
    passthough_args = (
        sys_libs_passthough_args
        + isa_ps_args_build_flags
        + cxx_abi_passthough_args
        + omp_passthough_args
    )

    # 返回所有的编译选项和配置作为元组
    return (
        definations,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthough_args,
    )


# CppTorchOptions 类，继承自 CppOptions 类
class CppTorchOptions(CppOptions):
    """
    This class is inherited from CppTorchOptions, which automatic contains
    base cxx build options. And then it will maintains torch related build
    args.
    1. Torch include_directories, libraries, libraries_directories.
    2. Python include_directories, libraries, libraries_directories.
    3. OpenMP related.
    4. Torch MACROs.
    5. MISC
    """
    # 初始化函数，用于创建一个新的对象实例
    def __init__(
        self,
        vec_isa: VecISA,
        include_pytorch: bool = False,
        warning_all: bool = True,
        aot_mode: bool = False,
        compile_only: bool = False,
        use_absolute_path: bool = False,
        use_mmap_weights: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
    ) -> None:
        # 调用父类的初始化函数，传递一些参数
        super().__init__(
            compile_only=compile_only,
            warning_all=warning_all,
            extra_flags=extra_flags,
            use_absolute_path=use_absolute_path,
        )
    
        # 设置当前对象的 AOT 模式属性
        self._aot_mode = aot_mode
    
        # 调用函数获取与 PyTorch C++ 编译相关的各种选项
        (
            torch_definations,
            torch_include_dirs,
            torch_cflags,
            torch_ldflags,
            torch_libraries_dirs,
            torch_libraries,
            torch_passthough_args,
        ) = get_cpp_torch_options(
            cpp_compiler=self._compiler,  # 使用当前对象的编译器属性
            vec_isa=vec_isa,  # 使用传入的向量 ISA 参数
            include_pytorch=include_pytorch,  # 是否包含 PyTorch 的相关选项
            aot_mode=aot_mode,  # AOT 模式参数
            compile_only=compile_only,  # 仅编译模式参数
            use_absolute_path=use_absolute_path,  # 是否使用绝对路径参数
            use_mmap_weights=use_mmap_weights,  # 是否使用内存映射权重参数
        )
    
        # 如果是仅编译模式，则清空库目录和库文件列表
        if compile_only:
            torch_libraries_dirs = []
            torch_libraries = []
    
        # 将获取到的各项编译选项附加到当前对象的各个列表属性中
        _append_list(self._definations, torch_definations)
        _append_list(self._include_dirs, torch_include_dirs)
        _append_list(self._cflags, torch_cflags)
        _append_list(self._ldflags, torch_ldflags)
        _append_list(self._libraries_dirs, torch_libraries_dirs)
        _append_list(self._libraries, torch_libraries)
        _append_list(self._passthough_args, torch_passthough_args)
        # 移除重复的选项，确保列表中的选项唯一
        self._remove_duplicate_options()
def _set_gpu_runtime_env() -> None:
    # 如果运行环境是在 Facebook 代码库中，且未使用 HIP（AMD GPU 加速），并且没有设置 CUDA_HOME 或 CUDA_PATH 环境变量
    if (
        config.is_fbcode()
        and torch.version.hip is None
        and "CUDA_HOME" not in os.environ
        and "CUDA_PATH" not in os.environ
    ):
        # 设置 CUDA_HOME 环境变量为构建路径中的 CUDA 路径
        os.environ["CUDA_HOME"] = build_paths.cuda()


def _transform_cuda_paths(lpaths):
    # 处理两种情况：
    # 1. Meta 内部的 cuda-12 安装，库位于 lib/cuda-12 和 lib/cuda-12/stubs 中
    # 2. Linux 机器可能将 CUDA 安装在 lib64/ 或 lib/ 下
    for i, path in enumerate(lpaths):
        # 如果 CUDA_HOME 环境变量已设置，并且当前路径以 CUDA_HOME 开头，并且路径中不存在 libcudart_static.a 文件
        if (
            "CUDA_HOME" in os.environ
            and path.startswith(os.environ["CUDA_HOME"])
            and not os.path.exists(f"{path}/libcudart_static.a")
        ):
            # 在当前路径下搜索 libcudart_static.a 文件，找到后将 lpaths[i] 更新为该路径，并添加其下的 "stubs" 目录到 lpaths
            for root, dirs, files in os.walk(path):
                if "libcudart_static.a" in files:
                    lpaths[i] = os.path.join(path, root)
                    lpaths.append(os.path.join(lpaths[i], "stubs"))
                    break


def get_cpp_torch_cuda_options(cuda: bool, aot_mode: bool = False):
    definations: List[str] = []  # 定义编译时的宏定义列表
    include_dirs: List[str] = []  # 定义包含文件目录列表
    cflags: List[str] = []  # 定义编译时的 CFLAGS 列表
    ldflags: List[str] = []  # 定义链接时的 LDFLAGS 列表
    libraries_dirs: List[str] = []  # 定义库文件目录列表
    libraries: List[str] = []  # 定义链接的库文件列表
    passthough_args: List[str] = []  # 定义传递的额外参数列表

    # 如果运行环境是在 Facebook 代码库中，并且未设置 CUDA_HOME 或 CUDA_PATH 环境变量
    if (
        config.is_fbcode()
        and "CUDA_HOME" not in os.environ
        and "CUDA_PATH" not in os.environ
    ):
        # 设置 CUDA_HOME 环境变量为构建路径中的 CUDA 路径
        os.environ["CUDA_HOME"] = build_paths.cuda()

    # 导入 torch.utils.cpp_extension 模块
    from torch.utils import cpp_extension

    # 获取 CUDA 模式下的包含文件路径
    include_dirs = cpp_extension.include_paths(cuda)
    # 获取 CUDA 模式下的库文件路径
    libraries_dirs = cpp_extension.library_paths(cuda)

    # 如果是 CUDA 模式
    if cuda:
        # 添加相应的宏定义到 definations 列表中
        definations.append(" USE_ROCM" if torch.version.hip else " USE_CUDA")

        # 如果使用 HIP（AMD GPU 加速）
        if torch.version.hip is not None:
            if config.is_fbcode():
                # 在 Facebook 代码库中，添加 amdhip64 库文件到 libraries 列表中
                libraries += ["amdhip64"]
            else:
                # 在非 Facebook 代码库中，添加 c10_hip 和 torch_hip 库文件到 libraries 列表中，并添加 __HIP_PLATFORM_AMD__ 到宏定义列表中
                libraries += ["c10_hip", "torch_hip"]
                definations.append(" __HIP_PLATFORM_AMD__")
        else:
            # 如果未使用 HIP（AMD GPU 加速）
            if config.is_fbcode():
                # 在 Facebook 代码库中，添加 cuda 库文件到 libraries 列表中
                libraries += ["cuda"]
            else:
                # 在非 Facebook 代码库中，添加 c10_cuda、cuda 和 torch_cuda 库文件到 libraries 列表中
                libraries += ["c10_cuda", "cuda", "torch_cuda"]

    # 如果是 AOT 编译模式
    if aot_mode:
        # 添加包含前缀目录到 include_dirs 列表中
        cpp_prefix_include_dir = [f"{os.path.dirname(_cpp_prefix_path())}"]
        include_dirs += cpp_prefix_include_dir

        # 如果是 CUDA 模式且未使用 HIP（AMD GPU 加速）
        if cuda and torch.version.hip is None:
            # 调用 _transform_cuda_paths 函数处理 libraries_dirs 列表
            _transform_cuda_paths(libraries_dirs)

    # 如果运行环境是在 Facebook 代码库中
    if config.is_fbcode():
        # 如果使用 HIP（AMD GPU 加速）
        if torch.version.hip is not None:
            # 添加 ROCm 的 include 目录到 include_dirs 列表中
            include_dirs.append(os.path.join(build_paths.rocm(), "include"))
        else:
            # 添加 CUDA 的 include 目录到 include_dirs 列表中
            include_dirs.append(os.path.join(build_paths.cuda(), "include"))

    # 如果是 AOT 编译模式且是 CUDA 模式，并且运行环境是在 Facebook 代码库中
    if aot_mode and cuda and config.is_fbcode():
        # 如果未使用 HIP（AMD GPU 加速）
        if torch.version.hip is None:
            # 添加静态链接参数到 passthough_args 列表中，以便在 Linux 上进行更好的静态链接
            passthough_args = ["-Wl,-Bstatic -lcudart_static -Wl,-Bdynamic"]
    return (
        definations,      # 返回定义的所有宏定义
        include_dirs,     # 返回包含文件的目录列表
        cflags,           # 返回编译器标志
        ldflags,          # 返回链接器标志
        libraries_dirs,   # 返回库文件目录列表
        libraries,        # 返回要链接的库列表
        passthough_args,  # 返回透传的额外参数
    )
class CppTorchCudaOptions(CppTorchOptions):
    """
    This class is inherited from CppTorchOptions, which automatically contains
    base C++ build options and Torch common build options. It manages CUDA device
    related build arguments.
    """

    def __init__(
        self,
        vec_isa: VecISA,
        include_pytorch: bool = False,
        cuda: bool = True,
        aot_mode: bool = False,
        compile_only: bool = False,
        use_absolute_path: bool = False,
        use_mmap_weights: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
    ) -> None:
        """
        Initialize the CppTorchCudaOptions object with specific build options.

        Args:
        - vec_isa: Vector instruction set architecture.
        - include_pytorch: Whether to include PyTorch.
        - cuda: Whether CUDA support is enabled.
        - aot_mode: Whether to enable Ahead-Of-Time compilation mode.
        - compile_only: Whether to compile only without linking.
        - use_absolute_path: Whether to use absolute paths.
        - use_mmap_weights: Whether to use memory-mapped weights.
        - shared: Whether to use shared libraries.
        - extra_flags: Additional build flags.
        """
        super().__init__(
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            compile_only=compile_only,
            use_absolute_path=use_absolute_path,
            use_mmap_weights=use_mmap_weights,
            extra_flags=extra_flags,
        )

        # Initialize lists to store CUDA specific build options
        cuda_definations: List[str] = []
        cuda_include_dirs: List[str] = []
        cuda_cflags: List[str] = []
        cuda_ldflags: List[str] = []
        cuda_libraries_dirs: List[str] = []
        cuda_libraries: List[str] = []
        cuda_passthough_args: List[str] = []

        # Retrieve CUDA specific build options using helper function
        (
            cuda_definations,
            cuda_include_dirs,
            cuda_cflags,
            cuda_ldflags,
            cuda_libraries_dirs,
            cuda_libraries,
            cuda_passthough_args,
        ) = get_cpp_torch_cuda_options(cuda=cuda, aot_mode=aot_mode)

        # Adjust CUDA libraries based on compile_only mode
        if compile_only:
            cuda_libraries_dirs = []
            cuda_libraries = []

        # Append CUDA build options to respective lists
        _append_list(self._definations, cuda_definations)
        _append_list(self._include_dirs, cuda_include_dirs)
        _append_list(self._cflags, cuda_cflags)
        _append_list(self._ldflags, cuda_ldflags)
        _append_list(self._libraries_dirs, cuda_libraries_dirs)
        _append_list(self._libraries, cuda_libraries)
        _append_list(self._passthough_args, cuda_passthough_args)

        # Remove duplicate build options across lists
        self._remove_duplicate_options()


def get_name_and_dir_from_output_file_path(
    aot_mode: bool, use_absolute_path: bool, file_path: str
):
    """
    Extracts the filename and directory from the given output file path.

    Args:
    - aot_mode: Whether in Ahead-Of-Time compilation mode.
    - use_absolute_path: Whether to use absolute path for the file.
    - file_path: Output file path to extract name and directory from.

    Returns:
    - name: Extracted filename without extension.
    - dir: Extracted directory path.
    """
    # Extract filename and extension
    name_and_ext = os.path.basename(file_path)
    name, ext = os.path.splitext(name_and_ext)
    # Extract directory path
    dir = os.path.dirname(file_path)

    # Adjust directory path based on FBCode environment
    if config.is_fbcode():
        if not (aot_mode and not use_absolute_path):
            dir = "."

    return name, dir


class CppBuilder:
    """
    CppBuilder is a C++ JIT builder that supports Windows, Linux, and MacOS.
    """
    Args:
        name:
            1. 构建目标文件的名称，最终的目标文件会自动附加扩展名。
            2. 由于 CppBuilder 支持多个操作系统，它会根据操作系统维护不同的扩展名。
        sources:
            待构建的源代码文件列表。
        BuildOption:
            构建器的构建选项。
        output_dir:
            1. 目标文件将输出到的输出目录。
            2. 默认值为空字符串，表示使用当前目录作为输出目录。
            3. 最终目标文件路径为: output_dir/name.ext
    """
    # 获取 Python 模块的扩展名
    def __get_python_module_ext(self) -> str:
        SHARED_LIB_EXT = ".pyd" if _IS_WINDOWS else ".so"
        return SHARED_LIB_EXT

    # 获取对象文件的扩展名
    def __get_object_ext(self) -> str:
        EXT = ".obj" if _IS_WINDOWS else ".o"
        return EXT

    # 初始化方法
    def __init__(
        self,
        name: str,
        sources: Union[str, List[str]],
        BuildOption: BuildOptionsBase,
        output_dir: str = "",
    ):
        # 获取构建命令行的字符串表示
        def get_command_line(self) -> str:
            def format_build_command(
                compiler,
                sources,
                include_dirs_args,
                definations_args,
                cflags_args,
                ldflags_args,
                libraries_args,
                libraries_dirs_args,
                passthougn_args,
                target_file,
            ):
                if _IS_WINDOWS:
                    # 构建 Windows 平台下的编译命令
                    # 参考链接：https://learn.microsoft.com/en-us/cpp/build/walkthrough-compile-a-c-program-on-the-command-line?view=msvc-1704
                    # 参考链接：https://stackoverflow.com/a/31566153
                    cmd = (
                        f"{compiler} {include_dirs_args} {definations_args} {cflags_args} {sources} "
                        f"{passthougn_args} /LD /Fe{target_file} /link {libraries_dirs_args} {libraries_args} {ldflags_args} "
                    )
                    cmd = cmd.replace("\\", "/")
                else:
                    compile_only_arg = "-c" if self._compile_only else ""
                    # 构建非 Windows 平台下的编译命令
                    cmd = re.sub(
                        r"[ \n]+",
                        " ",
                        f"""
                        {compiler} {sources} {definations_args} {cflags_args} {include_dirs_args}
                        {passthougn_args} {ldflags_args} {libraries_args} {libraries_dirs_args} {compile_only_arg} -o {target_file}
                        """,
                    ).strip()
                return cmd

            # 调用内部函数 format_build_command 获取构建命令行
            command_line = format_build_command(
                compiler=self._compiler,
                sources=self._sources_args,
                include_dirs_args=self._include_dirs_args,
                definations_args=self._definations_args,
                cflags_args=self._cflags_args,
                ldflags_args=self._ldflags_args,
                libraries_args=self._libraries_args,
                libraries_dirs_args=self._libraries_dirs_args,
                passthougn_args=self._passthough_parameters_args,
                target_file=self._target_file,
            )
            return command_line

        # 返回目标文件的路径
        def get_target_file_path(self):
            return self._target_file
    # 定义一个方法 `build`，返回一个元组，包含状态码和目标文件名
    def build(self) -> Tuple[int, str]:
        """
        必须在 Windows 系统中创建临时目录以存储对象文件。
        构建完成后，删除临时目录以节省磁盘空间。
        """
        # 如果输出目录不存在，则创建它
        _create_if_dir_not_exist(self._output_dir)
        # 构建临时目录路径，用于存放临时文件
        _build_tmp_dir = os.path.join(
            self._output_dir, f"{self._name}_{_BUILD_TEMP_DIR}"
        )
        # 如果临时目录不存在，则创建它
        _create_if_dir_not_exist(_build_tmp_dir)

        # 获取构建命令行
        build_cmd = self.get_command_line()

        # 运行构建命令行，设置当前工作目录为临时构建目录
        status = run_command_line(build_cmd, cwd=_build_tmp_dir)

        # 删除临时构建目录
        _remove_dir(_build_tmp_dir)
        # 返回运行结果状态码和目标文件名
        return status, self._target_file
```