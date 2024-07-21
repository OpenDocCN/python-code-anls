# `.\pytorch\torch\utils\cpp_extension.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块
import copy  # 导入copy模块，用于复制对象
import glob  # 导入glob模块，用于文件路径名匹配
import importlib  # 导入importlib模块，用于动态加载模块
import importlib.abc  # 导入importlib.abc模块，包含抽象基类
import os  # 导入os模块，提供与操作系统交互的功能
import re  # 导入re模块，用于正则表达式操作
import shlex  # 导入shlex模块，用于解析shell命令字符串
import shutil  # 导入shutil模块，提供高级文件操作功能
import setuptools  # 导入setuptools模块，用于打包和分发Python程序
import subprocess  # 导入subprocess模块，用于创建新的进程
import sys  # 导入sys模块，提供与Python解释器交互的功能
import sysconfig  # 导入sysconfig模块，获取Python的配置信息
import warnings  # 导入warnings模块，用于警告控制
import collections  # 导入collections模块，提供额外的数据结构
from pathlib import Path  # 从pathlib模块中导入Path类，用于处理文件和目录路径
import errno  # 导入errno模块，包含系统错误码

import torch  # 导入torch模块，PyTorch深度学习库的核心
import torch._appdirs  # 导入torch._appdirs模块，用于处理应用程序目录
from .file_baton import FileBaton  # 导入当前目录下file_baton模块中的FileBaton类
from ._cpp_extension_versioner import ExtensionVersioner  # 导入当前目录下_cpp_extension_versioner模块中的ExtensionVersioner类
from .hipify import hipify_python  # 导入当前目录下hipify模块中的hipify_python函数
from .hipify.hipify_python import GeneratedFileCleaner  # 导入当前目录下hipify.hipify_python模块中的GeneratedFileCleaner类
from typing import Dict, List, Optional, Union, Tuple  # 导入类型提示需要的类型

from torch.torch_version import TorchVersion, Version  # 从torch.torch_version模块中导入TorchVersion和Version类

from setuptools.command.build_ext import build_ext  # 从setuptools.command.build_ext模块中导入build_ext类

IS_WINDOWS = sys.platform == 'win32'  # 检查操作系统是否为Windows
IS_MACOS = sys.platform.startswith('darwin')  # 检查操作系统是否为macOS
IS_LINUX = sys.platform.startswith('linux')  # 检查操作系统是否为Linux
LIB_EXT = '.pyd' if IS_WINDOWS else '.so'  # 根据操作系统确定库文件扩展名
EXEC_EXT = '.exe' if IS_WINDOWS else ''  # 根据操作系统确定可执行文件扩展名
CLIB_PREFIX = '' if IS_WINDOWS else 'lib'  # 根据操作系统确定C库文件名前缀
CLIB_EXT = '.dll' if IS_WINDOWS else '.so'  # 根据操作系统确定C库文件扩展名
SHARED_FLAG = '/DLL' if IS_WINDOWS else '-shared'  # 根据操作系统确定共享库标志

_HERE = os.path.abspath(__file__)  # 获取当前脚本文件的绝对路径
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))  # 获取torch安装路径的上一级目录
TORCH_LIB_PATH = os.path.join(_TORCH_PATH, 'lib')  # 构建torch库的路径

SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()  # 如果是Windows，指定子进程解码参数为'oem'，否则为空元组

MINIMUM_GCC_VERSION = (5, 0, 0)  # 定义最低要求的GCC版本
MINIMUM_MSVC_VERSION = (19, 0, 24215)  # 定义最低要求的MSVC版本

VersionRange = Tuple[Tuple[int, ...], Tuple[int, ...]]  # 定义版本范围的类型别名
VersionMap = Dict[str, VersionRange]  # 定义版本映射的类型别名

# CUDA版本与最低要求的GCC版本范围的映射关系
CUDA_GCC_VERSIONS: VersionMap = {
    '11.0': (MINIMUM_GCC_VERSION, (10, 0)),
    '11.1': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.2': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.3': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.4': ((6, 0, 0), (12, 0)),
    '11.5': ((6, 0, 0), (12, 0)),
    '11.6': ((6, 0, 0), (12, 0)),
    '11.7': ((6, 0, 0), (12, 0)),
}

MINIMUM_CLANG_VERSION = (3, 3, 0)  # 定义最低要求的Clang版本
# CUDA版本与最低要求的Clang版本范围的映射关系
CUDA_CLANG_VERSIONS: VersionMap = {
    '11.1': (MINIMUM_CLANG_VERSION, (11, 0)),
    '11.2': (MINIMUM_CLANG_VERSION, (12, 0)),
    '11.3': (MINIMUM_CLANG_VERSION, (12, 0)),
    '11.4': (MINIMUM_CLANG_VERSION, (13, 0)),
    '11.5': (MINIMUM_CLANG_VERSION, (13, 0)),
    '11.6': (MINIMUM_CLANG_VERSION, (14, 0)),
    '11.7': (MINIMUM_CLANG_VERSION, (14, 0)),
}

__all__ = ["get_default_build_root", "check_compiler_ok_for_platform", "get_compiler_abi_compatibility_and_version", "BuildExtension",
           "CppExtension", "CUDAExtension", "include_paths", "library_paths", "load", "load_inline", "is_ninja_available",
           "verify_ninja_availability", "remove_extension_h_precompiler_headers", "get_cxx_compiler", "check_compiler_is_gcc"]
# 从Python标准库 < 3.9 直接获取
# 详情请参阅 https://github.com/pytorch/pytorch/issues/48617
def _nt_quote_args(args: Optional[List[str]]) -> List[str]:
    # 定义函数，用于在DOS/Windows约定中引用命令行参数
    # 将包含空格的每个参数用双引号包裹起来，并返回一个新的参数列表
    def quote_args(args):
        # 如果参数列表为空，则返回空列表
        if not args:
            return []
        # 遍历参数列表，对于包含空格的参数，在其两端添加双引号；否则保持原样
        return [f'"{arg}"' if ' ' in arg else arg for arg in args]
def _find_cuda_home() -> Optional[str]:
    """Find the CUDA install path."""
    # 获取环境变量中的 CUDA_HOME 或 CUDA_PATH，作为猜测 #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # 如果未找到 CUDA_HOME 或 CUDA_PATH，则进行猜测 #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            # 如果 nvcc 可执行文件存在，则基于其路径推测 CUDA_HOME
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # 如果在猜测 #2 中仍未找到，则根据操作系统查找 CUDA 安装路径
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            # 如果找到的路径不存在，则置为 None
            if not os.path.exists(cuda_home):
                cuda_home = None
    # 如果 cuda_home 被找到，并且 CUDA 并不可用，则打印警告信息
    if cuda_home and not torch.cuda.is_available():
        print(f"No CUDA runtime is found, using CUDA_HOME='{cuda_home}'",
              file=sys.stderr)
    return cuda_home

def _find_rocm_home() -> Optional[str]:
    """Find the ROCm install path."""
    # 获取环境变量中的 ROCM_HOME 或 ROCM_PATH，作为猜测 #1
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        # 如果未找到 ROCM_HOME 或 ROCM_PATH，则进行猜测 #2
        hipcc_path = shutil.which('hipcc')
        if hipcc_path is not None:
            # 如果 hipcc 可执行文件存在，则基于其路径推测 ROCM_HOME
            rocm_home = os.path.dirname(os.path.dirname(
                os.path.realpath(hipcc_path)))
            # 可能路径是 <ROCM_HOME>/hip/bin/hipcc 或 <ROCM_HOME>/bin/hipcc
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        else:
            # 如果在猜测 #2 中仍未找到，则使用预设的 ROCm 安装路径
            fallback_path = '/opt/rocm'
            if os.path.exists(fallback_path):
                rocm_home = fallback_path
    # 如果 rocm_home 被找到，并且当前环境不支持 ROCm，则打印警告信息
    if rocm_home and torch.version.hip is None:
        print(f"No ROCm runtime is found, using ROCM_HOME='{rocm_home}'",
              file=sys.stderr)
    return rocm_home


def _join_rocm_home(*paths) -> str:
    """
    Join paths with ROCM_HOME, or raises an error if ROCM_HOME is not set.

    This is basically a lazy way of raising an error for missing $ROCM_HOME
    only once we need to get any ROCm-specific path.
    """
    # 如果 ROCM_HOME 未设置，则抛出错误
    if ROCM_HOME is None:
        raise OSError('ROCM_HOME environment variable is not set. '
                      'Please set it to your ROCm install root.')
    elif IS_WINDOWS:
        raise OSError('Building PyTorch extensions using '
                      'ROCm and Windows is not supported.')
    # 使用 ROCM_HOME 和传入的路径拼接成完整路径
    return os.path.join(ROCM_HOME, *paths)
# 定义警告消息模板，用于指示用户使用错误的编译器来编译 PyTorch 扩展
WRONG_COMPILER_WARNING = '''
                               !! WARNING !!

Your compiler ({user_compiler}) is not compatible with the compiler Pytorch was
built with for this platform, which is {pytorch_compiler} on {platform}. Please
use {pytorch_compiler} to to compile your extension. Alternatively, you may
compile PyTorch from source using {user_compiler}, and then you can also use
{user_compiler} to compile your extension.

See https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md for help
with compiling PyTorch from source.
                               !! WARNING !!
'''

# 当检测到的 CUDA 版本与用于编译 PyTorch 的版本不匹配时显示的警告消息
CUDA_MISMATCH_MESSAGE = '''
The detected CUDA version ({0}) mismatches the version that was used to compile
PyTorch ({1}). Please make sure to use the same CUDA versions.
'''

# 当检测到的 CUDA 版本有轻微版本不匹配时显示的警告消息
CUDA_MISMATCH_WARN = "The detected CUDA version ({0}) has a minor version mismatch with the version that was used to compile PyTorch ({1}). Most likely this shouldn't be a problem."

# 当系统中未找到 CUDA 时显示的错误消息，提示用户设置 CUDA_HOME 或 CUDA_PATH 环境变量
CUDA_NOT_FOUND_MESSAGE = '''
CUDA was not found on the system, please set the CUDA_HOME or the CUDA_PATH
environment variable or add NVCC to your system PATH. The extension compilation will fail.
'''

# 检测 ROCm 的根目录，并设置 HIP_HOME
ROCM_HOME = _find_rocm_home()
HIP_HOME = _join_rocm_home('hip') if ROCM_HOME else None

# 根据当前是否为 HIP 扩展来设置 IS_HIP_EXTENSION 的值
IS_HIP_EXTENSION = True if ((ROCM_HOME is not None) and (torch.version.hip is not None)) else False

# 获取当前使用的 ROCm 版本
ROCM_VERSION = None
if torch.version.hip is not None:
    ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split('.')[:2])

# 如果 PyTorch 是使用 CUDA 编译的，则获取 CUDA 的根目录
CUDA_HOME = _find_cuda_home() if torch.cuda._is_compiled() else None

# 获取 CUDNN 的根目录，可以通过 CUDNN_HOME 或 CUDNN_PATH 环境变量获取
CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')

# 定义 PyTorch 发布版本号的正则表达式模式
BUILT_FROM_SOURCE_VERSION_PATTERN = re.compile(r'\d+\.\d+\.\d+\w+\+\w+')

# 常见的 MSVC 编译器标志列表，用于 Microsoft Visual Studio 编译器
COMMON_MSVC_FLAGS = ['/MD', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018', '/wd4190', '/wd4624', '/wd4067', '/wd4068', '/EHsc']

# NVCC 编译器常见标志列表，用于 NVIDIA CUDA 编译器
COMMON_NVCC_FLAGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
    '--expt-relaxed-constexpr'
]

# HIP 编译器常见标志列表，用于 ROCm 平台上的编译
COMMON_HIP_FLAGS = [
    '-fPIC',
    '-D__HIP_PLATFORM_AMD__=1',
    '-DUSE_ROCM=1',
    '-DHIPBLAS_V2',
]

# HIP C++ 编译器常见标志列表
COMMON_HIPCC_FLAGS = [
    '-DCUDA_HAS_FP16=1',
    '-D__HIP_NO_HALF_OPERATORS__=1',
    '-D__HIP_NO_HALF_CONVERSIONS__=1',
]
JIT_EXTENSION_VERSIONER = ExtensionVersioner()

PLAT_TO_VCVARS = {
    'win32' : 'x86',
    'win-amd64' : 'x86_amd64',
}

# 返回当前操作系统下的 C++ 编译器名称
def get_cxx_compiler():
    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')
    return compiler

# 检查当前 Torch 版本是否为二进制构建
def _is_binary_build() -> bool:
    return not BUILT_FROM_SOURCE_VERSION_PATTERN.match(torch.version.__version__)

# 返回当前操作系统下支持的编译器列表
def _accepted_compilers_for_platform() -> List[str]:
    # gnu-c++ 和 gnu-cc 是 conda 的 gcc 编译器
    return ['clang++', 'clang'] if IS_MACOS else ['g++', 'gcc', 'gnu-c++', 'gnu-cc', 'clang++', 'clang']

# 写入文件的内容，但如果内容已经正确则不写入（避免触发重新编译）
def _maybe_write(filename, new_content):
    r'''
    Equivalent to writing the content into the file but will not touch the file
    if it already had the right content (to avoid triggering recompile).
    '''
    if os.path.exists(filename):
        with open(filename) as f:
            content = f.read()

        if content == new_content:
            # 文件已包含正确内容！
            return

    with open(filename, 'w') as source_file:
        source_file.write(new_content)

# 返回默认的构建根目录，用于存放扩展模块的构建文件夹
def get_default_build_root() -> str:
    """
    Return the path to the root folder under which extensions will built.

    For each extension module built, there will be one folder underneath the
    folder returned by this function. For example, if ``p`` is the path
    returned by this function and ``ext`` the name of an extension, the build
    folder for the extension will be ``p/ext``.

    This directory is **user-specific** so that multiple users on the same
    machine won't meet permission issues.
    """
    return os.path.realpath(torch._appdirs.user_cache_dir(appname='torch_extensions'))


# 验证当前平台下的编译器是否符合预期
def check_compiler_ok_for_platform(compiler: str) -> bool:
    """
    Verify that the compiler is the expected one for the current platform.

    Args:
        compiler (str): The compiler executable to check.

    Returns:
        True if the compiler is gcc/g++ on Linux or clang/clang++ on macOS,
        and always True for Windows.
    """
    if IS_WINDOWS:
        return True
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        return False
    # 使用 os.path.realpath 解析任何符号链接，特别是从 'c++' 到 'g++' 的情况。
    compiler_path = os.path.realpath(compiler_path)
    # 检查编译器名称
    if any(name in compiler_path for name in _accepted_compilers_for_platform()):
        return True
    # 如果使用了编译器包装器，尝试通过带 -v 标志调用它来推断实际编译器
    env = os.environ.copy()
    env['LC_ALL'] = 'C'  # 不本地化输出
    version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    if IS_LINUX:
        # 如果运行环境是 Linux
        
        # 编译正则表达式模式，用于匹配以"COLLECT_GCC="开头的行
        pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
        
        # 在版本字符串中查找匹配正则表达式模式的结果
        results = re.findall(pattern, version_string)
        
        # 如果结果不唯一，说明存在多个匹配，这种情况下检查是否包含"clang version"来确定是否是 Clang 编译器
        if len(results) != 1:
            # 在 Ubuntu 系统上，Clang 有时被称为 "Ubuntu clang version"
            return 'clang version' in version_string
        
        # 获取真实的编译器路径并去除两端的空白字符
        compiler_path = os.path.realpath(results[0].strip())
        
        # 在 RHEL/CentOS 上，'c++' 是 gcc 的编译器包装器
        if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
            return True
        
        # 检查编译器路径是否包含平台支持的任何编译器名称
        return any(name in compiler_path for name in _accepted_compilers_for_platform())
    
    if IS_MACOS:
        # 如果运行环境是 macOS
        
        # 检查版本字符串是否以"Apple clang"开头，确定是否是 Apple 的 Clang 编译器
        return version_string.startswith("Apple clang")
    
    # 默认情况下，返回 False，表示不是 Linux 也不是 macOS
    return False
# 检查编译器是否与 PyTorch ABI 兼容，并返回编译器的版本信息
def get_compiler_abi_compatibility_and_version(compiler) -> Tuple[bool, TorchVersion]:
    """
    Determine if the given compiler is ABI-compatible with PyTorch alongside its version.

    Args:
        compiler (str): The compiler executable name to check (e.g. ``g++``).
            Must be executable in a shell process.

    Returns:
        A tuple that contains a boolean that defines if the compiler is (likely) ABI-incompatible with PyTorch,
        followed by a `TorchVersion` string that contains the compiler version separated by dots.
    """
    # 如果不是二进制构建，则假定不兼容，并返回默认版本 '0.0.0'
    if not _is_binary_build():
        return (True, TorchVersion('0.0.0'))
    
    # 如果设置了环境变量 TORCH_DONT_CHECK_COMPILER_ABI，假定不兼容，并返回默认版本 '0.0.0'
    if os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') in ['ON', '1', 'YES', 'TRUE', 'Y']:
        return (True, TorchVersion('0.0.0'))

    # 首先检查编译器是否符合特定平台的预期编译器
    if not check_compiler_ok_for_platform(compiler):
        # 如果不符合，发出警告，并返回不兼容状态和默认版本 '0.0.0'
        warnings.warn(WRONG_COMPILER_WARNING.format(
            user_compiler=compiler,
            pytorch_compiler=_accepted_compilers_for_platform()[0],
            platform=sys.platform))
        return (False, TorchVersion('0.0.0'))

    # 如果是 macOS 系统，不需要特定的最小版本要求，返回兼容状态和默认版本 '0.0.0'
    if IS_MACOS:
        return (True, TorchVersion('0.0.0'))
    
    try:
        if IS_LINUX:
            # 对于 Linux 系统，获取最小要求的 GCC 版本并检查编译器版本
            minimum_required_version = MINIMUM_GCC_VERSION
            versionstr = subprocess.check_output([compiler, '-dumpfullversion', '-dumpversion'])
            version = versionstr.decode(*SUBPROCESS_DECODE_ARGS).strip().split('.')
        else:
            # 对于 Windows 系统，获取最小要求的 MSVC 版本并检查编译器版本
            minimum_required_version = MINIMUM_MSVC_VERSION
            compiler_info = subprocess.check_output(compiler, stderr=subprocess.STDOUT)
            match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode(*SUBPROCESS_DECODE_ARGS).strip())
            version = ['0', '0', '0'] if match is None else list(match.groups())
    except Exception:
        # 捕获任何异常，发出警告并返回不兼容状态和默认版本 '0.0.0'
        _, error, _ = sys.exc_info()
        warnings.warn(f'Error checking compiler version for {compiler}: {error}')
        return (False, TorchVersion('0.0.0'))

    # 如果编译器版本大于等于所需的最小版本，返回兼容状态和编译器版本
    if tuple(map(int, version)) >= minimum_required_version:
        return (True, TorchVersion('.'.join(version)))

    # 如果编译器版本小于所需的最小版本，发出警告，并返回不兼容状态和编译器版本
    compiler = f'{compiler} {".".join(version)}'
    warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
    return (False, TorchVersion('.'.join(version)))


def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:
    # 检查 CUDA 是否安装，若未安装则抛出运行时错误
    if not CUDA_HOME:
        raise RuntimeError(CUDA_NOT_FOUND_MESSAGE)

    # 获取 nvcc 的路径并获取 CUDA 版本信息
    nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
    cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
    cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)

    if cuda_version is None:
        return

    # 解析 CUDA 版本并与 PyTorch 的 CUDA 版本进行比较
    cuda_str_version = cuda_version.group(1)
    cuda_ver = Version(cuda_str_version)
    if torch.version.cuda is None:
        return
    # 使用 torch.version.cuda 创建一个 Version 对象表示当前的 CUDA 版本
    torch_cuda_version = Version(torch.version.cuda)
    # 如果当前 CUDA 版本与给定的 cuda_ver 不匹配，则进行版本检查和报错处理
    if cuda_ver != torch_cuda_version:
        # 检查是否能访问到 major/minor 属性，这要求 setuptools 版本 >= 49.4.0
        if getattr(cuda_ver, "major", None) is None:
            raise ValueError("setuptools>=49.4.0 is required")
        # 检查主要版本号是否匹配，若不匹配则抛出错误
        if cuda_ver.major != torch_cuda_version.major:
            raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
        # 若版本号不匹配则发出警告
        warnings.warn(CUDA_MISMATCH_WARN.format(cuda_str_version, torch.version.cuda))

    # 检查当前操作系统是否为 Linux，环境变量 TORCH_DONT_CHECK_COMPILER_ABI 未设定为 ON/1/YES/TRUE/Y，且为二进制构建
    if not (sys.platform.startswith('linux') and
            os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') not in ['ON', '1', 'YES', 'TRUE', 'Y'] and
            _is_binary_build()):
        return

    # 根据编译器名称确定 CUDA 编译器版本边界字典
    cuda_compiler_bounds: VersionMap = CUDA_CLANG_VERSIONS if compiler_name.startswith('clang') else CUDA_GCC_VERSIONS

    # 检查给定的 CUDA 版本字符串是否存在于编译器版本边界字典中
    if cuda_str_version not in cuda_compiler_bounds:
        # 若没有定义给定 CUDA 版本的编译器版本边界，则发出警告
        warnings.warn(f'There are no {compiler_name} version bounds defined for CUDA version {cuda_str_version}')
    else:
        # 获取指定 CUDA 版本的最小和最大编译器版本边界
        min_compiler_version, max_excl_compiler_version = cuda_compiler_bounds[cuda_str_version]
        
        # 对特定的 CUDA 11.4.0 版本做特殊处理，调整最大编译器版本边界
        if "V11.4.48" in cuda_version_str and cuda_compiler_bounds == CUDA_GCC_VERSIONS:
            max_excl_compiler_version = (11, 0)
        
        # 将版本号转换为字符串形式
        min_compiler_version_str = '.'.join(map(str, min_compiler_version))
        max_excl_compiler_version_str = '.'.join(map(str, max_excl_compiler_version))

        # 构建版本边界描述字符串
        version_bound_str = f'>={min_compiler_version_str}, <{max_excl_compiler_version_str}'

        # 检查当前编译器版本是否符合最小要求，若不符合则抛出错误
        if compiler_version < TorchVersion(min_compiler_version_str):
            raise RuntimeError(
                f'The current installed version of {compiler_name} ({compiler_version}) is less '
                f'than the minimum required version by CUDA {cuda_str_version} ({min_compiler_version_str}). '
                f'Please make sure to use an adequate version of {compiler_name} ({version_bound_str}).'
            )
        # 检查当前编译器版本是否超过最大允许版本，若超过则抛出错误
        if compiler_version >= TorchVersion(max_excl_compiler_version_str):
            raise RuntimeError(
                f'The current installed version of {compiler_name} ({compiler_version}) is greater '
                f'than the maximum required version by CUDA {cuda_str_version}. '
                f'Please make sure to use an adequate version of {compiler_name} ({version_bound_str}).'
            )
class BuildExtension(build_ext):
    """
    A custom :mod:`setuptools` build extension.

    This :class:`setuptools.build_ext` subclass handles compiler flags
    (e.g. ``-std=c++17``) and supports mixed C++/CUDA compilation.

    When using :class:`BuildExtension`, you can provide a dictionary for
    ``extra_compile_args`` mapping languages (``cxx`` or ``nvcc``) to lists
    of additional compiler flags.

    ``use_ninja`` (bool): If ``True`` (default), attempts to use Ninja for
    faster compilation. Falls back to distutils if Ninja is unavailable.

    .. note::
        Ninja backend uses #CPUS + 2 workers by default. Adjust `MAX_JOBS`
        env var to control worker count.
    """

    @classmethod
    def with_options(cls, **options):
        """Returns a subclass with extended constructor arguments."""
        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get('use_ninja', True)
        if self.use_ninja:
            # Check if ninja is available; otherwise, fall back to distutils.
            msg = ('Attempted to use ninja as the BuildExtension backend but '
                   '{}. Falling back to using the slow distutils backend.')
            if not is_ninja_available():
                warnings.warn(msg.format('we could not find ninja.'))
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True
    # 获取扩展名对应的原始共享库文件名。对于 Python 3，这个名称将会带有 "<SOABI>.so" 后缀，
    # 其中 <SOABI> 通常是类似于 cpython-37m-x86_64-linux-gnu 的内容。
    ext_filename = super().get_ext_filename(ext_name)
    
    # 如果 `no_python_abi_suffix` 为 True，我们将省略 Python 3 ABI 组件。
    # 这样可以更好地使用 setuptools 构建不是 Python 模块的共享库。
    if self.no_python_abi_suffix:
        # 将 ext_filename 按照点号分隔成多个部分，例如 ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"]。
        ext_filename_parts = ext_filename.split('.')
        
        # 省略倒数第二个元素。
        without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        
        # 将列表 without_abi 重新拼接成一个字符串作为新的 ext_filename。
        ext_filename = '.'.join(without_abi)
    
    # 返回处理后的扩展文件名
    return ext_filename


# 在某些平台（如 Windows）上，compiler_cxx 可能不可用。
# 如果可用，使用 self.compiler.compiler_cxx[0] 作为编译器；否则调用 get_cxx_compiler()。
def _check_abi(self) -> Tuple[str, TorchVersion]:
    if hasattr(self.compiler, 'compiler_cxx'):
        compiler = self.compiler.compiler_cxx[0]
    else:
        compiler = get_cxx_compiler()
    
    # 获取编译器的 ABI 兼容性和版本信息
    _, version = get_compiler_abi_compatibility_and_version(compiler)
    
    # 如果在 Windows 下，并且当前环境中存在 'VSCMD_ARG_TGT_ARCH'，但不存在 'DISTUTILS_USE_SDK'，则发出用户警告。
    if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' in os.environ and 'DISTUTILS_USE_SDK' not in os.environ:
        msg = ('It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.'
               'This may lead to multiple activations of the VC env.'
               'Please set `DISTUTILS_USE_SDK=1` and try again.')
        raise UserWarning(msg)
    
    # 返回编译器和版本信息的元组
    return compiler, version


# 为扩展对象添加编译标志
def _add_compile_flag(self, extension, flag):
    # 深拷贝扩展对象的额外编译参数
    extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
    
    # 如果额外编译参数是字典类型，则为每个参数列表添加 flag
    if isinstance(extension.extra_compile_args, dict):
        for args in extension.extra_compile_args.values():
            args.append(flag)
    else:
        # 否则直接向列表中添加 flag
        extension.extra_compile_args.append(flag)


# 定义 Torch 扩展名，用于 pybind11 支持的扩展名称，避免名称中包含的点号。
def _define_torch_extension_name(self, extension):
    # 将扩展名按点号分隔成多个部分，取最后一个部分作为库名
    names = extension.name.split('.')
    name = names[-1]
    
    # 定义编译标志 -DTORCH_EXTENSION_NAME=library_name 并添加到扩展对象中
    define = f'-DTORCH_EXTENSION_NAME={name}'
    self._add_compile_flag(extension, define)


# 添加 GNU C++ ABI 编译标志，保持与 PyTorch 编译时的 CXX ABI 一致。
def _add_gnu_cpp_abi_flag(self, extension):
    # 添加编译标志 -D_GLIBCXX_USE_CXX11_ABI=<value> 到扩展对象中
    self._add_compile_flag(extension, '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))
# 创建一个用于 C++ 扩展的 setuptools.Extension 对象
def CppExtension(name, sources, *args, **kwargs):
    """
    Create a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor. Full list arguments can be found at
    https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#extension-api-reference

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CppExtension
        >>> setup(
        ...     name='extension',
        ...     ext_modules=[
        ...         CppExtension(
        ...             name='extension',
        ...             sources=['extension.cpp'],
        ...             extra_compile_args=['-g'],
        ...             extra_link_flags=['-Wl,--no-as-needed', '-lm'])
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })
    """
    # 获取用户提供的 include_dirs 参数，如果没有则为空列表
    include_dirs = kwargs.get('include_dirs', [])
    # 将系统默认的 include 路径添加到 include_dirs 中
    include_dirs += include_paths()
    # 更新 kwargs 中的 include_dirs 参数为新的列表
    kwargs['include_dirs'] = include_dirs

    # 获取用户提供的 library_dirs 参数，如果没有则为空列表
    library_dirs = kwargs.get('library_dirs', [])
    # 将系统默认的 library 路径添加到 library_dirs 中
    library_dirs += library_paths()
    # 更新 kwargs 中的 library_dirs 参数为新的列表
    kwargs['library_dirs'] = library_dirs

    # 获取用户提供的 libraries 参数，如果没有则为空列表
    libraries = kwargs.get('libraries', [])
    # 添加必要的 Torch 相关库到 libraries 列表中
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    # 如果运行在 Windows 平台上，添加额外的 Sleef 库
    if IS_WINDOWS:
        libraries.append("sleef")

    # 更新 kwargs 中的 libraries 参数为新的列表
    kwargs['libraries'] = libraries

    # 设置扩展的语言为 C++
    kwargs['language'] = 'c++'
    # 使用更新后的参数创建并返回一个 setuptools.Extension 对象
    return setuptools.Extension(name, sources, *args, **kwargs)
    # 导入必要的库和模块
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    
    # 设置安装参数和编译配置
    setup(
        name='cuda_extension',  # 包的名称
        ext_modules=[
            CUDAExtension(
                name='cuda_extension',  # CUDA 扩展的名称
                sources=['extension.cpp', 'extension_kernel.cu'],  # 扩展的源文件列表
                extra_compile_args={'cxx': ['-g'],  # C++ 编译器的额外参数
                                    'nvcc': ['-O2']},  # CUDA 编译器的额外参数
                extra_link_flags=['-Wl,--no-as-needed', '-lcuda']  # 链接时的额外参数
            )
        ],
        cmdclass={
            'build_ext': BuildExtension  # 使用 BuildExtension 类来构建扩展
        }
    )
    
    
    这段代码是一个 Python 脚本，用于构建并安装一个名为 `cuda_extension` 的 CUDA 扩展模块。它通过 `setup()` 函数设置了编译和安装参数，其中：
    
    - `CUDAExtension` 是用来描述一个 CUDA 扩展的类，指定了扩展的名称、源文件、编译选项和链接选项。
    - `ext_modules` 是 `setup()` 函数的一个参数，用于指定要构建的扩展模块列表，这里只有一个 CUDAExtension 类型的模块。
    - `extra_compile_args` 包含了额外的编译参数，分别对应 C++ 编译器（`cxx`）和 CUDA 编译器（`nvcc`）的选项。
    - `extra_link_flags` 是链接时的额外参数，这里包含了 `-Wl,--no-as-needed` 和 `-lcuda`，用于告诉链接器不要忽略未解析的符号，并链接 CUDA 库。
    
    通过这些设置，可以编译和安装一个支持 CUDA 的扩展模块，用于在 PyTorch 中执行 CUDA 加速的操作。
    """
    # 从 kwargs 参数中获取 library_dirs 的值，如果不存在则使用空列表
    library_dirs = kwargs.get('library_dirs', [])
    # 调用 library_paths 函数获取 CUDA 相关的库路径，并添加到 library_dirs 列表中
    library_dirs += library_paths(cuda=True)
    # 将更新后的 library_dirs 列表重新赋值给 kwargs 参数中的 library_dirs 键
    kwargs['library_dirs'] = library_dirs

    # 从 kwargs 参数中获取 libraries 的值，如果不存在则使用空列表
    libraries = kwargs.get('libraries', [])
    # 向 libraries 列表中依次添加 'c10', 'torch', 'torch_cpu', 'torch_python' 这几个库
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    # 如果是 HIP 扩展，还需要添加 'amdhip64', 'c10_hip', 'torch_hip' 这几个库
    if IS_HIP_EXTENSION:
        libraries.append('amdhip64')
        libraries.append('c10_hip')
        libraries.append('torch_hip')
    else:
        # 如果不是 HIP 扩展，则添加 'cudart', 'c10_cuda', 'torch_cuda' 这几个库
        libraries.append('cudart')
        libraries.append('c10_cuda')
        libraries.append('torch_cuda')
    # 将更新后的 libraries 列表重新赋值给 kwargs 参数中的 libraries 键
    kwargs['libraries'] = libraries

    # 从 kwargs 参数中获取 include_dirs 的值，如果不存在则使用空列表
    include_dirs = kwargs.get('include_dirs', [])
    # 如果是 HIP 扩展，执行以下操作
    if IS_HIP_EXTENSION:
        # 获取当前工作目录作为构建目录
        build_dir = os.getcwd()
        # 使用 hipify 进行代码转换
        hipify_result = hipify_python.hipify(
            project_directory=build_dir,  # 指定项目目录
            output_directory=build_dir,   # 输出目录与项目目录相同
            header_include_dirs=include_dirs,  # 头文件包含目录列表
            includes=[os.path.join(build_dir, '*')],  # 限制转换范围为 build_dir 目录下的文件
            extra_files=[os.path.abspath(s) for s in sources],  # 需要额外转换的文件列表
            show_detailed=True,   # 显示详细转换信息
            is_pytorch_extension=True,  # 标识为 PyTorch 扩展
            hipify_extra_files_only=True,  # 仅转换额外文件，不转换 includes 路径下的所有文件
        )

        hipified_sources = set()
        # 遍历原始源文件列表
        for source in sources:
            s_abs = os.path.abspath(source)
            # 如果源文件在 hipify_result 中且已经转换成功，则使用转换后的路径，否则使用原始路径
            hipified_s_abs = (hipify_result[s_abs].hipified_path if (s_abs in hipify_result and
                              hipify_result[s_abs].hipified_path is not None) else s_abs)
            # 将转换后的源文件路径相对于构建目录进行处理，以符合 setup() 函数的要求
            hipified_sources.add(os.path.relpath(hipified_s_abs, build_dir))

        sources = list(hipified_sources)  # 更新源文件列表为转换后的路径列表

    # 添加 CUDA 相关的包含路径到 include_dirs 中
    include_dirs += include_paths(cuda=True)
    # 将 include_dirs 添加到 kwargs 的 'include_dirs' 键中
    kwargs['include_dirs'] = include_dirs

    # 设置编译语言为 'c++'
    kwargs['language'] = 'c++'

    # 获取 dlink_libraries 参数，如果不存在则设为空列表
    dlink_libraries = kwargs.get('dlink_libraries', [])
    # 判断是否需要进行 Device Link 的编译
    dlink = kwargs.get('dlink', False) or dlink_libraries
    if dlink:
        # 获取额外的编译参数
        extra_compile_args = kwargs.get('extra_compile_args', {})

        # 获取 nvcc_dlink 的编译参数，如果不存在则设为空列表
        extra_compile_args_dlink = extra_compile_args.get('nvcc_dlink', [])
        # 添加 '-dlink' 参数到 nvcc_dlink 的编译参数中
        extra_compile_args_dlink += ['-dlink']
        # 添加所有 library_dirs 到 '-L' 参数列表中
        extra_compile_args_dlink += [f'-L{x}' for x in library_dirs]
        # 添加所有 dlink_libraries 到 '-l' 参数列表中
        extra_compile_args_dlink += [f'-l{x}' for x in dlink_libraries]

        # 如果当前 CUDA 版本存在且大于等于 11.2，则添加 '-dlto' 参数
        if (torch.version.cuda is not None) and TorchVersion(torch.version.cuda) >= '11.2':
            extra_compile_args_dlink += ['-dlto']   # CUDA 11.2 开始支持设备链接时优化

        # 更新 nvcc_dlink 的编译参数
        extra_compile_args['nvcc_dlink'] = extra_compile_args_dlink

        # 更新 kwargs 中的 'extra_compile_args' 参数
        kwargs['extra_compile_args'] = extra_compile_args

    # 返回 setuptools.Extension 对象，包含名称、源文件列表以及其它参数
    return setuptools.Extension(name, sources, *args, **kwargs)
# 返回构建 C++ 或 CUDA 扩展所需的包含路径列表
def include_paths(cuda: bool = False) -> List[str]:
    # 确定 libtorch 的包含路径
    lib_include = os.path.join(_TORCH_PATH, 'include')
    # 构建初始路径列表，包含 libtorch 和 torch/torch.h 的旧版本支持路径
    paths = [
        lib_include,
        # 一旦 torch/torch.h 正式不再支持 C++ 扩展，可以移除此路径
        os.path.join(lib_include, 'torch', 'csrc', 'api', 'include'),
        # 由于一些内部（旧版）Torch 头文件未正确添加前缀，因此需要包含 -Itorch/lib/include/TH
        os.path.join(lib_include, 'TH'),
        os.path.join(lib_include, 'THC')
    ]
    # 如果需要 CUDA 并且是 HIP 扩展
    if cuda and IS_HIP_EXTENSION:
        # 添加 HIP 特定的路径
        paths.append(os.path.join(lib_include, 'THH'))
        paths.append(_join_rocm_home('include'))
    # 如果需要 CUDA
    elif cuda:
        # 确定 CUDA 的主目录的包含路径
        cuda_home_include = _join_cuda_home('include')
        # 如果 CUDA 主目录不是 '/usr/include'，则添加路径
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)
        
        # 支持由 CMake 文件设置的 CUDA_INC_PATH 环境变量
        if (cuda_inc_path := os.environ.get("CUDA_INC_PATH", None)) and \
                cuda_inc_path != '/usr/include':
            paths.append(cuda_inc_path)
        
        # 如果定义了 CUDNN_HOME，则添加 CUDNN 的包含路径
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, 'include'))
    # 返回最终的路径列表
    return paths


# 返回构建 C++ 或 CUDA 扩展所需的库路径列表
def library_paths(cuda: bool = False) -> List[str]:
    # 必须链接到 libtorch.so
    paths = [TORCH_LIB_PATH]

    # 如果需要 CUDA 并且是 HIP 扩展
    if cuda and IS_HIP_EXTENSION:
        lib_dir = 'lib'
        paths.append(_join_rocm_home(lib_dir))
        # 如果定义了 HIP_HOME，则添加 HIP 的库路径
        if HIP_HOME is not None:
            paths.append(os.path.join(HIP_HOME, 'lib'))
    # 如果需要 CUDA
    elif cuda:
        # 如果是在 Windows 下
        if IS_WINDOWS:
            lib_dir = os.path.join('lib', 'x64')
        else:
            lib_dir = 'lib64'
            # 如果 _join_cuda_home(lib_dir) 不存在且 _join_cuda_home('lib') 存在，则选择 'lib' 目录
            if (not os.path.exists(_join_cuda_home(lib_dir)) and
                    os.path.exists(_join_cuda_home('lib'))):
                lib_dir = 'lib'
        
        # 添加 CUDA 主目录的库路径
        paths.append(_join_cuda_home(lib_dir))
        
        # 如果定义了 CUDNN_HOME，则添加 CUDNN 的库路径
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, lib_dir))
    
    # 返回最终的库路径列表
    return paths
# 定义一个函数，用于即时加载（JIT）PyTorch C++ 扩展

def load(name,
         sources: Union[str, List[str]],  # 接受一个字符串或字符串列表作为源文件
         extra_cflags=None,  # 额外的 C 编译标志，默认为 None
         extra_cuda_cflags=None,  # 额外的 CUDA 编译标志，默认为 None
         extra_ldflags=None,  # 额外的链接标志，默认为 None
         extra_include_paths=None,  # 额外的包含路径，默认为 None
         build_directory=None,  # 构建目录，默认为 None
         verbose=False,  # 是否启用详细输出，默认为 False
         with_cuda: Optional[bool] = None,  # 是否使用 CUDA，可选的布尔值，默认为 None
         is_python_module=True,  # 是否是 Python 模块，默认为 True
         is_standalone=False,  # 是否是独立模式，默认为 False
         keep_intermediates=True):  # 是否保留中间文件，默认为 True

    """
    Load a PyTorch C++ extension just-in-time (JIT).

    To load an extension, a Ninja build file is emitted, which is used to
    compile the given sources into a dynamic library. This library is
    subsequently loaded into the current Python process as a module and
    returned from this function, ready for use.

    By default, the directory to which the build file is emitted and the
    resulting library compiled to is ``<tmp>/torch_extensions/<name>``, where
    ``<tmp>`` is the temporary folder on the current platform and ``<name>``
    the name of the extension. This location can be overridden in two ways.
    First, if the ``TORCH_EXTENSIONS_DIR`` environment variable is set, it
    replaces ``<tmp>/torch_extensions`` and all extensions will be compiled
    into subfolders of this directory. Second, if the ``build_directory``
    argument to this function is supplied, it overrides the entire path, i.e.
    the library will be compiled into that folder directly.

    To compile the sources, the default system compiler (``c++``) is used,
    which can be overridden by setting the ``CXX`` environment variable. To pass
    additional arguments to the compilation process, ``extra_cflags`` or
    ``extra_ldflags`` can be provided. For example, to compile your extension
    with optimizations, pass ``extra_cflags=['-O3']``. You can also use
    ``extra_cflags`` to pass further include directories.

    CUDA support with mixed compilation is provided. Simply pass CUDA source
    files (``.cu`` or ``.cuh``) along with other sources. Such files will be
    detected and compiled with nvcc rather than the C++ compiler. This includes
    passing the CUDA lib64 directory as a library directory, and linking
    ``cudart``. You can pass additional flags to nvcc via
    ``extra_cuda_cflags``, just like with ``extra_cflags`` for C++. Various
    heuristics for finding the CUDA install directory are used, which usually
    work fine. If not, setting the ``CUDA_HOME`` environment variable is the
    safest option.
    """
    # 使用 pybind11 和 Torch 提供的工具加载和编译 C++ 扩展模块
    return _jit_compile(
        name,
        [sources] if isinstance(sources, str) else sources,  # 将源文件路径转换为列表形式，如果只有一个路径则放入列表中
        extra_cflags,  # 需要传递给编译器的额外编译选项
        extra_cuda_cflags,  # 需要传递给 nvcc 编译 CUDA 源文件的额外编译选项
        extra_ldflags,  # 需要传递给链接器的额外链接选项
        extra_include_paths,  # 需要传递给编译器的额外包含目录
        build_directory or _get_build_directory(name, verbose),  # 指定编译的工作目录，如果未提供则根据名称和详细日志获取默认目录
        verbose,  # 控制是否输出详细的加载步骤日志
        with_cuda,  # 确定是否包含 CUDA 头文件和库，如果为 None，则根据源文件是否包含 `.cu` 或 `.cuh` 自动确定
        is_python_module,  # 如果为 True，则作为 Python 模块导入生成的共享库
        is_standalone,  # 如果为 False，则将构建的扩展加载到进程中作为动态库，如果为 True，则构建一个独立可执行文件
        keep_intermediates=keep_intermediates  # 控制是否保留中间生成文件（可选）
    )
# 获取 Pybind11 ABI 构建标志
def _get_pybind11_abi_build_flags():
    # 注意事项 [Pybind11 ABI constants]
    #
    # 在 Pybind11 2.4 之前，ABI 字符串的构建模式如下：
    # f"__pybind11_internals_v{PYBIND11_INTERNALS_VERSION}{PYBIND11_INTERNALS_KIND}{PYBIND11_BUILD_TYPE}__"
    # 自 2.4 版本开始，还包括编译器类型、stdlib 和构建 ABI 参数，构建模式如下：
    # f"__pybind11_internals_v{PYBIND11_INTERNALS_VERSION}{PYBIND11_INTERNALS_KIND}{PYBIND11_COMPILER_TYPE}{PYBIND11_STDLIB}{PYBIND11_BUILD_ABI}{PYBIND11_BUILD_TYPE}__"
    #
    # 这样做是为了进一步缩小编译器 ABI 不兼容性的可能性，这可能导致难以调试的段错误。
    # 对于 PyTorch 扩展，我们希望放宽这些限制，并传递在 torch/csrc/Module.cpp 中编译 PyTorch 本地库时捕获的编译器、stdlib 和 ABI 属性。

    abi_cflags = []
    for pname in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
        pval = getattr(torch._C, f"_PYBIND11_{pname}")
        if pval is not None and not IS_WINDOWS:
            abi_cflags.append(f'-DPYBIND11_{pname}=\\"{pval}\\"')
    return abi_cflags

# 获取 glibcxx ABI 构建标志
def _get_glibcxx_abi_build_flags():
    glibcxx_abi_cflags = ['-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    return glibcxx_abi_cflags

# 检查编译器是否为 GCC
def check_compiler_is_gcc(compiler):
    if not IS_LINUX:
        return False

    env = os.environ.copy()
    env['LC_ALL'] = 'C'  # 不本地化输出
    try:
        version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    except Exception as e:
        try:
            version_string = subprocess.check_output([compiler, '--version'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
        except Exception as e:
            return False
    # 检查是否为 'gcc' 或 'g++'，适用于 sccache 包装器
    pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
    results = re.findall(pattern, version_string)
    if len(results) != 1:
        return False
    compiler_path = os.path.realpath(results[0].strip())
    # 在 RHEL/CentOS 上，c++ 是 gcc 编译器包装器
    if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
        return True
    return False

# 检查并构建扩展头文件的预编译器头文件（PCH）
def _check_and_build_extension_h_precompiler_headers(
        extra_cflags,
        extra_include_paths,
        is_standalone=False):
    r'''
    预编译头文件（PCH）可以预先构建相同的头文件，并减少 PyTorch load_inline 模块的构建时间。
    GCC 官方手册：https://gcc.gnu.org/onlinedocs/gcc-4.0.4/gcc/Precompiled-Headers.html
    PCH 仅在构建了 pch 文件（header.h.gch）且构建目标具有相同的构建参数时才有效。因此，我们需要
    添加一个签名文件来记录 PCH 文件的参数。如果构建参数（签名）发生变化，则应重新构建 PCH 文件。

    注意：
    1. Windows 和 MacOS 有不同的 PCH 机制。我们目前仅支持 Linux。
    '''
    # 如果不是在 Linux 系统上运行，则退出函数
    if not IS_LINUX:
        return

    # 获取当前的 C++ 编译器
    compiler = get_cxx_compiler()

    # 检查当前编译器是否为 GCC
    b_is_gcc = check_compiler_is_gcc(compiler)
    # 如果不是 GCC，则退出函数
    if b_is_gcc is False:
        return

    # 设置头文件的路径
    head_file = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h')
    # 预编译头文件的路径
    head_file_pch = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.gch')
    # 头文件签名文件的路径
    head_file_signature = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.sign')

    # 将列表转换为字符串的函数
    def listToString(s):
        # 初始化空字符串
        string = ""
        # 如果输入为空，则直接返回空字符串
        if s is None:
            return string

        # 遍历列表中的元素，拼接为一个字符串
        for element in s:
            string += (element + ' ')
        # 返回拼接好的字符串
        return string

    # 格式化预编译头文件的命令
    def format_precompiler_header_cmd(compiler, head_file, head_file_pch, common_cflags, torch_include_dirs, extra_cflags, extra_include_paths):
        return re.sub(
            r"[ \n]+",
            " ",
            f"""
                {compiler} -x c++-header {head_file} -o {head_file_pch} {torch_include_dirs} {extra_include_paths} {extra_cflags} {common_cflags}
            """,
        ).strip()

    # 将命令转换为签名字符串的函数
    def command_to_signature(cmd):
        # 将命令中的空格替换为下划线，用作文件签名
        signature = cmd.replace(' ', '_')
        return signature

    # 检查文件中的预编译头文件签名是否匹配
    def check_pch_signature_in_file(file_path, signature):
        # 检查文件是否存在
        b_exist = os.path.isfile(file_path)
        # 如果文件不存在，则返回 False
        if b_exist is False:
            return False

        # 打开文件并读取其内容
        with open(file_path) as file:
            content = file.read()
            # 检查文件内容是否与预期的签名字符串相符
            if signature == content:
                return True
            else:
                return False

    # 如果目录不存在，则创建目录
    def _create_if_not_exist(path_dir):
        if not os.path.exists(path_dir):
            try:
                # 创建目录及其所有父目录
                Path(path_dir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                # 如果创建失败，则抛出异常
                if exc.errno != errno.EEXIST:
                    raise RuntimeError(f"Fail to create path {path_dir}") from exc

    # 将预编译头文件的签名写入文件
    def write_pch_signature_to_file(file_path, pch_sign):
        # 如果文件所在目录不存在，则创建目录
        _create_if_not_exist(os.path.dirname(file_path))
        # 将签名写入文件中
        with open(file_path, "w") as f:
            f.write(pch_sign)
            f.close()

    # 编译预编译头文件
    def build_precompile_header(pch_cmd):
        try:
            # 执行预编译头文件的编译命令
            subprocess.check_output(pch_cmd, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            # 如果编译失败，则抛出异常
            raise RuntimeError(f"Compile PreCompile Header fail, command: {pch_cmd}") from e

    # 将额外编译标志列表转换为字符串
    extra_cflags_str = listToString(extra_cflags)
    # 将额外的包含路径列表转换为参数形式的字符串
    extra_include_paths_str = " ".join(
        [f"-I{include}" for include in extra_include_paths] if extra_include_paths else []
    )

    # 设置库文件的包含路径
    lib_include = os.path.join(_TORCH_PATH, 'include')
    ```
    # 定义包含 Torch 库的头文件目录列表
    torch_include_dirs = [
        f"-I {lib_include}",  # 添加指向库包含目录的编译器选项
        # 包含 Python.h 头文件路径
        "-I {}".format(sysconfig.get_path("include")),
        # 包含 torch/all.h 头文件路径
        "-I {}".format(os.path.join(lib_include, 'torch', 'csrc', 'api', 'include')),
    ]
    
    # 将包含 Torch 库的头文件目录列表转换为字符串
    torch_include_dirs_str = listToString(torch_include_dirs)
    
    # 定义通用的编译器标志列表
    common_cflags = []
    if not is_standalone:
        # 如果不是独立模式，添加 TORCH_API_INCLUDE_EXTENSION_H 宏定义
        common_cflags += ['-DTORCH_API_INCLUDE_EXTENSION_H']
    
    # 添加通用的 C++17 和位置无关代码（PIC）标志
    common_cflags += ['-std=c++17', '-fPIC']
    # 添加从 _get_pybind11_abi_build_flags() 函数获取的编译器标志
    common_cflags += [f"{x}" for x in _get_pybind11_abi_build_flags()]
    # 添加从 _get_glibcxx_abi_build_flags() 函数获取的编译器标志
    common_cflags += [f"{x}" for x in _get_glibcxx_abi_build_flags()]
    # 将通用的编译器标志列表转换为字符串
    common_cflags_str = listToString(common_cflags)
    
    # 格式化预编译头文件的命令
    pch_cmd = format_precompiler_header_cmd(
        compiler, head_file, head_file_pch, common_cflags_str, torch_include_dirs_str, extra_cflags_str, extra_include_paths_str)
    # 生成预编译头文件的签名
    pch_sign = command_to_signature(pch_cmd)
    
    # 如果预编译头文件不存在
    if os.path.isfile(head_file_pch) is not True:
        # 构建预编译头文件
        build_precompile_header(pch_cmd)
        # 将预编译头文件的签名写入文件
        write_pch_signature_to_file(head_file_signature, pch_sign)
    else:
        # 否则，检查当前的预编译头文件签名是否与记录的签名相同
        b_same_sign = check_pch_signature_in_file(head_file_signature, pch_sign)
        # 如果签名不相同
        if b_same_sign is False:
            # 重新构建预编译头文件
            build_precompile_header(pch_cmd)
            # 更新预编译头文件的签名
            write_pch_signature_to_file(head_file_signature, pch_sign)
def remove_extension_h_precompiler_headers():
    # 定义内部函数，用于删除存在的文件
    def _remove_if_file_exists(path_file):
        # 检查文件是否存在，如果存在则删除
        if os.path.exists(path_file):
            os.remove(path_file)

    # 拼接得到预编译头文件（.gch 文件）的路径
    head_file_pch = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.gch')
    # 拼接得到签名文件（.sign 文件）的路径
    head_file_signature = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.sign')

    # 调用内部函数，删除预编译头文件（如果存在）
    _remove_if_file_exists(head_file_pch)
    # 调用内部函数，删除签名文件（如果存在）
    _remove_if_file_exists(head_file_signature)

def load_inline(name,
                cpp_sources,
                cuda_sources=None,
                functions=None,
                extra_cflags=None,
                extra_cuda_cflags=None,
                extra_ldflags=None,
                extra_include_paths=None,
                build_directory=None,
                verbose=False,
                with_cuda=None,
                is_python_module=True,
                with_pytorch_error_handling=True,
                keep_intermediates=True,
                use_pch=False):
    r'''
    从字符串源加载 PyTorch C++ 扩展（即时编译）。

    此函数与 :func:`load` 的行为完全相同，但其源代码为字符串而非文件名。
    这些字符串在构建目录中存储为文件，之后 :func:`load_inline` 的行为与 :func:`load` 相同。

    参见 `测试用例 <https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions_jit.py>`_
    获取此函数的良好示例。

    源代码可以省略典型非内联 C++ 扩展的两个必需部分：
    必要的头文件包含以及（pybind11）绑定代码。具体而言，传递给 ``cpp_sources`` 的字符串首先连接为单个 ``.cpp`` 文件。
    然后，此文件以 ``#include <torch/extension.h>`` 开头。

    此外，如果提供了 ``functions`` 参数，则将为每个指定的函数自动生成绑定。
    ``functions`` 可以是函数名列表，或者是从函数名到文档字符串的字典。如果给定列表，则每个函数的名称将用作其文档字符串。

    ``cuda_sources`` 中的源代码连接为单独的 ``.cu`` 文件，并以 ``torch/types.h``、``cuda.h`` 和 ``cuda_runtime.h`` 开头。
    ``.cpp`` 和 ``.cu`` 文件分别编译，最终链接为单个库。请注意，``cuda_sources`` 中的函数不会自动绑定。
    要绑定到 CUDA 核心，必须创建一个调用它的 C++ 函数，并在 ``cpp_sources`` 中声明或定义此 C++ 函数（并在 ``functions`` 中包含其名称）。

    有关省略的参数的描述，请参见 :func:`load`。

    ```py
    '''
    Args:
        cpp_sources: A string, or list of strings, containing C++ source code.
        cuda_sources: A string, or list of strings, containing CUDA source code.
        functions: A list of function names for which to generate function
            bindings. If a dictionary is given, it should map function names to
            docstrings (which are otherwise just the function names).
        with_cuda: Determines whether CUDA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on whether ``cuda_sources`` is
            provided. Set it to ``True`` to force CUDA headers
            and libraries to be included.
        with_pytorch_error_handling: Determines whether pytorch error and
            warning macros are handled by pytorch instead of pybind. To do
            this, each function ``foo`` is called via an intermediary ``_safe_foo``
            function. This redirection might cause issues in obscure cases
            of cpp. This flag should be set to ``False`` when this redirect
            causes issues.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from torch.utils.cpp_extension import load_inline
        >>> source = """
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
          return x.sin() + y.sin();
        }
        """
        >>> module = load_inline(name='inline_extension',
        ...                      cpp_sources=[source],
        ...                      functions=['sin_add'])

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    '''

    # 如果未提供构建目录，根据名称获取构建目录并可选地显示详细信息
    build_directory = build_directory or _get_build_directory(name, verbose)

    # 如果 cpp_sources 是字符串，则转换为单元素列表
    if isinstance(cpp_sources, str):
        cpp_sources = [cpp_sources]

    # 如果未提供 cuda_sources，则置为空列表
    cuda_sources = cuda_sources or []

    # 如果 cuda_sources 是字符串，则转换为单元素列表
    if isinstance(cuda_sources, str):
        cuda_sources = [cuda_sources]

    # 在 cpp_sources 的开头插入 torch/extension.h 的 include 语句
    cpp_sources.insert(0, '#include <torch/extension.h>')

    # 如果 use_pch 是 True，则使用预编译头文件 'torch/extension.h' 来减少编译时间
    if use_pch is True:
        # 调用函数，检查并构建扩展头文件的预编译器头文件
        _check_and_build_extension_h_precompiler_headers(extra_cflags, extra_include_paths)
    else:
        # 移除扩展头文件的预编译器头文件
        remove_extension_h_precompiler_headers()

    # 如果提供了 functions 参数，则为用户生成 pybind11 的绑定
    # 在这里，functions 是一个从函数名映射到函数文档字符串（或者仅函数名）的映射表
    # 如果 functions 参数不是 None，则进行以下处理
    if functions is not None:
        # 初始化一个空列表，用于存储生成的 C++ 模块定义
        module_def = []
        # 将 PYBIND11_MODULE 宏的定义添加到模块定义列表中
        module_def.append('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {')
        
        # 如果 functions 是字符串，则转换为单元素列表以便统一处理
        if isinstance(functions, str):
            functions = [functions]
        
        # 如果 functions 是列表，则将其转换为字典，其中键和值相同，表示函数名和文档字符串相同
        if isinstance(functions, list):
            functions = {f: f for f in functions}
        # 如果 functions 不是列表或字典，则引发异常
        elif not isinstance(functions, dict):
            raise ValueError(f"Expected 'functions' to be a list or dict, but was {type(functions)}")
        
        # 遍历 functions 字典，生成每个函数的模块定义条目
        for function_name, docstring in functions.items():
            # 如果指定了 with_pytorch_error_handling，则使用 wrap_pybind_function 封装函数
            if with_pytorch_error_handling:
                module_def.append(f'm.def("{function_name}", torch::wrap_pybind_function({function_name}), "{docstring}");')
            # 否则直接使用函数名进行定义
            else:
                module_def.append(f'm.def("{function_name}", {function_name}, "{docstring}");')
        
        # 添加 C++ 模块定义的结束标记
        module_def.append('}')
        
        # 将生成的 C++ 模块定义列表添加到 cpp_sources 中
        cpp_sources += module_def

    # 构建生成的 main.cpp 文件的路径
    cpp_source_path = os.path.join(build_directory, 'main.cpp')
    # 将生成的 C++ 源码写入到 main.cpp 文件中
    _maybe_write(cpp_source_path, "\n".join(cpp_sources))

    # 将 main.cpp 文件路径添加到 sources 列表中
    sources = [cpp_source_path]

    # 如果存在 cuda_sources，则需要进行额外处理
    if cuda_sources:
        # 在 cuda_sources 列表的开头插入必要的头文件包含
        cuda_sources.insert(0, '#include <torch/types.h>')
        cuda_sources.insert(1, '#include <cuda.h>')
        cuda_sources.insert(2, '#include <cuda_runtime.h>')

        # 构建生成的 cuda.cu 文件的路径
        cuda_source_path = os.path.join(build_directory, 'cuda.cu')
        # 将生成的 CUDA 源码写入到 cuda.cu 文件中
        _maybe_write(cuda_source_path, "\n".join(cuda_sources))

        # 将 cuda.cu 文件路径添加到 sources 列表中
        sources.append(cuda_source_path)

    # 调用 _jit_compile 函数进行 JIT 编译，返回编译结果
    return _jit_compile(
        name,
        sources,
        extra_cflags,
        extra_cuda_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory,
        verbose,
        with_cuda,
        is_python_module,
        is_standalone=False,
        keep_intermediates=keep_intermediates)
# 编译即时（Just-In-Time）扩展模块的函数
def _jit_compile(name,
                 sources,
                 extra_cflags,
                 extra_cuda_cflags,
                 extra_ldflags,
                 extra_include_paths,
                 build_directory: str,
                 verbose: bool,
                 with_cuda: Optional[bool],
                 is_python_module,
                 is_standalone,
                 keep_intermediates=True) -> None:
    # 检查 `is_python_module` 和 `is_standalone` 是否同时为真，它们是互斥的
    if is_python_module and is_standalone:
        raise ValueError("`is_python_module` and `is_standalone` are mutually exclusive.")

    # 如果未明确指定 `with_cuda`，根据源文件列表检测是否需要使用 CUDA
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    # 检查 `extra_ldflags` 是否包含 cudnn，确定是否需要使用 cudnn
    with_cudnn = any('cudnn' in f for f in extra_ldflags or [])

    # 获取当前扩展模块的旧版本号
    old_version = JIT_EXTENSION_VERSIONER.get_version(name)
    # 根据输入条件和构建参数计算新版本号，可能会更新版本号
    version = JIT_EXTENSION_VERSIONER.bump_version_if_changed(
        name,
        sources,
        build_arguments=[extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths],
        build_directory=build_directory,
        with_cuda=with_cuda,
        is_python_module=is_python_module,
        is_standalone=is_standalone,
    )

    # 如果版本号大于零，且与旧版本号不同，并且需要输出详细信息，则打印版本更新信息
    if version > 0:
        if version != old_version and verbose:
            print(f'The input conditions for extension module {name} have changed. ' +
                  f'Bumping to version {version} and re-building as {name}_v{version}...',
                  file=sys.stderr)
        # 更新模块名称为包含版本号的形式
        name = f'{name}_v{version}'

    # 创建一个文件锁对象，用于确保编译过程中不会并行运行多个编译任务
    baton = FileBaton(os.path.join(build_directory, 'lock'))
    # 如果能获取到控制权（信号量 baton），则执行以下操作
    if baton.try_acquire():
        try:
            # 如果版本号与旧版本号不同
            if version != old_version:
                # 使用 GeneratedFileCleaner 管理临时文件的清理
                with GeneratedFileCleaner(keep_intermediates=keep_intermediates) as clean_ctx:
                    # 如果是 HIP 扩展并且使用 CUDA 或者 cuDNN
                    if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                        # 进行 HIP 转换
                        hipify_result = hipify_python.hipify(
                            project_directory=build_directory,
                            output_directory=build_directory,
                            header_include_dirs=(extra_include_paths if extra_include_paths is not None else []),
                            extra_files=[os.path.abspath(s) for s in sources],
                            ignores=[_join_rocm_home('*'), os.path.join(_TORCH_PATH, '*')],  # 不需要转换 ROCm 或 PyTorch 的头文件
                            show_detailed=verbose,
                            show_progress=verbose,
                            is_pytorch_extension=True,
                            clean_ctx=clean_ctx
                        )

                        # 收集已经进行了 HIP 转换的源文件路径
                        hipified_sources = set()
                        for source in sources:
                            s_abs = os.path.abspath(source)
                            # 如果该源文件在 hipify_result 中有对应的转换路径，则使用转换后的路径，否则使用原始路径
                            hipified_sources.add(hipify_result[s_abs].hipified_path if s_abs in hipify_result else s_abs)

                        # 更新 sources 列表为转换后的路径列表
                        sources = list(hipified_sources)

                    # 写入 Ninja 文件并构建库文件
                    _write_ninja_file_and_build_library(
                        name=name,
                        sources=sources,
                        extra_cflags=extra_cflags or [],
                        extra_cuda_cflags=extra_cuda_cflags or [],
                        extra_ldflags=extra_ldflags or [],
                        extra_include_paths=extra_include_paths or [],
                        build_directory=build_directory,
                        verbose=verbose,
                        with_cuda=with_cuda,
                        is_standalone=is_standalone)
            # 如果版本号与旧版本号相同，并且 verbose 为 True，则输出无需重新构建的消息
            elif verbose:
                print('No modifications detected for re-loaded extension '
                      f'module {name}, skipping build step...', file=sys.stderr)
        finally:
            # 释放信号量 baton
            baton.release()
    else:
        # 等待信号量 baton 的释放
        baton.wait()

    # 如果 verbose 为 True，则输出加载扩展模块的消息
    if verbose:
        print(f'Loading extension module {name}...', file=sys.stderr)

    # 如果是独立模块，则返回该模块的执行路径
    if is_standalone:
        return _get_exec_path(name, build_directory)

    # 否则，从库文件中导入该模块并返回
    return _import_module_from_library(name, build_directory, is_python_module)
# 定义一个函数，用于生成 Ninja 构建文件并编译对象文件
def _write_ninja_file_and_compile_objects(
        sources: List[str],
        objects,
        cflags,
        post_cflags,
        cuda_cflags,
        cuda_post_cflags,
        cuda_dlink_post_cflags,
        build_directory: str,
        verbose: bool,
        with_cuda: Optional[bool]) -> None:
    # 确认 Ninja 可用性
    verify_ninja_availability()

    # 获取 C++ 编译器
    compiler = get_cxx_compiler()

    # 获取编译器 ABI 兼容性和版本信息
    get_compiler_abi_compatibility_and_version(compiler)

    # 如果 with_cuda 未指定，则检查源文件中是否包含 CUDA 文件
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))

    # 构建 Ninja 文件的路径
    build_file_path = os.path.join(build_directory, 'build.ninja')

    # 如果 verbose 为 True，则输出正在生成 Ninja 构建文件的消息
    if verbose:
        print(f'Emitting ninja build file {build_file_path}...', file=sys.stderr)

    # 生成 Ninja 构建文件
    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        cuda_cflags=cuda_cflags,
        cuda_post_cflags=cuda_post_cflags,
        cuda_dlink_post_cflags=cuda_dlink_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_cuda=with_cuda)

    # 如果 verbose 为 True，则输出正在编译对象文件的消息
    if verbose:
        print('Compiling objects...', file=sys.stderr)

    # 运行 Ninja 构建过程来编译对象文件
    _run_ninja_build(
        build_directory,
        verbose,
        # 如果能告知用户构建失败的扩展名会更好，但这里无法获取
        error_prefix='Error compiling objects for extension')


# 定义一个函数，用于生成 Ninja 构建文件并构建库文件
def _write_ninja_file_and_build_library(
        name,
        sources: List[str],
        extra_cflags,
        extra_cuda_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory: str,
        verbose: bool,
        with_cuda: Optional[bool],
        is_standalone: bool = False) -> None:
    # 确认 Ninja 可用性
    verify_ninja_availability()

    # 获取 C++ 编译器
    compiler = get_cxx_compiler()

    # 获取编译器 ABI 兼容性和版本信息
    get_compiler_abi_compatibility_and_version(compiler)

    # 如果 with_cuda 未指定，则检查源文件中是否包含 CUDA 文件
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))

    # 准备额外的链接标志（ldflags）
    extra_ldflags = _prepare_ldflags(
        extra_ldflags or [],
        with_cuda,
        verbose,
        is_standalone)

    # 构建 Ninja 文件的路径
    build_file_path = os.path.join(build_directory, 'build.ninja')

    # 如果 verbose 为 True，则输出正在生成 Ninja 构建文件的消息
    if verbose:
        print(f'Emitting ninja build file {build_file_path}...', file=sys.stderr)

    # NOTE: 生成一个新的 Ninja 构建文件不会导致源文件未更改时重新编译，因此重新生成是安全且快速的
    # 生成 Ninja 构建库文件的指令
    _write_ninja_file_to_build_library(
        path=build_file_path,
        name=name,
        sources=sources,
        extra_cflags=extra_cflags or [],
        extra_cuda_cflags=extra_cuda_cflags or [],
        extra_ldflags=extra_ldflags or [],
        extra_include_paths=extra_include_paths or [],
        with_cuda=with_cuda,
        is_standalone=is_standalone)

    # 如果 verbose 为 True，则输出正在构建扩展模块的消息
    if verbose:
        print(f'Building extension module {name}...', file=sys.stderr)

    # 运行 Ninja 构建过程来构建库文件
    _run_ninja_build(
        build_directory,
        verbose,
        error_prefix=f"Error building extension '{name}'")


# 定义一个函数，用于检查 Ninja 的可用性
def is_ninja_available():
    # 检查系统上是否安装了 `ninja` 构建系统，如果安装了则返回 True，否则返回 False
    try:
        # 尝试运行命令 `ninja --version` 并获取输出，如果成功则说明 `ninja` 可用
        subprocess.check_output('ninja --version'.split())
    except Exception:
        # 如果出现任何异常，说明 `ninja` 不可用，返回 False
        return False
    else:
        # 如果没有异常，则说明命令成功执行，`ninja` 可用，返回 True
        return True
def verify_ninja_availability():
    """检查系统上是否可用 `ninja <https://ninja-build.org/>`_ 构建系统，如果不可用则抛出 ``RuntimeError`` 异常，否则什么也不做。"""
    # 检查是否有可用的 ninja 构建系统
    if not is_ninja_available():
        # 如果 ninja 不可用，则抛出异常
        raise RuntimeError("Ninja is required to load C++ extensions")


def _prepare_ldflags(extra_ldflags, with_cuda, verbose, is_standalone):
    """根据操作系统类型和其他条件准备链接标志列表。

    Args:
        extra_ldflags (list): 链接标志列表，用于存储额外的链接选项。
        with_cuda (bool): 是否包含 CUDA 支持。
        verbose (bool): 是否输出详细信息。
        is_standalone (bool): 是否为独立模式，影响链接方式。

    Returns:
        list: 更新后的链接标志列表。
    """
    if IS_WINDOWS:
        # 设置 Python 库路径
        python_lib_path = os.path.join(sys.base_exec_prefix, 'libs')

        # 添加常规库文件
        extra_ldflags.append('c10.lib')
        if with_cuda:
            extra_ldflags.append('c10_cuda.lib')
        extra_ldflags.append('torch_cpu.lib')
        if with_cuda:
            extra_ldflags.append('torch_cuda.lib')
            # 添加特定于 Windows 的链接选项，确保在依赖于 torch_cuda 的项目中正确链接
            # 相关问题：https://github.com/pytorch/pytorch/issues/31611
            extra_ldflags.append('-INCLUDE:?warp_size@cuda@at@@YAHXZ')
        extra_ldflags.append('torch.lib')
        # 添加 Torch 库路径
        extra_ldflags.append(f'/LIBPATH:{TORCH_LIB_PATH}')
        if not is_standalone:
            extra_ldflags.append('torch_python.lib')
            extra_ldflags.append(f'/LIBPATH:{python_lib_path}')

    else:
        # 非 Windows 系统下的链接选项设置
        extra_ldflags.append(f'-L{TORCH_LIB_PATH}')
        extra_ldflags.append('-lc10')
        if with_cuda:
            extra_ldflags.append('-lc10_hip' if IS_HIP_EXTENSION else '-lc10_cuda')
        extra_ldflags.append('-ltorch_cpu')
        if with_cuda:
            extra_ldflags.append('-ltorch_hip' if IS_HIP_EXTENSION else '-ltorch_cuda')
        extra_ldflags.append('-ltorch')
        if not is_standalone:
            extra_ldflags.append('-ltorch_python')

        if is_standalone:
            # 在独立模式下添加额外的动态链接库路径
            extra_ldflags.append(f"-Wl,-rpath,{TORCH_LIB_PATH}")

    # 处理 CUDA 相关选项
    if with_cuda:
        if verbose:
            # 如果需要输出详细信息，打印检测到 CUDA 文件的信息
            print('Detected CUDA files, patching ldflags', file=sys.stderr)
        if IS_WINDOWS:
            # Windows 下的 CUDA 相关库路径设置
            extra_ldflags.append(f'/LIBPATH:{_join_cuda_home("lib", "x64")}')
            extra_ldflags.append('cudart.lib')
            if CUDNN_HOME is not None:
                extra_ldflags.append(f'/LIBPATH:{os.path.join(CUDNN_HOME, "lib", "x64")}')
        elif not IS_HIP_EXTENSION:
            # 非 Windows 下的 CUDA 相关库路径设置
            extra_lib_dir = "lib64"
            if (not os.path.exists(_join_cuda_home(extra_lib_dir)) and
                    os.path.exists(_join_cuda_home("lib"))):
                # 可能存在于 "lib" 目录下的 64 位 CUDA
                # 注意：也有可能两者都不存在（参见 _find_cuda_home），此时保持 "lib64"
                extra_lib_dir = "lib"
            extra_ldflags.append(f'-L{_join_cuda_home(extra_lib_dir)}')
            extra_ldflags.append('-lcudart')
            if CUDNN_HOME is not None:
                extra_ldflags.append(f'-L{os.path.join(CUDNN_HOME, "lib64")}')
        elif IS_HIP_EXTENSION:
            # ROCm（HIP）扩展环境下的链接设置
            extra_ldflags.append(f'-L{_join_rocm_home("lib")}')
            extra_ldflags.append('-lamdhip64')
    return extra_ldflags
def _get_cuda_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    """
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    """
    # 如果传入了 cflags 参数，可能已经包含了用户提供的架构标志（来自 `extra_compile_args`）
    if cflags is not None:
        for flag in cflags:
            # 如果 flag 包含 'TORCH_EXTENSION_NAME'，跳过处理
            if 'TORCH_EXTENSION_NAME' in flag:
                continue
            # 如果 flag 包含 'arch'，说明已经设置了架构标志，直接返回空列表
            if 'arch' in flag:
                return []

    # 注意：在单个架构名称之前保持组合名称（"arch1+arch2"），否则字符串替换可能不起作用
    named_arches = collections.OrderedDict([
        ('Kepler+Tesla', '3.7'),
        ('Kepler', '3.5+PTX'),
        ('Maxwell+Tegra', '5.3'),
        ('Maxwell', '5.0;5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta+Tegra', '7.2'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
        ('Ampere+Tegra', '8.7'),
        ('Ampere', '8.0;8.6+PTX'),
        ('Ada', '8.9+PTX'),
        ('Hopper', '9.0+PTX'),
    ])

    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5', '8.0', '8.6', '8.7', '8.9', '9.0', '9.0a']
    valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]

    # 默认情况下，对于 CUDA 9.x 和 10.x，使用 sm_30
    # 首先检查环境变量（与主 setup.py 使用的相同）
    # 可能是一个或多个架构，例如 "6.1" 或 "3.5;5.2;6.0;6.1;7.0+PTX"
    # 参见 cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    _arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)

    # 如果未提供 _arch_list，确定对于找到的 GPU / CUDA 版本最佳的架构
    if not _arch_list:
        # 如果没有设置 _arch_list，则发出警告，说明将为所有可见的 GPU 编译所有架构。
        # 如果不希望这样，请设置 os.environ['TORCH_CUDA_ARCH_LIST']。
        warnings.warn(
            "TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. \n"
            "If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].")
        arch_list = []
        # 假设是扩展应在当前可见的任何类型的 GPU 上运行，
        # 因此应包含所有可见卡的所有架构。
        for i in range(torch.cuda.device_count()):
            # 获取第 i 个 GPU 设备的计算能力
            capability = torch.cuda.get_device_capability(i)
            # 获取支持的所有架构中的 sm 版本号列表
            supported_sm = [int(arch.split('_')[1])
                            for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
            # 获取支持的最大 sm 版本
            max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
            # 设备的实际能力可能高于用户 NVCC 支持的能力，
            # 这可能导致编译错误。期望用户的 NVCC 与构建 PyTorch 所用的 NVCC 匹配，
            # 因此使用 PyTorch 支持的最大能力来限制能力。
            capability = min(max_supported_sm, capability)
            # 格式化成 "X.Y" 形式的架构版本号
            arch = f'{capability[0]}.{capability[1]}'
            if arch not in arch_list:
                # 将生成的架构版本号添加到 arch_list 中
                arch_list.append(arch)
        # 对架构版本号列表进行排序
        arch_list = sorted(arch_list)
        # 将最后一个架构版本号添加 "+PTX" 后缀
        arch_list[-1] += '+PTX'
    else:
        # 处理以空格分隔的 _arch_list（仅在分号之后处理）
        _arch_list = _arch_list.replace(' ', ';')
        # 根据 named_arches 字典扩展命名的架构
        for named_arch, archval in named_arches.items():
            _arch_list = _arch_list.replace(named_arch, archval)

        # 将 _arch_list 按分号分割成 arch_list
        arch_list = _arch_list.split(';')

    flags = []
    # 遍历 arch_list 中的每个架构版本号
    for arch in arch_list:
        if arch not in valid_arch_strings:
            # 如果架构版本号不在有效的架构字符串列表中，则抛出 ValueError 异常
            raise ValueError(f"Unknown CUDA arch ({arch}) or GPU not supported")
        else:
            # 从架构版本号中提取数字部分，用于生成编译标志
            num = arch[0] + arch[2:].split("+")[0]
            # 添加编译标志，格式为 '-gencode=arch=compute_X.Y,code=sm_X.Y'
            flags.append(f'-gencode=arch=compute_{num},code=sm_{num}')
            if arch.endswith('+PTX'):
                # 如果架构版本号以 '+PTX' 结尾，则添加额外的 PTX 编译标志
                flags.append(f'-gencode=arch=compute_{num},code=compute_{num}')

    # 返回排序后的去重编译标志集合
    return sorted(set(flags))
# 根据给定的编译标志列表（如果提供），获取 ROCm 架构相关的编译标志
def _get_rocm_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    # 如果 cflags 已提供，则可能包含用户提供的架构标志（来自 `extra_compile_args`）
    if cflags is not None:
        for flag in cflags:
            # 检查是否包含 ROCm 或 offload 架构相关的标志
            if 'amdgpu-target' in flag or 'offload-arch' in flag:
                return ['-fno-gpu-rdc']  # 返回禁用 GPU RDC 的标志

    # 使用与构建 PyTorch 时相同的默认设置
    # 允许环境变量覆盖，就像在初始 cmake 构建期间一样。
    _archs = os.environ.get('PYTORCH_ROCM_ARCH', None)
    if not _archs:
        # 如果未设置环境变量 PYTORCH_ROCM_ARCH，则获取 Torch 提供的架构标志
        archFlags = torch._C._cuda_getArchFlags()
        if archFlags:
            archs = archFlags.split()
        else:
            archs = []
    else:
        # 如果设置了环境变量 PYTORCH_ROCM_ARCH，则从中获取架构列表
        archs = _archs.replace(' ', ';').split(';')

    # 构建 offload 架构相关的编译标志列表
    flags = [f'--offload-arch={arch}' for arch in archs]
    flags += ['-fno-gpu-rdc']  # 添加禁用 GPU RDC 的标志
    return flags

# 获取构建目录的路径
def _get_build_directory(name: str, verbose: bool) -> str:
    # 获取根扩展目录的路径
    root_extensions_directory = os.environ.get('TORCH_EXTENSIONS_DIR')
    if root_extensions_directory is None:
        # 如果未设置 TORCH_EXTENSIONS_DIR，则使用默认的构建根目录
        root_extensions_directory = get_default_build_root()
        # 根据当前环境设置 CPU 或 GPU 的构建文件夹名称
        cu_str = ('cpu' if torch.version.cuda is None else
                  f'cu{torch.version.cuda.replace(".", "")}')  # type: ignore[attr-defined]
        python_version = f'py{sys.version_info.major}{sys.version_info.minor}'
        build_folder = f'{python_version}_{cu_str}'

        # 构建完整的扩展目录路径
        root_extensions_directory = os.path.join(
            root_extensions_directory, build_folder)

    # 如果 verbose 为 True，则打印使用的 PyTorch 扩展根目录路径信息
    if verbose:
        print(f'Using {root_extensions_directory} as PyTorch extensions root...', file=sys.stderr)

    # 构建特定扩展名称的构建目录路径
    build_directory = os.path.join(root_extensions_directory, name)
    if not os.path.exists(build_directory):
        # 如果构建目录不存在，则根据需要创建
        if verbose:
            print(f'Creating extension directory {build_directory}...', file=sys.stderr)
        # 类似于 mkdir -p 的操作，将同时创建父目录
        os.makedirs(build_directory, exist_ok=True)

    return build_directory

# 获取可用的工作线程数目
def _get_num_workers(verbose: bool) -> Optional[int]:
    # 获取环境变量 MAX_JOBS 的值作为最大工作线程数
    max_jobs = os.environ.get('MAX_JOBS')
    if max_jobs is not None and max_jobs.isdigit():
        # 如果 MAX_JOBS 已定义且是数字，则将其作为工作线程数返回
        if verbose:
            print(f'Using envvar MAX_JOBS ({max_jobs}) as the number of workers...',
                  file=sys.stderr)
        return int(max_jobs)

    # 如果 verbose 为 True，则说明允许 ninja 设置默认的工作线程数
    if verbose:
        print('Allowing ninja to set a default number of workers... '
              '(overridable by setting the environment variable MAX_JOBS=N)',
              file=sys.stderr)

    return None  # 返回 None 表示未指定特定的工作线程数

# 执行使用 ninja 进行构建的操作
def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    # 定义执行 ninja 命令的基本参数列表
    command = ['ninja', '-v']
    # 获取可用的工作线程数
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        # 如果指定了工作线程数，则将其添加到命令参数中
        command.extend(['-j', str(num_workers)])
    # 复制当前环境变量作为执行命令的环境
    env = os.environ.copy()
    # 尝试激活用户的 vc 环境
    # 如果运行环境是 Windows，并且环境变量中没有 'VSCMD_ARG_TGT_ARCH'
    if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' not in env:
        # 导入 setuptools 的 distutils 模块
        from setuptools import distutils

        # 获取当前平台的名称
        plat_name = distutils.util.get_platform()
        # 根据平台名称从预定义的映射表 PLAT_TO_VCVARS 中获取对应的平台规范
        plat_spec = PLAT_TO_VCVARS[plat_name]

        # 调用 distutils 内部方法 _get_vc_env 获取与当前平台相关的 VC 编译环境变量
        vc_env = distutils._msvccompiler._get_vc_env(plat_spec)
        # 将获取到的环境变量键名转换为大写
        vc_env = {k.upper(): v for k, v in vc_env.items()}
        
        # 遍历原始环境变量，将其中未包含在 vc_env 中的项添加进 vc_env
        for k, v in env.items():
            uk = k.upper()
            if uk not in vc_env:
                vc_env[uk] = v
        
        # 将 env 更新为 vc_env
        env = vc_env

    try:
        # 刷新标准输出和标准错误流
        sys.stdout.flush()
        sys.stderr.flush()

        # 提示：不要将 stdout=None 传递给 subprocess.run 以获取输出。
        # subprocess.run 假定 sys.__stdout__ 未被修改，并默认尝试向其写入。
        # 然而，当我们从预编译的 cpp 扩展中调用 _run_ninja_build 时，会出现以下情况：
        # 1）如果 stdout 编码不是 utf-8，则 setuptools 会分离 __stdout__。
        #    https://github.com/pypa/setuptools/blob/7e97def47723303fafabe48b22168bbc11bb4821/setuptools/dist.py#L1110
        #    （这可能不应该这样做）
        # 2）subprocess.run（在 POSIX 系统上，没有 stdout 覆盖的情况下）依赖于
        #    __stdout__ 未被分离：
        #    https://github.com/python/cpython/blob/c352e6c7446c894b13643f538db312092b351789/Lib/subprocess.py#L1214
        # 为了解决这个问题，我们直接传入文件描述符，并希望它是有效的。
        stdout_fileno = 1
        # 使用 subprocess.run 调用外部命令
        subprocess.run(
            command,  # 要执行的命令
            stdout=stdout_fileno if verbose else subprocess.PIPE,  # 根据 verbose 设置输出方式
            stderr=subprocess.STDOUT,  # 将标准错误流合并到标准输出流
            cwd=build_directory,  # 设置命令执行的当前工作目录
            check=True,  # 如果命令返回非零退出码，则引发 CalledProcessError
            env=env)  # 设置命令执行时的环境变量
    except subprocess.CalledProcessError as e:
        # 获取异常信息
        _, error, _ = sys.exc_info()
        # 构造错误消息前缀
        message = error_prefix
        # 如果 error 对象有 'output' 属性，并且不为空
        if hasattr(error, 'output') and error.output:  # type: ignore[union-attr]
            # 将错误输出解码为字符串并添加到消息中
            message += f": {error.output.decode(*SUBPROCESS_DECODE_ARGS)}"  # type: ignore[union-attr]
        # 抛出自定义异常 RuntimeError
        raise RuntimeError(message) from e
# 构建执行路径，返回模块的完整执行路径
def _get_exec_path(module_name, path):
    # 如果在 Windows 平台且 TORCH_LIB_PATH 不在环境变量 PATH 中
    if IS_WINDOWS and TORCH_LIB_PATH not in os.getenv('PATH', '').split(';'):
        # 检查 TORCH_LIB_PATH 是否在任意 PATH 中，并且路径存在且相同
        torch_lib_in_path = any(
            os.path.exists(p) and os.path.samefile(p, TORCH_LIB_PATH)
            for p in os.getenv('PATH', '').split(';')
        )
        # 如果 TORCH_LIB_PATH 不在 PATH 中，则添加到环境变量 PATH 中
        if not torch_lib_in_path:
            os.environ['PATH'] = f"{TORCH_LIB_PATH};{os.getenv('PATH', '')}"
    # 返回模块的完整执行路径
    return os.path.join(path, f'{module_name}{EXEC_EXT}')


# 从库中导入模块
def _import_module_from_library(module_name, path, is_python_module):
    # 构建文件路径
    filepath = os.path.join(path, f"{module_name}{LIB_EXT}")
    if is_python_module:
        # 如果是 Python 模块，根据文件路径创建模块的规范
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        assert spec is not None  # 断言规范不为空
        # 根据规范创建模块
        module = importlib.util.module_from_spec(spec)
        assert isinstance(spec.loader, importlib.abc.Loader)  # 断言规范的加载器为 Loader 类型
        # 执行模块
        spec.loader.exec_module(module)
        return module  # 返回导入的模块
    else:
        torch.ops.load_library(filepath)  # 否则加载 Torch 操作的库文件


# 写入用于构建库的 Ninja 文件
def _write_ninja_file_to_build_library(path,
                                       name,
                                       sources,
                                       extra_cflags,
                                       extra_cuda_cflags,
                                       extra_ldflags,
                                       extra_include_paths,
                                       with_cuda,
                                       is_standalone) -> None:
    # 清理额外的编译标志，去除前后空格
    extra_cflags = [flag.strip() for flag in extra_cflags]
    extra_cuda_cflags = [flag.strip() for flag in extra_cuda_cflags]
    extra_ldflags = [flag.strip() for flag in extra_ldflags]
    extra_include_paths = [flag.strip() for flag in extra_include_paths]

    # 将用户指定的包含路径转换为绝对路径，以便在 Ninja 构建文件中使用
    user_includes = [os.path.abspath(file) for file in extra_include_paths]

    # 获取系统的包含路径，包括 torch/extension.h 的位置
    system_includes = include_paths(with_cuda)

    # 获取 Python.h 的位置，使用 'posix_prefix' 方案解决某些 MacOS 安装上的问题
    python_include_path = sysconfig.get_path('include', scheme='nt' if IS_WINDOWS else 'posix_prefix')
    if python_include_path is not None:
        system_includes.append(python_include_path)

    common_cflags = []
    if not is_standalone:
        # 如果不是独立模式，添加 TORCH_EXTENSION_NAME 和 TORCH_API_INCLUDE_EXTENSION_H 标志
        common_cflags.append(f'-DTORCH_EXTENSION_NAME={name}')
        common_cflags.append('-DTORCH_API_INCLUDE_EXTENSION_H')

    common_cflags += [f"{x}" for x in _get_pybind11_abi_build_flags()]

    # Windows 不支持 `-isystem`，将用户和系统包含路径都加入 common_cflags
    if IS_WINDOWS:
        common_cflags += [f'-I{include}' for include in user_includes + system_includes]
    else:
        # 如果不是 Windows 系统，则添加用户指定的头文件路径和系统头文件路径到编译标志中
        common_cflags += [f'-I{shlex.quote(include)}' for include in user_includes]
        common_cflags += [f'-isystem {shlex.quote(include)}' for include in system_includes]
    
    # 获取 GLIBCXX ABI 构建标志，并添加到编译标志中
    common_cflags += [f"{x}" for x in _get_glibcxx_abi_build_flags()]
    
    if IS_WINDOWS:
        # 如果是 Windows 系统，则设置 MSVC 编译标志，包括 C++17 标准和额外的编译标志
        cflags = common_cflags + COMMON_MSVC_FLAGS + ['/std:c++17'] + extra_cflags
        # 对编译标志进行引用处理
        cflags = _nt_quote_args(cflags)
    else:
        # 如果不是 Windows 系统，则设置通用的编译标志，包括 -fPIC 和 C++17 标准以及额外的编译标志
        cflags = common_cflags + ['-fPIC', '-std=c++17'] + extra_cflags
    
    if with_cuda and IS_HIP_EXTENSION:
        # 如果使用 CUDA 并且是 HIP 扩展，则设置 HIP 相关的编译标志
        cuda_flags = ['-DWITH_HIP'] + cflags + COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS
        cuda_flags += extra_cuda_cflags
        cuda_flags += _get_rocm_arch_flags(cuda_flags)
    elif with_cuda:
        # 如果使用 CUDA，则设置 NVIDIA CUDA 相关的编译标志
        cuda_flags = common_cflags + COMMON_NVCC_FLAGS + _get_cuda_arch_flags()
        if IS_WINDOWS:
            # 如果是 Windows 系统，则设置 MSVC 编译标志和 CUDA 特定的编译标志
            for flag in COMMON_MSVC_FLAGS:
                cuda_flags = ['-Xcompiler', flag] + cuda_flags
            for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                cuda_flags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cuda_flags
            cuda_flags = cuda_flags + ['-std=c++17']
            cuda_flags = _nt_quote_args(cuda_flags)
            cuda_flags += _nt_quote_args(extra_cuda_cflags)
        else:
            # 如果不是 Windows 系统，则设置通用的 CUDA 编译标志，包括 -fPIC 和额外的 CUDA 编译标志
            cuda_flags += ['--compiler-options', "'-fPIC'"]
            cuda_flags += extra_cuda_cflags
            # 如果 CUDA 编译标志中没有指定 C++17 标准，则添加该标准
            if not any(flag.startswith('-std=') for flag in cuda_flags):
                cuda_flags.append('-std=c++17')
            # 获取环境变量中的 C 编译器，并设置为 CUDA 的编译器
            cc_env = os.getenv("CC")
            if cc_env is not None:
                cuda_flags = ['-ccbin', cc_env] + cuda_flags
    else:
        # 如果不使用 CUDA，则设置为 None
        cuda_flags = None
    
    def object_file_path(source_file: str) -> str:
        # 将源文件路径转换为对象文件路径，例如 '/path/to/file.cpp' -> 'file.o' 或 'file.cuda.o'
        file_name = os.path.splitext(os.path.basename(source_file))[0]
        if _is_cuda_file(source_file) and with_cuda:
            # 如果是 CUDA 文件并且启用了 CUDA 支持，则使用不同的对象文件扩展名
            target = f'{file_name}.cuda.o'
        else:
            # 否则使用标准的对象文件扩展名
            target = f'{file_name}.o'
        return target
    
    # 根据源文件列表生成对象文件列表
    objects = [object_file_path(src) for src in sources]
    # 设置链接标志，如果不是独立库则包括共享库标志，以及额外的链接标志
    ldflags = ([] if is_standalone else [SHARED_FLAG]) + extra_ldflags
    
    # 如果是 macOS 系统，则需要显式指定允许未解析的符号
    if IS_MACOS:
        ldflags.append('-undefined dynamic_lookup')
    elif IS_WINDOWS:
        # 如果是 Windows 系统，则对链接标志进行引用处理
        ldflags = _nt_quote_args(ldflags)
    
    # 根据是否独立库设置输出文件扩展名
    ext = EXEC_EXT if is_standalone else LIB_EXT
    library_target = f'{name}{ext}'
    
    # 调用函数生成 Ninja 文件，用于构建相关任务
    _write_ninja_file(
        path=path,
        cflags=cflags,
        post_cflags=None,
        cuda_cflags=cuda_flags,
        cuda_post_cflags=None,
        cuda_dlink_post_cflags=None,
        sources=sources,
        objects=objects,
        ldflags=ldflags,
        library_target=library_target,
        with_cuda=with_cuda)
# 定义函数 _write_ninja_file，用于生成一个 ninja 文件来执行编译和链接操作
def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      cuda_dlink_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    # 定义内部函数 sanitize_flags，用于处理传入的标志列表，确保返回一个清理过的列表
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    # 对传入的各种标志进行清理处理
    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # 执行一些基本的断言检查，确保源文件列表和目标文件列表长度一致，且源文件数量大于零
    assert len(sources) == len(objects)
    assert len(sources) > 0

    # 获取 C++ 编译器的名称
    compiler = get_cxx_compiler()

    # 定义配置列表，指定所需的 ninja 版本
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')

    # 根据条件决定是否添加 nvcc 编译器配置，主要用于 CUDA 编译
    if with_cuda or cuda_dlink_post_cflags:
        if "PYTORCH_NVCC" in os.environ:
            nvcc = os.getenv("PYTORCH_NVCC")  # 用户可以通过环境变量设置 nvcc 编译器及其选项
        else:
            if IS_HIP_EXTENSION:
                nvcc = _join_rocm_home('bin', 'hipcc')  # 如果是 HIP 扩展，则使用 HIP 编译器
            else:
                nvcc = _join_cuda_home('bin', 'nvcc')  # 否则使用 CUDA 编译器
        config.append(f'nvcc = {nvcc}')

    # 如果是 HIP 扩展，则在 post_cflags 中添加通用 HIP 标志
    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags

    # 将各种标志转换为配置字符串，准备写入 ninja 文件
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # 将源文件的相对路径转换为绝对路径，以便在 ninja 构建文件中使用
    sources = [os.path.abspath(file) for file in sources]

    # 定义编译规则，指定编译的命令
    # 参考：https://ninja-build.org/build.ninja.html
    compile_rule = ['rule compile']
    # 如果运行环境是 Windows
    if IS_WINDOWS:
        # 添加编译规则到 compile_rule 列表，使用 MSVC 编译器的命令
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        # 设置依赖为 msvc
        compile_rule.append('  deps = msvc')
    else:
        # 如果运行环境不是 Windows，添加编译规则到 compile_rule 列表，使用 C++ 编译器的命令
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        # 设置生成依赖文件的名称为 $out.d
        compile_rule.append('  depfile = $out.d')
        # 设置依赖为 gcc
        compile_rule.append('  deps = gcc')

    # 如果启用了 CUDA 支持
    if with_cuda:
        # 初始化 cuda_compile_rule 列表，添加 CUDA 编译规则
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # 检查是否支持 Torch CUDA 版本且未设置 TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES 环境变量为 '1'
        if torch.version.cuda is not None and os.getenv('TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', '0') != '1':
            # 将依赖文件设置为 $out.d
            cuda_compile_rule.append('  depfile = $out.d')
            # 设置依赖为 gcc
            cuda_compile_rule.append('  deps = gcc')
            # 如果是 Linux 或未在 sccache 中使用 --generate-dependencies-with-compile 标志，添加 nvcc_gendeps 参数
            # 这有助于在 Windows 上使用 --generate-dependencies-with-compile
            nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        # 添加 CUDA 编译命令到 cuda_compile_rule 列表
        cuda_compile_rule.append(
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    # 针对每个源文件和目标文件的组合，生成单独的构建规则，以便支持增量构建
    build = []
    for source_file, object_file in zip(sources, objects):
        # 检查当前源文件是否是 CUDA 文件并且已启用 CUDA 支持
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        # 选择相应的编译规则名称
        rule = 'cuda_compile' if is_cuda_source else 'compile'
        # 如果运行环境是 Windows，将文件名中的冒号替换为 $:
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        # 将文件名中的空格替换为 $ 符号
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        # 将构建规则添加到 build 列表中
        build.append(f'build {object_file}: {rule} {source_file}')

    # 如果存在 cuda_dlink_post_cflags 参数
    if cuda_dlink_post_cflags:
        # 确定 devlink_out 文件的输出路径
        devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        # 初始化 devlink_rule 列表，添加 CUDA 设备链接规则
        devlink_rule = ['rule cuda_devlink']
        # 添加 CUDA 设备链接命令到 devlink_rule 列表
        devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        # 将设备链接规则添加到 devlink 列表中
        devlink = [f'build {devlink_out}: cuda_devlink {" ".join(objects)}']
        # 将 devlink_out 文件添加到 objects 列表中
        objects += [devlink_out]
    else:
        # 如果不存在 cuda_dlink_post_cflags 参数，初始化 devlink_rule 和 devlink 列表为空
        devlink_rule, devlink = [], []
    # 如果指定了库目标，则生成链接规则、链接命令和默认构建目标
    if library_target is not None:
        link_rule = ['rule link']
        
        # 如果运行环境是 Windows
        if IS_WINDOWS:
            # 使用 subprocess 调用系统命令 'where cl' 查找 cl.exe 的路径
            cl_paths = subprocess.check_output(['where', 'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            
            # 如果找到了至少一个 cl.exe 的路径
            if len(cl_paths) >= 1:
                # 获取 cl.exe 所在目录，并将 ':' 替换为 '$:'
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                # 如果未找到 cl.exe 路径，抛出运行时异常
                raise RuntimeError("MSVC is required to load C++ extensions")
            
            # 构建链接命令，使用找到的 cl.exe 的路径
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            # 如果不是 Windows 环境，构建默认的链接命令
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        # 生成构建规则，指定链接的目标和依赖的对象文件
        link = [f'build {library_target}: link {" ".join(objects)}']

        # 设置默认构建目标
        default = [f'default {library_target}']
    else:
        # 如果未指定库目标，清空链接规则、链接命令和默认构建目标
        link_rule, link, default = [], [], []

    # 将配置、标志、编译规则等各个部分组合成一个列表
    blocks = [config, flags, compile_rule]
    
    # 如果需要 CUDA 支持，添加 CUDA 编译规则到列表中
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
    
    # 将链接规则、构建规则、开发链接等加入列表中
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    
    # 将所有部分的内容用双空行分隔，形成最终的内容字符串
    content = "\n\n".join("\n".join(b) for b in blocks)
    
    # Ninja 构建系统要求 .ninja 文件以换行符结尾
    content += "\n"
    
    # 将生成的内容写入到指定的路径下
    _maybe_write(path, content)
# 将给定路径与 CUDA_HOME 联合，如果 CUDA_HOME 未设置则引发错误
def _join_cuda_home(*paths) -> str:
    """
    将路径与 CUDA_HOME 联合，如果 CUDA_HOME 未设置，则引发错误。

    这基本上是一种延迟引发错误的方式，用于在需要获取任何 CUDA 特定路径时仅引发一次错误。
    """
    if CUDA_HOME is None:
        raise OSError('CUDA_HOME 环境变量未设置。请将其设置为您的 CUDA 安装根目录。')
    return os.path.join(CUDA_HOME, *paths)


# 检查给定路径是否为 CUDA 文件
def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']  # 可接受的文件扩展名列表
    if IS_HIP_EXTENSION:
        valid_ext.append('.hip')  # 如果是 HIP 扩展，则接受 .hip 扩展名
    return os.path.splitext(path)[1] in valid_ext  # 返回路径的扩展名是否在可接受列表中
```