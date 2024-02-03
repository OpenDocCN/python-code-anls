# `bitsandbytes\bitsandbytes\cuda_setup\main.py`

```py
"""
extract factors the build is dependent on:
[X] compute capability
    [ ] TODO: Q - What if we have multiple GPUs of different makes?
- CUDA version
- Software:
    - CPU-only: only CPU quantization functions (no optimizer, no matrix multipl)
    - CuBLAS-LT: full-build 8-bit optimizer
    - no CuBLAS-LT: no 8-bit matrix multiplication (`nomatmul`)

evaluation:
    - if paths faulty, return meaningful error
    - else:
        - determine CUDA version
        - determine capabilities
        - based on that set the default path
"""

import ctypes as ct  # 导入 ctypes 库，用于调用 C 语言的函数
import errno  # 导入 errno 模块，用于处理错误码
import os  # 导入 os 模块，用于操作系统相关功能
from pathlib import Path  # 从 pathlib 模块导入 Path 类，用于处理文件路径
import platform  # 导入 platform 模块，用于获取系统信息
from typing import Set, Union  # 导入类型提示相关的模块
from warnings import warn  # 从 warnings 模块导入 warn 函数，用于发出警告

import torch  # 导入 torch 库，用于深度学习

from .env_vars import get_potentially_lib_path_containing_env_vars  # 从当前目录下的 env_vars 模块中导入函数

# these are the most common libs names
# libcudart.so is missing by default for a conda install with PyTorch 2.0 and instead
# we have libcudart.so.11.0 which causes a lot of errors before
# not sure if libcudart.so.12.0 exists in pytorch installs, but it does not hurt
system = platform.system()  # 获取当前系统信息
if system == 'Windows':  # 如果是 Windows 系统
    CUDA_RUNTIME_LIBS = ["nvcuda.dll"]  # 设置 CUDA 运行时库列表为 nvcuda.dll
else:  # 如果是 Linux 或其他系统
    CUDA_RUNTIME_LIBS = ["libcudart.so", 'libcudart.so.11.0', 'libcudart.so.12.0', 'libcudart.so.12.1', 'libcudart.so.12.2']  # 设置 CUDA 运行时库列表

# this is a order list of backup paths to search CUDA in, if it cannot be found in the main environmental paths
backup_paths = []  # 初始化备用路径列表
backup_paths.append('$CONDA_PREFIX/lib/libcudart.so.11.0')  # 将备用路径添加到列表中

class CUDASetup:  # 定义 CUDASetup 类
    _instance = None  # 类属性 _instance 初始化为 None

    def __init__(self):  # 定义初始化方法
        raise RuntimeError("Call get_instance() instead")  # 抛出运行时错误，提示应该调用 get_instance() 方法

    def initialize(self):  # 定义初始化方法
        if not getattr(self, 'initialized', False):  # 如果 initialized 属性不存在或为 False
            self.has_printed = False  # 初始化 has_printed 属性为 False
            self.lib = None  # 初始化 lib 属性为 None
            self.initialized = False  # 初始化 initialized 属性为 False
            self.error = False  # 初始化 error 属性为 False
    # 手动覆盖函数，用于检查是否有可用的 CUDA，并根据环境变量进行手动覆盖
    def manual_override(self):
        # 检查是否有可用的 CUDA
        if torch.cuda.is_available():
            # 检查环境变量中是否存在 BNB_CUDA_VERSION
            if 'BNB_CUDA_VERSION' in os.environ:
                # 检查 BNB_CUDA_VERSION 环境变量的长度是否大于0
                if len(os.environ['BNB_CUDA_VERSION']) > 0:
                    # 发出警告信息，提示手动覆盖 CUDA 版本
                    warn(
                        f'\n\n{"=" * 80}\n'
                        'WARNING: Manual override via BNB_CUDA_VERSION env variable detected!\n'
                        'BNB_CUDA_VERSION=XXX can be used to load a bitsandbytes version that is different from the PyTorch CUDA version.\n'
                        'If this was unintended set the BNB_CUDA_VERSION variable to an empty string: export BNB_CUDA_VERSION=\n'
                        'If you use the manual override make sure the right libcudart.so is in your LD_LIBRARY_PATH\n'
                        'For example by adding the following to your .bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_cuda_dir/lib64\n'
                        f'Loading CUDA version: BNB_CUDA_VERSION={os.environ["BNB_CUDA_VERSION"]}'
                        f'\n{"=" * 80}\n\n'
                    )
                    # 获取二进制文件名，并根据环境变量中的 CUDA 版本进行修改
                    binary_name = self.binary_name.rsplit(".", 1)[0]
                    suffix = ".so" if os.name != "nt" else ".dll"
                    self.binary_name = binary_name[:-3] + f'{os.environ["BNB_CUDA_VERSION"]}.{suffix}'

    # 添加日志条目函数，将消息和是否为警告信息添加到 cuda_setup_log 列表中
    def add_log_entry(self, msg, is_warning=False):
        self.cuda_setup_log.append((msg, is_warning))

    # 打印日志堆栈函数，遍历 cuda_setup_log 列表，根据是否为警告信息进行打印
    def print_log_stack(self):
        for msg, is_warning in self.cuda_setup_log:
            if is_warning:
                warn(msg)
            else:
                print(msg)

    # 类方法，获取单例实例
    @classmethod
    def get_instance(cls):
        # 如果 _instance 为 None，则创建新实例并初始化
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance
# 检查给定的 CUDA 计算能力是否与 cuBLASLt 兼容
def is_cublasLt_compatible(cc):
    # 初始化 cuBLASLt 兼容性标志为 False
    has_cublaslt = False
    # 如果 CUDA 计算能力不为空
    if cc is not None:
        # 将 CUDA 计算能力拆分为主版本号和次版本号
        cc_major, cc_minor = cc.split('.')
        # 如果主版本号小于 7 或者 (主版本号等于 7 且次版本号小于 5)
        if int(cc_major) < 7 or (int(cc_major) == 7 and int(cc_minor) < 5):
            # 记录警告日志，提示用户 CUDA 计算能力低于 7.5，只支持慢速的 8 位矩阵乘法
            CUDASetup.get_instance().add_log_entry("WARNING: Compute capability < 7.5 detected! Only slow 8-bit matmul is supported for your GPU! \
                    If you run into issues with 8-bit matmul, you can try 4-bit quantization: https://huggingface.co/blog/4bit-transformers-bitsandbytes", is_warning=True)
        else:
            # 设置 cuBLASLt 兼容性标志为 True
            has_cublaslt = True
    # 返回 cuBLASLt 兼容性标志
    return has_cublaslt

# 提取候选路径列表中的有效路径
def extract_candidate_paths(paths_list_candidate: str) -> Set[Path]:
    # 使用 os.pathsep 分割路径列表，创建路径对象集合
    return {Path(ld_path) for ld_path in paths_list_candidate.split(os.pathsep) if ld_path}

# 移除不存在的目录
def remove_non_existent_dirs(candidate_paths: Set[Path]) -> Set[Path]:
    # 初始化存在的目录集合
    existent_directories: Set[Path] = set()
    # 遍历候选路径集合
    for path in candidate_paths:
        try:
            # 如果路径存在
            if path.exists():
                # 将路径添加到存在的目录集合中
                existent_directories.add(path)
        except PermissionError:
            # 处理权限错误，因为它是 OSError 的子类型
            pass
        except OSError as exc:
            # 如果不是路径过长错误，则抛出异常
            if exc.errno != errno.ENAMETOOLONG:
                raise exc

    # 计算不存在的目录集合
    non_existent_directories: Set[Path] = candidate_paths - existent_directories
    # 如果存在不存在的目录
    if non_existent_directories:
        # 记录日志，提示用户存在非存在的目录
        CUDASetup.get_instance().add_log_entry(
            f"The following directories listed in your path were found to be non-existent: {non_existent_directories}",
            is_warning=False,
        )

    # 返回存在的目录集合
    return existent_directories

# 获取 CUDA 运行时库路径
def get_cuda_runtime_lib_paths(candidate_paths: Set[Path]) -> Set[Path]:
    # 初始化路径集合
    paths = set()
    # 遍历 CUDA 运行时库列表
    for libname in CUDA_RUNTIME_LIBS:
        # 遍历候选路径集合
        for path in candidate_paths:
            try:
                # 如果路径下的库文件存在
                if (path / libname).is_file():
                    # 将路径添加到路径集合中
                    paths.add(path / libname)
            except PermissionError:
                pass
    # 返回变量 paths，结束函数并将 paths 作为返回值
    return paths
# 解析路径列表候选字符串，返回有效路径集合
def resolve_paths_list(paths_list_candidate: str) -> Set[Path]:
    # 调用函数提取候选路径列表中存在的有效路径
    return remove_non_existent_dirs(extract_candidate_paths(paths_list_candidate))


# 在路径列表候选字符串中查找 CUDA 运行时库，返回路径集合
def find_cuda_lib_in(paths_list_candidate: str) -> Set[Path]:
    # 调用函数解析路径列表候选字符串，获取有效路径集合
    return get_cuda_runtime_lib_paths(
        resolve_paths_list(paths_list_candidate)
    )


# 如果存在重复的路径，发出警告
def warn_in_case_of_duplicates(results_paths: Set[Path]) -> None:
    if len(results_paths) > 1:
        # 构建警告消息字符串
        warning_msg = (
            f"Found duplicate {CUDA_RUNTIME_LIBS} files: {results_paths}.. "
            "We select the PyTorch default libcudart.so, which is {torch.version.cuda},"
            "but this might missmatch with the CUDA version that is needed for bitsandbytes."
            "To override this behavior set the BNB_CUDA_VERSION=<version string, e.g. 122> environmental variable"
            "For example, if you want to use the CUDA version 122"
            "BNB_CUDA_VERSION=122 python ..."
            "OR set the environmental variable in your .bashrc: export BNB_CUDA_VERSION=122"
            "In the case of a manual override, make sure you set the LD_LIBRARY_PATH, e.g."
            "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2")
        # 添加警告消息到日志
        CUDASetup.get_instance().add_log_entry(warning_msg, is_warning=True)


# 确定 CUDA 运行时库路径，返回路径或 None
def determine_cuda_runtime_lib_path() -> Union[Path, None]:
    """
        Searches for a cuda installations, in the following order of priority:
            1. active conda env
            2. LD_LIBRARY_PATH
            3. any other env vars, while ignoring those that
                - are known to be unrelated (see `bnb.cuda_setup.env_vars.to_be_ignored`)
                - don't contain the path separator `/`

        If multiple libraries are found in part 3, we optimistically try one,
        while giving a warning message.
    """
    # 获取可能包含库路径的环境变量
    candidate_env_vars = get_potentially_lib_path_containing_env_vars()
    # 创建一个空集合来存储 CUDA 运行时库的路径
    cuda_runtime_libs = set()
    
    # 检查环境变量中是否包含 CONDA_PREFIX
    if "CONDA_PREFIX" in candidate_env_vars:
        # 获取 CONDA_PREFIX 环境变量的路径
        conda_libs_path = Path(candidate_env_vars["CONDA_PREFIX"]) / "lib"
    
        # 在 conda_libs_path 路径下查找 CUDA 库
        conda_cuda_libs = find_cuda_lib_in(str(conda_libs_path))
        # 检查是否有重复的 CUDA 库路径
        warn_in_case_of_duplicates(conda_cuda_libs)
    
        # 如果找到 CUDA 库路径，则将其添加到 cuda_runtime_libs 集合中
        if conda_cuda_libs:
            cuda_runtime_libs.update(conda_cuda_libs)
    
        # 记录日志，指示 CONDA_PREFIX 路径不包含预期的 CUDA 运行时库
        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["CONDA_PREFIX"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)
    
    # 检查环境变量中是否包含 LD_LIBRARY_PATH
    if "LD_LIBRARY_PATH" in candidate_env_vars:
        # 在 LD_LIBRARY_PATH 路径下查找 CUDA 库
        lib_ld_cuda_libs = find_cuda_lib_in(candidate_env_vars["LD_LIBRARY_PATH"])
    
        # 如果找到 CUDA 库路径，则将其添加到 cuda_runtime_libs 集合中
        if lib_ld_cuda_libs:
            cuda_runtime_libs.update(lib_ld_cuda_libs)
        # 检查是否有重复的 CUDA 库路径
        warn_in_case_of_duplicates(lib_ld_cuda_libs)
    
        # 记录日志，指示 LD_LIBRARY_PATH 路径不包含预期的 CUDA 运行时库
        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["LD_LIBRARY_PATH"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)
    
    # 从 candidate_env_vars 中移除 CONDA_PREFIX 和 LD_LIBRARY_PATH 环境变量，剩余的存储在 remaining_candidate_env_vars 中
    remaining_candidate_env_vars = {
        env_var: value for env_var, value in candidate_env_vars.items()
        if env_var not in {"CONDA_PREFIX", "LD_LIBRARY_PATH"}
    }
    
    # 遍历剩余的环境变量，查找 CUDA 库路径并添加到 cuda_runtime_libs 集合中
    for env_var, value in remaining_candidate_env_vars.items():
        cuda_runtime_libs.update(find_cuda_lib_in(value))
    
    # 如果未找到任何 CUDA 运行时库路径，则在备用路径中查找 libcudart.so
    if len(cuda_runtime_libs) == 0:
        CUDASetup.get_instance().add_log_entry('CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching in backup paths...')
        cuda_runtime_libs.update(find_cuda_lib_in('/usr/local/cuda/lib64'))
    
    # 检查是否有重复的 CUDA 库路径
    warn_in_case_of_duplicates(cuda_runtime_libs)
    
    # 获取 CUDASetup 实例
    cuda_setup = CUDASetup.get_instance()
    # 记录日志，指示找到的可能的 libcudart.so 选项
    cuda_setup.add_log_entry(f'DEBUG: Possible options found for libcudart.so: {cuda_runtime_libs}')
    
    # 返回 cuda_runtime_libs 集合中的第一个路径，如果为空则返回 None
    return next(iter(cuda_runtime_libs)) if cuda_runtime_libs else None
# 获取当前 CUDA 版本号
def get_cuda_version():
    # 将 CUDA 版本号拆分成主版本号和次版本号
    major, minor = map(int, torch.version.cuda.split("."))

    # 如果主版本号小于 11，则输出警告信息
    if major < 11:
        CUDASetup.get_instance().add_log_entry('CUDA SETUP: CUDA version lower than 11 are currently not supported for LLM.int8(). You will be only to use 8-bit optimizers and quantization routines!!')

    # 返回 CUDA 版本号
    return f'{major}{minor}'

# 获取当前设备的计算能力
def get_compute_capabilities():
    ccs = []
    # 遍历所有 CUDA 设备，获取其计算能力
    for i in range(torch.cuda.device_count()):
        cc_major, cc_minor = torch.cuda.get_device_capability(torch.cuda.device(i))
        ccs.append(f"{cc_major}.{cc_minor}")

    # 按照计算能力排序
    ccs.sort(key=lambda v: tuple(map(int, str(v).split(".")))

    # 返回计算能力列表
    return ccs

# 评估 CUDA 设置
def evaluate_cuda_setup():
    # 获取 CUDA 设置实例
    cuda_setup = CUDASetup.get_instance()
    # 根据操作系统类型确定库文件后缀
    suffix = ".so" if os.name != "nt" else ".dll"
    # 如果环境变量中没有指定 BITSANDBYTES_NOWELCOME 或者为 0，则输出欢迎信息
    if 'BITSANDBYTES_NOWELCOME' not in os.environ or str(os.environ['BITSANDBYTES_NOWELCOME']) == '0':
        cuda_setup.add_log_entry('')
        cuda_setup.add_log_entry('='*35 + 'BUG REPORT' + '='*35)
        cuda_setup.add_log_entry(('Welcome to bitsandbytes. For bug reports, please run\n\npython -m bitsandbytes\n\n'),
              ('and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues'))
        cuda_setup.add_log_entry('='*80)
    # 如果当前设备不支持 CUDA，则返回相应信息
    if not torch.cuda.is_available(): return f'libbitsandbytes_cpu{suffix}', None, None, None

    # 确定 CUDA 运行时库路径
    cudart_path = determine_cuda_runtime_lib_path()
    # 获取设备的计算能力列表
    ccs = get_compute_capabilities()
    ccs.sort()
    # 获取最高计算能力
    cc = ccs[-1] # we take the highest capability
    # 获取当前 CUDA 版本号字符串
    cuda_version_string = get_cuda_version()

    # 输出 CUDA 设置信息
    cuda_setup.add_log_entry(f"CUDA SETUP: PyTorch settings found: CUDA_VERSION={cuda_version_string}, Highest Compute Capability: {cc}.")
    cuda_setup.add_log_entry(
        "CUDA SETUP: To manually override the PyTorch CUDA version please see:"
        "https://github.com/TimDettmers/bitsandbytes/blob/main/how_to_use_nonpytorch_cuda.md"
    )

    # 检查是否支持cublasLt，7.5是最低要求
    has_cublaslt = is_cublasLt_compatible(cc)

    # TODO:
    # (1) CUDA缺失情况（没有通过CUDA驱动程序安装CUDA（nvidia-smi不可访问）
    # (2) 安装了多个CUDA版本

    # 我们使用ls -l而不是nvcc来确定CUDA版本
    # 因为大多数安装将安装libcudart.so，但没有编译器

    # 根据是否支持cublasLt选择二进制文件名
    if has_cublaslt:
        binary_name = f"libbitsandbytes_cuda{cuda_version_string}"
    else:
        "如果不支持cublasLt（CC < 7.5），则必须选择_nocublaslt"
        binary_name = f"libbitsandbytes_cuda{cuda_version_string}_nocublaslt"

    # 添加后缀到二进制文件名
    binary_name = f"{binary_name}{suffix}"

    # 返回二进制文件名、cudart路径、CC、CUDA版本字符串
    return binary_name, cudart_path, cc, cuda_version_string
```