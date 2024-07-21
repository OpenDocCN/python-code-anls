# `.\pytorch\test\cpp_extensions\setup.py`

```py
# 导入必要的标准库和模块
import os
import sys

# 导入 setuptools 中的 setup 函数，用于包的安装和分发
from setuptools import setup

# 导入 torch.cuda 模块，用于检查 CUDA 设备的可用性
import torch.cuda

# 导入 torch.testing._internal.common_utils 模块中的 IS_WINDOWS 变量
from torch.testing._internal.common_utils import IS_WINDOWS

# 导入 torch.utils.cpp_extension 模块中的各种扩展相关类和常量
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDA_HOME,
    CUDAExtension,
    ROCM_HOME,
)

# 如果运行平台为 win32（Windows），根据环境变量 VCToolsVersion 设置编译标志
if sys.platform == "win32":
    vc_version = os.getenv("VCToolsVersion", "")
    if vc_version.startswith("14.16."):
        CXX_FLAGS = ["/sdl"]  # 使用 /sdl 编译标志
    else:
        CXX_FLAGS = ["/sdl", "/permissive-"]  # 使用 /sdl 和 /permissive- 编译标志
else:
    CXX_FLAGS = ["-g"]  # 非 Windows 平台默认使用 -g 编译标志

# 根据环境变量 USE_NINJA 设置是否使用 Ninja 构建系统
USE_NINJA = os.getenv("USE_NINJA") == "1"

# 定义扩展模块列表
ext_modules = [
    # 创建 CppExtension 对象，编译名为 torch_test_cpp_extension.cpp 的 C++ 扩展模块
    CppExtension(
        "torch_test_cpp_extension.cpp", ["extension.cpp"], extra_compile_args=CXX_FLAGS
    ),
    # 创建 CppExtension 对象，编译名为 torch_test_cpp_extension.maia 的 C++ 扩展模块
    CppExtension(
        "torch_test_cpp_extension.maia",
        ["maia_extension.cpp"],
        extra_compile_args=CXX_FLAGS,
    ),
    # 创建 CppExtension 对象，编译名为 torch_test_cpp_extension.rng 的 C++ 扩展模块
    CppExtension(
        "torch_test_cpp_extension.rng",
        ["rng_extension.cpp"],
        extra_compile_args=CXX_FLAGS,
    ),
]

# 如果 CUDA 可用且存在 CUDA_HOME 或 ROCM_HOME
if torch.cuda.is_available() and (CUDA_HOME is not None or ROCM_HOME is not None):
    # 创建 CUDAExtension 对象，编译名为 torch_test_cpp_extension.cuda 的 CUDA 扩展模块
    extension = CUDAExtension(
        "torch_test_cpp_extension.cuda",
        [
            "cuda_extension.cpp",
            "cuda_extension_kernel.cu",
            "cuda_extension_kernel2.cu",
        ],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": ["-O2"]},  # 设置编译标志
    )
    ext_modules.append(extension)  # 将 CUDA 扩展模块对象添加到扩展模块列表中

# 如果 CUDA 可用且存在 CUDA_HOME 或 ROCM_HOME
if torch.cuda.is_available() and (CUDA_HOME is not None or ROCM_HOME is not None):
    # 创建 CUDAExtension 对象，编译名为 torch_test_cpp_extension.torch_library 的 CUDA 扩展模块
    extension = CUDAExtension(
        "torch_test_cpp_extension.torch_library",
        ["torch_library.cu"],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": ["-O2"]},  # 设置编译标志
    )
    ext_modules.append(extension)  # 将 CUDA 扩展模块对象添加到扩展模块列表中

# 如果使用了 MPS（Memory Pooling Subsystem），创建名为 torch_test_cpp_extension.mps 的 C++ 扩展模块
if torch.backends.mps.is_available():
    extension = CppExtension(
        "torch_test_cpp_extension.mps",
        ["mps_extension.mm"],
        extra_compile_args=CXX_FLAGS,
    )
    ext_modules.append(extension)  # 将 MPS 扩展模块对象添加到扩展模块列表中

# todo(mkozuki): Figure out the root cause
# 如果不是 Windows 平台且 CUDA 可用并且存在 CUDA_HOME
if (not IS_WINDOWS) and torch.cuda.is_available() and CUDA_HOME is not None:
    # 创建名为 torch_test_cpp_extension.cublas_extension 的 CUDA 扩展模块
    cublas_extension = CUDAExtension(
        name="torch_test_cpp_extension.cublas_extension",
        sources=["cublas_extension.cpp"],
        libraries=["cublas"] if torch.version.hip is None else [],  # 根据 PyTorch 版本选择链接的库
    )
    ext_modules.append(cublas_extension)  # 将 cuBLAS 扩展模块对象添加到扩展模块列表中

    # 创建名为 torch_test_cpp_extension.cusolver_extension 的 CUDA 扩展模块
    cusolver_extension = CUDAExtension(
        name="torch_test_cpp_extension.cusolver_extension",
        sources=["cusolver_extension.cpp"],
        libraries=["cusolver"] if torch.version.hip is None else [],  # 根据 PyTorch 版本选择链接的库
    )
    ext_modules.append(cusolver_extension)  # 将 cuSolver 扩展模块对象添加到扩展模块列表中

# 如果使用了 Ninja 并且不是 Windows 平台且 CUDA 可用并且存在 CUDA_HOME
if USE_NINJA and (not IS_WINDOWS) and torch.cuda.is_available() and CUDA_HOME is not None:
    # 创建一个 CUDAExtension 对象，用于构建一个名为 torch_test_cpp_extension.cuda_dlink 的 CUDA 扩展
    extension = CUDAExtension(
        name="torch_test_cpp_extension.cuda_dlink",
        # 指定该扩展的源文件列表，包括 cpp 文件和多个 cu 文件
        sources=[
            "cuda_dlink_extension.cpp",
            "cuda_dlink_extension_kernel.cu",
            "cuda_dlink_extension_add.cu",
        ],
        # 指定此扩展需要链接到动态链接库 (dlink=True)
        dlink=True,
        # 提供额外的编译参数，对 C++ 编译器使用 CXX_FLAGS 变量，对 nvcc 使用指定的优化选项
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": ["-O2", "-dc"]},
    )
    # 将创建的 CUDAExtension 对象添加到 ext_modules 列表中，用于构建时编译和链接
    ext_modules.append(extension)
# 设置 Python 包的元数据，指定名称为 "torch_test_cpp_extension"
setup(
    # 包名为 "torch_test_cpp_extension"
    name="torch_test_cpp_extension",
    # 包含的 Python 子包列表
    packages=["torch_test_cpp_extension"],
    # 扩展模块的配置，这里使用了变量 ext_modules
    ext_modules=ext_modules,
    # 包含的头文件目录，此处似乎应为列表，但给定的是字符串 "self_compiler_include_dirs_test"
    include_dirs="self_compiler_include_dirs_test",
    # 自定义命令类的设置，此处指定了使用的命令为 "build_ext"，并传入了 BuildExtension 类的选项 use_ninja=USE_NINJA
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
```