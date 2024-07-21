# `.\pytorch\setup.py`

```py
# 欢迎来到 PyTorch 的 setup.py。
# 以下是可能对你感兴趣的环境变量说明：

# 调试模式，使用 -O0 和 -g 构建（包含调试符号）
DEBUG

# 使用优化和 -g 构建（包含调试符号）
REL_WITH_DEB_INFO

# 仅为指定文件使用调试信息进行构建，文件路径需以分号分隔
USE_CUSTOM_DEBINFO="path/to/file1.cpp;path/to/file2.cpp"

# 编译代码时最大并行任务数
MAX_JOBS

# 禁用 CUDA 构建
USE_CUDA=0

# 应用于所有 C 和 C++ 文件的编译标志，注意 CFLAGS 同样适用于 C++ 文件
CFLAGS

# 指定 C/C++ 编译器
CC

# 功能开关的环境变量：

# 如果与 DEBUG 或 REL_WITH_DEB_INFO 一同使用，将使用 -lineinfo --source-in-ptx 构建 CUDA 内核。
# 注意，在 CUDA 12 上可能导致 nvcc 内存溢出，默认情况下禁用。
DEBUG_CUDA=1

# 禁用 cuDNN 构建
USE_CUDNN=0

# 禁用 cuSPARSELt 构建
USE_CUSPARSELT=0

# 禁用 FBGEMM 构建
USE_FBGEMM=0

# 禁用 libkineto 库用于性能分析
USE_KINETO=0

# 禁用 NumPy 构建
USE_NUMPY=0

# 禁用测试构建
BUILD_TEST=0

# 禁用 MKLDNN 库的使用
USE_MKLDNN=0

# 启用 MKLDNN 在 Arm 架构上的 Compute Library 后端，需要显式启用 USE_MKLDNN
USE_MKLDNN_ACL

# MKL-DNN 的 CPU 运行时模式：TBB 或 OMP（默认）
MKLDNN_CPU_RUNTIME

# 在 Unix 下首选静态链接 MKL
USE_STATIC_MKL

# 禁用 Intel(R) VTune Profiler 的 ITT 功能
USE_ITT=0

# 禁用 NNPACK 构建
USE_NNPACK=0

# 禁用 QNNPACK 构建（量化的 8 位操作）
USE_QNNPACK=0

# 禁用分布式构建（c10d, gloo, mpi 等）
USE_DISTRIBUTED=0

# 禁用分布式 Tensorpipe 后端构建
USE_TENSORPIPE=0

# 禁用分布式 gloo 后端构建
USE_GLOO=0

# 禁用分布式 MPI 后端构建
USE_MPI=0

# 禁用系统范围的 nccl 使用（将使用第三方/nccl 中的副本）
USE_SYSTEM_NCCL=0

# 禁用 OpenMP 用于并行化
USE_OPENMP=0

# 禁用用于缩放点产品注意力的 flash attention 构建
USE_FLASH_ATTENTION=0

# 禁用用于缩放点产品注意力的内存高效 attention 构建
USE_MEM_EFF_ATTENTION=0

# 启用额外的二进制文件构建
BUILD_BINARY

# 如果 AVX512 在某些机器上性能不佳，可以设置 ATen AVX2 内核使用 32 个 ymm 寄存器（默认为 16 个）
ATEN_AVX512_256=TRUE

# 在 Xeon D 处理器上，FBGEMM 库也会使用 AVX512_256 内核
# but it also has some (optimized) assembly code.
#
# PYTORCH_BUILD_VERSION
# PYTORCH_BUILD_NUMBER
#   specify the version of PyTorch, rather than the hard-coded version
#   in this file; used when we're building binaries for distribution
#
# TORCH_CUDA_ARCH_LIST
#   specify which CUDA architectures to build for.
#   ie `TORCH_CUDA_ARCH_LIST="6.0;7.0"`
#   These are not CUDA versions, instead, they specify what
#   classes of NVIDIA hardware we should generate PTX for.
#
# PYTORCH_ROCM_ARCH
#   specify which AMD GPU targets to build for.
#   ie `PYTORCH_ROCM_ARCH="gfx900;gfx906"`
#
# ONNX_NAMESPACE
#   specify a namespace for ONNX built here rather than the hard-coded
#   one in this file; needed to build with other frameworks that share ONNX.
#
# BLAS
#   BLAS to be used by Caffe2. Can be MKL, Eigen, ATLAS, FlexiBLAS, or OpenBLAS. If set
#   then the build will fail if the requested BLAS is not found, otherwise
#   the BLAS will be chosen based on what is found on your system.
#
# MKL_THREADING
#   MKL threading mode: SEQ, TBB or OMP (default)
#
# USE_ROCM_KERNEL_ASSERT=1
#   Enable kernel assert in ROCm platform
#
# Environment variables we respect (these environment variables are
# conventional and are often understood/set by other software.)
#
# CUDA_HOME (Linux/OS X)
# CUDA_PATH (Windows)
#   specify where CUDA is installed; usually /usr/local/cuda or
#   /usr/local/cuda-x.y
# CUDAHOSTCXX
#   specify a different compiler than the system one to use as the CUDA
#   host compiler for nvcc.
#
# CUDA_NVCC_EXECUTABLE
#   Specify a NVCC to use. This is used in our CI to point to a cached nvcc
#
# CUDNN_LIB_DIR
# CUDNN_INCLUDE_DIR
# CUDNN_LIBRARY
#   specify where cuDNN is installed
#
# MIOPEN_LIB_DIR
# MIOPEN_INCLUDE_DIR
# MIOPEN_LIBRARY
#   specify where MIOpen is installed
#
# NCCL_ROOT
# NCCL_LIB_DIR
# NCCL_INCLUDE_DIR
#   specify where nccl is installed
#
# NVTOOLSEXT_PATH (Windows only)
#   specify where nvtoolsext is installed
#
# ACL_ROOT_DIR
#   specify where Compute Library is installed
#
# LIBRARY_PATH
# LD_LIBRARY_PATH
#   we will search for libraries in these paths
#
# ATEN_THREADING
#   ATen parallel backend to use for intra- and inter-op parallelism
#   possible values:
#     OMP - use OpenMP for intra-op and native backend for inter-op tasks
#     NATIVE - use native thread pool for both intra- and inter-op tasks
#
# USE_SYSTEM_LIBS (work in progress)
#   Use system-provided libraries to satisfy the build dependencies.
#   When turned on, the following cmake variables will be toggled as well:
#     USE_SYSTEM_CPUINFO=ON USE_SYSTEM_SLEEF=ON BUILD_CUSTOM_PROTOBUF=OFF
#
# USE_MIMALLOC
#   Static link mimalloc into C10, and use mimalloc in alloc_cpu & alloc_free.
#   By default, It is only enabled on Windows.
#
# USE_PRIORITIZED_TEXT_FOR_LD
# 使用 cmake/prioritized_text.txt 中的优先级文本形式来进行 LD
#
# 构建 libtorch.so 及其依赖项作为一个 wheel
#
# 使用独立的 wheel 构建 pytorch，使用来自 libtorch.so 的依赖项

import os
import sys

# 如果运行环境是 32 位的 Windows Python，不支持，提示用户切换到 64 位的 Python
if sys.platform == "win32" and sys.maxsize.bit_length() == 31:
    print(
        "32-bit Windows Python runtime is not supported. Please switch to 64-bit Python."
    )
    sys.exit(-1)

import platform

# 根据环境变量 BUILD_LIBTORCH_WHL 设置是否构建 libtorch 的 wheel
BUILD_LIBTORCH_WHL = os.getenv("BUILD_LIBTORCH_WHL", "0") == "1"
# 根据环境变量 BUILD_PYTHON_ONLY 设置是否仅构建 Python 的 wheel，并使用来自单独 wheel 的 libtorch.so
BUILD_PYTHON_ONLY = os.getenv("BUILD_PYTHON_ONLY", "0") == "1"

# 设置所需的最低 Python 版本为 3.8.0
python_min_version = (3, 8, 0)
python_min_version_str = ".".join(map(str, python_min_version))
if sys.version_info < python_min_version:
    # 如果当前 Python 版本低于所需版本，提示用户需要升级 Python
    print(
        f"You are using Python {platform.python_version()}. Python >={python_min_version_str} is required."
    )
    sys.exit(-1)

import filecmp
import glob
import importlib
import importlib.util
import json
import shutil
import subprocess
import sysconfig
import time
from collections import defaultdict

import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.sdist
from setuptools import Extension, find_packages, setup
from setuptools.dist import Distribution

# 导入自定义工具和函数
from tools.build_pytorch_libs import build_caffe2
from tools.generate_torch_version import get_torch_version
from tools.setup_helpers.cmake import CMake
from tools.setup_helpers.env import build_type, IS_DARWIN, IS_LINUX, IS_WINDOWS
from tools.setup_helpers.generate_linker_script import gen_linker_script


def _get_package_path(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec:
        # 如果包是命名空间包，get_data 可能会失败
        try:
            loader = spec.loader
            if loader is not None:
                file_path = loader.get_filename()  # 获取包的文件路径
                return os.path.dirname(file_path)
        except AttributeError:
            pass
    return None

# 如果设置了 BUILD_LIBTORCH_WHL 环境变量，设置环境变量 BUILD_FUNCTORCH 为 OFF
if BUILD_LIBTORCH_WHL:
    # 设置环境变量，仅构建 libtorch.so，不构建 libtorch_python.so
    # functorch 在没有 Python 支持时不被支持
    os.environ["BUILD_FUNCTORCH"] = "OFF"

# 如果设置了 BUILD_PYTHON_ONLY 环境变量，设置环境变量 BUILD_LIBTORCHLESS 为 ON
# 设置 LIBTORCH_LIB_PATH 为 torch 包的 lib 目录路径
if BUILD_PYTHON_ONLY:
    os.environ["BUILD_LIBTORCHLESS"] = "ON"
    os.environ["LIBTORCH_LIB_PATH"] = f"{_get_package_path('torch')}/lib"

################################################################################
# 从环境中解析的参数
################################################################################

# 是否输出详细的构建脚本信息
VERBOSE_SCRIPT = True
# 是否运行构建依赖项
RUN_BUILD_DEPS = True
# 查看用户是否在 setup.py 参数中传递了 quiet 标志，并在构建过程中尊重它
EMIT_BUILD_WARNING = False
# 是否重新运行 cmake
RERUN_CMAKE = False
# 是否仅运行 cmake
CMAKE_ONLY = False
# 过滤后的参数列表
filtered_args = []

# 遍历 sys.argv 中的参数，查找是否存在 "--cmake" 参数
for i, arg in enumerate(sys.argv):
    if arg == "--cmake":
        RERUN_CMAKE = True
        continue
    # 如果命令行参数 arg 等于 "--cmake-only"，表示只执行 CMake，之后停止。
    # 给用户机会来调整构建选项。
    if arg == "--cmake-only":
        # 设置 CMAKE_ONLY 标志为 True，表示仅执行 CMake，不继续后续构建步骤。
        CMAKE_ONLY = True
        # 继续处理下一个命令行参数。
        continue
    
    # 如果命令行参数 arg 等于 "rebuild" 或者 "build"，将其统一设为 "build"。
    # "rebuild" 已不再使用，改为使用 "build"，并设置 EMIT_BUILD_WARNING 为 True，发出构建警告。
    if arg == "rebuild" or arg == "build":
        arg = "build"  # rebuild is gone, make it build
        EMIT_BUILD_WARNING = True
    
    # 如果命令行参数 arg 等于 "--"，将之后的所有参数都添加到 filtered_args 中，并终止循环。
    if arg == "--":
        filtered_args += sys.argv[i:]
        break
    
    # 如果命令行参数 arg 等于 "-q" 或者 "--quiet"，将 VERBOSE_SCRIPT 设置为 False，表示静默执行脚本。
    if arg == "-q" or arg == "--quiet":
        VERBOSE_SCRIPT = False
    
    # 如果命令行参数 arg 在 ["clean", "egg_info", "sdist"] 中，则不执行构建依赖。
    if arg in ["clean", "egg_info", "sdist"]:
        RUN_BUILD_DEPS = False
    
    # 将当前命令行参数 arg 添加到 filtered_args 中，准备进入下一轮循环处理。
    filtered_args.append(arg)
# 将程序的命令行参数设为经过过滤的参数列表
sys.argv = filtered_args

# 如果设定了 VERBOSE_SCRIPT 标志
if VERBOSE_SCRIPT:

    # 定义 report 函数，用于打印参数
    def report(*args):
        print(*args)

else:

    # 定义 report 函数，但不执行任何操作（空函数）
    def report(*args):
        pass

    # 使 distutils 也能支持 --quiet 选项
    setuptools.distutils.log.warn = report

# 定义常量和已知变量，用于整个文件
cwd = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的绝对路径
lib_path = os.path.join(cwd, "torch", "lib")  # 构造 torch 库的路径
third_party_path = os.path.join(cwd, "third_party")  # 构造 third_party 文件夹的路径
caffe2_build_dir = os.path.join(cwd, "build")  # 构造 build 文件夹的路径

# CMAKE: Python 库的完整路径
if IS_WINDOWS:
    # 如果是 Windows 系统，构造 Python 库的路径
    cmake_python_library = "{}/libs/python{}.lib".format(
        sysconfig.get_config_var("prefix"), sysconfig.get_config_var("VERSION")
    )
    # 修正虚拟环境构建
    if not os.path.exists(cmake_python_library):
        cmake_python_library = "{}/libs/python{}.lib".format(
            sys.base_prefix, sysconfig.get_config_var("VERSION")
        )
else:
    # 如果是其他操作系统，构造 Python 库的路径
    cmake_python_library = "{}/{}".format(
        sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("INSTSONAME")
    )
cmake_python_include_dir = sysconfig.get_path("include")  # 获取 Python 头文件的路径

################################################################################
# Version, create_version_file, and package_name
################################################################################

# 获取环境变量中的 TORCH_PACKAGE_NAME，如果没有则默认为 "torch"
package_name = os.getenv("TORCH_PACKAGE_NAME", "torch")
# 获取环境变量中的 LIBTORCH_PACKAGE_NAME，如果正在构建 LIBTORCH 的 wheel 包，则将 package_name 设为 LIBTORCH_PACKAGE_NAME
LIBTORCH_PKG_NAME = os.getenv("LIBTORCH_PACKAGE_NAME", "torch_no_python")
if BUILD_LIBTORCH_WHL:
    package_name = LIBTORCH_PKG_NAME

# 获取环境变量中的 PACKAGE_TYPE，如果没有则默认为 "wheel"
package_type = os.getenv("PACKAGE_TYPE", "wheel")
# 获取当前 Torch 的版本号
version = get_torch_version()
report(f"Building wheel {package_name}-{version}")

# 创建 CMake 对象
cmake = CMake()


def get_submodule_folders():
    # 获取 .gitmodules 文件的路径
    git_modules_path = os.path.join(cwd, ".gitmodules")
    # 默认的 submodule 路径列表
    default_modules_path = [
        os.path.join(third_party_path, name)
        for name in [
            "gloo",
            "cpuinfo",
            "onnx",
            "foxi",
            "QNNPACK",
            "fbgemm",
            "cutlass",
        ]
    ]
    # 如果 .gitmodules 文件不存在，返回默认的 submodule 路径列表
    if not os.path.exists(git_modules_path):
        return default_modules_path
    # 否则，读取 .gitmodules 文件，获取 submodule 文件夹路径列表
    with open(git_modules_path) as f:
        return [
            os.path.join(cwd, line.split("=", 1)[1].strip())
            for line in f
            if line.strip().startswith("path")
        ]


def check_submodules():
    def check_for_files(folder, files):
        # 检查文件夹中是否存在指定的文件
        if not any(os.path.exists(os.path.join(folder, f)) for f in files):
            report("Could not find any of {} in {}".format(", ".join(files), folder))
            report("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)

    def not_exists_or_empty(folder):
        # 检查文件夹是否不存在或者为空
        return not os.path.exists(folder) or (
            os.path.isdir(folder) and len(os.listdir(folder)) == 0
        )

    # 如果设置了环境变量 USE_SYSTEM_LIBS，则直接返回
    if bool(os.getenv("USE_SYSTEM_LIBS", False)):
        return
    # 否则，获取 submodule 文件夹路径列表
    folders = get_submodule_folders()
    # 如果没有任何 submodule 文件夹存在，则提示用户初始化 submodule
    if all(not_exists_or_empty(folder) for folder in folders):
        report("Could not find any submodules. Did you run 'git submodule update --init --recursive'?")
        sys.exit(1)
    # 检查所有文件夹是否不存在或为空
    if all(not_exists_or_empty(folder) for folder in folders):
        try:
            # 尝试初始化子模块
            print(" --- Trying to initialize submodules")
            start = time.time()
            # 执行 git 命令来更新并初始化子模块（递归地）
            subprocess.check_call(
                ["git", "submodule", "update", "--init", "--recursive"], cwd=cwd
            )
            end = time.time()
            # 输出子模块初始化所花费的时间
            print(f" --- Submodule initialization took {end - start:.2f} sec")
        except Exception:
            # 如果初始化失败，打印错误信息
            print(" --- Submodule initalization failed")
            print("Please run:\n\tgit submodule update --init --recursive")
            # 退出程序，并返回状态码 1
            sys.exit(1)
    
    # 检查指定文件夹中是否存在特定文件
    for folder in folders:
        check_for_files(
            folder,
            [
                "CMakeLists.txt",
                "Makefile",
                "setup.py",
                "LICENSE",
                "LICENSE.md",
                "LICENSE.txt",
            ],
        )
    
    # 检查指定路径下的文件夹是否存在特定文件
    check_for_files(
        os.path.join(third_party_path, "fbgemm", "third_party", "asmjit"),
        ["CMakeLists.txt"],
    )
    
    # 检查指定路径下的文件夹是否存在特定文件
    check_for_files(
        os.path.join(third_party_path, "onnx", "third_party", "benchmark"),
        ["CMakeLists.txt"],
    )
# Windows 对符号链接的支持非常差。
# 因此，我们将使用文件复制而不是符号链接。
def mirror_files_into_torchgen():
    # 定义需要复制的文件路径对
    paths = [
        (
            "torchgen/packaged/ATen/native/native_functions.yaml",
            "aten/src/ATen/native/native_functions.yaml",
        ),
        ("torchgen/packaged/ATen/native/tags.yaml", "aten/src/ATen/native/tags.yaml"),
        ("torchgen/packaged/ATen/templates", "aten/src/ATen/templates"),
        ("torchgen/packaged/autograd", "tools/autograd"),
        ("torchgen/packaged/autograd/templates", "tools/autograd/templates"),
    ]
    # 遍历路径对，执行文件镜像操作
    for new_path, orig_path in paths:
        # 如果目标路径不存在，则创建相关目录结构
        if not os.path.exists(new_path):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

        # 复制原始位置的文件到新位置
        if os.path.isfile(orig_path):
            shutil.copyfile(orig_path, new_path)
            continue
        if os.path.isdir(orig_path):
            # 如果新路径已存在，则先删除以避免 shutil.copytree 失败
            if os.path.exists(new_path):
                shutil.rmtree(new_path)
            shutil.copytree(orig_path, new_path)
            continue
        # 如果原始文件既不是文件也不是目录，抛出运行时异常
        raise RuntimeError("Check the file paths in `mirror_files_into_torchgen()`")


# 在运行设置前所需的所有工作
def build_deps():
    # 打印版本信息
    report("-- Building version " + version)

    # 检查子模块
    check_submodules()
    # 检查 Python 依赖包 pyyaml
    check_pydep("yaml", "pyyaml")
    # 根据条件构建 Python 包
    build_python = not BUILD_LIBTORCH_WHL
    build_caffe2(
        version=version,
        cmake_python_library=cmake_python_library,
        build_python=build_python,
        rerun_cmake=RERUN_CMAKE,
        cmake_only=CMAKE_ONLY,
        cmake=cmake,
    )

    # 如果仅执行 CMake，则输出相应提示信息并退出
    if CMAKE_ONLY:
        report(
            'Finished running cmake. Run "ccmake build" or '
            '"cmake-gui build" to adjust build options and '
            '"python setup.py install" to build.'
        )
        sys.exit()

    # 在 Windows 上使用复制文件而不是符号链接
    sym_files = [
        "tools/shared/_utils_internal.py",
        "torch/utils/benchmark/utils/valgrind_wrapper/callgrind.h",
        "torch/utils/benchmark/utils/valgrind_wrapper/valgrind.h",
    ]
    orig_files = [
        "torch/_utils_internal.py",
        "third_party/valgrind-headers/callgrind.h",
        "third_party/valgrind-headers/valgrind.h",
    ]
    # 遍历符号链接文件和原始文件的列表，并进行比较和复制操作
    for sym_file, orig_file in zip(sym_files, orig_files):
        same = False
        # 如果符号链接文件存在
        if os.path.exists(sym_file):
            # 比较符号链接文件和原始文件是否相同
            if filecmp.cmp(sym_file, orig_file):
                same = True
            else:
                # 如果不同，删除符号链接文件
                os.remove(sym_file)
        # 如果不相同，则复制原始文件到符号链接位置
        if not same:
            shutil.copyfile(orig_file, sym_file)


################################################################################
# 构建依赖库
################################################################################

missing_pydep = """
Missing build dependency: Unable to `import {importname}`.
Please install it via `conda install {module}` or `pip install {module}`
""".strip()

# 定义一个多行字符串，用于生成缺失依赖的错误消息模板

def check_pydep(importname, module):
    # 检查是否能导入指定的模块，如果不能则抛出 ImportError
    try:
        importlib.import_module(importname)
    except ImportError as e:
        # 如果导入失败，抛出自定义的 RuntimeError 异常，包含详细的错误消息
        raise RuntimeError(
            missing_pydep.format(importname=importname, module=module)
        ) from e

# 定义一个函数，用于检查指定的 Python 依赖是否可用，如果不可用则抛出自定义异常

class build_ext(setuptools.command.build_ext.build_ext):
    # 创建一个自定义的 build_ext 类，继承自 setuptools.command.build_ext.build_ext
    def _embed_libomp(self):
        # 在 MacOS 的 wheel 包中复制 libiomp5.dylib/libomp.dylib
        lib_dir = os.path.join(self.build_lib, "torch", "lib")
        libtorch_cpu_path = os.path.join(lib_dir, "libtorch_cpu.dylib")
        # 如果 libtorch_cpu.dylib 不存在，则直接返回
        if not os.path.exists(libtorch_cpu_path):
            return
        # 解析 libtorch_cpu 的加载命令
        otool_cmds = (
            subprocess.check_output(["otool", "-l", libtorch_cpu_path])
            .decode("utf-8")
            .split("\n")
        )
        rpaths, libs = [], []
        for idx, line in enumerate(otool_cmds):
            # 如果是 LC_LOAD_DYLIB 命令，则获取加载的库名
            if line.strip() == "cmd LC_LOAD_DYLIB":
                lib_name = otool_cmds[idx + 2].strip()
                assert lib_name.startswith("name ")
                libs.append(lib_name.split(" ", 1)[1].rsplit("(", 1)[0][:-1])

            # 如果是 LC_RPATH 命令，则获取 rpath 路径
            if line.strip() == "cmd LC_RPATH":
                rpath = otool_cmds[idx + 2].strip()
                assert rpath.startswith("path ")
                rpaths.append(rpath.split(" ", 1)[1].rsplit("(", 1)[0][:-1])

        # 根据操作系统判断使用的 OpenMP 库名
        omp_lib_name = (
            "libomp.dylib" if os.uname().machine == "arm64" else "libiomp5.dylib"
        )
        # 构造 rpath 中 OpenMP 库的路径
        omp_rpath_lib_path = os.path.join("@rpath", omp_lib_name)
        # 如果 OpenMP 库路径不在已加载的库列表中，则返回
        if omp_rpath_lib_path not in libs:
            return

        # 从 rpath 路径复制 libomp/libiomp5
        for rpath in rpaths:
            source_lib = os.path.join(rpath, omp_lib_name)
            # 如果源库文件不存在，则继续下一个 rpath
            if not os.path.exists(source_lib):
                continue
            # 目标库文件路径
            target_lib = os.path.join(self.build_lib, "torch", "lib", omp_lib_name)
            # 复制文件
            self.copy_file(source_lib, target_lib)
            # 使用 install_name_tool 删除旧的 rpath，并将 @loader_path 添加到 rpath 中
            # 这样可以防止 delocate 尝试将另一个 OpenMP 库实例打包到 torch wheel 中，
            # 并且避免加载两个 libomp.dylib 到地址空间中，因为库会被其未解析的名称缓存
            subprocess.check_call(
                [
                    "install_name_tool",
                    "-rpath",
                    rpath,
                    "@loader_path",
                    libtorch_cpu_path,
                ]
            )
            break

        # 从 OpenMP_C_FLAGS 中获取 omp.h，并将其复制到 include 文件夹中
        omp_cflags = get_cmake_cache_vars()["OpenMP_C_FLAGS"]
        # 如果没有找到 OpenMP_C_FLAGS，则返回
        if not omp_cflags:
            return
        # 遍历 include 文件夹，找到 omp.h 并复制到目标路径
        for include_dir in [f[2:] for f in omp_cflags.split(" ") if f.startswith("-I")]:
            omp_h = os.path.join(include_dir, "omp.h")
            if not os.path.exists(omp_h):
                continue
            target_omp_h = os.path.join(self.build_lib, "torch", "include", "omp.h")
            # 复制 omp.h 文件
            self.copy_file(omp_h, target_omp_h)
            break
    def build_extensions(self):
        self.create_compile_commands()
        # 创建 caffe2 的 pybind 扩展
        # 这些扩展将被创建在 tmp_install/lib/pythonM.m/site-packages/caffe2/python/
        # 并需要复制到 build/lib.linux.... ，这是由 setuptools 的 "build" 命令创建的
        # 平台相关的构建文件夹。只有该文件夹的内容会默认被 "install" 命令安装。
        # 我们只为 Caffe2 的 pybind 扩展进行此复制
        caffe2_pybind_exts = [
            "caffe2.python.caffe2_pybind11_state",
            "caffe2.python.caffe2_pybind11_state_gpu",
            "caffe2.python.caffe2_pybind11_state_hip",
        ]
        if BUILD_LIBTORCH_WHL:
            caffe2_pybind_exts = []
        i = 0
        while i < len(self.extensions):
            ext = self.extensions[i]
            if ext.name not in caffe2_pybind_exts:
                i += 1
                continue
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            report(f"\nCopying extension {ext.name}")

            # 获取纯库路径，并去掉数据路径部分，得到相对的站点包路径
            relative_site_packages = (
                sysconfig.get_path("purelib")
                .replace(sysconfig.get_path("data"), "")
                .lstrip(os.path.sep)
            )
            # 构建源路径，用于复制文件
            src = os.path.join("torch", relative_site_packages, filename)
            if not os.path.exists(src):
                report(f"{src} does not exist")
                # 如果源文件不存在，则从 self.extensions 中删除该扩展
                del self.extensions[i]
            else:
                # 构建目标路径，准备复制文件
                dst = os.path.join(os.path.realpath(self.build_lib), filename)
                report(f"Copying {ext.name} from {src} to {dst}")
                dst_dir = os.path.dirname(dst)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                # 执行文件复制操作
                self.copy_file(src, dst)
                i += 1

        # 复制 functorch 扩展
        for i, ext in enumerate(self.extensions):
            if ext.name != "functorch._C":
                continue
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            fileext = os.path.splitext(filename)[1]
            src = os.path.join(os.path.dirname(filename), "functorch" + fileext)
            dst = os.path.join(os.path.realpath(self.build_lib), filename)
            if os.path.exists(src):
                report(f"Copying {ext.name} from {src} to {dst}")
                dst_dir = os.path.dirname(dst)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                # 执行文件复制操作
                self.copy_file(src, dst)

        # 调用父类方法继续构建扩展
        setuptools.command.build_ext.build_ext.build_extensions(self)

    def get_outputs(self):
        # 获取父类方法的输出
        outputs = setuptools.command.build_ext.build_ext.get_outputs(self)
        # 添加额外的输出目录
        outputs.append(os.path.join(self.build_lib, "caffe2"))
        report(f"setup.py::get_outputs returning {outputs}")
        return outputs
    def create_compile_commands(self):
        # 定义内部函数load，用于从文件中加载JSON数据并返回
        def load(filename):
            with open(filename) as f:
                return json.load(f)

        # 使用glob模块查找所有形如"build/*compile_commands.json"的文件
        ninja_files = glob.glob("build/*compile_commands.json")
        # 使用glob模块查找所有形如"torch/lib/build/*/compile_commands.json"的文件
        cmake_files = glob.glob("torch/lib/build/*/compile_commands.json")
        # 合并ninja_files和cmake_files中所有文件的内容，加载成JSON对象列表
        all_commands = [entry for f in ninja_files + cmake_files for entry in load(f)]

        # 对于all_commands中的每个条目，如果命令以"gcc "开头，则将其替换为"g++ "
        # 这是为了解决cquery在处理由python setup.py生成的以gcc开头的C++编译命令时可能遗漏C++头文件目录的问题
        for command in all_commands:
            if command["command"].startswith("gcc "):
                command["command"] = "g++ " + command["command"][4:]

        # 将更新后的all_commands转换为格式化良好的JSON字符串
        new_contents = json.dumps(all_commands, indent=2)
        contents = ""
        # 如果当前目录下存在"compile_commands.json"文件，则读取其内容
        if os.path.exists("compile_commands.json"):
            with open("compile_commands.json") as f:
                contents = f.read()
        # 如果"compile_commands.json"的当前内容与新生成的内容不同，则写入新内容
        if contents != new_contents:
            with open("compile_commands.json", "w") as f:
                f.write(new_contents)
class concat_license_files:
    """Merge LICENSE and LICENSES_BUNDLED.txt as a context manager

    LICENSE is the main PyTorch license, LICENSES_BUNDLED.txt is auto-generated
    from all the licenses found in ./third_party/. We concatenate them so there
    is a single license file in the sdist and wheels with all of the necessary
    licensing info.
    """

    def __init__(self, include_files=False):
        # 初始化两个文件的路径
        self.f1 = "LICENSE"
        self.f2 = "third_party/LICENSES_BUNDLED.txt"
        # 是否包含附加文件的标志
        self.include_files = include_files

    def __enter__(self):
        """Concatenate files"""
        
        # 保存旧的路径，并添加第三方路径到系统路径中
        old_path = sys.path
        sys.path.append(third_party_path)
        try:
            # 尝试导入构建脚本并创建合并的许可证文件
            from build_bundled import create_bundled
        finally:
            # 恢复原始系统路径
            sys.path = old_path
        
        # 打开LICENSE文件并读取其内容保存到bsd_text变量中
        with open(self.f1) as f1:
            self.bsd_text = f1.read()
        
        # 以追加模式打开LICENSE文件，并在末尾添加两个换行符
        with open(self.f1, "a") as f1:
            f1.write("\n\n")
            # 调用构建脚本来创建合并的许可证信息到f1文件中
            create_bundled(
                os.path.relpath(third_party_path), f1, include_files=self.include_files
            )

    def __exit__(self, exception_type, exception_value, traceback):
        """Restore content of f1"""
        # 将bsd_text中保存的原始LICENSE内容写回f1文件中
        with open(self.f1, "w") as f:
            f.write(self.bsd_text)


try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    # 当wheel未安装或未在命令行中指定bdist_wheel时，这段代码有用
    wheel_concatenate = None
else:
    # 需要为wheel创建适当的LICENSE.txt文件
    class wheel_concatenate(bdist_wheel):
        """check submodules on sdist to prevent incomplete tarballs"""

        def run(self):
            # 使用concat_license_files上下文管理器来合并许可证文件
            with concat_license_files(include_files=True):
                super().run()


class install(setuptools.command.install.install):
    def run(self):
        # 调用父类的run方法来执行安装操作
        super().run()


class clean(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import glob
        import re

        # 读取.gitignore文件中的忽略规则
        with open(".gitignore") as f:
            ignores = f.read()
            # 构建忽略文件的正则表达式模式
            pat = re.compile(r"^#( BEGIN NOT-CLEAN-FILES )?")
            for wildcard in filter(None, ignores.split("\n")):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # 如果找到标记，则停止读取.gitignore文件
                        break
                    # 忽略以'#'开头的行
                else:
                    # 去除路径中的'./'前缀
                    wildcard = wildcard.lstrip("./")

                    # 根据通配符删除匹配的文件或目录
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)
class sdist(setuptools.command.sdist.sdist):
    # 继承自 setuptools.command.sdist.sdist 类的 sdist 类
    def run(self):
        # 运行时，使用 concat_license_files 上下文管理器
        with concat_license_files():
            # 调用父类的 run 方法
            super().run()


def get_cmake_cache_vars():
    try:
        # 尝试获取 CMake 缓存变量并以 defaultdict 初始化，默认值为 False
        return defaultdict(lambda: False, cmake.get_cmake_cache_variables())
    except FileNotFoundError:
        # 如果 CMakeCache.txt 文件不存在，则返回一个默认值为 False 的 defaultdict
        # 可能是在干净的目录中运行 "python setup.py clean"
        return defaultdict(lambda: False)


def configure_extension_build():
    r"""Configures extension build options according to system environment and user's choice.

    Returns:
      The input to parameters ext_modules, cmdclass, packages, and entry_points as required in setuptools.setup.
    """
    
    # 获取 CMake 缓存变量
    cmake_cache_vars = get_cmake_cache_vars()

    ################################################################################
    # Configure compile flags
    ################################################################################

    library_dirs = []
    extra_install_requires = []

    if IS_WINDOWS:
        # 如果是在 Windows 系统下
        # /NODEFAULTLIB 确保只链接到 DLL 运行时，与 protobuf 和 ONNX 设置的标志匹配
        extra_link_args = ["/NODEFAULTLIB:LIBCMT.LIB"]
        # /MD 链接到 DLL 运行时，与 protobuf 和 ONNX 设置的标志匹配
        # /EHsc 关于标准 C++ 异常处理
        extra_compile_args = ["/MD", "/FS", "/EHsc"]
    else:
        # 如果不是在 Windows 系统下
        extra_link_args = []
        # 编译时的额外参数
        extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-Wno-strict-overflow",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers",
            "-Wno-unknown-pragmas",
            # Python 2.6 需要 -fno-strict-aliasing，参见
            # http://legacy.python.org/dev/peps/pep-3123/
            # 我们的代码中也依赖它（即使是 Python 3）。
            "-fno-strict-aliasing",
        ]

    # 将 lib_path 添加到 library_dirs
    library_dirs.append(lib_path)

    # 主编译参数
    main_compile_args = []
    # 主库
    main_libraries = ["torch_python"]

    # 主链接参数
    main_link_args = []
    # 主源文件
    main_sources = ["torch/csrc/stub.c"]

    # 如果构建 LIBTORCH_WHL
    if BUILD_LIBTORCH_WHL:
        main_libraries = ["torch"]
        main_sources = []

    # 如果构建类型是 debug
    if build_type.is_debug():
        if IS_WINDOWS:
            # Windows 下的调试参数
            extra_compile_args.append("/Z7")
            extra_link_args.append("/DEBUG:FULL")
        else:
            # 非 Windows 下的调试参数
            extra_compile_args += ["-O0", "-g"]
            extra_link_args += ["-O0", "-g"]

    # 如果构建类型是 rel_with_deb_info
    if build_type.is_rel_with_deb_info():
        if IS_WINDOWS:
            # Windows 下的调试信息参数
            extra_compile_args.append("/Z7")
            extra_link_args.append("/DEBUG:FULL")
        else:
            # 非 Windows 下的调试信息参数
            extra_compile_args += ["-g"]
            extra_link_args += ["-g"]

    # pypi cuda package 需要安装 cuda runtime、cudnn 和 cublas
    # 应包含在上传到 pypi 的所有 wheels 中
    pytorch_extra_install_requirements = os.getenv(
        "PYTORCH_EXTRA_INSTALL_REQUIREMENTS", ""
    )
    # 如果存在额外的 PyTorch 安装要求，则记录并加入到额外的安装依赖中
    if pytorch_extra_install_requirements:
        report(
            f"pytorch_extra_install_requirements: {pytorch_extra_install_requirements}"
        )
        extra_install_requires += pytorch_extra_install_requirements.split("|")

    # 如果运行环境是 macOS，并且目标架构为 M1
    if IS_DARWIN:
        # 获取当前的 macOS 目标架构设置
        macos_target_arch = os.getenv("CMAKE_OSX_ARCHITECTURES", "")
        # 如果目标架构是 arm64 或 x86_64
        if macos_target_arch in ["arm64", "x86_64"]:
            # 获取当前 macOS 系统根路径
            macos_sysroot_path = os.getenv("CMAKE_OSX_SYSROOT")
            # 如果系统根路径未设置，则使用 xcrun 命令获取
            if macos_sysroot_path is None:
                macos_sysroot_path = (
                    subprocess.check_output(
                        ["xcrun", "--show-sdk-path", "--sdk", "macosx"]
                    )
                    .decode("utf-8")
                    .strip()
                )
            # 添加额外的编译参数和链接参数以支持目标架构
            extra_compile_args += [
                "-arch",
                macos_target_arch,
                "-isysroot",
                macos_sysroot_path,
            ]
            extra_link_args += ["-arch", macos_target_arch]

    # 定义一个函数，根据当前系统平台返回相应的相对路径参数
    def make_relative_rpath_args(path):
        if IS_DARWIN:
            return ["-Wl,-rpath,@loader_path/" + path]
        elif IS_WINDOWS:
            return []
        else:
            return ["-Wl,-rpath,$ORIGIN/" + path]

    ################################################################################
    # 声明扩展和包
    ################################################################################

    # 初始化扩展列表和排除的包
    extensions = []
    excludes = ["tools", "tools.*"]
    # 如果不构建 Caffe2，则排除相关包
    if not cmake_cache_vars["BUILD_CAFFE2"]:
        excludes.extend(["caffe2", "caffe2.*"])
    # 如果不构建 Functorch，则排除相关包
    if not cmake_cache_vars["BUILD_FUNCTORCH"]:
        excludes.extend(["functorch", "functorch.*"])
    # 查找所有的包，排除指定的包
    packages = find_packages(exclude=excludes)
    # 定义 C 扩展对象，并配置其参数
    C = Extension(
        "torch._C",
        libraries=main_libraries,
        sources=main_sources,
        language="c",
        extra_compile_args=main_compile_args + extra_compile_args,
        include_dirs=[],
        library_dirs=library_dirs,
        extra_link_args=extra_link_args
        + main_link_args
        + make_relative_rpath_args("lib"),
    )
    # 将 C 扩展对象添加到扩展列表中
    extensions.append(C)

    # 下面的扩展由 cmake 构建，并在 build_extensions() 内部手动复制
    # 如果构建 Caffe2，则添加相应的 Pybind11 扩展
    if cmake_cache_vars["BUILD_CAFFE2"]:
        extensions.append(
            Extension(name="caffe2.python.caffe2_pybind11_state", sources=[]),
        )
        # 如果使用 CUDA，则添加相应的 GPU 扩展
        if cmake_cache_vars["USE_CUDA"]:
            extensions.append(
                Extension(name="caffe2.python.caffe2_pybind11_state_gpu", sources=[]),
            )
        # 如果使用 ROCm，则添加相应的 HIP 扩展
        if cmake_cache_vars["USE_ROCM"]:
            extensions.append(
                Extension(name="caffe2.python.caffe2_pybind11_state_hip", sources=[]),
            )
    # 如果构建 Functorch，则添加 Functorch 的 C 扩展
    if cmake_cache_vars["BUILD_FUNCTORCH"]:
        extensions.append(
            Extension(name="functorch._C", sources=[]),
        )
    # 定义一个字典，包含用于不同命令的类或函数
    cmdclass = {
        "bdist_wheel": wheel_concatenate,  # 将 "bdist_wheel" 命令与 wheel_concatenate 函数关联
        "build_ext": build_ext,  # 将 "build_ext" 命令与 build_ext 函数关联
        "clean": clean,  # 将 "clean" 命令与 clean 函数关联
        "install": install,  # 将 "install" 命令与 install 函数关联
        "sdist": sdist,  # 将 "sdist" 命令与 sdist 函数关联
    }
    
    # 定义一个字典，包含不同 entry points 的命令行脚本和对应的模块或函数
    entry_points = {
        "console_scripts": [
            "convert-caffe2-to-onnx = caffe2.python.onnx.bin.conversion:caffe2_to_onnx",
            # 将 "convert-caffe2-to-onnx" 命令关联到 caffe2.python.onnx.bin.conversion 模块的 caffe2_to_onnx 函数
            "convert-onnx-to-caffe2 = caffe2.python.onnx.bin.conversion:onnx_to_caffe2",
            # 将 "convert-onnx-to-caffe2" 命令关联到 caffe2.python.onnx.bin.conversion 模块的 onnx_to_caffe2 函数
            "torchrun = torch.distributed.run:main",  
            # 将 "torchrun" 命令关联到 torch.distributed.run 模块的 main 函数
        ],
        "torchrun.logs_specs": [
            "default = torch.distributed.elastic.multiprocessing:DefaultLogsSpecs",
            # 将 "default" logs spec 关联到 torch.distributed.elastic.multiprocessing 模块的 DefaultLogsSpecs 类
        ],
    }
    
    # 返回包含多个变量的元组，依次为扩展、命令字典、包、entry points、额外安装依赖
    return extensions, cmdclass, packages, entry_points, extra_install_requires
# 定义多行字符串变量，包含构建更新消息的说明和指导信息
build_update_message = """
    It is no longer necessary to use the 'build' or 'rebuild' targets

    To install:
      $ python setup.py install
    To develop locally:
      $ python setup.py develop
    To force cmake to re-generate native build files (off by default):
      $ python setup.py develop --cmake
"""

# 定义打印消息框函数，将给定消息输出为带框线的格式
def print_box(msg):
    # 拆分消息为多行，并计算消息中最长行的长度
    lines = msg.split("\n")
    size = max(len(l) + 1 for l in lines)
    # 打印顶部边框
    print("-" * (size + 2))
    # 打印消息内容，每行消息左右用竖线包围，使得每行长度一致
    for l in lines:
        print("|{}{}|".format(l, " " * (size - len(l))))
    # 打印底部边框
    print("-" * (size + 2))


# 主函数，程序入口
def main():
    # 如果同时设置了构建LibTorch Wheel和仅构建Python部分，抛出运行时错误
    if BUILD_LIBTORCH_WHL and BUILD_PYTHON_ONLY:
        raise RuntimeError(
            "Conflict: 'BUILD_LIBTORCH_WHL' and 'BUILD_PYTHON_ONLY' can't both be 1. Set one to 0 and rerun."
        )
    
    # 定义需要安装的Python依赖列表
    install_requires = [
        "filelock",
        "typing-extensions>=4.8.0",
        "sympy",
        "networkx",
        "jinja2",
        "fsspec",
    ]

    # 如果Python版本大于等于3.12.0，则添加setuptools到依赖列表中
    if sys.version_info >= (3, 12, 0):
        install_requires.append("setuptools")

    # 如果仅构建Python部分，添加特定版本的LibTorch到依赖列表中
    if BUILD_PYTHON_ONLY:
        install_requires.append(f"{LIBTORCH_PKG_NAME}=={get_torch_version()}")

    # 检查是否需要使用优先文本链接器脚本，根据条件打印警告消息
    use_prioritized_text = str(os.getenv("USE_PRIORITIZED_TEXT_FOR_LD", ""))
    if (
        use_prioritized_text == ""
        and platform.system() == "Linux"
        and platform.processor() == "aarch64"
    ):
        print_box(
            """
            WARNING: we strongly recommend enabling linker script optimization for ARM + CUDA.
            To do so please export USE_PRIORITIZED_TEXT_FOR_LD=1
            """
        )
    
    # 如果设置了使用优先文本链接器脚本，生成链接器脚本文件并设置环境变量
    if use_prioritized_text == "1" or use_prioritized_text == "True":
        gen_linker_script(
            filein="cmake/prioritized_text.txt", fout="cmake/linker_script.ld"
        )
        linker_script_path = os.path.abspath("cmake/linker_script.ld")
        os.environ["LDFLAGS"] = os.getenv("LDFLAGS", "") + f" -T{linker_script_path}"
        os.environ["CFLAGS"] = (
            os.getenv("CFLAGS", "") + " -ffunction-sections -fdata-sections"
        )
        os.environ["CXXFLAGS"] = (
            os.getenv("CXXFLAGS", "") + " -ffunction-sections -fdata-sections"
        )

    # 创建Distribution对象实例，设置脚本名称和命令行参数
    dist = Distribution()
    dist.script_name = os.path.basename(sys.argv[0])
    dist.script_args = sys.argv[1:]
    
    # 解析命令行参数，处理任何可能的错误
    try:
        dist.parse_command_line()
    except setuptools.distutils.errors.DistutilsArgError as e:
        print(e)
        sys.exit(1)

    # 将文件镜像到torchgen目录
    mirror_files_into_torchgen()

    # 如果设置了运行构建依赖，则执行构建依赖的操作
    if RUN_BUILD_DEPS:
        build_deps()

    # 调用configure_extension_build函数配置扩展构建所需的各种属性和依赖项
    (
        extensions,
        cmdclass,
        packages,
        entry_points,
        extra_install_requires,
    ) = configure_extension_build()

    # 将额外的安装依赖项添加到install_requires列表中
    install_requires += extra_install_requires

    # 定义额外的可选依赖字典
    extras_require = {
        "optree": ["optree>=0.11.0"],
        "opt-einsum": ["opt-einsum>=3.3"],
    }
    # 读取 README.md 文件作为长描述信息
    with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    # 确定版本范围的上限，将当前 Python 版本的第二部分与12进行比较并加1
    version_range_max = max(sys.version_info[1], 12) + 1
    # 如果不构建 LIBTORCH_WHL，则添加以下 Torch 包数据
    if not BUILD_LIBTORCH_WHL:
        torch_package_data.extend(
            [
                "lib/libtorch_python.so",
                "lib/libtorch_python.dylib",
                "lib/libtorch_python.dll",
            ]
        )
    # 如果不仅构建 Python，则扩展 Torch 包数据
    if not BUILD_PYTHON_ONLY:
        torch_package_data.extend(
            [
                "lib/*.so*",
                "lib/*.dylib*",
                "lib/*.dll",
                "lib/*.lib",
            ]
        )
    # 如果构建 CAFFE2，则扩展 Torch 包数据
    if get_cmake_cache_vars()["BUILD_CAFFE2"]:
        torch_package_data.extend(
            [
                "include/caffe2/**/*.h",
                "include/caffe2/utils/*.h",
                "include/caffe2/utils/**/*.h",
            ]
        )
    # 如果使用 TENSORPIPE，则扩展 Torch 包数据
    if get_cmake_cache_vars()["USE_TENSORPIPE"]:
        torch_package_data.extend(
            [
                "include/tensorpipe/*.h",
                "include/tensorpipe/channel/*.h",
                "include/tensorpipe/channel/basic/*.h",
                "include/tensorpipe/channel/cma/*.h",
                "include/tensorpipe/channel/mpt/*.h",
                "include/tensorpipe/channel/xth/*.h",
                "include/tensorpipe/common/*.h",
                "include/tensorpipe/core/*.h",
                "include/tensorpipe/transport/*.h",
                "include/tensorpipe/transport/ibv/*.h",
                "include/tensorpipe/transport/shm/*.h",
                "include/tensorpipe/transport/uv/*.h",
            ]
        )
    # 如果使用 KINETO，则扩展 Torch 包数据
    if get_cmake_cache_vars()["USE_KINETO"]:
        torch_package_data.extend(
            [
                "include/kineto/*.h",
            ]
        )
    # 设置 Torchgen 包数据，递归通配符在 setup.py 中不起作用
    torchgen_package_data = [
        # 递归通配符在 setup.py 中不起作用，详见 https://github.com/pypa/setuptools/issues/1806
        "packaged/ATen/*",
        "packaged/ATen/native/*",
        "packaged/ATen/templates/*",
        "packaged/autograd/*",
        "packaged/autograd/templates/*",
    ]
    # 定义包数据，主要为 "torch" 的包数据
    package_data = {
        "torch": torch_package_data,
    }

    # 如果不构建 LIBTORCH_WHL，则添加额外的 package_data 条目
    if not BUILD_LIBTORCH_WHL:
        package_data["torchgen"] = torchgen_package_data
        package_data["caffe2"] = [
            "python/serialized_test/data/operator_test/*.zip",
        ]
    else:
        # 在 BUILD_LIBTORCH_WHL 模式下没有扩展
        extensions = []
    setup(
        name=package_name,  # 设置包的名称
        version=version,  # 设置包的版本号
        description=(
            "Tensors and Dynamic neural networks in "
            "Python with strong GPU acceleration"  # 设置包的简短描述
        ),
        long_description=long_description,  # 设置包的详细描述
        long_description_content_type="text/markdown",  # 描述内容的类型为 Markdown 格式
        ext_modules=extensions,  # 设置扩展模块
        cmdclass=cmdclass,  # 设置命令类
        packages=packages,  # 设置包含的 Python 包
        entry_points=entry_points,  # 设置入口点
        install_requires=install_requires,  # 设置安装依赖
        extras_require=extras_require,  # 设置额外的依赖
        package_data=package_data,  # 设置包数据
        url="https://pytorch.org/",  # 设置包的官方网址
        download_url="https://github.com/pytorch/pytorch/tags",  # 设置包的下载链接
        author="PyTorch Team",  # 设置作者
        author_email="packages@pytorch.org",  # 设置作者邮箱
        python_requires=f">={python_min_version_str}",  # 设置所需的 Python 最低版本
        # 设置 PyPI 包的分类信息
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3",
        ]
        + [
            f"Programming Language :: Python :: 3.{i}"
            for i in range(python_min_version[1], version_range_max)
        ],  # 添加 Python 版本的分类信息
        license="BSD-3",  # 设置许可证类型
        keywords="pytorch, machine learning",  # 设置关键词
    )
    if EMIT_BUILD_WARNING:  # 如果需要发出构建警告
        print_box(build_update_message)  # 打印构建更新信息的盒子
# 如果当前脚本被作为主程序执行（而不是被导入到其他模块中执行），则执行 main() 函数
if __name__ == "__main__":
    main()
```