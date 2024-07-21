# `.\pytorch\tools\amd_build\build_amd.py`

```
#!/usr/bin/env python3
# 指定脚本的解释器为 Python 3

import argparse
import os
import sys

# 将当前脚本的父级目录添加到系统路径中，以便导入 hipify 模块
sys.path.append(
    os.path.realpath(
        os.path.join(
            __file__, os.path.pardir, os.path.pardir, os.path.pardir, "torch", "utils"
        )
    )
)

from hipify import hipify_python  # type: ignore[import]

# 创建命令行参数解析对象
parser = argparse.ArgumentParser(
    description="Top-level script for HIPifying, filling in most common parameters"
)

# 添加命令行参数：是否仅对源文件运行 hipify，采用 out-of-place 方式
parser.add_argument(
    "--out-of-place-only",
    action="store_true",
    help="Whether to only run hipify out-of-place on source files",
)

# 添加命令行参数：项目的根目录路径
parser.add_argument(
    "--project-directory",
    type=str,
    default="",
    help="The root of the project.",
    required=False,
)

# 添加命令行参数：hipify 后的项目存储目录路径
parser.add_argument(
    "--output-directory",
    type=str,
    default="",
    help="The directory to store the hipified project",
    required=False,
)

# 添加命令行参数：额外的 caffe2 目录列表，用于 hipify
parser.add_argument(
    "--extra-include-dir",
    type=str,
    default=[],
    nargs="+",
    help="The list of extra directories in caffe2 to hipify",
    required=False,
)

# 解析命令行参数
args = parser.parse_args()

# 获取当前脚本所在的目录路径
amd_build_dir = os.path.dirname(os.path.realpath(__file__))

# 获取项目的根目录路径，默认为当前脚本所在目录的父级目录
proj_dir = os.path.join(os.path.dirname(os.path.dirname(amd_build_dir)))

# 如果命令行参数中提供了项目的根目录路径，则使用命令行参数中的路径
if args.project_directory:
    proj_dir = args.project_directory

# 输出目录默认为项目的根目录路径
out_dir = proj_dir

# 如果命令行参数中提供了 hipify 后的项目存储目录路径，则使用命令行参数中的路径
if args.output_directory:
    out_dir = args.output_directory

# hipify 的包含路径列表
includes = [
    "caffe2/operators/*",
    "caffe2/sgd/*",
    "caffe2/image/*",
    "caffe2/transforms/*",
    "caffe2/video/*",
    "caffe2/distributed/*",
    "caffe2/queue/*",
    "caffe2/contrib/aten/*",
    "binaries/*",
    "caffe2/**/*_test*",
    "caffe2/core/*",
    "caffe2/db/*",
    "caffe2/utils/*",
    "caffe2/contrib/gloo/*",
    "caffe2/contrib/nccl/*",
    "c10/cuda/*",
    "c10/cuda/test/CMakeLists.txt",
    "modules/*",
    "third_party/nvfuser/*",
    # PyTorch paths
    # Keep this synchronized with is_pytorch_file in hipify_python.py
    "aten/src/ATen/cuda/*",
    "aten/src/ATen/native/cuda/*",
    "aten/src/ATen/native/cudnn/*",
    "aten/src/ATen/native/quantized/cudnn/*",
    "aten/src/ATen/native/nested/cuda/*",
    "aten/src/ATen/native/sparse/cuda/*",
    "aten/src/ATen/native/quantized/cuda/*",
    "aten/src/ATen/native/transformers/cuda/attention_backward.cu",
    "aten/src/ATen/native/transformers/cuda/attention.cu",
    "aten/src/ATen/native/transformers/cuda/sdp_utils.cpp",
    "aten/src/ATen/native/transformers/cuda/sdp_utils.h",
    "aten/src/ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h",
    "aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h",
    "aten/src/ATen/native/transformers/cuda/mem_eff_attention/pytorch_utils.h",
    "aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.h",
    "aten/src/THC/*",
    "aten/src/ATen/test/*",
    # CMakeLists.txt isn't processed by default, but there are a few
    # we do want to handle, so explicitly specify them
    "aten/src/THC/CMakeLists.txt",
    "torch/*",
    # 文件路径字符串，指定了要打开的文件位置和名称
    "tools/autograd/templates/python_variable_methods.cpp",
# 拼接包含文件的绝对路径列表，使用列表推导式遍历 includes 列表
includes = [os.path.join(proj_dir, include) for include in includes]

# 遍历额外的包含目录列表 args.extra_include_dir
for new_dir in args.extra_include_dir:
    # 拼接新目录的绝对路径
    abs_new_dir = os.path.join(proj_dir, new_dir)
    # 如果新目录存在，则进一步拼接路径并加入 includes 列表
    if os.path.exists(abs_new_dir):
        abs_new_dir = os.path.join(abs_new_dir, "**/*")
        includes.append(abs_new_dir)

# 忽略文件列表
ignores = [
    "caffe2/operators/depthwise_3x3_conv_op_cudnn.cu",  # 忽略的 CUDA 文件
    "caffe2/operators/pool_op_cudnn.cu",  # 忽略的 CUDA 文件
    "*/hip/*",  # 忽略的 HIP 相关文件和目录
    # 以下文件是 CUDA 和 HIP 兼容的
    "aten/src/ATen/core/*",  # 忽略的路径模式
    # 生成 HIPConfig.h 的正确路径：
    #   CUDAConfig.h.in -> (amd_build) HIPConfig.h.in -> (cmake) HIPConfig.h
    "aten/src/ATen/cuda/CUDAConfig.h",  # 忽略的 CUDA 文件
    "third_party/nvfuser/csrc/codegen.cpp",  # 忽略的文件
    "third_party/nvfuser/runtime/block_reduction.cu",  # 忽略的 CUDA 文件
    "third_party/nvfuser/runtime/block_sync_atomic.cu",  # 忽略的 CUDA 文件
    "third_party/nvfuser/runtime/block_sync_default_rocm.cu",  # 忽略的 CUDA 文件
    "third_party/nvfuser/runtime/broadcast.cu",  # 忽略的 CUDA 文件
    "third_party/nvfuser/runtime/grid_reduction.cu",  # 忽略的 CUDA 文件
    "third_party/nvfuser/runtime/helpers.cu",  # 忽略的 CUDA 文件
    "torch/csrc/jit/codegen/fuser/cuda/resource_strings.h",  # 忽略的文件
    "torch/csrc/jit/tensorexpr/ir_printer.cpp",  # 忽略的文件
    # 不应该进行更改的生成文件
    "torch/lib/tmp_install/*",  # 忽略的临时安装目录
    "torch/include/*",  # 忽略的 include 目录
]

# 将忽略文件列表转换为绝对路径形式
ignores = [os.path.join(proj_dir, ignore) for ignore in ignores]

# 检查编译器是否为 hip-clang
def is_hip_clang() -> bool:
    try:
        hip_path = os.getenv("HIP_PATH", "/opt/rocm/hip")
        # 打开 hip 的配置文件 .hipInfo，检查是否包含 "HIP_COMPILER=clang"
        with open(hip_path + "/lib/.hipInfo") as f:
            return "HIP_COMPILER=clang" in f.read()
    except OSError:
        return False


# TODO 待移除，一旦以下子模块更新完成
# 列出需要更新的 HIP 平台文件列表
hip_platform_files = [
    "third_party/fbgemm/fbgemm_gpu/CMakeLists.txt",
    "third_party/fbgemm/fbgemm_gpu/cmake/Hip.cmake",
    "third_party/fbgemm/fbgemm_gpu/codegen/embedding_backward_dense_host.cpp",
    "third_party/fbgemm/fbgemm_gpu/codegen/embedding_backward_split_host_template.cpp",
    "third_party/fbgemm/fbgemm_gpu/codegen/embedding_backward_split_template.cu",
    "third_party/fbgemm/fbgemm_gpu/codegen/embedding_forward_quantized_split_lookup.cu",
    "third_party/fbgemm/fbgemm_gpu/include/fbgemm_gpu/fbgemm_cuda_utils.cuh",
    "third_party/fbgemm/fbgemm_gpu/include/fbgemm_gpu/sparse_ops.cuh",
    "third_party/fbgemm/fbgemm_gpu/src/jagged_tensor_ops.cu",
    "third_party/fbgemm/fbgemm_gpu/src/quantize_ops.cu",
    "third_party/fbgemm/fbgemm_gpu/src/sparse_ops.cu",
    "third_party/fbgemm/fbgemm_gpu/src/split_embeddings_cache_cuda.cu",
    "third_party/fbgemm/fbgemm_gpu/src/topology_utils.cpp",
    "third_party/fbgemm/src/EmbeddingSpMDM.cc",
    "third_party/gloo/cmake/Dependencies.cmake",
    "third_party/gloo/gloo/cuda.cu",
    "third_party/kineto/libkineto/CMakeLists.txt",
    "third_party/nvfuser/CMakeLists.txt",
    "third_party/tensorpipe/cmake/Hip.cmake",
]

# 移除行中的 HCC 相关标识，替换为 AMD 平台和 clang 相关标识
def remove_hcc(line: str) -> str:
    line = line.replace("HIP_PLATFORM_HCC", "HIP_PLATFORM_AMD")
    line = line.replace("HIP_HCC_FLAGS", "HIP_CLANG_FLAGS")
    return line


注释：


    # 返回函数中的变量 line，这里是函数的返回语句
    # 在调用该函数时，返回 line 变量的值
    return line
for hip`
# 遍历 hip_platform_files 列表中的每个文件路径
for hip_platform_file in hip_platform_files:
    # 初始化一个标志，用于指示是否需要写入文件，默认为 False
    do_write = False
    # 检查当前文件路径是否存在
    if os.path.exists(hip_platform_file):
        # 打开文件并读取所有行内容
        with open(hip_platform_file) as sources:
            lines = sources.readlines()
        # 对每一行应用 remove_hcc 函数，生成新的行列表
        newlines = [remove_hcc(line) for line in lines]
        # 检查原始行列表和新行列表是否相同
        if lines == newlines:
            # 如果相同，表示文件内容未改变，输出文件被跳过的消息
            print(f"{hip_platform_file} skipped")
        else:
            # 如果不相同，将新的行列表写回到文件中
            with open(hip_platform_file, "w") as sources:
                for line in newlines:
                    sources.write(line)
            # 输出文件已更新的消息
            print(f"{hip_platform_file} updated")

# 调用 hipify_python.hipify 函数进行代码转换操作
hipify_python.hipify(
    project_directory=proj_dir,  # 指定项目目录
    output_directory=out_dir,     # 指定输出目录
    includes=includes,            # 指定需要转换的文件或目录列表
    ignores=ignores,              # 指定需要忽略的文件或目录列表
    extra_files=[
        "torch/_inductor/codegen/cpp_wrapper_cpu.py",   # 需要额外处理的 CPU 版本的文件
        "torch/_inductor/codegen/cpp_wrapper_cuda.py",  # 需要额外处理的 CUDA 版本的文件
        "torch/_inductor/codegen/wrapper.py",           # 需要额外处理的通用版本文件
    ],
    out_of_place_only=args.out_of_place_only,  # 根据命令行参数指定只进行 out-of-place 转换
    hip_clang_launch=is_hip_clang(),            # 检查是否使用 HIP clang 进行编译
)
```