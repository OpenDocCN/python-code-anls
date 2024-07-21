# `.\pytorch\torch\_inductor\codegen\rocm\compile_command.py`

```py
# mypy: allow-untyped-defs
# 导入日志模块和操作系统模块
import logging
import os
# 导入类型提示模块
from typing import List, Optional

# 导入 Torch 混合编程的配置模块
from torch._inductor import config
# 导入 Torch 混合编程的工具函数
from torch._inductor.utils import is_linux

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


def _rocm_include_paths() -> List[str]:
    # 导入 Torch 的 C++ 扩展模块
    from torch.utils import cpp_extension

    # 根据配置确定 ROCm 的头文件路径
    rocm_include = (
        os.path.join(config.rocm.rocm_home, "include")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("include")
    )
    # 如果未指定 Composable Kernel 的包含目录，则发出警告
    if not config.rocm.ck_dir:
        log.warning("Unspecified Composable Kernel include dir")
    # 获取 Composable Kernel 的包含目录路径
    ck_include = os.path.join(
        config.rocm.ck_dir or cpp_extension._join_rocm_home("composable_kernel"),
        "include",
    )
    # 返回 ROCm 和 Composable Kernel 的包含目录路径列表
    return [os.path.realpath(rocm_include), os.path.realpath(ck_include)]


def _rocm_lib_options() -> List[str]:
    # 导入 Torch 的 C++ 扩展模块
    from torch.utils import cpp_extension

    # 根据配置确定 ROCm 的库文件路径
    rocm_lib_dir = (
        os.path.join(config.rocm.rocm_home, "lib")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("lib")
    )
    # 根据配置确定 HIP 的库文件路径
    hip_lib_dir = (
        os.path.join(config.rocm.rocm_home, "hip", "lib")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("hip", "lib")
    )

    # 返回 ROCm 和 HIP 的库选项列表
    return [
        f"-L{os.path.realpath(rocm_lib_dir)}",
        f"-L{os.path.realpath(hip_lib_dir)}",
        "-lamdhip64",
    ]


def _rocm_compiler_options() -> List[str]:
    # 获取 ROCm 的架构列表，若未指定则默认为本地架构
    arch_list = config.rocm.arch or ["native"]
    # 生成针对 GPU 架构的编译选项
    gpu_arch_flags = [f"--offload-arch={arch}" for arch in arch_list]
    # 生成编译选项列表
    opts = [
        config.rocm.compile_opt_level,
        "-x",
        "hip",
        "-std=c++17",
        *gpu_arch_flags,
        "-fno-gpu-rdc",
        "-fPIC",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-mllvm",
        "-amdgpu-function-calls=false",
        "-mllvm",
        "-enable-post-misched=0",
    ]
    # 若开启调试模式，则添加调试相关选项
    if config.rocm.is_debug:
        opts += ["-DDEBUG_LOG=1", "-g"]
    # 若需要保存临时文件，则添加相应选项
    if config.rocm.save_temps:
        opts += ["--save-temps=obj"]
    # 若需要打印内核资源使用情况，则添加相应选项
    if config.rocm.print_kernel_resource_usage:
        opts += ["-Rpass-analysis=kernel-resource-usage"]
    # 若需要将 denormalized 浮点数规范化为零，则添加相应选项
    if config.rocm.flush_denormals:
        opts += ["-fgpu-flush-denormals-to-zero"]
    # 若开启快速数学计算，则添加相应选项
    if config.rocm.use_fast_math:
        opts += ["-ffast-math"]
    # 返回最终的编译选项列表
    return opts


def rocm_compiler() -> Optional[str]:
    # 若当前系统为 Linux
    if is_linux():
        # 若配置中指定了 ROCm 的安装路径，则返回其对应的编译器路径
        if config.rocm.rocm_home:
            return os.path.realpath(
                os.path.join(config.rocm.rocm_home, "llvm", "bin", "clang")
            )
        # 否则尝试使用 Torch 的 C++ 扩展模块中默认的 ROCm 安装路径
        try:
            from torch.utils import cpp_extension

            return os.path.realpath(
                cpp_extension._join_rocm_home("llvm", "bin", "clang")
            )
        # 若以上尝试失败，则返回默认的 clang 编译器路径
        except OSError:
            # 若配置中未设置 ROCm 安装路径且环境变量 ROCM_HOME 也未设置，则返回默认的 clang 编译器路径
            return "clang"
    # 若当前系统非 Linux，则返回空值
    return None


def rocm_compile_command(
    src_files: List[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: Optional[List[str]] = None,
) -> str:
    # 获取 ROCm 的包含路径列表
    include_paths = _rocm_include_paths()
    # 获取 ROCm 库的选项
    lib_options = _rocm_lib_options()
    # 获取 ROCm 编译器的选项
    compiler_options = _rocm_compiler_options()
    # 获取 ROCm 编译器路径
    compiler = rocm_compiler()
    # 构建编译选项，包括编译器选项、额外参数（如果有）、包含路径、库选项
    options = (
        compiler_options
        + (extra_args if extra_args else [])  # 加入额外的参数，如果有的话
        + ["-I" + path for path in include_paths]  # 将每个包含路径转换为 "-I<path>" 的形式
        + lib_options  # 加入 ROCm 库的选项
    )
    # 将源文件列表连接成一个字符串，用空格分隔
    src_file = " ".join(src_files)
    # 初始化结果字符串
    res = ""
    # 根据目标文件扩展名不同，构建不同的编译命令
    if dst_file_ext == "o":
        res = f"{compiler} {' '.join(options)} -c -o {dst_file} {src_file}"
    elif dst_file_ext == "so":
        options.append("-shared")  # 如果是共享库，添加 "-shared" 选项
        res = f"{compiler} {' '.join(options)} -o {dst_file} {src_file}"
    elif dst_file_ext == "exe":
        res = f"{compiler} {' '.join(options)} -o {dst_file} {src_file}"
    else:
        # 如果不支持的目标文件后缀，抛出 NotImplementedError 异常
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")
    # 返回构建好的编译命令字符串
    return res
```