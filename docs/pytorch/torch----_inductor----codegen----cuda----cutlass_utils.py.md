# `.\pytorch\torch\_inductor\codegen\cuda\cutlass_utils.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和库
import functools  # 导入 functools 模块
import logging  # 导入 logging 模块
import os  # 导入 os 模块
import sys  # 导入 sys 模块

from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from pathlib import Path  # 从 pathlib 模块导入 Path 类
from typing import Any, List, Optional  # 导入类型提示相关的类型

import sympy  # 导入 sympy 库

import torch  # 导入 torch 库
from ... import config  # 从相对路径导入 config 模块
from ...ir import Layout  # 从相对路径导入 Layout 类

from ...runtime.runtime_utils import cache_dir  # 从相对路径导入 cache_dir 函数
from .cuda_env import get_cuda_arch, get_cuda_version  # 从当前目录导入 get_cuda_arch 和 get_cuda_version 函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def _rename_cutlass_import(content: str, cutlass_modules: List[str]) -> str:
    # 将指定的 Cutlass 模块重命名为 cutlass_library 下的模块
    for cutlass_module in cutlass_modules:
        content = content.replace(
            f"from {cutlass_module} import ",
            f"from cutlass_library.{cutlass_module} import ",
        )
    return content


def _gen_cutlass_file(
    file_name: str, cutlass_modules: List[str], src_dir: str, dst_dir: str
) -> None:
    # 生成修改后的 Cutlass 文件到目标目录
    orig_full_path = os.path.abspath(os.path.join(src_dir, file_name))  # 获取源文件的绝对路径
    text = ""
    with open(orig_full_path) as f:
        text = f.read()  # 读取源文件内容
    text = _rename_cutlass_import(text, cutlass_modules)  # 替换源文件中的 Cutlass 模块导入
    dst_full_path = os.path.abspath(
        os.path.join(
            dst_dir,
            file_name,
        )
    )
    with open(dst_full_path, "w") as f:
        f.write(text)  # 将修改后的内容写入目标文件


@functools.lru_cache(None)
def try_import_cutlass() -> bool:
    if config.is_fbcode():  # 如果运行环境是在 Facebook 代码库中
        return True  # 直接返回 True

    # 将 CUTLASS Python 脚本复制到临时目录，并将该目录添加到 Python 搜索路径中。
    # 这是一个临时解决方案，用于避免 CUTLASS 模块命名冲突问题。
    # TODO(ipiszy): 当 CUTLASS 解决 Python 脚本打包结构问题时，移除此临时解决方案。

    cutlass_py_full_path = os.path.abspath(
        os.path.join(config.cuda.cutlass_dir, "python/cutlass_library")
    )  # 获取 CUTLASS Python 脚本的绝对路径
    tmp_cutlass_py_full_path = os.path.abspath(
        os.path.join(cache_dir(), "torch_cutlass_library")
    )  # 获取临时目录的绝对路径
    dst_link = os.path.join(tmp_cutlass_py_full_path, "cutlass_library")  # 目标链接路径

    if os.path.isdir(cutlass_py_full_path):  # 如果 CUTLASS Python 脚本目录存在
        if tmp_cutlass_py_full_path not in sys.path:  # 如果临时目录不在 Python 搜索路径中
            if os.path.exists(dst_link):  # 如果目标链接已经存在
                assert os.path.islink(
                    dst_link
                ), f"{dst_link} is not a symlink. Try to remove {dst_link} manually and try again."  # 断言目标链接是符号链接
                assert os.path.realpath(os.readlink(dst_link)) == os.path.realpath(
                    cutlass_py_full_path
                ), f"Symlink at {dst_link} does not point to {cutlass_py_full_path}"  # 断言链接的目标路径正确
            else:
                os.makedirs(tmp_cutlass_py_full_path, exist_ok=True)  # 创建临时目录
                os.symlink(cutlass_py_full_path, dst_link)  # 创建符号链接
            sys.path.append(tmp_cutlass_py_full_path)  # 将临时目录添加到 Python 搜索路径中
        try:
            import cutlass_library.generator  # 尝试导入 Cutlass 库的相关模块
            import cutlass_library.library
            import cutlass_library.manifest

            return True  # 导入成功，返回 True

        except ImportError as e:  # 如果导入失败
            log.debug(
                "Failed to import CUTLASS packages: %s, ignoring the CUTLASS backend.",
                str(e),
            )  # 记录调试信息，表示未能导入 CUTLASS 包，忽略 CUTLASS 后端
    else:
        # 如果导入 CUTLASS 包失败，则记录调试信息，指明 CUTLASS 仓库不存在的完整路径
        log.debug(
            "Failed to import CUTLASS packages: CUTLASS repo does not exist: %s",
            cutlass_py_full_path,
        )
    # 返回 False 表示导入失败
    return False
# 根据输入的 CUDA 架构版本号进行规范化处理，并返回规范化后的版本号字符串
def _normalize_cuda_arch(arch: str) -> str:
    if int(arch) >= 90:
        return "90"
    elif int(arch) >= 80:
        return "80"
    elif int(arch) >= 75:
        return "75"
    elif int(arch) >= 70:
        return "70"
    else:
        # 如果架构版本不在支持列表中，则抛出未实现错误
        raise NotImplementedError(f"Unsupported cuda arch: {arch}")


@dataclass
class CUTLASSArgs:
    """
    CUTLASS args used to initialize a CUTLASS Manifest.
    """

    architectures: Optional[str] = None  # 可选参数，CUDA 架构版本号字符串
    cuda_version: Optional[str] = None  # 可选参数，CUDA 版本号字符串

    operations = "all"  # 默认值为 "all"，表示所有操作
    build_dir = ""  # 默认空字符串，构建目录路径
    curr_build_dir = ""  # 默认空字符串，当前构建目录路径
    generator_target = ""  # 默认空字符串，生成器目标
    kernels = "all"  # 默认值为 "all"，表示所有内核
    ignore_kernels = ""  # 默认空字符串，忽略的内核列表
    # TODO: these three look dead?
    kernel_filter_file: None = None  # 默认为 None，内核过滤文件
    selected_kernel_list: None = None  # 默认为 None，选定的内核列表
    interface_dir: None = None  # 默认为 None，接口目录
    filter_by_cc = True  # 默认为 True，根据计算能力版本过滤
    disable_full_archs_compilation = False  # 默认为 False，禁止完整架构编译

    def __post_init__(self):
        # 在初始化后检查架构和CUDA版本是否为None，若是则引发运行时错误
        if self.architectures is None or self.cuda_version is None:
            raise RuntimeError(
                f"{self.architectures=} or {self.cuda_version=} is None!"
            )
        self.architectures = _normalize_cuda_arch(self.architectures)  # 规范化 CUDA 架构版本号


@functools.lru_cache(None)
def _gen_ops_cached(arch, version) -> List[Any]:
    # Note: Cache needs to be specific for cuda architecture and version
    # 缓存需基于特定的 CUDA 架构和版本

    # 导入 cutlass Python 脚本。
    assert try_import_cutlass()  # 断言确保 cutlass 库已成功导入
    import cutlass_library.generator as cutlass_generator
    import cutlass_library.manifest as cutlass_manifest

    if arch is None or version is None:
        # 如果未检测到CUDA架构或版本，则记录错误并返回空列表
        log.error(
            "Cannot detect cuda arch %s or cuda version %s. "
            "Will discard all cutlass ops. "
            "Please consider setting _inductor.cuda.arch and _inductor.cuda.version configs.",
            arch,
            version,
        )
        return list()
    arch = _normalize_cuda_arch(arch)  # 规范化 CUDA 架构版本号
    args = CUTLASSArgs(architectures=arch, cuda_version=version)
    manifest = cutlass_manifest.Manifest(args)  # 创建 CUTLASS Manifest 对象

    if arch == "90":
        cutlass_generator.GenerateSM90(manifest, args.cuda_version)  # 生成 SM90 架构的操作
        cutlass_generator.GenerateSM80(manifest, args.cuda_version)  # 生成 SM80 架构的操作
    else:
        try:
            func = getattr(cutlass_generator, "GenerateSM" + arch)
            func(manifest, args.cuda_version)  # 根据架构生成对应的操作
        except AttributeError as e:
            # 如果指定的架构不支持，则引发未实现错误
            raise NotImplementedError(
                "Arch " + arch + " is not supported by current cutlass lib."
            ) from e
    return manifest.operations  # 返回生成的操作列表


def gen_ops() -> List[Any]:
    """
    Generates all supported CUTLASS operations.
    """
    arch = get_cuda_arch()  # 获取 CUDA 架构版本号
    version = get_cuda_version()  # 获取 CUDA 版本号
    return _gen_ops_cached(arch, version)  # 调用缓存生成操作的函数并返回结果


def torch_dtype_to_cutlass_type(
    torch_dtype: torch.dtype,
) -> "cutlass_library.library.DataType":  # type: ignore[name-defined] # noqa: F821
    # 导入 cutlass Python 脚本。
    assert try_import_cutlass()  # 断言确保 cutlass 库已成功导入
    import cutlass_library  # type: ignore[import]

    if torch_dtype == torch.float:
        return cutlass_library.library.DataType.f32  # 如果输入类型是 float，则返回 CUTLASS 中对应的数据类型
    # 如果 Torch 数据类型是半精度（torch.half），返回 Cutlass 库中对应的 f16 数据类型
    elif torch_dtype == torch.half:
        return cutlass_library.library.DataType.f16
    # 如果 Torch 数据类型是 bfloat16，返回 Cutlass 库中对应的 bf16 数据类型
    elif torch_dtype == torch.bfloat16:
        return cutlass_library.library.DataType.bf16
    # 如果 Torch 数据类型不是上述两种情况之一，则抛出 NotImplementedError 异常
    else:
        raise NotImplementedError(f"Unsupported data type: {torch_dtype=}")
# 定义一个函数，用于比较 Torch 数据类型和 Cutlass 数据类型是否匹配
def dtype_match(
    torch_dtype: Optional[torch.dtype],
    cutlass_dtype: "cutlass_library.library.DataType",  # type: ignore[name-defined]  # noqa: F821
) -> bool:
    # 引入 Cutlass 的 Python 脚本
    assert try_import_cutlass()
    import cutlass_library

    # 根据 Torch 数据类型进行匹配判断
    if torch_dtype == torch.float:
        return (
            cutlass_dtype == cutlass_library.library.DataType.f32
            or cutlass_dtype == cutlass_library.library.DataType.tf32
        )
    elif torch_dtype == torch.half:
        return cutlass_dtype == cutlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_dtype == cutlass_library.library.DataType.bf16
    elif torch_dtype == torch.int8:
        return cutlass_dtype == cutlass_library.library.DataType.s8
    elif torch_dtype == torch.uint8:
        return cutlass_dtype == cutlass_library.library.DataType.u8
    elif torch_dtype == torch.int32:
        return cutlass_dtype == cutlass_library.library.DataType.s32
    else:
        return False


# 定义一个函数，根据输入的 Torch 数据类型列表，推断并返回累加器的 Torch 数据类型
def get_accumulator_dtype(
    input_torch_dtypes: List[torch.dtype],
) -> Optional[torch.dtype]:
    """
    给定一对输入的 Torch 数据类型，返回推断出的累加器 Torch 数据类型。
    """

    # 如果输入的 Torch 数据类型列表长度不为2，则返回 None
    if len(input_torch_dtypes) != 2:
        return None

    torch_dtype = None
    # 如果两个输入的 Torch 数据类型相同，则取第一个作为累加器类型
    if input_torch_dtypes[0] == input_torch_dtypes[1]:
        torch_dtype = input_torch_dtypes[0]
    else:
        # 根据元素大小判断，选择较大的类型作为主数据类型
        size0 = torch.tensor([], dtype=input_torch_dtypes[0]).element_size()
        size1 = torch.tensor([], dtype=input_torch_dtypes[1]).element_size()
        if size0 > size1:
            dtype0, dtype1 = input_torch_dtypes
        else:
            dtype1, dtype0 = input_torch_dtypes
        # 特定的数据类型组合可以选择较小的精度作为累加器类型
        if dtype0 in [torch.half, torch.bfloat16] and dtype1 in [
            torch.int8,
            torch.uint8,
        ]:
            torch_dtype = dtype0

    # 根据累加器类型进行进一步判断和返回
    if torch_dtype == torch.half:
        if torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction:
            return torch_dtype
        else:
            return torch.float
    if torch_dtype in {torch.bfloat16, torch.float}:
        return torch.float
    if torch_dtype == torch.int8:
        return torch.int32
    raise NotImplementedError(f"Unsupported data types: {input_torch_dtypes=}")


# 定义一个函数，返回给定 Torch 数据类型的所有可能有效的 CUTLASS 对齐方式列表
def get_alignments(torch_dtype: torch.dtype) -> List[int]:
    """
    返回给定数据类型的所有可能有效的 CUTLASS 对齐方式列表，以元素数量计算。
    CUTLASS gemm / conv SM80 APIs 支持最大 16 字节对齐，最小 2 字节对齐。
    """

    # 根据 Torch 数据类型进行不同的对齐方式选择和返回
    if torch_dtype in (torch.half, torch.bfloat16):
        return [8, 4, 2, 1]
    elif torch_dtype == torch.float:
        return [4, 2, 1]
    elif torch_dtype in (torch.uint8, torch.int8):
        return [16, 8, 4, 2]
    elif torch_dtype == torch.int32:
        return [4, 2, 1]
    else:
        raise NotImplementedError(f"unsupported {torch_dtype=} for alignments")


# 定义一个函数，返回给定 Inductor 布局的最大对齐方式
def get_max_alignment(inductor_layout: Layout) -> int:
    """
    返回给定 Inductor 布局的最大对齐方式。
    """
    # 返回给定电感器布局的最大对齐方式（以元素数量表示）。

    # 获取电感器布局的数据类型
    dtype = inductor_layout.dtype
    # 获取电感器布局的总元素数
    size = inductor_layout.size
    # 获取电感器布局的偏移量

    offset = inductor_layout.offset

    # 定义函数，用于检查一个数是否为静态整数（int 或 sympy.Integer 类型）
    def is_static_int(number):
        return isinstance(number, (int, sympy.Integer))

    try:
        # 查找第一个步长为1的维度
        contiguous_dim = inductor_layout.stride.index(1)
    except ValueError:
        # 如果没有步长为1的维度，返回默认对齐方式1
        return 1

    # 检查是否所有相关参数都是静态整数，且满足对齐条件
    if (
        is_static_int(size[contiguous_dim])
        and is_static_int(offset)
        and all(is_static_int(s) for s in inductor_layout.stride)
    ):
        # 获取可能的对齐方式列表
        alignments = get_alignments(dtype)
        # 遍历每种可能的对齐方式
        for alignment in alignments:
            # 检查是否大小和偏移量能被当前对齐方式整除
            if (
                int(size[contiguous_dim]) % alignment != 0
                or int(offset) % alignment != 0
            ):
                continue
            # 检查所有维度是否能被当前对齐方式整除
            if all(
                (dim == contiguous_dim)
                or (inductor_layout.stride[dim] % alignment == 0)
                for dim in range(len(size))
            ):
                # 如果满足所有条件，返回当前对齐方式
                return alignment

    # 如果没有找到合适的对齐方式，返回默认对齐方式1
    return 1
class CUDACompileSourceCapturingContext:
    # 用于捕获传递给 CUDACodeCache.compile 的源代码的辅助类
    # 可用于在隔离环境中进行 CUTLASS 内核的基准测试和测试

    def __init__(self):
        # 初始化函数，创建一个空列表来存储源代码
        self.sources = []
        # 初始化一个空的编译补丁对象
        self._compile_patch = None

    def __enter__(self, *args, **kwargs):
        # 进入上下文管理器时调用的方法
        import unittest.mock as mock
        import torch._inductor.codecache
        
        # 原始的 compile 方法保存在 _compile_method_orig 中
        _compile_method_orig = torch._inductor.codecache.CUDACodeCache.compile

        # 定义自定义的 compile 方法，用于捕获源代码并调用原始的 compile 方法
        def my_compile(source_code, dst_file_ext):
            self.sources.append(source_code)  # 将源代码添加到 sources 列表中
            return _compile_method_orig(source_code, dst_file_ext)  # 调用原始的 compile 方法

        # 使用 unittest.mock.patch 创建一个 patch 对象，替换 CUDACodeCache.compile 方法为 my_compile
        self._compile_patch = mock.patch(
            "torch._inductor.codecache.CUDACodeCache.compile", my_compile
        )

        # 返回 patch 对象的进入方法的调用结果
        return self._compile_patch.__enter__(*args, **kwargs)  # type: ignore[union-attr]

    def __exit__(self, *args, **kwargs):
        # 退出上下文管理器时调用的方法
        # 调用 patch 对象的退出方法
        return self._compile_patch.__exit__(*args, **kwargs)  # type: ignore[union-attr]


def cuda_standalone_runner_compile_command(srcpath: Path, exepath: Path):
    # 返回一个字符串命令，用于将（捕获的）CUDA GEMM Kernel源代码编译为一个独立可执行文件，准备运行
    # 传递正确的预处理器定义给 nvcc，以确保启用独立运行程序
    from torch._inductor.codecache import cuda_compile_command

    # 额外的编译参数，用于启用独立运行程序和设置 CUTLASS 调试追踪级别
    extra_args = ["-DGENERATE_STANDALONE_RUNNER=1", "-DCUTLASS_DEBUG_TRACE_LEVEL=1"]

    # 调用 cuda_compile_command 函数获取编译命令字符串
    compile_command = cuda_compile_command(
        [str(srcpath)], str(exepath), "exe", extra_args=extra_args
    )

    # 返回编译命令字符串
    return compile_command
```