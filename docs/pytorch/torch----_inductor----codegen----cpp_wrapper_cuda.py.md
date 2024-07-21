# `.\pytorch\torch\_inductor\codegen\cpp_wrapper_cuda.py`

```
# mypy: allow-untyped-defs
# 引入 functools、os 模块，从 itertools 中导入 chain 和 count 函数
import functools
import os
from itertools import chain, count
# 从 typing 模块导入 Any、List、Optional 和 TYPE_CHECKING 类型提示
from typing import Any, List, Optional, TYPE_CHECKING

# 导入 sympy 库
import sympy

# 从 torch 中导入 dtype 类型别名
from torch import dtype as torch_dtype
# 从 torch._inductor.codecache 中导入 get_cpp_wrapper_cubin_path_name 函数
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
# 从 torch._inductor.runtime.triton_heuristics 中导入 grid 变量作为默认 grid
from torch._inductor.runtime.triton_heuristics import grid as default_grid

# 从 .. 包中导入 config、CudaKernelParamCache 类
from .. import config
# 从 ..codecache 包中导入 CudaKernelParamCache 类
from ..codecache import CudaKernelParamCache
# 从 ..virtualized 包中导入 V 模块
from ..virtualized import V
# 从 .aoti_hipify_utils 中导入 maybe_hipify_code_wrapper 函数
from .aoti_hipify_utils import maybe_hipify_code_wrapper
# 从 .codegen_device_driver 中导入 cuda_kernel_driver、cuda_kernel_header 函数
from .codegen_device_driver import cuda_kernel_driver, cuda_kernel_header
# 从 .cpp_utils 中导入 DTYPE_TO_CPP 变量
from .cpp_utils import DTYPE_TO_CPP
# 从 .cpp_wrapper_cpu 中导入 CppWrapperCpu 类
from .cpp_wrapper_cpu import CppWrapperCpu
# 从 .wrapper 中导入 SymbolicCallArg 类

# 如果 TYPE_CHECKING 为 True，则从 ..graph 中导入 GraphLowering 类
if TYPE_CHECKING:
    from ..graph import GraphLowering


class CppWrapperCuda(CppWrapperCpu):
    """
    生成用于在 GPU 上运行并调用 CUDA 核心的 cpp 封装器
    """

    def __init__(self):
        # 初始化父类 CppWrapperCpu 的构造函数
        self.device = "cuda"
        super().__init__()
        # 创建一个计数器对象用于生成唯一的 grid_id
        self.grid_id = count()
        # 设置 cuda 标志为 True
        self.cuda = True

    def write_header(self):
        # 如果 V.graph.is_const_graph 为真，则不写常量图的头文件，这将由主模块处理
        if V.graph.is_const_graph:
            return

        # 调用父类的 write_header 方法
        super().write_header()

        # 添加头文件声明 "#include <filesystem>"
        self.header.splice("#include <filesystem>")
        # 如果 config.abi_compatible 为真，添加头文件 "#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>"
        if config.abi_compatible:
            self.header.splice(
                "#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>"
            )
        # 否则，调用 maybe_hipify_code_wrapper 函数处理 cuda_kernel_header()
        else:
            self.header.splice(maybe_hipify_code_wrapper(cuda_kernel_header()))
        # 调用 maybe_hipify_code_wrapper 函数处理 cuda_kernel_driver()
        self.header.splice(maybe_hipify_code_wrapper(cuda_kernel_driver()))

    def write_get_raw_stream(self, index, graph=None):
        # 定义流名称为 stream{index}
        name = f"stream{index}"
        # 写入流的声明语句，调用 maybe_hipify_code_wrapper 处理后的 "cudaStream_t {name};"
        self.writeline(maybe_hipify_code_wrapper(f"cudaStream_t {name};"))
        # 写入错误检查和获取当前 CUDA 流的代码行
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream({index}, (void**)&{name}));"
        )
        # 返回流的名称
        return name

    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, cuda=True
    ):
        # 如果不是 CUDA 内核，调用父类的 define_kernel 方法
        if not cuda:
            return super().define_kernel(name, kernel, metadata, cuda)

    def generate(self, is_inference):
        # 在前缀部分写入空行
        self.prefix.writeline("\n")
        # 如果不是 V.graph.aot_mode 模式
        if not V.graph.aot_mode:
            # 对于每个内核，静态声明 CUfunction {kernel} = nullptr;
            for kernel in chain(
                sorted(self.src_to_kernel.values()),
                sorted([entry[0] for entry in self.user_defined_kernel_cache.values()]),
            ):
                self.prefix.writeline(
                    maybe_hipify_code_wrapper(f"static CUfunction {kernel} = nullptr;")
                )
            # 写入空行
            self.prefix.writeline("\n")
        # 调用父类的 generate 方法
        return super().generate(is_inference)

    def generate_user_defined_triton_kernel(
        self, kernel_name, grid, configs, args, triton_meta, raw_args
    ):
        # 断言 grid 的长度不为 0
        assert len(grid) != 0
        # 如果 grid 只有一个元素，直接选取该元素作为 grid_decision
        if len(grid) == 1:
            grid_decision = grid[0]
        else:
            # 从 CudaKernelParamCache 中获取指定 kernel_name 的元数据
            meta = CudaKernelParamCache.get(kernel_name)
            # 断言 meta 不为 None
            assert meta is not None
            # 初始化 grid_decision 为 None
            grid_decision = None
            # 遍历 configs 中的元素和索引
            for i, c in enumerate(configs):
                # 检查是否所有的参数值与 meta 中的元数据匹配
                if all(arg == meta["meta"][key] for key, arg in c.kwargs.items()):
                    # 如果匹配，选择对应的 grid[i] 作为 grid_decision，并结束循环
                    grid_decision = grid[i]
                    break
            # 断言 grid_decision 不为 None
            assert grid_decision is not None

        # 获取 raw_args 中每个参数的类型信息，如果有 get_dtype 方法则调用它
        arg_types = [
            arg.get_dtype() if hasattr(arg, "get_dtype") else type(arg)
            for arg in raw_args
        ]
        # 调用 generate_kernel_call 方法，传入相应的参数
        self.generate_kernel_call(
            kernel_name,
            args,
            arg_types=arg_types,
            grid=grid_decision,
            cuda=True,
            triton=True,
            triton_meta=triton_meta,
        )

    @functools.lru_cache(None)  # noqa: B019
    def generate_load_kernel_once(
        self,
        name: str,
        mangled_name: str,
        cubin_path: str,
        shared_mem: int,
        graph: "GraphLowering",  # for per-graph caching
    ):
        # 如果处于 AOT 模式
        if V.graph.aot_mode:
            # 生成加载 kernel 的条件语句
            self.writeline(f"if (kernels.{name} == nullptr) {{")
            # 调用 loadKernel 函数加载指定的 kernel
            self.writeline(
                f"""    kernels.{name} = loadKernel("{cubin_path}", "{mangled_name}", {shared_mem}, this->cubin_dir_);"""
            )
            self.writeline("}")
        else:
            # 生成加载 kernel 的条件语句
            self.writeline(f"if ({name} == nullptr) {{")
            # 调用 loadKernel 函数加载指定的 kernel
            self.writeline(
                f"""    {name} = loadKernel("{cubin_path}", "{mangled_name}", {shared_mem});"""
            )
            self.writeline("}")
    def generate_args_decl(self, call_args, arg_types):
        # 初始化一个空列表，用于存储生成的参数声明
        new_args = []
        # 遍历调用参数和参数类型的元组列表
        for arg, arg_type in zip(call_args, arg_types):
            # 生成一个唯一的变量名，形如"var_X"，使用迭代器self.arg_var_id生成
            var_name = f"var_{next(self.arg_var_id)}"
            # 如果参数类型是torch_dtype类型
            if isinstance(arg_type, torch_dtype):
                # 如果参数以".item()"结尾，表示需要声明一个标量变量
                if arg.endswith(".item()"):
                    # 根据torch数据类型获取对应的C++数据类型
                    ctype = DTYPE_TO_CPP[arg_type]
                    # 去掉参数末尾的".item()"部分
                    arg = arg[:-7]
                    # 如果配置要求ABI兼容
                    if config.abi_compatible:
                        # 调用codegen_tensor_item方法生成对应的代码
                        self.codegen_tensor_item(
                            arg_type,
                            arg,
                            var_name,
                        )
                    else:
                        # 否则，生成普通的C++赋值语句
                        self.writeline(f"{ctype} {var_name} = {arg}.item<{ctype}>();")
                else:
                    # 如果参数不以".item()"结尾
                    if config.abi_compatible:
                        # 如果配置要求ABI兼容，生成CUdeviceptr变量声明
                        self.writeline(
                            maybe_hipify_code_wrapper(f"CUdeviceptr {var_name};")
                        )
                        # 调用aoti_torch_get_data_ptr函数获取数据指针，并进行错误检查
                        self.writeline(
                            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr({arg}, reinterpret_cast<void**>(&{var_name})));"
                        )
                    else:
                        # 否则，生成CUdeviceptr变量声明并获取数据指针
                        self.writeline(
                            maybe_hipify_code_wrapper(
                                f"CUdeviceptr {var_name} = reinterpret_cast<CUdeviceptr>({arg}.data_ptr());"
                            )
                        )
            # 如果参数类型是整数或者sympy.Integer类型
            elif arg_type in (sympy.Integer, int):
                # 生成整数变量声明语句
                self.writeline(f"int {var_name} = {self.expr_printer(arg)};")
            # 如果参数类型是浮点数或者sympy.Float类型
            elif arg_type in (sympy.Float, float):
                # 生成浮点数变量声明语句
                self.writeline(f"float {var_name} = {self.expr_printer(arg)};")
            else:
                # 否则，生成auto类型的变量声明语句
                self.writeline(f"auto {var_name} = {self.expr_printer(arg)};")
            # 将生成的变量名加上引用符"&"后添加到new_args列表中
            new_args.append(f"&{var_name}")

        # 将new_args列表中的所有元素用逗号连接成一个字符串并返回
        return ", ".join(new_args)

    def generate_default_grid(self, name: str, grid: List[Any], cuda: bool = True):
        """
        Generate grid configs for launching a CUDA kernel using the grid
        function from triton_heuristics.
        """
        # 如果不使用CUDA，则直接返回grid参数
        if not cuda:
            return grid
        # 断言grid是一个列表
        assert isinstance(grid, list), f"expected {grid=} to be a list"
        # 将grid列表中的每个元素如果是SymbolicCallArg类型，则使用其inner_expr属性，否则保持原样
        grid = [e.inner_expr if isinstance(e, SymbolicCallArg) else e for e in grid]
        # 调用default_grid函数生成默认的grid配置
        grid_fn = default_grid(*grid)
        # 从CudaKernelParamCache中获取指定name的参数信息
        params = CudaKernelParamCache.get(name)
        # 断言params不为None，应该已经存在name对应的CUDA内核参数
        assert (
            params is not None
        ), f"cuda kernel parameters for {name} should already exist at this moment, only found {CudaKernelParamCache.get_keys()}"
        # 构造块配置字典
        block_cfg = {
            "XBLOCK": params["x_block"],
            "YBLOCK": params["y_block"],
            "ZBLOCK": params["z_block"],
        }
        # 使用grid_fn函数对block_cfg进行配置，返回结果
        return grid_fn(block_cfg)
    # 定义生成内核调用的方法，通常用于生成GPU或其他加速设备上的函数调用
    def generate_kernel_call(
        # 内核函数的名称
        self,
        name,
        # 调用内核函数时传递的参数
        call_args,
        # GPU 计算时的网格设置，默认为 None
        grid=None,
        # 设备索引，指定使用的设备，默认为 None
        device_index=None,
        # 是否使用 CUDA，布尔类型，默认为 True
        cuda=True,
        # 是否使用 Triton（NVIDIA 的 GPU 服务器管理工具），布尔类型，默认为 True
        triton=True,
        # 参数类型列表，用于描述每个参数的数据类型，默认为 None
        arg_types=None,
        # 原始参数列表，用于传递给内核函数的未处理参数，默认为 None
        raw_args=None,
        # 用于描述 GPU 网格的函数名称字符串，默认为 "grid"
        grid_fn: str = "grid",
        # Triton 元数据，用于 Triton 支持的额外信息，默认为 None
        triton_meta=None,
```