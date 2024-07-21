# `.\pytorch\torch\_inductor\runtime\triton_heuristics.py`

```
# mypy: allow-untyped-defs
# 引入 Python 内置模块和第三方库
import builtins  # 内置模块
import copy  # 复制对象的模块
import functools  # 函数工具模块，提供高阶函数
import hashlib  # 哈希算法模块
import inspect  # 获取对象信息模块
import json  # JSON 编解码模块
import logging  # 日志记录模块
import math  # 数学函数模块
import operator  # 操作符函数模块
import os  # 操作系统相关模块
import os.path  # 操作系统路径模块
import re  # 正则表达式模块
import sys  # Python 系统相关模块
import threading  # 多线程模块
import time  # 时间模块
from typing import Any, Callable, Dict, List, Optional, Set, Tuple  # 类型提示相关模块

import torch  # PyTorch 深度学习库

from .coordinate_descent_tuner import CoordescTuner  # 导入自定义的调优器

from .hints import (
    _NUM_THREADS_PER_WARP,
    AutotuneHint,
    DeviceProperties,
    HeuristicType,
    ReductionHint,
    TileHint,
    TRITON_MAX_BLOCK,
)  # 从hints模块导入常量和类型提示

from .runtime_utils import (
    cache_dir,
    ceildiv,
    conditional_product,
    create_bandwidth_info_str,
    do_bench_gpu,
    dynamo_timed,
    get_first_attr,
    get_max_y_grid,
    get_num_bytes,
    next_power_of_2,
    triton_config_to_hashable,
)  # 从runtime_utils模块导入各种实用函数

try:
    import triton  # 尝试导入triton模块
except ImportError:
    triton = None  # 如果导入失败，则设置triton为None

if triton is not None:
    from triton import Config  # 如果triton可用，从triton导入Config类
    from triton.compiler import CompiledKernel  # 从triton.compiler导入CompiledKernel类
    from triton.runtime.autotuner import OutOfResources  # 从triton.runtime.autotuner导入OutOfResources类
    from triton.runtime.jit import KernelInterface  # 从triton.runtime.jit导入KernelInterface类

    try:
        from triton.compiler.compiler import ASTSource  # 尝试从triton.compiler.compiler导入ASTSource类
    except ImportError:
        ASTSource = None  # 如果导入失败，则设置ASTSource为None

    try:
        from triton.backends.compiler import GPUTarget  # 尝试从triton.backends.compiler导入GPUTarget类
    except ImportError:
        GPUTarget = None  # 如果导入失败，则设置GPUTarget为None
else:
    Config = object  # 如果triton不可用，则设置Config为object
    KernelInterface = object  # 设置KernelInterface为object
    OutOfResources = object  # 设置OutOfResources为object
    ASTSource = None  # 设置ASTSource为None
    GPUTarget = None  # 设置GPUTarget为None

try:
    autograd_profiler = torch.autograd.profiler  # 尝试从torch.autograd导入profiler模块
except AttributeError:
    # 如果属性错误，则说明在编译工作进程中只有torch的模拟版本
    class autograd_profiler:  # 定义一个空的autograd_profiler类
        _is_profiler_enabled = False  # 设置属性_is_profiler_enabled为False

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


def autotune_hints_to_configs(
    hints: Set[AutotuneHint], size_hints, block_size: int
) -> List[Config]:
    """
    AutotuneHints可以附加到triton内核的元数据中，提供有关自动调优的建议。
    如果有一些配置仅在特定场景下有用，则可以避免在不了解是否在这些场景中时浪费编译时间。

    根据这些提示，该函数将生成一个额外的自动调优配置列表。
    """
    xyz_options: Tuple[Tuple[int, Optional[int], Optional[int]], ...]  # 声明xyz_options变量的类型
    configs = []  # 初始化一个空列表用于存储配置
    # 遍历自动调整提示列表中的每一个提示
    for hint in hints:
        # 检查提示是否为 AutotuneHint.ELEMENTS_PER_WARP_32
        if hint == AutotuneHint.ELEMENTS_PER_WARP_32:
            # 根据 size_hints 的长度确定不同的 xyz_options
            if len(size_hints) == 1:
                # 如果 size_hints 长度为 1，设置 xyz_options 为一个元组
                xyz_options = ((block_size // 4, None, None),)
            elif len(size_hints) == 2:
                # 如果 size_hints 长度为 2，设置 xyz_options 包含两个元组
                xyz_options = ((block_size // 4, 1, None), (1, block_size // 4, None))
            elif len(size_hints) == 3:
                # 如果 size_hints 长度为 3，设置 xyz_options 包含三个元组
                xyz_options = (
                    (block_size // 4, 1, 1),
                    (1, block_size // 4, 1),
                    (1, 1, block_size // 4),
                )
            # 遍历 xyz_options 中的每一个 xyz 元组
            for xyz in xyz_options:
                # 将每个 xyz 元组与 num_elements_per_warp=32 一起作为参数，
                # 调用 triton_config 函数并将返回结果添加到 configs 列表中
                configs.append(
                    triton_config(
                        size_hints,
                        *xyz,
                        num_elements_per_warp=32,
                    )
                )

    # 返回配置列表 configs
    return configs
# 禁用逐点自动调优功能
def disable_pointwise_autotuning(inductor_meta):
    # 当启用确定性算法标志位时，禁用自动调优，因为自动调优可能导致运行时的不确定性结果。
    if inductor_meta.get("are_deterministic_algorithms_enabled"):
        return True
    # 如果未设置禁用自动调优标志，或者标志为假，则默认启用逐点自动调优。
    return not inductor_meta.get("autotune_pointwise", True)


def _dump_launch_params(args, kwargs, launcher, kernel_name):
    call_args = []
    call_kwargs = {}
    # 将所有参数转换为字符串形式
    for arg in args:
        if isinstance(arg, (int, bool)):
            call_args.append(str(arg))
        else:
            call_args.append("T")
    # 将所有关键字参数转换为字符串形式
    for k, v in kwargs.items():
        if isinstance(arg, (int, bool)):  # 应为 isinstance(arg, (int, bool))
            call_kwargs[k] = v
        else:
            call_kwargs[k] = v
    # 将启动器配置的所有关键字参数添加到调用参数中
    for k, v in launcher.config.kwargs.items():
        call_kwargs[k] = v
    # 添加特定于调用的参数：num_warps 和 num_stages
    call_kwargs["num_warps"] = launcher.config.num_warps
    call_kwargs["num_stages"] = launcher.config.num_stages
    # 构建参数字符串，以便记录调用的详细信息
    args_str = ""
    args_str += ", ".join(call_args)
    for k, v in call_kwargs.items():
        args_str += f", {k}={v}"

    # 获取当前脚本的绝对路径，并在其基础上创建一个以 .launch_params 结尾的文件
    abs_path = os.path.abspath(sys.argv[0])
    # 将核函数名称和调用参数写入文件，以记录启动参数
    with open(f"{abs_path}.launch_params", "a") as f:
        f.write(f"{kernel_name} | {args_str}\n")


class CachingAutotuner(KernelInterface):
    """
    简化版 Triton 自动调优器，没有无效化密钥，通过缓存最佳配置到磁盘以提高冷启动时间。
    与主要的 Triton 自动调优器不同，此版本可以预编译所有配置，并且不依赖 Triton JIT。
    """

    def __init__(
        self,
        fn,
        triton_meta,  # 直接传递给 Triton 的元数据
        configs,
        save_cache_hook,
        mutated_arg_names,
        heuristic_type,
        size_hints=None,
        inductor_meta=None,  # 不与 Triton 相关的元数据
        custom_kernel=False,  # 是否为生成的感应器内核或自定义内核
        filename: Optional[str] = None,
        ):
        super().__init__()  # 调用父类的构造方法

        assert len(configs) > 0, "Non-empty TritonConfig list required for compiling"  # 断言确保 configs 列表非空
        self.fn = fn  # 设置实例变量 fn，存储编译函数的引用
        self.device_props: DeviceProperties = triton_meta["device"]  # 设置实例变量 device_props，存储 Triton 设备属性
        self.triton_meta = {
            **triton_meta,
            "device": self.device_props.index,
            "device_type": self.device_props.type,
        }  # 创建 Triton 元数据字典，包括设备索引和类型等信息
        self.inductor_meta = {} if inductor_meta is None else inductor_meta  # 设置实例变量 inductor_meta，存储感应器元数据
        self.save_cache_hook = save_cache_hook  # 设置实例变量 save_cache_hook，存储保存缓存的钩子函数
        self.mutated_arg_names = mutated_arg_names  # 设置实例变量 mutated_arg_names，存储变异参数的名称列表
        self.configs = configs  # 设置实例变量 configs，存储编译配置列表
        self.heuristic_type = heuristic_type  # 设置实例变量 heuristic_type，存储启发式类型
        self.custom_kernel = custom_kernel  # 设置实例变量 custom_kernel，存储自定义内核
        self.cuda_kernel_saved = False  # 初始化实例变量 cuda_kernel_saved，用于标记 CUDA 内核是否已保存

        if log.isEnabledFor(logging.DEBUG):  # 如果 DEBUG 日志已启用
            log.debug(
                "CachingAutotuner gets %d configs for %s",
                len(self.configs),
                self.fn.__name__,
            )  # 记录调试信息，显示获取的配置数量和函数名称
            for c in self.configs:
                log.debug(c)  # 记录每个配置项的详细信息

        self.launchers = []  # 初始化实例变量 launchers，用于存储启动器对象列表
        self.lock = threading.Lock()  # 初始化实例变量 lock，用于多线程同步操作

        if os.getenv("TRITON_CACHE_DIR") is None:  # 如果未设置环境变量 TRITON_CACHE_DIR
            os.environ["TRITON_CACHE_DIR"] = os.path.join(
                cache_dir(),
                "triton",
                str(self.triton_meta.get("device", 0)),
            )  # 设置 Triton 缓存目录路径

        log.debug("Triton cache dir: %s", os.environ["TRITON_CACHE_DIR"])  # 记录 Triton 缓存目录路径的调试信息

        self.size_hints = size_hints  # 设置实例变量 size_hints，存储大小提示信息
        self.coordesc_tuner = CoordescTuner(
            is_mm=False,
            name=self.fn.__name__,
            size_hints=size_hints,
            inductor_meta=self.inductor_meta,
        )  # 创建 CoordescTuner 对象，用于协调参数调优

        self.filename = filename  # 设置实例变量 filename，存储文件名
    def bench(self, launcher, *args, grid, **kwargs):
        """Measure the performance of a given launcher"""
        # 如果不是自定义内核，并且 launcher.n_spills 大于等于自动调整的溢出阈值（默认为16）
        # 则跳过具有寄存器溢出的配置，因为对于某些复杂的自定义 Triton 内核，寄存器溢出配置可能提供最佳延迟。
        if not self.custom_kernel and launcher.n_spills > self.inductor_meta.get(
            "spill_threshold", 16
        ):
            log.debug(
                "Skip config %s because of register spilling: %d",
                launcher.config,
                launcher.n_spills,
            )
            return float("inf")  # 返回无穷大作为性能指标

        device_interface = self.get_device_interface()
        # 获取当前设备的原始流
        stream = device_interface.get_raw_stream(  # type: ignore[call-arg]
            device_interface.current_device()
        )

        def kernel_call():
            # 如果 launcher.config.pre_hook 不为 None，则执行预处理钩子函数
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook(
                    {**dict(zip(self.arg_names, args)), **launcher.config.kwargs}
                )

            # 克隆参数和关键字参数，用于避免自动调整污染处于原地存储的缓冲区
            cloned_args, cloned_kwargs = self.clone_args(*args, **kwargs)
            # 调用 launcher 函数进行内核启动
            launcher(
                *cloned_args,
                **cloned_kwargs,
                grid=grid,
                stream=stream,
            )

        return do_bench_gpu(kernel_call, rep=40, fast_flush=True)

    def clone_args(self, *args, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        from ..compile_fx import clone_preserve_strides

        # 克隆原地缓冲区，以避免自动调整污染它们，如果内核进行原地存储操作。避免克隆其他缓冲区，因为会增加内存使用
        cloned_args = []
        for i, arg in enumerate(args):
            if self.fn.arg_names[i] in self.mutated_arg_names:
                assert isinstance(arg, torch.Tensor)
                cloned_args.append(clone_preserve_strides(arg))
            else:
                cloned_args.append(arg)

        cloned_kwargs: Dict[str, Any] = {}
        for name, arg in kwargs.items():
            if name in self.mutated_arg_names:
                assert isinstance(arg, torch.Tensor)
                cloned_kwargs[name] = clone_preserve_strides(arg)
            else:
                cloned_kwargs[name] = arg

        return cloned_args, cloned_kwargs

    @dynamo_timed
    def benchmark_all_configs(self, *args, **kwargs):
        # 生成包含所有启动器的计时器字典
        timings = {
            launcher: self.bench(launcher, *args, **kwargs)
            for launcher in self.launchers
        }

        # 缓存每个配置的基准结果
        for k, v in timings.items():
            self.coordesc_tuner.cache_benchmark_result(k.config, v)

        # 如果日志启用了调试级别，输出基准测试结果到日志
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Benchmark all input configs for %s, get:", self.fn.__name__)
            for k, v in timings.items():
                log.debug(
                    "%s: %f, nreg %d, nspill %d, #shared-mem %s",
                    k.config,
                    v,
                    k.n_regs,
                    k.n_spills,
                    k.shared,
                )

        # 返回计时器字典
        return timings

    def autotune_to_one_config(self, *args, **kwargs):
        """执行实际的自动调整"""
        # 记录开始时间
        start_time = time.time_ns()
        # 对所有配置进行基准测试，并获取时间信息
        timings = self.benchmark_all_configs(*args, **kwargs)
        # 计算运行时间
        time_taken_ns = time.time_ns() - start_time
        # 选择性能最佳的启动器配置
        self.launchers = [builtins.min(timings, key=timings.get)]
        # 如果定义了保存缓存的钩子函数，调用它来保存缓存结果
        if self.save_cache_hook:
            self.save_cache_hook(self.launchers[0].config, time_taken_ns)

    def save_cuda_kernel(self, grid, stream, launcher):
        # 如果 grid 是可调用对象，使用其参数来获取网格大小
        if callable(grid):
            grid_x, grid_y, grid_z = grid(launcher.config.kwargs)
        else:
            grid_x, grid_y, grid_z = grid

        # 获取感应器元数据中的唯一内核名称
        key = self.inductor_meta.get("kernel_name", None)
        # 确保内核名称不为 None
        assert key is not None, "kernel_name can not be None"
        # 准备内核参数字典
        params = {
            "mangled_name": launcher.bin.metadata.name
            if hasattr(launcher.bin.metadata, "name")
            else launcher.bin.metadata["name"],
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z,
            "x_block": launcher.config.kwargs.get("XBLOCK", 1),
            "y_block": launcher.config.kwargs.get("YBLOCK", None),
            "z_block": launcher.config.kwargs.get("ZBLOCK", None),
            "num_warps": launcher.bin.num_warps
            if hasattr(launcher.bin, "num_warps")
            else launcher.bin.metadata.num_warps,
            "shared_mem": launcher.bin.shared
            if hasattr(launcher.bin, "shared")
            else launcher.bin.metadata.shared,
            "stream": stream,
            # 用户定义的 triton 内核可能具有任意的关键字参数名
            "meta": launcher.config.kwargs,
        }
        # 导入 CUDA 内核参数缓存类
        from torch._inductor.codecache import CudaKernelParamCache

        # 根据设备类型选择合适的二进制数据
        binary = (
            launcher.bin.asm["cubin"]
            if self.device_props.type != "hip"
            else launcher.bin.asm["hsaco"]
        )
        # 将内核参数和二进制数据存入 CUDA 内核参数缓存
        CudaKernelParamCache.set(key, params, binary)

        # 标记 CUDA 内核已保存
        self.cuda_kernel_saved = True
    def coordinate_descent_tuning(self, launcher, *args, **kwargs):
        """
        Coordinate descent tuning can be run with or without max-autotune.

        The only difference between these two is the starting config for coordinate_descent tuning.
        E.g., assuming regular autotune only get one config C1; while max-autotune get 4 configs C1, C2, C3, C4
        and max-autotune figure out C3 is the best.

        Then if coordinate desecnt tuning is run with max-autotune disabled, it will start from C1;
        while if coordinate descent tuning is run with max-autotune enabled, it will start from C3.
        """

        # Check if the heuristic type is TEMPLATE or USER_AUTOTUNE; if true, skip tuning
        if (
            self.heuristic_type == HeuristicType.TEMPLATE
            or self.heuristic_type == HeuristicType.USER_AUTOTUNE
        ):
            # skip tuning and return the launcher unchanged
            return launcher

        # Dictionary mapping launcher's configuration to the launcher object itself
        config2launcher = {launcher.config: launcher}

        # Function to benchmark performance for a given configuration
        def benchmark_one_config(config):
            # Acquire a lock to ensure thread safety
            with self.lock:
                # Precompile the configuration and retrieve the modified launcher
                _, launcher = self._precompile_config(config, False)
            # Store the launcher object corresponding to this configuration
            config2launcher[config] = launcher

            # Perform benchmarking using the modified launcher and provided arguments
            out = self.bench(launcher, *args, **kwargs)
            # Log benchmarking results including metrics like configuration, output, registers, spills, and shared memory
            log.debug(
                "COORDESC: %s: %f, nreg %d, nspill %d, #shared-mem %d",
                launcher.config,
                out,
                launcher.n_regs,
                launcher.n_spills,
                launcher.shared,
            )
            return out

        # Assertion to ensure persistent reduction's configuration does not have RBLOCK
        assert not (
            self.heuristic_type == HeuristicType.PERSISTENT_REDUCTION
            and "RBLOCK" in launcher.config.kwargs
        ), "Coordinate descent tuner relies on the assumption that persistent reduction's triton config does not have RBLOCK"

        # Measure the starting time for autotuning
        start_time = time.time_ns()
        # Perform autotuning for the best configuration using coordinate descent tuner
        best_config = self.coordesc_tuner.autotune(
            benchmark_one_config, launcher.config, None
        )
        # Calculate the time taken for autotuning in nanoseconds
        time_taken_ns = time.time_ns() - start_time
        # Mark the best configuration found by coordinate descent tuning
        best_config.found_by_coordesc = True

        # If a cache saving hook is defined, invoke it to save the best configuration and autotuning time
        if self.save_cache_hook:
            self.save_cache_hook(best_config, time_taken_ns, found_by_coordesc=True)
        
        # Return the launcher corresponding to the best configuration found
        return config2launcher.get(best_config)
    # 定义一个方法 `run`，接受任意位置参数 `args`，两个必选关键字参数 `grid` 和 `stream`，以及任意关键字参数 `kwargs`
    def run(self, *args, grid, stream, **kwargs):
        # 如果 `self.launchers` 列表的长度不为 1
        if len(self.launchers) != 1:
            # 如果 `self.launchers` 列表的长度为 0
            if len(self.launchers) == 0:
                # 调用 `precompile` 方法，预编译相关内容
                self.precompile()
            # 如果 `self.launchers` 列表的长度大于 1
            if len(self.launchers) > 1:
                # 调用 `autotune_to_one_config` 方法，自动调优到一个配置
                self.autotune_to_one_config(*args, grid=grid, **kwargs)

        # 如果 `self.launchers[0].config` 对象的属性 `found_by_coordesc` 为 False，并且 `self.inductor_meta` 的键 "coordinate_descent_tuning" 为 True
        if not getattr(
            self.launchers[0].config, "found_by_coordesc", False
        ) and self.inductor_meta.get("coordinate_descent_tuning", False):
            # 使用 `coordinate_descent_tuning` 方法对 `self.launchers[0]` 进行坐标下降调优
            self.launchers = [
                self.coordinate_descent_tuning(
                    self.launchers[0], *args, grid=grid, **kwargs
                )
            ]

        # 将 `self.launchers` 列表的唯一元素赋给变量 `launcher`
        (launcher,) = self.launchers
        # 如果 `launcher` 的 `store_cubin` 属性为 True
        if launcher.store_cubin:
            # 调用 `save_cuda_kernel` 方法保存 CUDA 内核
            self.save_cuda_kernel(grid, stream, launcher)

        # 如果 `launcher.config.pre_hook` 不为 None
        if launcher.config.pre_hook is not None:
            # 调用 `pre_hook` 方法，传入参数字典，包括位置参数、配置参数和关键字参数
            launcher.config.pre_hook(
                {**dict(zip(self.arg_names, args)), **launcher.config.kwargs, **kwargs}
            )

        # 如果环境变量 `TORCHINDUCTOR_DUMP_LAUNCH_PARAMS` 的值为 "1"
        if os.environ.get("TORCHINDUCTOR_DUMP_LAUNCH_PARAMS", 0) == "1":
            # 调用 `_dump_launch_params` 函数，输出启动参数的详细信息
            _dump_launch_params(args, kwargs, launcher, self.fn.__name__)

        # 速度比进入和退出上下文管理器更快，即使上下文管理器是空的。
        # 如果自动求导分析器启用
        if autograd_profiler._is_profiler_enabled:
            # 如果 `grid` 是一个整数元组或者是一个字符串
            if isinstance(grid, tuple):
                # 将 `grid` 转换为字符串形式
                grid_info = str(grid)
            else:
                # 否则使用 `grid` 对象的 `grid_fn_str` 属性值作为 `grid_info`
                grid_info = getattr(grid, "grid_fn_str", "")
            # 使用 Torch 的快速记录函数 `_RecordFunctionFast` 进行性能分析
            with torch._C._profiler._RecordFunctionFast(
                self.inductor_meta.get("kernel_name", "triton kernel"),  # 核心函数名称
                args,  # 传入的位置参数
                {
                    "kernel_file": "" if self.filename is None else self.filename,  # 内核文件名
                    "kernel_backend": "triton",  # 内核后端
                    "grid": grid_info,  # 网格信息
                    "stream": stream,  # 流信息
                },
            ):
                # 返回调用 `launcher` 对象，传入参数 `args` 和 `kwargs`，以及 `grid` 和 `stream` 关键字参数
                return launcher(
                    *args,
                    **kwargs,
                    grid=grid,
                    stream=stream,
                )
        else:
            # 如果自动求导分析器未启用，直接返回调用 `launcher` 对象，传入参数 `args` 和 `kwargs`，以及 `grid` 和 `stream` 关键字参数
            return launcher(
                *args,
                **kwargs,
                grid=grid,
                stream=stream,
            )
# 定义一个函数，用于查找引用了给定对象的所有变量名
def _find_names(obj):
    import gc  # 导入垃圾回收模块
    import inspect  # 导入检查模块

    # 获取当前调用栈帧
    frame = inspect.currentframe()
    while frame is not None:
        frame.f_locals  # 访问当前帧的局部变量
        frame = frame.f_back  # 向上遍历调用栈帧

    obj_names = []  # 存储找到的变量名列表
    # 遍历引用了给定对象的所有引用者
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):  # 仅考虑引用者为字典的情况
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)  # 将引用了给定对象的变量名加入列表

    return obj_names  # 返回找到的所有变量名列表


collected_calls: List[Any] = []  # 定义一个全局变量列表，用于存储收集到的调用信息


def start_graph():
    collected_calls.clear()  # 清空收集到的调用信息列表


def end_graph(output_file):
    if len(collected_calls) == 0:
        return  # 如果没有收集到任何调用信息，则直接返回

    overall_time = sum(call[0] for call in collected_calls)  # 计算总的调用时间
    overall_gb = sum(call[1] for call in collected_calls)  # 计算总的数据量（GB）
    cur_file = inspect.stack()[1].filename  # 获取调用栈中上一层的文件名
    summary_str = (
        f"SUMMARY ({cur_file})\n"  # 汇总信息的字符串，包含文件名
        f"{overall_time:.2f}ms   \t {overall_gb:.2f} GB\t {overall_gb/(overall_time/1e3):.2f}GB/s"
    )  # 汇总信息，包括总时间、总数据量和带宽

    print(summary_str)  # 打印汇总信息
    print()  # 输出一个空行

    if output_file is not None:
        # 按照运行时间降序排列调用信息
        sorted_calls = sorted(collected_calls, key=lambda c: float(c[0]), reverse=True)
        try:
            with open(output_file, "a") as file:
                log.debug("Save profile bandwidth results to %s", output_file)
                file.write("====================\n")  # 写入分隔线
                file.write(f"TRITON KERNELS BANDWIDTH INFO ({cur_file})\n")  # 写入标题和文件名
                for ms, num_gb, gb_per_s, kernel_name in sorted_calls:
                    # 计算每个内核的运行时间百分比
                    percentage = f"{ms/overall_time*100:.2f}%"
                    suffix = f" \t {percentage} \t {kernel_name}"
                    # 创建带宽信息字符串
                    bw_info_str = create_bandwidth_info_str(
                        ms,
                        num_gb,
                        gb_per_s,
                        suffix=suffix,
                        color=False,
                    )
                    file.write(bw_info_str + "\n")  # 将带宽信息写入文件
                file.write(f"{summary_str}\n\n")  # 将汇总信息写入文件
        except Exception as e:
            log.warning(
                "failed to write profile bandwidth result into %s: %s",
                output_file,
                e,
            )  # 如果写入文件出错，则记录警告信息


class DebugAutotuner(CachingAutotuner):
    def __init__(self, *args, regex_filter="", **kwargs):
        self.regex_filter = regex_filter  # 初始化正则表达式过滤器
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法
        self.cached = None  # 初始化缓存为空
    # 定义一个方法 `run`，接受任意参数 `args`，并接收 `grid` 和 `stream` 作为关键字参数
    def run(self, *args, grid, stream):
        # 查找可能的内核名称列表
        possible_names = _find_names(self)
        # 选择最长的内核名称作为 `kernel_name`
        kernel_name = f"{max(possible_names, key=len)}"
        # 如果 `kernel_name` 不符合正则表达式的过滤条件，则返回
        if not re.match(self.regex_filter, kernel_name):
            return
        # 调用父类的 `run` 方法，传递 `args`，`grid` 和 `stream`
        super().run(*args, grid=grid, stream=stream)
        # 获取 `self.launchers` 列表中的第一个元素作为 `launcher`
        (launcher,) = self.launchers

        # 如果缓存 `self.cached` 为空
        if self.cached is None:
            # 运行 `bench` 方法，使用 `launcher` 和 `args`，`grid` 运行负载测试，并获取执行时间 `ms`
            ms = self.bench(launcher, *args, grid=grid)
            # 计算以 `in_out_ptr` 开头的参数数量
            num_in_out_ptrs = len(
                [
                    arg_name
                    for arg_name in self.fn.arg_names
                    if arg_name.startswith("in_out_ptr")
                ]
            )
            # 从 `inductor_meta` 获取 `kernel_num_gb`，默认为 `None`
            num_gb = self.inductor_meta.get("kernel_num_gb", None)
            # 如果 `num_gb` 为 `None`，则通过 `get_num_bytes` 计算字节数并转换为 GB
            if num_gb is None:
                num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9
            # 计算每秒的传输速率 GB/s
            gb_per_s = num_gb / (ms / 1e3)
            # 将计算结果缓存到 `self.cached`，包括执行时间、数据量、传输速率和内核名称
            self.cached = ms, num_gb, gb_per_s, kernel_name
            # 将执行时间、数据量、传输速率和内核名称添加到 `collected_calls` 列表中
            collected_calls.append((ms, num_gb, gb_per_s, kernel_name))
            # 打印带宽信息字符串，显示执行时间、数据量、传输速率和内核名称
            print(
                create_bandwidth_info_str(
                    ms, num_gb, gb_per_s, suffix=f" \t {kernel_name}"
                )
            )
# 计算给定配置列表的哈希值，用于检查配置更改
def hash_configs(configs: List[Config]):
    # 创建一个 SHA-256 哈希对象
    hasher = hashlib.sha256()
    # 遍历每个配置对象
    for cfg in configs:
        # 对配置的关键字参数按键排序后，加入哈希对象中
        hasher.update(
            f"{sorted(cfg.kwargs.items())} {cfg.num_warps} {cfg.num_stages}\n".encode()
        )
    # 返回计算得到的哈希值的十六进制表示
    return hasher.hexdigest()


# 加载缓存的自动调优配置
def load_cached_autotuning(
    best_config,
    configs_hash: str,
    configs: List[Config],
    inductor_meta: Dict[str, Any],
):
    # 如果最佳配置为空，则返回 None
    if best_config is None:
        return None
    # 如果最佳配置中的哈希值与给定的配置哈希值不匹配，则返回 None
    if best_config.pop("configs_hash", None) != configs_hash:
        return None

    # 移除比较所用的时间（毫秒）
    best_config.pop("time_taken_ms", None)

    # 如果需要坐标下降调优，并且最佳配置是通过坐标下降发现的
    if inductor_meta.get("coordinate_descent_tuning") and best_config.pop(
        "found_by_coordesc", False
    ):
        # 获取数值参数 num_warps 和 num_stages
        num_warps = best_config.pop("num_warps")
        num_stages = best_config.pop("num_stages")
        # 创建新的配置对象 triton_config，并设置其找到方式为坐标下降
        triton_config = Config(best_config, num_warps=num_warps, num_stages=num_stages)
        triton_config.found_by_coordesc = True
        return triton_config

    # 查找与最佳配置匹配的配置对象列表
    matching_configs = [
        cfg
        for cfg in configs
        if all(val == best_config.get(key) for key, val in cfg.kwargs.items())
        and cfg.num_warps == best_config.get("num_warps")
        and cfg.num_stages == best_config.get("num_stages")
    ]
    # 如果找到多于一个匹配的配置，则返回 None
    if len(matching_configs) != 1:
        return None

    # 返回找到的唯一匹配配置
    return matching_configs[0]


# 决定是否使用远程自动调优缓存
def should_use_remote_autotune_cache(inductor_meta):
    # 如果设置了远程自动调优缓存，则返回 True
    if inductor_meta.get("autotune_remote_cache"):
        return True
    # 如果不是在 Facebook 代码库中，则返回 False
    if not inductor_meta.get("is_fbcode"):
        return False
    # 如果是在 HIP 环境中，则返回 False
    if inductor_meta.get("is_hip"):
        return False

    # 导入 MEMCACHE_VERSION，判断是否满足远程缓存版本要求
    from triton.fb.fb_memcache import MEMCACHE_VERSION

    return MEMCACHE_VERSION >= torch._utils_internal.justknobs_getval_int(
        "pytorch/remote_cache:autotune_memcache_version"
    )


# 缓存的自动调优函数
def cached_autotune(
    size_hints: Optional[List[int]],
    configs: List[Config],
    triton_meta,
    heuristic_type,
    filename=None,
    inductor_meta=None,
    custom_kernel=False,
):
    """
    这是 triton.autotune 的一个副本，调用我们的子类。我们的子类具有额外的调试、错误处理和磁盘缓存功能。
    """
    # 确保配置列表中的配置唯一
    configs = unique_configs(configs)
    # 断言配置列表长度为 1 或者提供了文件名
    assert len(configs) == 1 or filename

    # 初始化保存缓存钩子为 None
    save_cache_hook: Optional[Callable[[Any, Any, Any], Any]]
    # 如果未提供 inductor_meta，则设置为空字典
    inductor_meta = {} if inductor_meta is None else inductor_meta

    # 硬盘缓存逻辑和/或远程缓存
    if filename is not None and (
        len(configs) > 1 or inductor_meta.get("coordinate_descent_tuning")
    ):
        save_cache_hook = None
    else:
        save_cache_hook = None

    # 弹出并获取 inductor_meta 中的 mutated_arg_names
    mutated_arg_names = inductor_meta.pop("mutated_arg_names", ())
    def decorator(fn):
        # 定义一个装饰器函数，接受一个函数作为参数 fn

        # 如果函数 fn 的签名中不包含参数 "XBLOCK"
        if "XBLOCK" not in inspect.signature(fn.fn).parameters:
            # 遍历 configs 列表中的每个配置项 tconfig
            for tconfig in configs:
                # 如果 tconfig.kwargs 中包含键 "XBLOCK"
                if "XBLOCK" in tconfig.kwargs:
                    # 断言 tconfig.kwargs["XBLOCK"] 等于 1
                    assert tconfig.kwargs["XBLOCK"] == 1
                    # 移除 tconfig.kwargs 中的 "XBLOCK" 键
                    tconfig.kwargs.pop("XBLOCK")

        # 如果 inductor_meta 中存在 "profile_bandwidth" 键
        if inductor_meta.get("profile_bandwidth"):
            # 返回一个 DebugAutotuner 对象，使用给定参数实例化
            return DebugAutotuner(
                fn,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                regex_filter=inductor_meta["profile_bandwidth_regex"],
                configs=configs,
                save_cache_hook=save_cache_hook,
                mutated_arg_names=mutated_arg_names,
                heuristic_type=heuristic_type,
                size_hints=size_hints,
                custom_kernel=custom_kernel,
                filename=filename,
            )
        # 如果 inductor_meta 中不存在 "profile_bandwidth" 键
        # 返回一个 CachingAutotuner 对象，使用给定参数实例化
        return CachingAutotuner(
            fn,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            configs=configs,
            save_cache_hook=save_cache_hook,
            mutated_arg_names=mutated_arg_names,
            heuristic_type=heuristic_type,
            size_hints=size_hints,
            custom_kernel=custom_kernel,
            filename=filename,
        )

    # 返回装饰器函数 decorator
    return decorator
# 移除重复的配置项，保留唯一的配置项列表
def unique_configs(configs: List[Config]):
    seen = set()  # 创建一个空的集合，用于存储已经遇到的配置的哈希值
    pruned_configs = []  # 创建一个空列表，用于存储去重后的配置项列表

    for cfg in configs:
        key = triton_config_to_hashable(cfg)  # 将配置项转换为可哈希的键
        if key not in seen:  # 如果当前键不在集合中
            seen.add(key)  # 将当前键添加到集合中，标记为已见
            pruned_configs.append(cfg)  # 将当前配置项添加到去重后的列表中
    return pruned_configs  # 返回去重后的配置项列表


# 检查特定配置中的块大小是否符合预期要求
def check_config(cfg, *, xnumel=None, ynumel=None, znumel=None):
    for numel, label in zip((xnumel, ynumel, znumel), "XYZ"):  # 遍历给定的尺寸元组和对应的标签
        if numel is None:  # 如果当前尺寸为空，则跳过
            continue
        block = cfg[f"{label}BLOCK"]  # 获取配置中对应维度的块大小
        if numel == 1:
            assert block == 1, (  # 如果尺寸为1，则要求块大小也必须为1，否则触发断言异常
                f"TritonKernel.indexing assumes numel == 1 => BLOCK == 1"
                f" but {label.lower()}numel=={numel} and {label}BLOCK={block} (cfg={cfg})."
            )
        max_block = TRITON_MAX_BLOCK[label]  # 获取最大块大小限制
        max_block_str = f'config.triton.max_block["{label}"]'  # 获取最大块大小的字符串表示
        assert max_block % block == 0, (  # 断言当前块大小能整除最大块大小，否则触发异常
            f"TritonKernel.indexing assumes {label}BLOCK divides {max_block_str}"
            f" but {label}BLOCK={block} and {max_block_str}={max_block} (cfg={cfg})."
        )


# 根据给定的提示和参数构建 Triton 配置
def triton_config(
    size_hints,
    x,
    y=None,
    z=None,
    num_stages=1,
    num_elements_per_warp=256,
    min_elem_per_thread=0,
) -> Config:
    """
    Construct a pointwise triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.

    num_elements_per_warp is a suggestion for controlling how many warps
    the triton config should contain. e.g.: if x=16, y=8, z=4 then
    num_elements = 16*8*4 = 512. Then if we set num_elements_per_warp=128,
    we'll launch 512 (elem) / 128 (elem/warp) = 4 warps. Note that it's
    just a suggestion, and sometimes other adjustment heuristics will
    override the num_elements_per_warp.

    min_elem_per_thread controls the minimum number of elements
    processed by each thread. It's always enforced.
    """
    size_hints = list(reversed(size_hints))  # 将尺寸提示列表反转，以便适应特定的维度映射

    maxGridSize = [2147483647, 65535, 65535]  # 设置最大网格尺寸限制

    target = conditional_product(x, y, z)  # 计算目标尺寸，基于给定的 x, y, z 尺寸组合
    if conditional_product(*size_hints) < target:  # 如果尺寸提示的乘积小于目标尺寸，则目标尺寸除以8
        target //= 8

    x = min(x, size_hints[0])  # 缩小 x 尺寸，以符合尺寸提示的第一个维度
    if y:
        y = min(y, size_hints[1])  # 如果有 y 尺寸，缩小 y 尺寸
    if z:
        z = min(z, size_hints[2])  # 如果有 z 尺寸，缩小 z 尺寸

    while x < min(size_hints[0], TRITON_MAX_BLOCK["X"]) and (
        x * maxGridSize[0] < size_hints[0] or conditional_product(x, y, z) < target
    ):
        x *= 2  # 如果 x 小于原始块大小或计算的网格大小大于限制，将 x 尺寸加倍
    # 当 y 不为零且未超过给定限制和条件允许时，逐步倍增 y 的值
    while (
        y
        and y < min(size_hints[1], TRITON_MAX_BLOCK["Y"])
        and (
            y * maxGridSize[1] < size_hints[1] or conditional_product(x, y, z) < target
        )
    ):
        y *= 2
    
    # 当 z 不为零且未超过给定限制和条件允许时，逐步倍增 z 的值
    while (
        z
        and z < min(size_hints[2], TRITON_MAX_BLOCK["Z"])
        and (
            z * maxGridSize[2] < size_hints[2] or conditional_product(x, y, z) < target
        )
    ):
        z *= 2

    # 计算所需的 warps 数量，确保最小的元素数每个线程块的要求
    num_warps = next_power_of_2(
        min(max(conditional_product(x, y, z) // num_elements_per_warp, 1), 8)
    )
    
    # 根据条件设置最小 warps 数量为 4，以处理特定的 PTX 编译器问题
    num_warps = max(num_warps, 4) if conditional_product(x, y, z) >= 128 else num_warps
    
    # 设置 x, y, z 的元素数量
    xnumel = size_hints[0]
    ynumel = size_hints[1] if y else None
    znumel = size_hints[2] if z else None

    # 计算线程块的大小，以满足最小元素数每个线程的要求
    block_size = max(
        conditional_product(x, y, z),
        min_elem_per_thread * _NUM_THREADS_PER_WARP * num_warps,
    )
    
    # 调整 x 的值，以确保满足最小线程元素数的要求
    x *= math.ceil(block_size / conditional_product(x, y, z))

    # 构建配置字典 cfg，设置 XBLOCK、YBLOCK、ZBLOCK 的值
    cfg = {"XBLOCK": x}
    if y:
        cfg["YBLOCK"] = y
    if z:
        cfg["ZBLOCK"] = z
    
    # 检查配置是否满足指定的元素数目，并返回配置对象
    check_config(cfg, xnumel=xnumel, ynumel=ynumel, znumel=znumel)
    
    # 返回包含配置和 warps 数量的 Config 对象
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)
# 构建一个用于减少 Triton 配置的函数，根据 size_hints 进行一些调整启发。size_hints 是每个瓦片维度中的元素数的元组，并将其向上舍入到最接近的2的幂次方。

def triton_config_reduction(size_hints, x, r, num_stages=1, num_warps=None) -> Config:
    target = conditional_product(x, r)
    # 如果 size_hints 的乘积小于 target，则将 target 缩小为其八分之一
    if conditional_product(*size_hints) < target:
        target //= 8

    # 缩小 x 和 r 到 size_hints 指定的大小
    x = min(x, size_hints[0])
    r = min(r, size_hints[1])

    # 如果 x 小于 size_hints[0] 并且 x 和 r 的乘积小于 target，则将 x 扩展两倍
    while x < size_hints[0] and conditional_product(x, r) < target:
        x *= 2
    # 如果 r 小于 size_hints[1] 并且 x 和 r 的乘积小于 target，则将 r 扩展两倍
    while r < size_hints[1] and conditional_product(x, r) < target:
        r *= 2

    # 创建一个配置字典 cfg，包含 XBLOCK 和 RBLOCK
    cfg = {"XBLOCK": x, "RBLOCK": r}

    # 如果未提供 num_warps，则根据 x 和 r 的乘积确定 num_warps
    if num_warps is None:
        num_warps = conditional_product(x, r) // 128

    # 对于 AMD GPU，每个 warp 有64个lane，是 NV GPU 的两倍，因此这里使用一半的 num_warps
    default_num_warps = 4 if torch.version.hip else 8
    min_num_warps = 1 if torch.version.hip else 2
    # 将 num_warps 调整为大于等于 min_num_warps 且小于等于 default_num_warps 的最接近的2的幂次方
    num_warps = next_power_of_2(min(max(num_warps, min_num_warps), default_num_warps))

    # 检查配置是否合理，确保 xnumel 符合 size_hints[0] 的要求
    check_config(cfg, xnumel=size_hints[0])
    
    # 断言 r 小于等于 TRITON_MAX_BLOCK["R"]，否则输出错误信息
    assert r <= TRITON_MAX_BLOCK["R"], f"increase TRITON_MAX_BLOCK['r'] to {r}"

    # 返回一个 Config 对象，包含 cfg、num_warps 和 num_stages
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


# 构建一个分块减少 Triton 配置的函数，根据 size_hints 进行一些调整启发。
# size_hints 是每个瓦片维度中的元素数的元组，并将其向上舍入到最接近的2的幂次方。
def triton_config_tiled_reduction(size_hints, x, y, r, num_stages=1):
    target = conditional_product(x, y, r)
    # 如果 size_hints 的乘积小于 target，则将 target 缩小为其八分之一
    if conditional_product(*size_hints) < target:
        target //= 8

    # 缩小 x、y 和 r 到 size_hints 指定的大小
    x = min(x, size_hints[0])
    y = min(y, size_hints[1])
    r = min(r, size_hints[2])

    # 如果 x 小于 size_hints[0] 并且 x、y、r 的乘积小于 target，则将 x 扩展两倍
    while x < size_hints[0] and conditional_product(x, y, r) < target:
        x *= 2
    # 如果 r 小于 size_hints[2] 并且 x、y、r 的乘积小于 target，则将 r 扩展两倍
    while r < size_hints[2] and conditional_product(x, y, r) < target:
        r *= 2
    # 如果 y 小于 size_hints[1] 并且 x、y、r 的乘积小于 target，则将 y 扩展两倍
    while y < size_hints[1] and conditional_product(x, y, r) < target:
        y *= 2

    # 创建一个配置字典 cfg，包含 XBLOCK、YBLOCK 和 RBLOCK
    cfg = {"XBLOCK": x, "YBLOCK": y, "RBLOCK": r}

    # 根据 x、y、r 的乘积确定 num_warps
    num_warps = next_power_of_2(min(max(conditional_product(x, y, r) // 256, 1), 8))

    # 检查配置是否合理，确保 xnumel 和 ynumel 符合 size_hints 的要求
    check_config(cfg, xnumel=size_hints[0], ynumel=size_hints[1])

    # 断言 r 小于等于 TRITON_MAX_BLOCK["R"]，否则输出错误信息
    assert r <= TRITON_MAX_BLOCK["R"], f"increase TRITON_MAX_BLOCK['r'] to {r}"

    # 返回一个 Config 对象，包含 cfg、num_warps 和 num_stages
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


# 基于 size_hints 构建 pointwise 函数，生成 @triton.heuristics() 的配置。
def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
):
    # 如果未提供 inductor_meta，则设为一个空字典
    inductor_meta = {} if inductor_meta is None else inductor_meta

    # 断言 inductor_meta 中没有 "no_x_dim" 键
    assert not inductor_meta.get("no_x_dim")

    # 计算 size_hints 中元素的乘积
    numel = functools.reduce(operator.mul, size_hints)
    bs = max(256, min(numel // 128, 1024))
    # 计算块大小(bs)，取最大值为256，最小值为numel除以128与1024的较小者

    hinted_configs = autotune_hints_to_configs(
        inductor_meta.get("autotune_hints", set()), size_hints, bs
    )
    # 根据自动调优提示、尺寸提示和块大小生成配置列表(hinted_configs)

    triton_config_with_settings = functools.partial(
        triton_config, min_elem_per_thread=min_elem_per_thread
    )
    # 利用偏函数functools.partial创建triton_config_with_settings函数，设定参数min_elem_per_thread

    if len(size_hints) == 1:
        if disable_pointwise_autotuning(inductor_meta) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            # 如果禁用点级别自动调优且未启用最大自动调优或点级最大自动调优，则返回缓存的自动调优结果
            return cached_autotune(
                size_hints,
                [triton_config_with_settings(size_hints, bs)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        else:
            # 否则返回缓存的自动调优结果，包括不同配置选项
            return cached_autotune(
                size_hints,
                [
                    triton_config_with_settings(
                        size_hints, bs, num_elements_per_warp=256
                    ),
                    triton_config_with_settings(
                        size_hints, bs // 2, num_elements_per_warp=64
                    ),
                    *hinted_configs,
                ],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
    if len(size_hints) == 2:
        if (
            disable_pointwise_autotuning(inductor_meta) or tile_hint == TileHint.SQUARE
        ) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            # 如果禁用点级别自动调优或瓦片提示为正方形且未启用最大自动调优或点级最大自动调优，则返回缓存的自动调优结果
            return cached_autotune(
                size_hints,
                [triton_config_with_settings(size_hints, 32, 32)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        # 否则返回缓存的自动调优结果，包括多个不同的配置选项
        return cached_autotune(
            size_hints,
            [
                triton_config_with_settings(size_hints, 32, 32),
                triton_config_with_settings(size_hints, 64, 64),  # ~8% better for fp16
                triton_config_with_settings(size_hints, 256, 16),
                triton_config_with_settings(size_hints, 16, 256),
                triton_config_with_settings(size_hints, bs, 1),
                triton_config_with_settings(size_hints, 1, bs),
                *hinted_configs,
            ],
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    # 如果 size_hints 列表长度为 3
    if len(size_hints) == 3:
        # 如果禁用点积自动调整，返回经缓存的自动调整结果
        if disable_pointwise_autotuning(inductor_meta):
            return cached_autotune(
                size_hints,
                # 使用 triton_config_with_settings 函数生成设置为 (16, 16, 16) 的 Triton 配置
                [triton_config_with_settings(size_hints, 16, 16, 16)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        
        # 否则返回经缓存的自动调整结果，包括多个 Triton 配置
        return cached_autotune(
            size_hints,
            [
                triton_config_with_settings(size_hints, 16, 16, 16),
                triton_config_with_settings(size_hints, 64, 8, 8),
                triton_config_with_settings(size_hints, 8, 64, 8),
                triton_config_with_settings(size_hints, 8, 8, 64),
                triton_config_with_settings(size_hints, bs, 1, 1),
                triton_config_with_settings(size_hints, 1, bs, 1),
                triton_config_with_settings(size_hints, 1, 1, bs),
                *hinted_configs,
            ],
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    
    # 如果 size_hints 列表长度不为 3，则抛出未实现错误，包含 size_hints 的具体值
    raise NotImplementedError(f"size_hints: {size_hints}")
# 定义函数_reduction_configs，返回一组配置列表，用于指定数据减少操作的配置信息
def _reduction_configs(
    *, size_hints: List[int], inductor_meta: Dict[str, Any]
) -> List[Config]:
    # 从inductor_meta中获取减少提示（reduction_hint），默认为None
    reduction_hint = inductor_meta.get("reduction_hint", None)
    # 断言size_hints列表长度为2
    assert len(size_hints) == 2
    # 获取rnumel，即size_hints中的最后一个元素
    rnumel = size_hints[-1]

    # 定义最大的R块大小为2048
    MAX_RBLOCK = 2048
    # 如果size_hints的第一个元素大于等于1024，并且加载和减少的数量之和大于等于10
    if (
        size_hints[0] >= 1024
        and inductor_meta.get("num_load", 0) + inductor_meta.get("num_reduction", 0)
        >= 10
    ):
        # 根据启发式算法调整MAX_RBLOCK，如果内核可能需要多个寄存器，则减小R块大小
        # 考虑到加载需要将数据移到寄存器中，而减少则需要一个累加器。
        #
        # 这些魔法数字有些是任意选择的。
        #
        # 我们不能依赖于后期动态缩减RBLOCK的能力，因为有时triton可能会使用更少的寄存器但性能更差。参考：
        # https://github.com/pytorch/pytorch/issues/126463
        #
        # 这个启发式算法是一个非常简单的算法，因为寄存器可以被重用。但希望它能成为一个足够好的指标。
        MAX_RBLOCK = 1024

    # 创建一个连续配置，用于数据减少操作
    contiguous_config = triton_config_reduction(
        size_hints, 1, (rnumel if 256 <= rnumel < MAX_RBLOCK else MAX_RBLOCK)
    )
    # 创建一个外部配置，用于数据减少操作
    outer_config = triton_config_reduction(size_hints, 64, 8)
    # 创建一个微小配置，用于数据减少操作
    tiny_config = triton_config_reduction(
        size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, min(rnumel, MAX_RBLOCK)
    )
    # 如果存在max_autotune或max_autotune_pointwise，跳过所有这些情况
    if inductor_meta.get("max_autotune") or inductor_meta.get("max_autotune_pointwise"):
        pass  # 跳过所有这些情况
    # 如果reduction_hint为INNER，则返回连续配置的列表
    elif reduction_hint == ReductionHint.INNER:
        return [contiguous_config]
    # 如果reduction_hint为OUTER，则返回外部配置的列表
    elif reduction_hint == ReductionHint.OUTER:
        return [outer_config]
    # 如果reduction_hint为OUTER_TINY，则返回微小配置的列表
    elif reduction_hint == ReductionHint.OUTER_TINY:
        return [tiny_config]
    # 如果禁用点对点自动调整，则返回一个特定的配置列表
    if disable_pointwise_autotuning(inductor_meta):
        return [triton_config_reduction(size_hints, 32, 128)]
    # 默认返回一组默认的配置列表，用于数据减少操作
    return [
        contiguous_config,
        outer_config,
        tiny_config,
        triton_config_reduction(size_hints, 64, 64),
        triton_config_reduction(size_hints, 8, 512),
        # 减半XBLOCK/RBLOCK与outer_config相比
        # TODO: 可能仅在每次减少迭代非常耗时时才有利。例如：https://gist.github.com/shunting314/189a8ef69f90db9d614a823385147a72
        triton_config_reduction(size_hints, 64, 4, num_warps=8),
    ]


# 定义函数reduction，根据输入参数生成一个减少配置的列表
def reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """args to @triton.heuristics()"""
    # 如果inductor_meta为None，则设置为空字典
    inductor_meta = {} if inductor_meta is None else inductor_meta
    # 将reduction_hint设置为inductor_meta中的reduction_hint
    inductor_meta["reduction_hint"] = reduction_hint
    # 如果存在no_x_dim标志，则将size_hints的第一个元素设置为1
    if inductor_meta.get("no_x_dim"):
        size_hints = [1, *size_hints[1:]]

    # 断言triton_meta不为None
    assert triton_meta is not None
    # 获取rnumel，即size_hints中的最后一个元素
    rnumel = size_hints[-1]
    # 断言size_hints列表长度为2
    if len(size_hints) != 2:
        raise NotImplementedError(f"size_hints: {size_hints}")

    # 调用_reduction_configs函数，获取配置列表
    configs = _reduction_configs(size_hints=size_hints, inductor_meta=inductor_meta)
    # 调用 cached_autotune 函数，并传递以下参数：
    # - size_hints: 用于自动调优的大小提示
    # - configs: 配置参数，可能包含调优算法所需的设置
    # - triton_meta: Triton 元数据，可能包含与 Triton 相关的信息
    # - inductor_meta: 电感器元数据，可能包含与电感器相关的信息
    # - heuristic_type: 启发式类型，这里设置为减少类型（HeuristicType.REDUCTION）
    # - filename: 文件名，用于指定要进行自动调优的文件
    
    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.REDUCTION,
        filename=filename,
    )
def persistent_reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    # 如果没有传入inductor_meta，则设为一个空字典
    inductor_meta = {} if inductor_meta is None else inductor_meta
    # 将reduction_hint存入inductor_meta字典中
    inductor_meta["reduction_hint"] = reduction_hint
    # 如果inductor_meta中有"no_x_dim"键，则将size_hints的第一个元素设为1
    if inductor_meta.get("no_x_dim"):
        size_hints = [1, *size_hints[1:]]

    # 将size_hints解包为xnumel和rnumel
    xnumel, rnumel = size_hints

    # 生成一系列的triton配置列表，其中xblock分别为1, 8, 32, 128，满足条件的才会被保留
    configs = [
        triton_config_reduction(size_hints, xblock, rnumel)
        for xblock in (1, 8, 32, 128)
        if xblock == 1 or (rnumel * xblock <= 4096 and xblock <= xnumel)
    ]

    # TODO(jansel): we should be able to improve these heuristics
    # 根据reduction_hint的不同类型进行进一步的配置修正
    if reduction_hint == ReductionHint.INNER and rnumel >= 256:
        configs = configs[:1]
    elif reduction_hint == ReductionHint.OUTER:
        configs = configs[-1:]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        configs = [
            triton_config_reduction(
                size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, rnumel
            )
        ]
    
    # 对configs中的每个配置，移除"RBLOCK"参数
    for c in configs:
        c.kwargs.pop("RBLOCK")

    # 如果disable_pointwise_autotuning函数返回True，则只保留configs中的第一个配置
    if disable_pointwise_autotuning(inductor_meta):
        configs = configs[:1]

    # 调用cached_autotune函数进行缓存自动调优，返回结果
    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
    )


def split_scan(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """Heuristic for TritonSplitScanKernel"""
    # 如果没有传入inductor_meta，则设为一个空字典
    inductor_meta = {} if inductor_meta is None else inductor_meta
    # 将reduction_hint存入inductor_meta字典中
    inductor_meta["reduction_hint"] = reduction_hint
    # 如果inductor_meta中有"no_x_dim"键，则将size_hints的第一个元素设为1
    if inductor_meta.get("no_x_dim"):
        size_hints = [1, *size_hints[1:]]

    # 断言triton_meta不为None，若为None则抛出异常
    assert triton_meta is not None
    # 若size_hints的长度不为2，则抛出NotImplementedError异常
    if len(size_hints) != 2:
        raise NotImplementedError(f"size_hints: {size_hints}")

    # 生成一系列的减少扫描配置列表，根据size_hints和inductor_meta进行配置
    configs = _reduction_configs(size_hints=size_hints, inductor_meta=inductor_meta)

    # 获取inductor_meta中指定的最小RBLOCK值，若没有指定则为256
    min_rblock = inductor_meta.get("min_split_scan_rblock", 256)
    # 对configs中的每个配置，如果RBLOCK小于min_rblock，则设为min_rblock
    for cfg in configs:
        if cfg.kwargs["RBLOCK"] < min_rblock:
            cfg.kwargs["RBLOCK"] = min_rblock

    # 调用cached_autotune函数进行缓存自动调优，返回结果
    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.SPLIT_SCAN,
        filename=filename,
    )


def template(num_stages, num_warps, triton_meta, filename=None, inductor_meta=None):
    """
    Compile a triton template
    """
    # 调用cached_autotune函数进行缓存自动调优，返回结果
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=num_stages, num_warps=num_warps)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


def user_autotune(
    configs, triton_meta, filename=None, inductor_meta=None, custom_kernel=False
):
    """
    Autotune based on user-provided configurations
    """
    # 此函数功能为基于用户提供的配置进行自动调优，具体操作由cached_autotune函数完成
    # 编译用户定义的 Triton 内核
    """
    # 获取 triton.Config 的默认参数列表
    defaults = inspect.signature(triton.Config).parameters
    # 获取默认的 num_stages 参数值
    default_num_stages = defaults["num_stages"].default
    # 获取默认的 num_warps 参数值
    default_num_warps = defaults["num_warps"].default

    # 如果 configs 列表为空，则使用默认配置创建一个 triton.Config 对象列表
    if len(configs) == 0:
        configs = [
            triton.Config(
                {}, num_stages=default_num_stages, num_warps=default_num_warps
            )
        ]
    else:
        # 否则，根据 configs 列表中的每个元素创建对应的 triton.Config 对象列表
        configs = [
            triton.Config(
                c.get("kwargs", {}),
                num_stages=c.get("num_stages", default_num_stages),
                num_warps=c.get("num_warps", default_num_warps),
            )
            for c in configs
        ]

    # 调用 cached_autotune 函数，执行自动调优过程
    return cached_autotune(
        None,
        configs,
        triton_meta=triton_meta,
        heuristic_type=HeuristicType.USER_AUTOTUNE,
        filename=filename,
        inductor_meta=inductor_meta,
        custom_kernel=custom_kernel,
    )
# 定义一个函数 foreach，用于编译 triton 的 foreach 内核
def foreach(triton_meta, num_warps, filename=None, inductor_meta=None):
    """
    Compile a triton foreach kernel
    """
    # 调用 cached_autotune 函数，传入必要参数和配置，返回编译后的内核
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=1, num_warps=num_warps)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


# 定义一个辅助函数 grid，用于计算 triton 网格
def grid(*numels):
    """Helper function to compute triton grids"""
    # 根据参数长度决定 xnumel, ynumel, znumel 的赋值
    if len(numels) == 1:
        xnumel, ynumel, znumel = numels[0], None, None
    elif len(numels) == 2:
        xnumel, ynumel, znumel = numels[1], numels[0], None
    elif len(numels) == 3:
        xnumel, ynumel, znumel = numels[2], numels[1], numels[0]
    else:
        # 抛出断言错误，表示不支持的 numels 大小
        raise AssertionError(f"invalid size for numels {len(numels)}")

    # 定义内部函数 get_grid_dim，用于计算网格维度
    def get_grid_dim(numel, block):
        if numel is None:
            return 1
        if block is None:
            return numel
        return ceildiv(numel, block)

    # 定义 grid_fn 函数，根据 meta 数据计算并返回三个维度的网格
    def grid_fn(meta):
        x_grid = get_grid_dim(xnumel, meta.get("XBLOCK", 1))
        y_grid = get_grid_dim(ynumel, meta.get("YBLOCK", None))

        # 调用 get_max_y_grid 函数获取最大的 y 网格数
        max_y_grid = get_max_y_grid()
        if znumel is None:
            # 计算 y 和 z 网格数，确保不超出支持范围
            div = ceildiv(y_grid, max_y_grid)
            y_grid = ceildiv(y_grid, div)
            z_grid = div
        else:
            # 计算 z 网格数，并检查 y 网格数是否在支持范围内
            z_grid = get_grid_dim(znumel, meta.get("ZBLOCK", None))
            torch._check(
                y_grid <= max_y_grid,
                lambda: f"Generated y grid beyond 2^16 ({y_grid}) not supported with z dimension present. File issue",
            )

        return (
            x_grid,
            y_grid,
            z_grid,
        )

    # 设置 grid_fn 的属性 grid_fn_str，用于标识该函数的字符串表示
    setattr(grid_fn, "grid_fn_str", f"grid({numels})")  # noqa: B010

    return grid_fn


# 定义 split_scan_grid 函数，用于计算分裂扫描的 triton 网格
def split_scan_grid(xnumel, rnumel):
    def grid_fn(meta):
        # 断言 XBLOCK 等于 1，确保只有一个线程块
        assert meta.get("XBLOCK", 1) == 1
        # 返回网格维度元组，包含 rnumel 的块数和 xnumel
        return (ceildiv(rnumel, meta.get("RBLOCK", 1)), xnumel, 1)

    # 设置 grid_fn 的属性 grid_fn_str，用于标识该函数的字符串表示
    grid_fn_str = f"split_scan_grid({xnumel}, {rnumel})"
    setattr(grid_fn, "grid_fn_str", grid_fn_str)  # noqa: B010

    return grid_fn
```