# `.\pytorch\torch\_inductor\config.py`

```py
# mypy: allow-untyped-defs
# 引入os模块，并告知 linter 忽略 C101 规则
import os  # noqa: C101
# 引入sys模块
import sys
# 引入类型提示相关模块
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING, Union

# 引入torch模块
import torch

# 检查当前是否在 FBCode 环境中
def is_fbcode():
    return not hasattr(torch.version, "git_version")

# 设置默认的 FX 图远程缓存状态
def fx_graph_remote_cache_default():
    # 检查环境变量是否设置了 TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE
    if os.environ.get("TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE") == "1":
        return True
    if os.environ.get("TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE") == "0":
        return False
    return None

# 添加一些调试输出
debug = False

# 是否禁用自动调优进度条
disable_progress = True

# 是否启用打印每个未来的源代码
verbose_progress = False

# 使用 FX AOT 图代码生成缓存
fx_graph_cache = os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE") == "1"

# 使用远程 FX AOT 图代码生成缓存
# False: 禁用缓存
# True: 启用缓存
# None: 未设置 -- 对于 OSS 关闭，对于内部基于 JustKnobs
fx_graph_remote_cache: Optional[bool] = fx_graph_remote_cache_default()

# 启用自动调优本地缓存
autotune_local_cache = True

# 启用自动调优远程缓存
autotune_remote_cache = os.environ.get("TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE") == "1"

# 强制禁用所有引导器级别的缓存 -- 这将覆盖任何其他缓存标志
force_disable_caches = os.environ.get("TORCHINDUCTOR_FORCE_DISABLE_CACHES") == "1"

# 使用 CPP 包装器而不是 Python 包装器
cpp_wrapper = os.environ.get("TORCHINDUCTOR_CPP_WRAPPER", "0") == "1"

# 在 ABI 兼容模式下生成 CPP 包装器代码
abi_compatible = (
    os.environ.get("TORCHINDUCTOR_ABI_COMPATIBLE", "1" if is_fbcode() else "0") == "1"
)

# C 语言接口版本
c_shim_version = os.environ.get(
    "TORCHINDUCTOR_C_SHIM_VERSION", "1" if is_fbcode() else "2"
)

# 死代码消除
dce = False

# 假定权重张量具有固定大小
static_weight_shapes = True

# 在生成的代码中放置正确性断言
size_asserts = os.environ.get("TORCHINDUCTOR_SIZE_ASSERTS", "1") == "1"
nan_asserts = os.environ.get("TORCHINDUCTOR_NAN_ASSERTS") == "1"

# 基于输入顺序启用循环重排
pick_loop_orders = True

# 将输入用作输出的内核输入复用
inplace_buffers = True

# 为不是输出的张量启用池化分配
memory_planning = os.environ.get("TORCHINDUCTOR_MEMORY_PLANNING", "0") == "1"

# 在 memory_planning=True 时如何组织内存:
# - "none": 不尝试池化存储，只重用
# - "intermediates": 所有非输出共享存储，每个输出都有唯一存储
# - "outputs": 两个池，一个用于中间结果（返回时释放），一个用于输出
# - "combined": 单个池用于中间结果和输出
memory_pool = os.environ.get("TORCHINDUCTOR_MEMORY_POOL", "intermediates")

# 代码生成基准测试工具
benchmark_harness = True

# 将逐点操作融合到模板中
epilogue_fusion = True

# 在其他融合之前执行后记融合
epilogue_fusion_first = False

# 启用模式匹配+替换优化
pattern_matcher = True

# 设置一个布尔变量，用于标识是否启用模式匹配器

# register custom graph optimization pass hook. so far, pre/post passes are
# only applied before/after pattern_matcher in post_grad_passes.
#
# def my_custom_pre_pass(graph: torch.fx.graph.Graph):
#     # my custom graph optimization pass
#     ...
#
# def my_custom_post_pass(graph: torch.fx.graph.Graph):
#     # my custom graph optimization pass
#     ...
#
# torch._inductor.config.post_grad_custom_pre_pass = my_custom_pre_pass
# torch._inductor.config.post_grad_custom_post_pass = my_custom_post_pass
post_grad_custom_pre_pass: Optional[Callable[[torch.fx.graph.Graph], None]] = None
post_grad_custom_post_pass: Optional[Callable[[torch.fx.graph.Graph], None]] = None

# 注册自定义的图优化传递钩子。目前，在 post_grad_passes 中只有前后两个传递会在 pattern_matcher 之前/之后应用。

# Registers a custom joint graph pass.
joint_custom_pre_pass: Optional[Callable[[torch.fx.Graph], None]] = None
joint_custom_post_pass: Optional[Callable[[torch.fx.Graph], None]] = None

# 注册自定义的联合图传递钩子。

# Registers a custom pregrad pass. Note that the pre-grad IR is 1.
# non-functional, 2. non-normalized, and 3. prone to change. Ideally we should
# use post-grad passes.
pre_grad_custom_pass: Optional[Callable[[torch.fx.graph.Graph], None]] = None

# 注册自定义的 pregrad 传递。请注意，pregrad IR 是非功能性的、非标准化的，并且容易变化。理想情况下应该使用 post-grad 传递。

# Deprecated
split_cat_fx_passes = True

# 标记 split_cat_fx_passes 已被弃用

# Optimize conv-batchnorm if batchnorm is in eval mode. Slightly reduces numerical stability.
efficient_conv_bn_eval_fx_passes = False

# 如果 batchnorm 处于评估模式，则优化 conv-batchnorm。会略微降低数值稳定性。

# Enable predispatch aten IR for export
is_predispatch = False

# 启用预调度 aten IR 用于导出

# Deprecated
group_fusion = False

# 标记 group_fusion 已被弃用

# Deprecated
batch_fusion = True

# 标记 batch_fusion 已被弃用

# Pre grad fusion and options in order, set to empty dict to disable fusion.
# Call `torch._inductor.fx_passes.group_batch_fusion.list_group_batch_fusions()` to see available fusions.
# batch fusion options:
# batch_linear
# batch_linear_lhs
# batch_layernorm
# batch_tanh
# batch_relu
# batch_sigmoid
pre_grad_fusion_options: Dict[str, Dict[str, Any]] = {
    "batch_linear": {},
    "batch_linear_lhs": {},
    "batch_layernorm": {},
    "batch_tanh": {},
    "batch_relu": {},
    "batch_sigmoid": {},
}

# 预先梯度融合及其选项，按顺序排列，设置为空字典以禁用融合。
# 调用 `torch._inductor.fx_passes.group_batch_fusion.list_group_batch_fusions()` 查看可用的融合选项。

# Post grad fusion and options, set to empty dict to disable fusion.
# Call `torch._inductor.fx_passes.group_batch_fusion.list_group_batch_fusions(False)` to see available fusions.
post_grad_fusion_options: Dict[str, Dict[str, Any]] = {}

# 后梯度融合及其选项，设置为空字典以禁用融合。
# 调用 `torch._inductor.fx_passes.group_batch_fusion.list_group_batch_fusions(False)` 查看可用的融合选项。

# enable reordering pass for improving memory locality
reorder_for_locality = True

# 启用重新排序传递以改善内存局部性

# Scale down RBLOCK for better occupancy
dynamic_scale_rblock = os.environ.get("TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK", "1") == "1"

# 为了提高占用率而缩减 RBLOCK 的规模

# this forces fusion for int_mm with mul. Needed when you want to avoid realizing the int32
# but the mul gets fused with other pointwise ops instead.
force_fuse_int_mm_with_mul = False

# 这强制对 int_mm 和 mul 进行融合。在不想实现 int32 但 mul 要与其他逐点操作融合时需要。

# for pattern torch.mm(a, b.to(dtype)) with cuda tensors,
# enable torch._inductor.kernel.mm.tuned_mixed_mm fused kernel.
# Autotune will compare perf with normal cast->then->mm option
use_mixed_mm = True

# 对于具有 cuda 张量的 torch.mm(a, b.to(dtype)) 模式，
# 启用 torch._inductor.kernel.mm.tuned_mixed_mm 融合内核。
# 自动调优将比较与正常 cast->then->mm 选项的性能。
# 启用运行时数字检查以用于前/后梯度传递
# 浮点数提供有限的精度（单精度浮点数大约为7位小数，双精度浮点数大约为16位小数）
# 根据PyTorch文档。
# https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations
fx_passes_numeric_check: Dict[str, Any] = {
    "pre_grad": False,                  # 控制是否启用前梯度传递
    "precision": 1e-4,                  # 数字精度阈值
    "num_iterations": 1,                # 迭代次数
    "requires_optimizer": True,         # 是否需要优化器
}

# mixed_mm_choice用于控制模式torch.mm(a, b.to(dtype))在CUDA张量上的行为。
# 回退的aten实现是常规的类型转换->mm选项。
# 如果mixed_mm_choice是"default"：此标志将被忽略。
# 如果mixed_mm_choice是"triton"：
# - 总是使用torch._inductor.kernel.mm.tuned_mixed_mm的融合核心。
# - 自动调优不会与回退进行比较。
# 如果mixed_mm_choice是"aten"：总是使用回退的aten实现。
# 如果mixed_mm_choice是"heuristic"：
# - 启用启发式算法。
# - 如果启发式算法决定添加一个配置，则将其作为首选项。
# - 如果自动调优被禁用，则始终选择此配置。
# - 如果自动调优被启用，则还将与回退的aten实现和融合核心进行比较。
# 如果mixed_mm_choice != "default"，将忽略use_mixed_mm标志。
mixed_mm_choice = "heuristic"

# 启用重新排序传递以增加计算和通信之间的重叠
reorder_for_compute_comm_overlap = False

# 用于增加计算和通信之间重叠的传递（执行顺序）
# 对于内置传递，使用字符串名称；对于用户定义的传递，传递函数句柄
reorder_for_compute_comm_overlap_passes = [
    "reorder_compute_for_overlap",     # 重新排序以增加重叠
    "sink_waits",                      # 下沉等待
    "raise_comms",                     # 提升通信
]

# 运行时操作估计函数
# 对于内置估计函数，传入"default"；对于用户定义的估计函数，传递函数句柄
estimate_op_runtime = "default"

# 单位：GB/s，单向P2P卡间带宽
# 默认值为NVLink
intra_node_bw = 300

# 单位：GB/s，单向P2P节点间带宽
# 默认值为InfiniBand
inter_node_bw = 25

# 启用慢速自动调优传递以选择算法
max_autotune = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1"

# 启用慢速自动调优传递以选择逐点/减少算法
max_autotune_pointwise = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE") == "1"

# 启用慢速自动调优传递以选择GEMM算法
max_autotune_gemm = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM") == "1"

# 强制cublas和triton使用相同的精度；cublas支持TF32用于矩阵乘法操作
# 当m、n、k是16的倍数时，而triton支持TF32用于矩阵乘法操作
# 无论它们的对齐如何。设置此标志将确保
# 是否强制在 Triton 不使用 TF32 的情况下不使用 TF32
force_same_precision = (
    True if is_fbcode() else os.environ.get("TORCHINDUCTOR_FORCE_SAME_PRECISION") == "1"
)

# 指定用于 GEMM 自动调优的候选后端
# 可能的选择组合包括：ATen, Triton, CUTLASS, CK, CPP
max_autotune_gemm_backends = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON,CPP"
).upper()

# 指定 GEMM 自动调优的搜索空间大小
# DEFAULT     - 平衡编译时间开销和性能
# EXHAUSTIVE  - 最大化性能
max_autotune_gemm_search_space = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE", "DEFAULT"
).upper()

# 在自动调优中未找到匹配项时，是否回退到 ATen 或硬错误
autotune_fallback_to_aten = (
    os.environ.get("TORCHINDUCTOR_AUTOTUNE_FALLBACK_TO_ATEN", "1") == "1"
)

# 未支持的 SymInts 的回退值，可能出现在输入形状中（例如在自动调优中）
unbacked_symint_fallback = 8192

# 是否启用全局和本地缓存的搜索，不考虑 `max_autotune`
search_autotune_cache = os.environ.get("TORCHINDUCTOR_SEARCH_AUTOTUNE_CACHE") == "1"

# 是否保存参数
save_args = os.environ.get("TORCHINDUCTOR_SAVE_ARGS") == "1"

# 如果为 False，则禁用为自动调优创建子进程
autotune_in_subproc = os.environ.get("TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC") == "1"

# 如果启用自动调优子进程，则以下三个超时设置生效
# 自动调优子进程的最大结果超时时间
max_autotune_subproc_result_timeout_seconds = 60.0
# 超时后允许子进程优雅终止的额外时间
max_autotune_subproc_graceful_timeout_seconds = 1.0
# 发送 SIGTERM 后直到发送 SIGKILL 前的额外时间
max_autotune_subproc_terminate_timeout_seconds = 2.0

# 如果在子进程中进行自动调优，是否使用多个设备
autotune_multi_device = os.environ.get("TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE") == "1"

# 是否进行坐标下降调优
coordinate_descent_tuning = (
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING") == "1"
)

# 在坐标下降调优中是否检查所有方向
coordinate_descent_check_all_directions = (
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS") == "1"
)

# 坐标下降调优的搜索半径
coordinate_descent_search_radius = int(
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS", "1")
)

# ROCm 上默认禁用，如果模型使用 NHWC 卷积，则启用布局优化
layout_opt_default = "1" if not torch.version.hip else "0"
layout_optimization = (
    os.environ.get("TORCHINDUCTOR_LAYOUT_OPTIMIZATION", layout_opt_default) == "1"
)
# 从环境变量中获取是否强制进行布局优化的设置，如果设置为 "1" 则为 True，否则为 False
force_layout_optimization = os.environ.get("TORCHINDUCTOR_FORCE_LAYOUT_OPT", "0") == "1"

# 从环境变量中获取是否保持输出步幅与 eager 模式一致的设置，如果设置为 "1" 则为 True，否则为 False
keep_output_stride = os.environ.get("TORCHINDUCTOR_KEEP_OUTPUT_STRIDE", "1") == "1"

# 从环境变量中获取是否打印警告信息，如果设置为 "1" 则为 True，否则为 False
warn_mix_layout = os.environ.get("TORCHINDUCTOR_WARN_MIX_LAYOUT") == "1"

# 控制 store vs recompute 策略的阈值：对于 fanouts，重新计算可能导致指数级增长，所以设置较小的阈值
realize_reads_threshold = 4
realize_opcount_threshold = 30

# 用于降低时防止操作在一个缓冲区中过度积累的阈值
realize_acc_reads_threshold = 8

# 是否启用对随机数和 dropout 操作的回退到 eager 模式，用于调试目的，通常设置为 False
fallback_random = False

# 是否在遇到未处理操作时自动创建回退，通常设置为 True
implicit_fallbacks = True

# 是否进行更激进的融合，通常设置为 False
aggressive_fusion = False

# 是否启用融合调试模式，从环境变量中获取设置，如果设置为 "1" 则为 True，否则为 False
debug_fusion = os.environ.get("TORCHINDUCTOR_DEBUG_FUSION") == "1"
# 是否启用融合性能基准测试，从环境变量中获取设置，如果设置为 "1" 则为 True，否则为 False
benchmark_fusion = os.environ.get("TORCHINDUCTOR_BENCHMARK_FUSION") == "1"

# 从环境变量中获取启用的度量表名称列表，用逗号分隔
enabled_metric_tables = os.environ.get("TORCHINDUCTOR_ENABLED_METRIC_TABLES", "")

# 是否在 Triton 模板中启用最佳模板加 epilogue 和最佳模板加独立 epilogue 内核的基准性能测试
benchmark_epilogue_fusion = (
    os.environ.get("TORCHINDUCTOR_BENCHMARK_EPILOGUE_FUSION", "1") == "1"
)

# 用于基准测试 epilogue 的顶部 Triton 内核数目
max_epilogue_benchmarked_choices = 1

# 允许单个融合中的最大节点数
max_fusion_size = 64

# 用于点对点操作中生成 cat 的最大输入数
max_pointwise_cat_inputs = 8

# 小型 reduction 替换为 pointwise 操作的阈值，设置为 1 以禁用
unroll_reductions_threshold = 8

# 是否在输出代码中添加额外的注释，可能会导致编译缓存未命中，通常设置为 False
comment_origin = False

# 是否将 1x1 卷积转换为矩阵乘法
conv_1x1_as_mm = False

# 是否启用分割 reduction 以提高利用率，通常设置为 True
split_reductions = True

# 从环境变量中获取是否进行内核基准测试的设置，如果设置为 "1" 则为 True，否则为 False
benchmark_kernel = os.environ.get("TORCHINDUCTOR_BENCHMARK_KERNEL", "0") == "1"

# 是否启用常量和索引表达式的折叠
constant_and_index_propagation = True

# 是否始终将常量添加到 graph.constants 而不执行任何常量内联优化
always_keep_tensor_constants = False

# 是否断言间接索引不会读写超出界限
assert_indirect_indexing = True

# 是否对在 FX 图中不出现的变量计算 CSE 边界
compute_all_bounds = False

# 是否在联合图上进行常量折叠
joint_graph_constant_folding = True
# Enable indirect_indexing asserts for decompositions and lowerings
debug_index_asserts = False

# warnings intended for PyTorch developers, disable for point releases
is_nightly_or_source = "dev" in torch.__version__ or "git" in torch.__version__
developer_warnings = is_fbcode() or is_nightly_or_source

# This pattern matches a special usage of scatter
# 1. It's applied to a constant tensor
# 2. The index tensor has size 1 in the scatter dimension
# Such pattern generates a sparse matrix when the const tensor is all-zero.
# We can lower this pattern to a pointwise kernel for more fusion opportunities
# and saving memory footprint.
optimize_scatter_upon_const_tensor = (
    os.environ.get("TORCHINDUCTOR_OPTIMIZE_SCATTER_UPON_CONST_TENSOR", "1") == "1"
)

# The multiprocessing start method to use for inductor workers in the codecache.
# "subprocess", "fork", or "spawn"
def decide_worker_start_method():
    start_method = os.environ.get(
        "TORCHINDUCTOR_WORKER_START", "fork" if is_fbcode() else "subprocess"
    )
    assert start_method in [
        "subprocess",
        "fork",
        "spawn",
    ], f"Invalid start method: {start_method}"
    return start_method

worker_start_method = decide_worker_start_method()

# Flags to turn on all_reduce fusion. These 2 flags should be automaticaly turned
# on by DDP and should not be set by the users.
_fuse_ddp_communication = False
_fuse_ddp_bucket_size = 25

# Flag to control which fusion passes to apply. Functions in the list will
# be applied in order. There are two different different fusion passes
# --"fuse_ddp_with_concat_op" and "fuse_ddp_with_coalesced_op". The default
# one is "fuse_ddp_with_concat_op". Users can also change this to a customized
# fusion function.
#
# The fusion currently does not support multiple DDP with different PG or
# data type. This feature will be added in the future PRs.
#
# "schedule_comm_wait" is used to delay the wait ops to maximize comm/comp
# overlapping. At this moment, this pass performs better than
# reorder_for_compute_comm_overlap_passes but we will add the logic of
# "schedule_comm_wait" in the future and remove the one here.
_fuse_ddp_communication_passes: List[Union[Callable[..., None], str]] = [
    "fuse_ddp_with_concat_op",
    "schedule_comm_wait",
]

_micro_pipeline_tp: bool = False

def decide_compile_threads():
    """
    Here are the precedence to decide compile_threads
    1. User can override it by TORCHINDUCTOR_COMPILE_THREADS.  One may want to disable async compiling by
       setting this to 1 to make pdb happy.
    2. Set to 1 if it's win32 platform
    3. decide by the number of CPU cores
    """
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        return int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
    elif sys.platform == "win32":
        return 1
    elif is_fbcode() and worker_start_method != "subprocess":
        return 1
    else:
        # 如果操作系统支持获取CPU亲和力信息，则使用该方法获取CPU核心数
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count()
        )
        # 确保获取的CPU核心数大于0，即至少有一个CPU核心
        assert cpu_count
        # 返回CPU核心数的最小值，限制在32个核心以内
        return min(32, cpu_count)
# 决定编译线程数量
compile_threads = decide_compile_threads()

# 如果运行在 Facebook 代码库中
if is_fbcode():
    # 从 libfb.py 中导入 parutil 模块
    from libfb.py import parutil
    
    try:
        # 如果有 __package__ 属性，则根据其构建全局缓存目录路径
        if __package__:
            global_cache_dir = parutil.get_dir_path(
                os.path.join(__package__.replace(".", os.sep), "fb/cache")
            )
        else:
            # 否则直接构建全局缓存目录路径
            global_cache_dir = parutil.get_dir_path("fb/cache")
    except ValueError:
        # 如果发生 ValueError，则将全局缓存目录设置为 None
        global_cache_dir = None
else:
    # 如果不在 Facebook 代码库中，则全局缓存目录设置为 None
    global_cache_dir = None

# 如果 kernel 是融合的，从原始节点操作名称生成名称，限制在较大的内核上
kernel_name_max_ops = 10

# 用于控制是否对 matmul/bmm/addmm 的输入张量进行填充，以利用 NVIDIA GPU 上的 Tensor Cores
shape_padding = os.environ.get("TORCHINDUCTOR_SHAPE_PADDING", "1") == "1"

# 控制是否为 pointwise/reductions 进行填充
comprehensive_padding = (
    os.environ.get("TORCHINDUCTOR_COMPREHENSIVE_PADDING", "1") == "1"
)
pad_channels_last = False

# 将反向图的输出视为用户可见
bw_outputs_user_visible = True

# 如果启用且可能，始终使用形状填充
force_shape_pad: bool = False

# Fx-based linear/matmul/bmm + permute/transpose 垂直融合
permute_fusion = os.environ.get("TORCHINDUCTOR_PERMUTE_FUSION", "0") == "1"

# 在 PyTorch 分析器中标记包装器调用
profiler_mark_wrapper_call = False

# 为可以与原始 FX 图中的中间结果进行关联的每个中间结果生成挂钩调用到 torch._inductor.hooks.run_intermediate_hooks
generate_intermediate_hooks = False

# 在 IRNode 上填充 traceback 字段，用于调试为什么 origin_node 没有填充或查找 IRNode 构造的位置
debug_ir_traceback = False

# 用于调试，确保配置正确设置
_raise_error_for_testing = False

# 从环境变量中获取 TORCHINDUCTOR_PROFILE，用于判断是否启用带宽分析
_profile_var = os.environ.get("TORCHINDUCTOR_PROFILE", "")
profile_bandwidth = _profile_var != ""
profile_bandwidth_regex = "" if _profile_var == "1" else _profile_var
# 指定一个文件来打印分析结果，如果为 None 则不将结果写入文件
profile_bandwidth_output = os.environ.get("TORCHINDUCTOR_PROFILE_OUTPUT", None)

# TODO: 以后移除
# 禁用 CPP 代码生成
disable_cpp_codegen = False

# 冻结操作将尝试将权重内联为优化中的常量，并对它们运行常量折叠和其他优化。冻结后，权重将无法更新。
freezing: bool = os.environ.get("TORCHINDUCTOR_FREEZING", "0") == "1"

# 使冻结操作失效 nn 模块的 eager Parameters，以避免可能保留多个权重副本的内存开销
freezing_discard_parameters: bool = False

# 允许为临时张量分配堆栈数组。应该在打开和关闭此标志的情况下运行测试，以确保覆盖率。
allow_stack_allocation: bool = (
    # 获取环境变量"TORCHINDUCTOR_STACK_ALLOCATION"的值，如果该变量未设置，则根据is_fbcode()函数的返回值确定默认值为"1"或"0"。
    os.environ.get("TORCHINDUCTOR_STACK_ALLOCATION", "1" if is_fbcode() else "0") == "1"
# 启用备用的 DSO 接口（“minimal ArrayRef interface”），旨在通过牺牲通用性来最大化性能，适用于特定的使用案例。
# 简要来说：
# - 输入和输出是 ArrayRefTensor<T>（注意需要步长，但张量必须是连续的）
# - 常量处理方式不变，因为它不是每次推断迭代的瓶颈
#
# 在此模式下生成 DSO 时，通常接口仍然支持，但性能可能会降低。
use_minimal_arrayref_interface: bool = False

# 将一些内存限制的矩阵乘法（matmul/bmm）分解为乘法操作
decompose_mem_bound_mm: bool = False

# assume_aligned_inputs 表示我们假设输入已经对齐；我们根据此假设生成代码，并在使用前克隆未对齐的张量。
# 在常见情况下，大多数输入将是对齐的。
assume_aligned_inputs: bool = False


# 针对 codegen/cpp.py 的配置
class cpp:
    # 设置为 torch.get_num_threads()
    threads = -1

    # 当条件不满足时，不生成循环，例如：
    # for(long i0=4096; i0<4096; i0+=1)
    no_redundant_loops = (
        os.environ.get("TORCHINDUCTOR_CPP_NO_REDUNDANT_LOOPS", "1") == "1"
    )

    # 假设线程数是动态的，不专门化线程数。
    # 当开启此标志时，内核在线程数更改时不会重新编译。
    # 对于单线程工作负载，开启它会稍微降低性能。
    dynamic_threads = os.environ.get("TORCHINDUCTOR_CPP_DYNAMIC_THREADS", "0") == "1"

    simdlen: Optional[int] = None
    min_chunk_size = int(os.environ.get("TORCHINDUCTOR_CPP_MIN_CHUNK_SIZE", "4096"))
    cxx = (
        None,  # 如果安装了 conda，从 conda-forge 下载 gcc12
        # "g++-12",
        # "g++-11",
        # "g++-10",
        # "clang++",
        os.environ.get("CXX", "clang++" if sys.platform == "darwin" else "g++"),
        # "g++.par",
    )
    # 允许通过 PyTorch 分析器进行内核性能分析
    enable_kernel_profile = (
        os.environ.get("TORCHINDUCTOR_CPP_ENABLE_KERNEL_PROFILE", "0") == "1"
    )

    # 启用权重预打包以获得更好的性能；可能导致内存占用大
    weight_prepack = os.environ.get("TORCHINDUCTOR_CPP_WEIGHT_PREPACK", "1") == "1"

    # 向我们的 relu 实现注入一个 bug；用于测试我们的重现提取和简化功能。
    # 有效值："compile_error"，"runtime_error"，"accuracy"
    inject_relu_bug_TESTING_ONLY: Optional[str] = None
    inject_log1p_bug_TESTING_ONLY: Optional[str] = None

    # 如果为 None，则自动检测是否可以使用 AVX512/AVX2。否则，按指定强制使用，不进行测试。
    vec_isa_ok: Optional[bool] = None

    # 类似于 config.triton.descriptive_names
    descriptive_names = "original_aten"

    # 允许单个水平融合中的节点数量
    # 从环境变量中获取 TORCHINDUCTOR_CPP_MAX_HORIZONTAL_FUSION_SIZE 的值，并将其转换为整数赋给 max_horizontal_fusion_size
    max_horizontal_fusion_size = int(
        os.environ.get("TORCHINDUCTOR_CPP_MAX_HORIZONTAL_FUSION_SIZE", "16")
    )
    
    # 如果环境变量 TORCHINDUCTOR_CPP_FALLBACK_SCATTER_REDUCE_SUM 的值为 "1"，则将 fallback_scatter_reduce_sum 设置为 True，用于在 reduce 操作为 sum 时回退到 scatter_reduce，以避免性能退化和原子加法的问题
    fallback_scatter_reduce_sum = (
        os.environ.get("TORCHINDUCTOR_CPP_FALLBACK_SCATTER_REDUCE_SUM", "1") == "1"
    )
    
    # 如果环境变量 TORCHINDUCTOR_CPP_ENABLE_UNSAFE_MATH_OPT_FLAG 的值为 "1"，则将 enable_unsafe_math_opt_flag 设置为 True，启用不安全的数学优化标志（funsafe-math-optimizations）
    enable_unsafe_math_opt_flag = (
        os.environ.get("TORCHINDUCTOR_CPP_ENABLE_UNSAFE_MATH_OPT_FLAG", "0") == "1"
    )
    
    # 如果环境变量 TORCHINDUCTOR_CPP_ENABLE_FLOATING_POINT_CONTRACT_FLAG 的值为 "1"，则将 enable_floating_point_contract_flag 设置为 True，启用浮点数合同标志（ffp-contract）
    enable_floating_point_contract_flag = (
        os.environ.get("TORCHINDUCTOR_CPP_ENABLE_FLOATING_POINT_CONTRACT_FLAG", "0")
        == "1"
    )
# config specific to codegen/triton.py
class triton:
    # 是否在输出代码中使用 cudagraphs
    cudagraphs = os.environ.get("TORCHINDUCTOR_CUDAGRAPHS") == "1"

    # 如果 cudagraphs 为 True，则使用 cudagraph 树进行内存池化
    cudagraph_trees = True

    # 是否跳过具有动态形状输入的 cudagraph 图
    # 如果为 False，则对每组唯一的形状输入重新记录图形
    cudagraph_skip_dynamic_graphs = False

    # 在稳定状态下，断言不在快速路径上
    slow_path_cudagraph_asserts = True

    # TODO - 需要调试为什么这会阻止清理
    cudagraph_trees_history_recording = False

    # 如果不在 fbcode 中，启用 cudagraph 支持来自先前 cudagraph 池的突变输入
    cudagraph_support_input_mutation = False if is_fbcode() else True

    # cudagraph 调用后是否强制同步
    force_cudagraph_sync = False

    # 总是在急切热身阶段运行 cudagraphs，而不是记录和执行 cudagraphs
    force_cudagraphs_warmup = False

    # 在快速路径上启用断言
    fast_path_cudagraph_asserts = False

    # 跳过 cudagraph 树的热身
    skip_cudagraph_warmup = False

    # 在每个编译图之前和之后同步
    debug_sync_graph = False

    # 在每次内核启动后同步，以帮助确定错误
    debug_sync_kernel = False

    # 总是加载完整块（而不是在块内部进行广播）
    dense_indexing = False

    # 限制平铺维度
    max_tiles = 2

    # 使用 triton.autotune 用于具有复杂布局的点操作
    # 这应仅在调试/测试时禁用
    autotune_pointwise = True

    # 使用 cublasLt 进行最大的 autotune gemm
    autotune_cublasLt = True

    # 在编译时调整生成的 Triton 内核，而不是第一次运行时
    autotune_at_compile_time = False

    # 是否阻止融合以实现更好的平铺
    tiling_prevents_pointwise_fusion = True
    tiling_prevents_reduction_fusion = True

    # 是否为内核给出不同的名称
    # 注意：这与 descriptive_names 是正交的 - 这决定了是否我们的 triton 内核名称应全部是 `triton_`
    # （以最大化缓存）或它们是否应该是唯一的。
    unique_kernel_names = os.environ.get("TORCHINDUCTOR_UNIQUE_KERNEL_NAMES") == "1"

    # 是否将操作名称放入内核名称中
    # False: 没有特殊名称（只是 triton__1、triton__2 等）
    # "torch": 映射到 Dynamo 图中的 fx 操作（模块名称、方法名称等）
    # "original_aten": 映射到最高级别的 aten 操作（即预分解）
    # "inductor_node": 映射到传递给 Inductor 的 FX 图中的节点名称
    descriptive_names = "original_aten"

    # 用于较小的归约是否使用替代代码生成
    persistent_reductions = (
        os.environ.get("TORCHINDUCTOR_PERSISTENT_REDUCTIONS", "1") == "1"
    )

    # 0/False: 禁用
    # 1/True: 启用，使用调优来选择不同的子内核
    # 2: 获取环境变量 TORCHINDUCTOR_MULTI_KERNEL 的值，转换为整数，如果未设置则默认为 0
    multi_kernel = int(os.environ.get("TORCHINDUCTOR_MULTI_KERNEL", "0"))

    # 3: 提示 Triton 在参数可被 16 整除时的优化机会
    divisible_by_16 = True

    # Minimum RBLOCK 用于 TritonSplitScanKernel 的最小值，间接控制所需的工作空间缓冲区大小
    min_split_scan_rblock = 256

    # 是否存储生成的 cubin 文件，供 cpp 包装代码加载使用
    store_cubin = False

    # 允许的最大溢出寄存器数量，用于基准测试的配置
    # 设置为 0 表示如果一个配置甚至溢出了一个寄存器就跳过
    # 设置一个较大的值允许一个配置溢出少量寄存器进行基准测试
    #
    # 注意：使用 sin/cos 的内核总是报告 >0 寄存器溢出。
    # 我们目前看到使用 sin/cos 的内核固定溢出 8 个寄存器。
    # 将阈值提高到 16 以确保安全。
    # 一旦更多理解寄存器溢出的来源，应重新审视此设置。
    spill_threshold: int = 16

    # 生成包含较新的 tl.make_block_ptr() API 的代码，用于加载/存储操作
    use_block_ptr = False

    # 在 relu 实现中注入错误，用于测试复现抽取和最小化功能
    # 有效值包括："compile_error", "runtime_error", "accuracy"
    inject_relu_bug_TESTING_ONLY: Optional[str] = None
class aot_inductor:
    # AOTInductor output path
    # If an absolute path is specified, the generated lib files will be stored under the directory;
    # If a relative path is specified, it will be used as a subdirectory under the default caching path;
    # If not specified, a temp directory will be created under the default caching path.
    # If the specified path contains something like "model.so", the sub-string will be used
    # to name the generated library.
    output_path = ""

    # Determine if debug compilation is enabled based on the environment variable AOT_INDUCTOR_DEBUG_COMPILE
    debug_compile = os.environ.get("AOT_INDUCTOR_DEBUG_COMPILE", "0") == "1"

    # Determine if debugging of constant binary dumps is enabled based on the environment variable AOT_INDUCTOR_DEBUG_DUMP_CONSTS_BIN
    debug_dump_consts_bin: bool = (
        os.environ.get("AOT_INDUCTOR_DEBUG_DUMP_CONSTS_BIN", "0") == "1"
    )

    # Serialized tree spec for flattening inputs
    serialized_in_spec = ""

    # Serialized tree spec for flattening outputs
    serialized_out_spec = ""

    # Flag to decide whether to create a submodule for constant graph.
    use_runtime_constant_folding: bool = False

    # Flag to force weights to be appended to the shared library and mmaped by the runtime
    # rather than embedded into the data section. Needed to support 1B+ parameter models
    force_mmap_weights: bool = False


class cuda:
    # CUDA arch to use for CUDA template kernel compilation.
    # e.g. "70", "75", "80", "90", etc.
    # When arch is None, Inductor uses torch.cuda.get_device_capability(0).
    arch: Optional[str] = None

    # CUDA version to use for CUDA template kernel compilation.
    # e.g. "11.4", "12.1", etc.
    # When version is None, Inductor uses torch.version.cuda.
    version: Optional[str] = None

    # Optimization level for the host compiler.
    compile_opt_level = "-O1"

    # Whether to enable device LTO (link-time-optimization).
    enable_cuda_lto = False

    # Whether to keep intermediate files during compilation.
    enable_ptxas_info = False

    # Whether to enable debug info, e.g. line number, cutlass debug info.
    enable_debug_info = False

    # Whether to use fast math.
    use_fast_math = False

    # Path to the CUTLASS repo root directory.
    # The default path only works under PyTorch local development environment.
    cutlass_dir = os.environ.get(
        "TORCHINDUCTOR_CUTLASS_DIR",
        os.path.abspath(
            os.path.join(os.path.dirname(torch.__file__), "../third_party/cutlass/")
        ),
    )

    # Configures the maximum number of CUTLASS configs to profile in max_autotune.
    # By default it's None, so that all CUTLASS configs are tuned.
    # This is mainly used to reduce test time in CI.
    cutlass_max_profiling_configs: Optional[int] = None

    # Path to CUDA NVCC.
    # NVCC search order:
    # 1) cuda_cxx set in this config
    # 2) CUDACXX environment variable
    # 3) CUDA_HOME environment variable
    # 4) default system search PATH.
    cuda_cxx: Optional[str] = None

    # Minimum value of M*N*K to consider the CUTLASS backend for GEMM ops.
    cutlass_backend_min_gemm_size: int = 1
    # 是否生成CUDA CPP生成的代码中的独立运行器
    # 如果启用，可以将生成的代码编译成独立可执行文件。
    generate_test_runner: bool = (
        os.environ.get("INDUCTOR_CUDA_BACKEND_GENERATE_TEST_RUNNER_CODE", "1") == "1"
    )
    
    # 保留包含指定正则表达式模式的Cutlass操作配置
    # 将其设置为"warpspecialized_cooperative_epi_tma"以仅启用用于大型GEMM的SM90 TMA Cutlass内核。
    cutlass_op_allowlist_regex: Optional[str] = None
    
    # 注意：可以通过调用Cutlass操作实例的op.configuration_name()获取Cutlass操作的名称。
    # 例如，通过cutlass_utils.gen_ops()返回的实例或传递给CUTLASSGemmTemplate.render(...)的op参数。
    
    # 过滤包含指定正则表达式模式的Cutlass配置
    # 将其设置为"pingpong"以避免由某些Cutlass内核使用的"pingpong"内存访问模式的操作顺序引起的数值问题。
    cutlass_op_denylist_regex: Optional[str] = "pingpong"
class rocm:
    # Offload arch list for device code compilation, e.g. ["gfx941", "gfx942"].
    # If empty, the `native` arch is used
    arch: List[str] = []

    # Enable for CDNA3 only for now
    # Processor name reference: https://llvm.org/docs/AMDGPUUsage.html#processors
    supported_arch: Set[str] = {"gfx940", "gfx941", "gfx942"}

    # Optimization level, use to balance compilation speed and runtime performance
    compile_opt_level = "-O2"

    # Flag to keep debug information in compiled objects
    is_debug = False

    # Flag to keep intermediate files (assembly listings, preprocessed sources, etc.)
    save_temps = False

    # Flag to add `-ffast-math`` to compile flags
    use_fast_math = True

    # Flag to add `-fgpu-flush-denormals-to-zero` to compile flags
    flush_denormals = True

    # Flag to print register and LDS usage during compilation
    print_kernel_resource_usage = False

    # Path to ROCm installation, if None, use env variable ROCM_HOME
    rocm_home: Optional[str] = None

    # Path to Composable Kernel library.
    # Install with `pip install git+https://github.com/rocm/composable_kernel@develop`.
    ck_dir = os.environ.get("TORCHINDUCTOR_CK_DIR")

    # Number of op instance choices to trade off between runtime perf and compilation time
    n_max_profiling_configs: Optional[int] = None

    # Flag to use a short list of CK instances which perform well across a variety of shapes.
    # Currently RCR and F16 only
    use_preselected_instances: bool = False


# Backend to use for CPU codegen either "cpp" or "halide" (experimental)
cpu_backend = "cpp"

# Backend to use for CUDA codegen either "triton" or "halide" (experimental)
cuda_backend = "triton"


class halide:
    # Base halide target to use for CPU devices
    cpu_target = "host"

    # Base halide target to use for CUDA devices
    gpu_target = "host-cuda"

    # Halide autoscheduler to use, choices are:
    # "Anderson2021" (gpu-only), "Li2018", "Adams2019" (cpu-only), or "Mullapudi2016" (cpu-only)
    scheduler_cuda = "Anderson2021"
    scheduler_cpu = "Adams2019"

    # Controls `no_asserts` flag passed to Halide target (warning: can false positive)
    asserts = False

    # Controls `debug` flag passed to Halide target
    debug = False


# create a directory containing lots of debug information
class trace:
    # master switch for all debugging flags below
    enabled = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    # Save debug information to a temporary directory
    # If not specified, a temp directory will be created by system
    debug_dir: Optional[str] = None

    # Save python logger call >=logging.DEBUG
    debug_log = False

    # Save python logger call >=logging.INFO
    info_log = False

    # Save input FX graph (post decomps, pre optimization)
    fx_graph = True

    # Save FX graph after transformations
    fx_graph_transformed = True

    # Save TorchInductor IR before fusion pass
    ir_pre_fusion = True
    # 是否在融合处理后保存 TorchInductor IR
    ir_post_fusion = True
    
    # 将生成的代码复制到跟踪目录
    output_code = True
    
    # 是否生成显示融合后图形的 SVG 图
    graph_diagram = os.environ.get("INDUCTOR_POST_FUSION_SVG", "0") == "1"
    
    # 是否生成显示带融合的 fx 图的 SVG 图
    draw_orig_fx_graph = os.environ.get("INDUCTOR_ORIG_FX_SVG", "0") == "1"
    
    # 默认情况下，使用 "record" 形状属性绘制 fx 图形。
    # 在图形非常复杂时，可能会遇到 dot 错误，如下所示：
    #   "flat edge between adjacent nodes one of which has a record shape -
    #    replace records with HTML-like labels"
    # 因此，让用户可以指定 dot 图形的形状属性。
    # 例如，传递 INDUCTOR_DOT_GRAPH_SHAPE_SVG = "none" 将允许我们生成类似 HTML 的标签，
    # 以解决上述错误。
    dot_graph_shape = os.environ.get("INDUCTOR_DOT_GRAPH_SHAPE_SVG", None)
    
    # 如果不是 None，则是保存每个修改图形的传递的 SVG 文件的 URL。
    # 每个传递中正在转换的节点将以黄色着色。
    # 目前 URL 仅支持本地目录。
    log_url_for_graph_xform = os.environ.get("INDUCTOR_LOG_URL_FOR_GRAPH_XFORM", None)
    
    # 存储 cProfile (使用 snakeviz 查看)
    compile_profile = False
    
    # 上传 .tar.gz 文件
    # 根据特定环境需求进行覆盖
    upload_tar: Optional[Callable[[str], None]] = None
    
    # 记录自动调整结果的日志
    log_autotuning_results: bool = False
_save_config_ignore = [
    # 忽略保存配置时出现的无法序列化函数问题
    "trace.upload_tar",
]

_cache_config_ignore_prefix = [
    # 跟踪函数与配置缓存无关
    "trace",
    # 使用绝对路径
    "cuda.cutlass_dir",
    # 不相关
    "compile_threads",
]

if TYPE_CHECKING:
    # 导入类型检查所需的模块
    from torch.utils._config_typing import *  # noqa: F401, F403

from torch.utils._config_module import install_config_module

# 安装配置模块，包括补丁、保存配置等
install_config_module(sys.modules[__name__])
```