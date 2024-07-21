# `.\pytorch\torch\_dynamo\config.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import getpass         # 导入用于获取当前用户的模块
import inspect         # 导入用于获取对象信息的模块
import os              # 导入操作系统相关功能的模块
import re              # 导入正则表达式模块
import sys             # 导入系统相关的参数和功能
import tempfile        # 导入临时文件和目录创建模块
from os.path import abspath, dirname  # 导入路径操作相关的函数
from typing import Any, Callable, Dict, Optional, Set, Type, TYPE_CHECKING, Union  # 导入类型提示相关的功能

import torch           # 导入 PyTorch 深度学习库


def is_fbcode():
    # 检查是否处于 Facebook 的代码环境中
    return not hasattr(torch.version, "git_version")


# to configure logging for dynamo, aot, and inductor
# use the following API in the torch._logging module
# torch._logging.set_logs(dynamo=<level>, aot=<level>, inductor<level>)
# or use the environment variable TORCH_LOGS="dynamo,aot,inductor" (use a prefix + to indicate higher verbosity)
# see this design doc for more detailed info
# Design doc: https://docs.google.com/document/d/1ZRfTWKa8eaPq1AxaiHrq4ASTPouzzlPiuquSBEJYwS8/edit#
# the name of a file to write the logs to
# [@compile_ignored: debug]
log_file_name: Optional[str] = None

# [@compile_ignored: debug] Verbose will print full stack traces on warnings and errors
verbose = os.environ.get("TORCHDYNAMO_VERBOSE", "0") == "1"

# [@compile_ignored: runtime_behaviour] verify the correctness of optimized backend
verify_correctness = False

# need this many ops to create an FX graph
minimum_call_count = 1

# turn on/off DCE pass
dead_code_elimination = True

# disable (for a function) when cache reaches this size

# controls the maximum number of cache entries with a guard on same ID_MATCH'd
# object. It also controls the maximum size of cache entries if they don't have
# any ID_MATCH'd guards.
# [@compile_ignored: runtime_behaviour]
cache_size_limit = 8

# [@compile_ignored: runtime_behaviour] safeguarding to prevent horrible recomps
accumulated_cache_size_limit = 256

# whether or not to specialize on int inputs.  This only has an effect with
# dynamic_shapes; when dynamic_shapes is False, we ALWAYS specialize on int
# inputs.  Note that assume_static_by_default will also cause ints to get
# specialized, so this is mostly useful for export, where we want inputs
# to be dynamic, but accesses to ints should NOT get promoted into inputs.
specialize_int = False

# Whether or not to specialize on float inputs.  Dynamo will always promote
# float inputs into Tensor inputs, but at the moment, backends inconsistently
# support codegen on float (this is to be fixed).
specialize_float = True

# legacy config, does nothing now!
dynamic_shapes = True

use_lazy_graph_module = (
    os.environ.get("TORCH_COMPILE_USE_LAZY_GRAPH_MODULE", "1") == "1"
)

# This is a temporarily flag, which changes the behavior of dynamic_shapes=True.
# When assume_static_by_default is True, we only allocate symbols for shapes marked dynamic via mark_dynamic.
# NOTE - this flag can be removed once we can run dynamic_shapes=False w/ the mark_dynamic API
# see [Note - on the state of mark_dynamic]
assume_static_by_default = True

# This flag changes how dynamic_shapes=True works, and is meant to be used in conjunction
# with assume_static_by_default=True.
# 启用此标志后，第一次编译框架时始
# 维护一个集合，用于存储可追踪的张量子类的类型
traceable_tensor_subclasses: Set[Type[Any]] = set()

# 如果环境变量 TORCHDYNAMO_SUPPRESS_ERRORS 设为真，则在 torch._dynamo.optimize 中抑制错误，强制回退到急切模式
# 这是使模型能够以某种方式工作的一种方法，但可能会失去优化机会
suppress_errors = bool(os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", False))

# 如果环境变量 TORCH_COMPILE_REPLAY_RECORD 设为 '1'，则在遇到异常时，记录并将当前帧的执行记录写入文件
replay_record_enabled = os.environ.get("TORCH_COMPILE_REPLAY_RECORD", "0") == "1"

# 将 Python 中的 assert 语句重写为 torch._assert
rewrite_assert_with_torch_assert = True

# 如果环境变量 TORCH_COMPILE_DISABLE 为真，则禁用 dynamo
disable = os.environ.get("TORCH_COMPILE_DISABLE", False)

# 如果环境变量 TORCH_COMPILE_CPROFILE 为真，则获取 Dynamo 的 cprofile 跟踪
cprofile = os.environ.get("TORCH_COMPILE_CPROFILE", False)

# 跳过内联模块的白名单配置，但此处的 legacy config 已不再起作用
skipfiles_inline_module_allowlist: Dict[Any, Any] = {}

# 如果 PyTorch 模块的字符串在此 ignorelist 中，则在创建 FX IR 时，allowed_functions.is_allowed 函数将不会考虑它们
allowed_functions_module_string_ignorelist = {
    "torch.distributions",
    "torch.testing",
    "torch._refs",
    "torch._prims",
    "torch._decomp",
}

# 调试标志，尝试在不同阶段使用 minifier。可能的值为 {None, "aot", "dynamo"}
# None - 关闭 minifier
# dynamo - 在 TorchDynamo 生成的图上运行 minifier，如果编译失败
# aot - 在 Aot Autograd 生成的图上运行 minifier，如果编译失败
repro_after = os.environ.get("TORCHDYNAMO_REPRO_AFTER", None)

# 编译器编译调试信息级别
# 1: 如果编译失败，将原始图导出到 repro.py
# 2: 如果编译失败，生成一个 minifier_launcher.py
# 3: 总是生成一个 minifier_launcher.py。适用于段错误
# 4: 如果精度失败，生成一个 minifier_launcher.py
repro_level = int(os.environ.get("TORCHDYNAMO_REPRO_LEVEL", 2))
# 从环境变量中获取 TORCHDYNAMO_REPRO_FORWARD_ONLY 的值，判断是否设置为 "1"，然后赋值给 repro_forward_only 变量
repro_forward_only = os.environ.get("TORCHDYNAMO_REPRO_FORWARD_ONLY") == "1"

# 设置测试编译图形是否发散的容差值，当差异超过此值时将其视为精度失败
# [@compile_ignored: debug]
repro_tolerance = 1e-3

# 从环境变量中获取 TORCHDYNAMO_REPRO_IGNORE_NON_FP 的值，判断是否设置为 "1"，然后赋值给 repro_ignore_non_fp 变量
# 检查精度时是否忽略非浮点值，如布尔张量，以避免误报
# [@compile_ignored: debug]
repro_ignore_non_fp = os.environ.get("TORCHDYNAMO_REPRO_IGNORE_NON_FP") == "1"

# 设置是否在比较两个模型时使用 fp64 参考，并只在相对于 fp64 的 RMSE 较大时报告问题
# 这会增加内存使用，如果内存使用过高，可以禁用此选项
# [@compile_ignored: runtime_behaviour]
same_two_models_use_fp64 = True

# 设置是否捕获标量输出，某些后端不支持标量，如 .item() 返回标量类型时
# 当此标志为 False 时，引入图形断裂而不是捕获
# 这要求 dynamic_shapes 为 True
capture_scalar_outputs = os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS") == "1"

# 设置是否捕获具有动态输出形状的操作符，某些后端不支持具有动态输出形状的操作
# 当此标志为 False 时，引入图形断裂而不是捕获
# 这要求 dynamic_shapes 为 True
capture_dynamic_output_shape_ops = (
    os.environ.get("TORCHDYNAMO_CAPTURE_DYNAMIC_OUTPUT_SHAPE_OPS", "0") == "1"
)

# 设置是否优先选择延迟运行时断言而不是保护性检查
prefer_deferred_runtime_asserts_over_guards = False

# 设置是否允许将复杂的 guards 视为运行时断言
# 默认情况下，会引发约束违规错误或默认专门化
_allow_complex_guards_as_runtime_asserts = False

# 默认情况下，dynamo 将所有整数视为支持的 SymInts，强制将 _length_per_key 和 _offset_per_key 视为类似于未支持的 SymInts
# 可用于导出，以便它们能够立即泛化，并且不会与 0/1 相等
force_unspec_int_unbacked_size_like_on_torchrec_kjt = False

# 设置是否强制要求 cond 的 true_fn 和 false_fn 生成具有相同 guards 的代码
enforce_cond_guards_match = True
# Specify how to optimize a compiled DDP module. The flag accepts a boolean
# value or a string. There are 4 modes.
# 1. "ddp_optimizer" (or True): with "ddp_ptimizer", Dynamo will automatically
# split model graph into pieces to match DDP bucket sizes to allow DDP
# comm/compute overlap.
# 2. "python_reducer" (experimental): this optimization requires the usage
# of compiled_autograd. With "python_reducer", DDP will disable the C++ reducer
# and use the Python reducer to allow compiled_autograd to trace the
# communication and allow comm/compute overlap without graph-breaks.
# 3. "python_reducer_without_compiled_forward" (experimental): this mode is
# similar to "python_reducer". One should only use this optimization mode
# when compiled_autograd is used but the DDP module is not compiled.
# 4. "no_optimization" (or False): Dynamo won't split the model graph, nor
# will Python reducer be used. With this mode, there will be no graph-breaks
# and the original DDP C++ reducer will be used. There will no comm/compute
# overlap. This mode CANNOT be used with compiled_autograd.
# Note that to avoid breaking the existing usage, mode 1 and mode 4 can be
# specified with a boolean value. True is using ddp_optimizer and False is
# no optimization.
optimize_ddp: Union[bool, str] = True

# By default, Dynamo emits runtime asserts (e.g. torch._check, torch._check_is_size) in the graph.
# In some cases those asserts could be performance costly
# E.g. torch._check(tensor[0].item() > 2) for tensor on cuda will require cuda sync.
# Setting this to True keeps them hinting to symbolic shapes engine,
# but not be emitted in the graph.
do_not_emit_runtime_asserts: bool = (
    os.environ.get("TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS", "0") == "1"
)

_ddp_optimization_mode = [
    "ddp_optimizer",
    "python_reducer",  # experimental mode
    "python_reducer_without_compiled_forward",  # experimental mode
    "no_optimization",
]

# Function to determine the current optimization mode for DDP
def _get_optimize_ddp_mode():
    m = sys.modules[__name__]
    if isinstance(m.optimize_ddp, bool):
        if m.optimize_ddp:
            mode = "ddp_optimizer"
        else:
            mode = "no_optimization"
    elif isinstance(m.optimize_ddp, str):
        mode = m.optimize_ddp
    else:
        raise ValueError(f"Invalid type, {type(optimize_ddp)=}")

    assert mode in m._ddp_optimization_mode, f"Invalid mode {mode=}"
    return mode


# No longer used
optimize_ddp_lazy_compile = False

# Whether to skip guarding on FSDP-managed modules
skip_fsdp_guards = True
# Whether to apply torch._dynamo.disable() to per-param FSDP hooks
skip_fsdp_hooks = False

# Make dynamo skip guarding on hooks on nn modules
# Note: unsafe: if your model actually has hooks and you remove them, or doesn't and  you add them,
# dynamo will not notice and will execute whichever version you first compiled.
skip_nnmodule_hook_guards = True

# If True, raises exception if TorchDynamo is called with a context manager
raise_on_ctx_manager_usage = True
# 如果为 True，则在不安全时引发 aot autograd 异常
raise_on_unsafe_aot_autograd = False

# 如果为 True，在 torch.jit.trace 一个经过 dynamo 优化的函数时引发错误；如果为 False，则静默处理 dynamo
error_on_nested_jit_trace = True

# 如果为 True，在 symbolically trace 一个经过 dynamo 优化的函数时提供更好的错误消息；如果为 False，则静默处理 dynamo
error_on_nested_fx_trace = True

# 如果为 False，在 rnn 上禁用图断裂。使用不同的后端可能会有不同效果。
allow_rnn = False

# 如果为 True，在尝试编译一个之前已经见过的函数时引发错误
error_on_recompile = False

# 是否报告任何守卫失败（已弃用，不再起作用）
report_guard_failures = True

# 项目的根目录
base_dir = dirname(dirname(dirname(abspath(__file__))))

# 是否跟踪 NumPy 或进行图断裂
trace_numpy = True

# 使用 torch.compile 进行跟踪时的默认 NumPy 数据类型
numpy_default_float = "float64"
numpy_default_complex = "complex128"
numpy_default_int = "int64"

# 如果为 True，使用 NumPy 的随机数生成器；如果为 False，则使用 PyTorch 的随机数生成器
use_numpy_random_stream = False

# 是否使用 C++ 守卫管理器
enable_cpp_guard_manager = os.environ.get("TORCHDYNAMO_CPP_GUARD_MANAGER", "1") == "1"

# 是否内联内置的 nn 模块
inline_inbuilt_nn_modules = (os.environ.get("TORCHDYNAMO_INLINE_INBUILT_NN_MODULES", "0") == "1")

# 默认的调试目录根路径
def default_debug_dir_root():
    # 如果设置了环境变量 TORCH_COMPILE_DEBUG_DIR，则返回其指定的路径
    DEBUG_DIR_VAR_NAME = "TORCH_COMPILE_DEBUG_DIR"
    if DEBUG_DIR_VAR_NAME in os.environ:
        return os.path.join(os.environ[DEBUG_DIR_VAR_NAME], "torch_compile_debug")
    # 如果在 fbcode 环境下，则返回临时目录下以用户名命名的路径
    elif is_fbcode():
        return os.path.join(tempfile.gettempdir(), getpass.getuser(), "torch_compile_debug")
    # 否则返回当前工作目录下的 torch_compile_debug 目录
    else:
        return os.path.join(os.getcwd(), "torch_compile_debug")

# 调试目录的根路径
debug_dir_root = default_debug_dir_root()

# 需要忽略的保存配置项集合
_save_config_ignore = {
    "repro_after",
    "repro_level",
    # 临时解决方法："cannot pickle PyCapsule"
    "constant_functions",
    # 临时解决方法："cannot pickle module"
    "skipfiles_inline_module_allowlist",
}

# 对于 backend="cudagraphs"，是否将输入上的变异发送到 cudagraph 后端或在 aot_autograd 后处理中重放，默认为 False，
# 因为输入上的变异可能会阻止 cudagraphing
cudagraph_backend_keep_input_mutation = False

# 是否支持来自先前 cudagraph 池中变异输入的 cudagraph 支持
cudagraph_backend_support_input_mutation = False

# 如果为 True，则仅允许具有 torch.Tag.pt2_compliant 标签的操作进入图中；所有其他操作将被禁止，并且回退到 eager-mode PyTorch。
# 对于确保自定义操作的正确性很有用。
only_allow_pt2_compliant_ops = False

# 是否捕获 autograd 函数
capture_autograd_function = True

# 是否启用 dynamo 跟踪 `torch.func` 转换
capture_func_transforms = True
# 是否将 Dynamo 编译指标记录到日志文件（用于 OSS）和 Scuba 表（用于 fbcode）
log_compilation_metrics = True

# 一组日志函数，将被重新排序到图中断点的末尾，允许 Dynamo 构建更大的图。注意，这里有一些限制，比如它不能正确打印在打印语句之后被修改的对象。
reorderable_logging_functions: Set[Callable[[Any], None]] = set()

# 模拟如果我们没有支持 BUILD_SET 操作码会发生什么，用于测试
inject_BUILD_SET_unimplemented_TESTING_ONLY = False

# 在严格模式下禁止的自动求导操作
_autograd_backward_strict_mode_banned_ops = [
    "stride",
    "requires_grad",
    "storage_offset",
    "layout",
    "data",
]

# 将所有 torch.Tensor 中以 "is_" 开头的方法添加到禁止列表中
_autograd_backward_strict_mode_banned_ops.extend(
    [name for name, _ in inspect.getmembers(torch.Tensor) if re.match(r"^is_.*", name)]
)

# 启用对假张量调度的缓存
fake_tensor_cache_enabled = (
    os.environ.get("TORCH_FAKE_TENSOR_DISPATCH_CACHE", "1") == "1"
)

# 启用假张量缓存与调度之间的交叉检查
fake_tensor_cache_crosscheck_enabled = (
    os.environ.get("TORCH_FAKE_TENSOR_DISPATCH_CACHE_CROSSCHECK", "0") == "1"
)

# 启用编译自动求导引擎跟踪在 torch.compile() 下调用的 .backward() 方法
# 注意：AOT 自动求导仍将跟踪联合图
compiled_autograd = False

if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403

    # 定义一个函数，用于创建闭包补丁
    def _make_closure_patcher(**changes):
        ...

# 安装配置模块
from torch.utils._config_module import install_config_module

install_config_module(sys.modules[__name__])
```