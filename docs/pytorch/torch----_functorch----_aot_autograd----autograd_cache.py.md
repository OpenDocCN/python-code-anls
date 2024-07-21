# `.\pytorch\torch\_functorch\_aot_autograd\autograd_cache.py`

```
"""
Utils for caching the outputs of AOTAutograd
"""
# 引入必要的模块和库
import functools
import logging
import os
import pickle
import shutil

# 导入数据类装饰器
from dataclasses import dataclass

# 导入类型检查标志
from typing import Callable, List, Optional, TYPE_CHECKING, Union

# 导入 torch 库
import torch

# 导入计数器模块
from torch._dynamo.utils import counters

# 导入 Functorch 配置
from torch._functorch import config

# 导入代码缓存相关模块
from torch._inductor.codecache import (
    _ident,
    BypassFxGraphCache,
    CompiledFxGraph,
    FxGraphCache,
    FxGraphCachePickler,
    FxGraphHashDetails,
    get_code_hash,
    write_atomic,
)

# 导入运行时缓存目录函数
from torch._inductor.runtime.runtime_utils import cache_dir

# 导入假张量元数据提取模块
from torch._subclasses.fake_tensor import extract_tensor_metadata

# 导入运行时包装器模块
from .runtime_wrappers import (
    AOTDispatchAutograd,
    AOTDispatchSubclassWrapper,
    CompilerWrapper,
    FunctionalizedRngRuntimeWrapper,
    post_compile,
    RuntimeWrapper,
    SubclassMeta,
)

# 导入模式相关模块
from .schemas import AOTConfig, ViewAndMutationMeta  # noqa: F401

# 如果是类型检查阶段，导入额外的 Node 类
if TYPE_CHECKING:
    from torch.fx.node import Node

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


# 自定义异常，用于绕过 AOTAutograd 缓存
class BypassAOTAutogradCache(Exception):
    pass


# 当 FXGraphCache 未命中时引发的异常
class FXGraphCacheMiss(BypassAOTAutogradCache):
    pass


# 检查节点是否安全的函数
def check_node_safe(node: Node):
    """
    Checks that the node only uses supported operators. We are starting with very
    conservative cacheability constraints, and incrementally adding more support as we expand.
    """

    def is_torch_function(target):
        # 检查目标是否为内置的 torch 函数或方法
        is_builtin_fun_or_type = type(target).__name__ == "builtin_function_or_method"
        # TODO: 处理 torch.nn.functional 等非内联目标，它们不会编译成内置函数
        return is_builtin_fun_or_type

    def is_tensor(target):
        # 张量总是在 meta 字段中具有示例值
        return "example_value" in target.meta

    # 检查节点操作类型
    if node.op == "call_function":
        # 目前只支持 torch.* 函数
        # 可能还可以添加一些安全的非 torch 实现到白名单中
        if not is_torch_function(node.target):
            raise BypassAOTAutogradCache(
                f"Unsupported call_function target {node.target}"
            )
    elif node.op == "call_method":
        method_name = node.target
        method_target = node.args[0]
        # 只支持基本张量的方法调用
        if not is_tensor(method_target):
            raise BypassAOTAutogradCache(
                f"Unsupported call_method target {method_target}"
            )
        if (
            type(method_name) != str
            and type(method_name).__name__ != "method_descriptor"
        ):
            raise BypassAOTAutogradCache(
                f"Unsupported call_method method {node.target}: {method_name}"
            )
    # 缓存安全
    # 如果节点操作是 "placeholder", "get_attr", "call_module", "output" 中的一种，则执行以下逻辑：
    # 假设 call_module 是一个安全操作的前提：
    # (1) 当前仅允许从 "built-in-nn-modules" 中获取的 call_module 操作出现在图中，这些模块被 Dynamo 假设为可以安全追踪。
    #     如果 Dynamo 认为可以盲目追踪它们，那么它们也应该可以安全缓存。
    # (2) 在稳定状态下（可能在下半年？），我们不再应该看到这些操作，因为内联的内置 nn 模块将成为默认设置。
    # (3) 今天我们不允许用户自定义的 nn 模块出现在图中，只允许函数调用。
    pass  # 如果不满足以上条件，则抛出异常，指出不支持的节点操作类型
# 使用 functools 库中的 lru_cache 装饰器，对 get_autograd_code_hash 函数进行结果缓存
@functools.lru_cache(None)
def get_autograd_code_hash():
    # 获取当前文件的所在目录路径
    autograd_root = os.path.dirname(__file__)
    # 调用 get_code_hash 函数，计算 autograd_root 目录下文件的哈希值并返回
    return get_code_hash([autograd_root])


def check_cacheable(gm: torch.fx.GraphModule):
    """
    检查图模块是否只使用支持的运算符
    """
    # 获取图模块的所有节点
    nodes = gm.graph.nodes
    # 检查是否处于编译自动微分区域，如果是则抛出异常
    if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
        raise BypassAOTAutogradCache(
            "Cannot cache a graph with compiled autograd enabled"
        )

    # 检查是否启用了 FX 图缓存，如果没有则抛出异常
    if not torch._inductor.config.fx_graph_cache:
        raise BypassAOTAutogradCache("FX graph cache is not enabled")

    # 获取跟踪上下文，并检查是否启用了 fakify_first_call，如果是则抛出异常
    tracing_context = torch._guards.TracingContext.try_get()
    if tracing_context and tracing_context.fakify_first_call:
        raise BypassAOTAutogradCache(
            "Won't cache a graph with fakify_first_call enabled"
        )
    
    # 对图中的每个节点进行安全检查
    for node in nodes:
        check_node_safe(node)


class AOTAutogradCacheDetails(FxGraphHashDetails):
    """
    用于捕获与计算 AOTAutograd 安全且稳定缓存键相关的所有图模块详细信息的对象。
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs,
        aot_config: AOTConfig,
    ):
        # FxGraphHashDetails 包含与感应器相关的所有键。还包括一些系统信息
        self.aot_config = aot_config
        # 检查梯度是否启用
        self.grad_enabled = torch.is_grad_enabled()
        # 检查是否启用了自动混合精度
        self.disable_amp = torch._C._is_any_autocast_enabled()
        # 检查是否启用了确定性算法
        self.deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
        # 计算自动微分代码的哈希值
        self.code_hash = get_autograd_code_hash()
        # 保存自动微分配置
        self.autograd_config = config.save_config()
        try:
            # 我们不使用 FxGraphHashDetails 来对 example_inputs 进行哈希化，因为它期望
            # example_inputs 始终是 FakeTensors，但在 AOTAutograd 的入口点，
            # 它们仍然是常规张量。因此，我们在这里存储它们的元数据。
            # TODO: 这会导致比必要的更多缓存未命中，因为这是在我们添加
            # symints 到张量元数据之前。稍后改进这一点。
            # 提取 example_inputs 中张量的元数据
            self.example_input_metadata = [
                extract_tensor_metadata(t)
                for t in example_inputs
                if isinstance(t, torch.Tensor)
            ]
            # 调用父类构造函数初始化 FxGraphHashDetails
            super().__init__(gm, [], {}, [])
        except BypassFxGraphCache as e:
            # 有时感应器配置无法 pickle 并可能失败
            raise BypassAOTAutogradCache from e

    def debug_str(self) -> str:
        # 返回 AOTAutogradCachePickler 的调试信息字符串
        return AOTAutogradCachePickler.debug_str(self)


def _reduce_aot_config(aot_config: AOTConfig):
    """
    将配置减少为用于缓存的稳定键。
    """
    # 返回一个包含标识符和配置信息元组的元组
    return (
        # 返回模型的标识符
        _ident,
        (
            # 返回AOT编译配置中的参数缓冲区数量
            aot_config.num_params_buffers,
            # 返回AOT编译配置中是否保留推断输入变化
            aot_config.keep_inference_input_mutations,
            # 返回AOT编译配置中是否导出模型
            aot_config.is_export,
            # 返回AOT编译配置中是否禁用切线计算
            aot_config.no_tangents,
            # 返回AOT编译配置中是否启用动态形状
            aot_config.dynamic_shapes,
            # 返回AOT编译配置中自动微分参数位置到源的映射
            aot_config.aot_autograd_arg_pos_to_source,
            # 返回AOT编译配置中是否启用日志
            aot_config.enable_log,
            # 返回AOT编译配置中预先分派的设置
            aot_config.pre_dispatch,
        ),
    )
class AOTAutogradCachePickler(FxGraphCachePickler):
    # 继承自 FxGraphCachePickler 的 AOTAutogradCachePickler 类
    dispatch_table = FxGraphCachePickler.dispatch_table.copy()
    # 复制 FxGraphCachePickler 的调度表 dispatch_table 到当前类的 dispatch_table

    # 将 AOTConfig 类型映射到 _reduce_aot_config 函数
    dispatch_table[AOTConfig] = _reduce_aot_config


def autograd_cache_key(
    gm: torch.fx.GraphModule,
    example_inputs,
    config: AOTConfig,
    # TODO: add args and parameters
) -> str:
    """
    生成用于缓存的 FX 图的唯一哈希值。
    """
    # 检查 gm 是否可缓存
    check_cacheable(gm)
    # 创建 AOTAutogradCacheDetails 对象，包含 gm、example_inputs 和 config 的详细信息
    details = AOTAutogradCacheDetails(gm, example_inputs, config)
    # 使用 AOTAutogradCachePickler 获取 details 的哈希值，添加前缀 'a'
    key = "a" + AOTAutogradCachePickler.get_hash(details)
    # 记录调试信息，显示生成的哈希 key 和 details 的调试字符串
    log.debug(
        "Autograd graph cache hash details for key %s:\n%s", key, details.debug_str()
    )
    # 返回生成的哈希 key
    return key


@dataclass
class FXGraphCacheLoadable:
    """
    可加载的 FX 图缓存条目
    """
    fx_graph_cache_key: str

    def load(self, example_inputs) -> CompiledFxGraph:
        # [Note: AOTAutogradCache and FXGraphCache Guard interactions]
        # AOTAutogradCache 和 FXGraphCache 保护交互说明
        # AOTAutograd 从 dynamo 参数列表中获取 symint 输入。
        # FXGraphCache 序列化所需的守卫，并基于这些 symint 输入到图中的 shape_env。
        # AOTAutograd 在此使用的不变性是，其通过 dynamo 提供的 symint 的源与其传递给 Inductor 的源完全相同，
        # 无论是前向传递还是反向传递。（这并不意味着传递的张量值相同，只是它们的 symint 相同）。
        # 换句话说，AOTAutograd 和 Inductor 不会基于与其由 Inductor 提供的不同源的 symint 创建新的守卫。
        result = FxGraphCache._lookup_graph(
            self.fx_graph_cache_key, example_inputs, local=True, remote_cache=False
        )
        # 如果没有找到结果，记录缓存未命中的信息并抛出 FXGraphCacheMiss 异常
        if result is None:
            log.info("FXGraphCache cache miss for key %s", self.fx_graph_cache_key)
            raise FXGraphCacheMiss
        # 将结果标记为已封装调用
        result._boxed_call = True
        # 返回加载的结果
        return result


@dataclass
class CompiledForward(FXGraphCacheLoadable):
    """
    前向函数的可缓存条目
    """
    pass


@dataclass
class CompiledBackward(FXGraphCacheLoadable):
    """
    反向函数的可缓存条目

    Used by AOTDispatchAutograd.post_compile
    """
    backward_state_indices: List[int]
    num_symints_saved_for_bw_: int


@dataclass
class AOTAutogradCacheEntry:
    """
    缓存中的单个条目

    Forward 和 Backward 信息
    compiled_fw: CompiledForward
    compiled_bw: Optional[CompiledBackward]

    在编译之前保存的运行时元数据
    runtime_metadata: ViewAndMutationMeta

    每个 aot_dispatch_* 函数运行后运行的包装器
    dispatch_wrappers: List[CompilerWrapper]

    由 AOTSubclassWrapper 使用
    maybe_subclass_meta: Optional[SubclassMeta]
    num_fw_outs_saved_for_bw: Optional[int]

    由 RuntimeWrapepr 使用
    indices_of_inps_to_detach: List[int]

    将缓存条目转换为原始可调用对象
    """
    pass
    # 定义一个方法 wrap_post_compile，接受三个参数：self，一个包含 torch.Tensor 的列表 args，和一个 AOTConfig 类型的参数 aot_config
    def wrap_post_compile(
        self, args: List[torch.Tensor], aot_config: AOTConfig
    @staticmethod
    # 静态方法：清空缓存
    def clear():
        """Clear the cache"""
        # 尝试删除临时目录
        try:
            shutil.rmtree(AOTAutogradCache._get_tmp_dir())
        except FileNotFoundError:
            # 如果目录不存在，则忽略异常
            pass

    @staticmethod
    # 静态方法：加载缓存
    def load(
        dispatch_and_compile: Callable,
        mod: Union[torch.fx.GraphModule, torch._dynamo.utils.GmWrapper],
        args,
        aot_config: AOTConfig,
    ) -> Callable:
        """
        Load a result from the cache, and reconstruct a runtime wrapper around the object
        """
        # Determine the correct module to use, considering whether mod is a GmWrapper instance
        gm = mod.gm if isinstance(mod, torch._dynamo.utils.GmWrapper) else mod
        compiled_fn = None
        cache_key = None
        try:
            # Generate a cache key using autograd_cache_key function
            cache_key = autograd_cache_key(gm, args, aot_config)
            # Look up the cache entry for the generated cache key
            entry: Optional[AOTAutogradCacheEntry] = AOTAutogradCache._lookup(cache_key)
            if entry is not None:
                # Wrap the compiled function using the cache entry's post-compile method
                compiled_fn = entry.wrap_post_compile(args, aot_config)
                # Log cache hit information
                log.info("AOTAutograd cache hit for key %s", cache_key)
                # Increment cache hit counter
                counters["aot_autograd"]["autograd_cache_hit"] += 1
            if compiled_fn is None:
                # Log cache miss information
                log.info("AOTAutograd cache miss for key %s", cache_key)
                # Increment cache miss counter
                counters["aot_autograd"]["autograd_cache_miss"] += 1
        # Handle specific exceptions related to cache misses or bypasses
        except FXGraphCacheMiss as e:
            counters["aot_autograd"]["autograd_cache_miss"] += 1
            if config.strict_autograd_cache:
                raise e
        except BypassAOTAutogradCache as e:
            cache_key = None
            counters["aot_autograd"]["autograd_cache_bypass"] += 1
            if config.strict_autograd_cache:
                raise e
        # If compiled_fn is still None, set the cache key in aot_config and compile
        if compiled_fn is None:
            aot_config.cache_key = cache_key
            compiled_fn = dispatch_and_compile()
        # Return the compiled function
        return compiled_fn

    @staticmethod
    def _get_tmp_dir() -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        # Return the path to the top-level temporary directory used for AOT autograd caching
        return os.path.join(cache_dir(), "aotautograd")

    @staticmethod
    def _lookup(key: str) -> Optional[AOTAutogradCacheEntry]:
        """Given a key generated by AOTAutogradCachePickler, look up its location in the cache."""
        # Construct the subdirectory path where the cache entry is stored
        subdir = os.path.join(AOTAutogradCache._get_tmp_dir(), key)
        # Check if the subdirectory exists; if not, return None indicating cache miss
        if not os.path.exists(subdir):
            return None
        # Construct the full path to the cache entry file
        path = os.path.join(subdir, "entry")
        try:
            # Attempt to load the cache entry from the file
            with open(path, "rb") as f:
                entry: AOTAutogradCacheEntry = pickle.load(f)
            return entry  # Return the loaded cache entry
        except Exception as e:
            # Log a warning if loading the cache entry fails
            log.warning("AOTAutograd cache unable to load compiled graph: %s", e)
            # Raise the exception if strict_autograd_cache mode is enabled
            if config.strict_autograd_cache:
                raise e
            return None  # Return None indicating cache miss
    # 定义保存函数，用于将单个条目保存到缓存中
    def save(key: str, entry: AOTAutogradCacheEntry):
        """Save a single entry into the cache."""
        
        # 尝试将条目序列化为字节流
        try:
            content = pickle.dumps(entry)
        except Exception as e:
            # 如果序列化失败，记录警告信息，并根据配置处理严格的自动微分缓存要求
            log.warning("AOTAutograd cache unable to serialize compiled graph: %s", e)
            if config.strict_autograd_cache:
                raise e
            return None
        
        # 获取保存目录的子目录路径，确保子目录存在，不存在则创建
        subdir = os.path.join(AOTAutogradCache._get_tmp_dir(), key)
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)
        
        # 构建条目的完整路径
        path = os.path.join(subdir, "entry")
        
        # 记录将AOTAutograd缓存条目写入的操作信息
        log.info("Writing AOTAutograd cache entry to %s", path)
        
        # 使用原子写入方法将内容写入路径指定的文件
        write_atomic(path, content)
        
        # 增加自动微分缓存保存计数器
        counters["aot_autograd"]["autograd_cache_saved"] += 1
```