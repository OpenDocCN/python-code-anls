# `.\pytorch\torch\_dynamo\output_graph.py`

```py
# 导入必要的模块和库

# 导入用于处理默认类型的声明
mypy: allow-untyped-defs

# 导入集合模块，用于处理各种集合数据类型
import collections

# 提供对上下文管理的支持
import contextlib

# 提供对象复制操作的支持
import copy

# 提供对函数式编程的支持，如函数装饰器
import functools

# 提供对迭代工具的支持，如生成器函数
import itertools

# 提供日志记录功能
import logging

# 提供对操作符函数的支持
import operator

# 提供正则表达式操作的支持
import re

# 提供系统特定参数和功能的访问支持
import sys

# 提供异常追踪功能
import traceback

# 提供弱引用对象的支持
import weakref

# 提供数据类的支持
from dataclasses import dataclass

# 提供类型提示的支持
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union

# 提供符号计算的支持
import sympy

# 导入 Torch 相关的保护模块
import torch._guards

# 导入 Torch 相关的日志模块
import torch._logging

# 导入 Torch 的神经网络模块
import torch.nn

# 导入 Torch 的 Pytree 模块
import torch.utils._pytree as pytree

# 导入 Torch 的 fx 模块
from torch import fx

# 导入 Torch 的全局上下文检查点状态、源和跟踪上下文模块
from torch._guards import GlobalContextCheckpointState, Source, TracingContext

# 导入 Torch 内部工具的内部函数
from torch._utils_internal import signpost_event

# 导入 Torch 的 fx 的懒加载图模块
from torch.fx._lazy_graph_module import _make_graph_module  # type: ignore[attr-defined]

# 导入 Torch fx 实验性的反向状态模块
from torch.fx.experimental._backward_state import BackwardState

# 导入 Torch fx 实验性的符号形状模块
from torch.fx.experimental.symbolic_shapes import free_symbols, is_symbolic, ShapeEnv

# 导入 Torch fx 运行时断言插入模块
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts

# 导入 Torch 的 Python 调度工具模块
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

# 导入当前模块的配置、日志和变量模块
from . import config, logging as torchdynamo_logging, variables

# 导入后端注册表的编译和编译器函数
from .backends.registry import CompiledFn, CompilerFn

# 导入字节码转换模块
from .bytecode_transformation import (
    create_call_function,
    create_instruction,
    Instruction,
    unique_id,
)

# 导入代码上下文模块
from .code_context import code_context

# 导入代码生成模块
from .codegen import PyCodegen

# 导入当前作用域 ID 模块
from .current_scope_id import enter_new_scope

# 导入异常模块
from .exc import (
    BackendCompilerFailed,
    exceptions_allowed_to_be_fallback,
    SkipFrame,
    unimplemented,
    unimplemented_with_warning,
)

# 导入守卫生成器模块
from .guards import GuardBuilder, install_guard

# 导入变异守卫模块
from .mutation_guard import is_dynamic_nn_module

# 导入副作用模块
from .side_effects import AttributeMutationExisting, SideEffects

# 导入数据源模块
from .source import (
    AttrSource,
    BackwardStateSource,
    ConstantSource,
    GetItemSource,
    GlobalStateSource,
    is_constant_source,
    is_from_local_source,
    LocalSource,
    ParamBufferSource,
    ShapeEnvSource,
    SyntheticLocalSource,
    TensorProperty,
    TensorPropertySource,
)

# 导入实用函数模块
from .utils import (
    checkpoint_params,
    CleanupHook,
    clone_inputs,
    count_calls,
    counters,
    dynamo_timed,
    get_instruction_source_311,
    get_locals_to_steal,
    get_static_address_type,
    graph_break_reasons,
    increment_op_count,
    lazy_format_graph_code,
    LazyString,
    nn_module_proxy,
    same,
    set_example_value,
)

# 导入变量追踪模块
from .variables.base import VariableTracker

# 导入变量构建器模块
from .variables.builder import (
    BackwardStateGraphArg,
    GraphArg,
    TrackedFake,
    VariableBuilder,
    wrap_fx_proxy,
)

# 导入列表变量模块
from .variables.lists import BaseListVariable

# 导入杂项变量模块
from .variables.misc import NullVariable

# 导入神经网络模块变量模块
from .variables.nn_module import NNModuleVariable

# 导入张量变量模块
from .variables.tensor import (
    NumpyNdarrayVariable,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)

# 导入 Torch 功能覆盖的张量变量模块
from .variables.torch_function import TensorWithTFOverrideVariable

# 如果是类型检查模式，导入指令翻译器基类
if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)
# 使用 torch._logging 模块获取名为 "graph" 的日志记录器对象，并赋值给 graph_tabular_log 变量
graph_tabular_log = torch._logging.getArtifactLogger(__name__, "graph")
# 使用 torch._logging 模块获取名为 "graph_code" 的日志记录器对象，并赋值给 graph_code_log 变量
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code")
# 使用 torch._logging 模块获取名为 "graph_sizes" 的日志记录器对象，并赋值给 graph_sizes_log 变量
graph_sizes_log = torch._logging.getArtifactLogger(__name__, "graph_sizes")
# 使用 torch._logging 模块获取名为 "trace_call" 的日志记录器对象，并赋值给 trace_call_log 变量
trace_call_log = torch._logging.getArtifactLogger(__name__, "trace_call")

# 定义一个不可变的数据类 VariableTrackerCacheKey，表示变量追踪器缓存的键
@dataclass(frozen=True)
class VariableTrackerCacheKey:
    vt_id: int  # 变量追踪器的 ID
    source: Source  # 源对象，用于指示变量的来源

# 表示变量追踪器的缓存
class VariableTrackerCache:
    def __init__(self):
        self.cache = {}  # 初始化空的缓存字典

    # 查找缓存中是否存在指定值和来源的追踪器
    def lookup(self, value, source):
        key = VariableTrackerCacheKey(id(value), source)
        if key not in self.cache:
            return None
        return self.cache[key]

    # 将值、来源和追踪器对象添加到缓存中
    def add(self, value, source, vt):
        key = VariableTrackerCacheKey(id(value), source)
        self.cache[key] = vt

    # 克隆当前缓存对象，用于复制和恢复图状态
    def clone(self):
        new_cache = VariableTrackerCache()  # 创建一个新的缓存对象
        new_cache.cache.update(self.cache)  # 将当前缓存内容复制到新对象中
        return new_cache

    # 清空缓存
    def clear(self):
        self.cache.clear()

# 使用 functools.lru_cache 装饰器定义 _step_logger 函数，返回 torchdynamo_logging 模块的步骤记录器对象
@functools.lru_cache(None)
def _step_logger():
    return torchdynamo_logging.get_step_logger(log)

# 表示图编译原因的数据类，存储输出图被编译的原因和用户调用堆栈信息
@dataclass
class GraphCompileReason:
    reason: str  # 编译图断裂的原因描述
    user_stack: List[traceback.FrameSummary]  # 用户调用堆栈信息列表

    graph_break: bool = True  # 标识此原因是否是因为图断裂

    def __post_init__(self):
        if self.graph_break:
            graph_break_reasons.append(self)  # 如果是因为图断裂原因，则添加到全局列表中

# 根据随机调用列表创建一个生成随机值函数 _gen_rand_values_fn
def _get_gen_rand_values_fn(random_calls):
    def _gen_rand_values():
        return [fn(*args, **kwargs) for fn, args, kwargs in random_calls]  # 调用 random_calls 中每个函数生成随机值并返回

    return _gen_rand_values

# FakeRootModule 类，用于欺骗 fx.GraphModule 的构造函数
class FakeRootModule(torch.nn.Module):
    def __init__(self, nn_modules: Dict[str, torch.nn.Module]):
        super().__init__()
        for k, v in nn_modules.items():
            setattr(self, k, v)  # 将传入的 nn_modules 中的每个模块设置为当前对象的属性

    def __repr__(self):
        return "FakeRootModule(...)"  # 返回对象的字符串表示形式

# WrapperBackend 类，封装一个编译器函数作为其后端
class WrapperBackend:
    def __init__(self, backend: CompilerFn):
        self.backend: CompilerFn = backend  # 初始化时接收一个编译器函数作为参数并保存在 backend 属性中
    # 定义一个方法，允许对象被调用，接收图模块和示例输入列表作为参数
    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        # 备份当前模型参数
        self.restore = checkpoint_params(gm)
        # 存储传入的图模块对象
        self.gm = gm
        # 深度复制图模块对象，以便后续操作不影响原始对象
        copy_gm = copy.deepcopy(self.gm)
        # 使用后端模型对复制的图模块和示例输入进行计算
        self.candidate = self.backend(copy_gm, example_inputs)

        # 如果候选模型为空或者与原始模型的前向函数相同，则直接返回原始模型的前向函数
        if self.candidate is None or self.candidate is self.gm.forward:
            return self.gm.forward

        # 如果不需要验证正确性，则直接返回候选模型
        if not config.verify_correctness:
            return self.candidate

        # 如果需要验证正确性
        try:
            # 获取原始模型对示例输入的前向计算结果
            correct = self.gm.forward(*clone_inputs(example_inputs))
            # 获取候选模型对示例输入的前向计算结果
            result = self.candidate(*clone_inputs(example_inputs))

            # TODO: 使用测试中的 `same` 函数替换此处的比较方式
            # 比较两个计算结果是否相同
            if same(correct, result):
                return self.candidate

            # 如果计算结果不同，则抛出运行时异常
            raise RuntimeError(f"incorrect results of backend {self}")

            # 返回原始模型的前向函数
            return self.gm.forward

        # 捕获所有异常
        except Exception:
            # 记录异常信息
            log.exception("error in verify_correctness")
            # 重新抛出异常
            raise

        # 无论是否发生异常，最终都会执行的清理操作
        finally:
            # 恢复模型参数到之前的状态
            self.restore()
# Scope 类型定义，表示一个字典，其键为字符串，值为任意对象
Scope = Dict[str, object]

# OutputGraph 类用于包装 InstructionTranslator 的输出，主要是生成的 fx.Graph。
# 每个 OutputGraph 对象与一个正在处理的帧相关联。每个帧都与某个根 InstructionTranslator 相关联。
# 当用户代码调用函数时，我们构造一个 InliningInstructionTranslator，它继续向根 InstructionTranslator 的 OutputGraph 写入。
class OutputGraph:
    """
    Wrapper class to hold outputs of InstructionTranslator.  Mainly the
    generated fx.Graph.

    OutputGraph is 1:1 with a frame being processed. Each frame is associated
    with some root InstructionTranslator. When user code calls a function,
    we construct a InliningInstructionTranslator that continues to write into
    the root InstructionTranslator's OutputGraph.
    """

    # OutputGraph 的构造函数，初始化对象
    def __init__(
        self,
        code_options: Dict[str, Any],
        compiler_fn: Optional[CompilerFn],
        root_tx,  # 根 InstructionTranslator
        export: bool,
        export_constraints,  # 导出约束
        frame_state,  # 帧状态
        local_scope: Scope,  # 局部作用域
        global_scope: Scope,  # 全局作用域
        f_code,  # 函数代码对象
    ):
    
    # 在 f_globals["__builtins__"] 中安装内置字典
    def install_builtins_dict_in_fglobals(self):
        # f_globals["__builtins__"] 可以是字典或模块。这是一个实现细节 -
        # https://docs.python.org/3/library/builtins.html.

        # 这使得在任何内置对象上进行保护检查变得复杂，因为保护检查函数
        # 必须检查 __builtins__ 是否是模块或字典，然后相应地使用 getattr 或 getitem 访问。

        # 为了解决这个问题，我们在 f_globals 中插入一个新条目，指向内置字典 __dict__，
        # 然后我们在这个字典上保护任何内置对象。
        # 为了避免与现有键发生冲突，我们使用 install_global 来给我们一个唯一的字典键。

        f_builtins = self.global_scope["__builtins__"]
        if not isinstance(f_builtins, dict):
            f_builtins = f_builtins.__dict__
        return self.install_global("__builtins_dict__", f_builtins)

    # 向 OutputGraph 中添加后向状态钩子
    def add_backward_state_hook(self, hook: VariableTracker, prefix="hook"):
        name = f"{prefix}{len(self.backward_state)}"
        assert name not in self.backward_state
        self.backward_state[name] = hook
        return name, self.get_backward_state_proxy()

    # 获取后向状态的代理
    def get_backward_state_proxy(self):
        if self.backward_state_proxy is None:
            if self.export:
                unimplemented("backward_state does not support export")
            self.backward_state_proxy = self.root_tracer.create_graph_input(
                "dynamo_backward_state", BackwardState, source=BackwardStateSource()
            )
            self.backward_state_proxy.node.meta["grapharg"] = BackwardStateGraphArg()
            set_example_value(self.backward_state_proxy.node, BackwardState())
            self.backward_state_var = self.new_var()
        return self.backward_state_proxy

    # 这个方法有自己的辅助函数，以便 DEBUG 日志更加详细
    def init_ambient_guards(self):
        # Register a SHAPE_ENV guard to make sure we setup shape guards
        # that show up in ShapeEnv
        self.guards.add(ShapeEnvSource().make_guard(GuardBuilder.SHAPE_ENV))

        # Add a guard for DETERMINISTIC_ALGORITHMS to ensure deterministic behavior
        self.guards.add(
            GlobalStateSource().make_guard(GuardBuilder.DETERMINISTIC_ALGORITHMS)
        )

        # Add a guard for GRAD_MODE to manage gradient computation mode
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.GRAD_MODE))

        # Add a guard for DEFAULT_DEVICE to manage the default device for operations
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.DEFAULT_DEVICE))

        # Add a guard for TORCH_FUNCTION_STATE to track Torch function state
        self.guards.add(
            GlobalStateSource().make_guard(GuardBuilder.TORCH_FUNCTION_STATE)
        )

        # Peek at the interpreter stack for Functorch and add a guard if applicable
        ci = torch._C._functorch.peek_interpreter_stack()
        if ci is not None:
            self.guards.add(
                GlobalStateSource().make_guard(GuardBuilder.FUNCTORCH_STACK_MATCH)
            )

    def synthetic_graph_input(self, fn, args):
        """
        call fn(*args) before the graph runs and turn the result into a fake input.
        """
        example_value = fn(*args)
        varname = self.new_var()
        cg = PyCodegen(self.root_tx)
        cg.add_push_null(
            lambda: cg.load_import_from(
                fn.__module__,
                fn.__name__,
            )
        )
        cg.foreach(map(variables.ConstantVariable.create, args))
        cg.call_function(len(args), False)
        cg.store(varname)
        self.pregraph_bytecode.extend(cg.get_instructions())
        source = SyntheticLocalSource(varname)
        result = VariableBuilder(self.root_tx, source)(example_value)
        TracingContext.get().guards_context.dynamo_guards.remove_guards_with_source(
            source
        )
        return result

    def add_cleanup_hook(self, fn: Callable[[], Any]):
        # Add a cleanup hook function to be executed later
        self.cleanup_hooks.append(fn)

    def call_cleanup_hooks(self):
        # Execute cleanup hooks in reverse order and clear the list afterwards
        for hook in reversed(self.cleanup_hooks):
            hook()
        self.cleanup_hooks.clear()

    @property
    def root_tracer(self):
        # Return the root tracer object from the list of tracers
        return self.tracers[0]

    @property
    def current_tracer(self):
        # Return the current tracer object from the end of the tracers list
        return self.tracers[-1]

    def is_root_tracer(self):
        # Helper method to check if we are in the root tracer
        return len(self.tracers) == 1

    @property
    def graph(self):
        # Return the graph associated with the current tracer
        return self.current_tracer.graph

    # Setter method for 'graph' property to set the graph for the current tracer
    @graph.setter
    def graph(self, value):
        self.current_tracer.graph = value

    @property
    def input_name_to_proxy(self):
        # Return the mapping of input names to proxies from the current tracer
        return self.current_tracer.input_name_to_proxy

    @property
    def real_value_cache(self):
        # Return the real value cache from the current tracer
        return self.current_tracer.real_value_cache

    # Note from rzou: can delete after we refactor speculate_subgraph to use nested GraphTracer.
    # Property with a message guiding users to call create_graph_input on appropriate tracer instance
    # for clarity and to avoid ambiguity.
    # 调用当前追踪器的 create_proxy 方法，并返回结果
    def create_proxy(self, *args, **kwargs):
        return self.current_tracer.create_proxy(*args, **kwargs)

    # 调用当前追踪器的 create_node 方法，并返回结果
    def create_node(self, *args, **kwargs):
        return self.current_tracer.create_node(*args, **kwargs)

    # 调用当前追踪器的 remove_node 方法，并返回结果
    def remove_node(self, *args, **kwargs):
        return self.current_tracer.remove_node(*args, **kwargs)

    # 上下文管理器，用于创建子追踪器（tracer），管理作用域
    @contextlib.contextmanager
    def subtracer(self, source_target, prior_tracer):
        # 进入新的作用域上下文
        new_scope_ctx = enter_new_scope()
        try:
            if prior_tracer:
                # 断言：保持祖先关系不变
                assert prior_tracer.parent is self.current_tracer
            # 进入新的作用域上下文
            new_scope_ctx.__enter__()
            # 如果 prior_tracer 存在，则使用它作为新 tracer，否则创建新的 SubgraphTracer
            tracer = (
                prior_tracer
                if prior_tracer
                else SubgraphTracer(
                    self, parent=self.current_tracer, source_target=source_target
                )
            )
            # 将 tracer 添加到追踪器列表中
            self.tracers.append(tracer)
            # 生成 tracer，并将其传递给 yield 语句的调用方
            yield tracer
        finally:
            # 退出作用域上下文
            new_scope_ctx.__exit__(None, None, None)
            # 移除最后添加的 tracer
            self.tracers.pop()

    # 返回当前对象自身
    @property
    def output(self):
        return self

    # 返回追踪上下文中的 fake_mode 属性
    @property
    def fake_mode(self):
        return self.tracing_context.fake_mode

    # 返回追踪上下文中的 fake_mode.shape_env 属性
    @property
    def shape_env(self):
        return self.tracing_context.fake_mode.shape_env

    # 返回追踪上下文中的 guards_context.dynamo_guards 属性
    @property
    def guards(self) -> torch._guards.GuardsSet:
        return self.tracing_context.guards_context.dynamo_guards

    # 返回追踪上下文中的 module_context.nn_modules 属性
    @property
    def nn_modules(self) -> Dict[str, Any]:
        return self.tracing_context.module_context.nn_modules
    def save_global_state(self, out=None):
        """
        Saves to out if it is provided. Else saves to the tracing context's global_state.
        """
        # 确定要保存的全局状态对象，如果提供了out参数则使用out，否则使用tracing context的全局状态
        global_state = (
            out if out is not None else self.tracing_context.global_context.global_state
        )

        # 设置torch函数状态的全局变量
        global_state["torch_function_enabled"] = (
            self.set_torch_function_state,
            self.torch_function_enabled,
        )
        # 设置梯度计算状态的全局变量
        global_state["grad_enabled"] = (torch.set_grad_enabled, torch.is_grad_enabled())

        # 设置CUDA自动混合精度计算状态的全局变量
        global_state["autocast_enabled"] = (
            functools.partial(torch.set_autocast_enabled, "cuda"),
            torch.is_autocast_enabled("cuda"),
        )
        # 设置CPU自动混合精度计算状态的全局变量
        global_state["autocast_cpu_enabled"] = (
            functools.partial(torch.set_autocast_enabled, "cpu"),
            torch.is_autocast_enabled("cpu"),
        )
        # 设置CUDA自动混合精度计算的数据类型的全局变量
        global_state["autocast_gpu_dtype"] = (
            functools.partial(torch.set_autocast_dtype, "cuda"),
            torch.get_autocast_dtype("cuda"),
        )
        # 设置CPU自动混合精度计算的数据类型的全局变量
        global_state["autocast_cpu_dtype"] = (
            functools.partial(torch.set_autocast_dtype, "cpu"),
            torch.get_autocast_dtype("cpu"),
        )
        # 设置自动混合精度计算缓存状态的全局变量
        global_state["autocast_cache_enabled"] = (
            torch.set_autocast_cache_enabled,
            torch.is_autocast_cache_enabled(),
        )

    def push_tx(self, tx):
        # 将交易对象tx推入当前交易栈
        self._current_tx.append(tx)

    def pop_tx(self):
        # 从当前交易栈弹出一个交易对象tx并返回
        return self._current_tx.pop()

    @property
    def current_tx(self):
        # 返回当前交易对象，若当前交易栈为空则返回根交易对象root_tx
        return self.root_tx if not self._current_tx else self._current_tx[-1]
    # 添加符号绑定到图中的参数
    def add_symbol_bindings(self, arg: GraphArg):
        # 如果是导出模式，则直接返回，不做任何操作
        if self.export:
            return
        
        # 断言确保假张量不为空
        assert arg.fake_tensor is not None
        
        # 定义函数，用于绑定符号整数到图中
        def bind_symint(s, prop):
            # 如果符号不是符号表达式，则直接返回
            if not (is_symbolic(s) and isinstance(s.node.expr, sympy.Symbol)):
                return
            s0 = s.node.expr
            # 如果符号已经绑定过，则直接返回
            if s0 in self.bound_symbols:
                return
            # 将符号添加到已绑定符号集合中
            self.bound_symbols.add(s0)
            log.debug("bind_symint %s %s", s, prop.name())
            # TODO: 如果图中已经存在该符号整数，则不需要重新添加（因为后续会移除未使用的）
            # 创建符号整数的图输入代理
            proxy = self.root_tracer.create_graph_input(
                str(s0),
                torch.SymInt,
                before=True,
                source=prop,
            )
            # 设置代理节点的示例值为当前符号整数的值
            set_example_value(proxy.node, s)
            # 将图参数相关信息存储到代理节点的元数据中
            proxy.node.meta["grapharg"] = GraphArg(
                prop,
                s,
                pass_arg_as_tensor=False,
                fake_tensor=None,
                is_tensor=False,
            )

        # 处理张量，绑定张量大小相关的符号整数到图中
        def handle_tensor(t, src):
            for i, s in enumerate(t.size()):
                bind_symint(s, TensorPropertySource(src, TensorProperty.SIZE, i))
            # 如果张量布局为 strided
            if t.layout is torch.strided:
                for i, s in enumerate(t.stride()):
                    bind_symint(s, TensorPropertySource(src, TensorProperty.STRIDE, i))
                # 绑定张量的存储偏移符号整数到图中
                bind_symint(
                    t.storage_offset(),
                    TensorPropertySource(src, TensorProperty.STORAGE_OFFSET),
                )
            # 如果张量布局为 sparse_coo
            elif t.layout is torch.sparse_coo:
                handle_tensor(t._indices(), src)
                handle_tensor(t._values(), src)
            # 如果张量布局为 sparse_csr 或者 sparse_bsr
            elif t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                handle_tensor(t.crow_indices(), src)
                handle_tensor(t.col_indices(), src)
            # 如果张量布局为 sparse_csc 或者 sparse_bsc
            elif t.layout in {torch.sparse_csc, torch.sparse_bsc}:
                handle_tensor(t.ccol_indices(), src)
                handle_tensor(t.row_indices(), src)
            # 如果张量是可追踪包装器的子类
            if is_traceable_wrapper_subclass(t):
                # 展开张量属性并递归处理内部张量
                attrs, ctx = t.__tensor_flatten__()
                for attr in attrs:
                    inner_t = getattr(t, attr)
                    handle_tensor(inner_t, AttrSource(src, attr))

        # 调用处理张量的函数，处理假张量
        handle_tensor(arg.fake_tensor, arg.source)
    def get_submodule(self, keys):
        # 确保 keys 不为空
        assert keys
        # 初始时，将 self.nn_modules 赋给 obj，可以是 Module 或字典
        obj: Union[torch.nn.Module, Dict[str, torch.nn.Module]] = self.nn_modules
        # 根据 keys 中的点号分隔键，逐层获取子模块或属性
        for k in keys.split("."):
            if isinstance(obj, dict):
                obj = obj[k]  # 如果当前对象是字典，则获取键为 k 的值
            else:
                obj = getattr(obj, k)  # 否则，获取属性 k 的值
        # 返回获取的子模块或属性对象
        return obj

    def new_var(self, name="tmp"):
        existing = set(self.code_options["co_varnames"])
        # 在常见情况下，这是 O(1) 的操作
        while True:
            # 生成一个新的变量名，确保唯一性
            var = f"{name}_{next(self.unique_var_id)}"
            if var not in existing:
                self.code_options["co_varnames"] += (var,)  # 将新变量名添加到变量名列表中
                return var

    def update_co_names(self, name):
        """确保 self.code_options.co_names 包含 name"""
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)  # 将 name 添加到代码选项的名称列表中

    @staticmethod
    def module_key_name(*names):
        # 创建一个新的唯一名称
        name = "_".join(map(str, names))
        # 去除 L/G 保护查找的访问
        name = re.sub(r"^[GL]\['?(.*?)'?\]$", r"\1", name)
        # 例如，将 abc.xyz[123].qkv 替换为 abc.xyz_123.qkv
        name = re.sub(r"\[(\d+)\]", r"_\g<1>", name)
        # 例如，将 abc.xyz_123.qkv 替换为 abc_xyz_123_qkv
        name = re.sub(r"[^a-zA-Z0-9]", "_", name)

        if not name or not name[0].isalpha():
            name = "sub" + name  # 如果名称为空或不以字母开头，则添加前缀 "sub"

        return name

    def register_attr_or_module(
        self,
        target: Union[torch.nn.Module, torch.Tensor, Any],
        *names,
        **options,
    # 处理被窃取列表的别名，以保持它们在函数调用后的有效性
    def handle_aliases_for_stolen_lists(self, tx):
        # 如果被窃取的列表输入在函数调用后仍然需要，创建别名以保持其有效
        maybe_gm = self.local_scope.get("self")
        # 获取可能的全局变量 "self"
        stolen_list_names = get_locals_to_steal(maybe_gm)
        # 获取需要被窃取的局部变量名列表

        if not stolen_list_names:
            return []
        # 如果没有需要被窃取的局部变量，直接返回空列表

        alias_insts = []
        # 别名指令列表

        needs_alias: Dict[
            str, List[Union[VariableTracker, AttributeMutationExisting]]
        ] = {}
        # 需要别名的变量字典，键为变量名，值为变量追踪器或属性变异对象的列表

        queue = [
            *tx.stack,
            *tx.symbolic_locals.values(),
            *self.side_effects.store_attr_mutations.keys(),
        ]
        # 初始化队列，包含事务栈、符号化局部变量值、以及存储属性变异的键

        while queue:
            x = queue.pop()
            # 从队列中取出一个元素
            if isinstance(x, BaseListVariable):
                assert isinstance(x.items, List)
                # 断言 x 的 items 属性是一个列表
                queue += x.items
                continue
            # 如果 x 是 BaseListVariable 类型，则将其 items 属性加入队列继续处理

            if not (
                isinstance(x, (VariableTracker, AttributeMutationExisting))
                and isinstance(x.source, GetItemSource)
                and isinstance(x.source.base, LocalSource)
                and x.source.base.local_name in stolen_list_names
            ):
                continue
            # 如果 x 不是 VariableTracker 或 AttributeMutationExisting 类型，或者其来源不是 GetItemSource，或者来源的基本本地名称不在被窃取列表中，则继续下一轮循环

            stolen_name = x.source.base.local_name
            # 获取被窃取的列表名
            if stolen_name not in needs_alias:
                needs_alias[stolen_name] = []
            # 如果被窃取的列表名不在需要别名的字典中，则将其添加进去
            needs_alias[stolen_name].append(x)
            # 将 x 添加到需要别名的列表中

        visited = {}
        # 访问过的别名字典

        for arg in self.graphargs:
            # 遍历图形参数
            if not (
                isinstance(arg._example, list)
                and isinstance(arg.source, LocalSource)
                and arg.source.local_name in needs_alias
            ):
                continue
            # 如果参数不是列表类型，或者其来源不是本地变量，或者其本地名称不在需要别名的列表中，则继续下一轮循环

            # arg 是一个将由编译函数清除的列表
            list_name = arg.source.local_name
            # 获取列表的本地名称
            assert list_name in self.code_options["co_varnames"]
            # 断言列表名在代码选项的变量名称中存在
            for x in needs_alias[list_name]:
                # 遍历需要别名的列表名
    def codegen_suffix(self, tx, stack_values, cg):
        if self.backward_state:
            assert not self.export
            for name, val in self.backward_state.items():
                cg(val)  # 生成代码，处理后向状态的值
                cg.append_output(cg.create_load(self.backward_state_var))  # 将后向状态变量加载到生成的代码中
                cg.store_attr(name)  # 存储处理后的值到属性名为name的对象中
        self.side_effects.codegen_hooks(cg)  # 生成代码，处理副作用的钩子
        self.side_effects.codegen_save_tempvars(cg)  # 生成代码，保存临时变量

        # 返回用于调试记录的变量
        for debug_var, args in tx.debug_locals:
            cg.add_push_null(lambda: cg(debug_var))  # 添加生成代码，将调试变量推入堆栈
            for arg in args:
                cg(arg)  # 生成代码，处理每个调试变量的参数
            cg.extend_output(create_call_function(len(args), False))  # 扩展生成的输出，调用函数并传入参数
            cg.extend_output([create_instruction("POP_TOP")])  # 扩展生成的输出，弹出栈顶元素

        cg.restore_stack(stack_values, value_from_source=not tx.export)  # 生成代码，恢复堆栈值，根据是否导出获取值来源
        self.side_effects.codegen_update_mutated(cg)  # 生成代码，更新已变异的副作用

    def cleanup_graph(self):
        """
        Remove "creation_timestamp" from node meta

        Remove this pattern from the graph:
            torch._C._set_grad_enabled(False)
            torch._C._set_grad_enabled(True)
        """
        assert self.should_exit  # 断言，应该退出当前操作
        nodes = list(self.graph.nodes)  # 获取图中所有节点的列表
        for node in nodes:
            node.meta.pop("creation_timestamp", None)  # 从节点的元数据中移除"creation_timestamp"键

        grad_enabled = torch.is_grad_enabled()  # 获取当前梯度是否启用的状态
        for node1, node2 in zip(nodes, nodes[1:]):
            if (
                node1.target is torch._C._set_grad_enabled
                and tuple(node1.args) == (not grad_enabled,)
                and not node1._erased
            ):
                grad_enabled = node1.args[0]  # 更新梯度启用状态
                if (
                    node2.target is torch._C._set_grad_enabled
                    and tuple(node2.args) == (not grad_enabled,)
                    and not node2._erased
                ):
                    grad_enabled = node2.args[0]  # 更新梯度启用状态
                    self.graph.erase_node(node1)  # 从图中删除节点1
                    self.graph.erase_node(node2)  # 从图中删除节点2

    def get_graph_sizes_structured(self):
        ret = {}  # 初始化返回的空字典
        for node in self.graph.nodes:
            example_value = node.meta.get("example_value", None)  # 获取节点元数据中的"example_value"，默认为None
            if isinstance(example_value, torch._subclasses.FakeTensor):
                size = example_value.size()  # 获取示例值的大小
                ret[node.name] = [s if isinstance(s, int) else repr(s) for s in size]  # 将节点名称映射到示例值的大小列表
        return ret  # 返回结构化的图节点大小字典
    # 返回一个描述跟踪图中节点尺寸的字符串，包括给定名称的标题
    def get_graph_sizes(self, name: str):
        # 初始化跟踪图尺寸的字符串，包括标题
        graph_sizes_str = "TRACED GRAPH TENSOR SIZES\n"
        graph_sizes_str += f"===== {name} =====\n"
        # 遍历图中的每个节点
        for node in self.graph.nodes:
            # 获取节点的示例值（如果有），通常为 FakeTensor
            example_value = node.meta.get("example_value", None)
            # 如果示例值是 FakeTensor 类型
            if isinstance(example_value, torch._subclasses.FakeTensor):
                # 获取其尺寸并添加到 graph_sizes_str 中
                size = example_value.size()
                graph_sizes_str += f"{node.name}: {tuple(size)}\n"
                # 初始化一个列表来存储具体的尺寸值，包括可能的符号整数
                concrete_size = []
                has_symint = False
                # 遍历尺寸中的每个维度
                for sz in size:
                    # 如果是整数，直接添加到 concrete_size
                    if isinstance(sz, int):
                        concrete_size.append(sz)
                    # 如果是 SymInt 类型，记录其提示值并设置标志
                    elif isinstance(sz, torch.SymInt):
                        has_symint = True
                        concrete_size.append(sz.node.hint)
                    else:
                        break
                else:
                    # 如果存在符号整数，将具体尺寸信息添加到 graph_sizes_str 中
                    if has_symint:
                        graph_sizes_str += (
                            f"{node.name} (concrete): {tuple(concrete_size)}\n"
                        )
        # 返回描述跟踪图节点尺寸的完整字符串
        return graph_sizes_str

    @contextlib.contextmanager
    def restore_global_state(self):
        """
        临时将全局状态恢复到跟踪当前输出之前的状态
        """
        # 备份当前全局状态
        prior_global_state = self.tracing_context.global_context.copy_graphstate()
        # 初始化当前全局状态为一个空字典
        current_global_state: Dict[str, Tuple[Any, bool]] = {}
        # 保存当前全局状态到 current_global_state 中
        self.save_global_state(out=current_global_state)
        try:
            # 在 trace 图形之前恢复到先前的全局状态
            self.tracing_context.global_context.restore_graphstate(prior_global_state)
            yield
        finally:
            # 在当前时间点（例如在调用用户编译器之前）恢复到当前全局状态
            self.tracing_context.global_context.restore_graphstate(
                GlobalContextCheckpointState(current_global_state)
            )

    @torch._guards.TracingContext.clear_frame()
    @property
    def placeholders(self) -> List[fx.Node]:
        # 返回图中操作为 "placeholder" 的节点列表作为属性 placeholders
        return self.graph.find_nodes(op="placeholder")

    @property
    def graphargs(self) -> List[GraphArg]:
        # 返回所有占位符节点中的 "grapharg" 元数据作为属性 graphargs
        return [node.meta["grapharg"] for node in self.placeholders]

    @dynamo_timed(phase_name="backend_compile")
    # 调用用户编译器函数，将图模块作为参数传入，返回编译后的函数对象
    def call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:
        # 确保编译器函数已定义
        assert self.compiler_fn is not None
        
        # 初始化操作总数
        tot = 0
        # 存储所有占位符节点的列表
        placeholders = []
        
        # 遍历图模块中的所有节点
        for node in gm.graph.nodes:
            # 统计调用函数、方法或模块的节点数量
            if node.op in ("call_function", "call_method", "call_module"):
                tot += 1
            # 如果节点是占位符，则将其添加到占位符列表中
            if node.op == "placeholder":
                placeholders.append(node)
        
        # 调用外部函数增加操作总数
        increment_op_count(tot)
        
        # 针对每个占位符节点，设置其动态源
        for pl in placeholders:
            arg = pl.meta["grapharg"]
            # TODO: 为什么这个没有存储在 meta 中？ :think:
            pl._dynamo_source = arg.source
        
        # 将参数名到源的映射赋给图模块对象的属性
        gm._param_name_to_source = self.param_name_to_source  # type: ignore[assignment]
        # 将源到用户堆栈的映射赋给图模块对象的属性
        gm._source_to_user_stacks = self.source_to_user_stacks  # type: ignore[assignment]
        
        try:
            # 获取编译器函数的名称
            name = (
                self.compiler_fn.__name__
                if hasattr(self.compiler_fn, "__name__")
                else ""
            )
            # 记录调用编译器函数的日志信息
            _step_logger()(logging.INFO, f"calling compiler function {name}")
            
            # 获取编译器函数
            compiler_fn = self.compiler_fn
            # 如果配置要求验证正确性，则使用 WrapperBackend 包装编译器函数
            if config.verify_correctness:
                compiler_fn = WrapperBackend(compiler_fn)
            
            # 调用编译器函数，传入图模块和示例输入，获取编译后的函数对象
            compiled_fn = compiler_fn(gm, self.example_inputs())
            
            # 记录编译器函数执行完毕的日志信息
            _step_logger()(logging.INFO, f"done compiler function {name}")
            
            # 确保编译后的函数对象是可调用的
            assert callable(compiled_fn), "compiler_fn did not return callable"
        
        # 捕获允许作为回退的特定异常
        except exceptions_allowed_to_be_fallback as e:
            # 如果允许在图中定义的异常存在，则抛出后端编译器失败的异常
            if self.has_user_defined_allowed_in_graph:
                raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
                    e.__traceback__
                ) from None
            # 否则，记录后端编译器使用虚假张量异常失败的信息
            msg = (
                "Backend compiler failed with a fake tensor exception at \n"
                f"{self.root_tx.format_frame_summary()}"
                "Adding a graph break."
            )
            # 在消息中使用未实现的警告
            unimplemented_with_warning(e, self.root_tx.f_code, msg)
        
        # 捕获请求跳过当前帧的异常
        except SkipFrame as e:
            # 后端编译器请求跳过当前帧而非中止执行
            raise e
        
        # 捕获其他任何异常
        except Exception as e:
            # 抛出后端编译器失败的异常
            raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
                e.__traceback__
            ) from None
        
        # 记录标记事件，指示调用用户编译器的操作完成
        signpost_event(
            "dynamo",
            "OutputGraph.call_user_compiler",
            {
                **self.co_fields,
                "op_count": tot,
                "node_count": len(gm.graph.nodes),
                "input_count": len(placeholders),
            },
        )
        
        # 返回编译后的函数对象
        return compiled_fn

    # 返回所有图参数的示例输入列表
    def example_inputs(self) -> List[torch.Tensor]:
        result = []
        # 遍历所有图参数，将其示例输入添加到结果列表中
        for arg in self.graphargs:
            result.append(arg.example)
        return result
    def add_output_instructions(self, prefix: List[Instruction]) -> None:
        """
        在创建新的编译子图并插入用户代码之前调用此方法。
        """
        # 将指令列表 prefix 添加到输出指令列表中
        self.output_instructions.extend(prefix)
        # 设置 should_exit 标志为 True，表示需要退出
        self.should_exit = True

    def install_global_unsafe(self, name, value) -> None:
        """
        警告：更推荐使用安全的 `install_global_by_id/install_global` 方法。
        torch.compile 实例应该彼此独立；
        一个常见的问题是一个实例依赖于另一个实例安装的全局变量的存在。
        这可能发生在我们对两个实例中的全局变量进行相同方式的处理时。
        """
        # 断言确保要安装的全局变量名不在已安装的全局变量集合中
        assert name not in self.installed_globals
        # 将全局变量名 name 添加到已安装全局变量集合中
        self.installed_globals.add(name)
        # 创建一个清理钩子，用于清理时执行
        self.cleanups.append(CleanupHook.create(self.global_scope, name, value))

    def install_global_by_id(self, prefix, value) -> str:
        """
        如果尚未安装，则安装一个全局变量。
        这由 (prefix, id(value)) 对决定。

        返回新安装的全局变量的名称。
        """
        # 根据 value 的 id 和编译实例的编号创建全局变量名
        name = f"{prefix}_{id(value)}_c{self.compile_id}"
        # 如果该全局变量名已经在已安装的全局变量集合中，则直接返回
        if name in self.installed_globals:
            return name
        # 否则，使用不安全的方式安装全局变量
        self.install_global_unsafe(name, value)
        return name

    def install_global(self, prefix, value) -> str:
        """
        安装一个全局变量，并为其生成一个唯一的名称。

        返回新安装的全局变量的名称。
        """
        # 使用唯一的 id 为前缀生成全局变量名
        name = unique_id(prefix)
        # 使用不安全的方式安装全局变量
        self.install_global_unsafe(name, value)
        return name

    def cleanup(self) -> None:
        # 修正了 tracer 和 OutputGraph 之间的引用循环问题，
        # 导致一些张量对象的生存时间比必要的长。
        self.root_tx = None
        self.nn_modules.clear()
        self.param_name_to_source = None

        # 清理节点的元数据中的 "grapharg" 键
        for node in self.graph.nodes:
            if "grapharg" in node.meta:
                del node.meta["grapharg"]

        # 清空实际值缓存
        self.real_value_cache.clear()
        # 清空输入名到代理对象的映射
        self.input_name_to_proxy.clear()
        # 清空副作用集合
        self.side_effects.clear()
        # 清空变量追踪器缓存
        self.variable_tracker_cache.clear()
        # 清空注册的最终化函数列表
        self.register_finalizer_fns.clear()
        # 清空动态扁平名称到原始全限定名的映射
        self.dynamo_flat_name_to_original_fqn.clear()
        # 清空追踪上下文
        self.tracing_context.clear()

    def set_torch_function_state(self, enabled: bool) -> None:
        # 设置 torch 函数的状态
        self.torch_function_enabled = enabled

    def add_graph_finalizer(
        self, register_finalizer: Callable[[fx.GraphModule], None]
    ) -> None:
        # 将注册的最终化函数添加到列表中
        self.register_finalizer_fns.append(register_finalizer)
    def example_value_from_input_node(self, node: torch.fx.Node):
        """Extract the non-fake example tensor"""
        # 如果节点操作是"placeholder"，返回节点元数据中的示例张量
        if node.op == "placeholder":
            return node.meta["grapharg"].example
        # 如果节点操作是"get_attr"，则断言节点目标在nn_modules中存在，并返回对应的模块
        assert node.op == "get_attr"
        return self.nn_modules[node.target]  # type: ignore[index]
# 错误处理的结尾语句，当配置不符合要求时，将在所有没有 'pt2_compliant_tag' 标签的操作上图并退回到即时模式的 PyTorch。
err_epilogue = (
    "With the current config, we will graph break "
    "(and fall back to eager-mode PyTorch) on all ops "
    "that have do not have the 'pt2_compliant_tag'. "
    "Please see the following doc for how to mark this op as PT2 compliant "
    "https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html"
)


def check_pt2_compliant_op(output_graph, kind, target, args, kwargs):
    if kind != "call_function":
        return

    # 检查是否遇到了符合 PT2 标准的操作
    def encountered_compliant_op(target):
        if target.namespace in {"prim", "prims", "aten"}:
            return
        output_graph.compliant_custom_ops.add(target)

    # 检查是否遇到了不符合 PT2 标准的操作
    def encountered_non_compliant_op(target, msg):
        output_graph.non_compliant_ops.add(target)
        if config.only_allow_pt2_compliant_ops:
            unimplemented(msg + " " + err_epilogue)

    if isinstance(target, torch._ops.OpOverload):
        # 如果操作是 OpOverload 类型
        if torch.Tag.pt2_compliant_tag in target.tags:
            encountered_compliant_op(target)
            return
        # 如果不符合 PT2 标准，则调用处理不符合操作的函数
        encountered_non_compliant_op(
            target,
            f"Encountered the torch.ops.OpOverload {target} "
            f"that is not PT2 compliant.",
        )
        return

    if isinstance(target, torch._ops.OpOverloadPacket):
        # 如果操作是 OpOverloadPacket 类型
        overloads = tuple(target.overloads())
        # 优化：过载解析很昂贵。如果只有一个过载，我们知道它会解析到什么。
        if len(overloads) == 1:
            op = getattr(target, overloads[0])
            if torch.Tag.pt2_compliant_tag in op.tags:
                encountered_compliant_op(op)
                return
            encountered_non_compliant_op(
                op,
                f"Encountered the non-overloaded "
                f"torch.ops.OpOverloadPacket {target} "
                f"that is not PT2 compliant. ",
            )
            return

        # 从节点获取虚假值以进行过载解析
        args, kwargs = torch._dynamo.utils.get_fake_values_from_nodes(
            output_graph.current_tx, (args, kwargs), False
        )
        try:
            # 解析 OpOverloadPacket 的过载
            overload = torch._C._jit_resolve_packet(
                target._qualified_op_name, *args, **kwargs
            )
        except RuntimeError as e:
            unimplemented(str(e))

        # 获取解析后的操作
        op = getattr(target, overload)
        if torch.Tag.pt2_compliant_tag in op.tags:
            encountered_compliant_op(op)
        else:
            encountered_non_compliant_op(
                op,
                f"Encountered the torch.ops.OpOverloadPacket {target} "
                f"which resolves to the overload ({overload}) that is "
                f"not PT2 compliant.",
            )


# 编译 ID 计数器，使用 itertools 进行计数
_compile_id_counter = itertools.count()


class SubgraphTracer(fx.Tracer):
    """
    持有被追踪的 FX 图的子图追踪器。OutputGraph 拥有 SubgraphTracer，
    负责建立图形，而 OutputGraph 则负责编译和执行图形。
    """
    def __init__(
        self, output_graph, parent=None, export_root=False, source_target=None
    ):
        super().__init__()  # 调用父类的初始化方法

        self.output_graph = weakref.proxy(output_graph)  # 使用弱引用代理output_graph
        self.graph = torch.fx.Graph()  # 创建一个torch.fx.Graph对象

        # 导出仅对ROOT追踪器设置，控制是否允许添加某些输入。
        # 查看create_graph_input的调用位置以了解其用法。
        if export_root:
            assert parent is None  # 如果export_root为True，则parent必须为None

        self.export_root = export_root  # 设置是否为根追踪器的标志

        # 将图输入名映射到其占位符代理对象的字典，其中键给出当前所有占位符节点的名称，并可用于创建唯一的节点名称
        self.input_name_to_proxy: Dict[str, fx.Proxy] = {}

        # 节点 => 计算得到的实际值的缓存（参见utils.get_real_value）
        self.real_value_cache: Dict[fx.Node, torch.Tensor] = {}

        # SubgraphTracers可以是嵌套的。见NOTE [HigherOrderOperator tracing design]
        self.parent = parent  # 设置父追踪器对象

        # 一个字典，将先前的自由变量（Proxy对象）映射到包装到此子图输入的新Proxy对象
        # 此字典有两个目的：
        # - Proxies与VariableTracker关联。如果我们两次看到相同的VariableTracker（它是自由变量），
        #   那么我们希望在当前子图中使用相同的Proxy记录跟踪。
        # - 如果我们正在跟踪HigherOrderOperator的body_fn，则需要跟踪被提升的自由变量，以便我们可以使用跟踪的body_fn重写HigherOrderOperator调用。
        # 字典保持了HigherOrderOperator调用参数的顺序。
        self.lifted_freevars = {}

        self.prev_inst = None  # 初始化前一个实例为None

        self._cur_code = None  # 当前代码的占位符
        self._orig_gm_meta = None  # 原始的图模型元数据
        self._orig_gm_lineno_map = None  # 原始的图模型行号映射
        self._orig_gm_firstlineno = None  # 原始的图模型首行号

        # 每个SubgraphTracer与一个源目标相关联，指示此子图附加到哪个操作符。
        # 我们基于源目标计算一个source_fn_stack。对于根追踪器，它被设置为[]。
        # 这对于调试和转换导出的图形非常有用。
        if self.parent is None:
            self.source_fn_stack = []  # 如果没有父追踪器，则源函数堆栈为空列表
        else:
            self.source_fn_stack = self.parent.source_fn_stack + [
                (self.graph._target_to_str(source_target), source_target)
            ]  # 否则，基于父追踪器的源函数堆栈创建当前的源函数堆栈

    def create_proxy(
        self,
        kind,
        target,
        args,
        kwargs,
        name=None,
        type_expr=None,
        proxy_factory_fn=None,
    ):
        pass  # 方法体未提供，暂不需要注释

    def create_node(
        self, op, target, args=None, kwargs=None, name=None, type_expr=None
    ):
        pass  # 方法体未提供，暂不需要注释
    ):
        # 检查节点是否符合 Checkpoint 2 的兼容性，使用输出图、操作、目标、参数和关键字参数进行检查
        check_pt2_compliant_op(self.output_graph, op, target, args, kwargs)
        if self.parent is not None:
            # 展平参数列表，获取所有参数和关键字参数的叶子节点
            flat_args = pytree.arg_tree_leaves(*args, **kwargs)
            for arg in flat_args:
                if not isinstance(arg, torch.fx.Node):
                    continue
                # 确保创建的节点使用的图与当前 SubgraphTracer 的图相同
                assert (
                    arg.graph == self.graph
                ), "create_node using arg not from this SubgraphTracer"

        # 调用父类方法创建节点，并设置节点的创建时间戳为输出图的时间戳
        node = super().create_node(op, target, args, kwargs, name, type_expr)
        node.meta["creation_timestamp"] = self.output_graph.timestamp
        return node

    # 注意：我们没有重写 erase_node 方法，因为在其他地方调用了 self.graph.erase_node
    def remove_node(self, node):
        if len(node.users) > 0:
            user_graph_nodes: List[torch.fx.Node] = []
            for user in node.users.keys():
                # 对于 user.graph != self.graph 的情况，这是一个真正的 bug，并会引发异常
                if user.graph != self.graph:
                    # 这是一个嵌套图，需要被删除
                    # 如果不这样做，在尝试移除时将会引发异常
                    # 由于我们只会在恢复清理期间到达此处，这是合理的
                    user_graph_nodes.extend(reversed(list(user.graph.nodes)))
            for other_graph_node in user_graph_nodes:
                other_graph_node.graph.erase_node(other_graph_node)
        
        # 从图中删除节点，并从输入名称到代理的映射中移除节点的名称
        self.graph.erase_node(node)
        self.input_name_to_proxy.pop(node.name, None)

    # 当 before=True 时，我们会将此输入插入到最近插入的代理之前
    # 这是为了解决一个顺序问题的 hack，
    # 在这之前我们首先插入一个张量参数，然后插入可能出现在张量参数中的 SymInt 绑定
    # 如果 https://github.com/pytorch/pytorch/issues/99007 得到修复，可以移除这段代码
    # 创建图输入节点的方法，接受参数 name（节点名称）、type_expr（类型表达式，默认为None）、before（布尔值，指示是否在节点之前插入）、source（输入数据源）
    def create_graph_input(self, name, type_expr=None, before=False, source=None):
        # 打印调试信息，记录创建图输入节点的操作，包括节点名称和源名称（如果有的话）
        log.debug(
            "create_graph_input %s %s",
            name,
            source.name() if source is not None else "(none)",
        )
        # 如果没有提供数据源（source），则要求在根跟踪器上提供数据源
        if source is None:
            assert (
                self.parent is not None
            ), "you are required to provide a source for inputs on the root tracer"

        # 如果处于导出模式（export_root为True），则需要对输入源的来源进行检查和记录
        if self.export_root:
            # 如果来源不是本地源并且不允许使用单元格或自由变量，则记录源到用户堆栈
            if not is_from_local_source(source, allow_cell_or_freevar=False):
                self.output_graph.source_to_user_stacks.setdefault(source, []).append(
                    TracingContext.extract_stack()
                )

        # 确保节点名称是唯一的
        if name in self.input_name_to_proxy:
            for i in itertools.count():
                candidate_name = f"{name}_{i}"
                if candidate_name not in self.input_name_to_proxy:
                    name = candidate_name
                    break

        # 确定插入位置
        if self.input_name_to_proxy:
            prev_name = next(reversed(self.input_name_to_proxy))
            node = self.input_name_to_proxy[prev_name].node
            # 根据 before 参数决定是在节点之前还是之后插入
            if before:
                ctx = self.graph.inserting_before(node)
            else:
                ctx = self.graph.inserting_after(node)
        else:
            # 如果没有已有节点，插入到开头
            ctx = self.graph.inserting_before(None)
        
        # 在上下文 ctx 中创建代理节点
        with ctx:
            proxy = self.create_proxy("placeholder", name, (), {}, type_expr=type_expr)
            # 如果之前已有输入节点并且是在节点之前插入，调整顺序以保证正确性
            if self.input_name_to_proxy and before:
                k, v = self.input_name_to_proxy.popitem()
                self.input_name_to_proxy[name] = proxy
                self.input_name_to_proxy[k] = v
            else:
                self.input_name_to_proxy[name] = proxy
            return proxy

    # 查看 NOTE: [Nested SubgraphTracer and free_variable handling] 获取更多详情
    def lift_tracked_freevar_to_input(self, proxy):
        # 如果我们是根 SubgraphTracer，那么肯定出错了，因为 Dynamo 在创建代理之前就将张量添加到图输入中了。
        assert (
            self.parent is not None
        ), "lift_tracked_freevar_to_input 不应该在根 SubgraphTracer 上调用"
        
        # Proxies 与 VariableTracker 相关联。
        # 可能已经将代理升级为输入了。
        # 如果是这种情况，直接返回已经升级的代理。
        if proxy in self.lifted_freevars:
            return self.lifted_freevars[proxy]
        
        # 创建一个新的代理作为图的输入
        new_proxy = self.create_graph_input(proxy.node.name)
        
        # 设置新代理节点的示例值
        set_example_value(new_proxy.node, proxy.node.meta["example_value"])
        
        # 将新代理加入到 lifted_freevars 字典中
        self.lifted_freevars[proxy] = new_proxy
        
        # 如果父对象存在且代理的追踪器不是当前对象，则在父对象上再次调用 lift_tracked_freevar_to_input
        if self.parent is not None and proxy.tracer != self.parent:
            self.parent.lift_tracked_freevar_to_input(proxy)
        
        # 返回新创建的代理
        return new_proxy

    def maybe_lift_tracked_freevar_to_input(self, arg):
        """
        如果 arg 是自由变量，则将其升级为输入。
        返回升级后的新 arg（如果 arg 是自由变量），否则返回原始 arg。
        """
        # 如果 arg 不是 torch.fx.Proxy 类型，则直接返回原始 arg
        if not isinstance(arg, torch.fx.Proxy):
            return arg
        
        # 如果 arg 的追踪器是当前对象，则返回原始 arg
        elif arg.tracer == self:
            return arg
        
        # 调用 lift_tracked_freevar_to_input 将自由变量代理升级为输入
        return self.lift_tracked_freevar_to_input(arg)
# NOTE: [HigherOrderOperator tracing design]
# Ignoring HigherOrderOperators for a moment,
# OutputGraph represents the graph being built by Dynamo that may be compiled
# and executed. It holds a root SubgraphTracer where the FX graph is built.
#
# HigherOrderOperators are operators that take functions as their arguments.
# When Dynamo encounters a HigherOrderOperator, then it attempts to introspect
# the function passed to it (call this the "body function"), capture it into a
# GraphModule, and rewrite the call to the HigherOrderOperator to use the
# GraphModule.
#
# The way we handle the capture of body functions is through having
# (possibly nested) SubgraphTracers, one per body function.
#
# Mechanically, we do the introspection by:
# - Creating a new SubgraphTracer via OutputGraph.subtracer
# - Executing the body function.
# This constructs the graph of the body function in the new SubgraphTracer
# while modifying the state of the OutputGraph. For example:
# - the OutputGraph can receive new GraphArgs (if we discover any new
#   untracked Tensors)
# - side effects from the body function get accumulated into
#   OutputGraph.side_effects
# - guards produced by the body function get accumulated into OutputGraph.guards
#
# The traced function has some special properties that make it easier for us
# to transform later down the line:
# - we lift all free variables to being inputs.
#
# If the introspection fails (due to the existence of graph breaks), then
# we roll back the current OutputGraph state and graph break on the
# HigherOrderOperator.
```