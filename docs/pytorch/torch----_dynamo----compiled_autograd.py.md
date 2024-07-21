# `.\pytorch\torch\_dynamo\compiled_autograd.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类型声明
import contextlib                          # 上下文管理器的支持库
import functools                           # 函数装饰器和高阶函数支持库
from typing import Dict, List, Optional, TYPE_CHECKING  # 引入类型声明

import torch                               # PyTorch 深度学习框架
from torch._dynamo.external_utils import (
    call_backward,                        # 导入反向传播相关工具函数
    call_hook,                            # 导入调用钩子函数
    FakeCompiledAutogradEngine,            # 导入虚拟编译自动求导引擎
)
from torch._dynamo.source import GetItemSource, LocalSource  # 导入源码追踪相关工具
from torch._dynamo.utils import counters, lazy_format_graph_code, set_locals_to_steal  # 导入计数器、懒格式化图代码、设置本地变量的工具函数
from torch._logging import getArtifactLogger, trace_structured  # 导入日志记录和结构化追踪相关函数
from torch._prims_common import clone_preserve_strides  # 导入克隆保留步幅的函数
from torch._subclasses import FakeTensorMode  # 导入虚拟张量模式
from torch.fx import GraphModule           # 导入图模块
from torch.fx.experimental._backward_state import BackwardState  # 导入反向传播状态
from torch.fx.experimental.proxy_tensor import (
    decompose,                            # 导入解构函数
    disable_autocast_cache,               # 导入自动类型转换缓存禁用函数
    disable_proxy_modes_tracing,          # 导入跟踪代理模式禁用函数
    fetch_object_proxy,                   # 导入获取对象代理函数
    ProxyTorchDispatchMode,               # 导入代理调度模式
    PythonKeyTracer,                      # 导入 Python 键追踪器
    track_tensor_tree,                    # 导入跟踪张量树函数
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv  # 导入符号形状相关工具
from torch.fx.traceback import preserve_node_meta, set_stack_trace  # 导入节点元信息保留和设置堆栈追踪函数
from torch.utils._traceback import CapturedTraceback  # 导入捕获的堆栈追踪

if TYPE_CHECKING:
    from torch.fx.proxy import Proxy      # 类型检查导入代理类型声明

# 获取编译自动求导日志记录器和详细日志记录器
compiled_autograd_log = getArtifactLogger(__name__, "compiled_autograd")
verbose_log = getArtifactLogger(__name__, "compiled_autograd_verbose")


def snapshot_verbose_logging_enabled():
    # 检查是否启用了详细日志记录
    return torch._logging._internal.log_state.is_artifact_enabled(
        "compiled_autograd_verbose"
    )


def cpp_verbose_log_fn(msg: str) -> None:
    # 输出详细日志的调试信息
    verbose_log.debug(msg)


def snapshot_cudagraph_enabled():
    # 检查是否启用了 CUDA 图形
    return torch._inductor.config.triton.cudagraphs


def maybe_clone(x):
    # 克隆张量 x（如果不为 None）
    if x is not None:
        return clone_preserve_strides(x)
    return x


class AutogradCompilerInstance:
    def __init__(self, compiler_fn) -> None:
        # 初始化自动求导编译器实例
        self.compiler_fn = compiler_fn                    # 设置编译函数
        self.stack = contextlib.ExitStack()               # 创建上下文管理栈
        self.close = self.stack.close                     # 设置关闭方法
        self.shape_env = ShapeEnv()                       # 创建形状环境
        self.fake_tensor_mode = FakeTensorMode(           # 创建虚拟张量模式
            allow_fallback_kernels=True,
            allow_non_fake_inputs=True,
            shape_env=self.shape_env,
        )
        self.fx_tracer = PythonKeyTracer()                # 创建 Python 键追踪器
        self.proxy_mode = ProxyTorchDispatchMode(         # 创建代理 Torch 调度模式
            self.fx_tracer, "symbolic"
        )
        self.hooks_proxy: Optional[Proxy] = None          # 钩子代理初始化为 None

    def wrap_fake(self, x, source):
        # 包装张量 x 为虚拟张量，并指定源
        assert isinstance(x, torch.Tensor)
        return self.fake_tensor_mode.from_tensor(x, source=source)

    @staticmethod
    def source(name, idx) -> GetItemSource:
        # 静态方法：创建获取项源码的实例
        return GetItemSource(LocalSource(name), idx)
    # 增加计数器，记录调用次数
    counters["compiled_autograd"]["captures"] += 1

    # 创建一个空的根模块作为函数追踪器的根
    self.fx_tracer.root = torch.nn.Module()

    # 创建一个新的函数追踪图，使用PythonKeyTracer作为追踪器类
    self.fx_tracer.graph = torch.fx.Graph(tracer_cls=PythonKeyTracer)

    # 初始化一个空的 tensor_attrs 字典，用于存储张量属性
    self.fx_tracer.tensor_attrs = {}

    # 创建代理对象，用于表示输入参数 inputs
    args_proxy = self.fx_tracer.create_proxy("placeholder", "inputs", (), {})

    # 创建代理对象，用于表示输入参数 sizes
    sizes_proxy = self.fx_tracer.create_proxy("placeholder", "sizes", (), {})

    # 创建代理对象，用于表示 hooks，初始化为占位符
    self.hooks_proxy = self.fx_tracer.create_proxy("placeholder", "hooks", (), {})

    # 将真实的输入张量 inputs 转换为虚拟张量
    inputs = [
        self.wrap_fake(x, self.source("inputs", idx))
        for idx, x in enumerate(inputs)
    ]
    proxies = [args_proxy[i] for i in range(len(inputs))]
    self.bind_tensors_to_proxies(inputs, proxies)

    # 将输入的 sizes 转换为符号整数（symint）
    sizes = [
        self.shape_env.create_unspecified_symint_and_symbol(
            val,
            self.source("sizes", idx),
            DimDynamic.DYNAMIC,
        )
        for idx, val in enumerate(sizes)
    ]
    self.bind_tensors_to_proxies(sizes, sizes_proxy)

    # 进入以下上下文环境，控制特定模式
    self.stack.enter_context(decompose({}))  # 分解模式
    self.stack.enter_context(self.fake_tensor_mode)  # 虚拟张量模式
    self.stack.enter_context(self.proxy_mode.sym_mode)  # 代理模式中的符号模式
    self.stack.enter_context(self.proxy_mode)  # 一般代理模式
    self.stack.enter_context(disable_autocast_cache())  # 禁用自动转换缓存
    self.stack.enter_context(preserve_node_meta())  # 保留节点元数据

    # 返回处理后的 inputs 和 sizes
    return inputs, sizes


def proxy_call_backward(
    self,
    inputs,
    output_metadatas,
    saved_tensors,
    backward_idx: int,
):
    # 断言确保 hooks_proxy 不为空
    assert self.hooks_proxy is not None

    # 从 hooks_proxy 中获取反向传播函数的代理对象
    backward_c_function = self.hooks_proxy[backward_idx]  # type: ignore[index]

    # 创建一个代理对象，调用反向传播函数 call_backward
    proxies = self.fx_tracer.create_proxy(
        kind="call_function",
        target=call_backward,
        args=(
            backward_c_function,
            self.to_proxy(saved_tensors),
            *self.to_proxy(inputs),
        ),
        kwargs={},
    )

    # 使用 disable_proxy_modes_tracing 上下文环境
    with disable_proxy_modes_tracing():
        # 创建虚拟张量，用于梯度输入 grad_ins
        grad_ins: List[Optional[torch.Tensor]] = []
        for output_metadata in output_metadatas:
            if output_metadata is None:
                grad_ins.append(None)
                continue

            layout, device, dtype, size = output_metadata
            grad_ins.append(
                torch.empty(size=size, dtype=dtype, layout=layout, device=device)
            )

        # 将创建的虚拟张量绑定到代理对象 proxies 中
        self.bind_tensors_to_proxies(grad_ins, proxies)

    # 返回梯度输入 grad_ins 的元组
    return tuple(grad_ins)
    # 创建一个代理调用钩子的方法
    def proxy_call_hook(self, hook, *args):
        return self.fx_tracer.create_proxy(
            "call_function",  # 创建一个“call_function”类型的代理
            call_hook,  # 调用名为call_hook的函数
            (
                hook,  # 要调用的钩子对象
                *[self.to_proxy(x) for x in args],  # 将输入参数转换为代理对象
            ),
            {},  # 空字典作为额外参数传递给钩子
        )

    # Tensor预处理钩子
    def tensor_pre_hook(self, inputs, hook_id, i: int):
        assert self.hooks_proxy is not None  # 断言钩子代理不为空
        hook = self.hooks_proxy[hook_id]  # type: ignore[index] 获取特定id的钩子代理对象
        proxy = self.proxy_call_hook(
            hook,  # 调用的钩子代理对象
            inputs[i],  # 输入张量的第i个元素
        )
        with disable_proxy_modes_tracing():  # 禁用代理模式跟踪
            inputs[i] = maybe_clone(inputs[i])  # 可能克隆输入张量的第i个元素
            self.bind_tensors_to_proxies([inputs[i]], [proxy])  # 将输入张量和代理对象进行绑定
        return inputs  # 返回处理后的输入张量列表

    # 预处理钩子
    def pre_hook(self, inputs, hook_id):
        assert self.hooks_proxy is not None  # 断言钩子代理不为空
        hook = self.hooks_proxy[hook_id]  # type: ignore[index] 获取特定id的钩子代理对象
        proxies = self.proxy_call_hook(
            hook,  # 调用的钩子代理对象
            inputs,  # 输入张量列表
        )
        with disable_proxy_modes_tracing():  # 禁用代理模式跟踪
            inputs = [maybe_clone(x) for x in inputs]  # 可能克隆输入张量列表中的每个张量
            self.bind_tensors_to_proxies(inputs, proxies)  # 将输入张量和代理对象进行绑定
        return inputs  # 返回处理后的输入张量列表

    # 后处理钩子
    def post_hook(self, outputs, inputs, hook_id):
        assert self.hooks_proxy is not None  # 断言钩子代理不为空
        hook = self.hooks_proxy[hook_id]  # type: ignore[index] 获取特定id的钩子代理对象
        proxies = self.proxy_call_hook(
            hook,  # 调用的钩子代理对象
            outputs,  # 输出张量列表
            inputs,  # 输入张量列表
        )
        with disable_proxy_modes_tracing():  # 禁用代理模式跟踪
            outputs = [maybe_clone(x) for x in outputs]  # 可能克隆输出张量列表中的每个张量
            self.bind_tensors_to_proxies(outputs, proxies)  # 将输出张量和代理对象进行绑定
        return outputs  # 返回处理后的输出张量列表

    # 后处理梯度累积钩子
    def post_acc_grad_hook(self, input, hook_id):
        assert isinstance(input, torch.Tensor)  # 断言输入是torch.Tensor类型
        assert self.hooks_proxy is not None  # 断言钩子代理不为空
        hook = self.hooks_proxy[hook_id]  # type: ignore[index] 获取特定id的钩子代理对象
        proxies = self.proxy_call_hook(
            hook,  # 调用的钩子代理对象
            input,  # 输入张量
        )
        with disable_proxy_modes_tracing():  # 禁用代理模式跟踪
            input = [maybe_clone(input)]  # 可能克隆输入张量
            self.bind_tensors_to_proxies(input, proxies)  # 将输入张量和代理对象进行绑定
        return input  # 返回处理后的输入张量列表

    # 注释:
    # 编译的自动求导和cudagraphs
    # 急切的自动求导将标量实现为0维张量，见DivBackward0::other_。
    # 当编译的自动求导跟踪这些节点时，它会提升标量张量，导致图中存在一些cpu 0维张量输入。
    # 为了防止整个图跳过cudagraph，我们将标量张量移动到cuda上。
    # 这是因为ATen/prims操作也会接受cuda 0维张量。
    # 将图中的节点移动到 CUDA 设备上，返回需要移动的节点在运行时的索引列表
    def move_graph_nodes_to_cuda(self, graph) -> List[int]:
        # 用于存储需要移动的节点的字典，键为索引，值为节点对象
        to_move: Dict[int, torch.fx.Node] = {}
        # 是否存在 CUDA 输入标记
        has_cuda_inputs = False
        # 获取图中所有节点的列表
        nodes = list(graph.nodes)
        # 断言第一个节点的目标为 "inputs"
        assert nodes[0].target == "inputs"
        # 提取输入节点
        inputs = nodes[0]
        # 获取直接依赖于输入节点的节点列表
        inputs_users = list(inputs.users.keys())
        
        # 确认节点的顺序应为 [inputs, sizes, hooks, getitem, getitem1, ...]
        # 其中 getitemi 访问 inputs[i]
        first_getitem_idx = 3
        assert nodes[first_getitem_idx] == inputs_users[0]
        last_getitem_idx = first_getitem_idx + len(inputs_users) - 1
        assert nodes[last_getitem_idx] == inputs_users[-1]
        
        # 遍历直接依赖于输入节点的节点列表
        for i, node in enumerate(inputs_users):
            # 如果尚未发现 CUDA 输入，并且当前节点值的设备类型为 "cuda"，则设置标记为 True
            if not has_cuda_inputs and node.meta["val"].device.type == "cuda":
                has_cuda_inputs = True
                continue
            
            # 检查当前节点值是否为 CPU 设备且是标量
            is_cpu = node.meta["val"].device.type == "cpu"
            is_scalar = len(node.meta["val"].size()) == 0
            
            # 如果节点是 CPU 设备且是标量
            if is_cpu and is_scalar:
                # 获取当前节点的用户列表
                node_users = list(node.users.keys())
                # 如果所有用户均为 torch._ops.OpOverload 类型，并且命名空间为 ("prims", "aten")
                if all(
                    isinstance(user.target, torch._ops.OpOverload)
                    and user.target.namespace in ("prims", "aten")
                    for user in node_users
                ):
                    # 所有用户都是 prims/aten 操作，可以安全地移动此节点
                    to_move[i] = node

        # 如果图中存在 CUDA 输入，则将需要移动的节点的值移动到 CUDA 设备上
        if has_cuda_inputs:
            for node in to_move.values():
                node.meta["val"] = node.meta["val"].cuda()

            # 返回需要移动的节点在运行时的索引列表
            return list(to_move.keys())

        # 如果图中不存在 CUDA 输入，则返回空列表
        return []
    def end_capture(self, outputs):
        # 创建一个代理函数调用节点，用于最终回调函数
        self.fx_tracer.create_proxy(
            "call_function",
            FakeCompiledAutogradEngine._exec_final_callbacks_stub,
            (),
            {},
        )
        # 关闭堆栈
        self.stack.close()
        # 创建一个输出节点，并记录到跟踪器中
        self.fx_tracer.create_node(
            "output",
            "output",
            (self.fx_tracer.create_arg(self.to_proxy(outputs)),),
            {},
        )
        # 重新排序累积梯度节点，以模拟急切模式的行为
        self.reorder_accumulate_grad_nodes()
        # 如果启用了快照 cudagraph，则移动运行时输入到 CUDA
        runtime_inputs_to_move: List[int] = []
        if snapshot_cudagraph_enabled():
            runtime_inputs_to_move = self.move_graph_nodes_to_cuda(self.fx_tracer.graph)

        # 创建一个 GraphModule 对象来表示编译后的自动求导图
        graph = GraphModule(
            self.fx_tracer.root, self.fx_tracer.graph, "CompiledAutograd"
        )
        # 设置本地变量以窃取
        set_locals_to_steal(graph, ["inputs"])
        # 记录编译后自动求导图的信息到日志
        compiled_autograd_log.info(
            "%s", lazy_format_graph_code("Compiled autograd graph", graph, colored=True)
        )
        # 调试输出编译后自动求导图的信息，包括设备信息
        verbose_log.debug(
            "%s",
            lazy_format_graph_code(
                "Compiled autograd graph", graph, include_device=True, colored=True
            ),
        )
        # 跟踪结构化信息，打印可读的编译后自动求导图
        trace_structured(
            "compiled_autograd_graph",
            payload_fn=lambda: graph.print_readable(print_output=False),
        )

        # 运行时包装器函数，用于执行编译后的函数
        def runtime_wrapper(compiled_fn, inputs, sizes, hooks):
            global in_compiled_autograd_region
            try:
                # 进入编译后自动求导区域
                in_compiled_autograd_region = True
                # 将需要移动的运行时输入置于 CUDA 设备上
                for i in runtime_inputs_to_move:
                    inputs[i] = inputs[i].pin_memory().cuda(non_blocking=True)

                # 执行编译后的函数
                return compiled_fn(inputs, sizes, hooks)
            finally:
                # 离开编译后自动求导区域
                in_compiled_autograd_region = False

        # 返回运行时包装器函数和编译器函数处理后的图
        return runtime_wrapper, self.compiler_fn(graph)

    def reorder_accumulate_grad_nodes(self):
        """
        AOTAutograd 使用时会导致所有 accumulate_grad_ 节点被推送到图的末尾。
        这与急切模式不同，后者会尽早调度这些节点。此函数尝试重新排序图以模拟急切模式的行为。
        """
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=torch.ops.inductor.accumulate_grad_.default
        ):
            # 找到最后一个参数，并将该节点追加到参数的末尾
            arg = max(node.args)  # 最后一个参数
            if arg is not node.prev and arg.op != "placeholder":
                arg.append(node)

    def to_proxy(self, t):
        if t is None:
            return None
        if isinstance(t, list):
            return [self.to_proxy(x) for x in t]
        if isinstance(t, tuple):
            return tuple(self.to_proxy(x) for x in t)
        assert isinstance(t, (torch.Tensor, torch.SymInt))
        # 返回 t 的代理对象
        return fetch_object_proxy(self.fx_tracer)(t).proxy
    # 将张量绑定到代理对象上
    def bind_tensors_to_proxies(self, tensors, proxies):
        # 如果 proxies 是单个 Proxy 对象，则转换为列表，每个张量对应一个代理对象
        if isinstance(proxies, torch.fx.Proxy):
            proxies = [proxies[i] for i in range(len(tensors))]
        # 断言张量列表和代理列表长度相同
        assert len(tensors) == len(proxies)
        # 调用函数跟踪器，将张量和代理对象绑定起来
        track_tensor_tree(tensors, proxies, constant=None, tracer=self.fx_tracer)

    # 绑定反向状态
    def bind_backward_state(self, index: int):
        # 断言钩子代理对象不为空
        assert self.hooks_proxy is not None
        # 获取指定索引处的钩子代理对象
        proxy = self.hooks_proxy[index]  # type: ignore[index]
        # 创建一个反向状态对象
        bw_state = BackwardState()
        # 调用函数跟踪器，将反向状态对象和钩子代理对象绑定起来
        track_tensor_tree(bw_state, proxy, constant=None, tracer=self.fx_tracer)
        # 返回创建的反向状态对象
        return bw_state

    # 设置节点的源信息
    def set_node_origin(self, node_name, node_index):
        # 提取当前捕获的堆栈跟踪信息
        raw_stack_trace = CapturedTraceback.extract().format()[-1]
        # 构建新的代码字符串，表示节点名和节点索引
        new_code = f"{node_name} (NodeCall {node_index})"
        # 替换原始堆栈跟踪中的代码行，更新为新的节点信息
        new_stack_trace = raw_stack_trace.replace(
            "raw_stack_trace = CapturedTraceback.extract().format()[-1]", new_code
        )
        # 设置更新后的堆栈跟踪信息
        set_stack_trace(new_stack_trace)
# 设置编译自动求导引擎的状态为未启用
compiled_autograd_enabled = False

# 全局标志，用于检查是否正在处理编译自动求导图产生的图形
in_compiled_autograd_region = False


@contextlib.contextmanager
def enable(compiler_fn):
    # 保存当前的自动求导编译器状态，并设置新的编译器函数
    prior = torch._C._dynamo.compiled_autograd.set_autograd_compiler(
        functools.partial(AutogradCompilerInstance, compiler_fn)
    )
    # 如果启用了详细快照日志记录，则设置详细日志记录函数
    if snapshot_verbose_logging_enabled():
        torch._C._dynamo.compiled_autograd.set_verbose_logger(cpp_verbose_log_fn)
    # 设置全局编译自动求导引擎状态为启用
    global compiled_autograd_enabled
    compiled_autograd_enabled = True
    try:
        # 禁用多线程执行
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        # 恢复先前的自动求导编译器状态
        if not prior:
            compiled_autograd_enabled = False
        torch._C._dynamo.compiled_autograd.set_autograd_compiler(prior)


@contextlib.contextmanager
def disable():
    # 保存当前的自动求导编译器状态，并设置为None以禁用编译器
    prior = torch._C._dynamo.compiled_autograd.set_autograd_compiler(None)
    # 设置全局编译自动求导引擎状态为禁用
    global compiled_autograd_enabled
    compiled_autograd_enabled = False
    try:
        yield
    finally:
        # 如果先前的编译器状态存在，则设置全局编译自动求导引擎状态为启用
        if prior:
            compiled_autograd_enabled = True
        torch._C._dynamo.compiled_autograd.set_autograd_compiler(prior)


# 返回到新进程的起始状态
def reset() -> None:
    # 设置编译自动求导引擎状态为未启用
    compiled_autograd_enable = False
    # 断言不在编译自动求导区域中
    assert not in_compiled_autograd_region
    # 设置自动求导编译器和详细日志记录函数为None
    torch._C._dynamo.compiled_autograd.set_autograd_compiler(None)
    torch._C._dynamo.compiled_autograd.set_verbose_logger(None)
```