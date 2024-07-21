# `.\pytorch\torch\cuda\graphs.py`

```
# mypy: allow-untyped-defs
# 导入垃圾回收模块
import gc
# 导入类型提示模块
import typing

# 导入PyTorch模块
import torch

# 导入自定义的_dummy_type函数
from .._utils import _dummy_type

# 如果torch._C模块没有_CudaStreamBase属性
if not hasattr(torch._C, "_CudaStreamBase"):
    # 定义虚拟的基类
    torch._C.__dict__["_CUDAGraph"] = _dummy_type("_CUDAGraph")
    torch._C.__dict__["_graph_pool_handle"] = _dummy_type("_graph_pool_handle")
    torch._C.__dict__["_cuda_isCurrentStreamCapturing"] = _dummy_type(
        "_cuda_isCurrentStreamCapturing"
    )

# 从torch._C模块中导入以下内容
from torch._C import (  # noqa: F401
    _cuda_isCurrentStreamCapturing,
    _CUDAGraph,
    _graph_pool_handle,
)

# 定义函数is_current_stream_capturing
def is_current_stream_capturing():
    r"""Return True if CUDA graph capture is underway on the current CUDA stream, False otherwise.

    If a CUDA context does not exist on the current device, returns False without initializing the context.
    """
    # 调用C++扩展函数_cuda_isCurrentStreamCapturing()来获取当前CUDA流是否正在进行图形捕获
    return _cuda_isCurrentStreamCapturing()


# Python shim帮助Sphinx更可靠地处理文档字符串。
# 定义函数graph_pool_handle
def graph_pool_handle():
    r"""Return an opaque token representing the id of a graph memory pool.

    See :ref:`Graph memory management<graph-memory-management>`.

    .. warning::
        This API is in beta and may change in future releases.
    """
    # 返回一个表示图形内存池ID的不透明令牌
    return _graph_pool_handle()


# Python shim帮助Sphinx更可靠地处理文档字符串。
# 定义类CUDAGraph，继承自torch._C._CUDAGraph
class CUDAGraph(torch._C._CUDAGraph):
    r"""Wrapper around a CUDA graph.

    .. warning::
        This API is in beta and may change in future releases.
    """

    def __new__(cls):
        return super().__new__(cls)

    # 定义实例方法capture_begin
    def capture_begin(self, pool=None, capture_error_mode="global"):
        r"""Begin capturing CUDA work on the current stream.

        Typically, you shouldn't call ``capture_begin`` yourself.
        Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables`,
        which call ``capture_begin`` internally.

        Arguments:
            pool (optional): Token (returned by :func:`~torch.cuda.graph_pool_handle` or
                :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) that hints this graph may share memory
                with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.
            capture_error_mode (str, optional): specifies the cudaStreamCaptureMode for the graph capture stream.
                Can be "global", "thread_local" or "relaxed". During cuda graph capture, some actions, such as cudaMalloc,
                may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for
                actions in the current thread, and "relaxed" will not error on these actions. Do NOT change this setting
                unless you're familiar with `cudaStreamCaptureMode <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85>`_
        """  # noqa: B950
        # 调用父类的capture_begin方法，开始在当前流上捕获CUDA工作
        super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)
    def capture_end(self):
        r"""End CUDA graph capture on the current stream.

        After ``capture_end``, ``replay`` may be called on this instance.

        Typically, you shouldn't call ``capture_end`` yourself.
        Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables`,
        which call ``capture_end`` internally.
        """
        # 调用父类方法，结束当前流上的 CUDA 图捕获
        super().capture_end()

    def replay(self):
        r"""Replay the CUDA work captured by this graph."""
        # 调用父类方法，回放由该图捕获的 CUDA 工作
        super().replay()

    def reset(self):
        r"""Delete the graph currently held by this instance."""
        # 调用父类方法，删除当前实例持有的图
        super().reset()

    def pool(self):
        r"""Return an opaque token representing the id of this graph's memory pool.

        This id can optionally be passed to another graph's ``capture_begin``,
        which hints the other graph may share the same memory pool.
        """
        # 返回代表此图内存池 ID 的不透明令牌
        # 可以选择将此 ID 传递给另一个图的 ``capture_begin``，暗示另一个图可能共享相同的内存池
        return super().pool()

    def enable_debug_mode(self):
        r"""Enable debugging mode for CUDAGraph.debug_dump."""
        # 启用 CUDAGraph.debug_dump 的调试模式
        return super().enable_debug_mode()

    def debug_dump(self, debug_path):
        r"""
        Arguments:
            debug_path (required): Path to dump the graph to.

        Calls a debugging function to dump the graph if the debugging is
        enabled via CUDAGraph.enable_debug_mode()
        """
        # 调用一个调试函数来将图形转储到指定路径
        # 前提是通过 CUDAGraph.enable_debug_mode() 启用了调试
        return super().debug_dump(debug_path)
class graph:
    r"""Context-manager that captures CUDA work into a :class:`torch.cuda.CUDAGraph` object for later replay.

    See :ref:`CUDA Graphs <cuda-graph-semantics>` for a general introduction,
    detailed use, and constraints.

    Arguments:
        cuda_graph (torch.cuda.CUDAGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch.cuda.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) hinting this graph's capture
            may share memory from the specified pool. See :ref:`Graph memory management<graph-memory-management>`.
        stream (torch.cuda.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.
        capture_error_mode (str, optional): specifies the cudaStreamCaptureMode for the graph capture stream.
            Can be "global", "thread_local" or "relaxed". During cuda graph capture, some actions, such as cudaMalloc,
            may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for
            actions in the current thread, and "relaxed" will not error on actions. Do NOT change this setting
            unless you're familiar with `cudaStreamCaptureMode <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85>`_

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    .. warning::
        This API is in beta and may change in future releases.

    .. _cudaStreamCaptureMode:
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
    """  # noqa: B950

    default_capture_stream: typing.Optional["torch.cuda.Stream"] = None

    def __init__(
        self,
        cuda_graph,
        pool=None,
        stream=None,
        capture_error_mode: str = "global",
    ):
        # 初始化方法，创建一个新的 CUDA 图形对象捕获上下文管理器
        self.cuda_graph = cuda_graph
        # 可选参数，用于内存共享，指示此图的捕获可能共享来自指定池的内存
        self.pool = pool
        # 如果提供了 stream 参数，将其设置为当前上下文中的流；如果未提供，则使用自己的内部侧流
        self.stream = stream
        # 指定图形捕获流的 cudaStreamCaptureMode
        # 可选值有 "global", "thread_local" 或 "relaxed"，不要更改此设置，除非熟悉 cudaStreamCaptureMode
        self.capture_error_mode = capture_error_mode
    ):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        # 如果默认捕获流未初始化，则进行延迟初始化，以避免循环导入错误。
        # 虽然不是线程安全的，但图表已经明确文档化了一次只能在进程中进行一次捕获的一般限制。
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.cuda.Stream()

        self.pool = () if pool is None else (pool,)
        # 如果未提供stream参数，则使用默认的捕获流，否则使用给定的stream。
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        assert self.capture_stream is not None
        # 将当前CUDA流设置为捕获流的上下文管理器
        self.stream_ctx = torch.cuda.stream(self.capture_stream)
        self.cuda_graph = cuda_graph
        self.capture_error_mode = capture_error_mode

    def __enter__(self):
        # Free as much memory as we can for the graph
        # 同步所有CUDA设备上的操作，确保操作完成
        torch.cuda.synchronize()
        # 手动触发垃圾回收
        gc.collect()
        # 清空CUDA缓存，尽可能释放内存
        torch.cuda.empty_cache()

        # 使用流的上下文管理器进入该流，开始捕获CUDA图
        self.stream_ctx.__enter__()

        # 开始捕获CUDA图，传递捕获错误模式和池参数
        self.cuda_graph.capture_begin(
            *self.pool, capture_error_mode=self.capture_error_mode
        )

    def __exit__(self, exc_type, exc_value, traceback):
        # 结束CUDA图的捕获过程
        self.cuda_graph.capture_end()
        # 退出流的上下文管理器
        self.stream_ctx.__exit__(exc_type, exc_value, traceback)
        # 返回None以传播从capture_end或stream_ctx.__exit__()引发的异常
def make_graphed_callables(
    callables, sample_args, num_warmup_iters=3, allow_unused_input=False, pool=None
):
    r"""Accept callables (functions or :class:`nn.Module<torch.nn.Module>`\ s) and returns graphed versions.

    Each graphed callable's forward pass runs its source callable's
    forward CUDA work as a CUDA graph inside a single autograd node.

    The graphed callable's forward pass also appends
    a backward node to the autograd graph. During backward, this node runs the
    callable's backward work as a CUDA graph.

    Therefore, each graphed callable should be a drop-in replacement for its source callable
    in an autograd-enabled training loop.

    See :ref:`Partial-network capture<partial-network-capture>` for detailed use and constraints.

    If you pass a tuple of several callables, their captures will use the same memory pool.
    See :ref:`Graph memory management<graph-memory-management>` for when this is appropriate.

    Arguments:
        callables (torch.nn.Module or Python function, or tuple of these): Callable or callables to graph.
            See :ref:`Graph memory management<graph-memory-management>` for when passing a tuple of callables
            is appropriate.  If you pass a tuple of callables, their order in the tuple must be the same order
            they'll run in the live workload.
        sample_args (tuple of Tensors, or tuple of tuples of Tensors): Samples args for each callable.
            If a single callable was passed, ``sample_args`` must be a single tuple of argument Tensors.
            If a tuple of callables was passed, ``sample_args`` must be tuple of tuples of argument Tensors.
        num_warmup_iters (int): The number of warmup iterations. Currently, ``DataDistributedParallel`` needs
            11 iterations for warm up. Default: ``3``.
        allow_unused_input (bool): If False, specifying inputs that were not used when computing outputs
            (and therefore their grad is always zero) is an error. Defaults to False.
        pool (optional): Token (returned by :func:`~torch.cuda.graph_pool_handle` or
            :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) that hints this graph may share memory
            with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.

    .. note::
        The ``requires_grad`` state of each Tensor in ``sample_args`` must match the state
        that's expected for the corresponding real input in the training loop.

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        ``sample_args`` for each callable must contain only Tensors. Other types are not allowed.

    .. warning::
        Returned callables do not support higher order differentiation (e.g., double backward).
"""
    # 检查是否启用了自动混合精度（automatic mixed precision），且是否启用了自动混合精度的缓存
    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        # 如果启用了自动混合精度且启用了缓存，则抛出运行时错误
        raise RuntimeError(
            "make_graphed_callables does not support the autocast caching. Please set `cache_enabled=False`."
        )

    # 初始化一个布尔值，用于标记是否只传入了单个的 callable
    just_one_callable = False

    # 如果传入的 callables 不是元组，则认为只传入了一个 callable，将其转为包含单个元素的元组，并相应调整 sample_args
    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)

    # 将 sample_args 扁平化处理
    flatten_sample_args = []

    # 遍历每个 callable 和其对应的 sample_args
    for c, args in zip(callables, sample_args):
        # 如果当前 callable 是 torch.nn.Module 类型
        if isinstance(c, torch.nn.Module):
            # 断言当前模块没有注册任何 hooks
            assert (
                len(c._backward_hooks) == 0
                and len(c._forward_hooks) == 0
                and len(c._forward_pre_hooks) == 0
            ), (
                "Modules must not have hooks registered at the time they are passed. However, registering hooks "
                + "on modules after passing them through make_graphed_callables is allowed."
            )
            # 断言所有的 buffer 在传递给 make_graphed_callables 时都设置了 requires_grad=False
            assert all(b.requires_grad is False for b in c.buffers()), (
                "In any :class:`~torch.nn.Module` passed to "
                + ":func:`~make_graphed_callables`, only parameters may be trainable. All buffers must have "
                + "``requires_grad=False``."
            )

        # 将参数扁平化并添加到 flatten_sample_args 中
        flatten_arg = torch.utils._pytree.arg_tree_leaves(*args)
        flatten_sample_args.append(tuple(flatten_arg))

        # 断言扁平化后的参数只包含 torch.Tensor 类型，因为在 beta API 中，sample_args 中只能包含张量类型
        assert all(isinstance(arg, torch.Tensor) for arg in flatten_arg), (
            "In the beta API, sample_args "
            + "for each callable must contain only Tensors. Other types are not allowed."
        )

    # 每个 callable 的期望用户参数长度
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    # 为每个可调用对象生成参数元组，如果是 torch.nn.Module 类型，则获取其参数
    per_callable_module_params = [
        tuple(c.parameters()) if isinstance(c, torch.nn.Module) else ()
        for c in callables
    ]
    
    # 生成每个可调用对象的静态输入表面，结合扁平化的样本参数和模块参数
    per_callable_static_input_surfaces = [
        flatten_sample_args[i] + per_callable_module_params[i]
        for i in range(len(callables))
    ]

    # 创建与可调用对象数量相同的前向 CUDA 图和后向 CUDA 图
    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]

    # 如果未提供内存池，则创建新的图池句柄
    mempool = graph_pool_handle() if pool is None else pool

    # 预热阶段，确保 CUDA 核心的性能基准和其他懒初始化 CUDA 工作不会影响捕获过程
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        for func, args, static_input_surface in zip(
            callables, sample_args, per_callable_static_input_surfaces
        ):
            for _ in range(num_warmup_iters):
                # 执行函数并捕获输出
                outputs = torch.utils._pytree.tree_leaves(func(*args))
                # 计算梯度输入
                grad_inputs = torch.autograd.grad(
                    outputs=tuple(o for o in outputs if o.requires_grad),
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(
                        torch.empty_like(o) for o in outputs if o.requires_grad
                    ),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                )
            # 清理内存，删除不再需要的变量
            del outputs, grad_inputs  # type: ignore[possibly-undefined]
    torch.cuda.synchronize()

    # 所有捕获的过程共享一个内存池。为了避免重放过程相互干扰内存，按照指定顺序捕获：
    # 前向传播 1, 前向传播 2, ... 前向传播 N, 然后反向传播 N, 反向传播 N-1, ... 反向传播 1.

    # 捕获每个可调用对象的前向传播图
    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    for func, args, fwd_graph in zip(callables, sample_args, fwd_graphs):
        with torch.cuda.graph(fwd_graph, pool=mempool):
            outputs = func(*args)

        # 扁平化输出并记录规范
        flatten_outputs, spec = torch.utils._pytree.tree_flatten(outputs)
        per_callable_static_outputs.append(tuple(flatten_outputs))
        per_callable_output_unflatten_spec.append(spec)

    # 按相反顺序捕获每个可调用对象的反向传播图
    per_callable_static_grad_outputs = []
    per_callable_static_grad_inputs = []
    for static_input_surface, static_outputs, bwd_graph, module_params in zip(
        reversed(per_callable_static_input_surfaces),
        reversed(per_callable_static_outputs),
        reversed(bwd_graphs),
        reversed(per_callable_module_params),
        # 循环结束后，返回结果
    ):
        # For now, assumes all static_outputs require grad
        # assert all(o.requires_grad for o in static_outputs), "Outputs of graphed callables must require grad."
        static_grad_outputs = tuple(
            torch.empty_like(o) if o.requires_grad else None for o in static_outputs
        )

        with torch.cuda.graph(bwd_graph, pool=mempool):
            # Compute gradients using the backward graph and memory pool
            grad_inputs = torch.autograd.grad(
                outputs=tuple(o for o in static_outputs if o.requires_grad),
                inputs=tuple(i for i in static_input_surface if i.requires_grad),
                grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                only_inputs=True,
                allow_unused=allow_unused_input,
            )

        # Constructs a tuple suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs that don't require grad.
        # I couldn't think of a slick one-liner for this pattern.
        static_grad_inputs = []
        grad_idx = 0
        for arg in static_input_surface:
            if arg.requires_grad:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)  # type: ignore[arg-type]
        static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]

        per_callable_static_grad_outputs.append(static_grad_outputs)
        per_callable_static_grad_inputs.append(static_grad_inputs)

    # Reverses the most recent two lists
    per_callable_static_grad_outputs.reverse()
    per_callable_static_grad_inputs.reverse()
    # Now for every per_callable list, per_callable_*[i] holds the stuff for the ith callable.

    def make_graphed_autograd_function(
        fwd_graph,
        bwd_graph,
        module_params,
        len_user_args,
        output_unflatten_spec,
        static_input_surface,
        static_outputs,
        static_grad_outputs,
        static_grad_inputs,
        ):
            # 定义一个内部类 Graphed，继承自 torch.autograd.Function
            class Graphed(torch.autograd.Function):
                @staticmethod
                def forward(ctx, *inputs):
                    # 前向传播函数，接收参数 ctx 和 *inputs
                    # 在这个阶段，只有用户参数可能是新的张量。
                    for i in range(len_user_args):
                        # 如果 static_input_surface[i] 和 inputs[i] 的数据指针不同，则复制 inputs[i] 到 static_input_surface[i]
                        if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                            static_input_surface[i].copy_(inputs[i])
                    # 重放前向图
                    fwd_graph.replay()
                    # 断言 static_outputs 是一个元组
                    assert isinstance(static_outputs, tuple)
                    # 返回 static_outputs 的每个元素的 detach 副本组成的元组
                    return tuple(o.detach() for o in static_outputs)

                @staticmethod
                @torch.autograd.function.once_differentiable
                def backward(ctx, *grads):
                    # 反向传播函数，接收参数 ctx 和 *grads
                    assert len(grads) == len(static_grad_outputs)
                    # 对每对 static_grad_outputs 和 grads 进行循环
                    for g, grad in zip(static_grad_outputs, grads):
                        if g is not None:
                            # 如果 autograd 平台已经处理好，grad 已经在正确的位置，就不需要复制
                            if g.data_ptr() != grad.data_ptr():
                                g.copy_(grad)
                    # 重放反向图
                    bwd_graph.replay()

                    # 对于不需要梯度的输入参数，期望一个 None 梯度
                    assert isinstance(static_grad_inputs, tuple)
                    # 返回 static_grad_inputs 中每个元素的 detach 副本组成的元组
                    return tuple(
                        b.detach() if b is not None else b for b in static_grad_inputs
                    )

        def functionalized(*user_args):
            # 定义函数 functionalized，接收任意数量的用户参数 *user_args
            # 运行 autograd 函数，输入是图中可能需要梯度的所有输入（显式用户参数 + 模块参数）
            # 假设模块参数自捕获以来没有改变
            flatten_user_args = torch.utils._pytree.arg_tree_leaves(*user_args)
            # 调用 Graphed 的 apply 方法，传入展平的用户参数和模块参数，得到 out
            out = Graphed.apply(*(tuple(flatten_user_args) + module_params))
            # 使用 torch.utils._pytree.tree_unflatten 将 out 解展开，按照 output_unflatten_spec 规定解展开
            return torch.utils._pytree.tree_unflatten(out, output_unflatten_spec)

        # 返回 functionalized 函数
        return functionalized

    # 组合最终的图形调用
    ret = []
    # 遍历可调用对象列表，并使用索引 i 和对应的可调用对象 func 进行迭代
    for i, func in enumerate(callables):
        # 使用给定的参数创建图形化自动求导函数
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],  # 前向图
            bwd_graphs[i],  # 后向图
            per_callable_module_params[i],  # 每个可调用对象的模块参数
            per_callable_len_user_args[i],  # 每个可调用对象的用户参数长度
            per_callable_output_unflatten_spec[i],  # 每个可调用对象的输出展开规范
            per_callable_static_input_surfaces[i],  # 每个可调用对象的静态输入表面
            per_callable_static_outputs[i],  # 每个可调用对象的静态输出
            per_callable_static_grad_outputs[i],  # 每个可调用对象的静态梯度输出
            per_callable_static_grad_inputs[i],  # 每个可调用对象的静态梯度输入
        )

        # 如果 func 是一个 torch.nn.Module 对象
        if isinstance(func, torch.nn.Module):

            # 定义一个函数，用于替换 func 的 forward 方法，以便在图形化训练状态下执行图形化方法
            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):
                def new_fwd(*user_args):
                    # 如果模块的训练状态与图形化的训练状态相匹配，则执行图形化方法
                    if func.training == graph_training_state:
                        return graphed(*user_args)
                    else:
                        # 否则，执行原始的 forward 方法
                        return orig_fwd(*user_args)

                return new_fwd

            # 替换 func 的 forward 方法为图形化的新方法
            func.forward = make_graphed_forward(func, func.training, graphed, func.forward)  # type: ignore[assignment]
            # 将修改后的 func 添加到结果列表 ret 中
            ret.append(func)
        else:
            # 如果 func 不是 torch.nn.Module 对象，则直接将 graphed 添加到结果列表 ret 中
            ret.append(graphed)

    # 如果只有一个可调用对象，返回 ret 列表的第一个元素
    if just_one_callable:
        return ret[0]

    # 否则，返回结果列表 ret
    return tuple(ret)
```