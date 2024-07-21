# `.\pytorch\torch\_functorch\_aot_autograd\logging_utils.py`

```
# mypy: allow-untyped-defs
"""
Contains utils for logging in AOTAutograd, including managing the names of the graphs under
compilation, capturing user-friendly tracebacks, and debug messages.
"""

import collections
from contextlib import contextmanager
from typing import List, Tuple

import torch
import torch.fx.traceback as fx_traceback

# This is a list since looking forward, we can have this arbitrarily nested.
graph_being_compiled: List[str] = []
# TODO: It would be nice to reset the numbering every time aot_id goes
# up, but this is annoying to do right now (because we don't know if
# an aot_id will come back from the dead), so right now this also happens
# to be a globally unique number too (at the cost of wobbling if you change
# how the graphs compile)
nth_graph: int = 0
model_name: str = "model"


def set_model_name(name):
    global model_name
    model_name = name


def get_aot_compilation_context() -> Tuple[List[str], str, int]:
    """
    Returns the current context of AOT compilation, including the list of graphs being compiled,
    the current model name, and the nth graph number.
    """
    return list(graph_being_compiled), model_name, nth_graph


def get_aot_graph_name() -> str:
    """
    Returns the formatted name of the current AOT graph being compiled.
    """
    global model_name, graph_being_compiled, nth_graph
    return f"{model_name}__{'_'.join(graph_being_compiled)}_{nth_graph}"


get_graph_being_compiled = get_aot_graph_name


@contextmanager
def track_graph_compiling(aot_config, graph_name):
    """
    Context manager to track the compilation of a graph.
    Sets the current graph being compiled and handles the tracing context.
    """
    global graph_being_compiled
    graph_being_compiled = [f"{aot_config.aot_id}_{graph_name}"]
    old_name = None
    if tracing_context := torch._guards.TracingContext.try_get():
        old_name = tracing_context.aot_graph_name
        tracing_context.aot_graph_name = graph_being_compiled
        has_tracing_context = True
    else:
        has_tracing_context = False
    try:
        yield
    finally:
        global nth_graph
        nth_graph += 1
        graph_being_compiled = []
        if has_tracing_context:
            if tracing_context := torch._guards.TracingContext.try_get():
                tracing_context.aot_graph_name = old_name


# Set up hooks so that during backward the fx's stack_trace is properly set
callback_set = False


def setup_stacktrace_preservation_hooks(roots: List):
    """
    Sets up hooks to preserve stack traces during backward pass.
    """
    def iter_graph(roots):
        if not roots:
            return
        seen = set()
        q = collections.deque()  # type: ignore[var-annotated]
        for node in roots:
            if node is not None and node not in seen:
                seen.add(node)
                q.append(node)

        while q:
            node = q.popleft()
            for fn, _idx in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)

            yield node

    def get_callback(saved_stack_):
        def callback():
            global callback_set
            fx_traceback.set_stack_trace(saved_stack_)
            callback_set = False

        return callback
    # 定义一个函数，用于获取前处理钩子函数，以设置梯度计算回调
    def get_prehook(stack_, seq_nr):
        # 定义前处理钩子函数，接收梯度输出作为参数
        def prehook(grad_output):
            # 声明全局变量 callback_set
            global callback_set

            # 如果回调函数未设置
            if not callback_set:
                # 在执行引擎队列中添加回调函数，获取函数调用堆栈信息
                torch.autograd.variable.Variable._execution_engine.queue_callback(  # type: ignore[attr-defined]
                    get_callback(fx_traceback.format_stack())
                )
                # 标记回调函数已设置
                callback_set = True

            # 设置函数调用堆栈信息
            fx_traceback.set_stack_trace(stack_)
            # 设置梯度函数序列号
            fx_traceback.set_grad_fn_seq_nr(seq_nr)

        return prehook

    # 定义一个函数，用于获取后处理钩子函数，以重置梯度函数序列号
    def get_posthook(special_stack_, seq_nr):
        # 定义后处理钩子函数，接收梯度输入和梯度输出作为参数
        def posthook(grad_input, grad_output):
            # 设置特殊的函数调用堆栈信息
            fx_traceback.set_stack_trace(special_stack_)
            # 重置梯度函数序列号
            fx_traceback.reset_grad_fn_seq_nr()

        return posthook

    # 遍历计算图中的每个节点
    for node in iter_graph(roots):
        # 获取节点的前向传播堆栈信息
        forward_node_stack = node.metadata.get("traceback_", [])
        # 注册前处理钩子函数到节点，设置前处理函数的堆栈和序列号
        node.register_prehook(get_prehook(forward_node_stack, node._sequence_nr()))

        # 创建特殊的函数调用堆栈信息，复制前向传播堆栈并添加特殊信息
        special_stack = forward_node_stack.copy()
        special_stack.append(
            "Gradient addition node due to multiple use of tensor around:"
        )
        # 注册后处理钩子函数到节点，设置后处理函数的特殊堆栈和序列号
        node.register_hook(get_posthook(special_stack, node._sequence_nr()))
# 根据输入索引和AOT配置描述输入的类型或用途
def describe_input(i, aot_config):
    # 如果输入索引小于AOT配置中参数缓冲区的数量，则描述为参数或缓冲区
    if i < aot_config.num_params_buffers:
        return f"parameter/buffer {i}"
    else:
        # 否则，描述为输入（索引减去参数缓冲区的数量）
        return f"input {i - aot_config.num_params_buffers}"


# 格式化用于报告编译时发生的守卫错误的消息
def format_guard_bug_msg(aot_config, expected):
    return (
        # 返回一个格式化的字符串，指示在编译时的假设和运行时实际情况不符
        f"At compilation time, graph {aot_config.aot_id} was compiled under the "
        f"assumption that {expected}, but at runtime this was not the case.  "
        "This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch."
    )
```