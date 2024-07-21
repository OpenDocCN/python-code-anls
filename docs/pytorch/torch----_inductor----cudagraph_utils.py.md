# `.\pytorch\torch\_inductor\cudagraph_utils.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和函数
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch._dynamo.utils import counters

# 获取性能提示的日志记录器
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")

@dataclasses.dataclass(frozen=True)
class FunctionID:
    "Unique counter of a function wrapped in cudagraphify_impl"
    id: int

@dataclasses.dataclass(frozen=True)
class WrappedFunction:
    """
    Represents a function that you want to record for CUDA graph replay,
    with a little more metadata so we can identify if we have an applicable
    CUDA graph in our CUDA graph tree for it.
    """
    model: Callable[..., Any]
    static_input_idxs: List[int]
    id: FunctionID
    constants: Tuple[torch.Tensor, ...]
    placeholders: List[torch.fx.Node]
    mutated_input_idxs: List[int]

# 获取图中所有占位符节点
def get_placeholders(graph: torch.fx.Graph) -> List[torch.fx.Node]:
    return [node for node in graph.nodes if node.op == "placeholder"]

# 获取占位符节点的变异使用堆栈跟踪信息
def get_mutating_use_stack_trace(placeholder_node: torch.fx.Node) -> Optional[str]:
    # 如果占位符节点的用户数为1，则返回其唯一的使用节点的堆栈跟踪信息
    if len(placeholder_node.users) == 1:
        return next(iter(placeholder_node.users)).meta.get("stack_trace", None)

    # 遍历占位符节点的所有使用节点，寻找具有特定操作的节点并返回其堆栈跟踪信息
    for use in placeholder_node.users:
        if use.target == torch.ops.aten.copy_.default:
            if stack_trace := use.meta.get("stack_trace", None):
                return stack_trace

    return None

# 格式化默认的跳过消息
def format_default_skip_message(reason: str) -> str:
    return f"skipping cudagraphs due to {reason}"

# 获取变异使用占位符的堆栈跟踪信息
def get_mutation_stack_trace(
    placeholders: List[torch.fx.Node], mutation_indices: List[int]
) -> str:
    stack_trace: Optional[str] = ""

    # 遍历变异索引列表，寻找具有变异使用的占位符节点，并获取其堆栈跟踪信息
    for idx in mutation_indices:
        placeholder = placeholders[idx]
        if stack_trace := get_mutating_use_stack_trace(placeholder):
            break

    # 格式化消息，指示变异输入的数量，并附加找到的堆栈跟踪信息（如果有的话）
    msg = format_default_skip_message(
        f"mutated inputs ({len(mutation_indices)} instances)"
    )
    if stack_trace:
        return f"{msg}. Found from : \n {stack_trace}"

    return msg

# 检查函数是否具有变异输入，并返回相关的堆栈跟踪信息
def check_for_mutation(
    func: WrappedFunction,
    inputs: List[torch.Tensor],
    is_cuda_graph_recorded_tensor: Callable[[torch.Tensor], bool],
) -> Optional[str]:
    # 如果开启了 cudagraph_trees，则检查变异是否仅发生在参数/静态输入上
    if torch._inductor.config.triton.cudagraph_trees:
        mutation_indices = [
            idx
            for idx in func.mutated_input_idxs
            if not (
                idx in func.static_input_idxs
                or is_cuda_graph_recorded_tensor(inputs[idx])
            )
        ]
    else:
        mutation_indices = func.mutated_input_idxs

    # 返回变异堆栈跟踪信息（如果有变异输入），否则返回 None
    return (
        get_mutation_stack_trace(func.placeholders, mutation_indices)
        if mutation_indices
        else None
    )

# 获取节点的使用堆栈跟踪信息
def get_use_stack_trace(node) -> Optional[str]:
    # 遍历节点的用户列表，对每个用户进行迭代处理
    for use in node.users:
        # 检查当前用户的元数据中是否存在名为 "stack_trace" 的项，并将其赋给 stack_trace 变量
        if stack_trace := use.meta.get("stack_trace", None):
            # 如果找到了 "stack_trace" 项，则返回其值，表示找到了对应的堆栈跟踪信息
            return stack_trace
    # 如果遍历完所有用户都没有找到 "stack_trace" 项，则返回 None，表示未找到堆栈跟踪信息
    return None
# 检查设备节点映射中是否存在多个设备或任何 CPU 节点
def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
) -> Optional[str]:
    # 如果存在 CPU 设备节点，则获取其对应的节点
    if cpu_node := device_node_mapping.get(torch.device("cpu")):
        # 构建包含 CPU 节点信息的消息字符串
        msg = f"cpu device ({cpu_node.name})"
        # 获取 CPU 节点的使用堆栈信息
        if stack_trace := get_use_stack_trace(cpu_node):
            # 格式化默认的跳过消息，包含 CPU 节点信息和堆栈跟踪信息
            return format_default_skip_message(f"{msg}. Found from : \n {stack_trace}")

        # 如果未获取到堆栈跟踪信息，则返回仅包含 CPU 节点信息的默认跳过消息
        return format_default_skip_message(msg)

    # 如果设备节点映射中只有一个 CUDA 设备节点，则返回 None
    if (
        len(device_node_mapping) == 1
        and next(iter(device_node_mapping.keys())).type == "cuda"
    ):
        return None

    # 构建包含所有设备节点信息的消息字符串，并返回格式化后的默认跳过消息
    keys_repr = (repr(key) for key in device_node_mapping.keys())
    return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")


# 检查降低禁用 CUDA 图的情况，并调用检查多设备或任何 CPU 节点的函数
def check_lowering_disable_cudagraph(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
):
    return check_multiple_devices_or_any_cpu_nodes(device_node_mapping)


# 记录 CUDA 图跳过并增加计数器
def log_cudagraph_skip_and_bump_counter(msg):
    # 使用性能提示日志记录消息
    perf_hint_log.warning(msg)
    # 增加 CUDA 图跳过计数器的计数值
    counters["inductor"]["cudagraph_skips"] += 1


# 用于封装设备索引的数据类
@dataclasses.dataclass
class BoxedDeviceIndex:
    value: Optional[int]

    # 设置设备索引值，要求设备索引为 None 或整数类型
    def set(self, device_idx: Optional[int]):
        assert device_idx is None or isinstance(device_idx, int)
        self.value = device_idx


# 检查是否忽略 CUDA 图管理张量的变异
def check_for_mutation_ignore_cuda_graph_managed_tensor(
    gm: torch.fx.GraphModule, compiled_graph, static_input_idxs: List[int]
) -> Optional[str]:
    # 构建默认的变异输入消息
    default_msg = format_default_skip_message("mutated inputs")

    # 如果 Triton 的 CUDA 图配置为树状结构
    if torch._inductor.config.triton.cudagraph_trees:
        # 获取静态输入索引的唯一集合
        unique_idxs = set(static_input_idxs)
        # 检查编译图中变异不在静态输入索引中的变异索引
        mutation_indices = [
            idx for idx in compiled_graph.mutated_input_idxs if idx not in unique_idxs
        ]
        # 判断是否存在变异
        has_mutation = len(mutation_indices) != 0
        # 如果不存在变异，则返回 None
        if not has_mutation:
            return None
        # 获取所有占位符节点，并返回变异堆栈跟踪信息
        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        return get_mutation_stack_trace(placeholders, mutation_indices)

    else:
        # 判断编译图中是否存在变异的输入
        has_mutation = len(compiled_graph.mutated_inputs) != 0
        # 如果不存在变异，则返回 None，否则返回默认的变异输入消息
        return None if not has_mutation else default_msg


# 获取占位符节点的堆栈跟踪信息
def get_placeholder_stack_trace(placeholder: torch.fx.Node) -> Optional[str]:
    """
    获取占位符或其用户的第一个非空堆栈跟踪信息。
    """
    # 如果占位符节点有堆栈跟踪信息，则返回该信息
    if placeholder.stack_trace:
        return placeholder.stack_trace

    # 遍历占位符节点的用户，如果用户节点有堆栈跟踪信息，则返回该信息
    for user in placeholder.users:
        if user.stack_trace:
            return user.stack_trace

    # 如果没有找到非空的堆栈跟踪信息，则返回 None
    return None
```