# `.\pytorch\torch\_functorch\fx_minifier.py`

```
# 忽略类型检查错误，这通常用于标记代码中的类型系统问题，让类型检查工具跳过这些错误
# 导入必要的库和模块
import copy  # 导入 copy 模块，用于对象的深拷贝和浅拷贝操作
import math  # 导入 math 模块，提供数学运算函数
import os  # 导入 os 模块，提供与操作系统交互的功能
import sys  # 导入 sys 模块，提供对 Python 解释器的访问和控制
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器，用于创建数据类
from functools import partial, wraps  # 从 functools 模块导入 partial 和 wraps 函数，用于函数式编程中的工具
from typing import Callable, List  # 从 typing 模块导入 Callable 和 List 类型

import torch  # 导入 PyTorch 库
import torch.fx as fx  # 导入 PyTorch 的 fx 模块，用于图模式编程
from torch.hub import tqdm  # 从 torch.hub 模块导入 tqdm 函数，用于显示进度条
from torch.multiprocessing.reductions import StorageWeakRef  # 从 torch.multiprocessing.reductions 模块导入 StorageWeakRef 类
from torch.utils._content_store import ContentStoreWriter  # 从 torch.utils._content_store 模块导入 ContentStoreWriter 类
from .compile_utils import get_outputs, get_placeholders  # 导入本地模块中的函数或类

is_tuple = object()  # 创建一个 is_tuple 对象

@dataclass
class LoadTensorMeta:
    size: List[int]  # 定义 LoadTensorMeta 类的 size 属性，类型为 List[int]
    stride: List[int]  # 定义 LoadTensorMeta 类的 stride 属性，类型为 List[int]
    dtype: torch.dtype  # 定义 LoadTensorMeta 类的 dtype 属性，类型为 torch.dtype
    device: torch.device  # 定义 LoadTensorMeta 类的 device 属性，类型为 torch.device

class ConcreteProp(torch.fx.Interpreter):
    def __init__(self, mod, *, writer=None, skip_offload=False):
        super().__init__(mod)  # 调用父类的构造方法初始化
        self.writer = writer  # 设置属性 writer，用于写入数据
        self.skip_offload = skip_offload  # 设置属性 skip_offload，用于控制是否跳过卸载操作
        self.seen_storages = set()  # 创建空集合，用于跟踪已经处理过的存储对象

    def run_node(self, n):
        self.pbar.update(1)  # 更新进度条状态
        r = super().run_node(n)  # 调用父类的 run_node 方法执行节点 n
        name = n.name  # 获取节点的名称

        if isinstance(r, torch.Tensor):  # 如果结果 r 是 torch.Tensor 对象
            if self.writer is None:  # 如果没有指定 writer
                n.meta["concrete_value"] = r  # 将节点的 meta 属性中的 concrete_value 设置为 r
            else:
                if StorageWeakRef(r.untyped_storage()) in self.seen_storages:
                    # 如果 r 的存储已经在 seen_storages 中存在
                    # 拒绝卸载别名其他活动张量，因为这可能违反操作符合约
                    n.meta["concrete_value"] = None  # 将节点的 concrete_value 设置为 None
                else:
                    if not self.skip_offload:  # 如果不跳过卸载操作
                        self.writer.write_tensor(os.path.join("eager", name), r)  # 将张量写入指定路径
                    n.meta["concrete_value"] = LoadTensorMeta(
                        r.size(), r.stride(), r.dtype, r.device
                    )  # 设置节点的 concrete_value 为 LoadTensorMeta 对象
                    self.seen_storages.add(StorageWeakRef(r.untyped_storage()))  # 将 r 的存储加入 seen_storages
        else:
            n.meta["concrete_value"] = is_tuple  # 如果结果不是 torch.Tensor，将节点的 concrete_value 设置为 is_tuple

        return r  # 返回执行结果 r

    def propagate(self, *args):
        with tqdm(
            desc="Saving intermediates for delta debugging",
            total=len(self.module.graph.nodes),
            disable=self.writer is None,
        ) as pbar:
            self.pbar = pbar  # 将进度条赋值给实例属性 pbar
            r = super().run(*args)  # 调用父类的 run 方法执行传入的参数
            if not self.skip_offload:  # 如果不跳过卸载操作
                pbar.set_description(
                    "Saved!  To skip next time, run with --skip-saving-eager-intermediates"
                )  # 更新进度条的描述信息为 "Saved!"
            return r  # 返回执行结果 r

def is_load_tensor_node(node):
    return (
        node.op == "call_function"
        and node.target is torch.ops.debugprims.load_tensor.default
    )  # 检查节点是否为调用 load_tensor.default 函数的节点

# inplace modifies node/inps
def _convert_node_to_placeholder(graph, node, inps):
    if node.op == "output" or node.op == "placeholder":  # 如果节点的操作为 "output" 或 "placeholder"
        return False  # 返回 False，表示不进行转换

    if is_load_tensor_node(node):  # 如果节点是调用 load_tensor.default 函数的节点
        return False  # 返回 False，表示不进行转换

    concrete_val = node.meta.get("concrete_value", None)  # 获取节点的 concrete_value 属性

    if isinstance(concrete_val, torch.Tensor):  # 如果 concrete_value 是 torch.Tensor 对象
        node.op = "placeholder"  # 将节点的操作设置为 "placeholder"
        node.target = node.name  # 将节点的目标设置为节点的名称
        node.args = ()  # 清空节点的位置参数
        node.kwargs = {}  # 清空节点的关键字参数

        inps.append(concrete_val)  # 将 concrete_val 添加到 inps 列表中
        return True  # 返回 True，表示成功进行了转换

    elif concrete_val is None:  # 如果 concrete_value 为 None
        return False  # 返回 False，表示不进行转换
    elif concrete_val is is_tuple:
        r = False
        for tuple_user in list(node.users):
            r = _convert_node_to_placeholder(graph, tuple_user, inps) or r
        # NB: We must not erase the node at this point, because
        # we are iterating over the nodes and this would change
        # the iteration order
        # 在这个点上我们不能擦除节点，因为
        # 我们正在遍历节点，这会改变
        # 遍历的顺序
        # graph.erase_node(node)
        return r


```        
    elif isinstance(concrete_val, LoadTensorMeta):
        node.op = "call_function"
        node.target = torch.ops.debugprims.load_tensor.default
        node.args = (
            os.path.join("eager", node.name),
            concrete_val.size,
            concrete_val.stride,
        )
        node.kwargs = {
            "device": concrete_val.device,
            "dtype": concrete_val.dtype,
        }
        return True



    return False
# 定义一个函数，用于将给定的最小化 FX 图形转换为 HLO 格式并保存到本地目录
def create_minified_hlo_graph(minified_fx_graph, inputs):
    """
    Takes minified FX graph as primary input, and ports it to HLO via StableHLO
    Provides minified HLO graph as output, and archive them to local directory
    """
    # 获取当前工作目录，并创建一个子目录用于存放 HLO 文件
    hlo_dir = f"{os.getcwd()}/hlo_files"
    os.makedirs(hlo_dir, exists_ok=True)

    # 导入 StableHLO 模块，并将最小化的 FX 图形保存为 StableHLO 文件
    from torch_xla.stablehlo import save_torch_model_as_stablehlo
    save_torch_model_as_stablehlo(minified_fx_graph, inputs, hlo_dir)


# 定义一个函数，打印包含有关 FX 图形节点数量的工作复现信息，并生成输入数据的信息和初始化代码
def dump_state(fx_g, inps):
    print(
        f"""
# Working Repro with {len(fx_g.graph.nodes)} nodes
inps = {[(i.shape, i.dtype, i.device.type) for i in inps]}
inps = [torch.zeros(())] + [torch.ones(shape, dtype=dtype, device=device) for (shape, dtype, device) in inps]
{fx_g.code}
"""
    )


# 定义一个函数，用于判断一个整数是否为2的幂
def is_power_of_two(n):
    if n == 0:
        return False
    return (n & (n - 1)) == 0


# 定义一个数据类，用于保存复现状态，包括 FX 图形和输入数据列表
@dataclass
class ReproState:
    graph: fx.Graph
    inps: List[torch.Tensor]

    def __post_init__(self):
        # 在初始化完成后，检查 FX 图形中的占位符节点数与输入数据列表的长度是否相等
        ph_nodes = get_placeholders(self.graph)
        assert len(ph_nodes) == len(self.inps)


# 定义一个函数，用于最小化 FX 图形，使得经过最小化后的图形仍然能够通过模块失败函数检查
def minifier(
    fail_f: fx.GraphModule,
    inps,
    module_fails,
    dump_state: Callable = dump_state,
    *,
    save_dir=None,
    offload_to_disk=False,
    skip_offload=False,
    skip_sanity=False,
    max_granularity=None,
):
    """
    Minimizes a FX graph with given inputs, such that the resulting FX graph still returns True for module_fails.

    Does 2 main strategies:
    1. Truncates suffix: Removes some suffix from the graph and sets a new output.
    2. Delta Debugging: Tries replacing half of the graph with inputs. If fails,
        tries replacing quarter of the graph, etc.

    >>> # xdoctest: +SKIP(failing)
    >>> failing_function = fx.symbolic_trace(f)
    >>> minimize(failing_function, [torch.randn(5)], lambda fx_g, inps: fx_g(*inps))

    note: module_fails returns True if it fails.
    """
    assert isinstance(inps, (tuple, list))

    # 获取失败的 FX 图形，并记录其节点数
    failing_graph = fail_f.graph
    cur_size = len(failing_graph.nodes)

    # 如果设置了 max_granularity 参数，确保其为2的幂
    if max_granularity is not None and not is_power_of_two(max_granularity):
        raise RuntimeError(f"max_granularity {max_granularity} not power of two")

    # 记录查询次数的变量
    num_queries = 0

    # 深拷贝 FX 图形的函数
    def deepcopy_fx_graph(fx_graph):
        return fx.GraphModule(fail_f, copy.deepcopy(fx_graph)).graph

    # 检查图形是否失败的函数
    def graph_fails(graph, inps):
        nonlocal num_queries
        graph = copy.deepcopy(graph)
        num_queries += 1
        mod = fx.GraphModule(fail_f, graph)
        mod.graph.lint()
        return module_fails(mod, inps)

    # 如果需要将结果写入磁盘，则创建 ContentStoreWriter 对象
    writer = None
    if offload_to_disk:
        writer = ContentStoreWriter(save_dir)

    # 将失败的 FX 图形传播给 ConcreteProp 类处理，并根据需要将结果写入磁盘
    ConcreteProp(fail_f, writer=writer, skip_offload=skip_offload).propagate(*inps)

    # 如果未跳过健全性检查，并且输入图形未失败测试，则引发运行时错误
    if not skip_sanity and not graph_fails(failing_graph, inps):
        raise RuntimeError("Input graph did not fail the tester")
    
    # 打印初始节点数信息到标准错误流
    print(f"Started off with {cur_size} nodes", file=sys.stderr)
    def _register_strategy(strategy: Callable, name: str):
        # 定义一个内部函数，用于注册并执行给定的策略
        @wraps(strategy)
        def new_func(old_state: ReproState, granularity=1):
            # 打印空行到标准错误流
            print(file=sys.stderr)
            # 打印策略的执行信息，包括策略名称、粒度、图节点数和输入数量，输出到标准错误流
            print(
                f"Strategy: {name} (G: {granularity}) "
                f"({len(old_state.graph.nodes)} nodes, {len(old_state.inps)} inputs)",
                file=sys.stderr,
            )
            # 深拷贝原始状态的图，并执行给定的策略函数
            new_state = strategy(
                deepcopy_fx_graph(old_state.graph), list(old_state.inps), granularity
            )
            # 如果策略执行成功
            if new_state is not None:
                # 计算新旧状态的节点数、输入数和输出数
                new_nodes = len(new_state.graph.nodes)
                old_nodes = len(old_state.graph.nodes)
                new_inps = len(new_state.inps)
                old_inps = len(old_state.inps)
                new_outs = len(get_outputs(new_state.graph))
                old_outs = len(get_outputs(old_state.graph))
                progress_made = False
                # 检查是否有节点数减少
                if new_nodes < old_nodes:
                    progress_made = True
                    # 打印节点数减少的成功信息到标准错误流
                    print(
                        f"SUCCESS: Went from {old_nodes} to {new_nodes} nodes",
                        file=sys.stderr,
                    )
                # 检查是否有输入数增加
                if new_inps > old_inps:
                    progress_made = True
                    # 打印输入数增加的成功信息到标准错误流
                    print(
                        f"SUCCESS: Went from {old_inps} to {new_inps} inputs",
                        file=sys.stderr,
                    )
                # 检查是否有输出数减少
                if new_outs < old_outs:
                    progress_made = True
                    # 打印输出数减少的成功信息到标准错误流
                    print(
                        f"SUCCESS: Went from {old_outs} to {new_outs} outputs",
                        file=sys.stderr,
                    )

                # 如果没有任何进展则抛出运行时错误
                if not progress_made:
                    raise RuntimeError("Success raised but no progress made?")

                # 检查新状态的图是否存在问题，若有则打印警告信息到标准错误流，并返回None
                if not graph_fails(new_state.graph, new_state.inps):
                    print(
                        "WARNING: Something went wrong, not applying this minification",
                        file=sys.stderr,
                    )
                    return None
                # 返回新的状态
                return new_state
            else:
                # 打印策略执行失败的信息到标准错误流
                print(f"FAIL: {name}", file=sys.stderr)
            return None

        return new_func

    # 返回一个函数，用于注册给定名称的策略
    def register_strategy(name: str):
        return partial(_register_strategy, name=name)

    # 注册一个名为"Truncate suffix"的策略
    @register_strategy("Truncate suffix")
    # 从当前图中移除后缀操作，检查节点的影响范围和粒度
    def remove_suffix(cur_graph, cur_inps, granularity):
        tested = set()  # 用于存储已经测试过的节点索引集合
        new_graph = fx.Graph()  # 创建一个新的图对象
        env = {}  # 环境变量字典，用于存储节点映射关系
        for idx, node in enumerate(cur_graph.nodes):  # 遍历当前图中的节点
            # 复制当前节点到新图中，并通过环境变量字典解析其依赖节点
            new_node = new_graph.node_copy(node, lambda x: env[x])
            # 如果节点操作不是“placeholder”或“output”
            if node.op not in ["placeholder", "output"]:
                # 如果 idx 能被 granularity * 2 整除，则已经检查过，跳过
                if (
                    idx % granularity == 0
                    and (idx % (granularity * 2) != 0)
                    and idx not in tested
                ):
                    # 在新图中创建输出节点
                    output_node = new_graph.output((new_node,))
                    # 如果新图节点数小于当前图节点数且图检查失败
                    if len(new_graph.nodes) < len(cur_graph.nodes) and graph_fails(
                        new_graph, cur_inps
                    ):
                        # 返回包含新图和当前输入的重现状态对象
                        return ReproState(new_graph, cur_inps)
                    else:
                        tested.add(idx)  # 将当前节点索引添加到已测试集合中
                        new_graph.erase_node(output_node)  # 在新图中移除输出节点
            env[node] = new_node  # 更新环境变量字典
        return None  # 返回空，表示未找到重现状态

    # 注册“移除输出”策略，用于移除当前图中的输出节点
    @register_strategy("Remove outputs")
    def remove_outputs(cur_graph, cur_inps, granularity):
        granularity = max(1, granularity // 2)  # 更新粒度值
        for idx, node in enumerate(cur_graph.nodes):  # 遍历当前图中的节点
            node.idx = idx  # 更新节点的索引值
            if node.op == "output":  # 如果节点操作是“output”
                output = node  # 将该节点设为输出节点
                break

        if isinstance(output.args[0], fx.Node):  # 如果输出节点的参数是图节点
            return None  # 返回空，表示未找到重现状态

        # 对输出节点的参数进行排序，并根据索引或大数值排序
        output_args = sorted(
            output.args[0], key=lambda x: x.idx if isinstance(x, fx.Node) else int(1e9)
        )
        if len(output_args) == 1:  # 如果输出节点参数只有一个
            return None  # 返回空，表示未找到重现状态

        # 根据指定粒度循环处理输出节点的参数
        for idx in range(0, len(output_args), granularity):
            # 更新输出节点的参数，移除指定粒度范围内的参数
            output.args = (output_args[:idx] + output_args[idx + granularity :],)
            if graph_fails(cur_graph, cur_inps):  # 如果图检查失败
                # 返回包含当前图和当前输入的重现状态对象
                return ReproState(cur_graph, cur_inps)
        return None  # 返回空，表示未找到重现状态

    # 移除未使用的输入节点，不做检查
    def remove_unused_inputs_unchecked(cur_state: ReproState):
        cur_graph = cur_state.graph  # 获取当前状态的图对象
        cur_inps = cur_state.inps  # 获取当前状态的输入列表
        ph_nodes = get_placeholders(cur_graph)  # 获取当前图中的占位符节点列表
        assert len(ph_nodes) == len(cur_inps)  # 断言占位符节点数量与输入数量相等

        new_inps = []  # 创建一个新的输入列表
        for idx in range(len(ph_nodes)):  # 遍历占位符节点列表
            if len(ph_nodes[idx].users) == 0:  # 如果当前占位符节点没有使用者
                cur_graph.erase_node(ph_nodes[idx])  # 在当前图中移除该占位符节点
            else:
                new_inps.append(cur_inps[idx])  # 否则将当前输入添加到新输入列表中
        if len(new_inps) < len(cur_inps):  # 如果新输入列表长度小于当前输入列表
            return ReproState(cur_graph, new_inps)  # 返回包含当前图和新输入的重现状态对象
        return None  # 返回空，表示未找到重现状态

    # 移除未使用的输入节点，并进行检查
    def remove_unused_inputs_checked(cur_state: ReproState):
        new_state = remove_unused_inputs_unchecked(cur_state)  # 调用未检查的移除未使用输入节点函数
        if new_state is not None and graph_fails(new_state.graph, new_state.inps):
            # 如果新状态不为空且图检查失败，则返回新状态
            return new_state
        return None  # 返回空，表示未找到重现状态

    # 私有函数，包装移除未使用输入节点的检查功能
    def _remove_unused_wrapper(cur_graph, cur_inps, granularity):
        return remove_unused_inputs_checked(ReproState(cur_graph, cur_inps))

    # 注册“移除未使用输入”策略，并调用移除未使用输入节点的包装函数
    remove_unused_inputs = register_strategy("Remove unused inputs")(
        _remove_unused_wrapper
    )

    # 注册“消除死代码”策略
    @register_strategy("Eliminate dead code")
    # 消除死代码的函数，如果成功消除死代码并且图形失败，则返回一个新的状态对象
    def eliminate_dead_code(cur_graph, cur_inps, granularity):
        # 调用当前图形对象的消除死代码方法，并检查图形是否失败
        if cur_graph.eliminate_dead_code() and graph_fails(cur_graph, cur_inps):
            # 返回一个新的ReproState对象，包含当前图形和输入
            return ReproState(cur_graph, cur_inps)
        # 如果未满足条件，则返回空值
        return None

    # 合并占位符的函数，返回一个新的图形对象
    def _consolidate_placeholders(cur_graph, inps):
        # 创建一个新的空白图形对象
        new_graph = fx.Graph()
        # 创建一个空的环境字典
        env = {}
        # 标记是否已经看到非占位符节点
        seen_non_placeholder = False

        # 遍历当前图形中的每个节点
        # 将所有占位符移动到前面；如果任何load_tensor在前面，则将其转换为输入
        for node in cur_graph.nodes:
            # 如果节点的操作是"placeholder"
            if node.op == "placeholder":
                # 复制节点到新图形中，并根据环境字典进行映射
                new_node = new_graph.node_copy(node, lambda x: env[x])
                env[node] = new_node
            # 如果还没有看到非占位符，并且节点是load_tensor类型
            elif not seen_non_placeholder and is_load_tensor_node(node):
                # 在新图形中创建一个占位符节点，并将其添加到输入列表中
                new_node = new_graph.placeholder(node.name)
                env[node] = new_node
                inps.append(
                    torch.ops.debugprims.load_tensor.default(*node.args, **node.kwargs)
                )
            else:
                # 标记已经看到了非占位符节点
                seen_non_placeholder = True

        # 移动其他所有节点到新图形中
        for node in cur_graph.nodes:
            # 如果节点不在环境字典中
            if node not in env:
                # 复制节点到新图形中，并根据环境字典进行映射
                new_node = new_graph.node_copy(node, lambda x: env[x])
                env[node] = new_node
        # 返回新的图形对象
        return new_graph

    # 注册"Delta Debugging"策略的函数
    @register_strategy("Delta Debugging")
    def delta_debugging(cur_graph: fx.Graph, cur_inps, granularity):
        # 计算当前图形节点的数量
        num_nodes = len(cur_graph.nodes)
        # 按指定的粒度遍历图形节点
        for start_range in range(0, num_nodes, granularity):
            is_removing = False
            # 深度复制当前图形对象
            new_graph = deepcopy_fx_graph(cur_graph)
            # 复制当前输入列表
            new_inps = cur_inps[:]
            # 计算当前范围的结束位置
            end_range = min(num_nodes, start_range + granularity)
            # 遍历当前范围内的节点
            for idx in range(start_range, end_range):
                # 获取新图形中的节点
                new_node = list(new_graph.nodes)[idx]
                # 将节点转换为占位符，如果成功，标记is_removing为True
                if _convert_node_to_placeholder(new_graph, new_node, new_inps):
                    is_removing = True
            # 如果没有节点被移除，则继续下一次迭代
            if not is_removing:
                continue
            # 消除新图形中的死代码
            new_graph.eliminate_dead_code()
            # 合并新图形中的占位符
            new_graph = _consolidate_placeholders(new_graph, new_inps)
            # 移除未使用的输入，并获取新的状态对象
            new_state = remove_unused_inputs_unchecked(ReproState(new_graph, new_inps))
            # 如果移除后的状态对象为空，则使用原始状态对象
            if new_state is None:
                new_state = ReproState(new_graph, new_inps)
            # 如果图形失败，则返回新的状态对象
            if graph_fails(new_state.graph, new_state.inps):
                return ReproState(new_state.graph, new_state.inps)

        # 如果未找到任何失败的情况，则返回空值
        return None

    # 注册"Consolidate Inputs"策略的函数
    @register_strategy("Consolidate Inputs")
    def consolidate_inputs(cur_graph, cur_inps, granularity):
        # 记录旧输入列表的长度
        old_len = len(cur_inps)
        # 合并当前图形中的占位符
        cur_graph = _consolidate_placeholders(cur_graph, cur_inps)
        # 如果新输入列表的长度大于旧长度，并且图形失败，则返回新状态对象
        if len(cur_inps) > old_len and graph_fails(cur_graph, cur_inps):
            return ReproState(cur_graph, cur_inps)
        # 如果未找到失败的情况，则返回空值
        return None

    # 创建一个包含失败图形和输入列表的ReproState对象
    failing_state = ReproState(failing_graph, inps)
    # 打印当前尝试的粒度信息到标准错误输出
    print(f"Trying granularity {granularity}", file=sys.stderr)

    # 初始化策略列表
    strategies = []
    # 获取当前状态图中的节点数和输出数
    num_nodes = len(failing_state.graph.nodes)
    num_outputs = len(get_outputs(failing_state.graph))
    # 如果输出数超过节点数的一半，添加移除输出的策略
    if num_outputs > num_nodes // 2:
        strategies += [remove_outputs]

    # 如果使用非粒度化策略，添加以下几种策略
    if use_non_granular:
        strategies += [
            eliminate_dead_code,
            remove_unused_inputs,
            consolidate_inputs,
        ]

    # 始终添加移除后缀和增量调试的策略
    strategies += [remove_suffix, delta_debugging]

    # 遍历策略列表，依次尝试应用每个策略
    for strategy in strategies:
        new_state = strategy(failing_state, granularity)
        # 如果成功找到新状态，则返回该新状态
        if new_state is not None:
            return new_state
    # 如果所有策略均未成功，返回 None
    return None

while True:
    # 将当前状态的图模块和输入转储到标准输出
    dump_state(fx.GraphModule(fail_f, failing_state.graph), failing_state.inps)
    # 计算当前图节点数量的对数，以确定粒度
    granularity = int(2 ** (math.floor(math.log2(len(failing_state.graph.nodes)))))
    # 如果设置了最大粒度限制，则取最小值作为当前粒度
    if max_granularity is not None:
        granularity = min(max_granularity, granularity)
    # 尝试使用粒度优化策略，返回新状态
    new_state = try_granularity(failing_state, granularity, use_non_granular=True)
    # 如果找到新状态，则更新当前失败状态并继续循环
    if new_state is not None:
        failing_state = new_state
        continue

    # 将当前粒度减半，并标记未取得进展
    granularity //= 2
    has_progress = False
    # 在当前粒度大于等于 1 的情况下，反复尝试使用粒度优化策略
    while granularity >= 1:
        new_state = try_granularity(
            failing_state, granularity, use_non_granular=False
        )
        # 如果找到新状态，则更新当前失败状态并标记已取得进展，跳出循环
        if new_state is not None:
            failing_state = new_state
            has_progress = True
            break
        # 粒度减半
        granularity //= 2
    # 如果有进展，继续循环
    if has_progress:
        continue

    # 如果以上所有策略均未找到新状态，尝试移除输出并更新当前失败状态
    new_state = remove_outputs(failing_state, 1)
    if new_state is not None:
        failing_state = new_state
        continue

    # 如果所有策略均未找到新状态，跳出循环
    break

# 如果最终的状态图不再失败，抛出运行时错误
if not graph_fails(failing_state.graph, failing_state.inps):
    raise RuntimeError("Uh oh, something went wrong :( Final graph is not failing")

# 打印查询次数到标准错误输出
print(f"Made {num_queries} queries", file=sys.stderr)

# 构建最终的失败图模块
failing_fx = fx.GraphModule(fail_f, failing_state.graph)

# 如果启用了 XLA 调试环境，同时创建简化的 HLO 图
if "XLA_HLO_DEBUG" in os.environ:
    create_minified_hlo_graph(failing_fx, failing_state.inps)

# 将最终的状态图模块和输入转储到标准输出
dump_state(failing_fx, failing_state.inps)
# 打印信息指示最小化的重现代码已写入 repro.py 到标准错误输出
print("Wrote minimal repro out to repro.py", file=sys.stderr)

# 返回最终的失败图模块和输入
return failing_fx, failing_state.inps
```