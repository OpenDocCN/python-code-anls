# `.\pytorch\torch\ao\ns\fx\n_shadows_utils.py`

```py
# mypy: allow-untyped-defs
# 引入 torch 库
import torch
# 引入 torch.fx 模块
import torch.fx
# 从 torch.fx 中导入 Node, GraphModule, Graph 类
from torch.fx import (
    Node,
    GraphModule,
    Graph,
)

# 从 torch.ao.ns.fx.utils 中导入工具函数
from torch.ao.ns.fx.utils import (
    # TODO(future PR): make this work correctly for methods
    get_target_type_str,
    get_normalized_nth_input,
)
# 从 torch.ao.ns.fx.ns_types 中导入类型定义
from torch.ao.ns.fx.ns_types import (
    NSSingleResultValuesType,
    NSResultsType,
)
# 导入 torch.ao.ns.fx.graph_passes 中的 _maybe_get_fqn 函数
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
# 导入 torch.ao.quantization 中的 QConfigMapping 类
from torch.ao.quantization import QConfigMapping
# 从 torch.ao.quantization.qconfig 中导入 QConfigAny 类
from torch.ao.quantization.qconfig import QConfigAny
# 从 torch.ao.quantization.utils 中导入 getattr_from_fqn 函数
from torch.ao.quantization.utils import getattr_from_fqn
# 导入 torch.ao.quantization.fx.match_utils 中的 _MatchResult 类
from torch.ao.quantization.fx.match_utils import _MatchResult
# 导入 torch.utils._pytree 中的 tree_map 函数
from torch.utils._pytree import tree_map

# 引入 collections 模块
import collections
# 引入 copy 模块
import copy
# 从 typing 模块中导入 List, Dict, Set, Tuple, Callable, Any, Optional 类型
from typing import List, Dict, Set, Tuple, Callable, Any, Optional
# 导入 operator 模块
import operator

# 定义变量 SHADOW_NODE_NAME_PREFIX，表示阴影节点名称的前缀
SHADOW_NODE_NAME_PREFIX = 'shadow'
# 定义变量 SHADOW_WRAPPER_NODE_NAME_PREFIX，表示阴影包装节点名称的前缀
SHADOW_WRAPPER_NODE_NAME_PREFIX = 'shadow_wrapper'

# TODO(future PR): reuse existing mapping instead of creating a new one
# 定义集合 BINARY_FUNCTIONS，包含 torch 和 operator 模块中的加法和乘法函数
BINARY_FUNCTIONS = {
    torch.add,
    torch.Tensor.add,
    operator.add,
    torch.mul,
    torch.Tensor.mul,
    operator.mul,
}

# 定义函数 _get_attr_name(subgraph_idx, subgraph_candidate_idx)，返回阴影节点名称
def _get_attr_name(subgraph_idx, subgraph_candidate_idx):
    return f"{SHADOW_NODE_NAME_PREFIX}_{subgraph_idx}_{subgraph_candidate_idx}"

# 定义函数 _get_attr_wrapper_name(subgraph_idx, subgraph_candidate_idx)，返回阴影包装节点名称
def _get_attr_wrapper_name(subgraph_idx, subgraph_candidate_idx):
    return f"{SHADOW_WRAPPER_NODE_NAME_PREFIX}_{subgraph_idx}_{subgraph_candidate_idx}"


# 定义类 OutputProp，表示输出传播（基于形状传播模型）
class OutputProp:
    """
    Output propagation (modeled from shape propagation).

    Given a GraphModule and an example input, saves the output flowing
    through each node on `node.traced_result`.

    Code based on the example from
    https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
    """
    # 初始化方法，接收 GraphModule 实例 mod，并设置相关属性
    def __init__(self, mod):
        # 保存 GraphModule 实例
        self.mod = mod
        # 获取模块的计算图对象
        self.graph = mod.graph
        # 获取模块中所有命名模块的字典
        self.modules = dict(self.mod.named_modules())
    # 定义 propagate 方法，用于执行节点操作传播
    def propagate(self, *args):
        # 创建参数迭代器
        args_iter = iter(args)
        # 初始化环境变量字典，映射节点名称到节点对象
        env: Dict[str, Node] = {}

        # 定义加载参数的内部函数
        def load_arg(a):
            # 使用 env 字典映射节点名称获取对应的节点对象
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        # 定义获取属性的内部函数
        def fetch_attr(target: str):
            # 按点分隔目标字符串，逐级访问对象属性
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    # 如果属性不存在，引发运行时异常
                    raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        # 遍历图中的每个节点
        for node in self.graph.nodes:
            # 根据节点操作类型执行相应的操作
            if node.op == 'placeholder':
                # 对于占位符节点，从参数迭代器中取出结果
                result = next(args_iter)
            elif node.op == 'get_attr':
                # 对于获取属性的节点，调用 fetch_attr 函数获取属性值
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                # 对于调用函数的节点，调用函数并传入加载后的参数
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                # 对于调用方法的节点，加载自身对象和参数后调用方法
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                # 对于调用模块的节点，从模块字典中获取模块并调用其方法
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # 如果结果是 torch.Tensor 类型，则将结果赋给节点的 traced_result 属性
            if isinstance(result, torch.Tensor):  # type: ignore[possibly-undefined]
                node.traced_result = result

            # 将节点的结果存入环境变量字典中，以便后续节点可以引用
            env[node.name] = result

        # propagate 方法执行完毕，返回 None
        return None
# 定义一个函数，从给定的 matches 字典中获取去重的子图集合
def _get_dedup_subgraphs(
    matches: Dict[str, _MatchResult]
) -> Dict[str, List[Node]]:
    # 用于跟踪已经处理过的节点，以确保每个子图的唯一性
    seen_nodes = set()
    # 存储去重后的子图集合的字典
    subgraphs_dedup = {}

    # 由于在 Python 3.8 之前，字典的 items() 不是可逆的，因此我们手动逆序处理它
    matches_items_reversed: List[Tuple[str, _MatchResult]] = []
    for name, cur_match in matches.items():
        matches_items_reversed.insert(0, (name, cur_match))

    # 注意：处理 matches 的顺序很重要。当前 matches 是逆序的，
    # 我们希望以非逆序的方式处理，这样可以创建直观的命名方案，
    # 比如将第一个操作的子模块命名为 `shadow_0_0` 到 `shadow_0_(n-1)`
    
    # 返回去重后的子图集合字典
    return subgraphs_dedup


# 定义一个函数，为给定的模型和线性子图创建一个日志记录器
def _get_logger_for_subgraph(
    model: GraphModule,
    first_node: Node,
    last_node: Node,
    subgraph_idx: int,
    subgraph_candidate_idx: int,
    qconfig_str: str,
    logger_cls: Callable,
    fqn: Optional[str],
) -> torch.nn.Module:
    """
    给定一个模型和从 `first_node` 开始到 `last_node` 结束的线性子图，
    创建该子图末端的日志记录器。
    """
    # 如果 fqn 为 None，则设置为空字符串
    if fqn is None:
        fqn = ''
    # 创建一个以给定参数初始化的日志记录器模块
    logger_mod_orig = logger_cls(
        first_node.name,  # ref_node_name，参考节点名称
        last_node.name,  # prev_node_name，前一个节点名称
        f'subgraph_{subgraph_idx}_{subgraph_candidate_idx}',  # model_name，模型名称
        'model',  # ref_name，参考名称
        get_target_type_str(last_node, model),  # prev_node_target_type，前一个节点的目标类型
        get_target_type_str(first_node, model),  # ref_node_target_type，参考节点的目标类型
        NSSingleResultValuesType.NODE_OUTPUT.value,  # results_type，结果类型
        0,  # index_within_arg，参数内部索引
        0,  # index_of_arg，参数索引
        fqn,  # fqn，全限定名
        qconfig_str,  # qconfig_str，配置字符串
    )
    # 通常期望用户先添加日志记录器，然后进行校准，然后转换，最后填充日志记录器。
    # 这就是为什么日志记录器默认是禁用的。
    # TODO（未来 PR）：重新考虑设计，使其更加直观。
    logger_mod_orig.enabled = False
    # 返回初始化的日志记录器模块
    return logger_mod_orig


# 定义一个函数，从给定的模型和线性子图创建一个子模块
def create_submodule_from_subgraph(
    model: torch.nn.Module,
    first_node: Node,
    last_node: Node,
) -> GraphModule:
    """
    输入：一个模型，以及模型内从 first_node 到 last_node 的线性子图。
    
    输出：一个包含子图副本的新子模块，其中第一个节点的输入成为子模块的输入，
         子图中的所有其他节点均被复制。
    """
    #
    # 创建一个空的 GraphModule，并附带一个空图
    #
    class M(torch.nn.Module):
        def forward(self, x):
            pass

    # 实例化一个 M 类型的对象
    m = M()
    # 对模型 m 进行符号跟踪，得到符号跟踪后的 GraphModule
    gm = torch.fx.symbolic_trace(m)
    # 获取符号跟踪后的图对象
    g = gm.graph
    # 反向遍历图中的节点，并逐一删除
    for node in reversed(gm.graph.nodes):
        g.erase_node(node)

    #
    # 修改图以包含我们子图的副本
    #

    # 设置当前节点为第一个节点的原始版本
    cur_node_orig = first_node
    # 获取当前节点的参数
    cur_args_orig = cur_node_orig.args
    # 获取当前节点的关键字参数
    cur_kwargs_orig = cur_node_orig.kwargs

    # 当前名称索引初始化
    cur_name_idx = 0

    # 迭代次数限制设定为100次
    iteration_limit = 100
    # 当前迭代次数初始化
    cur_iteration = 0

    # 设置输出
    g.output(cur_node_copy)

    # 重新编译符号跟踪后的 GraphModule
    gm.recompile()
    # 返回重新编译后的 GraphModule 对象
    return gm
# 创建一个经过转换和记录的子图的副本
def create_one_transformed_and_logged_copy_of_subgraph(
    mt: GraphModule,
    subgraph_idx: int,
    subgraph_candidate_idx: int,
    first_node: Node,
    last_node: Node,
    fqn: Optional[str],
    list_of_node_name_to_qconfig: List[Dict[str, QConfigAny]],
    example_inputs: Any,
    last_added_shadow_node_list: List[Optional[Node]],
    custom_prepare_fn: Optional[Callable] = None,
    custom_prepare_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    给定模块 `mt` 中的一个子图和一个子图候选索引，插入子图候选副本并用记录器进行处理。

    如果 subgraph_candidate_idx 为 0，则这是基准的 fp32 子图，我们只需在末尾添加一个记录器。

    如果 subgraph_candidate_idx 不为 0，则我们创建子图的副本并使用 `prepare_fx` 准备它。
    """

    # TODO(未来 PR): 将记录器类移动到 utils 中以消除循环依赖
    from torch.ao.ns._numeric_suite_fx import OutputLogger, OutputComparisonLogger

    if subgraph_candidate_idx == 0:
        # idx = 0 是子图的浮点数（原始）版本
        # 我们保持子图不变，并在末尾添加一个记录器

        qconfig_str = ''
        logger_mod_orig = _get_logger_for_subgraph(
            mt, first_node, last_node, subgraph_idx, subgraph_candidate_idx,
            qconfig_str, OutputLogger, fqn)

        attr_name = _get_attr_name(subgraph_idx, subgraph_candidate_idx)
        assert not hasattr(mt, attr_name)
        setattr(mt, attr_name, logger_mod_orig)
        with mt.graph.inserting_after(last_node):
            new_node = mt.graph.call_module(attr_name, args=(last_node,), kwargs={})
            last_added_shadow_node_list[0] = new_node

    mt.recompile()

# 创建子图的 n 个经过转换和记录的副本
def create_n_transformed_and_logged_copies_of_subgraph(
    mt: GraphModule,
    subgraph_idx: int,
    match_name: str,
    nodes_in_this_subgraph: List[Any],
    qconfig_mappings: List[QConfigMapping],
    list_of_node_name_to_qconfig: List[Dict[str, QConfigAny]],
    custom_prepare_fn: Optional[Callable] = None,
    custom_prepare_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    给定模块 `mt` 和一个子图索引，为所有 qconfigs 创建所需的子图副本，并用记录器进行处理。
    """
    # 现在，假设
    # 1. 第一个节点有一个输入
    # 2. 最后一个节点有一个输出

    # 现在，忽略包含非节点（元组等）的所有子图
    # TODO(未来 PR): 实现这一点
    if any(
        not isinstance(node, Node)
        for node in nodes_in_this_subgraph
    ):
        return

    first_node = nodes_in_this_subgraph[0]
    last_node = nodes_in_this_subgraph[-1]
    # 我们使用输出传播来在每个节点上填充示例值。
    # 使用前一个节点的示例值作为当前节点的输入。
    prev_node = get_normalized_nth_input(first_node, mt, 0)
    # 如果 prev_node 是一个列表，则 example_inputs 是一个列表，包含所有 prev_node 中节点的 traced_result
    if isinstance(prev_node, list):
        example_inputs = [x.traced_result for x in prev_node]
    # 如果 prev_node 是一个元组，则 example_inputs 是一个生成器表达式，包含所有 prev_node 中节点的 traced_result
    elif isinstance(prev_node, tuple):
        example_inputs = (x.traced_result for x in prev_node)  # type: ignore[assignment]
    else:
        # 当前一些客户模型的节点并不都有 traced_result，
        # 因此需要在没有示例输入的情况下进行保护，因为没有示例输入就无法进行量化
        # TODO(将来的PR): 一旦我们有了一个简单的重现情况，为此添加一个测试用例，
        # 参见 https://github.com/pytorch/pytorch/pull/80521/files#r975940489
        # 以获取更多上下文信息
        if hasattr(prev_node, 'traced_result'):
            example_inputs = (prev_node.traced_result,)  # type: ignore[attr-defined, assignment]
        else:
            # 如果无法获取节点的示例输入，输出错误消息并返回
            print(
                'unable to get example input for node ' +
                f'{first_node.format_node()}, skipping')
            return

    # 如果这个子图没有量化配置，则跳过添加日志记录器，这样可以减少内存使用，
    # 对于并非所有层都进行量化的模型尤其如此
    # TODO(将来): 考虑使这个行为可配置化
    found_at_least_one_qconfig = False
    for subgraph_candidate_idx in range(len(qconfig_mappings) + 1):

        if subgraph_candidate_idx == 0:
            # fp32基线不需要量化配置
            continue

        # a. 我们有 N 个阴影，因此 len(qconfig_mappings) 是 N
        # b. 我们将有 fp32 层 + N 个阴影，因此原始操作 + 阴影的总数将是 N+1
        # c. 由于 `subgraph_candidate_idx` 表示 (b)，我们需要从 (a) 中减去 1 来查询
        node_name_to_qconfig = \
            list_of_node_name_to_qconfig[subgraph_candidate_idx - 1]
        qconfig = node_name_to_qconfig[first_node.name]
        # 如果找到了有效的量化配置，则设置标志为 True 并跳出循环
        if qconfig is not None:
            found_at_least_one_qconfig = True
            break
    # 如果没有找到至少一个量化配置，则输出错误消息并返回
    if not found_at_least_one_qconfig:
        print('unable to find at least one qconfig for node ' +
              f'{first_node.format_node()}, skipping')
        return

    # 获取完全限定名称（fqn）以备用
    fqn = _maybe_get_fqn(first_node, mt)

    # 我们希望结果中包含自然顺序的子图，并且图中还包含自然顺序的阴影包装器和阴影记录器
    # 如果我们仅仅反向迭代，图将按自然顺序，但最终结果将是反向顺序
    # 因此，我们跟踪最后添加的阴影记录器，并始终在其后插入
    last_added_shadow_node_list: List[Optional[Node]] = [None]
    for subgraph_candidate_idx in range(len(qconfig_mappings) + 1):

        create_one_transformed_and_logged_copy_of_subgraph(
            mt, subgraph_idx, subgraph_candidate_idx, first_node,
            last_node, fqn, list_of_node_name_to_qconfig,
            example_inputs, last_added_shadow_node_list, custom_prepare_fn,
            custom_prepare_kwargs)
    def create_add_loggers_graph(
        model: GraphModule,
        subgraphs_dedup: Dict[str, List[Node]],
        qconfig_mapping: QConfigMapping,
        node_name_to_qconfig: Dict[str, QConfigAny],
    ) -> None:
        r"""
        Given a model, a model graph partition (currently a set of matched
        subgraphs) and instructions how to transform each subgraph
        (currently quantizing it according to qconfig_mapping), modifies
        the model graph to create an alternate path through the original graph,
        with each of the subgraphs quantized.  This is useful to compare
        propagation error of a transformation such as quantization.
    
        For example, given layer op0 and op1, there are four cases when handling op1:
        1. op0 and op1 quantized
        2. op0 and op1 unquantized
        3. op0 quantized, op1 unquantized
        4. op0 unquantized, op1 quantized
    
        Example input, case 1:
    
        .. code::
    
          x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
           \                        \          \                 \       # noqa: W605
             ---> op0_1 -> x1_1 ----> clog    op1_1 -> x2_1 ----> clog
    
        Example output, case 1:
    
        .. code::
    
          x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
           \                        \                           \        # noqa: W605
             ---> op0_1 -> x1_1 ----> clog -> op1_1 -> x2_1 ----> clog
    
        """
        # TODO(future PR): move logger classes to utils to remove circular dependency
        from torch.ao.ns._numeric_suite_fx import OutputLogger, OutputComparisonLogger
    
        def _get_subgraph_containing_node(node, subgraphs_dedup):
            for subgraph in subgraphs_dedup.values():
                if node in subgraph:
                    return subgraph
            return None
    
        # First, we need to create shadow branches, going from
        #
        #   x0 -> op0 -> x1 -> ...
        #
        #
        # to
        #
        #   x0 -> op0_0 -> x1_0 -> log -> ...
        #    \                     \
        #      -> op0_1 -> x1_1 -> clog
        #
        # Later, the outputs of each shadow will be rerouted to calculate
        # propagation error.
    
        # Note: we cannot iterate over matched subgraphs because some nodes
        # may not be matched. So, we iterate over nodes in the graph, and
        # associate them to matched subgraphs if possible.
    
        nodes_to_skip = set()
        # for each subgraph, save a mapping from first node of subgraph
        # to first and last node of the shadow of this subgraph
        orig_first_node_to_shadow_in_node = {}
        orig_first_node_to_shadow_out_node = {}
        # need to record original list because we will mutate the graph as we go
        orig_nodes = list(model.graph.nodes)  # type: ignore[union-attr, arg-type]
        cur_subgraph_idx = 0
        model.recompile()
    
        # Now, we go from
        #
        #   x0 -> op0_0 -> x1_0 -> log -> x1 -> op1_0 -> ...
        #    \                     \       \
        #      -> op0_1 -> x1_1 -> clog      -> op1_1 -> ...
        #
        # to
        #
        #   x0 -> op0_0 -> x1_0 -> log --> x1_0 -> op1_0 -> ...
        #    \                     \
    # nodes_to_skip 是一个集合，用于存储需要跳过处理的节点
    nodes_to_skip = set()

    # 遍历原始节点列表 orig_nodes
    for n in orig_nodes:
        # 如果节点 n 的操作类型是 'placeholder', 'get_attr', 'output'，或者 n 已经在 nodes_to_skip 中，跳过当前循环
        if n.op in ('placeholder', 'get_attr', 'output') or n in nodes_to_skip:
            continue

        # 尝试获取包含节点 n 的子图
        maybe_subgraph = _get_subgraph_containing_node(n, subgraphs_dedup)

        # 如果找到子图
        if maybe_subgraph is not None:
            # 获取子图的第一个节点和最后一个节点
            first_node, last_node = maybe_subgraph[0], maybe_subgraph[-1]
            # 将子图中的所有节点添加到 nodes_to_skip 中，表示这些节点应该跳过后续处理
            nodes_to_skip.update(maybe_subgraph)
        else:
            # 如果未找到子图，则处理单个节点 n
            first_node, last_node = n, n

        # 定义函数 maybe_remap_node_to_shadow，用于根据需要将节点映射到其阴影版本
        def maybe_remap_node_to_shadow(node):
            """
            如果未阴影化的节点有阴影版本，则返回该版本；否则返回原节点。
            """
            if not isinstance(node, Node):
                # 处理标量类型的节点
                return node

            if node.op in ('placeholder', 'get_attr'):
                return node

            # 查找前一个子图中此参数的阴影版本
            prev_subgraph = _get_subgraph_containing_node(
                node, subgraphs_dedup)
            if prev_subgraph is None:
                prev_subgraph = [node]
            prev_first_node = prev_subgraph[0]
            prev_shadow_output = \
                orig_first_node_to_shadow_out_node[prev_first_node]
            return prev_shadow_output

        # 获取当前节点的阴影输入
        cur_shadow_input = \
            orig_first_node_to_shadow_in_node[first_node]

        # 断言当前阴影输入不为空
        assert cur_shadow_input is not None

        # 对当前阴影输入的 args 和 kwargs 应用 maybe_remap_node_to_shadow 函数，进行节点映射
        cur_shadow_input.args = tree_map(
            maybe_remap_node_to_shadow, cur_shadow_input.args)
        cur_shadow_input.kwargs = tree_map(
            maybe_remap_node_to_shadow, cur_shadow_input.kwargs)

        # 重新编译模型
        model.recompile()
# 从给定的 shadow_wrapper 模块中获取权重信息的函数
def _get_weight_info_from_shadow_wrapper(shadow_wrapper: torch.nn.Module):
    # 输入: shadow_wrapper 模块
    # 输出如果 shadow_wrapper 模块具有加权操作:
    #   (quantize_fn, (quantize_fn_args))
    # 输出如果 shadow_wrapper 模块不具有加权操作:
    #   None

    # 现在暂定权重是 shadow 模块的第二个输入。
    # 如果这一点发生变化，我们可以稍后进行修正。
    placeholders_seen = 0
    for shadow_n in shadow_wrapper.graph.nodes:  # type: ignore[union-attr]
        if shadow_n.op != 'placeholder':
            continue

        placeholders_seen += 1
        if placeholders_seen != 2:
            continue

        # 子图如下所示
        #
        #   _input_scale_1 = self._input_scale_1
        #   _input_zero_point_1 = self._input_zero_point_1
        #   quantize_per_channel = torch.quantize_per_channel(
        #       w2_0, _input_scale_1, _input_zero_point_1,
        #       0, torch.qint8)
        #
        # 我们有 `w2_0`，并且正在遍历这个子图来获取 `_input_scale_1` 和 `_input_zero_point_1`

        assert len(shadow_n.users) == 1
        quant_node = next(iter(shadow_n.users.keys()))
        new_args: Any = None
        if quant_node.target == torch.quantize_per_channel:
            _weight, scale_node, zp_node, axis, dtype = quant_node.args
            scale_val = getattr_from_fqn(
                shadow_wrapper, scale_node.target)
            zp_val = getattr_from_fqn(
                shadow_wrapper, zp_node.target)
            new_args = (scale_val, zp_val, axis, dtype)
        else:
            assert quant_node.target == torch.quantize_per_tensor
            _weight, scale_node, zp_node, dtype = quant_node.args
            scale_val = getattr_from_fqn(
                shadow_wrapper, scale_node.target)
            zp_val = getattr_from_fqn(
                shadow_wrapper, zp_node.target)
            new_args = (scale_val, zp_val, dtype)
        return (quant_node.target, new_args)

    # 如果没有找到适合条件的 placeholder，返回 None
    return None


# 从给定的 GraphModule 中提取权重比较的函数
def extract_weight_comparison(m: GraphModule) -> NSResultsType:

    # 示例图:
    #
    #   w1 = self.w1
    #   b1 = self.b1
    #   linear = torch._C._nn.linear(x, w1, b1)
    #   shadow_0_0 = self.shadow_0_0(linear)
    #   shadow_wrapper_0_1 = self.shadow_wrapper_0_1(x, w1, b1)
    #   shadow_0_1 = self.shadow_0_1(shadow_wrapper_0_1, linear)
    #
    # 算法:
    # 1. 对于每个匹配我们允许列表中的 call_function 节点:
    # 2.   如果存在对应的 shadow wrapper，提取权重对

    # 注意: 这不是特别健壮的实现，但这没关系，因为这只是为依赖先前的双模型版本的老客户提供支持。
    # 待定是否需要进一步完善这一点。
    # 注意: 不支持模块，因为现有客户仅使用函数。

    # TODO(未来的 PR): 将这些内容移到配置中
    weighted_ops = {
        torch.nn.functional.linear,
    }
    # 创建一个名为results的变量，类型为NSResultsType，初始化为包含'model'键的字典，
    # 其中'model'键对应的值是一个字典，初始包含一个空字典，表示存储模型权重相关的数据。
    results: NSResultsType = {
        'model': {NSSingleResultValuesType.WEIGHT.value: {}}
    }
    
    # 返回results变量，这个变量包含了模型权重相关的数据结构。
    return results
# 为给定结果创建一个按子图分组的结果比较
def group_results_by_subgraph(results: NSResultsType) -> Any:
    """
    创建结果的比较

    Input:

    {
      'model': {
        'node_output': {
          'subgraph_0_0': [
            'values': [torch.tensor(...), ...], ...
            'ref_node_name': ...,
            'ref_node_target_type': ...,
            'qconfig_str': ...,
            'comparisons': [], ...
            'comparison_fn_name': '',
            'fqn': '...',
          ],
          'subgraph_0_1': [
            'values': [torch.tensor(...), ...], ...
            'ref_node_name': ...,
            'ref_node_target_type': ...,
            'qconfig_str': ...,
            'comparisons': [torch.tensor(...), ...], ...
            'comparison_fn_name': '...',
            'fqn': '...',
          ],
          ...
        },
      },
    }

    Output:
    {
      'subgraph_0': {
        '0': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': None,
          'comparisons': [torch.tensor(...), ...], ...
          'comparison_fn_name': '...',
          'fqn': '...',
        },
        '1': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '...',
          'comparisons': [torch.tensor(...), ...], ...
          'comparison_fn_name': '...',
          'fqn': '...',
        },
      },
    }

    """
    # 使用默认字典创建子图名到子图结果的映射
    subgraph_name_to_subgraph_results: Any = collections.defaultdict(dict)

    # 选择要使用的键：'node_output' 或 'weight'
    key_to_use = next(iter(results['model'].keys()))

    # 遍历每个子图及其候选结果
    for subgraph_name_with_idx, subgraph_candidate_results in \
            results['model'][key_to_use].items():

        # 将形如 `subgraph_m_n` 的名称转换为 `subgraph_m` 和 `n`
        subgraph_str, subgraph_idx, subgraph_candidate_idx = \
            subgraph_name_with_idx.split('_')
        subgraph_name = f'{subgraph_str}_{subgraph_idx}'

        # 提取子图的结果信息
        subgraph_results = {
            'ref_node_name': subgraph_candidate_results[0]['ref_node_name'],
            'ref_node_target_type': subgraph_candidate_results[0]['ref_node_target_type'],
            'fqn': subgraph_candidate_results[0]['fqn'],
            'values': subgraph_candidate_results[0]['values'],
            'qconfig_str': subgraph_candidate_results[0]['qconfig_str'],
            'comparisons': subgraph_candidate_results[0]['comparisons'],
            'comparison_fn_name': subgraph_candidate_results[0]['comparison_fn_name'],
        }

        # 将子图结果存入字典中
        subgraph_name_to_subgraph_results[subgraph_name][subgraph_candidate_idx] = \
            subgraph_results

    # 返回组织好的结果字典
    return dict(subgraph_name_to_subgraph_results)


# TODO(future PR): redesign this to make it easier to consume outputs
def create_results_comparison(
    results_grouped,
) -> Any:
    """
    创建结果的比较

    Input:
    ```

    # TODO: 完成该函数的注释
    """
    # 此处需要补充函数的实现和注释
    pass
    {
      'subgraph_0': {
        '0': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '',
          'comparisons': [],
          'comparison_fn_name': '',
          'fqn': '...',
        },
        '1': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '...',
          'comparisons': [torch.tensor(...), ...],
          'comparison_fn_name': 'sqnr',
          'fqn': '...',
        },
      },
    }

    Output:
    {
      'subgraph_0': {
        'ref_node_name': '...',
        'ref_node_target_type': '...',
        'fqn': '...',
        'candidates': {
          '1': {
            'qconfig_str': ...,
            'comparison_fn_name': 'sqnr',
            'cmp_raw': [..., ...],
            'cmp_mean': ...,
          },
          ...,
        },
      },
    }
    """

    # 初始化一个空字典用于存储比较结果
    results_comparison = {}

    # 遍历分组后的结果，分别处理每个子图
    for subgraph_name, subgraph_results in results_grouped.items():

        # 初始化空字典，用于存储每个子图的候选项比较结果
        candidates = {}

        # 遍历每个子图内部的结果
        for subgraph_inner_name, subgraph_inner_result in subgraph_results.items():
            # 如果子图名称为 '0'，跳过，因为这是基线对基线的比较
            if subgraph_inner_name == '0':
                continue

            # 获取预先计算的比较结果
            cmp_raw = subgraph_inner_result['comparisons']
            cmp_raw_tensor = torch.stack(cmp_raw)

            # 存储每个候选项的比较结果和相关信息
            candidates[subgraph_inner_name] = {
                'qconfig_str': subgraph_inner_result['qconfig_str'],
                'comparison_fn_name': subgraph_inner_result['comparison_fn_name'],
                'cmp_raw': cmp_raw_tensor,
                'cmp_mean': torch.mean(cmp_raw_tensor),
            }

        # 将每个子图的基本信息和候选项比较结果存储在结果字典中
        results_comparison[subgraph_name] = {
            'ref_node_name': subgraph_results['0']['ref_node_name'],
            'ref_node_target_type': subgraph_results['0']['ref_node_target_type'],
            'fqn': subgraph_results['0']['fqn'],
            'candidates': candidates,
        }

    # 返回完整的比较结果字典
    return results_comparison
# TODO(future PR): redesign this to make it easier to consume outputs
# 定义一个打印影子摘要的函数，接受一个比较结果字典作为输入
def print_n_shadows_summary(
    results_comparison,
) -> None:
    """
    Input:

    {
      'subgraph_0': {
        'ref_node_name': 'linear1',
        'ref_node_target_type': '...',
        'fqn': '...',
        'candidates': {
          '1': {
            'qconfig_str': ...,
            'comparison_fn_name': ...,
            'cmp_raw': [45.0, 55.0],
            'cmp_mean': 50.0,
          },
          ...,
        },
      },
    }

    Prints:

    node_name | node_type | fqn | 0    | 1    | ...
    linear1   | ...       | ... | 45.0 | 50.0 | ...
    """

    try:
        # 尝试导入 tabulate 库
        from tabulate import tabulate
    except ImportError:
        # 如果导入失败，打印错误信息并提醒用户安装 tabulate 库
        print("`print_tabular` relies on the library `tabulate`, "
              "which could not be found on this machine. Run `pip "
              "install tabulate` to install the library.")
        return

    # 初始化结果列表
    results = []
    # 遍历比较结果的每个子图数据
    for subgraph_data in results_comparison.values():
        # 提取每个候选节点的平均比较值
        mean_all_candidates = [
            candidate['cmp_mean']
            for candidate_name, candidate in subgraph_data['candidates'].items()
        ]

        # 构建一行数据，包括参考节点名称、节点目标类型、全限定名及所有候选节点的平均值
        data_row = [
            subgraph_data['ref_node_name'],
            subgraph_data['ref_node_target_type'],
            subgraph_data['fqn'],
            *mean_all_candidates,
        ]
        # 将数据行添加到结果列表中
        results.append(data_row)

    # 计算候选索引的最大长度
    max_candidate_idx_len = -1
    for data_row in results:
        max_candidate_idx_len = max(max_candidate_idx_len, len(data_row[1]))
    candidate_idx_headers = [str(x) for x in range(max_candidate_idx_len)]

    # 定义表头，包括节点名称、节点类型、全限定名及所有候选节点的索引
    headers = ['node_name', 'node_type', 'fqn', *candidate_idx_headers]
    # 使用 tabulate 库打印结果表格
    print(tabulate(results, headers=headers))
```