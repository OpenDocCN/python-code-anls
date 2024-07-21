# `.\pytorch\torch\fx\passes\splitter_base.py`

```
# mypy: allow-untyped-defs
# 导入需要的模块和库
import argparse                   # 解析命令行参数的模块
import copy                       # 复制对象的模块
from collections import defaultdict  # 提供了默认值的字典模块
from dataclasses import dataclass  # 创建数据类的装饰器
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple  # 类型提示，用于静态类型检查
import logging                    # 日志记录模块

import torch                      # PyTorch 深度学习库
from torch.fx.passes.graph_manipulation import get_size_of_node  # 获取节点大小的函数
from torch.fx.node import map_arg  # 对节点参数进行映射的函数
from torch.fx._compatibility import compatibility  # 兼容性函数

from .operator_support import (   # 导入自定义模块中的操作符支持相关函数
    get_node_target,              # 获取节点目标的函数
    OperatorSupportBase,          # 操作符支持的基类
)
from .graph_drawer import FxGraphDrawer  # 导入自定义模块中的图形绘制类
from .shape_prop import ShapeProp  # 导入自定义模块中的形状属性类
from .split_utils import split_by_tags  # 导入自定义模块中的按标签拆分函数
from .tools_common import (        # 导入自定义工具模块中的公共工具
    FxNetAccFusionsFinder,         # 导入加速网络融合查找器类
    CALLABLE_NODE_OPS,             # 可调用节点操作列表
    Tensors,                       # 张量类型
    NodeList,                      # 节点列表类型
    NodeSet,                       # 节点集合类型
    is_node_output_tensor,         # 判断节点输出是否为张量的函数
)

__all__ = ['FxNetAccNodesFinder', 'FxNetSplitterInternalError', 'Subgraph', 'SplitResult', 'generate_inputs_for_submodules']
# 将这些类和函数添加到模块的公开接口中

_LOGGER = logging.getLogger(__name__)  # 创建一个 logger 对象，用于记录日志

DEFAULT_MIN_ACC_MODULE_SIZE = 1  # 默认的加速模块最小大小
DEFAULT_SKIP_FUSION = False      # 默认是否跳过融合
DEFAULT_ALLOW_NON_TENSOR = False  # 默认是否允许非张量数据

class _SplitterSettingBase:
    def __init__(
        self,
        min_acc_module_size=DEFAULT_MIN_ACC_MODULE_SIZE,
        skip_fusion=DEFAULT_SKIP_FUSION,
        allow_non_tensor=DEFAULT_ALLOW_NON_TENSOR
    ):
        parser = argparse.ArgumentParser()  # 创建命令行参数解析器对象
        parser.add_argument(
            "--min-acc-module-size",            # 解析最小加速模块大小参数
            "--min_acc_module_size",
            required=False,
            type=int,
            help="Minimum size limit of an accelerator subgraph.",
        )
        parser.add_argument(
            "--skip-fusion",                    # 解析跳过融合参数
            "--skip_fusion",
            default=False,
            action="store_true",
            help="If true then no fusion groups. Fusion group is used to "
            "enforce no non-tensor data flow between submodules. If we don't "
            "have this constrain, setting this to false is recommended as it "
            "can reduce overhead.",
        )
        parser.add_argument(
            "--allow-non-tensor",               # 解析允许非张量数据参数
            "--allow_non_tensor",
            default=False,
            action="store_true",
            help="For some backends non-tensor data flow between cpu and them "
            "are not allowed. Therefore, if a node supported by accelerator but "
            "it has non-tensor inputs or outputs to a cpu node we would want to "
            "consider it as a cpu node during splitting. However, for some backends "
            "we might not care about non-tensor data flow and we can set this option "
            "to true to disable the functionality that prevent non-tensor data flow.",
        )
        args, unknown = parser.parse_known_args()  # 解析命令行参数

        # 设置最小加速模块大小，默认为参数值或默认值
        self.min_acc_module_size: int = args.min_acc_module_size if args.min_acc_module_size else min_acc_module_size
        # 设置是否跳过融合，默认为参数值或默认值
        self.skip_fusion: bool = args.skip_fusion if args.skip_fusion else skip_fusion
        # 设置是否允许非张量数据，默认为参数值或默认值
        self.allow_non_tensor: bool = args.allow_non_tensor if args.allow_non_tensor else allow_non_tensor


@compatibility(is_backward_compatible=False)
class FxNetAccNodesFinder:
    """
    Finds a set of nodes that can be supported on ACC, excluding nodes that have non-tensor
    input/output to cpu nodes to prevent non-tensor data flow between backends and cpu.

    I.e. if we have a chain:

    ACC_NODE_1 -> ACC_NODE_2 -> ACC_NODE_3 -> CPU_NODE_1

    where every ACC node produces non-tensor output, then they all should be treated as CPU nodes.

    This behavior can be turned off by passing allow_non_tensor=True.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        operator_support: OperatorSupportBase,
        allow_non_tensor: bool,
    ):
        """
        Initializes the FxNetAccNodesFinder instance.

        Args:
            module (torch.fx.GraphModule): The PyTorch FX graph module.
            operator_support (OperatorSupportBase): Operator support object.
            allow_non_tensor (bool): Flag indicating whether to allow non-tensor data flow.
        """
        self.module = module
        self.operator_support = operator_support
        self.allow_non_tensor = allow_non_tensor
        self.acc_nodes: NodeSet = set()

    def reduce_acc_nodes_non_tensor_input_helper(
        self, cpu_worklist: NodeList
    ):
        """
        Transitively excludes nodes from ACC supported set based on non-tensor input criteria.

        Args:
            cpu_worklist (NodeList): List of nodes to process for non-tensor data flow.

        Modifies:
            self.acc_nodes: Updates ACC nodes set by excluding nodes based on non-tensor inputs.
        """
        while cpu_worklist:
            node = cpu_worklist.pop(0)

            for user in node.users:
                if user in self.acc_nodes:
                    self.acc_nodes.remove(user)
                    if not is_node_output_tensor(user):
                        cpu_worklist.append(user)

    def reduce_acc_nodes_non_tensor_input(self):
        """
        Excludes nodes from ACC supported set that have direct upstream CPU nodes producing non-tensor outputs.
        """
        non_tensor_cpu_nodes: NodeList = []

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            if node in self.acc_nodes:
                continue
            if is_node_output_tensor(node):
                continue
            non_tensor_cpu_nodes.append(node)

        self.reduce_acc_nodes_non_tensor_input_helper(non_tensor_cpu_nodes)

    def reduce_acc_nodes_non_tensor_output(self):
        """
        Excludes nodes from ACC supported set that produce non-tensor outputs and have downstream CPU nodes.
        """
        while True:
            new_cpu_nodes: NodeList = []

            for acc_node in self.acc_nodes:
                if is_node_output_tensor(acc_node):
                    continue
                for user in acc_node.users:
                    if user not in self.acc_nodes:
                        new_cpu_nodes.append(acc_node)
                        break

            if not new_cpu_nodes:
                break

            for new_cpu_node in new_cpu_nodes:
                self.acc_nodes.remove(new_cpu_node)

            self.reduce_acc_nodes_non_tensor_input_helper(new_cpu_nodes)
    # 定义一个特殊方法 __call__，使对象可以像函数一样被调用，并返回 NodeSet 对象
    def __call__(self) -> NodeSet:
        # 使用 self.module.named_modules() 方法获取模块中所有子模块的字典表示
        submodules = dict(self.module.named_modules())
        
        # 初始化一个空集合 self.acc_nodes，用来存储满足条件的节点
        self.acc_nodes = {
            n
            for n in self.module.graph.nodes
            # 遍历模块图中的节点，选择操作类型在 CALLABLE_NODE_OPS 中，并且被操作支持的节点
            if n.op in CALLABLE_NODE_OPS
            and self.operator_support.is_node_supported(submodules, n)
        }

        # 如果不允许非张量输入，则进行相关处理
        if not self.allow_non_tensor:
            # 调用方法处理非张量输入，减少符合条件的节点
            self.reduce_acc_nodes_non_tensor_input()
            # 调用方法处理非张量输出，减少符合条件的节点
            self.reduce_acc_nodes_non_tensor_output()

        # 返回存储满足条件节点的集合 self.acc_nodes
        return self.acc_nodes
@compatibility(is_backward_compatible=False)
# 定义一个自定义异常类，用于表示FxNetSplitter内部的错误
class FxNetSplitterInternalError(Exception):
    pass

@compatibility(is_backward_compatible=False)
# 用于存储子图信息的数据类，标记是否是累加器，节点列表及设备编号
@dataclass
class Subgraph:
    is_acc: bool
    nodes: NodeList
    device_ordinal: Optional[int] = None

@compatibility(is_backward_compatible=False)
# 表示SplitResult的命名元组，存储了分离器的结果
class SplitResult(NamedTuple):
    """
    存储分离器的结果。

    Attributes:
        split_module: 分离后的根模块。
        submodule_inputs: 将子模块名称映射到其输入的字典。
        non_acc_submodule_prefix: 非累加子模块的前缀。对于累加子模块，前缀始终为"_run_on_acc_"。
    """
    split_module: torch.fx.GraphModule
    submodule_inputs: Dict[str, Any]
    non_acc_submodule_prefix: str


@compatibility(is_backward_compatible=False)
# 生成目标子模块的输入数据的函数
def generate_inputs_for_submodules(
    model: torch.nn.Module,
    inputs: Sequence[Any],
    target_submodules: Iterable[str],
    deepcopy: bool = False,
) -> Dict[str, Any]:
    """
    为给定模型中的目标子模块生成输入数据。注意，如果两个子模块引用相同的对象，此函数将无法正常工作。

    Args:
        model: 根模型。
        inputs: 根模型的输入数据。
        target_submodules: 希望为其生成输入数据的子模块名称。

    Returns:
        映射子模块名称到其输入数据的字典。
    """

    handles = []
    results = {}
    submodule_to_names = {mod: name for name, mod in model.named_modules()}

    def pre_forward(module, module_inputs):
        results[submodule_to_names[module]] = copy.deepcopy(module_inputs) if deepcopy else module_inputs

    # 注册前向钩子，用于收集每个目标子模块的输入数据
    for name, mod in model.named_modules():
        if name in target_submodules:
            handles.append(mod.register_forward_pre_hook(pre_forward))

    # 清理钩子对象的函数
    def clean_up_handles():
        for h in handles:
            h.remove()

    try:
        with torch.no_grad():
            model(*inputs)  # 运行模型前向传播以收集输入数据
    except Exception as e:
        clean_up_handles()
        raise e

    clean_up_handles()
    return results


class _SplitterBase:
    """
    将GraphModule拆分为在CPU或加速器上执行的子GraphModule。
    输出是一个GraphModule，支持和不支持的操作符尽可能少地分组到子GraphModule中。
    假设只有"call_module"、"call_function"和"call_method"来自FX IR可能在加速器上执行。

    给定以下图表：
          ==> b ==>
        //         \\
       a             d
        \\         //
          ==> c ==>

    class SimpleModule(torch.nn.Module):
        def forward(self, a):
            b = torch.sin(a)
            c = torch.cos(a)
            d = b + c
            return d

    并提供"operator_support"指示'b'和'c'可以在加速器上执行时，我们将获得以下拆分结果：

    main:
    def forward(self, a):
        # 调用 _run_on_acc_0_0 方法处理输入 a，返回结果
        run_on_acc_0_0 = self._run_on_acc_0_0(a)
        # 从 run_on_acc_0_0 的结果中获取第一个元素
        getitem = run_on_acc_0_0[0]
        # 从 run_on_acc_0_0 的结果中获取第二个元素
        getitem_1 = run_on_acc_0_0[1]
        # 调用 _run_on_cpu_1_1 方法处理 getitem 和 getitem_1，返回结果
        run_on_cpu_1_1 = self._run_on_cpu_1_1(getitem, getitem_1)
        # 返回 run_on_cpu_1_1 的结果作为 forward 方法的输出
        return run_on_cpu_1_1

    _run_on_acc_0_0:
    def forward(self, a):
        # 计算输入张量 a 的正弦值
        sin_1 = torch.sin(a)
        # 计算输入张量 a 的余弦值
        cos_1 = torch.cos(a)
        # 返回正弦值和余弦值组成的元组
        return (sin_1, cos_1)

    _run_on_cpu_1_1:
    def forward(self, sin_1, cos_1):
        # 将正弦值和余弦值相加
        add_1 = sin_1 + cos_1
        # 返回相加后的结果
        return add_1

"""
    # PCIe bandwidth for the backend, default to 100 GB/s
    # 定义 PCIe 后端的带宽，默认为 100 GB/s
    PCIe_BW = 100 * 2 ** 30

    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Sequence[Any],
        operator_support: OperatorSupportBase,
        settings: _SplitterSettingBase,
        non_acc_submodule_name: str = "_run_on_cpu_",
        return_tuple: bool = False,
    ):
        """
        Preprocesses graph before splitting:
        - finds nodes supported by ACC,
        - finds fusion groups for ACC nodes having non-tensor IO,
        - builds a graph of direct dependencies,
        - builds a map of fused nodes to their fusions.
        As a result we get self.acc_nodes, self.deps and self.fusions.
        """
        assert isinstance(module, torch.fx.GraphModule)

        # 初始化函数，对图模块进行预处理：
        # - 查找由 ACC 支持的节点
        # - 查找具有非张量 IO 的 ACC 节点的融合组
        # - 构建直接依赖关系图
        # - 构建融合节点到其融合的映射
        # 最终得到 self.acc_nodes, self.deps 和 self.fusions
        self.module = module
        ShapeProp(self.module).propagate(*sample_input)

        self.settings = settings
        self.operator_support = operator_support
        self.sample_input = sample_input
        # 使用 FxNetAccNodesFinder 类查找 ACC 节点
        self.acc_nodes = FxNetAccNodesFinder(self.module, self.operator_support, self.settings.allow_non_tensor)()

        if self.settings.skip_fusion:
            self.fusions = {}
        else:
            # 使用 FxNetAccFusionsFinder 类查找 ACC 节点的融合组
            self.fusions = FxNetAccFusionsFinder(module, self.acc_nodes)()

        # 修改 deps 来添加更多融合节点的依赖关系
        self.deps = self.find_deps()
        self.update_deps_for_fusions()

        self.non_acc_submodule_name = non_acc_submodule_name
        self._node_submodule_map: Dict[str, str] = {}
        self._return_tuple = return_tuple

        self.tags: List[str] = []

    # ===============================================================
    # Helpers for ctor and initial state
    # ===============================================================

    def get_node_submodule_map(self) -> Dict[str, str]:
        """
        Returns a map from node name to submodule name, e.g.
        node: main_module_impl_impl_over_arch_unary_multiple_embedding
              _pooling_embedding_pooling_sparse_entity_equivalence_key
              _proxy_embedding_bag
        maps to submodule name of: _run_on_acc_1
        """
        # 返回一个从节点名称映射到子模块名称的字典
        return self._node_submodule_map
    def find_deps(self) -> Dict[torch.fx.Node, NodeSet]:
        """
        Builds a graph of node dependencies. Leaf nodes don't have any
        dependencies and the "output" node doesn't have nodes depending on it.

        Resulting graph has only direct dependencies, i.e. there are no
        transitive dependencies.
        """
        # 初始化一个空的字典，默认值是空集合，用于存储节点和它们的依赖关系
        deps: Dict[torch.fx.Node, NodeSet] = defaultdict(set)
        
        # 遍历模块的图中的每个节点
        for node in self.module.graph.nodes:
            # 如果节点操作不在可调用节点操作列表中，则跳过
            if node.op not in CALLABLE_NODE_OPS:
                continue
            
            # 遍历当前节点的用户
            for user in node.users:
                # 如果用户节点的操作不是 "output"，则将当前节点添加到用户节点的依赖集合中
                if user.op != "output":
                    deps[user].add(node)
        
        # 返回构建好的节点依赖关系图
        return deps

    def update_deps_for_fusions(self):
        """
        Updates graph of dependencies so that:
        - nodes from the same fusion depend on the same set of outer nodes,
        - outer nodes depending on a fusion depend on all nodes in that fusion.
        """
        # 遍历每个融合节点
        for node in self.fusions:
            # 获取当前融合节点的融合集合
            fusion = self.fusions[node]
            
            # 遍历融合集合中的每个融合邻居节点
            for fused_neighbor in fusion:
                # 更新当前节点的依赖集合，使其包含融合邻居节点依赖集合与融合集合的差集
                self.deps[node].update(self.deps[fused_neighbor] - fusion)

                # 遍历融合邻居节点的用户节点
                for user in fused_neighbor.users:
                    # 如果用户节点不在当前融合集合中，则将当前节点添加到用户节点的依赖集合中
                    if user not in fusion:
                        self.deps[user].add(node)

    # ===============================================================
    # Helpers for preview
    # ===============================================================

    def _lower_model_to_backend(
        self, mod: torch.fx.GraphModule, inputs: Tensors
    ) -> torch.nn.Module:
        """
        Lower the model to a backend.
        """
        # 返回未修改的模型对象
        return mod

    def _find_culprit(
        self, mod: torch.fx.GraphModule, inputs: Tensors
    ) -> str:
        """
        When an error occurs during lowering or running the lowered mod, we use this
        function to find culprits in the `mod` that causes the error.
        """
        # 返回一个字符串，指示未实现该函数来找到问题源头
        return "Unable to find a culprit because _find_culprit() function is not implemented."

    def _draw_graph_based_on_node_support(
        self, mod: torch.fx.GraphModule, supported_nodes: NodeList
    ):
        # 定义节点样式的颜色映射
        color_map = {
            "default": "AliceBlue",
            "supported": "chartreuse1",
            "unsupported": "crimson",
        }

        # 定义一个自定义的图形绘制类
        class CustomDrawer(FxGraphDrawer):
            def _get_node_style(self, node):
                # 调用父类方法获取节点样式模板
                template = super()._get_node_style(node)
                # 根据节点是否在支持节点列表中设置填充颜色
                if node in supported_nodes:
                    template["fillcolor"] = color_map["supported"]
                elif node.op in CALLABLE_NODE_OPS:
                    template["fillcolor"] = color_map["unsupported"]
                else:
                    template["fillcolor"] = color_map["default"]

                return template

        # 创建自定义绘图对象
        drawer = CustomDrawer(mod, "node_support", ignore_getattr=True)
        # 获取主要的 Dot 图形
        dot_graph = drawer.get_main_dot_graph()
        # 将 Dot 图形写入文件 "node_support.dot" 中
        dot_graph.write_raw("node_support.dot")
    # 定义一个方法，用于检查并预览节点的支持情况，可以选择是否输出图形
    def node_support_preview(self, dump_graph: bool = False):
        # 获取模块中所有子模块的字典表示
        submodules = dict(self.module.named_modules())

        # 初始化支持的节点列表和支持的节点类型集合
        supported_nodes: NodeList = []
        supported_node_types = defaultdict(set)
        unsupported_node_types = defaultdict(set)

        # 定义一个函数，用于获取节点参数的数据类型
        def get_dtype(arg):
            tensor_meta = arg.meta.get("tensor_meta")
            return getattr(tensor_meta, "dtype", None)

        # 遍历模块图中的每个节点
        for node in self.module.graph.nodes:
            # 如果节点操作不在可调用节点操作列表中，则跳过
            if node.op not in CALLABLE_NODE_OPS:
                continue

            # 获取节点的目标对象
            target = get_node_target(submodules, node)

            # 存储节点参数中参数的数据类型。如果参数不是张量，则数据类型为 None
            arg_dtypes = [
                get_dtype(arg) if isinstance(arg, torch.fx.Node) else None
                for arg in node.args
            ]

            # 查找最后一个非 None 元素的索引，如果所有元素都是 None，则返回 max_len
            last_index = len(arg_dtypes) - next(
                (
                    i
                    for i, dtype in enumerate(reversed(arg_dtypes))
                    if dtype is not None
                ),
                len(arg_dtypes),
            )

            # 去除末尾的 None 元素
            arg_dtypes_tuple = tuple(arg_dtypes[:last_index])

            # 获取关键字参数的数据类型并构成元组
            kwarg_dtypes_tuple = tuple(
                (k, get_dtype(arg))
                for k, arg in node.kwargs.items()
                if isinstance(arg, torch.fx.Node)
            )

            # 如果操作支持当前节点
            if self.operator_support.is_node_supported(submodules, node):
                # 将节点添加到支持的节点列表中
                supported_nodes.append(node)
                # 将节点类型及其参数数据类型添加到支持的节点类型字典中
                supported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))
            else:
                # 将节点类型及其参数数据类型添加到不支持的节点类型字典中
                unsupported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))

        # 如果需要输出图形，则基于支持的节点绘制模块图
        if dump_graph:
            self._draw_graph_based_on_node_support(self.module, supported_nodes)

        # 构建支持和不支持节点类型的报告字符串
        reports = "\nSupported node types in the model:\n"
        for t, dtypes in supported_node_types.items():
            for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
                reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

        reports += "\nUnsupported node types in the model:\n"
        for t, dtypes in unsupported_node_types.items():
            for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
                reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

        # 打印报告字符串
        print(reports)

        # 返回报告字符串，用于测试目的
        return reports

    # ===============================================================
    # extend_acc_subgraph() 方法的辅助函数
    # ===============================================================

    # 定义一个方法，用于查找特定标签ID的反向依赖
    def find_reverse_deps(
        self, tag_id: Optional[int] = None
    ) -> Dict[torch.fx.Node, NodeSet]:
        """
        Builds reversed topological node dependencies, if tag_id is specified,
        we ignore nodes that are in later subgraph i.e. nodes have greater tag_id.
        """
        # 初始化一个空的字典，默认值为集合，用于存储节点到其依赖节点集合的映射关系
        result: Dict[torch.fx.Node, NodeSet] = defaultdict(set)

        # 遍历模块的图中的每个节点
        for node in self.module.graph.nodes:
            # 如果节点的操作类型不在可调用节点操作列表中，跳过该节点
            if node.op not in CALLABLE_NODE_OPS:
                continue

            # 遍历每个节点的用户（即依赖于该节点的节点）
            for user in node.users:
                # 如果用户节点的操作类型不在可调用节点操作列表中，跳过该用户节点
                if user.op not in CALLABLE_NODE_OPS:
                    continue

                # 如果未指定tag_id或者用户节点的tag_id小于给定的tag_id，将用户节点添加到当前节点的依赖集合中
                if tag_id is None or (int(user.tag.split("_")[-1]) < tag_id):
                    result[node].add(user)

        # 返回构建好的节点依赖关系字典
        return result

    def update_reverse_deps_for_fusions(
        self, deps: Dict[torch.fx.Node, NodeSet]
    ):
        processed_node = set()

        # 遍历每个融合组中的节点
        for node, fusion in self.fusions.items():
            # 如果节点已经处理过，跳过该节点
            if node in processed_node:
                continue

            new_dep = set()

            # 创建一个新的依赖集合，包括所有融合组中节点的依赖节点
            for n in fusion:
                new_dep.update(deps[n])

            # 排除融合组中的节点自身
            new_dep.difference_update(fusion)

            # 更新依赖关系
            for n in fusion:
                deps[n] = new_dep

                # 更新每个输入节点的依赖关系，将不在融合组中的节点加入依赖关系集合
                for arg in n.all_input_nodes:
                    if arg not in fusion:
                        deps[arg].update(fusion)

                # 将当前节点标记为已处理
                processed_node.add(n)

    def find_parent_nodes_of_subgraph(self, tag: str) -> NodeSet:
        """
        Finds parent nodes of the `tag` subgraph.

        Traverse the inputs of nodes in the subgraph, if input doesn't belong to the subgraph
        and is not a placeholder, we consider it as the parent node of the subgraph.
        """
        # 初始化一个空集合，用于存储子图的父节点
        parent_nodes = set()

        # 遍历模块的图中的每个节点
        for node in self.module.graph.nodes:
            # 如果节点的操作类型在可调用节点操作列表中且节点的标签与给定的tag相符
            if node.op in CALLABLE_NODE_OPS and node.tag == tag:
                # 遍历节点的所有输入节点
                for arg in node.all_input_nodes:
                    # 如果输入节点的操作类型在可调用节点操作列表中且输入节点的标签与给定的tag不相符
                    if arg.op in CALLABLE_NODE_OPS and arg.tag != tag:
                        # 将输入节点视为子图的父节点
                        parent_nodes.add(arg)

        # 返回找到的父节点集合
        return parent_nodes
    # 扩展 acc 子图，以 `tag` 为起点进行反向拓扑方向扩展
    def extend_acc_subgraph(self, tag: str):
        """
        Extend the acc subgraph with `tag` going the reversed topological direction.
        """
        # 创建一个字典，将节点映射到其用户，并忽略那些位于具有更大标签的子图中的用户
        deps = self.find_reverse_deps(tag_id=int(tag.split("_")[-1]))
        # 更新融合节点的反向依赖关系
        self.update_reverse_deps_for_fusions(deps)

        # 查找子图的父节点
        parent_nodes = self.find_parent_nodes_of_subgraph(tag)

        visited_nodes: NodeSet = set()

        while parent_nodes:
            node = None

            # 查找一个 acc 节点，它仅依赖于已访问过的节点
            for n in parent_nodes:
                if deps[n] <= visited_nodes and n in self.acc_nodes:
                    node = n
                    break

            if node is None:
                break

            # 将节点放入 `tag` 子图中
            node.tag = tag  # type: ignore[attr-defined]
            parent_nodes.remove(node)
            visited_nodes.add(node)

            # 如果节点位于融合组中，将所有融合伙伴添加到父节点中
            if node in self.fusions:
                for fusion_node in self.fusions[node]:
                    if fusion_node not in visited_nodes:
                        parent_nodes.add(fusion_node)

            # 将节点的输入添加到父节点中
            for arg in node.all_input_nodes:
                if arg.op in CALLABLE_NODE_OPS and arg not in visited_nodes:
                    parent_nodes.add(arg)

    # ===============================================================
    # split() 方法的辅助函数
    # ===============================================================

    def starter_nodes(self) -> Tuple[NodeSet, NodeSet]:
        """
        Finds nodes that consume module inputs or get_attr nodes.
        """
        starter_cpu_nodes: NodeSet = set()
        starter_acc_nodes: NodeSet = set()
        for node in self.module.graph.nodes:
            if node.op not in {"placeholder", "get_attr"}:
                continue
            for user in node.users:
                if user in self.acc_nodes:
                    starter_acc_nodes.add(user)
                else:
                    starter_cpu_nodes.add(user)
        return starter_cpu_nodes, starter_acc_nodes
    def put_nodes_into_subgraphs(self) -> List[Subgraph]:
        # 从叶节点开始进行图遍历
        current_cpu_nodes, current_acc_nodes = self.starter_nodes()
        # 记录已访问过的节点集合
        visited_nodes: NodeSet = set()

        # 根据当前 CPU 和加速器节点中是否存在没有依赖的节点来确定起始的子图
        acc_subgraph: bool = not any(len(self.deps[n]) == 0 for n in current_cpu_nodes)

        # 当前子图中的节点列表
        current_subgraph_nodes: NodeList = []

        # 结果累加器，存放最终的子图列表
        subgraphs: List[Subgraph] = []
        while current_cpu_nodes or current_acc_nodes:
            # 选择当前子图中的节点，优先选择加速器节点或者 CPU 节点
            current_nodes = current_acc_nodes if acc_subgraph else current_cpu_nodes
            # 找到第一个应属于当前子图且所有依赖都已解析的节点
            node = next(
                (n for n in current_nodes if self.deps[n] <= visited_nodes),
                None,
            )

            # 如果找不到符合条件的节点，则需要切换子图模式，并开始新的子图
            if node is None:
                if not current_subgraph_nodes:
                    # 如果当前子图为空，则抛出异常
                    raise FxNetSplitterInternalError("Subgraph can't be empty")

                # 将当前子图添加到子图列表中
                subgraphs.append(
                    Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes)
                )
                # 切换子图模式
                acc_subgraph = not acc_subgraph
                current_subgraph_nodes = []
                continue

            # 将节点从当前节点集合中移除，并标记为已访问
            current_nodes.remove(node)
            visited_nodes.add(node)
            # 将节点添加到当前子图的节点列表中
            current_subgraph_nodes.append(node)

            # 添加融合伙伴节点
            if node in self.fusions:
                if node in self.acc_nodes:
                    # 如果节点是加速器节点，则将其融合伙伴节点添加到加速器节点集合中
                    current_acc_nodes.update(self.fusions[node] - visited_nodes)
                else:
                    # 如果节点是 CPU 节点，则将其融合伙伴节点添加到 CPU 节点集合中
                    current_cpu_nodes.update(self.fusions[node] - visited_nodes)

            # 将依赖于当前节点的节点添加到相应的队列中
            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue

                # 将依赖节点加入到相应的队列中
                if user in self.acc_nodes:
                    current_acc_nodes.add(user)
                else:
                    current_cpu_nodes.add(user)

        # 检查是否最后一个子图未创建
        if current_subgraph_nodes:
            subgraphs.append(
                Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes)
            )

        # 如果没有创建任何子图，则抛出异常
        if not subgraphs:
            raise FxNetSplitterInternalError("Couldn't create subgraphs")

        # 返回最终的子图列表
        return subgraphs
    def remove_small_acc_subgraphs(self, subgraphs: List[Subgraph]) -> List[Subgraph]:
        """
        This pass finds ACC submodules with less than specified size and merges
        them with adjacent CPU submodules.
        """
        # 初始化空的结果列表
        result: List[Subgraph] = []
        # 遍历每个子图
        for subgraph in subgraphs:
            # 如果子图是 ACC 模块
            if subgraph.is_acc:
                # 如果 ACC 模块的节点数大于等于最小 ACC 模块大小阈值
                if len(subgraph.nodes) >= self.settings.min_acc_module_size:
                    # 将该子图添加到结果列表中
                    result.append(subgraph)
                else:
                    # 打印信息，说明该 ACC 子图小于阈值，将被消除
                    print(
                        "Eliminating acc subgraph because it's smaller than the threshold: "
                        f"{len(subgraph.nodes)} < {self.settings.min_acc_module_size}"
                    )
                    # 如果结果列表不为空，将当前 ACC 子图的节点合并到上一个子图中
                    if result:
                        result[-1].nodes.extend(subgraph.nodes)
                    else:
                        # 否则，将当前 ACC 子图标记为非 ACC，然后加入结果列表
                        subgraph.is_acc = False
                        result.append(subgraph)
            else:
                # 如果当前子图不是 ACC，并且结果列表的最后一个子图也不是 ACC，将当前子图节点合并到上一个子图中
                if result and not result[-1].is_acc:
                    result[-1].nodes.extend(subgraph.nodes)
                else:
                    # 否则，将当前子图直接添加到结果列表中
                    result.append(subgraph)
        # 返回处理后的结果列表
        return result

    def tag(self, subgraphs: List[Subgraph]):
        # 清空当前标签列表
        self.tags = []
        # 遍历每个子图
        for subgraph in subgraphs:
            # 根据是否为 ACC 子图确定标签名
            tag = f"_run_on_acc_{len(self.tags)}" if subgraph.is_acc else f"{self.non_acc_submodule_name}{len(self.tags)}"
            # 将标签添加到标签列表中
            self.tags.append(tag)
            # 为子图中的每个节点设置标签
            for node in subgraph.nodes:
                # 如果节点已经有标签，抛出异常
                if hasattr(node, "tag"):
                    raise FxNetSplitterInternalError(f"Node {node} was already tagged")
                # 否则，将标签赋予节点
                node.tag = tag  # type: ignore[attr-defined]
                # 更新节点到子模块映射表
                self._node_submodule_map[node.name] = tag

    def split(self, remove_tag: bool = False) -> torch.fx.GraphModule:
        # 根据标签对模块进行拆分
        split_module = split_by_tags(self.module, self.tags, return_tuple=self._return_tuple)
        # 如果需要移除标签
        if remove_tag:
            # 遍历模块图中的每个节点
            for node in self.module.graph.nodes:
                # 如果节点有标签属性，删除标签
                if hasattr(node, "tag"):
                    del node.tag
        # 返回拆分后的模块
        return split_module

    def __call__(self) -> torch.fx.GraphModule:
        # 将节点放入子图中
        subgraphs = self.put_nodes_into_subgraphs()
        # 移除小于阈值的 ACC 子图
        subgraphs = self.remove_small_acc_subgraphs(subgraphs)
        # 统计 ACC 和非 ACC 子图的数量
        acc_subgraphs_count = len([s for s in subgraphs if s.is_acc])
        non_acc_subgraphs_count = len(subgraphs) - acc_subgraphs_count
        # 打印 ACC 和非 ACC 子图的数量
        print(f"Got {acc_subgraphs_count} acc subgraphs and {non_acc_subgraphs_count} non-acc subgraphs")
        # 为子图添加标签
        self.tag(subgraphs)
        # 执行模块拆分并返回结果
        return self.split()

    def generate_split_results(self) -> SplitResult:
        # 生成拆分结果
        split_module = self()
        # 收集拆分后的子模块名称
        submodule_names = []
        # 遍历拆分后的模块的子模块
        for name, mod in split_module.named_children():
            submodule_names.append(name)
        # 为拆分后的子模块生成输入数据
        submodule_inputs = generate_inputs_for_submodules(split_module, self.sample_input, submodule_names)
        # 返回拆分结果对象
        return SplitResult(split_module, submodule_inputs, self.non_acc_submodule_name)
```