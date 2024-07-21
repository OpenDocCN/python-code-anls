# `.\pytorch\torch\fx\passes\net_min_base.py`

```py
# 设置类型提示中允许未标记的函数定义
# 导入日志模块
import logging
# 导入用于数据类的装饰器
from dataclasses import dataclass
# 导入类型提示相关的对象
from typing import Any, Callable, Dict, List, Optional, Tuple

# 导入PyTorch相关模块
import torch
import torch.fx

# 导入兼容性模块
from torch.fx._compatibility import compatibility
# 导入用于映射参数的函数
from torch.fx.node import map_arg

# 导入本地模块
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
    CALLABLE_NODE_OPS,
    FxNetAccFusionsFinder,
    Names,
    NodeList,
    NodeSet,
    TensorOrTensors,
    Tensors,
)

# 所有对外公开的模块列表
__all__ = [
    "FxNetMinimizerBadModuleError",
    "FxNetMinimizerRunFuncError",
    "FxNetMinimizerResultMismatchError",
]

# 获取当前模块的日志记录器
_LOGGER = logging.getLogger(__name__)


@compatibility(is_backward_compatible=False)
class FxNetMinimizerBadModuleError(Exception):
    """
    如果无法分离出最小化模块，则引发此异常。
    """
    pass


@compatibility(is_backward_compatible=False)
class FxNetMinimizerRunFuncError(Exception):
    """
    在运行 run_a 或 run_b 函数期间发生错误时引发此异常。
    """
    pass


@compatibility(is_backward_compatible=False)
class FxNetMinimizerResultMismatchError(Exception):
    """
    如果比较函数认为结果不匹配，则引发此异常。
    """
    pass


@dataclass
class _MinimizerSettingBase:
    """
    FX模块的最小化器设置类。

    Args:
    `accumulate_error`: 是否使用每个转换模块的先前输出累积错误，而不是使用a的输入来验证。

    `traverse_method`: "sequential" 或 "binary" 或 "accumulate"，确定遍历FX模块中节点的方式。

    `find_all`: 是否遍历整个模型并返回所有有问题的节点。

    `return_intermediate`: 如果为True，在使用 `run_nodes()` 函数运行模型时，将返回所有操作的中间结果作为输出。
    """

    accumulate_error: bool = False
    traverse_method: str = "sequential"
    find_all: bool = False
    return_intermediate: bool = False

    def __str__(self):
        settings_str = "FX Minimizer Settings:\n"

        for k, v in vars(self).items():
            settings_str += f"\t{k}: {v}\n"

        return settings_str


class _MinimizerBase:
    """
    该类用于自动查找模型中的问题节点。它接受一个FX图模块并在遍历图时生成一些子模块。
    然后会使用 `run_a` 和 `run_b` 函数运行相同的子模块，并使用 `compare_fn` 函数比较结果。

    目前我们提供两种遍历图和生成子模块的方式：
        1. 顺序遍历：这将逐节点遍历图，并生成一个包含单个节点的子模块。
        2. 二进制搜索：这将以二进制搜索风格遍历图。

    对于内部用户，可以在 https://fb.quip.com/HDtuAgiKGfkP 找到指南。
    """
    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tensors,
        compare_fn: Callable[
            [TensorOrTensors, TensorOrTensors, Names], Tuple[float, bool]
        ],
        settings: _MinimizerSettingBase,
        module_exporter: Optional[
            Callable[
                [Tensors, torch.fx.GraphModule, str],
                None
            ]
        ] = None,
        exclusion_fn: Optional[
            Callable[[NodeList, int, int], None]
        ] = None,
    ):
        # 确保传入的 module 是 torch.fx.GraphModule 类型
        assert isinstance(module, torch.fx.GraphModule)

        # 初始化各种成员变量
        self.module = module  # 存储传入的 module
        self.sample_input = sample_input  # 存储样本输入数据
        self.compare_fn = compare_fn  # 存储用于比较输出的函数
        self.module_exporter = module_exporter  # 可选参数，用于导出模型
        self.settings = settings  # 存储最小化器的设置
        self.exclusion_fn = exclusion_fn  # 可选参数，用于排除特定节点的函数

        # 存储 run_a 函数的输出结果
        self.a_outputs: Dict[str, Any] = {}

        # 存储 run_b 函数的输出结果
        self.b_outputs: Dict[str, Any] = {}

        # 存储 compare_fn 函数的结果
        self.results: Dict[Any, Any] = {}

        # 存储运行报告的列表
        self.reports: List[List[str]] = []

        # 当前迭代次数
        self.iteration: int = 0

        # 获取可调用节点集合
        callable_nodes = {
            node for node in self.module.graph.nodes if node.op in CALLABLE_NODE_OPS
        }
        # 使用 ShapeProp 对象传播样本输入的形状信息
        ShapeProp(self.module).propagate(*self.sample_input)

        # 查找并保存模型中的融合操作
        self.fusions = FxNetAccFusionsFinder(self.module, callable_nodes)()

        # 检查 sample_input 中的输入数量是否与占位符节点数量匹配
        placeholders = [
            node.name for node in self.module.graph.nodes if node.op == "placeholder"
        ]
        assert len(placeholders) == len(self.sample_input)

        # 将样本输入存储到 a_outputs 和 b_outputs 中
        for i, name in enumerate(placeholders):
            self.a_outputs[name] = sample_input[i]
            self.b_outputs[name] = sample_input[i]

    def run_a(self, mod: torch.fx.GraphModule, inputs: Tensors, report_idx: int = -1) -> TensorOrTensors:
        """
        运行 `mod` 和 `inputs`，生成输出。输出将与 run_b() 的输出进行比较。
        """
        raise RuntimeError("run_a() is not implemented.")

    def run_b(self, mod: torch.fx.GraphModule, inputs: Tensors, report_idx: int = -1) -> TensorOrTensors:
        """
        运行 `mod` 和 `inputs`，生成输出。输出将与 run_a() 的输出进行比较。
        """
        raise RuntimeError("run_b() is not implemented.")

    def _store_outputs(
        self,
        a_result: TensorOrTensors,
        b_result: TensorOrTensors,
        submodule: torch.fx.GraphModule,
        report_idx: int,
    ):
        """
        存储 run_a() 和 run_b() 的输出结果，并生成报告。
        """
    ):
        """
        Store the outputs of self.run_a() and self.run_b() into self.a_outputs and
        self.b_outputs, so that we can use them when execute preceding nodes that
        use those outputs as inputs.

        Args:
            a_result: Output of self.run_a(). Could be a tensor or tensors.
            b_result: Output of self.run_b(). Could be a tensor or tensors.
            submodule: The module that generates a_result and b_result.
        """
        output_node = next(
            node for node in submodule.graph.nodes if node.op == "output"
        )

        # Only one output
        if isinstance(output_node.args[0], torch.fx.Node):
            # Store the output of self.run_a() with the corresponding node name
            self.a_outputs[output_node.args[0].name] = a_result
            # Store the output of self.run_b() with the corresponding node name
            self.b_outputs[output_node.args[0].name] = b_result
        # Multiple outputs
        else:
            # Iterate through each output node argument and store results accordingly
            for i, arg in enumerate(output_node.args[0]):
                self.a_outputs[arg.name] = a_result[i]
                self.b_outputs[arg.name] = b_result[i]

    def _get_submod_inputs(
        self, main_module: torch.fx.GraphModule, submod_path: str
    ) -> Tuple[Tensors, Tensors]:
        """
        Try get submodule inputs from stored outputs. If not found then use
        torch_glow.get_submod_inputs to get the inputs.

        If accumulate_error is False, use a_input for run_a() and run_b()
        otherwise use a_input for run_a and b_input for run_b.

        Args:
            main_module: Top-levlel fx module.
            submod_path: Path to the submodule we want to run and compare results.

        Returns:
            a_input: List of tensor(s) that will be used by run_a() as submodule inputs.
            b_input: List of tensor(s) that will be used by run_b() as submodule inputs.
        """
        a_input = []
        b_input = []
        submodule = getattr(main_module, submod_path)
        placeholders = [
            node.name for node in submodule.graph.nodes if node.op == "placeholder"
        ]

        # If all placeholders can be found in stored outputs, use stored
        # outputs as inputs. Otherwise, use `torch_glow.get_submod_inputs`
        # to get the inputs.
        if set(placeholders) <= self.a_outputs.keys():
            # Retrieve stored outputs as inputs from self.a_outputs and self.b_outputs
            for name in placeholders:
                a_input.append(self.a_outputs[name])
                b_input.append(self.b_outputs[name])
        else:
            if self.settings.accumulate_error:
                # Print a warning if stored outputs for placeholders are not found
                print(f"Can't find previous stored outputs named {placeholders}!")

            def get_inputs(self: torch.nn.Module, inputs: Any):
                nonlocal a_input
                # Capture inputs using a forward hook to submodule
                a_input = inputs

            # Use forward hook to capture inputs to the submodule
            handle = submodule.register_forward_pre_hook(get_inputs)
            # Execute main_module with sample_input to trigger forward hook
            main_module(*self.sample_input)
            handle.remove()

            # If accumulate_error is True, use a_input for run_a and b_input for run_b
            b_input = a_input

        if not self.settings.accumulate_error:
            # If accumulate_error is False, return a_input for both run_a and run_b
            return a_input, a_input

        # Otherwise, return a_input for run_a and b_input for run_b
        return a_input, b_input
    # 标记选定的节点为 "minimize" 标签。具有相同标签的节点将被分到同一个子模块中。
    def _tag_nodes(self, selected_nodes: NodeSet):
        """
        Tag selected nodes with tag "minimize". Nodes with the same tags will
        be split to the same submodule afterwards.

        Args:
            selected_nodes: Nodes that we want to minimize. We will tag those nodes
                with "minimize", all preceding nodes with "main_0" and all following
                nodes with "main_1".
        """
        # 遍历整个图的节点
        for node in self.module.graph.nodes:
            # 如果节点的操作不在可调用节点操作列表中，则跳过
            if node.op not in CALLABLE_NODE_OPS:
                continue

            # 如果节点在选定节点集合中，则将其标记为 "minimize"
            if node in selected_nodes:
                node.tag = "minimize"
            # 否则，如果节点的所有输入节点中包含任何一个标记为 {"minimize", "main_1"} 的节点
            # 并且该输入节点是可调用节点操作，则将当前节点标记为 "main_1"
            elif any(
                n.tag in {"minimize", "main_1"}
                for n in node.all_input_nodes
                if n.op in CALLABLE_NODE_OPS
            ):
                node.tag = "main_1"
            # 否则将当前节点标记为 "main_0"
            else:
                node.tag = "main_0"

    # 构建子模块，该子模块只包含指定的节点集合
    def _build_submodule(self, nodes: NodeSet) -> Tuple[torch.fx.GraphModule, str]:
        """
        Split self.module so that one submodule consists of `nodes` and only `nodes`.

        Args:
            nodes: Nodes that we want to include in the minimize submodule.

        Returns:
            split_module (torch.fx.GraphModule): the module after split.
            submodule_name (str): the name of the submodule that consists of `nodes`.
        """
        # 对提供的节点进行标记
        self._tag_nodes(nodes)

        # 根据节点的标记将模块分割成多个子模块
        split_module = split_by_tags(self.module, ["main_0", "minimize", "main_1"])

        # 查找包含带有标记节点的子模块
        submodule_name: str = ""
        for child_name, _ in split_module.named_children():
            # 跳过我们当前不感兴趣的子模块
            if "minimize" not in child_name:
                continue

            # 如果尚未找到包含标记节点的子模块，则将其记录下来
            if submodule_name == "":
                submodule_name = child_name
            else:
                # 如果找到多个包含标记节点的子模块，则抛出异常
                raise FxNetMinimizerBadModuleError(
                    f"Expected only one minimize submodule with nodes {nodes}"
                )

        # 如果未找到包含标记节点的子模块，则抛出异常
        if submodule_name == "":
            raise FxNetMinimizerBadModuleError(
                f"Minimize submodule was not found with nodes {nodes}"
            )

        return split_module, submodule_name

    def _run_and_compare(
        self,
        split_module: torch.fx.GraphModule,
        submod_name: str,
        output_names: Names,
        report_idx: int = -1
        # 继续填写下一部分代码的注释
    ):
        """
        Run the submodule in `split_module` that has name `submod_name`
        using `self.run_a` and `self.run_b` and compare their results.

        Args:
            split_module: Main module that contains the minimize submodule.
            submod_name: Name of the minimize submodule.
            output_names: Names of the node we want to output. If None, we
                will use the original output.
        """
        # 获取指定名称的子模块对象
        submodule = getattr(split_module, submod_name)
        # 获取子模块的输入
        a_input, b_input = self._get_submod_inputs(split_module, submod_name)

        # 如果报告列表为空，则初始化一个空列表并设置迭代次数为1
        if len(self.reports) == 0:
            self.reports.append([])
            self.iteration = 1

        # 获取当前报告列表中的指定索引位置的报告，如果索引为负数则使用迭代次数减1
        report = self.reports[report_idx if report_idx >= 0 else self.iteration - 1]
        # 向报告列表中添加一条运行和比较的描述信息
        report.append("Run and compare ...")

        # 如果有指定输出节点的名称
        if output_names:
            # 初始化存储输出节点的列表
            output_nodes: NodeList = []
            # 遍历子模块的图中的节点
            for node in submodule.graph.nodes:
                # 如果节点操作为 "output"，则擦除该节点
                if node.op == "output":
                    submodule.graph.erase_node(node)

                # 如果节点名称在输出节点名称列表中，则将节点添加到输出节点列表中
                if node.name in output_names:
                    output_nodes.append(node)

            # 设置子模块的输出节点
            submodule.graph.output(
                output_nodes[0] if len(output_nodes) == 1 else tuple(output_nodes)
            )
            # 对子模块的图进行静态检查
            submodule.graph.lint()
            # 重新编译子模块
            submodule.recompile()

        # 使用输出节点参数的名称作为存储比较结果的键名
        for node in submodule.graph.nodes:
            if node.op == "output":
                result_key = map_arg(node.args, lambda x: x.name)

        try:
            # 运行 self.run_a 函数，并获取结果
            a_result = self.run_a(submodule, a_input, report_idx)
            # 运行 self.run_b 函数，并获取结果
            b_result = self.run_b(submodule, b_input, report_idx)
            # 存储输出结果到对应的子模块中
            self._store_outputs(a_result, b_result, submodule)
        except Exception as e:
            # 如果出现异常，则记录异常信息到报告中并抛出异常
            report.append(f"Exception raised when running {submod_name}: {e}")
            raise FxNetMinimizerRunFuncError(  # noqa: B904
                f"Exception raised when running {submod_name}: {e}"
            )

        # 比较结果
        names: Names = output_names
        if output_names is None:
            # 如果输出节点名称为空，则将结果键名转换为字符串列表
            names = [str(v) for v in result_key]  # type: ignore[possibly-undefined]

        # 使用比较函数对结果进行比较，得到数值结果和布尔结果
        numeric_result, bool_result = self.compare_fn(a_result, b_result, names)

        # 将数值结果存储到 self.results 中，使用结果键名作为键
        self.results[result_key] = numeric_result  # type: ignore[possibly-undefined]
        # 将数值准确性信息添加到报告中
        report.append(f"Numerical accuracy = {numeric_result}")
        # 如果布尔结果为 False，则记录结果不匹配的信息，并根据需要导出模块
        if not bool_result:
            report.append(f"Result mismatch for {result_key}")
            if self.module_exporter:
                # 导出模块到指定文件名
                self.module_exporter(
                    a_input, submodule, str(result_key[0]) + "_cpu",
                )
                self.module_exporter(
                    b_input, submodule, str(result_key[0]) + "_acc",
                )
            # 抛出结果不匹配错误异常
            raise FxNetMinimizerResultMismatchError(f"Result mismatch for {result_key}")

    def _binary_search_impl(
        self, all_nodes: NodeList, start_idx: int, end_idx: int
    ) -> NodeSet:
        """
        递归二分搜索的实现。
        """
        culprits: NodeSet = set()  # 初始化一个空的 NodeSet 用于存储问题节点
        nodes: NodeList = all_nodes[start_idx:end_idx]  # 获取从 start_idx 到 end_idx 之间的节点列表

        report: List[str] = []  # 初始化一个空的报告列表

        if self.exclusion_fn is not None:
            # 如果有排除函数，则调用该函数排除节点
            self.exclusion_fn(nodes, start_idx, end_idx)
            if len(nodes) == 0:
                report = ["用户排除了所有节点"]
                self.reports.append(report)
                return culprits  # 如果所有节点都被用户排除，则返回空的 culprits 集合

        first_node_name = nodes[0].name  # 获取第一个节点的名称
        output_node_name = nodes[-1].name  # 获取最后一个节点的名称
        self.iteration += 1  # 迭代次数加一
        self.reports.append(report)  # 将当前报告列表添加到总报告中
        report.append(f"二分搜索迭代 {self.iteration}")  # 记录当前二分搜索的迭代次数
        report.append(
            f"从节点索引 {start_idx}:{first_node_name} 到 {end_idx-1}:{output_node_name}。"
            f"感兴趣节点列表的大小为 {len(nodes)}"
        )  # 记录当前搜索范围的起始和结束节点信息以及节点列表大小

        cur_nodes: NodeSet = set(nodes)  # 将当前节点列表转换为集合

        try:
            split_module, submod_name = self._build_submodule(cur_nodes)  # 构建子模块并获取子模块名称
            self._run_and_compare(split_module, submod_name, [output_node_name])  # 运行比较子模块结果

        except (FxNetMinimizerRunFuncError, FxNetMinimizerResultMismatchError):
            if len(nodes) == 1:
                report.append(
                    f"这是子模块中的最后一个节点。当前分支的搜索成功，问题节点为 {cur_nodes}."
                )
                self.print_report(report)
                return cur_nodes  # 如果当前节点列表只有一个节点，直接返回当前节点作为问题节点集合

            report.append(
                "继续单独处理当前子模块的两个半部分。"
            )
            self.print_report(report)  # 打印当前报告

            mid = len(nodes) // 2  # 计算节点列表的中间位置
            culprits = self._binary_search_impl(all_nodes, start_idx, start_idx + mid)  # 对左半部分进行二分搜索

            if len(culprits) != 0 and not self.settings.find_all:
                return culprits  # 如果找到问题节点且不需要找到所有问题节点，则直接返回问题节点集合

            culprits = self._binary_search_impl(all_nodes, start_idx + mid, end_idx)  # 对右半部分进行二分搜索

            if len(culprits) == 0:
                report.append(
                    f"进一步分割和降低未发现错误。无法最小化节点列表为 {nodes} 的子模块。"
                )
                self.print_report(report)  # 打印当前报告

            return culprits  # 返回问题节点集合
        else:
            report.append("未发现任何差异。")
            self.print_report(report)  # 打印当前报告
            return set()  # 返回空的集合作为未发现问题节点的结果集

    def _binary_traverse(self, nodes: NodeList) -> NodeSet:
        """
        对节点列表进行二分搜索查找问题节点。
        """
        return self._binary_search_impl(nodes, 0, len(nodes))  # 调用实际的二分搜索实现函数，并返回结果集合
    # 定义一个方法 `_sequential_traverse`，用于按顺序遍历节点并确定是否存在问题节点。
    def _sequential_traverse(self, nodes: NodeList) -> NodeSet:
        """
        Traverse `nodes` one by one and determine if any of them is a culprit.
        逐个遍历 `nodes`，并确定是否有问题节点。
        """
        # 初始化一个空集合 `culprits`，用于存储问题节点。
        culprits: NodeSet = set()

        # 遍历传入的节点列表 `nodes`
        for node in nodes:
            # 初始化一个空列表 `report`，用于记录每个节点的报告信息。
            report: List[str] = []
            # 将 `report` 添加到类成员变量 `self.reports` 中
            self.reports.append(report)
            # 迭代次数加一
            self.iteration += 1
            # 向 `report` 中添加遍历迭代信息
            report.append(f"Sequential traverse iteration {self.iteration}.")
            # 向 `report` 中添加访问节点信息
            report.append(f"Visit node: {node.name}")

            # 在日志中记录访问节点的信息
            _LOGGER.info("Visit node: %s", node.name)
            
            # 构建仅包含当前节点的节点列表 `node_list`
            node_list: NodeList = [node]
            
            # 如果存在排除函数 `exclusion_fn`
            if self.exclusion_fn is not None:
                # 调用排除函数来更新 `node_list`
                self.exclusion_fn(node_list, -1, -1)
                # 如果更新后的 `node_list` 为空，则将排除信息添加到 `report` 中并打印报告，然后返回空的 `culprits` 集合。
                if len(node_list) == 0:
                    report.append(f"User exclusion : {node.name}")
                    self.print_report(report)
                    return culprits

            # 初始化当前节点集合 `cur_nodes`，初始值为当前遍历的节点 `node`
            cur_nodes: NodeSet = {node}

            # 如果当前节点在 `self.fusions` 中存在对应的融合节点集合，则更新 `cur_nodes` 为融合节点集合
            if node in self.fusions:
                cur_nodes = self.fusions[node]

            # 尝试构建子模块并运行比较操作
            try:
                # 构建子模块和子模块名称
                split_module, submod_name = self._build_submodule(cur_nodes)
                # 运行并比较结果
                self._run_and_compare(split_module, submod_name, [node.name])
                # 打印报告
                self.print_report(report)
            # 捕获数值错误异常，将当前节点标记为问题节点，并打印报告
            except (FxNetMinimizerResultMismatchError):
                culprits.add(node)
                report.append(f"Found culprit from numeric error: {node}")
                self.print_report(report)
                # 如果不需要找到所有问题节点，则直接返回当前的问题节点集合 `culprits`
                if not self.settings.find_all:
                    return culprits
            # 捕获运行错误异常，将当前节点及其融合节点标记为问题节点，并打印报告
            except (FxNetMinimizerRunFuncError):
                culprits.update(cur_nodes)
                report.append(f"Found culprit from run error: {node}")
                self.print_report(report)
                # 如果不需要找到所有问题节点，则直接返回当前的问题节点集合 `culprits`
                if not self.settings.find_all:
                    return culprits

        # 循环结束后，返回所有找到的问题节点集合 `culprits`
        return culprits
    def _block_traverse(self, nodes: NodeList, find_last_node: Optional[bool]) -> NodeSet:
        """
        Traverse topologically sorted node list
        Find minimium block (start_idx, end_idx) which contains the culprit
        1st pass: search for end_idx by finding the last node in culprit block
        where Numerical accuracy (0, end_idx) > threshold
        2nd pass: search for start_idx by finding the first node in culprit block
        where Numerical accuracy (start_idx, end_idx) < threshold
        Form minimum block by (start_idx - 1, end_idx)
        """
        # 初始化一个空集合，用于存储找到的节点集合
        culprits: NodeSet = set()
        # 获取节点列表中第一个和最后一个节点的名称
        first_node_name = nodes[0].name
        last_node_name = nodes[-1].name
        # 创建一个报告，记录搜索范围从第一个节点到最后一个节点的信息
        last_node_report = [f"Block search from {first_node_name} to {last_node_name}"]
        last_node_report.append("*" * 50)
        # 将报告添加到对象的报告列表中
        self.reports.append(last_node_report)

        # 初始化开始索引和结束索引
        start_idx = 0
        end_idx = len(nodes) - 1
        # 根据参数决定是否运行两次搜索
        run_both = True if find_last_node is None else False

        # 第一步：查找（0, end_idx）中的罪犯块
        if run_both or find_last_node:
            # 添加搜索最后一个节点的报告
            last_node_report.append("Start searching for last node in culprit")
            # 打印最后一个节点的搜索报告
            self.print_report(last_node_report)
            # 调用内部方法进行搜索，并更新end_idx
            end_idx = self._block_traverse_impl(nodes, start_idx, end_idx, True)
            # 更新报告，记录找到的end_idx及其对应的节点名称
            last_node_report.extend(
                [
                    "Finish Pass 1",
                    f"Find end_idx = {end_idx}:{nodes[end_idx].name}"
                ]
            )
            # 打印最后一个节点搜索的报告
            self.print_report(last_node_report)

        # 第二步：将罪犯块缩小为（start_idx, end_idx）
        if run_both or not find_last_node:
            # 创建搜索第一个节点的报告
            first_node_report = ["Start searching for first node in culprit"]
            # 打印第一个节点的搜索报告
            self.print_report(first_node_report)
            # 调用内部方法进行搜索，并更新start_idx
            start_idx = self._block_traverse_impl(nodes[0:end_idx + 1], start_idx, end_idx, False)
            # 更新报告，记录找到的start_idx及其对应的节点名称
            first_node_report.append("*" * 50)
            # 将第一个节点的报告添加到对象的报告列表中
            self.reports.append(first_node_report)
            first_node_report.extend(
                [
                    "Finish Pass 2",
                    f"Find start_idx = {start_idx}:{nodes[start_idx].name}"
                ]
            )
            # 打印第一个节点搜索的报告
            self.print_report(first_node_report)

        # 第三步：形成包含最少罪犯的模块
        culprits.update(nodes[start_idx:end_idx + 1])
        # 创建结果报告，记录找到的最小块的起始和结束节点
        result_report = [f"Finish searching, found minimum block ({nodes[start_idx]},{nodes[end_idx]})"]
        # 将结果报告添加到对象的报告列表中
        self.reports.append(result_report)
        # 打印结果报告
        self.print_report(result_report)
        # 返回找到的罪犯节点集合
        return culprits
    # 定义一个私有方法 _defined_traverse，接受一个 NodeList 参数并返回一个 NodeSet 结果
    def _defined_traverse(self, nodes: NodeList) -> NodeSet:
        """
        run user defined `nodes` and determine if it is a culprit.
        """
        # 初始化一个空的 NodeSet 用于存储有问题的节点
        culprits: NodeSet = set()
        # 如果定义了排除函数，调用它排除节点
        if self.exclusion_fn is not None:
            self.exclusion_fn(nodes, -1, -1)
        # 如果节点列表为空，记录报告并返回所有节点都被用户排除的信息
        if len(nodes) == 0:
            report = ["All nodes are excluded by user"]
            self.reports.append(report)
            return culprits

        # 获取第一个和最后一个节点的名称
        first_node_name = nodes[0].name
        output_node_name = nodes[-1].name
        # 记录定义的图形从第一个节点到最后一个节点的信息
        report = [f"Defined graph from {first_node_name} to {output_node_name}"]
        # 当前节点集合为当前节点列表的副本
        cur_nodes: NodeSet = set(nodes)
        
        # 尝试构建子模块并运行比较
        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, [output_node_name])
            # 打印报告
            self.print_report(report)
        except (FxNetMinimizerResultMismatchError, FxNetMinimizerRunFuncError):
            # 如果发现问题，记录并返回有问题的节点集合
            report.append(f"Found culprit {cur_nodes}")
            self.print_report(report)
            return culprits

        # 返回空集合，表示没有找到有问题的节点
        return culprits

    # 定义一个私有方法 _accumulate_traverse，接受一个 NodeList 参数并返回一个 NodeSet 结果
    def _accumulate_traverse(self, nodes: NodeList) -> NodeSet:
        # 初始化一个空的 NodeSet 用于存储有问题的节点
        culprits: NodeSet = set()
        # 初始化一个空的节点集合，用于存储要运行的节点
        nodes_to_run: NodeSet = set()

        # 如果设置为查找所有节点，累积遍历不支持此模式，直接返回空集合
        if self.settings.find_all:
            print("'Find All' mode is not supported in accumulate traversal.")
            return culprits

        # 遍历传入的节点列表
        for node in nodes:
            # 初始化一个空的报告列表
            report: List[str] = []
            self.reports.append(report)  # 将报告列表添加到实例的报告列表中
            self.iteration += 1  # 增加迭代计数器
            report.append(f"Accumulate traverse iteration {self.iteration}.")  # 记录累积遍历的迭代次数

            nodes_to_run.add(node)  # 将当前节点添加到要运行的节点集合中

            # 获取节点的名称，如果是元组，则取第一个元素
            node_name = node.name
            if node_name is not None and isinstance(node_name, tuple):
                node_name = node_name[0]
            # 断言节点名称不为空且是字符串类型，用于调试时检查节点名称
            assert node_name is not None and isinstance(
                node_name, str
            ), f"minimize: node_name: {node_name}"

            report.append(f"Add node: {node_name}")  # 记录添加的节点名称到报告中

            try:
                split_module, submod_name = self._build_submodule(nodes_to_run)
                self._run_and_compare(split_module, submod_name, [node_name])
                # 打印报告
                self.print_report(report)
            except (FxNetMinimizerResultMismatchError, FxNetMinimizerRunFuncError):
                # 如果发现问题，记录有问题的节点并返回
                culprits.add(node)
                report.append(f"Found culprit {node}")
                self.print_report(report)
                return culprits

        # 返回没有找到有问题节点的空集合
        return culprits
    def _skip_traverse_impl(self, all_nodes: NodeList, start_idx: int, end_idx: int) -> NodeSet:
        """
        Skip certain nodes in graph based on settings
        """
        # 初始化一个空集合来存储问题节点
        culprits: NodeSet = set()
        # 从给定的节点列表中切片获取指定范围内的节点列表
        nodes: NodeList = all_nodes[start_idx:end_idx]
        # 当前处理的节点集合，开始时为切片得到的节点集合
        cur_nodes: NodeSet = set(nodes)
        
        # 如果定义了排除函数，使用它来更新当前节点集合
        if self.exclusion_fn is not None:
            self.exclusion_fn(nodes, start_idx, end_idx)
            cur_nodes = set(nodes)
        else:
            # 否则，检查每个节点是否存在于融合节点字典中，并将其融合节点添加到当前节点集合中
            for node in nodes:
                if node in self.fusions:
                    cur_nodes.update(self.fusions[node])
        
        # 初始化一个空列表，用于存储报告信息
        report: List[str] = []
        # 将报告列表添加到报告集合中
        self.reports.append(report)
        # 增加迭代计数器
        self.iteration += 1
        # 添加节点范围描述到报告列表中
        report.append(f" Nodes block {self.iteration}.")
        report.append(
            f"From node index {start_idx} to {end_idx-1}. "
            f"Size of the interested node list is {len(nodes)}"
        )

        try:
            # 构建子模块并进行比较
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, [])
        except (FxNetMinimizerResultMismatchError):
            # 如果出现数值错误，记录问题节点并打印报告
            culprits.update(cur_nodes)
            report.append(f"Found culprit from numeric error: {cur_nodes}")
            self.print_report(report)
            return culprits
        except (FxNetMinimizerRunFuncError):
            # 如果运行错误，记录问题节点并打印报告
            culprits.update(cur_nodes)
            report.append(f"Found culprit from run error: {cur_nodes}")
            self.print_report(report)
            return culprits
        else:
            # 如果没有发现问题，添加一条无差异的报告并打印
            report.append("No discrepancy found.")
            self.print_report(report)
            return set()


    def _skip_traverse(self, all_nodes: NodeList, skip_nodes: List) -> NodeSet:
        """
        Skip certain nodes in graph based on settings
        """
        # 初始化起始索引和节点总数
        start_idx = 0
        num_nodes = len(all_nodes)
        idx = 0
        # 初始化问题节点集合
        culprits = set()
        # 遍历所有节点
        while idx < num_nodes:
            node = all_nodes[idx]
            # 如果当前节点名在需要跳过的节点列表中，调用内部方法处理跳过逻辑
            if (node.name in skip_nodes):  # skip the node
                if idx > start_idx:
                    culprits = self._skip_traverse_impl(all_nodes, start_idx, idx)
                start_idx = idx + 1
            # 如果是最后一个节点，并且起始索引小于等于当前索引，也调用内部方法处理跳过逻辑
            elif idx == num_nodes - 1 and start_idx <= idx:  # last node
                culprits = self._skip_traverse_impl(all_nodes, start_idx, idx + 1)
            idx += 1

        return culprits



    def _collect_nodes(self, start: Optional[str], end: Optional[str]) -> NodeList:
        """
        Collect nodes in the model that between nodes with name of `start` and `end`.
        These two nodes are also included.
        """
        # 初始化一个空节点列表
        nodes: NodeList = []
        # 标志变量，用于指示是否开始添加节点
        add_node = start is None

        # 遍历模型中的所有节点
        for node in self.module.graph.nodes:
            # 如果节点的操作类型不在可调用节点操作列表中，则继续下一个节点
            if node.op not in CALLABLE_NODE_OPS:
                continue

            # 如果节点名称与起始节点名称匹配，设置标志变量以开始添加节点
            if node.name == start:
                add_node = True

            # 如果标志变量为真，将节点添加到节点列表中
            if add_node:
                nodes.append(node)

            # 如果节点名称与结束节点名称匹配，停止添加节点
            if node.name == end:
                break

        return nodes
    def run_nodes(self, start: Optional[str] = None, end: Optional[str] = None):
        """
        Run part of the model from `start` node to `end` node. If `start` is None
        then we start from the beginning of the model. If `end` is None then we
        stop at the end of the model.

        Args:
            start: The name of the node which is the first node of the submodule
                we want to run. If set to None, then we'll start with the first
                node of the model.
            end: The name of the node which is the last node of the submodule we
                want to run. If set to None, we'll end with the last node of the
                model.
        """
        # Collect nodes to run based on start and end constraints
        nodes = self._collect_nodes(start, end)
        # Initialize current nodes to be run
        cur_nodes = set(nodes)

        # Update current nodes with fused nodes if present
        for node in nodes:
            if node in self.fusions:
                cur_nodes.update(self.fusions[node])

        # Prepare output names if return_intermediate setting is enabled
        output_names = []
        if self.settings.return_intermediate:
            output_names = [node.name for node in nodes]

        try:
            # Build submodule and retrieve submodule name
            split_module, submod_name = self._build_submodule(cur_nodes)
            # Execute submodule run and compare outputs if necessary
            self._run_and_compare(split_module, submod_name, output_names)
        except (
            FxNetMinimizerRunFuncError,
            FxNetMinimizerResultMismatchError,
        ) as e:
            # Handle specific errors related to submodule execution
            print(e)

    def print_report(self, report: List[str]):
        # Print each entry in the report list with appropriate formatting
        for i in range(len(report)):
            if i > 0:
                print(" . " + report[i])
            else:
                print(report[i])

    def print_reports(self):
        # Print all reports stored in the instance
        for report in self.reports:
            self.print_report(report)

    def minimize(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        skip_nodes: Optional[List] = None,
        find_last_node: Optional[bool] = None,
    ) -> NodeSet:
        """
        Minimizing the model from node with name `start` to node with name `end` based
        on self.settings. Find culprits that cause FxNetMinimizerRunFuncError or
        FxNetMinimizerResultMismatchError errors.

        Args:
            start: The name of the node where we want to start minimizing. If set
                to None, then we'll start with the first node of the model.
            end: The name of the node where we want to terminate minimizing. If
                set to None, we'll end with the last node of the model.
            skip_nodes: The names of nodes where we want to skip during minimizing.
                It'll create subgraphs without these skip nodes under the hood.
                Only applicable in mode "skip".
            find_last_node: True if only the last node of a culprits is needed in mode "block".
                False if only the first node of a culprits is needed.
                Only applicable in mode "block".

        Returns:
            nodes: A list of nodes that cause FxNetMinimizerRunFuncError or
                FxNetMinimizerResultMismatchError errors during minimizing.
        """
        
        # 打印当前设置，用于调试
        print(self.settings)
        # 打印模型的图结构，用于调试
        print(self.module.graph)

        # 收集从 start 到 end 节点之间的所有节点
        nodes = self._collect_nodes(start, end)

        # 根据设置的遍历方法选择相应的节点遍历方式
        if self.settings.traverse_method == "sequential":
            return self._sequential_traverse(nodes)

        if self.settings.traverse_method == "binary":
            return self._binary_traverse(nodes)

        if self.settings.traverse_method == "accumulate":
            return self._accumulate_traverse(nodes)

        if self.settings.traverse_method == "skip":
            if skip_nodes is None:
                raise RuntimeError("'skip_nodes' can't be None when 'traverse_method' is 'skip'.")
            return self._skip_traverse(nodes, skip_nodes)

        if self.settings.traverse_method == "defined":
            return self._defined_traverse(nodes)

        if self.settings.traverse_method == "block":
            return self._block_traverse(nodes, find_last_node)

        # 如果设置了未知的遍历方法，抛出异常
        raise RuntimeError(f"Unknown traverse method {self.settings.traverse_method}!")
```