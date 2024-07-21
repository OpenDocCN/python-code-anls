# `.\pytorch\torch\_inductor\fx_passes\group_batch_fusion.py`

```py
# 添加类型检查允许未类型化的定义
# 导入必要的模块和库
import collections
import logging
import operator
from collections import OrderedDict
from typing import (
    Any,
    DefaultDict,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

# 导入 torch 库
import torch

# 导入内部工具函数和计数器
from torch._dynamo.utils import counters, optimus_scuba_log
# 导入内部工具函数用于上传图形
from torch._utils_internal import upload_graph
# 导入图形转换观察者类
from torch.fx.passes.graph_transform_observer import GraphTransformObserver

# 导入配置和模式匹配器
from .. import config
from ..pattern_matcher import (
    CallFunctionVarArgs,
    get_arg_value,
    stable_topological_sort,
)

try:
    # 尝试导入 deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings 模块，
    # 注册 fbgemm 的降低操作
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    # 如果导入失败，设置 has_fbgemm 为 False
    has_fbgemm = False
    pass

# 设置 torch 的 aten 操作为 aten
aten = torch.ops.aten

# 设置日志记录器
log = logging.getLogger(__name__)

# 定义融合组的最小和最大大小
MIN_FUSE_SET_SIZE = 5
MAX_FUSE_SET_SIZE = 300
MAX_FUSE_SEARCH_DEPTH = 5

# 融合组中张量的最大大小
MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR = 4096

# 是否只融合具有相同父节点的节点
FUSE_NODES_WITH_SAME_PARENT = False

# 是否在批线性加法中启用广播
SHAPE_BROADCAST_BATCH_LINEAR = False

# 是否启用与相同用户的节点融合
Fuse_NODES_WITH_SAME_USERS = False

# 从 BFS 排除的节点集合，例如排除 getitem 操作
SEARCH_EXCLUSIONS = {operator.getitem}

# 默认的图搜索选项
default_graph_search_options = {
    "min_fuse_set_size": MIN_FUSE_SET_SIZE,
    "max_fuse_set_size": MAX_FUSE_SET_SIZE,
    "max_fuse_search_depth": MAX_FUSE_SEARCH_DEPTH,
    "max_fuse_tensor_size_group_linear": MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR,
    "fuse_nodes_with_same_parent": FUSE_NODES_WITH_SAME_PARENT,
    "shape_broadcast_batch_linear": SHAPE_BROADCAST_BATCH_LINEAR,
    "fuse_nodes_with_same_users": Fuse_NODES_WITH_SAME_USERS,
}

# 当前图搜索选项默认为默认图搜索选项
graph_search_options = default_graph_search_options


def update_stack_example_value(node, metadata, dim=0, op=torch.stack):
    """
    更新图中节点的示例值，以便进行后续的拆分和连接优化。
    """
    if node is not None and hasattr(node, "meta"):
        # 如果节点不为空且具有 meta 属性
        if op == torch.stack:
            # 如果操作为 torch.stack
            example_value = torch.stack(metadata, dim=dim)
        elif op == torch.unbind:
            # 如果操作为 torch.unbind
            example_value = torch.unbind(metadata, dim=dim)  # type: ignore[assignment]
        else:
            return
        # 将示例值存储到节点的 meta 属性中
        node.meta["example_value"] = example_value


def update_pointwise_example_value(pointwise_node, input, other, op):
    """
    更新图中加法节点的示例值，以便进行后续的拆分和连接优化。
    """
    if pointwise_node is not None and hasattr(pointwise_node, "meta"):
        # 如果加法节点不为空且具有 meta 属性
        if op == torch.add:
            # 如果操作为 torch.add
            example_value = torch.add(input, other)
        elif op == torch.mul:
            # 如果操作为 torch.mul
            example_value = torch.mul(input, other)
        else:
            return
        # 将示例值存储到加法节点的 meta 属性中
        pointwise_node.meta["example_value"] = example_value


class GroupBatchFusionBase:
    # 待实现的批次融合基类
    # 初始化函数，用于对象的初始化操作
    def __init__(self, **kwargs):
        # 从关键字参数中弹出 'graph_search_options'，如果不存在则使用默认的搜索选项
        self.graph_search_options = kwargs.pop(
            "graph_search_options", default_graph_search_options
        )

    # 匹配函数，子类需要实现具体的匹配逻辑
    def match(self, node):
        # 抛出未实现错误，提示子类需要实现该方法
        raise NotImplementedError("match called on base")

    # 融合函数，子类需要实现具体的融合逻辑
    def fuse(self, graph, subset):
        # 抛出未实现错误，提示子类需要实现该方法
        raise NotImplementedError("fuse called on base")
# 定义一个空的字典，用于存储预梯度融合的名称与融合类对象的映射关系
PRE_GRAD_FUSIONS: Dict[str, GroupBatchFusionBase] = dict()
# 定义一个空的字典，用于存储后梯度融合的名称与融合类对象的映射关系
POST_GRAD_FUSIONS: Dict[str, GroupBatchFusionBase] = dict()


def register_fusion(name: str, pre_grad=True):
    # 定义一个装饰器函数，用于注册融合操作类到预梯度或后梯度的字典中
    def decorator(fusion_cls: GroupBatchFusionBase):
        # 根据 pre_grad 参数决定将融合类对象注册到预梯度字典还是后梯度字典中
        if pre_grad:
            PRE_GRAD_FUSIONS[name] = fusion_cls
        else:
            POST_GRAD_FUSIONS[name] = fusion_cls
        return fusion_cls

    return decorator


def list_group_batch_fusions(pre_grad=True) -> List[str]:
    # 返回当前已注册的预梯度或后梯度融合名称列表
    if pre_grad:
        return list(PRE_GRAD_FUSIONS.keys())
    else:
        return list(POST_GRAD_FUSIONS.keys())


def decompose_stack(graph: torch.fx.GraphModule, input_tensors: List[Any]) -> Any:
    # 将输入张量列表中的每个张量在图中进行展开操作，同时更新元数据
    unsqueezed_inputs = []
    unsqueezed_inputs_meta = []
    for input_tensor in input_tensors:
        unsqueezed_input = graph.call_function(
            aten.unsqueeze, args=(input_tensor,), kwargs={"dim": 0}
        )
        unsqueezed_inputs.append(unsqueezed_input)
        # 更新展开后的输入张量的元数据
        unsqueezed_input.meta["val"] = aten.unsqueeze(input_tensor.meta["val"], dim=0)  # type: ignore[assignment]
        unsqueezed_inputs_meta.append(unsqueezed_input.meta["val"])
    # 将展开后的输入张量堆叠起来形成一个新的张量
    stacked_inputs = graph.call_function(
        aten.cat, args=(unsqueezed_inputs,), kwargs={"dim": 0}
    )
    # 更新堆叠后的张量的元数据
    stacked_inputs.meta["val"] = aten.cat(unsqueezed_inputs_meta, dim=0)  # type: ignore[assignment]
    return stacked_inputs


class GroupFusion(GroupBatchFusionBase):
    """
    以组的方式融合操作，例如，将任意形状的 mm/addmm 操作与 fbgemm.gmm 进行融合。
    """
    pass


class BatchFusion(GroupBatchFusionBase):
    """
    以批的方式融合操作，例如，将相同形状的 mm/addmm 操作与 bmm 进行融合。
    """
    pass


class BatchPointwiseOpsFusionFactory(BatchFusion):
    def __init__(self, op, **kwargs):
        super().__init__(**kwargs)
        self.op = op


@register_fusion("batch_linear_post_grad", pre_grad=False)
class PostGradBatchLinearFusion(BatchFusion):
    """
    以后梯度的批量方式融合操作 (aten 级别)。
    """

    def _addmm_node_can_be_fused(self, node: torch.fx.Node) -> bool:
        # 检查节点是否可以融合的私有方法，判断 beta 和 alpha 参数是否为 1.0
        return (
            node.kwargs.get("beta", 1.0) == 1.0 and node.kwargs.get("alpha", 1.0) == 1.0  # type: ignore[return-value]
        )

    def _is_input_2d(self, input: torch.fx.Node) -> bool:
        # 判断输入张量是否为二维的私有方法，检查输入张量的形状
        input_shapes = input.meta["val"].shape
        return (
            len(input_shapes) == 2
            and isinstance(input_shapes[0], int)
            and isinstance(input_shapes[1], int)
        )

    def match(
        self, node: torch.fx.Node
        # 继承自父类的方法，用于在后梯度融合中匹配操作节点
        ) -> Optional[Tuple[str, int, int, int, bool, str]]:
        # 如果节点是 mm 函数调用，匹配并获取输入的张量
        if CallFunctionVarArgs(aten.mm).match(node):
            input_m, weight_m = node.args
            bias_m = None

        # 如果节点是 addmm 函数调用且可以融合，则获取输入的张量和偏置
        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias_m, input_m, weight_m = node.args
        else:
            return None

        # 获取节点的用户（即依赖于此节点的其他节点或操作）
        if self.graph_search_options.get("fuse_nodes_with_same_users", False):
            users = [user.target for user in node.users.keys()]
        else:
            users = ""  # type: ignore[assignment]

        # 只处理输入为二维张量的情况
        if not self._is_input_2d(input_m) or not self._is_input_2d(weight_m):  # type: ignore[arg-type]
            return None
        
        # 获取输入张量的形状信息
        m, k = input_m.meta["val"].shape  # type: ignore[union-attr]
        n = weight_m.meta["val"].shape[1]  # type: ignore[union-attr]
        
        # 构建批量线性后梯度计算的关键字
        batch_key = ("batch_linear_post_grad", m, k, n, bias_m is not None, str(users))
        return batch_key
# 注册一个自定义融合操作 "group_linear"，且不需要预先计算梯度
@register_fusion("group_linear", pre_grad=False)
# GroupLinearFusion 类继承自 GroupFusion 类
class GroupLinearFusion(GroupFusion):
    # 判断是否可以融合 _addmm_node_can_be_fused 方法
    def _addmm_node_can_be_fused(self, node: torch.fx.Node):
        # 获取第二个参数的形状信息
        input_shape = node.args[1].meta["val"].shape  # type: ignore[union-attr]
        # 获取第三个参数的形状信息
        weight_shape = node.args[2].meta["val"].shape  # type: ignore[union-attr]
        # 检查是否满足融合条件：beta 和 alpha 等于 1.0，输入和权重的维度为二维，且各维度大小均为偶数
        return (
            node.kwargs.get("beta", 1.0) == 1.0
            and node.kwargs.get("alpha", 1.0) == 1.0
            and len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    # 判断是否可以融合 _mm_node_can_be_fused 方法
    def _mm_node_can_be_fused(self, node: torch.fx.Node):
        # 获取第一个参数的形状信息
        input_shape = node.args[0].meta["val"].shape  # type: ignore[union-attr]
        # 获取第二个参数的形状信息
        weight_shape = node.args[1].meta["val"].shape  # type: ignore[union-attr]
        # 检查是否满足融合条件：输入和权重的维度为二维，且各维度大小均为偶数
        return (
            len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    # 匹配节点是否可以融合，返回融合操作的键值对或者 None
    def match(self, node: torch.fx.Node) -> Optional[Tuple[str, bool]]:
        # 如果节点匹配 torch 的 mm 默认函数，并且可以进行融合
        if CallFunctionVarArgs(aten.mm.default).match(
            node
        ) and self._mm_node_can_be_fused(node):
            # 设置融合键值对，指示为 "group_linear" 融合操作，并且无偏置
            group_key = ("group_linear", True)
        # 如果节点匹配 torch 的 addmm 默认函数，并且可以进行融合
        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            # 获取偏置参数
            bias = node.args[0]
            # 设置融合键值对，指示为 "group_linear" 融合操作，并且有偏置
            group_key = ("group_linear", bias is None)
        else:
            # 如果不匹配任何融合条件，设置键值对为 None
            group_key = None
        # 返回最终的融合键值对或者 None
        return group_key
    # 定义一个方法 `fuse`，用于融合部分节点为一个新节点
    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        # 初始化用于存储融合节点的输入、权重、偏置和节点本身的列表
        group_inputs = []
        group_weights = []
        group_biases = []
        group_nodes = []

        # 遍历给定的节点子集
        for node in subset:
            # 检查节点是否匹配 `aten.addmm.default` 函数调用
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                # 如果匹配，则解析出偏置、输入、权重
                bias, input, weight = node.args
            else:
                # 如果不匹配，则断言节点匹配 `aten.mm.default` 函数调用
                assert CallFunctionVarArgs(aten.mm.default).match(node)
                # 解析出输入、权重，偏置为 None
                input, weight = node.args
                bias = None
            
            # 将解析得到的节点信息存入相应的列表中
            group_nodes.append(node)
            group_inputs.append(input)
            group_weights.append(weight)
            group_biases.append(bias)

        # 如果所有融合节点的偏置都为 None，则将 `group_biases` 设置为 None
        if all(bias is None for bias in group_biases):
            group_biases = None  # type: ignore[assignment]

        # 在子集的第一个节点之前插入新的函数调用节点 `torch.ops.fbgemm.gmm.default`
        with graph.inserting_before(subset[0]):
            fused_mm = graph.call_function(
                torch.ops.fbgemm.gmm.default,
                args=(group_inputs, group_weights, group_biases),
                kwargs={"smart_fused": True},
            )

        # 遍历融合节点列表，为每个原始节点创建一个新的函数调用节点
        for i, original_mm in enumerate(group_nodes):
            # 在 `fused_mm` 节点之后插入新的函数调用节点
            with graph.inserting_after(fused_mm):
                new_mm = graph.call_function(operator.getitem, args=(fused_mm, i))
            # 替换所有使用原始节点的地方为新节点
            original_mm.replace_all_uses_with(new_mm)
            # 更新新节点的元数据
            new_mm.meta.update(original_mm.meta)
            # 从图中删除原始节点
            graph.erase_node(original_mm)

        # 增加计数器中的组合线性操作计数
        counters["inductor"]["group_linear"] += 1
    """
    Batch pointwise math operator (e.g., add, mul) in post grad pass.
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op  # 初始化操作符

    def _pointwise_node_can_be_fused(self, node: torch.fx.Node):
        # 注意：只考虑输入都是张量的情况
        # 对于混合精度训练，确保在堆叠时aten.cat的输入张量具有相同的dtype
        # 否则，aten.cat的输出可能与输入不同，并在mm或addmm中引发dtype不同的错误
        input, other = node.args
        return (
            input.meta["val"].shape == other.meta["val"].shape  # 确保输入张量形状相同
            if hasattr(input, "meta")
            and hasattr(other, "meta")
            and "val" in input.meta
            and "val" in other.meta
            else False
        )

    def match(self, node: torch.fx.Node):
        # 检查节点是否匹配操作符并且可以融合
        if CallFunctionVarArgs(self.op).match(node) and self._pointwise_node_can_be_fused(node):
            alpha = node.kwargs.get("alpha", 1.0)  # 获取alpha参数，默认为1.0
            rounding_mode = node.kwargs.get("rounding_mode", None)  # 获取rounding_mode参数，默认为None
            input, other = node.args  # 获取节点的输入参数
            shape = list(input.meta["val"].shape)  # 获取输入张量的形状
            if self.graph_search_options.get("fuse_nodes_with_same_parent", False):
                # 仅考虑线性情况
                if input.target == aten.select or other.target == aten.select:
                    # 如果其中一个输入目标是aten.select
                    parent = (
                        input.args[0] if input.target == aten.select else other.args[0]
                    )  # 获取父节点
                else:
                    parent = ""
            else:
                parent = ""
            # 组成融合键值，用于标识批量操作的唯一性
            group_key = (
                "batch_aten_" + self.op.__name__.lower().split(".")[0],  # 操作符名称的小写形式
                str(shape),  # 输入张量形状的字符串表示
                str(input.meta["val"].dtype),  # 输入张量的dtype
                str(other.meta["val"].dtype),  # 另一个输入张量的dtype
                str(alpha),  # alpha参数的字符串表示
                str(rounding_mode),  # rounding_mode参数的字符串表示
                str(parent),  # 父节点的字符串表示
            )
        else:
            group_key = None  # 如果节点不匹配或无法融合，则组键为None
        return group_key  # 返回组键作为匹配结果
    # 定义方法 `fuse`，用于融合图模块中的子图节点
    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        # 初始化空列表，用于存储批量输入和其他数据
        batch_inputs, batch_others = [], []
        # 从第一个子图节点的关键字参数中获取 `alpha` 值，默认为 1.0
        alpha = subset[0].kwargs.get("alpha", 1.0)
        # 初始化空列表，用于存储批量输入和其他数据的元信息
        batch_inputs_meta, batch_others_meta = [], []

        # 遍历子图节点列表
        for node in subset:
            # 获取当前节点的输入和其他数据
            input, other = node.args
            # 将输入和其他数据添加到对应的批量列表中
            batch_inputs.append(input)
            batch_others.append(other)
            # 将输入和其他数据的元信息添加到对应的批量元信息列表中
            batch_inputs_meta.append(input.meta)  # type: ignore[possibly-undefined, union-attr]
            batch_others_meta.append(other.meta)  # type: ignore[possibly-undefined, union-attr]

        # 在第一个子图节点之前插入操作
        with graph.inserting_before(subset[0]):
            # 分解批量输入和其他数据的堆栈
            stack_inputs = decompose_stack(graph, batch_inputs)
            stack_others = decompose_stack(graph, batch_others)
            # 将批量输入和其他数据的元信息堆叠为张量
            stack_inputs_meta = torch.stack(
                [input["val"] for input in batch_inputs_meta]
            )
            stack_others_meta = torch.stack(
                [other["val"] for other in batch_others_meta]
            )

            # 在图中调用函数操作 `self.op`，传入堆栈输入和其他数据，以及可能的 `alpha` 值
            batch_op = graph.call_function(
                self.op,
                args=(stack_inputs, stack_others),
                kwargs={"alpha": alpha} if self.op == aten.add.Tensor else {},
            )
            # 更新 `batch_op` 的元信息值
            batch_op.meta["val"] = self.op(stack_inputs_meta, stack_others_meta)
            
            # 遍历子图节点列表中的每个节点
            for i, original_add in enumerate(subset):
                # 在 `batch_op` 之后插入新的函数调用节点
                with graph.inserting_after(batch_op):
                    new_add = graph.call_function(
                        torch.ops.aten.select, args=((batch_op, 0, i))
                    )
                # 将原始节点 `original_add` 的所有使用点替换为新节点 `new_add`
                original_add.replace_all_uses_with(new_add)
                # 更新新节点 `new_add` 的元信息
                new_add.meta.update(original_add.meta)
                # 从图中删除原始节点 `original_add`
                graph.erase_node(original_add)
        
        # 增加计数器中关于当前操作类型的批处理统计信息
        counters["inductor"][
            "batch_aten_" + self.op.__name__.lower().split(".")[0]
        ] += 1
@register_fusion("batch_linear_lhs")
class BatchLinearLHSFusion(BatchFusion):
    """
    Batch linear left-hand side fusion. This pass tries to fuse the following patterns:

        torch.nn.functional.linear(x, w1), linear(x, w2),... * linear(x, wn)
        -> torch.mm(x, torch.cat([w1, w2,... * wn]).transpose(0, 1))

    We have a separate pass to eliminate contiguous transpose in a generic way.
    """

    def match(self, node: torch.fx.Node) -> Optional[Tuple[str, bool, Any]]:
        # 检查节点是否匹配 torch.nn.functional.linear 调用，并且可以被融合
        if CallFunctionVarArgs(torch.nn.functional.linear).match(
            node
        ) and is_linear_node_can_be_fused(node):
            # 获取输入、权重和偏置的值
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            # 确定分组的关键信息
            group_key = ("batch_linear_lhs", bias is None, input)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        # 初始化融合所需的变量和列表
        batch_nodes = []
        batch_input = None
        batch_weights = []
        batch_biases = []
        split_sections = []

        # 遍历要融合的节点列表
        for node in subset:
            # 获取节点的输入、权重和偏置值，并添加到对应的列表中
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            batch_nodes.append(node)
            if batch_input is None:
                batch_input = input
            else:
                assert batch_input is input
            batch_weights.append(weight)
            if bias:
                batch_biases.append(bias)
            split_sections.append(weight.meta["example_value"].shape[0])

        # 在融合节点集合的第一个节点之前插入新的图操作
        with graph.inserting_before(subset[0]):
            # 拼接权重张量，并进行转置操作
            cat_weights = graph.call_function(
                torch.cat, args=(batch_weights,), kwargs={"dim": 0}
            )
            transposed_weights = graph.call_function(
                torch.transpose, args=(cat_weights, 0, 1)
            )
            if len(batch_biases) > 0:
                # 如果存在偏置项，进行加权矩阵乘法
                cat_biases = graph.call_function(
                    torch.cat, args=(batch_biases,), kwargs={"dim": 0}
                )
                fused_lhs = graph.call_function(
                    torch.addmm,
                    args=(cat_biases, batch_input, transposed_weights),
                )
            else:
                # 否则，进行普通矩阵乘法
                fused_lhs = graph.call_function(
                    torch.mm,
                    args=(batch_input, transposed_weights),
                )
            # 将融合后的结果按照指定维度进行切分
            fused_lhs_list = graph.call_function(
                torch.split, args=(fused_lhs, split_sections), kwargs={"dim": 1}
            )

        # 遍历融合的节点，并替换其使用，并更新元数据
        for i, node in enumerate(batch_nodes):
            with graph.inserting_after(fused_lhs_list):
                new_node = graph.call_function(
                    operator.getitem, args=(fused_lhs_list, i)
                )
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)
        counters["inductor"]["batch_linear_lhs"] += 1
# 检查给定节点是否为有效节点，即非空节点
def is_node_meta_valid(node: Optional[torch.fx.Node]):
    if node is None:
        return True  # 如果节点为空，则认为是有效节点
    if "example_value" not in node.meta and "val" not in node.meta:
        return False  # 如果节点的元数据中缺少 'example_value' 和 'val'，则认为是无效节点
    return True  # 其他情况下认为是有效节点


# 判断节点是否为可变节点，即是否会修改其输入
def _is_mutable_node(tgt):
    if str(tgt).endswith("_"):
        return True  # 如果目标字符串以 '_' 结尾，通常表示会原地修改输入
    if (
        hasattr(tgt, "__module__")
        and tgt.__module__ == "_operator"
        and tgt.__name__.startswith("i")
    ):
        return True  # 如果目标是位于 '_operator' 模块且以 'i' 开头，也表示可能会原地修改输入
    return False  # 其他情况下认为不会原地修改输入


# 判断是否可以融合线性节点
def is_linear_node_can_be_fused(node: torch.fx.Node):
    input = get_arg_value(node, 0, "input")  # 获取节点的第一个参数作为输入
    weight = get_arg_value(node, 1, "weight")  # 获取节点的第二个参数作为权重
    return (
        is_node_meta_valid(node)  # 节点本身需要是有效节点
        and is_node_meta_valid(input)  # 输入参数需要是有效节点
        and is_node_meta_valid(weight)  # 权重参数需要是有效节点
        and len(input.meta["example_value"].shape) == 2  # 输入参数的例子值的形状必须是二维的
        and len(weight.meta["example_value"].shape) == 2  # 权重参数的例子值的形状必须是二维的
        # 防止由于 mm -> bmm 转换而添加 unbind() 操作，这种操作在 mm 输出被修改时不安全
        # 因此，如果 mm 的任何用户会修改输入，则不进行模式匹配
        and not any(_is_mutable_node(user.target) for user in node.users)
    )


# 注册批量线性融合类，继承自批量融合基类
@register_fusion("batch_linear")
class PreGradBatchLinearFusion(BatchFusion):
    """
    Batch linear fusion in pre grad pass.
    Fuse linear with same size with torch.baddmm
    """

    # 获取索引操作节点的参数
    def _getitem_args(self, getitem_node: torch.fx.Node):
        if getitem_node.target != operator.__getitem__ or (
            getitem_node.op != "call_function"
        ):
            return None  # 如果节点不是调用 'operator.__getitem__' 函数，则返回 None
        return getitem_node.args[0]  # 否则返回该节点的第一个参数

    # 判断节点是否符合融合条件
    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(torch.nn.functional.linear).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")  # 获取节点的第一个参数作为输入
            weight = get_arg_value(node, 1, "weight")  # 获取节点的第二个参数作为权重
            bias = get_arg_value(node, 2, "bias")  # 获取节点的第三个参数作为偏置
            if self.graph_search_options.get("fuse_nodes_with_same_users", False):
                users = [user.target for user in node.users.keys()]  # 获取所有用户的目标
            else:
                users = ""  # 如果不需要融合相同用户的节点，则设为空字符串
            # 创建融合组键，用于标识相同融合组的节点
            group_key = (
                "batch_linear",  # 标记为批量线性融合
                self._getitem_args(input),  # 输入参数的索引操作参数
                str(input.meta["example_value"].shape),  # 输入参数的例子值形状字符串表示
                str(weight.meta["example_value"].shape),  # 权重参数的例子值形状字符串表示
                bias is None,  # 偏置是否为空
                str(users),  # 用户目标的字符串表示
            )
        else:
            group_key = None  # 如果节点不符合融合条件，则组键为空
        return group_key  # 返回融合组键或 None
    # 定义一个方法，用于将给定的子图节点集合融合到主图中
    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        # 初始化用于批处理的各种列表
        batch_nodes = []  # 存储子图中的节点
        batch_inputs = []  # 存储节点的输入
        batch_weights = []  # 存储节点的权重
        batch_biases = []  # 存储节点的偏置
        batch_inputs_metadata = []  # 存储输入的元数据
        batch_weights_metadata = []  # 存储权重的元数据
        batch_biases_metadata = []  # 存储偏置的元数据

        # 遍历子图中的每个节点
        for node in subset:
            batch_nodes.append(node)  # 将节点添加到批处理节点列表中
            # 获取节点的第一个输入（通常是输入数据）
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)  # 将输入添加到批处理输入列表中
            batch_inputs_metadata.append(input.meta["example_value"])  # 获取输入的示例值元数据
            # 获取节点的第二个输入（通常是权重）
            weight = get_arg_value(node, 1, "weight")
            batch_weights.append(weight)  # 将权重添加到批处理权重列表中
            batch_weights_metadata.append(weight.meta["example_value"])  # 获取权重的示例值元数据
            # 获取节点的第三个输入（通常是偏置）
            bias = get_arg_value(node, 2, "bias")
            batch_biases.append(bias)  # 将偏置添加到批处理偏置列表中
            # 如果偏置不为空且具有 "meta" 属性，则获取偏置的示例值元数据
            if bias is not None and hasattr(bias, "meta"):
                batch_biases_metadata.append(bias.meta["example_value"])

        # 在子图的第一个节点之前插入新节点
        with graph.inserting_before(subset[0]):
            # 对输入数据进行堆叠操作，按照第一个维度（dim=0）堆叠
            stack_inputs = graph.call_function(
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            # 更新堆叠后的输入数据的示例值元数据
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            # 对权重进行堆叠操作，按照第一个维度（dim=0）堆叠
            stack_weights = graph.call_function(
                torch.stack, args=(batch_weights,), kwargs={"dim": 0}
            )
            # 更新堆叠后的权重的示例值元数据
            update_stack_example_value(stack_weights, batch_weights_metadata)
            # 对权重进行转置操作，交换维度1和2
            transpose_weight = graph.call_function(
                torch.transpose, args=(stack_weights, 1, 2)
            )
            # 如果批处理中所有偏置都为 None，则执行批量矩阵乘操作
            if all(bias is None for bias in batch_biases):
                bmm = graph.call_function(
                    torch.bmm,
                    args=(stack_inputs, transpose_weight),
                )
            else:
                # 对偏置进行堆叠操作，按照第一个维度（dim=0）堆叠
                stack_biases = graph.call_function(
                    torch.stack, args=(batch_biases,), kwargs={"dim": 0}
                )
                # 更新堆叠后的偏置的示例值元数据
                update_stack_example_value(stack_biases, batch_biases_metadata)
                # 对偏置进行维度扩展操作，在第1个维度上扩展
                unsqueeze_biases = graph.call_function(
                    torch.unsqueeze, args=(stack_biases, 1)
                )
                # 执行批量矩阵乘加操作，包括偏置
                bmm = graph.call_function(
                    torch.baddbmm,
                    args=(unsqueeze_biases, stack_inputs, transpose_weight),
                )

            # 对结果进行解绑操作，按照第一个维度（dim=0）解绑
            bmm = graph.call_function(torch.unbind, args=(bmm,), kwargs={"dim": 0})
            # 遍历批处理的每个节点和对应的结果
            for i, linear in enumerate(batch_nodes):
                # 在解绑结果之后插入新节点
                with graph.inserting_after(bmm):
                    # 获取解绑结果中的特定索引项
                    getitem = graph.call_function(operator.getitem, args=(bmm, i))
                # 替换原始节点的所有使用为获取的索引项
                linear.replace_all_uses_with(getitem)
                # 更新获取的索引项的元数据为原始节点的元数据
                getitem.meta.update(linear.meta)
                # 从图中删除原始节点
                graph.erase_node(linear)
        # 增加计数器中感应器的批处理线性计数器
        counters["inductor"]["batch_linear"] += 1
# 注册一个名为 "batch_layernorm" 的融合操作，用于批量层归一化
@register_fusion("batch_layernorm")
class BatchLayernormFusion(BatchFusion):
    """
    Batch layer norm fusion in pre grad pass
    """

    # 定义匹配函数，用于判断节点是否符合批量层归一化的条件
    def match(self, node: torch.fx.Node):
        # 检查节点是否调用了 torch.nn.functional.layer_norm 函数
        if CallFunctionVarArgs(torch.nn.functional.layer_norm).match(node):
            # 获取调用节点的输入、权重和偏置
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 2, "weight")
            bias = get_arg_value(node, 3, "bias")
            # 根据配置决定是否获取使用了同一用户的节点列表
            if self.graph_search_options.get("fuse_nodes_with_same_users", False):
                users = [user.target for user in node.users.keys()]
            else:
                users = ""  # type: ignore[assignment]
            # 构建用于分组的键值，包括操作类型、输入形状、权重形状、偏置形状、归一化形状、epsilon、用户列表等信息
            group_key = (
                (
                    "batch_layernorm",
                    str(input.meta["example_value"].shape),
                    str(weight.meta["example_value"].shape) if weight is not None else "",
                    str(bias.meta["example_value"].shape) if bias is not None else "",
                    str(get_arg_value(node, 1, "normalized_shape")),
                    str(get_arg_value(node, 4, "eps")),
                    str(users),
                )
                if "example_value" in input.meta
                and is_node_meta_valid(weight)
                and is_node_meta_valid(bias)
                else None
            )
        else:
            group_key = None
        return group_key

# 批量逐点操作（例如 sigmoid、relu、tanh）的融合工厂类，在预梯度传递阶段融合
class BatchPointwiseOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise ops (e.g., sigmoid, relu, tanh) fusion in pre grad pass.
    We fuse it in random place, and the introduced stack node may be merged in split cat.
    """

    # 初始化方法，接受操作符参数并调用父类的初始化方法
    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op

    # 匹配函数，用于判断节点是否符合批量逐点操作的融合条件
    def match(self, node: torch.fx.Node):
        # 获取节点的输入
        input = get_arg_value(node, 0, "input")
        # 检查节点是否调用了指定的操作符函数，并验证节点的元数据有效性
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            # 根据配置决定是否获取使用了同一父节点的信息
            if self.graph_search_options.get("fuse_nodes_with_same_parent", False):
                parent = node.args[0]  # pyre-fixme[16]
                parent = parent.target if parent is not None else ""  # type: ignore[union-attr]
            else:
                parent = ""
            # 构建用于分组的键值，包括操作类型、输入形状、是否原地操作、父节点信息等
            group_key = (
                "batch_" + self.op.__name__.lower().split(".")[0],
                str(input.meta["example_value"].shape),
                str(node.kwargs.get("inplace", False)),
                str(parent),
            )
        else:
            group_key = None
        return group_key
    # 定义一个方法 fuse，用于融合图中的节点子集
    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        # 创建空列表，用于存储批处理的节点、输入和输入的元数据
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []

        # 遍历给定的节点子集
        for node in subset:
            # 将当前节点添加到批处理节点列表中
            batch_nodes.append(node)
            # 获取节点的第一个参数作为输入
            input = get_arg_value(node, 0, "input")
            # 将输入添加到批处理输入列表中
            batch_inputs.append(input)
            # 将输入的示例值元数据添加到批处理输入元数据列表中
            batch_inputs_metadata.append(input.meta["example_value"])

        # 在节点子集的第一个节点之前插入操作
        with graph.inserting_before(subset[0]):
            # 对批处理的输入进行堆叠操作，沿指定维度堆叠
            stack_inputs = graph.call_function(
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            # 更新堆叠后的输入的示例值元数据
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            # 如果操作是 torch.nn.functional.relu，则调用操作并传入堆叠后的输入
            if self.op == torch.nn.functional.relu:
                batch_op = graph.call_function(
                    self.op,
                    args=(stack_inputs,),
                    kwargs={"inplace": subset[0].kwargs.get("inplace", False)},
                )
            else:
                # 否则，调用操作并传入堆叠后的输入
                batch_op = graph.call_function(
                    self.op,
                    args=(stack_inputs,),
                )
            # 对堆叠后的输出进行解绑操作，沿指定维度解绑
            unbind_op = graph.call_function(
                torch.unbind, args=(batch_op,), kwargs={"dim": 0}
            )
            # 遍历批处理节点列表中的节点
            for i, node in enumerate(batch_nodes):
                # 在解绑操作之后插入操作
                with graph.inserting_after(unbind_op):
                    # 获取解绑后的输出的指定项
                    getitem = graph.call_function(operator.getitem, args=(unbind_op, i))
                # 替换当前节点的所有用法为获取的项
                node.replace_all_uses_with(getitem)
                # 更新获取的项的元数据为当前节点的元数据
                getitem.meta.update(node.meta)
                # 从图中擦除当前节点
                graph.erase_node(node)
        
        # 增加与操作相关的计数器，用于记录批处理操作的次数
        counters["inductor"]["batch_" + self.op.__name__.lower().split(".")[0]] += 1
# 批处理逐点操作后传融合工厂，继承自批处理逐点操作融合工厂
class BatchPointwiseOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    """
    批处理逐点操作（例如 sigmoid、relu、tanh）在后向传播阶段的融合。
    引入的堆栈节点可能会在拆分连接中合并。
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        # 获取输入节点
        input = get_arg_value(node, 0, "input")
        
        # 检查节点是否匹配指定的操作，并验证节点元数据是否有效
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            # 对于 relu 操作，也使用 inplace 构建键
            # 我们批处理具有相同父节点的操作，以启用后续的拆分连接
            parent = node.args[0]
            parent = parent.target if self.graph_search_options.get("fuse_nodes_with_same_parent", False) else ""  # type: ignore[union-attr]
            # 构建分组键
            group_key = (
                "batch_aten_" + self.op.__name__.lower().split(".")[0],
                str(input.meta["val"].shape),
                str(node.kwargs.get("inplace", False)),
                # pyre-fixme[16]
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        # 初始化批处理节点、输入和输入元数据列表
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []

        # 遍历子集中的节点
        for node in subset:
            batch_nodes.append(node)
            # 获取节点的输入
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["val"])

        # 在子集的第一个节点之前插入操作
        with graph.inserting_before(subset[0]):
            # 分解堆栈输入
            stack_inputs = decompose_stack(graph, batch_inputs)
            # 更新堆栈示例值
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            # 在图中调用操作
            batch_op = graph.call_function(
                self.op,
                args=(stack_inputs,),
            )
            # 遍历批处理节点列表
            for i, node in enumerate(batch_nodes):
                # 在批处理操作之后插入节点
                with graph.inserting_after(batch_op):
                    getitem = graph.call_function(aten.select, args=(batch_op, 0, i))
                # 替换所有使用节点的引用
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)
        # 增加计数器中的批处理逐点操作计数
        counters["inductor"][
            "batch_aten_" + self.op.__name__.lower().split(".")[0]
        ] += 1


# 注册批处理 tanh 融合类，继承自批处理逐点操作前向传播融合类
@register_fusion("batch_tanh")
class BatchTanhPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.tanh, **kwargs)


# 注册批处理 sigmoid 融合类，继承自批处理逐点操作前向传播融合类
@register_fusion("batch_sigmoid")
class BatchSigmoidPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.sigmoid, **kwargs)


# 注册批处理 relu 融合类，继承自批处理逐点操作前向传播融合类
@register_fusion("batch_relu")
class BatchReLuPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.nn.functional.relu, **kwargs)


# 注册批处理 aten tanh 融合类，继承自批处理逐点操作后向传播融合类
@register_fusion("batch_aten_tanh", pre_grad=False)
class BatchTanhPostGradFusion(BatchPointwiseOpsPostGradFusion):
    # 定义类的初始化方法，接受任意关键字参数
    def __init__(self, **kwargs):
        # 调用父类的初始化方法，使用默认的 arctangent hyperbolic tangent (tanh) 函数
        super().__init__(aten.tanh.default, **kwargs)
@register_fusion("batch_aten_sigmoid", pre_grad=False)
class BatchSigmoidPostGradFusion(BatchPointwiseOpsPostGradFusion):
    # 注册批量的 sigmoid 函数后梯度融合，不需要预先计算梯度
    def __init__(self, **kwargs):
        # 调用父类构造函数，使用默认的 sigmoid 函数
        super().__init__(aten.sigmoid.default, **kwargs)


@register_fusion("batch_aten_relu", pre_grad=False)
class BatchReLuPostGradFusion(BatchPointwiseOpsPostGradFusion):
    # 注册批量的 ReLU 函数后梯度融合，不需要预先计算梯度
    def __init__(self, **kwargs):
        # 调用父类构造函数，使用默认的 ReLU 函数
        super().__init__(aten.relu.default, **kwargs)


@register_fusion("batch_aten_add", pre_grad=False)
class BatchAddPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    # 注册批量的加法后梯度融合，不需要预先计算梯度
    def __init__(self, **kwargs):
        # 调用父类构造函数，使用 Tensor 的加法操作
        super().__init__(aten.add.Tensor, **kwargs)


@register_fusion("batch_aten_sub", pre_grad=False)
class BatchSubPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    # 注册批量的减法后梯度融合，不需要预先计算梯度
    def __init__(self, **kwargs):
        # 调用父类构造函数，使用 Tensor 的减法操作
        super().__init__(aten.sub.Tensor, **kwargs)


@register_fusion("batch_aten_div", pre_grad=False)
class BatchDivPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    # 注册批量的除法后梯度融合，不需要预先计算梯度
    def __init__(self, **kwargs):
        # 调用父类构造函数，使用 Tensor 的除法操作
        super().__init__(aten.div.Tensor, **kwargs)


@register_fusion("batch_aten_mul", pre_grad=False)
class BatchMulPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    # 注册批量的乘法后梯度融合，不需要预先计算梯度
    def __init__(self, **kwargs):
        # 调用父类构造函数，使用 Tensor 的乘法操作
        super().__init__(aten.mul.Tensor, **kwargs)


class _OrderedSet:
    # 有序集合类，基于 OrderedDict 实现
    def __init__(self, param=None):
        if param:
            # 如果有参数传入，则使用参数创建有序字典
            self.rep = OrderedDict(dict.fromkeys(param))
        else:
            # 否则创建一个空的有序字典
            self.rep = OrderedDict()

    def __contains__(self, o):
        # 判断 o 是否在有序字典中
        return o in self.rep

    def __len__(self):
        # 返回有序字典中元素的个数
        return self.rep.__len__()

    def append(self, o):
        # 在有序字典中添加元素 o
        self.rep[o] = None

    def __iter__(self):
        # 返回有序字典键的迭代器
        return self.rep.keys().__iter__()


def find_independent_subset_greedy(
    node_list: Iterable[torch.fx.Node],
    graph_search_options: Dict[str, Any],
) -> Iterator[Iterable[torch.fx.Node]]:
    """
    Yields a list of subsets of `node_list` where no element in the subset
    depends on any other element in the subset. This results in a set of
    independent nodes which can be fused together.

    The order of `node_list` is preserved within each subset so we can benefit
    from split-cat elimination in later passes.

    During iteration it is only safe to mutate the graph by changing the nodes
    that have been returned.

    graph_search_options:
      - min_fuse_set_size: Minimum size of the subset to consider. Subsets below
        this size will be ignored.
      - max_fuse_set_size: Maximum size of the subset to consider. Subsets will
        be broken to be at most this size.
    """

    # 计算 `node_list` 中每个节点的子节点，这些子节点必须是 `interesting_nodes` 的成员
    # 定义一个函数，用于查找节点的依赖节点集合
    def find_dependent_nodes(node, interesting_nodes):
        # visited_node_set 用于存储已访问过的节点集合，初始包含当前节点
        visited_node_set: Set[torch.fx.Node] = {node}
        # dep_set 用于存储依赖节点的集合，初始为空集合
        dep_set: Set[torch.fx.Node] = set()

        # 使用深度优先搜索遍历节点及其依赖节点
        work = [node]
        while work:
            node = work.pop()
            # 遍历当前节点的所有输入节点
            for input_node in node.all_input_nodes:
                # 如果输入节点在感兴趣的节点集合中，则将其添加到依赖集合中
                if input_node in interesting_nodes:
                    dep_set.add(input_node)

                # 如果输入节点尚未访问过，则将其加入 visited_node_set，并继续遍历
                if input_node not in visited_node_set:
                    visited_node_set.add(input_node)
                    work.append(input_node)

        return dep_set

    # 从图搜索选项中获取最小融合集大小和最大融合集大小
    min_fuse_set_size = graph_search_options["min_fuse_set_size"]
    max_fuse_set_size = graph_search_options["max_fuse_set_size"]

    # node_list 需要是一个有序集合，因为我们只跟踪其中剩余的节点
    # （我们希望在集合上执行 `in` 操作，而不是在列表上）但我们希望保持正确的顺序。
    node_list = _OrderedSet(node_list)

    # cache 用于存储节点到其依赖节点集合的映射
    cache: Dict[torch.fx.Node, Set[torch.fx.Node]] = {}

    # 循环处理节点列表直到为空
    while node_list:
        subset: List[torch.fx.Node] = []      # 用于存储当前子集的节点列表
        subset_deps: Set[torch.fx.Node] = set()  # 用于存储当前子集的依赖节点集合

        next_round_node_list = _OrderedSet()  # 用于存储下一轮处理的节点列表

        # 遍历当前节点列表
        for node in node_list:
            # 如果当前子集大小已经达到最大融合集大小或当前节点已经在依赖集合中，则跳过该节点
            if len(subset) >= max_fuse_set_size or node in subset_deps:
                next_round_node_list.append(node)
                continue

            # 尝试从缓存中取出当前节点的依赖集合
            dep_set = cache.pop(node, None)
            if dep_set is None:
                # 如果缓存中不存在，则调用 find_dependent_nodes 函数查找依赖节点集合
                dep_set = find_dependent_nodes(node, node_list)

            # 如果当前节点的依赖集合与当前子集没有交集，则将该节点加入子集并更新依赖集合
            if not dep_set.intersection(subset):
                subset.append(node)
                subset_deps.update(dep_set)
            else:
                # 否则将该节点加入下一轮处理的节点列表，并将其依赖集合缓存起来
                next_round_node_list.append(node)
                cache[node] = dep_set

        # 如果当前子集大小达到最小融合集大小，则返回当前子集作为结果
        if len(subset) >= min_fuse_set_size:
            # 注意：调用者将使用这些子集来合并节点，因此需要清除任何包含返回节点的缓存条目，
            # 因为合并后的依赖列表可能会有所不同（更大）。
            cache = {k: v for k, v in cache.items() if v.isdisjoint(subset)}
            yield subset

        # 更新节点列表为下一轮处理的节点列表
        node_list = next_round_node_list
# 定义一个函数，用于获取指定规则的融合候选节点集合，从根节点开始进行 BFS 搜索。
# 返回一个默认字典，键是任意类型，值是包含 torch.fx.Node 的列表
def get_fusion_candidates(
    rule: GroupBatchFusionBase, root_node: torch.fx.Node, fused_set: Set[torch.fx.Node]
) -> DefaultDict[Any, List[torch.fx.Node]]:
    """
    从根节点开始使用 BFS 搜索，查找特定规则的融合候选节点。
    只在图搜索选项 graph_search_options["max_fuse_search_depth"] 内搜索子图。
    """
    # 使用双端队列初始化 BFS 队列，存储搜索深度和节点元组
    q: Deque[Tuple[int, torch.fx.Node]] = collections.deque()

    # 使用默认字典初始化候选节点字典
    candidate_dict: DefaultDict[Any, List[torch.fx.Node]] = collections.defaultdict(list)

    # 如果根节点的目标在搜索排除列表 SEARCH_EXCLUSIONS 中，直接返回空的候选节点字典
    if root_node.target in SEARCH_EXCLUSIONS:
        return candidate_dict

    # 初始化访问过的节点集合
    visited_set: Set[torch.fx.Node] = set()

    # 将根节点的所有输入节点加入 BFS 队列和访问集合
    for next_node in root_node.all_input_nodes:
        q.append((1, next_node))
        visited_set.add(next_node)

    # 开始 BFS 遍历
    while len(q) > 0:
        depth, node = q.popleft()

        # 如果节点已经在融合集合中，则跳过
        if node in fused_set:
            continue

        # 使用规则对象对当前节点进行匹配
        key = rule.match(node)
        if key is not None:
            # 如果匹配成功，则将节点添加到对应键的候选节点列表中
            candidate_nodes = candidate_dict[key]
            if node not in candidate_nodes:
                candidate_nodes.append(node)
        else:
            # 如果匹配不成功且深度未达到最大搜索深度，则继续向下搜索
            if depth < rule.graph_search_options["max_fuse_search_depth"]:
                for next_node in node.all_input_nodes:
                    if next_node not in visited_set:
                        visited_set.add(next_node)
                        q.append((depth + 1, next_node))

    # 返回最终的候选节点字典
    return candidate_dict


# 定义一个函数，将图中的节点按稳定拓扑排序排序
def apply_group_batch_fusion(graph: torch.fx.GraphModule, rule: GroupBatchFusionBase):
    stable_topological_sort(graph)  # type: ignore[arg-type]
    # 初始化一个空的节点集合，用于存储已经融合过的节点
    fused_set: Set[torch.fx.Node] = set()
    # 初始化一个标志，指示是否需要将日志上传到 Scuba
    log_to_scuba = False

    # 从图的最后一个节点开始向前遍历
    for node in reversed(graph.nodes):
        # 获取当前节点符合规则的融合候选节点
        candidates = get_fusion_candidates(rule, node, fused_set)

        # 遍历候选节点字典中的键值对
        for key, candidate_nodes in candidates.items():
            # 如果候选节点的数量小于最小融合集大小，则跳过
            if len(candidate_nodes) < rule.graph_search_options["min_fuse_set_size"]:
                continue

            # 使用贪婪算法找到独立子集
            for subset in find_independent_subset_greedy(
                candidate_nodes, rule.graph_search_options
            ):
                # 对子集进行规则定义的融合操作
                rule.fuse(graph, subset)
                # 更新已融合节点集合
                fused_set.update(subset)
                # 记录调试日志，标记需要上传到 Scuba
                log.debug(
                    f"{rule.__class__.__name__}: key = {key}; subset size = {len(list(subset))}"  # noqa: G004
                )
                log_to_scuba = True

    # 如果有需要上传到 Scuba 的日志，则将图上传到 Optimus 平台
    if log_to_scuba:
        optimus_scuba_log[rule.__class__.__name__] = upload_graph(graph)


# 定义一个函数，根据配置选项生成融合规则列表
def generate_fusion_from_config(config_options: Dict[str, Any], pre_grad=True):
    # 初始化一个空的融合规则列表
    fusions: List[GroupBatchFusionBase] = []

    # 遍历配置选项中的每个名称和对应选项
    for name, options in config_options.items():
        # 跳过来自 pattern_matcher passes 的所有模式（例如 split_cat）
        if name not in PRE_GRAD_FUSIONS and name not in POST_GRAD_FUSIONS:
            continue
        # 根据前向或后向梯度选择相应的融合类
        fusion_cls = PRE_GRAD_FUSIONS[name] if pre_grad else POST_GRAD_FUSIONS[name]
        # 复制图搜索选项并更新为当前融合选项
        _options = graph_search_options.copy()
        _options.update(options)
        # 创建融合规则对象并添加到融合列表中
        fusions.append(fusion_cls(graph_search_options=_options))  # type: ignore[operator]
    # 返回变量 fusions，这是函数的返回值，即合并后的列表
    return fusions
def group_batch_fusion_passes(graph: torch.fx.Graph, pre_grad=True):
    fusions: List[GroupBatchFusionBase] = []
    # 我们保留所有当前的预梯度融合以保持当前的实现，稍后会删除此部分

    # 如果 pre_grad 为 True，则生成并添加预梯度融合选项到 fusions 列表中
    if pre_grad:
        fusions += generate_fusion_from_config(
            config.pre_grad_fusion_options, pre_grad=True
        )
    else:
        # 获取需要使用 fbgemm 的后梯度融合选项的键列表
        fbgemm_fusion_keys = [
            x
            for x in config.post_grad_fusion_options
            if config.post_grad_fusion_options[x].get("require_fbgemm", False)
        ]
        # 根据 fbgemm_fusion_keys 构建包含 fbgemm 融合选项的字典
        fbgemm_fusions = {
            fusion: config.post_grad_fusion_options[fusion]
            for fusion in fbgemm_fusion_keys
        }
        # 构建不包含 fbgemm 融合选项的字典
        non_fbgemm_fusions = {
            fusion: config.post_grad_fusion_options[fusion]
            for fusion in config.post_grad_fusion_options.keys()
            if fusion not in fbgemm_fusion_keys
        }
        # 生成并添加非 fbgemm 融合选项到 fusions 列表中
        fusions += generate_fusion_from_config(non_fbgemm_fusions, pre_grad=False)
        # 如果存在 fbgemm，生成并添加 fbgemm 融合选项到 fusions 列表中
        if has_fbgemm:
            fusions += generate_fusion_from_config(fbgemm_fusions, pre_grad=False)

    # 对于每个生成的融合规则，应用到图中
    for i, rule in enumerate(fusions):
        # 使用 GraphTransformObserver 进行图变换观察，为每个融合规则创建一个命名空间
        with GraphTransformObserver(
            graph.owning_module,
            f"group_batch_fusion_{i}",
            config.trace.log_url_for_graph_xform,
        ):
            # 应用融合规则到图中
            apply_group_batch_fusion(graph, rule)  # type: ignore[arg-type]
```