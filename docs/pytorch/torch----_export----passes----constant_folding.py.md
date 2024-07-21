# `.\pytorch\torch\_export\passes\constant_folding.py`

```
# 设置 mypy：允许未标注的定义
import collections  # 导入 collections 模块
from collections import defaultdict  # 从 collections 模块中导入 defaultdict 类
from typing import Any, Callable, Dict, Optional  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
import torch.utils._pytree as pytree  # 导入 PyTorch 中的 _pytree 模块

aten = torch.ops.aten  # 设置变量 aten，指向 torch.ops.aten

# 将模块分为两个子图，以便正确进行运行时权重更新。
# 使用案例和更多信息可以在以下链接找到：
# https://docs.google.com/document/d/1inZC-8KarJ6gKB7G9egmYLx1V_dKX_apxon0w4zPC0Q/edit?usp=sharing
META_TAG = "MODULE_TYPE"  # 定义常量 META_TAG，表示模块类型
MODULE_TAG = "_MAIN_MODULE"  # 定义常量 MODULE_TAG，表示主模块
CONST_MODULE_TAG = "_CONST_MODULE"  # 定义常量 CONST_MODULE_TAG，表示常量模块


def replace_node_with_constant(gm, node, constant, name=None):
    """
    将节点替换为常量，并注册为 buffer。

    Args:
    - gm: 图模块对象
    - node: 要替换的节点
    - constant: 替换节点的常量值
    - name: 用于注册的名称，默认为 None

    Returns:
    - None
    """
    g = gm.graph  # 获取图模块的图对象

    if name:
        qualname = name  # 如果指定了 name 参数，则使用该名称
    else:
        if not hasattr(gm, "_frozen_param_count"):
            gm._frozen_param_count = 0
        i = gm._frozen_param_count

        while True:
            qualname = f"_frozen_param{i}"
            if not hasattr(gm, qualname):
                break
            i += 1

        gm._frozen_param_count = i + 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})  # 创建新的 get_attr 节点
        node.replace_all_uses_with(new_input_node)  # 替换所有使用该节点的地方为新节点
        new_input_node.meta.update(node.meta)  # 更新新节点的元数据信息
        g.erase_node(node)  # 删除原节点

    # 需要注册为 buffer，以抑制 `does not reference an nn.Module, nn.Parameter, or buffer` 警告
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


class ConstantFolder(torch.fx.Interpreter):
    """
    常量折叠器，继承自 torch.fx.Interpreter。

    Args:
    - gm: 图模块对象
    - skip_constructors: 是否跳过构造函数，默认为 False

    Attributes:
    - node_replacements: 节点替换字典
    - replaced_uses: 替换使用次数统计字典
    - unknown_value: 未知值对象
    - skip_constructors: 是否跳过构造函数标志

    Methods:
    - is_impure(node): 判断节点是否不纯
    """
    def __init__(
        self,
        gm,
        skip_constructors=False,
    ):
        super().__init__(gm)
        self.node_replacements: Dict[torch.fx.Node, Any] = {}  # 初始化节点替换字典
        self.replaced_uses: Dict[torch.fx.Node, int] = collections.Counter()  # 初始化替换使用次数统计
        self.unknown_value = object()  # 初始化未知值对象
        self.skip_constructors: bool = skip_constructors  # 初始化是否跳过构造函数标志

        # 覆盖此方法以释放环境值，如果它们的唯一剩余使用是输出
        self.user_to_last_uses = self.node_to_last_non_output_use()  # 初始化用户到最后非输出使用的映射

    def is_impure(self, node: torch.fx.node.Node):
        """
        判断节点是否不纯，即它是否会产生副作用。

        Args:
        - node: 要检查的节点对象

        Returns:
        - bool: 如果节点不纯返回 True，否则返回 False
        """
        if (
            node.target == torch.ops.prims.convert_element_type.default
            and node.args[0].op == "get_attr"  # type: ignore[union-attr]
            and node.args[0].meta["val"].dtype == torch.int8  # type: ignore[union-attr]
            and node.args[1] == torch.bfloat16
        ):
            # 对于 int8_weight -> dq -> bf16_weight 的情况，视为不纯
            return True
        if node.target in [
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
        ]:
            # 对于 fp32_weight -> q -> dq 的模式，仅折叠 fp32_weight -> q
            # 并保留 dq 在图中进行融合
            return True
        return False
    # 返回一个字典，键为每个节点，值为节点的最后一个非输出用途的列表
    def node_to_last_non_output_use(self):
        # 创建一个默认字典，值为列表，用于存储每个节点的最后一个非输出用途
        last_non_output_use = collections.defaultdict(list)
        # 记录已经遇到的使用过的输入
        seen_uses = set()
        # 获取图中的最后一个节点，通常是输出节点
        output_node = next(iter(reversed(self.module.graph.nodes)))

        # 反向遍历图中的节点
        for node in reversed(self.module.graph.nodes):
            # 如果节点的目标是 "output"，则跳过
            if node.target == "output":
                continue

            # 定义一个函数，用于将节点的输入添加到最后非输出用途中
            def add_use(inp):
                # 如果输入已经在已见过的使用集合中，则返回
                if inp in seen_uses:
                    return

                # 将输入添加到已见过的使用集合中
                seen_uses.add(inp)
                # 将输入添加到当前节点的最后非输出用途列表中
                last_non_output_use[node].append(inp)

            # 对节点的参数和关键字参数应用函数 add_use，只映射 torch.fx.Node 类型的对象
            pytree.tree_map_only_(torch.fx.Node, add_use, (node.args, node.kwargs))

            # 如果当前节点只在输出节点中被使用，我们希望立即进行垃圾收集
            if len(node.users) == 1 and output_node in node.users:
                # 将当前节点添加到其最后非输出用途列表中
                last_non_output_use[node].append(node)

        # 返回存储节点最后非输出用途的字典
        return last_non_output_use

    # 检查是否可以插入张量，始终返回 True
    def insertable_tensor_check(self, tensor: torch.Tensor) -> bool:
        return True

    # 将节点替换为给定张量
    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor

    # 运行方法，初始化环境并返回结果
    def run(self):
        # 创建一个空的环境字典
        env = {}
        # 遍历图中的所有占位符节点，将其初始化为未知值
        for n in self.module.graph.find_nodes(op="placeholder"):
            env[n] = self.unknown_value
        # 调用父类的 run 方法，传入初始化的环境，并返回结果
        return super().run(initial_env=env)
# 禁用当前模式装饰器，确保常量折叠过程中不会修改当前模式
@torch.utils._python_dispatch._disable_current_modes()
def constant_fold(gm, constraint_fn: Optional[Callable[[torch.fx.Node], bool]] = None):
    # 创建常量折叠器对象，跳过构造函数
    cf = ConstantFolder(gm, skip_constructors=True)
    # 运行常量折叠过程
    cf.run()

    # 遍历常量折叠器替换字典中的节点和常量
    for node, constant in cf.node_replacements.items():
        # 如果存在约束函数且节点不符合约束条件，则继续下一个节点
        if constraint_fn is not None and not constraint_fn(node):
            continue
        # 替换节点为对应的常量值
        replace_node_with_constant(gm, node, constant)

    # 存储被删除的参数列表
    erased_params = []

    # 根据节点的使用者信息，获取所有属性节点的用户
    # 由于在本例中 _tensor_constant0 和 _tensor_constant0_1 实际上引用了相同的张量，因此不直接使用 node.users 查找
    get_attr_node_users = defaultdict(list)
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            get_attr_node_users[node.target].extend(node.users.keys())

    # 遍历图中所有的 get_attr 节点
    for node in gm.graph.find_nodes(op="get_attr"):
        if node.op == "get_attr" and len(get_attr_node_users[node.target]) == 0:
            # 如果在 gm 中存在 node.target 对应的属性，则删除该属性
            if hasattr(gm, node.target):
                delattr(gm, node.target)
            # 将该节点添加到被删除参数列表中
            erased_params.append(node)

    # 遍历被删除参数列表中的节点，从图中擦除这些节点
    for node in erased_params:
        gm.graph.erase_node(node)

    # 消除死代码
    gm.graph.eliminate_dead_code()
    # 检查图的一致性
    gm.graph.lint()
    # 重新编译图模块
    gm.recompile()


# 禁用当前模式装饰器，标记常量图中的节点
@torch.utils._python_dispatch._disable_current_modes()
def constant_graph_tag(gm: torch.fx.GraphModule):
    # 创建常量折叠器对象，跳过构造函数
    cf = ConstantFolder(gm, skip_constructors=True)
    # 运行常量折叠过程
    cf.run()

    # 遍历图中的所有节点
    for node in gm.graph.nodes:
        # 如果节点是 get_attr、在节点替换字典中、或在替换使用字典中，则将节点标记为 CONST_MODULE_TAG
        if (
            node.op == "get_attr"
            or node in cf.node_replacements
            or node in cf.replaced_uses
        ):
            node.meta[META_TAG] = CONST_MODULE_TAG
        else:
            # 否则将节点标记为 MODULE_TAG
            node.meta[META_TAG] = MODULE_TAG


def run_and_get_constant_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    构建一个对应于在给定 gm 中可以常量折叠部分的 GraphModule。
    """
    # 对给定的 gm 标记常量图中的节点
    constant_graph_tag(gm)
    # 如果直接使用的是常量而不是通过折叠的机会，我们将保留它在主 gm 中
    # 遍历给定的图中的节点，查找操作为 "get_attr" 的节点
    for node in gm.graph.find_nodes(op="get_attr"):
        # 标记该节点是否被用于折叠（优化）
        used_to_fold = False
        # 检查该节点的每一个使用者
        for u in node.users:
            # 如果使用者的元数据中包含特定的标记 CONST_MODULE_TAG
            if u.meta[META_TAG] == CONST_MODULE_TAG:
                # 设置节点被用于折叠
                used_to_fold = True
                break
        # 如果节点未被用于折叠，则更新节点的元数据为 MODULE_TAG
        if not used_to_fold:
            node.meta[META_TAG] = MODULE_TAG
    
    # 创建一个新的空图
    new_graph = torch.fx.Graph()
    
    # 用于节点映射的字典，将旧图中的节点映射到新图中的节点
    node_remapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    # 用于保存新图的输出节点
    output_nodes = []
    
    # 遍历给定图中的每一个节点
    for node in gm.graph.nodes:
        # 如果节点的元数据为 MODULE_TAG，则跳过该节点
        if node.meta[META_TAG] == MODULE_TAG:
            continue
    
        # 复制当前节点到新图中，并建立节点映射关系
        new_node = new_graph.node_copy(node, lambda x: node_remapping[x])
        node_remapping[node] = new_node
    
        # 检查当前节点的每一个使用者
        for user in node.users:
            # 如果使用者的元数据为 MODULE_TAG
            if user.meta[META_TAG] == MODULE_TAG:
                # 将新节点添加到输出节点列表中
                output_nodes.append(new_node)
                break
    
    # 设置新图的输出节点为输出节点列表中的节点组成的元组
    new_graph.output(tuple(output_nodes))
    # 对新图进行 lint（代码检查）
    new_graph.lint()
    
    # 使用新图创建一个新的图模块
    new_gm = torch.fx.GraphModule(gm, new_graph)
    
    # 返回新创建的图模块
    return new_gm
```