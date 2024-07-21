# `.\pytorch\torch\_inductor\constant_folding.py`

```
# mypy: allow-untyped-defs
# 引入 collections 模块
import collections
# 从 typing 模块中引入需要的类型提示
from typing import Any, Callable, Dict, Optional

# 引入 PyTorch 库
import torch
# 导入 PyTorch 内部模块 _pytree
import torch.utils._pytree as pytree

# 使用 torch.ops.aten 别名为 aten
aten = torch.ops.aten

# 定义三个常量字符串用于标记不同的模块类型
META_TAG = "MODULE_TYPE"
MODULE_TAG = "_MAIN_MODULE"
CONST_MODULE_TAG = "_CONST_MODULE"

# 定义函数 replace_node_with_constant，用于替换图中节点为常量
def replace_node_with_constant(gm, node, constant, name=None):
    g = gm.graph

    # 如果提供了名字，则使用提供的名字
    if name:
        qualname = name
    else:
        # 如果 gm 没有 _frozen_param_count 属性，则初始化为 0
        if not hasattr(gm, "_frozen_param_count"):
            gm._frozen_param_count = 0
        i = gm._frozen_param_count

        # 循环直到找到一个可用的名字
        while True:
            qualname = f"_frozen_param{i}"
            if not hasattr(gm, qualname):
                break
            i += 1

        # 更新 _frozen_param_count 属性
        gm._frozen_param_count = i + 1

    # 在节点之前插入新节点
    with g.inserting_before(node):
        # 创建一个新的 get_attr 节点
        new_input_node = g.create_node("get_attr", qualname, (), {})
        # 替换原始节点的所有使用为新节点
        node.replace_all_uses_with(new_input_node)
        # 更新新节点的元数据
        new_input_node.meta.update(node.meta)
        # 删除原始节点
        g.erase_node(node)

    # 注册常量为 buffer，用于抑制警告
    gm.register_buffer(qualname, constant)
    # 将常量设置为 gm 对象的属性
    setattr(gm, qualname, constant)

# 定义 ConstantFolder 类，继承自 torch.fx.Interpreter
class ConstantFolder(torch.fx.Interpreter):
    def __init__(
        self,
        gm,
        skip_constructors=False,
    ):
        super().__init__(gm)
        # 初始化节点替换字典
        self.node_replacements: Dict[torch.fx.Node, Any] = {}
        # 初始化替换使用计数器
        self.replaced_uses: Dict[torch.fx.Node, int] = collections.Counter()
        # 初始化未知值
        self.unknown_value = object()
        # 是否跳过构造函数的标志
        self.skip_constructors: bool = skip_constructors

        # 重写以处理仅剩输出的环境值的释放
        self.user_to_last_uses = self.node_to_last_non_output_use()

    # 判断节点是否为不纯的方法
    def is_impure(self, node: torch.fx.node.Node):
        if (
            # 判断是否为将 int8 转换为 bfloat16 的操作
            node.target == torch.ops.prims.convert_element_type.default
            and node.args[0].op == "get_attr"  # type: ignore[union-attr]
            and node.args[0].meta["val"].dtype == torch.int8  # type: ignore[union-attr]
            and node.args[1] == torch.bfloat16
        ):
            # 对于 int8_weight -> dq -> bf16_weight 的模式，视为不纯
            return True
        if node.target in [
            # 判断是否为量化分解的反量化操作
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
        ]:
            # 对于 fp32_weight -> q -> dq 的模式，视为不纯
            # 只折叠 fp32_weight -> q 部分，保留 dq 在图中以进行融合
            return True
        # 默认为纯操作
        return False
    # 返回一个字典，记录每个节点最后一次非输出使用的位置
    def node_to_last_non_output_use(self):
        # 使用 defaultdict 创建一个字典，值为列表，用于记录每个节点的最后一次非输出使用的位置
        last_non_output_use = collections.defaultdict(list)
        # 用于记录已经遍历过的使用位置，避免重复记录
        seen_uses = set()
        # 获取图中的输出节点
        output_node = next(iter(reversed(self.module.graph.nodes)))

        # 反向遍历图中的节点
        for node in reversed(self.module.graph.nodes):
            # 如果节点目标是 "output"，则跳过
            if node.target == "output":
                continue

            # 定义一个函数，将节点的输入作为使用位置添加到 last_non_output_use 中
            def add_use(inp):
                # 如果输入已经在 seen_uses 中，直接返回
                if inp in seen_uses:
                    return
                # 将输入添加到 seen_uses 中
                seen_uses.add(inp)
                # 将输入添加到当前节点的最后一次非输出使用位置列表中
                last_non_output_use[node].append(inp)

            # 调用 pytree.tree_map_only_ 方法，对节点的参数和关键字参数应用 add_use 函数
            # 注意：这里的操作不会改变节点本身，因为 Node 对象是不可变的
            pytree.tree_map_only_(torch.fx.Node, add_use, (node.args, node.kwargs))

            # 如果该节点只在输出节点中使用，则将其添加到最后一次非输出使用位置列表中
            if len(node.users) == 1 and output_node in node.users:
                last_non_output_use[node].append(node)

        # 返回记录了每个节点最后一次非输出使用位置的字典
        return last_non_output_use

    # 对给定的 torch.Tensor 进行可插入性检查，始终返回 True
    def insertable_tensor_check(self, tensor: torch.Tensor) -> bool:
        return True

    # 将指定的节点和对应的 tensor 添加到节点替换字典中
    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor

    # 运行方法，初始化环境变量 env，将所有占位符节点设置为未知值
    def run(self):
        env = {}
        # 遍历图中所有操作为 "placeholder" 的节点，将其对应的 env 值设为未知值
        for n in self.module.graph.find_nodes(op="placeholder"):
            env[n] = self.unknown_value
        # 调用父类的 run 方法，传入初始化的环境变量 env，并返回运行结果
        return super().run(initial_env=env)
# 禁用当前的 TorchScript 模式，以便在常量折叠期间不触发不必要的模式检查
@torch.utils._python_dispatch._disable_current_modes()
def constant_fold(gm, constraint_fn: Optional[Callable[[torch.fx.Node], bool]] = None):
    # 创建常量折叠器对象，跳过构造函数
    cf = ConstantFolder(gm, skip_constructors=True)
    # 执行常量折叠操作
    cf.run()

    # 遍历被替换为常量的节点，并进行替换
    for node, constant in cf.node_replacements.items():
        # 如果定义了约束函数并且当前节点不满足约束，则继续下一个节点
        if constraint_fn is not None and not constraint_fn(node):
            continue
        # 替换节点为常量
        replace_node_with_constant(gm, node, constant)

    # 存储被删除的参数节点
    erased_params = []
    # 遍历图中所有的 "get_attr" 节点
    for node in gm.graph.find_nodes(op="get_attr"):
        # 如果节点没有被使用，则删除相关的属性
        if len(node.users) == 0:
            if hasattr(gm, node.target):
                delattr(gm, node.target)
            # 记录被删除的节点
            erased_params.append(node)

    # 删除所有被标记为删除的节点
    for node in erased_params:
        gm.graph.erase_node(node)

    # 消除死代码以及对图进行静态分析
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    # 重新编译图模块
    gm.recompile()


# 为给定的 TorchFX 图模块标记常量节点
@torch.utils._python_dispatch._disable_current_modes()
def constant_graph_tag(gm: torch.fx.GraphModule):
    # 创建常量折叠器对象，跳过构造函数
    cf = ConstantFolder(gm, skip_constructors=True)
    # 执行常量折叠操作
    cf.run()

    # 遍历图中的所有节点
    for node in gm.graph.nodes:
        # 如果节点是 "get_attr" 操作，或者节点被替换为常量，或者节点被替换使用
        if (
            node.op == "get_attr"
            or node in cf.node_replacements
            or node in cf.replaced_uses
        ):
            # 将节点的元信息标记为 CONST_MODULE_TAG
            node.meta[META_TAG] = CONST_MODULE_TAG
        else:
            # 否则将节点的元信息标记为 MODULE_TAG
            node.meta[META_TAG] = MODULE_TAG


# 运行并获取常量图模块
def run_and_get_constant_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    构造一个与提供的 gm 中可以常量折叠的部分对应的 GraphModule。
    """

    # 标记常量图中的节点
    constant_graph_tag(gm)
    # 重新编写标记，如果是直接使用常量而没有折叠机会，则保留在主 gm 中
    for node in gm.graph.find_nodes(op="get_attr"):
        used_to_fold = False
        for u in node.users:
            if u.meta[META_TAG] == CONST_MODULE_TAG:
                used_to_fold = True
                break
        if not used_to_fold:
            # 将节点的元信息标记为 MODULE_TAG
            node.meta[META_TAG] = MODULE_TAG

    # 创建一个新的图对象
    new_graph = torch.fx.Graph()

    # 节点映射字典
    node_remapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    output_nodes = []
    # 遍历图中的所有节点
    for node in gm.graph.nodes:
        # 如果节点的元信息标记为 MODULE_TAG，则跳过
        if node.meta[META_TAG] == MODULE_TAG:
            continue

        # 在新图中复制节点，并建立节点的映射关系
        new_node = new_graph.node_copy(node, lambda x: node_remapping[x])
        node_remapping[node] = new_node

        # 遍历节点的用户
        for user in node.users:
            if user.meta[META_TAG] == MODULE_TAG:
                output_nodes.append(new_node)
                break

    # 设置新图的输出节点
    new_graph.output(tuple(output_nodes))
    # 对新图进行静态分析
    new_graph.lint()
    # 创建新的图模块对象
    new_gm = torch.fx.GraphModule(gm, new_graph)

    # 返回新的图模块对象
    return new_gm
```