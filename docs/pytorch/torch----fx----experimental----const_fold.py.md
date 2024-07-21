# `.\pytorch\torch\fx\experimental\const_fold.py`

```
# mypy: allow-untyped-defs
# 引入正则表达式模块和类型提示模块
import re
from typing import Callable, Dict, Optional, Set, Union

# 引入torch的fx模块及其相关组件
import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module

# 导出的类和函数名
__all__ = ['FoldedGraphModule', 'get_unique_attr_name_in_module', 'split_const_subgraphs']

class FoldedGraphModule(torch.fx.GraphModule):
    """
    FoldedGraphModule是一个GraphModule，它还包含另一个const_subgraph_module，
    代表具有所有const attr输入的子图，可以在运行主标准图之前运行一次。
    const_output_names是attrs的有序列表名，表示const_subgraph的每个相应输出应设置在哪些attrs上。
    """

    def __init__(
        self,
        root: torch.nn.Module,
        graph: torch.fx.Graph,
        const_subgraph: Optional[torch.fx.Graph] = None,
        fx_const_folded_attrs_name: Optional[str] = None,
        device_for_folded_attrs: str = "cuda",
    ):
        # 调用父类构造函数初始化
        super().__init__(root, graph)
        
        # 初始化const_subgraph_module，如果没有给定const_subgraph则为None，否则将其封装为GraphModule对象
        self.const_subgraph_module = (
            None
            if const_subgraph is None
            else torch.fx.GraphModule(root, const_subgraph)
        )
        
        # 标记是否已经运行了折叠操作的状态
        self.has_folding_been_run = False
        
        # 存储fx_const_folded_attrs_name，表示折叠后的属性名称
        self.fx_const_folded_attrs_name = fx_const_folded_attrs_name
        
        # 设备名称，默认为cuda
        self.device_for_folded_attrs = device_for_folded_attrs

    def __call__(self, *args, **kwargs):
        # 如果还没有运行过折叠操作，则首先运行折叠操作
        if not self.has_folding_been_run:
            self.run_folding()
        # 调用父类的__call__方法执行forward过程
        return super().__call__(*args)

    def run_folding(self):
        # 如果没有const_subgraph_module或fx_const_folded_attrs_name，直接返回，没有折叠操作可执行
        if (
            self.const_subgraph_module is None
            or self.fx_const_folded_attrs_name is None
        ):
            return
        
        # 断言确保还没有运行过折叠操作
        assert not self.has_folding_been_run
        # 标记为已运行过折叠操作
        self.has_folding_been_run = True

        # 运行const_subgraph_module得到折叠后的属性
        folded_attrs = self.const_subgraph_module()

        # 定义一个函数，根据输入创建torch.nn.Parameter对象
        def _create_param(i):
            return torch.nn.Parameter(
                # 如果输入不是整数，则克隆其数据并去除梯度信息
                i.detach().clone() if not isinstance(i, int)
                # 如果是整数，则创建一个对应的Tensor，并指定设备为device_for_folded_attrs
                else torch.Tensor([i]).to(device=self.device_for_folded_attrs),
                # 是否需要梯度，对于Tensor类型的输入保留其原有梯度属性，对于整数则不需要梯度
                requires_grad=i.requires_grad if isinstance(i, torch.Tensor) else False,
            )

        # 根据folded_attrs的类型创建对应的参数对象
        params = (
            torch.nn.ParameterList([_create_param(i) for i in folded_attrs])
            if isinstance(folded_attrs, tuple)
            else _create_param(folded_attrs)
        )
        
        # 将创建的参数对象设置为当前对象的属性，属性名称由fx_const_folded_attrs_name指定
        setattr(self, self.fx_const_folded_attrs_name, params)


def _inline_module(gm: torch.fx.GraphModule, inline_mod_name: str):
    """
    给定gm和一个被称为inline_mod_name的图模块，
    ```
    # 将调用的图模块中的所有节点内联到 `gm` 中。
    """
    # 获取要内联到 `gm` 中的内部图模块。
    inline_mod = dict(gm.named_modules())[inline_mod_name]
    assert isinstance(inline_mod, torch.fx.GraphModule)
    
    # 查找要替换的调用模块节点。
    call_mod_node_to_replace = None
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target == inline_mod_name:
            call_mod_node_to_replace = node
            break
    assert call_mod_node_to_replace is not None

    # 开始实际的替换操作。注意我们必须跟踪被复制到 `gm` 中的新节点，通过 replacement_mapping 完成。
    call_mod_args = call_mod_node_to_replace.args
    replacement_mapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    ph_count = 0

    def replacement_fn(node):
        new_node = replacement_mapping[node]
        new_node.meta = node.meta.copy()
        return new_node

    # 遍历内联模块的节点进行处理。
    for inline_node in inline_mod.graph.nodes:
        if inline_node.op == "placeholder":
            replacement_mapping[inline_node] = call_mod_args[ph_count]
            ph_count += 1
            continue

        if inline_node.op == "output":
            # 处理输出节点，替换所有使用该节点的地方。
            outputs = inline_node.args[0]
            output_replacements = map_arg(outputs, replacement_fn)
            call_mod_node_to_replace.replace_all_uses_with(output_replacements)
            continue

        # 在 `call_mod_node_to_replace` 之前插入新节点。
        with gm.graph.inserting_before(call_mod_node_to_replace):
            new_node = gm.graph.node_copy(inline_node, replacement_fn)
        replacement_mapping[inline_node] = new_node

    # 消除无用代码。
    gm.graph.eliminate_dead_code()
# 确保名称在模块中是唯一的，并可以表示一个属性
def get_unique_attr_name_in_module(mod_traced: torch.fx.GraphModule, name: str) -> str:
    # 删除所有在Python标识符中非法的字符
    name = re.sub("[^0-9a-zA-Z_]+", "_", name)
    # 如果名称以数字开头，则在前面加上下划线
    if name[0].isdigit():
        name = f"_{name}"
    # 确保名称在模块中是唯一的，通过增加后缀值实现
    while hasattr(mod_traced, name):
        match = re.match(r"(.*)_(\d+)$", name)
        if match is None:
            name = name + "_1"
        else:
            base, num = match.group(1, 2)
            name = f"{base}_{int(num) + 1}"

    return name


# 将模块中所有具有全常量属性输入的节点分离出来，形成常量子图，并返回一个FoldedGraphModule
def split_const_subgraphs(
    module: Union[torch.nn.Module, torch.fx.GraphModule],
    skip_folding_node_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
    device_for_folded_attrs: str = "cpu",
) -> FoldedGraphModule:
    # 如果module不是torch.fx.GraphModule类型，则进行符号跟踪
    if not isinstance(module, torch.fx.GraphModule):
        mod_traced = torch.fx.symbolic_trace(module)
    else:
        mod_traced = module

    # 构建常量节点列表，这些节点本身是get_attrs，或者其所有输入都是常量节点
    const_nodes: Set[torch.fx.Node] = set()
    found_const_folding = False
    for node in mod_traced.graph.nodes:
        # 跳过占位符/输出节点，因为它们不能进行常量折叠，也不希望给它们添加标签
        if node.op in {"placeholder", "output"}:
            continue

        # 如果节点本身是常量，或者其所有输入都是常量，则将其标记为常量节点
        if node.op != "get_attr" and not set(node.all_input_nodes).issubset(
            const_nodes
        ):
            continue

        # 如果提供的跳过折叠函数说要跳过，则跳过
        if skip_folding_node_fn and skip_folding_node_fn(node):
            continue

        # 跳过具有副作用的函数
        if node.is_impure():
            continue

        # 到这一步必须是可以常量折叠的节点
        const_nodes.add(node)
        if node.op != "get_attr":
            found_const_folding = True

    # 如果没有找到任何常量折叠，则提前返回没有常量折叠子图的FoldedGraphModule
    if not found_const_folding:
        return FoldedGraphModule(mod_traced, mod_traced.graph)

    # 将模块分成两部分：submod_0用于常量折叠子图，submod_1用于其余部分
    def mod_partition(node: torch.fx.Node):
        return 0 if node in const_nodes else 1

    split = split_module(mod_traced, module, mod_partition)
    const_gm, non_const_gm = split.submod_0, split.submod_1
    const_mod_name, non_const_mod_name = "submod_0", "submod_1"

    # 在 split 模块中，将非常量模块和常量模块分别赋给 const_gm 和 non_const_gm
    # 同时设置常量模块和非常量模块的名称
    # split.submod_0 和 split.submod_1 是来自于 split 模块中的两个子模块

    for node in non_const_gm.graph.nodes:
        # 遍历非常量模块的计算图中的每个节点
        if node.op == "call_module":
            # 如果节点操作为 "call_module"
            # 将非常量模块中的对应模块赋给 split 模块的属性
            setattr(split, node.target, getattr(non_const_gm, node.target))

    for node in const_gm.graph.nodes:
        # 遍历常量模块的计算图中的每个节点
        if node.op == "call_module":
            # 如果节点操作为 "call_module"
            # 将常量模块中的对应模块赋给 split 模块的属性
            setattr(split, node.target, getattr(const_gm, node.target))

    # split_module 目前不使用 get_attrs 来获取属性。而是直接将它们作为参数从父模块传递。
    # 这里在 const_gm 中设置 get_attrs，允许在运行折叠（folding）时使用它们。
    call_const_gm_args = None
    for node in split.graph.nodes:
        # 遍历 split 模块的计算图中的每个节点
        if node.op == "call_module":
            if node.target == const_mod_name:
                # 找到目标为 const_mod_name 的 call_module 节点
                call_const_gm_args = node.args
                break
    assert call_const_gm_args is not None

    # 在这里执行占位符替换为 get_attrs 的实际操作。
    # 将 const_gm.graph 设置为一个新的 root_const_gm，其中 split 是根模块，
    # 因为我们直接从根模块获取属性，而不是从 const_gm 获取。
    root_const_gm = torch.fx.GraphModule(split, const_gm.graph)
    for node in root_const_gm.graph.nodes:
        # 遍历 root_const_gm 计算图中的每个节点
        if node.op == "output":
            multiple_outputs = isinstance(node.args[0], tuple)
            continue
        if node.op != "placeholder":
            continue
        # 找到 call_const_gm_args 中与节点目标名称相同的节点
        in_node = next(n for n in call_const_gm_args if n.name == node.target)
        assert in_node.op == "get_attr"
        # 在节点之前插入一个新节点，将 get_attr 替换为实际的属性获取
        with root_const_gm.graph.inserting_before(node):
            new_node = root_const_gm.graph.get_attr(in_node.target)
        new_node.meta = node.meta.copy()
        node.replace_all_uses_with(new_node)
        root_const_gm.graph.erase_node(node)
    assert "multiple_outputs" in locals()
    # 找到 split 函数内部对 const_gm 的调用，并将其替换为对常量折叠结果的 getattr 调用。
    # 注意，我们不需要担心这是一个还是多个张量，因为原始图正确使用 getitem 从折叠的多个张量中提取单个张量。
    fx_const_folded_attrs_name = get_unique_attr_name_in_module(
        mod_traced, "_FX_CONST_FOLDED_ATTRS"
    )
    # 设置 split 对象的新属性 fx_const_folded_attrs_name，用于存储常量折叠的结果
    setattr(
        split,
        fx_const_folded_attrs_name,
        torch.nn.ParameterList() if multiple_outputs else torch.nn.Parameter(),  # type: ignore[possibly-undefined]
    )
    # 遍历 split 函数的图中的每个节点
    for node in split.graph.nodes:
        # 如果节点是调用模块，并且目标是 const_mod_name
        if node.op == "call_module" and node.target == const_mod_name:
            # 在节点之前插入操作
            with node.graph.inserting_before(node):
                # 获取 fx_const_folded_attrs_name 对应的折叠属性
                folded_attrs = node.graph.get_attr(fx_const_folded_attrs_name)
            # 复制节点的元数据给折叠属性
            folded_attrs.meta = node.meta.copy()
            # 用折叠属性替换节点的所有使用
            node.replace_all_uses_with(folded_attrs)
            break

    # 删除死代码
    split.graph.eliminate_dead_code()

    # 最后，将非常量子模块内联到 split 子模块中。
    # 这样，原始调用者如果传入了一个图模块，将得到一个图模块，其图被跟踪到相同的粒度。
    _inline_module(split, non_const_mod_name)

    # 返回一个包含折叠图模块信息的 FoldedGraphModule 对象
    return FoldedGraphModule(
        split,
        split.graph,
        root_const_gm.graph,
        fx_const_folded_attrs_name,
        device_for_folded_attrs,
    )
```