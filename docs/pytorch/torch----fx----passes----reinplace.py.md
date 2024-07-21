# `.\pytorch\torch\fx\passes\reinplace.py`

```py
# mypy: allow-untyped-defs
# 导入 PyTorch 相关模块和类
import torch
from torch.fx import Node  # 导入 Node 类，用于表示计算图中的节点
from torch.fx._compatibility import compatibility  # 导入兼容性相关的装饰器
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor  # 导入虚拟张量相关的类和模式
from torch.utils._pytree import tree_map_only  # 导入用于处理 PyTree 的函数
from torch.utils import _pytree as pytree  # 导入 PyTree 相关的模块
from torch.multiprocessing.reductions import StorageWeakRef  # 导入多进程相关的类

import _operator  # 导入操作符模块
from enum import Enum  # 导入枚举类相关的模块
import itertools  # 导入 itertools 模块，用于高效循环和迭代
from typing import Set, Dict  # 导入类型提示相关的类
from collections import defaultdict  # 导入默认字典类

__all__ = ['reinplace']  # 模块的公开接口列表，仅包括 'reinplace'

class _ViewType(Enum):  # 定义枚举类 _ViewType，表示操作的视图类型
    NonView = 0  # 非视图类型
    SingleOutputView = 1  # 单输出视图类型
    MultiOutputView = 2  # 多输出视图类型

def _is_view_op(tgt):  # 定义函数 _is_view_op，判断是否为视图操作
    if tgt is not None and isinstance(tgt, torch._ops.OpOverload):
        schema = tgt._schema
        if len(schema.arguments) > 0:
            first_arg = schema.arguments[0]
            # 检查操作是否为视图
            return first_arg.alias_info is not None and not first_arg.alias_info.is_write

def _get_view_type(tgt) -> _ViewType:  # 定义函数 _get_view_type，获取操作的视图类型
    if tgt is not None and isinstance(tgt, torch._ops.OpOverload):
        schema = tgt._schema
        if len(schema.arguments) > 0:
            first_arg = schema.arguments[0]
            # 检查操作是否为视图
            if first_arg.alias_info is not None and not first_arg.alias_info.is_write:
                # 检查操作是否为多输出视图
                if '*' in first_arg.alias_info.after_set:
                    return _ViewType.MultiOutputView
                else:
                    return _ViewType.SingleOutputView
    return _ViewType.NonView  # 默认情况下，不是视图类型

# 存储与函数化每个节点相关的元数据
# 相关的元数据:
# n.meta['fake_result']: FakeTensor (与节点输出相同类型的虚拟张量，但使用 FakeTensor 而不是 Tensor)
#   运行当前节点时生成的虚拟张量输出
# n.meta['view_of']: Node
#   如果当前节点 n 是某个基础张量的视图，则 'view_of' 字段告诉我们是哪个视图节点生成了当前节点（一个视图张量）。
#   这些信息实际上使 `fake_result` 字段变得多余，但我们可以使用 `fake_result` 来检查我们的别名信息是否正确。
@compatibility(is_backward_compatible=False)
class _FunctionalizationMetadataProp(torch.fx.Interpreter):
    def run_node(self, node: Node):
        # 增加节点计数器
        self.node_counter += 1
        # 运行父类方法，获取结果
        result = super().run_node(node)
        # 将结果存入节点的元数据中
        node.meta['fake_result'] = result
        # 将节点索引存入节点的元数据中
        node.meta['node_idx'] = self.node_counter

        # (1) 使用节点的参数列表更新元数据，排除特定的 torch 操作
        node_args = node.args
        if node.target is torch.ops.aten.copy_.default:
            node_args = node_args[1:]

        # (2) 更新元数据以跟踪视图张量节点的别名信息
        if node.op == 'call_function':
            view_type = _get_view_type(node.target)
            if view_type == _ViewType.SingleOutputView:
                assert isinstance(node.args[0], Node)
                node.meta['view_of'] = node.args[0]
            elif view_type == _ViewType.MultiOutputView:
                self.multi_output_view_nodes[node] = node.args[0]

            # 检查是否返回了多输出视图，
            # 并且现在我们从输出中获取各个视图。
            elif node.target is _operator.getitem:
                list_arg = node.args[0]
                maybe_base_of_view = self.multi_output_view_nodes.get(list_arg, None)
                if maybe_base_of_view is not None:
                    assert isinstance(maybe_base_of_view, Node)
                    node.meta['view_of'] = maybe_base_of_view

        if 'view_of' in node.meta:
            # 链接当前节点及其第一个参数作为视图
            assert isinstance(node.meta['fake_result'], FakeTensor)
            assert isinstance(node.meta['view_of'].meta['fake_result'], FakeTensor)
            # 断言视图的存储与基本张量的存储相同
            view_storage = StorageWeakRef(node.meta['fake_result']._typed_storage())
            base_storage = StorageWeakRef(node.meta['view_of'].meta['fake_result']._typed_storage())
            assert view_storage == base_storage
        return result
    # 定义一个方法 propagate，接受可变数量的参数
    def propagate(self, *args):
        # 初始化一个空字典用于存储多输出视图节点
        self.multi_output_view_nodes = {}
        # 初始化节点计数器为 -1
        self.node_counter = -1

        # 使用 FakeTensorMode 上下文管理器进行操作
        with FakeTensorMode() as mode:
            # 将每个参数转换为模拟张量对象并存储在 fake_args 列表中
            fake_args = [mode.from_tensor(a) for a in args]
            # 调用父类的 run 方法，并传入模拟的参数列表，返回其结果
            return super().run(*fake_args)
# 检查两个函数操作模式的名称是否匹配，并且它们的参数类型是否一致
def _schemas_match(functional_schema, inplace_schema):
    # 检查函数操作模式的名称是否以下划线结尾，并且去除下划线后是否与另一个模式的名称匹配
    names_match = inplace_schema.name.endswith("_") and inplace_schema.name[:-1] == functional_schema.name
    # 检查函数操作模式的参数数量是否相同，并且每个对应位置的参数类型是否一致
    arg_types_match = len(functional_schema.arguments) == len(inplace_schema.arguments) and all(
        a1.type == a2.type for a1, a2 in zip(functional_schema.arguments, inplace_schema.arguments))
    # 对于就地操作模式，确保其第一个参数是可变的
    assert inplace_schema.arguments[0].alias_info is not None and inplace_schema.arguments[0].alias_info.is_write
    # 对于其余参数，确保没有别名信息，即不能对它们进行写入操作
    assert all(a.alias_info is None for a in inplace_schema.arguments[1:])
    # 返回名称和参数类型是否匹配的布尔结果
    return names_match and arg_types_match

# TODO: 需要加强此功能，使其能够正确地支持重新就地操作:
# - 使用变异操作 (例如 _fused_moving_avg_obs_fq_helper)
# - 使用 out= 操作 (例如 angle -> angle.out)
# TODO: 我们还应该使用 torchgen 来确定这些信息。
def _maybe_get_inplace_op(op):
    # 如果操作不是 torch._ops.OpOverload 类型，则返回 None
    if not isinstance(op, torch._ops.OpOverload):
        return None
    # 一些视图操作有就地变体 (如 as_strided_ 等),
    # 但我们不希望重新就地操作直接将这些操作添加到程序中。
    # (它们需要额外的特殊处理，并且在性能上并不真正有用)
    if _is_view_op(op):
        return None
    # 获取操作的命名空间和基本名称
    op_namespace = op.__module__.split(".")[-1]
    op_base_name = op.overloadpacket.__name__
    maybe_namespace_module = getattr(torch.ops, op_namespace)
    maybe_inplace_op = None if maybe_namespace_module is None else getattr(maybe_namespace_module, f'{op_base_name}_', None)
    # 如果没有找到就地操作，返回 None
    if maybe_inplace_op is None:
        return None

    # 获取所有就地操作的重载
    inplace_overloads = [
        getattr(maybe_inplace_op, overload_name) for overload_name in maybe_inplace_op.overloads()
    ]
    # 选取具有匹配模式的就地操作
    inplace_overloads_with_matching_schemas = [
        f
        for f in inplace_overloads
        if _schemas_match(op._schema, f._schema)
    ]
    # 如果没有找到具有匹配模式的就地操作，返回 None
    if len(inplace_overloads_with_matching_schemas) == 0:
        return None
    assert len(inplace_overloads_with_matching_schemas) == 1
    # 返回匹配模式的就地操作
    inplace_op = inplace_overloads_with_matching_schemas[0]
    return inplace_op

# 视图操作的逆映射字典，用于将散布操作映射回其原始视图操作
_VIEW_INVERSE_MAP = {
    torch.ops.aten.diagonal_scatter.default: torch.ops.aten.diagonal.default,
    torch.ops.aten.select_scatter.default: torch.ops.aten.select.int,
    torch.ops.aten.slice_scatter.default: torch.ops.aten.slice.Tensor,
    torch.ops.aten.as_strided_scatter.default: torch.ops.aten.as_strided.default,
}

# 此函数接收一组（别名）张量节点，并返回在图中使用任何别名的节点，
# 以及其在操作索引之后出现的任何节点。
# 给定一组节点别名和操作索引，返回所有在指定操作后使用的节点集合
def _get_all_later_node_usages(tensor_aliases: Set[Node], op_index: int):
    # 将 FakeTensor 类型的对象加入集合 set_，以存储其弱引用对象
    def _add_if_tensor(x, set_):
        if isinstance(x, FakeTensor):
            set_.add(StorageWeakRef(x._typed_storage()))

    # 用来存储在指定操作后使用的节点集合
    nodes_used_after = set()
    # 遍历所有的别名节点
    for t in tensor_aliases:
        # 获取使用当前别名的所有节点
        usage_nodes = t.users
        for n in usage_nodes:
            # 只关心在当前节点之后的使用情况
            if 'node_idx' not in n.meta or n.meta['node_idx'] <= op_index:
                continue
            # 不关心中间的视图操作，除非它们的输出被其他地方使用
            if n in tensor_aliases:
                # 如果节点是别名集合中的一部分，并且目标是一个特定操作或者索引操作，则跳过
                if isinstance(n.target, torch._ops.OpOverload) or n.target == _operator.getitem:
                    continue
            # 将满足条件的节点加入集合中
            nodes_used_after.add(n)
    return nodes_used_after

# 给定一个正在尝试重新赋值的操作 "b = foo(a)",
# 以及后续图中出现的 {view}_scatter 操作 "y = {view}_scatter(base, x, args...)"
# 如果在别名集合中存在满足以下条件的任何别名：
# (1) "alias" 的基础部分 "alias_base" 具有与 "base" 相同的大小、步幅和偏移量元数据
# (2) 运行 {view}(alias, args...) 的输出具有与 "alias" 相同的大小、步幅和偏移量元数据
# 则重新赋值 `foo()` 将允许我们完全删除 `{view}_scatter` 操作，如果：
# 如果在自身别名集合中存在任何别名满足这些条件。
def _get_view_inverse_node_usages(later_node_usages: Set[Node], self_aliases: Set[Node]) -> Set[Node]:
    # 检查两个对象的视图元数据是否匹配
    def matching_view_metadata(a, b):
        return a.size() == b.size() and \
            a.stride() == b.stride() and \
            a.storage_offset() == b.storage_offset()

    # 存储符合条件的节点集合
    view_inverse_nodes = set()
    # 按照节点顺序进行处理，这样可以查看视图散布操作链。
    # 对 `later_node_usages` 中的节点按照 `node_idx` 属性进行排序，获取排序后的节点列表
    for n in sorted(later_node_usages, key=lambda x: x.meta['node_idx']):
        # 如果当前节点的目标不在 `_VIEW_INVERSE_MAP` 中，则跳过本次循环
        if n.target not in _VIEW_INVERSE_MAP:
            continue
        
        # 获取当前节点的第一个参数作为 `base`，第二个参数作为 `mutated_view`
        base = n.args[0]
        mutated_view = n.args[1]
        
        # 断言 `base` 和 `mutated_view` 都是 `Node` 类型
        assert isinstance(base, Node)
        assert isinstance(base.meta['fake_result'], FakeTensor)
        assert isinstance(mutated_view, Node)
        assert isinstance(mutated_view.meta['fake_result'], FakeTensor)
        
        # 检查这个 view_inverse 操作是否实际上是对我们现有的 self_alias 节点之一进行反向操作
        original_view = _VIEW_INVERSE_MAP[n.target]
        
        # 遍历 `self_aliases` 列表，查找与当前 `self_arg` 的某个别名相关的 self_alias
        for self_alias in self_aliases:
            # 如果 `self_alias` 的元数据中没有 'view_of' 键，则跳过本次循环
            if 'view_of' not in self_alias.meta:
                continue
            
            # 获取 `self_alias` 的基本别名
            self_alias_base = self_alias.meta['view_of']
            
            try:
                # 尝试使用与当前视图散布调用内部相应视图操作中的参数来重新运行原始操作，并检查是否匹配步长
                view_replay_metadata = original_view(self_alias_base.meta['fake_result'], *n.args[2:], **n.kwargs)
                expected_metadata = self_alias.meta['fake_result']
                
                # 如果 `self_alias` 和其基本别名的元数据匹配，并且视图重播元数据与预期元数据匹配，则将当前节点添加到 `view_inverse_nodes`
                if matching_view_metadata(self_alias_base.meta['fake_result'], base.meta['fake_result']) and \
                        matching_view_metadata(view_replay_metadata, expected_metadata):
                    view_inverse_nodes.add(n)
            except Exception:
                # 如果出现异常则继续下一个循环
                continue
    
    # 返回所有符合条件的 `view_inverse` 节点集合
    return view_inverse_nodes
# 定义一个装饰器函数，标记函数为向后兼容的
@compatibility(is_backward_compatible=True)
# 定义一个函数，接受一个 fx.GraphModule 和一系列样本参数
def reinplace(gm, *sample_args):
    """
    给定一个 fx.GraphModule，修改其以执行“reinplacing”，即修改图的节点。
    我们查找类似 `b = a.add(...)` 的非原位操作调用点，
    并将它们转换为原位操作 (`b = a.add_(...)`)，
    前提是当前运算符的输入 ("a") 不会在图中的任何后续位置重用。

    此操作目前期望在一个**函数式、ATen**图上运行。
    可以通过运行 `make_fx(functionalize(f))` 来获取这样的图。

    需要样本输入来确定输入之间的别名关系。
    通常情况下，如果“a”与程序的任何输入别名，则无法对节点 `b = a.add(...)` 进行原位替换。

    给定一个节点 "b = foo(a, args...)"，进行原位替换的算法如下：
    """
    # (1) 对 "a" 和 "args..." 的元数据进行初始检查，这些检查可以排除它们不适合重新就地替换的情况。

    # (1a) 检查我们尝试重新就地替换的 self 参数是否具有可以重新替换的合适 dtype/size 元数据。

    #      例如，如果有以下情况：
    #        a = torch.ones(1)
    #        b = torch.ones(10)
    #        out = torch.add(a, b)
    #      我们无法将其转换为
    #        a.add_(b)
    #      因为这将需要调整 "a" 的大小。

    #      类似地，我们也不能将 torch.ge(a, b) 转换为 a.ge_(b)，
    #      因为这将需要改变 "a" 的 dtype（例如从 float32 改为 bool）。
    #      注意，在这个特定的例子中，我们技术上可以做得更好。

    #      如果我们看到以下模式：
    #        a_1 = a.ge(b)
    #        a_2 = aten._to_copy(a_1, a.dtype)
    #      那么完全重新就地替换应该是有效的
    #      （这正是当它看到 a.ge_(b) 时功能化会生成的内容）。

    #      这种优化对直接使用就地比较操作的用户程序非常重要。

    #      我们也不能对具有重叠内存的张量进行重新就地替换，
    #      例如 torch.ones(1).expand(4, 4).add_(1)

    # (1b) 检查 "a" 是否是程序输入的任何别名。

    #      如果是，跳过并移动到下一个节点。
    #      就地执行一个会导致程序变异的操作是不安全的，
    #      因为这会对用户可见的副作用。

    #      注意：有一个未来的优化我们应该实现：
    #      如果 "a" 是程序输入的别名，但稍后在程序中有一个节点看起来像 "a.copy_(...)"，
    #      那么重新就地替换是可以的 - 我们暂时重用 "a" 的缓冲区，
    #      这将在 copy_() 调用时被后续覆盖。

    #      对于修改其输入的程序来说，这将是一个重要的优化。但目前尚未实现。

    # (1c) 检查 "a" 和 "args..." 是否是别名。

    #      例如，重新就地替换以创建类似下面的代码
    #        aten.mul_(a, a)
    #      不能保证是安全的。
    # 实际执行 foo 函数的替换操作！

    # 这是常见情况，但是针对 {view}_scatter 需要特别注意 (3a)

    # {view}_scatter 操作。

    # 考虑以下程序示例：
    #   a = torch.zeros(2, 2)
    #   b = torch.ones(2)
    #   a[0] = b
    # 在功能化后，它将变成：
    #   a = torch.zeros(2)
    #   b = torch.ones(1)
    #   a_updated = torch.select_scatter(a, b, 0, 0)
    # 在这种情况下，没有“functional”操作来进行重新插入！
    # 相反，我们希望直接移除 select_scatter 调用。
    # 我们已经从 (3) 中知道这是有效的，因为在图中没有后续使用“a”的地方。

    # 我们执行如下对 {view}_scatter 操作的重新插入操作
    # 在之前：
    #   a_updated = torch.select_scatter(a, b, args...)
    # 之后：
    #   a_slice = a.select(a, args...)
    #   a_slice.copy_(b)

    # 否则，用其就地变体替换功能性操作。
    # 在之前：
    #   b = foo(a, args...)
    # 之后：
    #   a.foo_(args...)

    # 最后，在转换之后：
    #   在之前：
    #     b = foo(a)
    #   之后：
    #     foo_(a)
    # 或者
    #   在之前：
    #     b = {slice}_scatter(a, mutated_slice, args...)
    #   之后：
    #     slice = {slice}(a, args...)
    #     slice.copy_(mutated_slice)
    # 现在，我们需要找到所有后续使用“b”作为参数的节点，并更新它们以取代为“a”。

    # 注意，对于大多数就地操作来说，这实际上是不必要的
    # （因为大多数就地操作将“self”作为它们的输出返回）。
    # 对于所有可变操作来说，这并不一般适用，这就是为什么我们需要实际替换所有的参数。

    # 我们还需要更新我们的元数据 Dict[StorageWeakRef, Set[Node]]，
    # 将给定张量存储映射到所有以该存储作为输入的节点集合。
    # 具体而言，重新插入 `b = foo(a)` 导致了 "a" 和 "b" 的集合被合并。

    # 在步骤 (3) 期间被标识为“可以忽略”的所有“view_inverse/scatter”节点
    # 现在被手动从图中删除。
    # 它们的输出不再被使用，因此从技术上讲，标准的 DCE 可以做到这一点，
    # 但是现在我们不能再运行 FX 的 DCE pass 因为图中有可变操作。
    """
    _FunctionalizationMetadataProp(gm).propagate(*sample_args)

    # 有用的调试打印
    # def _print(x):
    # if isinstance(x, FakeTensor):
    # print(f'fake_result: {StorageWeakRef(x._typed_storage()).cdata}')

    # for n in gm.graph.nodes:
    # print(n.format_node())
    # if hasattr(n, 'meta'):
    # print(f'node_idx: {n.meta["node_idx"]}')
    # if 'fake_result' in n.meta:
    # tree_map(_print, n.meta['fake_result'])
    # if 'view_of' in n.meta:
    # print(f'view_of: {str(n.meta["view_of"])}')
    # print()
    ```
    # 我们需要知道哪些节点对应于输入（或其别名），以便知道不要对它们进行重新放置。
    # 注意：稍后，我们需要为那些修改输入的程序添加优化，以完全恢复性能。
    input_storages = {
        StorageWeakRef(
            node.meta['fake_result']._typed_storage()
        ) for node in gm.graph.nodes if node.op == 'placeholder'}

    # 我们还需要知道对于给定节点，其所有别名节点是什么。
    storage_to_nodes: Dict[StorageWeakRef, Set[Node]] = defaultdict(set)
    for n in gm.graph.nodes:
        if 'fake_result' in n.meta:
            # 使用树映射，因为一些操作可以返回张量列表。
            def _add_to_map(x):
                if isinstance(x, FakeTensor):
                    storage_to_nodes[StorageWeakRef(x._typed_storage())].add(n)
            pytree.tree_map_(_add_to_map, n.meta['fake_result'])

    # 将函数式操作修改为原地操作，受下面所述的约束限制。
    all_later_view_inverse_nodes_to_delete = set()
    # 步骤 4：删除所有我们取消函数化的 _scatter 节点。
    # 在完成对图的所有修改之前，务必小心不要删除任何这些节点。
    for to_delete in all_later_view_inverse_nodes_to_delete:
        gm.graph.erase_node(to_delete)

    # 重新编译图模型
    gm.recompile()
    # 返回更新后的图模型
    return gm
```