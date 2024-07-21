# `.\pytorch\torch\profiler\_memory_profiler.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和类
import collections  # 引入collections模块，用于定义特定数据结构
import dataclasses  # 引入dataclasses模块，用于定义数据类
import enum  # 引入enum模块，用于定义枚举类型
import itertools as it  # 引入itertools模块，用于高效循环和迭代操作
import logging  # 引入logging模块，用于记录日志信息
from typing import (  # 引入typing模块，定义类型提示
    Any,
    cast,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from typing_extensions import Literal  # 引入Literal类型提示扩展

import torch  # 引入PyTorch库
from torch._C import FunctionSchema  # 从torch._C模块导入FunctionSchema类
from torch._C._autograd import _ProfilerResult  # 从torch._C._autograd导入_ProfilerResult类
from torch._C._profiler import (  # 从torch._C._profiler导入多个类和函数
    _EventType,
    _ExtraFields_Allocation,
    _ExtraFields_TorchOp,
    _ProfilerEvent,
    _TensorMetadata,
    RecordScope,
)
from torch._utils import _element_size  # 从torch._utils导入_element_size函数
from torch.profiler import _utils  # 从torch.profiler导入_utils模块

# 定义类型别名KeyAndID和TensorAndID
KeyAndID = Tuple["Key", int]
TensorAndID = Tuple["TensorKey", int]

# 设置日志记录器
log = logging.getLogger(__name__)

# 定义枚举类Category，表示不同的类别
class Category(enum.Enum):
    INPUT = enum.auto()
    TEMPORARY = enum.auto()
    ACTIVATION = enum.auto()
    GRADIENT = enum.auto()
    AUTOGRAD_DETAIL = enum.auto()
    PARAMETER = enum.auto()
    OPTIMIZER_STATE = enum.auto()

# 定义类别到颜色的映射字典
_CATEGORY_TO_COLORS = {
    Category.PARAMETER: "darkgreen",
    Category.OPTIMIZER_STATE: "goldenrod",
    Category.INPUT: "black",
    Category.TEMPORARY: "mediumpurple",
    Category.ACTIVATION: "red",
    Category.GRADIENT: "mediumblue",
    Category.AUTOGRAD_DETAIL: "royalblue",
    None: "grey",
}

# 定义类别到索引的映射字典
_CATEGORY_TO_INDEX = {c: i for i, c in enumerate(_CATEGORY_TO_COLORS)}

# 定义枚举类Action，表示不同的动作
class Action(enum.Enum):
    PREEXISTING = enum.auto()
    CREATE = enum.auto()
    INCREMENT_VERSION = enum.auto()
    DESTROY = enum.auto()

# 定义动作到索引的映射字典
_ACTION_TO_INDEX = {i: i.value for i in Action}

# 定义数据类Key，表示具有设备信息的键
@dataclasses.dataclass(eq=True, unsafe_hash=False, frozen=True)
class Key:
    device: torch.device

# 定义数据类_Storage，表示存储指针和ID的捆绑
@dataclasses.dataclass
class _Storage:
    """Bundle storage pointer and id.

    All profiling logic should use `allocation_id`, however it is useful to
    print storage pointers for debugging and unit tests sometimes look up
    values using the storage data pointer of a live Tensor."""
    ptr: int
    allocation_id: int

    def __repr__(self) -> str:
        return f"{hex(self.ptr):>18} ({self.allocation_id})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Storage) and self.allocation_id == other.allocation_id

    def __hash__(self) -> int:
        return hash(self.allocation_id)

# 定义数据类TensorKey，表示分配了ID的存储的可哈希标识符
@dataclasses.dataclass(eq=True, unsafe_hash=True, frozen=True)
class TensorKey(Key):
    """Hashable identifier for a storage which has been asigned an ID.

    A detailed description of Tensor IDs and why they are needed is given in
    `torch/csrc/profiler/collection.h` when `TensorID` is declared. To
    summarize, multiple Storage buffers can map to the same logical Tensor.
    This dataclass is used to refer to a concrete in-memory StorageImpl of
    a Tensor.
    """
    id: int
    storage: _Storage

    def __repr__(self) -> str:
        return f"id={self.id}: {repr(self.storage):<24} ({self.device})"

    def __lt__(self, other: "TensorKey") -> bool:
        return self._as_sortable < other._as_sortable
    # 定义一个静态方法 `_make`，用于创建 `TensorKey` 实例
    @staticmethod
    def _make(
        tensor_id: Optional[int],
        storage_ptr: Optional[int],
        allocation_id: Optional[int],
        device: torch.device,
    ) -> Optional["TensorKey"]:
        # 检查传入的参数是否都不为 None
        if (
            tensor_id is not None
            and storage_ptr is not None
            and allocation_id is not None
        ):
            # 如果参数都不为 None，则创建并返回一个 `TensorKey` 实例
            return TensorKey(device, tensor_id, _Storage(storage_ptr, allocation_id))
        # 如果有任何一个参数为 None，则返回 None
        return None

    # 从 `_ExtraFields_Allocation` 对象创建 `TensorKey` 实例的类方法
    @classmethod
    def from_allocation(cls, alloc: _ExtraFields_Allocation) -> Optional["TensorKey"]:
        # 调用 `_make` 方法，传入 `_ExtraFields_Allocation` 对象的属性作为参数
        return cls._make(alloc.id, alloc.ptr, alloc.allocation_id, alloc.device)

    # 从 `_TensorMetadata` 对象创建 `TensorKey` 实例的类方法
    @classmethod
    def from_tensor(cls, t: Optional[_TensorMetadata]) -> Optional["TensorKey"]:
        # 如果 `_TensorMetadata` 对象不为 None，则调用 `_make` 方法
        if t is not None:
            return cls._make(t.id, t.storage_data_ptr, t.allocation_id, t.device)
        # 如果 `_TensorMetadata` 对象为 None，则返回 None
        return None

    # 返回一个元组，用于排序的属性
    @property
    def _as_sortable(self) -> Tuple[int, int, str, int]:
        # 返回 `TensorKey` 实例的属性作为元组，用于排序
        return self.id, self.storage.allocation_id, self.device.type, self.device.index
def _extract_parameters_and_gradients(
    node: _ProfilerEvent,
) -> Iterator[Tuple[Optional[TensorKey], Optional[TensorKey]]]:
    # 获取当前节点的所有子节点
    children = node.children

    # AccumulateGrad 用于处理自动求导引擎中的梯度更新。
    # 有两种可能的情况：
    # 1) 这是一个新创建的梯度张量。在这种情况下，没有什么需要累积的，
    #    因此自动求导简单地分离该张量（aten::detach）。
    #
    # 2) 存在一个预先存在的梯度张量，我们需要添加新计算的更新。
    #    这通过一个原地加法操作（aten::add_）完成。
    #    （下划线后缀表示“原地”操作。）
    if (
        node.typed[0] == _EventType.TorchOp
        and node.typed[1].scope == RecordScope.BACKWARD_FUNCTION
        # TODO(robieta): 远离具有载荷的名称
        and node.name == "torch::autograd::AccumulateGrad"
        and children
        and children[0].typed[0] == _EventType.TorchOp
        and children[0].name in ("aten::detach", "aten::add_")
        and children[0].typed[1].inputs
        and isinstance(children[0].typed[1].inputs[0], _TensorMetadata)
    ):
        # 如果满足条件，则生成一个空的 p 和梯度张量的键值对
        yield None, TensorKey.from_tensor(children[0].typed[1].inputs[0])

    # 我们直接检测 `torch.nn.Module` 和 `torch.optim.Optimizer`
    # 注意：Python 跟踪器捕获的值是缓存的；它们可以用于构建标签，
    #       但不意味着张量在特定时间是活跃的。
    elif node.typed[0] == _EventType.PyCall:
        typed_fields = node.typed[1]
        assert typed_fields.module is None or typed_fields.optimizer is None
        if typed_fields.module is not None:
            # 遍历模块的参数和梯度，生成对应的键值对
            for _, p, p_grad in typed_fields.module.parameters:
                yield TensorKey.from_tensor(p), TensorKey.from_tensor(p_grad)

        if typed_fields.optimizer is not None:
            # 遍历优化器的参数和梯度，生成对应的键值对
            for p, p_grad, _ in typed_fields.optimizer.parameters:
                yield TensorKey.from_tensor(p), TensorKey.from_tensor(p_grad)


def extract_parameters(node: _ProfilerEvent) -> Iterator[TensorKey]:
    # 调用 _extract_parameters_and_gradients 函数生成参数的迭代器
    for p, p_grad in _extract_parameters_and_gradients(node):
        if p is not None:
            yield p


def extract_gradients(
    node: _ProfilerEvent,
) -> Iterator[Tuple[Optional[TensorKey], TensorKey]]:
    # 调用 _extract_parameters_and_gradients 函数生成梯度的迭代器
    for p, p_grad in _extract_parameters_and_gradients(node):
        if p_grad is not None:
            yield p, p_grad


def get_scopes(event: Optional[_ProfilerEvent]) -> Tuple[RecordScope, ...]:
    scopes = []
    # 从当前事件追溯到其父事件，收集所有的作用域
    while event:
        if event.typed[0] == _EventType.TorchOp:
            scopes.append(event.typed[1].scope)
        event = event.parent
    return tuple(scopes)


class SchemaMatcher:
    """根据 profiled 名称查找运算符模式。

    在进行性能分析时，我们记录运算符的名称但不记录模式。
    然而，某些分析需要这些信息。幸运的是，我们可以根据记录的名称查找注册的模式。
    不过，我们不会记录
    """
    # 此方法判断给定的 _ExtraFields_TorchOp 对象 t 的输入是否可能被修改，并返回一个布尔值元组。
    @classmethod
    def inputs_are_mutable(cls, t: _ExtraFields_TorchOp) -> Tuple[Optional[bool], ...]:
        """Determine which inputs may have mutated based on function schema.

        Note that we don't need to resolve down to a single schema to perform
        this analysis. An input is mutable if it is mutable in any overload. In
        practice, however, it is overwhelmingly common to match a single
        overload. If we cannot find any valid schema then we must be
        conservative and assume all inputs are mutable.
        """
        # 初始化一个可选的布尔值列表，用于记录每个输入是否可变
        mutable: Optional[List[bool]] = None
        # 遍历匹配到的所有函数模式
        for schema in cls.match_schemas(t):
            # 如果 mutable 为 None，则初始化为一个与 schema.arguments 相同长度的 False 列表
            mutable = mutable or [False for _ in schema.arguments]
            # 遍历 schema.arguments，确定每个参数是否可变
            for i, arg in enumerate(schema.arguments):
                mutable[i] |= getattr(arg.alias_info, "is_write", False)

        # 将 mutable 转换为元组，或者如果 mutable 为 None，则创建与 t.inputs 相同长度的 None 元组
        return tuple(mutable or (None for _ in t.inputs))

    @classmethod
    def match_schemas(cls, t: _ExtraFields_TorchOp) -> Tuple[FunctionSchema, ...]:
        # 根据 t.inputs 的类型生成相应的签名，用于后续的函数模式匹配
        signature = tuple(
            # 如果是 _TensorMetadata 类型，则转换为 TensorKey
            TensorKey.from_tensor(i) if isinstance(i, _TensorMetadata)
            # 如果是列表类型，则转换为列表中每个元素的 TensorKey
            else [TensorKey.from_tensor(j) for j in i] if isinstance(i, list)
            # 否则直接使用输入 i
            else i
            for i in t.inputs
        )

        # 定义一个匹配函数，检查给定的 schema 是否与 signature 匹配
        def matches(schema) -> bool:
            return len(schema.arguments) == len(signature) and all(
                cls._types_match(observed, schema_arg.type)
                for observed, schema_arg in zip(signature, schema.arguments)
            )

        # 返回所有匹配的函数模式，这些模式是根据 t.name 查找得到的
        return tuple(s for s in cls.lookup_schemas(t.name) or () if matches(s))

    @classmethod
    def _types_match(cls, observed, schema_type) -> bool:
        # 检查是否为可选类型，若是，则获取其实际类型并递归调用_types_match函数
        if isinstance(schema_type, torch._C.OptionalType):
            schema_type = schema_type.getElementType()
            return observed is None or cls._types_match(observed, schema_type)

        # 检查是否为任意类型，是则直接返回True
        if isinstance(schema_type, torch._C.AnyType):
            return True

        # 检查是否为张量列表类型，同时检查observed是否为列表且列表中所有元素是否为TensorKey类型
        if schema_type.isSubtypeOf(torch._C.ListType.ofTensors()):
            return isinstance(observed, list) and all(
                isinstance(i, TensorKey) for i in observed
            )

        # 定义类型映射关系，用于检查schema_type是否与observed类型匹配
        type_map: Tuple[Tuple[Any, Union[type, Tuple[type, ...]]], ...] = (
            (torch._C.TensorType, TensorKey),
            (torch._C.NoneType, type(None)),
            (torch._C.BoolType, bool),
            (torch._C.IntType, int),
            (torch._C.FloatType, float),
            (torch._C.ComplexType, complex),
            (torch._C.NumberType, (bool, int, float, complex)),
        )

        # 遍历类型映射关系，检查schema_type是否属于其中一种类型，并验证observed是否匹配对应的Python类型
        for jit_type, py_types in type_map:
            if isinstance(schema_type, jit_type):
                return isinstance(observed, py_types)

        # 如果程序执行到这里，说明Profiler仅记录了部分可能的参数类型。
        # 如果schema_type要求的类型不在已知类型中，那么observed只能为None才能匹配。
        return observed is None

    @staticmethod
    def lookup_schemas(name: str) -> Optional[Tuple[FunctionSchema, ...]]:
        # TODO(robieta):
        #   _jit_get_schemas_for_operator是相当昂贵的操作。（每次调用大约100微秒）
        #   如果有性能问题，考虑添加`functools.lru_cache`。

        try:
            # 尝试获取操作符的schema，如果name格式不正确会抛出异常。
            # 如果name不包含"::"，则直接返回None表示name不是一个有效的操作符名称。
            #
            # 注意，record_function注解也会通过此路径，因此预期某些名称可能不对应于PyTorch操作符。
            if "::" not in name:
                return None
            return tuple(torch._C._jit_get_schemas_for_operator(name))
        except RuntimeError:
            return None
class OpTree:
    # 初始化函数，接受一个 _ProfilerResult 对象作为参数
    def __init__(self, result: _ProfilerResult) -> None:
        # 获取实验性事件树的根节点
        self._root_nodes = result.experimental_event_tree()
        # 深度优先搜索遍历节点，并按照开始时间排序，存储为元组
        self._sorted_nodes = tuple(sorted(self.dfs(), key=lambda x: x.start_time_ns))

    # 深度优先搜索函数，返回一个 _ProfilerEvent 的迭代器
    def dfs(self, *args, **kwargs) -> Iterator[_ProfilerEvent]:
        yield from _utils.traverse_dfs(self._root_nodes, *args, **kwargs)

    # 返回已排序的节点元组作为属性
    @property
    def sorted_nodes(self) -> Tuple[_ProfilerEvent, ...]:
        return self._sorted_nodes


class SizeMap:
    # 初始化函数，接受一个 OpTree 对象作为参数
    def __init__(self, op_tree: OpTree) -> None:
        # 创建一个空字典，用于存储 TensorKey 到 int 的映射关系
        self._values: Dict[TensorKey, int] = {}

        # 遍历 op_tree 中已排序的节点
        for node in op_tree.sorted_nodes:
            # 如果节点的类型为 TorchOp
            if node.typed[0] == _EventType.TorchOp:
                # 遍历节点的平展化张量输入
                for t in self._flat_tensor_inputs(node.typed[1]):
                    # 更新 _values 字典中的值
                    self._update_values(t)

            # 如果节点的类型为 PyCall
            elif node.typed[0] == _EventType.PyCall:
                typed_fields = node.typed[1]
                # 断言模块字段为空或者优化器字段为空
                assert typed_fields.module is None or typed_fields.optimizer is None
                # 如果模块字段不为空
                if typed_fields.module is not None:
                    # 遍历模块参数的三元组（name, parameter, gradient）
                    for _, p, p_grad in typed_fields.module.parameters:
                        # 更新 _values 字典中的值
                        self._update_values(p)
                        self._update_values(p_grad)

                # 如果优化器字段不为空
                if typed_fields.optimizer is not None:
                    # 遍历优化器参数的三元组（parameter, gradient, state）
                    for p, p_grad, state in typed_fields.optimizer.parameters:
                        # 更新 _values 字典中的值
                        self._update_values(p)
                        self._update_values(p_grad)
                        # 遍历状态字段的元组（name, tensor）
                        for _, t in state:
                            # 更新 _values 字典中的值
                            self._update_values(t)

        # 创建一个空字典，用于存储 TensorKey 到 int 的映射关系
        allocations: Dict[TensorKey, int] = {}
        # 再次遍历 op_tree 中已排序的节点
        for node in op_tree.sorted_nodes:
            # 如果节点的类型为 Allocation
            if node.typed[0] == _EventType.Allocation:
                # 获取节点的分配字段
                alloc_fields = node.typed[1]
                # 从分配字段创建 TensorKey
                key = TensorKey.from_allocation(alloc_fields)
                # 如果成功创建了 TensorKey
                if key:
                    # 计算新的尺寸大小
                    new_size = abs(alloc_fields.alloc_size)
                    # 获取先前的尺寸大小或者设置为新的尺寸大小
                    prior_size = allocations.setdefault(key, new_size)

                    # 如果先前的尺寸大小与新的尺寸大小不一致
                    if prior_size != new_size:
                        # 计算尺寸变化差异
                        delta = f"{prior_size} vs. {new_size}"
                        # 发出警告日志，指出分配与释放之间的不匹配
                        log.warning("Mismatch between allocation and free: %s", delta)

        # 更新 self._values 字典，将 allocations 中的映射关系合并进去
        self._values.update(allocations)
    # 更新值的方法，根据给定的张量元数据更新内部存储的值
    def _update_values(self, t: Optional[_TensorMetadata]) -> None:
        # 从张量元数据生成键
        key = TensorKey.from_tensor(t)
        # 如果键不为None且张量不为None且布局为torch.strided
        if key is not None and t is not None and t.layout == torch.strided:
            # 标量在内部被表示为零维张量
            # 计算张量的元素个数乘以元素的字节大小，得到总字节数
            n = max(i[0] * i[1] for i in zip(t.sizes or [1], t.strides or [1]))

            # 计算张量占用的字节数，确保字节数不小于零
            num_bytes = n * _element_size(t.dtype)
            assert num_bytes >= 0, f"{num_bytes}"
            # 更新self._values中对应键的值为已有值和新计算的字节数的较大值
            self._values[key] = max(self._values.get(key, 0), num_bytes)

    # 静态方法，用于扁平化张量输入列表，生成迭代器返回张量元数据
    @staticmethod
    def _flat_tensor_inputs(op: _ExtraFields_TorchOp) -> Iterator[_TensorMetadata]:
        # 遍历操作的输入
        for i in op.inputs:
            # 如果输入是张量元数据，则yield返回该元数据
            if isinstance(i, _TensorMetadata):
                yield i
            # 如果输入是列表，则递归yield其内部的所有元素
            elif isinstance(i, list):
                yield from i

    # 通过键获取对应的值，用于索引操作
    def __getitem__(self, key: TensorKey):
        return self._values[key]
@dataclasses.dataclass()
class DataFlowEdge:
    input_version: Optional[int] = None  # 定义一个可选的整数属性 input_version，默认为 None
    mutated: Optional[bool] = False  # 定义一个可选的布尔属性 mutated，默认为 False

    @property
    def is_allocation(self) -> bool:
        return self.input_version is None  # 返回是否为分配操作的布尔值，即 input_version 是否为 None

    @property
    def is_deletion(self) -> bool:
        return self.mutated is None  # 返回是否为删除操作的布尔值，即 mutated 是否为 None


class DataFlowNode:
    def __init__(self, event: _ProfilerEvent, graph: "DataFlowGraph") -> None:
        self._event = event  # 保存事件对象到私有属性 _event
        self._graph = graph  # 保存数据流图对象到私有属性 _graph
        self._edges: Dict[TensorKey, DataFlowEdge] = self._determine_edges()  # 使用 _determine_edges 方法确定节点的边缘关系，并保存为字典类型的私有属性 _edges

        # 对每个边缘进行遍历
        for key, edge in self._edges.items():
            # 如果边缘发生了变异且不是分配操作，调用数据流图的 bump 方法
            if edge.mutated and not edge.is_allocation:
                self._graph.bump(key)

        # 确保版本增加行为符合预期
        versions = {k: (v, self._graph.lookup(k)) for k, v in self.outputs.items()}
        assert all(i == j for i, j in versions.values()), f"{versions}, {self._edges}"
    def _determine_edges(self) -> Dict[TensorKey, DataFlowEdge]:
        # 使用深度优先搜索遍历子树，得到所有节点的元组
        subtree = tuple(_utils.traverse_dfs([self._event]))

        # 初始化一个字典，用于存储每个 TensorKey 对应的可变性信息集合
        mutable_by_key: Dict[Optional[TensorKey], Set[Optional[bool]]] = {}

        # 遍历子树中的每个 TorchOp 节点的操作
        for op in (i.typed[1] for i in subtree if i.typed[0] == _EventType.TorchOp):
            # 遍历每个操作的输入及其是否可变性的匹配
            for op_input, mutable in zip(
                op.inputs, SchemaMatcher.inputs_are_mutable(op)
            ):
                # 如果是 Tensor 类型
                if isinstance(op_input, _TensorMetadata):
                    # 根据 Tensor 生成 TensorKey，并将可变性信息添加到对应的集合中
                    key = TensorKey.from_tensor(op_input)
                    mutable_by_key.setdefault(key, set()).add(mutable)

                # 如果是 TensorList 类型
                elif isinstance(op_input, list):
                    # 对列表中的每个 Tensor 进行相同的操作
                    for op_input_i in op_input:
                        key = TensorKey.from_tensor(op_input_i)
                        mutable_by_key.setdefault(key, set()).add(mutable)

        # 初始化一个默认字典来存储 DataFlowEdge 对象
        edges: DefaultDict[Optional[TensorKey], DataFlowEdge]
        edges = collections.defaultdict(DataFlowEdge)

        # 遍历每个 TensorKey 及其可变性信息集合
        for key, mutable_set in mutable_by_key.items():
            if key is not None:
                # 设置输入版本号，如果 key 不存在则设置为 -1
                edges[key].input_version = self._graph.lookup(key) if key else -1

                # 判断 Tensor 是否被修改，基于可变性信息集合中是否存在 True 或全为 None 的情况
                mutated = (True in mutable_set) or (tuple(mutable_set) == (None,))
                edges[key].mutated = mutated

        # 处理删除操作，注意删除 Tensor 会隐式添加为输入边
        for i in subtree:
            if i.typed[0] == _EventType.Allocation and i.typed[1].alloc_size < 0:
                # 根据 Allocation 生成 TensorKey，并设置边的状态为未定义
                key = TensorKey.from_allocation(i.typed[1])
                edge = edges[key]
                assert key is None or edge.mutated is not None, f"Double delete: {key}"
                edge.mutated = None
                edge.input_version = self._graph.lookup(key) if key else -1

        # 处理分配操作，这一步骤必须放在最后，因为前两步乐观地添加了输入边
        for i in subtree:
            if i.typed[0] == _EventType.Allocation and i.typed[1].alloc_size > 0:
                # 根据 Allocation 生成 TensorKey，并设置输入版本为 None
                edges[TensorKey.from_allocation(i.typed[1])].input_version = None

        # 返回排序后的边字典，过滤掉键为 None 的项
        return dict(sorted((k, v) for k, v in edges.items() if k is not None))

    @property
    def inputs(self) -> Dict[TensorKey, Tuple[bool, int]]:
        # 返回一个字典，将每个 TensorKey 映射到其对应的可变性和输入版本信息
        return {
            k: (bool(v.mutated), cast(int, v.input_version))
            for k, v in self._edges.items()  # 遍历 self._edges 中的每一项
            if not v.is_allocation  # 排除 is_allocation 属性为 True 的项
        }
    # 返回一个字典，包含每个 TensorKey 对象到一个整数的映射
    def outputs(self) -> Dict[TensorKey, int]:
        return {
            # 如果输入版本为 None，则映射值为 0；否则映射值为输入版本号加 1
            k: 0 if v.input_version is None else v.input_version + 1
            for k, v in self._edges.items()
            # 只选择满足以下条件的边：
            # 1. 是分配操作且不是删除操作
            # 2. 或者边被修改过
            if (v.is_allocation and not v.is_deletion) or v.mutated
        }

    # 返回一个元组，包含所有被分配和删除标记的 TensorKey 对象
    @property
    def intermediates(self) -> Tuple[TensorKey, ...]:
        return tuple(
            k for k, v in self._edges.items()
            # 选择所有同时满足分配和删除标记的边
            if v.is_allocation and v.is_deletion
        )

    # 返回事件对象的开始时间，单位为纳秒
    @property
    def start_time(self) -> int:
        return self._event.start_time_ns
    # 数据流图类，用于表示操作树的数据流图
class DataFlowGraph:
    # 初始化方法，接受一个操作树对象作为参数
    def __init__(self, op_tree: OpTree) -> None:
        # 将操作树对象保存为私有属性 _op_tree
        self._op_tree = op_tree
        # 从操作树中提取叶子事件，并保存为私有属性 _leaf_events
        self._leaf_events = self._extract_leaf_events(op_tree)
        # 初始化一个空的字典，用于存储每个张量的活跃版本号
        self._active_version: Dict[TensorKey, Optional[int]] = {}
        # 根据叶子事件创建数据流节点列表，并保存为私有属性 _flow_nodes
        self._flow_nodes = [DataFlowNode(e, self) for e in self.leaf_events]
        # 根据节点的开始时间对 _flow_nodes 列表进行排序
        self._flow_nodes.sort(key=lambda x: x.start_time)
        # 调用验证方法，确保数据流图的一致性和有效性
        self.validate()

    # 获取数据流图中的流节点列表，作为只读属性
    @property
    def flow_nodes(self) -> Tuple[DataFlowNode, ...]:
        return tuple(self._flow_nodes)

    # 验证数据流图的方法
    def validate(self):
        # 检查每个（张量，版本号）对是否具有唯一的创建节点
        outputs: Set[Tuple[TensorKey, int]] = set()
        for node in self.flow_nodes:
            node_outputs = set(node.outputs.items())
            duplicates = outputs & node_outputs
            assert not duplicates, f"{node._event.name} {node._edges} {duplicates}"
            outputs |= node_outputs

        # 检查 self._flow_nodes 是否形成了一个有效的拓扑排序的有向无环图（DAG）
        tensor_versions: Dict[TensorKey, int] = {}
        for node in self.flow_nodes:
            # 检查每个输入张量的版本号是否符合预期
            for key, (_, version) in node.inputs.items():
                expected = tensor_versions.get(key, 0)
                assert expected == version, (expected, version)

            # 检查每个输出张量的版本号是否符合预期，并更新版本号信息
            for key, version in node.outputs.items():
                prior_version = tensor_versions.get(key, version)
                assert version >= prior_version, (version, prior_version)
                tensor_versions[key] = version

    # 获取数据流图中的叶子事件列表，作为只读属性
    @property
    def leaf_events(self) -> Tuple[_ProfilerEvent, ...]:
        return self._leaf_events

    # 静态方法的声明，未完待续
    def lookup(self, key: TensorKey) -> int:
        # 获取给定键的当前版本号，如果不存在则返回默认值0
        version = self._active_version.setdefault(key, 0)
        # 断言版本号不为None
        assert version is not None
        # 返回版本号
        return version

    def bump(self, key: TensorKey) -> None:
        # 获取给定键的当前版本号
        prior_version = self._active_version.get(key, None)
        # 断言先前版本号不为None
        assert prior_version is not None
        # 将给定键的版本号增加1
        self._active_version[key] = prior_version + 1

    def delete(self, key: TensorKey) -> None:
        # 确保给定键的当前版本号不为None
        assert self._active_version.setdefault(key, 0) is not None
        # 删除给定键的版本号，将其设为None
        self._active_version[key] = None
@dataclasses.dataclass
class CategoryElement:
    by_id: Optional[Category] = None
    by_key: Dict[TensorKey, Category] = dataclasses.field(default_factory=dict)
    by_version: Dict[TensorAndID, Category] = dataclasses.field(default_factory=dict)

    # Used by unit tests to check internals. (And consequently by
    # MemoryProfile.lookup) This should not be used in any other capacity.
    # 单元测试用于检查内部状态的属性。因此也被 MemoryProfile.lookup 使用，不应在其他情况下使用。
    _by_id_keyset: Set[TensorKey] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class CategoryDict:
    _values: DefaultDict[int, CategoryElement] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(CategoryElement)
    )

    def set_by_id(self, key: TensorKey, category: Category) -> None:
        # 设置按 ID 进行索引的分类
        self._values[key.id].by_id = category
        # 向 _by_id_keyset 集合中添加对应的 TensorKey
        self._values[key.id]._by_id_keyset.add(key)

    def set_by_key(self, key: TensorKey, category: Category) -> None:
        # 设置按 TensorKey 进行索引的分类
        self._values[key.id].by_key[key] = category

    def set_by_version(self, key: TensorKey, version: int, category: Category) -> None:
        # 设置按 TensorKey 和版本号进行索引的分类
        self._values[key.id].by_version[(key, version)] = category

    def setdefault_by_version(
        self, key: TensorKey, version: int, category: Category
    ) -> None:
        # 如果索引不存在则设置按 TensorKey 和版本号进行索引的分类
        self._values[key.id].by_version.setdefault((key, version), category)

    def get(self, key: Key, version: int) -> Optional[Category]:
        if isinstance(key, Key) and not isinstance(key, TensorKey):
            return None
        # 获取分类，优先按 ID，然后按 Key，最后按版本号进行索引
        element = self._values[key.id]
        return (
            element.by_id
            or element.by_key.get(key, None)
            or element.by_version.get((key, version), None)
        )


class MemoryProfile:
    def __init__(self, result: _ProfilerResult) -> None:
        # 初始化 MemoryProfile 对象，传入 _ProfilerResult 结果
        self._op_tree = OpTree(result)
        self._data_flow_graph = DataFlowGraph(self._op_tree)
        self._size_map = SizeMap(self._op_tree)
        self._categories = CategoryDict()

        # 调用私有方法设置梯度和临时变量
        self._set_gradients_and_temporaries()
        # 调用私有方法使用 Python 追踪器设置参数
        self._set_parameters_using_python_tracer()
        # 调用私有方法设置输入
        self._set_inputs()
        # 调用私有方法使用数据流设置参数
        self._set_parameters_using_data_flow()
        # 调用私有方法设置激活状态
        self._set_activations()
        # 调用私有方法设置优化器状态
        self._set_optimizer_state()
        # 调用私有方法设置自动求导详细信息
        self._set_autograd_detail()

    @property
    # 定义一个方法 `timeline`，返回类型为元组，包含元组组成的列表
    def timeline(self) -> Tuple[Tuple[int, Action, KeyAndID, int], ...]:
        # 初始化一个空列表 `output`，用于存储时间线事件元组
        output: List[Tuple[int, Action, KeyAndID, int]] = []
        # 初始化一个字典 `allocation_times`，用于存储分配时间
        allocation_times: Dict[Tuple[TensorKey, bool], int] = {}
        # 初始化一个字典 `live_unknown`，用于记录活跃但未知的对象
        live_unknown: Dict[Tuple[int, torch.device], Literal[True]] = {}
        
        # 遍历操作树 `_op_tree` 的深度优先遍历结果
        for event in self._op_tree.dfs():
            # 如果事件类型的第一个元素为分配事件
            if event.typed[0] == _EventType.Allocation:
                # 获取分配事件的详细字段
                alloc_fields = event.typed[1]
                # 获取分配大小
                alloc_size = alloc_fields.alloc_size
                # 判断是否为有效的分配操作
                is_allocation = alloc_size > 0
                # 获取事件的开始时间
                t = event.start_time_ns

                # 根据分配事件的字段创建张量键 `tkey`
                tkey = TensorKey.from_allocation(alloc_fields)
                # 如果张量键不为空，则记录分配时间
                if tkey is not None:
                    allocation_times[(tkey, is_allocation)] = t
                else:
                    # 否则，根据设备信息创建键 `key`
                    key = Key(alloc_fields.device)
                    ptr_and_device = (alloc_fields.ptr, key.device)
                    # 如果是分配操作且该指针和设备已经存在于 `live_unknown` 中
                    if is_allocation:
                        if ptr_and_device in live_unknown:
                            # 增加版本号
                            output.append((t, Action.INCREMENT_VERSION, (key, 0), alloc_size))
                        else:
                            # 将该指针和设备标记为活跃但未知，并创建对象
                            live_unknown[ptr_and_device] = True
                            output.append((t, Action.CREATE, (key, 0), alloc_size))
                    else:
                        # 如果是释放操作，销毁对象并记录
                        output.append((t, Action.DESTROY, (key, 0), -alloc_size))
                        # 如果该指针和设备已不存在于 `live_unknown` 中，则记录先前存在
                        if not live_unknown.pop(ptr_and_device, False):
                            output.append((-1, Action.PREEXISTING, (key, 0), -alloc_size))

        # 获取类别快照 `_category_snapshot`，并按键排序得到最后的版本号
        snapshot = self._category_snapshot()
        last_version = dict(sorted(snapshot.keys()))

        # 初始化事件列表 `events`，根据快照数据生成预存在的事件
        events: List[Tuple[int, Action, TensorAndID]] = [
            (-1, Action.PREEXISTING, (key, version))
            for key, version in snapshot.keys()
            # 排除在分配时间中出现且版本号为0的键
            if (key, True) not in allocation_times and version == 0
        ]

        # 遍历数据流图的节点 `_data_flow_graph.flow_nodes`
        for node in self._data_flow_graph.flow_nodes:
            # 遍历节点的边 `_edges`，处理分配、变异和删除事件
            for key, edge in node._edges.items():
                # 如果边表示分配事件
                if edge.is_allocation:
                    # 获取分配时间并记录创建事件
                    t = allocation_times[(key, True)]
                    events.append((t, Action.CREATE, (key, 0)))
                # 如果边表示变异事件
                elif edge.mutated:
                    # 获取事件的开始时间并记录版本增加事件
                    t = node._event.start_time_ns
                    version = edge.input_version
                    assert version is not None
                    events.append((t, Action.INCREMENT_VERSION, (key, version)))
                
                # 如果边表示删除事件
                if edge.is_deletion:
                    # 获取删除时间并记录销毁事件，并使用最后版本号作为参数
                    t = allocation_times[(key, False)]
                    events.append((t, Action.DESTROY, (key, last_version[key])))

        # 将事件列表 `events` 中的事件转化为输出格式，并扩展到 `output` 列表中
        output.extend(
            (time, action, (key, version), self._size_map[key])
            for time, action, (key, version) in events
        )

        # 根据时间和动作值对输出列表 `output` 进行排序
        output.sort(key=lambda x: (x[0], x[1].value))
        # 返回排序后的时间线事件元组作为结果
        return tuple(output)
    # 检查给定参数是否对应于梯度类别
    def _is_gradient(self, *args, **kwargs) -> bool:
        return self._categories.get(*args, **kwargs) == Category.GRADIENT

    # 返回一个字典，其键是 TensorAndID 对象，值是对应的 Category 或 None
    def _category_snapshot(self) -> Dict[TensorAndID, Optional[Category]]:
        # 创建一个空集合，用于存储所有 TensorAndID 对象
        all_tensor_versions: Set[TensorAndID] = set()

        # 遍历数据流图中的所有节点
        for node in self._data_flow_graph.flow_nodes:
            # 将节点输入中的每对键值添加到集合中
            all_tensor_versions.update(((k, v) for k, (_, v) in node.inputs.items()))
            # 将节点中间结果的每个键添加到集合中，值为 0
            all_tensor_versions.update((key, 0) for key in node.intermediates)
            # 将节点输出的每对键值添加到集合中
            all_tensor_versions.update(node.outputs.items())

        # 遍历 self._categories._values 中的每个值
        for i in self._categories._values.values():
            # 将每个值的 _by_id_keyset 中的每个键添加到集合中，值为 0
            all_tensor_versions.update((key, 0) for key in i._by_id_keyset)

        # 返回一个字典，键是 (key, version) 元组，值是 self._categories 中对应的 Category
        return {
            (key, version): self._categories.get(key, version)
            for key, version in sorted(all_tensor_versions)
        }

    # 返回一个集合，其中包含依赖于梯度的 Tensor 的 ID
    def _any_version_depends_on_gradient(self) -> Set[int]:
        """Extract IDs of Tensors which depend or will depend on a gradient.

        Note that this weakened definition of "depends" requires us to loop
        over the data flow graph multiple times because it allows dependency
        information to flow backward through edges and removes the guarantee
        that nodes are topologically sorted. (Or indeed, even that a valid
        topological order exists.) Put another way, we have converted an
        acyclic data flow graph into a cyclic graph and we are attempting to
        partition cycles involving a gradient from the rest of the graph.
        """
        # 创建一个空集合，用于存储依赖于梯度的 Tensor 的 ID
        depends_on_gradient: Set[int] = set()

        # 无限循环，直到不再有新的 Tensor ID 被添加
        while True:
            # 记录当前集合大小
            start_size = len(depends_on_gradient)

            # 遍历数据流图中的所有节点
            for node in self._data_flow_graph.flow_nodes:
                # 构造一个包含依赖于梯度的 Tensor ID 的元组
                ids = tuple(
                    key.id
                    for key, (_, version) in node.inputs.items()
                    if self._categories.get(key, version)
                    in (Category.GRADIENT, Category.PARAMETER)
                    or key.id in depends_on_gradient
                )

                # 如果 ids 非空，将其中的 ID 添加到 depends_on_gradient 中，并添加节点输出的 ID
                if ids:
                    depends_on_gradient.update(ids)
                    depends_on_gradient.update(key.id for key in node.outputs)

            # 当集合大小不再增加时，退出循环
            if len(depends_on_gradient) == start_size:
                return depends_on_gradient
    # 标记那些明显且容易推理的张量。

    # 梯度很容易检测。我们直接在 Python 追踪器中检查 `.grad` 属性，
    # 并且可以从 `AccumulateGrad` 操作中检测到任何新的梯度张量。
    for event in self._op_tree.dfs():
        for _, p_grad in extract_gradients(event):
            self._categories.set_by_id(p_grad, Category.GRADIENT)

    # 类似地，临时张量易于识别，并且标记它们非常有用，因为它们可能会使
    # 内存使用情况比预期的更"尖锐"。
    for node in self._data_flow_graph.flow_nodes:
        for i in node.intermediates:
            self._categories.set_by_key(i, Category.TEMPORARY)
    def _set_inputs(self) -> None:
        """
        Mark inputs based on which Tensors are updated using gradients.

        The process for differentiating between inputs and activations is more
        involved. Most Tensors in a training loop depend on at least one
        gradient: parameters depend on them through updates, and activations
        and optimizer state depend on them transitively through parameters.
        Critically, we do not need to know which Tensors are parameters to
        apply this method; we can simply walk the data flow graph to build the
        set of all values which depend on a gradient and then obtain the set
        of inputs from the conjugate set.

        There is, however, one hiccup. The first time we see a parameter is
        generally on the forward pass of the first step. We know from
        inspection of the data flow graph that v1 of that Tensor depends on
        a gradient (provided we profile an optimizer step), but not v0. To
        address this problem we weaken the definition of "depends on a
        gradient" to "any version of this Tensor depends on a gradient",
        which in turn strengthens the criteria for the input set enough to
        filter the activations in the forward pass of the first step.
        """

        # All of this analysis is predicated on using at least one training
        # step (or parameters from the python tracer) to partition the graph.
        # Absent that we cannot determine which Tensors are inputs and which
        # ones are part of the model.
        depends_on_gradient = self._any_version_depends_on_gradient()

        # We only want to annotate Tensors which actually contribute to the
        # model calculation.
        produces_gradient: Set[TensorAndID] = set()
        for node in reversed(self._data_flow_graph.flow_nodes):
            tensors = {(key, version) for key, (_, version) in node.inputs.items()}
            tensors |= node.outputs.items()
            if any(
                self._categories.get(*i) in (Category.GRADIENT, Category.PARAMETER)
                or i in produces_gradient
                for i in tensors
            ):
                produces_gradient |= tensors

        # Don't include Tensors created in the backward pass, as these are
        # generally Autograd implementation details rather than proper inputs.
        input_candidates = produces_gradient.copy()
        for node in self._data_flow_graph.flow_nodes:
            if RecordScope.BACKWARD_FUNCTION in get_scopes(node._event):
                input_candidates -= set(node.outputs.items())

        for key, version in input_candidates:
            if key.id not in depends_on_gradient:
                # Set the category of this Tensor version as an input if it does
                # not depend on any gradient.
                self._categories.setdefault_by_version(key, version, Category.INPUT)
    def _set_activations(self) -> None:
        """
        Flood the graph to identify activations.

        """
        
        # 定义所需的类别和允许的附加类别集合
        required = {Category.INPUT, Category.ACTIVATION}
        also_allowed = {Category.PARAMETER, Category.TEMPORARY}
        
        # 遍历数据流图中的每个节点
        for node in self._data_flow_graph.flow_nodes:
            # 收集节点输入的键值对集合
            inputs = {(key, value) for key, (_, value) in node.inputs.items()}
            # 获取输入键值对对应的类别集合
            input_categories = {self._categories.get(*i) for i in inputs}

            # 检查节点是否符合激活标识的条件
            if (
                (input_categories & required)  # 必需的类别都存在
                and not (input_categories - (required | also_allowed))  # 没有不允许的类别
                and RecordScope.BACKWARD_FUNCTION not in get_scopes(node._event)  # 不在反向传播函数中
            ):
                # 将节点输出的每个项设置为激活类别
                for i in node.outputs.items():
                    self._categories.setdefault_by_version(*i, Category.ACTIVATION)

    def _set_optimizer_state(self) -> None:
        """
        Set optimizer state based on PyCall events.

        """
        
        # 深度优先遍历操作树中的每个事件
        for event in self._op_tree.dfs():
            # 检查事件是否为 PyCall 类型且包含优化器
            if event.typed[0] == _EventType.PyCall and event.typed[1].optimizer:
                # 获取优化器的参数状态
                parameters = event.typed[1].optimizer.parameters
                # 遍历状态列表并设置优化器状态类别
                for _, t in it.chain(*[state for _, _, state in parameters]):
                    key = TensorKey.from_tensor(t)
                    if key is not None:
                        self._categories.set_by_id(key, Category.OPTIMIZER_STATE)

    def _set_autograd_detail(self):
        """
        Set autograd detail category for backward functions.

        """
        
        # 定义前一个版本的类别集合
        prior = {None, Category.AUTOGRAD_DETAIL}
        
        # 遍历数据流图中的每个节点
        for node in self._data_flow_graph.flow_nodes:
            # 检查节点是否在反向传播函数中
            if RecordScope.BACKWARD_FUNCTION in get_scopes(node._event):
                # 对于每个输出项，如果版本为 0 或前一个版本是自动梯度详细信息，则设置类别
                for key, version in node.outputs.items():
                    if version == 0 or self._categories.get(key, version - 1) in prior:
                        self._categories.setdefault_by_version(
                            key, version, Category.AUTOGRAD_DETAIL
                        )
# 定义一个类 MemoryProfileTimeline，用于处理内存分析的时间线数据
class MemoryProfileTimeline:
    def __init__(self, memory_profile):
        """The minimum representation of the memory profile timeline
        includes the memory timeline and categories. The timeline
        consists of [timestamp, action, (TensorKey, version), numbytes]
        elements, to denote any actions (pre-existing, create, destroy,
        or increment_version) that occurred to a specific Tensor for a
        chunk of memory. The categories help map each (TensorKey,
        version) pair into a category."""
        # 初始化函数，接收一个 memory_profile 对象作为参数
        # 从 memory_profile 中获取内存时间线和分类信息
        self.timeline = memory_profile.timeline
        self.categories = memory_profile._categories

    def _coalesce_timeline(self, device_str):
        """Convert the memory timeline and categories into a memory plot
        consisting of timestamps and their respective sizes by category
        for a given device.

        Input: device
        Output: [timestamps, sizes by category]
        """
        # 将内存时间线和分类信息转换为给定设备上的内存图
        device = torch.device(device_str)
        times: List[int] = []  # 时间戳列表
        sizes: List[List[int]] = []  # 按分类存储的尺寸列表

        def update(key, version, delta):
            # 更新指定键和版本的分类尺寸
            category = (
                self.categories.get(key, version)
                if isinstance(key, TensorKey)
                else None
            )
            index = _CATEGORY_TO_INDEX[category] + 1  # 根据分类获取索引
            sizes[-1][index] += int(delta)  # 更新对应分类的尺寸信息

        t_min = -1  # 最小时间戳的初始值
        for t, action, (key, version), numbytes in self.timeline:
            if key.device != device:
                continue  # 如果不是指定设备的操作则跳过

            # 将时间戳从纳秒转换为微秒，以匹配跟踪事件的单位
            if t != -1:
                t = int(t / 1000)

            # 保存最小时间戳以填充预先存在的分配
            if t_min == -1 or (t < t_min and t > 0):
                t_min = t

            # 处理时间步长
            if len(times) == 0:
                times.append(t)
                sizes.append([0] + [0 for _ in _CATEGORY_TO_INDEX])  # 初始化尺寸列表

            elif t != times[-1]:
                times.append(t)
                sizes.append(sizes[-1].copy())  # 复制上一个时间步长的尺寸信息

            # 处理内存和分类
            if action in (Action.PREEXISTING, Action.CREATE):
                update(key, version, numbytes)  # 创建或者更新内存使用情况

            elif action == Action.INCREMENT_VERSION:
                update(key, version, -numbytes)  # 减少旧版本内存使用
                update(key, version + 1, numbytes)  # 增加新版本内存使用

            elif action == Action.DESTROY:
                update(key, version, -numbytes)  # 销毁释放内存

            else:
                raise ValueError(f"Unknown action: {action}")  # 未知操作类型错误

        times = [t_min if t < 0 else t for t in times]  # 处理负时间戳为最小时间
        return times, sizes  # 返回处理后的时间戳和尺寸列表
    def export_memory_timeline(self, path, device_str) -> None:
        """将内存时间线保存为 JSON 格式文件，包含时间和按类别大小分类的数据，
        保存到指定路径下，针对指定设备。"""
        times, sizes = self._coalesce_timeline(device_str)
        # TODO: Write a faster serialize (orjson not available in CI)
        import json

        with open(path, "w") as f:
            json.dump([times, sizes], f)

    def export_memory_timeline_raw(self, path, device_str) -> None:
        """将内存时间线保存为原始内存事件元组的 JSON 格式文件，
        格式为 (时间戳, 操作, 字节数, 类别索引)，保存到指定路径下，针对指定设备。"""
        device = torch.device(device_str)
        raw_events: List[Tuple[int, int, int, int]] = []

        def get_category_index(key, version):
            category = (
                self.categories.get(key, version)
                if isinstance(key, TensorKey)
                else None
            )
            return _CATEGORY_TO_INDEX[category]

        for t, action, (key, version), numbytes in self.timeline:
            if key.device != device:
                continue

            if action in (Action.PREEXISTING, Action.CREATE):
                raw_events.append(
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        numbytes,
                        get_category_index(key, version),
                    )
                )

            elif action == Action.INCREMENT_VERSION:
                raw_events.append(
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        -numbytes,
                        get_category_index(key, version),
                    )
                )
                raw_events.append(
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        numbytes,
                        get_category_index(key, version + 1),
                    )
                )

            elif action == Action.DESTROY:
                raw_events.append(
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        -numbytes,
                        get_category_index(key, version),
                    )
                )

            else:
                raise ValueError(f"Unknown action: {action}")

        import json

        with open(path, "w") as f:
            json.dump(raw_events, f)

    def export_memory_timeline_html(
        self, path, device_str, figsize=(20, 12), title=None
    ):
        """导出内存时间线为 HTML 文件，包含指定设备的数据，可指定图表大小和标题。"""
        # 该函数未完全列出，请在代码中查找并继续注释。
    ) -> None:
        """Exports the memory timeline as an HTML file which contains
        the memory timeline plot embedded as a PNG file."""
        # 检查用户是否安装了 matplotlib，如果没有则优雅地返回
        import importlib.util

        matplotlib_spec = importlib.util.find_spec("matplotlib")
        if matplotlib_spec is None:
            print(
                "export_memory_timeline_html failed because matplotlib was not found."
            )
            return

        from base64 import b64encode
        from os import remove
        from tempfile import NamedTemporaryFile

        import matplotlib.pyplot as plt
        import numpy as np

        # 合并设备的内存时间线数据
        mt = self._coalesce_timeline(device_str)
        times, sizes = np.array(mt[0]), np.array(mt[1])
        # 将时间线起始时间设为0，以匹配 Chrome 追踪的格式
        t_min = min(times)
        times -= t_min
        stacked = np.cumsum(sizes, axis=1) / 1024**3
        device = torch.device(device_str)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)

        # 绘制内存时间线的堆叠数据图
        fig = plt.figure(figsize=figsize, dpi=80)
        axes = fig.gca()
        for category, color in _CATEGORY_TO_COLORS.items():
            i = _CATEGORY_TO_INDEX[category]
            axes.fill_between(
                times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
            )
        # 设置图例，显示不同类别的内存使用情况
        fig.legend(["Unknown" if i is None else i.name for i in _CATEGORY_TO_COLORS])
        # X轴为时间（毫秒），Y轴为内存（GB）
        axes.set_xlabel("Time (ms)")
        axes.set_ylabel("Memory (GB)")
        # 设置图表标题，包括最大内存分配和最大内存保留信息
        title = "\n\n".join(
            ([title] if title else [])
            + [
                f"Max memory allocated: {max_memory_allocated/(1024**3):.2f} GiB \n"
                f"Max memory reserved: {max_memory_reserved/(1024**3):.2f} GiB"
            ]
        )
        axes.set_title(title)

        # 将内存时间线图像嵌入到 HTML 文件中
        tmpfile = NamedTemporaryFile("wb", suffix=".png", delete=False)
        tmpfile.close()
        fig.savefig(tmpfile.name, format="png")

        with open(tmpfile.name, "rb") as tmp:
            encoded = b64encode(tmp.read()).decode("utf-8")
            # 生成包含内存时间线图像的 HTML 文件内容
            html = f"""<html>
<head><meta charset="utf-8" /><title>GPU Memory Timeline HTML</title></head>
<body>
  <img src='data:image/png;base64,{encoded}'>
</body>
</html>"""

页面的 HTML 结构，包括了一个 `<head>` 标签用于设置页面的字符集和标题，以及一个 `<body>` 标签用于页面主体内容，其中包含了一个 `<img>` 标签用于显示 base64 编码的 PNG 图片。


            with open(path, "w") as f:
                f.write(html)

使用 `with` 语句打开文件 `path`，以写入模式 (`"w"`) 打开，并将变量 `html` 中的内容写入文件 `f`。


        remove(tmpfile.name)

调用 `remove` 函数，删除临时文件 `tmpfile.name`。
```