# `.\pytorch\torch\_export\serde\serialize.py`

```py
# mypy: allow-untyped-defs
# 导入必要的库和模块
import base64                 # 导入base64编解码模块
import copy                   # 导入复制对象的模块
import copyreg                # 导入用于注册pickle支持的函数
import dataclasses            # 导入用于定义数据类的模块
import heapq                  # 导入堆队列算法的模块
import inspect                # 导入用于检查对象的模块
import io                     # 导入用于处理流的模块
import json                   # 导入用于处理JSON数据的模块
import logging                # 导入日志记录模块
import math                   # 导入数学函数的模块
import operator               # 导入运算符模块
import re                     # 导入正则表达式操作模块
import typing                 # 导入类型提示模块

from contextlib import contextmanager  # 导入上下文管理器模块
from dataclasses import dataclass, field  # 导入用于定义数据类和字段的装饰器
from enum import Enum         # 导入枚举类型的模块
from typing import (           # 导入多种类型提示，包括：
    Any,                      # 任意类型
    Callable,                 # 可调用对象类型
    cast,                     # 强制类型转换函数
    Dict,                     # 字典类型
    final,                    # 用于标记终态变量的装饰器
    Iterator,                 # 迭代器类型
    List,                     # 列表类型
    Optional,                 # 可选类型
    Set,                      # 集合类型
    Tuple,                    # 元组类型
    Type,                     # 类型对象
    Union,                    # 联合类型
)

import sympy                  # 导入符号计算库

import torch                  # 导入PyTorch深度学习库
import torch.export.exported_program as ep  # 导入PyTorch导出程序相关模块
from torch._export.serde.schema import SchemaVersion  # 导入PyTorch序列化模式版本
from torch._export.verifier import load_verifier     # 导入PyTorch加载验证器函数
from torch._library.fake_class_registry import FakeScriptObject  # 导入假脚本对象注册模块
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode  # 导入假张量相关模块
from torch.fx.experimental import symbolic_shapes  # 导入PyTorch符号形状实验模块
from torch.utils import _pytree as pytree           # 导入PyTorch树结构工具模块
from torch.utils._pytree import treespec_dumps, treespec_loads  # 导入树规范的序列化和反序列化函数
from torch.utils._sympy.value_ranges import ValueRanges  # 导入值范围处理模块
from torch.utils._sympy.numbers import int_oo        # 导入无穷大符号整数模块

from .schema import (  # 导入当前包中的schema模块，并列出如下类和对象：
    Argument,                     # 参数
    BufferMutationSpec,           # 缓冲区变异规范
    ConstantInputSpec,            # 常数输入规范
    ConstantValue,                # 常量值
    CustomObjArgument,            # 自定义对象参数
    Device,                       # 设备
    ExportedProgram,              # 导出程序
    GradientToParameterSpec,      # 梯度到参数规范
    GradientToUserInputSpec,      # 梯度到用户输入规范
    Graph,                        # 图
    GraphArgument,                # 图参数
    GraphModule,                  # 图模块
    GraphSignature,               # 图签名
    InputSpec,                    # 输入规范
    InputToBufferSpec,            # 输入到缓冲区规范
    InputToCustomObjSpec,         # 输入到自定义对象规范
    InputTokenSpec,               # 输入令牌规范
    InputToParameterSpec,         # 输入到参数规范
    InputToTensorConstantSpec,    # 输入到张量常量规范
    Layout,                       # 布局
    LossOutputSpec,               # 损失输出规范
    MemoryFormat,                 # 存储格式
    ModuleCallEntry,              # 模块调用条目
    ModuleCallSignature,          # 模块调用签名
    NamedArgument,                # 命名参数
    Node,                         # 节点
    OptionalTensorArgument,       # 可选张量参数
    OutputSpec,                   # 输出规范
    OutputTokenSpec,              # 输出令牌规范
    RangeConstraint,              # 范围约束
    ScalarType,                   # 标量类型
    SCHEMA_VERSION,               # 模式版本
    SymBool,                      # 符号布尔
    SymBoolArgument,              # 符号布尔参数
    SymExpr,                      # 符号表达式
    SymExprHint,                  # 符号表达式提示
    SymInt,                       # 符号整数
    SymIntArgument,               # 符号整数参数
    TensorArgument,               # 张量参数
    TensorMeta,                   # 张量元数据
    TokenArgument,                # 令牌参数
    TREESPEC_VERSION,             # 树规范版本
    UserInputMutationSpec,        # 用户输入变异规范
    UserInputSpec,                # 用户输入规范
    UserOutputSpec,               # 用户输出规范
)
from .union import _Union      # 从当前包的union模块中导入_Union对象


__all__ = [                    # 定义公开的模块成员列表
    "serialize",               # 序列化函数
    "GraphModuleSerializer",   # 图模块序列化器
    "ExportedProgramSerializer",  # 导出程序序列化器
    "GraphModuleDeserializer",  # 图模块反序列化器
    "ExportedProgramDeserializer",  # 导出程序反序列化器
]

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class SerializeError(RuntimeError):  # 定义自定义异常类SerializeError，继承自RuntimeError
    pass


def _reverse_map(d: Dict[Any, Enum]):  # 定义函数_reverse_map，接受一个字典参数d，返回一个反向映射的字典
    return {v.value: k for k, v in d.items()}  # 使用字典推导式实现反向映射


MetaType = Union[                 # 定义MetaType类型别名，可以是以下类型之一：
    FakeTensor,                   # 假张量
    int,                          # 整数
    torch.SymInt,                 # PyTorch符号整数
    bool,                         # 布尔值
    torch.SymBool,                # PyTorch符号布尔
    ep.CustomObjArgument          # 导出程序自定义对象参数
]


ST_DELIMITER = ";"  # 定义字符串分隔符ST_DELIMITER为分号

_TORCH_TO_SERIALIZE_DTYPE = {  # 定义映射_TORCH_TO_SERIALIZE_DTYPE，将PyTorch张量类型映射到序列化的标量类型
    torch.uint8: ScalarType.BYTE,         # 无符号8位整数映射为BYTE
    torch.int8: ScalarType.CHAR,           # 8位整数映射为CHAR
    torch.int16: ScalarType.SHORT,         # 16位整数映射为SHORT
    torch.int32: ScalarType.INT,           # 32位整数映射为INT
    torch.int64: ScalarType.LONG,          # 64位整数映射为LONG
    torch.float16: ScalarType.HALF,        # 16位浮点数映射为HALF
    torch.float32: ScalarType.FLOAT,       # 32位浮点数映射为FLOAT
    torch.float64: ScalarType.DOUBLE,      # 64位浮点数映射为DOUBLE
    torch.complex32: ScalarType.COMPLEXHALF,  # 32位复数映
# 将 _TORCH_TO_SERIALIZE_DTYPE 的键值反转，存储到 _SERIALIZE_TO_TORCH_DTYPE 中
_SERIALIZE_TO_TORCH_DTYPE = _reverse_map(_TORCH_TO_SERIALIZE_DTYPE)  # type: ignore[arg-type]


# 将 torch 稀疏张量布局映射到对应的 Layout 枚举值
_TORCH_TO_SERIALIZE_LAYOUT = {
    torch.sparse_coo: Layout.SparseCoo,
    torch.sparse_csr: Layout.SparseCsr,
    torch.sparse_csc: Layout.SparseCsc,
    torch.sparse_bsr: Layout.SparseBsr,
    torch.sparse_bsc: Layout.SparseBsc,
    torch._mkldnn: Layout._mkldnn,  # type: ignore[attr-defined]
    torch.strided: Layout.Strided,
}


# 将 _TORCH_TO_SERIALIZE_LAYOUT 的键值反转，存储到 _SERIALIZE_TO_TORCH_LAYOUT 中
_SERIALIZE_TO_TORCH_LAYOUT = _reverse_map(_TORCH_TO_SERIALIZE_LAYOUT)  # type: ignore[arg-type]


# 将 torch 张量内存格式映射到对应的 MemoryFormat 枚举值
_TORCH_TO_SERIALIZE_MEMORY_FORMAT = {
    torch.contiguous_format: MemoryFormat.ContiguousFormat,
    torch.channels_last: MemoryFormat.ChannelsLast,
    torch.channels_last_3d: MemoryFormat.ChannelsLast3d,
    torch.preserve_format: MemoryFormat.PreserveFormat,
}


# 将 _TORCH_TO_SERIALIZE_MEMORY_FORMAT 的键值反转，存储到 _SERIALIZE_TO_TORCH_MEMORY_FORMAT 中
_SERIALIZE_TO_TORCH_MEMORY_FORMAT = _reverse_map(_TORCH_TO_SERIALIZE_MEMORY_FORMAT)  # type: ignore[arg-type]


# 定义支持符号整数操作的集合
_SYM_INT_OPS = {
    operator.mul,
    operator.add,
    operator.sub,
    operator.floordiv,
    operator.mod,
    torch.sym_int,
    torch.sym_float,
    torch.sym_ite,
    torch.sym_max,
    torch.sym_min,
    torch.sym_sqrt,
}


# 定义支持符号布尔操作的集合
_SYM_BOOL_OPS = {
    operator.eq,
    operator.ne,
    operator.le,
    operator.ge,
    operator.lt,
    operator.gt,
    torch.sym_not,
}


@dataclass
class SerializedArtifact:
    # 序列化的艺术品类，包含导出程序、状态字典、常量和示例输入的字节数据
    exported_program: bytes
    state_dict: bytes
    constants: bytes
    example_inputs: bytes


@dataclass
class _SerializedProgram:
    # 序列化的程序类，包含导出程序、状态字典、常量和示例输入的字节数据
    exported_program: ExportedProgram
    state_dict: bytes
    constants: bytes
    example_inputs: bytes


def deserialize_device(d: Device) -> torch.device:
    # 根据输入的设备信息反序列化为 torch.device 对象
    if d.index is None:
        return torch.device(type=d.type)  # type: ignore[call-overload]
    return torch.device(type=d.type, index=d.index)


def serialize_sym_int(s: Union[int, torch.SymInt]) -> SymInt:
    # 序列化符号整数或整数为 SymInt 对象
    if isinstance(s, (torch.SymInt, int)):
        if symbolic_shapes.is_concrete_int(s):
            return SymInt.create(as_int=int(s))
        else:
            assert isinstance(s, torch.SymInt)
            if s.node.hint is None:
                return SymInt.create(as_expr=SymExpr(str(s)))
            else:
                return SymInt.create(
                    as_expr=SymExpr(str(s), hint=SymExprHint.create(as_int=s.node.hint))
                )
    else:
        raise SerializeError(
            f"SymInt should be either symbol or int, got `{s}` of type `{type(s)}`"
        )


def serialize_sym_bool(s: Union[bool, torch.SymBool]) -> SymBool:
    # 序列化符号布尔或布尔为 SymBool 对象
    if isinstance(s, (torch.SymBool, bool)):
        if symbolic_shapes.is_concrete_bool(s):
            return SymBool.create(as_bool=bool(s))
        else:
            return SymBool.create(as_expr=SymExpr(expr_str=str(s)))
    else:
        raise SerializeError(
            f"SymBool should be either symbol or bool, got `{s}` of type `{type(s)}`"
        )


def serialize_tensor_meta(t: torch.Tensor) -> TensorMeta:
    """
    Extract a TensorMeta describing `t`.
    """
    # 返回一个 TensorMeta 对象，描述给定张量的元信息
    return TensorMeta(
        # 将 PyTorch 张量的数据类型转换为序列化后的数据类型
        dtype=_TORCH_TO_SERIALIZE_DTYPE[t.dtype],
        # 将张量的每个维度大小序列化为符号整数列表
        sizes=[serialize_sym_int(s) for s in t.shape],
        # 指示张量是否需要梯度
        requires_grad=t.requires_grad,
        # 创建描述设备的 Device 对象，包括设备类型和索引
        device=Device(type=t.device.type, index=t.device.index),
        # 将张量的每个维度步长序列化为符号整数列表
        strides=[serialize_sym_int(s) for s in t.stride()],
        # 设置张量数据的起始偏移量，当前为固定值 0，需要修复
        storage_offset=serialize_sym_int(0),  # TODO 需要修复
        # 将 PyTorch 张量的布局转换为序列化后的布局
        layout=_TORCH_TO_SERIALIZE_LAYOUT[t.layout],
    )
# 当前反序列化器的全局变量，默认为 None
_CURRENT_DESERIALIZER: Optional["GraphModuleDeserializer"] = None


def _reduce_fake_tensor(fake_tensor: FakeTensor):
    # 判断 fake_tensor 是否为 torch.nn.Parameter 类型
    is_parameter = isinstance(fake_tensor, torch.nn.Parameter)
    # 序列化 fake_tensor 的元信息
    tensor_meta = serialize_tensor_meta(fake_tensor)
    # 将序列化后的元信息转换为 JSON 格式的字节流
    tensor_meta_bytes = json.dumps(
        _dataclass_to_dict(tensor_meta), cls=EnumEncoder
    ).encode("utf-8")
    # 返回重建 fake_tensor 所需的函数和参数元组
    return _reconstruct_fake_tensor, (tensor_meta_bytes, is_parameter)


def _reconstruct_fake_tensor(
    serialized_tensor_meta: bytes, is_parameter: bool
) -> FakeTensor:
    # 将字节流反序列化为 TensorMeta 对象
    json_tensor_meta = json.loads(serialized_tensor_meta.decode("utf-8"))
    tensor_meta = _dict_to_dataclass(TensorMeta, json_tensor_meta)
    # 获取当前的 fake mode 反序列化器
    assert (
        _CURRENT_DESERIALIZER is not None
    ), "Need access to current deserializer state"
    # 使用当前反序列化器解析 tensor_meta 生成 fake_tensor
    fake_tensor = _CURRENT_DESERIALIZER.deserialize_tensor_meta(tensor_meta)
    # 如果 is_parameter 为 True，则将 fake_tensor 转换为 torch.nn.Parameter 类型
    if is_parameter:
        fake_tensor = torch.nn.Parameter(fake_tensor)  # type: ignore[assignment]
    return fake_tensor


def serialize_torch_artifact(artifact: Optional[Any]) -> bytes:
    # 如果 artifact 为 None，则返回空字节串
    if artifact is None:
        return b""

    # 确保在复制注册分发表中没有 FakeTensor，以避免覆盖现有的 FakeTensor 缩减器
    assert (
        FakeTensor not in copyreg.dispatch_table
    ), "Refusing to stomp on existing FakeTensor reducer"
    try:
        # 注册 FakeTensor 类型的自定义缩减器 _reduce_fake_tensor
        copyreg.pickle(FakeTensor, _reduce_fake_tensor)
        buffer = io.BytesIO()
        # 由于后端的张量反序列化问题，暂时将 artifact 移至 CPU 后保存
        # TODO: 应通过反序列化修复此问题，而非在保存时进行修复
        torch.save(artifact, buffer)
        return buffer.getvalue()
    finally:
        # 清除 FakeTensor 在复制注册分发表中的缩减器
        del copyreg.dispatch_table[FakeTensor]


def deserialize_torch_artifact(serialized: Union[Dict[str, Any], Tuple[Any, ...], bytes]):
    # 如果 serialized 是字典或元组类型，则直接返回
    if isinstance(serialized, (dict, tuple)):
        return serialized
    # 如果 serialized 长度为 0，则返回空字典
    if len(serialized) == 0:
        return {}
    buffer = io.BytesIO(serialized)
    buffer.seek(0)
    # 从缓冲区中加载 artifact
    artifact = torch.load(buffer)
    assert isinstance(artifact, (tuple, dict))
    return artifact


def _sympy_int_to_int(val: sympy.Expr, adjust: str):
    # 将简单的 sympy 整数转换为具体的 int 类型
    if val in (sympy.oo, int_oo):
        return math.inf
    if val in (-sympy.oo, -int_oo):
        return -math.inf
    if isinstance(val, sympy.Integer):
        return int(val)

    # TODO: 当 Ed 去除分数范围时，删除此调整
    log.warning(
        "Export constraints cannot be non-integer expressions. Found "
        "type %s, and value %s. We will attempt to %s "
        "this value.", type(val), val, adjust
    )

    # 根据 adjust 参数进行数值调整
    if adjust == "floor":
        return math.floor(val)
    elif adjust == "ceil":
        return math.ceil(val)
    else:
        # 如果adjust不是预期的有效值，抛出运行时错误异常
        raise RuntimeError(f"Got invalid adjustment {adjust}")
# 将具体整数转换为简单的 sympy 整数表达式
def _int_to_sympy_int(val) -> sympy.Expr:
    if val == math.inf:
        return int_oo  # 如果值为正无穷大，则返回 sympy 的正无穷大表示
    if val == -math.inf:
        return -int_oo  # 如果值为负无穷大，则返回 sympy 的负无穷大表示
    return sympy.Integer(val)  # 否则，返回普通的 sympy 整数表示


# 序列化范围约束
def serialize_range_constraints(
    range_constraints: Dict[sympy.Symbol, ValueRanges]
) -> Dict[str, RangeConstraint]:
    return {
        str(k): RangeConstraint(
            _sympy_int_to_int(v.lower, "ceil"),  # 将下限转换为整数，并向上取整
            _sympy_int_to_int(v.upper, "floor"),  # 将上限转换为整数，并向下取整
        )
        for k, v in range_constraints.items()
    }


# 从目标对象获取模式(schema)
def _get_schema_from_target(target):
    if isinstance(target, torch._ops.OpOverload):
        return target._schema  # 如果目标对象是 torch 操作重载对象，则返回其模式(schema)
    elif type(target) in _serialization_registry:
        return _serialization_registry[type(target)].op_schema(type(target))  # 如果目标对象在序列化注册表中，返回其模式(schema)
    raise RuntimeError(f"Cannot find schema for {type(target)}")  # 否则，抛出运行时错误，表示找不到该类型的模式(schema)


# 判断是否返回单个张量
def _is_single_tensor_return(target: torch._ops.OpOverload) -> bool:
    schema = _get_schema_from_target(target)
    returns = schema.returns
    return len(returns) == 1 and isinstance(returns[0].real_type, torch.TensorType)


# 判断是否返回单个张量列表
def _is_single_tensor_list_return(target: Any) -> bool:
    schema = _get_schema_from_target(target)
    returns = schema.returns

    if len(returns) != 1:
        return False
    return_type = returns[0].real_type
    return isinstance(return_type, torch.ListType) and isinstance(
        return_type.getElementType(), torch.TensorType
    )


# 图状态类，用于表示计算图中的状态信息
@dataclass
class GraphState:
    inputs: List[Argument] = field(default_factory=list)  # 输入参数列表
    outputs: List[Argument] = field(default_factory=list)  # 输出参数列表
    nodes: List[Node] = field(default_factory=list)  # 节点列表
    tensor_values: Dict[str, TensorMeta] = field(default_factory=dict)  # 张量值的字典
    sym_int_values: Dict[str, SymInt] = field(default_factory=dict)  # 符号整数值的字典
    sym_bool_values: Dict[str, SymBool] = field(default_factory=dict)  # 符号布尔值的字典
    is_single_tensor_return: bool = False  # 是否返回单个张量
    custom_obj_values: Dict[str, CustomObjArgument] = field(default_factory=dict)  # 自定义对象值的字典


# Final 元类，用于禁止继承
class Final(type):
    def __new__(metacls, name, bases, classdict):
        for b in bases:
            if isinstance(b, Final):
                raise TypeError(f"type '{b.__name__}' is not an acceptable base type")
        return type.__new__(metacls, name, bases, dict(classdict))


# 图模块序列化器，使用 Final 元类以禁止继承
@final
class GraphModuleSerializer(metaclass=Final):
    def __init__(
        self,
        graph_signature: ep.ExportGraphSignature,  # 导出图签名对象
        module_call_graph: List[ep.ModuleCallEntry],  # 模块调用图列表
    ):
        self.graph_state = GraphState()  # 初始化图状态
        self.graph_signature = graph_signature  # 设置图签名对象
        self.module_call_graph = module_call_graph  # 设置模块调用图列表
        self.custom_objs: Dict[str, torch._C.ScriptObject] = {}  # 自定义对象字典
        self.duplicate_getitem_nodes: Dict[str, str] = {}  # 重复获取项节点字典

    # 保存图状态的上下文管理器
    @contextmanager
    def save_graph_state(self):
        saved = self.graph_state  # 保存当前图状态
        self.graph_state = GraphState()  # 重置图状态
        try:
            yield
        finally:
            self.graph_state = saved  # 恢复保存的图状态
    # 处理占位符节点，根据节点操作类型为"placeholder"进行处理
    def handle_placeholder(self, node: torch.fx.Node):
        # 断言节点操作为"placeholder"
        assert node.op == "placeholder"
        
        # 根据节点的值类型进行不同处理
        if isinstance(node.meta["val"], torch.Tensor):
            # 如果节点值是 torch.Tensor 类型，创建一个作为张量参数的 Argument 对象
            graph_input = Argument.create(as_tensor=TensorArgument(name=node.name))
            # 将张量的元数据序列化并存储到图状态的张量值字典中
            self.graph_state.tensor_values[node.name] = serialize_tensor_meta(
                node.meta["val"]
            )
        elif isinstance(node.meta["val"], torch.SymInt):
            # 如果节点值是 torch.SymInt 类型，抛出错误，暂未实现对 SymInt 的处理
            raise AssertionError("SymInt graph input is not implemented yet.")
        elif isinstance(node.meta["val"], (int, bool, str, float, type(None))):
            # 如果节点值是基本类型（int、bool、str、float、NoneType），序列化输入并创建 Argument 对象
            graph_input = self.serialize_input(node.meta["val"])
        elif isinstance(node.meta["val"], ep.CustomObjArgument):
            # 如果节点值是自定义对象类型，获取类的全限定名，创建一个作为自定义对象参数的 Argument 对象
            class_fqn = node.meta["val"].class_fqn
            graph_input = Argument.create(
                as_custom_obj=CustomObjArgument(name=node.name, class_fqn=class_fqn)
            )
            # 序列化自定义对象的元数据并存储到图状态的自定义对象值字典中
            self.graph_state.custom_obj_values[node.name] = (
                self.serialize_script_obj_meta(node.meta["val"])
            )
        else:
            # 如果节点值类型未被实现的情况下，抛出断言错误
            raise AssertionError(f"Unimplemented graph input type: {node.meta['val']}")
        
        # 将处理后的输入参数添加到图状态的输入列表中
        self.graph_state.inputs.append(graph_input)

    # 处理输出节点，根据节点操作类型为"output"进行处理
    def handle_output(self, node: torch.fx.Node):
        # 断言节点操作为"output"
        assert node.op == "output"
        # 断言节点的参数列表长度为1，FX.Node 的参数应该只有一个参数
        assert len(node.args) == 1, "FX.Node's args should have one arg"
        
        # 获取节点的参数
        node_args = node.args[0]
        
        if isinstance(node_args, torch.fx.Node):
            # 如果节点参数是 torch.fx.Node 类型，表示是单个张量的返回值
            self.graph_state.is_single_tensor_return = True
            # 将该节点参数序列化并作为图状态的输出列表中的唯一元素
            self.graph_state.outputs = [self.serialize_input(node_args)]
        else:
            # 否则，断言节点参数是 tuple 或者 list 类型
            assert isinstance(node_args, (tuple, list))
            # 将节点参数序列化后作为图状态的输出列表
            self.graph_state.outputs = [self.serialize_input(arg) for arg in node_args]

    # 序列化操作符，返回目标对象的字符串表示
    def serialize_operator(self, target) -> str:
        if isinstance(target, str):
            # 如果目标对象是字符串，直接返回
            return target
        elif target.__module__.startswith("torch._ops"):
            # 如果目标对象的模块名以"torch._ops"开头，替换为"torch.ops"，并返回完整的函数名路径
            # TODO(zhxchen17) Maybe provide a function name helper in FX.
            # From torch.fx.node._get_qualified_name
            module = target.__module__.replace("torch._ops", "torch.ops")
            return f"{module}.{target.__name__}"
        else:
            # 否则，返回目标对象的模块名和对象名的字符串表示
            return f"{target.__module__}.{target.__name__}"
    # 处理调用函数节点，根据节点类型进行不同的处理逻辑
    def handle_call_function(self, node: torch.fx.Node):
        # 断言节点操作为 "call_function"
        assert node.op == "call_function"

        # 如果节点的目标是 operator.getitem，则在生产者节点已经处理，这里跳过处理
        if node.target is operator.getitem:
            return

        # 如果节点的目标在 _SYM_INT_OPS 中，则处理为符号整数操作
        if node.target in _SYM_INT_OPS:
            assert len(node.kwargs) == 0
            # 获取节点的元数据中的值
            meta_val = node.meta["val"]
            # 创建新的节点对象
            ex_node = Node(
                target=self.serialize_operator(node.target),  # 序列化操作符
                inputs=self.serialize_sym_op_inputs(node.target, node.args),  # 序列化符号操作符的输入
                outputs=[
                    Argument.create(
                        as_sym_int=self.serialize_sym_int_output(node.name, meta_val)  # 序列化符号整数输出
                    )
                ],
                metadata=self.serialize_metadata(node),  # 序列化节点的元数据
            )
        # 如果节点的目标在 _SYM_BOOL_OPS 中，则处理为符号布尔操作
        elif node.target in _SYM_BOOL_OPS:
            assert len(node.kwargs) == 0
            # 获取节点的元数据中的值
            meta_val = node.meta["val"]
            # 创建新的节点对象
            ex_node = Node(
                target=self.serialize_operator(node.target),  # 序列化操作符
                inputs=self.serialize_sym_op_inputs(node.target, node.args),  # 序列化符号操作符的输入
                outputs=[
                    Argument.create(
                        as_sym_bool=self.serialize_sym_bool_output(node.name, meta_val)  # 序列化符号布尔输出
                    )
                ],
                metadata=self.serialize_metadata(node),  # 序列化节点的元数据
            )
        # 如果节点的目标是 torch._ops.OpOverload 类的实例，则处理为重载操作
        elif isinstance(node.target, torch._ops.OpOverload):
            # 创建新的节点对象
            ex_node = Node(
                target=self.serialize_operator(node.target),  # 序列化操作符
                inputs=self.serialize_inputs(node.target, node.args, node.kwargs),  # 序列化输入
                outputs=self.serialize_outputs(node),  # 序列化输出
                # TODO: 在此处创建一个新的 tensor_values，元数据可能包含 faketensor 信息
                metadata=self.serialize_metadata(node),  # 序列化节点的元数据
            )
        # 如果节点的目标是 torch._ops.HigherOrderOperator 类的实例，则处理为高阶操作符
        elif isinstance(node.target, torch._ops.HigherOrderOperator):
            # 创建新的节点对象
            ex_node = Node(
                target=self.serialize_operator(node.target),  # 序列化操作符
                inputs=self.serialize_hoo_inputs(node.args, node.kwargs),  # 序列化高阶操作符的输入
                outputs=self.serialize_hoo_outputs(node),  # 序列化高阶操作符的输出
                metadata=self.serialize_metadata(node),  # 序列化节点的元数据
            )
        # 如果节点的目标类型在 _serialization_registry 中注册过，则由自定义操作处理器处理
        elif type(node.target) in _serialization_registry:
            custom_op_handler = node.target

            # 对未处理的序列化进行健全性检查
            assert type(node.target) in _serialization_registry, f"Miss {type(node.target)} CustomOpHandler"

            handler = _serialization_registry[type(node.target)]
            # 创建新的节点对象
            ex_node = Node(
                target=f"${handler.namespace()}:{handler.op_name(node.target)}",  # 自定义操作的命名空间和操作名
                inputs=self.serialize_inputs(node.target, node.args, node.kwargs),  # 序列化输入
                outputs=self.serialize_outputs(node),  # 序列化输出
                metadata=self.serialize_metadata(node),  # 序列化节点的元数据
            )
        # 如果以上条件都不满足，则抛出序列化错误
        else:
            raise SerializeError(f"Serializing {node.target} is not supported")

        # 将处理后的节点对象添加到图的节点列表中
        self.graph_state.nodes.append(ex_node)

    # 处理获取属性节点，暂时未实现具体处理逻辑
    def handle_get_attr(self, node):
        pass
    # 返回与给定索引相关联的用户节点，用于获取操作
    def _output_node_at_index(self, node, index):
        user_node = None
        # 遍历节点的所有用户
        for user in node.users:
            # 断言用户操作为获取操作
            assert user.target is operator.getitem, f"{user} is not a getitem node"
            # 检查索引是否与用户操作的第二个参数相匹配
            if index == user.args[1]:
                # 如果尚未设置用户节点，则设置为当前用户
                if user_node is None:
                    user_node = user
                else:
                    # 如果已经设置了用户节点，则将重复的获取项节点记录到字典中
                    self.duplicate_getitem_nodes[user.name] = user_node.name
        # 返回与索引相关的用户节点
        return user_node

    # 序列化节点的元数据为字典
    def serialize_metadata(self, node: torch.fx.Node) -> Dict[str, str]:
        ret = {}
        # 如果节点具有堆栈跟踪信息，则添加到返回字典中
        if stack_trace := node.meta.get("stack_trace"):
            ret["stack_trace"] = stack_trace

        # 如果节点具有神经网络模块堆栈信息，则序列化为特定格式并添加到返回字典中
        if nn_module_stack := node.meta.get("nn_module_stack"):

            def export_nn_module_stack(val):
                assert isinstance(val, tuple) and len(val) == 2
                path, ty = val

                assert isinstance(path, str)
                assert isinstance(ty, str)

                return path + "," + ty

            # 将神经网络模块堆栈的每一项序列化为 "key,orig_path,type_str" 格式
            nn_module_list = [
                f"{k},{export_nn_module_stack(v)}" for k, v in nn_module_stack.items()
            ]
            ret["nn_module_stack"] = ST_DELIMITER.join(nn_module_list)

        # 如果节点具有源函数堆栈信息，则序列化为特定格式并添加到返回字典中
        if source_fn_st := node.meta.get("source_fn_stack"):
            source_fn_list = [
                f"{source_fn[0]},{self.serialize_operator(source_fn[1])}"
                for source_fn in source_fn_st
            ]
            ret["source_fn_stack"] = ST_DELIMITER.join(source_fn_list)

        # 如果节点具有 Torch 函数信息，则添加到返回字典中
        if torch_fn := node.meta.get("torch_fn"):
            ret["torch_fn"] = ST_DELIMITER.join(list(torch_fn))

        # 返回序列化后的元数据字典
        return ret

    # 序列化脚本对象的元数据
    def serialize_script_obj_meta(
        self, script_obj_meta: ep.CustomObjArgument
    ) -> CustomObjArgument:
        # 直接返回脚本对象的元数据
        return CustomObjArgument(
            name=script_obj_meta.name,
            class_fqn=script_obj_meta.class_fqn,
        )

    # 序列化符号操作的输入参数
    def serialize_sym_op_inputs(self, op, args) -> List[NamedArgument]:
        serialized_args = []
        # 获取操作的参数名称列表
        args_names = inspect.signature(op).parameters.keys()
        # 遍历参数名称和实际参数，序列化每个参数并添加到列表中
        for args_name, arg in zip(args_names, args):
            serialized_args.append(
                NamedArgument(name=args_name, arg=self.serialize_input(arg))
            )
        # 返回序列化后的参数列表
        return serialized_args

    # 序列化输入参数
    def serialize_inputs(
        self,
        target: Any,  # torch._ops.OpOverload and other custom operator types.
        args,
        kwargs=None
    ):
    ) -> List[NamedArgument]:
        # 断言目标类型为 OpOverload 或者允许的注册操作类型之一
        assert isinstance(target, (torch._ops.OpOverload, *allowed_registered_op_types()))
        # 如果 kwargs 为 None，则置为空字典
        kwargs = kwargs or {}
        # 初始化序列化后的参数列表
        serialized_args = []

        # 从目标对象获取其对应的模式（schema）
        schema = _get_schema_from_target(target)

        # 遍历模式中的每一个参数
        for i, schema_arg in enumerate(schema.arguments):
            # 如果参数在 kwargs 中存在，则序列化该参数的值
            if schema_arg.name in kwargs:
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(kwargs[schema_arg.name], schema_arg.type),
                    )
                )
            # 如果参数不是仅限关键字参数且在 args 中有对应位置的参数，则序列化该参数的值
            elif not schema_arg.kwarg_only and i < len(args):
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(args[i], schema_arg.type),
                    )
                )
            else:
                # 否则，不序列化缺失的参数（带默认值的情况）
                pass

        # 返回序列化后的参数列表
        return serialized_args

    def serialize_hoo_inputs(self, args, kwargs) -> List[NamedArgument]:
        """
        For serializing HOO inputs since HOOs do not have a schema.
        """
        # 序列化 HOO 输入参数，因为 HOO 没有模式（schema）
        inputs = [
            NamedArgument(
                name="",
                arg=self.serialize_input(a),
            )
            for a in args
        ]
        # 同时序列化关键字参数
        inputs.extend(
            [
                NamedArgument(name=name, arg=self.serialize_input(a))
                for name, a in kwargs.items()
            ]
        )
        return inputs

    def is_sym_int_arg(self, arg) -> bool:
        # 判断参数是否为整数或者是 FX 图中的节点，并且节点名存在于符号整数值列表中
        return isinstance(arg, int) or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_int_values
        )

    def is_sym_bool_arg(self, arg) -> bool:
        # 判断参数是否为布尔类型或者是 FX 图中的节点，并且节点名存在于符号布尔值列表中
        return isinstance(arg, bool) or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_bool_values
        )

    def serialize_input(
        self, arg, arg_type: Optional[torch._C.Argument] = None
    ):
        # 序列化输入参数
        pass

    def serialize_tensor_output(self, name, meta_val) -> TensorArgument:
        # 断言张量名称不在图状态的张量值列表中
        assert name not in self.graph_state.tensor_values
        # 将张量值及其元信息序列化，并存储在图状态的张量值列表中
        self.graph_state.tensor_values[name] = serialize_tensor_meta(meta_val)
        # 返回张量参数对象
        return TensorArgument(name=name)

    def serialize_sym_int_output(self, name, meta_val) -> SymIntArgument:
        # 断言符号整数名称不在图状态的符号整数值列表中
        assert name not in self.graph_state.sym_int_values
        # 将符号整数值及其元信息序列化，并存储在图状态的符号整数值列表中
        self.graph_state.sym_int_values[name] = serialize_sym_int(meta_val)
        # 返回符号整数参数对象
        return SymIntArgument.create(as_name=name)

    def serialize_sym_bool_output(self, name, meta_val) -> SymIntArgument:
        # 断言符号布尔名称不在图状态的符号布尔值列表中
        assert name not in self.graph_state.sym_bool_values
        # 将符号布尔值及其元信息序列化，并存储在图状态的符号布尔值列表中
        self.graph_state.sym_bool_values[name] = serialize_sym_bool(meta_val)
        # 返回符号布尔参数对象
        return SymBoolArgument.create(as_name=name)
    # 将输出规范对象序列化为输出规范对象
    def serialize_output_spec(self, spec: ep.OutputSpec) -> OutputSpec:
        # 如果输出规范的类型为用户输出
        if spec.kind == ep.OutputKind.USER_OUTPUT:
            # 创建用户输出规范对象，序列化其参数规范
            return OutputSpec.create(
                user_output=UserOutputSpec(arg=self.serialize_argument_spec(spec.arg))
            )
        # 如果输出规范的类型为损失输出
        elif spec.kind == ep.OutputKind.LOSS_OUTPUT:
            # 断言参数规范为张量参数
            assert isinstance(spec.arg, ep.TensorArgument)
            # 创建损失输出规范对象，指定其参数名为特定的张量参数名
            return OutputSpec.create(
                loss_output=LossOutputSpec(arg=TensorArgument(name=spec.arg.name))
            )
        # 如果输出规范的类型为缓冲区变异
        elif spec.kind == ep.OutputKind.BUFFER_MUTATION:
            # 断言目标名称不为空
            assert spec.target is not None
            # 断言参数规范为张量参数
            assert isinstance(spec.arg, ep.TensorArgument)
            # 创建缓冲区变异规范对象，指定其参数名和缓冲区名称
            return OutputSpec.create(
                buffer_mutation=BufferMutationSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    buffer_name=spec.target,
                )
            )
        # 如果输出规范的类型为梯度到参数
        elif spec.kind == ep.OutputKind.GRADIENT_TO_PARAMETER:
            # 断言目标名称不为空
            assert spec.target is not None
            # 断言参数规范为张量参数
            assert isinstance(spec.arg, ep.TensorArgument)
            # 创建梯度到参数规范对象，指定其参数名和参数名称
            return OutputSpec.create(
                gradient_to_parameter=GradientToParameterSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    parameter_name=spec.target,
                )
            )
        # 如果输出规范的类型为梯度到用户输入
        elif spec.kind == ep.OutputKind.GRADIENT_TO_USER_INPUT:
            # 断言目标名称不为空
            assert spec.target is not None
            # 断言参数规范为张量参数
            assert isinstance(spec.arg, ep.TensorArgument)
            # 创建梯度到用户输入规范对象，指定其参数名和用户输入名称
            return OutputSpec.create(
                gradient_to_user_input=GradientToUserInputSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    user_input_name=spec.target,
                )
            )
        # 如果输出规范的类型为用户输入变异
        elif spec.kind == ep.OutputKind.USER_INPUT_MUTATION:
            # 断言目标名称不为空
            assert spec.target is not None
            # 断言参数规范为张量参数
            assert isinstance(spec.arg, ep.TensorArgument)
            # 创建用户输入变异规范对象，指定其参数名和用户输入名称
            return OutputSpec.create(
                user_input_mutation=UserInputMutationSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    user_input_name=spec.target,
                )
            )
        # 如果输出规范的类型为令牌
        elif spec.kind == ep.OutputKind.TOKEN:
            # 断言参数规范为令牌参数
            assert isinstance(spec.arg, ep.TokenArgument)
            # 创建令牌输出规范对象，指定其参数名为特定的令牌参数名
            return OutputSpec.create(
                token=OutputTokenSpec(
                    arg=TokenArgument(name=spec.arg.name),
                )
            )
        else:
            # 如果出现未知的参数类型，抛出断言错误
            raise AssertionError(f"Unknown argument kind: {spec}")

    # 将导出图签名对象序列化为图签名对象
    def serialize_signature(self, sig: ep.ExportGraphSignature) -> GraphSignature:
        # 创建图签名对象，包括序列化输入规范和输出规范列表
        return GraphSignature(
            input_specs=[self.serialize_input_spec(s) for s in sig.input_specs],
            output_specs=[self.serialize_output_spec(s) for s in sig.output_specs],
        )
    # 将输入的参数规范对象序列化为通用的 Argument 对象
    def serialize_argument_spec(self, x: ep.ArgumentSpec) -> Argument:
        # 如果参数是 TensorArgument 类型，则创建对应的 Argument 对象
        if isinstance(x, ep.TensorArgument):
            return Argument.create(as_tensor=TensorArgument(name=x.name))
        # 如果参数是 SymIntArgument 类型，则创建对应的 Argument 对象
        elif isinstance(x, ep.SymIntArgument):
            return Argument.create(as_sym_int=SymIntArgument.create(as_name=x.name))
        # 如果参数是 ConstantArgument 类型，则序列化其值
        elif isinstance(x, ep.ConstantArgument):
            return self.serialize_input(x.value)
        # 如果参数是 CustomObjArgument 类型，则创建对应的 Argument 对象
        elif isinstance(x, ep.CustomObjArgument):
            return Argument.create(
                as_custom_obj=CustomObjArgument(name=x.name, class_fqn=x.class_fqn)
            )
        else:
            # 如果遇到未知类型的参数，则抛出断言错误
            raise AssertionError("TODO")

    # 序列化模块调用签名对象
    def serialize_module_call_signature(
        self, module_call_signature: ep.ModuleCallSignature
    ) -> ModuleCallSignature:
        # 返回序列化后的 ModuleCallSignature 对象，包括输入输出参数列表及规范
        return ModuleCallSignature(
            inputs=[
                self.serialize_argument_spec(x) for x in module_call_signature.inputs
            ],
            outputs=[
                self.serialize_argument_spec(x) for x in module_call_signature.outputs
            ],
            in_spec=treespec_dumps(module_call_signature.in_spec, TREESPEC_VERSION),
            out_spec=treespec_dumps(module_call_signature.out_spec, TREESPEC_VERSION),
        )

    # 序列化模块调用图
    def serialize_module_call_graph(
        self, module_call_graph: List[ep.ModuleCallEntry]
    ) -> List[ModuleCallEntry]:
        # 返回序列化后的模块调用图，包括每个模块调用的 FQN 和对应的签名
        return [
            ModuleCallEntry(
                fqn=entry.fqn,
                signature=(
                    self.serialize_module_call_signature(entry.signature)
                    if entry.signature
                    else None
                ),
            )
            for entry in module_call_graph
        ]
    def serialize_hoo_outputs(self, node: torch.fx.Node) -> List[Argument]:
        """
        For serializing HOO outputs since HOOs do not have a schema.
        """
        # 从节点的元数据中获取值对象
        meta_val = node.meta["val"]

        # 检查值对象是否为元组
        if isinstance(meta_val, tuple):
            # 注意：由于没有具体的模式(schema)，我们将所有元组输出序列化为值的列表。
            # 即使输出应为张量列表（Tensor[]），我们也将其序列化为张量的列表（Tensor, Tensor, Tensor）。
            # 一个例外是，如果有单一张量，则将其序列化为单一张量列表，以便反序列化器知道要插入getitem节点。

            # 如果元组中只有一个元素
            if len(meta_val) == 1:
                assert isinstance(meta_val[0], torch.Tensor)
                # 获取对应的用户节点（即输出节点）
                user_node = self._output_node_at_index(node, 0)
                # 获取节点名称或创建未使用节点名称
                name = (
                    user_node.name
                    if user_node is not None
                    else f"{node.name}_unused_0"
                )
                # 序列化单个张量输出，并创建参数对象列表
                return [Argument.create(as_tensors=[self.serialize_tensor_output(name, meta_val[0])])]

            # 处理元组中每个元素的情况
            outputs = []
            for i, element_meta_val in enumerate(meta_val):
                # 获取对应的用户节点（即输出节点）
                user_node = self._output_node_at_index(node, i)
                # 如果元素是列表（例如 "-> Tensor[]"）
                if isinstance(element_meta_val, list):
                    assert user_node is not None
                    # 序列化列表输出中的每个张量
                    tensors = []
                    for j, m in enumerate(element_meta_val):
                        if not isinstance(m, torch.Tensor):
                            raise SerializeError(f"Serialize list output with type {type(m)} nyi")
                        # 获取子节点的名称或创建未使用节点名称
                        sub_user_node = self._output_node_at_index(user_node, j)
                        name = (
                            sub_user_node.name
                            if sub_user_node is not None
                            else f"{user_node.name}_unused_{j}"
                        )
                        # 序列化张量输出，并添加到张量列表中
                        tensors.append(self.serialize_tensor_output(name, m))
                    # 创建参数对象，并添加到输出列表中
                    outputs.append(Argument.create(as_tensors=tensors))

                else:
                    # 获取节点名称或创建未使用节点名称
                    name = (
                        user_node.name
                        if user_node is not None
                        else f"{node.name}_unused_{i}"
                    )
                    # 序列化元素值，并添加到输出列表中
                    outputs.append(self.serialize_output(name, element_meta_val))

            return outputs
        else:
            # 如果值对象不是元组，则直接序列化单个输出，并创建参数对象列表
            return [self.serialize_output(node.name, meta_val)]
    def serialize_output(self, name: str, meta_val: Any) -> Argument:
        # 检查返回值是否为单一值
        if meta_val is None:
            return Argument.create(as_none=())
        if isinstance(meta_val, torch.Tensor):
            # 如果返回值是 torch.Tensor 类型
            return Argument.create(
                as_tensor=self.serialize_tensor_output(name, meta_val)
            )
        elif isinstance(meta_val, (int, torch.SymInt)):
            # 如果返回值是 int 或 torch.SymInt 类型
            return Argument.create(
                as_sym_int=self.serialize_sym_int_output(name, meta_val)
            )
        elif isinstance(meta_val, torch.SymBool):
            # 如果返回值是 torch.SymBool 类型
            return Argument.create(
                as_sym_bool=self.serialize_sym_bool_output(name, meta_val)
            )

        # 如果返回值是列表，应该在前面处理
        raise SerializeError(f"Unable to serialize output {meta_val}")

    def _handle_getitem_users(self, node: torch.fx.Node) -> List[TensorArgument]:
        meta_val = node.meta["val"]

        idx_to_name = {}
        for user in node.users:
            assert (
                user.target is operator.getitem
            ), f"User node {user} of {node} is incorrect"
            # 将使用了 getitem 操作的节点映射为索引到名称的字典
            idx_to_name[user.args[1]] = user.name

        for idx, _ in enumerate(meta_val):
            # 对于 FX 没有使用的输出，不会生成 getitem 节点。
            # 但是我们需要一个名称来确保输出数量与模式匹配。因此分配一个虚拟名称。
            if idx not in idx_to_name:
                idx_to_name[idx] = f"{node.name}_unused_{idx}"

        arg_list = []
        for i, element_meta_val in enumerate(meta_val):
            # 将每个元素的元数据值序列化为 TensorArgument，并添加到列表中
            arg_list.append(
                self.serialize_tensor_output(idx_to_name[i], element_meta_val)
            )

        return arg_list

    def serialize_graph(self, graph_module: torch.fx.GraphModule) -> Graph:
        assert isinstance(graph_module, torch.fx.GraphModule)
        for node in graph_module.graph.nodes:
            try:
                # 调用相应节点操作的处理方法
                getattr(self, f"handle_{node.op}")(node)
            except Exception as e:
                # 如果处理过程中出错，抛出序列化错误
                raise SerializeError(
                    f"Failed serializing node {node} in graph: {node.format_node()}"
                ) from e

        # 返回序列化后的图形结构
        return Graph(
            inputs=self.graph_state.inputs,
            nodes=self.graph_state.nodes,
            tensor_values=self.graph_state.tensor_values,
            sym_int_values=self.graph_state.sym_int_values,
            sym_bool_values=self.graph_state.sym_bool_values,
            custom_obj_values=self.graph_state.custom_obj_values,
            outputs=self.graph_state.outputs,
            is_single_tensor_return=self.graph_state.is_single_tensor_return,
        )
    # 定义一个方法 serialize，用于序列化给定的 torch.fx.GraphModule
    def serialize(self, graph_module: torch.fx.GraphModule) -> GraphModule:
        # 调用 self.serialize_graph 方法，将传入的 graph_module 序列化为图形表示
        graph = self.serialize_graph(graph_module)

        # 创建一个新的 GraphModule 对象，包括序列化后的图形表示、签名的序列化版本和模块调用图的序列化版本
        return GraphModule(
            graph=graph,
            signature=self.serialize_signature(self.graph_signature),
            module_call_graph=self.serialize_module_call_graph(self.module_call_graph),
        )
@final
class ExportedProgramSerializer(metaclass=Final):
    # 定义了一个不可继承的类 ExportedProgramSerializer，使用 Final 元类进行声明

    def __init__(self, opset_version: Optional[Dict[str, int]] = None):
        # 初始化方法，接受一个可选的 opset_version 参数作为字典
        self.opset_version: Dict[str, int] = {}
        if opset_version:
            self.opset_version.update(opset_version)
        # 如果提供了 opset_version 参数，则更新实例的 opset_version 属性

        if "aten" not in self.opset_version:
            self.opset_version["aten"] = torch._C._get_max_operator_version()
        # 如果 "aten" 不在 opset_version 中，则将其添加，并获取当前 torch 操作符的最大版本号

    def serialize(self, exported_program: ep.ExportedProgram) -> _SerializedProgram:
        """
        Args:
            exported_program: Exported Program to serialize
        """
        # 序列化方法，将 ExportedProgram 对象转换为 _SerializedProgram 对象

        exported_program._validate()
        # 调用 ExportedProgram 对象的 _validate 方法，确保其有效性

        gm_serializer = GraphModuleSerializer(
            exported_program.graph_signature, exported_program.module_call_graph
        )
        # 创建 GraphModuleSerializer 对象 gm_serializer，传入 ExportedProgram 的图签名和模块调用图

        serialized_graph_module = gm_serializer.serialize(exported_program.graph_module)
        # 调用 gm_serializer 的 serialize 方法，对图模块进行序列化

        serialized_range_constraints = serialize_range_constraints(
            exported_program.range_constraints
        )
        # 调用 serialize_range_constraints 函数，序列化 range_constraints

        # TODO: Directly serialize exported_program.constants once
        # CustomClassHolders get stored in the ExportedProgram rather than in
        # the graph
        constants = {}
        for n, c in gm_serializer.custom_objs.items():
            constants[n] = c
        # 遍历 gm_serializer 的 custom_objs 属性，将其作为 constants 字典的一部分

        for n, t in exported_program.constants.items():
            assert n not in constants
            constants[n] = t
        # 遍历 exported_program 的 constants 属性，确保没有重复的键，并将其添加到 constants 字典中

        serialized_ep = ExportedProgram(
            graph_module=serialized_graph_module,
            opset_version=self.opset_version,
            range_constraints=serialized_range_constraints,
            schema_version=SchemaVersion(
                major=SCHEMA_VERSION[0],
                minor=SCHEMA_VERSION[1],
            ),
            dialect=exported_program.dialect
        )
        # 创建 ExportedProgram 对象 serialized_ep，传入序列化后的图模块、opset_version、range_constraints、schema_version 和 dialect

        # Test canonical form is well defined.
        canonicalize(serialized_ep)
        # 调用 canonicalize 函数，确保 serialized_ep 的规范形式是明确定义的

        return _SerializedProgram(
            serialized_ep,
            serialize_torch_artifact(exported_program.state_dict),
            serialize_torch_artifact(constants),
            serialize_torch_artifact(exported_program.example_inputs),
        )
        # 返回 _SerializedProgram 对象，包含 serialized_ep 和序列化的 state_dict、constants 和 example_inputs


@final
class GraphModuleDeserializer(metaclass=Final):
    # 定义了一个不可继承的类 GraphModuleDeserializer，使用 Final 元类进行声明

    @dataclasses.dataclass
    class Result:
        graph_module: torch.fx.GraphModule
        signature: ep.ExportGraphSignature
        module_call_graph: List[ep.ModuleCallEntry]
        names_to_symbols: Dict[str, sympy.Symbol]
        state_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]]
        constants: Dict[str, Union[torch.Tensor, FakeScriptObject, torch.ScriptObject]]
        example_inputs: Optional[Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]]
        # 内部 Result 类，用于定义 GraphModuleDeserializer 的返回结果的数据结构

    def __init__(self):
        # 初始化方法，创建 GraphModuleDeserializer 实例时执行
        self.serialized_name_to_node: Dict[str, torch.fx.Node] = {}
        self.serialized_name_to_meta: Dict[str, MetaType] = {}
        # 初始化两个属性 serialized_name_to_node 和 serialized_name_to_meta，分别为字典类型

        self.graph = torch.fx.Graph()
        self.module = torch.nn.Module()
        # 初始化属性 graph 和 module，分别为 torch.fx.Graph 和 torch.nn.Module 的实例
    # 定义一个生成器方法，用于保存当前图、模块和序列化映射，并重置它们为新的对象
    def save_graph_module(self) -> Iterator[None]:
        # 保存当前的图、模块和序列化映射到 saved 元组中
        saved = (
            self.graph,
            self.module,
            self.serialized_name_to_node,
            self.serialized_name_to_meta,
        )
        # 创建一个新的空图对象
        self.graph = torch.fx.Graph()
        # 创建一个新的空模块对象
        self.module = torch.nn.Module()
        # 清空已序列化名称到节点的映射
        self.serialized_name_to_node = {}
        # 清空已序列化名称到元数据的映射
        self.serialized_name_to_meta = {}
        try:
            yield  # 返回生成器
        finally:
            # 在 finally 块中恢复保存的图、模块和序列化映射
            (
                self.graph,
                self.module,
                self.serialized_name_to_node,
                self.serialized_name_to_meta,
            ) = saved

    # 根据序列化目标字符串反序列化操作符
    def deserialize_operator(self, serialized_target: str):
        if serialized_target.startswith("_operator"):  # 如果以 "_operator" 开头
            module = operator  # 设置 module 为 operator 模块
            serialized_target_names = serialized_target.split(".")[1:]  # 提取后续的名称列表
        elif serialized_target.startswith("torch"):  # 如果以 "torch" 开头
            module = torch  # 设置 module 为 torch 模块
            serialized_target_names = serialized_target.split(".")[1:]  # 提取后续的名称列表
        else:  # 对于其他情况
            return serialized_target  # 直接返回序列化目标字符串

        target = module
        # 遍历序列化目标名称列表中的每个名称
        for name in serialized_target_names:
            if not hasattr(target, name):  # 如果目标对象中没有该属性名称
                return serialized_target  # 直接返回序列化目标字符串
            else:
                target = getattr(target, name)  # 获取目标对象的属性
        return target  # 返回最终的目标对象或属性

    # 根据 SymBool 对象反序列化成布尔值或 SymBool 对象
    def deserialize_sym_bool(self, s: SymBool) -> Union[bool, torch.SymBool]:
        val = s.value  # 获取 SymBool 对象的值
        if s.type == "as_expr":  # 如果 SymBool 类型为 "as_expr"
            # 使用 sympy 库根据表达式字符串创建表达式，并在符号名称到符号的本地环境中进行解析
            expr = sympy.sympify(val.expr_str, locals=self.symbol_name_to_symbol)
            return self.shape_env.create_symboolnode(expr)  # 创建符号布尔节点
        elif s.type == "as_bool":  # 如果 SymBool 类型为 "as_bool"
            assert isinstance(val, bool)  # 断言值是布尔类型
            return val  # 直接返回布尔值
        else:
            # 抛出序列化错误，指出 SymBool 对象具有无效的字段类型和值
            raise SerializeError(
                f"SymBool has invalid field type {s.type} with value {s.value}"
            )

    # 根据 TensorMeta 对象反序列化成 FakeTensor 对象
    def deserialize_tensor_meta(
        self,
        tensor_meta: TensorMeta,
    ) -> FakeTensor:
        with self.fake_tensor_mode:  # 使用 fake_tensor_mode 上下文管理器
            # 创建一个带有给定尺寸和步幅的空张量，使用反序列化的符号整数进行设置
            return cast(
                FakeTensor,
                torch.empty_strided(
                    tuple(self.deserialize_sym_int(val) for val in tensor_meta.sizes),  # 创建尺寸元组
                    tuple(self.deserialize_sym_int(val) for val in tensor_meta.strides),  # 创建步幅元组
                    device=deserialize_device(tensor_meta.device),  # 设备类型的反序列化
                    dtype=_SERIALIZE_TO_TORCH_DTYPE[tensor_meta.dtype],  # 数据类型的反序列化
                ),
            )

    # 根据 CustomObjArgument 对象反序列化成 ep.CustomObjArgument 对象
    def deserialize_script_obj_meta(
        self, script_obj_meta: CustomObjArgument
    ) -> ep.CustomObjArgument:
        # 创建并返回 ep.CustomObjArgument 对象，使用 script_obj_meta 的名称和类全限定名
        return ep.CustomObjArgument(
            name=script_obj_meta.name,
            class_fqn=script_obj_meta.class_fqn,
        )
    # 反序列化图输出，将序列化的输出对象转换为图节点或整数值
    def deserialize_graph_output(self, output) -> Optional[Union[torch.fx.Node, int]]:
        # 如果输出类型为 "as_tensor"，返回相应序列化名称对应的图节点
        if output.type == "as_tensor":
            return self.serialized_name_to_node[output.as_tensor.name]
        # 如果输出类型为 "as_sym_int"，返回相应序列化名称对应的图节点
        elif output.type == "as_sym_int":
            return self.serialized_name_to_node[output.as_sym_int.as_name]
        # 如果输出类型为 "as_sym_bool"，返回相应序列化名称对应的图节点
        elif output.type == "as_sym_bool":
            return self.serialized_name_to_node[output.as_sym_bool.as_name]
        # 如果输出类型为 "as_int"，直接返回整数值
        elif output.type == "as_int":
            return output.as_int
        # 如果输出类型为 "as_none"，返回 None
        elif output.type == "as_none":
            return None
        # 如果未知输出类型，则引发序列化错误，报告无法反序列化该输出节点
        else:
            raise SerializeError(f"Unable to deserialize output node {output}")
    # 反序列化给定的节点对象，将其映射为图中的函数调用节点
    def deserialize_node(self, serialized_node: Node, target: Callable) -> None:
        # 如果目标是符号布尔运算或符号整数运算之一
        if target in _SYM_BOOL_OPS or target in _SYM_INT_OPS:
            # 从序列化节点的输出中获取名称作为节点的名称
            name = serialized_node.outputs[0].value.as_name
            # 反序列化符号操作的输入参数
            args = self.deserialize_sym_op_inputs(serialized_node.inputs)

            # 在图中创建一个函数调用节点，表示符号操作
            fx_node = self.graph.create_node("call_function", target, args, {}, name)
            # 处理符号操作节点的输出
            self.deserialize_sym_op_outputs(serialized_node, fx_node)

        # 如果目标是 torch 的 HigherOrderOperator 类型
        elif isinstance(target, torch._ops.HigherOrderOperator):
            # 反序列化高阶操作的输入参数和关键字参数
            args, kwargs = self.deserialize_hoo_inputs(serialized_node.inputs)

            # 如果 HOP 返回单个张量，则使用该张量的名称作为新节点的名称
            name = (
                serialized_node.outputs[0].as_tensor.name
                if len(serialized_node.outputs) == 1
                and hasattr(serialized_node.outputs[0], "as_tensor")
                else None
            )

            # 在图中创建一个函数调用节点，表示高阶操作
            fx_node = self.graph.create_node(
                "call_function", target, args, kwargs, name
            )
            # 处理高阶操作节点的输出
            self.deserialize_outputs(serialized_node, fx_node)
            # 更新节点的元数据
            fx_node.meta.update(self.deserialize_metadata(serialized_node.metadata))

        # 如果目标是 torch 的 OpOverload 类型
        elif isinstance(target, torch._ops.OpOverload):
            # 如果此节点返回单个张量，则使用该张量的名称作为新节点的名称
            name = (
                serialized_node.outputs[0].as_tensor.name
                if _is_single_tensor_return(target)
                else None  # FX 将为我们生成一个名称
            )

            # 反序列化 OpOverload 类型的输入参数和关键字参数
            args, kwargs = self.deserialize_inputs(target, serialized_node)

            # 在图中创建一个函数调用节点，表示重载操作
            fx_node = self.graph.create_node(
                "call_function", target, args, kwargs, name
            )
            # 处理重载操作节点的输出
            self.deserialize_outputs(serialized_node, fx_node)

        else:
            # 如果目标类型不受支持，则抛出异常
            raise SerializeError(
                f"Unsupported target type for node {serialized_node}: {type(target)}"
            )

        # 更新节点的元数据
        fx_node.meta.update(self.deserialize_metadata(serialized_node.metadata))

        # 如果节点的操作不是占位符或输出，并且节点的元数据中不存在 'nn_module_stack' 键
        if fx_node.op not in ["placeholder", "output"] and "nn_module_stack" not in fx_node.meta:
            # 在节点的元数据中添加一个空字典，用于表示神经网络模块栈
            fx_node.meta["nn_module_stack"] = {}  # serialization throws away empty dicts
    def deserialize_input_spec(self, i: InputSpec) -> ep.InputSpec:
        # 如果输入类型为用户输入，则创建一个用户输入规范对象，没有目标
        if i.type == "user_input":
            return ep.InputSpec(
                kind=ep.InputKind.USER_INPUT,
                arg=self.deserialize_argument_spec(i.user_input.arg),
                target=None,
            )
        # 如果输入类型为参数，则创建一个参数输入规范对象，设置目标为参数名
        elif i.type == "parameter":
            return ep.InputSpec(
                kind=ep.InputKind.PARAMETER,
                arg=ep.TensorArgument(name=i.parameter.arg.name),
                target=i.parameter.parameter_name,
            )
        # 如果输入类型为缓冲区，则创建一个缓冲区输入规范对象，设置目标为缓冲区名，并指定是否持久化
        elif i.type == "buffer":
            return ep.InputSpec(
                kind=ep.InputKind.BUFFER,
                arg=ep.TensorArgument(name=i.buffer.arg.name),
                target=i.buffer.buffer_name,
                persistent=i.buffer.persistent,
            )
        # 如果输入类型为常量张量，则创建一个常量张量输入规范对象，设置目标为张量常量名
        elif i.type == "tensor_constant":
            return ep.InputSpec(
                kind=ep.InputKind.CONSTANT_TENSOR,
                arg=ep.TensorArgument(name=i.tensor_constant.arg.name),
                target=i.tensor_constant.tensor_constant_name,
            )
        # 如果输入类型为自定义对象，则创建一个自定义对象输入规范对象，设置目标为自定义对象名
        elif i.type == "custom_obj":
            return ep.InputSpec(
                kind=ep.InputKind.CUSTOM_OBJ,
                arg=ep.CustomObjArgument(
                    name=i.custom_obj.arg.name, class_fqn=i.custom_obj.arg.class_fqn
                ),
                target=i.custom_obj.custom_obj_name,
            )
        # 如果输入类型为令牌，则创建一个令牌输入规范对象，没有目标
        elif i.type == "token":
            return ep.InputSpec(
                kind=ep.InputKind.TOKEN,
                arg=ep.TokenArgument(name=i.token.arg.name),
                target=None
            )
        # 如果输入类型为常量输入，则创建一个用户输入规范对象，但其实际类型是常量输入，没有目标
        elif i.type == "constant_input":
            return ep.InputSpec(
                kind=ep.InputKind.USER_INPUT,
                arg=ep.ConstantArgument(
                    name=i.constant_input.name,
                    value=self.deserialize_constant_input(i.constant_input.value)
                ),
                target=None,
            )
        else:
            # 如果出现未知的输入类型，则抛出断言错误
            raise AssertionError(f"Unknown input spec {i}")
    # 反序列化输出规范对象，将其转换为对应的工程化推理输出规范
    def deserialize_output_spec(self, o: OutputSpec) -> ep.OutputSpec:
        # 如果输出类型为用户输出
        if o.type == "user_output":
            # 返回用户输出的规范对象
            return ep.OutputSpec(
                kind=ep.OutputKind.USER_OUTPUT,
                arg=self.deserialize_argument_spec(o.user_output.arg),
                target=None,
            )
        # 如果输出类型为损失输出
        elif o.type == "loss_output":
            # 返回损失输出的规范对象
            return ep.OutputSpec(
                kind=ep.OutputKind.LOSS_OUTPUT,
                arg=ep.TensorArgument(name=o.loss_output.arg.name),
                target=None,
            )
        # 如果输出类型为缓冲区变异
        elif o.type == "buffer_mutation":
            # 返回缓冲区变异的规范对象
            return ep.OutputSpec(
                kind=ep.OutputKind.BUFFER_MUTATION,
                arg=ep.TensorArgument(name=o.buffer_mutation.arg.name),
                target=o.buffer_mutation.buffer_name,
            )
        # 如果输出类型为梯度到参数
        elif o.type == "gradient_to_parameter":
            # 返回梯度到参数的规范对象
            return ep.OutputSpec(
                kind=ep.OutputKind.GRADIENT_TO_PARAMETER,
                arg=ep.TensorArgument(name=o.gradient_to_parameter.arg.name),
                target=o.gradient_to_parameter.parameter_name,
            )
        # 如果输出类型为梯度到用户输入
        elif o.type == "gradient_to_user_input":
            # 返回梯度到用户输入的规范对象
            return ep.OutputSpec(
                kind=ep.OutputKind.GRADIENT_TO_USER_INPUT,
                arg=ep.TensorArgument(name=o.gradient_to_user_input.arg.name),
                target=o.gradient_to_user_input.user_input_name,
            )
        # 如果输出类型为用户输入变异
        elif o.type == "user_input_mutation":
            # 返回用户输入变异的规范对象
            return ep.OutputSpec(
                kind=ep.OutputKind.USER_INPUT_MUTATION,
                arg=ep.TensorArgument(name=o.user_input_mutation.arg.name),
                target=o.user_input_mutation.user_input_name,
            )
        # 如果输出类型为令牌
        elif o.type == "token":
            # 返回令牌的规范对象
            return ep.OutputSpec(
                kind=ep.OutputKind.TOKEN,
                arg=ep.TokenArgument(name=o.token.arg.name),
                target=None
            )
        else:
            # 如果未知的输出类型，则抛出断言错误
            raise AssertionError(f"Unknown output spec {o}")

    # 反序列化图形签名对象，将其转换为对应的工程化推理导出图形签名
    def deserialize_signature(self, sig: GraphSignature) -> ep.ExportGraphSignature:
        # 返回导出图形签名对象
        return ep.ExportGraphSignature(
            # 反序列化输入规范列表
            input_specs=[self.deserialize_input_spec(i) for i in sig.input_specs],
            # 反序列化输出规范列表
            output_specs=[self.deserialize_output_spec(o) for o in sig.output_specs],
        )

    # 反序列化方法，将序列化的图形模块、状态字典、常量、示例输入和符号名范围转换为可用的工程化推理模块
    def deserialize(
        self,
        serialized_graph_module: GraphModule,
        serialized_state_dict: Union[Dict[str, torch.Tensor], bytes],
        constants: Union[Dict[str, Any], bytes],
        example_inputs: Optional[Union[Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]], bytes]] = None,
        symbol_name_to_range: Optional[Dict[str, symbolic_shapes.ValueRanges]] = None,
        ):
    ) -> Result:
        global _CURRENT_DESERIALIZER
        # 确保当前没有其他反序列化器在操作
        assert _CURRENT_DESERIALIZER is None
        # 将当前的反序列化器设置为自身
        _CURRENT_DESERIALIZER = self
        try:
            # 创建符号形状环境，假设默认为静态
            self.shape_env = symbolic_shapes.ShapeEnv(assume_static_by_default=True)
            # 设置假张量模式，不允许回退内核，允许非假输入，使用上述形状环境
            self.fake_tensor_mode = FakeTensorMode(
                allow_fallback_kernels=False,
                allow_non_fake_inputs=True,
                shape_env=self.shape_env,
            )
            # 符号名称到符号的映射字典
            self.symbol_name_to_symbol: Dict[str, sympy.Symbol] = {}
            # 反序列化常量
            self.constants = deserialize_torch_artifact(constants)
            # 反序列化签名
            self.signature = self.deserialize_signature(serialized_graph_module.signature)

            # 反序列化会进行与 0/1 相关的分析，因此创建假的范围约束，然后恢复原始的范围约束
            self.symbol_name_to_range = {}
            if symbol_name_to_range:
                for k, vr in symbol_name_to_range.items():
                    lower = vr.lower
                    if vr.upper >= 2:  # 最大值大于等于 2，不是符号布尔范围
                        lower = max(2, lower)
                    self.symbol_name_to_range[k] = symbolic_shapes.ValueRanges(_int_to_sympy_int(lower), vr.upper)

            # 如果存在示例输入且不为空，则反序列化示例输入
            if example_inputs is not None and len(example_inputs) > 0:
                self.example_inputs = deserialize_torch_artifact(example_inputs)
            else:
                self.example_inputs = None
            # 反序列化图形
            self.deserialize_graph(serialized_graph_module.graph)

            # 反序列化模块调用图
            module_call_graph = self.deserialize_module_call_graph(
                serialized_graph_module.module_call_graph
            )
            # 返回反序列化结果对象
            return GraphModuleDeserializer.Result(
                graph_module=ep._create_graph_module_for_export(
                    self.module, self.graph
                ),
                signature=self.signature,
                module_call_graph=module_call_graph,
                names_to_symbols=self.symbol_name_to_symbol,
                state_dict=deserialize_torch_artifact(serialized_state_dict),
                constants=self.constants,
                example_inputs=self.example_inputs,
            )
        finally:
            # 最终将当前的反序列化器设置为 None，表示反序列化结束
            _CURRENT_DESERIALIZER = None

    def sync_fx_node(self, name: str, fx_node: torch.fx.Node):
        # 如果节点名称已经在已序列化的名称到节点映射中存在，则抛出序列化错误
        if name in self.serialized_name_to_node:
            raise SerializeError(f"Node {name} has already been deserialized before.")
        # 将节点名称与节点对象映射存入已序列化的名称到节点映射中
        self.serialized_name_to_node[name] = fx_node
        # 确保节点的元数据中不存在 "val" 字段
        assert "val" not in fx_node.meta
        # 向节点的元数据中添加 "val" 字段，并赋值为相应名称的序列化元数据
        fx_node.meta["val"] = self.serialized_name_to_meta[name]

    def deserialize_sym_op_inputs(self, inputs):
        # 对输入列表中的每个输入进行反序列化处理，返回处理后的元组
        return tuple(self.deserialize_input(input.arg) for input in inputs)
    # 反序列化操作的输入参数，将序列化后的节点反序列化为目标操作的输入
    def deserialize_inputs(self, target: torch._ops.OpOverload, serialized_node: Node):
        # 获取目标操作的参数模式
        schema_args = target._schema.arguments
        # 实际参数字典，包含从序列化节点中获取的输入参数
        actual_args = {
            input.name: self.deserialize_input(input.arg)
            for input in serialized_node.inputs
        }
        # 位置参数列表和关键字参数字典初始化
        args = []
        kwargs = {}
        # 遍历目标操作的参数模式
        for schema_arg in schema_args:
            # 判断是否为位置参数
            is_positional = (
                not schema_arg.has_default_value() and not schema_arg.kwarg_only
            )
            if is_positional:
                # 添加到位置参数列表中
                args.append(actual_args[schema_arg.name])
            else:
                # 如果是关键字参数且存在于实际参数中，则添加到关键字参数字典中
                if schema_arg.name in actual_args:
                    kwargs[schema_arg.name] = actual_args[schema_arg.name]
        # 返回位置参数元组和关键字参数字典
        return tuple(args), kwargs

    # 反序列化高阶操作的输入参数列表
    def deserialize_hoo_inputs(self, inputs: List[NamedArgument]):
        """
        用于反序列化高阶操作的输入参数，因为高阶操作没有参数模式。
        """
        args = []
        kwargs = {}
        # 遍历输入参数列表
        for input_ in inputs:
            if input_.name != "":
                # 如果参数有名称，则作为关键字参数添加到 kwargs 中
                kwargs[input_.name] = self.deserialize_input(input_.arg)
            else:
                # 否则作为位置参数添加到 args 中
                args.append(self.deserialize_input(input_.arg))
        # 返回位置参数元组和关键字参数字典
        return (tuple(args), kwargs)

    # 反序列化常量输入参数
    def deserialize_constant_input(self, inp: ConstantValue) -> Any:
        # 根据常量值的类型进行反序列化
        if inp.type == "as_int":
            return int(inp.as_int)
        elif inp.type == "as_float":
            return float(inp.as_float)
        elif inp.type == "as_string":
            return str(inp.as_string)
        elif inp.type == "as_bool":
            return bool(inp.as_bool)
        elif inp.type == "as_none":
            return None
        else:
            # 抛出异常，处理未处理的常量参数类型
            raise SerializeError(f"Unhandled constant argument {inp} to deserialize")

    # 反序列化符号参数
    def deserialize_sym_argument(self, sym_arg):
        # 根据符号参数的类型进行反序列化
        if isinstance(sym_arg, SymIntArgument):
            if sym_arg.type == "as_int":
                return sym_arg.as_int
            elif sym_arg.type == "as_name":
                # 返回符号名称对应的节点
                return self.serialized_name_to_node[sym_arg.as_name]
        elif isinstance(sym_arg, SymBoolArgument):
            if sym_arg.type == "as_bool":
                return sym_arg.as_bool
            elif sym_arg.type == "as_name":
                # 返回符号名称对应的节点
                return self.serialized_name_to_node[sym_arg.as_name]
        # 抛出异常，处理未知的符号参数类型
        raise SerializeError(f"Unknown symbolic argument type: {sym_arg}")

    # 同步符号操作的输出节点
    def deserialize_sym_op_outputs(self, serialized_node: Node, fx_node: torch.fx.Node):
        # 同步序列化节点的输出节点到 fx_node
        self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)
    # 反序列化节点的输出结果
    def deserialize_outputs(self, serialized_node: Node, fx_node: torch.fx.Node):
        # 检查是否只有一个输出值
        if len(serialized_node.outputs) == 0:
            return  # 如果没有输出值，则直接返回

        # 检查是否只有一个输出值且类型为 "as_tensor"
        if (
            len(serialized_node.outputs) == 1
            and serialized_node.outputs[0].type == "as_tensor"
        ):
            # 同步 TorchFX 节点和序列化节点的信息
            self.sync_fx_node(serialized_node.outputs[0].as_tensor.name, fx_node)
            return  # 处理完毕后直接返回

        # 检查是否只有一个输出值且类型为 SymIntArgument 或 SymBoolArgument
        elif len(serialized_node.outputs) == 1 and isinstance(
            serialized_node.outputs[0].value, (SymIntArgument, SymBoolArgument)
        ):
            # 同步 TorchFX 节点和序列化节点的信息
            self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)
            return  # 处理完毕后直接返回

        # 如果有多个输出值，则调用方法处理
        self.deserialize_multiple_outputs(serialized_node, fx_node)

    # 反序列化多个输出结果的节点
    def deserialize_multiple_outputs(
        self, serialized_node: Node, fx_node: torch.fx.Node
    ):
        ) -> None:
            # 反序列化节点的元数据
            deserialized_metadata = self.deserialize_metadata(serialized_node.metadata)

            def generate_getitem(
                meta_val,
                fx_node: torch.fx.Node,
                arg: Union[TensorArgument, SymIntArgument],
                idx: int,
            ):
                # 根据参数类型确定节点名称
                if isinstance(arg, TensorArgument):
                    name = arg.name
                elif isinstance(arg, SymIntArgument):
                    name = arg.as_name
                else:
                    raise AssertionError(
                        f"generate_getitem got unknown argument type {type(arg)}"
                    )
                # 创建一个 `call_function` 节点，表示从 `fx_node` 中获取第 `idx` 个元素
                individual_output = self.graph.create_node(
                    "call_function",
                    operator.getitem,
                    (fx_node, idx),
                    name=name,
                )
                # 同步 `fx_node` 和 `individual_output` 的元数据
                self.sync_fx_node(name, individual_output)
                # 将序列化后的名称映射到元数据中的值追加到 `meta_val` 中
                meta_val.append(self.serialized_name_to_meta[name])
                # 派生的 `getitem` 节点应该与原始 `fx_node` 具有相同的堆栈跟踪信息
                individual_output.meta.update(deserialized_metadata)

            def generate_getitems(meta_val, fx_node: torch.fx.Node, args):
                # 遍历参数列表，并根据参数类型生成相应的 `getitem` 节点
                for idx, arg in enumerate(args):
                    if isinstance(arg, Argument):
                        arg = arg.value
                    if isinstance(arg, (TensorArgument, SymIntArgument)):
                        generate_getitem(meta_val, fx_node, arg, idx)
                    elif isinstance(arg, (list, tuple)):
                        # 对于列表或元组类型的参数，递归生成 `getitem` 节点
                        list_output = self.graph.create_node(
                            "call_function",
                            operator.getitem,
                            (fx_node, idx),
                        )
                        meta_val.append([])
                        generate_getitems(meta_val[-1], list_output, arg)
                        # 更新节点的元数据，指定值为 `meta_val` 的列表
                        list_output.meta.update(deserialized_metadata)
                        list_output.meta["val"] = meta_val[-1]
                    else:
                        raise NotImplementedError(f"Unimplemented node output type: {arg}")

            # 将多个返回类型转换为 FX 格式
            # 在 FX 中，每个节点只返回一个值。因此，为了表示多个返回值，我们必须为每个返回值发出一个 `getitem` 节点
            # 这执行了序列化中 `serialize_outputs` 调用的反向映射，请参见 [NOTE: Multiple outputs]
            meta_val: List[Any] = []
            if len(serialized_node.outputs) == 1:
                assert isinstance(serialized_node.outputs[0].value, list)
                assert isinstance(serialized_node.outputs[0].value[0], TensorArgument)
                generate_getitems(meta_val, fx_node, serialized_node.outputs[0].as_tensors)
            else:
                generate_getitems(meta_val, fx_node, serialized_node.outputs)

            # 更新 `fx_node` 的元数据，指定值为 `meta_val` 的元组
            fx_node.meta["val"] = tuple(meta_val)
            # 将 `fx_node` 的名称映射到 `fx_node` 本身
            self.serialized_name_to_node[fx_node.name] = fx_node
    # 反序列化元数据，将元数据字典转换为包含不同类型值的字典
    def deserialize_metadata(self, metadata: Dict[str, str]) -> Dict[str, Any]:
        # 初始化返回的字典
        ret: Dict[str, Any] = {}
        
        # 如果存在键为 "stack_trace" 的元数据，则将其添加到返回字典中
        if stack_trace := metadata.get("stack_trace"):
            ret["stack_trace"] = stack_trace

        # 定义一个内部函数，用于反序列化目标字符串
        def deserialize_meta_func(serialized_target: str):
            module = None
            # 如果序列化目标以 "torch.nn" 开头，则表示目标属于 torch.nn 模块
            if serialized_target.startswith("torch.nn"):
                module = torch.nn
                serialized_target_names = serialized_target.split(".")[2:]
            # 如果序列化目标以 "torch" 开头，则表示目标属于 torch 模块
            elif serialized_target.startswith("torch"):
                module = torch
                serialized_target_names = serialized_target.split(".")[1:]
            # 否则，调用 deserialize_operator 方法处理操作符
            else:
                return self.deserialize_operator(serialized_target)

            # 从模块中逐级获取目标对象
            target = module
            for name in serialized_target_names:
                if not hasattr(target, name):
                    return serialized_target
                else:
                    target = getattr(target, name)
            return target

        # 如果存在键为 "nn_module_stack" 的元数据，则进行处理
        if nn_module_stack_str := metadata.get("nn_module_stack"):
            # 定义一个辅助函数，将元数据分割成键、路径和类型
            def import_nn_module_stack(key, path, ty):
                return key, (path, ty)

            # 定义一个函数，根据特定规则分割元数据字符串
            def metadata_split(metadata):
                # 去除字符串中的括号及其中的内容
                metadata = re.sub(r'\(.*?\)', '', metadata)
                # 按逗号分割字符串，但排除括号内的逗号
                return re.split(r'(?<!\()\s*,\s*(?!\()', metadata)

            # 根据特定分隔符拆分 nn_module_stack_str，并构建字典
            nn_module_stack = dict(
                import_nn_module_stack(*metadata_split(item))
                for item in nn_module_stack_str.split(ST_DELIMITER)
            )
            ret["nn_module_stack"] = nn_module_stack

        # 如果存在键为 "source_fn_stack" 的元数据，则进行处理
        if source_fn_st_str := metadata.get("source_fn_stack"):
            # 按照特定分隔符拆分 source_fn_st_str，并反序列化其中的目标字符串
            source_fn_st = []
            for source_fn_str in source_fn_st_str.split(ST_DELIMITER):
                name, target_str = source_fn_str.split(",")
                source_fn_st.append((name, deserialize_meta_func(target_str)))
            ret["source_fn_stack"] = source_fn_st

        # 如果存在键为 "torch_fn" 的元数据，则按特定分隔符拆分为元组并添加到返回字典中
        if torch_fn_str := metadata.get("torch_fn"):
            ret["torch_fn"] = tuple(torch_fn_str.split(ST_DELIMITER))
        
        # 返回处理后的元数据字典
        return ret
    # 反序列化给定的参数规范对象为引擎的参数规范对象
    def deserialize_argument_spec(self, x: Argument) -> ep.ArgumentSpec:
        # 如果参数类型为 "as_tensor"，返回一个张量参数对象
        if x.type == "as_tensor":
            return ep.TensorArgument(name=x.as_tensor.name)
        # 如果参数类型为 "as_sym_int"，返回一个符号整数参数对象
        elif x.type == "as_sym_int":
            return ep.SymIntArgument(name=x.as_sym_int.as_name)
        # 如果参数类型为 "as_custom_obj"，返回一个常量参数对象，其值由反序列化输入确定
        elif x.type == "as_custom_obj":
            return ep.ConstantArgument(name=x.as_custom_obj.name, value=self.deserialize_input(x))
        # 如果类型未知或未定义，返回一个空名称的常量参数对象，其值由反序列化输入确定
        else:
            return ep.ConstantArgument(name="", value=self.deserialize_input(x))

    # 反序列化给定的模块调用签名对象为引擎的模块调用签名对象
    def deserialize_module_call_signature(
        self, module_call_signature: ModuleCallSignature
    ) -> ep.ModuleCallSignature:
        # 构造模块调用签名对象，其中包括输入参数列表和输出参数列表的反序列化结果，
        # 以及输入和输出规范的树形结构加载结果
        return ep.ModuleCallSignature(
            inputs=[
                self.deserialize_argument_spec(x) for x in module_call_signature.inputs
            ],
            outputs=[
                self.deserialize_argument_spec(x) for x in module_call_signature.outputs
            ],
            in_spec=treespec_loads(module_call_signature.in_spec),
            out_spec=treespec_loads(module_call_signature.out_spec),
        )

    # 反序列化给定的模块调用图对象为引擎的模块调用图条目列表
    def deserialize_module_call_graph(
        self, module_call_graph: List[ModuleCallEntry]
    ) -> List[ep.ModuleCallEntry]:
        # 使用列表推导式，对输入的模块调用图中的每个条目进行反序列化，
        # 构造相应的引擎模块调用图条目列表
        return [
            ep.ModuleCallEntry(
                fqn=entry.fqn,
                signature=(
                    self.deserialize_module_call_signature(entry.signature)
                    if entry.signature
                    else None
                ),
            )
            for entry in module_call_graph
        ]
@final
class ExportedProgramDeserializer(metaclass=Final):
    # 定义一个不可继承的最终类，用于反序列化导出的程序

    def __init__(self, expected_opset_version: Optional[Dict[str, int]] = None):
        # 初始化方法，设置期望的运算集版本号字典，默认为空字典
        self.expected_opset_version: Dict[str, int] = {}
        if expected_opset_version:
            self.expected_opset_version.update(expected_opset_version)
        if "aten" not in self.expected_opset_version:
            self.expected_opset_version["aten"] = torch._C._get_max_operator_version()
        # 如果传入了期望的运算集版本号，则更新到实例的属性中；如果未包含 "aten" 键，则从 torch 获取最大操作版本号

    def deserialize_range_constraints(
        self,
        symbol_name_to_range: Dict[str, symbolic_shapes.ValueRanges],
        symbol_name_to_symbol: Dict[str, sympy.Symbol],
    ) -> Dict[sympy.Symbol, ValueRanges]:
        # 反序列化符号名称到值范围的字典，返回符号到值范围的字典
        range_constraints = {}
        for k, v in symbol_name_to_range.items():
            if symbol := symbol_name_to_symbol.get(k):
                range_constraints[symbol] = v  # 将符号及其对应的值范围添加到结果字典中
            else:
                log.warning(f"Symbol {k} did not appear in the graph that was deserialized")  # 警告：符号未在反序列化的图中出现
        return range_constraints

    def deserialize(
        self,
        exported_program: ExportedProgram,
        state_dict: Union[Dict[str, torch.Tensor], bytes],
        constants: Union[Dict[str, torch.Tensor], bytes],
        example_inputs: Optional[Union[Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]], bytes]] = None,
    ) -> ep.ExportedProgram:
        # 反序列化方法，将导出的程序反序列化为 ExportedProgram 对象
        assert isinstance(exported_program, ExportedProgram)
        version = exported_program.schema_version

        # TODO(zhxchen17) blocked on thrift schema refactor
        # TODO: 阻塞在 thrift 模式重构上

        if version.major != SCHEMA_VERSION[0] and not (version.major == 0 and version.minor == 0):
            raise SerializeError(
                f"Serialized schema version {exported_program.schema_version} "
                f"does not match our current schema version {SCHEMA_VERSION}."
            )
        # 如果导出的模式版本与当前模式版本不匹配，则引发序列化错误

        symbol_name_to_range = {
            k: symbolic_shapes.ValueRanges(
                _int_to_sympy_int(v.min_val), _int_to_sympy_int(v.max_val)
            )
            for k, v in exported_program.range_constraints.items()
        }
        # 创建符号名称到值范围对象的字典，将导出程序的值范围项转换为符号形式

        res = (
            GraphModuleDeserializer()
            .deserialize(
                exported_program.graph_module,
                state_dict,
                constants,
                example_inputs,
                symbol_name_to_range,
            )
        )
        # 使用 GraphModuleDeserializer 反序列化导出程序的图模块，返回反序列化结果对象

        range_constraints = self.deserialize_range_constraints(
            symbol_name_to_range,
            res.names_to_symbols,
        )
        # 使用先前定义的方法反序列化值范围约束

        return ep.ExportedProgram(
            root=res.graph_module,
            graph=res.graph_module.graph,
            graph_signature=res.signature,
            state_dict=res.state_dict,  # 状态字典
            range_constraints=range_constraints,  # 值范围约束
            module_call_graph=res.module_call_graph,
            example_inputs=res.example_inputs,
            verifier=load_verifier(exported_program.dialect),
            constants=res.constants,
        )
        # 返回一个新的 ExportedProgram 对象，包含反序列化后的各项属性
# 定义一个自定义的 JSON 编码器类 EnumEncoder，继承自 json.JSONEncoder
class EnumEncoder(json.JSONEncoder):
    
    # 重写 default 方法，处理对象的序列化逻辑
    def default(self, obj):
        # 如果对象是枚举类型，则返回其值
        if isinstance(obj, Enum):
            return obj.value
        # 如果对象是字节类型，则将其 base64 编码后以 UTF-8 解码返回
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("utf-8")
        # 其他情况调用父类的 default 方法处理
        return super().default(obj)


# 定义一个函数 _dataclass_to_dict，将数据类对象转换为字典
def _dataclass_to_dict(obj):
    # 如果对象是 _Union 类型，递归处理其 value 属性并以 type 为键构建字典
    if isinstance(obj, _Union):
        return {obj.type: _dataclass_to_dict(obj.value)}
    # 如果对象是数据类（dataclass），将其字段及对应值构建为字典
    elif dataclasses.is_dataclass(obj):
        return {
            f.name: _dataclass_to_dict(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            # 排除默认值为 None 且属性值也为 None 的字段
            if not (f.default is None and getattr(obj, f.name) is None)
        }
    # 如果对象是列表，则递归处理列表中的每个元素
    elif isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    # 如果对象是元组，则递归处理元组中的每个元素
    elif isinstance(obj, tuple):
        return tuple(_dataclass_to_dict(x) for x in obj)
    # 如果对象是字典，则递归处理字典中的每个键值对
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    # 其他情况直接返回对象本身
    else:
        return obj


# 定义函数 serialize，将导出的程序对象序列化为 SerializedArtifact
def serialize(
    exported_program: ep.ExportedProgram,
    opset_version: Optional[Dict[str, int]] = None,
) -> SerializedArtifact:
    # 使用 ExportedProgramSerializer 对象对导出的程序对象进行序列化
    serialized_program = ExportedProgramSerializer(opset_version).serialize(
        exported_program
    )
    # 断言 serialized_program.exported_program 是 ExportedProgram 类型的实例
    assert isinstance(serialized_program.exported_program, ExportedProgram)
    
    # 将 serialized_program.exported_program 转换为字典并使用 EnumEncoder 进行 JSON 序列化
    json_program = json.dumps(
        _dataclass_to_dict(serialized_program.exported_program), cls=EnumEncoder
    )
    # 将 JSON 字符串转换为 UTF-8 编码的字节流
    json_bytes = json_program.encode("utf-8")
    # 创建 SerializedArtifact 对象，包括 JSON 字节流及其他属性
    artifact = SerializedArtifact(
        json_bytes,
        serialized_program.state_dict,
        serialized_program.constants,
        serialized_program.example_inputs
    )
    # 返回序列化后的 SerializedArtifact 对象
    return artifact


# 定义一个私有函数 _dict_to_dataclass，将字典数据转换为数据类对象
def _dict_to_dataclass(cls, data):
    # 断言 cls 不是字符串类型，如果是字符串则抛出异常
    assert not isinstance(cls, str), f"Unresolved class type: '{cls}'."
    
    # 处理 Union 类型，如果数据为 None，则返回 None；否则根据类型信息继续处理
    if typing.get_origin(cls) == typing.Union and type(None) in typing.get_args(cls):
        if data is None:
            return None
        ty_args = typing.get_args(cls)
        assert len(ty_args) == 2
        return _dict_to_dataclass(ty_args[0], data)
    
    # 处理 _Union 类型，根据字典中的信息创建 _Union 对象
    elif isinstance(cls, type) and issubclass(cls, _Union):
        assert isinstance(data, dict)
        assert len(data) == 1
        _type = next(iter(data.keys()))
        _value = next(iter(data.values()))
        assert isinstance(_type, str)
        field_type = cls.__annotations__[_type]
        return cls.create(**{_type: _dict_to_dataclass(field_type, _value)})
    
    # 处理数据类（dataclass），根据字典创建数据类对象
    elif dataclasses.is_dataclass(cls):
        obj = cls(**data)  # type: ignore[assignment]
        type_hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            name = f.name
            new_field_obj = _dict_to_dataclass(type_hints[name], getattr(obj, name))
            setattr(obj, name, new_field_obj)
        return obj
    
    # 处理列表，递归处理列表中的每个元素
    elif isinstance(data, list):
        if len(data) == 0:
            return data
        d_type = typing.get_args(cls)[0]
        return [_dict_to_dataclass(d_type, d) for d in data]
    # 如果 data 是字典类型，则进入这个分支处理
    elif isinstance(data, dict):
        # 获取类 cls 的第二个参数类型
        v_type = typing.get_args(cls)[1]
        # 使用递归调用 _dict_to_dataclass 处理字典中每个键值对，将值转换为对应的数据类
        return {k: _dict_to_dataclass(v_type, v) for k, v in data.items()}
    # 如果 data 不是字典类型，则直接返回 data
    return data
def _canonicalize_graph(
    sorted_inputs, sorted_outputs, graph
) -> Tuple[Graph, Dict[str, str]]:
    # 定义一个内部函数，用于根据不同的参数类型返回对应的值
    def _get_argument(a: Argument):
        if a.type == "as_none":
            return None
        elif a.type == "as_tensor":
            return a.as_tensor
        elif a.type == "as_tensors":
            return a.as_tensors
        elif a.type == "as_int":
            return None
        elif a.type == "as_ints":
            return None
        elif a.type == "as_float":
            return None
        elif a.type == "as_floats":
            return None
        elif a.type == "as_string":
            return None
        elif a.type == "as_strings":
            return None
        elif a.type == "as_sym_int":
            return a.as_sym_int
        elif a.type == "as_sym_ints":
            return a.as_sym_ints
        elif a.type == "as_scalar_type":
            return None
        elif a.type == "as_memory_format":
            return None
        elif a.type == "as_layout":
            return None
        elif a.type == "as_device":
            return None
        elif a.type == "as_bool":
            return None
        elif a.type == "as_bools":
            return None
        elif a.type == "as_sym_bool":
            return a.as_sym_bool
        elif a.type == "as_sym_bools":
            return a.as_sym_bools
        elif a.type == "as_graph":
            return None
        elif a.type == "as_optional_tensors":
            return a.as_optional_tensors
        elif a.type == "as_custom_obj":
            return None
        elif a.type == "as_operator":
            return None
        else:
            raise AssertionError(f"Unknown input type to the ExportedProgram: {a}")

    # Stage 1: Reorder named items.
    # 使用 sort_nodes 函数对图的节点进行排序
    sorted_nodes = sort_nodes(graph.nodes)
    # 断言排序后节点的数量与原节点数量相等
    assert len(sorted_nodes) == len(graph.nodes)

    # Stage 2: Rename nodes.
    # 创建一个空的字典，用于存储节点的重命名映射关系
    name_table: Dict[str, str] = {}
    def rename_def(a):
        def _rename(arg_name, values):
            new_name = f"_{len(name_table)}"
            assert arg_name not in name_table
            name_table[arg_name] = new_name
            assert arg_name in values
            values[new_name] = values.pop(arg_name)
            return new_name
        # 如果参数 a 为 None，则直接返回，不做任何操作
        if a is None:
            return
        # 如果 a 是 TensorArgument 类型，则调用 _rename 函数重命名其名称，并更新到图的张量值中
        if isinstance(a, TensorArgument):
            a.name = _rename(a.name, graph.tensor_values)
        # 如果 a 是 SymIntArgument 类型，并且其类型为 "as_name"，则调用 _rename 函数重命名其 as_name，并更新到图的符号整数值中
        elif isinstance(a, SymIntArgument):
            if a.type == "as_name":
                a.as_name = _rename(a.as_name, graph.sym_int_values)
        # 如果 a 是 SymBoolArgument 类型，并且其类型为 "as_name"，则调用 _rename 函数重命名其 as_name，并更新到图的符号布尔值中
        elif isinstance(a, SymBoolArgument):
            if a.type == "as_name":
                a.as_name = _rename(a.as_name, graph.sym_bool_values)
        # 如果以上条件都不满足，则抛出断言错误，表示未知的参数类型
        else:
            raise AssertionError(f"Unknown argument type: {a}")

    def replace_use(a):
        # 如果参数 a 为 None，则直接返回，不做任何操作
        if a is None:
            return
        # 如果 a 是 TensorArgument 类型，则尝试从 name_table 中获取其新名称，并更新到 a.name 中
        if isinstance(a, TensorArgument):
            a.name = name_table.get(a.name, a.name)
        # 如果 a 是 SymIntArgument 类型，并且其类型为 "as_name"，则尝试从 name_table 中获取其新名称，并更新到 a.as_name 中
        elif isinstance(a, SymIntArgument):
            if a.type == "as_name":
                a.as_name = name_table.get(a.as_name, a.as_name)
        # 如果 a 是 SymBoolArgument 类型，并且其类型为 "as_name"，则尝试从 name_table 中获取其新名称，并更新到 a.as_name 中
        elif isinstance(a, SymBoolArgument):
            if a.type == "as_name":
                a.as_name = name_table.get(a.as_name, a.as_name)
        # 如果 a 是 OptionalTensorArgument 类型，并且其类型为 "as_tensor"，则尝试从 name_table 中获取其新名称，并更新到 a.as_tensor.name 中
        elif isinstance(a, OptionalTensorArgument):
            if a.type == "as_tensor":
                a.as_tensor.name = name_table.get(a.as_tensor.name, a.as_tensor.name)
        # 如果以上条件都不满足，则抛出断言错误，表示未知的参数类型
        else:
            raise AssertionError(f"Unknown argument type: {a}")

    # 对 sorted_inputs 中的每个元素调用 rename_def 函数
    for i in sorted_inputs:
        for_args(rename_def, i)

    # 对 sorted_nodes 中的每个节点的输出调用 rename_def 函数
    for n in sorted_nodes:
        for o in n.outputs:
            for_args(rename_def, o)

    # 对 sorted_nodes 中的每个节点的输入调用 replace_use 函数
    for n in sorted_nodes:
        for i in n.inputs:
            for_args(replace_use, i.arg)

    # 对 sorted_outputs 中的每个元素调用 replace_use 函数
    for o in sorted_outputs:
        for_args(replace_use, o)

    # Stage 3: Remove unstable fields.
    # 清空每个 sorted_nodes 中节点的 metadata 字典
    for n in sorted_nodes:
        n.metadata.clear()

    # Stage 4: Aggregate values.
    # 对 graph.tensor_values 按键排序并形成字典 sorted_tensor_values
    sorted_tensor_values = dict(sorted(graph.tensor_values.items(), key=operator.itemgetter(0)))
    # 对 graph.sym_int_values 按键排序并形成字典 sorted_sym_int_values
    sorted_sym_int_values = dict(sorted(graph.sym_int_values.items(), key=operator.itemgetter(0)))
    # 对 graph.sym_bool_values 按键排序并形成字典 sorted_sym_bool_values
    sorted_sym_bool_values = dict(sorted(graph.sym_bool_values.items(), key=operator.itemgetter(0)))

    # Stage 5: Recurse in subgraphs.
    # 初始化计数器 counter
    counter = 0
    # 遍历 sorted_nodes 中的每个节点
    for node in sorted_nodes:
        # 遍历节点的输入
        for i in node.inputs:
            a = i.arg
            # 如果输入参数 a 的类型为 "as_graph"，则递归调用 _canonicalize_graph 函数进行规范化处理，并重新命名其 as_graph.name 属性
            if a.type == "as_graph":
                a.as_graph.graph, _ = _canonicalize_graph(
                    a.as_graph.graph.inputs, a.as_graph.graph.outputs, a.as_graph.graph
                )
                a.as_graph.name = f"_g{counter}"
                counter += 1
    # 创建一个名为 graph 的 Graph 对象，用于表示计算图
    graph = Graph(
        inputs=sorted_inputs,  # 排序后的输入节点列表
        outputs=sorted_outputs,  # 排序后的输出节点列表
        nodes=sorted_nodes,  # 排序后的节点列表
        tensor_values=sorted_tensor_values,  # 排序后的张量数值字典
        sym_int_values=sorted_sym_int_values,  # 排序后的符号整数值字典
        sym_bool_values=sorted_sym_bool_values,  # 排序后的符号布尔值字典
        is_single_tensor_return=graph.is_single_tensor_return,  # 计算图是否仅返回单个张量
    )
    
    # 返回创建的 graph 对象和 name_table
    return graph, name_table
    """
    Normalize a serialized ExportedProgram, so that different eager program which
    shares the same semantics can get a single representation on disk.

    This function canonicalizes an ExportedProgram by:

    1. Sorting nodes in topological order.
    2. Rename nodes to have unique names.
    3. Remove unstable fields.
    4. Aggregate the above program fields.
    5. Recurse in subgraphs.

    Args:
        ep (ExportedProgram): The ExportedProgram to canonicalize.

    Returns:
        ExportedProgram: The canonicalized exported program.
    """
    # 深度复制输入的 ExportedProgram 对象，以避免在规范化过程中修改原始对象
    ep = copy.deepcopy(ep)

    # 根据操作集版本号字典的键排序，返回新的有序字典
    opset_version = dict(sorted(ep.opset_version.items(), key=operator.itemgetter(0)))
    # 根据范围约束字典的键排序，返回新的有序字典
    range_constraints = dict(sorted(ep.range_constraints.items(), key=operator.itemgetter(0)))
    # 根据模块调用图中的完全限定名称排序子图模块
    module_call_graph = sorted(ep.graph_module.module_call_graph, key=lambda x: x.fqn)
    # 获取图模块的签名
    signature = ep.graph_module.signature
    # 获取图模块的图对象
    graph = ep.graph_module.graph

    # 断言输入节点数量与签名中输入规范数量相等
    assert len(graph.inputs) == len(signature.input_specs)
    # 断言输出节点数量与签名中输出规范数量相等
    assert len(graph.outputs) == len(signature.output_specs)

    # 定义排序输入的函数，返回排序后的元组列表
    def rank_input(inp) -> Tuple[int, Optional[str], int]:
        idx, (arg, spec) = inp
        assert isinstance(spec, InputSpec)
        # 根据输入规范的类型进行排序
        if spec.type == "user_input":
            return 5, None, idx
        elif spec.type == "parameter":
            return 1, spec.parameter.parameter_name, idx
        elif spec.type == "buffer":
            return 2, spec.buffer.buffer_name, idx
        elif spec.type == "tensor_constant":
            return 3, spec.tensor_constant.tensor_constant_name, idx
        elif spec.type == "custom_obj":
            return 4, spec.custom_obj.custom_obj_name, idx
        elif spec.type == "token":
            return 0, None, idx
        elif spec.type == "constant_input":
            return 6, spec.constant_input.name, idx
        else:
            raise AssertionError(f"Unknown input type: {spec}")

    # 定义排序输出的函数，返回排序后的元组列表
    def rank_output(out) -> Tuple[int, Optional[str], int]:
        idx, (arg, spec) = out
        assert isinstance(spec, OutputSpec)
        # 根据输出规范的类型进行排序
        if spec.type == "user_output":
            return 3, None, idx
        elif spec.type == "loss_output":
            return 3, None, idx
        elif spec.type == "buffer_mutation":
            return 1, spec.buffer_mutation.buffer_name, idx
        elif spec.type == "gradient_to_parameter":
            return 4, spec.gradient_to_parameter.parameter_name, idx
        elif spec.type == "gradient_to_user_input":
            return 5, None, idx
        elif spec.type == "user_input_mutation":
            return 2, None, idx
        elif spec.type == "token":
            return 0, None, idx
        else:
            raise AssertionError(f"Unknown output type: {spec}")

    # 对图输入和签名输入进行排序，返回排序后的输入列表和输入规范列表
    sorted_ins = sorted(
        enumerate(zip(graph.inputs, signature.input_specs)), key=rank_input
    )
    sorted_inputs, input_specs = zip(*(i for idx, i in sorted_ins))  # type: ignore[assignment]
    # 对 graph.outputs 和 signature.output_specs 进行排序，使用 rank_output 函数进行排序的依据
    sorted_outs = sorted(
        enumerate(zip(graph.outputs, signature.output_specs)), key=rank_output
    )
    
    # 将排序后的结果解压缩为两个列表 sorted_outputs 和 output_specs
    sorted_outputs, output_specs = zip(*(i for idx, i in sorted_outs))  # type: ignore[assignment]
    
    # 对输入、输出和图形进行规范化处理，生成规范化后的图和替换表
    sorted_graph, replace_table = _canonicalize_graph(
        sorted_inputs, sorted_outputs, graph
    )
    
    # 定义一个函数 replace_input，用于根据不同的输入类型替换相应的内容
    def replace_input(inp):
        # 确保 spec 是 InputSpec 类型的实例
        assert isinstance(spec, InputSpec)
        if spec.type == "user_input":
            # 如果输入类型是 "user_input"
            arg = spec.user_input.arg
            if arg.type == "as_tensor":
                # 如果参数类型是 "as_tensor"，更新 tensor 的名称为替换表中对应的名称
                t = arg.as_tensor
                t.name = replace_table[t.name]
            elif arg.type == "as_sym_int":
                # 如果参数类型是 "as_sym_int"
                s = arg.as_sym_int
                if s.type == "as_name":
                    # 如果符号整数类型是 "as_name"，更新其名称为替换表中对应的名称
                    s.as_name = replace_table[s.as_name]
                elif s.type == "as_int":
                    # 如果符号整数类型是 "as_int"，不做任何操作
                    pass
                else:
                    # 抛出异常，表示未知的符号整数类型
                    raise AssertionError(f"Unknown sym_int type: {s}")
            elif arg.type in (
                "as_none",
                "as_bool",
                "as_int",
                "as_float",
                "as_string",
                "as_custom_obj",
            ):
                # 如果参数类型在已知的基本类型中，直接返回
                return
            else:
                # 抛出异常，表示未知的输入类型
                raise AssertionError(f"Unknown input type: {arg}")
        elif spec.type == "parameter":
            # 如果输入类型是 "parameter"，更新参数的名称为替换表中对应的名称
            t = spec.parameter.arg
            t.name = replace_table[t.name]
        elif spec.type == "buffer":
            # 如果输入类型是 "buffer"，更新缓冲区的名称为替换表中对应的名称
            t = spec.buffer.arg
            t.name = replace_table[t.name]
        elif spec.type == "tensor_constant":
            # 如果输入类型是 "tensor_constant"，更新常量张量的名称为替换表中对应的名称
            t = spec.tensor_constant.arg
            t.name = replace_table[t.name]
        elif spec.type == "custom_obj":
            # 如果输入类型是 "custom_obj"，直接返回
            return
        elif spec.type == "token":
            # 如果输入类型是 "token"，更新 token 的名称为替换表中对应的名称
            tok = spec.token.arg
            tok.name = replace_table[tok.name]
        elif spec.type == "constant_input":
            # 如果输入类型是 "constant_input"，直接返回
            return
        else:
            # 抛出异常，表示未知的输入类型
            raise AssertionError(f"Unknown input type: {spec}")
    # 定义一个函数，用于替换输出规范中的参数名称
    def replace_output(out):
        # 确保 spec 是 OutputSpec 类型的对象
        assert isinstance(spec, OutputSpec)
        
        # 如果输出类型是用户输出
        if spec.type == "user_output":
            # 获取用户输出的参数
            arg = spec.user_output.arg
            
            # 如果参数类型是作为张量
            if arg.type == "as_tensor":
                # 获取作为张量的数据，并更新其名称
                t = arg.as_tensor
                t.name = replace_table[t.name]
            
            # 如果参数类型是作为符号整数
            elif arg.type == "as_sym_int":
                # 获取作为符号整数的数据
                s = arg.as_sym_int
                
                # 如果类型是作为名称，则更新其名称
                if s.type == "as_name":
                    s.as_name = replace_table[s.as_name]
                    
                # 如果类型是作为整数，则不进行任何操作
                elif s.type == "as_int":
                    pass
                
                # 否则，抛出异常，说明未知的符号整数类型
                else:
                    raise AssertionError(f"Unknown sym_int type: {s}")
            
            # 如果参数类型是作为无类型、整数、浮点数或字符串中的一种，则直接返回
            elif arg.type in ("as_none", "as_int", "as_float", "as_string"):
                return
            
            # 如果参数类型未知，则抛出异常
            else:
                raise AssertionError(f"Unknown input type: {arg}")
        
        # 如果输出类型是损失输出，则获取损失输出的参数，并更新其名称
        elif spec.type == "loss_output":
            t = spec.loss_output.arg
            t.name = replace_table[t.name]
        
        # 如果输出类型是缓冲区突变，则获取缓冲区突变的参数，并更新其名称
        elif spec.type == "buffer_mutation":
            t = spec.buffer_mutation.arg
            t.name = replace_table[t.name]
        
        # 如果输出类型是梯度到参数，则获取梯度到参数的参数，并更新其名称
        elif spec.type == "gradient_to_parameter":
            t = spec.gradient_to_parameter.arg
            t.name = replace_table[t.name]
        
        # 如果输出类型是梯度到用户输入，则获取梯度到用户输入的参数和用户输入名称，并更新它们的名称
        elif spec.type == "gradient_to_user_input":
            g = spec.gradient_to_user_input
            g.arg.name = replace_table[g.arg.name]
            g.user_input_name = replace_table[g.user_input_name]
        
        # 如果输出类型是用户输入突变，则获取用户输入突变的参数和用户输入名称，并更新它们的名称
        elif spec.type == "user_input_mutation":
            u = spec.user_input_mutation
            u.arg.name = replace_table[u.arg.name]
            u.user_input_name = replace_table[u.user_input_name]
        
        # 如果输出类型是令牌，则获取令牌的参数，并更新其名称
        elif spec.type == "token":
            tok = spec.token.arg
            tok.name = replace_table[tok.name]
        
        # 如果输出类型未知，则抛出异常
        else:
            raise AssertionError(f"Unknown output type: {spec}")
    
    # 遍历输入规范列表中的每个规范，并替换其输入
    for spec in input_specs:
        replace_input(spec)
    
    # 遍历输出规范列表中的每个规范，并替换其输出
    for spec in output_specs:
        replace_output(spec)
    
    # 返回一个导出程序对象，包括排序后的图形、输入规范、输出规范、模块调用图和其他信息
    return ExportedProgram(
        graph_module=GraphModule(
            graph=sorted_graph,
            signature=GraphSignature(
                input_specs=list(input_specs),
                output_specs=list(output_specs),
            ),
            module_call_graph=module_call_graph,
        ),
        opset_version=opset_version,
        range_constraints=range_constraints,
        schema_version=ep.schema_version,
        dialect=ep.dialect
    )
class CustomOpHandler:
    """
    Base class for handling custom operators.
    """
    @classmethod
    def namespace(cls):
        # 抽象方法：返回自定义操作处理器的命名空间
        raise NotImplementedError(f"{cls.__class__} namespace() must be implemented")

    @classmethod
    def op_name(cls, op_type):
        # 抽象方法：根据操作类型返回操作名称
        raise NotImplementedError(f"{cls.__class__} op_name() must be implemented")

    @classmethod
    def op_type(cls, op_name):
        # 抽象方法：根据操作名称返回操作类型
        raise NotImplementedError(f"{cls.__class__} op_type() must be implemented")

    @classmethod
    def op_schema(cls, op_type):
        # 抽象方法：根据操作类型返回操作的模式
        raise NotImplementedError(f"{cls.__class__} op_schema() must be implemented")


def register_custom_op_handler(
    op_handler: CustomOpHandler,
    op_type: Type[Any],
):
    """Register custom de/serialization method for a node."""
    assert isinstance(op_handler, CustomOpHandler), f"Expected CustomOpHandler, got {type(op_handler)}."
    _serialization_registry[op_type] = op_handler
    # FIXME: handles deserialization later.
    # 注册自定义操作处理器到反序列化注册表
    _deserialization_registry[op_handler.namespace()] = op_handler


def allowed_registered_op_types():
    return tuple(
        _serialization_registry.keys()
    )


# Registry to store all custom serialization implementations.
# The registry maps a operation to its serialization function (a callable), in their own
# namespace to avoid conflicts.
# Serialization: Op type --> custom handler.
# De-serialization: Namespace --> custom handler.
_serialization_registry: Dict[Type[Any], CustomOpHandler] = {}
_deserialization_registry: Dict[str, CustomOpHandler] = {}
```