# `.\pytorch\torch\_export\serde\schema.py`

```
# NOTE: This is a placeholder for iterating on export serialization schema design.
#       Anything is subject to change and no guarantee is provided at this point.
# 用于导出序列化模式设计的占位符，请注意这里的任何内容都可能会更改，目前不提供任何保证。

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from torch._export.serde.union import _Union

# NOTE: Please update this value if any modifications are made to the schema
# 如果对模式进行了任何修改，请更新此值
SCHEMA_VERSION = (5, 3)
TREESPEC_VERSION = 1

# 定义枚举类型 ScalarType，用于表示张量的标量类型
class ScalarType(IntEnum):
    UNKNOWN = 0
    BYTE = 1
    CHAR = 2
    SHORT = 3
    INT = 4
    LONG = 5
    HALF = 6
    FLOAT = 7
    DOUBLE = 8
    COMPLEXHALF = 9
    COMPLEXFLOAT = 10
    COMPLEXDOUBLE = 11
    BOOL = 12
    BFLOAT16 = 13

# 定义枚举类型 Layout，用于表示张量的布局类型
class Layout(IntEnum):
    Unknown = 0
    SparseCoo = 1
    SparseCsr = 2
    SparseCsc = 3
    SparseBsr = 4
    SparseBsc = 5
    _mkldnn = 6
    Strided = 7

# 定义枚举类型 MemoryFormat，用于表示张量的内存格式
class MemoryFormat(IntEnum):
    Unknown = 0
    ContiguousFormat = 1
    ChannelsLast = 2
    ChannelsLast3d = 3
    PreserveFormat = 4

# @dataclass 用于定义类的数据结构
@dataclass
class Device:
    type: str
    index: Optional[int] = None

# @dataclass(repr=False) 用于定义类的数据结构，不生成默认的 __repr__ 方法
@dataclass(repr=False)
class SymExprHint(_Union):
    as_int: int
    as_float: float
    as_bool: bool

# @dataclass 用于定义类的数据结构
@dataclass
class SymExpr:
    expr_str: str
    hint: Optional[SymExprHint] = None

# @dataclass(repr=False) 用于定义类的数据结构，不生成默认的 __repr__ 方法
@dataclass(repr=False)
class SymInt(_Union):
    as_expr: SymExpr
    as_int: int

# @dataclass(repr=False) 用于定义类的数据结构，不生成默认的 __repr__ 方法
@dataclass(repr=False)
class SymBool(_Union):
    as_expr: SymExpr
    as_bool: bool

# @dataclass 用于定义类的数据结构
@dataclass
class TensorMeta:
    dtype: ScalarType
    sizes: List[SymInt]
    requires_grad: bool
    device: Device
    strides: List[SymInt]
    storage_offset: SymInt
    layout: Layout

# @dataclass(repr=False) 用于定义类的数据结构，不生成默认的 __repr__ 方法
@dataclass(repr=False)
class SymIntArgument(_Union):
    as_name: str
    as_int: int

# @dataclass(repr=False) 用于定义类的数据结构，不生成默认的 __repr__ 方法
@dataclass(repr=False)
class SymBoolArgument(_Union):
    as_name: str
    as_bool: bool

# @dataclass 用于定义类的数据结构
@dataclass
class TensorArgument:
    name: str

# @dataclass 用于定义类的数据结构
@dataclass
class TokenArgument:
    name: str

# This is use for storing the contents of a list which contain optional tensors
# 用于存储包含可选张量的列表的内容
# (Tensor?[], ex. [Tensor, None, ...]), where the list will be serialized to the
# type List[OptionalTensorArgument], with tensor values seiralized to the
# "as_tensor" field, and None values serialized to the "as_none" field.
@dataclass(repr=False)
class OptionalTensorArgument(_Union):
    # Represents an optional tensor argument which can either be a tensor or None
    as_tensor: TensorArgument  # Represents a tensor argument
    as_none: Tuple[()]  # Represents None as an argument


@dataclass
class GraphArgument:
    name: str  # Name of the graph argument
    graph: 'Graph'  # Reference to a Graph object


@dataclass
class CustomObjArgument:
    name: str  # Name of the custom object argument
    class_fqn: str  # Fully qualified name of the class


# This is actually a union type
@dataclass(repr=False)
class Argument(_Union):
    as_none: Tuple[()]  # Represents None as an argument
    as_tensor: TensorArgument  # Represents a single tensor argument
    as_tensors: List[TensorArgument]  # Represents a list of tensor arguments
    as_int: int  # Represents an integer argument
    as_ints: List[int]  # Represents a list of integer arguments
    as_float: float  # Represents a float argument
    as_floats: List[float]  # Represents a list of float arguments
    as_string: str  # Represents a string argument
    as_strings: List[str]  # Represents a list of string arguments
    as_sym_int: SymIntArgument  # Represents a symbolic integer argument
    as_sym_ints: List[SymIntArgument]  # Represents a list of symbolic integer arguments
    as_scalar_type: ScalarType  # Represents a scalar type argument
    as_memory_format: MemoryFormat  # Represents a memory format argument
    as_layout: Layout  # Represents a layout argument
    as_device: Device  # Represents a device argument
    as_bool: bool  # Represents a boolean argument
    as_bools: List[bool]  # Represents a list of boolean arguments
    as_sym_bool: SymBoolArgument  # Represents a symbolic boolean argument
    as_sym_bools: List[SymBoolArgument]  # Represents a list of symbolic boolean arguments
    as_graph: GraphArgument  # Represents a graph argument
    as_optional_tensors: List[OptionalTensorArgument]  # Represents a list of optional tensor arguments
    as_custom_obj: CustomObjArgument  # Represents a custom object argument
    as_operator: str  # Represents an operator string


@dataclass
class NamedArgument:
    # Argument name from the operator schema
    name: str  # Name of the argument
    arg: Argument  # The argument itself


@dataclass
class Node:
    target: str  # Target of the node
    inputs: List[NamedArgument]  # List of named input arguments
    outputs: List[Argument]  # List of output arguments
    metadata: Dict[str, str]  # Metadata associated with the node


@dataclass
class Graph:
    inputs: List[Argument]  # List of input arguments to the graph
    outputs: List[Argument]  # List of output arguments from the graph
    nodes: List[Node]  # List of nodes in the graph
    tensor_values: Dict[str, TensorMeta]  # Dictionary mapping tensor names to TensorMeta objects
    sym_int_values: Dict[str, SymInt]  # Dictionary mapping symbolic integer names to SymInt objects
    sym_bool_values: Dict[str, SymBool]  # Dictionary mapping symbolic boolean names to SymBool objects
    # This is for deserializing the submodule graphs from higher order ops
    # (ex. cond, map) where single tensor returns will just return a single
    # tensor, rather than following export schema and returning a singleton
    # list.
    is_single_tensor_return: bool = False  # Indicates if the graph returns a single tensor
    custom_obj_values: Dict[str, CustomObjArgument] = field(default_factory=dict)  # Dictionary mapping custom object names to CustomObjArgument objects


@dataclass
class UserInputSpec:
    # Actually, only tensors and SymInts are allowed here
    arg: Argument  # Represents an argument allowed in user input


@dataclass(repr=False)
class ConstantValue(_Union):
    as_none: Tuple[()]  # Represents None as a constant value
    as_int: int  # Represents an integer constant value
    as_float: float  # Represents a float constant value
    as_string: str  # Represents a string constant value
    as_bool: bool  # Represents a boolean constant value


@dataclass
class ConstantInputSpec:
    name: str  # Name of the constant input specification
    value: ConstantValue  # Constant value associated with the specification


@dataclass
class InputToParameterSpec:
    arg: TensorArgument  # Tensor argument mapped to a parameter name
    parameter_name: str  # Name of the parameter


@dataclass
class InputToBufferSpec:
    arg: TensorArgument  # Tensor argument mapped to a buffer name
    buffer_name: str  # Name of the buffer
    persistent: bool  # Indicates if the buffer is persistent


@dataclass
class InputToTensorConstantSpec:
    arg: TensorArgument  # Tensor argument mapped to a tensor constant name
    tensor_constant_name: str  # Name of the tensor constant


@dataclass
class InputToCustomObjSpec:
    arg: CustomObjArgument  # Custom object argument mapped to a custom object name
    custom_obj_name: str  # Name of the custom object


@dataclass
class InputTokenSpec:
    arg: TokenArgument  # Token argument specification


@dataclass(repr=False)
class InputSpec(_Union):
    user_input: UserInputSpec  # Represents an input specification from user input
    parameter: InputToParameterSpec  # Represents an input specification mapped to a parameter
    buffer: InputToBufferSpec  # Represents an input specification mapped to a buffer
    tensor_constant: InputToTensorConstantSpec
    custom_obj: InputToCustomObjSpec
    token: InputTokenSpec
    constant_input: ConstantInputSpec
# 使用 `dataclass` 装饰器定义一个用户输出规范类，包含一个 `Argument` 类型的参数 `arg`
@dataclass
class UserOutputSpec:
    arg: Argument


# 使用 `dataclass` 装饰器定义一个损失输出规范类，包含一个 `TensorArgument` 类型的参数 `arg`
@dataclass
class LossOutputSpec:
    arg: TensorArgument


# 使用 `dataclass` 装饰器定义一个缓冲区变异规范类，包含一个 `TensorArgument` 类型的参数 `arg` 和一个字符串类型的参数 `buffer_name`
@dataclass
class BufferMutationSpec:
    arg: TensorArgument
    buffer_name: str


# 使用 `dataclass` 装饰器定义一个梯度到参数规范类，包含一个 `TensorArgument` 类型的参数 `arg` 和一个字符串类型的参数 `parameter_name`
@dataclass
class GradientToParameterSpec:
    arg: TensorArgument
    parameter_name: str


# 使用 `dataclass` 装饰器定义一个梯度到用户输入规范类，包含一个 `TensorArgument` 类型的参数 `arg` 和一个字符串类型的参数 `user_input_name`
@dataclass
class GradientToUserInputSpec:
    arg: TensorArgument
    user_input_name: str


# 使用 `dataclass` 装饰器定义一个用户输入变异规范类，包含一个 `TensorArgument` 类型的参数 `arg` 和一个字符串类型的参数 `user_input_name`
@dataclass
class UserInputMutationSpec:
    arg: TensorArgument
    user_input_name: str


# 使用 `dataclass` 装饰器定义一个输出令牌规范类，包含一个 `TokenArgument` 类型的参数 `arg`
@dataclass
class OutputTokenSpec:
    arg: TokenArgument


# 使用 `_Union` 类型定义一个输出规范类，可以是多个不同类型的输出规范类之一：`UserOutputSpec`, `LossOutputSpec`, `BufferMutationSpec`, `GradientToParameterSpec`, `GradientToUserInputSpec`, `UserInputMutationSpec`, `OutputTokenSpec`
@dataclass(repr=False)
class OutputSpec(_Union):
    user_output: UserOutputSpec
    loss_output: LossOutputSpec
    buffer_mutation: BufferMutationSpec
    gradient_to_parameter: GradientToParameterSpec
    gradient_to_user_input: GradientToUserInputSpec
    user_input_mutation: UserInputMutationSpec
    token: OutputTokenSpec


# 使用 `dataclass` 装饰器定义一个图签名类，包含输入规范和输出规范的列表
@dataclass
class GraphSignature:
    input_specs: List[InputSpec]
    output_specs: List[OutputSpec]


# 使用 `dataclass` 装饰器定义一个范围约束类，包含最小值和最大值两个整型参数
@dataclass
class RangeConstraint:
    min_val: int
    max_val: int


# 使用 `dataclass` 装饰器定义一个模块调用签名类，包含输入参数列表和输出参数列表，以及用于序列化的字符串输入和输出规范
@dataclass
class ModuleCallSignature:
    inputs: List[Argument]
    outputs: List[Argument]
    in_spec: str  # 这些由调用 `pytree.treespec_loads` 进行序列化，通过调用 `pytree.treespec_dumps` 进行反序列化
    out_spec: str


# 使用 `dataclass` 装饰器定义一个模块调用条目类，包含完全限定名和可选的模块调用签名
@dataclass
class ModuleCallEntry:
    fqn: str
    signature: Optional[ModuleCallSignature] = None


# 使用 `dataclass` 装饰器定义一个图模块类，包含图对象、图签名对象和模块调用图的列表
@dataclass
class GraphModule:
    graph: Graph
    signature: GraphSignature
    module_call_graph: List[ModuleCallEntry]
    # 不变性: 每次对模式进行更改时，应更新其中一个版本
    # 主要版本号在每次进行破坏性更改时增加
    # 次要版本号在进行兼容性更改时增加
    schema_version: SchemaVersion


# 使用 `dataclass` 装饰器定义一个模式版本类，包含主要版本号和次要版本号
@dataclass
class SchemaVersion:
    major: int
    minor: int


# 使用 `dataclass` 装饰器定义一个导出的程序类，包含图模块对象、操作集版本字典、范围约束字典、模式版本对象和方言字符串
@dataclass
class ExportedProgram:
    graph_module: GraphModule
    opset_version: Dict[str, int]  # 键是操作集命名空间（例如 aten），值是版本号
    range_constraints: Dict[str, RangeConstraint]
    schema_version: SchemaVersion
    dialect: str
```