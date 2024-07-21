# `.\pytorch\torchgen\api\lazy.py`

```py
# 导入必要的模块和类
from __future__ import annotations
from typing import Any
from torchgen.api.types import (
    BaseCppType,
    BaseCType,
    boolT,
    CType,
    deviceT,
    doubleT,
    generatorT,
    layoutT,
    ListCType,
    longT,
    memoryFormatT,
    NamedCType,
    OptionalCType,
    scalarT,
    scalarTypeT,
    stringT,
    SymIntT,
    VectorCType,
)
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    OperatorName,
    OptionalType,
    Return,
    TensorOptionsArguments,
    Type,
)

# 全局变量，用于存储 IR 类型
_valueT: BaseCppType | None = None


# 定义函数 getValueT，用于获取当前的 IR 类型
# IR 类型表示计算张量的内部表示
def getValueT() -> BaseCppType:
    global _valueT
    if not _valueT:
        # 如果 _valueT 未设置，则抛出未实现的错误
        raise NotImplementedError(
            "The value type needs to be set with setValueT() in run_gen_lazy_tensor()"
        )

    return _valueT


# 定义函数 setValueT，用于设置当前的 IR 类型
# 参数 val 表示要设置的 IR 类型
def setValueT(val: BaseCppType) -> None:
    global _valueT
    _valueT = val


# 定义一个特定的 BaseCppType 对象 tensorListValueT，表示 torch::lazy 的 Value 类型
# 用于表示张量的懒惰求值特性
tensorListValueT = BaseCppType("torch::lazy", "Value")


# 定义函数 process_ir_type，将 NativeFunctions 中的类型转换为 lazy tensor 代码生成所需的类型
# 参数 typ 表示要转换的类型，properties 表示懒惰 IR 属性，symint 表示是否为符号整数类型
# 返回值为转换后的类型，可以是 BaseCType、VectorCType、OptionalCType 或 ListCType
def process_ir_type(
    typ: Type, properties: LazyIrProperties, *, symint: bool
) -> BaseCType | VectorCType | OptionalCType | ListCType:
    """
    This function takes a type from NativeFunctions and converts it for use with
    lazy tensor codegen.

    Type conversion for lazy currently consists of
     (1) changing at::Tensors into lazy::Values
     (2) wrapping everything in a BaseCType
     (3) making cpp-reference types into cpp-value types (e.g. vector instead of IntArrayRef)

    (1) converts at::Tensors to lazy::Values (which wrap lazy::Nodes, with which Lazy IR represents tensors.)
    There is special handling for Optional[Tensor] or List[Tensor], etc- hence 'tensor-like'

    This is incomplete- there are assertions in places that it's expected to need to add
    more types as the codegen is used with more operators.
    """
    # 检查 typ 是否为 BaseType 类型的实例
    if isinstance(typ, BaseType):
        # 如果 typ 的名称为 BaseTy.Tensor
        if typ.name == BaseTy.Tensor:
            # 返回一个 BaseCType 对象，该对象使用 getValueT() 返回的值
            return BaseCType(getValueT())
        # 如果 typ 的名称为 BaseTy.Scalar
        elif typ.name == BaseTy.Scalar:
            # 如果 properties.TreatScalarsAsConstants 为真
            if properties.TreatScalarsAsConstants:
                # 返回一个 BaseCType 对象，该对象使用 scalarT 的值
                return BaseCType(scalarT)
            # 如果 properties.TreatScalarsAsConstants 为假
            else:
                # 返回一个 BaseCType 对象，该对象使用 getValueT() 返回的值
                return BaseCType(getValueT())
        # 如果 typ 的名称为 BaseTy.ScalarType
        elif typ.name == BaseTy.ScalarType:
            # 返回一个 BaseCType 对象，该对象使用 scalarTypeT 的值
            return BaseCType(scalarTypeT)
        # 如果 typ 的名称为 BaseTy.int
        elif typ.name == BaseTy.int:
            # 返回一个 BaseCType 对象，该对象使用 longT 的值
            return BaseCType(longT)
        # 如果 typ 的名称为 BaseTy.SymInt
        elif typ.name == BaseTy.SymInt:
            # 如果 symint 为真
            if symint:
                # 返回一个 BaseCType 对象，该对象使用 getValueT() 返回的值
                return BaseCType(getValueT())
            # 如果 symint 为假
            else:
                # 返回一个 BaseCType 对象，该对象使用 longT 的值
                return BaseCType(longT)
        # 如果 typ 的名称为 BaseTy.bool
        elif typ.name == BaseTy.bool:
            # 返回一个 BaseCType 对象，该对象使用 boolT 的值
            return BaseCType(boolT)
        # 如果 typ 的名称为 BaseTy.float
        elif typ.name == BaseTy.float:
            # 返回一个 BaseCType 对象，该对象使用 doubleT 的值
            return BaseCType(doubleT)
        # 如果 typ 的名称为 BaseTy.str
        elif typ.name == BaseTy.str:
            # 返回一个 BaseCType 对象，该对象使用 stringT 的值
            return BaseCType(stringT)
        # 如果 typ 的名称为 BaseTy.Device
        elif typ.name == BaseTy.Device:
            # 返回一个 BaseCType 对象，该对象使用 deviceT 的值
            return BaseCType(deviceT)
        # 如果 typ 的名称为 BaseTy.Generator
        elif typ.name == BaseTy.Generator:
            # 返回一个 BaseCType 对象，该对象使用 generatorT 的值
            return BaseCType(generatorT)
        # 如果 typ 的名称为 BaseTy.Layout
        elif typ.name == BaseTy.Layout:
            # 返回一个 BaseCType 对象，该对象使用 layoutT 的值
            return BaseCType(layoutT)
        # 如果 typ 的名称为 BaseTy.MemoryFormat
        elif typ.name == BaseTy.MemoryFormat:
            # 返回一个 BaseCType 对象，该对象使用 memoryFormatT 的值
            return BaseCType(memoryFormatT)
        # 如果 typ 的名称未匹配到以上任何情况
        else:
            # 抛出 AssertionError，提示需要添加对该类型的支持
            raise AssertionError(f"TODO add support for type {repr(typ)}")
    # 如果 typ 是 OptionalType 类型的实例
    elif isinstance(typ, OptionalType):
        # 递归处理 OptionalType 类型中的元素类型，返回其对应的 OptionalCType 对象
        return OptionalCType(process_ir_type(typ.elem, properties, symint=symint))
    # 如果 typ 是 ListType 类型的实例
    elif isinstance(typ, ListType):
        # 如果 typ 的元素类型的字符串表示为 "Tensor?"
        if str(typ.elem) == "Tensor?":
            # 返回一个 ListCType 对象，该对象包含一个 OptionalCType 对象，
            # 该对象又包含一个 BaseCType 对象，使用 getValueT() 返回的值
            return ListCType(OptionalCType(BaseCType(getValueT())))
        # 如果 typ 的元素类型的字符串表示为 "Tensor"
        elif str(typ.elem) == "Tensor":
            # 返回一个 BaseCType 对象，该对象使用 tensorListValueT 的值
            # 这是从 GetTensorList 作为 Value 输入的 TensorList
            return BaseCType(tensorListValueT)
        # 如果 typ 的元素类型为 BaseType(BaseTy.SymInt)
        elif typ.elem == BaseType(BaseTy.SymInt):
            # 返回一个 VectorCType 对象，该对象包含一个 BaseCType 对象，使用 longT 的值
            # TODO: 需要返回一个值类型。这里的问题类似于 tensorListValueT 的问题：
            # 如果有 SymInt[]，不能方便地直接保存 Value 的列表，因为节点期望将所有参数保存为向量。
            # 因此，需要一个单独的 IR 节点来表示所有大小节点组装成的列表。我不是 LTC 的开发人员，所以现在不想搞清楚。
            # 你们自己搞定吧……
            return VectorCType(BaseCType(longT))
        # 如果 typ 的元素类型未匹配到以上任何情况
        else:
            # 递归处理 ListType 类型中的元素类型，返回其对应的 VectorCType 对象
            return VectorCType(process_ir_type(typ.elem, properties, symint=symint))
    # 如果 typ 类型未被识别
    else:
        # 抛出 AssertionError，提示类型未被识别
        raise AssertionError(f"unrecognized type {repr(typ)}")
# TODO: 根据 CType 确定此值不好；应直接从 Type 计算；然后可以使用与 process_ir_type 相同的逻辑
# 
# 不变条件：传递的 typ 应为 *拥有* 的 CType（例如，我们会报告 ArrayRef<Value> 不是值类型）
def isValueType(typ: CType, properties: LazyIrProperties | None = None) -> bool:
    """
    给定一个类型，确定它是否类似于值类型。这相当于 Tensor 类型，但假定类型已经转换过。
    """
    if isinstance(typ, BaseCType):
        # 我对自己的命名约定感到后悔，但现在我们在 lazy value 中包装 at::scalar，
        # 同时在 IR 中将其他 'scalar' 类型保留为标量
        treat_scalars_as_constants = properties and properties.TreatScalarsAsConstants
        return (
            typ.type == getValueT()
            or (typ.type == scalarT and not treat_scalars_as_constants)
            or typ.type == SymIntT
        )
    elif typ == VectorCType(BaseCType(SymIntT)):
        # TODO: 为此报告 True
        return False
    elif isinstance(typ, (OptionalCType, ListCType, VectorCType)):
        return isValueType(typ.elem, properties)
    return False


def isSymIntType(typ: Type) -> bool:
    return isinstance(typ, BaseType) and typ.name == BaseTy.SymInt


def isWrappedScalarType(typ: Type) -> bool:
    """
    给定一个类型，确定它是否为我们将在 lazy Value 中包装的 c10::scalar。
    由于我们从 scalarT 直接更改类型为 valueT，信息会丢失。
    此函数有助于构建包含包装标量的列表，以保存该信息
    """
    if isinstance(typ, BaseType):
        # 我对自己的命名约定感到后悔，但现在我们在 lazy value 中包装 at::scalar，
        # 同时在 IR 中将其他 'scalar' 类型保留为标量
        return typ.name == BaseTy.Scalar
    elif isinstance(typ, (OptionalType, ListType)):
        return isWrappedScalarType(typ.elem)
    return False


# TODO: 与 Type.is_generator_like 合并
def isGeneratorType(typ: Type) -> bool:
    if isinstance(typ, BaseType):
        return typ.name == BaseTy.Generator
    elif isinstance(typ, (OptionalType)):
        return isGeneratorType(typ.elem)
    return False


# 此类缓存从 Argument 和 LazyIrProperties 计算得出的几个衍生属性
class LazyArgument:
    name: str
    orig_type: Type
    lazy_type_: CType | None
    is_wrapped_scalar: bool
    is_generator: bool
    # TODO: 这是错误的，对于 symint 列表是 false
    is_symint_or_list: bool

    # 是否将此视为 symint 或非 symint
    symint: bool

    # 如果此参数是或包含 lazy IR 值，则为 true
    is_lazy_value: bool

    def __init__(
        self, arg: Argument, properties: LazyIrProperties, *, symint: bool
    ):
    ) -> None:
        self.name = arg.name
        self.orig_type = arg.type
        self.symint = symint
        self.is_optional = isinstance(arg.type, OptionalType)
        self.is_generator = isGeneratorType(arg.type)
        self.lazy_type_ = process_ir_type(arg.type, properties, symint=symint)
        self.is_wrapped_scalar = isWrappedScalarType(arg.type)
        self.is_symint_or_list = symint and (
            isSymIntType(arg.type)
            or (isinstance(arg.type, OptionalType) and isSymIntType(arg.type.elem))
            # TODO: lists of symints are not currently treated as value types
            # or (isinstance(arg.type, ListType) and isSymIntType(arg.type.elem))
        )
        # 判断是否为延迟加载的值类型
        self.is_lazy_value = isValueType(self.lazy_type, properties)

    @property
    def lazy_type(self) -> CType:
        # 断言确保延迟类型不为 None
        assert (
            self.lazy_type_ is not None
        ), f"Attempted to access lazy_type for invalid argument {self.name}"
        return self.lazy_type_
class LazyIrProperties:
    """Collection of properties for an IR node

    The property groups are listed below. Each group is mutually
    exclusive, meaning that only one property from each group can be True
    at any one time. The properties can be accessed as if they were normal
    attributes. The mutual exclusivity is automatically handled.
    """

    Properties: tuple[tuple[str, ...], ...] = (
        (
            "ShapePrecompute",  # Assume shape has been precomputed
            "ShapeCompute",  # Need to compute the shape on construction
            "ShapeCache",  # Utilize the shape cache to defer computation
        ),
        (
            "Lower",  # Codegen full lower function
            "LowerDeclOnly",  # Codegen only lower function declaration
        ),
        (
            "CanBeReused",  # Codegen full reuse function
            "CanBeReusedDeclOnly",  # Codegen only reuse function declaration
        ),
        (
            "CreateFn",  # Codegen full create function
            "CreateFnDeclOnly",  # Codegen only create function declaration
        ),
        (
            "TreatScalarsAsConstants",  # Treat Scalars as constants instead of handling like values
        ),
    )

    def __init__(self, *default_properties: str) -> None:
        # Initialize a dictionary to store properties with their respective groups as keys
        properties: dict[tuple[str, ...], str | None] = dict.fromkeys(
            LazyIrProperties.Properties
        )
        self.__dict__["properties"] = properties
        # Set default properties provided during initialization
        for p in default_properties:
            setattr(self, p, True)

    def __getattr__(self, key: str) -> Any:
        # Retrieve properties dictionary
        properties = self.__dict__["properties"]
        # Check each property group for the requested key
        for values in LazyIrProperties.Properties:
            if key in values:
                return properties[values] == key

        # If key not found, fallback to default behavior
        return self.__getattribute__(key)

    def __setattr__(self, key: str, value: Any) -> Any:
        # Retrieve properties dictionary
        properties = self.__dict__["properties"]
        # Check each property group for the requested key
        for values in LazyIrProperties.Properties:
            if key in values:
                # Set property in the dictionary based on the value
                properties[values] = key if value else None
                return value

        # Raise error if key not found in any property group
        raise KeyError(f"Invalid property: {key}")


# Inspired by a FunctionSchema object, a LazyIrSchema holds the schema of a Lazy IR node.
# Unlike a FunctionSchema, it has no round-trippable string form (relating to the YAML),
# but carries type information from a native FunctionSchema modified for use with IR nodes,
# and preserving original argument names.
#
# TODO: This is not idiomatic with how other torchgen APIs transform on schema.
class LazyIrSchema:
    # The name of the operator this function schema describes.
    name: OperatorName

    positional_args: tuple[LazyArgument, ...]
    keyword_args: tuple[LazyArgument, ...]

    # TODO: Need to handle collisions with argument names at some point
    returns: tuple[Return, ...]

    # if this schema has a Generator arg, list its orig ctype/name but don't
    # build a LazyArgument since lazy IR doesn't support it
    generator_arg: NamedCType | None = None

# 定义一个变量 `generator_arg`，类型为 `NamedCType` 或 `None`，初始值为 `None`

    # original function schema
    func: FunctionSchema

# 定义一个变量 `func`，类型为 `FunctionSchema`，表示原始函数的架构信息


    # Whether or not we are code-genning for SymInt or not
    symint: bool

# 表示是否为 SymInt 代码生成的布尔值变量 `symint`


    properties: LazyIrProperties = LazyIrProperties(

# 定义一个变量 `properties`，类型为 `LazyIrProperties`，并初始化为默认属性列表

        # default properties
        "ShapePrecompute",
        "Lower",
        "CanBeReused",
    )

# 初始化 `properties` 变量的默认属性，包括 "ShapePrecompute"、"Lower"、"CanBeReused"


    opkind: str | None = None

# 定义一个变量 `opkind`，类型为 `str` 或 `None`，初始值为 `None`


    def __init__(
        self,
        func: FunctionSchema,
        properties: LazyIrProperties | None = None,
        *,
        symint: bool,
    ) -> None:

# 构造函数 `__init__`，接受 `func`（函数架构）、`properties`（惰性 IR 属性）、`symint`（SymInt 标志）

        if properties:
            self.properties = properties

# 如果传入了 `properties`，则使用传入的 `properties` 替换默认属性


        self.func = func
        self.symint = symint

# 将传入的 `func` 和 `symint` 分配给对象的属性


        positional_args: list[LazyArgument] = []

# 初始化一个列表 `positional_args`，用于存储位置参数的惰性参数对象


        for arg_field in ["pre_self_positional", "self_arg", "post_self_positional"]:

# 遍历位置参数的字段：前置自身位置参数、自身参数、后置自身位置参数


            if arg_field == "self_arg" and func.arguments.self_arg is not None:
                arg = func.arguments.self_arg.argument
                positional_args.append(
                    LazyArgument(arg, self.properties, symint=symint)
                )

# 如果当前字段是 `self_arg` 并且 `func.arguments.self_arg` 不为空，则创建对应的惰性参数对象并添加到 `positional_args`


            elif getattr(func.arguments, arg_field) is not None:
                positional_args.extend(
                    LazyArgument(arg, self.properties, symint=symint)
                    for arg in getattr(func.arguments, arg_field)
                )

# 否则，如果字段不为空，则遍历其中的参数列表，创建对应的惰性参数对象并添加到 `positional_args`


        self.positional_args = tuple(positional_args)

# 将所有位置参数的惰性参数对象转换为元组，并赋给对象的 `positional_args` 属性


        keyword_args: list[LazyArgument] = []

# 初始化一个列表 `keyword_args`，用于存储关键字参数的惰性参数对象


        for arg_field in [
            "pre_tensor_options_kwarg_only",
            "tensor_options",
            "post_tensor_options_kwarg_only",
            "out",
        ]:

# 遍历关键字参数的字段：前置张量选项关键字参数、张量选项、后置张量选项关键字参数、输出


            curr_args = getattr(func.arguments, arg_field)
            if curr_args is not None:
                if isinstance(curr_args, TensorOptionsArguments):
                    curr_args = curr_args.all()
                for arg in curr_args:
                    if isGeneratorType(arg.type):
                        assert (
                            self.generator_arg is None
                        ), "We expect there is only one generator arg"
                        self.generator_arg = NamedCType(
                            arg.name, arg.type  # type:ignore[arg-type]
                        )
                keyword_args.extend(
                    LazyArgument(arg, self.properties, symint=symint)
                    for arg in curr_args
                )

# 如果当前字段不为空，则处理其中的参数：
## - 如果是 `TensorOptionsArguments` 类型，则获取所有参数
## - 遍历参数列表，如果参数类型是生成器类型，则确保只有一个生成器参数，并将其赋给 `generator_arg`
## - 创建相应的惰性参数对象并添加到 `keyword_args`


        self.keyword_args = tuple(keyword_args)

# 将所有关键字参数的惰性参数对象转换为元组，并赋给对象的 `keyword_args` 属性


        self.name = func.name
        self.returns = func.returns

# 将函数的名称和返回类型赋给对象的 `name` 和 `returns` 属性


    @property
    def node_name(self) -> str:

# 定义一个属性 `node_name`，返回类型为 `str`

        """
        Return camel-case version of op in node.

        Note: This function also appends any `overload_name` in the operation.
        For example, if the op is `bitwise_and.Tensor`, the returned name
        will be `BitwiseAndTensor`.
        """

# 文档字符串，说明 `node_name` 方法的作用，将操作转换为驼峰命名形式


        op_name = f"{self.name.name}_{self.name.overload_name}".lower()

# 根据函数名称和重载名称创建操作名称，并转换为小写


        return "".join(word.capitalize() or "" for word in op_name.split("_"))

# 将操作名称按下划线拆分，并将每个单词的首字母大写后连接起来作为返回值
    # 返回属性的名称字符串表示形式
    def aten_name(self) -> str:
        return str(self.name.name)

    # 返回基本名称的字符串表示形式
    @property
    def base_name(self) -> str:
        return f"{self.name.name.base}"

    # 返回根据不同过滤条件得到的参数列表
    def filtered_args(
        self,
        positional: bool = True,
        keyword: bool = True,
        values: bool = True,
        scalars: bool = True,
        generator: bool = True,
    ) -> list[LazyArgument]:
        # 该函数维护参数的排序顺序，但提供不同的过滤视图
        # 代码的某些部分关心 kwargs vs args（TS lowerings），
        # 其他部分关心是否需要将参数包装在惰性值中或保持不变
        # 生成器是特殊情况，因为它们对于回退/形状推断是必需的，但不支持 TS lowerings，因此也从惰性 IR 中省略
        args: list[LazyArgument] = []
        if positional:
            args.extend(self.positional_args)
        if keyword:
            args.extend(self.keyword_args)

        if values and scalars and generator:
            return args
        elif values and scalars:
            return [a for a in args if not a.is_generator]
        elif values:
            return [a for a in args if a.is_lazy_value]
        elif scalars:
            return [
                a
                for a in args
                if not a.is_lazy_value and (generator or not a.is_generator)
            ]

        return []

    # 返回位置参数的值列表
    @property
    def positional_values(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=True, keyword=False, values=True, scalars=False
        )

    # 返回位置参数的标量列表
    @property
    def positional_scalars(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=True, keyword=False, values=False, scalars=True
        )

    # 返回关键字参数的值列表
    @property
    def keyword_values(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=False, keyword=True, values=True, scalars=False
        )

    # 返回关键字参数的标量列表
    @property
    def keyword_scalars(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=False, keyword=True, values=False, scalars=True
        )
```