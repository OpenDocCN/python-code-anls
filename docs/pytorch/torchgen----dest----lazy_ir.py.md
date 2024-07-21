# `.\pytorch\torchgen\dest\lazy_ir.py`

```py
from __future__ import annotations
# 导入未来版本的注释特性

import itertools
# 导入itertools模块，用于高效循环操作

from abc import ABC
# 从abc模块导入ABC，用于定义抽象基类

from dataclasses import dataclass
# 导入dataclass，用于创建数据类（用于存储数据）

from typing import Any
# 导入Any，表示可以是任意类型的对象

import torchgen.api.dispatcher as dispatcher
# 导入torchgen库中的dispatcher模块

from torchgen.api.lazy import (
    getValueT,
    isValueType,
    LazyArgument,
    LazyIrProperties,
    LazyIrSchema,
    tensorListValueT,
)
# 从torchgen.api.lazy模块导入多个符号，包括getValueT, isValueType, LazyArgument等

from torchgen.api.translate import translate
# 导入translate函数，用于执行翻译操作

from torchgen.api.types import (
    BaseCType,
    Binding,
    deviceT,
    DispatcherSignature,
    kernel_signature,
    NativeSignature,
    OptionalCType,
    VectorCType,
)
# 从torchgen.api.types模块导入多个符号，包括BaseCType, Binding, NativeSignature等

from torchgen.context import method_with_native_function
# 从torchgen.context模块导入method_with_native_function函数

from torchgen.dest.lazy_ts_lowering import ts_lowering_body
# 从torchgen.dest.lazy_ts_lowering模块导入ts_lowering_body函数

from torchgen.model import (
    Argument,
    BackendIndex,
    BackendMetadata,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    NativeFunctionsGroup,
)
# 从torchgen.model模块导入多个符号，包括Argument, FunctionSchema, NativeFunction等

def node_ctor_arg_rvalue_string(arg: LazyArgument) -> str:
    """
    Given a LazyArgument,
    generate a c++ string for materializing an rvalue of that arg for passing into
    a lazy Node constructor.
    """
    # 给定一个LazyArgument对象，
    # 生成一个C++字符串，用于实例化该参数的右值，以便传递给延迟节点构造函数。

    # TODO: Matching on CType seems wrong; should be matching on Type
    # TODO: 匹配CType似乎不正确；应该匹配Type

    if isValueType(arg.lazy_type):
        # 如果LazyArgument的类型是值类型
        if isinstance(arg.lazy_type, BaseCType):
            # 如果lazy_type是BaseCType的实例
            if arg.is_wrapped_scalar:
                return f"node_{arg.name}"
                # 返回包装标量的节点名称
            elif arg.lazy_type.type is tensorListValueT:
                return f"lazy_{arg.name}_tensorlist"
                # 返回张量列表的惰性名称
            elif arg.is_symint_or_list:
                return f"GetSymIntValue({arg.name})"
                # 返回符号整数值的名称
            return f"lazy_{arg.name}->GetIrValue()"
            # 返回惰性参数的IR值
        elif isinstance(arg.lazy_type, OptionalCType):
            # 如果lazy_type是OptionalCType的实例
            if arg.is_symint_or_list:
                # 如果是符号整数或列表
                # TODO: I don't understand when you should put lazy_ in the name
                # or not
                return f"{arg.name} ? std::make_optional(GetSymIntValue(*{arg.name})) : ::std::nullopt"
                # 返回可选符号整数值或空值
            elif arg.is_wrapped_scalar:
                return f"node_{arg.name}"
                # 返回包装标量的节点名称
            return (
                f"lazy_{arg.name} ? "
                f"std::make_optional(lazy_{arg.name}->GetIrValue()) : "
                "::std::nullopt"
            )
            # 返回惰性参数的IR值或空值
        else:
            raise AssertionError(
                f"TODO not sure if there are other valid types to handle here ({arg.lazy_type})"
            )
            # 抛出断言错误，提示还有其他有效类型需要处理
    else:
        # 如果参数的原始类型是列表类型并且元素类型是 SymInt
        if isinstance(arg.orig_type, ListType) and arg.orig_type.elem == BaseType(
            BaseTy.SymInt
        ):
            # 如果参数是符号整数类型，返回获取符号整数数组引用的表达式
            if arg.symint:
                return f"GetSymIntArrayRefValue({arg.name})"
            else:
                # 否则返回将数组转换为 std::vector<int64_t> 的表达式
                return f"std::vector<int64_t>({arg.name}.begin(), {arg.name}.end())"
        # 如果参数的惰性类型是向量类型，并且元素类型是基本类型
        elif isinstance(arg.lazy_type, VectorCType) and isinstance(
            arg.lazy_type.elem, BaseCType
        ):
            # 返回将数组转换为指定类型的 std::vector 表达式
            return f"std::vector<{arg.lazy_type.elem.type}>({arg.name}.begin(), {arg.name}.end())"
        # 如果参数的惰性类型是可选类型，并且元素类型是向量类型，并且向量元素是基本类型
        elif (
            isinstance(arg.lazy_type, OptionalCType)
            and isinstance(arg.lazy_type.elem, VectorCType)
            and isinstance(arg.lazy_type.elem.elem, BaseCType)
        ):
            # 返回将参数转换为 Torch 的可选向量类型的表达式
            return f"torch::lazy::ToOptionalVector<{arg.lazy_type.elem.elem.type}>({arg.name})"
        else:
            # 默认情况下返回参数的名称
            return f"{arg.name}"
# 根据 LazyIrSchema 生成节点构造函数的参数字符串
def node_ctor_inputs(schema: LazyIrSchema) -> str:
    # 使用 schema.filtered_args() 迭代生成节点构造函数参数的右值字符串列表
    node_ctor_values = [
        node_ctor_arg_rvalue_string(arg) for arg in schema.filtered_args()
    ]
    # 将生成的参数字符串列表连接成一个逗号分隔的字符串返回
    return ", ".join(node_ctor_values)


# 根据给定的 schema、sig 和 overload_name 生成回退代码
def gen_fallback_code(
    schema: LazyIrSchema,
    sig: DispatcherSignature | NativeSignature,
    overload_name: str,
) -> str:
    # 从 schema.func 中获取 DispatcherSignature
    dispatcher_sig = DispatcherSignature.from_schema(schema.func)
    # 将 sig.arguments() 和 dispatcher_sig.arguments() 转换成表达式列表
    exprs = translate(sig.arguments(), dispatcher_sig.arguments())
    # 将表达式列表中的表达式连接成多行字符串，每行缩进以提高可读性
    fallback_args = ",\n                ".join([a.expr for a in exprs])
    # 根据 overload_name 是否为空生成对应的 ATEN 操作字符串
    if len(overload_name):
        aten_op_str = f"ATEN_OP2({schema.aten_name}, {overload_name})"
    else:
        aten_op_str = f"ATEN_OP({schema.aten_name})"
    # 返回生成的条件回退代码块，其中包含了调用特定函数和参数的语句
    return f"""
        if (force_eager_fallback({aten_symbol(schema)})) {{
            return at::native::call_fallback_fn_symint<&ltc_eager_fallback, {aten_op_str}>::call(
                {fallback_args}
            );
        }}
"""


# 根据 LazyIrSchema 返回对应的 ATEN 符号字符串
def aten_symbol(schema: LazyIrSchema) -> str:
    missing_interned_strings = {
        "sigmoid_backward",
    }
    # 如果 schema.aten_name 在 missing_interned_strings 中，则返回特定格式的 ATEN 符号字符串
    if schema.aten_name in missing_interned_strings:
        return f'c10::Symbol::fromQualString("aten::{schema.aten_name}")'
    # 如果 schema.aten_name 不以 "at::" 开头，则返回带前缀 "at::aten::" 的 ATEN 符号字符串
    if not schema.aten_name.startswith("at::"):
        return f"at::aten::{schema.aten_name}"
    else:
        return schema.aten_name


# 将所有类似张量的参数转换为元张量，并返回两部分结果：
# (1) 包含所有转换逻辑的字符串
# (2) 用于 translate() 的上下文，包含所有相关绑定
def convert_to_meta_tensors(sig: DispatcherSignature) -> tuple[str, list[Binding]]:
    context: list[Binding] = []
    unwrapped_tensor_args: list[str] = []
    # 遍历 sig.arguments() 中的参数，如果是张量类型，则生成对应的元张量转换语句
    for arg in sig.arguments():
        if isinstance(arg.argument, Argument) and arg.argument.type.is_tensor_like():
            unwrapped_name = f"{arg.name}_meta"
            unwrapped_tensor_args.append(
                f"auto {unwrapped_name} = to_meta({arg.name});"
            )
            context.append(arg.with_name(unwrapped_name))
        else:
            context.append(arg)
    # 将生成的元张量转换语句用换行符连接成多行字符串返回
    unwrap_tensor_args_str = "\n        ".join(unwrapped_tensor_args)
    return unwrap_tensor_args_str, context


# 数据类，用于表示生成的懒惰 IR 的抽象基类，冻结其所有字段
@dataclass(frozen=True)
class GenLazyIR(ABC):
    backend_index: BackendIndex
    backend_name: str
    node_base: str
    use_lazy_shape: bool
    # 定义一个方法 __call__，用于处理传入的函数对象或函数组对象，并返回字符串列表
    def __call__(self, f: NativeFunctionsGroup | NativeFunction) -> list[str]:
        # 根据传入对象的类型选择对应的函数对象
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        # 获取函数对象的元数据，使用后端索引获取内核信息
        metadata = self.backend_index.get_kernel(
            f.functional if isinstance(f, NativeFunctionsGroup) else f
        )
        # 创建 LazyIrSchema 对象，用于惰性中间表示的模式
        schema = LazyIrSchema(
            func, symint=metadata is not None and metadata.supports_symint()
        )
        # 根据生成的模式生成代码并返回
        return self.gen(schema)

    # 如果没有子类化这个 IR 基类并作为特定后端节点实现，不会生成任何降级功能
    def lowering_function(self, schema: LazyIrSchema) -> str:
        # 返回空字符串，表示没有实现降级功能
        return ""

    # 创建函数的方法，返回空字符串
    def create_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        # 返回空字符串，表示未实现创建函数的功能
        return ""

    # 判断函数是否可以重用的方法，返回格式化的字符串
    def can_be_reused_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        # 返回格式化字符串，表明不支持重用
        return f"""bool CanBeReused({node_ctor_args}) const {{
    return false;
    }}"""
    def node_base_ctor_call(self, schema: LazyIrSchema) -> str:
        # 获取所有需要作为值传递给基类构造函数的参数
        value_args = schema.filtered_args(values=True, scalars=False)
        
        # 构建基类构造函数调用时传递的参数列表
        base_ctor_value_args_list = []
        for arg in value_args:
            if isinstance(arg.lazy_type, (BaseCType, VectorCType)):
                base_ctor_value_args_list.append(f"{arg.name}")
            elif isinstance(arg.lazy_type, OptionalCType):
                base_ctor_value_args_list.append(f"{arg.name}.value_or(kNullValue)")
            else:
                raise AssertionError(
                    f"Unsupported type ({arg.lazy_type}) - add support if necessary"
                )
        base_ctor_value_args = ", ".join(base_ctor_value_args_list)

        # 获取所有作为标量传递的参数
        scalar_args = schema.filtered_args(values=False, scalars=True)

        # 形状构造
        # 根据指定的形状属性条件性地构建形状
        if schema.properties.ShapePrecompute:
            shape_ctor_arg = "std::move(shapes),"
        elif schema.properties.ShapeCompute:
            shape_args = [a.name for a in value_args]
            shape_args.extend(a.name for a in scalar_args)
            shape_ctor_arg = f"compute_shape_{schema.name}({', '.join(shape_args)}),"
        elif schema.properties.ShapeCache:
            shape_args = [f"operand({i})" for i in range(len(value_args))]
            shape_args.extend(a.name for a in scalar_args)
            shape_ctor_arg = f"[&](){{ return compute_shape_{schema.name}({', '.join(shape_args)})[0]; }},"
        else:
            shape_ctor_arg = ""

        # 标量参数的哈希值
        scalar_hashes = ", ".join(f"{a.name}" for a in scalar_args)

        # 构建并返回最终的节点基类构造函数调用字符串
        return f"""{self.node_base}(
              {schema.node_name}::ClassOpKind(),
              OpList{{{base_ctor_value_args}}},
              {shape_ctor_arg}
              /* num_outputs */ {len(schema.returns)},
              torch::lazy::MHash({scalar_hashes}))"""
        # 获取操作类型，如果没有指定则根据 IR 模式推断
        opkind = schema.opkind or aten_symbol(schema)

        # 筛选所有的参数
        all_args = schema.filtered_args()
        # 筛选标量参数
        scalar_args = schema.filtered_args(values=False, scalars=True)

        # 构造函数参数声明
        ctor_args = [f"const {i.lazy_type.cpp_type()}& {i.name}" for i in all_args]
        # 重用构造函数参数字符串
        reuse_ctor_args = ", ".join(ctor_args)
        
        # 如果使用延迟形状且属性中包含 ShapePrecompute
        if self.use_lazy_shape and schema.properties.ShapePrecompute:
            ctor_args.append("std::vector<torch::lazy::Shape>&& shapes")
        # 节点构造函数参数字符串
        node_ctor_args = ", ".join(ctor_args)

        # 标量参数的初始化字符串
        scalar_initializers = ",\n        ".join(
            [
                # 处理特殊情况，将 string_view 映射到 string
                f"{a.name}({a.name}.has_value() ? ::std::make_optional(std::string(*{a.name})) : ::std::nullopt)"
                if a.lazy_type.cpp_type() == "::std::optional<c10::string_view>"
                else f"{a.name}({a.name})"
                for a in scalar_args
            ]
        )
        # 如果有标量初始化器，则添加逗号和换行符
        if len(scalar_initializers):
            scalar_initializers = f",\n        {scalar_initializers}"
        
        # 标量声明字符串
        scalar_decls = "\n  ".join(
            [
                f"std::string {a.name};"
                if a.lazy_type.cpp_type() == "c10::string_view"
                else f"::std::optional<std::string> {a.name};"
                if a.lazy_type.cpp_type() == "::std::optional<c10::string_view>"
                else f"{a.lazy_type.cpp_type()} {a.name};"
                for a in scalar_args
            ]
        )

        # 可选参数的布尔值声明
        optional_values = [
            arg.name
            for arg in schema.filtered_args(values=True, scalars=False)
            if isinstance(arg.lazy_type, OptionalCType)
        ]
        has_optional_decls = "\n  ".join(
            [f"bool has_{value}: 1;" for value in optional_values]
        )
        # 可选参数的布尔值定义
        has_optional_defs = "\n    ".join(
            [f"has_{value} = !!{value};" for value in optional_values]
        )

        # 生成成员变量字符串列表
        members_to_string = []
        for arg in scalar_args:
            if isinstance(arg.lazy_type, OptionalCType):
                value = f"{arg.name}.value()"
                if arg.is_generator:
                    value = '"torch.Generator()"'
                # 处理可选参数是否有值的情况
                members_to_string.append(
                    f"""if ({arg.name}.has_value()) {{
      ss << ", {arg.name}=" << {value};
    }} else {{
      ss << ", {arg.name}=null";
    }}"""
                )
            else:
                members_to_string.append(f'ss << ", {arg.name}=" << {arg.name};')
        # 生成成员变量字符串
        members_to_string_str = "\n    ".join(members_to_string)

        # 返回生成的字符串列表
        return [
            f"""\
class {schema.node_name} : public {self.node_base} {{
 public:
  // 返回该类的操作类型
  static torch::lazy::OpKind ClassOpKind() {{
    return torch::lazy::OpKind({opkind});
  }}

  // 构造函数，初始化节点并调用基类构造函数
  {schema.node_name}({node_ctor_args})
      : {self.node_base_ctor_call(schema)}{scalar_initializers}
  {{
    // 可选成员的定义
    {has_optional_defs}
  }}

  // 返回节点的字符串表示，重载自基类
  std::string ToString() const override {{
    std::stringstream ss;
    ss << {self.node_base}::ToString();
    // 将成员变量转换为字符串并添加到流中
    {members_to_string_str}
    return ss.str();
  }}

  // 创建函数的定义
  {self.create_function(schema, reuse_ctor_args)}

  // 是否可以重用函数的定义
  {self.can_be_reused_function(schema, reuse_ctor_args)}

  // 降低函数的定义
  {self.lowering_function(schema)}

  // 标量声明
  {scalar_decls}
  // 可选声明
  {has_optional_decls}

}};
    backend_index: BackendIndex
    tensor_class: str
    gen_forced_fallback_code: bool
    backend_namespace: str
    get_tensorlist: str
    get_tensor_or_wrap_number: str
    try_get_tensor: str
    metrics_counter: str
    create_tensor: str
    create_from_first_tensor: bool
    create_aten_from_ltc_tensor: str
    tuple_aten_from_ltc_tensors: str
    lazy_tensor_ptr: str
    get_device_fn: str

    def lazy_tensor_decls(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        # 从 schema 中过滤出需要处理的值参数（非标量）
        value_args = schema.filtered_args(values=True, scalars=False)
        # 生成 LazyTensor 的声明列表，用于包装输入张量
        lazy_tensor_decls: list[str] = []
        for arg in value_args:
            if arg.is_wrapped_scalar:
                if isinstance(arg.lazy_type, OptionalCType):
                    # 如果参数是可选的包装标量类型，生成对应的 LazyTensor 声明
                    lazy_tensor_decls.append(
                        f"""auto node_{arg.name} = {arg.name} ?
                std::make_optional(torch::lazy::LazyGraphExecutor::Get()->
                    GetIrValueForScalarFromCodegen(*{arg.name}, *common_device)):
                ::std::nullopt;"""
                    )
                else:
                    # 否则，直接生成 LazyTensor 声明
                    lazy_tensor_decls.append(
                        f"""auto node_{arg.name} = torch::lazy::LazyGraphExecutor::Get()->
                            GetIrValueForScalarFromCodegen({arg.name}, *common_device);"""
                    )
            elif arg.is_symint_or_list:
                # 如果参数是符号整数或列表，跳过处理（在 isValueType 中提取值）
                continue
            elif isinstance(arg.lazy_type, BaseCType):
                if arg.lazy_type.type is tensorListValueT:
                    # 如果参数类型是 tensorListValueT，生成对应的 LazyTensor 列表声明
                    lazy_tensor_decls.append(
                        f"auto lazy_{arg.name}_tensorlist = "
                        f"{self.backend_namespace}::{self.get_tensorlist}({arg.name});"
                    )
                else:
                    # 否则，生成对应的 LazyTensor 指针声明
                    lazy_tensor_decls.append(
                        f"{self.lazy_tensor_ptr} lazy_{arg.name} = "
                        f"{self.backend_namespace}::{self.get_tensor_or_wrap_number}({arg.name}, *common_device);"
                    )
            elif isinstance(arg.lazy_type, OptionalCType):
                assert arg.lazy_type.elem == BaseCType(getValueT()), arg.lazy_type.elem
                # 如果参数是可选类型，生成 LazyTensor 指针声明，可能需要使用默认值
                lazy_tensor_decls.append(
                    f"{self.lazy_tensor_ptr} lazy_{arg.name} = "
                    f"{self.backend_namespace}::{self.try_get_tensor}({arg.name}.value_or(at::Tensor()));"
                )
            else:
                # 如果遇到未知类型的参数，抛出断言错误
                raise AssertionError(
                    f"TODO not sure if there are other valid types to handle here ({arg.lazy_type})"
                )
        # 返回生成的 LazyTensor 声明代码块
        return ("\n        ").join(lazy_tensor_decls)
    # 强制使用 eager 模式的回退函数，生成回退代码
    def force_eager_fallback(
        self,
        func: NativeFunction,
        schema: LazyIrSchema,
        metadata: BackendMetadata,
        sig: DispatcherSignature | NativeSignature,
    ) -> str:
        # 如果需要生成强制回退代码，则调用生成回退代码函数
        if self.gen_forced_fallback_code:
            return gen_fallback_code(
                schema, sig, overload_name=func.func.name.overload_name
            )
        # 否则返回空字符串
        return ""

    # 生成度量指标的函数
    def metrics(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        # 返回度量计数器的值
        return f"{self.metrics_counter};"

    # 获取设备信息的函数
    def get_device(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        # 获取值参数和标量参数
        value_args = schema.filtered_args(values=True, scalars=False)
        scalar_args = schema.filtered_args(values=False, scalars=True)
        # 获取值参数的名称
        value_types_names = [f"{a.name}" for a in value_args if not a.is_wrapped_scalar]
        # 获取可选设备类型参数的名称
        optional_device = OptionalCType(BaseCType(deviceT))
        optional_devices = [
            a.name for a in scalar_args if a.lazy_type == optional_device
        ]
        # 断言至少存在一个值参数或设备参数
        assert (
            len(value_types_names) > 0 or len(optional_devices) > 0
        ), "Expected at least one Value or Device type"
        # 构建获取设备信息的字符串
        get_device_str = (
            f"{self.get_device_fn}({', '.join(value_types_names + optional_devices)})"
        )
        # 返回获取设备信息的代码块
        return f"""auto common_device = {get_device_str};
        TORCH_INTERNAL_ASSERT(common_device);
        """

    # 形状推断函数
    def shape_inference(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        # 获取后端索引中的内核元数据
        metadata = self.backend_index.get_kernel(func)
        assert metadata is not None
        # 获取所有参数
        all_args = schema.filtered_args()
        returns_length = len(schema.returns)
        # 调用元内核（如果存在）来计算我们的 IR 的输出形状/数据类型
        # 如果操作符是结构化的或者是 view_copy 操作符，则使用元张量进行形状推断
        is_view_copy_op = "view_copy" in func.tags
        is_structured = func.structured or func.structured_delegate is not None
        if is_structured or is_view_copy_op:
            # 返回元输出的代码块
            meta_out = """
// 如果返回的形状数量大于1
if (returns_length > 1) {
    // 定义一个函数，返回指定索引的形状字符串表示
    auto this_shape = [](int i) -> std::string {
        return f"torch::lazy::Shape(std::get<{i}>(out_meta).scalar_type(), std::get<{i}>(out_meta).sizes().vec())";
    };
    // 构建形状字符串列表
    std::string shapes_str = std::string::join(",", {this_shape(i) for i in range(returns_length)});
    // 生成最终的形状数据结构声明
    std::string meta_out = "std::vector<torch::lazy::Shape> shapes{" + shapes_str + "};"
}

// 从函数签名获取调度器签名
auto dispatcher_sig = DispatcherSignature::from_schema(func.func);
// 将输入张量转换为元数据设备并调用
auto [meta_conversion_str, meta_call_ctx] = convert_to_meta_tensors(dispatcher_sig);
// 构建元数据调用的参数列表
auto meta_call_args = {e.expr for e in translate(meta_call_ctx, dispatcher_sig.arguments(), method=False)};

// 如果是视图复制操作
if (is_view_copy_op) {
    // 断言函数具有复合显式自动微分非功能内核
    TORCH_INTERNAL_ASSERT(func.has_composite_explicit_autograd_non_functional_kernel);
    // 设置调度命名空间为复合显式自动微分非功能
    auto dispatch_ns = "compositeexplicitautogradnonfunctional";
} else {
    // 否则设置调度命名空间为meta
    auto dispatch_ns = "meta";
}
// 获取ATen操作的名称
auto aten_name = schema.aten_name;
// 如果函数具有符号整数并且元数据支持符号整数
if (func.func.has_symint() && metadata.supports_symint()) {
    // 修改ATen操作的名称以包含_symint后缀
    aten_name += "_symint";
}

// 构建形状字符串，包括元数据转换和ATen操作调用
auto shape_str = fmt::format(R"(
    {}
    auto out_meta = at::{dispatch_ns}::{aten_name}({});
    {}
)", meta_conversion_str, std::string::join(", ", meta_call_args), meta_out);

// 如果返回的形状数量为1
else {
    // 计算形状签名
    auto shape_sig = ComputeShapeSignature(metadata.kernel, func, symint=metadata.supports_symint());
    // 生成形状数据结构声明
    auto shape_str = fmt::format(R"(
        auto shapes = {};
    )", shape_sig.shape_call);
}

// 添加断言以确保返回的形状数量与期望的数量一致
shape_str += fmt::format(R"(
    TORCH_INTERNAL_ASSERT(shapes.size() == {});
)", returns_length);

// 计算哪些维度是符号化的
auto func_schema_str = "aten::" + str(func.func);
shape_str += fmt::format(R"(
    if(torch::lazy::symbolicShapeEnabled()) {{
        std::vector<torch::jit::IValue> inputs = {{ {} }};
        const char* schema_str = "{}";
        applySymbolicShapesOnLT(schema_str, inputs, shapes);
    }}
)", std::string::join(", ", {str(a.name) for a in all_args}), func_schema_str);

// 返回最终的形状字符串
return shape_str;
    def build_ir_node(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        node_ctor_input_str = node_ctor_inputs(schema)
        return f"""torch::lazy::NodePtr node = torch::lazy::ReuseNode<{schema.node_name}>({node_ctor_input_str});
        if (!node) {{
            {self.shape_inference(func, schema)}
            node = torch::lazy::MakeNode<{schema.node_name}>({node_ctor_input_str}, std::move(shapes));
            CacheNode(node);
        }}
        """
# 构建惰性 IR 节点。如果节点不存在，则根据给定的函数和模式进行形状推断，并创建新的节点。

    def create_lazy_tensor(self, first_tensor_name: str | None = None) -> str:
        # xla uses an instance method for tensor creation, for the time being
        if self.create_from_first_tensor:
            # TODO(whc) remove this if XLA switches to using static method for creation
            assert (
                first_tensor_name is not None
            ), "Requires first tensor to create lazy tensor"
            return f"{first_tensor_name}.{self.create_tensor}"
        return f"{self.backend_namespace}::{self.create_tensor}"
# 创建惰性张量。根据是否从第一个张量创建，返回相应的张量创建方法字符串。

    def return_aten_tensor(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        returns_length = len(schema.returns)
        value_args = schema.filtered_args(values=True, scalars=False)
        value_types_names = [f"{a.name}" for a in value_args if not a.is_wrapped_scalar]
        first_tensor_name = value_types_names[0] if len(value_types_names) > 0 else None
        bridge_str = f"""auto result = {self.create_aten_from_ltc_tensor}(
                {self.create_lazy_tensor(first_tensor_name)}(std::move(node), *common_device));"""
# 返回 ATen 张量。根据返回值数量，生成对应的桥接字符串，用于转换惰性张量为 ATen 张量。

        if returns_length > 1:
            assert (
                len(value_types_names) > 0
            ), "Code below assumes there is at least one tensor arg"
            bridge_str = f"""std::vector<{self.lazy_tensor_ptr}> lazy_tensors;
        for (int i = 0; i < {returns_length}; i++) {{
            lazy_tensors.push_back({self.create_lazy_tensor(first_tensor_name)}({getValueT()}(node, i), *common_device));
        }}
        auto result = {self.tuple_aten_from_ltc_tensors}<{returns_length}>(lazy_tensors);"""
# 如果返回值数量大于 1，则生成多个惰性张量并转换为 ATen 张量的桥接字符串。

        if schema.name.name.inplace or func.func.is_out_fn():
            assert returns_length == 1, (
                "We assumed there was no such case where an op is an in-place variant "
                f"and has tuple outputs, but got tuple of len {returns_length}."
            )
            bridge_str = f"""lazy_{first_tensor_name}->SetInPlaceIrValue(node);
        auto& result = {first_tensor_name};"""
# 如果函数是就地操作或者有输出函数，则设置就地 IR 值，并返回相应结果。

        bridge_str += """
        return result;"""
        return bridge_str
# 返回桥接字符串最终结果。

    @method_with_native_function
    def __call__(self, func: NativeFunction) -> list[str]:
        sig = kernel_signature(func, self.backend_index)
        metadata = self.backend_index.get_kernel(func)
        assert metadata is not None
        schema = LazyIrSchema(func.func, symint=metadata.supports_symint())
        return [
            f"""\
    # 构建包含类和方法信息的声明字符串，使用指定的类方法名和内核名称格式化字符串
    {sig.decl(name=f"{self.class_method_name}::{metadata.kernel}")} {{
        # 强制使用快速回退的方法，传入函数、模式、元数据和签名对象
        {self.force_eager_fallback(func, schema, metadata, sig)}
        # 计算函数的指标信息，传入函数和模式
        {self.metrics(func, schema)}
        # 获取函数的计算设备，传入函数和模式
        {self.get_device(func, schema)}
        # 延迟张量声明，传入函数和模式
        {self.lazy_tensor_decls(func, schema)}
        # 构建中间表示节点，传入函数和模式
        {self.build_ir_node(func, schema)}
        # 返回 ATen 张量，传入函数和模式
        {self.return_aten_tensor(func, schema)}
    }}\n
    """
class ComputeShapeSignature:
    """
    Here we use the base name as the suffix of the signature to avoid generating for in-place variants.
    """

    def __init__(self, kernel_name: str, f: NativeFunction, *, symint: bool) -> None:
        # Initialize with lazy intermediate representation (IR) schema for the function
        self.__schema = LazyIrSchema(f.func, symint=symint)
        # Generate dispatch arguments string for the function
        self.__dispatch_args = ", ".join(
            [a.decl() for a in dispatcher.arguments(f.func, symint=symint)]
        )
        # Generate call arguments string from filtered schema arguments
        self.__call_args = ", ".join(
            [f"{arg.name}" for arg in self.__schema.filtered_args(generator=True)]
        )
        # Store the kernel name associated with this signature
        self.__kernel_name = kernel_name

    def __decl_suffix(self) -> str:
        # Return the suffix string for the function declaration based on kernel name and dispatch arguments
        return f"{self.__kernel_name}({self.__dispatch_args})"

    def __call_suffix(self) -> str:
        # Return the suffix string for the function call based on kernel name and call arguments
        return f"{self.__kernel_name}({self.__call_args})"

    @property
    def shape_decl(self) -> str:
        # Return the declaration string for the shape computation function
        return f"TORCH_API std::vector<torch::lazy::Shape> compute_shape_{self.__decl_suffix()}"

    @property
    def shape_call(self) -> str:
        # Return the call string for invoking the shape computation function
        return f"torch::lazy::compute_shape_{self.__call_suffix()}"


@dataclass(frozen=True)
class GenLazyShapeInferenceDefinition:
    backend_index: BackendIndex
    tensor_class: str

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> list[str]:
        # Retrieve kernel metadata for the given function
        metadata = self.backend_index.get_kernel(f)
        assert metadata is not None

        # Check if the function is a view_copy operation
        is_view_copy_op = "view_copy" in f.tags
        # Check if the function is structured or has a structured delegate
        is_structured = f.structured or f.structured_delegate is not None

        if is_structured or is_view_copy_op:
            # If function is structured or a view_copy operation, return an empty list
            return []
        else:
            # Otherwise, create ComputeShapeSignature instance for shape computation
            shape_sig = ComputeShapeSignature(
                metadata.kernel, f, symint=metadata.supports_symint()
            )
            # Return a list containing the shape declaration string
            return ["\n".join([f"{shape_sig.shape_decl};"])]


def generate_non_native_lazy_ir_nodes(
    non_native: list[dict[str, Any]], gen_lazy_ir: GenLazyIR
) -> list[str]:
    """Generate the non-native lazy IR node classes"""
    nodes = []
    for op in non_native:
        # Set default properties for Non-Native IRs
        properties = LazyIrProperties("ShapeCache", "CanBeReused", "LowerDeclOnly")
        for p in op.get("properties", []):
            setattr(properties, p, True)

        # Parse function schema from the operation dictionary
        schema = LazyIrSchema(FunctionSchema.parse(op["func"]), properties, symint=True)
        # Assign optional operation kind to the schema
        schema.opkind = op.get("opkind")
        # Generate lazy IR node and append to nodes list
        nodes.append(gen_lazy_ir.gen(schema)[0])

    return nodes
```