# `.\pytorch\torchgen\static_runtime\generator.py`

```py
from __future__ import annotations

import json
import logging
import math
from typing import Sequence

import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
    Argument,
    BackendIndex,
    BaseTy,
    BaseType,
    FunctionSchema,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    OptionalType,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from torchgen.static_runtime import config

# 获取全局日志记录器
logger: logging.Logger = logging.getLogger()

# 判断参数列表中是否存在别名设置
def has_alias(
    arguments: Sequence[Argument | SelfArgument | TensorOptionsArguments],
) -> bool:
    for arg in arguments:
        annotation = getattr(arg, "annotation", None)
        if not annotation:
            continue
        alias_set = getattr(annotation, "alias_set", ())
        if alias_set:
            return True
    return False

# 定义禁止操作集合
BLOCKED_OPS = frozenset()

# 判断给定的函数组是否被支持
def is_supported(g: NativeFunctionsGroup | NativeFunctionsViewGroup) -> bool:
    base_op_name = ""
    func = None
    if isinstance(g, NativeFunctionsViewGroup):
        base_op_name = g.view.root_name
        func = g.view.func
    else:
        base_op_name = g.out.func.name.name.base
        func = g.out.func
    
    # 检查函数是否为手写实现
    if config.is_hand_written(g):
        logger.info("HAND WRITTEN: %s", base_op_name)
        return False
    
    # 检查函数是否在禁止操作集合中
    if base_op_name in BLOCKED_OPS:
        logger.info("BLOCKED: %s", base_op_name)
        return False
    
    # 检查函数的参数类型是否支持
    for arg in func.schema_order_arguments():
        maybe_method = ivalue_type_conversion_method(arg.type)
        if not maybe_method:
            # 类型转换目前不受支持
            logger.info("NOT SUPPORTED TYPE CONVERTING: %s", func)
            return False
    
    # 对于 NativeFunctionsViewGroup 类型的函数组
    if isinstance(g, NativeFunctionsViewGroup):
        # 检查函数返回类型是否为 "at::Tensor"
        if "at::Tensor" != cpp.returns_type(func.returns, symint=False).cpp_type():
            # 返回值不是 Tensor 类型
            logger.info("NON-TENSOR RET TYPE: %s", str(func))
            return False
        return True
    
    # 对于 NativeFunctionsGroup 类型的函数组，检查其功能函数的参数类型是否支持
    for arg in g.functional.func.schema_order_arguments():
        maybe_method = ivalue_type_conversion_method(arg.type)
        if not maybe_method:
            # 类型转换目前不受支持
            logger.info("NOT SUPPORTED TYPE CONVERTING: %s", g.functional.func)
            return False
    
    # 如果函数组不是结构化的，需要检查其是否具有输出变体实现
    if not g.structured:
        if (
            not hasattr(g, "out")
            or not str(func).endswith("Tensor(a!) out) -> Tensor(a!)")
            or not str(func.name).endswith(".out")
        ):
            return False
    return True
    # 检查返回类型是否为 "at::Tensor &"，如果不是则记录日志并返回 False
    if "at::Tensor &" != cpp.returns_type(func.returns, symint=False).cpp_type():
        logger.info("NON_TENSOR RET TYPE: %s", func)
        return False
    
    # 检查函数是否具有非输出参数的别名，如果有则记录日志并返回 False
    if has_alias(func.arguments.non_out):
        # 这个操作可能会创建输入的别名
        logger.info("INPUTS ALIAS: %s", base_op_name)
        return False
    
    # 如果没有发现返回类型不是 "at::Tensor &" 或者没有输入别名的情况，则返回 True
    return True
# 定义一个方法，用于转换传入类型 `arg_type` 的方法调用表达式，
# 以将 `c10::ivalue` 包含的值转换为预期的 `arg_type` 类型值。
def ivalue_type_conversion_method(
    arg_type: BaseType | OptionalType | Type,
) -> tuple[bool, str] | None:
    """
    Return the method call expression of `c10::ivalue' to convert its contained value to
    the expected value of `arg_type` type. For example, for `arg_type` == BaseTy.Tensor,
    this function returns ".toTensor()", so that it can be appended to the ivalue's
    variable name to get the value of the expected type.
    """
    # 定义不同基础类型 `arg_type` 的转换方法集合
    type_conversion_methods = {
        BaseTy.Tensor: ((True, "toTensor()"), (False, "toOptional<at::Tensor>()")),
        BaseTy.int: ((False, "toInt()"), (False, "toOptional<int64_t>()")),
        BaseTy.bool: ((False, "toBool()"), (False, "toOptional<bool>()")),
        BaseTy.Scalar: ((False, "toScalar()"), (False, "toOptional<at::Scalar>()")),
        BaseTy.ScalarType: (
            (False, "toScalarType()"),
            (False, "toOptional<at::ScalarType>()"),
        ),
        BaseTy.str: (
            (False, "toStringView()"),
            (False, "toOptional<c10::string_view>()"),
        ),
    }

    # 初始化基础类型对象
    base_ty_object = None
    # 根据 `arg_type` 的类型确定基础类型对象
    if isinstance(arg_type, BaseType):
        base_ty_object = arg_type.name
    elif isinstance(arg_type, OptionalType):
        # 如果是可选类型，则获取其元素类型的基础类型对象
        if not isinstance(arg_type.elem, BaseType):
            # 目前不支持列表类型
            return None
        base_ty_object = arg_type.elem.name
    else:
        return None

    # 如果基础类型对象不在转换方法集合中，则返回 None
    if base_ty_object not in type_conversion_methods:
        return None
    # 根据 `arg_type` 的具体类型选择相应的转换方法
    methods = type_conversion_methods[base_ty_object]
    if isinstance(arg_type, BaseType):
        return methods[0]
    return methods[1]


# 冻结集合，包含应使用整数张量操作的操作名称
should_use_int_tensor_ops_ = frozenset(
    (
        "bitwise_not",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "gcd",
        "lcm",
        "scatter",
        "gather",
        "_convert_indices_from_coo_to_csr",
        "_convert_indices_from_csr_to_coo",
    )
)
# 冻结集合，包含应使用复杂张量操作的操作名称
should_use_complex_tensor_ops_ = frozenset(("view_as_real", "imag", "_conj"))


# 判断给定操作名称 `op_name` 是否应使用整数张量操作
def should_use_int_tensor(op_name: str) -> bool:
    return op_name in should_use_int_tensor_ops_


# 判断给定操作名称 `op_name` 是否应使用复杂张量操作
def should_use_complex_tensor(op_name: str) -> bool:
    return op_name in should_use_complex_tensor_ops_


# 冻结集合，包含测试张量维度操作的操作名称集合1
test_tensor_dim_ops_1_ = frozenset(
    (
        "addmv",
        "index_add",
        "_convert_indices_from_coo_to_csr",
        "_convert_indices_from_csr_to_coo",
        "nll_loss_backward",
        "dot",
        "vdot",
        "outer",
        "ger",
    )
)
# 冻结集合，包含测试张量维度操作的操作名称集合2
test_tensor_dim_ops_2_ = frozenset(
    ("addmm", "mm", "nuclear_norm", "diag", "_addmm_activation", "matrix_H", "t")
)


# 根据给定操作名称 `op_name` 返回测试张量的维度：1、2 或者默认 3
def test_tensor_dim(op_name: str) -> int:
    if op_name in test_tensor_dim_ops_1_:
        return 1
    if op_name in test_tensor_dim_ops_2_:
        return 2
    return 3


# 定义一个测试张量形状的字符串表示
test_tensor_shapes_string = '{"view_as_complex": "{2, 2}"}'
# 将测试张量形状的 JSON 字符串解析为 Python 字典
test_tensor_shape_json: dict[str, str] = json.loads(test_tensor_shapes_string)
def test_tensor_shape(op_name: str) -> str:
    # 如果 op_name 存在于 test_tensor_shape_json 中，返回其对应的形状表达式
    if op_name in test_tensor_shape_json:
        return test_tensor_shape_json[op_name]
    else:
        # 否则返回空字符串
        return ""


def test_value_expression(
    arg_type: BaseType | OptionalType | Type, index: int, op_name: str
) -> str:
    # 调用 test_tensor_shape 函数获取张量的形状表达式
    tensor_size_ex = test_tensor_shape(op_name)
    # 如果张量的形状表达式为空
    if tensor_size_ex == "":
        # 根据索引值设置默认张量数量
        num_tensors = 16 if index == 0 else 64
        # 获取张量的维度数
        num_dim = test_tensor_dim(op_name)
        # 计算每个维度的大小，确保为偶数
        size_per_dim = math.ceil(num_tensors / float(num_dim))
        size_per_dim += size_per_dim % 2
        # 格式化生成张量的形状表达式
        tensor_size_ex = "{{{}}}".format(",".join([f"{size_per_dim}"] * num_dim))
    # 根据操作名称判断是否应该使用整数张量
    if should_use_int_tensor(op_name):
        tensor_expression = f"at::randint(1, 100, {tensor_size_ex}, at::kInt)"
    # 根据操作名称判断是否应该使用复数张量
    elif should_use_complex_tensor(op_name):
        tensor_expression = f"at::randn({tensor_size_ex}, at::kComplexFloat)"
    else:
        # 否则使用默认的随机张量生成表达式
        tensor_expression = f"at::rand({tensor_size_ex})"

    # 定义不同基本类型对应的值表达式
    value_expressions = {
        BaseTy.Tensor: tensor_expression,
        BaseTy.int: "1",
        BaseTy.bool: "false",
        BaseTy.Scalar: "2",
        BaseTy.ScalarType: "at::ScalarType::Float",
        BaseTy.str: '"floor"',
    }

    # 获取参数类型对应的基本类型对象
    base_ty_object = None
    if isinstance(arg_type, BaseType):
        base_ty_object = arg_type.name
    else:
        assert isinstance(arg_type, OptionalType) and isinstance(
            arg_type.elem, BaseType
        )
        base_ty_object = arg_type.elem.name
    # 断言基本类型对象存在于值表达式中，如果不存在抛出异常
    assert base_ty_object in value_expressions, "not expected type"
    # 获取基本类型对应的值表达式
    value_expression = value_expressions[base_ty_object]
    # 返回值表达式
    return value_expression


def generate_test_value_definitions(schema: FunctionSchema, index: int) -> str:
    # 断言函数模式不是输出函数
    assert not schema.is_out_fn()
    # 获取函数模式的名称
    schema_name = schema.name.name.base
    # 创建参数名称到测试值表达式的映射字典
    arg_map = {}
    # 遍历函数模式的参数列表
    for arg in schema.schema_order_arguments():
        # 调用 test_value_expression 获取参数的测试值表达式
        test_value_exp = test_value_expression(arg.type, index, schema_name)
        arg_map[arg.name] = test_value_exp
    # 使用配置函数覆盖测试值
    config.override_test_values(arg_map, schema_name, index)
    # 创建参数初始化语句列表
    arg_populations = []
    for arg_name, arg_value in arg_map.items():
        arg_populations.append(f"auto {arg_name}{index} = {arg_value}")
    # 返回初始化语句字符串
    return ";\n    ".join(arg_populations) + ";"


def generate_test_value_names(schema: FunctionSchema, index: int) -> str:
    # 断言函数模式不是输出函数
    assert not schema.is_out_fn()
    # 返回函数模式参数名称列表的逗号分隔字符串
    return ",".join(f"{arg.name}{index}" for arg in schema.schema_order_arguments())


generate_test_ir_arguments_base_ty_to_type_str_ = {
    BaseTy.Tensor: "Tensor",
    BaseTy.int: "int",
    BaseTy.float: "float",
    BaseTy.str: "str",
    BaseTy.Scalar: "int",
    BaseTy.ScalarType: "int",
    BaseTy.bool: "bool",
}


def generate_test_ir_arguments(
    schema: FunctionSchema,
) -> list[tuple[str, str | None]]:
    # 定义函数 ir_argument，接收一个 Argument 类型的参数 arg，并返回一个元组，包含参数名和类型字符串或 None
    def ir_argument(arg: Argument) -> tuple[str, str | None]:
        # 获取参数的类型
        t = arg.type
        # 初始化一个标志，用于指示是否为可选类型
        add_optional = False
        # 如果参数类型是 OptionalType 的实例
        if isinstance(t, OptionalType):
            # 获取 OptionalType 中的元素类型，并将 add_optional 标志设置为 True
            t = t.elem
            add_optional = True
        # 使用断言确保 t 是 BaseType 的实例
        assert isinstance(t, BaseType)
        # 初始化类型字符串为 None
        type_str = None
        # 如果 t 的名称在 generate_test_ir_arguments_base_ty_to_type_str_ 字典中
        if t.name in generate_test_ir_arguments_base_ty_to_type_str_:
            # 获取对应的类型字符串
            type_str = generate_test_ir_arguments_base_ty_to_type_str_[t.name]
        # 如果存在类型字符串且 add_optional 为 True
        if type_str and add_optional:
            # 在类型字符串末尾添加 '?' 表示可选类型
            type_str = f"{type_str}?"
        # 返回参数名以 '%' 开头和类型字符串组成的元组
        return ("%" + arg.name, type_str)

    # 调用 schema_order_arguments 方法获取参数列表，并对每个参数调用 ir_argument 函数，返回结果列表
    return [ir_argument(arg) for arg in schema.schema_order_arguments()]
# 生成参数提取的代码字符串，用于生成函数签名中的参数声明和输入节点的访问
def generate_arg_extraction(schema: FunctionSchema) -> str:
    arg_populations = []
    # 遍历函数参数的枚举列表
    for i, arg in enumerate(schema.schema_order_arguments()):
        # 获取参数类型对应的值转换方法
        maybe_method = ivalue_type_conversion_method(arg.type)
        # 确保获取到值转换方法
        assert maybe_method
        # 解构值转换方法，得到是否为引用和类型转换方法
        is_reference, type_conversion_method = maybe_method
        # 根据是否为引用生成对应的符号
        reference = "&" if is_reference else ""
        # 构建参数声明和访问节点的代码行，并添加到列表中
        arg_populations.append(
            f"const auto{reference} {arg.name} = p_node->Input({i}).{type_conversion_method}"
        )
    # 将所有参数声明和访问节点的代码行用分号连接成字符串返回
    return ";\n    ".join(arg_populations) + ";"


# 获取函数的内核名称
def get_kernel_name(g: NativeFunctionsGroup, backend_index: BackendIndex) -> str:
    # 根据功能组获取内核对象
    kernel = backend_index.get_kernel(g.functional)
    # 如果功能组结构化或者内核对象为空，则返回功能函数的 C++ 名称
    if g.structured or kernel is None:
        return cpp.name(g.functional.func)
    # 否则返回内核对象的内核名称
    return kernel.kernel


# 获取输出函数的内核名称
def get_out_kernel_name(g: NativeFunctionsGroup, backend_index: BackendIndex) -> str:
    # 根据输出功能组获取内核对象
    kernel = backend_index.get_kernel(g.out)
    # 如果输出功能组结构化或者内核对象为空，则返回输出功能函数的 C++ 名称
    if g.structured or kernel is None:
        return cpp.name(g.out.func)
    # 否则返回内核对象的内核名称
    return kernel.kernel


# 生成非输出变体调用的代码字符串
def generate_non_out_variant_call(
    g: NativeFunctionsGroup, backend_index: BackendIndex
) -> str:
    schema = g.functional.func
    # 确保函数模式不是输出函数
    assert not schema.is_out_fn()
    # 获取函数的内核名称
    kernel_name = get_kernel_name(g, backend_index)
    # 获取函数参数的名称生成器
    arg_names = (arg.name for arg in schema.schema_order_arguments())
    # 根据功能组是否结构化确定命名空间名称
    namespace_name = "cpu" if g.structured else "native"
    # 返回函数调用的字符串表示
    return f'at::{namespace_name}::{kernel_name}({",".join(arg_names)})'


# 生成调用视图操作的代码字符串
def generate_call_to_view_ops(
    g: NativeFunctionsViewGroup, backend_index: BackendIndex
) -> str:
    schema = g.view.func
    # 获取视图操作函数的 C++ 名称
    kernel_name = cpp.name(schema)
    # 根据视图操作获取内核对象
    kernel = backend_index.get_kernel(g.view)
    # 如果存在内核对象，则使用内核的内核名称
    if kernel:
        kernel_name = kernel.kernel
    # 获取函数参数的名称生成器
    arg_names = (arg.name for arg in schema.schema_order_arguments())
    # 设置命名空间名称为 native
    namespace_name = "native"
    # 返回视图操作函数调用的字符串表示
    return f'at::{namespace_name}::{kernel_name}({",".join(arg_names)})'


# 生成输出变体调用的代码字符串
def generate_out_variant_call(
    g: NativeFunctionsGroup, backend_index: BackendIndex
) -> str:
    schema = g.out.func
    # 确保函数模式是输出函数
    assert schema.is_out_fn()
    # 初始化参数名称列表
    arg_names = []
    # 获取输出函数的内核名称
    kernel_name = get_out_kernel_name(g, backend_index)
    # 如果功能组结构化
    if g.structured:
        # 从输出参数中获取输出参数的名称
        arg_names = [out_arg.name for out_arg in schema.arguments.out]
    else:
        arg_names = []
    # 遍历非输出参数列表
    for arg in schema.arguments.non_out:
        # 如果参数是 SelfArgument 类型
        if isinstance(arg, SelfArgument):
            # 添加自身参数的名称
            arg_names.append(arg.argument.name)
        else:
            # 确保参数是 Argument 类型，然后添加参数名称
            assert isinstance(arg, Argument)
            arg_names.append(arg.name)
    # 如果功能组非结构化，则确保输出参数列表中只有一个参数
    if not g.structured:
        assert len(schema.arguments.out) == 1
        arg_names.append(schema.arguments.out[0].name)
    # 将参数名称列表用逗号连接成字符串
    cpp_arg_names = ",".join(arg_names)
    # 根据功能组是否结构化确定命名空间名称
    namespace_name = "cpu" if g.structured else "native"
    # 返回输出变体函数调用的字符串表示
    return f"at::{namespace_name}::{kernel_name}({cpp_arg_names})"


# 不需要内存调整的操作集合
no_memory_resize_ops = frozenset(
    # 创建一个包含多个字符串的元组，每个字符串代表一个函数或方法名
    (
        "isin.Scalar_Tensor",         # 检查是否标量张量的成员
        "index_add",                  # 将张量按索引增加
        "dot",                        # 计算向量的点积
        "vdot",                       # 计算向量的内积
        "nuclear_norm",               # 计算矩阵的核范数
        "histc",                      # 计算直方图
        "l1_loss",                    # 计算 L1 损失
        "multi_margin_loss",          # 计算多重边界损失
        "multilabel_margin_loss",     # 计算多标签边界损失
        "nll_loss",                   # 计算负对数似然损失
        "nll_loss2d",                 # 计算二维负对数似然损失
        "prod",                       # 计算张量元素的乘积
    )
    )
    ) -> str:
        # 获取功能函数对象
        functional = g.functional
        # 获取功能函数对象的字符串表示形式（函数名）
        schema = str(functional.func)
        # 生成参数提取代码
        populated_argument = generate_arg_extraction(g.functional.func)
        # 生成非输出变体调用代码
        functional_variant_call = generate_non_out_variant_call(g, backend_index)
        # 断言输出参数列表长度为1
        assert len(g.out.func.arguments.out) == 1
        # 获取输出变量名
        out_variable_name = str(g.out.func.arguments.out[0].name)
        # 生成输出变体调用代码
        out_variant_call = generate_out_variant_call(g, backend_index)
        # 生成完整的 C++ Lambda 表达式字符串
        generated = f"""
      if (n->matches(torch::schema("aten::{schema}"))) {{
        return [](ProcessedNode* p_node) {{
          {populated_argument}
          if (p_node->Output(0).isNone()) {{
            p_node->Output(0) = {functional_variant_call};
            return;
          }}
          auto& {out_variable_name} = p_node->Output(0).toTensor();
          fastResizeToZero({out_variable_name});
          {out_variant_call};
        }};
      }}"""
        # 返回生成的 Lambda 表达式字符串
        return generated

    def view_op_generator(
        self, g: NativeFunctionsViewGroup, backend_index: BackendIndex
    ) -> str:
        # 获取视图操作函数对象的字符串表示形式（函数名）
        schema = str(g.view.func)
        # 生成参数提取代码
        populated_argument = generate_arg_extraction(g.view.func)
        # 生成调用视图操作函数的代码
        functional_variant_call = generate_call_to_view_ops(g, backend_index)
        # 生成完整的 C++ Lambda 表达式字符串
        generated = f"""
      if (n->matches(torch::schema("aten::{schema}"))) {{
        return [](ProcessedNode* p_node) {{
          {populated_argument}
          p_node->Output(0) = {functional_variant_call};
        }};
      }}"""
        # 返回生成的 Lambda 表达式字符串
        return generated
    class GenOpTestCase:
        def out_variant(self, groups: Sequence[NativeFunctionsGroup]) -> str:
            # 如果 groups 为空列表，则返回空字符串
            if not groups:
                return ""
            # 初始化生成的类型变体列表
            generated_type_variants = []
            # 遍历每个 NativeFunctionsGroup 对象 g
            for g in groups:
                # 使用 native_function_manager 管理当前的 NativeFunctionsGroup g
                with native_function_manager(g):
                    # 断言当前的 NativeFunctionsGroup g 是否被支持
                    assert is_supported(g)
                    # 断言 g 是 NativeFunctionsGroup 的实例
                    assert isinstance(g, NativeFunctionsGroup)
                    # 调用 out_variant_op_test_case_generator 方法生成测试用例字符串
                    generated_type_variant = self.out_variant_op_test_case_generator(g)
                    # 将生成的测试用例字符串添加到列表中
                    generated_type_variants.append(generated_type_variant)
            # 返回生成的所有测试用例字符串，用换行符连接
            return "\n".join(generated_type_variants)

        def view(self, groups: Sequence[NativeFunctionsViewGroup]) -> str:
            # 如果 groups 为空列表，则返回空字符串
            if not groups:
                return ""
            # 初始化生成的类型变体列表
            generated_type_variants = []
            # 遍历每个 NativeFunctionsViewGroup 对象 g
            for g in groups:
                # 使用 native_function_manager 管理当前的 NativeFunctionsViewGroup g
                with native_function_manager(g):
                    # 断言当前的 NativeFunctionsViewGroup g 是否被支持
                    assert is_supported(g)
                    # 断言 g 是 NativeFunctionsViewGroup 的实例
                    assert isinstance(g, NativeFunctionsViewGroup)
                    # 调用 view_op_test_case_generator 方法生成测试用例字符串
                    generated_type_variant = self.view_op_test_case_generator(g)
                    # 将生成的测试用例字符串添加到列表中
                    generated_type_variants.append(generated_type_variant)
            # 返回生成的所有测试用例字符串，用换行符连接
            return "\n".join(generated_type_variants)

        def out_variant_op_test_case_generator(self, g: NativeFunctionsGroup) -> str:
            # 获取 NativeFunctionsGroup 对象 g 的函数 schema
            schema = g.functional.func
            # 将 schema 转换为字符串表示
            schema_str = str(schema)
            # 断言 schema_str 中包含 "("
            assert schema_str.find("(") > 0
            # 从 schema_str 中提取类型变体操作的名称
            type_variant_op_name = schema_str[: schema_str.find("(")].replace(".", "_")
            # 根据 NativeFunctionsGroup g 获取操作名称
            op_name = op_name_from_group(g)
            # 断言类型变体操作名称以操作名称开头
            assert type_variant_op_name.startswith(op_name)

            # 生成测试用例参数的 IR 表达式
            arg_types = generate_test_ir_arguments(schema)
            # 生成参数声明的字符串，包括可选的类型注释
            arg_declarations = ", ".join(
                (
                    arg_name if arg_type is None else f"{arg_name}: {arg_type}"
                    for arg_name, arg_type in arg_types
                )
            )
            # 生成参数名称的字符串
            arg_names = ", ".join((arg_name for arg_name, _ in arg_types))
            # 断言 schema 的返回值列表长度为1，并且返回类型是 BaseType，并且类型名称是 BaseTy.Tensor
            assert (
                len(schema.returns) == 1
                and isinstance(schema.returns[0].type, BaseType)
                and schema.returns[0].type.name is BaseTy.Tensor
            )
            # 生成测试值定义的 IR 表达式
            test_value_definitions = generate_test_value_definitions(schema, 0)
            # 生成测试值名称的字符串
            test_value_names = generate_test_value_names(schema, 0)
            # 生成第二组测试值定义的 IR 表达式
            test_value_definitions2 = generate_test_value_definitions(schema, 1)
            # 生成第二组测试值名称的字符串
            test_value_names2 = generate_test_value_names(schema, 1)
            # 检查是否需要检查调整大小，生成相应的字符串
            check_resize = "true" if should_check_resize(schema) else "false"
            # 生成最终的测试用例字符串，使用 R"IR( 多行字符串表示
            generated = f"""
TEST(StaticRuntime, autogen_{type_variant_op_name}) {{
  const std::string script = R"IR(
  )IR";

  {test_value_definitions}
  // 创建一个包含测试值名称的参数向量
  std::vector<IValue> args{{{test_value_names}}};
  // 使用 testStaticRuntime 函数测试静态运行时，传入参数 args
  testStaticRuntime(script, args, {{}}, /*use_allclose=*/false, /*use_equalnan=*/false, /*check_resize=*/{check_resize});

  {test_value_definitions2}
  // 创建另一个包含测试值名称的参数向量
  std::vector<IValue> args2{{{test_value_names2}}};
  // 使用 testStaticRuntime 函数再次测试静态运行时，传入参数 args 和 args2
  testStaticRuntime(script, args, args2, /*use_allclose=*/false, /*use_equalnan=*/false, /*check_resize=*/{check_resize});
    """
    生成测试用例的字符串表示，以测试特定视图组的本地函数

    Parameters:
    - g: 本地函数视图组对象

    Returns:
    - 生成的测试用例字符串
    """
    schema = g.view.func  # 获取视图组的函数模式
    schema_str = str(schema)  # 将函数模式转换为字符串表示
    assert schema_str.find("(") > 0  # 断言函数模式字符串包含 '('

    # 提取函数类型变体的操作名，转换成合适的格式
    type_variant_op_name = schema_str[: schema_str.find("(")].replace(".", "_")

    op_name = g.view.root_name  # 获取根操作的名称
    assert type_variant_op_name.startswith(op_name)  # 断言类型变体操作名以操作名开头

    # 生成测试用例参数的 IR 表示，并拼接为字符串
    arg_types = generate_test_ir_arguments(schema)
    arg_declarations = ", ".join(
        (
            arg_name if arg_type is None else f"{arg_name}: {arg_type}"
            for arg_name, arg_type in arg_types
        )
    )
    arg_names = ", ".join((arg_name for arg_name, _ in arg_types))

    assert (
        len(schema.returns) == 1
        and isinstance(schema.returns[0].type, BaseType)
        and schema.returns[0].type.name is BaseTy.Tensor
    )  # 断言函数返回值的类型为 BaseType 的 Tensor

    # 生成测试值的定义和名称
    test_value_definitions = generate_test_value_definitions(schema, 0)
    test_value_names = generate_test_value_names(schema, 0)

    # 构造最终的测试用例字符串
    generated = f"""
TEST(StaticRuntime, autogen_{type_variant_op_name}) {{
  const std::string script = R"IR(
    graph({arg_declarations}):
        %bias: None = prim::Constant()
        %ret = aten::{op_name}({arg_names})
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  {test_value_definitions}
  std::vector<IValue> args{{{test_value_names}}};
  testStaticRuntime(script, args);
}}
"""

    return generated
```