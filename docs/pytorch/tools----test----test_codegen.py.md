# `.\pytorch\tools\test\test_codegen.py`

```py
# 从未来导入 annotations 特性，用于支持注解的类型提示
from __future__ import annotations

# 导入 dataclasses 模块，用于支持定义不可变数据类
import dataclasses
# 导入 typing 模块，支持类型提示和类型检查
import typing
# 导入 unittest 模块，用于编写和运行单元测试
import unittest
# 导入 defaultdict 类型，用于创建默认值为列表的字典
from collections import defaultdict

# 导入 yaml 模块，用于读取和写入 YAML 格式的数据
import yaml

# 从 tools.autograd 导入 gen_autograd_functions 和 load_derivatives 函数
from tools.autograd import gen_autograd_functions, load_derivatives

# 从 torchgen.dest 导入 dest 对象
from torchgen import dest
# 从 torchgen.api.types 导入 CppSignatureGroup 和 DispatcherSignature 类
from torchgen.api.types import CppSignatureGroup, DispatcherSignature
# 从 torchgen.context 导入 native_function_manager 对象
from torchgen.context import native_function_manager
# 从 torchgen.gen 导入多个函数和类
from torchgen.gen import (
    get_native_function_declarations,
    get_native_function_schema_registrations,
    LineLoader,
    static_dispatch,
)
# 从 torchgen.model 导入多个类
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    FunctionSchema,
    Location,
    NativeFunction,
    OperatorName,
)
# 从 torchgen.native_function_generation 导入 add_generated_native_functions 函数
from torchgen.native_function_generation import add_generated_native_functions
# 从 torchgen.selective_build.selector 导入 SelectiveBuilder 类
from torchgen.selective_build.selector import SelectiveBuilder

# 定义单元测试类 TestCreateDerivative，继承自 unittest.TestCase
class TestCreateDerivative(unittest.TestCase):

    # 定义测试方法 test_named_grads，返回类型为 None
    def test_named_grads(self) -> None:
        # 解析函数模式字符串，创建 FunctionSchema 对象
        schema = FunctionSchema.parse(
            "func(Tensor a, Tensor b) -> (Tensor x, Tensor y)"
        )
        # 使用给定的默认原生函数，替换 func 字段后创建原生函数对象
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        # 调用 load_derivatives.create_derivative 函数，创建导数对象 derivative
        derivative = load_derivatives.create_derivative(
            native_function,
            formula="func_backward(grad_x, grad_y)",
            var_names=(),
            available_named_gradients=["grad_x", "grad_y"],
        )
        # 断言 derivative 的命名梯度集合与 {"grad_x", "grad_y"} 相等
        self.assertSetEqual(derivative.named_gradients, {"grad_x", "grad_y"})

    # 定义测试方法 test_non_differentiable_output，返回类型为 None
    def test_non_differentiable_output(self) -> None:
        # 指定函数规范字符串
        specification = "func(Tensor a, Tensor b) -> (Tensor x, bool y, Tensor z)"
        # 解析函数模式字符串，创建 FunctionSchema 对象
        schema = FunctionSchema.parse(specification)
        # 使用给定的默认原生函数，替换 func 字段后创建原生函数对象
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        # 调用 load_derivatives.create_differentiability_info 函数，
        # 创建不同iability_info 对象和返回值元组 _
        _, differentiability_info = load_derivatives.create_differentiability_info(
            defn_dict={
                "name": specification,
                "dispatch": {"Default": {"a": "grads[0]", "b": "grads[2]"}},
            },
            functions_by_signature={schema.signature(): [native_function]},
            functions_by_schema={specification: native_function},
            op_counter=typing.Counter[str](),
            used_dispatch_keys=set(),
        )

        # 断言 differentiability_info["Default"].available_named_gradients 的序列与
        # ["grad_x", "grad_z"] 相等，说明可用的命名梯度
        self.assertSequenceEqual(
            differentiability_info["Default"].available_named_gradients,
            # grad_y is not present because y is a
            # bool and thus not differentiable.
            ["grad_x", "grad_z"],
        )

    # 定义测试方法 test_indexed_grads，返回类型为 None
    def test_indexed_grads(self) -> None:
        # 解析函数模式字符串，创建 FunctionSchema 对象
        schema = FunctionSchema.parse(
            "func(Tensor a, Tensor b) -> (Tensor x, Tensor y)"
        )
        # 使用给定的默认原生函数，替换 func 字段后创建原生函数对象
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        # 调用 load_derivatives.create_derivative 函数，创建导数对象 derivative
        derivative = load_derivatives.create_derivative(
            native_function,
            formula="func_backward(grads[0], grads[1])",
            var_names=(),
            available_named_gradients=["grad_x", "grad_y"],
        )
        # 断言 derivative 的命名梯度集合与空集相等
        self.assertSetEqual(derivative.named_gradients, set())
    # 定义一个测试方法，用于测试命名梯度和索引梯度的行为
    def test_named_grads_and_indexed_grads(self) -> None:
        # 定义一个函数规范字符串，描述了函数接受两个张量参数并返回两个张量的结构
        specification = "func(Tensor a, Tensor b) -> (Tensor x, Tensor y)"
        # 解析函数规范，生成函数模式对象
        schema = FunctionSchema.parse(specification)
        # 使用函数模式对象替换默认的本地函数对象，创建一个新的本地函数对象
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        # 在测试中，确保引发运行时错误，并检查错误消息中是否包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError, 'illegally mixes use of "grad_RETURN_NAME"'
        ):
            # 调用加载导数信息的函数，传入定义字典和其他参数
            load_derivatives.create_differentiability_info(
                defn_dict={
                    "name": specification,
                    # 定义派发字典，指定默认派发方式和参数的梯度名称或索引
                    "dispatch": {
                        "Default": {
                            "a": "grad_x",
                            "b": "grads[1]",
                        }
                    },
                },
                # 将函数模式对象映射到其对应的本地函数对象列表
                functions_by_signature={schema.signature(): [native_function]},
                # 将函数规范映射到其对应的本地函数对象
                functions_by_schema={specification: native_function},
                # 统计操作计数器，用于记录操作的数量
                op_counter=typing.Counter[str](),
                # 已使用的派发键集合为空
                used_dispatch_keys=set(),
            )
class TestGenAutogradFunctions(unittest.TestCase):
    # 定义测试方法，验证非可微输出的无效类型情况
    def test_non_differentiable_output_invalid_type(self) -> None:
        # 定义函数规范字符串
        specification = "func(Tensor a, Tensor b) -> (Tensor x, bool y, Tensor z)"
        # 解析函数规范，生成函数模式对象
        schema = FunctionSchema.parse(specification)
        # 创建带有指定函数规范的默认本地函数对象
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        # 调用加载导数信息的函数，创建不同导数信息对象
        _, differentiability_info = load_derivatives.create_differentiability_info(
            defn_dict={
                "name": specification,
                "dispatch": {
                    "Default": {
                        "a": "grad_x",
                        "b": "grad_z",
                    }
                },
            },
            functions_by_signature={schema.signature(): [native_function]},
            functions_by_schema={specification: native_function},
            op_counter=typing.Counter[str](),
            used_dispatch_keys=set(),
        )
        # 生成自动求导函数的定义，根据不同导数信息中的“Default”配置
        definition = gen_autograd_functions.process_function(
            differentiability_info["Default"],
            gen_autograd_functions.FUNCTION_DEFINITION,
        )
        # 断言语句，确保“grad_z = grads[2]”不在生成的函数定义中
        assert "grad_z = grads[2]" not in definition
        # 断言语句，确保“grad_z = grads[1]”在生成的函数定义中
        assert "grad_z = grads[1]" in definition
    # 定义测试方法，验证非可微输出的可微性
    def test_non_differentiable_output_output_differentiability(self) -> None:
        # 定义函数规范字符串
        specification = "func(Tensor a, Tensor b) -> (Tensor x, Tensor y, Tensor z)"
        # 解析函数模式，生成函数模式对象
        schema = FunctionSchema.parse(specification)
        # 创建默认的本地函数对象，替换其函数定义为给定的函数模式
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        # 调用加载导数信息的函数，创建不同可微性信息
        _, differentiability_info = load_derivatives.create_differentiability_info(
            # 定义函数信息字典
            defn_dict={
                "name": specification,
                "dispatch": {
                    # 默认分发策略映射
                    "Default": {
                        "a": "grad_x",  # a 映射到 grad_x
                        "b": "grad_z",  # b 映射到 grad_z
                    },
                    # AutogradNestedTensor 分发策略映射
                    "AutogradNestedTensor": {
                        "a": "grad_z",  # a 映射到 grad_z
                        "b": "grad_x",  # b 映射到 grad_x
                    },
                },
                # 输出的不同可微性信息列表
                "output_differentiability": [True, False, True],
            },
            # 函数按签名分组的字典
            functions_by_signature={schema.signature(): [native_function]},
            # 函数按规范名称分组的字典
            functions_by_schema={specification: native_function},
            # 操作计数器对象
            op_counter=typing.Counter[str](),
            # 使用的分发键集合
            used_dispatch_keys=set(),
        )

        # 处理默认函数定义，生成自动求导函数的定义字符串
        default_definition = gen_autograd_functions.process_function(
            differentiability_info["Default"],
            gen_autograd_functions.FUNCTION_DEFINITION,
        )
        # 断言默认定义中不应包含 "grad_z = grads[2]"，而应包含 "grad_z = grads[1]"，因为输出 1 (y) 不可微
        assert "grad_z = grads[2]" not in default_definition
        assert "grad_z = grads[1]" in default_definition

        # 处理 AutogradNestedTensor 函数定义，生成自动求导函数的定义字符串
        nested_tensor_definition = gen_autograd_functions.process_function(
            differentiability_info["AutogradNestedTensor"],
            gen_autograd_functions.FUNCTION_DEFINITION,
        )
        # 断言 AutogradNestedTensor 定义中不应包含 "grad_z = grads[2]"，而应包含 "grad_z = grads[1]"，因为输出 1 (y) 不可微
        assert "grad_z = grads[2]" not in nested_tensor_definition
        assert "grad_z = grads[1]" in nested_tensor_definition
    # 定义一个测试方法，用于测试注册无效的调度键情况
    def test_register_bogus_dispatch_key(self) -> None:
        # 定义函数规范字符串
        specification = "func(Tensor a, Tensor b) -> (Tensor x, bool y, Tensor z)"
        # 解析函数规范，生成函数模式对象
        schema = FunctionSchema.parse(specification)
        # 替换默认的本地函数对象，将其函数规范替换为新的函数模式对象
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        # 使用断言检查是否会抛出RuntimeError，并且错误消息包含特定文本
        with self.assertRaisesRegex(
            RuntimeError,
            "Invalid dispatch key AutogradRandomTensor in derivatives.yaml for",
        ):
            # 调用加载导数信息的函数，传入相关参数
            load_derivatives.create_differentiability_info(
                defn_dict={
                    "name": specification,
                    "dispatch": {
                        "Default": {
                            "a": "grad_x",
                            "b": "grad_z",
                        },
                        "AutogradRandomTensor": {
                            "a": "grad_x",
                            "b": "grad_z",
                        },
                    },
                },
                functions_by_signature={schema.signature(): [native_function]},
                functions_by_schema={specification: native_function},
                op_counter=typing.Counter[str](),
                used_dispatch_keys=set(),
            )
class TestGenSchemaRegistration(unittest.TestCase):
    # 设置测试环境
    def setUp(self) -> None:
        # 初始化一个空的选择器
        self.selector = SelectiveBuilder.get_nop_selector()
        # 从 YAML 文件中加载自定义原生函数对象
        self.custom_native_function, _ = NativeFunction.from_yaml(
            {"func": "custom::func() -> bool"},
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        # 从 YAML 文件中加载片段化的自定义原生函数对象
        (
            self.fragment_custom_native_function,
            _,
        ) = NativeFunction.from_yaml(
            {"func": "quantized_decomposed::func() -> bool"},
            loc=Location(__file__, 1),
            valid_tags=set(),
        )

    # 测试默认命名空间的模式注册代码是否有效
    def test_default_namespace_schema_registration_code_valid(self) -> None:
        # 获取默认原生函数的模式注册
        native_functions = [DEFAULT_NATIVE_FUNCTION]
        registrations, _ = get_native_function_schema_registrations(
            native_functions=native_functions,
            schema_selector=self.selector,
        )
        # 断言注册结果与预期相符
        self.assertEqual(registrations, ['m.def("func() -> bool", {});\n'])

    # 测试自定义命名空间的模式注册代码是否有效
    def test_custom_namespace_schema_registration_code_valid(self) -> None:
        # 获取自定义原生函数的模式注册
        _, registrations = get_native_function_schema_registrations(
            native_functions=[self.custom_native_function],
            schema_selector=self.selector,
        )
        # 断言注册结果与预期相符
        self.assertEqual(
            registrations,
            """
TORCH_LIBRARY(custom, m) {
  m.def("func() -> bool", {});

};""",
        )

    # 测试片段化自定义命名空间的模式注册代码是否有效
    def test_fragment_custom_namespace_schema_registration_code_valid(self) -> None:
        """有时我们想要扩展一个已存在的命名空间，例如已在 native/quantized/library.cpp 中定义的 quantized 命名空间。"""
        # 获取片段化自定义原生函数的模式注册
        _, registrations = get_native_function_schema_registrations(
            native_functions=[self.fragment_custom_native_function],
            schema_selector=self.selector,
        )
        # 断言注册结果与预期相符
        self.assertEqual(
            registrations,
            """
TORCH_LIBRARY_FRAGMENT(quantized_decomposed, m) {
  m.def("func() -> bool", {});

};""",
        )

    # 测试混合命名空间的模式注册代码是否有效
    def test_mixed_namespace_schema_registration_code_valid(self) -> None:
        # 获取默认原生函数和自定义原生函数的模式注册
        (
            aten_registrations,
            custom_registrations,
        ) = get_native_function_schema_registrations(
            native_functions=[DEFAULT_NATIVE_FUNCTION, self.custom_native_function],
            schema_selector=self.selector,
        )
        # 断言默认原生函数注册结果与预期相符
        self.assertEqual(aten_registrations, ['m.def("func() -> bool", {});\n'])
        # 断言自定义原生函数注册结果与预期相符
        self.assertEqual(
            custom_registrations,
            """
TORCH_LIBRARY(custom, m) {
  m.def("func() -> bool", {});

};""",
        )
    # 定义测试函数 test_3_namespaces_schema_registration_code_valid，无返回值
    def test_3_namespaces_schema_registration_code_valid(self) -> None:
        # 从 YAML 数据创建 NativeFunction 对象，获取 custom2::func() -> bool 函数
        custom2_native_function, _ = NativeFunction.from_yaml(
            {"func": "custom2::func() -> bool"},
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        # 调用 get_native_function_schema_registrations 函数，获取原生函数和自定义函数的注册信息
        (
            aten_registrations,
            custom_registrations,
        ) = get_native_function_schema_registrations(
            native_functions=[
                DEFAULT_NATIVE_FUNCTION,  # 默认原生函数
                self.custom_native_function,  # 自定义原生函数
                custom2_native_function,  # custom2 的原生函数
            ],
            schema_selector=self.selector,  # 模式选择器对象
        )
        # 断言 aten_registrations 是否等于 ['m.def("func() -> bool", {});\n']
        self.assertEqual(aten_registrations, ['m.def("func() -> bool", {});\n'])
        # 断言 custom_registrations 是否为空字符串
        self.assertEqual(
            custom_registrations,
            """
TORCH_LIBRARY(custom, m) {
  m.def("func() -> bool", {});
  // 定义一个名为 custom 的 Torch 库，注册一个返回布尔值的函数 func()
};

TORCH_LIBRARY(custom2, m) {
  m.def("func() -> bool", {});
  // 定义一个名为 custom2 的 Torch 库，注册一个返回布尔值的函数 func()
};

class TestGenNativeFunctionDeclaration(unittest.TestCase):
  // 定义一个测试类 TestGenNativeFunctionDeclaration，用于测试原生函数声明
    def setUp(self) -> None:
        // 设置测试前的准备工作
        self.op_1_native_function, op_1_backend_index = NativeFunction.from_yaml(
            {"func": "op_1() -> bool", "dispatch": {"CPU": "kernel_1"}},
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        // 使用 YAML 配置创建原生函数 op_1，并指定 CPU 分发到 kernel_1
        self.op_2_native_function, op_2_backend_index = NativeFunction.from_yaml(
            {
                "func": "op_2() -> bool",
                "dispatch": {"CPU": "kernel_2", "QuantizedCPU": "custom::kernel_3"},
            },
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        // 使用 YAML 配置创建原生函数 op_2，指定 CPU 和 QuantizedCPU 分发到不同的内核

        backend_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]] = {
            DispatchKey.CPU: {},
            DispatchKey.QuantizedCPU: {},
        }
        // 创建一个字典，用于存储各个 DispatchKey 对应的 OperatorName 和 BackendMetadata

        BackendIndex.grow_index(backend_indices, op_1_backend_index)
        // 扩展 backend_indices，添加 op_1 的 BackendIndex
        BackendIndex.grow_index(backend_indices, op_2_backend_index)
        // 扩展 backend_indices，添加 op_2 的 BackendIndex

        self.backend_indices = {
            k: BackendIndex(
                dispatch_key=k,
                use_out_as_primary=True,
                external=False,
                device_guard=False,
                index=backend_indices[k],
            )
            for k in backend_indices
        }
        // 创建一个包含各个 DispatchKey 的 BackendIndex 实例，用于管理后端索引

    def test_native_function_declaration_1_op_2_ns_error(self) -> None:
        // 定义一个测试函数，测试 op_2 的命名空间错误
        with self.assertRaises(AssertionError):
            get_native_function_declarations(
                grouped_native_functions=[
                    self.op_1_native_function,
                    self.op_2_native_function,
                ],
                backend_indices=self.backend_indices,
                native_function_decl_gen=dest.compute_native_function_declaration,
            )
        // 断言抛出 AssertionError，因为 op_2 命名空间错误

    def test_native_function_declaration_1_op_1_ns_valid(self) -> None:
        // 定义一个测试函数，测试 op_1 的命名空间有效性
        self.assertIsInstance(self.op_1_native_function, NativeFunction)
        // 断言 op_1_native_function 是 NativeFunction 类型
        declaration = get_native_function_declarations(
            grouped_native_functions=[
                self.op_1_native_function,
            ],
            backend_indices=self.backend_indices,
            native_function_decl_gen=dest.compute_native_function_declaration,
        )
        // 调用函数生成 op_1 的原生函数声明
        target = """
namespace at {
namespace native {
TORCH_API bool kernel_1();
} // namespace native
} // namespace at
        """
        // 期望的生成目标代码
        self.assertEqual("\n".join(declaration), target)
        // 断言生成的声明与目标代码一致

# Test for native_function_generation
class TestNativeFunctionGeneratrion(unittest.TestCase):
  // 定义一个测试类 TestNativeFunctionGeneratrion，用于测试原生函数生成

    def setUp(self) -> None:
        // 设置测试前的准备工作
        self.native_functions: list[NativeFunction] = []
        // 初始化原生函数列表为空
        self.backend_indices: dict[
            DispatchKey, dict[OperatorName, BackendMetadata]
        ] = defaultdict(dict)
        // 初始化后端索引为默认字典

        yaml_entry = """
- func: op(Tensor self) -> Tensor
  dispatch:
        """
        // YAML 配置示例，定义了一个 op 函数的签名和分发方式
    # 定义类 CompositeExplicitAutograd 的 op 属性
    CompositeExplicitAutograd: op
    # 从 op 的输出属性 autogen 获取数据
    autogen: op.out
        """
        从 YAML 数据中加载条目，使用 LineLoader 加载器
        es = yaml.load(yaml_entry, Loader=LineLoader)
        从第一个 YAML 条目创建 NativeFunction 对象，并获取返回函数和元数据 m
        self.one_return_func, m = NativeFunction.from_yaml(
            es[0], loc=Location(__file__, 1), valid_tags=set()
        )

        使用元数据 m 扩展 BackendIndex 中的索引
        BackendIndex.grow_index(self.backend_indices, m)

        从固定的 YAML 数据中创建带有两个返回值的 NativeFunction 对象
        self.two_returns_func, two_returns_backend_index = NativeFunction.from_yaml(
            {
                "func": "op_2() -> (Tensor, Tensor)",
                "dispatch": {"CPU": "kernel_1"},
                "autogen": "op_2.out",
            },
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        使用元数据 two_returns_backend_index 扩展 BackendIndex 中的索引
        BackendIndex.grow_index(self.backend_indices, two_returns_backend_index)
        
    # 定义测试功能变体 functional_variant_autogen_out_variant
    def test_functional_variant_autogen_out_variant(self) -> None:
        将 self.one_return_func 添加到 native_functions 列表中
        native_functions = [self.one_return_func]
        向 backend_indices 中添加生成的本地函数
        add_generated_native_functions(native_functions, self.backend_indices)
        断言 native_functions 的长度为 2
        self.assertEqual(len(native_functions), 2)
        断言 native_functions[1].func 的字符串表示与特定格式匹配
        self.assertEqual(
            str(native_functions[1].func),
            "op.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
        )
        获取操作名称 op_name
        op_name = native_functions[1].func.name
        从 backend_indices 中获取 DispatchKey.CompositeExplicitAutograd 下 op_name 的后端元数据
        backend_metadata = self.backend_indices[DispatchKey.CompositeExplicitAutograd][
            op_name
        ]
        断言 backend_metadata 的内核为 "op_out"
        self.assertEqual(backend_metadata.kernel, "op_out")

    # 定义测试功能变体 functional_variant_autogen_out_variant_two_returns
    def test_functional_variant_autogen_out_variant_two_returns(self) -> None:
        将 self.two_returns_func 添加到 native_functions 列表中
        native_functions = [self.two_returns_func]
        向 backend_indices 中添加生成的本地函数
        add_generated_native_functions(native_functions, self.backend_indices)
        断言 native_functions 的长度为 2
        self.assertEqual(len(native_functions), 2)
        断言 native_functions[1].func 的字符串表示与特定格式匹配
        self.assertEqual(
            str(native_functions[1].func),
            "op_2.out(*, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))",
        )
        获取操作名称 op_name
        op_name = native_functions[1].func.name
        从 backend_indices 中获取 DispatchKey.CompositeExplicitAutograd 下 op_name 的后端元数据
        backend_metadata = self.backend_indices[DispatchKey.CompositeExplicitAutograd][
            op_name
        ]
        断言 backend_metadata 的内核为 "op_2_out"
        self.assertEqual(backend_metadata.kernel, "op_2_out")
# Test for static_dispatch

# 定义测试类 TestStaticDispatchGeneratrion，继承自 unittest.TestCase
class TestStaticDispatchGeneratrion(unittest.TestCase):
    
    # 初始化设置方法
    def setUp(self) -> None:
        # 初始化 self.backend_indices 为默认字典，键为 DispatchKey，值为包含 OperatorName 和 BackendMetadata 的字典
        self.backend_indices: dict[
            DispatchKey, dict[OperatorName, BackendMetadata]
        ] = defaultdict(dict)
        
        # 定义 YAML 字符串
        yaml_entry = """
        - func: op.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
          dispatch:
            CompositeExplicitAutograd: op
        """
        
        # 使用 LineLoader 加载 YAML 字符串
        es = yaml.load(yaml_entry, Loader=LineLoader)
        
        # 从 YAML 数据创建 NativeFunction 对象并获取相关元数据 m
        self.one_return_func, m = NativeFunction.from_yaml(
            es[0], loc=Location(__file__, 1), valid_tags=set()
        )
        
        # 将 m 加入到 self.backend_indices 的对应项中
        BackendIndex.grow_index(self.backend_indices, m)
        
        # 设置 dispatch_key 为 DispatchKey.CompositeExplicitAutograd
        dispatch_key = DispatchKey.CompositeExplicitAutograd
        
        # 断言 dispatch_key 是否在 self.backend_indices 中
        self.assertTrue(dispatch_key in self.backend_indices)
        
        # 初始化 self.indices 为包含 BackendIndex 对象的列表
        self.indices = [
            BackendIndex(
                dispatch_key=dispatch_key,
                use_out_as_primary=True,
                external=False,
                device_guard=False,
                index=self.backend_indices[dispatch_key],
            )
        ]
    
    # 测试函数：测试具有单个后端生成静态分派
    def test_op_with_1_backend_generates_static_dispatch(self) -> None:
        # 从 self.one_return_func 的 func 字段创建 DispatcherSignature 对象 disp_sig
        disp_sig = DispatcherSignature.from_schema(self.one_return_func.func)
        
        # 使用 self.one_return_func 进入本地函数管理器
        with native_function_manager(self.one_return_func):
            # 调用 static_dispatch 函数并获取返回值 out
            out = static_dispatch(
                sig=disp_sig,
                f=self.one_return_func,
                backend_indices=self.indices,
            )
        
        # 断言 out 的值与指定字符串相等
        self.assertEqual(
            out, "return at::compositeexplicitautograd::op_out(out, self);"
        )
    
    # 测试函数：测试使用 cpp 签名生成静态分派
    def test_op_with_cpp_sig_generates_static_dispatch(self) -> None:
        # 从 NativeFunction 对象 self.one_return_func 创建 CppSignatureGroup
        sig_group = CppSignatureGroup.from_native_function(
            self.one_return_func,
            method=False,
            fallback_binding=self.one_return_func.manual_cpp_binding,
        )
        
        # 使用 self.one_return_func 进入本地函数管理器
        with native_function_manager(self.one_return_func):
            # 调用 static_dispatch 函数并获取返回值 out
            out = static_dispatch(
                sig=sig_group.signature,
                f=self.one_return_func,
                backend_indices=self.indices,
            )
        
        # 断言 out 的值与指定字符串相等
        self.assertEqual(
            out, "return at::compositeexplicitautograd::op_out(out, self);"
        )

# 表示最基本的 NativeFunction。使用 dataclasses.replace() 进行编辑。
# 初始化 DEFAULT_NATIVE_FUNCTION 和一个未使用的元组
DEFAULT_NATIVE_FUNCTION, _ = NativeFunction.from_yaml(
    {"func": "func() -> bool"},
    loc=Location(__file__, 1),
    valid_tags=set(),
)

# 如果当前脚本被作为主程序运行，则执行单元测试
if __name__ == "__main__":
    unittest.main()
```