# `.\pytorch\tools\test\test_gen_backend_stubs.py`

```py
# Owner(s): ["module: codegen"]

# 引入未来版本的注解特性
from __future__ import annotations

# 导入标准库模块
import os
import tempfile
import unittest

# 导入自定义模块
import expecttest

# 导入全局变量，但不使用，用于静态检查
from torchgen.gen import _GLOBAL_PARSE_NATIVE_YAML_CACHE  # noqa: F401

# 获取当前文件所在路径
path = os.path.dirname(os.path.realpath(__file__))
# 拼接生成后端存根文件的路径
gen_backend_stubs_path = os.path.join(path, "../torchgen/gen_backend_stubs.py")

# gen_backend_stubs.py 是一个被外部后端直接调用的集成点。
# 这里的测试用例用于确认输入格式不正确时是否能得到合理的错误信息。
class TestGenBackendStubs(expecttest.TestCase):
    
    # 每个测试方法执行前的设置方法
    def setUp(self) -> None:
        # 清空全局的本地 YAML 缓存
        global _GLOBAL_PARSE_NATIVE_YAML_CACHE
        _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()

    # 断言从 gen_backend_stubs 成功返回的方法
    def assert_success_from_gen_backend_stubs(self, yaml_str: str) -> None:
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_str)
            fp.flush()
            # 运行 gen_backend_stubs，并传入临时文件的路径名作为参数
            run(fp.name, "", True)

    # 从 gen_backend_stubs 获取错误信息的方法
    def get_errors_from_gen_backend_stubs(
        self, yaml_str: str, *, kernels_str: str | None = None
    ) -> str:
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_str)
            fp.flush()
            try:
                if kernels_str is None:
                    # 运行 gen_backend_stubs，并传入临时文件的路径名作为参数
                    run(fp.name, "", True)
                else:
                    with tempfile.NamedTemporaryFile(mode="w") as kernel_file:
                        kernel_file.write(kernels_str)
                        kernel_file.flush()
                        # 运行 gen_backend_stubs，并传入临时文件的路径名和内核文件路径名作为参数
                        run(fp.name, "", True, impl_path=kernel_file.name)
            except AssertionError as e:
                # 从错误消息中删除临时文件的路径名，以简化断言
                return str(e).replace(fp.name, "")
            # 如果没有抛出 AssertionError，则测试失败
            self.fail(
                "Expected gen_backend_stubs to raise an AssertionError, but it did not."
            )

    # 测试单个操作有效的方法
    def test_valid_single_op(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs"""
        # 调用断言成功方法，传入 YAML 字符串
        self.assert_success_from_gen_backend_stubs(yaml_str)

    # 测试多个操作有效的方法
    def test_valid_multiple_ops(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
- abs"""
        # 调用断言成功方法，传入 YAML 字符串
        self.assert_success_from_gen_backend_stubs(yaml_str)

    # 测试零个操作有效的方法
    def test_valid_zero_ops(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:"""
        # 调用断言成功方法，传入 YAML 字符串
        self.assert_success_from_gen_backend_stubs(yaml_str)

    # 测试零个操作不需要后端分发键的有效方法
    def test_valid_zero_ops_doesnt_require_backend_dispatch_key(self) -> None:
        yaml_str = """\
backend: BAD_XLA
cpp_namespace: torch_xla
supported:"""
        # 外部代码生成的 yaml 文件中没有操作符实际上是一个空操作，
        # 因此没有理由解析后端
        self.assert_success_from_gen_backend_stubs(yaml_str)

    # 测试包含自动微分操作的有效方法
    def test_valid_with_autograd_ops(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs
autograd:"""
        # 调用断言成功方法，传入 YAML 字符串
        self.assert_success_from_gen_backend_stubs(yaml_str)
    # External codegen on a yaml file with no operators is effectively a no-op,
    # so there's no reason to parse the backend
    self.assert_success_from_gen_backend_stubs(yaml_str)

def test_missing_backend(self) -> None:
    yaml_str = """\
cpp_namespace: torch_xla
supported:
- abs"""
    # Get errors from generating backend stubs without a specified backend
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # Assert that the error message indicates a missing "backend" value
    self.assertExpectedInline(
        output_error, '''You must provide a value for "backend"'''
    )

def test_empty_backend(self) -> None:
    yaml_str = """\
backend:
cpp_namespace: torch_xla
supported:
- abs"""
    # Get errors from generating backend stubs with an empty "backend" value
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # Assert that the error message indicates a missing "backend" value
    self.assertExpectedInline(
        output_error, '''You must provide a value for "backend"'''
    )

def test_backend_invalid_dispatch_key(self) -> None:
    yaml_str = """\
backend: NOT_XLA
cpp_namespace: torch_xla
supported:
- abs"""
    # Get errors from generating backend stubs with an invalid dispatch key
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # Assert that the error message indicates an unknown dispatch key
    self.assertExpectedInline(
        output_error,
        """\
unknown dispatch key NOT_XLA
  The provided value for "backend" must be a valid DispatchKey, but got NOT_XLA.""",
    )  # noqa: B950

def test_missing_cpp_namespace(self) -> None:
    yaml_str = """\
backend: XLA
supported:
- abs"""
    # Get errors from generating backend stubs without a specified "cpp_namespace"
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # Assert that the error message indicates a missing "cpp_namespace" value
    self.assertExpectedInline(
        output_error, '''You must provide a value for "cpp_namespace"'''
    )

def test_whitespace_cpp_namespace(self) -> None:
    yaml_str = """\
backend: XLA
cpp_namespace:\t
supported:
- abs"""
    # Get errors from generating backend stubs with a whitespace-only "cpp_namespace"
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # Assert that the error message indicates a missing "cpp_namespace" value
    self.assertExpectedInline(
        output_error, '''You must provide a value for "cpp_namespace"'''
    )

# supported is a single item (it should be a list)
def test_nonlist_supported(self) -> None:
    yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported: abs"""
    # Get errors from generating backend stubs with "supported" as a non-list item
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # Assert that the error message indicates "supported" should be a list
    self.assertExpectedInline(
        output_error,
        """expected "supported" to be a list, but got: abs (of type <class 'str'>)""",
    )

# supported contains an op that isn't in native_functions.yaml
def test_supported_invalid_op(self) -> None:
    yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs_BAD"""
    # Get errors from generating backend stubs with an unsupported operator
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # Assert that the error message indicates an invalid operator name
    self.assertExpectedInline(
        output_error, """Found an invalid operator name: abs_BAD"""
    )

# The backend is valid, but doesn't have a valid autograd key. They can't override autograd kernels in that case.
# Only using Vulkan here because it has a valid backend key but not an autograd key- if this changes we can update the test.
    # 定义名为 test_backend_has_no_autograd_key_but_provides_entries 的方法，无参数，无返回值
    def test_backend_has_no_autograd_key_but_provides_entries(self) -> None:
        # 定义包含 YAML 格式字符串的变量 yaml_str
        yaml_str = """\
# 设置后端为 Vulkan
backend: Vulkan
# 设置 C++ 命名空间为 torch_vulkan
cpp_namespace: torch_vulkan
# 定义支持的运算，这里仅支持 add
supported:
- add
# 调用函数获取生成的后端存根中的错误信息
output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
# 断言输出错误与预期的内联字符串匹配，这里是检查是否找到了无效的操作符名称 add
self.assertExpectedInline(
    output_error, """Found an invalid operator name: add"""
)  # noqa: B950

# 在操作符组中，当前所有操作符必须要么注册到后端，要么注册到自动微分内核中。
# 这里出现了功能不匹配的问题
def test_backend_autograd_kernel_mismatch_out_functional(self) -> None:
    # 定义 YAML 字符串
    yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
autograd:
- add.out"""
    # 调用函数获取生成的后端存根中的错误信息
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # 断言输出错误与预期的内联字符串匹配，这里是检查到了功能和 inplace 不匹配的问题
    self.assertExpectedInline(
        output_error,
        """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add_out is listed under "autograd".""",  # noqa: B950
    )

# 在操作符组中，当前所有操作符必须要么注册到后端，要么注册到自动微分内核中。
# 这里出现了功能和 inplace 不匹配的问题
def test_backend_autograd_kernel_mismatch_functional_inplace(self) -> None:
    # 定义 YAML 字符串
    yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
autograd:
- add_.Tensor"""
    # 调用函数获取生成的后端存根中的错误信息
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # 断言输出错误与预期的内联字符串匹配，这里是检查到了功能和 inplace 不匹配的问题
    self.assertExpectedInline(
        output_error,
        """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add_ is listed under "autograd".""",  # noqa: B950
    )

# 当前，同一个操作符不能同时列在 'supported' 和 'autograd' 中，
# 这意味着不能将相同的内核同时注册到 XLA 和 AutogradXLA 键中。
# 如果将来需要此功能，需要扩展代码生成。
def test_op_appears_in_supported_and_autograd_lists(self) -> None:
    # 定义 YAML 字符串
    yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
autograd:
- add.Tensor"""
    # 调用函数获取生成的后端存根中的错误信息
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # 断言输出错误与预期的内联字符串匹配，这里是检查到了操作符同时出现在 supported 和 autograd 列表中的问题
    self.assertExpectedInline(
        output_error,
        """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add is listed under "autograd".""",  # noqa: B950
    )

# 未识别的额外 YAML 键
def test_unrecognized_key(self) -> None:
    # 定义 YAML 字符串
    yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs
"""
    # 调用函数获取生成的后端存根中的错误信息
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # 断言输出错误与预期的内联字符串匹配，这里是检查到了未识别的额外 YAML 键的问题
    self.assertExpectedInline(
        output_error,
        """Unrecognized key 'abs' found in YAML."""  # noqa: B950
    )
    # 从生成的后端存根中获取错误信息
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # 断言期望的内联消息输出错误
    self.assertExpectedInline(
        output_error,
        """ contains unexpected keys: invalid_key. Only the following keys are supported: backend, class_name, cpp_namespace, extra_headers, supported, autograd, full_codegen, non_native, ir_gen, symint""",  # noqa: B950
    )

# 如果提供了 use_out_as_primary，它必须是布尔值
def test_use_out_as_primary_non_bool(self) -> None:
    yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
use_out_as_primary: frue
supported:
- abs"""
    # 从生成的后端存根中获取错误信息
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # 断言期望的内联消息输出错误
    self.assertExpectedInline(
        output_error,
        """You must provide either True or False for use_out_as_primary. Provided: frue""",
    )  # noqa: B950

# 如果提供了 device_guard，它必须是布尔值
def test_device_guard_non_bool(self) -> None:
    yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
device_guard: frue
supported:
- abs"""
    # 从生成的后端存根中获取错误信息
    output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
    # 断言期望的内联消息输出错误
    self.assertExpectedInline(
        output_error,
        """You must provide either True or False for device_guard. Provided: frue""",
    )  # noqa: B950

def test_incorrect_kernel_name(self) -> None:
    yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs
autograd:
- add.Tensor"""
    # 期望两个内核名称，并尝试使用正则表达式进行解析：
    # XLANativeFunctions::abs(...)
    # XLANativeFunctions::add(...)
    kernels_str = """\
at::Tensor& XLANativeFunctions::absWRONG(at::Tensor& self) {}
at::Tensor& XLANativeFunctions::add(at::Tensor& self) {}"""
    # 从生成的后端存根中获取错误信息，同时提供内核字符串以便检查
    output_error = self.get_errors_from_gen_backend_stubs(
        yaml_str, kernels_str=kernels_str
    )
    # 断言期望的内联消息输出错误
    self.assertExpectedInline(
        output_error,
        """\

XLANativeFunctions is missing a kernel definition for abs. We found 0 kernel(s) with that name,
but expected 1 kernel(s). The expected function schemas for the missing operator are:
at::Tensor abs(const at::Tensor & self)

""",
    )
```