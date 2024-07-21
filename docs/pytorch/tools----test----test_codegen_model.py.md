# `.\pytorch\tools\test\test_codegen_model.py`

```py
# Owner(s): ["module: codegen"]

# 导入所需的模块和类
import textwrap  # 导入文本包装模块，用于格式化文本输出
import unittest  # 导入单元测试框架模块
from typing import cast  # 导入类型提示模块中的类型强制转换功能

import expecttest  # 导入期望测试支持模块
import yaml  # 导入 YAML 解析器

import torchgen.dest as dest  # 导入目标生成模块中的目标命名空间
import torchgen.gen as gen  # 导入生成模块中的生成命名空间
from torchgen.gen import LineLoader, parse_native_yaml_struct  # 从生成模块导入行加载器和解析本地 YAML 结构函数
from torchgen.model import (  # 从模型模块导入多个类
    Annotation,
    CustomClassType,
    DispatchKey,
    NativeFunctionsGroup,
    Type,
)


class TestCodegenModel(expecttest.TestCase):
    # 自定义断言函数：验证解析错误行内信息是否与预期一致
    def assertParseErrorInline(self, yaml_str: str, expect: str) -> None:
        es = yaml.load(yaml_str, Loader=LineLoader)  # 解析 YAML 字符串为 Python 对象
        try:
            parse_native_yaml_struct(es, set())  # 尝试解析本地 YAML 结构
        except AssertionError as e:
            # hack to strip out the context
            msg, _ = str(e).split("  in ", 2)  # 提取错误消息并去除上下文信息
            self.assertExpectedInline("\n".join(textwrap.wrap(msg)), expect, skip=1)  # 断言错误消息与预期的行内消息一致
            return
        self.fail(msg="Did not raise when expected to")  # 若没有抛出预期的异常，则测试失败

    # 自定义断言函数：验证 ufunc 错误行内信息是否与预期一致
    def assertUfuncErrorInline(self, yaml_str: str, expect: str) -> None:
        es = yaml.load(yaml_str, Loader=LineLoader)  # 解析 YAML 字符串为 Python 对象
        parsed_yaml = parse_native_yaml_struct(es, set())  # 解析本地 YAML 结构
        native_functions, backend_indices = (
            parsed_yaml.native_functions,
            parsed_yaml.backend_indices,
        )
        grouped_native_functions = gen.get_grouped_native_functions(native_functions)  # 获取分组后的本地函数
        assert len(grouped_native_functions) == 1  # 断言仅有一个本地函数分组
        g = grouped_native_functions[0]  # 获取第一个本地函数分组
        assert isinstance(g, NativeFunctionsGroup)  # 断言 g 是 NativeFunctionsGroup 类的实例
        assert g.out.ufunc_inner_loop  # 断言 g 包含 ufunc 内部循环
        # 这里不是 ufunc 代码生成本身，但执行了一些基本的 ufunc 生成的健全性测试
        gen.compute_meta_function_declaration(g)  # 计算元函数声明
        dest.compute_native_function_declaration(g, backend_indices[DispatchKey.CPU])  # 计算 CPU 平台的本地函数声明
        dest.compute_native_function_declaration(g, backend_indices[DispatchKey.CUDA])  # 计算 CUDA 平台的本地函数声明
        try:
            # 实际的 ufunc 生成操作
            dest.compute_ufunc_cpu(g)  # 生成 CPU 平台的 ufunc
            dest.compute_ufunc_cpu_kernel(g)  # 生成 CPU 平台的 ufunc 内核
            dest.compute_ufunc_cuda(g)  # 生成 CUDA 平台的 ufunc
        except AssertionError as e:
            # hack to strip out the context
            msg, _ = str(e).split("  in ", 2)  # 提取错误消息并去除上下文信息
            self.assertExpectedInline("\n".join(textwrap.wrap(msg)), expect, skip=1)  # 断言错误消息与预期的行内消息一致
            return
        self.fail(msg="Did not raise when expected to")  # 若没有抛出预期的异常，则测试失败

    # 定义 binop_out 变量，包含二元操作函数的输出声明
    binop_out = (
        "func: binop.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"
    )
    # 定义 ti_binop_out 变量，包含结构化二元操作函数的输出声明及其继承关系
    ti_binop_out = f"""{binop_out}
  structured: True
  structured_inherits: TensorIteratorBase"""
    # 定义 ti_binop 变量，包含结构化二元操作函数的声明及其代理关系
    ti_binop = """func: binop(Tensor self, Tensor other) -> Tensor
  structured_delegate: binop.out
"""

    # 定义 ti_unop_out 变量，包含结构化一元操作函数的输出声明及其继承关系
    ti_unop_out = """func: unop.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase"""
    # 定义 ti_unop 变量，包含结构化一元操作函数的声明及其代理关系
    ti_unop = """func: unop(Tensor self) -> Tensor
  structured_delegate: unop.out
"""
    # 定义一个测试方法 `test_nonstructured_ufunc`，该方法没有参数限制，也没有返回值
    def test_nonstructured_ufunc(self) -> None:
        # 使用 f-string 构建一个多行的 YAML 字符串，赋值给变量 `yaml_str`
        yaml_str = f"""\
    def test_overlapping_ufunc_and_dispatch(self) -> None:
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
  dispatch:
    CPU: binop_cpu
"""
        # 断言解析错误，因为 ufunc 不应该有显式的 CPU 分发条目
        self.assertParseErrorInline(
            yaml_str,
            """\
ufunc should not have explicit dispatch entry for CPU""",
        )

    @unittest.expectedFailure
    def test_scalaronly_shadowed(self) -> None:
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
    ScalarOnly: binop (Bool)
"""
        # 断言解析错误，因为 ScalarOnly 不能与 Generic 同时存在
        self.assertParseErrorInline(
            yaml_str,
            """\
""",
        )

    def test_conflicting_ufunc(self) -> None:
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
    ScalarOnly: binop_scalar (Bool)
- {self.ti_binop}
"""
        # 断言 ufunc 错误，因为 ScalarOnly 和 Generic 必须有相同的 ufunc 名称
        self.assertUfuncErrorInline(
            yaml_str,
            """\
ScalarOnly and Generic must have same ufunc name""",
        )

    def test_invalid_cudafunctoronself_for_binary_op(self) -> None:
        yaml_str = f"""\
- {self.ti_unop_out}
  ufunc_inner_loop:
    Generic: unop (All)
    CUDAFunctorOnSelf: unop_self_cuda (All)
- {self.ti_unop}
"""
        # 断言 ufunc 错误，因为不能在非二元函数上使用 CUDAFunctorOnSelf
        self.assertUfuncErrorInline(
            yaml_str,
            """\
cannot use CUDAFunctorOnSelf on non-binary function""",
        )

    def test_parse_custom_class_type(self) -> None:
        custom_class_name = "namespace_foo.class_bar"
        custom_class_name_with_prefix = f"__torch__.torch.classes.{custom_class_name}"
        custom_class_type = cast(
            CustomClassType, Type.parse(custom_class_name_with_prefix)
        )
        # 断言自定义类类型解析是否正确
        self.assertTrue(isinstance(custom_class_type, CustomClassType))
        self.assertEqual(custom_class_name, custom_class_type.class_name)
        self.assertEqual(custom_class_name_with_prefix, str(custom_class_type))


class TestAnnotation(expecttest.TestCase):
    def test_single_alias_no_write(self) -> None:
        a = Annotation.parse("a")
        # 断言单个别名解析，无写入操作
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertFalse(a.is_write)
        self.assertEqual(a.alias_set_after, tuple())

    def test_single_alias_is_write(self) -> None:
        a = Annotation.parse("a!")
        # 断言单个别名解析，有写入操作
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, tuple())

    def test_single_alias_is_write_to_wildcard(self) -> None:
        a = Annotation.parse("a! -> *")
        # 断言单个别名解析，有写入操作到通配符
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, tuple("*"))

    def test_alias_set(self) -> None:
        a = Annotation.parse("a|b")
        # 断言别名集合解析
        self.assertEqual(a.alias_set, ("a", "b"))
    # 定义一个测试方法，用于测试别名设置是否会引发异常
    def test_alias_set_is_write_raises_exception(self) -> None:
        # 使用断言检查是否抛出指定异常类型和消息
        with self.assertRaisesRegex(
            AssertionError, r"alias set larger than 1 is not mutable"
        ):
            # 调用 Annotation 类的 parse 方法解析字符串 "a|b!"
            Annotation.parse("a|b!")

    # 定义一个测试方法，用于测试单个别名是否正确写入别名集合
    def test_single_alias_is_write_to_alias_set(self) -> None:
        # 解析字符串 "a! -> a|b" 并赋值给变量 a
        a = Annotation.parse("a! -> a|b")
        # 断言别名集合是否为 ('a')
        self.assertEqual(a.alias_set, tuple("a"))
        # 断言是否写操作
        self.assertTrue(a.is_write)
        # 断言修改后的别名集合是否为 ("a", "b")
        self.assertEqual(a.alias_set_after, ("a", "b"))

    # 定义一个测试方法，用于测试同时有多个别名集合大于 1 的情况是否会引发异常
    def test_before_and_after_alias_set_larger_than_1_raises_exception(self) -> None:
        # 使用断言检查是否抛出指定异常类型和消息
        with self.assertRaisesRegex(
            AssertionError,
            r"before alias set and after alias set cannot be larger than 1 at the same time",
        ):
            # 调用 Annotation 类的 parse 方法解析字符串 "a|b -> c|d"
            Annotation.parse("a|b -> c|d")
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 运行单元测试主程序
    unittest.main()
```