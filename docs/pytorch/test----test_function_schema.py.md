# `.\pytorch\test\test_function_schema.py`

```py
# Owner(s): ["module: unknown"]

# 导入PyTorch库
import torch
# 导入解析模式函数
from torch._C import parse_schema
# 导入测试相关的工具函数和类
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义一个测试类，继承自TestCase
class TestFunctionSchema(TestCase):
    
    # 测试序列化和反序列化函数架构
    def test_serialize_and_deserialize(self):
        # 获取所有注册的函数架构
        schemas = torch._C._jit_get_all_schemas()
        # 断言注册的函数架构数量大于1000个
        self.assertGreater(len(schemas), 1000)
        # 遍历每个函数架构
        for schema in schemas:
            # 解析函数架构并转换成字符串
            parsed_schema = parse_schema(str(schema))
            # 断言解析后的函数架构与原始函数架构相等
            self.assertEqual(parsed_schema, schema)
            # 断言解析后的函数架构在向后兼容性方面与原始函数架构兼容
            self.assertTrue(parsed_schema.is_backward_compatible_with(schema))

    # 测试带有输出参数的函数架构
    def test_out_schema(self):
        # 解析带有输出参数的函数架构
        schema_with_out = parse_schema(
            "any.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"
        )
        # 断言最后一个参数是否为输出参数
        self.assertTrue(schema_with_out.arguments[-1].is_out)
        
        # 解析不带输出参数的函数架构
        schema_without_out = parse_schema(
            "any.not_out(Tensor self, Tensor b) -> Tensor"
        )
        # 断言最后一个参数不是输出参数
        self.assertFalse(schema_without_out.arguments[-1].is_out)
    # 定义一个测试方法，用于测试哈希函数对不同模式的函数签名生成的哈希值是否正确
    def test_hash_schema(self):
        # 解析第一个函数签名模式并计算其哈希值
        schema1 = parse_schema("any.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
        # 解析第二个函数签名模式并计算其哈希值，应该与第一个相同
        schema2 = parse_schema("any.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
        self.assertEqual(hash(schema1), hash(schema2))

        # 解析第三个函数签名模式并计算其哈希值，与第二个应不同
        schema3 = parse_schema(
            "any.not_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"
        )
        self.assertNotEqual(hash(schema2), hash(schema3))

        # 解析第四个函数签名模式并计算其哈希值，与第二个应不同
        schema4 = parse_schema(
            "foo(Tensor self, *, int a, Tensor(a!) out) -> Tensor(a!)"
        )
        self.assertNotEqual(hash(schema2), hash(schema4))

        # 测试带有不同默认值或者关键字参数的函数签名应该生成不同的哈希值
        default_val_schema0 = parse_schema("foo(Tensor self, int a = 2) -> Tensor(a!)")
        default_val_schema1 = parse_schema("foo(Tensor self, int a = 3) -> Tensor(a!)")
        default_val_schema2 = parse_schema(
            "foo(Tensor self, *, int a = 2) -> Tensor(a!)"
        )
        self.assertNotEqual(hash(default_val_schema0), hash(default_val_schema1))
        self.assertNotEqual(hash(default_val_schema0), hash(default_val_schema2))

        # 测试带有不同别名注解的函数签名应该生成不同的哈希值
        alias_schema = parse_schema("foo(Tensor(a!) self, int a = 2) -> Tensor(a!)")
        self.assertNotEqual(hash(default_val_schema0), hash(alias_schema))
        alias_schema2 = parse_schema("foo(Tensor(b!) self, int a = 2) -> Tensor(a!)")
        self.assertNotEqual(hash(alias_schema), hash(alias_schema2))

        # 测试带有不同别名信息的函数签名应该生成不同的哈希值
        alias_schema3 = parse_schema(
            "foo(Tensor self, *, int a, int b=1, Tensor(a!) out, Tensor(b!) b) -> Tensor(a!)"
        )
        alias_schema4 = parse_schema(
            "foo(Tensor self, *, int a, int b=1, Tensor(a!) out, Tensor(b!) b) -> Tensor(b!)"
        )
        alias_schema5 = parse_schema(
            "foo(Tensor self, *, int a, int b=1, Tensor(b!) out, Tensor(a!) b) -> Tensor(a!)"
        )
        self.assertNotEqual(hash(alias_schema3), hash(alias_schema4))
        self.assertNotEqual(hash(alias_schema3), hash(alias_schema5))
    def test_backward_compatible_structure(self):
        # 解析旧模式的函数签名
        old_schema = parse_schema("any.over(Tensor self, *, Tensor b) -> Tensor")
        
        # BC: 一个没有变化的新模式
        # 解析没有变化的新模式的函数签名
        new_schema = parse_schema("any.over(Tensor self, *, Tensor b) -> Tensor")
        # 断言新模式是否与旧模式向后兼容
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        
        # No-BC: 一个具有不同名称的新模式
        # 解析具有不同名称的新模式的函数签名
        new_schema = parse_schema("any_.over(Tensor self, *, Tensor b) -> Tensor")
        # 断言新模式是否与旧模式不向后兼容
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        
        # No-BC: 一个具有不同重载名称的新模式
        # 解析具有不同重载名称的新模式的函数签名
        new_schema = parse_schema("any.other(Tensor self, *, Tensor b) -> Tensor")
        # 断言新模式是否与旧模式不向后兼容
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        
        # No-BC: 一个添加了可变参数的新模式
        # 解析添加了可变参数的新模式的函数签名
        new_schema = parse_schema("any.over(Tensor self, *, Tensor b, ...) -> Tensor")
        # 断言新模式是否与旧模式不向后兼容
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        
        # No-BC: 一个具有不同输出数量的新模式
        # 解析具有不同输出数量的新模式的函数签名
        new_schema = parse_schema(
            "any.over(Tensor self, *, Tensor b) -> (Tensor, Tensor)"
        )
        # 断言新模式是否与旧模式不向后兼容
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))

    def test_backward_compatible_outputs(self):
        # 解析旧模式的函数签名
        old_schema = parse_schema("any.over(Tensor self, *, Tensor b) -> Tensor")
        
        # No-BC: 一个新模式，输出变为可选类型
        # 解析输出变为可选类型的新模式的函数签名
        new_schema = parse_schema("any.over(Tensor self, *, Tensor b) -> Tensor?")
        # 断言新模式是否与旧模式不向后兼容
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        
        # BC: （相反的情况）输出不再是可选类型的模式
        # 断言旧模式是否与输出不再是可选类型的新模式向后兼容
        self.assertTrue(old_schema.is_backward_compatible_with(new_schema))
        
        # No-BC: 一个具有不同输出类型的新模式
        # 解析具有不同输出类型的新模式的函数签名
        new_schema = parse_schema("any.over(Tensor self, *, Tensor b) -> int")
        # 断言新模式是否与旧模式不向后兼容
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        
        # No-BC: 一个具有不同输出类型的新模式
        # 解析具有不同输出类型的新模式的函数签名
        new_schema = parse_schema("any.over(Tensor self, *, Tensor b) -> Tensor out")
        # 断言新模式是否与旧模式不向后兼容
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
    # 测试函数：测试是否向后兼容智能序列化
    def test_backward_compatible_with_smart_serialization(self):
        # 测试用例：提供了 out 参数的情况
        # 解析旧模式的函数签名
        old_schema = parse_schema(
            "foo(Tensor self, *, int a, Tensor(a!) out) -> Tensor(a!)"
        )
        # 解析新模式（保持相同 out 参数）的函数签名
        new_schema_same_out = parse_schema(
            "foo(Tensor self, *, int a, int b=1, Tensor(a!) out) -> Tensor(a!)"
        )
        # 解析新模式（默认值错误）的函数签名
        new_schema_wrong_default = parse_schema(
            "foo(Tensor self, *, int b=1, int a, Tensor(a!) out) -> Tensor(a!)"
        )
        # 解析新模式（增加了额外的 out 参数）的函数签名
        new_schema_more_out = parse_schema(
            "foo(Tensor self, *, int a, int b=1, Tensor(a!) out, Tensor(b!) b) -> Tensor(a!)"
        )
        # 解析新模式（参数顺序错误）的函数签名
        new_schema_wrong_pos = parse_schema(
            "foo(Tensor self, *, int a, int b=1, Tensor(b!) b, Tensor(a!) out) -> Tensor(a!)"
        )
        
        # 断言：新模式（保持相同 out 参数）是否向后兼容旧模式
        self.assertTrue(new_schema_same_out.is_backward_compatible_with(old_schema))
        # 断言：新模式（增加了额外的 out 参数）是否向后兼容旧模式
        self.assertTrue(new_schema_more_out.is_backward_compatible_with(old_schema))
        # 断言：新模式（默认值错误）不向后兼容旧模式
        self.assertFalse(
            new_schema_wrong_default.is_backward_compatible_with(old_schema)
        )
        # 断言：新模式（参数顺序错误）不向后兼容旧模式
        self.assertFalse(new_schema_wrong_pos.is_backward_compatible_with(old_schema))

        # 测试用例：没有提供 out 参数的情况
        # 解析旧模式的函数签名（不包含 out 参数）
        old_schema_without_arg = parse_schema("foo(Tensor self, int a, int b=1) -> int")
        # 解析新模式（不包含 out 参数）的函数签名
        new_schema_without_arg = parse_schema(
            "foo(Tensor self, int a, int b=1, int c=2) -> int"
        )
        # 解析新模式（不包含 out 参数，并且有多个默认值）的函数签名
        new_schema_without_arg_multiple_default = parse_schema(
            "foo(Tensor self, int a, int b=1, int c=2, int d=3) -> int"
        )
        # 解析新模式（不包含 out 参数，参数顺序错误）的函数签名
        new_schema_without_arg_wrong_pos = parse_schema(
            "foo(Tensor self, int a, int c=2, int b=1) -> int"
        )
        
        # 断言：新模式（不包含 out 参数）是否向后兼容旧模式（不包含 out 参数）
        self.assertTrue(
            new_schema_without_arg.is_backward_compatible_with(old_schema_without_arg)
        )
        # 断言：新模式（不包含 out 参数，并且有多个默认值）是否向后兼容旧模式（不包含 out 参数）
        self.assertTrue(
            new_schema_without_arg_multiple_default.is_backward_compatible_with(
                old_schema_without_arg
            )
        )
        # 断言：新模式（不包含 out 参数，参数顺序错误）不向后兼容旧模式（不包含 out 参数）
        self.assertFalse(
            new_schema_without_arg_wrong_pos.is_backward_compatible_with(
                old_schema_without_arg
            )
        )

    # 测试函数：测试字符串可选参数默认值
    def test_string_optional_parameter_default_value(self):
        # 解析模式 A 的函数签名
        schema_a = parse_schema('example::op(str? order="NCHW") -> (Tensor)')
        # 重新解析模式 A 的字符串表示
        schema_b = parse_schema(str(schema_a))
        # 断言：模式 A 和重新解析的模式 A 是否相等
        self.assertEqual(schema_a, schema_b)
    def test_forward_compatible_arguments_real_use_case(self):
        # 解析旧版本的函数签名字符串，返回一个旧版本的函数模式对象
        old_slice_schema = parse_schema(
            "slice(Tensor(a) self, int dim=0, int start=0, int end=0, int step=1) -> Tensor(a)"
        )
        # 解析新版本的函数签名字符串，返回一个新版本的函数模式对象
        new_slice_schema = parse_schema(
            "slice(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)"
        )
        # 检查新旧版本函数模式对象是否向前兼容，并返回结果和原因
        is_fc, reason = new_slice_schema.check_forward_compatible_with(old_slice_schema)
        # 断言新旧版本函数模式对象不向前兼容
        self.assertFalse(is_fc)
        # 断言原因是 'start' 参数与旧版本的函数模式不兼容
        self.assertEqual(
            reason,
            "'start' is not forward compatible with the older version of the schema",
        )

    def test_forward_compatible_arguments_with_out(self):
        # 解析旧版本的函数签名字符串，返回一个旧版本的函数模式对象
        old_schema = parse_schema(
            "any(Tensor self, *, int a, int b=1, Tensor(a!) out) -> Tensor(a!)"
        )
        # 解析新版本的函数签名字符串，返回一个新版本的函数模式对象
        new_schema = parse_schema(
            "any(Tensor self, *, int a, Tensor(a!) out) -> Tensor(a!)"
        )
        # 检查新旧版本函数模式对象是否向前兼容，并返回结果
        is_fc, _ = new_schema.check_forward_compatible_with(old_schema)
        # 断言新旧版本函数模式对象向前兼容
        self.assertTrue(is_fc)
        
        # 更新新版本的函数签名字符串，增加一个额外的参数 'c'
        new_schema = parse_schema(
            "any(Tensor self, *, int a, int b=1, int c=1, Tensor(a!) out) -> Tensor(a!)"
        )
        # 再次检查新旧版本函数模式对象是否向前兼容，并返回结果
        is_fc, _ = new_schema.check_forward_compatible_with(old_schema)
        # 断言新旧版本函数模式对象向前兼容
        self.assertTrue(is_fc)
        
        # 更新新版本的函数签名字符串，增加一个新的输出参数 'd'
        new_schema = parse_schema(
            "any(Tensor self, *, int a, Tensor(d!) d, int b=1, Tensor(a!) out) -> Tensor(a!)"
        )
        # 再次检查新旧版本函数模式对象是否向前兼容，并返回结果和原因
        is_fc, reason = new_schema.check_forward_compatible_with(old_schema)
        # 断言新旧版本函数模式对象不向前兼容
        self.assertFalse(is_fc)
        # 断言原因是 "函数模式应该具有相同数量的输出参数"
        self.assertEqual(
            reason, "Function schema should have the same number of out arguments"
        )

    def test_schema_error(self):
        # 使用断言上下文管理器，断言运行时错误中包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError, r"schemas with vararg \(...\) can't have default value args"
        ):
            # 解析特定的函数签名字符串，引发运行时错误
            schema = parse_schema("any.foo(int arg1, int arg2=0, ...)")

    def test_tensor_list_alias_annotation_properly_parsed(self):
        # 定义一个特定的函数签名字符串
        schema_str = "foo(Tensor self, *, Tensor(a!)[] out) -> ()"
        # 解析函数签名字符串，返回一个函数模式对象
        schema = parse_schema(schema_str)
        # 断言函数模式对象的最后一个参数的别名信息指示是写操作
        self.assertTrue(schema.arguments[-1].alias_info.is_write)
        # 断言函数模式对象的字符串表示与原始字符串相同
        self.assertEqual(str(schema), schema_str)
    # 定义测试函数，验证张量选项参数是否被正确解析
    def test_tensor_option_arguments_properly_parsed(self):
        # 定义表示函数签名的字符串
        schema_str = (
            "_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, "
            "bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor"
        )
        # 解析函数签名字符串，返回一个函数模式对象
        schema = parse_schema(schema_str)
        # 验证 MemoryFormat? 的假类型为 int?
        self.assertEqual(schema.arguments[-1].type.str(), "int?")
        # 验证 Layout? 的假类型为 int?
        self.assertEqual(schema.arguments[2].type.str(), "int?")
        # 验证 Device? 的假类型为 Device?
        self.assertEqual(schema.arguments[3].type.str(), "Device?")
        # 打印函数模式对象的实际类型
        self.assertEqual(str(schema), schema_str)

    # 定义测试函数，验证符号整数参数是否被正确解析
    def test_sym_int_argument_properly_parsed(self):
        # 定义表示函数签名的字符串
        schema_str = "sym_size.int(Tensor self, int dim) -> SymInt"
        # 解析函数签名字符串，返回一个函数模式对象
        schema = parse_schema(schema_str)
        # 验证 SymInt 的假类型为 int
        self.assertEqual(schema.returns[-1].type.str(), "int")
        # 验证 SymInt 的实际类型为 SymInt
        self.assertEqual(schema.returns[-1].real_type.str(), "SymInt")
        # 打印函数模式对象的实际类型
        self.assertEqual(str(schema), schema_str)
# 如果当前脚本被直接执行（而不是被导入为模块），则执行下面的代码
if __name__ == "__main__":
    # 调用名为 run_tests 的函数来执行测试代码
    run_tests()
```