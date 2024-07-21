# `.\pytorch\test\onnx\internal\test_registraion.py`

```py
# Owner(s): ["module: onnx"]
"""Unit tests for the internal registration wrapper module."""

# 导入必要的模块和类
from typing import Sequence

from torch.onnx import errors  # 导入 torch.onnx 的 errors 模块
from torch.onnx._internal import registration  # 导入 torch.onnx._internal 的 registration 模块
from torch.testing._internal import common_utils  # 导入 torch.testing._internal 的 common_utils 模块

# 使用装饰器实例化参数化测试类
@common_utils.instantiate_parametrized_tests
class TestGlobalHelpers(common_utils.TestCase):
    # 参数化测试函数，测试不同输入条件下的结果
    @common_utils.parametrize(
        "available_opsets, target, expected",
        [
            ((7, 8, 9, 10, 11), 16, 11),
            ((7, 8, 9, 10, 11), 11, 11),
            ((7, 8, 9, 10, 11), 10, 10),
            ((7, 8, 9, 10, 11), 9, 9),
            ((7, 8, 9, 10, 11), 8, 8),
            ((7, 8, 9, 10, 11), 7, 7),
            ((9, 10, 16), 16, 16),
            ((9, 10, 16), 15, 10),
            ((9, 10, 16), 10, 10),
            ((9, 10, 16), 9, 9),
            ((9, 10, 16), 8, 9),
            ((9, 10, 16), 7, 9),
            ((7, 9, 10, 16), 16, 16),
            ((7, 9, 10, 16), 10, 10),
            ((7, 9, 10, 16), 9, 9),
            ((7, 9, 10, 16), 8, 9),
            ((7, 9, 10, 16), 7, 7),
            ([17], 16, None),  # 新的操作在 opset 17 中添加
            ([9], 9, 9),
            ([9], 8, 9),
            ([], 16, None),
            ([], 9, None),
            ([], 8, None),
            # 当 target >= 9 时，opset 1 注册的操作作为回退选项
            ([1], 16, 1),
        ],
    )
    def test_dispatch_opset_version_returns_correct_version(
        self, available_opsets: Sequence[int], target: int, expected: int
    ):
        actual = registration._dispatch_opset_version(target, available_opsets)  # 调用 _dispatch_opset_version 方法
        self.assertEqual(actual, expected)


class TestOverrideDict(common_utils.TestCase):
    def setUp(self):
        self.override_dict: registration.OverrideDict[
            str, int
        ] = registration.OverrideDict()  # 初始化 OverrideDict 对象，指定键类型为 str，值类型为 int

    # 测试在没有覆盖时返回基本值
    def test_get_item_returns_base_value_when_no_override(self):
        self.override_dict.set_base("a", 42)  # 设置键 'a' 的基本值为 42
        self.override_dict.set_base("b", 0)   # 设置键 'b' 的基本值为 0

        self.assertEqual(self.override_dict["a"], 42)  # 断言获取键 'a' 的值为 42
        self.assertEqual(self.override_dict["b"], 0)   # 断言获取键 'b' 的值为 0
        self.assertEqual(len(self.override_dict), 2)   # 断言字典长度为 2

    # 测试在覆盖时返回覆盖的值
    def test_get_item_returns_overridden_value_when_override(self):
        self.override_dict.set_base("a", 42)  # 设置键 'a' 的基本值为 42
        self.override_dict.set_base("b", 0)   # 设置键 'b' 的基本值为 0
        self.override_dict.override("a", 100)  # 覆盖键 'a' 的值为 100
        self.override_dict.override("c", 1)    # 覆盖键 'c' 的值为 1

        self.assertEqual(self.override_dict["a"], 100)  # 断言获取键 'a' 的值为 100
        self.assertEqual(self.override_dict["b"], 0)    # 断言获取键 'b' 的值为 0
        self.assertEqual(self.override_dict["c"], 1)    # 断言获取键 'c' 的值为 1
        self.assertEqual(len(self.override_dict), 3)    # 断言字典长度为 3

    # 测试在键不存在时抛出 KeyError
    def test_get_item_raises_key_error_when_not_found(self):
        self.override_dict.set_base("a", 42)  # 设置键 'a' 的基本值为 42

        with self.assertRaises(KeyError):  # 断言抛出 KeyError 异常
            self.override_dict["nonexistent_key"]
    # 测试方法：测试当键被覆盖时，get方法是否返回覆盖后的值
    def test_get_returns_overridden_value_when_override(self):
        # 设置基础值"a"为42
        self.override_dict.set_base("a", 42)
        # 设置基础值"b"为0
        self.override_dict.set_base("b", 0)
        # 覆盖键"a"的值为100
        self.override_dict.override("a", 100)
        # 覆盖键"c"的值为1
        self.override_dict.override("c", 1)

        # 断言获取键"a"的值为100
        self.assertEqual(self.override_dict.get("a"), 100)
        # 断言获取键"b"的值为0
        self.assertEqual(self.override_dict.get("b"), 0)
        # 断言获取键"c"的值为1
        self.assertEqual(self.override_dict.get("c"), 1)
        # 断言字典长度为3
        self.assertEqual(len(self.override_dict), 3)

    # 测试方法：测试当键不存在时，get方法是否返回None
    def test_get_returns_none_when_not_found(self):
        # 设置基础值"a"为42
        self.override_dict.set_base("a", 42)

        # 断言获取不存在键"nonexistent_key"的值为None
        self.assertEqual(self.override_dict.get("nonexistent_key"), None)

    # 测试方法：测试当键在基础值中时，in_base方法是否返回True
    def test_in_base_returns_true_for_base_value(self):
        # 设置基础值"a"为42
        self.override_dict.set_base("a", 42)
        # 设置基础值"b"为0
        self.override_dict.set_base("b", 0)
        # 覆盖键"a"的值为100
        self.override_dict.override("a", 100)
        # 覆盖键"c"的值为1
        self.override_dict.override("c", 1)

        # 断言键"a"在字典中
        self.assertIn("a", self.override_dict)
        # 断言键"b"在字典中
        self.assertIn("b", self.override_dict)
        # 断言键"c"在字典中
        self.assertIn("c", self.override_dict)

        # 断言键"a"在基础值中为True
        self.assertTrue(self.override_dict.in_base("a"))
        # 断言键"b"在基础值中为True
        self.assertTrue(self.override_dict.in_base("b"))
        # 断言键"c"在基础值中为False
        self.assertFalse(self.override_dict.in_base("c"))
        # 断言不存在键"nonexistent_key"在基础值中为False
        self.assertFalse(self.override_dict.in_base("nonexistent_key"))

    # 测试方法：测试当键被覆盖时，overridden方法是否返回True
    def test_overridden_returns_true_for_overridden_value(self):
        # 设置基础值"a"为42
        self.override_dict.set_base("a", 42)
        # 设置基础值"b"为0
        self.override_dict.set_base("b", 0)
        # 覆盖键"a"的值为100
        self.override_dict.override("a", 100)
        # 覆盖键"c"的值为1
        self.override_dict.override("c", 1)

        # 断言键"a"被覆盖为True
        self.assertTrue(self.override_dict.overridden("a"))
        # 断言键"b"被覆盖为False
        self.assertFalse(self.override_dict.overridden("b"))
        # 断言键"c"被覆盖为True
        self.assertTrue(self.override_dict.overridden("c"))
        # 断言不存在键"nonexistent_key"被覆盖为False
        self.assertFalse(self.override_dict.overridden("nonexistent_key"))

    # 测试方法：测试remove_override方法是否正确移除覆盖的值
    def test_remove_override_removes_overridden_value(self):
        # 设置基础值"a"为42
        self.override_dict.set_base("a", 42)
        # 设置基础值"b"为0
        self.override_dict.set_base("b", 0)
        # 覆盖键"a"的值为100
        self.override_dict.override("a", 100)
        # 覆盖键"c"的值为1
        self.override_dict.override("c", 1)

        # 断言获取键"a"的值为100
        self.assertEqual(self.override_dict["a"], 100)
        # 断言获取键"c"的值为1
        self.assertEqual(self.override_dict["c"], 1)

        # 移除键"a"的覆盖
        self.override_dict.remove_override("a")
        # 移除键"c"的覆盖
        self.override_dict.remove_override("c")

        # 断言获取键"a"的值为42（恢复到基础值）
        self.assertEqual(self.override_dict["a"], 42)
        # 断言获取不存在键"c"的值为None
        self.assertEqual(self.override_dict.get("c"), None)
        # 断言键"a"不再被覆盖
        self.assertFalse(self.override_dict.overridden("a"))
        # 断言键"c"不再被覆盖
        self.assertFalse(self.override_dict.overridden("c"))

    # 测试方法：测试remove_override方法是否正确移除整个键及其覆盖值
    def test_remove_override_removes_overridden_key(self):
        # 覆盖键"a"的值为100
        self.override_dict.override("a", 100)
        # 断言获取键"a"的值为100
        self.assertEqual(self.override_dict["a"], 100)
        # 断言字典长度为1
        self.assertEqual(len(self.override_dict), 1)

        # 移除键"a"的覆盖
        self.override_dict.remove_override("a")

        # 断言字典长度为0
        self.assertEqual(len(self.override_dict), 0)
        # 断言键"a"不再在字典中
        self.assertNotIn("a", self.override_dict)
    # 测试覆盖字典 OverrideDict 中键值对顺序，无论插入顺序如何，覆盖的键值对优先
    def test_overriden_key_precededs_base_key_regardless_of_insert_order(self):
        # 设置键"a"的基础值为42
        self.override_dict.set_base("a", 42)
        # 覆盖键"a"的值为100
        self.override_dict.override("a", 100)
        # 再次设置键"a"的基础值为0
        self.override_dict.set_base("a", 0)
    
        # 断言键"a"的值为100，即覆盖值
        self.assertEqual(self.override_dict["a"], 100)
        # 断言字典长度为1
        self.assertEqual(len(self.override_dict), 1)
    
    # 测试 OverrideDict 当非空时布尔值为真
    def test_bool_is_true_when_not_empty(self):
        # 如果 OverrideDict 为空，则失败
        if self.override_dict:
            self.fail("OverrideDict should be false when empty")
        # 向 OverrideDict 中添加键"a"的覆盖值为1
        self.override_dict.override("a", 1)
        # 如果 OverrideDict 不为空，则失败
        if not self.override_dict:
            self.fail("OverrideDict should be true when not empty")
        # 设置键"a"的基础值为42
        self.override_dict.set_base("a", 42)
        # 如果 OverrideDict 不为空，则失败
        if not self.override_dict:
            self.fail("OverrideDict should be true when not empty")
        # 移除键"a"的覆盖值
        self.override_dict.remove_override("a")
        # 如果 OverrideDict 不为空，则失败
        if not self.override_dict:
            self.fail("OverrideDict should be true when not empty")
# 测试用例类，用于测试注册装饰器的功能
class TestRegistrationDecorators(common_utils.TestCase):

    # 在每个测试方法执行后执行，用于清理注册信息
    def tearDown(self) -> None:
        registration.registry._registry.pop("test::test_op", None)

    # 测试注册 ONNX 符号函数是否正常注册
    def test_onnx_symbolic_registers_function(self):
        # 断言在注册前 "test::test_op" 未注册
        self.assertFalse(registration.registry.is_registered_op("test::test_op", 9))

        # 使用装饰器注册 ONNX 符号函数
        @registration.onnx_symbolic("test::test_op", opset=9)
        def test(g, x):
            return g.op("test", x)

        # 断言注册后 "test::test_op" 已经注册
        self.assertTrue(registration.registry.is_registered_op("test::test_op", 9))
        # 获取注册的函数组
        function_group = registration.registry.get_function_group("test::test_op")
        assert function_group is not None
        # 断言注册的函数与定义的函数相同
        self.assertEqual(function_group.get(9), test)

    # 测试当提供了装饰器时，ONNX 符号函数是否正确注册
    def test_onnx_symbolic_registers_function_applied_decorator_when_provided(self):
        wrapper_called = False

        # 定义一个装饰器函数
        def decorator(func):
            def wrapper(*args, **kwargs):
                nonlocal wrapper_called
                wrapper_called = True
                return func(*args, **kwargs)
            return wrapper

        # 使用装饰器注册 ONNX 符号函数
        @registration.onnx_symbolic("test::test_op", opset=9, decorate=[decorator])
        def test():
            return

        # 获取注册的函数组
        function_group = registration.registry.get_function_group("test::test_op")
        assert function_group is not None
        # 获取注册的函数
        registered_function = function_group[9]
        # 断言装饰器尚未调用
        self.assertFalse(wrapper_called)
        # 调用注册的函数
        registered_function()
        # 断言装饰器已经调用
        self.assertTrue(wrapper_called)

    # 测试当尝试重复注册同一函数时是否会引发警告
    def test_onnx_symbolic_raises_warning_when_overriding_function(self):
        # 断言在注册前 "test::test_op" 未注册
        self.assertFalse(registration.registry.is_registered_op("test::test_op", 9))

        # 使用装饰器注册第一个 ONNX 符号函数
        @registration.onnx_symbolic("test::test_op", opset=9)
        def test1():
            return

        # 使用装饰器注册第二个同名的 ONNX 符号函数，预期会引发警告
        with self.assertWarnsRegex(
            errors.OnnxExporterWarning,
            "Symbolic function 'test::test_op' already registered",
        ):
            @registration.onnx_symbolic("test::test_op", opset=9)
            def test2():
                return

    # 测试自定义 ONNX 符号函数是否正常注册
    def test_custom_onnx_symbolic_registers_custom_function(self):
        # 断言在注册前 "test::test_op" 未注册
        self.assertFalse(registration.registry.is_registered_op("test::test_op", 9))

        # 使用自定义装饰器注册 ONNX 符号函数
        @registration.custom_onnx_symbolic("test::test_op", opset=9)
        def test(g, x):
            return g.op("test", x)

        # 断言注册后 "test::test_op" 已经注册
        self.assertTrue(registration.registry.is_registered_op("test::test_op", 9))
        # 获取注册的函数组
        function_group = registration.registry.get_function_group("test::test_op")
        assert function_group is not None
        # 断言注册的函数与定义的函数相同
        self.assertEqual(function_group.get(9), test)
    # 定义一个测试方法，用于测试自定义的 ONNX 符号覆盖现有函数
    def test_custom_onnx_symbolic_overrides_existing_function(self):
        # 断言名为 "test::test_op" 的操作在 opset=9 中尚未注册
        self.assertFalse(registration.registry.is_registered_op("test::test_op", 9))

        # 使用装饰器注册一个名为 "test::test_op" 的 ONNX 符号化函数，返回固定字符串 "original"
        @registration.onnx_symbolic("test::test_op", opset=9)
        def test_original():
            return "original"

        # 断言名为 "test::test_op" 的操作在 opset=9 中已经注册成功
        self.assertTrue(registration.registry.is_registered_op("test::test_op", 9))

        # 使用装饰器注册一个自定义的 ONNX 符号化函数，覆盖了之前的同名函数，返回固定字符串 "custom"
        @registration.custom_onnx_symbolic("test::test_op", opset=9)
        def test_custom():
            return "custom"

        # 获取名为 "test::test_op" 的函数组，确保它不为 None
        function_group = registration.registry.get_function_group("test::test_op")
        assert function_group is not None

        # 断言在 opset=9 中，名为 "test::test_op" 的函数组中包含 test_custom 函数
        self.assertEqual(function_group.get(9), test_custom)
if __name__ == "__main__":
    # 如果当前模块作为主程序执行（而不是被导入到其他模块中执行）
    common_utils.run_tests()
    # 调用 common_utils 模块中的 run_tests 函数，用于执行测试用例
```