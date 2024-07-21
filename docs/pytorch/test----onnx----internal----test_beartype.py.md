# `.\pytorch\test\onnx\internal\test_beartype.py`

```py
# Owner(s): ["module: onnx"]
"""Unit tests for the internal beartype wrapper module."""

# 导入 unittest 模块
import unittest

# 从 torch.onnx._internal 中导入 _beartype
from torch.onnx._internal import _beartype
# 从 torch.testing._internal 中导入 common_utils
from torch.testing._internal import common_utils


# 检查是否安装了 beartype
def beartype_installed():
    try:
        import beartype  # noqa: F401
    except ImportError:
        return False
    return True


# 如果未安装 beartype，则跳过测试
def skip_if_beartype_not_installed(test_case):
    return unittest.skipIf(not beartype_installed(), "beartype is not installed")(
        test_case
    )


# 函数带有类型提示的示例，输入和输出都是整数类型
def func_with_type_hint(x: int) -> int:
    return x


# 函数带有错误的类型提示，输入是整数类型，但输出被标注为字符串类型
def func_with_incorrect_type_hint(x: int) -> str:
    return x  # type: ignore[return-value]


# 使用 common_utils 中的装饰器，为测试类实例化参数化测试
@common_utils.instantiate_parametrized_tests
class TestBeartype(common_utils.TestCase):
    # 测试：当 beartype 被禁用时，_create_beartype_decorator 返回空操作装饰器
    def test_create_beartype_decorator_returns_no_op_decorator_when_disabled(self):
        decorator = _beartype._create_beartype_decorator(
            _beartype.RuntimeTypeCheckState.DISABLED,
        )
        decorated = decorator(func_with_incorrect_type_hint)
        decorated("string_input")  # type: ignore[arg-type]

    # 装饰器测试：当警告开启时，_create_beartype_decorator 应发出警告
    @skip_if_beartype_not_installed
    def test_create_beartype_decorator_warns_when_warnings(self):
        decorator = _beartype._create_beartype_decorator(
            _beartype.RuntimeTypeCheckState.WARNINGS,
        )
        decorated = decorator(func_with_incorrect_type_hint)
        with self.assertWarns(_beartype.CallHintViolationWarning):
            decorated("string_input")  # type: ignore[arg-type]

    # 装饰器测试：当错误开启时，_create_beartype_decorator 应抛出异常
    @common_utils.parametrize("arg", [1, "string_input"])
    @skip_if_beartype_not_installed
    def test_create_beartype_decorator_errors_when_errors(self, arg):
        import beartype

        decorator = _beartype._create_beartype_decorator(
            _beartype.RuntimeTypeCheckState.ERRORS,
        )
        decorated = decorator(func_with_incorrect_type_hint)
        with self.assertRaises(beartype.roar.BeartypeCallHintViolation):
            decorated(arg)

    # 装饰器测试：验证警告模式下，函数只调用一次
    @skip_if_beartype_not_installed
    def test_create_beartype_decorator_warning_calls_function_once(self):
        call_count = 0

        def func_with_incorrect_type_hint_and_side_effect(x: int) -> str:
            nonlocal call_count
            call_count += 1
            return x  # type: ignore[return-value]

        decorator = _beartype._create_beartype_decorator(
            _beartype.RuntimeTypeCheckState.WARNINGS,
        )
        decorated = decorator(func_with_incorrect_type_hint_and_side_effect)
        decorated("string_input")  # type: ignore[arg-type]
        self.assertEqual(call_count, 1)
        decorated(1)
        # 返回值违反了类型提示，但函数只调用了一次
        self.assertEqual(call_count, 2)


# 如果直接运行此脚本，则执行测试
if __name__ == "__main__":
    common_utils.run_tests()
```