# `D:\src\scipysrc\pandas\pandas\tests\util\test_deprecate_nonkeyword_arguments.py`

```
"""
Tests for the `deprecate_nonkeyword_arguments` decorator
"""

# 导入检查模块
import inspect

# 导入装饰器函数
from pandas.util._decorators import deprecate_nonkeyword_arguments

# 导入测试工具
import pandas._testing as tm

# 使用装饰器 `deprecate_nonkeyword_arguments` 对函数 `f` 进行修饰
@deprecate_nonkeyword_arguments(
    version="1.1", allowed_args=["a", "b"], name="f_add_inputs"
)
# 定义函数 `f`，接收参数 `a`、`b`、`c`、`d`
def f(a, b=0, c=0, d=0):
    return a + b + c + d

# 检查函数 `f` 的签名是否符合预期
def test_f_signature():
    assert str(inspect.signature(f)) == "(a, b=0, *, c=0, d=0)"

# 测试函数 `f` 只有一个参数的情况
def test_one_argument():
    with tm.assert_produces_warning(None):
        assert f(19) == 19

# 测试函数 `f` 有一个位置参数和一个关键字参数的情况
def test_one_and_one_arguments():
    with tm.assert_produces_warning(None):
        assert f(19, d=6) == 25

# 测试函数 `f` 有两个位置参数的情况
def test_two_arguments():
    with tm.assert_produces_warning(None):
        assert f(1, 5) == 6

# 测试函数 `f` 有两个位置参数和两个关键字参数的情况
def test_two_and_two_arguments():
    with tm.assert_produces_warning(None):
        assert f(1, 3, c=3, d=5) == 12

# 测试函数 `f` 有三个位置参数的情况
def test_three_arguments():
    with tm.assert_produces_warning(FutureWarning):
        assert f(6, 3, 3) == 12

# 测试函数 `f` 有四个位置参数的情况
def test_four_arguments():
    with tm.assert_produces_warning(FutureWarning):
        assert f(1, 2, 3, 4) == 10

# 测试函数 `f` 有三个位置参数且给出警告消息的情况
def test_three_arguments_with_name_in_warning():
    msg = (
        "Starting with pandas version 1.1 all arguments of f_add_inputs "
        "except for the arguments 'a' and 'b' will be keyword-only."
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert f(6, 3, 3) == 12

# 使用装饰器 `deprecate_nonkeyword_arguments` 对函数 `g` 进行修饰
@deprecate_nonkeyword_arguments(version="1.1")
# 定义函数 `g`，接收参数 `a` 和关键字参数 `b`、`c`、`d`
def g(a, b=0, c=0, d=0):
    with tm.assert_produces_warning(None):
        return a + b + c + d

# 检查函数 `g` 的签名是否符合预期
def test_g_signature():
    assert str(inspect.signature(g)) == "(a, *, b=0, c=0, d=0)"

# 测试函数 `g` 有一个位置参数和三个关键字参数的情况
def test_one_and_three_arguments_default_allowed_args():
    with tm.assert_produces_warning(None):
        assert g(1, b=3, c=3, d=5) == 12

# 测试函数 `g` 有三个位置参数的情况
def test_three_arguments_default_allowed_args():
    with tm.assert_produces_warning(FutureWarning):
        assert g(6, 3, 3) == 12

# 测试函数 `g` 有三个位置参数且给出警告消息的情况
def test_three_positional_argument_with_warning_message_analysis():
    msg = (
        "Starting with pandas version 1.1 all arguments of g "
        "except for the argument 'a' will be keyword-only."
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert g(6, 3, 3) == 12

# 使用装饰器 `deprecate_nonkeyword_arguments` 对函数 `h` 进行修饰
@deprecate_nonkeyword_arguments(version="1.1")
# 定义函数 `h`，只接收关键字参数 `a`、`b`、`c`、`d`
def h(a=0, b=0, c=0, d=0):
    return a + b + c + d

# 检查函数 `h` 的签名是否符合预期
def test_h_signature():
    assert str(inspect.signature(h)) == "(*, a=0, b=0, c=0, d=0)"

# 测试函数 `h` 全部为关键字参数的情况
def test_all_keyword_arguments():
    with tm.assert_produces_warning(None):
        assert h(a=1, b=2) == 3

# 测试函数 `h` 有一个位置参数的情况
def test_one_positional_argument():
    with tm.assert_produces_warning(FutureWarning):
        assert h(23) == 23

# 测试函数 `h` 有一个位置参数且给出警告消息的情况
def test_one_positional_argument_with_warning_message_analysis():
    msg = "Starting with pandas version 1.1 all arguments of h will be keyword-only."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert h(19) == 19

# 使用装饰器 `deprecate_nonkeyword_arguments` 对函数 `i` 进行修饰
@deprecate_nonkeyword_arguments(version="1.1")
# 定义函数 `i`，接收位置参数 `a`，关键字参数 `b`、`c`、`d`
def i(a=0, /, b=0, *, c=0, d=0):
    return a + b + c + d

# 检查函数 `i` 的签名是否符合预期
def test_i_signature():
    # 使用 inspect 模块的 signature 函数获取函数 i 的签名对象，然后将其转换成字符串形式
    assert str(inspect.signature(i)) == "(*, a=0, b=0, c=0, d=0)"
    # 断言：验证获取的函数签名字符串是否与指定的字符串相匹配
# 定义一个名为 Foo 的类
class Foo:
    # 使用装饰器将方法 baz 标记为废弃非关键字参数，并指定版本和允许的参数列表
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "bar"])
    # 定义方法 baz，其中 bar 是位置参数，foobar 是关键字参数，默认为 None
    def baz(self, bar=None, foobar=None): ...


# 定义测试方法 test_foo_signature，用于验证方法签名是否符合预期
def test_foo_signature():
    # 断言 Foo 类的方法 baz 的签名为 "(self, bar=None, *, foobar=None)"
    assert str(inspect.signature(Foo.baz)) == "(self, bar=None, *, foobar=None)"


# 定义测试类 test_class，用于验证废弃警告是否产生
def test_class():
    # 定义字符串消息，描述 pandas 未来版本中 Foo.baz 方法参数的变化
    msg = (
        r"In a future version of pandas all arguments of Foo\.baz "
        r"except for the argument \'bar\' will be keyword-only"
    )
    # 使用 assert_produces_warning 上下文管理器，验证调用 Foo().baz("qux", "quox") 会产生 FutureWarning 警告，并匹配消息 msg
    with tm.assert_produces_warning(FutureWarning, match=msg):
        Foo().baz("qux", "quox")
```