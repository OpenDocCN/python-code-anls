# `D:\src\scipysrc\pandas\pandas\tests\util\test_assert_attr_equal.py`

```
# 引入 SimpleNamespace 类，用于创建一个简单的命名空间对象
from types import SimpleNamespace

# 引入 pytest 模块，用于编写和运行测试用例
import pytest

# 从 pandas 库中引入 is_float 函数，用于检查一个对象是否为浮点数
from pandas.core.dtypes.common import is_float

# 引入 pandas 测试模块，命名为 tm，用于执行测试辅助功能
import pandas._testing as tm

# 定义一个测试函数，用于测试 assert_attr_equal 函数的行为
def test_assert_attr_equal(nulls_fixture):
    # 创建一个 SimpleNamespace 对象 obj，并将 nulls_fixture 赋给其属性 na_value
    obj = SimpleNamespace()
    obj.na_value = nulls_fixture
    
    # 调用 tm.assert_attr_equal 函数，断言 obj 和 obj 本身的 na_value 属性相等
    tm.assert_attr_equal("na_value", obj, obj)

# 定义另一个测试函数，用于测试 assert_attr_equal 函数处理不同 nulls_fixture 的情况
def test_assert_attr_equal_different_nulls(nulls_fixture, nulls_fixture2):
    # 创建第一个 SimpleNamespace 对象 obj，并将 nulls_fixture 赋给其属性 na_value
    obj = SimpleNamespace()
    obj.na_value = nulls_fixture

    # 创建第二个 SimpleNamespace 对象 obj2，并将 nulls_fixture2 赋给其属性 na_value
    obj2 = SimpleNamespace()
    obj2.na_value = nulls_fixture2

    # 检查 nulls_fixture 和 nulls_fixture2 是否引用同一对象
    if nulls_fixture is nulls_fixture2:
        # 如果是同一对象，则调用 tm.assert_attr_equal 函数，断言 obj 和 obj2 的 na_value 属性相等
        tm.assert_attr_equal("na_value", obj, obj2)
    # 检查 nulls_fixture 和 nulls_fixture2 是否为浮点数，且都是浮点数
    elif is_float(nulls_fixture) and is_float(nulls_fixture2):
        # 如果是浮点数，则调用 tm.assert_attr_equal 函数，断言 obj 和 obj2 的 na_value 属性相等
        # 根据注释说明，这里认为 float("nan") 和 np.float64("nan") 是等价的
        tm.assert_attr_equal("na_value", obj, obj2)
    # 检查 nulls_fixture 和 nulls_fixture2 的类型是否相同
    elif type(nulls_fixture) is type(nulls_fixture2):
        # 如果类型相同，则调用 tm.assert_attr_equal 函数，断言 obj 和 obj2 的 na_value 属性相等
        # 例如，Decimal("NaN") 的情况
        tm.assert_attr_equal("na_value", obj, obj2)
    else:
        # 如果以上条件都不满足，则使用 pytest.raises 检查是否抛出 AssertionError 异常
        # 并验证异常信息是否包含 '"na_value" are different'
        with pytest.raises(AssertionError, match='"na_value" are different'):
            tm.assert_attr_equal("na_value", obj, obj2)
```