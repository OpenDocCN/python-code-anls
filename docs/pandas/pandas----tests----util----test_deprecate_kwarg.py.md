# `D:\src\scipysrc\pandas\pandas\tests\util\test_deprecate_kwarg.py`

```
# 导入 pytest 模块，用于编写和运行测试
import pytest

# 从 pandas.util._decorators 模块中导入 deprecate_kwarg 装饰器
from pandas.util._decorators import deprecate_kwarg

# 导入 pandas._testing 模块，并使用别名 tm
import pandas._testing as tm

# 使用 deprecate_kwarg 装饰器，标记函数 _f1，将 old 参数标记为过时，使用 new 参数替代
@deprecate_kwarg("old", "new")
def _f1(new=False):
    return new

# 定义一个字典 _f2_mappings，用于将 "yes" 映射为 True，"no" 映射为 False
_f2_mappings = {"yes": True, "no": False}

# 使用 deprecate_kwarg 装饰器，标记函数 _f2，将 old 参数标记为过时，使用 new 参数替代，
# 同时使用 _f2_mappings 字典将字符串参数映射为布尔值
@deprecate_kwarg("old", "new", _f2_mappings)
def _f2(new=False):
    return new

# 定义一个函数 _f3_mapping，接受一个参数 x，并返回 x+1
def _f3_mapping(x):
    return x + 1

# 使用 deprecate_kwarg 装饰器，标记函数 _f3，将 old 参数标记为过时，使用 new 参数替代，
# 同时使用 _f3_mapping 函数处理 new 参数的映射
@deprecate_kwarg("old", "new", _f3_mapping)
def _f3(new=0):
    return new

# 使用 pytest.mark.parametrize 装饰器标记测试函数 test_deprecate_kwarg，
# 参数化 key 和 klass，分别为 "old"/FutureWarning 和 "new"/None
@pytest.mark.parametrize("key,klass", [("old", FutureWarning), ("new", None)])
def test_deprecate_kwarg(key, klass):
    # 设置变量 x 的值为 78
    x = 78

    # 使用 tm.assert_produces_warning 上下文，检查是否产生警告
    with tm.assert_produces_warning(klass):
        # 断言调用 _f1 函数时使用关键字参数 {key: x} 返回的结果与 x 相等
        assert _f1(**{key: x}) == x

# 使用 pytest.mark.parametrize 装饰器标记测试函数 test_dict_deprecate_kwarg，
# 参数化 key，key 的值为 _f2_mappings 字典的键列表
@pytest.mark.parametrize("key", list(_f2_mappings.keys()))
def test_dict_deprecate_kwarg(key):
    # 使用 tm.assert_produces_warning 上下文，检查是否产生 FutureWarning 警告
    with tm.assert_produces_warning(FutureWarning):
        # 断言调用 _f2 函数时使用关键字参数 old=key 返回的结果与 _f2_mappings[key] 相等
        assert _f2(old=key) == _f2_mappings[key]

# 使用 pytest.mark.parametrize 装饰器标记测试函数 test_missing_deprecate_kwarg，
# 参数化 key，key 的值为列表 ["bogus", 12345, -1.23]
@pytest.mark.parametrize("key", ["bogus", 12345, -1.23])
def test_missing_deprecate_kwarg(key):
    # 使用 tm.assert_produces_warning 上下文，检查是否产生 FutureWarning 警告
    with tm.assert_produces_warning(FutureWarning):
        # 断言调用 _f2 函数时使用关键字参数 old=key 返回的结果与 key 相等
        assert _f2(old=key) == key

# 使用 pytest.mark.parametrize 装饰器标记测试函数 test_callable_deprecate_kwarg，
# 参数化 x，x 的值为 [1, -1.4, 0]
@pytest.mark.parametrize("x", [1, -1.4, 0])
def test_callable_deprecate_kwarg(x):
    # 使用 tm.assert_produces_warning 上下文，检查是否产生 FutureWarning 警告
    with tm.assert_produces_warning(FutureWarning):
        # 断言调用 _f3 函数时使用关键字参数 old=x 返回的结果与 _f3_mapping(x) 相等
        assert _f3(old=x) == _f3_mapping(x)

# 定义测试函数 test_callable_deprecate_kwarg_fail，用于测试不支持的参数类型异常情况
def test_callable_deprecate_kwarg_fail():
    # 设置异常信息模式 msg，用于匹配错误消息
    msg = "((can only|cannot) concatenate)|(must be str)|(Can't convert)"

    # 使用 pytest.raises 检查是否引发 TypeError 异常，并匹配异常信息模式 msg
    with pytest.raises(TypeError, match=msg):
        # 调用 _f3 函数时使用关键字参数 old="hello"，预期会引发 TypeError 异常
        _f3(old="hello")

# 定义测试函数 test_bad_deprecate_kwarg，用于测试错误的参数映射类型异常情况
def test_bad_deprecate_kwarg():
    # 设置异常信息模式 msg，用于匹配错误消息
    msg = "mapping from old to new argument values must be dict or callable!"

    # 使用 pytest.raises 检查是否引发 TypeError 异常，并匹配异常信息模式 msg
    with pytest.raises(TypeError, match=msg):
        # 使用 deprecate_kwarg 装饰器定义函数 f4，但映射参数为 0，预期会引发 TypeError 异常
        @deprecate_kwarg("old", "new", 0)
        def f4(new=None):
            return new

# 使用 deprecate_kwarg 装饰器，标记函数 _f4，将 old 参数标记为过时，不使用新参数替代，
# unchanged 参数默认为 True
@deprecate_kwarg("old", None)
def _f4(old=True, unchanged=True):
    return old, unchanged

# 使用 pytest.mark.parametrize 装饰器标记测试函数 test_deprecate_keyword，
# 参数化 key，key 的值为 ["old", "unchanged"]
@pytest.mark.parametrize("key", ["old", "unchanged"])
def test_deprecate_keyword(key):
    # 设置变量 x 的值为 9
    x = 9

    # 根据 key 的值确定 klass 和 expected 的值
    if key == "old":
        klass = FutureWarning
        expected = (x, True)
    else:
        klass = None
        expected = (True, x)

    # 使用 tm.assert_produces_warning 上下文，检查是否产生相应的警告
    with tm.assert_produces_warning(klass):
        # 断言调用 _f4 函数时使用关键字参数 {key: x} 返回的结果与 expected 相等
        assert _f4(**{key: x}) == expected
```