# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_is_unique.py`

```
# 导入所需的库和模块
import numpy as np
import pytest

# 从 pandas 库中导入 Series 类
from pandas import Series

# 使用 pytest 的 parametrize 装饰器，定义了一个参数化测试函数 test_is_unique
@pytest.mark.parametrize(
    "data, expected",
    [
        (np.random.default_rng(2).integers(0, 10, size=1000), False),  # 随机整数数据，期望不唯一
        (np.arange(1000), True),  # 0 到 999 的整数序列，期望唯一
        ([], True),  # 空列表，期望唯一
        ([np.nan], True),  # 包含 NaN 的列表，期望唯一
        (["foo", "bar", np.nan], True),  # 包含字符串和 NaN 的列表，期望唯一
        (["foo", "foo", np.nan], False),  # 包含重复字符串和 NaN 的列表，期望不唯一
        (["foo", "bar", np.nan, np.nan], False),  # 包含多个 NaN 的列表，期望不唯一
    ],
)
def test_is_unique(data, expected):
    # GH#11946 / GH#25180
    # 创建一个 Series 对象，用于测试唯一性
    ser = Series(data)
    # 断言 Series 对象的唯一性是否符合期望
    assert ser.is_unique is expected


# 定义另一个测试函数 test_is_unique_class_ne，用于测试自定义类的 __ne__ 方法
def test_is_unique_class_ne(capsys):
    # GH#20661
    # 定义一个简单的自定义类 Foo
    class Foo:
        def __init__(self, val) -> None:
            self._value = val

        def __ne__(self, other):
            raise Exception("NEQ not supported")

    # 禁用 capsys 捕获，创建一组 Foo 对象并放入 Series 中
    with capsys.disabled():
        li = [Foo(i) for i in range(5)]
        ser = Series(li, index=list(range(5)))

    ser.is_unique  # 执行 Series 的 is_unique 方法
    captured = capsys.readouterr()  # 捕获 capsys 输出
    assert len(captured.err) == 0  # 断言没有捕获到异常信息
```