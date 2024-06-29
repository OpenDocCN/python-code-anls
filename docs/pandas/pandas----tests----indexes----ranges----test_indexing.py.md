# `D:\src\scipysrc\pandas\pandas\tests\indexes\ranges\test_indexing.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from pandas import (  # 从 Pandas 库中导入以下模块：
    Index,  # Index 类，用于创建索引对象
    RangeIndex,  # RangeIndex 类，用于创建整数范围索引对象
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块作为 tm 别名


class TestGetIndexer:
    def test_get_indexer(self):
        index = RangeIndex(start=0, stop=20, step=2)  # 创建一个起始为0，终止为20（不包括），步长为2的整数范围索引对象
        target = RangeIndex(10)  # 创建一个范围索引对象，从0到10（不包括）
        indexer = index.get_indexer(target)  # 获得将目标索引应用到当前索引的索引器
        expected = np.array([0, -1, 1, -1, 2, -1, 3, -1, 4, -1], dtype=np.intp)  # 期望的索引器结果数组
        tm.assert_numpy_array_equal(indexer, expected)  # 使用 Pandas 测试工具验证索引器结果数组与期望是否相等

    def test_get_indexer_pad(self):
        index = RangeIndex(start=0, stop=20, step=2)  # 创建一个起始为0，终止为20（不包括），步长为2的整数范围索引对象
        target = RangeIndex(10)  # 创建一个范围索引对象，从0到10（不包括）
        indexer = index.get_indexer(target, method="pad")  # 使用填充方法"pad"获得将目标索引应用到当前索引的索引器
        expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.intp)  # 期望的索引器结果数组
        tm.assert_numpy_array_equal(indexer, expected)  # 使用 Pandas 测试工具验证索引器结果数组与期望是否相等

    def test_get_indexer_backfill(self):
        index = RangeIndex(start=0, stop=20, step=2)  # 创建一个起始为0，终止为20（不包括），步长为2的整数范围索引对象
        target = RangeIndex(10)  # 创建一个范围索引对象，从0到10（不包括）
        indexer = index.get_indexer(target, method="backfill")  # 使用后向填充方法"backfill"获得将目标索引应用到当前索引的索引器
        expected = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.intp)  # 期望的索引器结果数组
        tm.assert_numpy_array_equal(indexer, expected)  # 使用 Pandas 测试工具验证索引器结果数组与期望是否相等

    def test_get_indexer_limit(self):
        # GH#28631
        idx = RangeIndex(4)  # 创建一个范围索引对象，从0到4（不包括）
        target = RangeIndex(6)  # 创建一个范围索引对象，从0到6（不包括）
        result = idx.get_indexer(target, method="pad", limit=1)  # 使用填充方法"pad"和限制参数1获得将目标索引应用到当前索引的索引器
        expected = np.array([0, 1, 2, 3, 3, -1], dtype=np.intp)  # 期望的索引器结果数组
        tm.assert_numpy_array_equal(result, expected)  # 使用 Pandas 测试工具验证索引器结果数组与期望是否相等

    @pytest.mark.parametrize("stop", [0, -1, -2])
    def test_get_indexer_decreasing(self, stop):
        # GH#28678
        index = RangeIndex(7, stop, -3)  # 创建一个起始为7，终止为 stop，步长为-3的整数范围索引对象
        result = index.get_indexer(range(9))  # 获得将目标索引应用到当前索引的索引器
        expected = np.array([-1, 2, -1, -1, 1, -1, -1, 0, -1], dtype=np.intp)  # 期望的索引器结果数组
        tm.assert_numpy_array_equal(result, expected)  # 使用 Pandas 测试工具验证索引器结果数组与期望是否相等


class TestTake:
    def test_take_preserve_name(self):
        index = RangeIndex(1, 5, name="foo")  # 创建一个起始为1，终止为5（不包括），名称为"foo"的范围索引对象
        taken = index.take([3, 0, 1])  # 从索引中获取指定位置的元素
        assert index.name == taken.name  # 断言索引对象的名称与取出的索引对象的名称相同

    def test_take_fill_value(self):
        # GH#12631
        idx = RangeIndex(1, 4, name="xxx")  # 创建一个起始为1，终止为4（不包括），名称为"xxx"的范围索引对象
        result = idx.take(np.array([1, 0, -1]))  # 从索引中获取指定位置的元素
        expected = Index([2, 1, 3], dtype=np.int64, name="xxx")  # 期望的索引对象结果
        tm.assert_index_equal(result, expected)  # 使用 Pandas 测试工具验证索引对象结果与期望是否相等

        # fill_value
        msg = "Unable to fill values because RangeIndex cannot contain NA"
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -1]), fill_value=True)  # 使用填充值时，抛出异常信息验证

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)  # 使用填充值，但不允许填充
        expected = Index([2, 1, 3], dtype=np.int64, name="xxx")  # 期望的索引对象结果
        tm.assert_index_equal(result, expected)  # 使用 Pandas 测试工具验证索引对象结果与期望是否相等

        msg = "Unable to fill values because RangeIndex cannot contain NA"
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)  # 使用填充值时，抛出异常信息验证
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)  # 使用填充值时，抛出异常信息验证
    # 定义测试函数，验证当使用 RangeIndex 对象调用 take 方法时是否会引发 IndexError 异常
    def test_take_raises_index_error(self):
        # 创建一个 RangeIndex 对象，范围为 [1, 2, 3]，命名为 "xxx"
        idx = RangeIndex(1, 4, name="xxx")

        # 定义错误消息模式，验证是否抛出特定的 IndexError 异常
        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        # 验证调用 take 方法时是否抛出 IndexError 异常，并匹配指定的错误消息模式
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

        # 定义另一个错误消息模式，验证是否抛出特定的 IndexError 异常
        msg = "index -4 is out of bounds for (axis 0 with )?size 3"
        # 验证调用 take 方法时是否抛出 IndexError 异常，并匹配指定的错误消息模式
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -4]))

        # 如果没有抛出异常，则验证结果
        # 调用 take 方法，传入数组 [1, -3]，返回结果
        result = idx.take(np.array([1, -3]))
        # 期望的结果是一个 Index 对象，内容为 [2, 1]，数据类型为 int64，命名为 "xxx"
        expected = Index([2, 1], dtype=np.int64, name="xxx")
        # 使用 assert_index_equal 函数验证结果与期望是否一致
        tm.assert_index_equal(result, expected)

    # 定义测试函数，验证当使用 RangeIndex 对象调用 take 方法时是否能接受空数组作为输入
    def test_take_accepts_empty_array(self):
        # 创建一个 RangeIndex 对象，范围为 [1, 2, 3]，命名为 "foo"
        idx = RangeIndex(1, 4, name="foo")
        # 调用 take 方法，传入空数组 []，返回结果
        result = idx.take(np.array([]))
        # 期望的结果是一个空的 Index 对象，数据类型为 int64，命名为 "foo"
        expected = Index([], dtype=np.int64, name="foo")
        # 使用 assert_index_equal 函数验证结果与期望是否一致
        tm.assert_index_equal(result, expected)

        # 创建一个空的 RangeIndex 对象，命名为 "foo"
        idx = RangeIndex(0, name="foo")
        # 调用 take 方法，传入空数组 []，返回结果
        result = idx.take(np.array([]))
        # 期望的结果是一个空的 Index 对象，数据类型为 int64，命名为 "foo"
        expected = Index([], dtype=np.int64, name="foo")
        # 使用 assert_index_equal 函数验证结果与期望是否一致
        tm.assert_index_equal(result, expected)

    # 定义测试函数，验证当使用 RangeIndex 对象调用 take 方法时是否能接受非 int64 类型的数组作为输入
    def test_take_accepts_non_int64_array(self):
        # 创建一个 RangeIndex 对象，范围为 [1, 2, 3]，命名为 "foo"
        idx = RangeIndex(1, 4, name="foo")
        # 调用 take 方法，传入数组 [2, 1]，数据类型为 uint32
        result = idx.take(np.array([2, 1], dtype=np.uint32))
        # 期望的结果是一个 Index 对象，内容为 [3, 2]，数据类型为 int64，命名为 "foo"
        expected = Index([3, 2], dtype=np.int64, name="foo")
        # 使用 assert_index_equal 函数验证结果与期望是否一致
        tm.assert_index_equal(result, expected)

    # 定义测试函数，验证当使用具有步长的 RangeIndex 对象调用 take 方法时的行为
    def test_take_when_index_has_step(self):
        # 创建一个具有步长的 RangeIndex 对象，范围为 [1, 4, 7, 10]，命名为 "foo"
        idx = RangeIndex(1, 11, 3, name="foo")
        # 调用 take 方法，传入数组 [1, 0, -1, -4]，返回结果
        result = idx.take(np.array([1, 0, -1, -4]))
        # 期望的结果是一个 Index 对象，内容为 [4, 1, 10, 1]，数据类型为 int64，命名为 "foo"
        expected = Index([4, 1, 10, 1], dtype=np.int64, name="foo")
        # 使用 assert_index_equal 函数验证结果与期望是否一致
        tm.assert_index_equal(result, expected)

    # 定义测试函数，验证当使用具有负步长的 RangeIndex 对象调用 take 方法时的行为
    def test_take_when_index_has_negative_step(self):
        # 创建一个具有负步长的 RangeIndex 对象，范围为 [11, 9, 7, 5, 3, 1, -1, -3]，命名为 "foo"
        idx = RangeIndex(11, -4, -2, name="foo")
        # 调用 take 方法，传入数组 [1, 0, -1, -8]，返回结果
        result = idx.take(np.array([1, 0, -1, -8]))
        # 期望的结果是一个 Index 对象，内容为 [9, 11, -3, 11]，数据类型为 int64，命名为 "foo"
        expected = Index([9, 11, -3, 11], dtype=np.int64, name="foo")
        # 使用 assert_index_equal 函数验证结果与期望是否一致
        tm.assert_index_equal(result, expected)
class TestWhere:
    # 定义一个测试类 TestWhere
    def test_where_putmask_range_cast(self):
        # 在测试方法 test_where_putmask_range_cast 中进行测试

        # 创建一个 RangeIndex 对象，范围是从 0 到 5，命名为 "test"
        idx = RangeIndex(0, 5, name="test")

        # 创建一个布尔数组作为掩码
        mask = np.array([True, True, False, False, False])

        # 使用 putmask 方法，根据 mask 将 idx 中符合条件的元素替换为 10
        result = idx.putmask(mask, 10)

        # 创建一个期望的 Index 对象，期望结果为 [10, 10, 2, 3, 4]，数据类型为 np.int64，命名为 "test"
        expected = Index([10, 10, 2, 3, 4], dtype=np.int64, name="test")

        # 断言 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)

        # 使用 where 方法，将 idx 中符合条件的元素替换为 10，与 putmask 方法结果相同
        result = idx.where(~mask, 10)

        # 断言 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)
```