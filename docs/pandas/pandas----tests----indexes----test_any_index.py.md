# `D:\src\scipysrc\pandas\pandas\tests\indexes\test_any_index.py`

```
"""
Tests that can be parametrized over _any_ Index object.
"""

# 导入所需的模块
import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm

# 测试函数：验证在布尔上下文中的行为兼容性
def test_boolean_context_compat(index):
    # GH#7897
    # 断言在布尔上下文中引发 ValueError 异常，匹配指定的错误信息
    with pytest.raises(ValueError, match="The truth value of a"):
        if index:
            pass
    
    # 另一种布尔上下文的断言方式，验证同样的异常和错误信息
    with pytest.raises(ValueError, match="The truth value of a"):
        bool(index)

# 测试函数：验证索引对象排序方法的异常行为
def test_sort(index):
    # 准备错误信息字符串
    msg = "cannot sort an Index object in-place, use sort_values instead"
    # 断言在排序时引发 TypeError 异常，匹配指定的错误信息
    with pytest.raises(TypeError, match=msg):
        index.sort()

# 测试函数：验证索引对象哈希化的异常行为
def test_hash_error(index):
    # 断言在尝试对索引对象进行哈希化时引发 TypeError 异常，匹配指定的错误信息
    with pytest.raises(TypeError, match=f"unhashable type: '{type(index).__name__}'"):
        hash(index)

# 测试函数：验证索引对象的可变性
def test_mutability(index):
    # 如果索引对象为空，则跳过测试
    if not len(index):
        pytest.skip("Test doesn't make sense for empty index")
    # 准备错误信息字符串
    msg = "Index does not support mutable operations"
    # 断言在尝试修改索引对象的第一个元素时引发 TypeError 异常，匹配指定的错误信息
    with pytest.raises(TypeError, match=msg):
        index[0] = index[0]

# 警告过滤标记，忽略特定的警告
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
# 测试函数：验证索引对象映射自身的行为
def test_map_identity_mapping(index, request):
    # GH#12766
    # 对索引对象进行映射操作，使用 lambda 函数映射到自身
    result = index.map(lambda x: x)
    # 如果索引对象和结果都是对象类型且结果是布尔类型
    if index.dtype == object and result.dtype == bool:
        # 断言索引对象和结果的每个元素都相等
        assert (index == result).all()
        # TODO: 可以将这部分工作整合到 'exact="equiv"' 中吗？
        return  # FIXME: doesn't belong in this file anymore!
    # 否则，比较映射结果与原索引对象，确保等价性
    tm.assert_index_equal(result, index, exact="equiv")

# 测试函数：验证索引对象名称列表长度不匹配的异常行为
def test_wrong_number_names(index):
    # 准备长度不匹配的名称列表
    names = index.nlevels * ["apple", "banana", "carrot"]
    # 断言在尝试设置索引对象的名称时引发 ValueError 异常，匹配指定的错误信息开头
    with pytest.raises(ValueError, match="^Length"):
        index.names = names

# 测试函数：验证索引对象视图保留名称的行为
def test_view_preserves_name(index):
    # 断言索引对象的视图的名称与原索引对象的名称相同
    assert index.view().name == index.name

# 测试函数：验证索引对象的 ravel 方法行为
def test_ravel(index):
    # GH#19956 ravel returning ndarray is deprecated, in 2.0 returns a view on self
    # 调用 ravel 方法，返回结果
    res = index.ravel()
    # 断言 ravel 方法返回的结果与原索引对象相等
    tm.assert_index_equal(res, index)

# 测试类：测试索引对象到 Series 对象的转换
class TestConversion:
    # 测试方法：验证索引对象转换为 Series 对象的行为
    def test_to_series(self, index):
        # 断言确保创建了索引对象的副本
        ser = index.to_series()
        assert ser.values is not index.values
        assert ser.index is not index
        assert ser.name == index.name
    
    # 测试方法：验证带参数的索引对象到 Series 对象的转换行为
    def test_to_series_with_arguments(self, index):
        # GH#18699
        # 使用 index 参数进行索引对象到 Series 对象的转换
        ser = index.to_series(index=index)
        assert ser.values is not index.values
        assert ser.index is index
        assert ser.name == index.name

        # 使用 name 参数进行索引对象到 Series 对象的转换
        ser = index.to_series(name="__test")
        assert ser.values is not index.values
        assert ser.index is not index
        assert ser.name != index.name

    # 测试方法：验证索引对象的 tolist 方法与 list 方法一致性
    def test_tolist_matches_list(self, index):
        assert index.tolist() == list(index)

# 测试类：测试序列化和反序列化操作
class TestRoundTrips:
    # 测试方法：验证索引对象的 pickle 序列化和反序列化往返
    def test_pickle_roundtrip(self, index):
        # 进行序列化和反序列化操作，返回结果
        result = tm.round_trip_pickle(index)
        # 断言序列化和反序列化的结果与原索引对象相等，且精确匹配
        tm.assert_index_equal(result, index, exact=True)
        # 如果结果的层级数大于 1，进行额外的比较以确保等级的相等性
        if result.nlevels > 1:
            # GH#8367 round-trip with timezone
            assert index.equal_levels(result)
    # 定义测试函数，用于验证 pickle 序列化后是否保留了索引的名称
    def test_pickle_preserves_name(self, index):
        # 临时保存原始索引名称，并将索引名称设置为 "foo"
        original_name, index.name = index.name, "foo"
        # 使用 round_trip_pickle 方法对索引进行 pickle 序列化和反序列化
        unpickled = tm.round_trip_pickle(index)
        # 断言：序列化前后的索引应当相等
        assert index.equals(unpickled)
        # 恢复原始的索引名称
        index.name = original_name
class TestIndexing:
    # 测试索引操作中当类引发无效索引错误时抛出异常
    def test_get_loc_listlike_raises_invalid_index_error(self, index):
        # 创建一个 numpy 数组作为测试的索引键
        key = np.array([0, 1], dtype=np.intp)

        # 使用 pytest 来验证调用 get_loc 方法时抛出 InvalidIndexError 异常，并匹配指定的错误消息
        with pytest.raises(InvalidIndexError, match=r"\[0 1\]"):
            index.get_loc(key)

        # 同样验证当传入布尔类型的数组时抛出异常，并匹配指定的错误消息
        with pytest.raises(InvalidIndexError, match=r"\[False  True\]"):
            index.get_loc(key.astype(bool))

    # 测试使用省略号（Ellipsis）索引操作
    def test_getitem_ellipsis(self, index):
        # 获取省略号索引的结果，并断言结果等同于原索引对象但不是同一对象
        result = index[...]
        assert result.equals(index)
        assert result is not index

    # 测试切片操作保留索引名称
    def test_slice_keeps_name(self, index):
        # 断言切片操作后的索引名称与原索引的切片名称一致
        assert index.name == index[1:].name

    # 参数化测试索引操作中的错误情况
    @pytest.mark.parametrize("item", [101, "no_int", 2.5])
    def test_getitem_error(self, index, item):
        # 构建预期的错误消息，包含多种可能的错误情况
        msg = "|".join(
            [
                r"index 101 is out of bounds for axis 0 with size [\d]+",
                re.escape(
                    "only integers, slices (`:`), ellipsis (`...`), "
                    "numpy.newaxis (`None`) and integer or boolean arrays "
                    "are valid indices"
                ),
                "index out of bounds",  # string[pyarrow]
            ]
        )
        # 使用 pytest 验证当传入错误的索引时抛出 IndexError 异常，并匹配预期的错误消息
        with pytest.raises(IndexError, match=msg):
            index[item]


class TestRendering:
    # 测试索引对象的字符串表示形式
    def test_str(self, index):
        # 设置索引名称为 "foo"，并断言其在索引对象的字符串表示中出现
        index.name = "foo"
        assert "'foo'" in str(index)
        # 断言索引对象的类型名称也出现在其字符串表示中
        assert type(index).__name__ in str(index)


class TestReductions:
    # 测试在进行归约操作时传入无效的 axis 参数时抛出 ValueError 异常
    def test_argmax_axis_invalid(self, index):
        # 定义预期的错误消息
        msg = r"`axis` must be fewer than the number of dimensions \(1\)"
        
        # 使用 pytest 验证在调用归约操作时传入无效的 axis 参数时抛出 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            index.argmax(axis=1)
        with pytest.raises(ValueError, match=msg):
            index.argmin(axis=2)
        with pytest.raises(ValueError, match=msg):
            index.min(axis=-2)
        with pytest.raises(ValueError, match=msg):
            index.max(axis=-3)
```