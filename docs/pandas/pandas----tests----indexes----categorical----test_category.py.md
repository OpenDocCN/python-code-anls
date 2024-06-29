# `D:\src\scipysrc\pandas\pandas\tests\indexes\categorical\test_category.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas._config import using_pyarrow_string_dtype  # 导入 pandas 内部配置模块中的 using_pyarrow_string_dtype 函数

from pandas._libs import index as libindex  # 导入 pandas 库中的 _libs.index 模块，并重命名为 libindex
from pandas._libs.arrays import NDArrayBacked  # 导入 pandas 库中的 _libs.arrays 模块中的 NDArrayBacked 类

import pandas as pd  # 导入 pandas 库，并使用 pd 别名
from pandas import (  # 导入 pandas 中的 Categorical 和 CategoricalDtype 类
    Categorical,
    CategoricalDtype,
)
import pandas._testing as tm  # 导入 pandas 库中的 _testing 模块，并使用 tm 别名
from pandas.core.indexes.api import (  # 从 pandas 库中的 core.indexes.api 模块中导入以下类
    CategoricalIndex,
    Index,
)


class TestCategoricalIndex:
    @pytest.fixture  # 使用 pytest 的 fixture 装饰器定义测试函数的装置
    def simple_index(self) -> CategoricalIndex:  # 返回一个简单的 CategoricalIndex 对象
        return CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)

    def test_can_hold_identifiers(self):
        idx = CategoricalIndex(list("aabbca"), categories=None, ordered=False)
        key = idx[0]  # 获取索引为 0 的元素
        assert idx._can_hold_identifiers_and_holds_name(key) is True  # 断言索引是否能够保存标识符并且保持名称不变

    def test_insert(self, simple_index):
        ci = simple_index  # 使用 simple_index fixture 创建 CategoricalIndex 对象
        categories = ci.categories  # 获取对象的分类列表

        # 测试插入第一个元素
        result = ci.insert(0, "a")
        expected = CategoricalIndex(list("aaabbca"), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)  # 断言插入结果与预期相同

        # 测试插入倒数第二个元素，类似 Python 列表的行为
        result = ci.insert(-1, "a")
        expected = CategoricalIndex(list("aabbcaa"), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)  # 断言插入结果与预期相同

        # 测试空索引的插入
        result = CategoricalIndex([], categories=categories).insert(0, "a")
        expected = CategoricalIndex(["a"], categories=categories)
        tm.assert_index_equal(result, expected, exact=True)  # 断言插入结果与预期相同

        # 插入无效值时，将对象转换为 object 类型
        expected = ci.astype(object).insert(0, "d")
        result = ci.insert(0, "d").astype(object)
        tm.assert_index_equal(result, expected, exact=True)  # 断言插入结果与预期相同

        # GH 18295 (测试缺失值)
        expected = CategoricalIndex(["a", np.nan, "a", "b", "c", "b"])
        for na in (np.nan, pd.NaT, None):
            result = CategoricalIndex(list("aabcb")).insert(1, na)
            tm.assert_index_equal(result, expected)  # 断言插入结果与预期相同

    def test_insert_na_mismatched_dtype(self):
        ci = CategoricalIndex([0, 1, 1])
        result = ci.insert(0, pd.NaT)  # 在索引 0 处插入 pd.NaT
        expected = Index([pd.NaT, 0, 1, 1], dtype=object)
        tm.assert_index_equal(result, expected)  # 断言插入结果与预期相同

    def test_delete(self, simple_index):
        ci = simple_index  # 使用 simple_index fixture 创建 CategoricalIndex 对象
        categories = ci.categories  # 获取对象的分类列表

        result = ci.delete(0)  # 删除索引为 0 的元素
        expected = CategoricalIndex(list("abbca"), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)  # 断言删除结果与预期相同

        result = ci.delete(-1)  # 删除倒数第一个元素
        expected = CategoricalIndex(list("aabbc"), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)  # 断言删除结果与预期相同

        with tm.external_error_raised((IndexError, ValueError)):
            # 取决于 NumPy 版本，可能会抛出 IndexError 或 ValueError
            ci.delete(10)

    @pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器定义多组参数化测试
        "data, non_lexsorted_data",
        [[[1, 2, 3], [9, 0, 1, 2, 3]], [list("abc"), list("fabcd")]],
    )
    # 定义测试方法，用于检查 CategoricalIndex 对象的单调性
    def test_is_monotonic(self, data, non_lexsorted_data):
        # 创建 CategoricalIndex 对象，并断言其是否单调递增和非单调递减
        c = CategoricalIndex(data)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False

        # 使用 ordered=True 创建 CategoricalIndex 对象，并断言其是否单调递增和非单调递减
        c = CategoricalIndex(data, ordered=True)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False

        # 使用倒序的 categories 创建 CategoricalIndex 对象，并断言其是否非单调递增和单调递减
        c = CategoricalIndex(data, categories=reversed(data))
        assert c.is_monotonic_increasing is False
        assert c.is_monotonic_decreasing is True

        # 使用 ordered=True 和倒序的 categories 创建 CategoricalIndex 对象，并断言其是否非单调递增和单调递减
        c = CategoricalIndex(data, categories=reversed(data), ordered=True)
        assert c.is_monotonic_increasing is False
        assert c.is_monotonic_decreasing is True

        # 当数据既不单调递增也不单调递减时，重新排列数据创建 CategoricalIndex 对象，并断言其是否非单调递增和非单调递减
        reordered_data = [data[0], data[2], data[1]]
        c = CategoricalIndex(reordered_data, categories=reversed(data))
        assert c.is_monotonic_increasing is False
        assert c.is_monotonic_decreasing is False

        # 测试非词法排序的 categories
        categories = non_lexsorted_data

        # 使用部分 categories 创建 CategoricalIndex 对象，并断言其是否单调递增和非单调递减
        c = CategoricalIndex(categories[:2], categories=categories)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False

        # 使用另一部分 categories 创建 CategoricalIndex 对象，并断言其是否单调递增和非单调递减
        c = CategoricalIndex(categories[1:3], categories=categories)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False
    @pytest.mark.parametrize(
        "data, categories, expected",
        [  # 使用 pytest.mark.parametrize 装饰器设置参数化测试，传入三个参数：data, categories, expected
            (
                [1, 1, 1],  # 第一个测试用例的 data 参数，包含三个元素 [1, 1, 1]
                [1, 2, 3],  # 第一个测试用例的 categories 参数，包含三个元素 [1, 2, 3]
                {  # 第一个测试用例的 expected 参数，包含三个键值对
                    "first": np.array([False, True, True]),  # 键 "first" 对应的值是一个 NumPy 数组
                    "last": np.array([True, True, False]),  # 键 "last" 对应的值是一个 NumPy 数组
                    False: np.array([True, True, True]),  # 键 False 对应的值是一个 NumPy 数组
                },
            ),
            (
                [1, 1, 1],  # 第二个测试用例的 data 参数，与第一个测试用例相同，包含三个元素 [1, 1, 1]
                list("abc"),  # 第二个测试用例的 categories 参数，包含字符 'a', 'b', 'c'
                {  # 第二个测试用例的 expected 参数，与第一个测试用例相同
                    "first": np.array([False, True, True]),  # 键 "first" 对应的值是一个 NumPy 数组
                    "last": np.array([True, True, False]),  # 键 "last" 对应的值是一个 NumPy 数组
                    False: np.array([True, True, True]),  # 键 False 对应的值是一个 NumPy 数组
                },
            ),
            (
                [2, "a", "b"],  # 第三个测试用例的 data 参数，包含三个元素 [2, "a", "b"]
                list("abc"),  # 第三个测试用例的 categories 参数，包含字符 'a', 'b', 'c'
                {  # 第三个测试用例的 expected 参数
                    "first": np.zeros(shape=(3), dtype=np.bool_),  # 键 "first" 对应的值是一个全零 NumPy 数组
                    "last": np.zeros(shape=(3), dtype=np.bool_),  # 键 "last" 对应的值是一个全零 NumPy 数组
                    False: np.zeros(shape=(3), dtype=np.bool_),  # 键 False 对应的值是一个全零 NumPy 数组
                },
            ),
            (
                list("abb"),  # 第四个测试用例的 data 参数，包含三个元素 ['a', 'b', 'b']
                list("abc"),  # 第四个测试用例的 categories 参数，包含字符 'a', 'b', 'c'
                {  # 第四个测试用例的 expected 参数
                    "first": np.array([False, False, True]),  # 键 "first" 对应的值是一个 NumPy 数组
                    "last": np.array([False, True, False]),  # 键 "last" 对应的值是一个 NumPy 数组
                    False: np.array([False, True, True]),  # 键 False 对应的值是一个 NumPy 数组
                },
            ),
        ],
    )
    def test_drop_duplicates(self, data, categories, expected):
        idx = CategoricalIndex(data, categories=categories, name="foo")  # 创建 CategoricalIndex 对象 idx
        for keep, e in expected.items():  # 遍历 expected 字典的键值对
            tm.assert_numpy_array_equal(idx.duplicated(keep=keep), e)  # 断言 idx.duplicated(keep=keep) 与 e 相等
            e = idx[~e]  # 将 e 取反后赋值给 e
            result = idx.drop_duplicates(keep=keep)  # 调用 idx.drop_duplicates 方法，传入参数 keep，结果赋给 result
            tm.assert_index_equal(result, e)  # 断言 result 与 e 索引相等

    @pytest.mark.parametrize(
        "data, categories, expected_data",
        [  # 使用 pytest.mark.parametrize 装饰器设置参数化测试，传入三个参数：data, categories, expected_data
            ([1, 1, 1], [1, 2, 3], [1]),  # 第一个测试用例的 data, categories, expected_data 参数
            ([1, 1, 1], list("abc"), [np.nan]),  # 第二个测试用例的 data, categories, expected_data 参数
            ([1, 2, "a"], [1, 2, 3], [1, 2, np.nan]),  # 第三个测试用例的 data, categories, expected_data 参数
            ([2, "a", "b"], list("abc"), [np.nan, "a", "b"]),  # 第四个测试用例的 data, categories, expected_data 参数
        ],
    )
    def test_unique(self, data, categories, expected_data, ordered):
        dtype = CategoricalDtype(categories, ordered=ordered)  # 创建 CategoricalDtype 对象 dtype

        idx = CategoricalIndex(data, dtype=dtype)  # 创建 CategoricalIndex 对象 idx
        expected = CategoricalIndex(expected_data, dtype=dtype)  # 创建预期的 CategoricalIndex 对象 expected
        tm.assert_index_equal(idx.unique(), expected)  # 断言 idx.unique() 与 expected 索引相等

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="repr doesn't roundtrip")
    def test_repr_roundtrip(self):
        ci = CategoricalIndex(["a", "b"], categories=["a", "b"], ordered=True)  # 创建 CategoricalIndex 对象 ci
        str(ci)  # 调用 str 方法输出 ci 的字符串表示形式
        tm.assert_index_equal(eval(repr(ci)), ci, exact=True)  # 断言 eval(repr(ci)) 与 ci 精确相等

        # formatting
        str(ci)  # 再次调用 str 方法输出 ci 的字符串表示形式

        # long format
        # this is not reprable
        ci = CategoricalIndex(np.random.default_rng(2).integers(0, 5, size=100))  # 创建具有随机整数数据的 CategoricalIndex 对象 ci
        str(ci)  # 调用 str 方法输出 ci 的字符串表示形式
    def test_isin(self):
        # 创建一个 CategoricalIndex 对象，用于测试，包含字符列表和一个 NaN 值，
        # 并指定类别顺序为 ["c", "a", "b"]
        ci = CategoricalIndex(list("aabca") + [np.nan], categories=["c", "a", "b"])
        
        # 断言 CategoricalIndex 对象的 isin 方法是否正确判断是否包含 "c"，预期结果为 False, False, False, True, False, False
        tm.assert_numpy_array_equal(
            ci.isin(["c"]), np.array([False, False, False, True, False, False])
        )
        
        # 断言 CategoricalIndex 对象的 isin 方法是否正确判断是否包含 ["c", "a", "b"]，预期结果为 True * 5 + False
        tm.assert_numpy_array_equal(
            ci.isin(["c", "a", "b"]), np.array([True] * 5 + [False])
        )
        
        # 断言 CategoricalIndex 对象的 isin 方法是否正确判断是否包含 ["c", "a", "b", np.nan]，预期结果为 True * 6
        tm.assert_numpy_array_equal(
            ci.isin(["c", "a", "b", np.nan]), np.array([True] * 6)
        )

        # 测试当类别不匹配时，isin 方法是否正常工作，预期结果为 True * 6
        result = ci.isin(ci.set_categories(list("abcdefghi")))
        expected = np.array([True] * 6)
        tm.assert_numpy_array_equal(result, expected)

        # 测试当类别不完全匹配时，isin 方法是否正常工作，预期结果为 False * 5 + True
        result = ci.isin(ci.set_categories(list("defghi")))
        expected = np.array([False] * 5 + [True])
        tm.assert_numpy_array_equal(result, expected)

    def test_isin_overlapping_intervals(self):
        # 测试在重叠区间的情况下，isin 方法是否正常工作，预期结果为 True, True
        # GH 34974
        idx = pd.IntervalIndex([pd.Interval(0, 2), pd.Interval(0, 1)])
        result = CategoricalIndex(idx).isin(idx)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_identical(self):
        # 创建两个 CategoricalIndex 对象，并测试它们的 identical 方法
        # ci1 包含类别为 ["a", "b"]，ci2 包含类别为 ["a", "b", "c"]
        ci1 = CategoricalIndex(["a", "b"], categories=["a", "b"], ordered=True)
        ci2 = CategoricalIndex(["a", "b"], categories=["a", "b", "c"], ordered=True)
        
        # 断言 ci1.identical(ci1) 返回 True
        assert ci1.identical(ci1)
        
        # 断言 ci1.identical(ci1.copy()) 返回 True
        assert ci1.identical(ci1.copy())
        
        # 断言 ci1.identical(ci2) 返回 False
        assert not ci1.identical(ci2)

    def test_ensure_copied_data(self):
        # 测试 CategoricalIndex 对象的构造函数中的 copy 参数是否生效
        # gh-12309: 检查每个 Index.__new__ 是否遵循 copy 参数。
        #
        # 必须与其他索引类型分开测试，因为 self.values 不是一个 ndarray。
        
        # 创建一个包含字符列表 "ab" 重复 5 次的 CategoricalIndex 对象
        index = CategoricalIndex(list("ab") * 5)

        # 测试 copy=True 时的情况
        result = CategoricalIndex(index.values, copy=True)
        # 断言 index 和 result 相等
        tm.assert_index_equal(index, result)
        # 断言 result._data._codes 和 index._data._codes 不共享内存
        assert not np.shares_memory(result._data._codes, index._data._codes)

        # 测试 copy=False 时的情况
        result = CategoricalIndex(index.values, copy=False)
        # 断言 result._data._codes 和 index._data._codes 共享内存
        assert result._data._codes is index._data._codes
class TestCategoricalIndex2:
    def test_view_i8(self):
        # GH#25464
        # 创建一个包含重复字符串 'ab' 50 次的分类索引对象
        ci = CategoricalIndex(list("ab") * 50)
        # 设置错误消息，用于捕获 ValueError 异常
        msg = "When changing to a larger dtype, its size must be a divisor"
        # 断言转换为 "i8" 类型时抛出 ValueError 异常，并匹配预期消息
        with pytest.raises(ValueError, match=msg):
            ci.view("i8")
        # 断言 ci._data 转换为 "i8" 类型时抛出 ValueError 异常，并匹配预期消息
        with pytest.raises(ValueError, match=msg):
            ci._data.view("i8")

        # 将 ci 切片，使其长度能被 8 整除
        ci = ci[:-4]

        # 转换 ci 为 "i8" 类型，并断言结果与 ci._data.codes 转换为 "i8" 后的结果相等
        res = ci.view("i8")
        expected = ci._data.codes.view("i8")
        tm.assert_numpy_array_equal(res, expected)

        # 获取 ci._data，断言其转换为 "i8" 类型后与 expected 相等
        cat = ci._data
        tm.assert_numpy_array_equal(cat.view("i8"), expected)

    @pytest.mark.parametrize(
        "dtype, engine_type",
        [
            (np.int8, libindex.Int8Engine),
            (np.int16, libindex.Int16Engine),
            (np.int32, libindex.Int32Engine),
            (np.int64, libindex.Int64Engine),
        ],
    )
    def test_engine_type(self, dtype, engine_type):
        if dtype != np.int64:
            # 根据 dtype 设置不同数量的唯一值来创建 CategoricalIndex
            num_uniques = {np.int8: 1, np.int16: 128, np.int32: 32768}[dtype]
            ci = CategoricalIndex(range(num_uniques))
        else:
            # 当 dtype 为 np.int64 时，创建一个包含 32768 个元素的 CategoricalIndex 对象
            # 这相当于 2**16 - 2**(16 - 1)
            ci = CategoricalIndex(range(32768))
            # 将 ci 的值数组转换为 int64 类型，并初始化 NDArrayBacked
            arr = ci.values._ndarray.astype("int64")
            NDArrayBacked.__init__(ci._data, arr, ci.dtype)
        # 断言 ci 的 codes 属性的 dtype 是 dtype 的子类型
        assert np.issubdtype(ci.codes.dtype, dtype)
        # 断言 ci 的 _engine 属性是 engine_type 的实例
        assert isinstance(ci._engine, engine_type)

    @pytest.mark.parametrize(
        "func,op_name",
        [
            (lambda idx: idx - idx, "__sub__"),
            (lambda idx: idx + idx, "__add__"),
            (lambda idx: idx - ["a", "b"], "__sub__"),
            (lambda idx: idx + ["a", "b"], "__add__"),
            (lambda idx: ["a", "b"] - idx, "__rsub__"),
            (lambda idx: ["a", "b"] + idx, "__radd__"),
        ],
    )
    def test_disallow_addsub_ops(self, func, op_name):
        # GH 10039
        # 设置操作名称和相应的错误消息
        # 对于 CategoricalIndex 类型的索引，禁止使用加法和减法操作
        idx = Index(Categorical(["a", "b"]))
        cat_or_list = "'(Categorical|list)' and '(Categorical|list)'"
        msg = "|".join(
            [
                f"cannot perform {op_name} with this index type: CategoricalIndex",
                "can only concatenate list",
                rf"unsupported operand type\(s\) for [\+-]: {cat_or_list}",
            ]
        )
        # 断言使用 func 执行操作时抛出 TypeError 异常，并匹配预期消息
        with pytest.raises(TypeError, match=msg):
            func(idx)
    # 测试方法委托

    # 创建一个包含重复元素的分类索引对象 ci，指定自定义的分类列表为 ['c', 'a', 'b', 'd', 'e', 'f']
    ci = CategoricalIndex(list("aabbca"), categories=list("cabdef"))
    # 使用 set_categories 方法设置新的分类列表为 ['c', 'a', 'b']，并返回修改后的结果 result
    result = ci.set_categories(list("cab"))
    # 断言 result 与预期的 CategoricalIndex 对象相等
    tm.assert_index_equal(
        result, CategoricalIndex(list("aabbca"), categories=list("cab"))
    )

    # 创建另一个包含重复元素的分类索引对象 ci，指定分类列表为 ['c', 'a', 'b']
    ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
    # 使用 rename_categories 方法将分类列表重命名为 ['e', 'f', 'g']，并返回修改后的结果 result
    result = ci.rename_categories(list("efg"))
    # 断言 result 与预期的 CategoricalIndex 对象相等
    tm.assert_index_equal(
        result, CategoricalIndex(list("ffggef"), categories=list("efg"))
    )

    # GH18862 (让 rename_categories 方法接受可调用对象)
    # 使用 lambda 函数将分类列表中的元素转换为大写字母，并返回修改后的结果 result
    result = ci.rename_categories(lambda x: x.upper())
    # 断言 result 与预期的 CategoricalIndex 对象相等
    tm.assert_index_equal(
        result, CategoricalIndex(list("AABBCA"), categories=list("CAB"))
    )

    # 创建另一个包含重复元素的分类索引对象 ci，指定分类列表为 ['c', 'a', 'b']
    ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
    # 使用 add_categories 方法添加新的分类元素 ['d']，并返回修改后的结果 result
    result = ci.add_categories(["d"])
    # 断言 result 与预期的 CategoricalIndex 对象相等
    tm.assert_index_equal(
        result, CategoricalIndex(list("aabbca"), categories=list("cabd"))
    )

    # 创建另一个包含重复元素的分类索引对象 ci，指定分类列表为 ['c', 'a', 'b']
    ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
    # 使用 remove_categories 方法移除分类元素 ['c']，并返回修改后的结果 result
    result = ci.remove_categories(["c"])
    # 断言 result 与预期的 CategoricalIndex 对象相等，包含了 np.nan 和 'a'
    tm.assert_index_equal(
        result,
        CategoricalIndex(list("aabb") + [np.nan] + ["a"], categories=list("ab")),
    )

    # 创建一个包含重复元素的分类索引对象 ci，指定自定义的分类列表为 ['c', 'a', 'b', 'd', 'e', 'f']
    ci = CategoricalIndex(list("aabbca"), categories=list("cabdef"))
    # 使用 as_unordered 方法将有序分类索引转换为无序，返回结果 result
    result = ci.as_unordered()
    # 断言 result 与原始的分类索引对象 ci 相等
    tm.assert_index_equal(result, ci)

    # 创建一个包含重复元素的分类索引对象 ci，指定自定义的分类列表为 ['c', 'a', 'b', 'd', 'e', 'f']
    ci = CategoricalIndex(list("aabbca"), categories=list("cabdef"))
    # 使用 as_ordered 方法将无序分类索引转换为有序，返回结果 result
    result = ci.as_ordered()
    # 断言 result 与预期的有序 CategoricalIndex 对象相等
    tm.assert_index_equal(
        result,
        CategoricalIndex(list("aabbca"), categories=list("cabdef"), ordered=True),
    )

    # 无效操作
    # 设置消息为 "cannot use inplace with CategoricalIndex"
    msg = "cannot use inplace with CategoricalIndex"
    # 使用 pytest.raises 检查 ValueError 异常，验证是否包含预期的错误消息
    with pytest.raises(ValueError, match=msg):
        # 尝试在 CategoricalIndex 上使用 set_categories 方法，传递 inplace=True 参数
        ci.set_categories(list("cab"), inplace=True)
```