# `D:\src\scipysrc\pandas\pandas\tests\libs\test_hashtable.py`

```
# 导入所需模块和类
from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc

# 导入第三方库
import numpy as np
import pytest

# 导入 pandas 相关模块
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin


# 上下文管理器，用于启动和停止内存跟踪
@contextmanager
def activated_tracemalloc() -> Generator[None, None, None]:
    tracemalloc.start()
    try:
        yield
    finally:
        tracemalloc.stop()


# 获取已分配内存中与哈希表相关的内存使用量
def get_allocated_khash_memory():
    # 获取当前内存快照
    snapshot = tracemalloc.take_snapshot()
    # 过滤内存跟踪，仅保留与哈希表有关的部分
    snapshot = snapshot.filter_traces(
        (tracemalloc.DomainFilter(True, ht.get_hashtable_trace_domain()),)
    )
    # 计算并返回内存使用总量
    return sum(x.size for x in snapshot.traces)


# 使用 pytest 的参数化装饰器指定不同的哈希表类型和数据类型进行测试
@pytest.mark.parametrize(
    "table_type, dtype",
    [
        (ht.PyObjectHashTable, np.object_),
        (ht.Complex128HashTable, np.complex128),
        (ht.Int64HashTable, np.int64),
        (ht.UInt64HashTable, np.uint64),
        (ht.Float64HashTable, np.float64),
        (ht.Complex64HashTable, np.complex64),
        (ht.Int32HashTable, np.int32),
        (ht.UInt32HashTable, np.uint32),
        (ht.Float32HashTable, np.float32),
        (ht.Int16HashTable, np.int16),
        (ht.UInt16HashTable, np.uint16),
        (ht.Int8HashTable, np.int8),
        (ht.UInt8HashTable, np.uint8),
        (ht.IntpHashTable, np.intp),
    ],
)
class TestHashTable:
    # 定义测试哈希表类
    def test_get_set_contains_len(self, table_type, dtype):
        # 设置索引值
        index = 5
        # 创建指定类型的哈希表对象
        table = table_type(55)

        # 断言哈希表长度为0
        assert len(table) == 0
        # 断言索引不在哈希表中
        assert index not in table

        # 设置索引为42的项，并进行断言验证
        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42

        # 设置索引为index+1的项，并进行断言验证
        table.set_item(index + 1, 41)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 42
        assert table.get_item(index + 1) == 41

        # 再次设置索引为index的项，进行更新验证
        table.set_item(index, 21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 21
        assert table.get_item(index + 1) == 41
        assert index + 2 not in table

        # 再次设置索引为index+1的项，进行更新验证
        table.set_item(index + 1, 21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 21
        assert table.get_item(index + 1) == 21

        # 使用 pytest 断言抛出 KeyError 异常，验证索引index+2不在哈希表中
        with pytest.raises(KeyError, match=str(index + 2)):
            table.get_item(index + 2)
    # 测试表格类型的 get/set/contains/len 掩码功能
    def test_get_set_contains_len_mask(self, table_type, dtype):
        # 如果表格类型为 PyObjectHashTable，则跳过测试，因为不支持掩码功能
        if table_type == ht.PyObjectHashTable:
            pytest.skip("Mask not supported for object")
        # 设置索引为 5
        index = 5
        # 创建指定类型的表格对象，使用掩码功能
        table = table_type(55, uses_mask=True)
        # 断言表格长度为 0
        assert len(table) == 0
        # 断言索引 5 不在表格中
        assert index not in table

        # 设置索引 5 的值为 42
        table.set_item(index, 42)
        # 断言表格长度为 1
        assert len(table) == 1
        # 断言索引 5 在表格中
        assert index in table
        # 断言获取索引 5 的值为 42
        assert table.get_item(index) == 42
        # 使用 pytest 检查获取 NA 值时抛出 KeyError 异常
        with pytest.raises(KeyError, match="NA"):
            table.get_na()

        # 设置索引 6 的值为 41，并将该值设为 NA
        table.set_item(index + 1, 41)
        table.set_na(41)
        # 断言 NA 值在表格中
        assert pd.NA in table
        # 断言索引 5 和 6 在表格中
        assert index in table
        assert index + 1 in table
        # 断言表格长度为 3
        assert len(table) == 3
        # 断言获取索引 5 的值为 42
        assert table.get_item(index) == 42
        # 断言获取索引 6 的值为 41
        assert table.get_item(index + 1) == 41
        # 断言获取 NA 值为 41
        assert table.get_na() == 41

        # 将索引 6 的值设为 NA 值 21
        table.set_na(21)
        # 断言索引 5 和 6 在表格中
        assert index in table
        assert index + 1 in table
        # 断言表格长度为 3
        assert len(table) == 3
        # 断言获取索引 6 的值为 41
        assert table.get_item(index + 1) == 41
        # 断言获取 NA 值为 21
        assert table.get_na() == 21
        # 断言索引 7 不在表格中
        assert index + 2 not in table

        # 使用 pytest 检查获取索引 7 的值时抛出 KeyError 异常
        with pytest.raises(KeyError, match=str(index + 2)):
            table.get_item(index + 2)

    # 测试将键映射到值的功能
    def test_map_keys_to_values(self, table_type, dtype, writable):
        # 只有 Int64HashTable 类型有此方法
        if table_type == ht.Int64HashTable:
            N = 77
            table = table_type()
            keys = np.arange(N).astype(dtype)
            vals = np.arange(N).astype(np.int64) + N
            keys.flags.writeable = writable
            vals.flags.writeable = writable
            # 使用键映射到值的方法
            table.map_keys_to_values(keys, vals)
            for i in range(N):
                # 断言键对应的值正确映射
                assert table.get_item(keys[i]) == i + N

    # 测试将位置映射到值的功能
    def test_map_locations(self, table_type, dtype, writable):
        N = 8
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        # 使用位置映射到值的方法
        table.map_locations(keys)
        for i in range(N):
            # 断言位置对应的值正确映射
            assert table.get_item(keys[i]) == i

    # 测试将位置映射到值的功能，并使用掩码
    def test_map_locations_mask(self, table_type, dtype, writable):
        # 如果表格类型为 PyObjectHashTable，则跳过测试，因为不支持掩码功能
        if table_type == ht.PyObjectHashTable:
            pytest.skip("Mask not supported for object")
        N = 3
        # 创建指定类型的表格对象，使用掩码功能
        table = table_type(uses_mask=True)
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        # 使用位置映射到值的方法，同时使用指定的掩码
        table.map_locations(keys, np.array([False, False, True]))
        for i in range(N - 1):
            # 断言位置对应的值正确映射
            assert table.get_item(keys[i]) == i

        # 使用 pytest 检查获取最后一个位置值时抛出 KeyError 异常
        with pytest.raises(KeyError, match=re.escape(str(keys[N - 1]))):
            table.get_item(keys[N - 1])

        # 断言获取 NA 值为 2
        assert table.get_na() == 2
    # 定义一个测试函数，用于测试查找功能
    def test_lookup(self, table_type, dtype, writable):
        # 定义测试数据的大小
        N = 3
        # 创建指定类型的表格对象
        table = table_type()
        # 生成一组包含N个元素的键，类型转换为指定的dtype
        keys = (np.arange(N) + N).astype(dtype)
        # 设置keys数组的可写标志
        keys.flags.writeable = writable
        # 将keys映射到表格中的位置
        table.map_locations(keys)
        # 查询表格中与keys对应的结果
        result = table.lookup(keys)
        # 生成预期的结果数组，从0到N-1
        expected = np.arange(N)
        # 断言结果与预期结果的数组相等
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))

    # 定义一个测试函数，用于测试查找错误键时的行为
    def test_lookup_wrong(self, table_type, dtype):
        # 根据数据类型选择合适的N值
        if dtype in (np.int8, np.uint8):
            N = 100
        else:
            N = 512
        # 创建指定类型的表格对象
        table = table_type()
        # 生成一组包含N个元素的键，类型转换为指定的dtype
        keys = (np.arange(N) + N).astype(dtype)
        # 将keys映射到表格中的位置
        table.map_locations(keys)
        # 创建一组错误的键，类型与keys相同
        wrong_keys = np.arange(N).astype(dtype)
        # 查询表格中与wrong_keys对应的结果
        result = table.lookup(wrong_keys)
        # 断言结果数组中所有元素均为-1
        assert np.all(result == -1)

    # 定义一个测试函数，用于测试带掩码的查找功能
    def test_lookup_mask(self, table_type, dtype, writable):
        # 如果表格类型是PyObjectHashTable，则跳过测试
        if table_type == ht.PyObjectHashTable:
            pytest.skip("Mask not supported for object")
        # 定义测试数据的大小
        N = 3
        # 创建支持掩码的指定类型的表格对象
        table = table_type(uses_mask=True)
        # 生成一组包含N个元素的键，类型转换为指定的dtype
        keys = (np.arange(N) + N).astype(dtype)
        # 创建一个掩码数组
        mask = np.array([False, True, False])
        # 设置keys数组的可写标志
        keys.flags.writeable = writable
        # 将带有掩码的keys映射到表格中的位置
        table.map_locations(keys, mask)
        # 使用带有掩码的keys查询表格中对应的结果
        result = table.lookup(keys, mask)
        # 生成预期的结果数组，从0到N-1
        expected = np.arange(N)
        # 断言结果与预期结果的数组相等
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))

        # 使用单个键和掩码进行查询
        result = table.lookup(np.array([1 + N]).astype(dtype), np.array([False]))
        # 断言结果数组与预期的值相等，预期值为-1
        tm.assert_numpy_array_equal(
            result.astype(np.int64), np.array([-1], dtype=np.int64)
        )

    # 定义一个测试函数，用于测试唯一性查询功能
    def test_unique(self, table_type, dtype, writable):
        # 根据数据类型选择合适的N值
        if dtype in (np.int8, np.uint8):
            N = 88
        else:
            N = 1000
        # 创建指定类型的表格对象
        table = table_type()
        # 生成预期的结果数组，从N到N-1，类型转换为指定的dtype
        expected = (np.arange(N) + N).astype(dtype)
        # 生成一个包含重复元素的键数组
        keys = np.repeat(expected, 5)
        # 设置keys数组的可写标志
        keys.flags.writeable = writable
        # 查询表格中唯一的键值
        unique = table.unique(keys)
        # 断言查询结果与预期的唯一键值数组相等
        tm.assert_numpy_array_equal(unique, expected)

    # 定义一个测试函数，用于测试内存分配追踪功能
    def test_tracemalloc_works(self, table_type, dtype):
        # 根据数据类型选择合适的N值
        if dtype in (np.int8, np.uint8):
            N = 256
        else:
            N = 30000
        # 生成一组包含N个元素的键，类型转换为指定的dtype
        keys = np.arange(N).astype(dtype)
        # 激活内存分配追踪功能
        with activated_tracemalloc():
            # 创建指定类型的表格对象
            table = table_type()
            # 将keys映射到表格中的位置
            table.map_locations(keys)
            # 获取已分配的内存大小
            used = get_allocated_khash_memory()
            # 获取表格对象的大小
            my_size = table.sizeof()
            # 断言已分配的内存大小与表格对象的大小相等
            assert used == my_size
            # 删除表格对象
            del table
            # 断言已分配的内存大小为0
            assert get_allocated_khash_memory() == 0

    # 定义一个测试函数，用于测试空表格的内存分配追踪功能
    def test_tracemalloc_for_empty(self, table_type, dtype):
        # 激活内存分配追踪功能
        with activated_tracemalloc():
            # 创建指定类型的表格对象
            table = table_type()
            # 获取已分配的内存大小
            used = get_allocated_khash_memory()
            # 获取表格对象的大小
            my_size = table.sizeof()
            # 断言已分配的内存大小与表格对象的大小相等
            assert used == my_size
            # 删除表格对象
            del table
            # 断言已分配的内存大小为0
            assert get_allocated_khash_memory() == 0
    # 定义一个测试方法，用于测试获取数据结构状态的方法
    def test_get_state(self, table_type, dtype):
        # 创建一个指定类型和大小的数据结构对象
        table = table_type(1000)
        # 获取当前数据结构对象的状态信息
        state = table.get_state()
        # 断言数据结构大小为零
        assert state["size"] == 0
        # 断言数据结构中被占用的元素数为零
        assert state["n_occupied"] == 0
        # 断言状态信息中包含键 "n_buckets"
        assert "n_buckets" in state
        # 断言状态信息中包含键 "upper_bound"
        assert "upper_bound" in state

    # 使用参数化装饰器定义一个参数化测试方法，测试数据结构不重新分配空间的情况
    @pytest.mark.parametrize("N", range(1, 110))
    def test_no_reallocation(self, table_type, dtype, N):
        # 创建一个包含 N 个元素的数组作为键，并指定数据类型
        keys = np.arange(N).astype(dtype)
        # 创建一个预先分配了 N 大小的数据结构对象
        preallocated_table = table_type(N)
        # 记录预分配表的初始桶数
        n_buckets_start = preallocated_table.get_state()["n_buckets"]
        # 将键映射到预分配表中
        preallocated_table.map_locations(keys)
        # 记录映射后预分配表的桶数
        n_buckets_end = preallocated_table.get_state()["n_buckets"]
        # 断言：映射前后桶数应保持不变
        assert n_buckets_start == n_buckets_end
        # 创建一个未预分配空间的新数据结构对象
        clean_table = table_type()
        # 将键映射到新表中
        clean_table.map_locations(keys)
        # 断言：映射前后新表的桶数应与初始桶数相同
        assert n_buckets_start == clean_table.get_state()["n_buckets"]
# 定义一个测试类 TestHashTableUnsorted，用于测试无序哈希表的功能
class TestHashTableUnsorted:
    # TODO: moved from test_algos; may be redundancies with other tests
    # 测试字符串哈希表的 set_item 方法签名是否正确
    def test_string_hashtable_set_item_signature(self):
        # GH#30419 fix typing in StringHashTable.set_item to prevent segfault
        # 创建一个 StringHashTable 的实例 tbl
        tbl = ht.StringHashTable()

        # 在 tbl 中设置键为 "key"，值为 1
        tbl.set_item("key", 1)
        # 断言获取键 "key" 对应的值为 1
        assert tbl.get_item("key") == 1

        # 使用 pytest 断言捕获 TypeError 异常，检查传入的键不是字符串类型时是否会抛出异常
        with pytest.raises(TypeError, match="'key' has incorrect type"):
            # key arg typed as string, not object
            tbl.set_item(4, 6)
        # 使用 pytest 断言捕获 TypeError 异常，检查传入的键不是字符串类型时是否会抛出异常
        with pytest.raises(TypeError, match="'val' has incorrect type"):
            tbl.get_item(4)

    # 测试处理 NaN 值的情况
    def test_lookup_nan(self, writable):
        # GH#21688 ensure we can deal with readonly memory views
        # 创建一个包含 NaN 值的 numpy 数组 xs
        xs = np.array([2.718, 3.14, np.nan, -7, 5, 2, 3])
        xs.setflags(write=writable)
        # 创建一个 Float64HashTable 的实例 m
        m = ht.Float64HashTable()
        # 将 xs 的位置映射到哈希表 m 中
        m.map_locations(xs)
        # 使用哈希表 m 查找 xs 中的元素，并断言结果与预期的索引数组相等
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    # 测试添加带符号的零值的情况
    def test_add_signed_zeros(self):
        # GH#21866 inconsistent hash-function for float64
        # default hash-function would lead to different hash-buckets
        # for 0.0 and -0.0 if there are more than 2^30 hash-buckets
        # but this would mean 16GB
        # 设置一个大数 N，如果内存足够则会触发错误
        N = 4  # 12 * 10**8 would trigger the error, if you have enough memory
        # 创建一个 Float64HashTable 的实例 m，指定大小为 N
        m = ht.Float64HashTable(N)
        # 在哈希表 m 中设置键 0.0 对应的值为 0
        m.set_item(0.0, 0)
        # 在哈希表 m 中设置键 -0.0 对应的值为 0
        m.set_item(-0.0, 0)
        # 断言哈希表 m 的长度为 1，说明 0.0 和 -0.0 在哈希表中被视为等价
        assert len(m) == 1  # 0.0 and -0.0 are equivalent

    # 测试添加不同的 NaN 值的情况
    def test_add_different_nans(self):
        # GH#21866 inconsistent hash-function for float64
        # create different nans from bit-patterns:
        # 使用结构体打包和解包创建不同的 NaN 值
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        # 断言不同的 NaN 值不相等
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        # 创建一个 Float64HashTable 的实例 m
        m = ht.Float64HashTable()
        # 在哈希表 m 中设置键 NAN1 对应的值为 0
        m.set_item(NAN1, 0)
        # 在哈希表 m 中设置键 NAN2 对应的值为 0
        m.set_item(NAN2, 0)
        # 断言哈希表 m 的长度为 1，说明 NAN1 和 NAN2 在哈希表中被视为等价
        assert len(m) == 1  # NAN1 and NAN2 are equivalent

    # 测试查找溢出的情况
    def test_lookup_overflow(self, writable):
        # 创建一个包含大整数的 numpy 数组 xs
        xs = np.array([1, 2, 2**63], dtype=np.uint64)
        # GH 21688 ensure we can deal with readonly memory views
        xs.setflags(write=writable)
        # 创建一个 UInt64HashTable 的实例 m
        m = ht.UInt64HashTable()
        # 将 xs 的位置映射到哈希表 m 中
        m.map_locations(xs)
        # 使用哈希表 m 查找 xs 中的元素，并断言结果与预期的索引数组相等
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    # 使用 pytest 的参数化标记，测试不同 nvals 的情况，包括特殊情况 nvals=0
    @pytest.mark.parametrize("nvals", [0, 10])  # resizing to 0 is special case
    @pytest.mark.parametrize(
        "htable, uniques, dtype, safely_resizes",
        [
            (ht.PyObjectHashTable, ht.ObjectVector, "object", False),
            (ht.StringHashTable, ht.ObjectVector, "object", True),
            (ht.Float64HashTable, ht.Float64Vector, "float64", False),
            (ht.Int64HashTable, ht.Int64Vector, "int64", False),
            (ht.Int32HashTable, ht.Int32Vector, "int32", False),
            (ht.UInt64HashTable, ht.UInt64Vector, "uint64", False),
        ],
    )
    def test_vector_resize(
        self, writable, htable, uniques, dtype, safely_resizes, nvals
    ):
        # Test for memory errors after internal vector
        # reallocations (GH 7157)
        # Changed from using np.random.default_rng(2).rand to range
        # which could cause flaky CI failures when safely_resizes=False

        # 创建一个包含1000个元素的 numpy 数组，元素类型由参数 dtype 指定
        vals = np.array(range(1000), dtype=dtype)

        # 根据 writable 参数设置 vals 数组是否可写
        vals.setflags(write=writable)

        # 创建哈希表和向量实例
        htable = htable()
        uniques = uniques()

        # 使用哈希表的方法将 vals 数组的前 nvals 个元素添加到 uniques 中
        htable.get_labels(vals[:nvals], uniques, 0, -1)
        
        # 将 uniques 转换为数组，并保存其原始形状
        tmp = uniques.to_array()
        oldshape = tmp.shape

        # 根据 safely_resizes 参数测试哈希表的行为，若为 False，预期引发 ValueError 异常
        if safely_resizes:
            htable.get_labels(vals, uniques, 0, -1)
        else:
            with pytest.raises(ValueError, match="external reference.*"):
                htable.get_labels(vals, uniques, 0, -1)

        # 验证转换后的 uniques 的形状未发生变化
        uniques.to_array()  # should not raise here
        assert tmp.shape == oldshape

    @pytest.mark.parametrize(
        "hashtable",
        [
            ht.PyObjectHashTable,
            ht.StringHashTable,
            ht.Float64HashTable,
            ht.Int64HashTable,
            ht.Int32HashTable,
            ht.UInt64HashTable,
        ],
    )
    def test_hashtable_large_sizehint(self, hashtable):
        # GH#22729 smoketest for not raising when passing a large size_hint

        # 设置一个较大的 size_hint 值，验证哈希表在此情况下不会引发异常
        size_hint = np.iinfo(np.uint32).max + 1
        hashtable(size_hint=size_hint)
class TestPyObjectHashTableWithNans:
    # 测试特殊情况下哈希表对 NaN 值的处理

    def test_nan_float(self):
        # 创建两个 NaN 浮点数对象
        nan1 = float("nan")
        nan2 = float("nan")
        # 断言两个 NaN 对象不是同一个对象
        assert nan1 is not nan2
        # 创建 PyObjectHashTable 对象
        table = ht.PyObjectHashTable()
        # 向哈希表中插入键值对
        table.set_item(nan1, 42)
        # 断言从哈希表中获取 nan2 对应的值为 42
        assert table.get_item(nan2) == 42

    def test_nan_complex_both(self):
        # 创建两个具有复数部分为 NaN 的复数对象
        nan1 = complex(float("nan"), float("nan"))
        nan2 = complex(float("nan"), float("nan"))
        # 断言两个复数对象不是同一个对象
        assert nan1 is not nan2
        # 创建 PyObjectHashTable 对象
        table = ht.PyObjectHashTable()
        # 向哈希表中插入键值对
        table.set_item(nan1, 42)
        # 断言从哈希表中获取 nan2 对应的值为 42
        assert table.get_item(nan2) == 42

    def test_nan_complex_real(self):
        # 创建具有实部为 NaN 的复数对象
        nan1 = complex(float("nan"), 1)
        nan2 = complex(float("nan"), 1)
        other = complex(float("nan"), 2)
        # 断言两个复数对象不是同一个对象
        assert nan1 is not nan2
        # 创建 PyObjectHashTable 对象
        table = ht.PyObjectHashTable()
        # 向哈希表中插入键值对
        table.set_item(nan1, 42)
        # 断言从哈希表中获取 nan2 对应的值为 42
        assert table.get_item(nan2) == 42
        # 断言从哈希表中获取 other 会引发 KeyError 异常，并检查错误消息
        with pytest.raises(KeyError, match=None) as error:
            table.get_item(other)
        assert str(error.value) == str(other)

    def test_nan_complex_imag(self):
        # 创建具有虚部为 NaN 的复数对象
        nan1 = complex(1, float("nan"))
        nan2 = complex(1, float("nan"))
        other = complex(2, float("nan"))
        # 断言两个复数对象不是同一个对象
        assert nan1 is not nan2
        # 创建 PyObjectHashTable 对象
        table = ht.PyObjectHashTable()
        # 向哈希表中插入键值对
        table.set_item(nan1, 42)
        # 断言从哈希表中获取 nan2 对应的值为 42
        assert table.get_item(nan2) == 42
        # 断言从哈希表中获取 other 会引发 KeyError 异常，并检查错误消息
        with pytest.raises(KeyError, match=None) as error:
            table.get_item(other)
        assert str(error.value) == str(other)

    def test_nan_in_tuple(self):
        # 创建包含 NaN 元组对象
        nan1 = (float("nan"),)
        nan2 = (float("nan"),)
        # 断言两个元组对象中的 NaN 元素不是同一个对象
        assert nan1[0] is not nan2[0]
        # 创建 PyObjectHashTable 对象
        table = ht.PyObjectHashTable()
        # 向哈希表中插入键值对
        table.set_item(nan1, 42)
        # 断言从哈希表中获取 nan2 对应的值为 42
        assert table.get_item(nan2) == 42

    def test_nan_in_nested_tuple(self):
        # 创建嵌套包含 NaN 的元组对象
        nan1 = (1, (2, (float("nan"),)))
        nan2 = (1, (2, (float("nan"),)))
        other = (1, 2)
        # 创建 PyObjectHashTable 对象
        table = ht.PyObjectHashTable()
        # 向哈希表中插入键值对
        table.set_item(nan1, 42)
        # 断言从哈希表中获取 nan2 对应的值为 42
        assert table.get_item(nan2) == 42
        # 断言从哈希表中获取 other 会引发 KeyError 异常，并检查错误消息
        with pytest.raises(KeyError, match=None) as error:
            table.get_item(other)
        assert str(error.value) == str(other)


# 测试当哈希表中包含 NaN 的元组时，哈希值和对象相等性的情况
def test_hash_equal_tuple_with_nans():
    a = (float("nan"), (float("nan"), float("nan")))
    b = (float("nan"), (float("nan"), float("nan")))
    # 断言使用 object_hash 函数计算的哈希值相等
    assert ht.object_hash(a) == ht.object_hash(b)
    # 断言使用 objects_are_equal 函数判断对象相等
    assert ht.objects_are_equal(a, b)


# 测试 Int64HashTable 类的 get_labels_groupby 方法
def test_get_labels_groupby_for_Int64(writable):
    table = ht.Int64HashTable()
    vals = np.array([1, 2, -1, 2, 1, -1], dtype=np.int64)
    vals.flags.writeable = writable
    # 调用 get_labels_groupby 方法
    arr, unique = table.get_labels_groupby(vals)
    expected_arr = np.array([0, 1, -1, 1, 0, -1], dtype=np.intp)
    expected_unique = np.array([1, 2], dtype=np.int64)
    # 断言返回的结果与预期的数组相等
    tm.assert_numpy_array_equal(arr, expected_arr)
    tm.assert_numpy_array_equal(unique, expected_unique)


# 测试 StringHashTable 类的 tracemalloc_works_for_StringHashTable 方法
def test_tracemalloc_works_for_StringHashTable():
    N = 1000
    keys = np.arange(N).astype(np.str_).astype(np.object_)
    # 此方法主要用于性能和内存分析，不需要额外注释
    # 使用 activated_tracemalloc() 启用内存分配跟踪
    with activated_tracemalloc():
        # 创建一个空的字符串哈希表对象
        table = ht.StringHashTable()
        # 将 keys 映射到哈希表中的位置
        table.map_locations(keys)
        # 获取当前分配给 khash 的内存量
        used = get_allocated_khash_memory()
        # 获取哈希表的大小（占用的内存量）
        my_size = table.sizeof()
        # 断言已使用的内存量与哈希表的大小相等
        assert used == my_size
        # 删除哈希表对象
        del table
        # 断言此时 khash 的内存分配为零
        assert get_allocated_khash_memory() == 0
# 定义测试函数，测试当 StringHashTable 为空时的内存分配情况
def test_tracemalloc_for_empty_StringHashTable():
    # 使用 activated_tracemalloc 上下文管理器激活内存跟踪
    with activated_tracemalloc():
        # 创建 StringHashTable 对象
        table = ht.StringHashTable()
        # 获取当前分配给 KHash 的内存量
        used = get_allocated_khash_memory()
        # 获取 StringHashTable 对象的大小
        my_size = table.sizeof()
        # 断言实际使用的内存等于 StringHashTable 对象的大小
        assert used == my_size
        # 删除 StringHashTable 对象
        del table
        # 断言所有 KHash 内存都已释放
        assert get_allocated_khash_memory() == 0


# 使用参数化测试，测试当 StringHashTable 不需要重新分配空间时的情况
@pytest.mark.parametrize("N", range(1, 110))
def test_no_reallocation_StringHashTable(N):
    # 创建包含 N 个字符串键的数组
    keys = np.arange(N).astype(np.str_).astype(np.object_)
    # 创建预分配 N 个桶的 StringHashTable 对象
    preallocated_table = ht.StringHashTable(N)
    # 获取预分配表的初始桶数
    n_buckets_start = preallocated_table.get_state()["n_buckets"]
    # 将键映射到预分配的表中
    preallocated_table.map_locations(keys)
    # 获取映射后的桶数
    n_buckets_end = preallocated_table.get_state()["n_buckets"]
    # 断言原始桶数足够，无需重新分配
    assert n_buckets_start == n_buckets_end
    # 创建一个干净的 StringHashTable 对象，并将键映射到其中
    clean_table = ht.StringHashTable()
    clean_table.map_locations(keys)
    # 再次断言桶数与预期一致
    assert n_buckets_start == clean_table.get_state()["n_buckets"]


# 使用参数化测试，测试不同类型的 HashTable 对象处理 NaN 值的情况
@pytest.mark.parametrize(
    "table_type, dtype",
    [
        (ht.Float64HashTable, np.float64),
        (ht.Float32HashTable, np.float32),
        (ht.Complex128HashTable, np.complex128),
        (ht.Complex64HashTable, np.complex64),
    ],
)
class TestHashTableWithNans:
    # 测试 HashTable 对象的 get/set/contains/len 方法
    def test_get_set_contains_len(self, table_type, dtype):
        # 创建 NaN 值的索引
        index = float("nan")
        # 创建指定类型的 HashTable 对象
        table = table_type()
        # 断言 NaN 值不在表中
        assert index not in table

        # 设置 NaN 值对应的条目为 42
        table.set_item(index, 42)
        # 断言表中的条目数为 1
        assert len(table) == 1
        # 断言 NaN 值现在在表中
        assert index in table
        # 断言获取到的 NaN 值对应的数据为 42
        assert table.get_item(index) == 42

        # 再次设置 NaN 值对应的条目为 41
        table.set_item(index, 41)
        # 断言表中的条目数为 1
        assert len(table) == 1
        # 断言 NaN 值仍然在表中
        assert index in table
        # 断言获取到的 NaN 值对应的数据为 41
        assert table.get_item(index) == 41

    # 测试 HashTable 对象的 map_locations 方法
    def test_map_locations(self, table_type, dtype):
        # 设置数组大小 N
        N = 10
        # 创建指定类型的 HashTable 对象
        table = table_type()
        # 创建包含 N 个 NaN 值的键数组
        keys = np.full(N, np.nan, dtype=dtype)
        # 将键映射到 HashTable 中
        table.map_locations(keys)
        # 断言表中的条目数为 1
        assert len(table) == 1
        # 断言获取到 NaN 值对应的数据为 N - 1
        assert table.get_item(np.nan) == N - 1

    # 测试 HashTable 对象的 unique 方法
    def test_unique(self, table_type, dtype):
        # 设置数组大小 N
        N = 1020
        # 创建指定类型的 HashTable 对象
        table = table_type()
        # 创建包含 N 个 NaN 值的键数组
        keys = np.full(N, np.nan, dtype=dtype)
        # 调用 unique 方法获取唯一值数组
        unique = table.unique(keys)
        # 断言唯一值数组中所有值都是 NaN，并且数组长度为 1
        assert np.all(np.isnan(unique)) and len(unique) == 1


# 测试处理浮点数 NaN 值的情况
def test_unique_for_nan_objects_floats():
    # 创建 PyObjectHashTable 对象
    table = ht.PyObjectHashTable()
    # 创建包含 50 个浮点数 NaN 值的键数组
    keys = np.array([float("nan") for i in range(50)], dtype=np.object_)
    # 调用 unique 方法获取唯一值数组
    unique = table.unique(keys)
    # 断言唯一值数组长度为 1
    assert len(unique) == 1


# 测试处理复数 NaN 值的情况
def test_unique_for_nan_objects_complex():
    # 创建 PyObjectHashTable 对象
    table = ht.PyObjectHashTable()
    # 创建包含 50 个复数 NaN 值的键数组
    keys = np.array([complex(float("nan"), 1.0) for i in range(50)], dtype=np.object_)
    # 调用 unique 方法获取唯一值数组
    unique = table.unique(keys)
    # 断言唯一值数组长度为 1
    assert len(unique) == 1


# 测试处理元组中包含 NaN 值的情况
def test_unique_for_nan_objects_tuple():
    # 创建 PyObjectHashTable 对象
    table = ht.PyObjectHashTable()
    # 创建包含 50 个元组，其中某个元素为 NaN 值的键数组
    keys = np.array(
        [1] + [(1.0, (float("nan"), 1.0)) for i in range(50)], dtype=np.object_
    )
    # 调用 unique 方法获取唯一值数组
    unique = table.unique(keys)
    # 断言唯一值数组长度为 2
    assert len(unique) == 2


# 使用参数化测试，测试处理不同类型数据的 PyObjectHashTable 对象的 unique 方法
@pytest.mark.parametrize(
    "dtype",
    [
        # numpy 的数据类型 np.object_
        np.object_,
        # 复数的 128 位表示 np.complex128
        np.complex128,
        # 64 位整数 np.int64
        np.int64,
        # 64 位无符号整数 np.uint64
        np.uint64,
        # 64 位浮点数 np.float64
        np.float64,
        # 64 位复数 np.complex64
        np.complex64,
        # 32 位整数 np.int32
        np.int32,
        # 32 位无符号整数 np.uint32
        np.uint32,
        # 32 位浮点数 np.float32
        np.float32,
        # 16 位整数 np.int16
        np.int16,
        # 16 位无符号整数 np.uint16
        np.uint16,
        # 8 位整数 np.int8
        np.int8,
        # 8 位无符号整数 np.uint8
        np.uint8,
        # 指针大小的整数 np.intp
        np.intp,
    ],
# 定义一个名为 TestHelpFunctions 的测试类
class TestHelpFunctions:
    
    # 测试函数：测试值计数功能
    def test_value_count(self, dtype, writable):
        # 设置常量 N 为 43
        N = 43
        # 生成期望的数组，类型转换为给定的 dtype，其中包含从 0 到 N-1 的整数
        expected = (np.arange(N) + N).astype(dtype)
        # 将期望的数组重复五次，得到一个长数组 values
        values = np.repeat(expected, 5)
        # 设置数组 values 是否可写属性
        values.flags.writeable = writable
        # 调用 ht.value_count 函数，获取返回的 keys, counts 和一个占位符
        keys, counts, _ = ht.value_count(values, False)
        # 断言排序后的 keys 等于期望的数组 expected
        tm.assert_numpy_array_equal(np.sort(keys), expected)
        # 断言 counts 中所有值均为 5
        assert np.all(counts == 5)

    # 测试函数：测试带掩码的值计数功能
    def test_value_count_mask(self, dtype):
        # 如果 dtype 是 np.object_ 类型，则跳过测试
        if dtype == np.object_:
            pytest.skip("mask not implemented for object dtype")
        # 创建一个数组 values，元素全为 1，类型为给定的 dtype
        values = np.array([1] * 5, dtype=dtype)
        # 创建一个布尔掩码数组 mask，长度为 5
        mask = np.zeros((5,), dtype=np.bool_)
        # 将 mask 的第二个和第五个元素设为 True
        mask[1] = True
        mask[4] = True
        # 调用 ht.value_count 函数，获取返回的 keys, counts 和 NA 计数器
        keys, counts, na_counter = ht.value_count(values, False, mask=mask)
        # 断言 keys 的长度为 2
        assert len(keys) == 2
        # 断言 NA 计数器为 2
        assert na_counter == 2

    # 测试函数：测试稳定性的值计数功能
    def test_value_count_stable(self, dtype, writable):
        # GH12679
        # 创建一个数组 values，包含一些整数，类型转换为给定的 dtype
        values = np.array([2, 1, 5, 22, 3, -1, 8]).astype(dtype)
        # 设置数组 values 是否可写属性
        values.flags.writeable = writable
        # 调用 ht.value_count 函数，获取返回的 keys, counts 和一个占位符
        keys, counts, _ = ht.value_count(values, False)
        # 断言 keys 与 values 数组相等
        tm.assert_numpy_array_equal(keys, values)
        # 断言 counts 中所有值均为 1
        assert np.all(counts == 1)

    # 测试函数：测试重复值检测功能
    def test_duplicated_first(self, dtype, writable):
        # 设置常量 N 为 100
        N = 100
        # 创建一个数组 values，包含 N 个数重复 5 次，类型转换为给定的 dtype
        values = np.repeat(np.arange(N).astype(dtype), 5)
        # 设置数组 values 是否可写属性
        values.flags.writeable = writable
        # 调用 ht.duplicated 函数，获取返回的结果 result
        result = ht.duplicated(values)
        # 创建一个期望的布尔数组 expected，与 values 大小相同，初始值为 True
        expected = np.ones_like(values, dtype=np.bool_)
        # 将 expected 中每隔 5 个元素设为 False
        expected[::5] = False
        # 断言 result 与 expected 数组相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试函数：测试是否成员功能，返回结果应为全部是 True
    def test_ismember_yes(self, dtype, writable):
        # 设置常量 N 为 127
        N = 127
        # 创建一个数组 arr，包含从 0 到 N-1 的整数，类型转换为给定的 dtype
        arr = np.arange(N).astype(dtype)
        # 创建一个数组 values，与 arr 相同
        values = np.arange(N).astype(dtype)
        # 设置数组 arr 和 values 是否可写属性
        arr.flags.writeable = writable
        values.flags.writeable = writable
        # 调用 ht.ismember 函数，获取返回的结果 result
        result = ht.ismember(arr, values)
        # 创建一个期望的布尔数组 expected，与 values 大小相同，初始值为 True
        expected = np.ones_like(values, dtype=np.bool_)
        # 断言 result 与 expected 数组相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试函数：测试是否成员功能，返回结果应为全部是 False
    def test_ismember_no(self, dtype):
        # 设置常量 N 为 17
        N = 17
        # 创建一个数组 arr，包含从 0 到 N-1 的整数，类型转换为给定的 dtype
        arr = np.arange(N).astype(dtype)
        # 创建一个数组 values，元素为 arr 中的元素加上 N，类型转换为给定的 dtype
        values = (np.arange(N) + N).astype(dtype)
        # 调用 ht.ismember 函数，获取返回的结果 result
        result = ht.ismember(arr, values)
        # 创建一个期望的布尔数组 expected，与 values 大小相同，初始值为 False
        expected = np.zeros_like(values, dtype=np.bool_)
        # 断言 result 与 expected 数组相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试函数：测试众数计算功能，期望结果为 42
    def test_mode(self, dtype, writable):
        # 根据 dtype 类型选择常量 N
        if dtype in (np.int8, np.uint8):
            N = 53
        else:
            N = 11111
        # 创建一个数组 values，包含 N 个数重复 5 次，类型转换为给定的 dtype
        values = np.repeat(np.arange(N).astype(dtype), 5)
        # 将 values 的第一个元素设为 42
        values[0] = 42
        # 设置数组 values 是否可写属性
        values.flags.writeable = writable
        # 调用 ht.mode 函数，获取返回的 keys
        result = ht.mode(values, False)[0]
        # 断言结果与期望的值 42 相等
        assert result == 42

    # 测试函数：测试稳定性的众数计算功能
    def test_mode_stable(self, dtype, writable):
        # 创建一个数组 values，包含一些整数，类型转换为给定的 dtype
        values = np.array([2, 1, 5, 22, 3, -1, 8]).astype(dtype)
        # 设置数组 values 是否可写属性
        values.flags.writeable = writable
        # 调用 ht.mode 函数，获取返回的 keys
        keys = ht.mode(values, False)[0]
        # 断言 keys 与 values 数组相等
        tm.assert_numpy_array_equal(keys, values)


# 测试函数：测试包含 NaN 的众数计算功能
def test_modes_with_nans():
    # GH42688, nans aren't mangled
    # 创建一个包含 NaN、np.nan、pd.NaT 和 None 的列表 nulls
    nulls = [pd.NA, np.nan, pd.NaT, None]
    # 创建一个数组 values，包含 True 和 nulls 列表的内容各重复两次，类型为 np.object_
    values = np.array([True] + nulls * 2, dtype=np.object_)
    # 调用 ht.mode 函数，获取返回的众数 modes
    modes = ht.mode(values, False)[0]
    # 断言 modes 的大小与 nulls 列表的长度相等
    assert modes.size == len(nulls)
def test_unique_label_indices_intp(writable):
    # 创建一个包含整数指针的 NumPy 数组作为测试数据
    keys = np.array([1, 2, 2, 2, 1, 3], dtype=np.intp)
    # 设置数组可写属性，用于测试
    keys.flags.writeable = writable
    # 调用被测试的函数，返回计算结果
    result = ht.unique_label_indices(keys)
    # 预期结果，作为比较基准
    expected = np.array([0, 1, 5], dtype=np.intp)
    # 使用测试框架断言两个 NumPy 数组相等
    tm.assert_numpy_array_equal(result, expected)


def test_unique_label_indices():
    # 生成一个大的随机整数指针 NumPy 数组
    a = np.random.default_rng(2).integers(1, 1 << 10, 1 << 15).astype(np.intp)

    # 调用函数获取左侧的唯一标签索引
    left = ht.unique_label_indices(a)
    # 使用 NumPy 自带函数获取右侧的唯一标签索引
    right = np.unique(a, return_index=True)[1]

    # 使用测试框架断言两个 NumPy 数组相等
    tm.assert_numpy_array_equal(left, right, check_dtype=False)

    # 将数组中的部分值设置为 -1
    a[np.random.default_rng(2).choice(len(a), 10)] = -1
    # 重新调用函数获取左侧的唯一标签索引
    left = ht.unique_label_indices(a)
    # 再次使用 NumPy 自带函数获取右侧的唯一标签索引，并且跳过第一个索引
    right = np.unique(a, return_index=True)[1][1:]
    # 使用测试框架断言两个 NumPy 数组相等
    tm.assert_numpy_array_equal(left, right, check_dtype=False)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float64,
        np.float32,
        np.complex128,
        np.complex64,
    ],
)
class TestHelpFunctionsWithNans:
    def test_value_count(self, dtype):
        # 创建一个包含 NaN 值的 NumPy 数组
        values = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        # 调用函数计算值的计数
        keys, counts, _ = ht.value_count(values, True)
        # 断言键数组的长度为 0
        assert len(keys) == 0
        # 再次调用函数，获取不忽略 NaN 值的计数结果
        keys, counts, _ = ht.value_count(values, False)
        # 断言键数组的长度为 1，且所有值都是 NaN
        assert len(keys) == 1 and np.all(np.isnan(keys))
        # 断言计数数组的第一个元素为 3
        assert counts[0] == 3

    def test_duplicated_first(self, dtype):
        # 创建一个包含 NaN 值的 NumPy 数组
        values = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        # 调用函数查找重复的值
        result = ht.duplicated(values)
        # 预期的重复标记数组
        expected = np.array([False, True, True])
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_yes(self, dtype):
        # 创建一个包含 NaN 值的 NumPy 数组
        arr = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        # 创建一个包含 NaN 值的查找数组
        values = np.array([np.nan, np.nan], dtype=dtype)
        # 调用函数检查成员关系
        result = ht.ismember(arr, values)
        # 预期的成员关系结果
        expected = np.array([True, True, True], dtype=np.bool_)
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_no(self, dtype):
        # 创建一个包含 NaN 值的 NumPy 数组
        arr = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        # 创建一个包含整数值的查找数组
        values = np.array([1], dtype=dtype)
        # 调用函数检查成员关系
        result = ht.ismember(arr, values)
        # 预期的成员关系结果
        expected = np.array([False, False, False], dtype=np.bool_)
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(result, expected)

    def test_mode(self, dtype):
        # 创建一个包含整数和 NaN 值的 NumPy 数组
        values = np.array([42, np.nan, np.nan, np.nan], dtype=dtype)
        # 调用函数计算众数
        assert ht.mode(values, True)[0] == 42
        # 调用函数计算众数，不忽略 NaN 值
        assert np.isnan(ht.mode(values, False)[0])


def test_ismember_tuple_with_nans():
    # GH-41836 测试案例
    # 创建一个空的对象数组，包含元组和 NaN 值
    values = np.empty(2, dtype=object)
    values[:] = [("a", float("nan")), ("b", 1)]
    # 创建一个包含元组的比较数组
    comps = [("a", float("nan"))]

    # 调用函数检查成员关系
    result = isin(values, comps)
    # 预期的成员关系结果
    expected = np.array([True, False], dtype=np.bool_)
    # 使用测试框架断言两个 NumPy 数组相等
    tm.assert_numpy_array_equal(result, expected)


def test_float_complex_int_are_equal_as_objects():
    # 创建一个包含不同类型对象的列表
    values = ["a", 5, 5.0, 5.0 + 0j]
    # 创建一个包含整数范围的列表
    comps = list(range(129))
    # 调用函数检查成员关系
    result = isin(np.array(values, dtype=object), np.asarray(comps))
    # 预期的成员关系结果
    expected = np.array([False, True, True, True], dtype=np.bool_)
    # 使用测试框架断言两个 NumPy 数组相等
    tm.assert_numpy_array_equal(result, expected)
```