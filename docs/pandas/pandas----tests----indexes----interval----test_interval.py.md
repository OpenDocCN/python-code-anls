# `D:\src\scipysrc\pandas\pandas\tests\indexes\interval\test_interval.py`

```
@pytest.mark.parametrize(
    "breaks",
    # 参数化测试用例的 breaks 参数，用于测试不同类型的间隔
    [
        [1, 1, 2, 5, 15, 53, 217, 1014, 5335, 31240, 201608],  # 整数列表
        [-np.inf, -100, -10, 0.5, 1, 1.5, 3.8, 101, 202, np.inf],  # 包含负无穷到正无穷的列表
        date_range("2017-01-01", "2017-01-04"),  # 日期范围对象
        pytest.param(
            date_range("2017-01-01", "2017-01-04", unit="s"),
            marks=pytest.mark.xfail(reason="mismatched result unit"),  # 使用 pytest.mark.xfail 标记的日期范围对象
        ),
        pd.to_timedelta(["1ns", "2ms", "3s", "4min", "5h", "6D"]),  # 时间增量对象
    ],
)
    # 定义测试方法，用于测试 IntervalIndex 的长度计算功能
    def test_length(self, closed, breaks):
        # GH 18789: 对于给定的断点 breaks 和闭合方式 closed，创建 IntervalIndex 对象
        index = IntervalIndex.from_breaks(breaks, closed=closed)
        # 获取 IntervalIndex 对象的长度
        result = index.length
        # 期望的结果是一个 Index 对象，包含每个区间的长度
        expected = Index(iv.length for iv in index)
        # 断言结果与期望值相等
        tm.assert_index_equal(result, expected)

        # 在包含 NaN 的情况下再次测试
        # 向 index 中插入一个 NaN 值
        index = index.insert(1, np.nan)
        # 获取新的 IntervalIndex 对象的长度
        result = index.length
        # 期望的结果是一个 Index 对象，如果区间不包含 NaN 则是其长度，否则是 NaN
        expected = Index(iv.length if notna(iv) else iv for iv in index)
        # 断言结果与期望值相等
        tm.assert_index_equal(result, expected)

    # 测试包含 NaN 值的情况
    def test_with_nans(self, closed):
        # 创建一个不包含 NaN 的 IntervalIndex 对象，并断言其没有 NaN 值
        index = self.create_index(closed=closed)
        assert index.hasnans is False

        # 测试 isna() 方法，期望所有值都不是 NaN
        result = index.isna()
        expected = np.zeros(len(index), dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        # 测试 notna() 方法，期望所有值都不是 NaN
        result = index.notna()
        expected = np.ones(len(index), dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        # 创建一个包含 NaN 值的 IntervalIndex 对象，并断言其包含 NaN 值
        index = self.create_index_with_nan(closed=closed)
        assert index.hasnans is True

        # 测试 isna() 方法，期望第二个值是 NaN，其余不是
        expected = np.array([False, True] + [False] * (len(index) - 2))
        tm.assert_numpy_array_equal(result, expected)

        # 测试 notna() 方法，期望第二个值不是 NaN，其余是
        expected = np.array([True, False] + [True] * (len(index) - 2))
        tm.assert_numpy_array_equal(result, expected)

    # 测试 copy() 方法
    def test_copy(self, closed):
        # 创建一个 IntervalIndex 对象
        expected = self.create_index(closed=closed)

        # 测试浅拷贝，断言拷贝后的对象与原对象相等
        result = expected.copy()
        assert result.equals(expected)

        # 测试深拷贝，断言拷贝后的对象与原对象相等，但对象不是同一个
        result = expected.copy(deep=True)
        assert result.equals(expected)
        assert result.left is not expected.left

    # 测试构造函数中的 copy 标志
    def test_ensure_copied_data(self, closed):
        # 测试构造函数中的 copy=False 参数

        # 不拷贝情况下创建一个 IntervalIndex 对象，并断言其左右端点数组相同
        index = self.create_index(closed=closed)
        result = IntervalIndex(index, copy=False)
        tm.assert_numpy_array_equal(
            index.left.values, result.left.values, check_same="same"
        )
        tm.assert_numpy_array_equal(
            index.right.values, result.right.values, check_same="same"
        )

        # 通过定义的方式强制拷贝创建一个 IntervalIndex 对象，并断言其左右端点数组不同
        result = IntervalIndex(np.array(index), copy=False)
        tm.assert_numpy_array_equal(
            index.left.values, result.left.values, check_same="copy"
        )
        tm.assert_numpy_array_equal(
            index.right.values, result.right.values, check_same="copy"
        )

    # 测试 delete() 方法
    def test_delete(self, closed):
        # 创建一个断点数组 breaks，并从中创建一个 IntervalIndex 对象
        breaks = np.arange(1, 11, dtype=np.int64)
        expected = IntervalIndex.from_breaks(breaks, closed=closed)
        # 删除指定位置的区间后，断言删除后的结果与期望值相同
        result = self.create_index(closed=closed).delete(0)
        tm.assert_index_equal(result, expected)

    # 参数化测试，测试不同数据情况下的 interval_range 方法
    @pytest.mark.parametrize(
        "data",
        [
            interval_range(0, periods=10, closed="neither"),
            interval_range(1.7, periods=8, freq=2.5, closed="both"),
            interval_range(Timestamp("20170101"), periods=12, closed="left"),
            interval_range(Timedelta("1 day"), periods=6, closed="right"),
        ],
    )
    # 定义测试方法 `test_insert`，用于测试数据插入操作
    def test_insert(self, data):
        # 从数据中取出第一个元素作为插入项
        item = data[0]
        # 将插入项构建成包含一个区间的 IntervalIndex 对象
        idx_item = IntervalIndex([item])

        # 插入操作 - 在列表头部插入元素
        expected = idx_item.append(data)
        result = data.insert(0, item)
        tm.assert_index_equal(result, expected)

        # 插入操作 - 在列表尾部插入元素
        expected = data.append(idx_item)
        result = data.insert(len(data), item)
        tm.assert_index_equal(result, expected)

        # 插入操作 - 在列表中间位置插入元素
        expected = data[:3].append(idx_item).append(data[3:])
        result = data.insert(3, item)
        tm.assert_index_equal(result, expected)

        # 插入操作 - 插入类型无效
        res = data.insert(1, "foo")
        expected = data.astype(object).insert(1, "foo")
        tm.assert_index_equal(res, expected)

        # 异常处理 - 插入非法类型
        msg = "can only insert Interval objects and NA into an IntervalArray"
        with pytest.raises(TypeError, match=msg):
            data._data.insert(1, "foo")

        # 异常处理 - 插入区间不合法
        msg = "'value.closed' is 'left', expected 'right'."
        for closed in {"left", "right", "both", "neither"} - {item.closed}:
            msg = f"'value.closed' is '{closed}', expected '{item.closed}'."
            bad_item = Interval(item.left, item.right, closed=closed)
            res = data.insert(1, bad_item)
            expected = data.astype(object).insert(1, bad_item)
            tm.assert_index_equal(res, expected)
            with pytest.raises(ValueError, match=msg):
                data._data.insert(1, bad_item)

        # 测试缺失值插入情况
        # GH 18295 (test missing)
        na_idx = IntervalIndex([np.nan], closed=data.closed)
        for na in [np.nan, None, pd.NA]:
            expected = data[:1].append(na_idx).append(data[1:])
            result = data.insert(1, na)
            tm.assert_index_equal(result, expected)

        # 检查数据类型，如果左边界不是日期类型，应当强制转换
        if data.left.dtype.kind not in ["m", "M"]:
            expected = data.astype(object).insert(1, pd.NaT)

            msg = "can only insert Interval objects and NA into an IntervalArray"
            with pytest.raises(TypeError, match=msg):
                data._data.insert(1, pd.NaT)

        # 最终插入操作，将 pd.NaT 插入指定位置
        result = data.insert(1, pd.NaT)
        tm.assert_index_equal(result, expected)
    def test_is_unique_interval(self, closed):
        """
        Interval specific tests for is_unique in addition to base class tests
        """
        # unique overlapping - distinct endpoints
        # 创建一个 IntervalIndex 对象，包含两个区间：(0, 1) 和 (0.5, 1.5)，并指定闭合属性
        idx = IntervalIndex.from_tuples([(0, 1), (0.5, 1.5)], closed=closed)
        # 断言 is_unique 属性为 True
        assert idx.is_unique is True

        # unique overlapping - shared endpoints
        # 创建一个 IntervalIndex 对象，包含三个区间：(1, 2), (1, 3), (2, 3)，并指定闭合属性
        idx = IntervalIndex.from_tuples([(1, 2), (1, 3), (2, 3)], closed=closed)
        # 断言 is_unique 属性为 True
        assert idx.is_unique is True

        # unique nested
        # 创建一个 IntervalIndex 对象，包含两个嵌套的区间：(-1, 1) 和 (-2, 2)，并指定闭合属性
        idx = IntervalIndex.from_tuples([(-1, 1), (-2, 2)], closed=closed)
        # 断言 is_unique 属性为 True
        assert idx.is_unique is True

        # unique NaN
        # 创建一个 IntervalIndex 对象，包含一个 NaN 区间
        idx = IntervalIndex.from_tuples([(np.nan, np.nan)], closed=closed)
        # 断言 is_unique 属性为 True
        assert idx.is_unique is True

        # non-unique NaN
        # 创建一个 IntervalIndex 对象，包含两个相同的 NaN 区间
        idx = IntervalIndex.from_tuples(
            [(np.nan, np.nan), (np.nan, np.nan)], closed=closed
        )
        # 断言 is_unique 属性为 False
        assert idx.is_unique is False

    def test_is_monotonic_with_nans(self):
        # GH#41831
        # 创建一个包含两个 NaN 值的 IntervalIndex 对象
        index = IntervalIndex([np.nan, np.nan])

        # 断言各种单调性属性为 False
        assert not index.is_monotonic_increasing
        assert not index._is_strictly_monotonic_increasing
        assert not index.is_monotonic_increasing
        assert not index._is_strictly_monotonic_decreasing
        assert not index.is_monotonic_decreasing

    @pytest.mark.parametrize(
        "breaks",
        [
            date_range("20180101", periods=4),
            date_range("20180101", periods=4, tz="US/Eastern"),
            timedelta_range("0 days", periods=4),
        ],
        ids=lambda x: str(x.dtype),
    )
    def test_maybe_convert_i8(self, breaks):
        # GH 20636
        # 根据给定的 breaks 创建一个 IntervalIndex 对象
        index = IntervalIndex.from_breaks(breaks)

        # 测试对 intervalindex 对象进行 _maybe_convert_i8 操作
        result = index._maybe_convert_i8(index)
        expected = IntervalIndex.from_breaks(breaks.asi8)
        # 断言操作后的结果与预期相等
        tm.assert_index_equal(result, expected)

        # 测试对 interval 对象进行 _maybe_convert_i8 操作
        interval = Interval(breaks[0], breaks[1])
        result = index._maybe_convert_i8(interval)
        expected = Interval(breaks[0]._value, breaks[1]._value)
        # 断言操作后的结果与预期相等
        assert result == expected

        # 测试对 datetimelike index 进行 _maybe_convert_i8 操作
        result = index._maybe_convert_i8(breaks)
        expected = Index(breaks.asi8)
        # 断言操作后的结果与预期相等
        tm.assert_index_equal(result, expected)

        # 测试对 datetimelike scalar 进行 _maybe_convert_i8 操作
        result = index._maybe_convert_i8(breaks[0])
        expected = breaks[0]._value
        # 断言操作后的结果与预期相等
        assert result == expected

        # 测试对列表形式的 datetimelike scalars 进行 _maybe_convert_i8 操作
        result = index._maybe_convert_i8(list(breaks))
        expected = Index(breaks.asi8)
        # 断言操作后的结果与预期相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "breaks",
        [date_range("2018-01-01", periods=5), timedelta_range("0 days", periods=5)],
    )
    # 测试函数，用于验证 `_maybe_convert_i8` 方法在处理 `pd.NaT` 时的行为
    def test_maybe_convert_i8_nat(self, breaks):
        # GH 20636
        # 使用 IntervalIndex 类的方法根据断点数组创建索引对象
        index = IntervalIndex.from_breaks(breaks)

        # 创建一个需要转换的数组，其中包含了多个 pd.NaT 值，并转换为纳秒单位
        to_convert = breaks._constructor([pd.NaT] * 3).as_unit("ns")
        # 创建预期的索引对象，其中包含了多个 np.nan 值，数据类型为 np.float64
        expected = Index([np.nan] * 3, dtype=np.float64)
        # 调用 _maybe_convert_i8 方法进行转换
        result = index._maybe_convert_i8(to_convert)
        # 使用测试工具验证 result 与 expected 是否相等
        tm.assert_index_equal(result, expected)

        # 将需要转换的数组插入到现有数组中的第一个位置
        to_convert = to_convert.insert(0, breaks[0])
        # 更新预期结果，将第一个断点的值转换为 float，并插入到对应位置
        expected = expected.insert(0, float(breaks[0]._value))
        # 再次调用 _maybe_convert_i8 方法进行转换
        result = index._maybe_convert_i8(to_convert)
        # 使用测试工具验证 result 与 expected 是否相等
        tm.assert_index_equal(result, expected)

    # 参数化测试函数，用于验证 `_maybe_convert_i8` 方法在处理数值类型数据时的行为
    @pytest.mark.parametrize(
        "make_key",
        [lambda breaks: breaks, list],
        ids=["lambda", "list"],
    )
    def test_maybe_convert_i8_numeric(self, make_key, any_real_numpy_dtype):
        # GH 20636
        # 创建一个 NumPy 数组作为断点数组
        breaks = np.arange(5, dtype=any_real_numpy_dtype)
        # 使用 IntervalIndex 类的方法根据断点数组创建索引对象
        index = IntervalIndex.from_breaks(breaks)
        # 通过 make_key 函数生成一个索引对象的键
        key = make_key(breaks)

        # 调用 _maybe_convert_i8 方法进行转换
        result = index._maybe_convert_i8(key)
        # 确定断点数组的数据类型的种类（整数、无符号整数、浮点数），选择相应的预期数据类型
        kind = breaks.dtype.kind
        expected_dtype = {"i": np.int64, "u": np.uint64, "f": np.float64}[kind]
        # 创建预期的索引对象，数据类型为上述确定的预期数据类型
        expected = Index(key, dtype=expected_dtype)
        # 使用测试工具验证 result 与 expected 是否相等
        tm.assert_index_equal(result, expected)

    # 参数化测试函数，用于验证 `_maybe_convert_i8` 方法在处理数值类型数据时的行为（相同对象）
    @pytest.mark.parametrize(
        "make_key",
        [
            IntervalIndex.from_breaks,
            lambda breaks: Interval(breaks[0], breaks[1]),
            lambda breaks: breaks[0],
        ],
        ids=["IntervalIndex", "Interval", "scalar"],
    )
    def test_maybe_convert_i8_numeric_identical(self, make_key, any_real_numpy_dtype):
        # GH 20636
        # 创建一个 NumPy 数组作为断点数组
        breaks = np.arange(5, dtype=any_real_numpy_dtype)
        # 使用 IntervalIndex 类的方法根据断点数组创建索引对象
        index = IntervalIndex.from_breaks(breaks)
        # 通过 make_key 函数生成一个索引对象的键
        key = make_key(breaks)

        # 调用 _maybe_convert_i8 方法进行转换
        # 确认如果 key 是 Interval 或 IntervalIndex，_maybe_convert_i8 方法不会改变 key
        result = index._maybe_convert_i8(key)
        assert result is key

    # 参数化测试函数，用于验证 `_maybe_convert_i8` 方法在处理错误输入时的行为
    @pytest.mark.parametrize(
        "breaks1, breaks2",
        permutations(
            [
                date_range("20180101", periods=4),
                date_range("20180101", periods=4, tz="US/Eastern"),
                timedelta_range("0 days", periods=4),
            ],
            2,
        ),
        ids=lambda x: str(x.dtype),
    )
    @pytest.mark.parametrize(
        "make_key",
        [
            IntervalIndex.from_breaks,
            lambda breaks: Interval(breaks[0], breaks[1]),
            lambda breaks: breaks,
            lambda breaks: breaks[0],
            list,
        ],
        ids=["IntervalIndex", "Interval", "Index", "scalar", "list"],
    )
    def test_maybe_convert_i8_errors(self, breaks1, breaks2, make_key):
        # GH 20636
        # 使用 IntervalIndex 类的方法根据断点数组 breaks1 创建索引对象
        index = IntervalIndex.from_breaks(breaks1)
        # 通过 make_key 函数生成一个可能引发错误的键
        key = make_key(breaks2)

        # 创建异常消息的预期字符串，反映断点数组数据类型的子类型和 key 的数据类型不匹配
        msg = (
            f"Cannot index an IntervalIndex of subtype {breaks1.dtype} with "
            f"values of dtype {breaks2.dtype}"
        )
        msg = re.escape(msg)
        # 使用 pytest 的断言，验证调用 _maybe_convert_i8 方法时是否抛出 ValueError 异常，并且异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            index._maybe_convert_i8(key)
    def test_contains_method(self):
        # 创建一个 IntervalIndex 对象，包含两个区间 [0, 1] 和 [1, 2]
        i = IntervalIndex.from_arrays([0, 1], [1, 2])

        # 期望的布尔数组，表示是否包含指定的值
        expected = np.array([False, False], dtype="bool")
        # 测试 contains 方法，检查是否包含值 0
        actual = i.contains(0)
        tm.assert_numpy_array_equal(actual, expected)
        # 再次测试 contains 方法，检查是否包含值 3
        actual = i.contains(3)
        tm.assert_numpy_array_equal(actual, expected)

        # 更新期望的布尔数组，表示是否包含指定的值
        expected = np.array([True, False], dtype="bool")
        # 测试 contains 方法，检查是否包含值 0.5
        actual = i.contains(0.5)
        tm.assert_numpy_array_equal(actual, expected)
        # 再次测试 contains 方法，检查是否包含值 1
        actual = i.contains(1)
        tm.assert_numpy_array_equal(actual, expected)

        # 对于 "interval in interval" 情况，contains 方法暂未实现
        with pytest.raises(
            NotImplementedError, match="contains not implemented for two"
        ):
            i.contains(Interval(0, 1))

    def test_dropna(self, closed):
        # 创建一个期望的 IntervalIndex 对象，包含两个区间 [0.0, 1.0] 和 [1.0, 2.0]
        expected = IntervalIndex.from_tuples([(0.0, 1.0), (1.0, 2.0)], closed=closed)

        # 创建一个 IntervalIndex 对象，包含三个区间 [0, 1], [1, 2], [NaN]
        ii = IntervalIndex.from_tuples([(0, 1), (1, 2), np.nan], closed=closed)
        # 测试 dropna 方法，移除 NaN 区间后的结果
        result = ii.dropna()
        tm.assert_index_equal(result, expected)

        # 创建一个 IntervalIndex 对象，包含三个区间 [0, 1], [1, 2], [NaN]
        ii = IntervalIndex.from_arrays([0, 1, np.nan], [1, 2, np.nan], closed=closed)
        # 测试 dropna 方法，移除 NaN 区间后的结果
        result = ii.dropna()
        tm.assert_index_equal(result, expected)

    def test_non_contiguous(self, closed):
        # 创建一个 IntervalIndex 对象，包含两个不连续的区间 [0, 1] 和 [2, 3]
        index = IntervalIndex.from_tuples([(0, 1), (2, 3)], closed=closed)
        # 目标列表
        target = [0.5, 1.5, 2.5]
        # 测试 get_indexer 方法，返回目标列表中每个元素的索引
        actual = index.get_indexer(target)
        # 期望的索引数组
        expected = np.array([0, -1, 1], dtype="intp")
        tm.assert_numpy_array_equal(actual, expected)

        # 断言 1.5 是否不在 IntervalIndex 对象中
        assert 1.5 not in index

    def test_isin(self, closed):
        # 创建一个 IntervalIndex 对象
        index = self.create_index(closed=closed)

        # 期望的布尔数组，表示是否在指定的索引中
        expected = np.array([True] + [False] * (len(index) - 1))
        # 测试 isin 方法，检查是否在 index[:1] 中
        result = index.isin(index[:1])
        tm.assert_numpy_array_equal(result, expected)

        # 测试 isin 方法，检查是否在 [index[0]] 中
        result = index.isin([index[0]])
        tm.assert_numpy_array_equal(result, expected)

        # 创建另一个 IntervalIndex 对象
        other = IntervalIndex.from_breaks(np.arange(-2, 10), closed=closed)
        # 期望的布尔数组，表示是否在 other 中
        expected = np.array([True] * (len(index) - 1) + [False])
        # 测试 isin 方法，检查是否在 other 中
        result = index.isin(other)
        tm.assert_numpy_array_equal(result, expected)

        # 测试 isin 方法，检查是否在 other 的列表表示中
        result = index.isin(other.tolist())
        tm.assert_numpy_array_equal(result, expected)

        # 对于不同的 closed 参数，进行循环测试
        for other_closed in ["right", "left", "both", "neither"]:
            other = self.create_index(closed=other_closed)
            # 期望的布尔数组，表示是否在 other 中
            expected = np.repeat(closed == other_closed, len(index))
            # 测试 isin 方法，检查是否在 other 中
            result = index.isin(other)
            tm.assert_numpy_array_equal(result, expected)

            # 测试 isin 方法，检查是否在 other 的列表表示中
            result = index.isin(other.tolist())
            tm.assert_numpy_array_equal(result, expected)
    # 定义一个测试方法，用于比较 Interval 对象和索引的关系
    def test_comparison(self):
        # 计算 Interval(0, 1) 是否小于当前索引 self.index，并进行断言比较
        actual = Interval(0, 1) < self.index
        expected = np.array([False, True])
        tm.assert_numpy_array_equal(actual, expected)

        # 计算 Interval(0.5, 1.5) 是否小于当前索引 self.index，并进行断言比较
        actual = Interval(0.5, 1.5) < self.index
        tm.assert_numpy_array_equal(actual, expected)

        # 计算当前索引 self.index 是否大于 Interval(0.5, 1.5)，并进行断言比较
        actual = self.index > Interval(0.5, 1.5)
        tm.assert_numpy_array_equal(actual, expected)

        # 计算当前索引 self.index 是否等于自身，并进行断言比较
        actual = self.index == self.index
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(actual, expected)

        # 计算当前索引 self.index 是否小于等于自身，并进行断言比较
        actual = self.index <= self.index
        tm.assert_numpy_array_equal(actual, expected)

        # 计算当前索引 self.index 是否大于等于自身，并进行断言比较
        actual = self.index >= self.index
        tm.assert_numpy_array_equal(actual, expected)

        # 计算当前索引 self.index 是否小于自身，并进行断言比较
        actual = self.index < self.index
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(actual, expected)

        # 计算当前索引 self.index 是否大于自身，并进行断言比较
        actual = self.index > self.index
        tm.assert_numpy_array_equal(actual, expected)

        # 计算当前索引 self.index 是否等于 IntervalIndex.from_breaks([0, 1, 2], "left")，并进行断言比较
        actual = self.index == IntervalIndex.from_breaks([0, 1, 2], "left")
        tm.assert_numpy_array_equal(actual, expected)

        # 计算当前索引 self.index 是否等于其值 self.index.values，并进行断言比较
        actual = self.index == self.index.values
        tm.assert_numpy_array_equal(actual, np.array([True, True]))

        # 计算当前索引的值 self.index.values 是否等于当前索引 self.index，并进行断言比较
        actual = self.index.values == self.index
        tm.assert_numpy_array_equal(actual, np.array([True, True]))

        # 计算当前索引 self.index 是否小于等于其值 self.index.values，并进行断言比较
        actual = self.index <= self.index.values
        tm.assert_numpy_array_equal(actual, np.array([True, True]))

        # 计算当前索引 self.index 是否不等于其值 self.index.values，并进行断言比较
        actual = self.index != self.index.values
        tm.assert_numpy_array_equal(actual, np.array([False, False]))

        # 计算当前索引 self.index 是否大于其值 self.index.values，并进行断言比较
        actual = self.index > self.index.values
        tm.assert_numpy_array_equal(actual, np.array([False, False]))

        # 计算当前索引的值 self.index.values 是否大于当前索引 self.index，并进行断言比较
        actual = self.index.values > self.index
        tm.assert_numpy_array_equal(actual, np.array([False, False]))

        # 对于不支持的比较操作，验证是否会抛出预期的 TypeError 异常，并进行匹配验证
        msg = "|".join(
            [
                "not supported between instances of 'int' and '.*.Interval'",
                r"Invalid comparison between dtype=interval\[int64, right\] and ",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            self.index > 0
        with pytest.raises(TypeError, match=msg):
            self.index <= 0
        with pytest.raises(TypeError, match=msg):
            self.index > np.arange(2)

        # 对于长度不匹配的比较操作，验证是否会抛出预期的 ValueError 异常，并进行匹配验证
        msg = "Lengths must match to compare"
        with pytest.raises(ValueError, match=msg):
            self.index > np.arange(3)
    # 测试缺失值处理函数，验证索引对象的创建和比较
    def test_missing_values(self, closed):
        # 创建包含 NaN 和 Interval 对象的索引
        idx = Index(
            [np.nan, Interval(0, 1, closed=closed), Interval(1, 2, closed=closed)]
        )
        # 使用 from_arrays 方法创建另一个索引对象 idx2，并断言两者相等
        idx2 = IntervalIndex.from_arrays([np.nan, 0, 1], [np.nan, 1, 2], closed=closed)
        assert idx.equals(idx2)

        # 定义错误消息，用于异常断言
        msg = (
            "missing values must be missing in the same location both left "
            "and right sides"
        )
        # 使用 pytest.raises 检查 ValueError 异常是否抛出，并验证错误消息
        with pytest.raises(ValueError, match=msg):
            IntervalIndex.from_arrays(
                [np.nan, 0, 1], np.array([0, 1, 2]), closed=closed
            )

        # 断言 idx 中的缺失值位置是否正确标记
        tm.assert_numpy_array_equal(isna(idx), np.array([True, False, False]))

    # 测试索引排序功能
    def test_sort_values(self, closed):
        # 创建具有指定关闭属性的索引对象
        index = self.create_index(closed=closed)

        # 测试默认排序结果是否与原索引相同
        result = index.sort_values()
        tm.assert_index_equal(result, index)

        # 测试降序排序结果是否正确
        result = index.sort_values(ascending=False)
        tm.assert_index_equal(result, index[::-1])

        # 包含 NaN 值的情况下进行排序测试
        index = IntervalIndex([Interval(1, 2), np.nan, Interval(0, 1)])

        # 测试升序排序结果是否符合预期
        result = index.sort_values()
        expected = IntervalIndex([Interval(0, 1), Interval(1, 2), np.nan])
        tm.assert_index_equal(result, expected)

        # 测试降序排序结果及 NaN 位置设置是否符合预期
        result = index.sort_values(ascending=False, na_position="first")
        expected = IntervalIndex([np.nan, Interval(1, 2), Interval(0, 1)])
        tm.assert_index_equal(result, expected)

    # 使用 pytest 的参数化功能来测试不同的时区参数
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    # 测试日期时间相关功能，使用给定的时区（tz）
    def test_datetime(self, tz):
        # 创建起始时间戳，指定时区
        start = Timestamp("2000-01-01", tz=tz)
        # 生成一个包含10个日期的日期范围
        dates = date_range(start=start, periods=10)
        # 从日期范围创建一个间隔索引
        index = IntervalIndex.from_breaks(dates)

        # 测试中间值（mid）
        start = Timestamp("2000-01-01T12:00", tz=tz)
        # 生成预期的日期范围，从指定的开始时间开始，生成9个日期
        expected = date_range(start=start, periods=9)
        # 检查索引的中间值是否与预期相等
        tm.assert_index_equal(index.mid, expected)

        # __contains__ 方法不检查单个时间点
        assert Timestamp("2000-01-01", tz=tz) not in index
        assert Timestamp("2000-01-01T12", tz=tz) not in index
        assert Timestamp("2000-01-02", tz=tz) not in index

        # 创建一个真区间和假区间用于测试
        iv_true = Interval(
            Timestamp("2000-01-02", tz=tz), Timestamp("2000-01-03", tz=tz)
        )
        iv_false = Interval(
            Timestamp("1999-12-31", tz=tz), Timestamp("2000-01-01", tz=tz)
        )
        # 检查真区间是否在索引中，假区间是否不在索引中
        assert iv_true in index
        assert iv_false not in index

        # .contains 方法检查单个时间点
        assert not index.contains(Timestamp("2000-01-01", tz=tz)).any()
        assert index.contains(Timestamp("2000-01-01T12", tz=tz)).any()
        assert index.contains(Timestamp("2000-01-02", tz=tz)).any()

        # 测试 get_indexer 方法
        start = Timestamp("1999-12-31T12:00", tz=tz)
        # 生成目标日期范围，从指定的开始时间开始，生成7个日期，频率为12小时
        target = date_range(start=start, periods=7, freq="12h")
        # 获取目标日期范围在索引中的索引值
        actual = index.get_indexer(target)
        # 预期的索引值数组
        expected = np.array([-1, -1, 0, 0, 1, 1, 2], dtype="intp")
        # 检查实际输出是否与预期相等
        tm.assert_numpy_array_equal(actual, expected)

        start = Timestamp("2000-01-08T18:00", tz=tz)
        # 生成目标日期范围，从指定的开始时间开始，生成7个日期，频率为6小时
        target = date_range(start=start, periods=7, freq="6h")
        # 获取目标日期范围在索引中的索引值
        actual = index.get_indexer(target)
        # 预期的索引值数组
        expected = np.array([7, 7, 8, 8, 8, 8, -1], dtype="intp")
        # 检查实际输出是否与预期相等
        tm.assert_numpy_array_equal(actual, expected)

    # 测试索引追加功能，使用给定的闭合方式（closed）
    def test_append(self, closed):
        # 创建两个间隔索引
        index1 = IntervalIndex.from_arrays([0, 1], [1, 2], closed=closed)
        index2 = IntervalIndex.from_arrays([1, 2], [2, 3], closed=closed)

        # 测试追加一个索引
        result = index1.append(index2)
        # 期望的间隔索引，包含追加后的所有间隔
        expected = IntervalIndex.from_arrays([0, 1, 1, 2], [1, 2, 2, 3], closed=closed)
        # 检查追加后的结果是否与期望相等
        tm.assert_index_equal(result, expected)

        # 测试追加一个索引列表
        result = index1.append([index1, index2])
        # 期望的间隔索引，包含追加后的所有间隔
        expected = IntervalIndex.from_arrays(
            [0, 1, 0, 1, 1, 2], [1, 2, 1, 2, 2, 3], closed=closed
        )
        # 检查追加后的结果是否与期望相等
        tm.assert_index_equal(result, expected)

        # 测试不同闭合方式的追加操作
        for other_closed in {"left", "right", "both", "neither"} - {closed}:
            index_other_closed = IntervalIndex.from_arrays(
                [0, 1], [1, 2], closed=other_closed
            )
            # 追加另一个闭合方式的索引
            result = index1.append(index_other_closed)
            # 期望的结果为对象类型的索引追加
            expected = index1.astype(object).append(index_other_closed.astype(object))
            # 检查追加后的结果是否与期望相等
            tm.assert_index_equal(result, expected)
    # 定义一个测试方法，用于验证 IntervalIndex 对象的非重叠单调性
    def test_is_non_overlapping_monotonic(self, closed):
        # 应始终为 True
        tpls = [(0, 1), (2, 3), (4, 5), (6, 7)]
        # 使用给定的元组列表创建 IntervalIndex 对象，指定是否包含端点
        idx = IntervalIndex.from_tuples(tpls, closed=closed)
        # 断言 IntervalIndex 对象的非重叠单调性为 True
        assert idx.is_non_overlapping_monotonic is True

        # 使用元组列表的反向列表创建 IntervalIndex 对象，测试是否仍为 True
        idx = IntervalIndex.from_tuples(tpls[::-1], closed=closed)
        assert idx.is_non_overlapping_monotonic is True

        # 应始终为 False
        tpls = [(0, 2), (1, 3), (4, 5), (6, 7)]
        # 使用给定的元组列表创建 IntervalIndex 对象，指定是否包含端点
        idx = IntervalIndex.from_tuples(tpls, closed=closed)
        # 断言 IntervalIndex 对象的非重叠单调性为 False
        assert idx.is_non_overlapping_monotonic is False

        # 使用元组列表的反向列表创建 IntervalIndex 对象，测试是否仍为 False
        idx = IntervalIndex.from_tuples(tpls[::-1], closed=closed)
        assert idx.is_non_overlapping_monotonic is False

        # 应始终为 False
        tpls = [(0, 1), (2, 3), (6, 7), (4, 5)]
        # 使用给定的元组列表创建 IntervalIndex 对象，指定是否包含端点
        idx = IntervalIndex.from_tuples(tpls, closed=closed)
        # 断言 IntervalIndex 对象的非重叠单调性为 False
        assert idx.is_non_overlapping_monotonic is False

        # 使用元组列表的反向列表创建 IntervalIndex 对象，测试是否仍为 False
        idx = IntervalIndex.from_tuples(tpls[::-1], closed=closed)
        assert idx.is_non_overlapping_monotonic is False

        # 对于 closed='both' 应为 False，否则为 True (GH16560)
        if closed == "both":
            # 使用断点范围创建 IntervalIndex 对象，指定是否包含端点
            idx = IntervalIndex.from_breaks(range(4), closed=closed)
            # 断言 IntervalIndex 对象的非重叠单调性为 False
            assert idx.is_non_overlapping_monotonic is False
        else:
            # 使用断点范围创建 IntervalIndex 对象，指定是否包含端点
            idx = IntervalIndex.from_breaks(range(4), closed=closed)
            # 断言 IntervalIndex 对象的非重叠单调性为 True
            assert idx.is_non_overlapping_monotonic is True
    def test_is_overlapping(self, start, shift, na_value, closed):
        # GH 23309
        # see test_interval_tree.py for extensive tests; interface tests here

        # non-overlapping
        tuples = [(start + n * shift, start + (n + 1) * shift) for n in (0, 2, 4)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        assert index.is_overlapping is False

        # non-overlapping with NA
        tuples = [(na_value, na_value)] + tuples + [(na_value, na_value)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        assert index.is_overlapping is False

        # overlapping
        tuples = [(start + n * shift, start + (n + 2) * shift) for n in range(3)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        assert index.is_overlapping is True

        # overlapping with NA
        tuples = [(na_value, na_value)] + tuples + [(na_value, na_value)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        assert index.is_overlapping is True

        # common endpoints
        tuples = [(start + n * shift, start + (n + 1) * shift) for n in range(3)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        result = index.is_overlapping
        expected = closed == "both"
        assert result is expected

        # common endpoints with NA
        tuples = [(na_value, na_value)] + tuples + [(na_value, na_value)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        result = index.is_overlapping
        assert result is expected

        # intervals with duplicate left values
        a = [10, 15, 20, 25, 30, 35, 40, 45, 45, 50, 55, 60, 65, 70, 75, 80, 85]
        b = [15, 20, 25, 30, 35, 40, 45, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
        index = IntervalIndex.from_arrays(a, b, closed="right")
        result = index.is_overlapping
        assert result is False

    @pytest.mark.parametrize(
        "tuples",
        [
            zip(range(10), range(1, 11)),
            zip(
                date_range("20170101", periods=10),
                date_range("20170101", periods=10),
            ),
            zip(
                timedelta_range("0 days", periods=10),
                timedelta_range("1 day", periods=10),
            ),
        ],
    )
    def test_to_tuples(self, tuples):
        # GH 18756
        # Convert tuples to list and create IntervalIndex from them
        tuples = list(tuples)
        idx = IntervalIndex.from_tuples(tuples)
        # Retrieve tuples representation from IntervalIndex
        result = idx.to_tuples()
        # Convert tuples to Index and verify equality with expected Index
        expected = Index(com.asarray_tuplesafe(tuples))
        tm.assert_index_equal(result, expected)
    @pytest.mark.parametrize(
        "tuples",
        [
            list(zip(range(10), range(1, 11))) + [np.nan],
            list(
                zip(
                    date_range("20170101", periods=10),
                    date_range("20170101", periods=10),
                )
            )
            + [np.nan],
            list(
                zip(
                    timedelta_range("0 days", periods=10),
                    timedelta_range("1 day", periods=10),
                )
            )
            + [np.nan],
        ],
    )
    @pytest.mark.parametrize("na_tuple", [True, False])
    def test_to_tuples_na(self, tuples, na_tuple):
        # GH 18756
        # 根据给定的元组列表创建区间索引对象
        idx = IntervalIndex.from_tuples(tuples)
        # 调用 to_tuples 方法，生成元组表示的索引结果
        result = idx.to_tuples(na_tuple=na_tuple)

        # 检查非缺失部分的结果
        # 期望的非缺失部分索引
        expected_notna = Index(com.asarray_tuplesafe(tuples[:-1]))
        result_notna = result[:-1]
        tm.assert_index_equal(result_notna, expected_notna)

        # 检查缺失部分的结果
        result_na = result[-1]
        if na_tuple:
            # 如果需要处理缺失值，确保结果是一个长度为 2 的元组，并且每个元素都是缺失值
            assert isinstance(result_na, tuple)
            assert len(result_na) == 2
            assert all(isna(x) for x in result_na)
        else:
            # 否则，确保整个结果是缺失的
            assert isna(result_na)

    def test_nbytes(self):
        # GH 19209
        # 创建两个 numpy 数组
        left = np.arange(0, 4, dtype="i8")
        right = np.arange(1, 5, dtype="i8")

        # 计算 IntervalIndex 对象的字节大小
        result = IntervalIndex.from_arrays(left, right).nbytes
        expected = 64  # 4 * 8 * 2
        assert result == expected

    @pytest.mark.parametrize("name", [None, "foo"])
    def test_set_closed(self, name, closed, other_closed):
        # GH 21670
        # 创建一个 interval_range 对象
        index = interval_range(0, 5, closed=closed, name=name)
        # 调用 set_closed 方法，设置新的 closed 参数
        result = index.set_closed(other_closed)
        # 创建期望的 interval_range 对象
        expected = interval_range(0, 5, closed=other_closed, name=name)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("bad_closed", ["foo", 10, "LEFT", True, False])
    def test_set_closed_errors(self, bad_closed):
        # GH 21670
        # 创建一个 interval_range 对象
        index = interval_range(0, 5)
        # 检查 set_closed 方法对于无效参数是否会引发 ValueError 异常
        msg = f"invalid option for 'closed': {bad_closed}"
        with pytest.raises(ValueError, match=msg):
            index.set_closed(bad_closed)

    def test_is_all_dates(self):
        # GH 23576
        # 创建一个表示 2017 年整年的 Interval 对象
        year_2017 = Interval(
            Timestamp("2017-01-01 00:00:00"), Timestamp("2018-01-01 00:00:00")
        )
        # 创建一个包含单个 Interval 对象的 IntervalIndex
        year_2017_index = IntervalIndex([year_2017])
        # 检查 _is_all_dates 方法是否正确标识了所有日期
        assert not year_2017_index._is_all_dates
# GH#27571 dir(interval_index) 应该不会引发异常
def test_dir():
    # 创建一个 IntervalIndex 对象，包含两个区间 [0, 1) 和 [1, 2)
    index = IntervalIndex.from_arrays([0, 1], [1, 2])
    # 获取 index 对象的所有属性和方法名，并存储在 result 变量中
    result = dir(index)
    # 断言结果中不应该包含字符串 "str"
    assert "str" not in result


# https://github.com/pandas-dev/pandas/issues/32762
def test_searchsorted_different_argument_classes(listlike_box):
    # 创建一个 IntervalIndex 对象，包含两个区间 [0, 1) 和 [1, 2)
    values = IntervalIndex([Interval(0, 1), Interval(1, 2)])
    # 使用 listlike_box 函数包装 values 对象，并对其进行 searchsorted 操作
    result = values.searchsorted(listlike_box(values))
    # 创建一个期望的 numpy 数组，其值为 [0, 1]，数据类型与 result 相同
    expected = np.array([0, 1], dtype=result.dtype)
    # 断言 result 和 expected 数组内容相等
    tm.assert_numpy_array_equal(result, expected)

    # 使用 values 对象的 _data 属性进行 searchsorted 操作
    result = values._data.searchsorted(listlike_box(values))
    # 断言 result 和 expected 数组内容相等
    tm.assert_numpy_array_equal(result, expected)


# 参数化测试用例，分别传入 [1, 2], ["a", "b"], 和带时区的 Timestamp 对象数组
@pytest.mark.parametrize(
    "arg", [[1, 2], ["a", "b"], [Timestamp("2020-01-01", tz="Europe/London")] * 2]
)
def test_searchsorted_invalid_argument(arg):
    # 创建一个 IntervalIndex 对象，包含两个区间 [0, 1) 和 [1, 2)
    values = IntervalIndex([Interval(0, 1), Interval(1, 2)])
    # 预期的错误信息部分字符串
    msg = "'<' not supported between instances of 'pandas._libs.interval.Interval' and "
    # 使用 pytest 的上下文管理器检查是否会抛出 TypeError，并匹配错误信息部分字符串
    with pytest.raises(TypeError, match=msg):
        # 对 values 对象进行 searchsorted 操作，传入参数 arg
        values.searchsorted(arg)
```