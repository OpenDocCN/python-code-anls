# `D:\src\scipysrc\pandas\pandas\tests\extension\base\methods.py`

```
    # 导入模块 inspect 用于检查对象的签名信息
    import inspect
    # 导入 operator 模块提供的运算符函数
    import operator

    # 导入 numpy 库，通常用 np 作为别名
    import numpy as np
    # 导入 pytest 库，用于编写和运行测试
    import pytest

    # 从 pandas._typing 模块导入 Dtype 类型
    from pandas._typing import Dtype

    # 从 pandas.core.dtypes.common 模块导入 is_bool_dtype 函数
    from pandas.core.dtypes.common import is_bool_dtype
    # 从 pandas.core.dtypes.dtypes 模块导入 NumpyEADtype 类型
    from pandas.core.dtypes.dtypes import NumpyEADtype
    # 从 pandas.core.dtypes.missing 模块导入 na_value_for_dtype 函数
    from pandas.core.dtypes.missing import na_value_for_dtype

    # 导入 pandas 库，通常用 pd 作为别名
    import pandas as pd
    # 导入 pandas._testing 模块，用于 pandas 的测试工具
    import pandas._testing as tm
    # 从 pandas.core.sorting 模块导入 nargsort 函数
    from pandas.core.sorting import nargsort


class BaseMethodsTests:
    """Various Series and DataFrame methods."""

    def test_hash_pandas_object(self, data):
        # _hash_pandas_object 应返回一个与数据长度相同的 uint64 数组
        # 导入 _default_hash_key 函数，用于生成哈希键
        from pandas.core.util.hashing import _default_hash_key

        # 调用 _hash_pandas_object 方法，对数据进行哈希化处理
        res = data._hash_pandas_object(
            encoding="utf-8", hash_key=_default_hash_key, categorize=False
        )
        # 断言返回结果的数据类型为 np.uint64
        assert res.dtype == np.uint64
        # 断言返回结果的形状与原始数据相同
        assert res.shape == data.shape

    def test_value_counts_default_dropna(self, data):
        # 确保默认情况下 dropna 参数的一致性
        if not hasattr(data, "value_counts"):
            pytest.skip(f"value_counts is not implemented for {type(data)}")
        # 获取 value_counts 方法的参数签名
        sig = inspect.signature(data.value_counts)
        # 获取 dropna 参数对象
        kwarg = sig.parameters["dropna"]
        # 断言 dropna 参数的默认值为 True
        assert kwarg.default is True

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna):
        # 对 all_data 进行切片，仅保留前 10 条记录
        all_data = all_data[:10]
        # 根据 dropna 参数的值选择不同的数据处理方式
        if dropna:
            other = all_data[~all_data.isna()]
        else:
            other = all_data

        # 调用 value_counts 方法，统计数据的值并按索引排序
        result = pd.Series(all_data).value_counts(dropna=dropna).sort_index()
        # 根据条件重新组织数据，以便与预期结果进行比较
        expected = pd.Series(other).value_counts(dropna=dropna).sort_index()

        # 使用测试工具方法 assert_series_equal 检查结果与预期结果是否相同
        tm.assert_series_equal(result, expected)

    def test_value_counts_with_normalize(self, data):
        # GH 33172
        # 对 data 进行切片，仅保留前 10 条记录，并获取唯一值
        data = data[:10].unique()
        # 获取非空值的 numpy 数组
        values = np.array(data[~data.isna()])
        # 创建 pandas Series 对象，指定数据类型
        ser = pd.Series(data, dtype=data.dtype)

        # 调用 value_counts 方法，对数据进行归一化处理并按索引排序
        result = ser.value_counts(normalize=True).sort_index()

        # 根据数据的类型不同设置预期结果
        if not isinstance(data, pd.Categorical):
            expected = pd.Series(
                [1 / len(values)] * len(values), index=result.index, name="proportion"
            )
        else:
            expected = pd.Series(0.0, index=result.index, name="proportion")
            expected[result > 0] = 1 / len(values)

        # 根据数据类型的存储方式，调整预期结果的数据类型
        if getattr(data.dtype, "storage", "") == "pyarrow" or isinstance(
            data.dtype, pd.ArrowDtype
        ):
            # TODO: 避免特殊情况处理
            expected = expected.astype("double[pyarrow]")
        elif getattr(data.dtype, "storage", "") == "pyarrow_numpy":
            # TODO: 避免特殊情况处理
            expected = expected.astype("float64")
        elif na_value_for_dtype(data.dtype) is pd.NA:
            # TODO(GH#44692): 避免特殊情况处理
            expected = expected.astype("Float64")

        # 使用测试工具方法 assert_series_equal 检查结果与预期结果是否相同
        tm.assert_series_equal(result, expected)
    # 定义一个测试方法，用于测试数据缺失情况下的 DataFrame 列计数功能
    def test_count(self, data_missing):
        # 创建一个 DataFrame，列名为"A"，数据来自参数 data_missing
        df = pd.DataFrame({"A": data_missing})
        # 对 DataFrame 进行列方向上的计数操作，生成结果
        result = df.count(axis="columns")
        # 预期结果为包含两个元素的 Series：[0, 1]
        expected = pd.Series([0, 1])
        # 使用测试框架中的方法来断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，用于测试数据缺失情况下的 Series 计数功能
    def test_series_count(self, data_missing):
        # 创建一个 Series，数据来自参数 data_missing
        ser = pd.Series(data_missing)
        # 对 Series 进行计数操作，生成结果
        result = ser.count()
        # 预期结果为整数 1
        expected = 1
        # 使用标准的 Python 断言来判断结果是否符合预期
        assert result == expected

    # 定义一个测试方法，测试简单 Series 对象的 apply 方法
    def test_apply_simple_series(self, data):
        # 对参数 data 构建的 Series 执行 apply 方法，传入 id 函数
        result = pd.Series(data).apply(id)
        # 断言结果是一个 pandas 的 Series 对象
        assert isinstance(result, pd.Series)

    # 用参数化测试标记标记一个测试方法，测试 Series 对象的 map 方法
    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data_missing, na_action):
        # 对 data_missing 所代表的 Series 对象应用 lambda 函数，并指定 na_action 参数
        result = data_missing.map(lambda x: x, na_action=na_action)
        # 预期结果为 data_missing 转换为 numpy 数组
        expected = data_missing.to_numpy()
        # 使用测试框架中的方法来断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试方法，测试 Series 对象的 argsort 方法
    def test_argsort(self, data_for_sorting):
        # 对 data_for_sorting 构建的 Series 执行 argsort 方法
        result = pd.Series(data_for_sorting).argsort()
        # 预期结果为 [2, 0, 1] 的 Series，数据类型为 np.intp
        expected = pd.Series(np.array([2, 0, 1], dtype=np.intp))
        # 使用测试框架中的方法来断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，测试处理有缺失值的数组的 argsort 方法
    def test_argsort_missing_array(self, data_missing_for_sorting):
        # 对 data_missing_for_sorting 执行 argsort 方法
        result = data_missing_for_sorting.argsort()
        # 预期结果为 [2, 0, 1] 的 numpy 数组，数据类型为 np.intp
        expected = np.array([2, 0, 1], dtype=np.intp)
        # 使用测试框架中的方法来断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试方法，测试处理有缺失值的 Series 对象的 argsort 方法
    def test_argsort_missing(self, data_missing_for_sorting):
        # 对 data_missing_for_sorting 构建的 Series 执行 argsort 方法
        result = pd.Series(data_missing_for_sorting).argsort()
        # 预期结果为 [2, 0, 1] 的 Series，数据类型为 np.intp
        expected = pd.Series(np.array([2, 0, 1], dtype=np.intp))
        # 使用测试框架中的方法来断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，测试处理排序数据和有缺失值数据的 argmin 和 argmax 方法
    def test_argmin_argmax(self, data_for_sorting, data_missing_for_sorting, na_value):
        # GH 24382
        # 检查 data_for_sorting 数组的数据类型是否为布尔值
        is_bool = data_for_sorting.dtype._is_boolean

        # 设置预期的 argmax 和 argmin 值
        exp_argmax = 1
        exp_argmax_repeated = 3
        if is_bool:
            # 如果是布尔值，根据数据的描述设置不同的预期值
            exp_argmax = 0
            exp_argmax_repeated = 1

        # 断言对 data_for_sorting 执行 argmax 和 argmin 方法的结果是否符合预期
        assert data_for_sorting.argmax() == exp_argmax
        assert data_for_sorting.argmin() == 2

        # 对包含重复值的 data_for_sorting 执行 argmax 和 argmin 方法，检查结果是否符合预期
        data = data_for_sorting.take([2, 0, 0, 1, 1, 2])
        assert data.argmax() == exp_argmax_repeated
        assert data.argmin() == 0

        # 对包含缺失值的 data_missing_for_sorting 执行 argmax 和 argmin 方法，检查结果是否符合预期
        assert data_missing_for_sorting.argmax() == 0
        assert data_missing_for_sorting.argmin() == 2

    # 用参数化测试标记标记一个测试方法，测试处理空数组的 argmin 和 argmax 方法
    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_empty_array(self, method, data):
        # GH 24382
        # 设置期望的错误消息
        err_msg = "attempt to get"
        # 使用 pytest 的上下文管理器来检查空数组执行指定方法时是否会抛出 ValueError 异常，并验证错误消息是否符合预期
        with pytest.raises(ValueError, match=err_msg):
            getattr(data[:0], method)()

    # 用参数化测试标记标记一个测试方法，测试处理空数组的 argmax 和 argmin 方法
    def test_argmin_argmax_all_na(self, method, data, na_value):
        # 处理所有数据缺失且 skipna=True 的情况，此时与空相同
        err_msg = "attempt to get"
        # 创建一个包含两个 na_value 的数据对象，数据类型与原数据相同
        data_na = type(data)._from_sequence([na_value, na_value], dtype=data.dtype)
        # 断言会抛出 ValueError 异常，异常信息包含 err_msg
        with pytest.raises(ValueError, match=err_msg):
            # 调用对象的指定方法（method）
            getattr(data_na, method)()

    @pytest.mark.parametrize(
        "op_name, skipna, expected",
        [
            ("idxmax", True, 0),  # 测试 idxmax 方法，skipna=True 时预期结果为 0
            ("idxmin", True, 2),  # 测试 idxmin 方法，skipna=True 时预期结果为 2
            ("argmax", True, 0),  # 测试 argmax 方法，skipna=True 时预期结果为 0
            ("argmin", True, 2),  # 测试 argmin 方法，skipna=True 时预期结果为 2
            ("idxmax", False, -1),  # 测试 idxmax 方法，skipna=False 时预期结果为 -1
            ("idxmin", False, -1),  # 测试 idxmin 方法，skipna=False 时预期结果为 -1
            ("argmax", False, -1),  # 测试 argmax 方法，skipna=False 时预期结果为 -1
            ("argmin", False, -1),  # 测试 argmin 方法，skipna=False 时预期结果为 -1
        ],
    )
    def test_argreduce_series(
        self, data_missing_for_sorting, op_name, skipna, expected
    ):
        # data_missing_for_sorting -> [B, NA, A]，其中 A < B，NA 为缺失值
        ser = pd.Series(data_missing_for_sorting)
        if expected == -1:
            # 断言会抛出 ValueError 异常，异常信息包含 "Encountered an NA value"
            with pytest.raises(ValueError, match="Encountered an NA value"):
                # 调用对象的指定方法（op_name），skipna 参数取决于测试参数
                getattr(ser, op_name)(skipna=skipna)
        else:
            # 调用对象的指定方法（op_name），skipna 参数取决于测试参数
            result = getattr(ser, op_name)(skipna=skipna)
            # 断言结果接近于预期值 expected
            tm.assert_almost_equal(result, expected)

    def test_argmax_argmin_no_skipna_notimplemented(self, data_missing_for_sorting):
        # GH#38733
        data = data_missing_for_sorting

        # 断言会抛出 ValueError 异常，异常信息包含 "Encountered an NA value"
        with pytest.raises(ValueError, match="Encountered an NA value"):
            # 调用数据对象的 argmin 方法，skipna=False
            data.argmin(skipna=False)

        # 断言会抛出 ValueError 异常，异常信息包含 "Encountered an NA value"
        with pytest.raises(ValueError, match="Encountered an NA value"):
            # 调用数据对象的 argmax 方法，skipna=False
            data.argmax(skipna=False)

    @pytest.mark.parametrize(
        "na_position, expected",
        [
            ("last", np.array([2, 0, 1], dtype=np.dtype("intp"))),  # 测试 na_position='last' 时的结果
            ("first", np.array([1, 2, 0], dtype=np.dtype("intp"))),  # 测试 na_position='first' 时的结果
        ],
    )
    def test_nargsort(self, data_missing_for_sorting, na_position, expected):
        # GH 25439
        # 调用自定义的 nargsort 函数，返回排序后的结果
        result = nargsort(data_missing_for_sorting, na_position=na_position)
        # 断言 numpy 数组的内容相等
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(self, data_for_sorting, ascending, sort_by_key):
        # 为 Series 创建一个排序后的结果
        ser = pd.Series(data_for_sorting)
        # 调用 Series 的 sort_values 方法进行排序
        result = ser.sort_values(ascending=ascending, key=sort_by_key)
        # 期望的排序结果
        expected = ser.iloc[[2, 0, 1]]
        if not ascending:
            # GH 35922. 期望是稳定排序
            if ser.nunique() == 2:
                expected = ser.iloc[[0, 1, 2]]
            else:
                expected = ser.iloc[[1, 0, 2]]

        # 断言 Series 相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_missing(
        self, data_missing_for_sorting, ascending, sort_by_key
    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_series(self, data_for_sorting, ascending):
        # 创建一个 Pandas Series，使用给定的数据
        ser = pd.Series(data_for_sorting)
        # 对 Series 进行排序，可以指定升序或降序以及排序键
        result = ser.sort_values(ascending=ascending, key=sort_by_key)
        # 根据升序或降序生成预期的排序结果
        if ascending:
            expected = ser.iloc[[2, 0, 1]]
        else:
            expected = ser.iloc[[0, 2, 1]]
        # 使用测试工具库（tm）来断言排序结果是否符合预期
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_frame(self, data_for_sorting, ascending):
        # 创建一个包含两列的 DataFrame，列"A"固定为[1, 2, 1]，列"B"使用给定的排序数据
        df = pd.DataFrame({"A": [1, 2, 1], "B": data_for_sorting})
        # 对 DataFrame 按列"A"和"B"进行排序
        result = df.sort_values(["A", "B"])
        # 创建预期的排序结果的 DataFrame
        expected = pd.DataFrame(
            {"A": [1, 1, 2], "B": data_for_sorting.take([2, 0, 1])}, index=[2, 0, 1]
        )
        # 使用测试工具库（tm）来断言排序结果是否符合预期
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("keep", ["first", "last", False])
    def test_duplicated(self, data, keep):
        # 创建一个包含重复数据的数组
        arr = data.take([0, 1, 0, 1])
        # 检查数组中的重复项，可以指定保留第一个、最后一个或不保留
        result = arr.duplicated(keep=keep)
        # 根据保留规则生成预期的重复检测结果数组
        if keep == "first":
            expected = np.array([False, False, True, True])
        elif keep == "last":
            expected = np.array([True, True, False, False])
        else:
            expected = np.array([True, True, True, True])
        # 使用测试工具库（tm）来断言重复检测结果是否符合预期
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("box", [pd.Series, lambda x: x])
    @pytest.mark.parametrize("method", [lambda x: x.unique(), pd.unique])
    def test_unique(self, data, box, method):
        # 创建一个包含重复数据的盒子（box）
        duplicated = box(data._from_sequence([data[0], data[0]], dtype=data.dtype))

        # 使用指定的方法来获取数据中的唯一值
        result = method(duplicated)

        # 断言结果数组长度为1，即所有数据都是重复的
        assert len(result) == 1
        # 断言结果的类型与原始数据的类型相同
        assert isinstance(result, type(data))
        # 断言结果中的唯一值等于原始数据中的第一个值
        assert result[0] == duplicated[0]

    def test_factorize(self, data_for_grouping):
        # 使用 Pandas 提供的 factorize 方法来获取数据的编码和唯一值
        codes, uniques = pd.factorize(data_for_grouping, use_na_sentinel=True)

        # 检查数据是否为布尔类型
        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # 如果数据是布尔类型，只有两个唯一值
            expected_codes = np.array([0, 0, -1, -1, 1, 1, 0, 0], dtype=np.intp)
            expected_uniques = data_for_grouping.take([0, 4])
        else:
            # 如果数据不是布尔类型，有三个唯一值
            expected_codes = np.array([0, 0, -1, -1, 1, 1, 0, 2], dtype=np.intp)
            expected_uniques = data_for_grouping.take([0, 4, 7])

        # 使用测试工具库（tm）来断言编码和唯一值是否符合预期
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_extension_array_equal(uniques, expected_uniques)

    def test_factorize_equivalence(self, data_for_grouping):
        # 使用 Pandas 提供的 factorize 方法和数据的 factorize 方法来获取编码和唯一值
        codes_1, uniques_1 = pd.factorize(data_for_grouping, use_na_sentinel=True)
        codes_2, uniques_2 = data_for_grouping.factorize(use_na_sentinel=True)

        # 断言两种方法得到的编码数组和唯一值数组是否相同
        tm.assert_numpy_array_equal(codes_1, codes_2)
        tm.assert_extension_array_equal(uniques_1, uniques_2)
        # 断言唯一值数组中的元素数量等于唯一值的数量
        assert len(uniques_1) == len(pd.unique(uniques_1))
        # 断言唯一值数组的数据类型与原始数据的数据类型相同
        assert uniques_1.dtype == data_for_grouping.dtype
    # 测试空数据的因子化处理
    def test_factorize_empty(self, data):
        # 使用 pandas 的 factorize 函数对空数据进行因子化处理，返回编码和唯一值
        codes, uniques = pd.factorize(data[:0])
        # 期望的编码为空的 numpy 数组
        expected_codes = np.array([], dtype=np.intp)
        # 期望的唯一值为空序列，类型与输入数据相同
        expected_uniques = type(data)._from_sequence([], dtype=data[:0].dtype)

        # 使用 pandas 测试工具比较编码和期望的编码
        tm.assert_numpy_array_equal(codes, expected_codes)
        # 使用 pandas 测试工具比较扩展数组和期望的唯一值
        tm.assert_extension_array_equal(uniques, expected_uniques)

    # 测试填充 DataFrame 中的缺失值，并限制填充次数
    def test_fillna_limit_frame(self, data_missing):
        # GH#58001
        # 创建一个包含缺失值的 DataFrame
        df = pd.DataFrame({"A": data_missing.take([0, 1, 0, 1])})
        # 创建期望的 DataFrame，对应的缺失值被填充
        expected = pd.DataFrame({"A": data_missing.take([1, 1, 0, 1])})
        # 使用 fillna 方法填充缺失值，并限制最多填充一次
        result = df.fillna(value=data_missing[1], limit=1)
        # 使用 pandas 测试工具比较结果和期望的 DataFrame
        tm.assert_frame_equal(result, expected)

    # 测试填充 Series 中的缺失值，并限制填充次数
    def test_fillna_limit_series(self, data_missing):
        # GH#58001
        # 创建一个包含缺失值的 Series
        ser = pd.Series(data_missing.take([0, 1, 0, 1]))
        # 创建期望的 Series，对应的缺失值被填充
        expected = pd.Series(data_missing.take([1, 1, 0, 1]))
        # 使用 fillna 方法填充缺失值，并限制最多填充一次
        result = ser.fillna(value=data_missing[1], limit=1)
        # 使用 pandas 测试工具比较结果和期望的 Series
        tm.assert_series_equal(result, expected)

    # 测试复制 DataFrame 中的缺失值处理
    def test_fillna_copy_frame(self, data_missing):
        # 创建一个包含缺失值的数组
        arr = data_missing.take([1, 1])
        # 创建包含该数组的 DataFrame
        df = pd.DataFrame({"A": arr})
        # 备份原始的 DataFrame
        df_orig = df.copy()

        # 获取需要填充的值
        filled_val = df.iloc[0, 0]
        # 使用 fillna 方法填充缺失值
        result = df.fillna(filled_val)

        # 修改结果的第一个元素，检查是否影响原始的 DataFrame
        result.iloc[0, 0] = filled_val

        # 使用 pandas 测试工具比较最终的 DataFrame 和原始备份的 DataFrame
        tm.assert_frame_equal(df, df_orig)

    # 测试复制 Series 中的缺失值处理
    def test_fillna_copy_series(self, data_missing):
        # 创建一个包含缺失值的数组
        arr = data_missing.take([1, 1])
        # 创建包含该数组的 Series，关闭复制选项
        ser = pd.Series(arr, copy=False)
        # 备份原始的 Series
        ser_orig = ser.copy()

        # 获取需要填充的值
        filled_val = ser[0]
        # 使用 fillna 方法填充缺失值
        result = ser.fillna(filled_val)
        # 修改结果的第一个元素，检查是否影响原始的 Series
        result.iloc[0] = filled_val

        # 使用 pandas 测试工具比较最终的 Series 和原始备份的 Series
        tm.assert_series_equal(ser, ser_orig)

    # 测试填充时长度不匹配的情况
    def test_fillna_length_mismatch(self, data_missing):
        # 期望的错误消息
        msg = "Length of 'value' does not match."
        # 使用 pytest 的上下文管理器检查是否引发了预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            data_missing.fillna(data_missing.take([1]))

    # 如果子类期望比较的数据类型为布尔值（如 Sparse[bool]、boolean、pyarrow[bool]），则可以重写此方法
    _combine_le_expected_dtype: Dtype = NumpyEADtype("bool")

    # 测试 Series 的 combine 方法，使用 <= 操作符进行比较
    def test_combine_le(self, data_repeated):
        # GH 20825
        # 测试 combine 方法在执行 <= 操作时的工作情况
        orig_data1, orig_data2 = data_repeated(2)
        # 创建两个 Series
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        # 使用 combine 方法比较两个 Series，使用 lambda 函数进行 <= 操作
        result = s1.combine(s2, lambda x1, x2: x1 <= x2)
        # 创建期望的 Series，结果为布尔数组，表示每个元素是否满足 <= 条件
        expected = pd.Series(
            pd.array(
                [a <= b for (a, b) in zip(list(orig_data1), list(orig_data2))],
                dtype=self._combine_le_expected_dtype,
            )
        )
        # 使用 pandas 测试工具比较结果和期望的 Series
        tm.assert_series_equal(result, expected)

        # 获取第一个元素的值
        val = s1.iloc[0]
        # 使用 combine 方法比较 Series 和单个值，使用 lambda 函数进行 <= 操作
        result = s1.combine(val, lambda x1, x2: x1 <= x2)
        # 创建期望的 Series，结果为布尔数组，表示每个元素是否满足 <= 条件
        expected = pd.Series(
            pd.array(
                [a <= val for a in list(orig_data1)],
                dtype=self._combine_le_expected_dtype,
            )
        )
        # 使用 pandas 测试工具比较结果和期望的 Series
        tm.assert_series_equal(result, expected)
    # 定义一个测试函数，用于测试 Series.combine_add 方法
    def test_combine_add(self, data_repeated):
        # GH 20825: GitHub issue reference
        
        # 从参数中获取两份重复数据
        orig_data1, orig_data2 = data_repeated(2)
        
        # 将数据转换为 Pandas Series 对象
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)

        # 检查我们的标量是否支持逐点操作。如果不支持，
        # 我们期望 Series.combine 也会抛出异常。
        try:
            with np.errstate(over="ignore"):
                # 创建预期的 Series 对象，使用原始数据1的序列和原始数据1与原始数据2相加的结果
                expected = pd.Series(
                    orig_data1._from_sequence(
                        [a + b for (a, b) in zip(list(orig_data1), list(orig_data2))]
                    )
                )
        except TypeError:
            # 如果我们的标量不支持逐点操作，则期望 Series.combine 也会抛出异常
            with pytest.raises(TypeError):
                s1.combine(s2, lambda x1, x2: x1 + x2)
            return

        # 执行 combine 操作，将结果与预期结果进行比较
        result = s1.combine(s2, lambda x1, x2: x1 + x2)
        tm.assert_series_equal(result, expected)

        # 从 s1 中获取第一个值
        val = s1.iloc[0]
        
        # 执行 combine 操作，将结果与预期结果进行比较
        result = s1.combine(val, lambda x1, x2: x1 + x2)
        expected = pd.Series(
            orig_data1._from_sequence([a + val for a in list(orig_data1)])
        )
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，用于测试 Series.combine_first 方法
    def test_combine_first(self, data):
        # https://github.com/pandas-dev/pandas/issues/24147: GitHub issue reference
        
        # 创建两个 Pandas Series 对象
        a = pd.Series(data[:3])
        b = pd.Series(data[2:5], index=[2, 3, 4])
        
        # 执行 combine_first 操作，将结果与预期结果进行比较
        result = a.combine_first(b)
        expected = pd.Series(data[:5])
        tm.assert_series_equal(result, expected)

    # 参数化测试，测试不同的输入条件
    @pytest.mark.parametrize("frame", [True, False])
    @pytest.mark.parametrize(
        "periods, indices",
        [(-2, [2, 3, 4, -1, -1]), (0, [0, 1, 2, 3, 4]), (2, [-1, -1, 0, 1, 2])],
    )
    def test_container_shift(self, data, frame, periods, indices):
        # https://github.com/pandas-dev/pandas/issues/22386: GitHub issue reference
        
        # 从数据中选择一个子集创建 Pandas Series 对象
        subset = data[:5]
        data = pd.Series(subset, name="A")
        
        # 创建预期结果的 Pandas Series 对象
        expected = pd.Series(subset.take(indices, allow_fill=True), name="A")

        if frame:
            # 如果 frame 为 True，则执行数据框转换和平移操作
            result = data.to_frame(name="A").assign(B=1).shift(periods)
            expected = pd.concat(
                [expected, pd.Series([1] * 5, name="B").shift(periods)], axis=1
            )
            compare = tm.assert_frame_equal
        else:
            # 如果 frame 为 False，则执行 Series 平移操作
            result = data.shift(periods)
            compare = tm.assert_series_equal

        # 比较结果与预期结果
        compare(result, expected)

    # 定义一个测试函数，测试在 periods=0 时的平移操作
    def test_shift_0_periods(self, data):
        # GH#33856: GitHub issue reference
        # 使用 periods=0 进行平移应返回一个副本，而不是同一个对象
        result = data.shift(0)
        
        # 断言第一个和第二个元素不相等，否则下面的断言就无效了
        assert data[0] != data[1]
        
        # 修改 data 的第一个元素为第二个元素的值
        data[0] = data[1]
        
        # 断言平移后的结果第一个和第二个元素不相等，即不是同一个对象或视图
        assert result[0] != result[1]

    # 参数化测试，测试不同 periods 的值
    @pytest.mark.parametrize("periods", [1, -2])
    # 测试不同的数据对象的差异操作
    def test_diff(self, data, periods):
        # 仅取数据的前五个元素
        data = data[:5]
        # 检查数据类型是否为布尔型
        if is_bool_dtype(data.dtype):
            # 如果是布尔型数据，选择异或操作
            op = operator.xor
        else:
            # 否则选择减法操作
            op = operator.sub
        try:
            # 尝试对数据执行指定的操作
            op(data, data)
        except Exception:
            # 如果操作失败，则跳过测试并输出提示信息
            pytest.skip(f"{type(data)} does not support diff")
        # 将数据转换为 Pandas 的 Series 对象
        s = pd.Series(data)
        # 计算 Series 对象的差分结果
        result = s.diff(periods)
        # 使用给定操作函数计算期望的结果
        expected = pd.Series(op(data, data.shift(periods)))
        # 检查计算结果是否与期望结果相等
        tm.assert_series_equal(result, expected)

        # 创建包含两列的 DataFrame 对象
        df = pd.DataFrame({"A": data, "B": [1.0] * 5})
        # 计算 DataFrame 对象的差分结果
        result = df.diff(periods)
        # 根据差分周期选择期望结果的填充值
        if periods == 1:
            b = [np.nan, 0, 0, 0, 0]
        else:
            b = [0, 0, 0, np.nan, np.nan]
        # 创建包含期望结果的 DataFrame 对象
        expected = pd.DataFrame({"A": expected, "B": b})
        # 检查 DataFrame 的计算结果是否与期望结果相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "periods, indices",
        [[-4, [-1, -1]], [-1, [1, -1]], [0, [0, 1]], [1, [-1, 0]], [4, [-1, -1]]],
    )
    # 测试非空数组的位移操作
    def test_shift_non_empty_array(self, data, periods, indices):
        # https://github.com/pandas-dev/pandas/issues/23911
        # 取数据的前两个元素作为子集
        subset = data[:2]
        # 执行位移操作并获取结果
        result = subset.shift(periods)
        # 根据指定的索引值获取期望结果
        expected = subset.take(indices, allow_fill=True)
        # 检查扩展数组的位移结果是否与期望结果相等
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize("periods", [-4, -1, 0, 1, 4])
    # 测试空数组的位移操作
    def test_shift_empty_array(self, data, periods):
        # https://github.com/pandas-dev/pandas/issues/23911
        # 创建一个空的数组
        empty = data[:0]
        # 执行位移操作并获取结果
        result = empty.shift(periods)
        # 期望结果即为空数组本身
        expected = empty
        # 检查扩展数组的位移结果是否与期望结果相等
        tm.assert_extension_array_equal(result, expected)

    # 测试位移操作是否进行了零拷贝
    def test_shift_zero_copies(self, data):
        # GH#31502
        # 执行零位移操作并检查结果是否为原始数据的引用
        result = data.shift(0)
        assert result is not data

        # 对空数组执行位移操作并检查结果是否为原始数据的引用
        result = data[:0].shift(2)
        assert result is not data

    # 测试位移操作的填充值功能
    def test_shift_fill_value(self, data):
        # 取数据的前四个元素作为数组
        arr = data[:4]
        # 设置填充值为数组的第一个元素，并执行正向位移操作
        fill_value = data[0]
        result = arr.shift(1, fill_value=fill_value)
        # 根据指定的索引值获取期望结果
        expected = data.take([0, 0, 1, 2])
        # 检查扩展数组的位移结果是否与期望结果相等
        tm.assert_extension_array_equal(result, expected)

        # 设置填充值为数组的第一个元素，并执行反向位移操作
        result = arr.shift(-2, fill_value=fill_value)
        # 根据指定的索引值获取期望结果
        expected = data.take([2, 3, 0, 0])
        # 检查扩展数组的位移结果是否与期望结果相等
        tm.assert_extension_array_equal(result, expected)

    # 测试不可哈希对象的哈希操作
    def test_not_hashable(self, data):
        # 我们一般是可变的，因此不可哈希
        # 检查是否会引发类型错误异常
        with pytest.raises(TypeError, match="unhashable type"):
            hash(data)

    @pytest.mark.parametrize("as_frame", [False, True])
    # 测试 Pandas 对象的哈希操作
    def test_hash_pandas_object_works(self, data, as_frame):
        # https://github.com/pandas-dev/pandas/issues/23066
        # 将数据转换为 Pandas 的 Series 对象
        data = pd.Series(data)
        # 根据参数决定是否将 Series 对象转换为 DataFrame 对象
        if as_frame:
            data = data.to_frame()
        # 计算 Pandas 对象的哈希值
        a = pd.util.hash_pandas_object(data)
        b = pd.util.hash_pandas_object(data)
        # 检查计算出的哈希值是否相等
        tm.assert_equal(a, b)
    # 定义一个测试方法，用于测试 searchsorted 方法在不同数据类型下的行为
    def test_searchsorted(self, data_for_sorting, as_series):
        # 如果数据类型是布尔类型，则调用专门的方法处理
        if data_for_sorting.dtype._is_boolean:
            return self._test_searchsorted_bool_dtypes(data_for_sorting, as_series)

        # 解包元组 data_for_sorting 中的元素
        b, c, a = data_for_sorting
        # 按指定顺序取出元素组成新的数组 arr，顺序为 [a, b, c]
        arr = data_for_sorting.take([2, 0, 1])  # to get [a, b, c]

        # 如果需要将 arr 转换为 Pandas Series，则执行转换
        if as_series:
            arr = pd.Series(arr)
        # 断言查找元素 a 在 arr 中的索引为 0
        assert arr.searchsorted(a) == 0
        # 断言查找元素 a 在 arr 中右侧的索引为 1
        assert arr.searchsorted(a, side="right") == 1

        # 断言查找元素 b 在 arr 中的索引为 1
        assert arr.searchsorted(b) == 1
        # 断言查找元素 b 在 arr 中右侧的索引为 2
        assert arr.searchsorted(b, side="right") == 2

        # 断言查找元素 c 在 arr 中的索引为 2
        assert arr.searchsorted(c) == 2
        # 断言查找元素 c 在 arr 中右侧的索引为 3
        assert arr.searchsorted(c, side="right") == 3

        # 对 arr 中取出的元素 [a, c] 进行 searchsorted 操作
        result = arr.searchsorted(arr.take([0, 2]))
        # 期望的结果数组
        expected = np.array([0, 2], dtype=np.intp)

        # 断言 result 和 expected 数组相等
        tm.assert_numpy_array_equal(result, expected)

        # sorter
        # 定义一个排序数组 sorter
        sorter = np.array([1, 2, 0])
        # 断言在指定排序下，查找元素 a 在 data_for_sorting 中的索引为 0
        assert data_for_sorting.searchsorted(a, sorter=sorter) == 0

    # 用于处理布尔类型数据的私有方法，从 test_searchsorted 方法中调用
    def _test_searchsorted_bool_dtypes(self, data_for_sorting, as_series):
        # 提取数据的 dtype
        dtype = data_for_sorting.dtype
        # 创建一个包含 True 和 False 的 Pandas 数组
        data_for_sorting = pd.array([True, False], dtype=dtype)
        # 解包元素
        b, a = data_for_sorting
        # 根据数据类型创建数组 arr
        arr = type(data_for_sorting)._from_sequence([a, b])

        # 如果需要将 arr 转换为 Pandas Series，则执行转换
        if as_series:
            arr = pd.Series(arr)
        # 断言查找元素 a 在 arr 中的索引为 0
        assert arr.searchsorted(a) == 0
        # 断言查找元素 a 在 arr 中右侧的索引为 1
        assert arr.searchsorted(a, side="right") == 1

        # 断言查找元素 b 在 arr 中的索引为 1
        assert arr.searchsorted(b) == 1
        # 断言查找元素 b 在 arr 中右侧的索引为 2
        assert arr.searchsorted(b, side="right") == 2

        # 对 arr 中取出的元素 [a, b] 进行 searchsorted 操作
        result = arr.searchsorted(arr.take([0, 1]))
        # 期望的结果数组
        expected = np.array([0, 1], dtype=np.intp)

        # 断言 result 和 expected 数组相等
        tm.assert_numpy_array_equal(result, expected)

        # sorter
        # 定义一个排序数组 sorter
        sorter = np.array([1, 0])
        # 断言在指定排序下，查找元素 a 在 data_for_sorting 中的索引为 0
        assert data_for_sorting.searchsorted(a, sorter=sorter) == 0
    # 测试 Series 对象的 where 方法，条件为第一个元素不等于第二个元素
    def test_where_series(self, data, na_value, as_frame):
        # 断言第一个元素不等于第二个元素
        assert data[0] != data[1]
        # 获取 data 对象的类型
        cls = type(data)
        # 将 data 的前两个元素分别赋值给 a 和 b
        a, b = data[:2]

        # 使用 data 中的元素创建原始 Series 对象 orig
        orig = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))
        # 复制 orig 生成新的 Series 对象 ser
        ser = orig.copy()
        # 创建布尔数组 cond，指示是否满足条件
        cond = np.array([True, True, False, False])

        # 如果 as_frame 为 True，则将 ser 转换为 DataFrame，并调整 cond 的形状
        if as_frame:
            ser = ser.to_frame(name="a")
            cond = cond.reshape(-1, 1)

        # 使用 where 方法根据条件 cond 过滤 ser，生成 result
        result = ser.where(cond)
        # 使用 data 中的元素创建预期的 Series 对象 expected
        expected = pd.Series(
            cls._from_sequence([a, a, na_value, na_value], dtype=data.dtype)
        )

        # 如果 as_frame 为 True，则将 expected 转换为 DataFrame
        if as_frame:
            expected = expected.to_frame(name="a")
        # 断言 result 和 expected 相等
        tm.assert_equal(result, expected)

        # 使用 mask 方法根据条件 ~cond 将 ser 中不满足条件的值替换为 na_value
        ser.mask(~cond, inplace=True)
        # 断言 ser 和 expected 相等
        tm.assert_equal(ser, expected)

        # array other
        # 复制 orig 生成新的 Series 对象 ser
        ser = orig.copy()
        # 如果 as_frame 为 True，则将 ser 转换为 DataFrame
        if as_frame:
            ser = ser.to_frame(name="a")
        # 创建新的布尔数组 cond
        cond = np.array([True, False, True, True])
        # 使用 data 中的元素创建新的 Series 对象 other
        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        # 如果 as_frame 为 True，则将 other 和 cond 转换为 DataFrame
        if as_frame:
            other = pd.DataFrame({"a": other})
            cond = pd.DataFrame({"a": cond})
        # 使用 where 方法根据条件 cond 过滤 ser，否则使用 other 替换不满足条件的值，生成 result
        result = ser.where(cond, other)
        # 使用 data 中的元素创建预期的 Series 对象 expected
        expected = pd.Series(cls._from_sequence([a, b, b, b], dtype=data.dtype))
        # 如果 as_frame 为 True，则将 expected 转换为 DataFrame
        if as_frame:
            expected = expected.to_frame(name="a")
        # 断言 result 和 expected 相等
        tm.assert_equal(result, expected)

        # 使用 mask 方法根据条件 ~cond 将 ser 中不满足条件的值替换为 other
        ser.mask(~cond, other, inplace=True)
        # 断言 ser 和 expected 相等
        tm.assert_equal(ser, expected)

    # 使用参数化测试重复操作
    @pytest.mark.parametrize("repeats", [0, 1, 2, [1, 2, 3]])
    def test_repeat(self, data, repeats, as_series, use_numpy):
        # 从 data 的前三个元素创建数组 arr
        arr = type(data)._from_sequence(data[:3], dtype=data.dtype)
        # 如果 as_series 为 True，则将 arr 转换为 Series 对象
        if as_series:
            arr = pd.Series(arr)

        # 使用 numpy 的 repeat 或 Series 对象的 repeat 方法对 arr 进行重复操作，生成 result
        result = np.repeat(arr, repeats) if use_numpy else arr.repeat(repeats)

        # 创建 repeats 的副本，如果 repeats 是整数，则将其重复三次
        repeats = [repeats] * 3 if isinstance(repeats, int) else repeats
        # 根据 repeats 中的值，生成预期的结果 expected
        expected = [x for x, n in zip(arr, repeats) for _ in range(n)]
        expected = type(data)._from_sequence(expected, dtype=data.dtype)
        # 如果 as_series 为 True，则将 expected 转换为 Series 对象，并设置其索引
        if as_series:
            expected = pd.Series(expected, index=arr.index.repeat(repeats))

        # 断言 result 和 expected 相等
        tm.assert_equal(result, expected)

    # 使用参数化测试测试重复操作时引发的异常情况
    @pytest.mark.parametrize(
        "repeats, kwargs, error, msg",
        [
            (2, {"axis": 1}, ValueError, "axis"),
            (-1, {}, ValueError, "negative"),
            ([1, 2], {}, ValueError, "shape"),
            (2, {"foo": "bar"}, TypeError, "'foo'"),
        ],
    )
    def test_repeat_raises(self, data, repeats, kwargs, error, msg, use_numpy):
        # 使用 pytest 的 raises 方法检查是否抛出指定的异常和消息
        with pytest.raises(error, match=msg):
            # 如果 use_numpy 为 True，则使用 numpy 的 repeat 方法，否则使用 data 的 repeat 方法
            if use_numpy:
                np.repeat(data, repeats, **kwargs)
            else:
                data.repeat(repeats, **kwargs)

    # 测试删除操作
    def test_delete(self, data):
        # 删除 data 的第一个元素，生成结果 result
        result = data.delete(0)
        # 生成预期的结果 expected，即删除 data 的第一个元素后的数组
        expected = data[1:]
        # 断言 result 和 expected 相等
        tm.assert_extension_array_equal(result, expected)

        # 删除 data 的指定索引的元素，生成结果 result
        result = data.delete([1, 3])
        # 生成预期的结果 expected，即删除指定索引后的数组拼接
        expected = data._concat_same_type([data[[0]], data[[2]], data[4:]])
        # 断言 result 和 expected 相等
        tm.assert_extension_array_equal(result, expected)
    # 定义一个测试方法，用于测试插入操作在给定数据上的行为
    def test_insert(self, data):
        # 在数据的开头插入元素
        result = data[1:].insert(0, data[0])
        # 使用测试工具验证插入后的结果与预期数据是否相等
        tm.assert_extension_array_equal(result, data)

        # 在数据的开头插入元素（另一种方式）
        result = data[1:].insert(-len(data[1:]), data[0])
        # 使用测试工具验证插入后的结果与预期数据是否相等
        tm.assert_extension_array_equal(result, data)

        # 在数据的中间位置插入元素
        result = data[:-1].insert(4, data[-1])

        # 创建一个索引数组，模拟在指定位置插入元素后的预期数据
        taker = np.arange(len(data))
        taker[5:] = taker[4:-1]
        taker[4] = len(data) - 1
        expected = data.take(taker)
        # 使用测试工具验证插入后的结果与预期数据是否相等
        tm.assert_extension_array_equal(result, expected)

    # 定义一个测试方法，用于测试在插入操作中处理无效情况的行为
    def test_insert_invalid(self, data, invalid_scalar):
        item = invalid_scalar

        # 检查在索引为0处插入无效元素时是否触发异常
        with pytest.raises((TypeError, ValueError)):
            data.insert(0, item)

        # 检查在索引为4处插入无效元素时是否触发异常
        with pytest.raises((TypeError, ValueError)):
            data.insert(4, item)

        # 检查在倒数第2个位置插入无效元素时是否触发异常
        with pytest.raises((TypeError, ValueError)):
            data.insert(len(data) - 1, item)

    # 定义一个测试方法，用于测试在插入操作中处理无效位置的行为
    def test_insert_invalid_loc(self, data):
        ub = len(data)

        # 检查在超出最大索引位置后插入元素是否触发 IndexError 异常
        with pytest.raises(IndexError):
            data.insert(ub + 1, data[0])

        # 检查在负数索引位置超出最小索引位置后插入元素是否触发 IndexError 异常
        with pytest.raises(IndexError):
            data.insert(-ub - 1, data[0])

        # 检查在插入浮点数索引位置时是否触发 TypeError 异常，以匹配 np.insert 的行为
        with pytest.raises(TypeError):
            data.insert(1.5, data[0])

    # 使用参数化测试装饰器，定义一个测试方法，用于测试数据对象之间相等性的行为
    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box):
        # 创建一个新的数据对象，内容为第一个元素重复的数组，用于测试相等性
        data2 = type(data)._from_sequence([data[0]] * len(data), dtype=data.dtype)
        # 创建一个包含指定缺失值的数据对象，用于测试相等性
        data_na = type(data)._from_sequence([na_value] * len(data), dtype=data.dtype)

        # 将原始数据对象装箱为指定类型，用于测试
        data = tm.box_expected(data, box, transpose=False)
        data2 = tm.box_expected(data2, box, transpose=False)
        data_na = tm.box_expected(data_na, box, transpose=False)

        # 使用显式断言来验证相等性检查的结果为 True
        assert data.equals(data) is True
        assert data.equals(data.copy()) is True

        # 检查与不同数据对象比较时相等性检查的结果为 False
        assert data.equals(data2) is False
        assert data.equals(data_na) is False

        # 检查与长度不同的数据对象比较时相等性检查的结果为 False
        assert data[:2].equals(data[:3]) is False

        # 检查空数据对象的相等性
        assert data[:0].equals(data[:0]) is True

        # 检查与其他类型比较时相等性检查的结果为 False
        assert data.equals(None) is False
        assert data[[0]].equals(data[0]) is False

    # 定义一个测试方法，用于测试即使数据相同但对象不同的情况下相等性的行为
    def test_equals_same_data_different_object(self, data):
        # 验证即使数据内容相同，但对象不同的情况下相等性检查的结果为 True
        assert pd.Series(data).equals(pd.Series(data))
```