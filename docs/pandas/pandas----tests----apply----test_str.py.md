# `D:\src\scipysrc\pandas\pandas\tests\apply\test_str.py`

```
    pytest.param(
        pytest.param({}, id="no_kwds"),
        pytest.param({"axis": 1}, id="on_axis"),
        pytest.param({"numeric_only": True}, id="func_kwds"),
        pytest.param({"axis": 1, "numeric_only": True}, id="axis_and_func_kwds"),
    ],
)
@pytest.mark.parametrize("how", ["agg", "apply"])
def test_apply_with_string_funcs(float_frame, func, kwds, how):
    # 调用 float_frame 的 agg 或 apply 方法，应用指定的函数 func 和关键字参数 kwds
    result = getattr(float_frame, how)(func, **kwds)
    # 调用 float_frame 的 func 方法，应用于所有列并带上关键字参数 kwds
    expected = getattr(float_frame, func)(**kwds)
    # 断言两个结果 Series 相等
    tm.assert_series_equal(result, expected)


def test_with_string_args(datetime_series, all_numeric_reductions):
    # 对 datetime_series 应用 all_numeric_reductions 函数
    result = datetime_series.apply(all_numeric_reductions)
    # 获取 datetime_series 上 all_numeric_reductions 方法的预期结果
    expected = getattr(datetime_series, all_numeric_reductions)()
    # 断言结果相等
    assert result == expected


@pytest.mark.parametrize("op", ["mean", "median", "std", "var"])
@pytest.mark.parametrize("how", ["agg", "apply"])
def test_apply_np_reducer(op, how):
    # 创建一个 DataFrame float_frame，包含列 'a' 和 'b' 的数据
    float_frame = DataFrame({"a": [1, 2], "b": [3, 4]})
    # 调用 float_frame 的 agg 或 apply 方法，应用指定的函数 op
    result = getattr(float_frame, how)(op)
    # 根据 numpy 中 op 函数的结果创建一个 Series，以 float_frame 的列名为索引
    kwargs = {"ddof": 1} if op in ("std", "var") else {}
    expected = Series(
        getattr(np, op)(float_frame, axis=0, **kwargs), index=float_frame.columns
    )
    # 断言两个 Series 相等
    tm.assert_series_equal(result, expected)


@pytest.mark.skipif(WASM, reason="No fp exception support in wasm")
@pytest.mark.parametrize(
    "op", ["abs", "ceil", "cos", "cumsum", "exp", "log", "sqrt", "square"]
)
@pytest.mark.parametrize("how", ["transform", "apply"])
def test_apply_np_transformer(float_frame, op, how):
    # 对 float_frame 中的第一个元素应用 op 操作
    float_frame.iloc[0, 0] = -1.0
    warn = None
    if op in ["log", "sqrt"]:
        warn = RuntimeWarning

    with tm.assert_produces_warning(warn, check_stacklevel=False):
        # 调用 float_frame 的 transform 或 apply 方法，应用指定的函数 op
        result = getattr(float_frame, how)(op)
        # 根据 numpy 中 op 函数的结果创建一个 DataFrame
        expected = getattr(np, op)(float_frame)
    # 断言两个 DataFrame 相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "series, func, expected",
    chain(
        # 调用测试模块中的函数获取 Cython 表格参数，传入一个空的浮点数类型的 Series
        # 并提供参数列表，描述需要计算的统计量及其初始值
        tm.get_cython_table_params(
            Series(dtype=np.float64),
            [
                ("sum", 0),         # 计算总和，初始值为 0
                ("max", np.nan),    # 计算最大值，初始值为 NaN
                ("min", np.nan),    # 计算最小值，初始值为 NaN
                ("all", True),      # 判断所有元素是否为真值，初始为 True
                ("any", False),     # 判断任意元素是否为真值，初始为 False
                ("mean", np.nan),   # 计算平均值，初始值为 NaN
                ("prod", 1),        # 计算所有元素的乘积，初始值为 1
                ("std", np.nan),    # 计算标准差，初始值为 NaN
                ("var", np.nan),    # 计算方差，初始值为 NaN
                ("median", np.nan), # 计算中位数，初始值为 NaN
            ],
        ),
        # 再次调用函数，传入一个包含 NaN、1、2、3 的 Series
        # 并提供相同的参数列表，以便测试函数对不同数据集的表现
        tm.get_cython_table_params(
            Series([np.nan, 1, 2, 3]),
            [
                ("sum", 6),         # 期望总和为 6
                ("max", 3),         # 期望最大值为 3
                ("min", 1),         # 期望最小值为 1
                ("all", True),      # 期望所有元素为真值
                ("any", True),      # 期望任意元素为真值
                ("mean", 2),        # 期望平均值为 2
                ("prod", 6),        # 期望所有元素的乘积为 6
                ("std", 1),         # 期望标准差为 1
                ("var", 1),         # 期望方差为 1
                ("median", 2),      # 期望中位数为 2
            ],
        ),
        # 最后一次调用函数，传入一个包含字符串 "a b c" 的 Series
        # 并提供参数列表，测试函数对字符串的处理能力
        tm.get_cython_table_params(
            Series("a b c".split()),
            [
                ("sum", "abc"),     # 期望总和为 "abc"
                ("max", "c"),       # 期望最大值为 "c"
                ("min", "a"),       # 期望最小值为 "a"
                ("all", True),      # 期望所有元素为真值
                ("any", True),      # 期望任意元素为真值
            ],
        ),
    ),
# GH 21224
# 在 pandas.core.base.SelectionMixin._cython_table 中测试聚合函数（sum, max, min, all, any, mean, prod, std, var, median）
@pytest.mark.parametrize(
    "df, func, expected",
    chain(
        # 获取 DataFrame 的 Cython 表参数
        tm.get_cython_table_params(
            DataFrame(),
            [
                ("sum", Series(dtype="float64")),  # 求和函数，期望返回 float64 的 Series
                ("max", Series(dtype="float64")),  # 最大值函数，期望返回 float64 的 Series
                ("min", Series(dtype="float64")),  # 最小值函数，期望返回 float64 的 Series
                ("all", Series(dtype=bool)),       # 全部为真函数，期望返回布尔类型的 Series
                ("any", Series(dtype=bool)),       # 任意为真函数，期望返回布尔类型的 Series
                ("mean", Series(dtype="float64")), # 均值函数，期望返回 float64 的 Series
                ("prod", Series(dtype="float64")), # 积函数，期望返回 float64 的 Series
                ("std", Series(dtype="float64")),  # 标准差函数，期望返回 float64 的 Series
                ("var", Series(dtype="float64")),  # 方差函数，期望返回 float64 的 Series
                ("median", Series(dtype="float64")), # 中位数函数，期望返回 float64 的 Series
            ],
        ),
        tm.get_cython_table_params(
            DataFrame([[np.nan, 1], [1, 2]]),
            [
                ("sum", Series([1.0, 3])),       # 求和函数，期望返回 [1.0, 3] 的 Series
                ("max", Series([1.0, 2])),       # 最大值函数，期望返回 [1.0, 2] 的 Series
                ("min", Series([1.0, 1])),       # 最小值函数，期望返回 [1.0, 1] 的 Series
                ("all", Series([True, True])),   # 全部为真函数，期望返回 [True, True] 的 Series
                ("any", Series([True, True])),   # 任意为真函数，期望返回 [True, True] 的 Series
                ("mean", Series([1, 1.5])),      # 均值函数，期望返回 [1, 1.5] 的 Series
                ("prod", Series([1.0, 2])),      # 积函数，期望返回 [1.0, 2] 的 Series
                ("std", Series([np.nan, 0.707107])),  # 标准差函数，期望返回 [np.nan, 0.707107] 的 Series
                ("var", Series([np.nan, 0.5])),  # 方差函数，期望返回 [np.nan, 0.5] 的 Series
                ("median", Series([1, 1.5])),    # 中位数函数，期望返回 [1, 1.5] 的 Series
            ],
        ),
    ),
)
def test_agg_cython_table_frame(df, func, expected, axis):
    # GH 21224
    # 在 pandas.core.base.SelectionMixin._cython_table 中测试聚合函数
    # 如果 func 是字符串类型，则将 warn 设置为 None，否则设置为 FutureWarning
    warn = None if isinstance(func, str) else FutureWarning
    
    # 使用 tm.assert_produces_warning 上下文管理器来确保在执行下面代码块期间生成警告，
    # 警告信息应匹配 "is currently using DataFrame.*"
    with tm.assert_produces_warning(warn, match="is currently using DataFrame.*"):
        # 执行聚合操作 func 在指定轴上的结果
        # GH#53425 是一个参考问题号，可能指示相关问题的修复或解决方案
        result = df.agg(func, axis=axis)
    
    # 使用 tm.assert_series_equal 检查 result 和预期的结果是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器来参数化测试用例
    "df, func, expected",  # 定义参数化的三个参数：df, func, expected
    chain(  # 使用 itertools.chain 将多个生成器链在一起作为参数
        tm.get_cython_table_params(  # 调用 tm 模块中的 get_cython_table_params 函数
            DataFrame(),  # 创建一个空的 pandas DataFrame
            [("cumprod", DataFrame()), ("cumsum", DataFrame())]  # 参数为包含元组的列表
        ),
        tm.get_cython_table_params(  # 再次调用 get_cython_table_params 函数
            DataFrame([[np.nan, 1], [1, 2]]),  # 创建一个包含 NaN 和数字的 DataFrame
            [
                ("cumprod", DataFrame([[np.nan, 1], [1, 2]])),  # 第一个操作为累积乘积
                ("cumsum", DataFrame([[np.nan, 1], [1, 3]])),  # 第二个操作为累积和
            ],
        ),
    ),
)
def test_agg_cython_table_transform_frame(df, func, expected, axis):
    # GH 21224
    # 对 pandas.core.base.SelectionMixin._cython_table 中的 cumprod 和 cumsum 函数进行测试转换操作

    if axis in ("columns", 1):
        # 如果 axis 是 "columns" 或者 1，则执行以下操作
        # 因为按列操作无法保留数据类型
        expected = expected.astype("float64")

    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match="is currently using DataFrame.*"):
        # GH#53425
        # 在执行 df.agg(func, axis=axis) 操作时，捕获 FutureWarning 警告信息
        result = df.agg(func, axis=axis)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("op", series_transform_kernels)
def test_transform_groupby_kernel_series(request, string_series, op):
    # GH 35964
    # 对于 Series 的 groupby 操作的转换核心功能进行测试
    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )
        # 如果操作是 "ngroup"，标记为预期失败，因为对 NDFrame 不支持 ngroup 操作

    args = [0.0] if op == "fillna" else []
    ones = np.ones(string_series.shape[0])

    warn = FutureWarning if op == "fillna" else None
    msg = "SeriesGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=msg):
        # 捕获针对 fillna 操作的 FutureWarning 警告信息
        expected = string_series.groupby(ones).transform(op, *args)
    result = string_series.transform(op, 0, *args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("op", frame_transform_kernels)
def test_transform_groupby_kernel_frame(request, float_frame, op):
    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )
        # 如果操作是 "ngroup"，标记为预期失败，因为对 NDFrame 不支持 ngroup 操作

    # GH 35964
    # 对于 DataFrame 的 groupby 操作的转换核心功能进行测试

    args = [0.0] if op == "fillna" else []
    ones = np.ones(float_frame.shape[0])
    gb = float_frame.groupby(ones)

    warn = FutureWarning if op == "fillna" else None
    op_msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=op_msg):
        # 捕获针对 fillna 操作的 FutureWarning 警告信息
        expected = gb.transform(op, *args)

    result = float_frame.transform(op, 0, *args)
    tm.assert_frame_equal(result, expected)

    # same thing, but ensuring we have multiple blocks
    assert "E" not in float_frame.columns
    float_frame["E"] = float_frame["A"].copy()
    assert len(float_frame._mgr.blocks) > 1

    ones = np.ones(float_frame.shape[0])
    gb2 = float_frame.groupby(ones)
    expected2 = gb2.transform(op, *args)
    result2 = float_frame.transform(op, 0, *args)
    tm.assert_frame_equal(result2, expected2)


@pytest.mark.parametrize("method", ["abs", "shift", "pct_change", "cumsum", "rank"])
def test_transform_method_name(method):
    # GH 19760
    # 测试 DataFrame 和 Series 的转换方法（abs, shift, pct_change, cumsum, rank）的功能
    # 创建一个包含两列的 DataFrame，列 "A" 包含值 [-1, 2]
    df = DataFrame({"A": [-1, 2]})
    # 对 DataFrame 进行转换操作，使用指定的方法 `method`
    result = df.transform(method)
    # 使用 operator 模块中的方法调用器 methodcaller 调用指定方法 `method` 并将其应用于 DataFrame `df`
    expected = operator.methodcaller(method)(df)
    # 使用测试框架中的函数 tm.assert_frame_equal 比较 `result` 和 `expected` 是否相等
    tm.assert_frame_equal(result, expected)
```