# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_encode.py`

```
    [
        (
            np.array([1, 2, 3, 4, 5]),  # 测试用整数数组
            np.array([1, 2, 3]),  # 预期唯一值数组
            np.array([4, 5]),  # 预期不包含在唯一值数组中的差异值
            np.array([True, True, True, False, False])  # 预期的有效掩码数组
        ),
        (
            np.array(["a", "b", "c", "d"]),  # 测试用对象数组
            np.array(["a", "b", "c"]),  # 预期唯一值数组
            np.array(["d"]),  # 预期不包含在唯一值数组中的差异值
            np.array([True, True, True, False])  # 预期的有效掩码数组
        ),
        (
            np.array([1.0, 2.0, np.nan]),  # 测试用浮点数数组（包括 NaN）
            np.array([1.0, 2.0]),  # 预期唯一值数组
            np.array([np.nan]),  # 预期不包含在唯一值数组中的差异值
            np.array([True, True, False])  # 预期的有效掩码数组
        ),
    ],
    ids=["int_array", "object_array", "float_nan_array"],  # 测试用例标识
)
def test_check_unknown(values, uniques, expected_diff, expected_mask):
    # 调用 _check_unknown 函数，比较差异值是否符合预期
    diff = _check_unknown(values, uniques)
    assert_array_equal(diff, expected_diff)

    # 调用 _check_unknown 函数，同时返回掩码数组，比较差异值和掩码数组是否符合预期
    diff, valid_mask = _check_unknown(values, uniques, return_mask=True)
    assert_array_equal(diff, expected_diff)
    assert_array_equal(valid_mask, expected_mask)
    [
        # 第一个元组：两个 NumPy 数组和两个列表
        (np.array([1, 2, 3, 4]), np.array([1, 2, 3]), [4], [True, True, True, False]),
        # 第二个元组：两个 NumPy 数组和两个列表
        (np.array([2, 1, 4, 5]), np.array([2, 5, 1]), [4], [True, True, False, True]),
        # 第三个元组：包含 NaN 的 NumPy 数组、一个包含 NaN 的列表
        (np.array([2, 1, np.nan]), np.array([2, 5, 1]), [np.nan], [True, True, False]),
        # 第四个元组：包含 NaN 的两个 NumPy 数组和两个列表
        (
            np.array([2, 1, 4, np.nan]),
            np.array([2, 5, 1, np.nan]),
            [4],
            [True, True, False, True],
        ),
        # 第五个元组：一个包含 NaN 的 NumPy 数组、一个不包含 NaN 的 NumPy 数组和两个列表
        (
            np.array([2, 1, 4, np.nan]),
            np.array([2, 5, 1]),
            [4, np.nan],
            [True, True, False, False],
        ),
        # 第六个元组：一个包含 NaN 的 NumPy 数组和一个包含 NaN 的 NumPy 数组以及两个列表
        (
            np.array([2, 1, 4, 5]),
            np.array([2, 5, 1, np.nan]),
            [4],
            [True, True, False, True],
        ),
        # 第七个元组：包含对象的 NumPy 数组、另一个包含对象的 NumPy 数组、一个包含对象的列表和一个布尔列表
        (
            np.array(["a", "b", "c", "d"], dtype=object),
            np.array(["a", "b", "c"], dtype=object),
            np.array(["d"], dtype=object),
            [True, True, True, False],
        ),
        # 第八个元组：包含对象的 NumPy 数组、另一个包含对象的 NumPy 数组、一个包含对象的列表和一个布尔列表
        (
            np.array(["d", "c", "a", "b"], dtype=object),
            np.array(["a", "c", "b"], dtype=object),
            np.array(["d"], dtype=object),
            [False, True, True, True],
        ),
        # 第九个元组：包含对象的 NumPy 数组、另一个包含对象的 NumPy 数组、一个包含对象的列表和一个布尔列表
        (
            np.array(["a", "b", "c", "d"]),
            np.array(["a", "b", "c"]),
            np.array(["d"]),
            [True, True, True, False],
        ),
        # 第十个元组：包含对象的 NumPy 数组、另一个包含对象的 NumPy 数组、一个包含对象的列表和一个布尔列表
        (
            np.array(["d", "c", "a", "b"]),
            np.array(["a", "c", "b"]),
            np.array(["d"]),
            [False, True, True, True],
        ),
    ],
# 定义测试函数 test_check_unknown，用于测试 _assert_check_unknown 函数
def test_check_unknown(values, uniques, expected_diff, expected_mask):
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)


# 使用 pytest.mark.parametrize 标记，参数化测试函数 test_check_unknown_missing_values
@pytest.mark.parametrize("missing_value", [None, np.nan, float("nan")])
@pytest.mark.parametrize("pickle_uniques", [True, False])
def test_check_unknown_missing_values(missing_value, pickle_uniques):
    # 测试带有缺失值的 check_unknown 函数，对应对象类型的数据
    values = np.array(["d", "c", "a", "b", missing_value], dtype=object)
    uniques = np.array(["c", "a", "b", missing_value], dtype=object)
    if pickle_uniques:
        # 使用 pickle 序列化和反序列化 uniques
        uniques = pickle.loads(pickle.dumps(uniques))

    expected_diff = ["d"]
    expected_mask = [False, True, True, True, True]
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)

    values = np.array(["d", "c", "a", "b", missing_value], dtype=object)
    uniques = np.array(["c", "a", "b"], dtype=object)
    if pickle_uniques:
        # 使用 pickle 序列化和反序列化 uniques
        uniques = pickle.loads(pickle.dumps(uniques))

    expected_diff = ["d", missing_value]
    expected_mask = [False, True, True, True, False]
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)

    values = np.array(["a", missing_value], dtype=object)
    uniques = np.array(["a", "b", "z"], dtype=object)
    if pickle_uniques:
        # 使用 pickle 序列化和反序列化 uniques
        uniques = pickle.loads(pickle.dumps(uniques))

    expected_diff = [missing_value]
    expected_mask = [True, False]
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)


# 使用 pytest.mark.parametrize 标记，参数化测试函数 test_unique_util_missing_values_objects
@pytest.mark.parametrize("missing_value", [np.nan, None, float("nan")])
@pytest.mark.parametrize("pickle_uniques", [True, False])
def test_unique_util_missing_values_objects(missing_value, pickle_uniques):
    # 测试带有缺失值的 _unique 和 _encode 函数，对应对象类型的数据
    values = np.array(["a", "c", "c", missing_value, "b"], dtype=object)
    expected_uniques = np.array(["a", "b", "c", missing_value], dtype=object)

    uniques = _unique(values)

    if missing_value is None:
        assert_array_equal(uniques, expected_uniques)
    else:  # missing_value == np.nan
        assert_array_equal(uniques[:-1], expected_uniques[:-1])
        assert np.isnan(uniques[-1])

    if pickle_uniques:
        # 使用 pickle 序列化和反序列化 uniques
        uniques = pickle.loads(pickle.dumps(uniques))

    encoded = _encode(values, uniques=uniques)
    assert_array_equal(encoded, np.array([0, 2, 2, 3, 1]))


# 定义测试函数 test_unique_util_missing_values_numeric
def test_unique_util_missing_values_numeric():
    # 检查数值类型数据中的缺失值
    values = np.array([3, 1, np.nan, 5, 3, np.nan], dtype=float)
    expected_uniques = np.array([1, 3, 5, np.nan], dtype=float)
    expected_inverse = np.array([1, 0, 3, 2, 1, 3])

    uniques = _unique(values)
    assert_array_equal(uniques, expected_uniques)

    uniques, inverse = _unique(values, return_inverse=True)
    assert_array_equal(uniques, expected_uniques)
    assert_array_equal(inverse, expected_inverse)

    encoded = _encode(values, uniques=uniques)
    assert_array_equal(encoded, expected_inverse)
def test_unique_util_with_all_missing_values():
    # test for all types of missing values for object dtype
    # 创建一个包含各种缺失值类型的对象数组
    values = np.array([np.nan, "a", "c", "c", None, float("nan"), None], dtype=object)

    # 调用 _unique 函数获取唯一值
    uniques = _unique(values)
    # 断言前面部分（除了最后一个）与预期的唯一值相等
    assert_array_equal(uniques[:-1], ["a", "c", None])
    # 最后一个值应为 NaN
    # 断言最后一个值是 NaN
    assert np.isnan(uniques[-1])

    # 预期的反向索引数组
    expected_inverse = [3, 0, 1, 1, 2, 3, 2]
    # 调用 _unique 函数返回反向索引
    _, inverse = _unique(values, return_inverse=True)
    # 断言反向索引与预期的反向索引数组相等
    assert_array_equal(inverse, expected_inverse)


def test_check_unknown_with_both_missing_values():
    # test for both types of missing values for object dtype
    # 创建一个包含两种类型缺失值的对象数组
    values = np.array([np.nan, "a", "c", "c", None, np.nan, None], dtype=object)

    # 使用 _check_unknown 函数检查未知值，已知值为 ["a", "c"]
    diff = _check_unknown(values, known_values=np.array(["a", "c"], dtype=object))
    # 断言第一个差异为 None
    assert diff[0] is None
    # 断言第二个差异为 NaN
    assert np.isnan(diff[1])

    # 使用 _check_unknown 函数返回差异和有效掩码
    diff, valid_mask = _check_unknown(
        values, known_values=np.array(["a", "c"], dtype=object), return_mask=True
    )

    # 断言第一个差异为 None
    assert diff[0] is None
    # 断言第二个差异为 NaN
    assert np.isnan(diff[1])
    # 断言有效掩码与预期的相等
    assert_array_equal(valid_mask, [False, True, True, True, False, False, False])


@pytest.mark.parametrize(
    "values, uniques, expected_counts",
    [
        # 参数化测试用例，测试不同的输入值、唯一值和预期计数
        (np.array([1] * 10 + [2] * 4 + [3] * 15), np.array([1, 2, 3]), [10, 4, 15]),
        (
            np.array([1] * 10 + [2] * 4 + [3] * 15),
            np.array([1, 2, 3, 5]),
            [10, 4, 15, 0],
        ),
        (
            np.array([np.nan] * 10 + [2] * 4 + [3] * 15),
            np.array([2, 3, np.nan]),
            [4, 15, 10],
        ),
        (
            np.array(["b"] * 4 + ["a"] * 16 + ["c"] * 20, dtype=object),
            ["a", "b", "c"],
            [16, 4, 20],
        ),
        (
            np.array(["b"] * 4 + ["a"] * 16 + ["c"] * 20, dtype=object),
            ["c", "b", "a"],
            [20, 4, 16],
        ),
        (
            np.array([np.nan] * 4 + ["a"] * 16 + ["c"] * 20, dtype=object),
            ["c", np.nan, "a"],
            [20, 4, 16],
        ),
        (
            np.array(["b"] * 4 + ["a"] * 16 + ["c"] * 20, dtype=object),
            ["a", "b", "c", "e"],
            [16, 4, 20, 0],
        ),
    ],
)
def test_get_counts(values, uniques, expected_counts):
    # 调用 _get_counts 函数计算值的计数
    counts = _get_counts(values, uniques)
    # 断言计数结果与预期的计数数组相等
    assert_array_equal(counts, expected_counts)
```