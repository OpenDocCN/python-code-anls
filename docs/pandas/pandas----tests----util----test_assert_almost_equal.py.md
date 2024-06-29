# `D:\src\scipysrc\pandas\pandas\tests\util\test_assert_almost_equal.py`

```
@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1.1),              # 参数化测试：a 和 b 相等的情况
        (1.1, 1.100001),         # 参数化测试：a 和 b 几乎相等的情况
        (np.int16(1), 1.000001), # 参数化测试：不同类型但几乎相等的情况
        (np.float64(1.1), 1.1),  # 参数化测试：不同类型但相等的情况
        (np.uint32(5), 5),       # 参数化测试：不同类型但相等的情况
    ],
)
def test_assert_almost_equal_numbers(a, b):
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1),                # 参数化测试：a 和 b 不相等的情况
        (1.1, True),             # 参数化测试：a 和 b 类型完全不同的情况
        (1, 2),                  # 参数化测试：a 和 b 不相等的情况
        (1.0001, np.int16(1)),   # 参数化测试：a 和 b 几乎相等但类型不同的情况
        # 以下两个示例由于 `tol` 的原因不是几乎相等的情况
        (0.1, 0.1001),           # 参数化测试：a 和 b 不几乎相等的情况
        (0.0011, 0.0012),        # 参数化测试：a 和 b 不几乎相等的情况
    ],
)
def test_assert_not_almost_equal_numbers(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1.1),              # 参数化测试：a 和 b 相等的情况
        (1.1, 1.100001),         # 参数化测试：a 和 b 几乎相等的情况
        (1.1, 1.1001),           # 参数化测试：a 和 b 几乎相等但不满足给定的 `atol` 和 `rtol`
        (0.000001, 0.000005),    # 参数化测试：a 和 b 几乎相等的情况
        (1000.0, 1000.0005),     # 参数化测试：a 和 b 几乎相等的情况
        # 根据 #13357 测试这个例子
        (0.000011, 0.000012),    # 参数化测试：a 和 b 几乎相等的情况
    ],
)
def test_assert_almost_equal_numbers_atol(a, b):
    # 使用指定的 `atol` 和 `rtol` 进行几乎相等性断言测试
    _assert_almost_equal_both(a, b, rtol=0.5e-3, atol=0.5e-3)


@pytest.mark.parametrize("a,b", [(1.1, 1.11), (0.1, 0.101), (0.000011, 0.001012)])
def test_assert_not_almost_equal_numbers_atol(a, b):
    # 使用指定的 `atol` 进行非几乎相等性断言测试
    _assert_not_almost_equal_both(a, b, atol=1e-3)


这些注释详细解释了每个参数化测试函数的目的和示例，确保了代码的可读性和理解性。
    [
        # 元组1: (1.1, 1.1) - 包含相同精度的浮点数对
        (1.1, 1.1),
        # 元组2: (1.1, 1.100001) - 稍微不同精度的浮点数对
        (1.1, 1.100001),
        # 元组3: (1.1, 1.1001) - 更显著不同精度的浮点数对
        (1.1, 1.1001),
        # 元组4: (1000.0, 1000.0005) - 大数和稍微不同精度的浮点数对
        (1000.0, 1000.0005),
        # 元组5: (1.1, 1.11) - 略微不同的浮点数对
        (1.1, 1.11),
        # 元组6: (0.1, 0.101) - 小数和稍微不同精度的浮点数对
        (0.1, 0.101),
    ],
# 测试函数，用于比较两个数近似相等的情况，允许相对误差（relative tolerance）为0.05
def test_assert_almost_equal_numbers_rtol(a, b):
    _assert_almost_equal_both(a, b, rtol=0.05)


# 参数化测试函数，用于比较两个数不近似相等的情况，允许相对误差（relative tolerance）为0.05
@pytest.mark.parametrize("a,b", [(0.000011, 0.000012), (0.000001, 0.000005)])
def test_assert_not_almost_equal_numbers_rtol(a, b):
    _assert_not_almost_equal_both(a, b, rtol=0.05)


# 复数参数化测试函数，用于比较复数近似相等的情况，允许指定的相对误差（relative tolerance）
@pytest.mark.parametrize(
    "a,b,rtol",
    [
        (1.00001, 1.00005, 0.001),
        (-0.908356 + 0.2j, -0.908358 + 0.2j, 1e-3),
        (0.1 + 1.009j, 0.1 + 1.006j, 0.1),
        (0.1001 + 2.0j, 0.1 + 2.001j, 0.01),
    ],
)
def test_assert_almost_equal_complex_numbers(a, b, rtol):
    _assert_almost_equal_both(a, b, rtol=rtol)
    _assert_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)


# 复数参数化测试函数，用于比较复数不近似相等的情况，允许指定的相对误差（relative tolerance）
@pytest.mark.parametrize(
    "a,b,rtol",
    [
        (0.58310768, 0.58330768, 1e-7),
        (-0.908 + 0.2j, -0.978 + 0.2j, 0.001),
        (0.1 + 1j, 0.1 + 2j, 0.01),
        (-0.132 + 1.001j, -0.132 + 1.005j, 1e-5),
        (0.58310768j, 0.58330768j, 1e-9),
    ],
)
def test_assert_not_almost_equal_complex_numbers(a, b, rtol):
    _assert_not_almost_equal_both(a, b, rtol=rtol)
    _assert_not_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_not_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)


# 参数化测试函数，用于比较数值与零近似相等的情况
@pytest.mark.parametrize("a,b", [(0, 0), (0, 0.0), (0, np.float64(0)), (0.00000001, 0)])
def test_assert_almost_equal_numbers_with_zeros(a, b):
    _assert_almost_equal_both(a, b)


# 参数化测试函数，用于比较数值与零不近似相等的情况
@pytest.mark.parametrize("a,b", [(0.001, 0), (1, 0)])
def test_assert_not_almost_equal_numbers_with_zeros(a, b):
    _assert_not_almost_equal_both(a, b)


# 参数化测试函数，用于比较数值类型不同的情况下不近似相等
@pytest.mark.parametrize("a,b", [(1, "abc"), (1, [1]), (1, object())])
def test_assert_not_almost_equal_numbers_with_mixed(a, b):
    _assert_not_almost_equal_both(a, b)


# 边缘情况测试函数，用于比较空数组时的近似相等
@pytest.mark.parametrize(
    "left_dtype", ["M8[ns]", "m8[ns]", "float64", "int64", "object"]
)
@pytest.mark.parametrize(
    "right_dtype", ["M8[ns]", "m8[ns]", "float64", "int64", "object"]
)
def test_assert_almost_equal_edge_case_ndarrays(left_dtype, right_dtype):
    # 空数组比较，不检查数据类型
    _assert_almost_equal_both(
        np.array([], dtype=left_dtype),
        np.array([], dtype=right_dtype),
        check_dtype=False,
    )


# 测试函数，用于比较集合的近似相等情况
def test_assert_almost_equal_sets():
    # GH#51727
    _assert_almost_equal_both({1, 2, 3}, {1, 2, 3})


# 测试函数，用于比较集合的不近似相等情况
def test_assert_almost_not_equal_sets():
    # GH#51727
    msg = r"{1, 2, 3} != {1, 2, 4}"
    with pytest.raises(AssertionError, match=msg):
        _assert_almost_equal_both({1, 2, 3}, {1, 2, 4})


# 测试函数，用于比较字典的近似相等情况
def test_assert_almost_equal_dicts():
    _assert_almost_equal_both({"a": 1, "b": 2}, {"a": 1, "b": 2})


# 参数化测试函数，用于比较字典的不近似相等情况
@pytest.mark.parametrize(
    "a,b",
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}),
        ({"a": 1}, 1),
        ({"a": 1}, "abc"),
        ({"a": 1}, [1]),
    ],
)
def test_assert_not_almost_equal_dicts(a, b):
    _assert_not_almost_equal_both(a, b)
@pytest.mark.parametrize("val", [1, 2])
def test_assert_almost_equal_dict_like_object(val):
    # 设定一个字典类型的值
    dict_val = 1
    # 创建一个真实的字典，包含键"a"和参数val对应的值
    real_dict = {"a": val}

    # 定义一个类DictLikeObj，模拟字典对象
    class DictLikeObj:
        # 实现keys方法，返回键"a"
        def keys(self):
            return ("a",)

        # 实现getitem方法，当键为"a"时返回dict_val的值
        def __getitem__(self, item):
            if item == "a":
                return dict_val

    # 根据val和dict_val的关系选择要调用的函数
    func = (
        _assert_almost_equal_both if val == dict_val else _assert_not_almost_equal_both
    )
    # 调用选择的函数，传入real_dict和DictLikeObj的实例，禁用数据类型检查
    func(real_dict, DictLikeObj(), check_dtype=False)


def test_assert_almost_equal_strings():
    # 调用_assert_almost_equal_both函数，比较两个字符串"abc"和"abc"
    _assert_almost_equal_both("abc", "abc")


@pytest.mark.parametrize("b", ["abcd", "abd", 1, [1]])
def test_assert_not_almost_equal_strings(b):
    # 调用_assert_not_almost_equal_both函数，分别比较字符串"abc"与参数b的值
    _assert_not_almost_equal_both("abc", b)


@pytest.mark.parametrize("box", [list, np.array])
def test_assert_almost_equal_iterables(box):
    # 调用_assert_almost_equal_both函数，分别比较包含[1, 2, 3]的box类型对象
    _assert_almost_equal_both(box([1, 2, 3]), box([1, 2, 3]))


@pytest.mark.parametrize(
    "a,b",
    [
        # 类型不同的情况
        (np.array([1, 2, 3]), [1, 2, 3]),
        # 数据类型不同的情况
        (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])),
        # 无法比较生成器的情况
        (iter([1, 2, 3]), [1, 2, 3]),
        # 其他不相等的情况
        ([1, 2, 3], [1, 2, 4]),
        ([1, 2, 3], [1, 2, 3, 4]),
        ([1, 2, 3], 1),
    ],
)
def test_assert_not_almost_equal_iterables(a, b):
    # 调用_assert_not_almost_equal函数，分别比较参数a和参数b的值
    _assert_not_almost_equal(a, b)


def test_assert_almost_equal_null():
    # 调用_assert_almost_equal_both函数，比较两个空值None
    _assert_almost_equal_both(None, None)


@pytest.mark.parametrize("a,b", [(None, np.nan), (None, 0), (np.nan, 0)])
def test_assert_not_almost_equal_null(a, b):
    # 调用_assert_not_almost_equal函数，分别比较参数a和参数b的值
    _assert_not_almost_equal(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        # 比较无穷大的情况
        (np.inf, np.inf),
        (np.inf, float("inf")),
        (np.array([np.inf, np.nan, -np.inf]), np.array([np.inf, np.nan, -np.inf])),
    ],
)
def test_assert_almost_equal_inf(a, b):
    # 调用_assert_almost_equal_both函数，分别比较参数a和参数b的值
    _assert_almost_equal_both(a, b)


objs = [NA, np.nan, NaT, None, np.datetime64("NaT"), np.timedelta64("NaT")]


@pytest.mark.parametrize("left", objs)
@pytest.mark.parametrize("right", objs)
def test_mismatched_na_assert_almost_equal(left, right):
    # 创建包含left和right的numpy对象数组，数据类型为object
    left_arr = np.array([left], dtype=object)
    right_arr = np.array([right], dtype=object)

    # 设置错误消息字符串
    msg = "Mismatched null-like values"

    # 如果left和right相同，则执行以下断言和比较
    if left is right:
        # 调用_assert_almost_equal_both函数，比较left和right的值，禁用数据类型检查
        _assert_almost_equal_both(left, right, check_dtype=False)
        # 断言左右两个数组相等
        tm.assert_numpy_array_equal(left_arr, right_arr)
        # 断言左右两个索引对象相等
        tm.assert_index_equal(
            Index(left_arr, dtype=object), Index(right_arr, dtype=object)
        )
        # 断言左右两个Series对象相等
        tm.assert_series_equal(
            Series(left_arr, dtype=object), Series(right_arr, dtype=object)
        )
        # 断言左右两个DataFrame对象相等
        tm.assert_frame_equal(
            DataFrame(left_arr, dtype=object), DataFrame(right_arr, dtype=object)
        )
    else:
        # 使用 pytest 断言检查 left 和 right 是否几乎相等，不检查数据类型
        with pytest.raises(AssertionError, match=msg):
            _assert_almost_equal_both(left, right, check_dtype=False)

        # TODO: 为了在 assert_numpy_array_equal 中获得相同的废弃警告，我们需要
        #  改变/废弃 strict_nan 的默认值为 True
        # TODO: 为了在 assert_index_equal 中获得相同的废弃警告，我们需要
        #  更改/废弃 array_equivalent_object 为更严格模式，因为
        #  assert_index_equal 使用 Index.equal，它又使用了 array_equivalent。
        # 使用 pytest 断言检查 Series 对象是否相等，dtype 为 object 类型
        with pytest.raises(AssertionError, match="Series are different"):
            tm.assert_series_equal(
                Series(left_arr, dtype=object), Series(right_arr, dtype=object)
            )
        # 使用 pytest 断言检查 DataFrame 的 iloc 内容是否相等，dtype 为 object 类型
        with pytest.raises(AssertionError, match="DataFrame.iloc.* are different"):
            tm.assert_frame_equal(
                DataFrame(left_arr, dtype=object), DataFrame(right_arr, dtype=object)
            )
def test_assert_not_almost_equal_inf():
    # 调用内部辅助函数 _assert_not_almost_equal_both，断言无穷大与零不近似相等
    _assert_not_almost_equal_both(np.inf, 0)


@pytest.mark.parametrize(
    "a,b",
    [
        # 测试不同 Pandas 对象之间的近似相等性断言
        (Index([1.0, 1.1]), Index([1.0, 1.100001])),
        (Series([1.0, 1.1]), Series([1.0, 1.100001])),
        (np.array([1.1, 2.000001]), np.array([1.1, 2.0])),
        (DataFrame({"a": [1.0, 1.1]}), DataFrame({"a": [1.0, 1.100001]})),
    ],
)
def test_assert_almost_equal_pandas(a, b):
    # 调用内部辅助函数 _assert_almost_equal_both，测试两个对象的近似相等性
    _assert_almost_equal_both(a, b)


def test_assert_almost_equal_object():
    # 测试包含 Timestamp 对象的列表的近似相等性
    a = [Timestamp("2011-01-01"), Timestamp("2011-01-01")]
    b = [Timestamp("2011-01-01"), Timestamp("2011-01-01")]
    _assert_almost_equal_both(a, b)


def test_assert_almost_equal_value_mismatch():
    # 使用 pytest 框架验证数值不近似相等的情况
    msg = "expected 2\\.00000 but got 1\\.00000, with rtol=1e-05, atol=1e-08"

    with pytest.raises(AssertionError, match=msg):
        # 断言数值 1 和 2 不近似相等
        tm.assert_almost_equal(1, 2)


@pytest.mark.parametrize(
    "a,b,klass1,klass2",
    [(np.array([1]), 1, "ndarray", "int"), (1, np.array([1]), "int", "ndarray")],
)
def test_assert_almost_equal_class_mismatch(a, b, klass1, klass2):
    # 测试不同类别对象之间的近似相等性断言
    msg = f"""numpy array are different

numpy array classes are different
\\[left\\]:  {klass1}
\\[right\\]: {klass2}"""

    with pytest.raises(AssertionError, match=msg):
        # 断言不同类型的对象不近似相等
        tm.assert_almost_equal(a, b)


def test_assert_almost_equal_value_mismatch1():
    # 使用 pytest 框架验证 numpy 数组值不近似相等的情况
    msg = """numpy array are different

numpy array values are different \\(66\\.66667 %\\)
\\[left\\]:  \\[nan, 2\\.0, 3\\.0\\]
\\[right\\]: \\[1\\.0, nan, 3\\.0\\]"""

    with pytest.raises(AssertionError, match=msg):
        # 断言两个 numpy 数组不近似相等
        tm.assert_almost_equal(np.array([np.nan, 2, 3]), np.array([1, np.nan, 3]))


def test_assert_almost_equal_value_mismatch2():
    # 使用 pytest 框架验证 numpy 数组值不近似相等的情况
    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        # 断言两个 numpy 数组不近似相等
        tm.assert_almost_equal(np.array([1, 2]), np.array([1, 3]))


def test_assert_almost_equal_value_mismatch3():
    # 使用 pytest 框架验证 numpy 数组值不近似相等的情况
    msg = """numpy array are different

numpy array values are different \\(16\\.66667 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        # 断言两个二维 numpy 数组不近似相等
        tm.assert_almost_equal(
            np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]])
        )


def test_assert_almost_equal_value_mismatch4():
    # 使用 pytest 框架验证 numpy 数组值不近似相等的情况
    msg = """numpy array are different

numpy array values are different \\(25\\.0 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        # 断言两个二维 numpy 数组不近似相等
        tm.assert_almost_equal(np.array([[1, 2], [3, 4]]), np.array([[1, 3], [3, 4]]))


def test_assert_almost_equal_shape_mismatch_override():
    # 使用 pytest 框架验证索引对象形状不近似相等的情况
    msg = """Index are different

Index shapes are different
\\[left\\]:  \\(2L*,\\)
\\[right\\]: \\(3L*,\\)"""
    # 使用 pytest 的上下文管理器 `pytest.raises` 来捕获 AssertionError 异常，并验证异常消息是否匹配给定的正则表达式 msg。
    with pytest.raises(AssertionError, match=msg):
        # 使用 `tm.assert_almost_equal` 函数比较两个 NumPy 数组的近似相等性，指定其中一个数组为 [1, 2]，另一个数组为 [3, 4, 5]。
        tm.assert_almost_equal(np.array([1, 2]), np.array([3, 4, 5]), obj="Index")
def test_assert_almost_equal_unicode():
    # 用于测试 numpy 的 assert_almost_equal 方法对 Unicode 字符串的断言功能
    # 引发 AssertionError，匹配给定的错误消息
    msg = """numpy array are different

numpy array values are different \\(33\\.33333 %\\)
\\[left\\]:  \\[á, à, ä\\]
\\[right\\]: \\[á, à, å\\]"""

    with pytest.raises(AssertionError, match=msg):
        # 调用 assert_almost_equal 方法，对两个 numpy 数组进行比较
        tm.assert_almost_equal(np.array(["á", "à", "ä"]), np.array(["á", "à", "å"]))


def test_assert_almost_equal_timestamp():
    # 用于测试 numpy 的 assert_almost_equal 方法对时间戳的断言功能
    a = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-01")])
    b = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-02")])

    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[2011-01-01 00:00:00, 2011-01-01 00:00:00\\]
\\[right\\]: \\[2011-01-01 00:00:00, 2011-01-02 00:00:00\\]"""

    with pytest.raises(AssertionError, match=msg):
        # 调用 assert_almost_equal 方法，对两个 numpy 数组进行比较
        tm.assert_almost_equal(a, b)


def test_assert_almost_equal_iterable_length_mismatch():
    # 用于测试 numpy 的 assert_almost_equal 方法对迭代器长度不匹配的断言功能
    msg = """Iterable are different

Iterable length are different
\\[left\\]:  2
\\[right\\]: 3"""

    with pytest.raises(AssertionError, match=msg):
        # 调用 assert_almost_equal 方法，对两个列表进行比较
        tm.assert_almost_equal([1, 2], [3, 4, 5])


def test_assert_almost_equal_iterable_values_mismatch():
    # 用于测试 numpy 的 assert_almost_equal 方法对迭代器值不匹配的断言功能
    msg = """Iterable are different

Iterable values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        # 调用 assert_almost_equal 方法，对两个列表进行比较
        tm.assert_almost_equal([1, 2], [1, 3])


subarr = np.empty(2, dtype=object)
subarr[:] = [np.array([None, "b"], dtype=object), np.array(["c", "d"], dtype=object)]

NESTED_CASES = [
    # nested array
    (
        np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object),
        np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object),
    ),
    # >1 level of nesting
    (
        np.array(
            [
                np.array([np.array([50, 70]), np.array([90])], dtype=object),
                np.array([np.array([20, 30])], dtype=object),
            ],
            dtype=object,
        ),
        np.array(
            [
                np.array([np.array([50, 70]), np.array([90])], dtype=object),
                np.array([np.array([20, 30])], dtype=object),
            ],
            dtype=object,
        ),
    ),
    # lists
    (
        np.array([[50, 70, 90], [20, 30]], dtype=object),
        np.array([[50, 70, 90], [20, 30]], dtype=object),
    ),
    # mixed array/list
    (
        np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object),
        np.array([[1, 2, 3], [4, 5]], dtype=object),
    ),
    (
        np.array(
            [
                np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object),
                np.array(
                    [np.array([6]), np.array([7, 8]), np.array([9])], dtype=object
                ),
            ],
            dtype=object,
        ),
        np.array([[[1, 2, 3], [4, 5]], [[6], [7, 8], [9]]], dtype=object),
    ),
    # same-length lists
]
    (
        np.array([subarr, None], dtype=object),
        np.array([[[None, "b"], ["c", "d"]], None], dtype=object),
    ),
    # 创建一个包含两个 numpy 数组的元组，每个数组都包含对象类型的数据
    # 第一个数组包含 subarr 变量和 None
    # 第二个数组是一个二维数组，包含两个子数组，每个子数组包含对象类型的数据

    # dicts
    (
        np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object),
        np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object),
    ),
    # 创建一个包含两个 numpy 数组的元组，每个数组都包含对象类型的数据
    # 每个数组包含一个字典，字典中有两个键值对："f1" 是整数 1，"f2" 是包含对象类型数据的 numpy 数组 ["a", "b"]

    (
        np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object),
        np.array([{"f1": 1, "f2": ["a", "b"]}], dtype=object),
    ),
    # 创建一个包含两个 numpy 数组的元组，每个数组都包含对象类型的数据
    # 第一个数组的字典中的 "f2" 键对应一个 numpy 数组 ["a", "b"]，数据类型是对象类型
    # 第二个数组的字典中的 "f2" 键对应一个列表 ["a", "b"]，数据类型是对象类型

    # array/list of dicts
    (
        np.array(
            [
                np.array(
                    [{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object
                ),
                np.array([], dtype=object),
            ],
            dtype=object,
        ),
        np.array([[{"f1": 1, "f2": ["a", "b"]}], []], dtype=object),
    ),
    # 创建一个包含两个 numpy 数组的元组，每个数组都包含对象类型的数据
    # 第一个数组是一个包含两个元素的 numpy 数组，每个元素都是一个包含对象类型数据的 numpy 数组的对象
    # 第一个元素的 numpy 数组的字典中的 "f2" 键对应一个列表 ["a", "b"]
    # 第二个数组是一个二维 numpy 数组，包含两个元素，第一个元素是一个包含一个字典的列表 [{"f1": 1, "f2": ["a", "b"]}]，第二个元素是一个空列表
# 使用 pytest 的 mark 函数标记测试用例，在运行测试时忽略特定的警告信息
@pytest.mark.filterwarnings("ignore:elementwise comparison failed:DeprecationWarning")
# 使用 pytest 的 mark 函数参数化测试用例，传入参数 a 和 b，这些参数来自于 NESTED_CASES
@pytest.mark.parametrize("a,b", NESTED_CASES)
# 定义一个测试函数，用于测试 assert_almost_equal_array_nested 函数的功能
def test_assert_almost_equal_array_nested(a, b):
    # 调用 _assert_almost_equal_both 函数来对参数 a 和 b 进行断言测试
    _assert_almost_equal_both(a, b)
```