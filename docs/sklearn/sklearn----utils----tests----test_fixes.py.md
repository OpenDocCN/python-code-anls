# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_fixes.py`

```
# 导入所需的库和模块
import numpy as np  # 导入NumPy库，用于处理数组
import pytest  # 导入pytest库，用于编写和运行测试

# 导入需要的测试辅助函数和类
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import _object_dtype_isnan, _smallest_admissible_index_dtype

# 定义测试函数，参数化测试不同的dtype和val组合
@pytest.mark.parametrize("dtype, val", ([object, 1], [object, "a"], [float, 1]))
def test_object_dtype_isnan(dtype, val):
    # 创建一个包含NaN的二维数组
    X = np.array([[val, np.nan], [np.nan, val]], dtype=dtype)

    # 期望得到的NaN掩码数组
    expected_mask = np.array([[False, True], [True, False]])

    # 调用_object_dtype_isnan函数计算NaN掩码
    mask = _object_dtype_isnan(X)

    # 使用assert_array_equal函数检查计算得到的掩码数组是否与期望相同
    assert_array_equal(mask, expected_mask)


# 定义测试函数，参数化测试不同的params和expected_dtype组合
@pytest.mark.parametrize(
    "params, expected_dtype",
    [
        ({}, np.int32),  # 默认行为，预期返回np.int32
        ({"maxval": np.iinfo(np.int32).max}, np.int32),  # 设置maxval为np.int32的最大值
        ({"maxval": np.iinfo(np.int32).max + 1}, np.int64),  # 设置maxval超出np.int32的最大值
    ],
)
def test_smallest_admissible_index_dtype_max_val(params, expected_dtype):
    """检查`smallest_admissible_index_dtype`函数根据`max_val`参数的行为。"""
    # 调用_smallest_admissible_index_dtype函数，验证返回的dtype是否与期望相同
    assert _smallest_admissible_index_dtype(**params) == expected_dtype


# 定义测试函数，参数化测试不同的params和expected_dtype组合
@pytest.mark.parametrize(
    "params, expected_dtype",
    [
        # 数组的dtype为int64，因此不应该在不检查内容的情况下降级为int32
        ({"arrays": np.array([1, 2], dtype=np.int64)}, np.int64),
        # 其中一个数组为int64，基于同样的原因，不应该降级为int32
        (
            {
                "arrays": (
                    np.array([1, 2], dtype=np.int32),
                    np.array([1, 2], dtype=np.int64),
                )
            },
            np.int64,
        ),
        # 两个数组已经是int32：应该保持这种dtype
        (
            {
                "arrays": (
                    np.array([1, 2], dtype=np.int32),
                    np.array([1, 2], dtype=np.int32),
                )
            },
            np.int32,
        ),
        # 数组应该提升至至少int32精度
        ({"arrays": np.array([1, 2], dtype=np.int8)}, np.int32),
        # 验证`maxval`优先于数组，并因此提升至int64
        (
            {
                "arrays": np.array([1, 2], dtype=np.int32),
                "maxval": np.iinfo(np.int32).max + 1,
            },
            np.int64,
        ),
    ],
)
def test_smallest_admissible_index_dtype_without_checking_contents(
    params, expected_dtype
):
    """检查`smallest_admissible_index_dtype`函数使用传递的数组但不检查数组内容的行为。"""
    # 调用_smallest_admissible_index_dtype函数，验证返回的dtype是否与期望相同
    assert _smallest_admissible_index_dtype(**params) == expected_dtype
    [
        # 空数组应始终转换为 int32 索引
        (
            {
                "arrays": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
                "check_contents": True,
            },
            np.int32,
        ),
        # 对于满足 np.iinfo(np.int32).min < x < np.iinfo(np.int32).max 的数组应转换为 int32
        (
            {"arrays": np.array([1], dtype=np.int64), "check_contents": True},
            np.int32,
        ),
        # 否则，应将其转换为 int64。我们需要创建一个 uint32 数组以容纳大于 np.iinfo(np.int32).max 的值
        (
            {
                "arrays": np.array([np.iinfo(np.int32).max + 1], dtype=np.uint32),
                "check_contents": True,
            },
            np.int64,
        ),
        # maxval 应优先于数组内容，并因此升级为 int64
        (
            {
                "arrays": np.array([1], dtype=np.int32),
                "check_contents": True,
                "maxval": np.iinfo(np.int32).max + 1,
            },
            np.int64,
        ),
        # 当 maxval 较小时，但 check_contents 为 True 且内容需要 np.int64 时，最终仍需要 np.int64 索引
        (
            {
                "arrays": np.array([np.iinfo(np.int32).max + 1], dtype=np.uint32),
                "check_contents": True,
                "maxval": 1,
            },
            np.int64,
        ),
    ],
# 定义测试函数，用于验证 smallest_admissible_index_dtype 函数的行为是否符合预期
def test_smallest_admissible_index_dtype_by_checking_contents(params, expected_dtype):
    """Check the behaviour of `smallest_admissible_index_dtype` using the dtype of the
    arrays but as well the contents.
    """
    # 断言 smallest_admissible_index_dtype 函数返回的结果与期望的数据类型一致
    assert _smallest_admissible_index_dtype(**params) == expected_dtype


# 使用 pytest 的 parametrize 装饰器，对不同参数进行测试
@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        (
            {"maxval": np.iinfo(np.int64).max + 1},  # 参数设置为超过 np.int64 能表示的最大值
            ValueError,  # 预期会抛出 ValueError 异常
            "is to large to be represented as np.int64",  # 异常消息应包含此文本
        ),
        (
            {"arrays": np.array([1, 2], dtype=np.float64)},  # 参数设置为浮点数组
            ValueError,  # 预期会抛出 ValueError 异常
            "Array dtype float64 is not supported",  # 异常消息应包含此文本
        ),
        (
            {"arrays": [1, 2]},  # 参数设置为 Python 列表而不是 np.ndarray
            TypeError,  # 预期会抛出 TypeError 异常
            "Arrays should be of type np.ndarray",  # 异常消息应包含此文本
        ),
    ],
)
def test_smallest_admissible_index_dtype_error(params, err_type, err_msg):
    """Check that we raise the proper error message."""
    # 使用 pytest.raises 验证调用 _smallest_admissible_index_dtype 函数时是否会抛出特定类型的异常，并且异常消息符合预期
    with pytest.raises(err_type, match=err_msg):
        _smallest_admissible_index_dtype(**params)
```