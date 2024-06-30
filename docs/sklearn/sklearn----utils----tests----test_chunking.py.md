# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_chunking.py`

```
# 导入警告模块
import warnings
# 导入 itertools 中的 chain 函数
from itertools import chain

# 导入 pytest 模块
import pytest

# 导入 sklearn 的配置上下文
from sklearn import config_context
# 导入 sklearn.utils._chunking 模块中的 gen_even_slices 和 get_chunk_n_rows 函数
from sklearn.utils._chunking import gen_even_slices, get_chunk_n_rows
# 导入 sklearn.utils._testing 模块中的 assert_array_equal 函数
from sklearn.utils._testing import assert_array_equal


# 定义测试函数 test_gen_even_slices
def test_gen_even_slices():
    # 检查 gen_even_slices 函数生成的切片是否包含所有样本
    some_range = range(10)
    # 将 gen_even_slices 函数生成的切片链表化，以验证是否包含所有样本
    joined_range = list(chain(*[some_range[slice] for slice in gen_even_slices(10, 3)]))
    assert_array_equal(some_range, joined_range)


# 使用 pytest 的参数化装饰器，定义测试函数 test_get_chunk_n_rows，传入多组参数进行测试
@pytest.mark.parametrize(
    ("row_bytes", "max_n_rows", "working_memory", "expected"),
    [
        (1024, None, 1, 1024),
        (1024, None, 0.99999999, 1023),
        (1023, None, 1, 1025),
        (1025, None, 1, 1023),
        (1024, None, 2, 2048),
        (1024, 7, 1, 7),
        (1024 * 1024, None, 1, 1),
    ],
)
def test_get_chunk_n_rows(row_bytes, max_n_rows, working_memory, expected):
    # 在测试中捕获警告，过滤 UserWarning 类型的警告
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # 调用 get_chunk_n_rows 函数，获取实际结果
        actual = get_chunk_n_rows(
            row_bytes=row_bytes,
            max_n_rows=max_n_rows,
            working_memory=working_memory,
        )

    # 断言实际结果与预期结果相等
    assert actual == expected
    # 断言实际结果的类型与预期结果的类型相同
    assert type(actual) is type(expected)

    # 在配置上下文中设置工作内存参数，再次测试
    with config_context(working_memory=working_memory):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            actual = get_chunk_n_rows(row_bytes=row_bytes, max_n_rows=max_n_rows)
        # 断言实际结果与预期结果相等
        assert actual == expected
        # 断言实际结果的类型与预期结果的类型相同
        assert type(actual) is type(expected)


# 定义测试函数 test_get_chunk_n_rows_warns，验证在工作内存过低时是否会发出警告
def test_get_chunk_n_rows_warns():
    # 设置测试参数
    row_bytes = 1024 * 1024 + 1
    max_n_rows = None
    working_memory = 1
    expected = 1

    # 预期的警告信息
    warn_msg = (
        "Could not adhere to working_memory config. Currently 1MiB, 2MiB required."
    )
    # 使用 pytest 的 warn 函数检查是否会发出 UserWarning 类型的警告，并匹配预期的警告信息
    with pytest.warns(UserWarning, match=warn_msg):
        # 调用 get_chunk_n_rows 函数，获取实际结果
        actual = get_chunk_n_rows(
            row_bytes=row_bytes,
            max_n_rows=max_n_rows,
            working_memory=working_memory,
        )

    # 断言实际结果与预期结果相等
    assert actual == expected
    # 断言实际结果的类型与预期结果的类型相同
    assert type(actual) is type(expected)

    # 在配置上下文中设置工作内存参数，再次测试
    with config_context(working_memory=working_memory):
        with pytest.warns(UserWarning, match=warn_msg):
            actual = get_chunk_n_rows(row_bytes=row_bytes, max_n_rows=max_n_rows)
        # 断言实际结果与预期结果相等
        assert actual == expected
        # 断言实际结果的类型与预期结果的类型相同
        assert type(actual) is type(expected)
```