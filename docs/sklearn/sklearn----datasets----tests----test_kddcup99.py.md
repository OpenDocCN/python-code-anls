# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_kddcup99.py`

```
"""
Test  kddcup99 loader, if the data is available,
or if specifically requested via environment variable
(e.g. for CI jobs).

Only 'percent10' mode is tested, as the full data
is too big to use in unit-testing.
"""

# 导入所需模块和函数
from functools import partial
import pytest
from sklearn.datasets.tests.test_common import (
    check_as_frame,
    check_pandas_dependency_message,
    check_return_X_y,
)

# 使用 pytest 装饰器标记参数化测试
@pytest.mark.parametrize("as_frame", [True, False])
@pytest.mark.parametrize(
    "subset, n_samples, n_features",
    [
        (None, 494021, 41),
        ("SA", 100655, 41),
        ("SF", 73237, 4),
        ("http", 58725, 3),
        ("smtp", 9571, 3),
    ],
)
# 测试函数：测试 fetch_kddcup99_fxt 函数在 percent10 模式下的行为
def test_fetch_kddcup99_percent10(
    fetch_kddcup99_fxt, as_frame, subset, n_samples, n_features
):
    # 调用 fetch_kddcup99_fxt 函数获取数据
    data = fetch_kddcup99_fxt(subset=subset, as_frame=as_frame)
    # 断言数据形状符合预期
    assert data.data.shape == (n_samples, n_features)
    assert data.target.shape == (n_samples,)
    # 如果 as_frame 为 True，则断言返回的数据帧形状也符合预期
    if as_frame:
        assert data.frame.shape == (n_samples, n_features + 1)
    # 断言数据描述信息以指定字符串开头
    assert data.DESCR.startswith(".. _kddcup99_dataset:")

# 测试函数：测试 fetch_kddcup99_fxt 函数返回 X 和 y 数据
def test_fetch_kddcup99_return_X_y(fetch_kddcup99_fxt):
    fetch_func = partial(fetch_kddcup99_fxt, subset="smtp")
    data = fetch_func()
    check_return_X_y(data, fetch_func)

# 测试函数：测试 fetch_kddcup99_fxt 函数返回数据集并转换为数据帧
def test_fetch_kddcup99_as_frame(fetch_kddcup99_fxt):
    bunch = fetch_kddcup99_fxt()
    check_as_frame(bunch, fetch_kddcup99_fxt)

# 测试函数：测试 fetch_kddcup99_fxt 函数在 shuffle=True 情况下的行为
def test_fetch_kddcup99_shuffle(fetch_kddcup99_fxt):
    dataset = fetch_kddcup99_fxt(
        random_state=0,
        subset="SA",
        percent10=True,
    )
    dataset_shuffled = fetch_kddcup99_fxt(
        random_state=0,
        subset="SA",
        shuffle=True,
        percent10=True,
    )
    # 断言洗牌后的目标值集合与未洗牌时一致
    assert set(dataset["target"]) == set(dataset_shuffled["target"])
    # 断言洗牌后的数据形状与未洗牌时一致
    assert dataset_shuffled.data.shape == dataset.data.shape
    assert dataset_shuffled.target.shape == dataset.target.shape

# 测试函数：检查当缺少 pandas 时的提示信息
def test_pandas_dependency_message(fetch_kddcup99_fxt, hide_available_pandas):
    check_pandas_dependency_message(fetch_kddcup99_fxt)

# 测试函数：检查当缓存文件损坏时是否能正确报错
def test_corrupted_file_error_message(fetch_kddcup99_fxt, tmp_path):
    kddcup99_dir = tmp_path / "kddcup99_10-py3"
    kddcup99_dir.mkdir()
    samples_path = kddcup99_dir / "samples"

    # 写入一个损坏的缓存文件
    with samples_path.open("wb") as f:
        f.write(b"THIS IS CORRUPTED")

    # 预期的报错信息
    msg = (
        "The cache for fetch_kddcup99 is invalid, please "
        f"delete {str(kddcup99_dir)} and run the fetch_kddcup99 again"
    )

    # 断言在调用 fetch_kddcup99_fxt 函数时能够抛出预期的 OSError 异常，并匹配预期的错误信息
    with pytest.raises(OSError, match=msg):
        fetch_kddcup99_fxt(data_home=str(tmp_path))
```