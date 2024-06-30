# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_20news.py`

```
"""Test the 20news downloader, if the data is available,
or if specifically requested via environment variable
(e.g. for CI jobs)."""

# 导入所需的库和模块
from functools import partial  # 导入偏函数的功能
from unittest.mock import patch  # 导入用于模拟的 patch 函数

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架
import scipy.sparse as sp  # 导入 SciPy 稀疏矩阵模块

from sklearn.datasets.tests.test_common import (
    check_as_frame,  # 导入用于检查转换为 DataFrame 的函数
    check_pandas_dependency_message,  # 导入用于检查 Pandas 依赖的函数
    check_return_X_y,  # 导入用于检查返回 X 和 y 的函数
)
from sklearn.preprocessing import normalize  # 导入数据归一化函数
from sklearn.utils._testing import assert_allclose_dense_sparse  # 导入用于比较密集和稀疏矩阵的函数


def test_20news(fetch_20newsgroups_fxt):
    # 调用 fetch_20newsgroups_fxt 函数获取数据集
    data = fetch_20newsgroups_fxt(subset="all", shuffle=False)
    # 断言数据集的描述信息以特定字符串开头
    assert data.DESCR.startswith(".. _20newsgroups_dataset:")

    # 提取一个减少后的数据集
    data2cats = fetch_20newsgroups_fxt(
        subset="all", categories=data.target_names[-1:-3:-1], shuffle=False
    )
    # 检查目标名称的顺序与完整数据集中的顺序是否相同
    assert data2cats.target_names == data.target_names[-2:]
    # 断言目标变量只包含0和1两个值
    assert np.unique(data2cats.target).tolist() == [0, 1]

    # 检查文件名数量与数据/目标变量数量一致
    assert len(data2cats.filenames) == len(data2cats.target)
    assert len(data2cats.filenames) == len(data2cats.data)

    # 检查减少后数据集的第一个条目与完整数据集相应类别的第一个条目是否对应
    entry1 = data2cats.data[0]
    category = data2cats.target_names[data2cats.target[0]]
    label = data.target_names.index(category)
    entry2 = data.data[np.where(data.target == label)[0][0]]
    assert entry1 == entry2

    # 检查返回 X_y 选项
    X, y = fetch_20newsgroups_fxt(subset="all", shuffle=False, return_X_y=True)
    assert len(X) == len(data.data)
    assert y.shape == data.target.shape


def test_20news_length_consistency(fetch_20newsgroups_fxt):
    """Checks the length consistencies within the bunch

    This is a non-regression test for a bug present in 0.16.1.
    """
    # 提取完整数据集
    data = fetch_20newsgroups_fxt(subset="all")
    # 断言数据和 'data' 键对应的长度一致
    assert len(data["data"]) == len(data.data)
    # 断言目标和 'target' 键对应的长度一致
    assert len(data["target"]) == len(data.target)
    # 断言文件名和 'filenames' 键对应的长度一致
    assert len(data["filenames"]) == len(data.filenames)


def test_20news_vectorized(fetch_20newsgroups_vectorized_fxt):
    # 测试 subset = train 的情况
    bunch = fetch_20newsgroups_vectorized_fxt(subset="train")
    # 断言数据是稀疏矩阵且格式为 'csr'
    assert sp.issparse(bunch.data) and bunch.data.format == "csr"
    # 断言数据的形状符合预期
    assert bunch.data.shape == (11314, 130107)
    # 断言目标变量的数量符合预期
    assert bunch.target.shape[0] == 11314
    # 断言数据类型为 np.float64
    assert bunch.data.dtype == np.float64
    # 断言数据集的描述信息以特定字符串开头
    assert bunch.DESCR.startswith(".. _20newsgroups_dataset:")

    # 测试 subset = test 的情况
    bunch = fetch_20newsgroups_vectorized_fxt(subset="test")
    # 断言数据是稀疏矩阵且格式为 'csr'
    assert sp.issparse(bunch.data) and bunch.data.format == "csr"
    # 断言数据的形状符合预期
    assert bunch.data.shape == (7532, 130107)
    # 断言目标变量的数量符合预期
    assert bunch.target.shape[0] == 7532
    # 断言数据类型为 np.float64
    assert bunch.data.dtype == np.float64
    # 断言数据集的描述字符串以指定的开头字符串开始
    assert bunch.DESCR.startswith(".. _20newsgroups_dataset:")

    # 测试 return_X_y 选项
    fetch_func = partial(fetch_20newsgroups_vectorized_fxt, subset="test")
    # 检查 fetch_func 返回的数据是否符合 return_X_y 的要求
    check_return_X_y(bunch, fetch_func)

    # 测试 subset = all
    # 获取完整数据集
    bunch = fetch_20newsgroups_vectorized_fxt(subset="all")
    # 断言数据集的数据部分为稀疏矩阵且格式为 csr
    assert sp.issparse(bunch.data) and bunch.data.format == "csr"
    # 断言数据集的数据部分形状正确
    assert bunch.data.shape == (11314 + 7532, 130107)
    # 断言目标标签数组的长度正确
    assert bunch.target.shape[0] == 11314 + 7532
    # 断言数据集的数据类型为 np.float64
    assert bunch.data.dtype == np.float64
    # 断言数据集的描述字符串以指定的开头字符串开始
    assert bunch.DESCR.startswith(".. _20newsgroups_dataset:")
# 测试函数：对未归一化和归一化后的20个新闻数据集进行规范化测试
def test_20news_normalization(fetch_20newsgroups_vectorized_fxt):
    # 获取未归一化的数据集
    X = fetch_20newsgroups_vectorized_fxt(normalize=False)
    # 获取归一化后的数据集
    X_ = fetch_20newsgroups_vectorized_fxt(normalize=True)
    # 获取归一化后数据集的前100个样本
    X_norm = X_["data"][:100]
    # 获取未归一化数据集的前100个样本
    X = X["data"][:100]

    # 断言两个稀疏矩阵的内容接近（使用自定义函数 assert_allclose_dense_sparse）
    assert_allclose_dense_sparse(X_norm, normalize(X))
    # 断言归一化后的矩阵每行的L2范数接近于1
    assert np.allclose(np.linalg.norm(X_norm.todense(), axis=1), 1)


# 测试函数：测试将20个新闻数据集作为 Pandas DataFrame 的情况
def test_20news_as_frame(fetch_20newsgroups_vectorized_fxt):
    # 导入 pandas 库，如果不存在则跳过该测试
    pd = pytest.importorskip("pandas")

    # 获取数据集并确保能够作为 DataFrame 使用
    bunch = fetch_20newsgroups_vectorized_fxt(as_frame=True)
    check_as_frame(bunch, fetch_20newsgroups_vectorized_fxt)

    # 获取 DataFrame
    frame = bunch.frame
    # 断言 DataFrame 的形状为 (11314, 130108)
    assert frame.shape == (11314, 130108)
    # 断言数据集的所有列都是稀疏数据类型 SparseDtype
    assert all([isinstance(col, pd.SparseDtype) for col in bunch.data.dtypes])

    # 检查一小部分特征是否在 DataFrame 的列中
    for expected_feature in [
        "beginner",
        "beginners",
        "beginning",
        "beginnings",
        "begins",
        "begley",
        "begone",
    ]:
        assert expected_feature in frame.keys()
    # 断言 "category_class" 列存在于 DataFrame 中
    assert "category_class" in frame.keys()
    # 断言目标变量的名称为 "category_class"
    assert bunch.target.name == "category_class"


# 测试函数：当没有安装 Pandas 时的处理情况
def test_as_frame_no_pandas(fetch_20newsgroups_vectorized_fxt, hide_available_pandas):
    # 检查在没有安装 Pandas 的情况下的依赖消息处理
    check_pandas_dependency_message(fetch_20newsgroups_vectorized_fxt)


# 测试函数：测试使用过时的 pickle 数据集的情况
def test_outdated_pickle(fetch_20newsgroups_vectorized_fxt):
    with patch("os.path.exists") as mock_is_exist:
        with patch("joblib.load") as mock_load:
            # 模拟数据集已被缓存
            mock_is_exist.return_value = True
            # 模拟只返回了过时的 pickle 数据集中的 X 和 y
            mock_load.return_value = ("X", "y")
            err_msg = "The cached dataset located in"
            # 使用 pytest 断言期望抛出 ValueError，并匹配给定的错误消息
            with pytest.raises(ValueError, match=err_msg):
                fetch_20newsgroups_vectorized_fxt(as_frame=True)
```