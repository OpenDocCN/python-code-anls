# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\tests\test_chi2.py`

```
"""
Tests for chi2, currently the only feature selection function designed
specifically to work with sparse matrices.
"""

import warnings  # 导入警告模块，用于处理警告信息

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 PyTest 库，用于编写和运行测试用例
import scipy.stats  # 导入 SciPy 统计模块

from sklearn.feature_selection import SelectKBest, chi2  # 从 scikit-learn 中导入特征选择相关函数
from sklearn.feature_selection._univariate_selection import _chisquare  # 导入卡方检验函数
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal  # 导入用于测试的数组比较函数
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS  # 导入用于稀疏矩阵容器的修复函数

# Feature 0 is highly informative for class 1;
# feature 1 is the same everywhere;
# feature 2 is a bit informative for class 2.
X = [[2, 1, 2], [9, 1, 1], [6, 1, 2], [0, 1, 2]]  # 创建一个特征矩阵 X，包含四个样本和三个特征
y = [0, 1, 2, 2]  # 创建目标变量 y，包含四个类标签


def mkchi2(k):
    """Make k-best chi2 selector"""
    return SelectKBest(chi2, k=k)  # 创建并返回一个 SelectKBest 对象，使用卡方检验选择 k 个最佳特征


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_chi2(csr_container):
    # Test Chi2 feature extraction
    chi2 = mkchi2(k=1).fit(X, y)  # 使用 k=1 创建卡方检验对象，拟合数据 X 和目标 y
    chi2 = mkchi2(k=1).fit(X, y)  # 使用 k=1 创建卡方检验对象，拟合数据 X 和目标 y
    assert_array_equal(chi2.get_support(indices=True), [0])  # 检查返回的最佳特征索引是否为 [0]
    assert_array_equal(chi2.transform(X), np.array(X)[:, [0]])  # 检查转换后的数据是否仅包含第一个特征列

    chi2 = mkchi2(k=2).fit(X, y)  # 使用 k=2 创建卡方检验对象，拟合数据 X 和目标 y
    assert_array_equal(sorted(chi2.get_support(indices=True)), [0, 2])  # 检查返回的最佳特征索引是否为 [0, 2]

    Xsp = csr_container(X, dtype=np.float64)  # 将原始数据 X 转换为指定稀疏矩阵格式
    chi2 = mkchi2(k=2).fit(Xsp, y)  # 使用 k=2 创建卡方检验对象，拟合稀疏数据 Xsp 和目标 y
    assert_array_equal(sorted(chi2.get_support(indices=True)), [0, 2])  # 检查返回的最佳特征索引是否为 [0, 2]
    Xtrans = chi2.transform(Xsp)  # 对稀疏数据进行转换
    assert_array_equal(Xtrans.shape, [Xsp.shape[0], 2])  # 检查转换后的数据形状是否为 [样本数, 2]

    # == doesn't work on scipy.sparse matrices
    Xtrans = Xtrans.toarray()  # 将稀疏矩阵转换为稠密数组
    Xtrans2 = mkchi2(k=2).fit_transform(Xsp, y).toarray()  # 使用 fit_transform 直接得到稠密数组
    assert_array_almost_equal(Xtrans, Xtrans2)  # 检查两个稠密数组是否近似相等


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_chi2_coo(coo_container):
    # Check that chi2 works with a COO matrix
    # (as returned by CountVectorizer, DictVectorizer)
    Xcoo = coo_container(X)  # 将原始数据 X 转换为 COO 格式的稀疏矩阵
    mkchi2(k=2).fit_transform(Xcoo, y)  # 使用 COO 格式的数据进行卡方检验转换
    # if we got here without an exception, we're safe


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_chi2_negative(csr_container):
    # Check for proper error on negative numbers in the input X.
    X, y = [[0, 1], [-1e-20, 1]], [0, 1]  # 创建包含负数的输入数据 X 和目标 y
    for X in (X, np.array(X), csr_container(X)):  # 对不同格式的 X 进行循环检查
        with pytest.raises(ValueError):  # 检查是否引发值错误异常
            chi2(X, y)


def test_chi2_unused_feature():
    # Unused feature should evaluate to NaN
    # and should issue no runtime warning
    with warnings.catch_warnings(record=True) as warned:  # 捕获警告信息
        warnings.simplefilter("always")  # 设置警告过滤器，始终捕获警告
        chi, p = chi2([[1, 0], [0, 0]], [1, 0])  # 执行卡方检验
        for w in warned:  # 遍历捕获的警告信息列表
            if "divide by zero" in repr(w):  # 如果警告信息中包含 "divide by zero"
                raise AssertionError("Found unexpected warning %s" % w)  # 抛出断言错误，表示发现了意外的警告
    assert_array_equal(chi, [1, np.nan])  # 检查卡方检验结果是否包含 NaN
    assert_array_equal(p[1], np.nan)  # 检查 p 值结果是否包含 NaN


def test_chisquare():
    # Test replacement for scipy.stats.chisquare against the original.
    obs = np.array([[2.0, 2.0], [1.0, 1.0]])  # 创建观测数据
    exp = np.array([[1.5, 1.5], [1.5, 1.5]])  # 创建期望数据
    # call SciPy first because our version overwrites obs
    # 使用 scipy 库中的 chisquare 函数计算观察值和期望值的卡方统计量和 p 值
    chi_scp, p_scp = scipy.stats.chisquare(obs, exp)
    # 调用自定义函数 _chisquare 计算相同的卡方统计量和 p 值
    chi_our, p_our = _chisquare(obs, exp)

    # 使用 numpy.testing.assert_array_almost_equal 函数断言两个卡方统计量的数组近似相等
    assert_array_almost_equal(chi_scp, chi_our)
    # 使用 numpy.testing.assert_array_almost_equal 函数断言两个 p 值的数组近似相等
    assert_array_almost_equal(p_scp, p_our)
```