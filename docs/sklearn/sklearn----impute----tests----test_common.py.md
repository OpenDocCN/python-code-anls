# `D:\src\scipysrc\scikit-learn\sklearn\impute\tests\test_common.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和矩阵
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer  # 导入三种数据缺失处理方法
from sklearn.utils._testing import (  # 导入测试工具函数
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入用于稀疏矩阵容器的修复工具


def imputers():
    return [IterativeImputer(tol=0.1), KNNImputer(), SimpleImputer()]  # 返回三种数据缺失处理方法的实例列表


def sparse_imputers():
    return [SimpleImputer()]  # 返回一个简单数据缺失处理方法的实例列表


# ConvergenceWarning will be raised by the IterativeImputer
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
def test_imputation_missing_value_in_test_array(imputer):
    # [Non Regression Test for issue #13968] Missing value in test set should
    # not throw an error and return a finite dataset
    train = [[1], [2]]
    test = [[3], [np.nan]]
    imputer.set_params(add_indicator=True)  # 设置参数，添加指示器来标记缺失值
    imputer.fit(train).transform(test)  # 在训练数据上拟合并转换测试数据


# ConvergenceWarning will be raised by the IterativeImputer
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("marker", [np.nan, -1, 0])
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
def test_imputers_add_indicator(marker, imputer):
    X = np.array(
        [
            [marker, 1, 5, marker, 1],
            [2, marker, 1, marker, 2],
            [6, 3, marker, marker, 3],
            [1, 2, 9, marker, 4],
        ]
    )
    X_true_indicator = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    imputer.set_params(missing_values=marker, add_indicator=True)  # 设置参数，标记缺失值并添加指示器

    X_trans = imputer.fit_transform(X)  # 拟合并转换数据
    assert_allclose(X_trans[:, -4:], X_true_indicator)  # 断言转换后的数据与预期的指示器数据接近
    assert_array_equal(imputer.indicator_.features_, np.array([0, 1, 2, 3]))  # 断言指示器的特征索引与预期相等

    imputer.set_params(add_indicator=False)  # 设置参数，不添加指示器
    X_trans_no_indicator = imputer.fit_transform(X)  # 再次拟合并转换数据
    assert_allclose(X_trans[:, :-4], X_trans_no_indicator)  # 断言转换后的数据与没有指示器的数据接近


# ConvergenceWarning will be raised by the IterativeImputer
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("marker", [np.nan, -1])
@pytest.mark.parametrize(
    "imputer", sparse_imputers(), ids=lambda x: x.__class__.__name__
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_imputers_add_indicator_sparse(imputer, marker, csr_container):
    X = csr_container(
        [
            [marker, 1, 5, marker, 1],
            [2, marker, 1, marker, 2],
            [6, 3, marker, marker, 3],
            [1, 2, 9, marker, 4],
        ]
    )
    X_true_indicator = csr_container(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # 设置缺失值替换器的参数：指定缺失值标记和添加指示器特征
    imputer.set_params(missing_values=marker, add_indicator=True)

    # 对输入数据 X 进行缺失值处理并转换，返回处理后的数据 X_trans
    X_trans = imputer.fit_transform(X)

    # 断言 X_trans 的最后四列与真实指示器矩阵 X_true_indicator 的值相近
    assert_allclose_dense_sparse(X_trans[:, -4:], X_true_indicator)

    # 断言缺失值替换器的指示器特征与预期的特征索引数组相等
    assert_array_equal(imputer.indicator_.features_, np.array([0, 1, 2, 3]))

    # 设置缺失值替换器的参数：不添加指示器特征
    imputer.set_params(add_indicator=False)

    # 对输入数据 X 进行缺失值处理并转换，返回处理后的数据 X_trans_no_indicator
    X_trans_no_indicator = imputer.fit_transform(X)

    # 断言 X_trans 的前面部分（不包括指示器列）与 X_trans_no_indicator 的值相近
    assert_allclose_dense_sparse(X_trans[:, :-4], X_trans_no_indicator)
# ConvergenceWarning将由IterativeImputer引发警告
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
# 参数化测试，使用imputers()生成的多个填充器
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
# 参数化测试，测试add_indicator参数为True和False的情况
@pytest.mark.parametrize("add_indicator", [True, False])
def test_imputers_pandas_na_integer_array_support(imputer, add_indicator):
    # 导入pytest前先检查是否有pandas模块，否则跳过测试
    pd = pytest.importorskip("pandas")
    # 使用np.nan作为标记值，设置填充器的参数
    marker = np.nan
    imputer = imputer.set_params(add_indicator=add_indicator, missing_values=marker)

    # 创建包含pd.NA的pandas IntegerArray的numpy数组
    X = np.array(
        [
            [marker, 1, 5, marker, 1],
            [2, marker, 1, marker, 2],
            [6, 3, marker, marker, 3],
            [1, 2, 9, marker, 4],
        ]
    )
    # 在numpy数组上进行拟合转换
    X_trans_expected = imputer.fit_transform(X)

    # 创建带有pd.NA的IntegerArrays的数据框
    X_df = pd.DataFrame(X, dtype="Int16", columns=["a", "b", "c", "d", "e"])

    # 在带有IntegerArrays的pandas数据框上进行拟合转换
    X_trans = imputer.fit_transform(X_df)

    # 断言转换后的结果近似相等
    assert_allclose(X_trans_expected, X_trans)


# 参数化测试，使用imputers()生成的多个填充器
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
# 参数化测试，测试add_indicator参数为True和False的情况
@pytest.mark.parametrize("add_indicator", [True, False])
def test_imputers_feature_names_out_pandas(imputer, add_indicator):
    """Check feature names out for imputers."""
    # 导入pytest前先检查是否有pandas模块，否则跳过测试
    pd = pytest.importorskip("pandas")
    # 使用np.nan作为标记值，设置填充器的参数
    marker = np.nan
    imputer = imputer.set_params(add_indicator=add_indicator, missing_values=marker)

    X = np.array(
        [
            [marker, 1, 5, 3, marker, 1],
            [2, marker, 1, 4, marker, 2],
            [6, 3, 7, marker, marker, 3],
            [1, 2, 9, 8, marker, 4],
        ]
    )
    # 创建具有自定义列名的数据框
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d", "e", "f"])
    # 在数据框上进行拟合
    imputer.fit(X_df)

    # 获取特征名列表
    names = imputer.get_feature_names_out()

    # 根据add_indicator参数进行断言，验证特征名输出是否符合预期
    if add_indicator:
        expected_names = [
            "a",
            "b",
            "c",
            "d",
            "f",
            "missingindicator_a",
            "missingindicator_b",
            "missingindicator_d",
            "missingindicator_e",
        ]
        assert_array_equal(expected_names, names)
    else:
        expected_names = ["a", "b", "c", "d", "f"]
        assert_array_equal(expected_names, names)


# 参数化测试，测试keep_empty_features参数为True和False的情况
@pytest.mark.parametrize("keep_empty_features", [True, False])
# 参数化测试，使用imputers()生成的多个填充器
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
def test_keep_empty_features(imputer, keep_empty_features):
    """Check that the imputer keeps features with only missing values."""
    # 创建包含NaN值的numpy数组
    X = np.array([[np.nan, 1], [np.nan, 2], [np.nan, 3]])
    # 设置填充器的参数，包括add_indicator=False和keep_empty_features参数
    imputer = imputer.set_params(
        add_indicator=False, keep_empty_features=keep_empty_features
    )

    # 循环调用fit_transform或transform方法，并验证结果形状是否符合预期
    for method in ["fit_transform", "transform"]:
        X_imputed = getattr(imputer, method)(X)
        if keep_empty_features:
            assert X_imputed.shape == X.shape
        else:
            assert X_imputed.shape == (X.shape[0], X.shape[1] - 1)
# 使用 pytest.mark.parametrize 装饰器，为每个 imputer 实例生成参数化的测试用例
# imputers() 返回一个生成 imputer 实例的迭代器
# ids=lambda x: x.__class__.__name__ 用于生成每个测试用例的标识符，显示 imputer 类名
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)

# 参数化测试用例，测试不同的缺失值情况，分别为 np.nan 和 1
@pytest.mark.parametrize("missing_value_test", [np.nan, 1])
def test_imputation_adds_missing_indicator_if_add_indicator_is_true(
    imputer, missing_value_test
):
    """Check that missing indicator always exists when add_indicator=True.

    Non-regression test for gh-26590.
    """
    # 创建训练数据集 X_train，包含两行样本和一个缺失值
    X_train = np.array([[0, np.nan], [1, 2]])

    # 创建测试数据集 X_test，其中 missing_value_test 可以是 np.nan 或 1
    X_test = np.array([[0, missing_value_test], [1, 2]])

    # 设置 imputer 参数 add_indicator=True，即启用缺失指示器
    imputer.set_params(add_indicator=True)
    # 在训练数据集上拟合 imputer 模型
    imputer.fit(X_train)

    # 对测试数据集 X_test 进行缺失值填充，并添加缺失指示器
    X_test_imputed_with_indicator = imputer.transform(X_test)
    # 断言填充后的数据形状为 (2, 3)
    assert X_test_imputed_with_indicator.shape == (2, 3)

    # 设置 imputer 参数 add_indicator=False，即不使用缺失指示器
    imputer.set_params(add_indicator=False)
    # 在训练数据集上重新拟合 imputer 模型
    imputer.fit(X_train)
    # 对测试数据集 X_test 进行缺失值填充，不添加缺失指示器
    X_test_imputed_without_indicator = imputer.transform(X_test)
    # 断言填充后的数据形状为 (2, 2)
    assert X_test_imputed_without_indicator.shape == (2, 2)

    # 断言填充后的数据（去除最后一列）与不添加缺失指示器的填充结果一致
    assert_allclose(
        X_test_imputed_with_indicator[:, :-1], X_test_imputed_without_indicator
    )

    # 根据 missing_value_test 的值确定预期的缺失指示器结果
    if np.isnan(missing_value_test):
        expected_missing_indicator = [1, 0]
    else:
        expected_missing_indicator = [0, 0]

    # 断言填充后的数据的最后一列（缺失指示器列）与预期的缺失指示器结果一致
    assert_allclose(X_test_imputed_with_indicator[:, -1], expected_missing_indicator)
```