# `D:\src\scipysrc\scikit-learn\sklearn\compose\tests\test_column_transformer.py`

```
"""
Test the ColumnTransformer.
"""

# 导入所需的库和模块
import pickle  # 用于序列化和反序列化 Python 对象
import re  # 正则表达式库
import warnings  # 警告处理库
from unittest.mock import Mock  # 创建模拟对象

import joblib  # 用于序列化和反序列化 Python 对象
import numpy as np  # 处理数值计算的核心库
import pytest  # Python 的单元测试框架
from numpy.testing import assert_allclose  # 用于比较数组是否接近的函数
from scipy import sparse  # 稀疏矩阵处理库

from sklearn.base import BaseEstimator, TransformerMixin  # Scikit-Learn 基类
from sklearn.compose import (  # 数据预处理和特征工程组合器
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.compose._column_transformer import _RemainderColsList  # 内部列转换器
from sklearn.exceptions import NotFittedError  # 未拟合异常
from sklearn.feature_selection import VarianceThreshold  # 方差阈值特征选择器
from sklearn.preprocessing import (  # 数据预处理器
    FunctionTransformer,
    Normalizer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tests.metadata_routing_common import (  # 元数据相关测试工具
    ConsumingTransformer,
    _Registry,
    check_recorded_metadata,
)
from sklearn.utils._indexing import _safe_indexing  # 安全索引访问工具
from sklearn.utils._testing import (  # Scikit-Learn 测试工具
    _convert_container,
    assert_allclose_dense_sparse,
    assert_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS, parse_version  # 兼容性修复工具


class Trans(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self  # 返回自身

    def transform(self, X, y=None):
        # 如果 X 是 1 维 Series，则转换成 2 维 DataFrame
        if hasattr(X, "to_frame"):
            return X.to_frame()
        # 如果 X 是 1 维数组，则将其转换成至少是 2 维的数组
        if getattr(X, "ndim", 2) == 1:
            return np.atleast_2d(X).T
        return X  # 否则返回原始 X


class DoubleTrans(BaseEstimator):
    def fit(self, X, y=None):
        return self  # 返回自身

    def transform(self, X):
        return 2 * X  # 返回输入的两倍


class SparseMatrixTrans(BaseEstimator):
    def __init__(self, csr_container):
        self.csr_container = csr_container  # 初始化稀疏矩阵容器类型

    def fit(self, X, y=None):
        return self  # 返回自身

    def transform(self, X, y=None):
        n_samples = len(X)  # 获取样本数
        return self.csr_container(sparse.eye(n_samples, n_samples))  # 返回稀疏单位矩阵


class TransNo2D(BaseEstimator):
    def fit(self, X, y=None):
        return self  # 返回自身

    def transform(self, X, y=None):
        return X  # 返回输入的 X


class TransRaise(BaseEstimator):
    def fit(self, X, y=None):
        raise ValueError("specific message")  # 抛出值错误异常

    def transform(self, X, y=None):
        raise ValueError("specific message")  # 抛出值错误异常


def test_column_transformer():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T  # 创建一个二维数组

    X_res_first1D = np.array([0, 1, 2])  # 第一列的一维数组
    X_res_second1D = np.array([2, 4, 6])  # 第二列的一维数组
    X_res_first = X_res_first1D.reshape(-1, 1)  # 第一列的二维数组
    X_res_both = X_array  # 原始二维数组

    cases = [
        # 单列 1 维 / 2 维
        (0, X_res_first),
        ([0], X_res_first),
        # 列表
        ([0, 1], X_res_both),
        (np.array([0, 1]), X_res_both),
        # 切片
        (slice(0, 1), X_res_first),
        (slice(0, 2), X_res_both),
        # 布尔掩码
        (np.array([True, False]), X_res_first),
        ([True, False], X_res_first),
        (np.array([True, True]), X_res_both),
        ([True, True], X_res_both),
    ]
    # 对每个测试用例进行迭代，其中 selection 是选择的列索引或名称，res 是预期的转换结果
    for selection, res in cases:
        # 创建 ColumnTransformer 对象，将指定列应用 Trans() 转换，其余列被丢弃
        ct = ColumnTransformer([("trans", Trans(), selection)], remainder="drop")
        # 断言使用 fit_transform() 方法得到的转换结果与预期结果相等
        assert_array_equal(ct.fit_transform(X_array), res)
        # 断言使用 fit() 和 transform() 方法得到的转换结果与预期结果相等
        assert_array_equal(ct.fit(X_array).transform(X_array), res)

        # 创建 ColumnTransformer 对象，使用 lambda 函数将指定的选择转换为 Trans() 转换
        ct = ColumnTransformer(
            [("trans", Trans(), lambda x: selection)], remainder="drop"
        )
        # 断言使用 fit_transform() 方法得到的转换结果与预期结果相等
        assert_array_equal(ct.fit_transform(X_array), res)
        # 断言使用 fit() 和 transform() 方法得到的转换结果与预期结果相等
        assert_array_equal(ct.fit(X_array).transform(X_array), res)

    # 创建 ColumnTransformer 对象，对多个列应用 Trans() 转换
    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", Trans(), [1])])
    # 断言使用 fit_transform() 方法得到的转换结果与预期的 X_res_both 相等
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    # 断言使用 fit() 和 transform() 方法得到的转换结果与预期的 X_res_both 相等
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    # 断言 ColumnTransformer 内部 transformers_ 属性的长度为 2
    assert len(ct.transformers_) == 2

    # 使用 transformer_weights 测试 ColumnTransformer
    transformer_weights = {"trans1": 0.1, "trans2": 10}
    both = ColumnTransformer(
        [("trans1", Trans(), [0]), ("trans2", Trans(), [1])],
        transformer_weights=transformer_weights,
    )
    # 计算加权后的预期结果 res
    res = np.vstack(
        [
            transformer_weights["trans1"] * X_res_first1D,
            transformer_weights["trans2"] * X_res_second1D,
        ]
    ).T
    # 断言使用 fit_transform() 方法得到的转换结果与加权后的预期结果 res 相等
    assert_array_equal(both.fit_transform(X_array), res)
    # 断言使用 fit() 和 transform() 方法得到的转换结果与加权后的预期结果 res 相等
    assert_array_equal(both.fit(X_array).transform(X_array), res)
    # 断言 ColumnTransformer 内部 transformers_ 属性的长度为 2
    assert len(both.transformers_) == 2

    # 创建仅对一组列应用 Trans() 转换的 ColumnTransformer 对象，使用指定的 transformer_weights
    both = ColumnTransformer(
        [("trans", Trans(), [0, 1])], transformer_weights={"trans": 0.1}
    )
    # 断言使用 fit_transform() 方法得到的转换结果与加权后的预期结果 0.1 * X_res_both 相等
    assert_array_equal(both.fit_transform(X_array), 0.1 * X_res_both)
    # 断言使用 fit() 和 transform() 方法得到的转换结果与加权后的预期结果 0.1 * X_res_both 相等
    assert_array_equal(both.fit(X_array).transform(X_array), 0.1 * X_res_both)
    # 断言 ColumnTransformer 内部 transformers_ 属性的长度为 1
    assert len(both.transformers_) == 1
def test_column_transformer_tuple_transformers_parameter():
    # 创建一个包含两列的 NumPy 数组
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    # 定义转换器列表，每个元素包含转换器的名称、转换器对象和要转换的列索引
    transformers = [("trans1", Trans(), [0]), ("trans2", Trans(), [1])]

    # 使用转换器列表创建 ColumnTransformer 对象，两种方式：一种是直接传入列表，另一种是传入元组
    ct_with_list = ColumnTransformer(transformers)
    ct_with_tuple = ColumnTransformer(tuple(transformers))

    # 断言两种方式转换后的结果是相等的
    assert_array_equal(
        ct_with_list.fit_transform(X_array), ct_with_tuple.fit_transform(X_array)
    )
    assert_array_equal(
        ct_with_list.fit(X_array).transform(X_array),
        ct_with_tuple.fit(X_array).transform(X_array),
    )


@pytest.mark.parametrize("constructor_name", ["dataframe", "polars"])
def test_column_transformer_dataframe(constructor_name):
    # 根据 constructor_name 导入相应的数据处理库，并进行必要的依赖检查
    if constructor_name == "dataframe":
        dataframe_lib = pytest.importorskip("pandas")
    else:
        dataframe_lib = pytest.importorskip(constructor_name)

    # 创建一个包含两列的 NumPy 数组和对应的 DataFrame
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = _convert_container(
        X_array, constructor_name, columns_name=["first", "second"]
    )

    # 预期的转换结果，用于后续的断言
    X_res_first = np.array([0, 1, 2]).reshape(-1, 1)
    X_res_both = X_array

    # 不同选择方式及其预期结果的组合，用于测试 ColumnTransformer
    cases = [
        # 字符串键：基于标签
        # 列表
        (["first"], X_res_first),
        (["first", "second"], X_res_both),
        # 切片
        (slice("first", "second"), X_res_both),
        # 整数键：基于位置
        # 列表
        ([0], X_res_first),
        ([0, 1], X_res_both),
        (np.array([0, 1]), X_res_both),
        # 切片
        (slice(0, 1), X_res_first),
        (slice(0, 2), X_res_both),
        # 布尔掩码
        (np.array([True, False]), X_res_first),
        ([True, False], X_res_first),
    ]
    if constructor_name == "dataframe":
        # 对于 pandas 数据帧，只支持标量值
        cases.extend(
            [
                # 标量值
                (0, X_res_first),
                ("first", X_res_first),
                (
                    dataframe_lib.Series([True, False], index=["first", "second"]),
                    X_res_first,
                ),
            ]
        )

    # 遍历每个测试用例，使用不同的选择方式测试 ColumnTransformer
    for selection, res in cases:
        # 创建 ColumnTransformer 对象，指定转换器、选择方式和处理方式
        ct = ColumnTransformer([("trans", Trans(), selection)], remainder="drop")
        # 断言转换后的结果与预期一致
        assert_array_equal(ct.fit_transform(X_df), res)
        assert_array_equal(ct.fit(X_df).transform(X_df), res)

        # 使用可调用对象作为选择方式，返回允许的任何选择方式
        ct = ColumnTransformer(
            [("trans", Trans(), lambda X: selection)], remainder="drop"
        )
        # 断言转换后的结果与预期一致
        assert_array_equal(ct.fit_transform(X_df), res)
        assert_array_equal(ct.fit(X_df).transform(X_df), res)

    # 创建包含两个转换器的 ColumnTransformer 对象
    ct = ColumnTransformer(
        [("trans1", Trans(), ["first"]), ("trans2", Trans(), ["second"])]
    )
    # 断言转换后的结果与预期一致
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
    # 断言转换器列表长度为 2，且最后一个转换器不是 "remainder"
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] != "remainder"

    # 创建包含两列的 ColumnTransformer 对象，指定每列的转换器和处理方式
    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", Trans(), [1])])
    # 使用 ColumnTransformer 对 X_df 进行拟合和转换，比较结果是否与 X_res_both 相等
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    
    # 使用 ColumnTransformer 先拟合再转换 X_df，比较结果是否与 X_res_both 相等
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
    
    # 检查 ColumnTransformer 的 transformers_ 属性长度是否为 2
    assert len(ct.transformers_) == 2
    
    # 检查 ColumnTransformer 的 transformers_ 属性最后一个元素的第一个元素是否不为 "remainder"
    assert ct.transformers_[-1][0] != "remainder"

    # 使用 transformer_weights 测试 ColumnTransformer 的行为
    transformer_weights = {"trans1": 0.1, "trans2": 10}
    both = ColumnTransformer(
        [("trans1", Trans(), ["first"]), ("trans2", Trans(), ["second"])],
        transformer_weights=transformer_weights,
    )
    # 计算期望的结果 res
    res = np.vstack(
        [
            transformer_weights["trans1"] * X_df["first"],
            transformer_weights["trans2"] * X_df["second"],
        ]
    ).T
    # 比较 ColumnTransformer 拟合并转换 X_df 后的结果是否与期望的 res 相等
    assert_array_equal(both.fit_transform(X_df), res)
    
    # 使用 ColumnTransformer 先拟合再转换 X_df，比较结果是否与期望的 res 相等
    assert_array_equal(both.fit(X_df).transform(X_df), res)
    
    # 检查 ColumnTransformer 的 transformers_ 属性长度是否为 2
    assert len(both.transformers_) == 2
    
    # 检查 ColumnTransformer 的 transformers_ 属性最后一个元素的第一个元素是否不为 "remainder"
    assert both.transformers_[-1][0] != "remainder"

    # 测试多列的情况
    both = ColumnTransformer(
        [("trans", Trans(), ["first", "second"])], transformer_weights={"trans": 0.1}
    )
    # 比较 ColumnTransformer 拟合并转换 X_df 后的结果是否与期望的 0.1 * X_res_both 相等
    assert_array_equal(both.fit_transform(X_df), 0.1 * X_res_both)
    
    # 使用 ColumnTransformer 先拟合再转换 X_df，比较结果是否与期望的 0.1 * X_res_both 相等
    assert_array_equal(both.fit(X_df).transform(X_df), 0.1 * X_res_both)
    
    # 检查 ColumnTransformer 的 transformers_ 属性长度是否为 1
    assert len(both.transformers_) == 1
    
    # 检查 ColumnTransformer 的 transformers_ 属性最后一个元素的第一个元素是否不为 "remainder"
    assert both.transformers_[-1][0] != "remainder"

    # 确保传递 pandas 对象通过
    
    # 定义 TransAssert 类，用于验证预期的转换类型
    class TransAssert(BaseEstimator):
        def __init__(self, expected_type_transform):
            self.expected_type_transform = expected_type_transform

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            # 断言 X 的类型是否符合预期的类型
            assert isinstance(X, self.expected_type_transform)
            if isinstance(X, dataframe_lib.Series):
                X = X.to_frame()
            return X

    # 使用 ColumnTransformer 验证 "first" 和 "second" 列的数据类型为 dataframe_lib.DataFrame
    ct = ColumnTransformer(
        [
            (
                "trans",
                TransAssert(expected_type_transform=dataframe_lib.DataFrame),
                ["first", "second"],
            )
        ]
    )
    ct.fit_transform(X_df)
    # 如果构造函数名称是 "dataframe"
    if constructor_name == "dataframe":
        # DataFrame 协议不支持 1 维列，因此我们只在 Pandas 数据框上进行测试。
        # 创建 ColumnTransformer 对象，设置一个转换器，验证预期的转换类型为 dataframe_lib.Series
        ct = ColumnTransformer(
            [
                (
                    "trans",
                    TransAssert(expected_type_transform=dataframe_lib.Series),
                    "first",
                )
            ],
            remainder="drop",
        )
        # 对输入数据 X_df 进行拟合转换
        ct.fit_transform(X_df)

        # 只在 Pandas 上进行测试，因为 DataFrame 协议要求列名必须是字符串
        # 整数列规范 + 整数列名 -> 仍然使用位置索引
        X_df2 = X_df.copy()
        X_df2.columns = [1, 0]
        # 创建另一个 ColumnTransformer 对象，设置一个转换器
        ct = ColumnTransformer([("trans", Trans(), 0)], remainder="drop")
        # 断言转换后的结果与预期结果 X_res_first 相等
        assert_array_equal(ct.fit_transform(X_df2), X_res_first)
        # 断言先拟合再转换的结果与预期结果 X_res_first 相等
        assert_array_equal(ct.fit(X_df2).transform(X_df2), X_res_first)

        # 断言转换器列表的长度为 2
        assert len(ct.transformers_) == 2
        # 断言最后一个转换器的名称为 "remainder"
        assert ct.transformers_[-1][0] == "remainder"
        # 断言最后一个转换器的转换方式为 "drop"
        assert ct.transformers_[-1][1] == "drop"
        # 断言最后一个转换器应用的列索引为 [1]
        assert_array_equal(ct.transformers_[-1][2], [1])
@pytest.mark.parametrize("pandas", [True, False], ids=["pandas", "numpy"])
@pytest.mark.parametrize(
    "column_selection",
    [[], np.array([False, False]), [False, False]],
    ids=["list", "bool", "bool_int"],
)
@pytest.mark.parametrize("callable_column", [False, True])
def test_column_transformer_empty_columns(pandas, column_selection, callable_column):
    # 测试用例确保列转换器在给定转换器没有任何列可操作时也能正常工作

    # 创建一个二维数组作为输入数据
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_res_both = X_array  # 保存原始输入数据的副本

    # 如果测试中需要使用 pandas 库，则导入并创建 DataFrame
    if pandas:
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X_array, columns=["first", "second"])
    else:
        X = X_array

    # 根据 callable_column 的值选择不同的列选择器函数或数组
    if callable_column:
        column = lambda X: column_selection  # noqa
    else:
        column = column_selection

    # 创建列转换器对象 ct，定义多个转换器及其对应的操作列
    ct = ColumnTransformer(
        [("trans1", Trans(), [0, 1]), ("trans2", TransRaise(), column)]
    )
    # 断言转换器对 X 的操作结果与预期结果 X_res_both 相等
    assert_array_equal(ct.fit_transform(X), X_res_both)
    assert_array_equal(ct.fit(X).transform(X), X_res_both)
    assert len(ct.transformers_) == 2
    assert isinstance(ct.transformers_[1][1], TransRaise)

    # 重新创建列转换器对象，改变转换器及其操作列的顺序
    ct = ColumnTransformer(
        [("trans1", TransRaise(), column), ("trans2", Trans(), [0, 1])]
    )
    assert_array_equal(ct.fit_transform(X), X_res_both)
    assert_array_equal(ct.fit(X).transform(X), X_res_both)
    assert len(ct.transformers_) == 2
    assert isinstance(ct.transformers_[0][1], TransRaise)

    # 创建列转换器对象，只有一个转换器和其对应的操作列，并指定 remainder 参数
    ct = ColumnTransformer([("trans", TransRaise(), column)], remainder="passthrough")
    assert_array_equal(ct.fit_transform(X), X_res_both)
    assert_array_equal(ct.fit(X).transform(X), X_res_both)
    assert len(ct.transformers_) == 2  # 包括 remainder
    assert isinstance(ct.transformers_[0][1], TransRaise)

    # 创建列转换器对象，只有一个转换器和其对应的操作列，并指定 remainder 参数为 drop
    fixture = np.array([[], [], []])
    ct = ColumnTransformer([("trans", TransRaise(), column)], remainder="drop")
    assert_array_equal(ct.fit_transform(X), fixture)
    assert_array_equal(ct.fit(X).transform(X), fixture)
    assert len(ct.transformers_) == 2  # 包括 remainder
    assert isinstance(ct.transformers_[0][1], TransRaise)


def test_column_transformer_output_indices():
    # 检查 output_indices_ 属性的输出情况
    X_array = np.arange(6).reshape(3, 2)

    # 创建列转换器对象 ct，并指定多个转换器及其对应的操作列
    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", Trans(), [1])])
    X_trans = ct.fit_transform(X_array)
    assert ct.output_indices_ == {
        "trans1": slice(0, 1),
        "trans2": slice(1, 2),
        "remainder": slice(0, 0),
    }
    assert_array_equal(X_trans[:, [0]], X_trans[:, ct.output_indices_["trans1"]])
    assert_array_equal(X_trans[:, [1]], X_trans[:, ct.output_indices_["trans2"]])

    # 使用 transformer_weights 和多个列创建列转换器对象 ct
    ct = ColumnTransformer(
        [("trans", Trans(), [0, 1])], transformer_weights={"trans": 0.1}
    )
    X_trans = ct.fit_transform(X_array)
    assert ct.output_indices_ == {"trans": slice(0, 2), "remainder": slice(0, 0)}


这些注释解释了每行代码的作用，包括测试函数和相关的列转换器操作。
    # 检查转换后的特征是否与指定索引处的列相等
    assert_array_equal(X_trans[:, [0, 1]], X_trans[:, ct.output_indices_["trans"]])
    # 检查转换后的特征是否为空，即检查是否没有指定索引处的列
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["remainder"]])

    # 创建一个 ColumnTransformer 对象，包含两个转换器，其中一个没有任何列要处理
    # 测试确保当给定的转换器没有任何列可处理时，属性也能正常工作
    ct = ColumnTransformer([("trans1", Trans(), [0, 1]), ("trans2", TransRaise(), [])])
    # 对输入数据进行转换
    X_trans = ct.fit_transform(X_array)
    # 检查输出索引是否与预期的字典相等
    assert ct.output_indices_ == {
        "trans1": slice(0, 2),
        "trans2": slice(0, 0),
        "remainder": slice(0, 0),
    }
    # 检查转换后的特征是否与指定索引处的列相等
    assert_array_equal(X_trans[:, [0, 1]], X_trans[:, ct.output_indices_["trans1"]])
    # 检查转换后的特征是否为空，即检查是否没有指定索引处的列
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["trans2"]])
    # 检查转换后的特征是否为空，即检查是否没有指定索引处的列
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["remainder"]])

    # 创建一个 ColumnTransformer 对象，只包含一个没有任何列要处理的转换器，并设置 remainder 参数为 "passthrough"
    ct = ColumnTransformer([("trans", TransRaise(), [])], remainder="passthrough")
    # 对输入数据进行转换
    X_trans = ct.fit_transform(X_array)
    # 检查输出索引是否与预期的字典相等
    assert ct.output_indices_ == {"trans": slice(0, 0), "remainder": slice(0, 2)}
    # 检查转换后的特征是否为空，即检查是否没有指定索引处的列
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["trans"]])
    # 检查转换后的特征是否与指定索引处的列相等
    assert_array_equal(X_trans[:, [0, 1]], X_trans[:, ct.output_indices_["remainder"]])
def test_column_transformer_output_indices_df():
    # 导入 pytest 库，如果库不存在则跳过测试
    pd = pytest.importorskip("pandas")

    # 创建一个包含数据的 DataFrame 对象 X_df
    X_df = pd.DataFrame(np.arange(6).reshape(3, 2), columns=["first", "second"])

    # 创建 ColumnTransformer 对象 ct，指定两个转换器并应用于不同的列
    ct = ColumnTransformer(
        [("trans1", Trans(), ["first"]), ("trans2", Trans(), ["second"])]
    )

    # 对 DataFrame X_df 进行转换，并将结果存储在 X_trans 中
    X_trans = ct.fit_transform(X_df)

    # 断言输出的列索引信息是否正确
    assert ct.output_indices_ == {
        "trans1": slice(0, 1),
        "trans2": slice(1, 2),
        "remainder": slice(0, 0),
    }

    # 断言列的转换结果是否与指定的输出索引相匹配
    assert_array_equal(X_trans[:, [0]], X_trans[:, ct.output_indices_["trans1"]])
    assert_array_equal(X_trans[:, [1]], X_trans[:, ct.output_indices_["trans2"]])
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["remainder"]])

    # 使用整数索引方式再次测试 ColumnTransformer 对象 ct
    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", Trans(), [1])])
    X_trans = ct.fit_transform(X_df)

    # 断言输出的列索引信息是否正确
    assert ct.output_indices_ == {
        "trans1": slice(0, 1),
        "trans2": slice(1, 2),
        "remainder": slice(0, 0),
    }

    # 断言列的转换结果是否与指定的输出索引相匹配
    assert_array_equal(X_trans[:, [0]], X_trans[:, ct.output_indices_["trans1"]])
    assert_array_equal(X_trans[:, [1]], X_trans[:, ct.output_indices_["trans2"]])
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["remainder"]])


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_column_transformer_sparse_array(csr_container):
    # 创建稀疏矩阵 X_sparse
    X_sparse = csr_container(sparse.eye(3, 2))

    # 对于不同的列选择方式和remainder参数设置，进行转换器测试
    # 遍历列选择方式和remainder参数的组合，确保转换器的稀疏性和转换结果的正确性
    for col in [(0,), [0], slice(0, 1)]:
        for remainder, res in [("drop", X_sparse[:, [0]]), ("passthrough", X_sparse)]:
            ct = ColumnTransformer(
                [("trans", Trans(), col)], remainder=remainder, sparse_threshold=0.8
            )
            assert sparse.issparse(ct.fit_transform(X_sparse))
            assert_allclose_dense_sparse(ct.fit_transform(X_sparse), res)
            assert_allclose_dense_sparse(ct.fit(X_sparse).transform(X_sparse), res)

    # 针对包含多列选择的情况，测试转换器的稀疏性和转换结果的正确性
    for col in [[0, 1], slice(0, 2)]:
        ct = ColumnTransformer([("trans", Trans(), col)], sparse_threshold=0.8)
        assert sparse.issparse(ct.fit_transform(X_sparse))
        assert_allclose_dense_sparse(ct.fit_transform(X_sparse), X_sparse)
        assert_allclose_dense_sparse(ct.fit(X_sparse).transform(X_sparse), X_sparse)


def test_column_transformer_list():
    # 创建一个包含列表的 X_list 和期望的结果 expected_result
    X_list = [[1, float("nan"), "a"], [0, 0, "b"]]
    expected_result = np.array(
        [
            [1, float("nan"), 1, 0],
            [-1, 0, 0, 1],
        ]
    )

    # 创建 ColumnTransformer 对象 ct，包含数值列和分类列的转换器
    ct = ColumnTransformer(
        [
            ("numerical", StandardScaler(), [0, 1]),
            ("categorical", OneHotEncoder(), [2]),
        ]
    )

    # 断言列表 X_list 经转换后的结果是否与期望的结果相等
    assert_array_equal(ct.fit_transform(X_list), expected_result)
    assert_array_equal(ct.fit(X_list).transform(X_list), expected_result)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_column_transformer_sparse_stacking(csr_container):
    # 创建一个 NumPy 数组，包含两行三列的数据，然后转置
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    
    # 创建 ColumnTransformer 对象，定义两个转换器：
    # - "trans1" 使用 Trans() 进行转换，作用于第一列
    # - "trans2" 使用 SparseMatrixTrans(csr_container) 进行转换，作用于第二列
    # sparse_threshold 设定为 0.8，表示当稀疏性大于等于 0.8 时认为是稀疏矩阵
    col_trans = ColumnTransformer(
        [("trans1", Trans(), [0]), ("trans2", SparseMatrixTrans(csr_container), 1)],
        sparse_threshold=0.8,
    )
    
    # 对 ColumnTransformer 进行拟合，使用 X_array 数据
    col_trans.fit(X_array)
    
    # 对 X_array 应用 ColumnTransformer 进行转换
    X_trans = col_trans.transform(X_array)
    
    # 断言结果 X_trans 是稀疏矩阵
    assert sparse.issparse(X_trans)
    
    # 断言 X_trans 的形状是 (行数, 行数+1)
    assert X_trans.shape == (X_trans.shape[0], X_trans.shape[0] + 1)
    
    # 断言 X_trans 去掉第一列后的数据等于单位矩阵
    assert_array_equal(X_trans.toarray()[:, 1:], np.eye(X_trans.shape[0]))
    
    # 断言 ColumnTransformer 中包含的转换器数量为 2
    assert len(col_trans.transformers_) == 2
    
    # 断言 ColumnTransformer 中最后一个转换器的名称不是 "remainder"
    assert col_trans.transformers_[-1][0] != "remainder"
    
    # 重新创建 ColumnTransformer 对象，定义与之前相同的转换器，但 sparse_threshold 设定为 0.1
    col_trans = ColumnTransformer(
        [("trans1", Trans(), [0]), ("trans2", SparseMatrixTrans(csr_container), 1)],
        sparse_threshold=0.1,
    )
    
    # 对新的 ColumnTransformer 对象进行拟合，使用相同的 X_array 数据
    col_trans.fit(X_array)
    
    # 再次对 X_array 应用新的 ColumnTransformer 进行转换
    X_trans = col_trans.transform(X_array)
    
    # 断言结果 X_trans 不是稀疏矩阵
    assert not sparse.issparse(X_trans)
    
    # 断言 X_trans 的形状是 (行数, 行数+1)
    assert X_trans.shape == (X_trans.shape[0], X_trans.shape[0] + 1)
    
    # 断言 X_trans 去掉第一列后的数据等于单位矩阵
    assert_array_equal(X_trans[:, 1:], np.eye(X_trans.shape[0]))
def test_column_transformer_mixed_cols_sparse():
    # 创建一个包含字符串、整数和布尔值的NumPy数组
    df = np.array([["a", 1, True], ["b", 2, False]], dtype="O")

    # 创建一个列转换器对象，应用OneHotEncoder到第一列，保持第二和第三列不变，并设置稀疏阈值为1.0
    ct = make_column_transformer(
        (OneHotEncoder(), [0]), ("passthrough", [1, 2]), sparse_threshold=1.0
    )

    # 对数据框进行转换
    X_trans = ct.fit_transform(df)
    # 断言转换后的输出格式为CSR稀疏矩阵
    assert X_trans.getformat() == "csr"
    # 断言转换后的数组与预期的稀疏矩阵一致
    assert_array_equal(X_trans.toarray(), np.array([[1, 0, 1, 1], [0, 1, 2, 0]]))

    # 创建另一个列转换器对象，尝试对第一列和第二列应用OneHotEncoder，但第一列包含字符串，预期引发值错误异常
    ct = make_column_transformer(
        (OneHotEncoder(), [0]), ("passthrough", [0]), sparse_threshold=1.0
    )
    with pytest.raises(ValueError, match="For a sparse output, all columns should"):
        # 断言该转换引发了值错误异常
        ct.fit_transform(df)


def test_column_transformer_sparse_threshold():
    # 创建一个包含字符串的NumPy数组，转置后的稀疏度为0.5
    X_array = np.array([["a", "b"], ["A", "B"]], dtype=object).T

    # 创建一个列转换器对象，应用OneHotEncoder到第一列和第二列，设置稀疏阈值为0.2
    col_trans = ColumnTransformer(
        [("trans1", OneHotEncoder(), [0]), ("trans2", OneHotEncoder(), [1])],
        sparse_threshold=0.2,
    )
    # 执行转换并断言结果不是稀疏矩阵，且不产生稀疏输出
    res = col_trans.fit_transform(X_array)
    assert not sparse.issparse(res)
    assert not col_trans.sparse_output_

    # 创建多个列转换器对象，应用OneHotEncoder到第一列和第二列，根据不同的稀疏阈值测试不同的稀疏性
    for thres in [0.75001, 1]:
        col_trans = ColumnTransformer(
            [
                ("trans1", OneHotEncoder(sparse_output=True), [0]),
                ("trans2", OneHotEncoder(sparse_output=False), [1]),
            ],
            sparse_threshold=thres,
        )
        # 执行转换并断言结果是稀疏矩阵，且产生稀疏输出
        res = col_trans.fit_transform(X_array)
        assert sparse.issparse(res)
        assert col_trans.sparse_output_

    # 创建多个列转换器对象，应用OneHotEncoder到第一列和第二列，根据不同的稀疏阈值测试不同的稀疏性
    for thres in [0.75, 0]:
        col_trans = ColumnTransformer(
            [
                ("trans1", OneHotEncoder(sparse_output=True), [0]),
                ("trans2", OneHotEncoder(sparse_output=False), [1]),
            ],
            sparse_threshold=thres,
        )
        # 执行转换并断言结果不是稀疏矩阵，且不产生稀疏输出
        res = col_trans.fit_transform(X_array)
        assert not sparse.issparse(res)
        assert not col_trans.sparse_output_

    # 创建多个列转换器对象，应用不产生稀疏输出的OneHotEncoder到第一列和第二列，根据不同的稀疏阈值测试不同的稀疏性
    for thres in [0.33, 0, 1]:
        col_trans = ColumnTransformer(
            [
                ("trans1", OneHotEncoder(sparse_output=False), [0]),
                ("trans2", OneHotEncoder(sparse_output=False), [1]),
            ],
            sparse_threshold=thres,
        )
        # 执行转换并断言结果不是稀疏矩阵，且不产生稀疏输出
        res = col_trans.fit_transform(X_array)
        assert not sparse.issparse(res)
        assert not col_trans.sparse_output_


def test_column_transformer_error_msg_1D():
    # 创建一个包含浮点数的NumPy数组，转置后的数据是一维的
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T

    # 创建一个列转换器对象，尝试对一维数据应用StandardScaler，预期引发值错误异常
    col_trans = ColumnTransformer([("trans", StandardScaler(), 0)])
    msg = "1D data passed to a transformer"
    with pytest.raises(ValueError, match=msg):
        # 断言该转换引发了值错误异常
        col_trans.fit(X_array)
    # 使用 pytest 测试框架来验证抛出 ValueError 异常，并匹配指定的错误消息
    with pytest.raises(ValueError, match=msg):
        col_trans.fit_transform(X_array)
    
    # 创建 ColumnTransformer 对象 col_trans，指定一个转换器 TransRaise() 应用于第 0 列
    col_trans = ColumnTransformer([("trans", TransRaise(), 0)])
    
    # 遍历 col_trans 对象中的 fit 和 fit_transform 方法
    for func in [col_trans.fit, col_trans.fit_transform]:
        # 使用 pytest 测试框架来验证调用 func 方法时抛出 ValueError 异常，并匹配指定的错误消息 "specific message"
        with pytest.raises(ValueError, match="specific message"):
            func(X_array)
# 定义一个测试函数，用于测试二维转换器的输出
def test_2D_transformer_output():
    # 创建一个二维 NumPy 数组
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    # 创建列转换器对象，其中第一个转换器会被丢弃，第二个转换器使用自定义的 TransNo2D 类
    ct = ColumnTransformer([("trans1", "drop", 0), ("trans2", TransNo2D(), 1)])

    # 预期抛出 ValueError 异常，匹配指定的错误信息
    msg = "the 'trans2' transformer should be 2D"
    with pytest.raises(ValueError, match=msg):
        # 对 X_array 进行拟合和转换
        ct.fit_transform(X_array)

    # 因为在拟合时也会执行转换，因此这里在拟合时抛出异常
    with pytest.raises(ValueError, match=msg):
        ct.fit(X_array)


# 定义另一个测试函数，用于测试 Pandas DataFrame 输入的二维转换器输出
def test_2D_transformer_output_pandas():
    # 导入 Pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")

    # 创建一个二维 NumPy 数组和对应的 Pandas DataFrame
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = pd.DataFrame(X_array, columns=["col1", "col2"])

    # 创建列转换器对象，使用自定义的 TransNo2D 类作为转换器，处理 'col1' 列
    ct = ColumnTransformer([("trans1", TransNo2D(), "col1")])

    # 预期抛出 ValueError 异常，匹配指定的错误信息
    msg = "the 'trans1' transformer should be 2D"
    with pytest.raises(ValueError, match=msg):
        # 对 X_df 进行拟合和转换
        ct.fit_transform(X_df)

    # 因为在拟合时也会执行转换，因此这里在拟合时抛出异常
    with pytest.raises(ValueError, match=msg):
        ct.fit(X_df)


# 使用参数化测试装饰器来测试列转换器的无效列处理情况
@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
def test_column_transformer_invalid_columns(remainder):
    # 创建一个二维 NumPy 数组
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    # 测试通用无效列情况
    for col in [1.5, ["string", 1], slice(1, "s"), np.array([1.0])]:
        ct = ColumnTransformer([("trans", Trans(), col)], remainder=remainder)
        # 预期抛出 ValueError 异常，匹配指定的错误信息
        with pytest.raises(ValueError, match="No valid specification"):
            ct.fit(X_array)

    # 测试数组输入时的无效列情况
    for col in ["string", ["string", "other"], slice("a", "b")]:
        ct = ColumnTransformer([("trans", Trans(), col)], remainder=remainder)
        # 预期抛出 ValueError 异常，匹配指定的错误信息
        with pytest.raises(ValueError, match="Specifying the columns"):
            ct.fit(X_array)

    # 测试拟合后的特征数量不匹配的情况
    col = [0, 1]
    ct = ColumnTransformer([("trans", Trans(), col)], remainder=remainder)
    ct.fit(X_array)
    X_array_more = np.array([[0, 1, 2], [2, 4, 6], [3, 6, 9]]).T
    msg = "X has 3 features, but ColumnTransformer is expecting 2 features as input."
    # 预期抛出 ValueError 异常，匹配指定的错误信息
    with pytest.raises(ValueError, match=msg):
        ct.transform(X_array_more)
    X_array_fewer = np.array([[0, 1, 2],]).T
    err_msg = "X has 1 features, but ColumnTransformer is expecting 2 features as input."
    # 预期抛出 ValueError 异常，匹配指定的错误信息
    with pytest.raises(ValueError, match=err_msg):
        ct.transform(X_array_fewer)


# 定义测试函数，用于测试未实现 transform 方法的转换器
def test_column_transformer_invalid_transformer():
    # 定义一个不实现 transform 方法的转换器类
    class NoTrans(BaseEstimator):
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return X

    # 创建一个二维 NumPy 数组
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    # 创建列转换器对象，使用未实现 transform 方法的转换器类
    ct = ColumnTransformer([("trans", NoTrans(), [0])])

    # 预期抛出 TypeError 异常，匹配指定的错误信息
    msg = "All estimators should implement fit and transform"
    with pytest.raises(TypeError, match=msg):
        # 对 X_array 进行拟合
        ct.fit(X_array)


# 定义一个测试函数，用于测试创建列转换器对象
def test_make_column_transformer():
    # 创建一个标准缩放器对象
    scaler = StandardScaler()
    # 创建一个 Normalizer 实例
    norm = Normalizer()
    # 创建一个 ColumnTransformer 实例，其中包含两个转换器：一个使用 scaler 对第一列进行转换，另一个使用 norm 对第二列进行转换
    ct = make_column_transformer((scaler, "first"), (norm, ["second"]))
    # 解包 ct.transformers，分别获取转换器的名称、实例和作用的列
    names, transformers, columns = zip(*ct.transformers)
    # 断言转换器的名称应该是 ("standardscaler", "normalizer")
    assert names == ("standardscaler", "normalizer")
    # 断言转换器的实例应该是 (scaler, norm)
    assert transformers == (scaler, norm)
    # 断言转换器作用的列应该是 ("first", ["second"])
    assert columns == ("first", ["second"])
def test_make_column_transformer_pandas():
    # 导入 pandas 库，并在不存在时跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个 NumPy 数组作为数据集 X_array
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    # 将 NumPy 数组转换为 pandas 的 DataFrame X_df，指定列名
    X_df = pd.DataFrame(X_array, columns=["first", "second"])
    # 创建 Normalizer 实例
    norm = Normalizer()
    # 使用 ColumnTransformer 创建 ct1，指定使用 Normalizer 对所有列进行转换
    ct1 = ColumnTransformer([("norm", Normalizer(), X_df.columns)])
    # 使用 make_column_transformer 创建 ct2，与 ct1 等效
    ct2 = make_column_transformer((norm, X_df.columns))
    # 断言两种方式转换的结果几乎相等
    assert_almost_equal(ct1.fit_transform(X_df), ct2.fit_transform(X_df))


def test_make_column_transformer_kwargs():
    # 创建 StandardScaler 和 Normalizer 实例
    scaler = StandardScaler()
    norm = Normalizer()
    # 使用 make_column_transformer 创建 ct，指定转换器、列、以及其他关键参数
    ct = make_column_transformer(
        (scaler, "first"),
        (norm, ["second"]),
        n_jobs=3,
        remainder="drop",
        sparse_threshold=0.5,
    )
    # 断言转换器的设定与预期相符
    assert (
        ct.transformers
        == make_column_transformer((scaler, "first"), (norm, ["second"])).transformers
    )
    assert ct.n_jobs == 3
    assert ct.remainder == "drop"
    assert ct.sparse_threshold == 0.5
    # 断言使用不支持的关键字参数时会引发 TypeError 异常
    msg = re.escape(
        "make_column_transformer() got an unexpected "
        "keyword argument 'transformer_weights'"
    )
    with pytest.raises(TypeError, match=msg):
        make_column_transformer(
            (scaler, "first"),
            (norm, ["second"]),
            transformer_weights={"pca": 10, "Transf": 1},
        )


def test_make_column_transformer_remainder_transformer():
    # 创建 StandardScaler 和 Normalizer 实例，以及一个额外的转换器 remainder
    scaler = StandardScaler()
    norm = Normalizer()
    remainder = StandardScaler()
    # 使用 make_column_transformer 创建 ct，指定转换器、列，并设置 remainder
    ct = make_column_transformer(
        (scaler, "first"), (norm, ["second"]), remainder=remainder
    )
    # 断言 ct 的 remainder 参数与预期相符
    assert ct.remainder == remainder


def test_column_transformer_get_set_params():
    # 使用 ColumnTransformer 创建 ct，包含两个转换器
    ct = ColumnTransformer(
        [("trans1", StandardScaler(), [0]), ("trans2", StandardScaler(), [1])]
    )

    # 预期的参数字典 exp
    exp = {
        "n_jobs": None,
        "remainder": "drop",
        "sparse_threshold": 0.3,
        "trans1": ct.transformers[0][1],
        "trans1__copy": True,
        "trans1__with_mean": True,
        "trans1__with_std": True,
        "trans2": ct.transformers[1][1],
        "trans2__copy": True,
        "trans2__with_mean": True,
        "trans2__with_std": True,
        "transformers": ct.transformers,
        "transformer_weights": None,
        "verbose_feature_names_out": True,
        "verbose": False,
        "force_int_remainder_cols": True,
    }

    # 断言获取的参数与预期字典 exp 相等
    assert ct.get_params() == exp

    # 修改 trans1__with_mean 参数，并验证修改成功
    ct.set_params(trans1__with_mean=False)
    assert not ct.get_params()["trans1__with_mean"]

    # 再次修改 trans1 参数为 "passthrough"，并验证修改成功
    ct.set_params(trans1="passthrough")
    exp = {
        "n_jobs": None,
        "remainder": "drop",
        "sparse_threshold": 0.3,
        "trans1": "passthrough",
        "trans2": ct.transformers[1][1],
        "trans2__copy": True,
        "trans2__with_mean": True,
        "trans2__with_std": True,
        "transformers": ct.transformers,
        "transformer_weights": None,
        "verbose_feature_names_out": True,
        "verbose": False,
        "force_int_remainder_cols": True,
    }
    # 使用断言检查对象 ct 的参数是否等于期望值 exp
    assert ct.get_params() == exp
def test_column_transformer_named_estimators():
    # 创建一个包含两列的二维数组
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    # 创建 ColumnTransformer 对象，指定两个转换器：第一个标准化第一列数据，第二个不标准化第二列数据
    ct = ColumnTransformer(
        [
            ("trans1", StandardScaler(), [0]),
            ("trans2", StandardScaler(with_std=False), [1]),
        ]
    )
    # 断言 ColumnTransformer 对象没有属性 transformers_
    assert not hasattr(ct, "transformers_")
    # 对 X_array 进行拟合
    ct.fit(X_array)
    # 断言 ColumnTransformer 对象有属性 transformers_
    assert hasattr(ct, "transformers_")
    # 断言 trans1 转换器是 StandardScaler 类型
    assert isinstance(ct.named_transformers_["trans1"], StandardScaler)
    # 断言可以通过属性访问 trans1 转换器，并且是 StandardScaler 类型
    assert isinstance(ct.named_transformers_.trans1, StandardScaler)
    # 断言 trans2 转换器是 StandardScaler 类型
    assert isinstance(ct.named_transformers_["trans2"], StandardScaler)
    # 断言可以通过属性访问 trans2 转换器，并且是 StandardScaler 类型
    assert isinstance(ct.named_transformers_.trans2, StandardScaler)
    # 断言 trans2 转换器的 with_std 属性为 False
    assert not ct.named_transformers_.trans2.with_std
    # 检查 trans1 转换器是否已拟合
    assert ct.named_transformers_.trans1.mean_ == 1.0


def test_column_transformer_cloning():
    # 创建一个包含两列的二维数组
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T

    # 创建 ColumnTransformer 对象，包含一个标准化转换器应用于第一列数据
    ct = ColumnTransformer([("trans", StandardScaler(), [0])])
    # 对 X_array 进行拟合
    ct.fit(X_array)
    # 断言 transformers 列表中的第一个转换器没有 mean_ 属性
    assert not hasattr(ct.transformers[0][1], "mean_")
    # 断言 transformers_ 列表中的第一个转换器有 mean_ 属性
    assert hasattr(ct.transformers_[0][1], "mean_")

    # 创建另一个 ColumnTransformer 对象，包含一个标准化转换器应用于第一列数据
    ct = ColumnTransformer([("trans", StandardScaler(), [0])])
    # 对 X_array 进行拟合和转换
    ct.fit_transform(X_array)
    # 断言 transformers 列表中的第一个转换器没有 mean_ 属性
    assert not hasattr(ct.transformers[0][1], "mean_")
    # 断言 transformers_ 列表中的第一个转换器有 mean_ 属性
    assert hasattr(ct.transformers_[0][1], "mean_")


def test_column_transformer_get_feature_names():
    # 创建一个包含两列的二维数组
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    # 创建 ColumnTransformer 对象，包含一个自定义转换器应用于第一列和第二列数据
    ct = ColumnTransformer([("trans", Trans(), [0, 1])])
    # 断言在未拟合状态下调用 get_feature_names_out() 会引发 NotFittedError 异常
    with pytest.raises(NotFittedError):
        ct.get_feature_names_out()
    # 对 X_array 进行拟合
    ct.fit(X_array)
    # 准备用于匹配的错误消息
    msg = re.escape(
        "Transformer trans (type Trans) does not provide get_feature_names_out"
    )
    # 断言调用 get_feature_names_out() 时会引发 AttributeError，并匹配特定错误消息
    with pytest.raises(AttributeError, match=msg):
        ct.get_feature_names_out()


def test_column_transformer_special_strings():
    # 创建一个包含两列的二维数组
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    # 创建 ColumnTransformer 对象，包含一个自定义转换器应用于第一列和第二列数据，第二个列使用 'drop' 忽略
    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", "drop", [1])])
    # 期望的结果数组
    exp = np.array([[0.0], [1.0], [2.0]])
    # 断言 fit_transform() 的结果与期望的结果数组相等
    assert_array_equal(ct.fit_transform(X_array), exp)
    # 断言拟合后再转换的结果与期望的结果数组相等
    assert_array_equal(ct.fit(X_array).transform(X_array), exp)
    # 断言 transformers_ 列表的长度为 2
    assert len(ct.transformers_) == 2
    # 断言 transformers_ 列表中最后一个转换器不是 "remainder"
    assert ct.transformers_[-1][0] != "remainder"

    # 创建 ColumnTransformer 对象，包含两个列都使用 'drop' 忽略
    ct = ColumnTransformer([("trans1", "drop", [0]), ("trans2", "drop", [1])])
    # 断言 fit_transform() 的结果形状为 (3, 0)
    assert_array_equal(ct.fit(X_array).transform(X_array).shape, (3, 0))
    # 断言 fit_transform() 的结果形状为 (3, 0)
    assert_array_equal(ct.fit_transform(X_array).shape, (3, 0))
    # 断言 transformers_ 列表的长度为 2
    assert len(ct.transformers_) == 2
    # 断言 transformers_ 列表中最后一个转换器不是 "remainder"

    # 创建 ColumnTransformer 对象，包含一个自定义转换器应用于第一列和第二列数据，第二个列使用 'passthrough'
    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", "passthrough", [1])])
    # 断言 fit_transform() 的结果与原始数据 X_array 相等
    assert_array_equal(ct.fit_transform(X_array), X_array)
    # 断言拟合后再转换的结果与原始数据 X_array 相等
    assert_array_equal(ct.fit(X_array).transform(X_array), X_array)
    # 断言检查数据转换器的数量是否为2
    assert len(ct.transformers_) == 2
    # 断言检查最后一个数据转换器的名称是否不是 "remainder"
    assert ct.transformers_[-1][0] != "remainder"
def test_column_transformer_remainder():
    # 创建一个包含两列的二维数组
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    # 分别创建第一列和第二列的预期结果数组
    X_res_first = np.array([0, 1, 2]).reshape(-1, 1)
    X_res_second = np.array([2, 4, 6]).reshape(-1, 1)
    X_res_both = X_array

    # 使用默认的删除策略创建 ColumnTransformer 对象
    ct = ColumnTransformer([("trans1", Trans(), [0])])
    assert_array_equal(ct.fit_transform(X_array), X_res_first)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_first)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert ct.transformers_[-1][1] == "drop"
    assert_array_equal(ct.transformers_[-1][2], [1])

    # 指定 'passthrough' 策略创建 ColumnTransformer 对象
    ct = ColumnTransformer([("trans", Trans(), [0])], remainder="passthrough")
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert isinstance(ct.transformers_[-1][1], FunctionTransformer)
    assert_array_equal(ct.transformers_[-1][2], [1])

    # 当指定列被添加到末尾时，列顺序不被保留
    ct = ColumnTransformer([("trans1", Trans(), [1])], remainder="passthrough")
    assert_array_equal(ct.fit_transform(X_array), X_res_both[:, ::-1])
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both[:, ::-1])
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert isinstance(ct.transformers_[-1][1], FunctionTransformer)
    assert_array_equal(ct.transformers_[-1][2], [0])

    # 当所有实际的转换器被跳过时，应用 'passthrough' 策略
    ct = ColumnTransformer([("trans1", "drop", [0])], remainder="passthrough")
    assert_array_equal(ct.fit_transform(X_array), X_res_second)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_second)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert isinstance(ct.transformers_[-1][1], FunctionTransformer)
    assert_array_equal(ct.transformers_[-1][2], [1])

    # 检查 make_column_transformer 的默认设置
    ct = make_column_transformer((Trans(), [0]))
    assert ct.remainder == "drop"


# TODO(1.7): check for deprecated force_int_remainder_cols
# TODO(1.9): remove force_int but keep the test
@pytest.mark.parametrize(
    "cols1, cols2",
    [
        ([0], [False, True, False]),  # mix types
        ([0], [1]),  # ints
        (lambda x: [0], lambda x: [1]),  # callables
    ],
)
@pytest.mark.parametrize("force_int", [False, True])
def test_column_transformer_remainder_dtypes_ints(force_int, cols1, cols2):
    """Check that the remainder columns are always stored as indices when
    other columns are not all specified as column names or masks, regardless of
    `force_int_remainder_cols`.
    """
    X = np.ones((1, 3))
    # 创建一个列转换器对象，用于对列进行转换
    ct = make_column_transformer(
        # 第一个转换器，使用 Trans() 对象处理 cols1 中的列
        (Trans(), cols1),
        # 第二个转换器，使用 Trans() 对象处理 cols2 中的列
        (Trans(), cols2),
        # 对于未指定的列保持不变，直接通过
        remainder="passthrough",
        # 指定是否强制将剩余列转换为整数类型
        force_int_remainder_cols=force_int,
    )
    # 忽略警告并捕获警告为异常
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 对输入数据 X 进行拟合转换
        ct.fit_transform(X)
        # 检查最后一个转换器中最后一个步骤是否为期望的值（假设为2）
        assert ct.transformers_[-1][-1][0] == 2
# TODO(1.7): check for deprecated force_int_remainder_cols
# TODO(1.9): remove force_int but keep the test
@pytest.mark.parametrize(
    "force_int, cols1, cols2, expected_cols",
    [
        (True, ["A"], ["B"], [2]),  # Test case where force_int is True
        (False, ["A"], ["B"], ["C"]),  # Test case where force_int is False
        (True, [True, False, False], [False, True, False], [2]),  # Test case with boolean arrays and force_int True
        (False, [True, False, False], [False, True, False], [False, False, True]),  # Test case with boolean arrays and force_int False
    ],
)
def test_column_transformer_remainder_dtypes(force_int, cols1, cols2, expected_cols):
    """Check that the remainder columns format matches the format of the other
    columns when they're all strings or masks, unless `force_int = True`.
    """
    X = np.ones((1, 3))

    if isinstance(cols1[0], str):
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X, columns=["A", "B", "C"])

    # if inputs are column names store remainder columns as column names unless
    # force_int_remainder_cols is True
    ct = make_column_transformer(
        (Trans(), cols1),
        (Trans(), cols2),
        remainder="passthrough",
        force_int_remainder_cols=force_int,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ct.fit_transform(X)

    if force_int:
        # If we forced using ints and we access the remainder columns a warning is shown
        match = "The format of the columns of the 'remainder' transformer"
        cols = ct.transformers_[-1][-1]
        with pytest.warns(FutureWarning, match=match):
            cols[0]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cols = ct.transformers_[-1][-1]
            cols[0]

    assert cols == expected_cols


def test_remainder_list_repr():
    cols = _RemainderColsList([0, 1], warning_enabled=False)
    assert str(cols) == "[0, 1]"
    assert repr(cols) == "[0, 1]"
    mock = Mock()
    cols._repr_pretty_(mock, False)
    mock.text.assert_called_once_with("[0, 1]")


@pytest.mark.parametrize(
    "key, expected_cols",
    [
        ([0], [1]),  # Test case with list key
        (np.array([0]), [1]),  # Test case with numpy array key
        (slice(0, 1), [1]),  # Test case with slice key
        (np.array([True, False]), [False, True]),  # Test case with boolean array key
    ],
)
def test_column_transformer_remainder_numpy(key, expected_cols):
    # test different ways that columns are specified with passthrough
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_res_both = X_array

    ct = ColumnTransformer(
        [("trans1", Trans(), key)],
        remainder="passthrough",
        force_int_remainder_cols=False,
    )
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert isinstance(ct.transformers_[-1][1], FunctionTransformer)
    assert ct.transformers_[-1][2] == expected_cols


@pytest.mark.parametrize(
    "key, expected_cols",
    [
        # 元组1: 包含一个整数列表和一个整数列表
        ([0], [1]),
        # 元组2: 包含一个切片对象和一个整数列表
        (slice(0, 1), [1]),
        # 元组3: 包含一个布尔数组和一个布尔数组
        (np.array([True, False]), [False, True]),
        # 元组4: 包含一个字符串列表和一个字符串列表
        (["first"], ["second"]),
        # 元组5: 包含一个字符串和一个字符串列表
        ("pd-index", ["second"]),
        # 元组6: 包含一个字符串数组和一个字符串列表
        (np.array(["first"]), ["second"]),
        # 元组7: 包含一个对象数组和一个字符串列表
        (np.array(["first"], dtype=object), ["second"]),
        # 元组8: 包含一个切片对象和一个字符串列表
        (slice(None, "first"), ["second"]),
        # 元组9: 包含一个切片对象和一个字符串列表
        (slice("first", "first"), ["second"]),
    ],
def test_column_transformer_remainder_pandas(key, expected_cols):
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    
    # 如果 key 是字符串并且为 "pd-index"，将其转换为 pandas.Index 对象
    if isinstance(key, str) and key == "pd-index":
        key = pd.Index(["first"])
    
    # 创建一个二维 NumPy 数组 X_array
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    
    # 使用 X_array 创建一个 pandas.DataFrame 对象 X_df，指定列名为 ["first", "second"]
    X_df = pd.DataFrame(X_array, columns=["first", "second"])
    
    # X_array 作为预期的转换结果
    X_res_both = X_array
    
    # 创建 ColumnTransformer 对象 ct，其中包含一个转换器，名称为 "trans1"，转换器为 Trans()，应用于 key
    ct = ColumnTransformer(
        [("trans1", Trans(), key)],
        remainder="passthrough",
        force_int_remainder_cols=False,
    )
    
    # 断言转换后的结果与预期的 X_res_both 相等
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    
    # 断言通过 fit 和 transform 后的结果与预期的 X_res_both 相等
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
    
    # 断言 transformers_ 属性的长度为 2
    assert len(ct.transformers_) == 2
    
    # 断言 transformers_ 中最后一个转换器的名称为 "remainder"
    assert ct.transformers_[-1][0] == "remainder"
    
    # 断言 transformers_ 中最后一个转换器的类型为 FunctionTransformer
    assert isinstance(ct.transformers_[-1][1], FunctionTransformer)
    
    # 断言 transformers_ 中最后一个转换器应用的列索引与 expected_cols 相等
    assert ct.transformers_[-1][2] == expected_cols


@pytest.mark.parametrize(
    "key, expected_cols",
    [
        ([0], [1, 2]),
        (np.array([0]), [1, 2]),
        (slice(0, 1), [1, 2]),
        (np.array([True, False, False]), [False, True, True]),
    ],
)
def test_column_transformer_remainder_transformer(key, expected_cols):
    # 创建一个二维 NumPy 数组 X_array
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T
    
    # X_array 作为预期的转换结果
    X_res_both = X_array.copy()
    
    # 第二和第三列在 remainder=DoubleTrans 时会被加倍处理
    X_res_both[:, 1:3] *= 2
    
    # 创建 ColumnTransformer 对象 ct，其中包含一个转换器，名称为 "trans1"，转换器为 Trans()，应用于 key
    # remainder 设置为 DoubleTrans()
    ct = ColumnTransformer(
        [("trans1", Trans(), key)],
        remainder=DoubleTrans(),
        force_int_remainder_cols=False,
    )
    
    # 断言转换后的结果与预期的 X_res_both 相等
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    
    # 断言通过 fit 和 transform 后的结果与预期的 X_res_both 相等
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    
    # 断言 transformers_ 属性的长度为 2
    assert len(ct.transformers_) == 2
    
    # 断言 transformers_ 中最后一个转换器的名称为 "remainder"
    assert ct.transformers_[-1][0] == "remainder"
    
    # 断言 transformers_ 中最后一个转换器的类型为 DoubleTrans
    assert isinstance(ct.transformers_[-1][1], DoubleTrans)
    
    # 断言 transformers_ 中最后一个转换器应用的列索引与 expected_cols 相等
    assert ct.transformers_[-1][2] == expected_cols


def test_column_transformer_no_remaining_remainder_transformer():
    # 创建一个二维 NumPy 数组 X_array
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T
    
    # 创建 ColumnTransformer 对象 ct，其中包含一个转换器，名称为 "trans1"，转换器为 Trans()，应用于列 [0, 1, 2]
    # remainder 设置为 DoubleTrans()
    ct = ColumnTransformer([("trans1", Trans(), [0, 1, 2])], remainder=DoubleTrans())
    
    # 断言转换后的结果与预期的 X_array 相等
    assert_array_equal(ct.fit_transform(X_array), X_array)
    
    # 断言通过 fit 和 transform 后的结果与预期的 X_array 相等
    assert_array_equal(ct.fit(X_array).transform(X_array), X_array)
    
    # 断言 transformers_ 属性的长度为 1
    assert len(ct.transformers_) == 1
    
    # 断言 transformers_ 中最后一个转换器不是 "remainder"
    assert ct.transformers_[-1][0] != "remainder"


def test_column_transformer_drops_all_remainder_transformer():
    # 创建一个二维 NumPy 数组 X_array
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T
    
    # columns 被 DoubleTrans 处理后会加倍
    X_res_both = 2 * X_array.copy()[:, 1:3]
    
    # 创建 ColumnTransformer 对象 ct，其中包含一个转换器，名称为 "trans1"，被设为 "drop"，应用于列 [0]
    # remainder 设置为 DoubleTrans()
    ct = ColumnTransformer([("trans1", "drop", [0])], remainder=DoubleTrans())
    
    # 断言转换后的结果与预期的 X_res_both 相等
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    
    # 断言通过 fit 和 transform 后的结果与预期的 X_res_both 相等
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    
    # 断言 transformers_ 属性的长度为 2
    assert len(ct.transformers_) == 2
    
    # 断言 transformers_ 中最后一个转换器的名称为 "remainder"
    assert ct.transformers_[-1][0] == "remainder"
    
    # 断言 transformers_ 中最后一个转换器的类型为 DoubleTrans
    assert isinstance(ct.transformers_[-1][1], DoubleTrans)
    
    # 断言 transformers_ 中最后一个转换器应用的列索引与预期的 [1, 2] 相等
    assert_array_equal(ct.transformers_[-1][2], [1, 2])
# 定义测试函数，用于测试 ColumnTransformer 在稀疏矩阵处理时的转换效果
def test_column_transformer_sparse_remainder_transformer(csr_container):
    # 创建一个包含 3 列的二维 NumPy 数组
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T

    # 创建 ColumnTransformer 对象，指定第一列使用 Trans() 进行转换，
    # 其余列使用 SparseMatrixTrans(csr_container) 进行转换，稀疏阈值为 0.8
    ct = ColumnTransformer(
        [("trans1", Trans(), [0])],
        remainder=SparseMatrixTrans(csr_container),
        sparse_threshold=0.8,
    )

    # 对输入数组 X_array 进行拟合和转换
    X_trans = ct.fit_transform(X_array)

    # 断言转换后的结果 X_trans 是稀疏矩阵
    assert sparse.issparse(X_trans)

    # 断言转换后的矩阵形状为 (3, 3 + 1)
    assert X_trans.shape == (3, 3 + 1)

    # 生成期望的数组，包含 X_array 的第一列和单位矩阵
    exp_array = np.hstack((X_array[:, 0].reshape(-1, 1), np.eye(3)))
    assert_array_equal(X_trans.toarray(), exp_array)

    # 断言 transformers_ 中包含的转换器数量为 2
    assert len(ct.transformers_) == 2

    # 断言 transformers_ 中最后一个转换器的名称为 "remainder"
    assert ct.transformers_[-1][0] == "remainder"

    # 断言 transformers_ 中最后一个转换器的类型为 SparseMatrixTrans
    assert isinstance(ct.transformers_[-1][1], SparseMatrixTrans)

    # 断言 transformers_ 中最后一个转换器的列索引为 [1, 2]
    assert_array_equal(ct.transformers_[-1][2], [1, 2])


# 使用参数化测试，测试在删除所有稀疏数据后的 ColumnTransformer 的转换效果
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_column_transformer_drop_all_sparse_remainder_transformer(csr_container):
    # 创建一个包含 3 列的二维 NumPy 数组
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T

    # 创建 ColumnTransformer 对象，指定第一列使用 "drop" 进行删除，
    # 其余列使用 SparseMatrixTrans(csr_container) 进行转换，稀疏阈值为 0.8
    ct = ColumnTransformer(
        [("trans1", "drop", [0])],
        remainder=SparseMatrixTrans(csr_container),
        sparse_threshold=0.8,
    )

    # 对输入数组 X_array 进行拟合和转换
    X_trans = ct.fit_transform(X_array)

    # 断言转换后的结果 X_trans 是稀疏矩阵
    assert sparse.issparse(X_trans)

    # 断言转换后的矩阵形状为 (3, 3)
    assert X_trans.shape == (3, 3)

    # 断言转换后的数组与单位矩阵相等
    assert_array_equal(X_trans.toarray(), np.eye(3))

    # 断言 transformers_ 中包含的转换器数量为 2
    assert len(ct.transformers_) == 2

    # 断言 transformers_ 中最后一个转换器的名称为 "remainder"
    assert ct.transformers_[-1][0] == "remainder"

    # 断言 transformers_ 中最后一个转换器的类型为 SparseMatrixTrans
    assert isinstance(ct.transformers_[-1][1], SparseMatrixTrans)

    # 断言 transformers_ 中最后一个转换器的列索引为 [1, 2]
    assert_array_equal(ct.transformers_[-1][2], [1, 2])


# 测试 ColumnTransformer 在带有 remainder 参数时的 get_params 和 set_params 方法
def test_column_transformer_get_set_params_with_remainder():
    # 创建 ColumnTransformer 对象，指定第一列使用 StandardScaler 进行转换，
    # remainder 参数也使用 StandardScaler
    ct = ColumnTransformer(
        [("trans1", StandardScaler(), [0])], remainder=StandardScaler()
    )

    # 预期的参数字典
    exp = {
        "n_jobs": None,
        "remainder": ct.remainder,
        "remainder__copy": True,
        "remainder__with_mean": True,
        "remainder__with_std": True,
        "sparse_threshold": 0.3,
        "trans1": ct.transformers[0][1],
        "trans1__copy": True,
        "trans1__with_mean": True,
        "trans1__with_std": True,
        "transformers": ct.transformers,
        "transformer_weights": None,
        "verbose_feature_names_out": True,
        "verbose": False,
        "force_int_remainder_cols": True,
    }

    # 断言获取的参数字典与预期相等
    assert ct.get_params() == exp

    # 设置 remainder__with_std 参数为 False，然后断言其值
    ct.set_params(remainder__with_std=False)
    assert not ct.get_params()["remainder__with_std"]

    # 设置 trans1 参数为 "passthrough"，然后更新预期的参数字典
    ct.set_params(trans1="passthrough")
    exp = {
        "n_jobs": None,
        "remainder": ct.remainder,
        "remainder__copy": True,
        "remainder__with_mean": True,
        "remainder__with_std": False,
        "sparse_threshold": 0.3,
        "trans1": "passthrough",
        "transformers": ct.transformers,
        "transformer_weights": None,
        "verbose_feature_names_out": True,
        "verbose": False,
        "force_int_remainder_cols": True,
    }
    # 使用断言验证对象 ct 的参数是否等于期望值 exp
    assert ct.get_params() == exp
# 定义一个测试函数，用于测试 ColumnTransformer 在没有估计器时的行为
def test_column_transformer_no_estimators():
    # 创建一个 numpy 数组作为输入数据 X_array，形状为 (3, 3)，并将其转换为浮点类型
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).astype("float").T
    # 创建一个 ColumnTransformer 对象 ct，指定空的估计器列表，同时设置 remainder 参数为 StandardScaler 对象
    ct = ColumnTransformer([], remainder=StandardScaler())

    # 获取 ColumnTransformer 对象的参数，并进行断言检查
    params = ct.get_params()
    assert params["remainder__with_mean"]

    # 对输入数据 X_array 进行拟合和转换，得到转换后的数据 X_trans
    X_trans = ct.fit_transform(X_array)
    # 断言转换后的数据 X_trans 的形状与原始输入数据 X_array 的形状相同
    assert X_trans.shape == X_array.shape
    # 断言 ColumnTransformer 对象中的 transformers_ 列表长度为 1
    assert len(ct.transformers_) == 1
    # 断言 transformers_ 列表中最后一个元素的第一个元素为 "remainder"
    assert ct.transformers_[-1][0] == "remainder"
    # 断言 transformers_ 列表中最后一个元素的第三个元素（转换后的特征索引列表）为 [0, 1, 2]
    assert ct.transformers_[-1][2] == [0, 1, 2]

# 使用 pytest 的 parametrize 装饰器进行参数化测试
@pytest.mark.parametrize(
    ["est", "pattern"],
    [
        (
            # 创建 ColumnTransformer 对象，指定两个转换器及其应用的列索引，剩余部分使用 DoubleTrans 处理
            ColumnTransformer(
                [("trans1", Trans(), [0]), ("trans2", Trans(), [1])],
                remainder=DoubleTrans(),
            ),
            # 匹配日志信息的正则表达式，确认每个转换器的处理顺序及总数
            (
                r"\[ColumnTransformer\].*\(1 of 3\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 3\) Processing trans2.* total=.*\n"
                r"\[ColumnTransformer\].*\(3 of 3\) Processing remainder.* total=.*\n$"
            ),
        ),
        (
            # 创建 ColumnTransformer 对象，指定两个转换器及其应用的列索引，剩余部分透传
            ColumnTransformer(
                [("trans1", Trans(), [0]), ("trans2", Trans(), [1])],
                remainder="passthrough",
            ),
            # 匹配日志信息的正则表达式，确认每个转换器的处理顺序及总数
            (
                r"\[ColumnTransformer\].*\(1 of 3\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 3\) Processing trans2.* total=.*\n"
                r"\[ColumnTransformer\].*\(3 of 3\) Processing remainder.* total=.*\n$"
            ),
        ),
        (
            # 创建 ColumnTransformer 对象，指定一个转换器和一个要丢弃的列索引，剩余部分透传
            ColumnTransformer(
                [("trans1", Trans(), [0]), ("trans2", "drop", [1])],
                remainder="passthrough",
            ),
            # 匹配日志信息的正则表达式，确认每个转换器的处理顺序及总数
            (
                r"\[ColumnTransformer\].*\(1 of 2\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 2\) Processing remainder.* total=.*\n$"
            ),
        ),
        (
            # 创建 ColumnTransformer 对象，指定一个转换器和一个透传的列索引，剩余部分透传
            ColumnTransformer(
                [("trans1", Trans(), [0]), ("trans2", "passthrough", [1])],
                remainder="passthrough",
            ),
            # 匹配日志信息的正则表达式，确认每个转换器的处理顺序及总数
            (
                r"\[ColumnTransformer\].*\(1 of 3\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 3\) Processing trans2.* total=.*\n"
                r"\[ColumnTransformer\].*\(3 of 3\) Processing remainder.* total=.*\n$"
            ),
        ),
        (
            # 创建 ColumnTransformer 对象，指定一个转换器和剩余部分透传
            ColumnTransformer([("trans1", Trans(), [0])], remainder="passthrough"),
            # 匹配日志信息的正则表达式，确认每个转换器的处理顺序及总数
            (
                r"\[ColumnTransformer\].*\(1 of 2\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 2\) Processing remainder.* total=.*\n$"
            ),
        ),
        (
            # 创建 ColumnTransformer 对象，指定两个转换器和要丢弃的剩余部分
            ColumnTransformer(
                [("trans1", Trans(), [0]), ("trans2", Trans(), [1])], remainder="drop"
            ),
            # 匹配日志信息的正则表达式，确认每个转换器的处理顺序及总数
            (
                r"\[ColumnTransformer\].*\(1 of 2\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 2\) Processing trans2.* total=.*\n$"
            ),
        ),
        (
            # 创建 ColumnTransformer 对象，指定一个转换器和要丢弃的剩余部分
            ColumnTransformer([("trans1", Trans(), [0])], remainder="drop"),
            # 匹配日志信息的正则表达式，确认每个转换器的处理顺序及总数
            r"\[ColumnTransformer\].*\(1 of 1\) Processing trans1.* total=.*\n$",
        ),
    ],
# 参数化测试，测试两种方法（'fit'和'fit_transform'）的行为
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_column_transformer_verbose(est, pattern, method, capsys):
    # 创建一个简单的二维数组作为输入数据
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T

    # 获取方法的引用（fit 或 fit_transform）
    func = getattr(est, method)
    # 设置估算器的参数 verbose 为 False
    est.set_params(verbose=False)
    # 执行方法并捕获输出
    func(X_array)
    # 断言不应有任何输出，因为 verbose 被设置为 False
    assert not capsys.readouterr().out, "Got output for verbose=False"

    # 设置估算器的参数 verbose 为 True
    est.set_params(verbose=True)
    # 再次执行方法并捕获输出
    func(X_array)
    # 使用正则表达式匹配输出是否符合预期的模式
    assert re.match(pattern, capsys.readouterr()[0])


# 测试当没有设置估算器时设置参数 n_jobs 的行为
def test_column_transformer_no_estimators_set_params():
    # 创建一个空的 ColumnTransformer 对象，并设置参数 n_jobs 为 2
    ct = ColumnTransformer([]).set_params(n_jobs=2)
    # 断言参数 n_jobs 是否被正确设置为 2
    assert ct.n_jobs == 2


# 测试使用可调用对象指定转换器行为的情况
def test_column_transformer_callable_specifier():
    # 创建一个简单的二维数组作为输入数据
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    # 期望的第一列数据
    X_res_first = np.array([[0, 1, 2]]).T

    # 定义一个函数，用于检查输入是否符合预期，并返回特定的值
    def func(X):
        assert_array_equal(X, X_array)
        return [0]

    # 创建一个 ColumnTransformer 对象，使用自定义转换器和指定的函数
    ct = ColumnTransformer([("trans", Trans(), func)], remainder="drop")
    # 断言 fit_transform 方法的输出是否符合预期
    assert_array_equal(ct.fit_transform(X_array), X_res_first)
    # 断言 fit 和 transform 方法的输出是否符合预期
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_first)
    # 断言转换器的第一个元素的转换函数是否为可调用对象
    assert callable(ct.transformers[0][2])
    # 断言转换器的第一个元素的转换器（transformer）属性是否为预期的值
    assert ct.transformers_[0][2] == [0]


# 测试在处理 DataFrame 时使用可调用对象指定转换器行为的情况
def test_column_transformer_callable_specifier_dataframe():
    # 导入 pandas 库，并确保存在该库
    pd = pytest.importorskip("pandas")
    # 创建一个简单的二维数组作为输入数据
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    # 期望的第一列数据
    X_res_first = np.array([[0, 1, 2]]).T

    # 创建一个 DataFrame 对象作为输入数据
    X_df = pd.DataFrame(X_array, columns=["first", "second"])

    # 定义一个函数，用于检查 DataFrame 是否符合预期，并返回特定的列
    def func(X):
        assert_array_equal(X.columns, X_df.columns)
        assert_array_equal(X.values, X_df.values)
        return ["first"]

    # 创建一个 ColumnTransformer 对象，使用自定义转换器和指定的函数
    ct = ColumnTransformer([("trans", Trans(), func)], remainder="drop")
    # 断言 fit_transform 方法的输出是否符合预期
    assert_array_equal(ct.fit_transform(X_df), X_res_first)
    # 断言 fit 和 transform 方法的输出是否符合预期
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_first)
    # 断言转换器的第一个元素的转换函数是否为可调用对象
    assert callable(ct.transformers[0][2])
    # 断言转换器的第一个元素的转换器（transformer）属性是否为预期的值
    assert ct.transformers_[0][2] == ["first"]


# 测试在处理负列索引时的行为
def test_column_transformer_negative_column_indexes():
    # 创建一个随机二维数组作为输入数据
    X = np.random.randn(2, 2)
    # 创建一个类别数据列，并将其连接到输入数据中
    X_categories = np.array([[1], [2]])
    X = np.concatenate([X, X_categories], axis=1)

    # 创建一个 OneHotEncoder 对象
    ohe = OneHotEncoder()

    # 创建两个 ColumnTransformer 对象，分别使用负数和正数列索引
    tf_1 = ColumnTransformer([("ohe", ohe, [-1])], remainder="passthrough")
    tf_2 = ColumnTransformer([("ohe", ohe, [2])], remainder="passthrough")
    # 断言两个 ColumnTransformer 对象的输出是否相等
    assert_array_equal(tf_1.fit_transform(X), tf_2.fit_transform(X))


# 参数化测试，测试列变换器在使用不同数组类型时的行为
@pytest.mark.parametrize("array_type", [np.asarray, *CSR_CONTAINERS])
def test_column_transformer_mask_indexing(array_type):
    # 回归测试，确保在稀疏矩阵中使用布尔数组时的行为
    X = np.transpose([[1, 2, 3], [4, 5, 6], [5, 6, 7], [8, 9, 10]])
    X = array_type(X)
    # 创建一个 ColumnTransformer 对象，使用函数转换器和布尔索引
    column_transformer = ColumnTransformer(
        [("identity", FunctionTransformer(), [False, True, False, True])]
    )
    # 执行 fit_transform 方法并断言转换后的形状是否符合预期
    X_trans = column_transformer.fit_transform(X)
    assert X_trans.shape == (3, 2)


# 测试 n_features_in 属性是否正确传递给列变换器的输入
def test_n_features_in():
    # 确保 n_features_in 属性被正确传递给列变换器的输入。
    # 创建一个包含子列表的二维数组 X
    X = [[1, 2], [3, 4], [5, 6]]
    
    # 创建一个 ColumnTransformer 对象 ct，用于对列进行转换
    # ("a", DoubleTrans(), [0]) 表示使用 DoubleTrans 对象处理第 0 列
    # ("b", DoubleTrans(), [1]) 表示使用 DoubleTrans 对象处理第 1 列
    ct = ColumnTransformer([("a", DoubleTrans(), [0]), ("b", DoubleTrans(), [1])])
    
    # 使用断言检查 ct 对象是否没有属性 "n_features_in_"
    assert not hasattr(ct, "n_features_in_")
    
    # 对 ct 对象使用 fit 方法，将 X 作为输入进行拟合
    ct.fit(X)
    
    # 使用断言检查 ct 对象的属性 "n_features_in_" 是否等于 2
    assert ct.n_features_in_ == 2
@pytest.mark.parametrize(
    "cols, pattern, include, exclude",
    [  # 参数化测试参数的组合
        (["col_int", "col_float"], None, np.number, None),  # 测试列为整数和浮点数，包含数值类型，不排除任何类型
        (["col_int", "col_float"], None, None, object),  # 测试列为整数和浮点数，不包含任何特定类型，但排除对象类型
        (["col_int", "col_float"], None, [int, float], None),  # 测试列为整数和浮点数，包含整数和浮点数类型，不排除任何类型
        (["col_str"], None, [object], None),  # 测试列为字符串，包含对象类型，不排除任何类型
        (["col_str"], None, object, None),  # 测试列为字符串，包含对象类型，不排除任何类型
        (["col_float"], None, float, None),  # 测试列为浮点数，包含浮点数类型，不排除任何类型
        (["col_float"], "at$", [np.number], None),  # 测试列为浮点数，模式匹配字符串"at$"，包含数值类型，不排除任何类型
        (["col_int"], None, [int], None),  # 测试列为整数，包含整数类型，不排除任何类型
        (["col_int"], "^col_int", [np.number], None),  # 测试列为整数，模式匹配"^col_int"，包含数值类型，不排除任何类型
        (["col_float", "col_str"], "float|str", None, None),  # 测试列为浮点数或字符串，模式匹配"float|str"，不包含任何特定类型，不排除任何类型
        (["col_str"], "^col_s", None, [int]),  # 测试列为字符串，模式匹配"^col_s"，不包含任何特定类型，排除整数类型
        ([], "str$", float, None),  # 测试空列，模式匹配"str$"，包含浮点数类型，不排除任何类型
        (["col_int", "col_float", "col_str"], None, [np.number, object], None),  # 测试整数、浮点数和字符串列，包含数值类型和对象类型，不排除任何类型
    ],
)
def test_make_column_selector_with_select_dtypes(cols, pattern, include, exclude):
    pd = pytest.importorskip("pandas")  # 导入 pandas 库，如果失败则跳过测试

    X_df = pd.DataFrame(  # 创建一个测试用的 pandas DataFrame
        {
            "col_int": np.array([0, 1, 2], dtype=int),
            "col_float": np.array([0.0, 1.0, 2.0], dtype=float),
            "col_str": ["one", "two", "three"],
        },
        columns=["col_int", "col_float", "col_str"],
    )

    selector = make_column_selector(  # 创建列选择器对象
        dtype_include=include, dtype_exclude=exclude, pattern=pattern
    )

    assert_array_equal(selector(X_df), cols)  # 断言选择器的输出与预期的列名相等


def test_column_transformer_with_make_column_selector():
    # 对列变换器和列选择器进行功能测试
    pd = pytest.importorskip("pandas")  # 导入 pandas 库，如果失败则跳过测试

    X_df = pd.DataFrame(  # 创建一个测试用的 pandas DataFrame
        {
            "col_int": np.array([0, 1, 2], dtype=int),
            "col_float": np.array([0.0, 1.0, 2.0], dtype=float),
            "col_cat": ["one", "two", "one"],
            "col_str": ["low", "middle", "high"],
        },
        columns=["col_int", "col_float", "col_cat", "col_str"],
    )
    X_df["col_str"] = X_df["col_str"].astype("category")  # 将列转换为分类类型

    cat_selector = make_column_selector(dtype_include=["category", object])  # 创建分类类型列选择器
    num_selector = make_column_selector(dtype_include=np.number)  # 创建数值类型列选择器

    ohe = OneHotEncoder()  # 创建独热编码器对象
    scaler = StandardScaler()  # 创建标准化缩放器对象

    ct_selector = make_column_transformer((ohe, cat_selector), (scaler, num_selector))  # 创建列变换器对象
    ct_direct = make_column_transformer(  # 创建另一个列变换器对象
        (ohe, ["col_cat", "col_str"]), (scaler, ["col_float", "col_int"])
    )

    X_selector = ct_selector.fit_transform(X_df)  # 对 DataFrame 进行列选择和变换
    X_direct = ct_direct.fit_transform(X_df)  # 直接对 DataFrame 进行列变换

    assert_allclose(X_selector, X_direct)  # 断言两种变换结果的近似相等性


def test_make_column_selector_error():
    selector = make_column_selector(dtype_include=np.number)  # 创建数值类型列选择器对象
    X = np.array([[0.1, 0.2]])  # 创建一个 NumPy 数组

    msg = "make_column_selector can only be applied to pandas dataframes"
    with pytest.raises(ValueError, match=msg):  # 断言捕获到特定错误信息
        selector(X)  # 尝试在非 pandas DataFrame 上应用列选择器


def test_make_column_selector_pickle():
    pd = pytest.importorskip("pandas")  # 导入 pandas 库，如果失败则跳过测试
    # 创建一个包含整数、浮点数和字符串的数据框架
    X_df = pd.DataFrame(
        {
            "col_int": np.array([0, 1, 2], dtype=int),
            "col_float": np.array([0.0, 1.0, 2.0], dtype=float),
            "col_str": ["one", "two", "three"],
        },
        columns=["col_int", "col_float", "col_str"],  # 指定数据框架的列顺序
    )
    
    # 使用 make_column_selector 函数创建一个选择器，选择包含对象类型的列
    selector = make_column_selector(dtype_include=[object])
    
    # 使用 pickle.dumps 序列化 selector 对象，并通过 pickle.loads 反序列化得到 selector_picked
    selector_picked = pickle.loads(pickle.dumps(selector))
    
    # 断言两个选择器对于 X_df 数据框架返回的结果是相等的
    assert_array_equal(selector(X_df), selector_picked(X_df))
@pytest.mark.parametrize(
    "empty_col",
    [[], np.array([], dtype=int), lambda x: []],
    ids=["list", "array", "callable"],
)
# 定义测试函数，用于测试处理空列时的特征名称生成
def test_feature_names_empty_columns(empty_col):
    pd = pytest.importorskip("pandas")
    
    # 创建一个包含两列的 Pandas 数据帧
    df = pd.DataFrame({"col1": ["a", "a", "b"], "col2": ["z", "z", "z"]})
    
    # 创建 ColumnTransformer 对象，包含两个转换器：OneHotEncoder 对象以及处理空列的 OneHotEncoder 对象
    ct = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(), ["col1", "col2"]),
            ("empty_features", OneHotEncoder(), empty_col),
        ],
    )
    
    # 对数据帧进行拟合
    ct.fit(df)
    
    # 断言生成的特征名称列表是否符合预期
    assert_array_equal(
        ct.get_feature_names_out(), ["ohe__col1_a", "ohe__col1_b", "ohe__col2_z"]
    )


@pytest.mark.parametrize(
    "selector",
    [
        [1],
        lambda x: [1],
        ["col2"],
        lambda x: ["col2"],
        [False, True],
        lambda x: [False, True],
    ],
)
# 定义测试函数，用于测试在 Pandas 数据帧中选择特定列时的特征名称生成
def test_feature_names_out_pandas(selector):
    """Checks name when selecting only the second column"""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"col1": ["a", "a", "b"], "col2": ["z", "z", "z"]})
    
    # 创建 ColumnTransformer 对象，只包含一个转换器：OneHotEncoder 对象
    ct = ColumnTransformer([("ohe", OneHotEncoder(), selector)])
    
    # 对数据帧进行拟合
    ct.fit(df)
    
    # 断言生成的特征名称列表是否符合预期
    assert_array_equal(ct.get_feature_names_out(), ["ohe__col2_z"])


@pytest.mark.parametrize(
    "selector", [[1], lambda x: [1], [False, True], lambda x: [False, True]]
)
# 定义测试函数，用于测试在非 Pandas 数据帧中选择特定列时的特征名称生成
def test_feature_names_out_non_pandas(selector):
    """Checks name when selecting the second column with numpy array"""
    X = [["a", "z"], ["a", "z"], ["b", "z"]]
    
    # 创建 ColumnTransformer 对象，只包含一个转换器：OneHotEncoder 对象
    ct = ColumnTransformer([("ohe", OneHotEncoder(), selector)])
    
    # 对数据进行拟合
    ct.fit(X)
    
    # 断言生成的特征名称列表是否符合预期
    assert_array_equal(ct.get_feature_names_out(), ["ohe__x1_z"])


@pytest.mark.parametrize("remainder", ["passthrough", StandardScaler()])
# 定义测试函数，测试在 ColumnTransformer 中使用 'remainder' 参数时的可视化展示
def test_sk_visual_block_remainder(remainder):
    # remainder='passthrough' or an estimator will be shown in repr_html
    ohe = OneHotEncoder()
    
    # 创建 ColumnTransformer 对象，包含一个 OneHotEncoder 转换器以及可能的 'remainder' 参数
    ct = ColumnTransformer(
        transformers=[("ohe", ohe, ["col1", "col2"])], remainder=remainder
    )
    
    # 获取 ColumnTransformer 对象的可视化展示块
    visual_block = ct._sk_visual_block_()
    
    # 断言展示块的属性是否符合预期
    assert visual_block.names == ("ohe", "remainder")
    assert visual_block.name_details == (["col1", "col2"], "")
    assert visual_block.estimators == (ohe, remainder)


def test_sk_visual_block_remainder_drop():
    # remainder='drop' is not shown in repr_html
    ohe = OneHotEncoder()
    
    # 创建 ColumnTransformer 对象，只包含一个 OneHotEncoder 转换器
    ct = ColumnTransformer(transformers=[("ohe", ohe, ["col1", "col2"])])
    
    # 获取 ColumnTransformer 对象的可视化展示块
    visual_block = ct._sk_visual_block_()
    
    # 断言展示块的属性是否符合预期
    assert visual_block.names == ("ohe",)
    assert visual_block.name_details == (["col1", "col2"],)
    assert visual_block.estimators == (ohe,)


@pytest.mark.parametrize("remainder", ["passthrough", StandardScaler()])
# 定义测试函数，测试在 ColumnTransformer 中使用 'remainder' 参数时的可视化展示（使用 Pandas 数据帧）
def test_sk_visual_block_remainder_fitted_pandas(remainder):
    # Remainder shows the columns after fitting
    pd = pytest.importorskip("pandas")
    ohe = OneHotEncoder()
    
    # 创建 ColumnTransformer 对象，包含一个 OneHotEncoder 转换器以及可能的 'remainder' 参数
    ct = ColumnTransformer(
        transformers=[("ohe", ohe, ["col1", "col2"])],
        remainder=remainder,
        force_int_remainder_cols=False,
    )
    # 创建一个包含四列的 Pandas 数据帧（DataFrame），每列有三个元素
    df = pd.DataFrame(
        {
            "col1": ["a", "b", "c"],    # 第一列名为 "col1"，包含字符串 "a", "b", "c"
            "col2": ["z", "z", "z"],    # 第二列名为 "col2"，包含字符串 "z", "z", "z"
            "col3": [1, 2, 3],          # 第三列名为 "col3"，包含整数 1, 2, 3
            "col4": [3, 4, 5],          # 第四列名为 "col4"，包含整数 3, 4, 5
        }
    )
    # 使用 ColumnTransformer 对象 ct 拟合数据帧 df
    ct.fit(df)
    # 从 ColumnTransformer 对象 ct 中获取可视化信息块
    visual_block = ct._sk_visual_block_()
    # 断言 visual_block 的 names 属性为 ("ohe", "remainder")
    assert visual_block.names == ("ohe", "remainder")
    # 断言 visual_block 的 name_details 属性为 (["col1", "col2"], ["col3", "col4"])
    assert visual_block.name_details == (["col1", "col2"], ["col3", "col4"])
    # 断言 visual_block 的 estimators 属性为 (ohe, remainder)，ohe 和 remainder 是预先定义好的变量
    assert visual_block.estimators == (ohe, remainder)
@pytest.mark.parametrize("remainder", ["passthrough", StandardScaler()])
# 使用 pytest 提供的参数化功能，remainder 参数可以是 "passthrough" 或 StandardScaler() 对象
def test_sk_visual_block_remainder_fitted_numpy(remainder):
    # Remainder shows the indices after fitting
    # 创建一个 2x3 的 numpy 数组作为输入数据 X
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    # 创建一个 StandardScaler() 的实例
    scaler = StandardScaler()
    # 创建一个 ColumnTransformer 实例 ct，应用 scaler 到索引为 0 和 2 的列上，剩余的列根据 remainder 参数处理
    ct = ColumnTransformer(
        transformers=[("scale", scaler, [0, 2])], remainder=remainder
    )
    # 对输入数据 X 进行拟合
    ct.fit(X)
    # 获取 ColumnTransformer 内部的可视化块对象 visual_block
    visual_block = ct._sk_visual_block_()
    # 断言 visual_block 的 names 属性是否为 ("scale", "remainder")
    assert visual_block.names == ("scale", "remainder")
    # 断言 visual_block 的 name_details 属性是否为 ([0, 2], [1])
    assert visual_block.name_details == ([0, 2], [1])
    # 断言 visual_block 的 estimators 属性是否为 (scaler, remainder)
    assert visual_block.estimators == (scaler, remainder)


@pytest.mark.parametrize("explicit_colname", ["first", "second", 0, 1])
@pytest.mark.parametrize("remainder", [Trans(), "passthrough", "drop"])
# 使用 pytest 的参数化功能，explicit_colname 参数可以是 "first", "second", 0, 1 中的一个，
# remainder 参数可以是 Trans() 对象，"passthrough" 或 "drop"
def test_column_transformer_reordered_column_names_remainder(
    explicit_colname, remainder
):
    """Test the interaction between remainder and column transformer"""
    pd = pytest.importorskip("pandas")
    
    # 创建一个 2x3 的 numpy 数组 X_fit_array，并转换为 DataFrame X_fit_df，列名为 ["first", "second"]
    X_fit_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_fit_df = pd.DataFrame(X_fit_array, columns=["first", "second"])
    
    # 创建一个 2x3 的 numpy 数组 X_trans_array，并转换为 DataFrame X_trans_df，列名为 ["second", "first"]
    X_trans_array = np.array([[2, 4, 6], [0, 1, 2]]).T
    X_trans_df = pd.DataFrame(X_trans_array, columns=["second", "first"])

    # 创建一个 ColumnTransformer 实例 tf，应用 Trans() 到 explicit_colname 指定的列上，剩余的列根据 remainder 参数处理
    tf = ColumnTransformer([("bycol", Trans(), explicit_colname)], remainder=remainder)

    # 对 X_fit_df 进行拟合
    tf.fit(X_fit_df)
    # 对 X_fit_df 进行变换，保存结果到 X_fit_trans
    X_fit_trans = tf.transform(X_fit_df)

    # 改变列顺序后仍然有效
    # 对 X_trans_df 进行变换，保存结果到 X_trans
    X_trans = tf.transform(X_trans_df)
    # 断言 X_trans 与 X_fit_trans 的值是否接近
    assert_allclose(X_trans, X_fit_trans)

    # 忽略额外的列
    # 复制 X_fit_df 并添加额外的列 "third"
    X_extended_df = X_fit_df.copy()
    X_extended_df["third"] = [3, 6]
    # 对 X_extended_df 进行变换，保存结果到 X_trans
    X_trans = tf.transform(X_extended_df)
    # 断言 X_trans 与 X_fit_trans 的值是否接近
    assert_allclose(X_trans, X_fit_trans)

    if isinstance(explicit_colname, str):
        # 如果列名是字符串，则抛出 ValueError，因为输入必须用位置来指定列而不是名称
        X_array = X_fit_array.copy()
        err_msg = "Specifying the columns"
        with pytest.raises(ValueError, match=err_msg):
            tf.transform(X_array)


def test_feature_name_validation_missing_columns_drop_passthough():
    """Test the interaction between {'drop', 'passthrough'} and
    missing column names."""
    pd = pytest.importorskip("pandas")
    
    # 创建一个 3x4 的全为 1 的 numpy 数组 X，转换为 DataFrame df，列名为 ["a", "b", "c", "d"]
    X = np.ones(shape=(3, 4))
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    # 删除 df 中的 "c" 列，保存到 df_dropped
    df_dropped = df.drop("c", axis=1)

    # 当 remainder='passthrough' 时，transform 中需要包含所有在 fit 过程中看到的列
    tf = ColumnTransformer([("bycol", Trans(), [1])], remainder="passthrough")
    tf.fit(df)
    msg = r"columns are missing: {'c'}"
    # 断言调用 transform(df_dropped) 时抛出 ValueError，错误信息中包含 "columns are missing: {'c'}"
    with pytest.raises(ValueError, match=msg):
        tf.transform(df_dropped)

    # 当 remainder='drop' 时，允许在 transform 中缺少 "c" 列
    tf = ColumnTransformer([("bycol", Trans(), [1])], remainder="drop")
    tf.fit(df)

    # 对 df_dropped 进行 transform，保存结果到 df_dropped_trans
    df_dropped_trans = tf.transform(df_dropped)
    # 对 df 进行 transform，保存结果到 df_fit_trans
    df_fit_trans = tf.transform(df)
    # 断言 df_dropped_trans 与 df_fit_trans 的值是否接近
    assert_allclose(df_dropped_trans, df_fit_trans)
    # 创建一个ColumnTransformer对象tf，用于处理DataFrame的列转换操作
    # "bycol"指定转换的名称，"drop"表示丢弃指定的列名列表["c"]
    # 由于指定的列名"c"在DataFrame中可能不存在，所以允许出现列名缺失的情况
    tf = ColumnTransformer([("bycol", "drop", ["c"])], remainder="passthrough")
    
    # 使用创建好的ColumnTransformer对象tf对DataFrame df进行拟合操作
    tf.fit(df)
    
    # 对已经丢弃了指定列的DataFrame df_dropped进行转换操作，生成转换后的DataFrame df_dropped_trans
    df_dropped_trans = tf.transform(df_dropped)
    
    # 对原始的DataFrame df进行转换操作，生成转换后的DataFrame df_fit_trans
    df_fit_trans = tf.transform(df)
    
    # 断言两个转换后的DataFrame df_dropped_trans和df_fit_trans非常接近
    assert_allclose(df_dropped_trans, df_fit_trans)
def test_feature_names_in_():
    """Feature names are stored in column transformer.

    Column transformer deliberately does not check for column name consistency.
    It only checks that the non-dropped names seen in `fit` are seen
    in `transform`. This behavior is already tested in
    `test_feature_name_validation_missing_columns_drop_passthough`"""

    pd = pytest.importorskip("pandas")  # 导入 pytest 库，并跳过如果导入失败的话

    feature_names = ["a", "c", "d"]  # 定义特征名列表
    df = pd.DataFrame([[1, 2, 3]], columns=feature_names)  # 创建包含特征名的 DataFrame
    ct = ColumnTransformer([("bycol", Trans(), ["a", "d"])], remainder="passthrough")  # 创建 ColumnTransformer 对象

    ct.fit(df)  # 使用 DataFrame 进行拟合

    assert_array_equal(ct.feature_names_in_, feature_names)  # 断言特征名是否与预期一致
    assert isinstance(ct.feature_names_in_, np.ndarray)  # 断言特征名类型为 ndarray
    assert ct.feature_names_in_.dtype == object  # 断言特征名数组的 dtype 为 object


class TransWithNames(Trans):
    def __init__(self, feature_names_out=None):
        self.feature_names_out = feature_names_out

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out is not None:
            return np.asarray(self.feature_names_out, dtype=object)
        return input_features


@pytest.mark.parametrize(
    "transformers, remainder, expected_names",
    ],  # 参数化测试的起始部分

)
def test_verbose_feature_names_out_true(transformers, remainder, expected_names):
    """Check feature_names_out for verbose_feature_names_out=True (default)"""

    pd = pytest.importorskip("pandas")  # 导入 pytest 库，并跳过如果导入失败的话
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"])  # 创建包含列名的 DataFrame
    ct = ColumnTransformer(
        transformers,
        remainder=remainder,
    )  # 创建 ColumnTransformer 对象

    ct.fit(df)  # 使用 DataFrame 进行拟合

    names = ct.get_feature_names_out()  # 获取输出特征名
    assert isinstance(names, np.ndarray)  # 断言输出特征名类型为 ndarray
    assert names.dtype == object  # 断言输出特征名数组的 dtype 为 object
    assert_array_equal(names, expected_names)  # 断言输出特征名是否与预期一致
    [
        (
            [
                ("bycol1", TransWithNames(), ["d", "c"]),
                ("bycol2", "passthrough", ["a"]),
            ],
            "passthrough",
            ["d", "c", "a", "b"],
        ),
        (
            [
                ("bycol1", TransWithNames(["a"]), ["d", "c"]),
                ("bycol2", "passthrough", ["d"]),
            ],
            "drop",
            ["a", "d"],
        ),
        (
            [
                ("bycol1", TransWithNames(), ["b"]),
                ("bycol2", "drop", ["d"]),
            ],
            "passthrough",
            ["b", "a", "c"],
        ),
        (
            [
                ("bycol1", TransWithNames(["pca1", "pca2"]), ["a", "b", "d"]),
            ],
            "passthrough",
            ["pca1", "pca2", "c"],
        ),
        (
            [
                ("bycol1", TransWithNames(["a", "c"]), ["d"]),
                ("bycol2", "passthrough", ["d"]),
            ],
            "drop",
            ["a", "c", "d"],
        ),
        (
            [
                ("bycol1", TransWithNames([f"pca{i}" for i in range(2)]), ["b"]),
                ("bycol2", TransWithNames([f"kpca{i}" for i in range(2)]), ["b"]),
            ],
            "passthrough",
            ["pca0", "pca1", "kpca0", "kpca1", "a", "c", "d"],
        ),
        (
            [
                ("bycol1", "drop", ["d"]),
            ],
            "drop",
            [],
        ),
        (
            [
                ("bycol1", TransWithNames(), slice(1, 2)),
                ("bycol2", "drop", ["d"]),
            ],
            "passthrough",
            ["b", "a", "c"],
        ),
        (
            [
                ("bycol1", TransWithNames(), ["b"]),
                ("bycol2", "drop", slice(3, 4)),
            ],
            "passthrough",
            ["b", "a", "c"],
        ),
        (
            [
                ("bycol1", TransWithNames(), ["d", "c"]),
                ("bycol2", "passthrough", slice(0, 2)),
            ],
            "drop",
            ["d", "c", "a", "b"],
        ),
        (
            [
                ("bycol1", TransWithNames(), slice("a", "b")),
                ("bycol2", "drop", ["d"]),
            ],
            "passthrough",
            ["a", "b", "c"],
        ),
        (
            [
                ("bycol1", TransWithNames(), ["b"]),
                ("bycol2", "drop", slice("c", "d")),
            ],
            "passthrough",
            ["b", "a"],
        ),
        (
            [
                ("bycol1", TransWithNames(), ["d", "c"]),
                ("bycol2", "passthrough", slice("a", "b")),
            ],
            "drop",
            ["d", "c", "a", "b"],
        ),
        (
            [
                ("bycol1", TransWithNames(), ["d", "c"]),
                ("bycol2", "passthrough", slice("b", "b")),
            ],
            "drop",
            ["d", "c", "b"],
        ),
    ]
)
# 定义一个测试函数，用于验证 verbose_feature_names_out=False 时的特征输出名称
def test_verbose_feature_names_out_false(transformers, remainder, expected_names):
    """Check feature_names_out for verbose_feature_names_out=False"""
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个包含一行数据的 DataFrame，列名为 ["a", "b", "c", "d"]
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"])
    # 创建 ColumnTransformer 对象，指定 transformers、remainder 和 verbose_feature_names_out=False
    ct = ColumnTransformer(
        transformers,
        remainder=remainder,
        verbose_feature_names_out=False,
    )
    # 对 DataFrame 进行拟合
    ct.fit(df)

    # 获取拟合后的特征名称输出
    names = ct.get_feature_names_out()
    # 断言输出的名称类型为 ndarray
    assert isinstance(names, np.ndarray)
    # 断言输出的名称数据类型为 object
    assert names.dtype == object
    # 断言输出的名称与预期名称数组相等
    assert_array_equal(names, expected_names)


# 使用 pytest 的 parametrize 装饰器，为下面的测试参数化设置
@pytest.mark.parametrize(
    "transformers, remainder, colliding_columns",
    [
        (
            # 第一个元组：定义了第一个转换流水线
            [
                ("bycol1", TransWithNames(), ["b"]),  # 第一个步骤：使用默认的 TransWithNames 转换，处理列 'b'
                ("bycol2", "passthrough", ["b"]),    # 第二个步骤：直接传递（不进行转换），处理列 'b'
            ],
            "drop",         # 在第一个转换流水线之后，丢弃列 'b'
            "['b']",        # 指明要丢弃的列名，这里是 'b'
        ),
        (
            # 第二个元组：定义了第二个转换流水线
            [
                ("bycol1", TransWithNames(["c", "d"]), ["c"]),  # 第一个步骤：使用自定义 TransWithNames 转换，处理列 'c'
                ("bycol2", "passthrough", ["c"]),              # 第二个步骤：直接传递，处理列 'c'
            ],
            "drop",         # 在第二个转换流水线之后，丢弃列 'c'
            "['c']",        # 指明要丢弃的列名，这里是 'c'
        ),
        (
            # 第三个元组：定义了第三个转换流水线
            [
                ("bycol1", TransWithNames(["a"]), ["b"]),     # 第一个步骤：使用自定义 TransWithNames 转换，处理列 'b'
                ("bycol2", "passthrough", ["b"]),            # 第二个步骤：直接传递，处理列 'b'
            ],
            "passthrough",  # 在第三个转换流水线之后，保留列 'a'（直接传递）
            "['a']",        # 指明要保留的列名，这里是 'a'
        ),
        (
            # 第四个元组：定义了第四个转换流水线
            [
                ("bycol1", TransWithNames(["a"]), ["b"]),     # 第一个步骤：使用自定义 TransWithNames 转换，处理列 'b'
                ("bycol2", "drop", ["b"]),                   # 第二个步骤：丢弃列 'b'
            ],
            "passthrough",  # 在第四个转换流水线之后，保留列 'a'（直接传递）
            "['a']",        # 指明要保留的列名，这里是 'a'
        ),
        (
            # 第五个元组：定义了第五个转换流水线
            [
                ("bycol1", TransWithNames(["c", "b"]), ["b"]),   # 第一个步骤：使用自定义 TransWithNames 转换，处理列 'b'
                ("bycol2", "passthrough", ["c", "b"]),          # 第二个步骤：直接传递，处理列 'c' 和 'b'
            ],
            "drop",         # 在第五个转换流水线之后，丢弃列 'b' 和 'c'
            "['b', 'c']",   # 指明要丢弃的列名，这里是 'b' 和 'c'
        ),
        (
            # 第六个元组：定义了第六个转换流水线
            [
                ("bycol1", TransWithNames(["a"]), ["b"]),     # 第一个步骤：使用自定义 TransWithNames 转换，处理列 'b'
                ("bycol2", "passthrough", ["a"]),            # 第二个步骤：直接传递，处理列 'a'
                ("bycol3", TransWithNames(["a"]), ["b"]),     # 第三个步骤：使用自定义 TransWithNames 转换，处理列 'b'
            ],
            "passthrough",  # 在第六个转换流水线之后，保留列 'a'（直接传递）
            "['a']",        # 指明要保留的列名，这里是 'a'
        ),
        (
            # 第七个元组：定义了第七个转换流水线
            [
                ("bycol1", TransWithNames(["a", "b"]), ["b"]),   # 第一个步骤：使用自定义 TransWithNames 转换，处理列 'b'
                ("bycol2", "passthrough", ["a"]),                # 第二个步骤：直接传递，处理列 'a'
                ("bycol3", TransWithNames(["b"]), ["c"]),        # 第三个步骤：使用自定义 TransWithNames 转换，处理列 'c'
            ],
            "passthrough",  # 在第七个转换流水线之后，保留列 'a' 和 'b'（直接传递）
            "['a', 'b']",   # 指明要保留的列名，这里是 'a' 和 'b'
        ),
        (
            # 第八个元组：定义了第八个转换流水线
            [
                ("bycol1", TransWithNames([f"pca{i}" for i in range(6)]), ["b"]),   # 第一个步骤：使用自定义 TransWithNames 转换，处理列 'b'
                ("bycol2", TransWithNames([f"pca{i}" for i in range(6)]), ["b"]),   # 第二个步骤：使用自定义 TransWithNames 转换，处理列 'b'
            ],
            "passthrough",  # 在第八个转换流水线之后，保留所有以 'pca' 开头的列（直接传递）
            "['pca0', 'pca1', 'pca2', 'pca3', 'pca4', ...]",   # 指明要保留的列名，这里是所有以 'pca' 开头的列
        ),
        (
            # 第九个元组：定义了第九个转换流水线
            [
                ("bycol1", TransWithNames(["a", "b"]), slice(1, 2)),   # 第一个步骤：使用自定义 TransWithNames 转换，处理列 'b'，从索引 1 到 2 的切片
                ("bycol2", "passthrough", ["a"]),                     # 第二个步骤：直接传递，处理列 'a'
                ("bycol3", TransWithNames(["b"]), ["c"]),             # 第三个步骤：使用自定义 TransWithNames 转换，处理列 'c'
            ],
            "passthrough",  # 在第九个转换流水线之后，保留列 'a' 和 'b'（直接传递）
            "['a', 'b']",   # 指明要保留的列名，这里是 'a' 和 'b'
        ),
        (
            # 第十个元组：定义了第十个转换流水线
            [
                ("bycol1", TransWithNames(["a", "b"]), ["b"]),         # 第一个步骤：使用自定义 TransWithNames 转换，处理列 'b'
                ("bycol2", "passthrough", slice(0, 1)),               # 第二个步骤：直接传递，处理从索引 0 到 1 的切片
                ("bycol3", TransWithNames(["b"]), ["c"]),             # 第三个步骤：使用自定义 TransWithNames 转换，处理列 'c'
            ],
            "passthrough",  # 在第十个转换流水线之后，保留列 'a' 和 'b'（直接传递）
            "['a', 'b']",   # 指明要保留的列名，这里是 'a' 和 'b'
        ),
        (
            # 第十一个元组：定义了第十一个转换流水线
            [
                ("bycol1", TransWithNames(["a", "b"]), ["b"]),         # 第一个步骤：使用自定义 TransWithNames 转换，处理列 'b'
                ("bycol2", "passthrough", slice("b", "c")),           # 第二个步骤：直接传递，处理从 'b' 到 'c' 的切片
                ("bycol3", TransWithNames(["b"]), ["c"]),             # 第
def test_verbose_feature_names_out_false_errors(
    transformers, remainder, colliding_columns
):
    """Check feature_names_out for verbose_feature_names_out=False"""

    # 导入 pytest 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    
    # 创建一个包含单行数据的 DataFrame
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"])
    
    # 创建 ColumnTransformer 对象，设置 verbose_feature_names_out=False
    ct = ColumnTransformer(
        transformers,
        remainder=remainder,
        verbose_feature_names_out=False,
    )
    
    # 对 DataFrame 进行拟合
    ct.fit(df)
    
    # 设置错误信息的正则表达式消息
    msg = re.escape(
        f"Output feature names: {colliding_columns} are not unique. Please set "
        "verbose_feature_names_out=True to add prefixes to feature names"
    )
    
    # 断言引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        ct.get_feature_names_out()


@pytest.mark.parametrize("verbose_feature_names_out", [True, False])
@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
def test_column_transformer_set_output(verbose_feature_names_out, remainder):
    """Check column transformer behavior with set_output."""
    
    # 导入 pytest 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    
    # 创建一个包含单行数据的 DataFrame，指定列名和索引
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"], index=[10])
    
    # 创建 ColumnTransformer 对象，根据参数设置 verbose_feature_names_out
    ct = ColumnTransformer(
        [("first", TransWithNames(), ["a", "c"]), ("second", TransWithNames(), ["d"])],
        remainder=remainder,
        verbose_feature_names_out=verbose_feature_names_out,
    )
    
    # 进行拟合转换，并获取转换后的 X_trans
    X_trans = ct.fit_transform(df)
    
    # 断言 X_trans 是 numpy 数组对象
    assert isinstance(X_trans, np.ndarray)
    
    # 设置输出为 pandas 格式
    ct.set_output(transform="pandas")
    
    # 创建另一个 DataFrame 用于测试
    df_test = pd.DataFrame([[1, 2, 3, 4]], columns=df.columns, index=[20])
    
    # 再次进行转换，并获取转换后的 X_trans
    X_trans = ct.transform(df_test)
    
    # 断言 X_trans 是 pandas DataFrame 对象
    assert isinstance(X_trans, pd.DataFrame)
    
    # 获取特征名输出，并断言列名匹配预期
    feature_names_out = ct.get_feature_names_out()
    assert_array_equal(X_trans.columns, feature_names_out)
    assert_array_equal(X_trans.index, df_test.index)


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
@pytest.mark.parametrize("fit_transform", [True, False])
def test_column_transform_set_output_mixed(remainder, fit_transform):
    """Check ColumnTransformer outputs mixed types correctly."""
    
    # 导入 pytest 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    
    # 创建包含不同类型数据的 DataFrame
    df = pd.DataFrame(
        {
            "pet": pd.Series(["dog", "cat", "snake"], dtype="category"),
            "color": pd.Series(["green", "blue", "red"], dtype="object"),
            "age": [1.4, 2.1, 4.4],
            "height": [20, 40, 10],
            "distance": pd.Series([20, pd.NA, 100], dtype="Int32"),
        }
    )
    
    # 创建 ColumnTransformer 对象，设置 verbose_feature_names_out=False，并将输出转换为 pandas 格式
    ct = ColumnTransformer(
        [
            (
                "color_encode",
                OneHotEncoder(sparse_output=False, dtype="int8"),
                ["color"],
            ),
            ("age", StandardScaler(), ["age"]),
        ],
        remainder=remainder,
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    
    # 根据条件选择拟合或拟合转换数据，并获取转换后的 X_trans
    if fit_transform:
        X_trans = ct.fit_transform(df)
    else:
        X_trans = ct.fit(df).transform(df)
    
    # 断言 X_trans 是 pandas DataFrame 对象
    assert isinstance(X_trans, pd.DataFrame)
    
    # 获取特征名输出，并断言列名匹配预期
    assert_array_equal(X_trans.columns, ct.get_feature_names_out())
    # 定义期望的数据类型字典，指定每个列名对应的预期数据类型
    expected_dtypes = {
        "color_blue": "int8",       # color_blue 列应为 int8 类型
        "color_green": "int8",      # color_green 列应为 int8 类型
        "color_red": "int8",        # color_red 列应为 int8 类型
        "age": "float64",           # age 列应为 float64 类型
        "pet": "category",          # pet 列应为 category 类型
        "height": "int64",          # height 列应为 int64 类型
        "distance": "Int32",        # distance 列应为 Int32 类型
    }
    
    # 遍历数据集 X_trans 的每一列及其数据类型
    for col, dtype in X_trans.dtypes.items():
        # 断言当前列的数据类型与预期的数据类型相符
        assert dtype == expected_dtypes[col]
@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
# 使用 pytest 的参数化标记，定义测试函数 test_column_transform_set_output_after_fitting
def test_column_transform_set_output_after_fitting(remainder):
    # 导入 pytest 和 pandas 库，如果导入失败则跳过测试
    pd = pytest.importorskip("pandas")
    
    # 创建一个包含 pet（类别数据）、age（浮点数）、height（整数）列的 DataFrame
    df = pd.DataFrame(
        {
            "pet": pd.Series(["dog", "cat", "snake"], dtype="category"),
            "age": [1.4, 2.1, 4.4],
            "height": [20, 40, 10],
        }
    )
    
    # 创建 ColumnTransformer 对象 ct，其中包含两个转换器
    ct = ColumnTransformer(
        [
            (
                "color_encode",
                OneHotEncoder(sparse_output=False, dtype="int16"),
                ["pet"],  # 对 pet 列使用 OneHotEncoder
            ),
            ("age", StandardScaler(), ["age"]),  # 对 age 列使用 StandardScaler
        ],
        remainder=remainder,  # 指定 remainder 参数的值
        verbose_feature_names_out=False,  # 禁用详细特征名称输出
    )

    # 在没有调用 set_output 的情况下拟合转换器
    X_trans = ct.fit_transform(df)
    assert isinstance(X_trans, np.ndarray)  # 断言转换结果是 numpy 数组
    assert X_trans.dtype == "float64"  # 断言转换结果的数据类型为 float64

    # 调用 set_output 方法设置输出为 pandas 格式
    ct.set_output(transform="pandas")
    X_trans_df = ct.transform(df)  # 使用转换器转换 DataFrame df
    expected_dtypes = {
        "pet_cat": "int16",
        "pet_dog": "int16",
        "pet_snake": "int16",
        "height": "int64",
        "age": "float64",
    }
    
    # 断言转换后的 DataFrame 的每列数据类型与预期一致
    for col, dtype in X_trans_df.dtypes.items():
        assert dtype == expected_dtypes[col]


# 定义一个自定义转换器 PandasOutTransformer，继承自 BaseEstimator 类
# 该转换器不定义 get_feature_names_out 方法，始终期望输入为 DataFrame
class PandasOutTransformer(BaseEstimator):
    def __init__(self, offset=1.0):
        self.offset = offset

    def fit(self, X, y=None):
        pd = pytest.importorskip("pandas")
        assert isinstance(X, pd.DataFrame)  # 断言 X 是 pandas 的 DataFrame
        return self

    def transform(self, X, y=None):
        pd = pytest.importorskip("pandas")
        assert isinstance(X, pd.DataFrame)  # 断言 X 是 pandas 的 DataFrame
        return X - self.offset  # 返回减去偏移量后的 DataFrame

    def set_output(self, transform=None):
        # 该转换器无论如何都返回 DataFrame
        return self


@pytest.mark.parametrize(
    "trans_1, expected_verbose_names, expected_non_verbose_names",
    [
        (
            PandasOutTransformer(offset=2.0),
            ["trans_0__feat1", "trans_1__feat0"],
            ["feat1", "feat0"],
        ),
        (
            "drop",
            ["trans_0__feat1"],
            ["feat1"],
        ),
        (
            "passthrough",
            ["trans_0__feat1", "trans_1__feat0"],
            ["feat1", "feat0"],
        ),
    ],
)
# 使用 pytest 的参数化标记，定义测试函数 test_transformers_with_pandas_out_but_not_feature_names_out
def test_transformers_with_pandas_out_but_not_feature_names_out(
    trans_1, expected_verbose_names, expected_non_verbose_names
):
    """Check that set_config(transform="pandas") is compatible with more transformers.

    Specifically, if transformers returns a DataFrame, but does not define
    `get_feature_names_out`.
    """
    pd = pytest.importorskip("pandas")

    # 创建一个包含 feat0 和 feat1 列的 DataFrame X_df
    X_df = pd.DataFrame({"feat0": [1.0, 2.0, 3.0], "feat1": [2.0, 3.0, 4.0]})
    
    # 创建 ColumnTransformer 对象 ct，其中包含两个转换器
    ct = ColumnTransformer(
        [
            ("trans_0", PandasOutTransformer(offset=3.0), ["feat1"]),
            ("trans_1", trans_1, ["feat0"]),
        ]
    )
    
    # 使用转换器拟合和转换 DataFrame X_df
    X_trans_np = ct.fit_transform(X_df)
    # 断言验证变量 X_trans_np 是 numpy 数组类型
    assert isinstance(X_trans_np, np.ndarray)

    # 使用 pytest 来断言 ct 对象调用 get_feature_names_out 方法会抛出 AttributeError 异常，
    # 并且异常信息应包含字符串 "not provide get_feature_names_out"
    with pytest.raises(AttributeError, match="not provide get_feature_names_out"):
        ct.get_feature_names_out()

    # 设置 ct 对象的输出格式为 "pandas"，这会使得输出的特征名具有前缀，因为 verbose_feature_names_out=True 是默认值
    ct.set_output(transform="pandas")
    # 对 X_df 进行拟合和转换，得到 DataFrame 类型的 X_trans_df0
    X_trans_df0 = ct.fit_transform(X_df)
    # 断言 X_trans_df0 的列名与预期的详细命名 expected_verbose_names 相等
    assert_array_equal(X_trans_df0.columns, expected_verbose_names)

    # 设置 ct 对象的 verbose_feature_names_out 参数为 False，关闭详细特征名输出
    ct.set_params(verbose_feature_names_out=False)
    # 再次对 X_df 进行拟合和转换，得到 DataFrame 类型的 X_trans_df1
    X_trans_df1 = ct.fit_transform(X_df)
    # 断言 X_trans_df1 的列名与预期的非详细命名 expected_non_verbose_names 相等
    assert_array_equal(X_trans_df1.columns, expected_non_verbose_names)
# 使用 pytest.mark.parametrize 来定义一个参数化测试，测试不同的空选择情况
@pytest.mark.parametrize(
    "empty_selection",  # 参数化的参数，表示空选择的不同情况
    [[], np.array([False, False]), [False, False]],  # 参数化的具体值：空列表、布尔数组、布尔列表
    ids=["list", "bool", "bool_int"],  # 每个参数化值的标识符
)
def test_empty_selection_pandas_output(empty_selection):
    """Check that pandas output works when there is an empty selection.

    Non-regression test for gh-25487
    """
    pd = pytest.importorskip("pandas")  # 导入 pandas，如果导入失败则跳过测试

    X = pd.DataFrame([[1.0, 2.2], [3.0, 1.0]], columns=["a", "b"])  # 创建一个 DataFrame X
    ct = ColumnTransformer(  # 创建 ColumnTransformer 对象 ct
        [
            ("categorical", "passthrough", empty_selection),  # 第一个转换器：类别特征通过传递
            ("numerical", StandardScaler(), ["a", "b"]),  # 第二个转换器：数值特征使用 StandardScaler
        ],
        verbose_feature_names_out=True,  # 设置输出详细特征名
    )
    ct.set_output(transform="pandas")  # 设置输出格式为 pandas
    X_out = ct.fit_transform(X)  # 对 X 进行转换并获得输出 X_out
    assert_array_equal(X_out.columns, ["numerical__a", "numerical__b"])  # 断言输出列名符合预期

    ct.set_params(verbose_feature_names_out=False)  # 设置参数 verbose_feature_names_out 为 False
    X_out = ct.fit_transform(X)  # 重新转换 X 并获取输出 X_out
    assert_array_equal(X_out.columns, ["a", "b"])  # 断言输出列名符合预期


def test_raise_error_if_index_not_aligned():
    """Check column transformer raises error if indices are not aligned.

    Non-regression test for gh-26210.
    """
    pd = pytest.importorskip("pandas")  # 导入 pandas，如果导入失败则跳过测试

    X = pd.DataFrame([[1.0, 2.2], [3.0, 1.0]], columns=["a", "b"], index=[8, 3])  # 创建一个带索引的 DataFrame X
    reset_index_transformer = FunctionTransformer(  # 创建一个功能转换器 reset_index_transformer
        lambda x: x.reset_index(drop=True), feature_names_out="one-to-one"  # 使用 lambda 函数重置索引
    )

    ct = ColumnTransformer(  # 创建 ColumnTransformer 对象 ct
        [
            ("num1", "passthrough", ["a"]),  # 第一个转换器：num1，直接传递特征 "a"
            ("num2", reset_index_transformer, ["b"]),  # 第二个转换器：num2，使用 reset_index_transformer 处理特征 "b"
        ],
    )
    ct.set_output(transform="pandas")  # 设置输出格式为 pandas
    msg = (
        "Concatenating DataFrames from the transformer's output lead to"
        " an inconsistent number of samples. The output may have Pandas"
        " Indexes that do not match."
    )
    with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言捕获 ValueError 异常，消息需匹配 msg
        ct.fit_transform(X)  # 对 X 进行转换并捕获异常


def test_remainder_set_output():
    """Check that the output is set for the remainder.

    Non-regression test for #26306.
    """

    pd = pytest.importorskip("pandas")  # 导入 pandas，如果导入失败则跳过测试
    df = pd.DataFrame({"a": [True, False, True], "b": [1, 2, 3]})  # 创建一个 DataFrame df

    ct = make_column_transformer(  # 创建 ColumnTransformer 对象 ct
        (VarianceThreshold(), make_column_selector(dtype_include=bool)),  # 第一个转换器：VarianceThreshold 选择布尔类型特征
        remainder=VarianceThreshold(),  # 设置剩余特征使用 VarianceThreshold
        verbose_feature_names_out=False,  # 设置输出详细特征名为 False
    )
    ct.set_output(transform="pandas")  # 设置输出格式为 pandas

    out = ct.fit_transform(df)  # 对 df 进行转换并获得输出 out
    pd.testing.assert_frame_equal(out, df)  # 使用 pandas 断言确保输出 out 与 df 相等

    ct.set_output(transform="default")  # 设置输出格式为默认
    out = ct.fit_transform(df)  # 重新转换 df 并获取输出 out
    assert isinstance(out, np.ndarray)  # 断言输出类型为 numpy 数组


# TODO(1.6): replace the warning by a ValueError exception
def test_transform_pd_na():
    """Check behavior when a tranformer's output contains pandas.NA

    It should emit a warning unless the output config is set to 'pandas'.
    """
    pd = pytest.importorskip("pandas")  # 导入 pandas，如果导入失败则跳过测试
    if not hasattr(pd, "Float64Dtype"):  # 检查 pandas 是否具有 Float64Dtype 扩展类型
        pytest.skip(
            "The issue with pd.NA tested here does not happen in old versions that do"
            " not have the extension dtypes"
        )
    df = pd.DataFrame({"a": [1.5, None]})  # 创建一个包含 NA 值的 DataFrame df
    # 创建一个列转换器对象 `ct`，将列 'a' 的数据不做任何转换直接通过
    ct = make_column_transformer(("passthrough", ["a"]))
    
    # 使用警告捕获机制，设置警告过滤器为错误级别，确保不会因为非扩展数据类型或 np.nan 而发出警告
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 对 DataFrame `df` 进行列转换器 `ct` 的拟合与转换操作
        ct.fit_transform(df)
    
    # 将 DataFrame `df` 中的数据类型转换为最合适的数据类型
    df = df.convert_dtypes()
    
    # 使用 pytest 的警告检测功能，预期发出 FutureWarning 警告，并匹配指定正则表达式的警告信息
    with pytest.warns(FutureWarning, match=r"set_output\(transform='pandas'\)"):
        # 对 DataFrame `df` 进行列转换器 `ct` 的拟合与转换操作，此时预期会有警告输出
        ct.fit_transform(df)
    
    # 使用警告捕获机制，设置警告过滤器为错误级别，确保不会因为输出设置为 pandas 而发出警告
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 将列转换器 `ct` 的输出设置为 pandas 格式，然后对 DataFrame `df` 进行拟合与转换操作
        ct.set_output(transform="pandas")
        ct.fit_transform(df)
    
    # 将列转换器 `ct` 的输出设置为默认格式
    ct.set_output(transform="default")
    
    # 使用警告捕获机制，设置警告过滤器为错误级别，确保不会因为 DataFrame `df` 中不存在 pd.NA 而发出警告
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 对 DataFrame `df` 进行列转换器 `ct` 的拟合与转换操作，此时 DataFrame 中已填充 -1.0 以替换 pd.NA
        ct.fit_transform(df.fillna(-1.0))
# 检查在 pandas 和 polars 数据框上的拟合和转换
def test_dataframe_different_dataframe_libraries():
    pd = pytest.importorskip("pandas")  # 导入并跳过如果不存在的话
    pl = pytest.importorskip("polars")  # 导入并跳过如果不存在的话
    X_train_np = np.array([[0, 1], [2, 4], [4, 5]])  # 创建一个 NumPy 数组
    X_test_np = np.array([[1, 2], [1, 3], [2, 3]])  # 创建另一个 NumPy 数组

    # 在 pandas 数据框上拟合，然后在 polars 数据框上进行转换
    X_train_pd = pd.DataFrame(X_train_np, columns=["a", "b"])  # 创建 pandas 数据框
    X_test_pl = pl.DataFrame(X_test_np, schema=["a", "b"])  # 创建 polars 数据框

    ct = make_column_transformer((Trans(), [0, 1]))  # 创建列变换器对象
    ct.fit(X_train_pd)  # 在 pandas 数据框上拟合列变换器

    out_pl_in = ct.transform(X_test_pl)  # 使用拟合好的变换器在 polars 数据框上进行转换
    assert_array_equal(out_pl_in, X_test_np)  # 断言转换结果与预期的 NumPy 数组相等

    # 在 polars 数据框上拟合，然后在 pandas 数据框上进行转换
    X_train_pl = pl.DataFrame(X_train_np, schema=["a", "b"])  # 创建 polars 数据框
    X_test_pd = pd.DataFrame(X_test_np, columns=["a", "b"])  # 创建 pandas 数据框
    ct.fit(X_train_pl)  # 在 polars 数据框上拟合列变换器

    out_pd_in = ct.transform(X_test_pd)  # 使用拟合好的变换器在 pandas 数据框上进行转换
    assert_array_equal(out_pd_in, X_test_np)  # 断言转换结果与预期的 NumPy 数组相等


# 检查 ColumnTransformer 的 __getitem__ 方法
def test_column_transformer__getitem__():
    X = np.array([[0, 1, 2], [3, 4, 5]])  # 创建一个 NumPy 数组
    ct = ColumnTransformer([("t1", Trans(), [0, 1]), ("t2", Trans(), [1, 2])])  # 创建列变换器对象

    msg = "ColumnTransformer is subscriptable after it is fitted"
    with pytest.raises(TypeError, match=msg):
        ct["t1"]  # 尝试获取未拟合时的列变换器项，预期引发 TypeError 异常

    ct.fit(X)  # 在 NumPy 数组 X 上拟合列变换器

    assert ct["t1"] is ct.named_transformers_["t1"]  # 断言获取的 "t1" 变换器等于命名变换器中的 "t1"
    assert ct["t2"] is ct.named_transformers_["t2"]  # 断言获取的 "t2" 变换器等于命名变换器中的 "t2"

    msg = "'does_not_exist' is not a valid transformer name"
    with pytest.raises(KeyError, match=msg):
        ct["does_not_exist"]  # 尝试获取不存在的变换器项，预期引发 KeyError 异常


# 检查在 `remainder="passthrough"` 时，处理不一致命名的正确性
def test_column_transformer_remainder_passthrough_naming_consistency(transform_output):
    pd = pytest.importorskip("pandas")  # 导入并跳过如果不存在的话
    X = pd.DataFrame(np.random.randn(10, 4))  # 创建一个随机数填充的 pandas 数据框

    # 创建列变换器对象，指定一个标准缩放器和需要应用的列索引，同时保留剩余列
    preprocessor = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), [0, 1])],
        remainder="passthrough",
    ).set_output(transform=transform_output)
    X_trans = preprocessor.fit_transform(X)  # 在 pandas 数据框上拟合列变换器

    assert X_trans.shape == X.shape  # 断言转换后的数据形状与原始数据框相同

    expected_column_names = [
        "scaler__x0",
        "scaler__x1",
        "remainder__x2",
        "remainder__x3",
    ]
    if hasattr(X_trans, "columns"):
        assert X_trans.columns.tolist() == expected_column_names  # 断言转换后的列名列表与预期相同
    assert preprocessor.get_feature_names_out().tolist() == expected_column_names  # 断言输出特征名列表与预期相同


# 检查在使用 ColumnTransformer 时，正确重命名列的处理
def test_column_transformer_column_renaming(dataframe_lib):
    lib = pytest.importorskip(dataframe_lib)  # 导入并跳过如果不存在的话
    # 创建一个包含三列的数据框，每列包含相应的整数列表
    df = lib.DataFrame({"x1": [1, 2, 3], "x2": [10, 20, 30], "x3": [100, 200, 300]})

    # 创建一个列转换器对象，用于对数据框中的列进行转换
    transformer = ColumnTransformer(
        transformers=[
            # 第一个转换器，将列 "x1", "x2", "x3" 保持不变（passthrough）
            ("A", "passthrough", ["x1", "x2", "x3"]),
            # 第二个转换器，使用 FunctionTransformer 对象对列 "x1", "x2" 进行转换
            ("B", FunctionTransformer(), ["x1", "x2"]),
            # 第三个转换器，使用 StandardScaler 对象对列 "x1", "x3" 进行标准化
            ("C", StandardScaler(), ["x1", "x3"]),
            # 第四个特殊情况的转换器，使用 lambda 函数对列 "x1", "x2", "x3" 进行特殊处理，返回0列
            (
                "D",
                FunctionTransformer(lambda x: _safe_indexing(x, [], axis=1)),
                ["x1", "x2", "x3"],
            ),
        ],
        # 打印转换后的特征名称
        verbose_feature_names_out=True,
    ).set_output(transform=dataframe_lib)
    
    # 对数据框 df 应用转换器并获取转换后的数据框
    df_trans = transformer.fit_transform(df)
    
    # 检查转换后的数据框列名是否与指定的列表相同
    assert list(df_trans.columns) == [
        "A__x1",
        "A__x2",
        "A__x3",
        "B__x1",
        "B__x2",
        "C__x1",
        "C__x3",
    ]
# 使用 pytest.mark.parametrize 装饰器定义一个参数化测试函数，测试不同的 dataframe 库
@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
def test_column_transformer_error_with_duplicated_columns(dataframe_lib):
    """Check that we raise an error when using `ColumnTransformer` and
    the columns names are duplicated between transformers."""
    # 动态导入指定的 dataframe 库，如果库不存在则跳过测试
    lib = pytest.importorskip(dataframe_lib)

    # 创建一个包含三列数据的 DataFrame
    df = lib.DataFrame({"x1": [1, 2, 3], "x2": [10, 20, 30], "x3": [100, 200, 300]})

    # 定义 ColumnTransformer 对象，包含多个转换器及其对应的列
    transformer = ColumnTransformer(
        transformers=[
            ("A", "passthrough", ["x1", "x2", "x3"]),
            ("B", FunctionTransformer(), ["x1", "x2"]),
            ("C", StandardScaler(), ["x1", "x3"]),
            # 特殊情况下的转换器返回0列，例如特征选择器
            (
                "D",
                FunctionTransformer(lambda x: _safe_indexing(x, [], axis=1)),
                ["x1", "x2", "x3"],
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform=dataframe_lib)

    # 预期引发 ValueError 异常，并匹配特定的错误消息
    err_msg = re.escape(
        "Duplicated feature names found before concatenating the outputs of the "
        "transformers: ['x1', 'x2', 'x3'].\n"
        "Transformer A has conflicting columns names: ['x1', 'x2', 'x3'].\n"
        "Transformer B has conflicting columns names: ['x1', 'x2'].\n"
        "Transformer C has conflicting columns names: ['x1', 'x3'].\n"
    )
    with pytest.raises(ValueError, match=err_msg):
        transformer.fit_transform(df)


# 使用 parse_version 函数检查 joblib 版本是否满足要求，如果不满足则跳过测试
@pytest.mark.skipif(
    parse_version(joblib.__version__) < parse_version("1.3"),
    reason="requires joblib >= 1.3",
)
def test_column_transformer_auto_memmap():
    """Check that ColumnTransformer works in parallel with joblib's auto-memmapping.

    non-regression test for issue #28781
    """
    # 创建一个随机数填充的数组作为输入数据
    X = np.random.RandomState(0).uniform(size=(3, 4))

    # 创建一个标准化的 Scaler 对象
    scaler = StandardScaler(copy=False)

    # 创建 ColumnTransformer 对象，将标准化应用于指定列，使用两个工作进程
    transformer = ColumnTransformer(
        transformers=[("scaler", scaler, [0])],
        n_jobs=2,
    )

    # 使用 joblib 的 loky 并行后端，最大允许使用 1 字节的内存进行自动内存映射
    with joblib.parallel_backend("loky", max_nbytes=1):
        # 对输入数据进行拟合和转换
        Xt = transformer.fit_transform(X)

    # 断言转换后的结果与单独标准化第一列后的结果相近
    assert_allclose(Xt, StandardScaler().fit_transform(X[:, [0]]))


# Metadata Routing Tests
# ======================


# 使用 pytest.mark.parametrize 装饰器定义一个参数化测试函数，测试不同的方法（transform, fit_transform, fit）
@pytest.mark.parametrize("method", ["transform", "fit_transform", "fit"])
def test_routing_passed_metadata_not_supported(method):
    """Test that the right error message is raised when metadata is passed while
    not supported when `enable_metadata_routing=False`."""

    # 创建一个简单的二维数组作为输入特征
    X = np.array([[0, 1, 2], [2, 4, 6]]).T
    y = [1, 2, 3]

    # 创建一个 ColumnTransformer 对象，包含一个自定义转换器并对第一列进行拟合
    trs = ColumnTransformer([("trans", Trans(), [0])]).fit(X, y)

    # 预期引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(
        ValueError, match="is only supported if enable_metadata_routing=True"
    ):
        # 调用 ColumnTransformer 对象的指定方法，传递额外的 sample_weight 和 prop 参数
        getattr(trs, method)([[1]], sample_weight=[1], prop="a")


# 使用 pytest.mark.usefixtures 装饰器确保在测试中启用 slep006（假设此处 slep006 是一个 fixture）
@pytest.mark.usefixtures("enable_slep006")
@pytest.mark.parametrize("method", ["transform", "fit_transform", "fit"])
def test_metadata_routing_for_column_transformer(method):
    """Test that metadata is routed correctly for column transformer."""
    # 创建一个 NumPy 数组 X，包含两个样本和三个特征，然后进行转置
    X = np.array([[0, 1, 2], [2, 4, 6]]).T
    # 创建一个包含目标值 y 的列表
    y = [1, 2, 3]
    # 创建一个 _Registry 类的实例，用于记录转换器
    registry = _Registry()
    # 定义样本权重 sample_weight 和元数据 metadata
    sample_weight, metadata = [1], "a"
    # 创建 ColumnTransformer 对象 trs，其中包含一个名为 "trans" 的转换器
    trs = ColumnTransformer(
        [
            (
                "trans",
                # 使用 ConsumingTransformer 类创建转换器实例，设置其需要的参数
                ConsumingTransformer(registry=registry)
                .set_fit_request(sample_weight=True, metadata=True)
                .set_transform_request(sample_weight=True, metadata=True),
                # 指定应用此转换器的列索引
                [0],
            )
        ]
    )

    # 根据 method 的值选择执行不同的操作
    if method == "transform":
        # 对 trs 执行拟合操作，使用 X、y、sample_weight 和 metadata 作为参数
        trs.fit(X, y, sample_weight=sample_weight, metadata=metadata)
        # 对 trs 执行转换操作，使用 X、sample_weight 和 metadata 作为参数
        trs.transform(X, sample_weight=sample_weight, metadata=metadata)
    else:
        # 根据 method 的值调用 trs 的相应方法，使用 X、y、sample_weight 和 metadata 作为参数
        getattr(trs, method)(X, y, sample_weight=sample_weight, metadata=metadata)

    # 断言 registry 中记录的转换器数量不为零
    assert len(registry)
    # 遍历 registry 中的每个转换器 _trs，并检查其记录的元数据
    for _trs in registry:
        check_recorded_metadata(
            obj=_trs,
            method=method,
            parent=method,
            sample_weight=sample_weight,
            metadata=metadata,
        )
@pytest.mark.usefixtures("enable_slep006")
# 使用 pytest.mark.usefixtures 装饰器确保在测试运行时启用了 slep006
def test_metadata_routing_no_fit_transform():
    """Test metadata routing when the sub-estimator doesn't implement
    ``fit_transform``."""
    # 测试当子估计器没有实现 ``fit_transform`` 时的元数据路由行为

    class NoFitTransform(BaseEstimator):
        # 定义一个没有 fit_transform 方法的子类 NoFitTransform
        def fit(self, X, y=None, sample_weight=None, metadata=None):
            # 实现 fit 方法，验证 sample_weight 和 metadata 参数的存在
            assert sample_weight
            assert metadata
            return self

        def transform(self, X, sample_weight=None, metadata=None):
            # 实现 transform 方法，验证 sample_weight 和 metadata 参数的存在
            assert sample_weight
            assert metadata
            return X

    X = np.array([[0, 1, 2], [2, 4, 6]]).T
    y = [1, 2, 3]
    sample_weight, metadata = [1], "a"
    # 创建 ColumnTransformer 对象 trs，配置子转换器 NoFitTransform，并设置请求为使用 sample_weight 和 metadata
    trs = ColumnTransformer(
        [
            (
                "trans",
                NoFitTransform()
                .set_fit_request(sample_weight=True, metadata=True)
                .set_transform_request(sample_weight=True, metadata=True),
                [0],
            )
        ]
    )

    trs.fit(X, y, sample_weight=sample_weight, metadata=metadata)
    # 测试 fit_transform 方法
    trs.fit_transform(X, y, sample_weight=sample_weight, metadata=metadata)


@pytest.mark.usefixtures("enable_slep006")
@pytest.mark.parametrize("method", ["transform", "fit_transform", "fit"])
# 使用 pytest.mark.parametrize 定义参数化测试方法，测试三种方法：transform、fit_transform、fit
def test_metadata_routing_error_for_column_transformer(method):
    """Test that the right error is raised when metadata is not requested."""
    # 测试当未请求 metadata 时是否会正确抛出异常

    X = np.array([[0, 1, 2], [2, 4, 6]]).T
    y = [1, 2, 3]
    sample_weight, metadata = [1], "a"
    # 创建 ColumnTransformer 对象 trs，使用 ConsumingTransformer，并指定转换的列
    trs = ColumnTransformer([("trans", ConsumingTransformer(), [0])])

    # 构造期望的错误消息
    error_message = (
        "[sample_weight, metadata] are passed but are not explicitly set as requested"
        f" or not requested for ConsumingTransformer.{method}"
    )
    # 使用 pytest.raises 确保当方法调用时抛出 ValueError 异常，异常信息匹配 error_message
    with pytest.raises(ValueError, match=re.escape(error_message)):
        if method == "transform":
            trs.fit(X, y)
            trs.transform(X, sample_weight=sample_weight, metadata=metadata)
        else:
            getattr(trs, method)(X, y, sample_weight=sample_weight, metadata=metadata)


@pytest.mark.usefixtures("enable_slep006")
def test_get_metadata_routing_works_without_fit():
    # Regression test for https://github.com/scikit-learn/scikit-learn/issues/28186
    # Make sure ct.get_metadata_routing() works w/o having called fit.
    # 确保 ct.get_metadata_routing() 在未调用 fit 的情况下正常工作，用于回归测试

    ct = ColumnTransformer([("trans", ConsumingTransformer(), [0])])
    ct.get_metadata_routing()


@pytest.mark.usefixtures("enable_slep006")
def test_remainder_request_always_present():
    # Test that remainder request is always present.
    # 测试 remainder 请求始终存在的情况

    ct = ColumnTransformer(
        [("trans", StandardScaler(), [0])],
        remainder=ConsumingTransformer()
        .set_fit_request(metadata=True)
        .set_transform_request(metadata=True),
    )
    router = ct.get_metadata_routing()
    assert router.consumes("fit", ["metadata"]) == set(["metadata"])


@pytest.mark.usefixtures("enable_slep006")
def test_unused_transformer_request_present():
    # Test that the request of a transformer is always present even when not
    # 测试即使没有使用到的转换器请求也会存在的情况
    # 创建一个 ColumnTransformer 对象，用于处理列的转换
    ct = ColumnTransformer(
        [
            (
                "trans",
                # 将 ConsumingTransformer 实例化并设置适合请求的元数据
                ConsumingTransformer().set_fit_request(metadata=True)
                # 同时设置转换请求的元数据
                .set_transform_request(metadata=True),
                # lambda 函数返回一个空列表，因为没有选择的列
                lambda X: [],
            )
        ]
    )
    # 获取 ColumnTransformer 对象的元数据路由
    router = ct.get_metadata_routing()
    # 使用断言验证路由器在“fit”阶段是否消耗了元数据，且返回集合包含“metadata”
    assert router.consumes("fit", ["metadata"]) == set(["metadata"])
# 元数据路由测试的结束标记
# =============================
```