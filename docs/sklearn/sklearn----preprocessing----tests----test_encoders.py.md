# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\tests\test_encoders.py`

```
# 导入必要的库和模块
import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS

# 定义测试函数，验证稀疏和稠密输出的一致性
def test_one_hot_encoder_sparse_dense():
    X = np.array([[3, 2, 1], [0, 1, 1]])
    
    # 使用稀疏输出的 OneHotEncoder
    enc_sparse = OneHotEncoder()
    # 使用稠密输出的 OneHotEncoder
    enc_dense = OneHotEncoder(sparse_output=False)
    
    # 对数据进行转换
    X_trans_sparse = enc_sparse.fit_transform(X)
    X_trans_dense = enc_dense.fit_transform(X)
    
    # 断言稀疏和稠密输出的形状相同
    assert X_trans_sparse.shape == (2, 5)
    assert X_trans_dense.shape == (2, 5)
    
    # 断言稀疏输出是稀疏矩阵，稠密输出不是稀疏矩阵
    assert sparse.issparse(X_trans_sparse)
    assert not sparse.issparse(X_trans_dense)
    
    # 检查转换后的结果是否一致
    assert_array_equal(
        X_trans_sparse.toarray(), [[0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0]]
    )
    assert_array_equal(X_trans_sparse.toarray(), X_trans_dense)


# 使用参数化测试，验证处理未知类别时的行为
@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
def test_one_hot_encoder_handle_unknown(handle_unknown):
    X = np.array([[0, 2, 1], [1, 0, 3], [1, 0, 2]])
    X2 = np.array([[4, 1, 1]])
    
    # 测试当出现未知特征时，OneHotEncoder 是否会引发错误
    oh = OneHotEncoder(handle_unknown="error")
    oh.fit(X)
    with pytest.raises(ValueError, match="Found unknown categories"):
        oh.transform(X2)
    
    # 测试 ignore 选项，忽略未知特征（转换为全0向量）
    oh = OneHotEncoder(handle_unknown=handle_unknown)
    oh.fit(X)
    X2_passed = X2.copy()
    assert_array_equal(
        oh.transform(X2_passed).toarray(),
        np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
    )
    # 确保转换后的数据没有在原地被修改
    assert_allclose(X2, X2_passed)


# 使用参数化测试，验证处理未知字符串类别时的行为
@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
def test_one_hot_encoder_handle_unknown_strings(handle_unknown):
    X = np.array(["11111111", "22", "333", "4444"]).reshape((-1, 1))
    X2 = np.array(["55555", "22"]).reshape((-1, 1))
    
    # 测试 ignore 选项，在类别为字符串类型时的行为
    oh = OneHotEncoder(handle_unknown=handle_unknown)
    oh.fit(X)
    X2_passed = X2.copy()
    assert_array_equal(
        oh.transform(X2_passed).toarray(),
        np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
    )
    # 确保转换后的数据没有在原地被修改
    assert_array_equal(X2, X2_passed)


# 使用参数化测试，验证不同数据类型输入输出类型的一致性
@pytest.mark.parametrize("output_dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("input_dtype", [np.int32, np.float32, np.float64])
def test_one_hot_encoder_dtype(input_dtype, output_dtype):
    # 创建一个2x1的NumPy数组X，元素为[0, 1]，使用指定的输入数据类型input_dtype
    X = np.asarray([[0, 1]], dtype=input_dtype).T
    
    # 创建一个2x2的NumPy数组X_expected，元素为[[1, 0], [0, 1]]，使用指定的输出数据类型output_dtype
    X_expected = np.asarray([[1, 0], [0, 1]], dtype=output_dtype)
    
    # 初始化一个OneHotEncoder对象oh，自动确定类别并指定输出数据类型output_dtype，然后进行转换并断言结果与X_expected相等
    assert_array_equal(oh.fit_transform(X).toarray(), X_expected)
    
    # 对X进行拟合和转换，并断言结果与X_expected相等
    assert_array_equal(oh.fit(X).transform(X).toarray(), X_expected)
    
    # 初始化一个OneHotEncoder对象oh，自动确定类别并指定输出数据类型output_dtype，同时设置稀疏输出sparse_output为False
    # 然后进行转换并断言结果与X_expected相等
    assert_array_equal(oh.fit_transform(X), X_expected)
    
    # 对X进行拟合和转换，并断言结果与X_expected相等
    assert_array_equal(oh.fit(X).transform(X), X_expected)
# 使用 pytest 的 parametrize 装饰器，为函数 test_one_hot_encoder_dtype_pandas 添加参数化测试
@pytest.mark.parametrize("output_dtype", [np.int32, np.float32, np.float64])
def test_one_hot_encoder_dtype_pandas(output_dtype):
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")

    # 创建一个 DataFrame X_df，包含两列"A"和"B"，每列两行数据
    X_df = pd.DataFrame({"A": ["a", "b"], "B": [1, 2]})
    
    # 期望的转换结果 X_expected，将 DataFrame 转换为 numpy 数组，根据 output_dtype 指定数据类型
    X_expected = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=output_dtype)

    # 创建 OneHotEncoder 对象 oh，指定数据类型为 output_dtype
    oh = OneHotEncoder(dtype=output_dtype)
    
    # 断言转换后的稀疏矩阵数组等于预期的 X_expected
    assert_array_equal(oh.fit_transform(X_df).toarray(), X_expected)
    
    # 使用 fit 方法拟合数据，并使用 transform 方法进行转换，再次断言转换结果等于预期的 X_expected
    assert_array_equal(oh.fit(X_df).transform(X_df).toarray(), X_expected)

    # 创建另一个 OneHotEncoder 对象 oh，关闭稀疏输出，指定数据类型为 output_dtype
    oh = OneHotEncoder(dtype=output_dtype, sparse_output=False)
    
    # 断言转换后的数组等于预期的 X_expected
    assert_array_equal(oh.fit_transform(X_df), X_expected)
    
    # 使用 fit 方法拟合数据，并使用 transform 方法进行转换，再次断言转换结果等于预期的 X_expected
    assert_array_equal(oh.fit(X_df).transform(X_df), X_expected)


# 测试获取 OneHotEncoder 的特征名
def test_one_hot_encoder_feature_names():
    # 创建 OneHotEncoder 对象 enc
    enc = OneHotEncoder()
    
    # 创建包含多个样本的列表 X，每个样本包含多个特征
    X = [
        ["Male", 1, "girl", 2, 3],
        ["Female", 41, "girl", 1, 10],
        ["Male", 51, "boy", 12, 3],
        ["Male", 91, "girl", 21, 30],
    ]
    
    # 对列表 X 进行拟合
    enc.fit(X)
    
    # 获取拟合后的特征名列表 feature_names
    feature_names = enc.get_feature_names_out()
    
    # 断言 feature_names 等于预期的列表
    assert_array_equal(
        [
            "x0_Female",
            "x0_Male",
            "x1_1",
            "x1_41",
            "x1_51",
            "x1_91",
            "x2_boy",
            "x2_girl",
            "x3_1",
            "x3_2",
            "x3_12",
            "x3_21",
            "x4_3",
            "x4_10",
            "x4_30",
        ],
        feature_names,
    )
    
    # 使用不同的输入特征名称列表调用 get_feature_names_out 方法，获取新的 feature_names2
    feature_names2 = enc.get_feature_names_out(["one", "two", "three", "four", "five"])
    
    # 断言 feature_names2 等于预期的列表
    assert_array_equal(
        [
            "one_Female",
            "one_Male",
            "two_1",
            "two_41",
            "two_51",
            "two_91",
            "three_boy",
            "three_girl",
            "four_1",
            "four_2",
            "four_12",
            "four_21",
            "five_3",
            "five_10",
            "five_30",
        ],
        feature_names2,
    )
    
    # 使用不符合长度要求的输入特征名称列表调用 get_feature_names_out 方法，预期抛出 ValueError 异常
    with pytest.raises(ValueError, match="input_features should have length"):
        enc.get_feature_names_out(["one", "two"])


# 测试获取 OneHotEncoder 的特征名，包含 Unicode 字符
def test_one_hot_encoder_feature_names_unicode():
    # 创建 OneHotEncoder 对象 enc
    enc = OneHotEncoder()
    
    # 创建包含 Unicode 字符的 numpy 数组 X
    X = np.array([["c❤t1", "dat2"]], dtype=object).T
    
    # 对数组 X 进行拟合
    enc.fit(X)
    
    # 获取拟合后的特征名列表 feature_names
    feature_names = enc.get_feature_names_out()
    
    # 断言 feature_names 等于预期的列表
    assert_array_equal(["x0_c❤t1", "x0_dat2"], feature_names)
    
    # 使用自定义输入特征名称列表调用 get_feature_names_out 方法，获取新的 feature_names
    feature_names = enc.get_feature_names_out(input_features=["n👍me"])
    
    # 断言 feature_names 等于预期的列表
    assert_array_equal(["n👍me_c❤t1", "n👍me_dat2"], feature_names)


# 测试自定义特征名组合器的行为
def test_one_hot_encoder_custom_feature_name_combiner():
    # 定义特征名组合器函数 name_combiner
    def name_combiner(feature, category):
        return feature + "_" + repr(category)
    
    # 创建 OneHotEncoder 对象 enc，指定特征名组合器为 name_combiner
    enc = OneHotEncoder(feature_name_combiner=name_combiner)
    
    # 创建包含 None 值的 numpy 数组 X
    X = np.array([["None", None]], dtype=object).T
    
    # 对数组 X 进行拟合
    enc.fit(X)
    
    # 获取拟合后的特征名列表 feature_names
    feature_names = enc.get_feature_names_out()
    
    # 断言 feature_names 等于预期的列表
    assert_array_equal(["x0_'None'", "x0_None"], feature_names)
    
    # 使用自定义输入特征名称列表调用 get_feature_names_out 方法，获取新的 feature_names
    feature_names = enc.get_feature_names_out(input_features=["a"])
    
    # 断言 feature_names 等于预期的列表
    assert_array_equal(["a_'None'", "a_None"], feature_names)
    # 定义一个名为 wrong_combiner 的函数，预期接收两个参数 feature 和 category，但函数实现不正确，应返回一个 Python 字符串。
    def wrong_combiner(feature, category):
        # we should be returning a Python string
        return 0  # 错误的实现，应该返回字符串而不是整数
    
    # 使用 OneHotEncoder 创建一个编码器对象 enc，其中 feature_name_combiner 参数被设置为 wrong_combiner 函数。
    # 它预期此函数应能够返回一个字符串，但实际上返回了一个整数。
    enc = OneHotEncoder(feature_name_combiner=wrong_combiner).fit(X)
    
    # 定义一个错误消息，用于检查异常的类型和消息内容
    err_msg = (
        "When `feature_name_combiner` is a callable, it should return a Python string."
    )
    
    # 使用 pytest 库来验证在调用 enc.get_feature_names_out() 时是否会抛出 TypeError 异常，并且异常消息匹配 err_msg 中定义的内容。
    with pytest.raises(TypeError, match=err_msg):
        enc.get_feature_names_out()
# 测试函数：测试 OneHotEncoder 类的 set_params 方法
def test_one_hot_encoder_set_params():
    # 创建一个二维数组 X，包含一个特征
    X = np.array([[1, 2]]).T
    # 创建一个 OneHotEncoder 实例
    oh = OneHotEncoder()
    # 设置尚未拟合的对象的参数
    oh.set_params(categories=[[0, 1, 2, 3]])
    # 断言检查参数是否设置成功
    assert oh.get_params()["categories"] == [[0, 1, 2, 3]]
    # 断言检查拟合并转换后的数组形状是否正确
    assert oh.fit_transform(X).toarray().shape == (2, 4)
    # 再次设置已拟合的对象的参数
    oh.set_params(categories=[[0, 1, 2, 3, 4]])
    # 断言检查拟合并转换后的数组形状是否正确
    assert oh.fit_transform(X).toarray().shape == (2, 5)


# 函数：检查 OneHotEncoder 类在不同参数设置下的转换结果
def check_categorical_onehot(X):
    # 创建 OneHotEncoder 实例，自动推断分类变量
    enc = OneHotEncoder(categories="auto")
    # 对输入 X 进行拟合和转换
    Xtr1 = enc.fit_transform(X)

    # 创建 OneHotEncoder 实例，自动推断分类变量，并指定稀疏输出为 False
    enc = OneHotEncoder(categories="auto", sparse_output=False)
    # 对输入 X 进行拟合和转换
    Xtr2 = enc.fit_transform(X)

    # 断言检查两种设置下的转换结果是否近似相等
    assert_allclose(Xtr1.toarray(), Xtr2)

    # 断言检查转换后的数组是否是稀疏矩阵，并且格式为 "csr"
    assert sparse.issparse(Xtr1) and Xtr1.format == "csr"
    # 返回转换后的数组的稠密表示
    return Xtr1.toarray()


# 使用参数化测试的标记，测试不同类型和设置下的 OneHotEncoder 转换
@pytest.mark.parametrize(
    "X",
    [
        [["def", 1, 55], ["abc", 2, 55]],  # 混合类型数据
        np.array([[10, 1, 55], [5, 2, 55]]),  # 数值类型数据
        np.array([["b", "A", "cat"], ["a", "B", "cat"]], dtype=object),  # 对象类型数据
        np.array([["b", 1, "cat"], ["a", np.nan, "cat"]], dtype=object),  # 混合类型数据，包含 NaN
        np.array([["b", 1, "cat"], ["a", float("nan"), "cat"]], dtype=object),  # 混合类型数据，包含 float NaN
        np.array([[None, 1, "cat"], ["a", 2, "cat"]], dtype=object),  # 混合类型数据，包含 None
        np.array([[None, 1, None], ["a", np.nan, None]], dtype=object),  # 混合类型数据，包含 None 和 NaN
        np.array([[None, 1, None], ["a", float("nan"), None]], dtype=object),  # 混合类型数据，包含 None 和 float NaN
    ],
    ids=[
        "mixed",
        "numeric",
        "object",
        "mixed-nan",
        "mixed-float-nan",
        "mixed-None",
        "mixed-None-nan",
        "mixed-None-float-nan",
    ],
)
# 测试函数：测试 OneHotEncoder 的不同输入类型和设置
def test_one_hot_encoder(X):
    # 检查仅包含第一列的 OneHot 编码结果
    Xtr = check_categorical_onehot(np.array(X)[:, [0]])
    # 断言检查编码结果是否与预期相等
    assert_allclose(Xtr, [[0, 1], [1, 0]])

    # 检查包含前两列的 OneHot 编码结果
    Xtr = check_categorical_onehot(np.array(X)[:, [0, 1]])
    # 断言检查编码结果是否与预期相等
    assert_allclose(Xtr, [[0, 1, 1, 0], [1, 0, 0, 1]])

    # 创建 OneHotEncoder 实例，自动推断分类变量，并对整个 X 进行拟合和转换
    Xtr = OneHotEncoder(categories="auto").fit_transform(X)
    # 断言检查编码结果是否与预期相等
    assert_allclose(Xtr.toarray(), [[0, 1, 1, 0, 1], [1, 0, 0, 1, 1]])


# 使用参数化测试的标记，测试 OneHotEncoder 的逆转换功能
@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
@pytest.mark.parametrize("sparse_", [False, True])
@pytest.mark.parametrize("drop", [None, "first"])
# 测试函数：测试 OneHotEncoder 的逆转换功能
def test_one_hot_encoder_inverse(handle_unknown, sparse_, drop):
    # 输入数据 X
    X = [["abc", 2, 55], ["def", 1, 55], ["abc", 3, 55]]
    # 创建 OneHotEncoder 实例，指定稀疏输出和丢弃策略
    enc = OneHotEncoder(sparse_output=sparse_, drop=drop)
    # 对 X 进行拟合和转换
    X_tr = enc.fit_transform(X)
    # 期望的逆转换结果
    exp = np.array(X, dtype=object)
    # 断言检查逆转换结果是否与期望相等
    assert_array_equal(enc.inverse_transform(X_tr), exp)

    # 输入数据 X
    X = [[2, 55], [1, 55], [3, 55]]
    # 创建 OneHotEncoder 实例，自动推断分类变量，指定稀疏输出和丢弃策略
    enc = OneHotEncoder(sparse_output=sparse_, categories="auto", drop=drop)
    # 对 X 进行拟合和转换
    X_tr = enc.fit_transform(X)
    # 期望的逆转换结果
    exp = np.array(X)
    # 断言检查逆转换结果是否与期望相等
    assert_array_equal(enc.inverse_transform(X_tr), exp)
    if drop is None:
        # 如果 drop 参数为 None，则处理未知类别
        # drop 参数与 handle_unknown=ignore 不兼容
        X = [["abc", 2, 55], ["def", 1, 55], ["abc", 3, 55]]
        # 创建 OneHotEncoder 对象，设置稀疏输出和处理未知类别的策略
        enc = OneHotEncoder(
            sparse_output=sparse_,
            handle_unknown=handle_unknown,
            categories=[["abc", "def"], [1, 2], [54, 55, 56]],
        )
        # 对数据 X 进行编码转换
        X_tr = enc.fit_transform(X)
        # 创建预期输出的 numpy 数组
        exp = np.array(X, dtype=object)
        exp[2, 1] = None  # 将第三行、第二列的元素设置为 None
        # 验证逆转换结果是否与预期一致
        assert_array_equal(enc.inverse_transform(X_tr), exp)

        # 当输出本应是数值类型，但未知类别时仍为对象类型
        X = [[2, 55], [1, 55], [3, 55]]
        # 创建另一个 OneHotEncoder 对象，设置稀疏输出和类别信息
        enc = OneHotEncoder(
            sparse_output=sparse_,
            categories=[[1, 2], [54, 56]],
            handle_unknown=handle_unknown,
        )
        # 再次对数据 X 进行编码转换
        X_tr = enc.fit_transform(X)
        # 创建预期输出的 numpy 数组
        exp = np.array(X, dtype=object)
        exp[2, 0] = None  # 将第三行、第一列的元素设置为 None
        exp[:, 1] = None   # 将所有行的第二列元素设置为 None
        # 验证逆转换结果是否与预期一致
        assert_array_equal(enc.inverse_transform(X_tr), exp)

    # 当输入数据的形状不正确时会引发异常
    X_tr = np.array([[0, 1, 1], [1, 0, 1]])
    msg = re.escape("Shape of the passed X data is not correct")
    # 使用 pytest 断言捕获 ValueError 异常，并验证异常消息是否符合预期
    with pytest.raises(ValueError, match=msg):
        enc.inverse_transform(X_tr)
@pytest.mark.parametrize("sparse_", [False, True])
# 参数化测试装饰器，用于测试稀疏矩阵的两种情况：False 和 True
@pytest.mark.parametrize(
    "X, X_trans",
    [
        ([[2, 55], [1, 55], [2, 55]], [[0, 1, 1], [0, 0, 0], [0, 1, 1]]),
        (
            [["one", "a"], ["two", "a"], ["three", "b"], ["two", "a"]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0]],
        ),
    ],
)
# 参数化测试装饰器，用于测试输入 X 和其转换 X_trans 的不同情况
def test_one_hot_encoder_inverse_transform_raise_error_with_unknown(
    X, X_trans, sparse_
):
    """Check that `inverse_transform` raise an error with unknown samples, no
    dropped feature, and `handle_unknow="error`.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/14934
    """
    # 使用 OneHotEncoder 对象对输入 X 进行拟合
    enc = OneHotEncoder(sparse_output=sparse_).fit(X)
    # 预期的错误信息正则表达式
    msg = (
        r"Samples \[(\d )*\d\] can not be inverted when drop=None and "
        r"handle_unknown='error' because they contain all zeros"
    )

    if sparse_:
        # 通过 _convert_container 函数模拟稀疏数据的转换，使用 "sparse" 类型
        X_trans = _convert_container(X_trans, "sparse")
    # 检查是否会抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        enc.inverse_transform(X_trans)


def test_one_hot_encoder_inverse_if_binary():
    # 定义输入数组 X，包含字符串和数字组合的对象数组
    X = np.array([["Male", 1], ["Female", 3], ["Female", 2]], dtype=object)
    # 创建 OneHotEncoder 对象，设置 drop="if_binary"，输出非稀疏矩阵
    ohe = OneHotEncoder(drop="if_binary", sparse_output=False)
    # 对输入 X 进行拟合和转换操作，并保存结果到 X_tr
    X_tr = ohe.fit_transform(X)
    # 断言逆转换后的结果与原始输入 X 相等
    assert_array_equal(ohe.inverse_transform(X_tr), X)


@pytest.mark.parametrize("drop", ["if_binary", "first", None])
@pytest.mark.parametrize("reset_drop", ["if_binary", "first", None])
def test_one_hot_encoder_drop_reset(drop, reset_drop):
    # 检查在不重新拟合的情况下重置 drop 参数不会引发错误
    # 定义输入数组 X，包含字符串和数字组合的对象数组
    X = np.array([["Male", 1], ["Female", 3], ["Female", 2]], dtype=object)
    # 创建 OneHotEncoder 对象，设置 drop 参数和输出为非稀疏矩阵
    ohe = OneHotEncoder(drop=drop, sparse_output=False)
    # 对输入 X 进行拟合和转换操作，并保存结果到 X_tr
    ohe.fit(X)
    X_tr = ohe.transform(X)
    # 获取特征名称列表
    feature_names = ohe.get_feature_names_out()
    # 设置参数 drop=reset_drop，不重新拟合
    ohe.set_params(drop=reset_drop)
    # 断言逆转换后的结果与原始输入 X 相等
    assert_array_equal(ohe.inverse_transform(X_tr), X)
    # 断言转换后的结果与之前的 X_tr 相等
    assert_allclose(ohe.transform(X), X_tr)
    # 断言特征名称列表未变化
    assert_array_equal(ohe.get_feature_names_out(), feature_names)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("X", [[1, 2], np.array([3.0, 4.0])])
def test_X_is_not_1D(X, method):
    # 检查输入 X 是否为一维数组的测试函数
    oh = OneHotEncoder()

    msg = "Expected 2D array, got 1D array instead"
    # 断言调用方法 method 时，会抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        getattr(oh, method)(X)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_X_is_not_1D_pandas(method):
    # 检查输入 X 是否为 Pandas Series 的测试函数
    pd = pytest.importorskip("pandas")
    # 创建 Pandas Series 对象作为输入 X
    X = pd.Series([6, 3, 4, 6])
    oh = OneHotEncoder()

    # 准备预期的错误消息
    msg = f"Expected a 2-dimensional container but got {type(X)} instead."
    # 断言调用方法 method 时，会抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        getattr(oh, method)(X)


@pytest.mark.parametrize(
    "X, cat_exp, cat_dtype",
    [
        # 测试用例1：混合类型数组，指定dtype为np.object_
        ( [["abc", 55], ["def", 55]],         # 输入数组
          [["abc", "def"], [55]],             # 预期输出数组
          np.object_),                        # 指定数据类型

        # 测试用例2：整数类型数组
        ( np.array([[1, 2], [3, 2]]),         # 输入数组
          [[1, 3], [2]],                      # 预期输出数组
          np.integer),                        # 指定数据类型

        # 测试用例3：对象类型数组
        ( np.array([["A", "cat"], ["B", "cat"]], dtype=object),   # 输入数组
          [["A", "B"], ["cat"]],                                  # 预期输出数组
          np.object_),                                             # 指定数据类型

        # 测试用例4：字符串类型数组
        ( np.array([["A", "cat"], ["B", "cat"]]),   # 输入数组
          [["A", "B"], ["cat"]],                    # 预期输出数组
          np.str_),                                # 指定数据类型

        # 测试用例5：包含缺失值（NaN）的浮点数类型数组
        ( np.array([[1, 2], [np.nan, 2]]),     # 输入数组
          [[1, np.nan], [2]],                  # 预期输出数组
          np.float64),                         # 指定数据类型

        # 测试用例6：包含缺失值（NaN）和None的对象类型数组
        ( np.array([["A", np.nan], [None, np.nan]], dtype=object),   # 输入数组
          [["A", None], [np.nan]],                                   # 预期输出数组
          np.object_),                                               # 指定数据类型

        # 测试用例7：包含缺失值（NaN）的对象类型数组，直接使用float("nan")
        ( np.array([["A", float("nan")], [None, float("nan")]], dtype=object),   # 输入数组
          [["A", None], [float("nan")]],                                         # 预期输出数组
          np.object_),                                                           # 指定数据类型
    ],
    ids=[
        "mixed",                    # 测试用例1的ID
        "numeric",                  # 测试用例2的ID
        "object",                   # 测试用例3的ID
        "string",                   # 测试用例4的ID
        "missing-float",            # 测试用例5的ID
        "missing-np.nan-object",    # 测试用例6的ID
        "missing-float-nan-object", # 测试用例7的ID
    ],
# 定义测试函数，用于测试OneHotEncoder类的行为
def test_one_hot_encoder_specified_categories(X, X2, cats, cat_dtype, handle_unknown):
    # 创建OneHotEncoder对象，指定要使用的类别
    enc = OneHotEncoder(categories=cats)
    
    # 预期的转换结果
    exp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    # 断言转换后的稀疏矩阵与预期结果相等
    assert_array_equal(enc.fit_transform(X).toarray(), exp)
    
    # 断言OneHotEncoder对象中的categories属性与给定的cats相等
    assert list(enc.categories[0]) == list(cats[0])
    
    # 断言OneHotEncoder对象中的categories_属性的列表形式与给定的cats[0]相等
    assert enc.categories_[0].tolist() == list(cats[0])
    
    # 断言OneHotEncoder对象中的categories_属性的数据类型与cat_dtype相等
    assert enc.categories_[0].dtype == cat_dtype
    
    # 当手动指定类别时，如果在拟合过程中遇到未知的类别，应该引发ValueError异常
    enc = OneHotEncoder(categories=cats)
    with pytest.raises(ValueError, match="Found unknown categories"):
        enc.fit(X2)
    
    # 使用handle_unknown参数处理未知的类别
    enc = OneHotEncoder(categories=cats, handle_unknown=handle_unknown)
    
    # 重新定义预期的转换结果
    exp = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    
    # 断言拟合后再转换的结果与新的预期结果相等
    assert_array_equal(enc.fit(X2).transform(X2).toarray(), exp)
def test_one_hot_encoder_unsorted_categories():
    # 创建一个包含单列的numpy数组X，每个元素是一个包含两个字符串的数组
    X = np.array([["a", "b"]], dtype=object).T

    # 使用给定的类别创建OneHotEncoder对象
    enc = OneHotEncoder(categories=[["b", "a", "c"]])

    # 预期的编码结果
    exp = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    # 断言转换后的数组与预期结果相等
    assert_array_equal(enc.fit(X).transform(X).toarray(), exp)

    # 断言直接使用fit_transform的结果与预期结果相等
    assert_array_equal(enc.fit_transform(X).toarray(), exp)

    # 断言编码器的第一个类别列表与预期的类别顺序相同
    assert enc.categories_[0].tolist() == ["b", "a", "c"]

    # 断言编码器的第一个类别的dtype是对象类型
    assert np.issubdtype(enc.categories_[0].dtype, np.object_)

    # 对于数值类型的未排序类别，预期会引发值错误
    X = np.array([[1, 2]]).T
    enc = OneHotEncoder(categories=[[2, 1, 3]])
    msg = "Unsorted categories are not supported"

    # 使用pytest断言引发值错误，并检查错误消息匹配
    with pytest.raises(ValueError, match=msg):
        enc.fit_transform(X)


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
def test_encoder_nan_ending_specified_categories(Encoder):
    """Test encoder for specified categories that nan is at the end.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27088
    """
    # 定义包含NaN值的类别数组
    cats = [np.array([0, np.nan, 1])]

    # 使用指定的类别数组创建编码器对象
    enc = Encoder(categories=cats)

    # 创建包含两列数据的numpy数组X，每列一个对象类型
    X = np.array([[0, 1]], dtype=object).T

    # 使用pytest断言应引发值错误，并检查错误消息匹配
    with pytest.raises(ValueError, match="Nan should be the last element"):
        enc.fit(X)


def test_one_hot_encoder_specified_categories_mixed_columns():
    # 创建包含两列数据的numpy数组X，每列包含字符串和整数对象
    X = np.array([["a", "b"], [0, 2]], dtype=object).T

    # 使用指定的类别创建OneHotEncoder对象，包含两个类别列表
    enc = OneHotEncoder(categories=[["a", "b", "c"], [0, 1, 2]])

    # 预期的编码结果
    exp = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])

    # 断言转换后的数组与预期结果相等
    assert_array_equal(enc.fit_transform(X).toarray(), exp)

    # 断言编码器的第一个类别列表与预期的类别顺序相同
    assert enc.categories_[0].tolist() == ["a", "b", "c"]

    # 断言编码器的第一个类别的dtype是对象类型
    assert np.issubdtype(enc.categories_[0].dtype, np.object_)

    # 断言编码器的第二个类别列表与预期的类别顺序相同
    assert enc.categories_[1].tolist() == [0, 1, 2]

    # 对于从对象类型数据生成的整数类别，预期dtype是对象类型
    assert np.issubdtype(enc.categories_[1].dtype, np.object_)


def test_one_hot_encoder_pandas():
    pd = pytest.importorskip("pandas")

    # 创建一个包含两列的DataFrame
    X_df = pd.DataFrame({"A": ["a", "b"], "B": [1, 2]})

    # 检查函数是否能正确处理分类数据的独热编码
    Xtr = check_categorical_onehot(X_df)
    assert_allclose(Xtr, [[1, 0, 1, 0], [0, 1, 0, 1]])


@pytest.mark.parametrize(
    "drop, expected_names",
    [
        ("first", ["x0_c", "x2_b"]),
        ("if_binary", ["x0_c", "x1_2", "x2_b"]),
        (["c", 2, "b"], ["x0_b", "x2_a"]),
    ],
    ids=["first", "binary", "manual"],
)
def test_one_hot_encoder_feature_names_drop(drop, expected_names):
    # 创建包含两个子数组的列表X
    X = [["c", 2, "a"], ["b", 2, "b"]]

    # 使用指定的drop参数创建OneHotEncoder对象
    ohe = OneHotEncoder(drop=drop)

    # 对列表X进行拟合
    ohe.fit(X)

    # 获取输出的特征名称
    feature_names = ohe.get_feature_names_out()

    # 使用pytest断言特征名称与预期的名称列表相等
    assert_array_equal(expected_names, feature_names)


def test_one_hot_encoder_drop_equals_if_binary():
    # 典型的案例
    X = [[10, "yes"], [20, "no"], [30, "yes"]]

    # 预期的编码结果和应该被丢弃的索引
    expected = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )
    expected_drop_idx = np.array([None, 0])

    # 使用指定的drop参数创建OneHotEncoder对象，禁用稀疏输出
    ohe = OneHotEncoder(drop="if_binary", sparse_output=False)

    # 对X进行拟合和转换
    result = ohe.fit_transform(X)
    # 使用 assert_array_equal 检查 ohe 对象的 drop_idx_ 属性是否与预期的 expected_drop_idx 相等
    assert_array_equal(ohe.drop_idx_, expected_drop_idx)
    # 使用 assert_allclose 检查 result 是否与预期的 expected 相近
    assert_allclose(result, expected)

    # 当只有一个类别时，行为等同于 drop=None
    X = [["true", "a"], ["false", "a"], ["false", "a"]]
    # 预期的独热编码结果
    expected = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    # 预期的 drop_idx_ 结果，其中第一个类别应当被删除，第二个类别不删除
    expected_drop_idx = np.array([0, None])

    # 创建一个 OneHotEncoder 对象，设定 drop="if_binary"，sparse_output=False
    ohe = OneHotEncoder(drop="if_binary", sparse_output=False)
    # 对输入数据 X 进行拟合和转换
    result = ohe.fit_transform(X)
    # 使用 assert_array_equal 检查 ohe 对象的 drop_idx_ 属性是否与预期的 expected_drop_idx 相等
    assert_array_equal(ohe.drop_idx_, expected_drop_idx)
    # 使用 assert_allclose 检查 result 是否与预期的 expected 相近
    assert_allclose(result, expected)
@pytest.mark.parametrize(
    "X",
    [
        [["abc", 2, 55], ["def", 1, 55]],  # 测试参数X：包含混合类型的列表
        np.array([[10, 2, 55], [20, 1, 55]]),  # 测试参数X：包含整数类型的NumPy数组
        np.array([["a", "B", "cat"], ["b", "A", "cat"]], dtype=object),  # 测试参数X：包含对象类型的NumPy数组
    ],
    ids=["mixed", "numeric", "object"],  # 测试用例的标识
)
def test_ordinal_encoder(X):
    enc = OrdinalEncoder()  # 创建OrdinalEncoder对象
    exp = np.array([[0, 1, 0], [1, 0, 0]], dtype="int64")  # 预期输出结果
    assert_array_equal(enc.fit_transform(X), exp.astype("float64"))  # 断言OrdinalEncoder的转换结果与预期结果相等
    enc = OrdinalEncoder(dtype="int64")  # 创建指定dtype为int64的OrdinalEncoder对象
    assert_array_equal(enc.fit_transform(X), exp)  # 断言OrdinalEncoder的转换结果与预期结果相等


@pytest.mark.parametrize(
    "X, X2, cats, cat_dtype",
    [
        (
            np.array([["a", "b"]], dtype=object).T,
            np.array([["a", "d"]], dtype=object).T,
            [["a", "b", "c"]],
            np.object_,
        ),
        (
            np.array([[1, 2]], dtype="int64").T,
            np.array([[1, 4]], dtype="int64").T,
            [[1, 2, 3]],
            np.int64,
        ),
        (
            np.array([["a", "b"]], dtype=object).T,
            np.array([["a", "d"]], dtype=object).T,
            [np.array(["a", "b", "c"])],
            np.object_,
        ),
    ],
    ids=["object", "numeric", "object-string-cat"],  # 测试用例的标识
)
def test_ordinal_encoder_specified_categories(X, X2, cats, cat_dtype):
    enc = OrdinalEncoder(categories=cats)  # 创建指定categories的OrdinalEncoder对象
    exp = np.array([[0.0], [1.0]])  # 预期输出结果
    assert_array_equal(enc.fit_transform(X), exp)  # 断言OrdinalEncoder的转换结果与预期结果相等
    assert list(enc.categories[0]) == list(cats[0])  # 断言OrdinalEncoder的类别与指定的categories相等
    assert enc.categories_[0].tolist() == list(cats[0])  # 断言OrdinalEncoder的类别与指定的categories相等
    # 手动指定的categories应该与数据的dtype一致
    assert enc.categories_[0].dtype == cat_dtype

    # 当手动指定categories时，如果有未知的类别应该在fit时引发异常
    enc = OrdinalEncoder(categories=cats)  # 再次创建指定categories的OrdinalEncoder对象
    with pytest.raises(ValueError, match="Found unknown categories"):  # 断言在fit时会引发值错误异常
        enc.fit(X2)


def test_ordinal_encoder_inverse():
    X = [["abc", 2, 55], ["def", 1, 55]]  # 输入数据X
    enc = OrdinalEncoder()  # 创建OrdinalEncoder对象
    X_tr = enc.fit_transform(X)  # 对X进行转换
    exp = np.array(X, dtype=object)  # 预期输出结果
    assert_array_equal(enc.inverse_transform(X_tr), exp)  # 断言逆转换结果与预期结果相等

    # 如果形状不正确，应该引发异常
    X_tr = np.array([[0, 1, 1, 2], [1, 0, 1, 0]])  # 错误的形状输入数据
    msg = re.escape("Shape of the passed X data is not correct")  # 异常信息模板
    with pytest.raises(ValueError, match=msg):  # 断言在逆转换时会引发形状错误的值错误异常
        enc.inverse_transform(X_tr)


def test_ordinal_encoder_handle_unknowns_string():
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2)  # 创建处理未知值的OrdinalEncoder对象
    X_fit = np.array([["a", "x"], ["b", "y"], ["c", "z"]], dtype=object)  # 用于fit的输入数据
    X_trans = np.array([["c", "xy"], ["bla", "y"], ["a", "x"]], dtype=object)  # 用于transform的输入数据
    enc.fit(X_fit)  # 对fit数据进行拟合

    X_trans_enc = enc.transform(X_trans)  # 对transform数据进行转换
    exp = np.array([[2, -2], [-2, 1], [0, 0]], dtype="int64")  # 预期输出结果
    assert_array_equal(X_trans_enc, exp)  # 断言转换结果与预期结果相等

    X_trans_inv = enc.inverse_transform(X_trans_enc)  # 对转换后的数据进行逆转换
    inv_exp = np.array([["c", None], [None, "y"], ["a", "x"]], dtype=object)  # 预期逆转换结果
    assert_array_equal(X_trans_inv, inv_exp)  # 断言逆转换结果与预期结果相等
@pytest.mark.parametrize("dtype", [float, int])
def test_ordinal_encoder_handle_unknowns_numeric(dtype):
    # 使用 pytest 的参数化装饰器，测试不同的数据类型（float 和 int）

    # 创建 OrdinalEncoder 对象，设定 handle_unknown="use_encoded_value" 和 unknown_value=-999
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-999)

    # 创建输入数据 X_fit 和 X_trans，分别使用指定的数据类型 dtype
    X_fit = np.array([[1, 7], [2, 8], [3, 9]], dtype=dtype)
    X_trans = np.array([[3, 12], [23, 8], [1, 7]], dtype=dtype)

    # 对 X_fit 进行拟合
    enc.fit(X_fit)

    # 对 X_trans 进行转换，并生成期望结果 exp
    X_trans_enc = enc.transform(X_trans)
    exp = np.array([[2, -999], [-999, 1], [0, 0]], dtype="int64")
    assert_array_equal(X_trans_enc, exp)

    # 对 X_trans_enc 进行逆转换，并生成逆转换的期望结果 inv_exp
    X_trans_inv = enc.inverse_transform(X_trans_enc)
    inv_exp = np.array([[3, None], [None, 8], [1, 7]], dtype=object)
    assert_array_equal(X_trans_inv, inv_exp)


def test_ordinal_encoder_handle_unknowns_nan():
    # 确保 unknown_value=np.nan 能够正确工作

    # 创建 OrdinalEncoder 对象，设定 handle_unknown="use_encoded_value" 和 unknown_value=np.nan
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)

    # 创建输入数据 X_fit，并对其进行拟合
    X_fit = np.array([[1], [2], [3]])
    enc.fit(X_fit)

    # 对 X_trans 进行转换，并验证结果
    X_trans = enc.transform([[1], [2], [4]])
    assert_array_equal(X_trans, [[0], [1], [np.nan]])


def test_ordinal_encoder_handle_unknowns_nan_non_float_dtype():
    # 确保当 unknown_value=np.nan 且 dtype 不是浮点型时会引发错误

    # 创建 OrdinalEncoder 对象，设定 handle_unknown="use_encoded_value"、unknown_value=np.nan 和 dtype=int
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=np.nan, dtype=int
    )

    # 创建输入数据 X_fit，并尝试进行拟合
    X_fit = np.array([[1], [2], [3]])

    # 使用 pytest 检查是否会引发预期的 ValueError 异常
    with pytest.raises(ValueError, match="dtype parameter should be a float dtype"):
        enc.fit(X_fit)


def test_ordinal_encoder_raise_categories_shape():
    # 检查当 categories 参数是一个数组时，其形状是否匹配输入数据的形状

    # 创建输入数据 X 和 categories 列表
    X = np.array([["Low", "Medium", "High", "Medium", "Low"]], dtype=object).T
    cats = ["Low", "Medium", "High"]

    # 创建 OrdinalEncoder 对象，使用指定的 categories 列表
    enc = OrdinalEncoder(categories=cats)

    # 使用 pytest 检查是否会引发预期的 ValueError 异常
    msg = "Shape mismatch: if categories is an array,"
    with pytest.raises(ValueError, match=msg):
        enc.fit(X)


def test_encoder_dtypes():
    # 检查在确定类别时，数据类型是否被保留不变

    # 创建 OneHotEncoder 对象，设定 categories="auto"
    enc = OneHotEncoder(categories="auto")

    # 创建预期结果 exp
    exp = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype="float64")

    # 遍历不同类型的输入数据 X，并验证结果
    for X in [
        np.array([[1, 2], [3, 4]], dtype="int64"),
        np.array([[1, 2], [3, 4]], dtype="float64"),
        np.array([["a", "b"], ["c", "d"]]),  # str dtype
        np.array([[b"a", b"b"], [b"c", b"d"]]),  # bytes dtype
        np.array([[1, "a"], [3, "b"]], dtype="object"),
    ]:
        enc.fit(X)
        # 检查每个类别的数据类型是否与输入数据 X 的数据类型匹配
        assert all([enc.categories_[i].dtype == X.dtype for i in range(2)])
        assert_array_equal(enc.transform(X).toarray(), exp)

    # 对特定类型的输入数据 X 进行拟合，并再次验证结果
    X = [[1, 2], [3, 4]]
    enc.fit(X)
    assert all([np.issubdtype(enc.categories_[i].dtype, np.integer) for i in range(2)])
    assert_array_equal(enc.transform(X).toarray(), exp)

    # 对另一种特定类型的输入数据 X 进行拟合，并再次验证结果
    X = [[1, "a"], [3, "b"]]
    enc.fit(X)
    assert all([enc.categories_[i].dtype == "object" for i in range(2)])
    assert_array_equal(enc.transform(X).toarray(), exp)


def test_encoder_dtypes_pandas():
    # 检查数据类型（类似于测试 dataframes 的 test_categorical_encoder_dtypes）

    # 导入 pandas 库，如果不存在则跳过该测试
    pd = pytest.importorskip("pandas")

    # 创建 OneHotEncoder 对象，设定 categories="auto"
    enc = OneHotEncoder(categories="auto")
    # 创建一个 NumPy 数组，包含两行六列的浮点数，用于表示期望的转换结果
    exp = np.array(
        [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]],
        dtype="float64",
    )

    # 创建一个包含三列的 Pandas DataFrame，用于进行编码器的拟合
    X = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]}, dtype="int64")
    # 对 DataFrame 进行编码器的拟合
    enc.fit(X)
    # 断言每个编码后的分类特征的数据类型为 int64
    assert all([enc.categories_[i].dtype == "int64" for i in range(2)])
    # 断言对 X 进行编码后的转换结果与期望的转换结果 exp 相等
    assert_array_equal(enc.transform(X).toarray(), exp)

    # 创建一个包含三列的 Pandas DataFrame，其中包含不同类型的数据
    X = pd.DataFrame({"A": [1, 2], "B": ["a", "b"], "C": [3.0, 4.0]})
    # 获取 DataFrame 中每列的数据类型
    X_type = [X["A"].dtype, X["B"].dtype, X["C"].dtype]
    # 对 DataFrame 进行编码器的拟合
    enc.fit(X)
    # 断言每个编码后的分类特征的数据类型与原始数据列的数据类型相同
    assert all([enc.categories_[i].dtype == X_type[i] for i in range(3)])
    # 断言对 X 进行编码后的转换结果与期望的转换结果 exp 相等
    assert_array_equal(enc.transform(X).toarray(), exp)
@pytest.mark.parametrize(
    "missing_value", [np.nan, None, float("nan")]
)
# 使用 pytest 的参数化装饰器，定义了一个名为 missing_value 的参数化测试用例，包括三种不同的缺失值表示方式
def test_one_hot_encoder_drop_manual(missing_value):
    # 定义要从编码器中删除的分类列表
    cats_to_drop = ["def", 12, 3, 56, missing_value]
    # 创建一个 OneHotEncoder 对象，指定要删除的分类列表
    enc = OneHotEncoder(drop=cats_to_drop)
    # 定义输入特征矩阵 X，包含多个样本，每个样本用列表表示
    X = [
        ["abc", 12, 2, 55, "a"],
        ["def", 12, 1, 55, "a"],
        ["def", 12, 3, 56, missing_value],
    ]
    # 对输入数据进行编码转换，并转换为稀疏矩阵表示
    trans = enc.fit_transform(X).toarray()
    # 预期的编码结果，作为对照
    exp = [[1, 0, 1, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0]]
    # 断言转换后的结果与预期结果一致
    assert_array_equal(trans, exp)
    # 断言编码器的 drop 属性与定义的 cats_to_drop 相等
    assert enc.drop is cats_to_drop

    # 获取在编码过程中被删除的分类值
    dropped_cats = [
        cat[feature] for cat, feature in zip(enc.categories_, enc.drop_idx_)
    ]
    # 对转换后的数据进行逆向转换
    X_inv_trans = enc.inverse_transform(trans)
    # 将原始输入 X 转换为 numpy 数组
    X_array = np.array(X, dtype=object)

    # 如果最后一个被删除的值是 np.nan
    if is_scalar_nan(cats_to_drop[-1]):
        # 断言被删除的分类列表中除了最后一个值外的所有值与定义的 cats_to_drop 相等
        assert_array_equal(dropped_cats[:-1], cats_to_drop[:-1])
        # 断言最后一个被删除的值是 np.nan
        assert is_scalar_nan(dropped_cats[-1])
        assert is_scalar_nan(cats_to_drop[-1])
        # 断言转换后的结果中不包含最后一列，即包含缺失值的列
        assert_array_equal(X_array[:, :-1], X_inv_trans[:, :-1])

        # 检查最后一列是否是缺失值
        assert_array_equal(X_array[-1, :-1], X_inv_trans[-1, :-1])
        assert is_scalar_nan(X_array[-1, -1])
        assert is_scalar_nan(X_inv_trans[-1, -1])
    else:
        # 断言被删除的分类列表与定义的 cats_to_drop 完全相等
        assert_array_equal(dropped_cats, cats_to_drop)
        # 断言转换后的数据与原始输入数据完全一致
        assert_array_equal(X_array, X_inv_trans)
    [
        # 第一个字典，设定最大分类数为2
        {"max_categories": 2},
        # 第二个字典，设定最小频率为11
        {"min_frequency": 11},
        # 第三个字典，设定最小频率为0.29
        {"min_frequency": 0.29},
        # 第四个字典，同时设定最大分类数为2和最小频率为6
        {"max_categories": 2, "min_frequency": 6},
        # 第五个字典，同时设定最大分类数为4和最小频率为12
        {"max_categories": 4, "min_frequency": 12},
    ],
@pytest.mark.parametrize("categories", ["auto", [["a", "b", "c", "d"]]])
# 使用 pytest 的 parametrize 装饰器为测试函数提供参数化测试的支持，categories 参数被设置为两个不同的测试参数

def test_ohe_infrequent_two_levels(kwargs, categories):
    """Test that different parameters for combine 'a', 'c', and 'd' into
    the infrequent category works as expected."""
    # 测试不同参数组合时，将 'a'、'c' 和 'd' 合并到罕见类别的功能是否按预期工作

    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    # 创建一个包含多个类别的训练数据集 X_train

    ohe = OneHotEncoder(
        categories=categories,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        **kwargs,
    ).fit(X_train)
    # 创建一个 OneHotEncoder 对象，使用给定的参数进行初始化和训练

    assert_array_equal(ohe.infrequent_categories_, [["a", "c", "d"]])
    # 检查 OneHotEncoder 对象中罕见类别是否包含预期的类别列表

    X_test = [["b"], ["a"], ["c"], ["d"], ["e"]]
    expected = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])
    # 准备用于测试的测试数据 X_test 和预期的转换结果 expected

    X_trans = ohe.transform(X_test)
    # 对测试数据 X_test 进行转换

    assert_allclose(expected, X_trans)
    # 检查转换后的结果是否与预期结果非常接近

    expected_inv = [[col] for col in ["b"] + ["infrequent_sklearn"] * 4]
    X_inv = ohe.inverse_transform(X_trans)
    # 对转换后的数据进行逆转换，并准备预期的逆转换结果 expected_inv

    assert_array_equal(expected_inv, X_inv)
    # 检查逆转换后的结果是否与预期的结果相等

    feature_names = ohe.get_feature_names_out()
    # 获取输出特征的名称列表

    assert_array_equal(["x0_b", "x0_infrequent_sklearn"], feature_names)
    # 检查输出特征的名称列表是否符合预期


@pytest.mark.parametrize("drop", ["if_binary", "first", ["b"]])
# 使用 pytest 的 parametrize 装饰器为测试函数提供参数化测试的支持，drop 参数被设置为三个不同的测试参数

def test_ohe_infrequent_two_levels_drop_frequent(drop):
    """Test two levels and dropping the frequent category."""
    # 测试在两个级别中删除常见类别的功能

    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    # 创建一个包含多个类别的训练数据集 X_train

    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        max_categories=2,
        drop=drop,
    ).fit(X_train)
    # 创建一个 OneHotEncoder 对象，使用给定的参数进行初始化和训练

    assert ohe.categories_[0][ohe.drop_idx_[0]] == "b"
    # 检查删除指定类别后的类别列表是否包含预期的结果

    X_test = np.array([["b"], ["c"]])
    X_trans = ohe.transform(X_test)
    # 对测试数据 X_test 进行转换

    assert_allclose([[0], [1]], X_trans)
    # 检查转换后的结果是否与预期结果非常接近

    feature_names = ohe.get_feature_names_out()
    # 获取输出特征的名称列表

    assert_array_equal(["x0_infrequent_sklearn"], feature_names)
    # 检查输出特征的名称列表是否符合预期

    X_inverse = ohe.inverse_transform(X_trans)
    # 对转换后的数据进行逆转换

    assert_array_equal([["b"], ["infrequent_sklearn"]], X_inverse)
    # 检查逆转换后的结果是否与预期的结果相等


@pytest.mark.parametrize("drop", [["a"], ["d"]])
# 使用 pytest 的 parametrize 装饰器为测试函数提供参数化测试的支持，drop 参数被设置为两个不同的测试参数

def test_ohe_infrequent_two_levels_drop_infrequent_errors(drop):
    """Test two levels and dropping any infrequent category removes the
    whole infrequent category."""
    # 测试删除任何罕见类别后是否删除整个罕见类别的功能

    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    # 创建一个包含多个类别的训练数据集 X_train

    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        max_categories=2,
        drop=drop,
    )

    msg = f"Unable to drop category {drop[0]!r} from feature 0 because it is infrequent"
    with pytest.raises(ValueError, match=msg):
        ohe.fit(X_train)
    # 检查在尝试删除罕见类别时是否引发预期的 ValueError 异常


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_categories": 3},
        {"min_frequency": 6},
        {"min_frequency": 9},
        {"min_frequency": 0.24},
        {"min_frequency": 0.16},
        {"max_categories": 3, "min_frequency": 8},
        {"max_categories": 4, "min_frequency": 6},
    ],
)
# 使用 pytest 的 parametrize 装饰器为测试函数提供参数化测试的支持，kwargs 参数包含多个不同的测试参数

def test_ohe_infrequent_three_levels(kwargs):
    """Test that different parameters for combing 'a', and 'd' into
    the infrequent category works as expected."""
    # 测试不同参数组合时，将 'a' 和 'd' 合并到罕见类别的功能是否按预期工作
    # 创建一个包含不同类别的数组 X_train，转置以符合 OneHotEncoder 的要求
    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    
    # 初始化 OneHotEncoder 对象 ohe，处理未知类别为 "infrequent_if_exist"，不使用稀疏矩阵表示，使用额外的参数 kwargs
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist", sparse_output=False, **kwargs
    ).fit(X_train)
    
    # 断言 ohe 对象中的 infrequent_categories_ 与预期相等
    assert_array_equal(ohe.infrequent_categories_, [["a", "d"]])

    # 准备用于转换的测试数据 X_test
    X_test = [["b"], ["a"], ["c"], ["d"], ["e"]]
    
    # 预期的转换结果
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1]])

    # 使用 ohe 对象对 X_test 进行转换得到 X_trans
    X_trans = ohe.transform(X_test)
    
    # 断言转换后的结果与预期结果非常接近
    assert_allclose(expected, X_trans)

    # 预期的逆转换结果
    expected_inv = [
        ["b"],
        ["infrequent_sklearn"],
        ["c"],
        ["infrequent_sklearn"],
        ["infrequent_sklearn"],
    ]
    
    # 使用 ohe 对象进行逆转换得到 X_inv
    X_inv = ohe.inverse_transform(X_trans)
    
    # 断言逆转换后的结果与预期结果相等
    assert_array_equal(expected_inv, X_inv)

    # 获取 OneHotEncoder 对象 ohe 的输出特征名称
    feature_names = ohe.get_feature_names_out()
    
    # 断言输出特征名称与预期的特征名称数组相等
    assert_array_equal(["x0_b", "x0_c", "x0_infrequent_sklearn"], feature_names)
@pytest.mark.parametrize("drop", ["first", ["b"]])
def test_ohe_infrequent_three_levels_drop_frequent(drop):
    """Test three levels and dropping the frequent category."""

    # 创建一个包含五个 'a'、二十个 'b'、十个 'c' 和三个 'd' 的特征矩阵
    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T

    # 初始化 OneHotEncoder 对象，设置 handle_unknown="infrequent_if_exist"，sparse_output=False，max_categories=3 和 drop=drop
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        max_categories=3,
        drop=drop,
    ).fit(X_train)

    # 创建测试数据 X_test 包含 ["b"], ["c"], ["d"]，并验证转换后的结果是否符合预期
    X_test = np.array([["b"], ["c"], ["d"]])
    assert_allclose([[0, 0], [1, 0], [0, 1]], ohe.transform(X_test))

    # 检查 handle_unknown="ignore"
    ohe.set_params(handle_unknown="ignore").fit(X_train)

    # 设置警告消息，并确保在转换未知类别时触发 UserWarning
    msg = "Found unknown categories"
    with pytest.warns(UserWarning, match=msg):
        X_trans = ohe.transform([["b"], ["e"]])

    assert_allclose([[0, 0], [0, 0]], X_trans)


@pytest.mark.parametrize("drop", [["a"], ["d"]])
def test_ohe_infrequent_three_levels_drop_infrequent_errors(drop):
    """Test three levels and dropping the infrequent category."""

    # 创建一个包含五个 'a'、二十个 'b'、十个 'c' 和三个 'd' 的特征矩阵
    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T

    # 初始化 OneHotEncoder 对象，设置 handle_unknown="infrequent_if_exist"，sparse_output=False，max_categories=3 和 drop=drop
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        max_categories=3,
        drop=drop,
    )

    # 设置异常消息，并确保在尝试删除 infrequent 类别时触发 ValueError
    msg = f"Unable to drop category {drop[0]!r} from feature 0 because it is infrequent"
    with pytest.raises(ValueError, match=msg):
        ohe.fit(X_train)


def test_ohe_infrequent_handle_unknown_error():
    """Test that different parameters for combining 'a', and 'd' into
    the infrequent category works as expected."""

    # 创建一个包含五个 'a'、二十个 'b'、十个 'c' 和三个 'd' 的特征矩阵
    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T

    # 初始化 OneHotEncoder 对象，设置 handle_unknown="error"，sparse_output=False 和 max_categories=3
    ohe = OneHotEncoder(
        handle_unknown="error", sparse_output=False, max_categories=3
    ).fit(X_train)

    # 验证 infrequent_categories_ 的正确性，确保包含 ["a", "d"]
    assert_array_equal(ohe.infrequent_categories_, [["a", "d"]])

    # 设置测试数据 X_test 包含 ["b"], ["a"], ["c"], ["d"]，并验证转换后的结果是否符合预期
    X_test = [["b"], ["a"], ["c"], ["d"]]
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])

    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)

    # 设置测试数据 X_test 包含 ["bad"]，并验证在发现未知类别时是否触发 ValueError
    X_test = [["bad"]]
    msg = r"Found unknown categories \['bad'\] in column 0"
    with pytest.raises(ValueError, match=msg):
        ohe.transform(X_test)


@pytest.mark.parametrize(
    "kwargs", [{"max_categories": 3, "min_frequency": 1}, {"min_frequency": 4}]
)
def test_ohe_infrequent_two_levels_user_cats_one_frequent(kwargs):
    """'a' is the only frequent category, all other categories are infrequent."""

    # 创建一个包含五个 'a' 和三十个 'e' 的特征矩阵
    X_train = np.array([["a"] * 5 + ["e"] * 30], dtype=object).T

    # 初始化 OneHotEncoder 对象，设置 categories=[["c", "d", "a", "b"]]，sparse_output=False，handle_unknown="infrequent_if_exist" 和传入的 kwargs 参数
    ohe = OneHotEncoder(
        categories=[["c", "d", "a", "b"]],
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
        **kwargs,
    ).fit(X_train)

    # 设置测试数据 X_test 包含 ["a"], ["b"], ["c"], ["d"], ["e"]，并验证转换后的结果是否符合预期
    X_test = [["a"], ["b"], ["c"], ["d"], ["e"]]
    expected = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])

    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)

    # 设置变量 drops 包含 ["first"], ["if_binary"], ["a"]，这些用于后续测试
    drops = ["first", "if_binary", ["a"]]
    # 定义测试数据集 X_test，包含两个子列表 ["a"] 和 ["c"]
    X_test = [["a"], ["c"]]
    # 遍历 drops 列表中的每个元素 drop
    for drop in drops:
        # 设置 OneHotEncoder 对象的 drop 参数为当前 drop 值，并拟合于 X_train 数据集
        ohe.set_params(drop=drop).fit(X_train)
        # 使用拟合后的 OneHotEncoder 对象 ohe 对 X_test 进行转换，并断言转换结果与预期结果相近
        assert_allclose([[0], [1]], ohe.transform(X_test))
def test_ohe_infrequent_two_levels_user_cats():
    """Test that the order of the categories provided by a user is respected."""
    # 创建一个包含多个重复值和各种类别的数组，并转置以便每行表示一个特征值
    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object
    ).T
    # 使用OneHotEncoder进行独热编码，指定用户提供的类别顺序，并设置其他参数
    ohe = OneHotEncoder(
        categories=[["c", "d", "a", "b"]],
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
        max_categories=2,
    ).fit(X_train)

    # 断言infrequent_categories_属性的值与预期相等
    assert_array_equal(ohe.infrequent_categories_, [["c", "d", "a"]])

    # 定义测试数据和预期输出
    X_test = [["b"], ["a"], ["c"], ["d"], ["e"]]
    expected = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])

    # 对测试数据进行转换，并断言转换结果与预期输出相近
    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)

    # 'infrequent'用于标记反向转换中的罕见类别
    expected_inv = [[col] for col in ["b"] + ["infrequent_sklearn"] * 4]
    # 对转换后的数据进行反向转换，并断言反向转换结果与预期输出相等
    X_inv = ohe.inverse_transform(X_trans)
    assert_array_equal(expected_inv, X_inv)


def test_ohe_infrequent_three_levels_user_cats():
    """Test that the order of the categories provided by a user is respected.
    In this case 'c' is encoded as the first category and 'b' is encoded
    as the second one."""
    # 创建一个包含多个重复值和各种类别的数组，并转置以便每行表示一个特征值
    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object
    ).T
    # 使用OneHotEncoder进行独热编码，指定用户提供的类别顺序，并设置其他参数
    ohe = OneHotEncoder(
        categories=[["c", "d", "b", "a"]],
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
        max_categories=3,
    ).fit(X_train)

    # 断言infrequent_categories_属性的值与预期相等
    assert_array_equal(ohe.infrequent_categories_, [["d", "a"]])

    # 定义测试数据和预期输出
    X_test = [["b"], ["a"], ["c"], ["d"], ["e"]]
    expected = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]])

    # 对测试数据进行转换，并断言转换结果与预期输出相近
    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)

    # 'infrequent'用于标记反向转换中的罕见类别
    expected_inv = [
        ["b"],
        ["infrequent_sklearn"],
        ["c"],
        ["infrequent_sklearn"],
        ["infrequent_sklearn"],
    ]
    # 对转换后的数据进行反向转换，并断言反向转换结果与预期输出相等
    X_inv = ohe.inverse_transform(X_trans)
    assert_array_equal(expected_inv, X_inv)


def test_ohe_infrequent_mixed():
    """Test infrequent categories where feature 0 has infrequent categories,
    and feature 1 does not."""
    # 创建包含多个特征的数组，其中第一个特征有罕见的类别，第二个特征没有
    X = np.c_[[0, 1, 3, 3, 3, 3, 2, 0, 3], [0, 0, 0, 0, 1, 1, 1, 1, 1]]

    # 使用OneHotEncoder进行独热编码，设置其他参数
    ohe = OneHotEncoder(max_categories=3, drop="if_binary", sparse_output=False)
    ohe.fit(X)

    # 定义测试数据和预期输出
    X_test = [[3, 0], [1, 1]]
    X_trans = ohe.transform(X_test)

    # 断言转换后的数据与预期输出相等
    assert_allclose(X_trans, [[0, 1, 0, 0], [0, 0, 1, 1]])


def test_ohe_infrequent_multiple_categories():
    """Test infrequent categories with feature matrix with 3 features."""
    # 创建包含多个特征的数组
    X = np.c_[
        [0, 1, 3, 3, 3, 3, 2, 0, 3],
        [0, 0, 5, 1, 1, 10, 5, 5, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1],
    ]

    # 使用OneHotEncoder进行独热编码，设置其他参数
    ohe = OneHotEncoder(
        categories="auto", max_categories=3, handle_unknown="infrequent_if_exist"
    )
    )
    # X[:, 0] 1 and 2 are infrequent
    # X[:, 1] 1 and 10 are infrequent
    # X[:, 2] nothing is infrequent

    # 使用OneHotEncoder对输入数据X进行编码转换，并转换为稀疏矩阵的数组表示
    X_trans = ohe.fit_transform(X).toarray()
    # 断言验证第一个列的稀疏编码中的非频繁类别为[1, 2]
    assert_array_equal(ohe.infrequent_categories_[0], [1, 2])
    # 断言验证第二个列的稀疏编码中的非频繁类别为[1, 10]
    assert_array_equal(ohe.infrequent_categories_[1], [1, 10])
    # 断言验证第三个列的稀疏编码中没有非频繁类别（即为None）
    assert_array_equal(ohe.infrequent_categories_[2], None)

    # 'infrequent' 用于表示非频繁类别
    # 对于第一列，1和2具有相同的频率。在这种情况下，选择1作为特征名称，因为在字典序中较小
    feature_names = ohe.get_feature_names_out()
    # 断言验证生成的特征名列表与期望的列表是否一致
    assert_array_equal(
        [
            "x0_0",
            "x0_3",
            "x0_infrequent_sklearn",
            "x1_0",
            "x1_5",
            "x1_infrequent_sklearn",
            "x2_0",
            "x2_1",
        ],
        feature_names,
    )

    # 期望的稀疏矩阵表示
    expected = [
        [1, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1],
    ]

    # 断言验证转换后的稀疏矩阵是否与期望的矩阵一致
    assert_allclose(expected, X_trans)

    # 测试数据集X_test的转换
    X_test = [[3, 1, 2], [4, 0, 3]]

    # 对测试数据集X_test进行转换
    X_test_trans = ohe.transform(X_test)

    # X[:, 2] 没有非频繁类别，因此编码为全零向量
    # 期望的稀疏矩阵表示
    expected = [[0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]]
    # 断言验证转换后的稀疏矩阵是否与期望的矩阵一致
    assert_allclose(expected, X_test_trans.toarray())

    # 对转换后的稀疏矩阵进行逆转换
    X_inv = ohe.inverse_transform(X_test_trans)
    # 期望的逆转换结果矩阵
    expected_inv = np.array(
        [[3, "infrequent_sklearn", None], ["infrequent_sklearn", 0, None]], dtype=object
    )
    # 断言验证逆转换后的结果矩阵是否与期望的矩阵一致
    assert_array_equal(expected_inv, X_inv)

    # 处理未知类别时的错误情况
    # 使用OneHotEncoder对数据集X进行拟合，设置未知类别处理方式为错误（error）
    ohe = OneHotEncoder(
        categories="auto", max_categories=3, handle_unknown="error"
    ).fit(X)
    # 使用pytest断言验证在转换未知类别时是否引发了预期的ValueError异常
    with pytest.raises(ValueError, match="Found unknown categories"):
        ohe.transform(X_test)

    # 只接受非频繁或已知类别的测试数据集X_test
    X_test = [[1, 1, 1], [3, 10, 0]]
    # 对测试数据集X_test进行转换
    X_test_trans = ohe.transform(X_test)

    # 期望的稀疏矩阵表示
    expected = [[0, 0, 1, 0, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1, 1, 0]]
    # 断言验证转换后的稀疏矩阵是否与期望的矩阵一致
    assert_allclose(expected, X_test_trans.toarray())

    # 对转换后的稀疏矩阵进行逆转换
    X_inv = ohe.inverse_transform(X_test_trans)

    # 期望的逆转换结果矩阵
    expected_inv = np.array(
        [["infrequent_sklearn", "infrequent_sklearn", 1], [3, "infrequent_sklearn", 0]],
        dtype=object,
    )
    # 断言验证逆转换后的结果矩阵是否与期望的矩阵一致
    assert_array_equal(expected_inv, X_inv)
# 定义测试函数，用于测试处理包含多种数据类型的 Pandas 数据帧中的稀有类别
def test_ohe_infrequent_multiple_categories_dtypes():
    """Test infrequent categories with a pandas dataframe with multiple dtypes."""

    # 导入 pytest，并在没有安装时跳过测试
    pd = pytest.importorskip("pandas")
    
    # 创建包含字符串和整数列的 Pandas 数据帧 X
    X = pd.DataFrame(
        {
            "str": ["a", "f", "c", "f", "f", "a", "c", "b", "b"],
            "int": [5, 3, 0, 10, 10, 12, 0, 3, 5],
        },
        columns=["str", "int"],
    )

    # 创建 OneHotEncoder 对象 ohe，设置处理策略为 infrequent_if_exist
    ohe = OneHotEncoder(
        categories="auto", max_categories=3, handle_unknown="infrequent_if_exist"
    )

    # 对数据帧 X 进行独热编码转换，得到稀疏矩阵 X_trans
    X_trans = ohe.fit_transform(X).toarray()
    
    # 断言检查每列的稀有类别列表是否符合预期
    assert_array_equal(ohe.infrequent_categories_[0], ["a", "b"])
    assert_array_equal(ohe.infrequent_categories_[1], [0, 3, 12])

    # 预期的独热编码结果
    expected = [
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 0],
    ]

    # 断言检查转换后的矩阵 X_trans 是否与预期结果一致
    assert_allclose(expected, X_trans)

    # 创建测试用的新数据帧 X_test，进行独热编码转换
    X_test = pd.DataFrame({"str": ["b", "f"], "int": [14, 12]}, columns=["str", "int"])
    expected = [[0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1]]
    X_test_trans = ohe.transform(X_test)
    
    # 断言检查转换后的矩阵 X_test_trans 是否与预期结果一致
    assert_allclose(expected, X_test_trans.toarray())

    # 对 X_test_trans 进行逆转换，得到原始数据
    X_inv = ohe.inverse_transform(X_test_trans)
    
    # 预期的逆转换结果
    expected_inv = np.array(
        [["infrequent_sklearn", "infrequent_sklearn"], ["f", "infrequent_sklearn"]],
        dtype=object,
    )
    
    # 断言检查逆转换后的结果是否与预期一致
    assert_array_equal(expected_inv, X_inv)

    # 创建只包含已知或稀有类别的测试数据帧 X_test，进行独热编码转换
    X_test = pd.DataFrame({"str": ["c", "b"], "int": [12, 5]}, columns=["str", "int"])
    X_test_trans = ohe.transform(X_test).toarray()
    
    # 预期的独热编码结果
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0]]
    
    # 断言检查转换后的矩阵 X_test_trans 是否与预期结果一致
    assert_allclose(expected, X_test_trans)

    # 对 X_test_trans 进行逆转换，得到原始数据
    X_inv = ohe.inverse_transform(X_test_trans)
    
    # 预期的逆转换结果
    expected_inv = np.array(
        [["c", "infrequent_sklearn"], ["infrequent_sklearn", 5]], dtype=object
    )
    
    # 断言检查逆转换后的结果是否与预期一致
    assert_array_equal(expected_inv, X_inv)


# 使用 pytest 的参数化装饰器，定义测试函数 test_ohe_infrequent_one_level_errors
@pytest.mark.parametrize("kwargs", [{"min_frequency": 21, "max_categories": 1}])
def test_ohe_infrequent_one_level_errors(kwargs):
    """All user provided categories are infrequent."""

    # 创建包含大量稀有类别的训练数据 X_train
    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 2]).T
    
    # 创建 OneHotEncoder 对象 ohe，设置处理策略为 infrequent_if_exist
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist", sparse_output=False, **kwargs
    )
    
    # 对数据 X_train 进行拟合
    ohe.fit(X_train)

    # 对单个样本进行转换，得到独热编码结果 X_trans
    X_trans = ohe.transform([["a"]])
    
    # 断言检查转换后的结果是否符合预期
    assert_allclose(X_trans, [[1]])


# 使用 pytest 的参数化装饰器，定义测试函数 test_ohe_infrequent_user_cats_unknown_training_errors
@pytest.mark.parametrize("kwargs", [{"min_frequency": 2, "max_categories": 3}])
def test_ohe_infrequent_user_cats_unknown_training_errors(kwargs):
    """All user provided categories are infrequent."""

    # 创建只包含稀有类别的训练数据 X_train
    X_train = np.array([["e"] * 3], dtype=object).T
    # 创建一个独热编码器对象，指定编码的类别为 ["c", "d", "a", "b"]，输出稠密矩阵而非稀疏矩阵，
    # 并指定未知类别处理策略为 "infrequent_if_exist"（如果存在则处理为罕见类别），同时传递其他关键字参数
    ohe = OneHotEncoder(
        categories=[["c", "d", "a", "b"]],
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
        **kwargs,
    ).fit(X_train)

    # 使用训练集 X_train 对独热编码器 ohe 进行拟合，学习类别映射关系
    X_trans = ohe.transform([["a"], ["e"]])

    # 断言转换后的结果 X_trans 与预期的独热编码值 [[1], [1]] 非常接近
    assert_allclose(X_trans, [[1], [1]])
# 在参数化测试中，指定输入和类别数据类型的组合进行测试
@pytest.mark.parametrize(
    "input_dtype, category_dtype", ["OO", "OU", "UO", "UU", "SO", "SU", "SS"]
)
# 参数化测试，对不同的数组类型进行测试
@pytest.mark.parametrize("array_type", ["list", "array", "dataframe"])
def test_encoders_string_categories(input_dtype, category_dtype, array_type):
    """Check that encoding work with object, unicode, and byte string dtypes.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/15616
    https://github.com/scikit-learn/scikit-learn/issues/15726
    https://github.com/scikit-learn/scikit-learn/issues/19677
    """

    # 创建输入数组 X，使用指定的输入数据类型
    X = np.array([["b"], ["a"]], dtype=input_dtype)
    # 创建类别数组，使用指定的类别数据类型
    categories = [np.array(["b", "a"], dtype=category_dtype)]
    # 使用 OneHotEncoder 对象进行编码，禁用稀疏输出，对 X 进行拟合
    ohe = OneHotEncoder(categories=categories, sparse_output=False).fit(X)

    # 创建测试用例 X_test，转换为指定的数组类型和数据类型
    X_test = _convert_container(
        [["a"], ["a"], ["b"], ["a"]], array_type, dtype=input_dtype
    )
    # 对 X_test 进行编码转换
    X_trans = ohe.transform(X_test)

    # 预期的转换结果
    expected = np.array([[0, 1], [0, 1], [1, 0], [0, 1]])
    # 断言实际转换结果与预期结果相等
    assert_allclose(X_trans, expected)

    # 使用 OrdinalEncoder 对象进行编码，对 X 进行拟合
    oe = OrdinalEncoder(categories=categories).fit(X)
    # 再次对 X_test 进行编码转换
    X_trans = oe.transform(X_test)

    # 另一个预期的转换结果
    expected = np.array([[1], [1], [0], [1]])
    # 断言实际转换结果与预期结果相等
    assert_array_equal(X_trans, expected)


def test_mixed_string_bytes_categoricals():
    """Check that this mixture of predefined categories and X raises an error.

    Categories defined as bytes can not easily be compared to data that is
    a string.
    """
    # 使用 Unicode 数据创建输入数组 X
    X = np.array([["b"], ["a"]], dtype="U")
    # 使用 Bytes 类型创建预定义的类别数组
    categories = [np.array(["b", "a"], dtype="S")]
    # 使用 OneHotEncoder 对象，传入类别数组并禁用稀疏输出
    ohe = OneHotEncoder(categories=categories, sparse_output=False)

    # 期望捕获的错误信息
    msg = re.escape(
        "In column 0, the predefined categories have type 'bytes' which is incompatible"
        " with values of type 'str_'."
    )

    # 使用 pytest.raises 断言捕获 ValueError，并验证错误信息
    with pytest.raises(ValueError, match=msg):
        ohe.fit(X)


@pytest.mark.parametrize("missing_value", [np.nan, None])
def test_ohe_missing_values_get_feature_names(missing_value):
    # 使用对象类型的数组 X，包含缺失值，进行编码器测试
    X = np.array([["a", "b", missing_value, "a", missing_value]], dtype=object).T
    # 使用 OneHotEncoder 对象，禁用稀疏输出，并忽略未知值进行拟合
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X)
    # 获取编码后的特征名称
    names = ohe.get_feature_names_out()
    # 断言特征名称与预期结果相等
    assert_array_equal(names, ["x0_a", "x0_b", f"x0_{missing_value}"])


def test_ohe_missing_value_support_pandas():
    # 检查对 Pandas 支持，包含混合数据类型和缺失值的情况
    pd = pytest.importorskip("pandas")
    # 创建包含混合数据类型和缺失值的 DataFrame
    df = pd.DataFrame(
        {
            "col1": ["dog", "cat", None, "cat"],
            "col2": np.array([3, 0, 4, np.nan], dtype=float),
        },
        columns=["col1", "col2"],
    )
    # 预期的 DataFrame 转换结果
    expected_df_trans = np.array(
        [
            [0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
        ]
    )

    # 使用 check_categorical_onehot 函数对 DataFrame 进行编码转换
    Xtr = check_categorical_onehot(df)
    # 断言转换后的结果与预期结果相等
    assert_allclose(Xtr, expected_df_trans)
@pytest.mark.parametrize("handle_unknown", ["infrequent_if_exist", "ignore"])
@pytest.mark.parametrize("pd_nan_type", ["pd.NA", "np.nan"])
def test_ohe_missing_value_support_pandas_categorical(pd_nan_type, handle_unknown):
    # 导入并检查是否存在 pandas 库
    pd = pytest.importorskip("pandas")

    # 根据 pd_nan_type 的值选择合适的缺失值表示
    pd_missing_value = pd.NA if pd_nan_type == "pd.NA" else np.nan

    # 创建一个包含分类特征的 pandas 数据帧
    df = pd.DataFrame(
        {
            "col1": pd.Series(["c", "a", pd_missing_value, "b", "a"], dtype="category"),
        }
    )
    
    # 预期的转换后的数组表示
    expected_df_trans = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )

    # 创建一个 OneHotEncoder 对象，并进行数据帧的转换
    ohe = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
    df_trans = ohe.fit_transform(df)
    assert_allclose(expected_df_trans, df_trans)

    # 断言编码器的分类数量为1
    assert len(ohe.categories_) == 1
    # 断言编码器的第一个分类的元素除了最后一个是["a", "b", "c"]
    assert_array_equal(ohe.categories_[0][:-1], ["a", "b", "c"])
    # 断言编码器的第一个分类的最后一个元素是 NaN
    assert np.isnan(ohe.categories_[0][-1])


@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
def test_ohe_drop_first_handle_unknown_ignore_warns(handle_unknown):
    """Check drop='first' and handle_unknown='ignore'/'infrequent_if_exist'
    during transform."""
    # 输入数据集
    X = [["a", 0], ["b", 2], ["b", 1]]

    # 创建一个 OneHotEncoder 对象，设置 drop='first' 和 handle_unknown 参数
    ohe = OneHotEncoder(
        drop="first", sparse_output=False, handle_unknown=handle_unknown
    )
    X_trans = ohe.fit_transform(X)

    # 预期的转换后的数组表示
    X_expected = np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )
    assert_allclose(X_trans, X_expected)

    # 测试数据集 X_test 包含未知的分类
    X_test = [["c", 3]]
    X_expected = np.array([[0, 0, 0]])

    # 断言在转换时，发出 UserWarning 警告，并检查警告信息
    warn_msg = (
        r"Found unknown categories in columns \[0, 1\] during "
        "transform. These unknown categories will be encoded as all "
        "zeros"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        X_trans = ohe.transform(X_test)
    assert_allclose(X_trans, X_expected)

    # 反向转换测试，将编码后的数组映射回原始形式
    X_inv = ohe.inverse_transform(X_expected)
    assert_array_equal(X_inv, np.array([["a", 0]], dtype=object))


@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
def test_ohe_drop_if_binary_handle_unknown_ignore_warns(handle_unknown):
    """Check drop='if_binary' and handle_unknown='ignore' during transform."""
    # 输入数据集
    X = [["a", 0], ["b", 2], ["b", 1]]

    # 创建一个 OneHotEncoder 对象，设置 drop='if_binary' 和 handle_unknown 参数
    ohe = OneHotEncoder(
        drop="if_binary", sparse_output=False, handle_unknown=handle_unknown
    )
    X_trans = ohe.fit_transform(X)

    # 预期的转换后的数组表示
    X_expected = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    assert_allclose(X_trans, X_expected)

    # 测试数据集 X_test 包含未知的分类
    X_test = [["c", 3]]
    X_expected = np.array([[0, 0, 0, 0]])
    # 定义警告信息，指出在转换过程中发现了未知的分类列[0, 1]，这些未知分类将被编码为全零
    warn_msg = (
        r"Found unknown categories in columns \[0, 1\] during "
        "transform. These unknown categories will be encoded as all "
        "zeros"
    )
    
    # 使用 pytest 的 warn 函数捕获 UserWarning，并匹配指定的警告信息 warn_msg
    with pytest.warns(UserWarning, match=warn_msg):
        # 对测试集 X_test 进行独热编码转换
        X_trans = ohe.transform(X_test)
    
    # 使用 assert_allclose 函数断言 X_trans 和预期的 X_expected 在数值上是接近的
    assert_allclose(X_trans, X_expected)

    # 使用 inverse_transform 函数将 X_expected 反向转换回原始数据
    X_inv = ohe.inverse_transform(X_expected)
    
    # 使用 assert_array_equal 函数断言 X_inv 和预期的 numpy 数组相等，其中包括了一个值为 ["a", None] 的对象类型数组
    assert_array_equal(X_inv, np.array([["a", None]], dtype=object))
@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
# 使用 pytest 的参数化装饰器，测试 handle_unknown 参数在 "ignore" 和 "infrequent_if_exist" 两种情况下的行为
def test_ohe_drop_first_explicit_categories(handle_unknown):
    """Check drop='first' and handle_unknown='ignore'/'infrequent_if_exist'
    during fit with categories passed in."""
    # 检查在指定 categories 的情况下，drop='first' 和 handle_unknown='ignore'/'infrequent_if_exist' 的行为

    X = [["a", 0], ["b", 2], ["b", 1]]

    ohe = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown=handle_unknown,
        categories=[["b", "a"], [1, 2]],
    )
    # 创建 OneHotEncoder 对象，设置 drop='first'、sparse_output=False 和 handle_unknown 参数，指定 categories
    ohe.fit(X)
    # 对输入数据 X 进行拟合

    X_test = [["c", 1]]
    X_expected = np.array([[0, 0]])

    warn_msg = (
        r"Found unknown categories in columns \[0\] during transform. "
        r"These unknown categories will be encoded as all zeros"
    )
    # 定义警告信息，指示在转换期间在列 [0] 中找到未知的类别，这些未知的类别将被编码为全零

    with pytest.warns(UserWarning, match=warn_msg):
        X_trans = ohe.transform(X_test)
    # 在转换 X_test 时，捕获预期的警告信息，进行转换并赋值给 X_trans

    assert_allclose(X_trans, X_expected)
    # 断言转换后的结果 X_trans 与预期结果 X_expected 非常接近


def test_ohe_more_informative_error_message():
    """Raise informative error message when pandas output and sparse_output=True."""
    # 在输出为 Pandas 且 sparse_output=True 时，提出更具信息性的错误消息

    pd = pytest.importorskip("pandas")
    # 导入 pandas 库，如果不存在则跳过测试
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["z", "b", "b"]}, columns=["a", "b"])

    ohe = OneHotEncoder(sparse_output=True)
    # 创建 sparse_output=True 的 OneHotEncoder 对象
    ohe.set_output(transform="pandas")
    # 设置输出为 Pandas 格式

    msg = (
        "Pandas output does not support sparse data. Set "
        "sparse_output=False to output pandas dataframes or disable Pandas output"
    )
    # 定义错误消息，指示 Pandas 输出不支持稀疏数据

    with pytest.raises(ValueError, match=msg):
        ohe.fit_transform(df)
    # 断言在拟合转换时，捕获预期的 ValueError 错误并匹配消息

    ohe.fit(df)
    with pytest.raises(ValueError, match=msg):
        ohe.transform(df)
    # 断言在转换时，捕获预期的 ValueError 错误并匹配消息


def test_ordinal_encoder_passthrough_missing_values_float_errors_dtype():
    """Test ordinal encoder with nan passthrough fails when dtype=np.int32."""
    # 测试当 dtype=np.int32 时，带有 NaN 传递的序数编码器失败的情况

    X = np.array([[np.nan, 3.0, 1.0, 3.0]]).T
    # 创建包含 NaN 的 numpy 数组 X
    oe = OrdinalEncoder(dtype=np.int32)
    # 创建 dtype=np.int32 的 OrdinalEncoder 对象

    msg = (
        r"There are missing values in features \[0\]. For OrdinalEncoder "
        f"to encode missing values with dtype: {np.int32}"
    )
    # 定义错误消息，指示在特征 [0] 中存在缺失值

    with pytest.raises(ValueError, match=msg):
        oe.fit(X)
    # 断言在拟合时，捕获预期的 ValueError 错误并匹配消息


@pytest.mark.parametrize("encoded_missing_value", [np.nan, -2])
# 使用 pytest 的参数化装饰器，测试 encoded_missing_value 参数分别为 np.nan 和 -2 的情况
def test_ordinal_encoder_passthrough_missing_values_float(encoded_missing_value):
    """Test ordinal encoder with nan on float dtypes."""
    # 测试在浮点数数据类型上，使用 NaN 的序数编码器

    X = np.array([[np.nan, 3.0, 1.0, 3.0]], dtype=np.float64).T
    # 创建包含 NaN 的浮点数类型的 numpy 数组 X
    oe = OrdinalEncoder(encoded_missing_value=encoded_missing_value).fit(X)
    # 创建具有指定 encoded_missing_value 参数的 OrdinalEncoder 对象，并进行拟合

    assert len(oe.categories_) == 1
    # 断言 oe.categories_ 的长度为 1

    assert_allclose(oe.categories_[0], [1.0, 3.0, np.nan])
    # 断言 oe.categories_[0] 与预期的 [1.0, 3.0, np.nan] 非常接近

    X_trans = oe.transform(X)
    # 进行转换并赋值给 X_trans
    assert_allclose(X_trans, [[encoded_missing_value], [1.0], [0.0], [1.0]])
    # 断言转换后的结果 X_trans 与预期结果非常接近

    X_inverse = oe.inverse_transform(X_trans)
    # 进行逆转换并赋值给 X_inverse
    assert_allclose(X_inverse, X)
    # 断言逆转换后的结果 X_inverse 与原始输入 X 非常接近


@pytest.mark.parametrize("pd_nan_type", ["pd.NA", "np.nan"])
@pytest.mark.parametrize("encoded_missing_value", [np.nan, -2])
# 使用 pytest 的参数化装饰器，测试 pd_nan_type 和 encoded_missing_value 参数的多种组合
def test_ordinal_encoder_missing_value_support_pandas_categorical(
    pd_nan_type, encoded_missing_value
):
    """Check ordinal encoder is compatible with pandas."""
    # 检查序数编码器与 pandas 的兼容性

    pd = pytest.importorskip("pandas")
    # 导入 pandas 库，如果不存在则跳过测试

    # 检查序数编码器与 pandas 的兼容性，尤其是处理分类特征中的缺失值情况
    # 检查 pandas 数据框中包含分类特征时的行为

    # 使用 pd_nan_type 参数指定的缺失值类型创建 pandas 数据框
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["z", "b", "b"]}, columns=["a", "b"])

    # 创建 OrdinalEncoder 对象，设置 encoded_missing_value 参数和 dtype
    oe = OrdinalEncoder(encoded_missing_value=encoded_missing_value)

    # 对 pandas 数据框进行拟合，检查对缺失值的处理方式
    oe.fit(df)

    # 断言类别数量与预期一致
    assert len(oe.categories_) == 1

    # 断言编码后的类别与预期结果非常接近
    assert_allclose(oe.categories_[0], [1.0, 2.0, 3.0])

    # 对数据进行转换并检查转换结果与预期结果的接近程度
    X_trans = oe.transform(df)
    assert_allclose(X_trans, [[0.0, encoded_missing_value], [1.0, 1.0], [2.0, 1.0]])

    # 对转换后的数据进行逆转换，并检查逆转换后的结果与原始数据框的接近程度
    X_inverse = oe.inverse_transform(X_trans)
    assert_frame_equal(X_inverse, df)
    # 如果 pd_nan_type 等于 "pd.NA"，则 pd_missing_value 设置为 pd.NA，否则设置为 np.nan
    pd_missing_value = pd.NA if pd_nan_type == "pd.NA" else np.nan

    # 创建一个包含一个列 "col1" 的 DataFrame，其中包括了字符串和可能的缺失值
    df = pd.DataFrame(
        {
            "col1": pd.Series(["c", "a", pd_missing_value, "b", "a"], dtype="category"),
        }
    )

    # 使用 OrdinalEncoder 对象处理 DataFrame，用于将类别数据编码为数值，并设置编码后的缺失值
    oe = OrdinalEncoder(encoded_missing_value=encoded_missing_value).fit(df)
    # 断言编码器的类别数量为1
    assert len(oe.categories_) == 1
    # 断言编码器的第一个类别的前三个值分别为 "a", "b", "c"
    assert_array_equal(oe.categories_[0][:3], ["a", "b", "c"])
    # 断言编码器的第一个类别的最后一个值是 NaN
    assert np.isnan(oe.categories_[0][-1])

    # 使用编码器转换 DataFrame，生成转换后的数据
    df_trans = oe.transform(df)

    # 断言转换后的 DataFrame 的值接近于指定的值
    assert_allclose(df_trans, [[2.0], [0.0], [encoded_missing_value], [1.0], [0.0]])

    # 使用编码器进行逆转换，得到原始数据的近似逆转
    X_inverse = oe.inverse_transform(df_trans)
    # 断言逆转换后的数据形状为 (5, 1)
    assert X_inverse.shape == (5, 1)
    # 断言逆转换后的前两行第一列的值分别为 "c", "a"
    assert_array_equal(X_inverse[:2, 0], ["c", "a"])
    # 断言逆转换后的第四行及之后的第一列的值分别为 "b", "a"
    assert_array_equal(X_inverse[3:, 0], ["b", "a"])
    # 断言逆转换后的第三行第一列的值为 NaN
    assert np.isnan(X_inverse[2, 0])
@pytest.mark.parametrize(
    "X, X2, cats, cat_dtype",
    [  # 参数化测试用例，测试不同的输入组合
        (
            np.array([["a", np.nan]], dtype=object).T,  # 第一个测试数据 X
            np.array([["a", "b"]], dtype=object).T,     # 第一个测试数据 X2
            [np.array(["a", "d", np.nan], dtype=object)],  # 第一个测试数据的分类数组 cats
            np.object_,  # 第一个测试数据的分类数组的数据类型 cat_dtype
        ),
        (
            np.array([["a", np.nan]], dtype=object).T,  # 第二个测试数据 X
            np.array([["a", "b"]], dtype=object).T,     # 第二个测试数据 X2
            [np.array(["a", "d", np.nan], dtype=object)],  # 第二个测试数据的分类数组 cats
            np.object_,  # 第二个测试数据的分类数组的数据类型 cat_dtype
        ),
        (
            np.array([[2.0, np.nan]], dtype=np.float64).T,  # 第三个测试数据 X
            np.array([[3.0]], dtype=np.float64).T,          # 第三个测试数据 X2
            [np.array([2.0, 4.0, np.nan])],                 # 第三个测试数据的分类数组 cats
            np.float64,  # 第三个测试数据的分类数组的数据类型 cat_dtype
        ),
    ],
    ids=[
        "object-None-missing-value",
        "object-nan-missing_value",
        "numeric-missing-value",
    ],  # 参数化测试用例的标识符
)
def test_ordinal_encoder_specified_categories_missing_passthrough(
    X, X2, cats, cat_dtype
):
    """Test ordinal encoder for specified categories."""
    oe = OrdinalEncoder(categories=cats)  # 使用给定的分类数组初始化 OrdinalEncoder
    exp = np.array([[0.0], [np.nan]])  # 预期输出结果
    assert_array_equal(oe.fit_transform(X), exp)  # 断言 OrdinalEncoder 的转换结果与预期结果一致

    # 手动指定的分类数组在从列表转换时应与数据的 dtype 保持一致
    assert oe.categories_[0].dtype == cat_dtype

    # 当手动指定分类时，未知的分类应在拟合时引发异常
    oe = OrdinalEncoder(categories=cats)  # 使用给定的分类数组初始化 OrdinalEncoder
    with pytest.raises(ValueError, match="Found unknown categories"):
        oe.fit(X2)  # 断言拟合过程中会引发 ValueError 异常，提示找到未知的分类


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
def test_encoder_duplicate_specified_categories(Encoder):
    """Test encoder for specified categories have duplicate values.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27088
    """
    cats = [np.array(["a", "b", "a"], dtype=object)]  # 包含重复值的分类数组
    enc = Encoder(categories=cats)  # 使用给定的分类数组初始化 Encoder
    X = np.array([["a", "b"]], dtype=object).T  # 输入数据 X
    with pytest.raises(
        ValueError, match="the predefined categories contain duplicate elements."
    ):
        enc.fit(X)  # 断言拟合过程中会引发 ValueError 异常，提示分类数组包含重复元素


@pytest.mark.parametrize(
    "X, expected_X_trans, X_test",
    [  # 参数化测试用例，测试不同的输入组合
        (
            np.array([[1.0, np.nan, 3.0]]).T,  # 第一个测试数据 X
            np.array([[0.0, np.nan, 1.0]]).T,  # 第一个预期转换后的输出 expected_X_trans
            np.array([[4.0]]),  # 第一个测试数据 X_test
        ),
        (
            np.array([[1.0, 4.0, 3.0]]).T,  # 第二个测试数据 X
            np.array([[0.0, 2.0, 1.0]]).T,  # 第二个预期转换后的输出 expected_X_trans
            np.array([[np.nan]]),  # 第二个测试数据 X_test
        ),
        (
            np.array([["c", np.nan, "b"]], dtype=object).T,  # 第三个测试数据 X
            np.array([[1.0, np.nan, 0.0]]).T,  # 第三个预期转换后的输出 expected_X_trans
            np.array([["d"]], dtype=object),  # 第三个测试数据 X_test
        ),
        (
            np.array([["c", "a", "b"]], dtype=object).T,  # 第四个测试数据 X
            np.array([[2.0, 0.0, 1.0]]).T,  # 第四个预期转换后的输出 expected_X_trans
            np.array([[np.nan]], dtype=object),  # 第四个测试数据 X_test
        ),
    ],
)
def test_ordinal_encoder_handle_missing_and_unknown(X, expected_X_trans, X_test):
    # 这个测试函数还未完整，后续会补充完整的测试代码
    """Test the interaction between missing values and handle_unknown"""

    # 创建一个OrdinalEncoder对象，指定handle_unknown为"use_encoded_value"，unknown_value为-1
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    # 使用OrdinalEncoder对象对数据集X进行拟合和转换
    X_trans = oe.fit_transform(X)
    
    # 断言转换后的结果X_trans与期望的结果expected_X_trans非常接近
    assert_allclose(X_trans, expected_X_trans)

    # 使用已经拟合好的OrdinalEncoder对象oe对测试集X_test进行转换，并断言转换后的结果
    assert_allclose(oe.transform(X_test), [[-1.0]])
# 使用 pytest.mark.parametrize 装饰器，为 test_ordinal_encoder_sparse 函数参数化，使其可以多次运行
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_ordinal_encoder_sparse(csr_container):
    """Check that we raise proper error with sparse input in OrdinalEncoder.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19878
    """
    # 创建一个示例稀疏矩阵 X_sparse，使用 csr_container 将稠密矩阵 X 转换为稀疏格式
    X = np.array([[3, 2, 1], [0, 1, 1]])
    X_sparse = csr_container(X)

    # 初始化 OrdinalEncoder 对象
    encoder = OrdinalEncoder()

    # 定义错误消息，用于断言异常类型和消息匹配
    err_msg = "Sparse data was passed, but dense data is required"

    # 使用 pytest 的 raises 断言，检查是否正确抛出 TypeError 异常
    with pytest.raises(TypeError, match=err_msg):
        encoder.fit(X_sparse)
    with pytest.raises(TypeError, match=err_msg):
        encoder.fit_transform(X_sparse)

    # 对稠密矩阵 X 进行编码转换，确保也能正确处理稠密输入
    X_trans = encoder.fit_transform(X)
    X_trans_sparse = csr_container(X_trans)

    # 使用 pytest 的 raises 断言，再次检查是否正确抛出 TypeError 异常
    with pytest.raises(TypeError, match=err_msg):
        encoder.inverse_transform(X_trans_sparse)


def test_ordinal_encoder_fit_with_unseen_category():
    """Check OrdinalEncoder.fit works with unseen category when
    `handle_unknown="use_encoded_value"`.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19872
    """
    # 创建包含未见过的类别的输入数据 X
    X = np.array([0, 0, 1, 0, 2, 5])[:, np.newaxis]

    # 初始化 OrdinalEncoder 对象，使用 `handle_unknown="use_encoded_value"` 和自定义未知值 -999
    oe = OrdinalEncoder(
        categories=[[-1, 0, 1]], handle_unknown="use_encoded_value", unknown_value=-999
    )

    # 使用 fit 方法进行编码器的拟合，确保能处理未见过的类别
    oe.fit(X)

    # 初始化另一个 OrdinalEncoder 对象，使用 `handle_unknown="error"`，期望抛出 ValueError 异常
    oe = OrdinalEncoder(categories=[[-1, 0, 1]], handle_unknown="error")

    # 使用 pytest 的 raises 断言，检查是否正确抛出 ValueError 异常，提示发现未知类别
    with pytest.raises(ValueError, match="Found unknown categories"):
        oe.fit(X)


@pytest.mark.parametrize(
    "X_train",
    [
        [["AA", "B"]],
        np.array([["AA", "B"]], dtype="O"),
        np.array([["AA", "B"]], dtype="U"),
    ],
)
@pytest.mark.parametrize(
    "X_test",
    [
        [["A", "B"]],
        np.array([["A", "B"]], dtype="O"),
        np.array([["A", "B"]], dtype="U"),
    ],
)
def test_ordinal_encoder_handle_unknown_string_dtypes(X_train, X_test):
    """Checks that `OrdinalEncoder` transforms string dtypes.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19872
    """
    # 初始化 OrdinalEncoder 对象，使用 `handle_unknown="use_encoded_value"` 和自定义未知值 -9
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-9)

    # 使用 fit 方法对训练集 X_train 进行拟合，检查字符串数据类型的转换是否有效
    enc.fit(X_train)

    # 使用 transform 方法对测试集 X_test 进行转换，断言结果与预期一致
    X_trans = enc.transform(X_test)
    assert_allclose(X_trans, [[-9, 0]])


def test_ordinal_encoder_python_integer():
    """Check that `OrdinalEncoder` accepts Python integers that are potentially
    larger than 64 bits.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20721
    """
    # 创建包含超过 64 位整数的 numpy 数组 X
    X = np.array(
        [
            44253463435747313673,
            9867966753463435747313673,
            44253462342215747313673,
            442534634357764313673,
        ]
    ).reshape(-1, 1)

    # 初始化 OrdinalEncoder 对象，确保能正确处理超大整数
    encoder = OrdinalEncoder().fit(X)

    # 使用 assert_array_equal 断言，检查编码器的分类是否与排序后的 X 相同
    assert_array_equal(encoder.categories_, np.sort(X, axis=0).T)

    # 使用 transform 方法对 X 进行转换，检查结果是否正确编码
    X_trans = encoder.transform(X)
    assert_array_equal(X_trans, [[0], [3], [2], [1]])


def test_ordinal_encoder_features_names_out_pandas():
    """Check feature names out is same as the input."""
    # 导入 pandas 库，并确保可以成功导入，否则跳过测试
    pd = pytest.importorskip("pandas")

    # 定义输入数据的特征名称列表
    names = ["b", "c", "a"]
    # 创建一个包含单行数据的 Pandas DataFrame，列名为 names
    X = pd.DataFrame([[1, 2, 3]], columns=names)
    # 创建一个 OrdinalEncoder 对象并使用 X 进行拟合
    enc = OrdinalEncoder().fit(X)
    # 获取经过编码后的特征名称列表
    feature_names_out = enc.get_feature_names_out()
    # 断言原始列名与编码后的特征名称列表相等，用于验证编码的正确性
    assert_array_equal(names, feature_names_out)
# 定义一个测试函数，用于检查编码器在处理未知值和缺失值编码时的交互作用
def test_ordinal_encoder_unknown_missing_interaction():
    # 创建一个包含字符串和NaN的NumPy数组
    X = np.array([["a"], ["b"], [np.nan]], dtype=object)

    # 初始化OrdinalEncoder对象，设置未知值处理策略为"use_encoded_value"，未知值编码为np.nan，缺失值编码为-3，并拟合输入数据
    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=-3,
    ).fit(X)

    # 对输入数据进行转换
    X_trans = oe.transform(X)
    # 断言转换后的结果与预期一致
    assert_allclose(X_trans, [[0], [1], [-3]])

    # 创建测试数据，包含未知值"c"和缺失值np.nan
    X_test = np.array([["c"], [np.nan]], dtype=object)
    # 对测试数据进行转换
    X_test_trans = oe.transform(X_test)
    # 断言转换后的结果与预期一致
    assert_allclose(X_test_trans, [[np.nan], [-3]])

    # 非回归测试，验证逆转换功能
    X_roundtrip = oe.inverse_transform(X_test_trans)

    # 断言np.nan被视为未知值，应映射回None
    assert X_roundtrip[0][0] is None

    # 断言-3被编码为缺失值，应映射回np.nan
    assert np.isnan(X_roundtrip[1][0])


# 使用参数化测试装饰器pytest.mark.parametrize，定义测试函数test_ordinal_encoder_encoded_missing_value_error的参数化测试
@pytest.mark.parametrize("with_pandas", [True, False])
def test_ordinal_encoder_encoded_missing_value_error(with_pandas):
    """Check OrdinalEncoder errors when encoded_missing_value is used by
    an known category."""
    
    # 创建包含字符串和NaN的NumPy数组
    X = np.array([["a", "dog"], ["b", "cat"], ["c", np.nan]], dtype=object)

    # 初始化OrdinalEncoder对象，设置编码缺失值为1
    oe = OrdinalEncoder(encoded_missing_value=1)

    # 如果with_pandas为True，则导入并使用pandas进行测试
    if with_pandas:
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X, columns=["letter", "pet"])
        error_msg = (
            r"encoded_missing_value \(1\) is already used to encode a known category "
            r"in features: \['pet'\]"
        )
    else:
        error_msg = (
            r"encoded_missing_value \(1\) is already used to encode a known category "
            r"in features: \[1\]"
        )

    # 使用pytest.raises断言捕获期望的ValueError异常，并匹配error_msg
    with pytest.raises(ValueError, match=error_msg):
        oe.fit(X)


# 使用参数化测试装饰器pytest.mark.parametrize，定义测试函数test_ordinal_encoder_unknown_missing_interaction_both_nan的参数化测试
@pytest.mark.parametrize(
    "X_train, X_test_trans_expected, X_roundtrip_expected",
    [
        (
            # 在训练集中不存在缺失值，逆转换将编码为未知值
            np.array([["a"], ["1"]], dtype=object),
            [[0], [np.nan], [np.nan]],
            np.asarray([["1"], [None], [None]], dtype=object),
        ),
        (
            # 在训练集中存在缺失值，逆转换将编码为缺失值
            np.array([[np.nan], ["1"], ["a"]], dtype=object),
            [[0], [np.nan], [np.nan]],
            np.asarray([["1"], [np.nan], [np.nan]], dtype=object),
        ),
    ],
)
def test_ordinal_encoder_unknown_missing_interaction_both_nan(
    X_train, X_test_trans_expected, X_roundtrip_expected
):
    """Check transform when unknown_value and encoded_missing_value is nan.

    Non-regression test for #24082.
    """
    # 初始化OrdinalEncoder对象，设置未知值处理策略为"use_encoded_value"，未知值编码和缺失值编码均为np.nan，并拟合训练数据
    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    ).fit(X_train)

    # 创建测试数据，包含字符串"1"和np.nan
    X_test = np.array([["1"], [np.nan], ["b"]])
    # 对测试数据进行转换
    X_test_trans = oe.transform(X_test)
    # 对于测试数据集进行断言，验证转换后的数据与预期数据是否相近
    assert_allclose(X_test_trans, X_test_trans_expected)
    
    # 使用逆变换将转换后的数据还原成原始数据
    X_roundtrip = oe.inverse_transform(X_test_trans)
    
    # 获取预期数据集的样本数
    n_samples = X_roundtrip_expected.shape[0]
    
    # 遍历每个样本进行验证
    for i in range(n_samples):
        # 获取当前样本在预期数据中的值
        expected_val = X_roundtrip_expected[i, 0]
        # 获取经过逆变换后当前样本的值
        val = X_roundtrip[i, 0]
    
        # 根据预期值的类型进行不同的断言
        if expected_val is None:
            # 如果预期值为 None，则验证逆变换后的值也为 None
            assert val is None
        elif is_scalar_nan(expected_val):
            # 如果预期值为 NaN，则验证逆变换后的值也为 NaN
            assert np.isnan(val)
        else:
            # 否则，直接比较逆变换后的值与预期值是否相等
            assert val == expected_val
# 测试函数：检查 OneHotEncoder 在 set_output 方法下的行为
def test_one_hot_encoder_set_output():
    """Check OneHotEncoder works with set_output."""
    # 导入 pytest 并检查其是否可用，否则跳过测试
    pd = pytest.importorskip("pandas")

    # 创建一个包含两列的 DataFrame
    X_df = pd.DataFrame({"A": ["a", "b"], "B": [1, 2]})
    
    # 创建一个 OneHotEncoder 实例
    ohe = OneHotEncoder()

    # 设置输出转换为 pandas 格式
    ohe.set_output(transform="pandas")

    # 预期的错误信息
    match = "Pandas output does not support sparse data. Set sparse_output=False"

    # 使用 pytest 的上下文管理来检查 ValueError 是否被引发，并验证错误信息
    with pytest.raises(ValueError, match=match):
        ohe.fit_transform(X_df)

    # 使用 sparse_output=False 创建两个不同的 OneHotEncoder 实例
    ohe_default = OneHotEncoder(sparse_output=False).set_output(transform="default")
    ohe_pandas = OneHotEncoder(sparse_output=False).set_output(transform="pandas")

    # 分别对 DataFrame 进行 fit_transform
    X_default = ohe_default.fit_transform(X_df)
    X_pandas = ohe_pandas.fit_transform(X_df)

    # 断言两种转换结果的近似性
    assert_allclose(X_pandas.to_numpy(), X_default)
    
    # 断言两种转换结果的列名相等
    assert_array_equal(ohe_pandas.get_feature_names_out(), X_pandas.columns)


# 测试函数：检查 OrdinalEncoder 在 set_output 方法下的行为
def test_ordinal_set_output():
    """Check OrdinalEncoder works with set_output."""
    # 导入 pytest 并检查其是否可用，否则跳过测试
    pd = pytest.importorskip("pandas")

    # 创建一个包含两列的 DataFrame
    X_df = pd.DataFrame({"A": ["a", "b"], "B": [1, 2]})

    # 使用不同输出转换创建两个 OrdinalEncoder 实例
    ord_default = OrdinalEncoder().set_output(transform="default")
    ord_pandas = OrdinalEncoder().set_output(transform="pandas")

    # 分别对 DataFrame 进行 fit_transform
    X_default = ord_default.fit_transform(X_df)
    X_pandas = ord_pandas.fit_transform(X_df)

    # 断言两种转换结果的近似性
    assert_allclose(X_pandas.to_numpy(), X_default)
    
    # 断言两种转换结果的列名相等
    assert_array_equal(ord_pandas.get_feature_names_out(), X_pandas.columns)


# 测试函数：检查 categories_ 的 dtype 是否为 `object`
def test_predefined_categories_dtype():
    """Check that the categories_ dtype is `object` for string categories

    Regression test for gh-25171.
    """
    # 定义字符串类型的 categories
    categories = [["as", "mmas", "eas", "ras", "acs"], ["1", "2"]]

    # 创建一个 OneHotEncoder 实例并使用定义的 categories 进行初始化
    enc = OneHotEncoder(categories=categories)

    # 对一个样本数据进行 fit_transform
    enc.fit([["as", "1"]])

    # 验证 categories_ 的长度与定义的 categories 相等，并且每个分类的 dtype 为 object
    assert len(categories) == len(enc.categories_)
    for n, cat in enumerate(enc.categories_):
        assert cat.dtype == object
        assert_array_equal(categories[n], cat)


# 测试函数：检查 OrdinalEncoder 处理缺失值或未知值编码的最大值行为
def test_ordinal_encoder_missing_unknown_encoding_max():
    """Check missing value or unknown encoding can equal the cardinality."""
    # 创建一个包含字符串的 numpy 数组
    X = np.array([["dog"], ["cat"], [np.nan]], dtype=object)

    # 使用 encoded_missing_value=2 进行转换
    X_trans = OrdinalEncoder(encoded_missing_value=2).fit_transform(X)
    assert_allclose(X_trans, [[1], [0], [2]])

    # 使用 handle_unknown="use_encoded_value", unknown_value=2 进行编码
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=2).fit(X)

    # 对一个未知值进行转换
    X_test = np.array([["snake"]])
    X_trans = enc.transform(X_test)
    assert_allclose(X_trans, [[2]])


# 测试函数：检查 infrequent categories 的 drop_idx 是否正确定义
def test_drop_idx_infrequent_categories():
    """Check drop_idx is defined correctly with infrequent categories.

    Non-regression test for gh-25550.
    """
    # 创建一个包含不频繁出现的类别的 numpy 数组
    X = np.array(
        [["a"] * 2 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["e"] * 4], dtype=object
    ).T
    
    # 创建一个 OneHotEncoder 实例，设置 min_frequency=4, sparse_output=False, drop="first" 并进行 fit
    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop="first").fit(X)

    # 验证输出特征的命名是否正确
    assert_array_equal(
        ohe.get_feature_names_out(), ["x0_c", "x0_d", "x0_e", "x0_infrequent_sklearn"]
    )
    
    # 验证 drop_idx 是否正确指定
    assert ohe.categories_[0][ohe.drop_idx_[0]] == "b"

    # 创建另一个包含不频繁出现的类别的 numpy 数组
    X = np.array([["a"] * 2 + ["b"] * 2 + ["c"] * 10], dtype=object).T
    # 使用 OneHotEncoder 对象进行独热编码，设置最小出现频率为4，稀疏输出为False，且在条件为二进制时删除特征
    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop="if_binary").fit(X)
    # 断言特征名称与预期的相等
    assert_array_equal(ohe.get_feature_names_out(), ["x0_infrequent_sklearn"])
    # 断言删除的特征在对应类别中为 "c"
    assert ohe.categories_[0][ohe.drop_idx_[0]] == "c"

    # 创建包含不同元素的数组 X，转置为列向量
    X = np.array(
        [["a"] * 2 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["e"] * 4], dtype=object
    ).T
    # 使用 OneHotEncoder 对象进行独热编码，设置最小出现频率为4，稀疏输出为False，删除特征 "d"
    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop=["d"]).fit(X)
    # 断言特征名称与预期的相等
    assert_array_equal(
        ohe.get_feature_names_out(), ["x0_b", "x0_c", "x0_e", "x0_infrequent_sklearn"]
    )
    # 断言删除的特征在对应类别中为 "d"
    assert ohe.categories_[0][ohe.drop_idx_[0]] == "d"

    # 使用 OneHotEncoder 对象进行独热编码，设置最小出现频率为4，稀疏输出为False，不删除任何特征
    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop=None).fit(X)
    # 断言特征名称与预期的相等
    assert_array_equal(
        ohe.get_feature_names_out(),
        ["x0_b", "x0_c", "x0_d", "x0_e", "x0_infrequent_sklearn"],
    )
    # 断言没有删除任何特征，因此 drop_idx_ 应为 None
    assert ohe.drop_idx_ is None
# 使用 pytest 的参数化装饰器，定义多组参数化测试用例
@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_categories": 3},  # 设置最大类别数为3
        {"min_frequency": 6},  # 设置最小频率为6
        {"min_frequency": 9},  # 设置最小频率为9
        {"min_frequency": 0.24},  # 设置最小频率为0.24
        {"min_frequency": 0.16},  # 设置最小频率为0.16
        {"max_categories": 3, "min_frequency": 8},  # 同时设置最大类别数为3和最小频率为8
        {"max_categories": 4, "min_frequency": 6},  # 同时设置最大类别数为4和最小频率为6
    ],
)
def test_ordinal_encoder_infrequent_three_levels(kwargs):
    """Test parameters for grouping 'a', and 'd' into the infrequent category."""
    
    # 创建训练数据 X_train，包含多个 'a' 和 'd'，用于测试稀有类别编码
    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    
    # 使用 OrdinalEncoder 进行编码，处理未知值为指定值，同时使用给定的参数
    ordinal = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, **kwargs
    ).fit(X_train)
    
    # 断言编码器的类别顺序和稀有类别的正确性
    assert_array_equal(ordinal.categories_, [["a", "b", "c", "d"]])
    assert_array_equal(ordinal.infrequent_categories_, [["a", "d"]])
    
    # 测试数据 X_test 和预期的转换结果 expected_trans
    X_test = [["a"], ["b"], ["c"], ["d"], ["z"]]
    expected_trans = [[2], [0], [1], [2], [-1]]
    
    # 使用编码器进行转换并断言转换结果的正确性
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)
    
    # 使用逆转换方法进行逆转换并断言逆转换结果的正确性
    X_inverse = ordinal.inverse_transform(X_trans)
    expected_inverse = [
        ["infrequent_sklearn"],
        ["b"],
        ["c"],
        ["infrequent_sklearn"],
        [None],
    ]
    assert_array_equal(X_inverse, expected_inverse)


def test_ordinal_encoder_infrequent_three_levels_user_cats():
    """Test that the order of the categories provided by a user is respected.

    In this case 'c' is encoded as the first category and 'b' is encoded
    as the second one.
    """
    
    # 创建训练数据 X_train，指定用户定义的类别顺序，用于测试类别顺序的尊重
    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object
    ).T
    
    # 使用 OrdinalEncoder 进行编码，指定用户定义的类别顺序和其他参数
    ordinal = OrdinalEncoder(
        categories=[["c", "d", "b", "a"]],
        max_categories=3,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ).fit(X_train)
    
    # 断言编码器的类别顺序和稀有类别的正确性
    assert_array_equal(ordinal.categories_, [["c", "d", "b", "a"]])
    assert_array_equal(ordinal.infrequent_categories_, [["d", "a"]])
    
    # 测试数据 X_test 和预期的转换结果 expected_trans
    X_test = [["a"], ["b"], ["c"], ["d"], ["z"]]
    expected_trans = [[2], [1], [0], [2], [-1]]
    
    # 使用编码器进行转换并断言转换结果的正确性
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)
    
    # 使用逆转换方法进行逆转换并断言逆转换结果的正确性
    X_inverse = ordinal.inverse_transform(X_trans)
    expected_inverse = [
        ["infrequent_sklearn"],
        ["b"],
        ["c"],
        ["infrequent_sklearn"],
        [None],
    ]
    assert_array_equal(X_inverse, expected_inverse)


def test_ordinal_encoder_infrequent_mixed():
    """Test when feature 0 has infrequent categories and feature 1 does not."""
    
    # 创建包含两列的数据 X，用于测试包含稀有类别的情况
    X = np.column_stack(([0, 1, 3, 3, 3, 3, 2, 0, 3], [0, 0, 0, 0, 1, 1, 1, 1, 1]))
    
    # 使用 OrdinalEncoder 进行编码，设置最大类别数为3，并拟合数据 X
    ordinal = OrdinalEncoder(max_categories=3).fit(X)
    
    # 断言第一个特征的稀有类别和第二个特征的稀有类别的正确性
    assert_array_equal(ordinal.infrequent_categories_[0], [1, 2])
    assert ordinal.infrequent_categories_[1] is None
    
    # 测试数据 X_test 和预期的转换结果 expected_trans
    X_test = [[3, 0], [1, 1]]
    expected_trans = [[1, 0], [2, 1]]
    
    # 使用编码器进行转换并断言转换结果的正确性
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)
    
    # 使用逆转换方法进行逆转换并断言逆转换结果的正确性
    X_inverse = ordinal.inverse_transform(X_trans)
    expected_inverse = np.array([[3, 0], ["infrequent_sklearn", 1]], dtype=object)
    assert_array_equal(X_inverse, expected_inverse)
    # 使用 assert_array_equal 函数比较 X_inverse 和 expected_inverse 两个数组是否相等
    assert_array_equal(X_inverse, expected_inverse)
# 定义测试函数，测试对多数据类型的 pandas DataFrame 使用序数编码器的行为
def test_ordinal_encoder_infrequent_multiple_categories_dtypes():
    """Test infrequent categories with a pandas DataFrame with multiple dtypes."""

    # 导入 pytest 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")

    # 创建分类数据类型，包括 "bird", "cat", "dog", "snake"
    categorical_dtype = pd.CategoricalDtype(["bird", "cat", "dog", "snake"])

    # 创建包含多种数据类型的 DataFrame X
    X = pd.DataFrame(
        {
            "str": ["a", "f", "c", "f", "f", "a", "c", "b", "b"],
            "int": [5, 3, 0, 10, 10, 12, 0, 3, 5],
            "categorical": pd.Series(
                ["dog"] * 4 + ["cat"] * 3 + ["snake"] + ["bird"],
                dtype=categorical_dtype,
            ),
        },
        columns=["str", "int", "categorical"],
    )

    # 使用最大类别数为 3 的序数编码器拟合 X
    ordinal = OrdinalEncoder(max_categories=3).fit(X)

    # 对注释的部分进行解释
    # X[:, 0] 'a', 'b', 'c' 有相同的频率。因为它们在排序时出现在前面，'a' 和 'b' 被视为不频繁
    # X[:, 1] 0, 3, 5, 10 的频率为 2，12 的频率为 1。0, 3, 12 在排序时出现在前面，被视为不频繁
    # X[:, 2] "snake" 和 "bird" 被视为不频繁

    # 断言不频繁的类别在 ordinal.infrequent_categories_ 中
    assert_array_equal(ordinal.infrequent_categories_[0], ["a", "b"])
    assert_array_equal(ordinal.infrequent_categories_[1], [0, 3, 12])
    assert_array_equal(ordinal.infrequent_categories_[2], ["bird", "snake"])

    # 创建测试用的 DataFrame X_test
    X_test = pd.DataFrame(
        {
            "str": ["a", "b", "f", "c"],
            "int": [12, 0, 10, 5],
            "categorical": pd.Series(
                ["cat"] + ["snake"] + ["bird"] + ["dog"],
                dtype=categorical_dtype,
            ),
        },
        columns=["str", "int", "categorical"],
    )

    # 预期的转换结果
    expected_trans = [[2, 2, 0], [2, 2, 2], [1, 1, 2], [0, 0, 1]]

    # 使用 ordinal 对象对 X_test 进行转换
    X_trans = ordinal.transform(X_test)

    # 断言转换后的结果与预期结果接近
    assert_allclose(X_trans, expected_trans)


# 测试自定义映射行为的序数编码器
def test_ordinal_encoder_infrequent_custom_mapping():
    """Check behavior of unknown_value and encoded_missing_value with infrequent."""

    # 创建训练数据 X_train，包括大量 "a", "b", "c", "d" 以及一个缺失值
    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3 + [np.nan]], dtype=object
    ).T

    # 使用自定义参数初始化序数编码器
    ordinal = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=2,
        max_categories=2,
        encoded_missing_value=3,
    ).fit(X_train)

    # 断言不频繁的类别在 ordinal.infrequent_categories_ 中
    assert_array_equal(ordinal.infrequent_categories_, [["a", "c", "d"]])

    # 创建测试数据 X_test
    X_test = np.array([["a"], ["b"], ["c"], ["d"], ["e"], [np.nan]], dtype=object)

    # 预期的转换结果
    expected_trans = [[1], [0], [1], [1], [2], [3]]

    # 使用 ordinal 对象对 X_test 进行转换
    X_trans = ordinal.transform(X_test)

    # 断言转换后的结果与预期结果接近
    assert_allclose(X_trans, expected_trans)


# 使用参数化测试检查所有类别均为频繁的情况
@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_categories": 6},  # 最大类别数为 6
        {"min_frequency": 2},   # 最小频率为 2
    ],
)
def test_ordinal_encoder_all_frequent(kwargs):
    """All categories are considered frequent have same encoding as default encoder."""

    # 创建训练数据 X_train，包含大量 "a", "b", "c", "d"
    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object
    ).T

    # 使用参数化的 kwargs 初始化序数编码器
    adjusted_encoder = OrdinalEncoder(
        **kwargs, handle_unknown="use_encoded_value", unknown_value=-1
    ).fit(X_train)
    # 使用 OrdinalEncoder 初始化一个默认的编码器，处理未知值时使用指定的编码值 -1
    default_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ).fit(X_train)

    # 定义测试数据集 X_test，包含五个列表，每个列表包含一个字符串元素
    X_test = [["a"], ["b"], ["c"], ["d"], ["e"]]

    # 使用 adjusted_encoder 对 X_test 进行转换，并使用 default_encoder 对相同数据集进行转换，
    # 然后使用 assert_allclose 进行两者结果的近似比较
    assert_allclose(
        adjusted_encoder.transform(X_test), default_encoder.transform(X_test)
    )
@pytest.mark.parametrize(
    "kwargs",  # 使用 pytest 的参数化装饰器，提供不同的参数组合进行测试
    [
        {"max_categories": 1},  # 第一组参数，设置 max_categories 为 1
        {"min_frequency": 100},  # 第二组参数，设置 min_frequency 为 100
    ],
)
def test_ordinal_encoder_all_infrequent(kwargs):
    """When all categories are infrequent, they are all encoded as zero."""
    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object
    ).T  # 创建一个包含分类数据的二维数组，转置以符合 sklearn 的输入要求
    encoder = OrdinalEncoder(
        **kwargs, handle_unknown="use_encoded_value", unknown_value=-1
    ).fit(X_train)  # 使用给定的参数初始化 OrdinalEncoder，并拟合训练数据 X_train

    X_test = [["a"], ["b"], ["c"], ["d"], ["e"]]  # 待测试的数据集
    assert_allclose(encoder.transform(X_test), [[0], [0], [0], [0], [-1]])  # 断言转换后的结果与预期一致


def test_ordinal_encoder_missing_appears_frequent():
    """Check behavior when missing value appears frequently."""
    X = np.array(
        [[np.nan] * 20 + ["dog"] * 10 + ["cat"] * 5 + ["snake"] + ["deer"]],
        dtype=object,
    ).T  # 创建包含缺失值和其他分类的二维数组，转置以符合 sklearn 的输入要求
    ordinal = OrdinalEncoder(max_categories=3).fit(X)  # 初始化 OrdinalEncoder，并拟合训练数据 X

    X_test = np.array([["snake", "cat", "dog", np.nan]], dtype=object).T  # 待测试的数据集
    X_trans = ordinal.transform(X_test)  # 对 X_test 进行转换
    assert_allclose(X_trans, [[2], [0], [1], [np.nan]])  # 断言转换后的结果与预期一致


def test_ordinal_encoder_missing_appears_infrequent():
    """Check behavior when missing value appears infrequently."""

    # feature 0 has infrequent categories
    # feature 1 has no infrequent categories
    X = np.array(
        [
            [np.nan] + ["dog"] * 10 + ["cat"] * 5 + ["snake"] + ["deer"],
            ["red"] * 9 + ["green"] * 9,
        ],
        dtype=object,
    ).T  # 创建包含缺失值和其他分类的二维数组，转置以符合 sklearn 的输入要求
    ordinal = OrdinalEncoder(min_frequency=4).fit(X)  # 初始化 OrdinalEncoder，并拟合训练数据 X

    X_test = np.array(
        [
            ["snake", "red"],
            ["deer", "green"],
            [np.nan, "green"],
            ["dog", "green"],
            ["cat", "red"],
        ],
        dtype=object,
    )  # 待测试的数据集
    X_trans = ordinal.transform(X_test)  # 对 X_test 进行转换
    assert_allclose(X_trans, [[2, 1], [2, 0], [np.nan, 0], [1, 0], [0, 1]])  # 断言转换后的结果与预期一致


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
def test_encoder_not_fitted(Encoder):
    """Check that we raise a `NotFittedError` by calling transform before fit with
    the encoders.

    One could expect that the passing the `categories` argument to the encoder
    would make it stateless. However, `fit` is making a couple of check, such as the
    position of `np.nan`.
    """
    X = np.array([["A"], ["B"], ["C"]], dtype=object)  # 创建一个包含分类数据的二维数组
    encoder = Encoder(categories=[["A", "B", "C"]])  # 使用给定的 Encoder 类型和参数初始化 encoder
    with pytest.raises(NotFittedError):  # 断言调用 transform 方法时会抛出 NotFittedError 异常
        encoder.transform(X)
```