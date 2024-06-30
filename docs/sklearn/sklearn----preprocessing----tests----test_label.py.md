# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\tests\test_label.py`

```
# 导入必要的库
import numpy as np  # 导入NumPy库，并简写为np
import pytest  # 导入pytest测试框架
from scipy.sparse import issparse  # 从scipy.sparse模块导入issparse函数

# 导入sklearn库中的模块和函数
from sklearn import config_context, datasets  # 从sklearn中导入config_context和datasets模块
from sklearn.preprocessing._label import (  # 导入标签处理相关的类和函数
    LabelBinarizer,  # 标签二值化器
    LabelEncoder,  # 标签编码器
    MultiLabelBinarizer,  # 多标签二值化器
    _inverse_binarize_multiclass,  # 多类反二值化
    _inverse_binarize_thresholding,  # 阈值反二值化
    label_binarize,  # 标签二值化函数
)
from sklearn.utils._array_api import (  # 导入数组API相关的函数
    _convert_to_numpy,  # 转换为NumPy数组
    get_namespace,  # 获取命名空间
    yield_namespace_device_dtype_combinations,  # 生成命名空间、设备和数据类型的组合
)
from sklearn.utils._testing import (  # 导入测试相关的函数
    _array_api_for_tests,  # 用于测试的数组API
    assert_array_equal,  # 断言数组相等
    ignore_warnings,  # 忽略警告
)
from sklearn.utils.fixes import (  # 导入修复相关的模块
    COO_CONTAINERS,  # COO格式容器
    CSC_CONTAINERS,  # CSC格式容器
    CSR_CONTAINERS,  # CSR格式容器
    DOK_CONTAINERS,  # DOK格式容器
    LIL_CONTAINERS,  # LIL格式容器
)
from sklearn.utils.multiclass import type_of_target  # 导入type_of_target函数，用于判断目标类型
from sklearn.utils.validation import _to_object_array  # 导入_to_object_array函数，将输入转换为对象数组

# 加载鸢尾花数据集
iris = datasets.load_iris()

# 定义一个函数，将稀疏矩阵或数组转换为普通的NumPy数组
def toarray(a):
    if hasattr(a, "toarray"):  # 如果对象a有toarray方法
        a = a.toarray()  # 将a转换为NumPy数组
    return a  # 返回转换后的结果

# 测试标签二值化器的功能
def test_label_binarizer():
    # 单类别情况，默认为负标签
    # 对于密集矩阵：
    inp = ["pos", "pos", "pos", "pos"]
    lb = LabelBinarizer(sparse_output=False)  # 创建标签二值化器对象，稠密输出
    expected = np.array([[0, 0, 0, 0]]).T  # 期望的输出
    got = lb.fit_transform(inp)  # 进行标签二值化
    assert_array_equal(lb.classes_, ["pos"])  # 断言类别数组是否符合预期
    assert_array_equal(expected, got)  # 断言输出是否符合预期
    assert_array_equal(lb.inverse_transform(got), inp)  # 断言逆转换后是否能够还原原始输入

    # 对于稀疏矩阵情况：
    lb = LabelBinarizer(sparse_output=True)  # 创建标签二值化器对象，稀疏输出
    got = lb.fit_transform(inp)  # 进行标签二值化
    assert issparse(got)  # 断言输出是否为稀疏矩阵
    assert_array_equal(lb.classes_, ["pos"])  # 断言类别数组是否符合预期
    assert_array_equal(expected, got.toarray())  # 断言输出是否符合预期（转换为稠密数组后）

    lb = LabelBinarizer(sparse_output=False)  # 再次创建标签二值化器对象，稠密输出
    # 二类别情况
    inp = ["neg", "pos", "pos", "neg"]
    expected = np.array([[0, 1, 1, 0]]).T  # 期望的输出
    got = lb.fit_transform(inp)  # 进行标签二值化
    assert_array_equal(lb.classes_, ["neg", "pos"])  # 断言类别数组是否符合预期
    assert_array_equal(expected, got)  # 断言输出是否符合预期

    to_invert = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # 待逆转的数组
    assert_array_equal(lb.inverse_transform(to_invert), inp)  # 断言逆转换后是否能够还原原始输入

    # 多类别情况
    inp = ["spam", "ham", "eggs", "ham", "0"]
    expected = np.array(  # 期望的输出
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    )
    got = lb.fit_transform(inp)  # 进行标签二值化
    assert_array_equal(lb.classes_, ["0", "eggs", "ham", "spam"])  # 断言类别数组是否符合预期
    assert_array_equal(expected, got)  # 断言输出是否符合预期
    assert_array_equal(lb.inverse_transform(got), inp)  # 断言逆转换后是否能够还原原始输入

# 测试标签二值化器处理未见过标签的情况
def test_label_binarizer_unseen_labels():
    lb = LabelBinarizer()

    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 期望的输出
    got = lb.fit_transform(["b", "d", "e"])  # 进行标签二值化
    assert_array_equal(expected, got)  # 断言输出是否符合预期

    expected = np.array(  # 期望的输出
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    )
    got = lb.transform(["a", "b", "c", "d", "e", "f"])  # 转换未见过的标签
    assert_array_equal(expected, got)  # 断言输出是否符合预期

# 测试设置标签编码的情况
def test_label_binarizer_set_label_encoding():
    lb = LabelBinarizer(neg_label=-2, pos_label=0)  # 创建标签二值化器对象，设置负标签为-2，正标签为0

    # 二类别情况，正标签为0
    inp = np.array([0, 1, 1, 0])  # 输入数组
    expected = np.array([[-2, 0, 0, -2]]).T  # 期望的输出
    # 使用 LabelBinarizer 对输入数据进行转换为二进制编码，返回编码结果
    got = lb.fit_transform(inp)
    # 断言编码后的结果与期望结果相等
    assert_array_equal(expected, got)
    # 断言反向转换后的结果与原始输入相等，验证逆变换的准确性
    assert_array_equal(lb.inverse_transform(got), inp)

    # 创建 LabelBinarizer 对象，指定负类标签为-2，正类标签为+2
    lb = LabelBinarizer(neg_label=-2, pos_label=2)

    # 多类别情况
    inp = np.array([3, 2, 1, 2, 0])
    expected = np.array(
        [
            [-2, -2, -2, +2],
            [-2, -2, +2, -2],
            [-2, +2, -2, -2],
            [-2, -2, +2, -2],
            [+2, -2, -2, -2],
        ]
    )
    # 对多类别输入进行二进制编码转换
    got = lb.fit_transform(inp)
    # 断言编码后的结果与期望结果相等
    assert_array_equal(expected, got)
    # 断言反向转换后的结果与原始输入相等，验证逆变换的准确性
    assert_array_equal(lb.inverse_transform(got), inp)
# 使用 pytest 的参数化装饰器，为 test_label_binarizer_pandas_nullable 函数指定参数组合
@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
@pytest.mark.parametrize("unique_first", [True, False])
def test_label_binarizer_pandas_nullable(dtype, unique_first):
    """Checks that LabelBinarizer works with pandas nullable dtypes.

    Non-regression test for gh-25637.
    """
    # 导入 pytest，如果导入失败则跳过该测试
    pd = pytest.importorskip("pandas")

    # 创建 pandas Series 对象 y_true，使用给定的数据类型 dtype
    y_true = pd.Series([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=dtype)
    if unique_first:
        # 如果 unique_first 为 True，则调用 unique 方法创建一个新的 pandas 数组
        # pandas 数组与 Series 有不同的接口，不支持 "iloc"
        y_true = y_true.unique()
    
    # 创建并拟合 LabelBinarizer 对象 lb，使用 y_true 数据
    lb = LabelBinarizer().fit(y_true)
    
    # 使用拟合好的 lb 对象对 [1, 0] 进行转换，得到二进制编码结果 y_out
    y_out = lb.transform([1, 0])

    # 断言 y_out 结果与预期结果 [[1], [0]] 相等
    assert_array_equal(y_out, [[1], [0]])


# 使用 ignore_warnings 装饰器，忽略测试中的警告信息
@ignore_warnings
def test_label_binarizer_errors():
    # 检查无效参数是否会引发 ValueError
    one_class = np.array([0, 0, 0, 0])
    lb = LabelBinarizer().fit(one_class)

    multi_label = [(2, 3), (0,), (0, 2)]
    err_msg = "You appear to be using a legacy multi-label data representation."
    # 使用 pytest.raises 检查是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=err_msg):
        lb.transform(multi_label)

    lb = LabelBinarizer()
    err_msg = "This LabelBinarizer instance is not fitted yet"
    with pytest.raises(ValueError, match=err_msg):
        lb.transform([])
    with pytest.raises(ValueError, match=err_msg):
        lb.inverse_transform([])

    input_labels = [0, 1, 0, 1]
    err_msg = "neg_label=2 must be strictly less than pos_label=1."
    # 创建具有不同参数的 LabelBinarizer 对象，检查是否会引发 ValueError，并匹配特定错误消息
    lb = LabelBinarizer(neg_label=2, pos_label=1)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)
    err_msg = "neg_label=2 must be strictly less than pos_label=2."
    lb = LabelBinarizer(neg_label=2, pos_label=2)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)
    err_msg = (
        "Sparse binarization is only supported with non zero pos_label and zero "
        "neg_label, got pos_label=2 and neg_label=1"
    )
    lb = LabelBinarizer(neg_label=1, pos_label=2, sparse_output=True)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)

    # 创建一个嵌套列表 y_seq_of_seqs，检查是否会引发 ValueError，并匹配特定错误消息
    y_seq_of_seqs = [[], [1, 2], [3], [0, 1, 3], [2]]
    err_msg = "You appear to be using a legacy multi-label data representation"
    with pytest.raises(ValueError, match=err_msg):
        LabelBinarizer().fit_transform(y_seq_of_seqs)

    # 创建一个 np.array 对象，检查是否会引发 ValueError，并匹配特定错误消息
    err_msg = "output_type='binary', but y.shape"
    with pytest.raises(ValueError, match=err_msg):
        _inverse_binarize_thresholding(
            y=np.array([[1, 2, 3], [2, 1, 3]]),
            output_type="binary",
            classes=[1, 2, 3],
            threshold=0,
        )

    # 创建一个 np.array 对象，检查是否会引发 ValueError，并匹配特定错误消息
    err_msg = "Multioutput target data is not supported with label binarization"
    with pytest.raises(ValueError, match=err_msg):
        LabelBinarizer().fit(np.array([[1, 3], [2, 1]]))
    # 使用 pytest 的上下文管理器 `raises` 来验证函数是否引发指定类型的异常，并检查异常消息是否匹配给定的错误消息
    with pytest.raises(ValueError, match=err_msg):
        # 调用 label_binarize 函数，尝试对输入数组进行标签二值化处理
        label_binarize(np.array([[1, 3], [2, 1]]), classes=[1, 2, 3])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用 pytest.mark.parametrize 装饰器，为 test_label_binarizer_sparse_errors 函数参数 csr_container 注入多个不同的容器类型
def test_label_binarizer_sparse_errors(csr_container):
    # Fail on y_type
    # 设置错误消息字符串，用于匹配异常信息
    err_msg = "foo format is not supported"
    # 使用 pytest.raises 检查是否会抛出 ValueError 异常，并匹配特定错误消息
    with pytest.raises(ValueError, match=err_msg):
        # 调用 _inverse_binarize_thresholding 函数，传入参数，期望抛出异常
        _inverse_binarize_thresholding(
            y=csr_container([[1, 2], [2, 1]]),
            output_type="foo",
            classes=[1, 2],
            threshold=0,
        )

    # Fail on the number of classes
    # 设置错误消息字符串，用于匹配异常信息
    err_msg = "The number of class is not equal to the number of dimension of y."
    # 使用 pytest.raises 检查是否会抛出 ValueError 异常，并匹配特定错误消息
    with pytest.raises(ValueError, match=err_msg):
        # 调用 _inverse_binarize_thresholding 函数，传入参数，期望抛出异常
        _inverse_binarize_thresholding(
            y=csr_container([[1, 2], [2, 1]]),
            output_type="foo",
            classes=[1, 2, 3],
            threshold=0,
        )


@pytest.mark.parametrize(
    "values, classes, unknown",
    [
        (
            np.array([2, 1, 3, 1, 3], dtype="int64"),
            np.array([1, 2, 3], dtype="int64"),
            np.array([4], dtype="int64"),
        ),
        (
            np.array(["b", "a", "c", "a", "c"], dtype=object),
            np.array(["a", "b", "c"], dtype=object),
            np.array(["d"], dtype=object),
        ),
        (
            np.array(["b", "a", "c", "a", "c"]),
            np.array(["a", "b", "c"]),
            np.array(["d"]),
        ),
    ],
    ids=["int64", "object", "str"],
)
# 使用 pytest.mark.parametrize 装饰器，为 test_label_encoder 函数参数 values, classes, unknown 注入多组不同的数据类型和值
def test_label_encoder(values, classes, unknown):
    # Test LabelEncoder's transform, fit_transform and
    # inverse_transform methods
    # 创建 LabelEncoder 对象
    le = LabelEncoder()
    # 对 values 进行拟合
    le.fit(values)
    # 检查拟合后的类别数组是否与预期的 classes 数组相等
    assert_array_equal(le.classes_, classes)
    # 检查 transform 方法是否正确转换 values
    assert_array_equal(le.transform(values), [1, 0, 2, 0, 2])
    # 检查 inverse_transform 方法是否正确逆转换
    assert_array_equal(le.inverse_transform([1, 0, 2, 0, 2]), values)
    # 创建新的 LabelEncoder 对象
    le = LabelEncoder()
    # 使用 fit_transform 方法对 values 进行拟合和转换
    ret = le.fit_transform(values)
    # 检查返回的数组是否与预期的转换结果一致
    assert_array_equal(ret, [1, 0, 2, 0, 2])

    # 检查是否会抛出 ValueError 异常，匹配 "unseen labels" 错误消息
    with pytest.raises(ValueError, match="unseen labels"):
        # 使用 transform 方法，传入 unknown 进行转换
        le.transform(unknown)


def test_label_encoder_negative_ints():
    # 创建 LabelEncoder 对象
    le = LabelEncoder()
    # 对包含负整数的列表进行拟合
    le.fit([1, 1, 4, 5, -1, 0])
    # 检查拟合后的类别数组是否与预期的结果一致
    assert_array_equal(le.classes_, [-1, 0, 1, 4, 5])
    # 检查 transform 方法是否正确转换给定的列表
    assert_array_equal(le.transform([0, 1, 4, 4, 5, -1, -1]), [1, 2, 3, 3, 4, 0, 0])
    # 检查 inverse_transform 方法是否正确逆转换
    assert_array_equal(
        le.inverse_transform([1, 2, 3, 3, 4, 0, 0]), [0, 1, 4, 4, 5, -1, -1]
    )
    # 检查是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 使用 transform 方法，传入一个超出类别范围的列表
        le.transform([0, 6])


@pytest.mark.parametrize("dtype", ["str", "object"])
# 使用 pytest.mark.parametrize 装饰器，为 test_label_encoder_str_bad_shape 函数参数 dtype 注入多个数据类型
def test_label_encoder_str_bad_shape(dtype):
    # 创建 LabelEncoder 对象
    le = LabelEncoder()
    # 对包含字符串的数组进行拟合
    le.fit(np.array(["apple", "orange"], dtype=dtype))
    # 设置错误消息字符串，用于匹配异常信息
    msg = "should be a 1d array"
    # 检查是否会抛出 ValueError 异常，匹配特定错误消息
    with pytest.raises(ValueError, match=msg):
        # 使用 transform 方法，传入一个字符串进行转换
        le.transform("apple")


def test_label_encoder_errors():
    # Check that invalid arguments yield ValueError
    # 创建 LabelEncoder 对象
    le = LabelEncoder()
    # 检查是否会抛出 ValueError 异常，因为空列表不能作为参数
    with pytest.raises(ValueError):
        le.transform([])
    # 检查是否会抛出 ValueError 异常，因为空列表不能作为参数
    with pytest.raises(ValueError):
        le.inverse_transform([])

    # 创建 LabelEncoder 对象
    le = LabelEncoder()
    # 对包含未见过标签的列表进行拟合
    le.fit([1, 2, 3, -1, 1])
    # 设置错误消息字符串，用于匹配异常信息
    msg = "contains previously unseen labels"
    # 使用 pytest 框架检测是否会引发 ValueError 异常，并验证异常消息是否匹配给定的 msg
    with pytest.raises(ValueError, match=msg):
        # 调用 le 对象的 inverse_transform 方法，尝试对输入 [-2] 进行逆转换
        le.inverse_transform([-2])

    # 使用 pytest 框架检测是否会引发 ValueError 异常，并验证异常消息是否匹配给定的 msg
    with pytest.raises(ValueError, match=msg):
        # 调用 le 对象的 inverse_transform 方法，尝试对输入 [-2, -3, -4] 进行逆转换
        le.inverse_transform([-2, -3, -4])

    # 设置变量 msg 以匹配预期的异常消息格式
    msg = r"should be a 1d array.+shape \(\)"
    # 使用 pytest 框架检测是否会引发 ValueError 异常，并验证异常消息是否匹配给定的 msg
    with pytest.raises(ValueError, match=msg):
        # 调用 le 对象的 inverse_transform 方法，尝试对输入 "" 进行逆转换
        le.inverse_transform("")
@pytest.mark.parametrize(
    "values",
    [
        np.array([2, 1, 3, 1, 3], dtype="int64"),  # 创建一个包含整数数组的参数化测试值
        np.array(["b", "a", "c", "a", "c"], dtype=object),  # 创建一个包含对象数组的参数化测试值
        np.array(["b", "a", "c", "a", "c"]),  # 创建一个包含字符串数组的参数化测试值，默认为对象类型
    ],
    ids=["int64", "object", "str"],  # 每个参数化测试值的标识
)
def test_label_encoder_empty_array(values):
    le = LabelEncoder()  # 创建一个标签编码器对象
    le.fit(values)  # 使用值来拟合标签编码器
    # 测试空转换
    transformed = le.transform([])  # 对空数组进行转换
    assert_array_equal(np.array([]), transformed)  # 断言转换结果为空数组
    # 测试空的逆转换
    inverse_transformed = le.inverse_transform([])  # 对空数组进行逆转换
    assert_array_equal(np.array([]), inverse_transformed)  # 断言逆转换结果为空数组


def test_sparse_output_multilabel_binarizer():
    # 测试输入为可迭代对象的可多标签二值化器
    inputs = [
        lambda: [(2, 3), (1,), (1, 2)],  # 元组列表作为输入
        lambda: ({2, 3}, {1}, {1, 2}),  # 集合作为输入
        lambda: iter([iter((2, 3)), iter((1,)), {1, 2}]),  # 迭代器的迭代器作为输入
    ]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])  # 预期的指示矩阵

    inverse = inputs[0]()  # 获取输入的逆转换结果
    for sparse_output in [True, False]:  # 遍历稀疏输出为真和假的情况
        for inp in inputs:
            # 使用 fit_transform 方法
            mlb = MultiLabelBinarizer(sparse_output=sparse_output)  # 创建一个多标签二值化器对象
            got = mlb.fit_transform(inp())  # 对输入进行拟合和转换
            assert issparse(got) == sparse_output  # 断言得到的结果是否为稀疏矩阵
            if sparse_output:
                # 验证 CSR 的假设：索引和索引指针具有相同的数据类型
                assert got.indices.dtype == got.indptr.dtype
                got = got.toarray()  # 转换为稠密数组
            assert_array_equal(indicator_mat, got)  # 断言转换后的结果与预期的指示矩阵相等
            assert_array_equal([1, 2, 3], mlb.classes_)  # 断言标签编码器的类与预期的类相等
            assert mlb.inverse_transform(got) == inverse  # 断言逆转换结果与预期的逆转换结果相等

            # 使用 fit 方法
            mlb = MultiLabelBinarizer(sparse_output=sparse_output)  # 创建一个多标签二值化器对象
            got = mlb.fit(inp()).transform(inp())  # 对输入进行拟合和转换
            assert issparse(got) == sparse_output  # 断言得到的结果是否为稀疏矩阵
            if sparse_output:
                # 验证 CSR 的假设：索引和索引指针具有相同的数据类型
                assert got.indices.dtype == got.indptr.dtype
                got = got.toarray()  # 转换为稠密数组
            assert_array_equal(indicator_mat, got)  # 断言转换后的结果与预期的指示矩阵相等
            assert_array_equal([1, 2, 3], mlb.classes_)  # 断言标签编码器的类与预期的类相等
            assert mlb.inverse_transform(got) == inverse  # 断言逆转换结果与预期的逆转换结果相等


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)  # 参数化测试，测试不同的 CSR 容器
def test_sparse_output_multilabel_binarizer_errors(csr_container):
    inp = iter([iter((2, 3)), iter((1,)), {1, 2}])  # 创建输入迭代器
    mlb = MultiLabelBinarizer(sparse_output=False)  # 创建一个不使用稀疏输出的多标签二值化器对象
    mlb.fit(inp)  # 对输入进行拟合
    with pytest.raises(ValueError):
        mlb.inverse_transform(
            csr_container(np.array([[0, 1, 1], [2, 0, 0], [1, 1, 0]]))  # 尝试使用不同的 CSR 容器进行逆转换
        )


def test_multilabel_binarizer():
    # 测试输入为可迭代对象的多标签二值化器
    inputs = [
        lambda: [(2, 3), (1,), (1, 2)],  # 元组列表作为输入
        lambda: ({2, 3}, {1}, {1, 2}),  # 集合作为输入
        lambda: iter([iter((2, 3)), iter((1,)), {1, 2}]),  # 迭代器的迭代器作为输入
    ]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])  # 预期的指示矩阵
    inverse = inputs[0]()  # 获取输入的逆转换结果
    # 对于每一个输入列表中的元素进行处理
    for inp in inputs:
        # 使用 fit_transform 方法对输入数据进行拟合和转换，并生成二进制标签矩阵
        mlb = MultiLabelBinarizer()
        got = mlb.fit_transform(inp())
        # 断言生成的二进制标签矩阵与预期的指示器矩阵相等
        assert_array_equal(indicator_mat, got)
        # 断言生成的类别列表与预期的类别列表相等
        assert_array_equal([1, 2, 3], mlb.classes_)
        # 使用 inverse_transform 方法将二进制标签矩阵转换回原始标签，并断言与预期的反转结果相等
        assert mlb.inverse_transform(got) == inverse

        # 使用 fit 方法对输入数据进行拟合，然后使用 transform 方法进行转换，并生成二进制标签矩阵
        mlb = MultiLabelBinarizer()
        got = mlb.fit(inp()).transform(inp())
        # 断言生成的二进制标签矩阵与预期的指示器矩阵相等
        assert_array_equal(indicator_mat, got)
        # 断言生成的类别列表与预期的类别列表相等
        assert_array_equal([1, 2, 3], mlb.classes_)
        # 使用 inverse_transform 方法将二进制标签矩阵转换回原始标签，并断言与预期的反转结果相等
        assert mlb.inverse_transform(got) == inverse
def test_multilabel_binarizer_empty_sample():
    # 创建一个 MultiLabelBinarizer 对象
    mlb = MultiLabelBinarizer()
    # 定义样本 y 和期望的转换结果 Y
    y = [[1, 2], [1], []]
    Y = np.array([[1, 1], [1, 0], [0, 0]])
    # 断言 mlb.fit_transform(y) 的结果与预期的 Y 相等
    assert_array_equal(mlb.fit_transform(y), Y)


def test_multilabel_binarizer_unknown_class():
    # 创建一个 MultiLabelBinarizer 对象
    mlb = MultiLabelBinarizer()
    # 定义样本 y 和期望的转换结果 Y
    y = [[1, 2]]
    Y = np.array([[1, 0], [0, 1]])
    # 设置警告消息内容
    warning_message = "unknown class.* will be ignored"
    # 使用 pytest 的 warns 断言捕获 UserWarning 异常，并匹配 warning_message
    with pytest.warns(UserWarning, match=warning_message):
        # 执行 mlb.fit(y).transform([[4, 1], [2, 0]]) 操作
        matrix = mlb.fit(y).transform([[4, 1], [2, 0]])

    # 重新定义 Y，预期值改为 [[1, 0, 0], [0, 1, 0]]
    Y = np.array([[1, 0, 0], [0, 1, 0]])
    # 创建一个新的 MultiLabelBinarizer 对象，指定类别为 [1, 2, 3]
    mlb = MultiLabelBinarizer(classes=[1, 2, 3])
    # 使用 pytest 的 warns 断言捕获 UserWarning 异常，并匹配 warning_message
    with pytest.warns(UserWarning, match=warning_message):
        # 执行 mlb.fit(y).transform([[4, 1], [2, 0]]) 操作
        matrix = mlb.fit(y).transform([[4, 1], [2, 0]])
    # 断言 matrix 的结果与预期的 Y 相等
    assert_array_equal(matrix, Y)


def test_multilabel_binarizer_given_classes():
    # 定义输入 inp 和期望的指示矩阵 indicator_mat
    inp = [(2, 3), (1,), (1, 2)]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
    
    # 使用指定的类别 [1, 3, 2] 创建 MultiLabelBinarizer 对象 mlb
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    # 断言 mlb.fit_transform(inp) 的结果与 indicator_mat 相等
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)
    # 断言 mlb.classes_ 的结果与 [1, 3, 2] 相等
    assert_array_equal(mlb.classes_, [1, 3, 2])

    # 使用 fit().transform() 方法进行操作
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    # 断言 mlb.fit(inp).transform(inp) 的结果与 indicator_mat 相等
    assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
    # 断言 mlb.classes_ 的结果与 [1, 3, 2] 相等
    assert_array_equal(mlb.classes_, [1, 3, 2])

    # 使用额外的类别 [4, 1, 3, 2] 创建 MultiLabelBinarizer 对象 mlb
    mlb = MultiLabelBinarizer(classes=[4, 1, 3, 2])
    # 断言 mlb.fit_transform(inp) 的结果与预期的结果连接在一起
    assert_array_equal(
        mlb.fit_transform(inp), np.hstack(([[0], [0], [0]], indicator_mat))
    )
    # 断言 mlb.classes_ 的结果与 [4, 1, 3, 2] 相等
    assert_array_equal(mlb.classes_, [4, 1, 3, 2])

    # 确保 fit 操作不会消耗可迭代对象
    inp = iter(inp)
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    # 断言 mlb.fit(inp).transform(inp) 的结果与 indicator_mat 相等
    assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)

    # 确保如果给出重复的类别，会抛出 ValueError 异常
    err_msg = (
        "The classes argument contains duplicate classes. Remove "
        "these duplicates before passing them to MultiLabelBinarizer."
    )
    mlb = MultiLabelBinarizer(classes=[1, 3, 2, 3])
    # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        mlb.fit(inp)


def test_multilabel_binarizer_multiple_calls():
    # 定义输入 inp 和两个预期的指示矩阵 indicator_mat 和 indicator_mat2
    inp = [(2, 3), (1,), (1, 2)]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
    indicator_mat2 = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])

    # 第一次调用，使用类别 [1, 3, 2] 创建 MultiLabelBinarizer 对象 mlb
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    # 断言 mlb.fit_transform(inp) 的结果与 indicator_mat 相等
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)
    # 修改 mlb 对象的类别顺序为 [1, 2, 3]
    mlb.classes = [1, 2, 3]
    # 断言 mlb.fit_transform(inp) 的结果与 indicator_mat2 相等
    assert_array_equal(mlb.fit_transform(inp), indicator_mat2)


def test_multilabel_binarizer_same_length_sequence():
    # 确保相同长度的序列不会被解释为二维数组
    inp = [[1], [0], [2]]
    indicator_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    
    # 使用默认参数创建 MultiLabelBinarizer 对象 mlb
    mlb = MultiLabelBinarizer()
    # 断言 mlb.fit_transform(inp) 的结果与 indicator_mat 相等
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)
    # 断言 mlb.inverse_transform(indicator_mat) 的结果与 inp 相等
    assert_array_equal(mlb.inverse_transform(indicator_mat), inp)

    # 继续测试 fit().transform() 方法
    mlb = MultiLabelBinarizer()
    # 断言输入数据经过多标签二进制化后的结果与指定的指示器矩阵相等
    assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
    
    # 断言通过逆转换指示器矩阵能够得到原始输入数据
    assert_array_equal(mlb.inverse_transform(indicator_mat), inp)
# 测试多标签二值化器处理非整数标签的函数
def test_multilabel_binarizer_non_integer_labels():
    # 准备元组形式的类别数据
    tuple_classes = _to_object_array([(1,), (2,), (3,)])
    # 输入数据包含三组元组，每组包含多个字符串或元组，以及对应的类别列表
    inputs = [
        ([("2", "3"), ("1",), ("1", "2")], ["1", "2", "3"]),
        ([("b", "c"), ("a",), ("a", "b")], ["a", "b", "c"]),
        ([((2,), (3,)), ((1,),), ((1,), (2,))], tuple_classes),
    ]
    # 预定义的指示矩阵
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])

    # 遍历所有输入数据和对应的类别
    for inp, classes in inputs:
        # 使用 MultiLabelBinarizer 初始化对象 mlb
        mlb = MultiLabelBinarizer()
        # 将输入数据转换为 numpy 数组类型为对象
        inp = np.array(inp, dtype=object)
        # 断言使用 fit_transform() 方法得到的二值化结果等于预期的指示矩阵
        assert_array_equal(mlb.fit_transform(inp), indicator_mat)
        # 断言 mlb 对象的 classes_ 属性等于预期的类别列表
        assert_array_equal(mlb.classes_, classes)
        # 使用 mlb 对象的 inverse_transform() 方法将指示矩阵反向转换回原始数据，断言结果与原输入数据相等
        indicator_mat_inv = np.array(mlb.inverse_transform(indicator_mat), dtype=object)
        assert_array_equal(indicator_mat_inv, inp)

        # 使用 fit().transform() 方法进行相同的测试流程
        mlb = MultiLabelBinarizer()
        assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
        assert_array_equal(mlb.classes_, classes)
        indicator_mat_inv = np.array(mlb.inverse_transform(indicator_mat), dtype=object)
        assert_array_equal(indicator_mat_inv, inp)

    # 对于异常情况，测试 MultiLabelBinarizer 处理非整数标签时是否能够引发 TypeError 异常
    mlb = MultiLabelBinarizer()
    with pytest.raises(TypeError):
        mlb.fit_transform([({}), ({}, {"a": "b"})])


# 测试多标签二值化器处理非唯一标签的函数
def test_multilabel_binarizer_non_unique():
    # 输入数据包含一个非唯一标签的元组列表
    inp = [(1, 1, 1, 0)]
    # 预定义的指示矩阵
    indicator_mat = np.array([[1, 1]])
    # 使用 MultiLabelBinarizer 初始化对象 mlb
    mlb = MultiLabelBinarizer()
    # 断言使用 fit_transform() 方法得到的二值化结果等于预期的指示矩阵
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)


# 测试多标签二值化器反向转换方法的验证函数
def test_multilabel_binarizer_inverse_validation():
    # 输入数据包含一个非唯一标签的元组列表
    inp = [(1, 1, 1, 0)]
    # 使用 MultiLabelBinarizer 初始化对象 mlb
    mlb = MultiLabelBinarizer()
    # 使用 fit_transform() 方法对输入数据进行二值化处理
    mlb.fit_transform(inp)
    # 验证是否能够正确处理不是二进制的情况，预期引发 ValueError 异常
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1, 3]]))
    # 验证以下二进制情况是否能够正常处理
    mlb.inverse_transform(np.array([[0, 0]]))
    mlb.inverse_transform(np.array([[1, 1]]))
    mlb.inverse_transform(np.array([[1, 0]]))

    # 验证输入形状不正确时是否能够引发 ValueError 异常
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1]]))
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1, 1, 1]]))


# 测试带有类别顺序的 label_binarize 函数
def test_label_binarize_with_class_order():
    # 使用指定的类别顺序对 [1, 6] 进行二值化处理
    out = label_binarize([1, 6], classes=[1, 2, 4, 6])
    # 预期的二值化结果
    expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    # 断言二值化结果与预期结果相等
    assert_array_equal(out, expected)

    # 修改类别顺序后再次进行二值化处理
    out = label_binarize([1, 6], classes=[1, 6, 4, 2])
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert_array_equal(out, expected)

    # 使用指定的类别顺序对 [0, 1, 2, 3] 进行二值化处理
    out = label_binarize([0, 1, 2, 3], classes=[3, 2, 0, 1])
    expected = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]])
    assert_array_equal(out, expected)
    # 对于稀疏输出为True和标签0不是正例或者负例不是0的情况，期望引发ValueError异常
    for sparse_output in [True, False]:
        if (pos_label == 0 or neg_label != 0) and sparse_output:
            with pytest.raises(ValueError):
                label_binarize(
                    y,
                    classes=classes,
                    neg_label=neg_label,
                    pos_label=pos_label,
                    sparse_output=sparse_output,
                )
            continue

        # 检查label_binarize函数的输出
        binarized = label_binarize(
            y,
            classes=classes,
            neg_label=neg_label,
            pos_label=pos_label,
            sparse_output=sparse_output,
        )
        # 断言转换后的数组与期望结果一致
        assert_array_equal(toarray(binarized), expected)
        # 断言稀疏性与预期一致
        assert issparse(binarized) == sparse_output

        # 检查反转操作
        y_type = type_of_target(y)
        if y_type == "multiclass":
            # 对多类问题进行反转二值化
            inversed = _inverse_binarize_multiclass(binarized, classes=classes)
        else:
            # 对于其他类型的问题，通过阈值反转二值化
            inversed = _inverse_binarize_thresholding(
                binarized,
                output_type=y_type,
                classes=classes,
                threshold=((neg_label + pos_label) / 2.0),
            )

        # 断言反转后的数组与原始数组一致
        assert_array_equal(toarray(inversed), toarray(y))

        # 检查LabelBinarizer类
        lb = LabelBinarizer(
            neg_label=neg_label, pos_label=pos_label, sparse_output=sparse_output
        )
        # 对数组进行拟合转换
        binarized = lb.fit_transform(y)
        # 断言转换后的数组与期望结果一致
        assert_array_equal(toarray(binarized), expected)
        # 断言稀疏性与预期一致
        assert issparse(binarized) == sparse_output
        # 对转换后的数组进行逆转换
        inverse_output = lb.inverse_transform(binarized)
        # 断言逆转换后的数组与原始数组一致
        assert_array_equal(toarray(inverse_output), toarray(y))
        # 断言稀疏性与原始数组一致
        assert issparse(inverse_output) == issparse(y)
def test_label_binarize_binary():
    # 定义二分类标签
    y = [0, 1, 0]
    # 类别列表
    classes = [0, 1]
    # 正类标签
    pos_label = 2
    # 负类标签
    neg_label = -1
    # 预期结果，numpy 数组，对应的二分类输出的第二列结果
    expected = np.array([[2, -1], [-1, 2], [2, -1]])[:, 1].reshape((-1, 1))

    # 调用检查二值化结果函数，验证结果是否符合预期
    check_binarized_results(y, classes, pos_label, neg_label, expected)

    # 二分类情况，当 sparse_output=True 时不应该抛出 ValueError
    y = [0, 1, 0]
    classes = [0, 1]
    pos_label = 3
    neg_label = 0
    expected = np.array([[3, 0], [0, 3], [3, 0]])[:, 1].reshape((-1, 1))

    # 再次调用检查二值化结果函数，验证结果是否符合预期
    check_binarized_results(y, classes, pos_label, neg_label, expected)


def test_label_binarize_multiclass():
    # 定义多分类标签
    y = [0, 1, 2]
    # 类别列表
    classes = [0, 1, 2]
    # 正类标签
    pos_label = 2
    # 负类标签
    neg_label = 0
    # 预期结果，2 倍的单位矩阵
    expected = 2 * np.eye(3)

    # 调用检查二值化结果函数，验证结果是否符合预期
    check_binarized_results(y, classes, pos_label, neg_label, expected)

    # 使用 pytest 验证，当 sparse_output=True 时应该抛出 ValueError
    with pytest.raises(ValueError):
        label_binarize(
            y, classes=classes, neg_label=-1, pos_label=pos_label, sparse_output=True
        )


@pytest.mark.parametrize(
    "arr_type",
    [np.array]
    + COO_CONTAINERS
    + CSC_CONTAINERS
    + CSR_CONTAINERS
    + DOK_CONTAINERS
    + LIL_CONTAINERS,
)
def test_label_binarize_multilabel(arr_type):
    # 定义多标签索引
    y_ind = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
    # 类别列表
    classes = [0, 1, 2]
    # 正类标签
    pos_label = 2
    # 负类标签
    neg_label = 0
    # 预期结果，使用正类标签乘以多标签索引
    expected = pos_label * y_ind
    # 转换为对应的数组类型
    y = arr_type(y_ind)

    # 调用检查二值化结果函数，验证结果是否符合预期
    check_binarized_results(y, classes, pos_label, neg_label, expected)

    # 使用 pytest 验证，当 sparse_output=True 时应该抛出 ValueError
    with pytest.raises(ValueError):
        label_binarize(
            y, classes=classes, neg_label=-1, pos_label=pos_label, sparse_output=True
        )


def test_invalid_input_label_binarize():
    # 使用 pytest 验证，应该抛出 ValueError，因为标签不匹配
    with pytest.raises(ValueError):
        label_binarize([0, 2], classes=[0, 2], pos_label=0, neg_label=1)
    # 使用 pytest 验证，应该抛出 ValueError，因为目标数据是连续的
    with pytest.raises(ValueError, match="continuous target data is not "):
        label_binarize([1.2, 2.7], classes=[0, 1])
    # 使用 pytest 验证，应该抛出 ValueError，因为标签不匹配
    with pytest.raises(ValueError, match="mismatch with the labels"):
        label_binarize([[1, 3]], classes=[1, 2, 3])


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_inverse_binarize_multiclass(csr_container):
    # 调用 _inverse_binarize_multiclass 函数，得到结果
    got = _inverse_binarize_multiclass(
        csr_container([[0, 1, 0], [-1, 0, -1], [0, 0, 0]]), np.arange(3)
    )
    # 使用 assert 比较得到的结果和预期的结果是否一致
    assert_array_equal(got, np.array([1, 1, 0]))


def test_nan_label_encoder():
    """Check that label encoder encodes nans in transform.

    Non-regression test for #22628.
    """
    # 创建 LabelEncoder 对象
    le = LabelEncoder()
    # 对标签进行编码
    le.fit(["a", "a", "b", np.nan])

    # 对 NaN 进行转换，验证转换后的结果是否符合预期
    y_trans = le.transform([np.nan])
    assert_array_equal(y_trans, [2])


@pytest.mark.parametrize(
    "encoder", [LabelEncoder(), LabelBinarizer(), MultiLabelBinarizer()]
)
def test_label_encoders_do_not_have_set_output(encoder):
    """Check that label encoders do not define set_output and work with y as a kwarg.

    Non-regression test for #26854.
    """
    # 验证 label encoder 对象没有定义 set_output 方法，并且可以使用 y 作为关键字参数
    assert not hasattr(encoder, "set_output")
    # 使用 y 作为关键字参数进行编码转换，并验证结果
    y_encoded_with_kwarg = encoder.fit_transform(y=["a", "b", "c"])
    # 使用位置参数进行编码转换，并验证结果
    y_encoded_positional = encoder.fit_transform(["a", "b", "c"])
    # 使用断言比较两个数组是否相等，如果不相等将引发 AssertionError
    assert_array_equal(y_encoded_with_kwarg, y_encoded_positional)
# 使用 pytest.mark.parametrize 装饰器，将 yield_namespace_device_dtype_combinations() 返回的参数组合应用于测试函数的 array_namespace, device, dtype 参数
@pytest.mark.parametrize(
    "array_namespace, device, dtype", yield_namespace_device_dtype_combinations()
)
# 使用 pytest.mark.parametrize 装饰器，将多个不同的 numpy 数组作为 y 参数传入测试函数
@pytest.mark.parametrize(
    "y",
    [
        np.array([2, 1, 3, 1, 3]),
        np.array([1, 1, 4, 5, -1, 0]),
        np.array([3, 5, 9, 5, 9, 3]),
    ],
)
# 定义测试函数 test_label_encoder_array_api_compliance，接受参数 y, array_namespace, device, dtype
def test_label_encoder_array_api_compliance(y, array_namespace, device, dtype):
    # 根据 array_namespace 和 device 获取相应的数组操作 API
    xp = _array_api_for_tests(array_namespace, device)
    # 使用 xp.asarray 方法将 y 转换为 xp 对象，并指定 device
    xp_y = xp.asarray(y, device=device)
    # 进入上下文，配置数组操作 API 的调度为真
    with config_context(array_api_dispatch=True):
        # 创建 xp_label 和 np_label 两个 LabelEncoder 实例
        xp_label = LabelEncoder()
        np_label = LabelEncoder()
        # 使用 xp_label.fit 方法拟合 xp_y 数据
        xp_label = xp_label.fit(xp_y)
        # 使用 xp_label.transform 方法对 xp_y 进行转换
        xp_transformed = xp_label.transform(xp_y)
        # 使用 xp_label.inverse_transform 方法对 xp_transformed 进行逆转换
        xp_inv_transformed = xp_label.inverse_transform(xp_transformed)
        # 使用 np_label.fit 方法拟合原始的 y 数据
        np_label = np_label.fit(y)
        # 使用 np_label.transform 方法对 y 进行转换
        np_transformed = np_label.transform(y)
        # 断言 xp_transformed 的命名空间的第一个元素的名称与 xp 的名称相同
        assert get_namespace(xp_transformed)[0].__name__ == xp.__name__
        # 断言 xp_inv_transformed 的命名空间的第一个元素的名称与 xp 的名称相同
        assert get_namespace(xp_inv_transformed)[0].__name__ == xp.__name__
        # 断言 xp_label.classes_ 的命名空间的第一个元素的名称与 xp 的名称相同
        assert get_namespace(xp_label.classes_)[0].__name__ == xp.__name__
        # 断言 _convert_to_numpy 方法将 xp_transformed 转换为 numpy 格式后与 np_transformed 相等
        assert_array_equal(_convert_to_numpy(xp_transformed, xp), np_transformed)
        # 断言 _convert_to_numpy 方法将 xp_inv_transformed 转换为 numpy 格式后与 y 相等
        assert_array_equal(_convert_to_numpy(xp_inv_transformed, xp), y)
        # 断言 _convert_to_numpy 方法将 xp_label.classes_ 转换为 numpy 格式后与 np_label.classes_ 相等
        assert_array_equal(_convert_to_numpy(xp_label.classes_, xp), np_label.classes_)

        # 重新创建 xp_label 和 np_label 两个 LabelEncoder 实例
        xp_label = LabelEncoder()
        np_label = LabelEncoder()
        # 使用 xp_label.fit_transform 方法直接对 xp_y 进行拟合和转换
        xp_transformed = xp_label.fit_transform(xp_y)
        # 使用 np_label.fit_transform 方法直接对 y 进行拟合和转换
        np_transformed = np_label.fit_transform(y)
        # 断言 xp_transformed 的命名空间的第一个元素的名称与 xp 的名称相同
        assert get_namespace(xp_transformed)[0].__name__ == xp.__name__
        # 断言 xp_label.classes_ 的命名空间的第一个元素的名称与 xp 的名称相同
        assert get_namespace(xp_label.classes_)[0].__name__ == xp.__name__
        # 断言 _convert_to_numpy 方法将 xp_transformed 转换为 numpy 格式后与 np_transformed 相等
        assert_array_equal(_convert_to_numpy(xp_transformed, xp), np_transformed)
        # 断言 _convert_to_numpy 方法将 xp_label.classes_ 转换为 numpy 格式后与 np_label.classes_ 相等
        assert_array_equal(_convert_to_numpy(xp_label.classes_, xp), np_label.classes_)
```