# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\tests\test_target_encoder.py`

```
# 导入正则表达式模块
import re

# 导入NumPy库，并使用简化名称 np
import numpy as np

# 导入 pytest 库，用于单元测试
import pytest

# 从 NumPy 测试模块中导入断言函数
from numpy.testing import assert_allclose, assert_array_equal

# 导入 scikit-learn 中的随机森林回归器和岭回归器
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# 导入 scikit-learn 中的交叉验证、数据集分割等功能模块
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

# 导入 scikit-learn 中的管道构建工具
from sklearn.pipeline import make_pipeline

# 导入 scikit-learn 中的数据预处理工具
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelBinarizer,
    LabelEncoder,
    TargetEncoder,
)


def _encode_target(X_ordinal, y_numeric, n_categories, smooth):
    """Simple Python implementation of target encoding."""
    # 初始化当前编码结果的数组，全部置零
    cur_encodings = np.zeros(n_categories, dtype=np.float64)
    # 计算目标变量 y 的均值
    y_mean = np.mean(y_numeric)

    # 根据 smooth 参数选择平滑方式
    if smooth == "auto":
        # 计算 y 的方差
        y_variance = np.var(y_numeric)
        # 遍历类别进行编码
        for c in range(n_categories):
            # 获取属于当前类别的目标变量子集
            y_subset = y_numeric[X_ordinal == c]
            # 获取子集大小
            n_i = y_subset.shape[0]

            # 如果子集为空，使用整体 y 的均值作为编码值
            if n_i == 0:
                cur_encodings[c] = y_mean
                continue

            # 计算子集的方差比例
            y_subset_variance = np.var(y_subset)
            m = y_subset_variance / y_variance
            lambda_ = n_i / (n_i + m)

            # 计算平滑后的编码值
            cur_encodings[c] = lambda_ * np.mean(y_subset) + (1 - lambda_) * y_mean
        return cur_encodings
    else:  # float 类型的 smooth 参数
        # 遍历类别进行编码
        for c in range(n_categories):
            # 获取属于当前类别的目标变量子集
            y_subset = y_numeric[X_ordinal == c]
            # 计算加权平均值
            current_sum = np.sum(y_subset) + y_mean * smooth
            current_cnt = y_subset.shape[0] + smooth
            cur_encodings[c] = current_sum / current_cnt
        return cur_encodings


@pytest.mark.parametrize(
    "categories, unknown_value",
    [
        # 不同类型的参数化测试用例
        ([np.array([0, 1, 2], dtype=np.int64)], 4),
        ([np.array([1.0, 3.0, np.nan], dtype=np.float64)], 6.0),
        ([np.array(["cat", "dog", "snake"], dtype=object)], "bear"),
        ("auto", 3),
    ],
)
@pytest.mark.parametrize("smooth", [5.0, "auto"])
@pytest.mark.parametrize("target_type", ["binary", "continuous"])
def test_encoding(categories, unknown_value, global_random_seed, smooth, target_type):
    """Check encoding for binary and continuous targets.

    Compare the values returned by `TargetEncoder.fit_transform` against the
    expected encodings for cv splits from a naive reference Python
    implementation in _encode_target.
    """
    # 类别数目
    n_categories = 3

    # 创建训练集和测试集数据数组
    X_train_int_array = np.array([[0] * 20 + [1] * 30 + [2] * 40], dtype=np.int64).T
    X_test_int_array = np.array([[0, 1, 2]], dtype=np.int64).T

    # 训练集样本数
    n_samples = X_train_int_array.shape[0]

    # 根据 categories 参数选择数据集
    if categories == "auto":
        X_train = X_train_int_array
        X_test = X_test_int_array
    else:
        X_train = categories[0][X_train_int_array]
        X_test = categories[0][X_test_int_array]

    # 在测试集中添加未知值
    X_test = np.concatenate((X_test, [[unknown_value]]))

    # 创建随机数生成器
    data_rng = np.random.RandomState(global_random_seed)

    # 交叉验证的折数
    n_splits = 3
    # 如果目标类型为 "binary"，生成一个随机的二元分类标签数组
    y_numeric = data_rng.randint(low=0, high=2, size=n_samples)
    # 定义二元分类的目标名称数组
    target_names = np.array(["cat", "dog"], dtype=object)
    # 将随机生成的数值映射到目标名称，构成训练集的目标标签数组
    y_train = target_names[y_numeric]

else:
    # 如果目标类型不是 "binary"，则必须是 "continuous"，进行断言验证
    assert target_type == "continuous"
    # 生成一个在指定范围内均匀分布的数值数组作为训练集的目标标签数组
    y_numeric = data_rng.uniform(low=-10, high=20, size=n_samples)
    y_train = y_numeric

# 对样本索引进行随机重排列
shuffled_idx = data_rng.permutation(n_samples)
# 根据重排后的索引对整数型特征数组进行重新排序
X_train_int_array = X_train_int_array[shuffled_idx]
# 根据重排后的索引对特征数组进行重新排序
X_train = X_train[shuffled_idx]
# 根据重排后的索引对目标标签数组进行重新排序
y_train = y_train[shuffled_idx]
# 根据重排后的索引对数值型目标标签数组进行重新排序
y_numeric = y_numeric[shuffled_idx]

# 定义交叉验证策略
if target_type == "binary":
    # 如果目标类型为 "binary"，使用分层 k 折交叉验证
    cv = StratifiedKFold(
        n_splits=n_splits, random_state=global_random_seed, shuffle=True
    )
else:
    # 否则使用普通 k 折交叉验证
    cv = KFold(n_splits=n_splits, random_state=global_random_seed, shuffle=True)

# 使用参考的 Python 实现计算预期的特征变换值
expected_X_fit_transform = np.empty_like(X_train_int_array, dtype=np.float64)

# 在交叉验证的每个训练集上计算目标编码值
for train_idx, test_idx in cv.split(X_train_int_array, y_train):
    # 获取当前训练集的特征和目标标签
    X_, y_ = X_train_int_array[train_idx, 0], y_numeric[train_idx]
    # 计算当前训练集的目标编码值
    cur_encodings = _encode_target(X_, y_, n_categories, smooth)
    # 将计算得到的编码值填充到预期的特征变换值中的相应位置
    expected_X_fit_transform[test_idx, 0] = cur_encodings[
        X_train_int_array[test_idx, 0]
    ]

# 检查是否可以通过在估算器上使用相同的 CV 参数调用 `fit_transform` 来获得相同的编码值
target_encoder = TargetEncoder(
    smooth=smooth,
    categories=categories,
    cv=n_splits,
    random_state=global_random_seed,
)

# 对训练数据进行目标编码转换
X_fit_transform = target_encoder.fit_transform(X_train, y_train)

# 断言目标编码器的目标类型属性与给定的目标类型相匹配
assert target_encoder.target_type_ == target_type
# 断言所有的目标编码值与预期的特征变换值非常接近
assert_allclose(X_fit_transform, expected_X_fit_transform)
# 断言目标编码器的编码器列表长度为 1
assert len(target_encoder.encodings_) == 1
if target_type == "binary":
    # 如果目标类型为 "binary"，断言编码器的类别数组与目标名称数组相等
    assert_array_equal(target_encoder.classes_, target_names)
else:
    # 否则断言编码器的类别数组为空
    assert target_encoder.classes_ is None

# 计算所有数据的编码值以验证 `transform` 方法
y_mean = np.mean(y_numeric)
# 计算预期的所有数据的编码值
expected_encodings = _encode_target(
    X_train_int_array[:, 0], y_numeric, n_categories, smooth
)
# 断言目标编码器的编码器列表中的第一个编码器与预期的编码值非常接近
assert_allclose(target_encoder.encodings_[0], expected_encodings)
# 断言目标编码器的目标均值属性与数值的均值非常接近
assert target_encoder.target_mean_ == pytest.approx(y_mean)

# 对测试数据进行转换，最后一个值未知，因此编码为目标均值
expected_X_test_transform = np.concatenate(
    (expected_encodings, np.array([y_mean]))
).reshape(-1, 1)

# 使用目标编码器对测试数据进行转换
X_test_transform = target_encoder.transform(X_test)
# 断言测试数据的转换结果与预期的测试数据编码值非常接近
assert_allclose(X_test_transform, expected_X_test_transform)
@pytest.mark.parametrize(
    "categories, unknown_values",
    [
        ([np.array([0, 1, 2], dtype=np.int64)], "auto"),
        ([np.array(["cat", "dog", "snake"], dtype=object)], ["bear", "rabbit"]),
    ],
)
@pytest.mark.parametrize(
    "target_labels", [np.array([1, 2, 3]), np.array(["a", "b", "c"])]
)
@pytest.mark.parametrize("smooth", [5.0, "auto"])
def test_encoding_multiclass(
    global_random_seed, categories, unknown_values, target_labels, smooth
):
    """Check encoding for multiclass targets."""
    rng = np.random.RandomState(global_random_seed)  # 使用全局随机种子初始化随机数生成器

    n_samples = 80  # 样本数
    n_features = 2  # 特征数
    feat_1_int = np.array(rng.randint(low=0, high=2, size=n_samples))  # 生成随机整数特征1
    feat_2_int = np.array(rng.randint(low=0, high=3, size=n_samples))  # 生成随机整数特征2
    feat_1 = categories[0][feat_1_int]  # 使用categories中的第一个数组，根据feat_1_int索引获取特征1值
    feat_2 = categories[0][feat_2_int]  # 使用categories中的第一个数组，根据feat_2_int索引获取特征2值
    X_train = np.column_stack((feat_1, feat_2))  # 将特征1和特征2堆叠成特征矩阵
    X_train_int = np.column_stack((feat_1_int, feat_2_int))  # 将整数形式的特征1和特征2堆叠成整数特征矩阵
    categories_ = [[0, 1], [0, 1, 2]]  # 特征的可能取值列表

    n_classes = 3  # 类别数
    y_train_int = np.array(rng.randint(low=0, high=n_classes, size=n_samples))  # 生成随机整数类别标签
    y_train = target_labels[y_train_int]  # 根据随机整数类别标签获取目标标签值
    y_train_enc = LabelBinarizer().fit_transform(y_train)  # 对目标标签进行二进制编码转换

    n_splits = 3  # 分割数
    cv = StratifiedKFold(
        n_splits=n_splits, random_state=global_random_seed, shuffle=True
    )  # 使用分层K折交叉验证进行数据集分割，保持分布和随机种子

    # Manually compute encodings for cv splits to validate `fit_transform`
    expected_X_fit_transform = np.empty(
        (X_train_int.shape[0], X_train_int.shape[1] * n_classes),
        dtype=np.float64,
    )  # 创建一个空的数组来存储预期的编码结果

    for f_idx, cats in enumerate(categories_):  # 遍历特征索引和可能取值列表
        for c_idx in range(n_classes):  # 遍历类别索引
            for train_idx, test_idx in cv.split(X_train, y_train):  # 遍历交叉验证的训练集索引和测试集索引
                y_class = y_train_enc[:, c_idx]  # 当前类别的二进制编码
                X_, y_ = X_train_int[train_idx, f_idx], y_class[train_idx]  # 当前特征和类别的训练集数据
                current_encoding = _encode_target(X_, y_, len(cats), smooth)  # 计算当前特征的目标编码
                # 计算在编码结果数组中的索引
                exp_idx = c_idx + (f_idx * n_classes)
                expected_X_fit_transform[test_idx, exp_idx] = current_encoding[
                    X_train_int[test_idx, f_idx]
                ]  # 将当前编码结果存储到预期的编码数组中

    target_encoder = TargetEncoder(
        smooth=smooth,
        cv=n_splits,
        random_state=global_random_seed,
    )  # 初始化目标编码器对象，使用平滑参数、交叉验证数和随机种子

    X_fit_transform = target_encoder.fit_transform(X_train, y_train)  # 对训练数据进行目标编码转换

    assert target_encoder.target_type_ == "multiclass"  # 断言目标编码类型为多类别
    assert_allclose(X_fit_transform, expected_X_fit_transform)  # 断言目标编码转换结果与预期的编码结果接近

    # Manually compute encoding to validate `transform`
    expected_encodings = []  # 存储预期的编码结果列表
    for f_idx, cats in enumerate(categories_):  # 遍历特征索引和可能取值列表
        for c_idx in range(n_classes):  # 遍历类别索引
            y_class = y_train_enc[:, c_idx]  # 当前类别的二进制编码
            current_encoding = _encode_target(
                X_train_int[:, f_idx], y_class, len(cats), smooth
            )  # 计算当前特征的目标编码
            expected_encodings.append(current_encoding)  # 将当前编码结果添加到预期的编码结果列表中

    assert len(target_encoder.encodings_) == n_features * n_classes  # 断言目标编码器的编码结果数量正确
    # 对每个编码值进行断言，确保与期望的编码值接近
    for i in range(n_features * n_classes):
        assert_allclose(target_encoder.encodings_[i], expected_encodings[i])
    
    # 断言目标编码器的类别与目标标签相等
    assert_array_equal(target_encoder.classes_, target_labels)

    # 将未知值包含在末尾
    X_test_int = np.array([[0, 1], [1, 2], [4, 5]])
    if unknown_values == "auto":
        X_test = X_test_int
    else:
        # 根据列索引将类别映射到测试数据中
        X_test = np.empty_like(X_test_int[:-1, :], dtype=object)
        for column_idx in range(X_test_int.shape[1]):
            X_test[:, column_idx] = categories[0][X_test_int[:-1, column_idx]]
        # 将未知值添加到末尾
        X_test = np.vstack((X_test, unknown_values))

    # 计算训练集目标的平均值
    y_mean = np.mean(y_train_enc, axis=0)
    
    # 创建一个空的预期测试转换矩阵，用于存储预期结果
    expected_X_test_transform = np.empty(
        (X_test_int.shape[0], X_test_int.shape[1] * n_classes),
        dtype=np.float64,
    )
    
    # 获取测试数据的行数
    n_rows = X_test_int.shape[0]
    
    # 特征索引，用于确定哪个特征的编码将被应用到预期转换中的哪个位置
    f_idx = [0, 0, 0, 1, 1, 1]
    
    # 对于每一行，根据预期的编码值填充预期的测试转换矩阵
    # 最后一行是未知值，稍后处理
    for row_idx in range(n_rows - 1):
        for i, enc in enumerate(expected_encodings):
            expected_X_test_transform[row_idx, i] = enc[X_test_int[row_idx, f_idx[i]]]

    # 将未知值编码为每个类别的目标均值
    # `y_mean` 包含每个类别的目标均值，因此循环遍历每个类别的均值，`n_features` 次
    mean_idx = [0, 1, 2, 0, 1, 2]
    for i in range(n_classes * n_features):
        expected_X_test_transform[n_rows - 1, i] = y_mean[mean_idx[i]]

    # 使用目标编码器对测试数据进行转换
    X_test_transform = target_encoder.transform(X_test)
    
    # 断言转换后的测试数据与预期的测试转换矩阵接近
    assert_allclose(X_test_transform, expected_X_test_transform)
@pytest.mark.parametrize(
    "X, categories",
    [
        (
            np.array([[0] * 10 + [1] * 10 + [3]], dtype=np.int64).T,  # 创建一个包含未知类别的输入数组 X
            [[0, 1, 2]],  # 指定类别列表 categories
        ),
        (
            np.array(
                [["cat"] * 10 + ["dog"] * 10 + ["snake"]], dtype=object
            ).T,  # 创建一个包含未知类别的输入数组 X
            [["dog", "cat", "cow"]],  # 指定类别列表 categories
        ),
    ],
)
@pytest.mark.parametrize("smooth", [4.0, "auto"])
def test_custom_categories(X, categories, smooth):
    """Custom categories with unknown categories that are not in training data."""
    rng = np.random.RandomState(0)  # 初始化一个随机数生成器
    y = rng.uniform(low=-10, high=20, size=X.shape[0])  # 生成随机目标变量 y
    enc = TargetEncoder(categories=categories, smooth=smooth, random_state=0).fit(X, y)

    # The last element is unknown and encoded as the mean
    y_mean = y.mean()  # 计算目标变量 y 的均值
    X_trans = enc.transform(X[-1:])  # 对最后一个样本进行转换
    assert X_trans[0, 0] == pytest.approx(y_mean)  # 断言转换结果与均值近似相等

    assert len(enc.encodings_) == 1  # 断言编码器的编码数量为 1
    # custom category that is not in training data
    assert enc.encodings_[0][-1] == pytest.approx(y_mean)  # 断言最后一个编码的值与均值近似相等


@pytest.mark.parametrize(
    "y, msg",
    [
        ([1, 2, 0, 1], "Found input variables with inconsistent"),  # 指定错误消息和目标变量 y
        (
            np.array([[1, 2, 0], [1, 2, 3]]).T,
            "Target type was inferred to be 'multiclass-multioutput'",  # 指定错误消息和目标变量 y
        ),
    ],
)
def test_errors(y, msg):
    """Check invalidate input."""
    X = np.array([[1, 0, 1]]).T  # 创建输入特征矩阵 X

    enc = TargetEncoder()  # 初始化编码器
    with pytest.raises(ValueError, match=msg):  # 断言抛出 ValueError 异常，并匹配指定消息
        enc.fit_transform(X, y)


def test_use_regression_target():
    """Check inferred and specified `target_type` on regression target."""
    X = np.array([[0, 1, 0, 1, 0, 1]]).T  # 创建输入特征矩阵 X
    y = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 4.0])  # 创建目标变量 y

    enc = TargetEncoder(cv=2)  # 初始化编码器
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "The least populated class in y has only 1 members, which is less than"
            " n_splits=2."
        ),  # 断言警告消息匹配
    ):
        enc.fit_transform(X, y)
    assert enc.target_type_ == "multiclass"  # 断言目标类型为多类分类

    enc = TargetEncoder(cv=2, target_type="continuous")  # 初始化编码器，指定目标类型为连续型
    enc.fit_transform(X, y)
    assert enc.target_type_ == "continuous"  # 断言目标类型为连续型


@pytest.mark.parametrize(
    "y, feature_names",
    [
        ([1, 2] * 10, ["A", "B"]),  # 指定目标变量 y 和特征名列表 feature_names
        ([1, 2, 3] * 6 + [1, 2], ["A_1", "A_2", "A_3", "B_1", "B_2", "B_3"]),  # 指定目标变量 y 和特征名列表 feature_names
        (
            ["y1", "y2", "y3"] * 6 + ["y1", "y2"],
            ["A_y1", "A_y2", "A_y3", "B_y1", "B_y2", "B_y3"],  # 指定目标变量 y 和特征名列表 feature_names
        ),
    ],
)
def test_feature_names_out_set_output(y, feature_names):
    """Check TargetEncoder works with set_output."""
    pd = pytest.importorskip("pandas")  # 导入并检查是否能成功导入 pandas 库

    X_df = pd.DataFrame({"A": ["a", "b"] * 10, "B": [1, 2] * 10})  # 创建 DataFrame X_df

    enc_default = TargetEncoder(cv=2, smooth=3.0, random_state=0)  # 初始化默认输出类型的编码器
    enc_default.set_output(transform="default")  # 设置输出类型为默认
    enc_pandas = TargetEncoder(cv=2, smooth=3.0, random_state=0)  # 初始化 pandas 输出类型的编码器
    enc_pandas.set_output(transform="pandas")  # 设置输出类型为 pandas

    X_default = enc_default.fit_transform(X_df, y)  # 使用默认输出类型的编码器对 X_df 进行拟合和转换
    # 使用 enc_pandas 对象对 X_df 数据集进行拟合和转换，同时使用 y 数据来指导转换（如果有必要）
    X_pandas = enc_pandas.fit_transform(X_df, y)

    # 断言 X_pandas 转换为 NumPy 数组后与 X_default 数组非常接近（数值近似相等）
    assert_allclose(X_pandas.to_numpy(), X_default)
    
    # 断言 enc_pandas 对象返回的输出特征名称与预期的 feature_names 数组完全相等
    assert_array_equal(enc_pandas.get_feature_names_out(), feature_names)
    
    # 再次断言 enc_pandas 对象返回的输出特征名称与 X_pandas 的列名完全相等
    assert_array_equal(enc_pandas.get_feature_names_out(), X_pandas.columns)
# 使用 pytest 的参数化功能，为 to_pandas、smooth 和 target_type 参数生成多组测试用例
@pytest.mark.parametrize("to_pandas", [True, False])
@pytest.mark.parametrize("smooth", [1.0, "auto"])
@pytest.mark.parametrize("target_type", ["binary-ints", "binary-str", "continuous"])
def test_multiple_features_quick(to_pandas, smooth, target_type):
    """Check target encoder with multiple features."""
    # 创建一个二维数组 X_ordinal，包含整数类型数据
    X_ordinal = np.array(
        [[1, 1], [0, 1], [1, 1], [2, 1], [1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int64
    )
    
    # 根据 target_type 类型进行不同的标签处理
    if target_type == "binary-str":
        # 如果标签类型是字符串，创建字符串数组 y_train，并用 LabelEncoder 编码
        y_train = np.array(["a", "b", "a", "a", "b", "b", "a", "b"])
        y_integer = LabelEncoder().fit_transform(y_train)
        # 创建 StratifiedKFold 交叉验证对象
        cv = StratifiedKFold(2, random_state=0, shuffle=True)
    elif target_type == "binary-ints":
        # 如果标签类型是整数，创建整数数组 y_train，并用 LabelEncoder 编码
        y_train = np.array([3, 4, 3, 3, 3, 4, 4, 4])
        y_integer = LabelEncoder().fit_transform(y_train)
        # 创建 StratifiedKFold 交叉验证对象
        cv = StratifiedKFold(2, random_state=0, shuffle=True)
    else:
        # 如果标签类型是连续值，创建浮点数数组 y_train
        y_train = np.array([3.0, 5.1, 2.4, 3.5, 4.1, 5.5, 10.3, 7.3], dtype=np.float32)
        y_integer = y_train
        # 创建 KFold 交叉验证对象
        cv = KFold(2, random_state=0, shuffle=True)
    
    # 计算标签的均值
    y_mean = np.mean(y_integer)
    # 定义多个类别列表
    categories = [[0, 1, 2], [0, 1]]

    # 创建测试数据 X_test，包含整数类型数据
    X_test = np.array(
        [
            [0, 1],
            [3, 0],  # 3 代表未知值
            [1, 10],  # 10 代表未知值
        ],
        dtype=np.int64,
    )

    # 根据 to_pandas 参数决定是否导入 pytest 和 pandas
    if to_pandas:
        pd = pytest.importorskip("pandas")
        # 将第二个特征转换为对象类型，并创建 DataFrame X_train
        X_train = pd.DataFrame(
            {
                "feat0": X_ordinal[:, 0],
                "feat1": np.array(["cat", "dog"], dtype=object)[X_ordinal[:, 1]],
            }
        )
        # "snake" 代表未知值
        X_test = pd.DataFrame({"feat0": X_test[:, 0], "feat1": ["dog", "cat", "snake"]})
    else:
        # 如果不使用 pandas，直接使用原始数据 X_ordinal
        X_train = X_ordinal

    # 手动计算 fit_transform 的预期结果
    expected_X_fit_transform = np.empty_like(X_ordinal, dtype=np.float64)
    for f_idx, cats in enumerate(categories):
        for train_idx, test_idx in cv.split(X_ordinal, y_integer):
            X_, y_ = X_ordinal[train_idx, f_idx], y_integer[train_idx]
            current_encoding = _encode_target(X_, y_, len(cats), smooth)
            expected_X_fit_transform[test_idx, f_idx] = current_encoding[
                X_ordinal[test_idx, f_idx]
            ]

    # 手动计算 transform 的预期编码
    expected_encodings = []
    for f_idx, cats in enumerate(categories):
        current_encoding = _encode_target(
            X_ordinal[:, f_idx], y_integer, len(cats), smooth
        )
        expected_encodings.append(current_encoding)

    # 创建 TargetEncoder 对象 enc
    enc = TargetEncoder(smooth=smooth, cv=2, random_state=0)
    # 对 X_train, y_train 进行 fit_transform
    X_fit_transform = enc.fit_transform(X_train, y_train)
    # 使用 assert_allclose 检查 X_fit_transform 和预期的 expected_X_fit_transform 是否近似相等
    assert_allclose(X_fit_transform, expected_X_fit_transform)
    # 断言编码器的编码数为2
    assert len(enc.encodings_) == 2
    # 遍历前两个编码，确保它们与期望的编码接近
    for i in range(2):
        assert_allclose(enc.encodings_[i], expected_encodings[i])

    # 对测试数据集进行编码转换
    X_test_transform = enc.transform(X_test)
    # 断言转换后的测试数据与期望的转换结果非常接近
    assert_allclose(X_test_transform, expected_X_test_transform)
`
@pytest.mark.parametrize(
    "y, y_mean",
    [
        (np.array([3.4] * 20), 3.4),  # 第一个测试用例，目标变量 y 为常数 3.4，y_mean 也是 3.4
        (np.array([0] * 20), 0),      # 第二个测试用例，目标变量 y 为常数 0，y_mean 也是 0
        (np.array(["a"] * 20, dtype=object), 0),  # 第三个测试用例，目标变量 y 为字符串 'a'，dtype 为 object，y_mean 设为 0
    ],
    ids=["continuous", "binary", "binary-string"],  # 定义测试用例的标识
)
@pytest.mark.parametrize("smooth", ["auto", 4.0, 0.0])  # 测试 smooth 参数的不同值，包括 "auto"、4.0 和 0.0
def test_constant_target_and_feature(y, y_mean, smooth):
    """Check edge case where feature and target is constant."""  # 测试特征和目标变量均为常数的情况
    X = np.array([[1] * 20]).T  # 创建特征矩阵 X，包含一列全为 1 的数据，共 20 行
    n_samples = X.shape[0]  # 获取样本数量

    enc = TargetEncoder(cv=2, smooth=smooth, random_state=0)  # 初始化 TargetEncoder，指定交叉验证折数和 smoothing 参数
    X_trans = enc.fit_transform(X, y)  # 对特征 X 和目标 y 进行拟合并转换
    assert_allclose(X_trans, np.repeat([[y_mean]], n_samples, axis=0))  # 断言转换后的 X_trans 数据与 y_mean 重复的结果接近
    assert enc.encodings_[0][0] == pytest.approx(y_mean)  # 断言编码器的第一个编码值接近 y_mean
    assert enc.target_mean_ == pytest.approx(y_mean)  # 断言目标均值接近 y_mean

    X_test = np.array([[1], [0]])  # 创建测试特征矩阵 X_test，包含两行数据，第一行为 1，第二行为 0
    X_test_trans = enc.transform(X_test)  # 使用已训练的编码器对测试特征进行转换
    assert_allclose(X_test_trans, np.repeat([[y_mean]], 2, axis=0))  # 断言测试特征转换结果与 y_mean 重复的结果接近


def test_fit_transform_not_associated_with_y_if_ordinal_categorical_is_not(
    global_random_seed,
):
    cardinality = 30  # 设置类别数目为 30，防止样本数量过大
    n_samples = 3000  # 设置样本数量为 3000
    rng = np.random.RandomState(global_random_seed)  # 初始化随机数生成器
    y_train = rng.normal(size=n_samples)  # 生成正态分布的目标变量 y_train
    X_train = rng.randint(0, cardinality, size=n_samples).reshape(-1, 1)  # 生成整数类型特征 X_train，取值范围为 [0, cardinality)，形状为 (n_samples, 1)

    y_sorted_indices = y_train.argsort()  # 获取目标 y_train 的排序索引
    y_train = y_train[y_sorted_indices]  # 根据排序索引对 y_train 进行排序
    X_train = X_train[y_sorted_indices]  # 根据排序索引对 X_train 进行排序

    target_encoder = TargetEncoder(shuffle=True, random_state=global_random_seed)  # 初始化 TargetEncoder，并设置 shuffle 为 True
    X_encoded_train_shuffled = target_encoder.fit_transform(X_train, y_train)  # 对训练集 X_train 和 y_train 进行拟合转换，启用 shuffle

    target_encoder = TargetEncoder(shuffle=False)  # 初始化 TargetEncoder，并设置 shuffle 为 False
    X_encoded_train_no_shuffled = target_encoder.fit_transform(X_train, y_train)  # 对训练集 X_train 和 y_train 进行拟合转换，不启用 shuffle

    regressor = RandomForestRegressor(
        n_estimators=10, min_samples_leaf=20, random_state=global_random_seed
    )  # 初始化回归模型，指定决策树数量和最小样本叶节点数

    cv = ShuffleSplit(n_splits=50, random_state=global_random_seed)  # 初始化 ShuffleSplit 交叉验证，设置折数为 50
    assert cross_val_score(regressor, X_train, y_train, cv=cv).mean() < 0.1  # 断言原始特征 X_train 在交叉验证中的均值得分小于 0.1
    assert (
        cross_val_score(regressor, X_encoded_train_shuffled, y_train, cv=cv).mean()
        < 0.1
    )  # 断言 shuffle 交叉验证后的特征 X_encoded_train_shuffled 在交叉验证中的均值得分小于 0.1

    assert (
        cross_val_score(regressor, X_encoded_train_no_shuffled, y_train, cv=cv).mean()
        > 0.5
    )  # 断言未 shuffle 的特征 X_encoded_train_no_shuffled 在交叉验证中的均值得分大于 0.5


def test_smooth_zero():
    """Check edge case with zero smoothing and cv does not contain category."""  # 测试 smooth 为 0 且交叉验证不包含类别的边缘情况
    # 创建一个包含10个元素的列向量 X，每个元素为 0 或 1
    X = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T
    
    # 创建一个包含10个元素的一维数组 y，表示对应的目标数值
    y = np.array([2.1, 4.3, 1.2, 3.1, 1.0, 9.0, 10.3, 14.2, 13.3, 15.0])
    
    # 创建一个 TargetEncoder 对象，设置平滑参数 smooth=0.0，不进行数据洗牌 shuffle=False，使用2折交叉验证 cv=2
    enc = TargetEncoder(smooth=0.0, shuffle=False, cv=2)
    
    # 使用 TargetEncoder 对象对 X 进行拟合和转换，得到转换后的结果 X_trans
    X_trans = enc.fit_transform(X, y)
    
    # 断言：对于第一个元素，因为 category 0 在后半部分不存在，所以它会被编码为后半部分 y[5:] 的均值
    assert_allclose(X_trans[0], np.mean(y[5:]))
    
    # 断言：对于最后一个元素，因为 category 1 在前半部分不存在，所以它会被编码为前半部分 y[:5] 的均值
    assert_allclose(X_trans[-1], np.mean(y[:5]))
@pytest.mark.parametrize("smooth", [0.0, 1e3, "auto"])
# 定义测试函数，参数化 smooth 参数，以便多次运行测试用例
def test_invariance_of_encoding_under_label_permutation(smooth, global_random_seed):
    # 检查编码是否不依赖于整数标签的值。这是一个相当简单的属性，但有助于理解下面的测试。
    rng = np.random.RandomState(global_random_seed)
    # 使用全局随机种子创建随机数生成器对象

    y = rng.normal(size=1000)
    # 生成一个大小为 1000 的正态分布随机数作为目标变量 y

    n_categories = 30
    # 定义类别数量为 30

    X = KBinsDiscretizer(n_bins=n_categories, encode="ordinal").fit_transform(
        y.reshape(-1, 1)
    )
    # 对目标变量 y 进行 KBinsDiscretizer 离散化处理，得到特征矩阵 X

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=global_random_seed
    )
    # 使用全局随机种子随机分割 X 和 y 为训练集和测试集

    permutated_labels = rng.permutation(n_categories)
    # 对类别标签进行随机排列，以验证编码是否对标签的排列不变

    X_train_permuted = permutated_labels[X_train.astype(np.int32)]
    X_test_permuted = permutated_labels[X_test.astype(np.int32)]
    # 根据随机排列的标签重新编码训练集和测试集的特征矩阵 X

    target_encoder = TargetEncoder(smooth=smooth, random_state=global_random_seed)
    # 初始化目标编码器对象，设置平滑参数和随机种子

    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    # 对训练集的特征进行目标编码

    X_test_encoded = target_encoder.transform(X_test)
    # 对测试集的特征进行目标编码

    X_train_permuted_encoded = target_encoder.fit_transform(X_train_permuted, y_train)
    # 对重新编码后的训练集特征进行目标编码

    X_test_permuted_encoded = target_encoder.transform(X_test_permuted)
    # 对重新编码后的测试集特征进行目标编码

    assert_allclose(X_train_encoded, X_train_permuted_encoded)
    # 断言：验证原始训练集编码结果与重新排列标签后训练集编码结果是否接近

    assert_allclose(X_test_encoded, X_test_permuted_encoded)
    # 断言：验证原始测试集编码结果与重新排列标签后测试集编码结果是否接近


@pytest.mark.parametrize("smooth", [0.0, "auto"])
# 定义测试函数，参数化 smooth 参数，以便多次运行测试用例
def test_target_encoding_for_linear_regression(smooth, global_random_seed):
    # 检查在对目标编码特征进行线性回归时的一些预期统计性质

    linear_regression = Ridge(alpha=1e-6, solver="lsqr", fit_intercept=False)
    # 初始化岭回归对象，设置 alpha 值、求解器为 "lsqr"，并且不拟合截距

    n_samples = 50_000
    # 定义样本数量为 50000

    rng = np.random.RandomState(global_random_seed)
    # 使用全局随机种子创建随机数生成器对象

    y = rng.randn(n_samples)
    # 生成大小为 n_samples 的正态分布随机数作为目标变量 y

    noise = 0.8 * rng.randn(n_samples)
    # 生成大小为 n_samples 的正态分布随机数作为噪声

    n_categories = 100
    # 定义类别数量为 100
    X_informative = KBinsDiscretizer(
        n_bins=n_categories,               # 使用 n_categories 个 bin 对数据进行分箱处理
        encode="ordinal",                  # 使用序数编码将数据编码为整数
        strategy="uniform",                # 使用均匀策略对数据进行分箱
        random_state=rng,                  # 设置随机数生成器的种子以保证可重复性
    ).fit_transform((y + noise).reshape(-1, 1))

    # 将标签进行随机置换，以隐藏这个特征对原始序数线性回归模型的信息影响
    permutated_labels = rng.permutation(n_categories)
    X_informative = permutated_labels[X_informative.astype(np.int32)]

    # 生成信息特征的洗牌副本，破坏其与目标变量的关系
    X_shuffled = rng.permutation(X_informative)

    # 同时包括一个非常高基数的分类特征，该特征本身与目标变量独立：在没有内部交叉验证的情况下对这种特征进行目标编码可能会导致下游回归器的灾难性过拟合，即使使用了收缩。这种特征通常表示样本的几乎唯一标识符。通常应从机器学习数据集中删除这些特征，但在这里我们希望研究目标编码的默认行为是否能够自动化地缓解它们。
    X_near_unique_categories = rng.choice(
        int(0.9 * n_samples), size=n_samples, replace=True
    ).reshape(-1, 1)

    # 组装数据集并进行训练测试分割：
    X = np.concatenate(
        [X_informative, X_shuffled, X_near_unique_categories],
        axis=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 首先检查在原始特征上训练的线性回归模型由于标签的无意义序数编码而导致欠拟合。
    raw_model = linear_regression.fit(X_train, y_train)
    assert raw_model.score(X_train, y_train) < 0.1
    assert raw_model.score(X_test, y_test) < 0.1

    # 现在使用内部 CV 机制进行目标编码，实现 fit_transform 时会自动进行
    model_with_cv = make_pipeline(
        TargetEncoder(smooth=smooth, random_state=rng), linear_regression
    ).fit(X_train, y_train)

    # 该模型应该能够很好地拟合数据，并且能够泛化到测试数据（假设分箱足够精细）。由于生成唯一信息特征时注入了噪声，R2 分数并不完美。
    coef = model_with_cv[-1].coef_
    assert model_with_cv.score(X_train, y_train) > 0.5, coef
    assert model_with_cv.score(X_test, y_test) > 0.5, coef

    # 目标编码恢复了目标编码后的唯一信息预测变量与目标之间斜率为1的线性关系。由于另外两个特征的目标编码不具信息性，这一点得以实现。
    # 使用内部交叉验证，多变量线性回归器将第一个特征的系数设为1，其他两个特征的系数设为0。
    assert coef[0] == pytest.approx(1, abs=1e-2)
    assert (np.abs(coef[1:]) < 0.2).all()

    # 现在通过分别在训练集上调用 fit 和 transform 来禁用内部交叉验证：
    # 初始化目标编码器，并在训练集上调用 fit 和 transform
    target_encoder = TargetEncoder(smooth=smooth, random_state=rng).fit(
        X_train, y_train
    )
    X_enc_no_cv_train = target_encoder.transform(X_train)
    X_enc_no_cv_test = target_encoder.transform(X_test)
    # 使用禁用交叉验证的编码后的数据在线性回归模型上进行拟合
    model_no_cv = linear_regression.fit(X_enc_no_cv_train, y_train)

    # 线性回归模型应该总是过拟合，因为相对于信息丰富的特征，它对高基数特征赋予了过多的权重。
    # 注意，即使使用了经验贝叶斯平滑，仅凭这一点还不足以防止这种过拟合。
    coef = model_no_cv.coef_
    assert model_no_cv.score(X_enc_no_cv_train, y_train) > 0.7, coef
    assert model_no_cv.score(X_enc_no_cv_test, y_test) < 0.5, coef

    # 模型过拟合是因为它对高基数但非信息丰富的特征赋予了过多的权重，而不是对低基数但信息丰富的特征。
    assert abs(coef[0]) < abs(coef[2])
def test_pandas_copy_on_write():
    """
    Test target-encoder cython code when y is read-only.

    The numpy array underlying df["y"] is read-only when copy-on-write is enabled.
    Non-regression test for gh-27879.
    """
    # 导入 pytest 库，如果版本小于 2.0 则跳过测试
    pd = pytest.importorskip("pandas", minversion="2.0")
    
    # 进入上下文环境，设置 pandas 的 copy-on-write 模式为 True
    with pd.option_context("mode.copy_on_write", True):
        # 创建一个包含两列 'x' 和 'y' 的 DataFrame
        df = pd.DataFrame({"x": ["a", "b", "b"], "y": [4.0, 5.0, 6.0]})
        
        # 使用 TargetEncoder 对象，针对 'x' 列进行拟合，'y' 列作为目标值
        TargetEncoder(target_type="continuous").fit(df[["x"]], df["y"])
```