# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\tests\test_dict_learning.py`

```
# 导入必要的库和模块
import itertools  # 提供迭代工具的函数，用于组合生成测试用例
import warnings  # 提供警告管理功能
from functools import partial  # 导入函数工具模块中的partial函数

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest，用于编写和运行测试用例

import sklearn  # 导入scikit-learn机器学习库
from sklearn.base import clone  # 导入clone函数，用于复制估计器对象
from sklearn.decomposition import (  # 导入分解模块中的特定类和函数
    DictionaryLearning,
    MiniBatchDictionaryLearning,
    SparseCoder,
    dict_learning,
    dict_learning_online,
    sparse_encode,
)
from sklearn.decomposition._dict_learning import _update_dict  # 导入更新字典的函数
from sklearn.exceptions import ConvergenceWarning  # 导入收敛警告类
from sklearn.utils import check_array  # 导入用于检查数组的函数
from sklearn.utils._testing import (  # 导入用于测试的辅助函数
    TempMemmap,
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.estimator_checks import (  # 导入用于估计器检查的函数
    check_transformer_data_not_an_array,
    check_transformer_general,
    check_transformers_unfitted,
)
from sklearn.utils.parallel import Parallel  # 导入并行计算模块

rng_global = np.random.RandomState(0)  # 创建全局随机种子
n_samples, n_features = 10, 8  # 定义样本数和特征数
X = rng_global.randn(n_samples, n_features)  # 生成服从正态分布的样本数据


def test_sparse_encode_shapes_omp():
    rng = np.random.RandomState(0)  # 创建本地随机种子
    algorithms = ["omp", "lasso_lars", "lasso_cd", "lars", "threshold"]  # 定义算法列表
    for n_components, n_samples in itertools.product([1, 5], [1, 9]):  # 组合生成测试用例
        X_ = rng.randn(n_samples, n_features)  # 生成服从正态分布的测试数据
        dictionary = rng.randn(n_components, n_features)  # 生成随机字典
        for algorithm, n_jobs in itertools.product(algorithms, [1, 2]):  # 组合生成算法和并行度
            code = sparse_encode(X_, dictionary, algorithm=algorithm, n_jobs=n_jobs)  # 执行稀疏编码
            assert code.shape == (n_samples, n_components)  # 断言编码结果的形状符合预期


def test_dict_learning_shapes():
    n_components = 5  # 设定字典的成分数
    dico = DictionaryLearning(n_components, random_state=0).fit(X)  # 创建并拟合字典学习模型
    assert dico.components_.shape == (n_components, n_features)  # 断言学习到的字典形状符合预期

    n_components = 1  # 设定字典的成分数为1
    dico = DictionaryLearning(n_components, random_state=0).fit(X)  # 创建并拟合字典学习模型
    assert dico.components_.shape == (n_components, n_features)  # 断言学习到的字典形状符合预期
    assert dico.transform(X).shape == (X.shape[0], n_components)  # 断言转换后的形状符合预期


def test_dict_learning_overcomplete():
    n_components = 12  # 设定超完备字典的成分数
    dico = DictionaryLearning(n_components, random_state=0).fit(X)  # 创建并拟合字典学习模型
    assert dico.components_.shape == (n_components, n_features)  # 断言学习到的字典形状符合预期


def test_max_iter():
    def ricker_function(resolution, center, width):
        """Discrete sub-sampled Ricker (Mexican hat) wavelet"""
        x = np.linspace(0, resolution - 1, resolution)  # 生成离散的x值数组
        x = (
            (2 / (np.sqrt(3 * width) * np.pi**0.25))  # 计算Ricker函数系数
            * (1 - (x - center) ** 2 / width**2)
            * np.exp(-((x - center) ** 2) / (2 * width**2))
        )
        return x  # 返回Ricker函数值数组

    def ricker_matrix(width, resolution, n_components):
        """Dictionary of Ricker (Mexican hat) wavelets"""
        centers = np.linspace(0, resolution - 1, n_components)  # 生成中心位置数组
        D = np.empty((n_components, resolution))  # 创建空的字典矩阵
        for i, center in enumerate(centers):  # 遍历中心位置
            D[i] = ricker_function(resolution, center, width)  # 填充字典矩阵的每一行
        D /= np.sqrt(np.sum(D**2, axis=1))[:, np.newaxis]  # 归一化字典矩阵
    # 根据分辨率和子采样计算波形字典中的成分数量
    n_components = resolution // subsampling

    # 计算一个多尺度小波字典
    D_multi = np.r_[
        tuple(
            # 生成一个 Ricker 小波矩阵，具有指定的宽度、分辨率和成分数量的五分之一
            ricker_matrix(
                width=w, resolution=resolution, n_components=n_components // 5
            )
            for w in (10, 50, 100, 500, 1000)
        )
    ]

    # 生成一个从 0 到 resolution-1 的等间隔数组 X
    X = np.linspace(0, resolution - 1, resolution)
    # 将 X 中小于分辨率四分之一的元素设为 3.0
    first_quarter = X < resolution / 4
    X[first_quarter] = 3.0
    # 将 X 中不在第一四分之一的元素设为 -1.0
    X[np.logical_not(first_quarter)] = -1.0
    # 将 X 调整为行向量形式
    X = X.reshape(1, -1)

    # 检查底层模型是否在不收敛时发出警告
    with pytest.warns(ConvergenceWarning):
        # 创建一个稀疏编码器模型，使用给定的小波字典和转换算法，最大迭代次数为 1
        model = SparseCoder(
            D_multi, transform_algorithm=transform_algorithm, transform_max_iter=1
        )
        # 对数据 X 进行拟合转换
        model.fit_transform(X)

    # 检查底层模型是否在收敛但没有警告的情况下运行
    with warnings.catch_warnings():
        # 设置警告过滤器，捕获 ConvergenceWarning 异常
        warnings.simplefilter("error", ConvergenceWarning)
        # 创建一个稀疏编码器模型，使用给定的小波字典和转换算法，最大迭代次数为 2000
        model = SparseCoder(
            D_multi, transform_algorithm=transform_algorithm, transform_max_iter=2000
        )
        # 对数据 X 进行拟合转换
        model.fit_transform(X)
# 测试用例：验证在 'lars' 编码方法中不支持正数约束的情况下抛出 ValueError 异常
def test_dict_learning_lars_positive_parameter():
    n_components = 5
    alpha = 1
    err_msg = "Positive constraint not supported for 'lars' coding method."
    # 使用 pytest 检查是否抛出 ValueError 异常，并验证异常信息与 err_msg 匹配
    with pytest.raises(ValueError, match=err_msg):
        dict_learning(X, n_components, alpha=alpha, positive_code=True)


# 参数化测试：验证不同的转换算法和正数编码/字典对的情况下的字典学习
@pytest.mark.parametrize(
    "transform_algorithm",
    [
        "lasso_lars",
        "lasso_cd",
        "threshold",
    ],
)
@pytest.mark.parametrize("positive_code", [False, True])
@pytest.mark.parametrize("positive_dict", [False, True])
def test_dict_learning_positivity(transform_algorithm, positive_code, positive_dict):
    n_components = 5
    # 创建字典学习对象并拟合数据集 X
    dico = DictionaryLearning(
        n_components,
        transform_algorithm=transform_algorithm,
        random_state=0,
        positive_code=positive_code,
        positive_dict=positive_dict,
        fit_algorithm="cd",
    ).fit(X)

    # 对学习到的字典进行编码
    code = dico.transform(X)
    # 根据 positive_dict 的取值验证学习到的字典元素符号性质
    if positive_dict:
        assert (dico.components_ >= 0).all()
    else:
        assert (dico.components_ < 0).any()
    # 根据 positive_code 的取值验证编码结果的符号性质
    if positive_code:
        assert (code >= 0).all()
    else:
        assert (code < 0).any()


# 参数化测试：验证在 'lars' 转换算法中不同的正数字典约束的情况
@pytest.mark.parametrize("positive_dict", [False, True])
def test_dict_learning_lars_dict_positivity(positive_dict):
    n_components = 5
    # 创建字典学习对象并拟合数据集 X，使用 'lars' 转换算法
    dico = DictionaryLearning(
        n_components,
        transform_algorithm="lars",
        random_state=0,
        positive_dict=positive_dict,
        fit_algorithm="cd",
    ).fit(X)

    # 根据 positive_dict 的取值验证学习到的字典元素符号性质
    if positive_dict:
        assert (dico.components_ >= 0).all()
    else:
        assert (dico.components_ < 0).any()


# 测试用例：验证在 'lars' 转换算法中不支持正数编码约束的情况下抛出 ValueError 异常
def test_dict_learning_lars_code_positivity():
    n_components = 5
    # 创建字典学习对象并拟合数据集 X，使用 'lars' 转换算法
    dico = DictionaryLearning(
        n_components,
        transform_algorithm="lars",
        random_state=0,
        positive_code=True,
        fit_algorithm="cd",
    ).fit(X)

    # 准备预期的 ValueError 异常信息
    err_msg = "Positive constraint not supported for '{}' coding method.".format("lars")
    # 使用 pytest 检查是否抛出 ValueError 异常，并验证异常信息与 err_msg 匹配
    with pytest.raises(ValueError, match=err_msg):
        dico.transform(X)


# 测试用例：验证字典学习后的重建准确性
def test_dict_learning_reconstruction():
    n_components = 12
    # 创建字典学习对象，使用 'omp' 转换算法和指定的参数
    dico = DictionaryLearning(
        n_components, transform_algorithm="omp", transform_alpha=0.001, random_state=0
    )
    # 拟合数据集 X 并进行编码
    code = dico.fit(X).transform(X)
    # 验证重建是否与原始数据 X 几乎相等
    assert_array_almost_equal(np.dot(code, dico.components_), X)

    # 修改转换算法为 'lasso_lars' 并重新编码
    dico.set_params(transform_algorithm="lasso_lars")
    code = dico.transform(X)
    # 再次验证重建是否与原始数据 X 几乎相等，精度为小数点后两位
    assert_array_almost_equal(np.dot(code, dico.components_), X, decimal=2)

    # 此处之前用于测试 'lars'，但无法保证非零原子的数量是正确的。


# 测试用例：验证并行重建是否在 n_jobs > 1 时正常工作
def test_dict_learning_reconstruction_parallel():
    n_components = 12
    # 创建字典学习对象，使用 'omp' 转换算法和指定的参数，并指定 n_jobs=4 进行并行处理
    dico = DictionaryLearning(
        n_components,
        transform_algorithm="omp",
        transform_alpha=0.001,
        random_state=0,
        n_jobs=4,
    )
    # 拟合数据集 X 并进行并行编码
    code = dico.fit(X).transform(X)
    # 验证并行重建结果是否与原始数据 X 几乎相等
    assert_array_almost_equal(np.dot(code, dico.components_), X)
    # 设置字典学习模型的参数，使用 Lasso Lars 算法作为变换算法
    dico.set_params(transform_algorithm="lasso_lars")
    # 对输入数据 X 进行字典学习的变换，得到编码后的结果
    code = dico.transform(X)
    # 使用近似相等的方式检查重构的数据是否与原始数据 X 接近，精确度为小数点后两位
    assert_array_almost_equal(np.dot(code, dico.components_), X, decimal=2)
# 测试使用 Lasso CD 变换算法和只读数据进行字典学习
def test_dict_learning_lassocd_readonly_data():
    n_components = 12
    # 使用 TempMemmap 类读取只读的 X 数据
    with TempMemmap(X) as X_read_only:
        # 创建 DictionaryLearning 对象，设置参数
        dico = DictionaryLearning(
            n_components,
            transform_algorithm="lasso_cd",  # 使用 Lasso CD 变换算法
            transform_alpha=0.001,
            random_state=0,
            n_jobs=4,
        )
        # 忽略收敛警告，拟合并转换 X_read_only
        with ignore_warnings(category=ConvergenceWarning):
            code = dico.fit(X_read_only).transform(X_read_only)
        # 断言转换后的代码乘以字典组件接近于 X_read_only，精度为小数点后两位
        assert_array_almost_equal(
            np.dot(code, dico.components_), X_read_only, decimal=2
        )


# 测试使用非零系数限制的字典学习
def test_dict_learning_nonzero_coefs():
    n_components = 4
    # 创建 DictionaryLearning 对象，设置参数
    dico = DictionaryLearning(
        n_components,
        transform_algorithm="lars",  # 使用 LARS 变换算法
        transform_n_nonzero_coefs=3,  # 设置非零系数的个数为 3
        random_state=0,
    )
    # 拟合并转换 X 的第二个样本，得到编码 code
    code = dico.fit(X).transform(X[np.newaxis, 1])
    # 断言 code 中非零元素的个数为 3
    assert len(np.flatnonzero(code)) == 3

    # 设置变换算法为 OMP
    dico.set_params(transform_algorithm="omp")
    # 再次转换 X 的第二个样本，得到新的 code
    code = dico.transform(X[np.newaxis, 1])
    # 断言新的 code 中非零元素的个数为 3
    assert len(np.flatnonzero(code)) == 3


# 测试使用阈值变换算法的字典学习
def test_dict_learning_split():
    n_components = 5
    # 创建 DictionaryLearning 对象，设置参数
    dico = DictionaryLearning(
        n_components, transform_algorithm="threshold", random_state=0
    )
    # 拟合并转换 X，得到编码 code
    code = dico.fit(X).transform(X)
    # 设置 split_sign 为 True
    dico.split_sign = True
    # 再次转换 X，得到拆分编码 split_code
    split_code = dico.transform(X)

    # 断言拆分编码 split_code 的前 n_components 列减去后 n_components 列接近于 code
    assert_array_almost_equal(
        split_code[:, :n_components] - split_code[:, n_components:], code
    )


# 测试在线字典学习的形状约束
def test_dict_learning_online_shapes():
    rng = np.random.RandomState(0)
    n_components = 8

    # 调用 dict_learning_online 函数进行在线字典学习，返回 code 和 dictionary
    code, dictionary = dict_learning_online(
        X,
        n_components=n_components,
        batch_size=4,
        max_iter=10,
        method="cd",
        random_state=rng,
        return_code=True,
    )
    # 断言 code 的形状为 (n_samples, n_components)
    assert code.shape == (n_samples, n_components)
    # 断言 dictionary 的形状为 (n_components, n_features)
    assert dictionary.shape == (n_components, n_features)
    # 断言 code 乘以 dictionary 的结果形状与 X 相同
    assert np.dot(code, dictionary).shape == X.shape

    # 调用 dict_learning_online 函数进行在线字典学习，返回 dictionary
    dictionary = dict_learning_online(
        X,
        n_components=n_components,
        batch_size=4,
        max_iter=10,
        method="cd",
        random_state=rng,
        return_code=False,
    )
    # 断言 dictionary 的形状为 (n_components, n_features)
    assert dictionary.shape == (n_components, n_features)


# 测试在线字典学习中 LARS 编码方法的正参数约束
def test_dict_learning_online_lars_positive_parameter():
    err_msg = "Positive constraint not supported for 'lars' coding method."
    # 断言调用 dict_learning_online 函数时使用 'lars' 方法并设定 positive_code=True 会引发 ValueError
    with pytest.raises(ValueError, match=err_msg):
        dict_learning_online(X, batch_size=4, max_iter=10, positive_code=True)


# 使用参数化测试检验小批量字典学习中正性约束的多种情况
@pytest.mark.parametrize(
    "transform_algorithm",
    [
        "lasso_lars",
        "lasso_cd",
        "threshold",
    ],
)
@pytest.mark.parametrize("positive_code", [False, True])
@pytest.mark.parametrize("positive_dict", [False, True])
def test_minibatch_dictionary_learning_positivity(
    transform_algorithm, positive_code, positive_dict
):
    n_components = 8
    # 使用MiniBatchDictionaryLearning类创建一个字典学习模型对象，通过对输入数据X进行拟合来训练这个模型
    dico = MiniBatchDictionaryLearning(
        n_components,                     # 字典中要学习的原子（atom）数量
        batch_size=4,                     # 每个迭代步骤中用于拟合的样本批量大小
        max_iter=10,                      # 拟合过程中允许的最大迭代次数
        transform_algorithm=transform_algorithm,  # 字典更新算法的选择
        random_state=0,                   # 随机数生成器的种子，用于重现结果
        positive_code=positive_code,      # 是否要求学习到的编码是非负的
        positive_dict=positive_dict,      # 是否要求学习到的字典（components）是非负的
        fit_algorithm="cd",               # 拟合算法的选择
    ).fit(X)                              # 对数据X进行拟合，返回拟合后的字典学习模型对象
    
    # 使用训练好的字典学习模型对象dico，对输入数据X进行变换，得到编码code
    code = dico.transform(X)
    
    # 如果要求学习到的字典是非负的，则断言所有学习到的字典元素都大于等于0
    if positive_dict:
        assert (dico.components_ >= 0).all()
    # 否则，断言至少存在一个学习到的字典元素小于0
    else:
        assert (dico.components_ < 0).any()
    
    # 如果要求学习到的编码是非负的，则断言所有学习到的编码元素都大于等于0
    if positive_code:
        assert (code >= 0).all()
    # 否则，断言至少存在一个学习到的编码元素小于0
    else:
        assert (code < 0).any()
# 使用 pytest 标记此函数为测试函数，并参数化 positive_dict 为 False 和 True 两种情况
@pytest.mark.parametrize("positive_dict", [False, True])
def test_minibatch_dictionary_learning_lars(positive_dict):
    # 设定字典学习的组件数为 8
    n_components = 8

    # 使用 MiniBatchDictionaryLearning 进行字典学习，采用 LARS 变换算法
    dico = MiniBatchDictionaryLearning(
        n_components,
        batch_size=4,
        max_iter=10,
        transform_algorithm="lars",
        random_state=0,
        positive_dict=positive_dict,  # 是否使用正字典的参数
        fit_algorithm="cd",  # 拟合算法选择为 coordinate descent
    ).fit(X)

    # 如果 positive_dict 为 True，则确保所有学习到的字典元素非负
    if positive_dict:
        assert (dico.components_ >= 0).all()
    else:
        # 如果 positive_dict 为 False，则至少有一个学习到的字典元素为负数
        assert (dico.components_ < 0).any()


# 使用 pytest 参数化 positive_code 和 positive_dict 为 False 和 True 两种情况
@pytest.mark.parametrize("positive_code", [False, True])
@pytest.mark.parametrize("positive_dict", [False, True])
def test_dict_learning_online_positivity(positive_code, positive_dict):
    # 设定随机种子
    rng = np.random.RandomState(0)
    # 设定字典学习的组件数为 8
    n_components = 8

    # 调用 dict_learning_online 函数进行在线字典学习
    code, dictionary = dict_learning_online(
        X,
        n_components=n_components,
        batch_size=4,
        method="cd",
        alpha=1,
        random_state=rng,
        positive_dict=positive_dict,  # 是否使用正字典的参数
        positive_code=positive_code,  # 是否使用正编码的参数
    )

    # 如果 positive_dict 为 True，则确保所有学习到的字典元素非负
    if positive_dict:
        assert (dictionary >= 0).all()
    else:
        # 如果 positive_dict 为 False，则至少有一个学习到的字典元素为负数
        assert (dictionary < 0).any()

    # 如果 positive_code 为 True，则确保所有学习到的编码值非负
    if positive_code:
        assert (code >= 0).all()
    else:
        # 如果 positive_code 为 False，则至少有一个学习到的编码值为负数
        assert (code < 0).any()


# 测试字典学习在线 API 的详细输出
def test_dict_learning_online_verbosity():
    # 测试更好的覆盖率
    n_components = 5
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()

        # 设置迭代停止的容忍度为 0.1 的 MiniBatchDictionaryLearning
        dico = MiniBatchDictionaryLearning(
            n_components, batch_size=4, max_iter=5, verbose=1, tol=0.1, random_state=0
        )
        dico.fit(X)

        # 设置最大不改进迭代次数为 2 的 MiniBatchDictionaryLearning
        dico = MiniBatchDictionaryLearning(
            n_components,
            batch_size=4,
            max_iter=5,
            verbose=1,
            max_no_improvement=2,
            random_state=0,
        )
        dico.fit(X)

        # 设置详细程度为 2 的 MiniBatchDictionaryLearning
        dico = MiniBatchDictionaryLearning(
            n_components, batch_size=4, max_iter=5, verbose=2, random_state=0
        )
        dico.fit(X)

        # 设置在线学习字典的详细程度为 1
        dict_learning_online(
            X,
            n_components=n_components,
            batch_size=4,
            alpha=1,
            verbose=1,
            random_state=0,
        )

        # 设置在线学习字典的详细程度为 2
        dict_learning_online(
            X,
            n_components=n_components,
            batch_size=4,
            alpha=1,
            verbose=2,
            random_state=0,
        )
    finally:
        sys.stdout = old_stdout

    # 确保 dico 的组件形状为 (n_components, n_features)
    assert dico.components_.shape == (n_components, n_features)


# 测试在线学习字典的估算器形状
def test_dict_learning_online_estimator_shapes():
    # 设定字典学习的组件数为 5
    n_components = 5
    # 使用 MiniBatchDictionaryLearning 进行字典学习
    dico = MiniBatchDictionaryLearning(
        n_components, batch_size=4, max_iter=5, random_state=0
    )
    dico.fit(X)
    # 确保学习到的字典的形状为 (n_components, n_features)
    assert dico.components_.shape == (n_components, n_features)


# 测试超完备在线学习字典
def test_dict_learning_online_overcomplete():
    # 设定字典学习的组件数为 12
    n_components = 12
    # 使用 MiniBatchDictionaryLearning 类初始化字典学习器对象，设定参数如下：
    # - n_components: 字典的组件数量
    # - batch_size: 每个小批量的样本数量为 4
    # - max_iter: 最大迭代次数为 5
    # - random_state: 随机数生成器的种子为 0
    dico = MiniBatchDictionaryLearning(
        n_components, batch_size=4, max_iter=5, random_state=0
    ).fit(X)
    
    # 断言条件，确保学习器学习后的字典成分的形状符合预期
    assert dico.components_.shape == (n_components, n_features)
# 定义测试函数，用于在线学习初始化字典
def test_dict_learning_online_initialization():
    # 设定字典中的成分数量
    n_components = 12
    # 使用随机数生成器创建随机状态
    rng = np.random.RandomState(0)
    # 从正态分布中生成形状为 (n_components, n_features) 的随机矩阵 V
    V = rng.randn(n_components, n_features)
    # 使用给定的初始化字典 V，以只读模式初始化在线字典学习器
    dico = MiniBatchDictionaryLearning(
        n_components, batch_size=4, max_iter=0, dict_init=V, random_state=0
    ).fit(X)
    # 断言学习得到的字典与初始化的 V 相等
    assert_array_equal(dico.components_, V)


# 定义测试函数，测试在线学习只读初始化字典
def test_dict_learning_online_readonly_initialization():
    # 设定字典中的成分数量
    n_components = 12
    # 使用随机数生成器创建随机状态
    rng = np.random.RandomState(0)
    # 从正态分布中生成形状为 (n_components, n_features) 的随机矩阵 V
    V = rng.randn(n_components, n_features)
    # 将 V 设置为只读状态
    V.setflags(write=False)
    # 初始化在线字典学习器，使用只读的 V 进行初始化
    MiniBatchDictionaryLearning(
        n_components,
        batch_size=4,
        max_iter=1,
        dict_init=V,
        random_state=0,
        shuffle=False,
    ).fit(X)


# 定义测试函数，测试在线学习的部分拟合功能
def test_dict_learning_online_partial_fit():
    # 设定字典中的成分数量
    n_components = 12
    # 使用随机数生成器创建随机状态
    rng = np.random.RandomState(0)
    # 从正态分布中生成形状为 (n_components, n_features) 的随机矩阵 V 进行随机初始化
    V = rng.randn(n_components, n_features)  # random init
    # 对 V 进行归一化处理，使得每个成分的平方和为 1
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    # 使用给定的 V 进行在线字典学习，设定相关参数
    dict1 = MiniBatchDictionaryLearning(
        n_components,
        max_iter=10,
        batch_size=1,
        alpha=1,
        shuffle=False,
        dict_init=V,
        max_no_improvement=None,
        tol=0.0,
        random_state=0,
    ).fit(X)
    # 初始化另一个在线字典学习器，使用相同的 V 进行初始化
    dict2 = MiniBatchDictionaryLearning(
        n_components, alpha=1, dict_init=V, random_state=0
    )
    # 通过部分拟合对 dict2 进行学习，重复 10 次
    for i in range(10):
        for sample in X:
            dict2.partial_fit(sample[np.newaxis, :])

    # 断言使用两种方式得到的字典编码结果不全为零
    assert not np.all(sparse_encode(X, dict1.components_, alpha=1) == 0)
    # 断言两个字典的成分在一定的精度下相等
    assert_array_almost_equal(dict1.components_, dict2.components_, decimal=2)

    # 额外断言部分拟合应该忽略 max_iter 参数 (#17433)
    assert dict1.n_steps_ == dict2.n_steps_ == 100


# 定义测试函数，测试稀疏编码的输出形状是否正确
def test_sparse_encode_shapes():
    # 设定字典中的成分数量
    n_components = 12
    # 使用随机数生成器创建随机状态
    rng = np.random.RandomState(0)
    # 从正态分布中生成形状为 (n_components, n_features) 的随机矩阵 V 进行随机初始化
    V = rng.randn(n_components, n_features)  # random init
    # 对 V 进行归一化处理，使得每个成分的平方和为 1
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    # 针对不同的算法（"lasso_lars", "lasso_cd", "lars", "omp", "threshold"），进行稀疏编码
    for algo in ("lasso_lars", "lasso_cd", "lars", "omp", "threshold"):
        # 调用稀疏编码函数，计算编码结果 code 的形状是否符合预期
        code = sparse_encode(X, V, algorithm=algo)
        assert code.shape == (n_samples, n_components)


# 使用参数化测试，测试稀疏编码是否能正确处理正性约束
@pytest.mark.parametrize("algo", ["lasso_lars", "lasso_cd", "threshold"])
@pytest.mark.parametrize("positive", [False, True])
def test_sparse_encode_positivity(algo, positive):
    # 设定字典中的成分数量
    n_components = 12
    # 使用随机数生成器创建随机状态
    rng = np.random.RandomState(0)
    # 从正态分布中生成形状为 (n_components, n_features) 的随机矩阵 V 进行随机初始化
    V = rng.randn(n_components, n_features)  # random init
    # 对 V 进行归一化处理，使得每个成分的平方和为 1
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    # 调用稀疏编码函数，测试是否能正确处理正性约束
    code = sparse_encode(X, V, algorithm=algo, positive=positive)
    if positive:
        assert (code >= 0).all()
    else:
        assert (code < 0).any()


# 使用参数化测试，测试当正性约束对于某些算法不可用时是否引发异常
@pytest.mark.parametrize("algo", ["lars", "omp"])
def test_sparse_encode_unavailable_positivity(algo):
    # 设定字典中的成分数量
    n_components = 12
    # 使用随机数生成器创建随机状态
    rng = np.random.RandomState(0)
    # 从正态分布中生成形状为 (n_components, n_features) 的随机矩阵 V 进行随机初始化
    V = rng.randn(n_components, n_features)  # random init
    # 对 V 进行归一化处理，使得每个成分的平方和为 1
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    # 准备错误信息，说明对于特定编码方法算法不支持正性约束
    err_msg = "Positive constraint not supported for '{}' coding method."
    err_msg = err_msg.format(algo)
    # 使用 pytest 的异常断言检测是否会引发预期的 ValueError 异常
    with pytest.raises(ValueError, match=err_msg):
        sparse_encode(X, V, algorithm=algo, positive=True)
def test_sparse_encode_input():
    # 设置稀疏编码所需的组件数
    n_components = 100
    # 创建随机数生成器对象，并指定种子以确保可重复性
    rng = np.random.RandomState(0)
    # 从正态分布中生成随机的 V 矩阵，形状为 (n_components, n_features)，用于稀疏编码
    V = rng.randn(n_components, n_features)  # random init
    # 对 V 进行归一化，使得每行的平方和为1
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    # 按 Fortran（列主序）顺序检查并转换输入数组 Xf
    Xf = check_array(X, order="F")
    # 对于给定的算法列表中的每个算法，执行稀疏编码
    for algo in ("lasso_lars", "lasso_cd", "lars", "omp", "threshold"):
        # 对输入 X 使用 V 进行稀疏编码，使用指定的算法
        a = sparse_encode(X, V, algorithm=algo)
        # 对 Fortran 顺序的输入 Xf 使用 V 进行稀疏编码，使用相同的算法
        b = sparse_encode(Xf, V, algorithm=algo)
        # 断言两种方法得到的稀疏编码结果近似相等
        assert_array_almost_equal(a, b)


def test_sparse_encode_error():
    # 设置稀疏编码所需的组件数
    n_components = 12
    # 创建随机数生成器对象，并指定种子以确保可重复性
    rng = np.random.RandomState(0)
    # 从正态分布中生成随机的 V 矩阵，形状为 (n_components, n_features)，用于稀疏编码
    V = rng.randn(n_components, n_features)  # random init
    # 对 V 进行归一化，使得每行的平方和为1
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    # 对输入数据 X 使用 V 进行稀疏编码，指定 alpha 参数为 0.001
    code = sparse_encode(X, V, alpha=0.001)
    # 断言稀疏编码结果中并非所有元素都为 0
    assert not np.all(code == 0)
    # 断言稀疏编码后的重建误差小于 0.1 的平方根
    assert np.sqrt(np.sum((np.dot(code, V) - X) ** 2)) < 0.1


def test_sparse_encode_error_default_sparsity():
    # 创建随机数生成器对象，并指定种子以确保可重复性
    rng = np.random.RandomState(0)
    # 从正态分布中生成随机的输入数据 X，形状为 (100, 64)
    X = rng.randn(100, 64)
    # 从正态分布中生成随机的字典 D，形状为 (2, 64)
    D = rng.randn(2, 64)
    # 使用 OMP 算法对输入数据 X 使用字典 D 进行稀疏编码，不限定非零系数的数量
    code = ignore_warnings(sparse_encode)(X, D, algorithm="omp", n_nonzero_coefs=None)
    # 断言稀疏编码结果的形状为 (100, 2)
    assert code.shape == (100, 2)


def test_sparse_coder_estimator():
    # 设置稀疏编码所需的组件数
    n_components = 12
    # 创建随机数生成器对象，并指定种子以确保可重复性
    rng = np.random.RandomState(0)
    # 从正态分布中生成随机的 V 矩阵，形状为 (n_components, n_features)，用于稀疏编码
    V = rng.randn(n_components, n_features)  # random init
    # 对 V 进行归一化，使得每行的平方和为1
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    # 使用 Lasso Lars 算法对输入数据 X 进行稀疏编码，指定 alpha 参数为 0.001
    coder = SparseCoder(
        dictionary=V, transform_algorithm="lasso_lars", transform_alpha=0.001
    ).transform(X)
    # 断言稀疏编码结果中并非所有元素都为 0
    assert not np.all(coder == 0)
    # 断言稀疏编码后的重建误差小于 0.1 的平方根
    assert np.sqrt(np.sum((np.dot(coder, V) - X) ** 2)) < 0.1


def test_sparse_coder_estimator_clone():
    # 设置稀疏编码所需的组件数
    n_components = 12
    # 创建随机数生成器对象，并指定种子以确保可重复性
    rng = np.random.RandomState(0)
    # 从正态分布中生成随机的 V 矩阵，形状为 (n_components, n_features)，用于稀疏编码
    V = rng.randn(n_components, n_features)  # random init
    # 对 V 进行归一化，使得每行的平方和为1
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    # 使用 Lasso Lars 算法对输入数据 X 进行稀疏编码，指定 alpha 参数为 0.001
    coder = SparseCoder(
        dictionary=V, transform_algorithm="lasso_lars", transform_alpha=0.001
    )
    # 克隆稀疏编码器对象
    cloned = clone(coder)
    # 断言克隆对象与原始对象不是同一个对象
    assert id(cloned) != id(coder)
    # 断言克隆对象的字典与原始对象的字典近似相等
    np.testing.assert_allclose(cloned.dictionary, coder.dictionary)
    # 断言克隆对象的组件数与原始对象的组件数相等
    assert cloned.n_components_ == coder.n_components_
    # 断言克隆对象的输入特征数与原始对象的输入特征数相等
    assert cloned.n_features_in_ == coder.n_features_in_
    # 生成随机的数据，使用克隆对象和原始对象进行稀疏编码，然后断言它们的结果近似相等
    data = np.random.rand(n_samples, n_features).astype(np.float32)
    np.testing.assert_allclose(cloned.transform(data), coder.transform(data))


def test_sparse_coder_parallel_mmap():
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/5956
    # Test that SparseCoder does not error by passing reading only
    # arrays to child processes

    # 创建随机数生成器对象，并指定种子以确保可重复性
    rng = np.random.RandomState(777)
    # 设置稀疏编码所需的组件数和特征数
    n_components, n_features = 40, 64
    # 从均匀分布中生成随机的初始字典，形状为 (n_components, n_features)
    init_dict = rng.rand(n_components, n_features)
    # 确保数据量大于2M。Joblib 将大于1MB的数组作为内存映射文件处理。
    n_samples = int(2e6) // (4 * n_features)
    # 从均匀分布中生成随机的数据，形状为 (n_samples, n_features)，数据类型为 float32
    data = np.random.rand(n_samples, n_features).astype(np.float32)

    # 创建 SparseCoder 对象，使用 OMP 算法，并指定并行处理的进程数为 2
    sc = SparseCoder(init_dict, transform_algorithm="omp", n_jobs=2)
    # 对数据进行拟合和转换
    sc.fit_transform(data)


def test_sparse_coder_common_transformer():
    # 创建随机数生成器对象，并指定种子以确保可重复
    # 定义变量 n_components 和 n_features，分别为稀疏编码器初始化字典的行数和列数
    n_components, n_features = 40, 3
    
    # 使用随机数生成器 rng 创建一个 n_components 行、n_features 列的随机矩阵作为稀疏编码器的初始字典
    init_dict = rng.rand(n_components, n_features)
    
    # 创建稀疏编码器对象 sc，使用上一步生成的随机矩阵作为初始化字典
    sc = SparseCoder(init_dict)
    
    # 调用函数 check_transformer_data_not_an_array，检查 sc 类的实例是否符合数据不是数组的标准
    check_transformer_data_not_an_array(sc.__class__.__name__, sc)
    
    # 调用函数 check_transformer_general，检查 sc 类的实例是否符合一般变换器的标准
    check_transformer_general(sc.__class__.__name__, sc)
    
    # 创建局部函数 check_transformer_general_memmap，用于检查只读内存映射的一般变换器的标准
    check_transformer_general_memmap = partial(
        check_transformer_general, readonly_memmap=True
    )
    
    # 调用 check_transformer_general_memmap 函数，检查 sc 类的实例是否符合只读内存映射的一般变换器的标准
    check_transformer_general_memmap(sc.__class__.__name__, sc)
    
    # 调用函数 check_transformers_unfitted，检查 sc 类的实例是否符合未拟合变换器的标准
    check_transformers_unfitted(sc.__class__.__name__, sc)
def test_sparse_coder_n_features_in():
    # 创建一个包含两行三列的 NumPy 数组作为测试数据
    d = np.array([[1, 2, 3], [1, 2, 3]])
    # 使用 SparseCoder 类初始化一个稀疏编码器对象
    sc = SparseCoder(d)
    # 断言稀疏编码器对象的输入特征数量等于数组 d 的列数
    assert sc.n_features_in_ == d.shape[1]


def test_update_dict():
    # 检查批处理模式和在线模式下字典更新的差异
    # 非回归测试用例，用于测试问题 #4866
    rng = np.random.RandomState(0)

    # 初始化编码矩阵和字典矩阵
    code = np.array([[0.5, -0.5], [0.1, 0.9]])
    dictionary = np.array([[1.0, 0.0], [0.6, 0.8]])

    # 构造模拟数据 X，用于更新字典
    X = np.dot(code, dictionary) + rng.randn(2, 2)

    # 复制字典以进行完全批处理更新
    newd_batch = dictionary.copy()
    _update_dict(newd_batch, X, code)

    # 在线更新方式
    A = np.dot(code.T, code)
    B = np.dot(X.T, code)
    newd_online = dictionary.copy()
    _update_dict(newd_online, X, code, A, B)

    # 断言完全批处理更新和在线更新得到的字典结果近似相等
    assert_allclose(newd_batch, newd_online)


@pytest.mark.parametrize(
    "algorithm", ("lasso_lars", "lasso_cd", "lars", "threshold", "omp")
)
@pytest.mark.parametrize("data_type", (np.float32, np.float64))
# 注意：不检查整数输入，因为 `lasso_lars` 和 `lars` 在 `_lars_path_solver` 中会引发 `ValueError`
def test_sparse_encode_dtype_match(data_type, algorithm):
    # 初始化随机数生成器和字典矩阵
    n_components = 6
    rng = np.random.RandomState(0)
    dictionary = rng.randn(n_components, n_features)

    # 调用 sparse_encode 函数进行稀疏编码
    code = sparse_encode(
        X.astype(data_type), dictionary.astype(data_type), algorithm=algorithm
    )

    # 断言编码结果的数据类型与输入数据类型一致
    assert code.dtype == data_type


@pytest.mark.parametrize(
    "algorithm", ("lasso_lars", "lasso_cd", "lars", "threshold", "omp")
)
def test_sparse_encode_numerical_consistency(algorithm):
    # 验证 np.float32 和 np.float64 之间的数值一致性
    rtol = 1e-4
    n_components = 6
    rng = np.random.RandomState(0)
    dictionary = rng.randn(n_components, n_features)

    # 分别对 np.float32 和 np.float64 类型的数据进行稀疏编码
    code_32 = sparse_encode(
        X.astype(np.float32), dictionary.astype(np.float32), algorithm=algorithm
    )
    code_64 = sparse_encode(
        X.astype(np.float64), dictionary.astype(np.float64), algorithm=algorithm
    )

    # 断言两种数据类型编码结果的数值近似性
    assert_allclose(code_32, code_64, rtol=rtol)


@pytest.mark.parametrize(
    "transform_algorithm", ("lasso_lars", "lasso_cd", "lars", "threshold", "omp")
)
@pytest.mark.parametrize("data_type", (np.float32, np.float64))
# 注意：不检查整数输入，因为 `lasso_lars` 和 `lars` 在 `_lars_path_solver` 中会引发 `ValueError`
def test_sparse_coder_dtype_match(data_type, transform_algorithm):
    # 验证稀疏编码器中变换过程中数据类型的一致性
    n_components = 6
    rng = np.random.RandomState(0)
    dictionary = rng.randn(n_components, n_features)

    # 初始化稀疏编码器对象
    coder = SparseCoder(
        dictionary.astype(data_type), transform_algorithm=transform_algorithm
    )

    # 对输入数据 X 进行数据类型匹配的稀疏编码
    code = coder.transform(X.astype(data_type))

    # 断言编码结果的数据类型与输入数据类型一致
    assert code.dtype == data_type


@pytest.mark.parametrize("fit_algorithm", ("lars", "cd"))
@pytest.mark.parametrize(
    "transform_algorithm", ("lasso_lars", "lasso_cd", "lars", "threshold", "omp")
)
@pytest.mark.parametrize(
    "data_type, expected_type",
    (
        # 定义了一个元组，每个元组包含两个元素，第一个元素是 np.float32 类型，第二个元素是 np.float32 类型
        (np.float32, np.float32),
        # 定义了一个元组，每个元组包含两个元素，第一个元素是 np.float64 类型，第二个元素是 np.float64 类型
        (np.float64, np.float64),
        # 定义了一个元组，每个元组包含两个元素，第一个元素是 np.int32 类型，第二个元素是 np.float64 类型
        (np.int32, np.float64),
        # 定义了一个元组，每个元组包含两个元素，第一个元素是 np.int64 类型，第二个元素是 np.float64 类型
        (np.int64, np.float64),
    ),
def test_dictionary_learning_dtype_match(
    data_type,
    expected_type,
    fit_algorithm,
    transform_algorithm,
):
    # 验证在字典学习类中对拟合和变换过程中的数据类型保持一致性

    # 创建一个 DictionaryLearning 实例，设置参数并初始化
    dict_learner = DictionaryLearning(
        n_components=8,
        fit_algorithm=fit_algorithm,
        transform_algorithm=transform_algorithm,
        random_state=0,
    )
    
    # 使用给定的数据类型将输入数据 X 转换为指定类型，并拟合到 dict_learner 中
    dict_learner.fit(X.astype(data_type))
    
    # 断言拟合后的成分(components)的数据类型与预期的类型相同
    assert dict_learner.components_.dtype == expected_type
    
    # 断言使用给定数据类型转换后的数据经过变换后的数据类型与预期类型相同
    assert dict_learner.transform(X.astype(data_type)).dtype == expected_type


@pytest.mark.parametrize("fit_algorithm", ("lars", "cd"))
@pytest.mark.parametrize(
    "transform_algorithm", ("lasso_lars", "lasso_cd", "lars", "threshold", "omp")
)
@pytest.mark.parametrize(
    "data_type, expected_type",
    (
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ),
)
def test_minibatch_dictionary_learning_dtype_match(
    data_type,
    expected_type,
    fit_algorithm,
    transform_algorithm,
):
    # 验证在小批量字典学习中对拟合和变换过程中的数据类型保持一致性

    # 创建一个 MiniBatchDictionaryLearning 实例，设置参数并初始化
    dict_learner = MiniBatchDictionaryLearning(
        n_components=8,
        batch_size=10,
        fit_algorithm=fit_algorithm,
        transform_algorithm=transform_algorithm,
        max_iter=100,
        tol=1e-1,
        random_state=0,
    )
    
    # 使用给定的数据类型将输入数据 X 转换为指定类型，并拟合到 dict_learner 中
    dict_learner.fit(X.astype(data_type))

    # 断言拟合后的成分(components)的数据类型与预期的类型相同
    assert dict_learner.components_.dtype == expected_type
    
    # 断言使用给定数据类型转换后的数据经过变换后的数据类型与预期类型相同
    assert dict_learner.transform(X.astype(data_type)).dtype == expected_type
    
    # 断言内部变量 _A 和 _B 的数据类型与预期类型相同
    assert dict_learner._A.dtype == expected_type
    assert dict_learner._B.dtype == expected_type


@pytest.mark.parametrize("method", ("lars", "cd"))
@pytest.mark.parametrize(
    "data_type, expected_type",
    (
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ),
)
def test_dict_learning_dtype_match(data_type, expected_type, method):
    # 验证在字典学习过程中输出矩阵的数据类型保持一致性

    # 创建一个随机数生成器
    rng = np.random.RandomState(0)
    n_components = 8
    
    # 调用 dict_learning 函数进行字典学习，使用给定的数据类型转换 X，并设置参数
    code, dictionary, _ = dict_learning(
        X.astype(data_type),
        n_components=n_components,
        alpha=1,
        random_state=rng,
        method=method,
    )
    
    # 断言输出矩阵 code 的数据类型与预期的类型相同
    assert code.dtype == expected_type
    
    # 断言输出矩阵 dictionary 的数据类型与预期的类型相同
    assert dictionary.dtype == expected_type


@pytest.mark.parametrize("method", ("lars", "cd"))
def test_dict_learning_numerical_consistency(method):
    # 验证在 np.float32 和 np.float64 之间字典学习的数值一致性

    rtol = 1e-6
    n_components = 4
    alpha = 2

    # 使用 np.float64 类型进行字典学习，并设置参数
    U_64, V_64, _ = dict_learning(
        X.astype(np.float64),
        n_components=n_components,
        alpha=alpha,
        random_state=0,
        method=method,
    )
    
    # 使用 np.float32 类型进行字典学习，并设置参数
    U_32, V_32, _ = dict_learning(
        X.astype(np.float32),
        n_components=n_components,
        alpha=alpha,
        random_state=0,
        method=method,
    )

    # Optimal solution (U*, V*) is not unique.
    # 如果 (U*, V*) 是最优解，那么 (-U*, -V*) 也是最优解，
    # 并且 (U* 经过列置换, V* 经过行置换) 也是可选的，
    # 只要保持 UV 乘积不变。
    # 因此这里验证 UV 乘积，||U||_1,1 范数以及 sum(||V_k||_2^2) 范数
    # 而不是直接比较 U 和 V。

    # 验证 UV 乘积是否在指定的相对误差范围内接近
    assert_allclose(np.matmul(U_64, V_64), np.matmul(U_32, V_32), rtol=rtol)
    
    # 验证 ||U||_1,1 范数是否在指定的相对误差范围内接近
    assert_allclose(np.sum(np.abs(U_64)), np.sum(np.abs(U_32)), rtol=rtol)
    
    # 验证 sum(||V_k||_2^2) 范数是否在指定的相对误差范围内接近
    assert_allclose(np.sum(V_64**2), np.sum(V_32**2), rtol=rtol)
    
    # 验证得到的解是否不是退化的
    assert np.mean(U_64 != 0.0) > 0.05
    
    # 验证 U_64 和 U_32 中非零元素的数量是否相等
    assert np.count_nonzero(U_64 != 0.0) == np.count_nonzero(U_32 != 0.0)
# 使用 pytest 的装饰器标记该测试函数为参数化测试，参数为 "lars" 和 "cd"
@pytest.mark.parametrize("method", ("lars", "cd"))
# 同时参数化测试函数的输入参数 data_type 和 expected_type
@pytest.mark.parametrize(
    "data_type, expected_type",
    (
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ),
)
# 定义测试函数 test_dict_learning_online_dtype_match，验证输出矩阵的数据类型匹配性
def test_dict_learning_online_dtype_match(data_type, expected_type, method):
    # 初始化随机数生成器
    rng = np.random.RandomState(0)
    # 设置字典学习在线函数的参数，并获取返回的 code 和 dictionary
    code, dictionary = dict_learning_online(
        X.astype(data_type),
        n_components=8,
        alpha=1,
        batch_size=10,
        random_state=rng,
        method=method,
    )
    # 断言 code 的数据类型与期望的数据类型相同
    assert code.dtype == expected_type
    # 断言 dictionary 的数据类型与期望的数据类型相同

    assert dictionary.dtype == expected_type


# 使用 pytest 的装饰器标记该测试函数为参数化测试，参数为 "lars" 和 "cd"
@pytest.mark.parametrize("method", ("lars", "cd"))
# 定义测试函数 test_dict_learning_online_numerical_consistency，验证 np.float32 和 np.float64 之间的数值一致性
def test_dict_learning_online_numerical_consistency(method):
    # 设置数值误差容忍度
    rtol = 1e-4
    n_components = 4
    alpha = 1

    # 使用 np.float64 类型运行字典学习在线函数，获取 U_64 和 V_64
    U_64, V_64 = dict_learning_online(
        X.astype(np.float64),
        n_components=n_components,
        max_iter=1_000,
        alpha=alpha,
        batch_size=10,
        random_state=0,
        method=method,
        tol=0.0,
        max_no_improvement=None,
    )
    # 使用 np.float32 类型运行字典学习在线函数，获取 U_32 和 V_32
    U_32, V_32 = dict_learning_online(
        X.astype(np.float32),
        n_components=n_components,
        max_iter=1_000,
        alpha=alpha,
        batch_size=10,
        random_state=0,
        method=method,
        tol=0.0,
        max_no_improvement=None,
    )

    # 对比 U_64 和 U_32 乘积、绝对值之和、平方之和的数值，验证其数值一致性
    assert_allclose(np.matmul(U_64, V_64), np.matmul(U_32, V_32), rtol=rtol)
    assert_allclose(np.sum(np.abs(U_64)), np.sum(np.abs(U_32)), rtol=rtol)
    assert_allclose(np.sum(V_64**2), np.sum(V_32**2), rtol=rtol)
    # 验证 U_64 不为零的平均比例大于 0.05
    assert np.mean(U_64 != 0.0) > 0.05
    # 验证 U_64 和 U_32 非零元素的数量相等
    assert np.count_nonzero(U_64 != 0.0) == np.count_nonzero(U_32 != 0.0)


# 参数化测试函数，参数为三个不同的字典学习估计器
@pytest.mark.parametrize(
    "estimator",
    [
        SparseCoder(X.T),
        DictionaryLearning(),
        MiniBatchDictionaryLearning(batch_size=4, max_iter=10),
    ],
    # 使用 lambda 函数设置测试函数的标识符为估计器的类名
    ids=lambda x: x.__class__.__name__,
)
# 定义测试函数 test_get_feature_names_out，验证字典学习估计器的特征名输出
def test_get_feature_names_out(estimator):
    # 对估计器进行拟合
    estimator.fit(X)
    # 获取输入数据 X 的特征数
    n_components = X.shape[1]

    # 获取估计器输出的特征名列表
    feature_names_out = estimator.get_feature_names_out()
    # 获取估计器的类名并转换为小写
    estimator_name = estimator.__class__.__name__.lower()
    # 断言估计器输出的特征名列表与预期的列表相等
    assert_array_equal(
        feature_names_out,
        [f"{estimator_name}{i}" for i in range(n_components)],
    )


# 定义测试函数 test_cd_work_on_joblib_memmapped_data，验证 cd 方法能够在 joblib 的内存映射数据上正常工作
def test_cd_work_on_joblib_memmapped_data(monkeypatch):
    # 使用 monkeypatch 替换 sklearn.decomposition._dict_learning 中的 "Parallel" 属性
    monkeypatch.setattr(
        sklearn.decomposition._dict_learning,
        "Parallel",
        partial(Parallel, max_nbytes=100),
    )
    )

    # 使用 NumPy 中的随机数生成器创建一个 10x10 的随机数组 X_train
    rng = np.random.RandomState(0)
    X_train = rng.randn(10, 10)

    # 创建一个 DictionaryLearning 对象 dict_learner，并设置参数
    dict_learner = DictionaryLearning(
        n_components=5,          # 设置字典中原子的数量为 5
        random_state=0,          # 设定随机数种子为 0，以便结果可重复
        n_jobs=2,                # 并行工作的线程数为 2
        fit_algorithm="cd",      # 使用坐标下降（Coordinate Descent）算法进行拟合
        max_iter=50,             # 最大迭代次数为 50
        verbose=True,            # 输出详细的拟合过程信息
    )

    # 运行拟合过程，确保能够无错误完成
    dict_learner.fit(X_train)
# TODO(1.6): remove in 1.6
# 定义名为 test_xxx 的函数，该函数可能在版本 1.6 中被移除
def test_xxx():
    # 设置警告消息，提醒 `max_iter=None` 在版本 1.4 中已被弃用并将被移除
    warn_msg = "`max_iter=None` is deprecated in version 1.4 and will be removed"
    
    # 使用 pytest 模块捕获 FutureWarning 类型的警告，且匹配特定的警告消息
    with pytest.warns(FutureWarning, match=warn_msg):
        # 使用 MiniBatchDictionaryLearning 类进行模型拟合，其中 max_iter 参数被设置为 None
        MiniBatchDictionaryLearning(max_iter=None, random_state=0).fit(X)
    
    # 再次使用 pytest 模块捕获 FutureWarning 类型的警告，且匹配特定的警告消息
    with pytest.warns(FutureWarning, match=warn_msg):
        # 调用 dict_learning_online 函数，其中 max_iter 参数被设置为 None
        dict_learning_online(X, max_iter=None, random_state=0)
```