# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\test_nca.py`

```
"""
Testing for Neighborhood Component Analysis module (sklearn.neighbors.nca)
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试
from numpy.testing import assert_array_almost_equal, assert_array_equal  # 导入NumPy的测试模块
from scipy.optimize import check_grad  # 导入SciPy的优化模块

from sklearn import clone  # 导入sklearn中的克隆函数
from sklearn.datasets import load_iris, make_blobs, make_classification  # 导入数据集生成函数
from sklearn.exceptions import ConvergenceWarning  # 导入收敛警告异常
from sklearn.metrics import pairwise_distances  # 导入用于计算成对距离的函数
from sklearn.neighbors import NeighborhoodComponentsAnalysis  # 导入邻域成分分析模块
from sklearn.preprocessing import LabelEncoder  # 导入标签编码器
from sklearn.utils import check_random_state  # 导入检查随机状态的函数

rng = check_random_state(0)  # 检查并设置随机状态对象

# 加载并随机排列鸢尾花数据集
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris_data = iris.data[perm]
iris_target = iris.target[perm]
EPS = np.finfo(float).eps  # 获取浮点数的机器精度

def test_simple_example():
    """Test on a simple example.

    Puts four points in the input space where the opposite labels points are
    next to each other. After transform the samples from the same class
    should be next to each other.

    """
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])  # 创建输入样本数组
    y = np.array([1, 0, 1, 0])  # 创建对应的标签数组
    nca = NeighborhoodComponentsAnalysis(
        n_components=2, init="identity", random_state=42
    )  # 创建邻域成分分析对象
    nca.fit(X, y)  # 在样本数据上拟合模型
    X_t = nca.transform(X)  # 对输入数据进行变换
    assert_array_equal(pairwise_distances(X_t).argsort()[:, 1], np.array([2, 3, 0, 1]))

def test_toy_example_collapse_points():
    """Test on a toy example of three points that should collapse

    We build a simple example: two points from the same class and a point from
    a different class in the middle of them. On this simple example, the new
    (transformed) points should all collapse into one single point. Indeed, the
    objective is 2/(1 + exp(d/2)), with d the euclidean distance between the
    two samples from the same class. This is maximized for d=0 (because d>=0),
    with an objective equal to 1 (loss=-1.).

    """
    rng = np.random.RandomState(42)  # 创建指定随机种子的随机状态对象
    input_dim = 5  # 设置输入维度
    two_points = rng.randn(2, input_dim)  # 生成符合正态分布的两个点
    X = np.vstack([two_points, two_points.mean(axis=0)[np.newaxis, :]])  # 堆叠样本数据
    y = [0, 0, 1]  # 设置标签数组

    class LossStorer:
        def __init__(self, X, y):
            self.loss = np.inf  # 初始化损失值为无穷大
            # 初始化一个虚拟的NCA对象和计算损失所需的变量：
            self.fake_nca = NeighborhoodComponentsAnalysis()
            self.fake_nca.n_iter_ = np.inf
            self.X, y = self.fake_nca._validate_data(X, y, ensure_min_samples=2)  # 验证并获取数据
            y = LabelEncoder().fit_transform(y)  # 对标签进行编码
            self.same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]  # 创建相同类别的掩码矩阵

        def callback(self, transformation, n_iter):
            """Stores the last value of the loss function"""
            self.loss, _ = self.fake_nca._loss_grad_lbfgs(
                transformation, self.X, self.same_class_mask, -1.0
            )  # 调用损失函数计算损失值

    loss_storer = LossStorer(X, y)  # 创建损失值存储对象
    # 使用随机种子为42初始化 NeighborhoodComponentsAnalysis 类的实例，并指定回调函数为 loss_storer.callback
    nca = NeighborhoodComponentsAnalysis(random_state=42, callback=loss_storer.callback)
    
    # 使用 NCA 模型拟合数据 X，并进行数据变换，得到转换后的数据 X_t
    X_t = nca.fit_transform(X, y)
    
    # 打印输出转换后的数据 X_t
    print(X_t)
    
    # 检验转换后的数据点是否被压缩为一个点的测试断言
    assert_array_almost_equal(X_t - X_t[0], 0.0)
    
    # 检验损失值是否接近 -1 的测试断言，误差小于 1e-10
    assert abs(loss_storer.loss + 1) < 1e-10
def test_finite_differences(global_random_seed):
    """Test gradient of loss function

    Assert that the gradient is almost equal to its finite differences
    approximation.
    """
    # 使用全局随机种子初始化随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 生成分类数据集 X 和 y
    X, y = make_classification(random_state=global_random_seed)
    # 生成随机变换矩阵 M，其行数在 1 到 X.shape[1]+1 之间随机选择
    M = rng.randn(rng.randint(1, X.shape[1] + 1), X.shape[1])
    # 初始化 NeighborhoodComponentsAnalysis 实例
    nca = NeighborhoodComponentsAnalysis()
    # 将迭代次数 n_iter_ 设为 0
    nca.n_iter_ = 0
    # 创建标签掩码，用于比较每对样本的标签是否相同
    mask = y[:, np.newaxis] == y[np.newaxis, :]

    def fun(M):
        return nca._loss_grad_lbfgs(M, X, mask)[0]

    def grad(M):
        return nca._loss_grad_lbfgs(M, X, mask)[1]

    # 比较梯度与有限差分近似值
    diff = check_grad(fun, grad, M.ravel())
    assert diff == pytest.approx(0.0, abs=1e-4)


def test_params_validation():
    # 测试无效参数是否会引发 ValueError 异常
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]
    NCA = NeighborhoodComponentsAnalysis
    rng = np.random.RandomState(42)

    init = rng.rand(5, 3)
    # 初始化信息字符串，验证输出维度是否大于输入维度
    msg = (
        f"The output dimensionality ({init.shape[0]}) "
        "of the given linear transformation `init` cannot be "
        f"greater than its input dimensionality ({init.shape[1]})."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        NCA(init=init).fit(X, y)
    n_components = 10
    # 初始化信息字符串，验证投影空间维度是否大于数据维度
    msg = (
        "The preferred dimensionality of the projected space "
        f"`n_components` ({n_components}) cannot be greater "
        f"than the given data dimensionality ({X.shape[1]})!"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        NCA(n_components=n_components).fit(X, y)


def test_transformation_dimensions():
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]

    # 如果变换矩阵的输入维度与输入数据维度不匹配，则测试失败
    transformation = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)

    # 如果变换矩阵的输出维度大于其输入维度，则测试失败
    transformation = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)

    # 否则测试通过
    transformation = np.arange(9).reshape(3, 3)
    NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)


def test_n_components():
    rng = np.random.RandomState(42)
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]

    init = rng.rand(X.shape[1] - 1, 3)

    # 当 n_components = X.shape[1] 时，变换矩阵的行数应等于输入数据的列数
    n_components = X.shape[1]
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    # 构造错误消息，当 `n_components` 与线性转换的输出维度不匹配时抛出异常
    msg = (
        "The preferred dimensionality of the projected space "
        f"`n_components` ({n_components}) does not match the output "
        "dimensionality of the given linear transformation "
        f"`init` ({init.shape[0]})!"
    )
    # 使用 pytest 的上下文管理器，验证期望的 ValueError 异常是否被引发，并且异常消息与预期的 msg 匹配
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)

    # 设置 n_components，确保其大于 X 的特征数量
    n_components = X.shape[1] + 2
    # 初始化 NeighborhoodComponentsAnalysis 类，传入初始化参数 init 和 n_components
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    # 构造错误消息，当 `n_components` 大于数据的维度时抛出异常
    msg = (
        "The preferred dimensionality of the projected space "
        f"`n_components` ({n_components}) cannot be greater than "
        f"the given data dimensionality ({X.shape[1]})!"
    )
    # 使用 pytest 的上下文管理器，验证期望的 ValueError 异常是否被引发，并且异常消息与预期的 msg 匹配
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)

    # 设置 n_components，确保其小于 X 的特征数量
    nca = NeighborhoodComponentsAnalysis(n_components=2, init="identity")
    # 执行 NeighborhoodComponentsAnalysis 的拟合操作
    nca.fit(X, y)
def test_init_transformation():
    rng = np.random.RandomState(42)  # 创建随机数生成器对象rng，种子为42
    X, y = make_blobs(n_samples=30, centers=6, n_features=5, random_state=0)
    # 生成包含30个样本、6个中心点、5个特征的数据集X和对应的标签y

    # 从单位矩阵开始学习
    nca = NeighborhoodComponentsAnalysis(init="identity")
    nca.fit(X, y)

    # 用随机值初始化
    nca_random = NeighborhoodComponentsAnalysis(init="random")
    nca_random.fit(X, y)

    # 自动选择初始化方式
    nca_auto = NeighborhoodComponentsAnalysis(init="auto")
    nca_auto.fit(X, y)

    # 用PCA初始化
    nca_pca = NeighborhoodComponentsAnalysis(init="pca")
    nca_pca.fit(X, y)

    # 用LDA初始化
    nca_lda = NeighborhoodComponentsAnalysis(init="lda")
    nca_lda.fit(X, y)

    # 使用随机生成的初始化矩阵init
    init = rng.rand(X.shape[1], X.shape[1])
    nca = NeighborhoodComponentsAnalysis(init=init)
    nca.fit(X, y)

    # init.shape[1]必须与X.shape[1]相匹配
    init = rng.rand(X.shape[1], X.shape[1] + 1)
    nca = NeighborhoodComponentsAnalysis(init=init)
    msg = (
        f"The input dimensionality ({init.shape[1]}) of the given "
        "linear transformation `init` must match the "
        f"dimensionality of the given inputs `X` ({X.shape[1]})."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)

    # init.shape[0]必须 <= init.shape[1]
    init = rng.rand(X.shape[1] + 1, X.shape[1])
    nca = NeighborhoodComponentsAnalysis(init=init)
    msg = (
        f"The output dimensionality ({init.shape[0]}) of the given "
        "linear transformation `init` cannot be "
        f"greater than its input dimensionality ({init.shape[1]})."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)

    # init.shape[0]必须与n_components相匹配
    init = rng.rand(X.shape[1], X.shape[1])
    n_components = X.shape[1] - 2
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    msg = (
        "The preferred dimensionality of the "
        f"projected space `n_components` ({n_components}) "
        "does not match the output dimensionality of the given "
        f"linear transformation `init` ({init.shape[0]})!"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)


@pytest.mark.parametrize("n_samples", [3, 5, 7, 11])
@pytest.mark.parametrize("n_features", [3, 5, 7, 11])
@pytest.mark.parametrize("n_classes", [5, 7, 11])
@pytest.mark.parametrize("n_components", [3, 5, 7, 11])
def test_auto_init(n_samples, n_features, n_classes, n_components):
    # 测试auto选项在各种配置下如何选择初始化
    rng = np.random.RandomState(42)  # 创建随机数生成器对象rng，种子为42
    nca_base = NeighborhoodComponentsAnalysis(
        init="auto", n_components=n_components, max_iter=1, random_state=rng
    )
    if n_classes >= n_samples:
        pass
        # 当n_classes >= n_samples时，会由于LDA算法的限制而抛出错误，但这是不可能的情况
        # n_classes == n_samples是一个荒谬的情况，也会导致错误
    # 否则分支：生成符合正态分布的随机数据矩阵 X，形状为 (n_samples, n_features)
    else:
        X = rng.randn(n_samples, n_features)
        # 创建目标标签 y，通过重复 n_classes 次的序列，保证样本数为 n_samples
        y = np.tile(range(n_classes), n_samples // n_classes + 1)[:n_samples]
        # 如果降维后的特征数大于原特征数，抛出 ValueError 异常，该情况已在 test_params_validation 中测试过
        if n_components > n_features:
            # 此处不执行任何操作，跳过异常情况的处理
            pass
        else:
            # 克隆 nca_base 对象并拟合数据 X, y
            nca = clone(nca_base)
            nca.fit(X, y)
            # 根据 n_components 的大小选择不同的初始化方法，设置为 "lda", "pca", 或 "identity"
            if n_components <= min(n_classes - 1, n_features):
                nca_other = clone(nca_base).set_params(init="lda")
            elif n_components < min(n_features, n_samples):
                nca_other = clone(nca_base).set_params(init="pca")
            else:
                nca_other = clone(nca_base).set_params(init="identity")
            # 使用相同的数据集 X, y 来拟合另一个 nca_other 对象
            nca_other.fit(X, y)
            # 断言 nca 和 nca_other 的成分向量 components_ 相似
            assert_array_almost_equal(nca.components_, nca_other.components_)
# 定义一个测试函数，用于验证在启用暖启动的情况下，NeighborhoodComponentsAnalysis类的行为是否符合预期
def test_warm_start_validation():
    # 创建一个具有特定特征的分类数据集 X 和对应的标签 y
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_classes=4,
        n_redundant=0,
        n_informative=5,
        random_state=0,
    )

    # 初始化 NeighborhoodComponentsAnalysis 对象，启用暖启动，并设置最大迭代次数为 5
    nca = NeighborhoodComponentsAnalysis(warm_start=True, max_iter=5)
    # 对数据集 X 和标签 y 进行拟合
    nca.fit(X, y)

    # 创建另一个具有较少特征的分类数据集 X_less_features 和对应的标签 y
    X_less_features, y = make_classification(
        n_samples=30,
        n_features=4,
        n_classes=4,
        n_redundant=0,
        n_informative=4,
        random_state=0,
    )
    # 创建用于比较维度不匹配的错误消息
    msg = (
        f"The new inputs dimensionality ({X_less_features.shape[1]}) "
        "does not match the input dimensionality of the previously learned "
        f"transformation ({nca.components_.shape[1]})."
    )
    # 使用 pytest 来检查是否抛出 ValueError，并验证错误消息是否符合预期
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X_less_features, y)


# 定义一个测试函数，用于验证暖启动对 NeighborhoodComponentsAnalysis 类的效果
def test_warm_start_effectiveness():
    # 在相同数据上进行第二次迭代，暖启动应该给出几乎相同的结果，而非暖启动会得到截然不同的结果

    # 初始化启用暖启动的 NeighborhoodComponentsAnalysis 对象
    nca_warm = NeighborhoodComponentsAnalysis(warm_start=True, random_state=0)
    # 对鸢尾花数据集 iris_data 进行拟合
    nca_warm.fit(iris_data, iris_target)
    # 获取变换后的组件
    transformation_warm = nca_warm.components_
    # 将最大迭代次数设置为 1，再次对 iris_data 进行拟合
    nca_warm.max_iter = 1
    nca_warm.fit(iris_data, iris_target)
    # 获取再次变换后的组件
    transformation_warm_plus_one = nca_warm.components_

    # 初始化不启用暖启动的 NeighborhoodComponentsAnalysis 对象
    nca_cold = NeighborhoodComponentsAnalysis(warm_start=False, random_state=0)
    # 对鸢尾花数据集 iris_data 进行拟合
    nca_cold.fit(iris_data, iris_target)
    # 获取变换后的组件
    transformation_cold = nca_cold.components_
    # 将最大迭代次数设置为 1，再次对 iris_data 进行拟合
    nca_cold.max_iter = 1
    nca_cold.fit(iris_data, iris_target)
    # 获取再次变换后的组件
    transformation_cold_plus_one = nca_cold.components_

    # 计算变换前后的差异
    diff_warm = np.sum(np.abs(transformation_warm_plus_one - transformation_warm))
    diff_cold = np.sum(np.abs(transformation_cold_plus_one - transformation_cold))
    
    # 断言：使用暖启动后，变换在一次迭代后应显著变化少于3.0
    assert diff_warm < 3.0, (
        "Transformer changed significantly after one "
        "iteration even though it was warm-started."
    )
    
    # 断言：不使用暖启动时，变换在一次迭代后应显著变化大于使用暖启动时的变换
    assert diff_cold > diff_warm, (
        "Cold-started transformer changed less "
        "significantly than warm-started "
        "transformer after one iteration."
    )


# 使用 pytest 参数化装饰器，定义一个测试函数，测试 NeighborhoodComponentsAnalysis 类的 verbose 参数
@pytest.mark.parametrize(
    "init_name", ["pca", "lda", "identity", "random", "precomputed"]
)
def test_verbose(init_name, capsys):
    # 断言：当 verbose = 1 时，每种初始化（除了 auto）都应有适当的输出
    # 生成一个随机数种子
    rng = np.random.RandomState(42)
    # 创建包含6个中心的聚类数据集 X 和对应的标签 y
    X, y = make_blobs(n_samples=30, centers=6, n_features=5, random_state=0)
    # 设置初始化正则表达式模式
    regexp_init = r"... done in \ *\d+\.\d{2}s"
    # 初始化消息字典，包含不同初始化方式的期望输出
    msgs = {
        "pca": "Finding principal components" + regexp_init,
        "lda": "Finding most discriminative components" + regexp_init,
    }
    # 根据不同的初始化名称设置不同的初始化方式
    if init_name == "precomputed":
        init = rng.randn(X.shape[1], X.shape[1])
    else:
        init = init_name
    # 初始化 NeighborhoodComponentsAnalysis 对象，设置 verbose = 1 和指定的初始化方式
    nca = NeighborhoodComponentsAnalysis(verbose=1, init=init)
    # 对数据集 X 和标签 y 进行拟合
    nca.fit(X, y)
    # 读取捕获的标准输出和标准错误输出
    out, _ = capsys.readouterr()

    # 检查输出结果是否符合预期
    lines = re.split("\n+", out)
    # 如果初始化名称为"pca"或"lda"，则检查并移除第一行以确保所有初始化情况下的一致性
    if init_name in ["pca", "lda"]:
        assert re.match(msgs[init_name], lines[0])
        lines = lines[1:]
    
    # 确认第一行应为固定的文本格式
    assert lines[0] == "[NeighborhoodComponentsAnalysis]"
    
    # 确认第二行应为特定格式的标题，包含表头信息
    header = "{:>10} {:>20} {:>10}".format("Iteration", "Objective Value", "Time(s)")
    assert lines[1] == "[NeighborhoodComponentsAnalysis] {}".format(header)
    
    # 确认第三行应为与表头长度相同的横线
    assert lines[2] == "[NeighborhoodComponentsAnalysis] {}".format("-" * len(header))
    
    # 遍历从第四行到倒数第三行（不包括最后两行），确保每行符合特定格式的正则表达式
    for line in lines[3:-2]:
        # 下面的正则表达式匹配类似于 '[NeighborhoodComponentsAnalysis]  0    6.988936e+01   0.01' 的行
        assert re.match(
            r"\[NeighborhoodComponentsAnalysis\] *\d+ *\d\.\d{6}e"
            r"[+|-]\d+\ *\d+\.\d{2}",
            line,
        )
    
    # 确认倒数第二行应为特定格式的文本，指示训练所用时间
    assert re.match(
        r"\[NeighborhoodComponentsAnalysis\] Training took\ *" r"\d+\.\d{2}s\.",
        lines[-2],
    )
    
    # 确认最后一行为空行
    assert lines[-1] == ""
def test_no_verbose(capsys):
    # assert by default there is no output (verbose=0)
    # 创建 NeighborhoodComponentsAnalysis 的实例，不指定 verbose 参数，默认无输出
    nca = NeighborhoodComponentsAnalysis()
    # 使用 iris_data 和 iris_target 进行拟合
    nca.fit(iris_data, iris_target)
    # 捕获标准输出和错误输出
    out, _ = capsys.readouterr()
    # 检查输出是否为空
    assert out == ""


def test_singleton_class():
    X = iris_data
    y = iris_target

    # one singleton class
    # 设置单例类别为 1
    singleton_class = 1
    # 找出 y 中等于 singleton_class 的索引
    (ind_singleton,) = np.where(y == singleton_class)
    # 将这些索引位置的类别改为 2
    y[ind_singleton] = 2
    # 将第一个单例索引位置的类别改回 singleton_class
    y[ind_singleton[0]] = singleton_class

    # 创建 NeighborhoodComponentsAnalysis 的实例，最大迭代次数为 30
    nca = NeighborhoodComponentsAnalysis(max_iter=30)
    # 使用 X 和 y 进行拟合
    nca.fit(X, y)

    # One non-singleton class
    # 找出 y 中等于 1 的索引
    (ind_1,) = np.where(y == 1)
    # 找出 y 中等于 2 的索引
    (ind_2,) = np.where(y == 2)
    # 将这些索引位置的类别改为 0
    y[ind_1] = 0
    # 将第一个非单例索引位置的类别改回 1
    y[ind_1[0]] = 1
    # 将这些索引位置的类别改为 0
    y[ind_2] = 0
    # 将第一个非单例索引位置的类别改回 2
    y[ind_2[0]] = 2

    # 创建 NeighborhoodComponentsAnalysis 的实例，最大迭代次数为 30
    nca = NeighborhoodComponentsAnalysis(max_iter=30)
    # 使用 X 和 y 进行拟合
    nca.fit(X, y)

    # Only singleton classes
    # 找出 y 中等于 0、1、2 的索引
    (ind_0,) = np.where(y == 0)
    (ind_1,) = np.where(y == 1)
    (ind_2,) = np.where(y == 2)
    # 仅保留 X 和 y 中第一个索引位置的数据点
    X = X[[ind_0[0], ind_1[0], ind_2[0]]]
    y = y[[ind_0[0], ind_1[0], ind_2[0]]]

    # 创建 NeighborhoodComponentsAnalysis 的实例，使用单位矩阵初始化，最大迭代次数为 30
    nca = NeighborhoodComponentsAnalysis(init="identity", max_iter=30)
    # 使用 X 和 y 进行拟合
    nca.fit(X, y)
    # 断言 X 转换后的结果与 nca.transform(X) 相等
    assert_array_equal(X, nca.transform(X))


def test_one_class():
    X = iris_data[iris_target == 0]
    y = iris_target[iris_target == 0]

    # 创建 NeighborhoodComponentsAnalysis 的实例，使用单位矩阵初始化，最大迭代次数为 30
    nca = NeighborhoodComponentsAnalysis(
        max_iter=30, n_components=X.shape[1], init="identity"
    )
    # 使用 X 和 y 进行拟合
    nca.fit(X, y)
    # 断言 X 转换后的结果与 nca.transform(X) 相等
    assert_array_equal(X, nca.transform(X))


def test_callback(capsys):
    max_iter = 10

    def my_cb(transformation, n_iter):
        # 断言 transformation 的形状为 (特征数 ** 2,)
        assert transformation.shape == (iris_data.shape[1] ** 2,)
        # 计算剩余迭代次数
        rem_iter = max_iter - n_iter
        # 打印剩余迭代次数信息
        print("{} iterations remaining...".format(rem_iter))

    # 创建 NeighborhoodComponentsAnalysis 的实例，最大迭代次数为 10，使用自定义回调函数 my_cb，输出详细信息
    nca = NeighborhoodComponentsAnalysis(max_iter=max_iter, callback=my_cb, verbose=1)
    # 使用 iris_data 和 iris_target 进行拟合
    nca.fit(iris_data, iris_target)
    # 捕获标准输出和错误输出
    out, _ = capsys.readouterr()

    # 检查输出中是否包含剩余迭代次数信息
    assert "{} iterations remaining...".format(max_iter - 1) in out


def test_expected_transformation_shape():
    """Test that the transformation has the expected shape."""
    X = iris_data
    y = iris_target

    class TransformationStorer:
        def __init__(self, X, y):
            # 初始化一个虚拟的 NCA 和调用损失函数所需的变量
            self.fake_nca = NeighborhoodComponentsAnalysis()
            self.fake_nca.n_iter_ = np.inf
            # 验证并处理输入数据 X 和 y，确保样本数不少于 2
            self.X, y = self.fake_nca._validate_data(X, y, ensure_min_samples=2)
            # 对目标标签进行编码
            y = LabelEncoder().fit_transform(y)
            # 创建相似类别掩码
            self.same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]

        def callback(self, transformation, n_iter):
            """Stores the last value of the transformation taken as input by
            the optimizer"""
            # 存储优化器输入的变换的最后一个值
            self.transformation = transformation

    transformation_storer = TransformationStorer(X, y)
    # 获取 callback 方法
    cb = transformation_storer.callback
    # 创建 NeighborhoodComponentsAnalysis 的实例，最大迭代次数为 5，使用自定义回调函数 cb
    nca = NeighborhoodComponentsAnalysis(max_iter=5, callback=cb)
    # 使用 X 和 y 进行拟合
    nca.fit(X, y)
    # 使用断言验证 transformation_storer.transformation 的大小是否等于 X 矩阵的平方大小
    assert transformation_storer.transformation.size == X.shape[1] ** 2
# 定义测试函数，用于测试 NeighborhoodComponentsAnalysis 类的收敛警告
def test_convergence_warning():
    # 创建 NeighborhoodComponentsAnalysis 对象，设置最大迭代次数为 2，启用详细输出
    nca = NeighborhoodComponentsAnalysis(max_iter=2, verbose=1)
    # 获取对象的类名
    cls_name = nca.__class__.__name__
    # 构造警告信息字符串，指示 NCA 未收敛
    msg = "[{}] NCA did not converge".format(cls_name)
    # 使用 pytest 的 warns 上下文，检查是否触发 ConvergenceWarning，且警告信息匹配预期
    with pytest.warns(ConvergenceWarning, match=re.escape(msg)):
        # 对 iris 数据进行拟合
        nca.fit(iris_data, iris_target)


# 使用参数化测试标记，测试参数的有效类型
@pytest.mark.parametrize(
    "param, value",
    [
        ("n_components", np.int32(3)),  # 设置 n_components 参数为 numpy 整数类型 3
        ("max_iter", np.int32(100)),    # 设置 max_iter 参数为 numpy 整数类型 100
        ("tol", np.float32(0.0001)),    # 设置 tol 参数为 numpy 浮点数类型 0.0001
    ],
)
def test_parameters_valid_types(param, value):
    # 创建 NeighborhoodComponentsAnalysis 对象，根据给定参数设置
    nca = NeighborhoodComponentsAnalysis(**{param: value})

    # 使用 iris 数据集作为输入特征 X 和目标标签 y
    X = iris_data
    y = iris_target

    # 对模型进行拟合
    nca.fit(X, y)


# 使用参数化测试标记，测试 n_components 参数为 None 或整数 2 时的特征名称输出
@pytest.mark.parametrize("n_components", [None, 2])
def test_nca_feature_names_out(n_components):
    """检查 `NeighborhoodComponentsAnalysis` 的 `get_feature_names_out` 方法。

    针对以下问题进行非回归测试：
    https://github.com/scikit-learn/scikit-learn/issues/28293
    """

    # 使用 iris 数据集作为输入特征 X 和目标标签 y
    X = iris_data
    y = iris_target

    # 创建 NeighborhoodComponentsAnalysis 对象，设置 n_components 参数
    est = NeighborhoodComponentsAnalysis(n_components=n_components).fit(X, y)
    # 获取特征名称输出
    names_out = est.get_feature_names_out()

    # 获取对象的类名并转换为小写
    class_name_lower = est.__class__.__name__.lower()

    # 根据 n_components 参数确定预期的特征数量
    if n_components is not None:
        expected_n_features = n_components
    else:
        expected_n_features = X.shape[1]

    # 生成预期的特征名称数组
    expected_names_out = np.array(
        [f"{class_name_lower}{i}" for i in range(expected_n_features)],
        dtype=object,
    )

    # 断言实际特征名称输出与预期相等
    assert_array_equal(names_out, expected_names_out)
```