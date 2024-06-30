# `D:\src\scipysrc\scikit-learn\sklearn\mixture\tests\test_mixture.py`

```
# 导入必要的库
import numpy as np  # 导入numpy库，用于科学计算
import pytest  # 导入pytest库，用于编写和运行测试用例

# 从sklearn.mixture模块中导入需要测试的类
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

# 参数化测试函数，分别使用GaussianMixture和BayesianGaussianMixture实例作为参数
@pytest.mark.parametrize("estimator", [GaussianMixture(), BayesianGaussianMixture()])
def test_gaussian_mixture_n_iter(estimator):
    # 检查n_iter是否等于实际迭代次数
    rng = np.random.RandomState(0)  # 创建随机数生成器实例rng，种子为0
    X = rng.rand(10, 5)  # 生成一个10行5列的随机数组成的矩阵X
    max_iter = 1  # 设置最大迭代次数为1
    estimator.set_params(max_iter=max_iter)  # 设置估计器(estimator)的最大迭代次数
    estimator.fit(X)  # 使用数据X拟合(estimator)
    assert estimator.n_iter_ == max_iter  # 断言实际迭代次数与设定的最大迭代次数相等


# 参数化测试函数，分别使用GaussianMixture和BayesianGaussianMixture实例作为参数
@pytest.mark.parametrize("estimator", [GaussianMixture(), BayesianGaussianMixture()])
def test_mixture_n_components_greater_than_n_samples_error(estimator):
    """检查当n_components <= n_samples时是否会抛出错误"""
    rng = np.random.RandomState(0)  # 创建随机数生成器实例rng，种子为0
    X = rng.rand(10, 5)  # 生成一个10行5列的随机数组成的矩阵X
    estimator.set_params(n_components=12)  # 设置估计器(estimator)的混合模型组件数量为12

    msg = "Expected n_samples >= n_components"  # 设置期望的错误信息
    with pytest.raises(ValueError, match=msg):  # 使用pytest断言捕获预期的ValueError异常，并匹配错误信息msg
        estimator.fit(X)  # 尝试拟合(estimator)使用数据X
```