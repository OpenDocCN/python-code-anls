# `numpy-ml\numpy_ml\tests\test_glm.py`

```
# 禁用 flake8 检查
# 导入 numpy 库并重命名为 np
import numpy as np

# 导入 statsmodels 库中的 api 模块并重命名为 sm
import statsmodels.api as sm
# 从 numpy_ml.linear_models 模块中导入 GeneralizedLinearModel 类
from numpy_ml.linear_models import GeneralizedLinearModel
# 从 numpy_ml.linear_models.glm 模块中导入 _GLM_LINKS 变量
from numpy_ml.linear_models.glm import _GLM_LINKS
# 从 numpy_ml.utils.testing 模块中导入 random_tensor 函数
from numpy_ml.utils.testing import random_tensor

# 定义一个测试函数 test_glm，参数 N 默认值为 20
def test_glm(N=20):
    # 设置随机种子为 12345
    np.random.seed(12345)
    # 如果 N 为 None，则将 N 设置为正无穷
    N = np.inf if N is None else N

    # 初始化变量 i 为 1
    i = 1
    # 循环执行直到 i 大于 N
    while i < N + 1:
        # 生成一个随机整数作为样本数量，范围在 10 到 100 之间
        n_samples = np.random.randint(10, 100)

        # 生成一个随机整数作为特征数量，确保特征数量远小于样本数量，避免完全分离或多个解决方案
        n_feats = np.random.randint(1, 1 + n_samples // 2)
        target_dim = 1

        # 随机选择是否拟合截距
        fit_intercept = np.random.choice([True, False])
        # 随机选择链接函数
        _link = np.random.choice(list(_GLM_LINKS.keys()))

        # 创建不同链接函数对应的家族
        families = {
            "identity": sm.families.Gaussian(),
            "logit": sm.families.Binomial(),
            "log": sm.families.Poisson(),
        }

        # 打印当前链接函数和是否拟合截距
        print(f"Link: {_link}")
        print(f"Fit intercept: {fit_intercept}")

        # 生成随机数据作为特征矩阵 X
        X = random_tensor((n_samples, n_feats), standardize=True)
        # 根据链接函数生成随机标签 y
        if _link == "logit":
            y = np.random.choice([0.0, 1.0], size=(n_samples, target_dim))
        elif _link == "log":
            y = np.random.choice(np.arange(0, 100), size=(n_samples, target_dim))
        elif _link == "identity":
            y = random_tensor((n_samples, target_dim), standardize=True)
        else:
            raise ValueError(f"Unknown link function {_link}")

        # 在整个数据集上拟合标准模型
        fam = families[_link]
        Xdesign = np.c_[np.ones(X.shape[0]), X] if fit_intercept else X

        glm_gold = sm.GLM(y, Xdesign, family=fam)
        glm_gold = glm_gold.fit()

        # 使用自定义的广义线性模型拟合数据
        glm_mine = GeneralizedLinearModel(link=_link, fit_intercept=fit_intercept)
        glm_mine.fit(X, y)

        # 检查模型系数是否匹配
        beta = glm_mine.beta.T.ravel()
        np.testing.assert_almost_equal(beta, glm_gold.params, decimal=6)
        print("\t1. Overall model coefficients match")

        # 检查模型预测是否匹配
        np.testing.assert_almost_equal(
            glm_mine.predict(X), glm_gold.predict(Xdesign), decimal=5
        )
        print("\t2. Overall model predictions match")

        # 打印测试通过信息
        print("\tPASSED\n")
        i += 1
```