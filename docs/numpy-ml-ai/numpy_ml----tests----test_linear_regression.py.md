# `numpy-ml\numpy_ml\tests\test_linear_regression.py`

```py
# 禁用 flake8 检查
# 导入 numpy 库并重命名为 np
import numpy as np

# 从 sklearn 线性模型中导入 LinearRegression 类并重命名为 LinearRegressionGold
from sklearn.linear_model import LinearRegression as LinearRegressionGold

# 从 numpy_ml 线性模型中导入 LinearRegression 类
from numpy_ml.linear_models import LinearRegression

# 从 numpy_ml 工具包中导入 random_tensor 函数
from numpy_ml.utils.testing import random_tensor

# 定义测试线性回归函数，参数 N 默认值为 10
def test_linear_regression(N=10):
    # 设置随机种子为 12345
    np.random.seed(12345)
    # 如果 N 为 None，则将 N 设置为正无穷
    N = np.inf if N is None else N

    # 初始化变量 i 为 1
    i = 1
```