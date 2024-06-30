# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\tests\test_report.py`

```
import pytest  # 导入 pytest 库，用于测试框架
import numpy as np  # 导入 NumPy 库，用于数值计算
from scipy.optimize import minimize, Bounds  # 从 SciPy 库中导入 minimize 函数和 Bounds 类

def test_gh10880():
    # 测试函数，用于检查 verbose 报告是否在有界约束问题中正常工作
    bnds = Bounds(1, 2)  # 创建一个包含上下界的 Bounds 对象，范围为 [1, 2]
    opts = {'maxiter': 1000, 'verbose': 2}  # 设置选项，包括最大迭代次数和详细程度为 2
    minimize(lambda x: x**2, x0=2., method='trust-constr',
             bounds=bnds, options=opts)  # 调用 minimize 函数进行优化

    opts = {'maxiter': 1000, 'verbose': 3}  # 设置选项，详细程度提升到 3
    minimize(lambda x: x**2, x0=2., method='trust-constr',
             bounds=bnds, options=opts)  # 再次调用 minimize 函数进行优化

@pytest.mark.xslow  # 标记该测试函数为较慢的测试
def test_gh12922():
    # 测试函数，用于检查 verbose 报告是否在一般约束问题中正常工作
    def objective(x):
        return np.array([(np.sum((x+1)**4))])  # 定义目标函数，计算给定表达式的数值

    cons = {'type': 'ineq', 'fun': lambda x: -x[0]**2}  # 定义约束条件，这里为一个不等式约束
    n = 25
    x0 = np.linspace(-5, 5, n)  # 生成初始点数组

    opts = {'maxiter': 1000, 'verbose': 2}  # 设置选项，包括最大迭代次数和详细程度为 2
    minimize(objective, x0=x0, method='trust-constr',
                      constraints=cons, options=opts)  # 调用 minimize 函数进行优化

    opts = {'maxiter': 1000, 'verbose': 3}  # 设置选项，详细程度提升到 3
    minimize(objective, x0=x0, method='trust-constr',
                      constraints=cons, options=opts)  # 再次调用 minimize 函数进行优化
```