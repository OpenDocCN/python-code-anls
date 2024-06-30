# `D:\src\scipysrc\scipy\benchmarks\benchmarks\cutest\calfun.py`

```
# 导入必要的库
import numpy as np
from .dfovec import dfovec

# 定义一个函数，计算向量的范数，默认是二范数
def norm(x, type=2):
    if type == 1:
        return np.sum(np.abs(x))  # 计算向量的一范数
    elif type == 2:
        return np.sqrt(x ** 2)  # 计算向量的二范数
    else:  # type==np.inf:
        return max(np.abs(x))  # 计算向量的无穷范数

# 定义一个函数，根据不同的问题类型计算函数值
def calfun(x, m, nprob, probtype="smooth", noise_level=1e-3):
    n = len(x)

    # 对于某些非可微问题，限制定义域
    xc = x
    if probtype == "nondiff":
        if (
            nprob == 8
            or nprob == 9
            or nprob == 13
            or nprob == 16
            or nprob == 17
            or nprob == 18
        ):
            xc = max(x, 0)  # 对于指定的问题，取向量中的非负部分

    # 生成向量
    fvec = dfovec(m, n, xc, nprob)

    # 计算函数值
    if probtype == "noisy3":
        sigma = noise_level
        u = sigma * (-np.ones(m) + 2 * np.random.rand(m))
        fvec = fvec * (1 + u)
        y = np.sum(fvec ** 2)  # 添加噪声后的函数值计算
    elif probtype == "wild3":
        sigma = noise_level
        phi = 0.9 * np.sin(100 * norm(x, 1)) * np.cos(
            100 * norm(x, np.inf)
        ) + 0.1 * np.cos(norm(x, 2))
        phi = phi * (4 * phi ** 2 - 3)
        y = (1 + sigma * phi) * sum(fvec ** 2)  # 野性问题类型的函数值计算
    elif probtype == "smooth":
        y = np.sum(fvec ** 2)  # 光滑问题类型的函数值计算
    elif probtype == "nondiff":
        y = np.sum(np.abs(fvec))  # 非可微问题类型的函数值计算
    else:
        print(f"invalid probtype {probtype}")
        return None

    # 永远不返回 NaN，返回无穷大，以便优化算法将其视为超出界限
    if np.isnan(y):
        return np.inf
    return y  # 返回计算得到的函数值
```