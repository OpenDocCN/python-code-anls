# `.\lucidrains\se3-transformer-pytorch\se3_transformer_pytorch\spherical_harmonics.py`

```py
# 从 math 模块中导入 pi 和 sqrt 函数
# 从 functools 模块中导入 reduce 函数
# 从 operator 模块中导入 mul 函数
# 导入 torch 模块
from math import pi, sqrt
from functools import reduce
from operator import mul
import torch

# 从 functools 模块中导入 lru_cache 装饰器
# 从 se3_transformer_pytorch.utils 模块中导入 cache 函数
from functools import lru_cache
from se3_transformer_pytorch.utils import cache

# 定义常量 CACHE，初始化为空字典
CACHE = {}

# 清空球谐函数缓存
def clear_spherical_harmonics_cache():
    CACHE.clear()

# 定义函数 lpmv_cache_key_fn，用于生成缓存键
def lpmv_cache_key_fn(l, m, x):
    return (l, m)

# 定义函数 semifactorial，使用 lru_cache 装饰器缓存结果
@lru_cache(maxsize = 1000)
def semifactorial(x):
    return reduce(mul, range(x, 1, -2), 1.)

# 定义函数 pochhammer，使用 lru_cache 装饰器缓存结果
@lru_cache(maxsize = 1000)
def pochhammer(x, k):
    return reduce(mul, range(x + 1, x + k), float(x))

# 定义函数 negative_lpmv，计算负的球谐函数
def negative_lpmv(l, m, y):
    if m < 0:
        y *= ((-1) ** m / pochhammer(l + m + 1, -2 * m))
    return y

# 定义函数 lpmv，使用 cache 装饰器缓存结果
@cache(cache=CACHE, key_fn=lpmv_cache_key_fn)
def lpmv(l, m, x):
    """Associated Legendre function including Condon-Shortley phase.

    Args:
        m: int order 
        l: int degree
        x: float argument tensor
    Returns:
        tensor of x-shape
    """
    # 检查是否有缓存版本
    m_abs = abs(m)

    if m_abs > l:
        return None

    if l == 0:
        return torch.ones_like(x)
    
    if m_abs == l:
        y = (-1)**m_abs * semifactorial(2*m_abs-1)
        y *= torch.pow(1-x*x, m_abs/2)
        return negative_lpmv(l, m, y)

    lpmv(l-1, m, x)

    y = ((2*l-1) / (l-m_abs)) * x * lpmv(l-1, m_abs, x)

    if l - m_abs > 1:
        y -= ((l+m_abs-1)/(l-m_abs)) * CACHE[(l-2, m_abs)]
    
    if m < 0:
        y = negative_lpmv(l, m, y)
    return y

# 定义函数 get_spherical_harmonics_element，计算球谐函数元素
def get_spherical_harmonics_element(l, m, theta, phi):
    """Tesseral spherical harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        m: int for order, where -l <= m < l
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape theta
    """
    m_abs = abs(m)
    assert m_abs <= l, "absolute value of order m must be <= degree l"

    N = sqrt((2*l + 1) / (4 * pi))
    leg = lpmv(l, m_abs, torch.cos(theta))

    if m == 0:
        return N * leg

    if m > 0:
        Y = torch.cos(m * phi)
    else:
        Y = torch.sin(m_abs * phi)

    Y *= leg
    N *= sqrt(2. / pochhammer(l - m_abs + 1, 2 * m_abs))
    Y *= N
    return Y

# 定义函数 get_spherical_harmonics，计算球谐函数
def get_spherical_harmonics(l, theta, phi):
    """ Tesseral harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape [*theta.shape, 2*l+1]
    """
    return torch.stack([get_spherical_harmonics_element(l, m, theta, phi) \
                        for m in range(-l, l+1)],
                        dim=-1)
```