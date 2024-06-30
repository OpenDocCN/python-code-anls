# `D:\src\scipysrc\scipy\scipy\special\tests\test_cosine_distr.py`

```
# 导入 numpy 库，并使用 np 作为别名
import numpy as np
# 从 numpy.testing 模块中导入 assert_allclose 函数
from numpy.testing import assert_allclose
# 导入 pytest 库，用于编写测试用例
import pytest
# 从 scipy.special._ufuncs 模块中导入 _cosine_cdf 和 _cosine_invcdf 函数
from scipy.special._ufuncs import _cosine_cdf, _cosine_invcdf


# 这些值是 (x, p)，其中 p 是 _cosine_cdf(x) 的预期精确值。这些值将被用于精确匹配测试。
_coscdf_exact = [
    (-4.0, 0.0),
    (0, 0.5),
    (np.pi, 1.0),
    (4.0, 1.0),
]

# 使用 pytest.mark.parametrize 装饰器定义参数化测试，对 _cosine_cdf 函数进行精确匹配测试
@pytest.mark.parametrize("x, expected", _coscdf_exact)
def test_cosine_cdf_exact(x, expected):
    assert _cosine_cdf(x) == expected


# 这些值是 (x, p)，其中 p 是 _cosine_cdf(x) 的预期值。这些值是使用 mpmath 计算得到的，保留了50位小数精度。
# 这些值将会使用非常小的相对容差进行测试，用于检验计算值与预期值的一致性。
# -np.pi 处的值不是 0，因为 -np.pi 不等于 -π。
_coscdf_close = [
    (3.1409, 0.999999999991185),
    (2.25, 0.9819328173287907),
    # -1.6 是使用 Pade 近似方法的阈值下限。
    (-1.599, 0.08641959838382553),
    (-1.601, 0.086110582992713),
    (-2.0, 0.0369709335961611),
    (-3.0, 7.522387241801384e-05),
    (-3.1415, 2.109869685443648e-14),
    (-3.14159, 4.956444476505336e-19),
    (-np.pi, 4.871934450264861e-50),
]

# 使用 pytest.mark.parametrize 装饰器定义参数化测试，对 _cosine_cdf 函数进行数值一致性测试
@pytest.mark.parametrize("x, expected", _coscdf_close)
def test_cosine_cdf(x, expected):
    assert_allclose(_cosine_cdf(x), expected, rtol=5e-15)


# 这些值是 (p, x)，其中 x 是 _cosine_invcdf(p) 的预期精确值。这些值将被用于精确匹配测试。
_cosinvcdf_exact = [
    (0.0, -np.pi),
    (0.5, 0.0),
    (1.0, np.pi),
]

# 使用 pytest.mark.parametrize 装饰器定义参数化测试，对 _cosine_invcdf 函数进行精确匹配测试
@pytest.mark.parametrize("p, expected", _cosinvcdf_exact)
def test_cosine_invcdf_exact(p, expected):
    assert _cosine_invcdf(p) == expected


# 定义一个测试函数，用于检查 _cosine_invcdf 函数在无效 p 值（超出 [0, 1] 范围）时返回 NaN。
def test_cosine_invcdf_invalid_p():
    assert np.isnan(_cosine_invcdf([-0.1, 1.1])).all()


# 这些值是 (p, x)，其中 x 是 _cosine_invcdf(p) 的预期值。
# 这些值是使用 mpmath 计算得到的，保留了50位小数精度。
_cosinvcdf_close = [
    (1e-50, -np.pi),
    (1e-14, -3.1415204137058454),
    (1e-08, -3.1343686589124524),
    (0.0018001, -2.732563923138336),
    (0.010, -2.41276589008678),
    (0.060, -1.7881244975330157),
    (0.125, -1.3752523669869274),
    (0.250, -0.831711193579736),
    (0.400, -0.3167954512395289),
    (0.419, -0.25586025626919906),
    (0.421, -0.24947570750445663),
    (0.750, 0.831711193579736),
    (0.940, 1.7881244975330153),
    (0.9999999996, 3.1391220839917167),
]

# 使用 pytest.mark.parametrize 装饰器定义参数化测试，对 _cosine_invcdf 函数进行数值一致性测试
@pytest.mark.parametrize("p, expected", _cosinvcdf_close)
def test_cosine_invcdf(p, expected):
    assert_allclose(_cosine_invcdf(p), expected, rtol=1e-14)
```