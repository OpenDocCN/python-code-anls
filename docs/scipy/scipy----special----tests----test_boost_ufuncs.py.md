# `D:\src\scipysrc\scipy\scipy\special\tests\test_boost_ufuncs.py`

```
# 导入 pytest 库，用于编写和运行测试用例
import pytest
# 导入 numpy 库，并将其重命名为 np，用于数值计算和数组操作
import numpy as np
# 从 numpy.testing 模块中导入 assert_allclose 函数，用于检查数组或者数值的近似相等性
from numpy.testing import assert_allclose
# 导入 scipy.special._ufuncs 模块，用于测试其中定义的函数
import scipy.special._ufuncs as scu

# 定义一个字典，将类型字符映射到其对应的 NumPy 类型和容差值
type_char_to_type_tol = {'f': (np.float32, 32*np.finfo(np.float32).eps),
                         'd': (np.float64, 32*np.finfo(np.float64).eps)}

# 定义一个测试数据列表，包含多个元组，每个元组包含函数、参数和期望值
# 这些数据用于对特定函数进行精确性测试，特别是检查不同数据类型的处理是否正确
test_data = [
    (scu._beta_pdf, (0.5, 2, 3), 1.5),
    (scu._beta_pdf, (0, 1, 5), 5.0),
    (scu._beta_pdf, (1, 5, 1), 5.0),
    (scu._binom_cdf, (1, 3, 0.5), 0.5),
    (scu._binom_pmf, (1, 4, 0.5), 0.25),
    (scu._hypergeom_cdf, (2, 3, 5, 6), 0.5),
    (scu._nbinom_cdf, (1, 4, 0.25), 0.015625),
    (scu._ncf_mean, (10, 12, 2.5), 1.5),
]

# 使用 pytest.mark.parametrize 装饰器，参数化测试函数 test_stats_boost_ufunc
# 该函数接受 func、args 和 expected 参数，分别表示函数、参数和预期结果
@pytest.mark.parametrize('func, args, expected', test_data)
def test_stats_boost_ufunc(func, args, expected):
    # 获取函数 func 的类型签名列表
    type_sigs = func.types
    # 从类型签名中提取返回类型的字符表示，并存储在 type_chars 列表中
    type_chars = [sig.split('->')[-1] for sig in type_sigs]
    # 遍历每个返回类型的字符表示
    for type_char in type_chars:
        # 从 type_char_to_type_tol 字典中获取对应的 NumPy 类型和容差值
        typ, rtol = type_char_to_type_tol[type_char]
        # 将参数 args 中的每个元素转换为 typ 类型
        args = [typ(arg) for arg in args]
        # 忽略潜在的溢出警告，确保测试在数据类型和精度上的准确性
        with np.errstate(over='ignore'):
            # 调用 func 函数，传入参数 args，并获取返回值
            value = func(*args)
        # 断言返回值 value 的类型为 typ
        assert isinstance(value, typ)
        # 使用 assert_allclose 函数检查返回值 value 是否接近于期望值 expected
        assert_allclose(value, expected, rtol=rtol)
```