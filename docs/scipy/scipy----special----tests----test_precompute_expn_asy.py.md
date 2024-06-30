# `D:\src\scipysrc\scipy\scipy\special\tests\test_precompute_expn_asy.py`

```
# 从 numpy.testing 模块中导入 assert_equal 函数，用于断言测试结果是否相等
from numpy.testing import assert_equal

# 从 scipy.special._testutils 模块中导入 check_version 和 MissingModule
from scipy.special._testutils import check_version, MissingModule
# 从 scipy.special._precompute.expn_asy 模块中导入 generate_A 函数
from scipy.special._precompute.expn_asy import generate_A

# 尝试导入 sympy 库，如果失败则将 sympy 替换为 MissingModule("sympy")
try:
    import sympy
    from sympy import Poly  # 从 sympy 库中导入 Poly 类
except ImportError:
    sympy = MissingModule("sympy")  # 若导入失败，将 sympy 替换为 MissingModule 对象

# 使用 @check_version 装饰器，检查 sympy 版本是否至少为 1.0
@check_version(sympy, "1.0")
def test_generate_A():
    # 使用 sympy.symbols 函数创建符号变量 x
    x = sympy.symbols('x')
    # 定义标准的多项式列表 Astd，这些多项式来自于 DLMF 8.20.5
    Astd = [Poly(1, x),
            Poly(1, x),
            Poly(1 - 2*x),
            Poly(1 - 8*x + 6*x**2)]
    # 调用 generate_A 函数生成与 Astd 相同长度的结果列表 Ares
    Ares = generate_A(len(Astd))

    # 使用 zip 函数遍历 Astd 和 Ares 列表，依次比较每对多项式 p 和 q 是否相等
    for p, q in zip(Astd, Ares):
        assert_equal(p, q)  # 断言 p 和 q 相等，若不相等则测试失败
```