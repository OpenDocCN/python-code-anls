# `D:\src\scipysrc\sympy\sympy\polys\tests\test_multivariate_resultants.py`

```
"""Tests for Dixon's and Macaulay's classes. """

# 导入所需的类和函数
from sympy.matrices.dense import Matrix  # 导入密集矩阵类
from sympy.polys.polytools import factor  # 导入多项式工具中的因式分解函数
from sympy.core import symbols  # 导入符号变量定义
from sympy.tensor.indexed import IndexedBase  # 导入索引基类

# 导入Dixon和Macaulay结果子模块中的类
from sympy.polys.multivariate_resultants import (DixonResultant,  # 导入Dixon结果类
                                                 MacaulayResultant)  # 导入Macaulay结果类

# 定义符号变量
c, d = symbols("a, b")
x, y = symbols("x, y")

# 定义两个多项式
p =  c * x + y  # 第一个多项式
q =  x + d * y  # 第二个多项式

# 创建Dixon和Macaulay结果对象
dixon = DixonResultant(polynomials=[p, q], variables=[x, y])  # Dixon结果对象
macaulay = MacaulayResultant(polynomials=[p, q], variables=[x, y])  # Macaulay结果对象

# 测试DixonResultant类的初始化方法
def test_dixon_resultant_init():
    """Test init method of DixonResultant."""
    a = IndexedBase("alpha")  # 创建索引基对象

    # 断言检查初始化后的属性
    assert dixon.polynomials == [p, q]
    assert dixon.variables == [x, y]
    assert dixon.n == 2
    assert dixon.m == 2
    assert dixon.dummy_variables == [a[0], a[1]]

# 测试DixonResultant类中获取Dixon多项式的数值示例方法
def test_get_dixon_polynomial_numerical():
    """Test Dixon's polynomial for a numerical example."""
    a = IndexedBase("alpha")  # 创建索引基对象

    # 定义新的多项式和预期的Dixon多项式
    p = x + y
    q = x ** 2 + y **3
    h = x ** 2 + y

    # 创建新的DixonResultant对象并计算预期的Dixon多项式
    dixon = DixonResultant([p, q, h], [x, y])
    polynomial = -x * y ** 2 * a[0] - x * y ** 2 * a[1] - x * y * a[0] \
    * a[1] - x * y * a[1] ** 2 - x * a[0] * a[1] ** 2 + x * a[0] - \
    y ** 2 * a[0] * a[1] + y ** 2 * a[1] - y * a[0] * a[1] ** 2 + y * \
    a[1] ** 2

    # 断言检查计算结果是否符合预期
    assert dixon.get_dixon_polynomial().as_expr().expand() == polynomial

# 测试获取最大次数的函数
def test_get_max_degrees():
    """Tests max degrees function."""

    # 定义新的多项式
    p = x + y
    q = x ** 2 + y **3
    h = x ** 2 + y

    # 创建新的DixonResultant对象并获取Dixon多项式
    dixon = DixonResultant(polynomials=[p, q, h], variables=[x, y])
    dixon_polynomial = dixon.get_dixon_polynomial()

    # 断言检查获取的最大次数是否符合预期
    assert dixon.get_max_degrees(dixon_polynomial) == [1, 2]

# 测试获取Dixon矩阵的方法
def test_get_dixon_matrix():
    """Test Dixon's resultant for a numerical example."""

    x, y = symbols('x, y')

    # 定义新的多项式
    p = x + y
    q = x ** 2 + y ** 3
    h = x ** 2 + y

    # 创建新的DixonResultant对象并获取Dixon多项式
    dixon = DixonResultant([p, q, h], [x, y])
    polynomial = dixon.get_dixon_polynomial()

    # 断言检查Dixon矩阵的行列式是否为零
    assert dixon.get_dixon_matrix(polynomial).det() == 0

# 测试从文献[Palancz08]_中的例子获取Dixon矩阵的方法
def test_get_dixon_matrix_example_two():
    """Test Dixon's matrix for example from [Palancz08]_."""
    x, y, z = symbols('x, y, z')

    # 定义新的多项式
    f = x ** 2 + y ** 2 - 1 + z * 0
    g = x ** 2 + z ** 2 - 1 + y * 0
    h = y ** 2 + z ** 2 - 1

    # 创建新的DixonResultant对象并获取Dixon多项式
    example_two = DixonResultant([f, g, h], [y, z])
    poly = example_two.get_dixon_polynomial()
    matrix = example_two.get_dixon_matrix(poly)

    # 断言检查Dixon矩阵的行列式是否与预期的表达式展开结果相等
    expr = 1 - 8 * x ** 2 + 24 * x ** 4 - 32 * x ** 6 + 16 * x ** 8
    assert (matrix.det() - expr).expand() == 0

# 测试KSY结果条件的前提条件
def test_KSY_precondition():
    """Tests precondition for KSY Resultant."""
    A, B, C = symbols('A, B, C')

    # 定义多个矩阵
    m1 = Matrix([[1, 2, 3],
                 [4, 5, 12],
                 [6, 7, 18]])

    m2 = Matrix([[0, C**2],
                 [-2 * C, -C ** 2]])

    m3 = Matrix([[1, 0],
                 [0, 1]])

    m4 = Matrix([[A**2, 0, 1],
                 [A, 1, 1 / A]])

    m5 = Matrix([[5, 1],
                 [2, B],
                 [0, 1],
                 [0, 0]])
    # 断言，验证 dixon 模块中的 KSY_precondition 函数对给定参数的预期行为是否符合预期
    assert dixon.KSY_precondition(m1) == False
    # 断言，验证 dixon 模块中的 KSY_precondition 函数对给定参数的预期行为是否符合预期
    assert dixon.KSY_precondition(m2) == True
    # 断言，验证 dixon 模块中的 KSY_precondition 函数对给定参数的预期行为是否符合预期
    assert dixon.KSY_precondition(m3) == True
    # 断言，验证 dixon 模块中的 KSY_precondition 函数对给定参数的预期行为是否符合预期
    assert dixon.KSY_precondition(m4) == False
    # 断言，验证 dixon 模块中的 KSY_precondition 函数对给定参数的预期行为是否符合预期
    assert dixon.KSY_precondition(m5) == True
# 定义一个测试函数，用于测试删除只包含零的行和列的方法
def test_delete_zero_rows_and_columns():
    """Tests method for deleting rows and columns containing only zeros."""
    # 导入符号变量 A, B, C
    A, B, C = symbols('A, B, C')

    # 创建矩阵 m1，包含部分元素为零的行和列
    m1 = Matrix([[0, 0],
                 [0, 0],
                 [1, 2]])

    # 创建矩阵 m2，包含部分元素为零的行和列
    m2 = Matrix([[0, 1, 2],
                 [0, 3, 4],
                 [0, 5, 6]])

    # 创建矩阵 m3，包含部分元素为零的行和列
    m3 = Matrix([[0, 0, 0, 0],
                 [0, 1, 2, 0],
                 [0, 3, 4, 0],
                 [0, 0, 0, 0]])

    # 创建矩阵 m4，包含部分元素为零的行和列
    m4 = Matrix([[1, 0, 2],
                 [0, 0, 0],
                 [3, 0, 4]])

    # 创建矩阵 m5，包含部分元素为零的行和列
    m5 = Matrix([[0, 0, 0, 1],
                 [0, 0, 0, 2],
                 [0, 0, 0, 3],
                 [0, 0, 0, 4]])

    # 创建矩阵 m6，包含部分元素为零的行和列，以及符号变量 A, B, C
    m6 = Matrix([[0, 0, A],
                 [B, 0, 0],
                 [0, 0, C]])

    # 断言调用 dixon 模块的 delete_zero_rows_and_columns 方法后的结果与期望值相等
    assert dixon.delete_zero_rows_and_columns(m1) == Matrix([[1, 2]])

    # 断言调用 dixon 模块的 delete_zero_rows_and_columns 方法后的结果与期望值相等
    assert dixon.delete_zero_rows_and_columns(m2) == Matrix([[1, 2],
                                                             [3, 4],
                                                             [5, 6]])

    # 断言调用 dixon 模块的 delete_zero_rows_and_columns 方法后的结果与期望值相等
    assert dixon.delete_zero_rows_and_columns(m3) == Matrix([[1, 2],
                                                             [3, 4]])

    # 断言调用 dixon 模块的 delete_zero_rows_and_columns 方法后的结果与期望值相等
    assert dixon.delete_zero_rows_and_columns(m4) == Matrix([[1, 2],
                                                             [3, 4]])

    # 断言调用 dixon 模块的 delete_zero_rows_and_columns 方法后的结果与期望值相等
    assert dixon.delete_zero_rows_and_columns(m5) == Matrix([[1],
                                                             [2],
                                                             [3],
                                                             [4]])

    # 断言调用 dixon 模块的 delete_zero_rows_and_columns 方法后的结果与期望值相等
    assert dixon.delete_zero_rows_and_columns(m6) == Matrix([[0, A],
                                                             [B, 0],
                                                             [0, C]])

# 定义一个测试函数，用于测试主导元素乘积方法
def test_product_leading_entries():
    """Tests product of leading entries method."""
    # 导入符号变量 A, B
    A, B = symbols('A, B')

    # 创建矩阵 m1，包含主导元素乘积的矩阵
    m1 = Matrix([[1, 2, 3],
                 [0, 4, 5],
                 [0, 0, 6]])

    # 创建矩阵 m2，包含主导元素乘积的矩阵
    m2 = Matrix([[0, 0, 1],
                 [2, 0, 3]])

    # 创建矩阵 m3，包含主导元素乘积的矩阵
    m3 = Matrix([[0, 0, 0],
                 [1, 2, 3],
                 [0, 0, 0]])

    # 创建矩阵 m4，包含主导元素乘积的矩阵，以及符号变量 A, B
    m4 = Matrix([[0, 0, A],
                 [1, 2, 3],
                 [B, 0, 0]])

    # 断言调用 dixon 模块的 product_leading_entries 方法后的结果与期望值相等
    assert dixon.product_leading_entries(m1) == 24
    # 断言调用 dixon 模块的 product_leading_entries 方法后的结果与期望值相等
    assert dixon.product_leading_entries(m2) == 2
    # 断言调用 dixon 模块的 product_leading_entries 方法后的结果与期望值相等
    assert dixon.product_leading_entries(m3) == 1
    # 断言调用 dixon 模块的 product_leading_entries 方法后的结果与期望值相等
    assert dixon.product_leading_entries(m4) == A * B

# 定义一个测试函数，用于测试 KSY Dixon 结果的示例一
def test_get_KSY_Dixon_resultant_example_one():
    """Tests the KSY Dixon resultant for example one"""
    # 导入符号变量 x, y, z
    x, y, z = symbols('x, y, z')

    # 定义多项式 p, q, h
    p = x * y * z
    q = x**2 - z**2
    h = x + y + z

    # 创建 DixonResultant 对象
    dixon = DixonResultant([p, q, h], [x, y])

    # 获取 Dixon 多项式
    dixon_poly = dixon.get_dixon_polynomial()

    # 获取 Dixon 矩阵
    dixon_matrix = dixon.get_dixon_matrix(dixon_poly)

    # 计算 KSY Dixon 结果
    D = dixon.get_KSY_Dixon_resultant(dixon_matrix)

    # 断言计算得到的结果与期望值相等
    assert D == -z**3

# 定义一个测试函数，用于测试 KSY Dixon 结果的示例二
def test_get_KSY_Dixon_resultant_example_two():
    """Tests the KSY Dixon resultant for example two"""
    # 导入符号变量 x, y, A
    x, y, A = symbols('x, y, A')
    # 计算表达式 p
    p = x * y + x * A + x - A**2 - A + y**2 + y
    # 计算表达式 q
    q = x**2 + x * A - x + x * y + y * A - y
    # 计算表达式 h
    h = x**2 + x * y + 2 * x - x * A - y * A - 2 * A

    # 创建 DixonResultant 对象，传入表达式列表 [p, q, h] 和变量列表 [x, y]
    dixon = DixonResultant([p, q, h], [x, y])
    # 调用对象方法获取 Dixon 多项式
    dixon_poly = dixon.get_dixon_polynomial()
    # 调用对象方法获取 Dixon 矩阵
    dixon_matrix = dixon.get_dixon_matrix(dixon_poly)
    # 调用对象方法获取 KSY Dixon 结果的因式分解
    D = factor(dixon.get_KSY_Dixon_resultant(dixon_matrix))

    # 使用断言检查 D 是否等于指定的结果
    assert D == -8*A*(A - 1)*(A + 2)*(2*A - 1)**2
def test_macaulay_resultant_init():
    """Test init method of MacaulayResultant."""

    # 检查初始化后的多项式列表是否正确
    assert macaulay.polynomials == [p, q]
    # 检查初始化后的变量列表是否正确
    assert macaulay.variables == [x, y]
    # 检查初始化后的多项式个数是否正确
    assert macaulay.n == 2
    # 检查每个多项式的次数是否正确
    assert macaulay.degrees == [1, 1]
    # 检查最大的多项式次数是否正确
    assert macaulay.degree_m == 1
    # 检查单项式集合的大小是否正确
    assert macaulay.monomials_size == 2

def test_get_degree_m():
    # 检查获取最大次数的方法是否返回正确的结果
    assert macaulay._get_degree_m() == 1

def test_get_size():
    # 检查获取多项式个数的方法是否返回正确的结果
    assert macaulay.get_size() == 2

def test_macaulay_example_one():
    """Tests the Macaulay for example from [Bruce97]_"""

    # 定义符号变量
    x, y, z = symbols('x, y, z')
    # 定义系数符号变量
    a_1_1, a_1_2, a_1_3 = symbols('a_1_1, a_1_2, a_1_3')
    a_2_2, a_2_3, a_3_3 = symbols('a_2_2, a_2_3, a_3_3')
    b_1_1, b_1_2, b_1_3 = symbols('b_1_1, b_1_2, b_1_3')
    b_2_2, b_2_3, b_3_3 = symbols('b_2_2, b_2_3, b_3_3')
    c_1, c_2, c_3 = symbols('c_1, c_2, c_3')

    # 定义多项式
    f_1 = a_1_1 * x ** 2 + a_1_2 * x * y + a_1_3 * x * z + \
          a_2_2 * y ** 2 + a_2_3 * y * z + a_3_3 * z ** 2
    f_2 = b_1_1 * x ** 2 + b_1_2 * x * y + b_1_3 * x * z + \
          b_2_2 * y ** 2 + b_2_3 * y * z + b_3_3 * z ** 2
    f_3 = c_1 * x + c_2 * y + c_3 * z

    # 创建 MacaulayResultant 对象
    mac = MacaulayResultant([f_1, f_2, f_3], [x, y, z])

    # 检查每个多项式的次数是否正确
    assert mac.degrees == [2, 2, 1]
    # 检查最大的多项式次数是否正确
    assert mac.degree_m == 3

    # 检查单项式集合是否包含正确的单项式
    assert mac.monomial_set == [x ** 3, x ** 2 * y, x ** 2 * z,
                                x * y ** 2,
                                x * y * z, x * z ** 2, y ** 3,
                                y ** 2 * z, y * z ** 2, z ** 3]
    # 检查单项式集合的大小是否正确
    assert mac.monomials_size == 10
    # 检查获取行系数的方法是否返回正确的结果
    assert mac.get_row_coefficients() == [[x, y, z], [x, y, z],
                                          [x * y, x * z, y * z, z ** 2]]

    # 获取 Macaulay 矩阵并检查其形状是否正确
    matrix = mac.get_matrix()
    assert matrix.shape == (mac.monomials_size, mac.monomials_size)
    # 检查获取子矩阵的方法是否返回正确的结果
    assert mac.get_submatrix(matrix) == Matrix([[a_1_1, a_2_2],
                                                [b_1_1, b_2_2]])

def test_macaulay_example_two():
    """Tests the Macaulay formulation for example from [Stiller96]_."""

    # 定义符号变量
    x, y, z = symbols('x, y, z')
    # 定义系数符号变量
    a_0, a_1, a_2 = symbols('a_0, a_1, a_2')
    b_0, b_1, b_2 = symbols('b_0, b_1, b_2')
    c_0, c_1, c_2, c_3, c_4 = symbols('c_0, c_1, c_2, c_3, c_4')

    # 定义多项式
    f = a_0 * y -  a_1 * x + a_2 * z
    g = b_1 * x ** 2 + b_0 * y ** 2 - b_2 * z ** 2
    h = c_0 * y - c_1 * x ** 3 + c_2 * x ** 2 * z - c_3 * x * z ** 2 + \
        c_4 * z ** 3

    # 创建 MacaulayResultant 对象
    mac = MacaulayResultant([f, g, h], [x, y, z])

    # 检查每个多项式的次数是否正确
    assert mac.degrees == [1, 2, 3]
    # 检查最大的多项式次数是否正确
    assert mac.degree_m == 4
    # 检查单项式集合的大小是否正确
    assert mac.monomials_size == 15
    # 检查获取行系数的方法是否返回正确的结果
    assert len(mac.get_row_coefficients()) == mac.n

    # 获取 Macaulay 矩阵并检查其形状是否正确
    matrix = mac.get_matrix()
    assert matrix.shape == (mac.monomials_size, mac.monomials_size)
    # 检查获取子矩阵的方法是否返回正确的结果
    assert mac.get_submatrix(matrix) == Matrix([[-a_1, a_0, a_2, 0],
                                                [0, -a_1, 0, 0],
                                                [0, 0, -a_1, 0],
                                                [0, 0, 0, -a_1]])
```