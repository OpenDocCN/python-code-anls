# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_dense.py`

```
# 导入用于测试的异常和断言的函数
from sympy.testing.pytest import raises

# 导入需要测试的符号数学库和矩阵模块
from sympy.polys import ZZ, QQ

# 导入双重对角矩阵（DDM）相关的矩阵操作函数
from sympy.polys.matrices.ddm import DDM

# 导入稠密矩阵相关的操作函数：
# 转置函数，加法、减法、取负、乘法、求逆、LU分解、LU分解的分步计算、LU分解后求解、伯克茨法
from sympy.polys.matrices.dense import (
        ddm_transpose,
        ddm_iadd, ddm_isub, ddm_ineg, ddm_imatmul, ddm_imul, ddm_irref,
        ddm_idet, ddm_iinv, ddm_ilu, ddm_ilu_split, ddm_ilu_solve, ddm_berk)

# 导入矩阵操作可能出现的异常
from sympy.polys.matrices.exceptions import (
    DMDomainError,              # 定义域错误
    DMNonInvertibleMatrixError, # 不可逆矩阵错误
    DMNonSquareMatrixError,     # 非方阵错误
    DMShapeError                # 矩阵形状错误
)


def test_ddm_transpose():
    a = [[1, 2], [3, 4]]
    # 测试矩阵转置函数，预期结果是转置后的矩阵
    assert ddm_transpose(a) == [[1, 3], [2, 4]]


def test_ddm_iadd():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    # 测试原地矩阵加法函数，预期结果是a矩阵被修改为原地相加后的结果
    ddm_iadd(a, b)
    assert a == [[6, 8], [10, 12]]


def test_ddm_isub():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    # 测试原地矩阵减法函数，预期结果是a矩阵被修改为原地相减后的结果
    ddm_isub(a, b)
    assert a == [[-4, -4], [-4, -4]]


def test_ddm_ineg():
    a = [[1, 2], [3, 4]]
    # 测试原地矩阵取负函数，预期结果是a矩阵被修改为取负后的结果
    ddm_ineg(a)
    assert a == [[-1, -2], [-3, -4]]


def test_ddm_matmul():
    a = [[1, 2], [3, 4]]
    # 测试原地矩阵标量乘法函数，预期结果是a矩阵被修改为标量乘法后的结果
    ddm_imul(a, 2)
    assert a == [[2, 4], [6, 8]]

    a = [[1, 2], [3, 4]]
    # 测试原地矩阵标量乘法函数，当乘数为0时的预期结果
    ddm_imul(a, 0)
    assert a == [[0, 0], [0, 0]]


def test_ddm_imatmul():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[1, 2], [3, 4], [5, 6]]

    c1 = [[0, 0], [0, 0]]
    # 测试原地矩阵乘法函数，预期结果是c1矩阵被修改为a矩阵和b矩阵乘法后的结果
    ddm_imatmul(c1, a, b)
    assert c1 == [[22, 28], [49, 64]]

    c2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # 测试原地矩阵乘法函数，预期结果是c2矩阵被修改为b矩阵和a矩阵乘法后的结果
    ddm_imatmul(c2, b, a)
    assert c2 == [[9, 12, 15], [19, 26, 33], [29, 40, 51]]

    b3 = [[1], [2], [3]]
    c3 = [[0], [0]]
    # 测试原地矩阵乘法函数，预期结果是c3矩阵被修改为a矩阵和b3矩阵乘法后的结果
    ddm_imatmul(c3, a, b3)
    assert c3 == [[14], [32]]


def test_ddm_irref():
    # 空矩阵
    A = []
    Ar = []
    pivots = []
    # 测试原地矩阵行阶梯形式函数，预期结果是返回主元列索引列表pivots，并且A矩阵被修改为行阶梯形式后的结果Ar
    assert ddm_irref(A) == pivots
    assert A == Ar

    # 标准方阵情况
    A = [[QQ(0), QQ(1)], [QQ(1), QQ(1)]]
    Ar = [[QQ(1), QQ(0)], [QQ(0), QQ(1)]]
    pivots = [0, 1]
    # 测试原地矩阵行阶梯形式函数，预期结果是返回主元列索引列表pivots，并且A矩阵被修改为行阶梯形式后的结果Ar
    assert ddm_irref(A) == pivots
    assert A == Ar

    # m < n  情况
    A = [[QQ(1), QQ(2), QQ(1)], [QQ(3), QQ(4), QQ(1)]]
    Ar = [[QQ(1), QQ(0), QQ(-1)], [QQ(0), QQ(1), QQ(1)]]
    pivots = [0, 1]
    # 测试原地矩阵行阶梯形式函数，预期结果是返回主元列索引列表pivots，并且A矩阵被修改为行阶梯形式后的结果Ar
    assert ddm_irref(A) == pivots
    assert A == Ar

    # m < n  但反向情况
    A = [[QQ(3), QQ(4), QQ(1)], [QQ(1), QQ(2), QQ(1)]]
    Ar = [[QQ(1), QQ(0), QQ(-1)], [QQ(0), QQ(1), QQ(1)]]
    pivots = [0, 1]
    # 测试原地矩阵行阶梯形式函数，预期结果是返回主元列索引列表pivots，并且A矩阵被修改为行阶梯形式后的结果Ar
    assert ddm_irref(A) == pivots
    assert A == Ar

    # m > n 情况
    A = [[QQ(1), QQ(0)], [QQ(1), QQ(3)], [QQ(0), QQ(1)]]
    Ar = [[QQ(1), QQ(0)], [QQ(0), QQ(1)], [QQ(0), QQ(0)]]
    pivots = [0, 1]
    # 测试原地矩阵行阶梯形式函数，预期结果是返回主元列索引列表pivots，并且A矩阵被修改为行阶梯形式后的结果Ar
    assert ddm_irref(A) == pivots
    assert A == Ar

    # 举例说明存在缺失主元的情况
    A = [[QQ(1), QQ(0), QQ(1)], [QQ(3), QQ(0), QQ(1)]]
    Ar = [[QQ(1), QQ(0),
    # 创建一个包含单个元素 ZZ(2) 的二维列表 A
    A = [[ZZ(2)]]
    # 使用 ddm_idet 函数计算 A 的行列式，期望结果是 ZZ(2)，进行断言
    assert ddm_idet(A, ZZ) == ZZ(2)
    
    # 创建一个 2x2 的整数矩阵 A
    A = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    # 使用 ddm_idet 函数计算 A 的行列式，期望结果是 ZZ(-2)，进行断言
    assert ddm_idet(A, ZZ) == ZZ(-2)
    
    # 创建一个 3x3 的整数矩阵 A
    A = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(3), ZZ(5)]]
    # 使用 ddm_idet 函数计算 A 的行列式，期望结果是 ZZ(-1)，进行断言
    assert ddm_idet(A, ZZ) == ZZ(-1)
    
    # 创建一个 3x3 的整数矩阵 A
    A = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(2), ZZ(5)]]
    # 使用 ddm_idet 函数计算 A 的行列式，期望结果是 ZZ(0)，进行断言
    assert ddm_idet(A, ZZ) == ZZ(0)
    
    # 创建一个 2x2 的有理数矩阵 A
    A = [[QQ(1, 2), QQ(1, 2)], [QQ(1, 3), QQ(1, 4)]]
    # 使用 ddm_idet 函数计算 A 的行列式，期望结果是 QQ(-1/24)，进行断言
    assert ddm_idet(A, QQ) == QQ(-1, 24)
# 定义一个函数用于测试 DDM 的逆运算功能
def test_ddm_inv():
    # 初始化空列表 A 和 Ainv
    A = []
    Ainv = []
    # 调用 ddm_iinv 函数，期望 Ainv 不变
    ddm_iinv(Ainv, A, QQ)
    # 断言 Ainv 等于 A
    assert Ainv == A

    # 再次初始化 A 和 Ainv
    A = []
    Ainv = []
    # 调用 ddm_iinv 函数，期望引发 DMDomainError 异常
    raises(DMDomainError, lambda: ddm_iinv(Ainv, A, ZZ))

    # 初始化 A 为一个非方阵
    A = [[QQ(1), QQ(2)]]
    Ainv = [[QQ(0), QQ(0)]]
    # 调用 ddm_iinv 函数，期望引发 DMNonSquareMatrixError 异常
    raises(DMNonSquareMatrixError, lambda: ddm_iinv(Ainv, A, QQ))

    # 初始化 A 和 Ainv，A 是一个可逆矩阵
    A = [[QQ(1, 1), QQ(2, 1)], [QQ(3, 1), QQ(4, 1)]]
    Ainv = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    # 期望计算得到的 Ainv 与预期结果 Ainv_expected 相等
    Ainv_expected = [[QQ(-2, 1), QQ(1, 1)], [QQ(3, 2), QQ(-1, 2)]]
    ddm_iinv(Ainv, A, QQ)
    assert Ainv == Ainv_expected

    # 初始化 A 和 Ainv，A 是一个不可逆矩阵
    A = [[QQ(1, 1), QQ(2, 1)], [QQ(2, 1), QQ(4, 1)]]
    Ainv = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    # 调用 ddm_iinv 函数，期望引发 DMNonInvertibleMatrixError 异常
    raises(DMNonInvertibleMatrixError, lambda: ddm_iinv(Ainv, A, QQ))


# 定义一个函数用于测试 DDM 的 LU 分解功能
def test_ddm_ilu():
    # 初始化空列表 A 和 Alu
    A = []
    Alu = []
    # 调用 ddm_ilu 函数，期望 A 不变，返回 swaps 为空列表
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []

    # 初始化 A 和 Alu 为空子列表
    A = [[]]
    Alu = [[]]
    # 调用 ddm_ilu 函数，期望 A 和 Alu 都不变，返回 swaps 为空列表
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []

    # 初始化 A 为一个二阶矩阵
    A = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    Alu = [[QQ(1), QQ(2)], [QQ(3), QQ(-2)]]
    # 调用 ddm_ilu 函数，进行 LU 分解，期望 A 变为 Alu，返回 swaps 为空列表
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []

    # 初始化 A 为一个具有交换行的二阶矩阵
    A = [[QQ(0), QQ(2)], [QQ(3), QQ(4)]]
    Alu = [[QQ(3), QQ(4)], [QQ(0), QQ(2)]]
    # 调用 ddm_ilu 函数，进行 LU 分解，期望 A 变为 Alu，返回 swaps 包含 (0, 1)
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == [(0, 1)]

    # 初始化 A 为一个三阶矩阵
    A = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)], [QQ(7), QQ(8), QQ(9)]]
    Alu = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(-3), QQ(-6)], [QQ(7), QQ(2), QQ(0)]]
    # 调用 ddm_ilu 函数，进行 LU 分解，期望 A 变为 Alu，返回 swaps 为空列表
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []

    # 初始化 A 为一个具有交换行的三阶矩阵
    A = [[QQ(0), QQ(1), QQ(2)], [QQ(0), QQ(1), QQ(3)], [QQ(1), QQ(1), QQ(2)]]
    Alu = [[QQ(1), QQ(1), QQ(2)], [QQ(0), QQ(1), QQ(3)], [QQ(0), QQ(1), QQ(-1)]]
    # 调用 ddm_ilu 函数，进行 LU 分解，期望 A 变为 Alu，返回 swaps 包含 (0, 2)
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == [(0, 2)]

    # 初始化 A 为一个具有交换行的二阶矩阵
    A = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]]
    Alu = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(-3), QQ(-6)]]
    # 调用 ddm_ilu 函数，进行 LU 分解，期望 A 变为 Alu，返回 swaps 为空列表
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []

    # 初始化 A 为一个具有交换行的三阶矩阵
    A = [[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]]
    Alu = [[QQ(1), QQ(2)], [QQ(3), QQ(-2)], [QQ(5), QQ(2)]]
    # 调用 ddm_ilu 函数，进行 LU 分解，期望 A 变为 Alu，返回 swaps 为空列表
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []


# 定义一个函数用于测试 DDM 的 LU 分解并返回分解结果的功能
def test_ddm_ilu_split():
    # 初始化空列表 U, L, Uexp, Lexp
    U = []
    L = []
    Uexp = []
    Lexp = []
    # 调用 ddm_ilu_split 函数，进行 LU 分解并返回分解结果
    swaps = ddm_ilu_split(L, U, QQ)
    # 断言 U 和 Uexp 相等，L 和 Lexp 相等，swaps 为空列表
    assert U == Uexp
    assert L == Lexp
    assert swaps == []

    # 初始化 U 和 L 为一个空子列表
    U = [[]]
    L = [[QQ(1)]]
    Uexp = [[]]
    Lexp = [[QQ(1)]]
    # 调用 ddm_ilu_split 函数，进行 LU 分解并返回分解结果
    swaps = ddm_ilu_split(L, U, QQ)
    # 断言 U 和 Uexp 相等，L 和 Lexp 相等，swaps 为空列表
    assert U == Uexp
    assert L == Lexp
    assert swaps == []

    # 初始化 U 和 L 为具有初始值的矩阵
    U = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    L = [[QQ(1), QQ(0)], [QQ(0), QQ(1)]]
    Uexp = [[QQ(1), QQ(2)], [QQ(0), QQ(-2)]]
    Lexp = [[QQ(1), QQ(0)], [QQ(3), QQ(1)]]
    # 调用 ddm_ilu_split 函数，进行 LU 分解并返回分解结果
    swaps = ddm_ilu_split(L, U, QQ)
    # 断言 U 和 Uexp 相等，L 和 Lexp 相等，swaps 为空列表
    assert U == Uexp
    assert L == Lexp
    assert swaps == []

    # 初始化 U 和 L 为具有不
    # 初始化一个上三角矩阵 U，其中元素为有理数 QQ 类型
    U = [[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]]
    # 初始化一个下三角矩阵 L，其中元素为有理数 QQ 类型
    L = [[QQ(1), QQ(0), QQ(0)], [QQ(0), QQ(1), QQ(0)], [QQ(0), QQ(0), QQ(1)]]
    # 预期的上三角矩阵 Uexp，用于比较结果
    Uexp = [[QQ(1), QQ(2)], [QQ(0), QQ(-2)], [QQ(0), QQ(0)]]
    # 预期的下三角矩阵 Lexp，用于比较结果
    Lexp = [[QQ(1), QQ(0), QQ(0)], [QQ(3), QQ(1), QQ(0)], [QQ(5), QQ(2), QQ(1)]]
    # 使用 ddm_ilu_split 函数计算 L 和 U 的因子分解，并返回交换操作列表
    swaps = ddm_ilu_split(L, U, QQ)
    # 断言 U 等于预期的 Uexp
    assert U == Uexp
    # 断言 L 等于预期的 Lexp
    assert L == Lexp
    # 断言 swaps 为空列表，表示没有交换操作
    assert swaps == []
def test_ddm_ilu_solve():
    # Basic example
    # 设置上三角矩阵 U 和下三角矩阵 L
    U = [[QQ(1), QQ(2)], [QQ(0), QQ(-2)]]
    L = [[QQ(1), QQ(0)], [QQ(3), QQ(1)]]
    # 初始化交换列表为空
    swaps = []
    # 设置向量 b 和解向量 x 的初始值
    b = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DDM([[QQ(0)], [QQ(0)]], (2, 1), QQ)
    # 预期的解向量
    xexp = DDM([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    # 调用 ddm_ilu_solve 函数求解
    ddm_ilu_solve(x, L, U, swaps, b)
    # 断言解向量与预期结果相等
    assert x == xexp

    # Example with swaps
    # 设置上三角矩阵 U 和下三角矩阵 L，包含交换信息
    U = [[QQ(3), QQ(4)], [QQ(0), QQ(2)]]
    L = [[QQ(1), QQ(0)], [QQ(0), QQ(1)]]
    # 初始化交换列表
    swaps = [(0, 1)]
    # 设置向量 b 和解向量 x 的初始值
    b = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DDM([[QQ(0)], [QQ(0)]], (2, 1), QQ)
    # 预期的解向量
    xexp = DDM([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    # 调用 ddm_ilu_solve 函数求解
    ddm_ilu_solve(x, L, U, swaps, b)
    # 断言解向量与预期结果相等
    assert x == xexp

    # Overdetermined, consistent
    # 设置上三角矩阵 U 和下三角矩阵 L
    U = [[QQ(1), QQ(2)], [QQ(0), QQ(-2)], [QQ(0), QQ(0)]]
    L = [[QQ(1), QQ(0), QQ(0)], [QQ(3), QQ(1), QQ(0)], [QQ(5), QQ(2), QQ(1)]]
    # 初始化交换列表为空
    swaps = []
    # 设置向量 b 和解向量 x 的初始值
    b = DDM([[QQ(1)], [QQ(2)], [QQ(3)]], (3, 1), QQ)
    x = DDM([[QQ(0)], [QQ(0)]], (2, 1), QQ)
    # 预期的解向量
    xexp = DDM([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    # 调用 ddm_ilu_solve 函数求解
    ddm_ilu_solve(x, L, U, swaps, b)
    # 断言解向量与预期结果相等
    assert x == xexp

    # Overdetermined, inconsistent
    # 设置向量 b 中的值使方程组不一致
    b = DDM([[QQ(1)], [QQ(2)], [QQ(4)]], (3, 1), QQ)
    # 断言调用 ddm_ilu_solve 函数会引发 DMNonInvertibleMatrixError 异常
    raises(DMNonInvertibleMatrixError, lambda: ddm_ilu_solve(x, L, U, swaps, b))

    # Square, noninvertible
    # 设置上三角矩阵 U 和下三角矩阵 L，使得矩阵不可逆
    U = [[QQ(1), QQ(2)], [QQ(0), QQ(0)]]
    L = [[QQ(1), QQ(0)], [QQ(1), QQ(1)]]
    # 初始化交换列表为空
    swaps = []
    # 设置向量 b 和解向量 x 的初始值
    b = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    # 断言调用 ddm_ilu_solve 函数会引发 DMNonInvertibleMatrixError 异常
    raises(DMNonInvertibleMatrixError, lambda: ddm_ilu_solve(x, L, U, swaps, b))

    # Underdetermined
    # 设置上三角矩阵 U 和下三角矩阵 L，使得方程组欠定
    U = [[QQ(1), QQ(2)]]
    L = [[QQ(1)]]
    # 初始化交换列表为空
    swaps = []
    # 设置向量 b 的初始值
    b = DDM([[QQ(3)]], (1, 1), QQ)
    # 断言调用 ddm_ilu_solve 函数会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: ddm_ilu_solve(x, L, U, swaps, b))

    # Shape mismatch
    # 设置向量 b 的维度与解向量 x 不匹配
    b3 = DDM([[QQ(1)], [QQ(2)], [QQ(3)]], (3, 1), QQ)
    # 断言调用 ddm_ilu_solve 函数会引发 DMShapeError 异常
    raises(DMShapeError, lambda: ddm_ilu_solve(x, L, U, swaps, b3))

    # Empty shape mismatch
    # 设置空的上三角矩阵 U 和下三角矩阵 L
    U = [[QQ(1)]]
    L = [[QQ(1)]]
    # 初始化交换列表为空
    swaps = []
    # 设置解向量 x 的初始值
    x = [[QQ(1)]]
    # 设置向量 b 为空列表
    b = []
    # 断言调用 ddm_ilu_solve 函数会引发 DMShapeError 异常
    raises(DMShapeError, lambda: ddm_ilu_solve(x, L, U, swaps, b))

    # Empty system
    # 设置空的上三角矩阵 U 和下三角矩阵 L
    U = []
    L = []
    # 初始化交换列表为空
    swaps = []
    # 设置向量 b 和解向量 x 为空列表
    b = []
    x = []
    # 调用 ddm_ilu_solve 函数求解
    ddm_ilu_solve(x, L, U, swaps, b)
    # 断言解向量 x 为空列表
    assert x == []
```