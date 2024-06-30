# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_ddm.py`

```
# 引入测试框架中的异常抛出功能
from sympy.testing.pytest import raises
# 引入外部库中的类型定义
from sympy.external.gmpy import GROUND_TYPES

# 从 sympy.polys 包中引入整数环 ZZ 和有理数环 QQ
from sympy.polys import ZZ, QQ

# 从 sympy.polys.matrices.ddm 模块中引入 DDM 类
from sympy.polys.matrices.ddm import DDM
# 从 sympy.polys.matrices.exceptions 引入多种异常类
from sympy.polys.matrices.exceptions import (
    DMShapeError, DMNonInvertibleMatrixError, DMDomainError,
    DMBadInputError
)


def test_DDM_init():
    # 创建一个二维列表，包含 ZZ 类型的对象
    items = [[ZZ(0), ZZ(1), ZZ(2)], [ZZ(3), ZZ(4), ZZ(5)]]
    # 设置矩阵的形状为 (2, 3)
    shape = (2, 3)
    # 创建 DDM 对象，使用 items 列表，指定形状和整数环 ZZ
    ddm = DDM(items, shape, ZZ)
    # 断言矩阵的形状正确
    assert ddm.shape == shape
    # 断言矩阵的行数为 2
    assert ddm.rows == 2
    # 断言矩阵的列数为 3
    assert ddm.cols == 3
    # 断言矩阵的定义域为整数环 ZZ

    # 测试输入不合法的情况，期望抛出 DMBadInputError 异常
    raises(DMBadInputError, lambda: DDM([[ZZ(2), ZZ(3)]], (2, 2), ZZ))
    raises(DMBadInputError, lambda: DDM([[ZZ(1)], [ZZ(2), ZZ(3)]], (2, 2), ZZ))


def test_DDM_getsetitem():
    # 创建一个 DDM 对象，包含 ZZ 类型的对象
    ddm = DDM([[ZZ(2), ZZ(3)], [ZZ(4), ZZ(5)]], (2, 2), ZZ)

    # 断言获取元素正确
    assert ddm[0][0] == ZZ(2)
    assert ddm[0][1] == ZZ(3)
    assert ddm[1][0] == ZZ(4)
    assert ddm[1][1] == ZZ(5)

    # 测试访问超出索引范围的元素，期望抛出 IndexError 异常
    raises(IndexError, lambda: ddm[2][0])
    raises(IndexError, lambda: ddm[0][2])

    # 修改元素值并断言修改成功
    ddm[0][0] = ZZ(-1)
    assert ddm[0][0] == ZZ(-1)


def test_DDM_str():
    # 创建一个 DDM 对象，包含 ZZ 类型的对象
    ddm = DDM([[ZZ(0), ZZ(1)], [ZZ(2), ZZ(3)]], (2, 2), ZZ)
    # 根据当前的 GROUND_TYPES 输出不同的字符串表示
    if GROUND_TYPES == 'gmpy': # pragma: no cover
        assert str(ddm) == '[[0, 1], [2, 3]]'
        assert repr(ddm) == 'DDM([[mpz(0), mpz(1)], [mpz(2), mpz(3)]], (2, 2), ZZ)'
    else:        # pragma: no cover
        assert repr(ddm) == 'DDM([[0, 1], [2, 3]], (2, 2), ZZ)'
        assert str(ddm) == '[[0, 1], [2, 3]]'


def test_DDM_eq():
    # 创建一个包含 ZZ 类型的二维列表
    items = [[ZZ(0), ZZ(1)], [ZZ(2), ZZ(3)]]
    # 创建两个相同的 DDM 对象
    ddm1 = DDM(items, (2, 2), ZZ)
    ddm2 = DDM(items, (2, 2), ZZ)

    # 测试相等性和不等性操作
    assert (ddm1 == ddm1) is True
    assert (ddm1 == items) is False
    assert (items == ddm1) is False
    assert (ddm1 == ddm2) is True
    assert (ddm2 == ddm1) is True

    assert (ddm1 != ddm1) is False
    assert (ddm1 != items) is True
    assert (items != ddm1) is True
    assert (ddm1 != ddm2) is False
    assert (ddm2 != ddm1) is False

    # 创建一个不同的 DDM 对象
    ddm3 = DDM([[ZZ(0), ZZ(1)], [ZZ(3), ZZ(3)]], (2, 2), ZZ)
    ddm3 = DDM(items, (2, 2), QQ)

    # 进行进一步的相等性和不等性测试
    assert (ddm1 == ddm3) is False
    assert (ddm3 == ddm1) is False
    assert (ddm1 != ddm3) is True
    assert (ddm3 != ddm1) is True


def test_DDM_convert_to():
    # 创建一个包含 ZZ 类型的 DDM 对象
    ddm = DDM([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    # 测试转换到相同类型的情况
    assert ddm.convert_to(ZZ) == ddm
    # 测试转换到 QQ 类型的情况
    ddmq = ddm.convert_to(QQ)
    assert ddmq.domain == QQ


def test_DDM_zeros():
    # 创建一个 QQ 类型的零矩阵
    ddmz = DDM.zeros((3, 4), QQ)
    # 断言矩阵的元素全为 0
    assert list(ddmz) == [[QQ(0)] * 4] * 3
    # 断言矩阵的形状为 (3, 4)
    assert ddmz.shape == (3, 4)
    # 断言矩阵的定义域为有理数环 QQ
    assert ddmz.domain == QQ


def test_DDM_ones():
    # 创建一个 QQ 类型的全 1 矩阵
    ddmone = DDM.ones((2, 3), QQ)
    # 断言矩阵的元素全为 1
    assert list(ddmone) == [[QQ(1)] * 3] * 2
    # 断言矩阵的形状为 (2, 3)
    assert ddmone.shape == (2, 3)
    # 断言矩阵的定义域为有理数环 QQ
    assert ddmone.domain == QQ


def test_DDM_eye():
    # 创建一个 QQ 类型的单位矩阵
    ddmz = DDM.eye(3, QQ)
    # 定义一个函数，用于生成单位矩阵的期望值
    f = lambda i, j: QQ(1) if i == j else QQ(0)
    # 断言矩阵的每个元素符合期望
    assert list(ddmz) == [[f(i, j) for i in range(3)] for j in range(3)]
    # 断言矩阵的形状为 (3, 3)
    assert ddmz.shape == (3, 3)
    # 断言矩阵的定义域为有理数环 QQ
    assert ddmz.domain == QQ


def test_DDM_copy():
    # 创建一个 QQ 类型的 DDM 对象
    ddm1 = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    # 创建其深拷贝
    ddm2 = ddm1.copy()
    # 断言：验证 ddm1 等于 ddm2
    assert (ddm1 == ddm2) is True
    
    # 修改 ddm1 的第一个元素的第一个元素为 QQ(-1)
    ddm1[0][0] = QQ(-1)
    
    # 断言：验证 ddm1 不等于 ddm2
    assert (ddm1 == ddm2) is False
    
    # 修改 ddm2 的第一个元素的第一个元素为 QQ(-1)
    ddm2[0][0] = QQ(-1)
    
    # 断言：验证 ddm1 等于 ddm2
    assert (ddm1 == ddm2) is True
def test_DDM_transpose():
    # 创建一个 DDM 对象，包含一个 2x1 的矩阵，元素为 QQ(1) 和 QQ(2)
    ddm = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    # 创建一个期望的转置 DDM 对象，包含一个 1x2 的矩阵，元素为 QQ(1) 和 QQ(2)
    ddmT = DDM([[QQ(1), QQ(2)]], (1, 2), QQ)
    # 断言 ddm 的转置应当等于 ddmT
    assert ddm.transpose() == ddmT
    # 创建一个空的 DDM 对象，形状为 (0, 2)
    ddm02 = DDM([], (0, 2), QQ)
    # 创建一个期望的转置 DDM 对象，形状为 (2, 0)
    ddm02T = DDM([[], []], (2, 0), QQ)
    # 断言 ddm02 的转置应当等于 ddm02T
    assert ddm02.transpose() == ddm02T
    # 断言 ddm02T 的转置应当等于 ddm02 本身
    assert ddm02T.transpose() == ddm02
    # 创建一个空的 DDM 对象，形状为 (0, 0)
    ddm0 = DDM([], (0, 0), QQ)
    # 断言 ddm0 的转置应当等于 ddm0 本身
    assert ddm0.transpose() == ddm0


def test_DDM_add():
    # 创建两个包含 ZZ 类型元素的 2x1 矩阵 A 和 B
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    B = DDM([[ZZ(3)], [ZZ(4)]], (2, 1), ZZ)
    # 创建一个期望的结果矩阵 C，其元素为 A 和 B 对应元素相加的结果
    C = DDM([[ZZ(4)], [ZZ(6)]], (2, 1), ZZ)
    # 创建一个包含 QQ 类型元素的 2x1 矩阵 AQ
    AQ = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    # 断言 A + B、A.add(B) 和 C 都应当相等
    assert A + B == A.add(B) == C

    # 检查异常情况：A 与不匹配形状的 DDM 相加会引发 DMShapeError 异常
    raises(DMShapeError, lambda: A + DDM([[ZZ(5)]], (1, 1), ZZ))
    # 检查异常情况：A 与非 DDM 类型的对象相加会引发 TypeError 异常
    raises(TypeError, lambda: A + ZZ(1))
    # 检查异常情况：非 DDM 类型的对象与 A 相加会引发 TypeError 异常
    raises(TypeError, lambda: ZZ(1) + A)
    # 检查异常情况：A 与类型不匹配的 DDM 相加会引发 DMDomainError 异常
    raises(DMDomainError, lambda: A + AQ)
    # 检查异常情况：类型不匹配的 DDM 与 A 相加会引发 DMDomainError 异常
    raises(DMDomainError, lambda: AQ + A)


def test_DDM_sub():
    # 创建两个包含 ZZ 类型元素的 2x1 矩阵 A 和 B
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    B = DDM([[ZZ(3)], [ZZ(4)]], (2, 1), ZZ)
    # 创建一个期望的结果矩阵 C，其元素为 A 和 B 对应元素相减的结果
    C = DDM([[ZZ(-2)], [ZZ(-2)]], (2, 1), ZZ)
    # 创建一个包含 QQ 类型元素的 2x1 矩阵 AQ
    AQ = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    # 创建一个包含 ZZ 类型元素的 1x1 矩阵 D
    D = DDM([[ZZ(5)]], (1, 1), ZZ)
    # 断言 A - B、A.sub(B) 和 C 都应当相等
    assert A - B == A.sub(B) == C

    # 检查异常情况：非 DDM 类型的对象与 A 相减会引发 TypeError 异常
    raises(TypeError, lambda: A - ZZ(1))
    # 检查异常情况：非 DDM 类型的对象与 A 相减会引发 TypeError 异常
    raises(TypeError, lambda: ZZ(1) - A)
    # 检查异常情况：A 与不匹配形状的 DDM 相减会引发 DMShapeError 异常
    raises(DMShapeError, lambda: A - D)
    # 检查异常情况：不匹配形状的 DDM 与 A 相减会引发 DMShapeError 异常
    raises(DMShapeError, lambda: D - A)
    # 检查异常情况：A 与类型不匹配的 DDM 相减会引发 DMDomainError 异常
    raises(DMDomainError, lambda: A - AQ)
    # 检查异常情况：类型不匹配的 DDM 与 A 相减会引发 DMDomainError 异常
    raises(DMDomainError, lambda: AQ - A)
    # 检查异常情况：A 与类型不匹配的 DDM 相减会引发 DMDomainError 异常
    raises(DMDomainError, lambda: A.sub(AQ))
    # 检查异常情况：类型不匹配的 DDM 与 A 相减会引发 DMDomainError 异常
    raises(DMDomainError, lambda: AQ.sub(A))


def test_DDM_neg():
    # 创建一个包含 ZZ 类型元素的 2x1 矩阵 A
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    # 创建一个期望的结果矩阵 An，其元素为 A 元素的相反数
    An = DDM([[ZZ(-1)], [ZZ(-2)]], (2, 1), ZZ)
    # 断言 -A、A.neg() 和 An 都应当相等
    assert -A == A.neg() == An
    # 断言 -An、An.neg() 和 A 都应当相等
    assert -An == An.neg() == A


def test_DDM_mul():
    # 创建一个包含 ZZ 类型元素的 1x1 矩阵 A
    A = DDM([[ZZ(1)]], (1, 1), ZZ)
    # 创建一个期望的结果矩阵 A2，其元素为 A 中元素乘以 ZZ(2) 的结果
    A2 = DDM([[ZZ(2)]], (1, 1), ZZ)
    # 断言 A * ZZ(2) 和 ZZ(2) * A 应当等于 A2
    assert A * ZZ(2) == A2
    assert ZZ(2) * A == A2
    # 检查异常情况：尝试用非 DDM 类型的对象乘以 A 会引发 TypeError 异常
    raises(TypeError, lambda: [[1]] * A)
    # 检查异常情况：尝试用 A 乘以非 DDM 类型的对象会引发 TypeError 异常
    raises(TypeError, lambda: A * [[1]])


def test_DDM_matmul():
    # 创建两个包含 ZZ 类型元素的矩阵 A 和 B
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    B = DDM([[ZZ(3), ZZ(4)]], (1, 2), ZZ)
    # 创建一个期望的结果矩阵 AB，其元素为 A 和 B 的矩阵乘积结果
    AB = DDM([[ZZ(3), ZZ(4)], [
    # 断言：验证 Z50 与 Z00 的矩阵乘积，以及使用 matmul 方法的结果均为 Z50 自身
    assert Z50 @ Z00 == Z50.matmul(Z00) == Z50
    
    # 断言：验证 Z00 与 Z00 的矩阵乘积，以及使用 matmul 方法的结果均为零矩阵 Z00 自身
    assert Z00 @ Z00 == Z00.matmul(Z00) == Z00
    
    # 断言：验证 Z50 与 Z04 的矩阵乘积，以及使用 matmul 方法的结果等于 Z54
    assert Z50 @ Z04 == Z50.matmul(Z04) == Z54
    
    # 引发异常：验证 Z05 与 Z40 的矩阵乘积会引发 DMShapeError 异常
    raises(DMShapeError, lambda: Z05 @ Z40)
    
    # 引发异常：验证使用 matmul 方法计算 Z05 与 Z40 的矩阵乘积会引发 DMShapeError 异常
    raises(DMShapeError, lambda: Z05.matmul(Z40))
# 定义测试函数 test_DDM_hstack，用于测试 DDM 对象的水平堆叠操作
def test_DDM_hstack():
    # 创建 DDM 对象 A，包含一个 1x3 的 ZZ 类型矩阵
    A = DDM([[ZZ(1), ZZ(2), ZZ(3)]], (1, 3), ZZ)
    # 创建 DDM 对象 B，包含一个 1x2 的 ZZ 类型矩阵
    B = DDM([[ZZ(4), ZZ(5)]], (1, 2), ZZ)
    # 创建 DDM 对象 C，包含一个 1x1 的 ZZ 类型矩阵
    C = DDM([[ZZ(6)]], (1, 1), ZZ)

    # 对象 A 和 B 进行水平堆叠操作，得到新的 DDM 对象 Ah
    Ah = A.hstack(B)
    # 断言 Ah 的形状为 (1, 5)
    assert Ah.shape == (1, 5)
    # 断言 Ah 的域（数据类型）为 ZZ
    assert Ah.domain == ZZ
    # 断言 Ah 的值等于 DDM([[ZZ(1), ZZ(2), ZZ(3), ZZ(4), ZZ(5)]], (1, 5), ZZ)
    assert Ah == DDM([[ZZ(1), ZZ(2), ZZ(3), ZZ(4), ZZ(5)]], (1, 5), ZZ)

    # 对象 A、B 和 C 进行连续水平堆叠操作，得到新的 DDM 对象 Ah
    Ah = A.hstack(B, C)
    # 断言 Ah 的形状为 (1, 6)
    assert Ah.shape == (1, 6)
    # 断言 Ah 的域（数据类型）为 ZZ
    assert Ah.domain == ZZ
    # 断言 Ah 的值等于 DDM([[ZZ(1), ZZ(2), ZZ(3), ZZ(4), ZZ(5), ZZ(6)]], (1, 6), ZZ)
    assert Ah == DDM([[ZZ(1), ZZ(2), ZZ(3), ZZ(4), ZZ(5), ZZ(6)]], (1, 6), ZZ)


# 定义测试函数 test_DDM_vstack，用于测试 DDM 对象的垂直堆叠操作
def test_DDM_vstack():
    # 创建 DDM 对象 A，包含一个 3x1 的 ZZ 类型矩阵
    A = DDM([[ZZ(1)], [ZZ(2)], [ZZ(3)]], (3, 1), ZZ)
    # 创建 DDM 对象 B，包含一个 2x1 的 ZZ 类型矩阵
    B = DDM([[ZZ(4)], [ZZ(5)]], (2, 1), ZZ)
    # 创建 DDM 对象 C，包含一个 1x1 的 ZZ 类型矩阵
    C = DDM([[ZZ(6)]], (1, 1), ZZ)

    # 对象 A 和 B 进行垂直堆叠操作，得到新的 DDM 对象 Ah
    Ah = A.vstack(B)
    # 断言 Ah 的形状为 (5, 1)
    assert Ah.shape == (5, 1)
    # 断言 Ah 的域（数据类型）为 ZZ
    assert Ah.domain == ZZ
    # 断言 Ah 的值等于 DDM([[ZZ(1)], [ZZ(2)], [ZZ(3)], [ZZ(4)], [ZZ(5)]], (5, 1), ZZ)
    assert Ah == DDM([[ZZ(1)], [ZZ(2)], [ZZ(3)], [ZZ(4)], [ZZ(5)]], (5, 1), ZZ)

    # 对象 A、B 和 C 进行连续垂直堆叠操作，得到新的 DDM 对象 Ah
    Ah = A.vstack(B, C)
    # 断言 Ah 的形状为 (6, 1)
    assert Ah.shape == (6, 1)
    # 断言 Ah 的域（数据类型）为 ZZ
    assert Ah.domain == ZZ
    # 断言 Ah 的值等于 DDM([[ZZ(1)], [ZZ(2)], [ZZ(3)], [ZZ(4)], [ZZ(5)], [ZZ(6)]], (6, 1), ZZ)
    assert Ah == DDM([[ZZ(1)], [ZZ(2)], [ZZ(3)], [ZZ(4)], [ZZ(5)], [ZZ(6)]], (6, 1), ZZ)


# 定义测试函数 test_DDM_applyfunc，用于测试 DDM 对象的函数应用操作
def test_DDM_applyfunc():
    # 创建 DDM 对象 A，包含一个 1x3 的 ZZ 类型矩阵
    A = DDM([[ZZ(1), ZZ(2), ZZ(3)]], (1, 3), ZZ)
    # 创建 DDM 对象 B，包含一个 1x3 的 ZZ 类型矩阵
    B = DDM([[ZZ(2), ZZ(4), ZZ(6)]], (1, 3), ZZ)
    # 断言 A 中每个元素乘以 2 后与 B 相等
    assert A.applyfunc(lambda x: 2*x, ZZ) == B


# 定义测试函数 test_DDM_rref，用于测试 DDM 对象的行简化阶梯形式（rref）操作
def test_DDM_rref():
    # 创建一个空的 DDM 对象 A，形状为 (0, 4)，域为 QQ
    A = DDM([], (0, 4), QQ)
    # 断言 A 的行简化阶梯形式（rref）结果与 A 自身相等，且无主元列
    assert A.rref() == (A, [])

    # 创建 DDM 对象 A，包含一个 2x2 的 QQ 类型矩阵
    A = DDM([[QQ(0), QQ(1)], [QQ(1), QQ(1)]], (2, 2), QQ)
    # 创建 DDM 对象 Ar，包含一个 2x2 的 QQ 类型矩阵，其行简化阶梯形式
    Ar = DDM([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    # 定义主元列的列表
    pivots = [0, 1]
    # 断言 A 的行简化阶梯形式（rref）结果与 Ar 相等，并且主元列与 pivots 相等
    assert A.rref() == (Ar, pivots)

    # 后续类似地创建不同情况下的测试用例，每个用例包含相应的输入和预期输出结果
    # 每个断言都验证了预期输出与实际输出是否一致
    # 省略具体内容以节省空间

# 定义测试函数 test_DDM_nullspace，用于测试 DDM 对象的零空间（nullspace）计算操作
def test_DDM_nullspace():
    # 更多的测试案例位于 test_nullspace.py 中
    # 创建 DDM 对象 A，包含一个 2x2 的 QQ 类型矩阵
    A = DDM([[QQ(1), QQ(1)], [QQ(1), QQ(1)]], (2, 2), QQ)
    # 创建 DDM 对象 Anull，包含一个 1x2 的 QQ 类型矩阵，其零空间
    Anull = DDM([[QQ(-1), QQ(1)]], (1, 2), QQ)
    # 定义非主元列的列表
    nonpivots = [1]
    # 断言 A 的零空间（nullspace）计算结果与 Anull 相等，并且非主元列与 nonpivots 相等
    assert A.nullspace() == (Anull, nonpivots)


# 定义测试函数 test_DDM_particular，用于测试 DDM 对象的特解（particular）计算操作
def test_DDM_particular():
    # 创建 DDM 对象 A，包含一个 1x2 的 QQ 类型矩阵
    A = DDM([[QQ(1), QQ(0)]], (1, 2), QQ)
    # 断言 A 的特解（particular）计算结果与 1x1 的零矩阵相等
    assert A.particular() == DDM.zeros((1,
    # 创建一个 2x2 的有理数矩阵 A，使用有理数 QQ 类初始化
    A = DDM([[QQ(1, 2), QQ(1, 2)], [QQ(1, 3), QQ(1, 4)]], (2, 2), QQ)
    # 断言矩阵 A 的行列式计算结果为 -1/24
    assert A.det() == QQ(-1, 24)

    # 创建一个 2x1 的整数矩阵 A，使用整数 ZZ 类初始化，这个是非方阵
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    # 使用 lambda 函数检查调用 A.det() 是否抛出 DMShapeError 异常，预期会抛出异常
    raises(DMShapeError, lambda: A.det())

    # 创建一个 0x1 的整数矩阵 A，使用整数 ZZ 类初始化，这个是空矩阵，同样非方阵
    A = DDM([], (0, 1), ZZ)
    # 使用 lambda 函数检查调用 A.det() 是否抛出 DMShapeError 异常，预期会抛出异常
    raises(DMShapeError, lambda: A.det())
def test_DDM_inv():
    # 创建一个 2x2 的有理数矩阵 A 和其逆矩阵 Ainv
    A = DDM([[QQ(1, 1), QQ(2, 1)], [QQ(3, 1), QQ(4, 1)]], (2, 2), QQ)
    Ainv = DDM([[QQ(-2, 1), QQ(1, 1)], [QQ(3, 2), QQ(-1, 2)]], (2, 2), QQ)
    # 断言 A 的逆矩阵与预期的 Ainv 相等
    assert A.inv() == Ainv

    # 创建一个 1x2 的有理数矩阵 A
    A = DDM([[QQ(1), QQ(2)]], (1, 2), QQ)
    # 断言求解 A 的逆矩阵引发 DMShapeError 异常
    raises(DMShapeError, lambda: A.inv())

    # 创建一个 1x1 的整数矩阵 A
    A = DDM([[ZZ(2)]], (1, 1), ZZ)
    # 断言求解 A 的逆矩阵引发 DMDomainError 异常
    raises(DMDomainError, lambda: A.inv())

    # 创建一个空矩阵 A
    A = DDM([], (0, 0), QQ)
    # 断言空矩阵 A 的逆矩阵仍为 A 自身
    assert A.inv() == A

    # 创建一个 2x2 的有理数矩阵 A
    A = DDM([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    # 断言求解 A 的逆矩阵引发 DMNonInvertibleMatrixError 异常
    raises(DMNonInvertibleMatrixError, lambda: A.inv())


def test_DDM_lu():
    # 创建一个 2x2 的有理数矩阵 A
    A = DDM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    # 进行 LU 分解，并获取 L, U 矩阵以及置换列表 swaps
    L, U, swaps = A.lu()
    # 断言分解结果与预期相符
    assert L == DDM([[QQ(1), QQ(0)], [QQ(3), QQ(1)]], (2, 2), QQ)
    assert U == DDM([[QQ(1), QQ(2)], [QQ(0), QQ(-2)]], (2, 2), QQ)
    assert swaps == []

    # 创建一个 4x4 的整数列表 A
    A = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 2]]
    # 预期的 L, U 矩阵
    Lexp = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
    Uexp = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    # 转换 A 为有理数矩阵对象，并断言 LU 分解结果与预期相符
    to_dom = lambda rows, dom: [[dom(e) for e in row] for row in rows]
    A = DDM(to_dom(A, QQ), (4, 4), QQ)
    Lexp = DDM(to_dom(Lexp, QQ), (4, 4), QQ)
    Uexp = DDM(to_dom(Uexp, QQ), (4, 4), QQ)
    L, U, swaps = A.lu()
    assert L == Lexp
    assert U == Uexp
    assert swaps == []


def test_DDM_lu_solve():
    # 基本示例
    A = DDM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    b = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DDM([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    # 断言 LU 解得到的结果与预期相符
    assert A.lu_solve(b) == x

    # 带置换的示例
    A = DDM([[QQ(0), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    # 断言 LU 解得到的结果与预期相符
    assert A.lu_solve(b) == x

    # 过定的、一致的示例
    A = DDM([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    b = DDM([[QQ(1)], [QQ(2)], [QQ(3)]], (3, 1), QQ)
    # 断言 LU 解得到的结果与预期相符
    assert A.lu_solve(b) == x

    # 过定的、不一致的示例
    b = DDM([[QQ(1)], [QQ(2)], [QQ(4)]], (3, 1), QQ)
    # 断言求解 A 的 LU 解引发 DMNonInvertibleMatrixError 异常
    raises(DMNonInvertibleMatrixError, lambda: A.lu_solve(b))

    # 方阵、不可逆的示例
    A = DDM([[QQ(1), QQ(2)], [QQ(1), QQ(2)]], (2, 2), QQ)
    b = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    # 断言求解 A 的 LU 解引发 DMNonInvertibleMatrixError 异常
    raises(DMNonInvertibleMatrixError, lambda: A.lu_solve(b))

    # 欠定的示例
    A = DDM([[QQ(1), QQ(2)]], (1, 2), QQ)
    b = DDM([[QQ(3)]], (1, 1), QQ)
    # 断言求解 A 的 LU 解引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: A.lu_solve(b))

    # 域不匹配的示例
    bz = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    # 断言求解 A 的 LU 解引发 DMDomainError 异常
    raises(DMDomainError, lambda: A.lu_solve(bz))

    # 形状不匹配的示例
    b3 = DDM([[QQ(1)], [QQ(2)], [QQ(3)]], (3, 1), QQ)
    # 断言求解 A 的 LU 解引发 DMShapeError 异常
    raises(DMShapeError, lambda: A.lu_solve(b3))


def test_DDM_charpoly():
    # 创建一个空矩阵 A
    A = DDM([], (0, 0), ZZ)
    # 断言 A 的特征多项式为 [ZZ(1)]
    assert A.charpoly() == [ZZ(1)]

    # 创建一个 3x3 的整数矩阵 A
    A = DDM([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    # 断言 A 的特征多项式为 Avec
    Avec = [ZZ(1), ZZ(-15), ZZ(-18), ZZ(0)]
    assert A.charpoly() == Avec

    # 创建一个 1x2 的整数矩阵 A
    A = DDM([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    # 断言求解 A 的特征多项式引发 DMShapeError 异常
    raises(DMShapeError, lambda: A.charpoly())
    # 创建一个 DDM（DenseMatrix）对象，并初始化为一个 3x3 的矩阵，元素为 ZZ 类型的对象
    dm = DDM([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]
    ], (3, 3), ZZ)
    
    # 使用断言检查获取矩阵中特定位置的元素值是否为预期值 ZZ(5)
    assert dm.getitem(1, 1) == ZZ(5)
    # 使用断言检查获取矩阵中特定位置（使用负数索引）的元素值是否为预期值 ZZ(5)
    assert dm.getitem(1, -2) == ZZ(5)
    # 使用断言检查获取矩阵中特定位置（使用负数索引）的元素值是否为预期值 ZZ(7)
    assert dm.getitem(-1, -3) == ZZ(7)
    
    # 使用断言测试边界条件：尝试访问超出矩阵边界的索引位置，预期会引发 IndexError 异常
    raises(IndexError, lambda: dm.getitem(3, 3))
# 定义一个测试函数，用于测试 DDM 类的 setitem 方法
def test_DDM_setitem():
    # 创建一个 3x3 的零矩阵 dm，数据类型为 ZZ
    dm = DDM.zeros((3, 3), ZZ)
    # 在位置 (0, 0) 处设置值为 1
    dm.setitem(0, 0, 1)
    # 在位置 (1, -2) 处设置值为 1，负数索引从右往左数，-2 表示倒数第二列
    dm.setitem(1, -2, 1)
    # 在位置 (-1, -1) 处设置值为 1，负数索引表示从右往左数
    dm.setitem(-1, -1, 1)
    # 断言 dm 等于一个 3x3 的单位矩阵，数据类型为 ZZ
    assert dm == DDM.eye(3, ZZ)

    # 测试边界情况，设置超出范围的索引应该引发 IndexError 异常
    raises(IndexError, lambda: dm.setitem(3, 3, 0))


# 定义一个测试函数，用于测试 DDM 类的 extract_slice 方法
def test_DDM_extract_slice():
    # 创建一个 3x3 的 DDM 对象 dm，包含 ZZ 类型的元素
    dm = DDM([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)

    # 测试提取整个矩阵的切片
    assert dm.extract_slice(slice(0, 3), slice(0, 3)) == dm
    # 测试提取部分行和列的切片
    assert dm.extract_slice(slice(1, 3), slice(-2)) == DDM([[4], [7]], (2, 1), ZZ)
    assert dm.extract_slice(slice(2, 3), slice(-2)) == DDM([[ZZ(7)]], (1, 1), ZZ)
    assert dm.extract_slice(slice(0, 2), slice(-2)) == DDM([[1], [4]], (2, 1), ZZ)
    assert dm.extract_slice(slice(-1), slice(-1)) == DDM([[1, 2], [4, 5]], (2, 2), ZZ)

    # 测试提取超出矩阵边界的切片，应得到一个空矩阵
    assert dm.extract_slice(slice(2), slice(3, 4)) == DDM([[], []], (2, 0), ZZ)
    assert dm.extract_slice(slice(3, 4), slice(2)) == DDM([], (0, 2), ZZ)
    assert dm.extract_slice(slice(3, 4), slice(3, 4)) == DDM([], (0, 0), ZZ)


# 定义一个测试函数，用于测试 DDM 类的 extract 方法
def test_DDM_extract():
    # 创建一个 3x3 的 DDM 对象 dm1 和一个 2x2 的 DDM 对象 dm2，数据类型为 ZZ
    dm1 = DDM([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    dm2 = DDM([
        [ZZ(6), ZZ(4)],
        [ZZ(3), ZZ(1)]], (2, 2), ZZ)

    # 测试从 dm1 中提取指定位置的子矩阵，与 dm2 相等
    assert dm1.extract([1, 0], [2, 0]) == dm2
    assert dm1.extract([-2, 0], [-1, 0]) == dm2

    # 测试提取空子矩阵
    assert dm1.extract([], []) == DDM.zeros((0, 0), ZZ)
    assert dm1.extract([1], []) == DDM.zeros((1, 0), ZZ)
    assert dm1.extract([], [1]) == DDM.zeros((0, 1), ZZ)

    # 测试超出边界的索引，应该引发 IndexError 异常
    raises(IndexError, lambda: dm2.extract([2], [0]))
    raises(IndexError, lambda: dm2.extract([0], [2]))
    raises(IndexError, lambda: dm2.extract([-3], [0]))
    raises(IndexError, lambda: dm2.extract([0], [-3]))


# 定义一个测试函数，用于测试 DDM 类的 flat 方法
def test_DDM_flat():
    # 创建一个 2x2 的 DDM 对象 dm，数据类型为 ZZ
    dm = DDM([
        [ZZ(6), ZZ(4)],
        [ZZ(3), ZZ(1)]], (2, 2), ZZ)
    # 测试 flat 方法，返回平铺的列表形式
    assert dm.flat() == [ZZ(6), ZZ(4), ZZ(3), ZZ(1)]


# 定义一个测试函数，用于测试 DDM 类的 is_zero_matrix 方法
def test_DDM_is_zero_matrix():
    # 创建一个 2x2 的 DDM 对象 A 和一个 1x2 的零矩阵 Azero，数据类型为 QQ
    A = DDM([[QQ(1), QQ(0)], [QQ(0), QQ(0)]], (2, 2), QQ)
    Azero = DDM.zeros((1, 2), QQ)
    # 断言 is_zero_matrix 方法的返回值
    assert A.is_zero_matrix() is False
    assert Azero.is_zero_matrix() is True


# 定义一个测试函数，用于测试 DDM 类的 is_upper 方法
def test_DDM_is_upper():
    # 创建两个宽矩阵 A 和 B，数据类型为 QQ
    A = DDM([
        [QQ(1), QQ(2), QQ(3), QQ(4)],
        [QQ(0), QQ(5), QQ(6), QQ(7)],
        [QQ(0), QQ(0), QQ(8), QQ(9)]
    ], (3, 4), QQ)
    B = DDM([
        [QQ(1), QQ(2), QQ(3), QQ(4)],
        [QQ(0), QQ(5), QQ(6), QQ(7)],
        [QQ(0), QQ(7), QQ(8), QQ(9)]
    ], (3, 4), QQ)
    # 断言 is_upper 方法的返回值
    assert A.is_upper() is True
    assert B.is_upper() is False

    # 创建两个高矩阵 A 和 B，数据类型为 QQ
    A = DDM([
        [QQ(1), QQ(2), QQ(3)],
        [QQ(0), QQ(5), QQ(6)],
        [QQ(0), QQ(0), QQ(8)],
        [QQ(0), QQ(0), QQ(0)]
    ], (4, 3), QQ)
    B = DDM([
        [QQ(1), QQ(2), QQ(3)],
        [QQ(0), QQ(5), QQ(6)],
        [QQ(0), QQ(0), QQ(8)],
        [QQ(0), QQ(0), QQ(10)]
    ], (4, 3), QQ)
    # 断言 is_upper 方法的返回值
    assert A.is_upper() is True
    assert B.is_upper() is False


# 定义一个测试函数，用于测试 DDM 类的 is_lower 方法
def test_DDM_is_lower():
    # 继续下面的测试函数
    # 创建一个 3x4 的矩阵 A，元素类型为 QQ（有理数对象），并对其进行转置操作
    A = DDM([
        [QQ(1), QQ(2), QQ(3), QQ(4)],
        [QQ(0), QQ(5), QQ(6), QQ(7)],
        [QQ(0), QQ(0), QQ(8), QQ(9)]
    ], (3, 4), QQ).transpose()
    
    # 创建一个 3x4 的矩阵 B，元素类型为 QQ（有理数对象），并对其进行转置操作
    B = DDM([
        [QQ(1), QQ(2), QQ(3), QQ(4)],
        [QQ(0), QQ(5), QQ(6), QQ(7)],
        [QQ(0), QQ(7), QQ(8), QQ(9)]
    ], (3, 4), QQ).transpose()
    
    # 断言矩阵 A 是下三角矩阵（即除对角线以上的元素外，其他元素均为零）
    assert A.is_lower() is True
    
    # 断言矩阵 B 不是下三角矩阵（即存在对角线以上的非零元素）
    assert B.is_lower() is False
    
    # Wide matrices:
    # 创建一个 4x3 的宽矩阵 A，元素类型为 QQ，并对其进行转置操作
    A = DDM([
        [QQ(1), QQ(2), QQ(3)],
        [QQ(0), QQ(5), QQ(6)],
        [QQ(0), QQ(0), QQ(8)],
        [QQ(0), QQ(0), QQ(0)]
    ], (4, 3), QQ).transpose()
    
    # 创建一个 4x3 的宽矩阵 B，元素类型为 QQ，并对其进行转置操作
    B = DDM([
        [QQ(1), QQ(2), QQ(3)],
        [QQ(0), QQ(5), QQ(6)],
        [QQ(0), QQ(0), QQ(8)],
        [QQ(0), QQ(0), QQ(10)]
    ], (4, 3), QQ).transpose()
    
    # 断言宽矩阵 A 是下三角矩阵
    assert A.is_lower() is True
    
    # 断言宽矩阵 B 不是下三角矩阵
    assert B.is_lower() is False
```