# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_sdm.py`

```
"""
Tests for the basic functionality of the SDM class.
"""

# 从 itertools 导入 product 函数，用于生成迭代器的笛卡尔积
from itertools import product

# 导入 Sympy 相关模块
from sympy.core.singleton import S
from sympy.external.gmpy import GROUND_TYPES
from sympy.testing.pytest import raises

# 导入 Sympy 多项式相关模块
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
                                             DMShapeError)


# 测试 SDM 类的基本功能
def test_SDM():
    # 创建一个 SDM 对象 A，包含一个非零元素 {0: {0: ZZ(1)}}
    A = SDM({0:{0:ZZ(1)}}, (2, 2), ZZ)
    assert A.domain == ZZ  # 检查 SDM 对象的域是否为 ZZ
    assert A.shape == (2, 2)  # 检查 SDM 对象的形状是否为 (2, 2)
    assert dict(A) == {0:{0:ZZ(1)}}  # 将 SDM 对象转换为字典并进行比较

    # 测试传入错误的输入是否会引发 DMBadInputError 异常
    raises(DMBadInputError, lambda: SDM({5:{1:ZZ(0)}}, (2, 2), ZZ))
    raises(DMBadInputError, lambda: SDM({0:{5:ZZ(0)}}, (2, 2), ZZ))


# 测试 SDM 类的字符串表示方法
def test_DDM_str():
    # 创建一个 SDM 对象 sdm，包含两个非零元素 {0: {0: ZZ(1)}, 1: {1: ZZ(1)}}
    sdm = SDM({0:{0:ZZ(1)}, 1:{1:ZZ(1)}}, (2, 2), ZZ)
    assert str(sdm) == '{0: {0: 1}, 1: {1: 1}}'  # 检查 SDM 对象的字符串表示形式
    if GROUND_TYPES == 'gmpy': # pragma: no cover
        assert repr(sdm) == 'SDM({0: {0: mpz(1)}, 1: {1: mpz(1)}}, (2, 2), ZZ)'  # 根据 GROUND_TYPES 检查 SDM 对象的详细表示形式
    else:        # pragma: no cover
        assert repr(sdm) == 'SDM({0: {0: 1}, 1: {1: 1}}, (2, 2), ZZ)'  # 检查 SDM 对象的详细表示形式


# 测试 SDM 类的 new 方法
def test_SDM_new():
    # 创建一个 SDM 对象 A，包含一个非零元素 {0: {0: ZZ(1)}}
    A = SDM({0:{0:ZZ(1)}}, (2, 2), ZZ)
    # 使用 new 方法创建一个空的 SDM 对象 B，并进行比较
    B = A.new({}, (2, 2), ZZ)
    assert B == SDM({}, (2, 2), ZZ)


# 测试 SDM 类的 copy 方法
def test_SDM_copy():
    # 创建一个 SDM 对象 A，包含一个非零元素 {0: {0: ZZ(1)}}
    A = SDM({0:{0:ZZ(1)}}, (2, 2), ZZ)
    # 使用 copy 方法创建一个新的 SDM 对象 B，并进行比较
    B = A.copy()
    assert A == B  # 检查两个 SDM 对象是否相等
    A[0][0] = ZZ(2)  # 修改 A 的元素
    assert A != B  # 再次检查两个 SDM 对象是否相等，预期应该不相等


# 测试 SDM 类的 from_list 方法
def test_SDM_from_list():
    # 使用 from_list 方法从列表创建一个 SDM 对象 A
    A = SDM.from_list([[ZZ(0), ZZ(1)], [ZZ(1), ZZ(0)]], (2, 2), ZZ)
    # 检查 SDM 对象 A 的内容是否符合预期
    assert A == SDM({0:{1:ZZ(1)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)

    # 测试传入错误的输入是否会引发 DMBadInputError 异常
    raises(DMBadInputError, lambda: SDM.from_list([[ZZ(0)], [ZZ(0), ZZ(1)]], (2, 2), ZZ))
    raises(DMBadInputError, lambda: SDM.from_list([[ZZ(0), ZZ(1)]], (2, 2), ZZ))


# 测试 SDM 类的 to_list 方法
def test_SDM_to_list():
    # 创建一个 SDM 对象 A，包含一个非零元素 {0: {1: ZZ(1)}}
    A = SDM({0:{1: ZZ(1)}}, (2, 2), ZZ)
    # 检查 SDM 对象 A 转换为列表后的内容是否符合预期
    assert A.to_list() == [[ZZ(0), ZZ(1)], [ZZ(0), ZZ(0)]]

    # 测试空形状的 SDM 对象转换为列表后的内容是否符合预期
    A = SDM({}, (0, 2), ZZ)
    assert A.to_list() == []

    A = SDM({}, (2, 0), ZZ)
    assert A.to_list() == [[], []]


# 测试 SDM 类的 to_list_flat 方法
def test_SDM_to_list_flat():
    # 创建一个 SDM 对象 A，包含一个非零元素 {0: {1: ZZ(1)}}
    A = SDM({0:{1: ZZ(1)}}, (2, 2), ZZ)
    # 检查 SDM 对象 A 转换为扁平列表后的内容是否符合预期
    assert A.to_list_flat() == [ZZ(0), ZZ(1), ZZ(0), ZZ(0)]


# 测试 SDM 类的 to_dok 方法
def test_SDM_to_dok():
    # 创建一个 SDM 对象 A，包含一个非零元素 {0: {1: ZZ(1)}}
    A = SDM({0:{1: ZZ(1)}}, (2, 2), ZZ)
    # 检查 SDM 对象 A 转换为 DOK 格式后的内容是否符合预期
    assert A.to_dok() == {(0, 1): ZZ(1)}


# 测试 SDM 类的 from_ddm 方法
def test_SDM_from_ddm():
    # 创建一个 DDM 对象 A，包含一个非零元素 [[ZZ(1), ZZ(0)], [ZZ(1), ZZ(0)]]
    A = DDM([[ZZ(1), ZZ(0)], [ZZ(1), ZZ(0)]], (2, 2), ZZ)
    # 使用 from_ddm 方法从 DDM 对象 A 创建一个 SDM 对象 B，并进行比较
    B = SDM.from_ddm(A)
    assert B.domain == ZZ  # 检查 SDM 对象 B 的域是否为 ZZ
    assert B.shape == (2, 2)  # 检查 SDM 对象 B 的形状是否为 (2, 2)
    assert dict(B) == {0:{0:ZZ(1)}, 1:{0:ZZ(1)}}  # 将 SDM 对象 B 转换为字典并进行比较


# 测试 SDM 类的 to_ddm 方法
def test_SDM_to_ddm():
    # 创建一个 SDM 对象 A，包含一个非零元素 {0: {1: ZZ(1)}}
    A = SDM({0:{1: ZZ(1)}}, (2, 2), ZZ)
    # 创建一个预期的 DDM 对象 B，包含 [[ZZ(0), ZZ(1)], [ZZ(0), ZZ(0)]]
    B = DDM([[ZZ(0), ZZ(1)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    # 检查 SDM 对象 A 转换为 DDM 对象后的内容是否符合预期
    assert A.to_ddm() == B


# 测试 SDM 类的 to_sdm 方法
def test
    # 调用 lambda 表达式来测试 A 对象在 (0, 2) 索引处是否引发 IndexError 异常
    raises(IndexError, lambda: A.getitem(0, 2))
# 测试 SDM 类的 setitem 方法
def test_SDM_setitem():
    # 创建一个 SDM 对象 A，包含一个非空的二维字典
    A = SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    # 在位置 (0, 0) 设置值为 ZZ(1)
    A.setitem(0, 0, ZZ(1))
    # 断言 A 是否等于修改后的 SDM 对象
    assert A == SDM({0:{0:ZZ(1), 1:ZZ(1)}}, (2, 2), ZZ)
    
    # 在位置 (1, 0) 设置值为 ZZ(1)
    A.setitem(1, 0, ZZ(1))
    # 断言 A 是否等于修改后的 SDM 对象
    assert A == SDM({0:{0:ZZ(1), 1:ZZ(1)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
    
    # 在位置 (1, 0) 设置值为 ZZ(0)
    A.setitem(1, 0, ZZ(0))
    # 断言 A 是否等于修改后的 SDM 对象
    assert A == SDM({0:{0:ZZ(1), 1:ZZ(1)}}, (2, 2), ZZ)
    
    # 重复上述测试，确保这次行为空
    A.setitem(1, 0, ZZ(0))
    # 断言 A 是否等于修改后的 SDM 对象
    assert A == SDM({0:{0:ZZ(1), 1:ZZ(1)}}, (2, 2), ZZ)
    
    # 在位置 (0, 0) 设置值为 ZZ(0)
    A.setitem(0, 0, ZZ(0))
    # 断言 A 是否等于修改后的 SDM 对象
    assert A == SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    
    # 这次行存在但列为空
    A.setitem(0, 0, ZZ(0))
    # 断言 A 是否等于修改后的 SDM 对象
    assert A == SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    
    # 测试索引超出范围的情况，期望引发 IndexError 异常
    raises(IndexError, lambda: A.setitem(2, 0, ZZ(1)))
    raises(IndexError, lambda: A.setitem(0, 2, ZZ(1)))


# 测试 SDM 类的 extract_slice 方法
def test_SDM_extract_slice():
    # 创建一个 SDM 对象 A，包含一个二维字典
    A = SDM({0:{0:ZZ(1), 1:ZZ(2)}, 1:{0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    # 提取切片 [1:2, 1:2] 的子集 B
    B = A.extract_slice(slice(1, 2), slice(1, 2))
    # 断言 B 是否等于预期的 SDM 对象
    assert B == SDM({0:{0:ZZ(4)}}, (1, 1), ZZ)


# 测试 SDM 类的 extract 方法
def test_SDM_extract():
    # 创建一个 SDM 对象 A，包含一个二维字典
    A = SDM({0:{0:ZZ(1), 1:ZZ(2)}, 1:{0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    # 提取位置列表 [1] 和 [1] 的子集 B
    B = A.extract([1], [1])
    # 断言 B 是否等于预期的 SDM 对象
    assert B == SDM({0:{0:ZZ(4)}}, (1, 1), ZZ)
    
    # 提取位置列表 [1, 0] 和 [1, 0] 的子集 B
    B = A.extract([1, 0], [1, 0])
    # 断言 B 是否等于预期的 SDM 对象
    assert B == SDM({0:{0:ZZ(4), 1:ZZ(3)}, 1:{0:ZZ(2), 1:ZZ(1)}}, (2, 2), ZZ)
    
    # 提取位置列表 [1, 1] 和 [1, 1] 的子集 B
    B = A.extract([1, 1], [1, 1])
    # 断言 B 是否等于预期的 SDM 对象
    assert B == SDM({0:{0:ZZ(4), 1:ZZ(4)}, 1:{0:ZZ(4), 1:ZZ(4)}}, (2, 2), ZZ)
    
    # 提取位置列表 [-1] 和 [-1] 的子集 B
    B = A.extract([-1], [-1])
    # 断言 B 是否等于预期的 SDM 对象
    assert B == SDM({0:{0:ZZ(4)}}, (1, 1), ZZ)
    
    # 创建一个空的 SDM 对象 A
    A = SDM({}, (2, 2), ZZ)
    # 提取位置列表 [0, 1, 0] 和 [0, 0] 的子集 B
    B = A.extract([0, 1, 0], [0, 0])
    # 断言 B 是否等于预期的空 SDM 对象
    assert B == SDM({}, (3, 2), ZZ)
    
    # 对空列表进行提取操作，期望返回全零的 SDM 对象
    assert A.extract([], []) == SDM.zeros((0, 0), ZZ)
    assert A.extract([1], []) == SDM.zeros((1, 0), ZZ)
    assert A.extract([], [1]) == SDM.zeros((0, 1), ZZ)
    
    # 测试索引超出范围的情况，期望引发 IndexError 异常
    raises(IndexError, lambda: A.extract([2], [0]))
    raises(IndexError, lambda: A.extract([0], [2]))
    raises(IndexError, lambda: A.extract([-3], [0]))
    raises(IndexError, lambda: A.extract([0], [-3)])


# 测试 SDM 类的 zeros 方法
def test_SDM_zeros():
    # 创建一个全零的 SDM 对象 A
    A = SDM.zeros((2, 2), ZZ)
    # 断言 A 的域和形状是否符合预期
    assert A.domain == ZZ
    assert A.shape == (2, 2)
    # 断言 A 的内容是否为空字典
    assert dict(A) == {}


# 测试 SDM 类的 ones 方法
def test_SDM_ones():
    # 创建一个全一的 SDM 对象 A
    A = SDM.ones((1, 2), QQ)
    # 断言 A 的域和形状是否符合预期
    assert A.domain == QQ
    assert A.shape == (1, 2)
    # 断言 A 的内容是否符合预期的字典形式
    assert dict(A) == {0:{0:QQ(1), 1:QQ(1)}}


# 测试 SDM 类的 eye 方法
def test_SDM_eye():
    # 创建一个单位矩阵的 SDM 对象 A
    A = SDM.eye((2, 2), ZZ)
    # 断言 A 的域和形状是否符合预期
    assert A.domain == ZZ
    assert A.shape == (2, 2)
    # 断言 A 的内容是否符合预期的字典形式
    assert dict(A) == {0:{0:ZZ(1)}, 1:{1:ZZ(1)}}


# 测试 SDM 类的 diag 方法
def test_SDM_diag():
    # 创建一个对角矩阵的 SDM 对象 A
    A = SDM.diag([ZZ(1), ZZ(2)], ZZ, (2, 3))
    # 断言 A 是否等于预期的 SDM 对象
    assert A == SDM({0:{0:ZZ(1)}, 1:{1:ZZ(2)}}, (2, 3), ZZ)


# 测试 SDM 类的 transpose 方法
def test_SDM_transpose():
    # 创建一个 SDM 对象 A，包含一个二维字典
    A = SDM({0:{0:ZZ(1), 1:ZZ(2)}, 1:{0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    # 创建预期的转置 SDM 对象 B
    B = SDM({
# 定义一个测试函数，用于测试 SDM 类的乘法运算
def test_SDM_mul():
    # 创建两个 SDM 对象 A 和 B，分别表示二维稀疏矩阵，元素类型为整数 ZZ
    A = SDM({0:{0:ZZ(2)}}, (2, 2), ZZ)
    B = SDM({0:{0:ZZ(4)}}, (2, 2), ZZ)
    # 断言 A 乘以整数 2 等于 B
    assert A*ZZ(2) == B
    # 断言 整数 2 乘以 A 等于 B
    assert ZZ(2)*A == B

    # 检查乘法运算中的类型错误，期望抛出 TypeError 异常
    raises(TypeError, lambda: A*QQ(1, 2))
    raises(TypeError, lambda: QQ(1, 2)*A)


# 定义一个测试函数，用于测试 SDM 类的逐元素乘法运算
def test_SDM_mul_elementwise():
    # 创建两个 SDM 对象 A 和 B，分别表示二维稀疏矩阵，元素类型为整数 ZZ
    A = SDM({0:{0:ZZ(2), 1:ZZ(2)}}, (2, 2), ZZ)
    B = SDM({0:{0:ZZ(4)}, 1:{0:ZZ(3)}}, (2, 2), ZZ)
    C = SDM({0:{0:ZZ(8)}}, (2, 2), ZZ)
    # 断言 A 逐元素乘以 B 等于 C
    assert A.mul_elementwise(B) == C
    # 断言 B 逐元素乘以 A 等于 C
    assert B.mul_elementwise(A) == C

    # 将 A 转换为有理数类型 QQ
    Aq = A.convert_to(QQ)
    # 创建一个只有一个元素的 SDM 对象 A1
    A1 = SDM({0:{0:ZZ(1)}}, (1, 1), ZZ)

    # 检查有理数类型的乘法运算中的域错误，期望抛出 DMDomainError 异常
    raises(DMDomainError, lambda: Aq.mul_elementwise(B))
    # 检查形状不匹配的乘法运算，期望抛出 DMShapeError 异常
    raises(DMShapeError, lambda: A1.mul_elementwise(B))


# 定义一个测试函数，用于测试 SDM 类的矩阵乘法运算
def test_SDM_matmul():
    # 创建两个 SDM 对象 A 和 B，分别表示二维稀疏矩阵，元素类型为整数 ZZ
    A = SDM({0:{0:ZZ(2)}}, (2, 2), ZZ)
    B = SDM({0:{0:ZZ(4)}}, (2, 2), ZZ)
    # 断言 A 乘以 A 等于 A 乘以 A 等于 B
    assert A.matmul(A) == A*A == B

    # 创建一个 SDM 对象 C，表示二维稀疏矩阵，元素类型为有理数 QQ
    C = SDM({0:{0:ZZ(2)}}, (2, 2), QQ)
    # 检查域不匹配的矩阵乘法，期望抛出 DMDomainError 异常
    raises(DMDomainError, lambda: A.matmul(C))

    # 创建一个 SDM 对象 A，表示二维稀疏矩阵，元素类型为整数 ZZ
    A = SDM({0:{0:ZZ(1), 1:ZZ(2)}, 1:{0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    # 创建一个 SDM 对象 B，表示二维稀疏矩阵，元素类型为整数 ZZ
    B = SDM({0:{0:ZZ(7), 1:ZZ(10)}, 1:{0:ZZ(15), 1:ZZ(22)}}, (2, 2), ZZ)
    # 断言 A 乘以 A 等于 A 乘以 A 等于 B
    assert A.matmul(A) == A*A == B

    # 创建不同形状的 SDM 对象进行矩阵乘法
    A22 = SDM({0:{0:ZZ(4)}}, (2, 2), ZZ)
    A32 = SDM({0:{0:ZZ(2)}}, (3, 2), ZZ)
    A23 = SDM({0:{0:ZZ(4)}}, (2, 3), ZZ)
    A33 = SDM({0:{0:ZZ(8)}}, (3, 3), ZZ)
    A22 = SDM({0:{0:ZZ(8)}}, (2, 2), ZZ)
    # 断言 A32 乘以 A23 等于 A33
    assert A32.matmul(A23) == A33
    # 断言 A23 乘以 A32 等于 A22
    assert A23.matmul(A32) == A22
    # XXX: @ not supported by SDM...
    #assert A32.matmul(A23) == A32 @ A23 == A33
    #assert A23.matmul(A32) == A23 @ A32 == A22
    #raises(DMShapeError, lambda: A23 @ A22)
    # 检查形状不匹配的矩阵乘法，期望抛出 DMShapeError 异常
    raises(DMShapeError, lambda: A23.matmul(A22))

    # 创建两个 SDM 对象 A 和 B，表示二维稀疏矩阵，元素类型为整数 ZZ
    A = SDM({0: {0: ZZ(-1), 1: ZZ(1)}}, (1, 2), ZZ)
    B = SDM({0: {0: ZZ(-1)}, 1: {0: ZZ(-1)}}, (2, 1), ZZ)
    # 断言 A 乘以 B 等于 A 乘以 B 等于 一个空的 SDM 对象，形状为 (1, 1)
    assert A.matmul(B) == A*B == SDM({}, (1, 1), ZZ)


# 定义一个测试函数，用于测试特殊的矩阵乘法
def test_matmul_exraw():

    # 定义一个函数 dm，用于将输入的稀疏矩阵数据转换为 SDM 对象
    def dm(d):
        result = {}
        for i, row in d.items():
            row = {j:val for j, val in row.items() if val}
            if row:
                result[i] = row
        return SDM(result, (2, 2), EXRAW)

    # 定义一个包含特定数学符号的列表
    values = [S.NegativeInfinity, S.NegativeOne, S.Zero, S.One, S.Infinity]
    # 遍历 values 中的每一种组合
    for a, b, c, d in product(*[values]*4):
        # 创建一个 SDM 对象 Ad，表示二维稀疏矩阵，元素类型为特殊的外部原始类型 EXRAW
        Ad = dm({0: {0:a, 1:b}, 1: {0:c, 1:d}})
        # 创建一个 SDM 对象 Ad2，表示二维稀疏矩阵，元素类型为特殊的外部原始类型 EXRAW
        Ad2 = dm({0: {0:a*a + b*c, 1:a*b + b*d}, 1:{0:c*a + d*c, 1: c*b + d*d}})
        # 断言 Ad 乘以 Ad 等于 Ad2
        assert Ad * Ad == Ad2


# 定义一个测试函数，用于测试 SDM 类的加法运算
def test_SDM_add():
    # 创建两个 SDM 对象 A 和 B，表示二维稀疏矩阵，元素类型为整数 ZZ
    A = SDM({0:{1:ZZ(1)}, 1:{0:ZZ(2), 1:ZZ(3)}}, (2, 2), ZZ)
    # 断言：A 对象调用 sub 方法并传入 B，结果应该等于 A 对象减去 B 后的结果，即等于 C
    assert A.sub(B) == A - B == C
    
    # 使用 lambda 函数在执行 A - [] 操作时应该引发 TypeError 异常
    raises(TypeError, lambda: A - [])
def test_SDM_neg():
    A = SDM({0:{1:ZZ(1)}, 1:{0:ZZ(2), 1:ZZ(3)}}, (2, 2), ZZ)
    B = SDM({0:{1:ZZ(-1)}, 1:{0:ZZ(-2), 1:ZZ(-3)}}, (2, 2), ZZ)
    # 施密特-迪克斯特-马赫森（SDM）矩阵 A 的负矩阵操作，验证负矩阵的正确性
    assert A.neg() == -A == B

def test_SDM_convert_to():
    A = SDM({0:{1:ZZ(1)}, 1:{0:ZZ(2), 1:ZZ(3)}}, (2, 2), ZZ)
    B = SDM({0:{1:QQ(1)}, 1:{0:QQ(2), 1:QQ(3)}}, (2, 2), QQ)
    # 将整数域（ZZ）的 SDM 矩阵 A 转换为有理数域（QQ）的矩阵 B
    C = A.convert_to(QQ)
    assert C == B
    assert C.domain == QQ

    # 测试转换回整数域 ZZ，应当恢复原矩阵 A
    D = A.convert_to(ZZ)
    assert D == A
    assert D.domain == ZZ

def test_SDM_hstack():
    A = SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    B = SDM({1:{1:ZZ(1)}}, (2, 2), ZZ)
    AA = SDM({0:{1:ZZ(1), 3:ZZ(1)}}, (2, 4), ZZ)
    AB = SDM({0:{1:ZZ(1)}, 1:{3:ZZ(1)}}, (2, 4), ZZ)
    # 测试 SDM 矩阵水平拼接函数 hstack 的功能
    assert SDM.hstack(A) == A
    assert SDM.hstack(A, A) == AA
    assert SDM.hstack(A, B) == AB

def test_SDM_vstack():
    A = SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    B = SDM({1:{1:ZZ(1)}}, (2, 2), ZZ)
    AA = SDM({0:{1:ZZ(1)}, 2:{1:ZZ(1)}}, (4, 2), ZZ)
    AB = SDM({0:{1:ZZ(1)}, 3:{1:ZZ(1)}}, (4, 2), ZZ)
    # 测试 SDM 矩阵垂直拼接函数 vstack 的功能
    assert SDM.vstack(A) == A
    assert SDM.vstack(A, A) == AA
    assert SDM.vstack(A, B) == AB

def test_SDM_applyfunc():
    A = SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    B = SDM({0:{1:ZZ(2)}}, (2, 2), ZZ)
    # 测试 SDM 矩阵的元素级函数应用功能
    assert A.applyfunc(lambda x: 2*x, ZZ) == B

def test_SDM_inv():
    A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
    B = SDM({0:{0:QQ(-2), 1:QQ(1)}, 1:{0:QQ(3, 2), 1:QQ(-1, 2)}}, (2, 2), QQ)
    # 测试 SDM 矩阵的逆矩阵计算功能
    assert A.inv() == B

def test_SDM_det():
    A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
    # 测试 SDM 矩阵的行列式计算功能
    assert A.det() == QQ(-2)

def test_SDM_lu():
    A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
    L = SDM({0:{0:QQ(1)}, 1:{0:QQ(3), 1:QQ(1)}}, (2, 2), QQ)
    # 测试 SDM 矩阵的 LU 分解功能
    assert A.lu()[0] == L

def test_SDM_lu_solve():
    A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
    b = SDM({0:{0:QQ(1)}, 1:{0:QQ(2)}}, (2, 1), QQ)
    x = SDM({1:{0:QQ(1, 2)}}, (2, 1), QQ)
    # 测试 SDM 矩阵的 LU 分解解线性方程组功能
    assert A.matmul(x) == b
    assert A.lu_solve(b) == x

def test_SDM_charpoly():
    A = SDM({0:{0:ZZ(1), 1:ZZ(2)}, 1:{0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    # 测试 SDM 矩阵的特征多项式计算功能
    assert A.charpoly() == [ZZ(1), ZZ(-5), ZZ(-2)]

def test_SDM_nullspace():
    A = SDM({0:{0:QQ(1), 1:QQ(1)}}, (2, 2), QQ)
    # 测试 SDM 矩阵的零空间计算功能
    assert A.nullspace()[0] == SDM({0:{0:QQ(-1), 1:QQ(1)}}, (1, 2), QQ)

def test_SDM_rref():
    A = SDM({0:{0:QQ(1), 1:QQ(2)},
             1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
    A_rref = SDM({0:{0:QQ(1)}, 1:{1:QQ(1)}}, (2, 2), QQ)
    # 测试 SDM 矩阵的行简化阶梯形计算功能
    assert A.rref() == (A_rref, [0, 1])

    A = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(2)},
             1: {0: QQ(3),           2: QQ(4)}}, (2, 3), ZZ)
    A_rref = SDM({0: {0: QQ(1,1), 2: QQ(4,3)},
                  1: {1: QQ(1,1), 2: QQ(1,3)}}, (2, 3), QQ)
    # 更复杂的 SDM 矩阵行简化阶梯形计算测试
    assert A.rref() == A_rref
    # 断言语句，用于验证矩阵 A 的行简化阶梯形式（Reduced Row Echelon Form, RREF）是否等于 (A_rref, [0, 1])
    assert A.rref() == (A_rref, [0, 1])
# 测试特定的稀疏矩阵 SDM 对象的特征
def test_SDM_particular():
    # 创建一个包含单个元素的稀疏矩阵 SDM 对象 A
    A = SDM({0:{0:QQ(1)}}, (2, 2), QQ)
    # 创建一个全零的稀疏矩阵 SDM 对象 Apart
    Apart = SDM.zeros((1, 2), QQ)
    # 断言 A 的特征等于 Apart
    assert A.particular() == Apart


# 测试稀疏矩阵 SDM 对象是否为零矩阵
def test_SDM_is_zero_matrix():
    # 创建一个非零元素的稀疏矩阵 SDM 对象 A
    A = SDM({0: {0: QQ(1)}}, (2, 2), QQ)
    # 创建一个全零的稀疏矩阵 SDM 对象 Azero
    Azero = SDM.zeros((1, 2), QQ)
    # 断言 A 不是零矩阵
    assert A.is_zero_matrix() is False
    # 断言 Azero 是零矩阵
    assert Azero.is_zero_matrix() is True


# 测试稀疏矩阵 SDM 对象是否为上三角矩阵
def test_SDM_is_upper():
    # 创建一个上三角稀疏矩阵 SDM 对象 A
    A = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(3), 3: QQ(4)},
             1: {1: QQ(5), 2: QQ(6), 3: QQ(7)},
             2: {2: QQ(8), 3: QQ(9)}}, (3, 4), QQ)
    # 创建一个非上三角的稀疏矩阵 SDM 对象 B
    B = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(3), 3: QQ(4)},
             1: {1: QQ(5), 2: QQ(6), 3: QQ(7)},
             2: {1: QQ(7), 2: QQ(8), 3: QQ(9)}}, (3, 4), QQ)
    # 断言 A 是上三角矩阵
    assert A.is_upper() is True
    # 断言 B 不是上三角矩阵
    assert B.is_upper() is False


# 测试稀疏矩阵 SDM 对象是否为下三角矩阵
def test_SDM_is_lower():
    # 创建一个下三角稀疏矩阵 SDM 对象 A
    A = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(3), 3: QQ(4)},
             1: {1: QQ(5), 2: QQ(6), 3: QQ(7)},
             2: {2: QQ(8), 3: QQ(9)}}, (3, 4), QQ
            ).transpose()
    # 创建一个非下三角的稀疏矩阵 SDM 对象 B
    B = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(3), 3: QQ(4)},
             1: {1: QQ(5), 2: QQ(6), 3: QQ(7)},
             2: {1: QQ(7), 2: QQ(8), 3: QQ(9)}}, (3, 4), QQ
            ).transpose()
    # 断言 A 是下三角矩阵
    assert A.is_lower() is True
    # 断言 B 不是下三角矩阵
    assert B.is_lower() is False
```