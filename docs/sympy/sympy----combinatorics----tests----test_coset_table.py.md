# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_coset_table.py`

```
# 导入所需的类和函数
from sympy.combinatorics.fp_groups import FpGroup
from sympy.combinatorics.coset_table import (CosetTable,
                    coset_enumeration_r, coset_enumeration_c)
from sympy.combinatorics.coset_table import modified_coset_enumeration_r
from sympy.combinatorics.free_groups import free_group
from sympy.testing.pytest import slow

"""
References
==========

[1] Holt, D., Eick, B., O'Brien, E.
"Handbook of Computational Group Theory"

[2] John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson
Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490.
"Implementation and Analysis of the Todd-Coxeter Algorithm"

"""

def test_scan_1():
    # Example 5.1 from [1]
    # 创建自由群 F 和生成元 x, y
    F, x, y = free_group("x, y")
    # 创建 FpGroup 对象 f，定义群的生成元和关系
    f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    # 创建 CosetTable 对象 c，指定群 f 和起始元素 [x]
    c = CosetTable(f, [x])

    # 使用 scan_and_fill 方法填充 coset 表格
    c.scan_and_fill(0, x)
    # 验证填充后的表格
    assert c.table == [[0, 0, None, None]]
    assert c.p == [0]
    assert c.n == 1
    assert c.omega == [0]

    c.scan_and_fill(0, x**3)
    assert c.table == [[0, 0, None, None]]
    assert c.p == [0]
    assert c.n == 1
    assert c.omega == [0]

    c.scan_and_fill(0, y**3)
    assert c.table == [[0, 0, 1, 2], [None, None, 2, 0], [None, None, 0, 1]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]

    c.scan_and_fill(0, x**-1*y**-1*x*y)
    assert c.table == [[0, 0, 1, 2], [None, None, 2, 0], [2, 2, 0, 1]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]

    c.scan_and_fill(1, x**3)
    assert c.table == [[0, 0, 1, 2], [3, 4, 2, 0], [2, 2, 0, 1], \
            [4, 1, None, None], [1, 3, None, None]]
    assert c.p == [0, 1, 2, 3, 4]
    assert c.n == 5
    assert c.omega == [0, 1, 2, 3, 4]

    c.scan_and_fill(1, y**3)
    assert c.table == [[0, 0, 1, 2], [3, 4, 2, 0], [2, 2, 0, 1], \
            [4, 1, None, None], [1, 3, None, None]]
    assert c.p == [0, 1, 2, 3, 4]
    assert c.n == 5
    assert c.omega == [0, 1, 2, 3, 4]

    c.scan_and_fill(1, x**-1*y**-1*x*y)
    assert c.table == [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1], \
            [None, 1, None, None], [1, 3, None, None]]
    assert c.p == [0, 1, 2, 1, 1]
    assert c.n == 3
    assert c.omega == [0, 1, 2]

    # Example 5.2 from [1]
    # 使用新的自由群 F 和生成元 x, y
    f = FpGroup(F, [x**2, y**3, (x*y)**3])
    # 创建新的 CosetTable 对象 c，指定群 f 和起始元素 [x*y]
    c = CosetTable(f, [x*y])

    c.scan_and_fill(0, x*y)
    assert c.table == [[1, None, None, 1], [None, 0, 0, None]]
    assert c.p == [0, 1]
    assert c.n == 2
    assert c.omega == [0, 1]

    c.scan_and_fill(0, x**2)
    assert c.table == [[1, 1, None, 1], [0, 0, 0, None]]
    assert c.p == [0, 1]
    assert c.n == 2
    assert c.omega == [0, 1]

    c.scan_and_fill(0, y**3)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]

    c.scan_and_fill(0, (x*y)**3)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]
    # 调用对象 c 的 scan_and_fill 方法，填充值 1 和 x 的平方
    c.scan_and_fill(1, x**2)
    # 断言表 c.table 的状态是否为预期的 [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    # 断言属性 c.p 的状态是否为预期的 [0, 1, 2]
    assert c.p == [0, 1, 2]
    # 断言属性 c.n 的状态是否为预期的 3
    assert c.n == 3
    # 断言属性 c.omega 的状态是否为预期的 [0, 1, 2]
    assert c.omega == [0, 1, 2]

    # 再次调用对象 c 的 scan_and_fill 方法，填充值 1 和 y 的立方
    c.scan_and_fill(1, y**3)
    # 断言表 c.table 的状态是否为预期的 [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    # 断言属性 c.p 的状态是否为预期的 [0, 1, 2]
    assert c.p == [0, 1, 2]
    # 断言属性 c.n 的状态是否为预期的 3
    assert c.n == 3
    # 断言属性 c.omega 的状态是否为预期的 [0, 1, 2]
    assert c.omega == [0, 1, 2]

    # 再次调用对象 c 的 scan_and_fill 方法，填充值 1 和 (x*y) 的立方
    c.scan_and_fill(1, (x*y)**3)
    # 断言表 c.table 的状态是否为预期的 [[1, 1, 2, 1], [0, 0, 0, 2], [3, 4, 1, 0], [None, 2, 4, None], [2, None, None, 3]]
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [3, 4, 1, 0], [None, 2, 4, None], [2, None, None, 3]]
    # 断言属性 c.p 的状态是否为预期的 [0, 1, 2, 3, 4]
    assert c.p == [0, 1, 2, 3, 4]
    # 断言属性 c.n 的状态是否为预期的 5
    assert c.n == 5
    # 断言属性 c.omega 的状态是否为预期的 [0, 1, 2, 3, 4]
    assert c.omega == [0, 1, 2, 3, 4]

    # 再次调用对象 c 的 scan_and_fill 方法，填充值 2 和 x 的平方
    c.scan_and_fill(2, x**2)
    # 断言表 c.table 的状态是否为预期的 [[1, 1, 2, 1], [0, 0, 0, 2], [3, 3, 1, 0], [2, 2, 3, 3], [2, None, None, 3]]
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [3, 3, 1, 0], [2, 2, 3, 3], [2, None, None, 3]]
    # 断言属性 c.p 的状态是否为预期的 [0, 1, 2, 3, 3]
    assert c.p == [0, 1, 2, 3, 3]
    # 断言属性 c.n 的状态是否为预期的 4
    assert c.n == 4
    # 断言属性 c.omega 的状态是否为预期的 [0, 1, 2, 3]
    assert c.omega == [0, 1, 2, 3]
@slow
def test_coset_enumeration():
    # 这个测试函数包含了两种策略（HLT 和 Felsch 策略）的组合测试

    # Example 5.1 from [1]
    F, x, y = free_group("x, y")
    # 创建自由群 FpGroup 对象
    f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    # 使用 HLT 策略进行余陪列举，并压缩和标准化结果
    C_r = coset_enumeration_r(f, [x])
    C_r.compress(); C_r.standardize()
    # 使用 Felsch 策略进行余陪列举，并压缩和标准化结果
    C_c = coset_enumeration_c(f, [x])
    C_c.compress(); C_c.standardize()
    # 预期的余陪表
    table1 = [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]
    assert C_r.table == table1
    assert C_c.table == table1

    # E1 from [2] Pg. 474
    F, r, s, t = free_group("r, s, t")
    # 创建自由群 FpGroup 对象
    E1 = FpGroup(F, [t**-1*r*t*r**-2, r**-1*s*r*s**-2, s**-1*t*s*t**-2])
    # 使用 HLT 策略进行余陪列举，并压缩结果
    C_r = coset_enumeration_r(E1, [])
    C_r.compress()
    # 使用 Felsch 策略进行余陪列举，并压缩结果
    C_c = coset_enumeration_c(E1, [])
    C_c.compress()
    # 预期的余陪表
    table2 = [[0, 0, 0, 0, 0, 0]]
    assert C_r.table == table2
    # issue #11449 的测试
    assert C_c.table == table2

    # Cox group from [2] Pg. 474
    F, a, b = free_group("a, b")
    # 创建自由群 FpGroup 对象
    Cox = FpGroup(F, [a**6, b**6, (a*b)**2, (a**2*b**2)**2, (a**3*b**3)**5])
    # 使用 HLT 策略进行余陪列举，并压缩和标准化结果
    C_r = coset_enumeration_r(Cox, [a])
    C_r.compress(); C_r.standardize()
    # 使用 Felsch 策略进行余陪列举，并压缩和标准化结果
    C_c = coset_enumeration_c(Cox, [a])
    C_c.compress(); C_c.standardize()
    # 预期的余陪表
    assert C_r.table == table3
    assert C_c.table == table3

    # Group denoted by B2,4 from [2] Pg. 474
    F, a, b = free_group("a, b")
    # 创建自由群 FpGroup 对象
    B_2_4 = FpGroup(F, [a**4, b**4, (a*b)**4, (a**-1*b)**4, (a**2*b)**4, \
            (a*b**2)**4, (a**2*b**2)**4, (a**-1*b*a*b)**4, (a*b**-1*a*b)**4])
    # 使用 HLT 策略进行余陪列举
    C_r = coset_enumeration_r(B_2_4, [a])
    # 使用 Felsch 策略进行余陪列举
    C_c = coset_enumeration_c(B_2_4, [a])
    # 计算并断言索引计数
    index_r = 0
    for i in range(len(C_r.p)):
        if C_r.p[i] == i:
            index_r += 1
    assert index_r == 1024

    index_c = 0
    for i in range(len(C_c.p)):
        if C_c.p[i] == i:
            index_c += 1
    assert index_c == 1024

    # trivial Macdonald group G(2,2) from [2] Pg. 480
    M = FpGroup(F, [b**-1*a**-1*b*a*b**-1*a*b*a**-2, a**-1*b**-1*a*b*a**-1*b*a*b**-2])
    # 使用 HLT 策略进行余陪列举，并压缩和标准化结果
    C_r = coset_enumeration_r(M, [a])
    C_r.compress(); C_r.standardize()
    # 使用 Felsch 策略进行余陪列举，并压缩和标准化结果
    C_c = coset_enumeration_c(M, [a])
    C_c.compress(); C_c.standardize()
    # 预期的余陪表
    table4 = [[0, 0, 0, 0]]
    assert C_r.table == table4
    assert C_c.table == table4


def test_look_ahead():
    # Section 3.2 [Test Example] Example (d) from [2]
    F, a, b, c = free_group("a, b, c")
    # 创建自由群 FpGroup 对象
    f = FpGroup(F, [a**11, b**5, c**4, (a*c)**3, b**2*c**-1*b**-1*c, a**4*b**-1*a**-1*b])
    # 定义生成元 H
    H = [c, b, c**2]
    # 预期的余陪表
    table0 = [[1, 2, 0, 0, 0, 0],
              [3, 0, 4, 5, 6, 7],
              [0, 8, 9, 10, 11, 12],
              [5, 1, 10, 13, 14, 15],
              [16, 5, 16, 1, 17, 18],
              [4, 3, 1, 8, 19, 20],
              [12, 21, 22, 23, 24, 1],
              [25, 26, 27, 28, 1, 24],
              [2, 10, 5, 16, 22, 28],
              [10, 13, 13, 2, 29, 30]]
    # 设置最大堆栈大小
    CosetTable.max_stack_size = 10
    # 使用 Felsch 策略进行余陪列举
    C_c = coset_enumeration_c(f, H)
    C_c.compress(); C_c.standardize()
    # 断言余陪表的前 10 行
    assert C_c.table[: 10] == table0

def test_modified_methods():
    # 这是一个空函数，用于测试修改后的方法，没有具体的实现和测试内容
    pass
    '''
    Tests for modified coset table methods.
    Example 5.7 from [1] Holt, D., Eick, B., O'Brien
    "Handbook of Computational Group Theory".
    '''
    
    # 创建自由群 F 和自由生成元 x, y
    F, x, y = free_group("x, y")
    
    # 创建 FpGroup，使用 F 作为自由群，生成的关系为 x^3, y^5, (xy)^2
    f = FpGroup(F, [x**3, y**5, (x*y)**2)
    
    # 定义 H 作为列表，包含两个群元 x*y 和 x^(-1)*y^(-1)*xyx
    H = [x*y, x**-1*y**-1*x*y*x]
    
    # 创建 CosetTable 对象 C，使用 f 和 H 初始化
    C = CosetTable(f, H)
    
    # 对 C 进行修改定义操作，设置第 0 行为 x
    C.modified_define(0, x)
    
    # 获取群的单位元素
    identity = C._grp.identity
    
    # 获取群的第一个生成元素
    a_0 = C._grp.generators[0]
    
    # 获取群的第二个生成元素
    a_1 = C._grp.generators[1]
    
    # 断言检查表 P 的初始化状态
    assert C.P == [[identity, None, None, None],
                   [None, identity, None, None]]
    
    # 断言检查表 table 的初始化状态
    assert C.table == [[1, None, None, None],
                       [None, 0, None, None]]
    
    # 对 C 进行修改定义操作，设置第 1 行为 x
    C.modified_define(1, x)
    
    # 断言检查修改后的表 table 的状态
    assert C.table == [[1, None, None, None],
                       [2, 0, None, None],
                       [None, 1, None, None]]
    
    # 断言检查修改后的表 P 的状态
    assert C.P == [[identity, None, None, None],
                   [identity, identity, None, None],
                   [None, identity, None, None]]
    
    # 对 C 进行修改扫描操作，使用 x^3 替换第 0 行的内容
    C.modified_scan(0, x**3, C._grp.identity, fill=False)
    
    # 断言检查修改后的表 P 的状态
    assert C.P == [[identity, identity, None, None],
                   [identity, identity, None, None],
                   [identity, identity, None, None]]
    
    # 断言检查修改后的表 table 的状态
    assert C.table == [[1, 2, None, None],
                       [2, 0, None, None],
                       [0, 1, None, None]]
    
    # 对 C 进行修改扫描操作，使用 xy 替换第 0 行的内容
    C.modified_scan(0, x*y, C._grp.generators[0], fill=False)
    
    # 断言检查修改后的表 P 的状态
    assert C.P == [[identity, identity, None, a_0**-1],
                   [identity, identity, a_0, None],
                   [identity, identity, None, None]]
    
    # 断言检查修改后的表 table 的状态
    assert C.table == [[1, 2, None, 1],
                       [2, 0, 0, None],
                       [0, 1, None, None]]
    
    # 对 C 进行修改定义操作，设置第 2 行为 y^(-1)
    C.modified_define(2, y**-1)
    
    # 断言检查修改后的表 table 的状态
    assert C.table == [[1, 2, None, 1],
                       [2, 0, 0, None],
                       [0, 1, None, 3],
                       [None, None, 2, None]]
    
    # 断言检查修改后的表 P 的状态
    assert C.P == [[identity, identity, None, a_0**-1],
                   [identity, identity, a_0, None],
                   [identity, identity, None, identity],
                   [None, None, identity, None]]
    
    # 对 C 进行修改扫描操作，使用 x^(-1)*y^(-1)*xyx 替换第 0 行的内容
    C.modified_scan(0, x**-1*y**-1*x*y*x, C._grp.generators[1])
    
    # 断言检查修改后的表 table 的状态
    assert C.table == [[1, 2, None, 1],
                       [2, 0, 0, None],
                       [0, 1, None, 3],
                       [3, 3, 2, None]]
    
    # 断言检查修改后的表 P 的状态
    assert C.P == [[identity, identity, None, a_0**-1],
                   [identity, identity, a_0, None],
                   [identity, identity, None, identity],
                   [a_1, a_1**-1, identity, None]]
    
    # 对 C 进行修改扫描操作，使用 (xy)^2 替换第 2 行的内容
    C.modified_scan(2, (x*y)**2, C._grp.identity)
    
    # 断言检查修改后的表 table 的状态
    assert C.table == [[1, 2, 3, 1],
                       [2, 0, 0, None],
                       [0, 1, None, 3],
                       [3, 3, 2, 0]]
    
    # 断言检查修改后的表 P 的状态
    assert C.P == [[identity, identity, a_1**-1, a_0**-1],
                   [identity, identity, a_0, None],
                   [identity, identity, None, identity],
                   [a_1, a_1**-1, identity, a_1]]
    
    # 对 C 进行修改定义操作，设置第 2 行为 y
    C.modified_define(2, y)
    # 断言，验证表格 C.table 的值是否与预期相符
    assert C.table == [[1, 2, 3, 1],
                        [2, 0, 0, None],
                        [0, 1, 4, 3],
                        [3, 3, 2, 0],
                        [None, None, None, 2]]
    
    # 断言，验证表格 C.P 的值是否与预期相符
    assert C.P == [[identity, identity, a_1**-1, a_0**-1],
                    [identity, identity, a_0, None],
                    [identity, identity, identity, identity],
                    [a_1, a_1**-1, identity, a_1],
                    [None, None, None, identity]]

    # 调用 C 对象的 modified_scan 方法，对表格进行修改
    C.modified_scan(0, y**5, C._grp.identity)
    
    # 再次断言，验证修改后的表格 C.table 的值是否与预期相符
    assert C.table == [[1, 2, 3, 1], [2, 0, 0, 4], [0, 1, 4, 3], [3, 3, 2, 0], [None, None, 1, 2]]
    
    # 再次断言，验证修改后的表格 C.P 的值是否与预期相符
    assert C.P == [[identity, identity, a_1**-1, a_0**-1],
                    [identity, identity, a_0, a_0*a_1**-1],
                    [identity, identity, identity, identity],
                    [a_1, a_1**-1, identity, a_1],
                    [None, None, a_1*a_0**-1, identity]]

    # 调用 C 对象的 modified_scan 方法，再次修改表格
    C.modified_scan(1, (x*y)**2, C._grp.identity)
    
    # 再次断言，验证再次修改后的表格 C.table 的值是否与预期相符
    assert C.table == [[1, 2, 3, 1],
                        [2, 0, 0, 4],
                        [0, 1, 4, 3],
                        [3, 3, 2, 0],
                        [4, 4, 1, 2]]
    
    # 再次断言，验证再次修改后的表格 C.P 的值是否与预期相符
    assert C.P == [[identity, identity, a_1**-1, a_0**-1],
                    [identity, identity, a_0, a_0*a_1**-1],
                    [identity, identity, identity, identity],
                    [a_1, a_1**-1, identity, a_1],
                    [a_0*a_1**-1, a_1*a_0**-1, a_1*a_0**-1, identity]]

    # 修改余类枚举的测试
    # 创建 FpGroup 对象 f，包含生成元 x 和关系 x**3, y**3, x**-1*y**-1*x*y
    f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    
    # 对 f 使用标准余类枚举算法，指定生成元 [x]
    C = coset_enumeration_r(f, [x])
    
    # 对 f 使用修改后的余类枚举算法，指定生成元 [x]
    C_m = modified_coset_enumeration_r(f, [x])
    
    # 断言，验证修改后的余类枚举 C_m 的表格是否与标准余类枚举 C 的表格相同
    assert C_m.table == C.table
```