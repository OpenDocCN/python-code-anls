# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_tensor_can.py`

```
from sympy.combinatorics.permutations import Permutation, Perm
from sympy.combinatorics.tensor_can import (perm_af_direct_product, dummy_sgs,
    riemann_bsgs, get_symmetric_group_sgs, canonicalize, bsgs_direct_product)
from sympy.combinatorics.testutil import canonicalize_naive, graph_certificate
from sympy.testing.pytest import skip, XFAIL

def test_perm_af_direct_product():
    gens1 = [[1,0,2,3], [0,1,3,2]]
    # 断言调用 perm_af_direct_product 函数，检查其返回的结果是否符合预期
    assert perm_af_direct_product(gens1, gens2, 0) == [[1, 0, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5], [0, 1, 2, 3, 5, 4]]

    gens1 = [[1,0,2,3,5,4], [0,1,3,2,4,5]]
    gens2 = [[1,0,2,3]]
    # 断言调用 perm_af_direct_product 函数，检查其返回的结果是否符合预期
    assert perm_af_direct_product(gens1, gens2, 0) == [[1, 0, 2, 3, 4, 5, 7, 6], [0, 1, 3, 2, 4, 5, 6, 7], [0, 1, 2, 3, 5, 4, 6, 7]]

def test_dummy_sgs():
    a = dummy_sgs([1,2], 0, 4)
    # 断言调用 dummy_sgs 函数，检查其返回的结果是否符合预期
    assert a == [[0,2,1,3,4,5]]

    a = dummy_sgs([2,3,4,5], 0, 8)
    # 断言调用 dummy_sgs 函数，检查其返回的结果是否符合预期，使用 Perm 对象转换成 _array_form 表示的列表
    assert a == [x._array_form for x in [Perm(9)(2,3), Perm(9)(4,5), Perm(9)(2,4)(3,5)]]

    a = dummy_sgs([2,3,4,5], 1, 8)
    # 断言调用 dummy_sgs 函数，检查其返回的结果是否符合预期，使用 Perm 对象转换成 _array_form 表示的列表
    assert a == [x._array_form for x in [Perm(2,3)(8,9), Perm(4,5)(8,9), Perm(9)(2,4)(3,5)]]

def test_get_symmetric_group_sgs():
    # 检查对 get_symmetric_group_sgs 函数的调用是否符合预期，返回元组形式的生成器和基
    assert get_symmetric_group_sgs(2) == ([0], [Permutation(3)(0,1)])
    assert get_symmetric_group_sgs(2, 1) == ([0], [Permutation(0,1)(2,3)])
    assert get_symmetric_group_sgs(3) == ([0,1], [Permutation(4)(0,1), Permutation(4)(1,2)])
    assert get_symmetric_group_sgs(3, 1) == ([0,1], [Permutation(0,1)(3,4), Permutation(1,2)(3,4)])
    assert get_symmetric_group_sgs(4) == ([0,1,2], [Permutation(5)(0,1), Permutation(5)(1,2), Permutation(5)(2,3)])
    assert get_symmetric_group_sgs(4, 1) == ([0,1,2], [Permutation(0,1)(4,5), Permutation(1,2)(4,5), Permutation(2,3)(4,5)])

def test_canonicalize_no_slot_sym():
    # 对 canonicalize_no_slot_sym 函数进行测试

    # 第一种情况，对称性不改变
    base1, gens1 = get_symmetric_group_sgs(1)
    dummies = [0, 1]
    g = Permutation([1,0,2,3])
    # 调用 canonicalize 函数，检查其返回的结果是否符合预期
    can = canonicalize(g, dummies, 0, (base1,gens1,1,0), (base1,gens1,1,0))
    assert can == [0,1,2,3]

    # 第二种情况，对称性不改变，简化的参数输入形式
    can = canonicalize(g, dummies, 0, (base1, gens1, 2, None))
    assert can == [0,1,2,3]

    # 第三种情况，有反对称度的度规
    can = canonicalize(g, dummies, 1, (base1,gens1,1,0), (base1,gens1,1,0))
    assert can == [0,1,3,2]

    # 第四种情况，无自由指标的对称度
    g = Permutation([0,1,2,3])
    dummies = []
    t0 = t1 = (base1, gens1, 1, 0)
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [0,1,2,3]

    # 对称度的顺序相反
    g = Permutation([1,0,2,3])
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [1,0,2,3]

    # 第五种情况，对称的 A
    # T_c = A^{b}_{d0}*A^{d0, a} order a,b,d0,-d0; T_c = A^{a d0}*A{b}_{d0}
    # g = [1,3,2,0,4,5]; can = [0,2,1,3,4,5]
    # 获取对称群的生成器和基础，针对群阶为2
    base2, gens2 = get_symmetric_group_sgs(2)
    # 设置虚拟变量
    dummies = [2,3]
    # 创建置换对象g=[1,3,2,0,4,5]
    g = Permutation([1,3,2,0,4,5])
    # 对g进行规范化处理，使用对称性metric=0
    can = canonicalize(g, dummies, 0, (base2, gens2, 2, 0))
    # 断言规范化结果符合预期
    assert can == [0, 2, 1, 3, 4, 5]
    
    # 使用反对称metric进行规范化
    can = canonicalize(g, dummies, 1, (base2, gens2, 2, 0))
    assert can == [0, 2, 1, 3, 4, 5]
    
    # 创建置换对象g=[0,3,2,1,4,5]
    g = Permutation([0,3,2,1,4,5])
    # 使用反对称metric进行规范化
    can = canonicalize(g, dummies, 1, (base2, gens2, 2, 0))
    assert can == [0, 2, 1, 3, 5, 4]

    # 设置虚拟变量
    dummies = [2,3]
    # 创建置换对象g=[1,3,2,0,4,5]
    g = Permutation([1,3,2,0,4,5])
    # 使用对称metric进行规范化，同时应用两组变量
    can = canonicalize(g, dummies, 0, (base2,gens2,1,0), (base2,gens2,1,0))
    assert can == [1,2,0,3,4,5]
    
    # 使用反对称metric进行规范化，同时应用两组变量
    can = canonicalize(g, dummies, 1, (base2,gens2,1,0), (base2,gens2,1,0))
    assert can == [1,2,0,3,5,4]

    # 获取对称群的生成器和基础，分别针对群阶为1和2
    base1, gens1 = get_symmetric_group_sgs(1)
    base2, gens2 = get_symmetric_group_sgs(2)
    # 创建置换对象g=[2,1,0,3,4,5]
    g = Permutation([2,1,0,3,4,5])
    # 设置虚拟变量
    dummies = [0,1,2,3]
    t0 = (base2, gens2, 1, 0)
    t1 = t2 = (base1, gens1, 1, 0)
    # 使用多组变量进行规范化处理
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [0, 2, 1, 3, 4, 5]

    # 创建置换对象g=[2,1,0,3,4,5]
    g = Permutation([2,1,0,3,4,5])
    # 设置虚拟变量
    dummies = [0,1,2,3]
    # 使用特定的非对称变量进行规范化处理
    t0 = ([], [Permutation(list(range(4)))], 1, 0)
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [0,2,3,1,4,5]
    
    # 设置虚拟变量
    t0 = t1 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [0,1,2,3]
    # 创建置换对象g=[2,1,3,0,4,5]
    g = Permutation([2,1,3,0,4,5])
    # 使用多组变量进行规范化处理
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [0, 2, 1, 3, 4, 5]
    
    # 创建置换对象g=[1,2,3,0,4,5]
    g = Permutation([1,2,3,0,4,5])
    # 使用多组变量进行规范化处理
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [0,2,3,1,4,5]

    # 设置虚拟变量
    t0 = t1 = t2 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [2,3,4,5]
    # 创建置换对象g=[4,2,0,3,5,1,6,7]
    g = Permutation([4,2,0,3,5,1,6,7])
    # 使用多组变量进行规范化处理
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [2,4,0,5,3,1,6,7]

    # 设置虚拟变量
    t0 = (base2,gens2,1,0)
    t1 = t2 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [2,3,4,5]
    # 创建置换对象g=[4,2,0,3,5,1,6,7]
    g = Permutation([4,2,0,3,5,1,6,7])
    # 使用多组变量进行规范化处理
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    # 断言：验证 can 是否等于 [2,4,0,3,5,1,6,7]
    assert can == [2,4,0,3,5,1,6,7]

    # A 对称，C 对称，B 无对称性
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # g=[4,2,0,3,5,1,6,7]
    # T_c = A^{d0 d1}*B_{a d0}*C_{b d1}; can = [2,4,0,3,1,5,6,7]
    t0 = t2 = (base2,gens2,1,0)
    t1 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [2,3,4,5]
    g = Permutation([4,2,0,3,5,1,6,7])
    # 规范化 g，使用指定的虚拟索引和参数，返回规范化后的列表
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    # 断言：验证 can 是否等于 [2,4,0,3,1,5,6,7]
    assert can == [2,4,0,3,1,5,6,7]

    # A 对称，B 无对称性，C 反对称
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # g=[4,2,0,3,5,1,6,7]
    # T_c = -A^{d0 d1}*B_{a d0}*C_{b d1}; can = [2,4,0,3,1,5,7,6]
    t0 = (base2,gens2, 1, 0)
    t1 = ([], [Permutation(list(range(4)))], 1, 0)
    # 获取对称群的生成元和基础
    base2a, gens2a = get_symmetric_group_sgs(2, 1)
    t2 = (base2a, gens2a, 1, 0)
    dummies = [2,3,4,5]
    g = Permutation([4,2,0,3,5,1,6,7])
    # 规范化 g，使用指定的虚拟索引和参数，返回规范化后的列表
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    # 断言：验证 can 是否等于 [2,4,0,3,1,5,7,6]
    assert can == [2,4,0,3,1,5,7,6]
def test_canonicalize_no_dummies():
    base1, gens1 = get_symmetric_group_sgs(1)
    base2, gens2 = get_symmetric_group_sgs(2)
    base2a, gens2a = get_symmetric_group_sgs(2, 1)

    # A commuting
    # A^c A^b A^a; ord = [a,b,c]; g = [2,1,0,3,4]
    # T_c = A^a A^b A^c; can = list(range(5))
    g = Permutation([2,1,0,3,4])
    can = canonicalize(g, [], 0, (base1, gens1, 3, 0))
    assert can == list(range(5))

    # A anticommuting
    # A^c A^b A^a; ord = [a,b,c]; g = [2,1,0,3,4]
    # T_c = -A^a A^b A^c; can = [0,1,2,4,3]
    g = Permutation([2,1,0,3,4])
    can = canonicalize(g, [], 0, (base1, gens1, 3, 1))
    assert can == [0,1,2,4,3]

    # A commuting and symmetric
    # A^{b,d}*A^{c,a}; ord = [a,b,c,d]; g = [1,3,2,0,4,5]
    # T_c = A^{a c}*A^{b d}; can = [0,2,1,3,4,5]
    g = Permutation([1,3,2,0,4,5])
    can = canonicalize(g, [], 0, (base2, gens2, 2, 0))
    assert can == [0,2,1,3,4,5]

    # A anticommuting and symmetric
    # A^{b,d}*A^{c,a}; ord = [a,b,c,d]; g = [1,3,2,0,4,5]
    # T_c = -A^{a c}*A^{b d}; can = [0,2,1,3,5,4]
    g = Permutation([1,3,2,0,4,5])
    can = canonicalize(g, [], 0, (base2, gens2, 2, 1))
    assert can == [0,2,1,3,5,4]

    # A^{c,a}*A^{b,d} ; g = [2,0,1,3,4,5]
    # T_c = A^{a c}*A^{b d}; can = [0,2,1,3,4,5]
    g = Permutation([2,0,1,3,4,5])
    can = canonicalize(g, [], 0, (base2, gens2, 2, 1))
    assert can == [0,2,1,3,4,5]

def test_no_metric_symmetry():
    # no metric symmetry
    # A^d1_d0 * A^d0_d1; ord = [d0,-d0,d1,-d1]; g= [2,1,0,3,4,5]
    # T_c = A^d0_d1 * A^d1_d0; can = [0,3,2,1,4,5]
    g = Permutation([2,1,0,3,4,5])
    can = canonicalize(g, list(range(4)), None, [[], [Permutation(list(range(4)))], 2, 0])
    assert can == [0,3,2,1,4,5]

    # A^d1_d2 * A^d0_d3 * A^d2_d1 * A^d3_d0
    # ord = [d0,-d0,d1,-d1,d2,-d2,d3,-d3]
    #        0    1  2  3  4   5   6   7
    # g = [2,5,0,7,4,3,6,1,8,9]
    # T_c = A^d0_d1 * A^d1_d0 * A^d2_d3 * A^d3_d2
    # can = [0,3,2,1,4,7,6,5,8,9]
    g = Permutation([2,5,0,7,4,3,6,1,8,9])
    can = canonicalize(g, list(range(8)), None, [[], [Permutation(list(range(4)))], 4, 0])
    assert can == [0,3,2,1,4,7,6,5,8,9]

    # A^d0_d2 * A^d1_d3 * A^d3_d0 * A^d2_d1
    # g = [0,5,2,7,6,1,4,3,8,9]
    # T_c = A^d0_d1 * A^d1_d2 * A^d2_d3 * A^d3_d0
    # can = [0,3,2,5,4,7,6,1,8,9]
    g = Permutation([0,5,2,7,6,1,4,3,8,9])
    can = canonicalize(g, list(range(8)), None, [[], [Permutation(list(range(4)))], 4, 0])
    assert can == [0,3,2,5,4,7,6,1,8,9]

    # Example with 16 elements
    g = Permutation([12,7,10,3,14,13,4,11,6,1,2,9,0,15,8,5,16,17])
    can = canonicalize(g, list(range(16)), None, [[], [Permutation(list(range(4)))], 8, 0])
    assert can == [0,3,2,5,4,7,6,1,8,11,10,13,12,15,14,9,16,17]

def test_canonical_free():
    # t = A^{d0 a1}*A_d0^a0
    # ord = [a0,a1,d0,-d0];  g = [2,1,3,0,4,5]; dummies = [[2,3]]
    # t_c = A_d0^a0*A^{d0 a1}
    g = Permutation([2,1,3,0,4,5])
    can = canonicalize(g, list(range(4)), None, [[], [Permutation(list(range(4)))], 2, 0])
    assert can == [0, 3, 2, 1, 4, 5]
    # 定义一个数组变量 can，存储排列 [3,0, 2,1, 4,5]
    # 创建一个 Permutation 对象 g，表示排列 [2,1,3,0,4,5]
    # 定义一个二维数组 dummies，包含一个子数组 [2,3]
    # 调用 canonicalize 函数，对排列 g 进行规范化处理，使用 dummies 和其他参数进行处理
    # 断言检查规范化后的结果 can 是否等于期望值 [3,0, 2,1, 4,5]
    g = Permutation([2,1,3,0,4,5])
    dummies = [[2,3]]
    can = canonicalize(g, dummies, [None], ([], [Permutation(3)], 2, 0))
    assert can == [3,0, 2,1, 4,5]
def test_canonicalize1():
    base1, gens1 = get_symmetric_group_sgs(1)
    base1a, gens1a = get_symmetric_group_sgs(1, 1)
    base2, gens2 = get_symmetric_group_sgs(2)
    base3, gens3 = get_symmetric_group_sgs(3)
    base2a, gens2a = get_symmetric_group_sgs(2, 1)
    base3a, gens3a = get_symmetric_group_sgs(3, 1)

    # A_d0*A^d0; ord = [d0,-d0]; g = [1,0,2,3]
    # T_c = A^d0*A_d0; can = [0,1,2,3]
    g = Permutation([1,0,2,3])
    can = canonicalize(g, [0, 1], 0, (base1, gens1, 2, 0))
    assert can == list(range(4))

    # A commuting
    # A_d0*A_d1*A_d2*A^d2*A^d1*A^d0; ord=[d0,-d0,d1,-d1,d2,-d2]
    # g = [1,3,5,4,2,0,6,7]
    # T_c = A^d0*A_d0*A^d1*A_d1*A^d2*A_d2; can = list(range(8))
    g = Permutation([1,3,5,4,2,0,6,7])
    can = canonicalize(g, list(range(6)), 0, (base1, gens1, 6, 0))
    assert can == list(range(8))

    # A anticommuting
    # A_d0*A_d1*A_d2*A^d2*A^d1*A^d0; ord=[d0,-d0,d1,-d1,d2,-d2]
    # g = [1,3,5,4,2,0,6,7]
    # T_c 0;  can = 0
    g = Permutation([1,3,5,4,2,0,6,7])
    can = canonicalize(g, list(range(6)), 0, (base1, gens1, 6, 1))
    assert can == 0
    can1 = canonicalize_naive(g, list(range(6)), 0, (base1, gens1, 6, 1))
    assert can1 == 0

    # A commuting symmetric
    # A^{d0 b}*A^a_d1*A^d1_d0; ord=[a,b,d0,-d0,d1,-d1]
    # g = [2,1,0,5,4,3,6,7]
    # T_c = A^{a d0}*A^{b d1}*A_{d0 d1}; can = [0,2,1,4,3,5,6,7]
    g = Permutation([2,1,0,5,4,3,6,7])
    can = canonicalize(g, list(range(2,6)), 0, (base2, gens2, 3, 0))
    assert can == [0,2,1,4,3,5,6,7]

    # A, B commuting symmetric
    # A^{d0 b}*A^d1_d0*B^a_d1; ord=[a,b,d0,-d0,d1,-d1]
    # g = [2,1,4,3,0,5,6,7]
    # T_c = A^{b d0}*A_d0^d1*B^a_d1; can = [1,2,3,4,0,5,6,7]
    g = Permutation([2,1,4,3,0,5,6,7])
    can = canonicalize(g, list(range(2,6)), 0, (base2,gens2,2,0), (base2,gens2,1,0))
    assert can == [1,2,3,4,0,5,6,7]

    # A commuting symmetric
    # A^{d1 d0 b}*A^{a}_{d1 d0}; ord=[a,b, d0,-d0,d1,-d1]
    # g = [4,2,1,0,5,3,6,7]
    # T_c = A^{a d0 d1}*A^{b}_{d0 d1}; can = [0,2,4,1,3,5,6,7]
    g = Permutation([4,2,1,0,5,3,6,7])
    can = canonicalize(g, list(range(2,6)), 0, (base3, gens3, 2, 0))
    assert can == [0,2,4,1,3,5,6,7]

    # A^{d3 d0 d2}*A^a0_{d1 d2}*A^d1_d3^a1*A^{a2 a3}_d0
    # ord = [a0,a1,a2,a3,d0,-d0,d1,-d1,d2,-d2,d3,-d3]
    #        0   1  2  3  4  5  6   7  8   9  10  11
    # g = [10,4,8, 0,7,9, 6,11,1, 2,3,5, 12,13]
    # T_c = A^{a0 d0 d1}*A^a1_d0^d2*A^{a2 a3 d3}*A_{d1 d2 d3}
    # can = [0,4,6, 1,5,8, 2,3,10, 7,9,11, 12,13]
    g = Permutation([10,4,8, 0,7,9, 6,11,1, 2,3,5, 12,13])
    can = canonicalize(g, list(range(4,12)), 0, (base3, gens3, 4, 0))
    assert can == [0,4,6, 1,5,8, 2,3,10, 7,9,11, 12,13]

    # A commuting symmetric, B antisymmetric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # ord = [d0,-d0,d1,-d1,d2,-d2,d3,-d3]
    # g = [0,2,4,5,7,3,1,6,8,9]
    # in this example and in the next three,
    # renaming dummy indices and using symmetry of A,
    # T = A^{d0 d1 d2} * A_{d0 d1 d3} * B_d2^d3
    # can = 0
    # 定义置换对象 g，置换列表表示置换顺序
    g = Permutation([0,2,4,5,7,3,1,6,8,9])
    # 使用 canonicalize 函数对 g 进行规范化处理，返回结果存入 can
    can = canonicalize(g, list(range(8)), 0, (base3, gens3,2,0), (base2a,gens2a,1,0))
    # 断言确保 can 的值为 0
    assert can == 0

    # A anticommuting symmetric, B anticommuting
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = A^{d0 d1 d2} * A_{d0 d1}^d3 * B_{d2 d3}
    # 定义置换对象 g，期望 can 的结果为 [0,2,4, 1,3,6, 5,7, 8,9]
    can = canonicalize(g, list(range(8)), 0, (base3, gens3,2,1), (base2a,gens2a,1,0))
    # 断言确保 can 的值符合预期结果
    assert can == [0,2,4, 1,3,6, 5,7, 8,9]

    # A anticommuting symmetric, B antisymmetric commuting, antisymmetric metric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = -A^{d0 d1 d2} * A_{d0 d1}^d3 * B_{d2 d3}
    # 定义置换对象 g，期望 can 的结果为 [0,2,4, 1,3,6, 5,7, 9,8]
    can = canonicalize(g, list(range(8)), 1, (base3, gens3,2,1), (base2a,gens2a,1,0))
    # 断言确保 can 的值符合预期结果
    assert can == [0,2,4, 1,3,6, 5,7, 9,8]

    # A anticommuting symmetric, B anticommuting anticommuting,
    # no metric symmetry
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = A^{d0 d1 d2} * A_{d0 d1 d3} * B_d2^d3
    # 定义置换对象 g，期望 can 的结果为 [0,2,4,1,3,7,5,6,8,9]
    can = canonicalize(g, list(range(8)), None, (base3, gens3,2,1), (base2a,gens2a,1,0))
    # 断言确保 can 的值符合预期结果
    assert can == [0,2,4,1,3,7,5,6,8,9]

    # Gamma anticommuting
    # Gamma_{mu nu} * gamma^rho * Gamma^{nu mu alpha}
    # ord = [alpha, rho, mu,-mu,nu,-nu]
    # g = [3,5,1,4,2,0,6,7]
    # T_c = -Gamma^{mu nu} * gamma^rho * Gamma_{alpha mu nu}
    # 定义置换对象 g，期望 can 的结果为 [2,4,1,0,3,5,7,6]
    g = Permutation([3,5,1,4,2,0,6,7])
    t0 = (base2a, gens2a, 1, None)
    t1 = (base1, gens1, 1, None)
    t2 = (base3a, gens3a, 1, None)
    can = canonicalize(g, list(range(2, 6)), 0, t0, t1, t2)
    # 断言确保 can 的值符合预期结果
    assert can == [2,4,1,0,3,5,7,6]

    # Gamma_{mu nu} * Gamma^{gamma beta} * gamma_rho * Gamma^{nu mu alpha}
    # ord = [alpha, beta, gamma, -rho, mu,-mu,nu,-nu]
    #         0      1      2     3    4   5   6  7
    # g = [5,7,2,1,3,6,4,0,8,9]
    # T_c = Gamma^{mu nu} * Gamma^{beta gamma} * gamma_rho * Gamma^alpha_{mu nu}
    # 定义置换对象 g，期望 can 的结果为 [4,6,1,2,3,0,5,7,8,9]
    t0 = (base2a, gens2a, 2, None)
    g = Permutation([5,7,2,1,3,6,4,0,8,9])
    can = canonicalize(g, list(range(4, 8)), 0, t0, t1, t2)
    # 断言确保 can 的值符合预期结果
    assert can == [4,6,1,2,3,0,5,7,8,9]

    # f^a_{b,c} antisymmetric in b,c; A_mu^a no symmetry
    # f^c_{d a} * f_{c e b} * A_mu^d * A_nu^a * A^{nu e} * A^{mu b}
    # ord = [mu,-mu,nu,-nu,a,-a,b,-b,c,-c,d,-d, e, -e]
    #         0  1  2   3  4  5 6  7 8  9 10 11 12 13
    # g = [8,11,5, 9,13,7, 1,10, 3,4, 2,12, 0,6, 14,15]
    # T_c = -f^{a b c} * f_a^{d e} * A^mu_b * A_{mu d} * A^nu_c * A_{nu e}
    # 定义置换对象 g，期望 can 的结果为 [4,6,8,5,10,12,0,7,1,11,2,9,3,13,15,14]
    g = Permutation([8,11,5, 9,13,7, 1,10, 3,4, 2,12, 0,6, 14,15])
    base_f, gens_f = bsgs_direct_product(base1, gens1, base2a, gens2a)
    base_A, gens_A = bsgs_direct_product(base1, gens1, base1, gens1)
    t0 = (base_f, gens_f, 2, 0)
    t1 = (base_A, gens_A, 4, 0)
    can = canonicalize(g, [list(range(4)), list(range(4, 14))], [0, 0], t0, t1)
    # 断言确保 can 的值符合预期结果
    assert can == [4,6,8,5,10,12,0,7,1,11,2,9,3,13,15,14]
    # 检查变量 can 中的值是否与预期列表匹配
    assert can == [4,6,8, 5,10,12, 0,7, 1,11, 2,9, 3,13, 15,14]
def test_riemann_invariants():
    baser, gensr = riemann_bsgs
    # 定义排列 g
    g = Permutation([0,2,3,1,4,5])
    # 使用 canonicalize 函数对 g 进行标准化处理，生成排列 can
    can = canonicalize(g, list(range(2, 4)), 0, (baser, gensr, 1, 0))
    # 断言标准化后的结果与预期结果相等
    assert can == [0,2,1,3,5,4]

    # 使用一个非最小的 BSGS 进行标准化
    can = canonicalize(g, list(range(2, 4)), 0, ([2, 0], [Permutation([1,0,2,3,5,4]), Permutation([2,3,0,1,4,5])], 1, 0))
    # 断言标准化后的结果与预期结果相等
    assert can == [0,2,1,3,5,4]

    """
    下面的测试在 test_riemann_invariants 和 test_riemann_invariants1 中
    使用了 xPerm 中的 xperm.c 和 cadabra 中的旧版本 test_xperm.cc 进行验证。

    [1] xPerm 由 J. M. Martin-Garcia 编写的 xperm.c
        http://www.xact.es/index.html
    [2] cadabra 由 Kasper Peeters 编写的 test_xperm.cc
        http://cadabra.phi-sci.com/
    """

    # 定义更复杂的排列 g
    g = Permutation([23,2,1,10,12,8,0,11,15,5,17,19,21,7,13,9,4,14,22,3,16,18,6,20,24,25])
    # 使用 canonicalize 函数对 g 进行标准化处理，生成排列 can
    can = canonicalize(g, list(range(24)), 0, (baser, gensr, 6, 0))
    # 断言标准化后的结果与预期结果相等
    assert can == [0,2,4,6,1,3,8,10,5,7,12,14,9,11,16,18,13,15,20,22,17,19,21,23,24,25]

    # 使用一个非最小的 BSGS 进行标准化
    can = canonicalize(g, list(range(24)), 0, ([2, 0], [Permutation([1,0,2,3,5,4]), Permutation([2,3,0,1,4,5])], 6, 0))
    # 断言标准化后的结果与预期结果相等
    assert can == [0,2,4,6,1,3,8,10,5,7,12,14,9,11,16,18,13,15,20,22,17,19,21,23,24,25]

    # 定义更复杂的排列 g
    g = Permutation([0,2,5,7,4,6,9,11,8,10,13,15,12,14,17,19,16,18,21,23,20,22,25,27,24,26,29,31,28,30,33,35,32,34,37,39,36,38,1,3,40,41])
    # 使用 canonicalize 函数对 g 进行标准化处理，生成排列 can
    can = canonicalize(g, list(range(40)), 0, (baser, gensr, 10, 0))
    # 断言标准化后的结果与预期结果相等
    assert can == [0,2,4,6,1,3,8,10,5,7,12,14,9,11,16,18,13,15,20,22,17,19,24,26,21,23,28,30,25,27,32,34,29,31,36,38,33,35,37,39,40,41]


@XFAIL
def test_riemann_invariants1():
    skip('takes too much time')
    baser, gensr = riemann_bsgs
    g = Permutation([17, 44, 11, 3, 0, 19, 23, 15, 38, 4, 25, 27, 43, 36, 22, 14, 8, 30, 41, 20, 2, 10, 12, 28, 18, 1, 29, 13, 37, 42, 33, 7, 9, 31, 24, 26, 39, 5, 34, 47, 32, 6, 21, 40, 35, 46, 45, 16, 48, 49])
    # 使用 canonicalize 函数对 g 进行标准化处理，生成排列 can
    can = canonicalize(g, list(range(48)), 0, (baser, gensr, 12, 0))
    # 断言标准化后的结果与预期结果相等
    assert can == [0, 2, 4, 6, 1, 3, 8, 10, 5, 7, 12, 14, 9, 11, 16, 18, 13, 15, 20, 22, 17, 19, 24, 26, 21, 23, 28, 30, 25, 27, 32, 34, 29, 31, 36, 38, 33, 35, 40, 42, 37, 39, 44, 46, 41, 43, 45, 47, 48, 49]

    g = Permutation([0,2,4,6, 7,8,10,12, 14,16,18,20, 19,22,24,26, 5,21,28,30, 32,34,36,38, 40,42,44,46, 13,48,50,52, 15,49,54,56, 17,33,41,58, 9,23,60,62, 29,35,63,64, 3,45,66,68, 25,37,47,57, 11,31,69,70, 27,39,53,72, 1,59,73,74, 55,61,67,76, 43,65,75,78, 51,71,77,79, 80,81])
    # 使用 canonicalize 函数对 g 进行标准化处理，生成排列 can
    can = canonicalize(g, list(range(80)), 0, (baser, gensr, 20, 0))
    # 断言语句用于检查条件是否满足，如果条件为假，则抛出异常
    assert can == [0,2,4,6, 1,8,10,12, 3,14,16,18, 5,20,22,24, 7,26,28,30, 9,15,32,34, 11,36,23,38, 13,40,42,44, 17,39,29,46, 19,48,43,50, 21,45,52,54, 25,56,33,58, 27,60,53,62, 31,51,64,66, 35,65,47,68, 37,70,49,72, 41,74,57,76, 55,67,59,78, 61,69,71,75, 63,79,73,77, 80,81]
    # 断言数组 `can` 是否与指定的列表相等，列表包含从 0 到 81 的数字
def test_riemann_products():
    # 获取Riemann基础和生成器
    baser, gensr = riemann_bsgs
    # 获取对称群S_1的基础和生成器
    base1, gens1 = get_symmetric_group_sgs(1)
    # 获取对称群S_2的基础和生成器
    base2, gens2 = get_symmetric_group_sgs(2)
    # 获取带有偏移的对称群S_2的基础和生成器
    base2a, gens2a = get_symmetric_group_sgs(2, 1)

    # R^{a b d0}_d0 = 0
    g = Permutation([0,1,2,3,4,5])
    # 规范化置换g，特定条件下返回0
    can = canonicalize(g, list(range(2,4)), 0, (baser, gensr, 1, 0))
    assert can == 0

    # R^{d0 b a}_d0 ; ord = [a,b,d0,-d0}; g = [2,1,0,3,4,5]
    # T_c = -R^{a d0 b}_d0;  can = [0,2,1,3,5,4]
    g = Permutation([2,1,0,3,4,5])
    # 规范化置换g，特定条件下返回列表[0,2,1,3,5,4]
    can = canonicalize(g, list(range(2, 4)), 0, (baser, gensr, 1, 0))
    assert can == [0,2,1,3,5,4]

    # R^d1_d2^b_d0 * R^{d0 a}_d1^d2; ord=[a,b,d0,-d0,d1,-d1,d2,-d2]
    # g = [4,7,1,3,2,0,5,6,8,9]
    # T_c = -R^{a d0 d1 d2}* R^b_{d0 d1 d2}
    # can = [0,2,4,6,1,3,5,7,9,8]
    g = Permutation([4,7,1,3,2,0,5,6,8,9])
    # 规范化置换g，特定条件下返回列表[0,2,4,6,1,3,5,7,9,8]
    can = canonicalize(g, list(range(2,8)), 0, (baser, gensr, 2, 0))
    assert can == [0,2,4,6,1,3,5,7,9,8]
    can1 = canonicalize_naive(g, list(range(2,8)), 0, (baser, gensr, 2, 0))
    assert can == can1

    # A symmetric commuting
    # R^{d6 d5}_d2^d1 * R^{d4 d0 d2 d3} * A_{d6 d0} A_{d3 d1} * A_{d4 d5}
    # g = [12,10,5,2, 8,0,4,6, 13,1, 7,3, 9,11,14,15]
    # T_c = -R^{d0 d1 d2 d3} * R_d0^{d4 d5 d6} * A_{d1 d4}*A_{d2 d5}*A_{d3 d6}
    g = Permutation([12,10,5,2,8,0,4,6,13,1,7,3,9,11,14,15])
    # 规范化置换g，特定条件下返回列表[0, 2, 4, 6, 1, 8, 10, 12, 3, 9, 5, 11, 7, 13, 15, 14]
    can = canonicalize(g, list(range(14)), 0, ((baser,gensr,2,0)), (base2,gens2,3,0))
    assert can == [0, 2, 4, 6, 1, 8, 10, 12, 3, 9, 5, 11, 7, 13, 15, 14]

    # R^{d2 a0 a2 d0} * R^d1_d2^{a1 a3} * R^{a4 a5}_{d0 d1}
    # ord = [a0,a1,a2,a3,a4,a5,d0,-d0,d1,-d1,d2,-d2]
    # can = [0, 6, 2, 8, 1, 3, 7, 10, 4, 5, 9, 11, 12, 13]
    # T_c = R^{a0 d0 a2 d1}*R^{a1 a3}_d0^d2*R^{a4 a5}_{d1 d2}
    g = Permutation([10,0,2,6,8,11,1,3,4,5,7,9,12,13])
    # 规范化置换g，特定条件下返回列表[0, 6, 2, 8, 1, 3, 7, 10, 4, 5, 9, 11, 12, 13]
    can = canonicalize(g, list(range(6,12)), 0, (baser, gensr, 3, 0))
    assert can == [0, 6, 2, 8, 1, 3, 7, 10, 4, 5, 9, 11, 12, 13]
    #can1 = canonicalize_naive(g, list(range(6,12)), 0, (baser, gensr, 3, 0))
    #assert can == can1

    # A^n_{i, j} antisymmetric in i,j
    # A_m0^d0_a1 * A_m1^a0_d0; ord = [m0,m1,a0,a1,d0,-d0]
    # g = [0,4,3,1,2,5,6,7]
    # T_c = -A_{m a1}^d0 * A_m1^a0_d0
    # can = [0,3,4,1,2,5,7,6]
    base, gens = bsgs_direct_product(base1, gens1, base2a, gens2a)
    dummies = list(range(4, 6))
    g = Permutation([0,4,3,1,2,5,6,7])
    # 规范化置换g，特定条件下返回列表[0, 3, 4, 1, 2, 5, 7, 6]
    can = canonicalize(g, dummies, 0, (base, gens, 2, 0))
    assert can == [0, 3, 4, 1, 2, 5, 7, 6]


    # A^n_{i, j} symmetric in i,j
    # A^m0_a0^d2 * A^n0_d2^d1 * A^n1_d1^d0 * A_{m0 d0}^a1
    # ord=[n0,n1,a0,a1, m0,-m0,d0,-d0,d1,-d1,d2,-d2]
    # g = [4,2,10, 0,11,8, 1,9,6, 5,7,3, 12,13]
    # T_c = A^{n0 d0 d1} * A^n1_d0^d2 * A^m0^a0_d1 * A_m0^a1_d2
    # can = [0, 6, 8, 1, 7, 10, 4, 2, 9, 5, 3, 11, 12, 13]
    g = Permutation([4,2,10,0,11,8,1,9,6,5,7,3,12,13])
    # 规范化置换g，特定条件下返回列表[0, 6, 8, 1, 7, 10, 4, 2, 9, 5, 3, 11, 12, 13]
    can = canonicalize(g, list(range(14)), 0, (baser, gensr, 4, 0))
    assert can == [0, 6, 8, 1, 7, 10, 4, 2, 9, 5, 3, 11, 12, 13]
    # 使用给定的函数进行直接乘积的 BSGS 算法，计算基数和生成元
    base, gens = bsgs_direct_product(base1, gens1, base2, gens2)
    # 创建一个包含指标范围的列表
    dummies = list(range(4, 12))
    # 创建一个置换对象 g，用于后续规范化处理
    g = Permutation([4,2,10, 0,11,8, 1,9,6, 5,7,3, 12,13])
    # 对给定的置换 g 进行规范化处理，使用指标列表 dummies
    can = canonicalize(g, dummies, 0, (base, gens, 4, 0))
    # 断言规范化处理后的结果是否符合预期值
    assert can == [0, 4, 6, 1, 5, 8, 10, 2, 7, 11, 3, 9, 12, 13]
    # 使用分离的指标列表 dummies 进行规范化处理
    dummies = [list(range(4, 6)), list(range(6,12))]
    # 创建一个对称性列表 sym，表示是否对应相应索引进行对称或反对称处理
    sym = [0, 0]
    # 再次对置换 g 进行规范化处理，使用分离的指标列表 dummies 和对称性列表 sym
    can = canonicalize(g, dummies, sym, (base, gens, 4, 0))
    # 断言规范化处理后的结果是否符合预期值
    assert can == [0, 6, 8, 1, 7, 10, 4, 2, 9, 5, 3, 11, 12, 13]
    # 再次使用不同的对称性列表 sym 进行规范化处理
    sym = [0, 1]
    # 再次对置换 g 进行规范化处理，使用相同的分离的指标列表 dummies 和新的对称性列表 sym
    # 由于指定了反对称的对称性，结果中的某些位置可能发生符号变化
    can = canonicalize(g, dummies, sym, (base, gens, 4, 0))
    # 断言规范化处理后的结果是否符合预期值
    assert can == [0, 6, 8, 1, 7, 10, 4, 2, 9, 5, 3, 11, 13, 12]
def test_graph_certificate():
    # 定义一个测试函数，用于测试从随机正则图构建的张量不变量；
    # 使用networkx检查图同构性

    import random

    def randomize_graph(size, g):
        # 定义一个函数，将图中节点的顺序随机打乱，并返回新的图
        p = list(range(size))
        random.shuffle(p)
        g1a = {}
        for k, v in g.items():
            g1a[p[k]] = [p[i] for i in v]
        return g1a

    # 示例图 g1 和 g2
    g1 = {0: [2, 3, 7], 1: [4, 5, 7], 2: [0, 4, 6], 3: [0, 6, 7], 4: [1, 2, 5], 5: [1, 4, 6], 6: [2, 3, 5], 7: [0, 1, 3]}
    g2 = {0: [2, 3, 7], 1: [2, 4, 5], 2: [0, 1, 5], 3: [0, 6, 7], 4: [1, 5, 6], 5: [1, 2, 4], 6: [3, 4, 7], 7: [0, 3, 6]}

    # 计算图 g1 和 g2 的图证书
    c1 = graph_certificate(g1)
    c2 = graph_certificate(g2)

    # 断言 g1 和 g2 的图证书不相等
    assert c1 != c2

    # 将 g1 的节点顺序随机化，并重新计算其图证书
    g1a = randomize_graph(8, g1)
    c1a = graph_certificate(g1a)

    # 断言随机化后的 g1a 的图证书与原始 g1 的图证书相等
    assert c1 == c1a

    # 另一组示例图 g1 和 g2
    g1 = {0: [8, 1, 9, 7], 1: [0, 9, 3, 4], 2: [3, 4, 6, 7], 3: [1, 2, 5, 6], 4: [8, 1, 2, 5], 5: [9, 3, 4, 7], 6: [8, 2, 3, 7], 7: [0, 2, 5, 6], 8: [0, 9, 4, 6], 9: [8, 0, 5, 1]}
    g2 = {0: [1, 2, 5, 6], 1: [0, 9, 5, 7], 2: [0, 4, 6, 7], 3: [8, 9, 6, 7], 4: [8, 2, 6, 7], 5: [0, 9, 8, 1], 6: [0, 2, 3, 4], 7: [1, 2, 3, 4], 8: [9, 3, 4, 5], 9: [8, 1, 3, 5]}

    # 计算新的图 g1 和 g2 的图证书
    c1 = graph_certificate(g1)
    c2 = graph_certificate(g2)

    # 断言 g1 和 g2 的图证书不相等
    assert c1 != c2

    # 将 g1 的节点顺序随机化，并重新计算其图证书
    g1a = randomize_graph(10, g1)
    c1a = graph_certificate(g1a)

    # 断言随机化后的 g1a 的图证书与原始 g1 的图证书相等
    assert c1 == c1a
```