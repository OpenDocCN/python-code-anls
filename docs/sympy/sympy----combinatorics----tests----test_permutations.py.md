# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_permutations.py`

```
from itertools import permutations  # 导入 permutations 函数，用于生成序列的所有排列方式
from copy import copy  # 导入 copy 函数，用于复制对象

from sympy.core.expr import unchanged  # 导入 unchanged 类，表示表达式不变
from sympy.core.numbers import Integer  # 导入 Integer 类，表示整数
from sympy.core.relational import Eq  # 导入 Eq 类，表示等式
from sympy.core.symbol import Symbol  # 导入 Symbol 类，表示符号
from sympy.core.singleton import S  # 导入 S 类，表示单例
from sympy.combinatorics.permutations import \  # 导入排列相关的模块和函数
    Permutation, _af_parity, _af_rmul, _af_rmuln, AppliedPermutation, Cycle
from sympy.printing import sstr, srepr, pretty, latex  # 导入打印函数和格式化函数
from sympy.testing.pytest import raises, warns_deprecated_sympy  # 导入测试相关函数

rmul = Permutation.rmul  # 给变量 rmul 赋值为 Permutation 类的 rmul 静态方法
a = Symbol('a', integer=True)  # 创建一个整数符号 a

def test_Permutation():  # 定义测试函数 test_Permutation
    # 检查不自动填充0时是否引发 ValueError 异常
    raises(ValueError, lambda: Permutation([1]))
    p = Permutation([0, 1, 2, 3])  # 创建一个排列 p
    # 作为双射调用
    assert [p(i) for i in range(p.size)] == list(p)
    # 作为运算符调用
    assert p(list(range(p.size))) == list(p)
    # 作为函数调用
    assert list(p(1, 2)) == [0, 2, 1, 3]
    raises(TypeError, lambda: p(-1))  # 检查是否引发 TypeError 异常
    raises(TypeError, lambda: p(5))  # 检查是否引发 TypeError 异常
    # 转换为列表
    assert list(p) == list(range(4))
    assert p.copy() == p
    assert copy(p) == p
    assert Permutation(size=4) == Permutation(3)
    assert Permutation(Permutation(3), size=5) == Permutation(4)
    # 带有指定大小的循环形式
    assert Permutation([[1, 2]], size=4) == Permutation([[1, 2], [0], [3]])
    # 随机生成
    assert Permutation.random(2) in (Permutation([1, 0]), Permutation([0, 1]))

    p = Permutation([2, 5, 1, 6, 3, 0, 4])  # 创建一个排列 p
    q = Permutation([[1], [0, 3, 5, 6, 2, 4]])  # 创建一个排列 q
    assert len({p, p}) == 1
    r = Permutation([1, 3, 2, 0, 4, 6, 5])  # 创建一个排列 r
    ans = Permutation(_af_rmuln(*[w.array_form for w in (p, q, r)])).array_form  # 计算排列乘积的数组形式
    assert rmul(p, q, r).array_form == ans
    # 确保没有其他排列可以产生相同的结果
    for a, b, c in permutations((p, q, r)):
        if (a, b, c) == (p, q, r):
            continue
        assert rmul(a, b, c).array_form != ans

    assert p.support() == list(range(7))
    assert q.support() == [0, 2, 3, 4, 5, 6]
    assert Permutation(p.cyclic_form).array_form == p.array_form
    assert p.cardinality == 5040
    assert q.cardinality == 5040
    assert q.cycles == 2
    assert rmul(q, p) == Permutation([4, 6, 1, 2, 5, 3, 0])
    assert rmul(p, q) == Permutation([6, 5, 3, 0, 2, 4, 1])
    assert _af_rmul(p.array_form, q.array_form) == \
        [6, 5, 3, 0, 2, 4, 1]

    assert rmul(Permutation([[1, 2, 3], [0, 4]]),
                Permutation([[1, 2, 4], [0], [3]])).cyclic_form == \
        [[0, 4, 2], [1, 3]]
    assert q.array_form == [3, 1, 4, 5, 0, 6, 2]
    assert q.cyclic_form == [[0, 3, 5, 6, 2, 4]]
    assert q.full_cyclic_form == [[0, 3, 5, 6, 2, 4], [1]]
    assert p.cyclic_form == [[0, 2, 1, 5], [3, 6, 4]]
    t = p.transpositions()
    assert t == [(0, 5), (0, 1), (0, 2), (3, 4), (3, 6)]
    assert Permutation.rmul(*[Permutation(Cycle(*ti)) for ti in (t)])
    assert Permutation([1, 0]).transpositions() == [(0, 1)]

    assert p**13 == p
    assert q**0 == Permutation(list(range(q.size)))
    # 断言 q 的负二次方等于 q 的按位取反后的平方
    assert q**-2 == ~q**2
    # 断言 q 的平方等于特定排列 [5, 1, 0, 6, 3, 2, 4]
    assert q**2 == Permutation([5, 1, 0, 6, 3, 2, 4])
    # 断言 q 的立方等于 q 的平方乘以 q 自身
    assert q**3 == q**2*q
    # 断言 q 的四次方等于 q 的平方乘以 q 的平方
    assert q**4 == q**2*q**2

    # 创建排列 a 和 b
    a = Permutation(1, 3)
    b = Permutation(2, 0, 3)
    # 创建单位排列 I
    I = Permutation(3)
    # 断言 a 的逆等于 a 的负一次幂
    assert ~a == a**-1
    # 断言 a 乘以其逆等于单位排列 I
    assert a*~a == I
    # 断言 a 乘以 b 的逆等于 a 乘以 b 的按位取反
    assert a*b**-1 == a*~b

    # 使用排列操作符构建 ans
    ans = Permutation(0, 5, 3, 1, 6)(2, 4)
    # 断言 (p + q.rank()) 的 rank 等于 ans 的 rank
    assert (p + q.rank()).rank() == ans.rank()
    # 断言 (p + q.rank()) 的 _rank 等于 ans 的 rank
    assert (p + q.rank())._rank == ans.rank()
    # 断言 (q + p.rank()) 的 rank 等于 ans 的 rank
    assert (q + p.rank()).rank() == ans.rank()
    # 断言对于 p + Permutation(list(range(10))) 会引发 TypeError
    raises(TypeError, lambda: p + Permutation(list(range(10))))

    # 断言 (p - q.rank()) 的 rank 等于特定排列的 rank
    assert (p - q.rank()).rank() == Permutation(0, 6, 3, 1, 2, 5, 4).rank()
    # 断言 p 的 rank 减去 q 的 rank 小于 0，用于覆盖: 确保使用了 mod 操作
    assert p.rank() - q.rank() < 0
    # 断言 (q - p.rank()) 的 rank 等于特定排列的 rank
    assert (q - p.rank()).rank() == Permutation(1, 4, 6, 2)(3, 5).rank()

    # 断言 p 乘以 q 等于通过 _af_rmuln 函数处理后的排列
    assert p*q == Permutation(_af_rmuln(*[list(w) for w in (q, p)]))
    # 断言 p 乘以空排列等于 p 自身
    assert p*Permutation([]) == p
    # 断言 空排列 乘以 p 等于 p 自身
    assert Permutation([])*p == p
    # 断言 p 乘以包含一个置换的排列等于特定排列
    assert p*Permutation([[0, 1]]) == Permutation([2, 5, 0, 6, 3, 1, 4])
    # 断言 包含一个置换的排列 乘以 p 等于特定排列
    assert Permutation([[0, 1]])*p == Permutation([5, 2, 1, 6, 3, 0, 4])

    # 创建 pq 作为 p 和 q 的排列交换
    pq = p ^ q
    # 断言 pq 等于特定排列
    assert pq == Permutation([5, 6, 0, 4, 1, 2, 3])
    # 断言 pq 等于通过 rmul 函数处理后的排列
    assert pq == rmul(q, p, ~q)
    # 创建 qp 作为 q 和 p 的排列交换
    qp = q ^ p
    # 断言 qp 等于特定排列
    assert qp == Permutation([4, 3, 6, 2, 1, 5, 0])
    # 断言 qp 等于通过 rmul 函数处理后的排列
    assert qp == rmul(p, q, ~p)
    # 断言对于 p ^ Permutation([]) 会引发 ValueError
    raises(ValueError, lambda: p ^ Permutation([]))

    # 断言 p 和 q 的 commutator 等于特定排列
    assert p.commutator(q) == Permutation(0, 1, 3, 4, 6, 5, 2)
    assert q.commutator(p) == Permutation(0, 2, 5, 6, 4, 3, 1)
    # 断言 p 和 q 的 commutator 等于 q 和 p 的 commutator 的逆
    assert p.commutator(q) == ~q.commutator(p)
    # 断言对于 p.commutator(Permutation([])) 会引发 ValueError
    raises(ValueError, lambda: p.commutator(Permutation([])))

    # 断言 p 的 atoms() 返回长度为 7
    assert len(p.atoms()) == 7
    # 断言 q 的 atoms() 返回集合 {0, 1, 2, 3, 4, 5, 6}
    assert q.atoms() == {0, 1, 2, 3, 4, 5, 6}

    # 断言 p 的 inversion_vector() 等于特定列表
    assert p.inversion_vector() == [2, 4, 1, 3, 1, 0]
    # 断言 q 的 inversion_vector() 等于特定列表
    assert q.inversion_vector() == [3, 1, 2, 2, 0, 1]

    # 断言从 p 的 inversion_vector() 构建的排列等于 p 自身
    assert Permutation.from_inversion_vector(p.inversion_vector()) == p
    # 断言从 q 的 inversion_vector() 构建的排列的 array_form 等于 q 的 array_form
    assert Permutation.from_inversion_vector(q.inversion_vector()).array_form \
        == q.array_form
    # 断言对于 Permutation.from_inversion_vector([0, 2]) 会引发 ValueError
    raises(ValueError, lambda: Permutation.from_inversion_vector([0, 2]))
    # 断言 Permutation(list(range(500, -1, -1))) 的 inversions() 等于 125250
    assert Permutation(list(range(500, -1, -1))).inversions() == 125250

    # 创建排列 s
    s = Permutation([0, 4, 1, 3, 2])
    # 断言 s 的 parity() 等于 0
    assert s.parity() == 0
    # 为了创建 _cyclic_form，获取 s 的 cyclic_form 属性
    _ = s.cyclic_form
    # 断言 s 的 _cyclic_form 的长度不等于 s 的 size，且 s 的 parity() 等于 0
    assert len(s._cyclic_form) != s.size and s.parity() == 0
    # 断言 s 不是奇排列
    assert not s.is_odd
    # 断言 s 是偶排列
    assert s.is_even
    # 断言 Permutation([0, 1, 4, 3, 2]) 的 parity() 等于 1
    assert Permutation([0, 1, 4, 3, 2]).parity() == 1
    # 断言 _af_parity([0, 4, 1, 3, 2]) 等于 0
    assert _af_parity([0, 4, 1, 3, 2]) == 0
    # 断言 _af_parity([0, 1, 4, 3, 2]) 等于 1
    assert _af_parity([0, 1, 4, 3, 2]) == 1

    # 创建排列 s
    s = Permutation([0])
    # 断言 s 是 Singleton
    assert s.is_Singleton
    # 断言
    # 断言 r 的升降序为空列表
    assert r.ascents() == []
    
    # 断言 p 的降序为 [1, 2, 5]
    assert p.descents() == [1, 2, 5]
    # 断言 q 的降序为 [0, 3, 5]
    assert q.descents() == [0, 3, 5]
    # 断言 r 的降序的排列是否是单位置换
    assert Permutation(r.descents()).is_Identity
    
    # 断言 p 的逆序数为 7
    assert p.inversions() == 7
    # 使用较长的排列进行归并排序的测试
    big = list(p) + list(range(p.max() + 1, p.max() + 130))
    # 断言大排列的逆序数为 7
    assert Permutation(big).inversions() == 7
    # 断言 p 的符号为 -1
    assert p.signature() == -1
    # 断言 q 的逆序数为 11
    assert q.inversions() == 11
    # 断言 q 的符号为 -1
    assert q.signature() == -1
    # 断言 p 和其逆元素的逆序数为 0
    assert rmul(p, ~p).inversions() == 0
    # 断言 p 和其逆元素的符号为 1
    assert rmul(p, ~p).signature() == 1
    
    # 断言 p 的阶为 6
    assert p.order() == 6
    # 断言 q 的阶为 10
    assert q.order() == 10
    # 断言 p 的 6 次幂是否是单位置换
    assert (p**(p.order())).is_Identity
    
    # 断言 p 的长度为 6
    assert p.length() == 6
    # 断言 q 的长度为 7
    assert q.length() == 7
    # 断言 r 的长度为 4
    assert r.length() == 4
    
    # 断言 p 的升序分段为 [[1, 5], [2], [0, 3, 6], [4]]
    assert p.runs() == [[1, 5], [2], [0, 3, 6], [4]]
    # 断言 q 的升序分段为 [[4], [2, 3, 5], [0, 6], [1]]
    assert q.runs() == [[4], [2, 3, 5], [0, 6], [1]]
    # 断言 r 的升序分段为 [[3], [2], [1], [0]]
    assert r.runs() == [[3], [2], [1], [0]]
    
    # 断言 p 的索引为 8
    assert p.index() == 8
    # 断言 q 的索引为 8
    assert q.index() == 8
    # 断言 r 的索引为 3
    assert r.index() == 3
    
    # 断言 p 和 q 的前序距离相等
    assert p.get_precedence_distance(q) == q.get_precedence_distance(p)
    # 断言 p 和 q 的邻接距离相等
    assert p.get_adjacency_distance(q) == p.get_adjacency_distance(q)
    # 断言 p 和 q 的位置距离相等
    assert p.get_positional_distance(q) == p.get_positional_distance(q)
    
    # 使用新的排列 p 和 q 进行测试
    p = Permutation([0, 1, 2, 3])
    q = Permutation([3, 2, 1, 0])
    # 断言 p 和 q 的前序距离为 6
    assert p.get_precedence_distance(q) == 6
    # 断言 p 和 q 的邻接距离为 3
    assert p.get_adjacency_distance(q) == 3
    # 断言 p 和 q 的位置距离为 8
    assert p.get_positional_distance(q) == 8
    
    # 使用新的排列 p 和 q 进行测试
    p = Permutation([0, 3, 1, 2, 4])
    q = Permutation.josephus(4, 5, 2)
    # 断言 p 和 q 的邻接距离为 3
    assert p.get_adjacency_distance(q) == 3
    # 断言对一个空排列获取邻接距离会引发 ValueError 异常
    raises(ValueError, lambda: p.get_adjacency_distance(Permutation([])))
    # 断言对一个空排列获取位置距离会引发 ValueError 异常
    raises(ValueError, lambda: p.get_positional_distance(Permutation([])))
    # 断言对一个空排列获取前序距离会引发 ValueError 异常
    raises(ValueError, lambda: p.get_precedence_distance(Permutation([])))
    
    # 创建一组非字典序排列
    a = [Permutation.unrank_nonlex(4, i) for i in range(5)]
    # 创建单位置换
    iden = Permutation([0, 1, 2, 3])
    # 对于所有的排列对 (i, j)，断言它们的交换关系是否等价于它们的乘积交换
    for i in range(5):
        for j in range(i + 1, 5):
            assert a[i].commutes_with(a[j]) == (rmul(a[i], a[j]) == rmul(a[j], a[i]))
            if a[i].commutes_with(a[j]):
                # 如果两排列可以交换，则它们的对换子为单位置换
                assert a[i].commutator(a[j]) == iden
                assert a[j].commutator(a[i]) == iden
    
    # 创建排列 a 和 b
    a = Permutation(3)
    b = Permutation(0, 6, 3)(1, 2)
    # 断言排列 a 的循环结构为 {1: 4}
    assert a.cycle_structure == {1: 4}
    # 断言排列 b 的循环结构为 {2: 1, 3: 1, 1: 2}
    assert b.cycle_structure == {2: 1, 3: 1, 1: 2}
    # 对于问题 11130，断言创建 3 大小排列会引发 ValueError 异常
    raises(ValueError, lambda: Permutation(3, size=3))
    # 对于问题 11130，断言创建包含超出大小的排列会引发 ValueError 异常
    raises(ValueError, lambda: Permutation([1, 2, 0, 3], size=3))
def test_Permutation_subclassing():
    # 定义一个自定义的排列子类，增加对可迭代对象的排列应用
    class CustomPermutation(Permutation):
        # 重载调用操作符，支持不同类型的参数传递
        def __call__(self, *i):
            try:
                # 尝试调用父类的同名方法
                return super().__call__(*i)
            except TypeError:
                pass

            try:
                # 如果传入参数是一个可迭代对象，返回根据当前排列重新排列后的结果
                perm_obj = i[0]
                return [self._array_form[j] for j in perm_obj]
            except TypeError:
                raise TypeError('unrecognized argument')

        # 重载相等操作符，比较两个排列是否相同
        def __eq__(self, other):
            if isinstance(other, Permutation):
                return self._hashable_content() == other._hashable_content()
            else:
                return super().__eq__(other)

        # 重载哈希操作符
        def __hash__(self):
            return super().__hash__()

    # 创建一个自定义排列对象 p 和一个普通排列对象 q
    p = CustomPermutation([1, 2, 3, 0])
    q = Permutation([1, 2, 3, 0])

    # 断言自定义排列对象和普通排列对象相等
    assert p == q
    # 断言对普通排列对象 q 执行带有非法参数的调用会抛出 TypeError 异常
    raises(TypeError, lambda: q([1, 2]))
    # 断言对自定义排列对象 p 执行特定排列后的结果
    assert [2, 3] == p([1, 2])

    # 断言 p 与 q 相乘后得到的类型是 CustomPermutation
    assert type(p * q) == CustomPermutation
    # 断言 q 与 p 相乘后得到的类型是 Permutation，这是因为 q.__mul__(p) 被调用了！
    assert type(q * p) == Permutation

    # 运行所有 Permutation 类的测试用例，同时也在子类上运行
    def wrapped_test_Permutation():
        # 在全局命名空间中用自定义排列类替换原始的 Permutation 类
        globals()['__Perm'] = globals()['Permutation']
        globals()['Permutation'] = CustomPermutation
        # 运行测试用例
        test_Permutation()
        # 恢复原始的 Permutation 类定义
        globals()['Permutation'] = globals()['__Perm']
        del globals()['__Perm']

    wrapped_test_Permutation()


def test_josephus():
    # 断言 Josephus 排列算法的运行结果
    assert Permutation.josephus(4, 6, 1) == Permutation([3, 1, 0, 2, 5, 4])
    # 断言 Josephus 排列的身份测试
    assert Permutation.josephus(1, 5, 1).is_Identity


def test_ranking():
    # 断言基于 Lexicographic 排名的排列对象
    assert Permutation.unrank_lex(5, 10).rank() == 10
    # 创建一个 Lexicographic 排列对象 p
    p = Permutation.unrank_lex(15, 225)
    # 断言 p 的排名为 225
    assert p.rank() == 225
    # 获取 p 的下一个 Lexicographic 排列对象 p1，并断言其排名为 226
    p1 = p.next_lex()
    assert p1.rank() == 226
    # 断言 Lexicographic 排名为 225 的排列对象与其自身相等，并且是身份排列
    assert Permutation.unrank_lex(15, 225).rank() == 225
    assert Permutation.unrank_lex(10, 0).is_Identity
    # 创建一个 Lexicographic 排列对象 p，其排列形式为 [3, 2, 1, 0]
    p = Permutation.unrank_lex(4, 23)
    # 断言 p 的排名为 23
    assert p.rank() == 23
    # 断言 p 的排列形式为 [3, 2, 1, 0]
    assert p.array_form == [3, 2, 1, 0]
    # 断言 p 的下一个 Lexicographic 排列对象为 None
    assert p.next_lex() is None

    # 创建一个排列对象 p 和一个嵌套的排列对象 q
    p = Permutation([1, 5, 2, 0, 3, 6, 4])
    q = Permutation([[1, 2, 3, 5, 6], [0, 4]])
    # 创建 Trotter-Johnson 排列的排名列表 a
    a = [Permutation.unrank_trotterjohnson(4, i).array_form for i in range(5)]
    # 断言 a 的结果与预期结果匹配
    assert a == [[0, 1, 2, 3], [0, 1, 3, 2], [0, 3, 1, 2], [3, 0, 1, 2], [3, 0, 2, 1]]
    # 断言 Trotter-Johnson 排名的结果与预期结果匹配
    assert [Permutation(pa).rank_trotterjohnson() for pa in a] == list(range(5))
    # 断言排列对象 q 的 Trotter-Johnson 排名为 2283
    assert q.rank_trotterjohnson() == 2283
    # 断言排列对象 p 的 Trotter-Johnson 排名为 3389
    assert p.rank_trotterjohnson() == 3389
    # 断言排列对象 [1, 0] 的 Trotter-Johnson 排名为 1
    assert Permutation([1, 0]).rank_trotterjohnson() == 1
    # 创建一个排列对象 a，并创建 b 作为其引用
    a = Permutation(list(range(3)))
    b = a
    l = []
    tj = []
    # 迭代生成 6 个排列对象，并将它们添加到列表 l 和 tj 中
    for i in range(6):
        l.append(a)
        tj.append(b)
        # 获取当前排列对象的下一个 Lexicographic 排列对象和 Trotter-Johnson 排列对象
        a = a.next_lex()
        b = b.next_trotterjohnson()
    # 断言当前的 Lexicographic 和 Trotter-Johnson 排列结果一致
    assert a == b is None
    assert {tuple(a) for a in l} == {tuple(a) for a in tj}

    # 创建一个排列对象 p
    p = Permutation([2, 5, 1, 6, 3, 0, 4])
    # 创建一个置换对象 `p`，表示置换 [[6], [5], [0, 1, 2, 3, 4]]
    q = Permutation([[6], [5], [0, 1, 2, 3, 4]])

    # 断言 `p` 的秩为 1964
    assert p.rank() == 1964

    # 断言 `q` 的秩为 870
    assert q.rank() == 870

    # 使用空置换的非词典序秩函数，期望结果为 0
    assert Permutation([]).rank_nonlex() == 0

    # 计算置换 `p` 的非词典序秩
    prank = p.rank_nonlex()

    # 断言 `prank` 的值为 1600
    assert prank == 1600

    # 使用非词典序秩还原置换，期望得到置换 `p`
    assert Permutation.unrank_nonlex(7, 1600) == p

    # 计算置换 `q` 的非词典序秩
    qrank = q.rank_nonlex()

    # 断言 `qrank` 的值为 41
    assert qrank == 41

    # 使用非词典序秩还原置换，期望得到与 `q` 数组形式相同的置换
    assert Permutation.unrank_nonlex(7, 41) == Permutation(q.array_form)

    # 使用非词典序秩生成所有长度为 4 的置换的数组形式列表 `a`
    a = [Permutation.unrank_nonlex(4, i).array_form for i in range(24)]

    # 断言 `a` 的结果与指定的数组形式列表相等
    assert a == [
        [1, 2, 3, 0], [3, 2, 0, 1], [1, 3, 0, 2], [1, 2, 0, 3], [2, 3, 1, 0],
        [2, 0, 3, 1], [3, 0, 1, 2], [2, 0, 1, 3], [1, 3, 2, 0], [3, 0, 2, 1],
        [1, 0, 3, 2], [1, 0, 2, 3], [2, 1, 3, 0], [2, 3, 0, 1], [3, 1, 0, 2],
        [2, 1, 0, 3], [3, 2, 1, 0], [0, 2, 3, 1], [0, 3, 1, 2], [0, 2, 1, 3],
        [3, 1, 2, 0], [0, 3, 2, 1], [0, 1, 3, 2], [0, 1, 2, 3]]

    # 定义一个常数 `N`，并创建置换 `p1`，作为数组形式列表 `a` 中第一个元素的置换
    N = 10
    p1 = Permutation(a[0])

    # 将数组形式列表 `a` 中的置换逐个与 `p1` 进行右乘操作
    for i in range(1, N+1):
        p1 = p1 * Permutation(a[i])

    # 使用静态方法 `rmul_with_af` 将数组形式列表 `a` 中的置换从右到左进行右乘操作，得到置换 `p2`
    p2 = Permutation.rmul_with_af(*[Permutation(h) for h in a[N::-1]])

    # 断言 `p1` 和 `p2` 相等
    assert p1 == p2

    # 定义空列表 `ok`，创建一个置换 `p`，其数组形式为 `[1, 0]`
    ok = []
    p = Permutation([1, 0])

    # 迭代三次，获取 `p` 的数组形式并添加到 `ok` 列表中，然后计算 `p` 的非词典序下一个置换
    for i in range(3):
        ok.append(p.array_form)
        p = p.next_nonlex()
        if p is None:
            ok.append(None)
            break

    # 断言 `ok` 的值为预期的列表 `[[1, 0], [0, 1], None]`
    assert ok == [[1, 0], [0, 1], None]

    # 断言置换 `[3, 2, 0, 1]` 的非词典序下一个置换是 `[1, 3, 0, 2]`
    assert Permutation([3, 2, 0, 1]).next_nonlex() == Permutation([1, 3, 0, 2])

    # 断言数组形式列表 `a` 中每个置换的非词典序秩与其索引相等
    assert [Permutation(pa).rank_nonlex() for pa in a] == list(range(24))
# 定义一个测试函数 test_mul，用于测试排列操作的乘法方法
def test_mul():
    # 初始化排列 a 和 b
    a, b = [0, 2, 1, 3], [0, 1, 3, 2]
    # 断言使用 _af_rmul 函数对 a 和 b 进行右乘的结果与预期的数组形式相等
    assert _af_rmul(a, b) == [0, 2, 3, 1]
    # 断言使用 _af_rmuln 函数对 a, b 和从 0 到 3 的列表进行右乘的结果与预期的数组形式相等
    assert _af_rmuln(a, b, list(range(4))) == [0, 2, 3, 1]
    # 断言对 Permutation(a) 和 Permutation(b) 进行右乘后的结果的数组形式与预期结果相等
    assert rmul(Permutation(a), Permutation(b)).array_form == [0, 2, 3, 1]

    # 将 a 初始化为 Permutation([0, 2, 1, 3])
    a = Permutation([0, 2, 1, 3])
    # 初始化 b 和 c
    b = (0, 1, 3, 2)
    c = (3, 1, 2, 0)
    # 断言对 Permutation.rmul 方法使用 a, b 和 c 进行右乘的结果与预期的排列形式相等
    assert Permutation.rmul(a, b, c) == Permutation([1, 2, 3, 0])
    # 断言对 Permutation.rmul 方法使用 a 和 c 进行右乘的结果与预期的排列形式相等
    assert Permutation.rmul(a, c) == Permutation([3, 2, 1, 0])
    # 断言对 Permutation.rmul 方法使用非 Permutation 对象会抛出 TypeError 异常
    raises(TypeError, lambda: Permutation.rmul(b, c))

    # 初始化 n 和 m
    n = 6
    m = 8
    # 使用 unrank_nonlex 方法生成长度为 m 的排列列表 a
    a = [Permutation.unrank_nonlex(n, i).array_form for i in range(m)]
    # 初始化 h 为 0 到 n-1 的列表
    h = list(range(n))
    # 遍历 a 中的排列，依次对 h 进行右乘，并断言 h 和 _af_rmuln(*a[:i+1]) 的结果相等
    for i in range(m):
        h = _af_rmul(h, a[i])
        h2 = _af_rmuln(*a[:i + 1])
        assert h == h2


# 定义一个测试函数 test_args，用于测试 Permutation 类的各种初始化方式和方法
def test_args():
    # 初始化 p 为 Permutation([(0, 3, 1, 2), (4, 5)])
    p = Permutation([(0, 3, 1, 2), (4, 5)])
    # 断言 p 的 _cyclic_form 属性为 None
    assert p._cyclic_form is None
    # 断言 Permutation(p) 等于 p
    assert Permutation(p) == p
    # 断言 p 的 cyclic_form 属性为 [[0, 3, 1, 2], [4, 5]]
    assert p.cyclic_form == [[0, 3, 1, 2], [4, 5]]
    # 断言 p 的 _array_form 属性为 [3, 2, 0, 1, 5, 4]
    assert p._array_form == [3, 2, 0, 1, 5, 4]
    # 初始化 p 为 Permutation((0, 3, 1, 2))
    p = Permutation((0, 3, 1, 2))
    # 断言 p 的 _cyclic_form 属性为 None
    assert p._cyclic_form is None
    # 断言 p 的 _array_form 属性为 [0, 3, 1, 2]
    assert p._array_form == [0, 3, 1, 2]
    # 断言 Permutation([0]) 等于 Permutation((0,))
    assert Permutation([0]) == Permutation((0,))
    # 断言 Permutation([[0], [1]]) 等于多种不同方式构造的 Permutation 对象
    assert Permutation([[0], [1]]) == Permutation(((0,), (1,))) == \
        Permutation(((0,), [1]))
    # 断言 Permutation([[1, 2]]) 等于 Permutation([0, 2, 1])
    assert Permutation([[1, 2]]) == Permutation([0, 2, 1])
    # 断言 Permutation([[1], [4, 2]]) 等于 Permutation([0, 1, 4, 3, 2])
    assert Permutation([[1], [4, 2]]) == Permutation([0, 1, 4, 3, 2])
    # 断言 Permutation([[1], [4, 2]], size=1) 等于 Permutation([0, 1, 4, 3, 2])
    assert Permutation([[1], [4, 2]], size=1) == Permutation([0, 1, 4, 3, 2])
    # 断言 Permutation([[1], [4, 2]], size=6) 等于 Permutation([0, 1, 4, 3, 2, 5])
    assert Permutation([[1], [4, 2]], size=6) == Permutation([0, 1, 4, 3, 2, 5])
    # 断言 Permutation([[0, 1], [0, 2]]) 等于 Permutation(0, 1, 2)
    assert Permutation([[0, 1], [0, 2]]) == Permutation(0, 1, 2)
    # 断言 Permutation([], size=3) 等于 Permutation([0, 1, 2])
    assert Permutation([], size=3) == Permutation([0, 1, 2])
    # 断言 Permutation(3).list(5) 的结果为 [0, 1, 2, 3, 4]
    assert Permutation(3).list(5) == [0, 1, 2, 3, 4]
    # 断言 Permutation(3).list(-1) 的结果为空列表
    assert Permutation(3).list(-1) == []
    # 断言 Permutation(5)(1, 2).list(-1) 的结果为 [0, 2, 1]
    assert Permutation(5)(1, 2).list(-1) == [0, 2, 1]
    # 断言 Permutation(5)(1, 2).list() 的结果为 [0, 2, 1, 3, 4, 5]
    assert Permutation(5)(1, 2).list() == [0, 2, 1, 3, 4, 5]
    # 断言对于 Permutation([1, 2], [0]) 构造会抛出 ValueError 异常
    raises(ValueError, lambda: Permutation([1, 2], [0]))
           # 需要封闭括号
    # 断言对于 Permutation([[1, 2], 0]) 构造会抛出 ValueError 异常
    raises(ValueError, lambda: Permutation([[1, 2], 0]))
           # 0 需要封闭括号
    # 断言对于 Permutation([1, 1, 0]) 构造会抛出 ValueError 异常
    raises(ValueError, lambda: Permutation([1, 1, 0]))
    # 断言对于 Permutation([4, 5], size=10) 构造会抛出 ValueError 异常
    raises(ValueError, lambda: Permutation([4, 5], size=10))  # 0 到 3 缺失？
    # 断言 Permutation(4, 5) 的结果等于 Permutation([0, 1, 2, 3, 5, 4])
    assert Permutation(4, 5) == Permutation([0, 1, 2, 3, 5, 4])


# 定义一个测试函数 test_Cycle，用于测试 Cycle 类的各种功能
def test_Cycle():
    # 断言空 Cycle 对象的字符串表示为
    # 验证 Cycle 类的功能
    assert str(Cycle(1, 2)(4, 5)) == '(1 2)(4 5)'
    # 检查将两个循环合并后的字符串表示是否正确
    
    assert str(Cycle(1, 2)) == '(1 2)'
    # 检查单个循环的字符串表示是否正确
    
    assert Cycle(Permutation(list(range(3)))) == Cycle()
    # 检查将包含所有元素的置换转换为循环后是否得到空循环
    
    assert Cycle(1, 2).list() == [0, 2, 1]
    # 检查循环对象的列表表示是否正确，列表长度为默认值 3
    
    assert Cycle(1, 2).list(4) == [0, 2, 1, 3]
    # 检查循环对象的列表表示是否正确，列表长度为 4
    
    assert Cycle().size == 0
    # 检查空循环对象的大小是否为 0
    
    raises(ValueError, lambda: Cycle((1, 2)))
    # 检查传递元组作为参数时是否会引发 ValueError 异常
    
    raises(ValueError, lambda: Cycle(1, 2, 1))
    # 检查包含重复元素的参数列表是否会引发 ValueError 异常
    
    raises(TypeError, lambda: Cycle(1, 2)*{})
    # 检查尝试用字典乘以循环对象是否会引发 TypeError 异常
    
    raises(ValueError, lambda: Cycle(4)[a])
    # 检查索引不存在的元素是否会引发 ValueError 异常
    
    raises(ValueError, lambda: Cycle(2, -4, 3))
    # 检查包含负数的参数列表是否会引发 ValueError 异常
    
    # 检查循环对象转换为置换对象再转换回循环对象后是否与原对象相同
    p = Permutation([[1, 2], [4, 3]], size=5)
    assert Permutation(Cycle(p)) == p
def test_from_sequence():
    # 测试从序列创建置换对象，使用默认大小
    assert Permutation.from_sequence('SymPy') == Permutation(4)(0, 1, 3)
    # 测试从序列创建置换对象，自定义键函数（忽略大小写）
    assert Permutation.from_sequence('SymPy', key=lambda x: x.lower()) == \
        Permutation(4)(0, 2)(1, 3)


def test_resize():
    p = Permutation(0, 1, 2)
    # 测试调整置换对象大小为5
    assert p.resize(5) == Permutation(0, 1, 2, size=5)
    # 测试调整置换对象大小为4
    assert p.resize(4) == Permutation(0, 1, 2, size=4)
    # 测试调整置换对象大小为3（不变）
    assert p.resize(3) == p
    # 测试调整置换对象到小于2（预期抛出异常）
    raises(ValueError, lambda: p.resize(2))

    p = Permutation(0, 1, 2)(3, 4)(5, 6)
    # 测试调整带有循环的置换对象大小为3
    assert p.resize(3) == Permutation(0, 1, 2)
    # 测试调整带有循环的置换对象大小为4（预期抛出异常）
    raises(ValueError, lambda: p.resize(4))


def test_printing_cyclic():
    p1 = Permutation([0, 2, 1])
    # 测试循环置换对象的字符串表示（短格式）
    assert repr(p1) == 'Permutation(1, 2)'
    # 测试循环置换对象的字符串表示（长格式）
    assert str(p1) == '(1 2)'
    p2 = Permutation()
    # 测试空置换对象的字符串表示（短格式）
    assert repr(p2) == 'Permutation()'
    # 测试空置换对象的字符串表示（长格式）
    assert str(p2) == '()'
    p3 = Permutation([1, 2, 0, 3])
    # 测试非循环置换对象的字符串表示（长格式）
    assert repr(p3) == 'Permutation(3)(0, 1, 2)'


def test_printing_non_cyclic():
    p1 = Permutation([0, 1, 2, 3, 4, 5])
    # 测试非循环置换对象的字符串表示（无循环）
    assert srepr(p1, perm_cyclic=False) == 'Permutation([], size=6)'
    assert sstr(p1, perm_cyclic=False) == 'Permutation([], size=6)'
    p2 = Permutation([0, 1, 2])
    # 测试非循环置换对象的字符串表示（包含元素）
    assert srepr(p2, perm_cyclic=False) == 'Permutation([0, 1, 2])'
    assert sstr(p2, perm_cyclic=False) == 'Permutation([0, 1, 2])'

    p3 = Permutation([0, 2, 1])
    # 测试非循环置换对象的字符串表示（包含元素，顺序不同）
    assert srepr(p3, perm_cyclic=False) == 'Permutation([0, 2, 1])'
    assert sstr(p3, perm_cyclic=False) == 'Permutation([0, 2, 1])'
    p4 = Permutation([0, 1, 3, 2, 4, 5, 6, 7])
    # 测试非循环置换对象的字符串表示（包含元素和大小）
    assert srepr(p4, perm_cyclic=False) == 'Permutation([0, 1, 3, 2], size=8)'


def test_deprecated_print_cyclic():
    p = Permutation(0, 1, 2)
    try:
        # 测试带有循环的置换对象的字符串表示（警告：使用废弃函数）
        Permutation.print_cyclic = True
        with warns_deprecated_sympy():
            assert sstr(p) == '(0 1 2)'
        with warns_deprecated_sympy():
            assert srepr(p) == 'Permutation(0, 1, 2)'
        with warns_deprecated_sympy():
            assert pretty(p) == '(0 1 2)'
        with warns_deprecated_sympy():
            assert latex(p) == r'\left( 0\; 1\; 2\right)'

        Permutation.print_cyclic = False
        # 测试不带循环的置换对象的字符串表示（警告：使用废弃函数）
        with warns_deprecated_sympy():
            assert sstr(p) == 'Permutation([1, 2, 0])'
        with warns_deprecated_sympy():
            assert srepr(p) == 'Permutation([1, 2, 0])'
        with warns_deprecated_sympy():
            assert pretty(p, use_unicode=False) == '/0 1 2\\\n\\1 2 0/'
        with warns_deprecated_sympy():
            assert latex(p) == \
                r'\begin{pmatrix} 0 & 1 & 2 \\ 1 & 2 & 0 \end{pmatrix}'
    finally:
        Permutation.print_cyclic = None


def test_permutation_equality():
    a = Permutation(0, 1, 2)
    b = Permutation(0, 1, 2)
    # 测试置换对象相等性
    assert Eq(a, b) is S.true
    c = Permutation(0, 2, 1)
    assert Eq(a, c) is S.false

    d = Permutation(0, 1, 2, size=4)
    # 测试带有指定大小的置换对象相等性
    assert unchanged(Eq, a, d)
    e = Permutation(0, 2, 1, size=4)
    assert unchanged(Eq, a, e)

    i = Permutation()
    # 测试空置换对象相等性
    assert unchanged(Eq, i, 0)
    assert unchanged(Eq, 0, i)


def test_issue_17661():
    # 测试修复问题 #17661 的特定情况
    # 创建 Cycle 类的实例 c1，参数为 1 和 2
    c1 = Cycle(1, 2)
    # 创建 Cycle 类的实例 c2，参数为 1 和 2
    c2 = Cycle(1, 2)
    # 使用断言检查 c1 和 c2 是否相等
    assert c1 == c2
    # 使用断言检查 repr(c1) 是否等于 'Cycle(1, 2)'
    assert repr(c1) == 'Cycle(1, 2)'
    # 使用断言再次检查 c1 和 c2 是否相等
    assert c1 == c2
# 定义测试函数 test_permutation_apply
def test_permutation_apply():
    # 创建符号变量 x
    x = Symbol('x')
    # 创建置换对象 p，表示置换 (0, 1, 2)
    p = Permutation(0, 1, 2)
    # 断言：应用置换 p 到 0，结果应为 1
    assert p.apply(0) == 1
    # 断言：应用置换 p 到 0，结果应为 Integer 类型
    assert isinstance(p.apply(0), Integer)
    # 断言：应用置换 p 到符号变量 x，结果应为 AppliedPermutation(p, x)
    assert p.apply(x) == AppliedPermutation(p, x)
    # 断言：对应用置换 p 到符号变量 x 的结果，再将 x 替换为 0，结果应为 1
    assert AppliedPermutation(p, x).subs(x, 0) == 1

    # 重新定义符号变量 x，指定为非整数
    x = Symbol('x', integer=False)
    # 断言：应用置换 p 到符号变量 x，抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: p.apply(x))
    # 重新定义符号变量 x，指定为负数
    x = Symbol('x', negative=True)
    # 断言：应用置换 p 到符号变量 x，抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: p.apply(x))


# 定义测试函数 test_AppliedPermutation
def test_AppliedPermutation():
    # 创建符号变量 x
    x = Symbol('x')
    # 创建置换对象 p，表示置换 (0, 1, 2)
    p = Permutation(0, 1, 2)
    # 断言：尝试创建 AppliedPermutation 对象时，传入的置换不是 Permutation 类型，抛出 ValueError 异常
    raises(ValueError, lambda: AppliedPermutation((0, 1, 2), x))
    # 断言：创建 AppliedPermutation 对象，对置换 p 应用到 1，evaluate=True 表示立即求值，结果应为 2
    assert AppliedPermutation(p, 1, evaluate=True) == 2
    # 断言：创建 AppliedPermutation 对象，对置换 p 应用到 1，evaluate=False 表示不立即求值，检查其类是否为 AppliedPermutation
    assert AppliedPermutation(p, 1, evaluate=False).__class__ == AppliedPermutation
```