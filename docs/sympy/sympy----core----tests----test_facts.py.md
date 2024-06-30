# `D:\src\scipysrc\sympy\sympy\core\tests\test_facts.py`

```
# 导入所需模块和函数
from sympy.core.facts import (deduce_alpha_implications,
        apply_beta_to_alpha_route, rules_2prereq, FactRules, FactKB)
from sympy.core.logic import And, Not  # 导入逻辑运算相关的模块和函数
from sympy.testing.pytest import raises  # 导入用于测试的 raises 函数

T = True  # 定义常量 True
F = False  # 定义常量 False
U = None  # 定义常量 None


def test_deduce_alpha_implications():
    def D(i):
        # 对给定的 alpha-implications 进行推导
        I = deduce_alpha_implications(i)
        # 将推导结果转换为先决条件
        P = rules_2prereq({
            (k, True): {(v, True) for v in S} for k, S in I.items()})
        return I, P

    # 测试传递性
    I, P = D([('a', 'b'), ('b', 'c')])
    assert I == {'a': {'b', 'c'}, 'b': {'c'}, Not('b'):
        {Not('a')}, Not('c'): {Not('a'), Not('b')}}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}

    # 测试重复条目处理
    I, P = D([('a', 'b'), ('b', 'c'), ('b', 'c')])
    assert I == {'a': {'b', 'c'}, 'b': {'c'}, Not('b'): {Not('a')}, Not('c'): {Not('a'), Not('b')}}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}

    # 测试循环容忍性
    assert D([('a', 'a'), ('a', 'a')]) == ({}, {})
    assert D([('a', 'b'), ('b', 'a')]) == (
        {'a': {'b'}, 'b': {'a'}, Not('a'): {Not('b')}, Not('b'): {Not('a')}})

    # 测试不一致性检测
    raises(ValueError, lambda: D([('a', Not('a'))]))
    raises(ValueError, lambda: D([('a', 'b'), ('b', Not('a'))]))
    raises(ValueError, lambda: D([('a', 'b'), ('b', 'c'), ('b', 'na'),
           ('na', Not('a'))]))

    # 测试带否定的推导
    I, P = D([('a', Not('b')), ('c', 'b')])
    assert I == {'a': {Not('b'), Not('c')}, 'b': {Not('a')}, 'c': {'b', Not('a')}, Not('b'): {Not('c')}}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}
    I, P = D([(Not('a'), 'b'), ('a', 'c')])
    assert I == {'a': {'c'}, Not('a'): {'b'}, Not('b'): {'a',
    'c'}, Not('c'): {Not('a'), 'b'},}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}


    # 长推导链测试
    I, P = D([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])
    assert I == {'a': {'b', 'c', 'd', 'e'}, 'b': {'c', 'd', 'e'},
        'c': {'d', 'e'}, 'd': {'e'}, Not('b'): {Not('a')},
        Not('c'): {Not('a'), Not('b')}, Not('d'): {Not('a'), Not('b'),
            Not('c')}, Not('e'): {Not('a'), Not('b'), Not('c'), Not('d')}}
    assert P == {'a': {'b', 'c', 'd', 'e'}, 'b': {'a', 'c', 'd',
        'e'}, 'c': {'a', 'b', 'd', 'e'}, 'd': {'a', 'b', 'c', 'e'},
        'e': {'a', 'b', 'c', 'd'}}

    # 实际应用相关推导测试
    I, P = D([('rat', 'real'), ('int', 'rat')])

    assert I == {'int': {'rat', 'real'}, 'rat': {'real'},
        Not('real'): {Not('rat'), Not('int')}, Not('rat'): {Not('int')}}
    assert P == {'rat': {'int', 'real'}, 'real': {'int', 'rat'},
        'int': {'rat', 'real'}}


# TODO move me to appropriate place
def test_apply_beta_to_alpha_route():
    APPLY = apply_beta_to_alpha_route

    # 表示带有附加 beta-rule #bidx 的空 alpha-chain
    def Q(bidx):
        return (set(), [bidx])
    # x -> a        &(a,b) -> x     --  x -> a
    A = {'x': {'a'}}
    B = [(And('a', 'b'), 'x')]
    assert APPLY(A, B) == {'x': ({'a'}, []), 'a': Q(0), 'b': Q(0)}

    # x -> a        &(a,!x) -> b    --  x -> a
    A = {'x': {'a'}}
    B = [(And('a', Not('x')), 'b')]
    assert APPLY(A, B) == {'x': ({'a'}, []), Not('x'): Q(0), 'a': Q(0)}

    # x -> a b      &(a,b) -> c     --  x -> a b c
    A = {'x': {'a', 'b'}}
    B = [(And('a', 'b'), 'c')]
    assert APPLY(A, B) == \
        {'x': ({'a', 'b', 'c'}, []), 'a': Q(0), 'b': Q(0)}

    # x -> a        &(a,b) -> y     --  x -> a [#0]
    A = {'x': {'a'}}
    B = [(And('a', 'b'), 'y')]
    assert APPLY(A, B) == {'x': ({'a'}, [0]), 'a': Q(0), 'b': Q(0)}

    # x -> a b c    &(a,b) -> c     --  x -> a b c
    A = {'x': {'a', 'b', 'c'}}
    B = [(And('a', 'b'), 'c')]
    assert APPLY(A, B) == \
        {'x': ({'a', 'b', 'c'}, []), 'a': Q(0), 'b': Q(0)}

    # x -> a b      &(a,b,c) -> y   --  x -> a b [#0]
    A = {'x': {'a', 'b'}}
    B = [(And('a', 'b', 'c'), 'y')]
    assert APPLY(A, B) == \
        {'x': ({'a', 'b'}, [0]), 'a': Q(0), 'b': Q(0), 'c': Q(0)}

    # x -> a b      &(a,b) -> c     --  x -> a b c d
    # c -> d                            c -> d
    A = {'x': {'a', 'b'}, 'c': {'d'}}
    B = [(And('a', 'b'), 'c')]
    assert APPLY(A, B) == {'x': ({'a', 'b', 'c', 'd'}, []),
        'c': ({'d'}, []), 'a': Q(0), 'b': Q(0)}

    # x -> a b      &(a,b) -> c     --  x -> a b c d e
    # c -> d        &(c,d) -> e         c -> d e
    A = {'x': {'a', 'b'}, 'c': {'d'}}
    B = [(And('a', 'b'), 'c'), (And('c', 'd'), 'e')]
    assert APPLY(A, B) == {'x': ({'a', 'b', 'c', 'd', 'e'}, []),
        'c': ({'d', 'e'}, []), 'a': Q(0), 'b': Q(0), 'd': Q(1)}

    # x -> a b      &(a,y) -> z     --  x -> a b y z
    #               &(a,b) -> y
    A = {'x': {'a', 'b'}}
    B = [(And('a', 'y'), 'z'), (And('a', 'b'), 'y')]
    assert APPLY(A, B) == {'x': ({'a', 'b', 'y', 'z'}, []),
        'a': (set(), [0, 1]), 'y': Q(0), 'b': Q(1)}

    # x -> a b      &(a,!b) -> c    --  x -> a b
    A = {'x': {'a', 'b'}}
    B = [(And('a', Not('b')), 'c')]
    assert APPLY(A, B) == \
        {'x': ({'a', 'b'}, []), 'a': Q(0), Not('b'): Q(0)}

    # !x -> !a !b   &(!a,b) -> c    --  !x -> !a !b
    A = {Not('x'): {Not('a'), Not('b')}}
    B = [(And(Not('a'), 'b'), 'c')]
    assert APPLY(A, B) == \
        {Not('x'): ({Not('a'), Not('b')}, []), Not('a'): Q(0), 'b': Q(0)}

    # x -> a b      &(b,c) -> !a    --  x -> a b
    A = {'x': {'a', 'b'}}
    B = [(And('b', 'c'), Not('a'))]
    assert APPLY(A, B) == {'x': ({'a', 'b'}, []), 'b': Q(0), 'c': Q(0)}

    # x -> a b      &(a, b) -> c    --  x -> a b c p
    # c -> p a
    A = {'x': {'a', 'b'}, 'c': {'p', 'a'}}
    B = [(And('a', 'b'), 'c')]
    assert APPLY(A, B) == {'x': ({'a', 'b', 'c', 'p'}, []),
        'c': ({'p', 'a'}, []), 'a': Q(0), 'b': Q(0)}
# 定义一个名为 test_FactRules_parse 的测试函数
def test_FactRules_parse():
    # 创建一个 FactRules 对象，传入字符串 'a -> b'，并赋值给变量 f
    f = FactRules('a -> b')
    # 使用断言验证 f 对象的 prereq 属性是否等于 {'b': {'a'}, 'a': {'b'}}
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}

    # 创建一个新的 FactRules 对象，传入字符串 'a -> !b'，并赋值给变量 f
    f = FactRules('a -> !b')
    # 使用断言验证 f 对象的 prereq 属性是否等于 {'b': {'a'}, 'a': {'b'}}
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}

    # 创建一个新的 FactRules 对象，传入字符串 '!a -> b'，并赋值给变量 f
    f = FactRules('!a -> b')
    # 使用断言验证 f 对象的 prereq 属性是否等于 {'b': {'a'}, 'a': {'b'}}
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}

    # 创建一个新的 FactRules 对象，传入字符串 '!a -> !b'，并赋值给变量 f
    f = FactRules('!a -> !b')
    # 使用断言验证 f 对象的 prereq 属性是否等于 {'b': {'a'}, 'a': {'b'}}
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}

    # 创建一个新的 FactRules 对象，传入字符串 '!z == nz'，并赋值给变量 f
    f = FactRules('!z == nz')
    # 使用断言验证 f 对象的 prereq 属性是否等于 {'z': {'nz'}, 'nz': {'z'}}
    assert f.prereq == {'z': {'nz'}, 'nz': {'z'}}


# 定义一个名为 test_FactRules_parse2 的测试函数
def test_FactRules_parse2():
    # 使用 raises 函数检查 ValueError 异常是否被抛出，传入 lambda 表达式和参数 'a -> !a'
    raises(ValueError, lambda: FactRules('a -> !a'))


# 定义一个名为 test_FactRules_deduce 的测试函数
def test_FactRules_deduce():
    # 创建一个 FactRules 对象，传入包含多个规则的列表，赋值给变量 f
    f = FactRules(['a -> b', 'b -> c', 'b -> d', 'c -> e'])

    # 定义一个名为 D 的函数，接受 facts 参数
    def D(facts):
        # 使用 FactKB 类创建 kb 对象，传入 f 对象
        kb = FactKB(f)
        # 调用 kb 对象的 deduce_all_facts 方法，传入 facts 参数
        kb.deduce_all_facts(facts)
        # 返回 kb 对象
        return kb

    # 使用断言验证 D 函数对 {'a': T} 的返回结果是否等于 {'a': T, 'b': T, 'c': T, 'd': T, 'e': T}
    assert D({'a': T}) == {'a': T, 'b': T, 'c': T, 'd': T, 'e': T}
    # 使用断言验证 D 函数对 {'b': T} 的返回结果是否等于 {'b': T, 'c': T, 'd': T, 'e': T}
    assert D({'b': T}) == {        'b': T, 'c': T, 'd': T, 'e': T}
    # 使用断言验证 D 函数对 {'c': T} 的返回结果是否等于 {'c': T, 'e': T}
    assert D({'c': T}) == {                'c': T,         'e': T}
    # 使用断言验证 D 函数对 {'d': T} 的返回结果是否等于 {'d': T}
    assert D({'d': T}) == {                        'd': T        }
    # 使用断言验证 D 函数对 {'e': T} 的返回结果是否等于 {'e': T}
    assert D({'e': T}) == {                                'e': T}

    # 使用断言验证 D 函数对 {'a': F} 的返回结果是否等于 {'a': F}
    assert D({'a': F}) == {'a': F                                }
    # 使用断言验证 D 函数对 {'b': F} 的返回结果是否等于 {'a': F, 'b': F}
    assert D({'b': F}) == {'a': F, 'b': F                        }
    # 使用断言验证 D 函数对 {'c': F} 的返回结果是否等于 {'a': F, 'b': F, 'c': F}
    assert D({'c': F}) == {'a': F, 'b': F, 'c': F                }
    # 使用断言验证 D 函数对 {'d': F} 的返回结果是否等于 {'a': F, 'b': F, 'd': F}
    assert D({'d': F}) == {'a': F, 'b': F,         'd': F        }

    # 使用断言验证 D 函数对 {'a': U} 的返回结果是否等于 {'a': U}，并附带注释 '# XXX ok?'
    assert D({'a': U}) == {'a': U}  # XXX ok?


# 定义一个名为 test_FactRules_deduce2 的测试函数
def test_FactRules_deduce2():
    # 创建一个 FactRules 对象，传入包含多个规则的列表，赋值给变量 f
    # pos/neg/zero, but the rules are not sufficient to derive all relations
    f = FactRules(['pos -> !neg', 'pos -> !z'])

    # 定义一个名为 D 的函数，接受 facts 参数
    def D(facts):
        # 使用 FactKB 类创建 kb 对象，传入 f 对象
        kb = FactKB(f)
        # 调用 kb 对象的 deduce_all_facts 方法，传入 facts 参数
        kb.deduce_all_facts(facts)
        # 返回 kb 对象
        return kb

    # 使用断言验证 D 函数对 {'pos': T} 的返回结果是否等于 {'pos': T, 'neg': F, 'z': F}
    assert D({'pos': T}) == {'pos': T, 'neg': F, 'z': F}
    # 使用断言验证 D 函数对 {'pos': F} 的返回结果是否等于 {'pos': F}
    assert D({'pos': F}) == {'pos': F                  }
    # 使用断言验证 D 函数对 {'neg': T} 的返回结果是否等于 {'pos': F, 'neg': T}
    assert D({'neg': T}) == {'pos': F, 'neg': T        }
    # 使用断言验证 D 函数对 {'neg': F} 的返回结果是否等于 {'neg': F}
    assert D({'neg': F}) == {          'neg': F        }
    # 使用断言验证 D 函数对 {'z': T} 的返回结果是否等于 {'pos': F, 'z': T}
    assert D({'z': T}) == {'pos': F,           'z': T}
    # 使用断言验证 D 函数对 {'z': F} 的返回结果是否等于 {'z': F}
    assert D({'z': F}) == {                    'z': F}

    # pos/neg/zero. rules are sufficient to derive all relations
    # 创建一个新的 FactRules 对象，传入包含多个规则的列表，赋值给变量 f
    f = FactRules(['pos -> !neg', 'neg -> !pos', 'pos -> !z', 'neg -> !z'])

    # 使用断言验证 D 函数对 {'pos': T} 的返回结果是否等于 {'pos': T, 'neg': F, 'z': F}
    assert D({'pos': T}) == {'pos': T, 'neg': F, 'z': F}
    # 使用断言验证 D 函数对 {'pos': F} 的返回结果是否等于 {'pos': F}
    assert D({'pos': F}) == {'pos': F                  }
    # 使用断言验证 D 函数对 {'neg': T} 的返回结果是否等于 {'pos': F, 'neg': T, 'z': F}
    assert D({'neg': T}) == {'pos': F, 'neg': T, 'z': F}
    # 使用断言验证 D 函数对 {'neg': F} 的返回结果是否等于 {'neg': F}
    assert D({'neg': F}) == {          'neg': F        }
    # 使用断言验证 D 函数对 {'z': T} 的返回
    # 调用函数 D 并断言其返回结果与预期字典 {'real': T, 'pos': F, 'npos': T} 相等
    assert D({'real': T, 'pos': F}) == {'real': T, 'pos': F, 'npos': T}
    
    # 调用函数 D 并断言其返回结果与预期字典 {'real': T, 'pos': T, 'npos': F} 相等
    assert D({'real': T, 'npos': F}) == {'real': T, 'pos': T, 'npos': F}
    
    # 调用函数 D 并断言其返回结果与预期字典 {'real': T, 'pos': T, 'npos': F} 相等
    assert D({'pos': T, 'npos': F}) == {'real': T, 'pos': T, 'npos': F}
    
    # 调用函数 D 并断言其返回结果与预期字典 {'real': T, 'pos': F, 'npos': T} 相等
    assert D({'pos': F, 'npos': T}) == {'real': T, 'pos': F, 'npos': T}
# 定义测试函数 test_FactRules_deduce_multiple2，用于测试 FactRules 类的 deduce_all_facts 方法
def test_FactRules_deduce_multiple2():
    # 创建 FactRules 对象 f，初始化规则为 ['real == neg | zero | pos']
    f = FactRules(['real == neg | zero | pos'])

    # 定义内部函数 D，接受 facts 参数，构建一个基于 f 的 FactKB 对象 kb，然后调用 deduce_all_facts 方法，返回结果
    def D(facts):
        kb = FactKB(f)
        kb.deduce_all_facts(facts)
        return kb

    # 断言不同的 facts 输入下，D 函数的返回结果
    assert D({'real': T}) == {'real': T}
    assert D({'real': F}) == {'real': F, 'neg': F, 'zero': F, 'pos': F}
    assert D({'neg': T}) == {'real': T, 'neg': T}
    assert D({'zero': T}) == {'real': T, 'zero': T}
    assert D({'pos': T}) == {'real': T, 'pos': T}

    # --- key tests below ---
    # 进行更多的断言，验证 deduce_all_facts 方法在不同条件下的输出
    assert D({'neg': F, 'zero': F, 'pos': F}) == {'real': F, 'neg': F,
             'zero': F, 'pos': F}
    assert D({'real': T, 'neg': F}) == {'real': T, 'neg': F}
    assert D({'real': T, 'zero': F}) == {'real': T, 'zero': F}
    assert D({'real': T, 'pos': F}) == {'real': T, 'pos': F}

    assert D({'real': T,           'zero': F, 'pos': F}) == {'real': T,
             'neg': T, 'zero': F, 'pos': F}
    assert D({'real': T, 'neg': F,            'pos': F}) == {'real': T,
             'neg': F, 'zero': T, 'pos': F}
    assert D({'real': T, 'neg': F, 'zero': F          }) == {'real': T,
             'neg': F, 'zero': F, 'pos': T}

    assert D({'neg': T, 'zero': F, 'pos': F}) == {'real': T, 'neg': T,
             'zero': F, 'pos': F}
    assert D({'neg': F, 'zero': T, 'pos': F}) == {'real': T, 'neg': F,
             'zero': T, 'pos': F}
    assert D({'neg': F, 'zero': F, 'pos': T}) == {'real': T, 'neg': F,
             'zero': F, 'pos': T}


# 定义测试函数 test_FactRules_deduce_base，用于验证从基础条件开始的推导过程
def test_FactRules_deduce_base():
    # 创建 FactRules 对象 f，初始化规则包括 'real == neg | zero | pos', 'neg -> real & !zero & !pos', 'pos -> real & !zero & !neg'
    f = FactRules(['real  == neg | zero | pos',
                   'neg   -> real & !zero & !pos',
                   'pos   -> real & !zero & !neg'])
    # 创建基于 f 的 FactKB 对象 base
    base = FactKB(f)

    # 在给定部分初始条件后进行断言，验证 deduce_all_facts 方法的输出结果
    base.deduce_all_facts({'real': T, 'neg': F})
    assert base == {'real': T, 'neg': F}

    base.deduce_all_facts({'zero': F})
    assert base == {'real': T, 'neg': F, 'zero': F, 'pos': T}


# 定义测试函数 test_FactRules_deduce_staticext，用于验证静态 beta-extensions 推导是否正确进行
def test_FactRules_deduce_staticext():
    # 创建 FactRules 对象 f，包括规则 'real == neg | zero | pos', 'neg -> real & !zero & !pos', 'pos -> real & !zero & !neg',
    # 'nneg == real & !neg', 'npos == real & !pos'
    f = FactRules(['real  == neg | zero | pos',
                   'neg   -> real & !zero & !pos',
                   'pos   -> real & !zero & !neg',
                   'nneg  == real & !neg',
                   'npos  == real & !pos'])

    # 验证不同的静态 beta-extensions 推导情况是否正确
    assert ('npos', True) in f.full_implications[('neg', True)]
    assert ('nneg', True) in f.full_implications[('pos', True)]
    assert ('nneg', True) in f.full_implications[('zero', True)]
    assert ('npos', True) in f.full_implications[('zero', True)]
```