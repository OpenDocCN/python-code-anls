# `D:\src\scipysrc\sympy\sympy\polys\tests\test_polyoptions.py`

```
# 导入所需模块和类
from sympy.polys.polyoptions import (
    Options, Expand, Gens, Wrt, Sort, Order, Field, Greedy, Domain,
    Split, Gaussian, Extension, Modulus, Symmetric, Strict, Auto,
    Frac, Formal, Polys, Include, All, Gen, Symbols, Method)
# 导入排序相关函数
from sympy.polys.orderings import lex
# 导入多项式定义相关模块
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX
# 导入异常处理相关类
from sympy.polys.polyerrors import OptionError, GeneratorsError
# 导入数学计算相关类和函数
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
# 导入测试框架相关类和函数
from sympy.testing.pytest import raises
# 导入符号变量
from sympy.abc import x, y, z


# 定义选项克隆测试函数
def test_Options_clone():
    # 创建选项对象opt，指定变量gens和域domain
    opt = Options((x, y, z), {'domain': 'ZZ'})

    # 断言检查gens是否包含指定的符号变量
    assert opt.gens == (x, y, z)
    # 断言检查domain是否为指定的整数环对象ZZ
    assert opt.domain == ZZ
    # 断言检查选项中是否没有order键
    assert ('order' in opt) is False

    # 克隆opt选项为new_opt，指定gens为(x, y)，order为'lex'
    new_opt = opt.clone({'gens': (x, y), 'order': 'lex'})

    # 再次检查原始opt中gens是否保持不变
    assert opt.gens == (x, y, z)
    # 再次检查原始opt中domain是否保持不变
    assert opt.domain == ZZ
    # 再次检查原始opt中是否没有order键
    assert ('order' in opt) is False

    # 断言检查new_opt中gens是否被成功更新为(x, y)
    assert new_opt.gens == (x, y)
    # 断言检查new_opt中domain是否保持为ZZ
    assert new_opt.domain == ZZ
    # 断言检查new_opt中是否存在order键
    assert ('order' in new_opt) is True


# 定义展开选项预处理测试函数
def test_Expand_preprocess():
    # 断言检查False作为输入时是否正确返回False
    assert Expand.preprocess(False) is False
    # 断言检查True作为输入时是否正确返回True
    assert Expand.preprocess(True) is True

    # 断言检查整数0作为输入时是否正确返回False
    assert Expand.preprocess(0) is False
    # 断言检查整数1作为输入时是否正确返回True
    assert Expand.preprocess(1) is True

    # 使用lambda函数检查传入符号变量x时是否引发OptionError异常
    raises(OptionError, lambda: Expand.preprocess(x))


# 定义展开选项后处理测试函数
def test_Expand_postprocess():
    # 创建包含展开选项的字典opt
    opt = {'expand': True}
    # 调用展开选项后处理函数
    Expand.postprocess(opt)

    # 断言检查opt字典是否保持不变
    assert opt == {'expand': True}


# 定义生成器选项预处理测试函数
def test_Gens_preprocess():
    # 断言检查包含None的元组作为输入时是否正确返回空元组()
    assert Gens.preprocess((None,)) == ()
    # 断言检查包含符号变量(x, y, z)的元组作为输入时是否正确返回该元组
    assert Gens.preprocess((x, y, z)) == (x, y, z)
    # 断言检查包含包含符号变量(x, y, z)的元组的元组作为输入时是否正确返回该元组
    assert Gens.preprocess(((x, y, z),)) == (x, y, z)

    # 创建一个不可交换的符号变量a
    a = Symbol('a', commutative=False)

    # 使用lambda函数检查包含重复符号变量(x, x, y)的元组作为输入时是否引发GeneratorsError异常
    raises(GeneratorsError, lambda: Gens.preprocess((x, x, y)))
    # 使用lambda函数检查包含不可交换符号变量(a)的元组作为输入时是否引发GeneratorsError异常
    raises(GeneratorsError, lambda: Gens.preprocess((x, y, a)))


# 定义生成器选项后处理测试函数
def test_Gens_postprocess():
    # 创建包含生成器选项的字典opt
    opt = {'gens': (x, y)}
    # 调用生成器选项后处理函数
    Gens.postprocess(opt)

    # 断言检查opt字典是否保持不变
    assert opt == {'gens': (x, y)}


# 定义排序选项预处理测试函数
def test_Wrt_preprocess():
    # 断言检查符号变量x作为输入时是否正确返回列表['x']
    assert Wrt.preprocess(x) == ['x']
    # 断言检查空字符串作为输入时是否正确返回空列表[]
    assert Wrt.preprocess('') == []
    # 断言检查空格字符作为输入时是否正确返回空列表[]
    assert Wrt.preprocess(' ') == []
    # 断言检查字符串'x,y'作为输入时是否正确返回列表['x', 'y']
    assert Wrt.preprocess('x,y') == ['x', 'y']
    # 断言检查字符串'x y'作为输入时是否正确返回列表['x', 'y']
    assert Wrt.preprocess('x y') == ['x', 'y']
    # 断言检查字符串'x, y'作为输入时是否正确返回列表['x', 'y']
    assert Wrt.preprocess('x, y') == ['x', 'y']
    # 断言检查字符串'x , y'作为输入时是否正确返回列表['x', 'y']
    assert Wrt.preprocess('x , y') == ['x', 'y']
    # 断言检查字符串' x, y'作为输入时是否正确返回列表['x', 'y']
    assert Wrt.preprocess(' x, y') == ['x', 'y']
    # 断言检查字符串' x,  y'作为输入时是否正确返回列表['x', 'y']
    assert Wrt.preprocess(' x,  y') == ['x', 'y']
    # 断言检查包含符号变量x和y的列表作为输入时是否正确返回列表['x', 'y']
    assert Wrt.preprocess([x, y]) == ['x', 'y']

    # 使用lambda函数检查包含逗号','的字符串作为输入时是否引发OptionError异常
    raises(OptionError, lambda: Wrt.preprocess(','))
    # 使用lambda函数检查整数0作为输入时是否引发OptionError异常
    raises(OptionError, lambda: Wrt.preprocess(0))


# 定义排序选项后处理测试函数
def test_Wrt_postprocess():
    # 创建包含排序选项的字典opt
    opt = {'wrt': ['x']}
    # 调用排序选项后处理函数
    Wrt.postprocess(opt)

    # 断言检查opt字典是否保持不变
    assert opt == {'wrt': ['x']}
    # 使用 raises 函数验证 Sort.preprocess({x, y, z}) 是否会引发 OptionError 异常
    raises(OptionError, lambda: Sort.preprocess({x, y, z}))
def test_Sort_postprocess():
    # 准备测试用例的选项字典
    opt = {'sort': 'x > y'}
    # 调用 Sort 类的 postprocess 方法，对选项字典进行处理
    Sort.postprocess(opt)

    # 断言处理后的选项字典与预期相符
    assert opt == {'sort': 'x > y'}


def test_Order_preprocess():
    # 断言 Order 类的 preprocess 方法能正确处理 'lex' 字符串
    assert Order.preprocess('lex') == lex


def test_Order_postprocess():
    # 准备测试用例的选项字典
    opt = {'order': True}
    # 调用 Order 类的 postprocess 方法，对选项字典进行处理
    Order.postprocess(opt)

    # 断言处理后的选项字典与预期相符
    assert opt == {'order': True}


def test_Field_preprocess():
    # 断言 Field 类的 preprocess 方法能正确处理布尔值 False 和 True
    assert Field.preprocess(False) is False
    assert Field.preprocess(True) is True

    # 断言 Field 类的 preprocess 方法能正确处理整数 0 和 1
    assert Field.preprocess(0) is False
    assert Field.preprocess(1) is True

    # 测试处理未知选项时是否引发 OptionError 异常
    raises(OptionError, lambda: Field.preprocess(x))


def test_Field_postprocess():
    # 准备测试用例的选项字典
    opt = {'field': True}
    # 调用 Field 类的 postprocess 方法，对选项字典进行处理
    Field.postprocess(opt)

    # 断言处理后的选项字典与预期相符
    assert opt == {'field': True}


def test_Greedy_preprocess():
    # 断言 Greedy 类的 preprocess 方法能正确处理布尔值 False 和 True
    assert Greedy.preprocess(False) is False
    assert Greedy.preprocess(True) is True

    # 断言 Greedy 类的 preprocess 方法能正确处理整数 0 和 1
    assert Greedy.preprocess(0) is False
    assert Greedy.preprocess(1) is True

    # 测试处理未知选项时是否引发 OptionError 异常
    raises(OptionError, lambda: Greedy.preprocess(x))


def test_Greedy_postprocess():
    # 准备测试用例的选项字典
    opt = {'greedy': True}
    # 调用 Greedy 类的 postprocess 方法，对选项字典进行处理
    Greedy.postprocess(opt)

    # 断言处理后的选项字典与预期相符
    assert opt == {'greedy': True}


def test_Domain_preprocess():
    # 断言 Domain 类的 preprocess 方法能正确处理给定的数学域对象
    assert Domain.preprocess(ZZ) == ZZ
    assert Domain.preprocess(QQ) == QQ
    assert Domain.preprocess(EX) == EX
    assert Domain.preprocess(FF(2)) == FF(2)
    assert Domain.preprocess(ZZ[x, y]) == ZZ[x, y]

    # 断言 Domain 类的 preprocess 方法能正确处理字符串表示的数学域
    assert Domain.preprocess('Z') == ZZ
    assert Domain.preprocess('Q') == QQ

    assert Domain.preprocess('ZZ') == ZZ
    assert Domain.preprocess('QQ') == QQ

    assert Domain.preprocess('EX') == EX

    assert Domain.preprocess('FF(23)') == FF(23)
    assert Domain.preprocess('GF(23)') == GF(23)

    # 测试处理不合法选项时是否引发 OptionError 异常
    raises(OptionError, lambda: Domain.preprocess('Z[]'))

    # 断言 Domain 类的 preprocess 方法能正确处理带变量的数学域字符串
    assert Domain.preprocess('Z[x]') == ZZ[x]
    assert Domain.preprocess('Q[x]') == QQ[x]
    assert Domain.preprocess('R[x]') == RR[x]
    assert Domain.preprocess('C[x]') == CC[x]

    assert Domain.preprocess('ZZ[x]') == ZZ[x]
    assert Domain.preprocess('QQ[x]') == QQ[x]
    assert Domain.preprocess('RR[x]') == RR[x]
    assert Domain.preprocess('CC[x]') == CC[x]

    assert Domain.preprocess('Z[x,y]') == ZZ[x, y]
    assert Domain.preprocess('Q[x,y]') == QQ[x, y]
    assert Domain.preprocess('R[x,y]') == RR[x, y]
    assert Domain.preprocess('C[x,y]') == CC[x, y]

    assert Domain.preprocess('ZZ[x,y]') == ZZ[x, y]
    assert Domain.preprocess('QQ[x,y]') == QQ[x, y]
    assert Domain.preprocess('RR[x,y]') == RR[x, y]
    assert Domain.preprocess('CC[x,y]') == CC[x, y]

    # 测试处理不合法选项时是否引发 OptionError 异常
    raises(OptionError, lambda: Domain.preprocess('Z()'))

    # 断言 Domain 类的 preprocess 方法能正确处理带参数的数学域字符串
    assert Domain.preprocess('Z(x)') == ZZ.frac_field(x)
    assert Domain.preprocess('Q(x)') == QQ.frac_field(x)

    assert Domain.preprocess('ZZ(x)') == ZZ.frac_field(x)
    assert Domain.preprocess('QQ(x)') == QQ.frac_field(x)

    assert Domain.preprocess('Z(x,y)') == ZZ.frac_field(x, y)
    assert Domain.preprocess('Q(x,y)') == QQ.frac_field(x, y)

    assert Domain.preprocess('ZZ(x,y)') == ZZ.frac_field(x, y)
    # 断言语句，验证 Domain.preprocess('QQ(x,y)') 的返回结果是否等于 QQ.frac_field(x, y)
    assert Domain.preprocess('QQ(x,y)') == QQ.frac_field(x, y)

    # 断言语句，验证 Domain.preprocess('Q<I>') 的返回结果是否等于 QQ.algebraic_field(I)
    assert Domain.preprocess('Q<I>') == QQ.algebraic_field(I)
    # 断言语句，验证 Domain.preprocess('QQ<I>') 的返回结果是否等于 QQ.algebraic_field(I)
    assert Domain.preprocess('QQ<I>') == QQ.algebraic_field(I)

    # 断言语句，验证 Domain.preprocess('Q<sqrt(2), I>') 的返回结果是否等于 QQ.algebraic_field(sqrt(2), I)
    assert Domain.preprocess('Q<sqrt(2), I>') == QQ.algebraic_field(sqrt(2), I)
    # 断言语句，验证 Domain.preprocess('QQ<sqrt(2), I>') 的返回结果是否等于 QQ.algebraic_field(sqrt(2), I)
    assert Domain.preprocess(
        'QQ<sqrt(2), I>') == QQ.algebraic_field(sqrt(2), I)

    # 断言语句，验证 Domain.preprocess('abc') 会引发 OptionError 异常
    raises(OptionError, lambda: Domain.preprocess('abc'))
def test_Domain_postprocess():
    # 测试用例：验证 Domain.postprocess 对于包含生成器和域的字典抛出 GeneratorsError 异常
    raises(GeneratorsError, lambda: Domain.postprocess({'gens': (x, y),
           'domain': ZZ[y, z]}))

    # 测试用例：验证 Domain.postprocess 对于空生成器的字典抛出 GeneratorsError 异常
    raises(GeneratorsError, lambda: Domain.postprocess({'gens': (),
           'domain': EX}))
    
    # 测试用例：验证 Domain.postprocess 对于缺少生成器的字典抛出 GeneratorsError 异常
    raises(GeneratorsError, lambda: Domain.postprocess({'domain': EX}))


def test_Split_preprocess():
    # 测试用例：验证 Split.preprocess 对于 False 返回 False
    assert Split.preprocess(False) is False
    # 测试用例：验证 Split.preprocess 对于 True 返回 True
    assert Split.preprocess(True) is True

    # 测试用例：验证 Split.preprocess 对于 0 返回 False
    assert Split.preprocess(0) is False
    # 测试用例：验证 Split.preprocess 对于 1 返回 True
    assert Split.preprocess(1) is True

    # 测试用例：验证 Split.preprocess 对于非预期的选项抛出 OptionError 异常
    raises(OptionError, lambda: Split.preprocess(x))


def test_Split_postprocess():
    # 测试用例：验证 Split.postprocess 对于包含 'split': True 的字典抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: Split.postprocess({'split': True}))


def test_Gaussian_preprocess():
    # 测试用例：验证 Gaussian.preprocess 对于 False 返回 False
    assert Gaussian.preprocess(False) is False
    # 测试用例：验证 Gaussian.preprocess 对于 True 返回 True
    assert Gaussian.preprocess(True) is True

    # 测试用例：验证 Gaussian.preprocess 对于 0 返回 False
    assert Gaussian.preprocess(0) is False
    # 测试用例：验证 Gaussian.preprocess 对于 1 返回 True
    assert Gaussian.preprocess(1) is True

    # 测试用例：验证 Gaussian.preprocess 对于非预期的选项抛出 OptionError 异常
    raises(OptionError, lambda: Gaussian.preprocess(x))


def test_Gaussian_postprocess():
    # 测试用例：验证 Gaussian.postprocess 对于传入的选项字典添加 'domain': QQ_I
    opt = {'gaussian': True}
    Gaussian.postprocess(opt)
    assert opt == {
        'gaussian': True,
        'domain': QQ_I,
    }


def test_Extension_preprocess():
    # 测试用例：验证 Extension.preprocess 对于 True 返回 True
    assert Extension.preprocess(True) is True
    # 测试用例：验证 Extension.preprocess 对于 1 返回 True
    assert Extension.preprocess(1) is True

    # 测试用例：验证 Extension.preprocess 对于空列表返回 None
    assert Extension.preprocess([]) is None

    # 测试用例：验证 Extension.preprocess 对于 sqrt(2) 返回 {sqrt(2)}
    assert Extension.preprocess(sqrt(2)) == {sqrt(2)}
    # 测试用例：验证 Extension.preprocess 对于包含 sqrt(2) 的列表返回 {sqrt(2)}
    assert Extension.preprocess([sqrt(2)]) == {sqrt(2)}

    # 测试用例：验证 Extension.preprocess 对于包含 sqrt(2) 和 I 的列表返回 {sqrt(2), I}
    assert Extension.preprocess([sqrt(2), I]) == {sqrt(2), I}

    # 测试用例：验证 Extension.preprocess 对于非预期的选项抛出 OptionError 异常
    raises(OptionError, lambda: Extension.preprocess(False))
    raises(OptionError, lambda: Extension.preprocess(0))


def test_Extension_postprocess():
    # 测试用例：验证 Extension.postprocess 对于传入的选项字典添加 'domain': QQ.algebraic_field(sqrt(2))
    opt = {'extension': {sqrt(2)}}
    Extension.postprocess(opt)
    assert opt == {
        'extension': {sqrt(2)},
        'domain': QQ.algebraic_field(sqrt(2)),
    }

    # 测试用例：验证 Extension.postprocess 对于 extension: True 的选项字典不修改
    opt = {'extension': True}
    Extension.postprocess(opt)
    assert opt == {'extension': True}


def test_Modulus_preprocess():
    # 测试用例：验证 Modulus.preprocess 对于 23 返回 23
    assert Modulus.preprocess(23) == 23
    # 测试用例：验证 Modulus.preprocess 对于 Integer(23) 返回 23
    assert Modulus.preprocess(Integer(23)) == 23

    # 测试用例：验证 Modulus.preprocess 对于 0 抛出 OptionError 异常
    raises(OptionError, lambda: Modulus.preprocess(0))
    # 测试用例：验证 Modulus.preprocess 对于非预期的选项抛出 OptionError 异常
    raises(OptionError, lambda: Modulus.preprocess(x))


def test_Modulus_postprocess():
    # 测试用例：验证 Modulus.postprocess 对于传入的选项字典添加 'domain': FF(5)
    opt = {'modulus': 5}
    Modulus.postprocess(opt)
    assert opt == {
        'modulus': 5,
        'domain': FF(5),
    }

    # 测试用例：验证 Modulus.postprocess 对于传入的选项字典添加 'domain': FF(5, False) 和 'symmetric': False
    opt = {'modulus': 5, 'symmetric': False}
    Modulus.postprocess(opt)
    assert opt == {
        'modulus': 5,
        'domain': FF(5, False),
        'symmetric': False,
    }


def test_Symmetric_preprocess():
    # 测试用例：验证 Symmetric.preprocess 对于 False 返回 False
    assert Symmetric.preprocess(False) is False
    # 测试用例：验证 Symmetric.preprocess 对于 True 返回 True
    assert Symmetric.preprocess(True) is True

    # 测试用例：验证 Symmetric.preprocess 对于 0 返回 False
    assert Symmetric.preprocess(0) is False
    # 测试用例：验证 Symmetric.preprocess 对于 1 返回 True
    assert Symmetric.preprocess(1) is True

    # 测试用例：验证 Symmetric.preprocess 对于非预期的选项抛出 OptionError 异常
    raises(OptionError, lambda: Symmetric.preprocess(x))


def test_Symmetric_postprocess():
    # 测试用例：验证 Symmetric.postprocess 对于传入的选项字典添加 'symmetric': True
    opt = {'symmetric': True}
    Symmetric.postprocess(opt)
    assert opt == {'symmetric': True}


def test_Strict_preprocess():
    # 测试用例：验证 Strict.preprocess 对于 False 返回 False
    assert Strict.preprocess(False) is False
    # 测试用例：验证 Strict.preprocess 对于 True 返回 True
    assert Strict.preprocess(True) is True
    # 使用 Strict 类中的 preprocess 方法，检查传入参数 0 是否返回 False
    assert Strict.preprocess(0) is False
    # 使用 Strict 类中的 preprocess 方法，检查传入参数 1 是否返回 True
    assert Strict.preprocess(1) is True

    # 使用 raises 函数验证传入参数 x 到 Strict.preprocess 方法时是否会引发 OptionError 异常
    raises(OptionError, lambda: Strict.preprocess(x))
# 测试 Strict 类的 postprocess 方法
def test_Strict_postprocess():
    # 创建一个包含 {'strict': True} 的选项字典
    opt = {'strict': True}
    # 调用 Strict 类的 postprocess 方法，不对 opt 进行修改
    Strict.postprocess(opt)

    # 断言 opt 字典仍然包含 {'strict': True}
    assert opt == {'strict': True}


# 测试 Auto 类的 preprocess 方法
def test_Auto_preprocess():
    # 断言 Auto.preprocess(False) 返回 False
    assert Auto.preprocess(False) is False
    # 断言 Auto.preprocess(True) 返回 True
    assert Auto.preprocess(True) is True

    # 断言 Auto.preprocess(0) 返回 False
    assert Auto.preprocess(0) is False
    # 断言 Auto.preprocess(1) 返回 True
    assert Auto.preprocess(1) is True

    # 断言调用 Auto.preprocess(x) 会引发 OptionError 异常
    raises(OptionError, lambda: Auto.preprocess(x))


# 测试 Auto 类的 postprocess 方法
def test_Auto_postprocess():
    # 创建一个包含 {'auto': True} 的选项字典
    opt = {'auto': True}
    # 调用 Auto 类的 postprocess 方法，不对 opt 进行修改
    Auto.postprocess(opt)

    # 断言 opt 字典仍然包含 {'auto': True}
    assert opt == {'auto': True}


# 测试 Frac 类的 preprocess 方法
def test_Frac_preprocess():
    # 断言 Frac.preprocess(False) 返回 False
    assert Frac.preprocess(False) is False
    # 断言 Frac.preprocess(True) 返回 True
    assert Frac.preprocess(True) is True

    # 断言 Frac.preprocess(0) 返回 False
    assert Frac.preprocess(0) is False
    # 断言 Frac.preprocess(1) 返回 True
    assert Frac.preprocess(1) is True

    # 断言调用 Frac.preprocess(x) 会引发 OptionError 异常
    raises(OptionError, lambda: Frac.preprocess(x))


# 测试 Frac 类的 postprocess 方法
def test_Frac_postprocess():
    # 创建一个包含 {'frac': True} 的选项字典
    opt = {'frac': True}
    # 调用 Frac 类的 postprocess 方法，不对 opt 进行修改
    Frac.postprocess(opt)

    # 断言 opt 字典仍然包含 {'frac': True}
    assert opt == {'frac': True}


# 测试 Formal 类的 preprocess 方法
def test_Formal_preprocess():
    # 断言 Formal.preprocess(False) 返回 False
    assert Formal.preprocess(False) is False
    # 断言 Formal.preprocess(True) 返回 True
    assert Formal.preprocess(True) is True

    # 断言 Formal.preprocess(0) 返回 False
    assert Formal.preprocess(0) is False
    # 断言 Formal.preprocess(1) 返回 True
    assert Formal.preprocess(1) is True

    # 断言调用 Formal.preprocess(x) 会引发 OptionError 异常
    raises(OptionError, lambda: Formal.preprocess(x))


# 测试 Formal 类的 postprocess 方法
def test_Formal_postprocess():
    # 创建一个包含 {'formal': True} 的选项字典
    opt = {'formal': True}
    # 调用 Formal 类的 postprocess 方法，不对 opt 进行修改
    Formal.postprocess(opt)

    # 断言 opt 字典仍然包含 {'formal': True}
    assert opt == {'formal': True}


# 测试 Polys 类的 preprocess 方法
def test_Polys_preprocess():
    # 断言 Polys.preprocess(False) 返回 False
    assert Polys.preprocess(False) is False
    # 断言 Polys.preprocess(True) 返回 True
    assert Polys.preprocess(True) is True

    # 断言 Polys.preprocess(0) 返回 False
    assert Polys.preprocess(0) is False
    # 断言 Polys.preprocess(1) 返回 True
    assert Polys.preprocess(1) is True

    # 断言调用 Polys.preprocess(x) 会引发 OptionError 异常
    raises(OptionError, lambda: Polys.preprocess(x))


# 测试 Polys 类的 postprocess 方法
def test_Polys_postprocess():
    # 创建一个包含 {'polys': True} 的选项字典
    opt = {'polys': True}
    # 调用 Polys 类的 postprocess 方法，不对 opt 进行修改
    Polys.postprocess(opt)

    # 断言 opt 字典仍然包含 {'polys': True}
    assert opt == {'polys': True}


# 测试 Include 类的 preprocess 方法
def test_Include_preprocess():
    # 断言 Include.preprocess(False) 返回 False
    assert Include.preprocess(False) is False
    # 断言 Include.preprocess(True) 返回 True
    assert Include.preprocess(True) is True

    # 断言 Include.preprocess(0) 返回 False
    assert Include.preprocess(0) is False
    # 断言 Include.preprocess(1) 返回 True
    assert Include.preprocess(1) is True

    # 断言调用 Include.preprocess(x) 会引发 OptionError 异常
    raises(OptionError, lambda: Include.preprocess(x))


# 测试 Include 类的 postprocess 方法
def test_Include_postprocess():
    # 创建一个包含 {'include': True} 的选项字典
    opt = {'include': True}
    # 调用 Include 类的 postprocess 方法，不对 opt 进行修改
    Include.postprocess(opt)

    # 断言 opt 字典仍然包含 {'include': True}
    assert opt == {'include': True}


# 测试 All 类的 preprocess 方法
def test_All_preprocess():
    # 断言 All.preprocess(False) 返回 False
    assert All.preprocess(False) is False
    # 断言 All.preprocess(True) 返回 True
    assert All.preprocess(True) is True

    # 断言 All.preprocess(0) 返回 False
    assert All.preprocess(0) is False
    # 断言 All.preprocess(1) 返回 True
    assert All.preprocess(1) is True

    # 断言调用 All.preprocess(x) 会引发 OptionError 异常
    raises(OptionError, lambda: All.preprocess(x))


# 测试 All 类的 postprocess 方法
def test_All_postprocess():
    # 创建一个包含 {'all': True} 的选项字典
    opt = {'all': True}
    # 调用 All 类的 postprocess 方法，不对 opt 进行修改
    All.postprocess(opt)

    # 断言 opt 字典仍然包含 {'all': True}
    assert opt == {'all': True}


# 测试 Gen 类的 postprocess 方法
def test_Gen_postprocess():
    # 创建一个包含 {'gen': x} 的选项字典
    opt = {'gen': x}
    # 调用 Gen 类的 postprocess 方法，不对 opt 进行修改
    Gen.postprocess(opt)

    # 断言 opt 字典仍然包含 {'gen': x}
    assert opt == {'gen': x}


# 测试 Symbols 类的 preprocess 方法
def test_Symbols_preprocess():
    # 断言调用 Symbols.preprocess(x) 会引发 OptionError 异常
    raises(OptionError, lambda: Symbols.preprocess(x))


# 测试 Symbols 类的 postprocess 方法
def test_Symbols_postprocess():
    # 创建一个包含 {'symbols': [x, y, z]} 的选项字典
    opt = {'symbols': [x, y, z]}
    # 调用 Symbols 类的 postprocess 方法，不对 opt 进行修改
    Symbols.postprocess(opt)

    # 断言 opt 字典仍然包含 {'symbols': [x, y, z]}
    assert opt == {'symbols': [x, y, z]}


# 测试 Method 类的 preprocess 方法
def test_Method_preprocess():
    # 断言调用 Method.preprocess(10) 会引发 OptionError 异常
    raises(OptionError, lambda: Method.preprocess(10))


# 测试 Method 类的 postprocess 方法
def test_Method_postprocess():
    # 创建一个包含 {'method': 'f5b'} 的选项字典
    opt = {'method': 'f5b'}
    # 调用 Method 类的 postprocess 方法，不对 opt 进行修改
    Method.postprocess(opt)

    # 断言 opt 字典仍然包含 {'method': 'f5b'}
    assert opt == {'method': 'f5b'}
```