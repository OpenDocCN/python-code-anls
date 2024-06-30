# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_matchpy_connector.py`

```
import pickle  # 导入pickle模块，用于对象的序列化和反序列化操作

from sympy.core.relational import (Eq, Ne)  # 导入Eq和Ne类，用于表示等式和不等式
from sympy.core.singleton import S  # 导入S类，表示单例符号
from sympy.core.symbol import symbols  # 导入symbols函数，用于创建符号变量
from sympy.functions.elementary.miscellaneous import sqrt  # 导入sqrt函数，用于计算平方根
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入cos和sin函数，用于三角函数计算
from sympy.external import import_module  # 导入import_module函数，用于动态导入模块
from sympy.testing.pytest import skip  # 导入skip函数，用于测试跳过
from sympy.utilities.matchpy_connector import WildDot, WildPlus, WildStar, Replacer  # 导入WildDot、WildPlus、WildStar和Replacer类

matchpy = import_module("matchpy")  # 导入matchpy模块，并赋值给matchpy变量

x, y, z = symbols("x y z")  # 创建符号变量x, y, z


def _get_first_match(expr, pattern):
    from matchpy import ManyToOneMatcher, Pattern  # 导入ManyToOneMatcher和Pattern类

    matcher = ManyToOneMatcher()  # 创建ManyToOneMatcher对象
    matcher.add(Pattern(pattern))  # 将Pattern对象添加到matcher中
    return next(iter(matcher.match(expr)))  # 返回匹配expr和pattern的第一个结果


def test_matchpy_connector():
    if matchpy is None:  # 如果matchpy模块未安装，则跳过测试
        skip("matchpy not installed")

    from multiset import Multiset  # 导入Multiset类
    from matchpy import Pattern, Substitution  # 导入Pattern和Substitution类

    w_ = WildDot("w_")  # 创建一个通配符WildDot对象w_
    w__ = WildPlus("w__")  # 创建一个通配符WildPlus对象w__
    w___ = WildStar("w___")  # 创建一个通配符WildStar对象w___

    expr = x + y  # 创建表达式x + y
    pattern = x + w_  # 创建模式x + w_
    p, subst = _get_first_match(expr, pattern)  # 获取表达式expr和模式pattern的第一个匹配结果
    assert p == Pattern(pattern)  # 断言匹配模式p等于Pattern(pattern)
    assert subst == Substitution({'w_': y})  # 断言替换字典subst包含键值对{'w_': y}

    expr = x + y + z  # 创建表达式x + y + z
    pattern = x + w__  # 创建模式x + w__
    p, subst = _get_first_match(expr, pattern)  # 获取表达式expr和模式pattern的第一个匹配结果
    assert p == Pattern(pattern)  # 断言匹配模式p等于Pattern(pattern)
    assert subst == Substitution({'w__': Multiset([y, z])})  # 断言替换字典subst包含键值对{'w__': Multiset([y, z])}

    expr = x + y + z  # 创建表达式x + y + z
    pattern = x + y + z + w___  # 创建模式x + y + z + w___
    p, subst = _get_first_match(expr, pattern)  # 获取表达式expr和模式pattern的第一个匹配结果
    assert p == Pattern(pattern)  # 断言匹配模式p等于Pattern(pattern)
    assert subst == Substitution({'w___': Multiset()})  # 断言替换字典subst包含键值对{'w___': Multiset()}


def test_matchpy_optional():
    if matchpy is None:  # 如果matchpy模块未安装，则跳过测试
        skip("matchpy not installed")

    from matchpy import Pattern, Substitution  # 导入Pattern和Substitution类

    p = WildDot("p", optional=1)  # 创建可选的通配符WildDot对象p
    q = WildDot("q", optional=0)  # 创建非可选的通配符WildDot对象q

    pattern = p*x + q  # 创建模式p*x + q

    expr1 = 2*x  # 创建表达式2*x
    pa, subst = _get_first_match(expr1, pattern)  # 获取表达式expr1和模式pattern的第一个匹配结果
    assert pa == Pattern(pattern)  # 断言匹配模式pa等于Pattern(pattern)
    assert subst == Substitution({'p': 2, 'q': 0})  # 断言替换字典subst包含键值对{'p': 2, 'q': 0}

    expr2 = x + 3  # 创建表达式x + 3
    pa, subst = _get_first_match(expr2, pattern)  # 获取表达式expr2和模式pattern的第一个匹配结果
    assert pa == Pattern(pattern)  # 断言匹配模式pa等于Pattern(pattern)
    assert subst == Substitution({'p': 1, 'q': 3})  # 断言替换字典subst包含键值对{'p': 1, 'q': 3}

    expr3 = x  # 创建表达式x
    pa, subst = _get_first_match(expr3, pattern)  # 获取表达式expr3和模式pattern的第一个匹配结果
    assert pa == Pattern(pattern)  # 断言匹配模式pa等于Pattern(pattern)
    assert subst == Substitution({'p': 1, 'q': 0})  # 断言替换字典subst包含键值对{'p': 1, 'q': 0}

    expr4 = x*y + z  # 创建表达式x*y + z
    pa, subst = _get_first_match(expr4, pattern)  # 获取表达式expr4和模式pattern的第一个匹配结果
    assert pa == Pattern(pattern)  # 断言匹配模式pa等于Pattern(pattern)
    assert subst == Substitution({'p': y, 'q': z})  # 断言替换字典subst包含键值对{'p': y, 'q': z}

    replacer = ManyToOneReplacer()  # 创建ManyToOneReplacer对象
    replacer.add(ReplacementRule(Pattern(pattern), lambda p, q: sin(p)*cos(q)))  # 添加替换规则
    assert replacer.replace(expr1) == sin(2)*cos(0)  # 断言替换结果符合预期
    assert replacer.replace(expr2) == sin(1)*cos(3)  # 断言替换结果符合预期
    assert replacer.replace(expr3) == sin(1)*cos(0)  # 断言替换结果符合预期
    assert replacer.replace(expr4) == sin(y)*cos(z)  # 断言替换结果符合预期


def test_replacer():
    if matchpy is None:  # 如果matchpy模块未安装，则跳过测试
        skip("matchpy not installed")

    for info in [True, False]:  # 遍历info的布尔值列表
        for lambdify in [True, False]:  # 遍历lambdify的布尔值列表
            _perform_test_replacer(info, lambdify)  # 执行测试


def _perform_test_replacer(info, lambdify):

    x1_ = WildDot("x1_")  # 创建通配符WildDot对象x1_
    x2_ = WildDot("x2_")  # 创建通配符WildDot对象x2_
    # 定义通配符，表示可能的表达式部分
    a_ = WildDot("a_", optional=S.One)
    b_ = WildDot("b_", optional=S.One)
    c_ = WildDot("c_", optional=S.Zero)

    # 创建一个替换器对象，用于模式匹配和替换
    replacer = Replacer(
        common_constraints=[
            # 自定义约束条件，确保表达式中不包含变量 x
            matchpy.CustomConstraint(lambda a_: not a_.has(x)),
            matchpy.CustomConstraint(lambda b_: not b_.has(x)),
            matchpy.CustomConstraint(lambda c_: not c_.has(x)),
        ],
        lambdify=lambdify,
        info=info
    )

    # 添加替换规则，将等式重写为隐式形式，除非已经是解决的形式：
    replacer.add(
        Eq(x1_, x2_),  # 模式匹配：x1_ 等于 x2_
        Eq(x1_ - x2_, 0),  # 替换为 x1_ - x2_ 等于 0
        conditions_nonfalse=[  # 条件限制：确保不为 False 的条件
            Ne(x2_, 0),  # x2_ 不等于 0
            Ne(x1_, 0),  # x1_ 不等于 0
            Ne(x1_, x),  # x1_ 不等于 x
            Ne(x2_, x)   # x2_ 不等于 x
        ],
        info=1  # 标识信息为 1
    )

    # 简单的实数方程求解器：
    replacer.add(
        Eq(a_*x + b_, 0),  # 模式匹配：a*x + b_ 等于 0
        Eq(x, -b_/a_),     # 替换为 x 等于 -b_/a_
        info=2              # 标识信息为 2
    )

    # 计算二次方程的判别式
    disc = b_**2 - 4*a_*c_

    # 添加替换规则，解二次方程：
    replacer.add(
        Eq(a_*x**2 + b_*x + c_, 0),  # 模式匹配：a*x**2 + b_*x + c_ 等于 0
        Eq(x, (-b_ - sqrt(disc))/(2*a_)) | Eq(x, (-b_ + sqrt(disc))/(2*a_)),  # 替换为二次方程的解
        conditions_nonfalse=[disc >= 0],  # 条件限制：确保判别式大于等于 0
        info=3  # 标识信息为 3
    )

    # 添加替换规则，解简化后的二次方程：
    replacer.add(
        Eq(a_*x**2 + c_, 0),  # 模式匹配：a*x**2 + c_ 等于 0
        Eq(x, sqrt(-c_/a_)) | Eq(x, -sqrt(-c_/a_)),  # 替换为简化后的二次方程的解
        conditions_nonfalse=[-c_*a_ > 0],  # 条件限制：确保条件 -c_*a_ 大于 0
        info=4  # 标识信息为 4
    )

    # 定义 lambda 函数 g，根据条件返回表达式及其信息
    g = lambda expr, infos: (expr, infos) if info else expr

    # 使用 assert 断言验证替换的结果是否符合预期
    assert replacer.replace(Eq(3*x, y)) == g(Eq(x, y/3), [1, 2])
    assert replacer.replace(Eq(x**2 + 1, 0)) == g(Eq(x**2 + 1, 0), [])
    assert replacer.replace(Eq(x**2, 4)) == g((Eq(x, 2) | Eq(x, -2)), [1, 4])
    assert replacer.replace(Eq(x**2 + 4*y*x + 4*y**2, 0)) == g(Eq(x, -2*y), [3])
# 定义一个测试函数，用于测试matchpy对象的序列化和反序列化是否正确
def test_matchpy_object_pickle():
    # 如果matchpy模块未导入，则直接返回，不进行测试
    if matchpy is None:
        return

    # 创建一个名为'a'的通配符对象a1，并将其序列化后再反序列化为a2
    a1 = WildDot("a")
    a2 = pickle.loads(pickle.dumps(a1))
    # 断言：序列化前后的通配符对象a1和a2应当相等
    assert a1 == a2

    # 创建一个名为'a'、绑定值为S(1)的通配符对象a1，并将其序列化后再反序列化为a2
    a1 = WildDot("a", S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    # 断言：序列化前后的通配符对象a1和a2应当相等
    assert a1 == a2

    # 创建一个名为'a'、绑定值为S(1)的通配符对象，并将其序列化后再反序列化为a2
    a1 = WildPlus("a", S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    # 断言：序列化前后的通配符对象a1和a2应当相等
    assert a1 == a2

    # 创建一个名为'a'、绑定值为S(1)的通配符对象，并将其序列化后再反序列化为a2
    a1 = WildStar("a", S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    # 断言：序列化前后的通配符对象a1和a2应当相等
    assert a1 == a2
```