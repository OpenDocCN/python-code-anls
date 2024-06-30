# `D:\src\scipysrc\sympy\sympy\categories\tests\test_baseclasses.py`

```
from sympy.categories import (Object, Morphism, IdentityMorphism,
                              NamedMorphism, CompositeMorphism,
                              Diagram, Category)
from sympy.categories.baseclasses import Class
from sympy.testing.pytest import raises
from sympy.core.containers import (Dict, Tuple)
from sympy.sets import EmptySet
from sympy.sets.sets import FiniteSet


def test_morphisms():
    A = Object("A")  # 创建一个名为"A"的对象A
    B = Object("B")  # 创建一个名为"B"的对象B
    C = Object("C")  # 创建一个名为"C"的对象C
    D = Object("D")  # 创建一个名为"D"的对象D

    # Test the base morphism.
    f = NamedMorphism(A, B, "f")  # 创建从A到B的命名态射"f"
    assert f.domain == A  # 断言f的定义域是A
    assert f.codomain == B  # 断言f的值域是B
    assert f == NamedMorphism(A, B, "f")  # 断言f等于一个具有相同参数的命名态射对象

    # Test identities.
    id_A = IdentityMorphism(A)  # 创建对象A上的恒同态射
    id_B = IdentityMorphism(B)  # 创建对象B上的恒同态射
    assert id_A.domain == A  # 断言id_A的定义域是A
    assert id_A.codomain == A  # 断言id_A的值域是A
    assert id_A == IdentityMorphism(A)  # 断言id_A等于一个具有相同参数的恒同态射对象
    assert id_A != id_B  # 断言id_A不等于id_B

    # Test named morphisms.
    g = NamedMorphism(B, C, "g")  # 创建从B到C的命名态射"g"
    assert g.name == "g"  # 断言g的名称是"g"
    assert g != f  # 断言g不等于f
    assert g == NamedMorphism(B, C, "g")  # 断言g等于一个具有相同参数的命名态射对象
    assert g != NamedMorphism(B, C, "f")  # 断言g不等于一个从B到C名称为"f"的命名态射对象

    # Test composite morphisms.
    assert f == CompositeMorphism(f)  # 断言f等于其自身的复合态射

    k = g.compose(f)  # 创建g和f的复合态射k
    assert k.domain == A  # 断言k的定义域是A
    assert k.codomain == C  # 断言k的值域是C
    assert k.components == Tuple(f, g)  # 断言k的组成部分是(f, g)
    assert g * f == k  # 断言g * f等于k
    assert CompositeMorphism(f, g) == k  # 断言具有f和g作为参数的复合态射等于k

    assert CompositeMorphism(g * f) == g * f  # 断言具有g * f作为参数的复合态射等于g * f

    # Test the associativity of composition.
    h = NamedMorphism(C, D, "h")  # 创建从C到D的命名态射"h"

    p = h * g  # 创建h和g的复合态射p
    u = h * g * f  # 创建h、g和f的复合态射u

    assert h * k == u  # 断言h * k等于u
    assert p * f == u  # 断言p * f等于u
    assert CompositeMorphism(f, g, h) == u  # 断言具有f、g和h作为参数的复合态射等于u

    # Test flattening.
    u2 = u.flatten("u")  # 将u扁平化为一个命名态射u2
    assert isinstance(u2, NamedMorphism)  # 断言u2是一个命名态射对象
    assert u2.name == "u"  # 断言u2的名称是"u"
    assert u2.domain == A  # 断言u2的定义域是A
    assert u2.codomain == D  # 断言u2的值域是D

    # Test identities.
    assert f * id_A == f  # 断言f与id_A的复合态射等于f
    assert id_B * f == f  # 断言id_B与f的复合态射等于f
    assert id_A * id_A == id_A  # 断言id_A与id_A的复合态射等于id_A
    assert CompositeMorphism(id_A) == id_A  # 断言具有id_A作为参数的复合态射等于id_A

    # Test bad compositions.
    raises(ValueError, lambda: f * g)  # 断言尝试对不兼容的态射进行复合会引发值错误

    raises(TypeError, lambda: f.compose(None))  # 断言尝试将None作为复合操作的参数会引发类型错误
    raises(TypeError, lambda: id_A.compose(None))  # 断言尝试将None作为复合操作的参数会引发类型错误
    raises(TypeError, lambda: f * None)  # 断言尝试将None作为复合操作的参数会引发类型错误
    raises(TypeError, lambda: id_A * None)  # 断言尝试将None作为复合操作的参数会引发类型错误

    raises(TypeError, lambda: CompositeMorphism(f, None, 1))  # 断言尝试使用不兼容的参数创建复合态射会引发类型错误

    raises(ValueError, lambda: NamedMorphism(A, B, ""))  # 断言尝试使用空字符串创建命名态射会引发值错误
    raises(NotImplementedError, lambda: Morphism(A, B))  # 断言尝试创建未实现的通用态射会引发未实现的错误


def test_diagram():
    A = Object("A")  # 创建一个名为"A"的对象A
    B = Object("B")  # 创建一个名为"B"的对象B
    C = Object("C")  # 创建一个名为"C"的对象C

    f = NamedMorphism(A, B, "f")  # 创建从A到B的命名态射"f"
    g = NamedMorphism(B, C, "g")  # 创建从B到C的命名态射"g"
    id_A = IdentityMorphism(A)  # 创建对象A上的恒同态射
    id_B = IdentityMorphism(B)  # 创建对象B上的恒同态射

    empty = EmptySet  # 创建一个空集对象empty

    # Test the addition of identities.
    d1 = Diagram([f])  # 创建一个包含f的图表d1

    assert d1.objects == FiniteSet(A, B)  # 断言d1的对象集合是{A, B}
    assert d1.hom(A, B) == (FiniteSet(f), empty)  # 断言d1中从A到B的态射包含(f, empty)
    assert d1.hom(A, A) == (FiniteSet(id_A), empty)  # 断言d1中从A到A的态射包含(id_A, empty)
    assert d1.hom(B, B) == (FiniteSet(id_B), empty)  # 断言d1中从B到B的态射包含(id_B, empty)

    assert d1 == Diagram([id_A, f])  # 断言d1等于包含id_A和f的图表
    assert d1 == Diagram([f, f])  # 断言d1等于包含两个f的图表

    # Test the addition of composites.
    d2 = Diagram([f, g])  # 创建一个包含f和g的图表d2
    homAC = d2.hom(A, C)[0]  # 获取d2中从A到C的态射集合的第一个元素

    assert d2.objects == FiniteSet(A, B, C)  # 断言d2的对象集合是{A, B, C}
    # 断言 g * f 是否在 d2 的前提字典的键中
    assert g * f in d2.premises.keys()
    # 断言 homAC 是否等于 FiniteSet(g * f)
    assert homAC == FiniteSet(g * f)

    # 测试相等性、不相等性和哈希值。
    # 创建包含单个箭头 f 的图表 d11
    d11 = Diagram([f])
    # 断言 d1 与 d11 相等
    assert d1 == d11
    # 断言 d1 与 d2 不相等
    assert d1 != d2
    # 断言 d1 的哈希值与 d11 的哈希值相等
    assert hash(d1) == hash(d11)

    # 使用具有新属性的复合箭头再次添加，检查是否按预期工作。
    # 创建包含箭头 f 和 g 的图表 d，同时指定 g * f 的属性为 "unique"
    d = Diagram([f, g], {g * f: "unique"})
    # 断言 d 的结论包含键 g * f，并且其对应的值是 FiniteSet("unique")
    assert d.conclusions == Dict({g * f: FiniteSet("unique")})

    # 当存在前提和结论时，检查同态集的情况。
    # 断言 d 中从 A 到 C 的同态集为 (FiniteSet(g * f), FiniteSet(g * f))
    assert d.hom(A, C) == (FiniteSet(g * f), FiniteSet(g * f))
    # 创建包含箭头 f 和 g 的图表 d，同时指定 g * f 作为结论
    d = Diagram([f, g], [g * f])
    # 断言 d 中从 A 到 C 的同态集为 (FiniteSet(g * f), FiniteSet(g * f))
    assert d.hom(A, C) == (FiniteSet(g * f), FiniteSet(g * f))

    # 检查如何计算复合箭头的属性。
    # 创建包含箭头 f 和 g 的图表 d，同时指定箭头 f 的属性为 ["unique", "isomorphism"]，箭头 g 的属性为 "unique"
    d = Diagram({f: ["unique", "isomorphism"], g: "unique"})
    # 断言 d 中前提 g * f 的值为 FiniteSet("unique")
    assert d.premises[g * f] == FiniteSet("unique")

    # 检查不允许具有新对象的结论箭头。
    # 创建包含箭头 f 的图表 d，同时指定箭头 g 作为结论
    d = Diagram([f], [g])
    # 断言 d 的结论为空字典
    assert d.conclusions == Dict({})

    # 测试空图表。
    d = Diagram()
    # 断言 d 的前提为空字典
    assert d.premises == Dict({})
    # 断言 d 的结论为空字典
    assert d.conclusions == Dict({})
    # 断言 d 的对象为 empty（假设为预定义的空对象）
    assert d.objects == empty

    # 检查 SymPy 的 Dict 对象。
    # 创建包含箭头 f 的图表 d，同时指定箭头 f 的属性为 FiniteSet("unique", "isomorphism")，箭头 g 的属性为 "unique"
    d = Diagram(Dict({f: FiniteSet("unique", "isomorphism"), g: "unique"}))
    # 断言 d 中前提 g * f 的值为 FiniteSet("unique")
    assert d.premises[g * f] == FiniteSet("unique")

    # 检查添加复合箭头的组件。
    # 创建包含箭头 g * f 的图表 d
    d = Diagram([g * f])
    # 断言 d 的前提包含箭头 f
    assert f in d.premises
    # 断言 d 的前提包含箭头 g
    assert g in d.premises

    # 检查子图表。
    # 创建包含箭头 f 和 g 的图表 d，同时指定 g * f 作为结论
    d = Diagram([f, g], {g * f: "unique"})

    # 创建包含箭头 f 的图表 d1
    d1 = Diagram([f])
    # 断言 d1 是 d 的子图表
    assert d.is_subdiagram(d1)
    # 断言 d 不是 d1 的子图表
    assert not d1.is_subdiagram(d)

    # 创建包含箭头 f' 的图表 d，其中 f' 是从 B 到 A 的命名箭头
    d = Diagram([NamedMorphism(B, A, "f'")])
    # 断言 d 不是 d1 的子图表
    assert not d.is_subdiagram(d1)
    # 断言 d1 不是 d 的子图表
    assert not d1.is_subdiagram(d)

    # 创建包含箭头 f 和 g 的图表 d1，同时指定 g * f 的属性为 ["unique", "something"]
    d1 = Diagram([f, g], {g * f: ["unique", "something"]})
    # 断言 d 不是 d1 的子图表
    assert not d.is_subdiagram(d1)
    # 断言 d1 不是 d 的子图表
    assert not d1.is_subdiagram(d)

    # 创建包含箭头 f 的图表 d，同时指定箭头 f 的属性为 "blooh"
    d = Diagram({f: "blooh"})
    # 创建包含箭头 f 的图表 d1，同时指定箭头 f 的属性为 "bleeh"
    d1 = Diagram({f: "bleeh"})
    # 断言 d 不是 d1 的子图表
    assert not d.is_subdiagram(d1)
    # 断言 d1 不是 d 的子图表
    assert not d1.is_subdiagram(d)

    # 创建包含箭头 f 和 g 的图表 d，同时指定箭头 f 的属性为 "unique"，箭头 g * f 的属性为 "veryunique"
    d = Diagram([f, g], {f: "unique", g * f: "veryunique"})
    # 从对象集合 FiniteSet(A, B) 中创建 d 的子图表 d1
    d1 = d.subdiagram_from_objects(FiniteSet(A, B))
    # 断言 d1 等于 Diagram([f], {f: "unique"})
    assert d1 == Diagram([f], {f: "unique"})
    # 使用 lambda 函数断言 d 从对象集合 FiniteSet(A, Object("D")) 中创建子图表时引发 ValueError 异常
    raises(ValueError, lambda: d.subdiagram_from_objects(FiniteSet(A,
           Object("D"))))

    # 使用 lambda 函数断言尝试创建具有 IdentityMorphism(A) 的图表时引发 ValueError 异常
    raises(ValueError, lambda: Diagram({IdentityMorphism(A): "unique"}))
# 定义一个测试函数，用于测试类别（Category）的相关功能
def test_category():
    # 创建三个对象 A、B、C，分别用字符串 "A"、"B"、"C" 初始化
    A = Object("A")
    B = Object("B")
    C = Object("C")

    # 创建从 A 到 B 的命名态射 f，从 B 到 C 的命名态射 g，它们的名称分别为 "f" 和 "g"
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")

    # 创建包含 f 和 g 两个命名态射的图表（Diagram）
    d1 = Diagram([f, g])
    # 创建只包含 f 一个命名态射的图表
    d2 = Diagram([f])

    # 将 d1 和 d2 图表中的所有对象合并成一个集合，并赋值给 objects 变量
    objects = d1.objects | d2.objects

    # 创建一个名为 "K" 的类别（Category），其对象集合为 objects，包含的交换图表为 d1 和 d2
    K = Category("K", objects, commutative_diagrams=[d1, d2])

    # 断言类别 K 的名称为 "K"
    assert K.name == "K"
    # 断言类别 K 的对象集合为 Class(objects)
    assert K.objects == Class(objects)
    # 断言类别 K 的交换图表集合为 FiniteSet(d1, d2)
    assert K.commutative_diagrams == FiniteSet(d1, d2)

    # 使用 lambda 表达式测试是否会引发 ValueError 异常，因为传递空字符串给 Category 构造函数会引发异常
    raises(ValueError, lambda: Category(""))
```