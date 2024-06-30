# `D:\src\scipysrc\sympy\sympy\categories\baseclasses.py`

```
from sympy.core import S, Basic, Dict, Symbol, Tuple, sympify  # 导入所需的符号计算模块
from sympy.core.symbol import Str  # 导入字符串符号类
from sympy.sets import Set, FiniteSet, EmptySet  # 导入集合相关类
from sympy.utilities.iterables import iterable  # 导入可迭代工具函数

class Class(Set):
    r"""
    The base class for any kind of class in the set-theoretic sense.

    Explanation
    ===========

    In axiomatic set theories, everything is a class.  A class which
    can be a member of another class is a set.  A class which is not a
    member of another class is a proper class.  The class `\{1, 2\}`
    is a set; the class of all sets is a proper class.

    This class is essentially a synonym for :class:`sympy.core.Set`.
    The goal of this class is to assure easier migration to the
    eventual proper implementation of set theory.
    """
    is_proper = False  # 设置类的属性为非 proper 类


class Object(Symbol):
    """
    The base class for any kind of object in an abstract category.

    Explanation
    ===========

    While technically any instance of :class:`~.Basic` will do, this
    class is the recommended way to create abstract objects in
    abstract categories.
    """


class Morphism(Basic):
    """
    The base class for any morphism in an abstract category.

    Explanation
    ===========

    In abstract categories, a morphism is an arrow between two
    category objects.  The object where the arrow starts is called the
    domain, while the object where the arrow ends is called the
    codomain.

    Two morphisms between the same pair of objects are considered to
    be the same morphisms.  To distinguish between morphisms between
    the same objects use :class:`NamedMorphism`.

    It is prohibited to instantiate this class.  Use one of the
    derived classes instead.

    See Also
    ========

    IdentityMorphism, NamedMorphism, CompositeMorphism
    """
    def __new__(cls, domain, codomain):
        raise(NotImplementedError(
            "Cannot instantiate Morphism.  Use derived classes instead."))

    @property
    def domain(self):
        """
        Returns the domain of the morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> f.domain
        Object("A")

        """
        return self.args[0]  # 返回箭头的起始对象作为域

    @property
    def codomain(self):
        """
        Returns the codomain of the morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> f.codomain
        Object("B")

        """
        return self.args[1]  # 返回箭头的结束对象作为余域
    def compose(self, other):
        r"""
        Composes self with the supplied morphism.

        The order of elements in the composition is the usual order,
        i.e., to construct `g\circ f` use ``g.compose(f)``.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> g * f
        CompositeMorphism((NamedMorphism(Object("A"), Object("B"), "f"),
        NamedMorphism(Object("B"), Object("C"), "g")))
        >>> (g * f).domain
        Object("A")
        >>> (g * f).codomain
        Object("C")

        """
        return CompositeMorphism(other, self)



        r"""
        Composes self with the supplied morphism.

        The semantics of this operation is given by the following
        equation: ``g * f == g.compose(f)`` for composable morphisms
        ``g`` and ``f``.

        See Also
        ========

        compose
        """
        return self.compose(other)
class IdentityMorphism(Morphism):
    """
    Represents an identity morphism.

    Explanation
    ===========

    An identity morphism is a morphism with equal domain and codomain,
    which acts as an identity with respect to composition.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, IdentityMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> f = NamedMorphism(A, B, "f")
    >>> id_A = IdentityMorphism(A)
    >>> id_B = IdentityMorphism(B)
    >>> f * id_A == f
    True
    >>> id_B * f == f
    True

    See Also
    ========

    Morphism
    """

    def __new__(cls, domain):
        # Create a new instance of IdentityMorphism with the specified domain
        return Basic.__new__(cls, domain)

    @property
    def codomain(self):
        # Return the codomain of this morphism, which is the same as its domain
        return self.domain


class NamedMorphism(Morphism):
    """
    Represents a morphism which has a name.

    Explanation
    ===========

    Names are used to distinguish between morphisms which have the
    same domain and codomain: two named morphisms are equal if they
    have the same domains, codomains, and names.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> f = NamedMorphism(A, B, "f")
    >>> f
    NamedMorphism(Object("A"), Object("B"), "f")
    >>> f.name
    'f'

    See Also
    ========

    Morphism
    """

    def __new__(cls, domain, codomain, name):
        # Ensure the name is not empty and convert it to a string if necessary
        if not name:
            raise ValueError("Empty morphism names not allowed.")

        if not isinstance(name, Str):
            name = Str(name)

        # Create a new instance of NamedMorphism with the specified domain, codomain, and name
        return Basic.__new__(cls, domain, codomain, name)

    @property
    def name(self):
        """
        Returns the name of the morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> f.name
        'f'

        """
        # Return the name stored in the third argument of the morphism's arguments tuple
        return self.args[2].name


class CompositeMorphism(Morphism):
    r"""
    Represents a morphism which is a composition of other morphisms.

    Explanation
    ===========

    Two composite morphisms are equal if the morphisms they were
    obtained from (components) are the same and were listed in the
    same order.

    The arguments to the constructor for this class should be listed
    in diagram order: to obtain the composition `g\circ f` from the
    instances of :class:`Morphism` ``g`` and ``f`` use
    ``CompositeMorphism(f, g)``.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, CompositeMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> g * f
    CompositeMorphism((NamedMorphism(Object("A"), Object("B"), "f"),
    NamedMorphism(Object("B"), Object("C"), "g")))
    >>> CompositeMorphism(f, g) == g * f
    True

    """
    @staticmethod
    @property
    def components(self):
        """
        返回此复合态射的组件。

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).components
        (NamedMorphism(Object("A"), Object("B"), "f"),
        NamedMorphism(Object("B"), Object("C"), "g"))

        """
        返回这个复合态射的组件（即构成这个复合态射的各个基本态射）。
        这个属性允许通过调用 .components 来访问复合态射对象的所有基本态射组件。
        在文档字符串中提供了一个示例，展示了如何使用这个属性来访问和操作复合态射的组件。
    def domain(self):
        """
        Returns the domain of this composite morphism.

        The domain of the composite morphism is the domain of its
        first component.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).domain
        Object("A")

        """
        # 返回此复合态射的定义域，即其第一个组成部分的定义域
        return self.components[0].domain

    @property
    def codomain(self):
        """
        Returns the codomain of this composite morphism.

        The codomain of the composite morphism is the codomain of its
        last component.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).codomain
        Object("C")

        """
        # 返回此复合态射的陪域，即其最后一个组成部分的陪域
        return self.components[-1].codomain

    def flatten(self, new_name):
        """
        Forgets the composite structure of this morphism.

        Explanation
        ===========

        If ``new_name`` is not empty, returns a :class:`NamedMorphism`
        with the supplied name, otherwise returns a :class:`Morphism`.
        In both cases the domain of the new morphism is the domain of
        this composite morphism and the codomain of the new morphism
        is the codomain of this composite morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).flatten("h")
        NamedMorphism(Object("A"), Object("C"), "h")

        """
        # 忽略此态射的复合结构，根据给定的名称创建一个新的命名态射或态射
        # 新态射的定义域是此复合态射的定义域，陪域是此复合态射的陪域
        return NamedMorphism(self.domain, self.codomain, new_name)
class Category(Basic):
    r"""
    An (abstract) category.

    Explanation
    ===========

    A category [JoyOfCats] is a quadruple `\mbox{K} = (O, \hom, id,
    \circ)` consisting of

    * a (set-theoretical) class `O`, whose members are called
      `K`-objects,

    * for each pair `(A, B)` of `K`-objects, a set `\hom(A, B)` whose
      members are called `K`-morphisms from `A` to `B`,

    * for a each `K`-object `A`, a morphism `id:A\rightarrow A`,
      called the `K`-identity of `A`,

    * a composition law `\circ` associating with every `K`-morphisms
      `f:A\rightarrow B` and `g:B\rightarrow C` a `K`-morphism `g\circ
      f:A\rightarrow C`, called the composite of `f` and `g`.

    Composition is associative, `K`-identities are identities with
    respect to composition, and the sets `\hom(A, B)` are pairwise
    disjoint.

    This class knows nothing about its objects and morphisms.
    Concrete cases of (abstract) categories should be implemented as
    classes derived from this one.

    Certain instances of :class:`Diagram` can be asserted to be
    commutative in a :class:`Category` by supplying the argument
    ``commutative_diagrams`` in the constructor.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram, Category
    >>> from sympy import FiniteSet
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> d = Diagram([f, g])
    >>> K = Category("K", commutative_diagrams=[d])
    >>> K.commutative_diagrams == FiniteSet(d)
    True

    See Also
    ========

    Diagram
    """
    def __new__(cls, name, objects=EmptySet, commutative_diagrams=EmptySet):
        # 检查类别名不为空
        if not name:
            raise ValueError("A Category cannot have an empty name.")
        
        # 确保类别名是字符串类型
        if not isinstance(name, Str):
            name = Str(name)
        
        # 确保对象集合是类的实例
        if not isinstance(objects, Class):
            objects = Class(objects)
        
        # 创建新的 Category 实例
        new_category = Basic.__new__(cls, name, objects,
                                     FiniteSet(*commutative_diagrams))
        return new_category

    @property
    def name(self):
        """
        Returns the name of this category.

        Examples
        ========

        >>> from sympy.categories import Category
        >>> K = Category("K")
        >>> K.name
        'K'

        """
        # 返回类别的名称
        return self.args[0].name

    @property
    def objects(self):
        """
        Returns the class of objects of this category.

        Examples
        ========

        >>> from sympy.categories import Object, Category
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> K = Category("K", FiniteSet(A, B))
        >>> K.objects
        Class({Object("A"), Object("B")})

        """
        # 返回类别的对象集合
        return self.args[1]

    @property
    def commutative_diagrams(self):
        """
        Returns the set of commutative diagrams in this category.

        """
        # 返回类别中的交换图集合
        return self.args[2]
    def commutative_diagrams(self):
        """
        Returns the :class:`~.FiniteSet` of diagrams which are known to
        be commutative in this category.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram, Category
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g])
        >>> K = Category("K", commutative_diagrams=[d])
        >>> K.commutative_diagrams == FiniteSet(d)
        True

        """
        # 返回当前类别中已知为可交换的图表的有限集合
        return self.args[2]

    def hom(self, A, B):
        # 抛出未实现错误，表示在类别中尚未实现同态集合
        raise NotImplementedError(
            "hom-sets are not implemented in Category.")

    def all_morphisms(self):
        # 抛出未实现错误，表示在类别中尚未实现获取所有态射的功能
        raise NotImplementedError(
            "Obtaining the class of morphisms is not implemented in Category.")
class Diagram(Basic):
    r"""
    Represents a diagram in a certain category.

    Explanation
    ===========

    Informally, a diagram is a collection of objects of a category and
    certain morphisms between them.  A diagram is still a monoid with
    respect to morphism composition; i.e., identity morphisms, as well
    as all composites of morphisms included in the diagram belong to
    the diagram.  For a more formal approach to this notion see
    [Pare1970].

    The components of composite morphisms are also added to the
    diagram.  No properties are assigned to such morphisms by default.

    A commutative diagram is often accompanied by a statement of the
    following kind: "if such morphisms with such properties exist,
    then such morphisms which such properties exist and the diagram is
    commutative".  To represent this, an instance of :class:`Diagram`
    includes a collection of morphisms which are the premises and
    another collection of conclusions.  ``premises`` and
    ``conclusions`` associate morphisms belonging to the corresponding
    categories with the :class:`~.FiniteSet`'s of their properties.

    The set of properties of a composite morphism is the intersection
    of the sets of properties of its components.  The domain and
    codomain of a conclusion morphism should be among the domains and
    codomains of the morphisms listed as the premises of a diagram.

    No checks are carried out of whether the supplied object and
    morphisms do belong to one and the same category.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy import pprint, default_sort_key
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> d = Diagram([f, g])
    >>> premises_keys = sorted(d.premises.keys(), key=default_sort_key)
    >>> pprint(premises_keys, use_unicode=False)
    [g*f:A-->C, id:A-->A, id:B-->B, id:C-->C, f:A-->B, g:B-->C]
    >>> pprint(d.premises, use_unicode=False)
    {g*f:A-->C: EmptySet, id:A-->A: EmptySet, id:B-->B: EmptySet,
     id:C-->C: EmptySet, f:A-->B: EmptySet, g:B-->C: EmptySet}
    >>> d = Diagram([f, g], {g * f: "unique"})
    >>> pprint(d.conclusions,use_unicode=False)
    {g*f:A-->C: {unique}}

    References
    ==========

    [Pare1970] B. Pareigis: Categories and functors.  Academic Press, 1970.

    """
    @staticmethod
    # 定义静态方法的装饰器，表明该方法与特定实例无关，可以直接通过类调用
    def _set_dict_union(dictionary, key, value):
        """
        If ``key`` is in ``dictionary``, set the new value of ``key``
        to be the union between the old value and ``value``.
        Otherwise, set the value of ``key`` to ``value``.

        Returns ``True`` if the key already was in the dictionary and
        ``False`` otherwise.
        """
        # 检查 key 是否已经在 dictionary 中
        if key in dictionary:
            # 如果是，则将 key 对应的值更新为其与 value 的并集
            dictionary[key] = dictionary[key] | value
            return True
        else:
            # 如果不是，则直接设置 key 对应的值为 value
            dictionary[key] = value
            return False

    @staticmethod
    def _add_morphism_closure(morphisms, morphism, props, add_identities=True,
                              recurse_composites=True):
        """
        Adds a morphism and its attributes to the supplied dictionary
        ``morphisms``.  If ``add_identities`` is True, also adds the
        identity morphisms for the domain and the codomain of
        ``morphism``.
        """
        # 调用 _set_dict_union 方法尝试添加 morphism 到 morphisms 字典中
        if not Diagram._set_dict_union(morphisms, morphism, props):
            # 如果成功添加了新的 morphism

            # 如果 morphism 是 IdentityMorphism 类型
            if isinstance(morphism, IdentityMorphism):
                if props:
                    # IdentityMorphism 实例不应该具有任何属性，因为它们是平凡的
                    raise ValueError(
                        "Instances of IdentityMorphism cannot have properties.")
                return

            # 如果 add_identities 为 True，则添加 domain 和 codomain 的 identity morphisms
            if add_identities:
                empty = EmptySet

                id_dom = IdentityMorphism(morphism.domain)
                id_cod = IdentityMorphism(morphism.codomain)

                Diagram._set_dict_union(morphisms, id_dom, empty)
                Diagram._set_dict_union(morphisms, id_cod, empty)

            # 遍历 morphisms 字典中的每个已有 morphism 和其对应的属性
            for existing_morphism, existing_props in list(morphisms.items()):
                new_props = existing_props & props
                # 如果当前 morphism 的 domain 与 existing_morphism 的 codomain 匹配
                if morphism.domain == existing_morphism.codomain:
                    left = morphism * existing_morphism
                    # 将 left 添加到 morphisms 字典，并更新其属性为 new_props 的并集
                    Diagram._set_dict_union(morphisms, left, new_props)
                # 如果当前 morphism 的 codomain 与 existing_morphism 的 domain 匹配
                if morphism.codomain == existing_morphism.domain:
                    right = existing_morphism * morphism
                    # 将 right 添加到 morphisms 字典，并更新其属性为 new_props 的并集
                    Diagram._set_dict_union(morphisms, right, new_props)

            # 如果 morphism 是 CompositeMorphism 类型并且 recurse_composites 为 True
            if isinstance(morphism, CompositeMorphism) and recurse_composites:
                # 这是一个复合态射，添加其组成部分
                empty = EmptySet
                for component in morphism.components:
                    # 递归调用 _add_morphism_closure 添加 component 到 morphisms 字典
                    Diagram._add_morphism_closure(morphisms, component, empty,
                                                  add_identities)

    @property
    def premises(self):
        """
        Returns the premises of this diagram.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import IdentityMorphism, Diagram
        >>> from sympy import pretty
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> id_A = IdentityMorphism(A)
        >>> id_B = IdentityMorphism(B)
        >>> d = Diagram([f])
        >>> print(pretty(d.premises, use_unicode=False))
        {id:A-->A: EmptySet, id:B-->B: EmptySet, f:A-->B: EmptySet}

        """
        # 返回当前对象的第一个参数，即图表的前提部分
        return self.args[0]

    @property
    def conclusions(self):
        """
        Returns the conclusions of this diagram.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import IdentityMorphism, Diagram
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g])
        >>> IdentityMorphism(A) in d.premises.keys()
        True
        >>> g * f in d.premises.keys()
        True
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> d.conclusions[g * f] == FiniteSet("unique")
        True

        """
        # 返回当前对象的第二个参数，即图表的结论部分
        return self.args[1]

    @property
    def objects(self):
        """
        Returns the :class:`~.FiniteSet` of objects that appear in this
        diagram.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g])
        >>> d.objects
        {Object("A"), Object("B"), Object("C")}

        """
        # 返回当前对象的第三个参数，即出现在图表中的对象的有限集合
        return self.args[2]
    def hom(self, A, B):
        """
        Returns a 2-tuple of sets of morphisms between objects ``A`` and
        ``B``: one set of morphisms listed as premises, and the other set
        of morphisms listed as conclusions.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> from sympy import pretty
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> print(pretty(d.hom(A, C), use_unicode=False))
        ({g*f:A-->C}, {g*f:A-->C})

        See Also
        ========
        Object, Morphism
        """
        # Initialize sets for premises and conclusions
        premises = EmptySet
        conclusions = EmptySet

        # Iterate through premises to find morphisms from A to B
        for morphism in self.premises.keys():
            if (morphism.domain == A) and (morphism.codomain == B):
                premises |= FiniteSet(morphism)

        # Iterate through conclusions to find morphisms from A to B
        for morphism in self.conclusions.keys():
            if (morphism.domain == A) and (morphism.codomain == B):
                conclusions |= FiniteSet(morphism)

        # Return a tuple containing premises and conclusions sets
        return (premises, conclusions)

    def is_subdiagram(self, diagram):
        """
        Checks whether ``diagram`` is a subdiagram of ``self``.
        Diagram `D'` is a subdiagram of `D` if all premises
        (conclusions) of `D'` are contained in the premises
        (conclusions) of `D`.  The morphisms contained
        both in `D'` and `D` should have the same properties for `D'`
        to be a subdiagram of `D`.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> d1 = Diagram([f])
        >>> d.is_subdiagram(d1)
        True
        >>> d1.is_subdiagram(d)
        False
        """
        # Check if all premises in diagram are also in self
        premises = all((m in self.premises) and
                       (diagram.premises[m] == self.premises[m])
                       for m in diagram.premises)
        
        # If premises are not fully contained, return False
        if not premises:
            return False

        # Check if all conclusions in diagram are also in self
        conclusions = all((m in self.conclusions) and
                          (diagram.conclusions[m] == self.conclusions[m])
                          for m in diagram.conclusions)

        # Since premises are True at this point, return conclusions
        return conclusions
    def subdiagram_from_objects(self, objects):
        """
        如果 ``objects`` 是 ``self`` 对象的子集，则返回一个图表，
        其中前提是 ``self`` 的所有前提中的定义域和值域都在 ``objects`` 中，
        结论亦然。属性将被保留。

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g], {f: "unique", g*f: "veryunique"})
        >>> d1 = d.subdiagram_from_objects(FiniteSet(A, B))
        >>> d1 == Diagram([f], {f: "unique"})
        True
        """
        # 检查提供的对象是否全部属于该图表的对象集合
        if not objects.is_subset(self.objects):
            raise ValueError(
                "Supplied objects should all belong to the diagram.")

        new_premises = {}
        # 遍历图表的前提
        for morphism, props in self.premises.items():
            # 如果前提的定义域和值域在提供的对象集合中
            if ((sympify(objects.contains(morphism.domain)) is S.true) and
                (sympify(objects.contains(morphism.codomain)) is S.true)):
                new_premises[morphism] = props

        new_conclusions = {}
        # 遍历图表的结论
        for morphism, props in self.conclusions.items():
            # 如果结论的定义域和值域在提供的对象集合中
            if ((sympify(objects.contains(morphism.domain)) is S.true) and
                (sympify(objects.contains(morphism.codomain)) is S.true)):
                new_conclusions[morphism] = props

        # 返回新的图表，其中包含符合提供对象条件的前提和结论
        return Diagram(new_premises, new_conclusions)
```