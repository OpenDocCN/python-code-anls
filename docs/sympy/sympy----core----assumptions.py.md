# `D:\src\scipysrc\sympy\sympy\core\assumptions.py`

```
"""
This module contains the machinery handling assumptions.
Do also consider the guide :ref:`assumptions-guide`.

All symbolic objects have assumption attributes that can be accessed via
``.is_<assumption name>`` attribute.

Assumptions determine certain properties of symbolic objects and can
have 3 possible values: ``True``, ``False``, ``None``.  ``True`` is returned if the
object has the property and ``False`` is returned if it does not or cannot
(i.e. does not make sense):

    >>> from sympy import I
    >>> I.is_algebraic
    True
    >>> I.is_real
    False
    >>> I.is_prime
    False

When the property cannot be determined (or when a method is not
implemented) ``None`` will be returned. For example,  a generic symbol, ``x``,
may or may not be positive so a value of ``None`` is returned for ``x.is_positive``.

By default, all symbolic values are in the largest set in the given context
without specifying the property. For example, a symbol that has a property
being integer, is also real, complex, etc.

Here follows a list of possible assumption names:

.. glossary::

    commutative
        object commutes with any other object with
        respect to multiplication operation. See [12]_.

    complex
        object can have only values from the set
        of complex numbers. See [13]_.

    imaginary
        object value is a number that can be written as a real
        number multiplied by the imaginary unit ``I``.  See
        [3]_.  Please note that ``0`` is not considered to be an
        imaginary number, see
        `issue #7649 <https://github.com/sympy/sympy/issues/7649>`_.

    real
        object can have only values from the set
        of real numbers.

    extended_real
        object can have only values from the set
        of real numbers, ``oo`` and ``-oo``.

    integer
        object can have only values from the set
        of integers.

    odd
    even
        object can have only values from the set of
        odd (even) integers [2]_.

    prime
        object is a natural number greater than 1 that has
        no positive divisors other than 1 and itself.  See [6]_.

    composite
        object is a positive integer that has at least one positive
        divisor other than 1 or the number itself.  See [4]_.

    zero
        object has the value of 0.

    nonzero
        object is a real number that is not zero.

    rational
        object can have only values from the set
        of rationals.

    algebraic
        object can have only values from the set
        of algebraic numbers [11]_.

    transcendental
        object can have only values from the set
        of transcendental numbers [10]_.

    irrational
        object value cannot be represented exactly by :class:`~.Rational`, see [5]_.

    finite
    infinite
        object absolute value is bounded (arbitrarily large).
        See [7]_, [8]_, [9]_.

    negative

"""

# 注释：
# 这段代码段并未执行任何实际的Python代码操作，而是提供了关于符号对象属性假设的详细文档说明。
# 文档列出了所有可能的假设名和它们对应的含义，例如"real"表示对象可以是实数，"prime"表示对象是素数等。
# 每个假设名后面都有详细的描述和相关引用，帮助理解这些属性在符号计算中的应用和意义。
    # 定义一个术语 "nonnegative"，表示对象只能拥有负值（非负值）[1]_。
    nonnegative
    
    # 定义术语 "positive"，表示对象只能有正值。
    positive
    
    # 定义术语 "nonpositive"，表示对象只能有非正值。
    nonpositive
    
    # 定义术语 "extended_negative" 和 "extended_nonnegative"，表示对象与其扩展部分一样，但也包括带有相应符号的无穷大，例如 extended_positive 包括 `oo`。
    extended_negative
    extended_nonnegative
    
    # 定义术语 "extended_positive"、"extended_nonpositive" 和 "extended_nonzero"，表示对象与其扩展部分一样，但也包括带有相应符号的无穷大，例如 extended_positive 包括 `oo`。
    extended_positive
    extended_nonpositive
    extended_nonzero
    
    # 定义术语 "hermitian" 和 "antihermitian"，表示对象属于埃尔米特（反埃尔米特）算符的领域。
    hermitian
    antihermitian
# 导入符号类 Symbol 从 sympy 模块
>>> from sympy import Symbol
# 创建一个名为 x 的符号，设定其为实数
>>> x = Symbol('x', real=True); x
x
# 检查符号 x 是否是实数
>>> x.is_real
True
# 检查符号 x 是否是复数
>>> x.is_complex
True
# 引入从Sympy库中的utilities.exceptions模块导入的sympy_deprecation_warning异常
from sympy.utilities.exceptions import sympy_deprecation_warning

# 从当前目录下的facts模块中导入FactRules和FactKB类
from .facts import FactRules, FactKB
# 从当前目录下的sympify模块中导入sympify函数
from .sympify import sympify

# 从Sympy核心模块中的random模块导入_assumptions_shuffle别名为shuffle
from sympy.core.random import _assumptions_shuffle as shuffle
# 从Sympy核心模块的assumptions_generated模块导入generated_assumptions别名为_assumptions
from sympy.core.assumptions_generated import generated_assumptions as _assumptions

# 定义一个函数_load_pre_generated_assumption_rules，返回类型为FactRules
def _load_pre_generated_assumption_rules() -> FactRules:
    """ Load the assumption rules from pre-generated data

    To update the pre-generated data, see :method::`_generate_assumption_rules`
    """
    # 使用FactRules类的_from_python方法从_python对象_assumptions创建_assume_rules对象
    _assume_rules = FactRules._from_python(_assumptions)
    # 返回_assume_rules对象
    return _assume_rules

# 定义一个函数_generate_assumption_rules，没有返回值
def _generate_assumption_rules():
    """ Generate the default assumption rules

    This method should only be called to update the pre-generated
    assumption rules.

    To update the pre-generated assumptions run: bin/ask_update.py

    """
    # 创建一个FactRules对象，包含预定义的假设规则列表
    _assume_rules = FactRules([
        'integer        ->  rational',
        'rational       ->  real',
        'rational       ->  algebraic',
        'algebraic      ->  complex',
        'transcendental ==  complex & !algebraic',
        'real           ->  hermitian',
        'imaginary      ->  complex',
        'imaginary      ->  antihermitian',
        'extended_real  ->  commutative',
        'complex        ->  commutative',
        'complex        ->  finite',

        'odd            ==  integer & !even',
        'even           ==  integer & !odd',

        'real           ->  complex',
        'extended_real  ->  real | infinite',
        'real           ==  extended_real & finite',

        'extended_real        ==  extended_negative | zero | extended_positive',
        'extended_negative    ==  extended_nonpositive & extended_nonzero',
        'extended_positive    ==  extended_nonnegative & extended_nonzero',

        'extended_nonpositive ==  extended_real & !extended_positive',
        'extended_nonnegative ==  extended_real & !extended_negative',

        'real           ==  negative | zero | positive',
        'negative       ==  nonpositive & nonzero',
        'positive       ==  nonnegative & nonzero',

        'nonpositive    ==  real & !positive',
        'nonnegative    ==  real & !negative',

        'positive       ==  extended_positive & finite',
        'negative       ==  extended_negative & finite',
        'nonpositive    ==  extended_nonpositive & finite',
        'nonnegative    ==  extended_nonnegative & finite',
        'nonzero        ==  extended_nonzero & finite',

        'zero           ->  even & finite',
        'zero           ==  extended_nonnegative & extended_nonpositive',
        'zero           ==  nonnegative & nonpositive',
        'nonzero        ->  real',

        'prime          ->  integer & positive',
        'composite      ->  integer & positive & !prime',
        '!composite     ->  !positive | !even | prime',

        'irrational     ==  real & !rational',

        'imaginary      ->  !extended_real',

        'infinite       ==  !finite',
        'noninteger     ==  extended_real & !integer',
    ])

    # 注意：此函数没有返回语句，仅用于生成预定义假设规则
    'extended_nonzero == extended_real & !zero',
    ```
    # 创建一个包含单个字符串的列表，该字符串为逻辑表达式
    ])
    # 返回包含该字符串的列表作为函数的结果
    return _assume_rules
# 加载预生成的假设规则
_assume_rules = _load_pre_generated_assumption_rules()
# 创建假设定义的副本集合
_assume_defined = _assume_rules.defined_facts.copy()
# 添加额外的假设 'polar' 到定义集合中
_assume_defined.add('polar')
# 将假设定义集合转换为不可变集合（frozenset），确保其内容不可更改
_assume_defined = frozenset(_assume_defined)


def assumptions(expr, _check=None):
    """返回表达式``expr``的真假假设"""
    # 将表达式转换为符号对象
    n = sympify(expr)
    if n.is_Symbol:
        # 如果是符号，则返回其假设属性字典
        rv = n.assumptions0  # 是否缺少重要假设？
        if _check is not None:
            # 如果提供了检查集合，则只保留该集合中的假设属性
            rv = {k: rv[k] for k in set(rv) & set(_check)}
        return rv
    rv = {}
    for k in _assume_defined if _check is None else _check:
        # 对于假设定义集合中的每个假设属性，检查其是否适用于表达式
        v = getattr(n, 'is_{}'.format(k))
        if v is not None:
            rv[k] = v
    return rv


def common_assumptions(exprs, check=None):
    """返回所有给定表达式共同的真假假设值

    Examples
    ========

    >>> from sympy.core import common_assumptions
    >>> from sympy import oo, pi, sqrt
    >>> common_assumptions([-4, 0, sqrt(2), 2, pi, oo])
    {'commutative': True, 'composite': False,
    'extended_real': True, 'imaginary': False, 'odd': False}

    默认情况下，测试所有假设；可以传入一个假设集合来限制报告的假设：

    >>> common_assumptions([0, 1, 2], ['positive', 'integer'])
    {'integer': True}
    """
    # 如果未提供假设集合或者表达式集合，则返回空字典
    check = _assume_defined if check is None else set(check)
    if not check or not exprs:
        return {}

    # 获取每个表达式的所有假设
    assume = [assumptions(i, _check=check) for i in sympify(exprs)]
    # 仅关注所有表达式中共同的真假假设
    for i, e in enumerate(assume):
        assume[i] = {k: e[k] for k in set(e) & check}
    # 找出所有表达式共同具有的假设
    common = set.intersection(*[set(i) for i in assume])
    # 确定这些共同假设在所有表达式中是否具有相同的值
    a = assume[0]
    return {k: a[k] for k in common if all(a[k] == b[k]
        for b in assume)}


def failing_assumptions(expr, **assumptions):
    """
    返回一个包含与传递假设不匹配的假设及其值的字典

    Examples
    ========

    >>> from sympy import failing_assumptions, Symbol

    >>> x = Symbol('x', positive=True)
    >>> y = Symbol('y')
    >>> failing_assumptions(6*x + y, positive=True)
    {'positive': None}

    >>> failing_assumptions(x**2 - 1, positive=True)
    {'positive': None}

    如果*expr*满足所有假设，则返回空字典。

    >>> failing_assumptions(x**2, positive=True)
    {}

    """
    # 将表达式转换为符号对象
    expr = sympify(expr)
    failed = {}
    for k in assumptions:
        # 检查表达式是否具有指定假设，并与期望的值进行比较
        test = getattr(expr, 'is_%s' % k, None)
        if test is not assumptions[k]:
            failed[k] = test
    return failed  # 返回假设与预期值不匹配的情况，可能为空字典或包含假设及其值


def check_assumptions(expr, against=None, **assume):
    """
    检查表达式的假设是否与给定的（或``against``中拥有的）假设相匹配。如果所有假设
    都匹配，则返回True。

    ```
    # 将表达式转换为符号表达式
    expr = sympify(expr)
    
    # 如果传入了 `against` 参数，则根据其进行假设
    if against is not None:
        # 如果已经定义了假设，则抛出值错误
        if assume:
            raise ValueError(
                'Expecting `against` or `assume`, not both.')
        # 获取 `against` 的假设
        assume = assumptions(against)
    
    # 默认为已知的假设
    known = True
    
    # 遍历假设字典中的每一项
    for k, v in assume.items():
        # 如果假设的值为 None，则继续下一项
        if v is None:
            continue
        # 获取表达式的属性，例如 is_real、is_positive 等
        e = getattr(expr, 'is_' + k, None)
        # 如果获取的属性为 None，则假设为不确定
        if e is None:
            known = None
        # 如果假设的值与属性不符合，则返回 False
        elif v != e:
            return False
    
    # 返回假设是否匹配的结果（True、False 或 None）
    return known
    """
    A specialized knowledge base for handling built-in rules.

    This class extends FactKB and is intended for Basic objects to utilize.

    Attributes:
        _generator (dict): A dictionary containing facts and their values.

    Methods:
        __init__(self, facts=None):
            Initializes the object with a set of initial facts.
        
        copy(self):
            Returns a copy of the current object.
        
        generator(self):
            Property method that returns a copy of _generator attribute.
    """
    def __init__(self, facts=None):
        # Initialize using FactKB's constructor with built-in rules
        super().__init__(_assume_rules)
        
        # Save a copy of the facts dict
        if not facts:
            self._generator = {}
        elif not isinstance(facts, FactKB):
            self._generator = facts.copy()
        else:
            self._generator = facts.generator
        
        # Deduce all facts from the provided facts
        if facts:
            self.deduce_all_facts(facts)

    def copy(self):
        """
        Returns:
            StdFactKB: A copy of the current object.
        """
        return self.__class__(self)

    @property
    def generator(self):
        """
        Returns:
            dict: A copy of the _generator attribute containing facts and their values.
        """
        return self._generator.copy()
    for fact_i in facts_to_check:
        # 对于每个待检查的事实 fact_i

        # 如果 assumptions 中已经存在 fact_i 的值，则无需重新运行处理器。
        # 但是在多线程代码中存在潜在的竞争条件，因为 fact_i 可能在另一个线程中被检查。
        # 下面循环的主要逻辑可能会跳过在这种情况下检查 assumptions[fact]，因此在循环后检查一次。
        if fact_i in assumptions:
            continue

        # 现在我们调用与 fact_i 相关联的处理器（如果存在）。
        fact_i_value = None
        handler_i = handler_map.get(fact_i)
        if handler_i is not None:
            fact_i_value = handler_i(obj)

        # 如果我们得到了 fact_i 的新值，则我们应该更新对 fact_i 的知识，
        # 以及可以通过推理规则推断出的任何相关事实。
        if fact_i_value is not None:
            assumptions.deduce_all_facts(((fact_i, fact_i_value),))

        # 通常，如果 assumptions[fact] 现在不为 None，则这是因为上面的 deduce_all_facts 调用。
        # fact_i 的处理器返回了 True 或 False，并且知道 fact_i（在第一次迭代中等于 fact）暗示了 fact 的一个值。
        # 不过也可能是独立的代码（例如间接由处理器调用或在多线程环境中由另一个线程调用）导致 assumptions[fact] 被设置。
        # 无论哪种方式，我们都将其返回。
        fact_value = assumptions.get(fact)
        if fact_value is not None:
            return fact_value

        # 将可能决定 fact_i 的其他事实加入队列。
        # 在这里我们随机化检查事实的顺序。如果所有处理器在事实的推理规则上都与逻辑一致，
        # 随机化检查顺序不应导致任何非确定性。处理器中的错误可能通过这个 shuffle 调用暴露为非确定性的假设查询，
        # 这些查询被推送到队列的末尾，意味着推理图以广度优先顺序遍历。
        new_facts_to_check = list(_assume_rules.prereq[fact_i] - facts_queued)
        shuffle(new_facts_to_check)
        facts_to_check.extend(new_facts_to_check)
        facts_queued.update(new_facts_to_check)

    # 在单线程上下文中，上述循环应该能很好地处理一切。
    # 但在多线程代码中，可能会发生这样的情况：本线程跳过计算另一个线程已经计算的特定事实（由于 continue）。
    # 在这种情况下，可能已经推断出 fact 并存储在 assumptions 字典中，但在循环体中未进行检查。
    # 这是一个微妙的情况，但为了确保
    # 如果已经在假设中找到了该事实，则直接返回该事实对应的值
    if fact in assumptions:
        return assumptions[fact]

    # 如果无法回答此查询，可能是因为其他线程已经将 None 存储为该事实的值，
    # 但是 assumptions._tell 方法不会在我们调用两次设置相同值时报错。
    # 如果此处抛出 InconsistentAssumptions 异常，则可能意味着另一个线程
    # 尝试计算该事实，并获得了 True 或 False 的值，而不是 None。
    # 在这种情况下，至少一个处理程序可能存在 bug。
    # 如果处理程序都是确定性的，并且符合推理规则，则所有线程中都应该计算出相同的值。
    assumptions._tell(fact, None)
    return None
# 准备类级别的假设和生成处理程序。

def _prepare_class_assumptions(cls):
    """Precompute class level assumptions and generate handlers.
    
    This is called by Basic.__init_subclass__ each time a Basic subclass is
    defined.
    """

    # 本地定义字典，存储属性的假设值
    local_defs = {}
    for k in _assume_defined:
        # 将假设名称转换为属性名
        attrname = as_property(k)
        # 获取类属性的值
        v = cls.__dict__.get(attrname, '')
        # 如果属性值是布尔型、整型或者None类型，则转换为布尔型
        if isinstance(v, (bool, int, type(None))):
            if v is not None:
                v = bool(v)
            local_defs[k] = v

    # 定义字典，存储继承链上的所有显式类级别假设
    defs = {}
    for base in reversed(cls.__bases__):
        assumptions = getattr(base, '_explicit_class_assumptions', None)
        if assumptions is not None:
            defs.update(assumptions)
    defs.update(local_defs)

    # 将计算得到的类级别假设存储到类的属性中
    cls._explicit_class_assumptions = defs
    # 创建一个标准事实知识库对象，基于类级别的假设
    cls.default_assumptions = StdFactKB(defs)

    # 创建属性处理程序的字典
    cls._prop_handler = {}
    for k in _assume_defined:
        # 检查类是否有对应的属性处理方法
        eval_is_meth = getattr(cls, '_eval_is_%s' % k, None)
        if eval_is_meth is not None:
            cls._prop_handler[k] = eval_is_meth

    # 将确定的结果直接放入类的字典中，以提高访问速度
    for k, v in cls.default_assumptions.items():
        setattr(cls, as_property(k), v)

    # 为了保护例如 Integer.is_even=F <- (Rational.is_integer=F) 这样的逻辑关系
    derived_from_bases = set()
    for base in cls.__bases__:
        default_assumptions = getattr(base, 'default_assumptions', None)
        if default_assumptions is not None:
            derived_from_bases.update(default_assumptions)

    # 根据继承链补充缺失的自动属性
    for fact in derived_from_bases - set(cls.default_assumptions):
        pname = as_property(fact)
        # 如果属性不存在于类的字典中，则创建该属性
        if pname not in cls.__dict__:
            setattr(cls, pname, make_property(fact))

    # 最后，为 Basic 类补充任何缺失的自动属性
    for fact in _assume_defined:
        pname = as_property(fact)
        # 如果属性不存在于类中，则创建该属性
        if not hasattr(cls, pname):
            setattr(cls, pname, make_property(fact))


# XXX: ManagedProperties 曾经是 Basic 的元类，但现在 Basic 不再使用元类。我们暂时保留这段代码以保证向后兼容性，
# 以防某些地方的代码仍然使用 ManagedProperties 类。之前需要使用 ManagedProperties 是因为在子类化一个类并希望使用元类时，
# 元类必须是被子类化的类的元类的子类。任何希望子类化 Basic 并在其子类中使用元类的人都需要子类化 ManagedProperties。
# 现在 ManagedProperties 不再是 Basic 的元类，但对于 Basic 的子类仍然可以使用它作为元类，因为它是 type 的子类，而 type 现在是 Basic 的元类。

class ManagedProperties(type):
    # 初始化函数，用于创建类的实例时调用
    def __init__(cls, *args, **kwargs):
        # 提示信息，指示 ManagedProperties 元类不再被 Basic 使用
        msg = ("The ManagedProperties metaclass. "
               "Basic does not use metaclasses any more")
        # 发出 sympy 的弃用警告，提醒自版本 "1.12" 起已不再使用此元类
        sympy_deprecation_warning(msg,
            deprecated_since_version="1.12",
            active_deprecations_target='managedproperties')

        # 在某些情况下仍调用此函数，尤其是对于不是 Basic 的子类使用 ManagedProperties 的情况。
        # 对于 Basic 的子类，__init_subclass__ 已经负责调用此函数，因此此元类不再需要。
        _prepare_class_assumptions(cls)
```