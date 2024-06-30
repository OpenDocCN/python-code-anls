# `D:\src\scipysrc\sympy\sympy\assumptions\cnf.py`

```
"""
The classes used here are for the internal use of assumptions system
only and should not be used anywhere else as these do not possess the
signatures common to SymPy objects. For general use of logic constructs
please refer to sympy.logic classes And, Or, Not, etc.
"""
# 导入 itertools 中的 combinations, product, zip_longest 函数
from itertools import combinations, product, zip_longest
# 导入 sympy 中的各种逻辑相关类和函数
from sympy.assumptions.assume import AppliedPredicate, Predicate
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.logic.boolalg import Or, And, Not, Xnor
from sympy.logic.boolalg import (Equivalent, ITE, Implies, Nand, Nor, Xor)


class Literal:
    """
    The smallest element of a CNF object.

    Parameters
    ==========

    lit : Boolean expression

    is_Not : bool

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import Literal
    >>> from sympy.abc import x
    >>> Literal(Q.even(x))
    Literal(Q.even(x), False)
    >>> Literal(~Q.even(x))
    Literal(Q.even(x), True)
    """
    
    def __new__(cls, lit, is_Not=False):
        # 如果 lit 是 Not 类型，则提取其参数并设置 is_Not 为 True
        if isinstance(lit, Not):
            lit = lit.args[0]
            is_Not = True
        # 如果 lit 是 AND, OR, Literal 之一，则根据 is_Not 返回相应的逻辑操作
        elif isinstance(lit, (AND, OR, Literal)):
            return ~lit if is_Not else lit
        # 使用父类的 __new__ 方法创建 Literal 对象
        obj = super().__new__(cls)
        obj.lit = lit
        obj.is_Not = is_Not
        return obj

    @property
    def arg(self):
        return self.lit

    def rcall(self, expr):
        # 如果 lit 是可调用的，则调用 lit(expr)，否则调用 lit.apply(expr)
        if callable(self.lit):
            lit = self.lit(expr)
        else:
            lit = self.lit.apply(expr)
        return type(self)(lit, self.is_Not)

    def __invert__(self):
        # 取反操作，即将 is_Not 取反
        is_Not = not self.is_Not
        return Literal(self.lit, is_Not)

    def __str__(self):
        return '{}({}, {})'.format(type(self).__name__, self.lit, self.is_Not)

    __repr__ = __str__

    def __eq__(self, other):
        # 判断两个 Literal 对象是否相等
        return self.arg == other.arg and self.is_Not == other.is_Not

    def __hash__(self):
        # 计算 Literal 对象的哈希值
        h = hash((type(self).__name__, self.arg, self.is_Not))
        return h


class OR:
    """
    A low-level implementation for Or
    """
    def __init__(self, *args):
        # 初始化 OR 类，接收多个参数
        self._args = args

    @property
    def args(self):
        # 返回排序后的参数列表
        return sorted(self._args, key=str)

    def rcall(self, expr):
        # 递归调用参数的 rcall 方法，并返回新的 OR 对象
        return type(self)(*[arg.rcall(expr)
                            for arg in self._args
                            ])

    def __invert__(self):
        # 对 OR 对象取反，即返回包含所有参数取反的 AND 对象
        return AND(*[~arg for arg in self._args])

    def __hash__(self):
        # 计算 OR 对象的哈希值
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(self, other):
        # 判断两个 OR 对象是否相等
        return self.args == other.args

    def __str__(self):
        # 返回 OR 对象的字符串表示形式
        s = '(' + ' | '.join([str(arg) for arg in self.args]) + ')'
        return s

    __repr__ = __str__


class AND:
    """
    A low-level implementation for And
    """
    def __init__(self, *args):
        # 初始化 AND 类，接收多个参数
        self._args = args

    def __invert__(self):
        # 对 AND 对象取反，即返回包含所有参数取反的 OR 对象
        return OR(*[~arg for arg in self._args])

    @property
    def args(self):
        # 返回参数列表
        return self._args
    # 返回已排序的参数列表，按照字符串的顺序排序
    def args(self):
        return sorted(self._args, key=str)

    # 递归调用每个参数的 rcall 方法，并使用结果创建一个新的对象
    def rcall(self, expr):
        return type(self)(*[arg.rcall(expr)
                            for arg in self._args
                            ])

    # 计算对象的哈希值，基于对象类型名称和参数元组的哈希值
    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.args))

    # 检查两个对象是否相等，比较它们的参数列表是否相同
    def __eq__(self, other):
        return self.args == other.args

    # 返回对象的字符串表示，形如 (arg1 & arg2 & ...)
    def __str__(self):
        s = '(' + ' & '.join([str(arg) for arg in self.args]) + ')'
        return s

    # 将 __repr__ 方法设置为 __str__ 方法的别名，返回对象的字符串表示
    __repr__ = __str__
    """
    Generates the Negation Normal Form of any boolean expression in terms
    of AND, OR, and Literal objects.

    Examples
    ========

    >>> from sympy import Q, Eq
    >>> from sympy.assumptions.cnf import to_NNF
    >>> from sympy.abc import x, y
    >>> expr = Q.even(x) & ~Q.positive(x)
    >>> to_NNF(expr)
    (Literal(Q.even(x), False) & Literal(Q.positive(x), True))

    Supported boolean objects are converted to corresponding predicates.

    >>> to_NNF(Eq(x, y))
    Literal(Q.eq(x, y), False)

    If ``composite_map`` argument is given, ``to_NNF`` decomposes the
    specified predicate into a combination of primitive predicates.

    >>> cmap = {Q.nonpositive: Q.negative | Q.zero}
    >>> to_NNF(Q.nonpositive, cmap)
    (Literal(Q.negative, False) | Literal(Q.zero, False))
    >>> to_NNF(Q.nonpositive(x), cmap)
    (Literal(Q.negative(x), False) | Literal(Q.zero(x), False))
    """
    from sympy.assumptions.ask import Q  # 导入 Q 对象用于表示谓词

    if composite_map is None:
        composite_map = {}  # 如果 composite_map 参数为 None，则设为空字典

    # 定义二元关系谓词到 Q 对象的映射
    binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}

    # 如果表达式的类型在 binrelpreds 中，则将其转换为对应的 Q 对象谓词
    if type(expr) in binrelpreds:
        pred = binrelpreds[type(expr)]
        expr = pred(*expr.args)

    # 如果表达式是 Not 类型，则对其参数求 NN 然后取反
    if isinstance(expr, Not):
        arg = expr.args[0]
        tmp = to_NNF(arg, composite_map)  # 策略：对 expr 的 NN 取反
        return ~tmp

    # 如果表达式是 Or 类型，则对其参数分别求 NN
    if isinstance(expr, Or):
        return OR(*[to_NNF(x, composite_map) for x in Or.make_args(expr)])

    # 如果表达式是 And 类型，则对其参数分别求 NN
    if isinstance(expr, And):
        return AND(*[to_NNF(x, composite_map) for x in And.make_args(expr)])

    # 如果表达式是 Nand 类型，则对其参数先求 NN 再取反
    if isinstance(expr, Nand):
        tmp = AND(*[to_NNF(x, composite_map) for x in expr.args])
        return ~tmp

    # 如果表达式是 Nor 类型，则对其参数先求 NN 再取反
    if isinstance(expr, Nor):
        tmp = OR(*[to_NNF(x, composite_map) for x in expr.args])
        return ~tmp

    # 如果表达式是 Xor 类型，则生成其对应的 CNF 表达式
    if isinstance(expr, Xor):
        cnfs = []
        for i in range(0, len(expr.args) + 1, 2):
            for neg in combinations(expr.args, i):
                clause = [~to_NNF(s, composite_map) if s in neg else to_NNF(s, composite_map)
                          for s in expr.args]
                cnfs.append(OR(*clause))
        return AND(*cnfs)

    # 如果表达式是 Xnor 类型，则生成其对应的 CNF 表达式再取反
    if isinstance(expr, Xnor):
        cnfs = []
        for i in range(0, len(expr.args) + 1, 2):
            for neg in combinations(expr.args, i):
                clause = [~to_NNF(s, composite_map) if s in neg else to_NNF(s, composite_map)
                          for s in expr.args]
                cnfs.append(OR(*clause))
        return ~AND(*cnfs)

    # 如果表达式是 Implies 类型，则生成其对应的 CNF 表达式
    if isinstance(expr, Implies):
        L, R = to_NNF(expr.args[0], composite_map), to_NNF(expr.args[1], composite_map)
        return OR(~L, R)

    # 如果表达式是 Equivalent 类型，则生成其对应的 CNF 表达式
    if isinstance(expr, Equivalent):
        cnfs = []
        for a, b in zip_longest(expr.args, expr.args[1:], fillvalue=expr.args[0]):
            a = to_NNF(a, composite_map)
            b = to_NNF(b, composite_map)
            cnfs.append(OR(~a, b))
        return AND(*cnfs)
    # 如果表达式是一个ITE表达式（If-Then-Else条件表达式）
    if isinstance(expr, ITE):
        # 递归地将条件表达式的三个分支转换为NNF形式
        L = to_NNF(expr.args[0], composite_map)
        M = to_NNF(expr.args[1], composite_map)
        R = to_NNF(expr.args[2], composite_map)
        # 返回经过转换后的NNF形式：(L' ∨ M') ∧ (¬L' ∨ R')
        return AND(OR(~L, M), OR(L, R))

    # 如果表达式是一个AppliedPredicate（已应用的谓词）
    if isinstance(expr, AppliedPredicate):
        # 提取谓词和参数
        pred, args = expr.function, expr.arguments
        # 在复合映射中查找谓词的新定义
        newpred = composite_map.get(pred, None)
        # 如果找到了新定义的谓词，递归地将其应用到参数上并转换为NNF形式
        if newpred is not None:
            return to_NNF(newpred.rcall(*args), composite_map)

    # 如果表达式是一个普通的Predicate（谓词）
    if isinstance(expr, Predicate):
        # 在复合映射中查找谓词的新定义
        newpred = composite_map.get(expr, None)
        # 如果找到了新定义的谓词，递归地将其转换为NNF形式
        if newpred is not None:
            return to_NNF(newpred, composite_map)

    # 如果表达式是一个Literal（字面量），直接返回它本身
    return Literal(expr)
def distribute_AND_over_OR(expr):
    """
    Distributes AND over OR in the NNF expression.
    Returns the result( Conjunctive Normal Form of expression)
    as a CNF object.
    """
    # 如果表达式不是 AND 或者 OR 类型，将其作为单一元素集合添加到临时集合中，返回其 CNF 表示
    if not isinstance(expr, (AND, OR)):
        tmp = set()
        tmp.add(frozenset((expr,)))
        return CNF(tmp)

    # 如果表达式是 OR 类型，递归地对其每个参数应用分配 AND 到 OR 的规则
    if isinstance(expr, OR):
        return CNF.all_or(*[distribute_AND_over_OR(arg)
                            for arg in expr._args])

    # 如果表达式是 AND 类型，递归地对其每个参数应用分配 AND 到 OR 的规则
    if isinstance(expr, AND):
        return CNF.all_and(*[distribute_AND_over_OR(arg)
                             for arg in expr._args])


class CNF:
    """
    Class to represent CNF of a Boolean expression.
    Consists of set of clauses, which themselves are stored as
    frozenset of Literal objects.

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.abc import x
    >>> cnf = CNF.from_prop(Q.real(x) & ~Q.zero(x))
    >>> cnf.clauses
    {frozenset({Literal(Q.zero(x), True)}),
    frozenset({Literal(Q.negative(x), False),
    Literal(Q.positive(x), False), Literal(Q.zero(x), False)})}
    """
    def __init__(self, clauses=None):
        # 初始化 CNF 实例，如果没有传入子句集合，则为空集合
        if not clauses:
            clauses = set()
        self.clauses = clauses

    def add(self, prop):
        # 将给定的逻辑表达式转换为 CNF，并将其子句添加到当前 CNF 实例中
        clauses = CNF.to_CNF(prop).clauses
        self.add_clauses(clauses)

    def __str__(self):
        # 将 CNF 实例转换为字符串形式，每个子句用 | 连接，整体用 & 连接
        s = ' & '.join(
            ['(' + ' | '.join([str(lit) for lit in clause]) +')'
            for clause in self.clauses]
        )
        return s

    def extend(self, props):
        # 将给定的逻辑表达式列表逐个添加到当前 CNF 实例中
        for p in props:
            self.add(p)
        return self

    def copy(self):
        # 返回当前 CNF 实例的深拷贝副本
        return CNF(set(self.clauses))

    def add_clauses(self, clauses):
        # 将给定的子句集合添加到当前 CNF 实例的子句集合中
        self.clauses |= clauses

    @classmethod
    def from_prop(cls, prop):
        # 根据给定的逻辑表达式创建一个新的 CNF 实例，并将其转换为 CNF 格式后返回
        res = cls()
        res.add(prop)
        return res

    def __iand__(self, other):
        # 将另一个 CNF 实例的子句集合并入当前 CNF 实例的子句集合中，并返回当前实例
        self.add_clauses(other.clauses)
        return self

    def all_predicates(self):
        # 返回当前 CNF 实例中所有出现过的逻辑命题的集合
        predicates = set()
        for c in self.clauses:
            predicates |= {arg.lit for arg in c}
        return predicates

    def _or(self, cnf):
        # 将当前 CNF 实例和另一个 CNF 实例的子句进行 OR 运算，并返回新的 CNF 实例
        clauses = set()
        for a, b in product(self.clauses, cnf.clauses):
            tmp = set(a)
            tmp.update(b)
            clauses.add(frozenset(tmp))
        return CNF(clauses)

    def _and(self, cnf):
        # 将当前 CNF 实例和另一个 CNF 实例的子句进行 AND 运算，并返回新的 CNF 实例
        clauses = self.clauses.union(cnf.clauses)
        return CNF(clauses)

    def _not(self):
        # 对当前 CNF 实例的子句集合中的每个子句取反，并返回新的 CNF 实例
        clss = list(self.clauses)
        ll = {frozenset((~x,)) for x in clss[-1]}
        ll = CNF(ll)

        for rest in clss[:-1]:
            p = {frozenset((~x,)) for x in rest}
            ll = ll._or(CNF(p))
        return ll

    def rcall(self, expr):
        # 对当前 CNF 实例的每个子句应用递归地分配 AND 到 OR 的规则，并返回最终的 CNF 结果
        clause_list = []
        for clause in self.clauses:
            lits = [arg.rcall(expr) for arg in clause]
            clause_list.append(OR(*lits))
        expr = AND(*clause_list)
        return distribute_AND_over_OR(expr)

    @classmethod
    # 定义一个类方法 all_or(cls, *cnfs)，将多个合取范式（CNF）对象进行逻辑或运算，返回新的合取范式对象
    def all_or(cls, *cnfs):
        # 复制第一个合取范式对象作为初始值
        b = cnfs[0].copy()
        # 遍历剩余的合取范式对象，逐个与初始值进行逻辑或运算
        for rest in cnfs[1:]:
            b = b._or(rest)
        # 返回运算后的合取范式对象
        return b

    # 定义一个类方法 all_and(cls, *cnfs)，将多个合取范式（CNF）对象进行逻辑与运算，返回新的合取范式对象
    @classmethod
    def all_and(cls, *cnfs):
        # 复制第一个合取范式对象作为初始值
        b = cnfs[0].copy()
        # 遍历剩余的合取范式对象，逐个与初始值进行逻辑与运算
        for rest in cnfs[1:]:
            b = b._and(rest)
        # 返回运算后的合取范式对象
        return b

    # 定义一个类方法 to_CNF(cls, expr)，将逻辑表达式转换为合取范式（CNF）
    @classmethod
    def to_CNF(cls, expr):
        # 导入必要的函数
        from sympy.assumptions.facts import get_composite_predicates
        # 将表达式转换为否定范式（NNF）
        expr = to_NNF(expr, get_composite_predicates())
        # 将表达式中的合取分配到析取上，转换为合取范式（CNF）
        expr = distribute_AND_over_OR(expr)
        # 返回转换后的合取范式表达式
        return expr

    # 定义一个类方法 CNF_to_cnf(cls, cnf)，将合取范式对象转换为 SymPy 的布尔表达式，并保留其形式
    @classmethod
    def CNF_to_cnf(cls, cnf):
        """
        将合取范式对象转换为 SymPy 的布尔表达式，并保留其形式。
        """
        # 定义一个函数，用于移除文字（literal）的否定
        def remove_literal(arg):
            return Not(arg.lit) if arg.is_Not else arg.lit

        # 对于每个合取范式中的子句，将其内部的文字转换为 SymPy 的布尔表达式，并进行析取运算后再进行合取运算
        return And(*(Or(*(remove_literal(arg) for arg in clause)) for clause in cnf.clauses))
    """
    CNF表达式编码的类。
    """
    # 初始化方法，接受data和encoding作为可选参数，如果都未提供，则使用空列表和空字典
    def __init__(self, data=None, encoding=None):
        if not data and not encoding:
            data = []  # 如果data和encoding都未提供，则初始化data为空列表
            encoding = {}  # 如果data和encoding都未提供，则初始化encoding为空字典
        self.data = data  # 初始化实例变量data，用于存储CNF表达式的数据
        self.encoding = encoding  # 初始化实例变量encoding，用于存储符号到整数的编码映射
        self._symbols = list(encoding.keys())  # 初始化实例变量_symbols，存储编码字典的所有键（符号列表）

    # 根据CNF对象设置编码，接受cnf作为参数
    def from_cnf(self, cnf):
        self._symbols = list(cnf.all_predicates())  # 获取CNF对象的所有谓词，存入_symbols列表
        n = len(self._symbols)  # 获取符号数量
        self.encoding = dict(zip(self._symbols, range(1, n + 1)))  # 创建符号到整数的映射编码
        self.data = [self.encode(clause) for clause in cnf.clauses]  # 将CNF子句编码存入data列表

    # 返回_symbols列表，存储所有符号
    @property
    def symbols(self):
        return self._symbols

    # 返回符号对应的整数范围，从1到符号数量加1
    @property
    def variables(self):
        return range(1, len(self._symbols) + 1)

    # 复制当前对象，返回一个新的EncodedCNF对象
    def copy(self):
        new_data = [set(clause) for clause in self.data]  # 复制data列表的每个子句为集合形式
        return EncodedCNF(new_data, dict(self.encoding))  # 返回新的EncodedCNF对象

    # 添加命题prop到当前对象，接受prop作为参数
    def add_prop(self, prop):
        cnf = CNF.from_prop(prop)  # 从命题prop创建CNF对象
        self.add_from_cnf(cnf)  # 调用add_from_cnf方法添加CNF对象的子句到当前对象的data列表中

    # 添加CNF对象的子句到当前对象的data列表，接受cnf作为参数
    def add_from_cnf(self, cnf):
        clauses = [self.encode(clause) for clause in cnf.clauses]  # 对CNF对象的每个子句进行编码
        self.data += clauses  # 将编码后的子句添加到当前对象的data列表末尾

    # 编码参数arg，接受arg作为参数
    def encode_arg(self, arg):
        literal = arg.lit  # 获取参数arg的文字
        value = self.encoding.get(literal, None)  # 获取文字的编码值，如果不存在则为None
        if value is None:
            n = len(self._symbols)  # 获取当前符号的数量
            self._symbols.append(literal)  # 将文字添加到符号列表中
            value = self.encoding[literal] = n + 1  # 为新增的文字分配一个编码值
        if arg.is_Not:
            return -value  # 如果参数arg是否定的，则返回其相反值的负数
        else:
            return value  # 否则返回参数arg的编码值

    # 对CNF子句进行编码，接受clause作为参数
    def encode(self, clause):
        return {self.encode_arg(arg) if not arg.lit == S.false else 0 for arg in clause}
        # 对子句中的每个参数arg进行编码，如果arg不是假，则返回其编码值；否则返回0
```