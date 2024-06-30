# `D:\src\scipysrc\sympy\sympy\holonomic\recurrence.py`

```
# 导入必要的模块和类：S 是 SymPy 的单例类，Symbol 和 symbols 是符号类和符号生成函数，sstr 是用于生成字符串表示的打印函数，sympify 是用于将字符串转换为 SymPy 对象的函数
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing import sstr
from sympy.core.sympify import sympify

# 定义一个函数 RecurrenceOperators，接受两个参数：base 是基础多项式环，generator 是生成器，可以是非交换符号或字符串
def RecurrenceOperators(base, generator):
    """
    返回一个递归操作代数及其移位操作符 `Sn`。
    第一个参数应该是代数的基础多项式环，第二个参数必须是一个生成器，可以是非交换符号或字符串。

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.recurrence import RecurrenceOperators
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    """

    # 创建一个 RecurrenceOperatorAlgebra 类的实例 ring，传入 base 和 generator 作为参数
    ring = RecurrenceOperatorAlgebra(base, generator)
    # 返回一个元组，包含代数 ring 和其移位操作符
    return (ring, ring.shift_operator)


class RecurrenceOperatorAlgebra:
    """
    递归操作代数是一组中间的非交换多项式 `Sn` 和基础环 A 中的系数。
    它遵循交换规则：
    Sn * a(n) = a(n + 1) * Sn

    该类表示递归操作代数，并作为递归操作的父环。

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.recurrence import RecurrenceOperators
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    >>> R
    Univariate Recurrence Operator Algebra in intermediate Sn over the base ring
    ZZ[n]

    See Also
    ========

    RecurrenceOperator
    """

    # 初始化函数，接受 base 和 generator 作为参数
    def __init__(self, base, generator):
        # 代数的基础环
        self.base = base
        # 移位操作符 `Sn`
        self.shift_operator = RecurrenceOperator(
            [base.zero, base.one], self)

        # 如果 generator 为 None，则创建一个非交换符号 'Sn'
        if generator is None:
            self.gen_symbol = symbols('Sn', commutative=False)
        else:
            # 如果 generator 是字符串，则创建一个非交换符号，否则直接使用 generator 作为 gen_symbol
            if isinstance(generator, str):
                self.gen_symbol = symbols(generator, commutative=False)
            elif isinstance(generator, Symbol):
                self.gen_symbol = generator

    # 返回对象的字符串表示形式
    def __str__(self):
        string = 'Univariate Recurrence Operator Algebra in intermediate '\
            + sstr(self.gen_symbol) + ' over the base ring ' + \
            (self.base).__str__()
        return string

    # __repr__ 方法与 __str__ 方法相同
    __repr__ = __str__

    # 定义对象相等性比较方法
    def __eq__(self, other):
        if self.base == other.base and self.gen_symbol == other.gen_symbol:
            return True
        else:
            return False


# 定义一个函数 _add_lists，接受两个列表 list1 和 list2 作为参数，返回它们的元素对应相加后的结果列表
def _add_lists(list1, list2):
    if len(list1) <= len(list2):
        sol = [a + b for a, b in zip(list1, list2)] + list2[len(list1):]
    else:
        sol = [a + b for a, b in zip(list1, list2)] + list1[len(list2):]
    return sol


class RecurrenceOperator:
    """
    递归操作由多项式列表定义
    """
    # 设置`
        # 文档说明部分，描述该类在基环和算子父环中的用途
        Explanation
        ===========
    
        # 说明接收一个多项式列表和父环，父环必须是 RecurrenceOperatorAlgebra 的实例
        Takes a list of polynomials for each power of Sn and the
        parent ring which must be an instance of RecurrenceOperatorAlgebra.
    
        # 说明如何使用 Sn 运算符创建一个递归算子，示例代码后面列出
        A Recurrence Operator can be created easily using
        the operator `Sn`. See examples below.
    
        # 示例部分，展示如何使用 RecurrenceOperator 和 RecurrenceOperators
        Examples
        ========
    
        # 导入必要的模块和符号
        >>> from sympy.holonomic.recurrence import RecurrenceOperator, RecurrenceOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols
        >>> n = symbols('n', integer=True)
        # 创建 RecurrenceOperators 对象，指定环为 ZZ.old_poly_ring(n) 和 'Sn'
        >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n),'Sn')
    
        # 创建 RecurrenceOperator 对象，指定多项式列表和父环 R
        >>> RecurrenceOperator([0, 1, n**2], R)
        (1)Sn + (n**2)Sn**2
    
        # Sn 运算符与变量 n 相乘
        >>> Sn*n
        (n + 1)Sn
    
        # 多项式表达式示例
        >>> n*Sn*n + 1 - Sn**2*n
        (1) + (n**2 + n)Sn + (-n - 2)Sn**2
    
        # 另见部分，列出相关模块
        See Also
        ========
    
        DifferentialOperatorAlgebra
        """
    
        # 类属性，指定运算符优先级
        _op_priority = 20
    
        # 初始化方法，接受多项式列表和父环作为参数
        def __init__(self, list_of_poly, parent):
            # 设置父环，必须是 RecurrenceOperatorAlgebra 对象
            self.parent = parent
            # 如果 list_of_poly 是列表，将多项式转换为环元素
            if isinstance(list_of_poly, list):
                for i, j in enumerate(list_of_poly):
                    # 如果多项式项是整数，转换为符号对象
                    if isinstance(j, int):
                        list_of_poly[i] = self.parent.base.from_sympy(S(j))
                    # 否则，如果不是父环的元素类型，转换为符号对象
                    elif not isinstance(j, self.parent.base.dtype):
                        list_of_poly[i] = self.parent.base.from_sympy(j)
    
                # 保存转换后的多项式列表
                self.listofpoly = list_of_poly
    # 定义乘法运算符的重载方法，用于计算两个操作符的乘积并返回一个新的 RecurrenceOperator 实例
    def __mul__(self, other):
        """
        Multiplies two Operators and returns another
        RecurrenceOperator instance using the commutation rule
        Sn * a(n) = a(n + 1) * Sn
        """

        # 获取当前操作符的多项式列表
        listofself = self.listofpoly
        # 获取当前操作符所属的父类对象的基础属性
        base = self.parent.base

        # 如果 `other` 不是 RecurrenceOperator 类型，则根据条件进行类型转换或创建
        if not isinstance(other, RecurrenceOperator):
            if not isinstance(other, self.parent.base.dtype):
                # 如果 `other` 不是当前操作符所属父类的数据类型，则将其转换为合适的类型
                listofother = [self.parent.base.from_sympy(sympify(other))]
            else:
                # 否则，将 `other` 包装为列表形式
                listofother = [other]
        else:
            # 如果 `other` 是 RecurrenceOperator 类型，则直接使用其多项式列表
            listofother = other.listofpoly

        # 定义一个内部函数，用于将多项式 `b` 与多项式列表 `listofother` 相乘
        def _mul_dmp_diffop(b, listofother):
            if isinstance(listofother, list):
                return [i * b for i in listofother]
            return [b * listofother]

        # 计算初始解 `sol`，即第一个多项式与 `listofother` 相乘的结果
        sol = _mul_dmp_diffop(listofself[0], listofother)

        # 定义一个内部函数，用于计算 Sn^i * b 的结果
        def _mul_Sni_b(b):
            sol = [base.zero]

            if isinstance(b, list):
                # 如果 `b` 是列表，则分别计算每个多项式 Sn^i * b 的结果
                for i in b:
                    j = base.to_sympy(i).subs(base.gens[0], base.gens[0] + S.One)
                    sol.append(base.from_sympy(j))
            else:
                # 否则，直接计算 Sn^i * b 的结果
                j = b.subs(base.gens[0], base.gens[0] + S.One)
                sol.append(base.from_sympy(j))

            return sol

        # 迭代计算 Sn^i * b 的结果，并更新 `sol`
        for i in range(1, len(listofself)):
            listofother = _mul_Sni_b(listofother)
            sol = _add_lists(sol, _mul_dmp_diffop(listofself[i], listofother))

        # 返回最终的 RecurrenceOperator 对象，其中多项式列表为 `sol`，父类为 `self.parent`
        return RecurrenceOperator(sol, self.parent)

    # 定义右乘法运算符的重载方法
    def __rmul__(self, other):
        # 如果 `other` 不是 RecurrenceOperator 类型，则根据条件进行类型转换或创建
        if not isinstance(other, RecurrenceOperator):
            if isinstance(other, int):
                other = S(other)
            if not isinstance(other, self.parent.base.dtype):
                other = (self.parent.base).from_sympy(other)

            # 计算 `other` 与当前操作符每个多项式的乘积结果
            sol = [other * j for j in self.listofpoly]
            return RecurrenceOperator(sol, self.parent)

    # 定义加法运算符的重载方法
    def __add__(self, other):
        # 如果 `other` 是 RecurrenceOperator 类型，则将当前操作符的多项式列表与 `other` 的多项式列表相加
        if isinstance(other, RecurrenceOperator):
            sol = _add_lists(self.listofpoly, other.listofpoly)
            return RecurrenceOperator(sol, self.parent)
        else:
            # 否则，根据条件将 `other` 转换为适当的数据类型，并与当前操作符的多项式列表相加
            if isinstance(other, int):
                other = S(other)
            list_self = self.listofpoly
            if not isinstance(other, self.parent.base.dtype):
                list_other = [((self.parent).base).from_sympy(other)]
            else:
                list_other = [other]
            sol = [list_self[0] + list_other[0]] + list_self[1:]

            return RecurrenceOperator(sol, self.parent)

    # 定义反向加法运算符，与加法运算符功能相同
    __radd__ = __add__

    # 定义减法运算符的重载方法，返回当前操作符与 `other` 相反数的和
    def __sub__(self, other):
        return self + (-1) * other

    # 定义反向减法运算符的重载方法，返回 `other` 与当前操作符的相反数的和
    def __rsub__(self, other):
        return (-1) * self + other
    # 定义一个特殊方法 `__pow__`，处理幂运算
    def __pow__(self, n):
        # 如果幂指数 n 等于 1，直接返回自身
        if n == 1:
            return self
        # 创建一个新的递归操作符对象 `result`，初始包含单位元素
        result = RecurrenceOperator([self.parent.base.one], self.parent)
        # 如果幂指数 n 等于 0，直接返回 `result`
        if n == 0:
            return result
        # 判断当前递归操作符 `self` 是否等于 `Sn`
        if self.listofpoly == self.parent.shift_operator.listofpoly:
            # 构建一个特殊的解 `sol`，用于 Sn 的特定情况
            sol = [self.parent.base.zero] * n + [self.parent.base.one]
            return RecurrenceOperator(sol, self.parent)
        # 初始化变量 `x` 为当前递归操作符 `self`
        x = self
        # 进行快速幂运算
        while True:
            # 如果当前幂指数 n 是奇数，则累乘 `result` 和 `x`
            if n % 2:
                result *= x
            # 将幂指数 n 右移一位
            n >>= 1
            # 如果幂指数 n 变成 0，则退出循环
            if not n:
                break
            # `x` 自乘，更新为 x^2
            x *= x
        # 返回最终结果 `result`
        return result

    # 定义一个特殊方法 `__str__`，返回递归操作符的字符串表示
    def __str__(self):
        # 获取递归操作符的多项式列表
        listofpoly = self.listofpoly
        # 初始化打印字符串
        print_str = ''
        # 遍历多项式列表
        for i, j in enumerate(listofpoly):
            # 如果当前系数 j 等于零，则跳过
            if j == self.parent.base.zero:
                continue
            # 将系数 j 转换为 SymPy 格式
            j = self.parent.base.to_sympy(j)
            # 如果是第一项
            if i == 0:
                print_str += '(' + sstr(j) + ')'
                continue
            # 如果已经有打印字符串，则添加加号分隔符
            if print_str:
                print_str += ' + '
            # 如果是第一次出现 i 等于 1
            if i == 1:
                print_str += '(' + sstr(j) + ')Sn'
                continue
            # 添加带幂次的字符串表示
            print_str += '(' + sstr(j) + ')' + 'Sn**' + sstr(i)
        # 返回最终打印字符串
        return print_str

    # `__repr__` 方法与 `__str__` 方法相同
    __repr__ = __str__

    # 定义一个特殊方法 `__eq__`，用于判断两个递归操作符是否相等
    def __eq__(self, other):
        # 如果 `other` 是递归操作符对象
        if isinstance(other, RecurrenceOperator):
            # 判断多项式列表和父级对象是否相等
            if self.listofpoly == other.listofpoly and self.parent == other.parent:
                return True
            else:
                return False
        # 如果 `other` 是单个系数
        return self.listofpoly[0] == other and \
            all(i is self.parent.base.zero for i in self.listofpoly[1:])
# 定义一个HolonomicSequence类，表示满足具有多项式系数的线性齐次递推关系的序列。
class HolonomicSequence:
    """
    A Holonomic Sequence is a type of sequence satisfying a linear homogeneous
    recurrence relation with Polynomial coefficients. Alternatively, A sequence
    is Holonomic if and only if its generating function is a Holonomic Function.
    """

    # 初始化方法，接受一个递推关系和一个初始条件列表u0。
    def __init__(self, recurrence, u0=[]):
        # 将递推关系保存在实例变量中
        self.recurrence = recurrence
        # 如果初始条件u0不是列表，则将其转换为包含u0的列表
        if not isinstance(u0, list):
            self.u0 = [u0]
        else:
            self.u0 = u0

        # 检查是否存在初始条件，根据初始条件列表u0是否为空来判断
        if len(self.u0) == 0:
            self._have_init_cond = False
        else:
            self._have_init_cond = True
        
        # 获取递推关系的基础元素名称，并保存在实例变量n中
        self.n = recurrence.parent.base.gens[0]

    # 返回对象的字符串表示形式，包括递推关系和初始条件（如果存在）
    def __repr__(self):
        str_sol = 'HolonomicSequence(%s, %s)' % ((self.recurrence).__repr__(), sstr(self.n))
        # 如果没有初始条件，则直接返回递推关系的字符串表示形式
        if not self._have_init_cond:
            return str_sol
        else:
            cond_str = ''
            seq_str = 0
            # 遍历初始条件列表u0，将其表示为字符串形式并添加到返回结果中
            for i in self.u0:
                cond_str += ', u(%s) = %s' % (sstr(seq_str), sstr(i))
                seq_str += 1

            # 将递推关系和初始条件组合成最终的字符串表示形式并返回
            sol = str_sol + cond_str
            return sol

    # 使用__repr__方法作为__str__方法的实现，返回对象的字符串表示形式
    __str__ = __repr__

    # 比较两个HolonomicSequence对象是否相等
    def __eq__(self, other):
        # 检查递推关系和基础元素是否相同
        if self.recurrence != other.recurrence or self.n != other.n:
            return False
        # 如果都有初始条件，则比较初始条件列表是否相同
        if self._have_init_cond and other._have_init_cond:
            return self.u0 == other.u0
        # 如果没有初始条件，且递推关系和基础元素相同，则认为相等
        return True
```