# `D:\src\scipysrc\sympy\sympy\vector\basisdependent.py`

```
# 引入从未来导入的注释，允许使用类型提示的特性
from __future__ import annotations
# 引入类型检查的标记，用于静态类型检查
from typing import TYPE_CHECKING

# 从 sympy.simplify 模块导入 simplify 别名为 simp，trigsimp 别名为 tsimp（类型忽略）
from sympy.simplify import simplify as simp, trigsimp as tsimp  # type: ignore
# 从 sympy.core.decorators 模块导入 call_highest_priority, _sympifyit
from sympy.core.decorators import call_highest_priority, _sympifyit
# 从 sympy.core.assumptions 模块导入 StdFactKB
from sympy.core.assumptions import StdFactKB
# 从 sympy.core.function 模块导入 diff 别名为 df
from sympy.core.function import diff as df
# 从 sympy.integrals.integrals 模块导入 Integral
from sympy.integrals.integrals import Integral
# 从 sympy.polys.polytools 模块导入 factor 别名为 fctr
from sympy.polys.polytools import factor as fctr
# 从 sympy.core 模块导入 S, Add, Mul
from sympy.core import S, Add, Mul
# 从 sympy.core.expr 模块导入 Expr
from sympy.core.expr import Expr

# 如果 TYPE_CHECKING 为 True，则从 sympy.vector.vector 模块导入 BaseVector
if TYPE_CHECKING:
    from sympy.vector.vector import BaseVector

# 定义一个名为 BasisDependent 的类，继承自 Expr 类
class BasisDependent(Expr):
    """
    Super class containing functionality common to vectors and
    dyadics.
    Named so because the representation of these quantities in
    sympy.vector is dependent on the basis they are expressed in.
    """

    # 类变量 zero，指向 BasisDependentZero 类的实例
    zero: BasisDependentZero

    # 装饰器 call_highest_priority 应用于 '__radd__' 方法
    @call_highest_priority('__radd__')
    # 定义 '__add__' 方法，实现加法操作
    def __add__(self, other):
        return self._add_func(self, other)

    # 装饰器 call_highest_priority 应用于 '__add__' 方法
    @call_highest_priority('__add__')
    # 定义 '__radd__' 方法，实现右加法操作
    def __radd__(self, other):
        return self._add_func(other, self)

    # 装饰器 call_highest_priority 应用于 '__rsub__' 方法
    @call_highest_priority('__rsub__')
    # 定义 '__sub__' 方法，实现减法操作
    def __sub__(self, other):
        return self._add_func(self, -other)

    # 装饰器 call_highest_priority 应用于 '__sub__' 方法
    @call_highest_priority('__sub__')
    # 定义 '__rsub__' 方法，实现右减法操作
    def __rsub__(self, other):
        return self._add_func(other, -self)

    # 装饰器 _sympifyit 应用于 '__mul__' 方法，确保参数可以转换为 SymPy 表达式
    @_sympifyit('other', NotImplemented)
    # 装饰器 call_highest_priority 应用于 '__rmul__' 方法
    def __mul__(self, other):
        return self._mul_func(self, other)

    # 装饰器 _sympifyit 应用于 '__rmul__' 方法，确保参数可以转换为 SymPy 表达式
    @_sympifyit('other', NotImplemented)
    # 装饰器 call_highest_priority 应用于 '__mul__' 方法
    def __rmul__(self, other):
        return self._mul_func(other, self)

    # 定义 '__neg__' 方法，实现取负操作
    def __neg__(self):
        return self._mul_func(S.NegativeOne, self)

    # 装饰器 _sympifyit 应用于 '__rtruediv__' 方法，确保参数可以转换为 SymPy 表达式
    @call_highest_priority('__rtruediv__')
    # 定义 '__truediv__' 方法，实现真除操作
    def __truediv__(self, other):
        return self._div_helper(other)

    # 装饰器 call_highest_priority 应用于 '__truediv__' 方法
    def __rtruediv__(self, other):
        return TypeError("Invalid divisor for division")

    # 定义 evalf 方法，实现对该量的数值化求值
    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """
        Implements the SymPy evalf routine for this quantity.

        evalf's documentation
        =====================

        """
        # 设置 evalf 方法的选项参数
        options = {'subs': subs, 'maxn': maxn, 'chop': chop, 'strict': strict,
                   'quad': quad, 'verbose': verbose}
        # 初始化 vec 为 self.zero
        vec = self.zero
        # 遍历 self.components 的键值对
        for k, v in self.components.items():
            # 将每个组件求值后乘以 k，并累加到 vec
            vec += v.evalf(n, **options) * k
        # 返回求得的 vec
        return vec

    # 将 Expr 类的 evalf 方法的文档字符串添加到 evalf 方法上
    evalf.__doc__ += Expr.evalf.__doc__  # type: ignore

    # 设置 n 属性为 evalf 方法的别名
    n = evalf

    # 定义 simplify 方法，实现对该量的简化
    def simplify(self, **kwargs):
        """
        Implements the SymPy simplify routine for this quantity.

        simplify's documentation
        ========================

        """
        # 对每个组件应用 simplify 函数并乘以 k，生成简化后的组件列表
        simp_components = [simp(v, **kwargs) * k for
                           k, v in self.components.items()]
        # 调用 _add_func 方法对简化后的组件进行累加求和
        return self._add_func(*simp_components)

    # 将 simplify 方法的文档字符串添加到 simplify 方法上
    simplify.__doc__ += simp.__doc__  # type: ignore
    def trigsimp(self, **opts):
        """
        Implements the SymPy trigsimp routine, for this quantity.

        trigsimp's documentation
        ========================
        """
        # 对每个分量应用 trigsimp 函数，并乘以相应的系数，得到经过三角简化后的分量列表
        trig_components = [tsimp(v, **opts) * k for
                           k, v in self.components.items()]
        # 将经过三角简化后的分量列表传递给 _add_func 方法进行求和
        return self._add_func(*trig_components)

    # 将 tsimp 函数的文档字符串添加到 trigsimp 方法的文档字符串中
    trigsimp.__doc__ += tsimp.__doc__  # type: ignore

    def _eval_simplify(self, **kwargs):
        # 调用 simplify 方法，并传递所有的关键字参数
        return self.simplify(**kwargs)

    def _eval_trigsimp(self, **opts):
        # 调用 trigsimp 方法，并传递所有的选项参数
        return self.trigsimp(**opts)

    def _eval_derivative(self, wrt):
        # 调用 diff 方法，并传递 wrt 参数
        return self.diff(wrt)

    def _eval_Integral(self, *symbols, **assumptions):
        # 构建积分组件列表，每个组件都是一个积分乘以其系数
        integral_components = [Integral(v, *symbols, **assumptions) * k
                               for k, v in self.components.items()]
        # 将积分组件列表传递给 _add_func 方法进行求和
        return self._add_func(*integral_components)

    def as_numer_denom(self):
        """
        Returns the expression as a tuple wrt the following
        transformation -

        expression -> a/b -> a, b
        """
        # 直接返回对象本身和 SymPy 中的单位元素 S.One 的元组
        return self, S.One

    def factor(self, *args, **kwargs):
        """
        Implements the SymPy factor routine, on the scalar parts
        of a basis-dependent expression.

        factor's documentation
        ========================
        """
        # 对每个分量应用 factor 函数，并乘以相应的系数，得到经过因式分解后的分量列表
        fctr_components = [fctr(v, *args, **kwargs) * k for
                           k, v in self.components.items()]
        # 将经过因式分解后的分量列表传递给 _add_func 方法进行求和
        return self._add_func(*fctr_components)

    # 将 fctr 函数的文档字符串添加到 factor 方法的文档字符串中
    factor.__doc__ += fctr.__doc__  # type: ignore

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        # 直接返回 SymPy 中的单位元素 S.One 和对象本身的元组
        return (S.One, self)

    def as_coeff_add(self, *deps):
        """Efficiently extract the coefficient of a summation."""
        # 返回一个元组，第一个元素为 0，第二个元素为各分量乘以其系数的和
        return 0, tuple(x * self.components[x] for x in self.components)

    def diff(self, *args, **kwargs):
        """
        Implements the SymPy diff routine, for vectors.

        diff's documentation
        ========================
        """
        # 检查参数列表中是否有 BasisDependent 类型的参数，如果有则抛出异常
        for x in args:
            if isinstance(x, BasisDependent):
                raise TypeError("Invalid arg for differentiation")
        # 对每个分量应用 diff 函数，并乘以相应的系数，得到经过求导后的分量列表
        diff_components = [df(v, *args, **kwargs) * k for
                           k, v in self.components.items()]
        # 将经过求导后的分量列表传递给 _add_func 方法进行求和
        return self._add_func(*diff_components)

    # 将 df 函数的文档字符串添加到 diff 方法的文档字符串中
    diff.__doc__ += df.__doc__  # type: ignore

    def doit(self, **hints):
        """Calls .doit() on each term in the Dyadic"""
        # 对每个分量调用 doit 方法，并传递所有的提示参数，然后乘以相应的系数
        doit_components = [self.components[x].doit(**hints) * x
                           for x in self.components]
        # 将经过 doit 处理后的分量列表传递给 _add_func 方法进行求和
        return self._add_func(*doit_components)
# 定义一个类 BasisDependentAdd，继承自 BasisDependent 和 Add 类
class BasisDependentAdd(BasisDependent, Add):
    """
    Denotes sum of basis dependent quantities such that they cannot
    be expressed as base or Mul instances.
    表示基于基础相关量的求和，这些量不能被表示为基础或乘积实例。
    """

    # 重载 __new__ 方法
    def __new__(cls, *args, **options):
        # 初始化空字典 components，用于存储组件及其系数
        components = {}

        # 检查每个参数并同时学习组件
        for arg in args:
            # 如果参数不是 cls._expr_type 类型的实例
            if not isinstance(arg, cls._expr_type):
                # 如果参数是 Mul 类的实例
                if isinstance(arg, Mul):
                    # 将参数转换为乘法表达式的函数应用结果
                    arg = cls._mul_func(*(arg.args))
                # 如果参数是 Add 类的实例
                elif isinstance(arg, Add):
                    # 将参数转换为加法表达式的函数应用结果
                    arg = cls._add_func(*(arg.args))
                else:
                    # 抛出类型错误，说明无法正确解释参数
                    raise TypeError(str(arg) +
                                    " cannot be interpreted correctly")
            
            # 如果参数为零，则忽略
            if arg == cls.zero:
                continue
            
            # 否则，根据参数更新 components 字典
            if hasattr(arg, "components"):
                for x in arg.components:
                    components[x] = components.get(x, 0) + arg.components[x]

        # 复制一份 components 字典的键列表
        temp = list(components.keys())
        # 遍历 temp 列表
        for x in temp:
            # 如果 components[x] 的值为零
            if components[x] == 0:
                # 则从 components 字典中删除该键值对
                del components[x]

        # 处理零向量的情况，如果 components 字典为空
        if len(components) == 0:
            # 返回类的零元素
            return cls.zero

        # 构建对象
        # 根据 components 中的每个键值对，生成新的参数列表 newargs
        newargs = [x * components[x] for x in components]
        # 使用父类的 __new__ 方法创建对象 obj
        obj = super().__new__(cls, *newargs, **options)
        # 如果 obj 是 Mul 类的实例
        if isinstance(obj, Mul):
            # 返回乘法函数应用的结果
            return cls._mul_func(*obj.args)
        # 设置假设为可交换的
        assumptions = {'commutative': True}
        # 初始化对象的 _assumptions 属性
        obj._assumptions = StdFactKB(assumptions)
        # 设置对象的 _components 属性为 components 字典
        obj._components = components
        # 设置对象的 _sys 属性为 components 字典中的第一个键的 _sys 属性
        obj._sys = (list(components.keys()))[0]._sys

        # 返回对象
        return obj


class BasisDependentMul(BasisDependent, Mul):
    """
    Denotes product of base- basis dependent quantity with a scalar.
    表示基于基础相关量与标量的乘积。
    """
    def __new__(cls, *args, **options):
        # 导入必要的向量运算模块
        from sympy.vector import Cross, Dot, Curl, Gradient
        # 计数器，用于统计特定类型参数出现的次数
        count = 0
        # 记录度量数，默认为1
        measure_number = S.One
        # 标记是否存在零向量
        zeroflag = False
        # 保存额外的参数列表
        extra_args = []

        # 确定向量的组成部分并检查参数
        # 同时保证不会对两个向量进行乘法操作
        for arg in args:
            # 如果参数是零向量函数的实例
            if isinstance(arg, cls._zero_func):
                count += 1
                zeroflag = True
            # 如果参数是零
            elif arg == S.Zero:
                zeroflag = True
            # 如果参数是基本函数或乘法函数的实例
            elif isinstance(arg, (cls._base_func, cls._mul_func)):
                count += 1
                expr = arg._base_instance
                measure_number *= arg._measure_number
            # 如果参数是加法函数的实例
            elif isinstance(arg, cls._add_func):
                count += 1
                expr = arg
            # 如果参数是交叉乘积、点乘积、旋度或梯度的实例
            elif isinstance(arg, (Cross, Dot, Curl, Gradient)):
                extra_args.append(arg)
            # 其他情况下，将参数纳入度量数的乘积中
            else:
                measure_number *= arg

        # 确保不兼容的类型没有进行乘法操作
        if count > 1:
            raise ValueError("Invalid multiplication")
        elif count == 0:
            # 如果没有有效参数，返回乘法运算的结果
            return Mul(*args, **options)

        # 处理零向量的情况
        if zeroflag:
            return cls.zero

        # 如果参数中存在VectorAdd实例，则返回相应的VectorAdd实例
        if isinstance(expr, cls._add_func):
            newargs = [cls._mul_func(measure_number, x) for
                       x in expr.args]
            return cls._add_func(*newargs)

        # 使用父类的__new__方法创建对象实例
        obj = super().__new__(cls, measure_number,
                              expr._base_instance,
                              *extra_args,
                              **options)
        
        # 如果创建的对象是加法对象，则返回对应的加法实例
        if isinstance(obj, Add):
            return cls._add_func(*obj.args)
        
        # 设置对象的基础实例和度量数
        obj._base_instance = expr._base_instance
        obj._measure_number = measure_number
        # 设置假设为可交换
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        # 设置对象的组件为基础实例和度量数的映射
        obj._components = {expr._base_instance: measure_number}
        # 设置对象的系统为基础实例的系统
        obj._sys = expr._base_instance._sys

        # 返回创建的对象
        return obj

    def _sympystr(self, printer):
        # 获取度量数的打印字符串
        measure_str = printer._print(self._measure_number)
        # 如果度量数字符串中包含括号、减号或加号，则添加额外的括号
        if ('(' in measure_str or '-' in measure_str or
                '+' in measure_str):
            measure_str = '(' + measure_str + ')'
        # 返回度量数字符串和基础实例的打印字符串的乘积
        return measure_str + '*' + printer._print(self._base_instance)
class BasisDependentZero(BasisDependent):
    """
    Class to denote a zero basis dependent instance.
    """
    # 用于存储向量的组件，字典形式，键为'BaseVector'类型，值为Expr类型
    components: dict['BaseVector', Expr] = {}
    # 用于存储 LaTeX 表示形式的字符串
    _latex_form: str

    def __new__(cls):
        # 调用父类的构造方法创建新的实例对象
        obj = super().__new__(cls)
        # 预先计算零向量的特定哈希值
        # 始终使用相同的哈希值
        obj._hash = (S.Zero, cls).__hash__()
        return obj

    def __hash__(self):
        # 返回预先计算的哈希值
        return self._hash

    @call_highest_priority('__req__')
    def __eq__(self, other):
        # 检查是否与另一个对象相等，使用 __req__ 方法重载 __eq__
        return isinstance(other, self._zero_func)

    __req__ = __eq__

    @call_highest_priority('__radd__')
    def __add__(self, other):
        # 实现右加法运算符重载，处理与表达式类型的加法
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for addition")

    @call_highest_priority('__add__')
    def __radd__(self, other):
        # 实现左加法运算符重载，处理与表达式类型的加法
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for addition")

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        # 实现右减法运算符重载，处理与表达式类型的减法
        if isinstance(other, self._expr_type):
            return -other
        else:
            raise TypeError("Invalid argument types for subtraction")

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        # 实现左减法运算符重载，处理与表达式类型的减法
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for subtraction")

    def __neg__(self):
        # 返回当前零向量的负向量
        return self

    def normalize(self):
        """
        Returns the normalized version of this vector.
        """
        # 返回当前向量的归一化版本，对于零向量来说就是它自身
        return self

    def _sympystr(self, printer):
        # 返回该零向量的 SymPy 字符串表示
        return '0'
```