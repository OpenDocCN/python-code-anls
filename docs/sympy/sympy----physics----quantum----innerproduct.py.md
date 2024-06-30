# `D:\src\scipysrc\sympy\sympy\physics\quantum\innerproduct.py`

```
"""Symbolic inner product."""

# 导入所需的模块和类
from sympy.core.expr import Expr
from sympy.functions.elementary.complexes import conjugate
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.state import KetBase, BraBase

__all__ = [
    'InnerProduct'
]

# InnerProduct 类继承自 Expr 类，用于表示未计算的态之间的内积
class InnerProduct(Expr):
    """An unevaluated inner product between a Bra and a Ket [1].

    Parameters
    ==========

    bra : BraBase or subclass
        The bra on the left side of the inner product.
    ket : KetBase or subclass
        The ket on the right side of the inner product.

    Examples
    ========

    Create an InnerProduct and check its properties:

        >>> from sympy.physics.quantum import Bra, Ket
        >>> b = Bra('b')
        >>> k = Ket('k')
        >>> ip = b*k
        >>> ip
        <b|k>
        >>> ip.bra
        <b|
        >>> ip.ket
        |k>

    In simple products of kets and bras inner products will be automatically
    identified and created::

        >>> b*k
        <b|k>

    But in more complex expressions, there is ambiguity in whether inner or
    outer products should be created::

        >>> k*b*k*b
        |k><b|*|k>*<b|

    A user can force the creation of a inner products in a complex expression
    by using parentheses to group the bra and ket::

        >>> k*(b*k)*b
        <b|k>*|k>*<b|

    Notice how the inner product <b|k> moved to the left of the expression
    because inner products are commutative complex numbers.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inner_product
    """
    
    # 类属性，表示内积是复数
    is_complex = True

    # 构造方法，验证输入的 bra 和 ket 类型，并创建 InnerProduct 对象
    def __new__(cls, bra, ket):
        if not isinstance(ket, KetBase):
            raise TypeError('KetBase subclass expected, got: %r' % ket)
        if not isinstance(bra, BraBase):
            raise TypeError('BraBase subclass expected, got: %r' % ket)
        obj = Expr.__new__(cls, bra, ket)
        return obj

    # 获取内积对象的 bra 属性
    @property
    def bra(self):
        return self.args[0]

    # 获取内积对象的 ket 属性
    @property
    def ket(self):
        return self.args[1]

    # 对内积对象进行共轭操作
    def _eval_conjugate(self):
        return InnerProduct(Dagger(self.ket), Dagger(self.bra))

    # 使用 SymPy 的打印机将内积对象转换为字符串表示
    def _sympyrepr(self, printer, *args):
        return '%s(%s,%s)' % (self.__class__.__name__,
            printer._print(self.bra, *args), printer._print(self.ket, *args))

    # 使用 SymPy 的打印机将内积对象转换为易读的字符串表示
    def _sympystr(self, printer, *args):
        sbra = printer._print(self.bra)
        sket = printer._print(self.ket)
        return '%s|%s' % (sbra[:-1], sket[1:])
    # 打印状态内容
    bra = self.bra._print_contents_pretty(printer, *args)
    ket = self.ket._print_contents_pretty(printer, *args)
    
    # 打印括号
    height = max(bra.height(), ket.height())  # 计算括号的最大高度
    use_unicode = printer._use_unicode  # 获取打印机是否使用Unicode
    lbracket, _ = self.bra._pretty_brackets(height, use_unicode)  # 打印左括号
    cbracket, rbracket = self.ket._pretty_brackets(height, use_unicode)  # 打印中间和右括号
    
    # 构建内积表达式
    pform = prettyForm(*bra.left(lbracket))
    pform = prettyForm(*pform.right(cbracket))
    pform = prettyForm(*pform.right(ket))
    pform = prettyForm(*pform.right(rbracket))
    
    # 返回美化后的表达式
    return pform

# 生成 LaTeX 格式的输出
def _latex(self, printer, *args):
    bra_label = self.bra._print_contents_latex(printer, *args)  # 打印 LaTeX 格式的 bra 标签
    ket = printer._print(self.ket, *args)  # 打印 ket 对象
    return r'\left\langle %s \right. %s' % (bra_label, ket)

# 执行操作，计算内积或返回自身
def doit(self, **hints):
    try:
        r = self.ket._eval_innerproduct(self.bra, **hints)  # 尝试计算内积
    except NotImplementedError:
        try:
            r = conjugate(
                self.bra.dual._eval_innerproduct(self.ket.dual, **hints)
            )  # 尝试计算对偶内积的共轭
        except NotImplementedError:
            r = None  # 如果都不支持，则结果为 None
    if r is not None:
        return r  # 如果计算得到结果，则返回结果
    return self  # 否则返回自身对象
```