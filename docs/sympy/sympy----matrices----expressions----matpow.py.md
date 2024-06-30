# `D:\src\scipysrc\sympy\sympy\matrices\expressions\matpow.py`

```
from .matexpr import MatrixExpr
from .special import Identity
from sympy.core import S  # 导入符号 S
from sympy.core.expr import ExprBuilder  # 导入表达式构建器 ExprBuilder
from sympy.core.cache import cacheit  # 导入缓存装饰器 cacheit
from sympy.core.power import Pow  # 导入幂运算 Pow
from sympy.core.sympify import _sympify  # 导入符号化函数 _sympify
from sympy.matrices import MatrixBase  # 导入矩阵基类 MatrixBase
from sympy.matrices.exceptions import NonSquareMatrixError  # 导入非方阵异常类 NonSquareMatrixError

# 定义矩阵幂表达式类 MatPow，继承自 MatrixExpr 类
class MatPow(MatrixExpr):
    # 构造函数，创建一个新的 MatPow 对象
    def __new__(cls, base, exp, evaluate=False, **options):
        # 将 base 符号化为 SymPy 对象
        base = _sympify(base)
        # 如果 base 不是矩阵，则抛出类型错误
        if not base.is_Matrix:
            raise TypeError("MatPow base should be a matrix")

        # 如果 base 不是方阵，则抛出非方阵异常
        if base.is_square is False:
            raise NonSquareMatrixError("Power of non-square matrix %s" % base)

        # 将 exp 符号化为 SymPy 对象
        exp = _sympify(exp)
        # 使用超类的 __new__ 方法创建 MatPow 对象
        obj = super().__new__(cls, base, exp)

        # 如果 evaluate 参数为 True，则进行计算
        if evaluate:
            obj = obj.doit(deep=False)

        return obj

    # 返回矩阵幂的基数（底数）
    @property
    def base(self):
        return self.args[0]

    # 返回矩阵幂的指数
    @property
    def exp(self):
        return self.args[1]

    # 返回矩阵幂的形状（shape）
    @property
    def shape(self):
        return self.base.shape

    # 使用缓存装饰器缓存计算得到的显式矩阵
    @cacheit
    def _get_explicit_matrix(self):
        return self.base.as_explicit()**self.exp

    # 计算矩阵幂的指定元素
    def _entry(self, i, j, **kwargs):
        from sympy.matrices.expressions import MatMul
        # 对矩阵幂对象进行计算
        A = self.doit()
        # 如果计算结果仍然是 MatPow 对象
        if isinstance(A, MatPow):
            # 创建一个显式的 MatMul 对象
            if A.exp.is_Integer and A.exp.is_positive:
                A = MatMul(*[A.base for k in range(A.exp)])
            # 如果形状不是符号化的，则返回显式矩阵的指定元素
            elif not self._is_shape_symbolic():
                return A._get_explicit_matrix()[i, j]
            else:
                # 留下表达式未计算：
                from sympy.matrices.expressions.matexpr import MatrixElement
                return MatrixElement(self, i, j)
        # 返回计算结果中的指定元素
        return A[i, j]

    # 执行矩阵幂对象的计算
    def doit(self, **hints):
        # 如果 hints 中有 'deep' 键，并且其值为 True
        if hints.get('deep', True):
            # 对基数和指数分别进行计算
            base, exp = (arg.doit(**hints) for arg in self.args)
        else:
            # 否则，直接获取基数和指数
            base, exp = self.args

        # 合并所有幂，例如 (A ** 2) ** 3 -> A ** 6
        while isinstance(base, MatPow):
            exp *= base.args[1]
            base = base.args[0]

        # 如果基数是 MatrixBase 的实例，则委托计算
        if isinstance(base, MatrixBase):
            return base ** exp

        # 处理简单情况，使得 MatrixExpr 子类中的 _eval_power() 方法可以忽略它们
        if exp == S.One:
            return base
        if exp == S.Zero:
            return Identity(base.rows)
        if exp == S.NegativeOne:
            from sympy.matrices.expressions import Inverse
            return Inverse(base).doit(**hints)

        # 获取基数对象的 _eval_power() 方法
        eval_power = getattr(base, '_eval_power', None)
        # 如果存在 _eval_power() 方法，则调用它
        if eval_power is not None:
            return eval_power(exp)

        # 返回计算后的 MatPow 对象
        return MatPow(base, exp)

    # 计算矩阵幂对象的转置
    def _eval_transpose(self):
        base, exp = self.args
        return MatPow(base.transpose(), exp)

    # 计算矩阵幂对象的伴随
    def _eval_adjoint(self):
        base, exp = self.args
        return MatPow(base.adjoint(), exp)
    # 计算当前幂对象的共轭
    def _eval_conjugate(self):
        # 分解出底数和指数
        base, exp = self.args
        # 返回底数的共轭作为新的底数，指数不变，返回一个新的幂对象
        return MatPow(base.conjugate(), exp)

    # 计算对变量 x 的导数
    def _eval_derivative(self, x):
        # 调用 Pow 类的导数计算方法
        return Pow._eval_derivative(self, x)

    # 计算对矩阵表达式的导数行
    def _eval_derivative_matrix_lines(self, x):
        # 导入所需的模块和类
        from sympy.tensor.array.expressions.array_expressions import ArrayContraction
        from ...tensor.array.expressions.array_expressions import ArrayTensorProduct
        from .matmul import MatMul
        from .inverse import Inverse
        
        # 获取当前幂对象的指数
        exp = self.exp
        
        # 如果底数为 (1, 1) 形状且指数不含 x，则进行下列操作
        if self.base.shape == (1, 1) and not exp.has(x):
            # 计算底数的导数行
            lr = self.base._eval_derivative_matrix_lines(x)
            # 对每行进行处理
            for i in lr:
                # 构建新的表达式，使用 ArrayContraction 和 ArrayTensorProduct
                subexpr = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                Identity(1),        # 单位矩阵
                                i._lines[0],        # 第一行
                                exp*self.base**(exp-1),  # 底数的指数项
                                i._lines[1],        # 第二行
                                Identity(1),        # 单位矩阵
                            ]
                        ),
                        (0, 3, 4), (5, 7, 8)    # 对应的索引位置
                    ],
                    validator=ArrayContraction._validate
                )
                # 更新指针和行
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 0
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 4
                i._lines = [subexpr]
            return lr
        
        # 根据指数的正负情况进行处理
        if (exp > 0) == True:
            # 对正指数的情况，创建新的矩阵乘积表达式
            newexpr = MatMul.fromiter([self.base for i in range(exp)])
        elif (exp == -1) == True:
            # 对指数为 -1 的情况，返回底数的逆的导数行
            return Inverse(self.base)._eval_derivative_matrix_lines(x)
        elif (exp < 0) == True:
            # 对负指数的情况，创建逆矩阵的乘积表达式
            newexpr = MatMul.fromiter([Inverse(self.base) for i in range(-exp)])
        elif (exp == 0) == True:
            # 对指数为 0 的情况，执行 doit() 方法后再次调用导数行的计算
            return self.doit()._eval_derivative_matrix_lines(x)
        else:
            # 若指数为其他情况，抛出未实现的异常
            raise NotImplementedError("cannot evaluate %s derived by %s" % (self, x))
        
        # 返回新表达式的导数行
        return newexpr._eval_derivative_matrix_lines(x)

    # 计算当前幂对象的逆
    def _eval_inverse(self):
        # 返回底数的负指数作为新的指数
        return MatPow(self.base, -self.exp)
```