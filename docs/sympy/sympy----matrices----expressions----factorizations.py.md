# `D:\src\scipysrc\sympy\sympy\matrices\expressions\factorizations.py`

```
from sympy.matrices.expressions import MatrixExpr  # 导入 MatrixExpr 类，用于表示矩阵表达式
from sympy.assumptions.ask import Q  # 导入 Q 对象，用于表达符号逻辑的假设

class Factorization(MatrixExpr):  # 定义 Factorization 类，继承自 MatrixExpr 类
    arg = property(lambda self: self.args[0])  # arg 属性，返回 Factorization 对象的第一个参数
    shape = property(lambda self: self.arg.shape)  # type: ignore，shape 属性，返回 Factorization 对象参数的形状

class LofLU(Factorization):  # 定义 LofLU 类，继承自 Factorization 类
    @property
    def predicates(self):  # predicates 属性，返回元组，指定 Q.lower_triangular 属性
        return (Q.lower_triangular,)

class UofLU(Factorization):  # 定义 UofLU 类，继承自 Factorization 类
    @property
    def predicates(self):  # predicates 属性，返回元组，指定 Q.upper_triangular 属性
        return (Q.upper_triangular,)

class LofCholesky(LofLU): pass  # 定义 LofCholesky 类，继承自 LofLU 类
class UofCholesky(UofLU): pass  # 定义 UofCholesky 类，继承自 UofLU 类

class QofQR(Factorization):  # 定义 QofQR 类，继承自 Factorization 类
    @property
    def predicates(self):  # predicates 属性，返回元组，指定 Q.orthogonal 属性
        return (Q.orthogonal,)

class RofQR(Factorization):  # 定义 RofQR 类，继承自 Factorization 类
    @property
    def predicates(self):  # predicates 属性，返回元组，指定 Q.upper_triangular 属性
        return (Q.upper_triangular,)

class EigenVectors(Factorization):  # 定义 EigenVectors 类，继承自 Factorization 类
    @property
    def predicates(self):  # predicates 属性，返回元组，指定 Q.orthogonal 属性
        return (Q.orthogonal,)

class EigenValues(Factorization):  # 定义 EigenValues 类，继承自 Factorization 类
    @property
    def predicates(self):  # predicates 属性，返回元组，指定 Q.diagonal 属性
        return (Q.diagonal,)

class UofSVD(Factorization):  # 定义 UofSVD 类，继承自 Factorization 类
    @property
    def predicates(self):  # predicates 属性，返回元组，指定 Q.orthogonal 属性
        return (Q.orthogonal,)

class SofSVD(Factorization):  # 定义 SofSVD 类，继承自 Factorization 类
    @property
    def predicates(self):  # predicates 属性，返回元组，指定 Q.diagonal 属性
        return (Q.diagonal,)

class VofSVD(Factorization):  # 定义 VofSVD 类，继承自 Factorization 类
    @property
    def predicates(self):  # predicates 属性，返回元组，指定 Q.orthogonal 属性
        return (Q.orthogonal,)

def lu(expr):  # 定义 lu 函数，接受一个表达式参数 expr
    return LofLU(expr), UofLU(expr)  # 返回 LofLU 和 UofLU 类的实例，分别传入 expr 参数

def qr(expr):  # 定义 qr 函数，接受一个表达式参数 expr
    return QofQR(expr), RofQR(expr)  # 返回 QofQR 和 RofQR 类的实例，分别传入 expr 参数

def eig(expr):  # 定义 eig 函数，接受一个表达式参数 expr
    return EigenValues(expr), EigenVectors(expr)  # 返回 EigenValues 和 EigenVectors 类的实例，分别传入 expr 参数

def svd(expr):  # 定义 svd 函数，接受一个表达式参数 expr
    return UofSVD(expr), SofSVD(expr), VofSVD(expr)  # 返回 UofSVD、SofSVD 和 VofSVD 类的实例，分别传入 expr 参数
```