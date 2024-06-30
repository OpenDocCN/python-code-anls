# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_funcmatrix.py`

```
# 导入所需模块和函数
from sympy.core import symbols, Lambda  # 导入符号和Lambda函数
from sympy.core.sympify import SympifyError  # 导入SympifyError异常
from sympy.functions import KroneckerDelta  # 导入KroneckerDelta函数
from sympy.matrices import Matrix  # 导入Matrix类
from sympy.matrices.expressions import FunctionMatrix, MatrixExpr, Identity  # 导入FunctionMatrix、MatrixExpr和Identity类
from sympy.testing.pytest import raises  # 导入raises函数


# 定义测试函数：测试FunctionMatrix的创建
def test_funcmatrix_creation():
    # 定义符号变量 i, j, k
    i, j, k = symbols('i j k')
    
    # 测试创建2x2的FunctionMatrix，元素为Lambda表达式，应当成功
    assert FunctionMatrix(2, 2, Lambda((i, j), 0))
    
    # 测试创建0x0的FunctionMatrix，元素为Lambda表达式，应当成功
    assert FunctionMatrix(0, 0, Lambda((i, j), 0))

    # 测试创建负数行数的FunctionMatrix，应当抛出 ValueError 异常
    raises(ValueError, lambda: FunctionMatrix(-1, 0, Lambda((i, j), 0)))
    
    # 测试创建浮点数行数的FunctionMatrix，应当抛出 ValueError 异常
    raises(ValueError, lambda: FunctionMatrix(2.0, 0, Lambda((i, j), 0)))
    
    # 测试创建虚数行数的FunctionMatrix，应当抛出 ValueError 异常
    raises(ValueError, lambda: FunctionMatrix(2j, 0, Lambda((i, j), 0)))
    
    # 测试创建负数列数的FunctionMatrix，应当抛出 ValueError 异常
    raises(ValueError, lambda: FunctionMatrix(0, -1, Lambda((i, j), 0)))
    
    # 测试创建浮点数列数的FunctionMatrix，应当抛出 ValueError 异常
    raises(ValueError, lambda: FunctionMatrix(0, 2.0, Lambda((i, j), 0)))
    
    # 测试创建虚数列数的FunctionMatrix，应当抛出 ValueError 异常
    raises(ValueError, lambda: FunctionMatrix(0, 2j, Lambda((i, j), 0)))
    
    # 测试Lambda表达式只有一个参数时创建FunctionMatrix，应当抛出 ValueError 异常
    raises(ValueError, lambda: FunctionMatrix(2, 2, Lambda(i, 0)))
    
    # 测试Lambda表达式为 lambda i, j: 0 时创建FunctionMatrix，应当抛出 SympifyError 异常
    raises(SympifyError, lambda: FunctionMatrix(2, 2, lambda i, j: 0))
    
    # 测试Lambda表达式参数不是元组时创建FunctionMatrix，应当抛出 ValueError 异常
    raises(ValueError, lambda: FunctionMatrix(2, 2, Lambda((i,), 0)))
    
    # 测试Lambda表达式参数个数超过两个时创建FunctionMatrix，应当抛出 ValueError 异常
    raises(ValueError, lambda: FunctionMatrix(2, 2, Lambda((i, j, k), 0)))
    
    # 测试Lambda表达式不是函数形式时创建FunctionMatrix，应当抛出 ValueError 异常
    raises(ValueError, lambda: FunctionMatrix(2, 2, i+j))
    
    # 测试字符串形式的Lambda表达式创建FunctionMatrix，应当与Lambda表达式对象相等
    assert FunctionMatrix(2, 2, "lambda i, j: 0") == \
        FunctionMatrix(2, 2, Lambda((i, j), 0))

    # 测试创建以KroneckerDelta函数为元素的FunctionMatrix
    m = FunctionMatrix(2, 2, KroneckerDelta)
    # 验证该矩阵转化为显式矩阵后与单位矩阵相等
    assert m.as_explicit() == Identity(2).as_explicit()
    # 验证该矩阵的第三个参数与Lambda表达式对象相等
    assert m.args[2].dummy_eq(Lambda((i, j), KroneckerDelta(i, j)))

    # 测试用符号变量 n 创建 n x n 的FunctionMatrix，应当成功
    n = symbols('n')
    assert FunctionMatrix(n, n, Lambda((i, j), 0))
    
    # 测试用非整数符号变量 n 创建 n x n 的FunctionMatrix，应当抛出 ValueError 异常
    n = symbols('n', integer=False)
    raises(ValueError, lambda: FunctionMatrix(n, n, Lambda((i, j), 0)))
    
    # 测试用负数符号变量 n 创建 n x n 的FunctionMatrix，应当抛出 ValueError 异常
    n = symbols('n', negative=True)
    raises(ValueError, lambda: FunctionMatrix(n, n, Lambda((i, j), 0)))


# 定义测试函数：测试FunctionMatrix的基本功能
def test_funcmatrix():
    # 定义符号变量 i, j
    i, j = symbols('i,j')
    
    # 创建一个3x3的FunctionMatrix，元素为Lambda表达式 i - j
    X = FunctionMatrix(3, 3, Lambda((i, j), i - j))
    
    # 验证矩阵中某个元素的值
    assert X[1, 1] == 0
    assert X[1, 2] == -1
    
    # 验证矩阵的形状
    assert X.shape == (3, 3)
    
    # 验证矩阵的行数和列数
    assert X.rows == X.cols == 3
    
    # 将FunctionMatrix转化为普通的SymPy矩阵后，与Lambda表达式创建的矩阵相等
    assert Matrix(X) == Matrix(3, 3, lambda i, j: i - j)
    
    # 验证矩阵是否为MatrixExpr类型
    assert isinstance(X*X + X, MatrixExpr)


# 定义测试函数：测试FunctionMatrix的替换功能
def test_replace_issue():
    # 创建一个3x3的FunctionMatrix，元素为KroneckerDelta函数
    X = FunctionMatrix(3, 3, KroneckerDelta)
    
    # 验证替换函数中返回True时，替换不改变矩阵
    assert X.replace(lambda x: True, lambda x: x) == X
```