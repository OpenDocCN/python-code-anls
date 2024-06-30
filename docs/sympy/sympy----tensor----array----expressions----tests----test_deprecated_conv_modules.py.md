# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\tests\test_deprecated_conv_modules.py`

```
from sympy import MatrixSymbol, symbols, Sum
from sympy.tensor.array.expressions import conv_array_to_indexed, from_array_to_indexed, ArrayTensorProduct, \
    ArrayContraction, conv_array_to_matrix, from_array_to_matrix, conv_matrix_to_array, from_matrix_to_array, \
    conv_indexed_to_array, from_indexed_to_array
from sympy.testing.pytest import warns
from sympy.utilities.exceptions import SymPyDeprecationWarning

# 定义一个测试函数，用于测试已弃用的转换模块的结果
def test_deprecated_conv_module_results():

    # 创建两个3x3的矩阵符号对象
    M = MatrixSymbol("M", 3, 3)
    N = MatrixSymbol("N", 3, 3)
    
    # 定义符号变量 i, j, d
    i, j, d = symbols("i j d")

    # 创建张量积表达式 x 和求和表达式 y
    x = ArrayContraction(ArrayTensorProduct(M, N), (1, 2))
    y = Sum(M[i, d]*N[d, j], (d, 0, 2))

    # 使用 SymPyDeprecationWarning 来捕获代码中的弃用警告
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        # 断言将数组表达式 x 转换为索引形式后的结果与期望值相等
        assert conv_array_to_indexed.convert_array_to_indexed(x, [i, j]).dummy_eq(
            from_array_to_indexed.convert_array_to_indexed(x, [i, j]))
        
        # 断言将数组表达式 x 转换为矩阵形式后的结果与期望值相等
        assert conv_array_to_matrix.convert_array_to_matrix(x) == from_array_to_matrix.convert_array_to_matrix(x)
        
        # 断言将矩阵乘积 M*N 转换为数组形式后的结果与期望值相等
        assert conv_matrix_to_array.convert_matrix_to_array(M*N) == from_matrix_to_array.convert_matrix_to_array(M*N)
        
        # 断言将索引形式表达式 y 转换为数组形式后的结果与期望值相等
        assert conv_indexed_to_array.convert_indexed_to_array(y) == from_indexed_to_array.convert_indexed_to_array(y)
```