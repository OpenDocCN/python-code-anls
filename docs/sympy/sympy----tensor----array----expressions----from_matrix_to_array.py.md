# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\from_matrix_to_array.py`

```
# 导入 Sympy 库中的特定类和函数
from sympy import KroneckerProduct
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.matrices.expressions.hadamard import (HadamardPower, HadamardProduct)
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.transpose import Transpose
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.tensor.array.expressions.array_expressions import \
    ArrayElementwiseApplyFunc, _array_tensor_product, _array_contraction, \
    _array_diagonal, _array_add, _permute_dims, Reshape

# 定义函数，将矩阵表达式转换为数组表达式
def convert_matrix_to_array(expr: Basic) -> Basic:
    # 如果表达式是矩阵乘法
    if isinstance(expr, MatMul):
        args_nonmat = []
        args = []
        # 分离出矩阵和非矩阵参数
        for arg in expr.args:
            if isinstance(arg, MatrixExpr):
                args.append(arg)
            else:
                args_nonmat.append(convert_matrix_to_array(arg))
        # 确定收缩维度的对
        contractions = [(2*i+1, 2*i+2) for i in range(len(args)-1)]
        # 计算标量项的张量积，若没有则为单位元
        scalar = _array_tensor_product(*args_nonmat) if args_nonmat else S.One
        if scalar == 1:
            # 如果标量为1，则计算矩阵参数的张量积
            tprod = _array_tensor_product(
                *[convert_matrix_to_array(arg) for arg in args])
        else:
            # 否则，将标量与矩阵参数的张量积计算
            tprod = _array_tensor_product(
                scalar,
                *[convert_matrix_to_array(arg) for arg in args])
        return _array_contraction(
                tprod,
                *contractions
        )
    # 如果表达式是矩阵加法
    elif isinstance(expr, MatAdd):
        # 对所有参数执行数组加法
        return _array_add(
                *[convert_matrix_to_array(arg) for arg in expr.args]
        )
    # 如果表达式是转置操作
    elif isinstance(expr, Transpose):
        # 对表达式的第一个参数执行维度置换，维度为[1, 0]
        return _permute_dims(
                convert_matrix_to_array(expr.args[0]), [1, 0]
        )
    # 如果表达式是迹运算
    elif isinstance(expr, Trace):
        # 将表达式的参数转换为数组表达式，并执行收缩操作
        inner_expr: MatrixExpr = convert_matrix_to_array(expr.arg) # type: ignore
        return _array_contraction(inner_expr, (0, len(inner_expr.shape) - 1))
    # 如果表达式是乘法操作
    elif isinstance(expr, Mul):
        # 对所有参数执行张量积
        return _array_tensor_product(*[convert_matrix_to_array(i) for i in expr.args])
    # 如果表达式是幂操作
    elif isinstance(expr, Pow):
        base = convert_matrix_to_array(expr.base)
        if (expr.exp > 0) == True:
            # 若指数大于0，则对基础表达式执行多次张量积
            return _array_tensor_product(*[base for i in range(expr.exp)])
        else:
            return expr
    # 如果表达式是矩阵幂操作
    elif isinstance(expr, MatPow):
        base = convert_matrix_to_array(expr.base)
        if expr.exp.is_Integer != True:
            # 若指数不是整数，则生成一个虚拟符号，并应用函数来处理矩阵幂
            b = symbols("b", cls=Dummy)
            return ArrayElementwiseApplyFunc(Lambda(b, b**expr.exp), convert_matrix_to_array(base))
        elif (expr.exp > 0) == True:
            # 若指数大于0，则对基础表达式执行多次矩阵乘法
            return convert_matrix_to_array(MatMul.fromiter(base for i in range(expr.exp)))
        else:
            return expr
    # 如果表达式是 HadamardProduct 类型
    elif isinstance(expr, HadamardProduct):
        # 将表达式中每个参数转换为数组，并进行张量积运算
        tp = _array_tensor_product(*[convert_matrix_to_array(arg) for arg in expr.args])
        # 创建一个对角线索引列表，用于提取对角线元素
        diag = [[2*i for i in range(len(expr.args))], [2*i+1 for i in range(len(expr.args))]]
        # 返回对张量积结果进行对角化处理后的数组
        return _array_diagonal(tp, *diag)
    # 如果表达式是 HadamardPower 类型
    elif isinstance(expr, HadamardPower):
        # 获取基数和指数
        base, exp = expr.args
        # 如果指数是正整数
        if isinstance(exp, Integer) and exp > 0:
            # 构建一个包含多个相同基数的 HadamardProduct，并转换为数组
            return convert_matrix_to_array(HadamardProduct.fromiter(base for i in range(exp)))
        else:
            # 创建一个虚拟变量
            d = Dummy("d")
            # 返回应用元素级函数的结果，该函数对基数中的每个元素进行指数运算
            return ArrayElementwiseApplyFunc(Lambda(d, d**exp), base)
    # 如果表达式是 KroneckerProduct 类型
    elif isinstance(expr, KroneckerProduct):
        # 将表达式中每个参数转换为数组，并进行张量积运算
        kp_args = [convert_matrix_to_array(arg) for arg in expr.args]
        # 创建一个排列索引列表，重新排列张量积结果的维度
        permutation = [2*i for i in range(len(kp_args))] + [2*i + 1 for i in range(len(kp_args))]
        # 返回重新排列维度后的张量积结果，形状保持不变
        return Reshape(_permute_dims(_array_tensor_product(*kp_args), permutation), expr.shape)
    # 如果表达式不是以上任何类型，直接返回表达式本身
    else:
        return expr
```