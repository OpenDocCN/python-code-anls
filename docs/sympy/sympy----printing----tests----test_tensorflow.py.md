# `D:\src\scipysrc\sympy\sympy\printing\tests\test_tensorflow.py`

```
# 导入随机数生成模块
import random
# 导入符号微分相关模块
from sympy.core.function import Derivative
# 导入符号变量定义模块
from sympy.core.symbol import symbols
# 导入张量运算相关模块
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
    PermuteDims, ArrayDiagonal
# 导入符号逻辑比较模块
from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
# 导入外部模块导入函数
from sympy.external import import_module
# 导入数学函数模块
from sympy.functions import \
    Abs, ceiling, exp, floor, sign, sin, asin, sqrt, cos, \
    acos, tan, atan, atan2, cosh, acosh, sinh, asinh, tanh, atanh, \
    re, im, arg, erf, loggamma, log
# 导入矩阵及相关运算模块
from sympy.matrices import Matrix, MatrixBase, eye, randMatrix
from sympy.matrices.expressions import \
    Determinant, HadamardProduct, Inverse, MatrixSymbol, Trace
# 导入打印为 TensorFlow 代码的模块
from sympy.printing.tensorflow import tensorflow_code
# 导入矩阵到数组转换模块
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
# 导入 Sympy 辅助函数模块
from sympy.utilities.lambdify import lambdify
# 导入 Sympy 测试框架中的跳过测试函数
from sympy.testing.pytest import skip
# 导入 Sympy 测试框架中的标记测试失败函数
from sympy.testing.pytest import XFAIL

# 尝试导入 TensorFlow 模块，并将其赋值给变量 tf 和 tensorflow
tf = tensorflow = import_module("tensorflow")

# 如果成功导入 TensorFlow 模块，则设置环境变量以隐藏 TensorFlow 的警告信息
if tensorflow:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义符号矩阵 M、N、P、Q，各为 3x3 大小的矩阵符号
M = MatrixSymbol("M", 3, 3)
N = MatrixSymbol("N", 3, 3)
P = MatrixSymbol("P", 3, 3)
Q = MatrixSymbol("Q", 3, 3)

# 定义符号变量 x、y、z、t
x, y, z, t = symbols("x y z t")

# 如果成功导入 TensorFlow 模块，则创建一个 3x3 的常量张量 m3x3sympy 和对应的 TensorFlow 张量 m3x3
if tf is not None:
    # 创建一个列表，其中包含了 [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    llo = [list(range(i, i+3)) for i in range(0, 9, 3)]
    # 使用 TensorFlow 创建一个常量张量 m3x3
    m3x3 = tf.constant(llo)
    # 使用 Sympy 创建一个 3x3 的矩阵对象 m3x3sympy
    m3x3sympy = Matrix(llo)

# 定义一个函数，用于比较 TensorFlow 中的矩阵表达式和 Sympy 中的表达式
def _compare_tensorflow_matrix(variables, expr, use_float=False):
    # 将 Sympy 的表达式 expr 转换为 TensorFlow 的可执行函数 f
    f = lambdify(variables, expr, 'tensorflow')
    
    # 根据 use_float 的值生成随机矩阵列表 random_matrices
    if not use_float:
        random_matrices = [randMatrix(v.rows, v.cols) for v in variables]
    else:
        random_matrices = [randMatrix(v.rows, v.cols)/100. for v in variables]

    # 创建一个 TensorFlow 的计算图 graph
    graph = tf.Graph()
    r = None
    with graph.as_default():
        # 将随机矩阵列表中的每个矩阵转换为 TensorFlow 对象
        random_variables = [eval(tensorflow_code(i)) for i in random_matrices]
        # 创建一个 TensorFlow 会话 session
        session = tf.compat.v1.Session(graph=graph)
        # 在 TensorFlow 会话中执行函数 f，并获取结果 r
        r = session.run(f(*random_variables))

    # 将表达式 expr 中的变量替换为随机矩阵，计算得到标准答案 e
    e = expr.subs(dict(zip(variables, random_matrices)))
    e = e.doit()
    # 如果 e 是一个矩阵，则将其转换为列表形式
    if e.is_Matrix:
        if not isinstance(e, MatrixBase):
            e = e.as_explicit()
        e = e.tolist()

    # 比较 TensorFlow 计算的结果 r 和标准答案 e
    if not use_float:
        assert (r == e).all()
    else:
        r = [i for row in r for i in row]
        e = [i for row in e for i in row]
        # 对浮点数进行近似比较，确保误差在可接受范围内
        assert all(
            abs(a-b) < 10**-(4-int(log(abs(a), 10))) for a, b in zip(r, e))

# 创建一个自定义逆矩阵测试函数
def _compare_tensorflow_matrix_inverse(variables, expr, use_float=False):
    # 将 Sympy 的表达式 expr 转换为 TensorFlow 的可执行函数 f
    f = lambdify(variables, expr, 'tensorflow')
    
    # 根据 use_float 的值生成适用于逆矩阵测试的随机矩阵列表 random_matrices
    if not use_float:
        random_matrices = [eye(v.rows, v.cols)*4 for v in variables]
    else:
        random_matrices = [eye(v.rows, v.cols)*3.14 for v in variables]

    # 创建一个 TensorFlow 的计算图 graph
    graph = tf.Graph()
    r = None
    with graph.as_default():
        # 将随机矩阵列表中的每个矩阵转换为 TensorFlow 对象
        random_variables = [eval(tensorflow_code(i)) for i in random_matrices]
        # 创建一个 TensorFlow 会话 session
        session = tf.compat.v1.Session(graph=graph)
        # 在 TensorFlow 会话中执行函数 f，并获取结果 r
        r = session.run(f(*random_variables))
    # 对表达式中的变量用随机矩阵进行代换，并计算表达式的值
    e = expr.subs(dict(zip(variables, random_matrices)))
    # 对表达式进行求值
    e = e.doit()
    
    # 如果计算结果是一个矩阵对象
    if e.is_Matrix:
        # 如果结果不是 MatrixBase 类型，则转换为显式矩阵
        if not isinstance(e, MatrixBase):
            e = e.as_explicit()
        # 将矩阵转换为 Python 原生列表形式
        e = e.tolist()

    # 如果不使用浮点数精度比较
    if not use_float:
        # 断言结果矩阵与期望值矩阵完全相等
        assert (r == e).all()
    else:
        # 展开期望值和实际值列表
        r = [i for row in r for i in row]
        e = [i for row in e for i in row]
        # 对每对值进行浮点数比较，根据值的大小动态确定比较精度
        assert all(
            abs(a-b) < 10**-(4-int(log(abs(a), 10))) for a, b in zip(r, e))
# 使用 SymPy 的 lambdify 函数将表达式 expr 转换为 TensorFlow 中的函数 f
def _compare_tensorflow_matrix_scalar(variables, expr):
    f = lambdify(variables, expr, 'tensorflow')
    # 生成随机矩阵列表，每个矩阵大小由 variables 中对应变量的行和列决定，然后将其转换为 TensorFlow 格式
    random_matrices = [
        randMatrix(v.rows, v.cols).evalf() / 100 for v in variables]

    # 创建 TensorFlow 的计算图
    graph = tf.Graph()
    r = None
    with graph.as_default():
        # 将 random_matrices 中的随机矩阵转换为 TensorFlow 对象
        random_variables = [eval(tensorflow_code(i)) for i in random_matrices]
        # 创建 TensorFlow 会话
        session = tf.compat.v1.Session(graph=graph)
        # 执行 TensorFlow 计算图中的函数 f，并获取结果 r
        r = session.run(f(*random_variables))

    # 将表达式中的变量替换为对应的随机矩阵，计算表达式的值 e
    e = expr.subs(dict(zip(variables, random_matrices)))
    e = e.doit()
    # 断言 TensorFlow 计算得到的结果 r 与表达式计算得到的结果 e 的误差小于 10^-6
    assert abs(r - e) < 10**-6


# 使用 SymPy 的 lambdify 函数将表达式 expr 转换为 TensorFlow 中的函数 f
def _compare_tensorflow_scalar(
    variables, expr, rng=lambda: random.randint(0, 10)):
    f = lambdify(variables, expr, 'tensorflow')
    # 生成随机数列表 rvs，用于构造表达式的变量值
    rvs = [rng() for v in variables]

    # 创建 TensorFlow 的计算图
    graph = tf.Graph()
    r = None
    with graph.as_default():
        # 将 rvs 中的随机数转换为 TensorFlow 对象
        tf_rvs = [eval(tensorflow_code(i)) for i in rvs]
        # 创建 TensorFlow 会话
        session = tf.compat.v1.Session(graph=graph)
        # 执行 TensorFlow 计算图中的函数 f，并获取结果 r
        r = session.run(f(*tf_rvs))

    # 将表达式中的变量替换为对应的随机数，计算表达式的值 e
    e = expr.subs(dict(zip(variables, rvs))).evalf().doit()
    # 断言 TensorFlow 计算得到的结果 r 与表达式计算得到的结果 e 的误差小于 10^-6
    assert abs(r - e) < 10**-6


# 使用 SymPy 的 lambdify 函数将表达式 expr 转换为 TensorFlow 中的函数 f
def _compare_tensorflow_relational(
    variables, expr, rng=lambda: random.randint(0, 10)):
    f = lambdify(variables, expr, 'tensorflow')
    # 生成随机数列表 rvs，用于构造表达式的变量值
    rvs = [rng() for v in variables]

    # 创建 TensorFlow 的计算图
    graph = tf.Graph()
    r = None
    with graph.as_default():
        # 将 rvs 中的随机数转换为 TensorFlow 对象
        tf_rvs = [eval(tensorflow_code(i)) for i in rvs]
        # 创建 TensorFlow 会话
        session = tf.compat.v1.Session(graph=graph)
        # 执行 TensorFlow 计算图中的函数 f，并获取结果 r
        r = session.run(f(*tf_rvs))

    # 将表达式中的变量替换为对应的随机数，计算表达式的值 e
    e = expr.subs(dict(zip(variables, rvs))).doit()
    # 断言 TensorFlow 计算得到的结果 r 与表达式计算得到的结果 e 相等
    assert r == e


# 测试 TensorFlow 代码生成函数
def test_tensorflow_printing():
    # 断言 SymPy 生成的眼睛矩阵的 TensorFlow 代码与预期一致
    assert tensorflow_code(eye(3)) == \
        "tensorflow.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])"

    # 定义一个表达式 expr
    expr = Matrix([[x, sin(y)], [exp(z), -t]])
    # 断言 SymPy 生成的表达式的 TensorFlow 代码与预期一致
    assert tensorflow_code(expr) == \
        "tensorflow.Variable(" \
            "[[x, tensorflow.math.sin(y)]," \
            " [tensorflow.math.exp(z), -t]])"

    # 下面是一系列用于测试数学函数在 TensorFlow 中的实现的断言和函数调用
    # 调用 _compare_tensorflow_scalar 函数，比较带有随机数作为参数的 TensorFlow 表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    # 计算 x 的余弦值表达式
    expr = cos(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.cos(x)"
    assert tensorflow_code(expr) == "tensorflow.math.cos(x)"
    # 再次调用 _compare_tensorflow_scalar 函数，比较余弦函数表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    # 计算 x 的反余弦值表达式
    expr = acos(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.acos(x)"
    assert tensorflow_code(expr) == "tensorflow.math.acos(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较反余弦函数表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(0, 0.95))

    # 计算 x 的正弦值表达式
    expr = sin(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.sin(x)"
    assert tensorflow_code(expr) == "tensorflow.math.sin(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较正弦函数表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    # 计算 x 的反正弦值表达式
    expr = asin(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.asin(x)"
    assert tensorflow_code(expr) == "tensorflow.math.asin(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较反正弦函数表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    # 计算 x 的正切值表达式
    expr = tan(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.tan(x)"
    assert tensorflow_code(expr) == "tensorflow.math.tan(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较正切函数表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    # 计算 x 的反正切值表达式
    expr = atan(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.atan(x)"
    assert tensorflow_code(expr) == "tensorflow.math.atan(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较反正切函数表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    # 计算 atan2(y, x) 的值表达式
    expr = atan2(y, x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.atan2(y, x)"
    assert tensorflow_code(expr) == "tensorflow.math.atan2(y, x)"
    # 调用 _compare_tensorflow_scalar 函数，比较 atan2 函数表达式的结果
    _compare_tensorflow_scalar((y, x), expr, rng=lambda: random.random())

    # 计算 x 的双曲余弦值表达式
    expr = cosh(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.cosh(x)"
    assert tensorflow_code(expr) == "tensorflow.math.cosh(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较双曲余弦函数表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    # 计算 x 的反双曲余弦值表达式
    expr = acosh(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.acosh(x)"
    assert tensorflow_code(expr) == "tensorflow.math.acosh(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较反双曲余弦函数表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    # 计算 x 的双曲正弦值表达式
    expr = sinh(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.sinh(x)"
    assert tensorflow_code(expr) == "tensorflow.math.sinh(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较双曲正弦函数表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    # 计算 x 的反双曲正弦值表达式
    expr = asinh(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.asinh(x)"
    assert tensorflow_code(expr) == "tensorflow.math.asinh(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较反双曲正弦函数表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    # 计算 x 的双曲正切值表达式
    expr = tanh(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.tanh(x)"
    assert tensorflow_code(expr) == "tensorflow.math.tanh(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较双曲正切函数表达式的结果
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    # 计算 x 的反双曲正切值表达式
    expr = atanh(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.atanh(x)"
    assert tensorflow_code(expr) == "tensorflow.math.atanh(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较反双曲正切函数表达式的结果
    _compare_tensorflow_scalar(
        (x,), expr, rng=lambda: random.uniform(-.5, .5))

    # 计算 x 的误差函数值表达式
    expr = erf(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.erf(x)"
    assert tensorflow_code(expr) == "tensorflow.math.erf(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较误差函数表达式的结果
    _compare_tensorflow_scalar(
        (x,), expr, rng=lambda: random.random())

    # 计算 x 的对数伽玛函数值表达式
    expr = loggamma(x)
    # 断言表达式的 TensorFlow 表示等于字符串 "tensorflow.math.lgamma(x)"
    assert tensorflow_code(expr) == "tensorflow.math.lgamma(x)"
    # 调用 _compare_tensorflow_scalar 函数，比较对数伽玛函数表达式的结果
    _compare_tensorflow_scalar(
        (x,), expr, rng=lambda: random.random())
# 测试 TensorFlow 中的复杂运算函数是否正确生成代码
def test_tensorflow_complexes():
    # 断言实部函数生成的 TensorFlow 代码是否正确
    assert tensorflow_code(re(x)) == "tensorflow.math.real(x)"
    # 断言虚部函数生成的 TensorFlow 代码是否正确
    assert tensorflow_code(im(x)) == "tensorflow.math.imag(x)"
    # 断言幅角函数生成的 TensorFlow 代码是否正确
    assert tensorflow_code(arg(x)) == "tensorflow.math.angle(x)"


# 测试 TensorFlow 中的关系运算是否正确生成代码
def test_tensorflow_relational():
    # 如果 TensorFlow 没有安装，则跳过测试
    if not tf:
        skip("TensorFlow not installed")

    # 创建一个相等表达式
    expr = Eq(x, y)
    # 断言相等表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.math.equal(x, y)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_relational((x, y), expr)

    # 创建一个不等表达式
    expr = Ne(x, y)
    # 断言不等表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.math.not_equal(x, y)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_relational((x, y), expr)

    # 创建一个大于等于表达式
    expr = Ge(x, y)
    # 断言大于等于表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.math.greater_equal(x, y)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_relational((x, y), expr)

    # 创建一个大于表达式
    expr = Gt(x, y)
    # 断言大于表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.math.greater(x, y)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_relational((x, y), expr)

    # 创建一个小于等于表达式
    expr = Le(x, y)
    # 断言小于等于表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.math.less_equal(x, y)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_relational((x, y), expr)

    # 创建一个小于表达式
    expr = Lt(x, y)
    # 断言小于表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.math.less(x, y)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_relational((x, y), expr)


# 这个（随机）测试标记为 XFAIL，因为它偶尔会失败
# 参见 https://github.com/sympy/sympy/issues/18469
@XFAIL
# 测试 TensorFlow 中的矩阵运算是否正确生成代码
def test_tensorflow_matrices():
    # 如果 TensorFlow 没有安装，则跳过测试
    if not tf:
        skip("TensorFlow not installed")

    # 创建一个矩阵表达式
    expr = M
    # 断言矩阵表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "M"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_matrix((M,), expr)

    # 创建一个矩阵加法表达式
    expr = M + N
    # 断言矩阵加法表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.math.add(M, N)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_matrix((M, N), expr)

    # 创建一个矩阵乘法表达式
    expr = M * N
    # 断言矩阵乘法表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.linalg.matmul(M, N)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_matrix((M, N), expr)

    # 创建一个哈达玛积表达式
    expr = HadamardProduct(M, N)
    # 断言哈达玛积表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.math.multiply(M, N)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_matrix((M, N), expr)

    # 创建一个连续矩阵乘法表达式
    expr = M * N * P * Q
    # 断言连续矩阵乘法表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == \
        "tensorflow.linalg.matmul(" \
            "tensorflow.linalg.matmul(" \
                "tensorflow.linalg.matmul(M, N), P), Q)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_matrix((M, N, P, Q), expr)

    # 创建一个矩阵的立方表达式
    expr = M**3
    # 断言矩阵的立方表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == \
        "tensorflow.linalg.matmul(tensorflow.linalg.matmul(M, M), M)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_matrix((M,), expr)

    # 创建一个矩阵的迹表达式
    expr = Trace(M)
    # 断言矩阵的迹表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.linalg.trace(M)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_matrix((M,), expr)

    # 创建一个矩阵的行列式表达式
    expr = Determinant(M)
    # 断言矩阵的行列式表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.linalg.det(M)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_matrix_scalar((M,), expr)

    # 创建一个矩阵的逆表达式
    expr = Inverse(M)
    # 断言矩阵的逆表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr) == "tensorflow.linalg.inv(M)"
    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_matrix_inverse((M,), expr, use_float=True)

    # 创建一个矩阵的转置表达式
    expr = M.T
    # 根据不同版本的 TensorFlow 断言矩阵的转置表达式生成的 TensorFlow 代码是否正确
    assert tensorflow_code(expr, tensorflow_version='1.14') == \
        "tensorflow.linalg.matrix_transpose(M)"
    assert tensorflow_code(expr, tensorflow_version='1.13') == \
        "tensorflow.matrix_transpose(M)"

    # 比较生成的 TensorFlow 代码和预期结果
    _compare_tensorflow_matrix((M,), expr)
# 定义一个测试函数，用于测试使用 TensorFlow 进行代码生成的情况
def test_codegen_einsum():
    # 如果 TensorFlow 模块未安装，则跳过测试
    if not tf:
        skip("TensorFlow not installed")

    # 创建一个新的计算图对象
    graph = tf.Graph()
    # 将当前计算图设为默认图
    with graph.as_default():
        # 在指定计算图中创建一个 TensorFlow 会话
        session = tf.compat.v1.Session(graph=graph)

        # 定义两个符号矩阵 M 和 N，每个矩阵是一个 2x2 的矩阵
        M = MatrixSymbol("M", 2, 2)
        N = MatrixSymbol("N", 2, 2)

        # 将矩阵乘法 M * N
    # 使用默认图形创建 TensorFlow 会话
    with graph.as_default():
        # 创建 TensorFlow v1 兼容的会话对象
        session = tf.compat.v1.Session()
    
        # 创建 2x2 矩阵符号 M, N, P, Q
        M = MatrixSymbol("M", 2, 2)
        N = MatrixSymbol("N", 2, 2)
        P = MatrixSymbol("P", 2, 2)
        Q = MatrixSymbol("Q", 2, 2)
    
        # 创建 TensorFlow 常量矩阵 ma, mb, mc, md
        ma = tf.constant([[1, 2], [3, 4]])
        mb = tf.constant([[1,-2], [-1, 3]])
        mc = tf.constant([[2, 0], [1, 2]])
        md = tf.constant([[1,-1], [4, 7]])
    
        # 创建 ArrayTensorProduct 对象 cg = M ⊗ N
        cg = ArrayTensorProduct(M, N)
        # 断言转换为 TensorFlow 代码是否符合预期
        assert tensorflow_code(cg) == \
            'tensorflow.linalg.einsum("ab,cd", M, N)'
        # 创建 TensorFlow 函数 f，运行并获取结果 y
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        # 使用 TensorFlow 的 einsum 计算结果 c
        c = session.run(tf.einsum("ij,kl", ma, mb))
        # 断言 y 与 c 的所有元素是否相等
        assert (y == c).all()
    
        # 创建 ArrayAdd 对象 cg = M + N
        cg = ArrayAdd(M, N)
        # 断言转换为 TensorFlow 代码是否符合预期
        assert tensorflow_code(cg) == 'tensorflow.math.add(M, N)'
        # 创建 TensorFlow 函数 f，运行并获取结果 y
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        # 使用 TensorFlow 的加法计算结果 c
        c = session.run(ma + mb)
        # 断言 y 与 c 的所有元素是否相等
        assert (y == c).all()
    
        # 创建 ArrayAdd 对象 cg = M + N + P
        cg = ArrayAdd(M, N, P)
        # 断言转换为 TensorFlow 代码是否符合预期
        assert tensorflow_code(cg) == \
            'tensorflow.math.add(tensorflow.math.add(M, N), P)'
        # 创建 TensorFlow 函数 f，运行并获取结果 y
        f = lambdify((M, N, P), cg, 'tensorflow')
        y = session.run(f(ma, mb, mc))
        # 使用 TensorFlow 的加法计算结果 c
        c = session.run(ma + mb + mc)
        # 断言 y 与 c 的所有元素是否相等
        assert (y == c).all()
    
        # 创建 ArrayAdd 对象 cg = M + N + P + Q
        cg = ArrayAdd(M, N, P, Q)
        # 断言转换为 TensorFlow 代码是否符合预期
        assert tensorflow_code(cg) == \
            'tensorflow.math.add(' \
                'tensorflow.math.add(tensorflow.math.add(M, N), P), Q)'
        # 创建 TensorFlow 函数 f，运行并获取结果 y
        f = lambdify((M, N, P, Q), cg, 'tensorflow')
        y = session.run(f(ma, mb, mc, md))
        # 使用 TensorFlow 的加法计算结果 c
        c = session.run(ma + mb + mc + md)
        # 断言 y 与 c 的所有元素是否相等
        assert (y == c).all()
    
        # 创建 PermuteDims 对象 cg = transpose(M, [1, 0])
        cg = PermuteDims(M, [1, 0])
        # 断言转换为 TensorFlow 代码是否符合预期
        assert tensorflow_code(cg) == 'tensorflow.transpose(M, [1, 0])'
        # 创建 TensorFlow 函数 f，运行并获取结果 y
        f = lambdify((M,), cg, 'tensorflow')
        y = session.run(f(ma))
        # 使用 TensorFlow 的转置函数计算结果 c
        c = session.run(tf.transpose(ma))
        # 断言 y 与 c 的所有元素是否相等
        assert (y == c).all()
    
        # 创建 PermuteDims 对象 cg = transpose(ArrayTensorProduct(M, N), [1, 2, 3, 0])
        cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
        # 断言转换为 TensorFlow 代码是否符合预期
        assert tensorflow_code(cg) == \
            'tensorflow.transpose(' \
                'tensorflow.linalg.einsum("ab,cd", M, N), [1, 2, 3, 0])'
        # 创建 TensorFlow 函数 f，运行并获取结果 y
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        # 使用 TensorFlow 的 einsum 和 transpose 计算结果 c
        c = session.run(tf.transpose(tf.einsum("ab,cd", ma, mb), [1, 2, 3, 0]))
        # 断言 y 与 c 的所有元素是否相等
        assert (y == c).all()
    
        # 创建 ArrayDiagonal 对象 cg = einsum("ab,bc->acb", M, N)
        cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
        # 断言转换为 TensorFlow 代码是否符合预期
        assert tensorflow_code(cg) == \
            'tensorflow.linalg.einsum("ab,bc->acb", M, N)'
        # 创建 TensorFlow 函数 f，运行并获取结果 y
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        # 使用 TensorFlow 的 einsum 计算结果 c
        c = session.run(tf.einsum("ab,bc->acb", ma, mb))
        # 断言 y 与 c 的所有元素是否相等
        assert (y == c).all()
# 定义测试函数，用于测试矩阵元素的打印功能
def test_MatrixElement_printing():
    # 创建矩阵符号 A, B, C，每个矩阵是 1 行 3 列的
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    # 断言：将 A[0, 0] 转换为 TensorFlow 代码后应该是字符串 "A[0, 0]"
    assert tensorflow_code(A[0, 0]) == "A[0, 0]"
    
    # 断言：将 3 * A[0, 0] 转换为 TensorFlow 代码后应该是字符串 "3*A[0, 0]"
    assert tensorflow_code(3 * A[0, 0]) == "3*A[0, 0]"

    # 计算表达式 F，替换 C 为 A - B，然后将 F 转换为 TensorFlow 代码
    F = C[0, 0].subs(C, A - B)
    assert tensorflow_code(F) == "(tensorflow.math.add((-1)*B, A))[0, 0]"


# 定义测试函数，用于测试 TensorFlow 中的导数计算
def test_tensorflow_Derivative():
    # 创建一个表达式 expr，表示 sin(x) 对 x 的导数
    expr = Derivative(sin(x), x)
    # 断言：将 expr 转换为 TensorFlow 代码后应该是字符串
    assert tensorflow_code(expr) == "tensorflow.gradients(tensorflow.math.sin(x), x)[0]"
```