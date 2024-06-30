# `D:\src\scipysrc\sympy\sympy\printing\tests\test_theanocode.py`

```
"""
Important note on tests in this module - the Theano printing functions use a
global cache by default, which means that tests using it will modify global
state and thus not be independent from each other. Instead of using the "cache"
keyword argument each time, this module uses the theano_code_ and
theano_function_ functions defined below which default to using a new, empty
cache instead.
"""

# 导入日志模块
import logging

# 导入 sympy 的外部模块，用于动态导入模块
from sympy.external import import_module
# 导入 sympy 的测试模块
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy

# 设置 Theano 日志记录器，级别为 CRITICAL
theanologger = logging.getLogger('theano.configdefaults')
theanologger.setLevel(logging.CRITICAL)
# 尝试动态导入 Theano 库，若导入成功则将日志级别设置为 WARNING
theano = import_module('theano')
theanologger.setLevel(logging.WARNING)

# 如果成功导入 Theano 库，则继续执行以下代码
if theano:
    # 导入 numpy 库
    import numpy as np
    # 导入 Theano 的标量和张量模块
    ts = theano.scalar
    tt = theano.tensor
    # 创建三个 Theano 标量变量
    xt, yt, zt = [tt.scalar(name, 'floatX') for name in 'xyz']
    # 创建三个 Theano 浮点数类型的张量变量
    Xt, Yt, Zt = [tt.tensor('floatX', (False, False), name=n) for n in 'XYZ']
else:
    # 若未能导入 Theano 库，则将 disabled 设为 True
    disabled = True

# 导入 sympy 库，并从其中导入一些符号和函数
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
        theano_function)

# 默认的矩阵符号集合，用于测试，设置为 4x4 方阵以便进行矩阵乘法和元素级操作
X, Y, Z = [sy.MatrixSymbol(n, 4, 4) for n in 'XYZ']

# 用于测试 AppliedUndef 的符号函数
f_t = sy.Function('f')(t)


def theano_code_(expr, **kwargs):
    """ Wrapper for theano_code that uses a new, empty cache by default. """
    # 默认使用一个新的空缓存，调用 theano_code 函数
    kwargs.setdefault('cache', {})
    # 使用 warns_deprecated_sympy 上下文管理器进行警告处理
    with warns_deprecated_sympy():
        return theano_code(expr, **kwargs)

def theano_function_(inputs, outputs, **kwargs):
    """ Wrapper for theano_function that uses a new, empty cache by default. """
    # 默认使用一个新的空缓存，调用 theano_function 函数
    kwargs.setdefault('cache', {})
    # 使用 warns_deprecated_sympy 上下文管理器进行警告处理
    with warns_deprecated_sympy():
        return theano_function(inputs, outputs, **kwargs)


def fgraph_of(*exprs):
    """ Transform SymPy expressions into Theano Computation.

    Parameters
    ==========
    exprs
        SymPy expressions

    Returns
    =======
    theano.gof.FunctionGraph
    """
    # 将 SymPy 表达式列表转换为 Theano 计算图
    outs = list(map(theano_code_, exprs))
    # 获取计算图的输入
    ins = theano.gof.graph.inputs(outs)
    # 克隆计算图的输入和输出
    ins, outs = theano.gof.graph.clone(ins, outs)
    # 返回 Theano 计算图
    return theano.gof.FunctionGraph(ins, outs)


def theano_simplify(fgraph):
    """ Simplify a Theano Computation.

    Parameters
    ==========
    fgraph : theano.gof.FunctionGraph

    Returns
    =======
    theano.gof.FunctionGraph
    """
    # 获取默认模式，并排除 "fusion" 优化
    mode = theano.compile.get_default_mode().excluding("fusion")
    # 克隆给定的 Theano 计算图
    fgraph = fgraph.clone()
    # 使用模式优化计算图
    mode.optimizer.optimize(fgraph)
    # 返回优化后的 Theano 计算图
    return fgraph


def theq(a, b):
    """ Test two Theano objects for equality.

    Also accepts numeric types and lists/tuples of supported types.

    Note - debugprint() has a bug where it will accept numeric types but does
    not respect the "file" argument and in this case and instead prints the number
    """
    """
    to stdout and returns an empty string. This can lead to tests passing where
    they should fail because any two numbers will always compare as equal. To
    prevent this we treat numbers as a separate case.
    """
    # 定义包含所有数值类型的元组
    numeric_types = (int, float, np.number)
    # 检查 a 是否为数值类型
    a_is_num = isinstance(a, numeric_types)
    # 检查 b 是否为数值类型
    b_is_num = isinstance(b, numeric_types)

    # 如果 a 或者 b 是数值类型，则使用普通的相等比较
    if a_is_num or b_is_num:
        # 如果 a 和 b 中有一个不是数值类型，则返回 False
        if not (a_is_num and b_is_num):
            return False

        # 返回 a 和 b 的相等比较结果
        return a == b

    # 检查 a 是否为序列类型（tuple 或 list）
    a_is_seq = isinstance(a, (tuple, list))
    # 检查 b 是否为序列类型（tuple 或 list）
    b_is_seq = isinstance(b, (tuple, list))

    # 如果 a 或者 b 是序列类型，则逐元素比较
    if a_is_seq or b_is_seq:
        # 如果 a 和 b 中有一个不是序列类型，或者它们的类型不相同，则返回 False
        if not (a_is_seq and b_is_seq) or type(a) != type(b):
            return False

        # 使用递归调用 theq 函数，逐元素比较 a 和 b
        return list(map(theq, a)) == list(map(theq, b))

    # 如果不是数值类型也不是序列类型，则假定由 debugprint() 处理
    # 使用 theano.printing.debugprint() 将 a 和 b 转换为字符串
    astr = theano.printing.debugprint(a, file='str')
    bstr = theano.printing.debugprint(b, file='str')

    # 检查上述提到的 bug
    for argname, argval, argstr in [('a', a, astr), ('b', b, bstr)]:
        if argstr == '':
            # 如果 debugprint() 返回空字符串，则抛出 TypeError 异常
            raise TypeError(
                'theano.printing.debugprint(%s) returned empty string '
                '(%s is instance of %r)'
                % (argname, argname, type(argval))
            )

    # 返回转换后的字符串的相等比较结果
    return astr == bstr
def test_example_symbols():
    """
    Check that the example symbols in this module print to their Theano
    equivalents, as many of the other tests depend on this.
    """
    # 检查示例符号是否正确映射到它们的 Theano 等效物，许多其他测试依赖于此
    assert theq(xt, theano_code_(x))
    # 断言示例符号 xt 和 x 的 Theano 表示是否相等
    assert theq(yt, theano_code_(y))
    # 断言示例符号 yt 和 y 的 Theano 表示是否相等
    assert theq(zt, theano_code_(z))
    # 断言示例符号 zt 和 z 的 Theano 表示是否相等
    assert theq(Xt, theano_code_(X))
    # 断言示例符号 Xt 和 X 的 Theano 表示是否相等
    assert theq(Yt, theano_code_(Y))
    # 断言示例符号 Yt 和 Y 的 Theano 表示是否相等
    assert theq(Zt, theano_code_(Z))
    # 断言示例符号 Zt 和 Z 的 Theano 表示是否相等


def test_Symbol():
    """ Test printing a Symbol to a theano variable. """
    # 测试将符号打印为 Theano 变量
    xx = theano_code_(x)
    # 将符号 x 转换为其对应的 Theano 表示
    assert isinstance(xx, (tt.TensorVariable, ts.ScalarVariable))
    # 断言转换后的 xx 是 Theano 的 TensorVariable 或 ScalarVariable 类型
    assert xx.broadcastable == ()
    # 断言 xx 的 broadcastable 属性为空元组
    assert xx.name == x.name
    # 断言 xx 的名称与 x 的名称相同

    xx2 = theano_code_(x, broadcastables={x: (False,)})
    # 使用 broadcastables 参数将 x 转换为 Theano 表示，设置 broadcastable 为 (False,)
    assert xx2.broadcastable == (False,)
    # 断言 xx2 的 broadcastable 属性为 (False,)
    assert xx2.name == x.name
    # 断言 xx2 的名称与 x 的名称相同


def test_MatrixSymbol():
    """ Test printing a MatrixSymbol to a theano variable. """
    # 测试将 MatrixSymbol 打印为 Theano 变量
    XX = theano_code_(X)
    # 将 MatrixSymbol X 转换为其对应的 Theano 表示
    assert isinstance(XX, tt.TensorVariable)
    # 断言转换后的 XX 是 Theano 的 TensorVariable 类型
    assert XX.broadcastable == (False, False)
    # 断言 XX 的 broadcastable 属性为 (False, False)


@SKIP  # TODO - this is currently not checked but should be implemented
def test_MatrixSymbol_wrong_dims():
    """ Test MatrixSymbol with invalid broadcastable. """
    # 测试具有非法 broadcastable 的 MatrixSymbol
    bcs = [(), (False,), (True,), (True, False), (False, True,), (True, True)]
    # 定义一组可能的 broadcastable 值
    for bc in bcs:
        with raises(ValueError):
            # 使用 raises 检查预期会引发 ValueError 的情况
            theano_code_(X, broadcastables={X: bc})


def test_AppliedUndef():
    """ Test printing AppliedUndef instance, which works similarly to Symbol. """
    # 测试打印 AppliedUndef 实例，其工作方式类似于 Symbol
    ftt = theano_code_(f_t)
    # 将 AppliedUndef 实例 f_t 转换为其对应的 Theano 表示
    assert isinstance(ftt, tt.TensorVariable)
    # 断言转换后的 ftt 是 Theano 的 TensorVariable 类型
    assert ftt.broadcastable == ()
    # 断言 ftt 的 broadcastable 属性为空元组
    assert ftt.name == 'f_t'
    # 断言 ftt 的名称为 'f_t'


def test_add():
    expr = x + y
    # 创建一个表达式 expr，表示 x + y
    comp = theano_code_(expr)
    # 将表达式 expr 转换为其对应的 Theano 表示
    assert comp.owner.op == theano.tensor.add
    # 断言转换后的 comp 的操作为 Theano 的加法操作


def test_trig():
    assert theq(theano_code_(sy.sin(x)), tt.sin(xt))
    # 断言将符号 x 的 sin 函数转换为 Theano 表示后与 xt 的 sin 函数表示是否相等
    assert theq(theano_code_(sy.tan(x)), tt.tan(xt))
    # 断言将符号 x 的 tan 函数转换为 Theano 表示后与 xt 的 tan 函数表示是否相等


def test_many():
    """ Test printing a complex expression with multiple symbols. """
    # 测试将包含多个符号的复杂表达式打印为 Theano 表示
    expr = sy.exp(x**2 + sy.cos(y)) * sy.log(2*z)
    # 创建一个复杂的表达式 expr
    comp = theano_code_(expr)
    # 将表达式 expr 转换为其对应的 Theano 表示
    expected = tt.exp(xt**2 + tt.cos(yt)) * tt.log(2*zt)
    # 创建预期的 Theano 表示
    assert theq(comp, expected)
    # 断言转换后的 comp 与预期的 expected 相等


def test_dtype():
    """ Test specifying specific data types through the dtype argument. """
    # 测试通过 dtype 参数指定特定的数据类型
    for dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']:
        assert theano_code_(x, dtypes={x: dtype}).type.dtype == dtype
        # 断言将符号 x 转换为指定 dtype 后的 Theano 表示的数据类型与 dtype 相等

    # "floatX" type
    assert theano_code_(x, dtypes={x: 'floatX'}).type.dtype in ('float32', 'float64')
    # 断言将符号 x 转换为 'floatX' 后的 Theano 表示的数据类型是 'float32' 或 'float64'

    # Type promotion
    assert theano_code_(x + 1, dtypes={x: 'float32'}).type.dtype == 'float32'
    # 断言将表达式 x + 1 转换为指定 dtype 后的 Theano 表示的数据类型为 'float32'
    assert theano_code_(x + y, dtypes={x: 'float64', y: 'float32'}).type.dtype == 'float64'
    # 断言将表达式 x + y 转换为指定 dtypes 后的 Theano 表示的数据类型为 'float64'


def test_broadcastables():
    """ Test the "broadcastables" argument when printing symbol-like objects. """

    # No restrictions on shape
    for s in [x, f_t]:
        for bc in [(), (False,), (True,), (False, False), (True, False)]:
            assert theano_code_(s, broadcastables={s: bc}).broadcastable == bc
            # 断言将符号 s 转换为指定 broadcastable 后的 Theano 表示的 broadcastable 属性与 bc 相等

    # TODO - matrix broadcasting?


def test_broadcasting():
    pass
    # 待实现的测试，目前未提供具体实现
    """ Test "broadcastable" attribute after applying element-wise binary op. """

    # 定义表达式，将两个变量 x 和 y 相加
    expr = x + y

    # 定义不同的测试用例，每个用例包含三个元组，分别代表 x 的广播性质 bc1、y 的广播性质 bc2 和预期的 expr 的广播性质 bc3
    cases = [
        [(), (), ()],                     # 空元组的情况
        [(False,), (False,), (False,)],   # 单个布尔值 False 的情况
        [(True,), (False,), (False,)],    # 单个布尔值 True 和 False 的情况
        [(False, True), (False, False), (False, False)],   # 布尔值 False 和 True 的情况
        [(True, False), (False, False), (False, False)],   # 布尔值 True 和 False 的情况
    ]

    # 对于每个测试用例，执行 theano_code_ 函数，传入表达式 expr 和指定的广播性质，获取计算结果的广播性质并进行断言比较
    for bc1, bc2, bc3 in cases:
        comp = theano_code_(expr, broadcastables={x: bc1, y: bc2})
        # 断言计算结果的广播性质与预期的 bc3 相同
        assert comp.broadcastable == bc3
def test_MatMul():
    # 创建表达式 X*Y*Z
    expr = X*Y*Z
    # 转换为 theano 代码表示
    expr_t = theano_code_(expr)
    # 断言表达式的操作类型为 tt.Dot
    assert isinstance(expr_t.owner.op, tt.Dot)
    # 断言表达式与预期值 Xt.dot(Yt).dot(Zt) 相等
    assert theq(expr_t, Xt.dot(Yt).dot(Zt))

def test_Transpose():
    # 断言 theano_code_(X.T) 的操作类型为 tt.DimShuffle
    assert isinstance(theano_code_(X.T).owner.op, tt.DimShuffle)

def test_MatAdd():
    # 创建表达式 X+Y+Z
    expr = X+Y+Z
    # 断言 theano_code_(expr) 的操作类型为 tt.Elemwise
    assert isinstance(theano_code_(expr).owner.op, tt.Elemwise)

def test_Rationals():
    # 断言 theano_code_(sy.Integer(2) / 3) 的值等于 tt.true_div(2, 3)
    assert theq(theano_code_(sy.Integer(2) / 3), tt.true_div(2, 3))
    # 断言 theano_code_(S.Half) 的值等于 tt.true_div(1, 2)
    assert theq(theano_code_(S.Half), tt.true_div(1, 2))

def test_Integers():
    # 断言 theano_code_(sy.Integer(3)) 的值等于 3
    assert theano_code_(sy.Integer(3)) == 3

def test_factorial():
    n = sy.Symbol('n')
    # 断言 theano_code_(sy.factorial(n)) 的值

def test_Derivative():
    simp = lambda expr: theano_simplify(fgraph_of(expr))
    # 断言简化后的 theano_code_(sy.Derivative(sy.sin(x), x, evaluate=False)) 等于简化后的 theano.grad(tt.sin(xt), xt)

def test_theano_function_simple():
    """ Test theano_function() with single output. """
    # 创建 theano_function_ 函数 f，输入为 x 和 y，输出为 x+y
    f = theano_function_([x, y], [x+y])
    # 断言 f(2, 3) 等于 5
    assert f(2, 3) == 5

def test_theano_function_multi():
    """ Test theano_function() with multiple outputs. """
    # 创建 theano_function_ 函数 f，输入为 x 和 y，输出为 [x+y, x-y]
    f = theano_function_([x, y], [x+y, x-y])
    # 将 f(2, 3) 的输出分别赋值给 o1 和 o2
    o1, o2 = f(2, 3)
    # 断言 o1 等于 5
    assert o1 == 5
    # 断言 o2 等于 -1
    assert o2 == -1

def test_theano_function_numpy():
    """ Test theano_function() vs Numpy implementation. """
    # 创建 theano_function_ 函数 f，输入为 x 和 y，输出为 [x+y]，数据类型为 'float64'
    f = theano_function_([x, y], [x+y], dim=1,
                         dtypes={x: 'float64', y: 'float64'})
    # 断言 f([1, 2], [3, 4]) 与 numpy 实现结果的 L2 范数小于 1e-9
    assert np.linalg.norm(f([1, 2], [3, 4]) - np.asarray([4, 6])) < 1e-9

    # 创建 theano_function_ 函数 f，输入为 x 和 y，输出为 [x+y]，数据类型为 'float64'
    f = theano_function_([x, y], [x+y], dtypes={x: 'float64', y: 'float64'},
                         dim=1)
    # 创建输入数组 xx 和 yy
    xx = np.arange(3).astype('float64')
    yy = 2*np.arange(3).astype('float64')
    # 断言 f(xx, yy) 与 3*xx 的 L2 范数小于 1e-9
    assert np.linalg.norm(f(xx, yy) - 3*np.arange(3)) < 1e-9

def test_theano_function_matrix():
    m = sy.Matrix([[x, y], [z, x + y + z]])
    expected = np.array([[1.0, 2.0], [3.0, 1.0 + 2.0 + 3.0]])
    # 创建 theano_function_ 函数 f，输入为 x, y, z，输出为 [m]
    f = theano_function_([x, y, z], [m])
    # 断言 f(1.0, 2.0, 3.0) 的结果与 expected 数组非常接近
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    # 创建 theano_function_ 函数 f，输入为 x, y, z，输出为 [m]，标量输出
    f = theano_function_([x, y, z], [m], scalar=True)
    # 断言 f(1.0, 2.0, 3.0) 的结果与 expected 数组非常接近
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    # 创建 theano_function_ 函数 f，输入为 x, y, z，输出为 [m, m]
    f = theano_function_([x, y, z], [m, m])
    # 断言 f(1.0, 2.0, 3.0) 返回值的类型为 list
    assert isinstance(f(1.0, 2.0, 3.0), type([]))
    # 断言 f(1.0, 2.0, 3.0)[0] 的结果与 expected 数组非常接近
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[0], expected)
    # 断言 f(1.0, 2.0, 3.0)[1] 的结果与 expected 数组非常接近
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[1], expected)

def test_dim_handling():
    # 断言 dim_handling([x], dim=2) 返回值为 {x: (False, False)}
    assert dim_handling([x], dim=2) == {x: (False, False)}
    # 断言 dim_handling([x, y], dims={x: 1, y: 2}) 返回值为 {x: (False, True), y: (False, False)}
    assert dim_handling([x, y], dims={x: 1, y: 2}) == {x: (False, True),
                                                       y: (False, False)}
    # 断言 dim_handling([x], broadcastables={x: (False,)}) 返回值为 {x: (False,)}
    assert dim_handling([x], broadcastables={x: (False,)}) == {x: (False,)}

def test_theano_function_kwargs():
    """
    Test passing additional kwargs from theano_function() to theano.function().
    """
    import numpy as np
    # 创建 theano_function_ 函数 f，输入为 x, y, z，输出为 [x+y]，数据类型为 'float64'，忽略未使用的输入
    f = theano_function_([x, y, z], [x+y], dim=1, on_unused_input='ignore',
            dtypes={x: 'float64', y: 'float64', z: 'float64'})
    # 断言：验证函数 f([1, 2], [3, 4], [0, 0]) 返回的结果与 np.asarray([4, 6]) 的二范数差异小于 1e-9
    assert np.linalg.norm(f([1, 2], [3, 4], [0, 0]) - np.asarray([4, 6])) < 1e-9
    
    # 使用 theano_function_ 函数创建一个 theano 函数 f，接受 x, y, z 三个参数，返回 x+y 的结果
    # 设置输入参数 x, y, z 的数据类型为 'float64'，维度为 1，忽略未使用的输入 z
    f = theano_function_([x, y, z], [x+y],
                        dtypes={x: 'float64', y: 'float64', z: 'float64'},
                        dim=1, on_unused_input='ignore')
    
    # 创建 numpy 数组 xx, yy, zz，分别为 [0. 1. 2.] 的浮点型数组，[0. 2. 4.] 的浮点型数组，[0. 2. 4.] 的浮点型数组
    xx = np.arange(3).astype('float64')
    yy = 2*np.arange(3).astype('float64')
    zz = 2*np.arange(3).astype('float64')
    
    # 断言：验证函数 f(xx, yy, zz) 返回的结果与 3*np.arange(3) 的二范数差异小于 1e-9
    assert np.linalg.norm(f(xx, yy, zz) - 3*np.arange(3)) < 1e-9
def test_theano_function_scalar():
    """ Test the "scalar" argument to theano_function(). """

    args = [
        ([x, y], [x + y], None, [0]),  # Single 0d output
        ([X, Y], [X + Y], None, [2]),  # Single 2d output
        ([x, y], [x + y], {x: 0, y: 1}, [1]),  # Single 1d output
        ([x, y], [x + y, x - y], None, [0, 0]),  # Two 0d outputs
        ([x, y, X, Y], [x + y, X + Y], None, [0, 2]),  # One 0d output, one 2d
    ]

    # Create and test functions with and without the scalar setting
    for inputs, outputs, in_dims, out_dims in args:
        for scalar in [False, True]:

            # Create a theano function using inputs, outputs, dimensions, and scalar setting
            f = theano_function_(inputs, outputs, dims=in_dims, scalar=scalar)

            # Check that the theano_function attribute is set whether wrapped or not
            assert isinstance(f.theano_function, theano.compile.function_module.Function)

            # Feed in inputs of the appropriate size and get outputs
            in_values = [
                np.ones([1 if bc else 5 for bc in i.type.broadcastable])
                for i in f.theano_function.input_storage
            ]
            out_values = f(*in_values)
            if not isinstance(out_values, list):
                out_values = [out_values]

            # Check output types and shapes
            assert len(out_dims) == len(out_values)
            for d, value in zip(out_dims, out_values):
                if scalar and d == 0:
                    # Should have been converted to a scalar value
                    assert isinstance(value, np.number)
                else:
                    # Otherwise should be an array
                    assert isinstance(value, np.ndarray)
                    assert value.ndim == d

def test_theano_function_bad_kwarg():
    """
    Passing an unknown keyword argument to theano_function() should raise an
    exception.
    """
    # Check that an exception is raised when an unknown keyword argument is passed
    raises(Exception, lambda : theano_function_([x], [x+1], foobar=3))


def test_slice():
    # Check that theano_code_ correctly handles slice objects
    assert theano_code_(slice(1, 2, 3)) == slice(1, 2, 3)

    def theq_slice(s1, s2):
        # Helper function to compare slices
        for attr in ['start', 'stop', 'step']:
            a1 = getattr(s1, attr)
            a2 = getattr(s2, attr)
            if a1 is None or a2 is None:
                if not (a1 is None or a2 is None):
                    return False
            elif not theq(a1, a2):
                return False
        return True

    dtypes = {x: 'int32', y: 'int32'}
    # Check that slices generated by theano_code_ are equivalent to expected slices
    assert theq_slice(theano_code_(slice(x, y), dtypes=dtypes), slice(xt, yt))
    assert theq_slice(theano_code_(slice(1, x, 3), dtypes=dtypes), slice(1, xt, 3))

def test_MatrixSlice():
    from theano import Constant

    cache = {}

    n = sy.Symbol('n', integer=True)
    X = sy.MatrixSymbol('X', n, n)

    Y = X[1:2:3, 4:5:6]
    Yt = theano_code_(Y, cache=cache)

    s = ts.Scalar('int64')
    # Check that the indices of the generated theano code match the expected slices
    assert tuple(Yt.owner.op.idx_list) == (slice(s, s, s), slice(s, s, s))
    assert Yt.owner.inputs[0] == theano_code_(X, cache=cache)
    # 使用 assert 语句来验证条件是否为真，如果不为真则抛出 AssertionError
    # 这里的条件是检查所有的 Yt.owner.inputs[i] 是否等于 Constant(s, i)，对于 i 在范围 [1, 7) 内
    assert all(Yt.owner.inputs[i].equals(Constant(s, i)) for i in range(1, 7))

    # 创建一个符号变量 k
    k = sy.Symbol('k')
    # 调用 theano_code_ 函数，并指定 k 的数据类型为 'int32'
    theano_code_(k, dtypes={k: 'int32'})
    
    # 设置 start, stop, step 的值分别为 4, k, 2
    start, stop, step = 4, k, 2
    # 对数组 X 进行切片操作，从索引 start 到 stop-1，步长为 step
    Y = X[start:stop:step]
    
    # 调用 theano_code_ 函数，并指定 Y 中的元素类型为 'int32'，同时 k 的类型也为 'int32'
    Yt = theano_code_(Y, dtypes={n: 'int32', k: 'int32'})
    
    # 使用 assert 语句检查 Yt.owner.op.idx_list[0].stop 是否等于 kt
    # 注意：代码中注释掉了对 kt 的实际引用
    # assert Yt.owner.op.idx_list[0].stop == kt
def test_BlockMatrix():
    # 定义一个整数符号变量 n
    n = sy.Symbol('n', integer=True)
    # 创建四个矩阵符号变量 A, B, C, D，每个都是 n x n 的矩阵
    A, B, C, D = [sy.MatrixSymbol(name, n, n) for name in 'ABCD']
    # 对每个矩阵应用 theano_code_ 函数，生成其 Theano 表达式
    At, Bt, Ct, Dt = map(theano_code_, (A, B, C, D))
    # 创建一个 BlockMatrix 对象 Block，由矩阵 A, B, C, D 组成的 2x2 块矩阵
    Block = sy.BlockMatrix([[A, B], [C, D]])
    # 将 Block 对象转换为 Theano 表达式 Blockt
    Blockt = theano_code_(Block)
    # 定义两个可能的解决方案，每个都是 Theano 表达式的组合
    solutions = [tt.join(0, tt.join(1, At, Bt), tt.join(1, Ct, Dt)),
                 tt.join(1, tt.join(0, At, Ct), tt.join(0, Bt, Dt))]
    # 断言：Blockt 与 solutions 中的任何一个表达式相等
    assert any(theq(Blockt, solution) for solution in solutions)

@SKIP
def test_BlockMatrix_Inverse_execution():
    # 定义变量 k, n，并指定数据类型为 'float32'
    k, n = 2, 4
    dtype = 'float32'
    # 创建矩阵符号变量 A 和 B
    A = sy.MatrixSymbol('A', n, k)
    B = sy.MatrixSymbol('B', n, n)
    inputs = A, B
    # 定义输出为 B 的逆乘以 A
    output = B.I * A

    # 定义 cutsizes 字典，指定输入矩阵 A 和 B 的切割尺寸
    cutsizes = {A: [(n//2, n//2), (k//2, k//2)],
                B: [(n//2, n//2), (n//2, n//2)]}
    # 对输入矩阵进行切割，得到 cutinputs 列表
    cutinputs = [sy.blockcut(i, *cutsizes[i]) for i in inputs]
    # 将输出中的符号变量用 cutinputs 替换，得到 cutoutput
    cutoutput = output.subs(dict(zip(inputs, cutinputs)))

    # 定义 dtypes 字典，指定每个输入矩阵的数据类型
    dtypes = dict(zip(inputs, [dtype]*len(inputs)))
    # 创建 Theano 函数 f，计算输出 output，使用给定的数据类型和空缓存
    f = theano_function_(inputs, [output], dtypes=dtypes, cache={})
    # 创建 Theano 函数 fblocked，计算简化后的 cutoutput，使用给定的数据类型和空缓存
    fblocked = theano_function_(inputs, [sy.block_collapse(cutoutput)],
                                dtypes=dtypes, cache={})

    # 生成随机输入数据 ninputs，确保两个函数的输出在给定误差范围内相等
    ninputs = [np.random.rand(*x.shape).astype(dtype) for x in inputs]
    assert np.allclose(f(*ninputs), fblocked(*ninputs), rtol=1e-5)

def test_DenseMatrix():
    # 定义符号变量 t
    t = sy.Symbol('theta')
    # 遍历 MatrixType 列表，分别创建包含 cos(t) 和 sin(t) 的 2x2 矩阵
    for MatrixType in [sy.Matrix, sy.ImmutableMatrix]:
        X = MatrixType([[sy.cos(t), -sy.sin(t)], [sy.sin(t), sy.cos(t)]])
        # 将矩阵 X 转换为 Theano 表达式 tX
        tX = theano_code_(X)
        # 断言：tX 是一个 TensorVariable 对象
        assert isinstance(tX, tt.TensorVariable)
        # 断言：tX 的操作为 tt.join_
        assert tX.owner.op == tt.join_

def test_cache_basic():
    """ Test single symbol-like objects are cached when printed by themselves. """

    # 需要缓存的对象对
    pairs = [
        (x, sy.Symbol('x')),
        (X, sy.MatrixSymbol('X', *X.shape)),
        (f_t, sy.Function('f')(sy.Symbol('t'))),
    ]

    for s1, s2 in pairs:
        cache = {}
        # 将 s1 转换为 Theano 表达式 st，并使用 cache 缓存
        st = theano_code_(s1, cache=cache)

        # 测试：使用相同的 cache，s1 转换后的结果应该与 st 相同
        assert theano_code_(s1, cache=cache) is st

        # 测试：使用新的 cache，s1 转换后的结果不应该与 st 相同
        assert theano_code_(s1, cache={}) is not st

        # 测试：使用相同的 cache，s2 转换后的结果应该与 st 相同
        assert theano_code_(s2, cache=cache) is st

def test_global_cache():
    """ Test use of the global cache. """
    from sympy.printing.theanocode import global_cache

    backup = dict(global_cache)
    try:
        # 临时清空全局缓存
        global_cache.clear()

        # 遍历符号变量列表 [x, X, f_t]
        for s in [x, X, f_t]:
            with warns_deprecated_sympy():
                # 将符号变量 s 转换为 Theano 表达式，并使用全局缓存
                st = theano_code(s)
                # 测试：使用相同的符号变量 s，转换后的结果应该与 st 相同
                assert theano_code(s) is st

    finally:
        # 恢复全局缓存
        global_cache.update(backup)

def test_cache_types_distinct():
    """
    Test that symbol-like objects of different types (Symbol, MatrixSymbol,
    AppliedUndef) are distinguished by the cache even if they have the same
    name.
    """
    symbols = [sy.Symbol('f_t'), sy.MatrixSymbol('f_t', 4, 4), f_t]

    cache = {}  # 单一共享缓存，用于存储已处理过的对象的输出
    printed = {}  # 存储已打印对象的输出结果

    for s in symbols:
        st = theano_code_(s, cache=cache)  # 将当前符号 s 转换为 Theano 代码字符串 st
        assert st not in printed.values()  # 确保当前生成的 st 在 printed 的值中不存在（保证唯一性）
        printed[s] = st  # 将当前符号 s 的输出结果 st 存入 printed 中

    # 检查所有打印的对象输出都是不同的
    assert len(set(map(id, printed.values()))) == len(symbols)

    # 检查获取操作的正确性
    for s, st in printed.items():
        with warns_deprecated_sympy():
            assert theano_code(s, cache=cache) is st  # 检查从缓存中检索到的 Theano 代码与预期的 st 相同
def test_symbols_are_created_once():
    """
    Test that a symbol is cached and reused when it appears in an expression
    more than once.
    """
    # 创建一个 SymPy 表达式，表示 x + x，但不立即求值
    expr = sy.Add(x, x, evaluate=False)
    # 将表达式转换为 Theano 代码
    comp = theano_code_(expr)

    # 断言 Theano 生成的代码与 xt + xt 相等
    assert theq(comp, xt + xt)
    # 断言 Theano 生成的代码与 xt + theano_code_(x) 不相等
    assert not theq(comp, xt + theano_code_(x))


def test_cache_complex():
    """
    Test caching on a complicated expression with multiple symbols appearing
    multiple times.
    """
    # 创建一个复杂的 SymPy 表达式
    expr = x ** 2 + (y - sy.exp(x)) * sy.sin(z - x * y)
    # 获取表达式中所有自由符号的名称
    symbol_names = {s.name for s in expr.free_symbols}
    # 将表达式转换为 Theano 代码
    expr_t = theano_code_(expr)

    # 遍历表达式在 Theano 计算图中依赖的变量
    seen = set()
    for v in theano.gof.graph.ancestors([expr_t]):
        # 如果变量没有 owner 并且不是常数，应为我们的符号
        if v.owner is None and not isinstance(v, theano.gof.graph.Constant):
            # 检查该变量是否对应一个符号并且只出现一次
            assert v.name in symbol_names
            assert v.name not in seen
            seen.add(v.name)

    # 断言所有的符号都已经被找到
    assert seen == symbol_names


def test_Piecewise():
    # 测试分段函数 Piecewise 的转换
    expr = sy.Piecewise((0, x<0), (x, x<2), (1, True))  # ___/III
    result = theano_code_(expr)
    # 断言结果的操作符是 Theano 的 switch 函数
    assert result.owner.op == tt.switch

    # 预期的 Theano 表达式
    expected = tt.switch(xt<0, 0, tt.switch(xt<2, xt, 1))
    assert theq(result, expected)

    expr = sy.Piecewise((x, x < 0))
    result = theano_code_(expr)
    expected = tt.switch(xt < 0, xt, np.nan)
    assert theq(result, expected)

    expr = sy.Piecewise((0, sy.And(x>0, x<2)), \
        (x, sy.Or(x>2, x<0)))
    result = theano_code_(expr)
    expected = tt.switch(tt.and_(xt>0,xt<2), 0, \
        tt.switch(tt.or_(xt>2, xt<0), xt, np.nan))
    assert theq(result, expected)


def test_Relationals():
    # 测试关系运算符的转换
    assert theq(theano_code_(sy.Eq(x, y)), tt.eq(xt, yt))
    # assert theq(theano_code_(sy.Ne(x, y)), tt.neq(xt, yt))  # TODO - implement
    assert theq(theano_code_(x > y), xt > yt)
    assert theq(theano_code_(x < y), xt < yt)
    assert theq(theano_code_(x >= y), xt >= yt)
    assert theq(theano_code_(x <= y), xt <= yt)


def test_complexfunctions():
    with warns_deprecated_sympy():
        # 使用 theano_code_ 转换复杂数学函数表达式，指定数据类型为 'complex128'
        xt, yt = theano_code_(x, dtypes={x:'complex128'}), theano_code_(y, dtypes={y: 'complex128'})
    from sympy.functions.elementary.complexes import conjugate
    from theano.tensor import as_tensor_variable as atv
    from theano.tensor import complex as cplx
    with warns_deprecated_sympy():
        # 断言 Theano 生成的代码与期望的结果相等
        assert theq(theano_code_(y*conjugate(x)), yt*(xt.conj()))
        assert theq(theano_code_((1+2j)*x), xt*(atv(1.0)+atv(2.0)*cplx(0,1)))


def test_constantfunctions():
    with warns_deprecated_sympy():
        # 测试返回常数的 Theano 函数
        tf = theano_function_([],[1+1j])
    assert(tf()==1+1j)


def test_Exp1():
    """
    Test that exp(1) prints without error and evaluates close to SymPy's E
    """
    # 这个测试确保 exp(1) 能够正常输出并且在数值上接近 SymPy 的 E
    # 计算自然对数的底 e 的两种方式：sy.exp(1) 和 sy.E 应该返回相同的 E 实例（单例），但为了确保，额外进行了检查。

    # 使用 sympy 库的 sy.exp(1) 计算 e 的近似值
    e_a = sy.exp(1)
    # 直接使用 sympy 库中预定义的常数 sy.E 获取 e 的值
    e_b = sy.E

    # 使用 NumPy 测试确保 e_a 的近似值与 np.e 非常接近
    np.testing.assert_allclose(float(e_a), np.e)
    # 使用 NumPy 测试确保 e_b 的近似值与 np.e 非常接近
    np.testing.assert_allclose(float(e_b), np.e)

    # 使用 theano_code_ 函数处理 e_a 的代码表示，并测试其值是否与 e_a 近似
    e = theano_code_(e_a)
    np.testing.assert_allclose(float(e_a), e.eval())

    # 使用 theano_code_ 函数处理 e_b 的代码表示，并测试其值是否与 e_b 近似
    e = theano_code_(e_b)
    np.testing.assert_allclose(float(e_b), e.eval())
```