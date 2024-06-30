# `D:\src\scipysrc\sympy\sympy\printing\tests\test_aesaracode.py`

```
"""
重要说明：本模块中的测试使用 Aesara 打印函数默认使用全局缓存，这意味着使用它的测试会修改全局状态，因此不会相互独立。为了避免每次都使用 "cache" 关键字参数，本模块使用下面定义的 aesara_code_ 和 aesara_function_ 函数，默认使用新的空缓存。

"""

# 导入日志模块
import logging

# 导入 sympy 的模块和函数
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP

# 导入忽略警告的装饰器
from sympy.utilities.exceptions import ignore_warnings

# 设置 Aesara 的日志记录器
aesaralogger = logging.getLogger('aesara.configdefaults')
aesaralogger.setLevel(logging.CRITICAL)

# 导入 aesara 模块，如果导入成功，则进一步导入以下内容
aesara = import_module('aesara')
if aesara:
    import numpy as np
    aet = aesara.tensor
    from aesara.scalar.basic import ScalarType
    from aesara.graph.basic import Variable
    from aesara.tensor.var import TensorVariable
    from aesara.tensor.elemwise import Elemwise, DimShuffle
    from aesara.tensor.math import Dot

    # 导入 sympy 打印 aesara 代码相关函数
    from sympy.printing.aesaracode import true_divide

    # 创建标量和张量变量
    xt, yt, zt = [aet.scalar(name, 'floatX') for name in 'xyz']
    Xt, Yt, Zt = [aet.tensor('floatX', (False, False), name=n) for n in 'XYZ']
else:
    # 如果 aesara 导入失败，禁用测试
    disabled = True

# 导入 sympy 库和相关符号
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.aesaracode import (aesara_code, dim_handling,
        aesara_function)

# 创建用于测试的默认矩阵符号集合，使其为方阵，以便可以进行乘法和元素操作
X, Y, Z = [sy.MatrixSymbol(n, 4, 4) for n in 'XYZ']

# 用于测试 AppliedUndef 的函数符号
f_t = sy.Function('f')(t)


def aesara_code_(expr, **kwargs):
    """ 使用新的空缓存，默认包装 aesara_code 函数。 """
    kwargs.setdefault('cache', {})
    return aesara_code(expr, **kwargs)

def aesara_function_(inputs, outputs, **kwargs):
    """ 使用新的空缓存，默认包装 aesara_function 函数。 """
    kwargs.setdefault('cache', {})
    return aesara_function(inputs, outputs, **kwargs)


def fgraph_of(*exprs):
    """
    将 SymPy 表达式转换为 Aesara 计算图。

    Parameters
    ==========
    exprs
        SymPy 表达式列表

    Returns
    =======
    aesara.graph.fg.FunctionGraph
    """
    # 将表达式列表映射为 Aesara 表达式
    outs = list(map(aesara_code_, exprs))
    # 获取输入变量列表
    ins = list(aesara.graph.basic.graph_inputs(outs))
    # 克隆输入和输出
    ins, outs = aesara.graph.basic.clone(ins, outs)
    # 返回函数图
    return aesara.graph.fg.FunctionGraph(ins, outs)


def aesara_simplify(fgraph):
    """
    简化 Aesara 计算图。

    Parameters
    ==========
    fgraph : aesara.graph.fg.FunctionGraph
        Aesara 计算图对象

    Returns
    =======
    aesara.graph.fg.FunctionGraph
        简化后的 Aesara 计算图对象
    """
    # 获取默认的编译模式，排除 "fusion" 优化
    mode = aesara.compile.get_default_mode().excluding("fusion")
    # 克隆计算图
    fgraph = fgraph.clone()
    # 优化计算图
    mode.optimizer.rewrite(fgraph)
    # 返回优化后的计算图
    return fgraph


def theq(a, b):
    """ 检查两个 Aesara 对象是否相等。 """
    # 检查两个对象是否相等
    return a == b
    """
    Also accepts numeric types and lists/tuples of supported types.

    Note - debugprint() has a bug where it will accept numeric types but does
    not respect the "file" argument and in this case and instead prints the number
    to stdout and returns an empty string. This can lead to tests passing where
    they should fail because any two numbers will always compare as equal. To
    prevent this we treat numbers as a separate case.
    """
    # 定义支持的数值类型，包括整数、浮点数和 numpy 的数值类型
    numeric_types = (int, float, np.number)
    # 检查 a 是否属于数值类型
    a_is_num = isinstance(a, numeric_types)
    # 检查 b 是否属于数值类型
    b_is_num = isinstance(b, numeric_types)

    # 如果 a 或 b 是数值类型，则直接使用普通的相等比较
    if a_is_num or b_is_num:
        # 如果 a 和 b 都是数值类型，才继续比较
        if not (a_is_num and b_is_num):
            return False
        
        # 直接返回数值相等比较的结果
        return a == b

    # 检查 a 是否为序列类型（tuple 或 list）
    a_is_seq = isinstance(a, (tuple, list))
    # 检查 b 是否为序列类型（tuple 或 list）
    b_is_seq = isinstance(b, (tuple, list))

    # 如果 a 或 b 是序列类型，则逐元素比较
    if a_is_seq or b_is_seq:
        # 如果 a 和 b 不同时为序列类型或者类型不同，则返回 False
        if not (a_is_seq and b_is_seq) or type(a) != type(b):
            return False
        
        # 使用递归方式逐元素比较序列的每个元素
        return list(map(theq, a)) == list(map(theq, b))

    # 如果不是数值类型也不是序列类型，则调用 debugprint() 函数处理
    astr = aesara.printing.debugprint(a, file='str')
    bstr = aesara.printing.debugprint(b, file='str')

    # 检查 debugprint() 返回的结果是否为空字符串（已知 bug 情况）
    for argname, argval, argstr in [('a', a, astr), ('b', b, bstr)]:
        if argstr == '':
            raise TypeError(
                'aesara.printing.debugprint(%s) returned empty string '
                '(%s is instance of %r)'
                % (argname, argname, type(argval))
            )

    # 最终比较 debugprint() 返回的字符串表示
    return astr == bstr
def test_example_symbols():
    """
    Check that the example symbols in this module print to their Aesara
    equivalents, as many of the other tests depend on this.
    """
    # 断言各个符号的 Aesara 等效性
    assert theq(xt, aesara_code_(x))
    assert theq(yt, aesara_code_(y))
    assert theq(zt, aesara_code_(z))
    assert theq(Xt, aesara_code_(X))
    assert theq(Yt, aesara_code_(Y))
    assert theq(Zt, aesara_code_(Z))


def test_Symbol():
    """ Test printing a Symbol to a aesara variable. """
    # 将符号打印到 Aesara 变量，并进行相关断言
    xx = aesara_code_(x)
    assert isinstance(xx, Variable)
    assert xx.broadcastable == ()
    assert xx.name == x.name

    xx2 = aesara_code_(x, broadcastables={x: (False,)})
    assert xx2.broadcastable == (False,)
    assert xx2.name == x.name


def test_MatrixSymbol():
    """ Test printing a MatrixSymbol to a aesara variable. """
    # 将矩阵符号打印到 Aesara 张量变量，并进行相关断言
    XX = aesara_code_(X)
    assert isinstance(XX, TensorVariable)
    assert XX.broadcastable == (False, False)


@SKIP  # TODO - this is currently not checked but should be implemented
def test_MatrixSymbol_wrong_dims():
    """ Test MatrixSymbol with invalid broadcastable. """
    # 使用无效的广播属性测试 MatrixSymbol
    bcs = [(), (False,), (True,), (True, False), (False, True,), (True, True)]
    for bc in bcs:
        with raises(ValueError):
            aesara_code_(X, broadcastables={X: bc})


def test_AppliedUndef():
    """ Test printing AppliedUndef instance, which works similarly to Symbol. """
    # 测试打印 AppliedUndef 实例，并进行相关断言
    ftt = aesara_code_(f_t)
    assert isinstance(ftt, TensorVariable)
    assert ftt.broadcastable == ()
    assert ftt.name == 'f_t'


def test_add():
    # 测试加法表达式的打印
    expr = x + y
    comp = aesara_code_(expr)
    assert comp.owner.op == aesara.tensor.add


def test_trig():
    # 测试三角函数的打印
    assert theq(aesara_code_(sy.sin(x)), aet.sin(xt))
    assert theq(aesara_code_(sy.tan(x)), aet.tan(xt))


def test_many():
    """ Test printing a complex expression with multiple symbols. """
    # 测试打印包含多个符号的复杂表达式
    expr = sy.exp(x**2 + sy.cos(y)) * sy.log(2*z)
    comp = aesara_code_(expr)
    expected = aet.exp(xt**2 + aet.cos(yt)) * aet.log(2*zt)
    assert theq(comp, expected)


def test_dtype():
    """ Test specifying specific data types through the dtype argument. """
    # 测试使用 dtype 参数指定特定数据类型
    for dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']:
        assert aesara_code_(x, dtypes={x: dtype}).type.dtype == dtype

    # "floatX" 类型
    assert aesara_code_(x, dtypes={x: 'floatX'}).type.dtype in ('float32', 'float64')

    # 类型提升
    assert aesara_code_(x + 1, dtypes={x: 'float32'}).type.dtype == 'float32'
    assert aesara_code_(x + y, dtypes={x: 'float64', y: 'float32'}).type.dtype == 'float64'


def test_broadcastables():
    """ Test the "broadcastables" argument when printing symbol-like objects. """

    # 对形状没有限制的测试
    for s in [x, f_t]:
        for bc in [(), (False,), (True,), (False, False), (True, False)]:
            assert aesara_code_(s, broadcastables={s: bc}).broadcastable == bc

    # TODO - 矩阵广播的测试?


def test_broadcasting():
    # TODO: Add test cases for broadcasting
    pass
    """ Test "broadcastable" attribute after applying element-wise binary op. """
    
    # 定义表达式，计算 x + y 的结果
    expr = x + y
    
    # 不同的测试案例，每个案例包含三个元组，分别表示 x 的广播属性 bc1、y 的广播属性 bc2 和 expr 的期望广播属性 bc3
    cases = [
        [(), (), ()],                   # 空元组的广播属性
        [(False,), (False,), (False,)], # 单元素元组的广播属性，都为 False
        [(True,), (False,), (False,)],  # 单元素元组的广播属性，x 为 True，y 为 False，结果为 False
        [(False, True), (False, False), (False, False)],  # 双元素元组的广播属性，x 的第二个元素为 True，结果的广播属性为 False
        [(True, False), (False, False), (False, False)],  # 双元素元组的广播属性，x 的第一个元素为 True，结果的广播属性为 False
    ]
    
    # 遍历每个测试案例
    for bc1, bc2, bc3 in cases:
        # 调用 aesara_code_ 函数，传入表达式 expr 和广播属性字典 {x: bc1, y: bc2}，获取计算结果的广播属性
        comp = aesara_code_(expr, broadcastables={x: bc1, y: bc2})
        # 断言计算结果的广播属性与预期的 bc3 相同
        assert comp.broadcastable == bc3
def test_MatMul():
    # 创建一个表达式 X*Y*Z
    expr = X*Y*Z
    # 调用 aesara_code_ 函数处理表达式，返回处理后的结果
    expr_t = aesara_code_(expr)
    # 断言表达式的操作对象是 Dot 类型
    assert isinstance(expr_t.owner.op, Dot)
    # 断言处理后的表达式等价于 Xt.dot(Yt).dot(Zt)
    assert theq(expr_t, Xt.dot(Yt).dot(Zt))

def test_Transpose():
    # 断言 aesara_code_(X.T) 的操作对象是 DimShuffle 类型
    assert isinstance(aesara_code_(X.T).owner.op, DimShuffle)

def test_MatAdd():
    # 创建一个表达式 X+Y+Z
    expr = X+Y+Z
    # 断言 aesara_code_(expr) 的操作对象是 Elemwise 类型
    assert isinstance(aesara_code_(expr).owner.op, Elemwise)

def test_Rationals():
    # 断言 aesara_code_(sy.Integer(2) / 3) 等于 true_divide(2, 3)
    assert theq(aesara_code_(sy.Integer(2) / 3), true_divide(2, 3))
    # 断言 aesara_code_(S.Half) 等于 true_divide(1, 2)
    assert theq(aesara_code_(S.Half), true_divide(1, 2))

def test_Integers():
    # 断言 aesara_code_(sy.Integer(3)) 等于 3
    assert aesara_code_(sy.Integer(3)) == 3

def test_factorial():
    # 创建一个符号变量 n
    n = sy.Symbol('n')
    # 断言 aesara_code_(sy.factorial(n)) 的值
    assert aesara_code_(sy.factorial(n))

def test_Derivative():
    # 使用 ignore_warnings 上下文，忽略 UserWarning
    with ignore_warnings(UserWarning):
        # 定义简化表达式函数 simp
        simp = lambda expr: aesara_simplify(fgraph_of(expr))
        # 断言简化后的 aesara_code_(sy.Derivative(sy.sin(x), x, evaluate=False)) 等于简化后的 aesara.grad(aet.sin(xt), xt)
        assert theq(simp(aesara_code_(sy.Derivative(sy.sin(x), x, evaluate=False))),
                    simp(aesara.grad(aet.sin(xt), xt)))

def test_aesara_function_simple():
    """ Test aesara_function() with single output. """
    # 使用 aesara_function_ 创建函数 f，计算 x+y
    f = aesara_function_([x, y], [x+y])
    # 断言 f(2, 3) 的结果等于 5
    assert f(2, 3) == 5

def test_aesara_function_multi():
    """ Test aesara_function() with multiple outputs. """
    # 使用 aesara_function_ 创建函数 f，计算 x+y 和 x-y
    f = aesara_function_([x, y], [x+y, x-y])
    # 分别将 f(2, 3) 的结果赋给 o1 和 o2
    o1, o2 = f(2, 3)
    # 断言 o1 等于 5
    assert o1 == 5
    # 断言 o2 等于 -1
    assert o2 == -1

def test_aesara_function_numpy():
    """ Test aesara_function() vs Numpy implementation. """
    # 使用 aesara_function_ 创建函数 f，计算 x+y，指定数据类型和维度
    f = aesara_function_([x, y], [x+y], dim=1,
                         dtypes={x: 'float64', y: 'float64'})
    # 断言 f([1, 2], [3, 4]) 和 numpy 实现的结果非常接近
    assert np.linalg.norm(f([1, 2], [3, 4]) - np.asarray([4, 6])) < 1e-9

    # 使用 aesara_function_ 创建函数 f，计算 x+y，指定数据类型
    f = aesara_function_([x, y], [x+y], dtypes={x: 'float64', y: 'float64'},
                         dim=1)
    # 创建 numpy 数组 xx 和 yy
    xx = np.arange(3).astype('float64')
    yy = 2*np.arange(3).astype('float64')
    # 断言 f(xx, yy) 和 3*xx 的结果非常接近
    assert np.linalg.norm(f(xx, yy) - 3*np.arange(3)) < 1e-9

def test_aesara_function_matrix():
    # 创建一个符号矩阵 m
    m = sy.Matrix([[x, y], [z, x + y + z]])
    # 创建预期的 numpy 数组 expected
    expected = np.array([[1.0, 2.0], [3.0, 1.0 + 2.0 + 3.0]])
    # 使用 aesara_function_ 创建函数 f，计算 m
    f = aesara_function_([x, y, z], [m])
    # 断言 f(1.0, 2.0, 3.0) 的结果与 expected 非常接近
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    # 使用 aesara_function_ 创建函数 f，计算 m，返回标量值
    f = aesara_function_([x, y, z], [m], scalar=True)
    # 断言 f(1.0, 2.0, 3.0) 的结果与 expected 非常接近
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    # 使用 aesara_function_ 创建函数 f，计算 m，返回数组
    f = aesara_function_([x, y, z], [m, m])
    # 断言 f(1.0, 2.0, 3.0) 返回的类型是列表
    assert isinstance(f(1.0, 2.0, 3.0), type([]))
    # 断言 f(1.0, 2.0, 3.0)[0] 的结果与 expected 非常接近
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[0], expected)
    # 断言 f(1.0, 2.0, 3.0)[1] 的结果与 expected 非常接近
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[1], expected)

def test_dim_handling():
    # 断言 dim_handling([x], dim=2) 的结果是 {x: (False, False)}
    assert dim_handling([x], dim=2) == {x: (False, False)}
    # 断言 dim_handling([x, y], dims={x: 1, y: 2}) 的结果是 {x: (False, True), y: (False, False)}
    assert dim_handling([x, y], dims={x: 1, y: 2}) == {x: (False, True),
                                                       y: (False, False)}
    # 断言 dim_handling([x], broadcastables={x: (False,)}) 的结果是 {x: (False,)}
    assert dim_handling([x], broadcastables={x: (False,)}) == {x: (False,)}

def test_aesara_function_kwargs():
    """
    Test passing additional kwargs from aesara_function() to aesara.function().
    """
    # 使用 aesara_function_ 创建函数 f，计算 x+y，传递额外参数给 aesara.function()
    f = aesara_function_([x, y, z], [x+y], dim=1, on_unused_input='ignore',
            dtypes={x: 'float64', y: 'float64', z: 'float64'})
    # 使用 NumPy 的线性代数模块计算向量 f([1, 2], [3, 4], [0, 0]) 减去向量 [4, 6] 的二范数，确保其小于 1e-9
    assert np.linalg.norm(f([1, 2], [3, 4], [0, 0]) - np.asarray([4, 6])) < 1e-9
    
    # 使用 aesara_function_ 函数创建一个处理三个输入变量 x, y, z 的函数 f，并指定数据类型为 'float64'，维度为 1，忽略未使用的输入
    f = aesara_function_([x, y, z], [x+y],
                        dtypes={x: 'float64', y: 'float64', z: 'float64'},
                        dim=1, on_unused_input='ignore')
    
    # 创建一个包含三个元素的 NumPy 数组 xx，元素为 [0.0, 1.0, 2.0]，数据类型为 'float64'
    xx = np.arange(3).astype('float64')
    
    # 创建一个包含三个元素的 NumPy 数组 yy，元素为 [0.0, 2.0, 4.0]，数据类型为 'float64'
    yy = 2*np.arange(3).astype('float64')
    
    # 创建一个包含三个元素的 NumPy 数组 zz，元素为 [0.0, 2.0, 4.0]，数据类型为 'float64'
    zz = 2*np.arange(3).astype('float64')
    
    # 使用 NumPy 的线性代数模块计算向量 f(xx, yy, zz) 减去向量 [0, 3, 6] 的二范数，确保其小于 1e-9
    assert np.linalg.norm(f(xx, yy, zz) - 3*np.arange(3)) < 1e-9
# 测试 aesara_function() 函数的标量参数
def test_aesara_function_scalar():
    # 从 aesara.compile.function.types 导入 Function 类
    from aesara.compile.function.types import Function

    # 定义不同的输入参数组合及其期望输出
    args = [
        ([x, y], [x + y], None, [0]),  # 单个0维输出
        ([X, Y], [X + Y], None, [2]),  # 单个2维输出
        ([x, y], [x + y], {x: 0, y: 1}, [1]),  # 单个1维输出
        ([x, y], [x + y, x - y], None, [0, 0]),  # 两个0维输出
        ([x, y, X, Y], [x + y, X + Y], None, [0, 2]),  # 一个0维输出，一个2维输出
    ]

    # 创建并测试带有标量设置和不带标量设置的函数
    for inputs, outputs, in_dims, out_dims in args:
        for scalar in [False, True]:

            f = aesara_function_(inputs, outputs, dims=in_dims, scalar=scalar)

            # 检查 aesara_function 属性是否设置，无论是否被包装
            assert isinstance(f.aesara_function, Function)

            # 提供适当大小的输入并获取输出
            in_values = [
                np.ones([1 if bc else 5 for bc in i.type.broadcastable])
                for i in f.aesara_function.input_storage
            ]
            out_values = f(*in_values)
            if not isinstance(out_values, list):
                out_values = [out_values]

            # 检查输出的类型和形状
            assert len(out_dims) == len(out_values)
            for d, value in zip(out_dims, out_values):

                if scalar and d == 0:
                    # 应该已转换为标量值
                    assert isinstance(value, np.number)

                else:
                    # 否则应该是一个数组
                    assert isinstance(value, np.ndarray)
                    assert value.ndim == d

# 测试 aesara_function() 函数传递未知关键字参数时是否引发异常
def test_aesara_function_bad_kwarg():
    """
    Passing an unknown keyword argument to aesara_function() should raise an
    exception.
    """
    raises(Exception, lambda : aesara_function_([x], [x+1], foobar=3))


# 测试 slice() 函数的行为
def test_slice():
    # 断言 aesara_code_(slice(1, 2, 3)) 的返回值等于 slice(1, 2, 3)
    assert aesara_code_(slice(1, 2, 3)) == slice(1, 2, 3)

    # 定义比较 slice 对象的函数 theq_slice
    def theq_slice(s1, s2):
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
    # 断言 aesara_code_(slice(x, y), dtypes=dtypes) 的行为等同于 slice(xt, yt)
    assert theq_slice(aesara_code_(slice(x, y), dtypes=dtypes), slice(xt, yt))
    # 断言 aesara_code_(slice(1, x, 3), dtypes=dtypes) 的行为等同于 slice(1, xt, 3)
    assert theq_slice(aesara_code_(slice(1, x, 3), dtypes=dtypes), slice(1, xt, 3))

# 测试 MatrixSlice 的行为
def test_MatrixSlice():
    cache = {}

    n = sy.Symbol('n', integer=True)
    X = sy.MatrixSymbol('X', n, n)

    # 创建符号矩阵切片 Y
    Y = X[1:2:3, 4:5:6]
    # 获取 Y 的 aesara 代码表示，并使用缓存 cache
    Yt = aesara_code_(Y, cache=cache)

    s = ScalarType('int64')
    # 断言 Yt.owner.op.idx_list 的值等于 (slice(s, s, s), slice(s, s, s))
    assert tuple(Yt.owner.op.idx_list) == (slice(s, s, s), slice(s, s, s))
    # 断言 Yt.owner.inputs[0] 的值等于 aesara_code_(X, cache=cache)
    assert Yt.owner.inputs[0] == aesara_code_(X, cache=cache)
    # == 在 Aesara 中不像在 SymPy 中一样工作，必须使用
    # 断言：验证所有的 Yt.owner.inputs[i].data 是否等于对应的索引 i（从 1 到 6）
    assert all(Yt.owner.inputs[i].data == i for i in range(1, 7))

    # 创建符号变量 k
    k = sy.Symbol('k')
    # 调用 aesara_code_ 函数，传入符号变量 k，并指定 k 的数据类型为 'int32'
    aesara_code_(k, dtypes={k: 'int32'})
    
    # 定义切片操作的起始点、终止点和步长
    start, stop, step = 4, k, 2
    # 对数组 X 执行切片操作，起始点为 start，终止点为 stop，步长为 step
    Y = X[start:stop:step]
    
    # 调用 aesara_code_ 函数，传入切片后的数组 Y，并指定变量 n 和 k 的数据类型均为 'int32'
    Yt = aesara_code_(Y, dtypes={n: 'int32', k: 'int32'})
    
    # 断言：验证 Yt 对象的 owner 属性中的 op 属性的 idx_list 列表的第一个元素的 stop 属性是否等于 kt
    # （注：这里的 kt 变量未在提供的代码中定义或说明）
    # assert Yt.owner.op.idx_list[0].stop == kt
def test_BlockMatrix():
    # 定义整数符号变量 n
    n = sy.Symbol('n', integer=True)
    # 创建符号矩阵变量 A, B, C, D，每个都是 n x n 的矩阵符号
    A, B, C, D = [sy.MatrixSymbol(name, n, n) for name in 'ABCD']
    # 将 A, B, C, D 映射为 Aesara 代码表示的变量 At, Bt, Ct, Dt
    At, Bt, Ct, Dt = map(aesara_code_, (A, B, C, D))
    # 创建块矩阵 Block，其内容为 [[A, B], [C, D]]
    Block = sy.BlockMatrix([[A, B], [C, D]])
    # 将 Block 转换为 Aesara 代码表示的变量 Blockt
    Blockt = aesara_code_(Block)
    # 定义可能的解决方案列表，每个解决方案都是 Aesara 表达式
    solutions = [aet.join(0, aet.join(1, At, Bt), aet.join(1, Ct, Dt)),
                 aet.join(1, aet.join(0, At, Ct), aet.join(0, Bt, Dt))]
    # 断言 Blockt 等于 solutions 中的任意一个
    assert any(theq(Blockt, solution) for solution in solutions)

@SKIP
def test_BlockMatrix_Inverse_execution():
    # 定义变量 k 和 n，分别为 2 和 4
    k, n = 2, 4
    # 定义数据类型为 'float32'
    dtype = 'float32'
    # 创建矩阵符号 A 和 B，分别为 n x k 和 n x n 的矩阵符号
    A = sy.MatrixSymbol('A', n, k)
    B = sy.MatrixSymbol('B', n, n)
    inputs = A, B
    # 定义输出为 B 的逆乘以 A
    output = B.I * A

    # 定义 cutsizes 字典，用于指定 A 和 B 的切割大小
    cutsizes = {A: [(n//2, n//2), (k//2, k//2)],
                B: [(n//2, n//2), (n//2, n//2)]}
    # 对 inputs 中的每个元素应用块切割，生成 cutinputs 列表
    cutinputs = [sy.blockcut(i, *cutsizes[i]) for i in inputs]
    # 对 output 应用替换，将 inputs 替换为 cutinputs，生成 cutoutput
    cutoutput = output.subs(dict(zip(inputs, cutinputs)))

    # 创建 dtypes 字典，指定 inputs 的数据类型为 'float32'
    dtypes = dict(zip(inputs, [dtype]*len(inputs)))
    # 创建 Aesara 函数 f，接受 inputs 作为输入，output 作为输出
    f = aesara_function_(inputs, [output], dtypes=dtypes, cache={})
    # 创建 Aesara 函数 fblocked，接受 cutinputs 作为输入，cutoutput 作为输出
    fblocked = aesara_function_(inputs, [sy.block_collapse(cutoutput)],
                                dtypes=dtypes, cache={})

    # 创建 ninputs，包含两个随机生成的数组，分别符合 A 和 B 的形状和 'float32' 类型
    ninputs = [np.random.rand(*x.shape).astype(dtype) for x in inputs]
    ninputs = [np.arange(n*k).reshape(A.shape).astype(dtype),
               np.eye(n).astype(dtype)]
    ninputs[1] += np.ones(B.shape)*1e-5

    # 断言 f 和 fblocked 对于 ninputs 的输出在给定相对误差范围内相等
    assert np.allclose(f(*ninputs), fblocked(*ninputs), rtol=1e-5)

def test_DenseMatrix():
    from aesara.tensor.basic import Join

    t = sy.Symbol('theta')
    # 对于每个 MatrixType，分别创建 2x2 的矩阵 X
    for MatrixType in [sy.Matrix, sy.ImmutableMatrix]:
        X = MatrixType([[sy.cos(t), -sy.sin(t)], [sy.sin(t), sy.cos(t)]])
        # 将 X 转换为 Aesara 代码表示的变量 tX
        tX = aesara_code_(X)
        # 断言 tX 是 TensorVariable 类型
        assert isinstance(tX, TensorVariable)
        # 断言 tX 的所有者操作是 Join
        assert isinstance(tX.owner.op, Join)


def test_cache_basic():
    """ Test single symbol-like objects are cached when printed by themselves. """

    # 定义符号对象对，这些对象在缓存时被视为等效
    pairs = [
        (x, sy.Symbol('x')),
        (X, sy.MatrixSymbol('X', *X.shape)),
        (f_t, sy.Function('f')(sy.Symbol('t'))),
    ]

    for s1, s2 in pairs:
        cache = {}
        # 获取 s1 的 Aesara 代码表示，并缓存到 cache 中
        st = aesara_code_(s1, cache=cache)

        # 测试缓存命中：相同实例
        assert aesara_code_(s1, cache=cache) is st

        # 测试缓存未命中：相同实例但新缓存
        assert aesara_code_(s1, cache={}) is not st

        # 测试缓存命中：不同但等效的实例
        assert aesara_code_(s2, cache=cache) is st

def test_global_cache():
    """ Test use of the global cache. """
    from sympy.printing.aesaracode import global_cache

    backup = dict(global_cache)
    try:
        # 暂时清空全局缓存
        global_cache.clear()

        # 对于每个对象 s，测试其 Aesara 代码表示是否被全局缓存
        for s in [x, X, f_t]:
            st = aesara_code(s)
            assert aesara_code(s) is st

    finally:
        # 恢复全局缓存
        global_cache.update(backup)

def test_cache_types_distinct():
    """
    Test that symbol-like objects of different types (Symbol, MatrixSymbol,
    AppliedUndef) are distinguished by the cache even if they have the same
    name.
    """
    symbols = [sy.Symbol('f_t'), sy.MatrixSymbol('f_t', 4, 4), f_t]  # 创建包含不同类型符号对象的列表

    cache = {}  # 单个共享缓存
    printed = {}  # 空字典用于存储打印结果

    for s in symbols:
        st = aesara_code_(s, cache=cache)  # 调用 aesara_code_ 函数生成表示符号 s 的代码字符串 st
        assert st not in printed.values()  # 断言生成的代码字符串 st 不在已打印结果的值中
        printed[s] = st  # 将生成的代码字符串 st 存储到 printed 字典中，以符号 s 作为键

    # 检查所有打印出的对象都是唯一的
    assert len(set(map(id, printed.values()))) == len(symbols)

    # 检查检索
    for s, st in printed.items():
        assert aesara_code(s, cache=cache) is st  # 断言调用 aesara_code 函数返回的结果与预期的代码字符串 st 相同
def test_symbols_are_created_once():
    """
    Test that a symbol is cached and reused when it appears in an expression
    more than once.
    """
    # 创建一个表达式，包含了符号 x 在 evaluate=False 的情况下加两次的结果
    expr = sy.Add(x, x, evaluate=False)
    # 将表达式转换为 Aesara 代码
    comp = aesara_code_(expr)

    # 断言：比较生成的 Aesara 代码是否等于两次 x 的和
    assert theq(comp, xt + xt)
    # 断言：生成的 Aesara 代码不等于 x 和 aesara_code_(x) 的和
    assert not theq(comp, xt + aesara_code_(x))


def test_cache_complex():
    """
    Test caching on a complicated expression with multiple symbols appearing
    multiple times.
    """
    # 创建一个复杂的表达式，包含多次出现的符号 x, y, z
    expr = x ** 2 + (y - sy.exp(x)) * sy.sin(z - x * y)
    # 获取表达式中的自由符号名称集合
    symbol_names = {s.name for s in expr.free_symbols}
    # 将表达式转换为 Aesara 代码
    expr_t = aesara_code_(expr)

    # 迭代 Aesara 计算图中与打印表达式相关的变量
    seen = set()
    for v in aesara.graph.basic.ancestors([expr_t]):
        # 对于没有所有者且不是常量的变量，应该是我们的符号
        if v.owner is None and not isinstance(v, aesara.graph.basic.Constant):
            # 断言：变量名在符号名称集合中
            assert v.name in symbol_names
            # 断言：变量名未在 seen 集合中出现过
            assert v.name not in seen
            seen.add(v.name)

    # 断言：检查所有的符号都已经出现过
    assert seen == symbol_names


def test_Piecewise():
    # 一个分段线性函数
    expr = sy.Piecewise((0, x<0), (x, x<2), (1, True))  # ___/III
    result = aesara_code_(expr)
    # 断言：结果的操作符应该是 aet.switch
    assert result.owner.op == aet.switch

    # 预期的 Aesara 表达式
    expected = aet.switch(xt<0, 0, aet.switch(xt<2, xt, 1))
    # 断言：比较生成的 Aesara 代码和预期的结果
    assert theq(result, expected)

    expr = sy.Piecewise((x, x < 0))
    result = aesara_code_(expr)
    expected = aet.switch(xt < 0, xt, np.nan)
    assert theq(result, expected)

    expr = sy.Piecewise((0, sy.And(x>0, x<2)), \
        (x, sy.Or(x>2, x<0)))
    result = aesara_code_(expr)
    expected = aet.switch(aet.and_(xt>0,xt<2), 0, \
        aet.switch(aet.or_(xt>2, xt<0), xt, np.nan))
    assert theq(result, expected)


def test_Relationals():
    assert theq(aesara_code_(sy.Eq(x, y)), aet.eq(xt, yt))
    # assert theq(aesara_code_(sy.Ne(x, y)), aet.neq(xt, yt))  # TODO - implement
    assert theq(aesara_code_(x > y), xt > yt)
    assert theq(aesara_code_(x < y), xt < yt)
    assert theq(aesara_code_(x >= y), xt >= yt)
    assert theq(aesara_code_(x <= y), xt <= yt)


def test_complexfunctions():
    dtypes = {x:'complex128', y:'complex128'}
    xt, yt = aesara_code(x, dtypes=dtypes), aesara_code(y, dtypes=dtypes)
    from sympy.functions.elementary.complexes import conjugate
    from aesara.tensor import as_tensor_variable as atv
    from aesara.tensor import complex as cplx
    assert theq(aesara_code(y*conjugate(x), dtypes=dtypes), yt*(xt.conj()))
    assert theq(aesara_code((1+2j)*x), xt*(atv(1.0)+atv(2.0)*cplx(0,1)))


def test_constantfunctions():
    tf = aesara_function([],[1+1j])
    assert(tf()==1+1j)
```