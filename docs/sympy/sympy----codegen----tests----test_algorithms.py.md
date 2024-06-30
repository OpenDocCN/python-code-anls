# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_algorithms.py`

```
import tempfile
from sympy import log, Min, Max, sqrt  # 导入对数、最小值、最大值、平方根函数
from sympy.core.numbers import Float  # 导入浮点数类
from sympy.core.symbol import Symbol, symbols  # 导入符号、符号列表
from sympy.functions.elementary.trigonometric import cos  # 导入余弦函数
from sympy.codegen.ast import Assignment, Raise, RuntimeError_, QuotedString  # 导入语法树节点类型
from sympy.codegen.algorithms import newtons_method, newtons_method_function  # 导入牛顿法算法及函数生成函数
from sympy.codegen.cfunctions import expm1  # 导入C函数
from sympy.codegen.fnodes import bind_C  # 导入绑定C语言节点
from sympy.codegen.futils import render_as_module as f_module  # 导入用于生成模块的函数（Fortran）
from sympy.codegen.pyutils import render_as_module as py_module  # 导入用于生成模块的函数（Python）
from sympy.external import import_module  # 导入模块导入函数
from sympy.printing.codeprinter import ccode  # 导入C代码打印器
from sympy.utilities._compilation import compile_link_import_strings, has_c, has_fortran  # 导入编译及检测C和Fortran编译器的函数
from sympy.utilities._compilation.util import may_xfail  # 导入测试装饰器
from sympy.testing.pytest import skip, raises  # 导入测试辅助函数

cython = import_module('cython')  # 导入Cython模块
wurlitzer = import_module('wurlitzer')  # 导入wurlitzer模块

def test_newtons_method():
    x, dx, atol = symbols('x dx atol')  # 声明符号变量
    expr = cos(x) - x**3  # 定义表达式
    algo = newtons_method(expr, x, atol, dx)  # 使用牛顿法求解表达式
    assert algo.has(Assignment(dx, -expr/expr.diff(x)))  # 断言表达式的导数分配

@may_xfail  # 标记为可能失败的测试
def test_newtons_method_function__ccode():
    x = Symbol('x', real=True)  # 声明实数符号变量x
    expr = cos(x) - x**3  # 定义表达式
    func = newtons_method_function(expr, x)  # 使用表达式生成函数

    if not cython:
        skip("cython not installed.")  # 如果没有安装Cython，跳过测试
    if not has_c():
        skip("No C compiler found.")  # 如果没有C编译器，跳过测试

    compile_kw = {"std": 'c99'}  # 编译选项
    with tempfile.TemporaryDirectory() as folder:  # 创建临时文件夹
        mod, info = compile_link_import_strings([
            ('newton.c', ('#include <math.h>\n'
                          '#include <stdio.h>\n') + ccode(func)),  # 编译C代码
            ('_newton.pyx', ("#cython: language_level={}\n".format("3") +
                             "cdef extern double newton(double)\n"
                             "def py_newton(x):\n"
                             "    return newton(x)\n"))  # 编译Cython代码
        ], build_dir=folder, compile_kwargs=compile_kw)  # 编译及链接字符串

        assert abs(mod.py_newton(0.5) - 0.865474033102) < 1e-12  # 断言计算结果精度

@may_xfail  # 标记为可能失败的测试
def test_newtons_method_function__fcode():
    x = Symbol('x', real=True)  # 声明实数符号变量x
    expr = cos(x) - x**3  # 定义表达式
    func = newtons_method_function(expr, x, attrs=[bind_C(name='newton')])  # 使用表达式生成函数，绑定C语言

    if not cython:
        skip("cython not installed.")  # 如果没有安装Cython，跳过测试
    if not has_fortran():
        skip("No Fortran compiler found.")  # 如果没有Fortran编译器，跳过测试

    f_mod = f_module([func], 'mod_newton')  # 生成Fortran模块
    with tempfile.TemporaryDirectory() as folder:  # 创建临时文件夹
        mod, info = compile_link_import_strings([
            ('newton.f90', f_mod),  # 编译Fortran代码
            ('_newton.pyx', ("#cython: language_level={}\n".format("3") +
                             "cdef extern double newton(double*)\n"
                             "def py_newton(double x):\n"
                             "    return newton(&x)\n"))  # 编译Cython代码
        ], build_dir=folder)  # 编译及链接字符串

        assert abs(mod.py_newton(0.5) - 0.865474033102) < 1e-12  # 断言计算结果精度

def test_newtons_method_function__pycode():
    x = Symbol('x', real=True)  # 声明实数符号变量x
    expr = cos(x) - x**3  # 定义表达式
    func = newtons_method_function(expr, x)  # 使用表达式生成函数
    # 调用py_module函数，返回一个Python模块对象py_mod，该模块包含func函数的定义或实现
    py_mod = py_module(func)
    
    # 创建一个空的命名空间字典namespace，用于执行动态生成的Python代码
    namespace = {}
    
    # 在命名空间namespace中执行py_mod字符串表示的Python代码，将代码中定义的变量和函数注册到namespace中
    exec(py_mod, namespace, namespace)
    
    # 在命名空间namespace中使用eval函数执行字符串'newton(0.5)'，返回表达式计算结果res
    res = eval('newton(0.5)', namespace)
    
    # 断言语句，检查res与期望值0.865474033102的绝对误差是否小于1e-12，若不满足则会引发AssertionError
    assert abs(res - 0.865474033102) < 1e-12
@may_xfail
# 定义一个测试函数，用于测试牛顿法函数的参数化处理
def test_newtons_method_function__ccode_parameters():
    # 定义符号变量 x, A, k, p，并将它们作为 args 元组
    args = x, A, k, p = symbols('x A k p')
    # 定义表达式 expr，这里是一个包含余弦和立方项的复杂数学表达式
    expr = A*cos(k*x) - p*x**3
    # 使用 newtons_method_function 函数预期引发 ValueError 异常
    raises(ValueError, lambda: newtons_method_function(expr, x))
    # 将 wurlitzer 模块引入并赋值给 use_wurlitzer 变量
    use_wurlitzer = wurlitzer

    # 调用 newtons_method_function 函数，传入表达式 expr 和符号变量 x，args 作为参数，设置调试标志为 use_wurlitzer
    func = newtons_method_function(expr, x, args, debug=use_wurlitzer)

    # 如果没有 C 编译器，则跳过测试
    if not has_c():
        skip("No C compiler found.")
    # 如果没有安装 cython，则跳过测试
    if not cython:
        skip("cython not installed.")

    # 设置编译选项
    compile_kw = {"std": 'c99'}
    # 使用临时目录创建临时文件夹
    with tempfile.TemporaryDirectory() as folder:
        # 编译、链接并导入字符串代码块
        mod, info = compile_link_import_strings([
            ('newton_par.c', ('#include <math.h>\n'
                          '#include <stdio.h>\n') + ccode(func)),
            ('_newton_par.pyx', ("#cython: language_level={}\n".format("3") +
                                 "cdef extern double newton(double, double, double, double)\n"
                             "def py_newton(x, A=1, k=1, p=1):\n"
                             "    return newton(x, A, k, p)\n"))
        ], compile_kwargs=compile_kw, build_dir=folder)

        # 如果使用了 wurlitzer，则捕获标准输出和标准错误
        if use_wurlitzer:
            with wurlitzer.pipes() as (out, err):
                result = mod.py_newton(0.5)
        else:
            result = mod.py_newton(0.5)

        # 断言计算结果与期望值之间的误差小于给定的容差
        assert abs(result - 0.865474033102) < 1e-12

        # 如果未使用 wurlitzer，则跳过测试
        if not use_wurlitzer:
            skip("C-level output only tested when package 'wurlitzer' is available.")

        # 读取标准输出和标准错误
        out, err = out.read(), err.read()
        # 断言标准错误为空
        assert err == ''
        # 断言标准输出与预期输出一致
        assert out == """\
x=         0.5
x=      1.1121 d_x=     0.61214
x=     0.90967 d_x=    -0.20247
x=     0.86726 d_x=   -0.042409
x=     0.86548 d_x=  -0.0017867
x=     0.86547 d_x= -3.1022e-06
x=     0.86547 d_x= -9.3421e-12
x=     0.86547 d_x=  3.6902e-17
"""  # try to run tests with LC_ALL=C if this assertion fails


def test_newtons_method_function__rtol_cse_nan():
    # 定义符号变量 a, b, c, N_geo, N_tot，并指定它们为实数和非负数
    a, b, c, N_geo, N_tot = symbols('a b c N_geo N_tot', real=True, nonnegative=True)
    # 定义整数符号变量 i，并指定它为非负整数
    i = Symbol('i', integer=True, nonnegative=True)
    # 计算 N_ari 和 delta_ari 的值
    N_ari = N_tot - N_geo - 1
    delta_ari = (c-b)/N_ari
    # 计算 ln_delta_geo 的值
    ln_delta_geo = log(b) + log(-expm1((log(a)-log(b))/N_geo))
    # 计算 eqb_log 的值，这是 ln_delta_geo 减去 delta_ari 的自然对数
    eqb_log = ln_delta_geo - log(delta_ari)

    # 定义一个辅助函数 _clamp，用于将表达式 expr 在指定范围内夹紧
    def _clamp(low, expr, high):
        return Min(Max(low, expr), high)

    # 定义一个包含不同牛顿法变体的字典 meth_kw
    meth_kw = {
        'clamped_newton': {'delta_fn': lambda e, x: _clamp(
            (sqrt(a*x)-x)*0.99,
            -e/e.diff(x),
            (sqrt(c*x)-x)*0.99
        )},
        'halley': {'delta_fn': lambda e, x: (-2*(e*e.diff(x))/(2*e.diff(x)**2 - e*e.diff(x, 2)))},
        'halley_alt': {'delta_fn': lambda e, x: (-e/e.diff(x)/(1-e/e.diff(x)*e.diff(x,2)/2/e.diff(x)))},
    }
    # 将 eqb_log 和 b 作为参数 args 返回
    args = eqb_log, b
    # 遍历使用两种不同的控制结构消除技术（CSE）的布尔值：False 和 True
    for use_cse in [False, True]:
        # 设置关键字参数 kwargs，包括参数元组 (b, a, c, N_geo, N_tot)，最大迭代次数 itermax 为 60，启用调试模式 debug 为 True
        # cse 参数根据当前循环的 use_cse 值设置
        kwargs = {
            'params': (b, a, c, N_geo, N_tot), 'itermax': 60, 'debug': True, 'cse': use_cse,
            'counter': i, 'atol': 1e-100, 'rtol': 2e-16, 'bounds': (a,c),
            'handle_nan': Raise(RuntimeError_(QuotedString("encountered NaN.")))
        }
        # 使用 newtons_method_function 函数为每个方法名称生成函数对象，并传入参数 kwargs 和 meth_kw 中对应的关键字参数
        func = {k: newtons_method_function(*args, func_name=f"{k}_b", **dict(kwargs, **kw)) for k, kw in meth_kw.items()}
        # 将每个函数对象转换为对应的 Python 模块对象
        py_mod = {k: py_module(v) for k, v in func.items()}
        # 初始化命名空间字典
        namespace = {}
        root_find_b = {}
        # 遍历每个 Python 模块对象
        for k, v in py_mod.items():
            # 为当前模块创建一个空的命名空间字典 ns，并在此命名空间中执行模块内容
            ns = namespace[k] = {}
            exec(v, ns, ns)
            # 从当前命名空间中获取函数对象并存储到 root_find_b 字典中
            root_find_b[k] = ns[f'{k}_b']
        # 参考值 ref 为 Float('13.2261515064168768938151923226496')
        ref = Float('13.2261515064168768938151923226496')
        # 定义不同方法的相对误差容限 reftol
        reftol = {'clamped_newton': 2e-16, 'halley': 2e-16, 'halley_alt': 3e-16}
        # 初始猜测值 guess 为 4.0
        guess = 4.0
        # 对每种方法和对应的函数执行根查找，并进行断言检查结果精度
        for meth, func in root_find_b.items():
            # 调用 func 函数进行根查找，传入猜测值 guess 和其他参数
            result = func(guess, 1e-2, 1e2, 50, 100)
            # 计算所需精度 req，基于参考值 ref 和 reftol 中指定的相对容限
            req = ref * reftol[meth]
            # 如果 use_cse 为 True，则将 req 加倍，用于检查使用 CSE 的情况
            if use_cse:
                req *= 2
            # 使用断言确保计算结果的绝对误差小于 req
            assert abs(result - ref) < req
```