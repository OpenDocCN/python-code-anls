# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_fnodes.py`

```
# 导入所需的模块和库
import os  # 导入操作系统相关功能的模块
import tempfile  # 导入处理临时文件和目录的模块
from sympy.core.symbol import (Symbol, symbols)  # 导入符号相关的类和函数
from sympy.codegen.ast import (  # 导入代码生成中的抽象语法树相关类和函数
    Assignment, Print, Declaration, FunctionDefinition, Return, real,
    FunctionCall, Variable, Element, integer
)
from sympy.codegen.fnodes import (  # 导入Fortran节点相关的类和函数
    allocatable, ArrayConstructor, isign, dsign, cmplx, kind, literal_dp,
    Program, Module, use, Subroutine, dimension, assumed_extent, ImpliedDoLoop,
    intent_out, size, Do, SubroutineCall, sum_, array, bind_C
)
from sympy.codegen.futils import render_as_module  # 导入用于生成Fortran模块的函数
from sympy.core.expr import unchanged  # 导入符号表达式相关的类和函数
from sympy.external import import_module  # 导入外部模块的导入函数
from sympy.printing.codeprinter import fcode  # 导入用于生成Fortran代码的函数
from sympy.utilities._compilation import (  # 导入用于编译运行Fortran代码的函数
    has_fortran, compile_run_strings, compile_link_import_strings
)
from sympy.utilities._compilation.util import may_xfail  # 导入用于测试中的装饰器和函数
from sympy.testing.pytest import skip, XFAIL  # 导入用于测试的装饰器和函数

# 尝试导入Cython和NumPy模块
cython = import_module('cython')
np = import_module('numpy')


# 定义测试函数test_size
def test_size():
    # 创建一个实数符号变量x
    x = Symbol('x', real=True)
    # 计算变量x的大小
    sx = size(x)
    # 断言生成的Fortran代码与预期的相符
    assert fcode(sx, source_format='free') == 'size(x)'


# 用may_xfail装饰的测试函数test_size_assumed_shape
@may_xfail
def test_size_assumed_shape():
    # 如果没有Fortran编译器，则跳过测试
    if not has_fortran():
        skip("No fortran compiler found.")
    # 创建一个实数符号变量a
    a = Symbol('a', real=True)
    # 定义函数体，计算根均方值并返回
    body = [Return((sum_(a**2)/size(a))**.5)]
    # 创建一个数组对象arr，标识其为输入
    arr = array(a, dim=[':'], intent='in')
    # 定义函数rms的函数定义
    fd = FunctionDefinition(real, 'rms', [arr], body)
    # 将函数定义渲染为Fortran模块
    render_as_module([fd], 'mod_rms')

    # 编译和运行Fortran字符串
    (stdout, stderr), info = compile_run_strings([
        ('rms.f90', render_as_module([fd], 'mod_rms')),
        ('main.f90', (
            'program myprog\n'
            'use mod_rms, only: rms\n'
            'real*8, dimension(4), parameter :: x = [4, 2, 2, 2]\n'
            'print *, dsqrt(7d0) - rms(x)\n'
            'end program\n'
        ))
    ], clean=True)
    # 断言输出的stdout包含预期结果
    assert '0.00000' in stdout
    assert stderr == ''
    assert info['exit_status'] == os.EX_OK


# 用may_xfail和XFAIL装饰的测试函数test_ImpliedDoLoop
@XFAIL  # https://github.com/sympy/sympy/issues/20265
@may_xfail
def test_ImpliedDoLoop():
    # 如果没有Fortran编译器，则跳过测试
    if not has_fortran():
        skip("No fortran compiler found.")

    # 创建整数符号变量a和循环索引变量i
    a, i = symbols('a i', integer=True)
    # 创建隐含循环对象idl，计算i的立方并返回
    idl = ImpliedDoLoop(i**3, i, -3, 3, 2)
    # 创建数组构造器对象ac，包含静态和循环产生的元素
    ac = ArrayConstructor([-28, idl, 28])
    # 创建一个可分配的数组a
    a = array(a, dim=[':'], attrs=[allocatable])
    # 创建一个程序prog，包含声明、赋值和打印操作
    prog = Program('idlprog', [
        a.as_Declaration(),
        Assignment(a, ac),
        Print([a])
    ])
    # 将程序对象转换为Fortran代码字符串
    fsrc = fcode(prog, standard=2003, source_format='free')
    # 编译和运行Fortran字符串
    (stdout, stderr), info = compile_run_strings([('main.f90', fsrc)], clean=True)
    # 断言输出的stdout包含预期结果
    for numstr in '-28 -27 -1 1 27 28'.split():
        assert numstr in stdout
    assert stderr == ''
    assert info['exit_status'] == os.EX_OK


# 用may_xfail装饰的测试函数test_Program
@may_xfail
def test_Program():
    # 创建一个实数符号变量x
    x = Symbol('x', real=True)
    # 创建一个推断变量vx，值为42
    vx = Variable.deduced(x, 42)
    # 创建变量vx的声明
    decl = Declaration(vx)
    # 创建打印语句prnt，打印x和x+1的值
    prnt = Print([x, x+1])
    # 创建一个程序prog，包含声明和打印操作
    prog = Program('foo', [decl, prnt])
    # 如果没有Fortran编译器，则跳过测试
    if not has_fortran():
        skip("No fortran compiler found.")

    # 编译和运行Fortran字符串
    (stdout, stderr), info = compile_run_strings([('main.f90', fcode(prog, standard=90))], clean=True)
    # 断言输出的stdout包含预期结果
    assert '42' in stdout
    assert '43' in stdout
    # 断言标准错误输出为空字符串
    assert stderr == ''
    # 断言信息字典中的退出状态为操作系统定义的正常退出状态
    assert info['exit_status'] == os.EX_OK
@may_xfail
# 声明一个测试函数，标记为可能会失败的测试用例
def test_Module():
    # 声明一个实数符号变量 'x'
    x = Symbol('x', real=True)
    # 使用变量 'x' 创建一个推导的变量 'v_x'
    v_x = Variable.deduced(x)
    # 定义一个实数函数 'sqr'，接受 'v_x' 作为参数，返回 x 的平方
    sq = FunctionDefinition(real, 'sqr', [v_x], [Return(x**2)])
    # 创建一个模块 'mod_sq'，包含前面定义的函数 'sq'
    mod_sq = Module('mod_sq', [], [sq])
    # 调用函数 'sqr'，参数是 42.0
    sq_call = FunctionCall('sqr', [42.])
    # 创建一个程序 'foobar'
    prg_sq = Program('foobar', [
        # 导入模块 'mod_sq'，但只使用 'sqr' 函数
        use('mod_sq', only=['sqr']),
        # 打印字符串和调用 'sqr(42.0)' 的结果
        Print(['"Square of 42 = "', sq_call])
    ])
    # 如果没有 Fortran 编译器，则跳过测试
    if not has_fortran():
        skip("No fortran compiler found.")
    # 编译并运行两个字符串，一个包含 'mod_sq' 的 Fortran 代码，另一个包含 'prg_sq' 的 Fortran 代码
    (stdout, stderr), info = compile_run_strings([
        ('mod_sq.f90', fcode(mod_sq, standard=90)),
        ('main.f90', fcode(prg_sq, standard=90))
    ], clean=True)
    # 断言在输出中找到字符串 '42'
    assert '42' in stdout
    # 断言在输出中找到字符串 '1764'（42 的平方）
    assert str(42**2) in stdout
    # 断言 stderr 为空
    assert stderr == ''


@XFAIL  # https://github.com/sympy/sympy/issues/20265
@may_xfail
# 声明一个测试函数，标记为可能会失败的测试用例
def test_Subroutine():
    # 声明一个实数符号变量 'r'
    r = Symbol('r', real=True)
    # 声明一个整数符号变量 'i'
    i = Symbol('i', integer=True)
    # 使用变量 'r' 创建一个推导的变量 'v_r'，带有属性：维度为未知范围，输出意图
    v_r = Variable.deduced(r, attrs=(dimension(assumed_extent), intent_out))
    # 使用变量 'i' 创建一个推导的变量 'v_i'
    v_i = Variable.deduced(i)
    # 声明一个整数变量 'v_n'，名称为 'n'
    v_n = Variable('n', integer)
    # 创建一个循环 'do_loop'，将 r[i] 赋值为 1/i**2
    do_loop = Do([
        Assignment(Element(r, [i]), literal_dp(1)/i**2)
    ], i, 1, v_n)
    # 定义一个子程序 'f'，接受 'v_r' 作为参数，包含声明 'v_n' 和 'v_i'，以及 'do_loop'
    sub = Subroutine("f", [v_r], [
        Declaration(v_n),
        Declaration(v_i),
        Assignment(v_n, size(r)),
        do_loop
    ])
    # 声明一个实数符号变量 'x'
    x = Symbol('x', real=True)
    # 使用变量 'x' 创建一个推导的变量 'v_x3'，带有属性：维度为3
    v_x3 = Variable.deduced(x, attrs=[dimension(3)])
    # 创建一个模块 'mymod'，包含前面定义的子程序 'sub'
    mod = Module('mymod', definitions=[sub])
    # 创建一个程序 'foo'
    prog = Program('foo', [
        # 导入模块 'mod'，但只使用子程序 'sub'
        use(mod, only=[sub]),
        # 声明变量 'v_x3'
        Declaration(v_x3),
        # 调用子程序 'sub'，参数是 'v_x3'
        SubroutineCall(sub, [v_x3]),
        # 打印 'v_x3' 的总和和它本身
        Print([sum_(v_x3), v_x3])
    ])
    # 如果没有 Fortran 编译器，则跳过测试
    if not has_fortran():
        skip("No fortran compiler found.")
    # 编译并运行两个字符串，一个包含 'mod' 的 Fortran 代码，另一个包含 'prog' 的 Fortran 代码
    (stdout, stderr), info = compile_run_strings([
        ('a.f90', fcode(mod, standard=90)),
        ('b.f90', fcode(prog, standard=90))
    ], clean=True)
    # 计算参考值 'ref'，即列表 [1.0/i**2 for i in range(1, 4)]
    ref = [1.0/i**2 for i in range(1, 4)]
    # 断言在输出中找到 'ref' 的总和的近似值
    assert str(sum(ref))[:-3] in stdout
    # 对于 'ref' 中的每个值，断言其近似值在输出中
    for _ in ref:
        assert str(_)[:-3] in stdout
    # 断言 stderr 为空
    assert stderr == ''


# 声明一个测试函数，测试函数 'isign'
def test_isign():
    # 声明一个整数符号变量 'x'
    x = Symbol('x', integer=True)
    # 断言 isign(1, x) 返回的值不变
    assert unchanged(isign, 1, x)
    # 断言将 isign(1, x) 转换成标准 95 年的自由格式 Fortran 代码等于 'isign(1, x)'
    assert fcode(isign(1, x), standard=95, source_format='free') == 'isign(1, x)'


# 声明一个测试函数，测试函数 'dsign'
def test_dsign():
    # 声明一个符号变量 'x'
    x = Symbol('x')
    # 断言 dsign(1, x) 返回的值不变
    assert unchanged(dsign, 1, x)
    # 断言将 dsign(literal_dp(1), x) 转换成标准 95 年的自由格式 Fortran 代码等于 'dsign(1d0, x)'
    assert fcode(dsign(literal_dp(1), x), standard=95, source_format='free') == 'dsign(1d0, x)'


# 声明一个测试函数，测试函数 'cmplx'
def test_cmplx():
    # 声明一个符号变量 'x'
    x = Symbol('x')
    # 断言 cmplx(1, x) 返回的值不变
    assert unchanged(cmplx, 1, x)


# 声明一个测试函数，测试函数 'kind'
def test_kind():
    # 声明一个符号变量 'x'
    x = Symbol('x')
    # 断言 kind(x) 返回的值不变
    assert unchanged(kind, x)


# 声明一个测试函数，测试函数 'literal_dp'
def test_literal_dp():
    # 断言将 literal_dp(0) 转换成自由格式 Fortran 代码等于 '0d0'
    assert fcode(literal_dp(0), source_format='free') == '0d0'


@may_xfail
# 声明一个测试函数，标记为可能会失败的测试用例
def test_bind_C():
    # 如果没有 Fortran 编译器，则跳过测试
    if not has_fortran():
        skip("No fortran compiler found.")
    # 如果没有 Cython，则跳过测试
    if not cython:
        skip("Cython not found.")
    # 如果没有 NumPy，则跳过测试
    if not np:
        skip("NumPy not found.")

    # 声明一个实数符号变量 'a'
    a = Symbol('a', real=True)
    # 声明一个整数符号变量 's'
    s = Symbol('s', integer=True)
    # 创建一个函数体，返回 (sum_(a**2)/s)**.5
    body = [Return((
    # 创建一个函数定义对象，表示一个名为 'rms' 的函数，接受参数 [arr, s]，具有指定的主体和属性列表
    fd = FunctionDefinition(real, 'rms', [arr, s], body, attrs=[bind_C('rms')])
    
    # 将函数定义对象列表渲染为一个模块 'mod_rms'
    f_mod = render_as_module([fd], 'mod_rms')
    
    # 使用临时目录作为编译和链接的工作目录
    with tempfile.TemporaryDirectory() as folder:
        # 编译、链接和导入字符串内容列表，返回编译后的模块及相关信息
        mod, info = compile_link_import_strings([
            ('rms.f90', f_mod),  # 将 'f_mod' 写入 'rms.f90' 文件
            ('_rms.pyx', (
                "#cython: language_level={}\n".format("3") +  # 设置 Cython 的语言级别为 3
                "cdef extern double rms(double*, int*)\n"  # 声明外部函数 rms
                "def py_rms(double[::1] x):\n"  # 定义名为 py_rms 的 Cython 函数
                "    cdef int s = x.size\n"  # 声明并初始化变量 s 为数组 x 的大小
                "    return rms(&x[0], &s)\n"))  # 调用外部函数 rms，传入数组首地址和大小的指针
        ], build_dir=folder)
    
        # 断言验证 mod.py_rms(np.array([2., 4., 2., 2.])) 的返回值接近于 7 的平方根
        assert abs(mod.py_rms(np.array([2., 4., 2., 2.])) - 7**0.5) < 1e-14
```