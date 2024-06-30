# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_applications.py`

```
# 导入必要的模块和函数
import tempfile  # 导入用于创建临时目录的模块
from sympy.external import import_module  # 导入用于动态导入模块的函数
from sympy.printing.codeprinter import ccode  # 导入用于生成C代码的函数
from sympy.utilities._compilation import compile_link_import_strings, has_c  # 导入编译和链接函数及C编译器检测函数
from sympy.utilities._compilation.util import may_xfail  # 导入用于标记预期失败的装饰器
from sympy.testing.pytest import skip  # 导入用于跳过测试的函数
from sympy.codegen.ast import (  # 导入抽象语法树节点类
    FunctionDefinition, FunctionPrototype, Variable, Pointer, real, Assignment,
    integer, CodeBlock, While
)
from sympy.codegen.cnodes import void, PreIncrement  # 导入C语言节点类和自增操作类
from sympy.codegen.cutils import render_as_source_file  # 导入将函数定义渲染为源文件的函数

cython = import_module('cython')  # 尝试导入Cython模块
np = import_module('numpy')  # 尝试导入NumPy模块

# 定义一个函数来生成一个函数定义对象
def _mk_func1():
    declars = n, inp, out = Variable('n', integer), Pointer('inp', real), Pointer('out', real)  # 定义变量声明
    i = Variable('i', integer)  # 定义整型变量i
    whl = While(i<n, [Assignment(out[i], inp[i]), PreIncrement(i)])  # 定义一个while循环节点对象
    body = CodeBlock(i.as_Declaration(value=0), whl)  # 定义一个代码块对象，包含变量声明和while循环
    return FunctionDefinition(void, 'our_test_function', declars, body)  # 返回一个void类型的函数定义对象

# 定义一个函数，渲染函数定义对象并进行编译和导入
def _render_compile_import(funcdef, build_dir):
    code_str = render_as_source_file(funcdef, settings={"contract": False})  # 渲染函数定义对象为源代码字符串
    declar = ccode(FunctionPrototype.from_FunctionDefinition(funcdef))  # 生成函数原型的C代码字符串
    # 编译和链接源代码字符串，并生成需要导入的字符串列表
    return compile_link_import_strings([
        ('our_test_func.c', code_str),  # C源文件名及对应代码字符串
        ('_our_test_func.pyx', ("#cython: language_level={}\n".format("3") +
                                "cdef extern {declar}\n"
                                "def _{fname}({typ}[:] inp, {typ}[:] out):\n"
                                "    {fname}(inp.size, &inp[0], &out[0])").format(
                                    declar=declar, fname=funcdef.name, typ='double'
                                ))  # Cython源文件名及对应代码字符串
    ], build_dir=build_dir)  # 编译链接并导入字符串列表，指定构建目录

# 标记该测试函数可能会预期失败
@may_xfail
def test_copying_function():
    if not np:
        skip("numpy not installed.")  # 如果NumPy未安装，则跳过测试
    if not has_c():
        skip("No C compiler found.")  # 如果没有找到C编译器，则跳过测试
    if not cython:
        skip("Cython not found.")  # 如果没有找到Cython，则跳过测试

    info = None  # 初始化info变量为None
    with tempfile.TemporaryDirectory() as folder:  # 使用临时目录
        mod, info = _render_compile_import(_mk_func1(), build_dir=folder)  # 渲染函数并编译导入
        inp = np.arange(10.0)  # 创建一个NumPy数组inp
        out = np.empty_like(inp)  # 创建一个和inp相同形状的空NumPy数组out
        mod._our_test_function(inp, out)  # 调用编译后的函数
        assert np.allclose(inp, out)  # 断言inp和out数组是否相等
```