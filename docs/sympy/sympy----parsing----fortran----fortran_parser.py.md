# `D:\src\scipysrc\sympy\sympy\parsing\fortran\fortran_parser.py`

```
from sympy.external import import_module
# 从 sympy.external 模块导入 import_module 函数

lfortran = import_module('lfortran')
# 尝试导入 lfortran 模块，如果导入成功，lfortran 将被赋予模块对象，否则为 None

if lfortran:
    # 如果成功导入 lfortran 模块，则执行以下代码块

    from sympy.codegen.ast import (Variable, IntBaseType, FloatBaseType, String,
                                   Return, FunctionDefinition, Assignment)
    # 从 sympy.codegen.ast 模块导入多个类（Variable, IntBaseType 等）

    from sympy.core import Add, Mul, Integer, Float
    # 从 sympy.core 模块导入 Add, Mul, Integer, Float 类

    from sympy.core.symbol import Symbol
    # 从 sympy.core.symbol 模块导入 Symbol 类

    asr_mod = lfortran.asr
    # 将 lfortran.asr 赋值给 asr_mod

    asr = lfortran.asr.asr
    # 将 lfortran.asr.asr 赋值给 asr

    src_to_ast = lfortran.ast.src_to_ast
    # 将 lfortran.ast.src_to_ast 赋值给 src_to_ast

    ast_to_asr = lfortran.semantic.ast_to_asr.ast_to_asr
    # 将 lfortran.semantic.ast_to_asr.ast_to_asr 赋值给 ast_to_asr

else:
    # 如果未能导入 lfortran 模块，则执行以下代码块

    class ASR2PyVisitor():  # type: ignore
        # 定义 ASR2PyVisitor 类，用于处理 lfortran 不可用的情况
        def __init__(self, *args, **kwargs):
            raise ImportError('lfortran not available')
            # 抛出 ImportError 异常，指示 lfortran 不可用

def call_visitor(fort_node):
    """Calls the AST Visitor on the Module

    This function is used to call the AST visitor for a program or module
    It imports all the required modules and calls the visit() function
    on the given node

    Parameters
    ==========

    fort_node : LFortran ASR object
        Node for the operation for which the NodeVisitor is called

    Returns
    =======

    res_ast : list
        list of SymPy AST Nodes

    """
    v = ASR2PyVisitor()
    # 创建 ASR2PyVisitor 的实例 v
    v.visit(fort_node)
    # 调用 v 的 visit 方法，处理 fort_node
    res_ast = v.ret_ast()
    # 获取处理结果并赋值给 res_ast
    return res_ast
    # 返回处理结果的 SymPy AST 节点列表

def src_to_sympy(src):
    """Wrapper function to convert the given Fortran source code to SymPy Expressions

    Parameters
    ==========

    src : string
        A string with the Fortran source code

    Returns
    =======

    py_src : string
        A string with the Python source code compatible with SymPy

    """
    a_ast = src_to_ast(src, translation_unit=False)
    # 将给定的 Fortran 源代码转换为 AST
    a = ast_to_asr(a_ast)
    # 将 AST 转换为 LFortran ASR
    py_src = call_visitor(a)
    # 调用 call_visitor 处理 ASR，并将结果赋值给 py_src
    return py_src
    # 返回转换后的 SymPy 兼容的 Python 源代码字符串
```