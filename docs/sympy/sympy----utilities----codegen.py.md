# `D:\src\scipysrc\sympy\sympy\utilities\codegen.py`

```
"""
module for generating C, C++, Fortran77, Fortran90, Julia, Rust
and Octave/Matlab routines that evaluate SymPy expressions.
This module is work in progress.
Only the milestones with a '+' character in the list below have been completed.

--- How is sympy.utilities.codegen different from sympy.printing.ccode? ---

We considered the idea to extend the printing routines for SymPy functions in
such a way that it prints complete compilable code, but this leads to a few
unsurmountable issues that can only be tackled with dedicated code generator:

- For C, one needs both a code and a header file, while the printing routines
  generate just one string. This code generator can be extended to support
  .pyf files for f2py.

- SymPy functions are not concerned with programming-technical issues, such
  as input, output and input-output arguments. Other examples are contiguous
  or non-contiguous arrays, including headers of other libraries such as gsl
  or others.

- It is highly interesting to evaluate several SymPy functions in one C
  routine, eventually sharing common intermediate results with the help
  of the cse routine. This is more than just printing.

- From the programming perspective, expressions with constants should be
  evaluated in the code generator as much as possible. This is different
  for printing.

--- Basic assumptions ---

* A generic Routine data structure describes the routine that must be
  translated into C/Fortran/... code. This data structure covers all
  features present in one or more of the supported languages.

* Descendants from the CodeGen class transform multiple Routine instances
  into compilable code. Each derived class translates into a specific
  language.

* In many cases, one wants a simple workflow. The friendly functions in the
  last part are a simple api on top of the Routine/CodeGen stuff. They are
  easier to use, but are less powerful.

--- Milestones ---

+ First working version with scalar input arguments, generating C code,
  tests
+ Friendly functions that are easier to use than the rigorous
  Routine/CodeGen workflow.
+ Integer and Real numbers as input and output
+ Output arguments
+ InputOutput arguments
+ Sort input/output arguments properly
+ Contiguous array arguments (numpy matrices)
+ Also generate .pyf code for f2py (in autowrap module)
+ Isolate constants and evaluate them beforehand in double precision
+ Fortran 90
+ Octave/Matlab

- Common Subexpression Elimination
- User defined comments in the generated code
- Optional extra include lines for libraries/objects that can eval special
  functions
- Test other C compilers and libraries: gcc, tcc, libtcc, gcc+gsl, ...
- Contiguous array arguments (SymPy matrices)
- Non-contiguous array arguments (SymPy matrices)
- ccode must raise an error when it encounters something that cannot be
  translated into c. ccode(integrate(sin(x)/x, x)) does not make sense.
- Complex numbers as input and output
- A default complex datatype
"""
# 导入必要的模块和函数
import os                  # 系统操作相关模块
import textwrap            # 文本包装模块
from io import StringIO    # 字符串IO操作相关模块

# 导入 sympy 相关模块和函数
from sympy import __version__ as sympy_version   # 导入 sympy 的版本信息
from sympy.core import Symbol, S, Tuple, Equality, Function, Basic   # 导入 sympy 的核心类和函数
from sympy.printing.c import c_code_printers      # 导入 sympy 的 C 代码打印器
from sympy.printing.codeprinter import AssignmentError   # 导入 sympy 的代码打印错误类
from sympy.printing.fortran import FCodePrinter    # 导入 sympy 的 Fortran 代码打印器
from sympy.printing.julia import JuliaCodePrinter  # 导入 sympy 的 Julia 代码打印器
from sympy.printing.octave import OctaveCodePrinter   # 导入 sympy 的 Octave 代码打印器
from sympy.printing.rust import RustCodePrinter    # 导入 sympy 的 Rust 代码打印器
from sympy.tensor import Idx, Indexed, IndexedBase   # 导入 sympy 的张量相关类
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
                            MatrixExpr, MatrixSlice)   # 导入 sympy 的矩阵相关类
from sympy.utilities.iterables import is_sequence   # 导入 sympy 的迭代工具函数

# 导出给外部使用的类和函数列表
__all__ = [
    # description of routines
    "Routine", "DataType", "default_datatypes", "get_default_datatype",
    "Argument", "InputArgument", "OutputArgument", "Result",
    # routines -> code
    "CodeGen", "CCodeGen", "FCodeGen", "JuliaCodeGen", "OctaveCodeGen",
    "RustCodeGen",
    # friendly functions
    "codegen", "make_routine",
]

#
# Description of routines
#

# 定义描述评估程序集合表达式的通用类
class Routine:
    """Generic description of evaluation routine for set of expressions.

    A CodeGen class can translate instances of this class into code in a
    particular language.  The routine specification covers all the features
    present in these languages.  The CodeGen part must raise an exception
    when certain features are not present in the target language.  For
    example, multiple return values are possible in Python, but not in C or
    Fortran.  Another example: Fortran and Python support complex numbers,
    while C does not.

    """
    def __init__(self, name, arguments, results, local_vars, global_vars):
        """Initialize a Routine instance.

        Parameters
        ==========

        name : string
            Name of the routine.

        arguments : list of Arguments
            These are things that appear in arguments of a routine, often
            appearing on the right-hand side of a function call.  These are
            commonly InputArguments but in some languages, they can also be
            OutputArguments or InOutArguments (e.g., pass-by-reference in C
            code).

        results : list of Results
            These are the return values of the routine, often appearing on
            the left-hand side of a function call.  The difference between
            Results and OutputArguments and when you should use each is
            language-specific.

        local_vars : list of Results
            These are variables that will be defined at the beginning of the
            function.

        global_vars : list of Symbols
            Variables which will not be passed into the function.

        """

        # extract all input symbols and all symbols appearing in an expression
        input_symbols = set()
        symbols = set()

        # Iterate over each argument to categorize symbols based on argument type
        for arg in arguments:
            if isinstance(arg, OutputArgument):
                # Update symbols with free symbols in the expression, excluding Indexed symbols
                symbols.update(arg.expr.free_symbols - arg.expr.atoms(Indexed))
            elif isinstance(arg, InputArgument):
                # Add the argument name to input_symbols
                input_symbols.add(arg.name)
            elif isinstance(arg, InOutArgument):
                # Add the argument name to input_symbols and update symbols similarly to OutputArgument
                input_symbols.add(arg.name)
                symbols.update(arg.expr.free_symbols - arg.expr.atoms(Indexed))
            else:
                # Raise an error if an unknown argument type is encountered
                raise ValueError("Unknown Routine argument: %s" % arg)

        # Iterate over each result to update symbols with free symbols in the expression
        for r in results:
            if not isinstance(r, Result):
                raise ValueError("Unknown Routine result: %s" % r)
            symbols.update(r.expr.free_symbols - r.expr.atoms(Indexed))

        local_symbols = set()

        # Iterate over local_vars to update symbols with free symbols in the expression and add local variable names
        for r in local_vars:
            if isinstance(r, Result):
                symbols.update(r.expr.free_symbols - r.expr.atoms(Indexed))
                local_symbols.add(r.name)
            else:
                local_symbols.add(r)

        # Convert symbols to labels if they are Idx objects, otherwise leave them as they are
        symbols = {s.label if isinstance(s, Idx) else s for s in symbols}

        # Check that all symbols in expressions are covered by input_arguments, local_symbols, or global_vars
        notcovered = symbols.difference(
            input_symbols.union(local_symbols).union(global_vars))
        if notcovered != set():
            raise ValueError("Symbols needed for output are not in input " +
                             ", ".join([str(x) for x in notcovered]))

        # Assign instance variables based on the provided parameters
        self.name = name
        self.arguments = arguments
        self.results = results
        self.local_vars = local_vars
        self.global_vars = global_vars
    # 返回描述实例的字符串表示形式，包括类名、名称、参数、结果、局部变量和全局变量
    def __str__(self):
        return self.__class__.__name__ + "({name!r}, {arguments}, {results}, {local_vars}, {global_vars})".format(**self.__dict__)

    # 将 __str__ 方法作为 __repr__ 方法的别名
    __repr__ = __str__

    @property
    # 返回可能在例程中使用的所有变量的集合
    # 对于没有命名返回值的例程，可能包含或不包含的虚拟变量也将包括在集合中
    def variables(self):
        v = set(self.local_vars)  # 将局部变量转换为集合
        v.update(arg.name for arg in self.arguments)  # 添加所有参数的名称到集合中
        v.update(res.result_var for res in self.results)  # 添加所有结果变量的名称到集合中
        return v

    @property
    # 返回一个包含 OutputArgument、InOutArgument 和 Result 的列表
    # 如果有返回值存在，它们将在列表的末尾
    def result_variables(self):
        args = [arg for arg in self.arguments if isinstance(
            arg, (OutputArgument, InOutArgument))]  # 筛选所有输出参数和输入/输出参数
        args.extend(self.results)  # 将所有结果添加到列表中
        return args
class DataType:
    """定义了一个包含特定数据类型在不同语言中命名的字符串的类。"""
    def __init__(self, cname, fname, pyname, jlname, octname, rsname):
        # 初始化方法，为每种语言设定对应的命名字符串
        self.cname = cname          # C 语言中的命名
        self.fname = fname          # Fortran 中的命名
        self.pyname = pyname        # Python 中的命名
        self.jlname = jlname        # Julia 中的命名
        self.octname = octname      # Octave 中的命名
        self.rsname = rsname        # Rust 中的命名


default_datatypes = {
    "int": DataType("int", "INTEGER*4", "int", "", "", "i32"),
    "float": DataType("double", "REAL*8", "float", "", "", "f64"),
    "complex": DataType("double", "COMPLEX*16", "complex", "", "", "float") #FIXME:
       # 复数仅在 Fortran、Python、Julia 和 Octave 中受支持。
       # 为了不破坏 C 或 Rust 代码生成，我们仍然使用 double 或 float，
       # 分别对应于复数的情况（但实际上应该对显式复数变量（x.is_complex==True）引发异常）。
}


COMPLEX_ALLOWED = False
def get_default_datatype(expr, complex_allowed=None):
    """根据表达式推断适当的数据类型。"""
    if complex_allowed is None:
        complex_allowed = COMPLEX_ALLOWED
    if complex_allowed:
        final_dtype = "complex"
    else:
        final_dtype = "float"
    if expr.is_integer:
        return default_datatypes["int"]
    elif expr.is_real:
        return default_datatypes["float"]
    elif isinstance(expr, MatrixBase):
        # 检查所有元素
        dt = "int"
        for element in expr:
            if dt == "int" and not element.is_integer:
                dt = "float"
            if dt == "float" and not element.is_real:
                return default_datatypes[final_dtype]
        return default_datatypes[dt]
    else:
        return default_datatypes[final_dtype]


class Variable:
    """代表一个带类型的变量。"""
    def __init__(self, name, datatype=None, dimensions=None, precision=None):
        """
        Return a new variable.

        Parameters
        ==========

        name : Symbol or MatrixSymbol
            符号或矩阵符号的名称。

        datatype : optional
            如果未指定，则根据符号参数的假设来猜测数据类型。

        dimensions : sequence containing tuples, optional
            如果存在，将被解释为数组，其中每个元组指定数组每个索引的（下限，上限）边界。

        precision : int, optional
            控制浮点常数的精度。
        """
        if not isinstance(name, (Symbol, MatrixSymbol)):
            raise TypeError("The first argument must be a SymPy symbol.")
        if datatype is None:
            datatype = get_default_datatype(name)
        elif not isinstance(datatype, DataType):
            raise TypeError("The (optional) `datatype' argument must be an "
                            "instance of the DataType class.")
        if dimensions and not isinstance(dimensions, (tuple, list)):
            raise TypeError(
                "The dimensions argument must be a sequence of tuples")

        self._name = name
        self._datatype = {
            'C': datatype.cname,
            'FORTRAN': datatype.fname,
            'JULIA': datatype.jlname,
            'OCTAVE': datatype.octname,
            'PYTHON': datatype.pyname,
            'RUST': datatype.rsname,
        }
        self.dimensions = dimensions  # 将维度信息存储在对象的属性中
        self.precision = precision    # 存储精度信息在对象的属性中

    def __str__(self):
        """
        返回对象的字符串表示形式。

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.utilities.codegen import Variable
        >>> x = Variable(Symbol('x'))
        >>> x.get_datatype('c')
        'double'
        >>> x.get_datatype('fortran')
        'REAL*8'
        """
        return "%s(%r)" % (self.__class__.__name__, self.name)

    __repr__ = __str__  # __repr__ 方法与 __str__ 方法相同

    @property
    def name(self):
        """
        返回变量的名称。
        """
        return self._name

    def get_datatype(self, language):
        """
        返回请求语言的数据类型字符串。

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.utilities.codegen import Variable
        >>> x = Variable(Symbol('x'))
        >>> x.get_datatype('c')
        'double'
        >>> x.get_datatype('fortran')
        'REAL*8'
        """
        try:
            return self._datatype[language.upper()]
        except KeyError:
            raise CodeGenError("Has datatypes for languages: %s" %
                    ", ".join(self._datatype))
class Argument(Variable):
    """An abstract Argument data structure: a name and a data type.

    This structure is refined in the descendants below.

    """
    pass



class InputArgument(Argument):
    """Represents an input argument, inheriting from Argument."""
    pass



class ResultBase:
    """Base class for all "outgoing" information from a routine.

    Objects of this class store a SymPy expression and a SymPy object
    representing a result variable that will be used in the generated code
    only if necessary.

    """
    def __init__(self, expr, result_var):
        """Initialize ResultBase with a SymPy expression and result variable.

        Parameters
        ----------
        expr : object
            The SymPy expression to store.
        result_var : object
            The SymPy object representing the result variable.

        """
        self.expr = expr
        self.result_var = result_var

    def __str__(self):
        """Return a string representation of ResultBase."""
        return "%s(%r, %r)" % (self.__class__.__name__, self.expr,
            self.result_var)

    __repr__ = __str__



class OutputArgument(Argument, ResultBase):
    """Represents an output argument initialized in the routine.

    Inherits from Argument and ResultBase.

    """
    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        """Initialize OutputArgument with various parameters.

        Parameters
        ----------
        name : Symbol, MatrixSymbol
            The name of the variable.
        result_var : Symbol, Indexed
            The object used to assign a value to the variable.
        expr : object
            The expression that should be output.
        datatype : optional
            The data type of the variable (guessed if not provided).
        dimensions : sequence containing tuples, optional
            Specifies the bounds for each index if the variable is an array.
        precision : int, optional
            Controls the precision of floating point constants.

        """
        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        """Return a string representation of OutputArgument."""
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.result_var, self.expr)

    __repr__ = __str__



class InOutArgument(Argument, ResultBase):
    """Represents an in-out argument not initialized in the routine.

    Inherits from Argument and ResultBase.

    """
    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        """Initialize InOutArgument with various parameters.

        Parameters
        ----------
        name : Symbol, MatrixSymbol
            The name of the variable.
        result_var : Symbol, Indexed
            The object used to assign a value to the variable.
        expr : object
            The expression that should be output.
        datatype : optional
            The data type of the variable (guessed if not provided).
        dimensions : sequence containing tuples, optional
            Specifies the bounds for each index if the variable is an array.
        precision : int, optional
            Controls the precision of floating point constants.

        """
        if not datatype:
            datatype = get_default_datatype(expr)
        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)
    __init__.__doc__ = OutputArgument.__init__.__doc__

    def __str__(self):
        """Return a string representation of InOutArgument."""
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.expr,
            self.result_var)

    __repr__ = __str__
# 表示一个结果变量，继承自Variable和ResultBase类
class Result(Variable, ResultBase):
    """An expression for a return value.

    The name result is used to avoid conflicts with the reserved word
    "return" in the Python language.  It is also shorter than ReturnValue.

    These may or may not need a name in the destination (e.g., "return(x*y)"
    might return a value without ever naming it).

    """

    def __init__(self, expr, name=None, result_var=None, datatype=None,
                 dimensions=None, precision=None):
        """Initialize a return value.

        Parameters
        ==========

        expr : SymPy expression
            表达式，必须是SymPy表达式类型或者MatrixBase类型的对象

        name : Symbol, MatrixSymbol, optional
            此返回变量的名称。在代码生成中，例如在返回值列表中的函数原型中可能出现。如果省略，则生成一个虚拟名称。

        result_var : Symbol, Indexed, optional
            用于分配值给此变量的对象。通常与`name`相同，但对于Indexed类型应为形如"y[i]"的形式，而`name`应为符号"y"。如果省略，默认为`name`。

        datatype : optional
            当未指定时，数据类型将根据expr参数的假设进行推断。

        dimensions : sequence containing tuples, optional
            如果存在，则此变量被解释为数组，其中此序列的元组指定数组每个索引的（下界，上界）。

        precision : int, optional
            控制浮点常数的精度。

        """
        # 检查表达式是否为Basic或MatrixBase类型，否则引发TypeError异常
        if not isinstance(expr, (Basic, MatrixBase)):
            raise TypeError("The first argument must be a SymPy expression.")

        # 如果未指定name，则根据表达式的哈希生成一个名称
        if name is None:
            name = 'result_%d' % abs(hash(expr))

        # 如果未指定datatype，则尝试从表达式推断数据类型
        if datatype is None:
            datatype = get_default_datatype(expr)

        # 如果name是字符串类型，则根据表达式的类型创建Symbol或MatrixSymbol对象
        if isinstance(name, str):
            if isinstance(expr, (MatrixBase, MatrixExpr)):
                name = MatrixSymbol(name, *expr.shape)
            else:
                name = Symbol(name)

        # 如果result_var未指定，则默认为name
        if result_var is None:
            result_var = name

        # 调用Variable类的初始化方法，初始化变量名称、数据类型等信息
        Variable.__init__(self, name, datatype=datatype,
                          dimensions=dimensions, precision=precision)
        
        # 调用ResultBase类的初始化方法，初始化表达式和结果变量
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        # 返回对象的字符串表示形式，包括类名、表达式、名称和结果变量
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.expr, self.name,
            self.result_var)

    __repr__ = __str__


#
# 转换程序对象为代码
#

class CodeGen:
    """Abstract class for the code generators."""

    printer = None  # 将被设置为CodePrinter子类的实例

    def _indent_code(self, codelines):
        # 调用printer对象的indent_code方法，缩进代码行
        return self.printer.indent_code(codelines)
    def _printer_method_with_settings(self, method, settings=None, *args, **kwargs):
        # 如果未提供 settings 参数，则设为一个空字典
        settings = settings or {}
        # 备份当前设置中与 settings 参数中键对应的原始值
        ori = {k: self.printer._settings[k] for k in settings}
        # 将 settings 中的设置值更新到 printer 对象的设置中
        for k, v in settings.items():
            self.printer._settings[k] = v
        # 调用 printer 对象的指定 method 方法，并传入其他位置参数和关键字参数
        result = getattr(self.printer, method)(*args, **kwargs)
        # 恢复原始的 printer 对象的设置值
        for k, v in ori.items():
            self.printer._settings[k] = v
        # 返回方法调用的结果
        return result

    def _get_symbol(self, s):
        """Returns the symbol as fcode prints it."""
        # 如果设置为人类可读，则调用 doprint 方法打印符号 s
        if self.printer._settings['human']:
            expr_str = self.printer.doprint(s)
        else:
            # 否则调用 doprint 方法，获取常量、不支持的内容和打印后的表达式字符串
            constants, not_supported, expr_str = self.printer.doprint(s)
            # 如果存在常量或不支持的内容，则引发 ValueError 异常
            if constants or not_supported:
                raise ValueError("Failed to print %s" % str(s))
        # 返回去除两端空白的表达式字符串
        return expr_str.strip()

    def __init__(self, project="project", cse=False):
        """Initialize a code generator.

        Derived classes will offer more options that affect the generated
        code.

        """
        # 初始化代码生成器对象，设置项目名称和 cse 标志位
        self.project = project
        self.cse = cse

    def write(self, routines, prefix, to_files=False, header=True, empty=True):
        """Writes all the source code files for the given routines.

        The generated source is returned as a list of (filename, contents)
        tuples, or is written to files (see below).  Each filename consists
        of the given prefix, appended with an appropriate extension.

        Parameters
        ==========

        routines : list
            A list of Routine instances to be written

        prefix : string
            The prefix for the output files

        to_files : bool, optional
            When True, the output is written to files.  Otherwise, a list
            of (filename, contents) tuples is returned.  [default: False]

        header : bool, optional
            When True, a header comment is included on top of each source
            file. [default: True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files. [default: True]

        """
        # 如果 to_files 参数为 True，则将生成的源代码写入文件
        if to_files:
            # 遍历所有的 dump 函数，并生成对应的文件
            for dump_fn in self.dump_fns:
                # 构建文件名，格式为 prefix + 文件扩展名
                filename = "%s.%s" % (prefix, dump_fn.extension)
                # 打开文件并写入生成的代码内容
                with open(filename, "w") as f:
                    dump_fn(self, routines, f, prefix, header, empty)
        else:
            # 如果 to_files 参数为 False，则返回生成的源代码作为列表
            result = []
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                # 使用 StringIO 缓存生成的代码内容
                contents = StringIO()
                # 将生成的代码写入到 contents 中
                dump_fn(self, routines, contents, prefix, header, empty)
                # 将文件名和内容的元组添加到结果列表中
                result.append((filename, contents.getvalue()))
            # 返回生成的源代码列表
            return result
    def dump_code(self, routines, f, prefix, header=True, empty=True):
        """Write the code by calling language specific methods.

        The generated file contains all the definitions of the routines in
        low-level code and refers to the header file if appropriate.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to refer to the proper header file.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """

        # 获取预处理语句的代码行
        code_lines = self._preprocessor_statements(prefix)

        # 遍历每个例程对象
        for routine in routines:
            # 如果需要空行，则添加一个空行
            if empty:
                code_lines.append("\n")
            # 添加例程的开头部分
            code_lines.extend(self._get_routine_opening(routine))
            # 添加例程的参数声明部分
            code_lines.extend(self._declare_arguments(routine))
            # 添加例程的全局变量声明部分
            code_lines.extend(self._declare_globals(routine))
            # 添加例程的局部变量声明部分
            code_lines.extend(self._declare_locals(routine))
            # 如果需要空行，则添加一个空行
            if empty:
                code_lines.append("\n")
            # 调用打印器，将例程的代码添加到代码行中
            code_lines.extend(self._call_printer(routine))
            # 如果需要空行，则添加一个空行
            if empty:
                code_lines.append("\n")
            # 添加例程的结束部分
            code_lines.extend(self._get_routine_ending(routine))

        # 缩进整理生成的代码
        code_lines = self._indent_code(''.join(code_lines))

        # 如果需要添加文件头部注释，则将其添加到代码行的开头
        if header:
            code_lines = ''.join(self._get_header() + [code_lines])

        # 将最终生成的代码写入文件对象
        if code_lines:
            f.write(code_lines)
# 定义自定义异常类 CodeGenError，继承自 Exception，用于代码生成错误的异常处理
class CodeGenError(Exception):
    pass


# 定义自定义异常类 CodeGenArgumentListError，继承自 Exception
class CodeGenArgumentListError(Exception):
    @property
    def missing_args(self):
        # 返回异常的第二个参数，表示缺失的参数列表
        return self.args[1]


# 头部注释字符串，包含 SymPy 的版本信息和项目信息
header_comment = """Code generated with SymPy %(version)s

See http://www.sympy.org/ for more information.

This file is part of '%(project)s'
"""


# 定义 CCodeGen 类，继承自 CodeGen
class CCodeGen(CodeGen):
    """Generator for C code.

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.c and <prefix>.h respectively.

    """

    # C 代码文件的扩展名
    code_extension = "c"
    # 接口文件的扩展名
    interface_extension = "h"
    # C 代码的标准版本
    standard = 'c99'

    def __init__(self, project="project", printer=None,
                 preprocessor_statements=None, cse=False):
        # 调用父类的构造函数初始化
        super().__init__(project=project, cse=cse)
        # 设置打印器，如果未指定，则使用标准的 C 代码打印器
        self.printer = printer or c_code_printers[self.standard.lower()]

        # 初始化预处理语句，如果未指定，则默认包含 math.h 头文件
        self.preprocessor_statements = preprocessor_statements
        if preprocessor_statements is None:
            self.preprocessor_statements = ['#include <math.h>']

    def _get_header(self):
        """Writes a common header for the generated files."""
        # 生成通用文件头部注释的代码行列表
        code_lines = []
        code_lines.append("/" + "*"*78 + '\n')
        # 根据 header_comment 模板格式化头部注释内容，并居中对齐添加到代码行列表
        tmp = header_comment % {"version": sympy_version,
                                "project": self.project}
        for line in tmp.splitlines():
            code_lines.append(" *%s*\n" % line.center(76))
        code_lines.append(" " + "*"*78 + "/\n")
        return code_lines

    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        # 获取 routine 的函数原型字符串表示

        # 如果 routine 的结果对象数大于 1，则抛出 CodeGenError 异常
        if len(routine.results) > 1:
            raise CodeGenError("C only supports a single or no return value.")
        elif len(routine.results) == 1:
            # 获取结果对象的 C 数据类型
            ctype = routine.results[0].get_datatype('C')
        else:
            ctype = "void"

        type_args = []
        # 遍历 routine 的参数列表
        for arg in routine.arguments:
            name = self.printer.doprint(arg.name)
            # 如果参数具有维度信息或者是 ResultBase 的实例，则将其指针化
            if arg.dimensions or isinstance(arg, ResultBase):
                type_args.append((arg.get_datatype('C'), "*%s" % name))
            else:
                type_args.append((arg.get_datatype('C'), name))
        # 构建参数列表字符串
        arguments = ", ".join([ "%s %s" % t for t in type_args])
        # 返回函数原型字符串
        return "%s %s(%s)" % (ctype, routine.name, arguments)

    def _preprocessor_statements(self, prefix):
        # 生成预处理语句的代码行列表
        code_lines = []
        # 添加包含接口文件的预处理语句
        code_lines.append('#include "{}.h"'.format(os.path.basename(prefix)))
        # 添加额外的预处理语句
        code_lines.extend(self.preprocessor_statements)
        # 将每行预处理语句格式化为字符串列表
        code_lines = ['{}\n'.format(l) for l in code_lines]
        return code_lines

    def _get_routine_opening(self, routine):
        # 获取 routine 的函数开头部分的代码行列表
        prototype = self.get_prototype(routine)
        return ["%s {\n" % prototype]

    def _declare_arguments(self, routine):
        # 参数在函数原型中已经声明，此处无需再声明，返回空列表
        return []
    # 定义一个方法，用于在特定程序例程中声明全局变量
    def _declare_globals(self, routine):
        # 在 C 函数内部不需要显式声明全局变量
        return []

    # 定义一个方法，用于在特定程序例程中声明局部变量
    def _declare_locals(self, routine):

        # 组成一个列表，包含函数体中需要通过引用指针解引用的符号
        # 这些符号是作为参数传递的，但不包括数组
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        # 初始化代码行列表
        code_lines = []
        for result in routine.local_vars:

            # 跳过简单符号形式的局部变量，例如用作循环索引的变量，这些在其他地方定义声明
            if not isinstance(result, Result):
                continue

            # 检查结果变量和名称是否匹配，否则引发代码生成异常
            if result.name != result.result_var:
                raise CodeGen("Result variable and name should match: {}".format(result))

            # 分配给局部变量的名称
            assign_to = result.name
            # 获取变量的 C 数据类型
            t = result.get_datatype('c')

            # 如果表达式是矩阵类型，则计算其维度并声明数组
            if isinstance(result.expr, (MatrixBase, MatrixExpr)):
                dims = result.expr.shape
                code_lines.append("{} {}[{}];\n".format(t, str(assign_to), dims[0]*dims[1]))
                prefix = ""
            else:
                prefix = "const {} ".format(t)

            # 使用打印设置的方法处理表达式，得到常量、非 C 表达式和 C 表达式
            constants, not_c, c_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "dereference": dereference, "strict": False},
                result.expr, assign_to=assign_to)

            # 将常量声明为 double const 类型
            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))

            # 将 C 表达式添加到代码行列表中
            code_lines.append("{}{}\n".format(prefix, c_expr))

        # 返回生成的代码行列表
        return code_lines
    # 定义一个方法用于调用打印机生成器，接受一个程序例行的参数
    def _call_printer(self, routine):
        # 初始化代码行列表
        code_lines = []

        # 组合一个需要在函数体中解引用的符号列表。这些是通过引用指针传递的参数，
        # 不包括数组。
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        return_val = None
        # 遍历例行的结果变量
        for result in routine.result_variables:
            if isinstance(result, Result):
                # 为结果分配一个变量名
                assign_to = routine.name + "_result"
                # 获取结果的数据类型 'c'
                t = result.get_datatype('c')
                code_lines.append("{} {};\n".format(t, str(assign_to)))
                return_val = assign_to
            else:
                # 对于不是 Result 实例的结果，使用其结果变量
                assign_to = result.result_var

            try:
                # 使用给定设置调用打印机方法，生成C表达式
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', {"human": False, "dereference": dereference, "strict": False},
                    result.expr, assign_to=assign_to)
            except AssignmentError:
                # 如果出现分配错误，使用结果变量作为分配目标
                assign_to = result.result_var
                code_lines.append(
                    "%s %s;\n" % (result.get_datatype('c'), str(assign_to)))
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', {"human": False, "dereference": dereference, "strict": False},
                    result.expr, assign_to=assign_to)

            # 将常量按名称排序后，作为 double const 类型添加到代码行列表
            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))
            # 将生成的 C 表达式添加到代码行列表
            code_lines.append("%s\n" % c_expr)

        # 如果存在返回值变量，将其作为返回语句添加到代码行列表
        if return_val:
            code_lines.append("   return %s;\n" % return_val)
        # 返回生成的代码行列表
        return code_lines

    # 返回一个包含单个 '}' 字符串的列表，表示例行结束
    def _get_routine_ending(self, routine):
        return ["}\n"]

    # 定义一个方法用于导出C代码，调用代码生成器的dump_code方法
    def dump_c(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)
    # 将 dump_c 方法的 extension 属性设置为 code_extension
    dump_c.extension = code_extension  # type: ignore
    # 将 dump_c 方法的文档字符串设置为 dump_code 方法的文档字符串
    dump_c.__doc__ = CodeGen.dump_code.__doc__
    # 定义一个方法 dump_h，用于生成 C 头文件
    def dump_h(self, routines, f, prefix, header=True, empty=True):
        """Writes the C header file.

        This file contains all the function declarations.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to construct the include guards.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """
        # 如果 header 参数为 True，则打印文件头部注释
        if header:
            print(''.join(self._get_header()), file=f)
        # 构建 include guards 的名称
        guard_name = "%s__%s__H" % (self.project.replace(
            " ", "_").upper(), prefix.replace("/", "_").upper())
        # 插入空行，如果 empty 参数为 True
        if empty:
            print(file=f)
        # 输出 ifndef 的预处理指令
        print("#ifndef %s" % guard_name, file=f)
        # 输出 define 的预处理指令
        print("#define %s" % guard_name, file=f)
        # 再次插入空行，如果 empty 参数为 True
        if empty:
            print(file=f)
        # 遍历 routines 列表中的每个 Routine 实例，输出其函数原型声明
        for routine in routines:
            prototype = self.get_prototype(routine)
            print("%s;" % prototype, file=f)
        # 输出 endif 的预处理指令
        if empty:
            print(file=f)
        print("#endif", file=f)
        # 最后再次插入空行，如果 empty 参数为 True
        if empty:
            print(file=f)
    # 将属性 extension 设置为 interface_extension，用于类型检查（type check）
    dump_h.extension = interface_extension  # type: ignore

    # 这个 dump 函数列表用于 CodeGen.write 方法，以确定要调用哪些 dump 函数。
    dump_fns = [dump_c, dump_h]
class C89CodeGen(CCodeGen):
    # 定义一个类 C89CodeGen，继承自 CCodeGen

    standard = 'C89'
    # 类变量 standard 被赋值为字符串 'C89'

class C99CodeGen(CCodeGen):
    # 定义一个类 C99CodeGen，继承自 CCodeGen

    standard = 'C99'
    # 类变量 standard 被赋值为字符串 'C99'

class FCodeGen(CodeGen):
    """Generator for Fortran 95 code

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.f90 and <prefix>.h respectively.

    """
    # 定义一个类 FCodeGen，继承自 CodeGen，用于生成 Fortran 95 代码

    code_extension = "f90"
    # 类变量 code_extension 被赋值为字符串 "f90"，代表代码文件的扩展名

    interface_extension = "h"
    # 类变量 interface_extension 被赋值为字符串 "h"，代表接口文件的扩展名

    def __init__(self, project='project', printer=None):
        # 初始化方法，接受项目名称和打印机对象作为参数

        super().__init__(project)
        # 调用父类的初始化方法，并传入项目名称

        self.printer = printer or FCodePrinter()
        # 如果没有传入打印机对象，则使用默认的 FCodePrinter 创建一个

    def _get_header(self):
        """Writes a common header for the generated files."""
        # 返回生成文件的通用头部内容的方法

        code_lines = []
        # 创建一个空列表，用于存储生成的代码行

        code_lines.append("!" + "*"*78 + '\n')
        # 向列表中添加一行以感叹号和78个星号开头的注释行

        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        # 使用 sympy_version 和类的项目名称，格式化生成文件的头部注释

        for line in tmp.splitlines():
            code_lines.append("!*%s*\n" % line.center(76))
        # 将格式化后的注释行逐行添加到列表中，每行最大宽度为76个字符

        code_lines.append("!" + "*"*78 + '\n')
        # 向列表中再添加一行以感叹号和78个星号结尾的注释行

        return code_lines
        # 返回生成的代码行列表

    def _preprocessor_statements(self, prefix):
        # 返回预处理语句的方法，参数为文件名前缀

        return []
        # 返回空列表，表示没有预处理语句

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the fortran routine."""
        # 返回 Fortran 例程的开头语句的方法，参数为例程对象

        code_list = []
        # 创建一个空列表，用于存储生成的代码行

        if len(routine.results) > 1:
            raise CodeGenError(
                "Fortran only supports a single or no return value.")
        # 如果例程的结果数量大于1，则抛出 CodeGenError 异常

        elif len(routine.results) == 1:
            result = routine.results[0]
            code_list.append(result.get_datatype('fortran'))
            code_list.append("function")
        # 如果例程的结果数量为1，则获取结果数据类型，并将 "function" 添加到代码列表中

        else:
            code_list.append("subroutine")
        # 如果例程没有结果，则将 "subroutine" 添加到代码列表中

        args = ", ".join("%s" % self._get_symbol(arg.name)
                        for arg in routine.arguments)
        # 构造例程参数的字符串表示形式，使用 _get_symbol 方法获取参数名称

        call_sig = "{}({})\n".format(routine.name, args)
        # 构造调用签名，包括例程名称和参数列表

        call_sig = ' &\n'.join(textwrap.wrap(call_sig,
                                             width=60,
                                             break_long_words=False)) + '\n'
        # 使用 textwrap.wrap 方法将调用签名进行换行处理，每行最大宽度为60个字符

        code_list.append(call_sig)
        # 将格式化后的调用签名添加到代码列表中

        code_list = [' '.join(code_list)]
        # 将代码列表转换为只含一个元素的列表，元素为合并后的代码字符串

        code_list.append('implicit none\n')
        # 向代码列表中添加 "implicit none"，表示禁用隐式类型声明

        return code_list
        # 返回生成的代码行列表
    # 声明函数用于生成参数类型声明的代码块
    def _declare_arguments(self, routine):
        # 初始化空列表，用于存储生成的代码行
        code_list = []
        # 初始化空列表，用于存储生成的数组声明代码行
        array_list = []
        # 初始化空列表，用于存储生成的标量声明代码行
        scalar_list = []
        # 遍历每个参数对象
        for arg in routine.arguments:

            # 根据参数类型的不同，生成不同的声明字符串
            if isinstance(arg, InputArgument):
                typeinfo = "%s, intent(in)" % arg.get_datatype('fortran')
            elif isinstance(arg, InOutArgument):
                typeinfo = "%s, intent(inout)" % arg.get_datatype('fortran')
            elif isinstance(arg, OutputArgument):
                typeinfo = "%s, intent(out)" % arg.get_datatype('fortran')
            else:
                raise CodeGenError("Unknown Argument type: %s" % type(arg))

            # 获取参数名对应的符号名称
            fprint = self._get_symbol

            # 如果参数有维度信息
            if arg.dimensions:
                # 生成 Fortran 数组维度声明，注意 Fortran 数组从 1 开始索引
                dimstr = ", ".join(["%s:%s" % (
                    fprint(dim[0] + 1), fprint(dim[1] + 1))
                    for dim in arg.dimensions])
                typeinfo += ", dimension(%s)" % dimstr
                # 将数组声明加入数组列表
                array_list.append("%s :: %s\n" % (typeinfo, fprint(arg.name)))
            else:
                # 将标量声明加入标量列表
                scalar_list.append("%s :: %s\n" % (typeinfo, fprint(arg.name)))

        # 首先加入标量声明，因为它们可能在数组声明中使用
        code_list.extend(scalar_list)
        # 然后加入数组声明
        code_list.extend(array_list)

        # 返回生成的代码列表
        return code_list

    # 生成全局变量声明的空列表，因为 Fortran 90 不需要显式声明全局变量
    def _declare_globals(self, routine):
        # 返回空列表
        return []

    # 生成本地变量声明代码块
    def _declare_locals(self, routine):
        # 初始化空列表，用于存储生成的本地变量声明代码行
        code_list = []
        # 遍历并按字符串顺序排序每个本地变量对象
        for var in sorted(routine.local_vars, key=str):
            # 获取变量的默认数据类型信息
            typeinfo = get_default_datatype(var)
            # 生成本地变量声明代码行，并加入到代码列表中
            code_list.append("%s :: %s\n" % (
                typeinfo.fname, self._get_symbol(var)))
        # 返回生成的本地变量声明代码列表
        return code_list

    # 根据函数或子程序的返回结果数目，生成相应的结束语句
    def _get_routine_ending(self, routine):
        # 如果结果数目为1，生成函数结束语句
        if len(routine.results) == 1:
            return ["end function\n"]
        else:
            # 否则，生成子程序结束语句
            return ["end subroutine\n"]

    # 生成函数接口的字符串表示
    def get_interface(self, routine):
        """Returns a string for the function interface.

        The routine should have a single result object, which can be None.
        If the routine has multiple result objects, a CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        # 初始化接口字符串列表
        prototype = [ "interface\n" ]
        # 加入函数或子程序的开始语句
        prototype.extend(self._get_routine_opening(routine))
        # 加入参数声明代码块
        prototype.extend(self._declare_arguments(routine))
        # 加入函数或子程序的结束语句
        prototype.extend(self._get_routine_ending(routine))
        # 加入接口结束语句
        prototype.append("end interface\n")

        # 返回连接后的接口字符串
        return "".join(prototype)
    # 定义一个私有方法 `_call_printer`，接收一个参数 `routine`
    def _call_printer(self, routine):
        # 声明一个空列表用于存储声明语句
        declarations = []
        # 声明一个空列表用于存储代码行
        code_lines = []
        
        # 遍历 routine 对象的 result_variables 属性
        for result in routine.result_variables:
            # 如果 result 是 Result 类型的实例
            if isinstance(result, Result):
                # 将 routine.name 赋给 assign_to
                assign_to = routine.name
            # 如果 result 是 OutputArgument 或者 InOutArgument 类型的实例
            elif isinstance(result, (OutputArgument, InOutArgument)):
                # 将 result.result_var 赋给 assign_to
                assign_to = result.result_var

            # 调用 self._printer_method_with_settings 方法执行 'doprint' 操作
            constants, not_fortran, f_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "source_format": 'free', "standard": 95, "strict": False},
                result.expr, assign_to=assign_to)

            # 遍历 constants 列表，按字符串排序后生成声明语句
            for obj, v in sorted(constants, key=str):
                t = get_default_datatype(obj)
                declarations.append(
                    "%s, parameter :: %s = %s\n" % (t.fname, obj, v))
            
            # 遍历 not_fortran 列表，按字符串排序后生成声明语句
            for obj in sorted(not_fortran, key=str):
                t = get_default_datatype(obj)
                # 如果 obj 是 Function 类型的实例，则取其 func 属性作为 name
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append("%s :: %s\n" % (t.fname, name))

            # 将 f_expr 添加到代码行列表中
            code_lines.append("%s\n" % f_expr)
        
        # 返回声明语句和代码行组成的列表
        return declarations + code_lines

    # 定义一个私有方法 `_indent_code`，接收一个参数 `codelines`
    def _indent_code(self, codelines):
        # 调用 self._printer_method_with_settings 方法执行 'indent_code' 操作
        return self._printer_method_with_settings(
            'indent_code', {"human": False, "source_format": 'free', "strict": False}, codelines)

    # 定义一个公有方法 `dump_f95`，接收 routines、f、prefix、header、empty 五个参数
    def dump_f95(self, routines, f, prefix, header=True, empty=True):
        # 检查 symbols 是否在忽略大小写的情况下唯一
        for r in routines:
            lowercase = {str(x).lower() for x in r.variables}
            orig_case = {str(x) for x in r.variables}
            if len(lowercase) < len(orig_case):
                # 如果存在相同的符号（忽略大小写），抛出 CodeGenError 异常
                raise CodeGenError("Fortran ignores case. Got symbols: %s" %
                        (", ".join([str(var) for var in r.variables])))
        
        # 调用 self.dump_code 方法，将 routines、f、prefix、header、empty 作为参数传递
        self.dump_code(routines, f, prefix, header, empty)
        
        # 将 dump_f95.extension 属性设置为 code_extension，类型为 'ignore'
        dump_f95.extension = code_extension  # type: ignore
        
        # 设置 dump_f95.__doc__ 属性为 CodeGen.dump_code.__doc__ 的文档字符串
        dump_f95.__doc__ = CodeGen.dump_code.__doc__
    def dump_h(self, routines, f, prefix, header=True, empty=True):
        """Writes the interface to a header file.

        This file contains all the function declarations.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """
        # 如果header为True，将接口的头部信息写入文件f中
        if header:
            print(''.join(self._get_header()), file=f)
        # 如果empty为True，在文件中插入空行
        if empty:
            print(file=f)
        # 对于每一个routine，生成其函数原型并写入文件f
        for routine in routines:
            prototype = self.get_interface(routine)
            f.write(prototype)
        # 如果empty为True，在文件末尾再插入一个空行
        if empty:
            print(file=f)
    # 设置一个属性，用于指定接口文件的扩展名
    dump_h.extension = interface_extension  # type: ignore

    # 这个列表包含了CodeGen.write要调用的dump函数列表
    dump_fns = [dump_f95, dump_h]
# 定义 Julia 代码生成器的类，继承自 CodeGen 类
class JuliaCodeGen(CodeGen):
    """Generator for Julia code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.jl.

    """

    # 类变量，指定生成的代码文件扩展名为 "jl"
    code_extension = "jl"

    # 初始化方法，接受项目名称和打印器对象作为参数
    def __init__(self, project='project', printer=None):
        # 调用父类 CodeGen 的初始化方法
        super().__init__(project)
        # 如果未提供打印器对象，则使用默认的 JuliaCodePrinter
        self.printer = printer or JuliaCodePrinter()

    # 内部方法，返回生成文件的公共头部注释内容
    def _get_header(self):
        """Writes a common header for the generated files."""
        # 初始化空的代码行列表
        code_lines = []
        # 生成头部注释内容，格式化 sympy_version 和项目名称
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        # 将注释按行处理
        for line in tmp.splitlines():
            if line == '':
                # 空行
                code_lines.append("#\n")
            else:
                # 添加带有 # 注释符的内容行
                code_lines.append("#   %s\n" % line)
        return code_lines

    # 内部方法，返回预处理语句列表，这里返回空列表
    def _preprocessor_statements(self, prefix):
        return []

    # 内部方法，返回例程（routine）的开头语句列表
    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        # 初始化代码列表
        code_list = []
        # 添加函数定义头部
        code_list.append("function ")

        # 处理输入参数
        args = []
        for arg in routine.arguments:
            if isinstance(arg, OutputArgument):
                # 如果参数是输出参数，抛出代码生成错误
                raise CodeGenError("Julia: invalid argument of type %s" %
                                   str(type(arg)))
            if isinstance(arg, (InputArgument, InOutArgument)):
                # 如果参数是输入或输入输出参数，添加参数名到参数列表中
                args.append("%s" % self._get_symbol(arg.name))
        # 将参数列表转换为字符串形式
        args = ", ".join(args)
        # 添加函数名和参数列表到代码列表中
        code_list.append("%s(%s)\n" % (routine.name, args))
        # 将代码列表转换为单个字符串形式的列表
        code_list = [ "".join(code_list) ]

        return code_list

    # 内部方法，返回声明参数的语句列表，这里返回空列表
    def _declare_arguments(self, routine):
        return []

    # 内部方法，返回声明全局变量的语句列表，这里返回空列表
    def _declare_globals(self, routine):
        return []

    # 内部方法，返回声明局部变量的语句列表，这里返回空列表
    def _declare_locals(self, routine):
        return []

    # 内部方法，返回例程（routine）的结尾语句列表
    def _get_routine_ending(self, routine):
        # 初始化结果列表
        outs = []
        # 处理例程中的结果
        for result in routine.results:
            if isinstance(result, Result):
                # 如果结果是 Result 类型，获取其符号名
                s = self._get_symbol(result.name)
            else:
                # 如果结果类型不符合预期，抛出代码生成错误
                raise CodeGenError("unexpected object in Routine results")
            # 添加符号名到结果列表中
            outs.append(s)
        # 返回结果列表的字符串形式，包含 return 关键字和结果变量列表，并结束函数定义
        return ["return " + ", ".join(outs) + "\nend\n"]
    # 定义一个方法 `_call_printer`，接受一个参数 `routine`
    def _call_printer(self, routine):
        # 用于存储声明的列表
        declarations = []
        # 用于存储代码行的列表
        code_lines = []
        
        # 遍历 `routine.results` 中的每个结果对象
        for result in routine.results:
            # 如果结果是 `Result` 类型的对象
            if isinstance(result, Result):
                # 将结果赋给 `assign_to`
                assign_to = result.result_var
            else:
                # 如果结果不是 `Result` 类型，抛出异常
                raise CodeGenError("unexpected object in Routine results")
            
            # 使用 `_printer_method_with_settings` 方法生成常量、不支持的对象和 Julia 表达式
            constants, not_supported, jl_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)
            
            # 将常量按字符串排序后加入声明列表
            for obj, v in sorted(constants, key=str):
                declarations.append(
                    "%s = %s\n" % (obj, v))
            
            # 将不支持的对象按字符串排序后加入声明列表
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append(
                    "# unsupported: %s\n" % (name))
            
            # 将 Julia 表达式加入代码行列表
            code_lines.append("%s\n" % (jl_expr))
        
        # 返回声明列表和代码行列表的组合结果
        return declarations + code_lines

    # 定义一个方法 `_indent_code`，接受一个参数 `codelines`
    def _indent_code(self, codelines):
        # 创建一个 JuliaCodePrinter 实例 `p`，设置打印选项为 {'human': False, "strict": False}
        p = JuliaCodePrinter({'human': False, "strict": False})
        # 调用 JuliaCodePrinter 的 `indent_code` 方法对代码行进行缩进处理，并返回结果
        return p.indent_code(codelines)

    # 定义一个方法 `dump_jl`，接受参数 `routines`, `f`, `prefix`, `header`, `empty`
    def dump_jl(self, routines, f, prefix, header=True, empty=True):
        # 调用 `dump_code` 方法，并传入参数 `routines`, `f`, `prefix`, `header`, `empty`
        self.dump_code(routines, f, prefix, header, empty)

    # 设置 `dump_jl.extension` 属性为 `code_extension`，这里忽略类型检查
    dump_jl.extension = code_extension  # type: ignore
    
    # 设置 `dump_jl.__doc__` 属性为 `CodeGen.dump_code.__doc__`
    dump_jl.__doc__ = CodeGen.dump_code.__doc__

    # 定义一个列表 `dump_fns`，包含单个元素 `dump_jl`，用于 `CodeGen.write` 方法调用
    # 以确定需要调用哪些 dump 函数
    dump_fns = [dump_jl]
class OctaveCodeGen(CodeGen):
    """Generator for Octave code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.m.

    Octave .m files usually contain one function.  That function name should
    match the filename (``prefix``).  If you pass multiple ``name_expr`` pairs,
    the latter ones are presumed to be private functions accessed by the
    primary function.

    You should only pass inputs to ``argument_sequence``: outputs are ordered
    according to their order in ``name_expr``.

    """

    code_extension = "m"

    def __init__(self, project='project', printer=None):
        super().__init__(project)
        self.printer = printer or OctaveCodePrinter()

    def _get_header(self):
        """Writes a common header for the generated files."""
        # Initialize an empty list to store lines of code
        code_lines = []
        # Generate the header comment string using template with version and project name
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        # Iterate through each line of the header comment
        for line in tmp.splitlines():
            # Check if the line is empty
            if line == '':
                # If empty, add a commented empty line to the code lines list
                code_lines.append("%\n")
            else:
                # Otherwise, format the line and add it as a commented line to code lines
                code_lines.append("%%   %s\n" % line)
        # Return the list of formatted comment lines
        return code_lines

    def _preprocessor_statements(self, prefix):
        # Return an empty list indicating no preprocessor statements for Octave
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        # Initialize an empty list to store code lines
        code_list = []
        # Add the Octave function declaration to the code list
        code_list.append("function ")

        # Outputs
        outs = []
        # Iterate through results in the routine
        for result in routine.results:
            # Check if the result is an instance of Result class
            if isinstance(result, Result):
                # Get the symbol associated with the result name
                s = self._get_symbol(result.name)
            else:
                # Raise an error if result is not of type Result
                raise CodeGenError("unexpected object in Routine results")
            # Append the symbol to outputs list
            outs.append(s)
        # Check number of outputs and format accordingly in code list
        if len(outs) > 1:
            code_list.append("[" + (", ".join(outs)) + "]")
        else:
            code_list.append("".join(outs))
        # Append assignment operator to code list
        code_list.append(" = ")

        # Inputs
        args = []
        # Iterate through arguments in the routine
        for arg in routine.arguments:
            # Check if argument is InputArgument type
            if isinstance(arg, InputArgument):
                # Append symbol associated with argument name to args list
                args.append("%s" % self._get_symbol(arg.name))
            # Raise error for invalid argument types
            elif isinstance(arg, (OutputArgument, InOutArgument)):
                raise CodeGenError("Octave: invalid argument of type %s" %
                                   str(type(arg)))
        # Join arguments with commas and format in code list
        args = ", ".join(args)
        code_list.append("%s(%s)\n" % (routine.name, args))
        # Convert code list to single string list and return
        code_list = [ "".join(code_list) ]
        return code_list

    def _declare_arguments(self, routine):
        # Return an empty list indicating no argument declarations for Octave
        return []

    def _declare_globals(self, routine):
        # Check if there are no global variables in routine
        if not routine.global_vars:
            # Return an empty list if no global variables
            return []
        # Get symbols for all global variables, sort and join them
        s = " ".join(sorted([self._get_symbol(g) for g in routine.global_vars]))
        # Return a list with global variable declaration in Octave syntax
        return ["global " + s + "\n"]

    def _declare_locals(self, routine):
        # Return an empty list indicating no local variable declarations for Octave
        return []

    def _get_routine_ending(self, routine):
        # Return a list containing the end statement to close the routine in Octave
        return ["end\n"]
    def _call_printer(self, routine):
        # 初始化声明列表和代码行列表
        declarations = []
        code_lines = []

        # 遍历例程中的结果
        for result in routine.results:
            # 如果结果是 Result 对象，则获取结果变量
            if isinstance(result, Result):
                assign_to = result.result_var
            else:
                # 如果结果不是 Result 对象，则抛出异常
                raise CodeGenError("unexpected object in Routine results")

            # 调用打印方法以生成常量、不支持的内容和八进制表达式
            constants, not_supported, oct_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)

            # 将常量声明添加到声明列表中
            for obj, v in sorted(constants, key=str):
                declarations.append(
                    "  %s = %s;  %% constant\n" % (obj, v))
            # 将不支持的声明添加到声明列表中
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append(
                    "  %% unsupported: %s\n" % (name))
            
            # 将八进制表达式添加到代码行列表中
            code_lines.append("%s\n" % (oct_expr))
        
        # 返回声明列表和代码行列表的组合
        return declarations + code_lines

    def _indent_code(self, codelines):
        # 调用打印方法来对代码行进行缩进处理
        return self._printer_method_with_settings(
            'indent_code', {"human": False, "strict": False}, codelines)

    def dump_m(self, routines, f, prefix, header=True, empty=True, inline=True):
        # 生成预处理语句
        code_lines = self._preprocessor_statements(prefix)

        # 遍历例程列表
        for i, routine in enumerate(routines):
            # 添加空行（如果允许）以分隔例程
            if i > 0:
                if empty:
                    code_lines.append("\n")
            
            # 添加例程的开头部分
            code_lines.extend(self._get_routine_opening(routine))

            # 如果是第一个例程，则检查函数名是否与前缀匹配，并根据需要添加头部信息
            if i == 0:
                if routine.name != prefix:
                    raise ValueError('Octave function name should match prefix')
                if header:
                    code_lines.append("%" + prefix.upper() +
                                      "  Autogenerated by SymPy\n")
                    code_lines.append(''.join(self._get_header()))

            # 声明参数、全局变量和局部变量
            code_lines.extend(self._declare_arguments(routine))
            code_lines.extend(self._declare_globals(routine))
            code_lines.extend(self._declare_locals(routine))

            # 添加空行（如果允许）以分隔不同部分
            if empty:
                code_lines.append("\n")

            # 调用打印方法以生成例程中的代码行
            code_lines.extend(self._call_printer(routine))

            # 添加空行（如果允许）以分隔不同部分
            if empty:
                code_lines.append("\n")

            # 添加例程的结尾部分
            code_lines.extend(self._get_routine_ending(routine))

        # 对生成的代码进行缩进处理
        code_lines = self._indent_code(''.join(code_lines))

        # 如果存在代码行，则将其写入文件对象 f 中
        if code_lines:
            f.write(code_lines)

    # 设置 dump_m 方法的扩展名属性
    dump_m.extension = code_extension  # type: ignore

    # 将 dump_m 方法的文档字符串设置为 CodeGen.dump_code 的文档字符串
    dump_m.__doc__ = CodeGen.dump_code.__doc__

    # 定义 dump 函数列表，用于指示 CodeGen.write 要调用的 dump 函数
    dump_fns = [dump_m]
    # RustCodeGen 类，继承自 CodeGen 类，用于生成 Rust 代码
    class RustCodeGen(CodeGen):

        """Generator for Rust code.
        
        The .write() method inherited from CodeGen will output a code file
        <prefix>.rs
        
        """

        # Rust 代码文件的扩展名为 "rs"
        code_extension = "rs"

        def __init__(self, project="project", printer=None):
            # 调用父类 CodeGen 的构造函数，设置项目名称和打印机对象
            super().__init__(project=project)
            self.printer = printer or RustCodePrinter()

        def _get_header(self):
            """Writes a common header for the generated files."""
            code_lines = []
            code_lines.append("/*\n")
            # 生成文件头部注释，包括版本号和项目名称
            tmp = header_comment % {"version": sympy_version,
                                    "project": self.project}
            for line in tmp.splitlines():
                # 将注释行按规定格式居中处理并添加到列表中
                code_lines.append((" *%s" % line.center(76)).rstrip() + "\n")
            code_lines.append(" */\n")
            return code_lines

        def get_prototype(self, routine):
            """Returns a string for the function prototype of the routine.

            If the routine has multiple result objects, an CodeGenError is
            raised.

            See: https://en.wikipedia.org/wiki/Function_prototype

            """
            # 获取 routine 的函数原型字符串，包括参数和返回类型
            results = [i.get_datatype('Rust') for i in routine.results]

            if len(results) == 1:
                rstype = " -> " + results[0]
            elif len(routine.results) > 1:
                rstype = " -> (" + ", ".join(results) + ")"
            else:
                rstype = ""

            type_args = []
            for arg in routine.arguments:
                name = self.printer.doprint(arg.name)
                if arg.dimensions or isinstance(arg, ResultBase):
                    type_args.append(("*%s" % name, arg.get_datatype('Rust')))
                else:
                    type_args.append((name, arg.get_datatype('Rust')))
            arguments = ", ".join([ "%s: %s" % t for t in type_args])
            return "fn %s(%s)%s" % (routine.name, arguments, rstype)

        def _preprocessor_statements(self, prefix):
            code_lines = []
            # 返回空列表，表示没有预处理语句
            # code_lines.append("use std::f64::consts::*;\n")
            return code_lines

        def _get_routine_opening(self, routine):
            # 获取 routine 函数的开头部分，包括函数原型
            prototype = self.get_prototype(routine)
            return ["%s {\n" % prototype]

        def _declare_arguments(self, routine):
            # 参数在函数原型中声明，此处返回空列表
            # arguments are declared in prototype
            return []

        def _declare_globals(self, routine):
            # Rust 中全局变量通常不会在函数内显式声明，返回空列表
            # global variables are not explicitly declared within C functions
            return []

        def _declare_locals(self, routine):
            # 局部变量通常在循环语句中声明，此处返回空列表
            # loop variables are declared in loop statement
            return []
    def _call_printer(self, routine):
        # 初始化空列表，用于存储函数体的代码行
        code_lines = []
        # 初始化空列表，用于存储声明语句
        declarations = []
        # 初始化空列表，用于存储返回值
        returns = []

        # 构建一个需要在函数体中解引用的符号列表。这些是通过引用指针传递的参数，排除数组。
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        # 遍历例行程序的结果
        for result in routine.results:
            if isinstance(result, Result):
                # 确定赋值目标
                assign_to = result.result_var
                returns.append(str(result.result_var))
            else:
                # 如果结果不是期望的类型，则抛出错误
                raise CodeGenError("unexpected object in Routine results")

            # 获取打印方法的常量、不支持项和 Rust 表达式
            constants, not_supported, rs_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)

            # 对常量按字符串排序后，构建声明语句
            for name, value in sorted(constants, key=str):
                declarations.append("const %s: f64 = %s;\n" % (name, value))

            # 对不支持的项按字符串排序后，构建注释
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append("// unsupported: %s\n" % (name))

            # 构建代码行，包含 Rust 表达式
            code_lines.append("let %s\n" % rs_expr);

        # 如果返回值超过一个，则将其组合成一个元组
        if len(returns) > 1:
            returns = ['(' + ', '.join(returns) + ')']

        # 添加空行到返回值列表末尾
        returns.append('\n')

        # 返回声明语句、代码行和返回值的组合列表
        return declarations + code_lines + returns

    def _get_routine_ending(self, routine):
        # 返回一个表示函数结束的列表
        return ["}\n"]

    def dump_rs(self, routines, f, prefix, header=True, empty=True):
        # 调用 dump_code 方法来将 Rust 代码转储到文件中
        self.dump_code(routines, f, prefix, header, empty)

    # 将 code_extension 赋值给 dump_rs 的 extension 属性，类型为忽略类型（type: ignore）
    dump_rs.extension = code_extension  # type: ignore
    # 将 dump_code 方法的文档字符串赋给 dump_rs 方法
    dump_rs.__doc__ = CodeGen.dump_code.__doc__

    # 这个 dump 函数列表用于 CodeGen.write 方法，以确定需要调用哪些 dump 函数
    dump_fns = [dump_rs]
# 根据给定的语言生成代码生成器对象
def get_code_generator(language, project=None, standard=None, printer=None):
    # 如果语言是 'C'，根据标准选择不同的 C 语言版本
    if language == 'C':
        if standard is None:
            pass  # 如果未指定标准，继续使用默认语言 'C'
        elif standard.lower() == 'c89':
            language = 'C89'  # 如果标准是 'c89'，选择 C89 版本
        elif standard.lower() == 'c99':
            language = 'C99'  # 如果标准是 'c99'，选择 C99 版本
    # 根据语言选择相应的代码生成器类
    CodeGenClass = {"C": CCodeGen, "C89": C89CodeGen, "C99": C99CodeGen,
                    "F95": FCodeGen, "JULIA": JuliaCodeGen,
                    "OCTAVE": OctaveCodeGen,
                    "RUST": RustCodeGen}.get(language.upper())
    # 如果未找到对应的代码生成器类，抛出异常
    if CodeGenClass is None:
        raise ValueError("Language '%s' is not supported." % language)
    # 返回相应语言的代码生成器对象，传入项目名称和打印机对象（如果有）
    return CodeGenClass(project, printer)


#
# Friendly functions
#


def codegen(name_expr, language=None, prefix=None, project="project",
            to_files=False, header=True, empty=True, argument_sequence=None,
            global_vars=None, standard=None, code_gen=None, printer=None):
    """Generate source code for expressions in a given language.

    Parameters
    ==========

    name_expr : tuple, or list of tuples
        A single (name, expression) tuple or a list of (name, expression)
        tuples.  Each tuple corresponds to a routine.  If the expression is
        an equality (an instance of class Equality) the left hand side is
        considered an output argument.  If expression is an iterable, then
        the routine will have multiple outputs.

    language : string,
        A string that indicates the source code language.  This is case
        insensitive.  Currently, 'C', 'F95' and 'Octave' are supported.
        'Octave' generates code compatible with both Octave and Matlab.

    prefix : string, optional
        A prefix for the names of the files that contain the source code.
        Language-dependent suffixes will be appended.  If omitted, the name
        of the first name_expr tuple is used.

    project : string, optional
        A project name, used for making unique preprocessor instructions.
        [default: "project"]

    to_files : bool, optional
        When True, the code will be written to one or more files with the
        given prefix, otherwise strings with the names and contents of
        these files are returned. [default: False]

    header : bool, optional
        When True, a header is written on top of each source file.
        [default: True]

    empty : bool, optional
        When True, empty lines are used to structure the code.
        [default: True]

    argument_sequence : iterable, optional
        Sequence of arguments for the routine in a preferred order.  A
        CodeGenError is raised if required arguments are missing.
        Redundant arguments are used without warning.  If omitted,
        arguments will be ordered alphabetically, but with all input
        arguments first, and then output or in-out arguments.
    # global_vars : iterable, optional
    #     Sequence of global variables used by the routine. Variables
    #     listed here will not show up as function arguments.

    # standard : string, optional

    # code_gen : CodeGen instance, optional
    #     An instance of a CodeGen subclass. Overrides ``language``.

    # printer : Printer instance, optional
    #     An instance of a Printer subclass.

    # Examples
    # ========

    # >>> from sympy.utilities.codegen import codegen
    # >>> from sympy.abc import x, y, z
    # >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    # ...     ("f", x+y*z), "C89", "test", header=False, empty=False)
    # >>> print(c_name)
    # test.c
    # >>> print(c_code)
    # #include "test.h"
    # #include <math.h>
    # double f(double x, double y, double z) {
    #    double f_result;
    #    f_result = x + y*z;
    #    return f_result;
    # }
    # <BLANKLINE>
    # >>> print(h_name)
    # test.h
    # >>> print(c_header)
    # #ifndef PROJECT__TEST__H
    # #define PROJECT__TEST__H
    # double f(double x, double y, double z);
    # #endif
    # <BLANKLINE>

    # Another example using Equality objects to give named outputs. Here the
    # filename (prefix) is taken from the first (name, expr) pair.

    # >>> from sympy.abc import f, g
    # >>> from sympy import Eq
    # >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    # ...      [("myfcn", x + y), ("fcn2", [Eq(f, 2*x), Eq(g, y)])],
    # ...      "C99", header=False, empty=False)
    # >>> print(c_name)
    # myfcn.c
    # >>> print(c_code)
    # #include "myfcn.h"
    # #include <math.h>
    # double myfcn(double x, double y) {
    #    double myfcn_result;
    #    myfcn_result = x + y;
    #    return myfcn_result;
    # }
    # void fcn2(double x, double y, double *f, double *g) {
    #    (*f) = 2*x;
    #    (*g) = y;
    # }
    # <BLANKLINE>

    # If the generated function(s) will be part of a larger project where various
    # global variables have been defined, the 'global_vars' option can be used
    # to remove the specified variables from the function signature

    # >>> from sympy.utilities.codegen import codegen
    # >>> from sympy.abc import x, y, z
    # >>> [(f_name, f_code), header] = codegen(
    # ...     ("f", x+y*z), "F95", header=False, empty=False,
    # ...     argument_sequence=(x, y), global_vars=(z,))
    # >>> print(f_code)
    # REAL*8 function f(x, y)
    # implicit none
    # REAL*8, intent(in) :: x
    # REAL*8, intent(in) :: y
    # f = x + y*z
    # end function
    # <BLANKLINE>

    """

    # Initialize the code generator.
    # 如果未指定语言，则需要确保 code_gen 已经定义，否则会引发异常
    if language is None:
        if code_gen is None:
            raise ValueError("Need either language or code_gen")
    else:
        # 如果同时指定了 language 和 code_gen，会引发异常
        if code_gen is not None:
            raise ValueError("You cannot specify both language and code_gen.")
        # 根据指定的 language、project、standard 和 printer，获取 CodeGen 实例
        code_gen = get_code_generator(language, project, standard, printer)

    # 如果 name_expr 的第一个元素是字符串，则将其转换为包含单个元组的列表
    if isinstance(name_expr[0], str):
        name_expr = [name_expr]
    # 如果未提供前缀，则使用第一个名称表达式的首字符作为前缀
    if prefix is None:
        prefix = name_expr[0][0]

    # 根据名称和表达式创建适用于该code_gen的Routine（程序段）
    routines = []
    for name, expr in name_expr:
        routines.append(code_gen.routine(name, expr, argument_sequence,
                                         global_vars))

    # 将生成的程序段列表写入代码生成器，并返回生成的结果
    return code_gen.write(routines, prefix, to_files, header, empty)
# 根据提供的表达式、参数等信息创建一个例行程序（Routine）
def make_routine(name, expr, argument_sequence=None,
                 global_vars=None, language="F95"):
    """A factory that makes an appropriate Routine from an expression.

    Parameters
    ==========

    name : string
        The name of this routine in the generated code.

    expr : expression or list/tuple of expressions
        A SymPy expression that the Routine instance will represent.  If
        given a list or tuple of expressions, the routine will be
        considered to have multiple return values and/or output arguments.

    argument_sequence : list or tuple, optional
        List arguments for the routine in a preferred order.  If omitted,
        the results are language dependent, for example, alphabetical order
        or in the same order as the given expressions.

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    language : string, optional
        Specify a target language.  The Routine itself should be
        language-agnostic but the precise way one is created, error
        checking, etc depend on the language.  [default: "F95"].

    Notes
    =====

    A decision about whether to use output arguments or return values is made
    depending on both the language and the particular mathematical expressions.
    For an expression of type Equality, the left hand side is typically made
    into an OutputArgument (or perhaps an InOutArgument if appropriate).
    Otherwise, typically, the calculated expression is made a return values of
    the routine.

    Examples
    ========

    >>> from sympy.utilities.codegen import make_routine
    >>> from sympy.abc import x, y, f, g
    >>> from sympy import Eq
    >>> r = make_routine('test', [Eq(f, 2*x), Eq(g, x + y)])
    >>> [arg.result_var for arg in r.results]
    []
    >>> [arg.name for arg in r.arguments]
    [x, y, f, g]
    >>> [arg.name for arg in r.result_variables]
    [f, g]
    >>> r.local_vars
    set()

    Another more complicated example with a mixture of specified and
    automatically-assigned names.  Also has Matrix output.

    >>> from sympy import Matrix
    >>> r = make_routine('fcn', [x*y, Eq(f, 1), Eq(g, x + g), Matrix([[x, 2]])])
    >>> [arg.result_var for arg in r.results]  # doctest: +SKIP
    [result_5397460570204848505]
    >>> [arg.expr for arg in r.results]
    [x*y]
    >>> [arg.name for arg in r.arguments]  # doctest: +SKIP
    [x, y, f, g, out_8598435338387848786]

    We can examine the various arguments more closely:

    >>> from sympy.utilities.codegen import (InputArgument, OutputArgument,
    ...                                      InOutArgument)
    >>> [a.name for a in r.arguments if isinstance(a, InputArgument)]
    [x, y]

    >>> [a.name for a in r.arguments if isinstance(a, OutputArgument)]  # doctest: +SKIP
    [f, out_8598435338387848786]
    """
    # 使用列表推导式从参数列表 r.arguments 中筛选出所有类型为 OutputArgument 的参数的表达式
    [a.expr for a in r.arguments if isinstance(a, OutputArgument)]
    # 返回结果为包含所有 OutputArgument 参数表达式的列表
    [1, Matrix([[x, 2]])]

    # 使用列表推导式从参数列表 r.arguments 中筛选出所有类型为 InOutArgument 的参数的名称
    [a.name for a in r.arguments if isinstance(a, InOutArgument)]
    # 返回结果为包含所有 InOutArgument 参数名称的列表
    [g]
    # 使用列表推导式从参数列表 r.arguments 中筛选出所有类型为 InOutArgument 的参数的表达式
    [a.expr for a in r.arguments if isinstance(a, InOutArgument)]
    # 返回结果为包含所有 InOutArgument 参数表达式的列表
    [g + x]

    """

    # 初始化一个新的代码生成器，根据指定的语言创建对应的实例
    code_gen = get_code_generator(language)

    # 调用代码生成器的 routine 方法，生成特定名称、表达式、参数序列和全局变量的代码
    return code_gen.routine(name, expr, argument_sequence, global_vars)
```