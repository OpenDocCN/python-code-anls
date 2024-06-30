# `D:\src\scipysrc\sympy\sympy\utilities\autowrap.py`

```
# 编译代码生成的模块，将二进制包装为 Python 可用的形式
"""Module for compiling codegen output, and wrap the binary for use in
python.

.. note:: To use the autowrap module it must first be imported

   >>> from sympy.utilities.autowrap import autowrap

This module provides a common interface for different external backends, such
as f2py, fwrap, Cython, SWIG(?) etc. (Currently only f2py and Cython are
implemented) The goal is to provide access to compiled binaries of acceptable
performance with a one-button user interface, e.g.,

    >>> from sympy.abc import x,y
    >>> expr = (x - y)**25
    >>> flat = expr.expand()
    >>> binary_callable = autowrap(flat)
    >>> binary_callable(2, 3)
    -1.0

Although a SymPy user might primarily be interested in working with
mathematical expressions and not in the details of wrapping tools
needed to evaluate such expressions efficiently in numerical form,
the user cannot do so without some understanding of the
limits in the target language. For example, the expanded expression
contains large coefficients which result in loss of precision when
computing the expression:

    >>> binary_callable(3, 2)
    0.0
    >>> binary_callable(4, 5), binary_callable(5, 4)
    (-22925376.0, 25165824.0)

Wrapping the unexpanded expression gives the expected behavior:

    >>> e = autowrap(expr)
    >>> e(4, 5), e(5, 4)
    (-1.0, 1.0)

The callable returned from autowrap() is a binary Python function, not a
SymPy object.  If it is desired to use the compiled function in symbolic
expressions, it is better to use binary_function() which returns a SymPy
Function object.  The binary callable is attached as the _imp_ attribute and
invoked when a numerical evaluation is requested with evalf(), or with
lambdify().

    >>> from sympy.utilities.autowrap import binary_function
    >>> f = binary_function('f', expr)
    >>> 2*f(x, y) + y
    y + 2*f(x, y)
    >>> (2*f(x, y) + y).evalf(2, subs={x: 1, y:2})
    0.e-110

When is this useful?

    1) For computations on large arrays, Python iterations may be too slow,
       and depending on the mathematical expression, it may be difficult to
       exploit the advanced index operations provided by NumPy.

    2) For *really* long expressions that will be called repeatedly, the
       compiled binary should be significantly faster than SymPy's .evalf()

    3) If you are generating code with the codegen utility in order to use
       it in another project, the automatic Python wrappers let you test the
       binaries immediately from within SymPy.

    4) To create customized ufuncs for use with numpy arrays.
       See *ufuncify*.

When is this module NOT the best approach?
    # 如果你真的关心速度或内存优化，你可能会通过直接使用包装工具和低级代码来获得更好的结果。
    # 然而，由此工具生成的文件可能提供一个有用的起点和参考代码。如果提供了关键字参数 tempdir="path/to/files/"，
    # 临时文件将被保留在指定的路径中。
    
    # 如果数组计算可以轻松地由 numpy 处理，并且你不需要这些二进制文件用于另一个项目。
"""

import sys  # 导入 sys 模块，用于访问系统相关的功能
import os  # 导入 os 模块，用于操作系统功能
import shutil  # 导入 shutil 模块，提供高级的文件操作功能
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
from subprocess import STDOUT, CalledProcessError, check_output  # 从 subprocess 模块导入相关函数和异常处理
from string import Template  # 导入 Template 类，用于字符串模板替换
from warnings import warn  # 导入 warn 函数，用于发出警告信息

from sympy.core.cache import cacheit  # 导入 sympy 相关模块
from sympy.core.function import Lambda
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.utilities.codegen import (make_routine, get_code_generator,  # 导入 sympy 提供的代码生成工具
                                     OutputArgument, InOutArgument,
                                     InputArgument, CodeGenArgumentListError,
                                     Result, ResultBase, C99CodeGen)
from sympy.utilities.iterables import iterable  # 导入 iterable 函数，用于检查对象是否可迭代
from sympy.utilities.lambdify import implemented_function  # 导入 implemented_function 函数，用于生成函数的数值评估
from sympy.utilities.decorator import doctest_depends_on  # 导入 doctest_depends_on 装饰器，用于依赖测试的装饰器

_doctest_depends_on = {'exe': ('f2py', 'gfortran', 'gcc'),  # 设置 doctest 的依赖项字典
                       'modules': ('numpy',)}


class CodeWrapError(Exception):  # 定义一个代码包装错误的异常类
    pass


class CodeWrapper:  # 定义代码包装器的基类
    """Base Class for code wrappers"""
    _filename = "wrapped_code"  # 类变量，包装后代码的文件名前缀
    _module_basename = "wrapper_module"  # 类变量，包装后代码的模块名前缀
    _module_counter = 0  # 类变量，模块计数器，用于生成唯一的文件名和模块名后缀

    @property
    def filename(self):  # 实例属性，返回包装后代码的文件名
        return "%s_%s" % (self._filename, CodeWrapper._module_counter)

    @property
    def module_name(self):  # 实例属性，返回包装后代码的模块名
        return "%s_%s" % (self._module_basename, CodeWrapper._module_counter)

    def __init__(self, generator, filepath=None, flags=[], verbose=False):
        """
        generator -- the code generator to use
        """
        self.generator = generator  # 实例化时传入的代码生成器对象
        self.filepath = filepath  # 实例化时传入的文件路径
        self.flags = flags  # 实例化时传入的标志列表
        self.quiet = not verbose  # 根据 verbose 参数设置静音模式标志

    @property
    def include_header(self):  # 实例属性，确定是否包含头部信息
        return bool(self.filepath)

    @property
    def include_empty(self):  # 实例属性，确定是否包含空文件
        return bool(self.filepath)

    def _generate_code(self, main_routine, routines):  # 定义生成代码的方法
        routines.append(main_routine)  # 将主要例程添加到例程列表中
        self.generator.write(  # 使用代码生成器对象将代码写入文件
            routines, self.filename, True, self.include_header,
            self.include_empty)
    # 定义一个方法，用于包装给定的代码块为一个函数，返回函数对象
    def wrap_code(self, routine, helpers=None):
        # 如果未提供 helpers 参数，则设为一个空列表
        helpers = helpers or []
        # 如果存在文件路径，则获取其绝对路径作为工作目录
        if self.filepath:
            workdir = os.path.abspath(self.filepath)
        else:
            # 否则创建一个临时目录作为工作目录
            workdir = tempfile.mkdtemp("_sympy_compile")
        # 如果工作目录不存在，则创建之
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        # 记录当前工作目录
        oldwork = os.getcwd()
        # 切换工作目录到创建的或者指定的工作目录
        os.chdir(workdir)
        try:
            # 将工作目录添加到系统路径中，以便 Python 模块能够找到生成的文件
            sys.path.append(workdir)
            # 生成代码（由子类实现具体细节）
            self._generate_code(routine, helpers)
            # 准备文件（由子类实现具体细节）
            self._prepare_files(routine)
            # 处理文件（由子类实现具体细节）
            self._process_files(routine)
            # 动态导入生成的模块
            mod = __import__(self.module_name)
        finally:
            # 从系统路径中移除工作目录
            sys.path.remove(workdir)
            # 增加模块计数器
            CodeWrapper._module_counter += 1
            # 恢复原始工作目录
            os.chdir(oldwork)
            # 如果工作目录是临时创建的，则尝试删除它
            if not self.filepath:
                try:
                    shutil.rmtree(workdir)
                except OSError:
                    # 可能在 Windows 上会有一些问题
                    pass

        # 返回包装后的函数对象
        return self._get_wrapped_function(mod, routine.name)

    # 处理生成的文件，执行指定的命令并获取其输出
    def _process_files(self, routine):
        # 获取预设的命令并扩展参数
        command = self.command
        command.extend(self.flags)
        try:
            # 执行命令，并捕获其输出
            retoutput = check_output(command, stderr=STDOUT)
        except CalledProcessError as e:
            # 如果命令执行出错，则抛出自定义异常
            raise CodeWrapError(
                "Error while executing command: %s. Command output is:\n%s" % (
                    " ".join(command), e.output.decode('utf-8')))
        # 如果非安静模式，打印命令输出
        if not self.quiet:
            print(retoutput)
class DummyWrapper(CodeWrapper):
    """Class used for testing independent of backends """

    template = """# dummy module for testing of SymPy
def %(name)s():
    return "%(expr)s"
%(name)s.args = "%(args)s"
%(name)s.returns = "%(retvals)s"
"""


# DummyWrapper 类继承自 CodeWrapper，用于测试独立于后端的功能

    # 字符串模板，用于生成测试 SymPy 代码的框架
def _prepare_files(self, routine):
        return


# 准备文件的方法，但当前未实现任何功能，只是简单返回
def _generate_code(self, routine, helpers):
        with open('%s.py' % self.module_name, 'w') as f:
            # 将 routine 的结果变量打印为字符串，并用逗号连接
            printed = ", ".join(
                [str(res.expr) for res in routine.result_variables])
            # 将 OutputArguments 转换为类似 f2py 的返回值
            args = filter(lambda x: not isinstance(
                x, OutputArgument), routine.arguments)
            # 准备返回值列表
            retvals = []
            for val in routine.result_variables:
                if isinstance(val, Result):
                    retvals.append('nameless')
                else:
                    retvals.append(val.result_var)

            # 使用 DummyWrapper.template 格式化生成代码，并写入文件
            print(DummyWrapper.template % {
                'name': routine.name,
                'expr': printed,
                'args': ", ".join([str(a.name) for a in args]),
                'retvals': ", ".join([str(val) for val in retvals])
            }, end="", file=f)


# 生成代码的方法，将根据传入的 routine 和 helpers 对象生成相应的代码文件
def _process_files(self, routine):
        return


# 处理文件的方法，目前并未实现任何功能，只是简单返回
    @classmethod
    # 获取被包装函数的方法，从给定的 mod 中获取指定名称的函数对象
def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)


class CythonCodeWrapper(CodeWrapper):
    """Wrapper that uses Cython"""

    setup_template = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {cythonize_options}
{np_import}
ext_mods = [Extension(
    {ext_args},
    include_dirs={include_dirs},
    library_dirs={library_dirs},
    libraries={libraries},
    extra_compile_args={extra_compile_args},
    extra_link_args={extra_link_args}
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
"""


# CythonCodeWrapper 类继承自 CodeWrapper，用于使用 Cython 进行封装

    # 设置模板，用于生成 Cython 扩展模块的 setup.py 文件
    _cythonize_options = {'compiler_directives':{'language_level' : "3"}}

    # 导入 numpy 并进行 cython 的相关设置
    pyx_imports = (
        "import numpy as np\n"
        "cimport numpy as np\n\n")

    # Cython 文件的头部模板，包括从指定头文件中导入原型
    pyx_header = (
        "cdef extern from '{header_file}.h':\n"
        "    {prototype}\n\n")

    # Cython 函数定义的模板，包括函数名称、参数字符串、声明和函数体
    pyx_func = (
        "def {name}_c({arg_string}):\n"
        "\n"
        "{declarations}"
        "{body}")

    # 标准的 C99 编译标志
    std_compile_flag = '-std=c99'


# 设置模板字符串，用于生成 Cython 的 setup.py 文件内容，包括导入必要的模块和设置
    def __init__(self, *args, **kwargs):
        """
        实例化一个 Cython 代码包装器。

        下列可选参数将传递给 `setuptools.Extension` 用于构建 Python 扩展模块。
        可以查阅其文档以获取更多信息。

        Parameters
        ==========
        include_dirs : [list of strings]
            用于搜索 C/C++ 头文件的目录列表（Unix 形式，用于可移植性）。
        library_dirs : [list of strings]
            在链接时搜索 C/C++ 库文件的目录列表。
        libraries : [list of strings]
            要链接的库名称列表（不是文件名或路径）。
        extra_compile_args : [list of strings]
            编译源文件时使用的任何额外的平台和编译器特定信息。
            对于支持“命令行”的平台和编译器，这通常是一组命令行参数，
            但对于其他平台，它可以是任何内容。注意将会追加 `std_compile_flag` 属性到这个列表中。
        extra_link_args : [list of strings]
            链接对象文件以创建扩展或创建新的静态 Python 解释器时使用的任何额外平台和编译器特定信息。
            与 `extra_compile_args` 的解释类似。
        cythonize_options : [dictionary]
            传递给 cythonize 的关键字参数。

        """

        self._include_dirs = kwargs.pop('include_dirs', [])
        self._library_dirs = kwargs.pop('library_dirs', [])
        self._libraries = kwargs.pop('libraries', [])
        self._extra_compile_args = kwargs.pop('extra_compile_args', [])
        self._extra_compile_args.append(self.std_compile_flag)  # 将 std_compile_flag 添加到额外的编译参数列表中
        self._extra_link_args = kwargs.pop('extra_link_args', [])
        self._cythonize_options = kwargs.pop('cythonize_options', self._cythonize_options)

        self._need_numpy = False  # 设定一个标志位，表明是否需要 NumPy

        super().__init__(*args, **kwargs)  # 调用父类的构造函数并传递所有参数

    @property
    def command(self):
        """
        返回一个命令列表，用于执行编译 Python 扩展模块的操作。

        Returns
        =======
        command : list
            包含命令及其参数的列表，用于执行编译操作。
        """
        command = [sys.executable, "setup.py", "build_ext", "--inplace"]
        return command
    # 准备文件用于构建，将生成的 Cython 文件保存为 .pyx 格式和指定的代码文件名
    def _prepare_files(self, routine, build_dir=os.curdir):
        # 注意：build_dir 用于测试目的。
        # 生成的 Cython 文件名
        pyxfilename = self.module_name + '.pyx'
        # 构建的代码文件名，格式为 module_name.code_extension
        codefilename = "%s.%s" % (self.filename, self.generator.code_extension)

        # 生成 .pyx 文件
        with open(os.path.join(build_dir, pyxfilename), 'w') as f:
            # 将指定的 routine 列表转换为 Cython 代码并写入文件 f 中
            self.dump_pyx([routine], f, self.filename)

        # 生成 setup.py 文件
        ext_args = [repr(self.module_name), repr([pyxfilename, codefilename])]
        # 如果需要使用 numpy，则添加 numpy 的导入语句和 include 路径
        if self._need_numpy:
            np_import = 'import numpy as np\n'
            self._include_dirs.append('np.get_include()')
        else:
            np_import = ''

        with open(os.path.join(build_dir, 'setup.py'), 'w') as f:
            # 将 setup.py 的模板内容写入文件 f 中，替换相关的变量内容
            includes = str(self._include_dirs).replace("'np.get_include()'",
                                                       'np.get_include()')
            f.write(self.setup_template.format(
                ext_args=", ".join(ext_args),
                np_import=np_import,
                include_dirs=includes,
                library_dirs=self._library_dirs,
                libraries=self._libraries,
                extra_compile_args=self._extra_compile_args,
                extra_link_args=self._extra_link_args,
                cythonize_options=self._cythonize_options
            ))

    @classmethod
    # 获取被包装的函数对象，返回模块中指定名称函数的包装版本
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name + '_c')
    def dump_pyx(self, routines, f, prefix):
        """
        Write a Cython file with Python wrappers

        This method generates a Cython file containing Python wrappers for C routines,
        with references to a specified header file.

        Arguments
        ---------
        routines
            List of Routine instances representing the C routines to wrap.
        f
            File-like object to write the generated Cython file.
        prefix
            The filename prefix used to refer to the appropriate header file.
            Only the basename of the prefix is used.
        """
        # Initialize lists to store headers and functions
        headers = []
        functions = []

        # Iterate over each routine provided
        for routine in routines:
            # Generate C function prototype for the routine
            prototype = self.generator.get_prototype(routine)

            # Add C function header import statement to headers list
            headers.append(self.pyx_header.format(header_file=prefix,
                                                  prototype=prototype))

            # Partition C function arguments into Python categories
            py_rets, py_args, py_loc, py_inf = self._partition_args(routine.arguments)

            # Generate function prototype
            name = routine.name
            arg_string = ", ".join(self._prototype_arg(arg) for arg in py_args)

            # Generate local variable declarations
            local_decs = []
            for arg, val in py_inf.items():
                proto = self._prototype_arg(arg)
                mat, ind = [self._string_var(v) for v in val]
                local_decs.append("    cdef {} = {}.shape[{}]".format(proto, mat, ind))
            local_decs.extend(["    cdef {}".format(self._declare_arg(a)) for a in py_loc])
            declarations = "\n".join(local_decs)
            if declarations:
                declarations = declarations + "\n"

            # Generate function body
            args_c = ", ".join([self._call_arg(a) for a in routine.arguments])
            rets = ", ".join([self._string_var(r.name) for r in py_rets])
            if routine.results:
                body = '    return %s(%s)' % (routine.name, args_c)
                if rets:
                    body = body + ', ' + rets
            else:
                body = '    %s(%s)\n' % (routine.name, args_c)
                body = body + '    return ' + rets

            # Format the function and add to functions list
            functions.append(self.pyx_func.format(name=name, arg_string=arg_string,
                                                  declarations=declarations, body=body))

        # Write necessary imports to the file
        if self._need_numpy:
            f.write(self.pyx_imports)

        # Write headers and functions to the file
        f.write('\n'.join(headers))
        f.write('\n'.join(functions))
    # 将函数参数分组到不同的类别中
    def _partition_args(self, args):
        """Group function arguments into categories."""
        # 存储用于返回值的参数列表
        py_args = []
        # 存储用于输入输出的参数列表
        py_returns = []
        # 存储本地变量参数列表
        py_locals = []
        # 存储推断得到的参数字典
        py_inferred = {}

        # 遍历所有参数
        for arg in args:
            # 如果参数是 OutputArgument 类型，则加入返回值和本地变量列表
            if isinstance(arg, OutputArgument):
                py_returns.append(arg)
                py_locals.append(arg)
            # 如果参数是 InOutArgument 类型，则加入返回值和输入输出参数列表
            elif isinstance(arg, InOutArgument):
                py_returns.append(arg)
                py_args.append(arg)
            # 否则，将参数加入输入参数列表
            else:
                py_args.append(arg)
            
            # 如果参数是 InputArgument 或者 InOutArgument，并且具有维度信息，则处理推断参数
            if isinstance(arg, (InputArgument, InOutArgument)) and arg.dimensions:
                # 提取维度信息，转换为符号的形式
                dims = [d[1] + 1 for d in arg.dimensions]
                sym_dims = [(i, d) for (i, d) in enumerate(dims) if isinstance(d, Symbol)]
                # 将推断的符号维度与参数名及索引关联起来
                for (i, d) in sym_dims:
                    py_inferred[d] = (arg.name, i)
        
        # 根据推断得到的参数，调整参数列表顺序
        for arg in args:
            if arg.name in py_inferred:
                py_inferred[arg] = py_inferred.pop(arg.name)
        
        # 从 py_args 中过滤掉已推断的参数
        py_args = [a for a in py_args if a not in py_inferred]
        
        # 返回分类后的参数列表及推断结果
        return py_returns, py_args, py_locals, py_inferred

    # 根据参数生成 Cython 函数原型声明
    def _prototype_arg(self, arg):
        mat_dec = "np.ndarray[{mtype}, ndim={ndim}] {name}"
        np_types = {'double': 'np.double_t',
                    'int': 'np.int_t'}
        # 获取参数的 C 语言数据类型
        t = arg.get_datatype('c')
        if arg.dimensions:
            # 如果参数具有维度信息，则需要 NumPy 支持
            self._need_numpy = True
            ndim = len(arg.dimensions)
            mtype = np_types[t]
            # 根据参数类型、维度生成 NumPy 数组的声明
            return mat_dec.format(mtype=mtype, ndim=ndim, name=self._string_var(arg.name))
        else:
            # 否则，直接返回参数类型及其名称
            return "%s %s" % (t, self._string_var(arg.name))

    # 根据参数生成声明语句
    def _declare_arg(self, arg):
        # 根据参数生成 Cython 的原型声明
        proto = self._prototype_arg(arg)
        if arg.dimensions:
            # 如果参数具有维度信息，则生成对应的 NumPy 数组声明
            shape = '(' + ','.join(self._string_var(i[1] + 1) for i in arg.dimensions) + ')'
            return proto + " = np.empty({shape})".format(shape=shape)
        else:
            # 否则，生成简单的赋值语句
            return proto + " = 0"

    # 根据参数生成调用语句
    def _call_arg(self, arg):
        if arg.dimensions:
            # 如果参数具有维度信息，则生成指向数组数据的指针形式
            t = arg.get_datatype('c')
            return "<{}*> {}.data".format(t, self._string_var(arg.name))
        elif isinstance(arg, ResultBase):
            # 如果参数是 ResultBase 类型，则生成其地址传递形式
            return "&{}".format(self._string_var(arg.name))
        else:
            # 否则，直接返回参数名称
            return self._string_var(arg.name)

    # 将变量名转换为字符串形式
    def _string_var(self, var):
        printer = self.generator.printer.doprint
        return printer(var)
class F2PyCodeWrapper(CodeWrapper):
    """Wrapper that uses f2py"""

    def __init__(self, *args, **kwargs):
        # 初始化方法，继承父类的初始化方法，传入的参数可能包含一些扩展选项
        ext_keys = ['include_dirs', 'library_dirs', 'libraries',
                    'extra_compile_args', 'extra_link_args']
        # 警告消息模板，用于提示不支持的编译选项
        msg = ('The compilation option kwarg {} is not supported with the f2py '
               'backend.')

        # 检查是否有不支持的编译选项，如果有则发出警告
        for k in ext_keys:
            if k in kwargs.keys():
                warn(msg.format(k))
            # 从 kwargs 中移除这些不支持的选项
            kwargs.pop(k, None)

        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

    @property
    def command(self):
        # 返回编译命令的列表
        filename = self.filename + '.' + self.generator.code_extension
        # 构造编译参数列表
        args = ['-c', '-m', self.module_name, filename]
        # 构造完整的命令，包括 Python 解释器和 f2py 的调用
        command = [sys.executable, "-c", "import numpy.f2py as f2py2e;f2py2e.main()"]+args
        return command

    def _prepare_files(self, routine):
        # 准备文件的方法，目前未实现具体逻辑，留空
        pass

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        # 类方法，用于从模块中获取包装后的函数对象
        return getattr(mod, name)
    # 参数列表，一个有序的可迭代对象，用于指定函数的参数顺序
    args : iterable, optional
        An ordered iterable of symbols. Specifies the argument sequence for the
        function.
    # 标志列表，一个可选的可迭代对象，包含将传递给后端的额外选项标志
    flags : iterable, optional
        Additional option flags that will be passed to the backend.
    # 是否详细输出，如果为 True，autowrap 将不会静音命令行后端，这对调试很有帮助
    verbose : bool, optional
        If True, autowrap will not mute the command line backends. This can be
        helpful for debugging.
    # 辅助函数列表，可以是一个3元组的可迭代对象或单个3元组，用于定义主表达式所需的辅助表达式
    helpers : 3-tuple or iterable of 3-tuples, optional
        Used to define auxiliary expressions needed for the main expr. If the
        main expression needs to call a specialized function it should be
        passed in via ``helpers``. Autowrap will then make sure that the
        compiled main expression can link to the helper routine. Items should
        be 3-tuples with (<function_name>, <sympy_expression>,
        <argument_tuple>). It is mandatory to supply an argument sequence to
        helper routines.
    # 代码生成器实例，必须是 CodeGen 的子类实例，用于生成代码
    code_gen : CodeGen instance
        An instance of a CodeGen subclass. Overrides ``language``.
    # 包含头文件的目录列表，用于搜索 C/C++ 头文件
    include_dirs : [string]
        A list of directories to search for C/C++ header files (in Unix form
        for portability).
    # 链接时搜索库文件的目录列表
    library_dirs : [string]
        A list of directories to search for C/C++ libraries at link time.
    # 链接时需要链接的库列表，不是文件名或路径名
    libraries : [string]
        A list of library names (not filenames or paths) to link against.
    # 编译源文件时使用的额外编译参数列表
    extra_compile_args : [string]
        Any extra platform- and compiler-specific information to use when
        compiling the source files in 'sources'.  For platforms and compilers
        where "command line" makes sense, this is typically a list of
        command-line arguments, but for other platforms it could be anything.
    # 链接对象文件以创建扩展或静态 Python 解释器时使用的额外链接参数列表
    extra_link_args : [string]
        Any extra platform- and compiler-specific information to use when
        linking object files together to create the extension (or to create a
        new static Python interpreter).  Similar interpretation as for
        'extra_compile_args'.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.autowrap import autowrap
    >>> expr = ((x - y + z)**(13)).expand()
    >>> binary_func = autowrap(expr)
    >>> binary_func(1, 4, 2)
    -1.0

    """
    # 如果指定了语言参数，则验证其类型是否符合要求，否则推断语言类型
    if language:
        if not isinstance(language, type):
            _validate_backend_language(backend, language)
    else:
        language = _infer_language(backend)

    # 处理 helpers 参数的两种情况：1) helpers 是一个3元组的可迭代对象；2) helpers 是一个3元组
    if iterable(helpers) and len(helpers) != 0 and iterable(helpers[0]):
        helpers = helpers if helpers else ()
    else:
        helpers = [helpers] if helpers else ()
    
    # 如果 args 是可迭代的（不包含集合），则将其转换为列表形式
    args = list(args) if iterable(args, exclude=set) else args

    # 如果未提供 code_gen 参数，则根据语言类型获取对应的代码生成器实例
    if code_gen is None:
        code_gen = get_code_generator(language, "autowrap")

    # 根据 backend 参数选择相应的 CodeWrapperClass 类型
    CodeWrapperClass = {
        'F2PY': F2PyCodeWrapper,
        'CYTHON': CythonCodeWrapper,
        'DUMMY': DummyWrapper
    }[backend.upper()]
    # 创建一个 CodeWrapperClass 的实例，用于包装生成的代码
    code_wrapper = CodeWrapperClass(code_gen, tempdir, flags if flags else (),
                                    verbose, **kwargs)

    # 初始化一个空列表，用于存储生成的帮助函数
    helps = []
    # 遍历 helpers 列表中的每一个元素，生成对应的代码并添加到 helps 列表中
    for name_h, expr_h, args_h in helpers:
        helps.append(code_gen.routine(name_h, expr_h, args_h))

    # 再次遍历 helpers 列表中的每一个元素，如果表达式 expr 中包含该帮助函数 expr_h，
    # 则将其替换为名为 name_h 的二元函数的调用结果
    for name_h, expr_h, args_h in helpers:
        if expr.has(expr_h):
            name_h = binary_function(name_h, expr_h, backend='dummy')
            expr = expr.subs(expr_h, name_h(*args_h))

    try:
        # 尝试生成一个名为 'autofunc' 的代码例程，基于给定的表达式和参数
        routine = code_gen.routine('autofunc', expr, args)
    except CodeGenArgumentListError as e:
        # 如果捕获到参数列表错误异常，检查其中缺失的参数是否仅为输出参数，
        # 若是，则将它们附加到参数列表的末尾，并重新尝试生成代码例程
        new_args = []
        for missing in e.missing_args:
            if not isinstance(missing, OutputArgument):
                raise
            new_args.append(missing.name)
        routine = code_gen.routine('autofunc', expr, args + new_args)

    # 使用 code_wrapper 对象将生成的代码例程 routine 进行包装，并附加帮助函数列表 helps
    return code_wrapper.wrap_code(routine, helpers=helps)
# 生成一个依赖于 doctest 的装饰器，检查 'f2py' 和 'gfortran' 是否存在，以及 'numpy' 是否已安装
@doctest_depends_on(exe=('f2py', 'gfortran'), modules=('numpy',))
def binary_function(symfunc, expr, **kwargs):
    """Returns a SymPy function with expr as binary implementation

    This is a convenience function that automates the steps needed to
    autowrap the SymPy expression and attaching it to a Function object
    with implemented_function().

    Parameters
    ==========

    symfunc : SymPy Function
        The function to bind the callable to.
    expr : SymPy Expression
        The expression used to generate the function.
    kwargs : dict
        Any kwargs accepted by autowrap.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.utilities.autowrap import binary_function
    >>> expr = ((x - y)**(25)).expand()
    >>> f = binary_function('f', expr)
    >>> type(f)
    <class 'sympy.core.function.UndefinedFunction'>
    >>> 2*f(x, y)
    2*f(x, y)
    >>> f(x, y).evalf(2, subs={x: 1, y: 2})
    -1.0

    """
    # 使用 autowrap 将 SymPy 表达式转换为二进制实现
    binary = autowrap(expr, **kwargs)
    # 使用 implemented_function 将二进制实现绑定到给定的 SymPy 函数上并返回
    return implemented_function(symfunc, binary)

#################################################################
#                           UFUNCIFY                            #
#################################################################

# 定义模板 _ufunc_top，用于生成包含必要头文件和模块方法的 C 代码
_ufunc_top = Template("""\
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include ${include_file}

static PyMethodDef ${module}Methods[] = {
        {NULL, NULL, 0, NULL}
};""")

# 定义模板 _ufunc_outcalls，用于生成将函数结果写入输出数组的 C 代码
_ufunc_outcalls = Template("*((double *)out${outnum}) = ${funcname}(${call_args});")

# 定义模板 _ufunc_body，用于生成 ufunc 的核心函数体，包含循环调用用户定义的函数并更新步长
_ufunc_body = Template("""\
#ifdef NPY_1_19_API_VERSION
static void ${funcname}_ufunc(char **args, const npy_intp *dimensions, const npy_intp* steps, void* data)
#else
static void ${funcname}_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
#endif
{
    npy_intp i;
    npy_intp n = dimensions[0];
    ${declare_args}
    ${declare_steps}
    for (i = 0; i < n; i++) {
        ${outcalls}
        ${step_increments}
    }
}

// 定义 PyUFuncGenericFunction 数组，包含生成的 ufunc 核心函数的指针
PyUFuncGenericFunction ${funcname}_funcs[1] = {&${funcname}_ufunc};

// 定义存储 ufunc 类型的数组 ${funcname}_types
static char ${funcname}_types[${n_types}] = ${types}

// 定义存储 ufunc 数据指针的数组 ${funcname}_data
static void *${funcname}_data[1] = {NULL};
""")

# 定义模板 _ufunc_bottom，生成模块初始化的 C 代码，适配不同 Python 版本
_ufunc_bottom = Template("""\
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "${module}",
    NULL,
    -1,
    ${module}Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_${module}(void)
{
    PyObject *m, *d;
    ${function_creation}
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ${ufunc_init}
    return m;
}
#else
PyMODINIT_FUNC init${module}(void)
{
    PyObject *m, *d;
    ${function_creation}
    m = Py_InitModule("${module}", ${module}Methods);
    if (m == NULL) {
        return;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ${ufunc_init}
}
#endif\
""")
"""
})

_ufunc_init_form = Template("""\
# 创建一个新的 ufunc 对象，并设置其函数指针、数据以及类型信息
ufunc${ind} = PyUFunc_FromFuncAndData(${funcname}_funcs, ${funcname}_data, ${funcname}_types, 1, ${n_in}, ${n_out},
            PyUFunc_None, "${module}", ${docstring}, 0);
# 将创建的 ufunc 对象添加到 Python 字典中，使用函数名作为键
PyDict_SetItemString(d, "${funcname}", ufunc${ind});
# 减少 ufunc 对象的引用计数，避免内存泄漏
Py_DECREF(ufunc${ind});
""")

_ufunc_setup = Template("""\
# 导入需要的模块和函数
from setuptools.extension import Extension
from setuptools import setup
from numpy import get_include

# 如果当前文件被直接执行
if __name__ == "__main__":
    # 配置编译参数并设置扩展模块
    setup(ext_modules=[
        Extension('${module}',
                  sources=['${module}.c', '${filename}.c'],
                  include_dirs=[get_include()])])
""")


class UfuncifyCodeWrapper(CodeWrapper):
    """Wrapper for Ufuncify"""

    def __init__(self, *args, **kwargs):
        # 初始化函数，检查是否包含不支持的编译选项
        ext_keys = ['include_dirs', 'library_dirs', 'libraries',
                    'extra_compile_args', 'extra_link_args']
        msg = ('The compilation option kwarg {} is not supported with the numpy'
               ' backend.')

        for k in ext_keys:
            if k in kwargs.keys():
                # 如果包含不支持的编译选项则发出警告信息
                warn(msg.format(k))
            # 从参数中移除不支持的编译选项
            kwargs.pop(k, None)

        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

    @property
    def command(self):
        # 返回编译命令列表
        command = [sys.executable, "setup.py", "build_ext", "--inplace"]
        return command

    def wrap_code(self, routines, helpers=None):
        # 重写 wrap_code 方法以支持 ufuncify
        # 设置函数名为一个唯一标识符
        funcname = 'wrapped_' + str(id(routines) + id(helpers))

        # 设置工作目录
        workdir = self.filepath or tempfile.mkdtemp("_sympy_compile")
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        oldwork = os.getcwd()
        os.chdir(workdir)
        try:
            sys.path.append(workdir)
            # 生成代码文件
            self._generate_code(routines, helpers)
            # 准备代码文件
            self._prepare_files(routines, funcname)
            # 处理代码文件
            self._process_files(routines)
            # 导入生成的模块
            mod = __import__(self.module_name)
        finally:
            sys.path.remove(workdir)
            # 增加模块计数器
            CodeWrapper._module_counter += 1
            os.chdir(oldwork)
            # 如果未指定文件路径，则删除临时工作目录
            if not self.filepath:
                try:
                    shutil.rmtree(workdir)
                except OSError:
                    # 在 Windows 上可能会出现问题
                    pass

        # 返回包装后的函数对象
        return self._get_wrapped_function(mod, funcname)

    def _generate_code(self, main_routines, helper_routines):
        # 生成代码的具体实现
        all_routines = main_routines + helper_routines
        self.generator.write(
            all_routines, self.filename, True, self.include_header,
            self.include_empty)
    # 准备生成 C 代码文件
    codefilename = self.module_name + '.c'
    with open(codefilename, 'w') as f:
        # 调用 dump_c 方法将函数体生成为 C 代码并写入文件
        self.dump_c(routines, f, self.filename, funcname=funcname)

    # 准备生成 setup.py 文件
    with open('setup.py', 'w') as f:
        # 调用 dump_setup 方法生成 setup.py 文件内容并写入文件
        self.dump_setup(f)


@classmethod
def _get_wrapped_function(cls, mod, name):
    # 返回指定模块中指定名称的函数对象
    return getattr(mod, name)


def dump_setup(self, f):
    # 使用 _ufunc_setup 模板替换参数生成 setup.py 文件内容并写入文件
    setup = _ufunc_setup.substitute(module=self.module_name,
                                    filename=self.filename)
    f.write(setup)


def _partition_args(self, args):
    """将函数参数分组为不同类别。"""
    py_in = []
    py_out = []
    for arg in args:
        if isinstance(arg, OutputArgument):
            # 将输出参数添加到 py_out 列表中
            py_out.append(arg)
        elif isinstance(arg, InOutArgument):
            # 如果是 InOutArgument 类型的参数，抛出异常
            raise ValueError("Ufuncify 不支持 InOutArguments")
        else:
            # 否则将输入参数添加到 py_in 列表中
            py_in.append(arg)
    # 返回输入参数列表和输出参数列表
    return py_in, py_out
# 使用装饰器 @cacheit 和 @doctest_depends_on 对 ufuncify 函数进行装饰，用于缓存和测试依赖管理
@cacheit
@doctest_depends_on(exe=('f2py', 'gfortran', 'gcc'), modules=('numpy',))
# 定义 ufuncify 函数，用于生成支持 numpy 数组广播的二进制函数
def ufuncify(args, expr, language=None, backend='numpy', tempdir=None,
             flags=None, verbose=False, helpers=None, **kwargs):
    """Generates a binary function that supports broadcasting on numpy arrays.

    Parameters
    ==========

    args : iterable
        Either a Symbol or an iterable of symbols. Specifies the argument
        sequence for the function.
    expr
        A SymPy expression that defines the element wise operation.
    language : string, optional
        If supplied, (options: 'C' or 'F95'), specifies the language of the
        generated code. If ``None`` [default], the language is inferred based
        upon the specified backend.
    backend : string, optional
        Backend used to wrap the generated code. Either 'numpy' [default],
        'cython', or 'f2py'.
    tempdir : string, optional
        Path to directory for temporary files. If this argument is supplied,
        the generated code and the wrapper input files are left intact in
        the specified path.
    flags : iterable, optional
        Additional option flags that will be passed to the backend.
    verbose : bool, optional
        If True, autowrap will not mute the command line backends. This can
        be helpful for debugging.
    helpers : iterable, optional
        Used to define auxiliary expressions needed for the main expr. If
        the main expression needs to call a specialized function it should
        be put in the ``helpers`` iterable. Autowrap will then make sure
        that the compiled main expression can link to the helper routine.
        Items should be tuples with (<funtion_name>, <sympy_expression>,
        <arguments>). It is mandatory to supply an argument sequence to
        helper routines.
    kwargs : dict
        These kwargs will be passed to autowrap if the `f2py` or `cython`
        backend is used and ignored if the `numpy` backend is used.

    Notes
    =====

    The default backend ('numpy') will create actual instances of
    ``numpy.ufunc``. These support ndimensional broadcasting, and implicit type
    conversion. Use of the other backends will result in a "ufunc-like"
    function, which requires equal length 1-dimensional arrays for all
    arguments, and will not perform any type conversions.

    References
    ==========

    .. [1] https://numpy.org/doc/stable/reference/ufuncs.html

    Examples
    ========

    >>> from sympy.utilities.autowrap import ufuncify
    >>> from sympy.abc import x, y
    >>> import numpy as np
    >>> f = ufuncify((x, y), y + x**2)
    >>> type(f)
    <class 'numpy.ufunc'>
    >>> f([1, 2, 3], 2)
    array([  3.,   6.,  11.])
    >>> f(np.arange(5), 3)
    array([  3.,   4.,   7.,  12.,  19.])

    For the 'f2py' and 'cython' backends, inputs are required to be equal length
    1-dimensional arrays. The 'f2py' backend will perform type conversion, but
    """
    # 函数主体部分，未提供更多具体代码
    the Cython backend will error if the inputs are not of the expected type.

    >>> f_fortran = ufuncify((x, y), y + x**2, backend='f2py')
    >>> f_fortran(1, 2)
    array([ 3.])
    >>> f_fortran(np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))
    array([  2.,   6.,  12.])
    >>> f_cython = ufuncify((x, y), y + x**2, backend='Cython')
    >>> f_cython(1, 2)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Argument '_x' has incorrect type (expected numpy.ndarray, got int)
    >>> f_cython(np.array([1.0]), np.array([2.0]))
    array([ 3.])

    """

    # 如果args是Symbol类型，则将其转换为单元素元组；否则，保持args不变
    if isinstance(args, Symbol):
        args = (args,)
    else:
        args = tuple(args)

    # 如果指定了language参数，则验证backend和language的匹配性；否则，根据backend推断language
    if language:
        _validate_backend_language(backend, language)
    else:
        language = _infer_language(backend)

    # 如果helpers为None，则设为空元组；否则保持不变
    helpers = helpers if helpers else ()
    # 如果flags为None，则设为空元组；否则保持不变
    flags = flags if flags else ()

    # 如果backend为'NUMPY'时执行以下代码块
    if backend.upper() == 'NUMPY':
        # 设置maxargs为32，这是numpy编译时的常量 NPY_MAXARGS 的值
        maxargs = 32
        helps = []
        # 遍历helpers，生成相关的routine并添加到helps列表中
        for name, expr, args in helpers:
            helps.append(make_routine(name, expr, args))
        # 使用C99CodeGen生成代码包装器，并传入相关参数
        code_wrapper = UfuncifyCodeWrapper(C99CodeGen("ufuncify"), tempdir,
                                           flags, verbose)
        # 如果expr不是列表或元组，则转换为列表形式
        if not isinstance(expr, (list, tuple)):
            expr = [expr]
        # 如果expr长度为0，则抛出ValueError异常
        if len(expr) == 0:
            raise ValueError('Expression iterable has zero length')
        # 如果expr和args的总长度超过maxargs，则抛出ValueError异常
        if len(expr) + len(args) > maxargs:
            msg = ('Cannot create ufunc with more than {0} total arguments: '
                   'got {1} in, {2} out')
            raise ValueError(msg.format(maxargs, len(args), len(expr)))
        # 生成routines列表，包含根据expr生成的routine
        routines = [make_routine('autofunc{}'.format(idx), exprx, args) for
                    idx, exprx in enumerate(expr)]
        # 返回通过code_wrapper包装后的代码
        return code_wrapper.wrap_code(routines, helpers=helps)
    else:
        # 为所有添加的表达式使用Dummy，以防止与原始表达式中的名称冲突
        y = IndexedBase(Dummy('y'))
        m = Dummy('m', integer=True)
        i = Idx(Dummy('i', integer=True), m)
        f_dummy = Dummy('f')
        # 使用implemented_function创建函数f，名称带有唯一标识符
        f = implemented_function('%s_%d' % (f_dummy.name, f_dummy.dummy_index), Lambda(args, expr))
        # 为每个参数args创建一个索引版本
        indexed_args = [IndexedBase(Dummy(str(a))) for a in args]
        # 排序参数列表（输出、args、dim）
        args = [y] + indexed_args + [m]
        # 创建带有索引的参数列表
        args_with_indices = [a[i] for a in indexed_args]
        # 调用autowrap函数生成代码，并返回结果
        return autowrap(Eq(y[i], f(*args_with_indices)), language, backend,
                        tempdir, args, flags, verbose, helpers, **kwargs)
```