# `.\numpy\numpy\distutils\command\config.py`

```py
# 导入必要的模块和库
import os
import signal
import subprocess
import sys
import textwrap
import warnings
# 导入 distutils 模块中的一些类和函数
from distutils.command.config import config as old_config
from distutils.command.config import LANG_EXT
from distutils import log
from distutils.file_util import copy_file
from distutils.ccompiler import CompileError, LinkError
import distutils
# 导入 numpy.distutils.exec_command 模块的函数
from numpy.distutils.exec_command import filepath_from_subprocess_output
# 导入 numpy.distutils.mingw32ccompiler 模块的函数
from numpy.distutils.mingw32ccompiler import generate_manifest
from numpy.distutils.command.autodist 模块中的函数
from numpy.distutils.command.autodist import (check_gcc_function_attribute,
                                              check_gcc_function_attribute_with_intrinsics,
                                              check_gcc_variable_attribute,
                                              check_gcc_version_at_least,
                                              check_inline,
                                              check_restrict,
                                              check_compiler_gcc)
# 将 Fortran 77 和 Fortran 90 对应的文件扩展名加入 LANG_EXT 字典
LANG_EXT['f77'] = '.f'
LANG_EXT['f90'] = '.f90'

# 创建 config 类
class config(old_config):
    # 增加了 fcompiler 选项，用于指定 Fortran 编译器类型
    old_config.user_options += [
        ('fcompiler=', None, "specify the Fortran compiler type"),
        ]

    # 初始化 fcompiler 选项
    def initialize_options(self):
        self.fcompiler = None
        old_config.initialize_options(self)
        # 检查编译器
        def _check_compiler (self):
            # 调用父类的_check_compiler方法
            old_config._check_compiler(self)
            # 导入FCompiler类和new_fcompiler函数
            from numpy.distutils.fcompiler import FCompiler, new_fcompiler

            # 如果是在Windows平台下，并且编译器类型是msvc、intelw、intelemw之一
            if sys.platform == 'win32' and (self.compiler.compiler_type in
                                            ('msvc', 'intelw', 'intelemw')):
                # XXX: hack to circumvent a python 2.6 bug with msvc9compiler:
                # 初始化调用query_vcvarsall，避免python 2.6的bug出现OSError，然后在这里捕获它并打印提示信息
                if not self.compiler.initialized:
                    try:
                        self.compiler.initialize()
                    except OSError as e:
                        # 打印错误信息，并通过DistutilsPlatformError抛出异常
                        msg = textwrap.dedent("""\
                            Could not initialize compiler instance: do you have Visual Studio
                            installed?  If you are trying to build with MinGW, please use "python setup.py
                            build -c mingw32" instead.  If you have Visual Studio installed, check it is
                            correctly installed, and the right version (VS 2015 as of this writing).

                            Original exception was: %s, and the Compiler class was %s
                            ============================================================================""") \
                            % (e, self.compiler.__class__.__name__)
                        print(textwrap.dedent("""\
                            ============================================================================"""))
                        raise distutils.errors.DistutilsPlatformError(msg) from e

                # 在MSVC初始化后，添加一个显式的/ MANIFEST到链接器标志
                from distutils import msvc9compiler
                # 如果是MSVC版本大于等于10
                if msvc9compiler.get_build_version() >= 10:
                    # 遍历链接器标志
                    for ldflags in [self.compiler.ldflags_shared,
                                    self.compiler.ldflags_shared_debug]:
                        # 如果链接器标志列表不包含'/ MANIFEST'，则添加
                        if '/MANIFEST' not in ldflags:
                            ldflags.append('/MANIFEST')

            # 如果self.fcompiler不是FCompiler的实例
            if not isinstance(self.fcompiler, FCompiler):
                # 创建新的编译器实例
                self.fcompiler = new_fcompiler(compiler=self.fcompiler,
                                               dry_run=self.dry_run, force=1,
                                               c_compiler=self.compiler)
                # 如果创建成功
                if self.fcompiler is not None:
                    # 定制化编译器
                    self.fcompiler.customize(self.distribution)
                    # 获取编译器版本，然后定制化cmd
                    if self.fcompiler.get_version():
                        self.fcompiler.customize_cmd(self)
                        self.fcompiler.show_customization()
    # 封装方法，用于指定语言和参数调用指定的方法
    def _wrap_method(self, mth, lang, args):
        # 导入异常类
        from distutils.ccompiler import CompileError
        from distutils.errors import DistutilsExecError
        # 保存当前编译器
        save_compiler = self.compiler
        # 如果语言是 'f77' 或 'f90'，则使用相应的编译器
        if lang in ['f77', 'f90']:
            self.compiler = self.fcompiler
        # 如果未设置编译器，则抛出编译错误
        if self.compiler is None:
            raise CompileError('%s compiler is not set' % (lang,))
        try:
            # 调用指定的方法，并处理可能出现的编译或执行错误
            ret = mth(*((self,)+args))
        except (DistutilsExecError, CompileError) as e:
            # 恢复原来的编译器并重新抛出异常
            self.compiler = save_compiler
            raise CompileError from e
        # 恢复原来的编译器
        self.compiler = save_compiler
        # 返回方法调用的结果
        return ret

    # 封装编译方法，处理编译产生的临时文件
    def _compile (self, body, headers, include_dirs, lang):
        # 调用上面封装方法，执行实际的编译操作
        src, obj = self._wrap_method(old_config._compile, lang,
                                     (body, headers, include_dirs, lang))
        # 在 unixcompiler.py 中的 _compile 有时会创建 .d 依赖文件，这里将其清理
        self.temp_files.append(obj + '.d')
        # 返回编译产生的源文件和目标文件
        return src, obj
    # 定义_link方法，接受参数 body, headers, include_dirs, libraries, library_dirs, lang
    def _link (self, body,
               headers, include_dirs,
               libraries, library_dirs, lang):
        # 如果编译器类型为msvc
        if self.compiler.compiler_type=='msvc':
            # 复制libraries列表，确保不修改原始列表
            libraries = (libraries or [:])
            # 复制library_dirs列表，确保不修改原始列表
            library_dirs = (library_dirs or [:])
            # 如果lang为f77或f90，则将其设为c，使用系统链接器
            if lang in ['f77', 'f90']:
                lang = 'c' # 使用MSVC编译器时总是使用系统链接器
                if self.fcompiler:
                    # 对于fcompiler中的每个library_dirs目录
                    for d in self.fcompiler.library_dirs or []:
                        # 当在Cygwin环境编译但使用的是正常的Windows Python时，修正路径
                        if d.startswith('/usr/lib'):
                            try:
                                d = subprocess.check_output(['cygpath', '-w', d])
                            except (OSError, subprocess.CalledProcessError):
                                pass
                            else:
                                d = filepath_from_subprocess_output(d)
                        # 将修正后的目录添加到library_dirs中
                        library_dirs.append(d)
                    # 对于fcompiler中的每个libraries库
                    for libname in self.fcompiler.libraries or []:
                        # 如果该库不在libraries列表中，则将其添加到libraries中
                        if libname not in libraries:
                            libraries.append(libname)
            # 对于libraries列表中的每个libname
            for libname in libraries:
                # 如果libname以msvc开头，则跳过
                if libname.startswith('msvc'): continue
                fileexists = False
                # 对于每个library_dirs目录
                for libdir in library_dirs or []:
                    # 拼接libdir和libname，得到libfile路径
                    libfile = os.path.join(libdir, '%s.lib' % (libname))
                    # 如果libfile文件存在，则设置fileexists为True，并跳出循环
                    if os.path.isfile(libfile):
                        fileexists = True
                        break
                # 如果fileexists为True，则继续循环
                if fileexists: continue
                # 如果编译器为g77，则执行以下操作
                for libdir in library_dirs:
                    libfile = os.path.join(libdir, 'lib%s.a' % (libname))
                    # 如果libfile文件存在，则将其复制为libname.lib，供MSVC链接器使用
                    if os.path.isfile(libfile):
                        libfile2 = os.path.join(libdir, '%s.lib' % (libname))
                        copy_file(libfile, libfile2)
                        self.temp_files.append(libfile2)
                        fileexists = True
                        break
                # 如果fileexists为True，则继续循环
                if fileexists: continue
                # 如果找不到库文件，则记录警告信息
                log.warn('could not find library %r in directories %s' \
                         % (libname, library_dirs))
        # 如果编译器类型为mingw32，则生成manifest
        elif self.compiler.compiler_type == 'mingw32':
            generate_manifest(self)
        # 返回_wrap_method方法的结果
        return self._wrap_method(old_config._link, lang,
                                 (body, headers, include_dirs,
                                  libraries, library_dirs, lang))
    # 检查头文件是否存在，包括指定的目录和库目录，默认语言为C
    def check_header(self, header, include_dirs=None, library_dirs=None, lang='c'):
        # 检查编译器是否准备好
        self._check_compiler()
        # 尝试编译指定的头文件
        return self.try_compile(
                "/* we need a dummy line to make distutils happy */",
                [header], include_dirs)

    # 检查声明是否存在
    def check_decl(self, symbol,
                   headers=None, include_dirs=None):
        # 检查编译器是否准备好
        self._check_compiler()
        # 准备用于试验编译的代码
        body = textwrap.dedent("""
            int main(void)
            {
            #ifndef %s
                (void) %s;
            #endif
                ;
                return 0;
            }""") % (symbol, symbol)

        return self.try_compile(body, headers, include_dirs)

    # 检查宏是否为真
    def check_macro_true(self, symbol,
                         headers=None, include_dirs=None):
        # 检查编译器是否准备好
        self._check_compiler()
        # 准备用于试验编译的代码
        body = textwrap.dedent("""
            int main(void)
            {
            #if %s
            #else
            #error false or undefined macro
            #endif
                ;
                return 0;
            }""") % (symbol,)

        return self.try_compile(body, headers, include_dirs)

    # 检查类型是否可用
    def check_type(self, type_name, headers=None, include_dirs=None,
            library_dirs=None):
        """Check type availability. Return True if the type can be compiled,
        False otherwise"""
        # 检查编译器是否准备好
        self._check_compiler()

        # 首先检查是否可以编译该类型
        body = textwrap.dedent(r"""
            int main(void) {
              if ((%(name)s *) 0)
                return 0;
              if (sizeof (%(name)s))
                return 0;
            }
            """) % {'name': type_name}

        st = False
        try:
            try:
                self._compile(body % {'type': type_name},
                        headers, include_dirs, 'c')
                st = True
            except distutils.errors.CompileError:
                st = False
        finally:
            self._clean()

        return st
    # 检查给定类型的大小
    def check_type_size(self, type_name, headers=None, include_dirs=None, library_dirs=None, expected=None):
        """Check size of a given type."""
        # 检查编译器是否可用
        self._check_compiler()

        # 首先检查类型是否可被编译
        body = textwrap.dedent(r"""
            typedef %(type)s npy_check_sizeof_type;
            int main (void)
            {
                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) >= 0)];
                test_array [0] = 0

                ;
                return 0;
            }
            """)
        self._compile(body % {'type': type_name},
                headers, include_dirs, 'c')
        self._clean()

        # 如果有预期大小，则检查预期大小
        if expected:
            body = textwrap.dedent(r"""
                typedef %(type)s npy_check_sizeof_type;
                int main (void)
                {
                    static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) == %(size)s)];
                    test_array [0] = 0

                    ;
                    return 0;
                }
                """)
            for size in expected:
                try:
                    self._compile(body % {'type': type_name, 'size': size},
                            headers, include_dirs, 'c')
                    self._clean()
                    return size
                except CompileError:
                    pass

        # 如果编译失败，说明大小超过了类型本身
        body = textwrap.dedent(r"""
            typedef %(type)s npy_check_sizeof_type;
            int main (void)
            {
                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) <= %(size)s)];
                test_array [0] = 0

                ;
                return 0;
            }
            """)

        # 原理很简单：我们首先找到类型大小的低值和高值，在对数尺度上进行查找。然后，我们进行二分查找，找到在低值和高值之间的确切大小
        low = 0
        mid = 0
        while True:
            try:
                self._compile(body % {'type': type_name, 'size': mid},
                        headers, include_dirs, 'c')
                self._clean()
                break
            except CompileError:
                low = mid + 1
                mid = 2 * mid + 1

        high = mid
        # 二分查找:
        while low != high:
            mid = (high - low) // 2 + low
            try:
                self._compile(body % {'type': type_name, 'size': mid},
                        headers, include_dirs, 'c')
                self._clean()
                high = mid
            except CompileError:
                low = mid + 1
        return low
    # 检查函数是否能够成功链接
    def check_func(self, func,
                   headers=None, include_dirs=None,
                   libraries=None, library_dirs=None,
                   decl=False, call=False, call_args=None):
        # 清理distutils的配置, 添加void给main()函数, 并返回一个值。
        self._check_compiler()
        # 初始化函数体
        body = []
        # 如果声明为真
        if decl:
            # 如果声明是字符串，则添加到函数体中
            if type(decl) == str:
                body.append(decl)
            # 否则添加一个int类型的声明到函数体中
            else:
                body.append("int %s (void);" % func)
        # 处理MSVC内置函数：强制MS编译器调用一个函数。用于在启用优化的情况下测试一些函数，避免因内置函数和我们的“伪”测试声明不匹配而导致构建错误。
        body.append("#ifdef _MSC_VER")
        body.append("#pragma function(%s)" % func)  # 添加MSVC内置函数的处理
        body.append("#endif")
        body.append("int main (void) {")  # 声明主函数
        # 如果调用为真
        if call:
            # 如果调用参数为空，则将空字符串赋给调用参数
            if call_args is None:
                call_args = ''
            body.append("  %s(%s);" % (func, call_args))
        else:
            body.append("  %s;" % func)
        body.append("  return 0;")
        body.append("}")  # 关闭主函数
        body = '\n'.join(body) + "\n"  # 将函数体按照换行符连接成字符串

        # 尝试链接函数体
        return self.try_link(body, headers, include_dirs,
                             libraries, library_dirs)
    def check_funcs_once(self, funcs,
                   headers=None, include_dirs=None,
                   libraries=None, library_dirs=None,
                   decl=False, call=False, call_args=None):
        """Check a list of functions at once.

        This is useful to speed up things, since all the functions in the funcs
        list will be put in one compilation unit.

        Arguments
        ---------
        funcs : seq
            list of functions to test
        include_dirs : seq
            list of header paths
        libraries : seq
            list of libraries to link the code snippet to
        library_dirs : seq
            list of library paths
        decl : dict
            for every (key, value), the declaration in the value will be
            used for function in key. If a function is not in the
            dictionary, no declaration will be used.
        call : dict
            for every item (f, value), if the value is True, a call will be
            done to the function f.
        """
        self._check_compiler()
        body = []
        if decl:
            for f, v in decl.items():
                if v:
                    body.append("int %s (void);" % f)

        # Handle MS intrinsics. See check_func for more info.
        body.append("#ifdef _MSC_VER")
        for func in funcs:
            body.append("#pragma function(%s)" % func)
        body.append("#endif")

        body.append("int main (void) {")
        if call:
            for f in funcs:
                if f in call and call[f]:
                    if not (call_args and f in call_args and call_args[f]):
                        args = ''
                    else:
                        args = call_args[f]
                    body.append("  %s(%s);" % (f, args))
                else:
                    body.append("  %s;" % f)
        else:
            for f in funcs:
                body.append("  %s;" % f)
        body.append("  return 0;")
        body.append("}")
        body = '\n'.join(body) + "\n"

        return self.try_link(body, headers, include_dirs,
                             libraries, library_dirs)

    def check_inline(self):
        """Return the inline keyword recognized by the compiler, empty string
        otherwise."""
        return check_inline(self)

    def check_restrict(self):
        """Return the restrict keyword recognized by the compiler, empty string
        otherwise."""
        return check_restrict(self)

    def check_compiler_gcc(self):
        """Return True if the C compiler is gcc"""
        return check_compiler_gcc(self)

    def check_gcc_function_attribute(self, attribute, name):
        return check_gcc_function_attribute(self, attribute, name)
    # 检查 GCC 函数属性并使用内置函数
    def check_gcc_function_attribute_with_intrinsics(self, attribute, name, code, include):
        # 调用函数，检查 GCC 函数属性并使用内置函数
        return check_gcc_function_attribute_with_intrinsics(self, attribute, name, code, include)
    
    # 检查 GCC 变量属性
    def check_gcc_variable_attribute(self, attribute):
        # 调用函数，检查 GCC 变量属性
        return check_gcc_variable_attribute(self, attribute)
    
    # 检查 GCC 的版本是否大于等于指定版本
    def check_gcc_version_at_least(self, major, minor=0, patchlevel=0):
        """Return True if the GCC version is greater than or equal to the specified version."""
        # 调用函数，检查 GCC 版本是否大于等于指定版本
        return check_gcc_version_at_least(self, major, minor, patchlevel)
    # 定义一个方法用于编译、链接和运行由'body'和'headers'创建的程序，返回程序的退出状态码和输出结果
    def get_output(self, body, headers=None, include_dirs=None,
                   libraries=None, library_dirs=None,
                   lang="c", use_tee=None):
        """Try to compile, link to an executable, and run a program
        built from 'body' and 'headers'. Returns the exit status code
        of the program and its output.
        """
        # 2008-11-16, RemoveMe
        # 发出警告，提示不要再使用get_output方法，因为它已经被弃用
        warnings.warn("\n+++++++++++++++++++++++++++++++++++++++++++++++++\n"
                      "Usage of get_output is deprecated: please do not \n"
                      "use it anymore, and avoid configuration checks \n"
                      "involving running executable on the target machine.\n"
                      "+++++++++++++++++++++++++++++++++++++++++++++++++\n",
                      DeprecationWarning, stacklevel=2)
        # 检查编译器是否可用
        self._check_compiler()
        # 初始化退出状态码和输出
        exitcode, output = 255, ''
        try:
            # 创建一个GrabStdout对象
            grabber = GrabStdout()
            try:
                # 进行链接操作，获取源码文件路径(src)、目标文件路径(obj)、可执行文件路径(exe)
                src, obj, exe = self._link(body, headers, include_dirs,
                                           libraries, library_dirs, lang)
                # 恢复标准输出
                grabber.restore()
            except Exception:
                output = grabber.data
                grabber.restore()
                raise
            # 将exe路径设定为当前目录
            exe = os.path.join('.', exe)
            try:
                # 使用subprocess模块的check_output方法运行可执行文件，获取输出结果
                output = subprocess.check_output([exe], cwd='.')
            except subprocess.CalledProcessError as exc:
                exitstatus = exc.returncode
                output = ''
            except OSError:
                # 保留EnvironmentError退出状态，这在历史上exec_command()方法中被使用
                exitstatus = 127
                output = ''
            else:
                # 对输出结果进行处理
                output = filepath_from_subprocess_output(output)
            if hasattr(os, 'WEXITSTATUS'):
                # 如果包含WEXITSTATUS属性，则使用其值进行处理
                exitcode = os.WEXITSTATUS(exitstatus)
                if os.WIFSIGNALED(exitstatus):
                    sig = os.WTERMSIG(exitstatus)
                    log.error('subprocess exited with signal %d' % (sig,))
                    if sig == signal.SIGINT:
                        # 控制C
                        raise KeyboardInterrupt
            else:
                exitcode = exitstatus
            log.info("success!")
        except (CompileError, LinkError):
            log.info("failure.")
        # 清理资源
        self._clean()
        # 返回退出状态码和输出结果
        return exitcode, output
# 定义一个名为 GrabStdout 的类，用于捕获标准输出流
class GrabStdout:

    # 初始化方法，设置初始状态
    def __init__(self):
        # 保存原始的 sys.stdout 对象，以便在恢复时使用
        self.sys_stdout = sys.stdout
        # 初始化一个空字符串来存储捕获到的输出数据
        self.data = ''
        # 将当前实例自身设置为 sys.stdout，以便捕获输出
        sys.stdout = self

    # 重写 write 方法，用于捕获输出并保存到 self.data 中
    def write(self, data):
        # 调用原始的 sys.stdout.write 方法输出到控制台
        self.sys_stdout.write(data)
        # 将捕获到的输出数据追加到 self.data 中
        self.data += data

    # 定义 flush 方法，用于刷新输出缓冲区
    def flush(self):
        # 调用原始的 sys.stdout.flush 方法刷新输出缓冲区
        self.sys_stdout.flush()

    # 定义 restore 方法，用于恢复原始的 sys.stdout 对象
    def restore(self):
        # 将 sys.stdout 恢复为初始保存的 self.sys_stdout
        sys.stdout = self.sys_stdout
```