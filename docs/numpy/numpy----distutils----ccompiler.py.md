# `.\numpy\numpy\distutils\ccompiler.py`

```
import os
import re
import sys
import platform
import shlex
import time
import subprocess
from copy import copy
from pathlib import Path
from distutils import ccompiler
from distutils.ccompiler import (
    compiler_class, gen_lib_options, get_default_compiler, new_compiler,
    CCompiler
)
from distutils.errors import (
    DistutilsExecError, DistutilsModuleError, DistutilsPlatformError,
    CompileError, UnknownFileError
)
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion

from numpy.distutils import log
from numpy.distutils.exec_command import (
    filepath_from_subprocess_output, forward_bytes_to_stdout
)
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
                                      get_num_build_jobs, \
                                      _commandline_dep_string, \
                                      sanitize_cxx_flags

# globals for parallel build management
import threading

_job_semaphore = None
_global_lock = threading.Lock()
_processing_files = set()


def _needs_build(obj, cc_args, extra_postargs, pp_opts):
    """
    Check if an objects needs to be rebuild based on its dependencies

    Parameters
    ----------
    obj : str
        object file

    Returns
    -------
    bool
        True if the object needs to be rebuilt, False otherwise
    """
    # defined in unixcompiler.py
    dep_file = obj + '.d'
    # 如果依赖文件不存在，则需要重新构建
    if not os.path.exists(dep_file):
        return True

    # dep_file 是一个包含 'object: dependencies' 格式的 makefile
    # 最后一行包含编译器命令行参数，因为某些项目可能会使用不同参数多次编译扩展
    with open(dep_file) as f:
        lines = f.readlines()

    # 生成当前编译器命令行的字符串表示
    cmdline = _commandline_dep_string(cc_args, extra_postargs, pp_opts)
    last_cmdline = lines[-1]
    # 如果最后一行命令行与当前命令行不同，则需要重新构建
    if last_cmdline != cmdline:
        return True

    contents = ''.join(lines[:-1])
    # 解析出依赖文件中的依赖项
    deps = [x for x in shlex.split(contents, posix=True)
            if x != "\n" and not x.endswith(":")]

    try:
        t_obj = os.stat(obj).st_mtime

        # 检查是否有任何依赖文件比对象文件更新
        # 依赖项包括用于创建对象的源文件
        for f in deps:
            if os.stat(f).st_mtime > t_obj:
                return True
    except OSError:
        # 如果发生 OSError，则认为需要重新构建（理论上不应发生）
        return True

    # 如果以上条件均不满足，则不需要重新构建
    return False


def replace_method(klass, method_name, func):
    """
    Replace a method in a class dynamically.

    Parameters
    ----------
    klass : class
        The class in which the method will be replaced.
    method_name : str
        The name of the method to be replaced.
    func : function
        The replacement function.

    Notes
    -----
    This function dynamically replaces a method in a class with
    another function.
    """
    # Py3k 不再具有未绑定方法，MethodType 不起作用，因此使用 lambda 表达式来替换方法
    m = lambda self, *args, **kw: func(self, *args, **kw)
    setattr(klass, method_name, m)


######################################################################
## Method that subclasses may redefine. But don't call this method,
## it is private to CCompiler class and may return unexpected
## results if used elsewhere. So, you have been warned..

def CCompiler_find_executables(self):
    """
    Placeholder method intended for redefinition by subclasses.

    Notes
    -----
    This method is intended to be redefined by subclasses of CCompiler.
    Calling this method directly outside its intended context (subclasses
    of CCompiler) may yield unexpected results.
    """
    """
    这是一个空函数，它本身不执行任何操作，但可以被`get_version`方法调用，并且可以被子类重写。
    特别是在`FCompiler`类中重新定义了这个方法，并且那里可以找到更多的文档说明。
    """
    pass
# 替换 CCompiler 类的 find_executables 方法为 CCompiler_find_executables 方法
replace_method(CCompiler, 'find_executables', CCompiler_find_executables)

# 使用定制的 CCompiler.spawn 方法执行命令的子进程操作。
def CCompiler_spawn(self, cmd, display=None, env=None):
    """
    在子进程中执行命令。

    Parameters
    ----------
    cmd : str
        要执行的命令。
    display : str 或者 str 序列，可选
        要添加到 `numpy.distutils` 日志文件的文本。如果未提供，则 `display` 等于 `cmd`。
    env : 字典类型的环境变量，可选

    Returns
    -------
    None

    Raises
    ------
    DistutilsExecError
        如果命令失败，即退出状态不为 0。
    """
    # 如果环境变量未提供，则使用当前操作系统的环境变量
    env = env if env is not None else dict(os.environ)
    # 如果没有指定显示文本，则使用命令本身作为显示文本
    if display is None:
        display = cmd
        # 如果显示文本是序列，则将其连接为字符串
        if is_sequence(display):
            display = ' '.join(list(display))
    # 记录信息到日志
    log.info(display)
    try:
        # 如果设置了详细输出，直接执行命令
        if self.verbose:
            subprocess.check_output(cmd, env=env)
        else:
            # 否则，将错误输出重定向到标准输出
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as exc:
        # 如果命令执行失败，捕获异常
        o = exc.output
        s = exc.returncode
    except OSError as e:
        # 如果出现 OSError，处理异常情况
        o = f"\n\n{e}\n\n\n"
        try:
            # 尝试根据当前系统的编码方式编码输出
            o = o.encode(sys.stdout.encoding)
        except AttributeError:
            o = o.encode('utf8')
        # 设置返回状态为 127
        s = 127
    else:
        # 如果没有异常，则返回 None
        return None

    # 如果命令是序列，则将其连接为字符串
    if is_sequence(cmd):
        cmd = ' '.join(list(cmd))

    # 如果设置了详细输出，将输出传递到标准输出
    if self.verbose:
        forward_bytes_to_stdout(o)

    # 如果输出包含 'Too many open files'，建议重新运行设置命令直至成功
    if re.search(b'Too many open files', o):
        msg = '\nTry rerunning setup command until build succeeds.'
    else:
        msg = ''
    # 抛出 DistutilsExecError 异常，指明命令执行失败及其退出状态
    raise DistutilsExecError('Command "%s" failed with exit status %d%s' %
                            (cmd, s, msg))

# 替换 CCompiler 类的 spawn 方法为 CCompiler_spawn 方法
replace_method(CCompiler, 'spawn', CCompiler_spawn)

# 定义 CCompiler 类的 object_filenames 方法
def CCompiler_object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
    """
    返回给定源文件的对象文件名。

    Parameters
    ----------
    source_filenames : str 列表
        源文件路径的列表。路径可以是相对路径或绝对路径，这会透明处理。
    strip_dir : bool, 可选
        是否从返回的路径中剥离目录。如果为 True，则返回文件名加上 `output_dir`。默认为 False。
    output_dir : str, 可选
        输出目录的路径。

    """
    # 如果未提供输出目录，则设为空字符串
    if output_dir is None:
        output_dir = ''
    # 初始化空列表用于存储目标文件路径
    obj_names = []
    # 遍历每个源文件名
    for src_name in source_filenames:
        # 分离文件名和扩展名，并规范化路径
        base, ext = os.path.splitext(os.path.normpath(src_name))
        # 如果路径包含驱动器信息，去除驱动器部分
        base = os.path.splitdrive(base)[1] # Chop off the drive
        # 如果路径是绝对路径，则去除开头的 '/'
        base = base[os.path.isabs(base):]  # If abs, chop off leading /
        # 处理以 '..' 开头的相对路径部分
        if base.startswith('..'):
            # 解析开头的相对路径部分，os.path.normpath 已经处理了中间的部分
            i = base.rfind('..')+2
            d = base[:i]
            # 获取绝对路径的基本名称
            d = os.path.basename(os.path.abspath(d))
            base = d + base[i:]
        # 如果文件扩展名不在支持的源文件扩展名列表中，抛出异常
        if ext not in self.src_extensions:
            raise UnknownFileError("unknown file type '%s' (from '%s')" % (ext, src_name))
        # 如果 strip_dir 为 True，则只保留基本文件名部分
        if strip_dir:
            base = os.path.basename(base)
        # 构建目标文件的完整路径，结合输出目录和目标文件的扩展名
        obj_name = os.path.join(output_dir, base + self.obj_extension)
        # 将目标文件路径添加到列表中
        obj_names.append(obj_name)
    # 返回所有目标文件路径的列表
    return obj_names
# 替换 CCompiler 类的 'object_filenames' 方法为自定义的 'CCompiler_object_filenames'
replace_method(CCompiler, 'object_filenames', CCompiler_object_filenames)

# 定义 CCompiler_compile 方法，用于编译一个或多个源文件
def CCompiler_compile(self, sources, output_dir=None, macros=None,
                      include_dirs=None, debug=0, extra_preargs=None,
                      extra_postargs=None, depends=None):
    """
    Compile one or more source files.

    Please refer to the Python distutils API reference for more details.

    Parameters
    ----------
    sources : list of str
        A list of filenames
    output_dir : str, optional
        Path to the output directory.
    macros : list of tuples
        A list of macro definitions.
    include_dirs : list of str, optional
        The directories to add to the default include file search path for
        this compilation only.
    debug : bool, optional
        Whether or not to output debug symbols in or alongside the object
        file(s).
    extra_preargs, extra_postargs : ?
        Extra pre- and post-arguments.
    depends : list of str, optional
        A list of file names that all targets depend on.

    Returns
    -------
    objects : list of str
        A list of object file names, one per source file `sources`.

    Raises
    ------
    CompileError
        If compilation fails.

    """
    
    # 获取并设置并行编译任务的数量
    global _job_semaphore
    jobs = get_num_build_jobs()
    
    # 设置信号量以限制并行编译任务的数量（适用于 Python >= 3.5 的扩展级并行编译）
    with _global_lock:
        if _job_semaphore is None:
            _job_semaphore = threading.Semaphore(jobs)

    # 如果没有源文件，则返回空列表
    if not sources:
        return []
    
    # 导入所需的编译器类和模块
    from numpy.distutils.fcompiler import (FCompiler,
                                           FORTRAN_COMMON_FIXED_EXTENSIONS,
                                           has_f90_header)
    
    # 如果 self 是 FCompiler 类的实例
    if isinstance(self, FCompiler):
        # 显示 Fortran 编译器的相关信息
        display = []
        for fc in ['f77', 'f90', 'fix']:
            fcomp = getattr(self, 'compiler_'+fc)
            if fcomp is None:
                continue
            display.append("Fortran %s compiler: %s" % (fc, ' '.join(fcomp)))
        display = '\n'.join(display)
    else:
        # 显示 C 编译器的相关信息
        ccomp = self.compiler_so
        display = "C compiler: %s\n" % (' '.join(ccomp),)
    
    # 记录编译器相关信息到日志中
    log.info(display)
    
    # 设置编译环境并获取编译所需的宏定义、对象文件名、额外的后处理参数、预处理选项和构建对象
    macros, objects, extra_postargs, pp_opts, build = \
            self._setup_compile(output_dir, macros, include_dirs, sources,
                                depends, extra_postargs)
    
    # 获取 C 编译器的命令行参数
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    
    # 显示编译选项到日志中
    display = "compile options: '%s'" % (' '.join(cc_args))
    if extra_postargs:
        display += "\nextra options: '%s'" % (' '.join(extra_postargs))
    
    # 记录编译选项到日志中
    log.info(display)
    def single_compile(args):
        obj, (src, ext) = args
        if not _needs_build(obj, cc_args, extra_postargs, pp_opts):
            return

        # 检查是否需要构建该目标文件
        while True:
            # 需要使用显式锁，因为 GIL 无法进行原子性的检查和修改操作
            with _global_lock:
                # 如果目标文件当前没有在处理中，则开始处理
                if obj not in _processing_files:
                    _processing_files.add(obj)
                    break
            # 等待处理结束
            time.sleep(0.1)

        try:
            # 从作业信号量中获取插槽并进行编译
            with _job_semaphore:
                self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        finally:
            # 注册处理完成
            with _global_lock:
                _processing_files.remove(obj)


    if isinstance(self, FCompiler):
        objects_to_build = list(build.keys())
        f77_objects, other_objects = [], []
        for obj in objects:
            if obj in objects_to_build:
                src, ext = build[obj]
                if self.compiler_type=='absoft':
                    obj = cyg2win32(obj)
                    src = cyg2win32(src)
                if Path(src).suffix.lower() in FORTRAN_COMMON_FIXED_EXTENSIONS \
                   and not has_f90_header(src):
                    # 将需要用 Fortran 77 编译的对象添加到列表中
                    f77_objects.append((obj, (src, ext)))
                else:
                    # 将需要用其他编译器编译的对象添加到列表中
                    other_objects.append((obj, (src, ext)))

        # Fortran 77 对象可以并行构建
        build_items = f77_objects
        # 串行构建 Fortran 90 模块，模块文件在编译期间生成，并可能被列表中后续文件使用，因此顺序很重要
        for o in other_objects:
            single_compile(o)
    else:
        build_items = build.items()

    if len(build) > 1 and jobs > 1:
        # 并行构建
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(jobs) as pool:
            res = pool.map(single_compile, build_items)
        list(res)  # 访问结果以引发错误
    else:
        # 串行构建
        for o in build_items:
            single_compile(o)

    # 返回所有目标文件名，而不仅仅是刚刚构建的那些
    return objects
# 替换 CCompiler 类的 'compile' 方法为自定义的 CCompiler_compile 方法
replace_method(CCompiler, 'compile', CCompiler_compile)

def CCompiler_customize_cmd(self, cmd, ignore=()):
    """
    自定义编译器使用 distutils 命令。

    Parameters
    ----------
    cmd : class instance
        继承自 ``distutils.cmd.Command`` 的实例。
    ignore : sequence of str, optional
        不应更改的 ``distutils.ccompiler.CCompiler`` 命令（不包括 ``'set_'``）的列表。检查的字符串有：
        ``('include_dirs', 'define', 'undef', 'libraries', 'library_dirs',
        'rpath', 'link_objects')``。

    Returns
    -------
    None

    """
    # 记录日志，显示正在使用哪个类定制编译器
    log.info('customize %s using %s' % (self.__class__.__name__,
                                        cmd.__class__.__name__))

    # 如果 self 中有 'compiler' 属性并且第一个元素是 'clang'，且不是在 arm64 架构的 macOS 上
    if (
        hasattr(self, 'compiler') and
        'clang' in self.compiler[0] and
        not (platform.machine() == 'arm64' and sys.platform == 'darwin')
    ):
        # clang 默认使用非严格的浮点错误点模型。
        # 但是，macosx_arm64 目前不支持 '-ftrapping-math'（2023-04-08）。
        # 因为 NumPy 和大多数 Python 库会对此发出警告，所以进行覆盖：
        self.compiler.append('-ftrapping-math')
        self.compiler_so.append('-ftrapping-math')

    # 定义一个函数 allow，用于检查命令是否允许对应的操作
    def allow(attr):
        return getattr(cmd, attr, None) is not None and attr not in ignore

    # 根据命令是否允许特定操作，设置相应的 include_dirs、define、undef、libraries、library_dirs、rpath 和 link_objects
    if allow('include_dirs'):
        self.set_include_dirs(cmd.include_dirs)
    if allow('define'):
        for (name, value) in cmd.define:
            self.define_macro(name, value)
    if allow('undef'):
        for macro in cmd.undef:
            self.undefine_macro(macro)
    if allow('libraries'):
        self.set_libraries(self.libraries + cmd.libraries)
    if allow('library_dirs'):
        self.set_library_dirs(self.library_dirs + cmd.library_dirs)
    if allow('rpath'):
        self.set_runtime_library_dirs(cmd.rpath)
    if allow('link_objects'):
        self.set_link_objects(cmd.link_objects)

# 替换 CCompiler 类的 'customize_cmd' 方法为自定义的 CCompiler_customize_cmd 方法
replace_method(CCompiler, 'customize_cmd', CCompiler_customize_cmd)

def _compiler_to_string(compiler):
    props = []
    mx = 0
    keys = list(compiler.executables.keys())
    # 检查需要的键是否存在，如果不存在则添加到 keys 列表中
    for key in ['version', 'libraries', 'library_dirs',
                'object_switch', 'compile_switch',
                'include_dirs', 'define', 'undef', 'rpath', 'link_objects']:
        if key not in keys:
            keys.append(key)
    # 遍历 keys 中的每个键，获取编译器对象的属性值，并格式化为字符串
    for key in keys:
        if hasattr(compiler, key):
            v = getattr(compiler, key)
            mx = max(mx, len(key))
            props.append((key, repr(v)))
    # 格式化输出属性，并使用换行符连接成一个字符串
    fmt = '%-' + repr(mx+1) + 's = %s'
    lines = [fmt % prop for prop in props]
    return '\n'.join(lines)

def CCompiler_show_customization(self):
    """
    打印编译器的定制信息到标准输出。

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    仅在 distutils 日志阈值 < 2 时执行打印操作。

    """
    try:
        # 获取编译器版本信息
        self.get_version()
    # 捕获并忽略任何异常，不对异常进行处理
    except Exception:
        pass
    # 如果全局日志对象 log 的阈值小于 2
    if log._global_log.threshold < 2:
        # 打印 80 个 '*' 字符，用于分隔线
        print('*'*80)
        # 打印当前对象的类名
        print(self.__class__)
        # 打印通过 _compiler_to_string 函数编译后的对象表示
        print(_compiler_to_string(self))
        # 打印 80 个 '*' 字符，用于分隔线
        print('*'*80)
# 替换 CCompiler 类的 show_customization 方法为 CCompiler_show_customization 函数
replace_method(CCompiler, 'show_customization', CCompiler_show_customization)

# 自定义编译器实例的平台特定定制
def CCompiler_customize(self, dist, need_cxx=0):
    """
    Do any platform-specific customization of a compiler instance.

    This method calls ``distutils.sysconfig.customize_compiler`` for
    platform-specific customization, as well as optionally remove a flag
    to suppress spurious warnings in case C++ code is being compiled.

    Parameters
    ----------
    dist : object
        This parameter is not used for anything.
    need_cxx : bool, optional
        Whether or not C++ has to be compiled. If True, the
        ``"-Wstrict-prototypes"`` option is removed to prevent spurious
        warnings. Default is False.

    Returns
    -------
    None

    Notes
    -----
    All the default options used by distutils can be extracted with::

      from distutils import sysconfig
      sysconfig.get_config_vars('CC', 'CXX', 'OPT', 'BASECFLAGS',
                                'CCSHARED', 'LDSHARED', 'SO')

    """
    # 输出日志信息，标明正在定制的编译器类名
    log.info('customize %s' % (self.__class__.__name__))
    # 调用外部定义的 customize_compiler 函数，对编译器实例进行定制
    customize_compiler(self)
    
    # 如果需要编译 C++ 代码
    if need_cxx:
        # 一般情况下，distutils 使用 -Wstrict-prototypes 选项，但该选项只对 C 代码有效，对 C++ 代码无效
        # 如果存在该选项，则移除，以避免每次编译时出现误报警告
        try:
            self.compiler_so.remove('-Wstrict-prototypes')
        except (AttributeError, ValueError):
            pass
        
        # 如果编译器存在，并且使用的是 cc 编译器
        if hasattr(self, 'compiler') and 'cc' in self.compiler[0]:
            # 如果编译器没有设置 compiler_cxx 属性，并且编译器以 'gcc' 开头，则替换为 'g++'
            if not self.compiler_cxx:
                if self.compiler[0].startswith('gcc'):
                    a, b = 'gcc', 'g++'
                else:
                    a, b = 'cc', 'c++'
                # 设置 compiler_cxx 属性为经过修正的编译器名称及其参数
                self.compiler_cxx = [self.compiler[0].replace(a, b)]\
                                    + self.compiler[1:]
        else:
            # 如果编译器存在，但没有设置 compiler_cxx 属性，则记录警告信息
            if hasattr(self, 'compiler'):
                log.warn("#### %s #######" % (self.compiler,))
            if not hasattr(self, 'compiler_cxx'):
                log.warn('Missing compiler_cxx fix for ' + self.__class__.__name__)

    # 检查编译器是否支持类似 gcc 风格的自动依赖性
    # 在每个扩展上运行，因此对于已知的好编译器跳过
    if hasattr(self, 'compiler') and ('gcc' in self.compiler[0] or
                                      'g++' in self.compiler[0] or
                                      'clang' in self.compiler[0]):
        # 设置 _auto_depends 属性为 True，表示支持自动依赖性
        self._auto_depends = True
    elif os.name == 'posix':
        # 如果操作系统为 POSIX 类型（Unix/Linux），执行以下操作
        import tempfile  # 导入临时文件模块
        import shutil    # 导入文件操作模块

        # 创建临时目录
        tmpdir = tempfile.mkdtemp()

        try:
            # 在临时目录下创建一个名为 "file.c" 的文件，并写入内容 "int a;\n"
            fn = os.path.join(tmpdir, "file.c")
            with open(fn, "w") as f:
                f.write("int a;\n")

            # 调用 self.compile 方法，编译文件 fn，并指定输出目录为 tmpdir，
            # 额外的编译参数为 ['-MMD', '-MF', fn + '.d']
            self.compile([fn], output_dir=tmpdir,
                         extra_preargs=['-MMD', '-MF', fn + '.d'])

            # 设置标志以表示自动依赖项处理已启用
            self._auto_depends = True

        except CompileError:
            # 如果编译过程中出现 CompileError 异常，则表示自动依赖项处理未能启用
            self._auto_depends = False

        finally:
            # 无论是否发生异常，都要删除临时目录及其内容
            shutil.rmtree(tmpdir)

    return
# 使用动态方法替换 CCompiler 类的 'customize' 方法为 CCompiler_customize 函数
replace_method(CCompiler, 'customize', CCompiler_customize)

# 定义一个简单的版本号匹配函数，用于 CCompiler 和 FCompiler
def simple_version_match(pat=r'[-.\d]+', ignore='', start=''):
    """
    Simple matching of version numbers, for use in CCompiler and FCompiler.

    Parameters
    ----------
    pat : str, optional
        A regular expression matching version numbers.
        Default is ``r'[-.\d]+'``.
    ignore : str, optional
        A regular expression matching patterns to skip.
        Default is ``''``, in which case nothing is skipped.
    start : str, optional
        A regular expression matching the start of where to start looking
        for version numbers.
        Default is ``''``, in which case searching is started at the
        beginning of the version string given to `matcher`.

    Returns
    -------
    matcher : callable
        A function that is appropriate to use as the ``.version_match``
        attribute of a ``distutils.ccompiler.CCompiler`` class. `matcher` takes a single parameter,
        a version string.

    """
    def matcher(self, version_string):
        # version string may appear in the second line, so getting rid
        # of new lines:
        version_string = version_string.replace('\n', ' ')
        pos = 0
        if start:
            m = re.match(start, version_string)
            if not m:
                return None
            pos = m.end()
        while True:
            m = re.search(pat, version_string[pos:])
            if not m:
                return None
            if ignore and re.match(ignore, m.group(0)):
                pos = m.end()
                continue
            break
        return m.group(0)
    return matcher

# 定义 CCompiler 类的 get_version 方法
def CCompiler_get_version(self, force=False, ok_status=[0]):
    """
    Return compiler version, or None if compiler is not available.

    Parameters
    ----------
    force : bool, optional
        If True, force a new determination of the version, even if the
        compiler already has a version attribute. Default is False.
    ok_status : list of int, optional
        The list of status values returned by the version look-up process
        for which a version string is returned. If the status value is not
        in `ok_status`, None is returned. Default is ``[0]``.

    Returns
    -------
    version : str or None
        Version string, in the format of ``distutils.version.LooseVersion``.

    """
    if not force and hasattr(self, 'version'):
        return self.version
    # 查找可执行文件
    self.find_executables()
    try:
        version_cmd = self.version_cmd
    except AttributeError:
        return None
    if not version_cmd or not version_cmd[0]:
        return None
    try:
        # 使用之前定义的版本号匹配器
        matcher = self.version_match
    # 捕获 AttributeError 异常，如果捕获到则执行以下代码块
    except AttributeError:
        try:
            # 尝试获取 self.version_pattern 属性
            pat = self.version_pattern
        except AttributeError:
            # 如果未定义 self.version_pattern 属性，则返回 None
            return None
        
        # 定义 matcher 函数，用于匹配版本字符串
        def matcher(version_string):
            # 使用正则表达式匹配版本字符串
            m = re.match(pat, version_string)
            if not m:
                return None
            # 获取匹配的版本号部分
            version = m.group('version')
            return version

    try:
        # 执行外部命令并获取输出，标准错误输出被合并到标准输出中
        output = subprocess.check_output(version_cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        # 如果外部命令抛出异常，则获取异常的输出和返回码
        output = exc.output
        status = exc.returncode
    except OSError:
        # 捕获 OSError 异常，表示操作系统错误
        # 匹配历史返回值，与 exec_command() 捕获的父异常类似
        status = 127
        output = b''
    else:
        # 如果外部命令执行成功，则处理输出，这里先将输出视为文件路径
        output = filepath_from_subprocess_output(output)
        status = 0

    # 初始化版本号为 None
    version = None
    # 如果返回码在允许的状态码列表中
    if status in ok_status:
        # 使用 matcher 函数匹配输出的版本号
        version = matcher(output)
        # 如果成功匹配到版本号，则转换为松散版本号对象
        if version:
            version = LooseVersion(version)
    
    # 将匹配到的版本号赋值给 self.version
    self.version = version
    # 返回版本号
    return version
# 将 CCompiler 类的 get_version 方法替换为 CCompiler_get_version 方法
replace_method(CCompiler, 'get_version', CCompiler_get_version)

# 定义 CCompiler 类的 cxx_compiler 方法，返回 C++ 编译器实例
def CCompiler_cxx_compiler(self):
    """
    Return the C++ compiler.

    Parameters
    ----------
    None

    Returns
    -------
    cxx : class instance
        The C++ compiler, as a ``distutils.ccompiler.CCompiler`` instance.

    """
    # 如果编译器类型是 'msvc', 'intelw', 'intelemw' 中的一种，则返回当前实例
    if self.compiler_type in ('msvc', 'intelw', 'intelemw'):
        return self

    # 复制当前实例，以便对其进行修改
    cxx = copy(self)
    
    # 设置 C++ 编译器和编译选项
    cxx.compiler_cxx = cxx.compiler_cxx
    cxx.compiler_so = [cxx.compiler_cxx[0]] + \
                      sanitize_cxx_flags(cxx.compiler_so[1:])
    
    # 如果运行在 'aix' 或 'os400' 平台上，并且链接器使用的是 ld_so_aix 脚本
    if (sys.platform.startswith(('aix', 'os400')) and
            'ld_so_aix' in cxx.linker_so[0]):
        # AIX 需要包含 Python 提供的 ld_so_aix 脚本
        cxx.linker_so = [cxx.linker_so[0], cxx.compiler_cxx[0]] \
                        + cxx.linker_so[2:]
    
    # 如果运行在 'os400' 平台上，添加特定的编译选项
    if sys.platform.startswith('os400'):
        # 这是为了在 i 7.4 及其以前版本中支持 printf() 中的 PRId64
        cxx.compiler_so.append('-D__STDC_FORMAT_MACROS')
        # 这是 gcc 10.3 的一个 bug，处理 TLS 初始化失败的情况
        cxx.compiler_so.append('-fno-extern-tls-init')
        cxx.linker_so.append('-fno-extern-tls-init')
    else:
        # 否则，将链接器设置为与 C++ 编译器相同
        cxx.linker_so = [cxx.compiler_cxx[0]] + cxx.linker_so[1:]
    
    # 返回修改后的 C++ 编译器实例
    return cxx

# 将 CCompiler 类的 cxx_compiler 方法替换为 CCompiler_cxx_compiler 方法
replace_method(CCompiler, 'cxx_compiler', CCompiler_cxx_compiler)

# 定义支持的编译器类别和相关描述
compiler_class['intel'] = ('intelccompiler', 'IntelCCompiler',
                           "Intel C Compiler for 32-bit applications")
compiler_class['intele'] = ('intelccompiler', 'IntelItaniumCCompiler',
                            "Intel C Itanium Compiler for Itanium-based applications")
compiler_class['intelem'] = ('intelccompiler', 'IntelEM64TCCompiler',
                             "Intel C Compiler for 64-bit applications")
compiler_class['intelw'] = ('intelccompiler', 'IntelCCompilerW',
                            "Intel C Compiler for 32-bit applications on Windows")
compiler_class['intelemw'] = ('intelccompiler', 'IntelEM64TCCompilerW',
                              "Intel C Compiler for 64-bit applications on Windows")
compiler_class['pathcc'] = ('pathccompiler', 'PathScaleCCompiler',
                            "PathScale Compiler for SiCortex-based applications")
compiler_class['arm'] = ('armccompiler', 'ArmCCompiler',
                            "Arm C Compiler")
compiler_class['fujitsu'] = ('fujitsuccompiler', 'FujitsuCCompiler',
                            "Fujitsu C Compiler")

# 将默认编译器设置添加到 ccompiler._default_compilers 元组中
ccompiler._default_compilers += (('linux.*', 'intel'),
                                 ('linux.*', 'intele'),
                                 ('linux.*', 'intelem'),
                                 ('linux.*', 'pathcc'),
                                 ('nt', 'intelw'),
                                 ('nt', 'intelemw'))

# 如果运行在 Windows 平台上
if sys.platform == 'win32':
    # 将键 'mingw32' 映射到一个元组，包含编译器模块名、编译器类名、描述信息
    compiler_class['mingw32'] = ('mingw32ccompiler', 'Mingw32CCompiler',
                                 "Mingw32 port of GNU C Compiler for Win32"
                                 "(for MSC built Python)")
    # 如果当前系统是 mingw32 平台
    if mingw32():
        # 在 Windows 平台上，默认使用 mingw32（gcc）作为编译器，
        # 因为 MSVC 无法构建 blitz 相关的内容。
        log.info('Setting mingw32 as default compiler for nt.')
        # 设置 mingw32 为 nt 平台的默认编译器，并保留原有默认编译器列表
        ccompiler._default_compilers = (('nt', 'mingw32'),) \
                                       + ccompiler._default_compilers
# 将全局变量 _distutils_new_compiler 指向函数 new_compiler
_distutils_new_compiler = new_compiler

# 定义函数 new_compiler，用于创建一个新的编译器对象
def new_compiler (plat=None,
                  compiler=None,
                  verbose=None,
                  dry_run=0,
                  force=0):
    # 如果未提供 verbose 参数，则根据日志级别设置 verbose
    if verbose is None:
        verbose = log.get_threshold() <= log.INFO
    # 如果未提供 plat 参数，则使用当前操作系统的名称
    if plat is None:
        plat = os.name
    try:
        # 如果未指定 compiler，则获取默认的编译器
        if compiler is None:
            compiler = get_default_compiler(plat)
        # 根据 compiler 获取对应的模块名、类名和描述
        (module_name, class_name, long_description) = compiler_class[compiler]
    except KeyError:
        # 如果未知平台或编译器，则抛出异常
        msg = "don't know how to compile C/C++ code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler" % compiler
        raise DistutilsPlatformError(msg)
    
    # 组装模块名
    module_name = "numpy.distutils." + module_name
    try:
        # 尝试导入模块 module_name
        __import__ (module_name)
    except ImportError as e:
        # 如果导入失败，则记录错误信息，并尝试从 distutils 中导入
        msg = str(e)
        log.info('%s in numpy.distutils; trying from distutils',
                 str(msg))
        module_name = module_name[6:]
        try:
            __import__(module_name)
        except ImportError as e:
            # 如果两次导入均失败，则抛出模块加载异常
            msg = str(e)
            raise DistutilsModuleError("can't compile C/C++ code: unable to load module '%s'" % \
                  module_name)
    
    try:
        # 尝试获取导入后的模块和类
        module = sys.modules[module_name]
        klass = vars(module)[class_name]
    except KeyError:
        # 如果未找到对应类，则抛出模块异常
        raise DistutilsModuleError(("can't compile C/C++ code: unable to find class '%s' " +
               "in module '%s'") % (class_name, module_name))
    
    # 使用找到的类创建编译器对象
    compiler = klass(None, dry_run, force)
    compiler.verbose = verbose
    # 记录调试信息
    log.debug('new_compiler returns %s' % (klass))
    return compiler

# 将 ccompiler 模块的 new_compiler 函数指向上面定义的 new_compiler 函数
ccompiler.new_compiler = new_compiler

# 将全局变量 _distutils_gen_lib_options 指向 gen_lib_options 函数
_distutils_gen_lib_options = gen_lib_options

# 定义函数 gen_lib_options，用于生成库选项列表
def gen_lib_options(compiler, library_dirs, runtime_library_dirs, libraries):
    # 使用 _distutils_gen_lib_options 函数生成原始的库选项列表 r
    r = _distutils_gen_lib_options(compiler, library_dirs,
                                   runtime_library_dirs, libraries)
    lib_opts = []
    # 遍历 r 中的每一项，将其扩展成 lib_opts 列表
    for i in r:
        if is_sequence(i):  # 如果 i 是序列，则展开后加入 lib_opts
            lib_opts.extend(list(i))
        else:  # 否则直接添加到 lib_opts
            lib_opts.append(i)
    return lib_opts

# 将 ccompiler 模块的 gen_lib_options 函数指向上面定义的 gen_lib_options 函数
ccompiler.gen_lib_options = gen_lib_options

# 对于一些特定的编译器模块，将其 gen_lib_options 函数指向上面定义的 gen_lib_options 函数
# 这些模块包括 'msvc9', 'msvc', '_msvc', 'bcpp', 'cygwinc', 'emxc', 'unixc'
for _cc in ['msvc9', 'msvc', '_msvc', 'bcpp', 'cygwinc', 'emxc', 'unixc']:
    _m = sys.modules.get('distutils.' + _cc + 'compiler')
    if _m is not None:
        setattr(_m, 'gen_lib_options', gen_lib_options)
```