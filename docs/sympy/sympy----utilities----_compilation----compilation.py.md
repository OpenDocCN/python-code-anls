# `D:\src\scipysrc\sympy\sympy\utilities\_compilation\compilation.py`

```
import glob  # 导入 glob 模块，用于文件路径的模式匹配
import os  # 导入 os 模块，提供与操作系统相关的功能
import shutil  # 导入 shutil 模块，用于高级文件操作
import subprocess  # 导入 subprocess 模块，允许生成新进程，连接其输入、输出和错误管道
import sys  # 导入 sys 模块，提供对 Python 解释器的访问
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import warnings  # 导入 warnings 模块，用于发出警告消息
from sysconfig import get_config_var, get_config_vars, get_path  # 从 sysconfig 模块中导入指定的函数

from .runners import (  # 从自定义模块中导入多个编译器运行器类
    CCompilerRunner,
    CppCompilerRunner,
    FortranCompilerRunner
)
from .util import (  # 从自定义模块中导入多个实用工具函数和类
    get_abspath, make_dirs, copy, Glob, ArbitraryDepthGlob,
    glob_at_depth, import_module_from_file, pyx_is_cplus,
    sha256_of_string, sha256_of_file, CompileError
)

if os.name == 'posix':  # 如果操作系统是 POSIX 类型
    objext = '.o'  # 设置对象文件的后缀为 '.o'
elif os.name == 'nt':  # 如果操作系统是 Windows
    objext = '.obj'  # 设置对象文件的后缀为 '.obj'
else:  # 如果操作系统不是 POSIX 也不是 Windows
    warnings.warn("Unknown os.name: {}".format(os.name))  # 发出警告消息，显示未知的操作系统名
    objext = '.o'  # 设置对象文件的后缀为 '.o'


def compile_sources(files, Runner=None, destdir=None, cwd=None, keep_dir_struct=False,
                    per_file_kwargs=None, **kwargs):
    """ Compile source code files to object files.

    Parameters
    ==========

    files : iterable of str
        Paths to source files, if ``cwd`` is given, the paths are taken as relative.
    Runner: CompilerRunner subclass (optional)
        Could be e.g. ``FortranCompilerRunner``. Will be inferred from filename
        extensions if missing.
    destdir: str
        Output directory, if cwd is given, the path is taken as relative.
    cwd: str
        Working directory. Specify to have compiler run in other directory.
        also used as root of relative paths.
    keep_dir_struct: bool
        Reproduce directory structure in `destdir`. default: ``False``
    per_file_kwargs: dict
        Dict mapping instances in ``files`` to keyword arguments.
    \\*\\*kwargs: dict
        Default keyword arguments to pass to ``Runner``.

    Returns
    =======
    List of strings (paths of object files).
    """
    _per_file_kwargs = {}  # 创建空字典 _per_file_kwargs，用于存储每个文件的特定关键字参数

    if per_file_kwargs is not None:  # 如果传入了 per_file_kwargs 参数
        for k, v in per_file_kwargs.items():  # 遍历 per_file_kwargs 中的每一对键值对
            if isinstance(k, Glob):  # 如果键 k 是 Glob 类型的实例
                for path in glob.glob(k.pathname):  # 使用 glob 模块匹配路径名，将路径名添加到 _per_file_kwargs 中
                    _per_file_kwargs[path] = v
            elif isinstance(k, ArbitraryDepthGlob):  # 如果键 k 是 ArbitraryDepthGlob 类型的实例
                for path in glob_at_depth(k.filename, cwd):  # 使用 glob_at_depth 函数匹配指定深度的文件名
                    _per_file_kwargs[path] = v
            else:  # 对于其他类型的键 k
                _per_file_kwargs[k] = v  # 直接将键值对添加到 _per_file_kwargs 中

    # 设置目标目录
    destdir = destdir or '.'  # 如果 destdir 为 None，则默认为当前目录 '.'
    if not os.path.isdir(destdir):  # 如果 destdir 不是一个目录
        if os.path.exists(destdir):  # 如果 destdir 存在但不是目录
            raise OSError("{} is not a directory".format(destdir))  # 抛出 OSError 异常，指示 destdir 不是一个目录
        else:  # 如果 destdir 既不存在也不是一个目录
            make_dirs(destdir)  # 创建目录 destdir

    if cwd is None:  # 如果未指定工作目录 cwd
        cwd = '.'  # 默认使用当前目录 '.'
        for f in files:  # 遍历文件列表 files
            copy(f, destdir, only_update=True, dest_is_dir=True)  # 复制文件到目标目录 destdir 中

    # 编译文件并返回对象文件路径列表
    dstpaths = []  # 创建空列表 dstpaths，用于存储编译后的对象文件路径
    for f in files:  # 遍历文件列表 files
        if keep_dir_struct:  # 如果 keep_dir_struct 为 True，保持目录结构
            name, ext = os.path.splitext(f)  # 分割文件名，获取文件名和扩展名
        else:  # 如果 keep_dir_struct 为 False，不保持目录结构
            name, ext = os.path.splitext(os.path.basename(f))  # 获取文件的基本名称和扩展名
        file_kwargs = kwargs.copy()  # 复制默认的关键字参数到 file_kwargs 中
        file_kwargs.update(_per_file_kwargs.get(f, {}))  # 更新特定文件的关键字参数到 file_kwargs 中
        dstpaths.append(src2obj(f, Runner, cwd=cwd, **file_kwargs))  # 将编译后的对象文件路径添加到 dstpaths 中
    return dstpaths  # 返回对象文件路径列表
def get_mixed_fort_c_linker(vendor=None, cplus=False, cwd=None):
    # 如果未提供 vendor 参数，则尝试从环境变量 SYMPY_COMPILER_VENDOR 中获取，默认为 'gnu'
    vendor = vendor or os.environ.get('SYMPY_COMPILER_VENDOR', 'gnu')

    # 根据 vendor 的值选择不同的编译器和标志
    if vendor.lower() == 'intel':
        # 如果 vendor 是 'intel'，并且 cplus 参数为 True，则返回 FortranCompilerRunner 类型及其特定标志
        if cplus:
            return (FortranCompilerRunner,
                    {'flags': ['-nofor_main', '-cxxlib']}, vendor)
        else:
            # 如果 cplus 参数为 False，则返回 FortranCompilerRunner 类型及其特定标志
            return (FortranCompilerRunner,
                    {'flags': ['-nofor_main']}, vendor)
    elif vendor.lower() == 'gnu' or 'llvm':
        # 如果 vendor 是 'gnu' 或 'llvm'，并且 cplus 参数为 True，则返回 CppCompilerRunner 类型及其特定标志
        if cplus:
            return (CppCompilerRunner,
                    {'lib_options': ['fortran']}, vendor)
        else:
            # 如果 cplus 参数为 False，则返回 FortranCompilerRunner 类型及其默认标志
            return (FortranCompilerRunner,
                    {}, vendor)
    else:
        # 如果 vendor 不在支持的列表中，则抛出 ValueError 异常
        raise ValueError("No vendor found.")


def link(obj_files, out_file=None, shared=False, Runner=None,
         cwd=None, cplus=False, fort=False, extra_objs=None, **kwargs):
    """ Link object files.

    Parameters
    ==========

    obj_files: iterable of str
        Paths to object files.
    out_file: str (optional)
        Path to executable/shared library, if ``None`` it will be
        deduced from the last item in obj_files.
    shared: bool
        Generate a shared library?
    Runner: CompilerRunner subclass (optional)
        If not given the ``cplus`` and ``fort`` flags will be inspected
        (fallback is the C compiler).
    cwd: str
        Path to the root of relative paths and working directory for compiler.
    cplus: bool
        C++ objects? default: ``False``.
    fort: bool
        Fortran objects? default: ``False``.
    extra_objs: list
        List of paths to extra object files / static libraries.
    \\*\\*kwargs: dict
        Keyword arguments passed to ``Runner``.

    Returns
    =======

    The absolute path to the generated shared object / executable.

    """
    # 如果未提供 out_file 参数，则根据 obj_files 中的最后一个文件名推断出输出文件名
    if out_file is None:
        out_file, ext = os.path.splitext(os.path.basename(obj_files[-1]))
        # 如果 shared 参数为 True，则根据系统配置变量获取共享库的文件后缀，并添加到文件名中
        if shared:
            out_file += get_config_var('EXT_SUFFIX')

    # 如果未指定 Runner，则根据 cplus 和 fort 参数选择合适的 Runner 类型
    if not Runner:
        if fort:
            # 获取混合 Fortran 和 C 编译器及其额外的关键字参数
            Runner, extra_kwargs, vendor = \
                get_mixed_fort_c_linker(
                    vendor=kwargs.get('vendor', None),
                    cplus=cplus,
                    cwd=cwd,
                )
            # 将额外的关键字参数合并到 kwargs 中
            for k, v in extra_kwargs.items():
                if k in kwargs:
                    kwargs[k].expand(v)
                else:
                    kwargs[k] = v
        else:
            # 根据 cplus 参数选择合适的默认 Runner 类型
            if cplus:
                Runner = CppCompilerRunner
            else:
                Runner = CCompilerRunner

    # 获取 flags 参数并移除 kwargs 中的 flags
    flags = kwargs.pop('flags', [])
    # 如果 shared 参数为 True，并且 flags 中不包含 '-shared'，则将 '-shared' 添加到 flags 中
    if shared:
        if '-shared' not in flags:
            flags.append('-shared')
    # 获取 run_linker 参数，默认应为 True，否则抛出 ValueError 异常
    run_linker = kwargs.pop('run_linker', True)
    if not run_linker:
        raise ValueError("run_linker was set to False (nonsensical).")

    # 获取绝对路径的 out_file
    out_file = get_abspath(out_file, cwd=cwd)
    # 创建 Runner 实例并运行链接过程
    runner = Runner(obj_files+(extra_objs or []), out_file, flags, cwd=cwd, **kwargs)
    runner.run()
    # 返回生成的共享对象或可执行文件的绝对路径
    return out_file
# 定义一个函数，用于链接 Python 扩展模块生成共享对象文件（.so 文件），以便导入

def link_py_so(obj_files, so_file=None, cwd=None, libraries=None,
               cplus=False, fort=False, extra_objs=None, **kwargs):
    """ Link Python extension module (shared object) for importing
    链接 Python 扩展模块生成共享对象文件（.so 文件），以供导入使用

    Parameters
    ==========

    obj_files: iterable of str
        Paths to object files to be linked.
        待链接的目标文件路径列表
    so_file: str
        Name (path) of shared object file to create. If not specified it will
        have the basname of the last object file in `obj_files` but with the
        extension '.so' (Unix).
        要创建的共享对象文件名（路径）。如果未指定，则默认使用 `obj_files` 中最后一个目标文件的基本名加上 '.so' 后缀（Unix 规范）
    cwd: path string
        Root of relative paths and working directory of linker.
        相对路径的根目录和链接器的工作目录
    libraries: iterable of strings
        Libraries to link against, e.g. ['m'].
        链接的库列表，例如 ['m']
    cplus: bool
        Any C++ objects? default: ``False``.
        是否包含 C++ 对象？默认为 `False`
    fort: bool
        Any Fortran objects? default: ``False``.
        是否包含 Fortran 对象？默认为 `False`
    extra_objs: list
        List of paths of extra object files / static libraries to link against.
        额外的目标文件或静态库路径列表，用于链接
    kwargs**: dict
        Keyword arguments passed to ``link(...)``.
        传递给 `link(...)` 函数的关键字参数

    Returns
    =======

    Absolute path to the generate shared object.
    返回生成的共享对象的绝对路径
    """
    
    libraries = libraries or []

    include_dirs = kwargs.pop('include_dirs', [])
    library_dirs = kwargs.pop('library_dirs', [])

    # Add Python include and library directories
    # PY_LDFLAGS does not available on all python implementations
    # e.g. when with pypy, so it's LDFLAGS we need to use
    if sys.platform == "win32":
        warnings.warn("Windows not yet supported.")
    elif sys.platform == 'darwin':
        cfgDict = get_config_vars()
        kwargs['linkline'] = kwargs.get('linkline', []) + [cfgDict['LDFLAGS']]
        library_dirs += [cfgDict['LIBDIR']]

        # In macOS, linker needs to compile frameworks
        # e.g. "-framework CoreFoundation"
        is_framework = False
        for opt in cfgDict['LIBS'].split():
            if is_framework:
                kwargs['linkline'] = kwargs.get('linkline', []) + ['-framework', opt]
                is_framework = False
            elif opt.startswith('-l'):
                libraries.append(opt[2:])
            elif opt.startswith('-framework'):
                is_framework = True
        # The python library is not included in LIBS
        libfile = cfgDict['LIBRARY']
        libname = ".".join(libfile.split('.')[:-1])[3:]
        libraries.append(libname)

    elif sys.platform[:3] == 'aix':
        # Don't use the default code below
        pass
    else:
        if get_config_var('Py_ENABLE_SHARED'):
            cfgDict = get_config_vars()
            kwargs['linkline'] = kwargs.get('linkline', []) + [cfgDict['LDFLAGS']]
            library_dirs += [cfgDict['LIBDIR']]
            for opt in cfgDict['BLDLIBRARY'].split():
                if opt.startswith('-l'):
                    libraries += [opt[2:]]
        else:
            pass

    flags = kwargs.pop('flags', [])
    needed_flags = ('-pthread',)
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)
    # 调用一个函数并返回其结果
    return link(obj_files, shared=True, flags=flags, cwd=cwd, cplus=cplus, fort=fort,
                include_dirs=include_dirs, libraries=libraries,
                library_dirs=library_dirs, extra_objs=extra_objs, **kwargs)
# 生成一个从 Cython 源文件生成 C 文件的函数
def simple_cythonize(src, destdir=None, cwd=None, **cy_kwargs):
    """ Generates a C file from a Cython source file.

    Parameters
    ==========

    src: str
        Cython 源文件的路径。
    destdir: str (optional)
        输出目录的路径（默认为当前目录）。
    cwd: path string (optional)
        相对路径的根目录（默认为当前目录）。
    **cy_kwargs:
        传递给 cy_compile 的第二个参数。如果在 cy_kwargs 中 cplus=True，则生成 .cpp 文件，否则生成 .c 文件。
    """
    from Cython.Compiler.Main import (
        default_options, CompilationOptions
    )
    from Cython.Compiler.Main import compile as cy_compile

    # 确保源文件以 .pyx 或 .py 结尾
    assert src.lower().endswith('.pyx') or src.lower().endswith('.py')
    cwd = cwd or '.'  # 如果 cwd 未指定，则默认为当前目录
    destdir = destdir or '.'  # 如果 destdir 未指定，则默认为当前目录

    # 根据 cy_kwargs 中的 cplus 参数确定生成的文件扩展名
    ext = '.cpp' if cy_kwargs.get('cplus', False) else '.c'
    # 生成输出的 C 文件名
    c_name = os.path.splitext(os.path.basename(src))[0] + ext

    # 生成输出文件的完整路径
    dstfile = os.path.join(destdir, c_name)

    # 切换当前工作目录以便执行 Cython 编译
    if cwd:
        ori_dir = os.getcwd()
    else:
        ori_dir = '.'
    os.chdir(cwd)
    try:
        cy_options = CompilationOptions(default_options)
        cy_options.__dict__.update(cy_kwargs)
        
        # 如果 cy_kwargs 中未设置 language_level，则设置为 3
        if 'language_level' not in cy_kwargs:
            cy_options.__dict__['language_level'] = 3
        # 执行 Cython 编译
        cy_result = cy_compile([src], cy_options)
        if cy_result.num_errors > 0:
            raise ValueError("Cython compilation failed.")

        # 将生成的 C 文件移动到目标目录
        # 在 macOS 中，生成的 C 文件在与源文件相同的目录中
        # 但 /var 是指向 /private/var 的符号链接，因此需要使用 realpath
        if os.path.realpath(os.path.dirname(src)) != os.path.realpath(destdir):
            if os.path.exists(dstfile):
                os.unlink(dstfile)
            shutil.move(os.path.join(os.path.dirname(src), c_name), destdir)
    finally:
        os.chdir(ori_dir)  # 恢复原始工作目录
    return dstfile


extension_mapping = {
    '.c': (CCompilerRunner, None),
    '.cpp': (CppCompilerRunner, None),
    '.cxx': (CppCompilerRunner, None),
    '.f': (FortranCompilerRunner, None),
    '.for': (FortranCompilerRunner, None),
    '.ftn': (FortranCompilerRunner, None),
    '.f90': (FortranCompilerRunner, None),  # ifort 仅支持 .f90 格式
    '.f95': (FortranCompilerRunner, 'f95'),
    '.f03': (FortranCompilerRunner, 'f2003'),
    '.f08': (FortranCompilerRunner, 'f2008'),
}


def src2obj(srcpath, Runner=None, objpath=None, cwd=None, inc_py=False, **kwargs):
    """ Compiles a source code file to an object file.

    Files ending with '.pyx' assumed to be cython files and
    are dispatched to pyx2obj.

    Parameters
    ==========

    srcpath: str
        源文件的路径。
    Runner: CompilerRunner subclass (optional)
        如果为 ``None``: 根据 srcpath 的扩展名推断。
    objpath : str (optional)
        生成的目标文件的路径。如果为 ``None``: 根据 ``srcpath`` 推断。

    cwd: str (optional)
        指定编译过程中的当前工作目录。

    inc_py: bool (optional)
        是否包含 Python 文件。

    **kwargs:
        其他传递给编译器的参数。
    """
    # cwd: str (optional)
    #     Working directory and root of relative paths. If ``None``: current dir.
    # inc_py: bool
    #     Add Python include path to kwarg "include_dirs". Default: False
    # **kwargs: dict
    #     keyword arguments passed to Runner or pyx2obj

    """
    根据给定的源文件路径(srcpath)获取文件名和扩展名
    """
    name, ext = os.path.splitext(os.path.basename(srcpath))
    
    """
    如果未提供目标文件路径(objpath)，则根据情况进行处理：
    - 如果源文件路径(srcpath)是绝对路径，则目标路径为当前目录('.')
    - 否则，目标路径为源文件路径(srcpath)所在的目录，若为空，则使用当前目录('.')
    """
    if objpath is None:
        if os.path.isabs(srcpath):
            objpath = '.'
        else:
            objpath = os.path.dirname(srcpath)
            objpath = objpath or '.'  # avoid objpath == ''

    """
    如果目标路径(objpath)是一个目录，则将目标文件名设置为该目录下的名称和目标文件扩展名(objext)的组合
    """
    if os.path.isdir(objpath):
        objpath = os.path.join(objpath, name + objext)

    """
    从kwargs中获取include_dirs参数，并确保其为列表类型
    """
    include_dirs = kwargs.pop('include_dirs', [])

    """
    如果inc_py为True，则获取Python包含路径，并将其添加到include_dirs中，如果尚未添加的话
    """
    if inc_py:
        py_inc_dir = get_path('include')
        if py_inc_dir not in include_dirs:
            include_dirs.append(py_inc_dir)

    """
    如果源文件的扩展名是'.pyx'，则调用pyx2obj函数进行编译，并传递相应的参数
    """
    if ext.lower() == '.pyx':
        return pyx2obj(srcpath, objpath=objpath, include_dirs=include_dirs, cwd=cwd,
                       **kwargs)

    """
    如果Runner未定义，则根据扩展名从extension_mapping中获取对应的Runner类和标准输出
    并将标准输出作为kwargs中的'std'参数，若未指定的话
    """
    if Runner is None:
        Runner, std = extension_mapping[ext.lower()]
        if 'std' not in kwargs:
            kwargs['std'] = std

    """
    将所需的编译标志(flags)从kwargs中获取并确保其为列表类型
    并添加所需的编译标志('-fPIC')，如果尚未添加的话
    """
    flags = kwargs.pop('flags', [])
    needed_flags = ('-fPIC',)
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)

    """
    检查是否运行链接器(run_linker)，如果设置为True，则抛出CompileError异常
    """
    run_linker = kwargs.pop('run_linker', False)
    if run_linker:
        raise CompileError("src2obj called with run_linker=True")

    """
    创建Runner对象，并运行编译操作
    """
    runner = Runner([srcpath], objpath, include_dirs=include_dirs,
                    run_linker=run_linker, cwd=cwd, flags=flags, **kwargs)
    runner.run()

    """
    返回目标文件路径(objpath)
    """
    return objpath
# 定义函数 pyx2obj，用于将 Cython 源文件编译为目标对象文件的便捷函数
def pyx2obj(pyxpath, objpath=None, destdir=None, cwd=None,
            include_dirs=None, cy_kwargs=None, cplus=None, **kwargs):
    """
    Convenience function

    If cwd is specified, pyxpath and dst are taken to be relative
    If only_update is set to `True` the modification time is checked
    and compilation is only run if the source is newer than the
    destination

    Parameters
    ==========

    pyxpath: str
        Path to Cython source file.
    objpath: str (optional)
        Path to object file to generate.
    destdir: str (optional)
        Directory to put generated C file. When ``None``: directory of ``objpath``.
    cwd: str (optional)
        Working directory and root of relative paths.
    include_dirs: iterable of path strings (optional)
        Passed onto src2obj and via cy_kwargs['include_path']
        to simple_cythonize.
    cy_kwargs: dict (optional)
        Keyword arguments passed onto `simple_cythonize`
    cplus: bool (optional)
        Indicate whether C++ is used. default: auto-detect using ``.util.pyx_is_cplus``.
    compile_kwargs: dict
        keyword arguments passed onto src2obj

    Returns
    =======

    Absolute path of generated object file.

    """
    # 断言 pyxpath 是以 '.pyx' 结尾的文件路径
    assert pyxpath.endswith('.pyx')
    # 设置默认工作目录为当前目录
    cwd = cwd or '.'
    # 如果未提供 objpath，设为当前目录
    objpath = objpath or '.'
    # 如果未提供 destdir，设为 objpath 的父目录
    destdir = destdir or os.path.dirname(objpath)

    # 获取绝对路径的 objpath
    abs_objpath = get_abspath(objpath, cwd=cwd)

    # 如果 abs_objpath 是目录，则生成对应的 object 文件路径
    if os.path.isdir(abs_objpath):
        pyx_fname = os.path.basename(pyxpath)
        name, ext = os.path.splitext(pyx_fname)
        objpath = os.path.join(objpath, name + objext)

    # 初始化 cy_kwargs，默认设置输出目录为当前工作目录
    cy_kwargs = cy_kwargs or {}
    cy_kwargs['output_dir'] = cwd
    # 自动检测是否使用 C++
    if cplus is None:
        cplus = pyx_is_cplus(pyxpath)
    cy_kwargs['cplus'] = cplus

    # 进行简单的 Cython 编译，生成中间的 C 文件
    interm_c_file = simple_cythonize(pyxpath, destdir=destdir, cwd=cwd, **cy_kwargs)

    # 如果未指定 include_dirs，则设置为空列表
    include_dirs = include_dirs or []
    # 获取 kwargs 中的 flags 参数，如果需要的标志不在 flags 中，则添加
    flags = kwargs.pop('flags', [])
    needed_flags = ('-fwrapv', '-pthread', '-fPIC')
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)

    # 获取 kwargs 中的 options 参数
    options = kwargs.pop('options', [])

    # 如果 strict_aliasing 设置为 True，则抛出 CompileError
    if kwargs.pop('strict_aliasing', False):
        raise CompileError("Cython requires strict aliasing to be disabled.")

    # 根据是否使用 C++ 设置 std 的值，默认为 'c99' 或 'c++98'
    if cplus:
        std = kwargs.pop('std', 'c++98')
    else:
        std = kwargs.pop('std', 'c99')

    # 调用 src2obj 函数进行最终的编译，返回生成的对象文件的绝对路径
    return src2obj(interm_c_file, objpath=objpath, cwd=cwd,
                   include_dirs=include_dirs, flags=flags, std=std,
                   options=options, inc_py=True, strict_aliasing=False,
                   **kwargs)


# 定义辅助函数 _any_X，用于检查列表中是否存在特定类型的源文件
def _any_X(srcs, cls):
    for src in srcs:
        name, ext = os.path.splitext(src)
        key = ext.lower()
        # 如果扩展名存在于映射表中，并且与指定的类匹配，则返回 True
        if key in extension_mapping:
            if extension_mapping[key][0] == cls:
                return True
    # 若未找到匹配的类型，则返回 False
    return False


# 定义函数 any_fortran_src，用于检查列表中是否存在 Fortran 源文件
def any_fortran_src(srcs):
    return _any_X(srcs, FortranCompilerRunner)


# 定义函数 any_cplus_src，用于检查列表中是否存在 C++ 源文件
def any_cplus_src(srcs):
    return _any_X(srcs, CppCompilerRunner)
# 编译、链接和导入 Python 扩展模块的函数
def compile_link_import_py_ext(sources, extname=None, build_dir='.', compile_kwargs=None,
                               link_kwargs=None, extra_objs=None):
    """ Compiles sources to a shared object (Python extension) and imports it

    Sources in ``sources`` which is imported. If shared object is newer than the sources, they
    are not recompiled but instead it is imported.

    Parameters
    ==========

    sources : list of strings
        List of paths to sources.
    extname : string
        Name of extension (default: ``None``).
        If ``None``: taken from the last file in ``sources`` without extension.
    build_dir: str
        Path to directory in which objects files etc. are generated.
    compile_kwargs: dict
        keyword arguments passed to ``compile_sources``
    link_kwargs: dict
        keyword arguments passed to ``link_py_so``
    extra_objs: list
        List of paths to (prebuilt) object files / static libraries to link against.

    Returns
    =======

    The imported module from of the Python extension.
    """
    # 如果未指定 extname，则从 sources 中的最后一个文件名（不含扩展名）获取
    if extname is None:
        extname = os.path.splitext(os.path.basename(sources[-1]))[0]

    # 设置编译和链接的关键字参数，默认为空字典
    compile_kwargs = compile_kwargs or {}
    link_kwargs = link_kwargs or {}

    try:
        # 尝试从文件导入模块
        mod = import_module_from_file(os.path.join(build_dir, extname), sources)
    except ImportError:
        # 如果导入失败，则编译源文件
        objs = compile_sources(list(map(get_abspath, sources)), destdir=build_dir,
                               cwd=build_dir, **compile_kwargs)
        # 链接生成共享对象文件
        so = link_py_so(objs, cwd=build_dir, fort=any_fortran_src(sources),
                        cplus=any_cplus_src(sources), extra_objs=extra_objs, **link_kwargs)
        # 从生成的共享对象文件导入模块
        mod = import_module_from_file(so)

    # 返回导入的模块
    return mod


def _write_sources_to_build_dir(sources, build_dir):
    # 如果未提供 build_dir，则创建临时目录
    build_dir = build_dir or tempfile.mkdtemp()
    # 检查目录是否存在，否则抛出异常
    if not os.path.isdir(build_dir):
        raise OSError("Non-existent directory: ", build_dir)

    # 存储源文件的路径
    source_files = []
    for name, src in sources:
        dest = os.path.join(build_dir, name)
        differs = True
        # 计算内存中源文件的 SHA256 值
        sha256_in_mem = sha256_of_string(src.encode('utf-8')).hexdigest()
        if os.path.exists(dest):
            # 如果目标文件存在，则比较磁盘上的 SHA256 值和内存中的值
            if os.path.exists(dest + '.sha256'):
                with open(dest + '.sha256') as fh:
                    sha256_on_disk = fh.read()
            else:
                sha256_on_disk = sha256_of_file(dest).hexdigest()

            differs = sha256_on_disk != sha256_in_mem
        
        # 如果文件不同，则写入新的源文件和其 SHA256 值
        if differs:
            with open(dest, 'wt') as fh:
                fh.write(src)
            with open(dest + '.sha256', 'wt') as fh:
                fh.write(sha256_in_mem)
        
        # 将目标文件路径添加到源文件列表中
        source_files.append(dest)
    
    # 返回源文件路径列表和构建目录路径
    return source_files, build_dir


def compile_link_import_strings(sources, build_dir=None, **kwargs):
    """ Compiles, links and imports extension module from source.

    Parameters
    ==========

    sources : iterable of name/source pair tuples
    """
    # 获取源文件和构建目录路径，通过调用 `_write_sources_to_build_dir` 函数完成
    source_files, build_dir = _write_sources_to_build_dir(sources, build_dir)
    
    # 调用 `compile_link_import_py_ext` 函数编译、链接并导入 Python 扩展模块
    mod = compile_link_import_py_ext(source_files, build_dir=build_dir, **kwargs)
    
    # 创建包含构建目录信息的字典，作为返回结果的一部分
    info = {"build_dir": build_dir}
    
    # 返回编译并导入的扩展模块 `mod` 和包含构建目录信息的字典 `info`
    return mod, info
# 编译、链接并运行由源代码构建的程序。

def compile_run_strings(sources, build_dir=None, clean=False, compile_kwargs=None, link_kwargs=None):
    """ Compiles, links and runs a program built from sources.

    Parameters
    ==========

    sources : iterable of name/source pair tuples
        源代码文件名和内容的可迭代元组
    build_dir : string (default: None)
        路径。``None`` 表示使用临时目录。
    clean : bool
        是否在使用后删除 build_dir。仅在 ``build_dir`` 为 ``None`` 时有效（创建临时目录）。
        如果 ``clean == True`` 且 ``build_dir != None``，则引发 ``ValueError``。
        这也会将返回的 info 字典中的 ``build_dir`` 设置为 ``None``。
    compile_kwargs: dict
        传递给 ``compile_sources`` 的关键字参数
    link_kwargs: dict
        传递给 ``link`` 的关键字参数

    Returns
    =======

    (stdout, stderr): pair of strings
        程序的标准输出和标准错误输出
    info: dict
        包含退出状态为 'exit_status' 和 ``build_dir`` 为 'build_dir'

    """
    if clean and build_dir is not None:
        raise ValueError("Automatic removal of build_dir is only available for temporary directory.")
    try:
        # 将源代码写入 build_dir，返回源文件列表和使用的 build_dir
        source_files, build_dir = _write_sources_to_build_dir(sources, build_dir)
        # 编译源文件成目标文件列表，并使用 get_abspath 转换为绝对路径
        objs = compile_sources(list(map(get_abspath, source_files)), destdir=build_dir,
                               cwd=build_dir, **(compile_kwargs or {}))
        # 链接目标文件成可执行程序，根据源文件类型选择是否使用 Fortran 或 C++
        prog = link(objs, cwd=build_dir,
                    fort=any_fortran_src(source_files),
                    cplus=any_cplus_src(source_files), **(link_kwargs or {}))
        # 启动子进程执行程序，捕获其标准输出和标准错误输出
        p = subprocess.Popen([prog], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        exit_status = p.wait()  # 等待子进程结束并获取退出状态
        stdout, stderr = [txt.decode('utf-8') for txt in p.communicate()]  # 解码子进程的输出文本
    finally:
        # 如果需要清理临时目录，且 build_dir 存在，则递归删除 build_dir 并将其设为 None
        if clean and os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
            build_dir = None
    # 返回执行结果的标准输出、标准错误输出和相关信息
    info = {"exit_status": exit_status, "build_dir": build_dir}
    return (stdout, stderr), info
```