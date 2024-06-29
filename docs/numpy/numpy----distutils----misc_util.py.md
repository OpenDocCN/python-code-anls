# `.\numpy\numpy\distutils\misc_util.py`

```
# 导入标准库模块
import os           # 提供与操作系统交互的功能
import re           # 提供正则表达式操作
import sys          # 提供与 Python 解释器交互的功能
import copy         # 提供复制对象的功能
import glob         # 提供文件通配符匹配
import atexit       # 提供退出时执行函数的注册和调用
import tempfile     # 提供创建临时文件和目录的功能
import subprocess   # 提供创建和管理子进程的功能
import shutil       # 提供高级文件操作功能
import multiprocessing  # 提供多进程处理支持
import textwrap     # 提供文本包装和填充功能
import importlib.util  # 提供导入模块的工具
from threading import local as tlocal  # 提供线程本地存储功能
from functools import reduce         # 提供高阶函数操作

import distutils    # Python 的标准库中的工具模块
from distutils.errors import DistutilsError  # 引入 distutils 中的错误异常类

# 线程本地存储，用于存储每个线程的临时目录，以确保每个线程只创建一个临时目录
_tdata = tlocal()

# 存储所有创建的临时目录，以便在退出时删除
_tmpdirs = []

def clean_up_temporary_directory():
    """
    清理临时目录的函数，在程序退出时被注册调用

    """
    if _tmpdirs is not None:
        for d in _tmpdirs:
            try:
                shutil.rmtree(d)  # 尝试删除临时目录及其内容
            except OSError:
                pass

atexit.register(clean_up_temporary_directory)  # 注册清理临时目录的函数，确保程序退出时执行

# 声明 __all__ 列表，指定模块中可以被外部导入的符号
__all__ = ['Configuration', 'get_numpy_include_dirs', 'default_config_dict',
           'dict_append', 'appendpath', 'generate_config_py',
           'get_cmd', 'allpath', 'get_mathlibs',
           'terminal_has_colors', 'red_text', 'green_text', 'yellow_text',
           'blue_text', 'cyan_text', 'cyg2win32', 'mingw32', 'all_strings',
           'has_f_sources', 'has_cxx_sources', 'filter_sources',
           'get_dependencies', 'is_local_src_dir', 'get_ext_source_files',
           'get_script_files', 'get_lib_source_files', 'get_data_files',
           'dot_join', 'get_frame', 'minrelpath', 'njoin',
           'is_sequence', 'is_string', 'as_list', 'gpaths', 'get_language',
           'get_build_architecture', 'get_info', 'get_pkg_info',
           'get_num_build_jobs', 'sanitize_cxx_flags',
           'exec_mod_from_location']

class InstallableLib:
    """
    可安装库的容器类，用于存储安装库的信息

    Parameters
    ----------
    name : str
        安装库的名称
    build_info : dict
        存储构建信息的字典
    target_dir : str
        指定安装库的绝对路径

    See Also
    --------
    Configuration.add_installed_library

    Notes
    -----
    这三个参数被存储为同名的属性。

    """
    def __init__(self, name, build_info, target_dir):
        self.name = name               # 设置名称属性
        self.build_info = build_info   # 设置构建信息属性
        self.target_dir = target_dir   # 设置目标目录属性

def get_num_build_jobs():
    """
    获取由 setup.py 的 --parallel 命令行参数设置的并行构建作业数
    如果未设置该命令，检查环境变量 NPY_NUM_BUILD_JOBS 的设置。如果未设置，返回系统的处理器数量，最大为 8（以防止过载）。

    Returns
    -------
    out : int
        可以运行的并行作业数

    """
    from numpy.distutils.core import get_distribution  # 导入 numpy.distutils 中的 get_distribution 函数
    try:
        cpu_count = len(os.sched_getaffinity(0))   # 尝试获取当前进程可用的 CPU 数量
    except AttributeError:
        cpu_count = multiprocessing.cpu_count()    # 获取系统中的 CPU 核心数
    cpu_count = min(cpu_count, 8)                  # 将 CPU 核心数限制在最大值为 8
    envjobs = int(os.environ.get("NPY_NUM_BUILD_JOBS", cpu_count))  # 获取环境变量中设置的并行作业数，如果未设置则使用 cpu_count
    # 获取当前项目的发行信息
    dist = get_distribution()
    # 如果发行信息为None，说明在配置阶段可能未定义，直接返回envjobs
    if dist is None:
        return envjobs

    # 获取三个构建命令对象中的并行属性，任意一个设置了并行作业数即可，选择最大的
    cmdattr = (getattr(dist.get_command_obj('build'), 'parallel', None),
               getattr(dist.get_command_obj('build_ext'), 'parallel', None),
               getattr(dist.get_command_obj('build_clib'), 'parallel', None))
    
    # 如果三个命令对象的并行属性都为None，则返回envjobs
    if all(x is None for x in cmdattr):
        return envjobs
    else:
        # 返回三个命令对象中并行属性不为None的最大值
        return max(x for x in cmdattr if x is not None)
# 引用警告模块
import warnings

# 定义函数：将参数列表中的每个参数进行引号处理
def quote_args(args):
    """Quote list of arguments.

    .. deprecated:: 1.22.
    """
    # 发出警告：'quote_args'已被弃用
    warnings.warn('"quote_args" is deprecated.',
                  DeprecationWarning, stacklevel=2)
    
    # 将参数列表转换为列表形式
    args = list(args)
    
    # 遍历参数列表，对包含空格但未被引号包围的参数进行引号处理
    for i in range(len(args)):
        a = args[i]
        if ' ' in a and a[0] not in '"\'':
            args[i] = '"%s"' % (a)
    
    # 返回处理后的参数列表
    return args

# 定义函数：将'/-'分隔的路径名转换为操作系统的路径分隔符
def allpath(name):
    "Convert a /-separated pathname to one using the OS's path separator."
    split = name.split('/')
    return os.path.join(*split)

# 定义函数：返回相对于父路径的路径
def rel_path(path, parent_path):
    """Return path relative to parent_path."""
    # 使用realpath避免符号链接目录的问题（参见gh-7707）
    pd = os.path.realpath(os.path.abspath(parent_path))
    apath = os.path.realpath(os.path.abspath(path))
    
    if len(apath) < len(pd):
        return path
    if apath == pd:
        return ''
    if pd == apath[:len(pd)]:
        assert apath[len(pd)] in [os.sep], repr((path, apath[len(pd)]))
        path = apath[len(pd)+1:]
    return path

# 定义函数：根据调用堆栈中的帧对象返回模块的路径
def get_path_from_frame(frame, parent_path=None):
    """Return path of the module given a frame object from the call stack.

    Returned path is relative to parent_path when given,
    otherwise it is absolute path.
    """
    # 尝试在帧中查找文件名
    try:
        caller_file = eval('__file__', frame.f_globals, frame.f_locals)
        d = os.path.dirname(os.path.abspath(caller_file))
    except NameError:
        # 如果__file__未定义，则尝试使用__name__
        caller_name = eval('__name__', frame.f_globals, frame.f_locals)
        __import__(caller_name)
        mod = sys.modules[caller_name]
        
        if hasattr(mod, '__file__'):
            d = os.path.dirname(os.path.abspath(mod.__file__))
        else:
            # 执行setup.py时，返回当前目录的绝对路径
            d = os.path.abspath('.')
    
    # 如果指定了父路径，则返回相对于父路径的路径
    if parent_path is not None:
        d = rel_path(d, parent_path)
    
    # 返回模块路径或者当前目录（如果未找到模块路径）
    return d or '.'

# 定义函数：连接两个或多个路径名组件，解析'..'和'.'，并使用操作系统的路径分隔符
def njoin(*path):
    """Join two or more pathname components +
    - convert a /-separated pathname to one using the OS's path separator.
    - resolve `..` and `.` from path.

    Either passing n arguments as in njoin('a','b'), or a sequence
    of n names as in njoin(['a','b']) is handled, or a mixture of such arguments.
    """
    paths = []
    
    # 遍历传入的路径名组件
    for p in path:
        if is_sequence(p):
            # 如果是序列，则递归处理
            paths.append(njoin(*p))
        else:
            assert is_string(p)
            paths.append(p)
    
    path = paths
    
    # 如果路径名组件为空，则返回空字符串
    if not path:
        # njoin()
        joined = ''
    else:
        # 否则连接路径名组件，并返回连接后的路径
        # njoin('a', 'b')
        joined = os.path.join(*path)
    # 检查操作系统路径分隔符是否为斜杠'/'
    if os.path.sep != '/':
        # 如果不是斜杠'/'，则用操作系统的路径分隔符替换路径中的斜杠'/'
        joined = joined.replace('/', os.path.sep)
    # 调用minrelpath函数计算路径的最短相对路径，并返回结果
    return minrelpath(joined)
# 返回numpyconfig.h中MATHLIB行的内容
def get_mathlibs(path=None):
    """Return the MATHLIB line from numpyconfig.h
    """
    # 如果提供了路径，则使用给定路径下的_numpyconfig.h文件
    if path is not None:
        config_file = os.path.join(path, '_numpyconfig.h')
    else:
        # 否则，在每个numpy包含目录中查找文件
        dirs = get_numpy_include_dirs()
        for path in dirs:
            fn = os.path.join(path, '_numpyconfig.h')
            # 找到文件后设置配置文件路径并退出循环
            if os.path.exists(fn):
                config_file = fn
                break
        else:
            # 如果在所有目录中都找不到文件，则引发异常
            raise DistutilsError('_numpyconfig.h not found in numpy include '
                'dirs %r' % (dirs,))
    
    # 打开配置文件并读取内容
    with open(config_file) as fid:
        mathlibs = []
        s = '#define MATHLIB'
        # 逐行读取文件内容
        for line in fid:
            # 如果行以指定的标识符开头，则提取并处理相应的数学库信息
            if line.startswith(s):
                value = line[len(s):].strip()
                if value:
                    mathlibs.extend(value.split(','))
    # 返回解析得到的数学库信息列表
    return mathlibs

# 解析路径中的`..`和`.`，返回规范化后的路径
def minrelpath(path):
    """Resolve `..` and '.' from path.
    """
    # 如果路径不是字符串，则直接返回
    if not is_string(path):
        return path
    # 如果路径中没有`.`，则直接返回
    if '.' not in path:
        return path
    l = path.split(os.sep)
    while l:
        try:
            i = l.index('.', 1)
        except ValueError:
            break
        del l[i]
    j = 1
    while l:
        try:
            i = l.index('..', j)
        except ValueError:
            break
        if l[i-1]=='..':
            j += 1
        else:
            del l[i], l[i-1]
            j = 1
    # 如果路径列表为空，则返回空字符串；否则返回重新连接后的路径
    if not l:
        return ''
    return os.sep.join(l)

# 对glob.glob返回的结果进行排序，以解决https://bugs.python.org/issue30461问题
def sorted_glob(fileglob):
    """sorts output of python glob for https://bugs.python.org/issue30461
    to allow extensions to have reproducible build results"""
    # 对glob.glob的结果进行排序并返回
    return sorted(glob.glob(fileglob))

# 对路径列表进行修正，确保它是一个序列，并且不是字符串
def _fix_paths(paths, local_path, include_non_existing):
    assert is_sequence(paths), repr(type(paths))
    new_paths = []
    # 断言路径不是字符串，避免意外的类型错误
    assert not is_string(paths), repr(paths)
    # 遍历给定的路径列表 paths
    for n in paths:
        # 检查当前路径 n 是否为字符串
        if is_string(n):
            # 如果路径中包含通配符 '*' 或 '?'，则使用 sorted_glob 函数获取匹配的路径列表 p
            if '*' in n or '?' in n:
                p = sorted_glob(n)
                # 使用 njoin 函数将 local_path 和 n 进行拼接，再使用 sorted_glob 函数获取匹配的路径列表 p2
                p2 = sorted_glob(njoin(local_path, n))
                # 如果 p2 列表非空，则将其添加到 new_paths 列表中
                if p2:
                    new_paths.extend(p2)
                # 否则，如果 p 列表非空，则将其添加到 new_paths 列表中
                elif p:
                    new_paths.extend(p)
                else:
                    # 如果 include_non_existing 为 True，则将当前路径 n 添加到 new_paths 列表中
                    if include_non_existing:
                        new_paths.append(n)
                    # 打印未能解析匹配模式的信息
                    print('could not resolve pattern in %r: %r' %
                            (local_path, n))
            else:
                # 使用 njoin 函数将 local_path 和 n 进行拼接，得到完整路径 n2
                n2 = njoin(local_path, n)
                # 如果 n2 存在于文件系统中，则将其添加到 new_paths 列表中
                if os.path.exists(n2):
                    new_paths.append(n2)
                else:
                    # 否则，如果 n 存在于文件系统中，则将 n 添加到 new_paths 列表中
                    if os.path.exists(n):
                        new_paths.append(n)
                    # 如果 include_non_existing 为 True，则将 n 添加到 new_paths 列表中
                    elif include_non_existing:
                        new_paths.append(n)
                    # 如果 n 依然不存在，则打印不存在路径的信息
                    if not os.path.exists(n):
                        print('non-existing path in %r: %r' %
                                (local_path, n))

        # 如果 n 是一个序列（如列表或元组），则递归调用 _fix_paths 函数处理其中的路径，并将结果扩展到 new_paths 列表中
        elif is_sequence(n):
            new_paths.extend(_fix_paths(n, local_path, include_non_existing))
        else:
            # 如果 n 不是字符串也不是序列，则直接将其添加到 new_paths 列表中
            new_paths.append(n)
    # 返回处理后的路径列表 new_paths，并对每个路径使用 minrelpath 函数进行最小化处理
    return [minrelpath(p) for p in new_paths]
# 将路径列表应用 glob 函数，并根据需要添加本地路径
def gpaths(paths, local_path='', include_non_existing=True):
    """Apply glob to paths and prepend local_path if needed.
    """
    # 如果 paths 是字符串，则将其转换成元组
    if is_string(paths):
        paths = (paths,)
    # 返回修正后的路径列表
    return _fix_paths(paths, local_path, include_non_existing)

# 创建临时文件，返回文件对象和文件名
def make_temp_file(suffix='', prefix='', text=True):
    # 如果 _tdata 没有 tempdir 属性，则创建临时目录
    if not hasattr(_tdata, 'tempdir'):
        _tdata.tempdir = tempfile.mkdtemp()
        _tmpdirs.append(_tdata.tempdir)
    # 创建临时文件，返回文件对象和文件名
    fid, name = tempfile.mkstemp(suffix=suffix,
                                 prefix=prefix,
                                 dir=_tdata.tempdir,
                                 text=text)
    fo = os.fdopen(fid, 'w')
    return fo, name

# 用于彩色终端输出的钩子
def terminal_has_colors():
    # 如果是 cygwin 平台且未设置 USE_COLOR 环境变量，则返回 0
    if sys.platform=='cygwin' and 'USE_COLOR' not in os.environ:
        return 0
    # 如果标准输出是终端并且支持颜色
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        try:
            # 尝试导入 curses 模块
            import curses
            curses.setupterm()
            # 如果终端支持颜色功能，则返回 1
            if (curses.tigetnum("colors") >= 0
                and curses.tigetnum("pairs") >= 0
                and ((curses.tigetstr("setf") is not None
                      and curses.tigetstr("setb") is not None)
                     or (curses.tigetstr("setaf") is not None
                         and curses.tigetstr("setab") is not None)
                     or curses.tigetstr("scp") is not None)):
                return 1
        except Exception:
            pass
    # 其他情况返回 0
    return 0

# 如果终端支持颜色，则定义颜色代码和文本修饰函数
if terminal_has_colors():
    _colour_codes = dict(black=0, red=1, green=2, yellow=3,
                         blue=4, magenta=5, cyan=6, white=7, default=9)
    def colour_text(s, fg=None, bg=None, bold=False):
        seq = []
        # 如果 bold 为真，则加入 '1' 到序列中
        if bold:
            seq.append('1')
        # 如果 fg 存在，则根据颜色返回对应的代码
        if fg:
            fgcode = 30 + _colour_codes.get(fg.lower(), 0)
            seq.append(str(fgcode))
        # 如果 bg 存在，则根据颜色返回对应的代码
        if bg:
            bgcode = 40 + _colour_codes.get(bg.lower(), 7)
            seq.append(str(bgcode))
        # 如果有需要修改文本颜色的指令，则返回修改后的文本，否则返回原始文本
        if seq:
            return '\x1b[%sm%s\x1b[0m' % (';'.join(seq), s)
        else:
            return s
else:
    # 如果终端不支持颜色，则定义文本颜色修改函数
    def colour_text(s, fg=None, bg=None):
        return s

# 定义默认文本颜色修改函数
def default_text(s):
    return colour_text(s, 'default')
# 定义红色文本颜色修改函数
def red_text(s):
    return colour_text(s, 'red')
# 定义绿色文本颜色修改函数
def green_text(s):
    return colour_text(s, 'green')
# 定义黄色文本颜色修改函数
def yellow_text(s):
    return colour_text(s, 'yellow')
# 定义青色文本颜色修改函数
def cyan_text(s):
    return colour_text(s, 'cyan')
# 定义蓝色文本颜色修改函数
def blue_text(s):
    return colour_text(s, 'blue')

# 将 cygwin 路径转换为 win32 路径
def cyg2win32(path: str) -> str:
    # 将路径从 Cygwin 本地格式转换为 Windows 本地格式
    # 使用 cygpath 工具（Base 安装的一部分）来进行实际转换。如果失败，则返回原始路径
    # 处理默认的“/cygdrive”挂载前缀，以及“/proc/cygdrive”便携前缀，自定义的 cygdrive 前缀，如“/”或“/mnt”，以及绝对路径，如“/usr/src/”或“/home/username”
    # 参数：
    # path：str，要转换的路径
    # 返回：
    # converted_path：str，转换后的路径
    # 注：
    # cygpath 工具的文档：
    # https://cygwin.com/cygwin-ug-net/cygpath.html
    # 它封装的 C 函数的文档：
    # https://cygwin.com/cygwin-api/func-cygwin-conv-path.html
    
    if sys.platform != "cygwin":
        # 如果不是在 Cygwin 平台上，直接返回原始路径
        return path
    # 调用子进程执行 cygpath 命令，传入参数"--windows"和路径，获取输出
    return subprocess.check_output(
        ["/usr/bin/cygpath", "--windows", path], text=True
    )
# 判断是否在mingw32环境中
def mingw32():
    """Return true when using mingw32 environment.
    """
    # 如果操作系统是win32
    if sys.platform=='win32':
        # 如果环境变量OSTYPE的值是'msys'
        if os.environ.get('OSTYPE', '')=='msys':
            return True
        # 如果环境变量MSYSTEM的值是'MINGW32'
        if os.environ.get('MSYSTEM', '')=='MINGW32':
            return True
    # 如果以上条件都不满足，则返回False
    return False

# 返回MSVC运行库的版本，由__MSC_VER__宏定义
def msvc_runtime_version():
    "Return version of MSVC runtime library, as defined by __MSC_VER__ macro"
    # 在sys.version中查找'MSC v.'的位置
    msc_pos = sys.version.find('MSC v.')
    if msc_pos != -1:
        # 如果找到'MSC v.'，则获取其后6到10位的数字作为版本号
        msc_ver = int(sys.version[msc_pos+6:msc_pos+10])
    else:
        # 如果没有找到'MSC v.'，则版本号为None
        msc_ver = None
    return msc_ver

# 返回Python是否使用MSVC构建的MSVC运行库的名称
def msvc_runtime_library():
    "Return name of MSVC runtime library if Python was built with MSVC >= 7"
    # 获取MSVC运行库的主要版本号
    ver = msvc_runtime_major ()
    if ver:
        # 如果版本号小于140，返回'msvcr'加上版本号的字符串
        if ver < 140:
            return "msvcr%i" % ver
        # 如果版本号大于等于140，返回'vcruntime'加上版本号的字符串
        else:
            return "vcruntime%i" % ver
    else:
        # 如果没有版本号，返回None
        return None

# 返回MSVC运行库的主要版本号
def msvc_runtime_major():
    "Return major version of MSVC runtime coded like get_build_msvc_version"
    # 构建一个字典，包含MSVC运行库版本号与主要版本号的对应关系
    major = {1300:  70,  # MSVC 7.0
             1310:  71,  # MSVC 7.1
             1400:  80,  # MSVC 8
             1500:  90,  # MSVC 9  (aka 2008)
             1600: 100,  # MSVC 10 (aka 2010)
             1900: 140,  # MSVC 14 (aka 2015)
    }.get(msvc_runtime_version(), None)
    return major

#########################

#XXX 需要支持同时为C和C++的文件.C
cxx_ext_match = re.compile(r'.*\.(cpp|cxx|cc)\Z', re.I).match
fortran_ext_match = re.compile(r'.*\.(f90|f95|f77|for|ftn|f)\Z', re.I).match
f90_ext_match = re.compile(r'.*\.(f90|f95)\Z', re.I).match
f90_module_name_match = re.compile(r'\s*module\s*(?P<name>[\w_]+)', re.I).match
# 获取Fortran f90模块的名称列表
def _get_f90_modules(source):
    """Return a list of Fortran f90 module names that
    given source file defines.
    """
    # 如果给定的源文件不是f90格式，返回空列表
    if not f90_ext_match(source):
        return []
    modules = []
    with open(source) as f:
        for line in f:
            # 匹配并提取f90模块的名称
            m = f90_module_name_match(line)
            if m:
                name = m.group('name')
                modules.append(name)
                # break  # XXX can we assume that there is one module per file?
    return modules

# 判断一个对象是否是字符串
def is_string(s):
    return isinstance(s, str)

# 判断列表中的所有项是否都是字符串对象
def all_strings(lst):
    """Return True if all items in lst are string objects. """
    for item in lst:
        if not is_string(item):
            return False
    return True

# 判断一个对象是否是序列（即可迭代的对象，如列表、元组、字符串）
def is_sequence(seq):
    if is_string(seq):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True

# 判断一个字符串是否是glob模式（带*或?的字符串）
def is_glob_pattern(s):
    return is_string(s) and ('*' in s or '?' in s)

# 将一个对象转换为列表
def as_list(seq):
    if is_sequence(seq):
        return list(seq)
    else:
        return [seq]

# 获取源文件的语言类型
def get_language(sources):
    # not used in numpy/scipy packages, use build_ext.detect_language instead
    """Determine language value (c,f77,f90) from sources """
    language = None
    # 遍历给定的源列表 sources
    for source in sources:
        # 检查当前源是否是字符串类型
        if isinstance(source, str):
            # 如果当前源的文件扩展名匹配 Fortran 90 的扩展名
            if f90_ext_match(source):
                # 设置语言类型为 'f90'
                language = 'f90'
                # 跳出循环，已确定语言类型
                break
            # 如果当前源的文件扩展名匹配 Fortran 77 的扩展名
            elif fortran_ext_match(source):
                # 设置语言类型为 'f77'
                language = 'f77'
    # 返回确定的语言类型
    return language
# 检查给定的源文件列表中是否包含 Fortran 文件，如果有则返回 True，否则返回 False
def has_f_sources(sources):
    for source in sources:
        # 调用 fortran_ext_match 函数检查文件名是否匹配 Fortran 文件扩展名
        if fortran_ext_match(source):
            return True
    return False

# 检查给定的源文件列表中是否包含 C++ 文件，如果有则返回 True，否则返回 False
def has_cxx_sources(sources):
    for source in sources:
        # 调用 cxx_ext_match 函数检查文件名是否匹配 C++ 文件扩展名
        if cxx_ext_match(source):
            return True
    return False

# 对给定的源文件列表进行过滤，返回四个文件名列表：C 文件、C++ 文件、Fortran 文件、Fortran 90 模块文件
def filter_sources(sources):
    c_sources = []
    cxx_sources = []
    f_sources = []
    fmodule_sources = []
    for source in sources:
        if fortran_ext_match(source):
            # 如果文件名匹配 Fortran 文件扩展名，则进一步检查是否是 Fortran 90 模块
            modules = _get_f90_modules(source)
            if modules:
                fmodule_sources.append(source)
            else:
                f_sources.append(source)
        elif cxx_ext_match(source):
            cxx_sources.append(source)
        else:
            c_sources.append(source)
    return c_sources, cxx_sources, f_sources, fmodule_sources

# 从目录列表中获取所有的 *.h 文件，并返回一个包含这些文件名的列表
def _get_headers(directory_list):
    headers = []
    for d in directory_list:
        # 使用 sorted_glob 函数获取指定目录下的所有 *.h 文件，并将结果添加到 headers 列表中
        head = sorted_glob(os.path.join(d, "*.h"))  # XXX: *.hpp files??
        headers.extend(head)
    return headers

# 从源文件列表中获取所有文件的父目录，并返回一个包含这些目录名的列表
def _get_directories(list_of_sources):
    direcs = []
    for f in list_of_sources:
        # 使用 os.path.split 函数获取文件的父目录，并确保目录名不重复
        d = os.path.split(f)
        if d[0] != '' and not d[0] in direcs:
            direcs.append(d[0])
    return direcs

# 构造用于确定是否需要重新编译文件的命令行表示，并返回该字符串
def _commandline_dep_string(cc_args, extra_postargs, pp_opts):
    cmdline = 'commandline: '
    cmdline += ' '.join(cc_args)
    cmdline += ' '.join(extra_postargs)
    cmdline += ' '.join(pp_opts) + '\n'
    return cmdline

# 分析源文件列表中的包含语句，获取所有被包含的头文件，并返回一个包含这些头文件名的列表
def get_dependencies(sources):
    # 调用 _get_directories 函数获取源文件列表中所有文件的父目录列表，然后调用 _get_headers 获取这些目录中的头文件
    return _get_headers(_get_directories(sources))

# 检查目录是否是本地目录，并返回 True 或 False
def is_local_src_dir(directory):
    if not is_string(directory):
        return False
    abs_dir = os.path.abspath(directory)
    c = os.path.commonprefix([os.getcwd(), abs_dir])
    new_dir = abs_dir[len(c):].split(os.sep)
    if new_dir and not new_dir[0]:
        new_dir = new_dir[1:]
    if new_dir and new_dir[0]=='build':
        return False
    new_dir = os.sep.join(new_dir)
    return os.path.isdir(new_dir)

# 生成指定路径下的源文件列表，排除特定的目录和文件类型，使用生成器实现
def general_source_files(top_path):
    pruned_directories = {'CVS':1, '.svn':1, 'build':1}
    prune_file_pat = re.compile(r'(?:[~#]|\.py[co]|\.o)$')
    for dirpath, dirnames, filenames in os.walk(top_path, topdown=True):
        pruned = [ d for d in dirnames if d not in pruned_directories ]
        dirnames[:] = pruned
        for f in filenames:
            # 排除指定文件类型的文件，并生成文件的完整路径
            if not prune_file_pat.search(f):
                yield os.path.join(dirpath, f)

# 生成指定路径下的源文件目录列表和文件列表，排除特定的目录和文件类型，使用生成器实现
def general_source_directories_files(top_path):
    # 返回相对于 top_path 的目录名和包含的文件列表
    """Return a directory name relative to top_path and
    files contained.
    """
    # 要忽略的目录列表，不包含在结果中
    pruned_directories = ['CVS', '.svn', 'build']
    # 用于匹配需要剔除的文件模式的正则表达式
    prune_file_pat = re.compile(r'(?:[~#]|\.py[co]|\.o)$')
    # 从 top_path 开始，递归遍历文件系统中的目录
    for dirpath, dirnames, filenames in os.walk(top_path, topdown=True):
        # 从当前目录的子目录列表中剔除 pruned_directories 中的目录
        pruned = [d for d in dirnames if d not in pruned_directories]
        dirnames[:] = pruned  # 更新 dirnames 列表，以便下一步的遍历不包含被剔除的目录
        for d in dirnames:
            # 构建子目录的完整路径
            dpath = os.path.join(dirpath, d)
            # 计算子目录相对于 top_path 的相对路径
            rpath = rel_path(dpath, top_path)
            files = []
            # 遍历子目录中的文件列表
            for f in os.listdir(dpath):
                fn = os.path.join(dpath, f)
                # 如果文件是普通文件且不匹配 prune_file_pat 的模式，则加入 files 列表
                if os.path.isfile(fn) and not prune_file_pat.search(fn):
                    files.append(fn)
            # 生成相对路径 rpath 和文件列表 files 的元组
            yield rpath, files
    
    # 处理 top_path 目录本身，生成其相对路径和包含的文件列表
    dpath = top_path
    # 计算 top_path 目录相对于自身的路径（即空字符串）
    rpath = rel_path(dpath, top_path)
    # 获取 top_path 目录下所有文件的完整路径列表
    filenames = [os.path.join(dpath, f) for f in os.listdir(dpath) if not prune_file_pat.search(f)]
    # 过滤出 filenames 中真正的文件路径（不是目录路径）
    files = [f for f in filenames if os.path.isfile(f)]
    # 生成相对路径 rpath 和文件列表 files 的元组
    yield rpath, files
# 返回具有指定扩展名的源文件及同一目录中的任何包含文件
def get_ext_source_files(ext):
    # 创建空文件名列表
    filenames = []
    # 获取所有源文件
    sources = [_m for _m in ext.sources if is_string(_m)]
    # 将源文件添加到文件名列表中
    filenames.extend(sources)
    # 获取源文件的依赖项，并将其添加到文件名列表中
    filenames.extend(get_dependencies(sources))
    # 遍历依赖列表
    for d in ext.depends:
        # 如果依赖是本地源文件目录，则将其下的所有通用源文件添加到文件名列表中
        if is_local_src_dir(d):
            filenames.extend(list(general_source_files(d)))
        # 如果依赖是文件，则将该文件添加到文件名列表中
        elif os.path.isfile(d):
            filenames.append(d)
    return filenames

# 获取脚本文件
def get_script_files(scripts):
    # 获取所有脚本文件并返回
    scripts = [_m for _m in scripts if is_string(_m)]
    return scripts

# 返回库的源文件
def get_lib_source_files(lib):
    # 创建空文件名列表
    filenames = []
    # 获取库的源文件
    sources = lib[1].get('sources', [])
    # 将源文件添加到文件名列表中
    sources = [_m for _m in sources if is_string(_m)]
    filenames.extend(sources)
    # 获取源文件的依赖项，并将其添加到文件名列表中
    filenames.extend(get_dependencies(sources))
    # 获取库的依赖项
    depends = lib[1].get('depends', [])
    for d in depends:
        # 如果依赖是本地源文件目录，则将其下的所有通用源文件添加到文件名列表中
        if is_local_src_dir(d):
            filenames.extend(list(general_source_files(d)))
        # 如果依赖是文件，则将该文件添加到文件名列表中
        elif os.path.isfile(d):
            filenames.append(d)
    return filenames

# 获取共享库的扩展名
def get_shared_lib_extension(is_python_ext=False):
    """Return the correct file extension for shared libraries.

    Parameters
    ----------
    is_python_ext : bool, optional
        Whether the shared library is a Python extension.  Default is False.

    Returns
    -------
    so_ext : str
        The shared library extension.

    Notes
    -----
    For Python shared libs, `so_ext` will typically be '.so' on Linux and OS X,
    and '.pyd' on Windows.  For Python >= 3.2 `so_ext` has a tag prepended on
    POSIX systems according to PEP 3149.

    """
    # 获取配置变量
    confvars = distutils.sysconfig.get_config_vars()
    # 获取共享库的扩展名
    so_ext = confvars.get('EXT_SUFFIX', '')

    # 如果不是Python扩展，则根据操作系统返回正确的共享库扩展名
    if not is_python_ext:
        # 硬编码已知的值，配置变量（包括SHLIB_SUFFIX）不可靠（参见＃3182）
        # 在3.3.1及更早版本中，darwin，windows和debug linux是错误的
        if (sys.platform.startswith('linux') or
            sys.platform.startswith('gnukfreebsd')):
            so_ext = '.so'
        elif sys.platform.startswith('darwin'):
            so_ext = '.dylib'
        elif sys.platform.startswith('win'):
            so_ext = '.dll'
        else:
            # 对于未知平台，回退到配置变量
            # 修复Python> = 3.2的长扩展，参见PEP 3149。
            if 'SOABI' in confvars:
                # 除非存在SOABI配置变量，否则不执行任何操作
                so_ext = so_ext.replace('.' + confvars.get('SOABI'), '', 1)

    return so_ext

# 获取数据文件
def get_data_files(data):
    # 如果数据是字符串，则返回其列表
    if is_string(data):
        return [data]
    # 否则，获取数据源并创建空文件名列表
    sources = data[1]
    filenames = []
    # 对于给定的源列表中的每个元素进行迭代
    for s in sources:
        # 检查当前元素是否是可调用的对象，如果是，则跳过本次迭代
        if hasattr(s, '__call__'):
            continue
        # 检查当前元素是否是本地源目录路径
        if is_local_src_dir(s):
            # 将该目录下所有一般源文件的文件名添加到文件名列表中
            filenames.extend(list(general_source_files(s)))
        # 如果当前元素是字符串类型
        elif is_string(s):
            # 检查该字符串是否是一个文件路径
            if os.path.isfile(s):
                # 将文件路径添加到文件名列表中
                filenames.append(s)
            else:
                # 打印出文件路径不存在的警告信息
                print('Not existing data file:', s)
        else:
            # 如果当前元素不是可调用对象，也不是本地源目录路径，也不是字符串，则引发类型错误
            raise TypeError(repr(s))
    # 返回最终的文件名列表
    return filenames
def dot_join(*args):
    # 将传入的参数按照"."连接成一个字符串
    return '.'.join([a for a in args if a])

def get_frame(level=0):
    """Return frame object from call stack with given level.
    """
    try:
        # 返回调用栈中指定层级的帧对象
        return sys._getframe(level+1)
    except AttributeError:
        # 如果没有找到指定层级的帧对象，返回当前异常的帧对象
        frame = sys.exc_info()[2].tb_frame
        for _ in range(level+1):
            frame = frame.f_back
        return frame


######################

class Configuration:

    _list_keys = ['packages', 'ext_modules', 'data_files', 'include_dirs',
                  'libraries', 'headers', 'scripts', 'py_modules',
                  'installed_libraries', 'define_macros']
    _dict_keys = ['package_dir', 'installed_pkg_config']
    _extra_keys = ['name', 'version']

    numpy_include_dirs = []

    def todict(self):
        """
        Return a dictionary compatible with the keyword arguments of distutils
        setup function.

        Examples
        --------
        >>> setup(**config.todict())                           #doctest: +SKIP
        """

        # 优化数据文件
        self._optimize_data_files()
        d = {}
        # 创建一个空字典
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        # 将配置中的列表、字典和额外的键合并在一起
        for n in known_keys:
            a = getattr(self, n)
            if a:
                # 如果属性值不为空，则将属性名和属性值添加到字典中
                d[n] = a
        return d

    def info(self, message):
        if not self.options['quiet']:
            # 如果选项中的quiet值为False，则输出消息
            print(message)

    def warn(self, message):
        sys.stderr.write('Warning: %s\n' % (message,))
        # 输出警告信息到标准错误流

    def set_options(self, **options):
        """
        Configure Configuration instance.

        The following options are available:
         - ignore_setup_xxx_py
         - assume_default_configuration
         - delegate_options_to_subpackages
         - quiet

        """
        for key, value in options.items():
            if key in self.options:
                # 如果选项名存在于配置中，则将选项值赋给配置的对应选项
                self.options[key] = value
            else:
                # 如果选项名不存在于配置中，则抛出值错误异常
                raise ValueError('Unknown option: '+key)

    def get_distribution(self):
        """Return the distutils distribution object for self."""
        from numpy.distutils.core import get_distribution
        # 导入并返回相应的distutils分发对象
        return get_distribution()

    def _wildcard_get_subpackage(self, subpackage_name,
                                 parent_name,
                                 caller_level = 1):
        l = subpackage_name.split('.')
        subpackage_path = njoin([self.local_path]+l)
        dirs = [_m for _m in sorted_glob(subpackage_path) if os.path.isdir(_m)]
        config_list = []
        for d in dirs:
            if not os.path.isfile(njoin(d, '__init__.py')):
                continue
            if 'build' in d.split(os.sep):
                continue
            n = '.'.join(d.split(os.sep)[-len(l):])
            c = self.get_subpackage(n,
                                    parent_name = parent_name,
                                    caller_level = caller_level+1)
            config_list.extend(c)
        return config_list
    # 从 setup.py 文件中获取配置信息
    def _get_configuration_from_setup_py(self, setup_py,
                                         subpackage_name,
                                         subpackage_path,
                                         parent_name,
                                         caller_level = 1):
        # 为了防止 setup.py 引入本地模块，将其所在目录添加到 sys.path 中
        sys.path.insert(0, os.path.dirname(setup_py)
        try:
            # 获取 setup.py 文件名（不带扩展名）
            setup_name = os.path.splitext(os.path.basename(setup_py))[0]
            # 组合模块名
            n = dot_join(self.name, subpackage_name, setup_name)
            # 从指定位置执行模块
            setup_module = exec_mod_from_location('_'.join(n.split('.')), setup_py)
            # 如果模块没有定义 configuration 属性
            if not hasattr(setup_module, 'configuration'):
                # 如果不假设默认配置，则警告
                if not self.options['assume_default_configuration']:
                    self.warn('Assuming default configuration (%s does not define configuration())' % (setup_module))
                # 创建默认配置
                config = Configuration(subpackage_name, parent_name, self.top_path, subpackage_path, caller_level = caller_level + 1)
            else:
                # 组合父模块名
                pn = dot_join(*([parent_name] + subpackage_name.split('.')[:-1]))
                args = (pn,)
                # 如果 configuration 函数参数大于 1，需要传入 self.top_path
                if setup_module.configuration.__code__.co_argcount > 1:
                    args = args + (self.top_path,)
                # 调用 setup_module 中的 configuration 函数
                config = setup_module.configuration(*args)
            # 如果配置的名称不与父模块和子模块的组合名称相同，发出警告
            if config.name != dot_join(parent_name, subpackage_name):
                self.warn('Subpackage %r configuration returned as %r' % (dot_join(parent_name, subpackage_name), config.name))
        finally:
            # 在 finally 块中删除刚刚添加的路径
            del sys.path[0]
        # 返回获取到的配置
        return config
    # 返回子包的配置列表
    def get_subpackage(self,subpackage_name,
                       subpackage_path=None,
                       parent_name=None,
                       caller_level = 1):
        """Return list of subpackage configurations.

        Parameters
        ----------
        subpackage_name : str or None
            子包的名称用于获取配置。在subpackage_name中的'*'将被处理为通配符。
        subpackage_path : str
            如果为None，则假定路径为本地路径加上subpackage_name。如果在subpackage_path中找不到setup.py文件，则使用默认配置。
        parent_name : str
            父名称。
        """
        if subpackage_name is None:
            if subpackage_path is None:
                raise ValueError(
                    "either subpackage_name or subpackage_path must be specified")
            subpackage_name = os.path.basename(subpackage_path)

        # 处理通配符
        l = subpackage_name.split('.')
        if subpackage_path is None and '*' in subpackage_name:
            return self._wildcard_get_subpackage(subpackage_name,
                                                 parent_name,
                                                 caller_level = caller_level+1)
        assert '*' not in subpackage_name, repr((subpackage_name, subpackage_path, parent_name))
        if subpackage_path is None:
            subpackage_path = njoin([self.local_path] + l)
        else:
            subpackage_path = njoin([subpackage_path] + l[:-1])
            subpackage_path = self.paths([subpackage_path])[0]
        setup_py = njoin(subpackage_path, self.setup_name)
        if not self.options['ignore_setup_xxx_py']:
            if not os.path.isfile(setup_py):
                setup_py = njoin(subpackage_path,
                                 'setup_%s.py' % (subpackage_name))
        if not os.path.isfile(setup_py):
            if not self.options['assume_default_configuration']:
                self.warn('Assuming default configuration '\
                          '(%s/{setup_%s,setup}.py was not found)' \
                          % (os.path.dirname(setup_py), subpackage_name))
            config = Configuration(subpackage_name, parent_name,
                                   self.top_path, subpackage_path,
                                   caller_level = caller_level+1)
        else:
            config = self._get_configuration_from_setup_py(
                setup_py,
                subpackage_name,
                subpackage_path,
                parent_name,
                caller_level = caller_level + 1)
        if config:
            return [config]
        else:
            return []
    # 添加一个子包到当前的 Configuration 实例
    def add_subpackage(self,subpackage_name,
                       subpackage_path=None,
                       standalone = False):
        """Add a sub-package to the current Configuration instance.
    
        This is useful in a setup.py script for adding sub-packages to a
        package.
    
        Parameters
        ----------
        subpackage_name : str
            name of the subpackage
        subpackage_path : str
            if given, the subpackage path such as the subpackage is in
            subpackage_path / subpackage_name. If None,the subpackage is
            assumed to be located in the local path / subpackage_name.
        standalone : bool
        """
    
        if standalone:
            parent_name = None
        else:
            parent_name = self.name
        config_list = self.get_subpackage(subpackage_name, subpackage_path,
                                          parent_name = parent_name,
                                          caller_level = 2)
        if not config_list:
            self.warn('No configuration returned, assuming unavailable.')
        for config in config_list:
            d = config
            if isinstance(config, Configuration):
                d = config.todict()
            assert isinstance(d, dict), repr(type(d))
    
            self.info('Appending %s configuration to %s' \
                      % (d.get('name'), self.name))
            self.dict_append(**d)
    
        # 获取分发情况
        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized,'\
                      ' it may be too late to add a subpackage '+ subpackage_name)
    
    # 优化数据文件
    def _optimize_data_files(self):
        data_dict = {}
        for p, files in self.data_files:
            if p not in data_dict:
                data_dict[p] = set()
            for f in files:
                data_dict[p].add(f)
        self.data_files[:] = [(p, list(files)) for p, files in data_dict.items()]
    
    ### XXX Implement add_py_modules
    
    # 添加宏定义到配置
    def add_define_macros(self, macros):
        """Add define macros to configuration
    
        Add the given sequence of macro name and value duples to the beginning
        of the define_macros list. This list will be visible to all extension
        modules of the current package.
        """
        dist = self.get_distribution()
        if dist is not None:
            if not hasattr(dist, 'define_macros'):
                dist.define_macros = []
            dist.define_macros.extend(macros)
        else:
            self.define_macros.extend(macros)
    # 将给定的路径添加到配置的包含目录中
    def add_include_dirs(self,*paths):
        """
        Add paths to configuration include directories.
    
        Add the given sequence of paths to the beginning of the include_dirs
        list. This list will be visible to all extension modules of the
        current package.
        """
        # 将给定的路径转换成包含目录
        include_dirs = self.paths(paths)
        # 获取当前包的分发对象
        dist = self.get_distribution()
        # 如果有分发对象，则将包含目录添加到分发对象中
        if dist is not None:
            if dist.include_dirs is None:
                dist.include_dirs = []
            dist.include_dirs.extend(include_dirs)
        # 如果没有分发对象，则将包含目录添加到当前对象中
        else:
            self.include_dirs.extend(include_dirs)
    
    # 将可安装的头文件添加到配置中
    def add_headers(self,*files):
        """
        Add installable headers to configuration.
    
        Add the given sequence of files to the beginning of the headers list.
        By default, headers will be installed under <python-
        include>/<self.name.replace('.','/')>/ directory. If an item of files
        is a tuple, then its first argument specifies the actual installation
        location relative to the <python-include> path.
    
        Parameters
        ----------
        files : str or seq
            Argument(s) can be either:
    
                * 2-sequence (<includedir suffix>,<path to header file(s)>)
                * path(s) to header file(s) where python includedir suffix will
                  default to package name.
        """
        headers = []
        # 遍历文件路径，根据类型进行处理
        for path in files:
            if is_string(path):
                [headers.append((self.name, p)) for p in self.paths(path)]
            else:
                if not isinstance(path, (tuple, list)) or len(path) != 2:
                    raise TypeError(repr(path))
                [headers.append((path[0], p)) for p in self.paths(path[1])]
        # 获取当前包的分发对象
        dist = self.get_distribution()
        # 如果有分发对象，则将头文件添加到分发对象中
        if dist is not None:
            if dist.headers is None:
                dist.headers = []
            dist.headers.extend(headers)
        # 如果没有分发对象，则将头文件添加到当前对象中
        else:
            self.headers.extend(headers)
    
    # 将路径应用于全局路径，并添加本地路径（如果需要的话）
    def paths(self,*paths,**kws):
        """
        Apply glob to paths and prepend local_path if needed.
    
        Applies glob.glob(...) to each path in the sequence (if needed) and
        pre-pends the local_path if needed. Because this is called on all
        source lists, this allows wildcard characters to be specified in lists
        of sources for extension modules and libraries and scripts and allows
        path-names be relative to the source directory.
        """
        include_non_existing = kws.get('include_non_existing', True)
        return gpaths(paths,
                      local_path = self.local_path,
                      include_non_existing=include_non_existing)
    
    # 修正路径字典
    def _fix_paths_dict(self, kw):
        for k in kw.keys():
            v = kw[k]
            if k in ['sources', 'depends', 'include_dirs', 'library_dirs',
                     'module_dirs', 'extra_objects']:
                new_v = self.paths(v)
                kw[k] = new_v
    # 将库添加到配置中
    def add_library(self,name,sources,**build_info):
        """
        Add library to configuration.

        Parameters
        ----------
        name : str
            Name of the extension.
        sources : sequence
            List of the sources. The list of sources may contain functions
            (called source generators) which must take an extension instance
            and a build directory as inputs and return a source file or list of
            source files or None. If None is returned then no sources are
            generated. If the Extension instance has no sources after
            processing all source generators, then no extension module is
            built.
        build_info : dict, optional
            The following keys are allowed:

                * depends
                * macros
                * include_dirs
                * extra_compiler_args
                * extra_f77_compile_args
                * extra_f90_compile_args
                * f2py_options
                * language

        """
        # 调用内部方法 _add_library，参数包括名称、源代码、安装目录和构建信息
        self._add_library(name, sources, None, build_info)

        # 获取分发对象
        dist = self.get_distribution()
        # 如果分发对象不为空则发出警告
        if dist is not None:
            self.warn('distutils distribution has been initialized,'\
                      ' it may be too late to add a library '+ name)

    # 内部方法，用于增加库和已安装库
    def _add_library(self, name, sources, install_dir, build_info):
        """Common implementation for add_library and add_installed_library. Do
        not use directly"""
        # 复制构建信息，将源代码添加到构建信息
        build_info = copy.copy(build_info)
        build_info['sources'] = sources

        # 有时候，依赖关系默认不为空列表，如果未给出依赖关系，则添加一个空列表
        if not 'depends' in build_info:
            build_info['depends'] = []

        # 修正路径字典
        self._fix_paths_dict(build_info)

        # 将库添加到库列表中，以便与 build_clib 一起构建
        self.libraries.append((name, build_info))
    # 定义一个方法，用于添加已安装的库
    def add_installed_library(self, name, sources, install_dir, build_info=None):
        """
        Similar to add_library, but the specified library is installed.
    
        Most C libraries used with ``distutils`` are only used to build python
        extensions, but libraries built through this method will be installed
        so that they can be reused by third-party packages.
    
        Parameters
        ----------
        name : str
            Name of the installed library.
        sources : sequence
            List of the library's source files. See `add_library` for details.
        install_dir : str
            Path to install the library, relative to the current sub-package.
        build_info : dict, optional
            The following keys are allowed:
    
                * depends
                * macros
                * include_dirs
                * extra_compiler_args
                * extra_f77_compile_args
                * extra_f90_compile_args
                * f2py_options
                * language
    
        Returns
        -------
        None
    
        See Also
        --------
        add_library, add_npy_pkg_config, get_info
    
        Notes
        -----
        The best way to encode the options required to link against the specified
        C libraries is to use a "libname.ini" file, and use `get_info` to
        retrieve the required options (see `add_npy_pkg_config` for more
        information).
    
        """
        # 如果未提供构建信息，将构建信息设为空字典
        if not build_info:
            build_info = {}
        
        # 将安装目录路径拼接到当前子包路径上
        install_dir = os.path.join(self.package_path, install_dir)
        
        # 调用私有方法 _add_library，并传入参数
        self._add_library(name, sources, install_dir, build_info)
        
        # 将已安装的库信息添加到已安装库列表中
        self.installed_libraries.append(InstallableLib(name, build_info, install_dir))
    
    # 定义一个方法，用于添加脚本文件
    def add_scripts(self,*files):
        """Add scripts to configuration.
    
        Add the sequence of files to the beginning of the scripts list.
        Scripts will be installed under the <prefix>/bin/ directory.
    
        """
        # 将传入的文件路径转换为绝对路径
        scripts = self.paths(files)
        
        # 获取当前的发行版
        dist = self.get_distribution()
        
        # 如果发行版存在
        if dist is not None:
            # 如果发行版的脚本列表为None，则将其设为空列表
            if dist.scripts is None:
                dist.scripts = []
            # 将文件列表添加到发行版的脚本列表中
            dist.scripts.extend(scripts)
        else:
            # 如果发行版不存在，则将文件列表添加到当前对象的脚本列表中
            self.scripts.extend(scripts)
    # 在字典属性中添加另一个字典的内容
    def dict_append(self,**dict):
        # 对于列表类型的属性，将传入的字典中对应的键的值添加到列表末尾
        for key in self.list_keys:
            a = getattr(self, key)
            a.extend(dict.get(key, []))
        # 对于字典类型的属性，将传入的字典中对应的键值对更新到字典中
        for key in self.dict_keys:
            a = getattr(self, key)
            a.update(dict.get(key, {}))
        # 获取已知的键的列表
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        # 循环遍历传入的字典中的键
        for key in dict.keys():
            # 如果键不在已知的键列表中
            if key not in known_keys:
                # 获取属性的值
                a = getattr(self, key, None)
                # 如果值存在且与传入字典中对应键的值相等，则不做任何操作
                if a and a==dict[key]: continue
                # 否则，发出警告并更新属性的值
                self.warn('Inheriting attribute %r=%r from %r' \
                          % (key, dict[key], dict.get('name', '?')))
                setattr(self, key, dict[key])
                # 更新额外的键列表
                self.extra_keys.append(key)
            # 如果键在额外的键列表中
            elif key in self.extra_keys:
                # 发出信息，忽略设置属性的尝试
                self.info('Ignoring attempt to set %r (from %r to %r)' \
                          % (key, getattr(self, key), dict[key]))
            # 如果键在已知的键列表中
            elif key in known_keys:
                # 键已在上面处理过，不做任何操作
                pass
            # 如果键未知
            else:
                # 抛出异常
                raise ValueError("Don't know about key=%r" % (key))

    # 返回实例的字符串表示
    def __str__(self):
        from pprint import pformat
        # 获取已知的键列表
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        # 初始化字符串
        s = '<'+5*'-' + '\n'
        s += 'Configuration of '+self.name+':\n'
        # 对已知的键列表进行排序
        known_keys.sort()
        # 遍历已知的键列表
        for k in known_keys:
            # 获取属性的值
            a = getattr(self, k, None)
            # 如果属性的值存在
            if a:
                # 将属性的键值对格式化为字符串，并添加到s中
                s += '%s = %s\n' % (k, pformat(a))
        s += 5*'-' + '>'
        return s

    # 返回numpy.distutils的配置命令实例
    def get_config_cmd(self):
        """
        返回numpy.distutils配置命令实例。
        """
        cmd = get_cmd('config')
        cmd.ensure_finalized()
        cmd.dump_source = 0
        cmd.noisy = 0
        old_path = os.environ.get('PATH')
        if old_path:
            path = os.pathsep.join(['.', old_path])
            os.environ['PATH'] = path
        return cmd

    # 返回临时构建文件的路径
    def get_build_temp_dir(self):
        """
        返回一个临时目录的路径，临时文件应该放在其中。
        """
        cmd = get_cmd('build')
        cmd.ensure_finalized()
        return cmd.build_temp

    # 检查Fortran 77编译器的可用性
    def have_f77c(self):
        """Check for availability of Fortran 77 compiler.

        在源代码生成函数中使用它，以确保已初始化设置分发实例。

        Notes
        -----
        如果Fortran 77编译器可用（因为能够成功编译简单的Fortran 77代码），则返回True。
        """
        # 简单的Fortran 77子例程
        simple_fortran_subroutine = '''
        subroutine simple
        end
        '''
        # 获取配置命令
        config_cmd = self.get_config_cmd()
        # 尝试编译简单的Fortran 77子例程，返回是否成功的标志
        flag = config_cmd.try_compile(simple_fortran_subroutine, lang='f77')
        return flag
    # 检查是否有 Fortran 90 编译器可用
    def have_f90c(self):
        """Check for availability of Fortran 90 compiler.

        Use it inside source generating function to ensure that
        setup distribution instance has been initialized.

        Notes
        -----
        True if a Fortran 90 compiler is available (because a simple Fortran
        90 code was able to be compiled successfully)
        """
        # 创建简单的 Fortran 90 子例程代码
        simple_fortran_subroutine = '''
        subroutine simple
        end
        '''
        # 获取配置命令
        config_cmd = self.get_config_cmd()
        # 尝试编译简单的 Fortran 90 子例程代码，并返回编译结果
        flag = config_cmd.try_compile(simple_fortran_subroutine, lang='f90')
        return flag

    # 向扩展或库项目的库和包含路径中添加库
    def append_to(self, extlib):
        """Append libraries, include_dirs to extension or library item.
        """
        # 如果 extlib 是序列
        if is_sequence(extlib):
            lib_name, build_info = extlib
            # 向 build_info 的 libraries 和 include_dirs 中添加库和包含路径
            dict_append(build_info,
                        libraries=self.libraries,
                        include_dirs=self.include_dirs)
        else:
            from numpy.distutils.core import Extension
            assert isinstance(extlib, Extension), repr(extlib)
            extlib.libraries.extend(self.libraries)
            extlib.include_dirs.extend(self.include_dirs)

    # 获取路径的 SVN 版本号
    def _get_svn_revision(self, path):
        """Return path's SVN revision number.
        """
        try:
            # 使用 subprocess 模块获取 SVN 版本号
            output = subprocess.check_output(['svnversion'], cwd=path)
        except (subprocess.CalledProcessError, OSError):
            pass
        else:
            # 使用正则表达式匹配 SVN 版本号
            m = re.match(rb'(?P<revision>\d+)', output)
            if m:
                return int(m.group('revision'))

        # 如果是 Windows 平台并且存在 SVN_ASP_DOT_NET_HACK 环境变量
        if sys.platform=='win32' and os.environ.get('SVN_ASP_DOT_NET_HACK', None):
            entries = njoin(path, '_svn', 'entries')
        else:
            entries = njoin(path, '.svn', 'entries')
        # 如果文件存在，打开它并读取内容
        if os.path.isfile(entries):
            with open(entries) as f:
                fstr = f.read()
            # 如果文件内容是以 '<?xml' 开头，使用正则表达式查找 SVN 版本号
            if fstr[:5] == '<?xml':  # pre 1.4
                m = re.search(r'revision="(?P<revision>\d+)"', fstr)
                if m:
                    return int(m.group('revision'))
            else:  # 非 xml 格式的文件，检查内容中的 SVN 版本号
                m = re.search(r'dir[\n\r]+(?P<revision>\d+)', fstr)
                if m:
                    return int(m.group('revision'))
        # 如果以上条件都不满足，返回 None
        return None
    # 定义一个方法用于获取指定路径下 Mercurial 版本控制系统的版本号
    def _get_hg_revision(self, path):
        """Return path's Mercurial revision number.
        """
        try:
            # 尝试运行命令行程序 'hg identify --num' 来获取版本号
            output = subprocess.check_output(
                ['hg', 'identify', '--num'], cwd=path)
        except (subprocess.CalledProcessError, OSError):
            # 如果出现异常（命令执行错误或系统错误），则忽略异常，不做任何操作
            pass
        else:
            # 如果成功获取到输出，使用正则表达式匹配输出中的版本号
            m = re.match(rb'(?P<revision>\d+)', output)
            if m:
                # 如果匹配成功，返回匹配到的版本号转换为整数类型
                return int(m.group('revision'))

        # 构造分支信息文件路径和缓存文件路径
        branch_fn = njoin(path, '.hg', 'branch')
        branch_cache_fn = njoin(path, '.hg', 'branch.cache')

        # 如果分支信息文件存在
        if os.path.isfile(branch_fn):
            branch0 = None
            # 打开分支信息文件并读取第一行作为当前分支信息
            with open(branch_fn) as f:
                revision0 = f.read().strip()

            # 初始化一个空的分支映射字典
            branch_map = {}
            # 打开分支缓存文件，逐行读取分支和版本信息，构建分支到版本号的映射
            with open(branch_cache_fn) as f:
                for line in f:
                    branch1, revision1  = line.split()[:2]
                    # 如果缓存中的版本号与当前版本号相同，则认为找到了当前分支
                    if revision1 == revision0:
                        branch0 = branch1
                    try:
                        # 尝试将版本号转换为整数类型，如果失败则继续下一次循环
                        revision1 = int(revision1)
                    except ValueError:
                        continue
                    # 将分支和对应的版本号加入映射字典中
                    branch_map[branch1] = revision1

            # 返回当前分支对应的版本号（如果找到的话）
            return branch_map.get(branch0)

        # 如果没有找到分支信息文件，则返回空值 None
        return None
    def get_version(self, version_file=None, version_variable=None):
        """尝试获取包的版本字符串。

        如果无法检测到版本信息，则返回当前包的版本字符串或 None。

        Notes
        -----
        该方法扫描文件 __version__.py、<packagename>_version.py、version.py、
        和 __svn_version__.py，查找字符串变量 version、__version__ 和
        <packagename>_version，直到找到版本号为止。
        """
        # 尝试从对象属性中获取版本号
        version = getattr(self, 'version', None)
        if version is not None:
            return version

        # 从版本文件中获取版本号
        if version_file is None:
            files = ['__version__.py',
                     self.name.split('.')[-1]+'_version.py',
                     'version.py',
                     '__svn_version__.py',
                     '__hg_version__.py']
        else:
            files = [version_file]

        # 指定版本变量名
        if version_variable is None:
            version_vars = ['version',
                            '__version__',
                            self.name.split('.')[-1]+'_version']
        else:
            version_vars = [version_variable]

        # 遍历文件列表，尝试获取版本号
        for f in files:
            fn = njoin(self.local_path, f)
            if os.path.isfile(fn):
                info = ('.py', 'U', 1)
                name = os.path.splitext(os.path.basename(fn))[0]
                n = dot_join(self.name, name)
                try:
                    # 从指定位置执行模块
                    version_module = exec_mod_from_location(
                                        '_'.join(n.split('.')), fn)
                except ImportError as e:
                    self.warn(str(e))
                    version_module = None

                # 如果模块为空，继续下一个文件
                if version_module is None:
                    continue

                # 尝试从模块中获取版本变量
                for a in version_vars:
                    version = getattr(version_module, a, None)
                    if version is not None:
                        break

                # 尝试使用 versioneer 模块获取版本号
                try:
                    version = version_module.get_versions()['version']
                except AttributeError:
                    pass

                if version is not None:
                    break

        # 如果找到版本号，则设置对象属性并返回版本号
        if version is not None:
            self.version = version
            return version

        # 尝试获取 SVN 或 Mercurial 的修订号作为版本号
        revision = self._get_svn_revision(self.local_path)
        if revision is None:
            revision = self._get_hg_revision(self.local_path)

        # 如果获取到修订号，则将其转换为字符串并设置对象属性
        if revision is not None:
            version = str(revision)
            self.version = version

        # 返回最终获取的版本号
        return version
    def make_svn_version_py(self, delete=True):
        """为 data_files 列表追加一个数据函数，用于生成当前包目录下的 __svn_version__.py 文件。

        从 SVN 的修订版本号生成包的 __svn_version__.py 文件，
        它在 Python 退出时将被删除，但在执行 sdist 等命令时将可用。

        注意
        -----
        如果 __svn_version__.py 文件已存在，则不执行任何操作。
        
        这个方法适用于带有 SVN 仓库的源代码目录。
        """
        target = njoin(self.local_path, '__svn_version__.py')
        # 获取当前目录下的 SVN 修订版本号
        revision = self._get_svn_revision(self.local_path)
        # 如果目标文件已经存在或者无法获取到 SVN 修订版本号，则直接返回
        if os.path.isfile(target) or revision is None:
            return
        else:
            def generate_svn_version_py():
                # 如果目标文件不存在，则创建
                if not os.path.isfile(target):
                    version = str(revision)
                    self.info('Creating %s (version=%r)' % (target, version))
                    # 写入修订版本号到目标文件
                    with open(target, 'w') as f:
                        f.write('version = %r\n' % (version))

                # 定义一个函数，用于删除目标文件及其编译后的版本
                def rm_file(f=target, p=self.info):
                    if delete:
                        try:
                            os.remove(f)
                            p('removed ' + f)
                        except OSError:
                            pass
                        try:
                            os.remove(f + 'c')
                            p('removed ' + f + 'c')
                        except OSError:
                            pass

                # 在程序退出时注册删除函数
                atexit.register(rm_file)

                return target

            # 将生成 __svn_version__.py 的函数添加到 data_files 列表中
            self.add_data_files(('', generate_svn_version_py()))
    # 为当前类定义一个方法，用于生成 __hg_version__.py 文件并添加到 data_files 列表中
    def make_hg_version_py(self, delete=True):
        """Appends a data function to the data_files list that will generate
        __hg_version__.py file to the current package directory.

        Generate package __hg_version__.py file from Mercurial revision,
        it will be removed after python exits but will be available
        when sdist, etc commands are executed.

        Notes
        -----
        If __hg_version__.py existed before, nothing is done.

        This is intended for working with source directories that are
        in an Mercurial repository.
        """
        # 指定目标文件路径为当前包目录下的 __hg_version__.py
        target = njoin(self.local_path, '__hg_version__.py')
        # 获取 Mercurial 版本信息
        revision = self._get_hg_revision(self.local_path)
        # 如果目标文件已存在或者无法获取到版本信息，则直接返回
        if os.path.isfile(target) or revision is None:
            return
        else:
            # 定义生成 __hg_version__.py 文件的函数
            def generate_hg_version_py():
                # 如果目标文件不存在，则创建文件并写入版本信息
                if not os.path.isfile(target):
                    version = str(revision)
                    self.info('Creating %s (version=%r)' % (target, version))
                    with open(target, 'w') as f:
                        f.write('version = %r\n' % (version))
                
                # 定义删除文件的函数
                def rm_file(f=target, p=self.info):
                    # 如果 delete 标志为 True，则尝试删除目标文件和其对应的编译文件
                    if delete:
                        try: 
                            os.remove(f)
                            p('removed ' + f)
                        except OSError:
                            pass
                        try:
                            os.remove(f + 'c')
                            p('removed ' + f + 'c')
                        except OSError:
                            pass
                
                # 注册在程序退出时执行删除文件操作
                atexit.register(rm_file)

                return target
            
            # 将生成 __hg_version__.py 文件的函数添加到数据文件列表中
            self.add_data_files(('', generate_hg_version_py()))

    # 为当前类定义一个方法，用于生成 __config__.py 文件并添加到 py_modules 列表中
    def make_config_py(self, name='__config__'):
        """Generate package __config__.py file containing system_info
        information used during building the package.

        This file is installed to the
        package installation directory.
        """
        # 将生成 __config__.py 文件的函数添加到 py_modules 列表中
        self.py_modules.append((self.name, name, generate_config_py))

    # 为当前类定义一个方法，用于获取多个资源信息并返回一个包含所有信息的字典
    def get_info(self, *names):
        """Get resources information.

        Return information (from system_info.get_info) for all of the names in
        the argument list in a single dictionary.
        """
        # 导入必要的函数
        from .system_info import get_info, dict_append
        # 初始化空字典用于存储信息
        info_dict = {}
        # 遍历参数中的每个名称，调用 get_info 函数获取信息并添加到 info_dict 中
        for a in names:
            dict_append(info_dict, **get_info(a))
        # 返回包含所有信息的字典
        return info_dict
# 根据命令名获取命令对象，使用缓存加速查找
def get_cmd(cmdname, _cache={}):
    if cmdname not in _cache:
        # 导入distutils核心模块
        import distutils.core
        # 获取distutils的设置分发对象
        dist = distutils.core._setup_distribution
        # 如果设置分发对象为None，则抛出内部错误异常
        if dist is None:
            from distutils.errors import DistutilsInternalError
            raise DistutilsInternalError(
                  'setup distribution instance not initialized')
        # 获取指定命令的命令对象
        cmd = dist.get_command_obj(cmdname)
        # 将命令对象存入缓存
        _cache[cmdname] = cmd
    # 返回命令对象
    return _cache[cmdname]

# 获取numpy包含目录
def get_numpy_include_dirs():
    # 复制numpy_include_dirs列表内容
    include_dirs = Configuration.numpy_include_dirs[:]
    # 如果列表为空，则导入numpy模块并获取其包含目录
    if not include_dirs:
        import numpy
        include_dirs = [ numpy.get_include() ]
    # 返回包含目录列表
    return include_dirs

# 获取npy-pkg-config目录路径
def get_npy_pkg_dir():
    """Return the path where to find the npy-pkg-config directory.

    If the NPY_PKG_CONFIG_PATH environment variable is set, the value of that
    is returned.  Otherwise, a path inside the location of the numpy module is
    returned.

    The NPY_PKG_CONFIG_PATH can be useful when cross-compiling, maintaining
    customized npy-pkg-config .ini files for the cross-compilation
    environment, and using them when cross-compiling.

    """
    # 获取环境变量NPY_PKG_CONFIG_PATH的值
    d = os.environ.get('NPY_PKG_CONFIG_PATH')
    # 如果环境变量不为None，则返回其值
    if d is not None:
        return d
    # 否则，查找numpy模块的位置并构建npy-pkg-config目录路径
    spec = importlib.util.find_spec('numpy')
    d = os.path.join(os.path.dirname(spec.origin),
            '_core', 'lib', 'npy-pkg-config')
    # 返回npy-pkg-config目录路径
    return d

# 获取指定包名的库信息
def get_pkg_info(pkgname, dirs=None):
    """
    Return library info for the given package.

    Parameters
    ----------
    pkgname : str
        Name of the package (should match the name of the .ini file, without
        the extension, e.g. foo for the file foo.ini).
    dirs : sequence, optional
        If given, should be a sequence of additional directories where to look
        for npy-pkg-config files. Those directories are searched prior to the
        NumPy directory.

    Returns
    -------
    pkginfo : class instance
        The `LibraryInfo` instance containing the build information.

    Raises
    ------
    PkgNotFound
        If the package is not found.

    See Also
    --------
    Configuration.add_npy_pkg_config, Configuration.add_installed_library,
    get_info

    """
    # 导入numpy包配置模块中的read_config函数
    from numpy.distutils.npy_pkg_config import read_config

    # 如果给定了dirs参数，则在其后追加npy-pkg-config目录路径
    if dirs:
        dirs.append(get_npy_pkg_dir())
    else:
        # 否则，设置dirs为包含npy-pkg-config目录路径的列表
        dirs = [get_npy_pkg_dir()]
    # 调用read_config函数读取指定包名的配置信息，并返回
    return read_config(pkgname, dirs)

# 获取指定C库的信息字典
def get_info(pkgname, dirs=None):
    """
    Return an info dict for a given C library.

    The info dict contains the necessary options to use the C library.

    Parameters
    ----------
    pkgname : str
        Name of the package (should match the name of the .ini file, without
        the extension, e.g. foo for the file foo.ini).

    """
    # 这个函数的实现与注释部分完全相符，无需添加额外的代码注释
    pass
    # dirs: 可选的序列，如果提供，则应为寻找 npy-pkg-config 文件的附加目录序列。在 NumPy 目录之前搜索这些目录。
    # 返回值: 包含构建信息的字典。

    # 如果找不到包，则引发 PkgNotFound 异常。

    # 参见: Configuration.add_npy_pkg_config, Configuration.add_installed_library, get_pkg_info

    # 示例:
    # 要获取来自 NumPy 的 npymath 库的必要信息:
    # >>> npymath_info = np.distutils.misc_util.get_info('npymath')
    # >>> npymath_info                                    # doctest: +SKIP
    # {'define_macros': [], 'libraries': ['npymath'], 'library_dirs': ['.../numpy/_core/lib'], 'include_dirs': ['.../numpy/_core/include']}
    # 然后可以将这个 info 字典作为 `Configuration` 实例的输入:
    # config.add_extension('foo', sources=['foo.c'], extra_info=npymath_info)

    """
    # 从 numpy.distutils.npy_pkg_config 导入 parse_flags 函数
    from numpy.distutils.npy_pkg_config import parse_flags
    # 使用 get_pkg_info 函数获取指定包的信息，并将结果存储在 pkg_info 中
    pkg_info = get_pkg_info(pkgname, dirs)

    # 将 LibraryInfo 实例解析为 build_info 字典
    info = parse_flags(pkg_info.cflags())
    # 遍历解析 pkg_info.libs() 的结果，将其项逐一添加到 info 字典的对应键中
    for k, v in parse_flags(pkg_info.libs()).items():
        info[k].extend(v)

    # add_extension 函数的 extra_info 参数是 ANAL
    # 将 info 字典中的 'macros' 键重命名为 'define_macros'，并删除 'macros' 和 'ignored' 键
    info['define_macros'] = info['macros']
    del info['macros']
    del info['ignored']

    # 返回构建信息字典
    return info
def is_bootstrapping():
    # 导入内置模块 builtins
    import builtins

    try:
        # 检查 builtins 中是否定义了 __NUMPY_SETUP__ 属性
        builtins.__NUMPY_SETUP__
        # 如果存在该属性，则返回 True，表示正在引导设置
        return True
    except AttributeError:
        # 如果不存在该属性，则返回 False，表示不是引导设置状态
        return False


#########################

def default_config_dict(name = None, parent_name = None, local_path=None):
    """Return a configuration dictionary for usage in
    configuration() function defined in file setup_<name>.py.
    """
    # 导入警告模块
    import warnings
    # 发出警告，提醒使用新的配置方式替代过时的函数
    warnings.warn('Use Configuration(%r,%r,top_path=%r) instead of '\
                  'deprecated default_config_dict(%r,%r,%r)'
                  % (name, parent_name, local_path,
                     name, parent_name, local_path,
                     ), stacklevel=2)
    # 创建 Configuration 对象 c，并返回其字典表示
    c = Configuration(name, parent_name, local_path)
    return c.todict()


def dict_append(d, **kws):
    # 遍历关键字参数的字典 kws
    for k, v in kws.items():
        # 如果字典 d 中已存在键 k
        if k in d:
            # 获取原来的值 ov
            ov = d[k]
            # 如果原来的值 ov 是字符串类型，则直接替换为新值 v
            if isinstance(ov, str):
                d[k] = v
            else:
                # 如果原来的值 ov 是列表类型，则扩展新值 v
                d[k].extend(v)
        else:
            # 如果字典 d 中不存在键 k，则直接赋值新值 v
            d[k] = v

def appendpath(prefix, path):
    # 如果操作系统路径分隔符不是 '/'，则替换为当前系统的路径分隔符
    if os.path.sep != '/':
        prefix = prefix.replace('/', os.path.sep)
        path = path.replace('/', os.path.sep)
    # 初始化驱动器为空字符串
    drive = ''
    # 如果路径是绝对路径
    if os.path.isabs(path):
        # 获取前缀的驱动器部分
        drive = os.path.splitdrive(prefix)[0]
        # 获取前缀的绝对路径部分
        absprefix = os.path.splitdrive(os.path.abspath(prefix))[1]
        # 获取路径的驱动器部分和路径部分
        pathdrive, path = os.path.splitdrive(path)
        # 获取前缀绝对路径和路径的最长公共前缀
        d = os.path.commonprefix([absprefix, path])
        # 如果拼接后的前缀不等于原前缀或者拼接后的路径不等于原路径，则处理无效路径
        if os.path.join(absprefix[:len(d)], absprefix[len(d):]) != absprefix \
           or os.path.join(path[:len(d)], path[len(d):]) != path:
            # 获取无效路径的父目录
            d = os.path.dirname(d)
        # 获取子路径
        subpath = path[len(d):]
        # 如果子路径是绝对路径，则去掉开头的斜杠
        if os.path.isabs(subpath):
            subpath = subpath[1:]
    else:
        # 如果路径不是绝对路径，则直接作为子路径
        subpath = path
    # 返回规范化的路径
    return os.path.normpath(njoin(drive + prefix, subpath))

def generate_config_py(target):
    """Generate config.py file containing system_info information
    used during building the package.

    Usage:
        config['py_modules'].append((packagename, '__config__',generate_config_py))
    """
    # 导入需要的模块
    from numpy.distutils.system_info import system_info
    from distutils.dir_util import mkpath
    # 创建目标路径的父目录
    mkpath(os.path.dirname(target))
    # 返回目标路径
    return target

def msvc_version(compiler):
    """Return version major and minor of compiler instance if it is
    MSVC, raise an exception otherwise."""
    # 如果编译器不是 MSVC，则抛出异常
    if not compiler.compiler_type == "msvc":
        raise ValueError("Compiler instance is not msvc (%s)"\
                         % compiler.compiler_type)
    # 返回 MSVC 编译器的主要版本和次要版本
    return compiler._MSVCCompiler__version

def get_build_architecture():
    # 在非 Windows 系统上导入 distutils.msvccompiler 会触发警告，因此延迟导入到此处
    from distutils.msvccompiler import get_build_architecture
    # 返回构建的架构信息
    return get_build_architecture()

_cxx_ignore_flags = {'-Werror=implicit-function-declaration', '-std=c99'}


def sanitize_cxx_flags(cxxflags):
    '''
    Some flags are valid for C but not C++. Prune them.
    '''
    # 移除对 C++ 无效的标志
    # 暂无具体实现
    # 返回一个列表，其中包含在 cxxflags 中但不在 _cxx_ignore_flags 中的所有标志
    return [flag for flag in cxxflags if flag not in _cxx_ignore_flags]
# 使用 importlib 工具来从指定位置的文件中导入模块 `modname`
def exec_mod_from_location(modname, modfile):
    # 根据文件路径和模块名创建一个模块规范对象
    spec = importlib.util.spec_from_file_location(modname, modfile)
    # 根据模块规范对象创建一个新的模块对象
    foo = importlib.util.module_from_spec(spec)
    # 使用加载器执行模块的代码，将其加入到系统模块列表中
    spec.loader.exec_module(foo)
    # 返回执行后的模块对象
    return foo
```