# `.\numpy\numpy\distutils\system_info.py`

```
#!/usr/bin/env python3
"""
This file defines a set of system_info classes for getting
information about various resources (libraries, library directories,
include directories, etc.) in the system. Usage:
    info_dict = get_info(<name>)
  where <name> is a string 'atlas','x11','fftw','lapack','blas',
  'lapack_src', 'blas_src', etc. For a complete list of allowed names,
  see the definition of get_info() function below.

  Returned info_dict is a dictionary which is compatible with
  distutils.setup keyword arguments. If info_dict == {}, then the
  asked resource is not available (system_info could not find it).

  Several *_info classes specify an environment variable to specify
  the locations of software. When setting the corresponding environment
  variable to 'None' then the software will be ignored, even when it
  is available in system.

Global parameters:
  system_info.search_static_first - search static libraries (.a)
             in precedence to shared ones (.so, .sl) if enabled.
  system_info.verbosity - output the results to stdout if enabled.

The file 'site.cfg' is looked for in

1) Directory of main setup.py file being run.
2) Home directory of user running the setup.py file as ~/.numpy-site.cfg
3) System wide directory (location of this file...)

The first one found is used to get system configuration options The
format is that used by ConfigParser (i.e., Windows .INI style). The
section ALL is not intended for general use.

Appropriate defaults are used if nothing is specified.

The order of finding the locations of resources is the following:
 1. environment variable
 2. section in site.cfg
 3. DEFAULT section in site.cfg
 4. System default search paths (see ``default_*`` variables below).
Only the first complete match is returned.

Currently, the following classes are available, along with their section names:

    Numeric_info:Numeric
    _numpy_info:Numeric
    _pkg_config_info:None
    accelerate_info:accelerate
    accelerate_lapack_info:accelerate
    agg2_info:agg2
    amd_info:amd
    atlas_3_10_blas_info:atlas
    atlas_3_10_blas_threads_info:atlas
    atlas_3_10_info:atlas
    atlas_3_10_threads_info:atlas
    atlas_blas_info:atlas
    atlas_blas_threads_info:atlas
    atlas_info:atlas
    atlas_threads_info:atlas
    blas64__opt_info:ALL               # usage recommended (general ILP64 BLAS, 64_ symbol suffix)
    blas_ilp64_opt_info:ALL            # usage recommended (general ILP64 BLAS)
    blas_ilp64_plain_opt_info:ALL      # usage recommended (general ILP64 BLAS, no symbol suffix)
    blas_info:blas
    blas_mkl_info:mkl
    blas_ssl2_info:ssl2
    blas_opt_info:ALL                  # usage recommended
    blas_src_info:blas_src
    blis_info:blis
    boost_python_info:boost_python
    dfftw_info:fftw
    dfftw_threads_info:fftw
    djbfft_info:djbfft
    f2py_info:ALL
    fft_opt_info:ALL
    fftw2_info:fftw
    fftw3_info:fftw3
    fftw_info:fftw
    fftw_threads_info:fftw
    flame_info:flame
"""
    freetype2_info:freetype2                     # 设置变量 freetype2_info 为 'freetype2'
    gdk_2_info:gdk_2                             # 设置变量 gdk_2_info 为 'gdk_2'
    gdk_info:gdk                                 # 设置变量 gdk_info 为 'gdk'
    gdk_pixbuf_2_info:gdk_pixbuf_2               # 设置变量 gdk_pixbuf_2_info 为 'gdk_pixbuf_2'
    gdk_pixbuf_xlib_2_info:gdk_pixbuf_xlib_2     # 设置变量 gdk_pixbuf_xlib_2_info 为 'gdk_pixbuf_xlib_2'
    gdk_x11_2_info:gdk_x11_2                     # 设置变量 gdk_x11_2_info 为 'gdk_x11_2'
    gtkp_2_info:gtkp_2                           # 设置变量 gtkp_2_info 为 'gtkp_2'
    gtkp_x11_2_info:gtkp_x11_2                   # 设置变量 gtkp_x11_2_info 为 'gtkp_x11_2'
    lapack64__opt_info:ALL                       # 设置变量 lapack64__opt_info 为 'ALL'，推荐使用通用 ILP64 LAPACK，符号后缀为 64_
    lapack_atlas_3_10_info:atlas                 # 设置变量 lapack_atlas_3_10_info 为 'atlas'
    lapack_atlas_3_10_threads_info:atlas         # 设置变量 lapack_atlas_3_10_threads_info 为 'atlas'
    lapack_atlas_info:atlas                      # 设置变量 lapack_atlas_info 为 'atlas'
    lapack_atlas_threads_info:atlas              # 设置变量 lapack_atlas_threads_info 为 'atlas'
    lapack_ilp64_opt_info:ALL                    # 设置变量 lapack_ilp64_opt_info 为 'ALL'，推荐使用通用 ILP64 LAPACK
    lapack_ilp64_plain_opt_info:ALL              # 设置变量 lapack_ilp64_plain_opt_info 为 'ALL'，推荐使用通用 ILP64 LAPACK，无符号后缀
    lapack_info:lapack                           # 设置变量 lapack_info 为 'lapack'
    lapack_mkl_info:mkl                          # 设置变量 lapack_mkl_info 为 'mkl'
    lapack_ssl2_info:ssl2                        # 设置变量 lapack_ssl2_info 为 'ssl2'
    lapack_opt_info:ALL                          # 设置变量 lapack_opt_info 为 'ALL'，推荐使用
    lapack_src_info:lapack_src                   # 设置变量 lapack_src_info 为 'lapack_src'
    mkl_info:mkl                                 # 设置变量 mkl_info 为 'mkl'
    ssl2_info:ssl2                               # 设置变量 ssl2_info 为 'ssl2'
    numarray_info:numarray                       # 设置变量 numarray_info 为 'numarray'
    numerix_info:numerix                         # 设置变量 numerix_info 为 'numerix'
    numpy_info:numpy                             # 设置变量 numpy_info 为 'numpy'
    openblas64__info:openblas64_                 # 设置变量 openblas64__info 为 'openblas64_'
    openblas64__lapack_info:openblas64_          # 设置变量 openblas64__lapack_info 为 'openblas64_'
    openblas_clapack_info:openblas               # 设置变量 openblas_clapack_info 为 'openblas'
    openblas_ilp64_info:openblas_ilp64           # 设置变量 openblas_ilp64_info 为 'openblas_ilp64'
    openblas_ilp64_lapack_info:openblas_ilp64    # 设置变量 openblas_ilp64_lapack_info 为 'openblas_ilp64'
    openblas_info:openblas                       # 设置变量 openblas_info 为 'openblas'
    openblas_lapack_info:openblas                # 设置变量 openblas_lapack_info 为 'openblas'
    sfftw_info:fftw                               # 设置变量 sfftw_info 为 'fftw'
    sfftw_threads_info:fftw                       # 设置变量 sfftw_threads_info 为 'fftw'
    system_info:ALL                              # 设置变量 system_info 为 'ALL'
    umfpack_info:umfpack                         # 设置变量 umfpack_info 为 'umfpack'
    wx_info:wx                                   # 设置变量 wx_info 为 'wx'
    x11_info:x11                                 # 设置变量 x11_info 为 'x11'
    xft_info:xft                                 # 设置变量 xft_info 为 'xft'
# 导入所需的模块和库
import sys  # 系统相关的功能
import os  # 操作系统相关的功能
import re  # 正则表达式的支持
import copy  # 复制对象的支持
import warnings  # 警告处理
import subprocess  # 启动子进程
import textwrap  # 文本格式化工具

# 导入glob模块中的glob函数，用于文件路径的模式匹配
from glob import glob

# 导入functools模块中的reduce函数，用于序列的归约操作
from functools import reduce

# 导入ConfigParser类，用于解析配置文件
from configparser import NoOptionError, RawConfigParser as ConfigParser

# 导入DistutilsError异常和Distribution类
from distutils.errors import DistutilsError
from distutils.dist import Distribution

# 导入sysconfig模块，用于Python的配置信息
import sysconfig

# 导入numpy.distutils中的log模块
from numpy.distutils import log

# 导入distutils.util模块中的get_platform函数
from distutils.util import get_platform

# 导入numpy.distutils.exec_command模块中的相关函数
from numpy.distutils.exec_command import (
    find_executable, filepath_from_subprocess_output,
)

# 导入numpy.distutils.misc_util模块中的函数和常量
from numpy.distutils.misc_util import (is_sequence, is_string,
                                       get_shared_lib_extension)

# 导入numpy.distutils.command.config模块中的config类
from numpy.distutils.command.config import config as cmd_config

# 导入numpy.distutils中的自定义C编译器
from numpy.distutils import customized_ccompiler as _customized_ccompiler

# 导入numpy.distutils中的_shell_utils模块
from numpy.distutils import _shell_utils

# 导入distutils.ccompiler模块
import distutils.ccompiler

# 导入tempfile模块，用于创建临时文件和目录
import tempfile

# 导入shutil模块，用于高级文件操作
import shutil

# 定义__all__列表，指定模块中公开的所有符号
__all__ = ['system_info']
# 导入 platform 模块，确定操作系统位数
import platform
# 以字典形式定义不同位数对应的值
_bits = {'32bit': 32, '64bit': 64}
# 获取当前操作系统的位数，存储到 platform_bits 变量中
platform_bits = _bits[platform.architecture()[0]]

# 定义全局变量 global_compiler，默认为 None
global_compiler = None

# 定义函数 customized_ccompiler
def customized_ccompiler():
    # 使用 global_compiler 全局变量，如果为 None 则调用 _customized_ccompiler() 函数赋值给 global_compiler
    global global_compiler
    if not global_compiler:
        global_compiler = _customized_ccompiler()
    return global_compiler

# 定义函数 _c_string_literal，将 Python 字符串转换为适用于 C 代码的字面量
def _c_string_literal(s):
    """
    Convert a python string into a literal suitable for inclusion into C code
    """
    # 将字符串中的反斜杠转义为双反斜杠，将双引号转义为双引号，将换行符转义为反斜杠加 n
    s = s.replace('\\', r'\\')
    s = s.replace('"',  r'\"')
    s = s.replace('\n', r'\n')
    return '"{}"'.format(s)

# 定义函数 libpaths，根据系统位数返回有效的库路径列表
def libpaths(paths, bits):
    """Return a list of library paths valid on 32 or 64 bit systems.

    Inputs:
      paths : sequence
        A sequence of strings (typically paths)
      bits : int
        An integer, the only valid values are 32 or 64.  A ValueError exception
      is raised otherwise.

    Examples:

    Consider a list of directories
    >>> paths = ['/usr/X11R6/lib','/usr/X11/lib','/usr/lib']

    For a 32-bit platform, this is already valid:
    >>> np.distutils.system_info.libpaths(paths,32)
    ['/usr/X11R6/lib', '/usr/X11/lib', '/usr/lib']

    On 64 bits, we prepend the '64' postfix
    >>> np.distutils.system_info.libpaths(paths,64)
    ['/usr/X11R6/lib64', '/usr/X11R6/lib', '/usr/X11/lib64', '/usr/X11/lib',
    '/usr/lib64', '/usr/lib']
    """
    # 如果 bits 不是 32 或 64，则抛出 ValueError 异常
    if bits not in (32, 64):
        raise ValueError("Invalid bit size in libpaths: 32 or 64 only")

    # 处理 32 位的情况
    if bits == 32:
        return paths

    # 处理 64 位的情况
    out = []
    for p in paths:
        out.extend([p + '64', p])

    return out

# 如果系统是 win32
if sys.platform == 'win32':
    # 定义默认的库目录
    default_lib_dirs = ['C:\\',
                        os.path.join(sysconfig.get_config_var('exec_prefix'),
                                     'libs')]
    default_runtime_dirs = [] # 定义默认的运行时目录
    default_include_dirs = [] # 定义默认的包含目录
    default_src_dirs = ['.']  # 定义默认的源码目录
    default_x11_lib_dirs = [] # 定义默认的 X11 库目录
    default_x11_include_dirs = [] # 定义默认的 X11 包含目录
    _include_dirs = [  # 定义包含目录列表
        'include',
        'include/suitesparse',
    ]
    _lib_dirs = [  # 定义库目录列表
        'lib',
    ]

    # 将包含目录中的斜杠替换为系统对应的分隔符
    _include_dirs = [d.replace('/', os.sep) for d in _include_dirs]
    _lib_dirs = [d.replace('/', os.sep) for d in _lib_dirs]
    
    # 定义函数 add_system_root，添加一个包管理器根目录到包含目录中
    def add_system_root(library_root):
        """Add a package manager root to the include directories"""
        # 使用全局变量 default_lib_dirs 和 default_include_dirs
        global default_lib_dirs
        global default_include_dirs
        
        # 对 library_root 变量进行规范化处理
        library_root = os.path.normpath(library_root)

        # 将 library_root 和包含目录列表中的元素进行组合，并添加到默认的库目录中
        default_lib_dirs.extend(
            os.path.join(library_root, d) for d in _lib_dirs)
        # 将 library_root 和包含目录列表中的元素进行组合，并添加到默认的包含目录中
        default_include_dirs.extend(
            os.path.join(library_root, d) for d in _include_dirs)

    # VCpkg 是 Windows 上用于 C/C++ 库的事实标准包管理器。如果它在环境变量 PATH 中，则将其路径追加到这里
    vcpkg = shutil.which('vcpkg')
    # 如果存在 vcpkg，则设置 vcpkg_dir 为 vcpkg 路径的上一级目录
    if vcpkg:
        vcpkg_dir = os.path.dirname(vcpkg)
        # 如果操作系统架构为 32 位，则 specifier 设置为 'x86'，否则设置为 'x64'
        if platform.architecture()[0] == '32bit':
            specifier = 'x86'
        else:
            specifier = 'x64'

        # 设置 vcpkg_installed 为 vcpkg_dir 下的 'installed' 目录
        vcpkg_installed = os.path.join(vcpkg_dir, 'installed')
        # 遍历指定的 vcpkg_root 路径，添加到系统路径中
        for vcpkg_root in [
            os.path.join(vcpkg_installed, specifier + '-windows'),
            os.path.join(vcpkg_installed, specifier + '-windows-static'),
        ]:
            add_system_root(vcpkg_root)

    # 判断是否存在 conda，若存在则设置 conda_dir 为 conda 路径的上一级目录
    conda = shutil.which('conda')
    if conda:
        conda_dir = os.path.dirname(conda)
        # 向系统路径中添加 conda_dir 下的 '..\Library' 和 'Library' 路径
        add_system_root(os.path.join(conda_dir, '..', 'Library'))
        add_system_root(os.path.join(conda_dir, 'Library'))
else:
    # 设置默认的库目录列表
    default_lib_dirs = libpaths(['/usr/local/lib', '/opt/lib', '/usr/lib',
                                 '/opt/local/lib', '/sw/lib'], platform_bits)
    # 空的运行时目录列表
    default_runtime_dirs = []
    # 设置默认的包含目录列表
    default_include_dirs = ['/usr/local/include',
                            '/opt/include',
                            # 在macports下umfpack的路径
                            '/opt/local/include/ufsparse',
                            '/opt/local/include', '/sw/include',
                            '/usr/include/suitesparse']
    # 设置默认的源码目录列表
    default_src_dirs = ['.', '/usr/local/src', '/opt/src', '/sw/src']

    # 设置默认的X11库目录列表
    default_x11_lib_dirs = libpaths(['/usr/X11R6/lib', '/usr/X11/lib',
                                     '/usr/lib'], platform_bits)
    # 设置默认的X11包含目录列表
    default_x11_include_dirs = ['/usr/X11R6/include', '/usr/X11/include']

    # 如果存在'/usr/lib/X11'目录
    if os.path.exists('/usr/lib/X11'):
        # 获取'libX11.so'文件的路径
        globbed_x11_dir = glob('/usr/lib/*/libX11.so')
        if globbed_x11_dir:
            x11_so_dir = os.path.split(globbed_x11_dir[0])[0]
            # 添加X11库的路径
            default_x11_lib_dirs.extend([x11_so_dir, '/usr/lib/X11'])
            # 添加X11包含目录的路径
            default_x11_include_dirs.extend(['/usr/lib/X11/include',
                                             '/usr/include/X11'])

    # 打开临时文件
    with open(os.devnull, 'w') as tmp:
        # 尝试执行命令行
        try:
            p = subprocess.Popen(["gcc", "-print-multiarch"], stdout=subprocess.PIPE,
                         stderr=tmp)
        except (OSError, DistutilsError):
            # 如果出现OSError，表示gcc未安装，或者出现SandboxViolation (DistutilsError的子类)错误
            pass
        else:
            triplet = str(p.communicate()[0].decode().strip())
            if p.returncode == 0:
                # gcc支持"-print-multiarch"选项，添加相关的库路径
                default_x11_lib_dirs += [os.path.join("/usr/lib/", triplet)]
                default_lib_dirs += [os.path.join("/usr/lib/", triplet)]


# 如果sys.prefix的lib目录不在默认的库目录列表中
if os.path.join(sys.prefix, 'lib') not in default_lib_dirs:
    # 将sys.prefix的lib目录插入到默认的库目录列表的最前面
    default_lib_dirs.insert(0, os.path.join(sys.prefix, 'lib'))
    # 将sys.prefix的include目录添加到默认的包含目录列表
    default_include_dirs.append(os.path.join(sys.prefix, 'include'))
    # 将sys.prefix的src目录添加到默认的源码目录列表
    default_src_dirs.append(os.path.join(sys.prefix, 'src'))

# 过滤默认的库目录，保留存在的目录
default_lib_dirs = [_m for _m in default_lib_dirs if os.path.isdir(_m)]
# 过滤默认的运行时目录，保留存在的目录
default_runtime_dirs = [_m for _m in default_runtime_dirs if os.path.isdir(_m)]
# 过滤默认的包含目录，保留存在的目录
default_include_dirs = [_m for _m in default_include_dirs if os.path.isdir(_m)]
# 过滤默认的源码目录，保留存在的目录
default_src_dirs = [_m for _m in default_src_dirs if os.path.isdir(_m)]

# 获取共享库的扩展名
so_ext = get_shared_lib_extension()


def get_standard_file(fname):
    """返回命名为'fname'的文件列表，顺序为：
    1) 系统全局目录（本模块的目录位置）
    2) 用户的HOME目录（os.environ['HOME']）
    3) 本地目录
    """
    # 系统全局文件
    filenames = []
    try:
        f = __file__
    except NameError:
        f = sys.argv[0]
    sysfile = os.path.join(os.path.split(os.path.abspath(f))[0],
                           fname)
    # 如果给定的文件路径是一个文件，将其添加到文件名列表中
    if os.path.isfile(sysfile):
        filenames.append(sysfile)
    
    # 尝试获取用户的主目录，并查找用户配置文件
    try:
        # 获取用户的主目录路径
        f = os.path.expanduser('~')
    except KeyError:
        # 忽略 Key Error 异常
        pass
    else:
        # 构建用户配置文件的完整路径
        user_file = os.path.join(f, fname)
        # 如果用户配置文件存在，将其添加到文件名列表中
        if os.path.isfile(user_file):
            filenames.append(user_file)
    
    # 如果本地文件存在，将其绝对路径添加到文件名列表中
    if os.path.isfile(fname):
        filenames.append(os.path.abspath(fname))
    
    # 返回所有文件名的列表
    return filenames
# 解析环境变量 `env`，根据 `,` 分割并仅返回在 `base_order` 中的元素

def _parse_env_order(base_order, env):
    """ Parse an environment variable `env` by splitting with "," and only returning elements from `base_order`
    
    This method will sequence the environment variable and check for their
    individual elements in `base_order`.
    
    The items in the environment variable may be negated via '^item' or '!itema,itemb'.
    It must start with ^/! to negate all options.
    
    Raises
    ------
    ValueError: for mixed negated and non-negated orders or multiple negated orders
    
    Parameters
    ----------
    base_order : list of str
       the base list of orders
    env : str
       the environment variable to be parsed, if none is found, `base_order` is returned
    
    Returns
    -------
    allow_order : list of str
        allowed orders in lower-case
    unknown_order : list of str
        for values not overlapping with `base_order`
    """
    # 获取环境变量 `env` 的值
    order_str = os.environ.get(env, None)
    
    # 确保所有基本订单都是小写（便于比较）
    base_order = [order.lower() for order in base_order]
    
    # 如果环境变量值为 None，则返回基本订单和空列表
    if order_str is None:
        return base_order, []
    
    # 判断是否为否定形式（以 '^' 或 '!' 开头）
    neg = order_str.startswith('^') or order_str.startswith('!')
    
    # 检查格式
    order_str_l = list(order_str)
    sum_neg = order_str_l.count('^') + order_str_l.count('!')
    if neg:
        if sum_neg > 1:
            raise ValueError(f"Environment variable '{env}' may only contain a single (prefixed) negation: {order_str}")
        # 去除前缀
        order_str = order_str[1:]
    elif sum_neg > 0:
        raise ValueError(f"Environment variable '{env}' may not mix negated an non-negated items: {order_str}")
    
    # 分割并转换为小写
    orders = order_str.lower().split(',')
    
    # 用于存放不重叠的元素
    unknown_order = []
    
    # 如果是否定形式，需要从基本订单中移除
    if neg:
        allow_order = base_order.copy()
        
        for order in orders:
            if not order:
                continue
            
            if order not in base_order:
                unknown_order.append(order)
                continue
            
            if order in allow_order:
                allow_order.remove(order)
    
    else:
        allow_order = []
        
        for order in orders:
            if not order:
                continue
            
            if order not in base_order:
                unknown_order.append(order)
                continue
            
            if order not in allow_order:
                allow_order.append(order)
    
    return allow_order, unknown_order
    In section '{section}' we found multiple appearances of options {options}.
# 定义 AtlasNotFoundError 类，继承自 NotFoundError
class AtlasNotFoundError(NotFoundError):
    """
    Atlas (http://github.com/math-atlas/math-atlas) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [atlas]) or by setting
    the ATLAS environment variable.
    """


# 定义 FlameNotFoundError 类，继承自 NotFoundError
class FlameNotFoundError(NotFoundError):
    """
    FLAME (http://www.cs.utexas.edu/~flame/web/) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [flame]).
    """


# 定义 LapackNotFoundError 类，继承自 NotFoundError
class LapackNotFoundError(NotFoundError):
    """
    Lapack (http://www.netlib.org/lapack/) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [lapack]) or by setting
    the LAPACK environment variable.
    """


# 定义 LapackSrcNotFoundError 类，继承自 LapackNotFoundError
class LapackSrcNotFoundError(LapackNotFoundError):
    """
    Lapack (http://www.netlib.org/lapack/) sources not found.
    Directories to search for the sources can be specified in the
    numpy/distutils/site.cfg file (section [lapack_src]) or by setting
    the LAPACK_SRC environment variable.
    """


# 定义 LapackILP64NotFoundError 类，继承自 NotFoundError
class LapackILP64NotFoundError(NotFoundError):
    """
    64-bit Lapack libraries not found.
    Known libraries in numpy/distutils/site.cfg file are:
    openblas64_, openblas_ilp64
    """


# 定义 BlasOptNotFoundError 类，继承自 NotFoundError
class BlasOptNotFoundError(NotFoundError):
    """
    Optimized (vendor) Blas libraries are not found.
    Falls back to netlib Blas library which has worse performance.
    A better performance should be easily gained by switching
    Blas library.
    """


# 定义 BlasNotFoundError 类，继承自 NotFoundError
class BlasNotFoundError(NotFoundError):
    """
    Blas (http://www.netlib.org/blas/) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [blas]) or by setting
    the BLAS environment variable.
    """


# 定义 BlasILP64NotFoundError 类，继承自 NotFoundError
class BlasILP64NotFoundError(NotFoundError):
    """
    64-bit Blas libraries not found.
    Known libraries in numpy/distutils/site.cfg file are:
    openblas64_, openblas_ilp64
    """


# 定义 BlasSrcNotFoundError 类，继承自 BlasNotFoundError
class BlasSrcNotFoundError(BlasNotFoundError):
    """
    Blas (http://www.netlib.org/blas/) sources not found.
    Directories to search for the sources can be specified in the
    numpy/distutils/site.cfg file (section [blas_src]) or by setting
    the BLAS_SRC environment variable.
    """


# 定义 FFTWNotFoundError 类，继承自 NotFoundError
class FFTWNotFoundError(NotFoundError):
    """
    FFTW (http://www.fftw.org/) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [fftw]) or by setting
    the FFTW environment variable.
    """


# 定义 DJBFFTNotFoundError 类，继承自 NotFoundError
class DJBFFTNotFoundError(NotFoundError):
    """
    DJBFFT (https://cr.yp.to/djbfft.html) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [djbfft]) or by setting
    the DJBFFT environment variable.
    """


# 定义 NumericNotFoundError 类，继承自 NotFoundError
class NumericNotFoundError(NotFoundError):
    """
    Placeholder for additional specific libraries not found.
    """
    # 如果找不到 Numeric 模块，则需要从 https://www.numpy.org/ 获取并安装，然后重试运行 setup.py 文件。
# 定义X11NotFoundError类，用于表示X11库未找到的错误
class X11NotFoundError(NotFoundError):
    """X11 libraries not found."""


# 定义UmfpackNotFoundError类，用于表示未找到UMFPACK稀疏求解器的错误
class UmfpackNotFoundError(NotFoundError):
    """
    UMFPACK sparse solver (https://www.cise.ufl.edu/research/sparse/umfpack/)
    not found. Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [umfpack]) or by setting
    the UMFPACK environment variable."""


# 定义system_info类
class system_info:

    """ get_info() is the only public method. Don't use others.
    """
    dir_env_var = None
    # XXX: search_static_first is disabled by default, may disappear in
    # future unless it is proved to be useful.
    search_static_first = 0
    # The base-class section name is a random word "ALL" and is not really
    # intended for general use. It cannot be None nor can it be DEFAULT as
    # these break the ConfigParser. See gh-15338
    section = 'ALL'
    saved_results = {}

    notfounderror = NotFoundError

    # 初始化方法
    def __init__(self,
                  default_lib_dirs=default_lib_dirs,
                  default_include_dirs=default_include_dirs,
                  ):
        self.__class__.info = {}
        self.local_prefixes = []
        defaults = {'library_dirs': os.pathsep.join(default_lib_dirs),
                    'include_dirs': os.pathsep.join(default_include_dirs),
                    'runtime_library_dirs': os.pathsep.join(default_runtime_dirs),
                    'rpath': '',
                    'src_dirs': os.pathsep.join(default_src_dirs),
                    'search_static_first': str(self.search_static_first),
                    'extra_compile_args': '', 'extra_link_args': ''}
        self.cp = ConfigParser(defaults)
        self.files = []
        self.files.extend(get_standard_file('.numpy-site.cfg'))
        self.files.extend(get_standard_file('site.cfg'))
        self.parse_config_files()

        # 如果section不是None，则尝试从配置文件中获取search_static_first的值
        if self.section is not None:
            self.search_static_first = self.cp.getboolean(
                self.section, 'search_static_first')
        assert isinstance(self.search_static_first, int)

    # 解析配置文件
    def parse_config_files(self):
        # 读取配置文件
        self.cp.read(self.files)
        # 如果配置文件中没有指定的section，则添加该section
        if not self.cp.has_section(self.section):
            if self.section is not None:
                self.cp.add_section(self.section) 
    # 计算并返回当前项目的库信息
    def calc_libraries_info(self):
        # 获取当前项目所使用的所有库
        libs = self.get_libraries()
        # 获取当前项目的库目录
        dirs = self.get_lib_dirs()
        # 获取当前项目的运行时库目录
        r_dirs = self.get_runtime_lib_dirs()
        # 对于内在的 distutils 使用 rpath，我们将两个条目视为一个条目来处理
        r_dirs.extend(self.get_runtime_lib_dirs(key='rpath'))
        # 创建一个空的字典来存储库信息
        info = {}
        # 遍历每个库
        for lib in libs:
            # 检查库是否存在于常规库目录中
            i = self.check_libs(dirs, [lib])
            if i is not None:
                # 将找到的库信息添加到 info 字典中
                dict_append(info, **i)
            else:
                # 记录日志，指出未找到该库并忽略
                log.info('Library %s was not found. Ignoring' % (lib))

            # 如果存在运行时库目录
            if r_dirs:
                # 检查库是否存在于运行时库目录中
                i = self.check_libs(r_dirs, [lib])
                if i is not None:
                    # 将找到的库信息添加到 info 字典中，并将关键字 'libraries' 改为 'runtime_library_dirs'
                    del i['libraries']
                    i['runtime_library_dirs'] = i.pop('library_dirs')
                    dict_append(info, **i)
                else:
                    # 记录日志，指出未找到该运行时库并忽略
                    log.info('Runtime library %s was not found. Ignoring' % (lib))

        # 返回计算得到的库信息字典
        return info

    # 将额外的信息添加到给定的 info 字典中
    def set_info(self, **info):
        # 如果传入了额外信息
        if info:
            # 计算并添加当前项目的库信息到 info 中
            lib_info = self.calc_libraries_info()
            dict_append(info, **lib_info)
            # 计算并添加额外的信息到 info 中
            extra_info = self.calc_extra_info()
            dict_append(info, **extra_info)
        # 将结果保存到 saved_results 中
        self.saved_results[self.__class__.__name__] = info

    # 在配置文件的特定部分（section）中查找单个选项
    def get_option_single(self, *options):
        """ Ensure that only one of `options` are found in the section

        Parameters
        ----------
        *options : list of str
           a list of options to be found in the section (``self.section``)

        Returns
        -------
        str :
            the option that is uniquely found in the section

        Raises
        ------
        AliasedOptionError :
            in case more than one of the options are found
        """
        # 检查每个选项是否存在于配置文件的特定部分中
        found = [self.cp.has_option(self.section, opt) for opt in options]
        # 如果恰好有一个选项被找到
        if sum(found) == 1:
            return options[found.index(True)]
        # 如果没有选项被找到，则返回第一个选项
        elif sum(found) == 0:
            return options[0]

        # 否则，如果有多于一个选项被找到，抛出 AliasedOptionError 异常
        if AliasedOptionError.__doc__ is None:
            raise AliasedOptionError()
        raise AliasedOptionError(AliasedOptionError.__doc__.format(
            section=self.section, options='[{}]'.format(', '.join(options))))

    # 检查当前项目是否有保存的信息
    def has_info(self):
        return self.__class__.__name__ in self.saved_results
    def calc_extra_info(self):
        """ 
        Updates the information in the current information with
        respect to these flags:
          extra_compile_args
          extra_link_args
        """
        # 初始化一个空字典用于存储信息
        info = {}
        # 遍历每个需要处理的键
        for key in ['extra_compile_args', 'extra_link_args']:
            # 获取配置文件中对应键的数值
            opt = self.cp.get(self.section, key)
            # 使用 NativeParser 的 split 方法对数值进行分割处理
            opt = _shell_utils.NativeParser.split(opt)
            # 如果分割结果非空，则将其加入到 info 字典中
            if opt:
                tmp = {key: opt}
                dict_append(info, **tmp)
        # 返回更新后的信息字典
        return info

    def get_info(self, notfound_action=0):
        """ 
        Return a dictionary with items that are compatible
        with numpy.distutils.setup keyword arguments.
        """
        # 初始化标志位
        flag = 0
        # 如果当前对象没有信息
        if not self.has_info():
            # 设置标志位为 1
            flag = 1
            # 记录日志：打印类名，并追加冒号
            log.info(self.__class__.__name__ + ':')
            # 如果对象有 calc_info 方法，则调用
            if hasattr(self, 'calc_info'):
                self.calc_info()
            # 如果 notfound_action 不为 0
            if notfound_action:
                # 如果依然没有信息
                if not self.has_info():
                    # 根据 notfound_action 的值选择性地发出警告或者抛出异常
                    if notfound_action == 1:
                        warnings.warn(self.notfounderror.__doc__, stacklevel=2)
                    elif notfound_action == 2:
                        raise self.notfounderror(self.notfounderror.__doc__)
                    else:
                        raise ValueError(repr(notfound_action))

            # 如果依然没有信息，记录日志表示不可用，并设置信息
            if not self.has_info():
                log.info('  NOT AVAILABLE')
                self.set_info()
            else:
                # 如果找到信息，记录日志表示找到信息
                log.info('  FOUND:')

        # 从保存的结果中获取当前类名对应的结果
        res = self.saved_results.get(self.__class__.__name__)
        # 如果日志的记录等级不低于 INFO 并且 flag 为真
        if log.get_threshold() <= log.INFO and flag:
            # 遍历结果字典中的每一对键值对
            for k, v in res.items():
                # 将值转换为字符串格式
                v = str(v)
                # 如果键为 'sources' 或者 'libraries' 并且长度超过 270
                if k in ['sources', 'libraries'] and len(v) > 270:
                    # 截取部分字符串，保留前后各 120 个字符，并用省略号表示中间部分
                    v = v[:120] + '...\n...\n...' + v[-120:]
                # 记录详细信息的日志条目
                log.info('    %s = %s', k, v)
            # 记录空行，用于分隔不同的日志条目
            log.info('')

        # 返回结果字典的深拷贝
        return copy.deepcopy(res)
    # 从配置文件中获取指定section和key对应的路径列表
    def get_paths(self, section, key):
        # 将配置文件中的路径字符串按分隔符切分成列表
        dirs = self.cp.get(section, key).split(os.pathsep)
        # 获取环境变量
        env_var = self.dir_env_var
        # 如果环境变量存在
        if env_var:
            # 如果环境变量是一个序列
            if is_sequence(env_var):
                # 设置默认环境变量值为序列中的最后一个值
                e0 = env_var[-1]
                # 遍历环境变量序列
                for e in env_var:
                    # 如果环境变量存在于系统环境变量中
                    if e in os.environ:
                        # 将默认环境变量值设为当前环境变量，并跳出循环
                        e0 = e
                        break
                # 如果默认环境变量不等于当前环境变量，则记录日志
                if not env_var[0] == e0:
                    log.info('Setting %s=%s' % (env_var[0], e0))
                # 将环境变量设置为当前环境变量
                env_var = e0
        # 如果环境变量存在并且在系统环境变量中
        if env_var and env_var in os.environ:
            # 获取环境变量对应的路径
            d = os.environ[env_var]
            # 如果路径为'None'，记录日志并返回空列表
            if d == 'None':
                log.info('Disabled %s: %s', self.__class__.__name__, '(%s is None)' % (env_var,))
                return []
            # 如果路径是文件
            if os.path.isfile(d):
                # 将文件所在目录添加到列表中
                dirs = [os.path.dirname(d)] + dirs
                # 如果_lib_names长度为1
                l = getattr(self, '_lib_names', [])
                if len(l) == 1:
                    # 获取文件名，如果开头是'lib'，记录日志并替换_lib_names[0]
                    b = os.path.basename(d)
                    b = os.path.splitext(b)[0]
                    if b[:3] == 'lib':
                        log.info('Replacing _lib_names[0]==%r with %r' % (self._lib_names[0], b[3:]))
                        self._lib_names[0] = b[3:]
            else:
                # 如果路径是目录
                ds = d.split(os.pathsep)
                ds2 = []
                for d in ds:
                    # 如果是有效目录
                    if os.path.isdir(d):
                        # 添加当前目录和include、lib目录到列表中
                        ds2.append(d)
                        for dd in ['include', 'lib']:
                            d1 = os.path.join(d, dd)
                            if os.path.isdir(d1):
                                ds2.append(d1)
                # 将有效目录添加到列表中
                dirs = ds2 + dirs
        # 获取默认路径，并添加到列表中
        default_dirs = self.cp.get(self.section, key).split(os.pathsep)
        dirs.extend(default_dirs)
        # 初始化结果列表
        ret = []
        # 遍历目录列表
        for d in dirs:
            # 如果目录不为空且不是有效目录，记录警告信息并继续下一个目录
            if len(d) > 0 and not os.path.isdir(d):
                warnings.warn('Specified path %s is invalid.' % d, stacklevel=2)
                continue
            # 如果目录不在结果列表中，则添加到结果列表中
            if d not in ret:
                ret.append(d)
        # 记录调试信息，返回结果列表
        log.debug('( %s = %s )', key, ':'.join(ret))
        return ret

    # 获取库目录
    def get_lib_dirs(self, key='library_dirs'):
        return self.get_paths(self.section, key)

    # 获取运行时库目录
    def get_runtime_lib_dirs(self, key='runtime_library_dirs'):
        # 获取运行时库目录
        path = self.get_paths(self.section, key)
        # 如果路径为空字符串，则返回空列表
        if path == ['']:
            path = []
        return path

    # 获取包含目录
    def get_include_dirs(self, key='include_dirs'):
        return self.get_paths(self.section, key)

    # 获取源代码目录
    def get_src_dirs(self, key='src_dirs'):
        return self.get_paths(self.section, key)

    # 获取库文件列表
    def get_libs(self, key, default):
        try:
            # 从配置文件中获取库文件列表
            libs = self.cp.get(self.section, key)
        except NoOptionError:
            # 如果配置文件中没有对应的key
            if not default:
                return []
            # 如果默认值是字符串，则将其转为列表返回
            if is_string(default):
                return [default]
            # 返回默认值
            return default
        # 返回去掉逗号并去掉空白字符的库文件列表
        return [b for b in [a.strip() for a in libs.split(',')] if b]
    def get_libraries(self, key='libraries'):
        # 检查当前对象是否具有 '_lib_names' 属性，如果有则返回相应的值
        if hasattr(self, '_lib_names'):
            return self.get_libs(key, default=self._lib_names)
        else:
            # 如果没有 '_lib_names' 属性，则返回默认值 ''
            return self.get_libs(key, '')

    def library_extensions(self):
        # 创建自定义的 C 编译器对象
        c = customized_ccompiler()
        static_exts = []
        # 如果编译器类型不是 'msvc'，则添加 '.a' 作为静态库扩展名
        if c.compiler_type != 'msvc':
            static_exts.append('.a')
        # 如果运行平台是 'win32'，则添加 '.lib' 作为静态库扩展名
        if sys.platform == 'win32':
            static_exts.append('.lib')  # .lib is used by MSVC and others
        # 根据search_static_first决定静态库和动态库的顺序
        if self.search_static_first:
            exts = static_exts + [so_ext]
        else:
            exts = [so_ext] + static_exts
        # 如果运行平台是 'cygwin'，则添加 '.dll.a' 作为共享库扩展名
        if sys.platform == 'cygwin':
            exts.append('.dll.a')
        # 如果运行平台是 'darwin'，则添加 '.dylib' 作为共享库扩展名
        if sys.platform == 'darwin':
            exts.append('.dylib')
        return exts

    def check_libs(self, lib_dirs, libs, opt_libs=[]):
        """If static or shared libraries are available then return
        their info dictionary.

        Checks for all libraries as shared libraries first, then
        static (or vice versa if self.search_static_first is True).
        """
        exts = self.library_extensions()
        info = None
        # 检查各个扩展名下库的可用性
        for ext in exts:
            info = self._check_libs(lib_dirs, libs, opt_libs, [ext])
            if info is not None:
                break
        if not info:
            log.info('  libraries %s not found in %s', ','.join(libs),
                     lib_dirs)
        return info

    def check_libs2(self, lib_dirs, libs, opt_libs=[]):
        """If static or shared libraries are available then return
        their info dictionary.

        Checks each library for shared or static.
        """
        exts = self.library_extensions()
        # 检查每个库是共享库还是静态库的可用性
        info = self._check_libs(lib_dirs, libs, opt_libs, exts)
        if not info:
            log.info('  libraries %s not found in %s', ','.join(libs),
                     lib_dirs)

        return info

    def _find_lib(self, lib_dir, lib, exts):
        assert is_string(lib_dir)
        # 如果是在 Windows 平台，尝试在没有 'lib' 前缀的情况下寻找库文件
        if sys.platform == 'win32':
            lib_prefixes = ['', 'lib']
        else:
            lib_prefixes = ['lib']
        # 对于每个库名，看能否找到相应的文件
        for ext in exts:
            for prefix in lib_prefixes:
                p = self.combine_paths(lib_dir, prefix + lib + ext)
                if p:
                    break
            if p:
                assert len(p) == 1
                # ??? splitext on p[0] would do this for cygwin
                # doesn't seem correct
                if ext == '.dll.a':
                    lib += '.dll'
                if ext == '.lib':
                    lib = prefix + lib
                return lib

        return False
    # 在给定的库目录中查找指定的库文件和扩展名，保持 libs 的顺序不变
    def _find_libs(self, lib_dirs, libs, exts):
        # 确保我们保留 libs 的顺序，因为这可能很重要
        found_dirs, found_libs = [], []
        for lib in libs:
            for lib_dir in lib_dirs:
                found_lib = self._find_lib(lib_dir, lib, exts)
                if found_lib:
                    found_libs.append(found_lib)
                    if lib_dir not in found_dirs:
                        found_dirs.append(lib_dir)
                    break
        return found_dirs, found_libs

    # 检查在预期路径中的必需和可选库
    # 未找到可选库将被忽略
    def _check_libs(self, lib_dirs, libs, opt_libs, exts):
        if not is_sequence(lib_dirs):
            lib_dirs = [lib_dirs]
        # 首先，尝试找到必需的库
        found_dirs, found_libs = self._find_libs(lib_dirs, libs, exts)
        if len(found_libs) > 0 and len(found_libs) == len(libs):
            # 现在，检查可选库
            opt_found_dirs, opt_found_libs = self._find_libs(lib_dirs, opt_libs, exts)
            found_libs.extend(opt_found_libs)
            for lib_dir in opt_found_dirs:
                if lib_dir not in found_dirs:
                    found_dirs.append(lib_dir)
            info = {'libraries': found_libs, 'library_dirs': found_dirs}
            return info
        else:
            return None

    # 返回由参数中的所有项目组合而成的所有组合的现有路径列表
    def combine_paths(self, *args):
        """Return a list of existing paths composed by all combinations
        of items from the arguments.
        """
        return combine_paths(*args)
# 创建 fft_opt_info 类，继承自 system_info 类
class fft_opt_info(system_info):

    # 计算 FFT 优化信息的方法，返回信息字典
    def calc_info(self):
        info = {}
        # 获取 FFTW3、FFTW2 或者 DFFTW 的信息
        fftw_info = get_info('fftw3') or get_info('fftw2') or get_info('dfftw')
        # 获取 DjbFFT 的信息
        djbfft_info = get_info('djbfft')
        # 如果存在 FFTW 的信息
        if fftw_info:
            # 将 FFTW 的信息加入到 info 字典中
            dict_append(info, **fftw_info)
            # 如果存在 DjbFFT 的信息
            if djbfft_info:
                # 将 DjbFFT 的信息加入到 info 字典中
                dict_append(info, **djbfft_info)
            # 设置当前对象的信息
            self.set_info(**info)
            # 返回结果
            return


# 创建 fftw_info 类，继承自 system_info 类
class fftw_info(system_info):
    # 需要覆盖的变量
    section = 'fftw'
    dir_env_var = 'FFTW'
    notfounderror = FFTWNotFoundError
    ver_info = [{'name':'fftw3',
                    'libs':['fftw3'],
                    'includes':['fftw3.h'],
                    'macros':[('SCIPY_FFTW3_H', None)]},
                  {'name':'fftw2',
                    'libs':['rfftw', 'fftw'],
                    'includes':['fftw.h', 'rfftw.h'],
                    'macros':[('SCIPY_FFTW_H', None)]}]

    # 计算版本信息的方法，根据 ver_param 返回 True 或 False
    def calc_ver_info(self, ver_param):
        """Returns True on successful version detection, else False"""
        lib_dirs = self.get_lib_dirs()
        incl_dirs = self.get_include_dirs()

        opt = self.get_option_single(self.section + '_libs', 'libraries')
        libs = self.get_libs(opt, ver_param['libs'])
        info = self.check_libs(lib_dirs, libs)
        if info is not None:
            flag = 0
            for d in incl_dirs:
                if len(self.combine_paths(d, ver_param['includes'])) \
                   == len(ver_param['includes']):
                    dict_append(info, include_dirs=[d])
                    flag = 1
                    break
            if flag:
                dict_append(info, define_macros=ver_param['macros'])
            else:
                info = None
        if info is not None:
            self.set_info(**info)
            return True
        else:
            log.info('  %s not found' % (ver_param['name']))
            return False

    # 计算信息的方法
    def calc_info(self):
        for i in self.ver_info:
            if self.calc_ver_info(i):
                break


# 创建 fftw2_info 类，继承自 fftw_info 类
class fftw2_info(fftw_info):
    # 需要覆盖的变量
    section = 'fftw'
    dir_env_var = 'FFTW'
    notfounderror = FFTWNotFoundError
    ver_info = [{'name':'fftw2',
                    'libs':['rfftw', 'fftw'],
                    'includes':['fftw.h', 'rfftw.h'],
                    'macros':[('SCIPY_FFTW_H', None)]}
                  ]


# 创建 fftw3_info 类，继承自 fftw_info 类
class fftw3_info(fftw_info):
    # 需要覆盖的变量
    section = 'fftw3'
    dir_env_var = 'FFTW3'
    notfounderror = FFTWNotFoundError
    ver_info = [{'name':'fftw3',
                    'libs':['fftw3'],
                    'includes':['fftw3.h'],
                    'macros':[('SCIPY_FFTW3_H', None)]},
                  ]


# 创建 fftw3_armpl_info 类，继承自 fftw_info 类
class fftw3_armpl_info(fftw_info):
    # 需要覆盖的变量
    section = 'fftw3'
    dir_env_var = 'ARMPL_DIR'
    notfounderror = FFTWNotFoundError
    # 定义一个包含版本信息的列表，每个元素是一个字典，描述不同的库和相关信息
    ver_info = [{'name': 'fftw3',  # 库的名称为 fftw3
                 'libs': ['armpl_lp64_mp'],  # 需要链接的库列表，这里包括 armpl_lp64_mp
                 'includes': ['fftw3.h'],  # 需要包含的头文件列表，这里包括 fftw3.h
                 'macros': [('SCIPY_FFTW3_H', None)]}]  # 预定义的宏列表，这里定义了 SCIPY_FFTW3_H
class dfftw_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    ver_info = [{'name':'dfftw',
                    'libs':['drfftw', 'dfftw'],
                    'includes':['dfftw.h', 'drfftw.h'],
                    'macros':[('SCIPY_DFFTW_H', None)]}]



# 定义一个名为 dfftw_info 的类，继承自 fftw_info 类
class dfftw_info(fftw_info):
    # 设置 section 属性为 'fftw'，指定了信息的部分
    section = 'fftw'
    # 设置 dir_env_var 属性为 'FFTW'，指定了环境变量名称
    dir_env_var = 'FFTW'
    # 设置 ver_info 属性为包含一个字典的列表，描述了不同版本的信息
    ver_info = [{'name':'dfftw',
                    'libs':['drfftw', 'dfftw'],
                    'includes':['dfftw.h', 'drfftw.h'],
                    'macros':[('SCIPY_DFFTW_H', None)]}]



class sfftw_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    ver_info = [{'name':'sfftw',
                    'libs':['srfftw', 'sfftw'],
                    'includes':['sfftw.h', 'srfftw.h'],
                    'macros':[('SCIPY_SFFTW_H', None)]}]



# 定义一个名为 sfftw_info 的类，继承自 fftw_info 类
class sfftw_info(fftw_info):
    # 设置 section 属性为 'fftw'，指定了信息的部分
    section = 'fftw'
    # 设置 dir_env_var 属性为 'FFTW'，指定了环境变量名称
    dir_env_var = 'FFTW'
    # 设置 ver_info 属性为包含一个字典的列表，描述了不同版本的信息
    ver_info = [{'name':'sfftw',
                    'libs':['srfftw', 'sfftw'],
                    'includes':['sfftw.h', 'srfftw.h'],
                    'macros':[('SCIPY_SFFTW_H', None)]}]



class fftw_threads_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    ver_info = [{'name':'fftw threads',
                    'libs':['rfftw_threads', 'fftw_threads'],
                    'includes':['fftw_threads.h', 'rfftw_threads.h'],
                    'macros':[('SCIPY_FFTW_THREADS_H', None)]}]



# 定义一个名为 fftw_threads_info 的类，继承自 fftw_info 类
class fftw_threads_info(fftw_info):
    # 设置 section 属性为 'fftw'，指定了信息的部分
    section = 'fftw'
    # 设置 dir_env_var 属性为 'FFTW'，指定了环境变量名称
    dir_env_var = 'FFTW'
    # 设置 ver_info 属性为包含一个字典的列表，描述了不同版本的信息
    ver_info = [{'name':'fftw threads',
                    'libs':['rfftw_threads', 'fftw_threads'],
                    'includes':['fftw_threads.h', 'rfftw_threads.h'],
                    'macros':[('SCIPY_FFTW_THREADS_H', None)]}]



class dfftw_threads_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    ver_info = [{'name':'dfftw threads',
                    'libs':['drfftw_threads', 'dfftw_threads'],
                    'includes':['dfftw_threads.h', 'drfftw_threads.h'],
                    'macros':[('SCIPY_DFFTW_THREADS_H', None)]}]



# 定义一个名为 dfftw_threads_info 的类，继承自 fftw_info 类
class dfftw_threads_info(fftw_info):
    # 设置 section 属性为 'fftw'，指定了信息的部分
    section = 'fftw'
    # 设置 dir_env_var 属性为 'FFTW'，指定了环境变量名称
    dir_env_var = 'FFTW'
    # 设置 ver_info 属性为包含一个字典的列表，描述了不同版本的信息
    ver_info = [{'name':'dfftw threads',
                    'libs':['drfftw_threads', 'dfftw_threads'],
                    'includes':['dfftw_threads.h', 'drfftw_threads.h'],
                    'macros':[('SCIPY_DFFTW_THREADS_H', None)]}]



class sfftw_threads_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    ver_info = [{'name':'sfftw threads',
                    'libs':['srfftw_threads', 'sfftw_threads'],
                    'includes':['sfftw_threads.h', 'srfftw_threads.h'],
                    'macros':[('SCIPY_SFFTW_THREADS_H', None)]}]



# 定义一个名为 sfftw_threads_info 的类，继承自 fftw_info 类
class sfftw_threads_info(fftw_info):
    # 设置 section 属性为 'fftw'，指定了信息的部分
    section = 'fftw'
    # 设置 dir_env_var 属性为 'FFTW'，指定了环境变量名称
    dir_env_var = 'FFTW'
    # 设置 ver_info 属性为包含一个字典的列表，描述了不同版本的信息
    ver_info = [{'name':'sfftw threads',
                    'libs':['srfftw_threads', 'sfftw_threads'],
                    'includes':['sfftw_threads.h', 'srfftw_threads.h'],
                    'macros':[('SCIPY_SFFTW_THREADS_H', None)]}]



class djbfft_info(system_info):
    section = 'djbfft'
    dir_env_var = 'DJBFFT'
    notfounderror = DJBFFTNotFoundError

    def get_paths(self, section, key):
        # 获取父类 system_info 的 get_paths 方法的返回值
        pre_dirs = system_info.get_paths(self, section, key)
        # 初始化一个空列表 dirs
        dirs = []
        # 遍历 pre_dirs 中的每个路径 d
        for d in pre_dirs:
            # 将 combine_paths 方法的结果（包含 'djbfft' 的路径）添加到 dirs 中，然后再添加原始路径 d
            dirs.extend(self.combine_paths(d, ['djbfft']) + [d])
        # 返回 dirs 列表，其中包含了扩展后的路径列表
        return [d for d in dirs if os.path.isdir(d)]

    def calc_info(self):
        # 获取库目录列表
        lib_dirs = self.get_lib_dirs()
        # 获取包含目录列表
        incl_dirs = self.get_include_dirs()
        # 初始化 info 变量为 None
        info = None
        # 遍历 lib_dirs 中的每个目录 d
        for d in lib_dirs:
            # 尝试获取包含 'djbfft.a' 的路径 p
            p = self.combine_paths(d, ['djbfft.a'])
            # 如果 p 不为空
            if p:
                # 设置 info 字典，包含额外的对象文件路径
                info = {'extra_objects': p}
                # 中断循环
                break
            # 尝试获取包含 'libdjbfft.a' 或 'libdjbfft.so' 的路径 p
            p = self.combine_paths(d, ['libdjbfft.a', 'libdjbfft' + so_ext])
            # 如果 p 不为空
            if p:
                # 设置 info 字典，包含库 'djbfft' 和库目录 d
                info = {'libraries': ['djbfft'], 'library_dirs': [d]}
                # 中断循环
                break
        # 如果 info 仍为 None，则返回
        if info is None:
            return
        # 遍历 incl_dirs 中的每个目录 d
        for d in incl_dirs:
            # 如果 combine_paths 方法返回包含 'fftc8.h' 和 'fftfreq.h' 两个文件的路径列表
            if len(self.combine_paths(d, ['fftc8.h', 'fftfreq.h'])) == 2:
                # 向 info 字典中添加包含目录 d 和宏定义 [('SCIPY_DJBFFT_H', None)]
                dict_append(info, include_dirs=[d],
                            define_macros=[('SCIPY_DJBFFT_H', None)])
                # 调用 set_info 方法，更新 info 信息
                self.set_info(**info)
                # 中断方法
                return
        # 方法结束
        return



class mkl_info(system_info):
    section = 'mkl'
    dir_env_var = 'MKLROOT'
    _lib_mkl = ['mkl_rt']



#
    # 获取 MKL 根目录的路径
    def get_mkl_rootdir(self):
        # 尝试从环境变量 MKLROOT 中获取 MKL 根目录路径
        mklroot = os.environ.get('MKLROOT', None)
        if mklroot is not None:
            return mklroot
        
        # 如果未从 MKLROOT 变量中获取到路径，则尝试从 LD_LIBRARY_PATH 环境变量中获取路径
        paths = os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
        
        # 检查系统中是否存在 ld.so.conf 文件
        ld_so_conf = '/etc/ld.so.conf'
        if os.path.isfile(ld_so_conf):
            # 逐行读取 ld.so.conf 文件，并将非空行的路径添加到 paths 列表中
            with open(ld_so_conf) as f:
                for d in f:
                    d = d.strip()
                    if d:
                        paths.append(d)
        
        # 初始化一个空列表，用于存储 Intel MKL 目录
        intel_mkl_dirs = []
        
        # 遍历 paths 列表中的每个路径
        for path in paths:
            # 将当前路径按分隔符分割成列表
            path_atoms = path.split(os.sep)
            # 遍历当前路径的各个部分
            for m in path_atoms:
                # 检查是否以 'mkl' 开头的部分
                if m.startswith('mkl'):
                    # 将路径截断到当前 'mkl' 目录及其上一级，并添加到 intel_mkl_dirs 列表中
                    d = os.sep.join(path_atoms[:path_atoms.index(m) + 2])
                    intel_mkl_dirs.append(d)
                    break
        
        # 遍历 paths 列表中的每个路径
        for d in paths:
            # 在当前路径下寻找以 'mkl' 开头的子目录，并将找到的目录添加到 dirs 列表中
            dirs = glob(os.path.join(d, 'mkl', '*'))
            dirs += glob(os.path.join(d, 'mkl*'))
            # 遍历 dirs 列表中的每个子目录
            for sub_dir in dirs:
                # 检查当前子目录下是否存在 'lib' 子目录，若存在则返回该子目录作为 MKL 根目录
                if os.path.isdir(os.path.join(sub_dir, 'lib')):
                    return sub_dir
        
        # 如果以上尝试均未找到合适的 MKL 根目录，则返回 None
        return None

    # 初始化方法，用于获取 MKL 根目录并设置系统信息
    def __init__(self):
        # 获取 MKL 根目录路径
        mklroot = self.get_mkl_rootdir()
        
        # 如果未找到 MKL 根目录路径，则调用系统信息的初始化方法
        if mklroot is None:
            system_info.__init__(self)
        else:
            # 导入 CPU 信息模块，并根据 CPU 类型设置 plt 变量
            from .cpuinfo import cpu
            if cpu.is_Itanium():
                plt = '64'
            elif cpu.is_Intel() and cpu.is_64bit():
                plt = 'intel64'
            else:
                plt = '32'
            
            # 使用找到的 MKL 根目录路径初始化系统信息对象
            system_info.__init__(
                self,
                default_lib_dirs=[os.path.join(mklroot, 'lib', plt)],  # 设置默认的库路径
                default_include_dirs=[os.path.join(mklroot, 'include')])  # 设置默认的头文件路径

    # 计算信息方法，用于检查库并设置相关信息
    def calc_info(self):
        # 获取当前系统的库路径列表
        lib_dirs = self.get_lib_dirs()
        # 获取当前系统的头文件路径列表
        incl_dirs = self.get_include_dirs()
        # 获取 mkl_libs 选项的单一值
        opt = self.get_option_single('mkl_libs', 'libraries')
        # 根据 mkl_libs 的值和预定义的库名字典获取 MKL 相关的库列表
        mkl_libs = self.get_libs(opt, self._lib_mkl)
        # 检查系统中是否存在所需的库文件，并返回相关信息
        info = self.check_libs2(lib_dirs, mkl_libs)
        
        # 如果未找到所需的库文件，则直接返回
        if info is None:
            return
        
        # 向信息字典中添加宏定义和头文件路径信息
        dict_append(info,
                    define_macros=[('SCIPY_MKL_H', None),
                                   ('HAVE_CBLAS', None)],
                    include_dirs=incl_dirs)
        
        # 如果系统平台为 win32，则不需要 pthread 库
        if sys.platform == 'win32':
            pass  # win32 系统不需要 pthread 库
        else:
            # 向信息字典中添加 pthread 库的信息
            dict_append(info, libraries=['pthread'])
        
        # 将收集到的信息应用到当前对象中
        self.set_info(**info)
class lapack_mkl_info(mkl_info):
    pass

class blas_mkl_info(mkl_info):
    pass

class ssl2_info(system_info):
    # SSL2 模块信息，继承自系统信息类
    section = 'ssl2'
    dir_env_var = 'SSL2_DIR'
    # 多线程版本。Python 必须由富士通编译器构建。
    _lib_ssl2 = ['fjlapackexsve']
    # 单线程版本
    # _lib_ssl2 = ['fjlapacksve']

    def get_tcsds_rootdir(self):
        # 获取环境变量中的 TCSDS_PATH 路径
        tcsdsroot = os.environ.get('TCSDS_PATH', None)
        if tcsdsroot is not None:
            return tcsdsroot
        return None

    def __init__(self):
        # 初始化 SSL2 模块信息
        tcsdsroot = self.get_tcsds_rootdir()
        if tcsdsroot is None:
            # 如果未找到 TCSDS_PATH，调用父类的初始化方法
            system_info.__init__(self)
        else:
            # 如果找到 TCSDS_PATH，使用指定的默认库和包含目录初始化父类
            system_info.__init__(
                self,
                default_lib_dirs=[os.path.join(tcsdsroot, 'lib64')],
                default_include_dirs=[os.path.join(tcsdsroot, 'clang-comp/include')])

    def calc_info(self):
        # 计算 SSL2 模块信息
        tcsdsroot = self.get_tcsds_rootdir()

        lib_dirs = self.get_lib_dirs()
        if lib_dirs is None:
            lib_dirs = os.path.join(tcsdsroot, 'lib64')

        incl_dirs = self.get_include_dirs()
        if incl_dirs is None:
            incl_dirs = os.path.join(tcsdsroot, 'clang-comp/include')

        ssl2_libs = self.get_libs('ssl2_libs', self._lib_ssl2)

        # 检查 SSL2 相关库是否可用并返回信息
        info = self.check_libs2(lib_dirs, ssl2_libs)
        if info is None:
            return
        # 将额外的定义和包含目录附加到信息中
        dict_append(info,
                    define_macros=[('HAVE_CBLAS', None),
                                   ('HAVE_SSL2', 1)],
                    include_dirs=incl_dirs,)
        self.set_info(**info)


class lapack_ssl2_info(ssl2_info):
    pass

class blas_ssl2_info(ssl2_info):
    pass

class armpl_info(system_info):
    # ARMPL 模块信息，继承自系统信息类
    section = 'armpl'
    dir_env_var = 'ARMPL_DIR'
    _lib_armpl = ['armpl_lp64_mp']

    def calc_info(self):
        # 计算 ARMPL 模块信息
        lib_dirs = self.get_lib_dirs()
        incl_dirs = self.get_include_dirs()
        armpl_libs = self.get_libs('armpl_libs', self._lib_armpl)
        # 检查 ARMPL 相关库是否可用并返回信息
        info = self.check_libs2(lib_dirs, armpl_libs)
        if info is None:
            return
        # 将额外的定义和包含目录附加到信息中
        dict_append(info,
                    define_macros=[('SCIPY_MKL_H', None),
                                   ('HAVE_CBLAS', None)],
                    include_dirs=incl_dirs)
        self.set_info(**info)

class lapack_armpl_info(armpl_info):
    pass

class blas_armpl_info(armpl_info):
    pass

class atlas_info(system_info):
    # ATLAS 模块信息，继承自系统信息类
    section = 'atlas'
    dir_env_var = 'ATLAS'
    _lib_names = ['f77blas', 'cblas']
    if sys.platform[:7] == 'freebsd':
        _lib_atlas = ['atlas_r']
        _lib_lapack = ['alapack_r']
    else:
        _lib_atlas = ['atlas']
        _lib_lapack = ['lapack']

    notfounderror = AtlasNotFoundError
    # 定义一个方法 `get_paths`，接受两个参数 `section` 和 `key`
    def get_paths(self, section, key):
        # 调用 `system_info` 对象的 `get_paths` 方法，返回路径列表 `pre_dirs`
        pre_dirs = system_info.get_paths(self, section, key)
        # 初始化一个空列表 `dirs` 用来存放最终的路径
        dirs = []
        # 遍历 `pre_dirs` 列表中的每个路径 `d`
        for d in pre_dirs:
            # 将 `d` 和列表 `['atlas*', 'ATLAS*', 'sse', '3dnow', 'sse2']` 合并后加入到 `dirs` 中
            dirs.extend(self.combine_paths(d, ['atlas*', 'ATLAS*',
                                               'sse', '3dnow', 'sse2']) + [d])
        # 返回过滤后的 `dirs` 列表，只保留其中是目录的路径
        return [d for d in dirs if os.path.isdir(d)]
# 定义一个名为 atlas_blas_info 的类，继承自 atlas_info 类
class atlas_blas_info(atlas_info):
    # 类变量，存储库的名称列表
    _lib_names = ['f77blas', 'cblas']

    # 计算信息的方法
    def calc_info(self):
        # 获取库目录
        lib_dirs = self.get_lib_dirs()
        # 创建空字典 info
        info = {}
        # 获取 'atlas_libs' 选项的单一值
        opt = self.get_option_single('atlas_libs', 'libraries')
        # 获取 ATLAS 库名称列表
        atlas_libs = self.get_libs(opt, self._lib_names + self._lib_atlas)
        # 检查库是否存在
        atlas = self.check_libs2(lib_dirs, atlas_libs, [])
        # 如果 ATLAS 库不存在，则返回
        if atlas is None:
            return
        # 获取包含目录
        include_dirs = self.get_include_dirs()
        # 尝试合并路径以获取 'cblas.h' 文件所在目录，返回列表并取第一个元素
        h = (self.combine_paths(lib_dirs + include_dirs, 'cblas.h') or [None])
        h = h[0]
        # 如果 h 不为空
        if h:
            # 获取 'cblas.h' 文件所在目录的父目录，并添加到 info 字典的 'include_dirs' 键下
            h = os.path.dirname(h)
            dict_append(info, include_dirs=[h])
        # 设置语言为 'c'
        info['language'] = 'c'
        # 定义宏 'HAVE_CBLAS'，添加到 info 字典的 'define_macros' 键下
        info['define_macros'] = [('HAVE_CBLAS', None)]

        # 获取 ATLAS 的版本信息和额外信息
        atlas_version, atlas_extra_info = get_atlas_version(**atlas)
        # 将额外信息添加到 atlas 字典中
        dict_append(atlas, **atlas_extra_info)

        # 将 atlas 字典的内容添加到 info 字典中
        dict_append(info, **atlas)

        # 设置信息，调用父类的 set_info 方法
        self.set_info(**info)
        # 返回
        return
    # 将LapackNotFoundError异常类赋值给变量notfounderror
    notfounderror = LapackNotFoundError

    # 定义方法calc_info，用于计算信息
    def calc_info(self):
        # 调用self对象的get_lib_dirs方法，获取库目录列表
        lib_dirs = self.get_lib_dirs()

        # 获取名为'lapack_libs'的单一选项值，并将其作为'libraries'参数传递给get_option_single方法
        opt = self.get_option_single('lapack_libs', 'libraries')

        # 调用self对象的get_libs方法，根据opt参数值和self._lib_names获取lapack库列表
        lapack_libs = self.get_libs(opt, self._lib_names)

        # 调用self对象的check_libs方法，检查库目录lib_dirs中是否存在lapack_libs所需的库，并传入空列表作为额外依赖项
        info = self.check_libs(lib_dirs, lapack_libs, [])

        # 如果info为None，直接返回，结束方法
        if info is None:
            return
        
        # 如果info不为None，设置info字典中的'language'键为'f77'
        info['language'] = 'f77'
        
        # 调用self对象的set_info方法，传入info字典作为关键字参数展开
        self.set_info(**info)
class lapack_src_info(system_info):
    # LAPACK_SRC is deprecated, please do not use this!
    # Build or install a BLAS library via your package manager or from
    # source separately.
    # lapack_src_info 类继承自 system_info 类，用于获取 LAPACK 源码路径信息。

    section = 'lapack_src'
    dir_env_var = 'LAPACK_SRC'
    notfounderror = LapackSrcNotFoundError

    def get_paths(self, section, key):
        # 覆盖父类方法，获取指定 section 和 key 的路径列表
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            # 将每个路径加入到 dirs 列表中，并与 ['LAPACK*/SRC', 'SRC'] 组合形成新路径
            dirs.extend([d] + self.combine_paths(d, ['LAPACK*/SRC', 'SRC']))
        return [d for d in dirs if os.path.isdir(d)]

atlas_version_c_text = r'''
/* This file is generated from numpy/distutils/system_info.py */
void ATL_buildinfo(void);
int main(void) {
  ATL_buildinfo();
  return 0;
}
'''

_cached_atlas_version = {}

def get_atlas_version(**config):
    # 获取 ATLAS 版本信息
    libraries = config.get('libraries', [])
    library_dirs = config.get('library_dirs', [])
    key = (tuple(libraries), tuple(library_dirs))
    if key in _cached_atlas_version:
        # 如果缓存中存在，则直接返回缓存中的 ATLAS 版本信息
        return _cached_atlas_version[key]
    c = cmd_config(Distribution())
    atlas_version = None
    info = {}
    try:
        # 尝试获取 ATLAS 版本信息的输出
        s, o = c.get_output(atlas_version_c_text,
                            libraries=libraries, library_dirs=library_dirs,
                           )
        if s and re.search(r'undefined reference to `_gfortran', o, re.M):
            # 如果出现链接错误，则尝试添加 gfortran 库重新获取版本信息
            s, o = c.get_output(atlas_version_c_text,
                                libraries=libraries + ['gfortran'],
                                library_dirs=library_dirs,
                               )
            if not s:
                # 如果还是无法成功，给出警告，并更新 info 字典
                warnings.warn(textwrap.dedent("""
                    *****************************************************
                    Linkage with ATLAS requires gfortran. Use

                      python setup.py config_fc --fcompiler=gnu95 ...

                    when building extension libraries that use ATLAS.
                    Make sure that -lgfortran is used for C++ extensions.
                    *****************************************************
                    """), stacklevel=2)
                dict_append(info, language='f90',
                            define_macros=[('ATLAS_REQUIRES_GFORTRAN', None)])
    except Exception:  # failed to get version from file -- maybe on Windows
        # 捕获异常：无法从文件中获取版本信息，可能是在 Windows 平台上
        # 查看库目录名
        for o in library_dirs:
            # 在目录名中查找形如 ATLAS_版本号_ 的字符串
            m = re.search(r'ATLAS_(?P<version>\d+[.]\d+[.]\d+)_', o)
            if m:
                # 提取版本号
                atlas_version = m.group('version')
            # 如果找到版本号则终止循环
            if atlas_version is not None:
                break

        # 最终选择 —— 查看 ATLAS_VERSION 环境变量
        if atlas_version is None:
            # 如果未找到版本号，则尝试从环境变量 ATLAS_VERSION 中获取
            atlas_version = os.environ.get('ATLAS_VERSION', None)
        if atlas_version:
            # 如果找到版本号，则将 ATLAS_INFO 宏添加到信息字典中
            dict_append(info, define_macros=[(
                'ATLAS_INFO', _c_string_literal(atlas_version))
            ])
        else:
            # 如果未找到版本号，则将 NO_ATLAS_INFO 宏添加到信息字典中
            dict_append(info, define_macros=[('NO_ATLAS_INFO', -1)])
        # 返回 ATLAS 版本号或默认字符串 '?.?.?' 以及处理后的信息字典
        return atlas_version or '?.?.?', info

    if not s:
        # 如果 s 为空，则从 o 中查找 ATLAS version 版本号信息
        m = re.search(r'ATLAS version (?P<version>\d+[.]\d+[.]\d+)', o)
        if m:
            # 提取版本号
            atlas_version = m.group('version')
    if atlas_version is None:
        # 如果未找到版本号，则根据特定字符串在 o 中查找
        if re.search(r'undefined symbol: ATL_buildinfo', o, re.M):
            # 如果找到特定字符串，则设置 atlas_version 为预定义的版本号
            atlas_version = '3.2.1_pre3.3.6'
        else:
            # 否则记录日志信息
            log.info('Status: %d', s)
            log.info('Output: %s', o)

    elif atlas_version == '3.2.1_pre3.3.6':
        # 如果 atlas_version 是预定义的版本号，则添加 NO_ATLAS_INFO 宏到信息字典中
        dict_append(info, define_macros=[('NO_ATLAS_INFO', -2)])
    else:
        # 否则将 ATLAS_INFO 宏添加到信息字典中
        dict_append(info, define_macros=[(
            'ATLAS_INFO', _c_string_literal(atlas_version))
        ])
    # 将结果缓存并返回 ATLAS 版本号以及信息字典
    result = _cached_atlas_version[key] = atlas_version, info
    return result
# LAPACK 优化信息类，继承自系统信息类
class lapack_opt_info(system_info):
    # LAPACK 未找到错误类
    notfounderror = LapackNotFoundError

    # 已知 LAPACK 库的默认顺序列表
    lapack_order = ['armpl', 'mkl', 'ssl2', 'openblas', 'flame',
                    'accelerate', 'atlas', 'lapack']
    # LAPACK 库顺序环境变量名称
    order_env_var_name = 'NPY_LAPACK_ORDER'
    
    # 计算 ARM Performance Libraries (ARMPL) 的信息
    def _calc_info_armpl(self):
        info = get_info('lapack_armpl')
        if info:
            self.set_info(**info)
            return True
        return False

    # 计算 Intel Math Kernel Library (MKL) 的信息
    def _calc_info_mkl(self):
        info = get_info('lapack_mkl')
        if info:
            self.set_info(**info)
            return True
        return False

    # 计算 SSL2 LAPACK 的信息
    def _calc_info_ssl2(self):
        info = get_info('lapack_ssl2')
        if info:
            self.set_info(**info)
            return True
        return False

    # 计算 OpenBLAS LAPACK 的信息
    def _calc_info_openblas(self):
        info = get_info('openblas_lapack')
        if info:
            self.set_info(**info)
            return True
        info = get_info('openblas_clapack')
        if info:
            self.set_info(**info)
            return True
        return False

    # 计算 FLAME LAPACK 的信息
    def _calc_info_flame(self):
        info = get_info('flame')
        if info:
            self.set_info(**info)
            return True
        return False

    # 计算 ATLAS LAPACK 的信息
    def _calc_info_atlas(self):
        info = get_info('atlas_3_10_threads')
        if not info:
            info = get_info('atlas_3_10')
        if not info:
            info = get_info('atlas_threads')
        if not info:
            info = get_info('atlas')
        if info:
            # 判断 ATLAS 是否包含 LAPACK...
            # 如果没有，我们需要 LAPACK 库，但不需要 BLAS！
            l = info.get('define_macros', [])
            if ('ATLAS_WITH_LAPACK_ATLAS', None) in l \
               or ('ATLAS_WITHOUT_LAPACK', None) in l:
                # 获取 LAPACK 信息（可能会有警告）
                lapack_info = self._get_info_lapack()
                if not lapack_info:
                    return False
                dict_append(info, **lapack_info)
            self.set_info(**info)
            return True
        return False

    # 计算 Accelerate Framework 的 LAPACK 信息
    def _calc_info_accelerate(self):
        info = get_info('accelerate')
        if info:
            self.set_info(**info)
            return True
        return False

    # 获取 BLAS 的信息
    def _get_info_blas(self):
        # 默认获取优化的 BLAS 实现
        info = get_info('blas_opt')
        if not info:
            # 如果未找到优化 BLAS，发出警告
            warnings.warn(BlasNotFoundError.__doc__ or '', stacklevel=3)
            info_src = get_info('blas_src')
            if not info_src:
                warnings.warn(BlasSrcNotFoundError.__doc__ or '', stacklevel=3)
                return {}
            dict_append(info, libraries=[('fblas_src', info_src)])
        return info
    # 获取 LAPACK 的信息，首先尝试从系统中获取
    info = get_info('lapack')
    # 如果未找到 LAPACK 信息，则发出警告
    if not info:
        # 在第三层堆栈上发出 LAPACK 未找到的警告
        warnings.warn(LapackNotFoundError.__doc__ or '', stacklevel=3)
        # 尝试获取 LAPACK 源码信息
        info_src = get_info('lapack_src')
        # 如果也未找到 LAPACK 源码信息，则发出相应警告并返回空字典
        if not info_src:
            warnings.warn(LapackSrcNotFoundError.__doc__ or '', stacklevel=3)
            return {}
        # 向 info 中添加源码库信息
        dict_append(info, libraries=[('flapack_src', info_src)])
    # 返回获取到的 LAPACK 信息字典
    return info

    # 计算 LAPACK 相关信息，首先获取 LAPACK 信息
    info = self._get_info_lapack()
    # 如果成功获取 LAPACK 信息，则继续获取 BLAS 相关信息并添加到 info 中
    if info:
        info_blas = self._get_info_blas()
        dict_append(info, **info_blas)
        # 设置 NO_ATLAS_INFO 宏定义
        dict_append(info, define_macros=[('NO_ATLAS_INFO', 1)])
        # 将整理好的信息传递给实例对象
        self.set_info(**info)
        return True
    # 若未能获取 LAPACK 信息，则返回 False
    return False

    # 从环境变量中计算信息
    info = {}
    info['language'] = 'f77'
    info['libraries'] = []
    info['include_dirs'] = []
    info['define_macros'] = []
    # 使用环境变量 NPY_LAPACK_LIBS 中的链接器参数
    info['extra_link_args'] = os.environ['NPY_LAPACK_LIBS'].split()
    # 将整理好的信息传递给实例对象
    self.set_info(**info)
    return True

    # 根据给定的名字计算信息，调用相应的计算方法
    return getattr(self, '_calc_info_{}'.format(name))()

    # 计算 LAPACK 相关信息的总入口
    lapack_order, unknown_order = _parse_env_order(self.lapack_order, self.order_env_var_name)
    # 如果存在不被识别的 LAPACK 顺序，抛出 ValueError 异常
    if len(unknown_order) > 0:
        raise ValueError("lapack_opt_info user defined "
                         "LAPACK order has unacceptable "
                         "values: {}".format(unknown_order))

    # 如果环境变量中存在 NPY_LAPACK_LIBS
    # 跳过自动检测，将语言设置为 F77，并直接使用环境变量中的链接器标志
    self._calc_info_from_envvar()
    return

    # 按照 LAPACK 的顺序依次计算信息
    for lapack in lapack_order:
        if self._calc_info(lapack):
            return

    # 如果 LAPACK 的顺序列表中不包含 'lapack'
    # 即用户可能请求不使用任何库，仍然需要发出警告以指示缺失的软件包！
    warnings.warn(LapackNotFoundError.__doc__ or '', stacklevel=2)
    warnings.warn(LapackSrcNotFoundError.__doc__ or '', stacklevel=2)
# 定义一个名为 _ilp64_opt_info_mixin 的类，用于处理 ILP64 架构的优化信息混合
class _ilp64_opt_info_mixin:
    # 符号后缀和前缀初始化为 None
    symbol_suffix = None
    symbol_prefix = None

    # 检查给定的信息是否符合要求
    def _check_info(self, info):
        # 将宏定义转换为字典格式
        macros = dict(info.get('define_macros', []))
        # 获取 BLAS 符号前缀和后缀
        prefix = macros.get('BLAS_SYMBOL_PREFIX', '')
        suffix = macros.get('BLAS_SYMBOL_SUFFIX', '')

        # 检查 BLAS 符号前缀是否匹配
        if self.symbol_prefix not in (None, prefix):
            return False

        # 检查 BLAS 符号后缀是否匹配
        if self.symbol_suffix not in (None, suffix):
            return False

        # 返回信息是否有效的布尔值
        return bool(info)


# 继承 lapack_opt_info 类和 _ilp64_opt_info_mixin 类的 lapack_ilp64_opt_info 类
class lapack_ilp64_opt_info(lapack_opt_info, _ilp64_opt_info_mixin):
    # 如果找不到 ILP64 版本的 LAPACK 库，引发 LapackILP64NotFoundError 异常
    notfounderror = LapackILP64NotFoundError
    # LAPACK 库的优先顺序
    lapack_order = ['openblas64_', 'openblas_ilp64', 'accelerate']
    # 环境变量中用于设置 LAPACK 库优先顺序的名称
    order_env_var_name = 'NPY_LAPACK_ILP64_ORDER'

    # 计算给定名称的 LAPACK 库信息
    def _calc_info(self, name):
        # 打印计算 LAPACK 信息的调试信息
        print('lapack_ilp64_opt_info._calc_info(name=%s)' % (name))
        # 获取指定名称的 LAPACK 库信息
        info = get_info(name + '_lapack')
        # 如果信息符合要求，则设置该信息并返回 True
        if self._check_info(info):
            self.set_info(**info)
            return True
        else:
            # 否则打印 LAPACK 库不存在的错误信息
            print('%s_lapack does not exist' % (name))
        # 返回 False 表示未找到符合要求的 LAPACK 库信息
        return False


# 继承 lapack_ilp64_opt_info 类的 lapack_ilp64_plain_opt_info 类，用于修正符号名称
class lapack_ilp64_plain_opt_info(lapack_ilp64_opt_info):
    # 符号前缀为空字符串
    symbol_prefix = ''
    # 符号后缀为空字符串
    symbol_suffix = ''


# 继承 lapack_ilp64_opt_info 类的 lapack64__opt_info 类，用于设置符号后缀为 '64_'
class lapack64__opt_info(lapack_ilp64_opt_info):
    # 符号前缀为空字符串
    symbol_prefix = ''
    # 符号后缀为 '64_'
    symbol_suffix = '64_'


# 继承 system_info 类的 blas_opt_info 类，用于处理 BLAS 库的优化信息
class blas_opt_info(system_info):
    # 如果找不到 BLAS 库，引发 BlasNotFoundError 异常
    notfounderror = BlasNotFoundError
    # 所有已知 BLAS 库的默认顺序列表
    blas_order = ['armpl', 'mkl', 'ssl2', 'blis', 'openblas',
                  'accelerate', 'atlas', 'blas']
    # 环境变量中用于设置 BLAS 库优先顺序的名称
    order_env_var_name = 'NPY_BLAS_ORDER'

    # 计算 armpl BLAS 库的信息
    def _calc_info_armpl(self):
        # 获取 armpl BLAS 库的信息
        info = get_info('blas_armpl')
        # 如果成功获取信息，则设置该信息并返回 True
        if info:
            self.set_info(**info)
            return True
        # 否则返回 False 表示未找到符合要求的 armpl BLAS 库信息
        return False

    # 计算 mkl BLAS 库的信息
    def _calc_info_mkl(self):
        # 获取 mkl BLAS 库的信息
        info = get_info('blas_mkl')
        # 如果成功获取信息，则设置该信息并返回 True
        if info:
            self.set_info(**info)
            return True
        # 否则返回 False 表示未找到符合要求的 mkl BLAS 库信息
        return False

    # 计算 ssl2 BLAS 库的信息
    def _calc_info_ssl2(self):
        # 获取 ssl2 BLAS 库的信息
        info = get_info('blas_ssl2')
        # 如果成功获取信息，则设置该信息并返回 True
        if info:
            self.set_info(**info)
            return True
        # 否则返回 False 表示未找到符合要求的 ssl2 BLAS 库信息
        return False

    # 计算 blis BLAS 库的信息
    def _calc_info_blis(self):
        # 获取 blis BLAS 库的信息
        info = get_info('blis')
        # 如果成功获取信息，则设置该信息并返回 True
        if info:
            self.set_info(**info)
            return True
        # 否则返回 False 表示未找到符合要求的 blis BLAS 库信息
        return False

    # 计算 openblas BLAS 库的信息
    def _calc_info_openblas(self):
        # 获取 openblas BLAS 库的信息
        info = get_info('openblas')
        # 如果成功获取信息，则设置该信息并返回 True
        if info:
            self.set_info(**info)
            return True
        # 否则返回 False 表示未找到符合要求的 openblas BLAS 库信息
        return False

    # 计算 atlas BLAS 库的信息
    def _calc_info_atlas(self):
        # 首先尝试获取 atlas_3_10_blas_threads 的信息
        info = get_info('atlas_3_10_blas_threads')
        # 如果信息不存在，则尝试获取 atlas_3_10_blas 的信息
        if not info:
            info = get_info('atlas_3_10_blas')
        # 如果信息不存在，则尝试获取 atlas_blas_threads 的信息
        if not info:
            info = get_info('atlas_blas_threads')
        # 如果信息不存在，则尝试获取 atlas_blas 的信息
        if not info:
            info = get_info('atlas_blas')
        # 如果成功获取信息，则设置该信息并返回 True
        if info:
            self.set_info(**info)
            return True
        # 否则返回 False 表示未找到符合要求的 atlas BLAS 库信息
        return False

    # 计算 accelerate BLAS 库的信息
    def _calc_info_accelerate(self):
        # 获取 accelerate BLAS 库的信息
        info = get_info('accelerate')
        # 如果成功获取信息，则设置该信息并返回 True
        if info:
            self.set_info(**info)
            return True
        # 否则返回 False 表示未找到符合要求的 accelerate BLAS 库信息
        return False
    # 计算 BLAS 相关信息并设置到实例中
    def _calc_info_blas(self):
        # 发出关于非优化 BLAS 库的警告
        warnings.warn(BlasOptNotFoundError.__doc__ or '', stacklevel=3)
        # 初始化信息字典
        info = {}
        # 将 ('NO_ATLAS_INFO', 1) 加入到宏定义中
        dict_append(info, define_macros=[('NO_ATLAS_INFO', 1)])

        # 获取 BLAS 库信息
        blas = get_info('blas')
        if blas:
            # 将 BLAS 相关信息加入到 info 字典中
            dict_append(info, **blas)
        else:
            # 如果没有找到 BLAS 库，则发出相应警告
            warnings.warn(BlasNotFoundError.__doc__ or '', stacklevel=3)
            # 获取 BLAS 源码路径信息
            blas_src = get_info('blas_src')
            if not blas_src:
                # 如果连 BLAS 源码路径也找不到，则发出相应警告并返回 False
                warnings.warn(BlasSrcNotFoundError.__doc__ or '', stacklevel=3)
                return False
            # 将 ('fblas_src', blas_src) 加入到库列表中
            dict_append(info, libraries=[('fblas_src', blas_src)])

        # 将计算得到的 info 设置到当前实例中
        self.set_info(**info)
        return True

    # 从环境变量中计算 BLAS 相关信息并设置到实例中
    def _calc_info_from_envvar(self):
        info = {}
        info['language'] = 'f77'
        info['libraries'] = []
        info['include_dirs'] = []
        info['define_macros'] = []
        # 将环境变量 NPY_BLAS_LIBS 的值按空格分割后加入额外链接参数中
        info['extra_link_args'] = os.environ['NPY_BLAS_LIBS'].split()
        if 'NPY_CBLAS_LIBS' in os.environ:
            # 如果环境变量 NPY_CBLAS_LIBS 存在，则添加宏定义 ('HAVE_CBLAS', None)
            info['define_macros'].append(('HAVE_CBLAS', None))
            # 将环境变量 NPY_CBLAS_LIBS 的值按空格分割后扩展到额外链接参数中
            info['extra_link_args'].extend(
                                        os.environ['NPY_CBLAS_LIBS'].split())
        # 将计算得到的 info 设置到当前实例中
        self.set_info(**info)
        return True

    # 根据给定的名字调用对应的计算信息方法
    def _calc_info(self, name):
        return getattr(self, '_calc_info_{}'.format(name))()

    # 计算 BLAS 相关信息的入口方法
    def calc_info(self):
        # 解析 BLAS 库的顺序和未知顺序
        blas_order, unknown_order = _parse_env_order(self.blas_order, self.order_env_var_name)
        if len(unknown_order) > 0:
            # 如果存在不可接受的 BLAS 库顺序值，则抛出 ValueError 异常
            raise ValueError("blas_opt_info user defined BLAS order has unacceptable values: {}".format(unknown_order))

        if 'NPY_BLAS_LIBS' in os.environ:
            # 如果环境变量 NPY_BLAS_LIBS 存在，则直接使用环境变量指定的链接标志
            self._calc_info_from_envvar()
            return

        # 按照优先级顺序尝试计算 BLAS 相关信息
        for blas in blas_order:
            if self._calc_info(blas):
                return

        if 'blas' not in blas_order:
            # 如果在 BLAS 库顺序中没有找到 'blas'，则发出警告信号缺少相关包
            warnings.warn(BlasNotFoundError.__doc__ or '', stacklevel=2)
            warnings.warn(BlasSrcNotFoundError.__doc__ or '', stacklevel=2)
# 定义一个名为 blas_ilp64_opt_info 的类，继承自 blas_opt_info 和 _ilp64_opt_info_mixin
class blas_ilp64_opt_info(blas_opt_info, _ilp64_opt_info_mixin):
    # 当找不到 ILP64 版本的 BLAS 时，抛出 BlasILP64NotFoundError 异常
    notfounderror = BlasILP64NotFoundError
    # BLAS 库的优先顺序列表
    blas_order = ['openblas64_', 'openblas_ilp64', 'accelerate']
    # 环境变量的名称，用于控制 BLAS 库的优先顺序
    order_env_var_name = 'NPY_BLAS_ILP64_ORDER'

    # 计算给定名称 BLAS 库的信息
    def _calc_info(self, name):
        # 调用 get_info 函数获取 BLAS 库的信息
        info = get_info(name)
        # 检查获取的信息是否符合要求
        if self._check_info(info):
            # 设置类的信息属性
            self.set_info(**info)
            return True
        return False


# 定义一个名为 blas_ilp64_plain_opt_info 的类，继承自 blas_ilp64_opt_info
class blas_ilp64_plain_opt_info(blas_ilp64_opt_info):
    # BLAS 符号前缀为空字符串
    symbol_prefix = ''
    # BLAS 符号后缀为空字符串
    symbol_suffix = ''


# 定义一个名为 blas64__opt_info 的类，继承自 blas_ilp64_opt_info
class blas64__opt_info(blas_ilp64_opt_info):
    # BLAS 符号前缀为空字符串
    symbol_prefix = ''
    # BLAS 符号后缀为 '64_'
    symbol_suffix = '64_'


# 定义一个名为 cblas_info 的类，继承自 system_info
class cblas_info(system_info):
    # 指定该类对应的配置文件部分为 'cblas'
    section = 'cblas'
    # 环境变量名称为 'CBLAS'
    dir_env_var = 'CBLAS'
    # 该列表不包含任何默认值，因为它仅在 blas_info 中使用
    _lib_names = []
    # 当找不到 CBLAS 时，抛出 BlasNotFoundError 异常
    notfounderror = BlasNotFoundError


# 定义一个名为 blas_info 的类，继承自 system_info
class blas_info(system_info):
    # 指定该类对应的配置文件部分为 'blas'
    section = 'blas'
    # 环境变量名称为 'BLAS'
    dir_env_var = 'BLAS'
    # BLAS 库的名称列表为 ['blas']
    _lib_names = ['blas']
    # 当找不到 BLAS 库时，抛出 BlasNotFoundError 异常
    notfounderror = BlasNotFoundError

    # 计算 BLAS 库的信息
    def calc_info(self):
        # 获取 BLAS 库的目录列表
        lib_dirs = self.get_lib_dirs()
        # 获取 'blas_libs' 或 'libraries' 选项的单一值
        opt = self.get_option_single('blas_libs', 'libraries')
        # 获取 BLAS 库的列表
        blas_libs = self.get_libs(opt, self._lib_names)
        # 检查 BLAS 库是否存在
        info = self.check_libs(lib_dirs, blas_libs, [])
        # 如果 BLAS 库不存在，则返回空值
        if info is None:
            return
        else:
            # 设置包含目录信息
            info['include_dirs'] = self.get_include_dirs()
        
        # 如果操作系统是 Windows
        if platform.system() == 'Windows':
            # Windows 下的特定处理：get_cblas_libs 使用与编译 Python 相同的编译器，
            # 当使用 mingw 时，通常没有安装 msvc，因此需要这样的处理。
            info['language'] = 'f77'  # XXX: 这通常正确吗？
            # 如果设置了 cblas 作为选项，则使用它们
            cblas_info_obj = cblas_info()
            cblas_opt = cblas_info_obj.get_option_single('cblas_libs', 'libraries')
            cblas_libs = cblas_info_obj.get_libs(cblas_opt, None)
            if cblas_libs:
                # 将 cblas 和 blas 库合并
                info['libraries'] = cblas_libs + blas_libs
                # 定义宏，表明系统有 CBLAS
                info['define_macros'] = [('HAVE_CBLAS', None)]
        else:
            # 获取 CBLAS 库的信息
            lib = self.get_cblas_libs(info)
            if lib is not None:
                info['language'] = 'c'
                info['libraries'] = lib
                # 定义宏，表明系统有 CBLAS
                info['define_macros'] = [('HAVE_CBLAS', None)]
        
        # 设置类的信息属性
        self.set_info(**info)
    def get_cblas_libs(self, info):
        """ Check whether we can link with CBLAS interface

        This method will search through several combinations of libraries
        to check whether CBLAS is present:

        1. Libraries in ``info['libraries']``, as is
        2. As 1. but also explicitly adding ``'cblas'`` as a library
        3. As 1. but also explicitly adding ``'blas'`` as a library
        4. Check only library ``'cblas'``
        5. Check only library ``'blas'``

        Parameters
        ----------
        info : dict
           system information dictionary for compilation and linking

        Returns
        -------
        libraries : list of str or None
            a list of libraries that enables the use of CBLAS interface.
            Returns None if not found or a compilation error occurs.

            Since 1.17 returns a list.
        """
        # 使用自定义的 C 编译器对象创建一个实例
        c = customized_ccompiler()
        # 创建临时目录来存放源代码文件
        tmpdir = tempfile.mkdtemp()
        # 定义 C 源代码，包含一个简单的 CBLAS 检查
        s = textwrap.dedent("""\
            #include <cblas.h>
            int main(int argc, const char *argv[])
            {
                double a[4] = {1,2,3,4};
                double b[4] = {5,6,7,8};
                return cblas_ddot(4, a, 1, b, 1) > 10;
            }""")
        # 将源代码写入到临时文件中
        src = os.path.join(tmpdir, 'source.c')
        try:
            with open(src, 'w') as f:
                f.write(s)

            try:
                # 尝试编译源代码，检查是否能够找到头文件
                obj = c.compile([src], output_dir=tmpdir,
                                include_dirs=self.get_include_dirs())
            except (distutils.ccompiler.CompileError, distutils.ccompiler.LinkError):
                # 如果编译或链接出错，返回 None
                return None

            # 尝试链接编译后的可执行文件，检查是否能够找到 CBLAS 或 BLAS 库
            # 一些系统可能有单独的 cblas 和 blas 库
            for libs in [info['libraries'], ['cblas'] + info['libraries'],
                         ['blas'] + info['libraries'], ['cblas'], ['blas']]:
                try:
                    c.link_executable(obj, os.path.join(tmpdir, "a.out"),
                                      libraries=libs,
                                      library_dirs=info['library_dirs'],
                                      extra_postargs=info.get('extra_link_args', []))
                    # 如果成功链接，则返回找到的库列表
                    return libs
                except distutils.ccompiler.LinkError:
                    pass
        finally:
            # 最终清理临时目录
            shutil.rmtree(tmpdir)
        # 如果所有尝试都失败，则返回 None
        return None
# 定义一个名为 openblas_info 的类，继承自 blas_info 类
class openblas_info(blas_info):
    # 定义类变量 section，表示配置文件中的节名为 'openblas'
    section = 'openblas'
    # 定义环境变量的名称为 'OPENBLAS'
    dir_env_var = 'OPENBLAS'
    # 定义一个列表 _lib_names 包含字符串 'openblas'
    _lib_names = ['openblas']
    # 定义一个空列表 _require_symbols
    _require_symbols = []
    # 定义 notfounderror 异常为 BlasNotFoundError

    # 定义一个属性方法 symbol_prefix，用于获取配置文件中节 'openblas' 下的 'symbol_prefix' 值
    @property
    def symbol_prefix(self):
        try:
            return self.cp.get(self.section, 'symbol_prefix')
        except NoOptionError:
            return ''

    # 定义一个属性方法 symbol_suffix，用于获取配置文件中节 'openblas' 下的 'symbol_suffix' 值
    @property
    def symbol_suffix(self):
        try:
            return self.cp.get(self.section, 'symbol_suffix')
        except NoOptionError:
            return ''

    # 定义一个方法 _calc_info，用于计算 OpenBLAS 的信息
    def _calc_info(self):
        # 创建一个自定义 C 编译器对象 c
        c = customized_ccompiler()
        
        # 获取库目录列表 lib_dirs
        lib_dirs = self.get_lib_dirs()

        # 获取配置文件中的 openblas_libs 选项值，或者默认为 'libraries'，然后根据 _lib_names 获取库列表 openblas_libs
        opt = self.get_option_single('openblas_libs', 'libraries')
        openblas_libs = self.get_libs(opt, self._lib_names)

        # 检查库文件存在性，返回信息存储在 info 中
        info = self.check_libs(lib_dirs, openblas_libs, [])

        # 如果编译器类型为 "msvc" 且未找到 info，则尝试使用 GNU 编译器
        if c.compiler_type == "msvc" and info is None:
            from numpy.distutils.fcompiler import new_fcompiler
            f = new_fcompiler(c_compiler=c)
            if f and f.compiler_type == 'gnu95':
                # 尝试使用兼容 gfortran 的库文件
                info = self.check_msvc_gfortran_libs(lib_dirs, openblas_libs)
                # 跳过 LAPACK 检查，需要 build_ext 来完成
                skip_symbol_check = True
        # 如果找到 info，则设置 skip_symbol_check 为 False，info 的语言为 'c'
        elif info:
            skip_symbol_check = False
            info['language'] = 'c'

        # 如果未找到 info，则返回 None
        if info is None:
            return None

        # 计算 OpenBLAS 的额外信息并添加到 info 中
        extra_info = self.calc_extra_info()
        dict_append(info, **extra_info)

        # 如果未跳过符号检查且检查符号失败，则返回 None
        if not (skip_symbol_check or self.check_symbols(info)):
            return None

        # 设置 define_macros 字段，添加宏定义 'HAVE_CBLAS'
        info['define_macros'] = [('HAVE_CBLAS', None)]
        # 如果存在 symbol_prefix，则添加宏定义 'BLAS_SYMBOL_PREFIX'，值为 symbol_prefix
        if self.symbol_prefix:
            info['define_macros'] += [('BLAS_SYMBOL_PREFIX', self.symbol_prefix)]
        # 如果存在 symbol_suffix，则添加宏定义 'BLAS_SYMBOL_SUFFIX'，值为 symbol_suffix
        if self.symbol_suffix:
            info['define_macros'] += [('BLAS_SYMBOL_SUFFIX', self.symbol_suffix)]

        # 返回计算得到的 info
        return info

    # 定义 calc_info 方法，计算 OpenBLAS 的信息，并根据结果设置信息
    def calc_info(self):
        # 调用 _calc_info 方法计算信息并获取结果 info
        info = self._calc_info()
        # 如果 info 不为 None，则调用 set_info 方法，设置类的信息为 info
        if info is not None:
            self.set_info(**info)
    # 检查 MSVC 和 gfortran 库是否存在于指定的库目录中
    def check_msvc_gfortran_libs(self, library_dirs, libraries):
        # 首先，查找每个库目录下各个库文件的完整路径
        library_paths = []
        for library in libraries:
            for library_dir in library_dirs:
                # MinGW 静态库的扩展名为 .a
                fullpath = os.path.join(library_dir, library + '.a')
                if os.path.isfile(fullpath):
                    library_paths.append(fullpath)
                    break
            else:
                # 如果某个库文件不存在于任何一个库目录中，则返回 None
                return None

        # 生成 numpy.distutils 虚拟静态库文件
        basename = self.__class__.__name__
        tmpdir = os.path.join(os.getcwd(), 'build', basename)
        if not os.path.isdir(tmpdir):
            os.makedirs(tmpdir)

        # 定义包含信息的字典，包括临时目录，库名和语言类型
        info = {'library_dirs': [tmpdir],
                'libraries': [basename],
                'language': 'f77'}

        # 创建虚拟的静态库文件和 C 语言对象文件
        fake_lib_file = os.path.join(tmpdir, basename + '.fobjects')
        fake_clib_file = os.path.join(tmpdir, basename + '.cobjects')
        with open(fake_lib_file, 'w') as f:
            f.write("\n".join(library_paths))  # 将找到的库文件路径写入文件
        with open(fake_clib_file, 'w') as f:
            pass  # 创建空的 C 语言对象文件

        return info  # 返回包含信息的字典作为结果

    # 检查给定的符号是否存在
    def check_symbols(self, info):
        res = False  # 初始化结果为 False
        c = customized_ccompiler()  # 创建自定义的 C 编译器对象

        tmpdir = tempfile.mkdtemp()  # 创建临时目录用于编译和链接操作

        # 生成符号的原型声明
        prototypes = "\n".join("void %s%s%s();" % (self.symbol_prefix,
                                                   symbol_name,
                                                   self.symbol_suffix)
                               for symbol_name in self._require_symbols)
        # 生成调用符号的代码
        calls = "\n".join("%s%s%s();" % (self.symbol_prefix,
                                         symbol_name,
                                         self.symbol_suffix)
                          for symbol_name in self._require_symbols)
        # 生成完整的 C 源代码
        s = textwrap.dedent("""\
            %(prototypes)s
            int main(int argc, const char *argv[])
            {
                %(calls)s
                return 0;
            }""") % dict(prototypes=prototypes, calls=calls)
        src = os.path.join(tmpdir, 'source.c')  # 源文件路径
        out = os.path.join(tmpdir, 'a.out')  # 可执行文件路径

        # 尝试获取额外的链接参数
        try:
            extra_args = info['extra_link_args']
        except Exception:
            extra_args = []

        try:
            with open(src, 'w') as f:
                f.write(s)  # 将生成的 C 源代码写入文件
            obj = c.compile([src], output_dir=tmpdir)  # 编译源文件，得到目标文件
            try:
                # 链接目标文件生成可执行文件
                c.link_executable(obj, out, libraries=info['libraries'],
                                  library_dirs=info['library_dirs'],
                                  extra_postargs=extra_args)
                res = True  # 如果成功链接，则结果为 True
            except distutils.ccompiler.LinkError:
                res = False  # 如果链接失败，则结果为 False
        finally:
            shutil.rmtree(tmpdir)  # 最后删除临时目录及其内容

        return res  # 返回链接结果
class openblas_lapack_info(openblas_info):
    # 设置模块的名称为 'openblas'，表示与 OpenBLAS 相关的信息
    section = 'openblas'
    # 环境变量名为 'OPENBLAS'，用于指定 OpenBLAS 的安装路径
    dir_env_var = 'OPENBLAS'
    # 库的名称列表，包括 'openblas'
    _lib_names = ['openblas']
    # 要求的符号列表，包括 'zungqr_'，表明需要这些符号来确定 BLAS 的可用性
    _require_symbols = ['zungqr_']
    # 找不到 BLAS 时引发的错误类型
    notfounderror = BlasNotFoundError

class openblas_clapack_info(openblas_lapack_info):
    # 增加了 LAPACK 支持，因此基于 openblas_lapack_info
    _lib_names = ['openblas', 'lapack']

class openblas_ilp64_info(openblas_info):
    # 设置模块的名称为 'openblas_ilp64'，表示与 ILP64 版本的 OpenBLAS 相关的信息
    section = 'openblas_ilp64'
    # 环境变量名为 'OPENBLAS_ILP64'，用于指定 ILP64 版本 OpenBLAS 的安装路径
    dir_env_var = 'OPENBLAS_ILP64'
    # 库的名称列表，包括 'openblas64'
    _lib_names = ['openblas64']
    # 要求的符号列表，包括 'dgemm_', 'cblas_dgemm'，表明需要这些符号来确定 ILP64 BLAS 的可用性
    _require_symbols = ['dgemm_', 'cblas_dgemm']
    # 找不到 ILP64 BLAS 时引发的错误类型
    notfounderror = BlasILP64NotFoundError

    def _calc_info(self):
        # 调用父类方法，获取基本信息
        info = super()._calc_info()
        if info is not None:
            # 如果获取到信息，添加宏定义 'HAVE_BLAS_ILP64'
            info['define_macros'] += [('HAVE_BLAS_ILP64', None)]
        return info

class openblas_ilp64_lapack_info(openblas_ilp64_info):
    # 增加了 LAPACK 支持，基于 openblas_ilp64_info
    _require_symbols = ['dgemm_', 'cblas_dgemm', 'zungqr_', 'LAPACKE_zungqr']

    def _calc_info(self):
        # 调用父类方法，获取基本信息
        info = super()._calc_info()
        if info:
            # 如果获取到信息，添加宏定义 'HAVE_LAPACKE'
            info['define_macros'] += [('HAVE_LAPACKE', None)]
        return info

class openblas64__info(openblas_ilp64_info):
    # ILP64 版本的 OpenBLAS，使用默认的符号后缀
    section = 'openblas64_'
    dir_env_var = 'OPENBLAS64_'
    # 库的名称列表，包括 'openblas64_'
    _lib_names = ['openblas64_']
    # 符号后缀为 '64_'
    symbol_suffix = '64_'
    symbol_prefix = ''

class openblas64__lapack_info(openblas_ilp64_lapack_info, openblas64__info):
    # ILP64 版本的 OpenBLAS，增加 LAPACK 支持
    pass

class blis_info(blas_info):
    # 设置模块的名称为 'blis'，表示与 BLIS 相关的信息
    section = 'blis'
    # 环境变量名为 'BLIS'，用于指定 BLIS 的安装路径
    dir_env_var = 'BLIS'
    # 库的名称列表，包括 'blis'
    _lib_names = ['blis']
    # 找不到 BLAS 时引发的错误类型
    notfounderror = BlasNotFoundError

    def calc_info(self):
        # 获取库目录
        lib_dirs = self.get_lib_dirs()
        # 获取 blis_libs 的选项
        opt = self.get_option_single('blis_libs', 'libraries')
        # 获取 blis 的库
        blis_libs = self.get_libs(opt, self._lib_names)
        # 检查库是否存在
        info = self.check_libs2(lib_dirs, blis_libs, [])
        if info is None:
            return

        # 添加包含目录
        incl_dirs = self.get_include_dirs()
        # 将信息添加到字典中
        dict_append(info,
                    language='c',
                    define_macros=[('HAVE_CBLAS', None)],
                    include_dirs=incl_dirs)
        # 设置信息
        self.set_info(**info)


class flame_info(system_info):
    """ Usage of libflame for LAPACK operations

    This requires libflame to be compiled with lapack wrappers:

    ./configure --enable-lapack2flame ...

    Be aware that libflame 5.1.0 has some missing names in the shared library, so
    if you have problems, try the static flame library.
    """
    # 设置模块的名称为 'flame'，表示使用 libflame 进行 LAPACK 操作的信息
    section = 'flame'
    # 库的名称列表，包括 'flame'
    _lib_names = ['flame']
    # 找不到 FLAME 时引发的错误类型
    notfounderror = FlameNotFoundError
    # 检查是否存在嵌入式 LAPACK 库
    def check_embedded_lapack(self, info):
        """ libflame does not necessarily have a wrapper for fortran LAPACK, we need to check """
        # 创建一个自定义的 C 编译器对象
        c = customized_ccompiler()

        # 创建临时目录用于存放源文件
        tmpdir = tempfile.mkdtemp()

        # 定义包含 LAPACK 函数调用的简单 C 源码字符串
        s = textwrap.dedent("""\
            void zungqr_();
            int main(int argc, const char *argv[])
            {
                zungqr_();
                return 0;
            }""")
        
        # 拼接并创建源文件路径
        src = os.path.join(tmpdir, 'source.c')
        
        # 拼接并创建输出文件路径
        out = os.path.join(tmpdir, 'a.out')

        # 获取额外的链接参数，如果存在的话
        extra_args = info.get('extra_link_args', [])

        try:
            # 将 C 源码写入文件
            with open(src, 'w') as f:
                f.write(s)
            
            # 编译源文件，输出文件到指定目录
            obj = c.compile([src], output_dir=tmpdir)
            
            try:
                # 链接可执行文件，使用指定的库和目录
                c.link_executable(obj, out, libraries=info['libraries'],
                                  library_dirs=info['library_dirs'],
                                  extra_postargs=extra_args)
                # 如果链接成功，则返回 True
                return True
            except distutils.ccompiler.LinkError:
                # 如果链接失败，则返回 False
                return False
        finally:
            # 无论如何都删除临时目录及其内容
            shutil.rmtree(tmpdir)

    # 计算库信息的方法
    def calc_info(self):
        # 获取库目录
        lib_dirs = self.get_lib_dirs()
        
        # 获取特定库的信息
        flame_libs = self.get_libs('libraries', self._lib_names)

        # 调用 check_libs2 方法检查库信息
        info = self.check_libs2(lib_dirs, flame_libs, [])
        
        # 如果 info 为空，则直接返回
        if info is None:
            return

        # 计算额外的信息并将其合并到 info 中
        extra_info = self.calc_extra_info()
        dict_append(info, **extra_info)

        # 检查嵌入式 LAPACK 库是否存在
        if self.check_embedded_lapack(info):
            # 如果存在，则设置对象的信息
            self.set_info(**info)
        else:
            # 如果不存在嵌入式 LAPACK 库，则尝试获取 BLAS 库信息
            blas_info = get_info('blas_opt')
            
            # 如果 BLAS 信息不存在，则直接返回
            if not blas_info:
                # 因为之前已经失败过一次，所以这次也不会成功
                return

            # 将 BLAS 信息合并到 info 中
            for key in blas_info:
                if isinstance(blas_info[key], list):
                    info[key] = info.get(key, []) + blas_info[key]
                elif isinstance(blas_info[key], tuple):
                    info[key] = info.get(key, ()) + blas_info[key]
                else:
                    info[key] = info.get(key, '') + blas_info[key]

            # 再次检查嵌入式 LAPACK 库是否存在
            if self.check_embedded_lapack(info):
                # 如果存在，则设置对象的信息
                self.set_info(**info)
class accelerate_info(system_info):
    section = 'accelerate'
    _lib_names = ['accelerate', 'veclib']
    notfounderror = BlasNotFoundError

    def calc_info(self):
        # 从配置文件或环境变量中获取加速库名称列表
        libraries = os.environ.get('ACCELERATE')
        if libraries:
            libraries = [libraries]
        else:
            # 如果未指定加速库，则使用预定义的默认库名称列表
            libraries = self.get_libs('libraries', self._lib_names)
        # 将库名称列表中的每个名称转换为小写并去除首尾空格
        libraries = [lib.strip().lower() for lib in libraries]

        if (sys.platform == 'darwin' and
                not os.getenv('_PYTHON_HOST_PLATFORM', None)):
            # 在 macOS 下，并且没有设置 _PYTHON_HOST_PLATFORM 环境变量时
            # 根据平台信息确定是否为 Intel 架构
            if get_platform()[-4:] == 'i386' or 'intel' in get_platform() or \
               'x86_64' in get_platform() or \
               'i386' in platform.platform():
                intel = 1
            else:
                intel = 0
            # 检查系统是否有 Accelerate framework，并且 'accelerate' 在指定的库中
            if (os.path.exists('/System/Library/Frameworks'
                              '/Accelerate.framework/') and
                    'accelerate' in libraries):
                # 如果是 Intel 架构，则添加 '-msse3' 编译参数
                if intel:
                    args.extend(['-msse3'])
                # 添加 Accelerate framework 的头文件路径作为编译参数
                args.extend([
                    '-I/System/Library/Frameworks/vecLib.framework/Headers'])
                # 添加链接参数 '-framework Accelerate'
                link_args.extend(['-Wl,-framework', '-Wl,Accelerate'])
            # 检查系统是否有 vecLib framework，并且 'veclib' 在指定的库中
            elif (os.path.exists('/System/Library/Frameworks'
                                 '/vecLib.framework/') and
                      'veclib' in libraries):
                # 如果是 Intel 架构，则添加 '-msse3' 编译参数
                if intel:
                    args.extend(['-msse3'])
                # 添加 vecLib framework 的头文件路径作为编译参数
                args.extend([
                    '-I/System/Library/Frameworks/vecLib.framework/Headers'])
                # 添加链接参数 '-framework vecLib'
                link_args.extend(['-Wl,-framework', '-Wl,vecLib'])

            # 如果有任何编译参数被设置，则设置一些宏定义
            if args:
                macros = [
                    ('NO_ATLAS_INFO', 3),
                    ('HAVE_CBLAS', None),
                    ('ACCELERATE_NEW_LAPACK', None),
                ]
                # 如果设置了环境变量 NPY_USE_BLAS_ILP64，则设置额外的宏定义
                if(os.getenv('NPY_USE_BLAS_ILP64', None)):
                    print('Setting HAVE_BLAS_ILP64')
                    macros += [
                        ('HAVE_BLAS_ILP64', None),
                        ('ACCELERATE_LAPACK_ILP64', None),
                    ]
                # 设置编译和链接的附加参数以及宏定义
                self.set_info(extra_compile_args=args,
                              extra_link_args=link_args,
                              define_macros=macros)

        return

class accelerate_lapack_info(accelerate_info):
    def _calc_info(self):
        # 调用父类的 calc_info 方法
        return super()._calc_info()

class blas_src_info(system_info):
    # BLAS_SRC 已废弃，请不要使用它！
    # 通过包管理器或从源代码单独构建或安装 BLAS 库。
    section = 'blas_src'
    dir_env_var = 'BLAS_SRC'
    notfounderror = BlasSrcNotFoundError
    def get_paths(self, section, key):
        # 调用系统信息对象的方法获取指定 section 和 key 的路径列表
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        # 遍历获取的路径列表
        for d in pre_dirs:
            # 将每个路径和 'blas' 组合成新的路径列表
            dirs.extend([d] + self.combine_paths(d, ['blas']))
        # 返回所有存在的目录路径列表
        return [d for d in dirs if os.path.isdir(d)]

    def calc_info(self):
        # 获取源代码目录列表
        src_dirs = self.get_src_dirs()
        src_dir = ''
        # 遍历源代码目录列表
        for d in src_dirs:
            # 如果在当前目录下找到 'daxpy.f' 文件，则将其设置为源代码目录并中断循环
            if os.path.isfile(os.path.join(d, 'daxpy.f')):
                src_dir = d
                break
        # 如果未找到源代码目录，则返回，标记可能需要从 netlib 获取源代码
        if not src_dir:
            # XXX: 从 netlib 获取源代码，可能需要先询问
            return
        
        # 定义 BLAS 库中的三个级别的函数列表
        blas1 = '''
        caxpy csscal dnrm2 dzasum saxpy srotg zdotc ccopy cswap drot
        dznrm2 scasum srotm zdotu cdotc dasum drotg icamax scnrm2
        srotmg zdrot cdotu daxpy drotm idamax scopy sscal zdscal crotg
        dcabs1 drotmg isamax sdot sswap zrotg cscal dcopy dscal izamax
        snrm2 zaxpy zscal csrot ddot dswap sasum srot zcopy zswap
        scabs1
        '''
        blas2 = '''
        cgbmv chpmv ctrsv dsymv dtrsv sspr2 strmv zhemv ztpmv cgemv
        chpr dgbmv dsyr lsame ssymv strsv zher ztpsv cgerc chpr2 dgemv
        dsyr2 sgbmv ssyr xerbla zher2 ztrmv cgeru ctbmv dger dtbmv
        sgemv ssyr2 zgbmv zhpmv ztrsv chbmv ctbsv dsbmv dtbsv sger
        stbmv zgemv zhpr chemv ctpmv dspmv dtpmv ssbmv stbsv zgerc
        zhpr2 cher ctpsv dspr dtpsv sspmv stpmv zgeru ztbmv cher2
        ctrmv dspr2 dtrmv sspr stpsv zhbmv ztbsv
        '''
        blas3 = '''
        cgemm csymm ctrsm dsyrk sgemm strmm zhemm zsyr2k chemm csyr2k
        dgemm dtrmm ssymm strsm zher2k zsyrk cher2k csyrk dsymm dtrsm
        ssyr2k zherk ztrmm cherk ctrmm dsyr2k ssyrk zgemm zsymm ztrsm
        '''
        
        # 将 BLAS 函数列表中的每个函数名加上 '.f' 扩展名，并拼接到源代码目录下
        sources = [os.path.join(src_dir, f + '.f') \
                   for f in (blas1 + blas2 + blas3).split()]
        
        # XXX: 在此处应该检查实际存在的源文件吗？
        # 过滤出真实存在的源文件路径列表
        sources = [f for f in sources if os.path.isfile(f)]
        
        # 设置对象的信息字典，包括源文件路径列表和语言类型为 'f77'
        info = {'sources': sources, 'language': 'f77'}
        self.set_info(**info)
# 定义一个继承自system_info的类x11_info，用于获取与X11相关的系统信息
class x11_info(system_info):
    # 定义section属性为'x11'，指示该类负责处理X11相关信息
    section = 'x11'
    # 定义notfounderror属性为X11NotFoundError，表示在未找到X11时引发的错误类型
    notfounderror = X11NotFoundError
    # 定义_lib_names属性为['X11']，表示需要检查的X11库的名称列表

    def __init__(self):
        # 调用父类system_info的构造函数初始化对象
        system_info.__init__(self,
                             # 设置默认的X11库目录
                             default_lib_dirs=default_x11_lib_dirs,
                             # 设置默认的X11包含文件目录
                             default_include_dirs=default_x11_include_dirs)

    def calc_info(self):
        # 如果系统平台是Windows（win32），直接返回，不执行后续的信息计算
        if sys.platform in ['win32']:
            return
        
        # 获取库目录列表
        lib_dirs = self.get_lib_dirs()
        # 获取包含文件目录列表
        include_dirs = self.get_include_dirs()
        # 获取x11_libs选项的单一值，即要使用的X11库的名称
        opt = self.get_option_single('x11_libs', 'libraries')
        # 获取与X11相关的库文件名列表
        x11_libs = self.get_libs(opt, self._lib_names)
        # 检查X11库的存在性，并获取相关信息
        info = self.check_libs(lib_dirs, x11_libs, [])
        # 如果未找到相关信息，直接返回
        if info is None:
            return
        
        # 初始化inc_dir为None
        inc_dir = None
        # 遍历包含文件目录列表
        for d in include_dirs:
            # 如果能够组合路径d和'X11/X.h'，则认为找到了X11的包含文件目录
            if self.combine_paths(d, 'X11/X.h'):
                inc_dir = d
                break
        
        # 如果找到了X11的包含文件目录
        if inc_dir is not None:
            # 将该目录添加到info字典中的include_dirs键对应的值中
            dict_append(info, include_dirs=[inc_dir])
        
        # 使用更新后的info字典来设置对象的信息
        self.set_info(**info)


# 定义一个继承自system_info的类_numpy_info，用于获取与Numeric模块相关的系统信息
class _numpy_info(system_info):
    # 定义section属性为'Numeric'，指示该类负责处理Numeric模块相关信息
    section = 'Numeric'
    # 定义modulename属性为'Numeric'，表示要处理的Numeric模块的名称
    modulename = 'Numeric'
    # 定义notfounderror属性为NumericNotFoundError，表示在未找到Numeric模块时引发的错误类型

    def __init__(self):
        # 初始化include_dirs为空列表
        include_dirs = []
        try:
            # 尝试导入Numeric模块
            module = __import__(self.modulename)
            prefix = []
            # 遍历模块文件路径中的每个部分
            for name in module.__file__.split(os.sep):
                # 如果遇到'lib'目录，则停止遍历
                if name == 'lib':
                    break
                # 将当前部分添加到prefix列表中
                prefix.append(name)

            # 在尝试任何其他操作之前，询问Numeric模块其自身的包含文件路径
            try:
                include_dirs.append(getattr(module, 'get_include')())
            except AttributeError:
                pass

            # 将系统配置中的include路径添加到include_dirs列表中
            include_dirs.append(sysconfig.get_path('include'))
        except ImportError:
            pass
        
        # 获取Python标准库的include路径，并添加到include_dirs列表中
        py_incl_dir = sysconfig.get_path('include')
        include_dirs.append(py_incl_dir)
        # 获取Python平台相关的include路径，并确保它不在include_dirs列表中
        py_pincl_dir = sysconfig.get_path('platinclude')
        if py_pincl_dir not in include_dirs:
            include_dirs.append(py_pincl_dir)
        
        # 遍历默认的包含文件目录列表
        for d in default_include_dirs:
            # 将Python标准库的include路径添加到每个目录d后面，并确保其不在include_dirs列表中
            d = os.path.join(d, os.path.basename(py_incl_dir))
            if d not in include_dirs:
                include_dirs.append(d)
        
        # 调用父类system_info的构造函数初始化对象
        system_info.__init__(self,
                             # 设置默认的库目录为空列表
                             default_lib_dirs=[],
                             # 设置默认的包含文件目录为include_dirs列表
                             default_include_dirs=include_dirs)
    # 定义一个方法用于计算模块信息
    def calc_info(self):
        # 尝试导入指定名称的模块，如果失败则返回
        try:
            module = __import__(self.modulename)
        except ImportError:
            return
        
        # 初始化一个空字典用于存储模块信息
        info = {}
        
        # 初始化一个空列表用于存储宏定义
        macros = []
        
        # 遍历指定的版本变量列表，尝试获取模块的版本信息
        for v in ['__version__', 'version']:
            vrs = getattr(module, v, None)
            if vrs is None:
                continue
            # 构建宏定义的元组列表，包括模块名称的大写版本号和版本信息
            macros = [(self.modulename.upper() + '_VERSION',
                       _c_string_literal(vrs)),
                      (self.modulename.upper(), None)]
            break
        
        # 将获取到的宏定义列表添加到信息字典中
        dict_append(info, define_macros=macros)
        
        # 获取包含文件的目录列表
        include_dirs = self.get_include_dirs()
        
        # 初始化一个变量用于存储匹配的包含目录路径
        inc_dir = None
        
        # 遍历包含文件的目录列表，尝试找到与模块相关的头文件路径
        for d in include_dirs:
            if self.combine_paths(d,
                                  os.path.join(self.modulename,
                                               'arrayobject.h')):
                inc_dir = d
                break
        
        # 如果找到了合适的包含目录路径，则将其添加到信息字典中
        if inc_dir is not None:
            dict_append(info, include_dirs=[inc_dir])
        
        # 如果信息字典非空，则调用实例方法设置模块信息
        if info:
            self.set_info(**info)
        
        # 方法执行完毕，返回
        return
class numarray_info(_numpy_info):
    section = 'numarray'
    modulename = 'numarray'

# 设置 `numarray_info` 类的 section 和 modulename 属性，用于标识 numarray 模块信息。


class Numeric_info(_numpy_info):
    section = 'Numeric'
    modulename = 'Numeric'

# 设置 `Numeric_info` 类的 section 和 modulename 属性，用于标识 Numeric 模块信息。


class numpy_info(_numpy_info):
    section = 'numpy'
    modulename = 'numpy'

# 设置 `numpy_info` 类的 section 和 modulename 属性，用于标识 numpy 模块信息。


class numerix_info(system_info):
    section = 'numerix'

    def calc_info(self):
        which = None, None
        if os.getenv("NUMERIX"):
            which = os.getenv("NUMERIX"), "environment var"
        # 如果上述均未成功，将默认选择 numpy。
        if which[0] is None:
            which = "numpy", "defaulted"
            try:
                import numpy  # noqa: F401
                which = "numpy", "defaulted"
            except ImportError as e:
                msg1 = str(e)
                try:
                    import Numeric  # noqa: F401
                    which = "numeric", "defaulted"
                except ImportError as e:
                    msg2 = str(e)
                    try:
                        import numarray  # noqa: F401
                        which = "numarray", "defaulted"
                    except ImportError as e:
                        msg3 = str(e)
                        log.info(msg1)
                        log.info(msg2)
                        log.info(msg3)
        which = which[0].strip().lower(), which[1]
        if which[0] not in ["numeric", "numarray", "numpy"]:
            raise ValueError("numerix selector must be either 'Numeric' "
                             "or 'numarray' or 'numpy' but the value obtained"
                             " from the %s was '%s'." % (which[1], which[0]))
        os.environ['NUMERIX'] = which[0]
        self.set_info(**get_info(which[0]))

# `numerix_info` 类的 `calc_info` 方法用于确定所选的数值计算库（numeric computing library），首先尝试从环境变量 NUMERIX 获取值。如果未设置，则默认选择 numpy。如果导入 numpy、Numeric 或 numarray 中的任何一个成功，将相应模块的名称和来源设置为选择的值。如果都失败，则记录 ImportError 消息并引发异常。最终，将选择的数值计算库名称设置为环境变量 NUMERIX，并获取并设置该库的详细信息。


class f2py_info(system_info):
    def calc_info(self):
        try:
            import numpy.f2py as f2py
        except ImportError:
            return
        f2py_dir = os.path.join(os.path.dirname(f2py.__file__), 'src')
        self.set_info(sources=[os.path.join(f2py_dir, 'fortranobject.c')],
                      include_dirs=[f2py_dir])
        return

# `f2py_info` 类的 `calc_info` 方法尝试导入 numpy.f2py 模块，如果导入失败则返回。成功导入后，设置 `f2py_info` 实例的源文件和包含目录信息，以便后续使用。


class boost_python_info(system_info):
    section = 'boost_python'
    dir_env_var = 'BOOST'

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + self.combine_paths(d, ['boost*']))
        return [d for d in dirs if os.path.isdir(d)]

# `boost_python_info` 类用于获取 boost_python 模块的路径信息。`get_paths` 方法继承自 `system_info` 类，获取预设路径并将其中每个目录与匹配 'boost*' 的路径组合。最终返回所有存在的目录路径列表。
    # 定义一个方法用于计算项目的信息并设置
    def calc_info(self):
        # 获取项目的源代码目录列表
        src_dirs = self.get_src_dirs()
        # 初始化源代码目录字符串
        src_dir = ''
        # 遍历源代码目录列表
        for d in src_dirs:
            # 检查是否存在特定的文件路径
            if os.path.isfile(os.path.join(d, 'libs', 'python', 'src', 'module.cpp')):
                # 如果找到目标文件，设置源代码目录并终止循环
                src_dir = d
                break
        # 如果没有找到目标文件，则直接返回
        if not src_dir:
            return
        # 获取 Python 的包含目录路径列表
        py_incl_dirs = [sysconfig.get_path('include')]
        # 获取 Python 平台相关的包含目录路径
        py_pincl_dir = sysconfig.get_path('platinclude')
        # 如果平台包含目录路径不在包含目录路径列表中，则添加进去
        if py_pincl_dir not in py_incl_dirs:
            py_incl_dirs.append(py_pincl_dir)
        # 组合源代码目录路径和特定子目录，形成源文件目录路径
        srcs_dir = os.path.join(src_dir, 'libs', 'python', 'src')
        # 获取指定目录及其子目录下的所有 .cpp 文件路径列表
        bpl_srcs = glob(os.path.join(srcs_dir, '*.cpp'))
        bpl_srcs += glob(os.path.join(srcs_dir, '*', '*.cpp'))
        # 构建项目信息的字典，包括库和包含目录
        info = {
            'libraries': [
                ('boost_python_src', {
                    'include_dirs': [src_dir] + py_incl_dirs,
                    'sources': bpl_srcs
                })
            ],
            'include_dirs': [src_dir],
        }
        # 如果信息字典不为空，则调用对象的 set_info 方法设置信息
        if info:
            self.set_info(**info)
        # 返回方法执行结果
        return
# 定义一个名为 agg2_info 的类，继承自 system_info 类
class agg2_info(system_info):
    # 类属性 section 表示配置信息的部分为 'agg2'
    section = 'agg2'
    # 类属性 dir_env_var 表示与 agg2 相关的目录环境变量名为 'AGG2'

    # 定义一个方法 get_paths，用于获取特定配置部分和键值对应的路径列表
    def get_paths(self, section, key):
        # 调用父类的 get_paths 方法获取预设的路径列表
        pre_dirs = system_info.get_paths(self, section, key)
        # 初始化一个空列表 dirs，用于存放扩展后的路径列表
        dirs = []
        # 遍历预设路径列表 pre_dirs
        for d in pre_dirs:
            # 将每个路径 d 及其与 'agg2*' 组合后的路径添加到 dirs 中
            dirs.extend([d] + self.combine_paths(d, ['agg2*']))
        # 返回所有存在的目录路径列表，过滤掉非目录路径
        return [d for d in dirs if os.path.isdir(d)]

    # 定义一个方法 calc_info，用于计算和设置关于 agg2 的信息
    def calc_info(self):
        # 获取所有源代码目录路径列表
        src_dirs = self.get_src_dirs()
        # 初始化一个空字符串 src_dir，用于存放符合条件的源代码目录路径
        src_dir = ''
        # 遍历所有源代码目录路径 src_dirs
        for d in src_dirs:
            # 如果在当前源代码目录 d 下找到 'src/agg_affine_matrix.cpp' 文件
            if os.path.isfile(os.path.join(d, 'src', 'agg_affine_matrix.cpp')):
                # 将找到的源代码目录路径赋值给 src_dir，并结束循环
                src_dir = d
                break
        # 如果未找到符合条件的源代码目录路径，则返回空
        if not src_dir:
            return
        
        # 如果操作系统平台是 'win32'
        if sys.platform == 'win32':
            # 获取所有符合条件的源文件路径列表
            agg2_srcs = glob(os.path.join(src_dir, 'src', 'platform',
                                          'win32', 'agg_win32_bmp.cpp'))
        else:
            # 获取所有符合条件的源文件路径列表，并添加 'agg_platform_support.cpp' 路径
            agg2_srcs = glob(os.path.join(src_dir, 'src', '*.cpp'))
            agg2_srcs += [os.path.join(src_dir, 'src', 'platform',
                                       'X11',
                                       'agg_platform_support.cpp')]

        # 定义一个字典 info，包含关于 agg2 的库和包含目录信息
        info = {'libraries':
                [('agg2_src',
                  {'sources': agg2_srcs,
                   'include_dirs': [os.path.join(src_dir, 'include')],
                  }
                 )],
                'include_dirs': [os.path.join(src_dir, 'include')],
                }
        
        # 如果 info 字典非空，则调用父类的 set_info 方法设置信息
        if info:
            self.set_info(**info)
        # 返回
        return


# 定义一个名为 _pkg_config_info 的类，继承自 system_info 类
class _pkg_config_info(system_info):
    # 类属性 section 为 None，表示无特定配置部分
    section = None
    # 类属性 config_env_var 表示配置环境变量名为 'PKG_CONFIG'
    config_env_var = 'PKG_CONFIG'
    # 类属性 default_config_exe 表示默认的配置执行文件名为 'pkg-config'
    default_config_exe = 'pkg-config'
    # 类属性 append_config_exe 为空字符串，表示不追加配置执行文件名
    append_config_exe = ''
    # 类属性 version_macro_name 和 release_macro_name 为 None，表示版本和发布宏名均未指定
    version_macro_name = None
    release_macro_name = None
    # 类属性 version_flag 表示版本标志为 '--modversion'
    version_flag = '--modversion'
    # 类属性 cflags_flag 表示编译标志为 '--cflags'

    # 定义一个方法 get_config_exe，用于获取配置执行文件的路径
    def get_config_exe(self):
        # 如果配置环境变量在当前环境中已定义，则返回其值
        if self.config_env_var in os.environ:
            return os.environ[self.config_env_var]
        # 否则返回默认的配置执行文件名
        return self.default_config_exe

    # 定义一个方法 get_config_output，用于执行配置执行文件并获取配置输出
    def get_config_output(self, config_exe, option):
        # 组装命令行指令 cmd，包括配置执行文件和选项
        cmd = config_exe + ' ' + self.append_config_exe + ' ' + option
        try:
            # 执行命令行指令，获取输出结果 o
            o = subprocess.check_output(cmd)
        except (OSError, subprocess.CalledProcessError):
            # 发生异常时返回空
            pass
        else:
            # 将输出结果 o 处理为文件路径并返回
            o = filepath_from_subprocess_output(o)
            return o
    # 定义一个方法，用于计算配置信息
    def calc_info(self):
        # 查找配置可执行文件的路径
        config_exe = find_executable(self.get_config_exe())
        
        # 如果找不到配置可执行文件，记录警告日志并返回
        if not config_exe:
            log.warn('File not found: %s. Cannot determine %s info.' \
                  % (config_exe, self.section))
            return
        
        # 初始化信息字典和列表
        info = {}
        macros = []
        libraries = []
        library_dirs = []
        include_dirs = []
        extra_link_args = []
        extra_compile_args = []
        
        # 获取配置信息的版本号
        version = self.get_config_output(config_exe, self.version_flag)
        if version:
            # 将版本号转换为宏定义，并添加到宏列表中
            macros.append((self.__class__.__name__.split('.')[-1].upper(),
                           _c_string_literal(version)))
            if self.version_macro_name:
                # 如果定义了版本宏名称，将其添加到宏列表中
                macros.append((self.version_macro_name + '_%s'
                               % (version.replace('.', '_')), None))
        
        # 如果定义了发布版本宏名称，获取并添加到宏列表中
        if self.release_macro_name:
            release = self.get_config_output(config_exe, '--release')
            if release:
                macros.append((self.release_macro_name + '_%s'
                               % (release.replace('.', '_')), None))
        
        # 获取配置信息的链接选项
        opts = self.get_config_output(config_exe, '--libs')
        if opts:
            # 解析链接选项，将库名、库目录和额外链接参数分别添加到相应列表中
            for opt in opts.split():
                if opt[:2] == '-l':
                    libraries.append(opt[2:])
                elif opt[:2] == '-L':
                    library_dirs.append(opt[2:])
                else:
                    extra_link_args.append(opt)
        
        # 获取配置信息的编译选项
        opts = self.get_config_output(config_exe, self.cflags_flag)
        if opts:
            # 解析编译选项，将包含目录、宏定义和额外编译参数分别添加到相应列表中
            for opt in opts.split():
                if opt[:2] == '-I':
                    include_dirs.append(opt[2:])
                elif opt[:2] == '-D':
                    if '=' in opt:
                        n, v = opt[2:].split('=')
                        macros.append((n, v))
                    else:
                        macros.append((opt[2:], None))
                else:
                    extra_compile_args.append(opt)
        
        # 如果存在宏定义，将其添加到信息字典中
        if macros:
            dict_append(info, define_macros=macros)
        
        # 如果存在库列表，将其添加到信息字典中
        if libraries:
            dict_append(info, libraries=libraries)
        
        # 如果存在库目录列表，将其添加到信息字典中
        if library_dirs:
            dict_append(info, library_dirs=library_dirs)
        
        # 如果存在包含目录列表，将其添加到信息字典中
        if include_dirs:
            dict_append(info, include_dirs=include_dirs)
        
        # 如果存在额外链接参数列表，将其添加到信息字典中
        if extra_link_args:
            dict_append(info, extra_link_args=extra_link_args)
        
        # 如果存在额外编译参数列表，将其添加到信息字典中
        if extra_compile_args:
            dict_append(info, extra_compile_args=extra_compile_args)
        
        # 如果信息字典不为空，将信息设置到对象中
        if info:
            self.set_info(**info)
        
        # 返回方法执行结果
        return
class wx_info(_pkg_config_info):
    # 定义 wx_info 类，继承自 _pkg_config_info 类
    section = 'wx'
    # 设置 section 属性为 'wx'
    config_env_var = 'WX_CONFIG'
    # 设置 config_env_var 属性为 'WX_CONFIG'
    default_config_exe = 'wx-config'
    # 设置 default_config_exe 属性为 'wx-config'
    append_config_exe = ''
    # 设置 append_config_exe 属性为空字符串
    version_macro_name = 'WX_VERSION'
    # 设置 version_macro_name 属性为 'WX_VERSION'
    release_macro_name = 'WX_RELEASE'
    # 设置 release_macro_name 属性为 'WX_RELEASE'
    version_flag = '--version'
    # 设置 version_flag 属性为 '--version'
    cflags_flag = '--cxxflags'
    # 设置 cflags_flag 属性为 '--cxxflags'


class gdk_pixbuf_xlib_2_info(_pkg_config_info):
    # 定义 gdk_pixbuf_xlib_2_info 类，继承自 _pkg_config_info 类
    section = 'gdk_pixbuf_xlib_2'
    # 设置 section 属性为 'gdk_pixbuf_xlib_2'
    append_config_exe = 'gdk-pixbuf-xlib-2.0'
    # 设置 append_config_exe 属性为 'gdk-pixbuf-xlib-2.0'
    version_macro_name = 'GDK_PIXBUF_XLIB_VERSION'
    # 设置 version_macro_name 属性为 'GDK_PIXBUF_XLIB_VERSION'


class gdk_pixbuf_2_info(_pkg_config_info):
    # 定义 gdk_pixbuf_2_info 类，继承自 _pkg_config_info 类
    section = 'gdk_pixbuf_2'
    # 设置 section 属性为 'gdk_pixbuf_2'
    append_config_exe = 'gdk-pixbuf-2.0'
    # 设置 append_config_exe 属性为 'gdk-pixbuf-2.0'
    version_macro_name = 'GDK_PIXBUF_VERSION'
    # 设置 version_macro_name 属性为 'GDK_PIXBUF_VERSION'


class gdk_x11_2_info(_pkg_config_info):
    # 定义 gdk_x11_2_info 类，继承自 _pkg_config_info 类
    section = 'gdk_x11_2'
    # 设置 section 属性为 'gdk_x11_2'
    append_config_exe = 'gdk-x11-2.0'
    # 设置 append_config_exe 属性为 'gdk-x11-2.0'
    version_macro_name = 'GDK_X11_VERSION'
    # 设置 version_macro_name 属性为 'GDK_X11_VERSION'


class gdk_2_info(_pkg_config_info):
    # 定义 gdk_2_info 类，继承自 _pkg_config_info 类
    section = 'gdk_2'
    # 设置 section 属性为 'gdk_2'
    append_config_exe = 'gdk-2.0'
    # 设置 append_config_exe 属性为 'gdk-2.0'
    version_macro_name = 'GDK_VERSION'
    # 设置 version_macro_name 属性为 'GDK_VERSION'


class gdk_info(_pkg_config_info):
    # 定义 gdk_info 类，继承自 _pkg_config_info 类
    section = 'gdk'
    # 设置 section 属性为 'gdk'
    append_config_exe = 'gdk'
    # 设置 append_config_exe 属性为 'gdk'
    version_macro_name = 'GDK_VERSION'
    # 设置 version_macro_name 属性为 'GDK_VERSION'


class gtkp_x11_2_info(_pkg_config_info):
    # 定义 gtkp_x11_2_info 类，继承自 _pkg_config_info 类
    section = 'gtkp_x11_2'
    # 设置 section 属性为 'gtkp_x11_2'
    append_config_exe = 'gtk+-x11-2.0'
    # 设置 append_config_exe 属性为 'gtk+-x11-2.0'
    version_macro_name = 'GTK_X11_VERSION'
    # 设置 version_macro_name 属性为 'GTK_X11_VERSION'


class gtkp_2_info(_pkg_config_info):
    # 定义 gtkp_2_info 类，继承自 _pkg_config_info 类
    section = 'gtkp_2'
    # 设置 section 属性为 'gtkp_2'
    append_config_exe = 'gtk+-2.0'
    # 设置 append_config_exe 属性为 'gtk+-2.0'
    version_macro_name = 'GTK_VERSION'
    # 设置 version_macro_name 属性为 'GTK_VERSION'


class xft_info(_pkg_config_info):
    # 定义 xft_info 类，继承自 _pkg_config_info 类
    section = 'xft'
    # 设置 section 属性为 'xft'
    append_config_exe = 'xft'
    # 设置 append_config_exe 属性为 'xft'
    version_macro_name = 'XFT_VERSION'
    # 设置 version_macro_name 属性为 'XFT_VERSION'


class freetype2_info(_pkg_config_info):
    # 定义 freetype2_info 类，继承自 _pkg_config_info 类
    section = 'freetype2'
    # 设置 section 属性为 'freetype2'
    append_config_exe = 'freetype2'
    # 设置 append_config_exe 属性为 'freetype2'
    version_macro_name = 'FREETYPE2_VERSION'
    # 设置 version_macro_name 属性为 'FREETYPE2_VERSION'


class amd_info(system_info):
    # 定义 amd_info 类，继承自 system_info 类
    section = 'amd'
    # 设置 section 属性为 'amd'
    dir_env_var = 'AMD'
    # 设置 dir_env_var 属性为 'AMD'
    _lib_names = ['amd']

    def calc_info(self):
        # 定义 calc_info 方法
        lib_dirs = self.get_lib_dirs()

        opt = self.get_option_single('amd_libs', 'libraries')
        # 获取名为 'amd_libs' 的选项值
        amd_libs = self.get_libs(opt, self._lib_names)
        info = self.check_libs(lib_dirs, amd_libs, [])
        # 检查库
        if info is None:
            return

        include_dirs = self.get_include_dirs()

        inc_dir = None
        for d in include_dirs:
            p = self.combine_paths(d, 'amd.h')
            if p:
                inc_dir = os.path.dirname(p[0])
                break
        # 遍历 include_dirs，查找 amd.h 文件并获取其所在目录
        if inc_dir is not None:
            dict_append(info, include_dirs=[inc_dir],
                        define_macros=[('SCIPY_AMD_H', None)],
                        swig_opts=['-I' + inc_dir])
            # 将 include_dirs 添加到 info 字典的 'include_dirs' 键下
            # 定义宏 'SCIPY_AMD_H'
            # 设置 swig_opts 为 ['-I' + inc_dir]

        self.set_info(**info)
        # 设置信息
        return


class umfpack_info(system_info):
    # 定义 umfpack_info 类，继承自 system_info 类
    section = 'umfpack'
    # 设置 section 属性为 'umfpack'
    dir_env_var = 'UMFPACK'
    # 设置 dir_env_var 属性为 'UMFPACK'
    notfounderror = UmfpackNotFoundError
    # 设置 notfounderror 属性为 UmfpackNotFoundError
    _lib_names = ['umfpack']
    # 计算库的信息配置
    def calc_info(self):
        # 获取库的目录路径
        lib_dirs = self.get_lib_dirs()

        # 获取单个选项的 umfpack_libs，即 UMFPACK 库的名称
        opt = self.get_option_single('umfpack_libs', 'libraries')
        # 获取 UMFPACK 库的完整路径列表
        umfpack_libs = self.get_libs(opt, self._lib_names)
        # 检查给定目录下是否存在必需的库文件，并返回信息字典
        info = self.check_libs(lib_dirs, umfpack_libs, [])
        # 如果没有找到有效的库信息，直接返回
        if info is None:
            return

        # 获取包含文件的目录列表
        include_dirs = self.get_include_dirs()

        # 初始化包含文件目录为 None
        inc_dir = None
        # 遍历包含文件的目录列表
        for d in include_dirs:
            # 尝试组合路径，以找到 umfpack.h 文件的位置
            p = self.combine_paths(d, ['', 'umfpack'], 'umfpack.h')
            # 如果找到了有效路径，则获取其所在目录
            if p:
                inc_dir = os.path.dirname(p[0])
                break
        # 如果找到有效的包含文件目录，则更新 info 字典
        if inc_dir is not None:
            dict_append(info, include_dirs=[inc_dir],
                        define_macros=[('SCIPY_UMFPACK_H', None)],
                        swig_opts=['-I' + inc_dir])

        # 获取 'amd' 相关的信息并更新到 info 字典中
        dict_append(info, **get_info('amd'))

        # 将整理好的 info 字典应用为当前对象的信息配置
        self.set_info(**info)
        # 返回计算得到的信息字典
        return
# 如果参数为空，则返回空列表
def combine_paths(*args, **kws):
    r = []
    for a in args:
        # 如果参数 a 为空，则跳过本次循环
        if not a:
            continue
        # 如果参数 a 是字符串，则转换为包含单个字符串的列表
        if is_string(a):
            a = [a]
        # 将参数 a 添加到结果列表 r 中
        r.append(a)
    # 将处理过的参数列表赋值给 args
    args = r
    # 如果参数列表为空，则返回空列表
    if not args:
        return []
    # 如果参数列表长度为 1，则使用 glob 函数处理第一个参数并将结果合并
    if len(args) == 1:
        result = reduce(lambda a, b: a + b, map(glob, args[0]), [])
    # 如果参数列表长度为 2，则生成所有可能的路径组合并使用 glob 函数处理
    elif len(args) == 2:
        result = []
        for a0 in args[0]:
            for a1 in args[1]:
                result.extend(glob(os.path.join(a0, a1)))
    # 如果参数列表长度大于 2，则递归调用 combine_paths 函数生成路径组合
    else:
        result = combine_paths(*(combine_paths(args[0], args[1]) + args[2:]))
    # 记录调试信息，打印生成的路径列表
    log.debug('(paths: %s)', ','.join(result))
    # 返回最终的路径列表
    return result

# 语言到索引的映射字典
language_map = {'c': 0, 'c++': 1, 'f77': 2, 'f90': 3}
# 索引到语言的反向映射字典
inv_language_map = {0: 'c', 1: 'c++', 2: 'f77', 3: 'f90'}

# 向字典 d 中添加元素
def dict_append(d, **kws):
    # 存储语言的列表
    languages = []
    # 遍历关键字参数
    for k, v in kws.items():
        # 如果键为 'language'，则将值添加到 languages 列表中并继续下一次循环
        if k == 'language':
            languages.append(v)
            continue
        # 如果键已存在于字典 d 中
        if k in d:
            # 如果键属于特定的列表，则将新值添加到列表中（仅当新值不在列表中时）
            if k in ['library_dirs', 'include_dirs',
                     'extra_compile_args', 'extra_link_args',
                     'runtime_library_dirs', 'define_macros']:
                [d[k].append(vv) for vv in v if vv not in d[k]]
            # 否则，直接将新值扩展到原来的列表中
            else:
                d[k].extend(v)
        # 如果键不存在于字典 d 中，则直接赋予新值
        else:
            d[k] = v
    # 如果存在语言列表
    if languages:
        # 找出语言列表中最大的索引映射到语言名称
        l = inv_language_map[max([language_map.get(l, 0) for l in languages])]
        # 将语言名称添加到字典 d 中
        d['language'] = l
    return

# 解析命令行参数并返回选项和参数列表
def parseCmdLine(argv=(None,)):
    import optparse
    # 创建 OptionParser 对象
    parser = optparse.OptionParser("usage: %prog [-v] [info objs]")
    # 添加 '-v' 或 '--verbose' 选项
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose',
                      default=False,
                      help='be verbose and print more messages')
    # 解析命令行参数并返回选项和参数列表
    opts, args = parser.parse_args(args=argv[1:])
    return opts, args

# 显示所有信息类的信息
def show_all(argv=None):
    import inspect
    # 如果未提供命令行参数，则使用默认参数 sys.argv
    if argv is None:
        argv = sys.argv
    # 解析命令行参数
    opts, args = parseCmdLine(argv)
    # 如果选项中包含 verbose，则设置日志级别为 DEBUG，否则设置为 INFO
    if opts.verbose:
        log.set_threshold(log.DEBUG)
    else:
        log.set_threshold(log.INFO)
    # 仅显示特定的信息类，初始化显示列表
    show_only = []
    for n in args:
        # 如果名称不以 '_info' 结尾，则添加 '_info' 后缀
        if n[-5:] != '_info':
            n = n + '_info'
        # 将名称添加到显示列表中
        show_only.append(n)
    # 如果显示列表为空，则显示所有信息类
    show_all = not show_only
    # 复制全局变量字典
    _gdict_ = globals().copy()
    # 遍历全局变量字典中的类定义
    for name, c in _gdict_.items():
        # 如果不是类或是系统信息类，则跳过本次循环
        if not inspect.isclass(c):
            continue
        if not issubclass(c, system_info) or c is system_info:
            continue
        # 如果不是显示所有信息类，并且名称不在显示列表中，则跳过本次循环
        if not show_all:
            if name not in show_only:
                continue
            # 从显示列表中移除当前名称
            del show_only[show_only.index(name)]
        # 创建信息类实例
        conf = c()
        # 设置详细程度为 2
        conf.verbosity = 2
        # 调用 get_info 方法获取信息，打印诊断信息（不需要结果）
        conf.get_info()
    # 如果显示列表不为空，则记录信息类未定义的日志信息
    if show_only:
        log.info('Info classes not defined: %s', ','.join(show_only))

# 如果脚本直接运行，则调用 show_all 函数
if __name__ == "__main__":
    show_all()
```