# `.\numpy\numpy\distutils\mingw32ccompiler.py`

```
"""
Support code for building Python extensions on Windows.

    # NT stuff
    # 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
    # 2. Force windows to use gcc (we're struggling with MSVC and g77 support)
    # 3. Force windows to use g77

"""
# 导入必要的库
import os                   # 导入操作系统功能模块
import sys                  # 导入系统相关功能模块
import subprocess           # 导入子进程管理模块，用于执行外部命令
import re                   # 导入正则表达式模块
import textwrap             # 导入文本包装模块

# Overwrite certain distutils.ccompiler functions:
import numpy.distutils.ccompiler  # noqa: F401  # 导入 NumPy 的 C 编译器相关功能模块，忽略 F401 警告
from numpy.distutils import log    # 导入 NumPy 日志功能模块
# NT stuff
# 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
# 2. Force windows to use gcc (we're struggling with MSVC and g77 support)
#    --> this is done in numpy/distutils/ccompiler.py
# 3. Force windows to use g77

import distutils.cygwinccompiler  # 导入 distutils Cygwin C 编译器模块
from distutils.unixccompiler import UnixCCompiler  # 导入 distutils Unix C 编译器类
from distutils.msvccompiler import get_build_version as get_build_msvc_version  # 导入 MSVC 编译器相关版本检查函数
from distutils.errors import UnknownFileError  # 导入未知文件错误类
from numpy.distutils.misc_util import (msvc_runtime_library,     # 导入 NumPy 的编译运行时库相关函数
                                       msvc_runtime_version,
                                       msvc_runtime_major,
                                       get_build_architecture)

def get_msvcr_replacement():
    """Replacement for outdated version of get_msvcr from cygwinccompiler"""
    msvcr = msvc_runtime_library()
    return [] if msvcr is None else [msvcr]

# Useful to generate table of symbols from a dll
_START = re.compile(r'\[Ordinal/Name Pointer\] Table')  # 编译正则表达式，用于匹配符号表的起始部分
_TABLE = re.compile(r'^\s+\[([\s*[0-9]*)\] ([a-zA-Z0-9_]*)')  # 编译正则表达式，用于匹配表格中的符号条目

# the same as cygwin plus some additional parameters
class Mingw32CCompiler(distutils.cygwinccompiler.CygwinCCompiler):
    """ A modified MingW32 compiler compatible with an MSVC built Python.

    """
    compiler_type = 'mingw32'  # 设置编译器类型为 'mingw32'
    def __init__ (self,
                  verbose=0,
                  dry_run=0,
                  force=0):
        # 调用父类构造函数初始化基类的属性
        distutils.cygwinccompiler.CygwinCCompiler.__init__ (self, verbose,
                                                            dry_run, force)

        # **changes: eric jones 4/11/01
        # 1. 检查在 Windows 上是否存在导入库文件。如果不存在则构建它。
        build_import_library()

        # 检查是否存在自定义的 MSVC 运行时库文件。如果不存在则构建它。
        msvcr_success = build_msvcr_library()
        msvcr_dbg_success = build_msvcr_library(debug=True)
        if msvcr_success or msvcr_dbg_success:
            # 添加预处理语句以使用自定义的 MSVC 运行时库
            self.define_macro('NPY_MINGW_USE_CUSTOM_MSVCR')

        # 为 MinGW 定义 MSVC 版本信息
        msvcr_version = msvc_runtime_version()
        if msvcr_version:
            self.define_macro('__MSVCRT_VERSION__', '0x%04i' % msvcr_version)

        # 当在 Windows 下为 amd64 架构构建时，应定义 MS_WIN64
        # Python 头文件只为 MS 编译器定义了 MS_WIN64，而这会导致一些问题，
        # 如使用 Py_ModuleInit4 而不是 Py_ModuleInit4_64 等，因此我们在这里添加它
        if get_build_architecture() == 'AMD64':
            self.set_executables(
                compiler='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall',
                compiler_so='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall '
                            '-Wstrict-prototypes',
                linker_exe='gcc -g',
                linker_so='gcc -g -shared')
        else:
            self.set_executables(
                compiler='gcc -O2 -Wall',
                compiler_so='gcc -O2 -Wall -Wstrict-prototypes',
                linker_exe='g++ ',
                linker_so='g++ -shared')
        
        # 为了支持 Python 2.3，我们单独设置 self.compiler_cxx
        # 因为在 2.2 之前的版本无法通过 set_executables 传递该参数
        self.compiler_cxx = ['g++']

        # 可能我们还应该添加 -mthreads，但这会导致生成的 DLL 需要另一个 DLL（mingwm10.dll 参见 Mingw32 文档）
        # (-mthreads: 在 Mingw32 上支持线程安全的异常处理)

        # 没有额外的库需要链接
        #self.dll_libraries=[]
        return
    def link(self,
             target_desc,
             objects,
             output_filename,
             output_dir,
             libraries,
             library_dirs,
             runtime_library_dirs,
             export_symbols = None,
             debug=0,
             extra_preargs=None,
             extra_postargs=None,
             build_temp=None,
             target_lang=None):
        # 根据 Python 使用的编译器确定要包含的 MSVC 运行时库
        runtime_library = msvc_runtime_library()
        if runtime_library:
            # 如果没有传入库列表，则初始化为空列表
            if not libraries:
                libraries = []
            # 将确定的运行时库添加到库列表中
            libraries.append(runtime_library)
        # 准备函数调用所需的参数元组
        args = (self,
                target_desc,
                objects,
                output_filename,
                output_dir,
                libraries,
                library_dirs,
                runtime_library_dirs,
                None, #export_symbols, 我们在定义文件中完成这一步骤
                debug,
                extra_preargs,
                extra_postargs,
                build_temp,
                target_lang)
        # 调用 UnixCCompiler 类的 link 方法来进行链接操作
        func = UnixCCompiler.link
        func(*args[:func.__code__.co_argcount])
        # 函数无返回值，直接结束
        return



    def object_filenames (self,
                          source_filenames,
                          strip_dir=0,
                          output_dir=''):
        # 如果未指定输出目录，则将其设为空字符串
        if output_dir is None: output_dir = ''
        # 初始化目标文件名列表为空
        obj_names = []
        # 遍历源文件名列表
        for src_name in source_filenames:
            # 使用 normcase 确保文件扩展名是正确的大小写
            (base, ext) = os.path.splitext (os.path.normcase(src_name))

            # 添加以下代码以去除 Windows 驱动器信息
            # 如果不这样做，.o 文件会被放在与 .c 文件相同的位置，而不是构建目录中
            drv, base = os.path.splitdrive(base)
            if drv:
                base = base[1:]

            # 如果文件扩展名不在源文件扩展名列表中，抛出未知文件类型异常
            if ext not in (self.src_extensions + ['.rc', '.res']):
                raise UnknownFileError(
                      "unknown file type '%s' (from '%s')" % \
                      (ext, src_name))
            # 如果要去除目录信息，则获取文件的基本名称
            if strip_dir:
                base = os.path.basename (base)
            # 如果文件扩展名是 .res 或者 .rc，将其编译成目标文件
            if ext == '.res' or ext == '.rc':
                obj_names.append (os.path.join (output_dir,
                                                base + ext + self.obj_extension))
            else:
                # 否则直接生成目标文件名并加入列表
                obj_names.append (os.path.join (output_dir,
                                                base + self.obj_extension))
        # 返回生成的目标文件名列表
        return obj_names
def find_python_dll():
    # We can't do much here:
    # - find it in the virtualenv (sys.prefix)
    # - find it in python main dir (sys.base_prefix, if in a virtualenv)
    # - in system32,
    # - otherwise (Sxs), I don't know how to get it.
    stems = [sys.prefix]
    if sys.base_prefix != sys.prefix:
        stems.append(sys.base_prefix)

    sub_dirs = ['', 'lib', 'bin']
    # generate possible combinations of directory trees and sub-directories
    lib_dirs = []
    for stem in stems:
        for folder in sub_dirs:
            lib_dirs.append(os.path.join(stem, folder))

    # add system directory as well
    if 'SYSTEMROOT' in os.environ:
        lib_dirs.append(os.path.join(os.environ['SYSTEMROOT'], 'System32'))

    # determine the Python DLL filename based on version and implementation
    major_version, minor_version = tuple(sys.version_info[:2])
    implementation = sys.implementation.name
    if implementation == 'cpython':
        dllname = f'python{major_version}{minor_version}.dll'
    elif implementation == 'pypy':
        dllname = f'libpypy{major_version}.{minor_version}-c.dll'
    else:
        dllname = f'Unknown platform {implementation}' 
    print("Looking for %s" % dllname)
    
    # search through the generated library directories for the Python DLL
    for folder in lib_dirs:
        dll = os.path.join(folder, dllname)
        if os.path.exists(dll):
            return dll

    # if the DLL is not found, raise an error
    raise ValueError("%s not found in %s" % (dllname, lib_dirs))


def dump_table(dll):
    # use objdump to get the symbol table of the given DLL
    st = subprocess.check_output(["objdump.exe", "-p", dll])
    return st.split(b'\n')


def generate_def(dll, dfile):
    """
    Given a dll file location, get all its exported symbols and dump them
    into the given def file.

    The .def file will be overwritten
    """
    # dump the symbol table from the DLL
    dump = dump_table(dll)

    # find the start of the symbol table
    for i in range(len(dump)):
        if _START.match(dump[i].decode()):
            break
    else:
        raise ValueError("Symbol table not found")

    # extract symbols from the symbol table
    syms = []
    for j in range(i + 1, len(dump)):
        m = _TABLE.match(dump[j].decode())
        if m:
            syms.append((int(m.group(1).strip()), m.group(2)))
        else:
            break

    # if no symbols are found, issue a warning
    if len(syms) == 0:
        log.warn('No symbols found in %s' % dll)

    # write the symbols into the .def file
    with open(dfile, 'w') as d:
        d.write('LIBRARY        %s\n' % os.path.basename(dll))
        d.write(';CODE          PRELOAD MOVEABLE DISCARDABLE\n')
        d.write(';DATA          PRELOAD SINGLE\n')
        d.write('\nEXPORTS\n')
        for s in syms:
            d.write('%s\n' % s[1])


def find_dll(dll_name):
    # determine the architecture and return the appropriate architecture string
    arch = {'AMD64': 'amd64',
            'Intel': 'x86'}[get_build_architecture()]
    # 在 WinSxS 目录中查找指定的 DLL 文件
    def _find_dll_in_winsxs(dll_name):
        # 获取系统的 Windows 目录（默认为 C:\WINDOWS）并拼接 WinSxS 目录路径
        winsxs_path = os.path.join(os.environ.get('WINDIR', r'C:\WINDOWS'),
                                   'winsxs')
        # 如果 WinSxS 目录不存在，则返回 None
        if not os.path.exists(winsxs_path):
            return None
        # 遍历 WinSxS 目录及其子目录
        for root, dirs, files in os.walk(winsxs_path):
            # 如果找到目标 DLL 文件且包含当前系统架构信息在路径中，则返回完整文件路径
            if dll_name in files and arch in root:
                return os.path.join(root, dll_name)
        # 如果未找到目标 DLL 文件，则返回 None
        return None

    # 在 Python 安装目录及系统 PATH 中查找指定的 DLL 文件
    def _find_dll_in_path(dll_name):
        # 首先在 Python 安装目录下查找
        for path in [sys.prefix] + os.environ['PATH'].split(';'):
            # 拼接文件路径
            filepath = os.path.join(path, dll_name)
            # 如果找到文件，则返回其绝对路径
            if os.path.exists(filepath):
                return os.path.abspath(filepath)

    # 返回在 WinSxS 目录或系统 PATH 中找到的 DLL 文件的绝对路径
    return _find_dll_in_winsxs(dll_name) or _find_dll_in_path(dll_name)
# 构建 MSVCR（Microsoft Visual C++ Runtime Library）库文件
def build_msvcr_library(debug=False):
    # 如果操作系统不是 Windows，则返回 False
    if os.name != 'nt':
        return False
    
    # 获取 MSVC runtime 的版本号
    msvcr_ver = msvc_runtime_major()
    # 如果版本号为 None，说明未找到 MSVC runtime，返回 False
    if msvcr_ver is None:
        log.debug('Skip building import library: Runtime is not compiled with MSVC')
        return False
    
    # 跳过版本小于 MSVC 8.0 的自定义库
    if msvcr_ver < 80:
        log.debug('Skip building msvcr library: custom functionality not present')
        return False
    
    # 获取 MSVC runtime 的库名称
    msvcr_name = msvc_runtime_library()
    # 如果 debug 为 True，则在库名称后添加 'd'
    if debug:
        msvcr_name += 'd'
    
    # 如果自定义库已经存在，直接返回 True
    out_name = "lib%s.a" % msvcr_name
    out_file = os.path.join(sys.prefix, 'libs', out_name)
    if os.path.isfile(out_file):
        log.debug('Skip building msvcr library: "%s" exists' % (out_file,))
        return True
    
    # 查找 msvcr.dll 文件
    msvcr_dll_name = msvcr_name + '.dll'
    dll_file = find_dll(msvcr_dll_name)
    # 如果未找到 msvcr.dll 文件，返回 False
    if not dll_file:
        log.warn('Cannot build msvcr library: "%s" not found' % msvcr_dll_name)
        return False
    
    # 生成 msvcr 库的符号定义文件
    def_name = "lib%s.def" % msvcr_name
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    log.info('Building msvcr library: "%s" (from %s)' % (out_file, dll_file))
    generate_def(dll_file, def_file)
    
    # 使用符号定义文件创建自定义 mingw 库
    cmd = ['dlltool', '-d', def_file, '-l', out_file]
    retcode = subprocess.call(cmd)
    
    # 清理符号定义文件
    os.remove(def_file)
    
    return (not retcode)

# 构建 Python 运行时的导入库
def build_import_library():
    # 如果操作系统不是 Windows，则直接返回
    if os.name != 'nt':
        return
    
    # 获取构建体系结构
    arch = get_build_architecture()
    # 根据不同的体系结构调用对应的函数进行构建
    if arch == 'AMD64':
        return _build_import_library_amd64()
    elif arch == 'Intel':
        return _build_import_library_x86()
    else:
        raise ValueError("Unhandled arch %s" % arch)

# 检查 Python 运行时的导入库是否已存在
def _check_for_import_lib():
    # 获取 Python 主版本号和次版本号
    major_version, minor_version = tuple(sys.version_info[:2])
    
    # 导入库文件名的模式
    patterns = ['libpython%d%d.a',
                'libpython%d%d.dll.a',
                'libpython%d.%d.dll.a']
    
    # 可能包含导入库的目录树
    stems = [sys.prefix]
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        stems.append(sys.base_prefix)
    elif hasattr(sys, 'real_prefix') and sys.real_prefix != sys.prefix:
        stems.append(sys.real_prefix)
    sub_dirs = ['libs', 'lib']

    # 生成候选位置列表
    candidates = []
    # 遍历模式列表
    for pat in patterns:
        # 根据当前主版本号和次版本号生成文件名
        filename = pat % (major_version, minor_version)
        # 遍历根目录列表
        for stem_dir in stems:
            # 遍历子目录列表
            for folder in sub_dirs:
                # 构建候选文件路径并添加到候选位置列表中
                candidates.append(os.path.join(stem_dir, folder, filename))

    # 检查文件系统以查找是否存在任何候选文件
    for fullname in candidates:
        if os.path.isfile(fullname):
            # 如果文件已存在于指定位置
            return (True, fullname)

    # 需要构建文件，首选位置放在候选位置列表的第一个
    return (False, candidates[0])
def _build_import_library_amd64():
    out_exists, out_file = _check_for_import_lib()
    if out_exists:
        log.debug('Skip building import library: "%s" exists', out_file)
        return

    # 获取当前正在构建导入库的运行时 DLL
    dll_file = find_python_dll()
    # 记录日志，指示正在构建导入库 (arch=AMD64)，显示相关的文件信息
    log.info('Building import library (arch=AMD64): "%s" (from %s)' %
             (out_file, dll_file))

    # 从 DLL 文件生成符号列表
    def_name = "python%d%d.def" % tuple(sys.version_info[:2])
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    generate_def(dll_file, def_file)

    # 使用符号列表生成导入库
    cmd = ['dlltool', '-d', def_file, '-l', out_file]
    subprocess.check_call(cmd)

def _build_import_library_x86():
    """ Build the import libraries for Mingw32-gcc on Windows
    """
    out_exists, out_file = _check_for_import_lib()
    if out_exists:
        log.debug('Skip building import library: "%s" exists', out_file)
        return

    # 根据 Python 版本信息生成导入库文件名
    lib_name = "python%d%d.lib" % tuple(sys.version_info[:2])
    lib_file = os.path.join(sys.prefix, 'libs', lib_name)
    if not os.path.isfile(lib_file):
        # 如果在虚拟环境中找不到库文件，尝试基本分发目录，并在那里找到使用
        # 对于 Python 2.7 的虚拟环境，基本目录是 real_prefix 而不是 base_prefix
        if hasattr(sys, 'base_prefix'):
            base_lib = os.path.join(sys.base_prefix, 'libs', lib_name)
        elif hasattr(sys, 'real_prefix'):
            base_lib = os.path.join(sys.real_prefix, 'libs', lib_name)
        else:
            base_lib = ''  # os.path.isfile('') == False

        if os.path.isfile(base_lib):
            lib_file = base_lib
        else:
            log.warn('Cannot build import library: "%s" not found', lib_file)
            return
    # 记录日志，指示正在构建导入库 (ARCH=x86)，显示相关的文件信息
    log.info('Building import library (ARCH=x86): "%s"', out_file)

    from numpy.distutils import lib2def

    # 根据 Python 版本信息生成符号定义文件名
    def_name = "python%d%d.def" % tuple(sys.version_info[:2])
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    # 使用 lib2def 生成符号列表
    nm_output = lib2def.getnm(
            lib2def.DEFAULT_NM + [lib_file], shell=False)
    dlist, flist = lib2def.parse_nm(nm_output)
    with open(def_file, 'w') as fid:
        # 将符号列表输出到定义文件中
        lib2def.output_def(dlist, flist, lib2def.DEF_HEADER, fid)

    # 查找 Python DLL 的路径
    dll_name = find_python_dll()

    # 使用 dlltool 创建导入库
    cmd = ["dlltool",
           "--dllname", dll_name,
           "--def", def_file,
           "--output-lib", out_file]
    status = subprocess.check_output(cmd)
    if status:
        log.warn('Failed to build import library for gcc. Linking will fail.')
    return

#=====================================
# Dealing with Visual Studio MANIFESTS
#=====================================

# 用于处理 Visual Studio 的清单文件的函数。清单文件是在 Windows 上强制 DLL 版本的一种机制，
# 与 distutils 的 MANIFEST 没有关系。清单文件是带有版本信息的 XML 文件，用于
# the OS loader; they are necessary when linking against a DLL not in the
# system path; in particular, official python 2.6 binary is built against the
# MS runtime 9 (the one from VS 2008), which is not available on most windows
# systems; python 2.6 installer does install it in the Win SxS (Side by side)
# directory, but this requires the manifest for this to work. This is a big
# mess, thanks MS for a wonderful system.

# XXX: ideally, we should use exactly the same version as used by python. I
# submitted a patch to get this version, but it was only included for python
# 2.6.1 and above. So for versions below, we use a "best guess".
_MSVCRVER_TO_FULLVER = {}
if sys.platform == 'win32':
    try:
        import msvcrt
        # I took one version in my SxS directory: no idea if it is the good
        # one, and we can't retrieve it from python
        _MSVCRVER_TO_FULLVER['80'] = "8.0.50727.42"
        _MSVCRVER_TO_FULLVER['90'] = "9.0.21022.8"
        # Value from msvcrt.CRT_ASSEMBLY_VERSION under Python 3.3.0
        # on Windows XP:
        _MSVCRVER_TO_FULLVER['100'] = "10.0.30319.460"
        crt_ver = getattr(msvcrt, 'CRT_ASSEMBLY_VERSION', None)
        if crt_ver is not None:  # Available at least back to Python 3.3
            maj, min = re.match(r'(\d+)\.(\d)', crt_ver).groups()
            _MSVCRVER_TO_FULLVER[maj + min] = crt_ver
            del maj, min
        del crt_ver
    except ImportError:
        # If we are here, means python was not built with MSVC. Not sure what
        # to do in that case: manifest building will fail, but it should not be
        # used in that case anyway
        log.warn('Cannot import msvcrt: using manifest will not be possible')

def msvc_manifest_xml(maj, min):
    """Given a major and minor version of the MSVCR, returns the
    corresponding XML file."""
    try:
        fullver = _MSVCRVER_TO_FULLVER[str(maj * 10 + min)]
    except KeyError:
        raise ValueError("Version %d,%d of MSVCRT not supported yet" %
                         (maj, min)) from None
    # Don't be fooled, it looks like an XML, but it is not. In particular, it
    # should not have any space before starting, and its size should be
    # divisible by 4, most likely for alignment constraints when the xml is
    # embedded in the binary...
    # This template was copied directly from the python 2.6 binary (using
    # strings.exe from mingw on python.exe).
    # 定义一个包含XML内容的模板字符串
    template = textwrap.dedent("""\
        <assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
          <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
            <security>
              <requestedPrivileges>
                <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>
              </requestedPrivileges>
            </security>
          </trustInfo>
          <dependency>
            <dependentAssembly>
              <assemblyIdentity type="win32" name="Microsoft.VC%(maj)d%(min)d.CRT" version="%(fullver)s" processorArchitecture="*" publicKeyToken="1fc8b3b9a1e18e3b"></assemblyIdentity>
            </dependentAssembly>
          </dependency>
        </assembly>""")
    
    # 使用模板字符串格式化替换，返回最终结果字符串
    return template % {'fullver': fullver, 'maj': maj, 'min': min}
# 返回用于生成嵌入为给定清单文件名的 res 文件的 rc 文件
def manifest_rc(name, type='dll'):
    """Return the rc file used to generate the res file which will be embedded
    as manifest for given manifest file name, of given type ('dll' or
    'exe').

    Parameters
    ----------
    name : str
            name of the manifest file to embed
    type : str {'dll', 'exe'}
            type of the binary which will embed the manifest

    """
    # 根据给定的类型 ('dll' or 'exe') 为给定的清单文件生成嵌入的 res 文件的 rc 文件
    if type == 'dll':
        rctype = 2
    elif type == 'exe':
        rctype = 1
    else:
        raise ValueError("Type %s not supported" % type)

    return """\
#include "winuser.h"
%d RT_MANIFEST %s""" % (rctype, name)

# 检查嵌入的 msvcr 是否与链接的 msvcr 版本匹配
def check_embedded_msvcr_match_linked(msver):
    """msver is the ms runtime version used for the MANIFEST."""
    # 检查链接和嵌入的 msvcr 主版本号是否相同
    maj = msvc_runtime_major()
    if maj:
        if not maj == int(msver):
            raise ValueError(
                  "Discrepancy between linked msvcr " \
                  "(%d) and the one about to be embedded " \
                  "(%d)" % (int(msver), maj))

# 获取配置测试的名称（包括后缀）
def configtest_name(config):
    base = os.path.basename(config._gen_temp_sourcefile("yo", [], "c"))
    return os.path.splitext(base)[0]

# 获取清单文件的名称
def manifest_name(config):
    # 获取配置测试的名称（包括后缀）
    root = configtest_name(config)
    exext = config.compiler.exe_extension
    return root + exext + ".manifest"

# 获取 rc 文件的名称
def rc_name(config):
    # 获取配置测试的名称（包括后缀）
    root = configtest_name(config)
    return root + ".rc"

# 生成清单文件
def generate_manifest(config):
    msver = get_build_msvc_version()
    if msver is not None:
        if msver >= 8:
            check_embedded_msvcr_match_linked(msver)
            ma_str, mi_str = str(msver).split('.')
            # 写入清单文件
            manxml = msvc_manifest_xml(int(ma_str), int(mi_str))
            with open(manifest_name(config), "w") as man:
                config.temp_files.append(manifest_name(config))
                man.write(manxml)
```