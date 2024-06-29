# `.\numpy\numpy\distutils\fcompiler\gnu.py`

```
# 导入正则表达式模块
import re
# 导入操作系统接口模块
import os
# 导入系统-specific 参数和功能模块
import sys
# 导入警告管理器模块
import warnings
# 导入平台相关模块
import platform
# 导入临时文件模块
import tempfile
# 导入哈希模块
import hashlib
# 导入BASE64 模块
import base64
# 导入子进程管理模块
import subprocess
# 从子进程管理模块导入进程和管道类
from subprocess import Popen, PIPE, STDOUT
# 从NumPy.distutils 模块导入文件路径和子进程输出
from numpy.distutils.exec_command import filepath_from_subprocess_output
# 从NumPy.distutils模块导入Fortran 编译器
from numpy.distutils.fcompiler import FCompiler
# 从distutils版本模块导入松散版本
from distutils.version import LooseVersion

# GnuFCompiler类，继承自FCompiler
class GnuFCompiler(FCompiler):
    # 编译器类型为gnu
    compiler_type = 'gnu'
    # 编译器别名为g77
    compiler_aliases = ('g77', )
    # 描述为GNU Fortran 77 编译器
    description = 'GNU Fortran 77 compiler'

    # GNU版本匹配方法
    def gnu_version_match(self, version_string):
        """Handle the different versions of GNU fortran compilers"""
        # 剥离可能被gfortran发出的警告
        while version_string.startswith('gfortran: warning'):
            # 从第一个换行符之后截取版本字符串并去除空格
            version_string =\
                version_string[version_string.find('\n') + 1:].strip()
        
        # 对GNU fortran不同版本进行处理
        if len(version_string) <= 20:
            # 尝试找到有效的版本字符串
            m = re.search(r'([0-9.]+)', version_string)
            if m:
                if version_string.startswith('GNU Fortran'):
                    # g77提供的版本字符串以GNU Fortran开头
                    return ('g77', m.group(1))
                elif m.start() == 0:
                    # gfortran仅输出类似#.#.#的版本字符串，因此检查匹配是否位于字符串的开头
                    return ('gfortran', m.group(1))
        else:
            # 从--version输出的版本信息，尝试更努力地获取版本信息
            m = re.search(r'GNU Fortran\s+95.*?([0-9-.]+)', version_string)
            if m:
                return ('gfortran', m.group(1))
            m = re.search(
                r'GNU Fortran.*?\-?([0-9-.]+\.[0-9-.]+)', version_string)
            if m:
                v = m.group(1)
                if v.startswith('0') or v.startswith('2') or v.startswith('3'):
                    return ('g77', v)
                else:
                    return ('gfortran', v)
        
        # 如果仍然找不到版本信息，引发异常以便找到问题
        err = 'A valid Fortran version was not found in this string:\n'
        raise ValueError(err + version_string)
    # 定义一个方法用于检查版本号是否匹配特定的 GNU 编译器
    def version_match(self, version_string):
        # 调用对象的方法检查是否匹配 GNU 版本
        v = self.gnu_version_match(version_string)
        # 如果版本不匹配或者第一个元素不是 'g77'，返回 None
        if not v or v[0] != 'g77':
            return None
        # 返回匹配的版本号
        return v[1]

    # 可能的执行文件列表，包括 'g77' 和 'f77'
    possible_executables = ['g77', 'f77']

    # 编译器和链接器等可执行文件的命令及选项
    executables = {
        'version_cmd'  : [None, "-dumpversion"],  # 获取版本号的命令
        'compiler_f77' : [None, "-g", "-Wall", "-fno-second-underscore"],  # f77 编译器的选项
        'compiler_f90' : None,  # 对于 f90 代码，使用 --fcompiler=gnu95
        'compiler_fix' : None,  # 修正编译器的选项
        'linker_so'    : [None, "-g", "-Wall"],  # 共享对象链接器的选项
        'archiver'     : ["ar", "-cr"],  # 归档工具的命令及选项
        'ranlib'       : ["ranlib"],  # ranlib 命令
        'linker_exe'   : [None, "-g", "-Wall"]  # 可执行文件链接器的选项
    }

    module_dir_switch = None  # 模块目录的开关，目前未设置
    module_include_switch = None  # 模块包含的开关，目前未设置

    # 对于非 Windows 平台且非 Cygwin Python 的情况，处理位置独立代码标志
    if os.name != 'nt' and sys.platform != 'cygwin':
        pic_flags = ['-fPIC']

    # 当 Python 运行在 Win32 平台时，为 g77 相关的几个键添加 -mno-cygwin 标志
    if sys.platform == 'win32':
        for key in ['version_cmd', 'compiler_f77', 'linker_so', 'linker_exe']:
            executables[key].append('-mno-cygwin')

    g2c = 'g2c'  # g2c 编译器的名称设为 'g2c'
    suggested_f90_compiler = 'gnu95'  # 建议使用的 f90 编译器设为 'gnu95'
    # 获取用于链接共享对象的选项
    def get_flags_linker_so(self):
        # 去除链接器选项中的第一个元素（通常是链接器名称）
        opt = self.linker_so[1:]
        # 如果运行在 macOS 上
        if sys.platform == 'darwin':
            # 获取环境变量中的 MACOSX_DEPLOYMENT_TARGET 设置
            target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', None)
            # 如果 MACOSX_DEPLOYMENT_TARGET 已设置，直接使用其值
            # 并且不作任何更改。但是，如果环境中的值与用于构建 Python 的
            # Makefile 中的值不一致，distutils 会报错。我们让 distutils
            # 处理这个错误检查。
            if not target:
                # 如果环境变量中没有设置 MACOSX_DEPLOYMENT_TARGET
                # 首先尝试从 sysconfig 中获取，然后默认设置为 10.9
                # 这是一个合理的默认值，即使使用官方的 Python 发行版
                # 或其衍生版本。
                import sysconfig
                target = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
                if not target:
                    target = '10.9'
                    s = f'Env. variable MACOSX_DEPLOYMENT_TARGET set to {target}'
                    # 发出警告，指示 MACOSX_DEPLOYMENT_TARGET 已设置为默认值
                    warnings.warn(s, stacklevel=2)
                # 将 MACOSX_DEPLOYMENT_TARGET 设置为 target
                os.environ['MACOSX_DEPLOYMENT_TARGET'] = str(target)
            # 添加特定于 macOS 的链接选项
            opt.extend(['-undefined', 'dynamic_lookup', '-bundle'])
        else:
            # 对于非 macOS 平台，添加 "-shared" 选项
            opt.append("-shared")
        # 如果运行在 Solaris 平台
        if sys.platform.startswith('sunos'):
            # SunOS 中静态库 libg2c.a 经常定义动态加载的符号
            # 链接器不能很好地处理这个问题。为了忽略这个问题，使用 -mimpure-text 标志。
            # 虽然不是最安全的方法，但似乎可以工作。gcc 的 man 手册说：
            # "相对于使用 -mimpure-text，你应该使用 -fpic 或 -fPIC 编译所有源代码。"
            opt.append('-mimpure-text')
        # 返回最终的链接器选项列表
        return opt

    # 获取 libgcc 库所在的目录
    def get_libgcc_dir(self):
        try:
            # 执行命令获取 libgcc 文件名
            output = subprocess.check_output(self.compiler_f77 +
                                            ['-print-libgcc-file-name'])
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            # 处理从子进程输出中获取的文件路径
            output = filepath_from_subprocess_output(output)
            # 返回 libgcc 文件所在的目录
            return os.path.dirname(output)
        # 如果出现异常或未能获取到输出，返回 None
        return None
    # 获取 libgfortran 库的路径
    def get_libgfortran_dir(self):
        # 如果运行在 Linux 平台上
        if sys.platform[:5] == 'linux':
            # 设置 libgfortran 库文件名
            libgfortran_name = 'libgfortran.so'
        # 如果运行在 Darwin（macOS）平台上
        elif sys.platform == 'darwin':
            # 设置 libgfortran 库文件名
            libgfortran_name = 'libgfortran.dylib'
        else:
            # 否则，平台不支持 libgfortran
            libgfortran_name = None

        # 初始化 libgfortran_dir 变量为 None
        libgfortran_dir = None
        # 如果找到了 libgfortran 库文件名
        if libgfortran_name:
            # 设置查找库文件的参数
            find_lib_arg = ['-print-file-name={0}'.format(libgfortran_name)]
            try:
                # 调用子进程执行命令查找库文件位置
                output = subprocess.check_output(
                                       self.compiler_f77 + find_lib_arg)
            except (OSError, subprocess.CalledProcessError):
                # 处理异常情况，比如命令执行错误或者子进程调用错误
                pass
            else:
                # 处理子进程输出，获取文件路径
                output = filepath_from_subprocess_output(output)
                # 获取库文件所在目录路径
                libgfortran_dir = os.path.dirname(output)
        # 返回找到的 libgfortran 库的目录路径
        return libgfortran_dir

    # 获取需要链接的库目录列表
    def get_library_dirs(self):
        # 初始化目录列表为空
        opt = []
        # 如果不是运行在 Linux 平台
        if sys.platform[:5] != 'linux':
            # 获取 libgcc 库的目录
            d = self.get_libgcc_dir()
            # 如果成功获取到 libgcc 库目录
            if d:
                # 如果运行在 Windows 平台且不是使用 cygwin
                if sys.platform == 'win32' and not d.startswith('/usr/lib'):
                    # 规范化目录路径
                    d = os.path.normpath(d)
                    # 构造 libg2c 库文件路径
                    path = os.path.join(d, "lib%s.a" % self.g2c)
                    # 如果路径不存在
                    if not os.path.exists(path):
                        # 计算根目录路径
                        root = os.path.join(d, *((os.pardir, ) * 4))
                        # 构造第二个可能的 lib 目录路径
                        d2 = os.path.abspath(os.path.join(root, 'lib'))
                        # 构造 libg2c 库文件路径
                        path = os.path.join(d2, "lib%s.a" % self.g2c)
                        # 如果路径存在
                        if os.path.exists(path):
                            # 将第二个 lib 目录路径添加到选项列表中
                            opt.append(d2)
                # 将 libgcc 库目录路径添加到选项列表中
                opt.append(d)
        # 获取 libgfortran 库的目录路径
        lib_gfortran_dir = self.get_libgfortran_dir()
        # 如果成功获取到 libgfortran 库目录路径
        if lib_gfortran_dir:
            # 将 libgfortran 库目录路径添加到选项列表中
            opt.append(lib_gfortran_dir)
        # 返回最终的选项列表
        return opt

    # 获取需要链接的库列表
    def get_libraries(self):
        # 初始化库列表为空
        opt = []
        # 获取 libgcc 库的目录路径
        d = self.get_libgcc_dir()
        # 如果成功获取到 libgcc 库目录路径
        if d is not None:
            # 设置 libg2c 的版本后缀
            g2c = self.g2c + '-pic'
            # 构造静态库的格式
            f = self.static_lib_format % (g2c, self.static_lib_extension)
            # 如果静态库文件不存在
            if not os.path.isfile(os.path.join(d, f)):
                # 重置 libg2c 的版本后缀
                g2c = self.g2c
        else:
            # 如果未获取到 libgcc 库目录路径，设置 libg2c 的版本
            g2c = self.g2c

        # 如果成功设置了 libg2c 版本
        if g2c is not None:
            # 将 libg2c 库版本添加到选项列表中
            opt.append(g2c)
        # 获取 C 编译器类型
        c_compiler = self.c_compiler
        # 如果运行在 Windows 平台且 C 编译器类型是 MSVC
        if sys.platform == 'win32' and c_compiler and \
                c_compiler.compiler_type == 'msvc':
            # 添加 gcc 到选项列表中
            opt.append('gcc')
        # 如果运行在 Darwin（macOS）平台
        if sys.platform == 'darwin':
            # 添加 cc_dynamic 到选项列表中
            opt.append('cc_dynamic')
        # 返回最终的选项列表
        return opt

    # 获取调试标志列表
    def get_flags_debug(self):
        # 返回调试标志列表 ['-g']
        return ['-g']

    # 获取优化标志列表
    def get_flags_opt(self):
        # 获取当前编译器版本
        v = self.get_version()
        # 如果版本存在且小于等于 '3.3.3'
        if v and v <= '3.3.3':
            # 对于该编译器版本，使用 '-O2' 优化标志
            opt = ['-O2']
        else:
            # 否则，使用 '-O3' 优化标志
            opt = ['-O3']
        # 添加 '-funroll-loops' 优化标志
        opt.append('-funroll-loops')
        # 返回最终的优化标志列表
        return opt
    # 返回从 CFLAGS 中检测到的架构标志
    def _c_arch_flags(self):
        """ Return detected arch flags from CFLAGS """
        import sysconfig
        try:
            # 获取 Python 的编译配置中的 CFLAGS
            cflags = sysconfig.get_config_vars()['CFLAGS']
        except KeyError:
            # 如果找不到 CFLAGS，则返回空列表
            return []
        # 匹配 "-arch" 后面跟随的架构标志，存入 arch_flags 列表
        arch_re = re.compile(r"-arch\s+(\w+)")
        arch_flags = []
        for arch in arch_re.findall(cflags):
            # 将每个架构标志转换为 ['-arch', arch] 的形式，添加到 arch_flags 中
            arch_flags += ['-arch', arch]
        return arch_flags

    # 返回空列表，占位函数，暂无具体实现
    def get_flags_arch(self):
        return []

    # 返回指定目录的运行时库目录选项
    def runtime_library_dir_option(self, dir):
        if sys.platform == 'win32' or sys.platform == 'cygwin':
            # Windows 和 Cygwin 平台不支持 RPATH，抛出未实现错误
            raise NotImplementedError

        # 断言目录字符串中不包含逗号
        assert "," not in dir

        if sys.platform == 'darwin':
            # macOS 平台使用 -Wl,-rpath,dir 作为运行时库目录选项
            return f'-Wl,-rpath,{dir}'
        elif sys.platform.startswith(('aix', 'os400')):
            # AIX 平台和 OS/400 平台使用 -Wl,-blibpath:dir 作为运行时库目录选项
            return f'-Wl,-blibpath:{dir}'
        else:
            # 其他 Unix-like 平台使用 -Wl,-rpath=dir 作为运行时库目录选项
            return f'-Wl,-rpath={dir}'
class Gnu95FCompiler(GnuFCompiler):
    # 指定编译器类型为 gnu95
    compiler_type = 'gnu95'
    # 设置编译器的别名为 gfortran
    compiler_aliases = ('gfortran', )
    # 描述信息为 GNU Fortran 95 编译器
    description = 'GNU Fortran 95 compiler'

    def version_match(self, version_string):
        # 调用父类方法，检查版本字符串是否匹配
        v = self.gnu_version_match(version_string)
        # 如果版本信息不匹配或者不是 gfortran，返回 None
        if not v or v[0] != 'gfortran':
            return None
        v = v[1]
        # 如果版本大于等于 "4"
        if LooseVersion(v) >= "4":
            # 对于 gcc-4 系列的版本不支持 -mno-cygwin 选项
            pass
        else:
            # 当 Python 不是 Cygwin-Python 时，在 win32 平台上添加 -mno-cygwin 标志
            if sys.platform == 'win32':
                for key in [
                        'version_cmd', 'compiler_f77', 'compiler_f90',
                        'compiler_fix', 'linker_so', 'linker_exe'
                ]:
                    self.executables[key].append('-mno-cygwin')
        return v

    # 可能的可执行文件列表
    possible_executables = ['gfortran', 'f95']
    # 不同类型的编译器及其选项
    executables = {
        'version_cmd'  : ["<F90>", "-dumpversion"],
        'compiler_f77' : [None, "-Wall", "-g", "-ffixed-form",
                          "-fno-second-underscore"],
        'compiler_f90' : [None, "-Wall", "-g",
                          "-fno-second-underscore"],
        'compiler_fix' : [None, "-Wall",  "-g","-ffixed-form",
                          "-fno-second-underscore"],
        'linker_so'    : ["<F90>", "-Wall", "-g"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"],
        'linker_exe'   : [None, "-Wall"]
    }

    # 模块目录选项
    module_dir_switch = '-J'
    # 模块包含选项
    module_include_switch = '-I'

    # 如果系统平台以 ('aix', 'os400') 开头
    if sys.platform.startswith(('aix', 'os400')):
        # 向 linker_so 执行文件添加 -lpthread 标志
        executables['linker_so'].append('-lpthread')
        # 如果平台架构是 64 位
        if platform.architecture()[0][:2] == '64':
            # 对于多个关键字，向可执行文件添加 -maix64 标志
            for key in ['compiler_f77', 'compiler_f90','compiler_fix','linker_so', 'linker_exe']:
                executables[key].append('-maix64')

    # g2c 变量设置为 'gfortran'
    g2c = 'gfortran'

    def _universal_flags(self, cmd):
        """Return a list of -arch flags for every supported architecture."""
        # 如果不是在 Darwin 平台上，返回空列表
        if not sys.platform == 'darwin':
            return []
        arch_flags = []
        # 获取 C 编译器支持的架构标志
        c_archs = self._c_arch_flags()
        # 如果 "i386" 在 C 编译器支持的架构中，将其替换为 "i686"
        if "i386" in c_archs:
            c_archs[c_archs.index("i386")] = "i686"
        # 检查 Fortran 编译器支持的架构，并与 C 编译器的进行比较
        for arch in ["ppc", "i686", "x86_64", "ppc64", "s390x"]:
            # 如果 cmd 可以以 arch 作为目标，并且 arch 在 C 编译器的架构中
            if _can_target(cmd, arch) and arch in c_archs:
                # 添加 -arch arch 标志到 arch_flags 列表中
                arch_flags.extend(["-arch", arch])
        return arch_flags

    def get_flags(self):
        # 调用父类方法获取 flags 列表
        flags = GnuFCompiler.get_flags(self)
        # 获取通用的架构标志
        arch_flags = self._universal_flags(self.compiler_f90)
        # 如果有通用的架构标志，将其插入到 flags 列表的开头
        if arch_flags:
            flags[:0] = arch_flags
        return flags

    def get_flags_linker_so(self):
        # 调用父类方法获取 linker_so 的 flags 列表
        flags = GnuFCompiler.get_flags_linker_so(self)
        # 获取通用的架构标志
        arch_flags = self._universal_flags(self.linker_so)
        # 如果有通用的架构标志，将其插入到 flags 列表的开头
        if arch_flags:
            flags[:0] = arch_flags
        return flags
    # 获得编译器库目录的方法
    def get_library_dirs(self):
        # 调用父类方法获取初始选项列表
        opt = GnuFCompiler.get_library_dirs(self)
        # 如果运行平台是 Windows
        if sys.platform == 'win32':
            # 获取 C 编译器
            c_compiler = self.c_compiler
            # 如果存在 C 编译器且编译器类型是 MSVC
            if c_compiler and c_compiler.compiler_type == "msvc":
                # 获取目标平台
                target = self.get_target()
                if target:
                    # 获取 libgcc 目录，并计算 mingw 目录
                    d = os.path.normpath(self.get_libgcc_dir())
                    root = os.path.join(d, *((os.pardir, ) * 4))
                    path = os.path.join(root, "lib")
                    mingwdir = os.path.normpath(path)
                    # 如果 mingw 目录下存在 libmingwex.a 文件，则添加到选项列表中
                    if os.path.exists(os.path.join(mingwdir, "libmingwex.a")):
                        opt.append(mingwdir)
        # 对于 Macports / Linux，libgfortran 和 libgcc 不是共存的
        # 获取 libgfortran 目录，如果存在则添加到选项列表中
        lib_gfortran_dir = self.get_libgfortran_dir()
        if lib_gfortran_dir:
            opt.append(lib_gfortran_dir)
        # 返回最终的选项列表
        return opt

    # 获取链接库的方法
    def get_libraries(self):
        # 调用父类方法获取初始选项列表
        opt = GnuFCompiler.get_libraries(self)
        # 如果运行平台是 Darwin (MacOS)
        if sys.platform == 'darwin':
            # 移除 cc_dynamic 库
            opt.remove('cc_dynamic')
        # 如果运行平台是 Windows
        if sys.platform == 'win32':
            # 获取 C 编译器
            c_compiler = self.c_compiler
            # 如果存在 C 编译器且编译器类型是 MSVC
            if c_compiler and c_compiler.compiler_type == "msvc":
                # 如果 opt 列表中包含 "gcc"，则在其后插入 "mingw32" 和 "mingwex"
                if "gcc" in opt:
                    i = opt.index("gcc")
                    opt.insert(i + 1, "mingwex")
                    opt.insert(i + 1, "mingw32")
            # 如果是 MSVC 编译器，返回空列表
            if c_compiler and c_compiler.compiler_type == "msvc":
                return []
            else:
                pass
        # 返回最终的选项列表
        return opt

    # 获取编译目标的方法
    def get_target(self):
        try:
            # 启动子进程执行编译器命令，并捕获输出
            p = subprocess.Popen(
                self.compiler_f77 + ['-v'],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = p.communicate()
            output = (stdout or b"") + (stderr or b"")
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            # 处理子进程输出的文件路径，通过正则表达式匹配目标平台信息
            output = filepath_from_subprocess_output(output)
            m = TARGET_R.search(output)
            if m:
                return m.group(1)
        # 如果没有匹配到目标平台信息，则返回空字符串
        return ""

    # 计算文件的哈希值的方法
    def _hash_files(self, filenames):
        # 创建 SHA1 哈希对象
        h = hashlib.sha1()
        # 遍历文件名列表
        for fn in filenames:
            # 以二进制读取文件内容并更新哈希对象
            with open(fn, 'rb') as f:
                while True:
                    block = f.read(131072)
                    if not block:
                        break
                    h.update(block)
        # 将哈希值进行 Base32 编码，并转换为 ASCII 字符串格式
        text = base64.b32encode(h.digest())
        text = text.decode('ascii')
        # 去除编码结果末尾的 '=' 符号并返回
        return text.rstrip('=')
    def _link_wrapper_lib(self, objects, output_dir, extra_dll_dir,
                          chained_dlls, is_archive):
        """Create a wrapper shared library for the given objects

        Return an MSVC-compatible lib
        """

        c_compiler = self.c_compiler
        # 检查编译器类型是否为 MSVC，否则抛出数值错误异常
        if c_compiler.compiler_type != "msvc":
            raise ValueError("This method only supports MSVC")

        # 计算对象文件和链式 DLL 的哈希值
        object_hash = self._hash_files(list(objects) + list(chained_dlls))

        # 根据系统位数确定标签
        if is_win64():
            tag = 'win_amd64'
        else:
            tag = 'win32'

        # 根据对象的基本名称和哈希值构造库文件名、动态链接库名和定义文件名
        basename = 'lib' + os.path.splitext(
            os.path.basename(objects[0]))[0][:8]
        root_name = basename + '.' + object_hash + '.gfortran-' + tag
        dll_name = root_name + '.dll'
        def_name = root_name + '.def'
        lib_name = root_name + '.lib'

        # 构造动态链接库、定义文件和库文件的路径
        dll_path = os.path.join(extra_dll_dir, dll_name)
        def_path = os.path.join(output_dir, def_name)
        lib_path = os.path.join(output_dir, lib_name)

        # 如果库文件已存在，则直接返回路径
        if os.path.isfile(lib_path):
            # Nothing to do
            return lib_path, dll_path

        # 如果需要创建归档文件，则修改对象列表
        if is_archive:
            objects = (["-Wl,--whole-archive"] + list(objects) +
                       ["-Wl,--no-whole-archive"])

        # 调用链接共享对象的方法，传入对象列表和相关参数
        self.link_shared_object(
            objects,
            dll_name,
            output_dir=extra_dll_dir,
            extra_postargs=list(chained_dlls) + [
                '-Wl,--allow-multiple-definition',
                '-Wl,--output-def,' + def_path,
                '-Wl,--export-all-symbols',
                '-Wl,--enable-auto-import',
                '-static',
                '-mlong-double-64',
            ])

        # 如果系统为 Win64，则指定 /MACHINE:X64，否则为 /MACHINE:X86
        if is_win64():
            specifier = '/MACHINE:X64'
        else:
            specifier = '/MACHINE:X86'

        # MSVC 特定的库参数列表
        lib_args = ['/def:' + def_path, '/OUT:' + lib_path, specifier]

        # 如果编译器尚未初始化，则进行初始化
        if not c_compiler.initialized:
            c_compiler.initialize()

        # 使用编译器的 spawn 方法调用 lib 命令来生成库文件
        c_compiler.spawn([c_compiler.lib] + lib_args)

        # 返回生成的库文件和动态链接库的路径
        return lib_path, dll_path

    def can_ccompiler_link(self, compiler):
        # 判断给定的编译器是否为 MSVC，如果是则返回 False，否则返回 True
        # 表示 MSVC 不能链接由 GNU fortran 编译的对象
        return compiler.compiler_type not in ("msvc", )
    # 将不兼容默认链接器的一组对象文件转换为与之兼容的文件。
    def wrap_unlinkable_objects(self, objects, output_dir, extra_dll_dir):
        """
        Convert a set of object files that are not compatible with the default
        linker, to a file that is compatible.
        """
        # 检查编译器类型是否为 MSVC
        if self.c_compiler.compiler_type == "msvc":
            # 编译一个 DLL 并返回 DLL 的 lib 作为对象。
            # 同时跟踪之前编译的 DLL，以便可以链接到它们。

            # 如果有 .a 归档文件，假设它们是独立的静态库，并为每个构建单独的 DLL。
            archives = []
            plain_objects = []
            for obj in objects:
                if obj.lower().endswith('.a'):
                    archives.append(obj)
                else:
                    plain_objects.append(obj)

            chained_libs = []
            chained_dlls = []
            # 反向遍历归档文件，以处理依赖顺序
            for archive in archives[::-1]:
                lib, dll = self._link_wrapper_lib(
                    [archive],
                    output_dir,
                    extra_dll_dir,
                    chained_dlls=chained_dlls,
                    is_archive=True)
                chained_libs.insert(0, lib)  # 在列表开头插入 lib
                chained_dlls.insert(0, dll)  # 在列表开头插入 dll

            if not plain_objects:
                return chained_libs  # 返回链接的库列表

            # 处理普通对象文件，构建相应的 DLL
            lib, dll = self._link_wrapper_lib(
                plain_objects,
                output_dir,
                extra_dll_dir,
                chained_dlls=chained_dlls,
                is_archive=False)
            return [lib] + chained_libs  # 返回链接的库列表，包括当前构建的库
        else:
            raise ValueError("Unsupported C compiler")
# 如果指定的体系结构支持 -arch 标志，则返回 True
def _can_target(cmd, arch):
    # 复制命令列表，避免修改原始命令
    newcmd = cmd[:]
    # 创建一个临时文件以 .f 为后缀
    fid, filename = tempfile.mkstemp(suffix=".f")
    os.close(fid)
    try:
        # 获取临时文件的目录
        d = os.path.dirname(filename)
        # 根据文件名生成输出文件名，将 .f 后缀改为 .o
        output = os.path.splitext(filename)[0] + ".o"
        try:
            # 向命令列表添加 -arch、体系结构、-c 和文件名参数
            newcmd.extend(["-arch", arch, "-c", filename])
            # 在指定目录中启动子进程，将标准错误输出到标准输出，捕获输出
            p = Popen(newcmd, stderr=STDOUT, stdout=PIPE, cwd=d)
            p.communicate()
            # 返回子进程的退出代码是否为 0，即命令执行成功
            return p.returncode == 0
        finally:
            # 如果生成了输出文件，删除它
            if os.path.exists(output):
                os.remove(output)
    finally:
        # 删除临时创建的文件
        os.remove(filename)


if __name__ == '__main__':
    # 导入日志模块
    from distutils import log
    # 导入自定义的 Fortran 编译器
    from numpy.distutils import customized_fcompiler
    # 设置日志的详细程度为 2
    log.set_verbosity(2)

    # 打印 GNU 编译器的版本信息
    print(customized_fcompiler('gnu').get_version())
    try:
        # 尝试打印 g95 编译器的版本信息，捕获可能的异常
        print(customized_fcompiler('g95').get_version())
    except Exception as e:
        # 打印捕获的异常对象
        print(e)
```