# `.\numpy\numpy\distutils\command\build_src.py`

```py
# 导入必要的模块
import os
import re
import sys
import shlex
import copy

from distutils.command import build_ext
from distutils.dep_util import newer_group, newer
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsSetupError

# 导入 numpy 模块中的一些功能
# 由于 numpy 模块中的功能只有在安装后才能使用，因此不能在此处进行导入
#import numpy.f2py
from numpy.distutils import log
from numpy.distutils.misc_util import (
    fortran_ext_match, appendpath, is_string, is_sequence, get_cmd
    )
from numpy.distutils.from_template import process_file as process_f_file
from numpy.distutils.conv_template import process_file as process_c_file

# 定义函数 subst_vars，用于从源文件中替换目标文件中的变量
def subst_vars(target, source, d):
    """Substitute any occurrence of @foo@ by d['foo'] from source file into
    target."""
    var = re.compile('@([a-zA-Z_]+)@')  # 匹配目标文件中需要替换的变量
    with open(source, 'r') as fs:
        with open(target, 'w') as ft:
            for l in fs:
                m = var.search(l)  # 在每行中搜索变量
                if m:
                    ft.write(l.replace('@%s@' % m.group(1), d[m.group(1)]))  # 将目标文件中的变量替换为指定内容
                else:
                    ft.write(l)  # 写入文件

# 创建 build_src 类，继承自 build_ext.build_ext 类
class build_src(build_ext.build_ext):

    description = "build sources from SWIG, F2PY files or a function"  # 描述类的作用

    # 用户选项
    user_options = [
        ('build-src=', 'd', "directory to \"build\" sources to"),  # 构建源文件的目录
        ('f2py-opts=', None, "list of f2py command line options"),  # f2py 命令行选项
        ('swig=', None, "path to the SWIG executable"),  # SWIG 可执行文件的路径
        ('swig-opts=', None, "list of SWIG command line options"),  # SWIG 命令行选项
        ('swig-cpp', None, "make SWIG create C++ files (default is autodetected from sources)"),  # 使 SWIG 创建 C++ 文件
        ('f2pyflags=', None, "additional flags to f2py (use --f2py-opts= instead)"),  # f2py 的附加标志
        ('swigflags=', None, "additional flags to swig (use --swig-opts= instead)"),  # swig 的附加标志
        ('force', 'f', "forcibly build everything (ignore file timestamps)"),  # 强制构建所有内容，忽略文件时间戳
        ('inplace', 'i',
         "ignore build-lib and put compiled extensions into the source " +
         "directory alongside your pure Python modules"),  # 忽略 build-lib，并将编译后的扩展放入源代码目录中，与纯 Python 模块并列
        ('verbose-cfg', None,
         "change logging level from WARN to INFO which will show all " +
         "compiler output")  # 将日志级别从 WARN 更改为 INFO，显示所有编译器输出
        ]

    boolean_options = ['force', 'inplace', 'verbose-cfg']  # 布尔类型的选项

    help_options = []  # 帮助选项

    # 初始化选项
    def initialize_options(self):
        self.extensions = None
        self.package = None
        self.py_modules = None
        self.py_modules_dict = None
        self.build_src = None
        self.build_lib = None
        self.build_base = None
        self.force = None
        self.inplace = None
        self.package_dir = None
        self.f2pyflags = None  # 附加标志，已过时
        self.f2py_opts = None
        self.swigflags = None  # 附加标志，已过时
        self.swig_opts = None
        self.swig_cpp = None
        self.swig = None
        self.verbose_cfg = None
    # 完成选项设置
    def finalize_options(self):
        # 设置未定义的选项，包括 build_base、build_lib、force
        self.set_undefined_options('build',
                                   ('build_base', 'build_base'),
                                   ('build_lib', 'build_lib'),
                                   ('force', 'force'))
        # 如果 package 未指定，则使用 distribution 的 ext_package
        if self.package is None:
            self.package = self.distribution.ext_package
        # 设置 extensions 和 libraries
        self.extensions = self.distribution.ext_modules
        self.libraries = self.distribution.libraries or []
        # 设置 py_modules 和 data_files
        self.py_modules = self.distribution.py_modules or []
        self.data_files = self.distribution.data_files or []
    
        # 如果 build_src 未指定，则根据平台信息和版本信息设置 build_src
        if self.build_src is None:
            plat_specifier = ".{}-{}.{}".format(get_platform(), *sys.version_info[:2])
            self.build_src = os.path.join(self.build_base, 'src'+plat_specifier)
    
        # py_modules_dict 用于 build_py.find_package_modules
        self.py_modules_dict = {}
    
        # 处理 f2pyflags 和 f2py_opts
        if self.f2pyflags:
            if self.f2py_opts:
                log.warn('ignoring --f2pyflags as --f2py-opts already used')
            else:
                self.f2py_opts = self.f2pyflags
            self.f2pyflags = None
        if self.f2py_opts is None:
            self.f2py_opts = []
        else:
            self.f2py_opts = shlex.split(self.f2py_opts)
    
        # 处理 swigflags 和 swig_opts
        if self.swigflags:
            if self.swig_opts:
                log.warn('ignoring --swigflags as --swig-opts already used')
            else:
                self.swig_opts = self.swigflags
            self.swigflags = None
        if self.swig_opts is None:
            self.swig_opts = []
        else:
            self.swig_opts = shlex.split(self.swig_opts)
    
        # 使用 build_ext 命令的选项
        build_ext = self.get_finalized_command('build_ext')
        # 处理 inplace 和 swig_cpp
        if self.inplace is None:
            self.inplace = build_ext.inplace
        if self.swig_cpp is None:
            self.swig_cpp = build_ext.swig_cpp
        # 对于 swig 和 swig_opt，使用 build_ext 命令的选项来覆盖自身的选项
        for c in ['swig', 'swig_opt']:
            o = '--'+c.replace('_', '-')
            v = getattr(build_ext, c, None)
            if v:
                if getattr(self, c):
                    log.warn('both build_src and build_ext define %s option' % (o))
                else:
                    log.info('using "%s=%s" option from build_ext command' % (o, v))
                    setattr(self, c, v)
    
    # 运行命令
    def run(self):
        # 输出提示信息
        log.info("build_src")
        # 如果没有 extensions 或 libraries，则返回
        if not (self.extensions or self.libraries):
            return
        # 构建源文件
        self.build_sources()
    # 构建源文件
    def build_sources(self):

        # 如果设置为 inplace，则获取包的目录
        if self.inplace:
            self.get_package_dir = \
                     self.get_finalized_command('build_py').get_package_dir

        # 构建 Python 模块的源文件
        self.build_py_modules_sources()

        # 对于每个库的信息，构建库的源文件
        for libname_info in self.libraries:
            self.build_library_sources(*libname_info)

        # 如果存在扩展，检查扩展列表，然后构建每个扩展的源文件
        if self.extensions:
            self.check_extensions_list(self.extensions)

            for ext in self.extensions:
                self.build_extension_sources(ext)

        # 构建数据文件的源文件
        self.build_data_files_sources()
        # 构建 numpy 包的配置
        self.build_npy_pkg_config()

    # 构建数据文件的源文件
    def build_data_files_sources(self):
        # 如果不存在数据文件，则返回
        if not self.data_files:
            return
        # 记录日志
        log.info('building data_files sources')
        # 导入必要的模块
        from numpy.distutils.misc_util import get_data_files
        new_data_files = []
        # 对每个数据文件进行处理
        for data in self.data_files:
            # 如果数据文件为字符串，则直接加入新的数据文件列表
            if isinstance(data, str):
                new_data_files.append(data)
            # 如果数据文件为元组，则按照条件进行处理
            elif isinstance(data, tuple):
                d, files = data
                # 根据 inplace 属性判断构建目录
                if self.inplace:
                    build_dir = self.get_package_dir('.'.join(d.split(os.sep)))
                else:
                    build_dir = os.path.join(self.build_src, d)
                funcs = [f for f in files if hasattr(f, '__call__')]
                files = [f for f in files if not hasattr(f, '__call__')]
                # 对于每个函数，根据参数个数进行处理，更新文件列表
                for f in funcs:
                    if f.__code__.co_argcount==1:
                        s = f(build_dir)
                    else:
                        s = f()
                    if s is not None:
                        if isinstance(s, list):
                            files.extend(s)
                        elif isinstance(s, str):
                            files.append(s)
                        else:
                            raise TypeError(repr(s))
                # 获取所有文件名，并加入新的数据文件列表
                filenames = get_data_files((d, files))
                new_data_files.append((d, filenames))
            else:
                raise TypeError(repr(data))
        # 更新数据文件列表
        self.data_files[:] = new_data_files


    # 构建 numpy 包的配置
    def _build_npy_pkg_config(self, info, gd):
        template, install_dir, subst_dict = info
        template_dir = os.path.dirname(template)
        # 根据信息字典更新替换字典
        for k, v in gd.items():
            subst_dict[k] = v

        # 根据 inplace 属性选择生成目录
        if self.inplace == 1:
            generated_dir = os.path.join(template_dir, install_dir)
        else:
            generated_dir = os.path.join(self.build_src, template_dir,
                    install_dir)
        generated = os.path.basename(os.path.splitext(template)[0])
        generated_path = os.path.join(generated_dir, generated)
        # 如果生成目录不存在，则创建
        if not os.path.exists(generated_dir):
            os.makedirs(generated_dir)

        # 替换模板变量并生成目标文件
        subst_vars(generated_path, template, subst_dict)

        # 安装目录相对于安装前缀的路径
        full_install_dir = os.path.join(template_dir, install_dir)
        return full_install_dir, generated_path
    # 构建 npy-pkg 配置文件
    def build_npy_pkg_config(self):
        # 记录日志，表示正在构建 npy-pkg 配置文件
        log.info('build_src: building npy-pkg config files')

        # XXX: 另一个丑陋的变通方法，旨在规避 distutils 的问题。我们需要安装前缀，但是当只构建源代码时，完成安装命令的选项会导致错误。
        # 相反，我们复制安装命令实例，并完成复制，以便它不会干扰原始安装命令实例的操作方式。
        install_cmd = copy.copy(get_cmd('install'))
        if not install_cmd.finalized == 1:
            install_cmd.finalize_options()
        build_npkg = False
        if self.inplace == 1:
            top_prefix = '.'
            build_npkg = True
        elif hasattr(install_cmd, 'install_libbase'):
            top_prefix = install_cmd.install_libbase
            build_npkg = True

        if build_npkg:
            # 遍历安装的 pkg-config 配置项
            for pkg, infos in self.distribution.installed_pkg_config.items():
                pkg_path = self.distribution.package_dir[pkg]
                prefix = os.path.join(os.path.abspath(top_prefix), pkg_path)
                d = {'prefix': prefix}
                # 构建 npy-pkg 配置文件，并将生成的文件添加到数据文件中
                for info in infos:
                    install_dir, generated = self._build_npy_pkg_config(info, d)
                    self.distribution.data_files.append((install_dir, [generated]))

    # 构建 Python 模块源代码
    def build_py_modules_sources(self):
        # 如果没有 Python 模块，则返回
        if not self.py_modules:
            return
        # 记录日志，表示正在构建 Python 模块的源代码
        log.info('building py_modules sources')
        new_py_modules = []
        # 遍历 Python 模块列表
        for source in self.py_modules:
            # 如果源文件是一个序列且长度为3
            if is_sequence(source) and len(source)==3:
                package, module_base, source = source
                # 如果是就地构建
                if self.inplace:
                    build_dir = self.get_package_dir(package)
                else:
                    # 否则，构建目录为 build_src 和 package 拼接而成
                    build_dir = os.path.join(self.build_src, os.path.join(*package.split('.')))
                # 如果源是一个可调用对象
                if hasattr(source, '__call__'):
                    target = os.path.join(build_dir, module_base + '.py')
                    source = source(target)
                # 如果源为空，则跳过
                if source is None:
                    continue
                # 将模块添加到新的 Python 模块列表中
                modules = [(package, module_base, source)]
                if package not in self.py_modules_dict:
                    self.py_modules_dict[package] = []
                self.py_modules_dict[package] += modules
            else:
                # 否则，将源添加到新的 Python 模块列表中
                new_py_modules.append(source)
        self.py_modules[:] = new_py_modules
    # 构建库的源文件
    def build_library_sources(self, lib_name, build_info):
        # 获取构建信息中的源文件列表
        sources = list(build_info.get('sources', []))

        # 若源文件列表为空，直接返回
        if not sources:
            return

        # 打印构建库源文件的信息
        log.info('building library "%s" sources' % (lib_name))

        # 生成源文件
        sources = self.generate_sources(sources, (lib_name, build_info))

        # 处理源文件模板
        sources = self.template_sources(sources, (lib_name, build_info))

        # 过滤出.h文件
        sources, h_files = self.filter_h_files(sources)

        # 若存在.h文件，打印信息并结束处理
        if h_files:
            log.info('%s - nothing done with h_files = %s',
                     self.package, h_files)

        # 更新构建信息中的源文件列表
        build_info['sources'] = sources
        return

    # 构建扩展模块的源文件
    def build_extension_sources(self, ext):
        # 获取扩展模块的源文件列表
        sources = list(ext.sources)

        # 打印构建扩展模块源文件的信息
        log.info('building extension "%s" sources' % (ext.name))

        # 获取扩展模块的完整名字
        fullname = self.get_ext_fullname(ext.name)
        
        # 拆分扩展模块的完整名字
        modpath = fullname.split('.')
        package = '.'.join(modpath[0:-1])

        # 若为就地构建，设置扩展模块的目标目录
        if self.inplace:
            self.ext_target_dir = self.get_package_dir(package)

        # 生成源文件
        sources = self.generate_sources(sources, ext)
        # 处理源文件模板
        sources = self.template_sources(sources, ext)
        # 处理SWIG源文件
        sources = self.swig_sources(sources, ext)
        # 处理F2PY源文件
        sources = self.f2py_sources(sources, ext)
        # 处理Pyrex源文件
        sources = self.pyrex_sources(sources, ext)

        # 过滤出.py文件
        sources, py_files = self.filter_py_files(sources)

        # 若包不在Python模块字典中，则添加
        if package not in self.py_modules_dict:
            self.py_modules_dict[package] = []
        modules = []
        for f in py_files:
            module = os.path.splitext(os.path.basename(f))[0]
            modules.append((package, module, f))
        self.py_modules_dict[package] += modules

        # 过滤出.h文件
        sources, h_files = self.filter_h_files(sources)

        # 若存在.h文件，打印信息并结束处理
        if h_files:
            log.info('%s - nothing done with h_files = %s',
                     package, h_files)

        # 更新扩展模块的源文件列表
        ext.sources = sources
    # 生成源文件列表，根据传入的 sources 和 extension
    def generate_sources(self, sources, extension):
        # 新的源文件列表
        new_sources = []
        # 函数源文件列表
        func_sources = []
        # 遍历传入的 sources
        for source in sources:
            # 如果 source 是字符串，添加到新的源文件列表中
            if is_string(source):
                new_sources.append(source)
            else:
                # 否则添加到函数源文件列表中
                func_sources.append(source)
        # 如果没有函数源文件，直接返回新的源文件列表
        if not func_sources:
            return new_sources
        # 如果是原地处理并且 extension 不是序列，设置构建目录为扩展目标目录
        if self.inplace and not is_sequence(extension):
            build_dir = self.ext_target_dir
        else:
            if is_sequence(extension):
                name = extension[0]
            #    if 'include_dirs' not in extension[1]:
            #        extension[1]['include_dirs'] = []
            #    incl_dirs = extension[1]['include_dirs']
            else:
                name = extension.name
            #    incl_dirs = extension.include_dirs
            #if self.build_src not in incl_dirs:
            #    incl_dirs.append(self.build_src)
            build_dir = os.path.join(*([self.build_src]
                                       +name.split('.')[:-1]))
        # 创建目录
        self.mkpath(build_dir)
        # 根据 verbose_cfg 设置日志级别
        if self.verbose_cfg:
            new_level = log.INFO
        else:
            new_level = log.WARN
        # 设置日志级别，并记录原始日志级别
        old_level = log.set_threshold(new_level)
        
        # 遍历函数源文件列表
        for func in func_sources:
            # 调用函数获取源文件
            source = func(extension, build_dir)
            # 如果没有获取到源文件，继续下一轮循环
            if not source:
                continue
            # 如果获取到的是源文件列表，将每个源文件添加到新的源文件列表中
            if is_sequence(source):
                [log.info("  adding '%s' to sources." % (s,)) for s in source]
                new_sources.extend(source)
            else:
                # 否则直接添加到新的源文件列表中
                log.info("  adding '%s' to sources." % (source,))
                new_sources.append(source)
        # 恢复原始日志级别
        log.set_threshold(old_level)
        # 返回新的源文件列表
        return new_sources
    
    # 过滤出 Python 文件
    def filter_py_files(self, sources):
        return self.filter_files(sources, ['.py'])
    
    # 过滤出头文件
    def filter_h_files(self, sources):
        return self.filter_files(sources, ['.h', '.hpp', '.inc'])
    
    # 根据扩展名过滤文件
    def filter_files(self, sources, exts = []):
        # 新的源文件列表
        new_sources = []
        # 符合扩展名的文件列表
        files = []
        # 遍历传入的 sources
        for source in sources:
            # 分割文件名和扩展名
            (base, ext) = os.path.splitext(source)
            # 如果扩展名在 exts 中，添加到文件列表，否则添加到新的源文件列表
            if ext in exts:
                files.append(source)
            else:
                new_sources.append(source)
        # 返回新的源文件列表和符合扩展名的文件列表
        return new_sources, files
    # 为给定的源文件和扩展名列表生成新的源文件列表
    def template_sources(self, sources, extension):
        new_sources = []
        # 如果扩展名是一个序列
        if is_sequence(extension):
            # 获取依赖关系
            depends = extension[1].get('depends')
            # 获取包含目录
            include_dirs = extension[1].get('include_dirs')
        else:
            # 获取依赖关系
            depends = extension.depends
            # 获取包含目录
            include_dirs = extension.include_dirs
        # 遍历源文件列表
        for source in sources:
            # 将文件名拆分成基本名称和扩展名
            (base, ext) = os.path.splitext(source)
            # 如果扩展名为 .src，表示是模板文件
            if ext == '.src':
                # 如果是原地处理
                if self.inplace:
                    target_dir = os.path.dirname(base)
                else:
                    target_dir = appendpath(self.build_src, os.path.dirname(base))
                # 创建目标目录
                self.mkpath(target_dir)
                target_file = os.path.join(target_dir, os.path.basename(base))
                # 如果需要强制更新或者源文件更新时间比目标文件新
                if (self.force or newer_group([source] + depends, target_file)):
                    # 如果是 .f 文件
                    if _f_pyf_ext_match(base):
                        log.info("from_template:> %s" % (target_file))
                        # 处理 .f 文件并获得输出文本
                        outstr = process_f_file(source)
                    # 如果是 .c 文件
                    else:
                        log.info("conv_template:> %s" % (target_file))
                        # 处理 .c 文件并获得输出文本
                        outstr = process_c_file(source)
                    # 将输出文本写入目标文件
                    with open(target_file, 'w') as fid:
                        fid.write(outstr)
                # 如果目标文件是头文件
                if _header_ext_match(target_file):
                    d = os.path.dirname(target_file)
                    # 如果目录不在包含目录中
                    if d not in include_dirs:
                        log.info("  adding '%s' to include_dirs." % (d))
                        include_dirs.append(d)
                # 添加目标文件到新源文件列表
                new_sources.append(target_file)
            else:
                # 添加源文件到新源文件列表
                new_sources.append(source)
        # 返回新源文件列表
        return new_sources

    # 为给定的源文件和扩展名列表生成新的源文件列表（用于 Pyrex 文件）
    def pyrex_sources(self, sources, extension):
        """Pyrex not supported; this remains for Cython support (see below)"""
        new_sources = []
        # 获取扩展名的名称
        ext_name = extension.name.split('.')[-1]
        # 遍历源文件列表
        for source in sources:
            # 将文件名拆分成基本名称和扩展名
            (base, ext) = os.path.splitext(source)
            # 如果扩展名为 .pyx
            if ext == '.pyx':
                # 生成一个 Pyrex 源文件
                target_file = self.generate_a_pyrex_source(base, ext_name,
                                                           source,
                                                           extension)
                # 添加目标文件到新源文件列表
                new_sources.append(target_file)
            else:
                # 添加源文件到新源文件列表
                new_sources.append(source)
        # 返回新源文件列表
        return new_sources

    # 生成一个 Pyrex 源文件（这是为了兼容性而保留的方法）
    def generate_a_pyrex_source(self, base, ext_name, source, extension):
        """Pyrex is not supported, but some projects monkeypatch this method.

        That allows compiling Cython code, see gh-6955.
        This method will remain here for compatibility reasons.
        """
        return []
# 定义一个正则表达式对象，用于匹配以.f90、.f95、.f77、.for、.ftn、.f、.pyf结尾的字符串（不区分大小写）
_f_pyf_ext_match = re.compile(r'.*\.(f90|f95|f77|for|ftn|f|pyf)\Z', re.I).match
# 定义一个正则表达式对象，用于匹配以.inc、.h、.hpp结尾的字符串（不区分大小写）
_header_ext_match = re.compile(r'.*\.(inc|h|hpp)\Z', re.I).match

#### SWIG相关的辅助函数 ####
# 定义一个正则表达式对象，用于匹配SWIG模块名的语句
_swig_module_name_match = re.compile(r'\s*%module\s*(.*\(\s*package\s*=\s*"(?P<package>[\w_]+)".*\)|)\s*(?P<name>[\w_]+)',
                                     re.I).match
# 定义一个正则表达式对象，用于匹配C头文件的语句
_has_c_header = re.compile(r'-\*-\s*c\s*-\*-', re.I).search
# 定义一个正则表达式对象，用于匹配C++头文件的语句
_has_cpp_header = re.compile(r'-\*-\s*c\+\+\s*-\*-', re.I).search

# 获取SWIG的目标语言
def get_swig_target(source):
    with open(source) as f:
        result = None
        # 读取文件的第一行
        line = f.readline()
        # 如果包含C++头文件语句，设置结果为'c++'
        if _has_cpp_header(line):
            result = 'c++'
        # 如果包含C头文件语句，设置结果为'c'
        if _has_c_header(line):
            result = 'c'
    # 返回结果
    return result

# 获取SWIG模块名
def get_swig_modulename(source):
    with open(source) as f:
        name = None
        # 遍历文件的每一行
        for line in f:
            # 使用正则表达式匹配SWIG模块名
            m = _swig_module_name_match(line)
            if m:
                name = m.group('name')
                break
    # 返回模块名
    return name

# 查找SWIG目标文件
def _find_swig_target(target_dir, name):
    for ext in ['.cpp', '.c']:
        # 组合文件路径
        target = os.path.join(target_dir, '%s_wrap%s' % (name, ext))
        # 如果文件存在，终止循环
        if os.path.isfile(target):
            break
    # 返回文件路径
    return target

#### F2PY相关的辅助函数 ####
# 定义一个正则表达式对象，用于匹配python module的语句
_f2py_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]+)',
                                     re.I).match
# 定义一个正则表达式对象，用于匹配python module中含有__user__的语句
_f2py_user_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]*?'
                                          r'__user__[\w_]*)', re.I).match

# 获取F2PY模块名
def get_f2py_modulename(source):
    name = None
    with open(source) as f:
        for line in f:
            # 使用正则表达式匹配F2PY模块名
            m = _f2py_module_name_match(line)
            if m:
                # 如果包含__user__的名称，跳过
                if _f2py_user_module_name_match(line): 
                    continue
                name = m.group('name')
                break
    # 返回模块名
    return name
```