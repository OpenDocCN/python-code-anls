# `.\numpy\numpy\distutils\extension.py`

```
"""
distutils.extension

Provides the Extension class, used to describe C/C++ extension
modules in setup scripts.

Overridden to support f2py.

"""
# 导入正则表达式模块
import re
# 从distutils.extension模块导入Extension类，并起别名为old_Extension
from distutils.extension import Extension as old_Extension

# 定义一个正则表达式，匹配C++源文件的文件名后缀
cxx_ext_re = re.compile(r'.*\.(cpp|cxx|cc)\Z', re.I).match
# 定义一个正则表达式，匹配Fortran和Python语言接口文件的文件名后缀
fortran_pyf_ext_re = re.compile(r'.*\.(f90|f95|f77|for|ftn|f|pyf)\Z', re.I).match


class Extension(old_Extension):
    """
    Parameters
    ----------
    name : str
        Extension name.
    sources : list of str
        List of source file locations relative to the top directory of
        the package.
    extra_compile_args : list of str
        Extra command line arguments to pass to the compiler.
    extra_f77_compile_args : list of str
        Extra command line arguments to pass to the fortran77 compiler.
    extra_f90_compile_args : list of str
        Extra command line arguments to pass to the fortran90 compiler.
    """
    # 初始化函数，用于初始化一个扩展模块对象，设置各种编译和链接选项
    def __init__(
            self, name, sources,
            include_dirs=None,
            define_macros=None,
            undef_macros=None,
            library_dirs=None,
            libraries=None,
            runtime_library_dirs=None,
            extra_objects=None,
            extra_compile_args=None,
            extra_link_args=None,
            export_symbols=None,
            swig_opts=None,
            depends=None,
            language=None,
            f2py_options=None,
            module_dirs=None,
            extra_c_compile_args=None,
            extra_cxx_compile_args=None,
            extra_f77_compile_args=None,
            extra_f90_compile_args=None,):
    
        # 调用旧版本的 Extension 类的初始化函数，初始化基类
        old_Extension.__init__(
                self, name, [],
                include_dirs=include_dirs,
                define_macros=define_macros,
                undef_macros=undef_macros,
                library_dirs=library_dirs,
                libraries=libraries,
                runtime_library_dirs=runtime_library_dirs,
                extra_objects=extra_objects,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                export_symbols=export_symbols)
    
        # 将传入的 sources 参数赋值给当前对象的 sources 属性
        # 避免 assert 语句检查 sources 是否包含字符串类型
        self.sources = sources
    
        # 设置 swig_opts 属性，如果未指定则为空列表
        self.swig_opts = swig_opts or []
        # 如果 swig_opts 被指定为字符串而非列表，则进行警告，并将其转换为列表
        if isinstance(self.swig_opts, str):
            import warnings
            msg = "swig_opts is specified as a string instead of a list"
            warnings.warn(msg, SyntaxWarning, stacklevel=2)
            self.swig_opts = self.swig_opts.split()
    
        # 设置 depends 属性，默认为空列表
        self.depends = depends or []
        # 设置 language 属性
        self.language = language
    
        # 设置 f2py_options、module_dirs、extra_c_compile_args、extra_cxx_compile_args、
        # extra_f77_compile_args、extra_f90_compile_args 属性，默认均为空列表
        self.f2py_options = f2py_options or []
        self.module_dirs = module_dirs or []
        self.extra_c_compile_args = extra_c_compile_args or []
        self.extra_cxx_compile_args = extra_cxx_compile_args or []
        self.extra_f77_compile_args = extra_f77_compile_args or []
        self.extra_f90_compile_args = extra_f90_compile_args or []
    
        return

    # 判断当前模块是否包含 C++ 源文件
    def has_cxx_sources(self):
        # 遍历 sources 属性，判断是否包含 C++ 源文件的扩展名
        for source in self.sources:
            if cxx_ext_re(str(source)):
                return True
        return False

    # 判断当前模块是否包含用于 f2py 的源文件
    def has_f2py_sources(self):
        # 遍历 sources 属性，判断是否包含用于 f2py 的 Fortran pyf 文件
        for source in self.sources:
            if fortran_pyf_ext_re(source):
                return True
        return False
# 定义一个名为 Extension 的类
class Extension:
```