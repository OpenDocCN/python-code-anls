# `D:\src\scipysrc\sympy\sympy\utilities\_compilation\runners.py`

```
# 引入未来的注解特性，以便在类中使用注解
from __future__ import annotations

# 引入类型提示相关模块
from typing import Callable, Optional

# 引入 OrderedDict 类来创建有序字典
from collections import OrderedDict

# 引入操作系统相关功能
import os

# 引入正则表达式模块
import re

# 引入子进程管理模块
import subprocess

# 引入警告模块
import warnings

# 从自定义模块中导入特定功能
from .util import (
    find_binary_of_command,  # 导入寻找命令二进制文件的功能
    unique_list,  # 导入生成唯一列表的功能
    CompileError  # 导入编译错误异常类
)


class CompilerRunner:
    """ CompilerRunner base class.

    Parameters
    ==========

    sources : list of str
        Paths to sources.
    out : str
    flags : iterable of str
        Compiler flags.
    run_linker : bool
    compiler_name_exe : (str, str) tuple
        Tuple of compiler name &  command to call.
    cwd : str
        Path of root of relative paths.
    include_dirs : list of str
        Include directories.
    libraries : list of str
        Libraries to link against.
    library_dirs : list of str
        Paths to search for shared libraries.
    std : str
        Standard string, e.g. ``'c++11'``, ``'c99'``, ``'f2003'``.
    define: iterable of strings
        macros to define
    undef : iterable of strings
        macros to undefine
    preferred_vendor : string
        name of preferred vendor e.g. 'gnu' or 'intel'

    Methods
    =======

    run():
        Invoke compilation as a subprocess.

    """

    # 环境变量中指定编译器的键名，例如 'CC', 'CXX', ...
    environ_key_compiler: str

    # 环境变量中指定编译器选项的键名，例如 'CFLAGS', 'CXXFLAGS', ...
    environ_key_flags: str

    # 环境变量中指定链接器选项的键名，默认为 'LDFLAGS'
    environ_key_ldflags: str = "LDFLAGS"

    # 子类到供应商/二进制文件名字典的映射关系
    compiler_dict: dict[str, str]

    # 支持的标准应该是一个元组，包含支持的标准名称（第一个为默认值）
    standards: tuple[None | str, ...]

    # 子类到格式化回调函数的映射关系，例如 {'gcc': Callable[[Optional[str]], str], ...}
    std_formater: dict[str, Callable[[Optional[str]], str]]

    # 子类到供应商名称的映射关系，例如 {'gcc': 'gnu', ...}
    compiler_name_vendor_mapping: dict[str, str]

    @classmethod
    def find_compiler(cls, preferred_vendor=None):
        """ Identify a suitable C/fortran/other compiler. """
        # 获取可用的编译器候选列表
        candidates = list(cls.compiler_dict.keys())
        
        # 如果有指定首选供应商，将其置于候选列表的最前面
        if preferred_vendor:
            if preferred_vendor in candidates:
                candidates = [preferred_vendor] + candidates
            else:
                # 如果指定的首选供应商不在候选列表中，则引发值错误异常
                raise ValueError("Unknown vendor {}".format(preferred_vendor))
        
        # 查找候选列表中第一个可用的编译器名字和路径
        name, path = find_binary_of_command([cls.compiler_dict[x] for x in candidates])
        
        # 返回找到的编译器名字、路径及其对应的供应商名称
        return name, path, cls.compiler_name_vendor_mapping[name]
    def cmd(self):
        """ List of arguments (str) to be passed to e.g. ``subprocess.Popen``. """
        # 构建编译命令的列表，包括编译器、标志、未定义宏、已定义宏、包含目录和源文件
        cmd = (
            [self.compiler_binary] +  # 编译器可执行文件路径
            self.flags +               # 编译器标志列表
            ['-U'+x for x in self.undef] +  # 未定义宏列表
            ['-D'+x for x in self.define] +  # 已定义宏列表
            ['-I'+x for x in self.include_dirs] +  # 包含目录列表
            self.sources  # 源文件列表
        )
        if self.run_linker:
            # 如果需要运行链接器，则添加链接器相关的标志、库目录和库文件
            cmd += (['-L'+x for x in self.library_dirs] +
                    ['-l'+x for x in self.libraries] +
                    self.linkline)
        counted = []
        # 检查命令中的环境变量引用，确保所有引用的环境变量都已定义
        for envvar in re.findall(r'\$\{(\w+)\}', ' '.join(cmd)):
            if os.getenv(envvar) is None:
                if envvar not in counted:
                    counted.append(envvar)
                    msg = "Environment variable '{}' undefined.".format(envvar)
                    raise CompileError(msg)
        return cmd

    def run(self):
        self.flags = unique_list(self.flags)

        # 将输出标志和输出文件名添加到标志列表的末尾
        self.flags.extend(['-o', self.out])
        env = os.environ.copy()
        env['PWD'] = self.cwd

        # 创建子进程执行编译命令
        p = subprocess.Popen(' '.join(self.cmd()),  # 将命令列表转换为字符串形式
                             shell=True,            # 在 shell 中执行命令
                             cwd=self.cwd,          # 指定工作目录
                             stdin=subprocess.PIPE,  # 标准输入管道
                             stdout=subprocess.PIPE,  # 标准输出管道
                             stderr=subprocess.STDOUT,  # 标准错误输出到标准输出
                             env=env)                # 指定子进程的环境变量
        comm = p.communicate()  # 等待子进程执行完成并获取输出

        try:
            self.cmd_outerr = comm[0].decode('utf-8')  # 解码子进程输出（UTF-8 编码）
        except UnicodeDecodeError:
            self.cmd_outerr = comm[0].decode('iso-8859-1')  # 解码子进程输出（ISO-8859-1 编码，适用于 win32）
        self.cmd_returncode = p.returncode  # 获取子进程的返回码

        # 错误处理：如果编译命令返回非零状态码，则抛出编译错误异常
        if self.cmd_returncode != 0:
            msg = "Error executing '{}' in {} (exited status {}):\n {}\n".format(
                ' '.join(self.cmd()), self.cwd, str(self.cmd_returncode), self.cmd_outerr
            )
            raise CompileError(msg)

        return self.cmd_outerr, self.cmd_returncode
class CCompilerRunner(CompilerRunner):
    # CCompilerRunner 类继承自 CompilerRunner 类，用于处理 C 语言编译器的运行相关功能

    environ_key_compiler = 'CC'
    # 环境变量中编译器的键名，用于设置 C 编译器

    environ_key_flags = 'CFLAGS'
    # 环境变量中编译器标志的键名，用于设置 C 编译器的编译选项

    compiler_dict = OrderedDict([
        ('gnu', 'gcc'),
        ('intel', 'icc'),
        ('llvm', 'clang'),
    ])
    # 编译器名称到实际编译器命令的有序字典映射，包括 GNU gcc、Intel icc 和 LLVM clang

    standards = ('c89', 'c90', 'c99', 'c11')  # First is default
    # 支持的 C 语言标准列表，默认为 c89

    std_formater = {
        'gcc': '-std={}'.format,
        'icc': '-std={}'.format,
        'clang': '-std={}'.format,
    }
    # 编译器到标准格式化字符串的映射，用于设置编译器的 C 语言标准

    compiler_name_vendor_mapping = {
        'gcc': 'gnu',
        'icc': 'intel',
        'clang': 'llvm'
    }
    # 编译器名称到供应商（vendor）的映射，例如 gcc 对应 gnu、icc 对应 intel、clang 对应 llvm


def _mk_flag_filter(cmplr_name):  # helper for class initialization
    # 用于类初始化的辅助函数，生成编译器标志的过滤器

    not_welcome = {'g++': ("Wimplicit-interface",)}  # "Wstrict-prototypes",)}
    # 不欢迎的编译器警告或错误列表，例如 g++ 中的 "Wimplicit-interface"

    if cmplr_name in not_welcome:
        # 如果编译器名称存在于不欢迎的列表中
        def fltr(x):
            for nw in not_welcome[cmplr_name]:
                if nw in x:
                    return False
            return True
    else:
        # 否则
        def fltr(x):
            return True
    return fltr
    # 返回生成的标志过滤函数


class CppCompilerRunner(CompilerRunner):
    # CppCompilerRunner 类继承自 CompilerRunner 类，用于处理 C++ 语言编译器的运行相关功能

    environ_key_compiler = 'CXX'
    # 环境变量中编译器的键名，用于设置 C++ 编译器

    environ_key_flags = 'CXXFLAGS'
    # 环境变量中编译器标志的键名，用于设置 C++ 编译器的编译选项

    compiler_dict = OrderedDict([
        ('gnu', 'g++'),
        ('intel', 'icpc'),
        ('llvm', 'clang++'),
    ])
    # 编译器名称到实际编译器命令的有序字典映射，包括 GNU g++、Intel icpc 和 LLVM clang++

    standards = ('c++98', 'c++0x')
    # 支持的 C++ 语言标准列表，默认为 c++98

    std_formater = {
        'g++': '-std={}'.format,
        'icpc': '-std={}'.format,
        'clang++': '-std={}'.format,
    }
    # 编译器到标准格式化字符串的映射，用于设置编译器的 C++ 语言标准

    compiler_name_vendor_mapping = {
        'g++': 'gnu',
        'icpc': 'intel',
        'clang++': 'llvm'
    }
    # 编译器名称到供应商（vendor）的映射，例如 g++ 对应 gnu、icpc 对应 intel、clang++ 对应 llvm


class FortranCompilerRunner(CompilerRunner):
    # FortranCompilerRunner 类继承自 CompilerRunner 类，用于处理 Fortran 语言编译器的运行相关功能

    environ_key_compiler = 'FC'
    # 环境变量中编译器的键名，用于设置 Fortran 编译器

    environ_key_flags = 'FFLAGS'
    # 环境变量中编译器标志的键名，用于设置 Fortran 编译器的编译选项

    standards = (None, 'f77', 'f95', 'f2003', 'f2008')
    # 支持的 Fortran 语言标准列表，包括 None（无特定标准）、f77、f95、f2003 和 f2008

    std_formater = {
        'gfortran': lambda x: '-std=gnu' if x is None else '-std=legacy' if x == 'f77' else '-std={}'.format(x),
        'ifort': lambda x: '-stand f08' if x is None else '-stand f{}'.format(x[-2:]),  # f2008 => f08
    }
    # 编译器到标准格式化字符串的映射，用于设置编译器的 Fortran 语言标准

    compiler_dict = OrderedDict([
        ('gnu', 'gfortran'),
        ('intel', 'ifort'),
    ])
    # 编译器名称到实际编译器命令的有序字典映射，包括 GNU gfortran 和 Intel ifort

    compiler_name_vendor_mapping = {
        'gfortran': 'gnu',
        'ifort': 'intel',
    }
    # 编译器名称到供应商（vendor）的映射，例如 gfortran 对应 gnu、ifort 对应 intel
```