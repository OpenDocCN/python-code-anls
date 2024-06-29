# `.\numpy\numpy\f2py\tests\util.py`

```py
"""
Utility functions for

- building and importing modules on test time, using a temporary location
- detecting if compilers are present
- determining paths to tests

"""
# 导入必要的库和模块
import glob
import os
import sys
import subprocess
import tempfile
import shutil
import atexit
import textwrap
import re
import pytest
import contextlib
import numpy
import concurrent.futures

from pathlib import Path
from numpy._utils import asunicode
from numpy.testing import temppath, IS_WASM
from importlib import import_module
from numpy.f2py._backends._meson import MesonBackend

#
# Maintaining a temporary module directory
#

# 初始化临时模块目录和模块编号
_module_dir = None
_module_num = 5403

# 如果运行平台是cygwin，设置numpy的安装根目录
if sys.platform == "cygwin":
    NUMPY_INSTALL_ROOT = Path(__file__).parent.parent.parent
    _module_list = list(NUMPY_INSTALL_ROOT.glob("**/*.dll"))


def _cleanup():
    global _module_dir
    # 清理临时模块目录
    if _module_dir is not None:
        try:
            sys.path.remove(_module_dir)
        except ValueError:
            pass
        try:
            shutil.rmtree(_module_dir)
        except OSError:
            pass
        _module_dir = None


def get_module_dir():
    global _module_dir
    # 获取临时模块目录，如果不存在则创建
    if _module_dir is None:
        _module_dir = tempfile.mkdtemp()
        atexit.register(_cleanup)
        if _module_dir not in sys.path:
            sys.path.insert(0, _module_dir)
    return _module_dir


def get_temp_module_name():
    # 获取一个唯一的临时模块名
    global _module_num
    get_module_dir()
    name = "_test_ext_module_%d" % _module_num
    _module_num += 1
    if name in sys.modules:
        # 检查是否已存在同名模块，理论上不应该出现这种情况
        raise RuntimeError("Temporary module name already in use.")
    return name


def _memoize(func):
    memo = {}

    def wrapper(*a, **kw):
        key = repr((a, kw))
        if key not in memo:
            try:
                memo[key] = func(*a, **kw)
            except Exception as e:
                memo[key] = e
                raise
        ret = memo[key]
        if isinstance(ret, Exception):
            raise ret
        return ret

    wrapper.__name__ = func.__name__
    return wrapper

#
# Building modules
#


@_memoize
def build_module(source_files, options=[], skip=[], only=[], module_name=None):
    """
    Compile and import a f2py module, built from the given files.

    """

    # 准备执行编译操作的代码字符串
    code = f"import sys; sys.path = {sys.path!r}; import numpy.f2py; numpy.f2py.main()"

    # 获取临时模块目录
    d = get_module_dir()

    # 复制文件到临时模块目录并准备编译所需的源文件列表
    dst_sources = []
    f2py_sources = []
    for fn in source_files:
        if not os.path.isfile(fn):
            raise RuntimeError("%s is not a file" % fn)
        dst = os.path.join(d, os.path.basename(fn))
        shutil.copyfile(fn, dst)
        dst_sources.append(dst)

        base, ext = os.path.splitext(dst)
        if ext in (".f90", ".f95", ".f", ".c", ".pyf"):
            f2py_sources.append(dst)

    assert f2py_sources

    # 准备编译选项
    # 如果模块名未指定，则生成临时模块名
    if module_name is None:
        module_name = get_temp_module_name()

    # 构建 f2py 的选项列表，包括编译选项、模块名和源文件列表
    f2py_opts = ["-c", "-m", module_name] + options + f2py_sources
    f2py_opts += ["--backend", "meson"]

    # 如果有跳过的选项，则添加到 f2py 的选项列表中
    if skip:
        f2py_opts += ["skip:"] + skip

    # 如果有仅包含的选项，则添加到 f2py 的选项列表中
    if only:
        f2py_opts += ["only:"] + only

    # 构建
    # 保存当前工作目录
    cwd = os.getcwd()
    try:
        # 切换到指定目录 d
        os.chdir(d)
        # 构建执行的命令列表，包括 Python 解释器、执行代码、f2py 选项
        cmd = [sys.executable, "-c", code] + f2py_opts
        # 启动子进程执行命令，捕获标准输出和标准错误
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        out, err = p.communicate()
        # 如果返回码不为 0，则抛出运行时错误
        if p.returncode != 0:
            raise RuntimeError("Running f2py failed: %s\n%s" %
                               (cmd[4:], asunicode(out)))
    finally:
        # 恢复到之前保存的工作目录
        os.chdir(cwd)

        # 部分清理
        # 删除指定的源文件列表中的文件
        for fn in dst_sources:
            os.unlink(fn)

    # 重新基址（仅适用于 Cygwin）
    if sys.platform == "cygwin":
        # 如果在导入后有人开始删除模块，则需要更改以记录每个模块的大小，
        # 而不是依赖 rebase 能够从文件中找到这些信息。
        # 将模块名匹配的文件列表扩展到 _module_list 中
        _module_list.extend(
            glob.glob(os.path.join(d, "{:s}*".format(module_name)))
        )
        # 调用 rebase 进行重新基址，使用数据库和详细模式
        subprocess.check_call(
            ["/usr/bin/rebase", "--database", "--oblivious", "--verbose"]
            + _module_list
        )

    # 导入模块
    return import_module(module_name)
# 装饰器函数，用于对 build_code 函数进行记忆化（memoization）
@_memoize
# 编译给定的 Fortran 代码并导入为模块
def build_code(source_code,
               options=[],
               skip=[],
               only=[],
               suffix=None,
               module_name=None):
    """
    Compile and import Fortran code using f2py.
    编译并导入 Fortran 代码，使用 f2py 工具。
    """
    # 如果未指定后缀名，则默认为 .f
    if suffix is None:
        suffix = ".f"
    # 利用临时路径创建一个文件，写入源代码
    with temppath(suffix=suffix) as path:
        with open(path, "w") as f:
            f.write(source_code)
        # 调用 build_module 函数，编译指定路径的代码文件为模块并返回
        return build_module([path],
                            options=options,
                            skip=skip,
                            only=only,
                            module_name=module_name)


#
# 检查是否至少有一个编译器可用...
#

# 检查指定语言的编译器是否可用
def check_language(lang, code_snippet=None):
    # 创建一个临时目录
    tmpdir = tempfile.mkdtemp()
    try:
        # 在临时目录下创建一个 Meson 构建文件
        meson_file = os.path.join(tmpdir, "meson.build")
        with open(meson_file, "w") as f:
            f.write("project('check_compilers')\n")
            f.write(f"add_languages('{lang}')\n")
            if code_snippet:
                f.write(f"{lang}_compiler = meson.get_compiler('{lang}')\n")
                f.write(f"{lang}_code = '''{code_snippet}'''\n")
                f.write(
                    f"_have_{lang}_feature ="
                    f"{lang}_compiler.compiles({lang}_code,"
                    f" name: '{lang} feature check')\n"
                )
        # 在临时目录下运行 Meson 进行设置
        runmeson = subprocess.run(
            ["meson", "setup", "btmp"],
            check=False,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # 如果 Meson 设置成功返回 True，否则返回 False
        if runmeson.returncode == 0:
            return True
        else:
            return False
    finally:
        # 最后删除临时目录及其内容
        shutil.rmtree(tmpdir)
    return False

# Fortran 77 示例代码
fortran77_code = '''
C Example Fortran 77 code
      PROGRAM HELLO
      PRINT *, 'Hello, Fortran 77!'
      END
'''

# Fortran 90 示例代码
fortran90_code = '''
! Example Fortran 90 code
program hello90
  type :: greeting
    character(len=20) :: text
  end type greeting

  type(greeting) :: greet
  greet%text = 'hello, fortran 90!'
  print *, greet%text
end program hello90
'''

# 用于缓存相关检查的虚拟类
class CompilerChecker:
    def __init__(self):
        self.compilers_checked = False
        self.has_c = False
        self.has_f77 = False
        self.has_f90 = False

    # 检查各种编译器的可用性
    def check_compilers(self):
        if (not self.compilers_checked) and (not sys.platform == "cygwin"):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(check_language, "c"),
                    executor.submit(check_language, "fortran", fortran77_code),
                    executor.submit(check_language, "fortran", fortran90_code)
                ]
                # 获取并记录每种语言的编译器可用性
                self.has_c = futures[0].result()
                self.has_f77 = futures[1].result()
                self.has_f90 = futures[2].result()

            self.compilers_checked = True

# 如果不是 WebAssembly 环境，创建一个编译器检查实例并进行检查
if not IS_WASM:
    checker = CompilerChecker()
    checker.check_compilers()

# 检查是否有 C 编译器可用
def has_c_compiler():
    # 返回变量 checker 的属性 has_c 的值
    return checker.has_c
# 检查当前系统是否具有 Fortran 77 编译器
def has_f77_compiler():
    return checker.has_f77

# 检查当前系统是否具有 Fortran 90 编译器
def has_f90_compiler():
    return checker.has_f90

#
# 使用 Meson 构建
#


class SimplifiedMesonBackend(MesonBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 编译方法，生成 Meson 构建文件并执行构建
    def compile(self):
        self.write_meson_build(self.build_dir)  # 写入 Meson 构建文件到指定目录
        self.run_meson(self.build_dir)  # 在指定目录运行 Meson 构建


def build_meson(source_files, module_name=None, **kwargs):
    """
    通过 Meson 构建并导入一个模块。
    """
    build_dir = get_module_dir()  # 获取模块的构建目录
    if module_name is None:
        module_name = get_temp_module_name()  # 获取临时模块名称

    # 初始化 SimplifiedMesonBackend 实例
    backend = SimplifiedMesonBackend(
        modulename=module_name,
        sources=source_files,
        extra_objects=kwargs.get("extra_objects", []),
        build_dir=build_dir,
        include_dirs=kwargs.get("include_dirs", []),
        library_dirs=kwargs.get("library_dirs", []),
        libraries=kwargs.get("libraries", []),
        define_macros=kwargs.get("define_macros", []),
        undef_macros=kwargs.get("undef_macros", []),
        f2py_flags=kwargs.get("f2py_flags", []),
        sysinfo_flags=kwargs.get("sysinfo_flags", []),
        fc_flags=kwargs.get("fc_flags", []),
        flib_flags=kwargs.get("flib_flags", []),
        setup_flags=kwargs.get("setup_flags", []),
        remove_build_dir=kwargs.get("remove_build_dir", False),
        extra_dat=kwargs.get("extra_dat", {}),
    )

    # 编译模块
    # 注意：由于没有 distutils，很难确定 CI 上使用的编译器栈，因此使用了一个捕获所有异常的通用处理
    try:
        backend.compile()
    except:
        pytest.skip("Failed to compile module")  # 若编译失败则跳过测试

    # 导入编译后的模块
    sys.path.insert(0, f"{build_dir}/{backend.meson_build_dir}")
    return import_module(module_name)


#
# 单元测试便利性
#


class F2PyTest:
    code = None
    sources = None
    options = []
    skip = []
    only = []
    suffix = ".f"
    module = None
    _has_c_compiler = None
    _has_f77_compiler = None
    _has_f90_compiler = None

    @property
    def module_name(self):
        cls = type(self)
        return f'_{cls.__module__.rsplit(".",1)[-1]}_{cls.__name__}_ext_module'

    @classmethod
    def setup_class(cls):
        if sys.platform == "win32":
            pytest.skip("Fails with MinGW64 Gfortran (Issue #9673)")
        F2PyTest._has_c_compiler = has_c_compiler()  # 检查当前系统是否具有 C 编译器
        F2PyTest._has_f77_compiler = has_f77_compiler()  # 检查当前系统是否具有 Fortran 77 编译器
        F2PyTest._has_f90_compiler = has_f90_compiler()  # 检查当前系统是否具有 Fortran 90 编译器
    # 设置测试方法的准备工作
    def setup_method(self):
        # 如果模块已经存在，则直接返回，避免重复设置
        if self.module is not None:
            return

        # 初始化代码列表，默认为空列表
        codes = self.sources if self.sources else []

        # 如果有单独的代码文件（self.code），将其添加到代码列表中
        if self.code:
            codes.append(self.suffix)

        # 检查代码列表中是否有需要使用 Fortran 77、Fortran 90 或 Python-Fortran 源文件的需求
        needs_f77 = any(str(fn).endswith(".f") for fn in codes)
        needs_f90 = any(str(fn).endswith(".f90") for fn in codes)
        needs_pyf = any(str(fn).endswith(".pyf") for fn in codes)

        # 如果需要 Fortran 77 编译器但系统没有，则跳过测试
        if needs_f77 and not self._has_f77_compiler:
            pytest.skip("No Fortran 77 compiler available")

        # 如果需要 Fortran 90 编译器但系统没有，则跳过测试
        if needs_f90 and not self._has_f90_compiler:
            pytest.skip("No Fortran 90 compiler available")

        # 如果需要 Python-Fortran 但系统没有支持的编译器，则跳过测试
        if needs_pyf and not (self._has_f90_compiler or self._has_f77_compiler):
            pytest.skip("No Fortran compiler available")

        # 如果指定了代码（self.code），则使用 build_code 函数构建模块
        if self.code is not None:
            self.module = build_code(
                self.code,
                options=self.options,
                skip=self.skip,
                only=self.only,
                suffix=self.suffix,
                module_name=self.module_name,
            )

        # 如果指定了源文件列表（self.sources），则使用 build_module 函数构建模块
        if self.sources is not None:
            self.module = build_module(
                self.sources,
                options=self.options,
                skip=self.skip,
                only=self.only,
                module_name=self.module_name,
            )
#
# Helper functions
#


# 根据给定的路径参数构造并返回路径对象
def getpath(*a):
    # 获取 numpy.f2py 模块的文件路径，并获取其父目录的绝对路径作为根目录
    d = Path(numpy.f2py.__file__).parent.resolve()
    return d.joinpath(*a)


# 上下文管理器，用于临时切换工作目录至指定路径，并在结束时恢复原工作目录
@contextlib.contextmanager
def switchdir(path):
    # 获取当前工作目录
    curpath = Path.cwd()
    # 切换工作目录至指定路径
    os.chdir(path)
    try:
        yield  # 执行被装饰函数体
    finally:
        # 最终恢复原工作目录
        os.chdir(curpath)
```