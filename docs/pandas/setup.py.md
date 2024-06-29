# `D:\src\scipysrc\pandas\setup.py`

```
# 指定 Python 解释器的位置
#!/usr/bin/env python3

"""
从 pyzmq 项目（https://github.com/zeromq/pyzmq）和 lxml 项目（https://github.com/lxml/lxml）中获取了部分代码，
根据 BSD 许可证允许使用这些部分。
"""

# 导入必要的库
import argparse  # 导入解析命令行参数的模块
import multiprocessing  # 多进程处理模块
import os  # 操作系统相关功能模块
from os.path import join as pjoin  # 导入路径拼接函数 join
import platform  # 获取平台信息的模块
import shutil  # 文件操作相关模块
import sys  # 系统相关的参数和功能
from sysconfig import get_config_vars  # 获取 Python 配置的变量

import numpy  # 数值计算库
from pkg_resources import parse_version  # 解析版本字符串的函数
from setuptools import (  # 使用 setuptools 中的相关模块
    Command,  # 扩展 setuptools 的命令基类
    Extension,  # 扩展模块的定义
    setup,  # 打包安装用的 setup 函数
)
from setuptools.command.build_ext import build_ext as _build_ext  # 构建扩展的命令
import versioneer  # 自动生成版本信息的工具

# 从 versioneer 中获取命令类
cmdclass = versioneer.get_cmdclass()

# 判断当前操作系统是否为 Windows
def is_platform_windows():
    return sys.platform in ("win32", "cygwin")

# 判断当前操作系统是否为 macOS
def is_platform_mac():
    return sys.platform == "darwin"

# 与 pyproject.toml、environment.yml 和 asv.conf.json 中的信息同步
min_cython_ver = "3.0"

try:
    from Cython import (  # 导入 Cython 相关模块
        Tempita,  # 模板处理工具
        __version__ as _CYTHON_VERSION,  # Cython 的版本信息
    )
    from Cython.Build import cythonize  # 使用 Cython 构建扩展模块

    # 判断当前安装的 Cython 版本是否符合最低要求
    _CYTHON_INSTALLED = parse_version(_CYTHON_VERSION) >= parse_version(min_cython_ver)
except ImportError:
    _CYTHON_VERSION = None
    _CYTHON_INSTALLED = False
    # 如果未安装 Cython，则将 cythonize 函数设置为一个占位符函数
    cythonize = lambda x, *args, **kwargs: x  # dummy func

# 定义用于存储模板依赖的字典
_pxi_dep_template = {
    "algos": ["_libs/algos_common_helper.pxi.in", "_libs/algos_take_helper.pxi.in"],
    "hashtable": [
        "_libs/hashtable_class_helper.pxi.in",
        "_libs/hashtable_func_helper.pxi.in",
        "_libs/khash_for_primitive_helper.pxi.in",
    ],
    "index": ["_libs/index_class_helper.pxi.in"],
    "sparse": ["_libs/sparse_op_helper.pxi.in"],
    "interval": ["_libs/intervaltree.pxi.in"],
}

# 初始化存储模板文件路径的列表和依赖字典
_pxifiles = []
_pxi_dep = {}
for module, files in _pxi_dep_template.items():
    pxi_files = [pjoin("pandas", x) for x in files]
    _pxifiles.extend(pxi_files)
    _pxi_dep[module] = pxi_files

# 自定义的构建扩展命令类，继承自 _build_ext
class build_ext(_build_ext):
    @classmethod
    def render_templates(cls, pxifiles) -> None:
        # 遍历所有的模板文件，生成相应的 .pxi 文件
        for pxifile in pxifiles:
            # 先构建 .pxi 文件，模板文件的扩展名必须是 .pxi.in
            assert pxifile.endswith(".pxi.in")
            outfile = pxifile[:-3]

            # 如果已经存在 .pxi 文件且模板文件没有更新，则不需要重新生成
            if (
                os.path.exists(outfile)
                and os.stat(pxifile).st_mtime < os.stat(outfile).st_mtime
            ):
                continue

            # 使用 Tempita 模块渲染模板文件
            with open(pxifile, encoding="utf-8") as f:
                tmpl = f.read()
            pyxcontent = Tempita.sub(tmpl)

            # 将渲染后的内容写入到 .pxi 文件中
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(pyxcontent)

    def build_extensions(self) -> None:
        # 如果使用的是从 c 文件构建的版本，则不需要生成模板文件的输出
        if _CYTHON_INSTALLED:
            self.render_templates(_pxifiles)

        # 调用父类的方法构建扩展模块
        super().build_extensions()

# 自定义的清理命令类，继承自 Command
class CleanCommand(Command):
    """Custom command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]
    # 初始化选项，设置默认值并准备待清理的文件和目录列表
    def initialize_options(self) -> None:
        self.all = True  # 设定一个布尔值选项，表示清理所有内容
        self._clean_me = []  # 待删除的文件列表
        self._clean_trees = []  # 待删除的目录列表

        # 设置基础路径和相关路径
        base = pjoin("pandas", "_libs", "src")
        parser = pjoin(base, "parser")
        vendored = pjoin(base, "vendored")
        dt = pjoin(base, "datetime")
        ujson_python = pjoin(vendored, "ujson", "python")
        ujson_lib = pjoin(vendored, "ujson", "lib")

        # 设置需要排除清理的文件列表
        self._clean_exclude = [
            pjoin(vendored, "numpy", "datetime", "np_datetime.c"),
            pjoin(vendored, "numpy", "datetime", "np_datetime_strings.c"),
            pjoin(dt, "date_conversions.c"),
            pjoin(parser, "tokenizer.c"),
            pjoin(parser, "io.c"),
            pjoin(ujson_python, "ujson.c"),
            pjoin(ujson_python, "objToJSON.c"),
            pjoin(ujson_python, "JSONtoObj.c"),
            pjoin(ujson_lib, "ultrajsonenc.c"),
            pjoin(ujson_lib, "ultrajsondec.c"),
            pjoin(dt, "pd_datetime.c"),
            pjoin(parser, "pd_parser.c"),
        ]

        # 遍历 "pandas" 目录及其子目录中的文件
        for root, dirs, files in os.walk("pandas"):
            for f in files:
                filepath = pjoin(root, f)
                # 如果文件在排除列表中，则跳过不处理
                if filepath in self._clean_exclude:
                    continue

                # 如果文件的扩展名指示它是需要清理的文件类型，则将其加入待删除列表
                if os.path.splitext(f)[-1] in (
                    ".pyc",
                    ".so",
                    ".o",
                    ".pyo",
                    ".pyd",
                    ".c",
                    ".cpp",
                    ".orig",
                ):
                    self._clean_me.append(filepath)

            # 将子目录名 "__pycache__" 加入待删除目录列表
            self._clean_trees.append(pjoin(root, d) for d in dirs if d == "__pycache__")

        # 清理生成的 pxi 文件
        for pxifile in _pxifiles:
            pxifile_replaced = pxifile.replace(".pxi.in", ".pxi")
            self._clean_me.append(pxifile_replaced)

        # 将 "build" 和 "dist" 目录加入待删除目录列表，如果存在的话
        self._clean_trees.append(d for d in ("build", "dist") if os.path.exists(d))

    # 完成选项的设置，无需特别操作
    def finalize_options(self) -> None:
        pass

    # 执行清理操作，删除待删除的文件和目录
    def run(self) -> None:
        # 删除待删除文件列表中的每个文件
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except OSError:
                pass
        
        # 删除待删除目录列表中的每个目录及其内容
        for clean_tree in self._clean_trees:
            try:
                shutil.rmtree(clean_tree)
            except OSError:
                pass
# 从 cmdclass 字典中获取 'sdist' 对应的类作为基类
sdist_class = cmdclass["sdist"]

# 定义一个名为 CheckSDist 的类，继承自 sdist_class
class CheckSDist(sdist_class):
    """Custom sdist that ensures Cython has compiled all pyx files to c."""

    # 包含需要编译的 .pyx 文件列表
    _pyxfiles = [
        "pandas/_libs/arrays.pyx",
        "pandas/_libs/lib.pyx",
        "pandas/_libs/hashtable.pyx",
        "pandas/_libs/tslib.pyx",
        "pandas/_libs/index.pyx",
        "pandas/_libs/internals.pyx",
        "pandas/_libs/algos.pyx",
        "pandas/_libs/join.pyx",
        "pandas/_libs/indexing.pyx",
        "pandas/_libs/interval.pyx",
        "pandas/_libs/hashing.pyx",
        "pandas/_libs/missing.pyx",
        "pandas/_libs/testing.pyx",
        "pandas/_libs/sparse.pyx",
        "pandas/_libs/ops.pyx",
        "pandas/_libs/parsers.pyx",
        "pandas/_libs/tslibs/base.pyx",
        "pandas/_libs/tslibs/ccalendar.pyx",
        "pandas/_libs/tslibs/dtypes.pyx",
        "pandas/_libs/tslibs/period.pyx",
        "pandas/_libs/tslibs/strptime.pyx",
        "pandas/_libs/tslibs/np_datetime.pyx",
        "pandas/_libs/tslibs/timedeltas.pyx",
        "pandas/_libs/tslibs/timestamps.pyx",
        "pandas/_libs/tslibs/timezones.pyx",
        "pandas/_libs/tslibs/conversion.pyx",
        "pandas/_libs/tslibs/fields.pyx",
        "pandas/_libs/tslibs/offsets.pyx",
        "pandas/_libs/tslibs/parsing.pyx",
        "pandas/_libs/tslibs/tzconversion.pyx",
        "pandas/_libs/tslibs/vectorized.pyx",
        "pandas/_libs/window/indexers.pyx",
        "pandas/_libs/writers.pyx",
        "pandas/_libs/sas.pyx",
        "pandas/_libs/byteswap.pyx",
    ]

    # 包含需要编译的 .pyx 文件列表，为 C++ 扩展
    _cpp_pyxfiles = [
        "pandas/_libs/window/aggregations.pyx",
    ]

    # 初始化选项方法，继承自父类的初始化选项
    def initialize_options(self) -> None:
        sdist_class.initialize_options(self)

    # 运行方法，继承自父类的运行方法
    def run(self) -> None:
        # 如果 'cython' 存在于 cmdclass 中
        if "cython" in cmdclass:
            # 运行 'cython' 命令
            self.run_command("cython")
        else:
            # 如果没有运行 'cython'
            # 则正确编译所有扩展

            # 定义需要处理的 .pyx 文件和对应的编译后文件类型
            pyx_files = [(self._pyxfiles, "c"), (self._cpp_pyxfiles, "cpp")]

            # 遍历每对 .pyx 文件和文件类型
            for pyxfiles, extension in pyx_files:
                for pyxfile in pyxfiles:
                    # 构建编译后的源文件路径
                    sourcefile = pyxfile[:-3] + extension
                    # 错误消息，显示源文件不存在
                    msg = (
                        f"{extension}-source file '{sourcefile}' not found.\n"
                        "Run 'setup.py cython' before sdist."
                    )
                    # 断言源文件存在，否则抛出错误消息
                    assert os.path.isfile(sourcefile), msg
        
        # 调用父类的运行方法
        sdist_class.run(self)


# 定义一个 CheckingBuildExt 类，用于构建扩展
class CheckingBuildExt(build_ext):
    """
    Subclass build_ext to get clearer report if Cython is necessary.
    """
    # 检查 Cython 扩展模块是否存在生成的源文件
    def check_cython_extensions(self, extensions) -> None:
        # 遍历传入的扩展模块列表
        for ext in extensions:
            # 遍历当前扩展模块的源文件列表
            for src in ext.sources:
                # 如果源文件不存在
                if not os.path.exists(src):
                    # 输出未找到源文件的错误信息，包含扩展模块名称和源文件路径列表
                    print(f"{ext.name}: -> [{ext.sources}]")
                    # 抛出异常，指示缺少生成的 Cython 源文件
                    raise Exception(
                        f"""Cython-generated file '{src}' not found.
                Cython is required to compile pandas from a development branch.
                Please install Cython or download a release package of pandas.
                """
                    )

    # 构建 Cython 扩展模块
    def build_extensions(self) -> None:
        # 检查当前实例的扩展模块是否有缺失的 Cython 源文件
        self.check_cython_extensions(self.extensions)
        # 调用父类的方法来构建扩展模块
        build_ext.build_extensions(self)
class CythonCommand(build_ext):
    """
    Custom command subclassed from Cython.Distutils.build_ext
    to compile pyx->c, and stop there. All this does is override the
    C-compile method build_extension() with a no-op.
    """

    def build_extension(self, ext) -> None:
        # 重写 build_extension 方法为空操作，用于停止编译过程
        pass


class DummyBuildSrc(Command):
    """numpy's build_src command interferes with Cython's build_ext."""

    user_options = []

    def initialize_options(self) -> None:
        # 初始化选项，设置一个空的 Python 模块字典
        self.py_modules_dict = {}

    def finalize_options(self) -> None:
        # 完成选项的设置，这里不执行任何操作
        pass

    def run(self) -> None:
        # 执行命令的核心逻辑，这里没有具体的实现
        pass


cmdclass["clean"] = CleanCommand
cmdclass["build_ext"] = CheckingBuildExt

if _CYTHON_INSTALLED:
    suffix = ".pyx"
    # 如果 Cython 已安装，则将 CythonCommand 加入 cmdclass 中
    cmdclass["cython"] = CythonCommand
else:
    suffix = ".c"
    # 如果 Cython 未安装，则将 DummyBuildSrc 加入 cmdclass 中
    cmdclass["build_src"] = DummyBuildSrc

# ----------------------------------------------------------------------
# 编译参数的准备工作

debugging_symbols_requested = "--with-debugging-symbols" in sys.argv
if debugging_symbols_requested:
    # 如果命令行参数中包含 '--with-debugging-symbols'，则移除该参数
    sys.argv.remove("--with-debugging-symbols")


if sys.byteorder == "big":
    # 如果系统字节顺序为大端序，定义一个大端序的宏
    endian_macro = [("__BIG_ENDIAN__", "1")]
else:
    # 如果系统字节顺序为小端序，定义一个小端序的宏
    endian_macro = [("__LITTLE_ENDIAN__", "1")]


extra_compile_args = []
extra_link_args = []
if is_platform_windows():
    if debugging_symbols_requested:
        # 如果在 Windows 平台且请求了调试符号，则添加编译和链接的调试选项
        extra_compile_args.append("/Z7")
        extra_link_args.append("/DEBUG")
else:
    # 在非 Windows 平台
    # 在 CI 环境中设置了环境变量 PANDAS_CI=1
    if os.environ.get("PANDAS_CI", "0") == "1":
        extra_compile_args.append("-Werror")
    if debugging_symbols_requested:
        # 如果请求了调试符号，则添加调试相关的编译选项
        extra_compile_args.append("-g3")
        extra_compile_args.append("-UNDEBUG")
        extra_compile_args.append("-O0")

# 当在 macOS 上编译时，至少构建支持 macOS 10.9 的二进制
if is_platform_mac():
    if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
        # 如果未设置 MACOSX_DEPLOYMENT_TARGET 环境变量，则根据当前系统版本设置目标版本
        current_system = platform.mac_ver()[0]
        python_target = get_config_vars().get(
            "MACOSX_DEPLOYMENT_TARGET", current_system
        )
        target_macos_version = "10.9"
        parsed_macos_version = parse_version(target_macos_version)
        if (
            parse_version(str(python_target))
            < parsed_macos_version
            <= parse_version(current_system)
        ):
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = target_macos_version

    if sys.version_info[:2] == (3, 8):  # GH 33239
        # 对于 Python 3.8 版本，忽略一些过时声明的警告
        extra_compile_args.append("-Wno-error=deprecated-declarations")

    # 解决 https://github.com/pandas-dev/pandas/issues/35559 的问题
    # 忽略一些不可达代码的警告
    extra_compile_args.append("-Wno-error=unreachable-code")

# 通过设置环境变量 "PANDAS_CYTHON_COVERAGE"（值为真）或使用 `--with-cython-coverage` 参数
# 开启 Cython 文件的覆盖率检查
# 从环境变量中获取 PANDAS_CYTHON_COVERAGE 的值，若不存在则使用默认值 False
linetrace = os.environ.get("PANDAS_CYTHON_COVERAGE", False)

# 如果命令行参数中包含 "--with-cython-coverage"，则将 linetrace 设置为 True，并移除该参数
if "--with-cython-coverage" in sys.argv:
    linetrace = True
    sys.argv.remove("--with-cython-coverage")

# 注意：如果不使用 `cythonize`，可以通过将 `ext.cython_directives = directives` 固定到每个扩展中来启用覆盖率。
# 参考：github.com/cython/cython/wiki/enhancements-compilerdirectives#in-setuppy
# 设置 Cython 编译指令的初始值
directives = {"linetrace": False, "language_level": 3, "always_allow_keywords": True}
macros = []

# 如果 linetrace 为 True，则启用 Cython 的代码覆盖率
if linetrace:
    # 参考：https://pypkg.com/pypi/pytest-cython/f/tests/example-project/setup.py
    directives["linetrace"] = True
    macros = [("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")]

# 抑制关于废弃 API 使用的构建警告
# 由于 Cython 和 numpy 版本不匹配，这些警告无法避免。
macros.append(("NPY_NO_DEPRECATED_API", "0"))


# ----------------------------------------------------------------------
# 依赖规范


# TODO(cython#4518): 需要检查例如 `linetrace` 是否有变化并可能重新编译。
# 定义函数 maybe_cythonize，用于在调用 cythonize 前渲染 tempita 模板。
def maybe_cythonize(extensions, *args, **kwargs):
    """
    在调用 cythonize 之前渲染 tempita 模板。在以下情况下跳过：

    * clean
    * sdist
    """
    if "clean" in sys.argv or "sdist" in sys.argv:
        # 参考：https://github.com/cython/cython/issues/1495
        return extensions

    elif not _CYTHON_INSTALLED:
        # GH#28836 抛出一个有用的错误信息
        if _CYTHON_VERSION:
            raise RuntimeError(
                f"Cannot cythonize with old Cython version ({_CYTHON_VERSION} "
                f"installed, needs {min_cython_ver})"
            )
        raise RuntimeError("Cannot cythonize without Cython installed.")

    # 如果需要调试符号，则设置 kwargs 中的 gdb_debug 为 True
    if debugging_symbols_requested:
        kwargs["gdb_debug"] = True

    # 解析并设置并行编译线程数
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", "-j", type=int, default=1)
    parsed, _ = parser.parse_known_args()
    kwargs["nthreads"] = parsed.parallel

    # 渲染 _pxifiles 中的模板文件
    build_ext.render_templates(_pxifiles)

    return cythonize(extensions, *args, **kwargs)


# 定义函数 srcpath，用于生成源文件路径，默认位于 "pandas/src/" 目录下
def srcpath(name=None, suffix=".pyx", subdir="src"):
    return pjoin("pandas", subdir, name + suffix)


# 定义 lib_depends 列表，包含 pandas 解析辅助相关的头文件依赖
lib_depends = ["pandas/_libs/include/pandas/parse_helper.h"]

# 定义 tseries_depends 列表，包含 pandas 时间序列相关的头文件依赖
tseries_depends = [
    "pandas/_libs/include/pandas/datetime/pd_datetime.h",
]

# 定义 ext_data 字典，包含各个 pandas 扩展模块的信息
ext_data = {
    "_libs.algos": {
        "pyxfile": "_libs/algos",
        "depends": _pxi_dep["algos"],
    },
    "_libs.arrays": {"pyxfile": "_libs/arrays"},
    "_libs.groupby": {"pyxfile": "_libs/groupby"},
    "_libs.hashing": {"pyxfile": "_libs/hashing", "depends": []},
}
    # 定义 "_libs.hashtable" 字典，包含与哈希表相关的信息
    "_libs.hashtable": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/hashtable",
        # 指定依赖项列表，包括特定头文件和其他依赖
        "depends": (
            [
                "pandas/_libs/include/pandas/vendored/klib/khash_python.h",
                "pandas/_libs/include/pandas/vendored/klib/khash.h",
            ]
            + _pxi_dep["hashtable"]  # 根据 _pxi_dep 中 "hashtable" 键的值来扩展依赖列表
        ),
    },
    
    # 定义 "_libs.index" 字典，包含与索引相关的信息
    "_libs.index": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/index",
        # 指定依赖项列表，根据 _pxi_dep 中 "index" 键的值来设置
        "depends": _pxi_dep["index"],
    },
    
    # 定义 "_libs.indexing" 字典，包含与索引操作相关的信息
    "_libs.indexing": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/indexing",
    },
    
    # 定义 "_libs.internals" 字典，包含与内部操作相关的信息
    "_libs.internals": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/internals",
    },
    
    # 定义 "_libs.interval" 字典，包含与区间相关的信息
    "_libs.interval": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/interval",
        # 指定依赖项列表，根据 _pxi_dep 中 "interval" 键的值来设置
        "depends": _pxi_dep["interval"],
    },
    
    # 定义 "_libs.join" 字典，包含与连接操作相关的信息
    "_libs.join": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/join",
    },
    
    # 定义 "_libs.lib" 字典，包含与库相关的信息
    "_libs.lib": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/lib",
        # 指定依赖项列表，包括 lib_depends 和 tseries_depends
        "depends": lib_depends + tseries_depends,
    },
    
    # 定义 "_libs.missing" 字典，包含与缺失值处理相关的信息
    "_libs.missing": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/missing",
        # 指定依赖项列表，仅包括 tseries_depends
        "depends": tseries_depends,
    },
    
    # 定义 "_libs.parsers" 字典，包含与解析器相关的信息
    "_libs.parsers": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/parsers",
        # 指定依赖项列表，包括解析器所需的特定头文件
        "depends": [
            "pandas/_libs/src/parser/tokenizer.h",
            "pandas/_libs/src/parser/io.h",
            "pandas/_libs/src/pd_parser.h",
        ],
    },
    
    # 定义 "_libs.ops" 字典，包含与操作相关的信息
    "_libs.ops": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/ops",
    },
    
    # 定义 "_libs.ops_dispatch" 字典，包含与操作分发相关的信息
    "_libs.ops_dispatch": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/ops_dispatch",
    },
    
    # 定义 "_libs.properties" 字典，包含与属性相关的信息
    "_libs.properties": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/properties",
    },
    
    # 定义 "_libs.reshape" 字典，包含与重塑相关的信息
    "_libs.reshape": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/reshape",
        # 指定空的依赖项列表
        "depends": [],
    },
    
    # 定义 "_libs.sparse" 字典，包含与稀疏数据相关的信息
    "_libs.sparse": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/sparse",
        # 指定依赖项列表，根据 _pxi_dep 中 "sparse" 键的值来设置
        "depends": _pxi_dep["sparse"],
    },
    
    # 定义 "_libs.tslib" 字典，包含与时间序列操作相关的信息
    "_libs.tslib": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/tslib",
        # 指定依赖项列表，仅包括 tseries_depends
        "depends": tseries_depends,
    },
    
    # 定义 "_libs.tslibs.base" 字典，包含与时间序列基础操作相关的信息
    "_libs.tslibs.base": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/tslibs/base",
    },
    
    # 定义 "_libs.tslibs.ccalendar" 字典，包含与日期日历相关的信息
    "_libs.tslibs.ccalendar": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/tslibs/ccalendar",
    },
    
    # 定义 "_libs.tslibs.dtypes" 字典，包含与数据类型相关的信息
    "_libs.tslibs.dtypes": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/tslibs/dtypes",
    },
    
    # 定义 "_libs.tslibs.conversion" 字典，包含与数据转换相关的信息
    "_libs.tslibs.conversion": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/tslibs/conversion",
        # 指定依赖项列表，仅包括 tseries_depends
        "depends": tseries_depends,
    },
    
    # 定义 "_libs.tslibs.fields" 字典，包含与时间序列字段相关的信息
    "_libs.tslibs.fields": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/tslibs/fields",
        # 指定依赖项列表，仅包括 tseries_depends
        "depends": tseries_depends,
    },
    
    # 定义 "_libs.tslibs.nattype" 字典，包含与 NAT 类型相关的信息
    "_libs.tslibs.nattype": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/tslibs/nattype",
    },
    
    # 定义 "_libs.tslibs.np_datetime" 字典，包含与 NumPy 日期时间相关的信息
    "_libs.tslibs.np_datetime": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/tslibs/np_datetime",
        # 指定依赖项列表，仅包括 tseries_depends
        "depends": tseries_depends,
    },
    
    # 定义 "_libs.tslibs.offsets" 字典，包含与时间序列偏移量相关的信息
    "_libs.tslibs.offsets": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/tslibs/offsets",
        # 指定依赖项列表，仅包括 tseries_depends
        "depends": tseries_depends,
    },
    
    # 定义 "_libs.tslibs.parsing" 字典，包含与时间序列解析相关的信息
    "_libs.tslibs.parsing": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/tslibs/parsing",
        # 指定源文件列表，包括特定的 C 文件
        "sources": ["pandas/_libs/src/parser/tokenizer.c"],
    },
    
    # 定义 "_libs.tslibs.period" 字典，包含与时间段相关的信息
    "_libs.tslibs.period": {
        # 指定对应的 Cython 文件路径
        "pyxfile": "_libs/tslibs/period",
        # 指定依赖项列表，仅包括 tseries_depends
        "depends": tseries_depends,
    },
    
    # 定义 "_libs.tslibs.strptime" 字典，包含与日期时间
    "_libs.tslibs.tzconversion": {
        "pyxfile": "_libs/tslibs/tzconversion",  # tzconversion 模块相关信息
        "depends": tseries_depends,  # 依赖于 tseries_depends 的模块信息
    },
    "_libs.tslibs.vectorized": {
        "pyxfile": "_libs/tslibs/vectorized",  # vectorized 模块相关信息
        "depends": tseries_depends,  # 依赖于 tseries_depends 的模块信息
    },
    "_libs.testing": {"pyxfile": "_libs/testing"},  # testing 模块相关信息
    "_libs.window.aggregations": {
        "pyxfile": "_libs/window/aggregations",  # aggregations 模块相关信息
        "language": "c++",  # 使用 C++ 语言
        "suffix": ".cpp",  # 文件后缀为 .cpp
        "depends": ["pandas/_libs/include/pandas/skiplist.h"],  # 依赖于 skiplist.h 头文件
    },
    "_libs.window.indexers": {"pyxfile": "_libs/window/indexers"},  # indexers 模块相关信息
    "_libs.writers": {"pyxfile": "_libs/writers"},  # writers 模块相关信息
    "_libs.sas": {"pyxfile": "_libs/sas"},  # sas 模块相关信息
    "_libs.byteswap": {"pyxfile": "_libs/byteswap"},  # byteswap 模块相关信息
# 扩展列表，用于存储所有的扩展模块对象
extensions = []

# 遍历 ext_data 字典，获取每个扩展模块的名称和数据
for name, data in ext_data.items():
    # 根据后缀选择源文件的后缀，如果 suffix 是 ".pyx" 则使用 ".pyx"，否则使用数据中的后缀或默认 ".c"
    source_suffix = suffix if suffix == ".pyx" else data.get("suffix", ".c")

    # 创建源文件列表，包含主要源文件和可能的其他源文件
    sources = [srcpath(data["pyxfile"], suffix=source_suffix, subdir="")]
    sources.extend(data.get("sources", []))

    # 包含的目录列表，添加 pandas 和 numpy 的头文件目录
    include = ["pandas/_libs/include", numpy.get_include()]

    # 未定义的宏列表
    undef_macros = []

    # 如果运行在 z/OS 平台，并且数据指定使用 C++ 语言，并且 C++ 编译器是 xlc 或 xlc++
    if (
        sys.platform == "zos"
        and data.get("language") == "c++"
        and os.path.basename(os.environ.get("CXX", "/bin/xlc++")) in ("xlc", "xlc++")
    ):
        # 向宏列表添加 __s390__ = 1
        data.get("macros", macros).append(("__s390__", "1"))
        # 添加额外的编译参数
        extra_compile_args.append("-qlanglvl=extended0x:nolibext")
        # 添加未定义的宏 _POSIX_THREADS
        undef_macros.append("_POSIX_THREADS")

    # 创建 Extension 对象，并将其添加到 extensions 列表中
    obj = Extension(
        f"pandas.{name}",
        sources=sources,
        depends=data.get("depends", []),
        include_dirs=include,
        language=data.get("language", "c"),
        define_macros=data.get("macros", macros),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        undef_macros=undef_macros,
    )
    extensions.append(obj)

# ----------------------------------------------------------------------
# ujson

# 如果 suffix 是 ".pyx"，修正 setuptools 的 bug，将 .pyx 源文件重置为 .c
if suffix == ".pyx":
    for ext in extensions:
        if ext.sources[0].endswith((".c", ".cpp")):
            root, _ = os.path.splitext(ext.sources[0])
            ext.sources[0] = root + suffix

# 创建 ujson_ext Extension 对象
ujson_ext = Extension(
    "pandas._libs.json",
    depends=[
        "pandas/_libs/include/pandas/vendored/ujson/lib/ultrajson.h",
        "pandas/_libs/include/pandas/datetime/pd_datetime.h",
    ],
    sources=(
        [
            "pandas/_libs/src/vendored/ujson/python/ujson.c",
            "pandas/_libs/src/vendored/ujson/python/objToJSON.c",
            "pandas/_libs/src/vendored/ujson/python/JSONtoObj.c",
            "pandas/_libs/src/vendored/ujson/lib/ultrajsonenc.c",
            "pandas/_libs/src/vendored/ujson/lib/ultrajsondec.c",
        ]
    ),
    include_dirs=[
        "pandas/_libs/include",
        numpy.get_include(),
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=macros,
)

# 将 ujson_ext 添加到 extensions 列表中
extensions.append(ujson_ext)

# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# pd_datetime

# 创建 pd_dt_ext Extension 对象
pd_dt_ext = Extension(
    "pandas._libs.pandas_datetime",
    depends=["pandas/_libs/tslibs/datetime/pd_datetime.h"],
    sources=(
        [
            "pandas/_libs/src/vendored/numpy/datetime/np_datetime.c",
            "pandas/_libs/src/vendored/numpy/datetime/np_datetime_strings.c",
            "pandas/_libs/src/datetime/date_conversions.c",
            "pandas/_libs/src/datetime/pd_datetime.c",
        ]
    ),
    include_dirs=[
        "pandas/_libs/include",
        numpy.get_include(),
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=macros,
)
    # 使用给定的宏定义列表来定义宏
    define_macros=macros,
# 将自定义的 pandas 扩展 pd_dt_ext 添加到扩展列表 extensions 中
extensions.append(pd_dt_ext)

# ----------------------------------------------------------------------
# pd_datetime

# 创建名为 pd_parser_ext 的 Extension 对象，用于编译 pandas 解析器模块
pd_parser_ext = Extension(
    "pandas._libs.pandas_parser",  # 模块名称
    depends=["pandas/_libs/include/pandas/parser/pd_parser.h"],  # 依赖的头文件
    sources=(
        [  # 源文件列表
            "pandas/_libs/src/parser/tokenizer.c",
            "pandas/_libs/src/parser/io.c",
            "pandas/_libs/src/parser/pd_parser.c",
        ]
    ),
    include_dirs=[
        "pandas/_libs/include",  # 包含的头文件目录
    ],
    extra_compile_args=(extra_compile_args),  # 额外的编译参数
    extra_link_args=extra_link_args,  # 额外的链接参数
    define_macros=macros,  # 宏定义
)

# 将 pd_parser_ext 添加到扩展列表 extensions 中
extensions.append(pd_parser_ext)

# ----------------------------------------------------------------------

# 如果当前脚本是作为主程序运行
if __name__ == "__main__":
    # 冻结支持并行编译，用于在 spawn 模式下代替 fork 时的支持
    multiprocessing.freeze_support()
    # 执行 setup 函数进行安装配置
    setup(
        version=versioneer.get_version(),  # 获取版本号
        ext_modules=maybe_cythonize(extensions, compiler_directives=directives),  # 编译扩展模块
        cmdclass=cmdclass,  # 自定义命令类
    )
```