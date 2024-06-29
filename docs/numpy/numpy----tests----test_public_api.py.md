# `.\numpy\numpy\tests\test_public_api.py`

```
# 导入系统相关模块
import sys
# 系统配置模块
import sysconfig
# 子进程管理模块
import subprocess
# 包工具模块
import pkgutil
# 类型模块
import types
# 模块导入工具
import importlib
# 检查模块属性模块
import inspect
# 警告模块
import warnings

# 导入 NumPy 库
import numpy as np
# 导入 NumPy 库的别名
import numpy
# 导入 NumPy 测试模块
from numpy.testing import IS_WASM

# 导入 Pytest 测试框架
import pytest

# 尝试导入 ctypes 模块，如果失败则置为 None
try:
    import ctypes
except ImportError:
    ctypes = None


def check_dir(module, module_name=None):
    """Returns a mapping of all objects with the wrong __module__ attribute."""
    # 如果未提供模块名，则使用模块自身的名称
    if module_name is None:
        module_name = module.__name__
    # 结果字典
    results = {}
    # 遍历模块中的所有对象
    for name in dir(module):
        # 跳过名为 "core" 的特定对象
        if name == "core":
            continue
        # 获取对象
        item = getattr(module, name)
        # 检查对象是否有 '__module__' 和 '__name__' 属性，并且 '__module__' 属性不等于模块名
        if (hasattr(item, '__module__') and hasattr(item, '__name__')
                and item.__module__ != module_name):
            # 将不符合预期的对象添加到结果字典中
            results[name] = item.__module__ + '.' + item.__name__
    return results


def test_numpy_namespace():
    # 需要显示的允许列表，包含需要跳过显示的成员
    allowlist = {
        'recarray': 'numpy.rec.recarray',
        'show_config': 'numpy.__config__.show',
    }
    # 检查 NumPy 命名空间中的不符合预期的成员
    bad_results = check_dir(np)
    # 使用内置的断言检查 pytest 提供更好的错误消息
    assert bad_results == allowlist


@pytest.mark.skipif(IS_WASM, reason="can't start subprocess")
@pytest.mark.parametrize('name', ['testing'])
def test_import_lazy_import(name):
    """Make sure we can actually use the modules we lazy load.

    While not exported as part of the public API, it was accessible.  With the
    use of __getattr__ and __dir__, this isn't always true It can happen that
    an infinite recursion may happen.

    This is the only way I found that would force the failure to appear on the
    badly implemented code.

    We also test for the presence of the lazily imported modules in dir

    """
    # 构造子进程执行命令
    exe = (sys.executable, '-c', "import numpy; numpy." + name)
    # 执行子进程并获取输出结果
    result = subprocess.check_output(exe)
    # 使用断言确保结果为空
    assert not result

    # 确保延迟加载模块仍然在 __dir__ 中
    assert name in dir(np)


def test_dir_testing():
    """Assert that output of dir has only one "testing/tester"
    attribute without duplicate"""
    # 使用断言确保 dir 输出的内容没有重复项
    assert len(dir(np)) == len(set(dir(np)))


def test_numpy_linalg():
    # 检查 NumPy 线性代数模块中的不符合预期的成员
    bad_results = check_dir(np.linalg)
    assert bad_results == {}


def test_numpy_fft():
    # 检查 NumPy FFT 模块中的不符合预期的成员
    bad_results = check_dir(np.fft)
    assert bad_results == {}


@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available in this python")
def test_NPY_NO_EXPORT():
    # 使用 ctypes 加载 _multiarray_tests 模块
    cdll = ctypes.CDLL(np._core._multiarray_tests.__file__)
    # 确保某个 NPY_NO_EXPORT 函数被正确隐藏
    f = getattr(cdll, 'test_not_exported', None)
    assert f is None, ("'test_not_exported' is mistakenly exported, "
                      "NPY_NO_EXPORT does not work")


# Historically NumPy has not used leading underscores for private submodules
# much.  This has resulted in lots of things that look like public modules
# (i.e. things that can be imported as `import numpy.somesubmodule.somefile`),
# 定义包含公共模块名称的列表，这些模块要么本身是公共的，要么包含了公共的函数/对象，没有其他命名空间中的类似内容，
# 因此应该被视为公共的。
PUBLIC_MODULES = ['numpy.' + s for s in [
    "ctypeslib",                     # ctypeslib 模块
    "dtypes",                        # dtypes 模块
    "exceptions",                    # exceptions 模块
    "f2py",                          # f2py 模块
    "fft",                           # fft 模块
    "lib",                           # lib 模块
    "lib.array_utils",               # lib.array_utils 子模块
    "lib.format",                    # lib.format 子模块
    "lib.introspect",                # lib.introspect 子模块
    "lib.mixins",                    # lib.mixins 子模块
    "lib.npyio",                     # lib.npyio 子模块
    "lib.recfunctions",              # lib.recfunctions 模块，需要清理，2.0 版本时被遗忘
    "lib.scimath",                   # lib.scimath 子模块
    "lib.stride_tricks",             # lib.stride_tricks 子模块
    "linalg",                        # linalg 模块
    "ma",                            # ma 模块
    "ma.extras",                     # ma.extras 子模块
    "ma.mrecords",                   # ma.mrecords 子模块
    "polynomial",                    # polynomial 模块
    "polynomial.chebyshev",          # polynomial.chebyshev 子模块
    "polynomial.hermite",            # polynomial.hermite 子模块
    "polynomial.hermite_e",          # polynomial.hermite_e 子模块
    "polynomial.laguerre",           # polynomial.laguerre 子模块
    "polynomial.legendre",           # polynomial.legendre 子模块
    "polynomial.polynomial",         # polynomial.polynomial 子模块
    "random",                        # random 模块
    "strings",                       # strings 模块
    "testing",                       # testing 模块
    "testing.overrides",             # testing.overrides 子模块
    "typing",                        # typing 模块
    "typing.mypy_plugin",            # typing.mypy_plugin 子模块
    "version",                       # version 模块
]]
# 如果 Python 版本低于 3.12，则添加以下模块到公共模块列表中
if sys.version_info < (3, 12):
    PUBLIC_MODULES += [
        'numpy.' + s for s in [
            "distutils",               # distutils 模块
            "distutils.cpuinfo",       # distutils.cpuinfo 子模块
            "distutils.exec_command",  # distutils.exec_command 子模块
            "distutils.misc_util",     # distutils.misc_util 子模块
            "distutils.log",           # distutils.log 子模块
            "distutils.system_info",   # distutils.system_info 子模块
        ]
    ]

# 包含公共模块名称的列表，这些模块是使用别名访问的
PUBLIC_ALIASED_MODULES = [
    "numpy.char",                    # numpy.char 模块
    "numpy.emath",                   # numpy.emath 模块
    "numpy.rec",                     # numpy.rec 模块
]

# 定义包含私有但实际存在的模块名称的列表，这些模块看起来像是公共的（没有下划线），但不应使用
PRIVATE_BUT_PRESENT_MODULES = ['numpy.' + s for s in [
    "compat",                        # compat 模块
    "compat.py3k",                   # compat.py3k 模块
    "conftest",                      # conftest 模块
    "core",                          # core 模块
    "core.multiarray",               # core.multiarray 子模块
    "core.numeric",                  # core.numeric 子模块
    "core.umath",                    # core.umath 子模块
    "core.arrayprint",               # core.arrayprint 子模块
    "core.defchararray",             # core.defchararray 子模块
    "core.einsumfunc",               # core.einsumfunc 子模块
    "core.fromnumeric",              # core.fromnumeric 子模块
    "core.function_base",            # core.function_base 子模块
    "core.getlimits",                # core.getlimits 子模块
    "core.numerictypes",             # core.numerictypes 子模块
    "core.overrides",                # core.overrides 子模块
    "core.records",                  # core.records 子模块
    "core.shape_base",               # core.shape_base 子模块
    "f2py.auxfuncs",                 # f2py.auxfuncs 子模块
    "f2py.capi_maps",                # f2py.capi_maps 子模块
    "f2py.cb_rules",                 # f2py.cb_rules 子模块
    "f2py.cfuncs",                   # f2py.cfuncs 子模块
    "f2py.common_rules",             # f2py.common_rules 子模块
    "f2py.crackfortran",             # f2py.crackfortran 子模块
    "f2py.diagnose",                 # f2py.diagnose 子模块
    "f2py.f2py2e",                   # f2py.f2py2e 子模块
    "f2py.f90mod_rules",             # f2py.f90mod_rules 子模块
    "f2py.func2subr",                # f2py.func2subr 子模块
    "f2py.rules",                    # f2py.rules 子模块
    "f2py.symbolic",                 # f2py.symbolic 子模块
    "f2py.use_rules",                # f2py.use_rules 子模块
    "fft.helper",                    # fft.helper 模块
    "lib.user_array",                # lib.user_array 模块，注意：不在 np.lib 中，但可能应该被删除
    "linalg.lapack_lite",            # linalg.lapack_lite 模块
    "linalg.linalg",                 # linalg.linalg 模块
    "ma.core",                       # ma.core 模块
    "ma.testutils",                  # ma.testutils 模块
    "ma.timer_comparison",           # ma.timer_comparison 模块
    "matlib",                        # matlib 模块
    "matrixlib",                     # matrixlib 模块
    "matrixlib.defmatrix",           # matrixlib.defmatrix 模块
    "polynomial.polyutils",          # polynomial.polyutils 模块
    "random.mtrand",                 # random.mtrand 模块
    "random.bit_generator",          # random.bit_generator 模块
    "testing.print_coercion_tables", # testing.print_coercion_tables 模块
]]
# 如果 Python 版本低于 3.12，则添加额外的模块到私有模块列表中
if sys.version_info < (3, 12):
    # 这部分的代码在示例中没有提供，但根据上下文和模式推测可能是继续添加私有模块的逻辑
    pass
    # 将特定的模块名列表添加到 PRIVATE_BUT_PRESENT_MODULES 列表中
    PRIVATE_BUT_PRESENT_MODULES += [
        # 使用列表推导式生成模块名，每个模块名以 'numpy.' 开头
        'numpy.' + s for s in [
            # 下面是各个模块的具体名称
            "distutils.armccompiler",
            "distutils.fujitsuccompiler",
            "distutils.ccompiler",
            'distutils.ccompiler_opt',
            "distutils.command",
            "distutils.command.autodist",
            "distutils.command.bdist_rpm",
            "distutils.command.build",
            "distutils.command.build_clib",
            "distutils.command.build_ext",
            "distutils.command.build_py",
            "distutils.command.build_scripts",
            "distutils.command.build_src",
            "distutils.command.config",
            "distutils.command.config_compiler",
            "distutils.command.develop",
            "distutils.command.egg_info",
            "distutils.command.install",
            "distutils.command.install_clib",
            "distutils.command.install_data",
            "distutils.command.install_headers",
            "distutils.command.sdist",
            "distutils.conv_template",
            "distutils.core",
            "distutils.extension",
            "distutils.fcompiler",
            "distutils.fcompiler.absoft",
            "distutils.fcompiler.arm",
            "distutils.fcompiler.compaq",
            "distutils.fcompiler.environment",
            "distutils.fcompiler.g95",
            "distutils.fcompiler.gnu",
            "distutils.fcompiler.hpux",
            "distutils.fcompiler.ibm",
            "distutils.fcompiler.intel",
            "distutils.fcompiler.lahey",
            "distutils.fcompiler.mips",
            "distutils.fcompiler.nag",
            "distutils.fcompiler.none",
            "distutils.fcompiler.pathf95",
            "distutils.fcompiler.pg",
            "distutils.fcompiler.nv",
            "distutils.fcompiler.sun",
            "distutils.fcompiler.vast",
            "distutils.fcompiler.fujitsu",
            "distutils.from_template",
            "distutils.intelccompiler",
            "distutils.lib2def",
            "distutils.line_endings",
            "distutils.mingw32ccompiler",
            "distutils.msvccompiler",
            "distutils.npy_pkg_config",
            "distutils.numpy_distribution",
            "distutils.pathccompiler",
            "distutils.unixccompiler",
        ]
    ]
# 检查给定模块名是否应考虑在内，若模块名包含 '._', '.tests' 或 '.setup'，则不需要考虑
def is_unexpected(name):
    if '._' in name or '.tests' in name or '.setup' in name:
        return False

    # 若模块名在 PUBLIC_MODULES 中，则不需要考虑
    if name in PUBLIC_MODULES:
        return False

    # 若模块名在 PUBLIC_ALIASED_MODULES 中，则不需要考虑
    if name in PUBLIC_ALIASED_MODULES:
        return False

    # 若模块名在 PRIVATE_BUT_PRESENT_MODULES 中，则不需要考虑
    if name in PRIVATE_BUT_PRESENT_MODULES:
        return False

    # 其他情况下需要考虑
    return True


# 根据 Python 版本决定跳过列表的内容
if sys.version_info < (3, 12):
    SKIP_LIST = ["numpy.distutils.msvc9compiler"]
else:
    SKIP_LIST = []


# 忽略来自废弃模块的警告
@pytest.mark.filterwarnings("ignore:.*np.compat.*:DeprecationWarning")
def test_all_modules_are_expected():
    """
    测试确保不会意外添加新的公共模块。检查基于文件名。
    """

    # 存储不符合预期的模块名
    modnames = []

    # 使用 pkgutil.walk_packages 遍历 numpy 的路径下的所有包和模块
    for _, modname, ispkg in pkgutil.walk_packages(path=np.__path__,
                                                   prefix=np.__name__ + '.',
                                                   onerror=None):
        # 如果模块名符合不符合预期的条件，并且不在跳过列表中，则添加到 modnames 中
        if is_unexpected(modname) and modname not in SKIP_LIST:
            modnames.append(modname)

    # 如果 modnames 不为空，则抛出 AssertionError，显示找到的意外模块名
    if modnames:
        raise AssertionError(f'Found unexpected modules: {modnames}')


# 下面的测试用例会检测明显不应在 API 中的内容
SKIP_LIST_2 = [
    'numpy.lib.math',
    'numpy.matlib.char',
    'numpy.matlib.rec',
    'numpy.matlib.emath',
    'numpy.matlib.exceptions',
    'numpy.matlib.math',
    'numpy.matlib.linalg',
    'numpy.matlib.fft',
    'numpy.matlib.random',
    'numpy.matlib.ctypeslib',
    'numpy.matlib.ma',
]

# 根据 Python 版本决定扩展 SKIP_LIST_2 的内容
if sys.version_info < (3, 12):
    SKIP_LIST_2 += [
        'numpy.distutils.log.sys',
        'numpy.distutils.log.logging',
        'numpy.distutils.log.warnings',
    ]


def test_all_modules_are_expected_2():
    """
    方法用于检查所有对象。与 test_all_modules_are_expected 中的基于 pkgutil 的方法不同，
    此测试更全面，检查类似于 import .lib.scimath as emath 这样的情况。
    """
    # 定义一个函数，用于查找指定模块中未预期的成员（通常是模块）
    def find_unexpected_members(mod_name):
        # 初始化一个空列表，用于存储未预期的成员
        members = []
        # 动态导入指定名称的模块
        module = importlib.import_module(mod_name)
        # 检查模块是否定义了 __all__ 属性，如果有则使用该属性作为对象名列表，否则使用 dir() 函数获取所有对象名
        if hasattr(module, '__all__'):
            objnames = module.__all__
        else:
            objnames = dir(module)
    
        # 遍历模块中的每个对象名
        for objname in objnames:
            # 排除私有对象（以 '_' 开头的对象名）
            if not objname.startswith('_'):
                # 构建完整的对象名，格式为 模块名.对象名
                fullobjname = mod_name + '.' + objname
                # 检查对象是否为模块类型
                if isinstance(getattr(module, objname), types.ModuleType):
                    # 如果是未预期的模块，则进一步检查是否应该跳过
                    if is_unexpected(fullobjname):
                        if fullobjname not in SKIP_LIST_2:
                            # 将符合条件的未预期模块添加到成员列表中
                            members.append(fullobjname)
    
        # 返回所有发现的未预期模块列表
        return members
    
    # 调用 find_unexpected_members 函数查找 "numpy" 模块中的未预期成员
    unexpected_members = find_unexpected_members("numpy")
    # 遍历预定义的公共模块列表，对每个模块调用 find_unexpected_members 函数，并将结果扩展到 unexpected_members 列表中
    for modname in PUBLIC_MODULES:
        unexpected_members.extend(find_unexpected_members(modname))
    
    # 如果 unexpected_members 列表非空，则抛出断言错误，指示找到了未预期的对象（通常是模块）
    if unexpected_members:
        raise AssertionError("Found unexpected object(s) that look like "
                             "modules: {}".format(unexpected_members))
def test_api_importable():
    """
    Check that all submodules listed higher up in this file can be imported

    Note that if a PRIVATE_BUT_PRESENT_MODULES entry goes missing, it may
    simply need to be removed from the list (deprecation may or may not be
    needed - apply common sense).
    """
    # 定义内部函数，检查模块是否可导入
    def check_importable(module_name):
        try:
            importlib.import_module(module_name)
        except (ImportError, AttributeError):
            return False

        return True

    # 初始化模块名列表
    module_names = []
    # 遍历公共模块列表，检查每个模块是否可导入
    for module_name in PUBLIC_MODULES:
        if not check_importable(module_name):
            module_names.append(module_name)

    # 如果有未能导入的模块，则抛出断言错误
    if module_names:
        raise AssertionError("Modules in the public API that cannot be "
                             "imported: {}".format(module_names))

    # 遍历公共别名模块列表，尝试评估每个模块名
    for module_name in PUBLIC_ALIASED_MODULES:
        try:
            eval(module_name)
        except AttributeError:
            module_names.append(module_name)

    # 如果有未找到的模块，则抛出断言错误
    if module_names:
        raise AssertionError("Modules in the public API that were not "
                             "found: {}".format(module_names))

    # 使用警告模块捕获警告
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', category=DeprecationWarning)
        warnings.filterwarnings('always', category=ImportWarning)
        # 遍历私有但存在的模块列表，检查每个模块是否可导入
        for module_name in PRIVATE_BUT_PRESENT_MODULES:
            if not check_importable(module_name):
                module_names.append(module_name)

    # 如果有不应公开但看起来公开且无法导入的模块，则抛出断言错误
    if module_names:
        raise AssertionError("Modules that are not really public but looked "
                             "public and can not be imported: "
                             "{}".format(module_names))


@pytest.mark.xfail(
    sysconfig.get_config_var("Py_DEBUG") not in (None, 0, "0"),
    reason=(
        "NumPy possibly built with `USE_DEBUG=True ./tools/travis-test.sh`, "
        "which does not expose the `array_api` entry point. "
        "See https://github.com/numpy/numpy/pull/19800"
    ),
)
def test_array_api_entry_point():
    """
    Entry point for Array API implementation can be found with importlib and
    returns the main numpy namespace.
    """
    # 对于没有通过 meson-python 安装的开发版本，确保这个测试仅在 numpy 位于 site-packages 中时失败
    numpy_in_sitepackages = sysconfig.get_path('platlib') in np.__file__

    # 获取所有入口点
    eps = importlib.metadata.entry_points()
    try:
        # 选择数组 API 组的入口点
        xp_eps = eps.select(group="array_api")
    except AttributeError:
        # 如果 select 接口在当前 Python 版本不可用，回退到 dict 键的方式查找数组 API 的入口点
        # 详见 https://github.com/numpy/numpy/pull/19800
        xp_eps = eps.get("array_api", [])
    # 如果 xp_eps 列表为空，则执行以下操作
    if len(xp_eps) == 0:
        # 如果 numpy 安装在 site-packages 中，则抛出错误消息
        if numpy_in_sitepackages:
            msg = "No entry points for 'array_api' found"
            raise AssertionError(msg) from None
        # 如果不是在 site-packages 中，直接返回
        return

    # 尝试从 xp_eps 中找到名为 "numpy" 的 entry point
    try:
        ep = next(ep for ep in xp_eps if ep.name == "numpy")
    except StopIteration:
        # 如果找不到名为 "numpy" 的 entry point，则根据情况抛出错误消息
        if numpy_in_sitepackages:
            msg = "'numpy' not in array_api entry points"
            raise AssertionError(msg) from None
        # 如果不是在 site-packages 中，直接返回
        return

    # 如果 entry point 的值是 'numpy.array_api'，表示当前 numpy 构建的 entry point 已经安装
    if ep.value == 'numpy.array_api':
        # 在进行 spin 或者 in-place 构建时，旧的 numpy 可能指向不存在的位置，这种情况下不会出错
        # 直接返回
        return

    # 加载 entry point 对应的模块或对象
    xp = ep.load()
    # 准备错误消息，如果加载的 xp 不是 numpy 对象
    msg = (
        f"numpy entry point value '{ep.value}' "
        "does not point to our Array API implementation"
    )
    # 断言加载的 xp 必须是 numpy 对象，否则抛出错误消息
    assert xp is numpy, msg
def test_main_namespace_all_dir_coherence():
    """
    检查 `dir(np)` 和 `np.__all__` 是否一致，并返回相同内容，排除异常和私有成员。
    """
    # 过滤掉私有成员
    def _remove_private_members(member_set):
        return {m for m in member_set if not m.startswith('_')}

    # 过滤掉异常成员
    def _remove_exceptions(member_set):
        return member_set.difference({
            "bool"  # 只在 __dir__ 中包含的成员
        })

    # 从 np.__all__ 中移除私有成员和异常成员
    all_members = _remove_private_members(np.__all__)
    all_members = _remove_exceptions(all_members)

    # 从 dir(np) 中移除私有成员和异常成员
    dir_members = _remove_private_members(np.__dir__())
    dir_members = _remove_exceptions(dir_members)

    # 断言：np.__all__ 和 dir(np) 的内容应该一致
    assert all_members == dir_members, (
        "破坏对称性的成员: "
        f"{all_members.symmetric_difference(dir_members)}"
    )


@pytest.mark.filterwarnings(
    r"ignore:numpy.core(\.\w+)? is deprecated:DeprecationWarning"
)
def test_core_shims_coherence():
    """
    检查 `numpy._core` 的所有“半公共”成员是否也能从 `numpy.core` shims 中访问。
    """
    import numpy.core as core

    # 遍历 np._core 的所有成员
    for member_name in dir(np._core):
        # 跳过私有成员、测试成员和已别名化的模块
        if (
            member_name.startswith("_")
            or member_name in ["tests", "strings"]
            or f"numpy.{member_name}" in PUBLIC_ALIASED_MODULES 
        ):
            continue

        # 获取成员对象
        member = getattr(np._core, member_name)

        # 如果是模块，则遍历其成员
        if inspect.ismodule(member):
            submodule = member
            submodule_name = member_name
            for submodule_member_name in dir(submodule):
                # 忽略双下划线开头的名称
                if submodule_member_name.startswith("__"):
                    continue
                # 获取子模块成员
                submodule_member = getattr(submodule, submodule_member_name)

                # 动态导入对应的 numpy.core 子模块
                core_submodule = __import__(
                    f"numpy.core.{submodule_name}",
                    fromlist=[submodule_member_name]
                )

                # 断言：子模块成员应与 np.core 中对应成员相等
                assert submodule_member is getattr(
                    core_submodule, submodule_member_name
                )

        else:
            # 断言：成员应与 np.core 中对应成员相等
            assert member is getattr(core, member_name)


def test_functions_single_location():
    """
    检查每个公共函数是否只能从一个位置访问。

    测试执行 BFS 搜索遍历 NumPy 的公共 API，标记从多个位置可访问的函数对象。
    """
    from typing import Any, Callable, Dict, List, Set, Tuple
    from numpy._core._multiarray_umath import (
        _ArrayFunctionDispatcher as dispatched_function
    )

    # 记录访问过的模块和函数
    visited_modules: Set[types.ModuleType] = {np}
    visited_functions: Set[Callable[..., Any]] = set()
    # Functions often have `__name__` overridden, therefore we need
    # to keep track of locations where functions have been found.
    # 定义一个字典，用于存储每个函数对象及其对应的原始路径
    functions_original_paths: Dict[Callable[..., Any], str] = dict()
    
    # Here we aggregate functions with more than one location.
    # It must be empty for the test to pass.
    # 用于存储存在多个位置定义的函数的列表，用于测试时必须为空
    duplicated_functions: List[Tuple] = []
    
    # 初始化一个模块队列，起始包含 numpy 模块
    modules_queue = [np]
    
    # 清除变量 visited_functions, visited_modules, functions_original_paths
    del visited_functions, visited_modules, functions_original_paths
    
    # 断言确保 duplicated_functions 列表为空，用于测试验证
    assert len(duplicated_functions) == 0, duplicated_functions
```