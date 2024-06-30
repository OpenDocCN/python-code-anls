# `D:\src\scipysrc\scipy\scipy\_lib\tests\test_public_api.py`

```
"""
This test script is adopted from:
    https://github.com/numpy/numpy/blob/main/numpy/tests/test_public_api.py
"""

import pkgutil  # 导入 Python 内置模块 pkgutil，用于访问包的工具和工具包
import types  # 导入 Python 内置模块 types，用于操作 Python 类型信息
import importlib  # 导入 Python 内置模块 importlib，用于导入其他模块
import warnings  # 导入 Python 内置模块 warnings，用于管理警告
from importlib import import_module  # 从 importlib 模块导入 import_module 函数

import pytest  # 导入 pytest 库，用于编写和运行测试

import scipy  # 导入 SciPy 库

from scipy.conftest import xp_available_backends  # 从 scipy.conftest 模块导入 xp_available_backends 变量


def test_dir_testing():
    """Assert that output of dir has only one "testing/tester"
    attribute without duplicate"""
    assert len(dir(scipy)) == len(set(dir(scipy)))  # 断言 SciPy 的 dir() 输出没有重复项

# Historically SciPy has not used leading underscores for private submodules
# much.  This has resulted in lots of things that look like public modules
# (i.e. things that can be imported as `import scipy.somesubmodule.somefile`),
# but were never intended to be public.  The PUBLIC_MODULES list contains
# modules that are either public because they were meant to be, or because they
# contain public functions/objects that aren't present in any other namespace
# for whatever reason and therefore should be treated as public.
# Historically SciPy has not used leading underscores for private submodules
# much.  This has resulted in lots of things that look like public modules
因 PUBLIC_MODULES = Specializations modules Sci following  containing
    # 导入大量的SciPy子模块，包括MATLAB文件读取、线性代数、优化、信号处理、稀疏矩阵等功能
    'scipy.io.matlab.mio4',
    'scipy.io.matlab.mio5',
    'scipy.io.matlab.mio5_params',
    'scipy.io.matlab.mio5_utils',
    'scipy.io.matlab.mio_utils',
    'scipy.io.matlab.miobase',
    'scipy.io.matlab.streams',
    'scipy.io.mmio',
    'scipy.io.netcdf',
    'scipy.linalg.basic',
    'scipy.linalg.decomp',
    'scipy.linalg.decomp_cholesky',
    'scipy.linalg.decomp_lu',
    'scipy.linalg.decomp_qr',
    'scipy.linalg.decomp_schur',
    'scipy.linalg.decomp_svd',
    'scipy.linalg.matfuncs',
    'scipy.linalg.misc',
    'scipy.linalg.special_matrices',
    'scipy.misc.common',
    'scipy.misc.doccer',
    'scipy.ndimage.filters',
    'scipy.ndimage.fourier',
    'scipy.ndimage.interpolation',
    'scipy.ndimage.measurements',
    'scipy.ndimage.morphology',
    'scipy.odr.models',
    'scipy.odr.odrpack',
    'scipy.optimize.cobyla',
    'scipy.optimize.cython_optimize',
    'scipy.optimize.lbfgsb',
    'scipy.optimize.linesearch',
    'scipy.optimize.minpack',
    'scipy.optimize.minpack2',
    'scipy.optimize.moduleTNC',
    'scipy.optimize.nonlin',
    'scipy.optimize.optimize',
    'scipy.optimize.slsqp',
    'scipy.optimize.tnc',
    'scipy.optimize.zeros',
    'scipy.signal.bsplines',
    'scipy.signal.filter_design',
    'scipy.signal.fir_filter_design',
    'scipy.signal.lti_conversion',
    'scipy.signal.ltisys',
    'scipy.signal.signaltools',
    'scipy.signal.spectral',
    'scipy.signal.spline',
    'scipy.signal.waveforms',
    'scipy.signal.wavelets',
    'scipy.signal.windows.windows',
    'scipy.sparse.base',
    'scipy.sparse.bsr',
    'scipy.sparse.compressed',
    'scipy.sparse.construct',
    'scipy.sparse.coo',
    'scipy.sparse.csc',
    'scipy.sparse.csr',
    'scipy.sparse.data',
    'scipy.sparse.dia',
    'scipy.sparse.dok',
    'scipy.sparse.extract',
    'scipy.sparse.lil',
    'scipy.sparse.linalg.dsolve',
    'scipy.sparse.linalg.eigen',
    'scipy.sparse.linalg.interface',
    'scipy.sparse.linalg.isolve',
    'scipy.sparse.linalg.matfuncs',
    'scipy.sparse.sparsetools',
    'scipy.sparse.spfuncs',
    'scipy.sparse.sputils',
    'scipy.spatial.ckdtree',
    'scipy.spatial.kdtree',
    'scipy.spatial.qhull',
    'scipy.spatial.transform.rotation',
    'scipy.special.add_newdocs',
    'scipy.special.basic',
    'scipy.special.cython_special',
    'scipy.special.orthogonal',
    'scipy.special.sf_error',
    'scipy.special.specfun',
    'scipy.special.spfun_stats',
    'scipy.stats.biasedurn',
    'scipy.stats.kde',
    'scipy.stats.morestats',
    'scipy.stats.mstats_basic',
    'scipy.stats.mstats_extras',
    'scipy.stats.mvn',
    'scipy.stats.stats',
# 检查给定模块名是否需要考虑在内，根据一些标准判断
def is_unexpected(name):
    # 排除以 '._', '.tests', '.setup' 结尾的模块名
    if '._' in name or '.tests' in name or '.setup' in name:
        return False
    
    # 排除在 PUBLIC_MODULES 列表中的模块名
    if name in PUBLIC_MODULES:
        return False
    
    # 排除在 PRIVATE_BUT_PRESENT_MODULES 列表中的模块名
    if name in PRIVATE_BUT_PRESENT_MODULES:
        return False
    
    # 其他情况视为需要考虑的模块名
    return True


# 定义不需要进行模块检查的模块名列表
SKIP_LIST = [
    'scipy.conftest',
    'scipy.version',
    'scipy.special.libsf_error_state'
]


# XXX: this test does more than it says on the tin - in using `pkgutil.walk_packages`,
# it will raise if it encounters any exceptions which are not handled by `ignore_errors`
# while attempting to import each discovered package.
# For now, `ignore_errors` only ignores what is necessary, but this could be expanded -
# for example, to all errors from private modules or git subpackages - if desired.
# 测试确保不会意外添加类似新公共模块的内容。检查基于文件名。
def test_all_modules_are_expected():
    """
    Test that we don't add anything that looks like a new public module by
    accident.  Check is based on filenames.
    """
    
    # 定义一个函数，用于忽略特定错误的导入
    def ignore_errors(name):
        # 如果安装了与当前 NumPy 版本不兼容的其他数组库版本，
        # 在导入 `array_api_compat` 时可能会出现错误。
        # 这仅在 SciPy 配置了该库作为可用后端时才会引发。
        backends = {'cupy': 'cupy',
                    'pytorch': 'torch',
                    'dask.array': 'dask.array'}
        for backend, dir_name in backends.items():
            path = f'array_api_compat.{dir_name}'
            if path in name and backend not in xp_available_backends:
                return
        raise
    
    modnames = []

    # 遍历 scipy 路径下的所有包，并通过 ignore_errors 函数来处理可能的异常
    for _, modname, _ in pkgutil.walk_packages(path=scipy.__path__,
                                               prefix=scipy.__name__ + '.',
                                               onerror=ignore_errors):
        # 如果模块名被视为意外并且不在 SKIP_LIST 中，则添加到 modnames 列表中
        if is_unexpected(modname) and modname not in SKIP_LIST:
            modnames.append(modname)
    
    # 如果发现了意外模块名，则引发断言错误
    if modnames:
        raise AssertionError(f'Found unexpected modules: {modnames}')


# Stuff that clearly shouldn't be in the API and is detected by the next test
# below
# 定义另一个不需要进行模块检查的模块名列表
SKIP_LIST_2 = [
    'scipy.char',
    'scipy.rec',
    'scipy.emath',
    'scipy.math',
    'scipy.random',
    'scipy.ctypeslib',
    'scipy.ma'
]


def test_all_modules_are_expected_2():
    """
    Method checking all objects. The pkgutil-based method in
    `test_all_modules_are_expected` does not catch imports into a namespace,
    only filenames.
    """
    # 定义一个函数，用于查找给定模块中未预期的成员（函数、类等）
    def find_unexpected_members(mod_name):
        # 初始化一个空列表，用于存储找到的未预期成员
        members = []
        # 动态导入指定名称的模块
        module = importlib.import_module(mod_name)
        # 检查模块是否定义了 '__all__' 属性，如果有则使用它，否则使用 dir(module) 的结果
        if hasattr(module, '__all__'):
            objnames = module.__all__
        else:
            objnames = dir(module)

        # 遍历模块中的所有名称
        for objname in objnames:
            # 排除以 '_' 开头的名称
            if not objname.startswith('_'):
                # 构建完整的对象名称，形如 '模块名.对象名'
                fullobjname = mod_name + '.' + objname
                # 检查对象是否为模块类型
                if isinstance(getattr(module, objname), types.ModuleType):
                    # 检查该模块是否是未预期的，并且不在 SKIP_LIST_2 中
                    if is_unexpected(fullobjname) and fullobjname not in SKIP_LIST_2:
                        # 将符合条件的模块名称添加到列表中
                        members.append(fullobjname)

        # 返回找到的所有未预期成员列表
        return members

    # 查找 "scipy" 模块中的未预期成员
    unexpected_members = find_unexpected_members("scipy")
    # 遍历 PUBLIC_MODULES 列表中的每个模块，查找它们的未预期成员并加入到列表中
    for modname in PUBLIC_MODULES:
        unexpected_members.extend(find_unexpected_members(modname))

    # 如果存在未预期的成员，抛出断言错误，显示找到的未预期模块名称列表
    if unexpected_members:
        raise AssertionError("Found unexpected object(s) that look like "
                             f"modules: {unexpected_members}")
# 检查所有在本文件中更高级别列出的子模块是否可以被导入
# 如果 PRIVATE_BUT_PRESENT_MODULES 中的条目消失，可能只需从列表中删除它（可能需要进行弃用 - 应用常识）。

def test_api_importable():
    def check_importable(module_name):
        try:
            importlib.import_module(module_name)
        except (ImportError, AttributeError):
            return False
        return True

    # 初始化一个空列表来存储不能导入的模块名称
    module_names = []

    # 遍历 PUBLIC_MODULES 中的每个模块名称
    for module_name in PUBLIC_MODULES:
        # 检查模块是否可以导入，如果不能，将模块名称添加到 module_names 中
        if not check_importable(module_name):
            module_names.append(module_name)

    # 如果有不能导入的模块，抛出 AssertionError 异常
    if module_names:
        raise AssertionError("Modules in the public API that cannot be "
                             f"imported: {module_names}")

    # 使用 warnings.catch_warnings() 捕获警告消息
    with warnings.catch_warnings(record=True):
        # 设置特定警告的过滤器
        warnings.filterwarnings('always', category=DeprecationWarning)
        warnings.filterwarnings('always', category=ImportWarning)
        
        # 遍历 PRIVATE_BUT_PRESENT_MODULES 中的每个模块名称
        for module_name in PRIVATE_BUT_PRESENT_MODULES:
            # 检查模块是否可以导入，如果不能，将模块名称添加到 module_names 中
            if not check_importable(module_name):
                module_names.append(module_name)

    # 如果有不能导入的模块，抛出 AssertionError 异常
    if module_names:
        raise AssertionError("Modules that are not really public but looked "
                             "public and can not be imported: "
                             f"{module_names}")


def test_private_but_present_deprecation(module_name, correct_module):
    # 解决了 gh-18279、gh-17572、gh-17771 中提到的来自私有模块的弃用警告误导问题。
    # 检查这是否已解决。

    # 导入指定名称的模块
    module = import_module(module_name)

    # 根据 correct_module 是否为 None 确定正确的导入名称
    if correct_module is None:
        import_name = f'scipy.{module_name.split(".")[1]}'
    else:
        import_name = f'scipy.{module_name.split(".")[1]}.{correct_module}'

    # 导入正确的模块
    correct_import = import_module(import_name)

    # 遍历 module.__all__ 中列出的所有属性名称
    for attr_name in module.__all__:
        # 对于特定情况 "varmats_from_mat"，延迟处理，参见指定的 GitHub 问题链接
        if attr_name == "varmats_from_mat":
            continue
        
        # 确保正确的导入中存在该属性
        assert getattr(correct_import, attr_name, None) is not None
        
        # 构造警告信息消息，提示用户从正确的模块中导入属性
        message = f"Please import `{attr_name}` from the `{import_name}`..."
        
        # 使用 pytest.deprecated_call 匹配警告消息，确保从 module 中获取 attr_name 时会出现警告
        with pytest.deprecated_call(match=message):
            getattr(module, attr_name)

    # 对于不在 module_name 中的属性，在获取时应抛出 AttributeError 异常，匹配特定的消息
    message = f"`{module_name}` is deprecated..."
    with pytest.raises(AttributeError, match=message):
        getattr(module, "ekki")


def test_misc_doccer_deprecation():
    # 解决了 gh-18279、gh-17572、gh-17771 中提到的来自私有模块的弃用警告误导问题。
    # 检查这是否已解决。
    # 导入 `scipy.misc.doccer` 模块
    module = import_module('scipy.misc.doccer')
    # 导入正确的模块 `scipy._lib.doccer`
    correct_import = import_module('scipy._lib.doccer')

    # 遍历 `module.__all__` 中定义的所有属性名
    for attr_name in module.__all__:
        # 尝试从 `scipy._lib.doccer` 中获取同名属性
        attr = getattr(correct_import, attr_name, None)
        # 如果属性不存在于 `scipy._lib.doccer` 中，生成相应的警告信息
        if attr is None:
            message = f"`scipy.misc.{attr_name}` is deprecated..."
        else:
            # 如果属性存在于 `scipy._lib.doccer` 中，生成不同的警告信息
            message = f"Please import `{attr_name}` from the `scipy._lib.doccer`..."
        # 使用 `pytest.deprecated_call` 检查是否产生了特定的警告消息
        with pytest.deprecated_call(match=message):
            # 获取 `module` 中的同名属性，可能会触发警告
            getattr(module, attr_name)

    # 尝试获取 `module` 中不存在的属性 "ekki"
    # 应该引发 `AttributeError`，指示 `scipy.misc.doccer` 已被弃用
    message = "`scipy.misc.doccer` is deprecated..."
    with pytest.raises(AttributeError, match=message):
        getattr(module, "ekki")
```