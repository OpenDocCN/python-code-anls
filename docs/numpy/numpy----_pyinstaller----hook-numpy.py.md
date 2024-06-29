# `.\numpy\numpy\_pyinstaller\hook-numpy.py`

```py
# 导入必要的模块和函数
"""This hook should collect all binary files and any hidden modules that numpy
needs.

Our (some-what inadequate) docs for writing PyInstaller hooks are kept here:
https://pyinstaller.readthedocs.io/en/stable/hooks.html

"""
from PyInstaller.compat import is_conda, is_pure_conda  # 导入 PyInstaller 兼容性模块中的 is_conda 和 is_pure_conda 函数
from PyInstaller.utils.hooks import collect_dynamic_libs, is_module_satisfies  # 导入 PyInstaller 工具模块中的 collect_dynamic_libs 和 is_module_satisfies 函数

# 收集 numpy 安装文件夹中所有的 DLL 文件，并将它们放置在构建后的应用程序根目录
binaries = collect_dynamic_libs("numpy", ".")

# 如果使用 Conda 而没有任何非 Conda 虚拟环境管理器：
if is_pure_conda:
    # 假定从 Conda-forge 运行 NumPy，并从共享的 Conda bin 目录收集其 DLL 文件。必须同时收集 NumPy 依赖项的 DLL 文件，以捕获 MKL、OpenBlas、OpenMP 等。
    from PyInstaller.utils.hooks import conda_support
    datas = conda_support.collect_dynamic_libs("numpy", dependencies=True)

# PyInstaller 无法检测到的子模块。'_dtype_ctypes' 仅从 C 语言中导入，'_multiarray_tests' 用于测试（不会被打包）。
hiddenimports = ['numpy._core._dtype_ctypes', 'numpy._core._multiarray_tests']

# 移除 NumPy 中引用但实际上并非依赖的测试和构建代码及包。
excludedimports = [
    "scipy",
    "pytest",
    "f2py",
    "setuptools",
    "numpy.f2py",
    "distutils",
    "numpy.distutils",
]
```