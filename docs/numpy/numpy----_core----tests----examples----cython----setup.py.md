# `.\numpy\numpy\_core\tests\examples\cython\setup.py`

```
"""
Provide python-space access to the functions exposed in numpy/__init__.pxd
for testing.
"""

# 导入所需的库
import numpy as np  # 导入NumPy库
from distutils.core import setup  # 导入setup函数，用于设置和安装扩展
from Cython.Build import cythonize  # 导入cythonize函数，用于将Cython代码编译为扩展模块
from setuptools.extension import Extension  # 导入Extension类，用于定义扩展模块
import os  # 导入os模块，用于操作系统相关功能

# 定义宏定义列表
macros = [
    ("NPY_NO_DEPRECATED_API", 0),  # 不使用已弃用的API
    # 要求NumPy版本至少为1.25以测试日期时间功能的新增
    ("NPY_TARGET_VERSION", "NPY_2_0_API_VERSION"),
]

# 定义Extension对象
checks = Extension(
    "checks",  # 扩展模块名为"checks"
    sources=[os.path.join('.', "checks.pyx")],  # 扩展模块的源文件为当前目录下的"checks.pyx"
    include_dirs=[np.get_include()],  # 包含NumPy头文件目录
    define_macros=macros,  # 使用前面定义的宏定义列表
)

# 将定义好的Extension对象放入列表中
extensions = [checks]

# 调用setup函数，配置和安装扩展模块
setup(
    ext_modules=cythonize(extensions)  # 将Extension对象列表传递给cythonize函数进行编译
)
```