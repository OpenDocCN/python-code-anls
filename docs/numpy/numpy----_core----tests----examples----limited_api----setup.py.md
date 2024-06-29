# `.\numpy\numpy\_core\tests\examples\limited_api\setup.py`

```py
"""
Build an example package using the limited Python C API.
"""

# 导入必要的库
import numpy as np  # 导入 NumPy 库
from setuptools import setup, Extension  # 导入 setuptools 库中的 setup 和 Extension 函数
import os  # 导入操作系统相关的库

# 定义宏，用于 Extension 对象的初始化
macros = [("NPY_NO_DEPRECATED_API", 0), ("Py_LIMITED_API", "0x03060000")]

# 创建 Extension 对象 limited_api
limited_api = Extension(
    "limited_api",  # 扩展模块名为 "limited_api"
    sources=[os.path.join('.', "limited_api.c")],  # 指定源文件为当前目录下的 limited_api.c
    include_dirs=[np.get_include()],  # 包含 NumPy 库的头文件目录
    define_macros=macros,  # 定义预处理宏
)

# 将 limited_api 加入到 extensions 列表中
extensions = [limited_api]

# 调用 setuptools 的 setup 函数，配置扩展模块
setup(
    ext_modules=extensions  # 指定扩展模块列表
)
```