# `.\numpy\tools\swig\test\setup.py`

```
#!/usr/bin/env python3
# System imports
# 导入 Extension 和 setup 函数
from distutils.core import Extension, setup

# Third-party modules - we depend on numpy for everything
# 导入 numpy 库
import numpy

# Obtain the numpy include directory.
# 获取 numpy 的头文件目录
numpy_include = numpy.get_include()

# Array extension module
# 定义名为 _Array 的扩展模块，包含多个源文件
_Array = Extension("_Array",
                   ["Array_wrap.cxx",
                    "Array1.cxx",
                    "Array2.cxx",
                    "ArrayZ.cxx"],
                   include_dirs=[numpy_include],
                   )

# Farray extension module
# 定义名为 _Farray 的扩展模块，包含多个源文件
_Farray = Extension("_Farray",
                    ["Farray_wrap.cxx",
                     "Farray.cxx"],
                    include_dirs=[numpy_include],
                    )

# _Vector extension module
# 定义名为 _Vector 的扩展模块，包含多个源文件
_Vector = Extension("_Vector",
                    ["Vector_wrap.cxx",
                     "Vector.cxx"],
                    include_dirs=[numpy_include],
                    )

# _Matrix extension module
# 定义名为 _Matrix 的扩展模块，包含多个源文件
_Matrix = Extension("_Matrix",
                    ["Matrix_wrap.cxx",
                     "Matrix.cxx"],
                    include_dirs=[numpy_include],
                    )

# _Tensor extension module
# 定义名为 _Tensor 的扩展模块，包含多个源文件
_Tensor = Extension("_Tensor",
                    ["Tensor_wrap.cxx",
                     "Tensor.cxx"],
                    include_dirs=[numpy_include],
                    )

# _Fortran extension module
# 定义名为 _Fortran 的扩展模块，包含多个源文件
_Fortran = Extension("_Fortran",
                    ["Fortran_wrap.cxx",
                     "Fortran.cxx"],
                    include_dirs=[numpy_include],
                    )

# _Flat extension module
# 定义名为 _Flat 的扩展模块，包含多个源文件
_Flat = Extension("_Flat",
                    ["Flat_wrap.cxx",
                     "Flat.cxx"],
                    include_dirs=[numpy_include],
                    )

# NumyTypemapTests setup
# 设置 NumyTypemapTests 包的信息和模块
setup(name="NumpyTypemapTests",
      description="Functions that work on arrays",
      author="Bill Spotz",
      py_modules=["Array", "Farray", "Vector", "Matrix", "Tensor",
                  "Fortran", "Flat"],
      ext_modules=[_Array, _Farray, _Vector, _Matrix, _Tensor,
                   _Fortran, _Flat]
      )
```