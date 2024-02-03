# `.\PaddleOCR\ppocr\postprocess\pse_postprocess\pse\setup.py`

```
# 导入必要的模块
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# 设置 Cython 编译的参数和选项
setup(
    # 将 Cython 编译后的扩展模块添加到 setup 中
    ext_modules=cythonize(Extension(
        'pse',  # 模块名
        sources=['pse.pyx'],  # Cython 源文件
        language='c++',  # 使用 C++ 语言
        include_dirs=[numpy.get_include()],  # 包含 numpy 库的头文件路径
        library_dirs=[],  # 库文件路径
        libraries=[],  # 需要链接的库
        extra_compile_args=['-O3'],  # 额外的编译参数，这里是优化级别为 3
        extra_link_args=[]  # 额外的链接参数
    ))
)
```