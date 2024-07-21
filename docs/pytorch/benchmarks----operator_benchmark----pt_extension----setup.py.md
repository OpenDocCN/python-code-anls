# `.\pytorch\benchmarks\operator_benchmark\pt_extension\setup.py`

```py
# 导入需要的模块和函数，从setuptools库中导入setup函数
from setuptools import setup

# 从torch.utils.cpp_extension库中导入BuildExtension和CppExtension类
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 调用setup函数来配置和安装Python包
setup(
    # 设置包的名称为"benchmark_cpp_extension"
    name="benchmark_cpp_extension",
    # 定义扩展模块的列表，包括一个CppExtension对象
    ext_modules=[CppExtension("benchmark_cpp_extension", ["extension.cpp"])],
    # 指定cmdclass参数，用于定义和配置构建扩展的类，这里使用BuildExtension类
    cmdclass={"build_ext": BuildExtension},
)
```