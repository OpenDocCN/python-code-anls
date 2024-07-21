# `.\pytorch\test\cpp_extensions\no_python_abi_suffix_test\setup.py`

```py
# 导入 setuptools 库中的 setup 函数，用于配置 Python 包的安装和发布
from setuptools import setup

# 从 torch.utils.cpp_extension 中导入 BuildExtension 和 CppExtension 类
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 使用 setup 函数配置 Python 包的信息和安装步骤
setup(
    # 指定包的名称为 "no_python_abi_suffix_test"
    name="no_python_abi_suffix_test",
    # 定义扩展模块的列表，包括一个 CppExtension 对象，指定了模块名称和源文件列表
    ext_modules=[
        CppExtension("no_python_abi_suffix_test", ["no_python_abi_suffix_test.cpp"])
    ],
    # 定义命令类的映射，使用 BuildExtension.with_options 方法配置 BuildExtension 类的选项
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
```