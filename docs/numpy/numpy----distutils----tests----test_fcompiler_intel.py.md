# `.\numpy\numpy\distutils\tests\test_fcompiler_intel.py`

```py
# 导入所需的模块和函数
import numpy.distutils.fcompiler  # 导入 numpy.distutils.fcompiler 模块
from numpy.testing import assert_  # 从 numpy.testing 模块导入 assert_ 函数

# 定义 Intel 32 位编译器版本字符串列表
intel_32bit_version_strings = [
    ("Intel(R) Fortran Intel(R) 32-bit Compiler Professional for applications"
     "running on Intel(R) 32, Version 11.1", '11.1'),
]

# 定义 Intel 64 位编译器版本字符串列表
intel_64bit_version_strings = [
    ("Intel(R) Fortran IA-64 Compiler Professional for applications"
     "running on IA-64, Version 11.0", '11.0'),
    ("Intel(R) Fortran Intel(R) 64 Compiler Professional for applications"
     "running on Intel(R) 64, Version 11.1", '11.1')
]

# 定义测试类 TestIntelFCompilerVersions，测试 Intel 32 位编译器版本匹配
class TestIntelFCompilerVersions:
    def test_32bit_version(self):
        # 创建一个 Intel 32 位编译器的新实例
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='intel')
        # 遍历 Intel 32 位编译器版本字符串列表
        for vs, version in intel_32bit_version_strings:
            # 使用 fc.version_match 方法匹配版本
            v = fc.version_match(vs)
            # 断言版本匹配结果是否与预期版本一致
            assert_(v == version)

# 定义测试类 TestIntelEM64TFCompilerVersions，测试 Intel 64 位编译器版本匹配
class TestIntelEM64TFCompilerVersions:
    def test_64bit_version(self):
        # 创建一个 Intel 64 位编译器的新实例
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='intelem')
        # 遍历 Intel 64 位编译器版本字符串列表
        for vs, version in intel_64bit_version_strings:
            # 使用 fc.version_match 方法匹配版本
            v = fc.version_match(vs)
            # 断言版本匹配结果是否与预期版本一致
            assert_(v == version)
```