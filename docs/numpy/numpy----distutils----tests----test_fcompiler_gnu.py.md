# `.\numpy\numpy\distutils\tests\test_fcompiler_gnu.py`

```
# 导入必要的断言函数
from numpy.testing import assert_

# 导入用于处理 Fortran 编译器的模块
import numpy.distutils.fcompiler

# 包含一组 G77 Fortran 版本字符串及其预期版本号的元组列表
g77_version_strings = [
    ('GNU Fortran 0.5.25 20010319 (prerelease)', '0.5.25'),
    ('GNU Fortran (GCC 3.2) 3.2 20020814 (release)', '3.2'),
    ('GNU Fortran (GCC) 3.3.3 20040110 (prerelease) (Debian)', '3.3.3'),
    ('GNU Fortran (GCC) 3.3.3 (Debian 20040401)', '3.3.3'),
    ('GNU Fortran (GCC 3.2.2 20030222 (Red Hat Linux 3.2.2-5)) 3.2.2'
       ' 20030222 (Red Hat Linux 3.2.2-5)', '3.2.2'),
]

# 包含一组 GFortran 版本字符串及其预期版本号的元组列表
gfortran_version_strings = [
    ('GNU Fortran 95 (GCC 4.0.3 20051023 (prerelease) (Debian 4.0.2-3))',
     '4.0.3'),
    ('GNU Fortran 95 (GCC) 4.1.0', '4.1.0'),
    ('GNU Fortran 95 (GCC) 4.2.0 20060218 (experimental)', '4.2.0'),
    ('GNU Fortran (GCC) 4.3.0 20070316 (experimental)', '4.3.0'),
    ('GNU Fortran (rubenvb-4.8.0) 4.8.0', '4.8.0'),
    ('4.8.0', '4.8.0'),
    ('4.0.3-7', '4.0.3'),
    ("gfortran: warning: couldn't understand kern.osversion '14.1.0\n4.9.1",
     '4.9.1'),
    ("gfortran: warning: couldn't understand kern.osversion '14.1.0\n"
     "gfortran: warning: yet another warning\n4.9.1",
     '4.9.1'),
    ('GNU Fortran (crosstool-NG 8a21ab48) 7.2.0', '7.2.0')
]

# 测试类：测试 G77 Fortran 版本匹配
class TestG77Versions:
    def test_g77_version(self):
        # 创建一个新的 G77 Fortran 编译器对象
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu')
        # 遍历预定义的 G77 版本字符串和预期版本号的元组列表
        for vs, version in g77_version_strings:
            # 使用编译器对象匹配版本字符串并获取结果
            v = fc.version_match(vs)
            # 断言版本号匹配预期版本号
            assert_(v == version, (vs, v))

    # 测试不是 G77 Fortran 的情况
    def test_not_g77(self):
        # 创建一个新的 G77 Fortran 编译器对象
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu')
        # 遍历预定义的 GFortran 版本字符串和预期版本号的元组列表
        for vs, _ in gfortran_version_strings:
            # 使用编译器对象匹配版本字符串并获取结果
            v = fc.version_match(vs)
            # 断言结果为空（不匹配）
            assert_(v is None, (vs, v))

# 测试类：测试 GFortran 版本匹配
class TestGFortranVersions:
    def test_gfortran_version(self):
        # 创建一个新的 GFortran 编译器对象
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu95')
        # 遍历预定义的 GFortran 版本字符串和预期版本号的元组列表
        for vs, version in gfortran_version_strings:
            # 使用编译器对象匹配版本字符串并获取结果
            v = fc.version_match(vs)
            # 断言版本号匹配预期版本号
            assert_(v == version, (vs, v))

    # 测试不是 GFortran 的情况
    def test_not_gfortran(self):
        # 创建一个新的 GFortran 编译器对象
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu95')
        # 遍历预定义的 G77 版本字符串和预期版本号的元组列表
        for vs, _ in g77_version_strings:
            # 使用编译器对象匹配版本字符串并获取结果
            v = fc.version_match(vs)
            # 断言结果为空（不匹配）
            assert_(v is None, (vs, v))
```