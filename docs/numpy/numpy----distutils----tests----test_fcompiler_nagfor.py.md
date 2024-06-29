# `.\numpy\numpy\distutils\tests\test_fcompiler_nagfor.py`

```
# 导入所需模块和函数
from numpy.testing import assert_
import numpy.distutils.fcompiler

# 定义一个包含多个元组的列表，每个元组包含编译器名称、版本字符串和期望的版本号
nag_version_strings = [('nagfor', 'NAG Fortran Compiler Release '
                        '6.2(Chiyoda) Build 6200', '6.2'),
                       ('nagfor', 'NAG Fortran Compiler Release '
                        '6.1(Tozai) Build 6136', '6.1'),
                       ('nagfor', 'NAG Fortran Compiler Release '
                        '6.0(Hibiya) Build 1021', '6.0'),
                       ('nagfor', 'NAG Fortran Compiler Release '
                        '5.3.2(971)', '5.3.2'),
                       ('nag', 'NAGWare Fortran 95 compiler Release 5.1'
                        '(347,355-367,375,380-383,389,394,399,401-402,407,'
                        '431,435,437,446,459-460,463,472,494,496,503,508,'
                        '511,517,529,555,557,565)', '5.1')]

# 定义一个测试类，用于测试 NAG 编译器版本匹配的功能
class TestNagFCompilerVersions:
    
    # 定义测试方法，验证每个版本字符串是否与预期的版本号匹配
    def test_version_match(self):
        # 遍历版本信息的列表
        for comp, vs, version in nag_version_strings:
            # 使用 numpy.distutils.fcompiler.new_fcompiler() 方法创建指定编译器的编译器对象
            fc = numpy.distutils.fcompiler.new_fcompiler(compiler=comp)
            # 调用编译器对象的 version_match() 方法，传入版本字符串 vs，返回实际版本号 v
            v = fc.version_match(vs)
            # 使用 numpy.testing.assert_ 函数断言实际版本号 v 是否等于预期版本号 version
            assert_(v == version)
```