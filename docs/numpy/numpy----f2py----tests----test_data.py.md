# `.\numpy\numpy\f2py\tests\test_data.py`

```
# 导入必要的模块：os、pytest、numpy
import os
import pytest
import numpy as np

# 从当前目录下的util模块中导入F2PyTest类
from . import util

# 从numpy.f2py.crackfortran中导入crackfortran函数
from numpy.f2py.crackfortran import crackfortran

# TestData类，继承自util.F2PyTest类
class TestData(util.F2PyTest):
    
    # 源文件路径列表，包含要测试的Fortran源文件路径
    sources = [util.getpath("tests", "src", "crackfortran", "data_stmts.f90")]

    # 标记为slow的测试方法，用于测试data_stmts.f90中的变量
    @pytest.mark.slow
    def test_data_stmts(self):
        # 断言各变量的值是否符合预期
        assert self.module.cmplxdat.i == 2
        assert self.module.cmplxdat.j == 3
        assert self.module.cmplxdat.x == 1.5
        assert self.module.cmplxdat.y == 2.0
        assert self.module.cmplxdat.pi == 3.1415926535897932384626433832795028841971693993751058209749445923078164062
        assert self.module.cmplxdat.medium_ref_index == np.array(1.+0.j)
        assert np.all(self.module.cmplxdat.z == np.array([3.5, 7.0]))
        assert np.all(self.module.cmplxdat.my_array == np.array([1.+2.j, -3.+4.j]))
        assert np.all(self.module.cmplxdat.my_real_array == np.array([1., 2., 3.]))
        assert np.all(self.module.cmplxdat.ref_index_one == np.array([13.0 + 21.0j]))
        assert np.all(self.module.cmplxdat.ref_index_two == np.array([-30.0 + 43.0j]))

    # 测试crackfortran函数
    def test_crackedlines(self):
        # 调用crackfortran函数处理源文件列表，返回处理结果mod
        mod = crackfortran(self.sources)
        # 断言处理结果中的变量值是否符合预期
        assert mod[0]['vars']['x']['='] == '1.5'
        assert mod[0]['vars']['y']['='] == '2.0'
        assert mod[0]['vars']['pi']['='] == '3.1415926535897932384626433832795028841971693993751058209749445923078164062d0'
        assert mod[0]['vars']['my_real_array']['='] == '(/1.0d0, 2.0d0, 3.0d0/)'
        assert mod[0]['vars']['ref_index_one']['='] == '(13.0d0, 21.0d0)'
        assert mod[0]['vars']['ref_index_two']['='] == '(-30.0d0, 43.0d0)'
        assert mod[0]['vars']['my_array']['='] == '(/(1.0d0, 2.0d0), (-3.0d0, 4.0d0)/)'
        assert mod[0]['vars']['z']['='] == '(/3.5,  7.0/)'

# TestDataF77类，继承自util.F2PyTest类
class TestDataF77(util.F2PyTest):
    
    # 源文件路径列表，包含要测试的Fortran源文件路径
    sources = [util.getpath("tests", "src", "crackfortran", "data_common.f")]

    # 测试data_common.f中的变量
    # For gh-23276
    def test_data_stmts(self):
        # 断言self.module.mycom.mydata的值是否为0
        assert self.module.mycom.mydata == 0

    # 测试crackfortran函数
    def test_crackedlines(self):
        # 使用str转换源文件路径为字符串，调用crackfortran函数处理源文件，返回处理结果mod
        mod = crackfortran(str(self.sources[0]))
        # 打印处理结果mod中的变量信息
        print(mod[0]['vars'])
        # 断言处理结果中mydata变量的值是否为0
        assert mod[0]['vars']['mydata']['='] == '0'

# TestDataMultiplierF77类，继承自util.F2PyTest类
class TestDataMultiplierF77(util.F2PyTest):
    
    # 源文件路径列表，包含要测试的Fortran源文件路径
    sources = [util.getpath("tests", "src", "crackfortran", "data_multiplier.f")]

    # 测试data_multiplier.f中的变量
    # For gh-23276
    def test_data_stmts(self):
        # 断言self.module.mycom中的各变量值是否符合预期
        assert self.module.mycom.ivar1 == 3
        assert self.module.mycom.ivar2 == 3
        assert self.module.mycom.ivar3 == 2
        assert self.module.mycom.ivar4 == 2
        assert self.module.mycom.evar5 == 0

# TestDataWithCommentsF77类，继承自util.F2PyTest类
class TestDataWithCommentsF77(util.F2PyTest):
    
    # 源文件路径列表，包含要测试的Fortran源文件路径
    sources = [util.getpath("tests", "src", "crackfortran", "data_with_comments.f")]

    # 测试data_with_comments.f中的变量
    # For gh-23276
    def test_data_stmts(self):
        # 断言self.module.mycom.mytab的长度是否为3，以及其各元素的值是否符合预期
        assert len(self.module.mycom.mytab) == 3
        assert self.module.mycom.mytab[0] == 0
        assert self.module.mycom.mytab[1] == 4
        assert self.module.mycom.mytab[2] == 0
```