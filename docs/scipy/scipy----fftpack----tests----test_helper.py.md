# `D:\src\scipysrc\scipy\scipy\fftpack\tests\test_helper.py`

```
# Created by Pearu Peterson, September 2002
# 定义脚本的使用方法，包括如何构建 fftpack 和运行测试
__usage__ = """
Build fftpack:
  python setup_fftpack.py build
Run tests if scipy is installed:
  python -c 'import scipy;scipy.fftpack.test(<level>)'
Run tests if fftpack is not installed:
  python tests/test_helper.py [<level>]
"""

# 导入必要的模块和函数
from numpy.testing import assert_array_almost_equal
from scipy.fftpack import fftshift, ifftshift, fftfreq, rfftfreq
from numpy import pi, random

# 定义一个测试类 TestFFTShift，用于测试 fftshift 和 ifftshift 函数
class TestFFTShift:

    # 测试 fftshift 和 ifftshift 函数的定义
    def test_definition(self):
        # 定义输入数据和期望输出
        x = [0,1,2,3,4,-4,-3,-2,-1]
        y = [-4,-3,-2,-1,0,1,2,3,4]
        # 断言 fftshift 函数的输出与期望输出几乎相等
        assert_array_almost_equal(fftshift(x),y)
        # 断言 ifftshift 函数的输出与输入 x 几乎相等
        assert_array_almost_equal(ifftshift(y),x)
        
        # 定义另一组输入数据和期望输出
        x = [0,1,2,3,4,-5,-4,-3,-2,-1]
        y = [-5,-4,-3,-2,-1,0,1,2,3,4]
        # 断言 fftshift 函数的输出与期望输出几乎相等
        assert_array_almost_equal(fftshift(x),y)
        # 断言 ifftshift 函数的输出与输入 x 几乎相等
        assert_array_almost_equal(ifftshift(y),x)

    # 测试 fftshift 和 ifftshift 函数的逆操作
    def test_inverse(self):
        # 遍历不同的长度 n
        for n in [1,4,9,100,211]:
            # 生成随机数据 x
            x = random.random((n,))
            # 断言 ifftshift(fftshift(x)) 的输出与输入 x 几乎相等
            assert_array_almost_equal(ifftshift(fftshift(x)),x)


# 定义一个测试类 TestFFTFreq，用于测试 fftfreq 函数
class TestFFTFreq:

    # 测试 fftfreq 函数的定义
    def test_definition(self):
        # 定义输入数据和期望输出
        x = [0,1,2,3,4,-4,-3,-2,-1]
        # 断言 9 倍 fftfreq(9) 的输出与期望输出几乎相等
        assert_array_almost_equal(9*fftfreq(9),x)
        # 断言 9*pi 倍 fftfreq(9,pi) 的输出与期望输出几乎相等
        assert_array_almost_equal(9*pi*fftfreq(9,pi),x)
        
        # 定义另一组输入数据和期望输出
        x = [0,1,2,3,4,-5,-4,-3,-2,-1]
        # 断言 10 倍 fftfreq(10) 的输出与期望输出几乎相等
        assert_array_almost_equal(10*fftfreq(10),x)
        # 断言 10*pi 倍 fftfreq(10,pi) 的输出与期望输出几乎相等
        assert_array_almost_equal(10*pi*fftfreq(10,pi),x)


# 定义一个测试类 TestRFFTFreq，用于测试 rfftfreq 函数
class TestRFFTFreq:

    # 测试 rfftfreq 函数的定义
    def test_definition(self):
        # 定义输入数据和期望输出
        x = [0,1,1,2,2,3,3,4,4]
        # 断言 9 倍 rfftfreq(9) 的输出与期望输出几乎相等
        assert_array_almost_equal(9*rfftfreq(9),x)
        # 断言 9*pi 倍 rfftfreq(9,pi) 的输出与期望输出几乎相等
        assert_array_almost_equal(9*pi*rfftfreq(9,pi),x)
        
        # 定义另一组输入数据和期望输出
        x = [0,1,1,2,2,3,3,4,4,5]
        # 断言 10 倍 rfftfreq(10) 的输出与期望输出几乎相等
        assert_array_almost_equal(10*rfftfreq(10),x)
        # 断言 10*pi 倍 rfftfreq(10,pi) 的输出与期望输出几乎相等
        assert_array_almost_equal(10*pi*rfftfreq(10,pi),x)
```