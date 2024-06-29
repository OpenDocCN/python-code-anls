# `.\numpy\numpy\lib\tests\test_type_check.py`

```py
# 导入 NumPy 库，并从中导入一些函数和类
import numpy as np
from numpy import (
    common_type, mintypecode, isreal, iscomplex, isposinf, isneginf,
    nan_to_num, isrealobj, iscomplexobj, real_if_close
    )
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_raises
    )

# 定义一个断言函数，用于验证条件是否为真
def assert_all(x):
    assert_(np.all(x), x)

# 定义一个测试类 TestCommonType，用于测试 common_type 函数
class TestCommonType:
    # 定义测试方法 test_basic，测试常见数据类型的 common_type 结果
    def test_basic(self):
        # 创建不同数据类型的 NumPy 数组
        ai32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        af16 = np.array([[1, 2], [3, 4]], dtype=np.float16)
        af32 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        af64 = np.array([[1, 2], [3, 4]], dtype=np.float64)
        acs = np.array([[1+5j, 2+6j], [3+7j, 4+8j]], dtype=np.complex64)
        acd = np.array([[1+5j, 2+6j], [3+7j, 4+8j]], dtype=np.complex128)
        
        # 断言不同数据类型的 common_type 结果是否符合预期
        assert_(common_type(ai32) == np.float64)
        assert_(common_type(af16) == np.float16)
        assert_(common_type(af32) == np.float32)
        assert_(common_type(af64) == np.float64)
        assert_(common_type(acs) == np.complex64)
        assert_(common_type(acd) == np.complex128)

# 定义一个测试类 TestMintypecode，用于测试 mintypecode 函数
class TestMintypecode:
    
    # 定义测试方法 test_default_1，测试默认情况下的 mintypecode 结果
    def test_default_1(self):
        # 遍历不同输入类型，验证 mintypecode 返回值是否符合预期
        for itype in '1bcsuwil':
            assert_equal(mintypecode(itype), 'd')
        assert_equal(mintypecode('f'), 'f')
        assert_equal(mintypecode('d'), 'd')
        assert_equal(mintypecode('F'), 'F')
        assert_equal(mintypecode('D'), 'D')

    # 定义测试方法 test_default_2，测试另一种情况下的 mintypecode 结果
    def test_default_2(self):
        # 遍历不同输入类型组合，验证 mintypecode 返回值是否符合预期
        for itype in '1bcsuwil':
            assert_equal(mintypecode(itype+'f'), 'f')
            assert_equal(mintypecode(itype+'d'), 'd')
            assert_equal(mintypecode(itype+'F'), 'F')
            assert_equal(mintypecode(itype+'D'), 'D')
        assert_equal(mintypecode('ff'), 'f')
        assert_equal(mintypecode('fd'), 'd')
        assert_equal(mintypecode('fF'), 'F')
        assert_equal(mintypecode('fD'), 'D')
        assert_equal(mintypecode('df'), 'd')
        assert_equal(mintypecode('dd'), 'd')
        #assert_equal(mintypecode('dF',savespace=1),'F')
        assert_equal(mintypecode('dF'), 'D')
        assert_equal(mintypecode('dD'), 'D')
        assert_equal(mintypecode('Ff'), 'F')
        #assert_equal(mintypecode('Fd',savespace=1),'F')
        assert_equal(mintypecode('Fd'), 'D')
        assert_equal(mintypecode('FF'), 'F')
        assert_equal(mintypecode('FD'), 'D')
        assert_equal(mintypecode('Df'), 'D')
        assert_equal(mintypecode('Dd'), 'D')
        assert_equal(mintypecode('DF'), 'D')
        assert_equal(mintypecode('DD'), 'D')
    # 定义一个测试方法，用于测试 mintypecode 函数的默认行为
    def test_default_3(self):
        # 断言 mintypecode('fdF') 的返回结果是否等于 'D'
        assert_equal(mintypecode('fdF'), 'D')
        #assert_equal(mintypecode('fdF',savespace=1),'F')  # 这行代码被注释掉了，不参与测试
        # 断言 mintypecode('fdD') 的返回结果是否等于 'D'
        assert_equal(mintypecode('fdD'), 'D')
        # 断言 mintypecode('fFD') 的返回结果是否等于 'D'
        assert_equal(mintypecode('fFD'), 'D')
        # 断言 mintypecode('dFD') 的返回结果是否等于 'D'
        assert_equal(mintypecode('dFD'), 'D')

        # 断言 mintypecode('ifd') 的返回结果是否等于 'd'
        assert_equal(mintypecode('ifd'), 'd')
        # 断言 mintypecode('ifF') 的返回结果是否等于 'F'
        assert_equal(mintypecode('ifF'), 'F')
        # 断言 mintypecode('ifD') 的返回结果是否等于 'D'
        assert_equal(mintypecode('ifD'), 'D')
        # 断言 mintypecode('idF') 的返回结果是否等于 'D'
        assert_equal(mintypecode('idF'), 'D')
        #assert_equal(mintypecode('idF',savespace=1),'F')  # 这行代码被注释掉了，不参与测试
        # 断言 mintypecode('idD') 的返回结果是否等于 'D'
        assert_equal(mintypecode('idD'), 'D')
class TestIsscalar:

    def test_basic(self):
        # 检查是否为标量（单个数值）
        assert_(np.isscalar(3))
        # 检查是否不是标量（列表不是标量）
        assert_(not np.isscalar([3]))
        # 检查是否不是标量（元组不是标量）
        assert_(not np.isscalar((3,)))
        # 检查是否为标量（复数是标量）
        assert_(np.isscalar(3j))
        # 检查是否为标量（浮点数是标量）
        assert_(np.isscalar(4.0))


class TestReal:

    def test_real(self):
        # 生成一个包含随机数的数组
        y = np.random.rand(10,)
        # 断言数组和其实部相等
        assert_array_equal(y, np.real(y))

        # 创建一个包含单个元素的数组
        y = np.array(1)
        # 获取数组的实部
        out = np.real(y)
        # 断言输入和输出数组相等
        assert_array_equal(y, out)
        # 断言输出为 ndarray 类型
        assert_(isinstance(out, np.ndarray))

        # 创建一个标量
        y = 1
        # 获取实部
        out = np.real(y)
        # 断言输入和输出相等
        assert_equal(y, out)
        # 断言输出不是 ndarray 类型
        assert_(not isinstance(out, np.ndarray))

    def test_cmplx(self):
        # 生成一个包含随机复数的数组
        y = np.random.rand(10,)+1j*np.random.rand(10,)
        # 断言实部数组和输入数组的实部相等
        assert_array_equal(y.real, np.real(y))

        # 创建一个包含单个复数的数组
        y = np.array(1 + 1j)
        # 获取数组的实部
        out = np.real(y)
        # 断言输入数组的实部和输出数组相等
        assert_array_equal(y.real, out)
        # 断言输出为 ndarray 类型
        assert_(isinstance(out, np.ndarray))

        # 创建一个复数标量
        y = 1 + 1j
        # 获取实部
        out = np.real(y)
        # 断言实部为 1.0
        assert_equal(1.0, out)
        # 断言输出不是 ndarray 类型
        assert_(not isinstance(out, np.ndarray))


class TestImag:

    def test_real(self):
        # 生成一个包含随机数的数组
        y = np.random.rand(10,)
        # 断言虚部数组为 0
        assert_array_equal(0, np.imag(y))

        # 创建一个包含单个元素的数组
        y = np.array(1)
        # 获取数组的虚部
        out = np.imag(y)
        # 断言虚部为 0
        assert_array_equal(0, out)
        # 断言输出为 ndarray 类型
        assert_(isinstance(out, np.ndarray))

        # 创建一个标量
        y = 1
        # 获取虚部
        out = np.imag(y)
        # 断言虚部为 0
        assert_equal(0, out)
        # 断言输出不是 ndarray 类型
        assert_(not isinstance(out, np.ndarray))

    def test_cmplx(self):
        # 生成一个包含随机复数的数组
        y = np.random.rand(10,)+1j*np.random.rand(10,)
        # 断言虚部数组和输入数组的虚部相等
        assert_array_equal(y.imag, np.imag(y))

        # 创建一个包含单个复数的数组
        y = np.array(1 + 1j)
        # 获取数组的虚部
        out = np.imag(y)
        # 断言输入数组的虚部和输出数组相等
        assert_array_equal(y.imag, out)
        # 断言输出为 ndarray 类型
        assert_(isinstance(out, np.ndarray))

        # 创建一个复数标量
        y = 1 + 1j
        # 获取虚部
        out = np.imag(y)
        # 断言虚部为 1.0
        assert_equal(1.0, out)
        # 断言输出不是 ndarray 类型
        assert_(not isinstance(out, np.ndarray))


class TestIscomplex:

    def test_fail(self):
        # 创建一个数组
        z = np.array([-1, 0, 1])
        # 检查数组中是否没有复数
        res = iscomplex(z)
        assert_(not np.any(res, axis=0))

    def test_pass(self):
        # 创建一个数组
        z = np.array([-1j, 1, 0])
        # 检查数组中每个元素是否为复数
        res = iscomplex(z)
        assert_array_equal(res, [1, 0, 0])


class TestIsreal:

    def test_pass(self):
        # 创建一个数组
        z = np.array([-1, 0, 1j])
        # 检查数组中每个元素是否为实数
        res = isreal(z)
        assert_array_equal(res, [1, 1, 0])

    def test_fail(self):
        # 创建一个数组
        z = np.array([-1j, 1, 0])
        # 检查数组中每个元素是否为实数
        res = isreal(z)
        assert_array_equal(res, [0, 1, 1])


class TestIscomplexobj:

    def test_basic(self):
        # 创建一个数组
        z = np.array([-1, 0, 1])
        # 检查数组是否包含复数对象
        assert_(not iscomplexobj(z))
        # 创建一个包含复数的数组
        z = np.array([-1j, 0, -1])
        # 检查数组是否包含复数对象
        assert_(iscomplexobj(z))

    def test_scalar(self):
        # 检查标量是否为复数对象
        assert_(not iscomplexobj(1.0))
        assert_(iscomplexobj(1+0j))

    def test_list(self):
        # 检查列表中是否包含复数对象
        assert_(iscomplexobj([3, 1+0j, True]))
        assert_(not iscomplexobj([3, 1, True]))

    def test_duck(self):
        # 创建一个虚拟的复数数组类
        class DummyComplexArray:
            @property
            def dtype(self):
                return np.dtype(complex)
        dummy = DummyComplexArray()
        # 检查虚拟复数数组是否为复数对象
        assert_(iscomplexobj(dummy))
    # 定义一个测试方法，用于验证自定义的 np.dtype 鸭子类型类，比如 pandas 使用的类（pandas.core.dtypes）
    def test_pandas_duck(self):
        # 定义一个继承自 np.complex128 的 pandas 复杂类型类
        class PdComplex(np.complex128):
            pass
        
        # 定义一个模拟的 pandas 数据类型类
        class PdDtype:
            name = 'category'  # 数据类型名称为 'category'
            names = None       # 名称列表为空
            type = PdComplex   # 数据类型为 PdComplex 类型
            kind = 'c'         # 类别标识为 'c'
            str = '<c16'       # 字符串描述为 '<c16'
            base = np.dtype('complex128')  # 基础数据类型为 np.complex128
        
        # 定义一个虚拟的 DummyPd 类，具有 dtype 属性，返回 PdDtype 类
        class DummyPd:
            @property
            def dtype(self):
                return PdDtype
        
        # 创建 DummyPd 类的实例 dummy
        dummy = DummyPd()
        
        # 断言 dummy 对象是否是复数对象
        assert_(iscomplexobj(dummy))

    # 定义另一个测试方法，用于验证自定义数据类型鸭子类型
    def test_custom_dtype_duck(self):
        # 定义一个继承自 list 的自定义数组类 MyArray
        class MyArray(list):
            # 定义 dtype 属性，返回复数类型
            @property
            def dtype(self):
                return complex
        
        # 创建 MyArray 类的实例 a，包含三个复数
        a = MyArray([1+0j, 2+0j, 3+0j])
        
        # 断言 a 对象是否是复数对象
        assert_(iscomplexobj(a))
class TestIsrealobj:
    def test_basic(self):
        # 创建一个包含三个元素的 numpy 数组，用于测试是否为实数对象
        z = np.array([-1, 0, 1])
        # 断言 z 是实数对象
        assert_(isrealobj(z))
        
        # 创建另一个 numpy 数组，包含复数元素，用于测试非实数对象
        z = np.array([-1j, 0, -1])
        # 断言 z 不是实数对象
        assert_(not isrealobj(z))


class TestIsnan:

    def test_goodvalues(self):
        # 创建一个包含三个浮点数的 numpy 数组，测试它们不是 NaN
        z = np.array((-1., 0., 1.))
        # 生成一个布尔数组，检查 z 中的元素是否不是 NaN
        res = np.isnan(z) == 0
        # 断言 res 中所有元素都为 True
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        # 使用 np.errstate 忽略除以零带来的警告
        with np.errstate(divide='ignore'):
            # 创建包含正无穷的 numpy 数组，测试它们不是 NaN
            assert_all(np.isnan(np.array((1.,))/0.) == 0)

    def test_neginf(self):
        # 使用 np.errstate 忽略除以零带来的警告
        with np.errstate(divide='ignore'):
            # 创建包含负无穷的 numpy 数组，测试它们不是 NaN
            assert_all(np.isnan(np.array((-1.,))/0.) == 0)

    def test_ind(self):
        # 使用 np.errstate 忽略除以零和无效操作带来的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建包含非法操作结果的 numpy 数组，测试它们是 NaN
            assert_all(np.isnan(np.array((0.,))/0.) == 1)

    def test_integer(self):
        # 创建整数值，测试它不是 NaN
        assert_all(np.isnan(1) == 0)

    def test_complex(self):
        # 创建复数值，测试它不是 NaN
        assert_all(np.isnan(1+1j) == 0)

    def test_complex1(self):
        # 使用 np.errstate 忽略除以零和无效操作带来的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建复数值进行除法运算，测试它是 NaN
            assert_all(np.isnan(np.array(0+0j)/0.) == 1)


class TestIsfinite:
    # Fixme, wrong place, isfinite now ufunc

    def test_goodvalues(self):
        # 创建一个包含三个浮点数的 numpy 数组，测试它们是有限数
        z = np.array((-1., 0., 1.))
        # 生成一个布尔数组，检查 z 中的元素是否是有限数
        res = np.isfinite(z) == 1
        # 断言 res 中所有元素都为 True
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        # 使用 np.errstate 忽略除以零和无效操作带来的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建包含正无穷的 numpy 数组，测试它们不是有限数
            assert_all(np.isfinite(np.array((1.,))/0.) == 0)

    def test_neginf(self):
        # 使用 np.errstate 忽略除以零和无效操作带来的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建包含负无穷的 numpy 数组，测试它们不是有限数
            assert_all(np.isfinite(np.array((-1.,))/0.) == 0)

    def test_ind(self):
        # 使用 np.errstate 忽略除以零和无效操作带来的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建包含非法操作结果的 numpy 数组，测试它们不是有限数
            assert_all(np.isfinite(np.array((0.,))/0.) == 0)

    def test_integer(self):
        # 创建整数值，测试它是有限数
        assert_all(np.isfinite(1) == 1)

    def test_complex(self):
        # 创建复数值，测试它是有限数
        assert_all(np.isfinite(1+1j) == 1)

    def test_complex1(self):
        # 使用 np.errstate 忽略除以零和无效操作带来的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建复数值进行除法运算，测试它不是有限数
            assert_all(np.isfinite(np.array(1+1j)/0.) == 0)


class TestIsinf:
    # Fixme, wrong place, isinf now ufunc

    def test_goodvalues(self):
        # 创建一个包含三个浮点数的 numpy 数组，测试它们不是无穷数
        z = np.array((-1., 0., 1.))
        # 生成一个布尔数组，检查 z 中的元素是否不是无穷数
        res = np.isinf(z) == 0
        # 断言 res 中所有元素都为 True
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        # 使用 np.errstate 忽略除以零和无效操作带来的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建包含正无穷的 numpy 数组，测试它们是正无穷数
            assert_all(np.isinf(np.array((1.,))/0.) == 1)

    def test_posinf_scalar(self):
        # 使用 np.errstate 忽略除以零和无效操作带来的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建单个正无穷数，测试它是正无穷数
            assert_all(np.isinf(np.array(1.,)/0.) == 1)

    def test_neginf(self):
        # 使用 np.errstate 忽略除以零和无效操作带来的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建包含负无穷的 numpy 数组，测试它们是负无穷数
            assert_all(np.isinf(np.array((-1.,))/0.) == 1)

    def test_neginf_scalar(self):
        # 使用 np.errstate 忽略除以零和无效操作带来的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建单个负无穷数，测试它是负无穷数
            assert_all(np.isinf(np.array(-1.)/0.) == 1)

    def test_ind(self):
        # 使用 np.errstate 忽略除以零和无效操作带来的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建包含非法操作结果的 numpy 数组，测试它们不是无穷数
            assert_all(np.isinf(np.array((0.,))/0.) == 0)


class TestIsposinf:
    # Fixme, wrong place, isposinf not a ufunc yet
    # 定义一个测试函数，用于测试通用情况
    def test_generic(self):
        # 在计算过程中忽略除以零的警告和无效操作的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建一个包含负无穷、零和正无穷的数组，并计算其是否为正无穷
            vals = isposinf(np.array((-1., 0, 1))/0.)
        # 断言：检查计算结果中第一个元素是否为零
        assert_(vals[0] == 0)
        # 断言：检查计算结果中第二个元素是否为零
        assert_(vals[1] == 0)
        # 断言：检查计算结果中第三个元素是否为正无穷
        assert_(vals[2] == 1)
class TestIsneginf:

    def test_generic(self):
        # 忽略除法和无效值错误，执行下面的代码块
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建一个包含负无穷值的数组
            vals = isneginf(np.array((-1., 0, 1))/0.)
        # 断言第一个值为1
        assert_(vals[0] == 1)
        # 断言第二个值为0
        assert_(vals[1] == 0)
        # 断言第三个值为0
        assert_(vals[2] == 0)


class TestNanToNum:

    def test_generic(self):
        # 忽略除法和无效值错误，执行下面的代码块
        with np.errstate(divide='ignore', invalid='ignore'):
            # 使用 nan_to_num 将包含无限值的数组处理为特定的值
            vals = nan_to_num(np.array((-1., 0, 1))/0.)
        # 断言第一个值小于 -1e10，并且是有限的
        assert_all(vals[0] < -1e10) and assert_all(np.isfinite(vals[0]))
        # 断言第二个值为0
        assert_(vals[1] == 0)
        # 断言第三个值大于 1e10，并且是有限的
        assert_all(vals[2] > 1e10) and assert_all(np.isfinite(vals[2]))
        # 断言结果的类型为 numpy 数组
        assert_equal(type(vals), np.ndarray)
        
        # 使用 nan=10, posinf=20, neginf=30 参数再次进行相同的测试
        with np.errstate(divide='ignore', invalid='ignore'):
            # 使用 nan_to_num 处理数组，将 nan 替换为 10，将正无穷替换为 20，将负无穷替换为 30
            vals = nan_to_num(np.array((-1., 0, 1))/0., 
                              nan=10, posinf=20, neginf=30)
        # 断言结果数组与期望的数组相等
        assert_equal(vals, [30, 10, 20])
        # 断言结果数组的第一个和第三个元素是有限的
        assert_all(np.isfinite(vals[[0, 2]]))
        # 断言结果的类型为 numpy 数组
        assert_equal(type(vals), np.ndarray)

        # 在原地进行相同的测试
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建一个包含无限值的数组
            vals = np.array((-1., 0, 1))/0.
        # 使用 nan_to_num 在原地处理数组
        result = nan_to_num(vals, copy=False)

        # 断言处理结果与原数组是同一个对象
        assert_(result is vals)
        # 断言原数组的第一个值小于 -1e10，并且是有限的
        assert_all(vals[0] < -1e10) and assert_all(np.isfinite(vals[0]))
        # 断言原数组的第二个值为0
        assert_(vals[1] == 0)
        # 断言原数组的第三个值大于 1e10，并且是有限的
        assert_all(vals[2] > 1e10) and assert_all(np.isfinite(vals[2]))
        # 断言结果的类型为 numpy 数组
        assert_equal(type(vals), np.ndarray)
        
        # 在原地进行相同的测试，但使用 nan=10, posinf=20, neginf=30 参数
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建一个包含无限值的数组
            vals = np.array((-1., 0, 1))/0.
        # 使用 nan_to_num 在原地处理数组，并将 nan 替换为 10，将正无穷替换为 20，将负无穷替换为 30
        result = nan_to_num(vals, copy=False, nan=10, posinf=20, neginf=30)

        # 断言处理结果与原数组是同一个对象
        assert_(result is vals)
        # 断言原数组等于期望的数组 [30, 10, 20]
        assert_equal(vals, [30, 10, 20])
        # 断言原数组的第一个和第三个元素是有限的
        assert_all(np.isfinite(vals[[0, 2]]))
        # 断言结果的类型为 numpy 数组
        assert_equal(type(vals), np.ndarray)

    def test_array(self):
        # 使用 nan_to_num 处理数组 [1]
        vals = nan_to_num([1])
        # 断言处理结果与期望的数组相等
        assert_array_equal(vals, np.array([1], int))
        # 断言结果的类型为 numpy 数组
        assert_equal(type(vals), np.ndarray)
        # 使用 nan=10, posinf=20, neginf=30 参数再次处理数组 [1]
        vals = nan_to_num([1], nan=10, posinf=20, neginf=30)
        # 断言处理结果与期望的数组相等
        assert_array_equal(vals, np.array([1], int))
        # 断言结果的类型为 numpy 数组
        assert_equal(type(vals), np.ndarray)

    def test_integer(self):
        # 使用 nan_to_num 处理整数 1
        vals = nan_to_num(1)
        # 断言处理结果等于 1
        assert_all(vals == 1)
        # 断言结果的类型为 np.int_
        assert_equal(type(vals), np.int_)
        # 使用 nan=10, posinf=20, neginf=30 参数再次处理整数 1
        vals = nan_to_num(1, nan=10, posinf=20, neginf=30)
        # 断言处理结果等于 1
        assert_all(vals == 1)
        # 断言结果的类型为 np.int_

    def test_float(self):
        # 使用 nan_to_num 处理浮点数 1.0
        vals = nan_to_num(1.0)
        # 断言处理结果等于 1.0
        assert_all(vals == 1.0)
        # 断言结果的类型为 np.float64
        assert_equal(type(vals), np.float64)
        # 使用 nan=10, posinf=20, neginf=30 参数再次处理浮点数 1.1
        vals = nan_to_num(1.1, nan=10, posinf=20, neginf=30)
        # 断言处理结果等于 1.1
        assert_all(vals == 1.1)
        # 断言结果的类型为 np.float64
        assert_equal(type(vals), np.float64)
    # 定义一个测试函数，用于测试处理复数的情况（正常情况）
    def test_complex_good(self):
        # 将复数中的 NaN 替换为 0，并返回处理后的值
        vals = nan_to_num(1+1j)
        # 断言所有处理后的值等于原始复数值
        assert_all(vals == 1+1j)
        # 断言处理后的值的数据类型为 np.complex128
        assert_equal(type(vals), np.complex128)
        
        # 将复数中的 NaN 替换为 10，正无穷替换为 20，负无穷替换为 30，并返回处理后的值
        vals = nan_to_num(1+1j, nan=10, posinf=20, neginf=30)
        # 断言所有处理后的值等于原始复数值
        assert_all(vals == 1+1j)
        # 断言处理后的值的数据类型为 np.complex128
        assert_equal(type(vals), np.complex128)

    # 定义一个测试函数，用于测试处理复数的情况（异常情况1）
    def test_complex_bad(self):
        # 在忽略除法错误和无效值的错误状态下执行以下操作
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建一个复数 v = 1 + 1j
            v = 1 + 1j
            # 将 v 增加一个除以零得到的复数数组的结果
            v += np.array(0+1.j)/0.
        # 将复数 v 中的 NaN 替换为 0，并返回处理后的值
        vals = nan_to_num(v)
        # 断言所有处理后的值是有限的（即不是 NaN 或 inf）
        # !! 这实际上是 (意外地) 得到了零
        assert_all(np.isfinite(vals))
        # 断言处理后的值的数据类型为 np.complex128
        assert_equal(type(vals), np.complex128)

    # 定义一个测试函数，用于测试处理复数的情况（异常情况2）
    def test_complex_bad2(self):
        # 在忽略除法错误和无效值的错误状态下执行以下操作
        with np.errstate(divide='ignore', invalid='ignore'):
            # 创建一个复数 v = 1 + 1j
            v = 1 + 1j
            # 将 v 增加一个除以零得到的复数数组的结果
            v += np.array(-1+1.j)/0.
        # 将复数 v 中的 NaN 替换为 0，并返回处理后的值
        vals = nan_to_num(v)
        # 断言所有处理后的值是有限的（即不是 NaN 或 inf）
        assert_all(np.isfinite(vals))
        # 断言处理后的值的数据类型为 np.complex128
        assert_equal(type(vals), np.complex128)
        # Fixme
        #assert_all(vals.imag > 1e10)  and assert_all(np.isfinite(vals))
        # !! 这实际上是 (意外地) 是正数
        # !! inf。暂时注释掉，并观察是否有变化
        #assert_all(vals.real < -1e10) and assert_all(np.isfinite(vals))

    # 定义一个测试函数，用于测试 nan_to_num 函数不会重写先前关键字的情况
    def test_do_not_rewrite_previous_keyword(self):
        # 在忽略除法错误和无效值的错误状态下执行以下操作
        with np.errstate(divide='ignore', invalid='ignore'):
            # 对数组中的 (-1., 0, 1)/0. 进行 NaN 替换为 np.inf，posinf 替换为 999
            vals = nan_to_num(np.array((-1., 0, 1))/0., nan=np.inf, posinf=999)
        # 断言数组中指定位置的值是有限的（不是 NaN 或 inf）
        assert_all(np.isfinite(vals[[0, 2]]))
        # 断言数组中第一个位置的值小于 -1e10
        assert_all(vals[0] < -1e10)
        # 断言数组中第二个和第三个位置的值分别等于 np.inf 和 999
        assert_equal(vals[[1, 2]], [np.inf, 999])
        # 断言处理后的值的数据类型为 np.ndarray
        assert_equal(type(vals), np.ndarray)
# 定义一个名为 TestRealIfClose 的测试类
class TestRealIfClose:

    # 定义一个测试方法 test_basic，用于测试 real_if_close 函数的基本功能
    def test_basic(self):
        # 生成一个包含 10 个随机数的数组 a
        a = np.random.rand(10)
        # 调用 real_if_close 函数，处理 a 中每个元素加上虚数部分 1e-15j
        b = real_if_close(a+1e-15j)
        # 断言处理后的数组 b 中所有元素都是实数
        assert_all(isrealobj(b))
        # 断言处理后的数组 b 与原始数组 a 相等
        assert_array_equal(a, b)
        # 再次调用 real_if_close 函数，处理 a 中每个元素加上虚数部分 1e-7j
        b = real_if_close(a+1e-7j)
        # 断言处理后的数组 b 中所有元素都是复数
        assert_all(iscomplexobj(b))
        # 第三次调用 real_if_close 函数，设置容差参数 tol 为 1e-6，处理 a 中每个元素加上虚数部分 1e-7j
        b = real_if_close(a+1e-7j, tol=1e-6)
        # 断言处理后的数组 b 中所有元素都是实数
        assert_all(isrealobj(b))
```