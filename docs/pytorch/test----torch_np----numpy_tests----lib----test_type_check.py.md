# `.\pytorch\test\torch_np\numpy_tests\lib\test_type_check.py`

```py
# Owner(s): ["module: dynamo"]

# 导入 functools 模块，用于创建偏函数
import functools

# 从 unittest 模块中导入 expectedFailure 和 skipIf，并使用 xfail 和 skipif 作为别名
from unittest import expectedFailure as xfail, skipIf as skipif

# 从 pytest 模块中导入 raises，并使用 assert_raises 作为别名
from pytest import raises as assert_raises

# 从 torch.testing._internal.common_utils 导入 run_tests、TEST_WITH_TORCHDYNAMO、TestCase、xpassIfTorchDynamo
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)

# 如果 TEST_WITH_TORCHDYNAMO 为真，则从 numpy 中导入以下函数和类
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import (
        common_type,
        iscomplex,
        iscomplexobj,
        isneginf,
        isposinf,
        isreal,
        isrealobj,
        nan_to_num,
        real_if_close,
    )
    from numpy.testing import assert_, assert_array_equal, assert_equal
# 否则，从 torch._numpy 中导入以下函数和类
else:
    import torch._numpy as np
    from torch._numpy import (
        common_type,
        iscomplex,
        iscomplexobj,
        isneginf,
        isposinf,
        isreal,
        isrealobj,
        nan_to_num,
        real_if_close,
    )
    from torch._numpy.testing import assert_, assert_array_equal, assert_equal

# 使用 functools.partial 创建 skip 函数，其行为类似于 skipif(True)
skip = functools.partial(skipif, True)


# 定义 assert_all 函数，用于断言参数 x 全为真
def assert_all(x):
    assert_(np.all(x), x)


@xpassIfTorchDynamo  # 标记测试类 TestCommonType 为 torch dynamo 的测试
class TestCommonType(TestCase):
    def test_basic(self):
        # 创建不同类型的 numpy 数组
        ai32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        af16 = np.array([[1, 2], [3, 4]], dtype=np.float16)
        af32 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        af64 = np.array([[1, 2], [3, 4]], dtype=np.float64)
        acs = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.csingle)
        acd = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.cdouble)
        
        # 断言不同类型数组的 common_type 结果符合预期
        assert_(common_type(ai32) == np.float64)
        assert_(common_type(af16) == np.float16)
        assert_(common_type(af32) == np.float32)
        assert_(common_type(af64) == np.float64)
        assert_(common_type(acs) == np.csingle)
        assert_(common_type(acd) == np.cdouble)


@xfail  # 标记测试类 TestMintypecode 为预期失败
class TestMintypecode(TestCase):
    def test_default_1(self):
        # 遍历字符串 "1bcsuwil" 中的每个字符，并断言 mintypecode 的结果符合预期
        for itype in "1bcsuwil":
            assert_equal(mintypecode(itype), "d")
        assert_equal(mintypecode("f"), "f")
        assert_equal(mintypecode("d"), "d")
        assert_equal(mintypecode("F"), "F")
        assert_equal(mintypecode("D"), "D")
    # 定义测试函数 test_default_2，用于测试 mintypecode 函数的多种输入情况
    def test_default_2(self):
        # 对字符串 "1bcsuwil" 中的每个字符进行迭代
        for itype in "1bcsuwil":
            # 断言调用 mintypecode 函数处理 itype + "f" 后返回值为 "f"
            assert_equal(mintypecode(itype + "f"), "f")
            # 断言调用 mintypecode 函数处理 itype + "d" 后返回值为 "d"
            assert_equal(mintypecode(itype + "d"), "d")
            # 断言调用 mintypecode 函数处理 itype + "F" 后返回值为 "F"
            assert_equal(mintypecode(itype + "F"), "F")
            # 断言调用 mintypecode 函数处理 itype + "D" 后返回值为 "D"
            assert_equal(mintypecode(itype + "D"), "D")
        
        # 对特定组合进行单独的断言测试
        assert_equal(mintypecode("ff"), "f")
        assert_equal(mintypecode("fd"), "d")
        assert_equal(mintypecode("fF"), "F")
        assert_equal(mintypecode("fD"), "D")
        assert_equal(mintypecode("df"), "d")
        assert_equal(mintypecode("dd"), "d")
        # assert_equal(mintypecode('dF',savespace=1),'F')
        assert_equal(mintypecode("dF"), "D")
        assert_equal(mintypecode("dD"), "D")
        assert_equal(mintypecode("Ff"), "F")
        # assert_equal(mintypecode('Fd',savespace=1),'F')
        assert_equal(mintypecode("Fd"), "D")
        assert_equal(mintypecode("FF"), "F")
        assert_equal(mintypecode("FD"), "D")
        assert_equal(mintypecode("Df"), "D")
        assert_equal(mintypecode("Dd"), "D")
        assert_equal(mintypecode("DF"), "D")
        assert_equal(mintypecode("DD"), "D")

    # 定义测试函数 test_default_3，用于进一步测试 mintypecode 函数的输入情况
    def test_default_3(self):
        # 断言调用 mintypecode 函数处理 "fdF" 后返回值为 "D"
        assert_equal(mintypecode("fdF"), "D")
        # assert_equal(mintypecode('fdF',savespace=1),'F')
        assert_equal(mintypecode("fdD"), "D")
        assert_equal(mintypecode("fFD"), "D")
        assert_equal(mintypecode("dFD"), "D")

        assert_equal(mintypecode("ifd"), "d")
        assert_equal(mintypecode("ifF"), "F")
        assert_equal(mintypecode("ifD"), "D")
        assert_equal(mintypecode("idF"), "D")
        # assert_equal(mintypecode('idF',savespace=1),'F')
        assert_equal(mintypecode("idD"), "D")
# 声明一个测试类 TestIsscalar，用于测试 np.isscalar 函数的行为
@xpassIfTorchDynamo  # (reason="TODO: decide on if [1] is a scalar or not")
class TestIsscalar(TestCase):
    
    # 定义一个测试方法 test_basic，测试 np.isscalar 的基本用法
    def test_basic(self):
        # 断言 3 是标量
        assert_(np.isscalar(3))
        # 断言列表 [3] 不是标量
        assert_(not np.isscalar([3]))
        # 断言元组 (3,) 不是标量
        assert_(not np.isscalar((3,)))
        # 断言 3j 是标量
        assert_(np.isscalar(3j))
        # 断言 4.0 是标量
        assert_(np.isscalar(4.0))


# 声明一个测试类 TestReal，用于测试 np.real 函数的行为
class TestReal(TestCase):
    
    # 定义一个测试方法 test_real，测试 np.real 在不同输入下的行为
    def test_real(self):
        # 生成一个形状为 (10,) 的随机数组 y
        y = np.random.rand(
            10,
        )
        # 断言 np.real(y) 等于 y 的元素本身（即实部）
        assert_array_equal(y, np.real(y))

        # 创建一个标量数组 np.array(1)
        y = np.array(1)
        # 调用 np.real(y)，out 接收结果
        out = np.real(y)
        # 断言 y 等于 out
        assert_array_equal(y, out)
        # 断言 out 的类型是 np.ndarray
        assert_(isinstance(out, np.ndarray))

        # 将标量 1 传递给 np.real(y)，out 接收结果
        y = 1
        out = np.real(y)
        # 断言 y 等于 out
        assert_equal(y, out)
        # 注释指出 out 不是 np.ndarray，而是 0 维张量（标量）

    # 定义一个测试方法 test_cmplx，测试 np.real 处理复数输入的行为
    def test_cmplx(self):
        # 生成一个形状为 (10,) 的随机复数数组 y
        y = np.random.rand(
            10,
        ) + 1j * np.random.rand(
            10,
        )
        # 断言 np.real(y) 等于 y 的实部
        assert_array_equal(y.real, np.real(y))

        # 创建一个复数数组 np.array(1 + 1j)
        y = np.array(1 + 1j)
        # 调用 np.real(y)，out 接收结果
        out = np.real(y)
        # 断言 y 的实部等于 out
        assert_array_equal(y.real, out)
        # 断言 out 的类型是 np.ndarray

        # 将复数 1 + 1j 传递给 np.real(y)，out 接收结果
        y = 1 + 1j
        out = np.real(y)
        # 断言 out 的实部等于 1.0
        assert_equal(1.0, out)
        # 注释指出 out 不是 np.ndarray，而是 0 维张量（标量）


# 声明一个测试类 TestImag，用于测试 np.imag 函数的行为
class TestImag(TestCase):
    
    # 定义一个测试方法 test_real，测试 np.imag 在不同输入下的行为
    def test_real(self):
        # 生成一个形状为 (10,) 的随机数组 y
        y = np.random.rand(
            10,
        )
        # 断言 np.imag(y) 等于 0
        assert_array_equal(0, np.imag(y))

        # 创建一个标量数组 np.array(1)
        y = np.array(1)
        # 调用 np.imag(y)，out 接收结果
        out = np.imag(y)
        # 断言 np.imag(y) 等于 0
        assert_array_equal(0, out)
        # 断言 out 的类型是 np.ndarray

        # 将标量 1 传递给 np.imag(y)，out 接收结果
        y = 1
        out = np.imag(y)
        # 断言 np.imag(y) 等于 0
        assert_equal(0, out)
        # 注释指出 out 不是 np.ndarray，而是 0 维张量（标量）

    # 定义一个测试方法 test_cmplx，测试 np.imag 处理复数输入的行为
    def test_cmplx(self):
        # 生成一个形状为 (10,) 的随机复数数组 y
        y = np.random.rand(
            10,
        ) + 1j * np.random.rand(
            10,
        )
        # 断言 np.imag(y) 等于 y 的虚部
        assert_array_equal(y.imag, np.imag(y))

        # 创建一个复数数组 np.array(1 + 1j)
        y = np.array(1 + 1j)
        # 调用 np.imag(y)，out 接收结果
        out = np.imag(y)
        # 断言 np.imag(y) 等于 y 的虚部
        assert_array_equal(y.imag, out)
        # 断言 out 的类型是 np.ndarray

        # 将复数 1 + 1j 传递给 np.imag(y)，out 接收结果
        y = 1 + 1j
        out = np.imag(y)
        # 断言 out 的虚部等于 1.0
        assert_equal(1.0, out)
        # 注释指出 out 不是 np.ndarray，而是 0 维张量（标量）


class TestIscomplex(TestCase):
    
    # 定义一个测试方法 test_fail，测试 iscomplex 函数对复数数组的行为
    def test_fail(self):
        # 创建一个数组 z 包含元素 [-1, 0, 1]
        z = np.array([-1, 0, 1])
        # 调用 iscomplex(z)，res 接收结果
        res = iscomplex(z)
        # 断言 res 中没有任何元素为 True
        assert_(not np.sometrue(res, axis=0))

    # 定义一个测试方法 test_pass，测试 iscomplex 函数对复数数组的行为
    def test_pass(self):
        # 创建一个数组 z 包含元素 [-1j, 1, 0]
        z = np.array([-1j, 1, 0])
        # 调用 iscomplex(z)，res 接收结果
        res = iscomplex(z)
        # 断言 res 等于 [1, 0, 0]
        assert_array_equal(res, [1, 0, 0])


class TestIsreal(TestCase):
    
    # 定义一个测试方法 test_pass，测试 isreal 函数对数组的行为
    def test_pass(self):
        # 创建一个数组 z 包含元素 [-1, 0, 1j]
        z = np.array([-1, 0, 1j])
        # 调用 isreal(z)，res 接收结果
        res = isreal(z)
        # 断言 res 等于 [1, 1, 0]
        assert_array_equal(res, [1, 1, 0])

    # 定义一个测试方法 test_fail，测试 isreal 函数对数组的行为
    def test_fail(self):
        # 创建一个数组 z 包含元素 [-1j, 1, 0]
        z = np.array([-1j, 1, 0])
        # 调用 isreal(z)，res 接收结果
        res = isreal(z)
        # 断言 res 等于 [0, 1, 1]
        assert_array_equal(res, [0, 1, 1])

    # 定义一个测试方法 test_isreal_real，测试 isreal 函数对数组的行为
    def test_isreal_real(self):
        # 创建一个数组 z 包含元素 [-1, 0, 1]
        z = np.array([-1, 0, 1])
        # 调用 isreal(z)，res 接收结果
        res = isreal(z)
        # 断言 res 中的所有元素都为 True
        assert res.all()


class TestIscomplexobj(TestCase):
    pass  # 这个类还未实现任何测试
    # 定义测试类，包含了多个测试方法用于测试 iscomplexobj 函数的行为
    def test_basic(self):
        # 创建一个包含整数的 NumPy 数组
        z = np.array([-1, 0, 1])
        # 断言：z 不是复数对象
        assert_(not iscomplexobj(z))
        
        # 将数组 z 修改为包含复数的 NumPy 数组
        z = np.array([-1j, 0, -1])
        # 断言：z 是复数对象
        assert_(iscomplexobj(z))
    
    # 定义测试类的第二个测试方法
    def test_scalar(self):
        # 断言：1.0 不是复数对象
        assert_(not iscomplexobj(1.0))
        # 断言：1 + 0j 是复数对象
        assert_(iscomplexobj(1 + 0j))
    
    # 定义测试类的第三个测试方法
    def test_list(self):
        # 断言：包含复数的列表是复数对象
        assert_(iscomplexobj([3, 1 + 0j, True]))
        # 断言：不包含复数的列表不是复数对象
        assert_(not iscomplexobj([3, 1, True]))
class TestIsrealobj(TestCase):
    # 测试是否为实数对象的函数
    def test_basic(self):
        # 创建一个包含负数、零、正数的NumPy数组
        z = np.array([-1, 0, 1])
        # 断言该数组为实数对象
        assert_(isrealobj(z))
        # 创建一个包含负虚数、零、负数的NumPy数组
        z = np.array([-1j, 0, -1])
        # 断言该数组不是实数对象
        assert_(not isrealobj(z))


class TestIsnan(TestCase):
    # 测试np.isnan函数的各种情况
    def test_goodvalues(self):
        # 创建一个包含负数、零、正数的NumPy数组
        z = np.array((-1.0, 0.0, 1.0))
        # 使用np.isnan检查数组中的值是否不是NaN，返回结果与0比较
        res = np.isnan(z) == 0
        # 断言数组中所有元素不是NaN
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        # 断言np.isnan检查正无穷除以0的结果是否不是NaN
        assert_all(np.isnan(np.array((1.0,)) / 0.0) == 0)

    def test_neginf(self):
        # 断言np.isnan检查负无穷除以0的结果是否不是NaN
        assert_all(np.isnan(np.array((-1.0,)) / 0.0) == 0)

    def test_ind(self):
        # 断言np.isnan检查零除以0的结果是否是NaN
        assert_all(np.isnan(np.array((0.0,)) / 0.0) == 1)

    def test_integer(self):
        # 断言np.isnan检查整数1是否不是NaN
        assert_all(np.isnan(1) == 0)

    def test_complex(self):
        # 断言np.isnan检查复数1+1j是否不是NaN
        assert_all(np.isnan(1 + 1j) == 0)

    def test_complex1(self):
        # 断言np.isnan检查复数0+0j除以0的结果是否是NaN
        assert_all(np.isnan(np.array(0 + 0j) / 0.0) == 1)


class TestIsfinite(TestCase):
    # Fixme, wrong place, isfinite now ufunc

    def test_goodvalues(self):
        # 创建一个包含负数、零、正数的NumPy数组
        z = np.array((-1.0, 0.0, 1.0))
        # 使用np.isfinite检查数组中的值是否有限，返回结果与1比较
        res = np.isfinite(z) == 1
        # 断言数组中所有元素是有限的
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        # 断言np.isfinite检查正无穷除以0的结果是否是有限的
        assert_all(np.isfinite(np.array((1.0,)) / 0.0) == 0)

    def test_neginf(self):
        # 断言np.isfinite检查负无穷除以0的结果是否是有限的
        assert_all(np.isfinite(np.array((-1.0,)) / 0.0) == 0)

    def test_ind(self):
        # 断言np.isfinite检查零除以0的结果是否是有限的
        assert_all(np.isfinite(np.array((0.0,)) / 0.0) == 0)

    def test_integer(self):
        # 断言np.isfinite检查整数1是否是有限的
        assert_all(np.isfinite(1) == 1)

    def test_complex(self):
        # 断言np.isfinite检查复数1+1j是否是有限的
        assert_all(np.isfinite(1 + 1j) == 1)

    def test_complex1(self):
        # 断言np.isfinite检查复数1+1j除以0的结果是否是有限的
        assert_all(np.isfinite(np.array(1 + 1j) / 0.0) == 0)


class TestIsinf(TestCase):
    # Fixme, wrong place, isinf now ufunc

    def test_goodvalues(self):
        # 创建一个包含负数、零、正数的NumPy数组
        z = np.array((-1.0, 0.0, 1.0))
        # 使用np.isinf检查数组中的值是否为无穷，返回结果与0比较
        res = np.isinf(z) == 0
        # 断言数组中所有元素不是无穷
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        # 断言np.isinf检查正无穷除以0的结果是否是无穷
        assert_all(np.isinf(np.array((1.0,)) / 0.0) == 1)

    def test_posinf_scalar(self):
        # 断言np.isinf检查标量正无穷除以0的结果是否是无穷
        assert_all(
            np.isinf(
                np.array(
                    1.0,
                )
                / 0.0
            )
            == 1
        )

    def test_neginf(self):
        # 断言np.isinf检查负无穷除以0的结果是否是无穷
        assert_all(np.isinf(np.array((-1.0,)) / 0.0) == 1)

    def test_neginf_scalar(self):
        # 断言np.isinf检查标量负无穷除以0的结果是否是无穷
        assert_all(np.isinf(np.array(-1.0) / 0.0) == 1)

    def test_ind(self):
        # 断言np.isinf检查零除以0的结果是否不是无穷
        assert_all(np.isinf(np.array((0.0,)) / 0.0) == 0)


class TestIsposinf(TestCase):
    # 测试是否为正无穷的函数
    def test_generic(self):
        # 使用isposinf检查数组中各元素除以0的结果是否为正无穷
        vals = isposinf(np.array((-1.0, 0, 1)) / 0.0)
        # 断言第一个元素不是正无穷，第二个元素不是正无穷，第三个元素是正无穷
        assert_(vals[0] == 0)
        assert_(vals[1] == 0)
        assert_(vals[2] == 1)


class TestIsneginf(TestCase):
    # 测试是否为负无穷的函数
    def test_generic(self):
        # 使用isneginf检查数组中各元素除以0的结果是否为负无穷
        vals = isneginf(np.array((-1.0, 0, 1)) / 0.0)
        # 断言第一个元素是负无穷，第二个元素不是负无穷，第三个元素不是负无穷
        assert_(vals[0] == 1)
        assert_(vals[1] == 0)
        assert_(vals[2] == 0)


# @xfail  #(reason="not implemented")
class TestNanToNum(TestCase):
    # 这个测试类还没有实现，使用xfail标记为预期失败
    # 定义一个测试函数，用于测试 nan_to_num 函数的通用情况
    def test_generic(self):
        # 使用 nan_to_num 处理数组中的 NaN 值，将其替换为特定的数值，确保第一个值远小于负一百亿，且为有限数值
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0)
        assert_all(vals[0] < -1e10) and assert_all(np.isfinite(vals[0]))
        # 确保第二个值为 0
        assert_(vals[1] == 0)
        # 确保第三个值远大于十亿，且为有限数值
        assert_all(vals[2] > 1e10) and assert_all(np.isfinite(vals[2]))
        # 确保 vals 是一个 NumPy 数组对象
        assert isinstance(vals, np.ndarray)

        # 使用 nan_to_num 处理数组中的 NaN 值，同时使用特定的替换数值（nan=10, posinf=20, neginf=30）
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0, nan=10, posinf=20, neginf=30)
        # 确保处理后的数组符合预期值 [30, 10, 20]
        assert_equal(vals, [30, 10, 20])
        # 确保处理后数组中指定索引位置的值为有限数值
        assert_all(np.isfinite(vals[[0, 2]]))
        # 确保 vals 是一个 NumPy 数组对象
        assert isinstance(vals, np.ndarray)

    # 定义一个测试函数，用于测试 nan_to_num 处理普通数组的情况
    def test_array(self):
        # 使用 nan_to_num 处理包含 NaN 值的普通数组
        vals = nan_to_num([1])
        # 确保处理后的数组与给定的 NumPy 数组相等
        assert_array_equal(vals, np.array([1], int))
        # 确保 vals 是一个 NumPy 数组对象
        assert isinstance(vals, np.ndarray)
        # 使用 nan_to_num 处理普通数组，同时使用特定的替换数值（nan=10, posinf=20, neginf=30）
        vals = nan_to_num([1], nan=10, posinf=20, neginf=30)
        # 确保处理后的数组与给定的 NumPy 数组相等
        assert_array_equal(vals, np.array([1], int))
        # 确保 vals 是一个 NumPy 数组对象
        assert isinstance(vals, np.ndarray)

    # 定义一个测试函数，用于测试 nan_to_num 处理整数的情况
    @skip(reason="we return OD arrays not scalars")
    def test_integer(self):
        # 使用 nan_to_num 处理整数
        vals = nan_to_num(1)
        # 确保处理后的值与原值相等
        assert_all(vals == 1)
        # 确保 vals 是一个 NumPy 整数对象
        assert isinstance(vals, np.int_)
        # 使用 nan_to_num 处理整数，同时使用特定的替换数值（nan=10, posinf=20, neginf=30）
        vals = nan_to_num(1, nan=10, posinf=20, neginf=30)
        # 确保处理后的值与原值相等
        assert_all(vals == 1)
        # 确保 vals 是一个 NumPy 整数对象
        assert isinstance(vals, np.int_)

    # 定义一个测试函数，用于测试 nan_to_num 处理浮点数的情况
    @skip(reason="we return OD arrays not scalars")
    def test_float(self):
        # 使用 nan_to_num 处理浮点数
        vals = nan_to_num(1.0)
        # 确保处理后的值与原值相等
        assert_all(vals == 1.0)
        # 确保 vals 的类型是 NumPy 浮点数对象
        assert_equal(type(vals), np.float_)
        # 使用 nan_to_num 处理浮点数，同时使用特定的替换数值（nan=10, posinf=20, neginf=30）
        vals = nan_to_num(1.1, nan=10, posinf=20, neginf=30)
        # 确保处理后的值与原值相等
        assert_all(vals == 1.1)
        # 确保 vals 的类型是 NumPy 浮点数对象
        assert_equal(type(vals), np.float_)

    # 定义一个测试函数，用于测试 nan_to_num 处理复数的情况（处理有效复数）
    @skip(reason="we return OD arrays not scalars")
    def test_complex_good(self):
        # 使用 nan_to_num 处理有效的复数
        vals = nan_to_num(1 + 1j)
        # 确保处理后的复数值与原复数值相等
        assert_all(vals == 1 + 1j)
        # 确保 vals 是一个 NumPy 复数对象
        assert isinstance(vals, np.complex_)
        # 使用 nan_to_num 处理有效的复数，同时使用特定的替换数值（nan=10, posinf=20, neginf=30）
        vals = nan_to_num(1 + 1j, nan=10, posinf=20, neginf=30)
        # 确保处理后的复数值与原复数值相等
        assert_all(vals == 1 + 1j)
        # 确保 vals 的类型是 NumPy 复数对象
        assert_equal(type(vals), np.complex_)

    # 定义一个测试函数，用于测试 nan_to_num 处理复数的情况（处理无效复数）
    @skip(reason="we return OD arrays not scalars")
    def test_complex_bad(self):
        v = 1 + 1j
        v += np.array(0 + 1.0j) / 0.0
        vals = nan_to_num(v)
        # 确保处理后的复数值是有限的
        assert_all(np.isfinite(vals))
        # 确保 vals 的类型是 NumPy 复数对象
        assert_equal(type(vals), np.complex_)

    # 定义一个测试函数，用于测试 nan_to_num 处理复数的情况（处理无效复数，第二种情况）
    @skip(reason="we return OD arrays not scalars")
    def test_complex_bad2(self):
        v = 1 + 1j
        v += np.array(-1 + 1.0j) / 0.0
        vals = nan_to_num(v)
        # 确保处理后的复数值是有限的
        assert_all(np.isfinite(vals))
        # 确保 vals 的类型是 NumPy 复数对象
        assert_equal(type(vals), np.complex_)
        # Fixme
        # assert_all(vals.imag > 1e10)  and assert_all(np.isfinite(vals))
        # !! This is actually (unexpectedly) positive
        # !! inf.  Comment out for now, and see if it
        # !! changes
        # assert_all(vals.real < -1e10) and assert_all(np.isfinite(vals))
    def test_do_not_rewrite_previous_keyword(self):
        # 定义一个测试函数，检查在特定情况下，确保当 nan=np.inf 时，这些值不被 posinf 关键字重新赋值为 posinf 的值。

        # 使用 nan_to_num 函数处理一个由 (-1.0, 0, 1) 除以 0.0 得到的 numpy 数组，将 nan 替换为 np.inf，将 posinf 替换为 999。
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0, nan=np.inf, posinf=999)

        # 断言处理后的 vals 数组中的值都是有限的
        assert_all(np.isfinite(vals[[0, 2]]))

        # 断言 vals[0] 的值小于 -1e10
        assert_all(vals[0] < -1e10)

        # 断言 vals 中索引为 1 和 2 的值分别等于 np.inf 和 999
        assert_equal(vals[[1, 2]], [np.inf, 999])

        # 断言 vals 是一个 numpy 数组的实例
        assert isinstance(vals, np.ndarray)
class TestRealIfClose(TestCase):
    # 定义测试类 TestRealIfClose，继承自 TestCase
    def test_basic(self):
        # 定义测试方法 test_basic
        a = np.random.rand(10)
        # 生成一个包含 10 个随机数的 NumPy 数组 a
        b = real_if_close(a + 1e-15j)
        # 调用 real_if_close 函数，处理 a + 1e-15j 的复数数组，返回结果赋给 b
        assert_all(isrealobj(b))
        # 断言 b 中的所有元素是实数
        assert_array_equal(a, b)
        # 断言 a 和 b 的内容完全相等
        b = real_if_close(a + 1e-7j)
        # 再次调用 real_if_close 函数，处理 a + 1e-7j 的复数数组，返回结果赋给 b
        assert_all(iscomplexobj(b))
        # 断言 b 中的所有元素是复数
        b = real_if_close(a + 1e-7j, tol=1e-6)
        # 使用指定的容差调用 real_if_close 函数，处理 a + 1e-7j 的复数数组，返回结果赋给 b
        assert_all(isrealobj(b))


@xfail  # (reason="not implemented")
# 标记为预期失败的测试类 TestArrayConversion
class TestArrayConversion(TestCase):
    # 定义测试类 TestArrayConversion，继承自 TestCase
    def test_asfarray(self):
        # 定义测试方法 test_asfarray
        a = asfarray(np.array([1, 2, 3]))
        # 调用 asfarray 函数，将整数数组 [1, 2, 3] 转换为浮点数数组 a
        assert_equal(a.__class__, np.ndarray)
        # 断言 a 的类是 np.ndarray
        assert_(np.issubdtype(a.dtype, np.floating))
        # 断言 a 的数据类型是浮点数类型

        # previously this would infer dtypes from arrays, unlike every single
        # other numpy function
        # 以前，它会从数组推断数据类型，与所有其他 NumPy 函数不同
        assert_raises(TypeError, asfarray, np.array([1, 2, 3]), dtype=np.array(1.0))
        # 断言调用 asfarray 函数时，使用 dtype 参数传入不合法参数时会抛出 TypeError 异常


if __name__ == "__main__":
    run_tests()
    # 如果作为主程序执行，则运行所有测试
```