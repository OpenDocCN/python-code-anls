# `.\numpy\numpy\_core\tests\test_ufunc.py`

```
# 引入警告模块，用于管理警告消息的显示
import warnings
# itertools 提供用于构建迭代器的工具
import itertools
# sys 提供了对 Python 解释器的访问
import sys
# ctypes 提供与 C 语言兼容的数据类型定义
import ctypes as ct
# pickle 提供对象序列化和反序列化的功能
import pickle

# 引入 pytest 测试框架及其参数化功能
import pytest
from pytest import param

# 引入 numpy 科学计算库
import numpy as np
# numpy 的核心数学运算模块
import numpy._core.umath as ncu
# numpy 内部数学测试
import numpy._core._umath_tests as umt
# numpy 线性代数运算
import numpy.linalg._umath_linalg as uml
# numpy 内核操作标志测试
import numpy._core._operand_flag_tests as opflag_tests
# numpy 内核有理数测试
import numpy._core._rational_tests as _rational_tests
# 引入异常处理模块，处理轴异常
from numpy.exceptions import AxisError
# 引入 numpy 测试模块中的断言函数
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_array_almost_equal, assert_no_warnings,
    assert_allclose, HAS_REFCOUNT, suppress_warnings, IS_WASM, IS_PYPY,
)
# 引入 numpy 测试模块中的内存需求装饰器
from numpy.testing._private.utils import requires_memory

# 从 numpy 核心数学模块中筛选出所有一元通用函数（ufunc）
UNARY_UFUNCS = [obj for obj in np._core.umath.__dict__.values()
                    if isinstance(obj, np.ufunc)]
# 筛选出所有以对象为输入和输出的一元 ufunc 函数
UNARY_OBJECT_UFUNCS = [uf for uf in UNARY_UFUNCS if "O->O" in uf.types]

# 移除不支持浮点数输入的函数，如 bitwise_count
# 这些函数无法接受浮点数作为输入参数
UNARY_OBJECT_UFUNCS.remove(getattr(np, 'bitwise_count'))


class TestUfuncKwargs:
    # 测试 ufunc 的关键字参数异常情况

    def test_kwarg_exact(self):
        # 测试在添加无效关键字参数时是否会抛出 TypeError
        assert_raises(TypeError, np.add, 1, 2, castingx='safe')
        assert_raises(TypeError, np.add, 1, 2, dtypex=int)
        assert_raises(TypeError, np.add, 1, 2, extobjx=[4096])
        assert_raises(TypeError, np.add, 1, 2, outx=None)
        assert_raises(TypeError, np.add, 1, 2, sigx='ii->i')
        assert_raises(TypeError, np.add, 1, 2, signaturex='ii->i')
        assert_raises(TypeError, np.add, 1, 2, subokx=False)
        assert_raises(TypeError, np.add, 1, 2, wherex=[True])

    def test_sig_signature(self):
        # 测试在同时提供 sig 和 signature 参数时是否会抛出 TypeError
        assert_raises(TypeError, np.add, 1, 2, sig='ii->i',
                      signature='ii->i')

    def test_sig_dtype(self):
        # 测试在提供 sig 或 signature 参数时，不能再提供 dtype 参数
        assert_raises(TypeError, np.add, 1, 2, sig='ii->i',
                      dtype=int)
        assert_raises(TypeError, np.add, 1, 2, signature='ii->i',
                      dtype=int)

    def test_extobj_removed(self):
        # 测试 extobj 参数在最新版本中已移除，尝试使用时是否会抛出 TypeError
        assert_raises(TypeError, np.add, 1, 2, extobj=[4096])


class TestUfuncGenericLoops:
    """Test generic loops.

    The loops to be tested are:

        PyUFunc_ff_f_As_dd_d
        PyUFunc_ff_f
        PyUFunc_dd_d
        PyUFunc_gg_g
        PyUFunc_FF_F_As_DD_D
        PyUFunc_DD_D
        PyUFunc_FF_F
        PyUFunc_GG_G
        PyUFunc_OO_O
        PyUFunc_OO_O_method
        PyUFunc_f_f_As_d_d
        PyUFunc_d_d
        PyUFunc_f_f
        PyUFunc_g_g
        PyUFunc_F_F_As_D_D
        PyUFunc_F_F
        PyUFunc_D_D
        PyUFunc_G_G
        PyUFunc_O_O
        PyUFunc_O_O_method
        PyUFunc_On_Om

    Where:

        f -- float
        d -- double
        g -- long double
        F -- complex float
        D -- complex double
        G -- complex long double
        O -- python object

    It is difficult to assure that each of these loops is entered from the
    Python level as the special cased loops are a moving target and the
    corresponding types are architecture dependent. We probably need to
    """
    # 测试通用循环的各种类型

    pass  # 由于这部分没有具体的代码，只有文档注释，因此使用 pass 占位
    # 定义 C 级别的测试函数，用于测试 Python UFuncs。目前，我只是查看构建目录中注册的签名，以找到相关的函数。
    
    """
    np_dtypes = [
        (np.single, np.single), (np.single, np.double),
        (np.csingle, np.csingle), (np.csingle, np.cdouble),
        (np.double, np.double), (np.longdouble, np.longdouble),
        (np.cdouble, np.cdouble), (np.clongdouble, np.clongdouble)]
    
    @pytest.mark.parametrize('input_dtype,output_dtype', np_dtypes)
    # 参数化测试，对于每组 input_dtype 和 output_dtype 进行测试
    def test_unary_PyUFunc(self, input_dtype, output_dtype, f=np.exp, x=0, y=1):
        # 创建一个填充了特定类型数据的数组 xs，数据从 x 转换而来，类型为 output_dtype
        xs = np.full(10, input_dtype(x), dtype=output_dtype)
        # 对 xs 应用函数 f，然后每隔一个取一个元素，存入 ys
        ys = f(xs)[::2]
        # 断言 ys 与预期值 y 在允许误差范围内相等
        assert_allclose(ys, y)
        # 断言 ys 的数据类型为 output_dtype
        assert_equal(ys.dtype, output_dtype)
    
    def f2(x, y):
        return x**y
    
    @pytest.mark.parametrize('input_dtype,output_dtype', np_dtypes)
    # 参数化测试，对于每组 input_dtype 和 output_dtype 进行测试
    def test_binary_PyUFunc(self, input_dtype, output_dtype, f=f2, x=0, y=1):
        # 创建一个填充了特定类型数据的数组 xs，数据从 x 转换而来，类型为 output_dtype
        xs = np.full(10, input_dtype(x), dtype=output_dtype)
        # 对 xs 应用函数 f，然后每隔一个取一个元素，存入 ys
        ys = f(xs, xs)[::2]
        # 断言 ys 与预期值 y 在允许误差范围内相等
        assert_allclose(ys, y)
        # 断言 ys 的数据类型为 output_dtype
        assert_equal(ys.dtype, output_dtype)
    
    # 用于测试对象方法循环的类
    class foo:
        def conjugate(self):
            return np.bool(1)
    
        def logical_xor(self, obj):
            return np.bool(1)
    
    def test_unary_PyUFunc_O_O(self):
        # 创建一个包含对象的数组 x，对象均为 1
        x = np.ones(10, dtype=object)
        # 断言对 x 应用 np.abs 函数结果全为 1
        assert_(np.all(np.abs(x) == 1))
    
    def test_unary_PyUFunc_O_O_method_simple(self, foo=foo):
        # 创建一个填充了 foo 类对象的数组 x
        x = np.full(10, foo(), dtype=object)
        # 断言对 x 应用 np.conjugate 方法结果全为 True
        assert_(np.all(np.conjugate(x) == True))
    
    def test_binary_PyUFunc_OO_O(self):
        # 创建一个包含对象的数组 x，对象均为 1
        x = np.ones(10, dtype=object)
        # 断言对 x 应用 np.add 函数结果全为 2
        assert_(np.all(np.add(x, x) == 2))
    
    def test_binary_PyUFunc_OO_O_method(self, foo=foo):
        # 创建一个填充了 foo 类对象的数组 x
        x = np.full(10, foo(), dtype=object)
        # 断言对 x 应用 np.logical_xor 方法结果全为 True
        assert_(np.all(np.logical_xor(x, x)))
    
    def test_binary_PyUFunc_On_Om_method(self, foo=foo):
        # 创建一个填充了 foo 类对象的多维数组 x
        x = np.full((10, 2, 3), foo(), dtype=object)
        # 断言对 x 应用 np.logical_xor 方法结果全为 True
        assert_(np.all(np.logical_xor(x, x)))
    
    def test_python_complex_conjugate(self):
        # 复数的共轭 ufunc 应该调用方法来处理：
        # 创建一个对象数组 arr，包含复数对象
        arr = np.array([1+2j, 3-4j], dtype="O")
        # 断言 arr 的第一个元素是 complex 类型
        assert isinstance(arr[0], complex)
        # 对 arr 应用 np.conjugate，结果应为对象类型
        res = np.conjugate(arr)
        # 断言 res 的数据类型为对象类型
        assert res.dtype == np.dtype("O")
        # 断言 res 与预期值数组相等
        assert_array_equal(res, np.array([1-2j, 3+4j], dtype="O"))
    
    @pytest.mark.parametrize("ufunc", UNARY_OBJECT_UFUNCS)
    # 定义一个测试方法，用于测试一元通用函数的对象和非对象计算结果的比较
    def test_unary_PyUFunc_O_O_method_full(self, ufunc):
        """Compare the result of the object loop with non-object one"""
        
        # 将 np.pi/4 转换为 np.float64 类型
        val = np.float64(np.pi/4)

        # 定义一个继承自 np.float64 的子类 MyFloat，重载 __getattr__ 方法
        class MyFloat(np.float64):
            # 当对象调用不存在的属性时，尝试从 np._core.umath 中获取对应属性并计算
            def __getattr__(self, attr):
                try:
                    return super().__getattr__(attr)
                except AttributeError:
                    return lambda: getattr(np._core.umath, attr)(val)

        # 创建一个包含 val 的 np.float64 类型的零维数组 num_arr
        num_arr = np.array(val, dtype=np.float64)
        # 创建一个包含 MyFloat(val) 的对象数组 obj_arr，元素类型为 "O"
        obj_arr = np.array(MyFloat(val), dtype="O")

        # 设置浮点运算错误状态为 "raise"，用于捕获异常
        with np.errstate(all="raise"):
            try:
                # 尝试计算 num_arr 上的 ufunc
                res_num = ufunc(num_arr)
            except Exception as exc:
                # 如果计算 num_arr 时出现异常，则期望在计算 obj_arr 上的 ufunc 时也会出现相同类型的异常
                with assert_raises(type(exc)):
                    ufunc(obj_arr)
            else:
                # 如果 num_arr 计算成功，计算 obj_arr 上的 ufunc，并近似比较两者的结果
                res_obj = ufunc(obj_arr)
                assert_array_almost_equal(res_num.astype("O"), res_obj)
# 定义一个空的函数 _pickleable_module_global，没有任何实际操作
def _pickleable_module_global():
    pass

# 定义一个名为 TestUfunc 的测试类
class TestUfunc:
    # 定义测试函数 test_pickle，用于测试 ufunc 的序列化和反序列化
    def test_pickle(self):
        # 遍历从协议版本 2 到最高协议版本的范围
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 断言序列化后的 np.sin 函数与其本身相等
            assert_(pickle.loads(pickle.dumps(np.sin,
                                              protocol=proto)) is np.sin)

            # 检查非位于顶层 numpy 命名空间的 ufunc（例如 numpy._core._rational_tests.test_add）是否能被序列化和反序列化
            res = pickle.loads(pickle.dumps(_rational_tests.test_add,
                                            protocol=proto))
            assert_(res is _rational_tests.test_add)

    # 定义测试函数 test_pickle_withstring，用于测试从字符串反序列化 ufunc
    def test_pickle_withstring(self):
        # 定义一个包含 ufunc 序列化字符串的字节串 astring
        astring = (b"cnumpy.core\n_ufunc_reconstruct\np0\n"
                   b"(S'numpy._core.umath'\np1\nS'cos'\np2\ntp3\nRp4\n.")
        # 断言从 astring 反序列化后的对象是 np.cos 函数
        assert_(pickle.loads(astring) is np.cos)

    # 根据条件跳过测试（在 PyPy 环境下不支持 'is' 操作）
    @pytest.mark.skipif(IS_PYPY, reason="'is' check does not work on PyPy")
    # 定义测试函数 test_pickle_name_is_qualname，测试 ufunc 的序列化和反序列化
    def test_pickle_name_is_qualname(self):
        # 将全局变量 _pickleable_module_global.ufunc 设为 umt._pickleable_module_global_ufunc
        _pickleable_module_global.ufunc = umt._pickleable_module_global_ufunc

        # 反序列化 _pickleable_module_global.ufunc，并断言其与 umt._pickleable_module_global_ufunc 是同一个对象
        obj = pickle.loads(pickle.dumps(_pickleable_module_global.ufunc))
        assert obj is umt._pickleable_module_global_ufunc

    # 定义测试函数 test_reduceat_shifting_sum，测试 np.add.reduceat 的使用
    def test_reduceat_shifting_sum(self):
        L = 6
        x = np.arange(L)
        # 构造索引数组 idx，用于指定 reduceat 操作的位置
        idx = np.array(list(zip(np.arange(L - 2), np.arange(L - 2) + 2))).ravel()
        # 断言 np.add.reduceat(x, idx)[::2] 的结果与指定的列表相等
        assert_array_equal(np.add.reduceat(x, idx)[::2], [1, 3, 5, 7])

    # 定义测试函数 test_signature0，测试 umt.test_signature 函数的行为
    def test_signature0(self):
        # 调用 umt.test_signature 函数，获取返回的各个参数
        # 2 是 nin，1 是 nout，"(i),(i)->()" 是 core_signature
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(i),(i)->()")
        # 断言返回的各个参数符合预期
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1,  1,  0))
        assert_equal(ixs, (0, 0))
        assert_equal(flags, (self.size_inferred,))
        assert_equal(sizes, (-1,))

    # 定义测试函数 test_signature1，测试 umt.test_signature 函数的行为（空的 core signature）
    def test_signature1(self):
        # 调用 umt.test_signature 函数，获取返回的各个参数
        # 2 是 nin，1 是 nout，"(),()->()" 是 core_signature
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(),()->()")
        # 断言返回的各个参数符合预期
        assert_equal(enabled, 0)
        assert_equal(num_dims, (0,  0,  0))
        assert_equal(ixs, ())
        assert_equal(flags, ())
        assert_equal(sizes, ())

    # 定义测试函数 test_signature2，测试 umt.test_signature 函数的行为（复杂的 core signature）
    def test_signature2(self):
        # 调用 umt.test_signature 函数，获取返回的各个参数
        # 2 是 nin，1 是 nout，"(i1,i2),(J_1)->(_kAB)" 是 core_signature
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(i1,i2),(J_1)->(_kAB)")
        # 断言返回的各个参数符合预期
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 1, 1))
        assert_equal(ixs, (0, 1, 2, 3))
        assert_equal(flags, (self.size_inferred,)*4)
        assert_equal(sizes, (-1, -1, -1, -1))
    # 测试函数 test_signature3，测试 umt.test_signature 函数的返回结果是否符合预期
    def test_signature3(self):
        # 调用 umt.test_signature 函数，传入参数并接收返回的多个变量
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(i1, i12),   (J_1)->(i12, i2)")
        # 断言各变量与预期值相等
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 1, 2))
        assert_equal(ixs, (0, 1, 2, 1, 3))
        assert_equal(flags, (self.size_inferred,)*4)
        assert_equal(sizes, (-1, -1, -1, -1))

    # 测试函数 test_signature4，测试 umt.test_signature 函数的返回结果是否符合预期
    def test_signature4(self):
        # 调用 umt.test_signature 函数，传入参数并接收返回的多个变量
        # matrix_multiply 签名来自 _umath_tests
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(n,k),(k,m)->(n,m)")
        # 断言各变量与预期值相等
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 2, 2))
        assert_equal(ixs, (0, 1, 1, 2, 0, 2))
        assert_equal(flags, (self.size_inferred,)*3)
        assert_equal(sizes, (-1, -1, -1))

    # 测试函数 test_signature5，测试 umt.test_signature 函数的返回结果是否符合预期
    def test_signature5(self):
        # 调用 umt.test_signature 函数，传入参数并接收返回的多个变量
        # matmul 签名来自 _umath_tests
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(n?,k),(k,m?)->(n?,m?)")
        # 断言各变量与预期值相等
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 2, 2))
        assert_equal(ixs, (0, 1, 1, 2, 0, 2))
        assert_equal(flags, (self.size_inferred | self.can_ignore,
                             self.size_inferred,
                             self.size_inferred | self.can_ignore))
        assert_equal(sizes, (-1, -1, -1))

    # 测试函数 test_signature6，测试 umt.test_signature 函数的返回结果是否符合预期
    def test_signature6(self):
        # 调用 umt.test_signature 函数，传入参数并接收返回的多个变量
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            1, 1, "(3)->()")
        # 断言各变量与预期值相等
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 0))
        assert_equal(ixs, (0,))
        assert_equal(flags, (0,))
        assert_equal(sizes, (3,))

    # 测试函数 test_signature7，测试 umt.test_signature 函数的返回结果是否符合预期
    def test_signature7(self):
        # 调用 umt.test_signature 函数，传入参数并接收返回的多个变量
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "(3),(03,3),(n)->(9)")
        # 断言各变量与预期值相等
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (0, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    # 测试函数 test_signature8，测试 umt.test_signature 函数的返回结果是否符合预期
    def test_signature8(self):
        # 调用 umt.test_signature 函数，传入参数并接收返回的多个变量
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "(3?),(3?,3?),(n)->(9)")
        # 断言各变量与预期值相等
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (self.can_ignore, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    # 测试函数 test_signature9，测试 umt.test_signature 函数的返回结果是否符合预期
    def test_signature9(self):
        # 调用 umt.test_signature 函数，传入参数并接收返回的多个变量
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            1, 1, "(  3)  -> ( )")
        # 断言各变量与预期值相等
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 0))
        assert_equal(ixs, (0,))
        assert_equal(flags, (0,))
        assert_equal(sizes, (3,))
    def test_signature10(self):
        # 调用 umt 模块的 test_signature 函数，测试签名解析功能
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "( 3? ) , (3? ,  3?) ,(n )-> ( 9)")
        # 断言各返回值与预期相等
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (self.can_ignore, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature_failure_extra_parenthesis(self):
        # 测试当输入签名中包含多余括号时是否能正确抛出 ValueError 异常
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "((i)),(i)->()")

    def test_signature_failure_mismatching_parenthesis(self):
        # 测试当输入签名中括号不匹配时是否能正确抛出 ValueError 异常
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "(i),)i(->()")

    def test_signature_failure_signature_missing_input_arg(self):
        # 测试当输入签名中缺少输入参数时是否能正确抛出 ValueError 异常
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "(i),->()")

    def test_signature_failure_signature_missing_output_arg(self):
        # 测试当输入签名中缺少输出参数时是否能正确抛出 ValueError 异常
        with assert_raises(ValueError):
            umt.test_signature(2, 2, "(i),(i)->()")

    def test_get_signature(self):
        # 断言 np.vecdot 的签名与预期相符
        assert_equal(np.vecdot.signature, "(n),(n)->()")

    def test_forced_sig(self):
        # 测试 np.add 函数的不同签名及参数组合是否能正确计算结果
        a = 0.5 * np.arange(3, dtype='f8')
        assert_equal(np.add(a, 0.5), [0.5, 1, 1.5])
        with pytest.warns(DeprecationWarning):
            assert_equal(np.add(a, 0.5, sig='i', casting='unsafe'), [0, 0, 1])
        assert_equal(np.add(a, 0.5, sig='ii->i', casting='unsafe'), [0, 0, 1])
        with pytest.warns(DeprecationWarning):
            assert_equal(np.add(a, 0.5, sig=('i4',), casting='unsafe'), [0, 0, 1])
        assert_equal(np.add(a, 0.5, sig=('i4', 'i4', 'i4'), casting='unsafe'), [0, 0, 1])

        b = np.zeros((3,), dtype='f8')
        np.add(a, 0.5, out=b)
        assert_equal(b, [0.5, 1, 1.5])
        b[:] = 0
        with pytest.warns(DeprecationWarning):
            np.add(a, 0.5, sig='i', out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        np.add(a, 0.5, sig='ii->i', out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        with pytest.warns(DeprecationWarning):
            np.add(a, 0.5, sig=('i4',), out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        np.add(a, 0.5, sig=('i4', 'i4', 'i4'), out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])

    def test_signature_all_None(self):
        # 测试在不提供签名时，使用所有参数为 None 的情况是否等同于不提供签名
        res1 = np.add([3], [4], sig=(None, None, None))
        res2 = np.add([3], [4])
        assert_array_equal(res1, res2)
        res1 = np.maximum([3], [4], sig=(None, None, None))
        res2 = np.maximum([3], [4])
        assert_array_equal(res1, res2)

        with pytest.raises(TypeError):
            # 当提供了 signature 参数时会引发 TypeError 异常
            np.add(3, 4, signature=(None,))
    def test_signature_dtype_type(self):
        # 在 NumPy 1.21 之后，这将是正常的行为
        # 我们已经支持这些类型：
        float_dtype = type(np.dtype(np.float64))
        np.add(3, 4, signature=(float_dtype, float_dtype, None))

    @pytest.mark.parametrize("get_kwarg", [
            lambda dt: dict(dtype=x),
            lambda dt: dict(signature=(x, None, None))])
    def test_signature_dtype_instances_allowed(self, get_kwarg):
        # 当存在明确的单例且给定的类型是等效的时候，允许特定的 dtype 实例；主要是为了向后兼容。
        int64 = np.dtype("int64")
        int64_2 = pickle.loads(pickle.dumps(int64))
        # 依赖于 pickle 的行为，如果断言失败，只需移除测试即可...
        assert int64 is not int64_2

        assert np.add(1, 2, **get_kwarg(int64_2)).dtype == int64
        td = np.timedelta(2, "s")
        assert np.add(td, td, **get_kwarg("m8")).dtype == "m8[s]"

    @pytest.mark.parametrize("get_kwarg", [
            param(lambda x: dict(dtype=x), id="dtype"),
            param(lambda x: dict(signature=(x, None, None)), id="signature")])
    def test_signature_dtype_instances_allowed(self, get_kwarg):
        msg = "The `dtype` and `signature` arguments to ufuncs"

        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg(np.dtype("int64").newbyteorder()))
        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg(np.dtype("m8[ns]")))
        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg("m8[ns]"))

    @pytest.mark.parametrize("casting", ["unsafe", "same_kind", "safe"])
    def test_partial_signature_mismatch(self, casting):
        # 如果第二个参数已经匹配，就不需要指定它：
        res = np.ldexp(np.float32(1.), np.int_(2), dtype="d")
        assert res.dtype == "d"
        res = np.ldexp(np.float32(1.), np.int_(2), signature=(None, None, "d"))
        assert res.dtype == "d"

        # ldexp 只对长整型输入有一个循环，覆盖输出不能帮助解决这个问题（无论转换如何）
        with pytest.raises(TypeError):
            np.ldexp(1., np.uint64(3), dtype="d")
        with pytest.raises(TypeError):
            np.ldexp(1., np.uint64(3), signature=(None, None, "d"))

    def test_partial_signature_mismatch_with_cache(self):
        with pytest.raises(TypeError):
            np.add(np.float16(1), np.uint64(2), sig=("e", "d", None))
        # 确保 e,d->None 在分派缓存中（双重循环）
        np.add(np.float16(1), np.float64(2))
        # 错误仍然必须被引发：
        with pytest.raises(TypeError):
            np.add(np.float16(1), np.uint64(2), sig=("e", "d", None))

    @pytest.mark.xfail(np._get_promotion_state() != "legacy",
            reason="NEP 50 impl breaks casting checks when `dtype=` is used "
                   "together with python scalars.")
    def test_use_output_signature_for_all_arguments(self):
        # 测试当仅提供 `dtype=` 或 `signature=(None, None, dtype)` 时，是否可以正确回退到同质签名。
        # 在这种情况下，选择 `intp, intp -> intp` 的循环。
        res = np.power(1.5, 2.8, dtype=np.intp, casting="unsafe")
        assert res == 1  # 先发生类型转换。
        res = np.power(1.5, 2.8, signature=(None, None, np.intp),
                       casting="unsafe")
        assert res == 1
        with pytest.raises(TypeError):
            # 通常会因为不安全的类型转换而引发错误：
            np.power(1.5, 2.8, dtype=np.intp)

    def test_signature_errors(self):
        with pytest.raises(TypeError,
                    match="the signature object to ufunc must be a string or"):
            np.add(3, 4, signature=123.)  # 不是字符串或元组

        with pytest.raises(ValueError):
            # 包含无法转换为数据类型的不良符号
            np.add(3, 4, signature="%^->#")

        with pytest.raises(ValueError):
            np.add(3, 4, signature=b"ii-i")  # 不完整且是字节字符串

        with pytest.raises(ValueError):
            np.add(3, 4, signature="ii>i")  # 不完整的字符串

        with pytest.raises(ValueError):
            np.add(3, 4, signature=(None, "f8"))  # 长度不正确

        with pytest.raises(UnicodeDecodeError):
            np.add(3, 4, signature=b"\xff\xff->i")

    def test_forced_dtype_times(self):
        # 签名仅设置类型编号（而不是实际的循环数据类型），因此在签名/dtype 中使用 `M` 通常是有效的：
        a = np.array(['2010-01-02', '1999-03-14', '1833-03'], dtype='>M8[D]')
        np.maximum(a, a, dtype="M")
        np.maximum.reduce(a, dtype="M")

        arr = np.arange(10, dtype="m8[s]")
        np.add(arr, arr, dtype="m")
        np.maximum(arr, arr, dtype="m")

    @pytest.mark.parametrize("ufunc", [np.add, np.sqrt])
    # 定义一个测试函数，用于测试安全的类型转换
    def test_cast_safety(self, ufunc):
        """Basic test for the safest casts, because ufuncs inner loops can
        indicate a cast-safety as well (which is normally always "no").
        """
        # 定义一个内部函数，调用给定的ufunc并返回结果
        def call_ufunc(arr, **kwargs):
            return ufunc(*(arr,) * ufunc.nin, **kwargs)

        # 创建一个包含三个浮点数的numpy数组，dtype为np.float32
        arr = np.array([1., 2., 3.], dtype=np.float32)
        # 将数组的字节顺序转换为与当前系统一致的顺序，并保存为新数组
        arr_bs = arr.astype(arr.dtype.newbyteorder())
        # 调用ufunc处理原始数组，并将结果保存为期望值
        expected = call_ufunc(arr)
        # 使用casting="no"参数调用ufunc，验证结果与期望值相等
        res = call_ufunc(arr, casting="no")
        assert_array_equal(expected, res)
        # 使用"no"参数调用ufunc，预期会引发TypeError异常，因为不允许字节交换
        with pytest.raises(TypeError):
            call_ufunc(arr_bs, casting="no")

        # 使用"equiv"参数调用ufunc，验证结果与期望值相等
        res = call_ufunc(arr_bs, casting="equiv")
        assert_array_equal(expected, res)

        # 使用"equiv"参数调用ufunc，预期会引发TypeError异常，因为向np.float64类型转换不是等效的
        with pytest.raises(TypeError):
            call_ufunc(arr_bs, dtype=np.float64, casting="equiv")

        # 使用"safe"参数调用ufunc，验证结果与期望值相等，这种类型转换是安全的
        res = call_ufunc(arr_bs, dtype=np.float64, casting="safe")
        expected = call_ufunc(arr.astype(np.float64))  # 向上转型为np.float64
        assert_array_equal(expected, res)

    # 测试和验证数组求和的稳定性
    def test_sum_stability(self):
        # 创建一个包含500个元素的全为1的numpy数组，dtype为np.float32
        a = np.ones(500, dtype=np.float32)
        # 验证数组元素除以10后的总和减去数组大小除以10的结果约等于0
        assert_almost_equal((a / 10.).sum() - a.size / 10., 0, 4)

        # 创建一个包含500个元素的全为1的numpy数组，dtype为np.float64
        a = np.ones(500, dtype=np.float64)
        # 验证数组元素除以10后的总和减去数组大小除以10的结果约等于0，精度为13位小数
        assert_almost_equal((a / 10.).sum() - a.size / 10., 0, 13)

    # 使用pytest.mark.skipif装饰器标记的测试函数，当条件IS_WASM为True时跳过测试
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_sum(self):
        # 遍历不同的数据类型和值，测试数组求和的行为和警告
        for dt in (int, np.float16, np.float32, np.float64, np.longdouble):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127,
                      128, 1024, 1235):
                # 捕获运行时警告信息
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", RuntimeWarning)

                    # 计算预期的目标值tgt，如果tgt不是有限的则标记为溢出
                    tgt = dt(v * (v + 1) / 2)
                    overflow = not np.isfinite(tgt)
                    # 验证运行时警告的数量是否为1，如果溢出则为1，否则为0
                    assert_equal(len(w), 1 * overflow)

                    # 创建一个包含1到v的整数数组，数据类型为dt
                    d = np.arange(1, v + 1, dtype=dt)

                    # 验证数组的总和是否接近预期的目标值tgt
                    assert_almost_equal(np.sum(d), tgt)
                    # 验证运行时警告的数量是否为2，如果溢出则为2，否则为0
                    assert_equal(len(w), 2 * overflow)

                    # 验证数组反向总和是否接近预期的目标值tgt
                    assert_almost_equal(np.sum(d[::-1]), tgt)
                    # 验证运行时警告的数量是否为3，如果溢出则为3，否则为0
                    assert_equal(len(w), 3 * overflow)

            # 创建一个包含500个元素的全为1的numpy数组，数据类型为dt
            d = np.ones(500, dtype=dt)
            # 验证数组偶数索引位置的元素之和接近250
            assert_almost_equal(np.sum(d[::2]), 250.)
            # 验证数组奇数索引位置的元素之和接近250
            assert_almost_equal(np.sum(d[1::2]), 250.)
            # 验证数组以3为步长的元素之和接近167
            assert_almost_equal(np.sum(d[::3]), 167.)
            # 验证数组以3为步长且从索引1开始的元素之和接近167
            assert_almost_equal(np.sum(d[1::3]), 167.)
            # 验证数组反向偶数索引位置的元素之和接近250
            assert_almost_equal(np.sum(d[::-2]), 250.)
            # 验证数组反向奇数索引位置的元素之和接近250
            assert_almost_equal(np.sum(d[-1::-2]), 250.)
            # 验证数组反向以3为步长的元素之和接近167
            assert_almost_equal(np.sum(d[::-3]), 167.)
            # 验证数组反向以3为步长且从索引-1开始的元素之和接近167
            assert_almost_equal(np.sum(d[-1::-3]), 167.)
            # 验证数组第一个元素增加后的值为2
            d = np.ones((1,), dtype=dt)
            d += d
            assert_almost_equal(d, 2.)
    # 测试复数求和函数
    def test_sum_complex(self):
        # 遍历不同复数数据类型
        for dt in (np.complex64, np.complex128, np.clongdouble):
            # 遍历不同的整数值
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127,
                      128, 1024, 1235):
                # 计算目标复数值
                tgt = dt(v * (v + 1) / 2) - dt((v * (v + 1) / 2) * 1j)
                # 创建长度为 v 的空数组，使用指定的复数数据类型
                d = np.empty(v, dtype=dt)
                # 设置数组的实部为 1 到 v
                d.real = np.arange(1, v + 1)
                # 设置数组的虚部为 -1 到 -v
                d.imag = -np.arange(1, v + 1)
                # 断言数组 d 的总和近似等于目标值 tgt
                assert_almost_equal(np.sum(d), tgt)
                # 断言数组 d 反向排列后的总和近似等于目标值 tgt
                assert_almost_equal(np.sum(d[::-1]), tgt)

            # 创建长度为 500 的全 1 复数数组
            d = np.ones(500, dtype=dt) + 1j
            # 断言数组 d 偶数索引位置元素的总和近似等于 250 + 250j
            assert_almost_equal(np.sum(d[::2]), 250. + 250j)
            # 断言数组 d 奇数索引位置元素的总和近似等于 250 + 250j
            assert_almost_equal(np.sum(d[1::2]), 250. + 250j)
            # 断言数组 d 以 3 为步长的元素的总和近似等于 167 + 167j
            assert_almost_equal(np.sum(d[::3]), 167. + 167j)
            # 断言数组 d 以 3 为步长的奇数索引位置元素的总和近似等于 167 + 167j
            assert_almost_equal(np.sum(d[1::3]), 167. + 167j)
            # 断言数组 d 反向以 2 为步长的元素的总和近似等于 250 + 250j
            assert_almost_equal(np.sum(d[::-2]), 250. + 250j)
            # 断言数组 d 反向以 2 为步长的奇数索引位置元素的总和近似等于 250 + 250j
            assert_almost_equal(np.sum(d[-1::-2]), 250. + 250j)
            # 断言数组 d 反向以 3 为步长的元素的总和近似等于 167 + 167j
            assert_almost_equal(np.sum(d[::-3]), 167. + 167j)
            # 断言数组 d 反向以 3 为步长的奇数索引位置元素的总和近似等于 167 + 167j
            assert_almost_equal(np.sum(d[-1::-3]), 167. + 167j)
            # 断言只有一个元素的数组 d 的和为其值的两倍
            # 注意：这里的数组 d 是复数数组，加法会使每个元素实部和虚部都加倍
            d = np.ones((1,), dtype=dt) + 1j
            d += d
            assert_almost_equal(d, 2. + 2j)

    # 测试带初始值的求和函数
    def test_sum_initial(self):
        # 整数数组，单轴求和，初始值为 2
        assert_equal(np.sum([3], initial=2), 5)

        # 浮点数数组，求和，初始值为 0.1
        assert_almost_equal(np.sum([0.2], initial=0.1), 0.3)

        # 多轴非相邻轴数组求和，初始值为 2
        assert_equal(np.sum(np.ones((2, 3, 5), dtype=np.int64), axis=(0, 2), initial=2),
                     [12, 12, 12])

    # 测试带条件的求和函数
    def test_sum_where(self):
        # 在指定条件下，对二维数组进行求和，不考虑第二行
        assert_equal(np.sum([[1., 2.], [3., 4.]], where=[True, False]), 4.)
        # 在指定条件下，对二维数组的列进行求和，初始值为 5.0
        assert_equal(np.sum([[1., 2.], [3., 4.]], axis=0, initial=5.,
                            where=[True, False]), [9., 5.])

    # 测试向量点积函数
    def test_vecdot(self):
        # 创建两个二维数组
        arr1 = np.arange(6).reshape((2, 3))
        arr2 = np.arange(3).reshape((1, 3))

        # 计算 arr1 和 arr2 的向量点积，期望得到 [5, 14]
        actual = np.vecdot(arr1, arr2)
        expected = np.array([5, 14])
        assert_array_equal(actual, expected)

        # 在指定轴上计算 arr1.T 和 arr2.T 的向量点积，期望得到 [5, 14]
        actual2 = np.vecdot(arr1.T, arr2.T, axis=-2)
        assert_array_equal(actual2, expected)

        # 将 arr1 和 arr2 转换为对象数组后，计算向量点积，期望得到 [10-4j]
        actual3 = np.vecdot(arr1.astype("object"), arr2)
        assert_array_equal(actual3, expected.astype("object"))

    # 测试复数向量点积函数
    def test_vecdot_complex(self):
        # 创建复数数组 arr1 和 arr2
        arr1 = np.array([1, 2j, 3])
        arr2 = np.array([1, 2, 3])

        # 计算 arr1 和 arr2 的复数向量点积，期望得到 [10-4j]
        actual = np.vecdot(arr1, arr2)
        expected = np.array([10-4j])
        assert_array_equal(actual, expected)

        # 在指定轴上计算 arr2 和 arr1 的复数向量点积的共轭，期望与上面结果相同
        actual2 = np.vecdot(arr2, arr1)
        assert_array_equal(actual2, expected.conj())

        # 将 arr1 和 arr2 转换为对象数组后，计算复数向量点积，期望得到 [10-4j]
        actual3 = np.vecdot(arr1.astype("object"), arr2.astype("object"))
        assert_array_equal(actual3, expected.astype("object"))
    # 定义一个测试用例，验证自定义子类继承自 np.ndarray 的行为
    def test_vecdot_subclass(self):
        # 定义一个简单的子类，继承自 np.ndarray
        class MySubclass(np.ndarray):
            pass

        # 创建一个 MySubclass 的实例 arr1，形状为 (2, 3)，并视图表示
        arr1 = np.arange(6).reshape((2, 3)).view(MySubclass)
        # 创建另一个 MySubclass 的实例 arr2，形状为 (1, 3)，并视图表示
        arr2 = np.arange(3).reshape((1, 3)).view(MySubclass)
        # 调用 np.vecdot 计算 arr1 和 arr2 的点积
        result = np.vecdot(arr1, arr2)
        # 断言 result 是 MySubclass 的实例
        assert isinstance(result, MySubclass)

    # 定义一个测试用例，验证处理包含对象类型数组的情况下，vecdot 是否会引发期望的异常
    def test_vecdot_object_no_conjugate(self):
        # 创建一个对象类型的数组 arr，包含字符串元素 "1" 和 "2"
        arr = np.array(["1", "2"], dtype=object)
        # 使用 pytest 检查调用 np.vecdot(arr, arr) 是否会引发 AttributeError 异常，并且异常信息匹配 "conjugate"
        with pytest.raises(AttributeError, match="conjugate"):
            np.vecdot(arr, arr)

    # 定义一个测试用例，验证处理对象类型数组时，vecdot 在遇到特定类型错误时是否能够正确停止外部循环
    def test_vecdot_object_breaks_outer_loop_on_error(self):
        # 创建一个形状为 (3, 3) 的对象类型数组 arr1，所有元素为 1
        arr1 = np.ones((3, 3)).astype(object)
        # 创建 arr1 的副本 arr2
        arr2 = arr1.copy()
        # 修改 arr2 的某个元素为 None
        arr2[1, 1] = None
        # 创建一个形状为 (3,) 的对象类型数组 out，所有元素为 0
        out = np.zeros(3).astype(object)
        # 使用 pytest 检查调用 np.vecdot(arr1, arr2, out=out) 是否会引发 TypeError 异常，并且异常信息匹配指定的正则表达式
        with pytest.raises(TypeError, match=r"\*: 'float' and 'NoneType'"):
            np.vecdot(arr1, arr2, out=out)
        # 断言 out 的前两个元素为 3，第三个元素为 0
        assert out[0] == 3
        assert out[1] == out[2] == 0

    # 定义一个测试用例，验证广播操作在 vecdot 中的行为是否符合预期
    def test_broadcast(self):
        # 设置错误消息
        msg = "broadcast"
        # 创建形状为 (2, 1, 2) 的数组 a 和形状为 (1, 2, 2) 的数组 b
        a = np.arange(4).reshape((2, 1, 2))
        b = np.arange(4).reshape((1, 2, 2))
        # 使用 assert_array_equal 检查 np.vecdot(a, b) 和 np.sum(a*b, axis=-1) 的结果是否相等，并提供错误消息
        assert_array_equal(np.vecdot(a, b), np.sum(a*b, axis=-1), err_msg=msg)
        
        # 更新错误消息
        msg = "extend & broadcast loop dimensions"
        # 重新定义数组 b 的形状为 (2, 2)
        b = np.arange(4).reshape((2, 2))
        # 再次使用 assert_array_equal 检查 np.vecdot(a, b) 和 np.sum(a*b, axis=-1) 的结果是否相等，并提供错误消息
        assert_array_equal(np.vecdot(a, b), np.sum(a*b, axis=-1), err_msg=msg)
        
        # 检查在核心维度上的广播操作是否会失败
        a = np.arange(8).reshape((4, 2))
        b = np.arange(4).reshape((4, 1))
        # 使用 assert_raises 检查调用 np.vecdot(a, b) 是否会引发 ValueError 异常
        assert_raises(ValueError, np.vecdot, a, b)
        
        # 检查在核心维度上的扩展操作是否会失败
        a = np.arange(8).reshape((4, 2))
        b = np.array(7)
        # 使用 assert_raises 检查调用 np.vecdot(a, b) 是否会引发 ValueError 异常
        assert_raises(ValueError, np.vecdot, a, b)
        
        # 检查广播操作是否会失败
        a = np.arange(2).reshape((2, 1, 1))
        b = np.arange(3).reshape((3, 1, 1))
        # 使用 assert_raises 检查调用 np.vecdot(a, b) 是否会引发 ValueError 异常
        assert_raises(ValueError, np.vecdot, a, b)
        
        # 在重叠广播数组写入时应发出警告，gh-2705
        a = np.arange(2)
        b = np.arange(4).reshape((2, 2))
        # 对数组进行广播，获取广播后的数组 u, v
        u, v = np.broadcast_arrays(a, b)
        # 断言 u 在第一维上的步长为 0
        assert_equal(u.strides[0], 0)
        # 计算 u 和 v 的和 x
        x = u + v
        # 使用 warnings.catch_warnings 检查对 u += v 操作是否会产生警告，并验证警告的数量和条件
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            u += v
            assert_equal(len(w), 1)
            assert_(x[0, 0] != u[0, 0])

        # 检查是否不允许输出降低
        # 参考 gh-15139
        a = np.arange(6).reshape(3, 2)
        b = np.ones(2)
        out = np.empty(())
        # 使用 assert_raises 检查调用 np.vecdot(a, b, out) 是否会引发 ValueError 异常
        assert_raises(ValueError, np.vecdot, a, b, out)
        out2 = np.empty(3)
        # 调用 np.vecdot(a, b, out2) 并保存返回值到 c
        c = np.vecdot(a, b, out2)
        # 断言 c 和 out2 是同一个对象
        assert_(c is out2)
    def test_out_broadcasts(self):
        # 对于 ufuncs 和 gufuncs（不适用于 reductions），我们目前允许输出导致输入数组的广播。
        # 包括在输入数组中维度为 1 的维度和根本不存在的维度。
        
        # 创建一个形状为 (1, 3) 的数组 arr
        arr = np.arange(3).reshape(1, 3)
        # 创建一个形状为 (5, 4, 3) 的空数组 out
        out = np.empty((5, 4, 3))
        # 使用 np.add 函数，将 arr 和 arr 相加，并将结果写入 out
        np.add(arr, arr, out=out)
        # 断言 out 中的值是否全部等于 np.arange(3) * 2
        assert (out == np.arange(3) * 2).all()

        # 对于 gufuncs 也是如此（gh-16484）
        # 使用 np.vecdot 函数，计算 arr 和 arr 的点积，并将结果写入 out
        np.vecdot(arr, arr, out=out)
        # 断言 out 中的值是否全部等于 5
        assert (out == 5).all()

    @pytest.mark.parametrize(["arr", "out"], [
                ([2], np.empty(())),
                ([1, 2], np.empty(1)),
                (np.ones((4, 3)), np.empty((4, 1)))],
            ids=["(1,)->()", "(2,)->(1,)", "(4, 3)->(4, 1)"])
    def test_out_broadcast_errors(self, arr, out):
        # 输出（当前）允许广播输入，但不能比实际结果更小。
        
        # 使用 pytest.raises 检查是否会抛出 ValueError 异常，匹配错误信息 "non-broadcastable"
        with pytest.raises(ValueError, match="non-broadcastable"):
            np.positive(arr, out=out)

        # 使用 pytest.raises 检查是否会抛出 ValueError 异常，匹配错误信息 "non-broadcastable"
        with pytest.raises(ValueError, match="non-broadcastable"):
            np.add(np.ones(()), arr, out=out)

    def test_type_cast(self):
        # 类型转换测试
        msg = "type cast"
        # 创建一个形状为 (2, 3) 的 short 类型数组 a
        a = np.arange(6, dtype='short').reshape((2, 3))
        # 断言 np.vecdot(a, a) 的结果与 np.sum(a*a, axis=-1) 相等
        assert_array_equal(np.vecdot(a, a), np.sum(a*a, axis=-1),
                           err_msg=msg)
        
        msg = "type cast on one argument"
        # 创建一个形状为 (2, 3) 的数组 a
        a = np.arange(6).reshape((2, 3))
        b = a + 0.1
        # 断言 np.vecdot(a, b) 的结果与 np.sum(a*b, axis=-1) 相等
        assert_array_almost_equal(np.vecdot(a, b), np.sum(a*b, axis=-1),
                                  err_msg=msg)

    def test_endian(self):
        # 字节序测试
        msg = "big endian"
        # 创建一个形状为 (2, 3)、大端序的数组 a
        a = np.arange(6, dtype='>i4').reshape((2, 3))
        # 断言 np.vecdot(a, a) 的结果与 np.sum(a*a, axis=-1) 相等
        assert_array_equal(np.vecdot(a, a), np.sum(a*a, axis=-1),
                           err_msg=msg)
        
        msg = "little endian"
        # 创建一个形状为 (2, 3)、小端序的数组 a
        a = np.arange(6, dtype='<i4').reshape((2, 3))
        # 断言 np.vecdot(a, a) 的结果与 np.sum(a*a, axis=-1) 相等
        assert_array_equal(np.vecdot(a, a), np.sum(a*a, axis=-1),
                           err_msg=msg)

        # 输出应始终是本地字节顺序
        Ba = np.arange(1, dtype='>f8')
        La = np.arange(1, dtype='<f8')
        # 断言 (Ba+Ba).dtype 的结果与 np.dtype('f8') 相等
        assert_equal((Ba+Ba).dtype, np.dtype('f8'))
        # 断言 (Ba+La).dtype 的结果与 np.dtype('f8') 相等
        assert_equal((Ba+La).dtype, np.dtype('f8'))
        # 断言 (La+Ba).dtype 的结果与 np.dtype('f8') 相等
        assert_equal((La+Ba).dtype, np.dtype('f8'))
        # 断言 (La+La).dtype 的结果与 np.dtype('f8') 相等
        assert_equal((La+La).dtype, np.dtype('f8'))

        # 断言 np.absolute(La).dtype 的结果与 np.dtype('f8') 相等
        assert_equal(np.absolute(La).dtype, np.dtype('f8'))
        # 断言 np.absolute(Ba).dtype 的结果与 np.dtype('f8') 相等
        assert_equal(np.absolute(Ba).dtype, np.dtype('f8'))
        # 断言 np.negative(La).dtype 的结果与 np.dtype('f8') 相等
        assert_equal(np.negative(La).dtype, np.dtype('f8'))
        # 断言 np.negative(Ba).dtype 的结果与 np.dtype('f8') 相等
        assert_equal(np.negative(Ba).dtype, np.dtype('f8'))
    # 测试不连续的数组内存布局
    def test_incontiguous_array(self):
        # 定义测试消息
        msg = "incontiguous memory layout of array"
        # 创建一个 6 维的数组，范围是 0 到 63，然后重塑为特定形状
        x = np.arange(64).reshape((2, 2, 2, 2, 2, 2))
        # 从 x 中选择特定的切片 a，选取每个轴的特定索引
        a = x[:, 0, :, 0, :, 0]
        # 从 x 中选择特定的切片 b，选取每个轴的特定索引
        b = x[:, 1, :, 1, :, 1]
        # 修改 a 中的元素
        a[0, 0, 0] = -1
        # 定义第二个测试消息
        msg2 = "make sure it references to the original array"
        # 使用断言确保修改反映在原始数组中
        assert_equal(x[0, 0, 0, 0, 0, 0], -1, err_msg=msg2)
        # 使用 np.vecdot 函数计算 a 和 b 的向量点积，并与用 np.sum 计算的结果进行比较
        assert_array_equal(np.vecdot(a, b), np.sum(a*b, axis=-1), err_msg=msg)
        
        # 重新分配 x，并修改其内容
        x = np.arange(24).reshape(2, 3, 4)
        # a 和 b 是 x 的转置
        a = x.T
        b = x.T
        # 修改 a 中的元素
        a[0, 0, 0] = -1
        # 确保修改反映在原始数组中
        assert_equal(x[0, 0, 0], -1, err_msg=msg2)
        # 再次使用 np.vecdot 函数计算 a 和 b 的向量点积，并与用 np.sum 计算的结果进行比较
        assert_array_equal(np.vecdot(a, b), np.sum(a*b, axis=-1), err_msg=msg)

    # 测试输出参数
    def test_output_argument(self):
        # 定义测试消息
        msg = "output argument"
        # 创建两个数组 a 和 b
        a = np.arange(12).reshape((2, 3, 2))
        b = np.arange(4).reshape((2, 1, 2)) + 1
        # 创建一个全零数组 c
        c = np.zeros((2, 3), dtype='int')
        # 使用 np.vecdot 计算 a 和 b 的向量点积，结果存储在 c 中
        np.vecdot(a, b, c)
        # 使用断言确保 c 等于用 np.sum 计算的结果
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)
        # 将 c 中的所有元素设置为 -1
        c[:] = -1
        # 再次使用 np.vecdot 计算 a 和 b 的向量点积，结果存储在 c 中
        np.vecdot(a, b, out=c)
        # 使用断言确保 c 等于用 np.sum 计算的结果
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)

        # 更换测试消息
        msg = "output argument with type cast"
        # 创建一个全零数组 c，数据类型为 int16
        c = np.zeros((2, 3), dtype='int16')
        # 使用 np.vecdot 计算 a 和 b 的向量点积，结果存储在 c 中
        np.vecdot(a, b, c)
        # 使用断言确保 c 等于用 np.sum 计算的结果
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)
        # 将 c 中的所有元素设置为 -1
        c[:] = -1
        # 再次使用 np.vecdot 计算 a 和 b 的向量点积，结果存储在 c 中
        np.vecdot(a, b, out=c)
        # 使用断言确保 c 等于用 np.sum 计算的结果
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)

        # 更换测试消息
        msg = "output argument with incontiguous layout"
        # 创建一个全零数组 c，形状为 (2, 3, 4)，数据类型为 int16
        c = np.zeros((2, 3, 4), dtype='int16')
        # 使用 np.vecdot 计算 a 和 b 的向量点积，结果存储在 c 的第一个轴上
        np.vecdot(a, b, c[..., 0])
        # 使用断言确保 c 的第一个轴等于用 np.sum 计算的结果
        assert_array_equal(c[..., 0], np.sum(a*b, axis=-1), err_msg=msg)
        # 将 c 中的所有元素设置为 -1
        c[:] = -1
        # 再次使用 np.vecdot 计算 a 和 b 的向量点积，结果存储在 c 的第一个轴上
        np.vecdot(a, b, out=c[..., 0])
        # 使用断言确保 c 的第一个轴等于用 np.sum 计算的结果
        assert_array_equal(c[..., 0], np.sum(a*b, axis=-1), err_msg=msg)
    # 定义测试方法，验证 np.vecdot 方法的 axis 参数
    def test_axis_argument(self):
        # vecdot 的签名为 '(n),(n)->()'
        # 创建一个 3x3x3 的数组 a，其中元素为 0 到 26
        a = np.arange(27.).reshape((3, 3, 3))
        # 创建一个 3x1x3 的数组 b，其中元素为 10 到 18
        b = np.arange(10., 19.).reshape((3, 1, 3))
        # 使用 np.vecdot 计算 a 和 b 的点乘结果赋值给 c
        c = np.vecdot(a, b)
        # 验证 c 是否等于 (a * b) 在最后一个维度上求和的结果
        assert_array_equal(c, (a * b).sum(-1))
        # 再次使用 np.vecdot 计算 a 和 b 的点乘结果，并指定 axis=-1
        c = np.vecdot(a, b, axis=-1)
        # 验证 c 是否等于 (a * b) 在最后一个维度上求和的结果
        assert_array_equal(c, (a * b).sum(-1))
        # 创建一个和 c 同样形状的零数组 out
        out = np.zeros_like(c)
        # 使用 np.vecdot 计算 a 和 b 的点乘结果，并指定 axis=-1 和 out=out
        d = np.vecdot(a, b, axis=-1, out=out)
        # 验证 d 和 out 是同一个数组对象
        assert_(d is out)
        # 验证 d 是否等于之前计算得到的 c
        assert_array_equal(d, c)
        # 使用 np.vecdot 计算 a 和 b 的点乘结果，并指定 axis=0
        c = np.vecdot(a, b, axis=0)
        # 验证 c 是否等于 (a * b) 在第一个维度上求和的结果
        assert_array_equal(c, (a * b).sum(0))
        
        # 对 innerwt 和 cumsum 进行基本的验证
        # 创建一个 2x3 的数组 a，元素为 0 到 5
        a = np.arange(6).reshape((2, 3))
        # 创建一个 2x3 的数组 b，元素为 10 到 15
        b = np.arange(10, 16).reshape((2, 3))
        # 创建一个 2x3 的数组 w，元素为 20 到 25
        w = np.arange(20, 26).reshape((2, 3))
        # 验证 umt.innerwt 的结果是否等于按照给定轴求和的结果
        assert_array_equal(umt.innerwt(a, b, w, axis=0),
                           np.sum(a * b * w, axis=0))
        # 验证 umt.cumsum 在 axis=0 上的结果是否等于 np.cumsum 的结果
        assert_array_equal(umt.cumsum(a, axis=0), np.cumsum(a, axis=0))
        # 验证 umt.cumsum 在 axis=-1 上的结果是否等于 np.cumsum 的结果
        assert_array_equal(umt.cumsum(a, axis=-1), np.cumsum(a, axis=-1))
        # 创建一个和 a 相同形状的空数组 out
        out = np.empty_like(a)
        # 使用 umt.cumsum 在 axis=0 上计算 a 的累加和，并将结果存储在 out 中
        b = umt.cumsum(a, out=out, axis=0)
        # 验证 out 和 b 是同一个数组对象
        assert_(out is b)
        # 验证 b 是否等于 np.cumsum(a, axis=0) 的结果
        assert_array_equal(b, np.cumsum(a, axis=0))
        # 使用 umt.cumsum 在 axis=1 上计算 a 的累加和，并将结果存储在 out 中
        b = umt.cumsum(a, out=out, axis=1)
        # 验证 out 和 b 是同一个数组对象
        assert_(out is b)
        # 验证 b 是否等于 np.cumsum(a, axis=-1) 的结果
        assert_array_equal(b, np.cumsum(a, axis=-1))
        
        # 检查错误情况
        # 不能同时传递 axis 和 axes 参数
        assert_raises(TypeError, np.vecdot, a, b, axis=0, axes=[0, 0])
        # axis 参数必须是整数
        assert_raises(TypeError, np.vecdot, a, b, axis=[0])
        # umt.matrix_multiply 不支持多于一个核心维度的操作
        mm = umt.matrix_multiply
        assert_raises(TypeError, mm, a, b, axis=1)
        # out 参数在指定的 axis 维度上大小不一致
        out = np.empty((1, 2, 3), dtype=a.dtype)
        assert_raises(ValueError, umt.cumsum, a, out=out, axis=0)
        # 普通的 ufunc 不应该接受 axis 参数
        assert_raises(TypeError, np.add, 1., 1., axis=0)

    # 定义测试方法，验证 umt.innerwt 方法的功能
    def test_innerwt(self):
        # 创建一个 2x3 的数组 a，元素为 0 到 5
        a = np.arange(6).reshape((2, 3))
        # 创建一个 2x3 的数组 b，元素为 10 到 15
        b = np.arange(10, 16).reshape((2, 3))
        # 创建一个 2x3 的数组 w，元素为 20 到 25
        w = np.arange(20, 26).reshape((2, 3))
        # 验证 umt.innerwt 的结果是否等于按照给定轴求和的结果
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a*b*w, axis=-1))
        # 创建一个 2x3x4 的数组 a，元素为 100 到 123
        a = np.arange(100, 124).reshape((2, 3, 4))
        # 创建一个 2x3x4 的数组 b，元素为 200 到 223
        b = np.arange(200, 224).reshape((2, 3, 4))
        # 创建一个 2x3x4 的数组 w，元素为 300 到 323
        w = np.arange(300, 324).reshape((2, 3, 4))
        # 验证 umt.innerwt 的结果是否等于按照给定轴求和的结果
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a*b*w, axis=-1))

    # 定义测试方法，验证 umt.innerwt 方法处理空数组的情况
    def test_innerwt_empty(self):
        """Test generalized ufunc with zero-sized operands"""
        # 创建空的浮点类型数组 a, b, w
        a = np.array([], dtype='f8')
        b = np.array([], dtype='f8')
        w = np.array([], dtype='f8')
        # 验证 umt.innerwt 在处理空数组时的结果是否正确
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a*b*w, axis=-1))
    # 定义一个测试方法，用于测试一维交叉乘积函数 `cross1d` 的各种情况
    def test_cross1d(self):
        """Test with fixed-sized signature."""
        # 创建一个 3x3 的单位矩阵 `a`
        a = np.eye(3)
        # 断言 `cross1d(a, a)` 的结果与 3x3 全零矩阵相等
        assert_array_equal(umt.cross1d(a, a), np.zeros((3, 3)))
        # 创建一个全零的 3x3 矩阵 `out`
        out = np.zeros((3, 3))
        # 调用 `cross1d` 函数，将结果存储到 `out` 中，断言返回的结果对象与 `out` 相同
        result = umt.cross1d(a[0], a, out)
        assert_(result is out)
        # 断言 `cross1d(a[0], a, out)` 的结果与预期结果相等，其中预期结果是竖直堆叠的 `[0, 0, 0]`、`a[2]` 和 `-a[1]`
        assert_array_equal(result, np.vstack((np.zeros(3), a[2], -a[1])))
        # 断言调用 `cross1d` 函数时，传递 4x4 的单位矩阵会抛出 ValueError 异常
        assert_raises(ValueError, umt.cross1d, np.eye(4), np.eye(4))
        # 断言调用 `cross1d` 函数时，传递 `a` 和长度为 4 的浮点数组会抛出 ValueError 异常
        assert_raises(ValueError, umt.cross1d, a, np.arange(4.))
        # 断言调用 `cross1d` 函数时，传递 `a`、长度为 3 的浮点数组和一个形状为 (3, 4) 的全零矩阵会抛出 ValueError 异常
        assert_raises(ValueError, umt.cross1d, a, np.arange(3.), np.zeros((3, 4)))
        # 断言调用 `cross1d` 函数时，传递 `a`、长度为 3 的浮点数组和一个长度为 3 的全零数组会抛出 ValueError 异常
        assert_raises(ValueError, umt.cross1d, a, np.arange(3.), np.zeros(3))

    # 定义一个测试矩阵乘法函数 `matrix_multiply` 的方法
    def test_matrix_multiply(self):
        self.compare_matrix_multiply_results(np.int64)
        self.compare_matrix_multiply_results(np.double)

    # 定义一个测试矩阵乘法函数 `matrix_multiply` 的特殊情况方法，用于处理空矩阵
    def test_matrix_multiply_umath_empty(self):
        # 调用 `matrix_multiply` 函数，传递两个空矩阵，断言结果与预期的全零矩阵相等
        res = umt.matrix_multiply(np.ones((0, 10)), np.ones((10, 0)))
        assert_array_equal(res, np.zeros((0, 0)))
        # 调用 `matrix_multiply` 函数，传递两个空矩阵（维度不同），断言结果与预期的 10x10 全零矩阵相等
        res = umt.matrix_multiply(np.ones((10, 0)), np.ones((0, 10)))
        assert_array_equal(res, np.zeros((10, 10)))
    # 定义一个方法，用于比较矩阵乘法的结果
    def compare_matrix_multiply_results(self, tp):
        # 创建两个随机生成的矩阵，形状为 (2, 3, 4)，数据类型为 tp
        d1 = np.array(np.random.rand(2, 3, 4), dtype=tp)
        d2 = np.array(np.random.rand(2, 3, 4), dtype=tp)
        # 生成一条消息，描述矩阵乘法的数据类型
        msg = "matrix multiply on type %s" % d1.dtype.name

        # 定义一个函数，用于生成长度为 n 的全排列
        def permute_n(n):
            if n == 1:
                return ([0],)
            ret = ()
            base = permute_n(n-1)
            for perm in base:
                for i in range(n):
                    new = perm + [n-1]
                    new[n-1] = new[i]
                    new[i] = n-1
                    ret += (new,)
            return ret

        # 定义一个函数，生成对 n 维数组的切片方式
        def slice_n(n):
            if n == 0:
                return ((),)
            ret = ()
            base = slice_n(n-1)
            for sl in base:
                ret += (sl+(slice(None),),)
                ret += (sl+(slice(0, 1),),)
            return ret

        # 定义一个函数，判断两个维度是否可广播
        def broadcastable(s1, s2):
            return s1 == s2 or s1 == 1 or s2 == 1

        # 生成三维数组的所有排列方式
        permute_3 = permute_n(3)
        # 生成三维数组的所有切片方式，并包括反向切片
        slice_3 = slice_n(3) + ((slice(None, None, -1),)*3,)

        # 初始化一个参考值
        ref = True
        # 对所有可能的排列和切片方式进行循环
        for p1 in permute_3:
            for p2 in permute_3:
                for s1 in slice_3:
                    for s2 in slice_3:
                        # 根据排列和切片方式对数组进行转置和切片操作
                        a1 = d1.transpose(p1)[s1]
                        a2 = d2.transpose(p2)[s2]
                        # 检查数组是否有基础数据，即非空
                        ref = ref and a1.base is not None
                        ref = ref and a2.base is not None
                        # 如果满足条件，断言矩阵乘法的结果与期望值相近
                        if (a1.shape[-1] == a2.shape[-2] and
                                broadcastable(a1.shape[0], a2.shape[0])):
                            assert_array_almost_equal(
                                umt.matrix_multiply(a1, a2),
                                np.sum(a2[..., np.newaxis].swapaxes(-3, -1) *
                                       a1[..., np.newaxis,:], axis=-1),
                                err_msg=msg + ' %s %s' % (str(a1.shape),
                                                          str(a2.shape)))

        # 最终断言参考值为 True，确保所有条件下的基础数据都非空
        assert_equal(ref, True, err_msg="reference check")

    # 定义一个测试方法，测试欧几里得距离的计算
    def test_euclidean_pdist(self):
        # 创建一个 4x3 的浮点数矩阵
        a = np.arange(12, dtype=float).reshape(4, 3)
        # 创建一个空数组，用于存储欧几里得距离的计算结果
        out = np.empty((a.shape[0] * (a.shape[0] - 1) // 2,), dtype=a.dtype)
        # 调用 umt 中的欧几里得距离计算函数
        umt.euclidean_pdist(a, out)
        # 使用 NumPy 计算每对点之间的欧几里得距离，并去除对角线元素
        b = np.sqrt(np.sum((a[:, None] - a)**2, axis=-1))
        b = b[~np.tri(a.shape[0], dtype=bool)]
        # 断言计算结果与期望值相近
        assert_almost_equal(out, b)
        # 断言调用欧几里得距离计算时需要提供一个输出数组
        assert_raises(ValueError, umt.euclidean_pdist, a)

    # 定义一个测试方法，测试累积求和的功能
    def test_cumsum(self):
        # 创建一个包含 0 到 9 的整数数组
        a = np.arange(10)
        # 使用 umt 中的累积求和函数计算结果
        result = umt.cumsum(a)
        # 断言计算结果与预期相等
        assert_array_equal(result, a.cumsum())
    def test_object_logical(self):
        # 创建包含多种类型元素的对象数组
        a = np.array([3, None, True, False, "test", ""], dtype=object)
        # 使用 logical_or 函数对数组中的每个元素与 None 做逻辑或运算
        assert_equal(np.logical_or(a, None),
                        np.array([x or None for x in a], dtype=object))
        # 使用 logical_or 函数对数组中的每个元素与 True 做逻辑或运算
        assert_equal(np.logical_or(a, True),
                        np.array([x or True for x in a], dtype=object))
        # 使用 logical_or 函数对数组中的每个元素与 12 做逻辑或运算
        assert_equal(np.logical_or(a, 12),
                        np.array([x or 12 for x in a], dtype=object))
        # 使用 logical_or 函数对数组中的每个元素与 "blah" 做逻辑或运算
        assert_equal(np.logical_or(a, "blah"),
                        np.array([x or "blah" for x in a], dtype=object))

        # 使用 logical_and 函数对数组中的每个元素与 None 做逻辑与运算
        assert_equal(np.logical_and(a, None),
                        np.array([x and None for x in a], dtype=object))
        # 使用 logical_and 函数对数组中的每个元素与 True 做逻辑与运算
        assert_equal(np.logical_and(a, True),
                        np.array([x and True for x in a], dtype=object))
        # 使用 logical_and 函数对数组中的每个元素与 12 做逻辑与运算
        assert_equal(np.logical_and(a, 12),
                        np.array([x and 12 for x in a], dtype=object))
        # 使用 logical_and 函数对数组中的每个元素与 "blah" 做逻辑与运算
        assert_equal(np.logical_and(a, "blah"),
                        np.array([x and "blah" for x in a], dtype=object))

        # 使用 logical_not 函数对数组中的每个元素进行逻辑非运算
        assert_equal(np.logical_not(a),
                        np.array([not x for x in a], dtype=object))

        # 对数组中的元素执行 logical_or.reduce 操作，期望结果为 3
        assert_equal(np.logical_or.reduce(a), 3)
        # 对数组中的元素执行 logical_and.reduce 操作，期望结果为 None
        assert_equal(np.logical_and.reduce(a), None)

    def test_object_comparison(self):
        # 定义一个具有比较方法的类
        class HasComparisons:
            def __eq__(self, other):
                return '=='

        # 创建一个零维对象数组，包含 HasComparisons 类的实例
        arr0d = np.array(HasComparisons())
        # 断言两个数组元素相等，期望结果为 True
        assert_equal(arr0d == arr0d, True)
        # 使用 np.equal 函数比较数组元素是否相等，期望结果为 True（正常情况下会进行类型转换）

        assert_equal(np.equal(arr0d, arr0d), True)

        # 创建一个一维对象数组，包含一个 HasComparisons 类的实例
        arr1d = np.array([HasComparisons()])
        # 断言两个数组元素相等，期望结果为包含 True 的数组
        assert_equal(arr1d == arr1d, np.array([True]))
        # 使用 np.equal 函数比较数组元素是否相等，期望结果为包含 True 的数组（正常情况下会进行类型转换）
        assert_equal(np.equal(arr1d, arr1d), np.array([True]))
        # 使用 np.equal 函数比较数组元素是否相等，指定 dtype=object，期望结果为包含 '==' 的数组

        assert_equal(np.equal(arr1d, arr1d, dtype=object), np.array(['==']))

    def test_object_array_reduction(self):
        # 在对象数组上进行归约操作
        a = np.array(['a', 'b', 'c'], dtype=object)
        # 对数组元素进行求和，期望结果为 'abc'
        assert_equal(np.sum(a), 'abc')
        # 获取数组中的最大值，期望结果为 'c'
        assert_equal(np.max(a), 'c')
        # 获取数组中的最小值，期望结果为 'a'
        assert_equal(np.min(a), 'a')
        # 创建一个包含 True、False 和 True 的对象数组
        a = np.array([True, False, True], dtype=object)
        # 对数组元素进行求和，期望结果为 2
        assert_equal(np.sum(a), 2)
        # 对数组元素进行累积乘积，期望结果为 0
        assert_equal(np.prod(a), 0)
        # 检查数组中是否有任意 True，期望结果为 True
        assert_equal(np.any(a), True)
        # 检查数组中是否所有元素都为 True，期望结果为 False
        assert_equal(np.all(a), False)
        # 获取数组中的最大值，期望结果为 True
        assert_equal(np.max(a), True)
        # 获取数组中的最小值，期望结果为 False
        assert_equal(np.min(a), False)
        # 对包含单个元素的对象数组进行求和，期望结果为 1
        assert_equal(np.array([[1]], dtype=object).sum(), 1)
        # 对包含嵌套数组的对象数组进行沿指定轴的求和，期望结果为 [1, 2]
        assert_equal(np.array([[[1, 2]]], dtype=object).sum((0, 1)), [1, 2])
        # 对包含单个元素的对象数组进行求和，初始值为 1，期望结果为 2
        assert_equal(np.array([1], dtype=object).sum(initial=1), 2)
        # 对包含嵌套数组的对象数组进行求和，指定 where 参数，期望结果为 [0, 2, 3]
        assert_equal(np.array([[1], [2, 3]], dtype=object)
                     .sum(initial=[0], where=[False, True]), [0, 2, 3])
    def test_object_array_accumulate_inplace(self):
        # 检查原地累积操作是否有效，参见gh-7402
        # 创建一个包含4个元素的对象数组，每个元素为包含一个整数1的列表
        arr = np.ones(4, dtype=object)
        arr[:] = [[1] for i in range(4)]
        # 对数组arr进行累积求和，并将结果存回arr中
        np.add.accumulate(arr, out=arr)
        # 再次对数组arr进行累积求和，并将结果存回arr中
        np.add.accumulate(arr, out=arr)
        # 断言数组arr是否与预期的对象数组相等
        assert_array_equal(arr,
                           np.array([[1]*i for i in [1, 3, 6, 10]], dtype=object),
                          )

        # 如果使用axis参数，结果应该相同
        # 创建一个形状为(2, 4)的对象数组，每个元素为包含一个整数2的列表
        arr = np.ones((2, 4), dtype=object)
        arr[0, :] = [[2] for i in range(4)]
        # 按指定轴（axis=-1，即最后一个轴）对数组arr进行累积求和
        np.add.accumulate(arr, out=arr, axis=-1)
        # 再次按指定轴对数组arr进行累积求和
        np.add.accumulate(arr, out=arr, axis=-1)
        # 断言数组arr的第一行是否与预期的对象数组相等
        assert_array_equal(arr[0, :],
                           np.array([[2]*i for i in [1, 3, 6, 10]], dtype=object),
                          )

    def test_object_array_accumulate_failure(self):
        # 对对象数组的典型累积操作如预期般有效
        res = np.add.accumulate(np.array([1, 0, 2], dtype=object))
        assert_array_equal(res, np.array([1, 1, 3], dtype=object))
        # 如果内部循环出现错误，错误应该被传播
        with pytest.raises(TypeError):
            np.add.accumulate([1, None, 2])

    def test_object_array_reduceat_inplace(self):
        # 检查原地reduceat操作是否有效，参见gh-7465
        # 创建一个包含4个元素的空对象数组，每个元素为包含一个整数1的列表
        arr = np.empty(4, dtype=object)
        arr[:] = [[1] for i in range(4)]
        # 创建一个形状和数据与arr相同的对象数组out
        out = np.empty(4, dtype=object)
        out[:] = [[1] for i in range(4)]
        # 对数组arr按照给定的索引位置进行reduceat操作，并将结果存回arr中
        np.add.reduceat(arr, np.arange(4), out=arr)
        # 再次对数组arr按照给定的索引位置进行reduceat操作，并将结果存回arr中
        np.add.reduceat(arr, np.arange(4), out=arr)
        # 断言数组arr是否与数组out相等
        assert_array_equal(arr, out)

        # 如果使用axis参数，结果应该相同
        # 创建一个形状为(2, 4)的对象数组，每个元素为包含一个整数2的列表
        arr = np.ones((2, 4), dtype=object)
        arr[0, :] = [[2] for i in range(4)]
        # 创建一个形状和数据与arr相同的对象数组out
        out = np.ones((2, 4), dtype=object)
        out[0, :] = [[2] for i in range(4)]
        # 按指定轴（axis=-1，即最后一个轴）对数组arr进行reduceat操作
        np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
        # 再次按指定轴对数组arr进行reduceat操作
        np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
        # 断言数组arr是否与数组out相等
        assert_array_equal(arr, out)

    def test_object_array_reduceat_failure(self):
        # 当没有无效操作发生时，reduceat操作如预期般有效（这里None不参与运算）
        res = np.add.reduceat(np.array([1, None, 2], dtype=object), [1, 2])
        assert_array_equal(res, np.array([None, 2], dtype=object))
        # 当None参与运算时，应该引发错误
        with pytest.raises(TypeError):
            np.add.reduceat([1, None, 2], [0, 2])

    def test_zerosize_reduction(self):
        # 使用默认dtype和对象dtype进行测试
        for a in [[], np.array([], dtype=object)]:
            # 断言对空数组a的求和结果为0
            assert_equal(np.sum(a), 0)
            # 断言对空数组a的乘积结果为1
            assert_equal(np.prod(a), 1)
            # 断言对空数组a的任意性检查结果为False
            assert_equal(np.any(a), False)
            # 断言对空数组a的全真性检查结果为True
            assert_equal(np.all(a), True)
            # 对空数组a调用np.max应该引发ValueError异常
            assert_raises(ValueError, np.max, a)
            # 对空数组a调用np.min应该引发ValueError异常
            assert_raises(ValueError, np.min, a)
    def test_axis_out_of_bounds(self):
        # 创建一个包含两个元素的布尔类型数组
        a = np.array([False, False])
        # 检查在指定轴向（axis=1）上调用 'all' 函数时是否抛出 AxisError 异常
        assert_raises(AxisError, a.all, axis=1)
        # 创建一个包含两个元素的布尔类型数组
        a = np.array([False, False])
        # 检查在指定轴向（axis=-2）上调用 'all' 函数时是否抛出 AxisError 异常
        assert_raises(AxisError, a.all, axis=-2)

        # 创建一个包含两个元素的布尔类型数组
        a = np.array([False, False])
        # 检查在指定轴向（axis=1）上调用 'any' 函数时是否抛出 AxisError 异常
        assert_raises(AxisError, a.any, axis=1)
        # 创建一个包含两个元素的布尔类型数组
        a = np.array([False, False])
        # 检查在指定轴向（axis=-2）上调用 'any' 函数时是否抛出 AxisError 异常
        assert_raises(AxisError, a.any, axis=-2)

    def test_scalar_reduction(self):
        # 对于标量，'sum'、'prod' 等函数允许指定 axis=0
        assert_equal(np.sum(3, axis=0), 3)
        assert_equal(np.prod(3.5, axis=0), 3.5)
        assert_equal(np.any(True, axis=0), True)
        assert_equal(np.all(False, axis=0), False)
        assert_equal(np.max(3, axis=0), 3)
        assert_equal(np.min(2.5, axis=0), 2.5)

        # 检查没有身份元的 ufunc 的标量行为
        assert_equal(np.power.reduce(3), 3)

        # 确保这些操作的输出是标量类型
        assert_(type(np.prod(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.sum(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.max(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.min(np.float32(2.5), axis=0)) is np.float32)

        # 检查标量/0维数组是否被转换
        assert_(type(np.any(0, axis=0)) is np.bool)

        # 确保 0维数组 被正确处理
        class MyArray(np.ndarray):
            pass
        a = np.array(1).view(MyArray)
        assert_(type(np.any(a)) is MyArray)

    def test_casting_out_param(self):
        # 测试可以对输出进行强制类型转换
        a = np.ones((200, 100), np.int64)
        b = np.ones((200, 100), np.int64)
        c = np.ones((200, 100), np.float64)
        np.add(a, b, out=c)
        assert_equal(c, 2)

        a = np.zeros(65536)
        b = np.zeros(65536, dtype=np.float32)
        np.subtract(a, 0, out=b)
        assert_equal(b, 0)

    def test_where_param(self):
        # 测试 where= ufunc 参数在常规数组上的工作
        a = np.arange(7)
        b = np.ones(7)
        c = np.zeros(7)
        np.add(a, b, out=c, where=(a % 2 == 1))
        assert_equal(c, [0, 2, 0, 4, 0, 6, 0])

        a = np.arange(4).reshape(2, 2) + 2
        np.power(a, [2, 3], out=a, where=[[0, 1], [1, 0]])
        assert_equal(a, [[2, 27], [16, 5]])
        # 广播 where= 参数
        np.subtract(a, 2, out=a, where=[True, False])
        assert_equal(a, [[0, 27], [14, 5]])

    def test_where_param_buffer_output(self):
        # 该测试暂时跳过，因为需要向 nditer 添加掩码特性才能正常工作

        # 在输出上进行强制类型转换
        a = np.ones(10, np.int64)
        b = np.ones(10, np.int64)
        c = 1.5 * np.ones(10, np.float64)
        np.add(a, b, out=c, where=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
        assert_equal(c, [2, 1.5, 1.5, 2, 1.5, 1.5, 2, 2, 2, 1.5])
    # 定义一个测试方法，用于验证带有参数分配的情况
    def test_where_param_alloc(self):
        # 创建一个包含一个整数的 NumPy 数组，指定数据类型为 int64
        a = np.array([1], dtype=np.int64)
        # 创建一个包含一个布尔值的 NumPy 数组，指定数据类型为 bool
        m = np.array([True], dtype=bool)
        # 调用 np.sqrt 函数，仅在 where 参数为 True 的位置计算平方根，返回的结果需要与 [1] 相等
        assert_equal(np.sqrt(a, where=m), [1])

        # 创建一个包含一个浮点数的 NumPy 数组，指定数据类型为 float64
        a = np.array([1], dtype=np.float64)
        # 创建一个包含一个布尔值的 NumPy 数组，指定数据类型为 bool
        m = np.array([True], dtype=bool)
        # 调用 np.sqrt 函数，仅在 where 参数为 True 的位置计算平方根，返回的结果需要与 [1] 相等
        assert_equal(np.sqrt(a, where=m), [1])

    # 定义一个测试方法，用于验证带有广播功能的 where 参数的情况
    def test_where_with_broadcasting(self):
        # 创建一个形状为 (5000, 4) 的随机浮点数 NumPy 数组
        a = np.random.random((5000, 4))
        # 创建一个形状为 (5000, 1) 的随机浮点数 NumPy 数组
        b = np.random.random((5000, 1))

        # 创建一个布尔数组 where，指示 a 中大于 0.3 的位置
        where = a > 0.3
        # 创建一个形状与 a 相同的全零数组 out
        out = np.full_like(a, 0)
        # 调用 np.less 函数，在 where 为 True 的位置，将 a < b 的比较结果存储到 out 中
        np.less(a, b, where=where, out=out)
        # 通过广播将 b 扩展到与 where 相同的形状，然后取出 where 为 True 的部分
        b_where = np.broadcast_to(b, a.shape)[where]
        # 断言 a 中 where 为 True 的元素小于 b_where 的元素，在 out 中也是 True
        assert_array_equal((a[where] < b_where), out[where].astype(bool))
        # 断言 out 中 where 为 False 的部分全为 0
        assert not out[~where].any()  # outside mask, out remains all 0
    def check_identityless_reduction(self, a):
        # 定义一个方法用于检查无身份元素的减少操作

        # 设置数组所有元素为1
        a[...] = 1
        # 将特定位置元素设为0，验证最小值减少操作是否能正确识别零值
        a[1, 0, 0] = 0
        # 检查在全局范围内执行最小值减少操作，应返回0
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        # 检查在指定轴（0、1）上执行最小值减少操作，应返回数组 [0, 1, 1, 1]
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [0, 1, 1, 1])
        # 检查在指定轴（0、2）上执行最小值减少操作，应返回数组 [0, 1, 1]
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [0, 1, 1])
        # 检查在指定轴（1、2）上执行最小值减少操作，应返回数组 [1, 0]
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [1, 0])
        # 检查在轴0上执行最小值减少操作，应返回二维数组
        assert_equal(np.minimum.reduce(a, axis=0),
                                    [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        # 检查在轴1上执行最小值减少操作，应返回二维数组
        assert_equal(np.minimum.reduce(a, axis=1),
                                    [[1, 1, 1, 1], [0, 1, 1, 1]])
        # 检查在轴2上执行最小值减少操作，应返回二维数组
        assert_equal(np.minimum.reduce(a, axis=2),
                                    [[1, 1, 1], [0, 1, 1]])
        # 检查在空轴上执行最小值减少操作，应返回原数组a
        assert_equal(np.minimum.reduce(a, axis=()), a)

        # 重置数组所有元素为1
        a[...] = 1
        # 将不同的位置元素设为0，再次验证最小值减少操作是否能正确识别零值
        a[0, 1, 0] = 0
        # 同上，检查在各种轴上执行最小值减少操作的期望结果
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [0, 1, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [1, 0, 1])
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [0, 1])
        assert_equal(np.minimum.reduce(a, axis=0),
                                    [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=1),
                                    [[0, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=2),
                                    [[1, 0, 1], [1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=()), a)

        # 重置数组所有元素为1
        a[...] = 1
        # 将另一个不同位置的元素设为0，再次验证最小值减少操作是否能正确识别零值
        a[0, 0, 1] = 0
        # 同上，检查在各种轴上执行最小值减少操作的期望结果
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [1, 0, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [0, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [0, 1])
        assert_equal(np.minimum.reduce(a, axis=0),
                                    [[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=1),
                                    [[1, 0, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=2),
                                    [[0, 1, 1], [1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=()), a)

    @requires_memory(6 * 1024**3)
    @pytest.mark.skipif(sys.maxsize < 2**32,
            reason="test array too large for 32bit platform")
    def test_identityless_reduction_huge_array(self):
        # 回归测试 gh-20921（错误地复制身份失败）
        # 创建一个巨大的数组进行无身份元素的减少操作测试
        arr = np.zeros((2, 2**31), 'uint8')
        arr[:, 0] = [1, 3]
        arr[:, -1] = [4, 1]
        # 在完成测试后删除数组以释放内存
        res = np.maximum.reduce(arr, axis=0)
        del arr
        # 验证最大值减少操作的预期结果
        assert res[0] == 3
        assert res[-1] == 4

    def test_identityless_reduction_corder(self):
        # 创建一个按'C'顺序排列的空数组
        a = np.empty((2, 3, 4), order='C')
        # 调用上面定义的检查无身份元素减少操作的方法
        self.check_identityless_reduction(a)
    # 定义测试函数，测试在 Fortran 顺序的空数组上执行无身份的缩减
    def test_identityless_reduction_forder(self):
        # 创建一个形状为 (2, 3, 4)，Fortran 顺序的空 NumPy 数组
        a = np.empty((2, 3, 4), order='F')
        # 调用自定义函数 check_identityless_reduction 对数组 a 进行测试
        self.check_identityless_reduction(a)

    # 定义测试函数，测试在其他顺序的空数组上执行无身份的缩减
    def test_identityless_reduction_otherorder(self):
        # 创建一个形状为 (2, 4, 3)，C 顺序并交换轴 1 和 2 的 NumPy 数组
        a = np.empty((2, 4, 3), order='C').swapaxes(1, 2)
        # 调用自定义函数 check_identityless_reduction 对数组 a 进行测试
        self.check_identityless_reduction(a)

    # 定义测试函数，测试在非连续数组上执行无身份的缩减
    def test_identityless_reduction_noncontig(self):
        # 创建一个形状为 (3, 5, 4)，C 顺序并交换轴 1 和 2 的 NumPy 数组
        a = np.empty((3, 5, 4), order='C').swapaxes(1, 2)
        # 对数组 a 切片，从第一个元素开始的所有维度，得到非连续的子数组
        a = a[1:, 1:, 1:]
        # 调用自定义函数 check_identityless_reduction 对数组 a 进行测试
        self.check_identityless_reduction(a)

    # 定义测试函数，测试在非连续且未对齐的数组上执行无身份的缩减
    def test_identityless_reduction_noncontig_unaligned(self):
        # 创建一个包含 3*4*5*8 + 1 个元素的 int8 类型的 NumPy 数组
        a = np.empty((3*4*5*8 + 1,), dtype='i1')
        # 对数组 a 进行切片，从第二个元素开始，同时将数据类型转换为 float64
        a = a[1:].view(dtype='f8')
        # 重新设置数组 a 的形状为 (3, 4, 5)
        a.shape = (3, 4, 5)
        # 对数组 a 进行切片，从第一个元素开始的所有维度，得到非连续的子数组
        a = a[1:, 1:, 1:]
        # 调用自定义函数 check_identityless_reduction 对数组 a 进行测试
        self.check_identityless_reduction(a)

    # 定义测试函数，测试初始缩减操作是否依赖循环
    def test_reduce_identity_depends_on_loop():
        """
        结果类型应该始终取决于所选的循环，而不一定是输出 (仅适用于对象数组)。
        """
        # 对空数组使用 np.add.reduce，指定 dtype=object，应返回 int 类型的默认值 0
        assert type(np.add.reduce([], dtype=object)) is int
        # 创建一个 dtype=object 类型的数组 out，并对空数组进行 np.add.reduce 操作，指定 dtype=np.float64
        np.add.reduce([], out=out, dtype=np.float64)
        # 当循环为 float64 类型，但 out 为 object 类型时，结果应为 float64 类型的浮点数
        assert type(out[()]) is float

    # 定义测试函数，测试初始缩减操作
    def test_initial_reduction(self):
        # np.minimum.reduce 是一种无身份的缩减操作

        # 对空数组使用 np.maximum.reduce，指定 initial=0，应返回 0
        assert_equal(np.maximum.reduce([], initial=0), 0)

        # 对空数组使用 np.minimum.reduce，指定 initial=np.inf，应返回 np.inf
        assert_equal(np.minimum.reduce([], initial=np.inf), np.inf)
        # 对空数组使用 np.maximum.reduce，指定 initial=-np.inf，应返回 -np.inf
        assert_equal(np.maximum.reduce([], initial=-np.inf), -np.inf)

        # 随机测试
        assert_equal(np.minimum.reduce([5], initial=4), 4)
        assert_equal(np.maximum.reduce([4], initial=5), 5)
        assert_equal(np.maximum.reduce([5], initial=4), 5)
        assert_equal(np.minimum.reduce([4], initial=5), 4)

        # 检查 initial=None 对两种类型的 ufunc 缩减操作引发 ValueError
        assert_raises(ValueError, np.minimum.reduce, [], initial=None)
        assert_raises(ValueError, np.add.reduce, [], initial=None)
        # 在特殊的对象案例中也是如此：
        with pytest.raises(ValueError):
            np.add.reduce([], initial=None, dtype=object)

        # 检查 np._NoValue 是否提供默认行为
        assert_equal(np.add.reduce([], initial=np._NoValue), 0)

        # 检查 dtype=object 时 initial 关键字的行为是否按预期工作
        a = np.array([10], dtype=object)
        res = np.add.reduce(a, initial=5)
        assert_equal(res, 15)
    def test_empty_reduction_and_identity(self):
        # 创建一个形状为 (0, 5) 的全零数组
        arr = np.zeros((0, 5))
        # 对 arr 按行进行真实除法的归约，预期结果形状为 (0,)
        assert np.true_divide.reduce(arr, axis=1).shape == (0,)
        # 当按列进行归约时，由于归约操作为空，抛出 ValueError 异常
        with pytest.raises(ValueError):
            np.true_divide.reduce(arr, axis=0)

        # 测试当数组为 (0, 0, 5) 时，按行进行归约是否会抛出异常
        arr = np.zeros((0, 0, 5))
        with pytest.raises(ValueError):
            np.true_divide.reduce(arr, axis=1)

        # 使用 initial=1 进行除法归约，无论数组是否为空，预期结果应为全一数组
        res = np.true_divide.reduce(arr, axis=1, initial=1)
        assert_array_equal(res, np.ones((0, 5)))

    @pytest.mark.parametrize('axis', (0, 1, None))
    @pytest.mark.parametrize('where', (np.array([False, True, True]),
                                       np.array([[True], [False], [True]]),
                                       np.array([[True, False, False],
                                                 [False, True, False],
                                                 [False, True, True]])))
    def test_reduction_with_where(self, axis, where):
        # 创建一个 3x3 的浮点数数组 a，并备份到 a_copy 中
        a = np.arange(9.).reshape(3, 3)
        a_copy = a.copy()
        # 创建一个与 a 相同形状的全零数组 a_check
        a_check = np.zeros_like(a)
        # 在符合 where 条件的位置上对 a 中的元素进行正数化操作
        np.positive(a, out=a_check, where=where)

        # 对数组 a 按指定轴和 where 条件进行加法归约
        res = np.add.reduce(a, axis=axis, where=where)
        # 计算 a_check 按指定轴的和作为对比结果
        check = a_check.sum(axis)
        # 断言归约结果与预期结果相等
        assert_equal(res, check)
        # 断言 a 的元素没有被内部操作修改
        assert_array_equal(a, a_copy)

    @pytest.mark.parametrize(('axis', 'where'),
                             ((0, np.array([True, False, True])),
                              (1, [True, True, False]),
                              (None, True)))
    @pytest.mark.parametrize('initial', (-np.inf, 5.))
    def test_reduction_with_where_and_initial(self, axis, where, initial):
        # 创建一个 3x3 的浮点数数组 a，并备份到 a_copy 中
        a = np.arange(9.).reshape(3, 3)
        a_copy = a.copy()
        # 创建一个与 a 相同形状的数组 a_check，初始化为 -inf
        a_check = np.full(a.shape, -np.inf)
        # 在符合 where 条件的位置上对 a 中的元素进行正数化操作
        np.positive(a, out=a_check, where=where)

        # 对数组 a 按指定轴和 where 条件进行 maximum 归约，使用指定的 initial 值
        res = np.maximum.reduce(a, axis=axis, where=where, initial=initial)
        # 计算 a_check 按指定轴的最大值，使用指定的 initial 值作为对比结果
        check = a_check.max(axis, initial=initial)
        # 断言归约结果与预期结果相等
        assert_equal(res, check)

    def test_reduction_where_initial_needed(self):
        # 创建一个 3x3 的浮点数数组 a
        a = np.arange(9.).reshape(3, 3)
        # 创建一个需要的 where 条件数组 m
        m = [False, True, False]
        # 断言对 a 按照给定的 where 条件进行 maximum 归约时会抛出 ValueError 异常
        assert_raises(ValueError, np.maximum.reduce, a, where=m)

    def test_identityless_reduction_nonreorderable(self):
        # 创建一个 2x3 的浮点数数组 a
        a = np.array([[8.0, 2.0, 2.0], [1.0, 0.5, 0.25]])

        # 对数组 a 按列进行除法归约
        res = np.divide.reduce(a, axis=0)
        # 断言归约结果与预期结果相等
        assert_equal(res, [8.0, 4.0, 8.0])

        # 对数组 a 按行进行除法归约
        res = np.divide.reduce(a, axis=1)
        # 断言归约结果与预期结果相等
        assert_equal(res, [2.0, 8.0])

        # 对数组 a 进行空轴归约
        res = np.divide.reduce(a, axis=())
        # 断言归约结果与数组 a 相等
        assert_equal(res, a)

        # 断言对数组 a 按 (0, 1) 轴进行除法归约时会抛出 ValueError 异常
        assert_raises(ValueError, np.divide.reduce, a, axis=(0, 1))
    def test_safe_casting(self):
        # 在旧版本的 numpy 中，就地操作使用了 'unsafe' 的类型转换规则。
        # 在版本 >= 1.10 中，默认使用 'same_kind'，不满足时会抛出异常而不是警告。
        # 创建一个整型数组 a
        a = np.array([1, 2, 3], dtype=int)
        # 非就地加法是允许的
        assert_array_equal(assert_no_warnings(np.add, a, 1.1),
                           [2.1, 3.1, 4.1])
        # 使用 out 参数时会抛出 TypeError 异常
        assert_raises(TypeError, np.add, a, 1.1, out=a)

        def add_inplace(a, b):
            # 就地加法函数
            a += b

        # 就地加法时也会抛出 TypeError 异常
        assert_raises(TypeError, add_inplace, a, 1.1)
        # 显式地使用 'unsafe' 类型转换规则不会抛出异常
        assert_no_warnings(np.add, a, 1.1, out=a, casting="unsafe")
        # 验证数组 a 的结果
        assert_array_equal(a, [2, 3, 4])

    def test_ufunc_custom_out(self):
        # 测试使用内置输入类型和自定义输出类型的 ufunc

        # 创建整型数组 a 和 b
        a = np.array([0, 1, 2], dtype='i8')
        b = np.array([0, 1, 2], dtype='i8')
        # 创建一个空数组 c，类型为 _rational_tests.rational
        c = np.empty(3, dtype=_rational_tests.rational)

        # 必须指定输出以便 numpy 知道要查找哪种 ufunc 签名
        result = _rational_tests.test_add(a, b, c)
        target = np.array([0, 2, 4], dtype=_rational_tests.rational)
        assert_equal(result, target)

        # 新的解析方式意味着我们通常可以找到匹配的自定义循环
        result = _rational_tests.test_add(a, b)
        assert_equal(result, target)

        # 即使默认的公共 dtype 提升也能正常工作：
        result = _rational_tests.test_add(a, b.astype(np.uint16), out=c)
        assert_equal(result, target)

        # 标量路径曾经进入传统的提升方式，但现在不会了：
        result = _rational_tests.test_add(a, np.uint16(2))
        target = np.array([2, 3, 4], dtype=_rational_tests.rational)
        assert_equal(result, target)

    def test_operand_flags(self):
        # 测试操作标志

        # 创建一个 4x4 的整型数组 a 和一个 3x3 的整型数组 b
        a = np.arange(16, dtype=int).reshape(4, 4)
        b = np.arange(9, dtype=int).reshape(3, 3)
        # 将 b 就地加到 a 的子集中
        opflag_tests.inplace_add(a[:-1, :-1], b)
        # 验证 a 的结果
        assert_equal(a, np.array([[0, 2, 4, 3], [7, 9, 11, 7],
            [14, 16, 18, 11], [12, 13, 14, 15]]))

        # 创建一个标量数组 a
        a = np.array(0)
        # 就地加法
        opflag_tests.inplace_add(a, 3)
        assert_equal(a, 3)
        # 再次就地加法
        opflag_tests.inplace_add(a, [3, 4])
        assert_equal(a, 10)

    def test_struct_ufunc(self):
        # 测试结构化 ufunc

        import numpy._core._struct_ufunc_tests as struct_ufunc

        # 创建一个结构化数组 a 和 b，包含三个无符号 8 字节整数字段
        a = np.array([(1, 2, 3)], dtype='u8,u8,u8')
        b = np.array([(1, 2, 3)], dtype='u8,u8,u8')

        # 使用结构化 ufunc 计算结果
        result = struct_ufunc.add_triplet(a, b)
        assert_equal(result, np.array([(2, 4, 6)], dtype='u8,u8,u8'))
        # 注册失败时会抛出 RuntimeError 异常
        assert_raises(RuntimeError, struct_ufunc.register_fail)
    # 定义测试函数 test_custom_ufunc，用于测试自定义的通用函数
    def test_custom_ufunc(self):
        # 创建包含有理数对象的 NumPy 数组 a
        a = np.array(
            [_rational_tests.rational(1, 2),
             _rational_tests.rational(1, 3),
             _rational_tests.rational(1, 4)],
            dtype=_rational_tests.rational)
        # 创建包含有理数对象的 NumPy 数组 b，与 a 结构相同
        b = np.array(
            [_rational_tests.rational(1, 2),
             _rational_tests.rational(1, 3),
             _rational_tests.rational(1, 4)],
            dtype=_rational_tests.rational)

        # 调用测试函数 test_add_rationals，对 a 和 b 进行有理数加法运算
        result = _rational_tests.test_add_rationals(a, b)
        # 预期结果是包含有理数对象的 NumPy 数组 expected
        expected = np.array(
            [_rational_tests.rational(1),
             _rational_tests.rational(2, 3),
             _rational_tests.rational(1, 2)],
            dtype=_rational_tests.rational)
        # 使用断言比较计算结果 result 和预期结果 expected 是否相等
        assert_equal(result, expected)

    # 定义测试函数 test_custom_ufunc_forced_sig，用于测试强制签名的自定义通用函数
    def test_custom_ufunc_forced_sig(self):
        # 在此测试中，验证当签名不匹配时是否能正确引发 TypeError 异常
        with assert_raises(TypeError):
            np.multiply(_rational_tests.rational(1), 1,
                        signature=(_rational_tests.rational, int, None))

    # 定义测试函数 test_custom_array_like，用于测试自定义类 MyThing
    def test_custom_array_like(self):

        # 定义类 MyThing，模拟 NumPy 中的数组行为
        class MyThing:
            __array_priority__ = 1000

            rmul_count = 0
            getitem_count = 0

            def __init__(self, shape):
                self.shape = shape

            def __len__(self):
                return self.shape[0]

            def __getitem__(self, i):
                # 记录 __getitem__ 方法的调用次数
                MyThing.getitem_count += 1
                if not isinstance(i, tuple):
                    i = (i,)
                if len(i) > self.ndim:
                    raise IndexError("boo")

                return MyThing(self.shape[len(i):])

            def __rmul__(self, other):
                # 记录 __rmul__ 方法的调用次数
                MyThing.rmul_count += 1
                return self

        # 测试 MyThing 类的 __rmul__ 方法是否正确调用
        np.float64(5)*MyThing((3, 3))
        # 使用断言验证 __rmul__ 方法是否仅调用了一次
        assert_(MyThing.rmul_count == 1, MyThing.rmul_count)
        # 使用断言验证 __getitem__ 方法调用次数不超过 2 次
        assert_(MyThing.getitem_count <= 2, MyThing.getitem_count)

    # 定义测试函数 test_ufunc_at_basic，用于测试 ufunc.at 的基本用法
    @pytest.mark.parametrize("a", (
                             np.arange(10, dtype=int),
                             np.arange(10, dtype=_rational_tests.rational),
                             ))
    def test_ufunc_at_basic(self, a):

        # 复制数组 a 到 aa
        aa = a.copy()
        # 在索引 [2, 5, 2] 处使用 add 操作增加值为 1
        np.add.at(aa, [2, 5, 2], 1)
        # 使用断言验证 aa 是否与预期相等
        assert_equal(aa, [0, 1, 4, 3, 4, 6, 6, 7, 8, 9])

        # 使用 pytest 验证，未提供第二个操作数时是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            np.add.at(aa, [2, 5, 3])

        # 复制数组 a 到 aa
        aa = a.copy()
        # 在索引 [2, 5, 3] 处使用 negative 操作
        np.negative.at(aa, [2, 5, 3])
        # 使用断言验证 aa 是否与预期相等
        assert_equal(aa, [0, 1, -2, -3, 4, -5, 6, 7, 8, 9])

        # 复制数组 a 到 aa
        aa = a.copy()
        b = np.array([100, 100, 100])
        # 在索引 [2, 5, 2] 处使用 add 操作增加数组 b
        np.add.at(aa, [2, 5, 2], b)
        # 使用断言验证 aa 是否与预期相等
        assert_equal(aa, [0, 1, 202, 3, 4, 105, 6, 7, 8, 9])

        # 使用 pytest 验证，当提供多余的第二个操作数时是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            np.negative.at(a, [2, 5, 3], [1, 2, 3])

        # 使用 pytest 验证，当第二个操作数无法转换为数组时是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            np.add.at(a, [2, 5, 3], [[1, 2], 1])

    # ufuncs with indexed loops for performance in ufunc.at
    # 定义包含一组 NumPy 通用函数的列表，用于测试
    indexed_ufuncs = [np.add, np.subtract, np.multiply, np.floor_divide,
                      np.maximum, np.minimum, np.fmax, np.fmin]

    @pytest.mark.parametrize(
                "typecode", np.typecodes['AllInteger'] + np.typecodes['Float'])
    @pytest.mark.parametrize("ufunc", indexed_ufuncs)
    # 定义测试类的方法，参数化类型码和通用函数
    def test_ufunc_at_inner_loops(self, typecode, ufunc):
        if ufunc is np.divide and typecode in np.typecodes['AllInteger']:
            # 对于整数除法，避免除以零和无穷大
            a = np.ones(100, dtype=typecode)
            indx = np.random.randint(100, size=30, dtype=np.intp)
            vals = np.arange(1, 31, dtype=typecode)
        else:
            a = np.ones(1000, dtype=typecode)
            indx = np.random.randint(1000, size=3000, dtype=np.intp)
            vals = np.arange(3000, dtype=typecode)
        atag = a.copy()
        # 执行计算两次并比较结果
        with warnings.catch_warnings(record=True) as w_at:
            warnings.simplefilter('always')
            ufunc.at(a, indx, vals)
        with warnings.catch_warnings(record=True) as w_loop:
            warnings.simplefilter('always')
            for i, v in zip(indx, vals):
                # 确保所有的工作都发生在通用函数内部
                # 以便复制错误/警告处理
                ufunc(atag[i], v, out=atag[i:i+1], casting="unsafe")
        # 断言结果相等
        assert_equal(atag, a)
        # 如果 w_loop 发出警告，则确保 w_at 也发出了警告
        if len(w_loop) > 0:
            #
            assert len(w_at) > 0
            assert w_at[0].category == w_loop[0].category
            assert str(w_at[0].message)[:10] == str(w_loop[0].message)[:10]

    @pytest.mark.parametrize("typecode", np.typecodes['Complex'])
    @pytest.mark.parametrize("ufunc", [np.add, np.subtract, np.multiply])
    # 定义测试复数类型的通用函数的方法
    def test_ufunc_at_inner_loops_complex(self, typecode, ufunc):
        a = np.ones(10, dtype=typecode)
        indx = np.concatenate([np.ones(6, dtype=np.intp),
                               np.full(18, 4, dtype=np.intp)])
        value = a.dtype.type(1j)
        ufunc.at(a, indx, value)
        expected = np.ones_like(a)
        if ufunc is np.multiply:
            expected[1] = expected[4] = -1
        else:
            expected[1] += 6 * (value if ufunc is np.add else -value)
            expected[4] += 18 * (value if ufunc is np.add else -value)

        # 断言数组是否与预期结果相等
        assert_array_equal(a, expected)

    def test_ufunc_at_ellipsis(self):
        # 确保索引循环检查不会在子空间的迭代中出错
        arr = np.zeros(5)
        np.add.at(arr, slice(None), np.ones(5))
        assert_array_equal(arr, np.ones(5))

    def test_ufunc_at_negative(self):
        arr = np.ones(5, dtype=np.int32)
        indx = np.arange(5)
        umt.indexed_negative.at(arr, indx)
        # 如果结果为 [-1, -1, -1, -100, 0]，则说明使用了常规的步进循环
        assert np.all(arr == [-1, -1, -1, -200, -1])
    # 定义一个测试函数，用于验证在大数组上使用 np.add.at 的行为
    def test_ufunc_at_large(self):
        # issue gh-23457
        # 创建一个长度为 8195 的整数类型的全零数组作为索引
        indices = np.zeros(8195, dtype=np.int16)
        # 创建一个长度为 8195 的浮点类型的全零数组作为数据
        b = np.zeros(8195, dtype=float)
        # 设置数据数组 b 的前两个元素为特定值，最后一个元素为 100
        b[0] = 10
        b[1] = 5
        b[8192:] = 100
        # 创建一个长度为 1 的浮点类型的全零数组
        a = np.zeros(1, dtype=float)
        # 使用 np.add.at 将数组 b 的值按照 indices 数组中的索引位置累加到数组 a 中
        np.add.at(a, indices, b)
        # 验证 a 数组的第一个元素是否等于 b 数组所有元素的和
        assert a[0] == b.sum()

    # 定义一个测试函数，用于验证在快速路径下进行索引转换的行为
    def test_cast_index_fastpath(self):
        # 创建一个长度为 10 的全零数组
        arr = np.zeros(10)
        # 创建一个长度为 100000 的全一数组
        values = np.ones(100000)
        # 创建一个长度与 values 数组相同的无符号字节类型的全零索引数组
        # index 必须被转换，可能会被分块缓存：
        index = np.zeros(len(values), dtype=np.uint8)
        # 使用 np.add.at 将 values 数组的值按照 index 数组中的索引位置累加到 arr 数组中
        np.add.at(arr, index, values)
        # 验证 arr 数组的第一个元素是否等于 values 数组的长度
        assert arr[0] == len(values)

    # 使用参数化装饰器，定义一个测试函数，验证在快速路径下对标量值进行索引的行为
    @pytest.mark.parametrize("value", [
        np.ones(1), np.ones(()), np.float64(1.), 1.])
    def test_ufunc_at_scalar_value_fastpath(self, value):
        # 创建一个长度为 1000 的全零数组
        arr = np.zeros(1000)
        # index 必须被转换，可能会被分块缓存：
        # 创建一个重复两次的 0 到 999 的数组作为索引
        index = np.repeat(np.arange(1000), 2)
        # 使用 np.add.at 将 value 添加到 arr 数组的 index 索引位置处
        np.add.at(arr, index, value)
        # 验证 arr 数组是否与一个全为 2 * value 的数组相等
        assert_array_equal(arr, np.full_like(arr, 2 * value))
    # 定义一个测试函数，用于测试多维数组上的 np.add.at 函数
    def test_ufunc_at_multiD(self):
        # 创建一个 3x3 的数组 a，其元素为 [0, 1, 2, ..., 8]
        a = np.arange(9).reshape(3, 3)
        # 创建一个 3x3 的数组 b，每行元素为 [100, 100, 100], [200, 200, 200], [300, 300, 300]
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        # 在 a 的特定位置进行累加操作，使用 np.add.at 函数
        np.add.at(a, (slice(None), [1, 2, 1]), b)
        # 断言 a 是否与预期的值相等
        assert_equal(a, [[0, 201, 102], [3, 404, 205], [6, 607, 308]])

        # 创建一个 3x3x3 的数组 a，其元素为 [0, 1, 2, ..., 26]
        a = np.arange(27).reshape(3, 3, 3)
        # 创建一个长度为 3 的数组 b，元素为 [100, 200, 300]
        b = np.array([100, 200, 300])
        # 在 a 的特定位置进行累加操作，使用 np.add.at 函数
        np.add.at(a, (slice(None), slice(None), [1, 2, 1]), b)
        # 断言 a 是否与预期的值相等
        assert_equal(a,
            [[[0, 401, 202],
              [3, 404, 205],
              [6, 407, 208]],

             [[9, 410, 211],
              [12, 413, 214],
              [15, 416, 217]],

             [[18, 419, 220],
              [21, 422, 223],
              [24, 425, 226]]])

        # 创建一个 3x3 的数组 a，其元素为 [0, 1, 2, ..., 8]
        a = np.arange(9).reshape(3, 3)
        # 创建一个 3x3 的数组 b，每行元素为 [100, 100, 100], [200, 200, 200], [300, 300, 300]
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        # 在 a 的特定位置进行累加操作，使用 np.add.at 函数
        np.add.at(a, ([1, 2, 1], slice(None)), b)
        # 断言 a 是否与预期的值相等
        assert_equal(a, [[0, 1, 2], [403, 404, 405], [206, 207, 208]])

        # 创建一个 3x3x3 的数组 a，其元素为 [0, 1, 2, ..., 26]
        a = np.arange(27).reshape(3, 3, 3)
        # 创建一个长度为 3 的数组 b，元素为 [100, 200, 300]
        b = np.array([100, 200, 300])
        # 在 a 的特定位置进行累加操作，使用 np.add.at 函数
        np.add.at(a, (slice(None), [1, 2, 1], slice(None)), b)
        # 断言 a 是否与预期的值相等
        assert_equal(a,
            [[[0,  1,  2],
              [203, 404, 605],
              [106, 207, 308]],

             [[9,  10, 11],
              [212, 413, 614],
              [115, 216, 317]],

             [[18, 19, 20],
              [221, 422, 623],
              [124, 225, 326]]])

        # 创建一个 3x3 的数组 a，其元素为 [0, 1, 2, ..., 8]
        a = np.arange(9).reshape(3, 3)
        # 创建一个长度为 3 的数组 b，元素为 [100, 200, 300]
        b = np.array([100, 200, 300])
        # 在 a 的特定位置进行累加操作，使用 np.add.at 函数
        np.add.at(a, (0, [1, 2, 1]), b)
        # 断言 a 是否与预期的值相等
        assert_equal(a, [[0, 401, 202], [3, 4, 5], [6, 7, 8]])

        # 创建一个 3x3x3 的数组 a，其元素为 [0, 1, 2, ..., 26]
        a = np.arange(27).reshape(3, 3, 3)
        # 创建一个长度为 3 的数组 b，元素为 [100, 200, 300]
        b = np.array([100, 200, 300])
        # 在 a 的特定位置进行累加操作，使用 np.add.at 函数
        np.add.at(a, ([1, 2, 1], 0, slice(None)), b)
        # 断言 a 是否与预期的值相等
        assert_equal(a,
            [[[0,  1,  2],
              [3,  4,  5],
              [6,  7,  8]],

             [[209, 410, 611],
              [12,  13, 14],
              [15,  16, 17]],

             [[118, 219, 320],
              [21,  22, 23],
              [24,  25, 26]]])

        # 创建一个 3x3x3 的数组 a，其元素为 [0, 1, 2, ..., 26]
        a = np.arange(27).reshape(3, 3, 3)
        # 创建一个长度为 3 的数组 b，元素为 [100, 200, 300]
        b = np.array([100, 200, 300])
        # 在 a 的每个位置进行累加操作，使用 np.add.at 函数
        np.add.at(a, (slice(None), slice(None), slice(None)), b)
        # 断言 a 是否与预期的值相等
        assert_equal(a,
            [[[100, 201, 302],
              [103, 204, 305],
              [106, 207, 308]],

             [[109, 210, 311],
              [112, 213, 314],
              [115, 216, 317]],

             [[118, 219, 320],
              [121, 222, 323],
              [124, 225, 326]]])

    # 定义一个测试函数，用于测试 np.add.at 函数在零维数组上的行为
    def test_ufunc_at_0D(self):
        # 创建一个零维数组 a，其元素为 0
        a = np.array(0)
        # 在 a 上的特定位置进行累加操作，使用 np.add.at 函数
        np.add.at(a, (), 1)
        # 断言 a 是否等于预期值 1
        assert_equal(a, 1)

        # 断言 np.add.at 在零维数组上的行为是否抛出 IndexError 异常
        assert_raises(IndexError, np.add.at, a, 0, 1)
        assert_raises(IndexError, np.add.at, a, [], 1)

    # 定义一个测试函数，用于测试 np.power.at 函数的行为
    def test_ufunc_at_dtypes(self):
        # 创建一个包含 0 到 9 的一维数组 a
        a = np.arange(10)
        # 在 a 的特定位置进行幂运算，使用 np.power.at 函数
        np.power.at(a, [1, 2, 3, 2], 3.5)
        # 断言 a 是否等于预期值
        assert_equal(a, np.array([0, 1, 4414, 46, 4, 5, 6, 7, 8, 9]))
    def test_ufunc_at_boolean(self):
        # Test boolean indexing and boolean ufuncs
        
        # 创建一个包含 0 到 9 的数组
        a = np.arange(10)
        
        # 使用模运算创建布尔索引
        index = a % 2 == 0
        
        # 在数组 a 中，根据布尔索引位置将指定值 [0, 2, 4, 6, 8] 添加
        np.equal.at(a, index, [0, 2, 4, 6, 8])
        
        # 断言数组 a 的值与期望值一致
        assert_equal(a, [1, 1, 1, 3, 1, 5, 1, 7, 1, 9])

        # Test unary operator
        
        # 创建一个包含 0 到 9 的无符号整数数组
        a = np.arange(10, dtype='u4')
        
        # 在数组 a 中指定位置应用按位取反操作
        np.invert.at(a, [2, 5, 2])
        
        # 断言数组 a 的值与期望值一致
        assert_equal(a, [0, 1, 2, 3, 4, 5 ^ 0xffffffff, 6, 7, 8, 9])

    def test_ufunc_at_advanced(self):
        # Test empty subspace
        
        # 创建一个包含 0 到 3 的原始数组
        orig = np.arange(4)
        
        # 创建一个空的子空间数组 a
        a = orig[:, None][:, 0:0]
        
        # 在数组 a 中指定位置添加值 3
        np.add.at(a, [0, 1], 3)
        
        # 断言数组 orig 与期望值一致
        assert_array_equal(orig, np.arange(4))

        # Test with swapped byte order
        
        # 创建一个具有交换字节顺序的索引数组
        index = np.array([1, 2, 1], np.dtype('i').newbyteorder())
        
        # 创建一个具有交换字节顺序的数值数组
        values = np.array([1, 2, 3, 4], np.dtype('f').newbyteorder())
        
        # 在值数组中指定位置添加值 3
        np.add.at(values, index, 3)
        
        # 断言数组 values 与期望值一致
        assert_array_equal(values, [1, 8, 6, 4])

        # Test exception thrown
        
        # 创建一个包含对象的数组 values
        values = np.array(['a', 1], dtype=object)
        
        # 断言在执行 np.add.at 操作时会引发 TypeError 异常
        assert_raises(TypeError, np.add.at, values, [0, 1], 1)
        
        # 断言数组 values 与期望值一致
        assert_array_equal(values, np.array(['a', 1], dtype=object))

        # Test multiple output ufuncs raise error, gh-5665
        
        # 断言执行 np.modf.at 操作时会引发 ValueError 异常
        assert_raises(ValueError, np.modf.at, np.arange(10), [1])

        # Test maximum
        
        # 创建一个包含 [1, 2, 3] 的数组 a
        a = np.array([1, 2, 3])
        
        # 在数组 a 中指定位置添加最大值
        np.maximum.at(a, [0], 0)
        
        # 断言数组 a 的值与期望值一致
        assert_equal(a, np.array([1, 2, 3]))

    @pytest.mark.parametrize("dtype",
            np.typecodes['AllInteger'] + np.typecodes['Float'])
    @pytest.mark.parametrize("ufunc",
            [np.add, np.subtract, np.divide, np.minimum, np.maximum])
    def test_at_negative_indexes(self, dtype, ufunc):
        # Test ufuncs with negative indexes
        
        # 创建一个包含 0 到 9 的指定类型数组 a
        a = np.arange(0, 10).astype(dtype)
        
        # 创建一个包含负数索引的数组 indxs 和相应的值数组 vals
        indxs = np.array([-1, 1, -1, 2]).astype(np.intp)
        vals = np.array([1, 5, 2, 10], dtype=a.dtype)

        # 创建期望的数组 expected，通过循环将值应用到指定索引处
        expected = a.copy()
        for i, v in zip(indxs, vals):
            expected[i] = ufunc(expected[i], v)

        # 在数组 a 中指定位置添加值 vals
        ufunc.at(a, indxs, vals)
        
        # 断言数组 a 的值与期望值一致
        assert_array_equal(a, expected)
        
        # 断言数组 indxs 的值与期望值一致
        assert np.all(indxs == [-1, 1, -1, 2])

    def test_at_not_none_signature(self):
        # Test ufuncs with non-trivial signature raise a TypeError
        
        # 创建一个包含全为 1 的多维数组 a 和 b
        a = np.ones((2, 2, 2))
        b = np.ones((1, 2, 2))
        
        # 断言在执行 np.matmul.at 操作时会引发 TypeError 异常
        assert_raises(TypeError, np.matmul.at, a, [0], b)

        # 创建一个多维数组 a
        a = np.array([[[1, 2], [3, 4]]])
        
        # 断言在执行 np.linalg._umath_linalg.det.at 操作时会引发 TypeError 异常
        assert_raises(TypeError, np.linalg._umath_linalg.det.at, a, [0])

    def test_at_no_loop_for_op(self):
        # str dtype does not have a ufunc loop for np.add
        
        # 创建一个包含字符串的数组 arr
        arr = np.ones(10, dtype=str)
        
        # 使用 pytest 断言，在执行 np.add.at 操作时会引发 np._core._exceptions._UFuncNoLoopError 异常
        with pytest.raises(np._core._exceptions._UFuncNoLoopError):
            np.add.at(arr, [0, 1], [0, 1])

    def test_at_output_casting(self):
        # Test output casting
        
        # 创建一个包含 [-1] 的数组 arr
        arr = np.array([-1])
        
        # 在数组 arr 中指定位置添加值 [0]
        np.equal.at(arr, [0], [0])
        
        # 使用 pytest 断言，数组 arr 第一个元素的值为 0
        assert arr[0] == 0

    def test_at_broadcast_failure(self):
        # Test broadcast failure
        
        # 创建一个包含 0 到 4 的数组 arr
        arr = np.arange(5)
        
        # 使用 pytest 断言，在执行 np.add.at 操作时会引发 ValueError 异常
        with pytest.raises(ValueError):
            np.add.at(arr, [0, 1], [1, 2, 3])
    # 定义测试函数 test_reduce_arguments(self)
    def test_reduce_arguments(self):
        # 将 numpy 的 add.reduce 函数赋给变量 f
        f = np.add.reduce
        # 创建一个形状为 (5, 2) 的全为 1 的整数数组 d
        d = np.ones((5,2), dtype=int)
        # 创建一个与 d 的 dtype 相同的形状为 (2,) 的全为 1 的数组 o
        o = np.ones((2,), dtype=d.dtype)
        # 计算 o 的每个元素乘以 5 的结果，并赋给变量 r
        r = o * 5
        # 断言 np.add.reduce(d) 的结果与 r 相等
        assert_equal(f(d), r)

        # 使用不同方式调用 np.add.reduce(d)，并与 r 进行断言
        assert_equal(f(d, axis=0), r)
        assert_equal(f(d, 0), r)
        assert_equal(f(d, 0, dtype=None), r)
        assert_equal(f(d, 0, dtype='i'), r)
        assert_equal(f(d, 0, 'i'), r)
        assert_equal(f(d, 0, None), r)
        assert_equal(f(d, 0, None, out=None), r)
        assert_equal(f(d, 0, None, out=o), r)
        assert_equal(f(d, 0, None, o), r)
        assert_equal(f(d, 0, None, None), r)
        assert_equal(f(d, 0, None, None, keepdims=False), r)
        assert_equal(f(d, 0, None, None, True), r.reshape((1,) + r.shape))
        assert_equal(f(d, 0, None, None, False, 0), r)
        assert_equal(f(d, 0, None, None, False, initial=0), r)
        assert_equal(f(d, 0, None, None, False, 0, True), r)
        assert_equal(f(d, 0, None, None, False, 0, where=True), r)

        # 使用多个关键字参数调用 np.add.reduce(d)，并与 r 进行断言
        assert_equal(f(d, axis=0, dtype=None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, dtype=None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, None, out=None, keepdims=False, initial=0,
                       where=True), r)

        # 断言抛出 TypeError 异常，因为没有传递足够的参数
        assert_raises(TypeError, f)
        # 断言抛出 TypeError 异常，因为传递了过多的参数
        assert_raises(TypeError, f, d, 0, None, None, False, 0, True, 1)
        # 断言抛出 TypeError 异常，因为 axis 参数的类型不正确
        assert_raises(TypeError, f, d, "invalid")
        assert_raises(TypeError, f, d, axis="invalid")
        assert_raises(TypeError, f, d, axis="invalid", dtype=None,
                      keepdims=True)
        # 断言抛出 TypeError 异常，因为 dtype 参数的类型不正确
        assert_raises(TypeError, f, d, 0, "invalid")
        assert_raises(TypeError, f, d, dtype="invalid")
        assert_raises(TypeError, f, d, dtype="invalid", out=None)
        # 断言抛出 TypeError 异常，因为 out 参数的类型不正确
        assert_raises(TypeError, f, d, 0, None, "invalid")
        assert_raises(TypeError, f, d, out="invalid")
        assert_raises(TypeError, f, d, out="invalid", dtype=None)
        # 断言抛出 TypeError 异常，因为 keepdims 参数的类型不正确
        assert_raises(TypeError, f, d, 0, None, None, False, 0, keepdims="invalid")

        # 断言抛出 TypeError 异常，因为传递了无效的关键字参数
        assert_raises(TypeError, f, d, axis=0, dtype=None, invalid=0)
        assert_raises(TypeError, f, d, invalid=0)
        assert_raises(TypeError, f, d, 0, keepdims=True, invalid="invalid",
                      out=None)
        assert_raises(TypeError, f, d, axis=0, dtype=None, keepdims=True,
                      out=None, invalid=0)
        assert_raises(TypeError, f, d, axis=0, dtype=None,
                      out=None, invalid=0)
    def test_structured_equal(self):
        # https://github.com/numpy/numpy/issues/4855
        # 定义一个继承自 np.ndarray 的自定义类 MyA
        class MyA(np.ndarray):
            # 重载 __array_ufunc__ 方法以支持数组运算
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                # 调用 ufunc 对应的方法，并将输入转换为 np.ndarray 类型后执行
                return getattr(ufunc, method)(*(input.view(np.ndarray)
                                              for input in inputs), **kwargs)
        # 创建一个形状为 (4,3) 的 ndarray
        a = np.arange(12.).reshape(4,3)
        # 将 a 视图转换为结构化数组，并去除多余的维度
        ra = a.view(dtype=('f8,f8,f8')).squeeze()
        # 将 ra 视图转换为自定义类 MyA 的实例
        mra = ra.view(MyA)

        # 目标数组，包含 True, False, False, False
        target = np.array([ True, False, False, False], dtype=bool)
        # 断言表达式，检查 mra 是否等于 ra[0] 的每个元素与 target 相等
        assert_equal(np.all(target == (mra == ra[0])), True)

    def test_scalar_equal(self):
        # 标量比较应始终有效，无需弃用警告。
        # 即使 ufunc 失败也应如此。
        a = np.array(0.)
        b = np.array('a')
        assert_(a != b)
        assert_(b != a)
        assert_(not (a == b))
        assert_(not (b == a))

    def test_NotImplemented_not_returned(self):
        # 参见 gh-5964 和 gh-2091。一些函数与操作符无关，但在过去的修复中有其他原因。
        # 二元函数列表
        binary_funcs = [
            np.power, np.add, np.subtract, np.multiply, np.divide,
            np.true_divide, np.floor_divide, np.bitwise_and, np.bitwise_or,
            np.bitwise_xor, np.left_shift, np.right_shift, np.fmax,
            np.fmin, np.fmod, np.hypot, np.logaddexp, np.logaddexp2,
            np.maximum, np.minimum, np.mod,
            np.greater, np.greater_equal, np.less, np.less_equal,
            np.equal, np.not_equal]

        a = np.array('1')
        b = 1
        c = np.array([1., 2.])
        # 遍历二元函数列表，检查各函数对 a, b 或 c 的调用是否引发 TypeError
        for f in binary_funcs:
            assert_raises(TypeError, f, a, b)
            assert_raises(TypeError, f, c, a)

    @pytest.mark.parametrize("ufunc",
             [np.logical_and, np.logical_or])  # logical_xor object loop is bad
    @pytest.mark.parametrize("signature",
             [(None, None, object), (object, None, None),
              (None, object, None)])
    def test_logical_ufuncs_object_signatures(self, ufunc, signature):
        # 使用对象类型数组 a，对 ufunc 执行指定签名的逻辑运算
        a = np.array([True, None, False], dtype=object)
        res = ufunc(a, a, signature=signature)
        # 断言结果 res 的数据类型为 object
        assert res.dtype == object

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    @pytest.mark.parametrize("signature",
                 [(bool, None, object), (object, None, bool),
                  (None, object, bool)])
    def test_logical_ufuncs_mixed_object_signatures(self, ufunc, signature):
        # 大多数混合签名会失败（除了 bool 输出的情况，如 `OO->?`）
        a = np.array([True, None, False])
        # 使用 pytest 检查对于给定签名，ufunc 对 a 运算是否会引发 TypeError
        with pytest.raises(TypeError):
            ufunc(a, a, signature=signature)
    def test_logical_ufuncs_support_anything(self, ufunc):
        # logical ufuncs支持即使不能提升的输入：
        a = np.array(b'1', dtype="V3")  # 创建一个二进制数组a，数据类型为V3（长度为3的void类型）
        c = np.array([1., 2.])  # 创建一个浮点数组c
        assert_array_equal(ufunc(a, c), ufunc([True, True], True))  # 断言ufunc对a和c的结果等于ufunc([True, True], True)
        assert ufunc.reduce(a) == True  # 断言对a进行reduce操作的结果为True
        # 检查输出是否无影响：
        out = np.zeros(2, dtype=np.int32)  # 创建一个全零数组out，数据类型为int32
        expected = ufunc([True, True], True).astype(out.dtype)  # 期望的结果，将ufunc([True, True], True)转换为out的数据类型
        assert_array_equal(ufunc(a, c, out=out), expected)  # 断言ufunc对a和c进行out输出的结果等于期望的结果
        out = np.zeros((), dtype=np.int32)  # 创建一个形状为空的全零数组out，数据类型为int32
        assert ufunc.reduce(a, out=out) == True  # 断言对a进行reduce操作并将结果输出到out的结果为True
        # 最后检查，测试当out和a匹配时的reduce操作（这里的复杂性在于"i,i->?"看似正确，但不应匹配）
        a = np.array([3], dtype="i")  # 创建一个整数数组a，数据类型为i
        out = np.zeros((), dtype=a.dtype)  # 创建一个形状为空的全零数组out，数据类型与a相同
        assert ufunc.reduce(a, out=out) == 1  # 断言对a进行reduce操作并将结果输出到out的结果为1

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    @pytest.mark.parametrize("dtype", ["S", "U"])
    @pytest.mark.parametrize("values", [["1", "hi", "0"], ["", ""]])
    def test_logical_ufuncs_supports_string(self, ufunc, dtype, values):
        # 注意values要么全部为True，要么全部为False
        arr = np.array(values, dtype=dtype)  # 创建一个包含字符串值的数组arr，数据类型为dtype指定的类型
        obj_arr = np.array(values, dtype=object)  # 创建一个包含对象类型的数组obj_arr，数据类型为object
        res = ufunc(arr, arr)  # 对数组arr进行ufunc操作
        expected = ufunc(obj_arr, obj_arr, dtype=bool)  # 使用对象数组obj_arr进行ufunc操作，期望结果的数据类型为bool

        assert_array_equal(res, expected)  # 断言结果数组res等于期望的结果数组expected

        res = ufunc.reduce(arr)  # 对数组arr进行reduce操作
        expected = ufunc.reduce(obj_arr, dtype=bool)  # 使用对象数组obj_arr进行reduce操作，期望结果的数据类型为bool
        assert_array_equal(res, expected)  # 断言reduce操作的结果数组res等于期望的结果数组expected

    @pytest.mark.parametrize("ufunc",
             [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_out_cast_check(self, ufunc):
        a = np.array('1')  # 创建一个包含字符串'1'的数组a
        c = np.array([1., 2.])  # 创建一个浮点数组c
        out = a.copy()  # 复制数组a到out
        with pytest.raises(TypeError):
            # 它可能是安全的，但是等效的类型转换不应该：
            ufunc(a, c, out=out, casting="equiv")  # 断言ufunc对a和c进行out输出时，使用"equiv"类型转换会引发TypeError异常

    def test_reducelike_byteorder_resolution(self):
        # 参见gh-20699，字节顺序的变化在类型解析中需要额外注意，以确保以下操作成功：
        arr_be = np.arange(10, dtype=">i8")  # 创建一个大端字节顺序的整数数组arr_be
        arr_le = np.arange(10, dtype="<i8")  # 创建一个小端字节顺序的整数数组arr_le

        assert np.add.reduce(arr_be) == np.add.reduce(arr_le)  # 断言对arr_be和arr_le进行reduce加法操作的结果相等
        assert_array_equal(np.add.accumulate(arr_be), np.add.accumulate(arr_le))  # 断言对arr_be和arr_le进行累积加法操作的结果数组相等
        assert_array_equal(
            np.add.reduceat(arr_be, [1]), np.add.reduceat(arr_le, [1]))  # 断言对arr_be和arr_le根据指定位置进行reduce加法操作的结果数组相等
    def test_reducelike_out_promotes(self):
        # 检查在归约操作中，是否考虑了输出参数（out），参见 issue gh-20455。
        # 注意，未来这些路径可能更偏向于使用 `initial=`，并且对于 add 和 prod 操作不会默认向上转型为整数。
        
        # 创建一个包含 1000 个元素的 uint8 类型数组，所有元素初始化为 1
        arr = np.ones(1000, dtype=np.uint8)
        # 创建一个空的 uint16 类型的数组作为输出参数 out
        out = np.zeros((), dtype=np.uint16)
        # 使用 np.add.reduce 对数组 arr 进行归约操作，将结果存入 out，预期结果应为 1000
        assert np.add.reduce(arr, out=out) == 1000
        # 将数组 arr 的前 10 个元素设置为 2
        arr[:10] = 2
        # 使用 np.multiply.reduce 对数组 arr 进行归约操作，将结果存入 out，预期结果应为 2 的 10 次方
        assert np.multiply.reduce(arr, out=out) == 2**10

        # 对于旧版数据类型，如果传递了 `out=` 参数，目前必须强制使用签名。
        # 下面的两条路径应该不同，没有 `dtype=` 的情况下，预期结果应为 `np.prod(arr.astype("f8")).astype("f4")`！

        # 创建一个包含 5 个元素的 int64 类型数组，所有元素初始化为 2^25-1
        arr = np.full(5, 2**25-1, dtype=np.int64)

        # 创建一个空的 float32 类型的数组作为输出参数 res
        res = np.zeros((), dtype=np.float32)
        # 如果传递了 `dtype=`，则强制计算结果为 float32
        single_res = np.zeros((), dtype=np.float32)
        np.multiply.reduce(arr, out=single_res, dtype=np.float32)
        # 断言单独计算的结果与 res 不相等
        assert single_res != res

    def test_reducelike_output_needs_identical_cast(self):
        # 检查简单的字节交换是否有效，主要测试在 reducelike 中是否要求描述符的一致性。
        
        # 创建一个包含 20 个元素的 float64 类型数组，所有元素初始化为 1.0
        arr = np.ones(20, dtype="f8")
        # 创建一个根据 arr 的字节顺序新建的空数组作为输出参数 out
        out = np.empty((), dtype=arr.dtype.newbyteorder())
        expected = np.add.reduce(arr)
        # 使用 np.add.reduce 将 arr 中的元素归约，并将结果存入 out
        np.add.reduce(arr, out=out)
        # 断言归约结果与预期结果相等
        assert_array_equal(expected, out)

        # 检查 reduceat：
        out = np.empty(2, dtype=arr.dtype.newbyteorder())
        expected = np.add.reduceat(arr, [0, 1])
        # 使用 np.add.reduceat 对 arr 执行归约操作，并将结果存入 out
        np.add.reduceat(arr, [0, 1], out=out)
        # 断言归约结果与预期结果相等
        assert_array_equal(expected, out)

        # 检查 accumulate：
        out = np.empty(arr.shape, dtype=arr.dtype.newbyteorder())
        expected = np.add.accumulate(arr)
        # 使用 np.add.accumulate 对 arr 执行累积操作，并将结果存入 out
        np.add.accumulate(arr, out=out)
        # 断言累积结果与预期结果相等
        assert_array_equal(expected, out)

    def test_reduce_noncontig_output(self):
        # 检查归约操作是否正确处理非连续的输出数组。
        #
        # gh-8036

        # 创建一个包含 7*13*8 个元素的 int16 类型数组，按指定方式重塑并切片
        x = np.arange(7*13*8, dtype=np.int16).reshape(7, 13, 8)
        x = x[4:6,1:11:6,1:5].transpose(1, 2, 0)
        # 创建一个包含 4*4 个元素的 int16 类型数组
        y_base = np.arange(4*4, dtype=np.int16).reshape(4, 4)
        # 对 y_base 进行切片和复制，创建 y
        y = y_base[::2,:]

        # 复制一份 y_base 的副本
        y_base_copy = y_base.copy()

        # 使用 np.add.reduce 对 x 进行归约操作，将结果存入 y 的副本，并指定轴为 2
        r0 = np.add.reduce(x, out=y.copy(), axis=2)
        # 再次使用 np.add.reduce 对 x 进行归约操作，将结果存入 y，并指定轴为 2
        r1 = np.add.reduce(x, out=y, axis=2)

        # 结果应该相等，且不应改变 y_base
        assert_equal(r0, r1)
        assert_equal(y_base[1,:], y_base_copy[1,:])
        assert_equal(y_base[3,:], y_base_copy[3,:])

    @pytest.mark.parametrize("with_cast", [True, False])
    def test_reduceat_and_accumulate_out_shape_mismatch(self, with_cast):
        # Should raise an error mentioning "shape" or "size"
        # 创建一个长度为5的NumPy数组
        arr = np.arange(5)
        # 创建一个维度不匹配的长度为3的NumPy数组
        out = np.arange(3)  # definitely wrong shape
        if with_cast:
            # 如果需要在输出上进行类型转换，则使用通用的NpyIter（非快速）路径
            out = out.astype(np.float64)

        # 断言np.add.reduceat会引发ValueError，并且错误信息中包含"shape"或"size"
        with pytest.raises(ValueError, match="(shape|size)"):
            np.add.reduceat(arr, [0, 3], out=out)

        # 断言np.add.accumulate会引发ValueError，并且错误信息中包含"shape"或"size"
        with pytest.raises(ValueError, match="(shape|size)"):
            np.add.accumulate(arr, out=out)

    @pytest.mark.parametrize('out_shape',
                             [(), (1,), (3,), (1, 1), (1, 3), (4, 3)])
    @pytest.mark.parametrize('keepdims', [True, False])
    @pytest.mark.parametrize('f_reduce', [np.add.reduce, np.minimum.reduce])
    def test_reduce_wrong_dimension_output(self, f_reduce, keepdims, out_shape):
        # 测试确保不会错误地广播维度
        # 参见gh-15144（以前np.add.reduce失败的情况）
        a = np.arange(12.).reshape(4, 3)
        # 创建一个与指定形状和dtype的空NumPy数组
        out = np.empty(out_shape, a.dtype)

        # 获取正确的输出
        correct_out = f_reduce(a, axis=0, keepdims=keepdims)
        if out_shape != correct_out.shape:
            # 如果输出形状与正确输出的形状不匹配，则断言会引发ValueError
            with assert_raises(ValueError):
                f_reduce(a, axis=0, out=out, keepdims=keepdims)
        else:
            # 否则，检查函数的返回结果，并断言它与out相等
            check = f_reduce(a, axis=0, out=out, keepdims=keepdims)
            assert_(check is out)
            assert_array_equal(check, correct_out)

    def test_reduce_output_does_not_broadcast_input(self):
        # 测试输出形状不能广播输入维度
        # （它永远不能增加维度，但可能会扩展现有的维度）
        a = np.ones((1, 10))
        out_correct = (np.empty((1, 1)))
        out_incorrect = np.empty((3, 1))
        np.add.reduce(a, axis=-1, out=out_correct, keepdims=True)
        np.add.reduce(a, axis=-1, out=out_correct[:, 0], keepdims=False)
        # 断言使用out_incorrect作为输出会引发ValueError
        with assert_raises(ValueError):
            np.add.reduce(a, axis=-1, out=out_incorrect, keepdims=True)
        with assert_raises(ValueError):
            np.add.reduce(a, axis=-1, out=out_incorrect[:, 0], keepdims=False)

    def test_reduce_output_subclass_ok(self):
        class MyArr(np.ndarray):
            pass

        out = np.empty(())
        np.add.reduce(np.ones(5), out=out)  # no subclass, all fine
        out = out.view(MyArr)
        # 断言np.add.reduce返回的类型是MyArr的实例
        assert np.add.reduce(np.ones(5), out=out) is out
        assert type(np.add.reduce(out)) is MyArr

    def test_no_doc_string(self):
        # gh-9337
        # 断言inner1d_no_doc的__doc__中不包含换行符
        assert_('\n' not in umt.inner1d_no_doc.__doc__)

    def test_invalid_args(self):
        # gh-7961
        # 断言调用np.sqrt(None)会引发TypeError异常，并且异常文本包含特定信息
        exc = pytest.raises(TypeError, np.sqrt, None)
        assert exc.match('loop of ufunc does not support')

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    # 定义一个测试函数，用于验证给定的值 nat 是否非有限数
    def test_nat_is_not_finite(self, nat):
        try:
            # 使用 NumPy 函数 np.isfinite() 检查 nat 是否为有限数
            assert not np.isfinite(nat)
        except TypeError:
            # 如果出现 TypeError 异常，表示该操作可能尚未实现，这种情况下不做处理
            pass  # ok, just not implemented
    
    # 使用 pytest 的参数化装饰器，针对不同的 nat 值进行多次测试
    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_nan(self, nat):
        try:
            # 使用 NumPy 函数 np.isnan() 检查 nat 是否为 NaN (Not a Number)
            assert np.isnan(nat)
        except TypeError:
            # 如果出现 TypeError 异常，表示该操作可能尚未实现，这种情况下不做处理
            pass  # ok, just not implemented
    
    # 使用 pytest 的参数化装饰器，针对不同的 nat 值进行多次测试
    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_not_inf(self, nat):
        try:
            # 使用 NumPy 函数 np.isinf() 检查 nat 是否为无穷大或无穷小
            assert not np.isinf(nat)
        except TypeError:
            # 如果出现 TypeError 异常，表示该操作可能尚未实现，这种情况下不做处理
            pass  # ok, just not implemented
# 使用 pytest 的 parametrize 装饰器来对所有 NumPy ufunc 进行参数化测试
@pytest.mark.parametrize('ufunc', [getattr(np, x) for x in dir(np)
                                if isinstance(getattr(np, x), np.ufunc)])
def test_ufunc_types(ufunc):
    '''
    检查所有的 ufunc 返回正确的类型。避免使用对象和布尔类型，因为很多操作对它们不适用。

    选择形状，以便即使 dot 和 matmul 也能成功
    '''
    for typ in ufunc.types:
        # types 是类似 'ii->i' 的字符串列表
        if 'O' in typ or '?' in typ:
            continue
        inp, out = typ.split('->')
        args = [np.ones((3, 3), t) for t in inp]
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            res = ufunc(*args)
        if isinstance(res, tuple):
            outs = tuple(out)
            assert len(res) == len(outs)
            for r, t in zip(res, outs):
                assert r.dtype == np.dtype(t)
        else:
            assert res.dtype == np.dtype(out)

# 使用 pytest 的 parametrize 装饰器来对所有 NumPy ufunc 进行参数化测试，并禁用 NEP50 警告
@pytest.mark.parametrize('ufunc', [getattr(np, x) for x in dir(np)
                                if isinstance(getattr(np, x), np.ufunc)])
@np._no_nep50_warning()
def test_ufunc_noncontiguous(ufunc):
    '''
    检查对 ufunc 的连续和非连续调用，在值范围为 1 到 6 的情况下应有相同的结果
    '''
    for typ in ufunc.types:
        # types 是类似 'ii->i' 的字符串列表
        if any(set('O?mM') & set(typ)):
            # 布尔值、对象、日期时间类型在这个简单测试中太不规则，跳过
            continue
        inp, out = typ.split('->')
        args_c = [np.empty(6, t) for t in inp]
        args_n = [np.empty(18, t)[::3] for t in inp]
        for a in args_c:
            a.flat = range(1,7)
        for a in args_n:
            a.flat = range(1,7)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            res_c = ufunc(*args_c)
            res_n = ufunc(*args_n)
        if len(out) == 1:
            res_c = (res_c,)
            res_n = (res_n,)
        for c_ar, n_ar in zip(res_c, res_n):
            dt = c_ar.dtype
            if np.issubdtype(dt, np.floating):
                # 对于浮点数结果，在比较中允许小的误差，因为不同的算法（libm vs. intrinsics）可能会使用不同的输入步长
                res_eps = np.finfo(dt).eps
                tol = 2 * res_eps
                assert_allclose(res_c, res_n, atol=tol, rtol=tol)
            else:
                assert_equal(c_ar, n_ar)

# 使用 pytest 的 parametrize 装饰器来对 np.sign 和 np.equal 进行参数化测试
@pytest.mark.parametrize('ufunc', [np.sign, np.equal])
def test_ufunc_warn_with_nan(ufunc):
    # issue gh-15127
    # 测试使用非标准的 `nan` 值调用某些 ufuncs 不会发出警告
    # `b` 包含一个 64 位的信号 NaN 值：尾数的最高位为零
    b = np.array([0x7ff0000000000001], 'i8').view('f8')
    assert np.isnan(b)
    if ufunc.nin == 1:
        ufunc(b)
    # 如果输入函数（ufunc）的输入参数个数是2，执行以下操作
    elif ufunc.nin == 2:
        # 使用ufunc对b数组进行操作，并传入b的副本作为第二个参数
        ufunc(b, b.copy())
    # 如果输入函数（ufunc）的输入参数个数不是2，则抛出数值错误异常
    else:
        raise ValueError('ufunc with more than 2 inputs')
# 使用 pytest 的装饰器标记此测试函数，如果没有引用计数（refcount），则跳过测试，理由是 Python 缺乏引用计数。
@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_ufunc_out_casterrors():
    # 测试确保正确报告类型转换错误并清除缓冲区。
    
    # 定义一个整数值 123，依赖于 Python 的缓存（内存泄漏检查仍然会找到它）。
    value = 123  
    # 创建一个包含多个元素的数组 arr，数组中包括整数值的重复、字符串 "string" 和整数值的重复，数据类型为对象（dtype=object）。
    arr = np.array([value] * int(ncu.BUFSIZE * 1.5) +
                   ["string"] +
                   [value] * int(1.5 * ncu.BUFSIZE), dtype=object)
    # 创建一个与 arr 长度相同的全为1的整数数组 out，数据类型为 np.intp。
    out = np.ones(len(arr), dtype=np.intp)

    # 记录当前 value 的引用计数
    count = sys.getrefcount(value)
    
    # 使用 pytest 断言语句，期望抛出 ValueError 异常：
    with pytest.raises(ValueError):
        # 执行 np.add 操作，将 arr 与自身相加，将结果存储在 out 中，指定转换方式为 "unsafe"。
        np.add(arr, arr, out=out, casting="unsafe")

    # 断言当前 value 的引用计数与记录的值相同
    assert count == sys.getrefcount(value)
    
    # 断言发生错误后 out 数组的最后一个元素仍为 1，表明迭代在错误发生时被中止（这不一定是定义良好的行为）。
    assert out[-1] == 1

    # 期望抛出 ValueError 异常：
    with pytest.raises(ValueError):
        # 执行 np.add 操作，将 arr 与自身相加，将结果存储在 out 中，指定数据类型为 np.intp，转换方式为 "unsafe"。
        np.add(arr, arr, out=out, dtype=np.intp, casting="unsafe")

    # 再次断言当前 value 的引用计数与记录的值相同
    assert count == sys.getrefcount(value)
    
    # 再次断言发生错误后 out 数组的最后一个元素仍为 1，表明迭代在错误发生时被中止（这不一定是定义良好的行为）。
    assert out[-1] == 1


# 使用 pytest 的装饰器标记此测试函数，参数化测试函数 bad_offset 的值为 [0, int(ncu.BUFSIZE * 1.5)]。
@pytest.mark.parametrize("bad_offset", [0, int(ncu.BUFSIZE * 1.5)])
def test_ufunc_input_casterrors(bad_offset):
    # 测试强制类型转换输入时的报错情况，但缓冲区中将 arr 转换为 intp 类型失败。

    # 定义一个整数值 123
    value = 123
    # 创建一个包含多个元素的数组 arr，数组中包括整数值的重复、字符串 "string" 和整数值的重复，数据类型为对象（dtype=object）。
    arr = np.array([value] * bad_offset +
                   ["string"] +
                   [value] * int(1.5 * ncu.BUFSIZE), dtype=object)
    
    # 期望抛出 ValueError 异常：
    with pytest.raises(ValueError):
        # 执行 np.add 操作，将 arr 与自身相加，指定数据类型为 np.intp，转换方式为 "unsafe"。
        np.add(arr, arr, dtype=np.intp, casting="unsafe")


# 使用 pytest 的装饰器标记此测试函数，如果运行环境是 WASM，则跳过测试。
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
# 参数化测试函数 bad_offset 的值为 [0, int(ncu.BUFSIZE * 1.5)]。
@pytest.mark.parametrize("bad_offset", [0, int(ncu.BUFSIZE * 1.5)])
def test_ufunc_input_floatingpoint_error(bad_offset):
    # 测试浮点错误在输入时的报错情况。

    # 定义一个整数值 123
    value = 123
    # 创建一个包含多个元素的数组 arr，数组中包括整数值的重复、NaN 和整数值的重复。
    arr = np.array([value] * bad_offset +
                   [np.nan] +
                   [value] * int(1.5 * ncu.BUFSIZE))
    
    # 在 np.add 操作中设置浮点错误状态为 "raise"，期望抛出 FloatingPointError 异常。
    with np.errstate(invalid="raise"), pytest.raises(FloatingPointError):
        # 执行 np.add 操作，将 arr 与自身相加，指定数据类型为 np.intp，转换方式为 "unsafe"。
        np.add(arr, arr, dtype=np.intp, casting="unsafe")


# 定义一个测试函数，用于测试 "invalid cast" 的快速路径，参见 gh-19904。
def test_trivial_loop_invalid_cast():
    # 使用 pytest 断言语句，期望抛出 TypeError 异常，并匹配错误信息 "cast ufunc 'add' input 0"。
    with pytest.raises(TypeError,
            match="cast ufunc 'add' input 0"):
        # 执行 np.add 操作，将一个包含整数 1 的数组与整数 3 相加，指定签名为 "dd->d"。
        # 这里 void dtype 明显无法转换为 double。
        np.add(np.array(1, "i,i"), 3, signature="dd->d")


# 使用 pytest 的装饰器标记此测试函数，如果没有引用计数（refcount），则跳过测试，理由是 Python 缺乏引用计数。
@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
# 参数化测试函数 offset 的值为 [0, ncu.BUFSIZE//2, int(1.5*ncu.BUFSIZE)]。
@pytest.mark.parametrize("offset",
        [0, ncu.BUFSIZE//2, int(1.5*ncu.BUFSIZE)])
def test_reduce_casterrors(offset):
    # 测试在减少操作中报告转换错误，测试不同的偏移量，因为这些错误可能发生在减少过程的不同位置。
    
    # 测试内容已省略，以示简洁。
    # value 变量赋值为 123，依赖于 Python 的缓存（内存泄漏检查仍会发现它）
    value = 123  # relies on python cache (leak-check will still find it)
    # 创建一个包含特定元素的 numpy 数组，元素包括 value * offset 次的 value，
    # 一个字符串 "string"，以及 1.5 倍 ncu.BUFSIZE 次的 value，dtype 设置为 object 类型
    arr = np.array([value] * offset +
                   ["string"] +
                   [value] * int(1.5 * ncu.BUFSIZE), dtype=object)
    # 创建一个仅包含一个元素为 -1 的 numpy 数组，数据类型为 np.intp
    out = np.array(-1, dtype=np.intp)

    # 获取 value 的引用计数
    count = sys.getrefcount(value)
    # 使用 pytest 的上下文管理器检测是否会抛出 ValueError 异常，并匹配错误消息 "invalid literal"
    with pytest.raises(ValueError, match="invalid literal"):
        # 这是一个不安全的类型转换，但我们目前总是允许这样做。
        # 注意，双重循环被选择，但是转换失败了。
        # `initial=None` 禁用了这里的身份使用，以测试失败情况，
        # 在复制第一个值路径时不使用身份（当存在身份时不使用）。
        np.add.reduce(arr, dtype=np.intp, out=out, initial=None)
    # 断言检查：value 的引用计数没有改变
    assert count == sys.getrefcount(value)
    # 如果在转换过程中发生错误，则操作最多进行到错误发生的地方
    # （其结果将是 `value * offset`），如果立即发生错误，则结果为 -1。
    # 这不定义行为，输出是无效的，因此是未定义的。
    assert out[()] < value * offset
# 测试函数，用于验证在失败时的对象清理行为
def test_object_reduce_cleanup_on_failure():
    # 断言 TypeError 异常被抛出，验证初始值的清理是否有效
    with pytest.raises(TypeError):
        # 使用初始值 4，尝试对数组 [1, 2, None] 执行 np.add.reduce 操作
        np.add.reduce([1, 2, None], initial=4)

    with pytest.raises(TypeError):
        # 尝试对数组 [1, 2, None] 执行 np.add.reduce 操作，期望抛出 TypeError 异常
        np.add.reduce([1, 2, None])


@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.parametrize("method",
        [np.add.accumulate, np.add.reduce,
         pytest.param(lambda x: np.add.reduceat(x, [0]), id="reduceat"),
         pytest.param(lambda x: np.log.at(x, [2]), id="at")])
def test_ufunc_methods_floaterrors(method):
    # 创建包含 np.inf, 0, -np.inf 的数组 arr
    arr = np.array([np.inf, 0, -np.inf])
    # 在错误状态下执行 method(arr)，期望触发 RuntimeWarning 警告，警告信息包含 "invalid value"
    with np.errstate(all="warn"):
        with pytest.warns(RuntimeWarning, match="invalid value"):
            method(arr)

    arr = np.array([np.inf, 0, -np.inf])
    # 在错误状态下执行 method(arr)，期望触发 FloatingPointError 异常
    with np.errstate(all="raise"):
        with pytest.raises(FloatingPointError):
            method(arr)


def _check_neg_zero(value):
    # 检查给定的值是否为 -0.0
    if value != 0.0:
        return False
    # 检查实部是否为负零
    if not np.signbit(value.real):
        return False
    # 对于复数类型，检查虚部是否为负零
    if value.dtype.kind == "c":
        return np.signbit(value.imag)
    return True

@pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
def test_addition_negative_zero(dtype):
    # 确定数据类型，并创建对应的负零值
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        neg_zero = dtype.type(complex(-0.0, -0.0))
    else:
        neg_zero = dtype.type(-0.0)

    # 创建数组 arr 和 arr2，分别包含负零值
    arr = np.array(neg_zero)
    arr2 = np.array(neg_zero)

    # 断言 arr + arr2 的结果是负零
    assert _check_neg_zero(arr + arr2)
    # 在原地操作中，结果可能会走不同的路径，参考 gh-21211
    arr += arr2
    # 再次断言 arr 的值是否为负零
    assert _check_neg_zero(arr)


@pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
@pytest.mark.parametrize("use_initial", [True, False])
def test_addition_reduce_negative_zero(dtype, use_initial):
    # 确定数据类型，并创建对应的负零值
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        neg_zero = dtype.type(complex(-0.0, -0.0))
    else:
        neg_zero = dtype.type(-0.0)

    kwargs = {}
    if use_initial:
        kwargs["initial"] = neg_zero
    else:
        pytest.xfail("-0. propagation in sum currently requires initial")

    # 测试多种数组长度，以验证 SIMD 路径或分块是否起作用
    for i in range(0, 150):
        arr = np.array([neg_zero] * i, dtype=dtype)
        # 执行 np.sum(arr, **kwargs)，获取其结果 res
        res = np.sum(arr, **kwargs)
        if i > 0 or use_initial:
            # 断言 res 的值是否为负零
            assert _check_neg_zero(res)
        else:
            # 对于空数组，sum([]) 的结果应该是 0.0 而不是 -0.0，与 sum([-0.0]) 不同
            assert not np.signbit(res.real)
            assert not np.signbit(res.imag)


@pytest.mark.parametrize(["dt1", "dt2"],
        [("S", "U"), ("U", "S"), ("S", "d"), ("S", "V"), ("U", "l")])
def test_addition_string_types(dt1, dt2):
    # 创建不同类型的数组 arr1 和 arr2
    arr1 = np.array([1234234], dtype=dt1)
    arr2 = np.array([b"423"], dtype=dt2)
    # 使用 pytest 模块来测试 np.add 函数是否会引发 np._core._exceptions.UFuncTypeError 异常
    with pytest.raises(np._core._exceptions.UFuncTypeError) as exc:
        # 调用 np.add 函数，尝试将 arr1 和 arr2 数组相加
        np.add(arr1, arr2)
# 使用 pytest 的参数化功能为以下测试用例提供多组参数进行测试
@pytest.mark.parametrize("order1,order2",
                         [(">", ">"), ("<", "<"), (">", "<"), ("<", ">")])
def test_addition_unicode_inverse_byte_order(order1, order2):
    # 定义字符串元素
    element = 'abcd'
    # 创建指定类型的数组 arr1 和 arr2
    arr1 = np.array([element], dtype=f"{order1}U4")
    arr2 = np.array([element], dtype=f"{order2}U4")
    # 对 arr1 和 arr2 进行加法运算
    result = arr1 + arr2
    # 断言结果等于二倍的 element
    assert result == 2*element


# 使用 pytest 的参数化功能为以下测试用例提供多组参数进行测试
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_find_non_long_args(dtype):
    # 定义字符串元素
    element = 'abcd'
    # 定义起始位置和结束位置
    start = dtype(0)
    end = dtype(len(element))
    # 创建包含元素的数组 arr
    arr = np.array([element])
    # 使用 np._core.umath.find 方法查找元素在数组中的位置
    result = np._core.umath.find(arr, "a", start, end)
    # 断言结果的数据类型为 np.dtype("intp")
    assert result.dtype == np.dtype("intp")
    # 断言结果等于 0
    assert result == 0


def test_find_access_past_buffer():
    # 这个测试检查在 string_fastsearch.h 中不会读取字符串缓冲区的末尾
    # 字符串缓冲区类会确保进行了检查
    # 如果要看到它的作用，可以移除缓冲区中的检查，这个测试在 valgrind 下会产生一个“Invalid read”错误
    # 创建包含元素的数组 arr
    arr = np.array([b'abcd', b'ebcd'])
    # 使用 np._core.umath.find 方法查找元素在数组中的位置
    result = np._core.umath.find(arr, b'cde', 0, np.iinfo(np.int64).max)
    # 断言所有结果等于 -1
    assert np.all(result == -1)


class TestLowlevelAPIAccess:
    def test_resolve_dtypes_basic(self):
        # 基本的 dtype 解析测试
        i4 = np.dtype("i4")
        f4 = np.dtype("f4")
        f8 = np.dtype("f8")
        # 调用 np.add.resolve_dtypes 方法进行解析
        r = np.add.resolve_dtypes((i4, f4, None))
        # 断言解析结果
        assert r == (f8, f8, f8)

        # Signature 使用与 ufunc 相同的逻辑进行解析（不那么严格）
        # 以下是“相同类型”强制转换，所以有效
        r = np.add.resolve_dtypes((
                i4, i4, None), signature=(None, None, "f4"))
        # 断言解析结果
        assert r == (f4, f4, f4)

        # 检查 NEP 50 “弱”升级
        r = np.add.resolve_dtypes((f4, int, None))
        # 断言解析结果
        assert r == (f4, f4, f4)

        # 抛出类型错误异常
        with pytest.raises(TypeError):
            np.add.resolve_dtypes((i4, f4, None), casting="no")

    def test_resolve_dtypes_comparison(self):
        i4 = np.dtype("i4")
        i8 = np.dtype("i8")
        b = np.dtype("?")
        # 调用 np.equal.resolve_dtypes 方法进行解析
        r = np.equal.resolve_dtypes((i4, i8, None))
        # 断言解析结果
        assert r == (i8, i8, b)

    def test_weird_dtypes(self):
        S0 = np.dtype("S0")
        # S0 在 NumPy 中经常被转换为 S1，但这里不是
        r = np.equal.resolve_dtypes((S0, S0, None))
        # 断言解析结果
        assert r == (S0, S0, np.dtype(bool))

        # Subarray dtypes 是奇怪的，可能无法完全工作，我们保留它们
        # 导致 TypeError （目前没有 void/structured 的 equal 循环）
        dts = np.dtype("10i")
        # 抛出类型错误异常
        with pytest.raises(TypeError):
            np.equal.resolve_dtypes((dts, dts, None))

    def test_resolve_dtypes_reduction(self):
        i2 = np.dtype("i2")
        default_int_ = np.dtype(np.int_)
        # 检查特殊的加法解析
        res = np.add.resolve_dtypes((None, i2, None), reduction=True)
        # 断言解析结果
        assert res == (default_int_, default_int_, default_int_)
    def test_resolve_dtypes_reduction_no_output(self):
        # 定义一个整数类型的数据类型对象 i4
        i4 = np.dtype("i4")
        # 使用 pytest 的断言检查是否会引发 TypeError 异常
        with pytest.raises(TypeError):
            # 调用 np.add.resolve_dtypes 函数，期望引发 TypeError 异常，因为 reduction=True 参数不合法
            np.add.resolve_dtypes((i4, i4, i4), reduction=True)

    @pytest.mark.parametrize("dtypes", [
            # 参数化测试用例，包含两个相同的整数类型 i 和 f 的组合
            (np.dtype("i"), np.dtype("i")),
            # 参数化测试用例，包含一个 None 和一个整数类型 i 和 f 的组合
            (None, np.dtype("i"), np.dtype("f")),
            # 参数化测试用例，包含一个整数类型 i 和一个 None 和一个整数类型 i4 的组合
            (np.dtype("i"), None, np.dtype("f")),
            # 参数化测试用例，包含三个字符串 "i4"，其中一个为 None
            ("i4", "i4", None)])
    def test_resolve_dtypes_errors(self, dtypes):
        # 使用 pytest 的断言检查是否会引发 TypeError 异常
        with pytest.raises(TypeError):
            # 调用 np.add.resolve_dtypes 函数，检查其参数是否合法
            np.add.resolve_dtypes(dtypes)

    def test_resolve_dtypes_reduction_errors(self):
        # 定义一个短整数类型的数据类型对象 i2
        i2 = np.dtype("i2")

        # 使用 pytest 的断言检查是否会引发 TypeError 异常
        with pytest.raises(TypeError):
            # 调用 np.add.resolve_dtypes 函数，检查其参数是否合法
            np.add.resolve_dtypes((None, i2, i2))

        # 使用 pytest 的断言检查是否会引发 TypeError 异常
        with pytest.raises(TypeError):
            # 调用 np.add.signature 函数，检查其参数是否合法
            np.add.signature((None, None, "i4"))

    @pytest.mark.skipif(not hasattr(ct, "pythonapi"),
            reason="`ctypes.pythonapi` required for capsule unpacking.")
    def test_loop_access(self):
        # 定义一个字符指针数组类型的数据类型对象 data_t
        data_t = ct.c_char_p * 2
        # 定义一个一维大小类型的数据类型对象 dim_t
        dim_t = ct.c_ssize_t * 1
        # 定义一个两维大小类型的数据类型对象 strides_t
        strides_t = ct.c_ssize_t * 2
        # 定义一个 C 函数指针类型 strided_loop_t
        strided_loop_t = ct.CFUNCTYPE(
                ct.c_int, ct.c_void_p, data_t, dim_t, strides_t, ct.c_void_p)

        # 定义一个调用信息结构体 call_info_t
        class call_info_t(ct.Structure):
            _fields_ = [
                ("strided_loop", strided_loop_t),
                ("context", ct.c_void_p),
                ("auxdata", ct.c_void_p),
                ("requires_pyapi", ct.c_byte),
                ("no_floatingpoint_errors", ct.c_byte),
            ]

        # 定义一个整数类型的数据类型对象 i4
        i4 = np.dtype("i4")
        # 调用 np.negative._resolve_dtypes_and_context 函数获取数据类型和上下文信息
        dt, call_info_obj = np.negative._resolve_dtypes_and_context((i4, i4))
        # 使用断言检查返回的数据类型 dt 是否与预期相同
        assert dt == (i4, i4)  # 可以在不进行类型转换的情况下使用

        # 填充其余信息：
        # 调用 np.negative._get_strided_loop 函数获取 strided_loop
        np.negative._get_strided_loop(call_info_obj)

        # 设置 PyCapsule_GetPointer 函数的返回类型为 void 指针
        ct.pythonapi.PyCapsule_GetPointer.restype = ct.c_void_p
        # 使用 PyCapsule_GetPointer 获取指向 call_info_obj 的指针
        call_info = ct.pythonapi.PyCapsule_GetPointer(
                ct.py_object(call_info_obj),
                ct.c_char_p(b"numpy_1.24_ufunc_call_info"))

        # 将 call_info 转换为 call_info_t 结构体指针，并获取其内容
        call_info = ct.cast(call_info, ct.POINTER(call_info_t)).contents

        # 创建一个包含 10 个元素的整数数组 arr
        arr = np.arange(10, dtype=i4)
        # 调用 call_info.strided_loop 函数执行循环访问
        call_info.strided_loop(
                call_info.context,
                data_t(arr.ctypes.data, arr.ctypes.data),
                arr.ctypes.shape,  # 这里是一个包含 10 个元素的 C 数组
                strides_t(arr.ctypes.strides[0], arr.ctypes.strides[0]),
                call_info.auxdata)

        # 直接调用了内部的负数循环，将 arr 中的元素取负
        assert_array_equal(arr, -np.arange(10, dtype=i4))

    @pytest.mark.parametrize("strides", [1, (1, 2, 3), (1, "2")])
    # 定义测试函数，用于测试 `_get_strided_loop` 在给定不良步长时是否引发异常
    def test__get_strided_loop_errors_bad_strides(self, strides):
        # 创建一个整数类型的数据类型对象
        i4 = np.dtype("i4")
        # 解析数据类型和调用信息
        dt, call_info = np.negative._resolve_dtypes_and_context((i4, i4))

        # 使用 pytest 断言检查是否引发 TypeError 异常，并匹配特定错误消息
        with pytest.raises(TypeError, match="fixed_strides.*tuple.*or None"):
            np.negative._get_strided_loop(call_info, fixed_strides=strides)

    # 定义测试函数，用于测试 `_get_strided_loop` 在不良调用信息时是否引发异常
    def test__get_strided_loop_errors_bad_call_info(self):
        # 创建一个整数类型的数据类型对象
        i4 = np.dtype("i4")
        # 解析数据类型和调用信息
        dt, call_info = np.negative._resolve_dtypes_and_context((i4, i4))

        # 使用 pytest 断言检查是否引发 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match="PyCapsule"):
            np.negative._get_strided_loop("not the capsule!")

        # 使用 pytest 断言检查是否引发 TypeError 异常，并匹配特定错误消息
        with pytest.raises(TypeError, match=".*incompatible context"):
            np.add._get_strided_loop(call_info)

        # 调用 `_get_strided_loop` 函数，没有期望异常被抛出
        np.negative._get_strided_loop(call_info)

        # 使用 pytest 断言检查是否引发 TypeError 异常
        with pytest.raises(TypeError):
            # 不能第二次调用 `_get_strided_loop` 函数:
            np.negative._get_strided_loop(call_info)

    # 定义测试函数，测试处理长数组时是否正确
    def test_long_arrays(self):
        # 创建一个形状为 (1029, 917) 的零数组，数据类型为单精度浮点数
        t = np.zeros((1029, 917), dtype=np.single)
        # 将数组中的第一个元素设置为 1
        t[0][0] = 1
        # 将数组中的某个元素（第 28 行，第 414 列）设置为 1
        t[28][414] = 1
        # 对数组中的元素应用余弦函数
        tc = np.cos(t)
        # 使用断言检查数组中两个特定位置的元素是否相等
        assert_equal(tc[0][0], tc[28][414])
```