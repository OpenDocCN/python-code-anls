# `D:\src\scipysrc\scipy\scipy\_lib\tests\test_array_api.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from scipy.conftest import array_api_compatible  # 从 scipy.conftest 模块导入 array_api_compatible 函数
from scipy._lib._array_api import (  # 导入 scipy._lib._array_api 模块的多个函数和变量
    _GLOBAL_CONFIG, array_namespace, _asarray, copy, xp_assert_equal, is_numpy
)
import scipy._lib.array_api_compat.numpy as np_compat  # 导入 scipy._lib.array_api_compat.numpy 模块，并重命名为 np_compat

skip_xp_backends = pytest.mark.skip_xp_backends  # 定义 skip_xp_backends 标记，用于跳过特定的后端测试


@pytest.mark.skipif(not _GLOBAL_CONFIG["SCIPY_ARRAY_API"],
                    reason="Array API test; set environment variable SCIPY_ARRAY_API=1 to run it")
class TestArrayAPI:
    # 测试类 TestArrayAPI，用于测试数组 API 相关功能

    def test_array_namespace(self):
        # 测试 array_namespace 函数的行为
        x, y = np.array([0, 1, 2]), np.array([0, 1, 2])  # 创建 NumPy 数组 x 和 y
        xp = array_namespace(x, y)  # 调用 array_namespace 函数
        assert 'array_api_compat.numpy' in xp.__name__  # 断言 xp 函数的名称中包含 'array_api_compat.numpy'

        _GLOBAL_CONFIG["SCIPY_ARRAY_API"] = False  # 设置全局配置的 SCIPY_ARRAY_API 为 False
        xp = array_namespace(x, y)  # 再次调用 array_namespace 函数
        assert 'array_api_compat.numpy' in xp.__name__  # 断言 xp 函数的名称中包含 'array_api_compat.numpy'
        _GLOBAL_CONFIG["SCIPY_ARRAY_API"] = True  # 恢复全局配置的 SCIPY_ARRAY_API 为 True

    @array_api_compatible
    def test_asarray(self, xp):
        # 使用 array_api_compatible 装饰器标记的测试函数 test_asarray
        x, y = _asarray([0, 1, 2], xp=xp), _asarray(np.arange(3), xp=xp)  # 使用 _asarray 函数将数组转换为 xp 所支持的数组格式
        ref = xp.asarray([0, 1, 2])  # 使用 xp.asarray 创建参考数组 ref
        xp_assert_equal(x, ref)  # 使用 xp_assert_equal 断言 x 与 ref 相等
        xp_assert_equal(y, ref)  # 使用 xp_assert_equal 断言 y 与 ref 相等

    @pytest.mark.filterwarnings("ignore: the matrix subclass")
    def test_raises(self):
        # 测试函数 test_raises，用于测试抛出异常的情况
        msg = "of type `numpy.ma.MaskedArray` are not supported"  # 设置异常消息
        with pytest.raises(TypeError, match=msg):  # 使用 pytest.raises 断言捕获 TypeError 异常并匹配异常消息
            array_namespace(np.ma.array(1), np.array(1))  # 调用 array_namespace 函数传入特定参数

        msg = "of type `numpy.matrix` are not supported"  # 设置异常消息
        with pytest.raises(TypeError, match=msg):  # 使用 pytest.raises 断言捕获 TypeError 异常并匹配异常消息
            array_namespace(np.array(1), np.matrix(1))  # 调用 array_namespace 函数传入特定参数

        msg = "only boolean and numerical dtypes are supported"  # 设置异常消息
        with pytest.raises(TypeError, match=msg):  # 使用 pytest.raises 断言捕获 TypeError 异常并匹配异常消息
            array_namespace([object()])  # 调用 array_namespace 函数传入特定参数
        with pytest.raises(TypeError, match=msg):  # 使用 pytest.raises 断言捕获 TypeError 异常并匹配异常消息
            array_namespace('abc')  # 调用 array_namespace 函数传入特定参数

    def test_array_likes(self):
        # 测试函数 test_array_likes，用于测试类似数组的行为
        # 应该不会引发异常
        array_namespace([0, 1, 2])  # 调用 array_namespace 函数传入列表参数
        array_namespace(1, 2, 3)  # 调用 array_namespace 函数传入多个参数
        array_namespace(1)  # 调用 array_namespace 函数传入单个参数

    @skip_xp_backends('jax.numpy',
                      reasons=["JAX arrays do not support item assignment"])
    @pytest.mark.usefixtures("skip_xp_backends")
    @array_api_compatible
    def test_copy(self, xp):
        # 使用 skip_xp_backends 和 array_api_compatible 装饰器标记的测试函数 test_copy
        for _xp in [xp, None]:  # 遍历 _xp 变量的可能取值
            x = xp.asarray([1, 2, 3])  # 使用 xp.asarray 创建数组 x
            y = copy(x, xp=_xp)  # 使用 copy 函数复制数组 x 到数组 y
            # with numpy we'd want to use np.shared_memory, but that's not specified
            # in the array-api
            x[0] = 10  # 修改数组 x 的元素
            x[1] = 11  # 修改数组 x 的元素
            x[2] = 12  # 修改数组 x 的元素

            assert x[0] != y[0]  # 断言 x 的第一个元素不等于 y 的第一个元素
            assert x[1] != y[1]  # 断言 x 的第二个元素不等于 y 的第二个元素
            assert x[2] != y[2]  # 断言 x 的第三个元素不等于 y 的第三个元素
            assert id(x) != id(y)  # 断言 x 和 y 的内存地址不同

    @array_api_compatible
    @pytest.mark.parametrize('dtype', ['int32', 'int64', 'float32', 'float64'])
    @pytest.mark.parametrize('shape', [(), (3,)])
    # 测试严格检查功能的方法，验证 `_strict_check` 的预期行为
    def test_strict_checks(self, xp, dtype, shape):
        # 将字符串表示的数据类型转换为对应的数据类型对象
        dtype = getattr(xp, dtype)
        # 将数值 1 转换为指定数据类型的数组，并广播到指定形状
        x = xp.broadcast_to(xp.asarray(1, dtype=dtype), shape)
        # 如果形状为空，则将数组 x 转换为标量
        x = x if shape else x[()]
        # 创建一个 NumPy 标量
        y = np_compat.asarray(1)[()]

        # 定义检查选项字典，用于控制断言函数的行为
        options = dict(check_namespace=True, check_dtype=False, check_shape=False)
        # 如果 xp 是 NumPy，则使用 xp_assert_equal 检查 x 和 y 是否相等
        if xp == np:
            xp_assert_equal(x, y, **options)
        else:
            # 否则，预期会抛出 AssertionError，并匹配指定的错误信息
            with pytest.raises(AssertionError, match="Namespaces do not match."):
                xp_assert_equal(x, y, **options)

        # 更新检查选项，仅检查数据类型
        options = dict(check_namespace=False, check_dtype=True, check_shape=False)
        # 如果 y 的数据类型名称包含在 x 的数据类型字符串中，则检查通过
        if y.dtype.name in str(x.dtype):
            xp_assert_equal(x, y, **options)
        else:
            # 否则，预期会抛出 AssertionError，并匹配指定的错误信息
            with pytest.raises(AssertionError, match="dtypes do not match."):
                xp_assert_equal(x, y, **options)

        # 更新检查选项，仅检查形状
        options = dict(check_namespace=False, check_dtype=False, check_shape=True)
        # 如果 x 和 y 的形状相同，则检查通过
        if x.shape == y.shape:
            xp_assert_equal(x, y, **options)
        else:
            # 否则，预期会抛出 AssertionError，并匹配指定的错误信息
            with pytest.raises(AssertionError, match="Shapes do not match."):
                xp_assert_equal(x, xp.asarray(y), allow_0d=True, **options)


    # 基于数组 API 兼容性装饰器的方法，测试标量检查功能
    @array_api_compatible
    def test_check_scalar(self, xp):
        # 如果 xp 不是 NumPy，则跳过测试
        if not is_numpy(xp):
            pytest.skip("Scalars only exist in NumPy")

        # 如果 xp 是 NumPy，则执行以下检查
        if is_numpy(xp):
            # 检查默认约定：不允许 0 维数组
            message = "Result is a NumPy 0d array. Many SciPy functions..."
            # 预期会抛出 AssertionError，并匹配指定的错误信息
            with pytest.raises(AssertionError, match=message):
                xp_assert_equal(xp.asarray(0.), xp.float64(0))
            with pytest.raises(AssertionError, match=message):
                xp_assert_equal(xp.asarray(0.), xp.asarray(0.))
            # 检查通过，x 和 y 是相等的浮点数
            xp_assert_equal(xp.float64(0), xp.asarray(0.))
            xp_assert_equal(xp.float64(0), xp.float64(0))

            # 检查 `allow_0d` 选项
            message = "Types do not match:\n..."
            # 预期会抛出 AssertionError，并匹配指定的错误信息
            with pytest.raises(AssertionError, match=message):
                xp_assert_equal(xp.asarray(0.), xp.float64(0), allow_0d=True)
            with pytest.raises(AssertionError, match=message):
                xp_assert_equal(xp.float64(0), xp.asarray(0.), allow_0d=True)
            # 检查通过，允许 0 维数组
            xp_assert_equal(xp.float64(0), xp.float64(0), allow_0d=True)
            xp_assert_equal(xp.asarray(0.), xp.asarray(0.), allow_0d=True)
```