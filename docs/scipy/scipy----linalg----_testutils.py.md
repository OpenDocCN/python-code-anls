# `D:\src\scipysrc\scipy\scipy\linalg\_testutils.py`

```
import numpy as np  # 导入 NumPy 库

# 定义一个模拟矩阵类，用于封装数据
class _FakeMatrix:
    def __init__(self, data):
        self._data = data  # 初始化存储数据的成员变量
        self.__array_interface__ = data.__array_interface__  # 将数据接口存储在私有成员变量中


# 定义另一个模拟矩阵类，支持 __array__ 方法
class _FakeMatrix2:
    def __init__(self, data):
        self._data = data  # 初始化存储数据的成员变量

    def __array__(self, dtype=None, copy=None):
        if copy:
            return self._data.copy()  # 如果指定了复制，返回数据的副本
        return self._data  # 否则返回数据本身


def _get_array(shape, dtype):
    """
    根据给定的形状和数据类型生成一个测试用的数组。
    当形状为 NxN 时返回正定矩阵，为 2xN 时返回带状正定矩阵。
    """
    if len(shape) == 2 and shape[0] == 2:
        # 生成带状正定矩阵
        x = np.zeros(shape, dtype=dtype)
        x[0, 1:] = -1
        x[1] = 2
        return x
    elif len(shape) == 2 and shape[0] == shape[1]:
        # 生成正定矩阵
        x = np.zeros(shape, dtype=dtype)
        j = np.arange(shape[0])
        x[j, j] = 2
        x[j[:-1], j[:-1]+1] = -1
        x[j[:-1]+1, j[:-1]] = -1
        return x
    else:
        np.random.seed(1234)
        return np.random.randn(*shape).astype(dtype)


def _id(x):
    return x  # 返回输入的对象本身


def assert_no_overwrite(call, shapes, dtypes=None):
    """
    测试调用函数不会改变其输入参数。
    """
    if dtypes is None:
        dtypes = [np.float32, np.float64, np.complex64, np.complex128]

    for dtype in dtypes:
        for order in ["C", "F"]:
            for faker in [_id, _FakeMatrix, _FakeMatrix2]:
                orig_inputs = [_get_array(s, dtype) for s in shapes]  # 生成原始输入数据
                inputs = [faker(x.copy(order)) for x in orig_inputs]  # 使用 faker 函数复制数据并传入调用函数
                call(*inputs)  # 调用函数处理复制后的数据
                msg = f"call modified inputs [{dtype!r}, {faker!r}]"  # 出错时的消息
                for a, b in zip(inputs, orig_inputs):
                    np.testing.assert_equal(a, b, err_msg=msg)  # 断言调用后数据未被修改
```