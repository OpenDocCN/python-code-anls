# `D:\src\scipysrc\scipy\scipy\ndimage\tests\test_ni_support.py`

```
import pytest  # 导入 pytest 库

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from .._ni_support import _get_output  # 导入模块中的 _get_output 函数


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，用于多组参数化测试
    'dtype',  # 参数化变量名为 dtype
    [  # 参数化的值列表开始
        # 字符串类型的数据类型标识符
        'f4', 'float32', 'complex64', 'complex128',
        # 类型和数据类型标识符的混合使用
        np.float32, float, np.dtype('f4'),
        # 根据输入推导
        None,
    ],  # 参数化的值列表结束
)
def test_get_output_basic(dtype):
    shape = (2, 3)  # 定义形状为 (2, 3)

    input_ = np.zeros(shape, 'float32')  # 创建一个形状为 shape 的 float32 类型的全零数组作为输入

    # 如果 dtype 是 None，则从输入中推导出数据类型为 'float32'
    expected_dtype = 'float32' if dtype is None else dtype

    # 调用 _get_output 函数，返回的结果的数据类型为 dtype，形状从输入中获取
    result = _get_output(dtype, input_)
    assert result.shape == shape  # 断言结果的形状与输入的形状相同
    assert result.dtype == np.dtype(expected_dtype)  # 断言结果的数据类型与预期的数据类型相同

    # 调用 _get_output 函数，返回的结果的数据类型为 dtype，形状被显式指定为 (3, 2)，覆盖输入的形状
    result = _get_output(dtype, input_, shape=(3, 2))
    assert result.shape == (3, 2)  # 断言结果的形状为 (3, 2)
    assert result.dtype == np.dtype(expected_dtype)  # 断言结果的数据类型与预期的数据类型相同

    # 调用 _get_output 函数，输出结果预先分配为数组 output，直接返回 output
    output = np.zeros(shape, dtype)
    result = _get_output(output, input_)
    assert result is output  # 断言返回的结果与预先分配的数组 output 是同一个对象


def test_get_output_complex():
    shape = (2, 3)  # 定义形状为 (2, 3)

    input_ = np.zeros(shape)  # 创建一个形状为 shape 的全零数组作为输入

    # 如果 dtype 是 None，则推广输入类型为复数类型
    result = _get_output(None, input_, complex_output=True)
    assert result.shape == shape  # 断言结果的形状与输入的形状相同
    assert result.dtype == np.dtype('complex128')  # 断言结果的数据类型为复数类型 'complex128'

    # 显式指定数据类型为 float，推广数据类型为复数类型
    with pytest.warns(UserWarning, match='promoting specified output dtype to complex'):
        result = _get_output(float, input_, complex_output=True)
    assert result.shape == shape  # 断言结果的形状与输入的形状相同
    assert result.dtype == np.dtype('complex128')  # 断言结果的数据类型为复数类型 'complex128'

    # 使用字符串标识符 'complex64'，验证结果为复数类型 'complex64'
    result = _get_output('complex64', input_, complex_output=True)
    assert result.shape == shape  # 断言结果的形状与输入的形状相同
    assert result.dtype == np.dtype('complex64')  # 断言结果的数据类型为复数类型 'complex64'


def test_get_output_error_cases():
    input_ = np.zeros((2, 3), 'float32')  # 创建一个形状为 (2, 3)，数据类型为 'float32' 的全零数组作为输入

    # 两个不同的路径可以引发相同的错误
    with pytest.raises(RuntimeError, match='output must have complex dtype'):
        _get_output('float32', input_, complex_output=True)
    with pytest.raises(RuntimeError, match='output must have complex dtype'):
        _get_output(np.zeros((2, 3)), input_, complex_output=True)

    # 引发 RuntimeError 错误，要求输出必须具有数值数据类型
    with pytest.raises(RuntimeError, match='output must have numeric dtype'):
        _get_output('void', input_)

    # 引发 RuntimeError 错误，形状不正确
    with pytest.raises(RuntimeError, match='shape not correct'):
        _get_output(np.zeros((3, 2)), input_)
```