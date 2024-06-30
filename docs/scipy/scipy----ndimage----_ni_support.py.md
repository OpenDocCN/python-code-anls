# `D:\src\scipysrc\scipy\scipy\ndimage\_ni_support.py`

```
# 导入所需模块和库
from collections.abc import Iterable  # 导入Iterable抽象基类，用于判断对象是否可迭代
import operator  # 导入operator模块，用于函数操作符的函数
import warnings  # 导入warnings模块，用于警告处理
import numpy as np  # 导入NumPy库，并使用np作为别名

# 将扩展模式转换为对应的整数代码
def _extend_mode_to_code(mode):
    """Convert an extension mode to the corresponding integer code.
    """
    if mode == 'nearest':
        return 0  # 如果模式为'nearest'，返回整数代码0
    elif mode == 'wrap':
        return 1  # 如果模式为'wrap'，返回整数代码1
    elif mode in ['reflect', 'grid-mirror']:
        return 2  # 如果模式为'reflect'或'grid-mirror'，返回整数代码2
    elif mode == 'mirror':
        return 3  # 如果模式为'mirror'，返回整数代码3
    elif mode == 'constant':
        return 4  # 如果模式为'constant'，返回整数代码4
    elif mode == 'grid-wrap':
        return 5  # 如果模式为'grid-wrap'，返回整数代码5
    elif mode == 'grid-constant':
        return 6  # 如果模式为'grid-constant'，返回整数代码6
    else:
        raise RuntimeError('boundary mode not supported')  # 抛出运行时错误，表示不支持的边界模式

# 标准化序列参数，确保其与输入数组的维度匹配
def _normalize_sequence(input, rank):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    is_str = isinstance(input, str)  # 检查输入是否为字符串
    if not is_str and isinstance(input, Iterable):  # 如果不是字符串且是可迭代对象
        normalized = list(input)  # 将可迭代对象转换为列表
        if len(normalized) != rank:  # 检查列表长度是否与输入数组的维度(rank)相等
            err = "sequence argument must have length equal to input rank"
            raise RuntimeError(err)  # 如果长度不匹配，抛出运行时错误
    else:
        normalized = [input] * rank  # 如果输入是标量，则用该标量创建长度为rank的序列
    return normalized  # 返回标准化后的序列

# 获取输出数组，如果未提供输出数组，则创建一个全零数组
def _get_output(output, input, shape=None, complex_output=False):
    if shape is None:
        shape = input.shape  # 如果未提供形状参数，则使用输入数组的形状
    if output is None:
        if not complex_output:
            output = np.zeros(shape, dtype=input.dtype.name)  # 如果不需要复数输出，创建相同形状的零数组
        else:
            complex_type = np.promote_types(input.dtype, np.complex64)
            output = np.zeros(shape, dtype=complex_type)  # 如果需要复数输出，创建相同形状的零数组，类型为复数类型
    # 如果输出是类或者 NumPy 的数据类型对象，则将其视作输出的数据类型
    elif isinstance(output, (type, np.dtype)):
        # 如果要求复数输出并且输出的数据类型不是复数类型，则发出警告并提升输出数据类型为复数类型
        if complex_output and np.dtype(output).kind != 'c':
            warnings.warn("promoting specified output dtype to complex", stacklevel=3)
            output = np.promote_types(output, np.complex64)
        # 使用指定的数据类型创建一个形状为 shape 的全零数组
        output = np.zeros(shape, dtype=output)
    
    # 如果输出是字符串类型，则将其解释为 NumPy 的数据类型对象
    elif isinstance(output, str):
        # 将字符串类型输出解释为 NumPy 的数据类型对象
        output = np.dtype(output)
        # 如果要求复数输出但输出的数据类型不是复数类型，则抛出运行时错误
        if complex_output and output.kind != 'c':
            raise RuntimeError("output must have complex dtype")
        # 如果输出数据类型不是 NumPy 数字类型的子类，则抛出运行时错误
        elif not issubclass(output.type, np.number):
            raise RuntimeError("output must have numeric dtype")
        # 使用指定的数据类型创建一个形状为 shape 的全零数组
        output = np.zeros(shape, dtype=output)
    
    # 如果输出数组的形状与指定的形状 shape 不一致，则抛出运行时错误
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    
    # 如果要求复数输出但输出数组的数据类型不是复数类型，则抛出运行时错误
    elif complex_output and output.dtype.kind != 'c':
        raise RuntimeError("output must have complex dtype")
    
    # 返回处理后的输出数组
    return output
# 根据给定的轴参数检查并返回有效的轴索引元组
def _check_axes(axes, ndim):
    # 如果轴参数为 None，则返回包含所有维度范围的元组
    if axes is None:
        return tuple(range(ndim))
    # 如果轴参数是一个标量，则将其转换为包含该标量的元组
    elif np.isscalar(axes):
        axes = (operator.index(axes),)
    # 如果轴参数是可迭代对象（如列表或元组）
    elif isinstance(axes, Iterable):
        for ax in axes:
            # 将所有轴索引转换为整数类型
            axes = tuple(operator.index(ax) for ax in axes)
            # 检查每个轴索引是否在有效范围内，若超出范围则抛出异常
            if ax < -ndim or ax > ndim - 1:
                raise ValueError(f"specified axis: {ax} is out of range")
        # 对所有负数轴索引进行循环处理，确保其在有效范围内
        axes = tuple(ax % ndim if ax < 0 else ax for ax in axes)
    else:
        # 如果轴参数既不是整数也不是可迭代对象，则抛出值错误异常
        message = "axes must be an integer, iterable of integers, or None"
        raise ValueError(message)
    
    # 检查所有轴索引是否唯一，若有重复则抛出值错误异常
    if len(tuple(set(axes))) != len(axes):
        raise ValueError("axes must be unique")
    
    # 返回处理后的轴索引元组
    return axes
```