# `.\numpy\numpy\lib\_user_array_impl.py`

```
"""
Container class for backward compatibility with NumArray.

The user_array.container class exists for backward compatibility with NumArray
and is not meant to be used in new code. If you need to create an array
container class, we recommend either creating a class that wraps an ndarray
or subclasses ndarray.

"""
from numpy._core import (
    array, asarray, absolute, add, subtract, multiply, divide,
    remainder, power, left_shift, right_shift, bitwise_and, bitwise_or,
    bitwise_xor, invert, less, less_equal, not_equal, equal, greater,
    greater_equal, shape, reshape, arange, sin, sqrt, transpose
)


class container:
    """
    container(data, dtype=None, copy=True)

    Standard container-class for easy multiple-inheritance.

    Methods
    -------
    copy
        Returns a copy of the container.
    tostring
        Converts the array to a string representation.
    byteswap
        Swaps the byte order of the array.
    astype
        Converts the array to the specified data type.

    """
    def __init__(self, data, dtype=None, copy=True):
        # Initialize the container with a NumPy array created from the provided data.
        self.array = array(data, dtype, copy=copy)

    def __repr__(self):
        # Returns a string representation of the container class and its array contents.
        if self.ndim > 0:
            return self.__class__.__name__ + repr(self.array)[len("array"):]
        else:
            return self.__class__.__name__ + "(" + repr(self.array) + ")"

    def __array__(self, t=None):
        # Converts the container's array to the specified data type if provided, otherwise returns as is.
        if t:
            return self.array.astype(t)
        return self.array

    # Array as sequence
    def __len__(self):
        # Returns the length of the container's array.
        return len(self.array)

    def __getitem__(self, index):
        # Retrieves an item or slice from the container's array.
        return self._rc(self.array[index])

    def __setitem__(self, index, value):
        # Sets an item or slice in the container's array.
        self.array[index] = asarray(value, self.dtype)

    def __abs__(self):
        # Returns the absolute values of the container's array elements.
        return self._rc(absolute(self.array))

    def __neg__(self):
        # Returns the negation of the container's array elements.
        return self._rc(-self.array)

    def __add__(self, other):
        # Adds another array or scalar to the container's array.
        return self._rc(self.array + asarray(other))

    __radd__ = __add__

    def __iadd__(self, other):
        # Performs in-place addition with another array or scalar.
        add(self.array, other, self.array)
        return self

    def __sub__(self, other):
        # Subtracts another array or scalar from the container's array.
        return self._rc(self.array - asarray(other))

    def __rsub__(self, other):
        # Subtracts the container's array from another array or scalar.
        return self._rc(asarray(other) - self.array)

    def __isub__(self, other):
        # Performs in-place subtraction with another array or scalar.
        subtract(self.array, other, self.array)
        return self

    def __mul__(self, other):
        # Multiplies the container's array by another array or scalar.
        return self._rc(multiply(self.array, asarray(other)))

    __rmul__ = __mul__

    def __imul__(self, other):
        # Performs in-place multiplication with another array or scalar.
        multiply(self.array, other, self.array)
        return self

    def __div__(self, other):
        # Divides the container's array by another array or scalar.
        return self._rc(divide(self.array, asarray(other)))

    def __rdiv__(self, other):
        # Divides another array or scalar by the container's array.
        return self._rc(divide(asarray(other), self.array))

    def __idiv__(self, other):
        # Performs in-place division with another array or scalar.
        divide(self.array, other, self.array)
        return self

    def __mod__(self, other):
        # Computes the remainder of the container's array divided by another array or scalar.
        return self._rc(remainder(self.array, other))

    def __rmod__(self, other):
        # Computes the remainder of another array or scalar divided by the container's array.
        return self._rc(remainder(other, self.array))

    def __imod__(self, other):
        # Performs in-place computation of remainder with another array or scalar.
        remainder(self.array, other, self.array)
        return self
    # 定义实例方法 __divmod__，实现对象自身数组与另一个对象的除法和取余操作
    def __divmod__(self, other):
        return (self._rc(divide(self.array, other)),
                self._rc(remainder(self.array, other)))

    # 定义反向实例方法 __rdivmod__，实现另一个对象与对象自身数组的除法和取余操作
    def __rdivmod__(self, other):
        return (self._rc(divide(other, self.array)),
                self._rc(remainder(other, self.array)))

    # 定义实例方法 __pow__，实现对象自身数组的幂运算
    def __pow__(self, other):
        return self._rc(power(self.array, asarray(other)))

    # 定义反向实例方法 __rpow__，实现另一个对象与对象自身数组的幂运算
    def __rpow__(self, other):
        return self._rc(power(asarray(other), self.array))

    # 定义增强赋值方法 __ipow__，实现对象自身数组的就地幂运算
    def __ipow__(self, other):
        power(self.array, other, self.array)
        return self

    # 定义实例方法 __lshift__，实现对象自身数组的左移位操作
    def __lshift__(self, other):
        return self._rc(left_shift(self.array, other))

    # 定义实例方法 __rshift__，实现对象自身数组的右移位操作
    def __rshift__(self, other):
        return self._rc(right_shift(self.array, other))

    # 定义反向实例方法 __rlshift__，实现另一个对象与对象自身数组的左移位操作
    def __rlshift__(self, other):
        return self._rc(left_shift(other, self.array))

    # 定义反向实例方法 __rrshift__，实现另一个对象与对象自身数组的右移位操作
    def __rrshift__(self, other):
        return self._rc(right_shift(other, self.array))

    # 定义增强赋值方法 __ilshift__，实现对象自身数组的就地左移位操作
    def __ilshift__(self, other):
        left_shift(self.array, other, self.array)
        return self

    # 定义增强赋值方法 __irshift__，实现对象自身数组的就地右移位操作
    def __irshift__(self, other):
        right_shift(self.array, other, self.array)
        return self

    # 定义实例方法 __and__，实现对象自身数组与另一个对象的按位与操作
    def __and__(self, other):
        return self._rc(bitwise_and(self.array, other))

    # 定义反向实例方法 __rand__，实现另一个对象与对象自身数组的按位与操作
    def __rand__(self, other):
        return self._rc(bitwise_and(other, self.array))

    # 定义增强赋值方法 __iand__，实现对象自身数组的就地按位与操作
    def __iand__(self, other):
        bitwise_and(self.array, other, self.array)
        return self

    # 定义实例方法 __xor__，实现对象自身数组与另一个对象的按位异或操作
    def __xor__(self, other):
        return self._rc(bitwise_xor(self.array, other))

    # 定义反向实例方法 __rxor__，实现另一个对象与对象自身数组的按位异或操作
    def __rxor__(self, other):
        return self._rc(bitwise_xor(other, self.array))

    # 定义增强赋值方法 __ixor__，实现对象自身数组的就地按位异或操作
    def __ixor__(self, other):
        bitwise_xor(self.array, other, self.array)
        return self

    # 定义实例方法 __or__，实现对象自身数组与另一个对象的按位或操作
    def __or__(self, other):
        return self._rc(bitwise_or(self.array, other))

    # 定义反向实例方法 __ror__，实现另一个对象与对象自身数组的按位或操作
    def __ror__(self, other):
        return self._rc(bitwise_or(other, self.array))

    # 定义增强赋值方法 __ior__，实现对象自身数组的就地按位或操作
    def __ior__(self, other):
        bitwise_or(self.array, other, self.array)
        return self

    # 定义实例方法 __pos__，实现对象自身数组的正运算
    def __pos__(self):
        return self._rc(self.array)

    # 定义实例方法 __invert__，实现对象自身数组的按位取反操作
    def __invert__(self):
        return self._rc(invert(self.array))

    # 定义私有方法 _scalarfunc，根据对象数组的维度返回相应的标量函数结果
    def _scalarfunc(self, func):
        if self.ndim == 0:
            return func(self[0])
        else:
            raise TypeError(
                "only rank-0 arrays can be converted to Python scalars.")

    # 定义实例方法 __complex__，实现对象自身数组的复数类型转换
    def __complex__(self):
        return self._scalarfunc(complex)

    # 定义实例方法 __float__，实现对象自身数组的浮点数类型转换
    def __float__(self):
        return self._scalarfunc(float)

    # 定义实例方法 __int__，实现对象自身数组的整数类型转换
    def __int__(self):
        return self._scalarfunc(int)

    # 定义实例方法 __hex__，实现对象自身数组的十六进制类型转换
    def __hex__(self):
        return self._scalarfunc(hex)

    # 定义实例方法 __oct__，实现对象自身数组的八进制类型转换
    def __oct__(self):
        return self._scalarfunc(oct)

    # 定义实例方法 __lt__，实现对象自身数组与另一个对象的小于比较操作
    def __lt__(self, other):
        return self._rc(less(self.array, other))

    # 定义实例方法 __le__，实现对象自身数组与另一个对象的小于等于比较操作
    def __le__(self, other):
        return self._rc(less_equal(self.array, other))

    # 定义实例方法 __eq__，实现对象自身数组与另一个对象的等于比较操作
    def __eq__(self, other):
        return self._rc(equal(self.array, other))

    # 定义实例方法 __ne__，实现对象自身数组与另一个对象的不等于比较操作
    def __ne__(self, other):
        return self._rc(not_equal(self.array, other))
    # 定义大于比较方法，返回数组中每个元素与给定值的比较结果
    def __gt__(self, other):
        return self._rc(greater(self.array, other))

    # 定义大于等于比较方法，返回数组中每个元素与给定值的比较结果
    def __ge__(self, other):
        return self._rc(greater_equal(self.array, other))

    # 复制数组
    def copy(self):
        ""
        return self._rc(self.array.copy())

    # 返回数组的二进制字符串表示
    def tostring(self):
        ""
        return self.array.tostring()

    # 返回数组的字节表示
    def tobytes(self):
        ""
        return self.array.tobytes()

    # 交换数组中的字节顺序
    def byteswap(self):
        ""
        return self._rc(self.array.byteswap())

    # 返回数组的副本，类型转换为指定类型
    def astype(self, typecode):
        ""
        return self._rc(self.array.astype(typecode))

    # 返回数组或数组副本
    def _rc(self, a):
        if len(shape(a)) == 0:
            return a
        else:
            return self.__class__(a)

    # 返回新创建的数组对象
    def __array_wrap__(self, *args):
        return self.__class__(args[0])

    # 设置对象属性值
    def __setattr__(self, attr, value):
        if attr == 'array':
            object.__setattr__(self, attr, value)
            return
        try:
            self.array.__setattr__(attr, value)
        except AttributeError:
            object.__setattr__(self, attr, value)

    # 在其它尝试失败后调用的方法，获取对象属性值
    def __getattr__(self, attr):
        if (attr == 'array'):
            return object.__getattribute__(self, attr)
        return self.array.__getattribute__(attr)
#############################################################
# Test of class container
#############################################################

# 检查是否处于主模块下执行，以便进行测试
if __name__ == '__main__':
    # 创建一个包含 10000 个元素的一维数组，然后将其重塑为 100x100 的二维数组
    temp = reshape(arange(10000), (100, 100))

    # 使用 container 类来包装二维数组 temp
    ua = container(temp)
    # 输出 ua 对象的属性列表
    print(dir(ua))
    # 打印数组的形状信息
    print(shape(ua), ua.shape)  # I have changed Numeric.py

    # 提取 ua 的一个子数组 ua_small，包含前三行和前五列的部分
    ua_small = ua[:3, :5]
    # 打印 ua_small 的内容
    print(ua_small)
    # 修改 ua_small 的第一个元素为 10，验证其对原数组 ua 的影响
    # 此处注释表明修改 ua_small 不应该影响 ua[0,0]
    ua_small[0, 0] = 10
    # 打印修改后的 ua_small 的第一个元素以及原始数组 ua 的第一个元素
    print(ua_small[0, 0], ua[0, 0])
    # 对 ua_small 应用数学函数操作
    print(sin(ua_small) / 3. * 6. + sqrt(ua_small ** 2))
    # 比较 ua_small 中的元素是否小于 103，并输出结果和类型
    print(less(ua_small, 103), type(less(ua_small, 103)))
    # 计算 ua_small 与一个形状相匹配的数组的乘积，并输出结果的类型
    print(type(ua_small * reshape(arange(15), shape(ua_small))))
    # 将 ua_small 重塑为 5x3 的数组，并打印结果
    print(reshape(ua_small, (5, 3)))
    # 对 ua_small 进行转置操作，并输出结果
    print(transpose(ua_small))
```