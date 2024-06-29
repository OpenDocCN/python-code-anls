# `.\numpy\numpy\doc\ufuncs.py`

```py
"""
===================
Universal Functions
===================

Ufuncs are, generally speaking, mathematical functions or operations that are
applied element-by-element to the contents of an array. That is, the result
in each output array element only depends on the value in the corresponding
input array (or arrays) and on no other array elements. NumPy comes with a
large suite of ufuncs, and scipy extends that suite substantially. The simplest
example is the addition operator: ::

 >>> np.array([0,2,3,4]) + np.array([1,1,-1,2])
 array([1, 3, 2, 6])

The ufunc module lists all the available ufuncs in numpy. Documentation on
the specific ufuncs may be found in those modules. This documentation is
intended to address the more general aspects of ufuncs common to most of
them. All of the ufuncs that make use of Python operators (e.g., +, -, etc.)
have equivalent functions defined (e.g. add() for +)

Type coercion
=============

What happens when a binary operator (e.g., +,-,\\*,/, etc) deals with arrays of
two different types? What is the type of the result? Typically, the result is
the higher of the two types. For example: ::

 float32 + float64 -> float64
 int8 + int32 -> int32
 int16 + float32 -> float32
 float32 + complex64 -> complex64

There are some less obvious cases generally involving mixes of types
(e.g. uints, ints and floats) where equal bit sizes for each are not
capable of saving all the information in a different type of equivalent
bit size. Some examples are int32 vs float32 or uint32 vs int32.
Generally, the result is the higher type of larger size than both
(if available). So: ::

 int32 + float32 -> float64
 uint32 + int32 -> int64

Finally, the type coercion behavior when expressions involve Python
scalars is different than that seen for arrays. Since Python has a
limited number of types, combining a Python int with a dtype=np.int8
array does not coerce to the higher type but instead, the type of the
array prevails. So the rules for Python scalars combined with arrays is
that the result will be that of the array equivalent the Python scalar
if the Python scalar is of a higher 'kind' than the array (e.g., float
vs. int), otherwise the resultant type will be that of the array.
For example: ::

  Python int + int8 -> int8
  Python float + int8 -> float64

ufunc methods
=============

Binary ufuncs support 4 methods.

**.reduce(arr)** applies the binary operator to elements of the array in
  sequence. For example: ::

 >>> np.add.reduce(np.arange(10))  # adds all elements of array
 45

For multidimensional arrays, the first dimension is reduced by default: ::

 >>> np.add.reduce(np.arange(10).reshape(2,5))
     array([ 5,  7,  9, 11, 13])

The axis keyword can be used to specify different axes to reduce: ::

 >>> np.add.reduce(np.arange(10).reshape(2,5),axis=1)
 array([10, 35])

**.accumulate(arr)** applies the binary operator and generates an
equivalently shaped array that includes the accumulated amount for each
"""
# np.add.accumulate(np.arange(10))
# 生成一个累积和数组，从0开始累积加法操作，结果为 [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
>>> np.add.accumulate(np.arange(10))

# np.multiply.accumulate(np.arange(1,9))
# 生成一个累积乘积数组，从1开始累积乘法操作，结果为 [1, 2, 6, 24, 120, 720, 5040, 40320]
>>> np.multiply.accumulate(np.arange(1,9))

# .reduceat(arr,indices) 允许对数组的指定部分应用 reduce 操作。这是一个较难理解的方法，详细文档可见：
# https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduceat.html

# .outer(arr1,arr2) 在两个数组 arr1 和 arr2 上执行外积操作。适用于多维数组，结果数组的形状是两个输入形状的串联。
# 示例：np.multiply.outer(np.arange(3),np.arange(4)) 的结果为一个二维数组。
>>> np.multiply.outer(np.arange(3),np.arange(4))

# np.add(np.arange(2, dtype=float), np.arange(2, dtype=float), x, casting='unsafe')
# 将两个浮点数数组相加，结果存入数组 x。使用 'unsafe' 模式进行类型转换。
# 示例：np.arange(2) 和 np.arange(2, dtype=float) 的结果为 [0.0, 2.0]
>>> x = np.arange(2)
>>> np.add(np.arange(2, dtype=float), np.arange(2, dtype=float), x, casting='unsafe')

# 使用 and 和 or 作为 ufunc 的逻辑操作符时会导致错误，应该使用其对应的 ufunc 函数 logical_and() 和 logical_or()。
# 也可以使用位运算符 & 和 |，但如果操作数不是布尔数组，结果可能不正确。
```