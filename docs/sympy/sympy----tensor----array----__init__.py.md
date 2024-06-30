# `D:\src\scipysrc\sympy\sympy\tensor\array\__init__.py`

```
r"""
N-dim array module for SymPy.

Four classes are provided to handle N-dim arrays, given by the combinations
dense/sparse (i.e. whether to store all elements or only the non-zero ones in
memory) and mutable/immutable (immutable classes are SymPy objects, but cannot
change after they have been created).

Examples
========

The following examples show the usage of ``Array``. This is an abbreviation for
``ImmutableDenseNDimArray``, that is an immutable and dense N-dim array, the
other classes are analogous. For mutable classes it is also possible to change
element values after the object has been constructed.

Array construction can detect the shape of nested lists and tuples:

>>> from sympy import Array
>>> a1 = Array([[1, 2], [3, 4], [5, 6]])
>>> a1
[[1, 2], [3, 4], [5, 6]]
>>> a1.shape
(3, 2)
>>> a1.rank()
2
>>> from sympy.abc import x, y, z
>>> a2 = Array([[[x, y], [z, x*z]], [[1, x*y], [1/x, x/y]]])
>>> a2
[[[x, y], [z, x*z]], [[1, x*y], [1/x, x/y]]]
>>> a2.shape
(2, 2, 2)
>>> a2.rank()
3

Otherwise one could pass a 1-dim array followed by a shape tuple:

>>> m1 = Array(range(12), (3, 4))
>>> m1
[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
>>> m2 = Array(range(12), (3, 2, 2))
>>> m2
[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]
>>> m2[1,1,1]
7
>>> m2.reshape(4, 3)
[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

Slice support:

>>> m2[:, 1, 1]
[3, 7, 11]

Elementwise derivative:

>>> from sympy.abc import x, y, z
>>> m3 = Array([x**3, x*y, z])
>>> m3.diff(x)
[3*x**2, y, 0]
>>> m3.diff(z)
[0, 0, 1]

Multiplication with other SymPy expressions is applied elementwisely:

>>> (1+x)*m3
[x**3*(x + 1), x*y*(x + 1), z*(x + 1)]

To apply a function to each element of the N-dim array, use ``applyfunc``:

>>> m3.applyfunc(lambda x: x/2)
[x**3/2, x*y/2, z/2]

N-dim arrays can be converted to nested lists by the ``tolist()`` method:

>>> m2.tolist()
[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]
>>> isinstance(m2.tolist(), list)
True

If the rank is 2, it is possible to convert them to matrices with ``tomatrix()``:

>>> m1.tomatrix()
Matrix([
[0, 1,  2,  3],
[4, 5,  6,  7],
[8, 9, 10, 11]])

Products and contractions
-------------------------

Tensor product between arrays `A_{i_1,\ldots,i_n}` and `B_{j_1,\ldots,j_m}`
creates the combined array `P = A \otimes B` defined as

`P_{i_1,\ldots,i_n,j_1,\ldots,j_m} := A_{i_1,\ldots,i_n}\cdot B_{j_1,\ldots,j_m}.`

It is available through ``tensorproduct(...)``:

>>> from sympy import Array, tensorproduct
>>> from sympy.abc import x,y,z,t
>>> A = Array([x, y, z, t])
>>> B = Array([1, 2, 3, 4])
>>> tensorproduct(A, B)
[[x, 2*x, 3*x, 4*x], [y, 2*y, 3*y, 4*y], [z, 2*z, 3*z, 4*z], [t, 2*t, 3*t, 4*t]]

In case you don't want to evaluate the tensor product immediately, you can use
``ArrayTensorProduct``, which creates an unevaluated tensor product expression:

>>> from sympy.tensor.array.expressions import ArrayTensorProduct
>>> ArrayTensorProduct(A, B)

"""
ArrayTensorProduct([x, y, z, t], [1, 2, 3, 4])
# 创建一个张量积对象，将给定的两个数组进行张量积运算

Calling ``.as_explicit()`` on ``ArrayTensorProduct`` is equivalent to just calling
``tensorproduct(...)``:

>>> ArrayTensorProduct(A, B).as_explicit()
[[x, 2*x, 3*x, 4*x], [y, 2*y, 3*y, 4*y], [z, 2*z, 3*z, 4*z], [t, 2*t, 3*t, 4*t]]
# 调用 `.as_explicit()` 方法将 `ArrayTensorProduct` 转换为显式的张量积结果，等价于直接调用 `tensorproduct(...)`

Tensor product between a rank-1 array and a matrix creates a rank-3 array:

>>> from sympy import eye
>>> p1 = tensorproduct(A, eye(4))
>>> p1
[[[x, 0, 0, 0], [0, x, 0, 0], [0, 0, x, 0], [0, 0, 0, x]], [[y, 0, 0, 0], [0, y, 0, 0], [0, 0, y, 0], [0, 0, 0, y]], [[z, 0, 0, 0], [0, z, 0, 0], [0, 0, z, 0], [0, 0, 0, z]], [[t, 0, 0, 0], [0, t, 0, 0], [0, 0, t, 0], [0, 0, 0, t]]]
# 将一个一阶数组和一个矩阵进行张量积运算，创建一个三阶数组

Now, to get back `A_0 \otimes \mathbf{1}` one can access `p_{0,m,n}` by slicing:

>>> p1[0,:,:]
[[x, 0, 0, 0], [0, x, 0, 0], [0, 0, x, 0], [0, 0, 0, x]]
# 通过切片方式获取 `p1` 中的特定部分 `p1[0,:,:]`，即张量积的部分结果

Tensor contraction sums over the specified axes, for example contracting
positions `a` and `b` means

`A_{i_1,\ldots,i_a,\ldots,i_b,\ldots,i_n} \implies \sum_k A_{i_1,\ldots,k,\ldots,k,\ldots,i_n}`
# 张量收缩在指定的轴上求和，例如对位置 `a` 和 `b` 进行收缩的含义

Remember that Python indexing is zero starting, to contract the a-th and b-th
axes it is therefore necessary to specify `a-1` and `b-1`

>>> from sympy import tensorcontraction
>>> C = Array([[x, y], [z, t]])
# 创建一个二阶数组 `C`

The matrix trace is equivalent to the contraction of a rank-2 array:

`A_{m,n} \implies \sum_k A_{k,k}`

>>> tensorcontraction(C, (0, 1))
t + x
# 矩阵的迹等价于对二阶数组进行收缩运算

To create an expression representing a tensor contraction that does not get
evaluated immediately, use ``ArrayContraction``, which is equivalent to
``tensorcontraction(...)`` if it is followed by ``.as_explicit()``:

>>> from sympy.tensor.array.expressions import ArrayContraction
>>> ArrayContraction(C, (0, 1))
ArrayContraction([[x, y], [z, t]], (0, 1))
>>> ArrayContraction(C, (0, 1)).as_explicit()
t + x
# 使用 `ArrayContraction` 创建一个表示张量收缩的表达式，如果接着调用 `.as_explicit()`，则等价于调用 `tensorcontraction(...)`

Matrix product is equivalent to a tensor product of two rank-2 arrays, followed
by a contraction of the 2nd and 3rd axes (in Python indexing axes number 1, 2).

`A_{m,n}\cdot B_{i,j} \implies \sum_k A_{m, k}\cdot B_{k, j}`

>>> D = Array([[2, 1], [0, -1]])
>>> tensorcontraction(tensorproduct(C, D), (1, 2))
[[2*x, x - y], [2*z, -t + z]]
# 矩阵乘积等价于两个二阶数组的张量积，接着对第二个和第三个轴进行收缩运算

One may verify that the matrix product is equivalent:

>>> from sympy import Matrix
>>> Matrix([[x, y], [z, t]])*Matrix([[2, 1], [0, -1]])
Matrix([
[2*x,  x - y],
[2*z, -t + z]])

or equivalently

>>> C.tomatrix()*D.tomatrix()
Matrix([
[2*x,  x - y],
[2*z, -t + z]])

Diagonal operator
-----------------

The ``tensordiagonal`` function acts in a similar manner as ``tensorcontraction``,
but the joined indices are not summed over, for example diagonalizing
positions `a` and `b` means

`A_{i_1,\ldots,i_a,\ldots,i_b,\ldots,i_n} \implies A_{i_1,\ldots,k,\ldots,k,\ldots,i_n}
\implies \tilde{A}_{i_1,\ldots,i_{a-1},i_{a+1},\ldots,i_{b-1},i_{b+1},\ldots,i_n,k}`
# `tensordiagonal` 函数类似于 `tensorcontraction`，但不对连接的索引进行求和，例如对位置 `a` 和 `b` 进行对角化的含义

where `\tilde{A}` is the array equivalent to the diagonal of `A` at positions
`a` and `b` moved to the last index slot.

Compare the difference between contraction and diagonal operators:
# 导入 sympy 库中的 tensordiagonal 函数
from sympy import tensordiagonal
# 从 sympy.abc 模块导入变量 a, b, c, d
from sympy.abc import a, b, c, d
# 创建一个 2x2 的矩阵 m，其元素为变量 a, b, c, d
m = Matrix([[a, b], [c, d]])
# 对矩阵 m 进行张量收缩操作，指定轴 [0, 1]，返回结果是 a + d
tensorcontraction(m, [0, 1])
# 对矩阵 m 进行 tensordiagonal 操作，指定轴 [0, 1]，返回结果是 [a, d]
tensordiagonal(m, [0, 1])
```