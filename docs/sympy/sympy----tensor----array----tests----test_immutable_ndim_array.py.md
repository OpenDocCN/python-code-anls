# `D:\src\scipysrc\sympy\sympy\tensor\array\tests\test_immutable_ndim_array.py`

```
# 从copy模块中导入copy函数
from copy import copy
# 导入ImmutableDenseNDimArray类，用于创建不可变的密集多维数组
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
# 导入Dict类，用于创建字典对象
from sympy.core.containers import Dict
# 导入diff函数，用于计算符号表达式的微分
from sympy.core.function import diff
# 导入Rational类，用于处理有理数
from sympy.core.numbers import Rational
# 导入S类，表示数学常量
from sympy.core.singleton import S
# 导入Symbol类和symbols函数，用于创建符号变量
from sympy.core.symbol import (Symbol, symbols)
# 导入SparseMatrix类，用于创建稀疏矩阵对象
from sympy.matrices import SparseMatrix
# 导入Indexed和IndexedBase类，用于创建索引对象和索引基类
from sympy.tensor.indexed import (Indexed, IndexedBase)
# 导入Matrix类，用于创建矩阵对象
from sympy.matrices import Matrix
# 导入ImmutableSparseNDimArray类，用于创建不可变的稀疏多维数组
from sympy.tensor.array.sparse_ndim_array import ImmutableSparseNDimArray
# 导入raises函数，用于检测是否抛出指定异常
from sympy.testing.pytest import raises


# 定义测试函数test_ndim_array_initiation，用于测试不可变多维数组的初始化和功能
def test_ndim_array_initiation():
    # 创建一个没有元素的不可变密集多维数组，形状为(0,)
    arr_with_no_elements = ImmutableDenseNDimArray([], shape=(0,))
    # 断言数组长度为0
    assert len(arr_with_no_elements) == 0
    # 断言数组的秩为1
    assert arr_with_no_elements.rank() == 1

    # 测试传入非法形状时是否会抛出异常
    raises(ValueError, lambda: ImmutableDenseNDimArray([0], shape=(0,)))
    raises(ValueError, lambda: ImmutableDenseNDimArray([1, 2, 3], shape=(0,)))
    raises(ValueError, lambda: ImmutableDenseNDimArray([], shape=()))

    # 测试传入非法形状时是否会抛出异常
    raises(ValueError, lambda: ImmutableSparseNDimArray([0], shape=(0,)))
    raises(ValueError, lambda: ImmutableSparseNDimArray([1, 2, 3], shape=(0,)))
    raises(ValueError, lambda: ImmutableSparseNDimArray([], shape=()))

    # 创建一个包含一个元素的不可变密集多维数组
    arr_with_one_element = ImmutableDenseNDimArray([23])
    # 断言数组长度为1
    assert len(arr_with_one_element) == 1
    # 断言数组第一个元素为23
    assert arr_with_one_element[0] == 23
    # 断言数组切片与原数组相同
    assert arr_with_one_element[:] == ImmutableDenseNDimArray([23])
    # 断言数组的秩为1
    assert arr_with_one_element.rank() == 1

    # 创建一个包含符号元素的不可变密集多维数组
    arr_with_symbol_element = ImmutableDenseNDimArray([Symbol('x')])
    # 断言数组长度为1
    assert len(arr_with_symbol_element) == 1
    # 断言数组第一个元素为符号变量x
    assert arr_with_symbol_element[0] == Symbol('x')
    # 断言数组切片与原数组相同
    assert arr_with_symbol_element[:] == ImmutableDenseNDimArray([Symbol('x')])
    # 断言数组的秩为1
    assert arr_with_symbol_element.rank() == 1

    # 创建一个全为零的不可变密集多维数组，长度为number5
    number5 = 5
    vector = ImmutableDenseNDimArray.zeros(number5)
    # 断言数组长度为number5
    assert len(vector) == number5
    # 断言数组形状为(number5,)
    assert vector.shape == (number5,)
    # 断言数组的秩为1
    assert vector.rank() == 1

    # 创建一个全为零的不可变稀疏多维数组，长度为number5
    vector = ImmutableSparseNDimArray.zeros(number5)
    # 断言数组长度为number5
    assert len(vector) == number5
    # 断言数组形状为(number5,)
    assert vector.shape == (number5,)
    # 断言数组的_sparse_array属性为Dict对象
    assert vector._sparse_array == Dict()
    # 断言数组的秩为1
    assert vector.rank() == 1

    # 创建一个指定形状的不可变密集多维数组，包含从0到3**4的连续整数
    n_dim_array = ImmutableDenseNDimArray(range(3**4), (3, 3, 3, 3,))
    # 断言数组长度为3 * 3 * 3 * 3
    assert len(n_dim_array) == 3 * 3 * 3 * 3
    # 断言数组形状为(3, 3, 3, 3)
    assert n_dim_array.shape == (3, 3, 3, 3)
    # 断言数组的秩为4
    assert n_dim_array.rank() == 4

    # 创建一个指定形状的不可变稀疏多维数组，全为零
    array_shape = (3, 3, 3, 3)
    sparse_array = ImmutableSparseNDimArray.zeros(*array_shape)
    # 断言_sparse_array属性为空字典
    assert len(sparse_array._sparse_array) == 0
    # 断言数组长度为3 * 3 * 3 * 3
    assert len(sparse_array) == 3 * 3 * 3 * 3
    # 断言数组形状与指定形状相同
    assert n_dim_array.shape == array_shape
    # 断言数组的秩为4
    assert n_dim_array.rank() == 4

    # 创建一个一维不可变密集多维数组，包含元素[2, 3, 1]
    one_dim_array = ImmutableDenseNDimArray([2, 3, 1])
    # 断言数组长度为3
    assert len(one_dim_array) == 3
    # 断言数组形状为(3,)
    assert one_dim_array.shape == (3,)
    # 断言数组的秩为1
    assert one_dim_array.rank() == 1
    # 断言数组转换为列表后与原列表相同
    assert one_dim_array.tolist() == [2, 3, 1]

    # 创建一个指定形状的不可变稀疏多维数组，全为零
    shape = (3, 3)
    array_with_many_args = ImmutableSparseNDimArray.zeros(*shape)
    # 断言数组长度为3 * 3
    assert len(array_with_many_args) == 3 * 3
    # 断言数组形状与指定形状相同
    assert array_with_many_args.shape == shape
    # 断言数组的第一个元素为0
    assert array_with_many_args[0, 0] == 0
    # 确保 array_with_many_args 是一个二维数组
    assert array_with_many_args.rank() == 2

    # 定义一个形状为 (3, 3) 的数组
    shape = (int(3), int(3))
    array_with_long_shape = ImmutableSparseNDimArray.zeros(*shape)
    # 确保数组的长度为 3 * 3 = 9
    assert len(array_with_long_shape) == 3 * 3
    # 确保数组的形状为 (3, 3)
    assert array_with_long_shape.shape == shape
    # 确保数组在索引 (0, 0) 处的值为 0
    assert array_with_long_shape[int(0), int(0)] == 0
    # 确保数组的秩为 2
    assert array_with_long_shape.rank() == 2

    # 创建一个长度为 5 的一维数组
    vector_with_long_shape = ImmutableDenseNDimArray(range(5), int(5))
    # 确保一维数组的长度为 5
    assert len(vector_with_long_shape) == 5
    # 确保一维数组的形状为 (5,)
    assert vector_with_long_shape.shape == (int(5),)
    # 确保一维数组的秩为 1
    assert vector_with_long_shape.rank() == 1
    # 尝试访问超出索引范围的元素，应该引发 ValueError 异常
    raises(ValueError, lambda: vector_with_long_shape[int(5)])

    # 导入符号 x
    from sympy.abc import x
    # 对于每种数组类型，创建一个秩为 0 的数组
    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        rank_zero_array = ArrayType(x)
        # 确保秩为 0 的数组长度为 1
        assert len(rank_zero_array) == 1
        # 确保秩为 0 的数组形状为空元组 ()
        assert rank_zero_array.shape == ()
        # 确保秩为 0 的数组的秩为 0
        assert rank_zero_array.rank() == 0
        # 确保秩为 0 的数组在空索引 () 处的值为 x
        assert rank_zero_array[()] == x
        # 尝试访问索引 0，应该引发 ValueError 异常
        raises(ValueError, lambda: rank_zero_array[0])
# 定义测试函数 test_reshape，用于测试数组的重塑操作
def test_reshape():
    # 创建一个长度为 50 的不可变密集多维数组，初始化值为 0 到 49
    array = ImmutableDenseNDimArray(range(50), 50)
    # 断言数组的形状为 (50,)
    assert array.shape == (50,)
    # 断言数组的秩为 1
    assert array.rank() == 1

    # 将数组重塑为形状为 (5, 5, 2) 的新数组
    array = array.reshape(5, 5, 2)
    # 断言数组的形状为 (5, 5, 2)
    assert array.shape == (5, 5, 2)
    # 断言数组的秩为 3
    assert array.rank() == 3
    # 断言数组的长度为 50
    assert len(array) == 50


# 定义测试函数 test_getitem，用于测试数组的索引和切片操作
def test_getitem():
    # 遍历不可变密集多维数组和不可变稀疏多维数组两种类型
    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        # 创建一个长度为 24 的数组，重塑为形状为 (2, 3, 4) 的新数组
        array = ArrayType(range(24)).reshape(2, 3, 4)
        # 断言数组转换为列表形式与预期列表相同
        assert array.tolist() == [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
        # 断言数组的索引 [0] 返回的结果符合预期的数组类型和值
        assert array[0] == ArrayType([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        # 断言数组的索引 [0, 0] 返回的结果符合预期的数组类型和值
        assert array[0, 0] == ArrayType([0, 1, 2, 3])
        
        # 循环遍历数组的每个元素，并依次断言其值与预期的递增序列相等
        value = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    assert array[i, j, k] == value
                    value += 1

    # 断言对超出数组边界的索引访问会引发 ValueError 异常
    raises(ValueError, lambda: array[3, 4, 5])
    raises(ValueError, lambda: array[3, 4, 5, 6])
    raises(ValueError, lambda: array[3, 4, 5, 3:4])


# 定义测试函数 test_iterator，用于测试数组的迭代操作
def test_iterator():
    # 创建一个长度为 4，形状为 (2, 2) 的不可变密集多维数组
    array = ImmutableDenseNDimArray(range(4), (2, 2))
    # 断言数组的索引 [0] 返回的结果符合预期的不可变密集多维数组类型和值
    assert array[0] == ImmutableDenseNDimArray([0, 1])
    # 断言数组的索引 [1] 返回的结果符合预期的不可变密集多维数组类型和值
    assert array[1] == ImmutableDenseNDimArray([2, 3])

    # 将数组重塑为形状为 (4,) 的新数组
    array = array.reshape(4)
    # 初始化迭代变量 j 为 0，依次断言迭代数组的每个元素与 j 相等
    j = 0
    for i in array:
        assert i == j
        j += 1


# 定义测试函数 test_sparse，用于测试稀疏数组的功能
def test_sparse():
    # 创建一个值为 [0, 0, 0, 1]，形状为 (2, 2) 的不可变稀疏多维数组
    sparse_array = ImmutableSparseNDimArray([0, 0, 0, 1], (2, 2))
    # 断言稀疏数组的长度为 4
    assert len(sparse_array) == 2 * 2
    # 断言稀疏数组内部使用的字典结构存储非零元素，其长度为 1
    assert len(sparse_array._sparse_array) == 1

    # 断言稀疏数组转换为列表形式与预期列表相同
    assert sparse_array.tolist() == [[0, 0], [0, 1]]

    # 使用 zip 函数遍历稀疏数组和预期列表，依次断言其元素类型和值相等
    for i, j in zip(sparse_array, [[0, 0], [0, 1]]):
        assert i == ImmutableSparseNDimArray(j)

    # 定义函数 sparse_assignment，尝试修改稀疏数组的元素，预期会引发 TypeError 异常
    def sparse_assignment():
        sparse_array[0, 0] = 123

    # 断言在调用 sparse_assignment 函数后，稀疏数组内部使用的字典结构仍然长度为 1
    assert len(sparse_array._sparse_array) == 1
    # 断言对稀疏数组的索引 [0, 0] 返回的结果为原始值 0
    raises(TypeError, sparse_assignment)
    assert len(sparse_array._sparse_array) == 1
    # 断言稀疏数组的索引 [0, 0] 返回的结果为原始值 0
    assert sparse_array[0, 0] == 0
    # 断言稀疏数组除以 0 返回的结果为形状为 (2, 2) 的新稀疏数组，包含 S.NaN 和 S.ComplexInfinity
    assert sparse_array/0 == ImmutableSparseNDimArray([[S.NaN, S.NaN], [S.NaN, S.ComplexInfinity]], (2, 2))

    # 对于大规模稀疏数组的相等性测试
    assert ImmutableSparseNDimArray.zeros(100000, 200000) == ImmutableSparseNDimArray.zeros(100000, 200000)

    # 测试稀疏数组的 __mul__ 和 __rmul__ 方法
    a = ImmutableSparseNDimArray({200001: 1}, (100000, 200000))
    assert a * 3 == ImmutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert 3 * a == ImmutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert a * 0 == ImmutableSparseNDimArray({}, (100000, 200000))
    assert 0 * a == ImmutableSparseNDimArray({}, (100000, 200000))

    # 测试稀疏数组的 __truediv__ 方法
    assert a/3 == ImmutableSparseNDimArray({200001: Rational(1, 3)}, (100000, 200000))

    # 测试稀疏数组的 __neg__ 方法
    assert -a == ImmutableSparseNDimArray({200001: -1}, (100000, 200000))


# 定义测试函数 test_calculation，用于测试数组的数学运算
def test_calculation():
    # 创建两个形状为 (3, 3) 的不可变密集多维数组 a 和 b
    a = ImmutableDenseNDimArray([1]*9, (3, 3))
    b = ImmutableDenseNDimArray([9]*9, (3, 3))

    # 对数组 a 和 b 进行加法运算，结果赋给 c
    c = a + b
    # 对列表 c 中的每个元素进行断言，确保每个元素都等于 ImmutableDenseNDimArray([10, 10, 10])
    for i in c:
        assert i == ImmutableDenseNDimArray([10, 10, 10])
    
    # 断言整个列表 c 等于 ImmutableDenseNDimArray([10]*9, (3, 3))，即一个维度为 (3, 3) 的 9 个元素全为 10 的数组
    assert c == ImmutableDenseNDimArray([10]*9, (3, 3))
    
    # 断言整个列表 c 等于 ImmutableSparseNDimArray([10]*9, (3, 3))，即一个稀疏数组，维度为 (3, 3)，9 个元素全为 10
    assert c == ImmutableSparseNDimArray([10]*9, (3, 3))
    
    # 计算数组 b 和数组 a 的差，结果存入数组 c
    c = b - a
    
    # 对列表 c 中的每个元素进行断言，确保每个元素都等于 ImmutableDenseNDimArray([8, 8, 8])
    for i in c:
        assert i == ImmutableDenseNDimArray([8, 8, 8])
    
    # 断言整个列表 c 等于 ImmutableDenseNDimArray([8]*9, (3, 3))，即一个维度为 (3, 3) 的 9 个元素全为 8 的数组
    assert c == ImmutableDenseNDimArray([8]*9, (3, 3))
    
    # 断言整个列表 c 等于 ImmutableSparseNDimArray([8]*9, (3, 3))，即一个稀疏数组，维度为 (3, 3)，9 个元素全为 8
    assert c == ImmutableSparseNDimArray([8]*9, (3, 3))
# 定义测试函数，用于测试多维不可变数组的转换和功能
def test_ndim_array_converting():
    # 创建一个密集型不可变多维数组，内容为 [1, 2, 3, 4]，形状为 (2, 2)
    dense_array = ImmutableDenseNDimArray([1, 2, 3, 4], (2, 2))
    # 将密集型不可变多维数组转换为普通 Python 列表
    alist = dense_array.tolist()

    # 断言转换后的列表与期望的二维列表 [[1, 2], [3, 4]] 相等
    assert alist == [[1, 2], [3, 4]]

    # 将密集型不可变多维数组转换为 Matrix 对象
    matrix = dense_array.tomatrix()
    # 断言转换后的对象类型为 Matrix
    assert (isinstance(matrix, Matrix))

    # 遍历密集型不可变多维数组的每个元素索引，确保索引访问 Matrix 对象后的值与原始数组相同
    for i in range(len(dense_array)):
        assert dense_array[dense_array._get_tuple_index(i)] == matrix[i]
    # 断言 Matrix 对象的形状与原始密集型数组的形状相同
    assert matrix.shape == dense_array.shape

    # 断言将 Matrix 对象转换为密集型不可变多维数组后与原始数组相等
    assert ImmutableDenseNDimArray(matrix) == dense_array
    # 断言将 Matrix 对象转换为不可变形式后与原始数组相等
    assert ImmutableDenseNDimArray(matrix.as_immutable()) == dense_array
    # 断言将 Matrix 对象转换为可变形式后与原始数组相等
    assert ImmutableDenseNDimArray(matrix.as_mutable()) == dense_array

    # 创建一个稀疏型不可变多维数组，内容为 [1, 2, 3, 4]，形状为 (2, 2)
    sparse_array = ImmutableSparseNDimArray([1, 2, 3, 4], (2, 2))
    # 将稀疏型不可变多维数组转换为普通 Python 列表
    alist = sparse_array.tolist()

    # 断言转换后的列表与期望的二维列表 [[1, 2], [3, 4]] 相等
    assert alist == [[1, 2], [3, 4]]

    # 将稀疏型不可变多维数组转换为 Matrix 对象
    matrix = sparse_array.tomatrix()
    # 断言转换后的对象类型为 SparseMatrix
    assert(isinstance(matrix, SparseMatrix))

    # 遍历稀疏型不可变多维数组的每个元素索引，确保索引访问 SparseMatrix 对象后的值与原始数组相同
    for i in range(len(sparse_array)):
        assert sparse_array[sparse_array._get_tuple_index(i)] == matrix[i]
    # 断言 SparseMatrix 对象的形状与原始稀疏型数组的形状相同
    assert matrix.shape == sparse_array.shape

    # 断言将 SparseMatrix 对象转换为稀疏型不可变多维数组后与原始数组相等
    assert ImmutableSparseNDimArray(matrix) == sparse_array
    # 断言将 SparseMatrix 对象转换为不可变形式后与原始数组相等
    assert ImmutableSparseNDimArray(matrix.as_immutable()) == sparse_array
    # 断言将 SparseMatrix 对象转换为可变形式后与原始数组相等
    assert ImmutableSparseNDimArray(matrix.as_mutable()) == sparse_array


# 定义测试函数，用于测试多维不可变数组的转换功能
def test_converting_functions():
    # 创建一个普通 Python 列表 [1, 2, 3, 4]
    arr_list = [1, 2, 3, 4]
    # 创建一个 Matrix 对象，内容为 ((1, 2), (3, 4))
    arr_matrix = Matrix(((1, 2), (3, 4)))

    # 将普通列表转换为密集型不可变多维数组，形状为 (2, 2)
    arr_ndim_array = ImmutableDenseNDimArray(arr_list, (2, 2))
    # 断言转换后的对象类型为 ImmutableDenseNDimArray
    assert (isinstance(arr_ndim_array, ImmutableDenseNDimArray))
    # 断言 Matrix 对象转换为列表后与密集型不可变多维数组转换的列表相等
    assert arr_matrix.tolist() == arr_ndim_array.tolist()

    # 将 Matrix 对象转换为密集型不可变多维数组
    arr_ndim_array = ImmutableDenseNDimArray(arr_matrix)
    # 断言转换后的对象类型为 ImmutableDenseNDimArray
    assert (isinstance(arr_ndim_array, ImmutableDenseNDimArray))
    # 断言 Matrix 对象转换为列表后与密集型不可变多维数组转换的列表相等
    assert arr_matrix.tolist() == arr_ndim_array.tolist()
    # 断言 Matrix 对象与密集型不可变多维数组的形状相等
    assert arr_matrix.shape == arr_ndim_array.shape


# 定义测试函数，用于测试多维不可变数组的相等性
def test_equality():
    # 创建两个相同的普通列表 [1, 2, 3, 4]
    first_list = [1, 2, 3, 4]
    second_list = [1, 2, 3, 4]
    # 创建一个不同的普通列表 [4, 3, 2, 1]
    third_list = [4, 3, 2, 1]
    # 断言两个相同的普通列表相等
    assert first_list == second_list
    # 断言两个不同的普通列表不相等
    assert first_list != third_list

    # 创建两个相同内容和形状的密集型不可变多维数组
    first_ndim_array = ImmutableDenseNDimArray(first_list, (2, 2))
    second_ndim_array = ImmutableDenseNDimArray(second_list, (2, 2))
    fourth_ndim_array = ImmutableDenseNDimArray(first_list, (2, 2))

    # 断言两个相同内容和形状的密集型不可变多维数组相等
    assert first_ndim_array == second_ndim_array

    # 定义一个尝试修改数组的函数，断言在尝试修改第二个数组时会引发 TypeError 异常
    def assignment_attempt(a):
        a[0, 0] = 0

    raises(TypeError, lambda: assignment_attempt(second_ndim_array))
    # 断言修改第二个数组后，第一个数组和第二个数组仍然相等
    assert first_ndim_array == second_ndim_array
    assert first_ndim_array == fourth_ndim_array


# 定义测试函数，用于测试多维不可变数组的算术运算
def test_arithmetic():
    # 创建一个内容全为 3 的密集型不可变多维数组，形状为 (3, 3)
    a = ImmutableDenseNDimArray([3 for i in range(9)], (3, 3))
    # 创建一个内容全为 7 的密集型不可变多维数组，形状为 (3, 3)
    b = ImmutableDenseNDimArray([7 for i in range(9)], (3, 3))

    # 计算两个数组的和并赋值给 c1 和 c2
    c1 = a + b
    c2 = b + a
    # 断言两种加法结果相等
    assert c1 == c2

    # 计算两个数组的差并赋值给 d1 和 d2
    d1 = a - b
    d2 = b - a
    # 断言 d1 是 d2 的相反数
    assert d1 == d2 * (-1)

    # 计算数组与标量的乘法，并赋值给 e1, e2, e3
    e1 = a * 5
    e2 = 5 * a
    e3 = copy(a)
    # 断言确保变量 a, b, c1, c2, d1, d2, e1, e2, e3, f1 全部具有相同的类型
    assert type(a) == type(b) == type(c1) == type(c2) == type(d1) == type(d2) \
        == type(e1) == type(e2) == type(e3) == type(f1)
    
    # 计算 z0 的值为 -a
    z0 = -a
    
    # 断言确保 z0 的值等于一个特定的不可变的稠密多维数组
    assert z0 == ImmutableDenseNDimArray([-3 for i in range(9)], (3, 3))
def test_higher_dimenions():
    # 创建一个 3 维的不可变密集数组，范围是从 10 到 33
    m3 = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))

    # 断言密集数组转换为列表后与预期的三维数组相等
    assert m3.tolist() == [[[10, 11, 12, 13],
            [14, 15, 16, 17],
            [18, 19, 20, 21]],

           [[22, 23, 24, 25],
            [26, 27, 28, 29],
            [30, 31, 32, 33]]]

    # 断言获取索引 0 时的元组索引
    assert m3._get_tuple_index(0) == (0, 0, 0)
    # 断言获取索引 1 时的元组索引
    assert m3._get_tuple_index(1) == (0, 0, 1)
    # 断言获取索引 4 时的元组索引
    assert m3._get_tuple_index(4) == (0, 1, 0)
    # 断言获取索引 12 时的元组索引
    assert m3._get_tuple_index(12) == (1, 0, 0)

    # 断言密集数组转换为字符串后与预期的字符串表示形式相等
    assert str(m3) == '[[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]]'

    # 重新构建一个相同的不可变密集数组，并断言两者相等
    m3_rebuilt = ImmutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]])
    assert m3 == m3_rebuilt

    # 创建另一个相同内容和形状的不可变密集数组，并断言两者相等
    m3_other = ImmutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]], (2, 3, 4))
    assert m3 == m3_other


def test_rebuild_immutable_arrays():
    # 创建一个不可变稀疏多维数组和一个不可变密集多维数组
    sparr = ImmutableSparseNDimArray(range(10, 34), (2, 3, 4))
    densarr = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))

    # 断言稀疏多维数组与其函数表示的相等性
    assert sparr == sparr.func(*sparr.args)
    # 断言密集多维数组与其函数表示的相等性
    assert densarr == densarr.func(*densarr.args)


def test_slices():
    # 创建一个不可变密集多维数组
    md = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))

    # 断言整个数组切片与原数组相等
    assert md[:] == ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))
    # 断言选取第三维度为 0 时的二维矩阵表示
    assert md[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    # 断言选取第一维度为 0，第二维度从索引 1 到 2 的三维矩阵表示
    assert md[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    # 断言选取第一维度为 0，第二维度从索引 1 到 3 的三维矩阵表示
    assert md[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    # 断言整个数组切片与原数组相等
    assert md[:, :, :] == md

    # 创建一个相同内容的不可变稀疏多维数组，并断言两者相等
    sd = ImmutableSparseNDimArray(range(10, 34), (2, 3, 4))
    assert sd == ImmutableSparseNDimArray(md)

    # 断言整个数组切片与原数组相等
    assert sd[:] == ImmutableSparseNDimArray(range(10, 34), (2, 3, 4))
    # 断言选取第三维度为 0 时的二维矩阵表示
    assert sd[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    # 断言选取第一维度为 0，第二维度从索引 1 到 2 的三维矩阵表示
    assert sd[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    # 断言选取第一维度为 0，第二维度从索引 1 到 3 的三维矩阵表示
    assert sd[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    # 断言整个数组切片与原数组相等
    assert sd[:, :, :] == sd


def test_diff_and_applyfunc():
    # 导入 sympy 库中的变量 x, y, z
    from sympy.abc import x, y, z
    # 创建一个不可变密集多维数组
    md = ImmutableDenseNDimArray([[x, y], [x*z, x*y*z]])
    # 断言对数组进行 x 方向的偏导数操作后的结果
    assert md.diff(x) == ImmutableDenseNDimArray([[1, 0], [z, y*z]])
    # 断言使用 sympy 库中的 diff 函数对数组进行 x 方向的偏导数操作后的结果
    assert diff(md, x) == ImmutableDenseNDimArray([[1, 0], [z, y*z]])

    # 创建一个不可变稀疏多维数组，并断言两者相等
    sd = ImmutableSparseNDimArray(md)
    assert sd == ImmutableSparseNDimArray([x, y, x*z, x*y*z], (2, 2))
    # 断言对数组进行 x 方向的偏导数操作后的结果
    assert sd.diff(x) == ImmutableSparseNDimArray([[1, 0], [z, y*z]])
    # 断言使用 sympy 库中的 diff 函数对数组进行 x 方向的偏导数操作后的结果
    assert diff(sd, x) == ImmutableSparseNDimArray([[1, 0], [z, y*z]])

    # 对不可变密集多维数组进行元素级乘以 3 的操作，并断言结果相等
    mdn = md.applyfunc(lambda x: x*3)
    assert mdn == ImmutableDenseNDimArray([[3*x, 3*y], [3*x*z, 3*x*y*z]])
    # 断言修改后的数组与原数组不相等
    assert md != mdn

    # 对不可变稀疏多维数组进行元素级除以 2 的操作，并断言结果相等
    sdn = sd.applyfunc(lambda x: x/2)
    assert sdn == ImmutableSparseNDimArray([[x/2, y/2], [x*z/2, x*y*z/2]])
    # 断言修改后的数组与原数组不相等
    # 断言：确保 sd 和 sdp 不相等
    assert sd != sdp
# 测试操作的优先级和表达式的相等性
def test_op_priority():
    # 从 sympy.abc 导入符号 x
    from sympy.abc import x
    # 创建一个不可变的稠密多维数组 md，包含元素 [1, 2, 3]
    md = ImmutableDenseNDimArray([1, 2, 3])
    # 创建表达式 e1，计算 (1+x)*md
    e1 = (1+x)*md
    # 创建表达式 e2，计算 md*(1+x)
    e2 = md*(1+x)
    # 断言 e1 等于 ImmutableDenseNDimArray([1+x, 2+2*x, 3+3*x])
    assert e1 == ImmutableDenseNDimArray([1+x, 2+2*x, 3+3*x])
    # 断言 e1 等于 e2
    assert e1 == e2

    # 创建一个不可变的稀疏多维数组 sd，包含元素 [1, 2, 3]
    sd = ImmutableSparseNDimArray([1, 2, 3])
    # 创建表达式 e3，计算 (1+x)*sd
    e3 = (1+x)*sd
    # 创建表达式 e4，计算 sd*(1+x)
    e4 = sd*(1+x)
    # 断言 e3 等于 ImmutableDenseNDimArray([1+x, 2+2*x, 3+3*x])
    assert e3 == ImmutableDenseNDimArray([1+x, 2+2*x, 3+3*x])
    # 断言 e3 等于 e4
    assert e3 == e4


# 测试符号索引操作
def test_symbolic_indexing():
    # 定义符号 x, y, z, w
    x, y, z, w = symbols("x y z w")
    # 创建一个不可变的稠密多维数组 M，包含元素 [[x, y], [z, w]]
    M = ImmutableDenseNDimArray([[x, y], [z, w]])
    # 定义符号 i, j
    i, j = symbols("i, j")
    # 获取 M 中索引为 (i, j) 的元素，存储为 Mij
    Mij = M[i, j]
    # 断言 Mij 的类型为 Indexed
    assert isinstance(Mij, Indexed)
    
    # 创建一个不可变的稀疏多维数组 Ms，包含元素 [[2, 3*x], [4, 5]]
    Ms = ImmutableSparseNDimArray([[2, 3*x], [4, 5]])
    # 获取 Ms 中索引为 (i, j) 的元素，存储为 msij
    msij = Ms[i, j]
    # 断言 msij 的类型为 Indexed
    assert isinstance(msij, Indexed)
    
    # 遍历索引对 (oi, oj) 的列表 [(0, 0), (0, 1), (1, 0), (1, 1)]
    for oi, oj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        # 断言用 {i: oi, j: oj} 替换后的 Mij 等于 M 中索引为 (oi, oj) 的元素
        assert Mij.subs({i: oi, j: oj}) == M[oi, oj]
        # 断言用 {i: oi, j: oj} 替换后的 msij 等于 Ms 中索引为 (oi, oj) 的元素
        assert msij.subs({i: oi, j: oj}) == Ms[oi, oj]
    
    # 创建 IndexedBase 对象 A，基于索引元组 (0, 2)
    A = IndexedBase("A", (0, 2))
    # 断言用 A 替换后的 A[0, 0] 等于 M 中索引为 (0, 0) 的元素 x
    assert A[0, 0].subs(A, M) == x
    # 断言用 A 替换后的 A[i, j] 等于 M 中索引为 (i, j) 的元素
    assert A[i, j].subs(A, M) == M[i, j]
    # 断言用 M 替换后的 M[i, j] 等于 A 中索引为 (i, j) 的元素
    assert M[i, j].subs(M, A) == A[i, j]

    # 断言 M 中索引为 (3*i-2, j) 的元素为 Indexed
    assert isinstance(M[3 * i - 2, j], Indexed)
    # 断言用 {i: 1, j: 0} 替换后的 M[3*i-2, j] 等于 M 中索引为 (1, 0) 的元素
    assert M[3 * i - 2, j].subs({i: 1, j: 0}) == M[1, 0]
    # 断言 M 中索引为 (i, 0) 的元素为 Indexed
    assert isinstance(M[i, 0], Indexed)
    # 断言用 i 替换后的 M[i, 0] 等于 M 中索引为 (0, 0) 的元素
    assert M[i, 0].subs(i, 0) == M[0, 0]
    # 断言用 i 替换后的 M[0, i] 等于 M 中索引为 (0, 1) 的元素
    assert M[0, i].subs(i, 1) == M[0, 1]

    # 断言 M 中索引为 (i, j) 的元素对 x 的偏导数等于对应的稠密多维数组 [[1, 0], [0, 0]] 中的元素
    assert M[i, j].diff(x) == ImmutableDenseNDimArray([[1, 0], [0, 0]])[i, j]
    # 断言 Ms 中索引为 (i, j) 的元素对 x 的偏导数等于对应的稀疏多维数组 [[0, 3], [0, 0]] 中的元素
    assert Ms[i, j].diff(x) == ImmutableSparseNDimArray([[0, 3], [0, 0]])[i, j]

    # 创建一个不可变的稠密多维数组 Mo，包含元素 [1, 2, 3]
    Mo = ImmutableDenseNDimArray([1, 2, 3])
    # 断言用 i 替换后的 Mo[i] 等于 2
    assert Mo[i].subs(i, 1) == 2
    # 创建一个不可变的稀疏多维数组 Mos，包含元素 [1, 2, 3]
    Mos = ImmutableSparseNDimArray([1, 2, 3])
    # 断言用 i 替换后的 Mos[i] 等于 2
    assert Mos[i].subs(i, 1) == 2

    # 断言调用 M[i, 2] 抛出 ValueError 异常
    raises(ValueError, lambda: M[i, 2])
    # 断言调用 M[i, -1] 抛出 ValueError 异常
    raises(ValueError, lambda: M[i, -1])
    # 断言调用 M[2, i] 抛出 ValueError 异常
    raises(ValueError, lambda: M[2, i])
    # 断言调用 M[-1, i] 抛出 ValueError 异常
    raises(ValueError, lambda: M[-1, i])

    # 断言调用 Ms[i, 2] 抛出 ValueError 异常
    raises(ValueError, lambda: Ms[i, 2])
    # 断言调用 Ms[i, -1] 抛出 ValueError 异常
    raises(ValueError, lambda: Ms[i, -1])
    # 断言调用 Ms[2, i] 抛出 ValueError 异常
    raises(ValueError, lambda: Ms[2, i])
    # 断言调用 Ms[-1, i] 抛出 ValueError 异常
    raises(ValueError, lambda: Ms[-1, i])


def test_issue_12665():
    # 测试不可变数组的哈希值在 Python 3 中
    arr = ImmutableDenseNDimArray([1, 2, 3])
    # 断言对 arr 的哈希操作不引发异常
```