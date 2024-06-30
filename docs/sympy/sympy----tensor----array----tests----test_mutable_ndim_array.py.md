# `D:\src\scipysrc\sympy\sympy\tensor\array\tests\test_mutable_ndim_array.py`

```
# 导入必要的模块和类
from copy import copy  # 导入 copy 函数
from sympy.tensor.array.dense_ndim_array import MutableDenseNDimArray  # 导入 MutableDenseNDimArray 类
from sympy.core.function import diff  # 导入 diff 函数
from sympy.core.numbers import Rational  # 导入 Rational 类
from sympy.core.singleton import S  # 导入 S 单例
from sympy.core.symbol import Symbol  # 导入 Symbol 类
from sympy.core.sympify import sympify  # 导入 sympify 函数
from sympy.matrices import SparseMatrix  # 导入 SparseMatrix 类
from sympy.matrices import Matrix  # 导入 Matrix 类
from sympy.tensor.array.sparse_ndim_array import MutableSparseNDimArray  # 导入 MutableSparseNDimArray 类
from sympy.testing.pytest import raises  # 导入 raises 函数

# 定义测试函数 test_ndim_array_initiation
def test_ndim_array_initiation():
    # 创建包含一个元素的 MutableDenseNDimArray 对象
    arr_with_one_element = MutableDenseNDimArray([23])
    # 断言数组长度为1
    assert len(arr_with_one_element) == 1
    # 断言数组第一个元素为23
    assert arr_with_one_element[0] == 23
    # 断言数组的秩为1
    assert arr_with_one_element.rank() == 1
    # 使用 lambda 表达式测试索引超出边界是否会引发 ValueError 异常
    raises(ValueError, lambda: arr_with_one_element[1])

    # 创建包含一个符号元素的 MutableDenseNDimArray 对象
    arr_with_symbol_element = MutableDenseNDimArray([Symbol('x')])
    # 断言数组长度为1
    assert len(arr_with_symbol_element) == 1
    # 断言数组第一个元素为符号 'x'
    assert arr_with_symbol_element[0] == Symbol('x')
    # 断言数组的秩为1
    assert arr_with_symbol_element.rank() == 1

    # 创建一个长度为5的零向量 MutableDenseNDimArray 对象
    number5 = 5
    vector = MutableDenseNDimArray.zeros(number5)
    # 断言数组长度为5
    assert len(vector) == number5
    # 断言数组形状为 (5,)
    assert vector.shape == (number5,)
    # 断言数组的秩为1
    assert vector.rank() == 1
    # 使用 lambda 表达式测试索引超出边界是否会引发 ValueError 异常
    raises(ValueError, lambda: arr_with_one_element[5])

    # 创建一个长度为5的零稀疏向量 MutableSparseNDimArray 对象
    vector = MutableSparseNDimArray.zeros(number5)
    # 断言数组长度为5
    assert len(vector) == number5
    # 断言数组形状为 (5,)
    assert vector.shape == (number5,)
    # 断言数组的稀疏数组为空字典
    assert vector._sparse_array == {}
    # 断言数组的秩为1
    assert vector.rank() == 1

    # 创建一个形状为 (3, 3, 3, 3) 的 MutableDenseNDimArray 对象
    n_dim_array = MutableDenseNDimArray(range(3**4), (3, 3, 3, 3,))
    # 断言数组长度为 3*3*3*3=81
    assert len(n_dim_array) == 3 * 3 * 3 * 3
    # 断言数组形状为 (3, 3, 3, 3)
    assert n_dim_array.shape == (3, 3, 3, 3)
    # 断言数组的秩为4
    assert n_dim_array.rank() == 4
    # 使用 lambda 表达式测试索引超出边界是否会引发 ValueError 异常
    raises(ValueError, lambda: n_dim_array[0, 0, 0, 3])
    raises(ValueError, lambda: n_dim_array[3, 0, 0, 0])
    raises(ValueError, lambda: n_dim_array[3**4])

    # 创建一个形状为 (3, 3, 3, 3) 的 MutableSparseNDimArray 对象
    array_shape = (3, 3, 3, 3)
    sparse_array = MutableSparseNDimArray.zeros(*array_shape)
    # 断言稀疏数组的稀疏数组字典长度为0
    assert len(sparse_array._sparse_array) == 0
    # 断言数组长度为 3*3*3*3=81
    assert len(sparse_array) == 3 * 3 * 3 * 3
    # 断言数组形状为 (3, 3, 3, 3)
    assert n_dim_array.shape == array_shape
    # 断言数组的秩为4
    assert n_dim_array.rank() == 4

    # 创建一个包含三个元素的 MutableDenseNDimArray 对象
    one_dim_array = MutableDenseNDimArray([2, 3, 1])
    # 断言数组长度为3
    assert len(one_dim_array) == 3
    # 断言数组形状为 (3,)
    assert one_dim_array.shape == (3,)
    # 断言数组的秩为1
    assert one_dim_array.rank() == 1
    # 断言将数组转换为列表后的结果正确
    assert one_dim_array.tolist() == [2, 3, 1]

    # 创建一个形状为 (3, 3) 的 MutableSparseNDimArray 对象
    shape = (3, 3)
    array_with_many_args = MutableSparseNDimArray.zeros(*shape)
    # 断言数组长度为 3*3=9
    assert len(array_with_many_args) == 3 * 3
    # 断言数组形状为 (3, 3)
    assert array_with_many_args.shape == shape
    # 断言数组第一个元素为0
    assert array_with_many_args[0, 0] == 0
    # 断言数组的秩为2
    assert array_with_many_args.rank() == 2

    # 创建一个形状为 (3, 3) 的 MutableSparseNDimArray 对象
    shape = (int(3), int(3))
    array_with_long_shape = MutableSparseNDimArray.zeros(*shape)
    # 断言数组长度为 3*3=9
    assert len(array_with_long_shape) == 3 * 3
    # 断言数组形状为 (3, 3)
    assert array_with_long_shape.shape == shape
    # 断言数组第一个元素为0
    assert array_with_long_shape[int(0), int(0)] == 0
    # 断言数组的秩为2
    assert array_with_long_shape.rank() == 2

    # 创建一个长度为5的 MutableDenseNDimArray 对象
    vector_with_long_shape = MutableDenseNDimArray(range(5), int(5))
    # 断言数组长度为5
    assert len(vector_with_long_shape) == 5
    # 断言数组形状为 (5,)
    assert vector_with_long_shape.shape == (int(5),)
    # 断言数组的秩为1
    assert vector_with_long_shape.rank() == 1
    # 确保 vector_with_long_shape[int(5)] 引发 ValueError 异常
    raises(ValueError, lambda: vector_with_long_shape[int(5)])
    
    # 从 sympy.abc 模块中导入 x 符号
    from sympy.abc import x
    
    # 对于每种数组类型 MutableDenseNDimArray 和 MutableSparseNDimArray
    for ArrayType in [MutableDenseNDimArray, MutableSparseNDimArray]:
        # 创建一个秩为零的数组 rank_zero_array，包含唯一元素 x
        rank_zero_array = ArrayType(x)
        
        # 断言数组的长度为 1
        assert len(rank_zero_array) == 1
        # 断言数组的形状为空元组 ()
        assert rank_zero_array.shape == ()
        # 断言数组的秩为 0
        assert rank_zero_array.rank() == 0
        # 断言数组索引 () 得到元素 x
        assert rank_zero_array[()] == x
        # 确保 rank_zero_array[0] 引发 ValueError 异常
        raises(ValueError, lambda: rank_zero_array[0])
# 定义测试函数 test_sympify，用于测试 sympify 函数的功能
def test_sympify():
    # 从 sympy.abc 模块中导入符号 x, y, z, t
    from sympy.abc import x, y, z, t
    # 创建一个二维可变密集数组 arr，包含符号 x, y 和表达式 z*t
    arr = MutableDenseNDimArray([[x, y], [1, z*t]])
    # 使用 sympify 函数将 arr 转换为一个符号化的表达式 arr_other
    arr_other = sympify(arr)
    # 断言 arr_other 的形状为 (2, 2)
    assert arr_other.shape == (2, 2)
    # 断言 arr_other 等于 arr
    assert arr_other == arr


# 定义测试函数 test_reshape，用于测试 MutableDenseNDimArray 的 reshape 方法
def test_reshape():
    # 创建一个包含 0 到 49 的可变密集数组 array，形状为 (50,)
    array = MutableDenseNDimArray(range(50), 50)
    # 断言 array 的形状为 (50,)
    assert array.shape == (50,)
    # 断言 array 的秩为 1
    assert array.rank() == 1

    # 使用 reshape 方法将 array 重新调整为形状 (5, 5, 2)
    array = array.reshape(5, 5, 2)
    # 断言 array 的形状为 (5, 5, 2)
    assert array.shape == (5, 5, 2)
    # 断言 array 的秩为 3
    assert array.rank() == 3
    # 断言 array 的元素个数为 50
    assert len(array) == 50


# 定义测试函数 test_iterator，用于测试 MutableDenseNDimArray 的迭代功能
def test_iterator():
    # 创建一个二维可变密集数组 array，包含元素 0 到 3，形状为 (2, 2)
    array = MutableDenseNDimArray(range(4), (2, 2))
    # 断言 array 的第一个元素是包含 [0, 1] 的可变密集数组
    assert array[0] == MutableDenseNDimArray([0, 1])
    # 断言 array 的第二个元素是包含 [2, 3] 的可变密集数组
    assert array[1] == MutableDenseNDimArray([2, 3])

    # 使用 reshape 方法将 array 重新调整为形状 (4,)
    array = array.reshape(4)
    j = 0
    # 遍历 array 的每个元素 i
    for i in array:
        # 断言 i 等于 j
        assert i == j
        j += 1


# 定义测试函数 test_getitem，用于测试 MutableDenseNDimArray 和 MutableSparseNDimArray 的索引功能
def test_getitem():
    # 对于 MutableDenseNDimArray 和 MutableSparseNDimArray 两种数组类型进行循环测试
    for ArrayType in [MutableDenseNDimArray, MutableSparseNDimArray]:
        # 创建一个 ArrayType 类型的数组 array，包含 0 到 23 的元素，形状为 (2, 3, 4)
        array = ArrayType(range(24)).reshape(2, 3, 4)
        # 断言 array 转换为嵌套列表后与给定列表相同
        assert array.tolist() == [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
        # 断言 array 的第一个切片与预期的 ArrayType 类型的数组相同
        assert array[0] == ArrayType([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        # 断言 array 的第一个元素的第一个切片与预期的 ArrayType 类型的数组相同
        assert array[0, 0] == ArrayType([0, 1, 2, 3])
        value = 0
        # 遍历 array 的每个索引 (i, j, k)，并断言其值等于 value
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    assert array[i, j, k] == value
                    value += 1

    # 断言对于超出数组索引范围的索引会引发 ValueError 异常
    raises(ValueError, lambda: array[3, 4, 5])
    raises(ValueError, lambda: array[3, 4, 5, 6])
    raises(ValueError, lambda: array[3, 4, 5, 3:4])


# 定义测试函数 test_sparse，用于测试 MutableSparseNDimArray 类的稀疏数组功能
def test_sparse():
    # 创建一个稀疏数组 sparse_array，包含元素 [0, 0, 0, 1]，形状为 (2, 2)
    sparse_array = MutableSparseNDimArray([0, 0, 0, 1], (2, 2))
    # 断言 sparse_array 的长度为 4
    assert len(sparse_array) == 2 * 2
    # 断言稀疏数组中存储数据的字典长度为 1
    assert len(sparse_array._sparse_array) == 1

    # 断言 sparse_array 转换为嵌套列表后与给定列表相同
    assert sparse_array.tolist() == [[0, 0], [0, 1]]

    # 使用 zip 函数遍历 sparse_array 和给定列表 j 的每个元素，并断言它们相等
    for i, j in zip(sparse_array, [[0, 0], [0, 1]]):
        assert i == MutableSparseNDimArray(j)

    # 修改 sparse_array 的元素 (0, 0) 为 123
    sparse_array[0, 0] = 123
    # 断言稀疏数组中存储数据的字典长度为 2
    assert len(sparse_array._sparse_array) == 2
    # 断言 sparse_array 的元素 (0, 0) 等于 123
    assert sparse_array[0, 0] == 123
    # 断言对稀疏数组进行除法运算得到的结果与预期的 MutableSparseNDimArray 数组相同
    assert sparse_array/0 == MutableSparseNDimArray([[S.ComplexInfinity, S.NaN], [S.NaN, S.ComplexInfinity]], (2, 2))

    # 将 sparse_array 的元素 (0, 0) 修改为 0 后，断言稀疏数组中存储数据的字典长度为 1
    sparse_array[0, 0] = 0
    assert len(sparse_array._sparse_array) == 1
    # 再次修改 sparse_array 的元素 (1, 1) 为 0 后，断言稀疏数组中存储数据的字典长度为 0
    sparse_array[1, 1] = 0
    assert len(sparse_array._sparse_array) == 0
    # 断言 sparse_array 的元素 (0, 0) 等于 0
    assert sparse_array[0, 0] == 0

    # 测试大规模稀疏数组的相等性
    # 创建两个大小为 (100000, 200000) 的零稀疏数组 a 和 b
    a = MutableSparseNDimArray.zeros(100000, 200000)
    b = MutableSparseNDimArray.zeros(100000, 200000)
    # 断言 a 等于 b
    assert a == b
    # 将 a 的元素 (1, 1) 设置为 1，将 b 的元素 (1, 1) 设置为 2
    a[1, 1] = 1
    b[1, 1] = 2
    # 断言 a 不等于 b
    assert a != b

    # 测试稀疏数组的乘法和除法运算
    # 断言 a 乘以
    # 断言：验证表达式 a/3 是否等于一个 MutableSparseNDimArray 对象
    assert a/3 == MutableSparseNDimArray({200001: Rational(1, 3)}, (100000, 200000))
    
    # 断言描述 __neg__ 方法的功能：验证取反运算符 -a 是否等于一个 MutableSparseNDimArray 对象
    assert -a == MutableSparseNDimArray({200001: -1}, (100000, 200000))
# 定义测试函数，用于验证 MutableDenseNDimArray 类的计算功能
def test_calculation():

    # 创建两个 3x3 的 MutableDenseNDimArray 实例，分别填充为 [1, 1, 1, 1, 1, 1, 1, 1, 1] 和 [9, 9, 9, 9, 9, 9, 9, 9, 9]
    a = MutableDenseNDimArray([1]*9, (3, 3))
    b = MutableDenseNDimArray([9]*9, (3, 3))

    # 计算 a + b，结果存储在 c 中，c 中每个元素为 MutableDenseNDimArray([10, 10, 10])
    c = a + b

    # 遍历 c 的每个元素，断言每个元素为 MutableDenseNDimArray([10, 10, 10])
    for i in c:
        assert i == MutableDenseNDimArray([10, 10, 10])

    # 断言 c 应与 MutableDenseNDimArray([10]*9, (3, 3)) 相等
    assert c == MutableDenseNDimArray([10]*9, (3, 3))

    # 断言 c 应与 MutableSparseNDimArray([10]*9, (3, 3)) 相等
    assert c == MutableSparseNDimArray([10]*9, (3, 3))

    # 计算 b - a，结果存储在 c 中，c 中每个元素为 MutableSparseNDimArray([8, 8, 8])
    c = b - a

    # 遍历 c 的每个元素，断言每个元素为 MutableSparseNDimArray([8, 8, 8])
    for i in c:
        assert i == MutableSparseNDimArray([8, 8, 8])

    # 断言 c 应与 MutableDenseNDimArray([8]*9, (3, 3)) 相等
    assert c == MutableDenseNDimArray([8]*9, (3, 3))

    # 断言 c 应与 MutableSparseNDimArray([8]*9, (3, 3)) 相等
    assert c == MutableSparseNDimArray([8]*9, (3, 3))


# 定义测试函数，用于验证 MutableDenseNDimArray 和 MutableSparseNDimArray 的转换功能
def test_ndim_array_converting():

    # 创建一个 2x2 的 MutableDenseNDimArray 实例 dense_array，内容为 [1, 2, 3, 4]
    dense_array = MutableDenseNDimArray([1, 2, 3, 4], (2, 2))

    # 将 dense_array 转换为普通列表 alist
    alist = dense_array.tolist()

    # 断言 alist 应为 [[1, 2], [3, 4]]
    assert alist == [[1, 2], [3, 4]]

    # 将 dense_array 转换为 Matrix 类型的实例 matrix
    matrix = dense_array.tomatrix()

    # 断言 matrix 应为 Matrix 类型的实例
    assert (isinstance(matrix, Matrix))

    # 遍历 dense_array 的每个元素的索引，断言 dense_array 的每个元素与 matrix 的相应元素相等
    for i in range(len(dense_array)):
        assert dense_array[dense_array._get_tuple_index(i)] == matrix[i]

    # 断言 matrix 的形状应与 dense_array 的形状相等
    assert matrix.shape == dense_array.shape

    # 断言从 matrix 转换回 MutableDenseNDimArray 应与 dense_array 相等
    assert MutableDenseNDimArray(matrix) == dense_array

    # 断言从 matrix 转换为不可变对象后再转换为 MutableDenseNDimArray 应与 dense_array 相等
    assert MutableDenseNDimArray(matrix.as_immutable()) == dense_array

    # 断言从 matrix 转换为可变对象后再转换为 MutableDenseNDimArray 应与 dense_array 相等
    assert MutableDenseNDimArray(matrix.as_mutable()) == dense_array

    # 创建一个 2x2 的 MutableSparseNDimArray 实例 sparse_array，内容为 [1, 2, 3, 4]
    sparse_array = MutableSparseNDimArray([1, 2, 3, 4], (2, 2))

    # 将 sparse_array 转换为普通列表 alist
    alist = sparse_array.tolist()

    # 断言 alist 应为 [[1, 2], [3, 4]]
    assert alist == [[1, 2], [3, 4]]

    # 将 sparse_array 转换为 SparseMatrix 类型的实例 matrix
    matrix = sparse_array.tomatrix()

    # 断言 matrix 应为 SparseMatrix 类型的实例
    assert(isinstance(matrix, SparseMatrix))

    # 遍历 sparse_array 的每个元素的索引，断言 sparse_array 的每个元素与 matrix 的相应元素相等
    for i in range(len(sparse_array)):
        assert sparse_array[sparse_array._get_tuple_index(i)] == matrix[i]

    # 断言 matrix 的形状应与 sparse_array 的形状相等
    assert matrix.shape == sparse_array.shape

    # 断言从 matrix 转换回 MutableSparseNDimArray 应与 sparse_array 相等
    assert MutableSparseNDimArray(matrix) == sparse_array

    # 断言从 matrix 转换为不可变对象后再转换为 MutableSparseNDimArray 应与 sparse_array 相等
    assert MutableSparseNDimArray(matrix.as_immutable()) == sparse_array

    # 断言从 matrix 转换为可变对象后再转换为 MutableSparseNDimArray 应与 sparse_array 相等


# 定义测试函数，用于验证 MutableDenseNDimArray 的转换功能
def test_converting_functions():

    # 创建一个普通列表 arr_list，内容为 [1, 2, 3, 4]
    arr_list = [1, 2, 3, 4]

    # 创建一个 Matrix 类型的实例 arr_matrix，内容为 ((1, 2), (3, 4))
    arr_matrix = Matrix(((1, 2), (3, 4)))

    # 创建一个 2x2 的 MutableDenseNDimArray 实例 arr_ndim_array，内容与 arr_list 相同
    arr_ndim_array = MutableDenseNDimArray(arr_list, (2, 2))

    # 断言 arr_ndim_array 应为 MutableDenseNDimArray 类型的实例
    assert (isinstance(arr_ndim_array, MutableDenseNDimArray))

    # 断言 arr_matrix 转换为普通列表应与 arr_ndim_array 的转换结果相等
    assert arr_matrix.tolist() == arr_ndim_array.tolist()

    # 创建一个 MutableDenseNDimArray 实例 arr_ndim_array，内容与 arr_matrix 相同
    arr_ndim_array = MutableDenseNDimArray(arr_matrix)

    # 断言 arr_ndim_array 应为 MutableDenseNDimArray 类型的实例
    assert (isinstance(arr_ndim_array, MutableDenseNDimArray))

    # 断言 arr_matrix 转换为普通列表应与 arr_ndim_array 的转换结果相等
    assert arr_matrix.tolist() == arr_ndim_array.tolist()

    # 断言 arr_matrix 的形状应与 arr_ndim_array 的形状相等
    assert arr_matrix.shape == arr_ndim_array.shape


# 定义测试函数，用于验证 MutableDenseNDimArray 的相等性比较功能
def test_equality():

    # 创建三个相同的普通列表 first_list, second_list, third_list
    first_list = [1, 2, 3, 4]
    second_list = [1, 2, 3, 4]
    third_list = [4, 3, 2, 1]

    # 断言 first_list 与 second_list 相等
    assert first_list == second_list

    # 断言 first_list 与 third_list 不相等
    assert first_list != third_list

    # 创建四个 2x2 的 MutableDenseNDimArray 实例，分别用 first_list, second_list, third_list 填充
    first_ndim_array = MutableDenseNDimArray(first_list, (2, 2))
    second_ndim_array = MutableDenseNDimArray(second_list, (2, 2))
    third_ndim_array = MutableDenseNDimArray(third_list, (2, 2
    # 创建一个3x3的可变密集多维数组，元素初始化为7
    b = MutableDenseNDimArray([7 for i in range(9)], (3, 3))
    
    # 对数组a和b进行加法操作，结果保存在c1和c2中
    c1 = a + b
    c2 = b + a
    # 断言c1和c2相等
    assert c1 == c2
    
    # 对数组a和b进行减法操作，结果保存在d1和d2中
    d1 = a - b
    d2 = b - a
    # 断言d1等于d2乘以-1
    assert d1 == d2 * (-1)
    
    # 将数组a与标量5进行乘法操作，结果分别保存在e1、e2和e3中
    e1 = a * 5
    e2 = 5 * a
    e3 = copy(a)
    e3 *= 5
    # 断言e1、e2和e3相等
    assert e1 == e2 == e3
    
    # 将数组a与标量5进行除法操作，结果分别保存在f1和f2中
    f1 = a / 5
    f2 = copy(a)
    f2 /= 5
    # 断言f1等于f2，并且f1的特定元素都等于有理数3/5
    assert f1 == f2
    assert f1[0, 0] == f1[0, 1] == f1[0, 2] == f1[1, 0] == f1[1, 1] == \
        f1[1, 2] == f1[2, 0] == f1[2, 1] == f1[2, 2] == Rational(3, 5)
    
    # 断言a、b、c1、c2、d1、d2、e1、e2、e3和f1的类型都相同
    assert type(a) == type(b) == type(c1) == type(c2) == type(d1) == type(d2) \
        == type(e1) == type(e2) == type(e3) == type(f1)
    
    # 对数组a进行取负操作，结果保存在z0中
    z0 = -a
    # 断言z0等于元素全部为-3的3x3数组
    assert z0 == MutableDenseNDimArray([-3 for i in range(9)], (3, 3))
# 定义测试函数 test_higher_dimenions
def test_higher_dimenions():
    # 创建一个 MutableDenseNDimArray 对象 m3，包含范围为 10 到 33 的元素，形状为 (2, 3, 4)
    m3 = MutableDenseNDimArray(range(10, 34), (2, 3, 4))

    # 断言 m3 转换为列表形式与指定的多维列表相等
    assert m3.tolist() == [[[10, 11, 12, 13],
            [14, 15, 16, 17],
            [18, 19, 20, 21]],

           [[22, 23, 24, 25],
            [26, 27, 28, 29],
            [30, 31, 32, 33]]]

    # 断言通过索引 0 获得的元组索引与预期相等
    assert m3._get_tuple_index(0) == (0, 0, 0)
    # 断言通过索引 1 获得的元组索引与预期相等
    assert m3._get_tuple_index(1) == (0, 0, 1)
    # 断言通过索引 4 获得的元组索引与预期相等
    assert m3._get_tuple_index(4) == (0, 1, 0)
    # 断言通过索引 12 获得的元组索引与预期相等
    assert m3._get_tuple_index(12) == (1, 0, 0)

    # 断言 m3 转换为字符串形式与指定的字符串表示相等
    assert str(m3) == '[[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]]'

    # 通过指定的多维列表创建一个新的 MutableDenseNDimArray 对象 m3_rebuilt
    m3_rebuilt = MutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]])
    # 断言 m3 与 m3_rebuilt 相等
    assert m3 == m3_rebuilt

    # 通过指定的多维列表和形状创建一个新的 MutableDenseNDimArray 对象 m3_other
    m3_other = MutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]], (2, 3, 4))

    # 断言 m3 与 m3_other 相等
    assert m3 == m3_other


# 定义测试函数 test_slices
def test_slices():
    # 创建一个 MutableDenseNDimArray 对象 md，包含范围为 10 到 33 的元素，形状为 (2, 3, 4)
    md = MutableDenseNDimArray(range(10, 34), (2, 3, 4))

    # 断言 md 的切片 [:] 与指定的 MutableDenseNDimArray 对象相等
    assert md[:] == MutableDenseNDimArray(range(10, 34), (2, 3, 4))
    # 断言 md 的切片 [:, :, 0] 转换为矩阵形式与指定的矩阵相等
    assert md[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    # 断言 md 的切片 [0, 1:2, :] 转换为矩阵形式与指定的矩阵相等
    assert md[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    # 断言 md 的切片 [0, 1:3, :] 转换为矩阵形式与指定的矩阵相等
    assert md[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    # 断言 md 的切片 [:, :, :] 与 md 相等
    assert md[:, :, :] == md

    # 创建一个 MutableSparseNDimArray 对象 sd，与 md 相同
    sd = MutableSparseNDimArray(range(10, 34), (2, 3, 4))
    # 断言 sd 与 MutableSparseNDimArray(md) 相等
    assert sd == MutableSparseNDimArray(md)

    # 断言 sd 的切片 [:] 与指定的 MutableSparseNDimArray 对象相等
    assert sd[:] == MutableSparseNDimArray(range(10, 34), (2, 3, 4))
    # 断言 sd 的切片 [:, :, 0] 转换为矩阵形式与指定的矩阵相等
    assert sd[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    # 断言 sd 的切片 [0, 1:2, :] 转换为矩阵形式与指定的矩阵相等
    assert sd[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    # 断言 sd 的切片 [0, 1:3, :] 转换为矩阵形式与指定的矩阵相等
    assert sd[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    # 断言 sd 的切片 [:, :, :] 与 sd 相等
    assert sd[:, :, :] == sd


# 定义测试函数 test_slices_assign
def test_slices_assign():
    # 创建一个 MutableDenseNDimArray 对象 a，包含范围为 0 到 11 的元素，形状为 (4, 3)
    a = MutableDenseNDimArray(range(12), shape=(4, 3))
    # 创建一个 MutableSparseNDimArray 对象 b，包含范围为 0 到 11 的元素，形状为 (4, 3)
    b = MutableSparseNDimArray(range(12), shape=(4, 3))

    # 对 a 和 b 进行迭代
    for i in [a, b]:
        # 断言 i 转换为列表形式与指定的多维列表相等
        assert i.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        # 修改 i 的切片 [0, :]，设置为 [2, 2, 2]
        i[0, :] = [2, 2, 2]
        # 断言 i 转换为列表形式与预期相等
        assert i.tolist() == [[2, 2, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        # 修改 i 的切片 [0, 1:]，设置为 [8, 8]
        i[0, 1:] = [8, 8]
        # 断言 i 转换为列表形式与预期相等
        assert i.tolist() == [[2, 8, 8], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        # 修改 i 的切片 [1:3, 1]，设置为 [20, 44]
        i[1:3, 1] = [20, 44]
        # 断言 i 转换为列表形式
```