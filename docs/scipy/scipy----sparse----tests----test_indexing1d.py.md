# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_indexing1d.py`

```
# 导入 contextlib 库，用于管理上下文
import contextlib
# 导入 pytest 库，用于编写和运行测试
import pytest
# 导入 numpy 库，并将其重命名为 np，用于数值计算
import numpy as np
# 从 numpy.testing 模块中导入 assert_allclose 和 assert_equal 函数，用于测试数值计算中的相等性和接近性
from numpy.testing import assert_allclose, assert_equal
# 从 scipy.sparse 库中导入 csr_array 和 dok_array 类，以及 SparseEfficiencyWarning 警告
from scipy.sparse import csr_array, dok_array, SparseEfficiencyWarning
# 从当前目录中的 test_arithmetic1d 模块中导入 toarray 函数
from .test_arithmetic1d import toarray

# 定义一个列表，包含了 csr_array 和 dok_array 类，用于参数化测试
formats_for_index1d = [csr_array, dok_array]

# 定义一个上下文管理器函数 check_remains_sorted，用于检查排序索引属性是否在操作中保持不变
@contextlib.contextmanager
def check_remains_sorted(X):
    """Checks that sorted indices property is retained through an operation"""
    yield
    # 如果 X 没有属性 'has_sorted_indices' 或者 'has_sorted_indices' 是 False，则直接返回
    if not hasattr(X, 'has_sorted_indices') or not X.has_sorted_indices:
        return
    # 备份当前的 indices 属性
    indices = X.indices.copy()
    # 将 'has_sorted_indices' 设为 False，并重新排序 indices
    X.has_sorted_indices = False
    X.sort_indices()
    # 检查排序后的 indices 是否与备份的一致，以确保排序属性得到保持
    assert_equal(indices, X.indices, 'Expected sorted indices, found unsorted')

# 使用 pytest.mark.parametrize 装饰器对 TestGetSet1D 类进行参数化测试，其中参数 spcreator 取自 formats_for_index1d 列表
@pytest.mark.parametrize("spcreator", formats_for_index1d)
class TestGetSet1D:
    # 定义测试方法 test_None_index，测试当索引为 None 时的情况
    def test_None_index(self, spcreator):
        # 创建一个 numpy 数组 D
        D = np.array([4, 3, 0])
        # 使用 spcreator 函数创建一个稀疏数组 A
        A = spcreator(D)

        # 计算数组 D 的长度
        N = D.shape[0]
        # 遍历 [-N, N) 范围内的索引 j
        for j in range(-N, N):
            # 断言获取 A[j, None] 的结果与 D[j, None] 相等
            assert_equal(A[j, None].toarray(), D[j, None])
            # 断言获取 A[None, j] 的结果与 D[None, j] 相等
            assert_equal(A[None, j].toarray(), D[None, j])
            # 断言获取 A[None, None, j] 的结果与 D[None, None, j] 相等
            assert_equal(A[None, None, j].toarray(), D[None, None, j])

    # 定义测试方法 test_getitem_shape，测试获取元素时返回的数组形状
    def test_getitem_shape(self, spcreator):
        # 使用 spcreator 函数创建一个稀疏数组 A
        A = spcreator(np.arange(3 * 4).reshape(3, 4))
        # 断言获取 A[1, 2] 的维度为 0
        assert A[1, 2].ndim == 0
        # 断言获取 A[1, 2:3] 的形状为 (1,)
        assert A[1, 2:3].shape == (1,)
        # 断言获取 A[None, 1, 2:3] 的形状为 (1, 1)
        assert A[None, 1, 2:3].shape == (1, 1)
        # 断言获取 A[None, 1, 2] 的形状为 (1,)
        assert A[None, 1, 2].shape == (1,)
        # 断言获取 A[None, 1, 2, None] 的形状为 (1, 1)
        assert A[None, 1, 2, None].shape == (1, 1)

        # 使用 pytest.raises 检查 IndexError 异常，匹配 'Only 1D or 2D arrays' 的错误信息
        with pytest.raises(IndexError, match='Only 1D or 2D arrays'):
            A[None, 2, 1, None, None]
        with pytest.raises(IndexError, match='Only 1D or 2D arrays'):
            A[None, 0:2, None, 1]
        with pytest.raises(IndexError, match='Only 1D or 2D arrays'):
            A[0:1, 1:, None]
        with pytest.raises(IndexError, match='Only 1D or 2D arrays'):
            A[1:, 1, None, None]

    # 定义测试方法 test_getelement，测试获取单个元素的情况
    def test_getelement(self, spcreator):
        # 创建一个 numpy 数组 D
        D = np.array([4, 3, 0])
        # 使用 spcreator 函数创建一个稀疏数组 A
        A = spcreator(D)

        # 计算数组 D 的长度
        N = D.shape[0]
        # 遍历 [-N, N) 范围内的索引 j
        for j in range(-N, N):
            # 断言获取 A[j] 的结果与 D[j] 相等
            assert_equal(A[j], D[j])

        # 对于不合法的索引值 ij，使用 pytest.raises 检查 IndexError 异常，匹配 'index (.*) out of (range|bounds)' 的错误信息
        for ij in [3, -4]:
            with pytest.raises(IndexError, match='index (.*) out of (range|bounds)'):
                A.__getitem__(ij)

        # 对于单元素元组，直接解包并获取元素值
        assert A[(0,)] == 4

        # 使用 pytest.raises 检查 IndexError 异常，匹配 'index (.*) out of (range|bounds)' 的错误信息
        with pytest.raises(IndexError, match='index (.*) out of (range|bounds)'):
            A.__getitem__((4,))
    # 定义一个测试方法，用于测试稀疏数据结构的元素设置功能，spcreator 是创建稀疏数组的函数
    def test_setelement(self, spcreator):
        # 指定数据类型为 np.float64
        dtype = np.float64
        # 调用 spcreator 函数创建一个形状为 (12,) 的稀疏数组 A，数据类型为 dtype
        A = spcreator((12,), dtype=dtype)
        # 使用 np.testing.suppress_warnings() 上下文管理器来忽略特定类型的警告
        with np.testing.suppress_warnings() as sup:
            # 设置警告过滤器，过滤 SparseEfficiencyWarning 类型的警告信息
            sup.filter(
                SparseEfficiencyWarning,
                "Changing the sparsity structure of .* is expensive",
            )
            # 设置稀疏数组 A 的各个元素的值
            A[0] = dtype(0)      # 设置第一个元素为 0
            A[1] = dtype(3)      # 设置第二个元素为 3
            A[8] = dtype(9.0)    # 设置第九个元素为 9.0
            A[-2] = dtype(7)     # 设置倒数第二个元素为 7
            A[5] = 9             # 设置第五个元素为 9

            A[-9,] = dtype(8)    # 使用索引 (-9,) 设置元素为 8
            A[1,] = dtype(5)     # 使用索引 (1,) 重写元素为 5（使用 1-tuple 索引）

            # 遍历索引列表 [13, -14, (13,), (14,)]
            for ij in [13, -14, (13,), (14,)]:
                # 使用 pytest.raises() 断言捕获 IndexError 异常，匹配包含 'out of range' 或 'out of bounds' 的错误信息
                with pytest.raises(IndexError, match='out of (range|bounds)'):
                    # 调用稀疏数组 A 的 __setitem__ 方法，尝试设置索引 ij 处的元素为 123.0
                    A.__setitem__(ij, 123.0)
@pytest.mark.parametrize("spcreator", formats_for_index1d)
class TestSlicingAndFancy1D:
    #######################
    #  Int-like Array Index
    #######################

    # 测试获取整数数组索引的功能
    def test_get_array_index(self, spcreator):
        # 创建一个NumPy数组
        D = np.array([4, 3, 0])
        # 使用 spcreator 函数创建稀疏矩阵 A
        A = spcreator(D)

        # 断言稀疏矩阵 A 使用空元组索引的结果与数组 D 使用空元组索引的结果相同
        assert_equal(A[()].toarray(), D[()])
        
        # 遍历测试以下索引：(0, 3) 和 (3,)
        for ij in [(0, 3), (3,)]:
            # 使用 pytest 来确保索引出界时会抛出 IndexError 异常，并匹配特定错误信息
            with pytest.raises(IndexError, match='out of (range|bounds)|many indices'):
                A.__getitem__(ij)

    # 测试设置整数数组索引的功能
    def test_set_array_index(self, spcreator):
        # 指定数据类型为 np.float64
        dtype = np.float64
        # 使用 spcreator 创建一个形状为 (12,) 的稀疏矩阵 A
        A = spcreator((12,), dtype=dtype)
        
        # 使用 np.testing.suppress_warnings 来忽略特定警告信息
        with np.testing.suppress_warnings() as sup:
            sup.filter(
                SparseEfficiencyWarning,
                "Changing the sparsity structure of .* is expensive",
            )
            
            # 使用标量索引对 A 进行赋值操作
            A[np.array(6)] = dtype(4.0)
            # 再次使用标量索引对 A 进行赋值操作
            A[np.array(6)] = dtype(2.0)
            # 断言 A 转换成普通数组后的结果符合预期
            assert_equal(A.toarray(), [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0])

            # 遍历测试以下索引：(13,), (-14,)
            for ij in [(13,), (-14,)]:
                # 使用 pytest 来确保索引超出范围时会抛出 IndexError 异常，并匹配特定错误信息
                with pytest.raises(IndexError, match='index .* out of (range|bounds)'):
                    A.__setitem__(ij, 123.0)

            # 遍历测试以下不合法的赋值操作
            for v in [(), (0, 3), [1, 2, 3], np.array([1, 2, 3])]:
                msg = 'Trying to assign a sequence to an item'
                # 使用 pytest 来确保赋值不合法时会抛出 ValueError 异常，并匹配特定错误信息
                with pytest.raises(ValueError, match=msg):
                    A.__setitem__(0, v)

    ####################
    #  1d Slice as index
    ####################

    # 测试切片作为索引时数据类型的保留功能
    def test_dtype_preservation(self, spcreator):
        # 断言切片索引的稀疏矩阵的数据类型与预期的一致
        assert_equal(spcreator((10,), dtype=np.int16)[1:5].dtype, np.int16)
        assert_equal(spcreator((6,), dtype=np.int32)[0:0:2].dtype, np.int32)
        assert_equal(spcreator((6,), dtype=np.int64)[:].dtype, np.int64)

    # 测试获取使用切片索引时的功能
    def test_get_1d_slice(self, spcreator):
        # 创建一个普通的 NumPy 数组 B
        B = np.arange(50.)
        # 使用 spcreator 函数创建稀疏矩阵 A
        A = spcreator(B)
        # 断言使用切片索引时稀疏矩阵 A 的结果与数组 B 的预期结果一致
        assert_equal(B[:], A[:].toarray())
        assert_equal(B[2:5], A[2:5].toarray())

        # 创建一个数组 C
        C = np.array([4, 0, 6, 0, 0, 0, 0, 0, 1])
        # 使用 spcreator 函数创建稀疏矩阵 D
        D = spcreator(C)
        # 断言使用切片索引时稀疏矩阵 D 的结果与数组 C 的预期结果一致
        assert_equal(C[1:3], D[1:3].toarray())

        # 测试当行包含全部为零时的切片索引情况
        E = np.array([0, 0, 0, 0, 0])
        F = spcreator(E)
        # 断言使用切片索引时稀疏矩阵 F 的结果与数组 E 的预期结果一致
        assert_equal(E[1:3], F[1:3].toarray())
        assert_equal(E[-2:], F[-2:].toarray())
        assert_equal(E[:], F[:].toarray())
        assert_equal(E[slice(None)], F[slice(None)].toarray())
    def test_slicing_idx_slice(self, spcreator):
        # 创建一个长度为50的NumPy数组B，用于测试
        B = np.arange(50)
        # 调用spcreator函数创建稀疏矩阵A，基于B的内容
        A = spcreator(B)

        # [i]
        # 验证索引操作：A[2]应该等于B[2]
        assert_equal(A[2], B[2])
        # 验证负数索引操作：A[-1]应该等于B[-1]
        assert_equal(A[-1], B[-1])
        # 使用np.array创建负数索引：A[np.array(-2)]应该等于B[-2]
        assert_equal(A[np.array(-2)], B[-2])

        # [1:2]
        # 验证切片操作：A[:]应该等于B[:]
        assert_equal(A[:].toarray(), B[:])
        # 验证带负数索引的切片操作：A[5:-2]应该等于B[5:-2]
        assert_equal(A[5:-2].toarray(), B[5:-2])
        # 验证带步长的切片操作：A[5:12:3]应该等于B[5:12:3]
        assert_equal(A[5:12:3].toarray(), B[5:12:3])

        # int8类型切片
        s = slice(np.int8(2), np.int8(4), None)
        # 验证int8类型的切片操作：A[s]应该等于B[2:4]
        assert_equal(A[s].toarray(), B[2:4])

        # 使用np.s_进行切片
        s_ = np.s_
        slices = [s_[:2], s_[1:2], s_[3:], s_[3::2],
                  s_[15:20], s_[3:2],
                  s_[8:3:-1], s_[4::-2], s_[:5:-1],
                  0, 1, s_[:], s_[1:5], -1, -2, -5,
                  np.array(-1), np.int8(-3)]

        for j, a in enumerate(slices):
            # 对每个切片a，从A中取出对应的子集x，从B中取出对应的子集y
            x = A[a]
            y = B[a]
            if y.shape == ():
                # 如果y是标量，则验证x与y相等
                assert_equal(x, y, repr(a))
            else:
                # 否则，验证稀疏矩阵x的稀疏表示是否等于B中对应的稀疏表示y
                if x.size == 0 and y.size == 0:
                    pass
                else:
                    assert_equal(x.toarray(), y, repr(a))

    def test_ellipsis_1d_slicing(self, spcreator):
        # 创建一个长度为50的NumPy数组B，用于测试
        B = np.arange(50)
        # 调用spcreator函数创建稀疏矩阵A，基于B的内容
        A = spcreator(B)
        # 验证省略号(...)的切片操作：A[...]应该等于B[...]
        assert_equal(A[...].toarray(), B[...])
        # 验证省略号(...)的切片操作（带逗号）：A[...,]应该等于B[...,]
        assert_equal(A[...,].toarray(), B[...,])

    ##########################
    #  Assignment with Slicing
    ##########################
    def test_slice_scalar_assign(self, spcreator):
        # 创建一个长度为5的稀疏矩阵A
        A = spcreator((5,))
        # 创建一个长度为5的全零NumPy数组B
        B = np.zeros((5,))
        # 忽略稀疏效率警告
        with np.testing.suppress_warnings() as sup:
            sup.filter(
                SparseEfficiencyWarning,
                "Changing the sparsity structure of .* is expensive",
            )
            # 对A和B进行赋值操作的测试
            for C in [A, B]:
                C[0:1] = 1
                C[2:0] = 4
                C[2:3] = 9
                C[3:] = 1
                C[3::-1] = 9
        # 验证A的稀疏表示是否等于B
        assert_equal(A.toarray(), B)

    def test_slice_assign_2(self, spcreator):
        shape = (10,)
        # 遍历不同的切片方式
        for idx in [slice(3), slice(None, 10, 4), slice(5, -2)]:
            # 创建一个shape为(10,)的稀疏矩阵A
            A = spcreator(shape)
            # 忽略稀疏效率警告
            with np.testing.suppress_warnings() as sup:
                sup.filter(
                    SparseEfficiencyWarning,
                    "Changing the sparsity structure of .* is expensive",
                )
                # 对A进行赋值操作
                A[idx] = 1
            # 创建一个全零NumPy数组B
            B = np.zeros(shape)
            B[idx] = 1
            msg = f"idx={idx!r}"
            # 验证A的稀疏表示是否等于B
            assert_allclose(A.toarray(), B, err_msg=msg)
    def test_self_self_assignment(self, spcreator):
        # Tests whether a row of one lil_matrix can be assigned to another.
        # 创建一个形状为 (5,) 的稀疏矩阵 B
        B = spcreator((5,))
        
        # 使用 numpy 的 suppress_warnings 上下文管理器，过滤 SparseEfficiencyWarning
        with np.testing.suppress_warnings() as sup:
            sup.filter(
                SparseEfficiencyWarning,
                "Changing the sparsity structure of .* is expensive",
            )
            
            # 对 B 的不同行进行赋值操作
            B[0] = 2
            B[1] = 0
            B[2] = 3
            B[3] = 10

            # 将 B 的每个元素除以 10，赋值给 A
            A = B / 10
            # 将 A 的内容复制到 B
            B[:] = A[:]
            # 断言 A 和 B 的内容是否一致
            assert_equal(A[:].toarray(), B[:].toarray())

            # 将 A 的部分内容赋值给 B
            A = B / 10
            B[:] = A[:1]
            # 断言 B 的前两行内容是否与 A 的第一行相等
            assert_equal(np.zeros((5,)) + A[0], B.toarray())

            # 将 A 的部分内容赋值给 B 的前四行
            A = B / 10
            B[:-1] = A[1:]
            # 断言 B 的前四行内容是否与 A 的后三行相等
            assert_equal(A[1:].toarray(), B[:-1].toarray())

    def test_slice_assignment(self, spcreator):
        # 创建一个形状为 (4,) 的稀疏矩阵 B
        B = spcreator((4,))
        # 预期结果为 [10, 0, 14, 0]
        expected = np.array([10, 0, 14, 0])
        # block 数组
        block = [2, 1]

        # 使用 numpy 的 suppress_warnings 上下文管理器，过滤 SparseEfficiencyWarning
        with np.testing.suppress_warnings() as sup:
            sup.filter(
                SparseEfficiencyWarning,
                "Changing the sparsity structure of .* is expensive",
            )
            
            # 对 B 的不同位置进行赋值操作
            B[0] = 5
            B[2] = 7
            # 将 B 本身加上 B 的内容并赋值给 B
            B[:] = B + B
            # 断言 B 的内容是否与 expected 相等
            assert_equal(B.toarray(), expected)

            # 将 block 数组赋值给 B 的前两行
            B[:2] = csr_array(block)
            # 断言 B 的前两行内容是否与 block 数组相等
            assert_equal(B.toarray()[:2], block)

    def test_set_slice(self, spcreator):
        # 创建一个形状为 (5,) 的稀疏矩阵 A
        A = spcreator((5,))
        # 创建一个形状为 (5,) 的全零数组 B
        B = np.zeros(5, float)
        s_ = np.s_
        # 定义一系列切片
        slices = [s_[:2], s_[1:2], s_[3:], s_[3::2],
                  s_[8:3:-1], s_[4::-2], s_[:5:-1],
                  0, 1, s_[:], s_[1:5], -1, -2, -5,
                  np.array(-1), np.int8(-3)]

        # 使用 numpy 的 suppress_warnings 上下文管理器，过滤 SparseEfficiencyWarning
        with np.testing.suppress_warnings() as sup:
            sup.filter(
                SparseEfficiencyWarning,
                "Changing the sparsity structure of .* is expensive",
            )
            
            # 遍历 slices 中的切片，将索引赋值给 A 和 B
            for j, a in enumerate(slices):
                A[a] = j
                B[a] = j
                # 断言 A 和 B 的内容是否相等，打印当前切片的表示形式
                assert_equal(A.toarray(), B, repr(a))

            # 将 range(1, 5, 2) 赋值给 A 的奇数索引位置（1, 3）
            A[1:10:2] = range(1, 5, 2)
            B[1:10:2] = range(1, 5, 2)
            # 断言 A 和 B 的内容是否相等
            assert_equal(A.toarray(), B)

        # 以下命令应该引发异常
        toobig = list(range(100))
        # 断言尝试将一个序列赋值给单个元素时是否引发 ValueError 异常
        with pytest.raises(ValueError, match='Trying to assign a sequence to an item'):
            A.__setitem__(0, toobig)
        # 断言尝试将大小不匹配的序列赋值给切片时是否引发 ValueError 异常
        with pytest.raises(ValueError, match='could not be broadcast together'):
            A.__setitem__(slice(None), toobig)

    def test_assign_empty(self, spcreator):
        # 创建一个形状为 (3,) 的稀疏矩阵 A，其元素为全 1
        A = spcreator(np.ones(3))
        # 创建一个形状为 (2,) 的稀疏矩阵 B
        B = spcreator((2,))
        # 将 B 赋值给 A 的前两个元素
        A[:2] = B
        # 断言 A 的内容是否为 [0, 0, 1]
        assert_equal(A.toarray(), [0, 0, 1])

    ####################
    #  1d Fancy Indexing
    ####################
    def test_dtype_preservation_empty_index(self, spcreator):
        # 创建一个形状为 (2,) 的稀疏矩阵 A，数据类型为 np.int16
        A = spcreator((2,), dtype=np.int16)
        # 断言 A 的空索引的数据类型是否为 np.int16
        assert_equal(A[[False, False]].dtype, np.int16)
        # 断言 A 的空列表索引的数据类型是否为 np.int16
        assert_equal(A[[]].dtype, np.int16)
    # 定义一个测试方法，用于测试在特殊矩阵创建器上的错误索引操作
    def test_bad_index(self, spcreator):
        # 创建一个特殊矩阵 A，基于全零数组
        A = spcreator(np.zeros(5))
        
        # 测试索引错误，期望引发 IndexError、ValueError 或 TypeError 异常，
        # 匹配指定的错误消息字符串
        with pytest.raises(
            (IndexError, ValueError, TypeError),
            match='Index dimension must be 1 or 2|only integers',
        ):
            # 尝试使用非整数索引"foo"，调用 __getitem__ 方法
            A.__getitem__("foo")
        
        # 再次测试索引错误，期望引发 IndexError、ValueError 或 TypeError 异常，
        # 匹配指定的错误消息字符串
        with pytest.raises(
            (IndexError, ValueError, TypeError),
            match='tuple index out of range|only integers',
        ):
            # 尝试使用元组索引中包含非整数元素"foo"，调用 __getitem__ 方法
            A.__getitem__((2, "foo"))

    # 定义一个测试方法，用于测试在特殊矩阵创建器上的高级索引操作
    def test_fancy_indexing(self, spcreator):
        # 创建一个数组 B，包含从 0 到 49 的整数
        B = np.arange(50)
        
        # 使用特殊矩阵创建器 spcreator 创建矩阵 A，基于数组 B
        A = spcreator(B)

        # 测试使用基本索引 [i] 的功能
        assert_equal(A[[3]].toarray(), B[[3]])

        # 测试使用 np.array 类型的索引数组
        assert_equal(A[[1, 3]].toarray(), B[[1, 3]])
        assert_equal(A[[2, -5]].toarray(), B[[2, -5]])
        assert_equal(A[np.array(-1)], B[-1])
        assert_equal(A[np.array([-1, 2])].toarray(), B[[-1, 2]])
        assert_equal(A[np.array(5)], B[np.array(5)])

        # 测试使用多维索引数组 [[[1],[2]]]
        ind = np.array([[1], [3]])
        assert_equal(A[ind].toarray(), B[ind])
        ind = np.array([[-1], [-3], [-2]])
        assert_equal(A[ind].toarray(), B[ind])

        # 测试混合正整数索引 [[1, 2]]
        assert_equal(A[[1, 3]].toarray(), B[[1, 3]])
        assert_equal(A[[-1, -3]].toarray(), B[[-1, -3]])
        assert_equal(A[np.array([-1, -3])].toarray(), B[[-1, -3]])

        # 测试混合正整数索引和高级索引 [[1, 2]][[1, 2]]
        assert_equal(A[[1, 5, 2, 8]][[1, 3]].toarray(),
                     B[[1, 5, 2, 8]][[1, 3]])
        assert_equal(A[[-1, -5, 2, 8]][[1, -4]].toarray(),
                     B[[-1, -5, 2, 8]][[1, -4]])

    # 定义一个测试方法，用于测试在特殊矩阵创建器上的布尔数组高级索引操作
    def test_fancy_indexing_boolean(self, spcreator):
        # 设置随机数种子，以确保结果可重复
        np.random.seed(1234)

        # 创建一个数组 B，包含从 0 到 49 的整数
        B = np.arange(50)
        
        # 使用特殊矩阵创建器 spcreator 创建矩阵 A，基于数组 B
        A = spcreator(B)

        # 创建一个布尔索引数组 I，其中元素为随机生成的 0 或 1
        I = np.array(np.random.randint(0, 2, size=50), dtype=bool)

        # 测试布尔数组索引的功能
        assert_equal(toarray(A[I]), B[I])
        assert_equal(toarray(A[B > 9]), B[B > 9])

        # 创建三个长度为 51 的布尔数组
        Z1 = np.zeros(51, dtype=bool)
        Z2 = np.zeros(51, dtype=bool)
        Z2[-1] = True
        Z3 = np.zeros(51, dtype=bool)
        Z3[0] = True

        # 测试错误的布尔数组索引，期望引发 IndexError 异常，
        # 匹配指定的错误消息字符串
        msg = 'bool index .* has shape|boolean index did not match'
        with pytest.raises(IndexError, match=msg):
            A.__getitem__(Z1)
        with pytest.raises(IndexError, match=msg):
            A.__getitem__(Z2)
        with pytest.raises(IndexError, match=msg):
            A.__getitem__(Z3)
    ############################
    #  1d Fancy Index Assignment
    ############################
    # 测试不良的索引赋值操作
    def test_bad_index_assign(self, spcreator):
        # 创建一个长度为5的稀疏数组
        A = spcreator(np.zeros(5))
        # 错误信息字符串，匹配其中包含 'Index dimension must be 1 or 2' 或 'only integers' 的异常
        msg = 'Index dimension must be 1 or 2|only integers'
        # 使用 pytest 检测是否会抛出 IndexError、ValueError 或 TypeError 异常，并匹配上述错误信息
        with pytest.raises((IndexError, ValueError, TypeError), match=msg):
            # 尝试对 A 使用字符串 "foo" 进行索引赋值操作
            A.__setitem__("foo", 2)

    # 测试复杂索引设置操作
    def test_fancy_indexing_set(self, spcreator):
        # 创建一个元组 M，包含单个元素 5
        M = (5,)

        # 对以下几种复杂索引 j 进行测试
        for j in [
            [2, 3, 4],                # 索引列表
            slice(None, 10, 4),       # 切片操作，从开头到10，步长为4
            np.arange(3),             # 使用 NumPy 的 arange 函数生成索引数组
            slice(5, -2),             # 切片操作，从5到倒数第二个元素
            slice(2, 5)               # 切片操作，从2到5
        ]:
            # 创建一个稀疏数组 A，其形状由 M 决定
            A = spcreator(M)
            # 创建一个与 A 形状相同的全零数组 B
            B = np.zeros(M)
            # 使用 np.testing.suppress_warnings() 对 SparseEfficiencyWarning 进行警告抑制
            with np.testing.suppress_warnings() as sup:
                # 过滤掉特定警告信息，该警告表明“改变稀疏结构是昂贵的”
                sup.filter(
                    SparseEfficiencyWarning,
                    "Changing the sparsity structure of .* is expensive",
                )
                # 将数组 B 的索引 j 处设置为 1
                B[j] = 1
                # 使用 check_remains_sorted 检查 A 的排序情况是否保持不变
                with check_remains_sorted(A):
                    # 将稀疏数组 A 的索引 j 处设置为 1
                    A[j] = 1
            # 使用 assert_allclose 检查 A 转换为密集数组后的结果是否与 B 相等
            assert_allclose(A.toarray(), B)
    # 测试稀疏矩阵的序列赋值操作
    def test_sequence_assignment(self, spcreator):
        # 创建稀疏矩阵 A 和 B
        A = spcreator((4,))
        B = spcreator((3,))

        # 不同类型的索引
        i0 = [0, 1, 2]  # 列表索引
        i1 = (0, 1, 2)  # 元组索引
        i2 = np.array(i0)  # NumPy 数组索引

        # 忽略 NumPy 的警告信息
        with np.testing.suppress_warnings() as sup:
            sup.filter(
                SparseEfficiencyWarning,
                "Changing the sparsity structure of .* is expensive",
            )
            # 使用 check_remains_sorted 上下文管理器验证 A 是否保持排序
            with check_remains_sorted(A):
                # 对 A 的指定索引赋值为 B 对应索引的值
                A[i0] = B[i0]
                # 检查索引越界时是否触发 IndexError 异常
                msg = "too many indices for array|tuple index out of range"
                with pytest.raises(IndexError, match=msg):
                    B.__getitem__(i1)
                A[i2] = B[i2]
            # 验证 A 的前三个元素是否与 B 的数组表示相等
            assert_equal(A[:3].toarray(), B.toarray())
            # 验证 A 的形状是否为 (4,)
            assert A.shape == (4,)

            # 切片赋值操作
            A = spcreator((4,))
            with check_remains_sorted(A):
                A[1:3] = [10, 20]
            # 验证 A 的数组表示是否符合预期
            assert_equal(A.toarray(), [0, 10, 20, 0])

            # 使用数组赋值操作
            A = spcreator((4,))
            B = np.zeros(4)
            with check_remains_sorted(A):
                # 对 A 和 B 中的指定索引同时赋值相同的数组
                for C in [A, B]:
                    C[[0, 1, 2]] = [4, 5, 6]
            # 验证 A 的数组表示是否与 B 相等
            assert_equal(A.toarray(), B)

    # 测试空索引赋值操作
    def test_fancy_assign_empty(self, spcreator):
        # 创建一个数组 B
        B = np.arange(50)
        B[2] = 0
        B[[3, 6]] = 0
        # 使用数组 B 创建稀疏矩阵 A
        A = spcreator(B)

        # 创建全为 False 的布尔数组 K
        K = np.array([False] * 50)
        # 对 A 中 K 为 True 的元素赋值为 42
        A[K] = 42
        # 验证 A 的数组表示是否与 B 相等
        assert_equal(A.toarray(), B)

        # 创建空数组 K
        K = np.array([], dtype=int)
        # 对 A 中 K 为 True 的元素赋值为 42
        A[K] = 42
        # 再次验证 A 的数组表示是否与 B 相等
        assert_equal(A.toarray(), B)
```