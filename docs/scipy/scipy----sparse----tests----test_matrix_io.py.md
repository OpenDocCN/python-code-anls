# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_matrix_io.py`

```
# 导入必要的库和模块
import os                    # 系统操作模块
import numpy as np           # 数值计算库
import tempfile              # 临时文件和目录模块

from pytest import raises as assert_raises  # 引入 pytest 的 raises 功能并重命名为 assert_raises
from numpy.testing import assert_equal, assert_  # 引入 numpy 测试相关的断言方法

from scipy.sparse import (sparray,     # 稀疏数组的基类
                          csc_matrix,  # CSC 格式的稀疏矩阵
                          csr_matrix,  # CSR 格式的稀疏矩阵
                          bsr_matrix,  # BSR 格式的稀疏矩阵
                          dia_matrix,  # DIA 格式的稀疏矩阵
                          coo_matrix,  # COO 格式的稀疏矩阵
                          dok_matrix,  # DOK 格式的稀疏矩阵
                          csr_array,   # CSR 格式的稀疏数组
                          save_npz,    # 保存为 NPZ 文件
                          load_npz)    # 加载 NPZ 文件


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')  # 设置数据目录为当前文件所在目录下的 'data' 子目录


def _save_and_load(matrix):
    # 创建临时文件，并返回文件描述符和文件名
    fd, tmpfile = tempfile.mkstemp(suffix='.npz')
    os.close(fd)  # 关闭文件描述符
    try:
        save_npz(tmpfile, matrix)  # 保存稀疏矩阵到 NPZ 文件
        loaded_matrix = load_npz(tmpfile)  # 从 NPZ 文件中加载矩阵
    finally:
        os.remove(tmpfile)  # 最后删除临时文件
    return loaded_matrix  # 返回加载后的矩阵对象


def _check_save_and_load(dense_matrix):
    # 针对不同的稀疏矩阵格式进行测试
    for matrix_class in [csc_matrix, csr_matrix, bsr_matrix, dia_matrix, coo_matrix]:
        matrix = matrix_class(dense_matrix)  # 根据密集矩阵创建稀疏矩阵
        loaded_matrix = _save_and_load(matrix)  # 调用保存和加载函数
        assert_(type(loaded_matrix) is matrix_class)  # 断言加载后的对象类型与原对象类型相同
        assert_(loaded_matrix.shape == dense_matrix.shape)  # 断言加载后的对象形状与原对象相同
        assert_(loaded_matrix.dtype == dense_matrix.dtype)  # 断言加载后的对象数据类型与原对象相同
        assert_equal(loaded_matrix.toarray(), dense_matrix)  # 断言加载后的稀疏矩阵转换为密集矩阵后与原始密集矩阵相等


def test_save_and_load_random():
    N = 10
    np.random.seed(0)
    dense_matrix = np.random.random((N, N))  # 创建一个随机的密集矩阵
    dense_matrix[dense_matrix > 0.7] = 0  # 将大于0.7的元素置为0，稀疏化矩阵
    _check_save_and_load(dense_matrix)  # 调用保存和加载测试函数进行测试


def test_save_and_load_empty():
    dense_matrix = np.zeros((4,6))  # 创建一个全零的密集矩阵
    _check_save_and_load(dense_matrix)  # 调用保存和加载测试函数进行测试


def test_save_and_load_one_entry():
    dense_matrix = np.zeros((4,6))  # 创建一个全零的密集矩阵
    dense_matrix[1,2] = 1  # 将某一个元素设为1，稀疏化矩阵
    _check_save_and_load(dense_matrix)  # 调用保存和加载测试函数进行测试


def test_sparray_vs_spmatrix():
    # 保存和加载稀疏矩阵及稀疏数组，并进行比较
    fd, tmpfile = tempfile.mkstemp(suffix='.npz')  # 创建临时文件，并返回文件描述符和文件名
    os.close(fd)  # 关闭文件描述符
    try:
        save_npz(tmpfile, csr_matrix([[1.2, 0, 0.9], [0, 0.3, 0]]))  # 保存 CSR 格式的稀疏矩阵到 NPZ 文件
        loaded_matrix = load_npz(tmpfile)  # 从 NPZ 文件中加载矩阵
    finally:
        os.remove(tmpfile)  # 最后删除临时文件

    fd, tmpfile = tempfile.mkstemp(suffix='.npz')  # 创建临时文件，并返回文件描述符和文件名
    os.close(fd)  # 关闭文件描述符
    try:
        save_npz(tmpfile, csr_array([[1.2, 0, 0.9], [0, 0.3, 0]]))  # 保存 CSR 格式的稀疏数组到 NPZ 文件
        loaded_array = load_npz(tmpfile)  # 从 NPZ 文件中加载数组
    finally:
        os.remove(tmpfile)  # 最后删除临时文件

    assert not isinstance(loaded_matrix, sparray)  # 断言加载后的矩阵不是 sparray 类型
    assert isinstance(loaded_array, sparray)  # 断言加载后的数组是 sparray 类型
    assert_(loaded_matrix.dtype == loaded_array.dtype)  # 断言加载后的矩阵和数组的数据类型相同
    assert_equal(loaded_matrix.toarray(), loaded_array.toarray())  # 断言加载后的矩阵和数组转换为密集矩阵后相等


def test_malicious_load():
    class Executor:
        def __reduce__(self):
            return (assert_, (False, 'unexpected code execution'))  # 返回一个包含断言的元组

    fd, tmpfile = tempfile.mkstemp(suffix='.npz')  # 创建临时文件，并返回文件描述符和文件名
    os.close(fd)  # 关闭文件描述符
    try:
        np.savez(tmpfile, format=Executor())  # 保存包含恶意代码的文件
        assert_raises(ValueError, load_npz, tmpfile)  # 断言加载时会引发 ValueError 异常
    finally:
        os.remove(tmpfile)  # 最后删除临时文件


def test_py23_compatibility():
    # 测试在 Python 2 和 Python 3 下加载保存的文件，由于 SciPy 版本 < 1.0.0 的文件可能包含 Unicode，因此结果可能不同
    a = load_npz(os.path.join(DATA_DIR, 'csc_py2.npz'))  # 加载 Python 2 下保存的文件
    b = load_npz(os.path.join(DATA_DIR, 'csc_py3.npz'))  # 加载 Python 3 下保存的文件
    c = csc_matrix([[0]])  # 创建一个 CSC 格式的稀疏矩阵
    # 断言：验证数组 a 转换为普通数组后是否与数组 c 转换为普通数组后相等
    assert_equal(a.toarray(), c.toarray())
    
    # 断言：验证数组 b 转换为普通数组后是否与数组 c 转换为普通数组后相等
    assert_equal(b.toarray(), c.toarray())
# 定义一个测试函数，用于测试保存不支持的数据类型时是否引发 NotImplementedError 异常
def test_implemented_error():
    # 创建一个稀疏矩阵 (DOK 格式)，大小为 (2,3)
    x = dok_matrix((2,3))
    # 在矩阵中指定位置 (0,1) 设置数值为 1
    x[0,1] = 1

    # 断言语句：期望调用 save_npz 函数保存对象 x 到 'x.npz' 文件时抛出 NotImplementedError 异常
    assert_raises(NotImplementedError, save_npz, 'x.npz', x)
```