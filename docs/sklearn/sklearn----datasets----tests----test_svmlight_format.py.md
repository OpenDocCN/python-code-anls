# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_svmlight_format.py`

```
import gzip  # 导入gzip模块，用于处理gzip压缩文件
import os  # 导入os模块，提供了处理操作系统相关任务的功能
import shutil  # 导入shutil模块，用于高级文件操作，如复制、删除等
from bz2 import BZ2File  # 从bz2模块导入BZ2File类，用于处理bzip2压缩文件
from importlib import resources  # 导入importlib中的resources模块，用于访问包内资源
from io import BytesIO  # 导入io模块中的BytesIO类，用于在内存中操作二进制数据流
from tempfile import NamedTemporaryFile  # 从tempfile模块导入NamedTemporaryFile类，用于创建临时文件

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例
import scipy.sparse as sp  # 导入SciPy的稀疏矩阵模块

import sklearn  # 导入scikit-learn机器学习库
from sklearn.datasets import dump_svmlight_file, load_svmlight_file, load_svmlight_files  # 从sklearn.datasets模块导入相关函数
from sklearn.utils._testing import (  # 导入sklearn.utils._testing模块中的函数和类
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    create_memmap_backed_data,
)
from sklearn.utils.fixes import CSR_CONTAINERS  # 从sklearn.utils.fixes模块导入CSR_CONTAINERS常量

TEST_DATA_MODULE = "sklearn.datasets.tests.data"  # 设置测试数据模块的路径
datafile = "svmlight_classification.txt"  # 设置用于分类的svmlight数据文件名
multifile = "svmlight_multilabel.txt"  # 设置多标签svmlight数据文件名
invalidfile = "svmlight_invalid.txt"  # 设置无效的svmlight数据文件名
invalidfile2 = "svmlight_invalid_order.txt"  # 设置顺序无效的svmlight数据文件名


def _svmlight_local_test_file_path(filename):
    """
    返回指定文件名在测试数据模块中的路径
    """
    return resources.files(TEST_DATA_MODULE) / filename


def _load_svmlight_local_test_file(filename, **kwargs):
    """
    使用importlib.resources加载资源文件，加载svmlight文件并返回数据
    """
    data_path = _svmlight_local_test_file_path(filename)
    with data_path.open("rb") as f:
        return load_svmlight_file(f, **kwargs)


def test_load_svmlight_file():
    """
    测试加载svmlight文件的功能
    """
    X, y = _load_svmlight_local_test_file(datafile)

    # 测试X的形状
    assert X.indptr.shape[0] == 7
    assert X.shape[0] == 6
    assert X.shape[1] == 21
    assert y.shape[0] == 6

    # 测试X的非零值
    for i, j, val in (
        (0, 2, 2.5),
        (0, 10, -5.2),
        (0, 15, 1.5),
        (1, 5, 1.0),
        (1, 12, -3),
        (2, 20, 27),
    ):
        assert X[i, j] == val

    # 测试X的零值
    assert X[0, 3] == 0
    assert X[0, 5] == 0
    assert X[1, 8] == 0
    assert X[1, 16] == 0
    assert X[2, 18] == 0

    # 测试可以改变X的值
    X[0, 2] *= 2
    assert X[0, 2] == 5

    # 测试y的值
    assert_array_equal(y, [1, 2, 3, 4, 1, 2])


def test_load_svmlight_file_fd():
    """
    测试从文件描述符加载svmlight文件
    """
    # 测试使用文件路径和文件描述符加载的load_svmlight_file结果是否相等
    data_path = resources.files(TEST_DATA_MODULE) / datafile
    data_path = str(data_path)
    X1, y1 = load_svmlight_file(data_path)

    fd = os.open(data_path, os.O_RDONLY)
    try:
        X2, y2 = load_svmlight_file(fd)
        assert_array_almost_equal(X1.data, X2.data)
        assert_array_almost_equal(y1, y2)
    finally:
        os.close(fd)


def test_load_svmlight_pathlib():
    """
    测试使用路径对象加载svmlight文件
    """
    data_path = _svmlight_local_test_file_path(datafile)
    X1, y1 = load_svmlight_file(str(data_path))
    X2, y2 = load_svmlight_file(data_path)

    assert_allclose(X1.data, X2.data)
    assert_allclose(y1, y2)


def test_load_svmlight_file_multilabel():
    """
    测试加载多标签svmlight文件
    """
    X, y = _load_svmlight_local_test_file(multifile, multilabel=True)
    assert y == [(0, 1), (2,), (), (1, 2)]


def test_load_svmlight_files():
    """
    测试加载多个svmlight文件
    """
    data_path = _svmlight_local_test_file_path(datafile)
    # 使用 load_svmlight_files 函数加载数据集，将训练集 X_train, y_train 和测试集 X_test, y_test 赋值给对应变量
    X_train, y_train, X_test, y_test = load_svmlight_files(
        [str(data_path)] * 2, dtype=np.float32
    )

    # 断言：验证训练集和测试集的稀疏矩阵表示是否相等
    assert_array_equal(X_train.toarray(), X_test.toarray())

    # 断言：验证训练集和测试集的标签数组是否近似相等
    assert_array_almost_equal(y_train, y_test)

    # 断言：验证训练集的数据类型是否为 np.float32
    assert X_train.dtype == np.float32

    # 断言：验证测试集的数据类型是否为 np.float32
    assert X_test.dtype == np.float32

    # 使用 load_svmlight_files 函数再次加载数据集，将三个数据集 X1, y1, X2, y2, X3, y3 赋值给对应变量，数据类型为 np.float64
    X1, y1, X2, y2, X3, y3 = load_svmlight_files([str(data_path)] * 3, dtype=np.float64)

    # 断言：验证三个数据集的特征矩阵数据类型是否相同
    assert X1.dtype == X2.dtype
    assert X2.dtype == X3.dtype

    # 断言：验证第三个数据集的特征矩阵数据类型是否为 np.float64
    assert X3.dtype == np.float64
def test_load_svmlight_file_n_features():
    # 从本地测试文件加载数据，指定特征数为22
    X, y = _load_svmlight_local_test_file(datafile, n_features=22)

    # 检验 X 的形状
    assert X.indptr.shape[0] == 7  # 确保 indptr 数组的长度为7
    assert X.shape[0] == 6  # 确保 X 的行数为6
    assert X.shape[1] == 22  # 确保 X 的列数为22

    # 检验 X 的非零值
    for i, j, val in ((0, 2, 2.5), (0, 10, -5.2), (1, 5, 1.0), (1, 12, -3)):
        assert X[i, j] == val  # 确保特定位置的值与预期值相等

    # 文件中有21个特征
    with pytest.raises(ValueError):
        _load_svmlight_local_test_file(datafile, n_features=20)  # 应当抛出 ValueError


def test_load_compressed():
    # 从本地测试文件加载数据
    X, y = _load_svmlight_local_test_file(datafile)

    # 测试 gzip 压缩文件
    with NamedTemporaryFile(prefix="sklearn-test", suffix=".gz") as tmp:
        tmp.close()  # 在 Windows 下必须关闭临时文件
        with _svmlight_local_test_file_path(datafile).open("rb") as f:
            with gzip.open(tmp.name, "wb") as fh_out:
                shutil.copyfileobj(f, fh_out)
        # 加载 gzip 压缩文件中的数据
        Xgz, ygz = load_svmlight_file(tmp.name)
        # 因为手动"关闭"并写入了临时文件，需要手动删除
        os.remove(tmp.name)
    assert_array_almost_equal(X.toarray(), Xgz.toarray())  # 确保加载的数据与原始数据一致
    assert_array_almost_equal(y, ygz)  # 确保加载的标签与原始数据一致

    # 测试 bz2 压缩文件
    with NamedTemporaryFile(prefix="sklearn-test", suffix=".bz2") as tmp:
        tmp.close()  # 在 Windows 下必须关闭临时文件
        with _svmlight_local_test_file_path(datafile).open("rb") as f:
            with BZ2File(tmp.name, "wb") as fh_out:
                shutil.copyfileobj(f, fh_out)
        # 加载 bz2 压缩文件中的数据
        Xbz, ybz = load_svmlight_file(tmp.name)
        # 因为手动"关闭"并写入了临时文件，需要手动删除
        os.remove(tmp.name)
    assert_array_almost_equal(X.toarray(), Xbz.toarray())  # 确保加载的数据与原始数据一致
    assert_array_almost_equal(y, ybz)  # 确保加载的标签与原始数据一致


def test_load_invalid_file():
    # 加载无效文件应当引发 ValueError
    with pytest.raises(ValueError):
        _load_svmlight_local_test_file(invalidfile)


def test_load_invalid_order_file():
    # 加载顺序无效的文件应当引发 ValueError
    with pytest.raises(ValueError):
        _load_svmlight_local_test_file(invalidfile2)


def test_load_zero_based():
    # 使用非零基索引加载文件应当引发 ValueError
    f = BytesIO(b"-1 4:1.\n1 0:1\n")
    with pytest.raises(ValueError):
        load_svmlight_file(f, zero_based=False)


def test_load_zero_based_auto():
    data1 = b"-1 1:1 2:2 3:3\n"
    data2 = b"-1 0:0 1:1\n"

    # 自动判断零基索引加载文件
    f1 = BytesIO(data1)
    X, y = load_svmlight_file(f1, zero_based="auto")
    assert X.shape == (1, 3)  # 确保加载的数据形状正确

    f1 = BytesIO(data1)
    f2 = BytesIO(data2)
    X1, y1, X2, y2 = load_svmlight_files([f1, f2], zero_based="auto")
    assert X1.shape == (1, 4)  # 确保加载的第一个数据集形状正确
    assert X2.shape == (1, 4)  # 确保加载的第二个数据集形状正确


def test_load_with_qid():
    # 加载带有 qid 属性的 svm 文件
    data = b"""
    3 qid:1 1:0.53 2:0.12
    2 qid:1 1:0.13 2:0.1
    7 qid:2 1:0.87 2:0.12"""
    X, y = load_svmlight_file(BytesIO(data), query_id=False)
    assert_array_equal(y, [3, 2, 7])  # 确保加载的标签正确
    assert_array_equal(X.toarray(), [[0.53, 0.12], [0.13, 0.1], [0.87, 0.12]])  # 确保加载的数据正确
    res1 = load_svmlight_files([BytesIO(data)], query_id=True)
    res2 = load_svmlight_file(BytesIO(data), query_id=True)
    # 遍历元组 (res1, res2)，每次迭代将元组中的元素分别赋值给 X, y, qid
    for X, y, qid in (res1, res2):
        # 断言 y 数组与 [3, 2, 7] 数组相等
        assert_array_equal(y, [3, 2, 7])
        # 断言 qid 数组与 [1, 1, 2] 数组相等
        assert_array_equal(qid, [1, 1, 2])
        # 断言 X 转换为稀疏矩阵后的数组与 [[0.53, 0.12], [0.13, 0.1], [0.87, 0.12]] 数组相等
        assert_array_equal(X.toarray(), [[0.53, 0.12], [0.13, 0.1], [0.87, 0.12]])
# 使用 pytest 的标记来跳过该测试，因为测试超出 32 位稀疏索引的溢出需要大量内存
@pytest.mark.skip(
    "testing the overflow of 32 bit sparse indexing requires a large amount of memory"
)
def test_load_large_qid():
    """
    读取带有 qid 属性的大型 libsvm / svmlight 文件。测试64位查询ID。
    """
    # 创建一个包含大量数据的字节串，每行数据格式为 "3 qid:{i} 1:0.53 2:0.12\n"，共计 40,000,000 行
    data = b"\n".join(
        (
            "3 qid:{0} 1:0.53 2:0.12\n2 qid:{0} 1:0.13 2:0.1".format(i).encode()
            for i in range(1, 40 * 1000 * 1000)
        )
    )
    # 调用 load_svmlight_file 函数加载数据，传入字节流和 query_id=True 参数
    X, y, qid = load_svmlight_file(BytesIO(data), query_id=True)
    # 断言最后四个元素 y[-4:] 应为 [3, 2, 3, 2]
    assert_array_equal(y[-4:], [3, 2, 3, 2])
    # 断言 qid 数组的唯一值应为从 1 到 40,000,000 的连续整数
    assert_array_equal(np.unique(qid), np.arange(1, 40 * 1000 * 1000))


def test_load_invalid_file2():
    # 使用 pytest.raises 检测 ValueError 异常
    with pytest.raises(ValueError):
        # 获取测试数据文件路径和无效文件路径
        data_path = _svmlight_local_test_file_path(datafile)
        invalid_path = _svmlight_local_test_file_path(invalidfile)
        # 调用 load_svmlight_files 函数，传入数据路径、无效路径和再次数据路径
        load_svmlight_files([str(data_path), str(invalid_path), str(data_path)])


def test_not_a_filename():
    # 使用 pytest.raises 检测 TypeError 异常
    with pytest.raises(TypeError):
        # 调用 load_svmlight_file 函数，传入一个浮点数参数 0.42
        load_svmlight_file(0.42)


def test_invalid_filename():
    # 使用 pytest.raises 检测 OSError 异常
    with pytest.raises(OSError):
        # 调用 load_svmlight_file 函数，传入一个无效的文件路径字符串
        load_svmlight_file("trou pic nic douille")


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_dump(csr_container):
    # 加载本地测试文件数据到稀疏矩阵 X_sparse 和密集向量 y_dense
    X_sparse, y_dense = _load_svmlight_local_test_file(datafile)
    # 将稀疏矩阵 X_sparse 转换为稠密数组 X_dense
    X_dense = X_sparse.toarray()
    # 使用 csr_container 将 y_dense 转换为稀疏容器 y_sparse
    y_sparse = csr_container(np.atleast_2d(y_dense))

    # 对 csr_matrix 进行切片可能会导致其 .indices 无序，因此测试确保正确排序
    X_sliced = X_sparse[np.arange(X_sparse.shape[0])]
    y_sliced = y_sparse[np.arange(y_sparse.shape[0])]
    # 迭代三种不同的输入数据类型：稀疏矩阵 (X_sparse), 密集矩阵 (X_dense), 切片矩阵 (X_sliced)
    for X in (X_sparse, X_dense, X_sliced):
        # 迭代三种不同的标签数据类型：稀疏矩阵 (y_sparse), 密集矩阵 (y_dense), 切片矩阵 (y_sliced)
        for y in (y_sparse, y_dense, y_sliced):
            # 迭代两种不同的 zero_based 参数值：True 和 False
            for zero_based in (True, False):
                # 迭代四种不同的数据类型：np.float32, np.float64, np.int32, np.int64
                for dtype in [np.float32, np.float64, np.int32, np.int64]:
                    # 创建一个字节流对象 f
                    f = BytesIO()
                    
                    # 对于 LibSVM 的兼容性，需要传递一个注释来获取版本信息；
                    # LibSVM 不支持注释，所以默认情况下不会添加注释。
                    
                    if sp.issparse(y) and y.shape[0] == 1:
                        # 当 y 是稀疏矩阵且其形状为 (1, n_labels) 时，
                        # 确保将 y 转置为 (n_samples, n_labels)
                        y = y.T
                    
                    # 注意：当 dtype=np.int32 时，我们进行了不安全的类型转换，
                    # 即 X.astype(dtype) 可能会溢出。结果因平台而异，X_dense.astype(dtype)
                    # 可能与 X_sparse.astype(dtype).asarray() 不同。
                    X_input = X.astype(dtype)
                    
                    # 将 X_input 和 y 以 SVMLight 格式写入到字节流 f 中，附带注释 "test"
                    dump_svmlight_file(
                        X_input, y, f, comment="test", zero_based=zero_based
                    )
                    
                    # 将文件指针移到字节流的开头
                    f.seek(0)
                    
                    # 从字节流中读取第一行作为注释
                    comment = f.readline()
                    comment = str(comment, "utf-8")  # 将字节转换为字符串
                    
                    # 断言 SVMLight 文件的注释中包含 "scikit-learn 版本号"
                    assert "scikit-learn %s" % sklearn.__version__ in comment
                    
                    # 继续读取下一行作为注释
                    comment = f.readline()
                    comment = str(comment, "utf-8")  # 将字节转换为字符串
                    
                    # 断言 SVMLight 文件的注释中包含 "zero-based" 或 "one-based"
                    assert ["one", "zero"][zero_based] + "-based" in comment
                    
                    # 从 SVMLight 格式的字节流中加载数据 X2 和标签 y2
                    X2, y2 = load_svmlight_file(f, dtype=dtype, zero_based=zero_based)
                    
                    # 断言加载后的 X2 数据类型为指定的 dtype
                    assert X2.dtype == dtype
                    
                    # 断言 X2 的排序索引与原始数据 X2.indices 的数组相等
                    assert_array_equal(X2.sorted_indices().indices, X2.indices)
                    
                    # 将稀疏矩阵 X2 转换为密集矩阵 X2_dense
                    X2_dense = X2.toarray()
                    
                    # 根据输入的数据类型，将 X_input 转换为密集矩阵 X_input_dense
                    if sp.issparse(X_input):
                        X_input_dense = X_input.toarray()
                    else:
                        X_input_dense = X_input
                    
                    # 如果数据类型为 np.float32，则允许在最后一位小数处存在舍入误差
                    if dtype == np.float32:
                        assert_array_almost_equal(X_input_dense, X2_dense, 4)
                        assert_array_almost_equal(
                            y_dense.astype(dtype, copy=False), y2, 4
                        )
                    else:
                        # 允许在最后一位小数处存在舍入误差
                        assert_array_almost_equal(X_input_dense, X2_dense, 15)
                        assert_array_almost_equal(
                            y_dense.astype(dtype, copy=False), y2, 15
                        )
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用 pytest 的 parametrize 标记来参数化测试函数，传入不同的 csr_container 参数
def test_dump_multilabel(csr_container):
    X = [[1, 0, 3, 0, 5], [0, 0, 0, 0, 0], [0, 5, 0, 1, 0]]
    y_dense = [[0, 1, 0], [1, 0, 1], [1, 1, 0]]
    # 根据 csr_container 创建稀疏表示的 y_sparse
    y_sparse = csr_container(y_dense)
    # 遍历两种格式的 y：y_dense 和 y_sparse
    for y in [y_dense, y_sparse]:
        f = BytesIO()
        # 将数据 X 和 y 以多标签格式写入字节流 f
        dump_svmlight_file(X, y, f, multilabel=True)
        f.seek(0)
        # 确保正确地以多标签格式写入
        assert f.readline() == b"1 0:1 2:3 4:5\n"
        assert f.readline() == b"0,2 \n"
        assert f.readline() == b"0,1 1:5 3:1\n"


def test_dump_concise():
    one = 1
    two = 2.1
    three = 3.01
    exact = 1.000000000000001
    # loses the last decimal place
    almost = 1.0000000000000001
    X = [
        [one, two, three, exact, almost],
        [1e9, 2e18, 3e27, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    y = [one, two, three, exact, almost]
    f = BytesIO()
    # 将数据 X 和 y 以最简洁的格式写入字节流 f
    dump_svmlight_file(X, y, f)
    f.seek(0)
    # 确保以最简洁的格式写入
    assert f.readline() == b"1 0:1 1:2.1 2:3.01 3:1.000000000000001 4:1\n"
    assert f.readline() == b"2.1 0:1000000000 1:2e+18 2:3e+27\n"
    assert f.readline() == b"3.01 \n"
    assert f.readline() == b"1.000000000000001 \n"
    assert f.readline() == b"1 \n"
    f.seek(0)
    # 确保也正确加载
    X2, y2 = load_svmlight_file(f)
    assert_array_almost_equal(X, X2.toarray())
    assert_array_almost_equal(y, y2)


def test_dump_comment():
    X, y = _load_svmlight_local_test_file(datafile)
    X = X.toarray()

    f = BytesIO()
    ascii_comment = "This is a comment\nspanning multiple lines."
    # 写入带有 ASCII 注释的数据 X 和 y 到字节流 f
    dump_svmlight_file(X, y, f, comment=ascii_comment, zero_based=False)
    f.seek(0)

    X2, y2 = load_svmlight_file(f, zero_based=False)
    assert_array_almost_equal(X, X2.toarray())
    assert_array_almost_equal(y, y2)

    # XXX we have to update this to support Python 3.x
    utf8_comment = b"It is true that\n\xc2\xbd\xc2\xb2 = \xc2\xbc"
    f = BytesIO()
    # 使用 UTF-8 编码的注释会引发 UnicodeDecodeError
    with pytest.raises(UnicodeDecodeError):
        dump_svmlight_file(X, y, f, comment=utf8_comment)

    unicode_comment = utf8_comment.decode("utf-8")
    f = BytesIO()
    # 写入带有 Unicode 注释的数据 X 和 y 到字节流 f
    dump_svmlight_file(X, y, f, comment=unicode_comment, zero_based=False)
    f.seek(0)

    X2, y2 = load_svmlight_file(f, zero_based=False)
    assert_array_almost_equal(X, X2.toarray())
    assert_array_almost_equal(y, y2)

    f = BytesIO()
    # 使用包含空字符的注释会引发 ValueError
    with pytest.raises(ValueError):
        dump_svmlight_file(X, y, f, comment="I've got a \0.")


def test_dump_invalid():
    X, y = _load_svmlight_local_test_file(datafile)

    f = BytesIO()
    y2d = [y]
    # 尝试写入一个维度不匹配的 y 会引发 ValueError
    with pytest.raises(ValueError):
        dump_svmlight_file(X, y2d, f)

    f = BytesIO()
    # 尝试写入一个长度不匹配的 y 会引发 ValueError
    with pytest.raises(ValueError):
        dump_svmlight_file(X, y[:-1], f)


def test_dump_query_id():
    # test dumping a file with query_id
    X, y = _load_svmlight_local_test_file(datafile)
    X = X.toarray()
    # 生成一个与样本数相同的查询ID数组，每两个样本共享一个查询ID
    query_id = np.arange(X.shape[0]) // 2
    
    # 创建一个字节流对象f，用于保存SVMLight格式的数据
    f = BytesIO()
    
    # 将数据集X和标签y以SVMLight格式写入字节流f，同时使用query_id作为查询ID，索引从零开始
    dump_svmlight_file(X, y, f, query_id=query_id, zero_based=True)
    
    # 将字节流f的指针移到起始位置
    f.seek(0)
    
    # 从字节流f中加载SVMLight格式的数据集和标签X1、y1，并返回相应的查询ID query_id1
    X1, y1, query_id1 = load_svmlight_file(f, query_id=True, zero_based=True)
    
    # 检查加载的数据集X1与原始数据集X（转为稀疏数组表示）的值是否几乎相等
    assert_array_almost_equal(X, X1.toarray())
    
    # 检查加载的标签y1与原始标签y的值是否几乎相等
    assert_array_almost_equal(y, y1)
    
    # 检查加载的查询ID query_id1 与原始的查询ID query_id 的值是否几乎相等
    assert_array_almost_equal(query_id, query_id1)
def test_load_with_long_qid():
    # load svmfile with longint qid attribute
    # 定义测试函数，用于加载带有长整型 qid 属性的 SVM 文件

    data = b"""
    1 qid:0 0:1 1:2 2:3
    0 qid:72048431380967004 0:1440446648 1:72048431380967004 2:236784985
    0 qid:-9223372036854775807 0:1440446648 1:72048431380967004 2:236784985
    3 qid:9223372036854775807  0:1440446648 1:72048431380967004 2:236784985"""
    # 定义测试数据，包括多行数据和相应的 qid 属性

    X, y, qid = load_svmlight_file(BytesIO(data), query_id=True)
    # 使用 load_svmlight_file 函数加载数据，包括特征矩阵 X、标签 y 和 query id qid

    true_X = [
        [1, 2, 3],
        [1440446648, 72048431380967004, 236784985],
        [1440446648, 72048431380967004, 236784985],
        [1440446648, 72048431380967004, 236784985],
    ]
    # 真实的特征矩阵 true_X，与加载的数据对应

    true_y = [1, 0, 0, 3]
    # 真实的标签 true_y

    trueQID = [0, 72048431380967004, -9223372036854775807, 9223372036854775807]
    # 真实的 query id trueQID

    assert_array_equal(y, true_y)
    assert_array_equal(X.toarray(), true_X)
    assert_array_equal(qid, trueQID)
    # 断言加载的数据与真实数据一致

    f = BytesIO()
    # 创建一个字节流对象 f

    dump_svmlight_file(X, y, f, query_id=qid, zero_based=True)
    # 将特征矩阵 X、标签 y 和 query id qid 写入到字节流 f 中，使用 zero_based=True

    f.seek(0)
    # 将字节流的读写位置移动到起始位置

    X, y, qid = load_svmlight_file(f, query_id=True, zero_based=True)
    # 从字节流 f 中加载特征矩阵 X、标签 y 和 query id qid，使用 zero_based=True

    assert_array_equal(y, true_y)
    assert_array_equal(X.toarray(), true_X)
    assert_array_equal(qid, trueQID)
    # 断言加载的数据与真实数据一致

    f.seek(0)
    # 将字节流的读写位置移动到起始位置

    X, y = load_svmlight_file(f, query_id=False, zero_based=True)
    # 从字节流 f 中加载特征矩阵 X、标签 y，不加载 query id，使用 zero_based=True

    assert_array_equal(y, true_y)
    assert_array_equal(X.toarray(), true_X)
    # 断言加载的数据与真实数据一致


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_load_zeros(csr_container):
    # 测试加载稀疏矩阵中的零值情况，使用不同的 CSR 容器类型

    f = BytesIO()
    # 创建一个字节流对象 f

    true_X = csr_container(np.zeros(shape=(3, 4)))
    # 生成一个稀疏矩阵的真实数据 true_X，使用给定的 CSR 容器类型

    true_y = np.array([0, 1, 0])
    # 真实的标签 true_y

    dump_svmlight_file(true_X, true_y, f)
    # 将稀疏矩阵 true_X 和标签 true_y 写入到字节流 f 中

    for zero_based in ["auto", True, False]:
        # 遍历不同的 zero_based 参数值

        f.seek(0)
        # 将字节流的读写位置移动到起始位置

        X, y = load_svmlight_file(f, n_features=4, zero_based=zero_based)
        # 从字节流 f 中加载特征矩阵 X 和标签 y，指定特征数量和 zero_based 参数

        assert_array_almost_equal(y, true_y)
        assert_array_almost_equal(X.toarray(), true_X.toarray())
        # 断言加载的数据与真实数据一致


@pytest.mark.parametrize("sparsity", [0, 0.1, 0.5, 0.99, 1])
@pytest.mark.parametrize("n_samples", [13, 101])
@pytest.mark.parametrize("n_features", [2, 7, 41])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_load_with_offsets(sparsity, n_samples, n_features, csr_container):
    # 测试在指定偏移和长度的情况下加载数据

    rng = np.random.RandomState(0)
    # 创建随机数生成器 rng

    X = rng.uniform(low=0.0, high=1.0, size=(n_samples, n_features))
    # 生成随机的特征矩阵 X

    if sparsity:
        X[X < sparsity] = 0.0
    # 如果指定了 sparsity，将小于 sparsity 的元素置为 0

    X = csr_container(X)
    # 使用给定的 CSR 容器类型包装特征矩阵 X

    y = rng.randint(low=0, high=2, size=n_samples)
    # 随机生成标签 y

    f = BytesIO()
    # 创建一个字节流对象 f

    dump_svmlight_file(X, y, f)
    # 将特征矩阵 X 和标签 y 写入到字节流 f 中

    f.seek(0)
    # 将字节流的读写位置移动到起始位置

    size = len(f.getvalue())
    # 获取字节流的大小

    # put some marks that are likely to happen anywhere in a row
    mark_0 = 0
    mark_1 = size // 3
    length_0 = mark_1 - mark_0
    mark_2 = 4 * size // 5
    length_1 = mark_2 - mark_1
    # 设置三个偏移和长度的标记点

    X_0, y_0 = load_svmlight_file(
        f, n_features=n_features, offset=mark_0, length=length_0
    )
    # 从字节流 f 中加载特征矩阵 X_0 和标签 y_0，指定特征数量、偏移和长度

    X_1, y_1 = load_svmlight_file(
        f, n_features=n_features, offset=mark_1, length=length_1
    )
    # 从字节流 f 中加载特征矩阵 X_1 和标签 y_1，指定特征数量、偏移和长度

    X_2, y_2 = load_svmlight_file(f, n_features=n_features, offset=mark_2)
    # 从字节流 f 中加载特征矩阵 X_2 和标签 y_2，指定特征数量和偏移

    y_concat = np.concatenate([y_0, y_1, y_2])
    # 将加载的标签拼接起来
    # 将稀疏矩阵 X_0、X_1 和 X_2 沿垂直方向堆叠，形成一个新的稀疏矩阵 X_concat
    X_concat = sp.vstack([X_0, X_1, X_2])
    
    # 断言数组 y 与数组 y_concat 几乎完全相等，若不相等则引发 AssertionError
    assert_array_almost_equal(y, y_concat)
    
    # 断言稀疏矩阵 X 转换为稀疏数组后与 X_concat 转换为稀疏数组后几乎完全相等，若不相等则引发 AssertionError
    assert_array_almost_equal(X.toarray(), X_concat.toarray())
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用pytest的@parametrize装饰器，以csr_container作为参数化测试的参数
def test_load_offset_exhaustive_splits(csr_container):
    # 初始化一个随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个二维数组X，表示特征数据
    X = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 0, 6],
            [1, 2, 3, 4, 0, 6],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
        ]
    )
    # 将二维数组X转换为CSR格式
    X = csr_container(X)
    # 获取样本数和特征数
    n_samples, n_features = X.shape
    # 生成随机分类标签y
    y = rng.randint(low=0, high=2, size=n_samples)
    # 创建查询ID，每两个样本一组
    query_id = np.arange(n_samples) // 2

    # 创建一个字节流对象f
    f = BytesIO()
    # 将X, y, query_id保存为SVMLight格式到字节流f中
    dump_svmlight_file(X, y, f, query_id=query_id)
    # 重置字节流f的位置到开头
    f.seek(0)

    # 获取字节流f的长度
    size = len(f.getvalue())

    # 在所有可能的字节偏移量下，分两部分加载相同的数据
    # 用于测试特定的边界情况
    for mark in range(size):
        # 重置字节流f的位置到开头
        f.seek(0)
        # 加载SVMLight格式文件的第一部分数据
        X_0, y_0, q_0 = load_svmlight_file(
            f, n_features=n_features, query_id=True, offset=0, length=mark
        )
        # 加载SVMLight格式文件的第二部分数据
        X_1, y_1, q_1 = load_svmlight_file(
            f, n_features=n_features, query_id=True, offset=mark, length=-1
        )
        # 合并查询ID
        q_concat = np.concatenate([q_0, q_1])
        # 合并分类标签y
        y_concat = np.concatenate([y_0, y_1])
        # 垂直堆叠特征矩阵X
        X_concat = sp.vstack([X_0, X_1])
        # 断言分类标签y与原始数据y的近似相等
        assert_array_almost_equal(y, y_concat)
        # 断言查询ID与原始数据query_id完全相等
        assert_array_equal(query_id, q_concat)
        # 断言特征矩阵X与原始数据X的近似相等（考虑稀疏矩阵情况）
        assert_array_almost_equal(X.toarray(), X_concat.toarray())


def test_load_with_offsets_error():
    # 使用pytest.raises断言，验证当缺少n_features参数时会抛出ValueError异常
    with pytest.raises(ValueError, match="n_features is required"):
        _load_svmlight_local_test_file(datafile, offset=3, length=3)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用pytest的@parametrize装饰器，以csr_container作为参数化测试的参数
def test_multilabel_y_explicit_zeros(tmp_path, csr_container):
    """
    Ensure that if y contains explicit zeros (i.e. elements of y.data equal to
    0) then those explicit zeros are not encoded.
    """
    # 设置保存路径
    save_path = str(tmp_path / "svm_explicit_zero")
    # 初始化随机数生成器
    rng = np.random.RandomState(42)
    # 创建特征矩阵X
    X = rng.randn(3, 5).astype(np.float64)
    # 创建稀疏矩阵的指针数组
    indptr = np.array([0, 2, 3, 6])
    # 创建稀疏矩阵的列索引数组
    indices = np.array([0, 2, 2, 0, 1, 2])
    # 创建稀疏矩阵的数据数组，包含显式的零值
    data = np.array([0, 1, 1, 1, 1, 0])
    # 使用csr_container创建稀疏矩阵y
    y = csr_container((data, indices, indptr), shape=(3, 3))
    # 将y作为稀疏矩阵保存为SVMLight格式文件
    dump_svmlight_file(X, y, save_path, multilabel=True)

    # 从保存的SVMLight格式文件中加载数据
    _, y_load = load_svmlight_file(save_path, multilabel=True)
    # 预期的y_true值
    y_true = [(2.0,), (2.0,), (0.0, 1.0)]
    # 断言加载的y与预期的y_true完全相等
    assert y_load == y_true


def test_dump_read_only(tmp_path):
    """Ensure that there is no ValueError when dumping a read-only `X`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28026
    """
    # 初始化随机数生成器
    rng = np.random.RandomState(42)
    # 创建随机特征矩阵X和标签y
    X = rng.randn(5, 2)
    y = rng.randn(5)

    # 将X和y转换为基于内存映射的只读格式
    X, y = create_memmap_backed_data([X, y])

    # 设置保存路径
    save_path = str(tmp_path / "svm_read_only")
    # 将数据保存为SVMLight格式文件
    dump_svmlight_file(X, y, save_path)
```