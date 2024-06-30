# `D:\src\scipysrc\scipy\scipy\io\matlab\tests\test_pathological.py`

```
"""
Test reading of files not conforming to matlab specification

We try and read any file that matlab reads, these files included
"""
# 导入必要的模块和函数
from os.path import dirname, join as pjoin
# 从 numpy.testing 中导入 assert_ 函数，用于断言测试结果
from numpy.testing import assert_
# 从 pytest 中导入 raises 函数，用于断言特定异常被抛出
from pytest import raises as assert_raises
# 从 scipy.io.matlab._mio 中导入 loadmat 函数，用于加载 MATLAB 文件
from scipy.io.matlab._mio import loadmat

# 定义测试数据的路径
TEST_DATA_PATH = pjoin(dirname(__file__), 'data')


def test_multiple_fieldnames():
    # Example provided by Dharhas Pothina
    # Extracted using mio5.varmats_from_mat
    # 拼接文件路径，加载 MATLAB 文件
    multi_fname = pjoin(TEST_DATA_PATH, 'nasty_duplicate_fieldnames.mat')
    vars = loadmat(multi_fname)
    # 获取加载的变量中的字段名集合
    funny_names = vars['Summary'].dtype.names
    # 断言特定字段名集合在加载的变量中
    assert_({'_1_Station_Q', '_2_Station_Q',
             '_3_Station_Q'}.issubset(funny_names))


def test_malformed1():
    # Example from gh-6072
    # Contains malformed header data, which previously resulted into a
    # buffer overflow.
    #
    # Should raise an exception, not segfault
    # 加载包含错误头数据的 MATLAB 文件，应该抛出 ValueError 异常而不是导致段错误
    fname = pjoin(TEST_DATA_PATH, 'malformed1.mat')
    with open(fname, 'rb') as f:
        assert_raises(ValueError, loadmat, f)
```