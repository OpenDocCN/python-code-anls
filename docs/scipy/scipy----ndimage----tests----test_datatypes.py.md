# `D:\src\scipysrc\scipy\scipy\ndimage\tests\test_datatypes.py`

```
""" Testing data types for ndimage calls
"""
# 导入所需的库和模块
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
import pytest

# 导入 scipy 库中的 ndimage 模块
from scipy import ndimage


# 定义测试函数 test_map_coordinates_dts
def test_map_coordinates_dts():
    # 检查 ndimage 对不同数据类型的插值是否正常工作
    data = np.array([[4, 1, 3, 2],
                     [7, 6, 8, 5],
                     [3, 5, 3, 6]])
    shifted_data = np.array([[0, 0, 0, 0],
                             [0, 4, 1, 3],
                             [0, 7, 6, 8]])
    idx = np.indices(data.shape)
    # 定义数据类型的集合
    dts = (np.uint8, np.uint16, np.uint32, np.uint64,
           np.int8, np.int16, np.int32, np.int64,
           np.intp, np.uintp, np.float32, np.float64)
    # 循环遍历插值的次序
    for order in range(0, 6):
        # 遍历不同的数据类型
        for data_dt in dts:
            these_data = data.astype(data_dt)
            # 遍历坐标数据类型
            for coord_dt in dts:
                # 进行仿射映射
                mat = np.eye(2, dtype=coord_dt)
                off = np.zeros((2,), dtype=coord_dt)
                out = ndimage.affine_transform(these_data, mat, off)
                assert_array_almost_equal(these_data, out)
                # 映射坐标
                coords_m1 = idx.astype(coord_dt) - 1
                coords_p10 = idx.astype(coord_dt) + 10
                out = ndimage.map_coordinates(these_data, coords_m1, order=order)
                assert_array_almost_equal(out, shifted_data)
                # 检查常数填充是否正常工作
                out = ndimage.map_coordinates(these_data, coords_p10, order=order)
                assert_array_almost_equal(out, np.zeros((3,4)))
            # 检查平移和缩放是否正常工作
            out = ndimage.shift(these_data, 1)
            assert_array_almost_equal(out, shifted_data)
            out = ndimage.zoom(these_data, 1)
            assert_array_almost_equal(these_data, out)


# 标记该测试为预期失败，原因是在多个平台上存在问题
@pytest.mark.xfail(True, reason="Broken on many platforms")
def test_uint64_max():
    # 测试插值是否尊重 uint64 的最大值。已知在某些平台上会失败，例如 win32 和 Debian on s390x
    # 插值总是以双精度浮点数进行，因此我们使用最大的 uint64 值
    big = 2**64 - 1025
    arr = np.array([big, big, big], dtype=np.uint64)
    # 测试几何变换（map_coordinates, affine_transform）
    inds = np.indices(arr.shape) - 0.1
    x = ndimage.map_coordinates(arr, inds)
    assert_(x[1] == int(float(big)))
    assert_(x[2] == int(float(big)))
    # 测试缩放和平移
    x = ndimage.shift(arr, 0.1)
    assert_(x[1] == int(float(big)))
    assert_(x[2] == int(float(big)))
```