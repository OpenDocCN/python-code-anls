# `D:\src\scipysrc\scipy\scipy\signal\tests\test_peak_finding.py`

```
import copy  # 导入 copy 模块，用于复制对象

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.testing import (  # 从 NumPy 的 testing 子模块导入多个函数
    assert_,  # 导入 assert_ 函数，用于断言测试
    assert_equal,  # 导入 assert_equal 函数，用于断言相等
    assert_allclose,  # 导入 assert_allclose 函数，用于断言数组近似相等
    assert_array_equal  # 导入 assert_array_equal 函数，用于断言数组相等
)
import pytest  # 导入 PyTest 库
from pytest import raises, warns  # 从 PyTest 模块导入 raises 和 warns 函数

from scipy.signal._peak_finding import (  # 从 SciPy 的 signal 子模块导入多个函数
    argrelmax,  # 导入 argrelmax 函数，用于寻找相对最大值的索引
    argrelmin,  # 导入 argrelmin 函数，用于寻找相对最小值的索引
    peak_prominences,  # 导入 peak_prominences 函数，用于计算峰值突出度
    peak_widths,  # 导入 peak_widths 函数，用于计算峰值宽度
    _unpack_condition_args,  # 导入 _unpack_condition_args 函数，用于解包条件参数
    find_peaks,  # 导入 find_peaks 函数，用于寻找峰值
    find_peaks_cwt,  # 导入 find_peaks_cwt 函数，用于连续小波变换寻找峰值
    _identify_ridge_lines  # 导入 _identify_ridge_lines 函数，用于识别岭线
)
from scipy.signal.windows import gaussian  # 从 SciPy 的 signal 子模块导入 gaussian 函数，用于生成高斯窗口
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning  # 从 SciPy 的 signal 子模块导入 _local_maxima_1d 函数和 PeakPropertyWarning 类


def _gen_gaussians(center_locs, sigmas, total_length):
    """
    Generate a signal consisting of multiple Gaussian peaks.

    Args:
    - center_locs: Array of center locations of Gaussians.
    - sigmas: Array of standard deviations (sigma) of Gaussians.
    - total_length: Total length of the output signal.

    Returns:
    - out_data: Signal consisting of Gaussian peaks.
    """
    xdata = np.arange(0, total_length).astype(float)  # 生成从 0 到 total_length 的浮点数数组
    out_data = np.zeros(total_length, dtype=float)  # 创建长度为 total_length 的全零浮点数数组
    for ind, sigma in enumerate(sigmas):  # 遍历 sigmas 数组的索引和值
        tmp = (xdata - center_locs[ind]) / sigma  # 计算高斯函数的指数部分
        out_data += np.exp(-(tmp**2))  # 计算高斯函数并累加到输出数据
    return out_data  # 返回生成的高斯峰信号


def _gen_gaussians_even(sigmas, total_length):
    """
    Generate a signal with evenly spaced Gaussian peaks.

    Args:
    - sigmas: Array of standard deviations (sigma) of Gaussians.
    - total_length: Total length of the output signal.

    Returns:
    - out_data: Signal consisting of Gaussian peaks.
    - center_locs: Array of center locations of Gaussians.
    """
    num_peaks = len(sigmas)  # 获取高斯峰数量
    delta = total_length / (num_peaks + 1)  # 计算峰之间的间隔
    center_locs = np.linspace(delta, total_length - delta, num=num_peaks).astype(int)  # 生成峰的中心位置数组
    out_data = _gen_gaussians(center_locs, sigmas, total_length)  # 生成高斯峰信号
    return out_data, center_locs  # 返回生成的信号和峰的中心位置数组


def _gen_ridge_line(start_locs, max_locs, length, distances, gaps):
    """
    Generate coordinates for a ridge line.

    Args:
    - start_locs: Starting coordinates (length 2).
    - max_locs: Maximum coordinates of the intended matrix.
    - length: Length of the ridge line.
    - distances: List of distances between adjacent columns.
    - gaps: List of gaps between adjacent rows.

    Returns:
    - List containing two arrays: x-coordinates and y-coordinates of the ridge line.
    """
    def keep_bounds(num, max_val):
        """Ensure a number stays within specified bounds."""
        out = max(num, 0)  # 确保数值不小于 0
        out = min(out, max_val)  # 确保数值不大于 max_val
        return out

    gaps = copy.deepcopy(gaps)  # 深拷贝 gaps 列表
    distances = copy.deepcopy(distances)  # 深拷贝 distances 列表

    locs = np.zeros([length, 2], dtype=int)  # 创建长度为 length 的二维整数数组
    locs[0, :] = start_locs  # 将起始坐标存入 locs 数组的第一行
    total_length = max_locs[0] - start_locs[0] - sum(gaps)  # 计算行坐标的有效长度
    if total_length < length:
        raise ValueError('Cannot generate ridge line according to constraints')  # 若长度不足则抛出 ValueError 异常
    dist_int = length / len(distances) - 1  # 计算距离间隔
    gap_int = length / len(gaps) - 1  # 计算间隙间隔
    for ind in range(1, length):
        nextcol = locs[ind - 1, 1]  # 获取下一个列坐标
        nextrow = locs[ind - 1, 0] + 1  # 获取下一个行坐标
        if (ind % dist_int == 0) and (len(distances) > 0):  # 判断是否需要更新列坐标
            nextcol += ((-1)**ind)*distances.pop()  # 根据距离列表更新列坐标
        if (ind % gap_int == 0) and (len(gaps) > 0):  # 判断是否需要更新行坐标
            nextrow += gaps.pop()  # 根据间隙列表更新行坐标
        nextrow = keep_bounds(nextrow, max_locs[0])  # 确保行坐标不超出边界
        nextcol = keep_bounds(nextcol, max_locs[1])  # 确保列坐标不超出边界
        locs[ind, :] = [nextrow, nextcol]  # 存储更新后的坐标到 locs 数组

    return [locs[:, 0], locs[:, 1]]  # 返回生成的岭线的 x 和 y 坐标数组


class TestLocalMaxima1d:
    """
    Unit tests for _local_maxima_1d function.
    """

    def test_empty(self):
        """
        Test case for empty input signal.
        """
        x = np.array([], dtype=np.float64)  # 创建空的浮点数数组 x
        for array in _local_maxima_1d(x):  # 对空数组调用 _local_maxima_1d 函数
            assert_equal(array, np.array([]))  # 断言输出数组为空数组
            assert_(array.base is None)  # 断言输出数组没有基础对象
   `
    # 测试线性信号的情况
    def test_linear(self):
        """Test with linear signal."""
        # 生成从0到100的等间距数组作为输入信号
        x = np.linspace(0, 100)
        # 对信号调用_local_maxima_1d函数，期望返回空数组，并验证其基础数据是None
        for array in _local_maxima_1d(x):
            assert_equal(array, np.array([]))
            assert_(array.base is None)

    # 测试简单信号的情况
    def test_simple(self):
        """Test with simple signal."""
        # 生成从-10到10之间的等间距50个点的数组作为输入信号
        x = np.linspace(-10, 10, 50)
        # 将数组每隔3个元素的切片加1
        x[2::3] += 1
        # 期望的局部最大值位置数组
        expected = np.arange(2, 50, 3)
        # 对信号调用_local_maxima_1d函数，期望返回预期的局部最大值位置数组，并验证其基础数据是None
        for array in _local_maxima_1d(x):
            # 对于大小为1的平台，边缘和中点是相同的
            assert_equal(array, expected)
            assert_(array.base is None)

    # 测试是否正确检测到平坦最大值的情况
    def test_flat_maxima(self):
        """Test if flat maxima are detected correctly."""
        # 输入包含平坦最大值的数组
        x = np.array([-1.3, 0, 1, 0, 2, 2, 0, 3, 3, 3, 2.99, 4, 4, 4, 4, -10,
                      -5, -5, -5, -5, -5, -10])
        # 调用_local_maxima_1d函数，期望返回平坦最大值的中点、左边缘和右边缘数组
        midpoints, left_edges, right_edges = _local_maxima_1d(x)
        assert_equal(midpoints, np.array([2, 4, 8, 12, 18]))
        assert_equal(left_edges, np.array([2, 4, 7, 11, 16]))
        assert_equal(right_edges, np.array([2, 5, 9, 14, 20]))

    # 使用参数化测试检查信号边缘行为是否正确
    @pytest.mark.parametrize('x', [
        np.array([1., 0, 2]),
        np.array([3., 3, 0, 4, 4]),
        np.array([5., 5, 5, 0, 6, 6, 6]),
    ])
    def test_signal_edges(self, x):
        """Test if behavior on signal edges is correct."""
        # 对信号调用_local_maxima_1d函数，期望返回空数组，并验证其基础数据是None
        for array in _local_maxima_1d(x):
            assert_equal(array, np.array([]))
            assert_(array.base is None)

    # 测试异常情况：输入验证和引发的异常
    def test_exceptions(self):
        """Test input validation and raised exceptions."""
        # 期望输入为(1, 1)维度数组引发 ValueError 异常
        with raises(ValueError, match="wrong number of dimensions"):
            _local_maxima_1d(np.ones((1, 1)))
        # 期望输入为整数类型数组引发 ValueError 异常
        with raises(ValueError, match="expected 'const float64_t'"):
            _local_maxima_1d(np.ones(1, dtype=int))
        # 期望输入为列表引发 TypeError 异常
        with raises(TypeError, match="list"):
            _local_maxima_1d([1., 2.])
        # 期望输入为None引发 TypeError 异常
        with raises(TypeError, match="'x' must not be None"):
            _local_maxima_1d(None)
class TestRidgeLines:
    # 测试空矩阵情况
    def test_empty(self):
        # 创建一个大小为20x100的全零矩阵作为测试矩阵
        test_matr = np.zeros([20, 100])
        # 调用函数_identify_ridge_lines识别岭线，预期返回空列表
        lines = _identify_ridge_lines(test_matr, np.full(20, 2), 1)
        # 断言返回的岭线列表长度为0
        assert_(len(lines) == 0)

    # 测试最小情况
    def test_minimal(self):
        # 创建一个大小为20x100的全零矩阵作为测试矩阵
        test_matr = np.zeros([20, 100])
        # 在位置(0, 10)处设置一个值为1的点
        test_matr[0, 10] = 1
        # 调用函数_identify_ridge_lines识别岭线，预期返回包含一条岭线的列表
        lines = _identify_ridge_lines(test_matr, np.full(20, 2), 1)
        # 断言返回的岭线列表长度为1
        assert_(len(lines) == 1)

        # 继续测试，设置多个点为1
        test_matr = np.zeros([20, 100])
        test_matr[0:2, 10] = 1
        lines = _identify_ridge_lines(test_matr, np.full(20, 2), 1)
        assert_(len(lines) == 1)

    # 测试单一通过情况
    def test_single_pass(self):
        # 定义距离和间隙列表
        distances = [0, 1, 2, 5]
        gaps = [0, 1, 2, 0, 1]
        # 创建一个大小为20x50的接近全零矩阵
        test_matr = np.zeros([20, 50]) + 1e-12
        # 定义岭线的长度
        length = 12
        # 生成一条岭线，并在测试矩阵中标记出来
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        # 计算最大距离数组
        max_distances = np.full(20, max(distances))
        # 识别岭线，预期返回包含生成的岭线的列表
        identified_lines = _identify_ridge_lines(test_matr,
                                                 max_distances,
                                                 max(gaps) + 1)
        # 断言返回的岭线列表与生成的岭线相等
        assert_array_equal(identified_lines, [line])

    # 测试单一大距离情况
    def test_single_bigdist(self):
        # 定义距离和间隙列表
        distances = [0, 1, 2, 5]
        gaps = [0, 1, 2, 4]
        # 创建一个大小为20x50的全零矩阵
        test_matr = np.zeros([20, 50])
        # 定义岭线的长度
        length = 12
        # 生成一条岭线，并在测试矩阵中标记出来
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        # 定义最大距离和最大距离数组
        max_dist = 3
        max_distances = np.full(20, max_dist)
        # 识别岭线，预期返回两条岭线的列表
        identified_lines = _identify_ridge_lines(test_matr,
                                                 max_distances,
                                                 max(gaps) + 1)
        # 断言返回的岭线列表长度为2
        assert_(len(identified_lines) == 2)

        # 对每条识别的岭线进行进一步的断言：距离和间隙满足条件
        for iline in identified_lines:
            adists = np.diff(iline[1])
            np.testing.assert_array_less(np.abs(adists), max_dist)

            agaps = np.diff(iline[0])
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)

    # 测试单一大间隙情况
    def test_single_biggap(self):
        # 定义距离、最大间隙和间隙列表
        distances = [0, 1, 2, 5]
        max_gap = 3
        gaps = [0, 4, 2, 1]
        # 创建一个大小为20x50的全零矩阵
        test_matr = np.zeros([20, 50])
        # 定义岭线的长度
        length = 12
        # 生成一条岭线，并在测试矩阵中标记出来
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        # 定义最大距离和最大距离数组
        max_dist = 6
        max_distances = np.full(20, max_dist)
        # 识别岭线，预期返回两条岭线的列表
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max_gap)
        # 断言返回的岭线列表长度为2
        assert_(len(identified_lines) == 2)

        # 对每条识别的岭线进行进一步的断言：距离和间隙满足条件
        for iline in identified_lines:
            adists = np.diff(iline[1])
            np.testing.assert_array_less(np.abs(adists), max_dist)

            agaps = np.diff(iline[0])
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)
    # 定义单元测试方法，用于测试单条脊线检测函数
    def test_single_biggaps(self):
        # 初始化距离列表，包含一个元素0
        distances = [0]
        # 设置最大间隙值为1
        max_gap = 1
        # 设置间隙列表为[3, 6]
        gaps = [3, 6]
        # 创建一个 50x50 的全零矩阵作为测试矩阵
        test_matr = np.zeros([50, 50])
        # 设定脊线长度为30
        length = 30
        # 使用 _gen_ridge_line 函数生成一条脊线，起点为 [0, 25]
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        # 在测试矩阵中标记生成的脊线
        test_matr[line[0], line[1]] = 1
        # 设置最大距离为1的全50个元素的数组
        max_dist = 1
        max_distances = np.full(50, max_dist)
        # 调用 _identify_ridge_lines 函数识别脊线，期望返回3条脊线
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max_gap)
        # 使用断言检查识别到的脊线数量是否为3
        assert_(len(identified_lines) == 3)

        # 遍历识别到的每条脊线
        for iline in identified_lines:
            # 计算每条脊线中列索引之间的差异
            adists = np.diff(iline[1])
            # 使用 NumPy 测试断言，验证列索引差异的绝对值是否小于最大距离
            np.testing.assert_array_less(np.abs(adists), max_dist)

            # 计算每条脊线中行索引之间的差异
            agaps = np.diff(iline[0])
            # 使用 NumPy 测试断言，验证行索引差异的绝对值是否小于最大间隙加上0.1
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)
class TestArgrel:

    def test_empty(self):
        # Regression test for gh-2832.
        # When there are no relative extrema, make sure that
        # the number of empty arrays returned matches the
        # dimension of the input.

        empty_array = np.array([], dtype=int)  # 创建一个空的 NumPy 数组

        z1 = np.zeros(5)  # 创建一个包含 5 个零的 NumPy 数组

        i = argrelmin(z1)  # 找出 z1 中的相对最小值的索引
        assert_equal(len(i), 1)  # 确保返回的索引数组的长度为 1
        assert_array_equal(i[0], empty_array)  # 确保返回的索引数组与空数组相等

        z2 = np.zeros((3,5))  # 创建一个形状为 (3,5) 的全零 NumPy 数组

        row, col = argrelmin(z2, axis=0)  # 找出 z2 每列的相对最小值的行和列索引
        assert_array_equal(row, empty_array)  # 确保行索引数组为空数组
        assert_array_equal(col, empty_array)  # 确保列索引数组为空数组

        row, col = argrelmin(z2, axis=1)  # 找出 z2 每行的相对最小值的行和列索引
        assert_array_equal(row, empty_array)  # 确保行索引数组为空数组
        assert_array_equal(col, empty_array)  # 确保列索引数组为空数组

    def test_basic(self):
        # Note: the docstrings for the argrel{min,max,extrema} functions
        # do not give a guarantee of the order of the indices, so we'll
        # sort them before testing.

        x = np.array([[1, 2, 2, 3, 2],
                      [2, 1, 2, 2, 3],
                      [3, 2, 1, 2, 2],
                      [2, 3, 2, 1, 2],
                      [1, 2, 3, 2, 1]])

        row, col = argrelmax(x, axis=0)  # 找出 x 每列的相对最大值的行和列索引
        order = np.argsort(row)  # 对行索引进行排序，以便进行比较
        assert_equal(row[order], [1, 2, 3])  # 确保排序后的行索引与预期值匹配
        assert_equal(col[order], [4, 0, 1])  # 确保排序后的列索引与预期值匹配

        row, col = argrelmax(x, axis=1)  # 找出 x 每行的相对最大值的行和列索引
        order = np.argsort(row)  # 对行索引进行排序，以便进行比较
        assert_equal(row[order], [0, 3, 4])  # 确保排序后的行索引与预期值匹配
        assert_equal(col[order], [3, 1, 2])  # 确保排序后的列索引与预期值匹配

        row, col = argrelmin(x, axis=0)  # 找出 x 每列的相对最小值的行和列索引
        order = np.argsort(row)  # 对行索引进行排序，以便进行比较
        assert_equal(row[order], [1, 2, 3])  # 确保排序后的行索引与预期值匹配
        assert_equal(col[order], [1, 2, 3])  # 确保排序后的列索引与预期值匹配

        row, col = argrelmin(x, axis=1)  # 找出 x 每行的相对最小值的行和列索引
        order = np.argsort(row)  # 对行索引进行排序，以便进行比较
        assert_equal(row[order], [1, 2, 3])  # 确保排序后的行索引与预期值匹配
        assert_equal(col[order], [1, 2, 3])  # 确保排序后的列索引与预期值匹配

    def test_highorder(self):
        order = 2  # 设置阶数为 2
        sigmas = [1.0, 2.0, 10.0, 5.0, 15.0]  # 定义多个标准差值
        test_data, act_locs = _gen_gaussians_even(sigmas, 500)  # 生成均匀分布的高斯数据和其实际位置
        test_data[act_locs + order] = test_data[act_locs]*0.99999  # 在指定位置上修改数据
        test_data[act_locs - order] = test_data[act_locs]*0.99999  # 在指定位置上修改数据
        rel_max_locs = argrelmax(test_data, order=order, mode='clip')[0]  # 找出 test_data 的相对最大值的位置

        assert_(len(rel_max_locs) == len(act_locs))  # 确保找到的相对最大值的位置数量与预期相符
        assert_((rel_max_locs == act_locs).all())  # 确保找到的相对最大值的位置与预期位置完全一致

    def test_2d_gaussians(self):
        sigmas = [1.0, 2.0, 10.0]  # 定义多个标准差值
        test_data, act_locs = _gen_gaussians_even(sigmas, 100)  # 生成均匀分布的高斯数据和其实际位置
        rot_factor = 20  # 设置旋转因子
        rot_range = np.arange(0, len(test_data)) - rot_factor  # 计算旋转范围
        test_data_2 = np.vstack([test_data, test_data[rot_range]])  # 在垂直方向堆叠数据
        rel_max_rows, rel_max_cols = argrelmax(test_data_2, axis=1, order=1)  # 找出 test_data_2 每行的相对最大值的行和列索引

        for rw in range(0, test_data_2.shape[0]):
            inds = (rel_max_rows == rw)  # 找出属于当前行的索引

            assert_(len(rel_max_cols[inds]) == len(act_locs))  # 确保找到的相对最大值的列索引数量与预期相符
            assert_((act_locs == (rel_max_cols[inds] - rot_factor*rw)).all())  # 确保找到的相对最大值的列索引与预期位置完全一致
    def test_empty(self):
        """
        Test if an empty array is returned if no peaks are provided.
        """
        # 调用 peak_prominences 函数，传入信号和空的峰值列表，期望返回空数组
        out = peak_prominences([1, 2, 3], [])
        # 遍历输出结果和数据类型列表，确保每个数组的大小为0且数据类型正确
        for arr, dtype in zip(out, [np.float64, np.intp, np.intp]):
            assert_(arr.size == 0)
            assert_(arr.dtype == dtype)

        # 再次调用 peak_prominences 函数，传入空信号和空的峰值列表，期望返回空数组
        out = peak_prominences([], [])
        # 遍历输出结果和数据类型列表，确保每个数组的大小为0且数据类型正确
        for arr, dtype in zip(out, [np.float64, np.intp, np.intp]):
            assert_(arr.size == 0)
            assert_(arr.dtype == dtype)

    def test_basic(self):
        """
        Test if height of prominences is correctly calculated in signal with
        rising baseline (peak widths are 1 sample).
        """
        # 准备基础信号
        x = np.array([-1, 1.2, 1.2, 1, 3.2, 1.3, 2.88, 2.1])
        peaks = np.array([1, 2, 4, 6])
        lbases = np.array([0, 0, 0, 5])
        rbases = np.array([3, 3, 5, 7])
        # 计算峰的突出高度，根据公式 proms = x[peaks] - max(x[lbases], x[rbases])
        proms = x[peaks] - np.max([x[lbases], x[rbases]], axis=0)
        # 测试计算结果是否与手工计算结果匹配
        out = peak_prominences(x, peaks)
        assert_equal(out[0], proms)
        assert_equal(out[1], lbases)
        assert_equal(out[2], rbases)

    def test_edge_cases(self):
        """
        Test edge cases.
        """
        # 峰的高度、突出度和基底相同
        x = [0, 2, 1, 2, 1, 2, 0]
        peaks = [1, 3, 5]
        proms, lbases, rbases = peak_prominences(x, peaks)
        assert_equal(proms, [2, 2, 2])
        assert_equal(lbases, [0, 0, 0])
        assert_equal(rbases, [6, 6, 6])

        # 峰的高度和突出度相同，但基底不同
        x = [0, 1, 0, 1, 0, 1, 0]
        peaks = np.array([1, 3, 5])
        proms, lbases, rbases = peak_prominences(x, peaks)
        assert_equal(proms, [1, 1, 1])
        assert_equal(lbases, peaks - 1)
        assert_equal(rbases, peaks + 1)

    def test_non_contiguous(self):
        """
        Test with non-C-contiguous input arrays.
        """
        x = np.repeat([-9, 9, 9, 0, 3, 1], 2)
        peaks = np.repeat([1, 2, 4], 2)
        # 测试非连续的输入数组
        proms, lbases, rbases = peak_prominences(x[::2], peaks[::2])
        assert_equal(proms, [9, 9, 2])
        assert_equal(lbases, [0, 0, 3])
        assert_equal(rbases, [3, 3, 5])

    def test_wlen(self):
        """
        Test if wlen actually shrinks the evaluation range correctly.
        """
        x = [0, 1, 2, 3, 1, 0, -1]
        peak = [3]
        # 测试 wlen 是否正确缩小评估范围
        assert_equal(peak_prominences(x, peak), [3., 0, 6])
        for wlen, i in [(8, 0), (7, 0), (6, 0), (5, 1), (3.2, 1), (3, 2), (1.1, 2)]:
            assert_equal(peak_prominences(x, peak, wlen), [3. - i, 0 + i, 6 - i])
    # 测试异常情况的函数
    def test_exceptions(self):
        """
        Verify that exceptions and warnings are raised.
        """
        # 当 x 的维度大于 1 时，应该引发 ValueError 异常，匹配错误信息 '1-D array'
        with raises(ValueError, match='1-D array'):
            peak_prominences([[0, 1, 1, 0]], [1, 2])
        
        # 当 peaks 的维度大于 1 时，应该引发 ValueError 异常，匹配错误信息 '1-D array'
        with raises(ValueError, match='1-D array'):
            peak_prominences([0, 1, 1, 0], [[1, 2]])
        
        # 当 x 的维度小于 1 时，应该引发 ValueError 异常，匹配错误信息 '1-D array'
        with raises(ValueError, match='1-D array'):
            peak_prominences(3, [0,])
        
        # 当提供了空的 x 时，应该引发 ValueError 异常，匹配错误信息 'not a valid index'
        with raises(ValueError, match='not a valid index'):
            peak_prominences([], [0])
        
        # 当 peaks 中包含无效的索引时，对于非空 x，应该引发 ValueError 异常，匹配错误信息 'not a valid index'
        for p in [-100, -1, 3, 1000]:
            with raises(ValueError, match='not a valid index'):
                peak_prominences([1, 0, 2], [p])
        
        # 当 peaks 的类型无法安全地转换为 np.intp 时，应该引发 TypeError 异常，匹配错误信息 'cannot safely cast'
        with raises(TypeError, match='cannot safely cast'):
            peak_prominences([0, 1, 1, 0], [1.1, 2.3])
        
        # 当 wlen 小于 3 时，应该引发 ValueError 异常，匹配错误信息 'wlen'
        with raises(ValueError, match='wlen'):
            peak_prominences(np.arange(10), [3, 5], wlen=1)

    # 测试警告情况的函数
    def test_warnings(self):
        """
        Verify that appropriate warnings are raised.
        """
        # 检查当一些峰的突出值为 0 时，是否引发 PeakPropertyWarning 警告，匹配警告信息 'some peaks have a prominence of 0'
        for p in [0, 1, 2]:
            with warns(PeakPropertyWarning, match='some peaks have a prominence of 0'):
                peak_prominences([1, 0, 2], [p,])
        
        # 检查当一些峰的突出值为 0 时，是否引发 PeakPropertyWarning 警告，匹配警告信息 'some peaks have a prominence of 0'
        with warns(PeakPropertyWarning, match='some peaks have a prominence of 0'):
            peak_prominences([0, 1, 1, 1, 0], [2], wlen=2)
# 定义一个名为 TestPeakWidths 的测试类
class TestPeakWidths:

    # 测试当没有峰值时返回空数组的情况
    def test_empty(self):
        """
        Test if an empty array is returned if no peaks are provided.
        """
        # 调用 peak_widths 函数，传入空数组作为参数，返回宽度数组的第一个元素
        widths = peak_widths([], [])[0]
        # 断言 widths 是 numpy 数组对象
        assert_(isinstance(widths, np.ndarray))
        # 断言 widths 的大小为 0
        assert_equal(widths.size, 0)
        
        # 再次调用 peak_widths 函数，传入包含峰值的数组和空数组作为参数
        widths = peak_widths([1, 2, 3], [])[0]
        # 断言 widths 是 numpy 数组对象
        assert_(isinstance(widths, np.ndarray))
        # 断言 widths 的大小为 0
        assert_equal(widths.size, 0)
        
        # 调用 peak_widths 函数，传入两个空数组作为参数
        out = peak_widths([], [])
        # 遍历结果数组 out 中的每个元素
        for arr in out:
            # 断言 arr 是 numpy 数组对象
            assert_(isinstance(arr, np.ndarray))
            # 断言 arr 的大小为 0
            assert_equal(arr.size, 0)

    # 使用 pytest 标记忽略特定警告信息
    @pytest.mark.filterwarnings("ignore:some peaks have a width of 0")
    # 测试简单用例，验证不同相对高度时的宽度计算
    def test_basic(self):
        """
        Test a simple use case with easy to verify results at different relative
        heights.
        """
        # 定义一个 numpy 数组 x
        x = np.array([1, 0, 1, 2, 1, 0, -1])
        # 设置显著性
        prominence = 2
        # 遍历不同相对高度的测试数据
        for rel_height, width_true, lip_true, rip_true in [
            (0., 0., 3., 3.),  # raises warning
            (0.25, 1., 2.5, 3.5),
            (0.5, 2., 2., 4.),
            (0.75, 3., 1.5, 4.5),
            (1., 4., 1., 5.),
            (2., 5., 1., 6.),
            (3., 5., 1., 6.)
        ]:
            # 调用 peak_widths 函数，计算宽度和相关信息
            width_calc, height, lip_calc, rip_calc = peak_widths(
                x, [3], rel_height)
            # 断言计算得到的宽度与预期的宽度值接近
            assert_allclose(width_calc, width_true)
            # 断言高度值与预期值接近
            assert_allclose(height, 2 - rel_height * prominence)
            # 断言左侧斜率与预期值接近
            assert_allclose(lip_calc, lip_true)
            # 断言右侧斜率与预期值接近
            assert_allclose(rip_calc, rip_true)

    # 测试非连续输入数组的情况
    def test_non_contiguous(self):
        """
        Test with non-C-contiguous input arrays.
        """
        # 创建一个非 C 连续的 numpy 数组 x
        x = np.repeat([0, 100, 50], 4)
        # 创建一个非 C 连续的 numpy 数组 peaks
        peaks = np.repeat([1], 3)
        # 调用 peak_widths 函数，对非连续输入数组进行宽度计算
        result = peak_widths(x[::4], peaks[::3])
        # 断言结果与预期值匹配
        assert_equal(result, [0.75, 75, 0.75, 1.5])
    def test_exceptions(self):
        """
        Verify that argument validation works as intended.
        """
        with raises(ValueError, match='1-D array'):
            # 当 x 的维度大于 1 时，抛出 ValueError 异常
            peak_widths(np.zeros((3, 4)), np.ones(3))
        with raises(ValueError, match='1-D array'):
            # 当 x 的维度小于 1 时，抛出 ValueError 异常
            peak_widths(3, [0])
        with raises(ValueError, match='1-D array'):
            # 当 peaks 的维度大于 1 时，抛出 ValueError 异常
            peak_widths(np.arange(10), np.ones((3, 2), dtype=np.intp))
        with raises(ValueError, match='1-D array'):
            # 当 peaks 的维度小于 1 时，抛出 ValueError 异常
            peak_widths(np.arange(10), 3)
        with raises(ValueError, match='not a valid index'):
            # 当 peak 位置超过 x.size 时，抛出 ValueError 异常
            peak_widths(np.arange(10), [8, 11])
        with raises(ValueError, match='not a valid index'):
            # 当 x 是空的且 peaks 被提供时，抛出 ValueError 异常
            peak_widths([], [1, 2])
        with raises(TypeError, match='cannot safely cast'):
            # 当 peak 不能安全地转换为 intp 类型时，抛出 TypeError 异常
            peak_widths(np.arange(10), [1.1, 2.3])
        with raises(ValueError, match='rel_height'):
            # 当 rel_height 小于 0 时，抛出 ValueError 异常
            peak_widths([0, 1, 0, 1, 0], [1, 3], rel_height=-1)
        with raises(TypeError, match='None'):
            # 当 prominence_data 包含 None 时，抛出 TypeError 异常
            peak_widths([1, 2, 1], [1], prominence_data=(None, None, None))

    def test_warnings(self):
        """
        Verify that appropriate warnings are raised.
        """
        msg = "some peaks have a width of 0"
        with warns(PeakPropertyWarning, match=msg):
            # Case: rel_height 为 0 的情况下，引发 PeakPropertyWarning 警告
            peak_widths([0, 1, 0], [1], rel_height=0)
        with warns(PeakPropertyWarning, match=msg):
            # Case: prominence 为 0 且 bases 相同的情况下，引发 PeakPropertyWarning 警告
            peak_widths(
                [0, 1, 1, 1, 0], [2],
                prominence_data=(np.array([0.], np.float64),
                                 np.array([2], np.intp),
                                 np.array([2], np.intp))
            )
    def test_mismatching_prominence_data(self):
        """Test with mismatching peak and / or prominence data."""
        x = [0, 1, 0]  # 示例输入数据 x
        peak = [1]  # 示例峰值列表 peak
        for i, (prominences, left_bases, right_bases) in enumerate([
            ((1.,), (-1,), (2,)),  # 情况1：左基准不在 x 中
            ((1.,), (0,), (3,)),  # 情况2：右基准不在 x 中
            ((1.,), (2,), (0,)),  # 情况3：交换的基准与峰值相同
            ((1., 1.), (0, 0), (2, 2)),  # 情况4：数组形状与峰值不匹配
            ((1., 1.), (0,), (2,)),  # 情况5：数组形状不同
            ((1.,), (0, 0), (2,)),  # 情况6：数组形状不同
            ((1.,), (0,), (2, 2))  # 情况7：数组形状不同
        ]):
            # 确保输入匹配 signal.peak_prominences 的输出
            prominence_data = (np.array(prominences, dtype=np.float64),
                               np.array(left_bases, dtype=np.intp),
                               np.array(right_bases, dtype=np.intp))
            # 测试正确的异常
            if i < 3:
                match = "prominence data is invalid for peak"
            else:
                match = "arrays in `prominence_data` must have the same shape"
            with raises(ValueError, match=match):
                peak_widths(x, peak, prominence_data=prominence_data)

    @pytest.mark.filterwarnings("ignore:some peaks have a width of 0")
    def test_intersection_rules(self):
        """Test if x == eval_height counts as an intersection."""
        x = [0, 1, 2, 1, 3, 3, 3, 1, 2, 1, 0]  # 示例输入数据 x
        # 在评估高度为 1 时，平坦峰可能有两个可能的交点
        # 相对高度为 0 -> 宽度也为 0，会引发警告
        assert_allclose(peak_widths(x, peaks=[5], rel_height=0),
                        [(0.,), (3.,), (5.,), (5.,)])
        # 宽度高度 == x 被视为交点 -> 选择最近的 1
        assert_allclose(peak_widths(x, peaks=[5], rel_height=2/3),
                        [(4.,), (1.,), (3.,), (7.,)])
# 定义一个测试函数，用于验证 `scipy.signal.find_peaks` 函数的条件参数解析
def test_unpack_condition_args():
    # 创建一个长度为10的数组
    x = np.arange(10)
    # amin_true 是 x 的复制
    amin_true = x
    # amax_true 是 amin_true 的每个元素加10
    amax_true = amin_true + 10
    # peaks 是 amin_true 中索引为1开始每隔2个取一个元素组成的数组
    peaks = amin_true[1::2]

    # 测试使用 None 或区间来解包条件参数
    assert_((None, None) == _unpack_condition_args((None, None), x, peaks))
    assert_((1, None) == _unpack_condition_args(1, x, peaks))
    assert_((1, None) == _unpack_condition_args((1, None), x, peaks))
    assert_((None, 2) == _unpack_condition_args((None, 2), x, peaks))
    assert_((3., 4.5) == _unpack_condition_args((3., 4.5), x, peaks))

    # 测试是否正确使用 `peaks` 缩小边界值
    amin_calc, amax_calc = _unpack_condition_args((amin_true, amax_true), x, peaks)
    assert_equal(amin_calc, amin_true[peaks])
    assert_equal(amax_calc, amax_true[peaks])

    # 测试当数组边界与 x 不匹配时是否引发异常
    with raises(ValueError, match="array size of lower"):
        _unpack_condition_args(amin_true, np.arange(11), peaks)
    with raises(ValueError, match="array size of upper"):
        _unpack_condition_args((None, amin_true), np.arange(11), peaks)


class TestFindPeaks:
    
    # 可选返回属性的键集合
    property_keys = {'peak_heights', 'left_thresholds', 'right_thresholds',
                     'prominences', 'left_bases', 'right_bases', 'widths',
                     'width_heights', 'left_ips', 'right_ips'}

    def test_constant(self):
        """
        测试没有局部最大值的信号行为。
        """
        open_interval = (None, None)
        # 调用 find_peaks 函数测试使用开放的区间参数
        peaks, props = find_peaks(np.ones(10),
                                  height=open_interval, threshold=open_interval,
                                  prominence=open_interval, width=open_interval)
        # 断言 peaks 数组大小为0
        assert_(peaks.size == 0)
        # 断言 props 中每个键对应的数组大小为0
        for key in self.property_keys:
            assert_(props[key].size == 0)

    def test_plateau_size(self):
        """
        测试用于峰值的高原大小条件。
        """
        # 准备包含具有 peak_height == plateau_size 的峰值的信号
        plateau_sizes = np.array([1, 2, 3, 4, 8, 20, 111])
        x = np.zeros(plateau_sizes.size * 2 + 1)
        x[1::2] = plateau_sizes
        repeats = np.ones(x.size, dtype=int)
        repeats[1::2] = x[1::2]
        x = np.repeat(x, repeats)

        # 测试完整的输出
        peaks, props = find_peaks(x, plateau_size=(None, None))
        assert_equal(peaks, [1, 3, 7, 11, 18, 33, 100])
        assert_equal(props["plateau_sizes"], plateau_sizes)
        assert_equal(props["left_edges"], peaks - (plateau_sizes - 1) // 2)
        assert_equal(props["right_edges"], peaks + plateau_sizes // 2)

        # 测试条件
        assert_equal(find_peaks(x, plateau_size=4)[0], [11, 18, 33, 100])
        assert_equal(find_peaks(x, plateau_size=(None, 3.5))[0], [1, 3, 7])
        assert_equal(find_peaks(x, plateau_size=(5, 50))[0], [18, 33])
    # 测试峰值的高度条件
    def test_height_condition(self):
        """
        Test height condition for peaks.
        """
        # 定义示例数据 x
        x = (0., 1/3, 0., 2.5, 0, 4., 0)
        # 使用 find_peaks 函数找到所有峰值及其属性，使用默认的高度条件 (None, None)
        peaks, props = find_peaks(x, height=(None, None))
        # 断言找到的峰值索引数组 peaks 符合预期的值
        assert_equal(peaks, np.array([1, 3, 5]))
        # 断言找到的峰值的高度属性 props['peak_heights'] 符合预期的值
        assert_equal(props['peak_heights'], np.array([1/3, 2.5, 4.]))
        # 使用 height=0.5 条件重新查找峰值，断言 peaks 数组符合预期的值
        assert_equal(find_peaks(x, height=0.5)[0], np.array([3, 5]))
        # 使用 height=(None, 3) 条件重新查找峰值，断言 peaks 数组符合预期的值
        assert_equal(find_peaks(x, height=(None, 3))[0], np.array([1, 3]))
        # 使用 height=(2, 3) 条件重新查找峰值，断言 peaks 数组符合预期的值
        assert_equal(find_peaks(x, height=(2, 3))[0], np.array([3]))

    # 测试峰值的阈值条件
    def test_threshold_condition(self):
        """
        Test threshold condition for peaks.
        """
        # 定义示例数据 x
        x = (0, 2, 1, 4, -1)
        # 使用 find_peaks 函数找到所有峰值及其属性，使用默认的阈值条件 (None, None)
        peaks, props = find_peaks(x, threshold=(None, None))
        # 断言找到的峰值索引数组 peaks 符合预期的值
        assert_equal(peaks, np.array([1, 3]))
        # 断言找到的左阈值属性 props['left_thresholds'] 符合预期的值
        assert_equal(props['left_thresholds'], np.array([2, 3]))
        # 断言找到的右阈值属性 props['right_thresholds'] 符合预期的值
        assert_equal(props['right_thresholds'], np.array([1, 5]))
        # 使用 threshold=2 条件重新查找峰值，断言 peaks 数组符合预期的值
        assert_equal(find_peaks(x, threshold=2)[0], np.array([3]))
        # 使用 threshold=3.5 条件重新查找峰值，断言 peaks 数组符合预期的值
        assert_equal(find_peaks(x, threshold=3.5)[0], np.array([]))
        # 使用 threshold=(None, 5) 条件重新查找峰值，断言 peaks 数组符合预期的值
        assert_equal(find_peaks(x, threshold=(None, 5))[0], np.array([1, 3]))
        # 使用 threshold=(None, 4) 条件重新查找峰值，断言 peaks 数组符合预期的值
        assert_equal(find_peaks(x, threshold=(None, 4))[0], np.array([1]))
        # 使用 threshold=(2, 4) 条件重新查找峰值，断言 peaks 数组符合预期的值
        assert_equal(find_peaks(x, threshold=(2, 4))[0], np.array([]))

    # 测试峰值的距离条件
    def test_distance_condition(self):
        """
        Test distance condition for peaks.
        """
        # 生成具有恒定距离的不同高度的峰值
        peaks_all = np.arange(1, 21, 3)
        x = np.zeros(21)
        x[peaks_all] += np.linspace(1, 2, peaks_all.size)

        # 测试是否仍然选择具有“最小”距离的峰值（距离 = 3）
        assert_equal(find_peaks(x, distance=3)[0], peaks_all)

        # 选择每隔一个峰值（距离 > 3）
        peaks_subset = find_peaks(x, distance=3.0001)[0]
        # 测试 peaks_subset 是否是 peaks_all 的子集
        assert_(
            np.setdiff1d(peaks_subset, peaks_all, assume_unique=True).size == 0
        )
        # 测试是否每隔一个峰值被移除
        assert_equal(np.diff(peaks_subset), 6)

        # 测试峰值移除的优先级
        x = [-2, 1, -1, 0, -3]
        peaks_subset = find_peaks(x, distance=10)[0]  # 使用大于 x 大小的距离
        assert_(peaks_subset.size == 1 and peaks_subset[0] == 1)
    def test_prominence_condition(self):
        """
        Test prominence condition for peaks.
        """
        # 创建一个包含100个元素的等间距数组作为测试数据
        x = np.linspace(0, 10, 100)
        # 创建一个包含奇数索引的数组，作为预期峰值位置
        peaks_true = np.arange(1, 99, 2)
        # 创建一个与 peaks_true 大小相同的数组，用于指定每个峰值的偏移量
        offset = np.linspace(1, 10, peaks_true.size)
        # 将 x 中 peaks_true 索引位置的元素增加对应的偏移量
        x[peaks_true] += offset
        # 计算峰值的显著性，即每个峰值的高度差
        prominences = x[peaks_true] - x[peaks_true + 1]
        # 指定一个显著性范围区间
        interval = (3, 9)
        # 选择满足显著性在指定区间内的峰值索引
        keep = np.nonzero(
            (interval[0] <= prominences) & (prominences <= interval[1]))

        # 调用 find_peaks 函数，计算实际的峰值位置和属性
        peaks_calc, properties = find_peaks(x, prominence=interval)
        # 断言计算得到的峰值位置与预期的一致
        assert_equal(peaks_calc, peaks_true[keep])
        # 断言属性中的显著性与预期的一致
        assert_equal(properties['prominences'], prominences[keep])
        # 断言属性中左侧基线的值为0
        assert_equal(properties['left_bases'], 0)
        # 断言属性中右侧基线的值为每个峰值位置加1
        assert_equal(properties['right_bases'], peaks_true[keep] + 1)

    def test_width_condition(self):
        """
        Test width condition for peaks.
        """
        # 创建一个示例数组，用于测试峰值的宽度条件
        x = np.array([1, 0, 1, 2, 1, 0, -1, 4, 0])
        # 调用 find_peaks 函数，指定宽度和相对高度条件
        peaks, props = find_peaks(x, width=(None, 2), rel_height=0.75)
        # 断言找到的峰值数量为1
        assert_equal(peaks.size, 1)
        # 断言找到的峰值位置为7
        assert_equal(peaks, 7)
        # 断言属性中宽度的值接近1.35
        assert_allclose(props['widths'], 1.35)
        # 断言属性中宽度高度的值接近1.0
        assert_allclose(props['width_heights'], 1.)
        # 断言属性中左侧端点位置的值接近6.4
        assert_allclose(props['left_ips'], 6.4)
        # 断言属性中右侧端点位置的值接近7.75
        assert_allclose(props['right_ips'], 7.75)

    def test_properties(self):
        """
        Test returned properties.
        """
        # 指定一个开放的高度、阈值、显著性和宽度区间
        open_interval = (None, None)
        # 创建一个示例数组，用于测试返回的峰值属性
        x = [0, 1, 0, 2, 1.5, 0, 3, 0, 5, 9]
        # 调用 find_peaks 函数，获取峰值位置和属性
        peaks, props = find_peaks(x,
                                  height=open_interval, threshold=open_interval,
                                  prominence=open_interval, width=open_interval)
        # 断言属性的数量与预期的属性键数量相同
        assert_(len(props) == len(self.property_keys))
        # 对于每个属性键，断言峰值位置的数量与属性值的数量相同
        for key in self.property_keys:
            assert_(peaks.size == props[key].size)

    def test_raises(self):
        """
        Test exceptions raised by function.
        """
        # 使用 pytest 的 raises 断言捕获函数抛出的特定异常
        with raises(ValueError, match="1-D array"):
            find_peaks(np.array(1))
        with raises(ValueError, match="1-D array"):
            find_peaks(np.ones((2, 2)))
        with raises(ValueError, match="distance"):
            find_peaks(np.arange(10), distance=-1)

    @pytest.mark.filterwarnings("ignore:some peaks have a prominence of 0",
                                "ignore:some peaks have a width of 0")
    def test_wlen_smaller_plateau(self):
        """
        Test behavior of prominence and width calculation if the given window
        length is smaller than a peak's plateau size.

        Regression test for gh-9110.
        """
        # 调用 find_peaks 函数，计算给定窗口长度情况下的显著性和宽度
        peaks, props = find_peaks([0, 1, 1, 1, 0], prominence=(None, None),
                                  width=(None, None), wlen=2)
        # 断言找到的峰值位置为2
        assert_equal(peaks, 2)
        # 断言属性中显著性的值为0
        assert_equal(props["prominences"], 0)
        # 断言属性中宽度的值为0
        assert_equal(props["widths"], 0)
        # 断言属性中宽度高度的值为1
        assert_equal(props["width_heights"], 1)
        # 对于每个端点的属性，断言其值与找到的峰值位置相同
        for key in ("left_bases", "right_bases", "left_ips", "right_ips"):
            assert_equal(props[key], peaks)
    # 使用 pytest.mark.parametrize 装饰器标记测试方法，参数化测试用例
    @pytest.mark.parametrize("kwargs", [
        {},  # 参数为空字典的情况
        {"distance": 3.0},  # 设置 distance 参数为 3.0
        {"prominence": (None, None)},  # 设置 prominence 参数为 (None, None)
        {"width": (None, 2)},  # 设置 width 参数为 (None, 2)
    ])
    # 定义测试只读数组被接受的情况
    def test_readonly_array(self, kwargs):
        """
        Test readonly arrays are accepted.
        """
        # 创建一个从 0 到 10 的包含 15 个元素的数组 x
        x = np.linspace(0, 10, 15)
        # 将数组 x 复制给 x_readonly，使其为只读
        x_readonly = x.copy()
        x_readonly.flags.writeable = False

        # 查找数组 x 中的峰值索引
        peaks, _ = find_peaks(x)
        # 查找只读数组 x_readonly 中的峰值索引，传入 kwargs 参数
        peaks_readonly, _ = find_peaks(x_readonly, **kwargs)

        # 使用 assert_allclose 函数断言 peaks 和 peaks_readonly 的值接近
        assert_allclose(peaks, peaks_readonly)
# 定义一个测试类 TestFindPeaksCwt，用于测试 find_peaks_cwt 函数的各种情况
class TestFindPeaksCwt:

    # 测试在精确情况下找到峰值位置
    def test_find_peaks_exact(self):
        """
        生成一系列高斯函数并尝试找到峰值位置。
        """
        # 设定高斯函数的标准差
        sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
        # 生成数据点的数量
        num_points = 500
        # 使用 _gen_gaussians_even 函数生成测试数据和实际峰值位置
        test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
        # 设定需要尝试的峰宽度范围
        widths = np.arange(0.1, max(sigmas))
        # 调用 find_peaks_cwt 函数查找峰值
        found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=0,
                                         min_length=None)
        # 使用 np.testing.assert_array_equal 断言找到的峰值位置与预期位置相等
        np.testing.assert_array_equal(found_locs, act_locs,
                        "Found maximum locations did not equal those expected")

    # 测试在带噪声情况下找到峰值位置
    def test_find_peaks_withnoise(self):
        """
        验证对加入噪声的一系列高斯函数能（大致）找到峰值位置。
        """
        # 设定高斯函数的标准差
        sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
        # 生成数据点的数量
        num_points = 500
        # 使用 _gen_gaussians_even 函数生成测试数据和实际峰值位置
        test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
        # 设定需要尝试的峰宽度范围
        widths = np.arange(0.1, max(sigmas))
        # 设定噪声的幅度
        noise_amp = 0.07
        # 设定随机数种子
        np.random.seed(18181911)
        # 在测试数据中加入随机噪声
        test_data += (np.random.rand(num_points) - 0.5)*(2*noise_amp)
        # 调用 find_peaks_cwt 函数查找峰值
        found_locs = find_peaks_cwt(test_data, widths, min_length=15,
                                         gap_thresh=1, min_snr=noise_amp / 5)

        # 使用 np.testing.assert_equal 断言找到的峰值数量与预期数量相等
        np.testing.assert_equal(len(found_locs), len(act_locs), 'Different number' +
                                'of peaks found than expected')
        # 计算找到峰值位置与实际位置的差异
        diffs = np.abs(found_locs - act_locs)
        # 设定最大允许的位置差异
        max_diffs = np.array(sigmas) / 5
        # 使用 np.testing.assert_array_less 断言位置差异在最大允许差异范围内
        np.testing.assert_array_less(diffs, max_diffs, 'Maximum location differed' +
                                     'by more than %s' % (max_diffs))

    # 测试在仅有噪声数据中找不到峰值
    def test_find_peaks_nopeak(self):
        """
        验证在仅有噪声数据中不能找到峰值。
        """
        # 设定噪声的幅度
        noise_amp = 1.0
        # 设定数据点的数量
        num_points = 100
        # 设定随机数种子
        np.random.seed(181819141)
        # 生成仅有噪声的测试数据
        test_data = (np.random.rand(num_points) - 0.5)*(2*noise_amp)
        # 设定需要尝试的峰宽度范围
        widths = np.arange(10, 50)
        # 调用 find_peaks_cwt 函数查找峰值
        found_locs = find_peaks_cwt(test_data, widths, min_snr=5, noise_perc=30)
        # 使用 np.testing.assert_equal 断言找到的峰值数量为 0
        np.testing.assert_equal(len(found_locs), 0)

    # 测试使用非默认小波变换函数查找峰值
    def test_find_peaks_with_non_default_wavelets(self):
        """
        验证使用非默认小波变换函数能找到峰值位置。
        """
        # 生成一个具有高斯分布的数据
        x = gaussian(200, 2)
        # 设定需要尝试的峰宽度范围
        widths = np.array([1, 2, 3, 4])
        # 调用 find_peaks_cwt 函数查找峰值
        a = find_peaks_cwt(x, widths, wavelet=gaussian)

        # 使用 np.testing.assert_equal 断言找到的峰值位置为 [100]
        np.testing.assert_equal(np.array([100]), a)
    def test_find_peaks_window_size(self):
        """
        Verify that window_size is passed correctly to private function and
        affects the result.
        """
        sigmas = [2.0, 2.0]  # 设置高斯函数的标准差
        num_points = 1000  # 生成数据点的数量
        test_data, act_locs = _gen_gaussians_even(sigmas, num_points)  # 使用均匀分布生成高斯数据和其位置
        widths = np.arange(0.1, max(sigmas), 0.2)  # 设置不同的宽度范围
        noise_amp = 0.05  # 噪声振幅
        np.random.seed(18181911)  # 设置随机种子
        test_data += (np.random.rand(num_points) - 0.5)*(2*noise_amp)  # 添加均匀分布的噪声

        # Possibly contrived negative region to throw off peak finding
        # when window_size is too large
        test_data[250:320] -= 1  # 为了干扰峰值查找，设置一个可能的负区域

        found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=3,
                                    min_length=None, window_size=None)  # 使用默认的窗口大小调用峰值查找函数
        with pytest.raises(AssertionError):  # 断言抛出异常
            assert found_locs.size == act_locs.size  # 检查找到的峰值数量是否与实际数量相等

        found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=3,
                                    min_length=None, window_size=20)  # 使用窗口大小为20调用峰值查找函数
        assert found_locs.size == act_locs.size  # 断言找到的峰值数量与实际数量相等

    def test_find_peaks_with_one_width(self):
        """
        Verify that the `width` argument
        in `find_peaks_cwt` can be a float
        """
        xs = np.arange(0, np.pi, 0.05)  # 生成0到π之间的数据点
        test_data = np.sin(xs)  # 计算正弦函数值作为测试数据
        widths = 1  # 设置宽度为1
        found_locs = find_peaks_cwt(test_data, widths)  # 调用峰值查找函数

        np.testing.assert_equal(found_locs, 32)  # 使用 NumPy 断言检查找到的峰值是否等于32
```