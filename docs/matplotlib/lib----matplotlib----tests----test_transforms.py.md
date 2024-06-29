# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_transforms.py`

```
import copy  # 导入copy模块

import numpy as np  # 导入NumPy库
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_equal, assert_array_almost_equal)  # 导入NumPy测试模块中的断言函数

import pytest  # 导入pytest测试框架

from matplotlib import scale  # 导入matplotlib中的scale模块
import matplotlib.pyplot as plt  # 导入matplotlib中的pyplot模块
import matplotlib.patches as mpatches  # 导入matplotlib中的patches模块，重命名为mpatches
import matplotlib.transforms as mtransforms  # 导入matplotlib中的transforms模块，重命名为mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox  # 导入matplotlib中的Affine2D、Bbox、TransformedBbox类
from matplotlib.path import Path  # 导入matplotlib中的Path类
from matplotlib.testing.decorators import image_comparison, check_figures_equal  # 导入matplotlib测试模块中的装饰器函数

class TestAffine2D:  # 定义测试类TestAffine2D

    single_point = [1.0, 1.0]  # 定义单个点的坐标
    multiple_points = [[0.0, 2.0], [3.0, 3.0], [4.0, 0.0]]  # 定义多个点的坐标列表
    pivot = single_point  # 使用单个点作为旋转的中心点

    def test_init(self):
        Affine2D([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 创建Affine2D对象并初始化矩阵
        Affine2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], int))  # 使用整数数组初始化Affine2D对象
        Affine2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float))  # 使用浮点数数组初始化Affine2D对象

    def test_values(self):
        np.random.seed(19680801)  # 设定随机数种子
        values = np.random.random(6)  # 生成长度为6的随机数数组
        assert_array_equal(Affine2D.from_values(*values).to_values(), values)  # 断言从值生成的Affine2D对象与原始值数组相等

    def test_modify_inplace(self):
        # Some polar transforms require modifying the matrix in place.
        trans = Affine2D()  # 创建Affine2D对象
        mtx = trans.get_matrix()  # 获取Affine2D对象的变换矩阵
        mtx[0, 0] = 42  # 修改矩阵中的特定元素
        assert_array_equal(trans.get_matrix(), [[42, 0, 0], [0, 1, 0], [0, 0, 1]])  # 断言修改后的矩阵与预期相等

    def test_clear(self):
        a = Affine2D(np.random.rand(3, 3) + 5)  # 创建一个非单位矩阵的Affine2D对象
        a.clear()  # 清空Affine2D对象
        assert_array_equal(a.get_matrix(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 断言清空后的矩阵为单位矩阵

    def test_rotate(self):
        r_pi_2 = Affine2D().rotate(np.pi / 2)  # 创建一个旋转90度的Affine2D对象
        r90 = Affine2D().rotate_deg(90)  # 创建一个旋转90度的Affine2D对象
        assert_array_equal(r_pi_2.get_matrix(), r90.get_matrix())  # 断言两个旋转矩阵相等
        assert_array_almost_equal(r90.transform(self.single_point), [-1, 1])  # 断言单个点经过旋转后的坐标近似于预期值
        assert_array_almost_equal(r90.transform(self.multiple_points),
                                  [[-2, 0], [-3, 3], [0, 4]])  # 断言多个点经过旋转后的坐标近似于预期值

        r_pi = Affine2D().rotate(np.pi)  # 创建一个旋转180度的Affine2D对象
        r180 = Affine2D().rotate_deg(180)  # 创建一个旋转180度的Affine2D对象
        assert_array_equal(r_pi.get_matrix(), r180.get_matrix())  # 断言两个旋转矩阵相等
        assert_array_almost_equal(r180.transform(self.single_point), [-1, -1])  # 断言单个点经过旋转后的坐标近似于预期值
        assert_array_almost_equal(r180.transform(self.multiple_points),
                                  [[0, -2], [-3, -3], [-4, 0]])  # 断言多个点经过旋转后的坐标近似于预期值

        r_pi_3_2 = Affine2D().rotate(3 * np.pi / 2)  # 创建一个旋转270度的Affine2D对象
        r270 = Affine2D().rotate_deg(270)  # 创建一个旋转270度的Affine2D对象
        assert_array_equal(r_pi_3_2.get_matrix(), r270.get_matrix())  # 断言两个旋转矩阵相等
        assert_array_almost_equal(r270.transform(self.single_point), [1, -1])  # 断言单个点经过旋转后的坐标近似于预期值
        assert_array_almost_equal(r270.transform(self.multiple_points),
                                  [[2, 0], [3, -3], [0, -4]])  # 断言多个点经过旋转后的坐标近似于预期值

        assert_array_equal((r90 + r90).get_matrix(), r180.get_matrix())  # 断言两个旋转90度的Affine2D对象相加得到旋转180度的Affine2D对象的矩阵
        assert_array_equal((r90 + r180).get_matrix(), r270.get_matrix())  # 断言一个旋转90度的Affine2D对象和一个旋转180度的Affine2D对象相加得到旋转270度的Affine2D对象的矩阵
    # 测试围绕某个点旋转的方法
    def test_rotate_around(self):
        # 创建一个围绕给定点self.pivot旋转π/2弧度的Affine2D对象
        r_pi_2 = Affine2D().rotate_around(*self.pivot, np.pi / 2)
        # 创建一个围绕给定点self.pivot旋转90度的Affine2D对象
        r90 = Affine2D().rotate_deg_around(*self.pivot, 90)
        # 断言两个旋转变换的矩阵是否相等
        assert_array_equal(r_pi_2.get_matrix(), r90.get_matrix())
        # 断言对单点进行旋转后的结果近似为[1, 1]
        assert_array_almost_equal(r90.transform(self.single_point), [1, 1])
        # 断言对多个点进行旋转后的结果近似为[[0, 0], [-1, 3], [2, 4]]

        # 创建一个围绕给定点self.pivot旋转π弧度的Affine2D对象
        r_pi = Affine2D().rotate_around(*self.pivot, np.pi)
        # 创建一个围绕给定点self.pivot旋转180度的Affine2D对象
        r180 = Affine2D().rotate_deg_around(*self.pivot, 180)
        # 断言两个旋转变换的矩阵是否相等
        assert_array_equal(r_pi.get_matrix(), r180.get_matrix())
        # 断言对单点进行旋转后的结果近似为[1, 1]
        assert_array_almost_equal(r180.transform(self.single_point), [1, 1])
        # 断言对多个点进行旋转后的结果近似为[[2, 0], [-1, -1], [-2, 2]]

        # 创建一个围绕给定点self.pivot旋转3π/2弧度的Affine2D对象
        r_pi_3_2 = Affine2D().rotate_around(*self.pivot, 3 * np.pi / 2)
        # 创建一个围绕给定点self.pivot旋转270度的Affine2D对象
        r270 = Affine2D().rotate_deg_around(*self.pivot, 270)
        # 断言两个旋转变换的矩阵是否相等
        assert_array_equal(r_pi_3_2.get_matrix(), r270.get_matrix())
        # 断言对单点进行旋转后的结果近似为[1, 1]
        assert_array_almost_equal(r270.transform(self.single_point), [1, 1])
        # 断言对多个点进行旋转后的结果近似为[[2, 2], [3, -1], [0, -2]]

        # 断言两个旋转变换之和的矩阵是否近似等于r180的矩阵
        assert_array_almost_equal((r90 + r90).get_matrix(), r180.get_matrix())
        # 断言两个旋转变换之和的矩阵是否近似等于r270的矩阵
        assert_array_almost_equal((r90 + r180).get_matrix(), r270.get_matrix())

    # 测试缩放方法
    def test_scale(self):
        # 创建一个在x轴上缩放3倍，在y轴上不变的Affine2D对象
        sx = Affine2D().scale(3, 1)
        # 创建一个在y轴上缩放2倍，在x轴上不变的Affine2D对象
        sy = Affine2D().scale(1, -2)
        # 创建一个同时在x轴和y轴上分别缩放3倍和-2倍的Affine2D对象
        trans = Affine2D().scale(3, -2)
        # 断言两个缩放变换之和的矩阵是否等于trans的矩阵
        assert_array_equal((sx + sy).get_matrix(), trans.get_matrix())
        # 断言对单点进行缩放后的结果是否等于[3, -2]
        assert_array_equal(trans.transform(self.single_point), [3, -2])
        # 断言对多个点进行缩放后的结果是否等于[[0, -4], [9, -6], [12, 0]]

    # 测试倾斜方法
    def test_skew(self):
        # 创建一个倾斜角度分别为π/8和π/12的Affine2D对象
        trans_rad = Affine2D().skew(np.pi / 8, np.pi / 12)
        # 创建一个倾斜角度分别为22.5度和15度的Affine2D对象
        trans_deg = Affine2D().skew_deg(22.5, 15)
        # 断言两个倾斜变换的矩阵是否相等
        assert_array_equal(trans_rad.get_matrix(), trans_deg.get_matrix())
        # 创建一个倾斜角度分别为26.5650512度和14.0362435度的Affine2D对象
        trans = Affine2D().skew_deg(26.5650512, 14.0362435)
        # 断言对单点进行倾斜后的结果近似为[1.5, 1.25]
        assert_array_almost_equal(trans.transform(self.single_point), [1.5, 1.25])
        # 断言对多个点进行倾斜后的结果近似为[[1, 2], [4.5, 3.75], [4, 1]]

    # 测试平移方法
    def test_translate(self):
        # 创建一个沿x轴平移23个单位，y轴不变的Affine2D对象
        tx = Affine2D().translate(23, 0)
        # 创建一个沿y轴平移42个单位，x轴不变的Affine2D对象
        ty = Affine2D().translate(0, 42)
        # 创建一个同时在x轴和y轴上分别平移23和42个单位的Affine2D对象
        trans = Affine2D().translate(23, 42)
        # 断言两个平移变换之和的矩阵是否等于trans的矩阵
        assert_array_equal((tx + ty).get_matrix(), trans.get_matrix())
        # 断言对单点进行平移后的结果是否等于[24, 43]
        assert_array_equal(trans.transform(self.single_point), [24, 43])
        # 断言对多个点进行平移后的结果是否等于[[23, 44], [26, 45], [27, 42]]
    # 定义一个测试方法，用于验证 Affine2D 类的旋转和组合操作
    def test_rotate_plus_other(self):
        # 创建一个仿射变换对象，先旋转90度，再围绕给定中心点旋转180度，并将结果赋给 trans
        trans = Affine2D().rotate_deg(90).rotate_deg_around(*self.pivot, 180)
        # 分别创建两个仿射变换对象，一个旋转90度，另一个先旋转90度再围绕给定中心点旋转180度，并将它们相加赋给 trans_added
        trans_added = (Affine2D().rotate_deg(90) +
                       Affine2D().rotate_deg_around(*self.pivot, 180))
        # 断言两个仿射变换对象的矩阵表示相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象 trans 对单个点进行变换后的结果近似为 [3, 1]
        assert_array_almost_equal(trans.transform(self.single_point), [3, 1])
        # 断言仿射变换对象 trans 对多个点进行变换后的结果近似为 [[4, 2], [5, -1], [2, -2]]
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[4, 2], [5, -1], [2, -2]])

        # 创建一个仿射变换对象，先旋转90度，再按指定比例缩放，将结果赋给 trans
        trans = Affine2D().rotate_deg(90).scale(3, -2)
        # 分别创建两个仿射变换对象，一个旋转90度，另一个按指定比例缩放，并将它们相加赋给 trans_added
        trans_added = Affine2D().rotate_deg(90) + Affine2D().scale(3, -2)
        # 断言两个仿射变换对象的矩阵表示相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象 trans 对单个点进行变换后的结果近似为 [-3, -2]
        assert_array_almost_equal(trans.transform(self.single_point), [-3, -2])
        # 断言仿射变换对象 trans 对多个点进行变换后的结果近似为 [[-6, 0], [-9, -6], [0, -8]]
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[-6, 0], [-9, -6], [0, -8]])

        # 创建一个仿射变换对象，先旋转90度，再按指定角度进行斜切，将结果赋给 trans
        trans = (Affine2D().rotate_deg(90)
                 .skew_deg(26.5650512, 14.0362435))  # ~atan(0.5), ~atan(0.25)
        # 分别创建两个仿射变换对象，一个旋转90度，另一个按指定角度进行斜切，并将它们相加赋给 trans_added
        trans_added = (Affine2D().rotate_deg(90) +
                       Affine2D().skew_deg(26.5650512, 14.0362435))
        # 断言两个仿射变换对象的矩阵表示相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象 trans 对单个点进行变换后的结果近似为 [-0.5, 0.75]
        assert_array_almost_equal(trans.transform(self.single_point), [-0.5, 0.75])
        # 断言仿射变换对象 trans 对多个点进行变换后的结果近似为 [[-2, -0.5], [-1.5, 2.25], [2, 4]]
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[-2, -0.5], [-1.5, 2.25], [2, 4]])

        # 创建一个仿射变换对象，先旋转90度，再按指定平移距离平移，将结果赋给 trans
        trans = Affine2D().rotate_deg(90).translate(23, 42)
        # 分别创建两个仿射变换对象，一个旋转90度，另一个按指定平移距离平移，并将它们相加赋给 trans_added
        trans_added = Affine2D().rotate_deg(90) + Affine2D().translate(23, 42)
        # 断言两个仿射变换对象的矩阵表示相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象 trans 对单个点进行变换后的结果近似为 [22, 43]
        assert_array_almost_equal(trans.transform(self.single_point), [22, 43])
        # 断言仿射变换对象 trans 对多个点进行变换后的结果近似为 [[21, 42], [20, 45], [23, 46]]
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[21, 42], [20, 45], [23, 46]])
    # 定义测试方法，用于测试仿射变换和其组合操作
    def test_rotate_around_plus_other(self):
        # 创建一个绕指定点旋转90度并绕原点旋转180度的仿射变换对象
        trans = Affine2D().rotate_deg_around(*self.pivot, 90).rotate_deg(180)
        # 创建两个仿射变换对象并将它们组合，分别旋转和添加
        trans_added = (Affine2D().rotate_deg_around(*self.pivot, 90) +
                       Affine2D().rotate_deg(180))
        # 断言两个仿射变换对象的变换矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单个点的变换结果是否正确
        assert_array_almost_equal(trans.transform(self.single_point), [-1, -1])
        # 断言仿射变换对象对多个点的变换结果是否正确
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[0, 0], [1, -3], [-2, -4]])

        # 创建一个绕指定点旋转90度并缩放3倍和-2倍的仿射变换对象
        trans = Affine2D().rotate_deg_around(*self.pivot, 90).scale(3, -2)
        # 创建两个仿射变换对象并将它们组合，分别旋转和缩放
        trans_added = (Affine2D().rotate_deg_around(*self.pivot, 90) +
                       Affine2D().scale(3, -2))
        # 断言两个仿射变换对象的变换矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单个点的变换结果是否正确
        assert_array_almost_equal(trans.transform(self.single_point), [3, -2])
        # 断言仿射变换对象对多个点的变换结果是否正确
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[0, 0], [-3, -6], [6, -8]])

        # 创建一个绕指定点旋转90度并剪切26.5650512度和14.0362435度的仿射变换对象
        trans = (Affine2D().rotate_deg_around(*self.pivot, 90)
                 .skew_deg(26.5650512, 14.0362435))  # ~atan(0.5), ~atan(0.25)
        # 创建两个仿射变换对象并将它们组合，分别旋转和剪切
        trans_added = (Affine2D().rotate_deg_around(*self.pivot, 90) +
                       Affine2D().skew_deg(26.5650512, 14.0362435))
        # 断言两个仿射变换对象的变换矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单个点的变换结果是否正确
        assert_array_almost_equal(trans.transform(self.single_point), [1.5, 1.25])
        # 断言仿射变换对象对多个点的变换结果是否正确
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[0, 0], [0.5, 2.75], [4, 4.5]])

        # 创建一个绕指定点旋转90度并平移23和42个单位的仿射变换对象
        trans = Affine2D().rotate_deg_around(*self.pivot, 90).translate(23, 42)
        # 创建两个仿射变换对象并将它们组合，分别旋转和平移
        trans_added = (Affine2D().rotate_deg_around(*self.pivot, 90) +
                       Affine2D().translate(23, 42))
        # 断言两个仿射变换对象的变换矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单个点的变换结果是否正确
        assert_array_almost_equal(trans.transform(self.single_point), [24, 43])
        # 断言仿射变换对象对多个点的变换结果是否正确
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[23, 42], [22, 45], [25, 46]])
    # 定义测试方法，用于测试仿射变换和其加法操作的结果
    def test_scale_plus_other(self):
        # 创建一个缩放比例为3倍、-2倍，再旋转90度的仿射变换对象
        trans = Affine2D().scale(3, -2).rotate_deg(90)
        # 创建两个独立的仿射变换对象，分别缩放和旋转，然后将它们相加
        trans_added = Affine2D().scale(3, -2) + Affine2D().rotate_deg(90)
        # 断言两个仿射变换对象的变换矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单点进行变换的结果是否符合预期
        assert_array_equal(trans.transform(self.single_point), [2, 3])
        # 断言仿射变换对象对多个点进行变换的结果是否符合预期
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[4, 0], [6, 9], [0, 12]])

        # 创建一个缩放比例为3倍、-2倍，围绕给定的旋转中心点旋转90度的仿射变换对象
        trans = Affine2D().scale(3, -2).rotate_deg_around(*self.pivot, 90)
        # 创建两个独立的仿射变换对象，分别缩放和围绕指定点旋转，然后将它们相加
        trans_added = (Affine2D().scale(3, -2) +
                       Affine2D().rotate_deg_around(*self.pivot, 90))
        # 断言两个仿射变换对象的变换矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单点进行变换的结果是否符合预期
        assert_array_equal(trans.transform(self.single_point), [4, 3])
        # 断言仿射变换对象对多个点进行变换的结果是否符合预期
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[6, 0], [8, 9], [2, 12]])

        # 创建一个缩放比例为3倍、-2倍，再以给定角度倾斜的仿射变换对象
        trans = (Affine2D().scale(3, -2)
                 .skew_deg(26.5650512, 14.0362435))  # ~atan(0.5), ~atan(0.25)
        # 创建两个独立的仿射变换对象，分别缩放和倾斜，然后将它们相加
        trans_added = (Affine2D().scale(3, -2) +
                       Affine2D().skew_deg(26.5650512, 14.0362435))
        # 断言两个仿射变换对象的变换矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单点进行变换的结果是否符合预期
        assert_array_almost_equal(trans.transform(self.single_point), [2, -1.25])
        # 断言仿射变换对象对多个点进行变换的结果是否符合预期
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[-2, -4], [6, -3.75], [12, 3]])

        # 创建一个缩放比例为3倍、-2倍，再平移至指定坐标的仿射变换对象
        trans = Affine2D().scale(3, -2).translate(23, 42)
        # 创建两个独立的仿射变换对象，分别缩放和平移，然后将它们相加
        trans_added = Affine2D().scale(3, -2) + Affine2D().translate(23, 42)
        # 断言两个仿射变换对象的变换矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单点进行变换的结果是否符合预期
        assert_array_equal(trans.transform(self.single_point), [26, 40])
        # 断言仿射变换对象对多个点进行变换的结果是否符合预期
        assert_array_equal(trans.transform(self.multiple_points),
                           [[23, 38], [32, 36], [35, 42]])
    def test_skew_plus_other(self):
        # 使用 ~atan(0.5), ~atan(0.25) 产生大致舍入的输出数值。
        trans = Affine2D().skew_deg(26.5650512, 14.0362435).rotate_deg(90)
        trans_added = (Affine2D().skew_deg(26.5650512, 14.0362435) +
                       Affine2D().rotate_deg(90))
        # 断言两个仿射变换对象的矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单个点进行变换后的结果
        assert_array_almost_equal(trans.transform(self.single_point), [-1.25, 1.5])
        # 断言仿射变换对象对多个点进行变换后的结果
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[-2, 1], [-3.75, 4.5], [-1, 4]])

        trans = (Affine2D().skew_deg(26.5650512, 14.0362435)
                 .rotate_deg_around(*self.pivot, 90))
        trans_added = (Affine2D().skew_deg(26.5650512, 14.0362435) +
                       Affine2D().rotate_deg_around(*self.pivot, 90))
        # 断言两个仿射变换对象的矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单个点进行变换后的结果
        assert_array_almost_equal(trans.transform(self.single_point), [0.75, 1.5])
        # 断言仿射变换对象对多个点进行变换后的结果
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[0, 1], [-1.75, 4.5], [1, 4]])

        trans = Affine2D().skew_deg(26.5650512, 14.0362435).scale(3, -2)
        trans_added = (Affine2D().skew_deg(26.5650512, 14.0362435) +
                       Affine2D().scale(3, -2))
        # 断言两个仿射变换对象的矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单个点进行变换后的结果
        assert_array_almost_equal(trans.transform(self.single_point), [4.5, -2.5])
        # 断言仿射变换对象对多个点进行变换后的结果
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[3, -4], [13.5, -7.5], [12, -2]])

        trans = Affine2D().skew_deg(26.5650512, 14.0362435).translate(23, 42)
        trans_added = (Affine2D().skew_deg(26.5650512, 14.0362435) +
                       Affine2D().translate(23, 42))
        # 断言两个仿射变换对象的矩阵是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言仿射变换对象对单个点进行变换后的结果
        assert_array_almost_equal(trans.transform(self.single_point), [24.5, 43.25])
        # 断言仿射变换对象对多个点进行变换后的结果
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[24, 44], [27.5, 45.75], [27, 43]])
    def test_translate_plus_other(self):
        # 创建一个平移矩阵，平移向量为 (23, 42)，并且添加一个90度的旋转
        trans = Affine2D().translate(23, 42).rotate_deg(90)
        # 分别创建一个平移矩阵和一个旋转矩阵，然后将它们相加
        trans_added = Affine2D().translate(23, 42) + Affine2D().rotate_deg(90)
        # 断言两个变换矩阵的数值是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言单个点经过变换后的结果是否接近预期值 [-43, 24]
        assert_array_almost_equal(trans.transform(self.single_point), [-43, 24])
        # 断言多个点经过变换后的结果是否接近预期值
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[-44, 23], [-45, 26], [-42, 27]])

        # 创建一个平移矩阵，平移向量为 (23, 42)，围绕给定的中心点进行90度的旋转
        trans = Affine2D().translate(23, 42).rotate_deg_around(*self.pivot, 90)
        # 分别创建一个平移矩阵和一个围绕中心点旋转的矩阵，然后将它们相加
        trans_added = (Affine2D().translate(23, 42) +
                       Affine2D().rotate_deg_around(*self.pivot, 90))
        # 断言两个变换矩阵的数值是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言单个点经过变换后的结果是否接近预期值 [-41, 24]
        assert_array_almost_equal(trans.transform(self.single_point), [-41, 24])
        # 断言多个点经过变换后的结果是否接近预期值
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[-42, 23], [-43, 26], [-40, 27]])

        # 创建一个平移矩阵，平移向量为 (23, 42)，并且添加一个缩放矩阵，缩放因子为 (3, -2)
        trans = Affine2D().translate(23, 42).scale(3, -2)
        # 分别创建一个平移矩阵和一个缩放矩阵，然后将它们相加
        trans_added = Affine2D().translate(23, 42) + Affine2D().scale(3, -2)
        # 断言两个变换矩阵的数值是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言单个点经过变换后的结果是否接近预期值 [72, -86]
        assert_array_almost_equal(trans.transform(self.single_point), [72, -86])
        # 断言多个点经过变换后的结果是否接近预期值
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[69, -88], [78, -90], [81, -84]])

        # 创建一个平移矩阵，平移向量为 (23, 42)，并添加一个倾斜矩阵，倾斜角度为 26.5650512°和14.0362435°
        trans = (Affine2D().translate(23, 42)
                 .skew_deg(26.5650512, 14.0362435))  # ~atan(0.5), ~atan(0.25)
        # 分别创建一个平移矩阵和一个倾斜矩阵，然后将它们相加
        trans_added = (Affine2D().translate(23, 42) +
                       Affine2D().skew_deg(26.5650512, 14.0362435))
        # 断言两个变换矩阵的数值是否相等
        assert_array_equal(trans.get_matrix(), trans_added.get_matrix())
        # 断言单个点经过变换后的结果是否接近预期值 [45.5, 49]
        assert_array_almost_equal(trans.transform(self.single_point), [45.5, 49])
        # 断言多个点经过变换后的结果是否接近预期值
        assert_array_almost_equal(trans.transform(self.multiple_points),
                                  [[45, 49.75], [48.5, 51.5], [48, 48.75]])

    def test_invalid_transform(self):
        # 创建一个空的仿射变换对象
        t = mtransforms.Affine2D()
        # 使用 pytest 检查在变换时传入错误维度的数据是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            t.transform(1)
        with pytest.raises(ValueError):
            t.transform([[[1]]])
        # 使用 pytest 检查在变换时传入空数组是否会引发 RuntimeError 异常
        with pytest.raises(RuntimeError):
            t.transform([])
        with pytest.raises(RuntimeError):
            t.transform([1])
        # 使用 pytest 检查在变换时传入不合适形状的数组是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            t.transform([[1]])
        with pytest.raises(ValueError):
            t.transform([[1, 2, 3]])
    # 定义一个测试函数，用于测试Affine2D对象的复制行为
    def test_copy(self):
        # 创建两个Affine2D对象a和b
        a = mtransforms.Affine2D()
        b = mtransforms.Affine2D()
        # 将a和b进行相加得到新的Affine2D对象s
        s = a + b
        # 更新依赖项应该使得依赖项的复制无效化。
        s.get_matrix()  # 解析它。
        # 对s进行浅拷贝，得到s1
        s1 = copy.copy(s)
        # 断言s和s1都不是无效的
        assert not s._invalid and not s1._invalid
        # 对a进行平移操作
        a.translate(1, 2)
        # 断言s被标记为无效，而s1没有被标记为无效
        assert s._invalid and s1._invalid
        # 断言s1的矩阵与a的矩阵相等
        assert (s1.get_matrix() == a.get_matrix()).all()
        # 更新依赖项的拷贝不应该使得依赖项本身无效。
        s.get_matrix()  # 解析它。
        # 对b进行浅拷贝，得到b1
        b1 = copy.copy(b)
        # 对b1进行平移操作
        b1.translate(3, 4)
        # 断言s没有被标记为无效
        assert not s._invalid
        # 断言s的矩阵与a的矩阵相等
        assert_array_equal(s.get_matrix(), a.get_matrix())

    # 定义一个测试函数，用于测试Affine2D对象的深拷贝行为
    def test_deepcopy(self):
        # 创建两个Affine2D对象a和b
        a = mtransforms.Affine2D()
        b = mtransforms.Affine2D()
        # 将a和b进行相加得到新的Affine2D对象s
        s = a + b
        # 更新依赖项应该使得依赖项的深拷贝无效化。
        s.get_matrix()  # 解析它。
        # 对s进行深拷贝，得到s1
        s1 = copy.deepcopy(s)
        # 断言s和s1都不是无效的
        assert not s._invalid and not s1._invalid
        # 对a进行平移操作
        a.translate(1, 2)
        # 断言s被标记为无效，而s1没有被标记为无效
        assert s._invalid and not s1._invalid
        # 断言s1的矩阵与一个新的Affine2D对象的矩阵相等
        assert_array_equal(s1.get_matrix(), mtransforms.Affine2D().get_matrix())
        # 更新依赖项的深拷贝不应该使得依赖项本身无效。
        s.get_matrix()  # 解析它。
        # 对b进行深拷贝，得到b1
        b1 = copy.deepcopy(b)
        # 对b1进行平移操作
        b1.translate(3, 4)
        # 断言s没有被标记为无效
        assert not s._invalid
        # 断言s的矩阵与a的矩阵相等
        assert_array_equal(s.get_matrix(), a.get_matrix())
def test_non_affine_caching():
    # 定义一个自定义的非仿射变换类 AssertingNonAffineTransform
    class AssertingNonAffineTransform(mtransforms.Transform):
        """
        This transform raises an assertion error when called when it
        shouldn't be and ``self.raise_on_transform`` is True.

        """
        input_dims = output_dims = 2
        is_affine = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 初始化时设置 raise_on_transform 为 False
            self.raise_on_transform = False
            # 创建一个基础的仿射变换对象，并缩放为 (10, 10)
            self.underlying_transform = mtransforms.Affine2D().scale(10, 10)

        # 处理非仿射路径变换的方法
        def transform_path_non_affine(self, path):
            # 如果 raise_on_transform 为 True，则断言失败
            assert not self.raise_on_transform, \
                'Invalidated affine part of transform unnecessarily.'
            # 否则使用 underlying_transform 对象对路径进行变换
            return self.underlying_transform.transform_path(path)
        transform_path = transform_path_non_affine

        # 处理非仿射变换的方法
        def transform_non_affine(self, path):
            # 如果 raise_on_transform 为 True，则断言失败
            assert not self.raise_on_transform, \
                'Invalidated affine part of transform unnecessarily.'
            # 否则使用 underlying_transform 对象对数据进行变换
            return self.underlying_transform.transform(path)
        transform = transform_non_affine

    # 创建一个 AssertingNonAffineTransform 实例
    my_trans = AssertingNonAffineTransform()
    # 在当前图中创建一个坐标轴
    ax = plt.axes()
    # 绘制一个图形，应用自定义的非仿射变换 my_trans + ax.transData
    plt.plot(np.arange(10), transform=my_trans + ax.transData)
    # 绘制图形
    plt.draw()
    # 启用非仿射变换触发异常的功能
    my_trans.raise_on_transform = True
    # 使 ax.transAxes 失效
    ax.transAxes.invalidate()
    # 重新绘制图形
    plt.draw()


def test_external_transform_api():
    # 定义一个简单的比例变换类 ScaledBy
    class ScaledBy:
        def __init__(self, scale_factor):
            self._scale_factor = scale_factor

        # 将比例变换转换为 Matplotlib 的变换对象
        def _as_mpl_transform(self, axes):
            return (mtransforms.Affine2D().scale(self._scale_factor)
                    + axes.transData)

    # 创建一个坐标轴
    ax = plt.axes()
    # 绘制一条曲线，应用比例变换 ScaledBy(10)
    line, = plt.plot(np.arange(10), transform=ScaledBy(10))
    # 设置坐标轴的 x 和 y 范围
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    # 断言线条的顶层变换是比例变换
    assert_allclose(line.get_transform()._a.get_matrix(),
                    mtransforms.Affine2D().scale(10).get_matrix())


@image_comparison(['pre_transform_data'], remove_text=True, style='mpl20',
                  tol=0.05)
def test_pre_transform_plotting():
    # 处理数据预变换的绘图方法的测试
    # 注意：此处图表的轴范围非常重要，应为数据建议的 10 倍

    # 创建一个坐标轴
    ax = plt.axes()
    # 创建一个 10 倍缩放的仿射变换对象 times10
    times10 = mtransforms.Affine2D().scale(10)

    # 绘制等高线图，应用 times10 + ax.transData 的变换
    ax.contourf(np.arange(48).reshape(6, 8), transform=times10 + ax.transData)

    # 绘制伪彩色图，应用 times10 + ax.transData 的变换
    ax.pcolormesh(np.linspace(0, 4, 7),
                  np.linspace(5.5, 8, 9),
                  np.arange(48).reshape(8, 6),
                  transform=times10 + ax.transData)

    # 绘制散点图，应用 times10 + ax.transData 的变换
    ax.scatter(np.linspace(0, 10), np.linspace(10, 0),
               transform=times10 + ax.transData)

    # 生成一些数据
    x = np.linspace(8, 10, 20)
    y = np.linspace(1, 5, 20)
    u = 2*np.sin(x) + np.cos(y[:, np.newaxis])
    v = np.sin(x) - np.cos(y[:, np.newaxis])
    # 使用 streamplot 在指定的坐标系上绘制流线图
    ax.streamplot(x, y, u, v, transform=times10 + ax.transData,
                  linewidth=np.hypot(u, v))

    # 减少向量数据密度，以便用于 barb 和 quiver 绘图
    x, y = x[::3], y[::3]  # 每隔三个点取一个，减少密度
    u, v = u[::3, ::3], v[::3, ::3]  # 每隔三个点取一个，减少密度

    # 在指定的坐标系上使用 quiver 绘制箭头图
    ax.quiver(x, y + 5, u, v, transform=times10 + ax.transData)

    # 在指定的坐标系上使用 barbs 绘制风羽图
    ax.barbs(x - 3, y + 5, u**2, v**2, transform=times10 + ax.transData)
def test_contour_pre_transform_limits():
    # 创建一个新的绘图对象
    ax = plt.axes()
    # 生成坐标网格
    xs, ys = np.meshgrid(np.linspace(15, 20, 15), np.linspace(12.4, 12.5, 20))
    # 绘制等高线图，并应用仿射变换到数据坐标系
    ax.contourf(xs, ys, np.log(xs * ys),
                transform=mtransforms.Affine2D().scale(0.1) + ax.transData)

    # 设置预期结果的二维数组
    expected = np.array([[1.5, 1.24],
                         [2., 1.25]])
    # 断言实际数据限制的点与预期是否接近
    assert_almost_equal(expected, ax.dataLim.get_points())


def test_pcolor_pre_transform_limits():
    # 基于 test_contour_pre_transform_limits() 编写
    ax = plt.axes()
    # 生成坐标网格
    xs, ys = np.meshgrid(np.linspace(15, 20, 15), np.linspace(12.4, 12.5, 20))
    # 绘制伪彩色图，并应用仿射变换到数据坐标系
    ax.pcolor(xs, ys, np.log(xs * ys)[:-1, :-1],
              transform=mtransforms.Affine2D().scale(0.1) + ax.transData)

    # 设置预期结果的二维数组
    expected = np.array([[1.5, 1.24],
                         [2., 1.25]])
    # 断言实际数据限制的点与预期是否接近
    assert_almost_equal(expected, ax.dataLim.get_points())


def test_pcolormesh_pre_transform_limits():
    # 基于 test_contour_pre_transform_limits() 编写
    ax = plt.axes()
    # 生成坐标网格
    xs, ys = np.meshgrid(np.linspace(15, 20, 15), np.linspace(12.4, 12.5, 20))
    # 绘制伪彩色网格，并应用仿射变换到数据坐标系
    ax.pcolormesh(xs, ys, np.log(xs * ys)[:-1, :-1],
                  transform=mtransforms.Affine2D().scale(0.1) + ax.transData)

    # 设置预期结果的二维数组
    expected = np.array([[1.5, 1.24],
                         [2., 1.25]])
    # 断言实际数据限制的点与预期是否接近
    assert_almost_equal(expected, ax.dataLim.get_points())


def test_pcolormesh_gouraud_nans():
    np.random.seed(19680801)

    # 准备数据
    values = np.linspace(0, 180, 3)
    radii = np.linspace(100, 1000, 10)
    z, y = np.meshgrid(values, radii)
    x = np.radians(np.random.rand(*z.shape) * 100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    # 设置极坐标的 r 范围，导致一些值为 NaN，但不应使程序崩溃，应该被其他路径操作忽略
    ax.set_rlim(101, 1000)
    # 绘制 gouraud 阴影的伪彩色网格
    ax.pcolormesh(x, y, z, shading="gouraud")

    # 刷新绘图区域
    fig.canvas.draw()


def test_Affine2D_from_values():
    # 准备测试点数据
    points = np.array([[0, 0],
                       [10, 20],
                       [-1, 0],
                       ])

    # 创建仿射变换对象，通过指定的值初始化
    t = mtransforms.Affine2D.from_values(1, 0, 0, 0, 0, 0)
    # 对点数据进行仿射变换
    actual = t.transform(points)
    # 设置预期结果的二维数组
    expected = np.array([[0, 0], [10, 0], [-1, 0]])
    # 断言仿射变换后的实际结果与预期是否接近
    assert_almost_equal(actual, expected)

    # 以不同的值初始化仿射变换对象，重复上述步骤进行断言
    t = mtransforms.Affine2D.from_values(0, 2, 0, 0, 0, 0)
    actual = t.transform(points)
    expected = np.array([[0, 0], [0, 20], [0, -2]])
    assert_almost_equal(actual, expected)

    t = mtransforms.Affine2D.from_values(0, 0, 3, 0, 0, 0)
    actual = t.transform(points)
    expected = np.array([[0, 0], [60, 0], [0, 0]])
    assert_almost_equal(actual, expected)

    t = mtransforms.Affine2D.from_values(0, 0, 0, 4, 0, 0)
    actual = t.transform(points)
    expected = np.array([[0, 0], [0, 80], [0, 0]])
    assert_almost_equal(actual, expected)

    t = mtransforms.Affine2D.from_values(0, 0, 0, 0, 5, 0)
    actual = t.transform(points)
    expected = np.array([[5, 0], [5, 0], [5, 0]])
    assert_almost_equal(actual, expected)
    # 创建一个 Affine2D 对象 t，使用指定的值构建仿射变换矩阵
    t = mtransforms.Affine2D.from_values(0, 0, 0, 0, 0, 6)
    # 使用 t 对 points 进行仿射变换，得到变换后的坐标 actual
    actual = t.transform(points)
    # 创建一个预期的 NumPy 数组 expected，表示仿射变换后的预期结果
    expected = np.array([[0, 6], [0, 6], [0, 6]])
    # 断言实际得到的变换结果 actual 与预期的 expected 数组非常接近
    assert_almost_equal(actual, expected)
def test_affine_inverted_invalidated():
    # 确保在访问时，仿射变换在反转后不被声明为有效
    point = [1.0, 1.0]
    t = mtransforms.Affine2D()  # 创建一个空的仿射变换对象

    # 断言：反转后再变换的结果与原始点坐标几乎相等
    assert_almost_equal(point, t.transform(t.inverted().transform(point)))

    # 改变并访问仿射变换
    t.translate(1.0, 1.0).get_matrix()  # 平移变换并获取变换矩阵
    assert_almost_equal(point, t.transform(t.inverted().transform(point)))


def test_clipping_of_log():
    # issue 804
    # 创建一个闭合路径对象，模拟在绘制对数直方图时的情况
    path = Path._create_closed([(0.2, -99), (0.4, -99), (0.4, 20), (0.2, 20)])

    # 创建一个混合泛用变换对象，结合仿射变换和对数变换
    trans = mtransforms.BlendedGenericTransform(
        mtransforms.Affine2D(), scale.LogTransform(10, 'clip'))

    # 对非仿射路径进行变换
    tpath = trans.transform_path_non_affine(path)

    # 迭代处理变换后的路径的各段，应用仿射变换矩阵，并进行裁剪
    result = tpath.iter_segments(trans.get_affine(),
                                 clip=(0, 0, 100, 100),
                                 simplify=False)
    tpoints, tcodes = zip(*result)

    # 断言：迭代后的路径代码与原始路径代码几乎相等，表示路径不再闭合
    assert_allclose(tcodes, path.codes[:-1])


class NonAffineForTest(mtransforms.Transform):
    """
    一个类，看起来像非仿射变换，但实际上执行给定变换的行为（即使是仿射变换）。
    用于测试非仿射行为与简单仿射变换的情况。
    """
    is_affine = False
    output_dims = 2
    input_dims = 2

    def __init__(self, real_trans, *args, **kwargs):
        self.real_trans = real_trans
        super().__init__(*args, **kwargs)

    def transform_non_affine(self, values):
        return self.real_trans.transform(values)

    def transform_path_non_affine(self, path):
        return self.real_trans.transform_path(path)


class TestBasicTransform:
    def setup_method(self):
        # 初始化各种仿射变换和非仿射变换对象

        self.ta1 = mtransforms.Affine2D(shorthand_name='ta1').rotate(np.pi / 2)
        self.ta2 = mtransforms.Affine2D(shorthand_name='ta2').translate(10, 0)
        self.ta3 = mtransforms.Affine2D(shorthand_name='ta3').scale(1, 2)

        self.tn1 = NonAffineForTest(mtransforms.Affine2D().translate(1, 2),
                                    shorthand_name='tn1')
        self.tn2 = NonAffineForTest(mtransforms.Affine2D().translate(1, 2),
                                    shorthand_name='tn2')
        self.tn3 = NonAffineForTest(mtransforms.Affine2D().translate(1, 2),
                                    shorthand_name='tn3')

        # 创建仿射变换堆栈，看起来像 ((A, (N, A)), A)
        self.stack1 = (self.ta1 + (self.tn1 + self.ta2)) + self.ta3
        # 创建仿射变换堆栈，看起来像 (((A, N), A), A)
        self.stack2 = self.ta1 + self.tn1 + self.ta2 + self.ta3
        # 创建仿射变换堆栈的子集
        self.stack2_subset = self.tn1 + self.ta2 + self.ta3

        # 在调试时，可以生成变换堆栈的 dot 图像:
#        self.stack1.write_graphviz(file('stack1.dot', 'w'))
#        self.stack2.write_graphviz(file('stack2.dot', 'w'))
    def test_transform_depth(self):
        # 断言 stack1 对象的深度为 4
        assert self.stack1.depth == 4
        # 断言 stack2 对象的深度为 4
        assert self.stack2.depth == 4
        # 断言 stack2_subset 对象的深度为 3
        assert self.stack2_subset.depth == 3

    def test_left_to_right_iteration(self):
        # 构建嵌套的表达式 stack3，并写入 Graphviz 文件 'stack3.dot'
#        stack3.write_graphviz(file('stack3.dot', 'w'))

        # 定义目标变换序列 target_transforms
        stack3 = (self.ta1 + (self.tn1 + (self.ta2 + self.tn2))) + self.ta3
        target_transforms = [stack3,
                             (self.tn1 + (self.ta2 + self.tn2)) + self.ta3,
                             (self.ta2 + self.tn2) + self.ta3,
                             self.tn2 + self.ta3,
                             self.ta3,
                             ]
        
        # 对 stack3 进行从左到右的迭代断点测试，并验证结果与目标变换序列一致
        r = [rh for _, rh in stack3._iter_break_from_left_to_right()]
        assert len(r) == len(target_transforms)

        for target_stack, stack in zip(target_transforms, r):
            assert target_stack == stack

    def test_transform_shortcuts(self):
        # 测试简化变换操作，验证 stack1 - stack2_subset 应等于 ta1
        assert self.stack1 - self.stack2_subset == self.ta1
        # 测试简化变换操作，验证 stack2 - stack2_subset 应等于 ta1
        assert self.stack2 - self.stack2_subset == self.ta1

        # 测试反向简化变换操作，验证 stack2_subset - stack2 应等于 ta1 的反向变换
        assert self.stack2_subset - self.stack2 == self.ta1.inverted()
        # 断言 stack2_subset - stack2 的深度为 1
        assert (self.stack2_subset - self.stack2).depth == 1

        # 预期抛出 ValueError 异常，因为 stack1 - stack2 无效
        with pytest.raises(ValueError):
            self.stack1 - self.stack2

        # 定义 aff1 和 aff2 为复合变换
        aff1 = self.ta1 + (self.ta2 + self.ta3)
        aff2 = self.ta2 + self.ta3

        # 测试简化变换操作，验证 aff1 - aff2 应等于 ta1
        assert aff1 - aff2 == self.ta1
        # 测试反向简化变换操作，验证 aff1 - ta2 应等于 aff1 + ta2 的反向变换
        assert aff1 - self.ta2 == aff1 + self.ta2.inverted()

        # 测试简化变换操作，验证 stack1 - ta3 应等于 ta1 + (tn1 + ta2)
        assert self.stack1 - self.ta3 == self.ta1 + (self.tn1 + self.ta2)
        # 测试简化变换操作，验证 stack2 - ta3 应等于 ta1 + tn1 + ta2
        assert self.stack2 - self.ta3 == self.ta1 + self.tn1 + self.ta2

        # 测试复合变换是否包含分支
        assert ((self.ta2 + self.ta3) - self.ta3 + self.ta3 ==
                self.ta2 + self.ta3)

    def test_contains_branch(self):
        # 定义 r1 和 r2 为相同的复合变换
        r1 = (self.ta2 + self.ta1)
        r2 = (self.ta2 + self.ta1)

        # 断言 r1 等于 r2
        assert r1 == r2
        # 断言 r1 不等于 ta1
        assert r1 != self.ta1
        # 断言 r1 包含分支 r2
        assert r1.contains_branch(r2)
        # 断言 r1 包含分支 ta1
        assert r1.contains_branch(self.ta1)
        # 断言 r1 不包含分支 ta2
        assert not r1.contains_branch(self.ta2)
        # 断言 r1 不包含分支 ta2 + ta2
        assert not r1.contains_branch(self.ta2 + self.ta2)

        # 断言 r1 等于 r2
        assert r1 == r2

        # 断言 stack1 包含分支 ta3
        assert self.stack1.contains_branch(self.ta3)
        # 断言 stack2 包含分支 ta3
        assert self.stack2.contains_branch(self.ta3)

        # 断言 stack1 包含分支 stack2_subset
        assert self.stack1.contains_branch(self.stack2_subset)
        # 断言 stack2 包含分支 stack2_subset
        assert self.stack2.contains_branch(self.stack2_subset)

        # 断言 stack2_subset 不包含分支 stack1
        assert not self.stack2_subset.contains_branch(self.stack1)
        # 断言 stack2_subset 不包含分支 stack2
        assert not self.stack2_subset.contains_branch(self.stack2)

        # 断言 stack1 包含分支 ta2 + ta3
        assert self.stack1.contains_branch(self.ta2 + self.ta3)
        # 断言 stack2 包含分支 ta2 + ta3
        assert self.stack2.contains_branch(self.ta2 + self.ta3)

        # 断言 stack1 不包含分支 tn1 + ta2
        assert not self.stack1.contains_branch(self.tn1 + self.ta2)
    def test_affine_simplification(self):
        # 测试仿射简化，确保转换堆栈仅调用绝对必要的部分，允许对复杂转换堆栈进行最佳优化。
        points = np.array([[0, 0], [10, 20], [np.nan, 1], [-1, 0]],
                          dtype=np.float64)
        # 使用 stack1 的非仿射变换方法处理点集，得到非仿射变换后的点
        na_pts = self.stack1.transform_non_affine(points)
        # 使用 stack1 的完整变换方法处理点集，得到完整变换后的点
        all_pts = self.stack1.transform(points)

        na_expected = np.array([[1., 2.], [-19., 12.],
                                [np.nan, np.nan], [1., 1.]], dtype=np.float64)
        all_expected = np.array([[11., 4.], [-9., 24.],
                                 [np.nan, np.nan], [11., 2.]],
                                dtype=np.float64)

        # 检查使用非仿射部分变换后的点是否与预期结果相近
        assert_array_almost_equal(na_pts, na_expected)
        # 检查使用完整变换后的点是否与预期结果相近
        assert_array_almost_equal(all_pts, all_expected)
        # 检查使用两步变换后的点是否与预期结果相近
        assert_array_almost_equal(self.stack1.transform_affine(na_pts),
                                  all_expected)
        # 检查先获取仿射变换，然后完全应用该变换是否与之前的结果相同
        assert_array_almost_equal(self.stack1.get_affine().transform(na_pts),
                                  all_expected)

        # 检查 stack1 和 stack2 的仿射部分是否等效（即优化是否生效）
        expected_result = (self.ta2 + self.ta3).get_matrix()
        result = self.stack1.get_affine().get_matrix()
        assert_array_equal(expected_result, result)

        result = self.stack2.get_affine().get_matrix()
        assert_array_equal(expected_result, result)
class TestTransformPlotInterface:
    # 测试线段在坐标轴坐标系中的范围
    def test_line_extent_axes_coords(self):
        # 创建一个新的坐标轴对象
        ax = plt.axes()
        # 在坐标轴坐标系中绘制简单的线段
        ax.plot([0.1, 1.2, 0.8], [0.9, 0.5, 0.8], transform=ax.transAxes)
        # 断言数据限制点是否满足预期
        assert_array_equal(ax.dataLim.get_points(),
                           np.array([[np.inf, np.inf],
                                     [-np.inf, -np.inf]]))

    # 测试线段在数据坐标系中的范围
    def test_line_extent_data_coords(self):
        # 创建一个新的坐标轴对象
        ax = plt.axes()
        # 在数据坐标系中绘制简单的线段
        ax.plot([0.1, 1.2, 0.8], [0.9, 0.5, 0.8], transform=ax.transData)
        # 断言数据限制点是否满足预期
        assert_array_equal(ax.dataLim.get_points(),
                           np.array([[0.1,  0.5], [1.2,  0.9]]))

    # 测试线段在复合坐标系中的范围（x轴为坐标轴坐标系，y轴为数据坐标系）
    def test_line_extent_compound_coords1(self):
        # 创建一个新的坐标轴对象
        ax = plt.axes()
        # 创建复合变换，结合坐标轴坐标系和数据坐标系
        trans = mtransforms.blended_transform_factory(ax.transAxes,
                                                      ax.transData)
        # 在复合坐标系中绘制简单的线段
        ax.plot([0.1, 1.2, 0.8], [35, -5, 18], transform=trans)
        # 断言数据限制点是否满足预期
        assert_array_equal(ax.dataLim.get_points(),
                           np.array([[np.inf, -5.],
                                     [-np.inf, 35.]]))

    # 测试线段在预数据变换坐标系中的范围
    def test_line_extent_predata_transform_coords(self):
        # 创建一个新的坐标轴对象
        ax = plt.axes()
        # 创建预数据变换，结合偏移和数据坐标系
        trans = mtransforms.Affine2D().scale(10) + ax.transData
        # 在预数据变换坐标系中绘制简单的线段
        ax.plot([0.1, 1.2, 0.8], [35, -5, 18], transform=trans)
        # 断言数据限制点是否满足预期
        assert_array_equal(ax.dataLim.get_points(),
                           np.array([[1., -50.], [12., 350.]]))

    # 测试线段在复合坐标系中的范围（x轴为坐标轴坐标系，y轴为预数据变换坐标系）
    def test_line_extent_compound_coords2(self):
        # 创建一个新的坐标轴对象
        ax = plt.axes()
        # 创建复合变换，结合坐标轴坐标系和预数据变换坐标系
        trans = mtransforms.blended_transform_factory(
            ax.transAxes, mtransforms.Affine2D().scale(10) + ax.transData)
        # 在复合坐标系中绘制简单的线段
        ax.plot([0.1, 1.2, 0.8], [35, -5, 18], transform=trans)
        # 断言数据限制点是否满足预期
        assert_array_equal(ax.dataLim.get_points(),
                           np.array([[np.inf, -50.], [-np.inf, 350.]]))

    # 测试线段在仿射坐标系中的范围
    def test_line_extents_affine(self):
        # 创建一个新的坐标轴对象
        ax = plt.axes()
        # 创建仿射变换，结合偏移和数据坐标系
        offset = mtransforms.Affine2D().translate(10, 10)
        # 在仿射坐标系中绘制简单的线段
        plt.plot(np.arange(10), transform=offset + ax.transData)
        # 预期的数据限制点
        expected_data_lim = np.array([[0., 0.], [9.,  9.]]) + 10
        # 断言数据限制点是否几乎等于预期
        assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)

    # 测试线段在非仿射坐标系中的范围
    def test_line_extents_non_affine(self):
        # 创建一个新的坐标轴对象
        ax = plt.axes()
        # 创建仿射变换，结合偏移和数据坐标系
        offset = mtransforms.Affine2D().translate(10, 10)
        # 创建用于测试的非仿射变换
        na_offset = NonAffineForTest(mtransforms.Affine2D().translate(10, 10))
        # 在非仿射坐标系中绘制简单的线段
        plt.plot(np.arange(10), transform=offset + na_offset + ax.transData)
        # 预期的数据限制点
        expected_data_lim = np.array([[0., 0.], [9.,  9.]]) + 20
        # 断言数据限制点是否几乎等于预期
        assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)
    def test_pathc_extents_non_affine(self):
        # 获取当前绘图区域对象
        ax = plt.axes()
        # 创建一个平移变换对象，偏移量为 (10, 10)
        offset = mtransforms.Affine2D().translate(10, 10)
        # 创建一个自定义非仿射变换对象，偏移量也为 (10, 10)
        na_offset = NonAffineForTest(mtransforms.Affine2D().translate(10, 10))
        # 创建一个简单路径对象，定义一个矩形路径
        pth = Path([[0, 0], [0, 10], [10, 10], [10, 0]])
        # 创建一个路径补丁对象，应用变换：offset + na_offset + ax.transData
        patch = mpatches.PathPatch(pth,
                                   transform=offset + na_offset + ax.transData)
        # 将路径补丁对象添加到绘图区域中
        ax.add_patch(patch)
        # 期望的数据边界，对当前的数据限制点进行偏移
        expected_data_lim = np.array([[0., 0.], [10.,  10.]]) + 20
        # 断言当前绘图区域的数据限制与期望的数据限制接近
        assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)

    def test_pathc_extents_affine(self):
        # 获取当前绘图区域对象
        ax = plt.axes()
        # 创建一个平移变换对象，偏移量为 (10, 10)
        offset = mtransforms.Affine2D().translate(10, 10)
        # 创建一个简单路径对象，定义一个矩形路径
        pth = Path([[0, 0], [0, 10], [10, 10], [10, 0]])
        # 创建一个路径补丁对象，应用变换：offset + ax.transData
        patch = mpatches.PathPatch(pth, transform=offset + ax.transData)
        # 将路径补丁对象添加到绘图区域中
        ax.add_patch(patch)
        # 期望的数据边界，对当前的数据限制点进行偏移
        expected_data_lim = np.array([[0., 0.], [10.,  10.]]) + 10
        # 断言当前绘图区域的数据限制与期望的数据限制接近
        assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)

    def test_line_extents_for_non_affine_transData(self):
        # 获取极坐标系的绘图区域对象
        ax = plt.axes(projection='polar')
        # 将数据的半径增加 10
        offset = mtransforms.Affine2D().translate(0, 10)

        # 绘制简单的线图，应用变换：offset + ax.transData
        plt.plot(np.arange(10), transform=offset + ax.transData)
        # 极坐标绘图的数据限制以极坐标系原始坐标为准
        # 因此，数据限制不反映在实际绘图中所显示的内容。
        expected_data_lim = np.array([[0., 0.], [9.,  9.]]) + [0, 10]
        # 断言当前绘图区域的数据限制与期望的数据限制接近
        assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)
# 检查两个边界框对象的边界是否相等，如果不相等则抛出异常
def assert_bbox_eq(bbox1, bbox2):
    assert_array_equal(bbox1.bounds, bbox2.bounds)


# 测试边界框对象的冻结副本功能是否正确复制了最小位置信息
def test_bbox_frozen_copies_minpos():
    # 创建一个新的边界框对象，设置最小位置参数为1.0
    bbox = mtransforms.Bbox.from_extents(0.0, 0.0, 1.0, 1.0, minpos=1.0)
    # 对边界框对象进行冻结，返回冻结后的副本
    frozen = bbox.frozen()
    # 检查冻结后的边界框的最小位置信息与原始边界框的最小位置信息是否相等
    assert_array_equal(frozen.minpos, bbox.minpos)


# 测试边界框对象的交集计算功能
def test_bbox_intersection():
    bbox_from_ext = mtransforms.Bbox.from_extents
    inter = mtransforms.Bbox.intersection

    r1 = bbox_from_ext(0, 0, 1, 1)
    r2 = bbox_from_ext(0.5, 0.5, 1.5, 1.5)
    r3 = bbox_from_ext(0.5, 0, 0.75, 0.75)
    r4 = bbox_from_ext(0.5, 1.5, 1, 2.5)
    r5 = bbox_from_ext(1, 1, 2, 2)

    # 自身交集 -> 返回自身
    assert_bbox_eq(inter(r1, r1), r1)
    # 简单交集计算
    assert_bbox_eq(inter(r1, r2), bbox_from_ext(0.5, 0.5, 1, 1))
    # r3 包含于 r1 内部
    assert_bbox_eq(inter(r1, r3), r3)
    # 无交集情况
    assert inter(r1, r4) is None
    # 单点交集
    assert_bbox_eq(inter(r1, r5), bbox_from_ext(1, 1, 1, 1))


# 测试边界框对象作为字符串表达时的正确性
def test_bbox_as_strings():
    # 创建一个新的边界框对象，定义其边界坐标
    b = mtransforms.Bbox([[.5, 0], [.75, .75]])
    # 检查边界框对象转换为字符串后，再转回对象的正确性
    assert_bbox_eq(b, eval(repr(b), {'Bbox': mtransforms.Bbox}))
    # 将边界框对象转换为字典格式，并逐一验证每个键值对的正确性
    asdict = eval(str(b), {'Bbox': dict})
    for k, v in asdict.items():
        assert getattr(b, k) == v
    # 定义格式化输出的格式
    fmt = '.1f'
    # 格式化输出边界框对象，并逐一验证每个键值对的正确性
    asdict = eval(format(b, fmt), {'Bbox': dict})
    for k, v in asdict.items():
        assert eval(format(getattr(b, k), fmt)) == v


# 测试字符串表示形式的变换功能
def test_str_transform():
    # 在极坐标投影中创建子图对象，并将其数据转换为字符串后进行验证
    assert str(plt.subplot(projection="polar").transData) == """\
CompositeGenericTransform(
    CompositeGenericTransform(
        CompositeGenericTransform(
            TransformWrapper(
                BlendedAffine2D(
                    IdentityTransform(),
                    IdentityTransform())),
            CompositeAffine2D(
                Affine2D().scale(1.0),
                Affine2D().scale(1.0))),
        PolarTransform(
            PolarAxes(0.125,0.1;0.775x0.8),
            use_rmin=True,
            apply_theta_transforms=False)),
    # 创建一个复合的通用变换对象，内部包含两个通用变换对象的组合
    CompositeGenericTransform(
        # 第一个通用变换对象，基于极坐标仿射变换
        CompositeGenericTransform(
            # 极坐标仿射变换对象，包装在变换封装器中的混合仿射变换对象
            PolarAffine(
                TransformWrapper(
                    BlendedAffine2D(
                        IdentityTransform(),  # 使用恒等仿射变换初始化
                        IdentityTransform()  # 使用恒等仿射变换初始化
                    )
                ),
                LockableBbox(
                    Bbox(x0=0.0, y0=0.0, x1=6.283185307179586, y1=1.0),  # 设置锁定的边界框
                    [[-- --]  # 二维数组，未提供具体数值
                     [-- --]]  # 二维数组，未提供具体数值
                )
            ),
            # Bbox 变换，从 _WedgeBbox 创建
            BboxTransformFrom(
                _WedgeBbox(
                    (0.5, 0.5),  # 设置中心点坐标
                    TransformedBbox(
                        Bbox(x0=0.0, y0=0.0, x1=6.283185307179586, y1=1.0),  # 初始边界框
                        CompositeAffine2D(
                            Affine2D().scale(1.0),  # 第一个仿射变换对象，进行 1.0 缩放
                            Affine2D().scale(1.0)   # 第二个仿射变换对象，进行 1.0 缩放
                        )
                    ),
                    LockableBbox(
                        Bbox(x0=0.0, y0=0.0, x1=6.283185307179586, y1=1.0),  # 设置锁定的边界框
                        [[-- --]  # 二维数组，未提供具体数值
                         [-- --]]  # 二维数组，未提供具体数值
                    )
                )
            )
        ),
        # Bbox 变换，到 TransformedBbox
        BboxTransformTo(
            TransformedBbox(
                Bbox(x0=0.125, y0=0.09999999999999998, x1=0.9, y1=0.9),  # 设置变换后的边界框
                BboxTransformTo(
                    TransformedBbox(
                        Bbox(x0=0.0, y0=0.0, x1=8.0, y1=6.0),  # 初始边界框
                        Affine2D().scale(80.0)  # 仿射变换对象，进行 80.0 缩放
                    )
                )
            )
        )
    )
def test_transform_single_point():
    t = mtransforms.Affine2D()
    r = t.transform_affine((1, 1))
    assert r.shape == (2,)


# 创建一个空的仿射变换对象
t = mtransforms.Affine2D()

# 对仿射变换对象进行仿射变换，将点 (1, 1) 转换为另一个坐标系中的点
r = t.transform_affine((1, 1))

# 检查变换后的点的形状是否为 (2,)，即二维坐标点
assert r.shape == (2,)



def test_log_transform():
    # Tests that the last line runs without exception (previously the
    # transform would fail if one of the axes was logarithmic).
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.transData.transform((1, 1))


# 测试对数变换功能，确保最后一行可以成功执行（先前如果其中一个坐标轴为对数尺度，则变换会失败）。

# 创建一个图形和坐标轴对象
fig, ax = plt.subplots()

# 设置坐标轴的 y 轴为对数尺度
ax.set_yscale('log')

# 对数据坐标系中的点 (1, 1) 进行坐标变换
ax.transData.transform((1, 1))



def test_nan_overlap():
    a = mtransforms.Bbox([[0, 0], [1, 1]])
    b = mtransforms.Bbox([[0, 0], [1, np.nan]])
    assert not a.overlaps(b)


# 测试两个边界框之间是否存在重叠，确保不会重叠。

# 创建两个边界框对象 a 和 b
a = mtransforms.Bbox([[0, 0], [1, 1]])
b = mtransforms.Bbox([[0, 0], [1, np.nan]])

# 检查边界框 a 和 b 是否重叠，预期结果是不重叠
assert not a.overlaps(b)



def test_transform_angles():
    t = mtransforms.Affine2D()  # Identity transform
    angles = np.array([20, 45, 60])
    points = np.array([[0, 0], [1, 1], [2, 2]])

    # Identity transform does not change angles
    new_angles = t.transform_angles(angles, points)
    assert_array_almost_equal(angles, new_angles)

    # points missing a 2nd dimension
    with pytest.raises(ValueError):
        t.transform_angles(angles, points[0:2, 0:1])

    # Number of angles != Number of points
    with pytest.raises(ValueError):
        t.transform_angles(angles, points[0:2, :])


# 测试角度变换功能

# 创建一个空的仿射变换对象（单位变换）
t = mtransforms.Affine2D()

# 创建角度数组和点数组
angles = np.array([20, 45, 60])
points = np.array([[0, 0], [1, 1], [2, 2]])

# 对于单位变换，角度不会改变
new_angles = t.transform_angles(angles, points)
assert_array_almost_equal(angles, new_angles)

# 当点数组缺少第二个维度时，应引发 ValueError
with pytest.raises(ValueError):
    t.transform_angles(angles, points[0:2, 0:1])

# 当角度数不等于点数时，应引发 ValueError
with pytest.raises(ValueError):
    t.transform_angles(angles, points[0:2, :])



def test_nonsingular():
    # test for zero-expansion type cases; other cases may be added later
    zero_expansion = np.array([-0.001, 0.001])
    cases = [(0, np.nan), (0, 0), (0, 7.9e-317)]
    for args in cases:
        out = np.array(mtransforms.nonsingular(*args))
        assert_array_equal(out, zero_expansion)


# 测试非奇异性功能，针对零扩展类型的情况进行测试；以后可能会添加其他情况。

# 定义零扩展数组
zero_expansion = np.array([-0.001, 0.001])

# 定义测试用例列表
cases = [(0, np.nan), (0, 0), (0, 7.9e-317)]

# 对每个测试用例进行测试
for args in cases:
    # 调用非奇异性函数，并将结果转换为数组
    out = np.array(mtransforms.nonsingular(*args))
    
    # 断言输出与预期的零扩展数组相等
    assert_array_equal(out, zero_expansion)



def test_transformed_path():
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    path = Path(points, closed=True)

    trans = mtransforms.Affine2D()
    trans_path = mtransforms.TransformedPath(path, trans)
    assert_allclose(trans_path.get_fully_transformed_path().vertices, points)

    # Changing the transform should change the result.
    r2 = 1 / np.sqrt(2)
    trans.rotate(np.pi / 4)
    assert_allclose(trans_path.get_fully_transformed_path().vertices,
                    [(0, 0), (r2, r2), (0, 2 * r2), (-r2, r2)],
                    atol=1e-15)

    # Changing the path does not change the result (it's cached).
    path.points = [(0, 0)] * 4
    assert_allclose(trans_path.get_fully_transformed_path().vertices,
                    [(0, 0), (r2, r2), (0, 2 * r2), (-r2, r2)],
                    atol=1e-15)


# 测试路径变换功能

# 定义点列表和路径对象
points = [(0, 0), (1, 0), (1, 1), (0, 1)]
path = Path(points, closed=True)

# 创建一个仿射变换对象
trans = mtransforms.Affine2D()

# 创建一个经过仿射变换的路径对象
trans_path = mtransforms.TransformedPath(path, trans)

# 断言完全变换后的路径顶点与原始点列表相等
assert_allclose(trans_path.get_fully_transformed_path().vertices, points)

# 改变仿射变换应该改变结果
r2 = 1 / np.sqrt(2)
trans.rotate(np.pi / 4)
assert_allclose(trans_path.get_fully_transformed_path().vertices,
                [(0, 0), (r2, r2), (0, 2 * r2), (-r2, r2)],
                atol=1e-15)

# 改变路径不应改变结果（结果被缓存）
path.points = [(0, 0)] * 4
assert_allclose(trans_path.get_fully_transformed_path().vertices,
                [(0, 0), (r2, r2), (0, 2 * r2), (-r2, r2)],
                atol=1e-15)



def test_transformed_patch_path():
    trans = mtransforms.Affine2D()
    patch = mpatches.Wedge((0, 0), 1, 45, 135, transform=trans)

    tpatch = mtransforms.TransformedPatchPath(patch)
    points = tpatch.get_fully_transformed_path().vertices

    # Changing the transform should change the result.
    trans.scale(2)
    assert_allclose(tpatch.get_fully_transformed_path().vertices, points * 2)

    # Changing the path should change the result (and cancel out the scaling
    # from the transform).
    patch.set_radius(0.5)
    assert_allclose(tpatch.get_fully_transformed_path().vertices, points)


# 测试变换后的路径图形功能

# 创建一个仿射变换对象
trans = mtransforms.Affine2D()

# 创建一个扇形路径对象，并应用仿射变换
patch = mpatches.Wedge((0, 0), 1, 45, 135, transform=trans)

# 创建一个经过变换的路径对象
tpatch = mtransforms.TransformedPatchPath(patch)

# 获取完全变换后的路径顶点
points = tpatch.get_fully_transformed_path().vertices

# 改变仿射变换应该改变结果
trans.scale(2)
assert_allclose(tpatch.get_fully_transformed_path().vertices, points
    # 定义一个包含除了被锁定元素以外的其他元素列表
    other_elements = ['x0', 'y0', 'x1', 'y1']
    # 从列表中移除当前被锁定的元素
    other_elements.remove(locked_element)

    # 创建一个单位的 Bbox 对象
    orig = mtransforms.Bbox.unit()
    # 使用 LockableBbox 类包装原始的 Bbox 对象，并指定被锁定的元素及其值为 2
    locked = mtransforms.LockableBbox(orig, **{locked_element: 2})

    # 断言锁定的元素在 LockableBbox 对象中的值为 2
    assert getattr(locked, locked_element) == 2
    # 断言 LockableBbox 对象中锁定的元素属性值为 2
    assert getattr(locked, 'locked_' + locked_element) == 2
    # 遍历其他元素列表，断言它们在 LockableBbox 对象中的值与原始 Bbox 对象中对应元素的值相同
    for elem in other_elements:
        assert getattr(locked, elem) == getattr(orig, elem)

    # 修改原始 Bbox 对象的点坐标，验证 LockableBbox 对象中被锁定元素的值不受影响
    orig.set_points(orig.get_points() + 10)
    assert getattr(locked, locked_element) == 2
    assert getattr(locked, 'locked_' + locked_element) == 2
    # 再次验证其他元素在 LockableBbox 对象中的值与原始 Bbox 对象中相应元素的值相同
    for elem in other_elements:
        assert getattr(locked, elem) == getattr(orig, elem)

    # 解锁被锁定的元素，验证 LockableBbox 对象中被锁定元素的值恢复为原始 Bbox 对象中的值
    setattr(locked, 'locked_' + locked_element, None)
    assert getattr(locked, 'locked_' + locked_element) is None
    # 断言原始 Bbox 对象的点坐标与 LockableBbox 对象的点坐标完全一致
    assert np.all(orig.get_points() == locked.get_points())

    # 重新锁定一个元素，验证锁定的元素值在 LockableBbox 对象中被正确修改，但其他元素不受影响
    setattr(locked, 'locked_' + locked_element, 3)
    assert getattr(locked, locked_element) == 3
    assert getattr(locked, 'locked_' + locked_element) == 3
    for elem in other_elements:
        assert getattr(locked, elem) == getattr(orig, elem)
# 定义测试函数 test_transformwrapper
def test_transformwrapper():
    # 创建 Affine2D 变换对象的包装器 TransformWrapper
    t = mtransforms.TransformWrapper(mtransforms.Affine2D())
    # 使用 pytest 检查是否会抛出 ValueError 异常，并验证异常信息
    with pytest.raises(ValueError, match=(
            r"The input and output dims of the new child \(1, 1\) "
            r"do not match those of current child \(2, 2\)")):
        # 设置变换对象为 LogTransform，并预期抛出异常
        t.set(scale.LogTransform(10))


# 使用装饰器 check_figures_equal 包装测试函数 test_scale_swapping
@check_figures_equal(extensions=["png"])
def test_scale_swapping(fig_test, fig_ref):
    # 设置随机种子
    np.random.seed(19680801)
    # 生成正态分布样本
    samples = np.random.normal(size=10)
    # 生成一组 x 值
    x = np.linspace(-5, 5, 10)

    # 遍历测试图形和参考图形，以及对应的对数状态
    for fig, log_state in zip([fig_test, fig_ref], [True, False]):
        # 创建子图
        ax = fig.subplots()
        # 绘制样本的直方图，根据 log_state 设置对数轴
        ax.hist(samples, log=log_state, density=True)
        # 绘制正态分布曲线
        ax.plot(x, np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi))
        # 刷新图形画布
        fig.canvas.draw()
        # 设置 y 轴为线性尺度
        ax.set_yscale('linear')


# 定义测试函数 test_offset_copy_errors
def test_offset_copy_errors():
    # 使用 pytest 检查是否会抛出 ValueError 异常，并验证异常信息
    with pytest.raises(ValueError,
                       match="'fontsize' is not a valid value for units;"
                             " supported values are 'dots', 'points', 'inches'"):
        # 调用 offset_copy 函数，期望抛出异常，指定 units 参数为 'fontsize'
        mtransforms.offset_copy(None, units='fontsize')

    # 使用 pytest 检查是否会抛出 ValueError 异常，并验证异常信息
    with pytest.raises(ValueError,
                       match='For units of inches or points a fig kwarg is needed'):
        # 调用 offset_copy 函数，期望抛出异常，指定 units 参数为 'inches'
        mtransforms.offset_copy(None, units='inches')


# 定义测试函数 test_transformedbbox_contains
def test_transformedbbox_contains():
    # 创建一个旋转 30 度的 Affine2D 变换对象，并构建相应的转换后的边界框
    bb = TransformedBbox(Bbox.unit(), Affine2D().rotate_deg(30))
    # 断言转换后的边界框是否包含指定点 (.8, .5)
    assert bb.contains(.8, .5)
    # 断言转换后的边界框是否包含指定点 (-.4, .85)
    assert bb.contains(-.4, .85)
    # 断言转换后的边界框是否不包含指定点 (.9, .5)
    assert not bb.contains(.9, .5)
    
    # 使用平移 (.25, .5) 创建新的 Affine2D 变换对象，并构建转换后的边界框
    bb = TransformedBbox(Bbox.unit(), Affine2D().translate(.25, .5))
    # 断言转换后的边界框是否包含指定点 (1.25, 1.5)
    assert bb.contains(1.25, 1.5)
    # 断言转换后的边界框是否不完全包含指定点 (1.25, 1.5)
    assert not bb.fully_contains(1.25, 1.5)
    # 断言转换后的边界框是否不完全包含指定点 (.1, .1)
    assert not bb.fully_contains(.1, .1)


# 定义测试函数 test_interval_contains
def test_interval_contains():
    # 断言区间 (0, 1) 是否包含值 0.5
    assert mtransforms.interval_contains((0, 1), 0.5)
    # 断言区间 (0, 1) 是否包含值 0
    assert mtransforms.interval_contains((0, 1), 0)
    # 断言区间 (0, 1) 是否包含值 1
    assert mtransforms.interval_contains((0, 1), 1)
    # 断言区间 (0, 1) 是否不包含值 -1
    assert not mtransforms.interval_contains((0, 1), -1)
    # 断言区间 (0, 1) 是否不包含值 2
    assert not mtransforms.interval_contains((0, 1), 2)
    # 断言区间 (1, 0) 是否包含值 0.5
    assert mtransforms.interval_contains((1, 0), 0.5)


# 定义测试函数 test_interval_contains_open
def test_interval_contains_open():
    # 断言开放区间 (0, 1) 是否包含值 0.5
    assert mtransforms.interval_contains_open((0, 1), 0.5)
    # 断言开放区间 (0, 1) 是否不包含值 0
    assert not mtransforms.interval_contains_open((0, 1), 0)
    # 断言开放区间 (0, 1) 是否不包含值 1
    assert not mtransforms.interval_contains_open((0, 1), 1)
    # 断言开放区间 (0, 1) 是否不包含值 -1
    assert not mtransforms.interval_contains_open((0, 1), -1)
    # 断言开放区间 (0, 1) 是否不包含值 2
    assert not mtransforms.interval_contains_open((0, 1), 2)
    # 断言开放区间 (1, 0) 是否包含值 0.5
    assert mtransforms.interval_contains_open((1, 0), 0.5)
```