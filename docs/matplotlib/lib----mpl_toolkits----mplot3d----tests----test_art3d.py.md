# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\mplot3d\tests\test_art3d.py`

```py
import numpy as np  # 导入 NumPy 库，用于数值计算

import matplotlib.pyplot as plt  # 导入 Matplotlib 库的 pyplot 模块，用于绘图

from matplotlib.backend_bases import MouseEvent  # 导入 MouseEvent 类，用于处理鼠标事件
from mpl_toolkits.mplot3d.art3d import Line3DCollection, _all_points_on_plane  # 导入 Line3DCollection 类和 _all_points_on_plane 函数，用于处理3D图形绘制


def test_scatter_3d_projection_conservation():
    fig = plt.figure()  # 创建一个新的图形窗口
    ax = fig.add_subplot(projection='3d')  # 在图形中添加一个3D子图
    # 设置固定的3D投影角度
    ax.roll = 0
    ax.elev = 0
    ax.azim = -45
    ax.stale = True  # 设置子图状态为stale，需要重新绘制

    x = [0, 1, 2, 3, 4]
    scatter_collection = ax.scatter(x, x, x)  # 在3D子图中绘制散点图
    fig.canvas.draw_idle()  # 更新图形显示

    # 获取散点图在画布上的位置并冻结数据
    scatter_offset = scatter_collection.get_offsets()
    scatter_location = ax.transData.transform(scatter_offset)

    # 尝试两个不同的azim角度来产生反向z-order的两组散点
    for azim in (-44, -46):
        ax.azim = azim  # 设置新的azim角度
        ax.stale = True  # 设置子图状态为stale，需要重新绘制
        fig.canvas.draw_idle()  # 更新图形显示

        for i in range(5):
            # 创建一个鼠标事件用于定位并获取每个点的索引
            event = MouseEvent("button_press_event", fig.canvas,
                               *scatter_location[i, :])
            contains, ind = scatter_collection.contains(event)
            assert contains is True  # 断言：散点包含鼠标事件位置
            assert len(ind["ind"]) == 1  # 断言：索引长度为1
            assert ind["ind"][0] == i  # 断言：索引值与预期相符


def test_zordered_error():
    # 烟雾测试：检查 https://github.com/matplotlib/matplotlib/issues/26497
    lc = [(np.fromiter([0.0, 0.0, 0.0], dtype="float"),
           np.fromiter([1.0, 1.0, 1.0], dtype="float"))]
    pc = [np.fromiter([0.0, 0.0], dtype="float"),
          np.fromiter([0.0, 1.0], dtype="float"),
          np.fromiter([1.0, 1.0], dtype="float")]

    fig = plt.figure()  # 创建一个新的图形窗口
    ax = fig.add_subplot(projection="3d")  # 在图形中添加一个3D子图
    ax.add_collection(Line3DCollection(lc))  # 在3D子图中添加线集合
    ax.scatter(*pc, visible=False)  # 在3D子图中添加散点（不可见）
    plt.draw()  # 绘制图形


def test_all_points_on_plane():
    # 测试所有点是否在同一平面的函数 _all_points_on_plane 的多种情况

    # 非共面点
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert not _all_points_on_plane(*points.T)

    # 重复点
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert _all_points_on_plane(*points.T)

    # 存在 NaN 值
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, np.nan]])
    assert _all_points_on_plane(*points.T)

    # 少于3个唯一点
    points = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert _all_points_on_plane(*points.T)

    # 所有点在一条直线上
    points = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])
    assert _all_points_on_plane(*points.T)

    # 所有点在两条线上，且反向向量
    points = np.array([[-2, 2, 0], [-1, 1, 0], [1, -1, 0],
                       [0, 0, 0], [2, 0, 0], [1, 0, 0]])
    assert _all_points_on_plane(*points.T)

    # 所有点在一个平面上
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0]])
    assert _all_points_on_plane(*points.T)
```