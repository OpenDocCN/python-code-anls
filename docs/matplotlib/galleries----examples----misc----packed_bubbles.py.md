# `D:\src\scipysrc\matplotlib\galleries\examples\misc\packed_bubbles.py`

```
"""
===================
Packed-bubble chart
===================

Create a packed-bubble chart to represent scalar data.
The presented algorithm tries to move all bubbles as close to the center of
mass as possible while avoiding some collisions by moving around colliding
objects. In this example we plot the market share of different desktop
browsers.
(source: https://gs.statcounter.com/browser-market-share/desktop/worldwidev)
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 库中的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 库，用于数值计算

browser_market_share = {
    'browsers': ['firefox', 'chrome', 'safari', 'edge', 'ie', 'opera'],
    'market_share': [8.61, 69.55, 8.36, 4.12, 2.76, 2.43],
    'color': ['#5A69AF', '#579E65', '#F9C784', '#FC944A', '#F24C00', '#00B825']
}

class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)  # 将输入的 area 转换为 NumPy 数组
        r = np.sqrt(area / np.pi)  # 计算每个泡泡的半径，使其面积与给定的 area 对应

        self.bubble_spacing = bubble_spacing  # 设置泡泡之间的最小间距
        self.bubbles = np.ones((len(area), 4))  # 创建一个形状为 (len(area), 4) 的数组，用于存储泡泡的位置和属性
        self.bubbles[:, 2] = r  # 将泡泡数组的第三列设置为计算得到的半径 r
        self.bubbles[:, 3] = area  # 将泡泡数组的第四列设置为输入的 area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing  # 计算泡泡之间的最大步长
        self.step_dist = self.maxstep / 2  # 计算步长的一半，用于调整泡泡位置

        # 计算初始的泡泡网格布局
        length = np.ceil(np.sqrt(len(self.bubbles)))  # 计算网格边长
        grid = np.arange(length) * self.maxstep  # 生成基础网格
        gx, gy = np.meshgrid(grid, grid)  # 创建网格坐标
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]  # 将网格坐标应用到泡泡数组的 x 坐标
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]  # 将网格坐标应用到泡泡数组的 y 坐标

        self.com = self.center_of_mass()  # 计算泡泡的质心位置

    def center_of_mass(self):
        """
        Calculate the center of mass of bubbles.

        Returns
        -------
        numpy.ndarray
            Coordinates of the center of mass (x, y).
        """
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )  # 使用泡泡的面积作为权重，计算泡泡群的质心位置

    def center_distance(self, bubble, bubbles):
        """
        Calculate distances from a bubble to all other bubbles' centers.

        Parameters
        ----------
        bubble : numpy.ndarray
            Coordinates and radius of the bubble.
        bubbles : numpy.ndarray
            Array of coordinates and radii of all bubbles.

        Returns
        -------
        numpy.ndarray
            Array of distances from the given bubble to all others.
        """
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])  # 计算给定泡泡到所有其他泡泡中心的距离

    def outline_distance(self, bubble, bubbles):
        """
        Calculate distances from a bubble to all other bubbles' outlines.

        Parameters
        ----------
        bubble : numpy.ndarray
            Coordinates and radius of the bubble.
        bubbles : numpy.ndarray
            Array of coordinates and radii of all bubbles.

        Returns
        -------
        numpy.ndarray
            Array of distances from the given bubble to all others' outlines.
        """
        center_distance = self.center_distance(bubble, bubbles)  # 获取到所有其他泡泡中心的距离
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing  # 计算给定泡泡到所有其他泡泡轮廓的距离

    def check_collisions(self, bubble, bubbles):
        """
        Check if a bubble collides with any others.

        Parameters
        ----------
        bubble : numpy.ndarray
            Coordinates and radius of the bubble.
        bubbles : numpy.ndarray
            Array of coordinates and radii of all bubbles.

        Returns
        -------
        int
            Number of collisions detected for the given bubble.
        """
        distance = self.outline_distance(bubble, bubbles)  # 获取给定泡泡到所有其他泡泡轮廓的距离
        return len(distance[distance < 0])  # 统计小于零的距离，即发生碰撞的泡泡数目

    def collides_with(self, bubble, bubbles):
        """
        Find the index of the closest bubble that a given bubble collides with.

        Parameters
        ----------
        bubble : numpy.ndarray
            Coordinates and radius of the bubble.
        bubbles : numpy.ndarray
            Array of coordinates and radii of all bubbles.

        Returns
        -------
        numpy.ndarray
            Index of the closest colliding bubble.
        """
        distance = self.outline_distance(bubble, bubbles)  # 获取给定泡泡到所有其他泡泡轮廓的距离
        return np.argmin(distance, keepdims=True)  # 返回距离最小的碰撞泡泡的索引
    # 对泡泡进行合并，使其朝质心移动

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # 尝试直接朝向质心移动的方向向量
                dir_vec = self.com - self.bubbles[i, :2]

                # 将方向向量缩短为长度为1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # 计算新的泡泡位置
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # 检查新泡泡位置是否与其他泡泡碰撞
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # 尝试绕过碰撞的泡泡移动
                    # 找到与其碰撞的泡泡
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # 计算方向向量
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # 计算正交向量
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # 测试向哪个方向移动
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            # 如果移动的泡泡比例小于10%，减小步长
            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2
    # 循环遍历所有泡泡的数据
    for i in range(len(self.bubbles)):
        # 创建一个圆形对象（圆心坐标为泡泡位置的前两个元素，半径为第三个元素，颜色为给定的颜色列表中的第i个颜色）
        circ = plt.Circle(self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
        # 将圆形对象添加到指定的绘图轴上
        ax.add_patch(circ)
        # 在泡泡的中心位置添加标签文本（文本为labels列表中的第i个标签，水平和垂直对齐方式均为居中）
        ax.text(*self.bubbles[i, :2], labels[i], horizontalalignment='center', verticalalignment='center')
# 创建一个气泡图对象，使用浏览器市场份额数据设置气泡区域大小，并设置气泡之间的间距为0.1
bubble_chart = BubbleChart(area=browser_market_share['market_share'],
                           bubble_spacing=0.1)

# 将气泡图对象进行折叠（这个方法的具体效果需要根据BubbleChart类的实现来理解）
bubble_chart.collapse()

# 创建一个新的图形窗口和轴对象，设置其纵横比为等比例
fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))

# 使用气泡图对象绘制气泡图到指定的轴对象上，使用浏览器名称作为标签，颜色数据用于着色
bubble_chart.plot(
    ax, browser_market_share['browsers'], browser_market_share['color'])

# 关闭轴的显示
ax.axis("off")

# 重新计算轴的限制
ax.relim()

# 自动调整轴的视图范围
ax.autoscale_view()

# 设置图形的标题为"Browser market share"
ax.set_title('Browser market share')

# 显示整个图形
plt.show()
```