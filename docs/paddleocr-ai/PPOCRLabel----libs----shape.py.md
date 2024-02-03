# `.\PaddleOCR\PPOCRLabel\libs\shape.py`

```
# 版权声明，允许任何人免费获取软件及相关文档，可以自由使用、复制、修改、合并、发布、分发、许可、出售软件的副本，
# 并允许将软件提供给他人，但需要包含版权声明和许可声明，不提供任何担保，作者或版权持有人不对任何索赔、损害或其他责任负责
# !/usr/bin/python
# -*- coding: utf-8 -*-
# 导入所需模块
import math
import sys

from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QPen, QPainterPath, QFont
# 导入自定义模块中的函数
from libs.utils import distance

# 定义默认线条颜色和填充颜色
DEFAULT_LINE_COLOR = QColor(0, 255, 0, 128)
DEFAULT_FILL_COLOR = QColor(255, 0, 0, 128)
DEFAULT_SELECT_LINE_COLOR = QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QColor(0, 128, 255, 155)
DEFAULT_VERTEX_FILL_COLOR = QColor(0, 255, 0, 255)
DEFAULT_HVERTEX_FILL_COLOR = QColor(255, 0, 0)
DEFAULT_LOCK_COLOR = QColor(255, 0, 255)
MIN_Y_LABEL = 10

# 定义形状类
class Shape(object):
    # 定义形状的类型
    P_SQUARE, P_ROUND = range(2)

    MOVE_VERTEX, NEAR_VERTEX = range(2)

    # 以下类变量影响所有形状对象的绘制
    line_color = DEFAULT_LINE_COLOR
    # 设置填充颜色为默认填充颜色
    fill_color = DEFAULT_FILL_COLOR
    # 设置选中线条颜色为默认选中线条颜色
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    # 设置选中填充颜色为默认选中填充颜色
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    # 设置顶点填充颜色为默认顶点填充颜色
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    # 设置高亮顶点填充颜色为默认高亮顶点填充颜色
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    # 设置点的类型为圆形
    point_type = P_ROUND
    # 设置点的大小为8
    point_size = 8
    # 设置缩放比例为1.0
    scale = 1.0

    # 初始化函数，接受标签、线条颜色、是否困难、关键类别、是否绘制标签、是否绘制索引作为参数
    def __init__(self, label=None, line_color=None, difficult=False, key_cls="None", paintLabel=False, paintIdx=False):
        # 初始化标签
        self.label = label
        # 初始化索引为None，仅用于表格注释的边界框顺序
        self.idx = None
        # 初始化点列表为空
        self.points = []
        # 初始化填充状态为False
        self.fill = False
        # 初始化选中状态为False
        self.selected = False
        # 初始化是否困难状态
        self.difficult = difficult
        # 初始化关键类别
        self.key_cls = key_cls
        # 初始化是否绘制标签
        self.paintLabel = paintLabel
        # 初始化是否绘制索引
        self.paintIdx = paintIdx
        # 初始化锁定状态为False
        self.locked = False
        # 初始化方向为0
        self.direction = 0
        # 初始化中心点为None
        self.center = None
        # 初始化容差为5，与画布相同
        self.epsilon = 5
        # 初始化高亮索引为None
        self._highlightIndex = None
        # 初始化高亮模式为NEAR_VERTEX
        self._highlightMode = self.NEAR_VERTEX
        # 初始化高亮设置
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }
        # 初始化字体大小为8
        self.fontsize = 8

        # 初始化闭合状态为False
        self._closed = False

        # 如果线条颜色不为None，则覆盖类的线条颜色属性
        # 使用对象属性。目前用于以不同颜色绘制待定线条。
        if line_color is not None:
            self.line_color = line_color

    # 旋转函数，接受旋转角度作为参数
    def rotate(self, theta):
        # 遍历点列表，旋转每个点
        for i, p in enumerate(self.points):
            self.points[i] = self.rotatePoint(p, theta)
        # 更新方向
        self.direction -= theta
        self.direction = self.direction % (2 * math.pi)

    # 旋转点函数，接受点和旋转角度作为参数
    def rotatePoint(self, p, theta):
        # 计算点相对于中心点的顺序
        order = p - self.center
        cosTheta = math.cos(theta)
        sinTheta = math.sin(theta)
        # 计算旋转后的点坐标
        pResx = cosTheta * order.x() + sinTheta * order.y()
        pResy = - sinTheta * order.x() + cosTheta * order.y()
        pRes = QPointF(self.center.x() + pResx, self.center.y() + pResy)
        return pRes
    # 计算多边形中心点坐标，取第一个点和第三个点的中点
    self.center = QPointF((self.points[0].x() + self.points[2].x()) / 2,
                          (self.points[0].y() + self.points[2].y()) / 2)
    # 标记多边形已闭合
    self._closed = True

    # 判断多边形是否达到最大点数
    def reachMaxPoints(self):
        if len(self.points) >= 4:
            return True
        return False

    # 添加点到多边形
    def addPoint(self, point):
        # 如果多边形已达到最大点数且新点与起始点足够接近，则闭合多边形
        if self.reachMaxPoints() and self.closeEnough(self.points[0], point):
            self.close()
        else:
            self.points.append(point)

    # 判断两点是否足够接近
    def closeEnough(self, p1, p2):
        return distance(p1 - p2) < self.epsilon

    # 弹出最后一个点
    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    # 判断多边形是否已闭合
    def isClosed(self):
        return self._closed

    # 将多边形标记为未闭合
    def setOpen(self):
        self._closed = False

    # 绘制多边形顶点
    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        # 如果当前顶点为高亮顶点，则根据高亮设置调整顶点大小和形状
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        # 根据是否有高亮顶点设置顶点填充颜色
        if self._highlightIndex is not None:
            self.vertex_fill_color = self.hvertex_fill_color
        else:
            self.vertex_fill_color = Shape.vertex_fill_color
        # 根据顶点形状绘制顶点
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    # 查找距离给定点最近的顶点
    def nearestVertex(self, point, epsilon):
        for i, p in enumerate(self.points):
            if distance(p - point) <= epsilon:
                return i
        return None

    # 判断给定点是否在多边形内
    def containsPoint(self, point):
        return self.makePath().contains(point)

    # 创建多边形路径
    def makePath(self):
        path = QPainterPath(self.points[0])
        for p in self.points[1:]:
            path.lineTo(p)
        return path

    # 计算多边形的边界矩形
    def boundingRect(self):
        return self.makePath().boundingRect()
    # 根据给定偏移量移动所有点的位置
    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    # 根据给定偏移量移动特定索引的点的位置
    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    # 高亮特定索引的点，并指定高亮动作
    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    # 清除高亮点
    def highlightClear(self):
        self._highlightIndex = None

    # 复制当前形状对象
    def copy(self):
        shape = Shape("%s" % self.label)
        shape.points = [p for p in self.points]
        shape.center = self.center
        shape.direction = self.direction
        shape.fill = self.fill
        shape.selected = self.selected
        shape._closed = self._closed
        # 如果线条颜色与默认值不同，则设置为当前值
        if self.line_color != Shape.line_color:
            shape.line_color = self.line_color
        # 如果填充颜色与默认值不同，则设置为当前值
        if self.fill_color != Shape.fill_color:
            shape.fill_color = self.fill_color
        shape.difficult = self.difficult
        shape.key_cls = self.key_cls
        return shape

    # 返回当前形状对象的点的数量
    def __len__(self):
        return len(self.points)

    # 获取指定索引的点
    def __getitem__(self, key):
        return self.points[key]

    # 设置指定索引的点的值
    def __setitem__(self, key, value):
        self.points[key] = value
```