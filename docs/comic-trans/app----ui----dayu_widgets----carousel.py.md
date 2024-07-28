# `.\comic-translate\app\ui\dayu_widgets\carousel.py`

```py
# 指定脚本使用的 Python 解释器
#!/usr/bin/env python
# 指定编码格式为 UTF-8
# -*- coding: utf-8 -*-

###################################################################
# 作者: Mu yanru
# 日期: 2019.3
# 邮箱: muyanru345@163.com
###################################################################

# 导入未来版本兼容模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入内置模块
import functools

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

# 导入本地模块
from . import dayu_theme  # 导入自定义的主题模块
from .mixin import property_mixin  # 导入属性混合类

# 创建 MGuidPrivate 类，继承自 QtWidgets.QFrame
class MGuidPrivate(QtWidgets.QFrame):
    # 定义信号 sig_go_to_page
    sig_go_to_page = QtCore.Signal()

    # 初始化方法
    def __init__(self, parent=None):
        super(MGuidPrivate, self).__init__(parent)
        # 设置鼠标形状为手型
        self.setCursor(QtCore.Qt.PointingHandCursor)
        # 调用 set_checked 方法，传入 False
        self.set_checked(False)

    # 设置选中状态的方法
    def set_checked(self, value):
        # 根据 value 的值设置样式表，选择主题颜色或者背景颜色
        self.setStyleSheet(
            "background-color:{}".format(dayu_theme.primary_color if value else dayu_theme.background_color)
        )
        # 根据 value 的值设置固定大小，20x4 或者 16x4
        self.setFixedSize(20 if value else 16, 4)

    # 鼠标按下事件处理方法
    def mousePressEvent(self, event):
        # 如果按下的是左键
        if event.buttons() == QtCore.Qt.LeftButton:
            # 发射 sig_go_to_page 信号
            self.sig_go_to_page.emit()
        # 调用父类的鼠标按下事件处理方法
        return super(MGuidPrivate, self).mousePressEvent(event)


# 使用 property_mixin 装饰器装饰 MCarousel 类
@property_mixin
class MCarousel(QtWidgets.QGraphicsView):
    def __init__(self, pix_list, autoplay=True, width=500, height=500, parent=None):
        super(MCarousel, self).__init__(parent)
        self.scene = QtWidgets.QGraphicsScene()  # 创建一个图形场景对象
        self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(dayu_theme.background_color)))  # 设置场景背景颜色
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # 设置水平滚动条策略为始终关闭
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # 设置垂直滚动条策略为始终关闭
        self.setScene(self.scene)  # 将场景设置给视图
        self.setRenderHints(QtGui.QPainter.Antialiasing)  # 设置渲染提示，这里是抗锯齿渲染
        self.hor_bar = self.horizontalScrollBar()  # 获取水平滚动条对象
        self.carousel_width = width  # 设置轮播器宽度
        self.carousel_height = height  # 设置轮播器高度

        pos = QtCore.QPoint(0, 0)  # 创建一个起始位置点
        pen = QtGui.QPen(QtCore.Qt.red)  # 创建一个红色画笔
        pen.setWidth(5)  # 设置画笔宽度为5
        self.page_count = len(pix_list)  # 计算图片列表的长度，即页数
        line_width = 20  # 设置线条宽度
        total_width = self.page_count * (line_width + 5)  # 计算总宽度，考虑到间距

        self.scene.setSceneRect(0, 0, self.page_count * width, height)  # 设置场景的矩形区域大小

        self.navigate_lay = QtWidgets.QHBoxLayout()  # 创建一个水平布局管理器
        self.navigate_lay.setSpacing(5)  # 设置布局内部控件的间距
        target_size = min(width, height)  # 计算目标尺寸，取宽和高的最小值
        for index, pix in enumerate(pix_list):  # 遍历图片列表
            if pix.width() > pix.height():  # 判断图片宽度是否大于高度
                new_pix = pix.scaledToWidth(target_size, QtCore.Qt.SmoothTransformation)  # 按宽度缩放图片
            else:
                new_pix = pix.scaledToHeight(target_size, QtCore.Qt.SmoothTransformation)  # 按高度缩放图片
            pix_item = QtWidgets.QGraphicsPixmapItem(new_pix)  # 创建图形像素项
            pix_item.setPos(pos)  # 设置图形项的位置
            pix_item.setTransformationMode(QtCore.Qt.SmoothTransformation)  # 设置平滑变换模式
            pos.setX(pos.x() + width)  # 更新下一个图形项的位置
            line_item = MGuidPrivate()  # 创建私有的导航线条项
            line_item.sig_go_to_page.connect(functools.partial(self.go_to_page, index))  # 连接跳转页面的信号
            self.navigate_lay.addWidget(line_item)  # 将线条项添加到布局管理器
            self.scene.addItem(pix_item)  # 将图形像素项添加到场景中

        hud_widget = QtWidgets.QWidget(self)  # 创建一个HUD小部件
        hud_widget.setLayout(self.navigate_lay)  # 将布局管理器设置为HUD小部件的布局
        hud_widget.setStyleSheet("background:transparent")  # 设置HUD小部件的样式表，使背景透明
        hud_widget.move(int(width / 2 - total_width / 2), height - 30)  # 移动HUD小部件到合适的位置

        self.setFixedWidth(width + 2)  # 设置固定宽度
        self.setFixedHeight(height + 2)  # 设置固定高度
        self.loading_ani = QtCore.QPropertyAnimation()  # 创建属性动画对象
        self.loading_ani.setTargetObject(self.hor_bar)  # 设置动画对象为水平滚动条
        self.loading_ani.setEasingCurve(QtCore.QEasingCurve.InOutQuad)  # 设置动画的缓和曲线
        self.loading_ani.setDuration(500)  # 设置动画持续时间为500毫秒
        self.loading_ani.setPropertyName(b"value")  # 设置动画的属性名
        self.autoplay_timer = QtCore.QTimer(self)  # 创建定时器对象
        self.autoplay_timer.setInterval(2000)  # 设置定时器间隔为2000毫秒
        self.autoplay_timer.timeout.connect(self.next_page)  # 连接定时器超时信号到下一页的槽函数

        self.current_index = 0  # 设置当前索引为0
        self.go_to_page(0)  # 调用跳转到第一页的方法
        self.set_autoplay(autoplay)  # 设置自动播放

    def set_autoplay(self, value):
        self.setProperty("autoplay", value)  # 设置自动播放属性

    def _set_autoplay(self, value):
        if value:
            self.autoplay_timer.start()  # 如果值为真，启动自动播放定时器
        else:
            self.autoplay_timer.stop()  # 如果值为假，停止自动播放定时器

    def set_interval(self, ms):
        self.autoplay_timer.setInterval(ms)  # 设置自动播放定时器的间隔时间
    # 定义一个方法用于显示下一页的内容
    def next_page(self):
        # 计算下一页的索引，如果当前索引加1小于总页数，则取当前索引加1，否则取0
        index = self.current_index + 1 if self.current_index + 1 < self.page_count else 0
        # 调用内部方法，跳转到计算得到的索引页
        self.go_to_page(index)

    # 定义一个方法用于显示上一页的内容
    def pre_page(self):
        # 计算上一页的索引，如果当前索引大于0，则取当前索引减1，否则取总页数减1
        index = self.current_index - 1 if self.current_index > 0 else self.page_count - 1
        # 调用内部方法，跳转到计算得到的索引页
        self.go_to_page(index)

    # 定义一个方法用于跳转到指定索引的页
    def go_to_page(self, index):
        # 设置动画的起始值为当前索引乘以每页的宽度
        self.loading_ani.setStartValue(self.current_index * self.carousel_width)
        # 设置动画的结束值为指定索引乘以每页的宽度
        self.loading_ani.setEndValue(index * self.carousel_width)
        # 启动动画效果
        self.loading_ani.start()
        # 更新当前索引为指定索引
        self.current_index = index
        # 遍历导航布局中的所有项
        for i in range(self.navigate_lay.count()):
            # 获取导航布局中第i个项对应的窗口部件
            frame = self.navigate_lay.itemAt(i).widget()
            # 设置第i个项是否被选中，根据当前索引是否等于i来决定
            frame.set_checked(i == self.current_index)
```