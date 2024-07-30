# `.\comic-translate\app\ui\dayu_widgets\menu.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################

# 导入未来模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入内置模块
from functools import partial  # 导入 functools 模块的 partial 函数
import re  # 导入 re 模块（正则表达式模块）

# 导入第三方模块
# from qtpy import QtCompat
from PySide6 import QtCore  # 导入 PySide6 模块的 QtCore
from PySide6 import QtGui  # 导入 PySide6 模块的 QtGui
from PySide6 import QtWidgets  # 导入 PySide6 模块的 QtWidgets
import six  # 导入 six 模块（用于 Python 2 和 3 兼容性）

# 导入本地模块
from .line_edit import MLineEdit  # 从当前包中导入 line_edit 模块的 MLineEdit 类
from .mixin import property_mixin  # 从当前包中导入 mixin 模块的 property_mixin 函数
from .popup import MPopup  # 从当前包中导入 popup 模块的 MPopup 类
from .import utils as utils  # 从当前包中导入 utils 模块并重命名为 utils

@property_mixin
class ScrollableMenuBase(QtWidgets.QMenu):
    """
    https://www.pythonfixing.com/2021/10/fixed-how-to-have-scrollable-context.html
    """

    deltaY = 0  # 定义类变量 deltaY，初始值为 0
    dirty = True  # 定义类变量 dirty，初始值为 True
    ignoreAutoScroll = False  # 定义类变量 ignoreAutoScroll，初始值为 False

    def __init__(self, *args, **kwargs):
        super(ScrollableMenuBase, self).__init__(*args, **kwargs)
        self._maximumHeight = self.maximumHeight()  # 记录菜单的最大高度
        self._actionRects = []  # 初始化用于存放动作区域的列表

        self.scrollTimer = QtCore.QTimer(self, interval=50, singleShot=True, timeout=self.checkScroll)
        # 创建定时器 scrollTimer，设置间隔为 50 毫秒，单次触发，超时时调用 self.checkScroll 方法
        self.scrollTimer.setProperty("defaultInterval", 50)  # 设置 scrollTimer 的属性 defaultInterval 为 50

        self.delayTimer = QtCore.QTimer(self, interval=100, singleShot=True)
        # 创建延迟定时器 delayTimer，设置间隔为 100 毫秒，单次触发

        self.setMaxItemCount(0)  # 调用自定义方法设置最大项数为 0

    def _set_max_scroll_count(self, value):
        self.setMaxItemCount(value * 2.2)  # 设置最大项数为 value 的 2.2 倍

    @property
    def actionRects(self):
        if self.dirty or not self._actionRects:
            del self._actionRects[:]  # 清空动作区域列表
            offset = self.offset()  # 获取菜单的偏移量
            for action in self.actions():
                geo = super(ScrollableMenuBase, self).actionGeometry(action)
                if offset:
                    geo.moveTop(geo.y() - offset)  # 根据偏移量调整动作区域的位置
                self._actionRects.append(geo)  # 将调整后的动作区域添加到列表中
            self.dirty = False  # 将 dirty 标记为 False，表示动作区域列表已更新
        return self._actionRects  # 返回动作区域列表

    def iterActionRects(self):
        for action, rect in zip(self.actions(), self.actionRects):
            yield action, rect  # 生成器函数，依次返回动作和对应的动作区域
    # 设置菜单项的最大数量，根据给定的数量调整菜单项的最大高度
    def setMaxItemCount(self, count):
        # 获取当前菜单的样式
        style = self.style()
        # 初始化菜单项的样式选项
        opt = QtWidgets.QStyleOptionMenuItem()
        opt.initFrom(self)

        # 创建一个虚拟动作（用于获取字体相关信息）
        a = QtGui.QAction("fake action", self)
        # 使用当前菜单的样式初始化样式选项
        self.initStyleOption(opt, a)
        
        # 计算菜单项的默认宽度
        size = QtCore.QSize()
        fm = self.fontMetrics()
        qfm = opt.fontMetrics
        size.setWidth(fm.boundingRect(QtCore.QRect(), QtCore.Qt.TextSingleLine, a.text()).width())
        size.setHeight(max(fm.height(), qfm.height()))
        
        # 计算菜单项的默认高度
        self.defaultItemHeight = style.sizeFromContents(QtWidgets.QStyle.CT_MenuItem, opt, size, self).height()

        # 根据给定的数量调整菜单的最大高度
        if not count:
            # 如果数量为零，则恢复到原始的最大高度
            self.setMaximumHeight(self._maximumHeight)
        else:
            # 否则，根据样式和其他度量值计算新的最大高度
            fw = style.pixelMetric(QtWidgets.QStyle.PM_MenuPanelWidth, None, self)
            vmargin = style.pixelMetric(QtWidgets.QStyle.PM_MenuHMargin, opt, self)
            scrollHeight = self.scrollHeight(style)
            self.setMaximumHeight(self.defaultItemHeight * count + (fw + vmargin + scrollHeight) * 2)
        
        # 设置标志表示菜单状态已改变
        self.dirty = True

    # 返回菜单滚动条的高度
    def scrollHeight(self, style):
        return style.pixelMetric(QtWidgets.QStyle.PM_MenuScrollerHeight, None, self) * 2

    # 检查菜单是否可滚动
    def isScrollable(self):
        return self.property("scrollable") and self.height() < super(ScrollableMenuBase, self).sizeHint().height()

    # 检查并执行菜单的滚动操作
    def checkScroll(self):
        # 获取鼠标当前位置相对于菜单的位置
        pos = self.mapFromGlobal(QtGui.QCursor.pos())
        # 计算滚动的增量
        delta = max(2, int(self.defaultItemHeight * 0.25))
        
        # 根据鼠标位置确定滚动方向
        if self.scrollUpRect.contains(pos):
            delta *= -1
        elif not self.scrollDownRect.contains(pos):
            return
        
        # 执行滚动操作，并启动滚动定时器
        if self.scrollBy(delta):
            self.scrollTimer.start(self.scrollTimer.property("defaultInterval"))

    # 返回当前菜单的垂直偏移量
    def offset(self):
        # 如果菜单可滚动，返回当前垂直偏移量减去滚动条的高度
        if self.isScrollable():
            return self.deltaY - self.scrollHeight(self.style())
        return 0

    # 返回给定动作的矩形区域
    def translatedActionGeometry(self, action):
        return self.actionRects[self.actions().index(action)]

    # 确保给定动作在可见区域内，如果不在则进行滚动
    def ensureVisible(self, action):
        style = self.style()
        # 获取菜单的各种度量值
        fw = style.pixelMetric(QtWidgets.QStyle.PM_MenuPanelWidth, None, self)
        hmargin = style.pixelMetric(QtWidgets.QStyle.PM_MenuHMargin, None, self)
        vmargin = style.pixelMetric(QtWidgets.QStyle.PM_MenuVMargin, None, self)
        scrollHeight = self.scrollHeight(style)
        extent = fw + hmargin + vmargin + scrollHeight
        
        # 计算可见区域
        r = self.rect().adjusted(0, extent, 0, -extent)
        geo = self.translatedActionGeometry(action)
        
        # 确保动作在可见区域内，否则进行滚动
        if geo.top() < r.top():
            self.scrollBy(-(r.top() - geo.top()))
        elif geo.bottom() > r.bottom():
            self.scrollBy(geo.bottom() - r.bottom())
    # 按指定步长滚动内容
    def scrollBy(self, step):
        # 如果步长为负数
        if step < 0:
            # 计算新的滚动位置，确保不小于0
            newDelta = max(0, self.deltaY + step)
            # 如果新的滚动位置与当前位置相同，返回False
            if newDelta == self.deltaY:
                return False
        # 如果步长为正数
        elif step > 0:
            # 计算新的滚动位置
            newDelta = self.deltaY + step
            # 获取当前控件的样式
            style = self.style()
            # 获取整体内容的滚动高度
            scrollHeight = self.scrollHeight(style)
            # 计算控件底部位置
            bottom = self.height() - scrollHeight

            # 遍历动作列表，查找最后一个可见动作
            for lastAction in reversed(self.actions()):
                if lastAction.isVisible():
                    break
            # 计算最后一个可见动作的底部位置
            lastBottom = self.actionGeometry(lastAction).bottom() - newDelta + scrollHeight
            # 如果最后一个可见动作的底部位置小于控件底部位置，调整滚动位置
            if lastBottom < bottom:
                newDelta -= bottom - lastBottom
            # 如果新的滚动位置与当前位置相同，返回False
            if newDelta == self.deltaY:
                return False

        # 更新滚动位置并标记为脏区域
        self.deltaY = newDelta
        self.dirty = True
        # 更新界面显示
        self.update()
        return True

    # 根据位置返回对应的动作
    def actionAt(self, pos):
        for action, rect in self.iterActionRects():
            # 如果位置在动作的矩形区域内，返回该动作
            if rect.contains(pos):
                return action

    # 重新实现父类方法
    def sizeHint(self):
        # 调用父类的 sizeHint 方法获取推荐大小
        hint = super(ScrollableMenuBase, self).sizeHint()
        # 如果推荐高度超过最大高度，限制推荐高度为最大高度
        if hint.height() > self.maximumHeight():
            hint.setHeight(self.maximumHeight())
        return hint

    # 事件过滤器方法
    def eventFilter(self, source, event):
        # 如果事件类型为显示事件
        if event.type() == event.Show:
            # 如果可滚动且有滚动偏移量
            if self.isScrollable() and self.deltaY:
                # 获取源对象的动作
                action = source.menuAction()
                # 确保该动作可见并滚动到可视区域
                self.ensureVisible(action)
                # 获取动作在控件内的矩形区域
                rect = self.translatedActionGeometry(action)
                # 计算源对象移动的偏移量
                delta = rect.topLeft() - self.actionGeometry(action).topLeft()
                # 移动源对象到新的位置
                source.move(source.pos() + delta)
            # 返回False，不拦截事件继续传递
            return False
        # 其他事件类型，调用父类的事件过滤器方法处理
        return super(ScrollableMenuBase, self).eventFilter(source, event)
    # 处理事件函数，响应特定的事件类型
    def event(self, event):
        # 如果当前不可滚动，则调用父类的事件处理方法
        if not self.isScrollable():
            return super(ScrollableMenuBase, self).event(event)
        
        # 处理按键事件
        if event.type() == event.KeyPress and event.key() in (
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Down,
        ):
            # 调用父类的事件处理方法，并保存返回结果
            res = super(ScrollableMenuBase, self).event(event)
            # 获取当前活动的动作项
            action = self.activeAction()
            if action:
                # 确保当前活动的动作项可见
                self.ensureVisible(action)
                # 更新界面显示
                self.update()
            return res
        
        # 处理鼠标按下或双击事件
        elif event.type() in (event.MouseButtonPress, event.MouseButtonDblClick):
            pos = event.pos()
            # 检查鼠标位置是否在上下滚动区域内
            if self.scrollUpRect.contains(pos) or self.scrollDownRect.contains(pos):
                if event.button() == QtCore.Qt.LeftButton:
                    # 计算滚动步长，最小为默认项高度的四分之一
                    step = max(2, int(self.defaultItemHeight * 0.25))
                    if self.scrollUpRect.contains(pos):
                        step *= -1
                    # 执行滚动操作
                    self.scrollBy(step)
                    # 启动滚动定时器
                    self.scrollTimer.start(200)
                    # 忽略自动滚动标志设为True
                    self.ignoreAutoScroll = True
                return True
        
        # 处理鼠标释放事件
        elif event.type() == event.MouseButtonRelease:
            pos = event.pos()
            # 停止滚动定时器
            self.scrollTimer.stop()
            # 如果鼠标释放位置不在上下滚动区域内
            if not (self.scrollUpRect.contains(pos) or self.scrollDownRect.contains(pos)):
                # 获取释放位置下的动作项
                action = self.actionAt(pos)
                if action:
                    # 触发动作项的触发操作
                    action.trigger()
                    # 关闭菜单
                    self.close()
                return True
        
        # 对于其他类型的事件，调用父类的事件处理方法
        return super(ScrollableMenuBase, self).event(event)

    # 定时器事件处理函数，用于处理滚动菜单的定时事件
    def timerEvent(self, event):
        if not self.isScrollable():
            # 如果当前不可滚动，则忽略内部定时事件
            super(ScrollableMenuBase, self).timerEvent(event)
    # 处理鼠标移动事件的方法，继承自父类 ScrollableMenuBase
    def mouseMoveEvent(self, event):
        # 如果当前菜单不可滚动，则调用父类的鼠标移动事件处理方法并返回
        if not self.isScrollable():
            super(ScrollableMenuBase, self).mouseMoveEvent(event)
            return

        # 获取鼠标事件发生的位置
        pos = event.pos()
        
        # 检查鼠标是否在滚动区域之外，若是，则启动自动滚动定时器
        if pos.y() < self.scrollUpRect.bottom() or pos.y() > self.scrollDownRect.top():
            if not self.ignoreAutoScroll and not self.scrollTimer.isActive():
                self.scrollTimer.start(200)
            return
        
        # 将忽略自动滚动标志设为 False
        self.ignoreAutoScroll = False

        # 获取当前激活的动作
        oldAction = self.activeAction()

        # 如果鼠标位置不在当前控件的矩形范围内，则将动作设为 None
        if not self.rect().contains(pos):
            action = None
        else:
            # 获取鼠标位置处对应的动作和其所在的矩形区域
            y = event.y()
            for action, rect in self.iterActionRects():
                if rect.y() <= y <= rect.y() + rect.height():
                    break
            else:
                action = None

        # 设置当前激活的动作
        self.setActiveAction(action)

        # 如果有有效动作且该动作不是分隔符
        if action and not action.isSeparator():

            # 定义确保动作可见的函数，并连接到延时定时器的超时信号
            def ensureVisible():
                self.delayTimer.timeout.disconnect()
                self.ensureVisible(action)

            try:
                # 尝试断开延时定时器的连接
                self.delayTimer.disconnect()
            except:
                pass
            
            # 连接延时定时器的超时信号到确保可见函数，并启动定时器
            self.delayTimer.timeout.connect(ensureVisible)
            self.delayTimer.start(150)
        
        # 如果有旧的动作且其菜单是可见的
        elif oldAction and oldAction.menu() and oldAction.menu().isVisible():

            # 定义关闭菜单的函数，并连接到延时定时器的超时信号
            def closeMenu():
                self.delayTimer.timeout.disconnect()
                oldAction.menu().hide()

            # 连接延时定时器的超时信号到关闭菜单函数，并启动定时器
            self.delayTimer.timeout.connect(closeMenu)
            self.delayTimer.start(50)
        
        # 更新界面显示
        self.update()

    # 处理滚轮事件的方法
    def wheelEvent(self, event):
        # 如果当前菜单不可滚动，则直接返回
        if not self.isScrollable():
            return
        
        # 停止延时定时器
        self.delayTimer.stop()

        # 根据滚轮滚动的方向，调整滚动位置
        if event.angleDelta().y() < 0:
            self.scrollBy(self.defaultItemHeight)
        else:
            self.scrollBy(-self.defaultItemHeight)

    # 当菜单显示时触发的事件处理方法
    def showEvent(self, event):
        # 如果菜单可滚动
        if self.isScrollable():
            # 初始化 deltaY 和 dirty 标志
            self.deltaY = 0
            self.dirty = True
            
            # 对所有动作安装事件过滤器，以便捕获菜单事件
            for action in self.actions():
                if action.menu():
                    action.menu().installEventFilter(self)
            
            # 将忽略自动滚动标志设为 False
            self.ignoreAutoScroll = False
        
        # 调用父类的显示事件处理方法
        super(ScrollableMenuBase, self).showEvent(event)

    # 当菜单隐藏时触发的事件处理方法
    def hideEvent(self, event):
        # 对所有动作移除事件过滤器
        for action in self.actions():
            if action.menu():
                action.menu().removeEventFilter(self)
        
        # 调用父类的隐藏事件处理方法
        super(ScrollableMenuBase, self).hideEvent(event)
    # 调整窗口大小时触发的事件处理函数，继承自父类 ScrollableMenuBase
    def resizeEvent(self, event):
        # 调用父类的 resizeEvent 方法处理事件
        super(ScrollableMenuBase, self).resizeEvent(event)

        # 获取当前窗口的样式信息
        style = self.style()

        # 获取窗口内容区域的边距信息
        margins = self.contentsMargins()
        l, t, r, b = margins.left(), margins.top(), margins.right(), margins.bottom()

        # 获取当前窗口样式中的像素度量信息
        fw = style.pixelMetric(QtWidgets.QStyle.PM_MenuPanelWidth, None, self)
        hmargin = style.pixelMetric(QtWidgets.QStyle.PM_MenuHMargin, None, self)
        vmargin = style.pixelMetric(QtWidgets.QStyle.PM_MenuVMargin, None, self)

        # 计算左侧和顶部的边距
        leftMargin = fw + hmargin + l
        topMargin = fw + vmargin + t

        # 计算右侧和底部的边距
        bottomMargin = fw + vmargin + b

        # 计算内容区域的宽度，排除边距
        contentWidth = self.width() - (fw + hmargin) * 2 - l - r

        # 计算滚动条区域的高度
        scrollHeight = self.scrollHeight(style)

        # 设置滚动条向上按钮的位置和大小
        self.scrollUpRect = QtCore.QRect(leftMargin, topMargin, contentWidth, scrollHeight)

        # 设置滚动条向下按钮的位置和大小
        self.scrollDownRect = QtCore.QRect(
            leftMargin,
            self.height() - scrollHeight - bottomMargin,
            contentWidth,
            scrollHeight,
        )
@property_mixin
class SearchableMenuBase(ScrollableMenuBase):
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super(SearchableMenuBase, self).__init__(*args, **kwargs)
        
        # 创建一个弹出搜索框的对象，并设置为不可见状态
        self.search_popup = MPopup(self)
        self.search_popup.setVisible(False)
        
        # 创建一个搜索输入框和一个用于显示搜索结果的标签
        self.search_bar = MLineEdit(parent=self)
        self.search_label = QtWidgets.QLabel()
        
        # 连接搜索输入框的文本变化信号到槽函数 slot_search_change
        self.search_bar.textChanged.connect(self.slot_search_change)
        
        # 重载搜索输入框的按键事件，调用自定义的搜索按键处理函数 search_key_event
        self.search_bar.keyPressEvent = partial(self.search_key_event, self.search_bar.keyPressEvent)
        
        # 当弹出框即将隐藏时，清空搜索输入框的文本
        self.aboutToHide.connect(lambda: self.search_bar.setText(""))

        # 创建垂直布局，并将搜索标签和搜索输入框添加到布局中
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.search_label)
        layout.addWidget(self.search_bar)
        self.search_popup.setLayout(layout)

        # 设置对象的属性，用于国际化支持的搜索占位符和搜索标签
        self.setProperty("search_placeholder", self.tr("Search Action..."))
        self.setProperty("search_label", self.tr("Search Action..."))

        # 设置对象的属性，用于标识该对象支持搜索和搜索的正则表达式模式
        self.setProperty("searchable", True)
        self.setProperty("search_re", "I")

    def search_key_event(self, call, event):
        key = event.key()
        # 支持在搜索栏上的菜单原始按键事件的注释
        if key in (
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Down,
            QtCore.Qt.Key_Return,
            QtCore.Qt.Key_Enter,
        ):
            # 调用父类的按键事件处理方法
            super(SearchableMenuBase, self).keyPressEvent(event)
        elif key == QtCore.Qt.Key_Tab:
            # 设置焦点到搜索输入框
            self.search_bar.setFocus()
        return call(event)

    def _set_search_label(self, value):
        # 设置搜索标签的文本内容
        self.search_label.setText(value)

    def _set_search_placeholder(self, value):
        # 设置搜索输入框的占位符文本
        self.search_bar.setPlaceholderText(value)

    def _set_search_re(self, value):
        # 检查搜索正则表达式属性的类型，应为字符串类型
        if not isinstance(value, six.text_type):
            raise TypeError("`search_re` property should be a string type")

    def slot_search_change(self, text):
        # 根据搜索正则表达式属性的设置，创建一个新的搜索正则表达式对象
        flags = 0
        for m in self.property("search_re") or "":
            flags |= getattr(re, m.upper(), 0)
        search_reg = re.compile(r".*%s.*" % text, flags)
        
        # 更新搜索结果的显示
        self._update_search(search_reg)

    def _update_search(self, search_reg, parent_menu=None):
        # 获取当前菜单或父菜单下的所有操作动作
        actions = parent_menu.actions() if parent_menu else self.actions()
        vis_list = []
        
        # 遍历所有操作动作，根据搜索正则表达式匹配结果设置动作的可见性
        for action in actions:
            menu = action.menu()
            if not menu:
                is_match = bool(re.match(search_reg, action.text()))
                action.setVisible(is_match)
                is_match and vis_list.append(action)
            else:
                is_match = bool(re.match(search_reg, menu.title()))
                # 递归更新子菜单的搜索匹配
                self._update_search("" if is_match else search_reg, menu)

        # 如果存在父菜单，则设置父菜单动作的可见性，根据搜索结果是否为空
        if parent_menu:
            parent_menu.menuAction().setVisible(bool(vis_list) or not search_reg)
    # 重写父类的按键事件处理函数，处理按键事件
    def keyPressEvent(self, event):
        # 获取按下的键值
        key = event.key()
        # 检查当前对象是否具有 "searchable" 属性
        if self.property("searchable"):
            # 如果按下的是 A-Z 中的字符键（ASCII 65-90）
            # NOTES(timmyliang): 26 character trigger search bar
            if 65 <= key <= 90:
                # 将按键转换为对应的字符
                char = chr(key)
                # 在搜索栏中设置文本为按键对应的字符
                self.search_bar.setText(char)
                # 设置焦点到搜索栏
                self.search_bar.setFocus()
                # 选择搜索栏中的所有文本
                self.search_bar.selectAll()
                # 获取当前对象的推荐大小宽度
                width = self.sizeHint().width()
                # 确保宽度至少为50
                width = width if width >= 50 else 50
                # 计算弹出窗口的偏移量，使其显示在当前对象右侧
                offset = QtCore.QPoint(width, 0)
                # 将弹出窗口移动到当前对象位置加上偏移量处
                self.search_popup.move(self.pos() + offset)
                # 显示搜索弹出窗口
                self.search_popup.show()
            # 如果按下的是 Escape 键
            elif key == QtCore.Qt.Key_Escape:
                # 清空搜索栏文本
                self.search_bar.setText("")
                # 隐藏搜索弹出窗口
                self.search_popup.hide()
        # 调用父类的按键事件处理函数，并返回处理结果
        return super(SearchableMenuBase, self).keyPressEvent(event)


这段代码是一个 PyQt 应用中的按键事件处理函数。根据按下的键执行不同的操作，主要用于实现一个可搜索的菜单功能，包括根据按键触发搜索栏的显示与隐藏，以及根据按键设置搜索栏的文本内容等功能。
@property_mixin
class MMenu(SearchableMenuBase):
    # 定义信号，用于数值变化的通知
    sig_value_changed = QtCore.Signal(object)

    # 初始化函数，设置菜单的基本属性和动作组
    def __init__(self, exclusive=True, cascader=False, title="", parent=None):
        super(MMenu, self).__init__(title=title, parent=parent)
        self.setProperty("cascader", cascader)  # 设置是否级联选择
        self.setCursor(QtCore.Qt.PointingHandCursor)  # 设置鼠标光标类型
        self._action_group = QtGui.QActionGroup(self)  # 创建动作组
        self._action_group.setExclusive(exclusive)  # 设置动作组是否互斥
        self._action_group.triggered.connect(self.slot_on_action_triggered)  # 连接动作触发信号到槽函数
        self._load_data_func = None  # 初始化数据加载回调函数为空
        self.set_value("")  # 设置初始值为空字符串
        self.set_data([])  # 设置初始数据为空列表
        self.set_separator("/")  # 设置分隔符为斜杠 '/'

    # 设置分隔符的方法
    def set_separator(self, chr):
        self.setProperty("separator", chr)  # 设置分隔符属性为传入的字符

    # 设置数据加载回调函数的方法
    def set_load_callback(self, func):
        assert callable(func)  # 断言传入的参数是可调用对象
        self._load_data_func = func  # 将传入的函数赋值给数据加载回调函数
        self.aboutToShow.connect(self.slot_fetch_data)  # 连接菜单显示前的信号到数据获取槽函数

    # 数据获取槽函数，从_load_data_func获取数据并设置菜单数据
    def slot_fetch_data(self):
        data_list = self._load_data_func()  # 调用_load_data_func获取数据
        self.set_data(data_list)  # 设置菜单的数据

    # 设置菜单数值的方法
    def set_value(self, data):
        assert isinstance(data, (list, six.string_types, six.integer_types, float))  # 断言数据类型符合预期
        # 如果属性表明是级联选择，并且数据是字符串类型，则按分隔符拆分为列表
        if self.property("cascader") and isinstance(data, six.string_types):
            data = data.split(self.property("separator"))
        self.setProperty("value", data)  # 设置菜单的数值属性为传入的数据

    # 私有方法，用于设置数值的具体实现
    def _set_value(self, value):
        data_list = value if isinstance(value, list) else [value]  # 如果value是列表则直接使用，否则转为单元素列表
        flag = False  # 标记是否有变化需要通知
        for act in self._action_group.actions():  # 遍历动作组中的所有动作
            if act.property("long_path"):  # 如果动作有"long_path"属性
                # 确保所有值都是字符串类型，将列表元素用斜杠连接成字符串
                selected = "/".join(map(str, data_list))
                checked = act.property("long_path") == selected  # 检查动作的"long_path"属性是否等于选中的字符串
            else:
                checked = act.property("value") in data_list  # 检查动作的"value"属性是否在数据列表中
            if act.isChecked() != checked:  # 如果动作的选中状态与需要的状态不一致
                act.setChecked(checked)  # 更新动作的选中状态
                flag = True  # 设置标记为需要通知变化
        if flag:
            self.sig_value_changed.emit(value)  # 如果有变化，则发射数值变化信号
    def _add_menu(self, parent_menu, data_dict, long_path=None):
        # 如果数据字典中有子菜单项
        if "children" in data_dict:
            # 创建一个菜单对象，使用数据字典中的标签作为标题
            menu = MMenu(title=data_dict.get("label"), parent=self)
            # 设置菜单的值属性为数据字典中的值
            menu.setProperty("value", data_dict.get("value"))
            # 将菜单添加到父菜单中
            parent_menu.addMenu(menu)
            # 如果父菜单不是当前对象自身
            if not (parent_menu is self):
                # 将父菜单作为属性添加到当前菜单对象中，以备将来使用
                menu.setProperty("parent_menu", parent_menu)
            # 遍历子菜单项
            for i in data_dict.get("children"):
                # 如果提供了长路径，则使用长路径作为根路径；否则使用数据字典中的标签作为根路径
                long_path = long_path or data_dict.get("label")
                # 组装完整的长路径，格式为 "根路径/标签"
                assemble_long_path = "{root}/{label}".format(root=long_path, label=i.get("label"))
                # 如果有有效的组装长路径
                if assemble_long_path:
                    # 递归调用添加菜单方法，传入当前菜单作为父菜单，子菜单数据字典和组装的长路径
                    self._add_menu(menu, i, assemble_long_path)
                else:
                    # 如果没有有效的组装长路径，则直接递归调用添加菜单方法，传入当前菜单和子菜单数据字典
                    self._add_menu(menu, i)
        else:
            # 如果数据字典中没有子菜单项，则创建一个动作对象并添加到动作组中
            action = self._action_group.addAction(utils.display_formatter(data_dict.get("label")))
            # 设置动作的值属性为数据字典中的值
            action.setProperty("value", data_dict.get("value"))
            # 设置动作为可选中状态
            action.setCheckable(True)
            # 将长路径作为属性添加到动作对象中
            action.setProperty("long_path", long_path)
            # 将父菜单作为属性添加到动作对象中
            action.setProperty("parent_menu", parent_menu)
            # 将动作添加到父菜单中
            parent_menu.addAction(action)

    def set_data(self, option_list):
        # 断言选项列表是一个列表类型
        assert isinstance(option_list, list)
        # 如果选项列表非空
        if option_list:
            # 如果选项列表中所有元素均为字符串类型
            if all(isinstance(i, six.string_types) for i in option_list):
                # 使用工具方法将字符串列表转换为嵌套字典，分隔符为属性中的分隔符
                option_list = utils.from_list_to_nested_dict(option_list, sep=self.property("separator"))
            # 如果选项列表中所有元素均为整数或浮点数类型
            if all(isinstance(i, (int, float)) for i in option_list):
                # 将选项列表转换为包含字典的列表，每个字典有"value"和"label"键
                option_list = [{"value": i, "label": str(i)} for i in option_list]
        # 将数据属性设置为处理后的选项列表
        self.setProperty("data", option_list)

    def _set_data(self, option_list):
        # 清空当前对象中的所有菜单和动作
        self.clear()
        for act in self._action_group.actions():
            # 从动作组中移除每一个动作
            self._action_group.removeAction(act)

        # 遍历选项列表中的每个数据字典，并添加到当前对象中作为菜单
        for data_dict in option_list:
            self._add_menu(self, data_dict)

    def _get_parent(self, result, obj):
        # 如果对象有父菜单属性
        if obj.property("parent_menu"):
            # 获取对象的父菜单
            parent_menu = obj.property("parent_menu")
            # 将父菜单的值属性添加到结果列表的开头
            result.insert(0, parent_menu.property("value"))
            # 递归调用获取父菜单方法，传入结果列表和父菜单
            self._get_parent(result, parent_menu)

    def slot_on_action_triggered(self, action):
        # 获取当前动作的值属性
        current_data = action.property("value")
        # 如果当前对象具有“级联选择”属性
        if self.property("cascader"):
            # 创建一个选中数据的列表，初始值为当前动作的值属性
            selected_data = [current_data]
            # 获取动作的所有父菜单数据
            self._get_parent(selected_data, action)
        else:
            # 如果动作组是排他性的
            if self._action_group.isExclusive():
                # 选中数据为当前动作的值属性
                selected_data = current_data
            else:
                # 否则，选中数据为动作组中所有已选中动作的值属性列表
                selected_data = [act.property("value") for act in self._action_group.actions() if act.isChecked()]
        # 设置当前对象的值为选中数据
        self.set_value(selected_data)
        # 发射值变更信号
        self.sig_value_changed.emit(selected_data)

    def set_loader(self, func):
        # 将加载数据函数设置为当前对象的私有属性
        self._load_data_func = func
```