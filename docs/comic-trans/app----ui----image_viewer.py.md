# `.\comic-translate\app\ui\image_viewer.py`

```py
# Import third-party modules
# 从 PySide6 模块中导入 QtWidgets、QtCore 和 QtGui
from PySide6 import QtWidgets
from PySide6 import QtCore, QtGui

# 导入 OpenCV (cv2) 和 numpy (np) 库
import cv2
import numpy as np
# 导入 math 模块
import math
# 导入类型提示 List 和 Dict
from typing import List, Dict

# 定义 ImageViewer 类，继承自 QtWidgets.QGraphicsView
class ImageViewer(QtWidgets.QGraphicsView):
    # 定义信号 rectangle_selected 和 rectangle_changed，类型为 QtCore.QRectF
    rectangle_selected = QtCore.Signal(QtCore.QRectF)
    rectangle_changed = QtCore.Signal(QtCore.QRectF)

    # 初始化函数，接受父类对象 parent
    def __init__(self, parent):
        # 调用父类的初始化函数
        super().__init__(parent)
        
        # 初始化变量
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._photo.setShapeMode(QtWidgets.QGraphicsPixmapItem.BoundingRectShape)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        
        # 初始化工具相关变量
        self._current_tool = None
        self._box_mode = False
        self._dragging = False
        self._start_point = None
        self._current_rect = None
        self._rectangles = []
        self._selected_rect = None
        self._drag_start = None
        self._drag_offset = None
        self._resize_handle = None
        self._resize_start = None
        self._panning = False
        self._pan_start_pos = None

        # 设置默认的笔刷颜色和大小
        self._brush_color = QtGui.QColor(255, 0, 0, 100)
        self._brush_size = 25
        self._drawing_path = None
        self._drawing_items = []
        self._undo_brush_stack = []
        self._redo_brush_stack = []
        self._eraser_size = 25

        # 创建笔刷和橡皮擦的光标
        self._brush_cursor = self.create_inpaint_cursor('brush', self._brush_size)
        self._eraser_cursor = self.create_inpaint_cursor('eraser', self._eraser_size)

        self._current_path = None
        self._current_path_item = None
        
        # 初始化上次的平移位置
        self._last_pan_pos = QtCore.QPoint()

    # 判断是否有照片加载
    def hasPhoto(self):
        return not self._empty

    # 处理视口事件
    def viewportEvent(self, event):
        # 如果事件类型是手势事件
        if event.type() == QtCore.QEvent.Gesture:
            return self.gestureEvent(event)
        # 调用父类的视口事件处理函数
        return super().viewportEvent(event)

    # 处理手势事件
    def gestureEvent(self, event):
        # 获取手势中的平移手势和缩放手势
        pan = event.gesture(QtCore.Qt.GestureType.PanGesture)
        pinch = event.gesture(QtCore.Qt.GestureType.PinchGesture)
        
        # 如果存在平移手势
        if pan:
            return self.handlePanGesture(pan)
        # 如果存在缩放手势
        elif pinch:
            return self.handlePinchGesture(pinch)
        
        return False
    # 处理滑动手势
    def handlePanGesture(self, gesture):
        # 获取手势的位移
        delta = gesture.delta()
        # 计算新的位置
        new_pos = self._last_pan_pos + delta
        
        # 设置水平滚动条的值，以平移视图
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() - (new_pos.x() - self._last_pan_pos.x())
        )
        # 设置垂直滚动条的值，以平移视图
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() - (new_pos.y() - self._last_pan_pos.y())
        )
        
        # 更新上次滑动位置
        self._last_pan_pos = new_pos
        return True

    # 处理捏合手势
    def handlePinchGesture(self, gesture):
        # 获取捏合手势的缩放因子
        scale_factor = gesture.scaleFactor()
        # 获取手势的中心点
        center = gesture.centerPoint()
        
        # 如果捏合手势刚开始，记录捏合中心点在场景中的位置
        if gesture.state() == QtCore.Qt.GestureState.GestureStarted:
            self._pinch_center = self.mapToScene(center.toPoint())
        
        # 如果缩放因子不为1，则进行缩放操作
        if scale_factor != 1:
            self.scale(scale_factor, scale_factor)
            # 更新缩放级别
            self._zoom += (scale_factor - 1)
        
        # 如果捏合手势结束，清空捏合中心点
        if gesture.state() == QtCore.Qt.GestureState.GestureFinished:
            self._pinch_center = QtCore.QPointF()
        
        return True

    # 处理滚轮事件
    def wheelEvent(self, event):
        # 如果视图中有照片
        if self.hasPhoto():
            # 如果同时按下了Ctrl键
            if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                # 使用Ctrl + 滚轮进行缩放
                factor = 1.25
                if event.angleDelta().y() > 0:
                    self.scale(factor, factor)
                    self._zoom += 1
                else:
                    self.scale(1 / factor, 1 / factor)
                    self._zoom -= 1
            else:
                # 没有按下Ctrl键时进行滚动操作
                super().wheelEvent(event)

    # 将图像适配到视图中
    def fitInView(self):
        # 获取图像的矩形区域
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            # 设置场景的矩形区域
            self.setSceneRect(rect)
            if self.hasPhoto():
                # 计算单位矩形的大小
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                # 获取视口和场景的矩形区域
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                # 计算缩放因子，使图像适应视图大小
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
                # 将视图中心设置为图像中心
                self.centerOn(rect.center())

    # 设置当前工具
    def set_tool(self, tool: str):
        self._current_tool = tool
        
        # 如果工具是平移
        if tool == 'pan':
            # 设置拖拽模式为手型滚动拖拽
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        elif tool in ['brush', 'eraser']:
            # 如果工具是画笔或橡皮擦，设置拖拽模式为无拖拽
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            # 根据工具设置光标形状
            if tool == 'brush':
                self.setCursor(self._brush_cursor)
            else:
                self.setCursor(self._eraser_cursor)
        else:
            # 其他情况下设置拖拽模式为无拖拽
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
    # 如果存在选中的矩形对象
    def delete_selected_rectangle(self):
        if self._selected_rect:
            # 从场景中移除选中的矩形对象
            self._scene.removeItem(self._selected_rect)
            # 从矩形对象列表中移除选中的矩形对象
            self._rectangles.remove(self._selected_rect)
            # 将选中的矩形对象设为 None，表示没有选中任何对象
            self._selected_rect = None

    # 处理鼠标按下事件
    def mousePressEvent(self, event):

        # 如果鼠标中键被按下
        if event.button() == QtCore.Qt.MiddleButton:
            # 开始平移操作
            self._panning = True
            # 记录平移操作的起始位置
            self._pan_start_pos = event.position()
            # 设置视口的鼠标形状为闭合手指形状
            self.viewport().setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            # 接受事件，不再传递
            event.accept()
            return
        
        # 如果当前工具为 'brush' 或 'eraser'，并且存在照片（场景中有内容）
        if self._current_tool in ['brush', 'eraser'] and self.hasPhoto():
            # 创建一个 QPainterPath 对象用于绘制路径
            self._drawing_path = QtGui.QPainterPath()
            # 将鼠标事件位置映射到场景坐标系中
            scene_pos = self.mapToScene(event.position().toPoint())
            # 如果场景中包含这个位置
            if self._photo.contains(scene_pos):
                # 将路径的起始点移动到这个位置
                self._drawing_path.moveTo(scene_pos)
                # 创建当前路径对象
                self._current_path = QtGui.QPainterPath()
                self._current_path.moveTo(scene_pos)
                # 在场景中添加当前路径，并设置画笔的颜色、大小等属性
                self._current_path_item = self._scene.addPath(self._current_path, 
                                                              QtGui.QPen(self._brush_color, self._brush_size, 
                                                                         QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, 
                                                                         QtCore.Qt.RoundJoin))

        # 如果当前工具为 'box' 并且存在照片（场景中有内容）
        if self._current_tool == 'box' and self.hasPhoto():
            # 将鼠标事件位置映射到场景坐标系中
            scene_pos = self.mapToScene(event.position().toPoint())
            # 如果照片包含这个位置
            if self._photo.contains(scene_pos):
                # 获取当前位置下的图形项
                item = self.itemAt(event.position().toPoint())
                # 如果图形项是矩形项，并且不是照片本身
                if isinstance(item, QtWidgets.QGraphicsRectItem) and item != self._photo:
                    # 选择这个矩形项
                    self.select_rectangle(item)
                    # 获取缩放手柄的位置
                    handle = self.get_resize_handle(item, scene_pos)
                    # 如果存在缩放手柄
                    if handle:
                        # 设置当前缩放手柄和起始位置
                        self._resize_handle = handle
                        self._resize_start = scene_pos
                    else:
                        # 开始拖拽操作
                        self._dragging = True
                        # 记录拖拽的起始位置和偏移量
                        self._drag_start = scene_pos
                        self._drag_offset = scene_pos - item.rect().topLeft()
                else:
                    # 取消选择所有项
                    self.deselect_all()
                    # 进入绘制矩形框模式
                    self._box_mode = True
                    # 记录起始点位置
                    self._start_point = scene_pos
                    # 创建当前矩形项并添加到场景中
                    self._current_rect = QtWidgets.QGraphicsRectItem(self._photo.mapRectToItem(self._photo, QtCore.QRectF(self._start_point, self._start_point)))
                    self._current_rect.setBrush(QtGui.QBrush(QtGui.QColor(255, 192, 203, 125)))  # Transparent pink
                    self._scene.addItem(self._current_rect)

        # 如果当前工具为 'pan'
        elif self._current_tool == 'pan':
            # 调用父类的鼠标按下事件处理方法
            super().mousePressEvent(event)
    # 处理鼠标移动事件，调用父类的同名方法处理事件
    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
    
        # 如果正在进行平移操作
        if self._panning:
            # 获取当前鼠标位置
            new_pos = event.position()
            # 计算鼠标移动的距离
            delta = new_pos - self._pan_start_pos
            # 调整水平滚动条的值
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            # 调整垂直滚动条的值
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            # 更新平移起始位置为当前位置
            self._pan_start_pos = new_pos
            # 接受该事件，处理完成
            event.accept()
            return
        
        # 如果当前工具是画笔或橡皮擦，并且当前路径有效
        if self._current_tool in ['brush', 'eraser'] and self._current_path:
            # 将事件位置映射到场景坐标系中
            scene_pos = self.mapToScene(event.position().toPoint())
            # 如果位置在照片范围内
            if self._photo.contains(scene_pos):
                # 将当前路径延伸至新位置
                self._current_path.lineTo(scene_pos)
                # 如果当前工具是画笔，更新当前路径项的路径
                if self._current_tool == 'brush':
                    self._current_path_item.setPath(self._current_path)
                # 如果当前工具是橡皮擦，在指定位置擦除
                elif self._current_tool == 'eraser':
                    self.erase_at(scene_pos)

        # 如果当前工具是矩形框
        if self._current_tool == 'box':
            # 将事件位置映射到场景坐标系中
            scene_pos = self.mapToScene(event.position().toPoint())
            # 如果处于绘制模式
            if self._box_mode:
                # 根据起始点和当前点创建矩形并规范化
                end_point = self.constrain_point(scene_pos)
                rect = QtCore.QRectF(self._start_point, end_point).normalized()
                # 将当前矩形设置为照片坐标系中的规范化矩形
                self._current_rect.setRect(self._photo.mapRectToItem(self._photo, rect))
            # 如果存在选定的矩形
            elif self._selected_rect:
                # 如果在调整大小手柄上
                if self._resize_handle:
                    # 调整矩形大小
                    self.resize_rectangle(scene_pos)
                # 如果在拖动矩形
                elif self._dragging:
                    # 移动矩形
                    self.move_rectangle(scene_pos)
                else:
                    # 获取适合矩形框工具的鼠标光标
                    cursor = self.get_cursor(self._selected_rect, scene_pos)
                    # 设置视口光标
                    self.viewport().setCursor(cursor)
            else:
                # 获取适合矩形框工具的鼠标光标
                cursor = self.get_cursor_for_box_tool(scene_pos)
                # 设置视口光标
                self.viewport().setCursor(cursor)

    # 移动选定矩形的方法，scene_pos 为新的位置在场景坐标系中的位置
    def move_rectangle(self, scene_pos: QtCore.QPointF):
        # 获取新的位置
        new_pos = scene_pos
        # 计算新的矩形的左上角位置
        new_top_left = new_pos - self._drag_offset
        # 创建新的矩形并约束在有效范围内
        new_rect = QtCore.QRectF(new_top_left, self._selected_rect.rect().size())
        constrained_rect = self.constrain_rect(new_rect)
        # 设置选定矩形的位置
        self._selected_rect.setRect(constrained_rect)
        
        # 发射 rectangle_moved 信号，通知矩形已移动
        self.rectangle_changed.emit(constrained_rect)
    # 当鼠标释放时触发的事件处理函数
    def mouseReleaseEvent(self, event):
        # 如果释放的是中间按钮
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            # 停止平移操作，并设置鼠标指针为箭头形状
            self._panning = False
            self.viewport().setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        
        # 如果当前工具是画笔或橡皮擦
        if self._current_tool in ['brush', 'eraser']:
            # 如果有当前路径项，则将其添加到绘制项列表中
            if self._current_path_item:
                self._drawing_items.append(self._current_path_item)
            # 重置当前路径和路径项
            self._current_path = None
            self._current_path_item = None
            # 将绘制项列表添加到撤销画笔堆栈中，并清空绘制项列表
            self._undo_brush_stack.append(self._drawing_items)
            self._drawing_items = []
            # 清空重做画笔堆栈
            self._redo_brush_stack.clear()

        # 如果当前工具是矩形框选
        if self._current_tool == 'box':
            # 如果处于框选模式
            if self._box_mode:
                self._box_mode = False
                # 如果存在当前矩形并且宽度和高度大于0，则将其添加到矩形列表中
                if self._current_rect and self._current_rect.rect().width() > 0 and self._current_rect.rect().height() > 0:
                    self._rectangles.append(self._current_rect)
                else:
                    self._scene.removeItem(self._current_rect)
                self._current_rect = None
            else:
                # 如果不处于框选模式，则重置拖动和调整大小的状态
                self._dragging = False
                self._drag_offset = None
                self._resize_handle = None
                self._resize_start = None

        # 如果当前工具是平移
        elif self._current_tool == 'pan':
            # 调用父类的鼠标释放事件处理函数
            super().mouseReleaseEvent(event)
            
    # 在指定位置使用橡皮擦进行擦除操作
    def erase_at(self, pos: QtCore.QPointF):
        # 创建椭圆形状的橡皮擦路径
        erase_path = QtGui.QPainterPath()
        erase_path.addEllipse(pos, self._eraser_size, self._eraser_size)
        
        # 增加橡皮擦路径的精度
        precise_erase_path = QtGui.QPainterPath()
        for i in range(36):
            angle = i * 10 * 3.14159 / 180
            point = QtCore.QPointF(pos.x() + self._eraser_size * math.cos(angle),
                                pos.y() + self._eraser_size * math.sin(angle))
            if i == 0:
                precise_erase_path.moveTo(point)
            else:
                precise_erase_path.lineTo(point)
        precise_erase_path.closeSubpath()

        # 获取与橡皮擦路径相交的场景中的图形项，并进行擦除操作
        items = self._scene.items(erase_path)
        for item in items:
            if isinstance(item, QtWidgets.QGraphicsPathItem) and item != self._photo:
                path = item.path()
                intersected_path = path.intersected(precise_erase_path)
                if not intersected_path.isEmpty():
                    new_path = QtGui.QPainterPath(path)
                    new_path = new_path.subtracted(intersected_path)
                    item.setPath(new_path)
                    # 如果图形项的路径为空，则移除该图形项
                    if new_path.isEmpty():
                        self._scene.removeItem(item)
                        if item in self._drawing_items:
                            self._drawing_items.remove(item)
    # 保存当前场景中所有的画笔路径信息，并返回列表
    def save_brush_strokes(self):
        brush_strokes = []
        # 遍历场景中的所有项
        for item in self._scene.items():
            # 如果是 QGraphicsPathItem 类型的项
            if isinstance(item, QtWidgets.QGraphicsPathItem):
                # 将路径、画笔颜色（包括透明度）、填充颜色和线宽保存到字典中
                brush_strokes.append({
                    'path': item.path(),
                    'pen': item.pen().color().name(QtGui.QColor.HexArgb),  # 以十六进制ARGB格式保存画笔颜色
                    'brush': item.brush().color().name(QtGui.QColor.HexArgb),  # 以十六进制ARGB格式保存填充颜色
                    'width': item.pen().width()
                })
        return brush_strokes

    # 加载给定的画笔路径信息列表并添加到场景中
    def load_brush_strokes(self, brush_strokes: List[Dict]):
        # 清空当前场景中的所有画笔路径
        self.clear_brush_strokes()
        # 将画笔路径信息列表反转顺序
        reversed_brush_strokes = brush_strokes[::-1]
        # 遍历反转后的画笔路径信息列表
        for stroke in reversed_brush_strokes:
            # 创建画笔对象并设置颜色、线宽和样式
            pen = QtGui.QPen()
            pen.setColor(QtGui.QColor(stroke['pen']))
            pen.setWidth(stroke['width'])
            pen.setStyle(QtCore.Qt.SolidLine)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            pen.setJoinStyle(QtCore.Qt.RoundJoin)

            # 创建填充颜色对象
            brush = QtGui.QBrush(QtGui.QColor(stroke['brush']))
            brush_color = QtGui.QColor(stroke['brush'])
            # 如果填充颜色为半透明红色
            if brush_color == "#80ff0000": # generated 
                # 将路径和画笔、填充颜色添加到场景中的 QGraphicsPathItem 对象
                path_item = self._scene.addPath(stroke['path'], pen, brush)
            else:
                # 将路径和画笔添加到场景中的 QGraphicsPathItem 对象
                path_item = self._scene.addPath(stroke['path'], pen)
            # 将添加的路径项存储到绘制项列表中
            self._drawing_items.append(path_item)
            # 将绘制项列表存储到撤销操作的堆栈中
            self._undo_brush_stack.append(self._drawing_items)
            # 清空绘制项列表
            self._drawing_items = []
            # 清空重做操作的堆栈（已注释）
            # self._redo_brush_stack.clear()

    # 撤销最后一次绘制的画笔路径
    def undo_brush_stroke(self):
        # 如果撤销操作堆栈不为空
        if self._undo_brush_stack:
            # 弹出最近一次绘制的路径项列表
            items = self._undo_brush_stack.pop()
            # 从场景中移除每个路径项
            for item in items:
                self._scene.removeItem(item)
            # 将移除的路径项列表存储到重做操作的堆栈中
            self._redo_brush_stack.append(items)

    # 重做上一次撤销的画笔路径
    def redo_brush_stroke(self):
        # 如果重做操作堆栈不为空
        if self._redo_brush_stack:
            # 弹出最近一次撤销的路径项列表
            items = self._redo_brush_stack.pop()
            # 将每个路径项添加回场景中
            for item in items:
                self._scene.addItem(item)
            # 将添加的路径项列表存储到撤销操作的堆栈中
            self._undo_brush_stack.append(items)

    # 清除当前场景中所有的画笔路径
    def clear_brush_strokes(self):
        items_to_remove = []
        # 遍历场景中的所有项
        for item in self._scene.items():
            # 如果是 QGraphicsPathItem 类型的项且不是照片项
            if isinstance(item, QtWidgets.QGraphicsPathItem) and item != self._photo:
                # 将该项添加到待移除列表中
                items_to_remove.append(item)
        
        # 从场景中移除待移除列表中的所有项
        for item in items_to_remove:
            self._scene.removeItem(item)
        
        # 清空绘制项列表、撤销操作堆栈和重做操作堆栈
        self._drawing_items.clear()
        self._undo_brush_stack.clear()
        self._redo_brush_stack.clear()
        
        # 更新场景以反映变更
        self._scene.update()
    # 获取用于调整大小的手柄的函数，根据指定的矩形和位置返回相应的手柄标识符
    def get_resize_handle(self, rect: QtWidgets.QGraphicsRectItem, pos: QtCore.QPointF):
        # 定义手柄的大小
        handle_size = 20
        # 获取矩形的边界矩形
        rect_rect = rect.rect()
        # 获取矩形的左上角和右下角坐标
        top_left = rect_rect.topLeft()
        bottom_right = rect_rect.bottomRight()
        
        # 定义包含所有可能手柄的字典
        handles = {
            'top_left': QtCore.QRectF(top_left.x() - handle_size/2, top_left.y() - handle_size/2, handle_size, handle_size),
            'top_right': QtCore.QRectF(bottom_right.x() - handle_size/2, top_left.y() - handle_size/2, handle_size, handle_size),
            'bottom_left': QtCore.QRectF(top_left.x() - handle_size/2, bottom_right.y() - handle_size/2, handle_size, handle_size),
            'bottom_right': QtCore.QRectF(bottom_right.x() - handle_size/2, bottom_right.y() - handle_size/2, handle_size, handle_size),
            'top': QtCore.QRectF(top_left.x(), top_left.y() - handle_size/2, rect_rect.width(), handle_size),
            'bottom': QtCore.QRectF(top_left.x(), bottom_right.y() - handle_size/2, rect_rect.width(), handle_size),
            'left': QtCore.QRectF(top_left.x() - handle_size/2, top_left.y(), handle_size, rect_rect.height()),
            'right': QtCore.QRectF(bottom_right.x() - handle_size/2, top_left.y(), handle_size, rect_rect.height()),
        }
        
        # 遍历所有手柄，检查位置是否在某个手柄的范围内
        for handle, handle_rect in handles.items():
            if handle_rect.contains(pos):
                return handle  # 返回匹配的手柄标识符
        return None  # 如果没有匹配的手柄，则返回 None

    # 获取用于矩形工具的光标形状的函数，根据指定的矩形和位置返回相应的光标形状
    def get_cursor(self, rect: QtWidgets.QGraphicsRectItem, pos: QtCore.QPointF):
        # 获取调整大小的手柄标识符
        handle = self.get_resize_handle(rect, pos)
        if handle:
            # 定义不同手柄对应的光标形状字典
            cursors = {
                'top_left': QtCore.Qt.CursorShape.SizeFDiagCursor,
                'top_right': QtCore.Qt.CursorShape.SizeBDiagCursor,
                'bottom_left': QtCore.Qt.CursorShape.SizeBDiagCursor,
                'bottom_right': QtCore.Qt.CursorShape.SizeFDiagCursor,
                'top': QtCore.Qt.CursorShape.SizeVerCursor,
                'bottom': QtCore.Qt.CursorShape.SizeVerCursor,
                'left': QtCore.Qt.CursorShape.SizeHorCursor,
                'right': QtCore.Qt.CursorShape.SizeHorCursor,
            }
            return cursors.get(handle, QtCore.Qt.CursorShape.ArrowCursor)  # 返回匹配的光标形状，若无匹配则默认为箭头形状
        elif rect.rect().contains(pos):
            return QtCore.Qt.CursorShape.SizeAllCursor  # 当光标在矩形内部时返回移动光标
        return QtCore.Qt.CursorShape.ArrowCursor  # 默认返回箭头光标，当光标不在矩形内部时

    # 获取用于框选工具的光标形状的函数，根据指定位置返回相应的光标形状
    def get_cursor_for_box_tool(self, pos: QtCore.QPointF):
        if self._photo.contains(pos):  # 如果位置在图像范围内
            for rect in self._rectangles:
                if rect.rect().contains(pos):
                    return QtCore.Qt.CursorShape.PointingHandCursor  # 当光标悬停在矩形框上时返回点击光标
            return QtCore.Qt.CursorShape.CrossCursor  # 当光标在图像内但不在矩形框上时返回十字光标
        return QtCore.Qt.CursorShape.ArrowCursor  # 当光标在图像外部时返回默认箭头光标
    # 根据鼠标位置调整选定矩形的大小
    def resize_rectangle(self, pos: QtCore.QPointF):
        # 如果没有选定的矩形或者没有选定的调整手柄，则退出函数
        if not self._selected_rect or not self._resize_handle:
            return

        # 获取当前选定矩形的位置和大小
        rect = self._selected_rect.rect()
        # 计算鼠标移动的距离
        dx = pos.x() - self._resize_start.x()
        dy = pos.y() - self._resize_start.y()

        # 创建一个新的矩形对象来存储调整后的矩形
        new_rect = QtCore.QRectF(rect)

        # 根据调整手柄的位置，调整新矩形的边界
        if self._resize_handle in ['top_left', 'left', 'bottom_left']:
            new_rect.setLeft(rect.left() + dx)
        if self._resize_handle in ['top_left', 'top', 'top_right']:
            new_rect.setTop(rect.top() + dy)
        if self._resize_handle in ['top_right', 'right', 'bottom_right']:
            new_rect.setRight(rect.right() + dx)
        if self._resize_handle in ['bottom_left', 'bottom', 'bottom_right']:
            new_rect.setBottom(rect.bottom() + dy)

        # 确保矩形不会被反转到内部
        if new_rect.width() < 10:
            if 'left' in self._resize_handle:
                new_rect.setLeft(new_rect.right() - 10)
            else:
                new_rect.setRight(new_rect.left() + 10)
        if new_rect.height() < 10:
            if 'top' in self._resize_handle:
                new_rect.setTop(new_rect.bottom() - 10)
            else:
                new_rect.setBottom(new_rect.top() + 10)

        # 对新矩形进行约束处理，确保在照片区域内
        constrained_rect = self.constrain_rect(new_rect)
        # 更新选定矩形的位置和大小
        self._selected_rect.setRect(constrained_rect)
        self._resize_start = pos

        # 发射矩形调整完成的信号
        self.rectangle_changed.emit(constrained_rect)

    # 对矩形进行约束处理，确保在照片区域内
    def constrain_rect(self, rect: QtCore.QRectF):
        photo_rect = self._photo.boundingRect()
        new_x = max(0, min(rect.x(), photo_rect.width() - rect.width()))
        new_y = max(0, min(rect.y(), photo_rect.height() - rect.height()))
        return QtCore.QRectF(new_x, new_y, rect.width(), rect.height())

    # 对点进行约束处理，确保在照片区域内
    def constrain_point(self, point: QtCore.QPointF):
        return QtCore.QPointF(
            max(0, min(point.x(), self._photo.pixmap().width())),
            max(0, min(point.y(), self._photo.pixmap().height()))
        )

    # 选定指定的矩形，并设置为选中状态
    def select_rectangle(self, rect: QtWidgets.QGraphicsRectItem):
        # 取消所有选定状态
        self.deselect_all()
        # 设置选定矩形的填充颜色为半透明红色
        rect.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 100)))
        self._selected_rect = rect

        # 发射选定矩形信号
        self.rectangle_selected.emit(rect.rect())

    # 取消所有矩形的选定状态
    def deselect_all(self):
        for rect in self._rectangles:
            # 设置所有矩形的填充颜色为半透明粉红色
            rect.setBrush(QtGui.QBrush(QtGui.QColor(255, 192, 203, 125)))
        self._selected_rect = None

    # 获取所有矩形的属性信息列表
    def get_rectangle_properties(self):
        return [
            {
                'x': rect.rect().x(),
                'y': rect.rect().y(),
                'width': rect.rect().width(),
                'height': rect.rect().height(),
                'selected': rect == self._selected_rect
            }
            for rect in self._rectangles
        ]
    def get_rectangle_coordinates(self):
        """获取矩形框的坐标列表。

        返回一个列表，其中每个元素是一个包含矩形左上角和右下角坐标的元组。
        """
        return [
            (
                int(rect.rect().x()),
                int(rect.rect().y()),
                int(rect.rect().x() + rect.rect().width()),
                int(rect.rect().y() + rect.rect().height())
            )
            for rect in self._rectangles
        ]
    
    def get_cv2_image(self):
        """获取当前加载的图像作为 cv2 格式的图像。

        如果当前没有加载图像，则返回 None。否则，将图像转换为 RGB888 格式的 cv2 图像并返回。
        """
        if self._photo.pixmap() is None:
            return None

        qimage = self._photo.pixmap().toImage()
        qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGB888)

        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()

        byte_count = qimage.sizeInBytes()
        expected_size = height * bytes_per_line  # bytes per line can include padding

        if byte_count != expected_size:
            print(f"QImage sizeInBytes: {byte_count}, Expected size: {expected_size}")
            print(f"Image dimensions: ({width}, {height}), Format: {qimage.format()}")
            raise ValueError(f"Byte count mismatch: got {byte_count} but expected {expected_size}")

        ptr = qimage.bits()

        # Convert memoryview to a numpy array considering the complete data with padding
        arr = np.array(ptr).reshape((height, bytes_per_line))

        # Exclude the padding bytes, keeping only the relevant image data
        arr = arr[:, :width * 3]

        # Reshape to the correct dimensions without the padding bytes
        arr = arr.reshape((height, width, 3))

        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    
    def display_cv2_image(self, cv2_img: np.ndarray):
        """显示给定的 cv2 格式图像。

        将给定的 cv2 图像转换为 QtGui.QPixmap，并设置为当前场景的图像显示。
        """
        height, width, channel = cv2_img.shape
        bytes_per_line = 3 * width
        qimage = QtGui.QImage(cv2_img.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.setPhoto(pixmap)

    def clear_scene(self):
        """清空场景。

        移除所有图形项、清空矩形列表并重置选择的矩形。
        """
        self._scene.clear()
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._photo.setShapeMode(QtWidgets.QGraphicsPixmapItem.BoundingRectShape)
        self._scene.addItem(self._photo)
        self._rectangles = []
        self._selected_rect = None

    def clear_rectangles(self):
        """清空矩形列表。

        从场景中移除所有矩形图形项，并清空矩形列表。
        """
        for rect in self._rectangles:
            self._scene.removeItem(rect)
        self._rectangles.clear()
        self._selected_rect = None

    def setPhoto(self, pixmap: QtGui.QPixmap =None):
        """设置照片。

        将给定的 QtGui.QPixmap 设置为当前的照片，并根据需要调整视图以适应照片大小。
        如果没有给定照片或照片为空，则将场景标记为空。
        """
        self.clear_scene()
        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
            self.fitInView()
        else:
            self._empty = True
        self._zoom = 0

    def has_drawn_elements(self):
        """检查场景中是否存在绘制元素。

        遍历场景中的图形项，如果发现任何 QGraphicsPathItem 类型的项（不包括照片自身），则返回 True。
        否则返回 False。
        """
        for item in self._scene.items():
            if isinstance(item, QtWidgets.QGraphicsPathItem):
                if item != self._photo:
                    return True
        return False
    # 从笔划生成遮罩图像的方法，如果没有照片则返回空
    def generate_mask_from_strokes(self):
        if not self.hasPhoto():
            return None

        # 获取图像的尺寸
        image_rect = self._photo.boundingRect()
        width = int(image_rect.width())
        height = int(image_rect.height())

        # 创建两个空白的遮罩图像
        human_mask = np.zeros((height, width), dtype=np.uint8)
        generated_mask = np.zeros((height, width), dtype=np.uint8)

        # 创建两个用于分别绘制路径的 QImage
        human_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_Grayscale8)
        generated_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_Grayscale8)
        human_qimage.fill(QtGui.QColor(0, 0, 0))
        generated_qimage.fill(QtGui.QColor(0, 0, 0))

        # 创建用于两个 QImage 的 QPainters
        human_painter = QtGui.QPainter(human_qimage)
        generated_painter = QtGui.QPainter(generated_qimage)
        
        # 设置绘制笔的颜色和大小
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255), self._brush_size)
        human_painter.setPen(pen)
        generated_painter.setPen(pen)
        
        # 设置绘制笔的刷子颜色
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        human_painter.setBrush(brush)
        generated_painter.setBrush(brush)

        # 遍历场景中的所有项
        for item in self._scene.items():
            # 如果是 QGraphicsPathItem 并且不是照片本身
            if isinstance(item, QtWidgets.QGraphicsPathItem) and item != self._photo:
                # 获取路径项的刷子颜色
                brush_color = QtGui.QColor(item.brush().color().name(QtGui.QColor.HexArgb))
                # 如果刷子颜色为生成的笔划颜色
                if brush_color == "#80ff0000":  # generated stroke
                    generated_painter.drawPath(item.path())
                else:  # 人工绘制的笔划
                    human_painter.drawPath(item.path())

        # 结束绘制
        human_painter.end()
        generated_painter.end()

        # 将 QImage 转换为 numpy 数组，考虑填充
        bytes_per_line = human_qimage.bytesPerLine()
        human_ptr = human_qimage.constBits()
        generated_ptr = generated_qimage.constBits()
        human_arr = np.array(human_ptr).reshape(height, bytes_per_line)
        generated_arr = np.array(generated_ptr).reshape(height, bytes_per_line)
        
        # 去除填充字节，保留有效的图像数据
        human_mask = human_arr[:, :width]
        generated_mask = generated_arr[:, :width]

        # 对人工绘制的笔划应用膨胀操作
        kernel = np.ones((5,5), np.uint8)
        human_mask = cv2.dilate(human_mask, kernel, iterations=2)

        # 合并两个遮罩图像
        final_mask = cv2.bitwise_or(human_mask, generated_mask)

        return final_mask
    # 生成修复图像所需的掩膜
    def get_mask_for_inpainting(self):
        # 从用户绘制的笔画生成掩膜
        mask = self.generate_mask_from_strokes()

        if mask is None:
            return None

        # 获取当前加载的图像
        cv2_image = self.get_cv2_image()

        if cv2_image is None:
            return None

        # 确保掩膜和图像具有相同的尺寸
        mask = cv2.resize(mask, (cv2_image.shape[1], cv2_image.shape[0]))

        return mask
    
    # 绘制分割线
    def draw_segmentation_lines(self, segmentation_points: np.ndarray, layers: int = 1, scale_factor: float = 0.95, smoothness: float = 0.35):
        # 检查是否加载了照片
        if not self.hasPhoto():
            print("No photo loaded.")
            return

        # 检查分割点数量是否足够
        if len(segmentation_points) < 3:
            print("Not enough points to create a filled area.")
            return

        # 计算分割点的质心
        centroid = np.mean(segmentation_points, axis=0)

        # 将分割点缩放至质心
        scaled_points = (segmentation_points - centroid) * scale_factor + centroid

        # 创建平滑曲线的 QPainterPath
        path = QtGui.QPainterPath()
        
        # 将路径移动到第一个缩放后的分割点
        path.moveTo(scaled_points[0][0], scaled_points[0][1])
        
        # 函数：计算控制点
        def get_control_points(p1, p2, p3, smoothness):
            vec1 = p2 - p1
            vec2 = p3 - p2
            d1 = np.linalg.norm(vec1)
            d2 = np.linalg.norm(vec2)
            
            c1 = p2 - vec1 * smoothness
            c2 = p2 + vec2 * smoothness
            
            return c1, c2

        # 绘制平滑曲线
        for i in range(len(scaled_points)):
            p1 = scaled_points[i]
            p2 = scaled_points[(i + 1) % len(scaled_points)]
            p3 = scaled_points[(i + 2) % len(scaled_points)]
            
            c1, c2 = get_control_points(p1, p2, p3, smoothness)
            
            path.cubicTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])

        # 创建路径项并添加到场景中
        fill_color = QtGui.QColor(255, 0, 0, 128)  # 半透明红色
        outline_color = QtGui.QColor(255, 0, 0)  # 实线红色

        for _ in range(layers):
            path_item = QtWidgets.QGraphicsPathItem(path)
            path_item.setPen(QtGui.QPen(outline_color, 2, QtCore.Qt.SolidLine))
            path_item.setBrush(QtGui.QBrush(fill_color))
            self._scene.addItem(path_item)
            self._drawing_items.append(path_item)  # 添加到绘制项目列表，用于保存

        # 确保填充区域可见
        self._scene.update()
    # 加载给定状态到当前对象中，用于恢复视图状态
    def load_state(self, state: Dict):
        # 设置图形变换矩阵
        self.setTransform(QtGui.QTransform(*state['transform']))
        # 在指定点居中显示视图
        self.centerOn(QtCore.QPointF(*state['center']))
        # 设置场景矩形范围
        self.setSceneRect(QtCore.QRectF(*state['scene_rect']))
        
        # 遍历状态中的矩形数据，创建矩形图形项，并设置画刷颜色为透明粉红色
        for rect_data in state['rectangles']:
            rect_item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(*rect_data), self._photo)
            rect_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 192, 203, 125)))  # Transparent pink
            self._rectangles.append(rect_item)

    # 保存当前对象的状态信息，用于序列化
    def save_state(self):
        # 获取当前的图形变换矩阵
        transform = self.transform()
        # 将视口中心点映射到场景坐标系中，获取视口中心点坐标
        center = self.mapToScene(self.viewport().rect().center())
        # 返回包含当前状态信息的字典
        return {
            'rectangles': [rect.rect().getRect() for rect in self._rectangles],  # 获取所有矩形的矩形数据
            'transform': (
                transform.m11(), transform.m12(), transform.m13(),
                transform.m21(), transform.m22(), transform.m23(),
                transform.m31(), transform.m32(), transform.m33()
            ),
            'center': (center.x(), center.y()),  # 场景中心点坐标
            'scene_rect': (self.sceneRect().x(), self.sceneRect().y(), 
                           self.sceneRect().width(), self.sceneRect().height())  # 场景矩形的位置和大小
        }

    # 创建并返回一个用于修复图标的光标对象
    def create_inpaint_cursor(self, cursor_type, size):
        from PySide6.QtGui import QPixmap, QPainter, QBrush, QColor, QCursor
        from PySide6.QtCore import Qt

        # 确保光标大小至少为1像素
        size = max(1, size)
        
        # 创建指定大小的透明像素图
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)

        # 根据指定的光标类型设置画笔和画刷
        if cursor_type == "brush":
            painter.setBrush(QBrush(QColor(255, 0, 0, 127)))  # 红色，半透明
            painter.setPen(Qt.PenStyle.NoPen)  # 不绘制边框线
        elif cursor_type == "eraser":
            painter.setBrush(QBrush(QColor(0, 0, 0, 0)))  # 完全透明
            painter.setPen(QColor(0, 0, 0, 127))  # 灰色，半透明边框
        else:
            painter.setBrush(QBrush(QColor(0, 0, 0, 127)))  # 默认为半透明黑色
            painter.setPen(Qt.PenStyle.NoPen)

        # 在图标上绘制一个椭圆
        painter.drawEllipse(0, 0, (size - 1), (size - 1))
        painter.end()

        # 创建并返回一个新的光标对象
        return QCursor(pixmap, size // 2, size // 2)
    
    # 设置笔刷工具的大小，并更新相应的修复图标光标
    def set_brush_size(self, size):
        self._brush_size = size
        self._brush_cursor = self.create_inpaint_cursor("brush", size)
        # 如果当前工具是笔刷，则设置当前光标为笔刷修复图标
        if self._current_tool == "brush":
            self.setCursor(self._brush_cursor)

    # 设置橡皮擦工具的大小，并更新相应的修复图标光标
    def set_eraser_size(self, size):
        self._eraser_size = size
        self._eraser_cursor = self.create_inpaint_cursor("eraser", size)
        # 如果当前工具是橡皮擦，则设置当前光标为橡皮擦修复图标
        if self._current_tool == "eraser":
            self.setCursor(self._eraser_cursor)
```