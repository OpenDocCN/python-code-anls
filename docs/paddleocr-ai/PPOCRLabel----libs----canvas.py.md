# `.\PaddleOCR\PPOCRLabel\libs\canvas.py`

```
# 版权声明，允许在特定条件下使用和分发软件
# 导入所需的库和模块
import copy

from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QPoint
from PyQt5.QtGui import QPainter, QBrush, QColor, QPixmap
from PyQt5.QtWidgets import QWidget, QMenu, QApplication
from libs.shape import Shape
from libs.utils import distance

# 定义不同的鼠标光标类型常量
CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor

# 定义一个名为 Canvas 的类，继承自 QWidget
class Canvas(QWidget):
    # 定义信号
    zoomRequest = pyqtSignal(int)
    scrollRequest = pyqtSignal(int, int)
    newShape = pyqtSignal()
    # selectionChanged = pyqtSignal(bool)
    selectionChanged = pyqtSignal(list)
    shapeMoved = pyqtSignal()
    drawingPolygon = pyqtSignal(bool)

    # 定义常量 CREATE 和 EDIT，表示创建和编辑状态
    CREATE, EDIT = list(range(2))
    # 定义一个变量 _fill_drawing，用于标记是否绘制阴影效果
    _fill_drawing = False
    # 设置一个常量 epsilon 为 5.0
    epsilon = 5.0

    # 初始化 Canvas 类的实例，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类的构造函数
        super(Canvas, self).__init__(*args, **kwargs)
        # 初始化本地状态
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []
        self.selectedShape = None  # 保存选定的形状
        self.selectedShapesCopy = []
        self.drawingLineColor = QColor(0, 0, 255)
        self.drawingRectColor = QColor(0, 0, 255)
        self.line = Shape(line_color=self.drawingLineColor)
        self.prevPoint = QPointF()
        self.offsets = QPointF(), QPointF()
        self.scale = 1.0
        self.pixmap = QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.hVertex = None
        self._painter = QPainter()
        self._cursor = CURSOR_DEFAULT
        # 菜单
        self.menus = (QMenu(), QMenu())
        # 设置小部件选项
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)
        self.verified = False
        self.drawSquare = False
        self.fourpoint = True  # 添加
        self.pointnum = 0
        self.movingShape = False
        self.selectCountShape = False

        # 初始化平移
        self.pan_initial_pos = QPoint()

        # 锁定形状相关
        self.lockedShapes = []
        self.isInTheSameImage = False

    # 设置绘制颜色
    def setDrawingColor(self, qColor):
        self.drawingLineColor = qColor
        self.drawingRectColor = qColor

    # 鼠标进入事件
    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    # 鼠标离开事件
    def leaveEvent(self, ev):
        self.restoreCursor()

    # 失去焦点事件
    def focusOutEvent(self, ev):
        self.restoreCursor()

    # 检查形状是否可见
    def isVisible(self, shape):
        return self.visible.get(shape, True)

    # 是否处于绘制模式
    def drawing(self):
        return self.mode == self.CREATE

    # 是否处于编辑模式
    def editing(self):
        return self.mode == self.EDIT
    # 设置编辑模式，如果 value 为 True，则设置为编辑模式，否则设置为创建模式
    def setEditing(self, value=True):
        # 根据 value 的值设置编辑模式
        self.mode = self.EDIT if value else self.CREATE
        # 如果不是编辑模式，则取消高亮显示，取消选择的形状
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()
        # 重置上一个点的位置
        self.prevPoint = QPointF()
        # 重绘界面
        self.repaint()

    # 取消高亮显示
    def unHighlight(self):
        # 如果存在高亮的形状，则清除高亮
        if self.hShape:
            self.hShape.highlightClear()
        # 重置高亮的顶点和形状
        self.hVertex = self.hShape = None

    # 返回是否选择了顶点
    def selectedVertex(self):
        return self.hVertex is not None
    # 处理鼠标按下事件
    def mousePressEvent(self, ev):
        # 将鼠标点击位置转换为画布上的坐标
        pos = self.transformPos(ev.pos())
        # 如果是左键点击
        if ev.button() == Qt.LeftButton:
            # 如果正在绘制
            if self.drawing():
                # 如果当前有选中的形状
                if self.current:
                    # 如果是四点形状
                    if self.fourpoint: # ADD IF
                        # 将点添加到当前形状中
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        # 如果当前形状是封闭的
                        if self.current.isClosed():
                            self.finalise()
                    # 如果是绘制正方形
                    elif self.drawSquare:
                        # 确保当前形状只有一个点
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                # 如果当前没有选中形状且点击位置在画布内
                elif not self.outOfPixmap(pos):
                    # 创建新的形状
                    self.current = Shape()
                    self.current.addPoint(pos)
                    self.line.points = [pos, pos]
                    self.setHiding()
                    self.drawingPolygon.emit(True)
                    self.update()

            else:
                # 检查是否按下了 Ctrl 键
                group_mode = int(ev.modifiers()) == Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.pan_initial_pos = pos

        # 如果是右键点击且处于编辑模式
        elif ev.button() == Qt.RightButton and self.editing():
            # 检查是否按下了 Ctrl 键
            group_mode = int(ev.modifiers()) == Qt.ControlModifier
            self.selectShapePoint(pos, multiple_selection_mode=group_mode)
            self.prevPoint = pos
        # 更新画布
        self.update()
    # 当鼠标释放时触发的事件处理函数
    def mouseReleaseEvent(self, ev):
        # 如果释放的是右键
        if ev.button() == Qt.RightButton:
            # 根据是否有选中的形状复制来选择菜单
            menu = self.menus[bool(self.selectedShapesCopy)]
            # 恢复鼠标光标
            self.restoreCursor()
            # 如果菜单未执行且存在选中的形状复制
            if not menu.exec_(self.mapToGlobal(ev.pos()))\
               and self.selectedShapesCopy:
                # 取消移动操作，删除阴影复制
                # self.selectedShapeCopy = None
                self.selectedShapesCopy = []
                self.repaint()

        # 如果释放的是左键且存在选中的形状
        elif ev.button() == Qt.LeftButton and self.selectedShapes:
            # 如果选中了顶点
            if self.selectedVertex():
                self.overrideCursor(CURSOR_POINT)
            else:
                self.overrideCursor(CURSOR_GRAB)

        # 如果释放的是左键且不是四点标注模式
        elif ev.button() == Qt.LeftButton and not self.fourpoint:
            # 将鼠标位置转换为画布坐标
            pos = self.transformPos(ev.pos())
            # 如果正在绘制
            if self.drawing():
                self.handleDrawing(pos)
            else:
                # 平移操作
                QApplication.restoreOverrideCursor() # 恢复鼠标光标？

        # 如果正在移动形状且存在高亮的形状
        if self.movingShape and self.hShape:
            # 如果高亮的形状在形状列表中
            if self.hShape in self.shapes:
                index = self.shapes.index(self.hShape)
                # 如果形状的点集与备份的不同
                if (
                    self.shapesBackups[-1][index].points
                    != self.shapes[index].points
                ):
                    # 存储形状备份
                    self.storeShapes()
                    # 发送形状移动信号，连接到 PPOCRLabel.py 中的 updateBoxlist 函数
                    self.shapeMoved.emit()

                self.movingShape = False
    # 结束移动操作，可选择是否复制
    def endMove(self, copy=False):
        # 断言已选择形状和已选择形状的副本存在
        assert self.selectedShapes and self.selectedShapesCopy
        # 断言已选择形状和已选择形状的副本数量相等
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        # 如果选择复制
        if copy:
            # 遍历已选择形状的副本
            for i, shape in enumerate(self.selectedShapesCopy):
                # 设置当前形状的索引为已有形状数量
                shape.idx = len(self.shapes) # add current box index
                # 将形状添加到形状列表中
                self.shapes.append(shape)
                # 取消已选择形状的选中状态，并替换为新形状
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        # 如果不选择复制
        else:
            # 遍历已选择形状的副本
            for i, shape in enumerate(self.selectedShapesCopy):
                # 将已选择形状的点设置为副本形状的点
                self.selectedShapes[i].points = shape.points
        # 清空已选择形状的副本
        self.selectedShapesCopy = []
        # 重新绘制
        self.repaint()
        # 存储形状
        self.storeShapes()
        # 返回True
        return True

    # 隐藏背景形状
    def hideBackroundShapes(self, value):
        # 设置隐藏背景的值
        self.hideBackround = value
        # 如果存在已选择形状
        if self.selectedShapes:
            # 只有在存在当前选择时隐藏其他形状
            # 否则用户将无法选择形状
            self.setHiding(True)
            # 重新绘制
            self.repaint()
    # 处理绘图操作，根据当前状态和点的位置进行相应操作
    def handleDrawing(self, pos):
        # 如果当前存在图形且未达到最大点数
        if self.current and self.current.reachMaxPoints() is False:
            # 如果是四个点的图形
            if self.fourpoint:
                # 获取目标点位置
                targetPos = self.line[self.pointnum]
                # 将目标点添加到当前图形中
                self.current.addPoint(targetPos)
                print('current points in handleDrawing is ', self.line[self.pointnum])
                self.update()
                # 如果已经添加了四个点，完成图形绘制
                if self.pointnum == 3:
                    self.finalise()

            else:
                # 获取初始点位置
                initPos = self.current[0]
                print('initPos', self.current[0])
                minX = initPos.x()
                minY = initPos.y()
                targetPos = self.line[1]
                maxX = targetPos.x()
                maxY = targetPos.y()
                # 添加四个点构成矩形
                self.current.addPoint(QPointF(maxX, minY))
                self.current.addPoint(targetPos)
                self.current.addPoint(QPointF(minX, maxY))
                self.finalise()

        # 如果鼠标位置未超出图像范围
        elif not self.outOfPixmap(pos):
            print('release')
            # 创建新的图形并添加点
            self.current = Shape()
            self.current.addPoint(pos)
            self.line.points = [pos, pos]
            self.setHiding()
            self.drawingPolygon.emit(True)
            self.update()

    # 设置是否隐藏背景
    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    # 判断是否可以闭合图形
    def canCloseShape(self):
        return self.drawing() and self.current and len(self.current) > 2

    # 处理鼠标双击事件
    def mouseDoubleClickEvent(self, ev):
        # 至少需要四个点才能闭合图形，因为鼠标按下事件会在此事件之前添加一个额外点
        if self.canCloseShape() and len(self.current) > 3:
            if not self.fourpoint:
                self.current.popPoint()
            self.finalise()

    # 选择图形
    def selectShapes(self, shapes):
        for s in shapes: s.seleted = True
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()
    # 选择包含给定点的第一个创建的形状
    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        # 如果已选择顶点，则标记为选择
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
            return self.hVertex
        else:
            # 遍历形状列表，查找包含给定点的可见形状
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    # 计算偏移量
                    self.calculateOffsets(shape, point)
                    self.setHiding()
                    # 如果是多选模式，则添加到已选择形状列表
                    if multiple_selection_mode:
                        if shape not in self.selectedShapes: # list
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape]
                            )
                    else:
                        self.selectionChanged.emit([shape])
                    return
        # 取消选择形状
        self.deSelectShape()

    # 计算形状相对于给定点的偏移量
    def calculateOffsets(self, shape, point):
        rect = shape.boundingRect()
        x1 = rect.x() - point.x()
        y1 = rect.y() - point.y()
        x2 = (rect.x() + rect.width()) - point.x()
        y2 = (rect.y() + rect.height()) - point.y()
        self.offsets = QPointF(x1, y1), QPointF(x2, y2)

    # 将点 x,y 移动到画布边界内
    def snapPointToCanvas(self, x, y):
        """
        Moves a point x,y to within the boundaries of the canvas.
        :return: (x,y,snapped) where snapped is True if x or y were changed, False if not.
        """
        # 如果点超出画布边界，则将其移动到边界上
        if x < 0 or x > self.pixmap.width() or y < 0 or y > self.pixmap.height():
            x = max(x, 0)
            y = max(y, 0)
            x = min(x, self.pixmap.width())
            y = min(y, self.pixmap.height())
            return x, y, True

        return x, y, False
    # 限制移动顶点的方法，接受一个位置参数
    def boundedMoveVertex(self, pos):
        # 获取当前顶点和形状的索引
        index, shape = self.hVertex, self.hShape
        # 获取当前顶点的坐标
        point = shape[index]
        
        # 如果移动位置超出了图片范围
        if self.outOfPixmap(pos):
            # 获取图片大小
            size = self.pixmap.size()
            # 对移动位置进行裁剪，确保在图片范围内
            clipped_x = min(max(0, pos.x()), size.width())
            clipped_y = min(max(0, pos.y()), size.height())
            pos = QPointF(clipped_x, clipped_y)

        # 如果需要绘制正方形
        if self.drawSquare:
            # 计算对角点的索引
            opposite_point_index = (index + 2) % 4
            opposite_point = shape[opposite_point_index]

            # 计算移动的最小距离
            min_size = min(abs(pos.x() - opposite_point.x()), abs(pos.y() - opposite_point.y()))
            # 确定移动方向
            directionX = -1 if pos.x() - opposite_point.x() < 0 else 1
            directionY = -1 if pos.y() - opposite_point.y() < 0 else 1
            # 计算移动后的位置
            shiftPos = QPointF(opposite_point.x() + directionX * min_size - point.x(),
                               opposite_point.y() + directionY * min_size - point.y())
        else:
            # 计算移动后的位置
            shiftPos = pos - point

        # 如果形状是矩形
        if [shape[0].x(), shape[0].y(), shape[2].x(), shape[2].y()] \
                == [shape[3].x(),shape[1].y(),shape[1].x(),shape[3].y()]:
            # 移动当前顶点
            shape.moveVertexBy(index, shiftPos)
            # 计算左右顶点的索引
            lindex = (index + 1) % 4
            rindex = (index + 3) % 4
            lshift = None
            rshift = None
            # 根据当前顶点的位置确定左右顶点的移动方向
            if index % 2 == 0:
                rshift = QPointF(shiftPos.x(), 0)
                lshift = QPointF(0, shiftPos.y())
            else:
                lshift = QPointF(shiftPos.x(), 0)
                rshift = QPointF(0, shiftPos.y())
            # 移动左右顶点
            shape.moveVertexBy(rindex, rshift)
            shape.moveVertexBy(lindex, lshift)

        else:
            # 移动当前顶点
            shape.moveVertexBy(index, shiftPos)
    # 限制移动形状的方法，接受形状列表和位置参数
    def boundedMoveShape(self, shapes, pos):
        # 如果传入的形状不是列表，则将其转换为列表
        if type(shapes).__name__ != 'list': shapes = [shapes]
        # 如果位置超出了画布范围，则不需要移动形状，直接返回False
        if self.outOfPixmap(pos):
            return False  # No need to move
        # 计算第一个偏移后的位置
        o1 = pos + self.offsets[0]
        # 如果偏移后的位置超出了画布范围，则调整位置
        if self.outOfPixmap(o1):
            pos -= QPointF(min(0, o1.x()), min(0, o1.y()))
        # 计算第二个偏移后的位置
        o2 = pos + self.offsets[1]
        # 如果偏移后的位置超出了画布范围，则调整位置
        if self.outOfPixmap(o2):
            pos += QPointF(min(0, self.pixmap.width() - o2.x()),
                           min(0, self.pixmap.height() - o2.y()))
        # 计算位置变化量
        dp = pos - self.prevPoint
        # 如果位置有变化
        if dp:
            # 遍历选中的形状，移动形状并更新位置
            for shape in shapes:
                shape.moveBy(dp)
                shape.close()
            self.prevPoint = pos
            return True
        return False

    # 取消选中形状的方法
    def deSelectShape(self):
        # 如果有选中的形状
        if self.selectedShapes:
            # 取消选中所有形状
            for shape in self.selectedShapes: shape.selected=False
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.update()

    # 删除选中的形状的方法
    def deleteSelected(self):
        deleted_shapes = []
        # 如果有选中的形状
        if self.selectedShapes:
            # 遍历选中的形状，从形状列表中移除，并添加到已删除形状列表中
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()

        # 更新形状索引
        self.updateShapeIndex()

        return deleted_shapes

    # 存储形状的方法
    def storeShapes(self):
        shapesBackup = []
        # 备份当前所有形状
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        # 如果形状备份列表长度超过10，则保留最近的9个备份
        if len(self.shapesBackups) >= 10:
            self.shapesBackups = self.shapesBackups[-9:]
        # 将当前形状备份添加到形状备份列表中
        self.shapesBackups.append(shapesBackup)
    # 复制选定的形状
    def copySelectedShape(self):
        # 如果存在选定的形状
        if self.selectedShapes:
            # 复制选定的形状列表
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            # 对复制的形状进行边界移动
            self.boundedShiftShapes(self.selectedShapesCopy)
            # 结束移动操作，复制形状
            self.endMove(copy=True)
        # 返回选定的形状列表
        return self.selectedShapes

    # 对形状进行边界移动
    def boundedShiftShapes(self, shapes):
        # 尝试向一个方向移动，如果失败则尝试另一个方向
        # 如果两个方向都失败则放弃
        for shape in shapes:
            # 获取形状的起始点
            point = shape[0]
            # 设置偏移量
            offset = QPointF(5.0, 5.0)
            # 计算偏移量
            self.calculateOffsets(shape, point)
            # 设置前一个点
            self.prevPoint = point
            # 如果向偏移量方向移动失败，则尝试向相反方向移动
            if not self.boundedMoveShape(shape, point - offset):
                self.boundedMoveShape(shape, point + offset)

    # 返回填充绘图的状态
    def fillDrawing(self):
        return self._fill_drawing

    # 将点从窗口逻辑坐标转换为绘图逻辑坐标
    def transformPos(self, point):
        return point / self.scale - self.offsetToCenter()

    # 计算偏移量以将绘图居中
    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPointF(x, y)

    # 检查点是否超出绘图范围
    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w and 0 <= p.y() <= h)

    # 完成当前绘图操作
    def finalise(self):
        # 断言当前存在绘图对象
        assert self.current
        # 如果当前绘图对象的起始点和结束点相同，则取消绘图操作
        if self.current.points[0] == self.current.points[-1]:
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
            return

        # 封闭当前绘图对象
        self.current.close()
        # 设置当前绘图对象的索引并添加到形状列表中
        self.current.idx = len(self.shapes)
        self.shapes.append(self.current)
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()
    # 检查两个点之间的距离是否小于给定的阈值 epsilon
    def closeEnough(self, p1, p2):
        # 计算两点之间的距离
        #d = distance(p1 - p2)
        # 计算两点之间的曼哈顿距离
        #m = (p1-p2).manhattanLength()
        # 打印距离和曼哈顿距离的差值
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # 返回两点之间距离是否小于阈值的布尔值
        return distance(p1 - p2) < self.epsilon

    # 这两个方法与调用 adjustSize 方法一起用于滚动区域
    # 返回部件的推荐大小
    def sizeHint(self):
        return self.minimumSizeHint()

    # 返回部件的最小大小
    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    # 处理鼠标滚轮事件
    def wheelEvent(self, ev):
        # 根据事件对象是否有 delta 属性判断 Qt 版本
        qt_version = 4 if hasattr(ev, "delta") else 5
        if qt_version == 4:
            # 如果是 Qt4，根据滚轮方向设置垂直和水平滚动值
            if ev.orientation() == Qt.Vertical:
                v_delta = ev.delta()
                h_delta = 0
            else:
                h_delta = ev.delta()
                v_delta = 0
        else:
            # 如果是 Qt5，根据角度差设置垂直和水平滚动值
            delta = ev.angleDelta()
            h_delta = delta.x()
            v_delta = delta.y()

        # 获取事件的修饰键
        mods = ev.modifiers()
        # 如果按下了 Ctrl 键并且有垂直滚动值，则发出缩放请求信号
        if Qt.ControlModifier == int(mods) and v_delta:
            self.zoomRequest.emit(v_delta)
        else:
            # 如果有垂直滚动值，则发出垂直滚动请求信号
            v_delta and self.scrollRequest.emit(v_delta, Qt.Vertical)
            # 如果有水平滚动值，则发出水平滚动请求信号
            h_delta and self.scrollRequest.emit(h_delta, Qt.Horizontal)
        # 接受事件
        ev.accept()
    # 处理键盘按键事件的方法
    def keyPressEvent(self, ev):
        # 获取按下的键值
        key = ev.key()
        # 备份当前图形列表
        shapesBackup = copy.deepcopy(self.shapes)
        # 如果图形列表为空，则返回
        if len(shapesBackup) == 0:
            return
        # 弹出最后一个备份
        self.shapesBackups.pop()
        # 将备份添加到备份列表中
        self.shapesBackups.append(shapesBackup)
        # 处理不同按键的操作
        if key == Qt.Key_Escape and self.current:
            # 如果按下 ESC 键且当前有图形在绘制，则取消当前操作
            print('ESC press')
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key == Qt.Key_Return and self.canCloseShape():
            # 如果按下回车键且可以关闭当前图形，则完成图形绘制
            self.finalise()
        elif key == Qt.Key_Left and self.selectedShapes:
            # 如果按下左箭头键且有选中的图形，则向左移动一个像素
            self.moveOnePixel('Left')
        elif key == Qt.Key_Right and self.selectedShapes:
            # 如果按下右箭头键且有选中的图形，则向右移动一个像素
            self.moveOnePixel('Right')
        elif key == Qt.Key_Up and self.selectedShapes:
            # 如果按下上箭头键且有选中的图形，则向上移动一个像素
            self.moveOnePixel('Up')
        elif key == Qt.Key_Down and self.selectedShapes:
            # 如果按下下箭头键且有选中的图形，则向下移动一个像素
            self.moveOnePixel('Down')
        elif key == Qt.Key_X and self.selectedShapes:
            # 如果按下 X 键且有选中的图形，则旋转选中的图形
            for i in range(len(self.selectedShapes)):
                self.selectedShape = self.selectedShapes[i]
                # 如果旋转后超出边界，则跳过
                if self.rotateOutOfBound(0.01):
                    continue
                self.selectedShape.rotate(0.01)
            self.shapeMoved.emit()
            self.update()
        elif key == Qt.Key_C and self.selectedShapes:
            # 如果按下 C 键且有选中的图形，则逆时针旋转选中的图形
            for i in range(len(self.selectedShapes)):
                self.selectedShape = self.selectedShapes[i]
                # 如果旋转后超出边界，则跳过
                if self.rotateOutOfBound(-0.01):
                    continue
                self.selectedShape.rotate(-0.01)
            self.shapeMoved.emit()
            self.update()

    # 检查旋转后是否超出边界的方法
    def rotateOutOfBound(self, angle):
        # 遍历选中的图形
        for shape in range(len(self.selectedShapes)):
            self.selectedShape = self.selectedShapes[shape]
            # 遍历图形的每个点
            for i, p in enumerate(self.selectedShape.points):
                # 如果旋转后超出边界，则返回 True
                if self.outOfPixmap(self.selectedShape.rotatePoint(p, angle)):
                    return True
            return False
    # 根据步长移动选定形状的四个点，判断是否超出边界
    def moveOutOfBound(self, step):
        # 计算移动后的四个点的坐标
        points = [p1+p2 for p1, p2 in zip(self.selectedShape.points, [step]*4)]
        # 判断移动后的点是否超出边界
        return True in map(self.outOfPixmap, points)

    # 设置最后一个标签的文本、线条颜色、填充颜色和关键类别
    def setLastLabel(self, text, line_color=None, fill_color=None, key_cls=None):
        assert text
        # 设置最后一个形状的标签文本
        self.shapes[-1].label = text
        # 如果有线条颜色，则设置最后一个形状的线条颜色
        if line_color:
            self.shapes[-1].line_color = line_color
        # 如果有填充颜色，则设置最后一个形状的填充颜色
        if fill_color:
            self.shapes[-1].fill_color = fill_color
        # 如果有关键类别，则设置最后一个形状的关键类别
        if key_cls:
            self.shapes[-1].key_cls = key_cls
        # 存储形状信息
        self.storeShapes()
        # 返回最后一个形状
        return self.shapes[-1]

    # 撤销最后一条线段
    def undoLastLine(self):
        assert self.shapes
        # 弹出最后一个形状
        self.current = self.shapes.pop()
        # 设置最后一个形状为开放状态
        self.current.setOpen()
        # 设置线段的两个端点
        self.line.points = [self.current[-1], self.current[0]]
        # 发送绘制多边形信号
        self.drawingPolygon.emit(True)

    # 撤销最后一个点
    def undoLastPoint(self):
        # 如果当前没有形状或者当前形状已经闭合，则返回
        if not self.current or self.current.isClosed():
            return
        # 移除最后一个点
        self.current.popPoint()
        # 如果当前形状还有点，则更新线段的起点
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            # 发送绘制多边形信号
            self.drawingPolygon.emit(False)
        # 重绘
        self.repaint()

    # 重置所有线段
    def resetAllLines(self):
        assert self.shapes
        # 弹出最后一个形状
        self.current = self.shapes.pop()
        # 设置最后一个形状为开放状态
        self.current.setOpen()
        # 设置线段的两个端点
        self.line.points = [self.current[-1], self.current[0]]
        # 发送绘制多边形信号
        self.drawingPolygon.emit(True)
        self.current = None
        # 发送绘制多边形信号
        self.drawingPolygon.emit(False)
        self.update()

    # 加载像素图
    def loadPixmap(self, pixmap):
        self.pixmap = pixmap
        self.shapes = []
        # 重绘
        self.repaint()

    # 加载形状
    def loadShapes(self, shapes, replace=True):
        # 如果替换形状，则将形状列表替换为给定形状列表
        if replace:
            self.shapes = list(shapes)
        else:
            # 否则将给定形状列表添加到形状列表中
            self.shapes.extend(shapes)
        self.current = None
        self.hShape = None
        self.hVertex = None
        # self.hEdge = None
        # 存储形状信息
        self.storeShapes()
        # 更新形状索引
        self.updateShapeIndex()
        # 重绘
        self.repaint()
    # 设置指定形状的可见性
    def setShapeVisible(self, shape, value):
        # 更新形状的可见性
        self.visible[shape] = value
        # 重新绘制
        self.repaint()

    # 获取当前光标形状
    def currentCursor(self):
        # 获取当前覆盖的光标
        cursor = QApplication.overrideCursor()
        # 如果存在覆盖的光标，则获取其形状
        if cursor is not None:
            cursor = cursor.shape()
        return cursor

    # 覆盖光标形状
    def overrideCursor(self, cursor):
        # 设置私有属性 _cursor 为指定光标
        self._cursor = cursor
        # 如果当前光标为空，则设置覆盖光标
        if self.currentCursor() is None:
            QApplication.setOverrideCursor(cursor)
        else:
            QApplication.changeOverrideCursor(cursor)

    # 恢复光标
    def restoreCursor(self):
        QApplication.restoreOverrideCursor()

    # 重置状态
    def resetState(self):
        # 恢复光标
        self.restoreCursor()
        # 重置 pixmap 为 None
        self.pixmap = None
        # 更新界面
        self.update()
        # 清空形状备份列表
        self.shapesBackups = []

    # 设置绘制形状为正方形
    def setDrawingShapeToSquare(self, status):
        self.drawSquare = status

    # 恢复形状
    def restoreShape(self):
        # 如果无法恢复形状，则直接返回
        if not self.isShapeRestorable:
            return

        # 弹出最新的形状备份
        self.shapesBackups.pop()  # latest
        shapesBackup = self.shapesBackups.pop()
        # 恢复形状列表
        self.shapes = shapesBackup
        self.selectedShapes = []
        # 取消所有形状的选中状态
        for shape in self.shapes:
            shape.selected = False
        # 更新形状索引
        self.updateShapeIndex()
        # 重新绘制
        self.repaint()
    
    # 判断是否可以恢复形状
    @property
    def isShapeRestorable(self):
        # 如果形状备份列表长度小于2，则无法恢复
        if len(self.shapesBackups) < 2:
            return False
        return True

    # 更新形状索引
    def updateShapeIndex(self):
        # 遍历形状列表，更新索引值
        for i in range(len(self.shapes)):
            self.shapes[i].idx = i
        # 更新界面
        self.update()
```