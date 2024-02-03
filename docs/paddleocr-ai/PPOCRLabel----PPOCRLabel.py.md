# `.\PaddleOCR\PPOCRLabel\PPOCRLabel.py`

```
# 版权声明，允许在特定条件下使用和分发软件
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 导入所需的库和模块
import argparse
import ast
import codecs
import json
import os.path
import platform
import subprocess
import sys
import xlrd
from functools import partial

# 导入 PyQt5 中需要的类和函数
from PyQt5.QtCore import QSize, Qt, QPoint, QByteArray, QTimer, QFileInfo, QPointF, QProcess
from PyQt5.QtGui import QImage, QCursor, QPixmap, QImageReader
from PyQt5.QtWidgets import QMainWindow, QListWidget, QVBoxLayout, QToolButton, QHBoxLayout, QDockWidget, QWidget, \
    QSlider, QGraphicsOpacityEffect, QMessageBox, QListView, QScrollArea, QWidgetAction, QApplication, QLabel, QGridLayout, \
    QFileDialog, QListWidgetItem, QComboBox, QDialog, QAbstractItemView, QSizePolicy
# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))

# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上一级目录添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
# 将当前文件所在目录的上一级目录下的 PaddleOCR 目录添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(__dir__, '../PaddleOCR')))
# 将当前文件所在目录的上一级目录添加到系统路径中
sys.path.append("..")

# 导入 PaddleOCR 和 PPStructure 类
from paddleocr import PaddleOCR, PPStructure
# 导入常量
from libs.constants import *
# 导入工具函数
from libs.utils import *
# 导入标签颜色映射
from libs.labelColor import label_colormap
# 导入设置
from libs.settings import Settings
# 导入形状类
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR, DEFAULT_LOCK_COLOR
# 导入字符串捆绑类
from libs.stringBundle import StringBundle
# 导入画布类
from libs.canvas import Canvas
# 导入缩放小部件类
from libs.zoomWidget import ZoomWidget
# 导入自动对话框类
from libs.autoDialog import AutoDialog
# 导入标签对话框类
from libs.labelDialog import LabelDialog
# 导入颜色对话框类
from libs.colorDialog import ColorDialog
# 导入字符串转换函数
from libs.ustr import ustr
# 导入可哈希的 QListWidgetItem 类
from libs.hashableQListWidgetItem import HashableQListWidgetItem
# 导入列表编辑类
from libs.editinlist import EditInList
# 导入唯一标签 QListWidget 类
from libs.unique_label_qlist_widget import UniqueLabelQListWidget
# 导入键盘对话框类
from libs.keyDialog import KeyDialog

# 设置应用程序名称
__appname__ = 'PPOCRLabel'

# 标签颜色映射
LABEL_COLORMAP = label_colormap()

# 主窗口类
class MainWindow(QMainWindow):
    # 定义适应窗口、适应宽度、手动缩放三种模式
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    # 定义菜单函数
    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    # 键盘释放事件处理函数
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    # 键盘按下事件处理函数
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # 如果按下 Ctrl 键，则绘制矩形
            self.canvas.setDrawingShapeToSquare(True)

    # 没有形状的函数
    def noShapes(self):
        return not self.itemsToShapes
    # 填充模式操作菜单
    def populateModeActions(self):
        # 清空画布菜单
        self.canvas.menus[0].clear()
        # 向画布菜单添加动作
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        # 清空编辑菜单
        self.menus.edit.clear()
        # 根据用户级别选择不同的动作
        actions = (self.actions.create,)  # if self.beginner() else (self.actions.createMode, self.actions.editMode)
        # 向编辑菜单添加动作
        addActions(self.menus.edit, actions + self.actions.editMenu)

    # 设置为脏状态
    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    # 设置为干净状态
    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)
        self.actions.createpoly.setEnabled(True)

    # 切换动作的启用/禁用状态
    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        # 设置缩放动作的启用/禁用状态
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        # 设置加载时激活的动作的启用/禁用状态
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    # 在事件队列中添加函数
    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    # 显示状态消息
    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    # 重置状态
    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.itemsToShapesbox.clear()  # ADD
        self.shapesToItemsbox.clear()
        self.labelList.clear()
        self.BoxList.clear()
        self.indexList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()
        # self.comboBox.cb.clear()
        self.result_dic = []

    # 获取当前选中的项目
    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    # 获取当前选中的框
    def currentBox(self):
        items = self.BoxList.selectedItems()
        if items:
            return items[0]
        return None
    # 将文件路径添加到最近文件列表中
    def addRecentFile(self, filePath):
        # 如果文件路径已经在最近文件列表中，则移除
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        # 如果最近文件列表已经达到最大数量，则移除最旧的文件路径
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        # 将文件路径插入到最近文件列表的最前面
        self.recentFiles.insert(0, filePath)

    # 返回是否为初学者模式
    def beginner(self):
        return self._beginner

    # 返回是否为高级模式
    def advanced(self):
        return not self.beginner()

    # 获取可用的屏幕录像查看器
    def getAvailableScreencastViewer(self):
        osName = platform.system()

        # 根据操作系统返回不同的屏幕录像查看器路径
        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## 回调函数 ##
    # 显示教程对话框
    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    # 显示信息对话框
    def showInfoDialog(self):
        from libs.__init__ import __version__
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    # 显示步骤对话框
    def showStepsDialog(self):
        msg = stepsInfo(self.lang)
        QMessageBox.information(self, u'Information', msg)

    # 显示按键对话框
    def showKeysDialog(self):
        msg = keysInfo(self.lang)
        QMessageBox.information(self, u'Information', msg)

    # 创建形状
    def createShape(self):
        # 确保当前为初学者模式
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)
        self.actions.createpoly.setEnabled(False)
        self.canvas.fourpoint = False

    # 创建多边形
    def createPolygon(self):
        # 确保当前为初学者模式
        assert self.beginner()
        self.canvas.setEditing(False)
        self.canvas.fourpoint = True
        self.actions.create.setEnabled(False)
        self.actions.createpoly.setEnabled(False)
        self.actions.undoLastPoint.setEnabled(True)
    # 旋转图片并保存，根据传入的旋转次数和值来确定旋转方向和角度
    def rotateImg(self, filename, k, _value):
        # 设置旋转右按钮是否可用
        self.actions.rotateRight.setEnabled(_value)
        # 读取图片文件
        pix = cv2.imread(filename)
        # 对图片进行旋转操作
        pix = np.rot90(pix, k)
        # 保存旋转后的图片
        cv2.imwrite(filename, pix)
        # 更新画布显示
        self.canvas.update()
        # 加载旋转后的图片
        self.loadFile(filename)

    # 提示用户旋转图片可能会打乱标注框
    def rotateImgWarn(self):
        # 根据语言设置不同的提示信息
        if self.lang == 'ch':
            self.msgBox.warning(self, "提示", "\n 该图片已经有标注框,旋转操作会打乱标注,建议清除标注框后旋转。")
        else:
            self.msgBox.warning(self, "Warn", "\n The picture already has a label box, "
                                              "and rotation will disrupt the label. "
                                              "It is recommended to clear the label box and rotate it.")

    # 执行旋转图片操作，根据传入的参数决定是否进行旋转
    def rotateImgAction(self, k=1, _value=False):
        # 获取当前图片文件名
        filename = self.mImgList[self.currIndex]
        # 判断文件是否存在
        if os.path.exists(filename):
            # 如果存在标注框，则提示用户
            if self.itemsToShapesbox:
                self.rotateImgWarn()
            else:
                # 保存当前文件，重置dirty标志，执行图片旋转操作
                self.saveFile()
                self.dirty = False
                self.rotateImg(filename=filename, k=k, _value=True)
        else:
            # 如果文件不存在，则提示用户
            self.rotateImgWarn()
            # 禁用旋转按钮
            self.actions.rotateRight.setEnabled(False)
            self.actions.rotateLeft.setEnabled(False)

    # 切换绘图敏感模式，根据传入的参数决定是否允许绘图
    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        # 设置编辑模式按钮是否可用
        self.actions.editMode.setEnabled(not drawing)
        # 如果不在绘图状态且为初学者，则取消创建
        if not drawing and self.beginner():
            print('Cancel creation.')
            # 设置画布为编辑模式
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)
            self.actions.createpoly.setEnabled(True)

    # 切换绘图模式，根据传入的参数决定是否进入编辑模式
    def toggleDrawMode(self, edit=True):
        # 设置画布的编辑模式
        self.canvas.setEditing(edit)
        # 设置创建模式按钮是否可用
        self.actions.createMode.setEnabled(edit)
        # 设置编辑模式按钮是否可用
        self.actions.editMode.setEnabled(not edit)

    # 设置创建模式，确保为高级用户
    def setCreateMode(self):
        assert self.advanced()
        # 切换为非绘图模式
        self.toggleDrawMode(False)
    # 设置编辑模式，确保当前为高级模式
    def setEditMode(self):
        assert self.advanced()
        # 切换至绘制模式
        self.toggleDrawMode(True)
        # 标签选择改变
        self.labelSelectionChanged()

    # 更新文件菜单
    def updateFileMenu(self):
        # 获取当前文件路径
        currFilePath = self.filePath

        # 检查文件是否存在
        def exists(filename):
            return os.path.exists(filename)

        # 清空最近文件菜单
        menu = self.menus.recentFiles
        menu.clear()
        # 获取最近文件列表，排除当前文件和不存在的文件
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            # 创建动作
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    # 弹出标签列表菜单
    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    # 编辑标签
    def editLabel(self):
        # 如果不在编辑模式，则返回
        if not self.canvas.editing():
            return
        # 获取当前项
        item = self.currentItem()
        if not item:
            return
        # 弹出标签对话框
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            # item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    # 检测框相关函数
    def boxItemChanged(self, item):
        # 获取形状对象
        shape = self.itemsToShapesbox[item]

        # 将文本转换为列表
        box = ast.literal_eval(item.text())
        # 如果框不同于形状的点列表，则更新形状的点列表
        if box != [(int(p.x()), int(p.y())) for p in shape.points]:
            shape.points = [QPointF(p[0], p[1]) for p in box]

            # QPointF(x,y)
            # shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # 用户可能改变了项目的可见性
            self.canvas.setShapeVisible(shape, True)  # item.checkState() == Qt.Checked
    def editBox(self):  # 定义一个编辑框方法
        # 如果画布不在编辑状态，则返回
        if not self.canvas.editing():
            return
        # 获取当前选中的框
        item = self.currentBox()
        # 如果没有选中框，则返回
        if not item:
            return
        # 弹出标签对话框，获取用户输入的文本
        text = self.labelDialog.popUp(item.text())

        # 获取图像尺寸
        imageSize = str(self.image.size())
        width, height = self.image.width(), self.image.height()
        # 如果用户输入了文本
        if text:
            try:
                # 尝试将文本转换为列表
                text_list = eval(text)
            except:
                # 如果转换失败，弹出警告框
                msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Please enter the correct format')
                msg_box.exec_()
                return
            # 如果列表长度小于4，弹出警告框
            if len(text_list) < 4:
                msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Please enter the coordinates of 4 points')
                msg_box.exec_()
                return
            # 遍历列表中的坐标点，检查是否超出图像范围
            for box in text_list:
                if box[0] > width or box[0] < 0 or box[1] > height or box[1] < 0:
                    msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Out of picture size')
                    msg_box.exec_()
                    return

            # 设置框的文本为用户输入的文本
            item.setText(text)
            # 设置框的背景颜色
            # item.setBackground(generateColorByText(text))
            # 标记为脏数据
            self.setDirty()
            # 更新下拉框
            self.updateComboBox()

    # 更新框列表
    def updateBoxlist(self):
        # 清空选中的形状
        self.canvas.selectedShapes_hShape = []
        # 如果有悬停形状
        if self.canvas.hShape != None:
            # 将选中的形状和悬停形状添加到选中的形状列表中
            self.canvas.selectedShapes_hShape = self.canvas.selectedShapes + [self.canvas.hShape]
        else:
            # 否则，只添加选中的形状
            self.canvas.selectedShapes_hShape = self.canvas.selectedShapes
        # 遍历选中的形状列表
        for shape in self.canvas.selectedShapes_hShape:
            # 如果形状在shapesToItemsbox字典中
            if shape in self.shapesToItemsbox.keys():
                # 获取对应的列表项
                item = self.shapesToItemsbox[shape]  # listitem
                # 将形状的坐标点转换为文本，并设置为列表项的文本
                text = [(int(p.x()), int(p.y())) for p in shape.points]
                item.setText(str(text))
        # 启用撤销操作
        self.actions.undo.setEnabled(True)
        # 标记为脏数据
        self.setDirty()
    # 将当前索引转换为包含5个文件的列表
    def indexTo5Files(self, currIndex):
        # 如果当前索引小于2，返回前5个文件
        if currIndex < 2:
            return self.mImgList[:5]
        # 如果当前索引大于总文件数减3，返回最后5个文件
        elif currIndex > len(self.mImgList) - 3:
            return self.mImgList[-5:]
        # 否则返回当前索引前后各2个文件
        else:
            return self.mImgList[currIndex - 2: currIndex + 3]

    # Tzutalin 20160906 : 添加文件列表和 dock 以便更快移动
    def fileitemDoubleClicked(self, item=None):
        # 获取当前文件索引
        self.currIndex = self.mImgList.index(ustr(os.path.join(os.path.abspath(self.dirname), item.text())))
        filename = self.mImgList[self.currIndex]
        if filename:
            # 获取当前文件索引前后各2个文件
            self.mImgList5 = self.indexTo5Files(self.currIndex)
            # 加载当前文件
            self.loadFile(filename)

    def iconitemDoubleClicked(self, item=None):
        # 获取当前文件索引
        self.currIndex = self.mImgList.index(ustr(os.path.join(item.toolTip())))
        filename = self.mImgList[self.currIndex]
        if filename:
            # 获取当前文件索引前后各2个文件
            self.mImgList5 = self.indexTo5Files(self.currIndex)
            # 加载当前文件
            self.loadFile(filename)

    def CanvasSizeChange(self):
        # 如果文件列表不为空且图像滑块有焦点
        if len(self.mImgList) > 0 and self.imageSlider.hasFocus():
            # 设置缩放小部件的值为图像滑块的值
            self.zoomWidget.setValue(self.imageSlider.value())
    # 为给定的形状添加标签
    def addLabel(self, shape):
        # 设置形状的标签绘制选项
        shape.paintLabel = self.displayLabelOption.isChecked()
        # 设置形状的索引绘制选项
        shape.paintIdx = self.displayIndexOption.isChecked()

        # 创建可哈希的 QListWidgetItem 对象
        item = HashableQListWidgetItem(shape.label)
        
        # 当前困难复选框被禁用
        # item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        # 如果形状困难，则设置复选框为未选中，否则设置为选中
        # item.setCheckState(Qt.Unchecked) if shape.difficult else item.setCheckState(Qt.Checked)

        # 选中表示困难为 False
        # 根据标签文本生成背景颜色
        # item.setBackground(generateColorByText(shape.label))
        
        # 将 QListWidgetItem 对象与形状关联存储
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        
        # 在标签字符串之前添加当前标签项索引
        current_index = QListWidgetItem(str(self.labelList.count()))
        current_index.setTextAlignment(Qt.AlignHCenter)
        self.indexList.addItem(current_index)
        self.labelList.addItem(item)
        
        # 添加用于框的信息
        item = HashableQListWidgetItem(str([(int(p.x()), int(p.y())) for p in shape.points]))
        self.itemsToShapesbox[item] = shape
        self.shapesToItemsbox[shape] = item
        self.BoxList.addItem(item)
        
        # 启用所有形状存在时的操作
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        
        # 更新组合框
        self.updateComboBox()

        # 更新显示计数
        self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
        self.labelListDock.setWindowTitle(self.labelListDockName + f" ({self.labelList.count()})")
    # 从标签列表中移除指定的形状
    def remLabels(self, shapes):
        # 如果形状列表为空，则直接返回
        if shapes is None:
            # print('rm empty label')
            return
        # 遍历每个形状
        for shape in shapes:
            # 获取形状对应的列表项
            item = self.shapesToItems[shape]
            # 从标签列表中移除该列表项
            self.labelList.takeItem(self.labelList.row(item))
            # 删除形状到列表项的映射
            del self.shapesToItems[shape]
            # 删除列表项到形状的映射
            del self.itemsToShapes[item]
            # 更新下拉框
            self.updateComboBox()

            # ADD:
            # 获取形状对应的列表项
            item = self.shapesToItemsbox[shape]
            # 从框列表中移除该列表项
            self.BoxList.takeItem(self.BoxList.row(item))
            # 删除形状到列表项的映射
            del self.shapesToItemsbox[shape]
            # 删除列表项到形状的映射
            del self.itemsToShapesbox[item]
            # 更新下拉框
            self.updateComboBox()
        # 更新索引列表
        self.updateIndexList()

    # 加载标签
    def loadLabels(self, shapes):
        # 创建一个空列表用于存储形状
        s = []
        shape_index = 0
        # 遍历每个标签、点、线颜色、关键类别和难度
        for label, points, line_color, key_cls, difficult in shapes:
            # 创建一个形状对象
            shape = Shape(label=label, line_color=line_color, key_cls=key_cls)
            # 遍历每个点
            for x, y in points:

                # 确保标签在图像边界内，如果不在，则修正
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                # 将点添加到形状中
                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.idx = shape_index
            shape_index += 1
            # shape.locked = False
            shape.close()
            s.append(shape)

            # 更新形状的颜色
            self._update_shape_color(shape)
            # 添加标签
            self.addLabel(shape)

        # 更新下拉框
        self.updateComboBox()
        # 加载形状到画布中
        self.canvas.loadShapes(s)

    # 单个标签
    def singleLabel(self, shape):
        # 如果形状为空，则直接返回
        if shape is None:
            # print('rm empty label')
            return
        # 获取形状对应的列表项
        item = self.shapesToItems[shape]
        # 设置列表项的文本为形状的标签
        item.setText(shape.label)
        # 更新下拉框

        # ADD:
        # 获取形状对应的列表项
        item = self.shapesToItemsbox[shape]
        # 设置列表项的文本为形状的点坐标
        item.setText(str([(int(p.x()), int(p.y())) for p in shape.points]))
        # 更新下拉框
        self.updateComboBox()
    # 更新下拉框中的选项，获取唯一标签并添加到下拉框中
    def updateComboBox(self):
        # 获取标签列表中每个项的文本内容并转换为字符串列表
        itemsTextList = [str(self.labelList.item(i).text()) for i in range(self.labelList.count())]

        # 获取唯一的文本内容列表
        uniqueTextList = list(set(itemsTextList))
        # 添加一个空行用于显示所有标签
        uniqueTextList.append("")
        uniqueTextList.sort()

        # 更新下拉框中的选项
        # self.comboBox.update_items(uniqueTextList)

    # 更新索引列表
    def updateIndexList(self):
        # 清空索引列表
        self.indexList.clear()
        # 遍历标签列表的数量，为每个索引创建一个列表项
        for i in range(self.labelList.count()):
            string = QListWidgetItem(str(i))
            string.setTextAlignment(Qt.AlignHCenter)
            self.indexList.addItem(string)

    # 复制选定的形状
    def copySelectedShape(self):
        # 复制选定的形状并添加到标签列表中
        for shape in self.canvas.copySelectedShape():
            self.addLabel(shape)
        # 修复复制和删除
        # self.shapeSelectionChanged(True)

    # 移动滚动条
    def move_scrollbar(self, value):
        # 设置标签列表滚动条的值
        self.labelListBar.setValue(value)
        # 设置索引列表滚动条的值
        self.indexListBar.setValue(value)

    # 标签选择改变时的操作
    def labelSelectionChanged(self):
        # 如果没有选择槽，则返回
        if self._noSelectionSlot:
            return
        # 如果画布正在编辑
        if self.canvas.editing():
            selected_shapes = []
            # 遍历标签列表中选定的项，将其对应的形状添加到选定形状列表中
            for item in self.labelList.selectedItems():
                selected_shapes.append(self.itemsToShapes[item])
            # 如果有选定的形状，则在画布中选中这些形状
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                # 否则，取消选择形状
                self.canvas.deSelectShape()
    # 当索引列表的选择发生变化时触发的槽函数
    def indexSelectionChanged(self):
        # 如果没有选择槽函数，则直接返回
        if self._noSelectionSlot:
            return
        # 如果画布正在编辑状态
        if self.canvas.editing():
            # 初始化一个空列表用于存储选中的形状
            selected_shapes = []
            # 遍历索引列表中选中的项
            for item in self.indexList.selectedItems():
                # 将索引项映射到标签项
                index = self.indexList.indexFromItem(item).row()
                item = self.labelList.item(index)
                selected_shapes.append(self.itemsToShapes[item])
            # 如果有选中的形状，则调用画布的选择形状方法
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    # 当框选列表的选择发生变化时触发的槽函数
    def boxSelectionChanged(self):
        # 如果没有选择槽函数，则直接返回
        if self._noSelectionSlot:
            # 滚动到当前框选项
            # self.BoxList.scrollToItem(self.currentBox(), QAbstractItemView.PositionAtCenter)
            return
        # 如果画布正在编辑状态
        if self.canvas.editing():
            # 初始化一个空列表用于存储选中的形状
            selected_shapes = []
            # 遍历框选列表中选中的项
            for item in self.BoxList.selectedItems():
                selected_shapes.append(self.itemsToShapesbox[item])
            # 如果有选中的形状，则调用画布的选择形状方法
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()
    # 当标签项发生改变时触发的方法
    def labelItemChanged(self, item):
        # 避免使用不可哈希的项意外触发itemChanged信号
        # 未知的触发条件
        if type(item) == HashableQListWidgetItem:
            # 获取与该项对应的形状对象
            shape = self.itemsToShapes[item]
            # 获取标签项的文本内容
            label = item.text()
            # 如果标签内容与形状对象的标签不同，则更新形状对象的标签
            if label != shape.label:
                shape.label = item.text()
                # shape.line_color = generateColorByText(shape.label)
                # 标记数据已更改
                self.setDirty()
            # 如果标签项的选中状态与形状对象的difficult属性不同，则更新difficult属性
            elif not ((item.checkState() == Qt.Unchecked) ^ (not shape.difficult)):
                shape.difficult = True if item.checkState() == Qt.Unchecked else False
                # 标记数据已更改
                self.setDirty()
            else:  # 用户可能改变了项的可见性
                # 设置形状对象为可见状态
                self.canvas.setShapeVisible(shape, True)  # item.checkState() == Qt.Checked
                # self.actions.save.setEnabled(True)
        else:
            # 打印出使用不可哈希的项进入labelItemChanged槽的信息
            print('enter labelItemChanged slot with unhashable item: ', item, item.text())
    # 当拖放事件发生时调用的方法
    def drag_drop_happened(self):
        '''
        label list drag drop signal slot
        '''
        # 仅选择单个项目
        for item in self.labelList.selectedItems():
            # 获取选中项目的索引
            newIndex = self.labelList.indexFromItem(item).row()

        # 仅支持拖放一个项目
        assert len(self.canvas.selectedShapes) > 0
        for shape in self.canvas.selectedShapes:
            # 获取选中形状的索引
            selectedShapeIndex = shape.idx
        
        # 如果新索引和选中形状的索引相同，则直接返回
        if newIndex == selectedShapeIndex:
            return

        # 移动形状列表中对应的项目
        shape = self.canvas.shapes.pop(selectedShapeIndex)
        self.canvas.shapes.insert(newIndex, shape)
            
        # 更新边界框索引
        self.canvas.updateShapeIndex()

        # 同步更新边界框列表
        item = self.BoxList.takeItem(selectedShapeIndex)
        self.BoxList.insertItem(newIndex, item)

        # 标记发生了变化
        self.setDirty()

    # 回调函数:
    # 定义一个方法用于创建新的形状，弹出并将焦点放在标签编辑器上
    def newShape(self, value=True):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        # 如果标签历史记录不为空，则创建一个标签对话框
        if len(self.labelHist) > 0:
            self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        # 如果 value 为 True，则弹出标签对话框并获取文本
        if value:
            text = self.labelDialog.popUp(text=self.prevLabelText)
            self.lastLabel = text
        else:
            text = self.prevLabelText

        # 如果文本不为空
        if text is not None:
            # 设置上一个标签文本为临时标签
            self.prevLabelText = self.stringBundle.getString('tempLabel')

            # 在画布上设置最后一个标签的文本，并返回形状
            shape = self.canvas.setLastLabel(text, None, None, None)  # generate_color, generate_color
            # 如果处于 kie_mode 模式
            if self.kie_mode:
                # 弹出键盘对话框并获取键盘文本
                key_text, _ = self.keyDialog.popUp(self.key_previous_text)
                if key_text is not None:
                    # 在画布上设置最后一个标签的文本和键盘文本，并返回形状
                    shape = self.canvas.setLastLabel(text, None, None, key_text)  # generate_color, generate_color
                    self.key_previous_text = key_text
                    # 如果键盘列表中不存在该键盘文本，则创建新的键盘项
                    if not self.keyList.findItemsByLabel(key_text):
                        item = self.keyList.createItemFromLabel(key_text)
                        self.keyList.addItem(item)
                        rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                        self.keyList.setItemLabel(item, key_text, rgb)

                    # 更新形状颜色
                    self._update_shape_color(shape)
                    # 将键盘文本添加到历史记录中
                    self.keyDialog.addLabelHistory(key_text)

            # 添加标签形状
            self.addLabel(shape)
            # 如果是初学者，则切换到编辑模式
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
                self.actions.createpoly.setEnabled(True)
                self.actions.undoLastPoint.setEnabled(False)
                self.actions.undo.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            # 设置为脏状态
            self.setDirty()

        else:
            # 撤销最后一条线
            # self.canvas.undoLastLine()
            # 重置所有线
            self.canvas.resetAllLines()
    # 更新形状的颜色属性
    def _update_shape_color(self, shape):
        # 根据标签和模式获取 RGB 颜色值
        r, g, b = self._get_rgb_by_label(shape.key_cls, self.kie_mode)
        # 设置形状的线条颜色
        shape.line_color = QColor(r, g, b)
        # 设置形状的顶点填充颜色
        shape.vertex_fill_color = QColor(r, g, b)
        # 设置形状的高亮顶点填充颜色
        shape.hvertex_fill_color = QColor(255, 255, 255)
        # 设置形状的填充颜色
        shape.fill_color = QColor(r, g, b, 128)
        # 设置形状被选中时的线条颜色
        shape.select_line_color = QColor(255, 255, 255)
        # 设置形状被选中时的填充颜色
        shape.select_fill_color = QColor(r, g, b, 155)

    # 根据标签和模式获取 RGB 颜色值
    def _get_rgb_by_label(self, label, kie_mode):
        # 用于随机颜色的偏移量
        shift_auto_shape_color = 2
        # 如果是 KIE 模式且标签不为 "None"
        if kie_mode and label != "None":
            # 获取标签对应的列表项
            item = self.keyList.findItemsByLabel(label)[0]
            # 获取标签的索引
            label_id = self.keyList.indexFromItem(item).row() + 1
            label_id += shift_auto_shape_color
            # 返回标签对应的颜色值
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        else:
            # 默认返回绿色
            return (0, 255, 0)

    # 滚动请求处理
    def scrollRequest(self, delta, orientation):
        # 计算滚动单位
        units = - delta / (8 * 15)
        # 获取指定方向的滚动条
        bar = self.scrollBars[orientation]
        # 设置滚动条的值
        bar.setValue(bar.value() + bar.singleStep() * units)

    # 设置缩放值
    def setZoom(self, value):
        # 取消适应宽度和适应窗口的选中状态
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        # 设置缩放模式为手动缩放
        self.zoomMode = self.MANUAL_ZOOM
        # 设置缩放部件的值
        self.zoomWidget.setValue(value)

    # 增加缩放值
    def addZoom(self, increment=10):
        # 设置缩放值为当前值加上增量
        self.setZoom(self.zoomWidget.value() + increment)
        # 设置图像滑块的值为当前值加上增量
        self.imageSlider.setValue(self.zoomWidget.value() + increment)
    # 定义一个方法用于处理缩放请求，接收缩放增量参数
    def zoomRequest(self, delta):
        # 获取当前滚动条位置
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # 获取当前最大值，用于计算缩放后的差值
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # 获取光标位置和画布大小
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # 缩放从0到1之间有一些填充
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # 将值限制在0到1之间
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # 缩放
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # 获取滚动条值的差值
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # 获取新的滚动条值
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    # 定义一个方法用于设置适应窗口大小
    def setFitWindow(self, value=True):
        # 如果值为真，则取消适应宽度选项
        if value:
            self.actions.fitWidth.setChecked(False)
        # 根据值设置缩放模式
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        # 调整缩放比例
        self.adjustScale()
    # 设置是否适应宽度
    def setFitWidth(self, value=True):
        # 如果值为真，则取消适应窗口选项
        if value:
            self.actions.fitWindow.setChecked(False)
        # 根据值设置缩放模式
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        # 调整比例
        self.adjustScale()

    # 切换多边形的可见性
    def togglePolygons(self, value):
        # 遍历项目和形状的映射，设置形状的可见性
        for item, shape in self.itemsToShapes.items():
            self.canvas.setShapeVisible(shape, value)

    # 从 PP 标签中显示边界框
    def showBoundingBoxFromPPlabel(self, filePath):
        # 获取图像的宽度和高度
        width, height = self.image.width(), self.image.height()
        # 获取图像标签的索引
        imgidx = self.getImglabelidx(filePath)
        shapes = []
        # 遍历锁定的形状，将形状的四个角坐标与图像的高度和宽度的比例添加到 shapes 中
        for box in self.canvas.lockedShapes:
            key_cls = 'None' if not self.kie_mode else box['key_cls']
            if self.canvas.isInTheSameImage:
                shapes.append((box['transcription'], [[s[0] * width, s[1] * height] for s in box['ratio']],
                               DEFAULT_LOCK_COLOR, key_cls, box['difficult']))
            else:
                shapes.append(('锁定框：待检测', [[s[0] * width, s[1] * height] for s in box['ratio']],
                               DEFAULT_LOCK_COLOR, key_cls, box['difficult']))
        # 如果图像索引在 PP 标签中存在，则将其边界框添加到 shapes 中
        if imgidx in self.PPlabel.keys():
            for box in self.PPlabel[imgidx]:
                key_cls = 'None' if not self.kie_mode else box.get('key_cls', 'None')
                shapes.append((box['transcription'], box['points'], None, key_cls, box.get('difficult', False)))

        # 如果 shapes 不为空，则加载标签并将 canvas 的 verified 属性设置为 False
        if shapes != []:
            self.loadLabels(shapes)
            self.canvas.verified = False

    # 验证文件状态
    def validFilestate(self, filePath):
        # 如果文件路径不在文件状态字典中，则返回 None
        if filePath not in self.fileStatedict.keys():
            return None
        # 如果文件状态为 1，则返回 True，否则返回 False
        elif self.fileStatedict[filePath] == 1:
            return True
        else:
            return False
    # 当窗口大小改变时触发的事件处理函数
    def resizeEvent(self, event):
        # 如果画布存在且图像不为空且缩放模式不是手动缩放
        if self.canvas and not self.image.isNull() \
                and self.zoomMode != self.MANUAL_ZOOM:
            # 调整缩放比例
            self.adjustScale()
        # 调用父类的resizeEvent方法
        super(MainWindow, self).resizeEvent(event)

    # 绘制画布的函数
    def paintCanvas(self):
        # 断言图像不为空，否则抛出异常
        assert not self.image.isNull(), "cannot paint null image"
        # 根据缩放部件的值设置画布的缩放比例
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        # 调整画布大小
        self.canvas.adjustSize()
        # 更新画布
        self.canvas.update()

    # 调整缩放比例的函数
    def adjustScale(self, initial=False):
        # 根据缩放模式选择缩放值
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        # 设置缩放部件的值
        self.zoomWidget.setValue(int(100 * value))
        # 设置图像滑块的值
        self.imageSlider.setValue(self.zoomWidget.value())  # set zoom slider value

    # 适应窗口大小的缩放函数
    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        # 设置一个小的偏移量，以避免生成滚动条
        e = 2.0
        # 计算主窗口的宽度和高度
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e - 110
        a1 = w1 / h1
        # 根据图像的宽高比计算新的缩放值
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    # 适应宽度的缩放函数
    def scaleFitWidth(self):
        # 计算主窗口的宽度
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()
    # 当窗口关闭时触发的事件处理函数
    def closeEvent(self, event):
        # 如果不允许继续关闭窗口，则忽略关闭事件
        if not self.mayContinue():
            event.ignore()
        else:
            # 获取窗口的设置信息
            settings = self.settings
            # 如果加载图片的目录为空，则将文件路径设置为设置中的文件路径，否则设置为空字符串
            if self.dirname is None:
                settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
            else:
                settings[SETTING_FILENAME] = ''

            # 保存窗口的大小、位置、状态等信息到设置中
            settings[SETTING_WIN_SIZE] = self.size()
            settings[SETTING_WIN_POSE] = self.pos()
            settings[SETTING_WIN_STATE] = self.saveState()
            settings[SETTING_LINE_COLOR] = self.lineColor
            settings[SETTING_FILL_COLOR] = self.fillColor
            settings[SETTING_RECENT_FILES] = self.recentFiles
            settings[SETTING_ADVANCE_MODE] = not self._beginner
            # 如果默认保存目录存在，则设置保存目录为默认保存目录，否则设置为空字符串
            if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
                settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
            else:
                settings[SETTING_SAVE_DIR] = ''

            # 如果最近打开目录存在，则设置最近打开目录为最近打开目录，否则设置为空字符串
            if self.lastOpenDir and os.path.exists(self.lastOpenDir):
                settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
            else:
                settings[SETTING_LAST_OPEN_DIR] = ''

            # 设置是否显示标签、索引、方框等信息到设置中
            settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
            settings[SETTING_PAINT_INDEX] = self.displayIndexOption.isChecked()
            settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
            # 保存设置信息
            settings.save()
            # 尝试保存标签文件，如果出错则忽略
            try:
                self.saveLabelFile()
            except:
                pass

    # 加载最近打开的文件
    def loadRecent(self, filename):
        # 如果可以继续加载文件
        if self.mayContinue():
            # 打印文件名
            print(filename, "======")
            # 加载文件
            self.loadFile(filename)
    # 扫描指定文件夹中的所有图片文件，并返回文件路径列表
    def scanAllImages(self, folderPath):
        # 获取支持的图片格式列表
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        # 初始化图片路径列表
        images = []

        # 遍历文件夹中的所有文件
        for file in os.listdir(folderPath):
            # 判断文件是否为图片文件
            if file.lower().endswith(tuple(extensions)):
                # 构建文件的相对路径
                relativePath = os.path.join(folderPath, file)
                # 获取文件的绝对路径
                path = ustr(os.path.abspath(relativePath))
                # 将文件路径添加到图片路径列表中
                images.append(path)
        # 对图片路径列表进行自然排序
        natural_sort(images, key=lambda x: x.lower())
        # 返回图片路径列表
        return images

    # 打开文件夹选择对话框，并导入文件夹中的图片
    def openDirDialog(self, _value=False, dirpath=None, silent=False):
        # 如果不允许继续操作，则直接返回
        if not self.mayContinue():
            return

        # 设置默认打开文件夹路径
        defaultOpenDirPath = dirpath if dirpath else '.'
        # 如果上次打开的文件夹路径存在，则使用该路径作为默认打开路径
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            # 否则使用当前文件路径的目录作为默认打开路径
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        
        # 如果不是静默模式，则弹出文件夹选择对话框
        if silent != True:
            targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                                  '%s - Open Directory' % __appname__,
                                                                  defaultOpenDirPath,
                                                                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            # 否则使用默认打开路径
            targetDirPath = ustr(defaultOpenDirPath)
        
        # 更新最后打开的文件夹路径
        self.lastOpenDir = targetDirPath
        # 导入文件夹中的图片
        self.importDirImages(targetDirPath)
    # 打开数据集目录对话框的方法
    def openDatasetDirDialog(self):
        # 如果上次打开的目录存在
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            # 如果操作系统是 Windows
            if platform.system() == 'Windows':
                # 使用系统默认程序打开目录
                os.startfile(self.lastOpenDir)
            else:
                # 在其他操作系统上使用系统命令打开目录
                os.system('open ' + os.path.normpath(self.lastOpenDir))
            # 设置默认打开目录为上次打开的目录
            defaultOpenDirPath = self.lastOpenDir

        else:
            # 如果上次打开的目录不存在
            # 根据语言设置不同的提示信息
            if self.lang == 'ch':
                self.msgBox.warning(self, "提示", "\n 原文件夹已不存在,请从新选择数据集路径!")
            else:
                self.msgBox.warning(self, "Warn",
                                    "\n The original folder no longer exists, please choose the data set path again!")
            # 禁用打开数据集目录的操作
            self.actions.open_dataset_dir.setEnabled(False)
            # 设置默认打开目录为当前文件路径的父目录，如果文件路径不存在则为当前目录
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
    # 初始化关键字列表，根据标签字典更新关键字列表
    def init_key_list(self, label_dict):
        # 如果不是关键字提取模式，则直接返回
        if not self.kie_mode:
            return
        # 遍历标签字典，加载关键字类别
        for image, info in label_dict.items():
            for box in info:
                # 如果盒子中没有关键字类别，则添加一个默认值
                if "key_cls" not in box:
                    box.update({"key_cls": "None"})
                # 将已存在的关键字类别添加到集合中
                self.existed_key_cls_set.add(box["key_cls"])
        # 如果存在关键字类别，则更新关键字列表
        if len(self.existed_key_cls_set) > 0:
            for key_text in self.existed_key_cls_set:
                # 如果关键字列表中没有该关键字类别，则创建新的项并添加到列表中
                if not self.keyList.findItemsByLabel(key_text):
                    item = self.keyList.createItemFromLabel(key_text)
                    self.keyList.addItem(item)
                    # 根据关键字类别获取对应的颜色，并更新关键字列表项的标签和颜色
                    rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                    self.keyList.setItemLabel(item, key_text, rgb)

        # 如果关键字对话框不存在，则创建关键字对话框
        if self.keyDialog is None:
            # 关键字列表对话框
            self.keyDialog = KeyDialog(
                text=self.key_dialog_tip,
                parent=self,
                labels=self.existed_key_cls_set,
                sort_labels=True,
                show_text_field=True,
                completion="startswith",
                fit_to_content={'column': True, 'row': False},
                flags=None
            )

    # 打开上一张图片
    def openPrevImg(self, _value=False):
        # 如果图片列表为空，则直接返回
        if len(self.mImgList) <= 0:
            return

        # 如果文件路径为空，则直接返回
        if self.filePath is None:
            return

        # 获取当前图片在图片列表中的索引
        currIndex = self.mImgList.index(self.filePath)
        # 获取当前图片的前5张图片
        self.mImgList5 = self.mImgList[:5]
        # 如果当前索引大于0，则加载前一张图片
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            self.mImgList5 = self.indexTo5Files(currIndex - 1)
            if filename:
                self.loadFile(filename)
    # 打开下一张图片，如果不允许继续则返回
    def openNextImg(self, _value=False):
        if not self.mayContinue():
            return

        # 如果图片列表为空则返回
        if len(self.mImgList) <= 0:
            return

        filename = None
        # 如果文件路径为空，则选择第一张图片
        if self.filePath is None:
            filename = self.mImgList[0]
            self.mImgList5 = self.mImgList[:5]
        else:
            # 获取当前文件路径在图片列表中的索引
            currIndex = self.mImgList.index(self.filePath)
            # 如果当前索引加一小于图片列表长度，则选择下一张图片
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]
                self.mImgList5 = self.indexTo5Files(currIndex + 1)
            else:
                self.mImgList5 = self.indexTo5Files(currIndex)
        # 如果有文件名，则打印文件名并加载文件
        if filename:
            print('file name in openNext is ', filename)
            self.loadFile(filename)

    # 更新文件列表图标
    def updateFileListIcon(self, filename):
        pass

    # 保存文件，手动模式用于用户手动点击“保存”，将改变图片状态
    def saveFile(self, _value=False, mode='Manual'):
        if self.filePath:
            # 获取图片标签索引并保存文件
            imgidx = self.getImglabelidx(self.filePath)
            self._saveFile(imgidx, mode=mode)

    # 保存锁定的形状
    def saveLockedShapes(self):
        # 清空锁定的形状和选中的形状
        self.canvas.lockedShapes = []
        self.canvas.selectedShapes = []
        # 遍历画布上的形状，将线条颜色为默认锁定颜色的形状添加到选中的形状中
        for s in self.canvas.shapes:
            if s.line_color == DEFAULT_LOCK_COLOR:
                self.canvas.selectedShapes.append(s)
        # 锁定选中的形状
        self.lockSelectedShape()
        # 再次遍历画布上的形状，将线条颜色为默认锁定颜色的形状从选中的形状和形状列表中移除
        for s in self.canvas.shapes:
            if s.line_color == DEFAULT_LOCK_COLOR:
                self.canvas.selectedShapes.remove(s)
                self.canvas.shapes.remove(s)
    # 保存标注文件，根据不同模式执行不同操作
    def _saveFile(self, annotationFilePath, mode='Manual'):
        # 如果有锁定的形状，则保存锁定的形状
        if len(self.canvas.lockedShapes) != 0:
            self.saveLockedShapes()

        # 如果是手动模式
        if mode == 'Manual':
            # 初始化锁定结果字典
            self.result_dic_locked = []
            # 读取图像文件
            img = cv2.imread(self.filePath)
            # 获取图像的宽度和高度
            width, height = self.image.width(), self.image.height()
            # 遍历锁定的形状
            for shape in self.canvas.lockedShapes:
                # 将形状的比例坐标转换为像素坐标
                box = [[int(p[0] * width), int(p[1] * height)] for p in shape['ratio']]
                # 创建结果列表，包含形状的文本和标记
                result = [(shape['transcription'], 1)]
                result.insert(0, box)
                # 将结果添加到锁定结果字典中
                self.result_dic_locked.append(result)
            # 将锁定结果字典合并到总结果字典中
            self.result_dic += self.result_dic_locked
            # 清空锁定结果字典
            self.result_dic_locked = []
            # 如果有注释文件路径，并且成功保存标签
            if annotationFilePath and self.saveLabels(annotationFilePath, mode=mode):
                # 设置为已保存状态
                self.setClean()
                # 在状态栏显示保存成功信息
                self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
                self.statusBar().show()
                # 更新文件列表中的图标
                currIndex = self.mImgList.index(self.filePath)
                item = self.fileListWidget.item(currIndex)
                item.setIcon(newIcon('done'))

                # 更新文件状态字典
                self.fileStatedict[self.filePath] = 1
                # 每达到自动保存数量时保存文件状态和标签
                if len(self.fileStatedict) % self.autoSaveNum == 0:
                    self.saveFilestate()
                    self.savePPlabel(mode='Auto')

                # 在当前位置插入文件列表项
                self.fileListWidget.insertItem(int(currIndex), item)
                # 如果不在同一图像中，则打开下一个图像
                if not self.canvas.isInTheSameImage:
                    self.openNextImg()
                # 启用保存推荐和保存标签的操作
                self.actions.saveRec.setEnabled(True)
                self.actions.saveLabel.setEnabled(True)
                self.actions.exportJSON.setEnabled(True) 

        # 如果是自动模式
        elif mode == 'Auto':
            # 如果有注释文件路径，并且成功保存标签
            if annotationFilePath and self.saveLabels(annotationFilePath, mode=mode):
                # 设置为已保存状态
                self.setClean()
                # 在状态栏显示保存成功信息
                self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
                self.statusBar().show()
    # 关闭文件操作，可选择是否保存
    def closeFile(self, _value=False):
        # 如果不能继续操作，则直接返回
        if not self.mayContinue():
            return
        # 重置状态
        self.resetState()
        # 设置为干净状态
        self.setClean()
        # 切换操作为关闭状态
        self.toggleActions(False)
        # 禁用画布
        self.canvas.setEnabled(False)
        # 禁用保存操作
        self.actions.saveAs.setEnabled(False)

    # 删除图片文件
    def deleteImg(self):
        # 获取要删除的文件路径
        deletePath = self.filePath
        # 如果文件路径不为空
        if deletePath is not None:
            # 弹出删除确认对话框
            deleteInfo = self.deleteImgDialog()
            # 如果确认删除
            if deleteInfo == QMessageBox.Yes:
                # 如果是 Windows 系统
                if platform.system() == 'Windows':
                    # 使用 win32com.shell 删除文件
                    from win32com.shell import shell, shellcon
                    shell.SHFileOperation((0, shellcon.FO_DELETE, deletePath, None,
                                           shellcon.FOF_SILENT | shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION,
                                           None, None))
                    # 如果是 Linux 系统
                elif platform.system() == 'Linux':
                    # 使用 trash 命令删除文件
                    cmd = 'trash ' + deletePath
                    os.system(cmd)
                    # 如果是 macOS 系统
                elif platform.system() == 'Darwin':
                    # 使用 osascript 命令将文件移动到回收站
                    import subprocess
                    absPath = os.path.abspath(deletePath).replace('\\', '\\\\').replace('"', '\\"')
                    cmd = ['osascript', '-e',
                           'tell app "Finder" to move {the POSIX file "' + absPath + '"} to trash']
                    print(cmd)
                    subprocess.call(cmd, stdout=open(os.devnull, 'w'))

                # 如果文件路径在文件状态字典中
                if self.filePath in self.fileStatedict.keys():
                    # 从文件状态字典中移除该文件路径
                    self.fileStatedict.pop(self.filePath)
                # 获取图片标签索引
                imgidx = self.getImglabelidx(self.filePath)
                # 如果图片标签索引在 PPlabel 中
                if imgidx in self.PPlabel.keys():
                    # 从 PPlabel 中移除该索引
                    self.PPlabel.pop(imgidx)
                # 打开下一个图片
                self.openNextImg()
                # 导入目录中的图片，标记为删除
                self.importDirImages(self.lastOpenDir, isDelete=True)
    # 弹出对话框询问用户是否删除图片，返回用户的选择
    def deleteImgDialog(self):
        # 定义对话框的按钮选项
        yes, cancel = QMessageBox.Yes, QMessageBox.Cancel
        # 设置对话框的提示信息
        msg = u'The image will be deleted to the recycle bin'
        # 弹出警告对话框，提示用户删除图片
        return QMessageBox.warning(self, u'Attention', msg, yes | cancel)

    # 重置所有设置并关闭当前窗口，重新启动程序
    def resetAll(self):
        # 重置所有设置
        self.settings.reset()
        # 关闭当前窗口
        self.close()
        # 创建一个新的进程对象
        proc = QProcess()
        # 启动一个独立的进程，重新运行当前文件
        proc.startDetached(os.path.abspath(__file__))

    # 检查是否可以继续操作
    def mayContinue(self):  #
        # 如果没有未保存的更改，直接返回 True
        if not self.dirty:
            return True
        else:
            # 弹出对话框询问用户是否放弃更改
            discardChanges = self.discardChangesDialog()
            if discardChanges == QMessageBox.No:
                return True
            elif discardChanges == QMessageBox.Yes:
                # 保存文件并继续操作
                self.canvas.isInTheSameImage = True
                self.saveFile()
                self.canvas.isInTheSameImage = False
                return True
            else:
                return False

    # 弹出对话框询问用户是否放弃更改
    def discardChangesDialog(self):
        # 定义对话框的按钮选项
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        # 根据语言设置对话框的提示信息
        if self.lang == 'ch':
            msg = u'您有未保存的变更, 您想保存再继续吗?\n点击 "No" 丢弃所有未保存的变更.'
        else:
            msg = u'You have unsaved changes, would you like to save them and proceed?\nClick "No" to undo all changes.'
        # 弹出警告对话框，提示用户是否放弃更改
        return QMessageBox.warning(self, u'Attention', msg, yes | no | cancel)

    # 弹出错误消息对话框
    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    # 返回当前文件的路径
    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    # 弹出颜色选择对话框，返回用户选择的颜色
    def chooseColor(self):
        # 获取用户选择的颜色
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        # 如果用户选择了颜色
        if color:
            # 更新线条颜色
            self.lineColor = color
            Shape.line_color = color
            # 设置画布的绘制颜色
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            # 标记文件已更改
            self.setDirty()
    # 删除选定的形状
    def deleteSelectedShape(self):
        # 从画布中删除选定的形状，并更新标签
        self.remLabels(self.canvas.deleteSelected())
        # 启用撤销操作
        self.actions.undo.setEnabled(True)
        # 设置为脏状态
        self.setDirty()
        # 如果没有形状存在
        if self.noShapes():
            # 禁用所有形状存在时的操作
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)
        # 更新 BoxListDock 和 labelListDock 的标题
        self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
        self.labelListDock.setWindowTitle(self.labelListDockName + f" ({self.labelList.count()})")

    # 更改形状的线条颜色
    def chshapeLineColor(self):
        # 获取用户选择的线条颜色
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color', default=DEFAULT_LINE_COLOR)
        if color:
            # 更新选定形状的线条颜色
            for shape in self.canvas.selectedShapes: shape.line_color = color
            # 更新画布显示
            self.canvas.update()
            # 设置为脏状态
            self.setDirty()

    # 更改形状的填充颜色
    def chshapeFillColor(self):
        # 获取用户选择的填充颜色
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color', default=DEFAULT_FILL_COLOR)
        if color:
            # 更新选定形状的填充颜色
            for shape in self.canvas.selectedShapes: shape.fill_color = color
            # 更新画布显示
            self.canvas.update()
            # 设置为脏状态
            self.setDirty()

    # 复制形状
    def copyShape(self):
        # 结束移动操作并复制选定的形状
        self.canvas.endMove(copy=True)
        # 添加标签
        self.addLabel(self.canvas.selectedShape)
        # 设置为脏状态
        self.setDirty()

    # 移动形状
    def moveShape(self):
        # 结束移动操作
        self.canvas.endMove(copy=False)
        # 设置为脏状态
        self.setDirty()

    # 加载预定义类别
    def loadPredefinedClasses(self, predefClassesFile):
        # 如果预定义类别文件存在
        if os.path.exists(predefClassesFile) is True:
            # 打开文件并逐行读取
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    # 如果标签历史为空，则创建列表并添加标签
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)
    # 切换绘制标签选项时的操作
    def togglePaintLabelsOption(self):
        # 取消显示索引选项的勾选状态
        self.displayIndexOption.setChecked(False)
        # 遍历画布上的所有形状
        for shape in self.canvas.shapes:
            # 设置形状是否绘制标签的状态
            shape.paintLabel = self.displayLabelOption.isChecked()
            # 设置形状是否绘制索引的状态
            shape.paintIdx = self.displayIndexOption.isChecked()
        # 重绘画布
        self.canvas.repaint()

    # 切换绘制索引选项时的操作
    def togglePaintIndexOption(self):
        # 取消显示标签选项的勾选状态
        self.displayLabelOption.setChecked(False)
        # 遍历画布上的所有形状
        for shape in self.canvas.shapes:
            # 设置形状是否绘制标签的状态
            shape.paintLabel = self.displayLabelOption.isChecked()
            # 设置形状是否绘制索引的状态
            shape.paintIdx = self.displayIndexOption.isChecked()
        # 重绘画布
        self.canvas.repaint()

    # 切换绘制正方形选项时的操作
    def toogleDrawSquare(self):
        # 设置画布绘制形状为正方形的状态
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())

    # 添加项目到列表中的操作
    def additems(self, dirpath):
        # 遍历图片列表中的文件
        for file in self.mImgList:
            # 从文件路径中获取文件名
            pix = QPixmap(file)
            _, filename = os.path.split(file)
            # 获取文件名并去除扩展名
            filename, _ = os.path.splitext(filename)
            # 创建列表项，包含缩略图和文件名
            item = QListWidgetItem(QIcon(pix.scaled(100, 100, Qt.IgnoreAspectRatio, Qt.FastTransformation)),
                                   filename[:10])
            # 设置列表项的工具提示为文件路径
            item.setToolTip(file)
            # 将列表项添加到图标列表中
            self.iconlist.addItem(item)
    # 为类中的图像列表中的每个文件添加条目
    def additems5(self, dirpath):
        # 遍历图像列表中的每个文件
        for file in self.mImgList5:
            # 从文件创建 QPixmap 对象
            pix = QPixmap(file)
            # 分离文件路径和文件名
            _, filename = os.path.split(file)
            # 分离文件名和扩展名
            filename, _ = os.path.splitext(filename)
            # 获取文件名的前10个字符
            pfilename = filename[:10]
            # 如果文件名长度小于10，则在两侧填充空格使其长度为10
            if len(pfilename) < 10:
                lentoken = 12 - len(pfilename)
                prelen = lentoken // 2
                bfilename = prelen * " " + pfilename + (lentoken - prelen) * " "
            # 创建 QListWidgetItem 对象，设置图标和文件名
            item = QListWidgetItem(QIcon(pix.scaled(100, 100, Qt.IgnoreAspectRatio, Qt.FastTransformation)), pfilename)
            # 设置条目的工具提示为文件路径
            item.setToolTip(file)
            # 将条目添加到图标列表中
            self.iconlist.addItem(item)
        # 计算所有条目的总宽度
        owidth = 0
        for index in range(len(self.mImgList5)):
            item = self.iconlist.item(index)
            itemwidget = self.iconlist.visualItemRect(item)
            owidth += itemwidget.width()
        # 设置图标列表的最小宽度为所有条目宽度之和加上50
        self.iconlist.setMinimumWidth(owidth + 50)
    def gen_quad_from_poly(self, poly):
        """
        从多边形生成最小面积四边形。
        """
        # 获取多边形的顶点数
        point_num = poly.shape[0]
        # 创建一个存储四个点坐标的数组
        min_area_quad = np.zeros((4, 2), dtype=np.float32)
        # 使用多边形的整数坐标创建最小外接矩形
        rect = cv2.minAreaRect(poly.astype(np.int32))  # (center (x,y), (width, height), angle of rotation)
        # 获取最小外接矩形的四个顶点坐标
        box = np.array(cv2.boxPoints(rect))

        # 初始化变量
        first_point_idx = 0
        min_dist = 1e4
        # 计算四个顶点到多边形顶点的距离和，找到距离最小的起始点
        for i in range(4):
            dist = np.linalg.norm(box[(i + 0) % 4] - poly[0]) + \
                   np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1]) + \
                   np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2]) + \
                   np.linalg.norm(box[(i + 3) % 4] - poly[-1])
            if dist < min_dist:
                min_dist = dist
                first_point_idx = i
        # 根据起始点顺序获取最小面积四边形的四个顶点坐标
        for i in range(4):
            min_area_quad[i] = box[(first_point_idx + i) % 4]

        # 将最小面积四边形的坐标转换为列表形式
        bbox_new = min_area_quad.tolist()
        bbox = []

        # 将坐标转换为整数类型，并添加到结果列表中
        for box in bbox_new:
            box = list(map(int, box))
            bbox.append(box)

        return bbox

    def getImglabelidx(self, filePath):
        # 根据操作系统类型设置路径分隔符
        if platform.system() == 'Windows':
            spliter = '\\'
        else:
            spliter = '/'
        # 根据路径分隔符将路径拆分为文件夹名和文件名
        filepathsplit = filePath.split(spliter)[-2:]
        # 返回文件夹名和文件名组成的路径
        return filepathsplit[0] + '/' + filepathsplit[1]
    # 自动识别图片中的文本内容
    def autoRecognition(self):
        # 断言图片列表不为空
        assert self.mImgList is not None
        # 打印所使用的模型
        print('Using model from ', self.model)

        # 获取未检查过的图片列表
        uncheckedList = [i for i in self.mImgList if i not in self.fileStatedict.keys()]
        # 创建自动识别对话框
        self.autoDialog = AutoDialog(parent=self, ocr=self.ocr, mImgList=uncheckedList, lenbar=len(uncheckedList))
        # 弹出对话框
        self.autoDialog.popUp()
        # 设置当前索引为图片列表长度减一
        self.currIndex = len(self.mImgList) - 1
        # 加载文件
        self.loadFile(self.filePath)  # ADD
        # 标记已进行自动识别
        self.haveAutoReced = True
        # 禁用自动识别按钮
        self.AutoRecognition.setEnabled(False)
        # 禁用相关操作
        self.actions.AutoRec.setEnabled(False)
        # 设置为脏状态
        self.setDirty()
        # 保存缓存标签
        self.saveCacheLabel()

        # 初始化键列表
        self.init_key_list(self.Cachelabel)

    # 单个重新识别
    def singleRerecognition(self):
        # 解码图片
        img = cv2.imdecode(np.fromfile(self.filePath,dtype=np.uint8),1)
        # 遍历选定的形状
        for shape in self.canvas.selectedShapes:
            # 获取形状的边界框
            box = [[int(p.x()), int(p.y())] for p in shape.points]
            # 如果边界框点数大于4，则生成四边形
            if len(box) > 4:
                box = self.gen_quad_from_poly(np.array(box))
            # 断言边界框点数为4
            assert len(box) == 4
            # 裁剪图片
            img_crop = get_rotate_crop_image(img, np.array(box, np.float32))
            # 如果裁剪后的图片为空
            if img_crop is None:
                msg = 'Can not recognise the detection box in ' + self.filePath + '. Please change manually'
                # 弹出消息框
                QMessageBox.information(self, "Information", msg)
                return
            # 使用OCR识别文本
            result = self.ocr.ocr(img_crop, cls=True, det=False)[0]
            # 如果识别结果不为空
            if result[0][0] != '':
                result.insert(0, box)
                print('result in reRec is ', result)
                # 如果识别结果与形状标签相同
                if result[1][0] == shape.label:
                    print('label no change')
                else:
                    shape.label = result[1][0]
            else:
                print('Can not recognise the box')
                # 如果无文本标签与形状标签相同
                if self.noLabelText == shape.label:
                    print('label no change')
                else:
                    shape.label = self.noLabelText
            # 更新单个标签
            self.singleLabel(shape)
            # 设置为脏状态
            self.setDirty()
    # 创建垂直布局管理器
    vbox = QVBoxLayout()
    # 创建水平布局管理器
    hbox = QHBoxLayout()
    # 创建标签并设置文本内容
    self.panel = QLabel()
    self.panel.setText(self.stringBundle.getString('choseModelLg'))
    self.panel.setAlignment(Qt.AlignLeft)
    # 创建下拉框并添加选项
    self.comboBox = QComboBox()
    self.comboBox.setObjectName("comboBox")
    self.comboBox.addItems(['Chinese & English', 'English', 'French', 'German', 'Korean', 'Japanese'])
    # 将标签和下拉框添加到垂直布局管理器中
    vbox.addWidget(self.panel)
    vbox.addWidget(self.comboBox)
    # 创建对话框并设置大小
    self.dialog = QDialog()
    self.dialog.resize(300, 100)
    # 创建确定和取消按钮
    self.okBtn = QPushButton(self.stringBundle.getString('ok'))
    self.cancelBtn = QPushButton(self.stringBundle.getString('cancel'))

    # 点击确定按钮连接到 modelChoose 方法
    self.okBtn.clicked.connect(self.modelChoose)
    # 点击取消按钮连接到 cancel 方法
    self.cancelBtn.clicked.connect(self.cancel)
    # 设置对话框标题
    self.dialog.setWindowTitle(self.stringBundle.getString('choseModelLg'))

    # 将确定和取消按钮添加到水平布局管理器中
    hbox.addWidget(self.okBtn)
    hbox.addWidget(self.cancelBtn)

    # 将标签和水平布局管理器添加到垂直布局管理器中
    vbox.addWidget(self.panel)
    vbox.addLayout(hbox)
    # 设置对话框的布局为垂直布局管理器
    self.dialog.setLayout(vbox)
    # 设置对话框为应用程序模态
    self.dialog.setWindowModality(Qt.ApplicationModal)
    # 执行对话框
    self.dialog.exec_()
    # 如果存在文件路径，则启用自动识别功能
    if self.filePath:
        self.AutoRecognition.setEnabled(True)
        self.actions.AutoRec.setEnabled(True)
    # 选择模型的语言类型，并打印当前选择的语言
    def modelChoose(self):
        print(self.comboBox.currentText())
        # 定义不同语言对应的索引字典
        lg_idx = {'Chinese & English': 'ch', 'English': 'en', 'French': 'french', 'German': 'german',
                  'Korean': 'korean', 'Japanese': 'japan'}
        # 删除之前的 OCR 对象
        del self.ocr
        # 创建新的 PaddleOCR 对象，根据选择的语言类型进行初始化
        self.ocr = PaddleOCR(use_pdserving=False, use_angle_cls=True, det=True, cls=True, use_gpu=False,
                             lang=lg_idx[self.comboBox.currentText()])
        # 删除之前的 table_ocr 对象
        del self.table_ocr
        # 创建新的 PPStructure 对象，根据选择的语言类型进行初始化
        self.table_ocr = PPStructure(use_pdserving=False,
                                     use_gpu=False,
                                     lang=lg_idx[self.comboBox.currentText()],
                                     layout=False,
                                     show_log=False)
        # 关闭当前对话框
        self.dialog.close()

    # 取消操作，关闭当前对话框
    def cancel(self):
        self.dialog.close()

    # 加载文件状态，根据保存目录加载文件状态信息
    def loadFilestate(self, saveDir):
        # 设置文件状态文件路径
        self.fileStatepath = saveDir + '/fileState.txt'
        # 初始化文件状态字典
        self.fileStatedict = {}
        # 如果文件状态文件不存在，则创建一个新文件
        if not os.path.exists(self.fileStatepath):
            f = open(self.fileStatepath, 'w', encoding='utf-8')
        else:
            # 读取文件状态文件内容
            with open(self.fileStatepath, 'r', encoding='utf-8') as f:
                states = f.readlines()
                # 遍历文件状态信息，更新文件状态字典
                for each in states:
                    file, state = each.split('\t')
                    self.fileStatedict[file] = 1
                # 启用保存标签、保存记录和导出 JSON 操作
                self.actions.saveLabel.setEnabled(True)
                self.actions.saveRec.setEnabled(True)
                self.actions.exportJSON.setEnabled(True)

    # 保存文件状态信息到文件
    def saveFilestate(self):
        # 将文件状态字典内容写入文件
        with open(self.fileStatepath, 'w', encoding='utf-8') as f:
            for key in self.fileStatedict:
                f.write(key + '\t')
                f.write(str(self.fileStatedict[key]) + '\n')
    # 从标签文件中加载标签数据，返回文件名到标签数据的字典
    def loadLabelFile(self, labelpath):
        labeldict = {}
        # 如果标签文件不存在，则创建一个新的空文件
        if not os.path.exists(labelpath):
            f = open(labelpath, 'w', encoding='utf-8')
        # 如果标签文件存在，则读取其中的数据
        else:
            with open(labelpath, 'r', encoding='utf-8') as f:
                data = f.readlines()
                # 遍历每一行数据，提取文件名和标签信息
                for each in data:
                    file, label = each.split('\t')
                    # 如果标签信息存在，则替换其中的字符串并转换为对应的数据类型
                    if label:
                        label = label.replace('false', 'False')
                        label = label.replace('true', 'True')
                        labeldict[file] = eval(label)
                    # 如果标签信息为空，则将文件名对应的标签设为空列表
                    else:
                        labeldict[file] = []
        # 返回标签数据字典
        return labeldict

    # 将已经处理过的图片标签保存到文件中
    def savePPlabel(self, mode='Manual'):
        # 获取已保存的文件列表
        savedfile = [self.getImglabelidx(i) for i in self.fileStatedict.keys()]
        # 打开文件，将图片标签信息写入文件
        with open(self.PPlabelpath, 'w', encoding='utf-8') as f:
            for key in self.PPlabel:
                # 如果图片在已保存文件列表中且标签不为空，则写入文件
                if key in savedfile and self.PPlabel[key] != []:
                    f.write(key + '\t')
                    f.write(json.dumps(self.PPlabel[key], ensure_ascii=False) + '\n')

        # 如果模式为手动保存，则显示保存信息提示框
        if mode == 'Manual':
            if self.lang == 'ch':
                msg = '已将检查过的图片标签保存在 ' + self.PPlabelpath + " 文件中"
            else:
                msg = 'Images that have been checked are saved in ' + self.PPlabelpath
            QMessageBox.information(self, "Information", msg)

    # 将缓存中的标签数据保存到文件中
    def saveCacheLabel(self):
        # 打开文件，将缓存中的标签数据写入文件
        with open(self.Cachelabelpath, 'w', encoding='utf-8') as f:
            for key in self.Cachelabel:
                f.write(key + '\t')
                f.write(json.dumps(self.Cachelabel[key], ensure_ascii=False) + '\n')

    # 保存标签文件，包括保存文件状态和已处理过的图片标签
    def saveLabelFile(self):
        # 保存文件状态
        self.saveFilestate()
        # 保存已处理过的图片标签
        self.savePPlabel()
    # 保存识别结果的方法
    def saveRecResult(self):
        # 检查是否有缺失的路径或数据，如果有则弹出提示框并返回
        if {} in [self.PPlabelpath, self.PPlabel, self.fileStatedict]:
            QMessageBox.information(self, "Information", "Check the image first")
            return

        # 设置保存识别结果的文件路径
        rec_gt_dir = os.path.dirname(self.PPlabelpath) + '/rec_gt.txt'
        crop_img_dir = os.path.dirname(self.PPlabelpath) + '/crop_img/'
        ques_img = []
        # 如果裁剪图像文件夹不存在，则创建
        if not os.path.exists(crop_img_dir):
            os.mkdir(crop_img_dir)

        # 打开文件，准备写入识别结果
        with open(rec_gt_dir, 'w', encoding='utf-8') as f:
            # 遍历文件状态字典中的每个键
            for key in self.fileStatedict:
                # 获取图像标签的索引
                idx = self.getImglabelidx(key)
                try:
                    # 读取图像文件
                    img = cv2.imread(key)
                    # 遍历每个标签
                    for i, label in enumerate(self.PPlabel[idx]):
                        # 如果标签为困难，则跳过
                        if label['difficult']:
                            continue
                        # 获取裁剪后的图像
                        img_crop = get_rotate_crop_image(img, np.array(label['points'], np.float32))
                        # 设置裁剪后图像的文件名
                        img_name = os.path.splitext(os.path.basename(idx))[0] + '_crop_' + str(i) + '.jpg'
                        # 保存裁剪后的图像
                        cv2.imwrite(crop_img_dir + img_name, img_crop)
                        # 写入裁剪后图像的路径和标签到文件
                        f.write('crop_img/' + img_name + '\t')
                        f.write(label['transcription'] + '\n')
                except Exception as e:
                    # 记录无法读取的图像
                    ques_img.append(key)
                    print("Can not read image ", e)
        # 如果有无法保存的图像，则弹出提示框
        if ques_img:
            QMessageBox.information(self,
                                    "Information",
                                    "The following images can not be saved, please check the image path and labels.\n"
                                    + "".join(str(i) + '\n' for i in ques_img))
        # 弹出提示框，显示裁剪后图像保存的路径
        QMessageBox.information(self, "Information", "Cropped images have been saved in " + str(crop_img_dir))
    # 根据用户选择的选项来设置绘图速度
    def speedChoose(self):
        # 如果选择了标签对话框选项
        if self.labelDialogOption.isChecked():
            # 断开原有的连接，连接到新的函数，传入参数为True
            self.canvas.newShape.disconnect()
            self.canvas.newShape.connect(partial(self.newShape, True))
        else:
            # 断开原有的连接，连接到新的函数，传入参数为False
            self.canvas.newShape.disconnect()
            self.canvas.newShape.connect(partial(self.newShape, False))

    # 自动保存功能
    def autoSaveFunc(self):
        # 如果自动保存选项被选中
        if self.autoSaveOption.isChecked():
            # 设置自动保存次数为1，尝试保存标签文件
            self.autoSaveNum = 1  # Real auto_Save
            try:
                self.saveLabelFile()
            except:
                pass
            print('The program will automatically save once after confirming an image')
        else:
            # 设置自动保存次数为5，用于备份
            self.autoSaveNum = 5  # Used for backup
            print('The program will automatically save once after confirming 5 images (default)')

    # 更改标签键
    def change_box_key(self):
        # 如果不是关键模式，则返回
        if not self.kie_mode:
            return
        # 弹出对话框，获取用户输入的键值
        key_text, _ = self.keyDialog.popUp(self.key_previous_text)
        if key_text is None:
            return
        self.key_previous_text = key_text
        # 遍历所选形状，设置键值，并更新界面
        for shape in self.canvas.selectedShapes:
            shape.key_cls = key_text
            if not self.keyList.findItemsByLabel(key_text):
                item = self.keyList.createItemFromLabel(key_text)
                self.keyList.addItem(item)
                rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                self.keyList.setItemLabel(item, key_text, rgb)

            self._update_shape_color(shape)
            self.keyDialog.addLabelHistory(key_text)
            
        # 保存更改后的形状
        self.setDirty()

    # 撤销形状编辑
    def undoShapeEdit(self):
        # 恢复形状，清空标签列表、索引列表和框列表，加载形状，设置撤销按钮是否可用
        self.canvas.restoreShape()
        self.labelList.clear()
        self.indexList.clear()
        self.BoxList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
    # 加载形状到画布中，可以选择是否替换原有形状
    def loadShapes(self, shapes, replace=True):
        # 禁用选择槽
        self._noSelectionSlot = True
        # 遍历每个形状，添加标签
        for shape in shapes:
            self.addLabel(shape)
        # 清除标签列表的选择
        self.labelList.clearSelection()
        # 清除索引列表的选择
        self.indexList.clearSelection()
        # 启用选择槽
        self._noSelectionSlot = False
        # 调用画布对象的loadShapes方法，加载形状到画布中
        self.canvas.loadShapes(shapes, replace=replace)
        # 打印信息，表示加载形状完成
        print("loadShapes")  # 1
    def lockSelectedShape(self):
        """锁定选定的形状。

        将 self.selectedShapes 添加到 self.canvas.lockedShapes 中，其中包含被锁定形状的四个坐标与图像宽度和高度的比率
        """
        # 获取图像的宽度和高度
        width, height = self.image.width(), self.image.height()

        def format_shape(s):
            return dict(label=s.label,  # 字符串
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        ratio=[[int(p.x()) / width, int(p.y()) / height] for p in s.points],  # QPonitF
                        difficult=s.difficult,  # 布尔值
                        key_cls=s.key_cls,  # 布尔值
                        )

        # 锁定
        if len(self.canvas.lockedShapes) == 0:
            for s in self.canvas.selectedShapes:
                s.line_color = DEFAULT_LOCK_COLOR
                s.locked = True
            shapes = [format_shape(shape) for shape in self.canvas.selectedShapes]
            trans_dic = []
            for box in shapes:
                trans_dict = {"transcription": box['label'], "ratio": box['ratio'], "difficult": box['difficult']}
                if self.kie_mode:
                    trans_dict.update({"key_cls": box["key_cls"]})
                trans_dic.append(trans_dict)
            self.canvas.lockedShapes = trans_dic
            self.actions.save.setEnabled(True)

        # 解锁
        else:
            for s in self.canvas.shapes:
                s.line_color = DEFAULT_LINE_COLOR
            self.canvas.lockedShapes = []
            self.result_dic_locked = []
            self.setDirty()
            self.actions.save.setEnabled(True)
# 定义一个函数，将颜色值取反，返回一个新的颜色对象
def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


# 定义一个函数，读取指定文件的内容，以二进制形式返回，如果读取失败则返回默认值
def read(filename, default=None):
    try:
        # 使用 with 语句打开文件，读取文件内容
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


# 定义一个函数，将字符串转换为布尔值
def str2bool(v):
    return v.lower() in ("true", "t", "1")


# 定义一个函数，获取主应用程序对象
def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    # 创建一个 Qt 应用程序对象
    app = QApplication(argv)
    # 设置应用程序名称
    app.setApplicationName(__appname__)
    # 设置应用程序图标
    app.setWindowIcon(newIcon("app"))
    # 创建命令行参数解析器
    arg_parser = argparse.ArgumentParser()
    # 添加命令行参数
    arg_parser.add_argument("--lang", type=str, default='en', nargs="?")
    arg_parser.add_argument("--gpu", type=str2bool, default=True, nargs="?")
    arg_parser.add_argument("--kie", type=str2bool, default=False, nargs="?")
    arg_parser.add_argument("--predefined_classes_file",
                            default=os.path.join(os.path.dirname(__file__), "data", "predefined_classes.txt"),
                            nargs="?")
    # 解析命令行参数
    args = arg_parser.parse_args(argv[1:])

    # 创建主窗口对象
    win = MainWindow(lang=args.lang,
                     gpu=args.gpu,
                     kie_mode=args.kie,
                     default_predefined_class_file=args.predefined_classes_file)
    # 显示主窗口
    win.show()
    return app, win


# 定义一个函数，构建主应用程序并运行
def main():
    """construct main app and run it"""
    # 获取主应用程序对象和主窗口对象
    app, _win = get_main_app(sys.argv)
    # 运行应用程序
    return app.exec_()


# 如果当前脚本为主程序入口
if __name__ == '__main__':

    # 检查资源文件是否存在，如果不存在则生成资源文件
    resource_file = './libs/resources.py'
    if not os.path.exists(resource_file):
        output = os.system('pyrcc5 -o libs/resources.py resources.qrc')
        assert output == 0, "operate the cmd have some problems ,please check  whether there is a in the lib " \
                            "directory resources.py "

    # 退出程序并返回应用程序的退出代码
    sys.exit(main())
```