# `.\comic-translate\app\ui\dayu_widgets\qt\__init__.py`

```py
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.3
# Email : muyanru345@163.com
###################################################################

# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import built-in modules
import contextlib
import signal
import sys

# Import third-party modules
from PySide6 import QtCore  # 导入 PySide6 中的 QtCore 模块
from PySide6 import QtGui   # 导入 PySide6 中的 QtGui 模块
from PySide6 import QtWidgets  # 导入 PySide6 中的 QtWidgets 模块
from PySide6.QtGui import QGuiApplication  # 导入 PySide6.QtGui 中的 QGuiApplication 类
from PySide6.QtSvg import QSvgRenderer   # 导入 PySide6.QtSvg 中的 QSvgRenderer 类
import six   # 导入 six 库

class MCacheDict(object):
    _render = QSvgRenderer()  # 创建一个 QSvgRenderer 对象 _render

    def __init__(self, cls):
        super(MCacheDict, self).__init__()
        self.cls = cls  # 初始化实例属性 cls，表示要缓存的对象类型
        self._cache_pix_dict = {}  # 初始化实例属性 _cache_pix_dict，用于缓存像素图对象

    def _render_svg(self, svg_path, replace_color=None):
        # Import local modules
        from .. import dayu_theme  # 从本地模块导入 dayu_theme

        replace_color = replace_color or dayu_theme.icon_color  # 如果未指定替换颜色，则使用 dayu_theme 中的 icon_color
        if (self.cls is QtGui.QIcon) and (replace_color is None):
            return QtGui.QIcon(svg_path)  # 如果缓存对象类型是 QIcon 并且没有替换颜色，则返回 QIcon 对象
        with open(svg_path, "r") as f:
            data_content = f.read()  # 读取 SVG 文件内容
            if replace_color is not None:
                data_content = data_content.replace("#555555", replace_color)  # 替换 SVG 文件中的颜色信息
            self._render.load(QtCore.QByteArray(six.b(data_content)))  # 将 SVG 数据加载到 _render 中
            pix = QtGui.QPixmap(128, 128)  # 创建一个大小为 128x128 的 QPixmap 对象
            pix.fill(QtCore.Qt.transparent)  # 填充 QPixmap 对象为透明
            painter = QtGui.QPainter(pix)  # 创建一个 QPainter 对象，并使用 QPixmap 进行初始化
            self._render.render(painter)  # 将 _render 中的 SVG 渲染到 QPixmap 中
            painter.end()  # 结束 QPainter 的绘制过程
            if self.cls is QtGui.QPixmap:
                return pix  # 如果缓存对象类型是 QPixmap，则返回 QPixmap 对象
            else:
                return self.cls(pix)  # 否则，使用 QPixmap 创建相应的对象并返回

    def __call__(self, path, color=None):
        # Import local modules
        from .. import utils  # 从本地模块导入 utils

        full_path = utils.get_static_file(path)  # 获取静态文件的完整路径
        if full_path is None:
            return self.cls()  # 如果路径为空，则返回一个空的缓存对象
        key = "{}{}".format(full_path.lower(), color or "")  # 生成缓存对象的键，包含路径和颜色信息
        pix_map = self._cache_pix_dict.get(key, None)  # 从缓存中获取对应键的像素图对象
        if pix_map is None:
            if full_path.endswith("svg"):
                pix_map = self._render_svg(full_path, color)  # 如果文件以 svg 结尾，则渲染 SVG 文件
            else:
                pix_map = self.cls(full_path)  # 否则，使用路径创建相应的对象
            self._cache_pix_dict.update({key: pix_map})  # 更新缓存字典
        return pix_map  # 返回获取到的像素图对象


def get_scale_factor():
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication([])  # 如果没有实例，则创建一个 QtWidgets.QApplication 实例
    standard_dpi = 96.0  # 标准 DPI 值
    scale_factor_x = QGuiApplication.primaryScreen().logicalDotsPerInchX() / standard_dpi  # 计算 X 方向的缩放因子
    scale_factor_y = QGuiApplication.primaryScreen().logicalDotsPerInchX() / standard_dpi  # 计算 Y 方向的缩放因子
    return scale_factor_x, scale_factor_y  # 返回计算得到的缩放因子


@contextlib.contextmanager
def application(*args):
    app = QtWidgets.QApplication.instance()  # 获取当前 QtWidgets.QApplication 实例

    if not app:
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # 设置信号处理器，处理 SIGINT 信号
        app = QtWidgets.QApplication(sys.argv)  # 创建 QtWidgets.QApplication 实例
        yield app  # 返回应用实例
        app.exec_()  # 执行应用事件循环
    else:
        yield app  # 返回当前应用实例


MPixmap = MCacheDict(QtGui.QPixmap)  # 创建 MCacheDict 类的实例 MPixmap，并指定缓存对象类型为 QtGui.QPixmap
# 创建一个 MCacheDict 对象，使用 QtGui.QIcon 作为默认值的类型参数 MIcon
MIcon = MCacheDict(QtGui.QIcon)
```