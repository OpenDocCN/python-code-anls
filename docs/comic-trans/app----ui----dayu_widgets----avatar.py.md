# `.\comic-translate\app\ui\dayu_widgets\avatar.py`

```py
"""
MAvatar.
"""
# 导入未来模块，确保代码向后兼容性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入第三方模块
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

# 导入本地模块
from . import dayu_theme
from .qt import MPixmap

class MAvatar(QtWidgets.QLabel):
    """
    Avatar component. It can be used to represent people or object.

    Property:
        image: avatar image, should be QPixmap.
        dayu_size: the size of image.
    """

    def __init__(self, parent=None, flags=QtCore.Qt.Widget):
        """
        Initialize MAvatar object.
        
        :param parent: Parent widget, default is None.
        :param flags: Widget flags, default is QtCore.Qt.Widget.
        """
        super(MAvatar, self).__init__(parent, flags)
        # 设置默认图像为"user_fill.svg"
        self._default_pix = MPixmap("user_fill.svg")
        self._pixmap = self._default_pix
        self._dayu_size = 0
        # 设置默认大小为dayu_theme.default_size
        self.set_dayu_size(dayu_theme.default_size)

    def set_dayu_size(self, value):
        """
        Set the avatar size.
        
        :param value: Integer value representing size.
        :return: None
        """
        self._dayu_size = value
        self._set_dayu_size()

    def _set_dayu_size(self):
        """
        Set the fixed size of the avatar widget.
        """
        self.setFixedSize(QtCore.QSize(self._dayu_size, self._dayu_size))  # 设置固定大小

    def _set_dayu_image(self):
        """
        Set the pixmap of the avatar with aspect ratio and smooth transformation.
        """
        self.setPixmap(self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def set_dayu_image(self, value):
        """
        Set avatar image.
        
        :param value: QPixmap or None.
        :return: None
        :raises TypeError: If value is not QPixmap or None.
        """
        if value is None:
            self._pixmap = self._default_pix
        elif isinstance(value, QtGui.QPixmap):
            self._pixmap = self._default_pix if value.isNull() else value
        else:
            raise TypeError("Input argument 'value' should be QPixmap or None, but get {}".format(type(value)))
        self._set_dayu_image()

    def get_dayu_image(self):
        """
        Get the avatar image.
        
        :return: QPixmap
        """
        return self._pixmap

    def get_dayu_size(self):
        """
        Get the avatar size.
        
        :return: Integer
        """
        return self._dayu_size

    # 定义属性dayu_image，用于获取和设置avatar的图像
    dayu_image = QtCore.Property(QtGui.QPixmap, get_dayu_image, set_dayu_image)
    # 定义属性dayu_size，用于获取和设置avatar的大小
    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    @classmethod
    def huge(cls, image=None):
        """
        Create a MAvatar with huge size.
        
        :param image: Optional QPixmap image, default is None.
        :return: Instance of MAvatar with huge size.
        """
        inst = cls()
        inst.set_dayu_size(dayu_theme.huge)
        inst.set_dayu_image(image)
        return inst

    @classmethod
    @classmethod
    def large(cls, image=None):
        """Create a MAvatar with large size"""
        # 创建一个大尺寸的 MAvatar 实例
        inst = cls()
        # 设置实例的尺寸为大尺寸
        inst.set_dayu_size(dayu_theme.large)
        # 设置实例的图像
        inst.set_dayu_image(image)
        return inst

    @classmethod
    def medium(cls, image=None):
        """Create a MAvatar with medium size"""
        # 创建一个中等尺寸的 MAvatar 实例
        inst = cls()
        # 设置实例的尺寸为中等尺寸
        inst.set_dayu_size(dayu_theme.medium)
        # 设置实例的图像
        inst.set_dayu_image(image)
        return inst

    @classmethod
    def small(cls, image=None):
        """Create a MAvatar with small size"""
        # 创建一个小尺寸的 MAvatar 实例
        inst = cls()
        # 设置实例的尺寸为小尺寸
        inst.set_dayu_size(dayu_theme.small)
        # 设置实例的图像
        inst.set_dayu_image(image)
        return inst

    @classmethod
    def tiny(cls, image=None):
        """Create a MAvatar with tiny size"""
        # 创建一个微小尺寸的 MAvatar 实例
        inst = cls()
        # 设置实例的尺寸为微小尺寸
        inst.set_dayu_size(dayu_theme.tiny)
        # 设置实例的图像
        inst.set_dayu_image(image)
        return inst
```