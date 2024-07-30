# `.\comic-translate\app\ui\dayu_widgets\browser.py`

```py
"""
MClickBrowserFilePushButton, MClickBrowserFileToolButton
MClickBrowserFolderPushButton, MClickBrowserFolderToolButton
Browser files or folders by selecting.

MDragFileButton, MDragFolderButton
Browser files or folders by dragging.
"""

# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import built-in modules
import os

# Import third-party modules
from PySide6 import QtCore
from PySide6 import QtWidgets
import six

# Import local modules
from . import dayu_theme
from .mixin import cursor_mixin
from .mixin import property_mixin
from .push_button import MPushButton
from .tool_button import MToolButton


# NOTE PySide2 Crash without QObject wrapper

# Slot function to handle file browsing
def _slot_browser_file(self):
    # Prepare filter list based on dayu filters
    filter_list = (
        "File(%s)" % (" ".join(["*" + e for e in self.get_dayu_filters()]))
        if self.get_dayu_filters()
        else "Any File(*)"
    )
    
    # Open file dialog for multiple files if specified
    if self.get_dayu_multiple():
        r_files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Browser File", self.get_dayu_path(), filter_list)
        if r_files:
            # Emit signal with list of selected files
            self.sig_files_changed.emit(r_files)
            # Set the path to the first selected file
            self.set_dayu_path(r_files[0])
    else:
        # Open file dialog for single file selection
        r_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Browser File", self.get_dayu_path(), filter_list)
        if r_file:
            # Emit signal with selected file path
            self.sig_file_changed.emit(r_file)
            # Set the path to the selected file
            self.set_dayu_path(r_file)


# Slot function to handle folder browsing
def _slot_browser_folder(self):
    # Open folder dialog to select a folder
    r_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Browser Folder", self.get_dayu_path())
    if r_folder:
        # Emit signal with selected folder path
        if self.get_dayu_multiple():
            self.sig_folders_changed.emit([r_folder])
        else:
            self.sig_folder_changed.emit(r_folder)
        # Set the path to the selected folder
        self.set_dayu_path(r_folder)


# Class definition for a clickable push button to browse files
class MClickBrowserFilePushButton(MPushButton):
    """A Clickable push button to browser files"""

    # Signal emitted when a file is changed
    sig_file_changed = QtCore.Signal(str)
    # Signal emitted when files are changed (multiple selection)
    sig_files_changed = QtCore.Signal(list)
    # Assign the slot function for browsing files
    slot_browser_file = _slot_browser_file
    # 初始化函数，设置按钮文本，默认是否允许多选，和父对象
    def __init__(self, text="Browser", multiple=False, parent=None):
        # 调用父类的初始化函数，设置按钮文本和父对象
        super(MClickBrowserFilePushButton, self).__init__(text=text, parent=parent)
        # 设置自定义属性"multiple"，表示是否允许多选文件
        self.setProperty("multiple", multiple)
        # 连接按钮的点击事件到自定义的文件浏览方法
        self.clicked.connect(self.slot_browser_file)
        # 设置按钮的工具提示
        self.setToolTip(self.tr("Click to browser file"))

        # 初始化私有变量
        self._path = None  # 文件浏览器的当前路径
        self._multiple = multiple  # 是否允许多选文件
        self._filters = []  # 文件格式过滤器列表

    # 获取文件格式过滤器列表
    def get_dayu_filters(self):
        """
        Get browser's format filters
        :return: list
        """
        return self._filters

    # 设置文件格式过滤器列表
    def set_dayu_filters(self, value):
        """
        Set browser file format filters
        :param value:
        :return: None
        """
        self._filters = value

    # 获取最近一次浏览的文件路径
    def get_dayu_path(self):
        """
        Get last browser file path
        :return: str
        """
        return self._path

    # 设置文件浏览器的起始路径
    def set_dayu_path(self, value):
        """
        Set browser file start path
        :param value: str
        :return: None
        """
        self._path = value

    # 获取是否允许多选文件
    def get_dayu_multiple(self):
        """
        Get browser can select multiple file or not
        :return: bool
        """
        return self._multiple

    # 设置是否允许多选文件
    def set_dayu_multiple(self, value):
        """
        Set browser can select multiple file or not
        :param value: bool
        :return: None
        """
        self._multiple = value

    # 定义多选文件属性
    dayu_multiple = QtCore.Property(bool, get_dayu_multiple, set_dayu_multiple)
    # 定义文件路径属性
    dayu_path = QtCore.Property(six.string_types[0], get_dayu_path, set_dayu_path)
    # 定义文件格式过滤器属性
    dayu_filters = QtCore.Property(list, get_dayu_filters, set_dayu_filters)
class MClickBrowserFileToolButton(MToolButton):
    """A Clickable tool button to browser files"""

    # Signal emitted when a single file is selected
    sig_file_changed = QtCore.Signal(str)
    # Signal emitted when multiple files are selected
    sig_files_changed = QtCore.Signal(list)
    # Assign slot function for file browsing
    slot_browser_file = _slot_browser_file

    def __init__(self, multiple=False, parent=None):
        super(MClickBrowserFileToolButton, self).__init__(parent=parent)
        # Set the icon using a Dayu SVG icon
        self.set_dayu_svg("cloud_line.svg")
        # Display only the icon on the button
        self.icon_only()
        # Connect the 'clicked' signal of the button to slot_browser_file method
        self.clicked.connect(self.slot_browser_file)
        # Set the tooltip for the button
        self.setToolTip(self.tr("Click to browser file"))

        # Initialize instance variables
        self._path = None
        self._multiple = multiple
        self._filters = []

    def get_dayu_filters(self):
        """
        Get browser's format filters
        :return: list
        """
        return self._filters

    def set_dayu_filters(self, value):
        """
        Set browser file format filters
        :param value: list
        :return: None
        """
        self._filters = value

    def get_dayu_path(self):
        """
        Get last browser file path
        :return: str
        """
        return self._path

    def set_dayu_path(self, value):
        """
        Set browser file start path
        :param value: str
        :return: None
        """
        self._path = value

    def get_dayu_multiple(self):
        """
        Get whether browser can select multiple files or not
        :return: bool
        """
        return self._multiple

    def set_dayu_multiple(self, value):
        """
        Set whether browser can select multiple files or not
        :param value: bool
        :return: None
        """
        self._multiple = value

    # Define dayu_multiple property using QtCore.Property
    dayu_multiple = QtCore.Property(bool, get_dayu_multiple, set_dayu_multiple)
    # Define dayu_path property using QtCore.Property
    dayu_path = QtCore.Property(six.string_types[0], get_dayu_path, set_dayu_path)
    # Define dayu_filters property using QtCore.Property
    dayu_filters = QtCore.Property(list, get_dayu_filters, set_dayu_filters)


# Modified the Original Dayu Class
class MClickSaveFileToolButton(MToolButton):
    """A Clickable tool button to browse and save files"""

    # Signal emitted when a file is saved
    sig_file_changed = QtCore.Signal(str)

    def __init__(self, file_types=None, parent=None):
        super(MClickSaveFileToolButton, self).__init__(parent=parent)
        # Set the icon using a Dayu SVG icon for save operation
        self.set_dayu_svg("save.svg")
        # Display only the icon on the button
        self.icon_only()
        # Connect the 'clicked' signal of the button to slot_save_file method
        self.clicked.connect(self.slot_save_file)
        # Set the tooltip for the button
        self.setToolTip(self.tr("Click to save file"))

        # Initialize instance variables
        self._path = ""
        self._file_types = file_types or []
        self._last_used_filter = ""
        # Initialize last used directory with user's home directory
        self._last_directory = os.path.expanduser('~')

    def get_file_types(self):
        """
        Get file types allowed for saving
        :return: list
        """
        return self._file_types

    def set_file_types(self, value):
        """
        Set file types allowed for saving
        :param value: list
        :return: None
        """
        self._file_types = value

    def get_path(self):
        """
        Get the current save file path
        :return: str
        """
        return self._path

    def set_path(self, value):
        """
        Set the current save file path
        :param value: str
        :return: None
        """
        self._path = value
        # Update the last used directory based on the new path
        if value:
            self._last_directory = os.path.dirname(value)

    # Define path property using QtCore.Property
    path = QtCore.Property(str, get_path, set_path)
    # Define file_types property using QtCore.Property
    file_types = QtCore.Property(list, get_file_types, set_file_types)
    # 创建用于文件选择对话框的筛选字符串，根据已配置的文件类型生成
    def _create_filter_string(self):
        filter_parts = []
        # 遍历每种文件类型
        for file_type in self._file_types:
            # 如果文件类型的扩展名是一个列表
            if isinstance(file_type[1], list):
                # 生成多个扩展名的字符串，以空格分隔
                extensions = " ".join(f"*.{ext}" for ext in file_type[1])
                filter_parts.append(f"{file_type[0]} ({extensions})")
            else:
                # 生成单个扩展名的字符串
                filter_parts.append(f"{file_type[0]} (*.{file_type[1]})")
        # 使用双分号连接所有文件类型字符串
        return ";;".join(filter_parts)

    # 获取默认文件扩展名
    def _get_default_extension(self):
        # 如果已配置文件类型列表不为空
        if self._file_types:
            # 如果第一个文件类型的扩展名是一个列表，则返回列表中的第一个扩展名
            if isinstance(self._file_types[0][1], list):
                return self._file_types[0][1][0]
            else:
                # 否则返回第一个文件类型的扩展名
                return self._file_types[0][1]
        # 如果文件类型列表为空，则返回空字符串
        return ""

    # 槽函数，用于保存文件操作
    def slot_save_file(self):
        # 创建文件选择对话框的筛选字符串
        filter_string = self._create_filter_string()
        # 创建默认文件名，使用第一个文件类型的默认扩展名
        default_name = f"untitled.{self._get_default_extension()}"
        # 获取最近一次使用的目录路径
        initial_dir = self._last_directory
        
        # 打开保存文件对话框，返回选择的文件名和筛选器
        file_name, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", os.path.join(initial_dir, default_name), filter_string,
            self._last_used_filter
        )
        
        # 如果用户选择了文件名
        if file_name:
            # 更新最近使用的筛选器
            self._last_used_filter = selected_filter
            # 更新文件路径
            self._path = file_name
            # 更新最近一次使用的目录路径为所选文件的父目录
            self._last_directory = os.path.dirname(file_name)
            
            # 提取所选文件类型及其扩展名
            selected_type = next((ft for ft in self._file_types if ft[0] in selected_filter), None)
            if selected_type:
                # 如果所选文件类型的扩展名是一个列表，则获取列表，否则生成包含单个扩展名的列表
                valid_extensions = selected_type[1] if isinstance(selected_type[1], list) else [selected_type[1]]
                
                # 确保文件名具有正确的扩展名
                file_base, file_extension = os.path.splitext(file_name)
                if file_extension[1:].lower() not in valid_extensions:
                    file_name = f"{file_base}.{valid_extensions[0]}"
            
            # 发送文件路径改变的信号
            self.sig_file_changed.emit(file_name)
        else:
            # 即使用户取消了保存，也要更新最近一次使用的目录路径
            dialog = QtWidgets.QFileDialog()
            self._last_directory = dialog.directory().absolutePath()
class MDragFileButton(MToolButton):
    """A Clickable and draggable tool button to upload files"""

    # Signal emitted when a single file is selected
    sig_file_changed = QtCore.Signal(str)
    # Signal emitted when multiple files are selected
    sig_files_changed = QtCore.Signal(list)
    
    # Connect the slot method _slot_browser_file to handle button clicks
    slot_browser_file = _slot_browser_file

    def __init__(self, text="", multiple=False, parent=None):
        # Initialize the button with parent widget
        super(MDragFileButton, self).__init__(parent=parent)
        
        # Enable drag-and-drop functionality
        self.setAcceptDrops(True)
        # Enable mouse tracking for the button
        self.setMouseTracking(True)
        # Set the text to display under the icon (if any)
        self.text_under_icon()
        # Set the text displayed on the button
        self.setText(text)
        
        # Set the size of the button based on the dayu_theme configuration
        size = dayu_theme.drag_size
        self.set_dayu_size(size)
        # Set the size of the icon displayed on the button
        self.setIconSize(QtCore.QSize(size, size))
        # Set the SVG icon for the button
        self.set_dayu_svg("cloud_line.svg")

        # Connect the clicked signal of the button to the slot_browser_file slot method
        self.clicked.connect(self.slot_browser_file)
        # Set the size policy of the button to expand horizontally and vertically
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Set the tooltip text for the button
        self.setToolTip(self.tr("Click to browser file"))

        # Initialize instance variables
        self._path = None
        self._multiple = multiple
        self._filters = []

    def get_dayu_filters(self):
        """
        Get browser's format filters
        :return: list
        """
        return self._filters

    def set_dayu_filters(self, value):
        """
        Set browser file format filters
        :param value:
        :return: None
        """
        self._filters = value

    def get_dayu_path(self):
        """
        Get last browser file path
        :return: str
        """
        return self._path

    def set_dayu_path(self, value):
        """
        Set browser file start path
        :param value: str
        :return: None
        """
        self._path = value

    def get_dayu_multiple(self):
        """
        Get browser can select multiple file or not
        :return: bool
        """
        return self._multiple

    def set_dayu_multiple(self, value):
        """
        Set browser can select multiple file or not
        :param value: bool
        :return: None
        """
        self._multiple = value

    # Define properties using QtCore.Property to access instance variables
    dayu_multiple = QtCore.Property(bool, get_dayu_multiple, set_dayu_multiple)
    dayu_path = QtCore.Property(six.string_types[0], get_dayu_path, set_dayu_path)
    dayu_filters = QtCore.Property(list, get_dayu_filters, set_dayu_filters)

    def dragEnterEvent(self, event):
        """Override dragEnterEvent. Validate dragged files"""
        # Check if the dragged content is a list of URIs
        if event.mimeData().hasFormat("text/uri-list"):
            # Get a list of valid file paths from the dropped URIs
            file_list = self._get_valid_file_list(event.mimeData().urls())
            count = len(file_list)
            # Accept the drop action if there's exactly one file or if multiple files are allowed
            if count == 1 or (count > 1 and self.get_dayu_multiple()):
                event.acceptProposedAction()
                return

    def dropEvent(self, event):
        """Override dropEvent to accept the dropped files"""
        # Get a list of valid file paths from the dropped URIs
        file_list = self._get_valid_file_list(event.mimeData().urls())
        # Emit the appropriate signal based on whether multiple files are allowed
        if self.get_dayu_multiple():
            self.sig_files_changed.emit(file_list)
            self.set_dayu_path(file_list)
        else:
            self.sig_file_changed.emit(file_list[0])
            self.set_dayu_path(file_list[0])
    # 定义一个方法，用于从给定的 URL 列表中获取有效的文件列表
    def _get_valid_file_list(self, url_list):
        # 导入内置模块 subprocess 和 sys
        import subprocess
        import sys

        # 初始化空的文件列表
        file_list = []

        # 遍历传入的 URL 列表
        for url in url_list:
            # 将 URL 转换为本地文件路径
            file_name = url.toLocalFile()

            # 如果操作系统是 macOS
            if sys.platform == "darwin":
                # 使用 subprocess 执行 AppleScript 命令，获取文件的 POSIX 路径
                sub_process = subprocess.Popen(
                    "osascript -e 'get posix path of posix file \"file://{}\" -- kthxbai'".format(file_name),
                    stdout=subprocess.PIPE,
                    shell=True,
                )
                # 从 subprocess 的输出中获取并去除首尾空白字符
                file_name = sub_process.communicate()[0].strip()
                sub_process.wait()

            # 检查文件名对应的路径是否存在且是一个文件
            if os.path.isfile(file_name):
                # 如果定义了过滤器函数 get_dayu_filters()，并且文件的扩展名在过滤器中
                if self.get_dayu_filters():
                    if os.path.splitext(file_name)[-1] in self.get_dayu_filters():
                        # 将符合条件的文件路径添加到文件列表中
                        file_list.append(file_name)
                else:
                    # 如果没有定义过滤器函数，则直接将文件路径添加到文件列表中
                    file_list.append(file_name)

        # 返回最终的有效文件列表
        return file_list
class MClickBrowserFolderPushButton(MPushButton):
    """A Clickable push button to browser folders"""

    # 定义信号，用于通知文件夹路径变更，传递字符串参数
    sig_folder_changed = QtCore.Signal(str)
    # 定义信号，用于通知多个文件夹路径变更，传递列表参数
    sig_folders_changed = QtCore.Signal(list)
    # 将槽函数设置为用于浏览文件夹的函数 _slot_browser_folder
    slot_browser_folder = _slot_browser_folder

    def __init__(self, text="", multiple=False, parent=None):
        # 调用父类构造函数初始化按钮文本和父组件
        super(MClickBrowserFolderPushButton, self).__init__(text=text, parent=parent)
        # 设置属性 "multiple"，表示是否支持多选
        self.setProperty("multiple", multiple)
        # 连接按钮的点击事件到浏览文件夹的槽函数
        self.clicked.connect(self.slot_browser_folder)
        # 设置工具提示文本为 "Click to browser folder"
        self.setToolTip(self.tr("Click to browser folder"))

        # 初始化路径变量为 None
        self._path = None
        # 初始化是否支持多选变量
        self._multiple = multiple

    def get_dayu_path(self):
        """
        Get last browser file path
        :return: str
        """
        # 返回最后浏览的文件夹路径
        return self._path

    def set_dayu_path(self, value):
        """
        Set browser file start path
        :param value: str
        :return: None
        """
        # 设置初始浏览文件夹路径
        self._path = value

    def get_dayu_multiple(self):
        """
        Get browser can select multiple file or not
        :return: bool
        """
        # 返回是否支持多选文件夹的布尔值
        return self._multiple

    def set_dayu_multiple(self, value):
        """
        Set browser can select multiple file or not
        :param value: bool
        :return: None
        """
        # 设置是否支持多选文件夹的布尔值
        self._multiple = value

    # 定义 dayu_multiple 属性，用于访问和设置是否支持多选的状态
    dayu_multiple = QtCore.Property(bool, get_dayu_multiple, set_dayu_multiple)
    # 定义 dayu_path 属性，用于访问和设置最后浏览文件夹的路径
    dayu_path = QtCore.Property(six.string_types[0], get_dayu_path, set_dayu_path)


@property_mixin
class MClickBrowserFolderToolButton(MToolButton):
    """A Clickable tool button to browser folders"""

    # 定义信号，用于通知文件夹路径变更，传递字符串参数
    sig_folder_changed = QtCore.Signal(str)
    # 定义信号，用于通知多个文件夹路径变更，传递列表参数
    sig_folders_changed = QtCore.Signal(list)
    # 将槽函数设置为用于浏览文件夹的函数 _slot_browser_folder

    slot_browser_folder = _slot_browser_folder

    def __init__(self, multiple=False, parent=None):
        # 调用父类构造函数初始化父组件
        super(MClickBrowserFolderToolButton, self).__init__(parent=parent)

        # 设置工具按钮的图标为 "folder_line.svg"
        self.set_dayu_svg("folder_line.svg")
        # 设置按钮仅显示图标
        self.icon_only()
        # 连接按钮的点击事件到浏览文件夹的槽函数
        self.clicked.connect(self.slot_browser_folder)
        # 设置工具提示文本为 "Click to browser folder"
        self.setToolTip(self.tr("Click to browser folder"))

        # 初始化路径变量为 None
        self._path = None
        # 初始化是否支持多选变量
        self._multiple = multiple

    def get_dayu_path(self):
        """
        Get last browser file path
        :return: str
        """
        # 返回最后浏览的文件夹路径
        return self._path

    def set_dayu_path(self, value):
        """
        Set browser file start path
        :param value: str
        :return: None
        """
        # 设置初始浏览文件夹路径
        self._path = value

    def get_dayu_multiple(self):
        """
        Get browser can select multiple file or not
        :return: bool
        """
        # 返回是否支持多选文件夹的布尔值
        return self._multiple

    def set_dayu_multiple(self, value):
        """
        Set browser can select multiple file or not
        :param value: bool
        :return: None
        """
        # 设置是否支持多选文件夹的布尔值
        self._multiple = value

    # 定义 dayu_multiple 属性，用于访问和设置是否支持多选的状态
    dayu_multiple = QtCore.Property(bool, get_dayu_multiple, set_dayu_multiple)
    # 定义 dayu_path 属性，用于访问和设置最后浏览文件夹的路径
    dayu_path = QtCore.Property(six.string_types[0], get_dayu_path, set_dayu_path)


@property_mixin
@cursor_mixin
class MDragFolderButton(MToolButton):
    """A tool button for dragging folders"""
    """A Clickable and draggable tool button to browser folders"""

    # 信号：当文件夹路径改变时发出的信号
    sig_folder_changed = QtCore.Signal(str)
    # 信号：当多个文件夹路径改变时发出的信号
    sig_folders_changed = QtCore.Signal(list)
    # 槽函数：处理浏览文件夹的操作
    slot_browser_folder = _slot_browser_folder

    def __init__(self, multiple=False, parent=None):
        # 调用父类构造函数，设置父级部件
        super(MDragFolderButton, self).__init__(parent=parent)
        # 设置接受拖拽操作
        self.setAcceptDrops(True)
        # 开启鼠标跟踪
        self.setMouseTracking(True)
        # 设置按钮文字显示在图标下方
        self.text_under_icon()
        # 设置按钮的图标使用特定的 SVG 文件
        self.set_dayu_svg("folder_line.svg")
        # 获取并设置按钮大小
        size = dayu_theme.drag_size
        self.set_dayu_size(size)
        # 设置图标大小
        self.setIconSize(QtCore.QSize(size, size))
        # 设置按钮文本
        self.setText(self.tr("Click or drag folder here"))
        # 点击按钮时连接到浏览文件夹的槽函数
        self.clicked.connect(self.slot_browser_folder)
        # 设置大小策略，使按钮在父布局中可以扩展
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # 设置工具提示文本
        self.setToolTip(self.tr("Click to browser folder or drag folder here"))

        # 初始化路径为 None
        self._path = None
        # 初始化是否允许选择多个文件夹的标志
        self._multiple = multiple

    def get_dayu_path(self):
        """
        Get last browser file path
        :return: str
        """
        # 返回最后浏览的文件夹路径
        return self._path

    def set_dayu_path(self, value):
        """
        Set browser file start path
        :param value: str
        :return: None
        """
        # 设置浏览文件夹的起始路径
        self._path = value

    def get_dayu_multiple(self):
        """
        Get browser can select multiple file or not
        :return: bool
        """
        # 返回是否允许选择多个文件夹的标志
        return self._multiple

    def set_dayu_multiple(self, value):
        """
        Set browser can select multiple file or not
        :param value: bool
        :return: None
        """
        # 设置是否允许选择多个文件夹的标志
        self._multiple = value

    # 定义属性：允许通过属性方式访问是否允许选择多个文件夹的标志
    dayu_multiple = QtCore.Property(bool, get_dayu_multiple, set_dayu_multiple)
    # 定义属性：允许通过属性方式访问最后浏览的文件夹路径
    dayu_path = QtCore.Property(bool, get_dayu_path, set_dayu_path)

    def dragEnterEvent(self, event):
        """Override dragEnterEvent. Validate dragged folders"""
        # 拖拽进入事件重写：验证拖拽的文件夹是否有效
        if event.mimeData().hasFormat("text/uri-list"):
            # 获取拖拽的所有 URL 中本地文件夹路径列表
            folder_list = [url.toLocalFile() for url in event.mimeData().urls() if os.path.isdir(url.toLocalFile())]
            # 统计有效文件夹的数量
            count = len(folder_list)
            # 如果只有一个文件夹或者允许选择多个文件夹，则接受该操作
            if count == 1 or (count > 1 and self.get_dayu_multiple()):
                event.acceptProposedAction()
                return

    def dropEvent(self, event):
        """Override dropEvent to accept the dropped folders"""
        # 拖拽释放事件重写：接受被拖拽的文件夹
        # 获取拖拽的所有 URL 中本地文件夹路径列表
        folder_list = [url.toLocalFile() for url in event.mimeData().urls() if os.path.isdir(url.toLocalFile())]
        # 如果允许选择多个文件夹，则发送包含所有文件夹路径的信号
        if self.get_dayu_multiple():
            self.sig_folders_changed.emit(folder_list)
        else:
            # 否则，发送包含第一个文件夹路径的信号
            self.sig_folder_changed.emit(folder_list[0])
        # 设置最后浏览的文件夹路径为第一个文件夹路径
        self.set_dayu_path(folder_list[0])
```