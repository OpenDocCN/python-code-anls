# `D:\src\scipysrc\matplotlib\tools\triage_tests.py`

```py
"""
This is a developer utility to help analyze and triage image
comparison failures.

It allows the failures to be quickly compared against the expected
results, and the new results to be either accepted (by copying the new
results to the source tree) or rejected (by copying the original
expected result to the source tree).

To start:

    If you ran the tests from the top-level of a source checkout, simply run:

        python tools/triage_tests.py

    Otherwise, you can manually select the location of `result_images`
    on the commandline.

Keys:

    left/right: Move between test, expected and diff images
    up/down:    Move between tests
    A:          Accept test.  Copy the test result to the source tree.
    R:          Reject test.  Copy the expected result to the source tree.
"""

import os
from pathlib import Path
import shutil
import sys

from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.backends.qt_compat import _exec


# matplotlib stores the baseline images under two separate subtrees,
# but these are all flattened in the result_images directory.  In
# order to find the source, we need to search for a match in one of
# these two places.

BASELINE_IMAGES = [
    Path('lib/matplotlib/tests/baseline_images'),
    *Path('lib/mpl_toolkits').glob('*/tests/baseline_images'),
]

# Non-png image extensions
exts = ['pdf', 'svg', 'eps']


class Thumbnail(QtWidgets.QFrame):
    """
    Represents one of the three thumbnails at the top of the window.
    """
    def __init__(self, parent, index, name):
        super().__init__()

        self.parent = parent
        self.index = index

        layout = QtWidgets.QVBoxLayout()

        label = QtWidgets.QLabel(name)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter |
                           QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label, 0)

        self.image = QtWidgets.QLabel()
        self.image.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter |
                                QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.image.setMinimumSize(800 // 3, 600 // 3)
        layout.addWidget(self.image)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        self.parent.set_large_image(self.index)


class EventFilter(QtCore.QObject):
    # A hack keypresses can be handled globally and aren't swallowed
    # by the individual widgets

    def __init__(self, window):
        super().__init__()
        self.window = window

    def eventFilter(self, receiver, event):
        if event.type() == QtCore.QEvent.Type.KeyPress:
            self.window.keyPressEvent(event)
            return True
        else:
            return super().eventFilter(receiver, event)


class Dialog(QtWidgets.QDialog):
    """
    The main dialog window.
    """
    # 初始化函数，用于初始化一个包含条目的窗口
    def __init__(self, entries):
        # 调用父类的初始化方法
        super().__init__()

        # 将传入的条目列表保存到对象属性中
        self.entries = entries
        # 初始化当前条目和当前缩略图的索引为-1
        self.current_entry = -1
        self.current_thumbnail = -1

        # 创建一个事件过滤器并安装到当前窗口上
        event_filter = EventFilter(self)
        self.installEventFilter(event_filter)

        # 创建一个 QListWidget 实例，用于显示文件列表，设置最小宽度为 400 像素
        self.filelist = QtWidgets.QListWidget()
        self.filelist.setMinimumWidth(400)
        # 遍历条目列表，将每个条目的显示名称添加到文件列表中
        for entry in entries:
            self.filelist.addItem(entry.display)
        # 连接文件列表的当前行改变信号到对应的槽函数 set_entry
        self.filelist.currentRowChanged.connect(self.set_entry)

        # 创建一个包含缩略图的容器 QWidget
        thumbnails_box = QtWidgets.QWidget()
        thumbnails_layout = QtWidgets.QVBoxLayout()
        self.thumbnails = []
        # 遍历指定的缩略图名称列表，创建对应的 Thumbnail 实例并添加到布局中和缩略图列表中
        for i, name in enumerate(('test', 'expected', 'diff')):
            thumbnail = Thumbnail(self, i, name)
            thumbnails_layout.addWidget(thumbnail)
            self.thumbnails.append(thumbnail)
        thumbnails_box.setLayout(thumbnails_layout)

        # 创建一个垂直布局管理器用于显示图片
        images_layout = QtWidgets.QVBoxLayout()
        images_box = QtWidgets.QWidget()
        # 创建一个 QLabel 实例用于显示图片，设置对齐方式和最小大小
        self.image_display = QtWidgets.QLabel()
        self.image_display.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignHCenter |
            QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.image_display.setMinimumSize(800, 600)
        # 将图片显示部件添加到图片布局中，并指定布局权重为 6
        images_layout.addWidget(self.image_display, 6)
        images_box.setLayout(images_layout)

        # 创建一个按钮容器部件
        buttons_box = QtWidgets.QWidget()
        buttons_layout = QtWidgets.QHBoxLayout()
        # 创建“接受”按钮，并连接到对应的槽函数 accept_test
        accept_button = QtWidgets.QPushButton("Accept (A)")
        accept_button.clicked.connect(self.accept_test)
        buttons_layout.addWidget(accept_button)
        # 创建“拒绝”按钮，并连接到对应的槽函数 reject_test
        reject_button = QtWidgets.QPushButton("Reject (R)")
        reject_button.clicked.connect(self.reject_test)
        buttons_layout.addWidget(reject_button)
        buttons_box.setLayout(buttons_layout)
        # 将按钮容器添加到图片布局中
        images_layout.addWidget(buttons_box)

        # 创建一个水平布局管理器用于整体布局
        main_layout = QtWidgets.QHBoxLayout()
        # 将文件列表、缩略图容器和图片容器依次添加到主布局中，指定文件列表和缩略图容器的布局权重为 1，图片容器为 3
        main_layout.addWidget(self.filelist, 1)
        main_layout.addWidget(thumbnails_box, 1)
        main_layout.addWidget(images_box, 3)

        # 将主布局应用到当前窗口
        self.setLayout(main_layout)

        # 设置窗口标题
        self.setWindowTitle("matplotlib test triager")

        # 初始化显示第一个条目的内容
        self.set_entry(0)

    # 设置当前显示的条目
    def set_entry(self, index):
        # 如果索引未改变，则直接返回
        if self.current_entry == index:
            return

        # 更新当前条目索引为指定索引
        self.current_entry = index
        # 获取当前条目对象
        entry = self.entries[index]

        # 清空之前的 pixmap 列表
        self.pixmaps = []
        # 遍历当前条目的缩略图文件名列表和对应的缩略图部件列表
        for fname, thumbnail in zip(entry.thumbnails, self.thumbnails):
            # 从文件名创建 QPixmap 对象
            pixmap = QtGui.QPixmap(os.fspath(fname))
            # 缩放 QPixmap 对象以适应缩略图部件的大小，并设置缩放模式为保持纵横比和平滑转换
            scaled_pixmap = pixmap.scaled(
                thumbnail.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation)
            # 将缩放后的 QPixmap 设置为缩略图部件的显示内容，并将其添加到 pixmap 列表中
            thumbnail.image.setPixmap(scaled_pixmap)
            self.pixmaps.append(scaled_pixmap)

        # 设置当前显示的大图像为第一张缩略图
        self.set_large_image(0)
        # 设置文件列表当前行为当前条目索引
        self.filelist.setCurrentRow(self.current_entry)
    # 设置大图像显示的方法，根据给定的索引切换当前缩略图的边框样式为无边框
    self.thumbnails[self.current_thumbnail].setFrameShape(
        QtWidgets.QFrame.Shape.NoFrame)
    # 更新当前缩略图索引为给定索引
    self.current_thumbnail = index
    # 根据当前条目和当前缩略图索引加载对应的图像文件路径，并创建 QPixmap 对象
    pixmap = QtGui.QPixmap(os.fspath(
        self.entries[self.current_entry]
        .thumbnails[self.current_thumbnail]))
    # 将创建的 QPixmap 对象设置为图片显示框的图像
    self.image_display.setPixmap(pixmap)
    # 将新的当前缩略图的边框样式设置为有边框
    self.thumbnails[self.current_thumbnail].setFrameShape(
        QtWidgets.QFrame.Shape.Box)

# 处理接受测试的方法，更新条目状态和文件列表中的显示，自动移动到下一个条目
def accept_test(self):
    # 获取当前条目对象
    entry = self.entries[self.current_entry]
    # 如果当前条目状态为 'autogen'，打印提示信息并返回
    if entry.status == 'autogen':
        print('Cannot accept autogenerated test cases.')
        return
    # 否则，接受当前条目
    entry.accept()
    # 更新文件列表中当前条目显示的文本
    self.filelist.currentItem().setText(
        self.entries[self.current_entry].display)
    # 自动移动到下一个条目
    self.set_entry(min((self.current_entry + 1), len(self.entries) - 1))

# 处理拒绝测试的方法，更新条目状态和文件列表中的显示，自动移动到下一个条目
def reject_test(self):
    # 获取当前条目对象
    entry = self.entries[self.current_entry]
    # 如果当前条目状态为 'autogen'，打印提示信息并返回
    if entry.status == 'autogen':
        print('Cannot reject autogenerated test cases.')
        return
    # 否则，拒绝当前条目
    entry.reject()
    # 更新文件列表中当前条目显示的文本
    self.filelist.currentItem().setText(
        self.entries[self.current_entry].display)
    # 自动移动到下一个条目
    self.set_entry(min((self.current_entry + 1), len(self.entries) - 1))

# 处理键盘按下事件的方法，根据按键执行相应的操作
def keyPressEvent(self, e):
    # 如果按下左箭头键，调用 set_large_image 方法，显示前一个缩略图
    if e.key() == QtCore.Qt.Key.Key_Left:
        self.set_large_image((self.current_thumbnail - 1) % 3)
    # 如果按下右箭头键，调用 set_large_image 方法，显示后一个缩略图
    elif e.key() == QtCore.Qt.Key.Key_Right:
        self.set_large_image((self.current_thumbnail + 1) % 3)
    # 如果按下上箭头键，调用 set_entry 方法，显示上一个条目
    elif e.key() == QtCore.Qt.Key.Key_Up:
        self.set_entry(max(self.current_entry - 1, 0))
    # 如果按下下箭头键，调用 set_entry 方法，显示下一个条目
    elif e.key() == QtCore.Qt.Key.Key_Down:
        self.set_entry(min(self.current_entry + 1, len(self.entries) - 1))
    # 如果按下 A 键，调用 accept_test 方法，接受当前条目的测试
    elif e.key() == QtCore.Qt.Key.Key_A:
        self.accept_test()
    # 如果按下 R 键，调用 reject_test 方法，拒绝当前条目的测试
    elif e.key() == QtCore.Qt.Key.Key_R:
        self.reject_test()
    # 对于其他按键，调用父类的 keyPressEvent 方法处理
    else:
        super().keyPressEvent(e)
class Entry:
    """
    A model for a single image comparison test.
    """

    def __init__(self, path, root, source):
        self.source = source  # 设置实例变量source为传入的source参数
        self.root = root  # 设置实例变量root为传入的root参数
        self.dir = path.parent  # 设置实例变量dir为传入路径的父目录
        self.diff = path.name  # 设置实例变量diff为传入路径的文件名部分
        self.reldir = self.dir.relative_to(self.root)  # 计算相对路径，相对于root

        basename = self.diff[:-len('-failed-diff.png')]
        for ext in exts:  # 遍历扩展名列表exts
            if basename.endswith(f'_{ext}'):  # 如果basename以某个扩展名结尾
                display_extension = f'_{ext}'  # 设置显示的扩展名
                extension = ext  # 设置扩展名
                basename = basename[:-len(display_extension)]  # 去除basename中的显示扩展名部分
                break
        else:  # 如果未找到匹配的扩展名
            display_extension = ''  # 设置显示的扩展名为空字符串
            extension = 'png'  # 默认扩展名为'png'

        self.basename = basename  # 设置实例变量basename为处理后的basename
        self.extension = extension  # 设置实例变量extension为扩展名
        self.generated = f'{basename}.{extension}'  # 设置生成的文件名
        self.expected = f'{basename}-expected.{extension}'  # 设置预期的文件名
        self.expected_display = f'{basename}-expected{display_extension}.png'  # 设置预期的显示文件名
        self.generated_display = f'{basename}{display_extension}.png'  # 设置生成的显示文件名
        self.name = self.reldir / self.basename  # 设置实例变量name为相对路径与basename的组合
        self.destdir = self.get_dest_dir(self.reldir)  # 计算目标目录路径

        self.thumbnails = [
            self.generated_display,  # 添加生成的显示文件名到缩略图列表
            self.expected_display,  # 添加预期的显示文件名到缩略图列表
            self.diff  # 添加diff文件名到缩略图列表
        ]
        self.thumbnails = [self.dir / x for x in self.thumbnails]  # 组合缩略图路径

        if not Path(self.destdir, self.generated).exists():  # 如果目标目录中生成的文件不存在
            self.status = 'autogen'  # 设置状态为'autogen'
        elif ((self.dir / self.generated).read_bytes()  # 否则，如果当前目录中生成的文件内容与目标目录中的相同
              == (self.destdir / self.generated).read_bytes()):
            self.status = 'accept'  # 设置状态为'accept'
        else:
            self.status = 'unknown'  # 否则，设置状态为'unknown'

    def get_dest_dir(self, reldir):
        """
        Find the source tree directory corresponding to the given
        result_images subdirectory.
        """
        for baseline_dir in BASELINE_IMAGES:  # 遍历基准图像目录列表
            path = self.source / baseline_dir / reldir  # 构造潜在的目标路径
            if path.is_dir():  # 如果该路径存在且是一个目录
                return path  # 返回找到的目标目录路径
        raise ValueError(f"Can't find baseline dir for {reldir}")  # 如果未找到匹配目录，则引发异常

    @property
    def display(self):
        """
        Get the display string for this entry.  This is the text that
        appears in the list widget.
        """
        status_map = {  # 状态映射表，将状态映射到对应的Unicode图标
            'unknown': '\N{BALLOT BOX}',  # 未知状态
            'accept': '\N{BALLOT BOX WITH CHECK}',  # 接受状态
            'reject': '\N{BALLOT BOX WITH X}',  # 拒绝状态（未在代码中使用）
            'autogen': '\N{WHITE SQUARE CONTAINING BLACK SMALL SQUARE}',  # 自动生成状态
        }
        box = status_map[self.status]  # 获取状态对应的Unicode图标
        return f'{box} {self.name} [{self.extension}]'  # 返回带有状态图标、名称和扩展名的显示字符串

    def accept(self):
        """
        Accept this test by copying the generated result to the source tree.
        """
        copy_file(self.dir / self.generated, self.destdir / self.generated)  # 复制生成的结果到目标目录
        self.status = 'accept'  # 设置状态为'accept'
    def reject(self):
        """
        Reject this test by copying the expected result to the source tree.
        """
        # 获取预期结果文件的路径
        expected = self.dir / self.expected
        # 如果预期结果文件不是符号链接
        if not expected.is_symlink():
            # 将预期结果文件复制到目标目录下的生成文件
            copy_file(expected, self.destdir / self.generated)
        # 设置测试状态为拒绝
        self.status = 'reject'
# 复制文件从路径 *a* 到路径 *b*
def copy_file(a, b):
    # 打印正在复制的文件路径
    print(f'copying: {a} to {b}')
    # 使用 shutil 库的 copyfile 函数执行文件复制操作
    shutil.copyfile(a, b)


# 查找所有失败的测试条目，通过查找文件名以 `-failed-diff` 结尾的文件来实现
def find_failing_tests(result_images, source):
    # 使用列表推导式创建一个 Entry 对象的列表，每个对象代表一个失败的测试条目
    return [Entry(path, result_images, source)
            for path in sorted(Path(result_images).glob("**/*-failed-diff.*"))]


# 启动图形用户界面 (GUI)
def launch(result_images, source):
    # 查找所有失败的测试条目
    entries = find_failing_tests(result_images, source)

    # 如果没有找到失败的测试条目，则打印信息并退出程序
    if len(entries) == 0:
        print("No failed tests")
        sys.exit(0)

    # 初始化 Qt 应用程序
    app = QtWidgets.QApplication(sys.argv)
    # 创建对话框并显示
    dialog = Dialog(entries)
    dialog.show()
    # 创建事件过滤器并安装到应用程序中
    filter = EventFilter(dialog)
    app.installEventFilter(filter)
    # 运行 Qt 应用程序的事件循环，并在退出时返回状态码
    sys.exit(_exec(app))


if __name__ == '__main__':
    import argparse

    # 设置源目录为当前文件的父级目录的父级目录
    source_dir = Path(__file__).parent.parent

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Triage image comparison test failures.

If no arguments are provided, it assumes you ran the tests at the
top-level of a source checkout as `pytest .`.

Keys:
    left/right: Move between test, expected and diff images
    up/down:    Move between tests
    A:          Accept test.  Copy the test result to the source tree.
    R:          Reject test.  Copy the expected result to the source tree.
""")
    # 添加位置参数 result_images 和 source，默认值分别为 result_images 目录和源码树的根目录
    parser.add_argument("result_images", type=Path, nargs='?',
                        default=source_dir / 'result_images',
                        help="The location of the result_images directory")
    parser.add_argument("source", type=Path, nargs='?', default=source_dir,
                        help="The location of the matplotlib source tree")
    # 解析命令行参数
    args = parser.parse_args()

    # 启动主函数，传入 result_images 和 source 参数
    launch(args.result_images, args.source)
```