# `.\comic-translate\app\ui\dayu_widgets\progress_bar.py`

```py
# 导入必要的模块
from PySide6 import QtCore  # 导入 PySide6 的 QtCore 模块
from PySide6 import QtWidgets  # 导入 PySide6 的 QtWidgets 模块


class MProgressBar(QtWidgets.QProgressBar):
    """
    props:
        status: str  # 属性，表示进度条的状态，可以是 'error', 'primary', 'success'
    """

    ErrorStatus = "error"  # 错误状态常量
    NormalStatus = "primary"  # 普通状态常量
    SuccessStatus = "success"  # 成功状态常量

    def __init__(self, parent=None):
        super(MProgressBar, self).__init__(parent=parent)
        self.setAlignment(QtCore.Qt.AlignCenter)  # 设置进度条文本居中显示
        self._status = MProgressBar.NormalStatus  # 初始化进度条状态为普通状态

    def auto_color(self):
        self.valueChanged.connect(self._update_color)  # 连接 valueChanged 信号到 _update_color 槽函数
        return self

    @QtCore.Slot(int)
    def _update_color(self, value):
        if value >= self.maximum():  # 如果当前值大于等于最大值
            self.set_dayu_status(MProgressBar.SuccessStatus)  # 设置进度条状态为成功状态
        else:
            self.set_dayu_status(MProgressBar.NormalStatus)  # 否则设置进度条状态为普通状态

    def get_dayu_status(self):
        return self._status  # 获取当前进度条状态

    def set_dayu_status(self, value):
        self._status = value  # 设置进度条状态
        self.style().polish(self)  # 更新进度条的样式

    dayu_status = QtCore.Property(str, get_dayu_status, set_dayu_status)  # 定义 dayu_status 属性

    def normal(self):
        self.set_dayu_status(MProgressBar.NormalStatus)  # 设置进度条状态为普通状态
        return self

    def error(self):
        self.set_dayu_status(MProgressBar.ErrorStatus)  # 设置进度条状态为错误状态
        return self

    def success(self):
        self.set_dayu_status(MProgressBar.SuccessStatus)  # 设置进度条状态为成功状态
        return self

    # def paintEvent(self, event):
    #     pass
```