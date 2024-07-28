# `.\comic-translate\app\ui\dayu_widgets\form.py`

```py
# 定义一个名为 MForm 的自定义窗口部件类，继承自 QtWidgets.QWidget
class MForm(QtWidgets.QWidget):
    # 定义类常量，表示布局方式为水平、垂直和内联
    Horizontal = "horizontal"
    Vertical = "vertical"
    Inline = "inline"

    # 初始化方法，接受布局参数和父级窗口部件
    def __init__(self, layout=None, parent=None):
        # 调用父类 QtWidgets.QWidget 的初始化方法
        super(MForm, self).__init__(parent)
        
        # 如果未提供布局参数，则默认为水平布局
        layout = layout or MForm.Horizontal
        
        # 根据布局参数选择主要布局方式
        if layout == MForm.Inline:
            self._main_layout = QtWidgets.QHBoxLayout()
        elif layout == MForm.Vertical:
            self._main_layout = QtWidgets.QVBoxLayout()
        else:
            self._main_layout = QtWidgets.QFormLayout()
        
        # 初始化模型变量为 None
        self._model = None
        
        # 初始化标签列表为空列表
        self._label_list = []

    # 设置模型的方法
    def set_model(self, m):
        self._model = m

    # 设置标签对齐方式的方法
    def set_label_align(self, align):
        # 遍历标签列表，设置每个标签的对齐方式
        for label in self._label_list:
            label.setAlignment(align)
        
        # 设置主要布局的标签对齐方式
        self._main_layout.setLabelAlignment(align)

    # 类方法，返回水平布局的 MForm 对象
    @classmethod
    def horizontal(cls):
        return cls(layout=cls.Horizontal)

    # 类方法，返回垂直布局的 MForm 对象
    @classmethod
    def vertical(cls):
        return cls(layout=cls.Vertical)

    # 类方法，返回内联布局的 MForm 对象
    @classmethod
    def inline(cls):
        return cls(layout=cls.Inline)
```