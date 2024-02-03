# `stable-diffusion-webui\modules\ui_components.py`

```py
# 导入 gradio 库
import gradio as gr

# 定义一个表单组件类，用于指定期望的父组件为 gr.components.Form
class FormComponent:
    def get_expected_parent(self):
        return gr.components.Form

# 将 Dropdown 类的期望父组件设置为 FormComponent 类
gr.Dropdown.get_expected_parent = FormComponent.get_expected_parent

# 定义一个工具按钮类，继承自 FormComponent 和 gr.Button，用于在 gradio 表单中显示一个带有单个表情符号的小按钮
class ToolButton(FormComponent, gr.Button):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, *args, **kwargs):
        classes = kwargs.pop("elem_classes", [])
        super().__init__(*args, elem_classes=["tool", *classes], **kwargs)

    def get_block_name(self):
        return "button"

# 定义一个调整大小的行类，继承自 gr.Row，用于在 gradio 表单中显示一个可以调整大小的行
class ResizeHandleRow(gr.Row):
    """Same as gr.Row but fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.elem_classes.append("resize-handle-row")

    def get_block_name(self):
        return "row"

# 定义一个表单行类，继承自 FormComponent 和 gr.Row，用于在 gradio 表单中显示一个行
class FormRow(FormComponent, gr.Row):
    """Same as gr.Row but fits inside gradio forms"""

    def get_block_name(self):
        return "row"

# 定义一个表单列类，继承自 FormComponent 和 gr.Column，用于在 gradio 表单中显示一个列
class FormColumn(FormComponent, gr.Column):
    """Same as gr.Column but fits inside gradio forms"""

    def get_block_name(self):
        return "column"

# 定义一个表单组类，继承自 FormComponent 和 gr.Group，用于在 gradio 表单中显示一个组
class FormGroup(FormComponent, gr.Group):
    """Same as gr.Group but fits inside gradio forms"""

    def get_block_name(self):
        return "group"

# 定义一个表单 HTML 类，继承自 FormComponent 和 gr.HTML，用于在 gradio 表单中显示 HTML 内容
class FormHTML(FormComponent, gr.HTML):
    """Same as gr.HTML but fits inside gradio forms"""

    def get_block_name(self):
        return "html"

# 定义一个表单颜色选择器类，继承自 FormComponent 和 gr.ColorPicker，用于在 gradio 表单中显示一个颜色选择器
class FormColorPicker(FormComponent, gr.ColorPicker):
    """Same as gr.ColorPicker but fits inside gradio forms"""

    def get_block_name(self):
        return "colorpicker"

# 定义一个多选下拉框类，继承自 FormComponent 和 gr.Dropdown，用于在 gradio 表单中显示一个多选下拉框
class DropdownMulti(FormComponent, gr.Dropdown):
    """Same as gr.Dropdown but always multiselect"""
    def __init__(self, **kwargs):
        super().__init__(multiselect=True, **kwargs)

    def get_block_name(self):
        return "dropdown"

# 定义一个可编辑下拉框类，继承自 FormComponent 和 gr.Dropdown，用于在 gradio 表单中显示一个可编辑的下拉框
class DropdownEditable(FormComponent, gr.Dropdown):
    """Same as gr.Dropdown but allows editing value"""
    # 初始化函数，继承父类的初始化方法，允许自定义值
    def __init__(self, **kwargs):
        super().__init__(allow_custom_value=True, **kwargs)

    # 返回块的名称为"dropdown"
    def get_block_name(self):
        return "dropdown"
class InputAccordion(gr.Checkbox):
    """A gr.Accordion that can be used as an input - returns True if open, False if closed.

    Actaully just a hidden checkbox, but creates an accordion that follows and is followed by the state of the checkbox.
    """

    global_index = 0

    def __init__(self, value, **kwargs):
        # 初始化 InputAccordion 类，设置初始值和其他参数
        self.accordion_id = kwargs.get('elem_id')
        # 获取参数中的 elem_id，如果不存在则生成一个唯一的 ID
        if self.accordion_id is None:
            self.accordion_id = f"input-accordion-{InputAccordion.global_index}"
            InputAccordion.global_index += 1

        kwargs_checkbox = {
            **kwargs,
            "elem_id": f"{self.accordion_id}-checkbox",
            "visible": False,
        }
        # 设置隐藏的复选框参数
        super().__init__(value, **kwargs_checkbox)

        self.change(fn=None, _js='function(checked){ inputAccordionChecked("' + self.accordion_id + '", checked); }', inputs=[self])
        # 设置复选框状态改变时的回调函数

        kwargs_accordion = {
            **kwargs,
            "elem_id": self.accordion_id,
            "label": kwargs.get('label', 'Accordion'),
            "elem_classes": ['input-accordion'],
            "open": value,
        }
        # 设置 Accordion 参数
        self.accordion = gr.Accordion(**kwargs_accordion)

    def extra(self):
        """Allows you to put something into the label of the accordion.

        Use it like this:

        ```
        with InputAccordion(False, label="Accordion") as acc:
            with acc.extra():
                FormHTML(value="hello", min_width=0)

            ...
        ```py
        """
        # 返回一个用于在 Accordion 标签中放置内容的 Column 对象
        return gr.Column(elem_id=self.accordion_id + '-extra', elem_classes='input-accordion-extra', min_width=0)

    def __enter__(self):
        # 进入 Accordion 上下文
        self.accordion.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出 Accordion 上下文
        self.accordion.__exit__(exc_type, exc_val, exc_tb)

    def get_block_name(self):
        # 返回块的名称
        return "checkbox"
```