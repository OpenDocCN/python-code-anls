# `stable-diffusion-webui\modules\ui_loadsave.py`

```py
# 导入所需的模块
import json
import os

# 导入 gradio 库并重命名为 gr
import gradio as gr

# 导入自定义模块 errors 和 ui_components 中的 ToolButton 和 InputAccordion
from modules import errors
from modules.ui_components import ToolButton, InputAccordion

# 定义函数 radio_choices，用于处理 gradio 3.41 版本中选择项的变化
def radio_choices(comp):  # gradio 3.41 changes choices from list of values to list of pairs
    return [x[0] if isinstance(x, tuple) else x for x in getattr(comp, 'choices', [])]

# 定义类 UiLoadsave，用于保存和恢复 gradio 组件的默认值
class UiLoadsave:
    """allows saving and restoring default values for gradio components"""

    def __init__(self, filename):
        # 初始化类的属性
        self.filename = filename
        self.ui_settings = {}
        self.component_mapping = {}
        self.error_loading = False
        self.finalized_ui = False

        self.ui_defaults_view = None
        self.ui_defaults_apply = None
        self.ui_defaults_review = None

        try:
            # 如果文件存在，则读取文件中的设置
            if os.path.exists(self.filename):
                self.ui_settings = self.read_from_file()
        except Exception as e:
            # 如果读取文件出错，则显示错误信息
            self.error_loading = True
            errors.display(e, "loading settings")

    # 定义方法 add_block，用于将 gradio 块内的所有组件添加到跟踪组件的注册表中
    def add_block(self, x, path=""):
        """adds all components inside a gradio block x to the registry of tracked components"""

        if hasattr(x, 'children'):
            if isinstance(x, gr.Tabs) and x.elem_id is not None:
                # Tabs 元素不能有标签，必须使用 elem_id
                self.add_component(f"{path}/Tabs@{x.elem_id}", x)
            for c in x.children:
                self.add_block(c, path)
        elif x.label is not None:
            self.add_component(f"{path}/{x.label}", x)
        elif isinstance(x, gr.Button) and x.value is not None:
            self.add_component(f"{path}/{x.value}", x)

    # 定义方法 read_from_file，用于从文件中读取设置
    def read_from_file(self):
        with open(self.filename, "r", encoding="utf8") as file:
            return json.load(file)

    # 定义方法 write_to_file，用于将当前 UI 设置写入文件
    def write_to_file(self, current_ui_settings):
        with open(self.filename, "w", encoding="utf8") as file:
            json.dump(current_ui_settings, file, indent=4, ensure_ascii=False)
    # 将默认值保存到文件，除非文件存在且在启动时加载默认值时出现错误
    def dump_defaults(self):
        if self.error_loading and os.path.exists(self.filename):
            return

        # 将 UI 设置写入文件
        self.write_to_file(self.ui_settings)

    # 遍历默认值字典和当前 UI 元素值字典，返回不同的值的元组迭代器
    # 元组内容为：路径，旧值，新值
    def iter_changes(self, current_ui_settings, values):
        for (path, component), new_value in zip(self.component_mapping.items(), values):
            old_value = current_ui_settings.get(path)

            # 获取单选框的选项
            choices = radio_choices(component)
            if isinstance(new_value, int) and choices:
                if new_value >= len(choices):
                    continue

                new_value = choices[new_value]
                if isinstance(new_value, tuple):
                    new_value = new_value[0]

            # 如果新值等于旧值，则跳过
            if new_value == old_value:
                continue

            # 如果旧值为 None 且新值为空字符串或空列表，则跳过
            if old_value is None and new_value == '' or new_value == []:
                continue

            yield path, old_value, new_value

    # 生成 UI 视图
    def ui_view(self, *values):
        text = ["<table><thead><tr><th>Path</th><th>Old value</th><th>New value</th></thead><tbody>"]

        # 遍历读取文件的默认值和传入的值，获取不同的值并添加到文本中
        for path, old_value, new_value in self.iter_changes(self.read_from_file(), values):
            if old_value is None:
                old_value = "<span class='ui-defaults-none'>None</span>"

            text.append(f"<tr><td>{path}</td><td>{old_value}</td><td>{new_value}</td></tr>")

        # 如果没有变化，则添加提示信息
        if len(text) == 1:
            text.append("<tr><td colspan=3>No changes</td></tr>")

        text.append("</tbody>")
        return "".join(text)
    # 应用用户界面更改，接受一系列值作为参数
    def ui_apply(self, *values):
        # 记录更改的次数
        num_changed = 0

        # 从文件中读取当前的 UI 设置
        current_ui_settings = self.read_from_file()

        # 遍历当前 UI 设置的更改，将新值应用到相应路径
        for path, _, new_value in self.iter_changes(current_ui_settings.copy(), values):
            num_changed += 1
            current_ui_settings[path] = new_value

        # 如果没有任何更改，则返回 "No changes."
        if num_changed == 0:
            return "No changes."

        # 将更改后的 UI 设置写入文件
        self.write_to_file(current_ui_settings)

        # 返回写入更改的次数
        return f"Wrote {num_changed} changes."

    # 创建用于编辑默认 UI 的 UI 元素，不添加任何逻辑
    def create_ui(self):
        """creates ui elements for editing defaults UI, without adding any logic to them"""

        # 显示页面内容，提供更改默认值的说明
        gr.HTML(
            f"This page allows you to change default values in UI elements on other tabs.<br />"
            f"Make your changes, press 'View changes' to review the changed default values,<br />"
            f"then press 'Apply' to write them to {self.filename}.<br />"
            f"New defaults will apply after you restart the UI.<br />"
        )

        # 创建按钮行，包括 'View changes' 和 'Apply' 两个按钮
        with gr.Row():
            self.ui_defaults_view = gr.Button(value='View changes', elem_id="ui_defaults_view", variant="secondary")
            self.ui_defaults_apply = gr.Button(value='Apply', elem_id="ui_defaults_apply", variant="primary")

        # 创建用于显示更改的 HTML 元素
        self.ui_defaults_review = gr.HTML("")

    # 设置 UI 元素的逻辑，所有 add_block 类必须在此之前创建
    def setup_ui(self):
        """adds logic to elements created with create_ui; all add_block class must be made before this"""

        # 确保 UI 设置未被最终确定
        assert not self.finalized_ui
        self.finalized_ui = True

        # 给 'View changes' 按钮添加点击事件，调用 ui_view 函数，设置输入和输出
        self.ui_defaults_view.click(fn=self.ui_view, inputs=list(self.component_mapping.values()), outputs=[self.ui_defaults_review])
        # 给 'Apply' 按钮添加点击事件，调用 ui_apply 函数，设置输入和输出
        self.ui_defaults_apply.click(fn=self.ui_apply, inputs=list(self.component_mapping.values()), outputs=[self.ui_defaults_review])
```