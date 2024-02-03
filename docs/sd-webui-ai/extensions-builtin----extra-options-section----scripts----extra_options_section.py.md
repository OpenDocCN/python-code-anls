# `stable-diffusion-webui\extensions-builtin\extra-options-section\scripts\extra_options_section.py`

```py
# 导入 math 模块
import math

# 导入 gradio 模块，并将其重命名为 gr
import gradio as gr
# 从 modules 模块中导入 scripts、shared、ui_components、ui_settings、generation_parameters_copypaste
from modules import scripts, shared, ui_components, ui_settings, generation_parameters_copypaste
# 从 modules.ui_components 模块中导入 FormColumn
from modules.ui_components import FormColumn

# 定义 ExtraOptionsSection 类，继承自 scripts.Script 类
class ExtraOptionsSection(scripts.Script):
    # 定义类属性 section，值为 "extra_options"
    section = "extra_options"

    # 初始化方法
    def __init__(self):
        # 初始化实例属性 comps，值为 None
        self.comps = None
        # 初始化实例属性 setting_names，值为 None
        self.setting_names = None

    # 定义 title 方法，返回 "Extra options"
    def title(self):
        return "Extra options"

    # 定义 show 方法，接受参数 is_img2img
    def show(self, is_img2img):
        # 返回 scripts.AlwaysVisible
        return scripts.AlwaysVisible
    # 定义一个 UI 方法，根据参数 is_img2img 决定显示图片到图片转换的界面还是文本到图片转换的界面
    def ui(self, is_img2img):
        # 初始化组件列表、设置名称列表和信息文本字段列表
        self.comps = []
        self.setting_names = []
        self.infotext_fields = []
        # 根据 is_img2img 决定使用图片到图片转换的额外选项还是文本到图片转换的额外选项
        extra_options = shared.opts.extra_options_img2img if is_img2img else shared.opts.extra_options_txt2img
        # 根据 is_img2img 构建元素 ID 的标签名
        elem_id_tabname = "extra_options_" + ("img2img" if is_img2img else "txt2img")

        # 将信息文本到设置名称的映射关系转换为字典
        mapping = {k: v for v, k in generation_parameters_copypaste.infotext_to_setting_name_mapping}

        # 创建界面块
        with gr.Blocks() as interface:
            # 如果额外选项存在且设置为手风琴模式，则创建手风琴组件，否则创建组件组
            with gr.Accordion("Options", open=False, elem_id=elem_id_tabname) if shared.opts.extra_options_accordion and extra_options else gr.Group(elem_id=elem_id_tabname):

                # 计算行数
                row_count = math.ceil(len(extra_options) / shared.opts.extra_options_cols)

                # 遍历每一行
                for row in range(row_count):
                    with gr.Row():
                        # 遍历每一列
                        for col in range(shared.opts.extra_options_cols):
                            index = row * shared.opts.extra_options_cols + col
                            # 如果超出额外选项的数量，则跳出循环
                            if index >= len(extra_options):
                                break

                            setting_name = extra_options[index]

                            # 创建设置组件
                            with FormColumn():
                                comp = ui_settings.create_setting_component(setting_name)

                            # 将设置组件添加到组件列表和设置名称列表中
                            self.comps.append(comp)
                            self.setting_names.append(setting_name)

                            # 获取设置名称对应的信息文本名称
                            setting_infotext_name = mapping.get(setting_name)
                            # 如果存在信息文本名称，则将设置组件和信息文本名称添加到信息文本字段列表中
                            if setting_infotext_name is not None:
                                self.infotext_fields.append((comp, setting_infotext_name))

        # 定义获取设置数值的方法
        def get_settings_values():
            res = [ui_settings.get_value_for_setting(key) for key in self.setting_names]
            return res[0] if len(res) == 1 else res

        # 加载界面，设置输入为空，输出为组件列表，不显示进度条
        interface.load(fn=get_settings_values, inputs=[], outputs=self.comps, queue=False, show_progress=False)

        # 返回组件列表
        return self.comps
    # 在处理之前，为处理器设置参数
    def before_process(self, p, *args):
        # 遍历设置名称和对应的值
        for name, value in zip(self.setting_names, args):
            # 如果设置名称不在处理器的覆盖设置中
            if name not in p.override_settings:
                # 将设置名称和值添加到处理器的覆盖设置中
                p.override_settings[name] = value
# 更新共享选项模板，将设置在 UI 中的设置添加到 txt2img 和 img2img 标签的主界面中
shared.options_templates.update(shared.options_section(('settings_in_ui', "Settings in UI", "ui"), {
    # 设置在 UI 中的说明文本
    "settings_in_ui": shared.OptionHTML("""
This page allows you to add some settings to the main interface of txt2img and img2img tabs.
"""),
    # txt2img 的额外选项设置
    "extra_options_txt2img": shared.OptionInfo([], "Settings for txt2img", ui_components.DropdownMulti, lambda: {"choices": list(shared.opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("setting entries that also appear in txt2img interfaces").needs_reload_ui(),
    # img2img 的额外选项设置
    "extra_options_img2img": shared.OptionInfo([], "Settings for img2img", ui_components.DropdownMulti, lambda: {"choices": list(shared.opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("setting entries that also appear in img2img interfaces").needs_reload_ui(),
    # 添加设置的列数
    "extra_options_cols": shared.OptionInfo(1, "Number of columns for added settings", gr.Slider, {"step": 1, "minimum": 1, "maximum": 20}).info("displayed amount will depend on the actual browser window width").needs_reload_ui(),
    # 将添加的设置放入手风琴中
    "extra_options_accordion": shared.OptionInfo(False, "Place added settings into an accordion").needs_reload_ui()
}))
```