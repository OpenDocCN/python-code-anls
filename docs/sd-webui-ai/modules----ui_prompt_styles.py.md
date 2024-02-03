# `stable-diffusion-webui\modules\ui_prompt_styles.py`

```
# 导入 gradio 库
import gradio as gr

# 导入自定义模块
from modules import shared, ui_common, ui_components, styles

# 符号常量定义
styles_edit_symbol = '\U0001f58c\uFE0F'  # 🖌️
styles_materialize_symbol = '\U0001f4cb'  # 📋
styles_copy_symbol = '\U0001f4dd'  # 📝

# 选择样式函数
def select_style(name):
    # 获取指定名称的样式
    style = shared.prompt_styles.styles.get(name)
    # 检查是否存在样式
    existing = style is not None
    # 检查名称是否为空
    empty = not name

    # 获取样式的提示信息和负面提示信息
    prompt = style.prompt if style else gr.update()
    negative_prompt = style.negative_prompt if style else gr.update()

    # 返回提示信息、负面提示信息、是否存在样式的更新状态、名称是否为空的更新状态
    return prompt, negative_prompt, gr.update(visible=existing), gr.update(visible=not empty)

# 保存样式函数
def save_style(name, prompt, negative_prompt):
    # 如果名称为空，则返回不可见的更新状态
    if not name:
        return gr.update(visible=False)

    # 创建新的样式对象
    style = styles.PromptStyle(name, prompt, negative_prompt)
    # 将样式添加到共享的样式字典中
    shared.prompt_styles.styles[style.name] = style
    # 保存样式到文件
    shared.prompt_styles.save_styles(shared.styles_filename)

    # 返回可见的更新状态
    return gr.update(visible=True)

# 删除样式函数
def delete_style(name):
    # 如果名称为空，则直接返回
    if name == "":
        return

    # 从共享的样式字典中删除指定名称的样式
    shared.prompt_styles.styles.pop(name, None)
    # 保存样式到文件
    shared.prompt_styles.save_styles(shared.styles_filename)

    # 返回空字符串
    return '', '', ''

# 应用样式函数
def materialize_styles(prompt, negative_prompt, styles):
    # 将样式应用到提示信息和负面提示信息
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
    negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(negative_prompt, styles)

    # 返回更新后的提示信息、负面提示信息和下拉框
    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=negative_prompt), gr.Dropdown.update(value=[])]

# 刷新样式函数
def refresh_styles():
    # 返回样式字典的名称列表
    return gr.update(choices=list(shared.prompt_styles.styles)), gr.update(choices=list(shared.prompt_styles.styles))

# 定义 UiPromptStyles 类
class UiPromptStyles:
    # 设置应用按钮的点击事件处理函数
    def setup_apply_button(self, button):
        # 点击按钮时执行 materialize_styles 函数，将 self.main_ui_prompt, self.main_ui_negative_prompt, self.dropdown 作为输入，更新它们的样式
        # 不显示进度条
        button.click(
            fn=materialize_styles,
            inputs=[self.main_ui_prompt, self.main_ui_negative_prompt, self.dropdown],
            outputs=[self.main_ui_prompt, self.main_ui_negative_prompt, self.dropdown],
            show_progress=False,
        ).then(fn=None, _js="function(){update_"+self.tabname+"_tokens(); closePopup();}", show_progress=False)
        # 点击按钮后执行 update_tabname_tokens 函数，关闭弹出窗口，不显示进度条
```