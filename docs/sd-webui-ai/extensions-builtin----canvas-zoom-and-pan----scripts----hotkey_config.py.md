# `stable-diffusion-webui\extensions-builtin\canvas-zoom-and-pan\scripts\hotkey_config.py`

```
# 导入 gradio 库
import gradio as gr
# 从 modules 模块中导入 shared 模块
from modules import shared

# 更新共享选项模板，添加画布热键部分的选项
shared.options_templates.update(shared.options_section(('canvas_hotkey', "Canvas Hotkeys"), {
    # 设置缩放画布的热键选项
    "canvas_hotkey_zoom": shared.OptionInfo("Alt", "Zoom canvas", gr.Radio, {"choices": ["Shift","Ctrl", "Alt"]}).info("If you choose 'Shift' you cannot scroll horizontally, 'Alt' can cause a little trouble in firefox"),
    # 设置调整画笔大小的热键选项
    "canvas_hotkey_adjust": shared.OptionInfo("Ctrl", "Adjust brush size", gr.Radio, {"choices": ["Shift","Ctrl", "Alt"]}).info("If you choose 'Shift' you cannot scroll horizontally, 'Alt' can cause a little trouble in firefox"),
    # 设置移动画布的热键选项
    "canvas_hotkey_move": shared.OptionInfo("F", "Moving the canvas").info("To work correctly in firefox, turn off 'Automatically search the page text when typing' in the browser settings"),
    # 设置全屏模式的热键选项
    "canvas_hotkey_fullscreen": shared.OptionInfo("S", "Fullscreen Mode, maximizes the picture so that it fits into the screen and stretches it to its full width "),
    # 设置重置缩放和画布位置的热键选项
    "canvas_hotkey_reset": shared.OptionInfo("R", "Reset zoom and canvas positon"),
    # 设置切换重叠的热键选项
    "canvas_hotkey_overlap": shared.OptionInfo("O", "Toggle overlap").info("Technical button, neededs for testing"),
    # 设置在画布上启用工具提示的选项
    "canvas_show_tooltip": shared.OptionInfo(True, "Enable tooltip on the canvas"),
    # 设置自动展开不完全适合画布区域的图像的选项
    "canvas_auto_expand": shared.OptionInfo(True, "Automatically expands an image that does not fit completely in the canvas area, similar to manually pressing the S and R buttons"),
    # 设置在使用画布时将焦点从提示中移开的选项
    "canvas_blur_prompt": shared.OptionInfo(False, "Take the focus off the prompt when working with a canvas"),
    # 设置禁用不使用的功能的选项
    "canvas_disabled_functions": shared.OptionInfo(["Overlap"], "Disable function that you don't use", gr.CheckboxGroup, {"choices": ["Zoom","Adjust brush size", "Moving canvas","Fullscreen","Reset Zoom","Overlap"]}),
}))
```