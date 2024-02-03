# `stable-diffusion-webui\scripts\postprocessing_caption.py`

```py
# 从 modules 模块中导入所需的模块
from modules import scripts_postprocessing, ui_components, deepbooru, shared
# 导入 gradio 库并重命名为 gr
import gradio as gr

# 创建 ScriptPostprocessingCeption 类，继承自 scripts_postprocessing.ScriptPostprocessing 类
class ScriptPostprocessingCeption(scripts_postprocessing.ScriptPostprocessing):
    # 定义类属性 name 为 "Caption"
    name = "Caption"
    # 定义类属性 order 为 4000
    order = 4000

    # 定义 ui 方法，用于创建用户界面
    def ui(self):
        # 创建一个 InputAccordion 组件，label 为 "Caption"，默认展开
        with ui_components.InputAccordion(False, label="Caption") as enable:
            # 创建一个 CheckboxGroup 组件，初始值为 ["Deepbooru"]，选项为 ["Deepbooru", "BLIP"]，不显示标签
            option = gr.CheckboxGroup(value=["Deepbooru"], choices=["Deepbooru", "BLIP"], show_label=False)

        # 返回一个字典，包含 "enable" 和 "option" 两个键值对
        return {
            "enable": enable,
            "option": option,
        }

    # 定义 process 方法，用于处理后处理图像
    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, option):
        # 如果 enable 为假，则直接返回
        if not enable:
            return

        # 创建一个列表 captions，初始值为 pp.caption
        captions = [pp.caption]

        # 如果 "Deepbooru" 在 option 中，则调用 deepbooru.model.tag 方法并将结果添加到 captions 中
        if "Deepbooru" in option:
            captions.append(deepbooru.model.tag(pp.image))

        # 如果 "BLIP" 在 option 中，则调用 shared.interrogator.generate_caption 方法并将结果添加到 captions 中
        if "BLIP" in option:
            captions.append(shared.interrogator.generate_caption(pp.image))

        # 将 captions 中非空的元素用逗号连接起来，更新 pp.caption
        pp.caption = ", ".join([x for x in captions if x])
```