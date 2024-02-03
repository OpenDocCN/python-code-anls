# `stable-diffusion-webui\scripts\postprocessing_create_flipped_copies.py`

```
# 从 PIL 库中导入 ImageOps 和 Image 模块
from PIL import ImageOps, Image

# 从 modules 模块中导入 scripts_postprocessing 和 ui_components 模块
from modules import scripts_postprocessing, ui_components
# 导入 gradio 库并重命名为 gr
import gradio as gr

# 定义一个名为 ScriptPostprocessingCreateFlippedCopies 的类，继承自 scripts_postprocessing.ScriptPostprocessing 类
class ScriptPostprocessingCreateFlippedCopies(scripts_postprocessing.ScriptPostprocessing):
    # 类属性：名称为 "Create flipped copies"
    name = "Create flipped copies"
    # 类属性：顺序为 4000
    order = 4000

    # 定义一个 ui 方法
    def ui(self):
        # 创建一个 InputAccordion 组件，label 为 "Create flipped copies"，默认展开状态为 False
        with ui_components.InputAccordion(False, label="Create flipped copies") as enable:
            # 创建一个行布局
            with gr.Row():
                # 创建一个 CheckboxGroup 组件，初始值为 ["Horizontal"]，选项为 ["Horizontal", "Vertical", "Both"]，不显示标签
                option = gr.CheckboxGroup(value=["Horizontal"], choices=["Horizontal", "Vertical", "Both"], show_label=False)

        # 返回一个字典，包含 "enable" 和 "option" 两个键值对
        return {
            "enable": enable,
            "option": option,
        }

    # 定义一个 process 方法，接受 pp、enable、option 三个参数
    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, option):
        # 如果 enable 为 False，则直接返回
        if not enable:
            return

        # 如果 "Horizontal" 在 option 中
        if "Horizontal" in option:
            # 将 pp.image 水平镜像后的图像添加到 pp.extra_images 中
            pp.extra_images.append(ImageOps.mirror(pp.image))

        # 如果 "Vertical" 在 option 中
        if "Vertical" in option:
            # 将 pp.image 垂直翻转后的图像添加到 pp.extra_images 中
            pp.extra_images.append(pp.image.transpose(Image.Transpose.FLIP_TOP_BOTTOM))

        # 如果 "Both" 在 option 中
        if "Both" in option:
            # 将 pp.image 先垂直翻转再水平镜像后的图像添加到 pp.extra_images 中
            pp.extra_images.append(pp.image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).transpose(Image.Transpose.FLIP_LEFT_RIGHT))
```