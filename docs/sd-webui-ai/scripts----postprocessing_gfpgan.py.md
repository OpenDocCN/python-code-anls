# `stable-diffusion-webui\scripts\postprocessing_gfpgan.py`

```py
# 从 PIL 库中导入 Image 模块
from PIL import Image
# 导入 numpy 库并使用别名 np
import numpy as np

# 从 modules 模块中导入 scripts_postprocessing、gfpgan_model、ui_components 模块
from modules import scripts_postprocessing, gfpgan_model, ui_components
# 导入 gradio 库并使用别名 gr
import gradio as gr

# 定义 ScriptPostprocessingGfpGan 类，继承自 scripts_postprocessing.ScriptPostprocessing 类
class ScriptPostprocessingGfpGan(scripts_postprocessing.ScriptPostprocessing):
    # 类属性 name 赋值为 "GFPGAN"
    name = "GFPGAN"
    # 类属性 order 赋值为 2000
    order = 2000

    # 定义 ui 方法
    def ui(self):
        # 创建一个 InputAccordion 对象，label 为 "GFPGAN"，enable 为 False
        with ui_components.InputAccordion(False, label="GFPGAN") as enable:
            # 创建一个 Slider 对象，设置最小值、最大值、步长、标签、初始值和元素 ID
            gfpgan_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Visibility", value=1.0, elem_id="extras_gfpgan_visibility")

        # 返回一个字典，包含 "enable" 和 "gfpgan_visibility" 两个键值对
        return {
            "enable": enable,
            "gfpgan_visibility": gfpgan_visibility,
        }

    # 定义 process 方法，接受 pp、enable、gfpgan_visibility 三个参数
    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, gfpgan_visibility):
        # 如果 gfpgan_visibility 为 0 或 enable 为 False，则直接返回
        if gfpgan_visibility == 0 or not enable:
            return

        # 使用 gfpgan_model 模块的 gfpgan_fix_faces 方法处理 pp.image，并转换为 Image 对象
        restored_img = gfpgan_model.gfpgan_fix_faces(np.array(pp.image, dtype=np.uint8))
        res = Image.fromarray(restored_img)

        # 如果 gfpgan_visibility 小于 1.0，则使用 Image.blend 方法混合 pp.image 和 res
        if gfpgan_visibility < 1.0:
            res = Image.blend(pp.image, res, gfpgan_visibility)

        # 更新 pp.image 为混合后的结果图像
        pp.image = res
        # 在 pp.info 中添加 "GFPGAN visibility" 键，值为 gfpgan_visibility 的四舍五入值（保留三位小数）
        pp.info["GFPGAN visibility"] = round(gfpgan_visibility, 3)
```