# `stable-diffusion-webui\scripts\postprocessing_codeformer.py`

```
# 从 PIL 库中导入 Image 模块
from PIL import Image
# 导入 numpy 库并使用别名 np
import numpy as np

# 从 modules 模块中导入 scripts_postprocessing、codeformer_model、ui_components 模块
from modules import scripts_postprocessing, codeformer_model, ui_components
# 导入 gradio 库并使用别名 gr
import gradio as gr

# 定义 ScriptPostprocessingCodeFormer 类，继承自 scripts_postprocessing.ScriptPostprocessing 类
class ScriptPostprocessingCodeFormer(scripts_postprocessing.ScriptPostprocessing):
    # 定义类属性 name 为 "CodeFormer"
    name = "CodeFormer"
    # 定义类属性 order 为 3000
    order = 3000

    # 定义 ui 方法，返回 UI 组件
    def ui(self):
        # 创建一个 InputAccordion 组件，label 为 "CodeFormer"
        with ui_components.InputAccordion(False, label="CodeFormer") as enable:
            # 创建一个 Row 组件
            with gr.Row():
                # 创建一个 Slider 组件，用于调整可见度
                codeformer_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Visibility", value=1.0, elem_id="extras_codeformer_visibility")
                # 创建一个 Slider 组件，用于调整权重
                codeformer_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Weight (0 = maximum effect, 1 = minimum effect)", value=0, elem_id="extras_codeformer_weight")

        # 返回 UI 组件字典
        return {
            "enable": enable,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
        }

    # 定义 process 方法，处理后处理图像
    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, codeformer_visibility, codeformer_weight):
        # 如果可见度为 0 或未启用 CodeFormer，则直接返回
        if codeformer_visibility == 0 or not enable:
            return

        # 使用 codeformer_model.codeformer 恢复图像
        restored_img = codeformer_model.codeformer.restore(np.array(pp.image, dtype=np.uint8), w=codeformer_weight)
        # 将恢复后的图像转换为 PIL Image 对象
        res = Image.fromarray(restored_img)

        # 如果可见度小于 1.0，则混合原始图像和恢复后的图像
        if codeformer_visibility < 1.0:
            res = Image.blend(pp.image, res, codeformer_visibility)

        # 更新后处理图像和信息
        pp.image = res
        pp.info["CodeFormer visibility"] = round(codeformer_visibility, 3)
        pp.info["CodeFormer weight"] = round(codeformer_weight, 3)
```