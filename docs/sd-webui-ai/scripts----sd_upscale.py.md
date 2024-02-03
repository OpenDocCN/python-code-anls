# `stable-diffusion-webui\scripts\sd_upscale.py`

```py
# 导入 math 模块
import math

# 导入自定义模块 modules.scripts，并重命名为 scripts
import modules.scripts as scripts

# 导入 gradio 模块，并重命名为 gr
import gradio as gr

# 从 PIL 模块中导入 Image 类
from PIL import Image

# 从自定义模块 modules 中导入 processing, shared, images, devices
from modules import processing, shared, images, devices

# 从 processing 模块中导入 Processed 类
from modules.processing import Processed

# 从 shared 模块中导入 opts, state
from modules.shared import opts, state

# 定义 Script 类，继承自 scripts.Script 类
class Script(scripts.Script):
    
    # 定义 title 方法，返回字符串 "SD upscale"
    def title(self):
        return "SD upscale"

    # 定义 show 方法，接受参数 is_img2img，返回 is_img2img
    def show(self, is_img2img):
        return is_img2img

    # 定义 ui 方法，接受参数 is_img2img
    def ui(self, is_img2img):
        # 创建 HTML 元素，显示文本内容
        info = gr.HTML("<p style=\"margin-bottom:0.75em\">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>")
        
        # 创建滑块元素，用于设置 Tile overlap 的值
        overlap = gr.Slider(minimum=0, maximum=256, step=16, label='Tile overlap', value=64, elem_id=self.elem_id("overlap"))
        
        # 创建滑块元素，用于设置 Scale Factor 的值
        scale_factor = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label='Scale Factor', value=2.0, elem_id=self.elem_id("scale_factor"))
        
        # 创建单选按钮元素，用于选择 Upscaler
        upscaler_index = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index", elem_id=self.elem_id("upscaler_index"))

        # 返回包含所有 UI 元素的列表
        return [info, overlap, upscaler_index, scale_factor]
```