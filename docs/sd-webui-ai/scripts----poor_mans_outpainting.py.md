# `stable-diffusion-webui\scripts\poor_mans_outpainting.py`

```
# 导入 math 模块
import math

# 导入 modules.scripts 模块并重命名为 scripts
import modules.scripts as scripts

# 导入 gradio 模块并重命名为 gr
import gradio as gr

# 从 PIL 模块中导入 Image 和 ImageDraw 类
from PIL import Image, ImageDraw

# 从 modules 中导入 images 和 devices 模块
from modules import images, devices

# 从 modules.processing 中导入 Processed 和 process_images 函数
from modules.processing import Processed, process_images

# 从 modules.shared 中导入 opts 和 state 变量
from modules.shared import opts, state

# 定义 Script 类，继承自 scripts.Script 类
class Script(scripts.Script):
    
    # 定义 title 方法，返回字符串 "Poor man's outpainting"
    def title(self):
        return "Poor man's outpainting"

    # 定义 show 方法，返回参数 is_img2img
    def show(self, is_img2img):
        return is_img2img

    # 定义 ui 方法，根据 is_img2img 参数返回不同的 UI 元素
    def ui(self, is_img2img):
        # 如果 is_img2img 为 False，则返回 None
        if not is_img2img:
            return None

        # 创建滑块元素 pixels，用于调整扩展像素数
        pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=256, step=8, value=128, elem_id=self.elem_id("pixels"))
        
        # 创建滑块元素 mask_blur，用于调整遮罩模糊度
        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id=self.elem_id("mask_blur"))
        
        # 创建单选按钮元素 inpainting_fill，用于选择填充方式
        inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='fill', type="index", elem_id=self.elem_id("inpainting_fill"))
        
        # 创建复选框元素 direction，用于选择扩展方向
        direction = gr.CheckboxGroup(label="Outpainting direction", choices=['left', 'right', 'up', 'down'], value=['left', 'right', 'up', 'down'], elem_id=self.elem_id("direction"))

        # 返回 UI 元素列表
        return [pixels, mask_blur, inpainting_fill, direction]
```