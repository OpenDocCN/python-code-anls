# `stable-diffusion-webui\scripts\processing_autosized_crop.py`

```py
# 导入所需的模块
from PIL import Image
from modules import scripts_postprocessing, ui_components
import gradio as gr

# 定义函数，根据指定宽度和高度对图像进行中心裁剪
def center_crop(image: Image, w: int, h: int):
    # 获取图像的宽度和高度
    iw, ih = image.size
    # 根据比例确定裁剪区域
    if ih / h < iw / w:
        sw = w * ih / h
        box = (iw - sw) / 2, 0, iw - (iw - sw) / 2, ih
    else:
        sh = h * iw / w
        box = 0, (ih - sh) / 2, iw, ih - (ih - sh) / 2
    # 返回裁剪后的图像
    return image.resize((w, h), Image.Resampling.LANCZOS, box)

# 定义函数，根据指定条件对图像进行多次裁剪
def multicrop_pic(image: Image, mindim, maxdim, minarea, maxarea, objective, threshold):
    # 获取图像的宽度和高度
    iw, ih = image.size
    # 定义误差函数
    err = lambda w, h: 1 - (lambda x: x if x < 1 else 1 / x)(iw / ih / (w / h))
    # 根据条件选择最佳裁剪尺寸
    wh = max(((w, h) for w in range(mindim, maxdim + 1, 64) for h in range(mindim, maxdim + 1, 64)
              if minarea <= w * h <= maxarea and err(w, h) <= threshold),
             key=lambda wh: (wh[0] * wh[1], -err(*wh))[::1 if objective == 'Maximize area' else -1],
             default=None
             )
    # 如果找到最佳裁剪尺寸，则进行中心裁剪
    return wh and center_crop(image, *wh)

# 定义一个脚本后处理类，实现自动尺寸裁剪功能
class ScriptPostprocessingAutosizedCrop(scripts_postprocessing.ScriptPostprocessing):
    name = "Auto-sized crop"
    order = 4000
    # 定义用户界面函数，创建一个输入折叠面板，用于自动裁剪图片
    def ui(self):
        # 创建一个输入折叠面板，设置为不展开，标题为"Auto-sized crop"
        with ui_components.InputAccordion(False, label="Auto-sized crop") as enable:
            # 在折叠面板中添加说明文本
            gr.Markdown('Each image is center-cropped with an automatically chosen width and height.')
            # 创建一个水平排列的行
            with gr.Row():
                # 创建一个滑块，用于设置裁剪的最小尺寸，初始值为384
                mindim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension lower bound", value=384, elem_id="postprocess_multicrop_mindim")
                # 创建一个滑块，用于设置裁剪的最大尺寸，初始值为768
                maxdim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension upper bound", value=768, elem_id="postprocess_multicrop_maxdim")
            # 创建另一个水平排列的行
            with gr.Row():
                # 创建一个滑块，用于设置裁剪的最小面积，初始值为64*64
                minarea = gr.Slider(minimum=64 * 64, maximum=2048 * 2048, step=1, label="Area lower bound", value=64 * 64, elem_id="postprocess_multicrop_minarea")
                # 创建一个滑块，用于设置裁剪的最大面积，初始值为640*640
                maxarea = gr.Slider(minimum=64 * 64, maximum=2048 * 2048, step=1, label="Area upper bound", value=640 * 640, elem_id="postprocess_multicrop_maxarea")
            # 创建另一个水平排列的行
            with gr.Row():
                # 创建一个单选按钮，用于选择裁剪的目标，初始值为"Maximize area"
                objective = gr.Radio(["Maximize area", "Minimize error"], value="Maximize area", label="Resizing objective", elem_id="postprocess_multicrop_objective")
                # 创建一个滑块，用于设置错误阈值，初始值为0.1
                threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="Error threshold", value=0.1, elem_id="postprocess_multicrop_threshold")

        # 返回用户界面元素的字典
        return {
            "enable": enable,
            "mindim": mindim,
            "maxdim": maxdim,
            "minarea": minarea,
            "maxarea": maxarea,
            "objective": objective,
            "threshold": threshold,
        }

    # 定义处理函数，根据用户设置裁剪图片
    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, mindim, maxdim, minarea, maxarea, objective, threshold):
        # 如果未启用自动裁剪，则直接返回
        if not enable:
            return

        # 调用多裁剪函数，裁剪图片
        cropped = multicrop_pic(pp.image, mindim, maxdim, minarea, maxarea, objective, threshold)
        # 如果成功裁剪，则更新后处理图片
        if cropped is not None:
            pp.image = cropped
        else:
            # 如果无法找到符合错误阈值的合适尺寸，则打印跳过信息
            print(f"skipped {pp.image.width}x{pp.image.height} image (can't find suitable size within error threshold)")
```