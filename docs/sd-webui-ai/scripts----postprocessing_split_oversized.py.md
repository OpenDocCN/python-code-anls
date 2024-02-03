# `stable-diffusion-webui\scripts\postprocessing_split_oversized.py`

```py
# 导入 math 模块
import math

# 从 modules 模块中导入 scripts_postprocessing 和 ui_components
from modules import scripts_postprocessing, ui_components
# 导入 gradio 模块并重命名为 gr
import gradio as gr

# 定义一个函数用于将图片分割
def split_pic(image, inverse_xy, width, height, overlap_ratio):
    # 如果 inverse_xy 为真，则交换宽高
    if inverse_xy:
        from_w, from_h = image.height, image.width
        to_w, to_h = height, width
    else:
        from_w, from_h = image.width, image.height
        to_w, to_h = width, height
    # 计算新的高度
    h = from_h * to_w // from_w
    # 根据 inverse_xy 来调整图片大小
    if inverse_xy:
        image = image.resize((h, to_w))
    else:
        image = image.resize((to_w, h))

    # 计算分割次数
    split_count = math.ceil((h - to_h * overlap_ratio) / (to_h * (1.0 - overlap_ratio)))
    # 计算每次分割的步长
    y_step = (h - to_h) / (split_count - 1)
    # 遍历分割次数
    for i in range(split_count):
        y = int(y_step * i)
        # 根据 inverse_xy 来裁剪图片
        if inverse_xy:
            splitted = image.crop((y, 0, y + to_h, to_w))
        else:
            splitted = image.crop((0, y, to_w, y + to_h))
        # 生成裁剪后的图片
        yield splitted

# 定义一个类用于处理脚本后处理的分割过大图片
class ScriptPostprocessingSplitOversized(scripts_postprocessing.ScriptPostprocessing):
    name = "Split oversized images"
    order = 4000

    # 定义用户界面
    def ui(self):
        with ui_components.InputAccordion(False, label="Split oversized images") as enable:
            with gr.Row():
                # 创建一个滑块用于设置阈值
                split_threshold = gr.Slider(label='Threshold', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="postprocess_split_threshold")
                # 创建一个滑块用于设置重叠比例
                overlap_ratio = gr.Slider(label='Overlap ratio', value=0.2, minimum=0.0, maximum=0.9, step=0.05, elem_id="postprocess_overlap_ratio")

        # 返回用户界面元素
        return {
            "enable": enable,
            "split_threshold": split_threshold,
            "overlap_ratio": overlap_ratio,
        }
    # 处理后处理图像，根据参数进行处理
    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, split_threshold, overlap_ratio):
        # 如果未启用处理，则直接返回
        if not enable:
            return

        # 获取目标宽度和高度
        width = pp.shared.target_width
        height = pp.shared.target_height

        # 如果宽度或高度不存在，则直接返回
        if not width or not height:
            return

        # 根据图像的高度和宽度比较，计算比例和是否需要交换xy坐标
        if pp.image.height > pp.image.width:
            ratio = (pp.image.width * height) / (pp.image.height * width)
            inverse_xy = False
        else:
            ratio = (pp.image.height * width) / (pp.image.width * height)
            inverse_xy = True

        # 如果比例大于等于1且大于分割阈值，则直接返回
        if ratio >= 1.0 and ratio > split_threshold:
            return

        # 调用split_pic函数对图像进行分割，获取结果图像和其他图像
        result, *others = split_pic(pp.image, inverse_xy, width, height, overlap_ratio)

        # 更新处理后的图像为结果图像
        pp.image = result
        # 更新额外图像为其他图像的副本列表
        pp.extra_images = [pp.create_copy(x) for x in others]
```