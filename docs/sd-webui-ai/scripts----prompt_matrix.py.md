# `stable-diffusion-webui\scripts\prompt_matrix.py`

```py
# 导入 math 模块
import math

# 导入 modules.scripts 模块并重命名为 scripts
import modules.scripts as scripts

# 导入 gradio 模块并重命名为 gr
import gradio as gr

# 从 modules 中导入 images 模块
from modules import images

# 从 modules.processing 中导入 process_images 函数
from modules.processing import process_images

# 从 modules.shared 中导入 opts 和 state 变量
from modules.shared import opts, state

# 从 modules 中导入 sd_samplers 模块
import modules.sd_samplers

# 定义一个函数，用于绘制 xy 坐标网格
def draw_xy_grid(xs, ys, x_label, y_label, cell):
    # 初始化结果列表
    res = []

    # 生成纵向文本标注
    ver_texts = [[images.GridAnnotation(y_label(y))] for y in ys]
    # 生成横向文本标注
    hor_texts = [[images.GridAnnotation(x_label(x))] for x in xs]

    # 初始化第一个处理结果
    first_processed = None

    # 设置作业总数
    state.job_count = len(xs) * len(ys)

    # 遍历 y 坐标
    for iy, y in enumerate(ys):
        # 遍历 x 坐标
        for ix, x in enumerate(xs):
            # 设置当前作业信息
            state.job = f"{ix + iy * len(xs) + 1} out of {len(xs) * len(ys)}"

            # 对当前坐标进行处理
            processed = cell(x, y)
            # 如果是第一个处理结果，保存下来
            if first_processed is None:
                first_processed = processed

            # 将处理结果的第一张图片添加到结果列表中
            res.append(processed.images[0])

    # 生成图片网格
    grid = images.image_grid(res, rows=len(ys))
    # 在图片网格上绘制文本标注
    grid = images.draw_grid_annotations(grid, res[0].width, res[0].height, hor_texts, ver_texts)

    # 将图片网格作为第一个处理结果的图片
    first_processed.images = [grid]

    # 返回第一个处理结果
    return first_processed

# 定义一个 Script 类，继承自 scripts.Script 类
class Script(scripts.Script):
    # 定义一个方法，返回标题为 "Prompt matrix"
    def title(self):
        return "Prompt matrix"
    # 定义一个 UI 方法，根据 is_img2img 参数确定是否为图片到图片的 UI
    def ui(self, is_img2img):
        # 在页面中插入一个换行
        gr.HTML('<br />')
        # 创建一个行元素
        with gr.Row():
            # 创建一个列元素
            with gr.Column():
                # 创建一个复选框，用于选择是否将变量部分放在提示的开头
                put_at_start = gr.Checkbox(label='Put variable parts at start of prompt', value=False, elem_id=self.elem_id("put_at_start"))
                # 创建一个复选框，用于选择是否为每张图片使用不同的种子
                different_seeds = gr.Checkbox(label='Use different seed for each picture', value=False, elem_id=self.elem_id("different_seeds"))
            # 创建一个列元素
            with gr.Column():
                # 创建一个单选框，用于选择提示类型（正面或负面）
                prompt_type = gr.Radio(["positive", "negative"], label="Select prompt", elem_id=self.elem_id("prompt_type"), value="positive")
                # 创建一个单选框，用于选择变量之间的连接字符（逗号或空格）
                variations_delimiter = gr.Radio(["comma", "space"], label="Select joining char", elem_id=self.elem_id("variations_delimiter"), value="comma")
            # 创建一个列元素
            with gr.Column():
                # 创建一个滑块，用于调整网格边距的大小
                margin_size = gr.Slider(label="Grid margins (px)", minimum=0, maximum=500, value=0, step=2, elem_id=self.elem_id("margin_size"))

        # 返回 UI 元素列表
        return [put_at_start, different_seeds, prompt_type, variations_delimiter, margin_size]
```