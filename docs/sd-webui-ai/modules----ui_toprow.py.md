# `stable-diffusion-webui\modules\ui_toprow.py`

```
# 导入 gradio 库
import gradio as gr

# 从 modules 中导入 shared 和 ui_prompt_styles 模块
from modules import shared, ui_prompt_styles
# 从 modules 中导入 images 模块
import modules.images

# 从 modules.ui_components 中导入 ToolButton 类
from modules.ui_components import ToolButton

# 定义 Toprow 类
class Toprow:
    """Creates a top row UI with prompts, generate button, styles, extra little buttons for things, and enables some functionality related to their operation"""

    # 初始化类属性
    prompt = None
    prompt_img = None
    negative_prompt = None

    button_interrogate = None
    button_deepbooru = None

    interrupt = None
    skip = None
    submit = None

    paste = None
    clear_prompt_button = None
    apply_styles = None
    restore_progress_button = None

    token_counter = None
    token_button = None
    negative_token_counter = None
    negative_token_button = None

    ui_styles = None

    submit_box = None

    # 初始化方法
    def __init__(self, is_img2img, is_compact=False, id_part=None):
        # 如果 id_part 为 None，则根据 is_img2img 的值设置 id_part
        if id_part is None:
            id_part = "img2img" if is_img2img else "txt2img"

        self.id_part = id_part
        self.is_img2img = is_img2img
        self.is_compact = is_compact

        # 如果不是紧凑模式，则创建经典的顶部行
        if not is_compact:
            with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
                self.create_classic_toprow()
        # 如果是紧凑模式，则创建提交框
        else:
            self.create_submit_box()

    # 创建经典的顶部行
    def create_classic_toprow(self):
        self.create_prompts()

        # 创建包含操作的列
        with gr.Column(scale=1, elem_id=f"{self.id_part}_actions_column"):
            self.create_submit_box()

            self.create_tools_row()

            self.create_styles_ui()

    # 创建内联的顶部行提示
    def create_inline_toprow_prompts(self):
        # 如果不是紧凑模式，则返回
        if not self.is_compact:
            return

        self.create_prompts()

        with gr.Row(elem_classes=["toprow-compact-stylerow"]):
            with gr.Column(elem_classes=["toprow-compact-tools"]):
                self.create_tools_row()
            with gr.Column():
                self.create_styles_ui()

    # 创建内联的顶部行图片
    def create_inline_toprow_image(self):
        # 如果不是紧凑模式，则返回
        if not self.is_compact:
            return

        self.submit_box.render()
    # 创建提示文本框和文件上传框
    def create_prompts(self):
        # 创建一个列容器，设置容器的 ID 和样式
        with gr.Column(elem_id=f"{self.id_part}_prompt_container", elem_classes=["prompt-container-compact"] if self.is_compact else [], scale=6):
            # 在列容器中创建一个行容器，设置行容器的 ID 和样式
            with gr.Row(elem_id=f"{self.id_part}_prompt_row", elem_classes=["prompt-row"]):
                # 创建一个文本框用于输入提示文本，设置文本框的 ID、行数、占位符和样式
                self.prompt = gr.Textbox(label="Prompt", elem_id=f"{self.id_part}_prompt", show_label=False, lines=3, placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)", elem_classes=["prompt"])
                # 创建一个文件上传框用于上传图片，设置文件上传框的 ID、类型和可见性
                self.prompt_img = gr.File(label="", elem_id=f"{self.id_part}_prompt_image", file_count="single", type="binary", visible=False)

            # 在列容器中创建另一个行容器，设置行容器的 ID 和样式
            with gr.Row(elem_id=f"{self.id_part}_neg_prompt_row", elem_classes=["prompt-row"]):
                # 创建一个文本框用于输入负面提示文本，设置文本框的 ID、行数、占位符和样式
                self.negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{self.id_part}_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)", elem_classes=["prompt"])

        # 当上传框内容改变时，调用指定的函数处理图片数据，设置输入和输出，不显示进度条
        self.prompt_img.change(
            fn=modules.images.image_data,
            inputs=[self.prompt_img],
            outputs=[self.prompt, self.prompt_img],
            show_progress=False,
        )
    # 创建提交框，包括中断、跳过和生成按钮
    def create_submit_box(self):
        # 创建一个行元素作为提交框容器，根据是否紧凑设置不同的样式类，并根据是否紧凑决定是否渲染
        with gr.Row(elem_id=f"{self.id_part}_generate_box", elem_classes=["generate-box"] + (["generate-box-compact"] if self.is_compact else []), render=not self.is_compact) as submit_box:
            # 将提交框赋值给实例变量
            self.submit_box = submit_box

            # 创建中断按钮
            self.interrupt = gr.Button('Interrupt', elem_id=f"{self.id_part}_interrupt", elem_classes="generate-box-interrupt")
            # 创建跳过按钮
            self.skip = gr.Button('Skip', elem_id=f"{self.id_part}_skip", elem_classes="generate-box-skip")
            # 创建生成按钮
            self.submit = gr.Button('Generate', elem_id=f"{self.id_part}_generate", variant='primary')

            # 设置跳过按钮的点击事件，调用共享状态的跳过方法
            self.skip.click(
                fn=lambda: shared.state.skip(),
                inputs=[],
                outputs=[],
            )

            # 设置中断按钮的点击事件，调用共享状态的中断方法
            self.interrupt.click(
                fn=lambda: shared.state.interrupt(),
                inputs=[],
                outputs=[],
            )

    # 创建样式界面
    def create_styles_ui(self):
        # 实例化 UiPromptStyles 类，传入部分 ID、提示和负面提示
        self.ui_styles = ui_prompt_styles.UiPromptStyles(self.id_part, self.prompt, self.negative_prompt)
        # 设置应用按钮的样式
        self.ui_styles.setup_apply_button(self.apply_styles)
```