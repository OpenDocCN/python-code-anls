# `stable-diffusion-webui\modules\processing_scripts\refiner.py`

```py
# 导入 gradio 库
import gradio as gr

# 从 modules 模块中导入 scripts 和 sd_models
from modules import scripts, sd_models
# 从 modules 模块中导入 ui_common 模块中的 create_refresh_button 函数
from modules.ui_common import create_refresh_button
# 从 modules 模块中导入 ui_components 模块中的 InputAccordion 类
from modules.ui_components import InputAccordion

# 定义 ScriptRefiner 类，继承自 scripts.ScriptBuiltinUI 类
class ScriptRefiner(scripts.ScriptBuiltinUI):
    # 设置 section 属性为 "accordions"
    section = "accordions"
    # 设置 create_group 属性为 False
    create_group = False

    # 初始化方法
    def __init__(self):
        pass

    # 返回标题为 "Refiner" 的方法
    def title(self):
        return "Refiner"

    # 显示方法，根据 is_img2img 返回 scripts.AlwaysVisible
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # UI 方法，根据 is_img2img 创建 UI 组件
    def ui(self, is_img2img):
        # 创建一个 InputAccordion 组件，label 为 "Refiner"，elem_id 为 self.elem_id("enable")
        with InputAccordion(False, label="Refiner", elem_id=self.elem_id("enable")) as enable_refiner:
            # 创建一个 Row 组件
            with gr.Row():
                # 创建一个 Dropdown 组件，label 为 'Checkpoint'，elem_id 为 self.elem_id("checkpoint")，choices 为 sd_models.checkpoint_tiles()，value 为 ''，tooltip 为 "switch to another model in the middle of generation"
                refiner_checkpoint = gr.Dropdown(label='Checkpoint', elem_id=self.elem_id("checkpoint"), choices=sd_models.checkpoint_tiles(), value='', tooltip="switch to another model in the middle of generation")
                # 调用 create_refresh_button 函数，传入参数 refiner_checkpoint, sd_models.list_models, lambda 函数和 self.elem_id("checkpoint_refresh")
                create_refresh_button(refiner_checkpoint, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, self.elem_id("checkpoint_refresh"))

                # 创建一个 Slider 组件，value 为 0.8，label 为 "Switch at"，minimum 为 0.01，maximum 为 1.0，step 为 0.01，elem_id 为 self.elem_id("switch_at")，tooltip 为 "fraction of sampling steps when the switch to refiner model should happen; 1=never, 0.5=switch in the middle of generation"
                refiner_switch_at = gr.Slider(value=0.8, label="Switch at", minimum=0.01, maximum=1.0, step=0.01, elem_id=self.elem_id("switch_at"), tooltip="fraction of sampling steps when the switch to refiner model should happen; 1=never, 0.5=switch in the middle of generation")

        # 定义 lookup_checkpoint 方法，根据 title 查找最接近的 checkpoint
        def lookup_checkpoint(title):
            info = sd_models.get_closet_checkpoint_match(title)
            return None if info is None else info.title

        # 设置 self.infotext_fields 属性为包含元组的列表，元组包含 enable_refiner, refiner_checkpoint 和 refiner_switch_at 以及对应的 lambda 函数或字符串
        self.infotext_fields = [
            (enable_refiner, lambda d: 'Refiner' in d),
            (refiner_checkpoint, lambda d: lookup_checkpoint(d.get('Refiner'))),
            (refiner_switch_at, 'Refiner switch at'),
        ]

        # 返回 enable_refiner, refiner_checkpoint, refiner_switch_at 组件
        return enable_refiner, refiner_checkpoint, refiner_switch_at
    # 设置函数，用于配置参数
    def setup(self, p, enable_refiner, refiner_checkpoint, refiner_switch_at):
        # 实际的实现在 sd_samplers_common.py 中的 apply_refiner 函数中

        # 如果不启用 refiner 或者 refiner_checkpoint 为 None、空字符串或者 "None"
        if not enable_refiner or refiner_checkpoint in (None, "", "None"):
            # 将 refiner_checkpoint 和 refiner_switch_at 设置为 None
            p.refiner_checkpoint = None
            p.refiner_switch_at = None
        else:
            # 否则将 refiner_checkpoint 和 refiner_switch_at 设置为传入的参数值
            p.refiner_checkpoint = refiner_checkpoint
            p.refiner_switch_at = refiner_switch_at
```