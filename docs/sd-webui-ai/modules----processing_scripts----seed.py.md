# `stable-diffusion-webui\modules\processing_scripts\seed.py`

```py
# 导入 json 模块
import json

# 导入 gradio 模块，并重命名为 gr
import gradio as gr

# 从 modules 中导入 scripts、ui、errors 模块
from modules import scripts, ui, errors

# 从 modules.shared 模块中导入 cmd_opts
from modules.shared import cmd_opts

# 从 modules.ui_components 模块中导入 ToolButton
from modules.ui_components import ToolButton

# 定义 ScriptSeed 类，继承自 ScriptBuiltinUI 类
class ScriptSeed(scripts.ScriptBuiltinUI):
    # 设置 section 属性为 "seed"
    section = "seed"
    # 设置 create_group 属性为 False
    create_group = False

    # 初始化方法
    def __init__(self):
        # 初始化 seed、reuse_seed、reuse_subseed 属性为 None
        self.seed = None
        self.reuse_seed = None
        self.reuse_subseed = None

    # 返回标题为 "Seed" 的方法
    def title(self):
        return "Seed"

    # 根据 is_img2img 参数返回 AlwaysVisible 类
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # 设置种子相关参数的方法
    def setup(self, p, seed, seed_checkbox, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h):
        # 将 seed 参数赋值给 p 对象的 seed 属性
        p.seed = seed

        # 如果 seed_checkbox 为 True 且 subseed_strength 大于 0
        if seed_checkbox and subseed_strength > 0:
            # 将 subseed 赋值给 p 对象的 subseed 属性
            p.subseed = subseed
            # 将 subseed_strength 赋值给 p 对象的 subseed_strength 属性

        # 如果 seed_checkbox 为 True 且 seed_resize_from_w 大于 0 且 seed_resize_from_h 大于 0
        if seed_checkbox and seed_resize_from_w > 0 and seed_resize_from_h > 0:
            # 将 seed_resize_from_w 赋值给 p 对象的 seed_resize_from_w 属性
            p.seed_resize_from_w = seed_resize_from_w
            # 将 seed_resize_from_h 赋值给 p 对象的 seed_resize_from_h 属性

# 定义 connect_reuse_seed 函数，接受 seed、reuse_seed、generation_info、is_subseed 参数
def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, is_subseed):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""
    # 定义一个函数，用于复制种子信息
    def copy_seed(gen_info_string: str, index):
        # 初始化结果为-1
        res = -1

        # 尝试解析传入的生成信息字符串为 JSON 格式
        try:
            gen_info = json.loads(gen_info_string)
            # 根据传入的索引值和生成信息中的第一个图像索引值计算新的索引值
            index -= gen_info.get('index_of_first_image', 0)

            # 如果是子种子并且子种子强度大于0
            if is_subseed and gen_info.get('subseed_strength', 0) > 0:
                # 获取所有子种子列表
                all_subseeds = gen_info.get('all_subseeds', [-1])
                # 根据新的索引值获取对应的子种子
                res = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
            else:
                # 获取所有种子列表
                all_seeds = gen_info.get('all_seeds', [-1])
                # 根据新的索引值获取对应的种子
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]

        # 如果解析 JSON 失败
        except json.decoder.JSONDecodeError:
            # 如果生成信息字符串不为空，报告错误
            if gen_info_string:
                errors.report(f"Error parsing JSON generation info: {gen_info_string}")

        # 返回结果和更新图形
        return [res, gr.update()]

    # 点击复用种子按钮，调用 copy_seed 函数
    reuse_seed.click(
        fn=copy_seed,
        _js="(x, y) => [x, selected_gallery_index()]",
        show_progress=False,
        inputs=[generation_info, seed],
        outputs=[seed, seed]
    )
```