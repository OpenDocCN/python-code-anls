# `stable-diffusion-webui\scripts\prompts_from_file.py`

```
# 导入 copy 模块，用于复制对象
import copy
# 导入 random 模块，用于生成随机数
import random
# 导入 shlex 模块，用于解析命令行参数
import shlex

# 导入 modules.scripts 模块，并重命名为 scripts
import modules.scripts as scripts
# 导入 gradio 模块，并重命名为 gr
import gradio as gr

# 从 modules 中导入 sd_samplers、errors、sd_models 模块
from modules import sd_samplers, errors, sd_models
# 从 modules.processing 中导入 Processed、process_images 函数
from modules.processing import Processed, process_images
# 从 modules.shared 中导入 state 对象
from modules.shared import state

# 定义函数，处理模型标签
def process_model_tag(tag):
    # 获取与标签最接近的检查点信息
    info = sd_models.get_closet_checkpoint_match(tag)
    # 断言检查点信息不为空
    assert info is not None, f'Unknown checkpoint: {tag}'
    # 返回检查点名称
    return info.name

# 定义函数，处理字符串标签
def process_string_tag(tag):
    return tag

# 定义函数，处理整数标签
def process_int_tag(tag):
    return int(tag)

# 定义函数，处理浮点数标签
def process_float_tag(tag):
    return float(tag)

# 定义函数，处理布尔值标签
def process_boolean_tag(tag):
    return True if (tag == "true") else False

# 定义字典，将标签与处理函数对应起来
prompt_tags = {
    "sd_model": process_model_tag,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}

# 定义函数，解析命令行参数
def cmdargs(line):
    # 使用 shlex 模块解析命令行参数
    args = shlex.split(line)
    # 初始化位置
    pos = 0
    # 初始化结果字典
    res = {}
    # 当前位置小于参数列表长度时，继续循环
    while pos < len(args):
        # 获取当前位置的参数
        arg = args[pos]

        # 断言参数以 "--" 开头，否则抛出异常
        assert arg.startswith("--"), f'must start with "--": {arg}'
        # 断言当前位置加一小于参数列表长度，否则抛出异常
        assert pos+1 < len(args), f'missing argument for command line option {arg}'

        # 提取标签
        tag = arg[2:]

        # 如果标签是 "prompt" 或 "negative_prompt"
        if tag == "prompt" or tag == "negative_prompt":
            # 移动到下一个参数位置
            pos += 1
            # 获取提示信息
            prompt = args[pos]
            # 继续移动到下一个参数位置，直到遇到下一个标签
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            # 将提示信息存入结果字典
            res[tag] = prompt
            continue

        # 获取标签对应的处理函数
        func = prompt_tags.get(tag, None)
        # 断言函数存在，否则抛出异常
        assert func, f'unknown commandline option: {arg}'

        # 获取标签对应的值
        val = args[pos+1]
        # 如果标签是 "sampler_name"，将值转换为小写并获取对应的采样器对象
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        # 将值经过处理函数处理后存入结果字典
        res[tag] = func(val)

        # 移动到下一个参数位置
        pos += 2

    # 返回结果字典
    return res
# 从文件中加载提示信息，如果文件为空则返回空值，否则返回处理后的文件内容和更新的控件信息
def load_prompt_file(file):
    # 如果文件为空，则返回空值和更新的控件信息
    if file is None:
        return None, gr.update(), gr.update(lines=7)
    else:
        # 将文件内容解码为 UTF-8 格式，去除空格并按行分割，组成列表
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]
        # 返回空值和连接后的文件内容，以及更新的控件信息
        return None, "\n".join(lines), gr.update(lines=7)

# 定义一个名为 Script 的类，继承自 scripts.Script 类
class Script(scripts.Script):
    # 返回脚本的标题
    def title(self):
        return "Prompts from file or textbox"

    # 定义用户界面
    def ui(self, is_img2img):
        # 创建复选框控件，用于选择是否在每行迭代种子
        checkbox_iterate = gr.Checkbox(label="Iterate seed every line", value=False, elem_id=self.elem_id("checkbox_iterate"))
        # 创建复选框控件，用于选择是否在所有行使用相同的随机种子
        checkbox_iterate_batch = gr.Checkbox(label="Use same random seed for all lines", value=False, elem_id=self.elem_id("checkbox_iterate_batch"))
        # 创建单选框控件，用于选择在哪里插入提示信息
        prompt_position = gr.Radio(["start", "end"], label="Insert prompts at the", elem_id=self.elem_id("prompt_position"), value="start")

        # 创建文本框控件，用于输入提示信息
        prompt_txt = gr.Textbox(label="List of prompt inputs", lines=1, elem_id=self.elem_id("prompt_txt"))
        # 创建文件上传控件，用于上传提示信息文件
        file = gr.File(label="Upload prompt inputs", type='binary', elem_id=self.elem_id("file"))

        # 当文件上传控件内容改变时，调用 load_prompt_file 函数处理文件内容，并更新相关控件信息
        file.change(fn=load_prompt_file, inputs=[file], outputs=[file, prompt_txt, prompt_txt], show_progress=False)

        # 当文本框控件内容改变时，根据内容是否包含换行符来更新文本框的行数
        # 如果包含换行符，则更新为 7 行，否则更新为 2 行
        # 不会缩小回 1 行，因为这会导致控件忽略 [enter] 键，用户可能不清楚需要使用 shift-enter
        prompt_txt.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[prompt_txt], outputs=[prompt_txt], show_progress=False)
        
        # 返回所有创建的控件列表
        return [checkbox_iterate, checkbox_iterate_batch, prompt_position, prompt_txt]
```