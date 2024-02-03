# `stable-diffusion-webui\modules\generation_parameters_copypaste.py`

```
# 导入必要的模块
from __future__ import annotations
import base64
import io
import json
import os
import re

# 导入 gradio 模块
import gradio as gr
# 导入自定义模块
from modules.paths import data_path
from modules import shared, ui_tempdir, script_callbacks, processing
# 导入 PIL 模块中的 Image 类
from PIL import Image

# 定义正则表达式模式
re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")
re_hypernet_hash = re.compile("\(([0-9a-f]+)\)$"
# 定义类型为 gr.update() 的变量
type_of_gr_update = type(gr.update())

# 定义 ParamBinding 类
class ParamBinding:
    def __init__(self, paste_button, tabname, source_text_component=None, source_image_component=None, source_tabname=None, override_settings_component=None, paste_field_names=None):
        self.paste_button = paste_button
        self.tabname = tabname
        self.source_text_component = source_text_component
        self.source_image_component = source_image_component
        self.source_tabname = source_tabname
        self.override_settings_component = override_settings_component
        self.paste_field_names = paste_field_names or []

# 初始化 paste_fields 和 registered_param_bindings
paste_fields: dict[str, dict] = {}
registered_param_bindings: list[ParamBinding] = []

# 重置 paste_fields 和 registered_param_bindings
def reset():
    paste_fields.clear()
    registered_param_bindings.clear()

# 对文本进行引用处理
def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)

# 对引用进行解除处理
def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text

    try:
        return json.loads(text)
    except Exception:
        return text

# 从 URL 文本中获取图像数据
def image_from_url_text(filedata):
    if filedata is None:
        return None

    if type(filedata) == list and filedata and type(filedata[0]) == dict and filedata[0].get("is_file", False):
        filedata = filedata[0]
    # 检查 filedata 是否为字典类型且包含 "is_file" 键
    if type(filedata) == dict and filedata.get("is_file", False):
        # 获取文件名
        filename = filedata["name"]
        # 检查文件是否在正确的目录中
        is_in_right_dir = ui_tempdir.check_tmp_file(shared.demo, filename)
        # 断言文件在允许的目录中，否则抛出异常
        assert is_in_right_dir, 'trying to open image file outside of allowed directories'
        
        # 去除文件名中的查询参数
        filename = filename.rsplit('?', 1)[0]
        # 返回打开的图像文件
        return Image.open(filename)

    # 如果 filedata 是列表类型
    if type(filedata) == list:
        # 如果列表为空，返回 None
        if len(filedata) == 0:
            return None
        
        # 取列表中的第一个元素
        filedata = filedata[0]

    # 如果 filedata 以 "data:image/png;base64," 开头
    if filedata.startswith("data:image/png;base64,"):
        # 去除前缀
        filedata = filedata[len("data:image/png;base64,"):]

    # 将 base64 编码的数据解码为字节流
    filedata = base64.decodebytes(filedata.encode('utf-8'))
    # 从字节流中打开图像
    image = Image.open(io.BytesIO(filedata))
    # 返回图像对象
    return image
# 将粘贴字段添加到 paste_fields 字典中，包括标签名、初始图像、字段和覆盖设置组件
def add_paste_fields(tabname, init_img, fields, override_settings_component=None):
    paste_fields[tabname] = {"init_img": init_img, "fields": fields, "override_settings_component": override_settings_component}

    # 为了向后兼容现有扩展，导入 modules.ui 模块
    import modules.ui
    # 如果标签名为 'txt2img'，则将字段赋值给 modules.ui.txt2img_paste_fields
    if tabname == 'txt2img':
        modules.ui.txt2img_paste_fields = fields
    # 如果标签名为 'img2img'，则将字段赋值给 modules.ui.img2img_paste_fields


# 创建按钮字典，键为标签名，值为对应的按钮对象
def create_buttons(tabs_list):
    buttons = {}
    for tab in tabs_list:
        buttons[tab] = gr.Button(f"Send to {tab}", elem_id=f"{tab}_tab")
    return buttons


# 绑定按钮功能，用于向后兼容旧版本；不建议使用此函数，应使用 register_paste_params_button
def bind_buttons(buttons, send_image, send_generate_info):
    for tabname, button in buttons.items():
        source_text_component = send_generate_info if isinstance(send_generate_info, gr.components.Component) else None
        source_tabname = send_generate_info if isinstance(send_generate_info, str) else None

        register_paste_params_button(ParamBinding(paste_button=button, tabname=tabname, source_text_component=source_text_component, source_image_component=send_image, source_tabname=source_tabname))


# 注册粘贴参数按钮，将参数绑定添加到 registered_param_bindings 列表中
def register_paste_params_button(binding: ParamBinding):
    registered_param_bindings.append(binding)


# 连接粘贴参数按钮
def connect_paste_params_buttons():
# 发送图像和尺寸
def send_image_and_dimensions(x):
    if isinstance(x, Image.Image):
        img = x
    else:
        img = image_from_url_text(x)

    if shared.opts.send_size and isinstance(img, Image.Image):
        w = img.width
        h = img.height
    else:
        w = gr.update()
        h = gr.update()

    return img, w, h


# 恢复旧的高清修复参数
def restore_old_hires_fix_params(res):
    """for infotexts that specify old First pass size parameter, convert it into
    width, height, and hr scale"""

    firstpass_width = res.get('First pass size-1', None)
    firstpass_height = res.get('First pass size-2', None)
    # 如果使用旧的高清修复宽度和高度选项
    if shared.opts.use_old_hires_fix_width_height:
        # 从结果中获取高清修复的宽度和高度，转换为整数
        hires_width = int(res.get("Hires resize-1", 0))
        hires_height = int(res.get("Hires resize-2", 0))

        # 如果高清修复的宽度和高度都存在
        if hires_width and hires_height:
            # 更新结果字典中的尺寸信息
            res['Size-1'] = hires_width
            res['Size-2'] = hires_height
            return

    # 如果第一次传递的宽度或高度为 None，则返回
    if firstpass_width is None or firstpass_height is None:
        return

    # 将第一次传递的宽度和高度转换为整数
    firstpass_width, firstpass_height = int(firstpass_width), int(firstpass_height)
    # 获取结果字典中的宽度和高度信息，如果不存在则默认为 512
    width = int(res.get("Size-1", 512))
    height = int(res.get("Size-2", 512))

    # 如果第一次传递的宽度或高度为 0
    if firstpass_width == 0 or firstpass_height == 0:
        # 使用旧的高清修复方法获取第一次传递的宽度和高度
        firstpass_width, firstpass_height = processing.old_hires_fix_first_pass_dimensions(width, height)

    # 更新结果字典中的尺寸信息
    res['Size-1'] = firstpass_width
    res['Size-2'] = firstpass_height
    res['Hires resize-1'] = width
    res['Hires resize-2'] = height
def parse_generation_parameters(x: str):
    """解析生成参数字符串，即在 UI 图片下的文本字段中看到的字符串:
    女孩戴着艺术家的贝雷帽，坚定，蓝眼睛，沙漠场景，电脑显示器，浓妆，由Alphonse Mucha和Charlie Bowater创作，((眼影))，(娇俏)，详细，复杂
    负面提示: 丑陋，胖，肥胖，圆滚滚，(((畸形)))，[模糊]，糟糕的解剖，毁容，画得不好的脸，变异，变异，(额外肢体)，(丑陋)，(画得不好的手)，乱七八糟的画
    步骤: 20，采样器: Euler a，CFG 比例: 7，种子: 965400086，尺寸: 512x512，模型哈希: 45dee52b
    返回一个包含字段值的字典
    """

    res = {}  # 创建一个空字典用于存储结果

    prompt = ""  # 初始化 prompt 字符串
    negative_prompt = ""  # 初始化 negative_prompt 字符串

    done_with_prompt = False  # 初始化 done_with_prompt 标志为 False

    *lines, lastline = x.strip().split("\n")  # 将输入字符串按行分割，最后一行单独处理
    if len(re_param.findall(lastline)) < 3:  # 如果最后一行不包含足够的参数信息
        lines.append(lastline)  # 将最后一行添加到行列表中
        lastline = ''  # 清空最后一行

    for line in lines:  # 遍历每一行
        line = line.strip()  # 去除首尾空格
        if line.startswith("Negative prompt:"):  # 如果行以"Negative prompt:"开头
            done_with_prompt = True  # 设置 done_with_prompt 标志为 True
            line = line[16:].strip()  # 截取行内容
        if done_with_prompt:  # 如果已经处理完 prompt
            negative_prompt += ("" if negative_prompt == "" else "\n") + line  # 将行内容添加到 negative_prompt 中
        else:
            prompt += ("" if prompt == "" else "\n") + line  # 将行内容添加到 prompt 中

    if shared.opts.infotext_styles != "Ignore":  # 如果样式不是"忽略"
        found_styles, prompt, negative_prompt = shared.prompt_styles.extract_styles_from_prompt(prompt, negative_prompt)  # 从 prompt 中提取样式

        if shared.opts.infotext_styles == "Apply":  # 如果样式是"应用"
            res["Styles array"] = found_styles  # 将找到的样式添加到结果字典中
        elif shared.opts.infotext_styles == "Apply if any" and found_styles:  # 如果样式是"如果有任何样式应用"
            res["Styles array"] = found_styles  # 将找到的样式添加到结果字典中

    res["Prompt"] = prompt  # 将 prompt 添加到结果字典中
    res["Negative prompt"] = negative_prompt  # 将 negative_prompt 添加到结果字典中
    # 从最后一行中使用正则表达式匹配参数键值对
    for k, v in re_param.findall(lastline):
        try:
            # 如果值以双引号开头和结尾，则解码
            if v[0] == '"' and v[-1] == '"':
                v = unquote(v)

            # 使用正则表达式匹配图片尺寸信息
            m = re_imagesize.match(v)
            if m is not None:
                # 将图片尺寸信息存入结果字典中
                res[f"{k}-1"] = m.group(1)
                res[f"{k}-2"] = m.group(2)
            else:
                # 否则将键值对存入结果字典中
                res[k] = v
        except Exception:
            # 捕获异常并打印错误信息
            print(f"Error parsing \"{k}: {v}\"")

    # 如果结果字典中缺少"Clip skip"键，则设置默认值为"1"
    if "Clip skip" not in res:
        res["Clip skip"] = "1"

    # 获取"Hypernet"键对应的值
    hypernet = res.get("Hypernet", None)
    if hypernet is not None:
        # 如果存在"Hypernet"键，则将相关信息添加到"Prompt"键中
        res["Prompt"] += f"""<hypernet:{hypernet}:{res.get("Hypernet strength", "1.0")}>"""

    # 如果结果字典中缺少"Hires resize-1"键，则设置默认值为0
    if "Hires resize-1" not in res:
        res["Hires resize-1"] = 0
        res["Hires resize-2"] = 0

    # 如果结果字典中缺少"Hires sampler"键，则设置默认值为"Use same sampler"
    if "Hires sampler" not in res:
        res["Hires sampler"] = "Use same sampler"

    # 如果结果字典中缺少"Hires checkpoint"键，则设置默认值为"Use same checkpoint"
    if "Hires checkpoint" not in res:
        res["Hires checkpoint"] = "Use same checkpoint"

    # 如果结果字典中缺少"Hires prompt"键，则设置默认值为空字符串
    if "Hires prompt" not in res:
        res["Hires prompt"] = ""

    # 如果结果字典中缺少"Hires negative prompt"键，则设置默认值为空字符串
    if "Hires negative prompt" not in res:
        res["Hires negative prompt"] = ""

    # 恢复旧的高分辨率修复参数
    restore_old_hires_fix_params(res)

    # 如果结果字典中缺少"RNG"键，则设置默认值为"GPU"
    if "RNG" not in res:
        res["RNG"] = "GPU"

    # 如果结果字典中缺少"Schedule type"键，则设置默认值为"Automatic"
    if "Schedule type" not in res:
        res["Schedule type"] = "Automatic"

    # 如果结果字典中缺少"Schedule max sigma"键，则设置默认值为0
    if "Schedule max sigma" not in res:
        res["Schedule max sigma"] = 0

    # 如果结果字典中缺少"Schedule min sigma"键，则设置默认值为0
    if "Schedule min sigma" not in res:
        res["Schedule min sigma"] = 0

    # 如果结果字典中缺少"Schedule rho"键，则设置默认值为0
    if "Schedule rho" not in res:
        res["Schedule rho"] = 0

    # 如果结果字典中缺少"VAE Encoder"键，则设置默认值为"Full"
    if "VAE Encoder" not in res:
        res["VAE Encoder"] = "Full"

    # 如果结果字典中缺少"VAE Decoder"键，则设置默认值为"Full"
    if "VAE Decoder" not in res:
        res["VAE Decoder"] = "Full"

    # 创建跳过集合
    skip = set(shared.opts.infotext_skip_pasting)
    # 从结果字典中移除跳过集合中的键值对
    res = {k: v for k, v in res.items() if k not in skip}

    # 返回结果字典
    return res
# infotext标签到设置名称的映射。仅保留以保持向后兼容性 - 使用OptionInfo(..., infotext='...')代替。
# 示例内容：
# infotext_to_setting_name_mapping = [
#     ('Conditional mask weight', 'inpainting_mask_weight'),
#     ('Model hash', 'sd_model_checkpoint'),
#     ('ENSD', 'eta_noise_seed_delta'),
#     ('Schedule type', 'k_sched_type'),
# ]
infotext_to_setting_name_mapping = [

]

def create_override_settings_dict(text_pairs):
    """从gradio的多选框中创建处理的override_settings参数

    示例输入：
        ['Clip skip: 2', 'Model hash: e6e99610c4', 'ENSD: 31337']

    示例输出：
        {'CLIP_stop_at_last_layers': 2, 'sd_model_checkpoint': 'e6e99610c4', 'eta_noise_seed_delta': 31337}
    """
    res = {}

    params = {}
    for pair in text_pairs:
        k, v = pair.split(":", maxsplit=1)

        params[k] = v.strip()

    # 从共享的选项数据标签中获取infotext和设置名称的映射
    mapping = [(info.infotext, k) for k, info in shared.opts.data_labels.items() if info.infotext]
    # 遍历infotext和设置名称的映射以及infotext_to_setting_name_mapping
    for param_name, setting_name in mapping + infotext_to_setting_name_mapping:
        value = params.get(param_name, None)

        if value is None:
            continue

        # 将值转换为设置名称对应的类型，并存储在结果字典中
        res[setting_name] = shared.opts.cast_value(setting_name, value)

    return res


def connect_paste(button, paste_fields, input_comp, override_settings_component, tabname):
    # 定义一个函数，用于处理粘贴操作
    def paste_func(prompt):
        # 如果没有提示信息并且未隐藏 UI 目录配置，则读取参数文件中的内容
        if not prompt and not shared.cmd_opts.hide_ui_dir_config:
            # 拼接参数文件的路径
            filename = os.path.join(data_path, "params.txt")
            # 如果参数文件存在，则读取文件内容
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf8") as file:
                    prompt = file.read()

        # 解析生成参数
        params = parse_generation_parameters(prompt)
        # 调用脚本回调函数，传入提示信息和参数
        script_callbacks.infotext_pasted_callback(prompt, params)
        # 初始化结果列表
        res = []

        # 遍历粘贴字段列表
        for output, key in paste_fields:
            # 如果 key 是可调用的，则调用 key 函数获取值
            if callable(key):
                v = key(params)
            else:
                v = params.get(key, None)

            # 如果值为 None，则添加更新到结果列表
            if v is None:
                res.append(gr.update())
            # 如果值为特定类型，则添加更新到结果列表
            elif isinstance(v, type_of_gr_update):
                res.append(v)
            else:
                try:
                    # 获取输出值的类型
                    valtype = type(output.value)

                    # 如果值类型为布尔型且值为字符串 "False"，则将值设为 False
                    if valtype == bool and v == "False":
                        val = False
                    else:
                        val = valtype(v)

                    # 尝试将值转换为输出值类型，并添加更新到结果列表
                    res.append(gr.update(value=val))
                except Exception:
                    # 发生异常时添加更新到结果列表
                    res.append(gr.update())

        # 返回结果列表
        return res
    # 如果传入了覆盖设置组件，则初始化已处理字段字典，用于跟踪已处理的字段
    if override_settings_component is not None:
        already_handled_fields = {key: 1 for _, key in paste_fields}

        # 定义一个函数，用于将参数粘贴到设置中
        def paste_settings(params):
            vals = {}

            # 构建参数名和设置名的映射关系
            mapping = [(info.infotext, k) for k, info in shared.opts.data_labels.items() if info.infotext]
            # 遍历映射关系和额外的映射关系
            for param_name, setting_name in mapping + infotext_to_setting_name_mapping:
                # 如果参数名已经在已处理字段中，则跳过
                if param_name in already_handled_fields:
                    continue

                # 获取参数值
                v = params.get(param_name, None)
                if v is None:
                    continue

                # 如果设置名为"sd_model_checkpoint"且禁用了权重自动交换，则跳过
                if setting_name == "sd_model_checkpoint" and shared.opts.disable_weights_auto_swap:
                    continue

                # 将参数值转换为设置值
                v = shared.opts.cast_value(setting_name, v)
                # 获取当前设置值
                current_value = getattr(shared.opts, setting_name, None)

                # 如果参数值等于当前设置值，则跳过
                if v == current_value:
                    continue

                # 将参数名和值添加到vals字典中
                vals[param_name] = v

            # 构建参数名和值的字符串对列表
            vals_pairs = [f"{k}: {v}" for k, v in vals.items()]

            # 返回一个下拉菜单更新对象
            return gr.Dropdown.update(value=vals_pairs, choices=vals_pairs, visible=bool(vals_pairs))

        # 将覆盖设置组件和粘贴设置函数添加到粘贴字段列表中
        paste_fields = paste_fields + [(override_settings_component, paste_settings)]

    # 点击按钮执行粘贴函数
    button.click(
        fn=paste_func,
        inputs=[input_comp],
        outputs=[x[0] for x in paste_fields],
        show_progress=False,
    )
    # 点击按钮执行空函数，触发JavaScript重新计算提示
    button.click(
        fn=None,
        _js=f"recalculate_prompts_{tabname}",
        inputs=[],
        outputs=[],
        show_progress=False,
    )
```