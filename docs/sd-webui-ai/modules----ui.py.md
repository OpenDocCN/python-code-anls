# `stable-diffusion-webui\modules\ui.py`

```
# 导入所需的模块
import datetime
import mimetypes
import os
import sys
from functools import reduce
import warnings
from contextlib import ExitStack

import gradio as gr
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin  # noqa: F401
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call

from modules import gradio_extensons  # noqa: F401
from modules import sd_hijack, sd_models, script_callbacks, ui_extensions, deepbooru, extra_networks, ui_common, ui_postprocessing, progress, ui_loadsave, shared_items, ui_settings, timer, sysinfo, ui_checkpoint_merger, scripts, sd_samplers, processing, ui_extra_networks, ui_toprow
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML, InputAccordion, ResizeHandleRow
from modules.paths import script_path
from modules.ui_common import create_refresh_button
from modules.ui_gradio_extensions import reload_javascript

from modules.shared import opts, cmd_opts

import modules.generation_parameters_copypaste as parameters_copypaste
import modules.hypernetworks.ui as hypernetworks_ui
import modules.textual_inversion.ui as textual_inversion_ui
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.shared as shared
from modules import prompt_parser
from modules.sd_hijack import model_hijack
from modules.generation_parameters_copypaste import image_from_url_text

create_setting_component = ui_settings.create_setting_component

# 根据用户设置决定是否显示警告信息
warnings.filterwarnings("default" if opts.show_warnings else "ignore", category=UserWarning)
warnings.filterwarnings("default" if opts.show_gradio_deprecation_warnings else "ignore", category=gr.deprecation.GradioDeprecationWarning)

# 为 Windows 用户修复问题，确保 JavaScript 文件以正确的 content-type 被提供，以便浏览器正确显示 UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# 同样，为某些缺失的图片类型添加显式的 content-type 头部
# 添加 MIME 类型映射，将 'image/webp' 映射到 '.webp' 文件扩展名
mimetypes.add_type('image/webp', '.webp')

# 如果未设置分享和监听选项，则执行以下操作
if not cmd_opts.share and not cmd_opts.listen:
    # 修复 Gradio 的远程连接
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

# 如果设置了 ngrok 选项，则执行以下操作
if cmd_opts.ngrok is not None:
    # 导入 ngrok 模块
    import modules.ngrok as ngrok
    # 打印提示信息
    print('ngrok authtoken detected, trying to connect...')
    # 连接 ngrok
    ngrok.connect(
        cmd_opts.ngrok,
        cmd_opts.port if cmd_opts.port is not None else 7860,
        cmd_opts.ngrok_options
        )

# 定义函数 gr_show，用于控制显示
def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

# 设置示例图片路径
sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

# 定义一些常量，用于显示特定符号
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅
restore_progress_symbol = '\U0001F300' # 🌀
detect_image_size_symbol = '\U0001F4D0'  # 📐

# 将纯文本转换为 HTML
plaintext_to_html = ui_common.plaintext_to_html

# 将 Gradio 图库发送到图像
def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return image_from_url_text(x[0])

# 计算高分辨率分辨率
def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    if not enable:
        return ""

    # 创建 StableDiffusionProcessingTxt2Img 对象
    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    # 计算目标分辨率
    p.calculate_target_resolution()

    # 返回分辨率计算结果
    return f"from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"
# 根据指定的宽度、高度和缩放比例计算目标宽度和高度
def resize_from_to_html(width, height, scale_by):
    # 计算目标宽度
    target_width = int(width * scale_by)
    # 计算目标高度
    target_height = int(height * scale_by)

    # 如果目标宽度或目标高度为0，则返回提示信息
    if not target_width or not target_height:
        return "no image selected"

    # 返回调整大小的信息
    return f"resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"


# 处理询问函数的调用
def process_interrogate(interrogation_function, mode, ii_input_dir, ii_output_dir, *ii_singles):
    # 根据不同的模式选择不同的处理方式
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    elif mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    elif mode == 5:
        # 检查是否启用了隐藏 UI 目录配置选项
        assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
        # 获取输入目录下的所有图片文件
        images = shared.listfiles(ii_input_dir)
        print(f"Will process {len(images)} images.")
        # 如果输出目录不为空，则创建输出目录
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir

        # 遍历处理每张图片
        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            # 打印处理结果到输出文件中
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, f"{left}.txt"), 'a', encoding='utf-8'))

        return [gr.update(), None]


# 对图片进行询问处理
def interrogate(image):
    # 对图片进行 RGB 转换后进行询问
    prompt = shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


# 对图片进行深度标记处理
def interrogate_deepbooru(image):
    # 使用 deepbooru 模型对图片进行标记
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


# 设置清除提示按钮的点击事件
def connect_clear_prompt(button):
    button.click(
        _js="clear_prompt",
        fn=None,
        inputs=[],
        outputs=[],
    )


# 更新令牌计数器
def update_token_counter(text, steps, *, is_positive=True):
    try:
        # 尝试解析输入的文本，获取解析后的文本和额外网络信息
        text, _ = extra_networks.parse_prompt(text)

        # 如果是正面情况
        if is_positive:
            # 获取多条件提示列表
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        else:
            # 否则，将文本添加到单条件提示列表中
            prompt_flat_list = [text]

        # 获取学习条件提示的时间表
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # 捕获异常，可能是在输入时发生解析错误，不想在控制台中打印相关消息
        prompt_schedules = [[[steps, text]]]

    # 将多个时间表合并成一个平面列表
    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    # 从平面列表中提取提示文本
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    # 获取提示文本中的令牌数量和最大长度
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    # 返回包含令牌数量和最大长度的 HTML 字符串
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"
# 更新负面提示令牌计数器
def update_negative_prompt_token_counter(text, steps):
    # 调用 update_token_counter 函数，传入文本、步骤和 is_positive 参数为 False
    return update_token_counter(text, steps, is_positive=False)


# 设置进度条
def setup_progressbar(*args, **kwargs):
    # 空函数，不执行任何操作
    pass


# 应用设置
def apply_setting(key, value):
    # 如果值为 None，则返回 gr.update()
    if value is None:
        return gr.update()

    # 如果 shared.cmd_opts.freeze_settings 为真，则返回 gr.update()
    if shared.cmd_opts.freeze_settings:
        return gr.update()

    # 如果 key 为 "sd_model_checkpoint" 并且 opts.disable_weights_auto_swap 为真，则返回 gr.update()
    if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
        return gr.update()

    # 如果 key 为 "sd_model_checkpoint"
    if key == "sd_model_checkpoint":
        # 获取与 value 最接近的检查点信息
        ckpt_info = sd_models.get_closet_checkpoint_match(value)

        # 如果存在最接近的检查点信息
        if ckpt_info is not None:
            # 将 value 更新为检查点信息的标题
            value = ckpt_info.title
        else:
            return gr.update()

    # 获取组件参数
    comp_args = opts.data_labels[key].component_args

    # 如果 comp_args 存在且为字典，并且 visible 为 False
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return

    # 获取值类型
    valtype = type(opts.data_labels[key].default)
    # 获取旧值
    oldval = opts.data.get(key, None)
    # 更新 opts.data[key] 为 value 的值
    opts.data[key] = valtype(value) if valtype != type(None) else value
    # 如果旧值不等于 value 并且 opts.data_labels[key].onchange 不为 None
    if oldval != value and opts.data_labels[key].onchange is not None:
        # 调用 opts.data_labels[key].onchange()
        opts.data_labels[key].onchange()

    # 保存设置到配置文件
    opts.save(shared.config_filename)
    # 返回 opts 中 key 对应的值
    return getattr(opts, key)


# 创建输出面板
def create_output_panel(tabname, outdir, toprow=None):
    # 调用 ui_common.create_output_panel 函数，传入标签名、输出目录和顶部行
    return ui_common.create_output_panel(tabname, outdir, toprow)


# 创建采样器和步骤选择
def create_sampler_and_steps_selection(choices, tabname):
    # 如果 opts.samplers_in_dropdown 为真
    if opts.samplers_in_dropdown:
        # 在表单行中创建下拉框和滑块
        with FormRow(elem_id=f"sampler_selection_{tabname}"):
            sampler_name = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=choices, value=choices[0])
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
    # 如果条件不满足，则执行以下代码块
    else:
        # 创建一个表单组，设置元素ID为"sampler_selection_{tabname}"
        with FormGroup(elem_id=f"sampler_selection_{tabname}"):
            # 创建一个滑块，设置最小值为1，最大值为150，步长为1，元素ID为"{tabname}_steps"，标签为"Sampling steps"，初始值为20
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
            # 创建一个单选框，设置标签为"Sampling method"，元素ID为"{tabname}_sampling"，选项为choices列表，初始值为choices列表的第一个元素
            sampler_name = gr.Radio(label='Sampling method', elem_id=f"{tabname}_sampling", choices=choices, value=choices[0])

    # 返回步数和采样方法
    return steps, sampler_name
# 生成按用户指定顺序排列的 UI 分类
def ordered_ui_categories():
    # 根据用户指定的顺序生成一个字典，键为分类名称，值为对应的顺序值
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder_list)}

    # 遍历已排序的 UI 分类列表，根据用户指定的顺序值或默认顺序值排序
    for _, category in sorted(enumerate(shared_items.ui_reorder_categories()), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


# 创建一个用于覆盖设置的下拉菜单
def create_override_settings_dropdown(tabname, row):
    # 创建一个下拉菜单对象，设置标签和可见性
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)

    # 当下拉菜单改变时执行指定函数，更新可见性
    dropdown.change(
        fn=lambda x: gr.Dropdown.update(visible=bool(x)),
        inputs=[dropdown],
        outputs=[dropdown],
    )

    return dropdown


# 创建用户界面
def create_ui():
    # 导入必要的模块
    import modules.img2img
    import modules.txt2img

    # 重新加载 JavaScript
    reload_javascript()

    # 重置参数复制粘贴
    parameters_copypaste.reset()

    # 设置当前脚本为文本转图片脚本，并初始化
    scripts.scripts_current = scripts.scripts_txt2img
    scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

    # 设置当前脚本为图片转图片脚本，并初始化
    scripts.scripts_current = scripts.scripts_img2img
    scripts.scripts_img2img.initialize_scripts(is_img2img=True)

    # 重置当前脚本
    scripts.scripts_current = None

    # 创建用户界面，关闭分析功能
    with gr.Blocks(analytics_enabled=False) as extras_interface:
        ui_postprocessing.create_ui()
    # 创建一个不启用分析功能的 Blocks 对象，并将其赋值给 pnginfo_interface
    with gr.Blocks(analytics_enabled=False) as pnginfo_interface:
        # 创建一个行布局，子元素高度不相等
        with gr.Row(equal_height=False):
            # 创建一个列布局，样式为 panel
            with gr.Column(variant='panel'):
                # 创建一个图片组件，可交互，类型为 PIL
                image = gr.Image(elem_id="pnginfo_image", label="Source", source="upload", interactive=True, type="pil")
    
            # 创建一个列布局，样式为 panel
            with gr.Column(variant='panel'):
                # 创建一个 HTML 组件
                html = gr.HTML()
                # 创建一个文本框组件，不可见，元素 ID 为 pnginfo_generation_info
                generation_info = gr.Textbox(visible=False, elem_id="pnginfo_generation_info")
                # 创建一个 HTML 组件
                html2 = gr.HTML()
                # 创建一个行布局
                with gr.Row():
                    # 使用 parameters_copypaste 模块的 create_buttons 方法创建按钮
                    buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
    
                # 遍历按钮字典
                for tabname, button in buttons.items():
                    # 注册粘贴参数按钮
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname, source_text_component=generation_info, source_image_component=image,
                    ))
    
        # 图片组件改变时执行 wrap_gradio_call(modules.extras.run_pnginfo) 函数
        image.change(
            fn=wrap_gradio_call(modules.extras.run_pnginfo),
            inputs=[image],
            outputs=[html, generation_info, html2],
        )
    
    # 创建一个 UiCheckpointMerger 对象
    modelmerger_ui = ui_checkpoint_merger.UiCheckpointMerger()
    
    # 创建一个 UiLoadsave 对象，传入参数 cmd_opts.ui_config_file
    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config_file)
    
    # 创建一个 UiSettings 对象
    settings = ui_settings.UiSettings()
    # 创建设置界面，传入参数 loadsave 和 dummy_component
    settings.create_ui(loadsave, dummy_component)
    
    # 创建界面列表
    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (img2img_interface, "img2img", "img2img"),
        (extras_interface, "Extras", "extras"),
        (pnginfo_interface, "PNG Info", "pnginfo"),
        (modelmerger_ui.blocks, "Checkpoint Merger", "modelmerger"),
        (train_interface, "Train", "train"),
    ]
    
    # 添加脚本回调的选项卡
    interfaces += script_callbacks.ui_tabs_callback()
    # 添加设置界面
    interfaces += [(settings.interface, "Settings", "settings")]
    
    # 创建扩展界面
    extensions_interface = ui_extensions.create_ui()
    # 添加扩展界面
    interfaces += [(extensions_interface, "Extensions", "extensions")]
    
    # 将共享的选项卡名称列表清空
    shared.tab_names = []
    # 遍历interfaces列表中的每个元素，元素包含_interface, label, _ifid三个值
    for _interface, label, _ifid in interfaces:
        # 将label值添加到shared.tab_names列表中
        shared.tab_names.append(label)
    # 创建一个包含 Gradio 主题和禁用分析的 Blocks 对象，标题为"Stable Diffusion"
    with gr.Blocks(theme=shared.gradio_theme, analytics_enabled=False, title="Stable Diffusion") as demo:
        # 添加快速设置
        settings.add_quicksettings()

        # 连接粘贴参数按钮
        parameters_copypaste.connect_paste_params_buttons()

        # 创建 Tabs 对象
        with gr.Tabs(elem_id="tabs") as tabs:
            # 根据 UI 标签顺序创建标签页
            tab_order = {k: i for i, k in enumerate(opts.ui_tab_order)}
            sorted_interfaces = sorted(interfaces, key=lambda x: tab_order.get(x[1], 9999))

            # 遍历排序后的接口
            for interface, label, ifid in sorted_interfaces:
                # 如果标签在隐藏标签列表中，则跳过
                if label in shared.opts.hidden_tabs:
                    continue
                # 创建标签页
                with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                    interface.render()

                # 如果标签页不是"extensions"或"settings"，则添加块
                if ifid not in ["extensions", "settings"]:
                    loadsave.add_block(interface, ifid)

            # 添加组件
            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)

            # 设置 UI
            loadsave.setup_ui()

        # 如果存在通知音频文件并且启用通知音频，则添加音频组件
        if os.path.exists(os.path.join(script_path, "notification.mp3")) and shared.opts.notification_audio:
            gr.Audio(interactive=False, value=os.path.join(script_path, "notification.mp3"), elem_id="audio_notification", visible=False)

        # 加载页脚 HTML
        footer = shared.html("footer.html")
        footer = footer.format(versions=versions_html(), api_docs="/docs" if shared.cmd_opts.api else "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API")
        gr.HTML(footer, elem_id="footer")

        # 添加功能
        settings.add_functionality(demo)

        # 更新图像配置比例可见性
        update_image_cfg_scale_visibility = lambda: gr.update(visible=shared.sd_model and shared.sd_model.cond_stage_key == "edit")
        settings.text_settings.change(fn=update_image_cfg_scale_visibility, inputs=[], outputs=[image_cfg_scale])
        demo.load(fn=update_image_cfg_scale_visibility, inputs=[], outputs=[image_cfg_scale])

        # 设置模型合并器 UI
        modelmerger_ui.setup_ui(dummy_component=dummy_component, sd_model_checkpoint_component=settings.component_dict['sd_model_checkpoint'])

    # 导出默认设置
    loadsave.dump_defaults()
    # 将 loadsave 对象赋值给 demo 的 ui_loadsave 属性
    demo.ui_loadsave = loadsave
    # 返回变量 demo 的值
    return demo
# 定义一个函数，生成包含各种版本信息的 HTML 内容
def versions_html():
    # 导入必要的库
    import torch
    import launch

    # 获取 Python 版本信息
    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    # 获取提交哈希值
    commit = launch.commit_hash()
    # 获取 Git 标签
    tag = launch.git_tag()

    # 如果 xformers 可用
    if shared.xformers_available:
        # 导入 xformers 库
        import xformers
        # 获取 xformers 版本信息
        xformers_version = xformers.__version__
    else:
        # 否则设置 xformers 版本为 N/A
        xformers_version = "N/A"

    # 返回包含版本信息的 HTML 内容
    return f"""
version: <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/{commit}">{tag}</a>
&#x2000;•&#x2000;
python: <span title="{sys.version}">{python_version}</span>
&#x2000;•&#x2000;
torch: {getattr(torch, '__long_version__',torch.__version__)}
&#x2000;•&#x2000;
xformers: {xformers_version}
&#x2000;•&#x2000;
gradio: {gr.__version__}
&#x2000;•&#x2000;
checkpoint: <a id="sd_checkpoint_hash">N/A</a>
"""

# 设置 UI API
def setup_ui_api(app):
    # 导入必要的库
    from pydantic import BaseModel, Field

    # 定义 QuicksettingsHint 类
    class QuicksettingsHint(BaseModel):
        name: str = Field(title="Name of the quicksettings field")
        label: str = Field(title="Label of the quicksettings field")

    # 定义函数返回 QuicksettingsHint 列表
    def quicksettings_hint():
        return [QuicksettingsHint(name=k, label=v.label) for k, v in opts.data_labels.items()]

    # 添加 API 路由，返回 QuicksettingsHint 列表
    app.add_api_route("/internal/quicksettings-hint", quicksettings_hint, methods=["GET"], response_model=list[QuicksettingsHint])

    # 添加 API 路由，返回空字典
    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])

    # 添加 API 路由，返回启动记录
    app.add_api_route("/internal/profile-startup", lambda: timer.startup_record, methods=["GET"])

    # 定义下载系统信息函数
    def download_sysinfo(attachment=False):
        # 导入必要的库
        from fastapi.responses import PlainTextResponse

        # 获取系统信息
        text = sysinfo.get()
        # 设置文件名
        filename = f"sysinfo-{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.json"

        # 返回纯文本响应
        return PlainTextResponse(text, headers={'Content-Disposition': f'{"attachment" if attachment else "inline"}; filename="{filename}"'})

    # 添加 API 路由，下载系统信息
    app.add_api_route("/internal/sysinfo", download_sysinfo, methods=["GET"])
    # 为应用程序添加一个 API 路由，当接收到 GET 请求时，调用 download_sysinfo 函数并设置附件为 True
    app.add_api_route("/internal/sysinfo-download", lambda: download_sysinfo(attachment=True), methods=["GET"])
```