# `stable-diffusion-webui\modules\ui_common.py`

```
# 导入所需的模块
import json
import html
import os
import platform
import sys

# 导入第三方库 gradio 并重命名为 gr
import gradio as gr
# 导入第三方库 subprocess 并重命名为 sp
import subprocess as sp

# 导入自定义模块 call_queue 和 shared
from modules import call_queue, shared
# 导入自定义模块 generation_parameters_copypaste 中的 image_from_url_text 函数
from modules.generation_parameters_copypaste import image_from_url_text
# 导入自定义模块 modules.images
import modules.images
# 导入自定义模块 ui_components 中的 ToolButton 类
from modules.ui_components import ToolButton
# 导入自定义模块 generation_parameters_copypaste 中的 parameters_copypaste 模块
import modules.generation_parameters_copypaste as parameters_copypaste

# 定义常量 folder_symbol 和 refresh_symbol，分别表示文件夹和刷新的 Unicode 符号
folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄

# 更新生成信息的函数，根据传入的生成信息、HTML 信息和图片索引返回更新后的 HTML 信息和 gr 更新
def update_generation_info(generation_info, html_info, img_index):
    try:
        # 尝试将生成信息解析为 JSON 格式
        generation_info = json.loads(generation_info)
        # 如果图片索引超出范围，则返回原始 HTML 信息和 gr 更新
        if img_index < 0 or img_index >= len(generation_info["infotexts"]):
            return html_info, gr.update()
        # 根据图片索引获取对应的信息文本，转换为 HTML 格式并返回更新后的 HTML 信息和 gr 更新
        return plaintext_to_html(generation_info["infotexts"][img_index]), gr.update()
    except Exception:
        pass
    # 如果 JSON 解析或其他操作失败，则返回原始 HTML 信息和 gr 更新
    return html_info, gr.update()

# 将纯文本转换为 HTML 格式的函数，支持指定类名
def plaintext_to_html(text, classname=None):
    # 对文本进行 HTML 转义处理，并以 <br> 分隔每行文本
    content = "<br>\n".join(html.escape(x) for x in text.split('\n'))
    # 根据是否指定类名，返回带有类名或不带类名的 HTML 段落
    return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"

# 保存文件的函数，根据传入的 JSON 数据、图片数据、是否创建 ZIP 文件和索引进行保存
def save_files(js_data, images, do_make_zip, index):
    import csv
    filenames = []
    fullfns = []

    # 快速将字典转换为类对象，用于 apply_filename_pattern 函数的要求
    class MyObject:
        def __init__(self, d=None):
            if d is not None:
                for key, value in d.items():
                    setattr(self, key, value)

    # 解析传入的 JSON 数据
    data = json.loads(js_data)

    # 创建 MyObject 类对象 p，并设置保存路径、是否使用保存到目录、文件扩展名等参数
    p = MyObject(data)
    path = shared.opts.outdir_save
    save_to_dirs = shared.opts.use_save_to_dirs_for_ui
    extension: str = shared.opts.samples_format
    start_index = 0
    only_one = False
    # 检查条件：确保 index 大于 -1，且 save_selected_only 为真，并且 index 大于等于 data["index_of_first_image"]
    if index > -1 and shared.opts.save_selected_only and (index >= data["index_of_first_image"]):  
        # 设置 only_one 为 True
        only_one = True
        # 将 images 中的第 index 个元素作为列表中唯一的元素
        images = [images[index]]
        # 将 index 赋值给 start_index

    # 创建目录 shared.opts.outdir_save，如果目录已存在则不做任何操作
    os.makedirs(shared.opts.outdir_save, exist_ok=True)

    # 打开文件 shared.opts.outdir_save 下的 log.csv 文件，以追加模式写入，编码为 utf8，每行末尾不加换行符
    with open(os.path.join(shared.opts.outdir_save, "log.csv"), "a", encoding="utf8", newline='') as file:
        # 检查文件指针是否在文件开头
        at_start = file.tell() == 0
        # 创建 CSV writer 对象
        writer = csv.writer(file)
        # 如果文件在开头，写入表头
        if at_start:
            writer.writerow(["prompt", "seed", "width", "height", "sampler", "cfgs", "steps", "filename", "negative_prompt"])

        # 遍历 images 列表中的元素及其索引，起始索引为 start_index
        for image_index, filedata in enumerate(images, start_index):
            # 从 filedata 中获取图像数据
            image = image_from_url_text(filedata)

            # 判断是否为网格图像
            is_grid = image_index < p.index_of_first_image
            # 计算 i 的值
            i = 0 if is_grid else (image_index - p.index_of_first_image)

            # 设置 p.batch_index 为 image_index-1
            p.batch_index = image_index-1
            # 保存图像到指定路径，并返回保存的文件名和文本文件名
            fullfn, txt_fullfn = modules.images.save_image(image, path, "", seed=p.all_seeds[i], prompt=p.all_prompts[i], extension=extension, info=p.infotexts[image_index], grid=is_grid, p=p, save_to_dirs=save_to_dirs)

            # 获取相对路径的文件名
            filename = os.path.relpath(fullfn, path)
            # 将文件名添加到列表 filenames 中
            filenames.append(filename)
            # 将完整文件名添加到列表 fullfns 中
            fullfns.append(fullfn)
            # 如果存在文本文件名，则将其添加到列表 filenames 和 fullfns 中
            if txt_fullfn:
                filenames.append(os.path.basename(txt_fullfn))
                fullfns.append(txt_fullfn)

        # 写入一行数据到 CSV 文件中
        writer.writerow([data["prompt"], data["seed"], data["width"], data["height"], data["sampler_name"], data["cfg_scale"], data["steps"], filenames[0], data["negative_prompt"]])

    # 创建 Zip 文件
    # 如果需要创建 ZIP 文件
    if do_make_zip:
        # 根据是否只有一个种子选择使用哪个种子
        zip_fileseed = p.all_seeds[index-1] if only_one else p.all_seeds[0]
        # 创建文件名生成器对象
        namegen = modules.images.FilenameGenerator(p, zip_fileseed, p.all_prompts[0], image, True)
        # 应用文件名生成器生成 ZIP 文件名
        zip_filename = namegen.apply(shared.opts.grid_zip_filename_pattern or "[datetime]_[[model_name]]_[seed]-[seed_last]")
        # 拼接 ZIP 文件路径
        zip_filepath = os.path.join(path, f"{zip_filename}.zip")

        # 导入 ZipFile 类
        from zipfile import ZipFile
        # 创建 ZipFile 对象，以写入模式打开
        with ZipFile(zip_filepath, "w") as zip_file:
            # 遍历文件列表
            for i in range(len(fullfns)):
                # 以二进制只读模式打开文件
                with open(fullfns[i], mode="rb") as f:
                    # 将文件内容写入 ZIP 文件
                    zip_file.writestr(filenames[i], f.read())
        # 将 ZIP 文件路径插入到文件列表的第一个位置
        fullfns.insert(0, zip_filepath)

    # 返回更新后的文件列表和保存成功的提示信息
    return gr.File.update(value=fullfns, visible=True), plaintext_to_html(f"Saved: {filenames[0]}")
# 创建输出面板，用于显示输出结果
def create_output_panel(tabname, outdir, toprow=None):

    # 打开指定文件夹
    def open_folder(f):
        # 如果文件夹不存在，则打印提示信息
        if not os.path.exists(f):
            print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
            return
        # 如果路径不是文件夹，则打印警告信息
        elif not os.path.isdir(f):
            print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
            return

        # 根据操作系统打开文件夹
        if not shared.cmd_opts.hide_ui_dir_config:
            path = os.path.normpath(f)
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                sp.Popen(["open", path])
            elif "microsoft-standard-WSL2" in platform.uname().release:
                sp.Popen(["wsl-open", path])
            else:
                sp.Popen(["xdg-open", path])

# 创建刷新按钮，用于刷新组件
def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    refresh_components = refresh_component if isinstance(refresh_component, list) else [refresh_component]

    label = None
    for comp in refresh_components:
        label = getattr(comp, 'label', None)
        if label is not None:
            break

    # 刷新方法
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        # 更新组件属性
        for k, v in args.items():
            for comp in refresh_components:
                setattr(comp, k, v)

        # 更新组件
        return [gr.update(**(args or {})) for _ in refresh_components] if len(refresh_components) > 1 else gr.update(**(args or {}))

    # 创建刷新按钮
    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id, tooltip=f"{label}: refresh" if label else "Refresh")
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=refresh_components
    )
    return refresh_button

# 设置对话框
def setup_dialog(button_show, dialog, *, button_close=None):
    # 设置 UI，使对话框（gr.Box）不可见，只有在点击 buttons_show 后才会显示在全屏模态窗口中
    dialog.visible = False
    
    # 当点击 button_show 时，执行函数以更新可见性，设置输入和输出，然后执行 JavaScript 函数以显示对话框
    button_show.click(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[dialog],
    ).then(fn=None, _js="function(){ popupId('" + dialog.elem_id + "'); }")
    
    # 如果存在 button_close，则在点击时执行 JavaScript 函数以关闭弹出窗口
    if button_close:
        button_close.click(fn=None, _js="closePopup")
```