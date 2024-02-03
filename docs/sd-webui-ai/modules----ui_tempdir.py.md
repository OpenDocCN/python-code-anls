# `stable-diffusion-webui\modules\ui_tempdir.py`

```
# 导入必要的库
import os
import tempfile
from collections import namedtuple
from pathlib import Path

import gradio.components

from PIL import PngImagePlugin

from modules import shared

# 定义一个命名元组 Savedfile，包含一个字段 name
Savedfile = namedtuple("Savedfile", ["name"])

# 注册临时文件，将文件名添加到临时文件集合或临时目录集合中
def register_tmp_file(gradio, filename):
    # 如果 gradio 对象有 temp_file_sets 属性
    if hasattr(gradio, 'temp_file_sets'):  # gradio 3.15
        # 将文件名添加到第一个临时文件集合中
        gradio.temp_file_sets[0] = gradio.temp_file_sets[0] | {os.path.abspath(filename)}

    # 如果 gradio 对象有 temp_dirs 属性
    if hasattr(gradio, 'temp_dirs'):  # gradio 3.9
        # 将文件所在目录添加到临时目录集合中
        gradio.temp_dirs = gradio.temp_dirs | {os.path.abspath(os.path.dirname(filename))}

# 检查临时文件是否存在于临时文件集合或临时目录集合中
def check_tmp_file(gradio, filename):
    # 如果 gradio 对象有 temp_file_sets 属性
    if hasattr(gradio, 'temp_file_sets'):
        # 检查文件名是否存在于任意一个临时文件集合中
        return any(filename in fileset for fileset in gradio.temp_file_sets)

    # 如果 gradio 对象有 temp_dirs 属性
    if hasattr(gradio, 'temp_dirs'):
        # 检查文件所在目录是否存在于任意一个临时目录的父目录中
        return any(Path(temp_dir).resolve() in Path(filename).resolve().parents for temp_dir in gradio.temp_dirs)

    return False

# 将 PIL 图像保存到文件
def save_pil_to_file(self, pil_image, dir=None, format="png"):
    # 获取已保存的文件名
    already_saved_as = getattr(pil_image, 'already_saved_as', None)
    # 如果已保存的文件名存在且是一个文件
    if already_saved_as and os.path.isfile(already_saved_as):
        # 注册临时文件
        register_tmp_file(shared.demo, already_saved_as)
        filename = already_saved_as

        # 如果不需要为文件名添加时间戳
        if not shared.opts.save_images_add_number:
            filename += f'?{os.path.getmtime(already_saved_as)}'

        return filename

    # 如果指定了临时目录
    if shared.opts.temp_dir != "":
        dir = shared.opts.temp_dir
    else:
        # 否则创建目录
        os.makedirs(dir, exist_ok=True)

    # 是否使用元数据
    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    # 遍历图像信息，添加到元数据中
    for key, value in pil_image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True

    # 创建临时文件对象
    file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=dir)
    # 保存图像到文件
    pil_image.save(file_obj, pnginfo=(metadata if use_metadata else None))
    return file_obj.name

# 覆盖保存到文件函数，使其也写入 PNG 信息
def install_ui_tempdir_override():
    """override save to file function so that it also writes PNG info"""
    # 将 gradio.components.IOComponent 类中的 pil_to_temp_file 方法设置为 save_pil_to_file 方法
    gradio.components.IOComponent.pil_to_temp_file = save_pil_to_file
# 当临时目录改变时执行的函数
def on_tmpdir_changed():
    # 如果临时目录为空或者共享的演示对象为空，则返回
    if shared.opts.temp_dir == "" or shared.demo is None:
        return

    # 创建临时目录，如果目录已存在则不报错
    os.makedirs(shared.opts.temp_dir, exist_ok=True)

    # 注册临时文件，将演示对象保存到临时目录下的文件"x"中
    register_tmp_file(shared.demo, os.path.join(shared.opts.temp_dir, "x"))


# 清理临时目录的函数
def cleanup_tmpdr():
    # 获取临时目录路径
    temp_dir = shared.opts.temp_dir
    # 如果临时目录为空或者临时目录不存在，则返回
    if temp_dir == "" or not os.path.isdir(temp_dir):
        return

    # 遍历临时目录下的所有文件
    for root, _, files in os.walk(temp_dir, topdown=False):
        for name in files:
            # 获取文件名和扩展名
            _, extension = os.path.splitext(name)
            # 如果文件不是".png"格式，则跳过
            if extension != ".png":
                continue

            # 构建文件的完整路径
            filename = os.path.join(root, name)
            # 删除文件
            os.remove(filename)
```