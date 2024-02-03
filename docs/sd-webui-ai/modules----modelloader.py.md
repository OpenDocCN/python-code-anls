# `stable-diffusion-webui\modules\modelloader.py`

```py
# 导入必要的模块和库
from __future__ import annotations
import os
import shutil
import importlib
from urllib.parse import urlparse
from modules import shared
from modules.upscaler import Upscaler, UpscalerLanczos, UpscalerNearest, UpscalerNone
from modules.paths import script_path, models_path

# 从指定的URL下载文件到指定的目录中，如果可能的话使用已存在的文件
def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    # 确保目录存在，如果不存在则创建
    os.makedirs(model_dir, exist_ok=True)
    # 如果没有指定文件名，则从URL中提取文件名
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    # 拼接文件路径
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    # 如果文件不存在，则下载文件
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    # 返回下载文件的路径
    return cached_file

# 尝试在指定目录中查找所需的模型文件
def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None, ext_blacklist=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    # 初始化输出列表
    output = []
    # 初始化一个空列表用于存放路径
    try:
        places = []

        # 如果命令路径存在且不等于模型路径，则设置预训练路径为命令路径下的'experiments/pretrained_models'目录
        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(command_path, 'experiments/pretrained_models')
            # 如果预训练路径存在，则将其添加到路径列表中
            if os.path.exists(pretrained_path):
                print(f"Appending path: {pretrained_path}")
                places.append(pretrained_path)
            # 如果预训练路径不存在但命令路径存在，则将命令路径添加到路径列表中
            elif os.path.exists(command_path):
                places.append(command_path)

        # 将模型路径添加到路径列表中
        places.append(model_path)

        # 遍历路径列表中的每个路径
        for place in places:
            # 遍历当前路径下的所有文件，根据扩展名过滤文件
            for full_path in shared.walk_files(place, allowed_extensions=ext_filter):
                # 如果当前文件是一个损坏的符号链接，则跳过
                if os.path.islink(full_path) and not os.path.exists(full_path):
                    print(f"Skipping broken symlink: {full_path}")
                    continue
                # 如果当前文件的扩展名在黑名单中，则跳过
                if ext_blacklist is not None and any(full_path.endswith(x) for x in ext_blacklist):
                    continue
                # 如果当前文件不在输出列表中，则将其添加到输出列表中
                if full_path not in output:
                    output.append(full_path)

        # 如果模型 URL 存在且输出列表为空，则根据下载名称从 URL 下载文件并添加到输出列表中
        if model_url is not None and len(output) == 0:
            if download_name is not None:
                output.append(load_file_from_url(model_url, model_dir=places[0], file_name=download_name))
            else:
                output.append(model_url)

    # 捕获任何异常并忽略
    except Exception:
        pass

    # 返回输出列表
    return output
# 定义一个函数，用于获取友好的文件名
def friendly_name(file: str):
    # 如果文件路径以"http"开头，则提取路径部分
    if file.startswith("http"):
        file = urlparse(file).path

    # 获取文件的基本名称（不包含路径）
    file = os.path.basename(file)
    # 将文件名分割为模型名称和扩展名
    model_name, extension = os.path.splitext(file)
    # 返回模型名称
    return model_name


# 定义一个函数，用于清理模型文件
def cleanup_models():
    # 这段代码可能更有效率，如果我们使用元组列表或其他方式来存储源/目标路径，然后枚举它，但目前这种方式也可以工作。
    # 未来，最好是让每个“模型”缩放器自动注册并执行这些操作...
    # 设置根路径为脚本路径
    root_path = script_path
    # 设置源路径为模型路径
    src_path = models_path
    # 设置目标路径为模型路径下的"Stable-diffusion"目录
    dest_path = os.path.join(models_path, "Stable-diffusion")
    # 移动所有以".ckpt"结尾的文件到目标路径
    move_files(src_path, dest_path, ".ckpt")
    # 移动所有以".safetensors"结尾的文件到目标路径
    move_files(src_path, dest_path, ".safetensors")
    # 设置源路径为根路径下的"ESRGAN"目录
    src_path = os.path.join(root_path, "ESRGAN")
    # 设置目标路径为模型路径下的"ESRGAN"目录
    dest_path = os.path.join(models_path, "ESRGAN")
    # 移动所有文件到目标路径
    move_files(src_path, dest_path)
    # 设置源路径为模型路径下的"BSRGAN"目录
    src_path = os.path.join(models_path, "BSRGAN")
    # 设置目标路径为模型路径下的"ESRGAN"目录
    dest_path = os.path.join(models_path, "ESRGAN")
    # 移动所有以".pth"结尾的文件到目标路径
    move_files(src_path, dest_path, ".pth")
    # 设置源路径为根路径下的"gfpgan"目录
    src_path = os.path.join(root_path, "gfpgan")
    # 设置目标路径为模型路径下的"GFPGAN"目录
    dest_path = os.path.join(models_path, "GFPGAN")
    # 移动所有文件到目标路径
    move_files(src_path, dest_path)
    # 设置源路径为根路径下的"SwinIR"目录
    src_path = os.path.join(root_path, "SwinIR")
    # 设置目标路径为模型路径下的"SwinIR"目录
    dest_path = os.path.join(models_path, "SwinIR")
    # 移动所有文件到目标路径
    move_files(src_path, dest_path)
    # 设置源路径为根路径下的"repositories/latent-diffusion/experiments/pretrained_models/"目录
    src_path = os.path.join(root_path, "repositories/latent-diffusion/experiments/pretrained_models/")
    # 设置目标路径为模型路径下的"LDSR"目录
    dest_path = os.path.join(models_path, "LDSR")
    # 移动所有文件到目标路径
    move_files(src_path, dest_path)


# 定义一个函数，用于移动文件
def move_files(src_path: str, dest_path: str, ext_filter: str = None):
    # 尝试创建目标路径，如果已存在则不报错
    try:
        os.makedirs(dest_path, exist_ok=True)
        # 如果源路径存在
        if os.path.exists(src_path):
            # 遍历源路径下的所有文件
            for file in os.listdir(src_path):
                # 获取文件的完整路径
                fullpath = os.path.join(src_path, file)
                # 如果是文件
                if os.path.isfile(fullpath):
                    # 如果有文件扩展名过滤器，并且文件名不包含该扩展名，则跳过
                    if ext_filter is not None:
                        if ext_filter not in file:
                            continue
                    # 打印移动文件的信息
                    print(f"Moving {file} from {src_path} to {dest_path}.")
                    # 尝试移动文件到目标路径
                    try:
                        shutil.move(fullpath, dest_path)
                    except Exception:
                        pass
            # 如果源路径下没有文件了
            if len(os.listdir(src_path)) == 0:
                # 打印移除空文件夹的信息
                print(f"Removing empty folder: {src_path}")
                # 递归删除空文件夹
                shutil.rmtree(src_path, True)
    except Exception:
        pass
# 加载所有的上采样器类
def load_upscalers():
    # 只有在引用时才能动态加载上采样器，因此我们会尝试在查找 __subclasses__ 之前导入任何 _model.py 文件
    modules_dir = os.path.join(shared.script_path, "modules")
    for file in os.listdir(modules_dir):
        if "_model.py" in file:
            model_name = file.replace("_model.py", "")
            full_model = f"modules.{model_name}_model"
            try:
                importlib.import_module(full_model)
            except Exception:
                pass

    datas = []
    commandline_options = vars(shared.cmd_opts)

    # 一些上采样器类在重新加载模块后不会消失，我们会得到两份这些类的副本。最新的副本总是列表中的最后一个，
    # 所以我们从后往前遍历并忽略重复的类
    used_classes = {}
    for cls in reversed(Upscaler.__subclasses__()):
        classname = str(cls)
        if classname not in used_classes:
            used_classes[classname] = cls

    for cls in reversed(used_classes.values()):
        name = cls.__name__
        cmd_name = f"{name.lower().replace('upscaler', '')}_models_path"
        commandline_model_path = commandline_options.get(cmd_name, None)
        scaler = cls(commandline_model_path)
        scaler.user_path = commandline_model_path
        scaler.model_download_path = commandline_model_path or scaler.model_path
        datas += scaler.scalers

    # 将上采样器数据按名称排序
    shared.sd_upscalers = sorted(
        datas,
        # 对于 UpscalerNone，将其保持在列表的开头
        key=lambda x: x.name.lower() if not isinstance(x.scaler, (UpscalerNone, UpscalerLanczos, UpscalerNearest)) else ""
    )
```