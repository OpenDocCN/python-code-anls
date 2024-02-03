# `stable-diffusion-webui\modules\sd_models.py`

```
# 导入必要的库
import collections
import os.path
import sys
import threading

import torch
import re
import safetensors.torch
from omegaconf import OmegaConf, ListConfig
from os import mkdir
from urllib import request
import ldm.modules.midas as midas

# 从自定义模块中导入必要的函数和类
from ldm.util import instantiate_from_config

from modules import paths, shared, modelloader, devices, script_callbacks, sd_vae, sd_disable_initialization, errors, hashes, sd_models_config, sd_unet, sd_models_xl, cache, extra_networks, processing, lowvram, sd_hijack, patches
from modules.timer import Timer
import tomesd
import numpy as np

# 定义模型目录和模型路径
model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))

# 初始化存储检查点信息的字典和别名字典
checkpoints_list = {}
checkpoint_aliases = {}
checkpoint_alisases = checkpoint_aliases  # 为了与旧名称兼容
checkpoints_loaded = collections.OrderedDict()

# 定义一个函数，用于替换字典中的键值对
def replace_key(d, key, new_key, value):
    keys = list(d.keys())

    # 添加新的键值对
    d[new_key] = value

    # 如果原键不存在，则直接返回
    if key not in keys:
        return d

    # 获取原键的索引，并替换为新键
    index = keys.index(key)
    keys[index] = new_key

    # 根据新的键顺序创建新的字典
    new_d = {k: d[k] for k in keys}

    # 清空原字典并更新为新字典
    d.clear()
    d.update(new_d)
    return d

# 定义一个类，用于注册检查点信息
class CheckpointInfo:
    def register(self):
        # 将检查点信息添加到检查点列表和别名字典中
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_aliases[id] = self
    # 计算文件的短哈希值
    def calculate_shorthash(self):
        # 使用文件名和路径计算 SHA256 哈希值
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        # 如果 SHA256 值为空，则返回
        if self.sha256 is None:
            return

        # 提取 SHA256 值的前10位作为短哈希值
        shorthash = self.sha256[0:10]
        # 如果当前短哈希值与 SHA256 值的前10位相同，则返回当前短哈希值
        if self.shorthash == self.sha256[0:10]:
            return self.shorthash

        # 更新对象的短哈希值
        self.shorthash = shorthash

        # 如果短哈希值不在对象的 ids 列表中，则将其添加到 ids 列表中
        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]', f'{self.name_for_extra} [{self.shorthash}]']

        # 保存旧标题，更新对象的标题和短标题
        old_title = self.title
        self.title = f'{self.name} [{self.shorthash}]'
        self.short_title = f'{self.name_for_extra} [{self.shorthash}]'

        # 在检查点列表中替换旧标题为新标题
        replace_key(checkpoints_list, old_title, self.title, self)
        # 注册对象
        self.register()

        # 返回短哈希值
        return self.shorthash
# 尝试导入 transformers 模块中的 logging 和 CLIPModel 类，同时禁止 F401 错误提示
try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging, CLIPModel  # noqa: F401

    # 设置日志级别为 error，避免输出冗长的日志信息
    logging.set_verbosity_error()
except Exception:
    pass


# 用于在启动时执行与 SD 模型相关的各种一次性任务
def setup_model():
    """called once at startup to do various one-time tasks related to SD models"""

    # 创建模型路径
    os.makedirs(model_path, exist_ok=True)

    # 启用 midas 自动下载
    enable_midas_autodownload()
    # 修补给定的贝塔值
    patch_given_betas()


# 返回检查点标题列表，根据 use_short 参数决定是否使用短标题
def checkpoint_tiles(use_short=False):
    return [x.short_title if use_short else x.title for x in checkpoints_list.values()]


# 列出模型
def list_models():
    # 清空检查点列表和检查点别名
    checkpoints_list.clear()
    checkpoint_aliases.clear()

    # 获取命令行参数中的检查点路径
    cmd_ckpt = shared.cmd_opts.ckpt
    # 如果禁止下载 SD 模型或者检查点路径已存在，则不需要下载模型
    if shared.cmd_opts.no_download_sd_model or cmd_ckpt != shared.sd_model_file or os.path.exists(cmd_ckpt):
        model_url = None
    else:
        # 设置模型下载链接
        model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"

    # 加载模型列表
    model_list = modelloader.load_models(model_path=model_path, model_url=model_url, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="v1-5-pruned-emaonly.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"])

    # 如果检查点路径存在，则注册检查点信息
    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()

        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    # 如果命令行参数中指定的检查点路径不存在，则输出错误信息
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}", file=sys.stderr)

    # 遍历模型列表，注册检查点信息
    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


# 用于匹配最接近的检查点
re_strip_checksum = re.compile(r"\s*\[[^]]+]\s*$")


def get_closet_checkpoint_match(search_string):
    if not search_string:
        return None

    # 获取检查点别名对应的检查点信息
    checkpoint_info = checkpoint_aliases.get(search_string, None)
    # 如果检查点信息不为空，则直接返回该信息
    if checkpoint_info is not None:
        return checkpoint_info

    # 在检查点列表中查找包含搜索字符串的检查点信息，并按标题长度排序
    found = sorted([info for info in checkpoints_list.values() if search_string in info.title], key=lambda x: len(x.title))
    # 如果找到匹配的检查点信息，则返回第一个找到的信息
    if found:
        return found[0]

    # 去除搜索字符串中的校验和信息，再在检查点列表中查找包含修改后搜索字符串的检查点信息，并按标题长度排序
    search_string_without_checksum = re.sub(re_strip_checksum, '', search_string)
    found = sorted([info for info in checkpoints_list.values() if search_string_without_checksum in info.title], key=lambda x: len(x.title))
    # 如果找到匹配的检查点信息，则返回第一个找到的信息
    if found:
        return found[0]

    # 如果没有找到匹配的检查点信息，则返回空
    return None
# 计算文件的哈希值，但只考虑文件的一小部分，容易发生碰撞
def model_hash(filename):
    try:
        # 以二进制只读方式打开文件
        with open(filename, "rb") as file:
            # 导入 hashlib 模块
            import hashlib
            # 创建 SHA-256 哈希对象
            m = hashlib.sha256()

            # 将文件指针移动到十六进制位置 0x100000
            file.seek(0x100000)
            # 读取十六进制大小的数据并更新哈希对象
            m.update(file.read(0x10000))
            # 返回哈希值的前八位
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


# 选择检查点，如果找不到检查点则引发 `FileNotFoundError`
def select_checkpoint():
    model_checkpoint = shared.opts.sd_model_checkpoint

    # 获取检查点信息
    checkpoint_info = checkpoint_aliases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    # 如果检查点列表为空，则引发错误
    if len(checkpoints_list) == 0:
        error_message = "No checkpoints found. When searching for checkpoints, looked at:"
        if shared.cmd_opts.ckpt is not None:
            error_message += f"\n - file {os.path.abspath(shared.cmd_opts.ckpt)}"
        error_message += f"\n - directory {model_path}"
        if shared.cmd_opts.ckpt_dir is not None:
            error_message += f"\n - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}"
        error_message += "Can't run without a checkpoint. Find and place a .ckpt or .safetensors file into any of those locations."
        raise FileNotFoundError(error_message)

    # 获取第一个检查点信息
    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


# 替换检查点字典中的键值对
checkpoint_dict_replacements_sd1 = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

# 转换 SD 2.1 Turbo 从 SGM 到 LDM 格式的检查点字典替换
checkpoint_dict_replacements_sd2_turbo = {
    # 将字符串'conditioner.embedders.0.'作为键，'cond_stage_model.'作为值，添加到字典中
    'conditioner.embedders.0.': 'cond_stage_model.',
# 结束当前函数定义
}


# 将给定键转换为替换后的键
def transform_checkpoint_dict_key(k, replacements):
    # 遍历替换字典中的文本和替换值
    for text, replacement in replacements.items():
        # 如果键以指定文本开头，则替换为新键
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


# 从检查点中获取状态字典
def get_state_dict_from_checkpoint(pl_sd):
    # 从字典中弹出键为"state_dict"的值，如果不存在则返回原字典
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    # 从字典中弹出键为"state_dict"的值，如果不存在则返回None
    pl_sd.pop("state_dict", None)

    # 检查是否为 sd2 turbo 模型
    is_sd2_turbo = 'conditioner.embedders.0.model.ln_final.weight' in pl_sd and pl_sd['conditioner.embedders.0.model.ln_final.weight'].size()[0] == 1024

    # 创建空字典 sd
    sd = {}
    # 遍历原字典中的键值对
    for k, v in pl_sd.items():
        # 根据模型类型选择替换规则
        if is_sd2_turbo:
            new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd2_turbo)
        else:
            new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd1)

        # 如果新键不为空，则将新键和值添加到 sd 字典中
        if new_key is not None:
            sd[new_key] = v

    # 清空原字典并更新为 sd 字典
    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


# 从 SafeTensors 文件中读取元数据
def read_metadata_from_safetensors(filename):
    import json

    # 以二进制模式打开文件
    with open(filename, mode="rb") as file:
        # 读取元数据长度
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        # 读取 JSON 数据起始标记
        json_start = file.read(2)

        # 断言元数据长度大于2且 JSON 起始标记正确，否则抛出异常
        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        # 读取 JSON 数据
        json_data = json_start + file.read(metadata_len-2)
        # 解析 JSON 数据
        json_obj = json.loads(json_data)

        # 创建空字典 res
        res = {}
        # 遍历 JSON 数据中的元数据
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            # 如果值为字符串且以 '{' 开头，则尝试解析为 JSON 对象
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception:
                    pass

        return res


# 从检查点文件中读取状态字典
def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    # 如果文件扩展名为.safetensors
    if extension.lower() == ".safetensors":
        # 确定设备位置，优先使用map_location，其次使用shared.weight_load_location，最后使用devices.get_optimal_device_name()
        device = map_location or shared.weight_load_location or devices.get_optimal_device_name()

        # 如果不禁用mmap加载safetensors
        if not shared.opts.disable_mmap_load_safetensors:
            # 使用safetensors.torch.load_file加载checkpoint_file到指定设备
            pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
        else:
            # 以二进制方式打开checkpoint_file并加载数据
            pl_sd = safetensors.torch.load(open(checkpoint_file, 'rb').read())
            # 将加载的数据转移到指定设备
            pl_sd = {k: v.to(device) for k, v in pl_sd.items()}
    else:
        # 使用torch.load加载checkpoint_file到指定设备
        pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)

    # 如果需要打印全局状态并且pl_sd中包含"global_step"键
    if print_global_state and "global_step" in pl_sd:
        # 打印全局步数
        print(f"Global Step: {pl_sd['global_step']}")

    # 从pl_sd中获取状态字典
    sd = get_state_dict_from_checkpoint(pl_sd)
    # 返回状态字典
    return sd
# 获取检查点状态字典，根据检查点信息和计时器
def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):
    # 计算模型哈希值
    sd_model_hash = checkpoint_info.calculate_shorthash()
    # 记录计时器
    timer.record("calculate hash")

    # 如果检查点信息在已加载的检查点中
    if checkpoint_info in checkpoints_loaded:
        # 使用检查点缓存
        print(f"Loading weights [{sd_model_hash}] from cache")
        # 将该检查点移到最后，作为最新的
        checkpoints_loaded.move_to_end(checkpoint_info)
        return checkpoints_loaded[checkpoint_info]

    # 打印加载权重信息
    print(f"Loading weights [{sd_model_hash}] from {checkpoint_info.filename}")
    # 从磁盘中读取状态字典
    res = read_state_dict(checkpoint_info.filename)
    # 记录加载权重的时间
    timer.record("load weights from disk")

    return res

# 跳过写入配置的上下文管理器
class SkipWritingToConfig:
    """This context manager prevents load_model_weights from writing checkpoint name to the config when it loads weight."""

    skip = False
    previous = None

    # 进入上下文管理器
    def __enter__(self):
        self.previous = SkipWritingToConfig.skip
        SkipWritingToConfig.skip = True
        return self

    # 退出上下文管理器
    def __exit__(self, exc_type, exc_value, exc_traceback):
        SkipWritingToConfig.skip = self.previous

# 加载模型权重
def load_model_weights(model, checkpoint_info: CheckpointInfo, state_dict, timer):
    # 计算模型哈希值
    sd_model_hash = checkpoint_info.calculate_shorthash()
    # 记录计时器
    timer.record("calculate hash")

    # 如果不跳过写入配置
    if not SkipWritingToConfig.skip:
        shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title

    # 如果状态字典为空
    if state_dict is None:
        # 获取检查点状态字典
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    # 判断模型类型
    model.is_sdxl = hasattr(model, 'conditioner')
    model.is_sd2 = not model.is_sdxl and hasattr(model.cond_stage_model, 'model')
    model.is_sd1 = not model.is_sdxl and not model.is_sd2
    model.is_ssd = model.is_sdxl and 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight' not in state_dict.keys()
    
    # 如果是 sdxl 模型
    if model.is_sdxl:
        # 扩展 sdxl 模型
        sd_models_xl.extend_sdxl(model)

    # 如果是 ssd 模型
    if model.is_ssd:
        # 转换 sdxl 到 ssd 模型
        sd_hijack.model_hijack.convert_sdxl_to_ssd(model)
    # 如果设置了检查点缓存大小大于0，则缓存新加载的模型
    if shared.opts.sd_checkpoint_cache > 0:
        checkpoints_loaded[checkpoint_info] = state_dict.copy()

    # 加载模型的状态字典，strict=False表示允许不严格匹配参数
    model.load_state_dict(state_dict, strict=False)
    timer.record("apply weights to model")

    # 删除状态字典，释放内存
    del state_dict

    # 如果命令行参数中指定了opt_channelslast，则将模型转换为channels_last内存格式
    if shared.cmd_opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)
        timer.record("apply channels_last")

    # 如果命令行参数中指定了no_half，则将模型转换为float类型
    if shared.cmd_opts.no_half:
        model.float()
        devices.dtype_unet = torch.float32
        timer.record("apply float()")
    else:
        # 获取模型中的VAE和深度模型
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)

        # 如果指定了no_half_vae，则在进行half()转换时，从模型中移除VAE，防止其权重被转换为float16
        if shared.cmd_opts.no_half_vae:
            model.first_stage_model = None
        # 如果指定了upcast_sampling，并且存在深度模型，则不将深度模型的权重转换为float16
        if shared.cmd_opts.upcast_sampling and depth_model:
            model.depth_model = None

        # 将模型转换为半精度float16
        model.half()
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model

        devices.dtype_unet = torch.float16
        timer.record("apply half()")

    # 判断是否需要将unet模型的数据类型转换为float16
    devices.unet_needs_upcast = shared.cmd_opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16

    # 将VAE模型的数据类型转换为指定的数据类型
    model.first_stage_model.to(devices.dtype_vae)
    timer.record("apply dtype to VAE")

    # 清理缓存，如果超过了设定的缓存大小
    while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
        checkpoints_loaded.popitem(last=False)

    # 设置模型的哈希值、检查点文件名和检查点信息
    model.sd_model_hash = sd_model_hash
    model.sd_model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    # 如果模型具有'logvar'属性，则将其转换为指定设备的数据类型，用于训练修复
    if hasattr(model, 'logvar'):
        model.logvar = model.logvar.to(devices.device)  # fix for training

    # 删除基础VAE模型
    sd_vae.delete_base_vae()
    # 清除已加载的 VAE 模型
    sd_vae.clear_loaded_vae()
    # 解析 VAE 模型的文件名和来源
    vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename).tuple()
    # 加载 VAE 模型
    sd_vae.load_vae(model, vae_file, vae_source)
    # 记录加载 VAE 模型的时间
    timer.record("load VAE")
def enable_midas_autodownload():
    """
    Gives the ldm.modules.midas.api.load_model function automatic downloading.

    When the 512-depth-ema model, and other future models like it, is loaded,
    it calls midas.api.load_model to load the associated midas depth model.
    This function applies a wrapper to download the model to the correct
    location automatically.
    """

    midas_path = os.path.join(paths.models_path, 'midas')

    # stable-diffusion-stability-ai hard-codes the midas model path to
    # a location that differs from where other scripts using this model look.
    # HACK: Overriding the path here.
    for k, v in midas.api.ISL_PATHS.items():
        file_name = os.path.basename(v)
        midas.api.ISL_PATHS[k] = os.path.join(midas_path, file_name)

    midas_urls = {
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    }

    midas.api.load_model_inner = midas.api.load_model

    def load_model_wrapper(model_type):
        path = midas.api.ISL_PATHS[model_type]
        if not os.path.exists(path):
            if not os.path.exists(midas_path):
                mkdir(midas_path)

            print(f"Downloading midas model weights for {model_type} to {path}")
            request.urlretrieve(midas_urls[model_type], path)
            print(f"{model_type} downloaded")

        return midas.api.load_model_inner(model_type)

    midas.api.load_model = load_model_wrapper


def patch_given_betas():
    import ldm.models.diffusion.ddpm
    # 定义一个修补后的 register_schedule 函数，将 Omegaconf 中的普通列表转换为 numpy 数组
    def patched_register_schedule(*args, **kwargs):
        """a modified version of register_schedule function that converts plain list from Omegaconf into numpy"""

        # 如果第二个参数是 ListConfig 类型，则将其转换为 numpy 数组
        if isinstance(args[1], ListConfig):
            args = (args[0], np.array(args[1]), *args[2:])

        # 调用原始的 register_schedule 函数
        original_register_schedule(*args, **kwargs)

    # 保存原始的 register_schedule 函数
    original_register_schedule = patches.patch(__name__, ldm.models.diffusion.ddpm.DDPM, 'register_schedule', patched_register_schedule)
# 修复配置文件中的一些参数，确保配置文件中的参数正确性
def repair_config(sd_config):

    # 如果配置文件中的模型参数中没有"use_ema"属性，则设置为False
    if not hasattr(sd_config.model.params, "use_ema"):
        sd_config.model.params.use_ema = False

    # 如果配置文件中的模型参数中有'unet_config'属性
    if hasattr(sd_config.model.params, 'unet_config'):
        # 如果命令行参数中没有'no_half'选项，则将'unet_config'中的'use_fp16'属性设置为False
        if shared.cmd_opts.no_half:
            sd_config.model.params.unet_config.params.use_fp16 = False
        # 如果命令行参数中有'upcast_sampling'选项，则将'unet_config'中的'use_fp16'属性设置为True
        elif shared.cmd_opts.upcast_sampling:
            sd_config.model.params.unet_config.params.use_fp16 = True

    # 如果配置文件中的第一阶段配置参数中的'attn_type'属性为"vanilla-xformers"且xformers库不可用，则将其设置为"vanilla"
    if getattr(sd_config.model.params.first_stage_config.params.ddconfig, "attn_type", None) == "vanilla-xformers" and not shared.xformers_available:
        sd_config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla"

    # 对于UnCLIP-L，覆盖硬编码的karlo目录
    if hasattr(sd_config.model.params, "noise_aug_config") and hasattr(sd_config.model.params.noise_aug_config.params, "clip_stats_path"):
        karlo_path = os.path.join(paths.models_path, 'karlo')
        sd_config.model.params.noise_aug_config.params.clip_stats_path = sd_config.model.params.noise_aug_config.params.clip_stats_path.replace("checkpoints/karlo_models", karlo_path)


# 定义一些Clip权重的路径
sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'
sdxl_clip_weight = 'conditioner.embedders.1.model.ln_final.weight'
sdxl_refiner_clip_weight = 'conditioner.embedders.0.model.ln_final.weight'


# 定义SdModelData类
class SdModelData:
    def __init__(self):
        self.sd_model = None
        self.loaded_sd_models = []
        self.was_loaded_at_least_once = False
        self.lock = threading.Lock()
    # 获取稳定扩散模型
    def get_sd_model(self):
        # 如果至少加载过一次，则返回已加载的稳定扩散模型
        if self.was_loaded_at_least_once:
            return self.sd_model

        # 如果稳定扩散模型为空
        if self.sd_model is None:
            # 使用锁确保线程安全
            with self.lock:
                # 如果稳定扩散模型不为空或者至少加载过一次，则返回已加载的稳定扩散模型
                if self.sd_model is not None or self.was_loaded_at_least_once:
                    return self.sd_model

                try:
                    # 加载模型
                    load_model()

                except Exception as e:
                    # 显示加载模型时的错误信息
                    errors.display(e, "loading stable diffusion model", full_traceback=True)
                    print("", file=sys.stderr)
                    print("Stable diffusion model failed to load", file=sys.stderr)
                    self.sd_model = None

        # 返回稳定扩散模型
        return self.sd_model

    # 设置稳定扩散模型
    def set_sd_model(self, v, already_loaded=False):
        # 设置稳定扩散模型为输入的值
        self.sd_model = v
        # 如果已经加载过，则更新相关属性
        if already_loaded:
            sd_vae.base_vae = getattr(v, "base_vae", None)
            sd_vae.loaded_vae_file = getattr(v, "loaded_vae_file", None)
            sd_vae.checkpoint_info = v.sd_checkpoint_info

        try:
            # 移除已加载的稳定扩散模型
            self.loaded_sd_models.remove(v)
        except ValueError:
            pass

        # 如果输入的值不为空，则将其插入到已加载的稳定扩散模型列表的最前面
        if v is not None:
            self.loaded_sd_models.insert(0, v)
# 创建一个 SdModelData 实例
model_data = SdModelData()

# 获取一个空的条件
def get_empty_cond(sd_model):
    # 创建一个 StableDiffusionProcessingTxt2Img 实例
    p = processing.StableDiffusionProcessingTxt2Img()
    # 激活额外的网络
    extra_networks.activate(p, {})

    # 如果 sd_model 有 'conditioner' 属性
    if hasattr(sd_model, 'conditioner'):
        # 获取学习到的条件
        d = sd_model.get_learned_conditioning([""])
        return d['crossattn']
    else:
        return sd_model.cond_stage_model([""])

# 将模型发送到 CPU
def send_model_to_cpu(m):
    # 如果模型的 lowvram 属性为真
    if m.lowvram:
        # 发送所有内容到 CPU
        lowvram.send_everything_to_cpu()
    else:
        # 将模型发送到 CPU
        m.to(devices.cpu)

    # 执行 torch 垃圾回收
    devices.torch_gc()

# 模型的目标设备
def model_target_device(m):
    # 如果需要 lowvram
    if lowvram.is_needed(m):
        return devices.cpu
    else:
        return devices.device

# 将模型发送到设备
def send_model_to_device(m):
    # 应用 lowvram
    lowvram.apply(m)

    # 如果模型不需要 lowvram
    if not m.lowvram:
        # 将模型发送到共享设备
        m.to(shared.device)

# 将模型发送到垃圾桶
def send_model_to_trash(m):
    # 将模型发送到 "meta" 设备
    m.to(device="meta")
    # 执行 torch 垃圾回收
    devices.torch_gc()

# 加载模型
def load_model(checkpoint_info=None, already_loaded_state_dict=None):
    # 导入 sd_hijack 模块
    from modules import sd_hijack
    # 如果没有指定 checkpoint_info，则选择一个检查点
    checkpoint_info = checkpoint_info or select_checkpoint()

    # 创建一个计时器
    timer = Timer()

    # 如果 model_data 中已经存在 sd_model
    if model_data.sd_model:
        # 将 sd_model 发送到垃圾桶
        send_model_to_trash(model_data.sd_model)
        model_data.sd_model = None
        # 执行 torch 垃圾回收
        devices.torch_gc()

    timer.record("unload existing model")

    # 如果已经加载了 state_dict
    if already_loaded_state_dict is not None:
        state_dict = already_loaded_state_dict
    else:
        # 获取检查点的 state_dict
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    # 查找检查点配置
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    # 检查是否包含 clip
    clip_is_included_into_sd = any(x for x in [sd1_clip_weight, sd2_clip_weight, sdxl_clip_weight, sdxl_refiner_clip_weight] if x in state_dict)

    timer.record("find config")

    # 加载检查点配置
    sd_config = OmegaConf.load(checkpoint_config)
    # 修复配置
    repair_config(sd_config)

    timer.record("load config")

    print(f"Creating model from config: {checkpoint_config}")

    sd_model = None
    # 尝试创建模型实例，禁用初始化过程中的剪辑操作
    try:
        with sd_disable_initialization.DisableInitialization(disable_clip=clip_is_included_into_sd or shared.cmd_opts.do_not_download_clip):
            # 在元数据上初始化模型
            with sd_disable_initialization.InitializeOnMeta():
                # 从配置文件实例化模型
                sd_model = instantiate_from_config(sd_config.model)

    # 捕获异常并显示错误信息
    except Exception as e:
        errors.display(e, "creating model quickly", full_traceback=True)

    # 如果模型实例为空，使用慢速方法重试创建模型
    if sd_model is None:
        print('Failed to create model quickly; will retry using slow method.', file=sys.stderr)

        with sd_disable_initialization.InitializeOnMeta():
            # 从配置文件实例化模型
            sd_model = instantiate_from_config(sd_config.model)

    # 设置模型使用的配置
    sd_model.used_config = checkpoint_config

    # 记录创建模型的时间
    timer.record("create model")

    # 如果命令行选项中没有使用半精度，则权重数据类型转换为 None 或 torch.float16
    if shared.cmd_opts.no_half:
        weight_dtype_conversion = None
    else:
        weight_dtype_conversion = {
            'first_stage_model': None,
            '': torch.float16,
        }

    # 在元数据上加载状态字典到模型
    with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=model_target_device(sd_model), weight_dtype_conversion=weight_dtype_conversion):
        # 加载模型权重
        load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    timer.record("load weights from state dict")

    # 将模型发送到设备
    send_model_to_device(sd_model)
    timer.record("move model to device")

    # 模型劫持
    sd_hijack.model_hijack.hijack(sd_model)

    timer.record("hijack")

    # 设置模型为评估模式
    sd_model.eval()
    # 设置模型数据
    model_data.set_sd_model(sd_model)
    model_data.was_loaded_at_least_once = True

    # 加载文本反转嵌入
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)  # Reload embeddings after model load as they may or may not fit the model

    timer.record("load textual inversion embeddings")

    # 模型加载后的脚本回调
    script_callbacks.model_loaded_callback(sd_model)

    timer.record("scripts callbacks")

    # 使用自动混合精度和禁用梯度计算
    with devices.autocast(), torch.no_grad():
        # 计算空提示
        sd_model.cond_stage_model_empty_prompt = get_empty_cond(sd_model)

    timer.record("calculate empty prompt")

    # 打印模型加载时间
    print(f"Model loaded in {timer.summary()}.")

    # 返回加载的模型实例
    return sd_model
def reuse_model_from_already_loaded(sd_model, checkpoint_info, timer):
    """
    检查所需的检查点是否未加载在 model_data.loaded_sd_models 中。
    如果已加载，则返回该模型（必要时将其移动到 GPU，并将当前加载的模型移动到 CPU）。
    如果未加载，则返回可用于从 checkpoint_info 的文件加载权重的模型。
    如果没有这样的模型存在，则返回 None。
    此外，删除超过设置中限制（sd_checkpoints_limit）的加载模型。
    """

    already_loaded = None
    for i in reversed(range(len(model_data.loaded_sd_models))):
        loaded_model = model_data.loaded_sd_models[i]
        # 检查已加载的模型是否与所需的检查点文件名相同
        if loaded_model.sd_checkpoint_info.filename == checkpoint_info.filename:
            already_loaded = loaded_model
            continue

        # 如果加载的模型数量超过设置中的限制，并且限制大于0
        if len(model_data.loaded_sd_models) > shared.opts.sd_checkpoints_limit > 0:
            print(f"Unloading model {len(model_data.loaded_sd_models)} over the limit of {shared.opts.sd_checkpoints_limit}: {loaded_model.sd_checkpoint_info.title}")
            # 从加载的模型列表中移除最后一个模型
            model_data.loaded_sd_models.pop()
            # 将模型发送到垃圾箱
            send_model_to_trash(loaded_model)
            timer.record("send model to trash")

        # 如果设置中指定要保留加载的模型在 CPU 上
        if shared.opts.sd_checkpoints_keep_in_cpu:
            # 将模型发送到 CPU
            send_model_to_cpu(sd_model)
            timer.record("send model to cpu")
    # 如果已经加载了模型，则将其发送到设备
    if already_loaded is not None:
        send_model_to_device(already_loaded)
        # 记录发送模型到设备的时间
        timer.record("send model to device")

        # 设置模型数据的已加载模型
        model_data.set_sd_model(already_loaded, already_loaded=True)

        # 如果不跳过写入配置文件
        if not SkipWritingToConfig.skip:
            # 更新共享选项中的数据，包括模型检查点信息和哈希值
            shared.opts.data["sd_model_checkpoint"] = already_loaded.sd_checkpoint_info.title
            shared.opts.data["sd_checkpoint_hash"] = already_loaded.sd_checkpoint_info.sha256

        # 打印已加载模型的信息和所花费的时间
        print(f"Using already loaded model {already_loaded.sd_checkpoint_info.title}: done in {timer.summary()}")
        # 重新加载 VAE 权重
        sd_vae.reload_vae_weights(already_loaded)
        # 返回模型数据中的 SD 模型
        return model_data.sd_model
    # 如果共享选项中的 SD 检查点限制大于 1 且已加载的 SD 模型数量小于限制
    elif shared.opts.sd_checkpoints_limit > 1 and len(model_data.loaded_sd_models) < shared.opts.sd_checkpoints_limit:
        # 打印正在加载的模型信息
        print(f"Loading model {checkpoint_info.title} ({len(model_data.loaded_sd_models) + 1} out of {shared.opts.sd_checkpoints_limit})")

        # 将 SD 模型设置为 None
        model_data.sd_model = None
        # 加载模型
        load_model(checkpoint_info)
        # 返回模型数据中的 SD 模型
        return model_data.sd_model
    # 如果已加载的 SD 模型数量大于 0
    elif len(model_data.loaded_sd_models) > 0:
        # 弹出已加载的 SD 模型
        sd_model = model_data.loaded_sd_models.pop()
        # 将模型数据中的 SD 模型设置为弹出的模型

        model_data.sd_model = sd_model

        # 更新 SD VAE 的基础 VAE、加载的 VAE 文件和检查点信息
        sd_vae.base_vae = getattr(sd_model, "base_vae", None)
        sd_vae.loaded_vae_file = getattr(sd_model, "loaded_vae_file", None)
        sd_vae.checkpoint_info = sd_model.sd_checkpoint_info

        # 打印重用已加载模型的信息
        print(f"Reusing loaded model {sd_model.sd_checkpoint_info.title} to load {checkpoint_info.title}")
        # 返回 SD 模型
        return sd_model
    # 如果没有符合条件的情况，则返回 None
    else:
        return None
# 重新加载模型的权重
def reload_model_weights(sd_model=None, info=None):
    # 获取检查点信息，如果没有提供则选择一个检查点
    checkpoint_info = info or select_checkpoint()

    # 创建计时器对象
    timer = Timer()

    # 如果没有提供模型，则使用全局变量中的模型
    if not sd_model:
        sd_model = model_data.sd_model

    # 如果模型为空，则表示之前加载模型失败
    if sd_model is None:
        current_checkpoint_info = None
    else:
        current_checkpoint_info = sd_model.sd_checkpoint_info
        # 如果当前模型的检查点文件名与目标检查点文件名相同，则直接返回当前模型
        if sd_model.sd_model_checkpoint == checkpoint_info.filename:
            return sd_model

    # 重用已加载的模型
    sd_model = reuse_model_from_already_loaded(sd_model, checkpoint_info, timer)
    # 如果成功重用模型且检查点文件名相同，则直接返回模型
    if sd_model is not None and sd_model.sd_checkpoint_info.filename == checkpoint_info.filename:
        return sd_model

    # 如果模型不为空，则执行以下操作
    if sd_model is not None:
        # 应用 UNet 模型
        sd_unet.apply_unet("None")
        # 将模型发送到 CPU
        send_model_to_cpu(sd_model)
        # 撤销模型劫持
        sd_hijack.model_hijack.undo_hijack(sd_model)

    # 获取检查点状态字典
    state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    # 查找检查点配置
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)

    timer.record("find config")

    # 如果模型为空或检查点配置与已使用配置不同，则执行以下操作
    if sd_model is None or checkpoint_config != sd_model.used_config:
        if sd_model is not None:
            # 将模型发送到垃圾箱
            send_model_to_trash(sd_model)

        # 加载模型
        load_model(checkpoint_info, already_loaded_state_dict=state_dict)
        return model_data.sd_model

    try:
        # 加载模型权重
        load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    except Exception:
        print("Failed to load checkpoint, restoring previous")
        # 加载之前的检查点
        load_model_weights(sd_model, current_checkpoint_info, None, timer)
        raise
    finally:
        # 模型劫持
        sd_hijack.model_hijack.hijack(sd_model)
        timer.record("hijack")

        # 执行脚本回调函数
        script_callbacks.model_loaded_callback(sd_model)
        timer.record("script callbacks")

        # 如果内存不足，则将模型移动到指定设备
        if not sd_model.lowvram:
            sd_model.to(devices.device)
            timer.record("move model to device")

    # 打印加载权重的时间
    print(f"Weights loaded in {timer.summary()}.")

    # 设置全局变量中的模型
    model_data.set_sd_model(sd_model)
    # 应用 UNet 模型
    sd_unet.apply_unet()

    return sd_model
# 卸载模型权重，将模型发送到 CPU
def unload_model_weights(sd_model=None, info=None):
    # 如果未指定模型，则使用共享的 sd_model
    send_model_to_cpu(sd_model or shared.sd_model)
    # 返回卸载后的模型
    return sd_model


# 应用来自 tomesd 的速度和内存优化
def apply_token_merging(sd_model, token_merging_ratio):
    # 获取当前应用的 token 合并比例
    current_token_merging_ratio = getattr(sd_model, 'applied_token_merged_ratio', 0)

    # 如果当前应用的 token 合并比例与传入的比例相同，则直接返回
    if current_token_merging_ratio == token_merging_ratio:
        return

    # 如果当前应用的 token 合并比例大于 0，则移除 tomesd 的补丁
    if current_token_merging_ratio > 0:
        tomesd.remove_patch(sd_model)

    # 如果传入的 token 合并比例大于 0，则应用 tomesd 的补丁
    if token_merging_ratio > 0:
        tomesd.apply_patch(
            sd_model,
            ratio=token_merging_ratio,
            use_rand=False,  # 可能会导致某些采样器出现问题
            merge_attn=True,
            merge_crossattn=False,
            merge_mlp=False
        )

    # 更新模型的应用 token 合并比例
    sd_model.applied_token_merged_ratio = token_merging_ratio
```