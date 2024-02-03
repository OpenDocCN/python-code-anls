# `stable-diffusion-webui\modules\sd_vae.py`

```py
# 导入必要的模块
import os
import collections
from dataclasses import dataclass

# 从自定义模块中导入指定内容
from modules import paths, shared, devices, script_callbacks, sd_models, extra_networks, lowvram, sd_hijack, hashes

# 导入 glob 函数和 deepcopy 函数
import glob
from copy import deepcopy

# 定义变量 vae_path 为 VAE 模型的绝对路径
vae_path = os.path.abspath(os.path.join(paths.models_path, "VAE"))
# 定义忽略的键集合
vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
# 创建空字典 vae_dict
vae_dict = {}

# 初始化变量 base_vae, loaded_vae_file, checkpoint_info
base_vae = None
loaded_vae_file = None
checkpoint_info = None

# 创建有序字典 checkpoints_loaded
checkpoints_loaded = collections.OrderedDict()

# 定义函数 get_loaded_vae_name，返回加载的 VAE 模型文件名
def get_loaded_vae_name():
    if loaded_vae_file is None:
        return None
    return os.path.basename(loaded_vae_file)

# 定义函数 get_loaded_vae_hash，返回加载的 VAE 模型文件的哈希值
def get_loaded_vae_hash():
    if loaded_vae_file is None:
        return None
    sha256 = hashes.sha256(loaded_vae_file, 'vae')
    return sha256[0:10] if sha256 else None

# 定义函数 get_base_vae，返回基础 VAE 模型
def get_base_vae(model):
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None

# 定义函数 store_base_vae，存储基础 VAE 模型
def store_base_vae(model):
    global base_vae, checkpoint_info
    if checkpoint_info != model.sd_checkpoint_info:
        assert not loaded_vae_file, "Trying to store non-base VAE!"
        base_vae = deepcopy(model.first_stage_model.state_dict())
        checkpoint_info = model.sd_checkpoint_info

# 定义函数 delete_base_vae，删除基础 VAE 模型
def delete_base_vae():
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None

# 定义函数 restore_base_vae，恢复基础 VAE 模型
def restore_base_vae(model):
    global loaded_vae_file
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        print("Restoring base VAE")
        _load_vae_dict(model, base_vae)
        loaded_vae_file = None
    delete_base_vae()

# 定义函数 get_filename，返回文件路径的文件名部分
def get_filename(filepath):
    return os.path.basename(filepath)

# 清空 VAE 字典
def refresh_vae_list():
    vae_dict.clear()
    # 定义需要搜索的文件路径列表
    paths = [
        os.path.join(sd_models.model_path, '**/*.vae.ckpt'),  # 包含模型路径下所有以.vae.ckpt结尾的文件
        os.path.join(sd_models.model_path, '**/*.vae.pt'),    # 包含模型路径下所有以.vae.pt结尾的文件
        os.path.join(sd_models.model_path, '**/*.vae.safetensors'),  # 包含模型路径下所有以.vae.safetensors结尾的文件
        os.path.join(vae_path, '**/*.ckpt'),  # 包含vae路径下所有以.ckpt结尾的文件
        os.path.join(vae_path, '**/*.pt'),    # 包含vae路径下所有以.pt结尾的文件
        os.path.join(vae_path, '**/*.safetensors'),  # 包含vae路径下所有以.safetensors结尾的文件
    ]
    
    # 如果指定了检查点目录并且该目录存在
    if shared.cmd_opts.ckpt_dir is not None and os.path.isdir(shared.cmd_opts.ckpt_dir):
        # 将检查点目录下符合条件的文件路径添加到列表中
        paths += [
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.ckpt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.pt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.safetensors'),
        ]
    
    # 如果指定了VAE目录并且该目录存在
    if shared.cmd_opts.vae_dir is not None and os.path.isdir(shared.cmd_opts.vae_dir):
        # 将VAE目录下符合条件的文件路径添加到列表中
        paths += [
            os.path.join(shared.cmd_opts.vae_dir, '**/*.ckpt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.pt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.safetensors'),
        ]
    
    # 初始化候选文件列表
    candidates = []
    # 遍历文件路径列表，使用glob模块搜索匹配的文件路径并添加到候选文件列表中
    for path in paths:
        candidates += glob.iglob(path, recursive=True)
    
    # 遍历候选文件列表中的文件路径
    for filepath in candidates:
        # 获取文件名
        name = get_filename(filepath)
        # 将文件名和文件路径添加到VAE字典中
        vae_dict[name] = filepath
    
    # 更新VAE字典，按文件名进行自然排序
    vae_dict.update(dict(sorted(vae_dict.items(), key=lambda item: shared.natural_sort_key(item[0]))))
# 根据给定的检查点文件名查找与之相关的 VAE 模型文件
def find_vae_near_checkpoint(checkpoint_file):
    # 从检查点文件名中提取文件名（不包括扩展名）
    checkpoint_path = os.path.basename(checkpoint_file).rsplit('.', 1)[0]
    # 遍历 VAE 字典中的值，查找与检查点文件名匹配的 VAE 文件
    for vae_file in vae_dict.values():
        if os.path.basename(vae_file).startswith(checkpoint_path):
            return vae_file

    return None


# 定义 VAE 分辨率类，包含 VAE 文件路径、来源和是否已解析的信息
@dataclass
class VaeResolution:
    vae: str = None
    source: str = None
    resolved: bool = True

    # 返回 VAE 分辨率的元组形式
    def tuple(self):
        return self.vae, self.source


# 检查是否为自动模式
def is_automatic():
    return shared.opts.sd_vae in {"Automatic", "auto"}  # "auto" for people with old config


# 从设置中解析 VAE 分辨率
def resolve_vae_from_setting() -> VaeResolution:
    if shared.opts.sd_vae == "None":
        return VaeResolution()

    # 从选项中获取 VAE 文件路径
    vae_from_options = vae_dict.get(shared.opts.sd_vae, None)
    if vae_from_options is not None:
        return VaeResolution(vae_from_options, 'specified in settings')

    # 如果不是自动模式，则打印未找到指定 VAE 的消息
    if not is_automatic():
        print(f"Couldn't find VAE named {shared.opts.sd_vae}; using None instead")

    return VaeResolution(resolved=False)


# 从用户元数据中解析 VAE 分辨率
def resolve_vae_from_user_metadata(checkpoint_file) -> VaeResolution:
    # 获取检查点文件的用户元数据
    metadata = extra_networks.get_user_metadata(checkpoint_file)
    vae_metadata = metadata.get("vae", None)
    if vae_metadata is not None and vae_metadata != "Automatic":
        if vae_metadata == "None":
            return VaeResolution()

        # 从元数据中获取 VAE 文件路径
        vae_from_metadata = vae_dict.get(vae_metadata, None)
        if vae_from_metadata is not None:
            return VaeResolution(vae_from_metadata, "from user metadata")

    return VaeResolution(resolved=False)


# 从检查点文件附近解析 VAE 分辨率
def resolve_vae_near_checkpoint(checkpoint_file) -> VaeResolution:
    # 查找与检查点文件附近的 VAE 文件
    vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
    # 如果找到 VAE 文件且未覆盖每个模型首选项或为自动模式，则返回找到的 VAE 文件路径
    if vae_near_checkpoint is not None and (not shared.opts.sd_vae_overrides_per_model_preferences or is_automatic()):
        return VaeResolution(vae_near_checkpoint, 'found near the checkpoint')

    return VaeResolution(resolved=False)


# 解析 VAE 分辨率
def resolve_vae(checkpoint_file) -> VaeResolution:
    # 如果命令行参数中指定了 VAE 路径，则返回该 VAE 对象
    if shared.cmd_opts.vae_path is not None:
        return VaeResolution(shared.cmd_opts.vae_path, 'from commandline argument')

    # 如果设置了 VAE 覆盖每个模型的偏好，并且不是自动模式，则从设置中解析 VAE
    if shared.opts.sd_vae_overrides_per_model_preferences and not is_automatic():
        return resolve_vae_from_setting()

    # 从用户元数据中解析 VAE
    res = resolve_vae_from_user_metadata(checkpoint_file)
    if res.resolved:
        return res

    # 在检查点附近解析 VAE
    res = resolve_vae_near_checkpoint(checkpoint_file)
    if res.resolved:
        return res

    # 从设置中解析 VAE
    res = resolve_vae_from_setting()

    # 返回解析的结果
    return res
# 加载 VAE 模型的权重字典
def load_vae_dict(filename, map_location):
    # 从文件中读取 VAE 模型的状态字典
    vae_ckpt = sd_models.read_state_dict(filename, map_location=map_location)
    # 从状态字典中筛选出需要的键值对，不包括以"loss"开头和在忽略键列表中的键值对
    vae_dict_1 = {k: v for k, v in vae_ckpt.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    return vae_dict_1

# 加载 VAE 模型
def load_vae(model, vae_file=None, vae_source="from unknown source"):
    global vae_dict, base_vae, loaded_vae_file
    # save_settings = False

    # 检查是否启用了缓存
    cache_enabled = shared.opts.sd_vae_checkpoint_cache > 0

    if vae_file:
        if cache_enabled and vae_file in checkpoints_loaded:
            # 使用 VAE 检查点缓存
            print(f"Loading VAE weights {vae_source}: cached {get_filename(vae_file)}")
            store_base_vae(model)
            _load_vae_dict(model, checkpoints_loaded[vae_file])
        else:
            # 检查 VAE 文件是否存在
            assert os.path.isfile(vae_file), f"VAE {vae_source} doesn't exist: {vae_file}"
            print(f"Loading VAE weights {vae_source}: {vae_file}")
            store_base_vae(model)

            # 加载 VAE 字典
            vae_dict_1 = load_vae_dict(vae_file, map_location=shared.weight_load_location)
            _load_vae_dict(model, vae_dict_1)

            if cache_enabled:
                # 缓存新加载的 VAE
                checkpoints_loaded[vae_file] = vae_dict_1.copy()

        # 清理缓存，如果达到限制
        if cache_enabled:
            while len(checkpoints_loaded) > shared.opts.sd_vae_checkpoint_cache + 1: # 我们需要计算当前模型
                checkpoints_loaded.popitem(last=False)  # LRU

        # 如果使用的 VAE 不在字典中，则更新它
        # 但在刷新时会被移除
        vae_opt = get_filename(vae_file)
        if vae_opt not in vae_dict:
            vae_dict[vae_opt] = vae_file

    elif loaded_vae_file:
        restore_base_vae(model)

    loaded_vae_file = vae_file
    model.base_vae = base_vae
    model.loaded_vae_file = loaded_vae_file

# 不要从外部调用此函数
def _load_vae_dict(model, vae_dict_1):
    # 加载第一个阶段的模型参数
    model.first_stage_model.load_state_dict(vae_dict_1)
    # 将第一个阶段的模型移动到指定设备上
    model.first_stage_model.to(devices.dtype_vae)
# 清除已加载的 VAE 模型文件
def clear_loaded_vae():
    # 使用全局变量 loaded_vae_file，将其设为 None
    global loaded_vae_file
    loaded_vae_file = None

# 未指定的对象
unspecified = object()

# 重新加载 VAE 模型权重
def reload_vae_weights(sd_model=None, vae_file=unspecified):
    # 如果未传入 sd_model 参数，则使用 shared.sd_model
    if not sd_model:
        sd_model = shared.sd_model

    # 获取 sd_model 的检查点信息
    checkpoint_info = sd_model.sd_checkpoint_info
    checkpoint_file = checkpoint_info.filename

    # 如果未指定 vae_file，则通过 resolve_vae 函数解析检查点文件来获取 vae_file 和 vae_source
    if vae_file == unspecified:
        vae_file, vae_source = resolve_vae(checkpoint_file).tuple()
    else:
        vae_source = "from function argument"

    # 如果已加载的 vae_file 与当前 vae_file 相同，则直接返回
    if loaded_vae_file == vae_file:
        return

    # 如果 sd_model.lowvram 为真，则将所有内容发送到 CPU
    if sd_model.lowvram:
        lowvram.send_everything_to_cpu()
    else:
        sd_model.to(devices.cpu)

    # 撤销 hijack 操作
    sd_hijack.model_hijack.undo_hijack(sd_model)

    # 加载 vae 模型
    load_vae(sd_model, vae_file, vae_source)

    # hijack 操作
    sd_hijack.model_hijack.hijack(sd_model)
    script_callbacks.model_loaded_callback(sd_model)

    # 如果 sd_model.lowvram 为假，则将 sd_model 切换回设备
    if not sd_model.lowvram:
        sd_model.to(devices.device)

    # 打印信息，表示 VAE 权重已加载
    print("VAE weights loaded.")
    return sd_model
```