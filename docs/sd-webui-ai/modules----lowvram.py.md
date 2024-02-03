# `stable-diffusion-webui\modules\lowvram.py`

```
# 导入 torch 库
import torch
# 从 modules 模块中导入 devices 和 shared
from modules import devices, shared

# 初始化一个全局变量 module_in_gpu
module_in_gpu = None
# 创建一个 CPU 设备对象
cpu = torch.device("cpu")

# 将所有内容发送到 CPU
def send_everything_to_cpu():
    global module_in_gpu

    # 如果 module_in_gpu 不为空，则将其转移到 CPU
    if module_in_gpu is not None:
        module_in_gpu.to(cpu)

    # 将 module_in_gpu 置为 None
    module_in_gpu = None

# 判断是否需要进行操作
def is_needed(sd_model):
    # 返回 shared.cmd_opts.lowvram 或 shared.cmd_opts.medvram 或 shared.cmd_opts.medvram_sdxl 为真，并且 sd_model 具有 'conditioner' 属性
    return shared.cmd_opts.lowvram or shared.cmd_opts.medvram or shared.cmd_opts.medvram_sdxl and hasattr(sd_model, 'conditioner')

# 应用操作
def apply(sd_model):
    # 判断是否需要进行操作
    enable = is_needed(sd_model)
    # 设置 shared.parallel_processing_allowed 为非 enable
    shared.parallel_processing_allowed = not enable

    # 如果需要进行操作
    if enable:
        # 设置用于低显存的配置
        setup_for_low_vram(sd_model, not shared.cmd_opts.lowvram)
    else:
        # 将 sd_model 的 lowvram 属性设置为 False
        sd_model.lowvram = False

# 为低显存配置设置
def setup_for_low_vram(sd_model, use_medvram):
    # 如果 sd_model 的 lowvram 属性为真，则直接返回
    if getattr(sd_model, 'lowvram', False):
        return

    # 将 sd_model 的 lowvram 属性设置为 True
    sd_model.lowvram = True

    # 初始化父模块字典
    parents = {}

    # 将模块发送到 GPU 的函数
    def send_me_to_gpu(module, _):
        """send this module to GPU; send whatever tracked module was previous in GPU to CPU;
        we add this as forward_pre_hook to a lot of modules and this way all but one of them will
        be in CPU
        """
        global module_in_gpu

        # 获取模块的父模块
        module = parents.get(module, module)

        # 如果当前模块已经在 GPU 上，则直接返回
        if module_in_gpu == module:
            return

        # 如果之前有模块在 GPU 上，则将其转移到 CPU
        if module_in_gpu is not None:
            module_in_gpu.to(cpu)

        # 将当前模块发送到指定设备
        module.to(devices.device)
        module_in_gpu = module

    # 替换 first_stage_model 的 encode 和 decode 方法
    first_stage_model = sd_model.first_stage_model
    first_stage_model_encode = sd_model.first_stage_model.encode
    first_stage_model_decode = sd_model.first_stage_model.decode

    # 包装 first_stage_model 的 encode 方法
    def first_stage_model_encode_wrap(x):
        send_me_to_gpu(first_stage_model, None)
        return first_stage_model_encode(x)

    # 包装 first_stage_model 的 decode 方法
    def first_stage_model_decode_wrap(z):
        send_me_to_gpu(first_stage_model, None)
        return first_stage_model_decode(z)
    # 定义需要保留在 CPU 上的模块列表，每个元组包含模型对象和字段名
    to_remain_in_cpu = [
        (sd_model, 'first_stage_model'),
        (sd_model, 'depth_model'),
        (sd_model, 'embedder'),
        (sd_model, 'model'),
        (sd_model, 'embedder'),
    ]

    # 检查模型是否为 SDXL 类型
    is_sdxl = hasattr(sd_model, 'conditioner')
    # 检查模型是否为 SD2 类型
    is_sd2 = not is_sdxl and hasattr(sd_model.cond_stage_model, 'model')

    # 根据模型类型添加需要保留在 CPU 上的模块
    if is_sdxl:
        to_remain_in_cpu.append((sd_model, 'conditioner'))
    elif is_sd2:
        to_remain_in_cpu.append((sd_model.cond_stage_model, 'model'))
    else:
        to_remain_in_cpu.append((sd_model.cond_stage_model, 'transformer')

    # 移除几个大模块：cond, first_stage, depth/embedder（如果适用）和 unet 从模型中
    stored = []
    for obj, field in to_remain_in_cpu:
        # 获取模块对象
        module = getattr(obj, field, None)
        stored.append(module)
        # 将模块置空
        setattr(obj, field, None)

    # 将模型发送到 GPU
    sd_model.to(devices.device)

    # 将模块放回到 CPU 上
    for (obj, field), module in zip(to_remain_in_cpu, stored):
        setattr(obj, field, module)

    # 为前三个模型注册钩子函数
    if is_sdxl:
        sd_model.conditioner.register_forward_pre_hook(send_me_to_gpu)
    elif is_sd2:
        sd_model.cond_stage_model.model.register_forward_pre_hook(send_me_to_gpu)
        sd_model.cond_stage_model.model.token_embedding.register_forward_pre_hook(send_me_to_gpu)
        parents[sd_model.cond_stage_model.model] = sd_model.cond_stage_model
        parents[sd_model.cond_stage_model.model.token_embedding] = sd_model.cond_stage_model
    else:
        sd_model.cond_stage_model.transformer.register_forward_pre_hook(send_me_to_gpu)
        parents[sd_model.cond_stage_model.transformer] = sd_model.cond_stage_model

    # 为 first_stage_model 注册钩子函数
    sd_model.first_stage_model.register_forward_pre_hook(send_me_to_gpu)
    # 重写 first_stage_model 的 encode 和 decode 方法
    sd_model.first_stage_model.encode = first_stage_model_encode_wrap
    sd_model.first_stage_model.decode = first_stage_model_decode_wrap
    # 如果存在深度模型，则注册前向钩子函数将数据发送到 GPU
    if sd_model.depth_model:
        sd_model.depth_model.register_forward_pre_hook(send_me_to_gpu)
    # 如果存在嵌入器，则注册前向钩子函数将数据发送到 GPU
    if sd_model.embedder:
        sd_model.embedder.register_forward_pre_hook(send_me_to_gpu)

    # 如果使用 MedVRAM，则注册前向钩子函数将数据发送到 GPU
    if use_medvram:
        sd_model.model.register_forward_pre_hook(send_me_to_gpu)
    else:
        # 否则，获取第三个模型的差异模型
        diff_model = sd_model.model.diffusion_model

        # 第三个剩余模型仍然太大，无法在 4GB 内存中，因此我们也对其子模块执行相同操作
        # 以便一次只有一个模块在 GPU 上
        stored = diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = None, None, None, None
        # 将模型移动到指定设备
        sd_model.model.to(devices.device)
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = stored

        # 为第三个模型的部分安装钩子函数
        diff_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.input_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
        diff_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.output_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
# 检查给定的sd_model对象是否启用了低内存模式，并返回相应的布尔值
def is_enabled(sd_model):
    # 返回sd_model对象的lowvram属性，表示是否启用了低内存模式
    return sd_model.lowvram
```