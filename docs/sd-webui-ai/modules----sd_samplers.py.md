# `stable-diffusion-webui\modules\sd_samplers.py`

```py
# 从 modules 模块中导入 sd_samplers_kdiffusion、sd_samplers_timesteps 和 shared 模块
from modules import sd_samplers_kdiffusion, sd_samplers_timesteps, shared

# 从 sd_samplers_common 模块中导入 samples_to_image_grid 和 sample_to_image 函数，忽略 F401 错误
from modules.sd_samplers_common import samples_to_image_grid, sample_to_image  # noqa: F401

# 将 sd_samplers_kdiffusion 和 sd_samplers_timesteps 模块中的所有采样器合并到 all_samplers 列表中
all_samplers = [
    *sd_samplers_kdiffusion.samplers_data_k_diffusion,
    *sd_samplers_timesteps.samplers_data_timesteps,
]

# 创建一个字典，将所有采样器的名称映射到对应的采样器对象
all_samplers_map = {x.name: x for x in all_samplers}

# 初始化空列表 samplers、samplers_for_img2img、samplers_map 和 samplers_hidden
samplers = []
samplers_for_img2img = []
samplers_map = {}
samplers_hidden = {}

# 根据给定的采样器名称查找对应的配置信息
def find_sampler_config(name):
    if name is not None:
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]

    return config

# 根据给定的采样器名称和模型创建采样器对象
def create_sampler(name, model):
    config = find_sampler_config(name)

    assert config is not None, f'bad sampler name: {name}'

    if model.is_sdxl and config.options.get("no_sdxl", False):
        raise Exception(f"Sampler {config.name} is not supported for SDXL")

    sampler = config.constructor(model)
    sampler.config = config

    return sampler

# 设置全局变量 samplers、samplers_for_img2img 和 samplers_hidden
def set_samplers():
    global samplers, samplers_for_img2img, samplers_hidden

    samplers_hidden = set(shared.opts.hide_samplers)
    samplers = all_samplers
    samplers_for_img2img = all_samplers

    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name

# 返回所有可见采样器的名称列表
def visible_sampler_names():
    return [x.name for x in samplers if x.name not in samplers_hidden]

# 调用 set_samplers() 函数，初始化采样器相关的全局变量
set_samplers()
```