# `stable-diffusion-webui\modules\shared_items.py`

```
# 导入 sys 模块
import sys

# 导入 shared_cmd_options 模块中的 cmd_opts 对象
from modules.shared_cmd_options import cmd_opts

# 获取 realesrgan 模型的名称列表
def realesrgan_models_names():
    # 导入 realesrgan_model 模块
    import modules.realesrgan_model
    # 返回 realesrgan 模型的名称列表
    return [x.name for x in modules.realesrgan_model.get_realesrgan_models(None)]

# 获取后处理脚本列表
def postprocessing_scripts():
    # 导入 scripts 模块
    import modules.scripts
    # 返回后处理脚本列表
    return modules.scripts.scripts_postproc.scripts

# 获取 sd_vae 项目列表
def sd_vae_items():
    # 导入 sd_vae 模块
    import modules.sd_vae
    # 返回 sd_vae 项目列表
    return ["Automatic", "None"] + list(modules.sd_vae.vae_dict)

# 刷新 vae 列表
def refresh_vae_list():
    # 导入 sd_vae 模块
    import modules.sd_vae
    # 刷新 vae 列表
    modules.sd_vae.refresh_vae_list()

# 获取交叉注意力优化列表
def cross_attention_optimizations():
    # 导入 sd_hijack 模块
    import modules.sd_hijack
    # 返回交叉注意力优化列表
    return ["Automatic"] + [x.title() for x in modules.sd_hijack.optimizers] + ["None"]

# 获取 sd_unet 项目列表
def sd_unet_items():
    # 导入 sd_unet 模块
    import modules.sd_unet
    # 返回 sd_unet 项目列表
    return ["Automatic"] + [x.label for x in modules.sd_unet.unet_options] + ["None"]

# 刷新 unet 列表
def refresh_unet_list():
    # 导入 sd_unet 模块
    import modules.sd_unet
    # 刷新 unet 列表
    modules.sd_unet.list_unets()

# 列出检查点瓷砖
def list_checkpoint_tiles(use_short=False):
    # 导入 sd_models 模块
    import modules.sd_models
    # 返回检查点瓷砖列表
    return modules.sd_models.checkpoint_tiles(use_short)

# 刷新检查点
def refresh_checkpoints():
    # 导入 sd_models 模块
    import modules.sd_models
    # 刷新检查点列表
    return modules.sd_models.list_models()

# 列出采样器
def list_samplers():
    # 导入 sd_samplers 模块
    import modules.sd_samplers
    # 返回所有采样器列表
    return modules.sd_samplers.all_samplers

# 重新加载超网络
def reload_hypernetworks():
    # 导入 hypernetwork 模块中的 hypernetwork 对象和 shared 模块
    from modules.hypernetworks import hypernetwork
    from modules import shared
    # 将 hypernetworks 设置为超网络列表
    shared.hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)

# 获取信息文本名称列表
def get_infotext_names():
    # 导入 generation_parameters_copypaste 和 shared 模块
    from modules import generation_parameters_copypaste, shared
    # 创建空字典 res
    res = {}
    # 遍历数据标签中的信息
    for info in shared.opts.data_labels.values():
        # 如果信息文本存在，则将其添加到 res 中
        if info.infotext:
            res[info.infotext] = 1
    # 遍历粘贴字段中的数据
    for tab_data in generation_parameters_copypaste.paste_fields.values():
        for _, name in tab_data.get("fields") or []:
            # 如果名称是字符串，则将其添加到 res 中
            if isinstance(name, str):
                res[name] = 1
    # 返回 res 的键列表
    return list(res)

# UI 重新排序内置项目列表
ui_reorder_categories_builtin_items = [
    "prompt",
    "image",
    "inpaint",
    "sampler",
    "accordions",
    "checkboxes",  # 定义一个字符串 "checkboxes"
    "dimensions",  # 定义一个字符串 "dimensions"
    "cfg",  # 定义一个字符串 "cfg"
    "denoising",  # 定义一个字符串 "denoising"
    "seed",  # 定义一个字符串 "seed"
    "batch",  # 定义一个字符串 "batch"
    "override_settings",  # 定义一个字符串 "override_settings"
# 定义一个函数用于重新排序类别
def ui_reorder_categories():
    # 从模块中导入脚本
    from modules import scripts

    # 从内置项中生成迭代器
    yield from ui_reorder_categories_builtin_items

    # 创建一个空字典用于存储各个部分
    sections = {}
    # 遍历所有脚本，将其所属部分加入字典中
    for script in scripts.scripts_txt2img.scripts + scripts.scripts_img2img.scripts:
        if isinstance(script.section, str) and script.section not in ui_reorder_categories_builtin_items:
            sections[script.section] = 1

    # 生成部分的迭代器
    yield from sections

    # 添加脚本部分
    yield "scripts"

# 创建一个类 Shared，用于提供 sd_model 字段作为属性，以便在需要时创建和加载，而不是在程序启动时
class Shared(sys.modules[__name__].__class__):
    """
    this class is here to provide sd_model field as a property, so that it can be created and loaded on demand rather than
    at program startup.
    """

    # 初始化 sd_model_val 为 None
    sd_model_val = None

    # 定义 sd_model 属性，用于获取 sd_model 数据
    @property
    def sd_model(self):
        # 导入 sd_models 模块
        import modules.sd_models

        # 返回 sd_model 数据
        return modules.sd_models.model_data.get_sd_model()

    # 定义 sd_model 属性的 setter 方法，用于设置 sd_model 数据
    @sd_model.setter
    def sd_model(self, value):
        # 导入 sd_models 模块
        import modules.sd_models

        # 设置 sd_model 数据
        modules.sd_models.model_data.set_sd_model(value)

# 将模块 'modules.shared' 的类设置为 Shared 类
sys.modules['modules.shared'].__class__ = Shared
```