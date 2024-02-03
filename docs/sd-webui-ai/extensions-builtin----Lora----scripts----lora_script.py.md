# `stable-diffusion-webui\extensions-builtin\Lora\scripts\lora_script.py`

```py
# 导入正则表达式模块
import re

# 导入 gradio 和 FastAPI 模块
import gradio as gr
from fastapi import FastAPI

# 导入自定义模块
import network
import networks
import lora  # noqa:F401
import lora_patches
import extra_networks_lora
import ui_extra_networks_lora
from modules import script_callbacks, ui_extra_networks, extra_networks, shared

# 定义卸载函数，用于撤销网络设置
def unload():
    networks.originals.undo()

# 在 UI 加载前执行的函数
def before_ui():
    # 注册额外网络页面
    ui_extra_networks.register_page(ui_extra_networks_lora.ExtraNetworksPageLora())

    # 创建 Lora 额外网络对象
    networks.extra_network_lora = extra_networks_lora.ExtraNetworkLora()
    extra_networks.register_extra_network(networks.extra_network_lora)
    extra_networks.register_extra_network_alias(networks.extra_network_lora, "lyco")

# 初始化网络原始设置
networks.originals = lora_patches.LoraPatches()

# 在模型加载时执行的回调函数
script_callbacks.on_model_loaded(networks.assign_network_names_to_compvis_modules)
# 在脚本卸载时执行的回调函数
script_callbacks.on_script_unloaded(unload)
# 在 UI 加载前执行的回调函数
script_callbacks.on_before_ui(before_ui)
# 在信息文本粘贴时执行的回调函数
script_callbacks.on_infotext_pasted(networks.infotext_pasted)

# 更新共享选项模板
shared.options_templates.update(shared.options_section(('extra_networks', "Extra Networks"), {
    "sd_lora": shared.OptionInfo("None", "Add network to prompt", gr.Dropdown, lambda: {"choices": ["None", *networks.available_networks]}, refresh=networks.list_available_networks),
    "lora_preferred_name": shared.OptionInfo("Alias from file", "When adding to prompt, refer to Lora by", gr.Radio, {"choices": ["Alias from file", "Filename"]}),
    "lora_add_hashes_to_infotext": shared.OptionInfo(True, "Add Lora hashes to infotext"),
    "lora_show_all": shared.OptionInfo(False, "Always show all networks on the Lora page").info("otherwise, those detected as for incompatible version of Stable Diffusion will be hidden"),
    "lora_hide_unknown_for_versions": shared.OptionInfo([], "Hide networks of unknown versions for model versions", gr.CheckboxGroup, {"choices": ["SD1", "SD2", "SDXL"]}),
    "lora_in_memory_limit": shared.OptionInfo(0, "Number of Lora networks to keep cached in memory", gr.Number, {"precision": 0}),
}))
# 更新共享选项模板，将兼容性部分的配置信息添加到选项模板中
shared.options_templates.update(shared.options_section(('compatibility', "Compatibility"), {
    "lora_functional": shared.OptionInfo(False, "Lora/Networks: use old method that takes longer when you have multiple Loras active and produces same results as kohya-ss/sd-webui-additional-networks extension"),
}))

# 创建一个 JSON 对象，表示网络对象的信息
def create_lora_json(obj: network.NetworkOnDisk):
    return {
        "name": obj.name,
        "alias": obj.alias,
        "path": obj.filename,
        "metadata": obj.metadata,
    }

# 定义 API 网络端点
def api_networks(_: gr.Blocks, app: FastAPI):
    # 获取 Lora 网络信息
    @app.get("/sdapi/v1/loras")
    async def get_loras():
        return [create_lora_json(obj) for obj in networks.available_networks.values()]

    # 刷新 Lora 网络信息
    @app.post("/sdapi/v1/refresh-loras")
    async def refresh_loras():
        return networks.list_available_networks()

# 在应用启动时注册 API 网络端点
script_callbacks.on_app_started(api_networks)

# 编译正则表达式，用于匹配 <lora:...: 格式的字符串
re_lora = re.compile("<lora:([^:]+):")

# 处理粘贴的信息文本
def infotext_pasted(infotext, d):
    # 获取 Lora hashes
    hashes = d.get("Lora hashes")
    if not hashes:
        return

    # 解析 Lora hashes 字符串，生成哈希映射
    hashes = [x.strip().split(':', 1) for x in hashes.split(",")]
    hashes = {x[0].strip().replace(",", ""): x[1].strip() for x in hashes}

    # 替换网络别名
    def network_replacement(m):
        alias = m.group(1)
        shorthash = hashes.get(alias)
        if shorthash is None:
            return m.group(0)

        network_on_disk = networks.available_network_hash_lookup.get(shorthash)
        if network_on_disk is None:
            return m.group(0)

        return f'<lora:{network_on_disk.get_alias()}:'

    # 替换 Prompt 中的网络信息
    d["Prompt"] = re.sub(re_lora, network_replacement, d["Prompt"])

# 在信息文本粘贴时触发处理函数
script_callbacks.on_infotext_pasted(infotext_pasted)

# 当 lora_in_memory_limit 选项发生变化时，从内存中清除网络信息
shared.opts.onchange("lora_in_memory_limit", networks.purge_networks_from_memory)
```