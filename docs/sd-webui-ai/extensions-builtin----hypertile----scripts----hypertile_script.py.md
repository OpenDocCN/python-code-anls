# `stable-diffusion-webui\extensions-builtin\hypertile\scripts\hypertile_script.py`

```
# 导入hypertile模块
import hypertile
# 从modules模块中导入scripts, script_callbacks, shared
from modules import scripts, script_callbacks, shared

# 从scripts模块中导入hypertile_xyz中的add_axis_options函数
from scripts.hypertile_xyz import add_axis_options

# 定义ScriptHypertile类，继承自scripts.Script类
class ScriptHypertile(scripts.Script):
    # 定义类属性name为"Hypertile"
    name = "Hypertile"

    # 定义title方法，返回name属性值
    def title(self):
        return self.name

    # 定义show方法，参数is_img2img，返回scripts.AlwaysVisible
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # 定义process方法，参数p和*args
    def process(self, p, *args):
        # 设置hypertile种子为all_seeds列表的第一个元素
        hypertile.set_hypertile_seed(p.all_seeds[0])

        # 调用configure_hypertile函数，设置宽度、高度和是否启用unet
        configure_hypertile(p.width, p.height, enable_unet=shared.opts.hypertile_enable_unet)

        # 调用add_infotext方法，传入参数p
        self.add_infotext(p)

    # 定义before_hr方法，参数p和*args
    def before_hr(self, p, *args):

        # 根据shared.opts中的设置，确定是否启用unet的第二次传递
        enable = shared.opts.hypertile_enable_unet_secondpass or shared.opts.hypertile_enable_unet

        # 如果启用unet的第二次传递
        if enable:
            # 设置hypertile种子为all_seeds列表的第一个元素
            hypertile.set_hypertile_seed(p.all_seeds[0])

        # 调用configure_hypertile函数，设置高分辨率放大到的宽度、高度和是否启用unet
        configure_hypertile(p.hr_upscale_to_x, p.hr_upscale_to_y, enable_unet=enable)

        # 如果启用unet的第二次传递且未启用unet
        if enable and not shared.opts.hypertile_enable_unet:
            # 在extra_generation_params中添加"Hypertile U-Net second pass"键值对
            p.extra_generation_params["Hypertile U-Net second pass"] = True

            # 调用add_infotext方法，传入参数p和add_unet_params=True
            self.add_infotext(p, add_unet_params=True)
    # 添加信息文本到参数对象中，可选择是否添加 U-Net 参数
    def add_infotext(self, p, add_unet_params=False):
        # 定义内部函数，用于获取参数值
        def option(name):
            # 获取参数值
            value = getattr(shared.opts, name)
            # 获取参数默认值
            default_value = shared.opts.get_default(name)
            # 如果参数值等于默认值，则返回 None，否则返回参数值
            return None if value == default_value else value

        # 如果启用了 Hypertile U-Net，则将参数对象中的额外生成参数设置为 True
        if shared.opts.hypertile_enable_unet:
            p.extra_generation_params["Hypertile U-Net"] = True

        # 如果启用了 Hypertile U-Net 或者需要添加 U-Net 参数，则设置额外生成参数
        if shared.opts.hypertile_enable_unet or add_unet_params:
            p.extra_generation_params["Hypertile U-Net max depth"] = option('hypertile_max_depth_unet')
            p.extra_generation_params["Hypertile U-Net max tile size"] = option('hypertile_max_tile_unet')
            p.extra_generation_params["Hypertile U-Net swap size"] = option('hypertile_swap_size_unet')

        # 如果启用了 Hypertile VAE，则设置参数对象中的额外生成参数
        if shared.opts.hypertile_enable_vae:
            p.extra_generation_params["Hypertile VAE"] = True
            p.extra_generation_params["Hypertile VAE max depth"] = option('hypertile_max_depth_vae')
            p.extra_generation_params["Hypertile VAE max tile size"] = option('hypertile_max_tile_vae')
            p.extra_generation_params["Hypertile VAE swap size"] = option('hypertile_swap_size_vae')
# 配置 Hypertile，用于优化 U-Net 和 VAE 模型中的自注意力层
def configure_hypertile(width, height, enable_unet=True):
    # 钩住第一阶段模型，设置 Hypertile 参数
    hypertile.hypertile_hook_model(
        shared.sd_model.first_stage_model,
        width,
        height,
        swap_size=shared.opts.hypertile_swap_size_vae,
        max_depth=shared.opts.hypertile_max_depth_vae,
        tile_size_max=shared.opts.hypertile_max_tile_vae,
        enable=shared.opts.hypertile_enable_vae,
    )

    # 钩住模型，设置 Hypertile 参数
    hypertile.hypertile_hook_model(
        shared.sd_model.model,
        width,
        height,
        swap_size=shared.opts.hypertile_swap_size_unet,
        max_depth=shared.opts.hypertile_max_depth_unet,
        tile_size_max=shared.opts.hypertile_max_tile_unet,
        enable=enable_unet,
        is_sdxl=shared.sd_model.is_sdxl
    )


# 处理用户界面设置
def on_ui_settings():
    # 导入 gradio 库
    import gradio as gr

    # 设置 Hypertile 说明
    options = {
        "hypertile_explanation": shared.OptionHTML("""
    <a href='https://github.com/tfernd/HyperTile'>Hypertile</a> optimizes the self-attention layer within U-Net and VAE models,
    resulting in a reduction in computation time ranging from 1 to 4 times. The larger the generated image is, the greater the
    benefit.
        # 启用 Hypertile U-Net
        "hypertile_enable_unet": shared.OptionInfo(False, "Enable Hypertile U-Net", infotext="Hypertile U-Net").info("enables hypertile for all modes, including hires fix second pass; noticeable change in details of the generated picture"),
        
        # 启用 Hypertile U-Net 用于 hires fix 的第二次处理
        "hypertile_enable_unet_secondpass": shared.OptionInfo(False, "Enable Hypertile U-Net for hires fix second pass", infotext="Hypertile U-Net second pass").info("enables hypertile just for hires fix second pass - regardless of whether the above setting is enabled"),
        
        # 设置 Hypertile U-Net 的最大深度
        "hypertile_max_depth_unet": shared.OptionInfo(3, "Hypertile U-Net max depth", gr.Slider, {"minimum": 0, "maximum": 3, "step": 1}, infotext="Hypertile U-Net max depth").info("larger = more neural network layers affected; minor effect on performance"),
        
        # 设置 Hypertile U-Net 的最大瓦片大小
        "hypertile_max_tile_unet": shared.OptionInfo(256, "Hypertile U-Net max tile size", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}, infotext="Hypertile U-Net max tile size").info("larger = worse performance"),
        
        # 设置 Hypertile U-Net 的交换大小
        "hypertile_swap_size_unet": shared.OptionInfo(3, "Hypertile U-Net swap size", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}, infotext="Hypertile U-Net swap size"),
        
        # 启用 Hypertile VAE
        "hypertile_enable_vae": shared.OptionInfo(False, "Enable Hypertile VAE", infotext="Hypertile VAE").info("minimal change in the generated picture"),
        
        # 设置 Hypertile VAE 的最大深度
        "hypertile_max_depth_vae": shared.OptionInfo(3, "Hypertile VAE max depth", gr.Slider, {"minimum": 0, "maximum": 3, "step": 1}, infotext="Hypertile VAE max depth"),
        
        # 设置 Hypertile VAE 的最大瓦片大小
        "hypertile_max_tile_vae": shared.OptionInfo(128, "Hypertile VAE max tile size", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}, infotext="Hypertile VAE max tile size"),
        
        # 设置 Hypertile VAE 的交换大小
        "hypertile_swap_size_vae": shared.OptionInfo(3, "Hypertile VAE swap size ", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}, infotext="Hypertile VAE swap size"),
    # 遍历 options 字典，获取每个选项的名称和选项对象
    for name, opt in options.items():
        # 设置选项对象的 section 属性为 ('hypertile', "Hypertile")
        opt.section = ('hypertile', "Hypertile")
        # 将该选项添加到 shared.opts 中
        shared.opts.add_option(name, opt)
# 将 on_ui_settings 函数注册到 script_callbacks 的 UI 设置回调中
script_callbacks.on_ui_settings(on_ui_settings)
# 将 add_axis_options 函数注册到 script_callbacks 的 UI 设置前回调中
script_callbacks.on_before_ui(add_axis_options)
```