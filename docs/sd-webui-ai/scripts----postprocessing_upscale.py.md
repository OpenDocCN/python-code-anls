# `stable-diffusion-webui\scripts\postprocessing_upscale.py`

```
# 从 PIL 库中导入 Image 模块
from PIL import Image
# 导入 numpy 库并使用别名 np
import numpy as np

# 从 modules 包中导入 scripts_postprocessing 和 shared 模块
from modules import scripts_postprocessing, shared
# 从 gradio 库中导入 gr 模块
import gradio as gr

# 从 modules.ui_components 模块中导入 FormRow 和 ToolButton 类
from modules.ui_components import FormRow, ToolButton
# 从 modules.ui 模块中导入 switch_values_symbol 函数

# 创建一个空字典用于缓存放大后的图片
upscale_cache = {}

# 定义一个名为 ScriptPostprocessingUpscale 的类，继承自 ScriptPostprocessing 类
class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    # 类属性：名称为 "Upscale"
    name = "Upscale"
    # 类属性：排序顺序为 1000
    order = 1000

    # 定义 upscale 方法，接受多个参数用于对图片进行放大处理
    def upscale(self, image, info, upscaler, upscale_mode, upscale_by,  upscale_to_width, upscale_to_height, upscale_crop):
        # 如果放大模式为 1
        if upscale_mode == 1:
            # 计算放大比例
            upscale_by = max(upscale_to_width/image.width, upscale_to_height/image.height)
            # 更新信息字典中的键值对
            info["Postprocess upscale to"] = f"{upscale_to_width}x{upscale_to_height}"
        else:
            # 更新信息字典中的键值对
            info["Postprocess upscale by"] = upscale_by

        # 生成缓存键
        cache_key = (hash(np.array(image.getdata()).tobytes()), upscaler.name, upscale_mode, upscale_by,  upscale_to_width, upscale_to_height, upscale_crop)
        # 从缓存中获取已缓存的图片
        cached_image = upscale_cache.pop(cache_key, None)

        # 如果已缓存的图片存在，则使用缓存的图片
        if cached_image is not None:
            image = cached_image
        else:
            # 否则使用指定的放大器对图片进行放大处理
            image = upscaler.scaler.upscale(image, upscale_by, upscaler.data_path)

        # 将处理后的图片存入缓存
        upscale_cache[cache_key] = image
        # 如果缓存中的图片数量超过设定的最大值，则移除最早的图片
        if len(upscale_cache) > shared.opts.upscaling_max_images_in_cache:
            upscale_cache.pop(next(iter(upscale_cache), None), None)

        # 如果放大模式为 1 且启用了裁剪
        if upscale_mode == 1 and upscale_crop:
            # 创建一个新的图片对象用于裁剪
            cropped = Image.new("RGB", (upscale_to_width, upscale_to_height))
            # 将原图片粘贴到新图片中心位置
            cropped.paste(image, box=(upscale_to_width // 2 - image.width // 2, upscale_to_height // 2 - image.height // 2))
            # 更新信息字典中的键值对
            info["Postprocess crop to"] = f"{image.width}x{image.height}"

        # 返回处理后的图片
        return image
    # 处理第一遍后处理，根据参数进行图像的放大处理
    def process_firstpass(self, pp: scripts_postprocessing.PostprocessedImage, upscale_mode=1, upscale_by=2.0, upscale_to_width=None, upscale_to_height=None, upscale_crop=False, upscaler_1_name=None, upscaler_2_name=None, upscaler_2_visibility=0.0):
        # 如果放大模式为1，则设置目标宽度和高度为指定数值
        if upscale_mode == 1:
            pp.shared.target_width = upscale_to_width
            pp.shared.target_height = upscale_to_height
        # 否则，根据放大倍数计算目标宽度和高度
        else:
            pp.shared.target_width = int(pp.image.width * upscale_by)
            pp.shared.target_height = int(pp.image.height * upscale_by)
    # 处理后处理图像，根据指定参数进行图像放大处理
    def process(self, pp: scripts_postprocessing.PostprocessedImage, upscale_mode=1, upscale_by=2.0, upscale_to_width=None, upscale_to_height=None, upscale_crop=False, upscaler_1_name=None, upscaler_2_name=None, upscaler_2_visibility=0.0):
        # 如果第一个放大器名称为 "None"，则将其设置为 None
        if upscaler_1_name == "None":
            upscaler_1_name = None

        # 查找第一个放大器对象
        upscaler1 = next(iter([x for x in shared.sd_upscalers if x.name == upscaler_1_name]), None)
        # 断言找到第一个放大器对象或者第一个放大器名称为 None
        assert upscaler1 or (upscaler_1_name is None), f'could not find upscaler named {upscaler_1_name}'

        # 如果未找到第一个放大器对象，则返回
        if not upscaler1:
            return

        # 如果第二个放大器名称为 "None"，则将其设置为 None
        if upscaler_2_name == "None":
            upscaler_2_name = None

        # 查找第二个放大器对象
        upscaler2 = next(iter([x for x in shared.sd_upscalers if x.name == upscaler_2_name and x.name != "None"]), None)
        # 断言找到第二个放大器对象或者第二个放大器名称为 None
        assert upscaler2 or (upscaler_2_name is None), f'could not find upscaler named {upscaler_2_name}'

        # 对图像进行第一次放大处理
        upscaled_image = self.upscale(pp.image, pp.info, upscaler1, upscale_mode, upscale_by, upscale_to_width, upscale_to_height, upscale_crop)
        # 将第一个放大器名称添加到信息字典中
        pp.info["Postprocess upscaler"] = upscaler1.name

        # 如果存在第二个放大器对象且第二个放大器可见度大于 0
        if upscaler2 and upscaler_2_visibility > 0:
            # 对图像进行第二次放大处理
            second_upscale = self.upscale(pp.image, pp.info, upscaler2, upscale_mode, upscale_by, upscale_to_width, upscale_to_height, upscale_crop)
            # 使用混合模式将第一次放大处理和第二次放大处理的图像混合
            upscaled_image = Image.blend(upscaled_image, second_upscale, upscaler_2_visibility)
            # 将第二个放大器名称添加到信息字典中
            pp.info["Postprocess upscaler 2"] = upscaler2.name

        # 更新后处理图像为放大处理后的图像
        pp.image = upscaled_image

    # 图像发生改变时，清空放大缓存
    def image_changed(self):
        upscale_cache.clear()
# 定义一个继承自ScriptPostprocessingUpscale的类，用于简单的图像放大处理
class ScriptPostprocessingUpscaleSimple(ScriptPostprocessingUpscale):
    # 定义类的名称为"Simple Upscale"
    name = "Simple Upscale"
    # 定义类的执行顺序为900
    order = 900

    # 定义用户界面
    def ui(self):
        # 创建一个表单行
        with FormRow():
            # 创建一个下拉框，用于选择图像放大器
            upscaler_name = gr.Dropdown(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name)
            # 创建一个滑块，用于设置放大倍数
            upscale_by = gr.Slider(minimum=0.05, maximum=8.0, step=0.05, label="Upscale by", value=2)

        # 返回用户界面元素
        return {
            "upscale_by": upscale_by,
            "upscaler_name": upscaler_name,
        }

    # 处理第一遍的图像处理
    def process_firstpass(self, pp: scripts_postprocessing.PostprocessedImage, upscale_by=2.0, upscaler_name=None):
        # 计算目标图像的宽度和高度
        pp.shared.target_width = int(pp.image.width * upscale_by)
        pp.shared.target_height = int(pp.image.height * upscale_by)

    # 处理图像
    def process(self, pp: scripts_postprocessing.PostprocessedImage, upscale_by=2.0, upscaler_name=None):
        # 如果没有选择图像放大器或选择了"None"，则直接返回
        if upscaler_name is None or upscaler_name == "None":
            return

        # 查找指定名称的图像放大器
        upscaler1 = next(iter([x for x in shared.sd_upscalers if x.name == upscaler_name]), None)
        # 断言找到了指定的图像放大器，否则抛出异常
        assert upscaler1, f'could not find upscaler named {upscaler_name}'

        # 使用指定的图像放大器对图像进行放大处理
        pp.image = self.upscale(pp.image, pp.info, upscaler1, 0, upscale_by, 0, 0, False)
        # 记录处理过程中使用的图像放大器名称
        pp.info["Postprocess upscaler"] = upscaler1.name
```