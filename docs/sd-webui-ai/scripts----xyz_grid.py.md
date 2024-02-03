# `stable-diffusion-webui\scripts\xyz_grid.py`

```
# 导入必要的模块
from collections import namedtuple
from copy import copy
from itertools import permutations, chain
import random
import csv
import os.path
from io import StringIO
from PIL import Image
import numpy as np

# 导入自定义模块
import modules.scripts as scripts
import gradio as gr

from modules import images, sd_samplers, processing, sd_models, sd_vae, sd_samplers_kdiffusion, errors
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, state
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import modules.sd_vae
import re

from modules.ui_components import ToolButton

# 定义填充值符号
fill_values_symbol = "\U0001f4d2"  # 📒

# 定义命名元组 AxisInfo，包含 axis 和 values 两个字段
AxisInfo = namedtuple('AxisInfo', ['axis', 'values'])

# 定义函数 apply_field，用于设置对象的属性值
def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)
    return fun

# 定义函数 apply_prompt，用于替换 Prompt 中的内容
def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")
    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)

# 定义函数 apply_order，用于按顺序替换 Prompt 中的内容
def apply_order(p, x, xs):
    token_order = []

    # 初始化 token_order 列表，按照在 prompt 中出现的顺序存储 token
    for token in x:
        token_order.append((p.prompt.find(token), token))

    # 按照 token 在 prompt 中出现的位置排序
    token_order.sort(key=lambda t: t[0])

    prompt_parts = []

    # 将 prompt 拆分，取出 token
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]

    # 重新构建 prompt，按照指定顺序插入 token
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt

# 定义函数 confirm_samplers，用于确认采样器
def confirm_samplers(p, xs):
    # 遍历列表 xs 中的每个元素
    for x in xs:
        # 检查元素 x 转换为小写后是否不在 sd_samplers.samplers_map 中
        if x.lower() not in sd_samplers.samplers_map:
            # 如果不在 samplers_map 中，则抛出运行时错误，显示未知采样器的信息
            raise RuntimeError(f"Unknown sampler: {x}")
# 应用检查点到参数中，根据给定的检查点名称获取最接近的检查点匹配信息
def apply_checkpoint(p, x, xs):
    # 获取最接近的检查点匹配信息
    info = modules.sd_models.get_closet_checkpoint_match(x)
    # 如果没有找到匹配的检查点信息，则抛出运行时错误
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    # 将检查点名称和信息存储到参数的覆盖设置中
    p.override_settings['sd_model_checkpoint'] = info.name


# 确认给定的检查点是否存在
def confirm_checkpoints(p, xs):
    # 遍历所有检查点
    for x in xs:
        # 如果找不到最接近的检查点匹配信息，则抛出运行时错误
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


# 确认给定的检查点是否存在，或者为 None
def confirm_checkpoints_or_none(p, xs):
    # 遍历所有检查点
    for x in xs:
        # 如果检查点为 None、空字符串或字符串 "None"、"none"，则跳过
        if x in (None, "", "None", "none"):
            continue
        # 如果找不到最接近的检查点匹配信息，则抛出运行时错误
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


# 应用剪辑跳过到参数中，设置 CLIP_stop_at_last_layers 为给定值
def apply_clip_skip(p, x, xs):
    opts.data["CLIP_stop_at_last_layers"] = x


# 应用放大潜在空间到参数中，根据给定值设置 use_scale_latent_for_hires_fix
def apply_upscale_latent_space(p, x, xs):
    # 如果给定值不为 '0'，则设置 use_scale_latent_for_hires_fix 为 True，否则为 False
    if x.lower().strip() != '0':
        opts.data["use_scale_latent_for_hires_fix"] = True
    else:
        opts.data["use_scale_latent_for_hires_fix"] = False


# 查找 VAE 模型，根据给定名称返回对应的 VAE 模型
def find_vae(name: str):
    # 如果名称为 'auto' 或 'automatic'，返回未指定的 VAE 模型
    if name.lower() in ['auto', 'automatic']:
        return modules.sd_vae.unspecified
    # 如果名称为 'none'，返回 None
    if name.lower() == 'none':
        return None
    else:
        # 在 VAE 字典中查找包含给定名称的 VAE 模型
        choices = [x for x in sorted(modules.sd_vae.vae_dict, key=lambda x: len(x)) if name.lower().strip() in x.lower()]
        # 如果没有找到匹配的 VAE 模型，则打印提示信息并返回未指定的 VAE 模型
        if len(choices) == 0:
            print(f"No VAE found for {name}; using automatic")
            return modules.sd_vae.unspecified
        else:
            # 返回第一个匹配的 VAE 模型
            return modules.sd_vae.vae_dict[choices[0]]


# 应用 VAE 模型到参数中，重新加载 VAE 权重
def apply_vae(p, x, xs):
    # 重新加载 VAE 权重，根据给定的 VAE 模型名称
    modules.sd_vae.reload_vae_weights(shared.sd_model, vae_file=find_vae(x))


# 应用样式到参数中，将给定的样式字符串拆分并添加到参数的样式列表中
def apply_styles(p: StableDiffusionProcessingTxt2Img, x: str, _):
    p.styles.extend(x.split(','))


# 应用统一 PC 顺序到参数中，设置 uni_pc_order 为给定值和步数减一的最小值
def apply_uni_pc_order(p, x, xs):
    opts.data["uni_pc_order"] = min(x, p.steps - 1)


# 应用人脸恢复到参数中，根据给定值设置人脸恢复模型
def apply_face_restore(p, opt, x):
    opt = opt.lower()
    # 如果值为 'codeformer'，设置人脸恢复模型为 'CodeFormer'
    if opt == 'codeformer':
        is_active = True
        p.face_restoration_model = 'CodeFormer'
    # 如果值为 'gfpgan'，设置人脸恢复模型为 'GFPGAN'
    elif opt == 'gfpgan':
        is_active = True
        p.face_restoration_model = 'GFPGAN'
    # 如果选项为'true', 'yes', 'y', '1'中的任意一个，则将is_active设置为True，否则设置为False
    is_active = opt in ('true', 'yes', 'y', '1')
    
    # 将is_active的值赋给p对象的restore_faces属性
    p.restore_faces = is_active
# 定义一个函数，用于设置字段的覆盖值
def apply_override(field, boolean: bool = False):
    # 定义一个内部函数，根据布尔值来设置字段的值
    def fun(p, x, xs):
        # 如果布尔值为真，则根据输入值设置字段为 True 或 False
        if boolean:
            x = True if x.lower() == "true" else False
        # 将设置好的值赋给字段
        p.override_settings[field] = x
    return fun

# 定义一个函数，用于返回布尔值选择列表
def boolean_choice(reverse: bool = False):
    # 定义一个内部函数，根据布尔值返回不同的布尔值选择列表
    def choice():
        return ["False", "True"] if reverse else ["True", "False"]
    return choice

# 定义一个函数，用于格式化带标签的值
def format_value_add_label(p, opt, x):
    # 如果值的类型为浮点数，则保留小数点后8位
    if type(x) == float:
        x = round(x, 8)
    return f"{opt.label}: {x}"

# 定义一个函数，用于格式化值
def format_value(p, opt, x):
    # 如果值的类型为浮点数，则保留小数点后8位
    if type(x) == float:
        x = round(x, 8)
    return x

# 定义一个函数，用于将列表元素连接成字符串
def format_value_join_list(p, opt, x):
    return ", ".join(x)

# 定义一个函数，什么也不做
def do_nothing(p, x, xs):
    pass

# 定义一个函数，返回空字符串
def format_nothing(p, opt, x):
    return ""

# 定义一个函数，用于格式化移除路径后的值
def format_remove_path(p, opt, x):
    return os.path.basename(x)

# 定义一个虚拟函数，用于指定在 AxisOption 的类型中，当需要获取排列组合列表时
def str_permutations(x):
    return x

# 定义一个函数，将列表转换为 CSV 格式的字符串
def list_to_csv_string(data_list):
    with StringIO() as o:
        csv.writer(o).writerow(data_list)
        return o.getvalue().strip()

# 定义一个函数，将 CSV 格式的字符串转换为列表并去除空格
def csv_string_to_list_strip(data_str):
    return list(map(str.strip, chain.from_iterable(csv.reader(StringIO(data_str))))

# 定义一个类 AxisOption，用于表示轴选项
class AxisOption:
    def __init__(self, label, type, apply, format_value=format_value_add_label, confirm=None, cost=0.0, choices=None, prepare=None):
        self.label = label
        self.type = type
        self.apply = apply
        self.format_value = format_value
        self.confirm = confirm
        self.cost = cost
        self.prepare = prepare
        self.choices = choices

# 定义一个类 AxisOptionImg2Img，继承自 AxisOption，表示图像到图像的轴选项
class AxisOptionImg2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = True

# 定义一个类 AxisOptionTxt2Img，继承自 AxisOption，表示文本到图像的轴选项
class AxisOptionTxt2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = False

# 定义一个轴选项列表
axis_options = [
    AxisOption("Nothing", str, do_nothing, format_value=format_nothing),
    # 创建一个轴选项对象，设置名称为"Seed"，类型为整数，应用于字段"seed"
    AxisOption("Seed", int, apply_field("seed")),
    # 创建一个轴选项对象，设置名称为"Var. seed"，类型为整数，应用于字段"subseed"
    AxisOption("Var. seed", int, apply_field("subseed")),
    # 创建一个轴选项对象，设置名称为"Var. strength"，类型为浮点数，应用于字段"subseed_strength"
    AxisOption("Var. strength", float, apply_field("subseed_strength")),
    # 创建一个轴选项对象，设置名称为"Steps"，类型为整数，应用于字段"steps"
    AxisOption("Steps", int, apply_field("steps")),
    # 创建一个文本到图像的轴选项对象，设置名称为"Hires steps"，类型为整数，应用于字段"hr_second_pass_steps"
    AxisOptionTxt2Img("Hires steps", int, apply_field("hr_second_pass_steps")),
    # 创建一个轴选项对象，设置名称为"CFG Scale"，类型为浮点数，应用于字段"cfg_scale"
    AxisOption("CFG Scale", float, apply_field("cfg_scale")),
    # 创建一个图像到图像的轴选项对象，设置名称为"Image CFG Scale"，类型为浮点数，应用于字段"image_cfg_scale"
    AxisOptionImg2Img("Image CFG Scale", float, apply_field("image_cfg_scale")),
    # 创建一个轴选项对象，设置名称为"Prompt S/R"，类型为字符串，应用于应用提示，格式化值
    AxisOption("Prompt S/R", str, apply_prompt, format_value=format_value),
    # 创建一个轴选项对象，设置名称为"Prompt order"，类型为字符串排列，应用于应用排序，格式化值连接列表
    AxisOption("Prompt order", str_permutations, apply_order, format_value=format_value_join_list),
    # 创建一个文本到图像的轴选项对象，设置名称为"Sampler"，类型为字符串，应用于字段"sampler_name"，格式化值，确认采样器，选择
    AxisOptionTxt2Img("Sampler", str, apply_field("sampler_name"), format_value=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers if x.name not in opts.hide_samplers]),
    # 创建一个文本到图像的轴选项对象，设置名称为"Hires sampler"，类型为字符串，应用于字段"hr_sampler_name"，确认采样器，选择
    AxisOptionTxt2Img("Hires sampler", str, apply_field("hr_sampler_name"), confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img if x.name not in opts.hide_samplers]),
    # 创建一个图像到图像的轴选项对象，设置名称为"Sampler"，类型为字符串，应用于字段"sampler_name"，格式化值，确认采样器，选择
    AxisOptionImg2Img("Sampler", str, apply_field("sampler_name"), format_value=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img if x.name not in opts.hide_samplers]),
    # 创建一个轴选项对象，设置名称为"Checkpoint name"，类型为字符串，应用于应用检查点，格式化值移除路径，确认检查点，成本，选择
    AxisOption("Checkpoint name", str, apply_checkpoint, format_value=format_remove_path, confirm=confirm_checkpoints, cost=1.0, choices=lambda: sorted(sd_models.checkpoints_list, key=str.casefold)),
    # 创建一个轴选项对象，设置名称为"Negative Guidance minimum sigma"，类型为浮点数，应用于字段"s_min_uncond"
    AxisOption("Negative Guidance minimum sigma", float, apply_field("s_min_uncond")),
    # 创建一个轴选项对象，设置名称为"Sigma Churn"，类型为浮点数，应用于字段"s_churn"
    AxisOption("Sigma Churn", float, apply_field("s_churn")),
    # 创建一个轴选项对象，设置名称为"Sigma min"，类型为浮点数，应用于字段"s_tmin"
    AxisOption("Sigma min", float, apply_field("s_tmin")),
    # 创建一个轴选项对象，设置名称为"Sigma max"，类型为浮点数，应用于字段"s_tmax"
    AxisOption("Sigma max", float, apply_field("s_tmax")),
    # 创建一个轴选项对象，设置名称为"Sigma noise"，类型为浮点数，应用于字段"s_noise"
    AxisOption("Sigma noise", float, apply_field("s_noise")),
    # 创建一个轴选项对象，设置名称为"Schedule type"，类型为字符串，应用于覆盖"k_sched_type"，选择
    AxisOption("Schedule type", str, apply_override("k_sched_type"), choices=lambda: list(sd_samplers_kdiffusion.k_diffusion_scheduler)),
    # 创建一个名为"Schedule min sigma"的轴选项，类型为float，应用override函数，参数为"sigma_min"
    AxisOption("Schedule min sigma", float, apply_override("sigma_min")),
    # 创建一个名为"Schedule max sigma"的轴选项，类型为float，应用override函数，参数为"sigma_max"
    AxisOption("Schedule max sigma", float, apply_override("sigma_max")),
    # 创建一个名为"Schedule rho"的轴选项，类型为float，应用override函数，参数为"rho"
    AxisOption("Schedule rho", float, apply_override("rho")),
    # 创建一个名为"Eta"的轴选项，类型为float，应用field函数，参数为"eta"
    AxisOption("Eta", float, apply_field("eta")),
    # 创建一个名为"Clip skip"的轴选项，类型为int，应用clip_skip函数
    AxisOption("Clip skip", int, apply_clip_skip),
    # 创建一个名为"Denoising"的轴选项，类型为float，应用field函数，参数为"denoising_strength"
    AxisOption("Denoising", float, apply_field("denoising_strength")),
    # 创建一个名为"Initial noise multiplier"的轴选项，类型为float，应用field函数，参数为"initial_noise_multiplier"
    AxisOption("Initial noise multiplier", float, apply_field("initial_noise_multiplier")),
    # 创建一个名为"Extra noise"的轴选项，类型为float，应用override函数，参数为"img2img_extra_noise"
    AxisOption("Extra noise", float, apply_override("img2img_extra_noise")),
    # 创建一个名为"Hires upscaler"的轴选项，类型为str，应用field函数，参数为"hr_upscaler"，choices为lambda表达式
    AxisOptionTxt2Img("Hires upscaler", str, apply_field("hr_upscaler"), choices=lambda: [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]),
    # 创建一个名为"Cond. Image Mask Weight"的轴选项，类型为float，应用field函数，参数为"inpainting_mask_weight"
    AxisOptionImg2Img("Cond. Image Mask Weight", float, apply_field("inpainting_mask_weight")),
    # 创建一个名为"VAE"的轴选项，类型为str，应用vae函数，cost为0.7，choices为lambda表达式
    AxisOption("VAE", str, apply_vae, cost=0.7, choices=lambda: ['None'] + list(sd_vae.vae_dict)),
    # 创建一个名为"Styles"的轴选项，类型为str，应用styles函数，choices为lambda表达式
    AxisOption("Styles", str, apply_styles, choices=lambda: list(shared.prompt_styles.styles)),
    # 创建一个名为"UniPC Order"的轴选项，类型为int，应用uni_pc_order函数，cost为0.5
    AxisOption("UniPC Order", int, apply_uni_pc_order, cost=0.5),
    # 创建一个名为"Face restore"的轴选项，类型为str，应用face_restore函数，format_value为format_value
    AxisOption("Face restore", str, apply_face_restore, format_value=format_value),
    # 创建一个名为"Token merging ratio"的轴选项，类型为float，应用override函数，参数为"token_merging_ratio"
    AxisOption("Token merging ratio", float, apply_override('token_merging_ratio')),
    # 创建一个名为"Token merging ratio high-res"的轴选项，类型为float，应用override函数，参数为"token_merging_ratio_hr"
    AxisOption("Token merging ratio high-res", float, apply_override('token_merging_ratio_hr')),
    # 创建一个名为"Always discard next-to-last sigma"的轴选项，类型为str，应用override函数，参数为"always_discard_next_to_last_sigma"，choices为boolean_choice
    AxisOption("Always discard next-to-last sigma", str, apply_override('always_discard_next_to_last_sigma', boolean=True), choices=boolean_choice(reverse=True)),
    # 创建一个名为"SGM noise multiplier"的轴选项，类型为str，应用override函数，参数为"sgm_noise_multiplier"，choices为boolean_choice
    AxisOption("SGM noise multiplier", str, apply_override('sgm_noise_multiplier', boolean=True), choices=boolean_choice(reverse=True)),
    # 创建一个名为"Refiner checkpoint"的轴选项，类型为str，应用field函数，参数为'refiner_checkpoint'，format_value为format_remove_path，confirm为confirm_checkpoints_or_none，cost为1.0，choices为lambda表达式
    AxisOption("Refiner checkpoint", str, apply_field('refiner_checkpoint'), format_value=format_remove_path, confirm=confirm_checkpoints_or_none, cost=1.0, choices=lambda: ['None'] + sorted(sd_models.checkpoints_list, key=str.casefold)),
    # 创建一个名为"Refiner switch at"的轴选项，类型为float，应用field函数，参数为'refiner_switch_at'
    AxisOption("Refiner switch at", float, apply_field('refiner_switch_at')),
    # 创建一个名为"RNG source"的轴选项，类型为字符串，应用覆盖函数"apply_override("randn_source")"，选项为["GPU", "CPU", "NV"]
    AxisOption("RNG source", str, apply_override("randn_source"), choices=lambda: ["GPU", "CPU", "NV"]),
# 定义一个函数，用于绘制三维网格
def draw_xyz_grid(p, xs, ys, zs, x_labels, y_labels, z_labels, cell, draw_legend, include_lone_images, include_sub_grids, first_axes_processed, second_axes_processed, margin_size):
    # 为 x 轴标签创建水平文本列表
    hor_texts = [[images.GridAnnotation(x)] for x in x_labels]
    # 为 y 轴标签创建垂直文本列表
    ver_texts = [[images.GridAnnotation(y)] for y in y_labels]
    # 为 z 轴标签创建标题文本列表
    title_texts = [[images.GridAnnotation(z)] for z in z_labels]

    # 计算列表的大小
    list_size = (len(xs) * len(ys) * len(zs))

    # 初始化处理结果变量
    processed_result = None

    # 计算作业数量并存储在状态对象中
    state.job_count = list_size * p.n_iter
    # 处理给定坐标和索引的细胞数据
    def process_cell(x, y, z, ix, iy, iz):
        # 声明 processed_result 变量为非局部变量
        nonlocal processed_result

        # 定义一个函数，根据给定的索引计算在一维数组中的位置
        def index(ix, iy, iz):
            return ix + iy * len(xs) + iz * len(xs) * len(ys)

        # 更新状态信息，显示当前处理的细胞索引
        state.job = f"{index(ix, iy, iz) + 1} out of {list_size}"

        # 调用 cell 函数处理给定坐标和索引的细胞数据
        processed: Processed = cell(x, y, z, ix, iy, iz)

        # 如果 processed_result 为空，则使用第一个 processed 结果对象作为模板容器来保存完整结果
        if processed_result is None:
            processed_result = copy(processed)
            processed_result.images = [None] * list_size
            processed_result.all_prompts = [None] * list_size
            processed_result.all_seeds = [None] * list_size
            processed_result.infotexts = [None] * list_size
            processed_result.index_of_first_image = 1

        # 计算当前细胞在结果数组中的索引
        idx = index(ix, iy, iz)
        # 如果 processed.images 不为空
        if processed.images:
            # 将第一个 processed 图像存入结果数组
            processed_result.images[idx] = processed.images[0]
            processed_result.all_prompts[idx] = processed.prompt
            processed_result.all_seeds[idx] = processed.seed
            processed_result.infotexts[idx] = processed.infotexts[0]
        else:
            # 如果 processed.images 为空，则创建一个新的图像对象
            cell_mode = "P"
            cell_size = (processed_result.width, processed_result.height)
            if processed_result.images[0] is not None:
                cell_mode = processed_result.images[0].mode
                cell_size = processed_result.images[0].size
            processed_result.images[idx] = Image.new(cell_mode, cell_size)
    # 如果第一个轴已处理为 'x'
    if first_axes_processed == 'x':
        # 遍历 xs 列表，获取索引 ix 和值 x
        for ix, x in enumerate(xs):
            # 如果第二个轴已处理为 'y'
            if second_axes_processed == 'y':
                # 遍历 ys 列表，获取索引 iy 和值 y
                for iy, y in enumerate(ys):
                    # 遍历 zs 列表，获取索引 iz 和值 z
                    for iz, z in enumerate(zs):
                        # 处理单元格，传入 x, y, z 以及它们的索引 ix, iy, iz
                        process_cell(x, y, z, ix, iy, iz)
            else:
                # 遍历 zs 列表，获取索引 iz 和值 z
                for iz, z in enumerate(zs):
                    # 遍历 ys 列表，获取索引 iy 和值 y
                    for iy, y in enumerate(ys):
                        # 处理单元格，传入 x, y, z 以及它们的索引 ix, iy, iz
                        process_cell(x, y, z, ix, iy, iz)
    # 如果第一个轴已处理为 'y'
    elif first_axes_processed == 'y':
        # 遍历 ys 列表，获取索引 iy 和值 y
        for iy, y in enumerate(ys):
            # 如果第二个轴已处理为 'x'
            if second_axes_processed == 'x':
                # 遍历 xs 列表，获取索引 ix 和值 x
                for ix, x in enumerate(xs):
                    # 遍历 zs 列表，获取索引 iz 和值 z
                    for iz, z in enumerate(zs):
                        # 处理单元格，传入 x, y, z 以及它们的索引 ix, iy, iz
                        process_cell(x, y, z, ix, iy, iz)
            else:
                # 遍历 zs 列表，获取索引 iz 和值 z
                for iz, z in enumerate(zs):
                    # 遍历 xs 列表，获取索引 ix 和值 x
                    for ix, x in enumerate(xs):
                        # 处理单元格，传入 x, y, z 以及它们的索引 ix, iy, iz
                        process_cell(x, y, z, ix, iy, iz)
    # 如果第一个轴已处理为 'z'
    elif first_axes_processed == 'z':
        # 遍历 zs 列表，获取索引 iz 和值 z
        for iz, z in enumerate(zs):
            # 如果第二个轴已处理为 'x'
            if second_axes_processed == 'x':
                # 遍历 xs 列表，获取索引 ix 和值 x
                for ix, x in enumerate(xs):
                    # 遍历 ys 列表，获取索引 iy 和值 y
                    for iy, y in enumerate(ys):
                        # 处理单元格，传入 x, y, z 以及它们的索引 ix, iy, iz
                        process_cell(x, y, z, ix, iy, iz)
            else:
                # 遍历 ys 列表，获取索引 iy 和值 y
                for iy, y in enumerate(ys):
                    # 遍历 xs 列表，获取索引 ix 和值 x
                    for ix, x in enumerate(xs):
                        # 处理单元格，传入 x, y, z 以及它们的索引 ix, iy, iz
                        process_cell(x, y, z, ix, iy, iz)

    # 如果没有处理结果
    if not processed_result:
        # 输出错误信息，提示可能需要刷新标签页或重新启动服务
        print("Unexpected error: Processing could not begin, you may need to refresh the tab or restart the service.")
        # 返回处理结果为空的 Processed 对象
        return Processed(p, [])
    # 如果处理结果中没有任何图像
    elif not any(processed_result.images):
        # 输出错误信息，提示 draw_xyz_grid 失败返回任何处理图像
        print("Unexpected error: draw_xyz_grid failed to return even a single processed image")
        # 返回处理结果为空的 Processed 对象
        return Processed(p, [])

    # 计算 zs 列表的长度，即 z 轴的数量
    z_count = len(zs)
    # 遍历 z_count 次
    for i in range(z_count):
        # 计算起始索引
        start_index = (i * len(xs) * len(ys)) + i
        # 计算结束索引
        end_index = start_index + len(xs) * len(ys)
        # 从 processed_result.images 中提取一部分图像，创建图像网格
        grid = images.image_grid(processed_result.images[start_index:end_index], rows=len(ys))
        # 如果需要绘制图例
        if draw_legend:
            # 在图像网格上绘制图例
            grid = images.draw_grid_annotations(grid, processed_result.images[start_index].size[0], processed_result.images[start_index].size[1], hor_texts, ver_texts, margin_size)
        # 将生成的图像网格插入到 processed_result.images 中
        processed_result.images.insert(i, grid)
        # 复制相关数据到新插入的位置
        processed_result.all_prompts.insert(i, processed_result.all_prompts[start_index])
        processed_result.all_seeds.insert(i, processed_result.all_seeds[start_index])
        processed_result.infotexts.insert(i, processed_result.infotexts[start_index])

    # 获取子网格大小
    sub_grid_size = processed_result.images[0].size
    # 创建 z_count 个图像的网格
    z_grid = images.image_grid(processed_result.images[:z_count], rows=1)
    # 如果需要绘制图例
    if draw_legend:
        # 在 z_grid 上绘制图例
        z_grid = images.draw_grid_annotations(z_grid, sub_grid_size[0], sub_grid_size[1], title_texts, [[images.GridAnnotation()]])
    # 将 z_grid 插入到 processed_result.images 的开头
    processed_result.images.insert(0, z_grid)
    # 插入 infotexts 数据到开头位置
    processed_result.infotexts.insert(0, processed_result.infotexts[0])

    # 返回处理后的结果
    return processed_result
# 定义一个类 SharedSettingsStackHelper，用于管理共享设置的堆栈
class SharedSettingsStackHelper(object):
    # 进入上下文时执行的方法
    def __enter__(self):
        # 保存当前的 CLIP_stop_at_last_layers 设置
        self.CLIP_stop_at_last_layers = opts.CLIP_stop_at_last_layers
        # 保存当前的 vae 设置
        self.vae = opts.sd_vae
        # 保存当前的 uni_pc_order 设置
        self.uni_pc_order = opts.uni_pc_order

    # 退出上下文时执行的方法
    def __exit__(self, exc_type, exc_value, tb):
        # 将保存的 vae 设置更新到全局设置中
        opts.data["sd_vae"] = self.vae
        # 将保存的 uni_pc_order 设置更新到全局设置中
        opts.data["uni_pc_order"] = self.uni_pc_order
        # 重新加载模型权重
        modules.sd_models.reload_model_weights()
        # 重新加载 VAE 权重
        modules.sd_vae.reload_vae_weights()

        # 将保存的 CLIP_stop_at_last_layers 设置更新到全局设置中
        opts.data["CLIP_stop_at_last_layers"] = self.CLIP_stop_at_last_layers

# 编译正则表达式，用于匹配整数范围
re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
# 编译正则表达式，用于匹配浮点数范围
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

# 编译正则表达式，用于匹配整数范围和计数
re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*])?\s*")
# 编译正则表达式，用于匹配浮点数范围和计数
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*])?\s*")

# 定义一个类 Script，继承自 scripts.Script 类
class Script(scripts.Script):
    # 定义一个方法 title，返回字符串 "X/Y/Z plot"
    def title(self):
        return "X/Y/Z plot"
```