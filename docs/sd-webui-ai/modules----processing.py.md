# `stable-diffusion-webui\modules\processing.py`

```
# 导入必要的库
from __future__ import annotations
import json
import logging
import math
import os
import sys
import hashlib
from dataclasses import dataclass, field

import torch
import numpy as np
from PIL import Image, ImageOps
import random
import cv2
from skimage import exposure
from typing import Any

# 导入自定义模块
import modules.sd_hijack
from modules import devices, prompt_parser, masking, sd_samplers, lowvram, generation_parameters_copypaste, extra_networks, sd_vae_approx, scripts, sd_samplers_common, sd_unet, errors, rng
from modules.rng import slerp # noqa: F401
from modules.sd_hijack import model_hijack
from modules.sd_samplers_common import images_tensor_to_samples, decode_first_stage, approximation_indexes
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.paths as paths
import modules.face_restoration
import modules.images as images
import modules.styles
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
from ldm.data.util import AddMiDaS
from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion

from einops import repeat, rearrange
from blendmodes.blend import blendLayers, BlendType

# 一些不应更改的选项，因此从选项中删除了它们
opt_C = 4
opt_f = 8

# 设置颜色校正
def setup_color_correction(image):
    logging.info("Calibrating color correction.")
    # 将原始图像转换为LAB颜色空间
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target

# 应用颜色校正
def apply_color_correction(correction, original_image):
    logging.info("Applying color correction.")
    # 将原始图像转换为LAB颜色空间，匹配直方图，然后转换回RGB颜色空间
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(
            np.asarray(original_image),
            cv2.COLOR_RGB2LAB
        ),
        correction,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8"))

    # 使用LUMINOSITY混合模式混合图像
    image = blendLayers(image, original_image, BlendType.LUMINOSITY)

    return image.convert('RGB')
# 应用覆盖层到图像上
def apply_overlay(image, paste_loc, index, overlays):
    # 如果覆盖层为空或者索引超出范围，则返回原图像
    if overlays is None or index >= len(overlays):
        return image

    # 获取当前索引对应的覆盖层
    overlay = overlays[index]

    # 如果有指定粘贴位置
    if paste_loc is not None:
        x, y, w, h = paste_loc
        # 创建一个与覆盖层相同大小的基础图像
        base_image = Image.new('RGBA', (overlay.width, overlay.height))
        # 调整原图像大小
        image = images.resize_image(1, image, w, h)
        # 在基础图像上粘贴原图像
        base_image.paste(image, (x, y))
        image = base_image

    # 将图像转换为 RGBA 模式
    image = image.convert('RGBA')
    # 将覆盖层叠加到图像上
    image.alpha_composite(overlay)
    # 将图像转换为 RGB 模式
    image = image.convert('RGB')

    return image

# 创建二进制掩码
def create_binary_mask(image):
    # 如果图像为 RGBA 模式且不是全白色
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        # 取出 alpha 通道，转换为灰度图像
        image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
    else:
        # 否则将图像转换为灰度图像
        image = image.convert('L')
    return image

# 对图像进行处理以用于文本到图像生成
def txt2img_image_conditioning(sd_model, x, width, height):
    # 如果条件键为 'hybrid' 或 'concat'，即修复模型
    if sd_model.model.conditioning_key in {'hybrid', 'concat'}:
        # 创建一个全为 0.5 的图像条件
        image_conditioning = torch.ones(x.shape[0], 3, height, width, device=x.device) * 0.5
        image_conditioning = images_tensor_to_samples(image_conditioning, approximation_indexes.get(opts.sd_vae_encode_method))

        # 在第一维度上添加一个全为 1 的假掩码
        image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
        image_conditioning = image_conditioning.to(x.dtype)

        return image_conditioning

    # 如果条件键为 'crossattn-adm'，即 UnCLIP 模型
    elif sd_model.model.conditioning_key == "crossattn-adm":
        # 返回一个全为 0 的张量
        return x.new_zeros(x.shape[0], 2*sd_model.noise_augmentor.time_embed.dim, dtype=x.dtype, device=x.device)
    else:
        # 如果不使用修补或解除裁剪模型，则进行虚拟零填充。
        # 仍然会占用一些内存，但不会调用编码器。
        # 我们可以将其制作成一个1x1的图像，因为除了其批量大小外，它不会被使用。
        # 返回一个与输入张量相同数据类型和设备的全零张量，形状为(x.shape[0], 5, 1, 1)
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)
# 定义一个数据类 StableDiffusionProcessing，用于存储稳定扩散处理相关的参数和属性
@dataclass(repr=False)
class StableDiffusionProcessing:
    # 模型对象
    sd_model: object = None
    # 输出样本路径
    outpath_samples: str = None
    # 输出网格路径
    outpath_grids: str = None
    # 提示信息
    prompt: str = ""
    # 用于显示的提示信息
    prompt_for_display: str = None
    # 负面提示信息
    negative_prompt: str = ""
    # 样式列表
    styles: list[str] = None
    # 随机种子
    seed: int = -1
    # 子种子
    subseed: int = -1
    # 子种子强度
    subseed_strength: float = 0
    # 从高度调整种子
    seed_resize_from_h: int = -1
    # 从宽度调整种子
    seed_resize_from_w: int = -1
    # 启用额外生成参数
    seed_enable_extras: bool = True
    # 采样器名称
    sampler_name: str = None
    # 批处理大小
    batch_size: int = 1
    # 迭代次数
    n_iter: int = 1
    # 步数
    steps: int = 50
    # 配置比例
    cfg_scale: float = 7.0
    # 宽度
    width: int = 512
    # 高度
    height: int = 512
    # 恢复面部
    restore_faces: bool = None
    # 平铺
    tiling: bool = None
    # 不保存样本
    do_not_save_samples: bool = False
    # 不保存网格
    do_not_save_grid: bool = False
    # 额外生成参数
    extra_generation_params: dict[str, Any] = None
    # 叠加图像列表
    overlay_images: list = None
    # eta 参数
    eta: float = None
    # 不重新加载嵌入
    do_not_reload_embeddings: bool = False
    # 降噪强度
    denoising_strength: float = None
    # ddim 离散化
    ddim_discretize: str = None
    # s_min_uncond 参数
    s_min_uncond: float = None
    # s_churn 参数
    s_churn: float = None
    # s_tmax 参数
    s_tmax: float = None
    # s_tmin 参数
    s_tmin: float = None
    # s_noise 参数
    s_noise: float = None
    # 覆盖设置
    override_settings: dict[str, Any] = None
    # 执行后恢复覆盖设置
    override_settings_restore_afterwards: bool = True
    # 采样器索引
    sampler_index: int = None
    # 优化器检查点
    refiner_checkpoint: str = None
    # 在指定时间切换优化器
    refiner_switch_at: float = None
    # token 合并比例
    token_merging_ratio = 0
    # token 合并比例（高分辨率）
    token_merging_ratio_hr = 0
    # 禁用额外网络
    disable_extra_networks: bool = False

    # 脚本值
    scripts_value: scripts.ScriptRunner = field(default=None, init=False)
    # 脚本参数值
    script_args_value: list = field(default=None, init=False)
    # 脚本设置完成标志
    scripts_setup_complete: bool = field(default=False, init=False)

    # 缓存无条件
    cached_uc = [None, None]
    # 缓存条件
    cached_c = [None, None]

    # 注释
    comments: dict = None
    # 采样器
    sampler: sd_samplers_common.Sampler | None = field(default=None, init=False)
    # 是否使用修补条件
    is_using_inpainting_conditioning: bool = field(default=False, init=False)
    # 粘贴到
    paste_to: tuple | None = field(default=None, init=False)

    # 是否高分辨率传递
    is_hr_pass: bool = field(default=False, init=False)

    # 条件
    c: tuple = field(default=None, init=False)
    # 定义一个不可变的元组uc，默认值为None，用于存储用户自定义的元组数据
    uc: tuple = field(default=None, init=False)

    # 定义一个ImageRNG对象rng，默认值为None，用于生成图像的随机数生成器
    rng: rng.ImageRNG | None = field(default=None, init=False)
    # 定义一个整数变量step_multiplier，默认值为1，用于控制步长的倍数
    step_multiplier: int = field(default=1, init=False)
    # 定义一个列表color_corrections，默认值为None，用于存储颜色校正数据

    color_corrections: list = field(default=None, init=False)

    # 定义一个列表all_prompts，默认值为None，用于存储所有提示数据
    all_prompts: list = field(default=None, init=False)
    # 定义一个列表all_negative_prompts，默认值为None，用于存储所有负面提示数据
    all_negative_prompts: list = field(default=None, init=False)
    # 定义一个列表all_seeds，默认值为None，用于存储所有种子数据
    all_seeds: list = field(default=None, init=False)
    # 定义一个列表all_subseeds，默认值为None，用于存储所有子种子数据
    all_subseeds: list = field(default=None, init=False)
    # 定义一个整数变量iteration，默认值为0，用于记录迭代次数
    iteration: int = field(default=0, init=False)
    # 定义一个字符串变量main_prompt，默认值为None，用于存储主提示数据
    main_prompt: str = field(default=None, init=False)
    # 定义一个字符串变量main_negative_prompt，默认值为None，用于存储主负面提示数据

    main_negative_prompt: str = field(default=None, init=False)

    # 定义一个列表prompts，默认值为None，用于存储提示数据
    prompts: list = field(default=None, init=False)
    # 定义一个列表negative_prompts，默认值为None，用于存储负面提示数据
    negative_prompts: list = field(default=None, init=False)
    # 定义一个列表seeds，默认值为None，用于存储种子数据
    seeds: list = field(default=None, init=False)
    # 定义一个列表subseeds，默认值为None，用于存储子种子数据
    subseeds: list = field(default=None, init=False)
    # 定义一个字典extra_network_data，默认值为None，用于存储额外的网络数据

    extra_network_data: dict = field(default=None, init=False)

    # 定义一个字符串变量user，默认值为None，用于存储用户数据
    user: str = field(default=None, init=False)

    # 定义一个字符串变量sd_model_name，默认值为None，用于存储模型名称数据
    sd_model_name: str = field(default=None, init=False)
    # 定义一个字符串变量sd_model_hash，默认值为None，用于存储模型哈希数据
    sd_model_hash: str = field(default=None, init=False)
    # 定义一个字符串变量sd_vae_name，默认值为None，用于存储VAE模型名称数据
    sd_vae_name: str = field(default=None, init=False)
    # 定义一个字符串变量sd_vae_hash，默认值为None，用于存储VAE模型哈希数据

    sd_vae_hash: str = field(default=None, init=False)

    # 定义一个布尔变量is_api，默认值为False，用于标识是否为API请求
    is_api: bool = field(default=False, init=False)
    # 初始化方法，用于设置对象的初始状态
    def __post_init__(self):
        # 如果传入的sampler_index参数不为None，则打印警告信息
        if self.sampler_index is not None:
            print("sampler_index argument for StableDiffusionProcessing does not do anything; use sampler_name", file=sys.stderr)

        # 初始化comments属性为一个空字典
        self.comments = {}

        # 如果styles属性为None，则将其初始化为空列表
        if self.styles is None:
            self.styles = []

        # 初始化sampler_noise_scheduler_override为None
        self.sampler_noise_scheduler_override = None
        # 初始化s_min_uncond为传入值或默认值opts.s_min_uncond
        self.s_min_uncond = self.s_min_uncond if self.s_min_uncond is not None else opts.s_min_uncond
        # 初始化s_churn为传入值或默认值opts.s_churn
        self.s_churn = self.s_churn if self.s_churn is not None else opts.s_churn
        # 初始化s_tmin为传入值或默认值opts.s_tmin
        self.s_tmin = self.s_tmin if self.s_tmin is not None else opts.s_tmin
        # 初始化s_tmax为传入值或默认值opts.s_tmax，如果传入值为None，则设为无穷大
        self.s_tmax = (self.s_tmax if self.s_tmax is not None else opts.s_tmax) or float('inf')
        # 初始化s_noise为传入值或默认值opts.s_noise
        self.s_noise = self.s_noise if self.s_noise is not None else opts.s_noise

        # 初始化extra_generation_params为传入值或空字典
        self.extra_generation_params = self.extra_generation_params or {}
        # 初始化override_settings为传入值或空字典
        self.override_settings = self.override_settings or {}
        # 初始化script_args为传入值或空字典
        self.script_args = self.script_args or {}

        # 初始化refiner_checkpoint_info为None
        self.refiner_checkpoint_info = None

        # 如果seed_enable_extras为False，则设置一些属性为默认值
        if not self.seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

        # 初始化cached_uc和cached_c为StableDiffusionProcessing类的属性值
        self.cached_uc = StableDiffusionProcessing.cached_uc
        self.cached_c = StableDiffusionProcessing.cached_c

    # sd_model属性的getter方法，返回shared.sd_model的值
    @property
    def sd_model(self):
        return shared.sd_model

    # sd_model属性的setter方法，不做任何操作
    @sd_model.setter
    def sd_model(self, value):
        pass

    # scripts属性的getter方法，返回scripts_value的值
    @property
    def scripts(self):
        return self.scripts_value

    # scripts属性的setter方法，设置scripts_value的值，并在条件满足时调用setup_scripts方法
    @scripts.setter
    def scripts(self, value):
        self.scripts_value = value

        # 如果scripts_value和script_args_value都存在且scripts_setup_complete为False，则调用setup_scripts方法
        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self.setup_scripts()

    # script_args属性的getter方法，返回script_args_value的值
    @property
    def script_args(self):
        return self.script_args_value

    # script_args属性的setter方法
    # 设置脚本参数值
    def script_args(self, value):
        self.script_args_value = value

        # 如果脚本和脚本参数都存在且脚本设置未完成，则进行脚本设置
        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self.setup_scripts()

    # 设置脚本
    def setup_scripts(self):
        self.scripts_setup_complete = True

        # 设置脚本
        self.scripts.setup_scrips(self, is_ui=not self.is_api)

    # 添加注释
    def comment(self, text):
        self.comments[text] = 1

    # 将文本转换为图像并进行条件处理
    def txt2img_image_conditioning(self, x, width=None, height=None):
        # 检查是否使用了混合或连接的条件键
        self.is_using_inpainting_conditioning = self.sd_model.model.conditioning_key in {'hybrid', 'concat'}

        return txt2img_image_conditioning(self.sd_model, x, width or self.width, height or self.height)

    # 将深度图像进行条件处理
    def depth2img_image_conditioning(self, source_image):
        # 使用AddMiDaS助手将源图像格式化以适应MiDaS模型
        transformer = AddMiDaS(model_type="dpt_hybrid")
        transformed = transformer({"jpg": rearrange(source_image[0], "c h w -> h w c")})
        midas_in = torch.from_numpy(transformed["midas_in"][None, ...]).to(device=shared.device)
        midas_in = repeat(midas_in, "1 ... -> n ...", n=self.batch_size)

        conditioning_image = images_tensor_to_samples(source_image*0.5+0.5, approximation_indexes.get(opts.sd_vae_encode_method))
        conditioning = torch.nn.functional.interpolate(
            self.sd_model.depth_model(midas_in),
            size=conditioning_image.shape[2:],
            mode="bicubic",
            align_corners=False,
        )

        # 对条件进行归一化处理
        (depth_min, depth_max) = torch.aminmax(conditioning)
        conditioning = 2. * (conditioning - depth_min) / (depth_max - depth_min) - 1.
        return conditioning

    # 编辑图像的条件处理
    def edit_image_conditioning(self, source_image):
        conditioning_image = shared.sd_model.encode_first_stage(source_image).mode()

        return conditioning_image
    # 对输入的源图像进行解码，生成条件向量
    def unclip_image_conditioning(self, source_image):
        # 使用 StyleGAN 模型的嵌入器对源图像进行编码，生成条件向量
        c_adm = self.sd_model.embedder(source_image)
        # 如果存在噪声增强器
        if self.sd_model.noise_augmentor is not None:
            # 设置噪声级别为0
            noise_level = 0 # TODO: Allow other noise levels?
            # 使用噪声增强器对条件向量进行增强
            c_adm, noise_level_emb = self.sd_model.noise_augmentor(c_adm, noise_level=repeat(torch.tensor([noise_level]).to(c_adm.device), '1 -> b', b=c_adm.shape[0]))
            # 将噪声级别嵌入到条件向量中
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        # 返回生成的条件向量
        return c_adm

    # 对输入的源图像和潜在图像进行编码，生成条件向量
    def img2img_image_conditioning(self, source_image, latent_image, image_mask=None):
        # 将源图像转换为浮点数类型
        source_image = devices.cond_cast_float(source_image)

        # HACK: 使用内省作为 Depth2Image 模型似乎没有一个所有模型共有的字段来唯一标识自己。条件键也是混合的。
        # 如果模型是 LatentDepth2ImageDiffusion 类型
        if isinstance(self.sd_model, LatentDepth2ImageDiffusion):
            # 调用 depth2img_image_conditioning 方法对源图像进行编码
            return self.depth2img_image_conditioning(source_image)

        # 如果条件阶段键为 "edit"
        if self.sd_model.cond_stage_key == "edit":
            # 调用 edit_image_conditioning 方法对源图像进行编码
            return self.edit_image_conditioning(source_image)

        # 如果采样器的条件键为 {'hybrid', 'concat'}
        if self.sampler.conditioning_key in {'hybrid', 'concat'}:
            # 调用 inpainting_image_conditioning 方法对源图像和潜在图像进行编码
            return self.inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask)

        # 如果采样器的条件键为 "crossattn-adm"
        if self.sampler.conditioning_key == "crossattn-adm":
            # 调用 unclip_image_conditioning 方法对源图像进行编码
            return self.unclip_image_conditioning(source_image)

        # 如果不使用修补或深度模型，则返回全零条件
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    # 初始化方法，接受所有提示、所有种子、所有子种子作为参数
    def init(self, all_prompts, all_seeds, all_subseeds):
        # 空方法，不执行任何操作
        pass

    # 采样方法，接受条件、无条件条件、种子、子种子、子种子强度、提示作为参数
    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        # 抛出未实现错误
        raise NotImplementedError()

    # 关闭方法
    def close(self):
        # 将采样器、条件和无条件条件设置为 None
        self.sampler = None
        self.c = None
        self.uc = None
        # 如果不使用持久条件缓存
        if not opts.persistent_cond_cache:
            # 将缓存的条件和无条件条件设置为 [None, None]
            StableDiffusionProcessing.cached_c = [None, None]
            StableDiffusionProcessing.cached_uc = [None, None]
    # 获取 token 合并比例，根据是否为 HR 模式返回不同的值
    def get_token_merging_ratio(self, for_hr=False):
        # 如果是 HR 模式，返回 HR 模式下的 token 合并比例，否则返回默认值
        if for_hr:
            return self.token_merging_ratio_hr or opts.token_merging_ratio_hr or self.token_merging_ratio or opts.token_merging_ratio

        # 返回默认的 token 合并比例
        return self.token_merging_ratio or opts.token_merging_ratio

    # 设置提示信息
    def setup_prompts(self):
        # 如果提示信息是列表，则将所有提示信息存储在 all_prompts 中
        if isinstance(self.prompt,list):
            self.all_prompts = self.prompt
        # 如果负面提示信息是列表，则将所有负面提示信息存储在 all_prompts 中
        elif isinstance(self.negative_prompt, list):
            self.all_prompts = [self.prompt] * len(self.negative_prompt)
        else:
            # 否则将提示信息重复 batch_size * n_iter 次存储在 all_prompts 中
            self.all_prompts = self.batch_size * self.n_iter * [self.prompt]

        # 如果负面提示信息是列表，则将所有负面提示信息存储在 all_negative_prompts 中
        if isinstance(self.negative_prompt, list):
            self.all_negative_prompts = self.negative_prompt
        else:
            # 否则将负面提示信息重复 len(all_prompts) 次存储在 all_negative_prompts 中
            self.all_negative_prompts = [self.negative_prompt] * len(self.all_prompts)

        # 如果提示信息和负面提示信息的数量不一致，则抛出异常
        if len(self.all_prompts) != len(self.all_negative_prompts):
            raise RuntimeError(f"Received a different number of prompts ({len(self.all_prompts)}) and negative prompts ({len(self.all_negative_prompts)})")

        # 对所有提示信息应用样式
        self.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, self.styles) for x in self.all_prompts]
        # 对所有负面提示信息应用负面样式
        self.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles) for x in self.all_negative_prompts]

        # 设置主提示信息和主负面提示信息
        self.main_prompt = self.all_prompts[0]
        self.main_negative_prompt = self.all_negative_prompts[0]

    # 返回缓存参数，用于在更改时使条件缓存无效
    def cached_params(self, required_prompts, steps, extra_network_data, hires_steps=None, use_old_scheduling=False):
        return (
            required_prompts,
            steps,
            hires_steps,
            use_old_scheduling,
            opts.CLIP_stop_at_last_layers,
            shared.sd_model.sd_checkpoint_info,
            extra_network_data,
            opts.sdxl_crop_left,
            opts.sdxl_crop_top,
            self.width,
            self.height,
        )
    def get_conds_with_caching(self, function, required_prompts, steps, caches, extra_network_data, hires_steps=None):
        """
        Returns the result of calling function(shared.sd_model, required_prompts, steps)
        using a cache to store the result if the same arguments have been used before.

        cache is an array containing two elements. The first element is a tuple
        representing the previously used arguments, or None if no arguments
        have been used before. The second element is where the previously
        computed result is stored.

        caches is a list with items described above.
        """

        # 如果使用旧的调度策略
        if shared.opts.use_old_scheduling:
            # 获取旧的和新的调度提示时间表
            old_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(required_prompts, steps, hires_steps, False)
            new_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(required_prompts, steps, hires_steps, True)
            # 如果旧的和新的调度提示时间表不相同
            if old_schedules != new_schedules:
                # 设置额外生成参数中的"Old prompt editing timelines"为True
                self.extra_generation_params["Old prompt editing timelines"] = True

        # 获取缓存参数
        cached_params = self.cached_params(required_prompts, steps, extra_network_data, hires_steps, shared.opts.use_old_scheduling)

        # 遍历缓存列表
        for cache in caches:
            # 如果缓存参数不为空且与当前缓存参数相同
            if cache[0] is not None and cached_params == cache[0]:
                # 返回缓存结果
                return cache[1]

        # 获取第一个缓存
        cache = caches[0]

        # 使用自动转换上下文
        with devices.autocast():
            # 调用函数计算结果，并存入缓存
            cache[1] = function(shared.sd_model, required_prompts, steps, hires_steps, shared.opts.use_old_scheduling)

        # 更新缓存参数
        cache[0] = cached_params
        # 返回缓存结果
        return cache[1]
    # 设置条件
    def setup_conds(self):
        # 创建正向条件对象
        prompts = prompt_parser.SdConditioning(self.prompts, width=self.width, height=self.height)
        # 创建负向条件对象
        negative_prompts = prompt_parser.SdConditioning(self.negative_prompts, width=self.width, height=self.height, is_negative_prompt=True)

        # 查找采样器配置
        sampler_config = sd_samplers.find_sampler_config(self.sampler_name)
        # 计算总步数
        total_steps = sampler_config.total_steps(self.steps) if sampler_config else self.steps
        # 计算步数倍增器
        self.step_multiplier = total_steps // self.steps
        # 设置第一遍步数
        self.firstpass_steps = total_steps

        # 获取学习条件
        self.uc = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, total_steps, [self.cached_uc], self.extra_network_data)
        # 获取多条件学习条件
        self.c = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, total_steps, [self.cached_c], self.extra_network_data)

    # 获取条件
    def get_conds(self):
        return self.c, self.uc

    # 解析额外网络提示
    def parse_extra_network_prompts(self):
        self.prompts, self.extra_network_data = extra_networks.parse_prompts(self.prompts)

    # 保存样本
    def save_samples(self) -> bool:
        """Returns whether generated images need to be written to disk"""
        return opts.samples_save and not self.do_not_save_samples and (opts.save_incomplete_images or not state.interrupted and not state.skipped)
# 定义一个类 Processed
class Processed:
    # 定义一个方法 js，返回包含对象属性的 JSON 字符串
    def js(self):
        # 创建一个包含对象属性的字典 obj
        obj = {
            "prompt": self.all_prompts[0],  # 提取第一个 all_prompts 属性值
            "all_prompts": self.all_prompts,  # 提取 all_prompts 属性值
            "negative_prompt": self.all_negative_prompts[0],  # 提取第一个 all_negative_prompts 属性值
            "all_negative_prompts": self.all_negative_prompts,  # 提取 all_negative_prompts 属性值
            "seed": self.seed,  # 提取 seed 属性值
            "all_seeds": self.all_seeds,  # 提取 all_seeds 属性值
            "subseed": self.subseed,  # 提取 subseed 属性值
            "all_subseeds": self.all_subseeds,  # 提取 all_subseeds 属性值
            "subseed_strength": self.subseed_strength,  # 提取 subseed_strength 属性值
            "width": self.width,  # 提取 width 属性值
            "height": self.height,  # 提取 height 属性值
            "sampler_name": self.sampler_name,  # 提取 sampler_name 属性值
            "cfg_scale": self.cfg_scale,  # 提取 cfg_scale 属性值
            "steps": self.steps,  # 提取 steps 属性值
            "batch_size": self.batch_size,  # 提取 batch_size 属性值
            "restore_faces": self.restore_faces,  # 提取 restore_faces 属性值
            "face_restoration_model": self.face_restoration_model,  # 提取 face_restoration_model 属性值
            "sd_model_name": self.sd_model_name,  # 提取 sd_model_name 属性值
            "sd_model_hash": self.sd_model_hash,  # 提取 sd_model_hash 属性值
            "sd_vae_name": self.sd_vae_name,  # 提取 sd_vae_name 属性值
            "sd_vae_hash": self.sd_vae_hash,  # 提取 sd_vae_hash 属性值
            "seed_resize_from_w": self.seed_resize_from_w,  # 提取 seed_resize_from_w 属性值
            "seed_resize_from_h": self.seed_resize_from_h,  # 提取 seed_resize_from_h 属性值
            "denoising_strength": self.denoising_strength,  # 提取 denoising_strength 属性值
            "extra_generation_params": self.extra_generation_params,  # 提取 extra_generation_params 属性值
            "index_of_first_image": self.index_of_first_image,  # 提取 index_of_first_image 属性值
            "infotexts": self.infotexts,  # 提取 infotexts 属性值
            "styles": self.styles,  # 提取 styles 属性值
            "job_timestamp": self.job_timestamp,  # 提取 job_timestamp 属性值
            "clip_skip": self.clip_skip,  # 提取 clip_skip 属性值
            "is_using_inpainting_conditioning": self.is_using_inpainting_conditioning,  # 提取 is_using_inpainting_conditioning 属性值
            "version": self.version,  # 提取 version 属性值
        }
        # 返回 obj 的 JSON 字符串表示
        return json.dumps(obj)

    # 定义一个方法 infotext，返回处理后的文本信息
    def infotext(self, p: StableDiffusionProcessing, index):
        # 调用 create_infotext 方法生成文本信息
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], position_in_batch=index % self.batch_size, iteration=index // self.batch_size)
    # 返回 token_merging_ratio_hr 或 token_merging_ratio，取决于 for_hr 参数的取值
    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio_hr if for_hr else self.token_merging_ratio
# 创建随机张量，用于生成随机数据
def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    # 使用给定种子创建图像随机数生成器对象
    g = rng.ImageRNG(shape, seeds, subseeds=subseeds, subseed_strength=subseed_strength, seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
    # 生成下一个随机张量
    return g.next()


# 定义一个类，用于存储解码后的样本
class DecodedSamples(list):
    # 标记已经解码
    already_decoded = True


# 解码潜在空间批次
def decode_latent_batch(model, batch, target_device=None, check_for_nans=False):
    # 创建一个存储解码后样本的对象
    samples = DecodedSamples()

    # 遍历批次中的每个样本
    for i in range(batch.shape[0]):
        # 解码第一阶段的样本
        sample = decode_first_stage(model, batch[i:i + 1])[0]

        # 检查是否存在 NaN 值
        if check_for_nans:
            try:
                # 检查样本是否包含 NaN 值
                devices.test_for_nans(sample, "vae")
            except devices.NansException as e:
                if devices.dtype_vae == torch.float32 or not shared.opts.auto_vae_precision:
                    raise e

                # 打印错误信息
                errors.print_error_explanation(
                    "A tensor with all NaNs was produced in VAE.\n"
                    "Web UI will now convert VAE into 32-bit float and retry.\n"
                    "To disable this behavior, disable the 'Automatically revert VAE to 32-bit floats' setting.\n"
                    "To always start with 32-bit VAE, use --no-half-vae commandline flag."
                )

                # 转换数据类型为 32 位浮点数
                devices.dtype_vae = torch.float32
                model.first_stage_model.to(devices.dtype_vae)
                batch = batch.to(devices.dtype_vae)

                # 重新解码样本
                sample = decode_first_stage(model, batch[i:i + 1])[0]

        # 如果指定了目标设备，则将样本转移到目标设备
        if target_device is not None:
            sample = sample.to(target_device)

        # 将解码后的样本添加到 samples 中
        samples.append(sample)

    # 返回解码后的样本集合
    return samples


# 获取固定的种子值
def get_fixed_seed(seed):
    if seed == '' or seed is None:
        seed = -1
    elif isinstance(seed, str):
        try:
            seed = int(seed)
        except Exception:
            seed = -1

    if seed == -1:
        return int(random.randrange(4294967294))

    return seed


# 修正种子值
def fix_seed(p):
    # 获取固定的种子值
    p.seed = get_fixed_seed(p.seed)
    # 调用函数 get_fixed_seed() 获取固定的种子值，并将其赋值给 p.subseed
    p.subseed = get_fixed_seed(p.subseed)
# 返回程序版本号
def program_version():
    # 导入 launch 模块
    import launch

    # 获取 git 标签
    res = launch.git_tag()
    # 如果标签为 "<none>"，则将 res 设为 None
    if res == "<none>":
        res = None

    # 返回版本号
    return res


# 创建信息文本
def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0, use_main_prompt=False, index=None, all_negative_prompts=None):
    # 如果 index 为 None，则计算 index
    if index is None:
        index = position_in_batch + iteration * p.batch_size

    # 如果 all_negative_prompts 为 None，则使用 p.all_negative_prompts
    if all_negative_prompts is None:
        all_negative_prompts = p.all_negative_prompts

    # 获取 clip_skip 和 enable_hr 参数
    clip_skip = getattr(p, 'clip_skip', opts.CLIP_stop_at_last_layers)
    enable_hr = getattr(p, 'enable_hr', False)
    # 获取 token_merging_ratio 和 token_merging_ratio_hr 参数
    token_merging_ratio = p.get_token_merging_ratio()
    token_merging_ratio_hr = p.get_token_merging_ratio(for_hr=True)

    # 检查是否使用 eta_noise_seed_delta
    uses_ensd = opts.eta_noise_seed_delta != 0
    if uses_ensd:
        uses_ensd = sd_samplers_common.is_sampler_using_eta_noise_seed_delta(p)

    # 生成参数文本
    generation_params_text = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

    # 获取主提示文本和负面提示文本
    prompt_text = p.main_prompt if use_main_prompt else all_prompts[index]
    negative_prompt_text = f"\nNegative prompt: {p.main_negative_prompt if use_main_prompt else all_negative_prompts[index]}" if all_negative_prompts[index] else ""

    # 返回信息文本
    return f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()


# 处理图片
def process_images(p: StableDiffusionProcessing) -> Processed:
    # 如果存在脚本，则在处理之前执行
    if p.scripts is not None:
        p.scripts.before_process(p)

    # 存储选项
    stored_opts = {k: opts.data[k] if k in opts.data else opts.get_default(k) for k in p.override_settings.keys() if k in opts.data}
    try:
        # 尝试执行以下代码块，如果出现异常则执行 finally 块中的代码
        # 如果没有指定检查点覆盖或找不到覆盖检查点，则移除覆盖条目并加载 opts 检查点
        # 如果在运行 refiner 后，refiner 模型没有被卸载 - webui 在这里切换回主模型，如果存在模型覆盖，则之后会重新加载
        if sd_models.checkpoint_aliases.get(p.override_settings.get('sd_model_checkpoint')) is None:
            p.override_settings.pop('sd_model_checkpoint', None)
            sd_models.reload_model_weights()

        # 遍历覆盖设置中的键值对
        for k, v in p.override_settings.items():
            # 设置 opts 中的键值对，is_api=True 表示通过 API 设置，run_callbacks=False 表示不运行回调函数
            opts.set(k, v, is_api=True, run_callbacks=False)

            # 如果键为 'sd_model_checkpoint'，重新加载模型权重
            if k == 'sd_model_checkpoint':
                sd_models.reload_model_weights()

            # 如果键为 'sd_vae'，重新加载 VAE 权重
            if k == 'sd_vae':
                sd_vae.reload_vae_weights()

        # 应用 token 合并到 sd_model 中，使用 p.get_token_merging_ratio() 获取 token 合并比例
        sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())

        # 调用 process_images_inner 函数处理图片
        res = process_images_inner(p)

    finally:
        # 最终执行的代码块，无论是否发生异常都会执行
        # 将 token 合并应用到 sd_model 中，比例为 0
        sd_models.apply_token_merging(p.sd_model, 0)

        # 恢复 opts 到原始状态
        if p.override_settings_restore_afterwards:
            for k, v in stored_opts.items():
                # 设置 opts 对象的属性为原始值
                setattr(opts, k, v)

                # 如果键为 'sd_vae'，重新加载 VAE 权重
                if k == 'sd_vae':
                    sd_vae.reload_vae_weights()

    # 返回处理结果
    return res
def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    # 检查输入的 prompt 是否为列表，如果是则长度必须大于 0
    if isinstance(p.prompt, list):
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    # 执行 torch 的垃圾回收
    devices.torch_gc()

    # 获取固定的随机种子
    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    # 如果 restore_faces 为 None，则使用默认值 opts.face_restoration
    if p.restore_faces is None:
        p.restore_faces = opts.face_restoration

    # 如果 tiling 为 None，则使用默认值 opts.tiling
    if p.tiling is None:
        p.tiling = opts.tiling

    # 如果 refiner_checkpoint 不在指定的值范围内，则获取最接近的检查点信息
    if p.refiner_checkpoint not in (None, "", "None", "none"):
        p.refiner_checkpoint_info = sd_models.get_closet_checkpoint_match(p.refiner_checkpoint)
        if p.refiner_checkpoint_info is None:
            raise Exception(f'Could not find checkpoint with name {p.refiner_checkpoint}')

    # 设置模型名称和哈希值
    p.sd_model_name = shared.sd_model.sd_checkpoint_info.name_for_extra
    p.sd_model_hash = shared.sd_model.sd_model_hash
    p.sd_vae_name = sd_vae.get_loaded_vae_name()
    p.sd_vae_hash = sd_vae.get_loaded_vae_hash()

    # 应用模型劫持，清除注释
    modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    modules.sd_hijack.model_hijack.clear_comments()

    # 设置提示
    p.setup_prompts()

    # 如果 seed 是列表，则使用 seed，否则生成 seed 列表
    if isinstance(seed, list):
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    # 如果 subseed 是列表，则使用 subseed，否则生成 subseed 列表
    if isinstance(subseed, list):
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    # 如果 embeddings_dir 存在且不禁止重新加载 embeddings，则加载文本反演 embeddings
    if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
        model_hijack.embedding_db.load_textual_inversion_embeddings()

    # 如果存在 scripts，则处理
    if p.scripts is not None:
        p.scripts.process(p)

    # 初始化变量
    infotexts = []
    output_images = []

    # 如果不禁用额外网络且存在额外网络数据，则停用额外网络
    if not p.disable_extra_networks and p.extra_network_data:
        extra_networks.deactivate(p, p.extra_network_data)

    # 执行 torch 的垃圾回收
    devices.torch_gc()
    # 创建一个Processed对象，传入参数为p、output_images、p.all_seeds[0]、infotexts[0]、p.all_subseeds[0]、index_of_first_image、infotexts
    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotexts[0],
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )

    # 如果p.scripts不为None，则调用p.scripts.postprocess方法对p和res进行后处理
    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    # 返回Processed对象res
    return res
# 旧算法用于自动计算第一次处理的尺寸
def old_hires_fix_first_pass_dimensions(width, height):
    # 设定期望的像素数量为 512*512
    desired_pixel_count = 512 * 512
    # 计算实际像素数量
    actual_pixel_count = width * height
    # 计算缩放比例
    scale = math.sqrt(desired_pixel_count / actual_pixel_count)
    # 根据缩放比例计算宽度和高度
    width = math.ceil(scale * width / 64) * 64
    height = math.ceil(scale * height / 64) * 64

    return width, height


# 定义一个类，继承自 StableDiffusionProcessing 类
@dataclass(repr=False)
class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    # 是否启用高分辨率处理
    enable_hr: bool = False
    # 降噪强度
    denoising_strength: float = 0.75
    # 第一阶段的宽度和高度
    firstphase_width: int = 0
    firstphase_height: int = 0
    # 高分辨率的缩放比例
    hr_scale: float = 2.0
    # 高分辨率的放大器
    hr_upscaler: str = None
    # 高分辨率第二次处理的步骤数
    hr_second_pass_steps: int = 0
    # 高分辨率调整的 x 和 y 值
    hr_resize_x: int = 0
    hr_resize_y: int = 0
    # 高分辨率的检查点名称和采样器名称
    hr_checkpoint_name: str = None
    hr_sampler_name: str = None
    # 高分辨率的提示和负面提示
    hr_prompt: str = ''
    hr_negative_prompt: str = ''

    # 缓存的高分辨率 uc 和 c 值
    cached_hr_uc = [None, None]
    cached_hr_c = [None, None]

    # 高分辨率的检查点信息
    hr_checkpoint_info: dict = field(default=None, init=False)
    # 高分辨率的放大到 x 和 y 值
    hr_upscale_to_x: int = field(default=0, init=False)
    hr_upscale_to_y: int = field(default=0, init=False)
    # 截断的 x 和 y 值
    truncate_x: int = field(default=0, init=False)
    truncate_y: int = field(default=0, init=False)
    # 应用旧的高分辨率行为到的元组
    applied_old_hires_behavior_to: tuple = field(default=None, init=False)
    # 潜在缩放模式
    latent_scale_mode: dict = field(default=None, init=False)
    # 高分辨率的 c 值
    hr_c: tuple | None = field(default=None, init=False)
    # 高分辨率的 uc 值
    hr_uc: tuple | None = field(default=None, init=False)
    # 所有高分辨率提示的列表
    all_hr_prompts: list = field(default=None, init=False)
    all_hr_negative_prompts: list = field(default=None, init=False)
    # 高分辨率提示的列表
    hr_prompts: list = field(default=None, init=False)
    hr_negative_prompts: list = field(default=None, init=False)
    # 高分辨率额外网络数据的列表
    hr_extra_network_data: list = field(default=None, init=False)
    # 在对象初始化之后执行一些额外的操作
    def __post_init__(self):
        # 调用父类的初始化方法
        super().__post_init__()

        # 如果第一阶段的宽度或高度不为0，则更新属性值
        if self.firstphase_width != 0 or self.firstphase_height != 0:
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height
            self.width = self.firstphase_width
            self.height = self.firstphase_height

        # 设置属性值为预先缓存的高分辨率无条件和有条件
        self.cached_hr_uc = StableDiffusionProcessingTxt2Img.cached_hr_uc
        self.cached_hr_c = StableDiffusionProcessingTxt2Img.cached_hr_c
    # 计算目标分辨率
    def calculate_target_resolution(self):
        # 如果使用旧的高清修复宽高，并且之前应用的旧的高清行为不是当前宽高
        if opts.use_old_hires_fix_width_height and self.applied_old_hires_behavior_to != (self.width, self.height):
            # 设置高清调整的宽高为当前宽高
            self.hr_resize_x = self.width
            self.hr_resize_y = self.height
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height

            # 调用旧的高清修复第一次通行的维度函数，更新宽高
            self.width, self.height = old_hires_fix_first_pass_dimensions(self.width, self.height)
            self.applied_old_hires_behavior_to = (self.width, self.height)

        # 如果高清调整的宽高为0
        if self.hr_resize_x == 0 and self.hr_resize_y == 0:
            # 设置额外生成参数中的"Hires upscale"为高清缩放比例
            self.extra_generation_params["Hires upscale"] = self.hr_scale
            self.hr_upscale_to_x = int(self.width * self.hr_scale)
            self.hr_upscale_to_y = int(self.height * self.hr_scale)
        else:
            # 设置额外生成参数中的"Hires resize"为高清调整的宽高
            self.extra_generation_params["Hires resize"] = f"{self.hr_resize_x}x{self.hr_resize_y}"

            # 根据高清调整的宽高进行计算
            if self.hr_resize_y == 0:
                self.hr_upscale_to_x = self.hr_resize_x
                self.hr_upscale_to_y = self.hr_resize_x * self.height // self.width
            elif self.hr_resize_x == 0:
                self.hr_upscale_to_x = self.hr_resize_y * self.width // self.height
                self.hr_upscale_to_y = self.hr_resize_y
            else:
                target_w = self.hr_resize_x
                target_h = self.hr_resize_y
                src_ratio = self.width / self.height
                dst_ratio = self.hr_resize_x / self.hr_resize_y

                # 根据比例关系进行计算高清调整后的宽高
                if src_ratio < dst_ratio:
                    self.hr_upscale_to_x = self.hr_resize_x
                    self.hr_upscale_to_y = self.hr_resize_x * self.height // self.width
                else:
                    self.hr_upscale_to_x = self.hr_resize_y * self.width // self.height
                    self.hr_upscale_to_y = self.hr_resize_y

                # 计算截断值
                self.truncate_x = (self.hr_upscale_to_x - target_w) // opt_f
                self.truncate_y = (self.hr_upscale_to_y - target_h) // opt_f
    # 定义一个方法用于生成样本，接受多种条件和种子参数
    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        # 创建一个采样器对象
        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)

        # 生成一个随机数
        x = self.rng.next()
        # 使用采样器对象生成样本
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))
        # 删除随机数变量
        del x

        # 如果不启用高分辨率模式，则直接返回生成的样本
        if not self.enable_hr:
            return samples
        # 执行 Torch 的垃圾回收
        devices.torch_gc()

        # 如果潜在尺度模式为 None，则解码生成的样本
        if self.latent_scale_mode is None:
            decoded_samples = torch.stack(decode_latent_batch(self.sd_model, samples, target_device=devices.cpu, check_for_nans=True)).to(dtype=torch.float32)
        else:
            decoded_samples = None

        # 重新加载模型权重
        with sd_models.SkipWritingToConfig():
            sd_models.reload_model_weights(info=self.hr_checkpoint_info)

        # 返回高分辨率样本
        return self.sample_hr_pass(samples, decoded_samples, seeds, subseeds, subseed_strength, prompts)

    # 关闭方法
    def close(self):
        # 调用父类的关闭方法
        super().close()
        # 清空高分辨率条件和无条件条件
        self.hr_c = None
        self.hr_uc = None
        # 如果不使用持久化条件缓存，则清空缓存的高分辨率无条件条件和高分辨率条件
        if not opts.persistent_cond_cache:
            StableDiffusionProcessingTxt2Img.cached_hr_uc = [None, None]
            StableDiffusionProcessingTxt2Img.cached_hr_c = [None, None]
    # 设置提示信息
    def setup_prompts(self):
        # 调用父类的设置提示信息方法
        super().setup_prompts()

        # 如果不启用人力资源提示，则直接返回
        if not self.enable_hr:
            return

        # 如果人力资源提示为空，则使用默认提示信息
        if self.hr_prompt == '':
            self.hr_prompt = self.prompt

        # 如果人力资源负面提示为空，则使用默认负面提示信息
        if self.hr_negative_prompt == '':
            self.hr_negative_prompt = self.negative_prompt

        # 如果人力资源提示是列表，则使用所有提示信息
        if isinstance(self.hr_prompt, list):
            self.all_hr_prompts = self.hr_prompt
        else:
            self.all_hr_prompts = self.batch_size * self.n_iter * [self.hr_prompt]

        # 如果人力资源负面提示是列表，则使用所有负面提示信息
        if isinstance(self.hr_negative_prompt, list):
            self.all_hr_negative_prompts = self.hr_negative_prompt
        else:
            self.all_hr_negative_prompts = self.batch_size * self.n_iter * [self.hr_negative_prompt]

        # 对所有人力资源提示应用样式
        self.all_hr_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, self.styles) for x in self.all_hr_prompts]
        # 对所有人力资源负面提示应用负面样式
        self.all_hr_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles) for x in self.all_hr_negative_prompts]
    # 计算 HR 条件
    def calculate_hr_conds(self):
        # 如果 HR 条件已经计算过，则直接返回
        if self.hr_c is not None:
            return

        # 对 HR 提示进行条件化处理
        hr_prompts = prompt_parser.SdConditioning(self.hr_prompts, width=self.hr_upscale_to_x, height=self.hr_upscale_to_y)
        # 对负面 HR 提示进行条件化处理
        hr_negative_prompts = prompt_parser.SdConditioning(self.hr_negative_prompts, width=self.hr_upscale_to_x, height=self.hr_upscale_to_y, is_negative_prompt=True)

        # 查找采样器配置
        sampler_config = sd_samplers.find_sampler_config(self.hr_sampler_name or self.sampler_name)
        # 获取 HR 第二遍处理的步骤数
        steps = self.hr_second_pass_steps or self.steps
        # 计算总步骤数
        total_steps = sampler_config.total_steps(steps) if sampler_config else steps

        # 使用缓存获取 HR UC 条件
        self.hr_uc = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, hr_negative_prompts, self.firstpass_steps, [self.cached_hr_uc, self.cached_uc], self.hr_extra_network_data, total_steps)
        # 使用缓存获取 HR C 条件
        self.hr_c = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, hr_prompts, self.firstpass_steps, [self.cached_hr_c, self.cached_c], self.hr_extra_network_data, total_steps)
    # 设置条件
    def setup_conds(self):
        # 如果当前处于 HR 通道，则调用来自细化器，不需要设置第一通道条件或切换模型
        if self.is_hr_pass:
            self.hr_c = None
            # 计算 HR 条件
            self.calculate_hr_conds()
            return

        # 调用父类的设置条件方法
        super().setup_conds()

        self.hr_uc = None
        self.hr_c = None

        # 如果启用 HR 并且 HR 检查点信息为 None
        if self.enable_hr and self.hr_checkpoint_info is None:
            # 如果启用 hires_fix_use_firstpass_conds
            if shared.opts.hires_fix_use_firstpass_conds:
                self.calculate_hr_conds()

            # 如果处于低 VRAM 模式，并且 SD 模型的检查点信息等于选择的检查点
            elif lowvram.is_enabled(shared.sd_model) and shared.sd_model.sd_checkpoint_info == sd_models.select_checkpoint():
                # 激活额外网络
                with devices.autocast():
                    extra_networks.activate(self, self.hr_extra_network_data)

                # 计算 HR 条件
                self.calculate_hr_conds()

                with devices.autocast():
                    extra_networks.activate(self, self.extra_network_data)

    # 获取条件
    def get_conds(self):
        # 如果当前处于 HR 通道
        if self.is_hr_pass:
            return self.hr_c, self.hr_uc

        return super().get_conds()

    # 解析额外网络提示
    def parse_extra_network_prompts(self):
        # 调用父类的解析额外网络提示方法
        res = super().parse_extra_network_prompts()

        # 如果启用 HR
        if self.enable_hr:
            # 获取 HR 提示和负面提示
            self.hr_prompts = self.all_hr_prompts[self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]
            self.hr_negative_prompts = self.all_hr_negative_prompts[self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]

            # 解析 HR 提示
            self.hr_prompts, self.hr_extra_network_data = extra_networks.parse_prompts(self.hr_prompts)

        return res
# 定义一个类 StableDiffusionProcessingImg2Img，继承自 StableDiffusionProcessing，并且不会在 repr 中显示
@dataclass(repr=False)
class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    # 初始化属性
    init_images: list = None
    resize_mode: int = 0
    denoising_strength: float = 0.75
    image_cfg_scale: float = None
    mask: Any = None
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    mask_blur: int = None
    inpainting_fill: int = 0
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 0
    inpainting_mask_invert: int = 0
    initial_noise_multiplier: float = None
    latent_mask: Image = None

    # 初始化属性，不会在初始化时设置
    image_mask: Any = field(default=None, init=False)

    # 初始化属性，不会在初始化时设置
    nmask: torch.Tensor = field(default=None, init=False)
    # 初始化属性，不会在初始化时设置
    image_conditioning: torch.Tensor = field(default=None, init=False)
    # 初始化属性，不会在初始化时设置
    init_img_hash: str = field(default=None, init=False)
    # 初始化属性，不会在初始化时设置
    mask_for_overlay: Image = field(default=None, init=False)
    # 初始化属性，不会在初始化时设置
    init_latent: torch.Tensor = field(default=None, init=False)

    # 初始化方法，在初始化后调用父类的初始化方法
    def __post_init__(self):
        super().__post_init__()

        # 将 self.mask 赋值给 self.image_mask，并将 self.mask 设置为 None
        self.image_mask = self.mask
        self.mask = None
        # 如果 self.initial_noise_multiplier 为 None，则将其设置为 opts.initial_noise_multiplier
        self.initial_noise_multiplier = opts.initial_noise_multiplier if self.initial_noise_multiplier is None else self.initial_noise_multiplier

    # 定义 mask_blur 属性的 getter 方法
    @property
    def mask_blur(self):
        # 如果 mask_blur_x 等于 mask_blur_y，则返回 mask_blur_x，否则返回 None
        if self.mask_blur_x == self.mask_blur_y:
            return self.mask_blur_x
        return None

    # 定义 mask_blur 属性的 setter 方法
    @mask_blur.setter
    def mask_blur(self, value):
        # 如果 value 是整数，则将其赋值给 mask_blur_x 和 mask_blur_y
        if isinstance(value, int):
            self.mask_blur_x = value
            self.mask_blur_y = value
    # 定义一个方法用于生成样本，接受多个参数：conditioning，unconditional_conditioning，seeds，subseeds，subseed_strength，prompts
    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        # 生成一个随机数
        x = self.rng.next()

        # 如果初始噪声乘数不为1.0，则将"Noise multiplier"参数添加到额外生成参数中，并将随机数乘以初始噪声乘数
        if self.initial_noise_multiplier != 1.0:
            self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
            x *= self.initial_noise_multiplier

        # 使用sampler对象的sample_img2img方法生成样本，传入参数self，self.init_latent，x，conditioning，unconditional_conditioning，image_conditioning=self.image_conditioning
        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)

        # 如果存在mask，则将样本乘以nmask并加上init_latent乘以mask
        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask

        # 删除变量x
        del x
        # 执行torch的垃圾回收
        devices.torch_gc()

        # 返回生成的样本
        return samples

    # 定义一个方法用于获取token合并比例，接受一个参数for_hr，默认为False
    def get_token_merging_ratio(self, for_hr=False):
        # 返回token_merging_ratio，如果override_settings中包含"token_merging_ratio"并且opts中包含token_merging_ratio，则返回opts中的token_merging_ratio，否则返回opts中的token_merging_ratio_img2img或opts中的token_merging_ratio
        return self.token_merging_ratio or ("token_merging_ratio" in self.override_settings and opts.token_merging_ratio) or opts.token_merging_ratio_img2img or opts.token_merging_ratio
```