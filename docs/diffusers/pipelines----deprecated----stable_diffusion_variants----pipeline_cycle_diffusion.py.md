# `.\diffusers\pipelines\deprecated\stable_diffusion_variants\pipeline_cycle_diffusion.py`

```py
# 版权声明，表明版权归 HuggingFace 团队所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证授权（“许可证”）；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，
# 否则根据许可证分发的软件是按“原样”提供的，
# 不附带任何形式的保证或条件，无论是明示或暗示的。
# 请参见许可证以了解管理权限和限制的具体条款。

# 导入 inspect 模块，用于获取对象的信息
import inspect
# 从 typing 模块导入类型提示所需的类型
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 numpy 库，用于数值计算和数组操作
import numpy as np
# 导入 PIL.Image，用于图像处理
import PIL.Image
# 导入 torch 库，提供深度学习功能
import torch
# 从 packaging 导入版本控制功能
from packaging import version
# 从 transformers 导入 CLIP 相关的类
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

# 从相对路径导入 FrozenDict 类
from ....configuration_utils import FrozenDict
# 从相对路径导入图像处理相关类
from ....image_processor import PipelineImageInput, VaeImageProcessor
# 从相对路径导入加载器类
from ....loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 从相对路径导入模型类
from ....models import AutoencoderKL, UNet2DConditionModel
# 从相对路径导入 Lora 调整函数
from ....models.lora import adjust_lora_scale_text_encoder
# 从相对路径导入调度器类
from ....schedulers import DDIMScheduler
# 从相对路径导入常用工具函数和变量
from ....utils import PIL_INTERPOLATION, USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
# 从相对路径导入随机数生成相关的工具函数
from ....utils.torch_utils import randn_tensor
# 从相对路径导入扩散管道类
from ...pipeline_utils import DiffusionPipeline
# 从相对路径导入稳定扩散的输出类
from ...stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
# 从相对路径导入稳定扩散的安全检查器类
from ...stable_diffusion.safety_checker import StableDiffusionSafetyChecker

# 创建一个日志记录器实例，用于记录模块中的日志
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img 中复制的预处理函数
def preprocess(image):
    # 定义弃用信息，提示用户该方法将在未来版本中被移除
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    # 调用弃用函数，记录警告信息
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    # 如果输入是一个张量，则直接返回
    if isinstance(image, torch.Tensor):
        return image
    # 如果输入是 PIL 图像，则将其放入列表中
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    # 如果列表中的第一个元素是 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        # 获取图像的宽和高
        w, h = image[0].size
        # 将宽和高调整为 8 的整数倍
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        # 对每个图像进行调整大小，并转换为数组格式
        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        # 将图像数组沿第一个维度拼接
        image = np.concatenate(image, axis=0)
        # 将像素值归一化到 [0, 1] 范围，并转换为 float32 类型
        image = np.array(image).astype(np.float32) / 255.0
        # 调整数组维度顺序，从 (N, H, W, C) 转换为 (N, C, H, W)
        image = image.transpose(0, 3, 1, 2)
        # 将像素值从 [0, 1] 线性映射到 [-1, 1]
        image = 2.0 * image - 1.0
        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
    # 如果列表中的第一个元素是张量，则在第 0 维拼接这些张量
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    # 返回处理后的图像
    return image

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img 中复制的检索潜在变量函数
def retrieve_latents(
    # 输入的编码器输出，类型为 torch.Tensor
    encoder_output: torch.Tensor, 
    # 可选的随机数生成器
    generator: Optional[torch.Generator] = None, 
    # 采样模式，默认为 "sample"
    sample_mode: str = "sample"
):
    # 检查 encoder_output 是否具有 "latent_dist" 属性，并且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从 latent_dist 中随机采样并返回生成器生成的样本
        return encoder_output.latent_dist.sample(generator)
    # 检查 encoder_output 是否具有 "latent_dist" 属性，并且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回 latent_dist 的众数作为结果
        return encoder_output.latent_dist.mode()
    # 检查 encoder_output 是否具有 "latents" 属性
    elif hasattr(encoder_output, "latents"):
        # 返回 latents 属性的值
        return encoder_output.latents
    # 如果以上条件都不满足，抛出属性错误
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
# 后验采样函数，生成前一时刻的潜变量
def posterior_sample(scheduler, latents, timestep, clean_latents, generator, eta):
    # 1. 获取前一步的时间值（t-1）
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 如果前一步时间值小于等于0，返回干净的潜变量
    if prev_timestep <= 0:
        return clean_latents

    # 2. 计算 alpha 和 beta 值
    alpha_prod_t = scheduler.alphas_cumprod[timestep]  # 当前时间步的累积 alpha
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )  # 前一步时间步的累积 alpha

    # 获取当前和前一步的方差
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)  # 计算标准差

    # 指向 x_t 的方向
    e_t = (latents - alpha_prod_t ** (0.5) * clean_latents) / (1 - alpha_prod_t) ** (0.5)
    dir_xt = (1.0 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * e_t  # 计算 x_t 的方向
    # 生成噪声
    noise = std_dev_t * randn_tensor(
        clean_latents.shape, dtype=clean_latents.dtype, device=clean_latents.device, generator=generator
    )
    # 计算前一步的潜变量
    prev_latents = alpha_prod_t_prev ** (0.5) * clean_latents + dir_xt + noise

    # 返回前一步的潜变量
    return prev_latents


# 计算噪声的函数
def compute_noise(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    # 1. 获取前一步的时间值（t-1）
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 2. 计算 alpha 和 beta 值
    alpha_prod_t = scheduler.alphas_cumprod[timestep]  # 当前时间步的累积 alpha
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )  # 前一步时间步的累积 alpha

    beta_prod_t = 1 - alpha_prod_t  # 计算 beta 值

    # 3. 根据预测的噪声计算预测的原始样本
    pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

    # 4. 对预测的原始样本进行剪辑
    if scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)  # 限制值范围在[-1, 1]之间

    # 5. 计算方差
    variance = scheduler._get_variance(timestep, prev_timestep)  # 获取方差
    std_dev_t = eta * variance ** (0.5)  # 计算标准差

    # 6. 计算指向 x_t 的方向
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

    # 计算噪声
    noise = (prev_latents - (alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction)) / (
        variance ** (0.5) * eta
    )
    return noise


# 文本引导图像生成的扩散管道类
class CycleDiffusionPipeline(DiffusionPipeline, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin):
    r"""
    基于文本引导的图像生成管道，使用稳定扩散模型。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档以获取所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。
    # 管道继承以下加载方法：
        # - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        # - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        # - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
    
    # 参数说明：
        # vae ([`AutoencoderKL`]）：变分自编码器模型，用于将图像编码和解码为潜在表示。
        # text_encoder ([`~transformers.CLIPTextModel`]）：冻结的文本编码器，使用 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)。
        # tokenizer ([`~transformers.CLIPTokenizer`]）：用于文本标记化的 `CLIPTokenizer`。
        # unet ([`UNet2DConditionModel`]）：用于去噪编码图像潜在的 `UNet2DConditionModel`。
        # scheduler ([`SchedulerMixin`]）：与 `unet` 结合使用以去噪编码图像潜在的调度器。只能是 [`DDIMScheduler`] 的实例。
        # safety_checker ([`StableDiffusionSafetyChecker`]）：分类模块，估计生成的图像是否可能被认为是冒犯性或有害的。
        # 请参考 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 获取更多关于模型潜在危害的详细信息。
        # feature_extractor ([`~transformers.CLIPImageProcessor`]）：用于从生成图像中提取特征的 `CLIPImageProcessor`；作为输入用于 `safety_checker`。
    
    # 定义模型在 CPU 上的卸载顺序
        model_cpu_offload_seq = "text_encoder->unet->vae"
        # 定义可选组件列表
        _optional_components = ["safety_checker", "feature_extractor"]
    
        # 初始化方法，定义构造函数的参数
        def __init__(
            self,
            vae: AutoencoderKL,  # 变分自编码器
            text_encoder: CLIPTextModel,  # 文本编码器
            tokenizer: CLIPTokenizer,  # 文本标记器
            unet: UNet2DConditionModel,  # 去噪模型
            scheduler: DDIMScheduler,  # 调度器
            safety_checker: StableDiffusionSafetyChecker,  # 安全检查器
            feature_extractor: CLIPImageProcessor,  # 特征提取器
            requires_safety_checker: bool = True,  # 是否需要安全检查器的布尔值
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制
        def _encode_prompt(
            self,
            prompt,  # 输入的提示信息
            device,  # 设备信息（CPU/GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器引导
            negative_prompt=None,  # 可选的负面提示
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            lora_scale: Optional[float] = None,  # 可选的 LoRA 缩放因子
            **kwargs,  # 其他关键字参数
    # 定义函数结束的括号
        ):
            # 设置弃用警告信息，提示用户该方法将在未来版本中被移除
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用弃用函数，记录弃用信息，版本为 1.0.0，且不使用标准警告
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用 encode_prompt 方法，生成提示嵌入元组
            prompt_embeds_tuple = self.encode_prompt(
                # 传入提示内容
                prompt=prompt,
                # 传入设备信息
                device=device,
                # 每个提示生成的图像数量
                num_images_per_prompt=num_images_per_prompt,
                # 是否执行无分类器引导
                do_classifier_free_guidance=do_classifier_free_guidance,
                # 传入负提示内容
                negative_prompt=negative_prompt,
                # 传入提示嵌入（可选）
                prompt_embeds=prompt_embeds,
                # 传入负提示嵌入（可选）
                negative_prompt_embeds=negative_prompt_embeds,
                # 传入 LoRA 比例（可选）
                lora_scale=lora_scale,
                # 传入额外参数（可变参数）
                **kwargs,
            )
    
            # 将提示嵌入元组中的两个部分进行连接，以兼容旧版
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回最终的提示嵌入
            return prompt_embeds
    
        # 定义 encode_prompt 方法，来自稳定扩散管道
        # 该方法用于编码提示信息
        def encode_prompt(
            self,
            # 提示内容
            prompt,
            # 设备信息
            device,
            # 每个提示生成的图像数量
            num_images_per_prompt,
            # 是否执行无分类器引导
            do_classifier_free_guidance,
            # 负提示内容（可选）
            negative_prompt=None,
            # 提示嵌入（可选）
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入（可选）
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # LoRA 比例（可选）
            lora_scale: Optional[float] = None,
            # 跳过的剪辑（可选）
            clip_skip: Optional[int] = None,
        # 定义 check_inputs 方法，检查输入参数
        def check_inputs(
            self,
            # 提示内容
            prompt,
            # 强度参数
            strength,
            # 回调步骤
            callback_steps,
            # 负提示内容（可选）
            negative_prompt=None,
            # 提示嵌入（可选）
            prompt_embeds=None,
            # 负提示嵌入（可选）
            negative_prompt_embeds=None,
            # 在步骤结束时的回调张量输入（可选）
            callback_on_step_end_tensor_inputs=None,
    ):
        # 检查 strength 的值是否在 [0.0, 1.0] 范围内，若不在则引发 ValueError
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # 检查 callback_steps 是否为正整数，若不是则引发 ValueError
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查 callback_on_step_end_tensor_inputs 中的所有键是否都在 _callback_tensor_inputs 中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        # 检查是否同时提供了 prompt 和 prompt_embeds，若是则引发 ValueError
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否两个都未提供，若是则引发 ValueError
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 的类型是否为 str 或 list，若不是则引发 ValueError
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了 negative_prompt 和 negative_prompt_embeds，若是则引发 ValueError
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 的形状是否一致，若不一致则引发 ValueError
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 复制自 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    # 准备调度器步骤的额外参数，因为并不是所有调度器都有相同的参数签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅用于 DDIMScheduler，其他调度器将忽略该参数
        # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
        # 并且应该在 [0, 1] 范围内

        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数的字典
        extra_step_kwargs = {}
        # 如果接受 eta 参数，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator 参数，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制
    def run_safety_checker(self, image, device, dtype):
        # 如果没有安全检查器，则设置 NSFW 概念为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果图像是张量格式
            if torch.is_tensor(image):
                # 将张量后处理为 PIL 格式以供特征提取器使用
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果图像不是张量，将其转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 将处理后的图像输入特征提取器，并转移到指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 使用安全检查器对图像进行检查并获取 NSFW 概念
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回检查后的图像和 NSFW 概念的状态
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制
    def decode_latents(self, latents):
        # 提示：decode_latents 方法已弃用，并将在 1.0.0 中删除，请使用 VaeImageProcessor.postprocess(...) 代替
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 发出弃用警告
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 将潜变量按缩放因子进行调整
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜变量以获得图像，并返回第一个元素
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像值归一化到 [0, 1] 范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为 float32 格式以确保兼容性
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回解码后的图像
        return image

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps 复制
    # 定义获取时间步长的方法，接收推理步数、强度和设备作为参数
    def get_timesteps(self, num_inference_steps, strength, device):
        # 计算初始时间步长，取 num_inference_steps 和 strength 的乘积的最小值
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
        # 计算开始时间步，确保不小于零
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从调度器中获取时间步，基于 t_start 和调度器的顺序
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        # 如果调度器有设置开始索引的方法，则调用该方法
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
    
        # 返回时间步和剩余推理步数
        return timesteps, num_inference_steps - t_start
    # 准备潜在向量，进行图像处理
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        # 将图像移动到指定设备并转换为指定数据类型
        image = image.to(device=device, dtype=dtype)
    
        # 获取图像的批次大小
        batch_size = image.shape[0]
    
        # 检查图像的通道数是否为4
        if image.shape[1] == 4:
            # 如果是4通道图像，初始化潜在向量为图像本身
            init_latents = image
    
        else:
            # 检查生成器是否是列表且其长度与批次大小不匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    # 抛出值错误，提示生成器长度与请求的批次大小不匹配
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果生成器是列表
            if isinstance(generator, list):
                # 使用每个生成器和对应图像获取潜在向量
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(image.shape[0])
                ]
                # 将所有潜在向量拼接成一个张量
                init_latents = torch.cat(init_latents, dim=0)
            else:
                # 使用单个生成器获取潜在向量
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)
    
            # 对初始化的潜在向量进行缩放
            init_latents = self.vae.config.scaling_factor * init_latents
    
        # 检查批次大小是否大于初始化潜在向量的数量且可以整除
        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # 扩展初始化潜在向量以匹配批次大小
            deprecation_message = (
                # 创建弃用消息，提醒用户初始图像的数量与文本提示不匹配
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            # 发出弃用警告
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            # 计算每个提示所需的附加图像数量
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            # 扩展潜在向量以匹配文本提示的数量
            init_latents = torch.cat([init_latents] * additional_image_per_prompt * num_images_per_prompt, dim=0)
        # 如果批次大小大于初始化潜在向量的数量且不能整除
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                # 抛出值错误，提示无法将图像复制到文本提示
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            # 如果批次大小小于等于初始化潜在向量数量，则复制潜在向量
            init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)
    
        # 使用时间步长向潜在向量添加噪声
        shape = init_latents.shape
        # 生成与潜在向量相同形状的随机噪声
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    
        # 获取清晰的潜在向量
        clean_latents = init_latents
        # 将噪声添加到初始化潜在向量中
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        # 结果潜在向量
        latents = init_latents
    
        # 返回处理后的潜在向量和清晰的潜在向量
        return latents, clean_latents
    
    # 禁用梯度计算，节省内存
    @torch.no_grad()
    # 定义可调用的类方法，用于生成图像
        def __call__(
            self,
            # 输入提示，可以是字符串或字符串列表
            prompt: Union[str, List[str]],
            # 源提示，可以是字符串或字符串列表
            source_prompt: Union[str, List[str]],
            # 输入图像，可选参数，默认为 None
            image: PipelineImageInput = None,
            # 强度参数，默认为 0.8
            strength: float = 0.8,
            # 推理步骤数量，默认为 50
            num_inference_steps: Optional[int] = 50,
            # 指导比例，默认为 7.5
            guidance_scale: Optional[float] = 7.5,
            # 源指导比例，默认为 1
            source_guidance_scale: Optional[float] = 1,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 噪声参数，默认为 0.1
            eta: Optional[float] = 0.1,
            # 随机数生成器，可以是单个或多个生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 预计算的提示嵌入，可选参数
            prompt_embeds: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式，默认为 True
            return_dict: bool = True,
            # 回调函数，可选参数，接收进度信息
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调调用的步数，默认为 1
            callback_steps: int = 1,
            # 交叉注意力的额外参数，可选
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 跳过的剪辑层数，可选
            clip_skip: Optional[int] = None,
```