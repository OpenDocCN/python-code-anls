# `.\diffusers\pipelines\stable_diffusion\pipeline_stable_diffusion_instruct_pix2pix.py`

```py
# 版权声明，包含版权信息和许可证使用条款
# Copyright 2024 The InstructPix2Pix Authors and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版授权
# 您不得在不遵守许可证的情况下使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件在许可证下分发
# 是基于“按现状”原则，没有任何明示或暗示的担保或条件。
# 查看许可证以获取特定的权限和限制。

import inspect  # 导入 inspect 模块，用于获取有关活跃对象的信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示相关的类

import numpy as np  # 导入 numpy，常用的数值计算库
import PIL.Image  # 导入 PIL.Image，处理图像的库
import torch  # 导入 PyTorch，深度学习框架
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 导入 transformers 库中的 CLIP 相关类

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 导入回调类
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入图像处理相关类
from ...loaders import IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入加载器混合类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 导入模型类
from ...schedulers import KarrasDiffusionSchedulers  # 导入调度器类
from ...utils import PIL_INTERPOLATION, deprecate, logging  # 导入实用工具类
from ...utils.torch_utils import randn_tensor  # 导入随机张量生成工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入管道工具类
from . import StableDiffusionPipelineOutput  # 导入管道输出类
from .safety_checker import StableDiffusionSafetyChecker  # 导入安全检查器类

logger = logging.get_logger(__name__)  # 创建一个日志记录器，用于记录信息，禁用 pylint 名称无效警告

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess 复制的函数
def preprocess(image):
    # 设置弃用警告消息，指示预处理方法已弃用
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    # 调用弃用函数，显示警告
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    # 如果输入是张量，则直接返回
    if isinstance(image, torch.Tensor):
        return image
    # 如果输入是 PIL 图像，则将其封装在列表中
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    # 如果列表中的第一个元素是 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size  # 获取图像的宽和高
        # 将宽高调整为 8 的整数倍
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        # 将图像调整为新的宽高，并转换为 NumPy 数组，增加一个维度
        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        # 将所有图像沿第0维合并
        image = np.concatenate(image, axis=0)
        # 将数组转换为 float32 类型并归一化到 [0, 1] 范围
        image = np.array(image).astype(np.float32) / 255.0
        # 调整数组维度顺序，从 (N, H, W, C) 转为 (N, C, H, W)
        image = image.transpose(0, 3, 1, 2)
        # 将像素值范围调整到 [-1, 1]
        image = 2.0 * image - 1.0
        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
    # 如果输入是张量列表
    elif isinstance(image[0], torch.Tensor):
        # 将多个张量沿第0维合并
        image = torch.cat(image, dim=0)
    # 返回处理后的图像
    return image


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    # 检查 encoder_output 是否具有 "latent_dist" 属性，并且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从 latent_dist 中进行采样，使用指定的生成器
        return encoder_output.latent_dist.sample(generator)
    # 检查 encoder_output 是否具有 "latent_dist" 属性，并且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回 latent_dist 的众数
        return encoder_output.latent_dist.mode()
    # 检查 encoder_output 是否具有 "latents" 属性
    elif hasattr(encoder_output, "latents"):
        # 返回 latents 属性的值
        return encoder_output.latents
    # 如果以上条件都不满足，则抛出 AttributeError
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
# 定义一个用于像素级图像编辑的管道类，继承多个混合类以实现功能
class StableDiffusionInstructPix2PixPipeline(
    # 继承 DiffusionPipeline 类以获得基础功能
    DiffusionPipeline,
    # 继承 StableDiffusionMixin 以获取稳定扩散相关功能
    StableDiffusionMixin,
    # 继承 TextualInversionLoaderMixin 以支持文本反转加载
    TextualInversionLoaderMixin,
    # 继承 StableDiffusionLoraLoaderMixin 以支持 LoRA 权重加载和保存
    StableDiffusionLoraLoaderMixin,
    # 继承 IPAdapterMixin 以支持 IP 适配器加载
    IPAdapterMixin,
):
    r"""
    管道用于通过遵循文本指令进行像素级图像编辑（基于稳定扩散）。

    该模型从 [`DiffusionPipeline`] 继承。有关所有管道实现的通用方法（下载、保存、在特定设备上运行等）的文档，请查看超类文档。

    此管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    参数：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于对文本进行分词的 `CLIPTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于去噪编码图像潜在表示的 `UNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用的调度器，用于去噪编码图像潜在表示。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            分类模块，估计生成的图像是否可能被认为是冒犯性或有害的。
            有关模型潜在危害的更多详细信息，请参阅 [模型卡](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            用于从生成的图像中提取特征的 `CLIPImageProcessor`；作为输入用于 `safety_checker`。
    """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件列表
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义从 CPU 卸载时排除的组件
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义回调张量输入
    _callback_tensor_inputs = ["latents", "prompt_embeds", "image_latents"]

    # 构造函数，初始化管道所需的各个组件
    def __init__(
        # 初始化变分自编码器（VAE）
        self,
        vae: AutoencoderKL,
        # 初始化文本编码器
        text_encoder: CLIPTextModel,
        # 初始化分词器
        tokenizer: CLIPTokenizer,
        # 初始化去噪模型
        unet: UNet2DConditionModel,
        # 初始化调度器
        scheduler: KarrasDiffusionSchedulers,
        # 初始化安全检查器
        safety_checker: StableDiffusionSafetyChecker,
        # 初始化特征提取器
        feature_extractor: CLIPImageProcessor,
        # 可选图像编码器
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
        # 是否需要安全检查器的标志
        requires_safety_checker: bool = True,
    # 初始化父类
        ):
            super().__init__()
    
            # 检查安全检查器是否为 None，并且需要安全检查器时发出警告
            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    # 警告信息，提醒用户禁用安全检查器的风险和使用条款
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 检查安全检查器不为 None 时，特征提取器必须定义
            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    # 抛出异常，提示用户需要定义特征提取器
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )
    
            # 注册各个模块，以便后续使用
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
            )
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            # 将配置注册到类中，指明是否需要安全检查器
            self.register_to_config(requires_safety_checker=requires_safety_checker)
    
        # 装饰器，指示该函数不需要计算梯度
        @torch.no_grad()
        def __call__(
            # 输入参数，包括提示文本、图像、推理步骤等
            prompt: Union[str, List[str]] = None,
            image: PipelineImageInput = None,
            num_inference_steps: int = 100,
            guidance_scale: float = 7.5,
            image_guidance_scale: float = 1.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            # 回调函数定义，处理步骤结束时的操作
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 定义步骤结束时的张量输入
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 交叉注意力的额外参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 接收额外参数
            **kwargs,
    # 定义一个编码提示的私有方法
        def _encode_prompt(
            self,  # 方法的第一个参数，表示调用该方法时传入的提示文本
            prompt,  # 提示文本
            device,  # 设备类型（如 CPU 或 GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器的引导
            negative_prompt=None,  # 可选的负提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入，类型为 Torch 张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入，类型为 Torch 张量
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制而来
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):  # 定义一个编码图像的方法
            dtype = next(self.image_encoder.parameters()).dtype  # 获取图像编码器参数的数据类型
    
            if not isinstance(image, torch.Tensor):  # 如果图像不是 Torch 张量
                image = self.feature_extractor(image, return_tensors="pt").pixel_values  # 使用特征提取器将图像转换为张量
    
            image = image.to(device=device, dtype=dtype)  # 将图像移动到指定设备并转换为相应的数据类型
            if output_hidden_states:  # 如果需要输出隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]  # 编码图像并获取倒数第二层的隐藏状态
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)  # 根据每个提示的图像数量重复隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(  # 编码零图像以获取无条件隐藏状态
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]  # 获取倒数第二层的隐藏状态
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(  # 根据每个提示的图像数量重复无条件隐藏状态
                    num_images_per_prompt, dim=0
                )
                return image_enc_hidden_states, uncond_image_enc_hidden_states  # 返回编码后的隐藏状态
            else:  # 如果不需要输出隐藏状态
                image_embeds = self.image_encoder(image).image_embeds  # 编码图像并获取图像嵌入
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)  # 根据每个提示的图像数量重复图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)  # 创建与图像嵌入相同形状的零张量作为无条件嵌入
    
                return image_embeds, uncond_image_embeds  # 返回编码后的图像嵌入和无条件嵌入
    
        # 定义一个准备图像嵌入的适配器方法
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 检查 ip_adapter_image_embeds 是否为 None
        if ip_adapter_image_embeds is None:
            # 如果 ip_adapter_image 不是列表，则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 确保 ip_adapter_image 的长度与 IP Adapters 的数量相同
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    # 抛出错误提示，指出图像数量与 IP Adapters 数量不匹配
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 初始化图像嵌入列表
            image_embeds = []
            # 遍历每个单独的图像和对应的投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 确定是否需要输出隐藏状态
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码图像，获取正向和负向图像嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                # 将正向图像嵌入重复 num_images_per_prompt 次
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                # 将负向图像嵌入重复 num_images_per_prompt 次
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                # 如果启用了分类器自由引导
                if do_classifier_free_guidance:
                    # 将正向和负向图像嵌入拼接在一起
                    single_image_embeds = torch.cat(
                        [single_image_embeds, single_negative_image_embeds, single_negative_image_embeds]
                    )
                    # 将图像嵌入转移到指定设备
                    single_image_embeds = single_image_embeds.to(device)

                # 将单个图像嵌入添加到图像嵌入列表中
                image_embeds.append(single_image_embeds)
        else:
            # 定义重复维度
            repeat_dims = [1]
            # 初始化图像嵌入列表
            image_embeds = []
            # 遍历已有的图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用了分类器自由引导
                if do_classifier_free_guidance:
                    # 将图像嵌入分割为正向和负向嵌入
                    (
                        single_image_embeds,
                        single_negative_image_embeds,
                        single_negative_image_embeds,
                    ) = single_image_embeds.chunk(3)
                    # 重复正向图像嵌入
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    # 重复负向图像嵌入
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    # 将正向和负向图像嵌入拼接在一起
                    single_image_embeds = torch.cat(
                        [single_image_embeds, single_negative_image_embeds, single_negative_image_embeds]
                    )
                else:
                    # 重复单个图像嵌入
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                # 将单个图像嵌入添加到图像嵌入列表中
                image_embeds.append(single_image_embeds)

        # 返回最终的图像嵌入列表
        return image_embeds
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，设置 nsfw 概念为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入为张量，使用图像处理器将其后处理为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入为 NumPy 数组，转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 使用特征提取器处理输入，返回张量并移动到指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 调用安全检查器，返回处理后的图像和 nsfw 概念判断结果
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 nsfw 概念判断结果
        return image, has_nsfw_concept
    
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，因为并非所有调度器都有相同的签名
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器会忽略该参数。
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 值应在 [0, 1] 之间
    
        # 检查调度器的步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        # 如果接受 eta，将其添加到额外参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器的步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，将其添加到额外参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数
        return extra_step_kwargs
    
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制
    def decode_latents(self, latents):
        # 警告信息，表示该方法已弃用，将在 1.0.0 中移除，建议使用 VaeImageProcessor.postprocess(...)
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
        # 按比例缩放 latents
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码 latents，获取图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像像素值规范化到 [0, 1] 范围
        image = (image / 2 + 0.5).clamp(0, 1)
        # 始终转换为 float32，确保与 bfloat16 兼容且不会造成显著开销
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像
        return image
    
    # 定义检查输入参数的方法
    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        # 检查 callback_steps 是否为正整数，如果不是则引发 ValueError
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                # 报告 callback_steps 不是正整数的错误信息
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查 callback_on_step_end_tensor_inputs 是否为 None，且是否包含在 _callback_tensor_inputs 中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                # 报告 callback_on_step_end_tensor_inputs 中的某些元素不在 _callback_tensor_inputs 的错误信息
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查 prompt 和 prompt_embeds 是否同时存在
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                # 报告同时提供 prompt 和 prompt_embeds 的错误信息
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否都为 None
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                # 报告需要提供 prompt 或 prompt_embeds 之一的错误信息
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 的类型是否为 str 或 list
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查 negative_prompt 和 negative_prompt_embeds 是否同时存在
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                # 报告同时提供 negative_prompt 和 negative_prompt_embeds 的错误信息
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 是否同时存在且形状一致
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    # 报告 prompt_embeds 和 negative_prompt_embeds 形状不一致的错误信息
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # 检查 ip_adapter_image 和 ip_adapter_image_embeds 是否同时存在
        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                # 报告需要提供 ip_adapter_image 或 ip_adapter_image_embeds 之一的错误信息
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        # 检查 ip_adapter_image_embeds 是否存在且类型为 list
        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    # 报告 ip_adapter_image_embeds 不是 list 类型的错误信息
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            # 检查 ip_adapter_image_embeds 中第一个元素的维度是否为 3D 或 4D
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    # 报告 ip_adapter_image_embeds 中的张量维度不正确的错误信息
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的代码
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义形状，包含批大小、通道数和缩放后的高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,  # 按 VAE 缩放因子调整高度
            int(width) // self.vae_scale_factor,    # 按 VAE 缩放因子调整宽度
        )
        # 检查生成器是否为列表且长度与批大小不匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 抛出值错误，提示生成器长度与批大小不匹配
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有传入潜在变量
        if latents is None:
            # 生成随机张量作为潜在变量，使用指定的生成器、设备和数据类型
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果已传入潜在变量，将其转移到指定的设备
            latents = latents.to(device)

        # 按调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    def prepare_image_latents(
        # 准备图像潜在变量的方法，接受图像及相关参数
        self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        # 检查输入的图像类型是否为 torch.Tensor、PIL.Image.Image 或列表
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            # 如果类型不匹配，则抛出错误并显示当前类型
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        # 将图像转换到指定的设备和数据类型
        image = image.to(device=device, dtype=dtype)

        # 根据提示数量调整批处理大小
        batch_size = batch_size * num_images_per_prompt

        # 如果图像有4个通道，则直接使用它
        if image.shape[1] == 4:
            image_latents = image
        else:
            # 编码图像并以 "argmax" 模式检索潜在表示
            image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")

        # 如果批处理大小大于潜在表示的数量且可以整除
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # 生成警告消息
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            # 发出弃用警告
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            # 计算每个提示需要额外的图像数量
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            # 扩展潜在表示以匹配批处理大小
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        # 如果批处理大小大于潜在表示数量但不能整除
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            # 抛出错误，表示无法复制图像以匹配文本提示
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            # 将潜在表示展开为一个批次
            image_latents = torch.cat([image_latents], dim=0)

        # 如果启用分类器自由引导
        if do_classifier_free_guidance:
            # 创建与潜在表示形状相同的零张量
            uncond_image_latents = torch.zeros_like(image_latents)
            # 将潜在表示和未条件潜在表示合并
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        # 返回处理后的潜在表示
        return image_latents

    # 定义属性：引导比例
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 定义属性：图像引导比例
    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale

    # 定义属性：时间步数
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 此处的 `guidance_scale` 定义类似于 Imagen 论文中方程 (2) 的引导权重 `w`
    # `guidance_scale = 1` 表示不使用分类器自由引导。
    @property
    def do_classifier_free_guidance(self):
        # 根据引导比例和图像引导比例决定是否使用分类器自由引导
        return self.guidance_scale > 1.0 and self.image_guidance_scale >= 1.0
```