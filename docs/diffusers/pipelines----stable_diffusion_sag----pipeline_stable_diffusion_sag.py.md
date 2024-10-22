# `.\diffusers\pipelines\stable_diffusion_sag\pipeline_stable_diffusion_sag.py`

```py
# 版权声明，说明此文件的版权所有者和许可信息
# Copyright 2024 Susung Hong and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可协议授权使用此文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 使用此文件前需遵循许可协议
# you may not use this file except in compliance with the License.
# 可在以下链接获取许可的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按 "AS IS" 基础分发，不提供任何形式的担保
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解特定的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect  # 导入 inspect 模块，用于检查对象的属性和方法
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示所需的类型

import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn.functional as F  # 导入 PyTorch 的函数式 API，用于常用的神经网络功能
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 从 transformers 导入 CLIP 相关类

from ...image_processor import PipelineImageInput, VaeImageProcessor  # 从图像处理模块导入输入和处理器类
from ...loaders import IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入加载器混合类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 导入模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 Lora 的函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入调度器类
from ...utils import (  # 导入常用工具函数
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor  # 导入用于生成随机张量的函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和混合类
from ..stable_diffusion import StableDiffusionPipelineOutput  # 导入稳定扩散管道的输出类
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 导入安全检查器类

logger = logging.get_logger(__name__)  # 初始化记录器，使用当前模块名作为标识

EXAMPLE_DOC_STRING = """  # 示例文档字符串，用于展示如何使用管道
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import StableDiffusionSAGPipeline  # 从 diffusers 导入 StableDiffusionSAGPipeline

        >>> pipe = StableDiffusionSAGPipeline.from_pretrained(  # 从预训练模型创建管道
        ...     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16  # 指定模型和数据类型
        ... )
        >>> pipe = pipe.to("cuda")  # 将管道移动到 GPU

        >>> prompt = "a photo of an astronaut riding a horse on mars"  # 定义生成图像的提示
        >>> image = pipe(prompt, sag_scale=0.75).images[0]  # 生成图像并获取第一个图像
        ```py
"""


# 处理和存储注意力概率的类
class CrossAttnStoreProcessor:
    def __init__(self):  # 初始化方法
        self.attention_probs = None  # 创建一个用于存储注意力概率的属性

    def __call__(  # 定义可调用对象的方法
        self,
        attn,  # 注意力张量
        hidden_states,  # 隐藏状态张量
        encoder_hidden_states=None,  # 编码器隐藏状态（可选）
        attention_mask=None,  # 注意力掩码（可选）
    ):
        # 获取隐藏状态的批次大小、序列长度和特征维度
        batch_size, sequence_length, _ = hidden_states.shape
        # 准备注意力掩码，以便后续计算
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # 将隐藏状态转换为查询向量
        query = attn.to_q(hidden_states)

        # 检查编码器隐藏状态是否为 None
        if encoder_hidden_states is None:
            # 如果为 None，则将其设置为隐藏状态
            encoder_hidden_states = hidden_states
        # 如果需要进行归一化处理
        elif attn.norm_cross:
            # 对编码器隐藏状态进行归一化
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 将编码器隐藏状态转换为键向量
        key = attn.to_k(encoder_hidden_states)
        # 将编码器隐藏状态转换为值向量
        value = attn.to_v(encoder_hidden_states)

        # 将查询向量从头维度转换为批次维度
        query = attn.head_to_batch_dim(query)
        # 将键向量从头维度转换为批次维度
        key = attn.head_to_batch_dim(key)
        # 将值向量从头维度转换为批次维度
        value = attn.head_to_batch_dim(value)

        # 计算注意力分数
        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 使用注意力分数和值向量进行批次矩阵乘法，得到新的隐藏状态
        hidden_states = torch.bmm(self.attention_probs, value)
        # 将隐藏状态从批次维度转换回头维度
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 进行线性变换以获取最终的隐藏状态
        hidden_states = attn.to_out[0](hidden_states)
        # 应用 dropout 以减少过拟合
        hidden_states = attn.to_out[1](hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
# 修改以获取自注意力引导缩放，在此论文中作为输入 (https://arxiv.org/pdf/2210.00939.pdf)
class StableDiffusionSAGPipeline(DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, IPAdapterMixin):
    r"""
    使用稳定扩散的文本到图像生成管道。

    此模型继承自 [`DiffusionPipeline`]。查看超类文档以获取所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    此管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    参数：
        vae ([`AutoencoderKL`]):
            用于编码和解码图像到潜在表示的变分自编码器 (VAE) 模型。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            一个 `CLIPTokenizer` 用于对文本进行标记化。
        unet ([`UNet2DConditionModel`]):
            一个 `UNet2DConditionModel` 用于去噪编码的图像潜在。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 结合使用以去噪编码的图像潜在的调度器。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            估计生成的图像是否可能被视为冒犯或有害的分类模块。
            请参考 [模型卡](https://huggingface.co/runwayml/stable-diffusion-v1-5) 获取更多
            关于模型潜在危害的详细信息。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            一个 `CLIPImageProcessor` 用于从生成的图像中提取特征；用作 `safety_checker` 的输入。
    """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件的列表
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义不包括在 CPU 卸载中的组件
    _exclude_from_cpu_offload = ["safety_checker"]

    # 初始化方法，接受多个参数以配置管道
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        # 可选的图像编码器，默认为 None
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
        # 指示是否需要安全检查器的布尔值，默认为 True
        requires_safety_checker: bool = True,
    # 初始化父类
        ):
            super().__init__()
    
            # 注册各个模块，供后续使用
            self.register_modules(
                vae=vae,  # 注册变分自编码器
                text_encoder=text_encoder,  # 注册文本编码器
                tokenizer=tokenizer,  # 注册分词器
                unet=unet,  # 注册 U-Net 模型
                scheduler=scheduler,  # 注册调度器
                safety_checker=safety_checker,  # 注册安全检查器
                feature_extractor=feature_extractor,  # 注册特征提取器
                image_encoder=image_encoder,  # 注册图像编码器
            )
            # 计算 VAE 的缩放因子，基于块输出通道的长度
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器，使用 VAE 缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            # 将配置注册到当前实例，指定是否需要安全检查器
            self.register_to_config(requires_safety_checker=requires_safety_checker)
    
        # 从 StableDiffusionPipeline 复制的编码提示方法
        def _encode_prompt(
            self,
            prompt,  # 输入提示
            device,  # 设备信息
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器的引导
            negative_prompt=None,  # 负提示（可选）
            prompt_embeds: Optional[torch.Tensor] = None,  # 提示嵌入（可选）
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负提示嵌入（可选）
            lora_scale: Optional[float] = None,  # Lora 缩放因子（可选）
            **kwargs,  # 其他参数
        ):
            # 设置废弃警告信息
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 发出废弃警告
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用 encode_prompt 方法获取提示嵌入元组
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                **kwargs,
            )
    
            # 将嵌入元组连接以兼容旧版本
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回合并后的提示嵌入
            return prompt_embeds
    
        # 从 StableDiffusionPipeline 复制的编码提示方法
        def encode_prompt(
            self,
            prompt,  # 输入提示
            device,  # 设备信息
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器的引导
            negative_prompt=None,  # 负提示（可选）
            prompt_embeds: Optional[torch.Tensor] = None,  # 提示嵌入（可选）
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负提示嵌入（可选）
            lora_scale: Optional[float] = None,  # Lora 缩放因子（可选）
            clip_skip: Optional[int] = None,  # 跳过剪辑的数量（可选）
        # 从 StableDiffusionPipeline 复制的编码图像方法
    # 定义一个方法，用于编码输入图像，返回图像的嵌入表示
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        # 获取图像编码器参数的数据类型
        dtype = next(self.image_encoder.parameters()).dtype

        # 如果输入的图像不是张量，则使用特征提取器将其转换为张量
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        # 将图像移动到指定的设备上，并转换为正确的数据类型
        image = image.to(device=device, dtype=dtype)
        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 编码图像，并获取倒数第二层的隐藏状态
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            # 重复隐藏状态以匹配每个提示的图像数量
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            # 编码全零图像以获取无条件隐藏状态
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            # 重复无条件隐藏状态以匹配每个提示的图像数量
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            # 返回有条件和无条件的隐藏状态
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            # 编码图像以获取图像嵌入
            image_embeds = self.image_encoder(image).image_embeds
            # 重复图像嵌入以匹配每个提示的图像数量
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            # 创建与图像嵌入形状相同的全零张量作为无条件嵌入
            uncond_image_embeds = torch.zeros_like(image_embeds)

            # 返回有条件和无条件的图像嵌入
            return image_embeds, uncond_image_embeds

    # 定义一个方法，用于准备 IP 适配器的图像嵌入
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 检查 ip_adapter_image_embeds 是否为 None
        if ip_adapter_image_embeds is None:
            # 检查 ip_adapter_image 是否为列表，如果不是则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查 ip_adapter_image 的长度是否与 IP 适配器的数量一致
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                # 如果不一致，抛出一个 ValueError 异常
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 初始化一个空列表，用于存储图像嵌入
            image_embeds = []
            # 遍历每个图像适配器图像和对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 判断输出隐藏状态是否为 True，条件是图像投影层不是 ImageProjection 实例
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个图像，获取嵌入和负嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                # 将单个图像嵌入复制 num_images_per_prompt 次，形成一个新维度
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                # 同样处理负嵌入
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                # 如果执行无分类器引导
                if do_classifier_free_guidance:
                    # 将负嵌入和正嵌入连接在一起
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                    # 将嵌入数据移动到指定设备
                    single_image_embeds = single_image_embeds.to(device)

                # 将处理后的图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)
        else:
            # 如果 ip_adapter_image_embeds 不是 None，则直接使用它
            image_embeds = ip_adapter_image_embeds
        # 返回最终的图像嵌入列表
        return image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制
    def run_safety_checker(self, image, device, dtype):
        # 检查安全检查器是否为 None
        if self.safety_checker is None:
            # 如果安全检查器为 None，设置 has_nsfw_concept 为 None
            has_nsfw_concept = None
        else:
            # 检查图像是否为张量
            if torch.is_tensor(image):
                # 如果是张量，使用图像处理器后处理图像，输出类型为 PIL
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果不是张量，将 NumPy 数组转换为 PIL 图像
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 提取特征，返回张量并移动到指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 运行安全检查器，返回处理后的图像和 nsfw 概念的存在情况
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 nsfw 概念的存在情况
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制
    # 解码潜在向量的方法
    def decode_latents(self, latents):
        # 设置过时警告信息，提示该方法将在 1.0.0 版本中被移除
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用 deprecate 函数记录该方法的使用和警告信息
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 将潜在向量根据缩放因子进行缩放
        latents = 1 / self.vae.config.scaling_factor * latents
        # 使用 VAE 解码潜在向量，返回图像数据
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像数据标准化到 [0, 1] 范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为 float32 类型，兼容 bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回解码后的图像数据
        return image

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 中复制
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备调度器步骤的额外参数，因为并不是所有调度器都有相同的签名
        # eta（η）仅用于 DDIMScheduler，其他调度器将忽略它。
        # eta 对应 DDIM 论文中的 η，范围应在 [0, 1] 之间

        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外参数字典
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数
        return extra_step_kwargs

    # 从 diffusers.pipelines.stable_diffusion_k_diffusion.pipeline_stable_diffusion_k_diffusion.StableDiffusionKDiffusionPipeline.check_inputs 中复制
    def check_inputs(
        # 定义检查输入参数的方法，包含 prompt 和图像尺寸等
        self,
        prompt,
        height,
        width,
        callback_steps,
        # 可选的负面提示
        negative_prompt=None,
        # 可选的提示嵌入
        prompt_embeds=None,
        # 可选的负面提示嵌入
        negative_prompt_embeds=None,
        # 可选的回调步骤结束时的张量输入
        callback_on_step_end_tensor_inputs=None,
    ):
        # 检查高度和宽度是否能够被8整除，若不能则抛出错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步骤是否为正整数，若不满足条件则抛出错误
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        # 检查在步骤结束时的回调张量输入是否在已定义的回调张量输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查是否同时提供了 `prompt` 和 `prompt_embeds`
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否同时未提供 `prompt` 和 `prompt_embeds`
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 `prompt` 是否为字符串或列表类型
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了 `negative_prompt` 和 `negative_prompt_embeds`
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 `prompt_embeds` 和 `negative_prompt_embeds` 的形状是否一致
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
    # 准备潜在向量的函数，接收多个参数以配置潜在向量的形状和生成方式
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在向量的形状，基于输入的批大小和通道数，调整高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,  # 高度缩放
            int(width) // self.vae_scale_factor,    # 宽度缩放
        )
        # 检查生成器是否为列表，并确保列表长度与批大小一致
        if isinstance(generator, list) and len(generator) != batch_size:
            # 如果不一致，抛出值错误
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果未提供潜在向量，则生成随机潜在向量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在向量，则将其转移到指定设备
            latents = latents.to(device)

        # 按调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回准备好的潜在向量
        return latents

    # 禁用梯度计算以节省内存和加快推理速度
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象，接收多个参数以执行生成过程
    def __call__(
        # 提示文本，可以是单个字符串或字符串列表
        prompt: Union[str, List[str]] = None,
        # 生成图像的高度
        height: Optional[int] = None,
        # 生成图像的宽度
        width: Optional[int] = None,
        # 推理步骤的数量
        num_inference_steps: int = 50,
        # 指导尺度，用于控制生成的质量
        guidance_scale: float = 7.5,
        # SAG尺度，用于调整特定生成效果
        sag_scale: float = 0.75,
        # 负面提示文本，可以是单个字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量
        num_images_per_prompt: Optional[int] = 1,
        # 用于生成的 eta 参数
        eta: float = 0.0,
        # 随机数生成器，可以是单个生成器或生成器列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜在向量，如果有的话
        latents: Optional[torch.Tensor] = None,
        # 提示嵌入向量
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负面提示嵌入向量
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 输入适配器图像
        ip_adapter_image: Optional[PipelineImageInput] = None,
        # 输入适配器图像嵌入向量列表
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # 输出类型，默认为 PIL 图像
        output_type: Optional[str] = "pil",
        # 是否返回字典格式的输出
        return_dict: bool = True,
        # 可选的回调函数，用于在推理过程中执行某些操作
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调步骤的频率
        callback_steps: Optional[int] = 1,
        # 交叉注意力的额外参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 跳过的 CLIP 层数
        clip_skip: Optional[int] = None,
    # 定义一个方法用于SAG掩膜处理
        def sag_masking(self, original_latents, attn_map, map_size, t, eps):
            # 按照SAG论文中的掩膜处理流程
            bh, hw1, hw2 = attn_map.shape  # 解包注意力图的维度
            b, latent_channel, latent_h, latent_w = original_latents.shape  # 解包原始潜变量的维度
            h = self.unet.config.attention_head_dim  # 获取注意力头的维度
            if isinstance(h, list):  # 如果h是列表
                h = h[-1]  # 取最后一个维度值
    
            # 生成注意力掩膜
            attn_map = attn_map.reshape(b, h, hw1, hw2)  # 重新调整注意力图的形状
            attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0  # 计算掩膜，阈值为1.0
            attn_mask = (
                attn_mask.reshape(b, map_size[0], map_size[1])  # 重新调整掩膜形状以匹配map_size
                .unsqueeze(1)  # 增加一个维度
                .repeat(1, latent_channel, 1, 1)  # 在channel维度上重复掩膜
                .type(attn_map.dtype)  # 转换为与注意力图相同的数据类型
            )
            attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))  # 按照潜变量的高度和宽度插值调整掩膜
    
            # 根据自注意力掩膜进行模糊处理
            degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)  # 对原始潜变量进行高斯模糊
            degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)  # 按掩膜加权融合模糊与原始潜变量
    
            # 重新加噪声以匹配噪声水平
            degraded_latents = self.scheduler.add_noise(degraded_latents, noise=eps, timesteps=t[None])  # 添加噪声
    
            return degraded_latents  # 返回处理后的潜变量
    
        # 从diffusers.schedulers.scheduling_ddim.DDIMScheduler.step修改而来
        # 注意：有些调度器会裁剪或不返回x_0（PNDMScheduler, DDIMScheduler等）
        def pred_x0(self, sample, model_output, timestep):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(sample.device)  # 获取当前时间步的累积alpha值
    
            beta_prod_t = 1 - alpha_prod_t  # 计算beta值
            if self.scheduler.config.prediction_type == "epsilon":  # 如果预测类型为“epsilon”
                pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)  # 计算原始样本的预测值
            elif self.scheduler.config.prediction_type == "sample":  # 如果预测类型为“sample”
                pred_original_sample = model_output  # 直接使用模型输出作为预测值
            elif self.scheduler.config.prediction_type == "v_prediction":  # 如果预测类型为“v_prediction”
                pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output  # 计算预测值V
                # 预测V
                model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample  # 更新模型输出
            else:  # 如果预测类型不匹配
                raise ValueError(  # 抛出异常
                    f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " or `v_prediction`"
                )
    
            return pred_original_sample  # 返回预测的原始样本
    # 定义一个方法，用于根据给定样本和模型输出计算预测值
        def pred_epsilon(self, sample, model_output, timestep):
            # 获取当前时间步的累积 alpha 值
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
    
            # 计算当前时间步的 beta 值
            beta_prod_t = 1 - alpha_prod_t
            # 根据预测类型选择对应的计算方式
            if self.scheduler.config.prediction_type == "epsilon":
                # 直接使用模型输出作为预测值
                pred_eps = model_output
            elif self.scheduler.config.prediction_type == "sample":
                # 通过样本和模型输出计算预测值
                pred_eps = (sample - (alpha_prod_t**0.5) * model_output) / (beta_prod_t**0.5)
            elif self.scheduler.config.prediction_type == "v_prediction":
                # 计算加权组合的预测值
                pred_eps = (beta_prod_t**0.5) * sample + (alpha_prod_t**0.5) * model_output
            else:
                # 如果预测类型不合法，抛出错误
                raise ValueError(
                    f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " or `v_prediction`"
                )
    
            # 返回计算得到的预测值
            return pred_eps
# 高斯模糊
def gaussian_blur_2d(img, kernel_size, sigma):
    # 计算卷积核的一半大小
    ksize_half = (kernel_size - 1) * 0.5

    # 创建一个从 -ksize_half 到 ksize_half 的线性空间，步长为 kernel_size
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    # 计算概率密度函数（PDF），用于高斯分布
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    # 将 PDF 归一化为高斯核
    x_kernel = pdf / pdf.sum()
    # 将高斯核转换为与输入图像相同的设备和数据类型
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    # 生成二维高斯核，使用外积
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    # 扩展高斯核以适应输入图像的通道数
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    # 计算填充的大小，用于处理边界
    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    # 使用反射模式对图像进行填充
    img = F.pad(img, padding, mode="reflect")
    # 对填充后的图像应用卷积，使用高斯核
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    # 返回模糊处理后的图像
    return img
```