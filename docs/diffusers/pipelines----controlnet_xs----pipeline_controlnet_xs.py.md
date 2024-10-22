# `.\diffusers\pipelines\controlnet_xs\pipeline_controlnet_xs.py`

```py
# 版权声明，表示该代码的版权所有者及其保留权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证 2.0 版本许可该文件，只有在遵守许可证的情况下才能使用
# Licensed under the Apache License, Version 2.0 (the "License");
# 许可证可在以下网址获取
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件按“原样”分发，不提供任何明示或暗示的担保
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 参见许可证中规定的特定语言的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect  # 导入 inspect 模块，用于获取活跃对象的信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示，提供可选类型注解

import numpy as np  # 导入 NumPy 库，用于数值计算
import PIL.Image  # 导入 PIL 库，用于图像处理
import torch  # 导入 PyTorch 库，进行深度学习相关操作
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer  # 导入 Hugging Face 的 CLIP 相关类

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 导入回调相关的类
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入图像处理相关的类
from ...loaders import FromSingleFileMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入文件加载相关的类
from ...models import AutoencoderKL, ControlNetXSAdapter, UNet2DConditionModel, UNetControlNetXSModel  # 导入模型相关的类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 Lora 文本编码器比例的函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器
from ...utils import (  # 导入多个工具函数和变量
    USE_PEFT_BACKEND,  # 指示是否使用 PEFT 后端的标志
    deprecate,  # 用于标记过时功能的装饰器
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的函数
    scale_lora_layers,  # 缩放 Lora 层的函数
    unscale_lora_layers,  # 反缩放 Lora 层的函数
)
from ...utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor  # 导入与 PyTorch 相关的工具函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道相关的类
from ..stable_diffusion.pipeline_output import StableDiffusionPipelineOutput  # 导入稳定扩散管道输出的类
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 导入稳定扩散安全检查器的类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，禁用 Pylint 对命名的警告
    # 示例用法
        Examples:
            ```py
            >>> # !pip install opencv-python transformers accelerate
            # 导入所需的库和模块
            >>> from diffusers import StableDiffusionControlNetXSPipeline, ControlNetXSAdapter
            >>> from diffusers.utils import load_image
            >>> import numpy as np
            >>> import torch
    
            >>> import cv2
            >>> from PIL import Image
    
            # 设置生成图像的提示词
            >>> prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
            # 设置反向提示词，限制不想要的特征
            >>> negative_prompt = "low quality, bad quality, sketches"
    
            # 下载一张图像
            >>> image = load_image(
            ...     "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
            ... )
    
            # 初始化模型和管道
            >>> controlnet_conditioning_scale = 0.5
    
            # 从预训练模型加载 ControlNetXS 适配器
            >>> controlnet = ControlNetXSAdapter.from_pretrained(
            ...     "UmerHA/Testing-ConrolNetXS-SD2.1-canny", torch_dtype=torch.float16
            ... )
            # 从预训练模型加载稳定扩散管道
            >>> pipe = StableDiffusionControlNetXSPipeline.from_pretrained(
            ...     "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=torch.float16
            ... )
            # 启用模型 CPU 卸载，以节省内存
            >>> pipe.enable_model_cpu_offload()
    
            # 获取 Canny 边缘图像
            >>> image = np.array(image)  # 将图像转换为 NumPy 数组
            >>> image = cv2.Canny(image, 100, 200)  # 应用 Canny 边缘检测
            >>> image = image[:, :, None]  # 添加一个维度以适应图像格式
            >>> image = np.concatenate([image, image, image], axis=2)  # 将单通道图像转换为三通道
            >>> canny_image = Image.fromarray(image)  # 将 NumPy 数组转换为 PIL 图像
            # 生成图像
            >>> image = pipe(
            ...     prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image
            ... ).images[0]  # 通过管道生成图像，并提取第一张生成的图像
# 定义一个名为 StableDiffusionControlNetXSPipeline 的类，继承自多个基类
class StableDiffusionControlNetXSPipeline(
    # 继承 DiffusionPipeline 类，提供扩散管道的基本功能
    DiffusionPipeline,
    # 继承 StableDiffusionMixin 类，包含与稳定扩散相关的混合功能
    StableDiffusionMixin,
    # 继承 TextualInversionLoaderMixin 类，用于加载文本反转嵌入
    TextualInversionLoaderMixin,
    # 继承 StableDiffusionLoraLoaderMixin 类，负责加载 LoRA 权重
    StableDiffusionLoraLoaderMixin,
    # 继承 FromSingleFileMixin 类，用于从单个文件加载模型
    FromSingleFileMixin,
):
    r"""
    用于文本到图像生成的管道，使用 Stable Diffusion 和 ControlNet-XS 指导。

    该模型从 [`DiffusionPipeline`] 继承。请查看父类文档以获取所有管道实现的通用方法（下载、保存、在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件

    参数：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器 (VAE) 模型。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于对文本进行分词的 `CLIPTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于创建 UNetControlNetXSModel 的 [`UNet2DConditionModel`]，用于去噪编码后的图像潜变量。
        controlnet ([`ControlNetXSAdapter`]):
            [`ControlNetXSAdapter`] 与 `unet` 结合使用以去噪编码的图像潜变量。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用以去噪编码的图像潜变量的调度器。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            分类模块，用于估计生成的图像是否可能被视为冒犯或有害。
            请参阅 [模型卡](https://huggingface.co/runwayml/stable-diffusion-v1-5) 了解模型潜在危害的更多细节。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            用于提取生成图像特征的 `CLIPImageProcessor`；作为输入提供给 `safety_checker`。
    """

    # 定义模型 CPU 离线处理顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件列表
    _optional_components = ["safety_checker", "feature_extractor"]
    # 定义不参与 CPU 离线处理的组件
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义需要回调的张量输入
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # 初始化方法，设置模型的各项参数
        def __init__(
            self,
            vae: AutoencoderKL,  # 变分自编码器模型
            text_encoder: CLIPTextModel,  # 文本编码器模型
            tokenizer: CLIPTokenizer,  # 用于文本处理的分词器
            unet: Union[UNet2DConditionModel, UNetControlNetXSModel],  # U-Net 模型，用于生成图像
            controlnet: ControlNetXSAdapter,  # 控制网络适配器
            scheduler: KarrasDiffusionSchedulers,  # 扩散调度器
            safety_checker: StableDiffusionSafetyChecker,  # 安全检查器
            feature_extractor: CLIPImageProcessor,  # 特征提取器
            requires_safety_checker: bool = True,  # 是否需要安全检查器
        ):
            # 调用父类的初始化方法
            super().__init__()
    
            # 如果传入的 UNet 是 UNet2DConditionModel，则转换为 UNetControlNetXSModel
            if isinstance(unet, UNet2DConditionModel):
                unet = UNetControlNetXSModel.from_unet(unet, controlnet)
    
            # 如果安全检查器为 None 且要求安全检查，则发出警告
            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 如果安全检查器不为 None，但特征提取器为 None，则抛出错误
            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )
    
            # 注册各个模块，设置其属性
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器，用于处理 VAE 输出
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
            # 创建控制图像处理器，用于处理控制网络的图像
            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
            # 注册配置参数，是否需要安全检查器
            self.register_to_config(requires_safety_checker=requires_safety_checker)
    
        # 从 StableDiffusionPipeline 复制的编码提示的方法
        def _encode_prompt(
            self,
            prompt,  # 输入的提示文本
            device,  # 设备类型，例如 CPU 或 GPU
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否执行无分类器引导
            negative_prompt=None,  # 负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            lora_scale: Optional[float] = None,  # 可选的 LoRA 缩放因子
            **kwargs,  # 其他可选参数
    ):
        # 定义弃用警告信息，提示用户该方法已被弃用
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用 deprecate 函数记录弃用信息，标记版本号为 "1.0.0"
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用 encode_prompt 方法，获取提示嵌入元组
        prompt_embeds_tuple = self.encode_prompt(
            # 传入提示文本
            prompt=prompt,
            # 设备参数
            device=device,
            # 每个提示生成的图像数量
            num_images_per_prompt=num_images_per_prompt,
            # 是否进行无分类引导
            do_classifier_free_guidance=do_classifier_free_guidance,
            # 负提示文本
            negative_prompt=negative_prompt,
            # 提示嵌入
            prompt_embeds=prompt_embeds,
            # 负提示嵌入
            negative_prompt_embeds=negative_prompt_embeds,
            # LORA 缩放因子
            lora_scale=lora_scale,
            # 额外参数
            **kwargs,
        )

        # 将提示嵌入元组中的两个部分连接，兼容以前的实现
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回合并后的提示嵌入
        return prompt_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 中复制的 encode_prompt 方法
    def encode_prompt(
        # 提示文本
        self,
        prompt,
        # 设备参数
        device,
        # 每个提示生成的图像数量
        num_images_per_prompt,
        # 是否进行无分类引导
        do_classifier_free_guidance,
        # 可选的负提示文本
        negative_prompt=None,
        # 可选的提示嵌入
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的 LORA 缩放因子
        lora_scale: Optional[float] = None,
        # 可选的剪切跳过参数
        clip_skip: Optional[int] = None,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 中复制的 run_safety_checker 方法
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，设置 nsfw 概念为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入图像为张量，则后处理为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入图像为 numpy 格式，则转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 使用特征提取器生成安全检查器输入
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 调用安全检查器进行图像检查，并获取 nsfw 概念的状态
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 nsfw 概念的状态
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 中复制的 decode_latents 方法
    # 解码潜在变量的函数
        def decode_latents(self, latents):
            # 生成弃用警告信息，提示用户该方法将被移除
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            # 调用弃用函数，传递方法名、版本和警告信息
            deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
            # 根据配置的缩放因子调整潜在变量
            latents = 1 / self.vae.config.scaling_factor * latents
            # 解码潜在变量，返回的第一个元素为图像
            image = self.vae.decode(latents, return_dict=False)[0]
            # 对图像进行归一化处理，并限制值在 [0, 1] 范围内
            image = (image / 2 + 0.5).clamp(0, 1)
            # 将图像转换为 float32 格式，适配 bfloat16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            # 返回解码后的图像
            return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的函数
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备额外的参数，以便调度器的步骤可以接受不同的签名
            # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略该参数
            # eta 对应于 DDIM 论文中的 η，应在 [0, 1] 之间
    
            # 检查调度器的步骤是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外参数字典
            extra_step_kwargs = {}
            # 如果接受 eta，添加到额外参数字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受 generator，添加到额外参数字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回额外参数字典
            return extra_step_kwargs
    
        # 检查输入参数的函数
        def check_inputs(
            self,
            prompt,
            image,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            controlnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            callback_on_step_end_tensor_inputs=None,
    # 检查输入图像和提示的类型和大小
        def check_image(self, image, prompt, prompt_embeds):
            # 检查图像是否为 PIL 图像类型
            image_is_pil = isinstance(image, PIL.Image.Image)
            # 检查图像是否为 PyTorch 张量类型
            image_is_tensor = isinstance(image, torch.Tensor)
            # 检查图像是否为 NumPy 数组类型
            image_is_np = isinstance(image, np.ndarray)
            # 检查图像是否为 PIL 图像列表
            image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
            # 检查图像是否为 PyTorch 张量列表
            image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
            # 检查图像是否为 NumPy 数组列表
            image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)
    
            # 如果图像不是上述任一类型，则引发类型错误
            if (
                not image_is_pil
                and not image_is_tensor
                and not image_is_np
                and not image_is_pil_list
                and not image_is_tensor_list
                and not image_is_np_list
            ):
                raise TypeError(
                    f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
                )
    
            # 如果图像是 PIL 图像，设置图像批次大小为 1
            if image_is_pil:
                image_batch_size = 1
            else:
                # 否则，批次大小为图像的长度
                image_batch_size = len(image)
    
            # 如果提示不为空且为字符串，设置提示批次大小为 1
            if prompt is not None and isinstance(prompt, str):
                prompt_batch_size = 1
            # 如果提示为列表，设置提示批次大小为列表的长度
            elif prompt is not None and isinstance(prompt, list):
                prompt_batch_size = len(prompt)
            # 如果提示嵌入不为空，设置提示批次大小为嵌入的第一个维度大小
            elif prompt_embeds is not None:
                prompt_batch_size = prompt_embeds.shape[0]
    
            # 如果图像批次大小不为 1 且与提示批次大小不相同，则引发值错误
            if image_batch_size != 1 and image_batch_size != prompt_batch_size:
                raise ValueError(
                    f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
                )
    
        # 准备图像以进行进一步处理
        def prepare_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
        ):
            # 使用控制图像处理器预处理图像，并转换为 float32 类型
            image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
            # 获取图像批次大小
            image_batch_size = image.shape[0]
    
            # 如果图像批次大小为 1，则按批次大小重复图像
            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # 否则，按每个提示的图像数量重复
                repeat_by = num_images_per_prompt
    
            # 在第 0 维重复图像
            image = image.repeat_interleave(repeat_by, dim=0)
    
            # 将图像移动到指定设备和数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 如果启用无分类器自由引导，将图像重复两次
            if do_classifier_free_guidance:
                image = torch.cat([image] * 2)
    
            # 返回处理后的图像
            return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的代码
    # 准备潜在变量，设置形状和初始化参数
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 计算潜在变量的形状，考虑批量大小、通道数和缩放因子
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器列表的长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 抛出值错误，提示生成器长度与批量大小不匹配
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有提供潜在变量，则生成新的潜在变量
        if latents is None:
            # 使用随机张量生成潜在变量，指定形状、生成器、设备和数据类型
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在变量，则将其转移到指定设备
            latents = latents.to(device)

        # 将初始噪声按调度器要求的标准差进行缩放
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 属性装饰器，获取引导比例
    @property
    # 从稳定扩散管道复制的引导比例属性
    def guidance_scale(self):
        # 返回当前的引导比例值
        return self._guidance_scale

    # 属性装饰器，获取剪切跳过设置
    @property
    # 从稳定扩散管道复制的剪切跳过属性
    def clip_skip(self):
        # 返回剪切跳过的配置值
        return self._clip_skip

    # 属性装饰器，判断是否执行无分类器自由引导
    @property
    # 从稳定扩散管道复制的无分类器自由引导属性
    def do_classifier_free_guidance(self):
        # 判断条件：引导比例大于1且时间条件投影维度为空
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    # 属性装饰器，获取交叉注意力参数
    @property
    # 从稳定扩散管道复制的交叉注意力参数属性
    def cross_attention_kwargs(self):
        # 返回交叉注意力的参数设置
        return self._cross_attention_kwargs

    # 属性装饰器，获取时间步数
    @property
    # 从稳定扩散管道复制的时间步数属性
    def num_timesteps(self):
        # 返回当前的时间步数
        return self._num_timesteps

    # 无梯度上下文装饰器，禁止梯度计算
    @torch.no_grad()
    # 替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义调用方法，允许对象被调用并执行某些操作
        def __call__(
            # 提示文本，可以是单个字符串或字符串列表，默认为 None
            self,
            prompt: Union[str, List[str]] = None,
            # 输入图像，类型为 PipelineImageInput，默认为 None
            image: PipelineImageInput = None,
            # 输出图像高度，类型为可选整型，默认为 None
            height: Optional[int] = None,
            # 输出图像宽度，类型为可选整型，默认为 None
            width: Optional[int] = None,
            # 推理步骤数量，默认为 50
            num_inference_steps: int = 50,
            # 指导比例，默认为 7.5
            guidance_scale: float = 7.5,
            # 负提示文本，可以是单个字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 噪声参数，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，可以是单个或多个 torch.Generator，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，可以是 torch.Tensor 类型，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，可以是 torch.Tensor 类型，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入，可以是 torch.Tensor 类型，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典形式的结果，默认为 True
            return_dict: bool = True,
            # 交叉注意力参数的可选字典，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # ControlNet 的条件比例，默认为 1.0
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            # ControlNet 指导开始比例，默认为 0.0
            control_guidance_start: float = 0.0,
            # ControlNet 指导结束比例，默认为 1.0
            control_guidance_end: float = 1.0,
            # 可选的跳过剪辑参数，默认为 None
            clip_skip: Optional[int] = None,
            # 步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 回调函数结束时张量输入的列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
```