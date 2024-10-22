# `.\diffusers\pipelines\controlnet\pipeline_controlnet_sd_xl_img2img.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件
# 在“按原样”基础上分发，不提供任何形式的保证或条件，
# 无论是明示或暗示的。
# 请参阅许可证以了解管理权限的具体语言和
# 限制条款。


import inspect  # 导入 inspect 模块，用于获取对象的摘要信息
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型注解模块

import numpy as np  # 导入 numpy，用于数组和矩阵计算
import PIL.Image  # 导入 PIL.Image，用于处理图像
import torch  # 导入 PyTorch，用于深度学习
import torch.nn.functional as F  # 导入 PyTorch 的函数式 API
from transformers import (  # 从 transformers 导入模型和处理器
    CLIPImageProcessor,  # 导入 CLIP 图像处理器
    CLIPTextModel,  # 导入 CLIP 文本模型
    CLIPTextModelWithProjection,  # 导入带投影的 CLIP 文本模型
    CLIPTokenizer,  # 导入 CLIP 分词器
    CLIPVisionModelWithProjection,  # 导入带投影的 CLIP 视觉模型
)

from diffusers.utils.import_utils import is_invisible_watermark_available  # 导入检查是否可用的隐形水印功能

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 导入多管道回调和管道回调类
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入图像处理相关类
from ...loaders import (  # 导入加载器相关类
    FromSingleFileMixin,  # 从单文件加载的混合类
    IPAdapterMixin,  # 图像处理适配器混合类
    StableDiffusionXLLoraLoaderMixin,  # StableDiffusionXL Lora 加载混合类
    TextualInversionLoaderMixin,  # 文本反转加载混合类
)
from ...models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel  # 导入不同模型
from ...models.attention_processor import (  # 导入注意力处理器
    AttnProcessor2_0,  # 注意力处理器版本 2.0
    XFormersAttnProcessor,  # XFormers 注意力处理器
)
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 Lora 标度文本编码器的函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器
from ...utils import (  # 导入常用工具
    USE_PEFT_BACKEND,  # 指示是否使用 PEFT 后端的常量
    deprecate,  # 导入弃用装饰器
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 导入替换示例文档字符串的工具
    scale_lora_layers,  # 导入缩放 Lora 层的工具
    unscale_lora_layers,  # 导入反缩放 Lora 层的工具
)
from ...utils.torch_utils import is_compiled_module, randn_tensor  # 导入与 PyTorch 相关的工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和稳定扩散混合类
from ..stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput  # 导入稳定扩散 XL 管道输出类


if is_invisible_watermark_available():  # 如果隐形水印功能可用
    from ..stable_diffusion_xl.watermark import StableDiffusionXLWatermarker  # 导入稳定扩散 XL 水印类

from .multicontrolnet import MultiControlNetModel  # 导入多控制网模型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，禁止 pylint 检查


EXAMPLE_DOC_STRING = """  # 示例文档字符串的空模板
"""


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(  # 定义函数以检索潜在变量
    encoder_output: torch.Tensor,  # 输入为编码器输出的张量
    generator: Optional[torch.Generator] = None,  # 可选的随机数生成器
    sample_mode: str = "sample"  # 采样模式，默认为“sample”
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":  # 如果编码器输出有潜在分布并且模式为采样
        return encoder_output.latent_dist.sample(generator)  # 从潜在分布中采样并返回
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":  # 如果编码器输出有潜在分布并且模式为“argmax”
        return encoder_output.latent_dist.mode()  # 返回潜在分布的众数
    elif hasattr(encoder_output, "latents"):  # 如果编码器输出有潜在变量
        return encoder_output.latents  # 直接返回潜在变量
    else:  # 如果以上条件都不满足
        raise AttributeError("Could not access latents of provided encoder_output")  # 抛出属性错误，说明无法访问潜在变量


class StableDiffusionXLControlNetImg2ImgPipeline(  # 定义 StableDiffusionXL 控制网络图像到图像的管道类
    DiffusionPipeline,  # 继承自扩散管道
    # 继承稳定扩散模型的混合类
        StableDiffusionMixin,
        # 继承文本反转加载器的混合类
        TextualInversionLoaderMixin,
        # 继承稳定扩散 XL Lora 加载器的混合类
        StableDiffusionXLLoraLoaderMixin,
        # 继承单文件加载器的混合类
        FromSingleFileMixin,
        # 继承 IP 适配器的混合类
        IPAdapterMixin,
# 文档字符串，描述使用 ControlNet 指导的图像生成管道
    r"""
    Pipeline for image-to-image generation using Stable Diffusion XL with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    """

    # 定义模型在 CPU 上卸载的顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    # 定义可选组件的列表，用于管道的初始化
    _optional_components = [
        "tokenizer",  # 词汇表，用于文本编码
        "tokenizer_2",  # 第二个词汇表，用于文本编码
        "text_encoder",  # 文本编码器，用于生成文本嵌入
        "text_encoder_2",  # 第二个文本编码器，可能有不同的功能
        "feature_extractor",  # 特征提取器，用于图像特征的提取
        "image_encoder",  # 图像编码器，将图像转换为嵌入
    ]
    # 定义回调张量输入的列表，用于处理管道中的输入
    _callback_tensor_inputs = [
        "latents",  # 潜在变量，用于生成模型的输入
        "prompt_embeds",  # 正向提示的嵌入表示
        "negative_prompt_embeds",  # 负向提示的嵌入表示
        "add_text_embeds",  # 额外文本嵌入，用于补充输入
        "add_time_ids",  # 附加的时间标识符，用于时间相关的处理
        "negative_pooled_prompt_embeds",  # 负向池化提示的嵌入表示
        "add_neg_time_ids",  # 附加的负向时间标识符
    ]

    # 构造函数，初始化管道所需的组件
    def __init__(
        self,  # 构造函数的第一个参数，指向类的实例
        vae: AutoencoderKL,  # 变分自编码器，用于图像的重建
        text_encoder: CLIPTextModel,  # 文本编码器，使用 CLIP 模型
        text_encoder_2: CLIPTextModelWithProjection,  # 第二个文本编码器，带投影功能的 CLIP 模型
        tokenizer: CLIPTokenizer,  # 第一个 CLIP 词汇表
        tokenizer_2: CLIPTokenizer,  # 第二个 CLIP 词汇表
        unet: UNet2DConditionModel,  # U-Net 模型，用于生成图像
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],  # 控制网络模型，用于引导生成
        scheduler: KarrasDiffusionSchedulers,  # 调度器，控制扩散过程
        requires_aesthetics_score: bool = False,  # 是否需要美学评分，默认为 False
        force_zeros_for_empty_prompt: bool = True,  # 对于空提示强制使用零值，默认为 True
        add_watermarker: Optional[bool] = None,  # 是否添加水印，默认为 None
        feature_extractor: CLIPImageProcessor = None,  # 特征提取器，默认为 None
        image_encoder: CLIPVisionModelWithProjection = None,  # 图像编码器，默认为 None
    ):
        # 调用父类的构造函数进行初始化
        super().__init__()

        # 检查 controlnet 是否为列表或元组，如果是则将其封装为 MultiControlNetModel 对象
        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        # 注册多个模块，包括 VAE、文本编码器、tokenizer、UNet 等
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        # 计算 VAE 的缩放因子，通常用于图像尺寸调整
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建 VAE 图像处理器，设置缩放因子并开启 RGB 转换
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        # 创建控制图像处理器，设置缩放因子，开启 RGB 转换，但不进行标准化
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        # 根据输入参数或默认值确定是否添加水印
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        # 如果需要水印，则初始化水印对象
        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            # 否则将水印设置为 None
            self.watermark = None

        # 注册配置，强制空提示使用零值
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        # 注册配置，标记是否需要美学评分
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)

    # 从 StableDiffusionXLPipeline 复制的 encode_prompt 方法
    def encode_prompt(
        self,
        # 定义 prompt 字符串及其相关参数
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    # 从 StableDiffusionPipeline 复制的 encode_image 方法
    # 定义一个方法来编码图像，参数包括图像、设备、每个提示的图像数量和可选的隐藏状态输出
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 检查输入的图像是否为张量类型
            if not isinstance(image, torch.Tensor):
                # 如果不是，将其转换为张量，并提取像素值
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备并转换为相应的数据类型
            image = image.to(device=device, dtype=dtype)
            # 检查是否需要输出隐藏状态
            if output_hidden_states:
                # 获取图像编码器的隐藏状态，选择倒数第二个隐藏层
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 将隐藏状态按每个提示的图像数量重复
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 获取无条件图像编码的隐藏状态，使用全零张量作为输入
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 将无条件隐藏状态按每个提示的图像数量重复
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回图像编码的隐藏状态和无条件图像编码的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 获取图像编码的嵌入表示
                image_embeds = self.image_encoder(image).image_embeds
                # 将嵌入表示按每个提示的图像数量重复
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入同样形状的全零张量作为无条件嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和无条件图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的方法
        def prepare_ip_adapter_image_embeds(
            # 定义方法的参数，包括 IP 适配器图像、图像嵌入、设备、每个提示的图像数量和分类器自由引导的标志
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化一个空列表，用于存储图像嵌入
        image_embeds = []
        # 如果启用了无分类器自由引导，则初始化负图像嵌入列表
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器图像嵌入为 None
        if ip_adapter_image_embeds is None:
            # 检查输入适配器图像是否为列表类型，如果不是，则转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入适配器图像的长度是否与 IP 适配器数量相等
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                # 如果不相等，抛出值错误
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历输入适配器图像和相应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 确定是否输出隐藏状态，依据图像投影层的类型
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个图像，获取嵌入和负嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将图像嵌入添加到列表中，增加一个维度
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用了无分类器自由引导，则将负图像嵌入添加到列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 如果输入适配器图像嵌入已存在
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用了无分类器自由引导，将嵌入分成负嵌入和正嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 添加负图像嵌入到列表中
                    negative_image_embeds.append(single_negative_image_embeds)
                # 添加正图像嵌入到列表中
                image_embeds.append(single_image_embeds)

        # 初始化一个空列表，用于存储处理后的输入适配器图像嵌入
        ip_adapter_image_embeds = []
        # 遍历图像嵌入，执行重复操作以匹配每个提示的图像数量
        for i, single_image_embeds in enumerate(image_embeds):
            # 将单个图像嵌入沿着维度 0 重复指定次数
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用了无分类器自由引导，处理负嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负嵌入与正嵌入合并
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将嵌入移动到指定的设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的嵌入添加到列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回处理后的输入适配器图像嵌入列表
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的内容
    # 准备额外的参数用于调度器步骤，因为并非所有调度器具有相同的参数签名
        def prepare_extra_step_kwargs(self, generator, eta):
            # eta (η) 仅在 DDIMScheduler 中使用，其他调度器会忽略此参数
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 该值应在 [0, 1] 之间
    
            # 检查调度器的步骤方法是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 创建一个空字典用于存放额外参数
            extra_step_kwargs = {}
            # 如果调度器接受 eta，则将其添加到额外参数中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤方法是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果调度器接受 generator，则将其添加到额外参数中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回包含额外参数的字典
            return extra_step_kwargs
    
        # 检查输入参数的有效性
        def check_inputs(
            self,
            prompt,
            prompt_2,
            image,
            strength,
            num_inference_steps,
            callback_steps,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            controlnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            callback_on_step_end_tensor_inputs=None,
        # 从 diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl.StableDiffusionXLControlNetPipeline.check_image 复制的参数
    # 检查输入图像的类型和形状，确保与提示的批量大小一致
    def check_image(self, image, prompt, prompt_embeds):
        # 判断输入是否为 PIL 图像
        image_is_pil = isinstance(image, PIL.Image.Image)
        # 判断输入是否为 PyTorch 张量
        image_is_tensor = isinstance(image, torch.Tensor)
        # 判断输入是否为 NumPy 数组
        image_is_np = isinstance(image, np.ndarray)
        # 判断输入是否为 PIL 图像列表
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        # 判断输入是否为 PyTorch 张量列表
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        # 判断输入是否为 NumPy 数组列表
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)
    
        # 如果输入不符合任何类型，抛出类型错误
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
    
        # 如果输入为 PIL 图像，设置批量大小为 1
        if image_is_pil:
            image_batch_size = 1
        else:
            # 否则，根据输入的长度确定批量大小
            image_batch_size = len(image)
    
        # 如果提示不为 None 且为字符串，设置提示批量大小为 1
        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        # 如果提示为列表，根据列表长度设置批量大小
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        # 如果提示嵌入不为 None，使用其第一维的大小作为批量大小
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]
    
        # 如果图像批量大小不为 1，且与提示批量大小不一致，抛出值错误
        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )
    
        # 从 diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl 导入的 prepare_image 方法
        def prepare_control_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
        ):
            # 预处理输入图像并转换为指定的数据类型
            image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
            # 获取图像批量大小
            image_batch_size = image.shape[0]
    
            # 如果图像批量大小为 1，重复次数设置为 batch_size
            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # 如果图像批量大小与提示批量大小相同，设置重复次数为每个提示的图像数量
                repeat_by = num_images_per_prompt
    
            # 重复图像以匹配所需的批量大小
            image = image.repeat_interleave(repeat_by, dim=0)
    
            # 将图像转移到指定设备和数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 如果启用分类器自由引导并且不在猜测模式下，复制图像以增加维度
            if do_classifier_free_guidance and not guess_mode:
                image = torch.cat([image] * 2)
    
            # 返回处理后的图像
            return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img 导入的 get_timesteps 方法
    # 获取时间步的函数，接收推理步骤数、强度和设备参数
        def get_timesteps(self, num_inference_steps, strength, device):
            # 计算原始时间步，使用 init_timestep，确保不超过推理步骤数
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
            # 计算开始时间步，确保不小于零
            t_start = max(num_inference_steps - init_timestep, 0)
            # 从调度器获取时间步，截取从 t_start 开始的所有时间步
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            # 如果调度器具有设置开始索引的方法，则调用该方法
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)
    
            # 返回时间步和剩余的推理步骤数
            return timesteps, num_inference_steps - t_start
    
        # 从 StableDiffusionXLImg2ImgPipeline 复制的准备潜在变量的函数
        def prepare_latents(
            self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
        # 从 StableDiffusionXLImg2ImgPipeline 复制的获取附加时间 ID 的函数
        def _get_add_time_ids(
            self,
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype,
            text_encoder_projection_dim=None,
    ):
        # 检查配置是否需要美学评分
        if self.config.requires_aesthetics_score:
            # 创建包含原始大小、裁剪坐标及美学评分的列表
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            # 创建包含负样本原始大小、裁剪坐标及负美学评分的列表
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            # 创建包含原始大小、裁剪坐标和目标大小的列表
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            # 创建包含负样本原始大小、裁剪坐标及负目标大小的列表
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        # 计算通过添加时间嵌入维度和文本编码器投影维度得到的通过嵌入维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取模型期望的添加嵌入维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 检查期望的嵌入维度是否大于传递的嵌入维度，并符合特定条件
        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 抛出值错误，说明创建的嵌入维度不符合预期
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        # 检查期望的嵌入维度是否小于传递的嵌入维度，并符合特定条件
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 抛出值错误，说明创建的嵌入维度不符合预期
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        # 检查期望的嵌入维度是否与传递的嵌入维度不相等
        elif expected_add_embed_dim != passed_add_embed_dim:
            # 抛出值错误，说明模型配置不正确
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将添加的时间 ID 转换为张量，并指定数据类型
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 将添加的负时间 ID 转换为张量，并指定数据类型
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        # 返回添加的时间 ID 和添加的负时间 ID
        return add_time_ids, add_neg_time_ids

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae 复制而来
    # 定义一个方法，用于将 VAE 模型的参数类型提升
    def upcast_vae(self):
        # 获取当前 VAE 模型的数据类型
        dtype = self.vae.dtype
        # 将 VAE 模型转换为 float32 数据类型
        self.vae.to(dtype=torch.float32)
        # 检查 VAE 解码器中第一个注意力处理器的类型，以确定是否使用了特定版本的处理器
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # 如果使用了 xformers 或 torch_2_0，注意力块不需要为 float32 类型，从而节省大量内存
        if use_torch_2_0_or_xformers:
            # 将后量化卷积层转换为原始数据类型
            self.vae.post_quant_conv.to(dtype)
            # 将解码器输入卷积层转换为原始数据类型
            self.vae.decoder.conv_in.to(dtype)
            # 将解码器中间块转换为原始数据类型
            self.vae.decoder.mid_block.to(dtype)

    # 定义一个属性，返回当前的引导缩放比例
    @property
    def guidance_scale(self):
        # 返回内部存储的引导缩放比例
        return self._guidance_scale

    # 定义一个属性，返回当前的剪辑跳过值
    @property
    def clip_skip(self):
        # 返回内部存储的剪辑跳过值
        return self._clip_skip

    # 定义一个属性，用于判断是否进行无分类器引导，依据是引导缩放比例是否大于 1
    # 此属性的定义参考了 Imagen 论文中的方程 (2)
    # 当 `guidance_scale = 1` 时，相当于不进行无分类器引导
    @property
    def do_classifier_free_guidance(self):
        # 如果引导缩放比例大于 1，返回 True，否则返回 False
        return self._guidance_scale > 1

    # 定义一个属性，返回当前的交叉注意力参数
    @property
    def cross_attention_kwargs(self):
        # 返回内部存储的交叉注意力参数
        return self._cross_attention_kwargs

    # 定义一个属性，返回当前的时间步数
    @property
    def num_timesteps(self):
        # 返回内部存储的时间步数
        return self._num_timesteps

    # 装饰器，表示在执行下面的方法时不计算梯度
    @torch.no_grad()
    # 装饰器，用于替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的类方法，接受多个参数用于处理图像生成
    def __call__(
        # 主提示字符串或字符串列表，默认为 None
        self,
        prompt: Union[str, List[str]] = None,
        # 第二个提示字符串或字符串列表，默认为 None
        prompt_2: Optional[Union[str, List[str]]] = None,
        # 输入图像，用于图像生成的基础，默认为 None
        image: PipelineImageInput = None,
        # 控制图像，用于影响生成的图像，默认为 None
        control_image: PipelineImageInput = None,
        # 输出图像的高度，默认为 None
        height: Optional[int] = None,
        # 输出图像的宽度，默认为 None
        width: Optional[int] = None,
        # 图像生成的强度，默认为 0.8
        strength: float = 0.8,
        # 进行推理的步数，默认为 50
        num_inference_steps: int = 50,
        # 引导尺度，控制图像生成的引导程度，默认为 5.0
        guidance_scale: float = 5.0,
        # 负面提示字符串或字符串列表，默认为 None
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 第二个负面提示字符串或字符串列表，默认为 None
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: Optional[int] = 1,
        # 采样的 eta 值，默认为 0.0
        eta: float = 0.0,
        # 随机数生成器，可选，默认为 None
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜在变量，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 提示的嵌入向量，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负面提示的嵌入向量，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 聚合的提示嵌入向量，默认为 None
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 负面聚合提示嵌入向量，默认为 None
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 输入适配器图像，默认为 None
        ip_adapter_image: Optional[PipelineImageInput] = None,
        # 输入适配器图像的嵌入向量，默认为 None
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典，默认为 True
        return_dict: bool = True,
        # 交叉注意力参数，默认为 None
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 控制网络的条件缩放，默认为 0.8
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        # 猜测模式，默认为 False
        guess_mode: bool = False,
        # 控制引导的开始位置，默认为 0.0
        control_guidance_start: Union[float, List[float]] = 0.0,
        # 控制引导的结束位置，默认为 1.0
        control_guidance_end: Union[float, List[float]] = 1.0,
        # 原始图像的尺寸，默认为 None
        original_size: Tuple[int, int] = None,
        # 裁剪坐标的左上角，默认为 (0, 0)
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        # 目标尺寸，默认为 None
        target_size: Tuple[int, int] = None,
        # 负面原始图像的尺寸，默认为 None
        negative_original_size: Optional[Tuple[int, int]] = None,
        # 负面裁剪坐标的左上角，默认为 (0, 0)
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        # 负目标尺寸，默认为 None
        negative_target_size: Optional[Tuple[int, int]] = None,
        # 审美分数，默认为 6.0
        aesthetic_score: float = 6.0,
        # 负面审美分数，默认为 2.5
        negative_aesthetic_score: float = 2.5,
        # 跳过的剪辑层数，默认为 None
        clip_skip: Optional[int] = None,
        # 步骤结束时的回调函数，可选，默认为 None
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        # 结束步骤时的张量输入回调，默认为 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 其他额外参数，默认为空
        **kwargs,
```