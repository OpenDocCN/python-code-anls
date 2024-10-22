# `.\diffusers\pipelines\stable_diffusion_gligen\pipeline_stable_diffusion_gligen_text_image.py`

```py
# 版权所有 2024 GLIGEN 作者及 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有规定，
# 否则根据许可证分发的软件是在“按现状”基础上分发的，
# 不提供任何形式的明示或暗示的担保或条件。
# 有关许可证所涵盖的特定权限和限制，请参阅许可证。

import inspect  # 导入 inspect 模块，用于获取对象的信息
import warnings  # 导入 warnings 模块，用于发出警告
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示所需的类型

import PIL.Image  # 导入 PIL.Image 模块，用于图像处理
import torch  # 导入 PyTorch 库，用于张量运算
from transformers import (  # 从 transformers 库中导入多个类和函数
    CLIPImageProcessor,  # 图像处理器，用于 CLIP 模型
    CLIPProcessor,  # 通用 CLIP 处理器
    CLIPTextModel,  # CLIP 文本模型
    CLIPTokenizer,  # CLIP 令牌化器
    CLIPVisionModelWithProjection,  # CLIP 视觉模型，包含投影功能
)

from ...image_processor import VaeImageProcessor  # 导入 VAE 图像处理器
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入用于加载的混合类
from ...models import AutoencoderKL, UNet2DConditionModel  # 导入模型类
from ...models.attention import GatedSelfAttentionDense  # 导入密集门控自注意力类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 LORA 缩放的函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器
from ...utils import USE_PEFT_BACKEND, logging, replace_example_docstring, scale_lora_layers, unscale_lora_layers  # 导入工具函数和常量
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和稳定扩散混合类
from ..stable_diffusion import StableDiffusionPipelineOutput  # 导入稳定扩散管道输出类
from ..stable_diffusion.clip_image_project_model import CLIPImageProjection  # 导入 CLIP 图像投影模型
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 导入稳定扩散安全检查器


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，使用 pylint 禁用无效名称警告

EXAMPLE_DOC_STRING = """  # 示例文档字符串，可能用于说明使用方式或示例
"""


class StableDiffusionGLIGENTextImagePipeline(DiffusionPipeline, StableDiffusionMixin):  # 定义一个类，继承自 DiffusionPipeline 和 StableDiffusionMixin
    r"""  # 类的文档字符串，说明该类的功能
    使用 Stable Diffusion 进行文本到图像生成的管道，结合基于语言的图像生成（GLIGEN）。

    此模型继承自 [`DiffusionPipeline`]。有关库为所有管道实现的通用方法（例如下载或保存、在特定设备上运行等），请查看超类文档。
    # 定义参数列表及其说明
        Args:
            vae ([`AutoencoderKL`]):
                用于将图像编码和解码为潜在表示的变分自编码器 (VAE) 模型。
            text_encoder ([`~transformers.CLIPTextModel`]):
                冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
            tokenizer ([`~transformers.CLIPTokenizer`]):
                用于对文本进行分词的 `CLIPTokenizer`。
            processor ([`~transformers.CLIPProcessor`]):
                用于处理参考图像的 `CLIPProcessor`。
            image_encoder ([`transformers.CLIPVisionModelWithProjection`]):
                冻结的图像编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
            image_project ([`CLIPImageProjection`]):
                用于将图像嵌入投影到短语嵌入空间的 `CLIPImageProjection`。
            unet ([`UNet2DConditionModel`]):
                用于去噪编码图像潜在表示的 `UNet2DConditionModel`。
            scheduler ([`SchedulerMixin`]):
                与 `unet` 结合使用以去噪编码图像潜在表示的调度器。可以是
                [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 中的一个。
            safety_checker ([`StableDiffusionSafetyChecker`]):
                分类模块，用于评估生成的图像是否可能被视为冒犯或有害。
                请参阅 [模型卡](https://huggingface.co/runwayml/stable-diffusion-v1-5) 获取有关模型潜在危害的更多细节。
            feature_extractor ([`~transformers.CLIPImageProcessor`]):
                用于从生成图像中提取特征的 `CLIPImageProcessor`；用于作为 `safety_checker` 的输入。
        """
    
        # 定义模型在 CPU 上的卸载顺序
        model_cpu_offload_seq = "text_encoder->unet->vae"
        # 定义可选组件列表
        _optional_components = ["safety_checker", "feature_extractor"]
        # 定义不包含在 CPU 卸载中的组件
        _exclude_from_cpu_offload = ["safety_checker"]
    
        # 初始化方法
        def __init__(
            # 接收变分自编码器
            self,
            vae: AutoencoderKL,
            # 接收文本编码器
            text_encoder: CLIPTextModel,
            # 接收分词器
            tokenizer: CLIPTokenizer,
            # 接收图像处理器
            processor: CLIPProcessor,
            # 接收图像编码器
            image_encoder: CLIPVisionModelWithProjection,
            # 接收图像投影器
            image_project: CLIPImageProjection,
            # 接收 U-Net 模型
            unet: UNet2DConditionModel,
            # 接收调度器
            scheduler: KarrasDiffusionSchedulers,
            # 接收安全检查器
            safety_checker: StableDiffusionSafetyChecker,
            # 接收特征提取器
            feature_extractor: CLIPImageProcessor,
            # 是否需要安全检查器的布尔值
            requires_safety_checker: bool = True,
    # 初始化父类
        ):
            super().__init__()
    
            # 如果未提供安全检查器且需要安全检查器，记录警告信息
            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    # 生成警告内容，提醒用户安全检查器已禁用，并提供相关指导信息
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 如果提供了安全检查器但未提供特征提取器，抛出错误
            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    # 提示用户在加载该类时需要定义特征提取器
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )
    
            # 注册各个模块，初始化相关组件
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                image_encoder=image_encoder,
                processor=processor,
                image_project=image_project,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建 VAE 图像处理器，设置为 RGB 转换
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
            # 将所需的安全检查器信息注册到配置中
            self.register_to_config(requires_safety_checker=requires_safety_checker)
    
        # 从 StableDiffusionPipeline 复制的 encode_prompt 方法
        def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            lora_scale: Optional[float] = None,
            clip_skip: Optional[int] = None,
        # 从 StableDiffusionPipeline 复制的 run_safety_checker 方法
    # 执行安全检查器，检查输入图像是否符合安全标准
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，则初始化为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入是张量格式，则进行后处理为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入是 NumPy 数组，则转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 提取特征并将输入转移到指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 使用安全检查器处理图像，并获取是否存在不当内容的概念
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和不当内容概念
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    # 准备调度步骤的额外参数，因为并非所有调度器的参数签名相同
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta（η）仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # eta 的取值范围应在 [0, 1] 之间

        # 检查调度器是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            # 如果接受 eta，则将其添加到额外参数中
            extra_step_kwargs["eta"] = eta

        # 检查调度器是否接受生成器参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            # 如果接受生成器，则将其添加到额外参数中
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数字典
        return extra_step_kwargs

    # 从 diffusers.pipelines.stable_diffusion_k_diffusion.pipeline_stable_diffusion_k_diffusion.StableDiffusionKDiffusionPipeline.check_inputs 复制
    # 检查输入参数的有效性和一致性
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        # 检查高度和宽度是否都能被 8 整除
        if height % 8 != 0 or width % 8 != 0:
            # 如果不能整除，抛出一个值错误，说明高度和宽度不符合要求
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步骤是否被设置，并且检查它是否为正整数
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            # 如果条件不满足，抛出值错误，说明回调步骤无效
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        
        # 检查回调结束时的张量输入是否有效
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 如果有无效的输入，抛出值错误，说明输入不在允许的范围内
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查是否同时提供了提示和提示嵌入
        if prompt is not None and prompt_embeds is not None:
            # 如果同时提供，抛出值错误，说明只能提供其中一个
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            # 如果都没有提供，抛出值错误，说明至少需要提供一个
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 检查提示的类型是否有效
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了负提示和负提示嵌入
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 如果同时提供，抛出值错误，说明只能提供其中一个
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查提示嵌入和负提示嵌入的形状是否一致
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 如果形状不一致，抛出值错误，说明形状必须相同
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的代码
    # 准备潜在变量的形状和初始值
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在变量的形状，包括批量大小和通道数
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器列表的长度是否与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果没有提供潜在变量，则生成随机张量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果提供了潜在变量，则将其移动到指定设备
                latents = latents.to(device)
    
            # 将初始噪声按调度器要求的标准差进行缩放
            latents = latents * self.scheduler.init_noise_sigma
            # 返回准备好的潜在变量
            return latents
    
        # 启用或禁用门控自注意力模块
        def enable_fuser(self, enabled=True):
            # 遍历 UNet 模块
            for module in self.unet.modules():
                # 检查模块类型是否为 GatedSelfAttentionDense
                if type(module) is GatedSelfAttentionDense:
                    # 设置模块的启用状态
                    module.enabled = enabled
    
        # 根据给定的框创建修复掩码
        def draw_inpaint_mask_from_boxes(self, boxes, size):
            """
            Create an inpainting mask based on given boxes. This function generates an inpainting mask using the provided
            boxes to mark regions that need to be inpainted.
            """
            # 创建一个全白的修复掩码
            inpaint_mask = torch.ones(size[0], size[1])
            # 遍历每个框
            for box in boxes:
                # 根据框计算对应的像素坐标
                x0, x1 = box[0] * size[0], box[2] * size[0]
                y0, y1 = box[1] * size[1], box[3] * size[1]
                # 在掩码上标记需要修复的区域
                inpaint_mask[int(y0) : int(y1), int(x0) : int(x1)] = 0
            # 返回修复掩码
            return inpaint_mask
    
        # 裁剪输入图像到指定尺寸
        def crop(self, im, new_width, new_height):
            """
            Crop the input image to the specified dimensions.
            """
            # 获取原始图像的宽度和高度
            width, height = im.size
            # 计算裁剪区域的左、上、右、下边界
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2
            # 返回裁剪后的图像
            return im.crop((left, top, right, bottom))
    
        # 裁剪并调整图像到目标尺寸，保持中心
        def target_size_center_crop(self, im, new_hw):
            """
            Crop and resize the image to the target size while keeping the center.
            """
            # 获取图像的宽度和高度
            width, height = im.size
            # 如果宽高不相等，进行中心裁剪
            if width != height:
                im = self.crop(im, min(height, width), min(height, width))
            # 返回调整后的图像
            return im.resize((new_hw, new_hw), PIL.Image.LANCZOS)
    # 根据输入的掩码值（0或1）为每个短语和图像掩蔽特征
    def complete_mask(self, has_mask, max_objs, device):
        # 创建一个全1的掩码，形状为(1, max_objs)，数据类型与文本编码器一致，转移到指定设备
        mask = torch.ones(1, max_objs).type(self.text_encoder.dtype).to(device)
        # 如果没有掩码，则返回全1的掩码
        if has_mask is None:
            return mask
    
        # 如果掩码是一个整数，则返回乘以该整数的掩码
        if isinstance(has_mask, int):
            return mask * has_mask
        else:
            # 遍历掩码列表，将值填入掩码中
            for idx, value in enumerate(has_mask):
                mask[0, idx] = value
            # 返回填充后的掩码
            return mask
    
    # 使用 CLIP 预训练模型获取图像和短语的嵌入
    def get_clip_feature(self, input, normalize_constant, device, is_image=False):
        # 如果处理的是图像
        if is_image:
            # 如果输入为 None，返回 None
            if input is None:
                return None
            # 处理图像输入，转换为张量并转移到设备
            inputs = self.processor(images=[input], return_tensors="pt").to(device)
            # 将像素值转换为图像编码器的数据类型
            inputs["pixel_values"] = inputs["pixel_values"].to(self.image_encoder.dtype)
    
            # 使用图像编码器获取嵌入输出
            outputs = self.image_encoder(**inputs)
            # 提取图像嵌入
            feature = outputs.image_embeds
            # 通过投影将特征转化并压缩维度
            feature = self.image_project(feature).squeeze(0)
            # 归一化特征并乘以归一化常数
            feature = (feature / feature.norm()) * normalize_constant
            # 添加维度以符合输出要求
            feature = feature.unsqueeze(0)
        else:
            # 如果处理的是文本
            if input is None:
                return None
            # 将文本输入转换为张量并转移到设备
            inputs = self.tokenizer(input, return_tensors="pt", padding=True).to(device)
            # 使用文本编码器获取嵌入输出
            outputs = self.text_encoder(**inputs)
            # 提取池化输出作为特征
            feature = outputs.pooler_output
        # 返回提取的特征
        return feature
    
    # 定义获取带有基础的交叉注意力参数的方法
    def get_cross_attention_kwargs_with_grounded(
        self,
        hidden_size,
        gligen_phrases,
        gligen_images,
        gligen_boxes,
        input_phrases_mask,
        input_images_mask,
        repeat_batch,
        normalize_constant,
        max_objs,
        device,
    ):
        """
        准备交叉注意力的关键字参数，包含有关基础输入的信息（框，掩码，图像嵌入，短语嵌入）。
        """
        # 将输入的短语和图像分别赋值给变量
        phrases, images = gligen_phrases, gligen_images
        # 如果图像为 None，则为每个短语创建一个 None 列表
        images = [None] * len(phrases) if images is None else images
        # 如果短语为 None，则为每个图像创建一个 None 列表
        phrases = [None] * len(images) if phrases is None else phrases

        # 创建一个张量用于存储每个对象的框（四个坐标）
        boxes = torch.zeros(max_objs, 4, device=device, dtype=self.text_encoder.dtype)
        # 创建一个张量用于存储每个对象的掩码
        masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        # 创建一个张量用于存储每个短语的掩码
        phrases_masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        # 创建一个张量用于存储每个图像的掩码
        image_masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        # 创建一个张量用于存储每个短语的嵌入
        phrases_embeddings = torch.zeros(max_objs, hidden_size, device=device, dtype=self.text_encoder.dtype)
        # 创建一个张量用于存储每个图像的嵌入
        image_embeddings = torch.zeros(max_objs, hidden_size, device=device, dtype=self.text_encoder.dtype)

        # 初始化存储文本特征和图像特征的列表
        text_features = []
        image_features = []
        # 遍历短语和图像，获取特征
        for phrase, image in zip(phrases, images):
            # 获取短语的特征并添加到列表
            text_features.append(self.get_clip_feature(phrase, normalize_constant, device, is_image=False))
            # 获取图像的特征并添加到列表
            image_features.append(self.get_clip_feature(image, normalize_constant, device, is_image=True))

        # 遍历框、文本特征和图像特征，填充相应的张量
        for idx, (box, text_feature, image_feature) in enumerate(zip(gligen_boxes, text_features, image_features)):
            # 将框转换为张量并赋值
            boxes[idx] = torch.tensor(box)
            # 设置掩码为 1
            masks[idx] = 1
            # 如果文本特征不为空，则赋值并设置掩码
            if text_feature is not None:
                phrases_embeddings[idx] = text_feature
                phrases_masks[idx] = 1
            # 如果图像特征不为空，则赋值并设置掩码
            if image_feature is not None:
                image_embeddings[idx] = image_feature
                image_masks[idx] = 1

        # 完成输入短语的掩码
        input_phrases_mask = self.complete_mask(input_phrases_mask, max_objs, device)
        # 通过重复输入短语的掩码来扩展短语掩码
        phrases_masks = phrases_masks.unsqueeze(0).repeat(repeat_batch, 1) * input_phrases_mask
        # 完成输入图像的掩码
        input_images_mask = self.complete_mask(input_images_mask, max_objs, device)
        # 通过重复输入图像的掩码来扩展图像掩码
        image_masks = image_masks.unsqueeze(0).repeat(repeat_batch, 1) * input_images_mask
        # 通过重复来扩展框的维度
        boxes = boxes.unsqueeze(0).repeat(repeat_batch, 1, 1)
        # 通过重复来扩展掩码的维度
        masks = masks.unsqueeze(0).repeat(repeat_batch, 1)
        # 通过重复来扩展短语嵌入的维度
        phrases_embeddings = phrases_embeddings.unsqueeze(0).repeat(repeat_batch, 1, 1)
        # 通过重复来扩展图像嵌入的维度
        image_embeddings = image_embeddings.unsqueeze(0).repeat(repeat_batch, 1, 1)

        # 将所有处理后的数据组织成字典
        out = {
            "boxes": boxes,
            "masks": masks,
            "phrases_masks": phrases_masks,
            "image_masks": image_masks,
            "phrases_embeddings": phrases_embeddings,
            "image_embeddings": image_embeddings,
        }

        # 返回包含所有信息的字典
        return out
    # 定义一个方法，用于获取无基于输入信息的交叉注意力参数
    def get_cross_attention_kwargs_without_grounded(self, hidden_size, repeat_batch, max_objs, device):
        """
        准备无关于基础输入（框、掩码、图像嵌入、短语嵌入）信息的交叉注意力参数（均为零张量）。
        """
        # 创建一个形状为 (max_objs, 4) 的全零张量，用于表示物体框
        boxes = torch.zeros(max_objs, 4, device=device, dtype=self.text_encoder.dtype)
        # 创建一个形状为 (max_objs,) 的全零张量，用于表示物体掩码
        masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        # 创建一个形状为 (max_objs,) 的全零张量，用于表示短语掩码
        phrases_masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        # 创建一个形状为 (max_objs,) 的全零张量，用于表示图像掩码
        image_masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        # 创建一个形状为 (max_objs, hidden_size) 的全零张量，用于表示短语嵌入
        phrases_embeddings = torch.zeros(max_objs, hidden_size, device=device, dtype=self.text_encoder.dtype)
        # 创建一个形状为 (max_objs, hidden_size) 的全零张量，用于表示图像嵌入
        image_embeddings = torch.zeros(max_objs, hidden_size, device=device, dtype=self.text_encoder.dtype)

        # 创建一个字典，包含多个张量，均为扩展并重复的零张量
        out = {
            # 扩展 boxes 张量并重复，生成形状为 (repeat_batch, max_objs, 4)
            "boxes": boxes.unsqueeze(0).repeat(repeat_batch, 1, 1),
            # 扩展 masks 张量并重复，生成形状为 (repeat_batch, max_objs)
            "masks": masks.unsqueeze(0).repeat(repeat_batch, 1),
            # 扩展 phrases_masks 张量并重复，生成形状为 (repeat_batch, max_objs)
            "phrases_masks": phrases_masks.unsqueeze(0).repeat(repeat_batch, 1),
            # 扩展 image_masks 张量并重复，生成形状为 (repeat_batch, max_objs)
            "image_masks": image_masks.unsqueeze(0).repeat(repeat_batch, 1),
            # 扩展 phrases_embeddings 张量并重复，生成形状为 (repeat_batch, max_objs, hidden_size)
            "phrases_embeddings": phrases_embeddings.unsqueeze(0).repeat(repeat_batch, 1, 1),
            # 扩展 image_embeddings 张量并重复，生成形状为 (repeat_batch, max_objs, hidden_size)
            "image_embeddings": image_embeddings.unsqueeze(0).repeat(repeat_batch, 1, 1),
        }

        # 返回包含交叉注意力参数的字典
        return out

    # 装饰器，禁用梯度计算以减少内存使用
    @torch.no_grad()
    # 装饰器，用于替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义调用方法，支持多种参数
    def __call__(
        # 可选的字符串或字符串列表，作为提示输入
        prompt: Union[str, List[str]] = None,
        # 可选的整数，指定生成图像的高度
        height: Optional[int] = None,
        # 可选的整数，指定生成图像的宽度
        width: Optional[int] = None,
        # 指定推理步骤的数量，默认为 50
        num_inference_steps: int = 50,
        # 指定引导比例，默认为 7.5
        guidance_scale: float = 7.5,
        # 指定 Gligen 计划采样的 beta 值，默认为 0.3
        gligen_scheduled_sampling_beta: float = 0.3,
        # 可选的短语列表，用于 Gligen
        gligen_phrases: List[str] = None,
        # 可选的图像列表，用于 Gligen
        gligen_images: List[PIL.Image.Image] = None,
        # 可选的短语掩码，单个整数或整数列表
        input_phrases_mask: Union[int, List[int]] = None,
        # 可选的图像掩码，单个整数或整数列表
        input_images_mask: Union[int, List[int]] = None,
        # 可选的边框列表，用于 Gligen
        gligen_boxes: List[List[float]] = None,
        # 可选的图像，用于填充，Gligen
        gligen_inpaint_image: Optional[PIL.Image.Image] = None,
        # 可选的负提示，单个字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 可选的整数，指定每个提示生成的图像数量，默认为 1
        num_images_per_prompt: Optional[int] = 1,
        # 指定噪声比例，默认为 0.0
        eta: float = 0.0,
        # 可选的生成器，用于随机数生成
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 可选的潜在张量
        latents: Optional[torch.Tensor] = None,
        # 可选的提示嵌入张量
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负提示嵌入张量
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 可选的布尔值，指定是否返回字典格式
        return_dict: bool = True,
        # 可选的回调函数，用于处理中间结果
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 指定回调函数调用的步长，默认为 1
        callback_steps: int = 1,
        # 可选的字典，包含交叉注意力的参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # Gligen 正常化常量，默认为 28.7
        gligen_normalize_constant: float = 28.7,
        # 可选的整数，指定跳过的剪辑步骤
        clip_skip: int = None,
```