# `.\diffusers\pipelines\stable_diffusion\pipeline_stable_diffusion_depth2img.py`

```py
# 版权信息，声明该文件的所有权归 HuggingFace 团队所有
# 许可信息，指明该文件遵循 Apache License 2.0
# 说明在使用该文件时需遵循该许可证的条款
# 可通过下面的链接获取许可证
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则该软件按“原样”提供，不附带任何明示或暗示的担保或条件
# 查看许可证以获取特定语言管理权限和限制的详细信息

import contextlib  # 导入上下文管理库，用于处理上下文
import inspect  # 导入检查库，用于获取对象的信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示相关工具

import numpy as np  # 导入 NumPy 库，用于数组和数值计算
import PIL.Image  # 导入 PIL 库中的 Image 模块，用于图像处理
import torch  # 导入 PyTorch 库，用于深度学习计算
from packaging import version  # 导入版本控制工具，用于处理版本信息
from transformers import CLIPTextModel, CLIPTokenizer, DPTForDepthEstimation, DPTImageProcessor  # 导入 Transformers 库中的模型和处理器

from ...configuration_utils import FrozenDict  # 从配置工具中导入 FrozenDict，用于不可变字典
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入图像处理相关工具
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入加载器工具，用于模型加载
from ...models import AutoencoderKL, UNet2DConditionModel  # 导入模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入 LoRA 调整工具
from ...schedulers import KarrasDiffusionSchedulers  # 导入调度器工具
from ...utils import PIL_INTERPOLATION, USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers  # 导入实用工具
from ...utils.torch_utils import randn_tensor  # 从 PyTorch 工具导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 导入扩散管道和图像输出工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    # 检查 encoder_output 是否具有 latent_dist 属性且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)  # 从潜在分布中采样
    # 检查 encoder_output 是否具有 latent_dist 属性且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()  # 返回潜在分布的众数
    # 检查 encoder_output 是否具有 latents 属性
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents  # 返回潜在表示
    else:
        raise AttributeError("Could not access latents of provided encoder_output")  # 如果没有访问到潜在表示，抛出异常

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess 复制的函数
def preprocess(image):
    # 设置弃用提示信息
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    # 调用弃用函数记录弃用信息
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    # 检查输入是否为 PyTorch 张量
    if isinstance(image, torch.Tensor):
        return image  # 如果是张量，则直接返回
    # 检查输入是否为 PIL 图像
    elif isinstance(image, PIL.Image.Image):
        image = [image]  # 如果是图像，则将其封装为列表
    # 检查 image 列表的第一个元素是否为 PIL 的图像对象
    if isinstance(image[0], PIL.Image.Image):
        # 获取图像的宽度和高度
        w, h = image[0].size
        # 将宽度和高度调整为 8 的整数倍
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        # 将每个图像调整为新的宽高，并转换为 NumPy 数组，增加一个新的维度
        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        # 将所有图像沿第 0 维连接成一个大数组
        image = np.concatenate(image, axis=0)
        # 将数组转换为浮点数并归一化到 [0, 1] 范围
        image = np.array(image).astype(np.float32) / 255.0
        # 调整数组的维度顺序，从 (N, H, W, C) 转换为 (N, C, H, W)
        image = image.transpose(0, 3, 1, 2)
        # 将图像数据的值范围从 [0, 1] 变换到 [-1, 1]
        image = 2.0 * image - 1.0
        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
    # 检查 image 列表的第一个元素是否为 PyTorch 张量
    elif isinstance(image[0], torch.Tensor):
        # 沿第 0 维连接所有 PyTorch 张量
        image = torch.cat(image, dim=0)
    # 返回处理后的图像
    return image
# 定义一个名为 StableDiffusionDepth2ImgPipeline 的类，继承多个混合类
class StableDiffusionDepth2ImgPipeline(DiffusionPipeline, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin):
    r"""
    使用稳定扩散进行基于深度的图像生成的管道，支持文本引导。

    该模型继承自 [`DiffusionPipeline`]，可查看超类文档以了解所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    此管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重

    参数：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器 (VAE) 模型。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于对文本进行标记化的 `CLIPTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于去噪编码图像潜在表示的 `UNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 结合使用的调度器，以去噪编码图像潜在表示。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
    """

    # 定义模型 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义需要作为回调的张量输入
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "depth_mask"]

    # 初始化方法，接受多个参数以配置管道
    def __init__(
        self,
        vae: AutoencoderKL,  # 变分自编码器模型
        text_encoder: CLIPTextModel,  # 文本编码器
        tokenizer: CLIPTokenizer,  # 文本标记化工具
        unet: UNet2DConditionModel,  # 去噪模型
        scheduler: KarrasDiffusionSchedulers,  # 调度器
        depth_estimator: DPTForDepthEstimation,  # 深度估计模型
        feature_extractor: DPTImageProcessor,  # 特征提取器
    # 定义构造函数
        ):
            # 调用父类的构造函数
            super().__init__()
    
            # 检查 unet 配置是否有 diffusers 版本属性，并判断其是否小于 0.9.0
            is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
                version.parse(unet.config._diffusers_version).base_version
            ) < version.parse("0.9.0.dev0")
            # 检查 unet 配置的样本大小是否小于 64
            is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
            # 如果 unet 版本小于 0.9.0 且样本大小小于 64，则给出弃用警告
            if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
                # 创建弃用消息，提示用户更新配置文件
                deprecation_message = (
                    "The configuration file of the unet has set the default `sample_size` to smaller than"
                    " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                    " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                    " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                    " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                    " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                    " in the config might lead to incorrect results in future versions. If you have downloaded this"
                    " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                    " the `unet/config.json` file"
                )
                # 调用弃用函数，记录样本大小小于 64 的警告
                deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
                # 创建新的配置字典，修改样本大小为 64
                new_config = dict(unet.config)
                new_config["sample_size"] = 64
                # 更新 unet 的内部字典
                unet._internal_dict = FrozenDict(new_config)
    
            # 注册各个模块
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                depth_estimator=depth_estimator,
                feature_extractor=feature_extractor,
            )
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建 VAE 图像处理器实例
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
        # 从 StableDiffusionPipeline 的 _encode_prompt 方法复制
        def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            # 可选参数，用于嵌入提示和负面提示
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            lora_scale: Optional[float] = None,
            # 接收额外的关键字参数
            **kwargs,
    # 结束括号，表示函数参数列表的结束
        ):
            # 警告信息，说明 `_encode_prompt()` 已被弃用，未来版本中将移除，建议使用 `encode_prompt()`
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用 deprecate 函数记录弃用信息，指定版本和自定义警告选项
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用 encode_prompt 方法生成提示嵌入元组，传入多个参数
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt,  # 输入提示
                device=device,  # 设备信息
                num_images_per_prompt=num_images_per_prompt,  # 每个提示的图像数量
                do_classifier_free_guidance=do_classifier_free_guidance,  # 是否进行无分类器引导
                negative_prompt=negative_prompt,  # 负提示内容
                prompt_embeds=prompt_embeds,  # 提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,  # 负提示嵌入
                lora_scale=lora_scale,  # Lora 缩放因子
                **kwargs,  # 其他关键字参数
            )
    
            # 连接嵌入元组中的两个部分，以支持向后兼容
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回组合后的提示嵌入
            return prompt_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制的函数
        def encode_prompt(
            self,
            prompt,  # 输入提示
            device,  # 设备信息
            num_images_per_prompt,  # 每个提示的图像数量
            do_classifier_free_guidance,  # 是否进行无分类器引导
            negative_prompt=None,  # 负提示内容，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入
            lora_scale: Optional[float] = None,  # 可选的 Lora 缩放因子
            clip_skip: Optional[int] = None,  # 可选的剪裁跳过参数
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的函数
        def run_safety_checker(self, image, device, dtype):  # 安全检查器函数，检查输入图像的安全性
            # 检查安全检查器是否存在，如果不存在则将 nsfw 概念标记为 None
            if self.safety_checker is None:
                has_nsfw_concept = None
            else:
                # 如果图像是张量，使用图像处理器进行后处理并转换为 PIL 格式
                if torch.is_tensor(image):
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    # 如果不是张量，则将 NumPy 数组转换为 PIL 图像
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 使用特征提取器处理图像输入，并将其转换为指定设备的张量
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 调用安全检查器，检查图像并返回处理后的图像和 nsfw 概念标记
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回检查后的图像和 nsfw 概念标记
            return image, has_nsfw_concept
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制的函数
    # 解码潜在变量的函数
    def decode_latents(self, latents):
        # 定义一个弃用警告信息，提示用户该方法将在未来版本中移除
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用弃用函数，发出警告，指明该方法弃用的版本
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 根据 VAE 的缩放因子调整潜在变量
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在变量，获取解码后的图像，返回的结果是一个元组，取第一个元素
        image = self.vae.decode(latents, return_dict=False)[0]
        # 对图像进行归一化处理，将值范围调整到 [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像数据转为 float32 类型，便于与 bfloat16 兼容，且不会造成显著开销
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像数据
        return image

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的函数
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的参数，因为并非所有调度器的参数签名相同
        # eta (η) 仅用于 DDIMScheduler，其他调度器将忽略此参数
        # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
        # eta 的取值应在 [0, 1] 之间

        # 检查调度器的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 创建一个字典以存放额外的步骤参数
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs

    # 检查输入参数的函数
    def check_inputs(
        self,
        prompt,  # 输入的提示文本
        strength,  # 强度参数
        callback_steps,  # 回调步骤
        negative_prompt=None,  # 可选的负面提示文本
        prompt_embeds=None,  # 可选的提示嵌入
        negative_prompt_embeds=None,  # 可选的负面提示嵌入
        callback_on_step_end_tensor_inputs=None,  # 可选的回调输入
    ):
        # 检查 strength 是否在有效范围内 [0.0, 1.0]
        if strength < 0 or strength > 1:
            # 如果不在范围内，抛出值错误
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # 检查 callback_steps 是否为正整数
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            # 如果不是正整数，抛出值错误
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查 callback_on_step_end_tensor_inputs 是否在允许的回调张量输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 如果有不在允许输入中的项，抛出值错误
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        
        # 检查同时提供 prompt 和 prompt_embeds 是否有效
        if prompt is not None and prompt_embeds is not None:
            # 抛出值错误，提醒只能提供其中一个
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            # 抛出值错误，提醒必须提供其中一个
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 检查 prompt 的类型是否有效
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查同时提供 negative_prompt 和 negative_prompt_embeds 是否有效
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 抛出值错误，提醒只能提供其中一个
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 的形状是否一致
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 抛出值错误，提醒两者形状必须一致
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps 复制的代码
    # 获取时间步，进行推理步骤的处理
    def get_timesteps(self, num_inference_steps, strength, device):
        # 计算初始时间步，取 num_inference_steps 与 strength 的乘积和 num_inference_steps 的最小值
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
        # 计算开始时间步，确保不小于0
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从调度器中获取相应时间步的切片
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        # 如果调度器有设置开始索引的方法，则调用它
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
    
        # 返回时间步和剩余的推理步骤数
        return timesteps, num_inference_steps - t_start
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.prepare_latents 中复制的
    # 准备深度图，处理输入图像和深度图，适应批量大小及其他参数
    def prepare_depth_map(self, image, depth_map, batch_size, do_classifier_free_guidance, dtype, device):
        # 如果输入的图像是单个 PIL 图像，则将其转换为列表
        if isinstance(image, PIL.Image.Image):
            image = [image]
        else:
            # 如果输入是多个图像，则将其转换为列表
            image = list(image)

        # 检查图像的类型并获取宽度和高度
        if isinstance(image[0], PIL.Image.Image):
            width, height = image[0].size  # 从 PIL 图像中获取宽高
        elif isinstance(image[0], np.ndarray):
            width, height = image[0].shape[:-1]  # 从 numpy 数组中获取宽高
        else:
            height, width = image[0].shape[-2:]  # 从其他格式中获取宽高

        # 如果没有提供深度图，则计算深度图
        if depth_map is None:
            # 使用特征提取器提取图像的像素值，并将其转换为张量
            pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
            # 将像素值移动到指定的设备并转换为指定的数据类型
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            # DPT-Hybrid 模型使用批量归一化层，不支持 fp16，因此使用自动混合精度
            if torch.backends.mps.is_available():
                autocast_ctx = contextlib.nullcontext()  # 创建一个空上下文
                logger.warning(
                    "The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16, but autocast is not yet supported on MPS."
                )  # 记录警告
            else:
                # 在支持的设备上创建自动混合精度上下文
                autocast_ctx = torch.autocast(device.type, dtype=dtype)

            with autocast_ctx:  # 进入自动混合精度上下文
                # 使用深度估计器计算深度图
                depth_map = self.depth_estimator(pixel_values).predicted_depth
        else:
            # 如果提供了深度图，则将其移动到指定的设备和数据类型
            depth_map = depth_map.to(device=device, dtype=dtype)

        # 调整深度图的大小以适应 VAE 的缩放因子
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),  # 增加一个维度以适应插值操作
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),  # 目标大小
            mode="bicubic",  # 使用双三次插值
            align_corners=False,  # 不对齐角点
        )

        # 计算深度图的最小值和最大值
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)  # 获取深度图的最小值
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)  # 获取深度图的最大值
        # 将深度图归一化到 [-1, 1] 的范围
        depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
        # 将深度图转换为指定的数据类型
        depth_map = depth_map.to(dtype)

        # 如果深度图的批量大小小于给定的批量大小，则重复深度图以匹配批量大小
        if depth_map.shape[0] < batch_size:
            repeat_by = batch_size // depth_map.shape[0]  # 计算重复次数
            depth_map = depth_map.repeat(repeat_by, 1, 1, 1)  # 重复深度图

        # 根据是否使用无分类器引导来调整深度图
        depth_map = torch.cat([depth_map] * 2) if do_classifier_free_guidance else depth_map
        # 返回处理后的深度图
        return depth_map

    # 返回指导缩放因子
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 返回剪辑跳过的参数
    @property
    def clip_skip(self):
        return self._clip_skip

    # 这里的 `guidance_scale` 定义类似于 Imagen 论文中公式 (2) 的指导权重 `w`
    # `guidance_scale = 1` 表示不使用无分类器引导
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1  # 判断是否使用无分类器引导

    # 返回交叉注意力的参数
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 返回时间步数
    @property
    def num_timesteps(self):
        return self._num_timesteps
    # 使用装饰器禁用梯度计算，以节省内存和计算资源
        @torch.no_grad()
        # 定义可调用方法，接受多个参数以生成图像
        def __call__(
            # 输入提示，字符串或字符串列表，决定生成内容
            self,
            prompt: Union[str, List[str]] = None,
            # 输入图像，类型为 PipelineImageInput，用于图像生成
            image: PipelineImageInput = None,
            # 深度图，类型为可选的 torch.Tensor，用于提供深度信息
            depth_map: Optional[torch.Tensor] = None,
            # 强度参数，决定生成的图像变化程度，默认为 0.8
            strength: float = 0.8,
            # 推理步骤数，决定生成过程的迭代次数，默认为 50
            num_inference_steps: Optional[int] = 50,
            # 指导比例，用于调整生成图像与提示的一致性，默认为 7.5
            guidance_scale: Optional[float] = 7.5,
            # 负向提示，字符串或字符串列表，提供生成限制条件
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 随机性参数，控制生成过程中的随机性，默认为 0.0
            eta: Optional[float] = 0.0,
            # 生成器，用于控制随机数生成的可选参数
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 提示的嵌入，类型为可选的 torch.Tensor，提供编码后的提示信息
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负向提示的嵌入，类型为可选的 torch.Tensor，提供编码后的负向提示信息
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"，指示生成结果的格式
            output_type: Optional[str] = "pil",
            # 返回字典标志，决定是否以字典形式返回结果，默认为 True
            return_dict: bool = True,
            # 交叉注意力的额外参数，可选字典类型
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 跳过的剪辑层数，可选整数，控制模型层的使用
            clip_skip: Optional[int] = None,
            # 每一步结束时的回调函数，接受步数、总步数和字典作为参数
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 在回调函数中包含的张量输入名称，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 额外的关键字参数，允许用户自定义输入
            **kwargs,
```