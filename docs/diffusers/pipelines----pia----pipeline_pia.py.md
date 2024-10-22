# `.\diffusers\pipelines\pia\pipeline_pia.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 根据许可证分发是在“按现状”基础上提供的，
# 不附带任何形式的明示或暗示的担保或条件。
# 请参阅许可证以获取有关权限和
# 限制的特定语言。

import inspect  # 导入 inspect 模块，用于检查对象的类型和属性
from dataclasses import dataclass  # 从 dataclasses 导入 dataclass 装饰器，用于简化类的定义
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型注解，用于类型提示

import numpy as np  # 导入 NumPy 库，通常用于数值计算和数组操作
import PIL  # 导入 PIL 库，用于图像处理
import torch  # 导入 PyTorch 库，用于深度学习
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 从 transformers 导入 CLIP 相关类

from ...image_processor import PipelineImageInput  # 从本地模块导入 PipelineImageInput 类
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 从本地模块导入各种混合类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel  # 从本地模块导入模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 从本地模块导入调整 Lora 缩放的函数
from ...models.unets.unet_motion_model import MotionAdapter  # 从本地模块导入 MotionAdapter 类
from ...schedulers import (  # 从本地模块导入调度器类
    DDIMScheduler,  # 导入 DDIM 调度器
    DPMSolverMultistepScheduler,  # 导入 DPM 多步调度器
    EulerAncestralDiscreteScheduler,  # 导入 Euler 祖先离散调度器
    EulerDiscreteScheduler,  # 导入 Euler 离散调度器
    LMSDiscreteScheduler,  # 导入 LMS 离散调度器
    PNDMScheduler,  # 导入 PNDM 调度器
)
from ...utils import (  # 从本地模块导入各种工具函数和常量
    USE_PEFT_BACKEND,  # 导入标识是否使用 PEFT 后端的常量
    BaseOutput,  # 导入 BaseOutput 类，通常用于输出格式
    logging,  # 导入 logging 模块，用于记录日志
    replace_example_docstring,  # 导入替换示例文档字符串的函数
    scale_lora_layers,  # 导入缩放 Lora 层的函数
    unscale_lora_layers,  # 导入取消缩放 Lora 层的函数
)
from ...utils.torch_utils import randn_tensor  # 从本地模块导入生成随机张量的函数
from ...video_processor import VideoProcessor  # 从本地模块导入视频处理类
from ..free_init_utils import FreeInitMixin  # 从上级模块导入 FreeInitMixin 类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 从上级模块导入扩散管道和稳定扩散混合类


logger = logging.get_logger(__name__)  # 创建一个记录器实例，用于记录当前模块的日志，禁用 pylint 对名称的警告

EXAMPLE_DOC_STRING = """  # 定义一个示例文档字符串的常量
```  # 该常量的开始部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量的结束部分
```py  # 该常量的结束部分
```  # 该常量
    # 示例代码的使用说明
        Examples:
            # 导入所需的库
            ```py
            >>> import torch  # 导入 PyTorch 库
            >>> from diffusers import EulerDiscreteScheduler, MotionAdapter, PIAPipeline  # 从 diffusers 导入相关类
            >>> from diffusers.utils import export_to_gif, load_image  # 导入工具函数
    
            # 从预训练模型加载运动适配器
            >>> adapter = MotionAdapter.from_pretrained("openmmlab/PIA-condition-adapter")
            # 从预训练模型创建 PIAPipeline 对象，并指定适配器和数据类型
            >>> pipe = PIAPipeline.from_pretrained(
            ...     "SG161222/Realistic_Vision_V6.0_B1_noVAE", motion_adapter=adapter, torch_dtype=torch.float16
            ... )
    
            # 设置调度器为 EulerDiscreteScheduler，并使用现有配置
            >>> pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            # 从指定 URL 加载图像
            >>> image = load_image(
            ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png?download=true"
            ... )
            # 调整图像大小为 512x512 像素
            >>> image = image.resize((512, 512))
            # 定义正向提示内容
            >>> prompt = "cat in a hat"
            # 定义反向提示内容，以减少不良效果
            >>> negative_prompt = "wrong white balance, dark, sketches, worst quality, low quality, deformed, distorted"
            # 创建一个随机数生成器并设置种子
            >>> generator = torch.Generator("cpu").manual_seed(0)
            # 生成输出图像，通过管道处理输入图像和提示
            >>> output = pipe(image=image, prompt=prompt, negative_prompt=negative_prompt, generator=generator)
            # 获取输出结果中的第一帧
            >>> frames = output.frames[0]
            # 将帧导出为 GIF 动画文件
            >>> export_to_gif(frames, "pia-animation.gif")
"""
# 定义一个包含不同运动范围的列表，每个子列表代表不同类型的运动
RANGE_LIST = [
    [1.0, 0.9, 0.85, 0.85, 0.85, 0.8],  # 0 小运动
    [1.0, 0.8, 0.8, 0.8, 0.79, 0.78, 0.75],  # 中等运动
    [1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.5],  # 大运动
    [1.0, 0.9, 0.85, 0.85, 0.85, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.85, 0.85, 0.9, 1.0],  # 循环
    [1.0, 0.8, 0.8, 0.8, 0.79, 0.78, 0.75, 0.75, 0.75, 0.75, 0.75, 0.78, 0.79, 0.8, 0.8, 1.0],  # 循环
    [1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.8, 1.0],  # 循环
    [0.5, 0.4, 0.4, 0.4, 0.35, 0.3],  # 风格迁移候选小运动
    [0.5, 0.4, 0.4, 0.4, 0.35, 0.35, 0.3, 0.25, 0.2],  # 风格迁移中等运动
    [0.5, 0.2],  # 风格迁移大运动
]


# 定义一个函数，根据统计信息准备掩码系数
def prepare_mask_coef_by_statistics(num_frames: int, cond_frame: int, motion_scale: int):
    # 确保视频帧数大于 0
    assert num_frames > 0, "video_length should be greater than 0"

    # 确保视频帧数大于条件帧
    assert num_frames > cond_frame, "video_length should be greater than cond_frame"

    # 将 RANGE_LIST 赋值给 range_list
    range_list = RANGE_LIST

    # 确保运动缩放类型在范围列表中可用
    assert motion_scale < len(range_list), f"motion_scale type{motion_scale} not implemented"

    # 根据运动缩放类型获取对应的系数
    coef = range_list[motion_scale]
    # 用最后一个系数填充至 num_frames 长度
    coef = coef + ([coef[-1]] * (num_frames - len(coef)))

    # 计算每帧与条件帧的距离
    order = [abs(i - cond_frame) for i in range(num_frames)]
    # 根据距离重新排列系数
    coef = [coef[order[i]] for i in range(num_frames)]

    # 返回重新排列后的系数
    return coef


@dataclass
# 定义一个数据类，用于 PIAPipeline 的输出
class PIAPipelineOutput(BaseOutput):
    r"""
    PIAPipeline 的输出类。

    参数：
        frames (`torch.Tensor`, `np.ndarray`, 或 List[List[PIL.Image.Image]]):
            长度为 `batch_size` 的嵌套列表，包含每个 `num_frames` 的去噪 PIL 图像序列，形状为
            `(batch_size, num_frames, channels, height, width)` 的 NumPy 数组，或形状为 
            `(batch_size, num_frames, channels, height, width)` 的 Torch 张量。
    """

    # 输出帧，可以是多种数据类型
    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]


# 定义一个用于文本到视频生成的管道类
class PIAPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    FromSingleFileMixin,
    FreeInitMixin,
):
    r"""
    文本到视频生成的管道。

    此模型继承自 [`DiffusionPipeline`]. 请查看超类文档以了解所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    此管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器
    # 函数参数说明
    Args:
        vae ([`AutoencoderKL`]):
            # 变分自编码器 (VAE) 模型，用于将图像编码和解码为潜在表示
        text_encoder ([`CLIPTextModel`]):
            # 冻结的文本编码器，使用 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
        tokenizer (`CLIPTokenizer`):
            # [`~transformers.CLIPTokenizer`] 用于对文本进行标记化
        unet ([`UNet2DConditionModel`]):
            # [`UNet2DConditionModel`] 用于创建 UNetMotionModel，以去噪编码后的视频潜在特征
        motion_adapter ([`MotionAdapter`]):
            # [`MotionAdapter`] 与 `unet` 结合使用，以去噪编码后的视频潜在特征
        scheduler ([`SchedulerMixin`]):
            # 与 `unet` 结合使用的调度器，用于去噪编码后的图像潜在特征。可以是
            # [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]
    """

    # 定义模型 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 定义可选组件列表
    _optional_components = ["feature_extractor", "image_encoder", "motion_adapter"]
    # 定义回调张量输入列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        # 初始化方法的参数列表
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: Union[UNet2DConditionModel, UNetMotionModel],
        scheduler: Union[
            # 允许的调度器类型
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        motion_adapter: Optional[MotionAdapter] = None,
        feature_extractor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 如果 unet 是 UNet2DConditionModel 的实例，则转换为 UNetMotionModel
        if isinstance(unet, UNet2DConditionModel):
            unet = UNetMotionModel.from_unet2d(unet, motion_adapter)

        # 注册模块，初始化各个组件
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            motion_adapter=motion_adapter,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化视频处理器，设置是否调整大小和 VAE 缩放因子
        self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor=self.vae_scale_factor)

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 中复制，num_images_per_prompt -> num_videos_per_prompt
    # 定义一个编码提示的函数，接受多个参数
        def encode_prompt(
            self,  # 类的实例
            prompt,  # 输入的提示文本
            device,  # 计算设备（如 CPU 或 GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器引导
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入张量
            lora_scale: Optional[float] = None,  # 可选的 LoRA 缩放因子
            clip_skip: Optional[int] = None,  # 可选的剪辑跳过参数
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制的函数
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):  # 定义图像编码函数
            dtype = next(self.image_encoder.parameters()).dtype  # 获取图像编码器参数的数据类型
    
            if not isinstance(image, torch.Tensor):  # 如果输入的图像不是张量
                image = self.feature_extractor(image, return_tensors="pt").pixel_values  # 使用特征提取器将其转换为张量
    
            image = image.to(device=device, dtype=dtype)  # 将图像张量移动到指定设备并设置数据类型
            if output_hidden_states:  # 如果需要输出隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]  # 编码图像并获取倒数第二个隐藏状态
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)  # 根据生成图像数量重复隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(  # 编码全零图像以获取无条件隐藏状态
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]  # 获取无条件图像的倒数第二个隐藏状态
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(  # 重复无条件隐藏状态
                    num_images_per_prompt, dim=0
                )
                return image_enc_hidden_states, uncond_image_enc_hidden_states  # 返回编码的图像隐藏状态和无条件隐藏状态
            else:  # 如果不需要输出隐藏状态
                image_embeds = self.image_encoder(image).image_embeds  # 编码图像以获取图像嵌入
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)  # 根据生成图像数量重复图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)  # 创建与图像嵌入相同形状的全零无条件嵌入
    
                return image_embeds, uncond_image_embeds  # 返回图像嵌入和无条件嵌入
    
        # 从 diffusers.pipelines.text_to_video_synthesis/pipeline_text_to_video_synth.TextToVideoSDPipeline.decode_latents 复制的函数
        def decode_latents(self, latents):  # 定义解码潜变量的函数
            latents = 1 / self.vae.config.scaling_factor * latents  # 按缩放因子调整潜变量
    
            batch_size, channels, num_frames, height, width = latents.shape  # 获取潜变量的形状信息
            latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)  # 重新排列并重塑潜变量
    
            image = self.vae.decode(latents).sample  # 使用 VAE 解码潜变量以获取图像样本
            video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)  # 处理图像以生成视频格式
            # 我们总是将其转换为 float32，因为这不会造成显著开销，并且与 bfloat16 兼容
            video = video.float()  # 将视频数据转换为 float32 类型
            return video  # 返回解码后的视频数据
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的函数
    # 定义准备额外参数的方法，供调度器步骤使用
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，因为不同调度器的参数签名可能不同
        # eta（η）仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
        # 并且应该在 [0, 1] 之间

        # 检查调度器的步骤函数是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数字典
        extra_step_kwargs = {}
        # 如果接受 eta 参数，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的步骤函数是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator 参数，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤参数字典
        return extra_step_kwargs

    # 定义检查输入参数的方法
    def check_inputs(
        self,
        prompt,  # 输入提示
        height,  # 输出图像的高度
        width,   # 输出图像的宽度
        negative_prompt=None,  # 可选的负面提示
        prompt_embeds=None,    # 可选的提示嵌入
        negative_prompt_embeds=None,  # 可选的负面提示嵌入
        ip_adapter_image=None,  # 可选的图像适配器输入图像
        ip_adapter_image_embeds=None,  # 可选的图像适配器输入嵌入
        callback_on_step_end_tensor_inputs=None,  # 可选的步骤结束回调输入
    ):
        # 检查高度和宽度是否能被8整除
        if height % 8 != 0 or width % 8 != 0:
            # 抛出值错误，提示高度和宽度不符合要求
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调输入是否存在且不全在已注册的回调输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 抛出值错误，提示未找到有效的回调输入
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查是否同时传入了 prompt 和 prompt_embeds
        if prompt is not None and prompt_embeds is not None:
            # 抛出值错误，提示不能同时传入两者
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否同时未传入 prompt 和 prompt_embeds
        elif prompt is None and prompt_embeds is None:
            # 抛出值错误，提示至少需要提供一个
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 的类型
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 抛出值错误，提示 prompt 的类型不正确
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时传入了 negative_prompt 和 negative_prompt_embeds
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 抛出值错误，提示不能同时传入两者
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 是否形状一致
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 抛出值错误，提示两者形状不一致
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # 检查是否同时传入了 ip_adapter_image 和 ip_adapter_image_embeds
        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            # 抛出值错误，提示不能同时传入两者
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        # 检查 ip_adapter_image_embeds 是否存在
        if ip_adapter_image_embeds is not None:
            # 检查类型是否为列表
            if not isinstance(ip_adapter_image_embeds, list):
                # 抛出值错误，提示类型不正确
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            # 检查列表中的第一个元素的维度
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                # 抛出值错误，提示维度不符合要求
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的代码
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
        ):
            # 初始化图像嵌入列表
            image_embeds = []
            # 如果启用分类器自由引导，初始化负图像嵌入列表
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 如果没有提供图像嵌入
            if ip_adapter_image_embeds is None:
                # 确保输入的图像是列表格式
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
                # 检查图像数量是否与适配器数量匹配
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    raise ValueError(
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
                # 遍历每个图像和对应的投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 确定是否输出隐藏状态
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 编码单个图像，获取嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
                    # 将嵌入添加到列表中
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果启用分类器自由引导，添加负嵌入
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 如果提供了图像嵌入，遍历嵌入列表
                for single_image_embeds in ip_adapter_image_embeds:
                    # 处理负图像嵌入（如果启用引导）
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 将嵌入添加到列表中
                    image_embeds.append(single_image_embeds)
    
            # 初始化适配器图像嵌入列表
            ip_adapter_image_embeds = []
            # 遍历图像嵌入，为每个图像复制指定数量
            for i, single_image_embeds in enumerate(image_embeds):
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果启用分类器自由引导，处理负嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
    
                # 将嵌入移动到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 添加到适配器图像嵌入列表
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回适配器图像嵌入
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.prepare_latents 复制的代码
        def prepare_latents(
            self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        # 定义输出张量的形状，包括批量大小、通道数、帧数以及缩放后的高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        # 检查生成器是否为列表且其长度与批量大小不匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 抛出错误，提示生成器的长度与请求的有效批量大小不匹配
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果潜变量为 None，则生成随机张量作为潜变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 将给定的潜变量转移到指定设备
            latents = latents.to(device)

        # 将初始噪声按调度器要求的标准差进行缩放
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜变量
        return latents

    def prepare_masked_condition(
        self,
        image,
        batch_size,
        num_channels_latents,
        num_frames,
        height,
        width,
        dtype,
        device,
        generator,
        motion_scale=0,
    ):
        # 定义输出张量的形状，包括批量大小、通道数、帧数以及缩放后的高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        # 解包形状信息，获取缩放后的高度和宽度
        _, _, _, scaled_height, scaled_width = shape

        # 对输入图像进行预处理
        image = self.video_processor.preprocess(image)
        # 将预处理后的图像转移到指定设备并转换为指定数据类型
        image = image.to(device, dtype)

        # 如果生成器是列表，则逐个编码每个图像并采样
        if isinstance(generator, list):
            image_latent = [
                self.vae.encode(image[k : k + 1]).latent_dist.sample(generator[k]) for k in range(batch_size)
            ]
            # 将所有潜变量张量按维度 0 连接起来
            image_latent = torch.cat(image_latent, dim=0)
        else:
            # 否则直接编码图像并采样
            image_latent = self.vae.encode(image).latent_dist.sample(generator)

        # 将潜变量转移到指定设备并转换为指定数据类型
        image_latent = image_latent.to(device=device, dtype=dtype)
        # 对潜变量进行插值调整，改变大小到缩放后的高度和宽度
        image_latent = torch.nn.functional.interpolate(image_latent, size=[scaled_height, scaled_width])
        # 复制潜变量并按配置的缩放因子进行缩放
        image_latent_padding = image_latent.clone() * self.vae.config.scaling_factor

        # 创建一个全零的掩码张量，大小与批量大小、帧数、缩放后的高度和宽度匹配
        mask = torch.zeros((batch_size, 1, num_frames, scaled_height, scaled_width)).to(device=device, dtype=dtype)
        # 根据统计信息准备掩码系数
        mask_coef = prepare_mask_coef_by_statistics(num_frames, 0, motion_scale)
        # 创建一个全零的张量用于存储被掩盖的图像，形状与 image_latent_padding 匹配
        masked_image = torch.zeros(batch_size, 4, num_frames, scaled_height, scaled_width).to(
            device=device, dtype=self.unet.dtype
        )
        # 遍历每一帧，更新掩码和被掩盖的图像
        for f in range(num_frames):
            mask[:, :, f, :, :] = mask_coef[f]
            masked_image[:, :, f, :, :] = image_latent_padding.clone()

        # 根据条件决定是否复制掩码以支持分类器自由引导
        mask = torch.cat([mask] * 2) if self.do_classifier_free_guidance else mask
        # 根据条件决定是否复制被掩盖的图像以支持分类器自由引导
        masked_image = torch.cat([masked_image] * 2) if self.do_classifier_free_guidance else masked_image

        # 返回掩码和被掩盖的图像
        return mask, masked_image

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps 复制
    # 定义获取时间步长的方法，接受推理步数、强度和设备作为参数
    def get_timesteps(self, num_inference_steps, strength, device):
        # 计算初始时间步，确保不超过总推理步数
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
        # 计算开始时间步，确保不小于零
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从调度器中获取相应的时间步长
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        # 如果调度器具有设置开始索引的方法，则调用该方法
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
    
        # 返回计算得到的时间步和剩余推理步数
        return timesteps, num_inference_steps - t_start
    
        # 定义一个属性，用于获取引导比例
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 定义一个属性，用于获取剪切跳过参数
        @property
        def clip_skip(self):
            return self._clip_skip
    
        # 定义一个属性，判断是否使用无分类器引导
        @property
        def do_classifier_free_guidance(self):
            return self._guidance_scale > 1
    
        # 定义一个属性，用于获取交叉注意力参数
        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs
    
        # 定义一个属性，用于获取时间步数
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 禁用梯度计算，装饰器用于优化性能
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义调用方法，接受多个输入参数
        def __call__(
            self,
            image: PipelineImageInput,
            prompt: Union[str, List[str]] = None,
            strength: float = 1.0,
            num_frames: Optional[int] = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            motion_scale: int = 0,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
```