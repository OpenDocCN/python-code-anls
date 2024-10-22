# `.\diffusers\pipelines\unclip\pipeline_unclip_image_variation.py`

```py
# 版权所有 2024 Kakao Brain 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守该许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件在许可证下分发时是按“原样”提供的，
# 不附有任何明示或暗示的担保或条件。
# 有关许可证下特定权利和限制，请参阅许可证。

import inspect  # 导入 inspect 模块以获取有关活跃对象的信息
from typing import List, Optional, Union  # 从 typing 模块导入类型提示工具

import PIL.Image  # 导入 PIL.Image 以处理图像文件
import torch  # 导入 PyTorch 库以进行张量运算
from torch.nn import functional as F  # 导入 PyTorch 的功能性 API 以进行各种神经网络操作
from transformers import (  # 从 transformers 库导入必要的模型和处理器
    CLIPImageProcessor,  # 导入 CLIP 图像处理器
    CLIPTextModelWithProjection,  # 导入 CLIP 文本模型，带有投影
    CLIPTokenizer,  # 导入 CLIP 分词器
    CLIPVisionModelWithProjection,  # 导入 CLIP 视觉模型，带有投影
)

from ...models import UNet2DConditionModel, UNet2DModel  # 从相对路径导入 UNet 模型
from ...schedulers import UnCLIPScheduler  # 从相对路径导入 UnCLIP 调度器
from ...utils import logging  # 从相对路径导入 logging 工具
from ...utils.torch_utils import randn_tensor  # 从相对路径导入随机张量生成工具
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从相对路径导入 DiffusionPipeline 和 ImagePipelineOutput
from .text_proj import UnCLIPTextProjModel  # 从当前目录导入 UnCLIP 文本投影模型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例； pylint 禁用无效名称警告


class UnCLIPImageVariationPipeline(DiffusionPipeline):  # 定义 UnCLIP 图像变体生成管道类，继承自 DiffusionPipeline
    """
    使用 UnCLIP 从输入图像生成图像变体的管道。

    该模型继承自 [`DiffusionPipeline`]。有关所有管道通用方法的文档（下载、保存、在特定设备上运行等），请查看超类文档。
    # 参数说明
    Args:
        text_encoder ([`~transformers.CLIPTextModelWithProjection`]):
            # 冻结的文本编码器
            Frozen text-encoder.
        tokenizer ([`~transformers.CLIPTokenizer`]):
            # 用于对文本进行分词的 CLIPTokenizer
            A `CLIPTokenizer` to tokenize text.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            # 从生成的图像中提取特征以作为图像编码器的输入的模型
            Model that extracts features from generated images to be used as inputs for the `image_encoder`.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            # 冻结的 CLIP 图像编码器，使用 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
            Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        text_proj ([`UnCLIPTextProjModel`]):
            # 准备和组合嵌入的工具类，嵌入将传递给解码器
            Utility class to prepare and combine the embeddings before they are passed to the decoder.
        decoder ([`UNet2DConditionModel`]):
            # 将图像嵌入反转为图像的解码器
            The decoder to invert the image embedding into an image.
        super_res_first ([`UNet2DModel`]):
            # 超分辨率 UNet，用于超分辨率扩散过程的所有步骤，除了最后一步
            Super resolution UNet. Used in all but the last step of the super resolution diffusion process.
        super_res_last ([`UNet2DModel`]):
            # 超分辨率 UNet，用于超分辨率扩散过程的最后一步
            Super resolution UNet. Used in the last step of the super resolution diffusion process.
        decoder_scheduler ([`UnCLIPScheduler`]):
            # 在解码器去噪过程中使用的调度器（修改后的 [`DDPMScheduler`])
            Scheduler used in the decoder denoising process (a modified [`DDPMScheduler`]).
        super_res_scheduler ([`UnCLIPScheduler`]):
            # 在超分辨率去噪过程中使用的调度器（修改后的 [`DDPMScheduler`])
            Scheduler used in the super resolution denoising process (a modified [`DDPMScheduler`]).
    """

    # 定义解码器的类型
    decoder: UNet2DConditionModel
    # 定义文本嵌入的处理模型
    text_proj: UnCLIPTextProjModel
    # 定义文本编码器的类型
    text_encoder: CLIPTextModelWithProjection
    # 定义分词器的类型
    tokenizer: CLIPTokenizer
    # 定义特征提取器的类型
    feature_extractor: CLIPImageProcessor
    # 定义图像编码器的类型
    image_encoder: CLIPVisionModelWithProjection
    # 定义超分辨率模型（第一步）
    super_res_first: UNet2DModel
    # 定义超分辨率模型（最后一步）
    super_res_last: UNet2DModel

    # 定义解码器和超分辨率过程中的调度器
    decoder_scheduler: UnCLIPScheduler
    super_res_scheduler: UnCLIPScheduler
    # 定义模型的 CPU 卸载顺序，表示组件之间的执行顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->text_proj->decoder->super_res_first->super_res_last"

    # 初始化方法
    def __init__(
        # 定义解码器的输入参数
        self,
        decoder: UNet2DConditionModel,
        # 定义文本编码器的输入参数
        text_encoder: CLIPTextModelWithProjection,
        # 定义分词器的输入参数
        tokenizer: CLIPTokenizer,
        # 定义文本嵌入处理的输入参数
        text_proj: UnCLIPTextProjModel,
        # 定义特征提取器的输入参数
        feature_extractor: CLIPImageProcessor,
        # 定义图像编码器的输入参数
        image_encoder: CLIPVisionModelWithProjection,
        # 定义超分辨率模型（第一步）的输入参数
        super_res_first: UNet2DModel,
        # 定义超分辨率模型（最后一步）的输入参数
        super_res_last: UNet2DModel,
        # 定义解码器调度器的输入参数
        decoder_scheduler: UnCLIPScheduler,
        # 定义超分辨率调度器的输入参数
        super_res_scheduler: UnCLIPScheduler,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册各个模块，便于管理和调用
        self.register_modules(
            decoder=decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_proj=text_proj,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            super_res_first=super_res_first,
            super_res_last=super_res_last,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )

    # 从 diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents 复制而来
    # 准备潜在变量，生成或处理给定形状的潜在张量
        def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
            # 如果潜在张量为 None，则随机生成一个具有指定形状的张量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果给定的潜在张量形状与期望的形状不匹配，则引发错误
                if latents.shape != shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
                # 将潜在张量移动到指定设备
                latents = latents.to(device)
    
            # 将潜在张量乘以调度器的初始噪声标准差
            latents = latents * scheduler.init_noise_sigma
            # 返回处理后的潜在张量
            return latents
    
        # 编码图像以生成图像嵌入
        def _encode_image(self, image, device, num_images_per_prompt, image_embeddings: Optional[torch.Tensor] = None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果没有提供图像嵌入，则进行图像处理
            if image_embeddings is None:
                # 如果输入的图像不是张量，则使用特征提取器处理图像
                if not isinstance(image, torch.Tensor):
                    image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
    
                # 将图像张量移动到指定设备，并转换为正确的数据类型
                image = image.to(device=device, dtype=dtype)
                # 通过图像编码器生成图像嵌入
                image_embeddings = self.image_encoder(image).image_embeds
    
            # 将图像嵌入按指定的提示数量进行重复
            image_embeddings = image_embeddings.repeat_interleave(num_images_per_prompt, dim=0)
    
            # 返回生成的图像嵌入
            return image_embeddings
    
        # 定义调用方法，禁用梯度计算以节省内存
        @torch.no_grad()
        def __call__(
            # 接收可选的输入图像，可以是单个图像、图像列表或张量
            image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor]] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: int = 1,
            # 解码器推理步骤的数量
            decoder_num_inference_steps: int = 25,
            # 超分辨率推理步骤的数量
            super_res_num_inference_steps: int = 7,
            # 可选的随机数生成器
            generator: Optional[torch.Generator] = None,
            # 可选的解码器潜在张量
            decoder_latents: Optional[torch.Tensor] = None,
            # 可选的超分辨率潜在张量
            super_res_latents: Optional[torch.Tensor] = None,
            # 可选的图像嵌入
            image_embeddings: Optional[torch.Tensor] = None,
            # 解码器引导尺度
            decoder_guidance_scale: float = 8.0,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果
            return_dict: bool = True,
```