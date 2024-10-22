# `.\diffusers\pipelines\controlnet\pipeline_controlnet_blip_diffusion.py`

```py
# 版权所有 2024 Salesforce.com, inc.
# 版权所有 2024 The HuggingFace Team. 保留所有权利。
#
# 根据 Apache 许可证，第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，使用许可证下分发的软件是基于“原样”提供的，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参阅许可证以获取有关特定权限和
# 限制的具体说明。
from typing import List, Optional, Union  # 导入类型注解 List、Optional 和 Union，以用于类型提示

import PIL.Image  # 导入 PIL.Image 模块，以便进行图像处理
import torch  # 导入 PyTorch 库，用于深度学习操作
from transformers import CLIPTokenizer  # 从 transformers 库导入 CLIPTokenizer，用于处理文本输入

from ...models import AutoencoderKL, ControlNetModel, UNet2DConditionModel  # 从上级目录导入模型类
from ...schedulers import PNDMScheduler  # 从上级目录导入调度器类
from ...utils import (  # 从上级目录导入工具函数
    logging,  # 导入日志记录工具
    replace_example_docstring,  # 导入函数用于替换示例文档字符串
)
from ...utils.torch_utils import randn_tensor  # 从上级目录导入生成随机张量的工具函数
from ..blip_diffusion.blip_image_processing import BlipImageProcessor  # 从下级目录导入图像处理类
from ..blip_diffusion.modeling_blip2 import Blip2QFormerModel  # 从下级目录导入 BLIP2 模型类
from ..blip_diffusion.modeling_ctx_clip import ContextCLIPTextModel  # 从下级目录导入上下文 CLIP 文本模型
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从下级目录导入扩散管道和图像输出类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例，方便日志输出
EXAMPLE_DOC_STRING = """  # 定义示例文档字符串的多行字符串，可能用于说明或文档
```  
    # 示例用法说明
        Examples:
            ```py
            # 从 diffusers 库导入 BlipDiffusionControlNetPipeline 类
            >>> from diffusers.pipelines import BlipDiffusionControlNetPipeline
            # 从 diffusers.utils 导入 load_image 函数
            >>> from diffusers.utils import load_image
            # 从 controlnet_aux 导入 CannyDetector 类
            >>> from controlnet_aux import CannyDetector
            # 导入 PyTorch 库
            >>> import torch
    
            # 从预训练模型加载 BlipDiffusionControlNetPipeline 实例，并设置数据类型为 float16
            >>> blip_diffusion_pipe = BlipDiffusionControlNetPipeline.from_pretrained(
            ...     "Salesforce/blipdiffusion-controlnet", torch_dtype=torch.float16
            ... ).to("cuda")  # 将模型移动到 GPU
    
            # 定义风格主体为 "flower"
            >>> style_subject = "flower"
            # 定义目标主体为 "teapot"
            >>> tgt_subject = "teapot"
            # 定义文本提示为 "on a marble table"
            >>> text_prompt = "on a marble table"
    
            # 加载并调整目标条件图像的大小
            >>> cldm_cond_image = load_image(
            ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/kettle.jpg"
            ... ).resize((512, 512))  # 调整图像大小到 512x512 像素
            # 创建 Canny 边缘检测器实例
            >>> canny = CannyDetector()
            # 应用 Canny 边缘检测，输出类型为 PIL 图像
            >>> cldm_cond_image = canny(cldm_cond_image, 30, 70, output_type="pil")
            # 加载风格图像
            >>> style_image = load_image(
            ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/flower.jpg"
            ... )
            # 定义引导尺度
            >>> guidance_scale = 7.5
            # 定义推理步骤数量
            >>> num_inference_steps = 50
            # 定义负提示
            >>> negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
    
            # 运行 BlipDiffusion 控制管道，生成图像输出
            >>> output = blip_diffusion_pipe(
            ...     text_prompt,
            ...     style_image,
            ...     cldm_cond_image,
            ...     style_subject,
            ...     tgt_subject,
            ...     guidance_scale=guidance_scale,
            ...     num_inference_steps=num_inference_steps,
            ...     neg_prompt=negative_prompt,
            ...     height=512,
            ...     width=512,
            ... ).images  # 生成的图像存储在 output 中
            # 保存生成的第一张图像为 "image.png"
            >>> output[0].save("image.png")
            ``` 
# 定义 BlipDiffusionControlNetPipeline 类，继承自 DiffusionPipeline
class BlipDiffusionControlNetPipeline(DiffusionPipeline):
    """
    基于 Canny 边缘的受控主体驱动生成的 Blip Diffusion 管道。

    该模型继承自 [`DiffusionPipeline`]。有关库为所有管道实现的通用方法（如下载或保存、在特定设备上运行等），请查看超类文档。

    参数：
        tokenizer ([`CLIPTokenizer`]):
            文本编码器的分词器
        text_encoder ([`ContextCLIPTextModel`]):
            用于编码文本提示的文本编码器
        vae ([`AutoencoderKL`]):
            将潜变量映射到图像的 VAE 模型
        unet ([`UNet2DConditionModel`]):
            用于去噪图像嵌入的条件 U-Net 架构。
        scheduler ([`PNDMScheduler`]):
             用于与 `unet` 结合生成图像潜变量的调度器。
        qformer ([`Blip2QFormerModel`]):
            从文本和图像获取多模态嵌入的 QFormer 模型。
        controlnet ([`ControlNetModel`]):
            用于获取条件图像嵌入的 ControlNet 模型。
        image_processor ([`BlipImageProcessor`]):
            用于预处理和后处理图像的图像处理器。
        ctx_begin_pos (int, `optional`, defaults to 2):
            文本编码器中上下文令牌的位置。
    """

    # 定义 CPU 离线加载序列
    model_cpu_offload_seq = "qformer->text_encoder->unet->vae"

    # 初始化方法，接受多个模型和参数
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: ContextCLIPTextModel,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        qformer: Blip2QFormerModel,
        controlnet: ControlNetModel,
        image_processor: BlipImageProcessor,
        ctx_begin_pos: int = 2,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        # 调用父类构造函数
        super().__init__()

        # 注册模型模块
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            qformer=qformer,
            controlnet=controlnet,
            image_processor=image_processor,
        )
        # 注册配置参数
        self.register_to_config(ctx_begin_pos=ctx_begin_pos, mean=mean, std=std)

    # 获取查询嵌入的函数，接受输入图像和源主题
    def get_query_embeddings(self, input_image, src_subject):
        # 调用 QFormer 模型获取嵌入
        return self.qformer(image_input=input_image, text_input=src_subject, return_dict=False)

    # 从原始 Blip Diffusion 代码，指定目标主题并通过重复提示增强提示
    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0, prompt_reps=20):
        rv = []  # 初始化返回值列表
        # 遍历每个提示和目标主题
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            # 构建增强的提示
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # 一种放大提示的技巧
            rv.append(", ".join([prompt] * int(prompt_strength * prompt_reps)))

        # 返回增强后的提示列表
        return rv
    # 从 diffusers.pipelines.consistency_models.pipeline_consistency_models.ConsistencyModelPipeline.prepare_latents 复制
    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        # 定义潜在张量的形状，基于输入的批量大小和通道数
        shape = (batch_size, num_channels, height, width)
        # 检查生成器列表的长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
    
        # 如果没有提供潜在张量，则生成随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在张量，则将其转移到指定设备和数据类型
            latents = latents.to(device=device, dtype=dtype)
    
        # 按调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在张量
        return latents
    
    def encode_prompt(self, query_embeds, prompt, device=None):
        # 设置设备为给定的设备或默认执行设备
        device = device or self._execution_device
    
        # 获取与查询嵌入相关的提示的最大长度
        max_len = self.text_encoder.text_model.config.max_position_embeddings
        max_len -= self.qformer.config.num_query_tokens
    
        # 对提示进行标记化，并转换为张量
        tokenized_prompt = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
    
        # 获取批量大小和上下文开始位置
        batch_size = query_embeds.shape[0]
        ctx_begin_pos = [self.config.ctx_begin_pos] * batch_size
    
        # 将标记化的提示输入到文本编码器中，得到文本嵌入
        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=ctx_begin_pos,
        )[0]
    
        # 返回文本嵌入
        return text_embeddings
    
    # 从 diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image 适配
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
    ):
        # 对输入图像进行预处理，调整大小并转换为张量
        image = self.image_processor.preprocess(
            image,
            size={"width": width, "height": height},
            do_rescale=True,
            do_center_crop=False,
            do_normalize=False,
            return_tensors="pt",
        )["pixel_values"].to(device)
        # 获取图像的批量大小
        image_batch_size = image.shape[0]
    
        # 根据图像批量大小设置重复次数
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # 图像批量大小与提示批量大小相同
            repeat_by = num_images_per_prompt
    
        # 通过重复图像来扩展图像张量
        image = image.repeat_interleave(repeat_by, dim=0)
    
        # 将图像张量转移到指定设备和数据类型
        image = image.to(device=device, dtype=dtype)
    
        # 如果启用无分类器自由引导，则重复图像
        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)
    
        # 返回处理后的图像
        return image
    
    # 不跟踪梯度
    @torch.no_grad()
    # 使用装饰器替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义可调用的方法，接收多个参数用于处理图像生成
        def __call__(
            # 输入的提示文本列表
            self,
            prompt: List[str],
            # 参考图像，类型为 PIL.Image.Image
            reference_image: PIL.Image.Image,
            # 条件图像，类型为 PIL.Image.Image
            condtioning_image: PIL.Image.Image,
            # 源主题类别列表
            source_subject_category: List[str],
            # 目标主题类别列表
            target_subject_category: List[str],
            # 可选的潜在变量，类型为 torch.Tensor
            latents: Optional[torch.Tensor] = None,
            # 指导尺度，默认值为 7.5
            guidance_scale: float = 7.5,
            # 生成图像的高度，默认值为 512
            height: int = 512,
            # 生成图像的宽度，默认值为 512
            width: int = 512,
            # 推理步骤的数量，默认值为 50
            num_inference_steps: int = 50,
            # 可选的随机数生成器，类型为 torch.Generator 或其列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的负提示文本，默认值为空字符串
            neg_prompt: Optional[str] = "",
            # 提示强度，默认值为 1.0
            prompt_strength: float = 1.0,
            # 提示重复次数，默认值为 20
            prompt_reps: int = 20,
            # 输出类型，默认值为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认值为 True
            return_dict: bool = True,
```