# `.\diffusers\pipelines\kandinsky2_2\pipeline_kandinsky2_2_prior.py`

```py
# 从 typing 模块导入必要的类型提示
from typing import Callable, Dict, List, Optional, Union

# 导入 PIL 库中的 Image 模块
import PIL.Image
# 导入 torch 库
import torch
# 从 transformers 库导入相关的模型和处理器
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

# 从本地模型文件中导入 PriorTransformer 类
from ...models import PriorTransformer
# 从调度器模块中导入 UnCLIPScheduler 类
from ...schedulers import UnCLIPScheduler
# 从工具模块中导入 logging 和 replace_example_docstring 函数
from ...utils import (
    logging,
    replace_example_docstring,
)
# 从 torch_utils 模块中导入 randn_tensor 函数
from ...utils.torch_utils import randn_tensor
# 从 kandinsky 模块中导入 KandinskyPriorPipelineOutput 类
from ..kandinsky import KandinskyPriorPipelineOutput
# 从 pipeline_utils 模块中导入 DiffusionPipeline 类
from ..pipeline_utils import DiffusionPipeline


# 创建一个 logger 实例用于记录日志，禁用 pylint 的命名检查
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，包含使用示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
        >>> import torch

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
        >>> pipe_prior.to("cuda")
        >>> prompt = "red cat, 4k photo"
        >>> image_emb, negative_image_emb = pipe_prior(prompt).to_tuple()

        >>> pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
        >>> pipe.to("cuda")
        >>> image = pipe(
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=negative_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=50,
        ... ).images
        >>> image[0].save("cat.png")
        ```py
"""

# 示例文档字符串，包含插值的使用示例
EXAMPLE_INTERPOLATE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
        >>> from diffusers.utils import load_image
        >>> import PIL
        >>> import torch
        >>> from torchvision import transforms

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ... )
        >>> pipe_prior.to("cuda")
        >>> img1 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )
        >>> img2 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/starry_night.jpeg"
        ... )
        >>> images_texts = ["a cat", img1, img2]
        >>> weights = [0.3, 0.3, 0.4]
        >>> out = pipe_prior.interpolate(images_texts, weights)
        >>> pipe = KandinskyV22Pipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> image = pipe(
        ...     image_embeds=out.image_embeds,
        ...     negative_image_embeds=out.negative_image_embeds,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("starry_cat.png")
        ```py
"""


# 定义 KandinskyV22PriorPipeline 类，继承自 DiffusionPipeline
class KandinskyV22PriorPipeline(DiffusionPipeline):
    """
    # 该文档字符串描述了生成 Kandinsky 图像先验的管道

    # 该模型继承自 [`DiffusionPipeline`]。可以查看超类文档以了解库为所有管道实现的通用方法
    # （例如下载、保存、在特定设备上运行等）。

    # 参数：
        # prior ([`PriorTransformer`]):
            # 用于从文本嵌入近似图像嵌入的标准 unCLIP 先验。
        # image_encoder ([`CLIPVisionModelWithProjection`]):
            # 冻结的图像编码器。
        # text_encoder ([`CLIPTextModelWithProjection`]):
            # 冻结的文本编码器。
        # tokenizer (`CLIPTokenizer`):
            # 类的标记器
            # [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)。
        # scheduler ([`UnCLIPScheduler`]):
            # 用于与 `prior` 结合生成图像嵌入的调度器。
        # image_processor ([`CLIPImageProcessor`]):
            # 用于从 CLIP 预处理图像的图像处理器。
    """

    # 定义 CPU 卸载序列，包括文本编码器、图像编码器和先验
    model_cpu_offload_seq = "text_encoder->image_encoder->prior"
    # 定义在 CPU 卸载时排除的模块，先验不参与卸载
    _exclude_from_cpu_offload = ["prior"]
    # 定义需要作为回调的张量输入
    _callback_tensor_inputs = ["latents", "prompt_embeds", "text_encoder_hidden_states", "text_mask"]

    def __init__(
        # 初始化方法，接受多个参数用于模型的构建
        self,
        prior: PriorTransformer,  # 标准 unCLIP 先验
        image_encoder: CLIPVisionModelWithProjection,  # 冻结的图像编码器
        text_encoder: CLIPTextModelWithProjection,  # 冻结的文本编码器
        tokenizer: CLIPTokenizer,  # 标记器
        scheduler: UnCLIPScheduler,  # 调度器
        image_processor: CLIPImageProcessor,  # 图像处理器
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册模型所需的模块
        self.register_modules(
            prior=prior,  # 注册先验
            text_encoder=text_encoder,  # 注册文本编码器
            tokenizer=tokenizer,  # 注册标记器
            scheduler=scheduler,  # 注册调度器
            image_encoder=image_encoder,  # 注册图像编码器
            image_processor=image_processor,  # 注册图像处理器
        )

    @torch.no_grad()  # 在此装饰器下，不计算梯度以节省内存和加快计算
    @replace_example_docstring(EXAMPLE_INTERPOLATE_DOC_STRING)  # 替换示例文档字符串为预定义字符串
    def interpolate(
        # 定义插值方法，接受多个参数以处理图像和提示
        self,
        images_and_prompts: List[Union[str, PIL.Image.Image, torch.Tensor]],  # 输入的图像和提示列表
        weights: List[float],  # 与每个图像和提示对应的权重
        num_images_per_prompt: int = 1,  # 每个提示生成的图像数量，默认为1
        num_inference_steps: int = 25,  # 推理步骤的数量，默认为25
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 可选的随机数生成器
        latents: Optional[torch.Tensor] = None,  # 可选的潜在变量张量
        negative_prior_prompt: Optional[str] = None,  # 可选的负先验提示
        negative_prompt: str = "",  # 负提示，默认为空字符串
        guidance_scale: float = 4.0,  # 指导比例，默认为4.0
        device=None,  # 可选的设备参数
    # 从 diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents 复制的部分
    # 准备潜在变量，生成或处理输入的潜在张量
        def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
            # 如果没有提供潜在张量，则随机生成一个
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果提供的潜在张量形状不匹配，则抛出异常
                if latents.shape != shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
                # 将潜在张量移动到指定设备
                latents = latents.to(device)
    
            # 将潜在张量乘以调度器的初始化噪声标准差
            latents = latents * scheduler.init_noise_sigma
            # 返回处理后的潜在张量
            return latents
    
        # 从 KandinskyPriorPipeline 获取零嵌入的复制
        def get_zero_embed(self, batch_size=1, device=None):
            # 如果没有指定设备，则使用默认设备
            device = device or self.device
            # 创建一个零图像张量，大小为图像编码器配置的图像大小
            zero_img = torch.zeros(1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size).to(
                device=device, dtype=self.image_encoder.dtype
            )
            # 将零图像传递给图像编码器以获取图像嵌入
            zero_image_emb = self.image_encoder(zero_img)["image_embeds"]
            # 将嵌入复制以匹配批量大小
            zero_image_emb = zero_image_emb.repeat(batch_size, 1)
            # 返回零图像嵌入
            return zero_image_emb
    
        # 从 KandinskyPriorPipeline 复制的提示编码方法
        def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
        # 分类器自由引导的属性，检查引导比例是否大于1
        @property
        def do_classifier_free_guidance(self):
            return self._guidance_scale > 1
    
        # 引导比例的属性，返回当前引导比例
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 时间步数的属性，返回当前时间步数
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 禁用梯度计算并替换示例文档字符串
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: int = 1,
            num_inference_steps: int = 25,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            guidance_scale: float = 4.0,
            output_type: Optional[str] = "pt",  # 仅支持 pt 格式
            return_dict: bool = True,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 指定在步骤结束时的张量输入列表
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
```