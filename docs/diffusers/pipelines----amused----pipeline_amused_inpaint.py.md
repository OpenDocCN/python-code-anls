# `.\diffusers\pipelines\amused\pipeline_amused_inpaint.py`

```py
# 版权所有 2024 HuggingFace 团队，保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件
# 根据许可证分发是在“按现状”基础上，
# 不提供任何形式的担保或条件，无论是明示或暗示。
# 请参阅许可证以获取有关特定语言的权限和
# 限制的详细信息。


# 从 typing 模块导入类型提示
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 transformers 模块导入 CLIP 模型和分词器
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

# 从当前模块的相对路径导入图像处理器和模型
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models import UVit2DModel, VQModel
from ...schedulers import AmusedScheduler
from ...utils import replace_example_docstring
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


# 示例文档字符串，提供用法示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AmusedInpaintPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = AmusedInpaintPipeline.from_pretrained(
        ...     "amused/amused-512", variant="fp16", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "fall mountains"
        >>> input_image = (
        ...     load_image(
        ...         "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1.jpg"
        ...     )
        ...     .resize((512, 512))
        ...     .convert("RGB")
        ... )
        >>> mask = (
        ...     load_image(
        ...         "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1_mask.png"
        ...     )
        ...     .resize((512, 512))
        ...     .convert("L")
        ... )
        >>> pipe(prompt, input_image, mask).images[0].save("out.png")
        ```py
"""


# 定义 AmusedInpaintPipeline 类，继承自 DiffusionPipeline
class AmusedInpaintPipeline(DiffusionPipeline):
    # 定义类属性，表示图像处理器的类型
    image_processor: VaeImageProcessor
    # 定义类属性，表示 VQ 模型的类型
    vqvae: VQModel
    # 定义类属性，表示分词器的类型
    tokenizer: CLIPTokenizer
    # 定义类属性，表示文本编码器的类型
    text_encoder: CLIPTextModelWithProjection
    # 定义类属性，表示变换器的类型
    transformer: UVit2DModel
    # 定义类属性，表示调度器的类型
    scheduler: AmusedScheduler

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->transformer->vqvae"

    # TODO - 解决在调用 self.vqvae.quantize 时，未能调用钩子来移动参数的问题
    _exclude_from_cpu_offload = ["vqvae"]

    # 初始化方法，接受多个参数以构建管道
    def __init__(
        self,
        vqvae: VQModel,  # VQ模型
        tokenizer: CLIPTokenizer,  # CLIP分词器
        text_encoder: CLIPTextModelWithProjection,  # 文本编码器
        transformer: UVit2DModel,  # 变换器模型
        scheduler: AmusedScheduler,  # 调度器
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册模块，包括 VQ-VAE、tokenizer、文本编码器、转换器和调度器
        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
        # 计算 VAE 的缩放因子，根据 VQ-VAE 的块输出通道数
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        # 创建图像处理器，配置 VAE 缩放因子，并设置不进行归一化
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)
        # 创建掩码处理器，配置 VAE 缩放因子，设置多项处理选项
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,  # 开启二值化处理
            do_convert_grayscale=True,  # 开启灰度转换
            do_resize=True,  # 开启缩放处理
        )
        # 将掩码调度注册到配置中，使用线性调度
        self.scheduler.register_to_config(masking_schedule="linear")

    @torch.no_grad()  # 禁用梯度计算，提高推理效率
    @replace_example_docstring(EXAMPLE_DOC_STRING)  # 用示例文档字符串替换默认文档
    def __call__(
        self,
        # 提示信息，可以是字符串或字符串列表
        prompt: Optional[Union[List[str], str]] = None,
        # 输入图像，类型为 PipelineImageInput
        image: PipelineImageInput = None,
        # 掩码图像，类型为 PipelineImageInput
        mask_image: PipelineImageInput = None,
        # 强度参数，影响生成图像的效果
        strength: float = 1.0,
        # 推理步骤数，影响生成质量
        num_inference_steps: int = 12,
        # 指导比例，影响生成图像的多样性
        guidance_scale: float = 10.0,
        # 负面提示信息，可以是字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量
        num_images_per_prompt: Optional[int] = 1,
        # 随机数生成器，控制随机性
        generator: Optional[torch.Generator] = None,
        # 提示的嵌入表示，类型为 torch.Tensor
        prompt_embeds: Optional[torch.Tensor] = None,
        # 编码器隐藏状态，类型为 torch.Tensor
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 负面提示的嵌入表示，类型为 torch.Tensor
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 负面编码器隐藏状态，类型为 torch.Tensor
        negative_encoder_hidden_states: Optional[torch.Tensor] = None,
        # 输出类型，默认为 "pil"
        output_type="pil",
        # 是否返回字典格式的结果
        return_dict: bool = True,
        # 回调函数，处理推理过程中的信息
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调频率，控制回调调用的步骤
        callback_steps: int = 1,
        # 跨注意力参数，调整模型的注意力机制
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 微调条件美学评分，默认值为 6
        micro_conditioning_aesthetic_score: int = 6,
        # 微调裁剪坐标，默认值为 (0, 0)
        micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
        # 温度参数，影响生成的随机性
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
```