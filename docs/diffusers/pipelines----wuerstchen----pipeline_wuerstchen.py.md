# `.\diffusers\pipelines\wuerstchen\pipeline_wuerstchen.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，按许可证分发的软件
# 是在“按原样”基础上提供的，没有任何形式的保证或条件，
# 明示或暗示。有关许可的特定权限和
# 限制，请参阅许可证。

from typing import Callable, Dict, List, Optional, Union  # 从 typing 模块导入类型注解工具

import numpy as np  # 导入 NumPy 库，常用于数值计算
import torch  # 导入 PyTorch 库，支持深度学习
from transformers import CLIPTextModel, CLIPTokenizer  # 从 transformers 库导入 CLIP 模型和分词器

from ...schedulers import DDPMWuerstchenScheduler  # 从调度器模块导入 DDPMWuerstchenScheduler
from ...utils import deprecate, logging, replace_example_docstring  # 从 utils 模块导入实用工具
from ...utils.torch_utils import randn_tensor  # 从 PyTorch 工具模块导入 randn_tensor 函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从管道工具模块导入 DiffusionPipeline 和 ImagePipelineOutput
from .modeling_paella_vq_model import PaellaVQModel  # 从 Paella VQ 模型模块导入 PaellaVQModel
from .modeling_wuerstchen_diffnext import WuerstchenDiffNeXt  # 从 Wuerstchen DiffNeXt 模型模块导入 WuerstchenDiffNeXt

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，禁用 pylint 对无效名称的警告

EXAMPLE_DOC_STRING = """  # 示例文档字符串，提供用法示例
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import WuerstchenPriorPipeline, WuerstchenDecoderPipeline  # 导入 Wuerstchen 管道

        >>> prior_pipe = WuerstchenPriorPipeline.from_pretrained(  # 从预训练模型创建 WuerstchenPriorPipeline 实例
        ...     "warp-ai/wuerstchen-prior", torch_dtype=torch.float16  # 指定模型名称和数据类型
        ... ).to("cuda")  # 将管道移动到 CUDA 设备
        >>> gen_pipe = WuerstchenDecoderPipeline.from_pretrain("warp-ai/wuerstchen", torch_dtype=torch.float16).to(  # 从预训练模型创建 WuerstchenDecoderPipeline 实例
        ...     "cuda"  # 将生成管道移动到 CUDA 设备
        ... )

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"  # 定义生成图像的提示
        >>> prior_output = pipe(prompt)  # 使用提示生成先前输出
        >>> images = gen_pipe(prior_output.image_embeddings, prompt=prompt)  # 使用生成管道从图像嵌入生成图像
        ```py
"""

class WuerstchenDecoderPipeline(DiffusionPipeline):  # 定义 WuerstchenDecoderPipeline 类，继承自 DiffusionPipeline
    """
    Pipeline for generating images from the Wuerstchen model.  # 类文档字符串，说明该管道的功能

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)  # 说明该模型继承自 DiffusionPipeline，并提醒用户查看父类文档以获取通用方法
    # 参数说明
    Args:
        tokenizer (`CLIPTokenizer`):  # CLIP 模型使用的分词器
            The CLIP tokenizer.
        text_encoder (`CLIPTextModel`):  # CLIP 模型使用的文本编码器
            The CLIP text encoder.
        decoder ([`WuerstchenDiffNeXt`]):  # WuerstchenDiffNeXt 解码器
            The WuerstchenDiffNeXt unet decoder.
        vqgan ([`PaellaVQModel`]):  # VQGAN 模型，用于图像生成
            The VQGAN model.
        scheduler ([`DDPMWuerstchenScheduler`]):  # 调度器，用于图像嵌入生成
            A scheduler to be used in combination with `prior` to generate image embedding.
        latent_dim_scale (float, `optional`, defaults to 10.67):  # 用于确定 VQ 潜在空间大小的乘数
            Multiplier to determine the VQ latent space size from the image embeddings. If the image embeddings are
            height=24 and width=24, the VQ latent shape needs to be height=int(24*10.67)=256 and
            width=int(24*10.67)=256 in order to match the training conditions.
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->decoder->vqgan"
    # 定义需要回调的张量输入列表
    _callback_tensor_inputs = [
        "latents",  # 潜在变量
        "text_encoder_hidden_states",  # 文本编码器的隐藏状态
        "negative_prompt_embeds",  # 负面提示的嵌入
        "image_embeddings",  # 图像嵌入
    ]

    # 构造函数
    def __init__(
        self,
        tokenizer: CLIPTokenizer,  # 初始化时传入的分词器
        text_encoder: CLIPTextModel,  # 初始化时传入的文本编码器
        decoder: WuerstchenDiffNeXt,  # 初始化时传入的解码器
        scheduler: DDPMWuerstchenScheduler,  # 初始化时传入的调度器
        vqgan: PaellaVQModel,  # 初始化时传入的 VQGAN 模型
        latent_dim_scale: float = 10.67,  # 可选参数，默认值为 10.67
    ) -> None:
        super().__init__()  # 调用父类构造函数
        # 注册模型的各个模块
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            decoder=decoder,
            scheduler=scheduler,
            vqgan=vqgan,
        )
        # 将潜在维度缩放因子注册到配置中
        self.register_to_config(latent_dim_scale=latent_dim_scale)

    # 从 diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline 复制的方法，准备潜在变量
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果潜在变量为 None，则生成随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 检查传入的潜在变量形状是否与预期形状匹配
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在变量移动到指定设备
            latents = latents.to(device)

        # 用调度器的初始噪声标准差调整潜在变量
        latents = latents * scheduler.init_noise_sigma
        # 返回调整后的潜在变量
        return latents

    # 编码提示的方法
    def encode_prompt(
        self,
        prompt,  # 输入的提示
        device,  # 目标设备
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否进行无分类器引导
        negative_prompt=None,  # 负面提示（可选）
    @property
    # 获取引导缩放比例的属性
    def guidance_scale(self):
        return self._guidance_scale

    @property
    # 判断是否使用无分类器引导
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    # 获取时间步数的属性
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()  # 不计算梯度
    @replace_example_docstring(EXAMPLE_DOC_STRING)  # 替换示例文档字符串
    # 定义可调用对象的 __call__ 方法，允许实例像函数一样被调用
    def __call__(
            self,
            # 输入图像的嵌入，支持单个张量或张量列表
            image_embeddings: Union[torch.Tensor, List[torch.Tensor]],
            # 提示文本，可以是单个字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 推理步骤的数量，默认值为 12
            num_inference_steps: int = 12,
            # 指定时间步的列表，默认为 None
            timesteps: Optional[List[float]] = None,
            # 指导比例，控制生成的多样性，默认值为 0.0
            guidance_scale: float = 0.0,
            # 负提示文本，可以是单个字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认值为 1
            num_images_per_prompt: int = 1,
            # 随机数生成器，可选，支持单个或多个生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，可选，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 输出类型，默认值为 "pil"
            output_type: Optional[str] = "pil",
            # 返回字典标志，默认为 True
            return_dict: bool = True,
            # 结束步骤回调函数，可选，接收步骤信息
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 结束步骤回调函数使用的张量输入列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 其他可选参数，以关键字参数形式传递
            **kwargs,
```