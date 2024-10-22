# `.\diffusers\pipelines\stable_cascade\pipeline_stable_cascade.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）许可；
# 除非遵循该许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，按照许可证分发的软件均按“原样”提供，
# 不附带任何形式的明示或暗示的担保或条件。
# 有关许可证下的特定语言和权限限制，请参见许可证。

from typing import Callable, Dict, List, Optional, Union  # 从 typing 模块导入类型提示所需的类

import torch  # 导入 PyTorch 库
from transformers import CLIPTextModel, CLIPTokenizer  # 从 transformers 导入 CLIP 文本模型和标记器

from ...models import StableCascadeUNet  # 从当前包导入 StableCascadeUNet 模型
from ...schedulers import DDPMWuerstchenScheduler  # 从当前包导入调度器 DDPMWuerstchenScheduler
from ...utils import is_torch_version, logging, replace_example_docstring  # 从 utils 导入工具函数
from ...utils.torch_utils import randn_tensor  # 从 torch_utils 导入 randn_tensor 函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从 pipeline_utils 导入 DiffusionPipeline 和 ImagePipelineOutput
from ..wuerstchen.modeling_paella_vq_model import PaellaVQModel  # 从 wuerstchen 模块导入 PaellaVQModel

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

EXAMPLE_DOC_STRING = """  # 示例文档字符串，提供使用示例
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline  # 导入管道类

        >>> prior_pipe = StableCascadePriorPipeline.from_pretrained(  # 从预训练模型加载 StableCascadePriorPipeline
        ...     "stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16  # 指定模型名称和数据类型
        ... ).to("cuda")  # 将管道移至 CUDA 设备
        >>> gen_pipe = StableCascadeDecoderPipeline.from_pretrain(  # 从预训练模型加载 StableCascadeDecoderPipeline
        ...     "stabilityai/stable-cascade", torch_dtype=torch.float16  # 指定模型名称和数据类型
        ... ).to("cuda")  # 将管道移至 CUDA 设备

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"  # 定义生成图像的提示
        >>> prior_output = pipe(prompt)  # 使用提示生成初步输出
        >>> images = gen_pipe(prior_output.image_embeddings, prompt=prompt)  # 基于初步输出生成最终图像
        ```py  # 结束示例代码块
"""

class StableCascadeDecoderPipeline(DiffusionPipeline):  # 定义 StableCascadeDecoderPipeline 类，继承自 DiffusionPipeline
    """
    用于从 Stable Cascade 模型生成图像的管道。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档以了解库为所有管道实现的通用方法
    （如下载或保存，运行在特定设备等）。
    # 参数说明
    Args:
        tokenizer (`CLIPTokenizer`):
            # CLIP 的分词器，用于处理文本输入
            The CLIP tokenizer.
        text_encoder (`CLIPTextModel`):
            # CLIP 的文本编码器，负责将文本转化为向量表示
            The CLIP text encoder.
        decoder ([`StableCascadeUNet`]):
            # 稳定的级联解码器 UNet，用于生成图像
            The Stable Cascade decoder unet.
        vqgan ([`PaellaVQModel`]):
            # VQGAN 模型，用于图像生成
            The VQGAN model.
        scheduler ([`DDPMWuerstchenScheduler`]):
            # 调度器，与 `prior` 结合用于生成图像嵌入
            A scheduler to be used in combination with `prior` to generate image embedding.
        latent_dim_scale (float, `optional`, defaults to 10.67):
            # 用于从图像嵌入计算 VQ 潜在空间大小的倍数
            Multiplier to determine the VQ latent space size from the image embeddings. If the image embeddings are
            height=24 and width=24, the VQ latent shape needs to be height=int(24*10.67)=256 and
            width=int(24*10.67)=256 in order to match the training conditions.
    """

    # 设置解码器名称
    unet_name = "decoder"
    # 设置文本编码器名称
    text_encoder_name = "text_encoder"
    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->decoder->vqgan"
    # 注册需要回调的张量输入
    _callback_tensor_inputs = [
        "latents",  # 潜在变量
        "prompt_embeds_pooled",  # 处理后的提示嵌入
        "negative_prompt_embeds",  # 负面提示嵌入
        "image_embeddings",  # 图像嵌入
    ]

    # 初始化方法
    def __init__(
        self,
        decoder: StableCascadeUNet,  # 解码器模型
        tokenizer: CLIPTokenizer,  # 分词器
        text_encoder: CLIPTextModel,  # 文本编码器
        scheduler: DDPMWuerstchenScheduler,  # 调度器
        vqgan: PaellaVQModel,  # VQGAN 模型
        latent_dim_scale: float = 10.67,  # 潜在维度缩放因子
    ) -> None:
        # 调用父类构造方法
        super().__init__()
        # 注册模块
        self.register_modules(
            decoder=decoder,  # 注册解码器
            tokenizer=tokenizer,  # 注册分词器
            text_encoder=text_encoder,  # 注册文本编码器
            scheduler=scheduler,  # 注册调度器
            vqgan=vqgan,  # 注册 VQGAN
        )
        # 将潜在维度缩放因子注册到配置中
        self.register_to_config(latent_dim_scale=latent_dim_scale)

    # 准备潜在变量的方法
    def prepare_latents(
        self, 
        batch_size,  # 批大小
        image_embeddings,  # 图像嵌入
        num_images_per_prompt,  # 每个提示生成的图像数量
        dtype,  # 数据类型
        device,  # 设备信息
        generator,  # 随机数生成器
        latents,  # 潜在变量
        scheduler  # 调度器
    ):
        # 获取图像嵌入的形状信息
        _, channels, height, width = image_embeddings.shape
        # 定义潜在变量的形状
        latents_shape = (
            batch_size * num_images_per_prompt,  # 总图像数量
            4,  # 通道数
            int(height * self.config.latent_dim_scale),  # 潜在图像高度
            int(width * self.config.latent_dim_scale),  # 潜在图像宽度
        )

        # 如果没有提供潜在变量，则生成随机潜在变量
        if latents is None:
            latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供的潜在变量形状不符合预期，则抛出异常
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            # 将潜在变量转移到指定设备
            latents = latents.to(device)

        # 将潜在变量与调度器的初始噪声标准差相乘
        latents = latents * scheduler.init_noise_sigma
        # 返回准备好的潜在变量
        return latents

    # 编码提示的方法
    def encode_prompt(
        self,
        device,  # 设备信息
        batch_size,  # 批大小
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否进行无分类器引导
        prompt=None,  # 提示文本
        negative_prompt=None,  # 负面提示文本
        prompt_embeds: Optional[torch.Tensor] = None,  # 提示嵌入（可选）
        prompt_embeds_pooled: Optional[torch.Tensor] = None,  # 处理后的提示嵌入（可选）
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负面提示嵌入（可选）
        negative_prompt_embeds_pooled: Optional[torch.Tensor] = None,  # 处理后的负面提示嵌入（可选）
    # 检查输入参数的有效性
        def check_inputs(
            self,  # 当前类实例
            prompt,  # 正向提示文本
            negative_prompt=None,  # 负向提示文本，默认为 None
            prompt_embeds=None,  # 正向提示的嵌入表示，默认为 None
            negative_prompt_embeds=None,  # 负向提示的嵌入表示，默认为 None
            callback_on_step_end_tensor_inputs=None,  # 回调函数输入，默认为 None
        ):
            # 检查回调函数输入是否为 None，并验证每个输入是否在预定义的回调输入列表中
            if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
            ):
                # 如果输入无效，抛出错误并列出无效的输入
                raise ValueError(
                    f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
                )
    
            # 检查正向提示和正向嵌入是否同时提供
            if prompt is not None and prompt_embeds is not None:
                # 抛出错误，说明不能同时提供两者
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            # 检查正向提示和正向嵌入是否都未提供
            elif prompt is None and prompt_embeds is None:
                # 抛出错误，要求至少提供一个
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            # 检查正向提示的类型是否正确
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                # 抛出错误，说明类型不符合要求
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
            # 检查负向提示和负向嵌入是否同时提供
            if negative_prompt is not None and negative_prompt_embeds is not None:
                # 抛出错误，说明不能同时提供两者
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
    
            # 检查正向嵌入和负向嵌入的形状是否匹配
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    # 抛出错误，说明形状不匹配
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    
        # 定义一个属性，获取指导比例
        @property
        def guidance_scale(self):
            # 返回指导比例的值
            return self._guidance_scale
    
        # 定义一个属性，判断是否进行无分类器指导
        @property
        def do_classifier_free_guidance(self):
            # 返回指导比例是否大于 1
            return self._guidance_scale > 1
    
        # 定义一个属性，获取时间步数
        @property
        def num_timesteps(self):
            # 返回时间步数的值
            return self._num_timesteps
    
        # 禁用梯度计算的装饰器
        @torch.no_grad()
        # 替换示例文档字符串的装饰器
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用类的方法，用于生成图像
        def __call__(
            self,
            # 输入的图像嵌入，可以是单个张量或张量列表
            image_embeddings: Union[torch.Tensor, List[torch.Tensor]],
            # 提示词，可以是字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 推理步骤的数量，默认为10
            num_inference_steps: int = 10,
            # 引导尺度，默认为0.0
            guidance_scale: float = 0.0,
            # 负提示词，可以是字符串或字符串列表，默认为None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 提示词嵌入，默认为None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 提示词池化后的嵌入，默认为None
            prompt_embeds_pooled: Optional[torch.Tensor] = None,
            # 负提示词嵌入，默认为None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示词池化后的嵌入，默认为None
            negative_prompt_embeds_pooled: Optional[torch.Tensor] = None,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: int = 1,
            # 随机数生成器，可以是单个或列表，默认为None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，默认为None
            latents: Optional[torch.Tensor] = None,
            # 输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式，默认为True
            return_dict: bool = True,
            # 每步结束时的回调函数，默认为None
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 每步结束时使用的张量输入列表，默认为包含"latents"
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
```