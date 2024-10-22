# `.\diffusers\pipelines\kandinsky3\pipeline_kandinsky3_img2img.py`

```py
import inspect  # 导入 inspect 模块，用于获取对象的详细信息
from typing import Callable, Dict, List, Optional, Union  # 导入类型提示相关的模块

import numpy as np  # 导入 numpy 库，用于数值计算
import PIL  # 导入 PIL 库，用于图像处理
import PIL.Image  # 从 PIL 导入 Image 模块，用于图像对象的操作
import torch  # 导入 PyTorch 库，用于深度学习
from transformers import T5EncoderModel, T5Tokenizer  # 从 transformers 导入 T5 模型和分词器

from ...loaders import StableDiffusionLoraLoaderMixin  # 导入混合类，用于加载 Lora 模型
from ...models import Kandinsky3UNet, VQModel  # 导入 Kandinsky3 UNet 和 VQModel 模型
from ...schedulers import DDPMScheduler  # 导入 DDPMScheduler，用于调度器
from ...utils import (  # 导入工具函数
    deprecate,  # 导入弃用装饰器
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 导入用于替换示例文档字符串的函数
)
from ...utils.torch_utils import randn_tensor  # 从工具模块导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 导入扩散管道和图像输出模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，使用模块名称作为标识，禁用 pylint 的无效名称警告

EXAMPLE_DOC_STRING = """  # 示例文档字符串，用于展示如何使用该管道
    Examples:  # 示例部分的开始
        ```py  # Python 示例代码块开始
        >>> from diffusers import AutoPipelineForImage2Image  # 导入图像转图像的自动管道
        >>> from diffusers.utils import load_image  # 导入加载图像的工具函数
        >>> import torch  # 导入 PyTorch 库

        >>> pipe = AutoPipelineForImage2Image.from_pretrained(  # 从预训练模型加载图像转图像的管道
        ...     "kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16  # 指定模型和数据类型
        ... )
        >>> pipe.enable_model_cpu_offload()  # 启用模型的 CPU 内存释放

        >>> prompt = "A painting of the inside of a subway train with tiny raccoons."  # 定义图像生成的提示
        >>> image = load_image(  # 加载图像
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/t2i.png"  # 图像的 URL
        ... )

        >>> generator = torch.Generator(device="cpu").manual_seed(0)  # 创建 CPU 设备上的随机数生成器并设置种子
        >>> image = pipe(prompt, image=image, strength=0.75, num_inference_steps=25, generator=generator).images[0]  # 生成图像并获取结果
        ```py  # 示例代码块结束
"""

def downscale_height_and_width(height, width, scale_factor=8):  # 定义函数以按比例缩放高度和宽度
    new_height = height // scale_factor**2  # 计算新的高度
    if height % scale_factor**2 != 0:  # 检查高度是否可以整除缩放因子平方
        new_height += 1  # 如果不能，增加高度
    new_width = width // scale_factor**2  # 计算新的宽度
    if width % scale_factor**2 != 0:  # 检查宽度是否可以整除缩放因子平方
        new_width += 1  # 如果不能，增加宽度
    return new_height * scale_factor, new_width * scale_factor  # 返回按缩放因子调整后的新高度和新宽度

def prepare_image(pil_image):  # 定义函数以准备 PIL 图像
    arr = np.array(pil_image.convert("RGB"))  # 将 PIL 图像转换为 RGB 并转为 NumPy 数组
    arr = arr.astype(np.float32) / 127.5 - 1  # 将数组转换为浮点数并归一化到 [-1, 1] 范围
    arr = np.transpose(arr, [2, 0, 1])  # 转换数组维度，从 (H, W, C) 到 (C, H, W)
    image = torch.from_numpy(arr).unsqueeze(0)  # 将 NumPy 数组转换为 PyTorch 张量并增加一个维度
    return image  # 返回处理后的图像张量

class Kandinsky3Img2ImgPipeline(DiffusionPipeline, StableDiffusionLoraLoaderMixin):  # 定义 Kandinsky 3 图像到图像的管道类，继承自 DiffusionPipeline 和 StableDiffusionLoraLoaderMixin
    model_cpu_offload_seq = "text_encoder->movq->unet->movq"  # 定义模型 CPU 内存释放的顺序
    _callback_tensor_inputs = [  # 定义需要回调的张量输入
        "latents",  # 潜在表示
        "prompt_embeds",  # 提示嵌入
        "negative_prompt_embeds",  # 负提示嵌入
        "negative_attention_mask",  # 负注意力掩码
        "attention_mask",  # 注意力掩码
    ]

    def __init__(  # 初始化方法
        self,
        tokenizer: T5Tokenizer,  # T5 分词器
        text_encoder: T5EncoderModel,  # T5 文本编码器
        unet: Kandinsky3UNet,  # Kandinsky3 UNet 模型
        scheduler: DDPMScheduler,  # DDPMScheduler 实例
        movq: VQModel,  # VQModel 实例
    ):
        super().__init__()  # 调用父类的初始化方法

        self.register_modules(  # 注册各个模块
            tokenizer=tokenizer, text_encoder=text_encoder, unet=unet, scheduler=scheduler, movq=movq  # 注册分词器、文本编码器、UNet、调度器和 VQModel
        )
    def get_timesteps(self, num_inference_steps, strength, device):
        # 获取初始时间步，使用给定的推理步骤数量和强度
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算开始时间步，确保不小于0
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从计算的开始时间步获取调度器中的时间步
        timesteps = self.scheduler.timesteps[t_start:]

        # 返回时间步和剩余的推理步骤数量
        return timesteps, num_inference_steps - t_start

    def _process_embeds(self, embeddings, attention_mask, cut_context):
        # 返回处理后的嵌入和注意力掩码
        if cut_context:
            # 将注意力掩码为0的嵌入置为零
            embeddings[attention_mask == 0] = torch.zeros_like(embeddings[attention_mask == 0])
            # 计算最大序列长度，并在此基础上切片嵌入和注意力掩码
            max_seq_length = attention_mask.sum(-1).max() + 1
            embeddings = embeddings[:, :max_seq_length]
            attention_mask = attention_mask[:, :max_seq_length]
        # 返回处理后的嵌入和注意力掩码
        return embeddings, attention_mask

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt,
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        device=None,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        _cut_context=False,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
    ):
        # 准备潜变量，处理输入图像和时间步
        def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
            # 检查输入图像类型是否合法
            if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
                raise ValueError(
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
                )

            # 将图像转移到指定设备，并设置数据类型
            image = image.to(device=device, dtype=dtype)

            # 计算实际批次大小
            batch_size = batch_size * num_images_per_prompt

            # 如果图像的通道数为4，初始化潜变量为输入图像
            if image.shape[1] == 4:
                init_latents = image

            else:
                # 如果生成器是列表且长度与批次大小不符，抛出错误
                if isinstance(generator, list) and len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )

                # 如果生成器是列表，逐个编码图像并采样
                elif isinstance(generator, list):
                    init_latents = [
                        self.movq.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                    ]
                    # 将所有潜变量合并到一起
                    init_latents = torch.cat(init_latents, dim=0)
                else:
                    # 否则直接编码并采样
                    init_latents = self.movq.encode(image).latent_dist.sample(generator)

                # 根据配置的缩放因子调整潜变量
                init_latents = self.movq.config.scaling_factor * init_latents

            # 将初始化的潜变量在第一维上进行合并
            init_latents = torch.cat([init_latents], dim=0)

            # 获取初始化潜变量的形状
            shape = init_latents.shape
            # 生成噪声张量，用于后续处理
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            # 将噪声添加到潜变量中，获取最终潜变量
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

            # 设置最终的潜变量
            latents = init_latents

            # 返回潜变量
            return latents
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制而来
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，因为并不是所有的调度器都有相同的参数签名
        # eta（η）仅在 DDIMScheduler 中使用，其他调度器将忽略它。
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 其值应在 [0, 1] 之间

        # 检查调度器的步骤方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数字典
        extra_step_kwargs = {}
        # 如果调度器接受 eta 参数，则将其添加到额外步骤参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的步骤方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator 参数，则将其添加到额外步骤参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤参数字典
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        attention_mask=None,
        negative_attention_mask=None,
    @property
    # 定义属性方法，获取指导尺度
    def guidance_scale(self):
        # 返回当前指导尺度的值
        return self._guidance_scale

    @property
    # 定义属性方法，检查是否进行无分类器自由指导
    def do_classifier_free_guidance(self):
        # 如果指导尺度大于 1，则返回 True
        return self._guidance_scale > 1

    @property
    # 定义属性方法，获取时间步数
    def num_timesteps(self):
        # 返回当前时间步数的值
        return self._num_timesteps

    @torch.no_grad()
    # 装饰器，表示该方法在推理时不计算梯度
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 定义可调用对象的方法，接受多个参数用于生成图像
        prompt: Union[str, List[str]] = None,
        image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]] = None,
        strength: float = 0.3,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 定义在步骤结束时调用的回调，默认为只包含 "latents" 的列表
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
```