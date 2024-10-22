# `.\diffusers\pipelines\deprecated\versatile_diffusion\pipeline_versatile_diffusion_text_to_image.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，否则根据许可证分发的软件在“按原样”基础上分发，
# 不提供任何形式的明示或暗示的担保或条件。
# 有关特定语言所管辖的权限和限制，请参阅许可证。

# 导入 inspect 模块以进行代码检查
import inspect
# 导入用于类型注释的相关类型
from typing import Callable, List, Optional, Union

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的检查点工具
import torch.utils.checkpoint
# 从 transformers 库导入 CLIP 相关模型和处理器
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer

# 从相对路径导入自定义图像处理器
from ....image_processor import VaeImageProcessor
# 从相对路径导入自定义模型
from ....models import AutoencoderKL, Transformer2DModel, UNet2DConditionModel
# 从相对路径导入调度器
from ....schedulers import KarrasDiffusionSchedulers
# 从相对路径导入工具函数和日志记录
from ....utils import deprecate, logging
from ....utils.torch_utils import randn_tensor
# 从相对路径导入 DiffusionPipeline 和 ImagePipelineOutput
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
# 从相对路径导入文本 UNet 模型
from .modeling_text_unet import UNetFlatConditionModel

# 初始化日志记录器，使用当前模块名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义 VersatileDiffusionTextToImagePipeline 类，继承自 DiffusionPipeline
class VersatileDiffusionTextToImagePipeline(DiffusionPipeline):
    r"""
    用于文本到图像生成的管道，使用 Versatile Diffusion。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档，以获取所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    参数：
        vqvae ([`VQModel`]):
            向量量化（VQ）模型，用于将图像编码和解码为潜在表示。
        bert ([`LDMBertModel`]):
            基于 [`~transformers.BERT`] 的文本编码器模型。
        tokenizer ([`~transformers.BertTokenizer`]):
            用于标记文本的 `BertTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于对编码的图像潜在数据进行去噪的 `UNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 一起去噪编码图像潜在数据的调度器。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "bert->unet->vqvae"

    # 声明 tokenizer 类型为 CLIPTokenizer
    tokenizer: CLIPTokenizer
    # 声明图像特征提取器类型为 CLIPImageProcessor
    image_feature_extractor: CLIPImageProcessor
    # 声明文本编码器类型为 CLIPTextModelWithProjection
    text_encoder: CLIPTextModelWithProjection
    # 声明图像 UNet 类型为 UNet2DConditionModel
    image_unet: UNet2DConditionModel
    # 声明文本 UNet 类型为 UNetFlatConditionModel
    text_unet: UNetFlatConditionModel
    # 声明 VAE 类型为 AutoencoderKL
    vae: AutoencoderKL
    # 声明调度器类型为 KarrasDiffusionSchedulers
    scheduler: KarrasDiffusionSchedulers

    # 定义可选组件列表，包含文本 UNet
    _optional_components = ["text_unet"]

    # 初始化方法，接受多个参数
    def __init__(
        # 初始化 tokenizer，类型为 CLIPTokenizer
        self,
        tokenizer: CLIPTokenizer,
        # 初始化文本编码器，类型为 CLIPTextModelWithProjection
        text_encoder: CLIPTextModelWithProjection,
        # 初始化图像 UNet，类型为 UNet2DConditionModel
        image_unet: UNet2DConditionModel,
        # 初始化文本 UNet，类型为 UNetFlatConditionModel
        text_unet: UNetFlatConditionModel,
        # 初始化 VAE，类型为 AutoencoderKL
        vae: AutoencoderKL,
        # 初始化调度器，类型为 KarrasDiffusionSchedulers
        scheduler: KarrasDiffusionSchedulers,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 注册各个模块，提供必要的组件
        self.register_modules(
            tokenizer=tokenizer,  # 注册分词器
            text_encoder=text_encoder,  # 注册文本编码器
            image_unet=image_unet,  # 注册图像 UNet 模型
            text_unet=text_unet,  # 注册文本 UNet 模型
            vae=vae,  # 注册变分自编码器
            scheduler=scheduler,  # 注册调度器
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器，使用计算得到的缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # 如果文本 UNet 模型存在，则交换其注意力块
        if self.text_unet is not None:
            self._swap_unet_attention_blocks()

    def _swap_unet_attention_blocks(self):
        """
        在图像和文本 UNet 之间交换 `Transformer2DModel` 块
        """
        # 遍历图像 UNet 的所有命名模块
        for name, module in self.image_unet.named_modules():
            # 如果模块是 Transformer2DModel 类型
            if isinstance(module, Transformer2DModel):
                # 分离父模块名称和索引
                parent_name, index = name.rsplit(".", 1)
                index = int(index)  # 将索引转换为整数
                # 交换图像 UNet 和文本 UNet 的相应模块
                self.image_unet.get_submodule(parent_name)[index], self.text_unet.get_submodule(parent_name)[index] = (
                    self.text_unet.get_submodule(parent_name)[index],
                    self.image_unet.get_submodule(parent_name)[index],
                )

    def remove_unused_weights(self):
        # 注册文本 UNet 为 None，以移除未使用的权重
        self.register_modules(text_unet=None)

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的 decode_latents 方法
    def decode_latents(self, latents):
        # 设置弃用警告信息
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 发出弃用警告
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 根据 VAE 的缩放因子调整潜在变量
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在变量，得到图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 对图像进行归一化处理
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为 float32 格式，兼容 bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像
        return image

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的 prepare_extra_step_kwargs 方法
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备调度器步骤的额外参数，因为不同调度器的签名不同
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将被忽略。
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应该在 [0, 1] 之间

        # 检查调度器是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}  # 初始化额外参数字典
        if accepts_eta:
            extra_step_kwargs["eta"] = eta  # 如果接受，添加 eta 参数

        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator  # 如果接受，添加 generator 参数
        # 返回准备好的额外参数字典
        return extra_step_kwargs
    # 定义一个检查输入参数的函数，确保传入的参数符合要求
    def check_inputs(
        self,  # 类的实例对象
        prompt,  # 文本提示，可能是字符串或列表
        height,  # 图像高度
        width,  # 图像宽度
        callback_steps,  # 回调的步骤数
        negative_prompt=None,  # 负面提示，可选参数
        prompt_embeds=None,  # 提示的嵌入向量，可选参数
        negative_prompt_embeds=None,  # 负面提示的嵌入向量，可选参数
        callback_on_step_end_tensor_inputs=None,  # 在步骤结束时的回调输入，可选参数
    ):
        # 检查高度和宽度是否是8的倍数
        if height % 8 != 0 or width % 8 != 0:
            # 如果不是，抛出值错误，提示高度和宽度的要求
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步骤的类型和有效性
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            # 如果回调步骤无效，抛出值错误
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        # 检查给定的回调输入是否在允许的输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 如果有不在允许输入中的项，抛出值错误
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查提示和提示嵌入的互斥性
        if prompt is not None and prompt_embeds is not None:
            # 如果两者都提供，抛出值错误
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否至少提供一个提示
        elif prompt is None and prompt_embeds is None:
            # 如果两个都未提供，抛出值错误
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查提示的类型是否有效
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 如果类型无效，抛出值错误
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查负面提示和负面提示嵌入的互斥性
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 如果两者都提供，抛出值错误
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查提示嵌入和负面提示嵌入的形状是否一致
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 如果形状不一致，抛出值错误
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制而来
    # 准备潜在变量，设置其形状与参数
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在变量的形状，考虑到 VAE 的缩放因子
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器列表的长度是否与批处理大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
    
        # 如果没有传入潜在变量，则生成随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在变量，则将其移动到指定设备
            latents = latents.to(device)
    
        # 根据调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回准备好的潜在变量
        return latents
    
    # 禁用梯度计算，以节省内存和提高性能
    @torch.no_grad()
    def __call__(
        # 提示信息，可以是字符串或字符串列表
        prompt: Union[str, List[str]],
        # 图像高度，可选
        height: Optional[int] = None,
        # 图像宽度，可选
        width: Optional[int] = None,
        # 推理步骤数量，默认为50
        num_inference_steps: int = 50,
        # 引导比例，默认为7.5
        guidance_scale: float = 7.5,
        # 负提示信息，可选
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为1
        num_images_per_prompt: Optional[int] = 1,
        # eta 值，默认为0.0
        eta: float = 0.0,
        # 生成器，可选
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜在变量，可选
        latents: Optional[torch.Tensor] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典格式的输出，默认为 True
        return_dict: bool = True,
        # 回调函数，可选
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调步骤间隔，默认为1
        callback_steps: int = 1,
        # 其他关键字参数
        **kwargs,
```