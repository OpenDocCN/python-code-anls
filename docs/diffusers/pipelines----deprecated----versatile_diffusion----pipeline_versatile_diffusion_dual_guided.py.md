# `.\diffusers\pipelines\deprecated\versatile_diffusion\pipeline_versatile_diffusion_dual_guided.py`

```py
# 版权声明，标识文件归 HuggingFace 团队所有，保留所有权利
# 使用 Apache 2.0 许可证，要求遵守许可证条款
# 许可证的获取地址
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用的法律或书面协议另有约定，软件按 "原样" 分发
# 不提供任何形式的明示或暗示的担保或条件
# 详细信息见许可证中的具体权限和限制条款

# 导入 inspect 模块，用于获取活跃的对象信息
import inspect
# 从 typing 模块导入类型提示，用于类型注释
from typing import Callable, List, Optional, Tuple, Union

# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 PIL.Image，用于图像处理
import PIL.Image
# 导入 torch 库，提供深度学习功能
import torch
# 导入 torch.utils.checkpoint，用于模型检查点功能
import torch.utils.checkpoint
# 从 transformers 库导入 CLIP 相关类，用于图像和文本处理
from transformers import (
    CLIPImageProcessor,  # 图像处理器
    CLIPTextModelWithProjection,  # 带投影的文本模型
    CLIPTokenizer,  # 文本分词器
    CLIPVisionModelWithProjection,  # 带投影的视觉模型
)

# 从本地模块导入图像处理和模型相关的类
from ....image_processor import VaeImageProcessor  # VAE 图像处理器
from ....models import AutoencoderKL, DualTransformer2DModel, Transformer2DModel, UNet2DConditionModel  # 各种模型
from ....schedulers import KarrasDiffusionSchedulers  # 调度器
from ....utils import deprecate, logging  # 工具函数和日志记录
from ....utils.torch_utils import randn_tensor  # 随机张量生成
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 管道相关工具
from .modeling_text_unet import UNetFlatConditionModel  # 文本条件模型

# 创建日志记录器实例，用于当前模块
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个多功能扩散双重引导管道类
class VersatileDiffusionDualGuidedPipeline(DiffusionPipeline):
    r""" 
    使用多功能扩散的图像-文本双重引导生成的管道。
    
    该模型继承自 [`DiffusionPipeline`]。查阅超类文档以获取所有管道的通用方法 
    （下载、保存、在特定设备上运行等）。
    
    参数：
        vqvae ([`VQModel`]):
            向量量化（VQ）模型，用于将图像编码和解码为潜在表示。
        bert ([`LDMBertModel`]):
            基于 [`~transformers.BERT`] 的文本编码器模型。
        tokenizer ([`~transformers.BertTokenizer`]):
            用于文本分词的 `BertTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于去噪编码图像潜在的 `UNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 结合使用的调度器，以去噪编码的图像潜在。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 中的任何一个。
    """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "bert->unet->vqvae"

    # 定义类属性，包含不同模型组件
    tokenizer: CLIPTokenizer  # 文本分词器
    image_feature_extractor: CLIPImageProcessor  # 图像特征提取器
    text_encoder: CLIPTextModelWithProjection  # 文本编码器
    image_encoder: CLIPVisionModelWithProjection  # 图像编码器
    image_unet: UNet2DConditionModel  # 图像去噪模型
    text_unet: UNetFlatConditionModel  # 文本条件去噪模型
    vae: AutoencoderKL  # 自编码器
    scheduler: KarrasDiffusionSchedulers  # 调度器

    # 定义可选组件
    _optional_components = ["text_unet"]  # 可选的文本去噪模型组件
    # 初始化方法，设置模型的基本组件
        def __init__(
            self,
            tokenizer: CLIPTokenizer,  # 用于文本的分词器
            image_feature_extractor: CLIPImageProcessor,  # 图像特征提取器
            text_encoder: CLIPTextModelWithProjection,  # 文本编码器，带有投影层
            image_encoder: CLIPVisionModelWithProjection,  # 图像编码器，带有投影层
            image_unet: UNet2DConditionModel,  # 用于图像处理的 UNet 模型
            text_unet: UNetFlatConditionModel,  # 用于文本处理的 UNet 模型
            vae: AutoencoderKL,  # 变分自编码器
            scheduler: KarrasDiffusionSchedulers,  # 调度器，用于控制训练过程
        ):
            # 调用父类的初始化方法
            super().__init__()
            # 注册各个模块，便于管理和调用
            self.register_modules(
                tokenizer=tokenizer,
                image_feature_extractor=image_feature_extractor,
                text_encoder=text_encoder,
                image_encoder=image_encoder,
                image_unet=image_unet,
                text_unet=text_unet,
                vae=vae,
                scheduler=scheduler,
            )
            # 计算 VAE 的缩放因子，用于图像处理
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器实例，使用计算出的缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
            # 检查文本 UNet 是否存在且图像 UNet 配置中不包含双交叉注意力
            if self.text_unet is not None and (
                "dual_cross_attention" not in self.image_unet.config or not self.image_unet.config.dual_cross_attention
            ):
                # 如果从通用检查点加载而非保存的双引导管道，转换为双注意力
                self._convert_to_dual_attention()
    
        # 移除未使用的权重
        def remove_unused_weights(self):
            # 将 text_unet 注册为 None，以释放资源
            self.register_modules(text_unet=None)
    # 定义一个私有方法，用于将图像的 UNet 转换为双重注意力机制
        def _convert_to_dual_attention(self):
            """
            替换 image_unet 的 `Transformer2DModel` 块为包含来自 `image_unet` 和 `text_unet` 的 transformer 块的 `DualTransformer2DModel`
            """
            # 遍历 image_unet 中的所有命名模块
            for name, module in self.image_unet.named_modules():
                # 检查当前模块是否为 Transformer2DModel 的实例
                if isinstance(module, Transformer2DModel):
                    # 分割模块名称，获取父级名称和索引
                    parent_name, index = name.rsplit(".", 1)
                    index = int(index)
    
                    # 获取图像和文本的 transformer 模块
                    image_transformer = self.image_unet.get_submodule(parent_name)[index]
                    text_transformer = self.text_unet.get_submodule(parent_name)[index]
    
                    # 获取图像 transformer 的配置
                    config = image_transformer.config
                    # 创建双重 transformer 模型
                    dual_transformer = DualTransformer2DModel(
                        num_attention_heads=config.num_attention_heads,
                        attention_head_dim=config.attention_head_dim,
                        in_channels=config.in_channels,
                        num_layers=config.num_layers,
                        dropout=config.dropout,
                        norm_num_groups=config.norm_num_groups,
                        cross_attention_dim=config.cross_attention_dim,
                        attention_bias=config.attention_bias,
                        sample_size=config.sample_size,
                        num_vector_embeds=config.num_vector_embeds,
                        activation_fn=config.activation_fn,
                        num_embeds_ada_norm=config.num_embeds_ada_norm,
                    )
                    # 将图像 transformer 和文本 transformer 分别赋值给双重 transformer
                    dual_transformer.transformers[0] = image_transformer
                    dual_transformer.transformers[1] = text_transformer
    
                    # 替换原有的模块为双重 transformer 模块
                    self.image_unet.get_submodule(parent_name)[index] = dual_transformer
                    # 注册配置，启用双重交叉注意力
                    self.image_unet.register_to_config(dual_cross_attention=True)
    
        # 定义一个私有方法，用于将双重注意力机制还原为图像 UNet 的标准 transformer
        def _revert_dual_attention(self):
            """
            将 image_unet 的 `DualTransformer2DModel` 块还原为带有 image_unet 权重的 `Transformer2DModel` 
            如果在另一个管道中重用 `image_unet`，例如 `VersatileDiffusionPipeline`，请调用此函数
            """
            # 遍历 image_unet 中的所有命名模块
            for name, module in self.image_unet.named_modules():
                # 检查当前模块是否为 DualTransformer2DModel 的实例
                if isinstance(module, DualTransformer2DModel):
                    # 分割模块名称，获取父级名称和索引
                    parent_name, index = name.rsplit(".", 1)
                    index = int(index)
                    # 将双重 transformer 的第一个 transformer 还原到原有模块
                    self.image_unet.get_submodule(parent_name)[index] = module.transformers[0]
    
            # 注册配置，禁用双重交叉注意力
            self.image_unet.register_to_config(dual_cross_attention=False)
    # 定义一个私有方法用于将提示编码为文本编码器的隐藏状态
    def _encode_image_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance):
        r"""
        将提示编码为文本编码器的隐藏状态。
    
        参数:
            prompt (`str` 或 `List[str]`):
                要编码的提示
            device: (`torch.device`):
                PyTorch 设备
            num_images_per_prompt (`int`):
                每个提示生成的图像数量
            do_classifier_free_guidance (`bool`):
                是否使用无分类器引导
        """
    
        # 定义一个私有方法用于标准化嵌入
        def normalize_embeddings(encoder_output):
            # 对编码器输出进行层归一化
            embeds = self.image_encoder.vision_model.post_layernorm(encoder_output.last_hidden_state)
            # 进行视觉投影以获得嵌入
            embeds = self.image_encoder.visual_projection(embeds)
            # 取第一个嵌入进行池化
            embeds_pooled = embeds[:, 0:1]
            # 归一化嵌入
            embeds = embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)
            return embeds
    
        # 根据提示类型确定批大小
        batch_size = len(prompt) if isinstance(prompt, list) else 1
    
        # 获取提示文本的嵌入
        image_input = self.image_feature_extractor(images=prompt, return_tensors="pt")
        # 将像素值移动到指定设备并转换为相应的数据类型
        pixel_values = image_input.pixel_values.to(device).to(self.image_encoder.dtype)
        # 通过图像编码器获取图像嵌入
        image_embeddings = self.image_encoder(pixel_values)
        # 对图像嵌入进行标准化处理
        image_embeddings = normalize_embeddings(image_embeddings)
    
        # 复制图像嵌入以适应每个提示的生成，采用适合 MPS 的方法
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        # 重塑图像嵌入的形状以适应批处理
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
    
        # 获取无条件的嵌入以用于无分类器引导
        if do_classifier_free_guidance:
            # 创建一个形状为 (512, 512, 3) 的全零图像数组，值为0.5
            uncond_images = [np.zeros((512, 512, 3)) + 0.5] * batch_size
            # 获取无条件图像的特征
            uncond_images = self.image_feature_extractor(images=uncond_images, return_tensors="pt")
            # 将无条件图像的像素值移动到指定设备并转换为相应的数据类型
            pixel_values = uncond_images.pixel_values.to(device).to(self.image_encoder.dtype)
            # 通过图像编码器获取负提示嵌入
            negative_prompt_embeds = self.image_encoder(pixel_values)
            # 对负提示嵌入进行标准化处理
            negative_prompt_embeds = normalize_embeddings(negative_prompt_embeds)
    
            # 复制无条件嵌入以适应每个提示的生成，采用适合 MPS 的方法
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            # 重塑无条件嵌入的形状以适应批处理
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    
            # 为了进行无分类器引导，需要进行两次前向传递
            # 这里将无条件嵌入和条件嵌入连接成一个批次
            # 以避免进行两次前向传递
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])
    
        # 返回最终的图像嵌入
        return image_embeddings
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制
    def decode_latents(self, latents):
        # 定义弃用信息，说明该方法将在 1.0.0 版本中移除，并提供替代方法
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用弃用函数，发出警告
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 根据 VAE 配置的缩放因子对潜在向量进行缩放
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在向量，获取图像数据
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像数据规范化到 [0, 1] 范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为 float32 格式，以确保兼容性且不增加显著开销
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像
        return image

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备额外的参数供调度器步骤使用，不同调度器的参数签名可能不同
        # eta（η）仅在 DDIMScheduler 中使用，其他调度器将忽略它
        # eta 对应于 DDIM 论文中的 η，应在 [0, 1] 范围内

        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数字典
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外步骤参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外步骤参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回额外步骤参数字典
        return extra_step_kwargs

    def check_inputs(self, prompt, image, height, width, callback_steps):
        # 检查 prompt 类型，必须为 str、PIL.Image 或 list
        if not isinstance(prompt, str) and not isinstance(prompt, PIL.Image.Image) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` `PIL.Image` or `list` but is {type(prompt)}")
        # 检查 image 类型，必须为 str、PIL.Image 或 list
        if not isinstance(image, str) and not isinstance(image, PIL.Image.Image) and not isinstance(image, list):
            raise ValueError(f"`image` has to be of type `str` `PIL.Image` or `list` but is {type(image)}")

        # 检查 height 和 width 是否为 8 的倍数
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查 callback_steps 是否为正整数
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
    # 准备潜在变量，定义形状和相关参数
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在变量的形状，包括批量大小、通道数和缩放后的高度和宽度
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器是否为列表且长度与批量大小不匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    # 抛出错误，提示生成器数量与批量大小不匹配
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果没有提供潜在变量，随机生成潜在变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果提供了潜在变量，将其移动到指定设备
                latents = latents.to(device)
    
            # 将初始噪声按调度器要求的标准差进行缩放
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在变量
            return latents
    
        # 设置变换器的参数，包括混合比例和条件类型
        def set_transformer_params(self, mix_ratio: float = 0.5, condition_types: Tuple = ("text", "image")):
            # 遍历命名模块，查找 DualTransformer2DModel 模块
            for name, module in self.image_unet.named_modules():
                if isinstance(module, DualTransformer2DModel):
                    # 设置模块的混合比例
                    module.mix_ratio = mix_ratio
    
                    # 遍历条件类型，设置每种条件的参数
                    for i, type in enumerate(condition_types):
                        if type == "text":
                            # 为文本条件设置长度和变换器索引
                            module.condition_lengths[i] = self.text_encoder.config.max_position_embeddings
                            module.transformer_index_for_condition[i] = 1  # 使用第二个（文本）变换器
                        else:
                            # 为图像条件设置长度和变换器索引
                            module.condition_lengths[i] = 257
                            module.transformer_index_for_condition[i] = 0  # 使用第一个（图像）变换器
    
        # 不计算梯度的调用方法，处理输入参数
        @torch.no_grad()
        def __call__(
            # 输入的提示，可以是单张或多张图像
            prompt: Union[PIL.Image.Image, List[PIL.Image.Image]],
            # 输入的图像文件路径
            image: Union[str, List[str]],
            # 文本到图像的强度
            text_to_image_strength: float = 0.5,
            # 可选的图像高度
            height: Optional[int] = None,
            # 可选的图像宽度
            width: Optional[int] = None,
            # 推理步骤的数量
            num_inference_steps: int = 50,
            # 指导比例
            guidance_scale: float = 7.5,
            # 每个提示生成的图像数量
            num_images_per_prompt: Optional[int] = 1,
            # 超参数
            eta: float = 0.0,
            # 随机数生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜在变量
            latents: Optional[torch.Tensor] = None,
            # 输出类型，默认为 PIL 图像
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果
            return_dict: bool = True,
            # 回调函数，用于推理过程中的处理
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调步骤
            callback_steps: int = 1,
            # 其他可选参数
            **kwargs,
```