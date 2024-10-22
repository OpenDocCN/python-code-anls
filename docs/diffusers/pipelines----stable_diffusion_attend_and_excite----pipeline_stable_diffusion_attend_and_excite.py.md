# `.\diffusers\pipelines\stable_diffusion_attend_and_excite\pipeline_stable_diffusion_attend_and_excite.py`

```py
# 版权声明，说明该文件的版权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守该许可证，否则不得使用此文件。
# 可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，软件
# 根据许可证分发是以“原样”基础进行的，
# 不提供任何形式的明示或暗示的担保或条件。
# 查看许可证以了解有关权限和
# 限制的具体信息。

# 导入 inspect 模块，用于获取对象的信息
import inspect
# 导入 math 模块，提供数学函数
import math
# 从 typing 模块导入各种类型注解
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 numpy 库，进行数值计算
import numpy as np
# 导入 torch 库，进行深度学习操作
import torch
# 从 torch.nn.functional 导入常用的神经网络功能
from torch.nn import functional as F
# 从 transformers 库导入 CLIP 相关的处理器和模型
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

# 从相对路径导入 VaeImageProcessor 类
from ...image_processor import VaeImageProcessor
# 从相对路径导入 Lora 和 Textual Inversion 的加载器混合类
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 从相对路径导入自动编码器和 U-Net 模型
from ...models import AutoencoderKL, UNet2DConditionModel
# 从相对路径导入 Attention 类
from ...models.attention_processor import Attention
# 从相对路径导入调整 LoRA 规模的函数
from ...models.lora import adjust_lora_scale_text_encoder
# 从相对路径导入 Karras Diffusion 调度器
from ...schedulers import KarrasDiffusionSchedulers
# 从相对路径导入实用工具
from ...utils import (
    USE_PEFT_BACKEND,       # 导入标志以使用 PEFT 后端
    deprecate,             # 导入用于标记弃用功能的工具
    logging,               # 导入日志记录功能
    replace_example_docstring, # 导入用于替换示例文档字符串的工具
    scale_lora_layers,     # 导入缩放 LoRA 层的函数
    unscale_lora_layers,   # 导入取消缩放 LoRA 层的函数
)
# 从相对路径导入随机张量生成工具
from ...utils.torch_utils import randn_tensor
# 从相对路径导入扩散管道和稳定扩散混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从相对路径导入稳定扩散的输出模型
from ..stable_diffusion import StableDiffusionPipelineOutput
# 从相对路径导入稳定扩散的安全检查器
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker

# 创建日志记录器，使用当前模块的名称
logger = logging.get_logger(__name__)

# 示例文档字符串，展示如何使用此类的示例代码
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionAttendAndExcitePipeline

        >>> pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        ... ).to("cuda")

        >>> prompt = "a cat and a frog"

        >>> # 使用 get_indices 函数查找要更改的令牌的索引
        >>> pipe.get_indices(prompt)
        {0: '<|startoftext|>', 1: 'a</w>', 2: 'cat</w>', 3: 'and</w>', 4: 'a</w>', 5: 'frog</w>', 6: '<|endoftext|>'}

        >>> token_indices = [2, 5]
        >>> seed = 6141
        >>> generator = torch.Generator("cuda").manual_seed(seed)

        >>> images = pipe(
        ...     prompt=prompt,
        ...     token_indices=token_indices,
        ...     guidance_scale=7.5,
        ...     generator=generator,
        ...     num_inference_steps=50,
        ...     max_iter_to_alter=25,
        ... ).images

        >>> image = images[0]
        >>> image.save(f"../images/{prompt}_{seed}.png")
        ```py
"""

# 定义 AttentionStore 类
class AttentionStore:
    # 定义静态方法，用于获取一个空的注意力存储
    @staticmethod
    def get_empty_store():
        # 返回一个包含三个空列表的字典，分别对应不同的注意力层
        return {"down": [], "mid": [], "up": []}
    # 定义一个可调用的函数，处理注意力矩阵
        def __call__(self, attn, is_cross: bool, place_in_unet: str):
            # 如果当前注意力层索引有效且为交叉注意力
            if self.cur_att_layer >= 0 and is_cross:
                # 检查注意力矩阵的形状是否与期望的分辨率一致
                if attn.shape[1] == np.prod(self.attn_res):
                    # 将当前注意力矩阵存储到相应位置
                    self.step_store[place_in_unet].append(attn)
    
            # 更新当前注意力层索引
            self.cur_att_layer += 1
            # 如果达到最后一层，重置索引并调用间隔步骤方法
            if self.cur_att_layer == self.num_att_layers:
                self.cur_att_layer = 0
                self.between_steps()
    
        # 定义间隔步骤方法，用于更新注意力存储
        def between_steps(self):
            # 将步骤存储的注意力矩阵赋值给注意力存储
            self.attention_store = self.step_store
            # 获取一个空的步骤存储
            self.step_store = self.get_empty_store()
    
        # 获取平均注意力矩阵
        def get_average_attention(self):
            # 将注意力存储返回为平均注意力
            average_attention = self.attention_store
            return average_attention
    
        # 聚合来自不同层和头部的注意力矩阵
        def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
            """在指定的分辨率下聚合不同层和头部的注意力。"""
            out = []  # 初始化输出列表
            attention_maps = self.get_average_attention()  # 获取平均注意力
            # 遍历来源位置
            for location in from_where:
                # 遍历对应的注意力矩阵
                for item in attention_maps[location]:
                    # 重塑注意力矩阵为适当形状
                    cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
                    # 将重塑的矩阵添加到输出列表
                    out.append(cross_maps)
            # 沿第0维连接所有注意力矩阵
            out = torch.cat(out, dim=0)
            # 计算所有矩阵的平均值
            out = out.sum(0) / out.shape[0]
            return out  # 返回聚合后的注意力矩阵
    
        # 重置注意力存储和索引
        def reset(self):
            self.cur_att_layer = 0  # 重置当前注意力层索引
            self.step_store = self.get_empty_store()  # 重置步骤存储
            self.attention_store = {}  # 清空注意力存储
    
        # 初始化方法，设置初始参数
        def __init__(self, attn_res):
            """
            初始化一个空的 AttentionStore :param step_index: 用于可视化扩散过程中的特定步骤
            """
            self.num_att_layers = -1  # 初始化注意力层数量
            self.cur_att_layer = 0  # 初始化当前注意力层索引
            self.step_store = self.get_empty_store()  # 初始化步骤存储
            self.attention_store = {}  # 初始化注意力存储
            self.curr_step_index = 0  # 初始化当前步骤索引
            self.attn_res = attn_res  # 设置注意力分辨率
# 定义一个 AttendExciteAttnProcessor 类
class AttendExciteAttnProcessor:
    # 初始化方法，接受注意力存储器和 UNet 中的位置
    def __init__(self, attnstore, place_in_unet):
        # 调用父类的初始化方法
        super().__init__()
        # 存储传入的注意力存储器
        self.attnstore = attnstore
        # 存储在 UNet 中的位置
        self.place_in_unet = place_in_unet

    # 定义调用方法，处理注意力计算
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # 获取批次大小和序列长度
        batch_size, sequence_length, _ = hidden_states.shape
        # 准备注意力掩码
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # 将隐藏状态转换为查询向量
        query = attn.to_q(hidden_states)

        # 判断是否为交叉注意力
        is_cross = encoder_hidden_states is not None
        # 如果没有编码器隐藏状态，则使用隐藏状态
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # 将编码器隐藏状态转换为键向量
        key = attn.to_k(encoder_hidden_states)
        # 将编码器隐藏状态转换为值向量
        value = attn.to_v(encoder_hidden_states)

        # 将查询、键和值转换为批次维度
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 计算注意力分数
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # 仅在 Attend 和 Excite 过程中存储注意力图
        if attention_probs.requires_grad:
            # 存储注意力概率
            self.attnstore(attention_probs, is_cross, self.place_in_unet)

        # 使用注意力概率和值向量计算新的隐藏状态
        hidden_states = torch.bmm(attention_probs, value)
        # 将隐藏状态转换回头维度
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 进行线性变换
        hidden_states = attn.to_out[0](hidden_states)
        # 应用 dropout
        hidden_states = attn.to_out[1](hidden_states)

        # 返回更新后的隐藏状态
        return hidden_states


# 定义一个 StableDiffusionAttendAndExcitePipeline 类，继承多个基类
class StableDiffusionAttendAndExcitePipeline(DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin):
    r"""
    使用 Stable Diffusion 和 Attend-and-Excite 进行文本到图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档，以获取所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反演嵌入
    # 定义函数参数的说明文档，描述各参数的作用
    Args:
        vae ([`AutoencoderKL`]):
            # Variational Auto-Encoder (VAE) 模型，用于将图像编码和解码为潜在表示
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            # 冻结的文本编码器，使用 CLIP 模型进行文本特征提取
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            # 用于将文本进行分词的 CLIPTokenizer
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            # UNet 模型，用于对编码后的图像潜在特征进行去噪处理
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            # 调度器，用于与 UNet 结合去噪编码后的图像潜在特征，可以是 DDIM、LMS 或 PNDM 调度器
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            # 分类模块，用于评估生成的图像是否可能被认为是冒犯性或有害的
            Classification module that estimates whether generated images could be considered offensive or harmful.
            # 参见模型卡以获取有关模型潜在危害的更多详细信息
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            # CLIP 图像处理器，用于从生成的图像中提取特征；这些特征作为输入提供给安全检查器
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件列表
    _optional_components = ["safety_checker", "feature_extractor"]
    # 定义在 CPU 卸载时排除的组件
    _exclude_from_cpu_offload = ["safety_checker"]

    # 初始化方法定义，接收各个参数
    def __init__(
        # VAE 模型实例
        self,
        vae: AutoencoderKL,
        # 文本编码器实例
        text_encoder: CLIPTextModel,
        # 分词器实例
        tokenizer: CLIPTokenizer,
        # UNet 模型实例
        unet: UNet2DConditionModel,
        # 调度器实例
        scheduler: KarrasDiffusionSchedulers,
        # 安全检查器实例
        safety_checker: StableDiffusionSafetyChecker,
        # 特征提取器实例
        feature_extractor: CLIPImageProcessor,
        # 是否需要安全检查器的布尔标志，默认为 True
        requires_safety_checker: bool = True,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 检查安全检查器是否未定义且需要安全检查器时发出警告
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 检查安全检查器已定义但特征提取器未定义时引发错误
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 注册多个模块，包括 VAE、文本编码器、标记器等
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器，使用 VAE 缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 将配置中是否需要安全检查器进行注册
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制
    def _encode_prompt(
        # 定义编码提示所需的参数
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        # 定义弃用消息，告知用户 `_encode_prompt()` 方法将被移除，建议使用 `encode_prompt()`
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用 deprecate 函数记录弃用信息，设置标准警告为 False
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用 encode_prompt 方法生成提示嵌入元组，传入必要参数
        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # 连接嵌入元组中的两个部分以兼容以前的实现
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回最终的提示嵌入
        return prompt_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的 encode_prompt 方法
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的 run_safety_checker 方法
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，则设置 has_nsfw_concept 为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入为张量，进行后处理以转换为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入为 numpy 数组，转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 提取特征并将其移动到指定设备上
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 使用安全检查器处理图像，并返回处理后的图像和 NSFW 概念标识
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 NSFW 概念标识
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的 decode_latents 方法
    # 解码潜在变量
    def decode_latents(self, latents):
        # 定义弃用警告信息
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用弃用函数，发出警告
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
        # 按照缩放因子调整潜在变量
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在变量，返回第一个输出
        image = self.vae.decode(latents, return_dict=False)[0]
        # 归一化图像并限制在[0, 1]范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为float32格式，适应bfloat16并避免显著开销
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回最终图像
        return image
    
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备调度器步骤的额外参数，因为不是所有调度器具有相同的签名
        # eta (η) 仅用于DDIMScheduler，其他调度器将被忽略
        # eta 对应于DDIM论文中的η: https://arxiv.org/abs/2010.02502
        # 应在 [0, 1] 范围内
    
        # 检查调度器是否接受eta参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数字典
        extra_step_kwargs = {}
        # 如果接受eta，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器是否接受generator参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受generator，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回额外步骤参数字典
        return extra_step_kwargs
    
    # 检查输入参数
    def check_inputs(
        self,
        prompt,
        indices,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义形状以匹配潜在变量的尺寸
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器的数量是否与批大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果潜在变量为空，则生成随机张量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 将给定的潜在变量移动到指定设备
                latents = latents.to(device)
    
            # 按调度器要求的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回调整后的潜在变量
            return latents
    
        @staticmethod
    # 计算每个需要修改的 token 的最大注意力值
    def _compute_max_attention_per_index(
        # 输入的注意力图张量
        attention_maps: torch.Tensor,
        # 需要关注的 token 索引列表
        indices: List[int],
    ) -> List[torch.Tensor]:
        """计算我们希望改变的每个 token 的最大注意力值。"""
        # 获取注意力图中去掉首尾 token 的部分
        attention_for_text = attention_maps[:, :, 1:-1]
        # 将注意力值放大 100 倍
        attention_for_text *= 100
        # 对注意力值进行 softmax 处理，规范化
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # 因为去掉了第一个 token，调整索引
        indices = [index - 1 for index in indices]

        # 提取最大值的列表
        max_indices_list = []
        # 遍历每个索引
        for i in indices:
            # 获取指定索引的注意力图
            image = attention_for_text[:, :, i]
            # 创建高斯平滑对象并移动到相应设备
            smoothing = GaussianSmoothing().to(attention_maps.device)
            # 对图像进行填充以反射模式
            input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
            # 应用高斯平滑，并去掉多余的维度
            image = smoothing(input).squeeze(0).squeeze(0)
            # 将最大值添加到结果列表
            max_indices_list.append(image.max())
        # 返回最大值列表
        return max_indices_list

    # 聚合每个 token 的注意力并计算最大激活值
    def _aggregate_and_get_max_attention_per_token(
        self,
        # 需要关注的 token 索引列表
        indices: List[int],
    ):
        """聚合每个 token 的注意力，并计算每个 token 的最大激活值。"""
        # 从注意力存储中聚合注意力图
        attention_maps = self.attention_store.aggregate_attention(
            # 从不同来源获取的注意力图
            from_where=("up", "down", "mid"),
        )
        # 计算每个 token 的最大注意力值
        max_attention_per_index = self._compute_max_attention_per_index(
            # 传入注意力图和索引
            attention_maps=attention_maps,
            indices=indices,
        )
        # 返回最大注意力值
        return max_attention_per_index

    @staticmethod
    # 计算损失值
    def _compute_loss(max_attention_per_index: List[torch.Tensor]) -> torch.Tensor:
        """使用每个 token 的最大注意力值计算 attend-and-excite 损失。"""
        # 计算损失列表，确保不低于 0
        losses = [max(0, 1.0 - curr_max) for curr_max in max_attention_per_index]
        # 获取损失中的最大值
        loss = max(losses)
        # 返回最大损失
        return loss

    @staticmethod
    # 更新潜在变量
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """根据计算出的损失更新潜在变量。"""
        # 计算损失对潜在变量的梯度
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        # 更新潜在变量，使用学习率乘以梯度
        latents = latents - step_size * grad_cond
        # 返回更新后的潜在变量
        return latents

    # 进行迭代细化步骤
    def _perform_iterative_refinement_step(
        # 潜在变量张量
        latents: torch.Tensor,
        # 需要关注的 token 索引列表
        indices: List[int],
        # 当前损失值
        loss: torch.Tensor,
        # 阈值，用于判断
        threshold: float,
        # 文本嵌入张量
        text_embeddings: torch.Tensor,
        # 学习率
        step_size: float,
        # 当前迭代步数
        t: int,
        # 最大细化步骤，默认 20
        max_refinement_steps: int = 20,
    ):
        """
        执行论文中引入的迭代潜在优化。我们根据损失目标持续更新潜在代码，直到所有令牌达到给定的阈值。
        """
        # 初始化迭代计数器
        iteration = 0
        # 计算目标损失值，确保不小于 0
        target_loss = max(0, 1.0 - threshold)
        # 当当前损失大于目标损失时，持续迭代
        while loss > target_loss:
            # 迭代计数加一
            iteration += 1

            # 克隆潜在变量并准备计算梯度
            latents = latents.clone().detach().requires_grad_(True)
            # 使用 UNet 模型处理潜在变量，生成样本
            self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            # 清零 UNet 模型的梯度
            self.unet.zero_grad()

            # 获取每个主题令牌的最大激活值
            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                indices=indices,
            )

            # 计算当前的损失值
            loss = self._compute_loss(max_attention_per_index)

            # 如果损失不为零，更新潜在变量
            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            # 记录当前迭代和损失信息
            logger.info(f"\t Try {iteration}. loss: {loss}")

            # 如果达到最大迭代步数，记录并退出循环
            if iteration >= max_refinement_steps:
                logger.info(f"\t Exceeded max number of iterations ({max_refinement_steps})! ")
                break

        # 再次运行但不计算梯度，也不更新潜在变量，仅计算新损失
        latents = latents.clone().detach().requires_grad_(True)
        _ = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
        self.unet.zero_grad()

        # 获取每个主题令牌的最大激活值
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            indices=indices,
        )
        # 计算当前损失值
        loss = self._compute_loss(max_attention_per_index)
        # 记录最终损失信息
        logger.info(f"\t Finished with loss of: {loss}")
        # 返回损失、潜在变量和最大激活值索引
        return loss, latents, max_attention_per_index

    def register_attention_control(self):
        # 初始化注意力处理器字典
        attn_procs = {}
        # 交叉注意力计数
        cross_att_count = 0
        # 遍历 UNet 中的注意力处理器
        for name in self.unet.attn_processors.keys():
            # 根据名称确定位置
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            # 交叉注意力计数加一
            cross_att_count += 1
            # 创建注意力处理器并添加到字典
            attn_procs[name] = AttendExciteAttnProcessor(attnstore=self.attention_store, place_in_unet=place_in_unet)

        # 设置 UNet 的注意力处理器
        self.unet.set_attn_processor(attn_procs)
        # 更新注意力层的数量
        self.attention_store.num_att_layers = cross_att_count

    def get_indices(self, prompt: str) -> Dict[str, int]:
        """用于列出要更改的令牌的索引的实用函数"""
        # 将提示转换为输入 ID
        ids = self.tokenizer(prompt).input_ids
        # 创建令牌到索引的映射字典
        indices = {i: tok for tok, i in zip(self.tokenizer.convert_ids_to_tokens(ids), range(len(ids)))}
        # 返回索引字典
        return indices

    @torch.no_grad()
    # 替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用的方法，接受多个参数以生成图像
        def __call__(
            # 提示信息，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]],
            # 令牌索引，可以是整数列表或列表的列表
            token_indices: Union[List[int], List[List[int]]],
            # 可选的图像高度，默认为 None
            height: Optional[int] = None,
            # 可选的图像宽度，默认为 None
            width: Optional[int] = None,
            # 生成的推理步骤数，默认为 50
            num_inference_steps: int = 50,
            # 指导比例，默认为 7.5
            guidance_scale: float = 7.5,
            # 可选的负提示，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: int = 1,
            # ETA 值，默认为 0.0
            eta: float = 0.0,
            # 可选的随机数生成器，可以是单个生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜在变量张量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 可选的提示嵌入张量，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负提示嵌入张量，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典形式的结果，默认为 True
            return_dict: bool = True,
            # 可选的回调函数，用于处理生成过程中的状态
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调函数调用的步骤间隔，默认为 1
            callback_steps: int = 1,
            # 可选的交叉注意力参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 最大迭代次数，默认为 25
            max_iter_to_alter: int = 25,
            # 阈值字典，定义不同步长下的阈值
            thresholds: dict = {0: 0.05, 10: 0.5, 20: 0.8},
            # 缩放因子，默认为 20
            scale_factor: int = 20,
            # 可选的注意力分辨率元组，默认为 (16, 16)
            attn_res: Optional[Tuple[int]] = (16, 16),
            # 可选的跳过剪辑次数，默认为 None
            clip_skip: Optional[int] = None,
# 定义一个继承自 PyTorch 模块的高斯平滑类
class GaussianSmoothing(torch.nn.Module):
    """
    参数：
    对 1D、2D 或 3D 张量应用高斯平滑。每个通道分别使用深度卷积进行过滤。
        channels (int, sequence): 输入张量的通道数。输出将具有相同数量的通道。
        kernel_size (int, sequence): 高斯核的大小。 sigma (float, sequence): 高斯核的标准差。
        dim (int, optional): 数据的维度数量。默认值为 2（空间）。
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    # 初始化方法，设置高斯平滑的参数
    def __init__(
        self,
        channels: int = 1,  # 输入通道数，默认为1
        kernel_size: int = 3,  # 高斯核的大小，默认为3
        sigma: float = 0.5,  # 高斯核的标准差，默认为0.5
        dim: int = 2,  # 数据维度，默认为2
    ):
        super().__init__()  # 调用父类的初始化方法

        # 如果 kernel_size 是一个整数，则将其转换为对应维度的列表
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        # 如果 sigma 是一个浮点数，则将其转换为对应维度的列表
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # 高斯核是每个维度高斯函数的乘积
        kernel = 1  # 初始化高斯核为1
        # 创建高斯核的网格，生成每个维度的网格
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        # 遍历每个维度的大小、标准差和网格
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2  # 计算高斯分布的均值
            # 更新高斯核，计算当前维度的高斯值并与核相乘
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # 确保高斯核的值之和等于1
        kernel = kernel / torch.sum(kernel)

        # 将高斯核重塑为深度卷积权重的形状
        kernel = kernel.view(1, 1, *kernel.size())  # 重塑为卷积所需的格式
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))  # 复制以适应输入通道数

        # 注册高斯核作为模块的缓冲区
        self.register_buffer("weight", kernel)
        self.groups = channels  # 设置分组卷积的通道数

        # 根据维度选择相应的卷积操作
        if dim == 1:
            self.conv = F.conv1d  # 1D 卷积
        elif dim == 2:
            self.conv = F.conv2d  # 2D 卷积
        elif dim == 3:
            self.conv = F.conv3d  # 3D 卷积
        else:
            # 如果维度不支持，则抛出运行时错误
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    # 前向传播方法，应用高斯滤波
    def forward(self, input):
        """
        参数：
        对输入应用高斯滤波。
            input (torch.Tensor): 需要应用高斯滤波的输入。
        返回：
            filtered (torch.Tensor): 滤波后的输出。
        """
        # 使用选择的卷积方法对输入进行卷积，返回滤波结果
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)
```