# `.\diffusers\pipelines\lumina\pipeline_lumina.py`

```py
# 版权所有 2024 Alpha-VLLM 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）进行许可；
# 您只能在遵守许可证的情况下使用此文件。
# 您可以在以下地址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，分发的软件均按“原样”提供，
# 不附带任何形式的明示或暗示的担保或条件。
# 有关许可证的具体权限和限制，请参见许可证。

import html  # 导入处理 HTML 实体的模块
import inspect  # 导入用于获取对象信息的模块
import math  # 导入数学函数模块
import re  # 导入正则表达式模块
import urllib.parse as ul  # 导入用于 URL 解析的模块，并将其重命名为 ul
from typing import List, Optional, Tuple, Union  # 导入类型注解模块中的相关类型

import torch  # 导入 PyTorch 库
from transformers import AutoModel, AutoTokenizer  # 从 transformers 导入自动模型和自动分词器

from ...image_processor import VaeImageProcessor  # 从上级模块导入 VAE 图像处理器
from ...models import AutoencoderKL  # 从上级模块导入自动编码器模型
from ...models.embeddings import get_2d_rotary_pos_embed_lumina  # 从上级模块导入获取 2D 旋转位置嵌入的函数
from ...models.transformers.lumina_nextdit2d import LuminaNextDiT2DModel  # 从上级模块导入 LuminaNextDiT2D 模型
from ...schedulers import FlowMatchEulerDiscreteScheduler  # 从上级模块导入调度器
from ...utils import (  # 从上级模块导入多个实用工具
    BACKENDS_MAPPING,  # 后端映射
    is_bs4_available,  # 检查 bs4 库是否可用的函数
    is_ftfy_available,  # 检查 ftfy 库是否可用的函数
    logging,  # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的函数
)
from ...utils.torch_utils import randn_tensor  # 从上级模块导入生成随机张量的函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从上级模块导入扩散管道和图像输出类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

if is_bs4_available():  # 如果 bs4 库可用
    from bs4 import BeautifulSoup  # 导入 BeautifulSoup 库用于解析 HTML

if is_ftfy_available():  # 如果 ftfy 库可用
    import ftfy  # 导入 ftfy 库用于文本修复

EXAMPLE_DOC_STRING = """  # 示例文档字符串，用于展示用法
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import LuminaText2ImgPipeline  # 从 diffusers 导入 LuminaText2ImgPipeline

        >>> pipe = LuminaText2ImgPipeline.from_pretrained(  # 从预训练模型加载管道
        ...     "Alpha-VLLM/Lumina-Next-SFT-diffusers", torch_dtype=torch.bfloat16  # 指定模型路径和数据类型
        ... ).cuda()  # 将管道移到 GPU
        >>> # 启用内存优化。
        >>> pipe.enable_model_cpu_offload()  # 启用模型的 CPU 卸载以节省内存

        >>> prompt = "Upper body of a young woman in a Victorian-era outfit with brass goggles and leather straps. Background shows an industrial revolution cityscape with smoky skies and tall, metal structures"  # 定义生成图像的提示
        >>> image = pipe(prompt).images[0]  # 生成图像并提取第一张图像
        ```py
"""

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 导入的函数
def retrieve_timesteps(  # 定义函数以检索时间步
    scheduler,  # 调度器对象
    num_inference_steps: Optional[int] = None,  # 可选推理步骤数
    device: Optional[Union[str, torch.device]] = None,  # 可选设备类型
    timesteps: Optional[List[int]] = None,  # 可选时间步列表
    sigmas: Optional[List[float]] = None,  # 可选 sigma 值列表
    **kwargs,  # 接收其他关键字参数
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器检索时间步。处理自定义时间步。
    任何关键字参数将传递给 `scheduler.set_timesteps`。
    # 参数说明
        Args:
            scheduler (`SchedulerMixin`):  # 定义调度器类型，用于获取时间步长
                The scheduler to get timesteps from.  # 调度器的描述
            num_inference_steps (`int`):  # 定义推理步骤数量
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`  # 生成样本时的扩散步骤数，如果使用，`timesteps` 必须为 `None`
                must be `None`.  # 说明条件
            device (`str` or `torch.device`, *optional*):  # 定义设备类型（字符串或torch.device），可选
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.  # 指定将时间步移动到的设备，如果为 `None`，则不移动
            timesteps (`List[int]`, *optional*):  # 自定义时间步，列表类型，可选
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,  # 自定义时间步，覆盖调度器的时间步策略，如果传递了 `timesteps`
                `num_inference_steps` and `sigmas` must be `None`.  # 则 `num_inference_steps` 和 `sigmas` 必须为 `None`
            sigmas (`List[float]`, *optional*):  # 自定义sigmas，列表类型，可选
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,  # 自定义sigmas，覆盖调度器的时间步策略，如果传递了 `sigmas`
                `num_inference_steps` and `timesteps` must be `None`.  # 则 `num_inference_steps` 和 `timesteps` 必须为 `None`
    
        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the  # 返回类型为元组，包含调度器的时间步计划和推理步骤数量
            second element is the number of inference steps.  # 返回第二个元素为推理步骤数量
        """
        if timesteps is not None and sigmas is not None:  # 检查是否同时传入了时间步和sigmas
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")  # 抛出错误，要求只能选择一个
        if timesteps is not None:  # 如果传入了时间步
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())  # 检查调度器是否接受时间步
            if not accepts_timesteps:  # 如果不接受
                raise ValueError(  # 抛出错误，说明当前调度器不支持自定义时间步
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"  # 当前调度器不支持自定义时间步
                    f" timestep schedules. Please check whether you are using the correct scheduler."  # 检查是否使用了正确的调度器
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)  # 设置时间步，传入设备和其他参数
            timesteps = scheduler.timesteps  # 从调度器获取设置后的时间步
            num_inference_steps = len(timesteps)  # 计算推理步骤数量
        elif sigmas is not None:  # 如果传入了sigmas
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())  # 检查调度器是否接受sigmas
            if not accept_sigmas:  # 如果不接受
                raise ValueError(  # 抛出错误，说明当前调度器不支持自定义sigmas
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"  # 当前调度器不支持自定义sigmas
                    f" sigmas schedules. Please check whether you are using the correct scheduler."  # 检查是否使用了正确的调度器
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)  # 设置sigmas，传入设备和其他参数
            timesteps = scheduler.timesteps  # 从调度器获取设置后的时间步
            num_inference_steps = len(timesteps)  # 计算推理步骤数量
        else:  # 如果没有传入时间步和sigmas
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)  # 直接使用推理步骤设置时间步，传入设备和其他参数
            timesteps = scheduler.timesteps  # 从调度器获取设置后的时间步
        return timesteps, num_inference_steps  # 返回时间步和推理步骤数量
# 定义 LuminaText2ImgPipeline 类，继承自 DiffusionPipeline
class LuminaText2ImgPipeline(DiffusionPipeline):
    r"""
    Lumina-T2I 的文本到图像生成管道。

    此模型继承自 [`DiffusionPipeline`]。请查看父类文档以了解库为所有管道实现的通用方法（例如下载或保存、在特定设备上运行等）。

    参数：
        vae ([`AutoencoderKL`]):
            用于编码和解码图像到潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`AutoModel`]):
            冻结的文本编码器。Lumina-T2I 使用
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.AutoModel)，具体是
            [t5-v1_1-xxl](https://huggingface.co/Alpha-VLLM/tree/main/t5-v1_1-xxl) 变体。
        tokenizer (`AutoModel`):
            [AutoModel](https://huggingface.co/docs/transformers/model_doc/t5#transformers.AutoModel) 类的分词器。
        transformer ([`Transformer2DModel`]):
            一种文本条件的 `Transformer2DModel`，用于去噪编码的图像潜在表示。
        scheduler ([`SchedulerMixin`]):
            用于与 `transformer` 结合使用的调度器，以去噪编码的图像潜在表示。
    """

    # 编译一个正则表达式，用于匹配不良标点
    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    # 定义可选组件的空列表
    _optional_components = []
    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    # 初始化方法，接受多个模型组件作为参数
    def __init__(
        self,
        transformer: LuminaNextDiT2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: AutoModel,
        tokenizer: AutoTokenizer,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册各个模块，便于后续使用
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        # 设置 VAE 的缩放因子
        self.vae_scale_factor = 8
        # 创建图像处理器，用于处理 VAE 输出
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 设置最大序列长度
        self.max_sequence_length = 256
        # 设置默认样本大小，根据 transformer 的配置或设置为 128
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        # 计算默认图像大小
        self.default_image_size = self.default_sample_size * self.vae_scale_factor

    # 定义获取 gemma 提示嵌入的方法，接受多个参数
    def _get_gemma_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clean_caption: Optional[bool] = False,
        max_length: Optional[int] = None,
    # 处理输入的设备，使用给定设备或默认执行设备
        ):
            device = device or self._execution_device
            # 将字符串类型的提示转为列表形式
            prompt = [prompt] if isinstance(prompt, str) else prompt
            # 获取提示的批次大小
            batch_size = len(prompt)
    
            # 对提示文本进行预处理，选项可清理标题
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            # 使用 tokenizer 处理提示文本，设置填充、最大长度等参数
            text_inputs = self.tokenizer(
                prompt,
                pad_to_multiple_of=8,
                max_length=self.max_sequence_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            # 将输入 ID 移动到指定设备
            text_input_ids = text_inputs.input_ids.to(device)
            # 获取未截断的输入 ID，使用最长填充
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(device)
    
            # 检查是否存在截断，且文本输入 ID 与未截断 ID 不同
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                # 解码被截断的文本，并记录警告信息
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.max_sequence_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because Gemma can only handle sequences up to"
                    f" {self.max_sequence_length} tokens: {removed_text}"
                )
    
            # 将提示的注意力掩码移动到指定设备
            prompt_attention_mask = text_inputs.attention_mask.to(device)
            # 使用文本编码器获取提示的嵌入，输出隐藏状态
            prompt_embeds = self.text_encoder(
                text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
            )
            # 获取倒数第二层的隐藏状态作为提示嵌入
            prompt_embeds = prompt_embeds.hidden_states[-2]
    
            # 确定数据类型，优先使用文本编码器的数据类型
            if self.text_encoder is not None:
                dtype = self.text_encoder.dtype
            elif self.transformer is not None:
                dtype = self.transformer.dtype
            else:
                dtype = None
    
            # 将嵌入移动到指定的数据类型和设备
            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
            # 解包嵌入的形状以获取序列长度
            _, seq_len, _ = prompt_embeds.shape
            # 为每个提示生成多个图像，重复嵌入和注意力掩码
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            # 重新调整嵌入的形状
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            # 重复注意力掩码以适应生成的图像数量
            prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
            # 重新调整注意力掩码的形状
            prompt_attention_mask = prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)
    
            # 返回嵌入和注意力掩码
            return prompt_embeds, prompt_attention_mask
    
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt 适配
        def encode_prompt(
            # 定义编码提示的函数参数，包括提示和其他可选参数
            prompt: Union[str, List[str]],
            do_classifier_free_guidance: bool = True,
            negative_prompt: Union[str, List[str]] = None,
            num_images_per_prompt: int = 1,
            device: Optional[torch.device] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            prompt_attention_mask: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            clean_caption: bool = False,
            **kwargs,
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    # 定义一个方法，准备额外的参数以供调度器步骤使用
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，因为不同调度器的签名可能不同
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略它
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # eta 的值应在 [0, 1] 之间

        # 检查调度器的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化一个空的字典以存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果调度器接受 eta 参数，将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator 参数，将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs

    # 定义一个方法，用于检查输入的有效性
    def check_inputs(
        self,
        prompt,  # 输入的提示文本
        height,  # 图像的高度
        width,   # 图像的宽度
        negative_prompt,  # 输入的负面提示文本
        prompt_embeds=None,  # 可选的提示嵌入
        negative_prompt_embeds=None,  # 可选的负面提示嵌入
        prompt_attention_mask=None,  # 可选的提示注意力掩码
        negative_prompt_attention_mask=None,  # 可选的负面提示注意力掩码
    ):
        # 检查高度和宽度是否是8的倍数，若不是，则抛出值错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查是否同时提供了 `prompt` 和 `prompt_embeds`，若是，则抛出值错误
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否都没有提供 `prompt` 和 `prompt_embeds`，若是，则抛出值错误
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 `prompt` 是否为字符串或列表，若不是，则抛出值错误
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了 `prompt` 和 `negative_prompt_embeds`，若是，则抛出值错误
        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查是否同时提供了 `negative_prompt` 和 `negative_prompt_embeds`，若是，则抛出值错误
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查如果提供了 `prompt_embeds`，则必须提供 `prompt_attention_mask`，若没有，则抛出值错误
        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        # 检查如果提供了 `negative_prompt_embeds`，则必须提供 `negative_prompt_attention_mask`，若没有，则抛出值错误
        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        # 检查 `prompt_embeds` 和 `negative_prompt_embeds` 的形状是否一致，若不一致，则抛出值错误
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            # 检查 `prompt_attention_mask` 和 `negative_prompt_attention_mask` 的形状是否一致，若不一致，则抛出值错误
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing 复制的内容
    # 定义文本预处理的私有方法，接受文本和一个可选的清理标题参数
        def _text_preprocessing(self, text, clean_caption=False):
            # 如果设置了清理标题且 bs4 库不可用，记录警告并将清理标题设为 False
            if clean_caption and not is_bs4_available():
                logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
                logger.warning("Setting `clean_caption` to False...")
                clean_caption = False
    
            # 如果设置了清理标题且 ftfy 库不可用，记录警告并将清理标题设为 False
            if clean_caption and not is_ftfy_available():
                logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
                logger.warning("Setting `clean_caption` to False...")
                clean_caption = False
    
            # 如果输入文本不是元组或列表，则将其转换为列表
            if not isinstance(text, (tuple, list)):
                text = [text]
    
            # 定义内部处理文本的函数
            def process(text: str):
                # 如果设置了清理标题，则进行标题清理两次
                if clean_caption:
                    text = self._clean_caption(text)
                    text = self._clean_caption(text)
                else:
                    # 否则将文本转换为小写并去除空格
                    text = text.lower().strip()
                return text
    
            # 对文本列表中的每个元素进行处理，并返回处理后的结果列表
            return [process(t) for t in text]
    
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption 复制而来
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在变量的形状，根据输入参数计算
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 如果生成器是列表且长度与批量大小不匹配，则引发错误
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果潜在变量为空，则生成随机张量作为潜在变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果潜在变量已给出，则将其转移到指定设备
                latents = latents.to(device)
    
            # 返回生成的或已转换的潜在变量
            return latents
    
        # 返回指导比例的属性值
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 此处的指导比例与 Imagen 论文中方程 (2) 的指导权重 `w` 相似： https://arxiv.org/pdf/2205.11487.pdf 
        # 指导比例 = 1 表示没有进行无分类器的指导
        @property
        def do_classifier_free_guidance(self):
            return self._guidance_scale > 1
    
        # 返回时间步数的属性值
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 关闭梯度计算以节省内存
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的方法
        def __call__(
            # 输入的提示，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 输出图像的宽度
            width: Optional[int] = None,
            # 输出图像的高度
            height: Optional[int] = None,
            # 推理步骤的数量，默认为30
            num_inference_steps: int = 30,
            # 定义时间步的列表，默认为None
            timesteps: List[int] = None,
            # 引导尺度，默认为4.0
            guidance_scale: float = 4.0,
            # 负提示，可以是字符串或字符串列表
            negative_prompt: Union[str, List[str]] = None,
            # sigma值的列表，默认为None
            sigmas: List[float] = None,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # 随机数生成器，可以是单个或多个torch生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，默认为None
            latents: Optional[torch.Tensor] = None,
            # 提示的嵌入向量，默认为None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示的嵌入向量，默认为None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 提示的注意力掩码，默认为None
            prompt_attention_mask: Optional[torch.Tensor] = None,
            # 负提示的注意力掩码，默认为None
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            # 输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式，默认为True
            return_dict: bool = True,
            # 是否清理提示文本，默认为True
            clean_caption: bool = True,
            # 最大序列长度，默认为256
            max_sequence_length: int = 256,
            # 缩放阈值，默认为1.0
            scaling_watershed: Optional[float] = 1.0,
            # 是否使用比例注意力，默认为True
            proportional_attn: Optional[bool] = True,
```