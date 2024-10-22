# `.\diffusers\pipelines\stable_diffusion\pipeline_flax_stable_diffusion.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）进行授权；
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有约定，
# 根据许可证分发的软件均按“原样”提供，
# 不提供任何形式的明示或暗示的保证或条件。
# 请参见许可证以获取管理权限的特定语言和
# 限制条件。

import warnings  # 导入警告模块，用于显示警告信息
from functools import partial  # 从functools导入partial，用于部分函数应用
from typing import Dict, List, Optional, Union  # 导入类型提示相关的类

import jax  # 导入jax库，用于高效的数值计算
import jax.numpy as jnp  # 导入jax的numpy模块
import numpy as np  # 导入numpy库
from flax.core.frozen_dict import FrozenDict  # 从flax导入FrozenDict，用于不可变字典
from flax.jax_utils import unreplicate  # 导入unreplicate，用于将数据从多个设备上收集回单个设备
from flax.training.common_utils import shard  # 从flax导入shard，用于数据分片
from packaging import version  # 导入version，用于处理版本号
from PIL import Image  # 从PIL导入Image类，用于图像处理
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel  # 导入Transformers库中与CLIP相关的类

from ...models import FlaxAutoencoderKL, FlaxUNet2DConditionModel  # 从相对路径导入FlaxAutoencoderKL和FlaxUNet2DConditionModel
from ...schedulers import (  # 从相对路径导入多个调度器
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
from ...utils import deprecate, logging, replace_example_docstring  # 从相对路径导入实用工具函数
from ..pipeline_flax_utils import FlaxDiffusionPipeline  # 从上级模块导入FlaxDiffusionPipeline
from .pipeline_output import FlaxStableDiffusionPipelineOutput  # 从当前模块导入FlaxStableDiffusionPipelineOutput
from .safety_checker_flax import FlaxStableDiffusionSafetyChecker  # 从当前模块导入FlaxStableDiffusionSafetyChecker

logger = logging.get_logger(__name__)  # 创建一个记录器实例，用于记录模块的日志信息

# 设置为True时使用Python循环而不是jax.fori_loop，以便更容易调试
DEBUG = False  # 定义DEBUG常量，初始值为False

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串，包含使用示例
    Examples:
        ```py
        >>> import jax  # 导入jax库
        >>> import numpy as np  # 导入numpy库
        >>> from flax.jax_utils import replicate  # 从flax导入replicate，用于数据复制
        >>> from flax.training.common_utils import shard  # 从flax导入shard，用于数据分片

        >>> from diffusers import FlaxStableDiffusionPipeline  # 从diffusers库导入FlaxStableDiffusionPipeline

        >>> pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(  # 从预训练模型加载管道和参数
        ...     "runwayml/stable-diffusion-v1-5", variant="bf16", dtype=jax.numpy.bfloat16
        ... )

        >>> prompt = "a photo of an astronaut riding a horse on mars"  # 定义生成图像的提示文本

        >>> prng_seed = jax.random.PRNGKey(0)  # 创建一个随机数生成器的种子
        >>> num_inference_steps = 50  # 定义推理的步数

        >>> num_samples = jax.device_count()  # 获取可用设备的数量
        >>> prompt = num_samples * [prompt]  # 为每个设备创建相同的提示文本列表
        >>> prompt_ids = pipeline.prepare_inputs(prompt)  # 准备输入的提示文本ID
        # shard inputs and rng  # 注释，说明将输入和随机数分片

        >>> params = replicate(params)  # 复制参数到所有设备
        >>> prng_seed = jax.random.split(prng_seed, jax.device_count())  # 将随机种子分割到每个设备
        >>> prompt_ids = shard(prompt_ids)  # 将提示文本ID分片到每个设备

        >>> images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images  # 生成图像
        >>> images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))  # 将生成的图像转换为PIL格式
        ```py
"""  # 示例文档字符串的结束

class FlaxStableDiffusionPipeline(FlaxDiffusionPipeline):  # 定义FlaxStableDiffusionPipeline类，继承自FlaxDiffusionPipeline
    r"""  # 定义类文档字符串
    Flax-based pipeline for text-to-image generation using Stable Diffusion.  # 描述此类的功能
    # 该模型继承自 [`FlaxDiffusionPipeline`]。请查看父类文档以获取所有管道实现的通用方法（如下载、保存、在特定设备上运行等）。
    
    # 参数说明：
    # vae ([`FlaxAutoencoderKL`]):
    #    变分自编码器（VAE）模型，用于将图像编码和解码为潜在表示。
    # text_encoder ([`~transformers.FlaxCLIPTextModel`]):
    #    冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
    # tokenizer ([`~transformers.CLIPTokenizer`]):
    #    用于文本分词的 `CLIPTokenizer`。
    # unet ([`FlaxUNet2DConditionModel`]):
    #    用于对编码图像潜在值进行去噪的 `FlaxUNet2DConditionModel`。
    # scheduler ([`SchedulerMixin`]):
    #    用于与 `unet` 结合使用以去噪编码图像潜在值的调度器。可以是
    #    [`FlaxDDIMScheduler`], [`FlaxLMSDiscreteScheduler`], [`FlaxPNDMScheduler`] 或
    #    [`FlaxDPMSolverMultistepScheduler`] 之一。
    # safety_checker ([`FlaxStableDiffusionSafetyChecker`]):
    #    分类模块，估计生成的图像是否可能被认为是冒犯或有害的。
    #    有关模型潜在危害的更多详细信息，请参阅 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
    # feature_extractor ([`~transformers.CLIPImageProcessor`]):
    #    从生成图像中提取特征的 `CLIPImageProcessor`；用作 `safety_checker` 的输入。
    # dtype: jnp.dtype = jnp.float32,  # 默认数据类型为 jnp.float32

    def __init__(  # 初始化方法，用于创建类的实例
        self,  # 实例自身
        vae: FlaxAutoencoderKL,  # 变分自编码器实例
        text_encoder: FlaxCLIPTextModel,  # 文本编码器实例
        tokenizer: CLIPTokenizer,  # 分词器实例
        unet: FlaxUNet2DConditionModel,  # UNet去噪模型实例
        scheduler: Union[  # 可选调度器，支持多种类型
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],
        safety_checker: FlaxStableDiffusionSafetyChecker,  # 安全检查器实例
        feature_extractor: CLIPImageProcessor,  # 特征提取器实例
        dtype: jnp.dtype = jnp.float32,  # 数据类型参数，默认值为 jnp.float32
    # 初始化方法，调用父类构造函数
        ):
            super().__init__()
            # 设置数据类型
            self.dtype = dtype
    
            # 检查安全检查器是否为 None
            if safety_checker is None:
                # 记录警告信息，提醒用户安全检查器已禁用
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 检查 UNet 版本是否低于 0.9.0
            is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
                version.parse(unet.config._diffusers_version).base_version
            ) < version.parse("0.9.0.dev0")
            # 检查 UNet 采样大小是否小于 64
            is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
            # 如果版本和采样大小都不符合要求，给出弃用警告
            if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
                # 创建弃用信息，提示用户修改配置文件
                deprecation_message = (
                    "The configuration file of the unet has set the default `sample_size` to smaller than"
                    " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                    " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                    " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                    " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                    " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                    " in the config might lead to incorrect results in future versions. If you have downloaded this"
                    " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                    " the `unet/config.json` file"
                )
                # 调用弃用函数，记录弃用警告
                deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
                # 创建新的配置字典并将采样大小设为 64
                new_config = dict(unet.config)
                new_config["sample_size"] = 64
                # 更新 UNet 内部字典
                unet._internal_dict = FrozenDict(new_config)
    
            # 注册模块，将各个组件关联起来
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
    # 准备输入，接受字符串或字符串列表
    def prepare_inputs(self, prompt: Union[str, List[str]]):
        # 检查 prompt 是否为字符串或列表类型，若不是则抛出异常
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
        # 使用分词器处理 prompt，生成带有填充和截断的张量
        text_input = self.tokenizer(
            prompt,
            padding="max_length",  # 填充至最大长度
            max_length=self.tokenizer.model_max_length,  # 使用模型最大长度
            truncation=True,  # 启用截断
            return_tensors="np",  # 返回 NumPy 格式的张量
        )
        # 返回处理后的输入 ID
        return text_input.input_ids
    
    # 获取是否存在 NSFW 概念
    def _get_has_nsfw_concepts(self, features, params):
        # 使用安全检查器处理特征和参数，返回是否存在 NSFW 概念
        has_nsfw_concepts = self.safety_checker(features, params)
        # 返回检测结果
        return has_nsfw_concepts
    
    # 运行安全检查器
    def _run_safety_checker(self, images, safety_model_params, jit=False):
        # 将输入的图像数组转换为 PIL 图像
        pil_images = [Image.fromarray(image) for image in images]
        # 提取特征，返回 NumPy 格式的张量
        features = self.feature_extractor(pil_images, return_tensors="np").pixel_values
    
        # 如果启用 JIT，则对特征进行分片处理
        if jit:
            features = shard(features)  # 分片特征
            has_nsfw_concepts = _p_get_has_nsfw_concepts(self, features, safety_model_params)  # 检查 NSFW 概念
            has_nsfw_concepts = unshard(has_nsfw_concepts)  # 合并分片结果
            safety_model_params = unreplicate(safety_model_params)  # 取消复制安全模型参数
        else:
            # 直接获取 NSFW 概念
            has_nsfw_concepts = self._get_has_nsfw_concepts(features, safety_model_params)
    
        images_was_copied = False  # 标记图像是否已复制
        # 遍历每个图像的 NSFW 概念检测结果
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:  # 如果检测到 NSFW 概念
                if not images_was_copied:  # 如果图像还没有复制
                    images_was_copied = True  # 标记为已复制
                    images = images.copy()  # 复制图像数组
    
                images[idx] = np.zeros(images[idx].shape, dtype=np.uint8)  # 替换为黑色图像
    
            # 如果有任何 NSFW 概念
            if any(has_nsfw_concepts):
                # 发出警告，提示可能检测到不适合的内容
                warnings.warn(
                    "Potential NSFW content was detected in one or more images. A black image will be returned"
                    " instead. Try again with a different prompt and/or seed."
                )
    
        # 返回处理后的图像和 NSFW 概念检测结果
        return images, has_nsfw_concepts
    
    # 生成图像的主函数
    def _generate(
        self,
        prompt_ids: jnp.array,  # 输入的提示 ID
        params: Union[Dict, FrozenDict],  # 模型参数
        prng_seed: jax.Array,  # 随机种子
        num_inference_steps: int,  # 推理步骤数
        height: int,  # 生成图像的高度
        width: int,  # 生成图像的宽度
        guidance_scale: float,  # 引导尺度
        latents: Optional[jnp.ndarray] = None,  # 可选的潜在变量
        neg_prompt_ids: Optional[jnp.ndarray] = None,  # 可选的负提示 ID
        @replace_example_docstring(EXAMPLE_DOC_STRING)  # 替换示例文档字符串的装饰器
        def __call__(  # 定义调用方法
            self,
            prompt_ids: jnp.array,  # 输入的提示 ID
            params: Union[Dict, FrozenDict],  # 模型参数
            prng_seed: jax.Array,  # 随机种子
            num_inference_steps: int = 50,  # 推理步骤数，默认为 50
            height: Optional[int] = None,  # 可选的图像高度
            width: Optional[int] = None,  # 可选的图像宽度
            guidance_scale: Union[float, jnp.ndarray] = 7.5,  # 引导尺度，默认为 7.5
            latents: jnp.ndarray = None,  # 可选的潜在变量
            neg_prompt_ids: jnp.ndarray = None,  # 可选的负提示 ID
            return_dict: bool = True,  # 是否返回字典格式的结果，默认为 True
            jit: bool = False,  # 是否启用 JIT 编译
# 静态参数包括管道、推理步数、高度和宽度。任何更改都会触发重新编译。
# 非静态参数是映射在其第一维上的（分片）输入张量（因此为 `0`）。
@partial(
    jax.pmap,  # 应用并行映射以支持多设备计算
    in_axes=(None, 0, 0, 0, None, None, None, 0, 0, 0),  # 指定输入张量的维度映射
    static_broadcasted_argnums=(0, 4, 5, 6),  # 静态广播参数的索引
)
def _p_generate(
    pipe,  # 生成管道
    prompt_ids,  # 提示的 ID 列表
    params,  # 模型参数
    prng_seed,  # 随机数生成种子
    num_inference_steps,  # 推理步骤的数量
    height,  # 输出图像的高度
    width,  # 输出图像的宽度
    guidance_scale,  # 引导尺度，用于控制生成的效果
    latents,  # 潜在向量
    neg_prompt_ids,  # 负提示的 ID 列表
):
    # 调用生成管道的方法以生成输出
    return pipe._generate(
        prompt_ids,  # 提示 ID
        params,  # 模型参数
        prng_seed,  # 随机种子
        num_inference_steps,  # 推理步骤
        height,  # 高度
        width,  # 宽度
        guidance_scale,  # 引导尺度
        latents,  # 潜在向量
        neg_prompt_ids,  # 负提示 ID
    )


@partial(jax.pmap, static_broadcasted_argnums=(0,))  # 应用并行映射，静态广播第一个参数
def _p_get_has_nsfw_concepts(pipe, features, params):
    # 调用管道方法以检查是否有不适宜内容概念
    return pipe._get_has_nsfw_concepts(features, params)


def unshard(x: jnp.ndarray):
    # 使用 einops 对输入进行重排，将设备和批次维度合并
    num_devices, batch_size = x.shape[:2]  # 获取设备数和批次大小
    rest = x.shape[2:]  # 获取剩余维度
    # 重塑张量，使得设备和批次维度合并
    return x.reshape(num_devices * batch_size, *rest)
```