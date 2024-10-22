# `.\diffusers\pipelines\stable_diffusion\pipeline_flax_stable_diffusion_inpaint.py`

```py
# 版权声明，指明该文件的版权信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 按照 Apache 许可证第 2.0 版许可使用本文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 除非遵循许可证，否则不得使用本文件
# you may not use this file except in compliance with the License.
# 可以通过以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议规定，否则根据许可证分发的软件
# Unless required by applicable law or agreed to in writing, software
# 是按“原样”基础分发的，不提供任何形式的担保或条件
# distributed under the License is distributed on an "AS IS" BASIS,
# 不论是明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 参见许可证以了解适用权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings  # 导入 warnings 模块以处理警告
from functools import partial  # 从 functools 导入 partial，用于部分应用函数
from typing import Dict, List, Optional, Union  # 导入类型注解工具

import jax  # 导入 jax 库，用于高性能数值计算
import jax.numpy as jnp  # 导入 jax 的 numpy 作为 jnp
import numpy as np  # 导入 numpy 库以进行数组操作
from flax.core.frozen_dict import FrozenDict  # 从 flax 导入 FrozenDict 用于不可变字典
from flax.jax_utils import unreplicate  # 从 flax 导入 unreplicate，用于去除复制
from flax.training.common_utils import shard  # 从 flax 导入 shard，用于数据分片
from packaging import version  # 导入 version 用于版本比较
from PIL import Image  # 从 PIL 导入 Image 用于图像处理
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel  # 导入 transformers 库的相关组件

from ...models import FlaxAutoencoderKL, FlaxUNet2DConditionModel  # 导入自定义模型
from ...schedulers import (  # 从自定义调度器导入各类调度器
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
from ...utils import PIL_INTERPOLATION, deprecate, logging, replace_example_docstring  # 导入工具函数
from ..pipeline_flax_utils import FlaxDiffusionPipeline  # 导入 FlaxDiffusionPipeline 类
from .pipeline_output import FlaxStableDiffusionPipelineOutput  # 导入输出类
from .safety_checker_flax import FlaxStableDiffusionSafetyChecker  # 导入安全检查器类

logger = logging.get_logger(__name__)  # 创建日志记录器，使用当前模块名称

# 设置为 True 时使用 Python 的 for 循环而不是 jax.fori_loop，以便于调试
DEBUG = False

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串，通常用于文档生成
```  
    # 示例代码块，用于展示如何使用 JAX 和 Flax 进行图像处理
        Examples:
            ```py
            # 导入必要的库
            >>> import jax
            >>> import numpy as np
            >>> from flax.jax_utils import replicate
            >>> from flax.training.common_utils import shard
            >>> import PIL
            >>> import requests
            >>> from io import BytesIO
            >>> from diffusers import FlaxStableDiffusionInpaintPipeline
    
            # 定义一个函数，用于下载图像并转换为 RGB 格式
            >>> def download_image(url):
            ...     # 发送 GET 请求以获取图像内容
            ...     response = requests.get(url)
            ...     # 打开下载的内容并转换为 RGB 图像
            ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")
    
            # 定义图像和掩码的 URL
            >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
            >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    
            # 下载并调整初始图像和掩码图像的大小
            >>> init_image = download_image(img_url).resize((512, 512))
            >>> mask_image = download_image(mask_url).resize((512, 512))
    
            # 从预训练模型加载管道和参数
            >>> pipeline, params = FlaxStableDiffusionInpaintPipeline.from_pretrained(
            ...     "xvjiarui/stable-diffusion-2-inpainting"
            ... )
    
            # 定义处理图像时使用的提示
            >>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
            # 初始化随机种子
            >>> prng_seed = jax.random.PRNGKey(0)
            # 定义推理步骤的数量
            >>> num_inference_steps = 50
    
            # 获取设备数量以便并行处理
            >>> num_samples = jax.device_count()
            # 将提示、初始图像和掩码图像扩展为设备数量的列表
            >>> prompt = num_samples * [prompt]
            >>> init_image = num_samples * [init_image]
            >>> mask_image = num_samples * [mask_image]
            # 准备输入，得到提示 ID 和处理后的图像
            >>> prompt_ids, processed_masked_images, processed_masks = pipeline.prepare_inputs(
            ...     prompt, init_image, mask_image
            ... )
            # 分割输入和随机数生成器
    
            # 复制参数以适应每个设备
            >>> params = replicate(params)
            # 根据设备数量分割随机种子
            >>> prng_seed = jax.random.split(prng_seed, jax.device_count())
            # 将提示 ID 和处理后的图像分割以适应每个设备
            >>> prompt_ids = shard(prompt_ids)
            >>> processed_masked_images = shard(processed_masked_images)
            >>> processed_masks = shard(processed_masks)
    
            # 运行管道以生成图像
            >>> images = pipeline(
            ...     prompt_ids, processed_masks, processed_masked_images, params, prng_seed, num_inference_steps, jit=True
            ... ).images
            # 将生成的图像数组转换为 PIL 图像格式
            >>> images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
            ```  
# FlaxStableDiffusionInpaintPipeline 类定义，继承自 FlaxDiffusionPipeline
class FlaxStableDiffusionInpaintPipeline(FlaxDiffusionPipeline):
    r"""
    Flax 基于 Stable Diffusion 的文本引导图像修补的管道。

    <Tip warning={true}>
    
    🧪 这是一个实验性功能！

    </Tip>

    该模型继承自 [`FlaxDiffusionPipeline`]。有关所有管道通用方法（下载、保存、在特定设备上运行等）的实现，请查看父类文档。

    参数:
        vae ([`FlaxAutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`~transformers.FlaxCLIPTextModel`]):
            冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于标记化文本的 `CLIPTokenizer`。
        unet ([`FlaxUNet2DConditionModel`]):
            用于去噪编码图像潜在表示的 `FlaxUNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用以去噪编码图像潜在表示的调度器。可以是以下之一
            [`FlaxDDIMScheduler`], [`FlaxLMSDiscreteScheduler`], [`FlaxPNDMScheduler`] 或
            [`FlaxDPMSolverMultistepScheduler`]。
        safety_checker ([`FlaxStableDiffusionSafetyChecker`]):
            估计生成图像是否可能被认为是冒犯性或有害的分类模块。
            请参考 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 以获取有关模型潜在危害的更多详细信息。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            从生成图像中提取特征的 `CLIPImageProcessor`；用作 `safety_checker` 的输入。
    """

    # 构造函数初始化
    def __init__(
        # 变分自编码器（VAE）模型实例
        vae: FlaxAutoencoderKL,
        # 文本编码器模型实例
        text_encoder: FlaxCLIPTextModel,
        # 标记器实例
        tokenizer: CLIPTokenizer,
        # 去噪模型实例
        unet: FlaxUNet2DConditionModel,
        # 调度器实例，指定可用的调度器类型
        scheduler: Union[
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],
        # 安全检查模块实例
        safety_checker: FlaxStableDiffusionSafetyChecker,
        # 特征提取器实例
        feature_extractor: CLIPImageProcessor,
        # 数据类型，默认为 float32
        dtype: jnp.dtype = jnp.float32,
    # 定义初始化方法，接收多个参数
        ):
            # 调用父类的初始化方法
            super().__init__()
            # 设置数据类型属性
            self.dtype = dtype
    
            # 检查安全检查器是否为 None
            if safety_checker is None:
                # 记录警告信息，提醒用户禁用安全检查器的风险
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 检查 UNet 版本是否小于 0.9.0
            is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
                version.parse(unet.config._diffusers_version).base_version
            ) < version.parse("0.9.0.dev0")
            # 检查 UNet 的样本大小是否小于 64
            is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
            # 如果满足两个条件，构造弃用警告信息
            if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
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
                # 调用弃用函数，传递警告信息
                deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
                # 创建新配置字典，并更新样本大小为 64
                new_config = dict(unet.config)
                new_config["sample_size"] = 64
                # 将新配置赋值给 UNet 的内部字典
                unet._internal_dict = FrozenDict(new_config)
    
            # 注册多个模块以供使用
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
    
        # 定义准备输入的方法，接收多个参数
        def prepare_inputs(
            self,
            # 输入提示，可以是字符串或字符串列表
            prompt: Union[str, List[str]],
            # 输入图像，可以是单张图像或图像列表
            image: Union[Image.Image, List[Image.Image]],
            # 输入掩码，可以是单张掩码或掩码列表
            mask: Union[Image.Image, List[Image.Image]],
    ):
        # 检查 prompt 是否为字符串或列表类型，不符合则抛出异常
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查 image 是否为 PIL 图像或列表类型，不符合则抛出异常
        if not isinstance(image, (Image.Image, list)):
            raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")

        # 如果 image 是单个 PIL 图像，则将其转为列表
        if isinstance(image, Image.Image):
            image = [image]

        # 检查 mask 是否为 PIL 图像或列表类型，不符合则抛出异常
        if not isinstance(mask, (Image.Image, list)):
            raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")

        # 如果 mask 是单个 PIL 图像，则将其转为列表
        if isinstance(mask, Image.Image):
            mask = [mask]

        # 对图像进行预处理，并合并为一个数组
        processed_images = jnp.concatenate([preprocess_image(img, jnp.float32) for img in image])
        # 对掩膜进行预处理，并合并为一个数组
        processed_masks = jnp.concatenate([preprocess_mask(m, jnp.float32) for m in mask])
        # 将处理后的掩膜中小于0.5的值设为0
        processed_masks = processed_masks.at[processed_masks < 0.5].set(0)
        # 将处理后的掩膜中大于等于0.5的值设为1
        processed_masks = processed_masks.at[processed_masks >= 0.5].set(1)

        # 根据掩膜对图像进行遮罩处理
        processed_masked_images = processed_images * (processed_masks < 0.5)

        # 将 prompt 进行编码，并设置最大长度、填充和截断
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        # 返回编码后的输入 ID、处理后的图像和掩膜
        return text_input.input_ids, processed_masked_images, processed_masks

    def _get_has_nsfw_concepts(self, features, params):
        # 使用安全检查器检查特征中是否存在 NSFW 概念
        has_nsfw_concepts = self.safety_checker(features, params)
        # 返回 NSFW 概念的检测结果
        return has_nsfw_concepts

    def _run_safety_checker(self, images, safety_model_params, jit=False):
        # 将传入的图像数组转换为 PIL 图像
        pil_images = [Image.fromarray(image) for image in images]
        # 提取图像特征并返回张量形式的像素值
        features = self.feature_extractor(pil_images, return_tensors="np").pixel_values

        # 如果开启 JIT 优化，则对特征进行分片
        if jit:
            features = shard(features)
            # 使用 NSFW 概念检测函数获取结果
            has_nsfw_concepts = _p_get_has_nsfw_concepts(self, features, safety_model_params)
            # 对结果进行反分片处理
            has_nsfw_concepts = unshard(has_nsfw_concepts)
            safety_model_params = unreplicate(safety_model_params)
        else:
            # 否则直接调用获取 NSFW 概念的函数
            has_nsfw_concepts = self._get_has_nsfw_concepts(features, safety_model_params)

        images_was_copied = False
        # 遍历每个 NSFW 概念的检测结果
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                # 如果发现 NSFW 概念且尚未复制图像，则进行复制
                if not images_was_copied:
                    images_was_copied = True
                    images = images.copy()

                # 将对应图像替换为黑色图像
                images[idx] = np.zeros(images[idx].shape, dtype=np.uint8)  # black image

            # 如果检测到任何 NSFW 概念，则发出警告
            if any(has_nsfw_concepts):
                warnings.warn(
                    "Potential NSFW content was detected in one or more images. A black image will be returned"
                    " instead. Try again with a different prompt and/or seed."
                )

        # 返回处理后的图像和 NSFW 概念的检测结果
        return images, has_nsfw_concepts
    # 定义一个生成函数，处理图像生成的相关操作
        def _generate(
            # 输入的提示ID数组，通常用于模型输入
            self,
            prompt_ids: jnp.ndarray,
            # 输入的掩码数组，指示哪些部分需要处理
            mask: jnp.ndarray,
            # 被掩码的图像数组，作为生成过程的基础
            masked_image: jnp.ndarray,
            # 模型参数，可以是字典或冻结字典类型
            params: Union[Dict, FrozenDict],
            # 随机数种子，用于生成可重复的结果
            prng_seed: jax.Array,
            # 推理步骤的数量，控制生成的细致程度
            num_inference_steps: int,
            # 生成图像的高度
            height: int,
            # 生成图像的宽度
            width: int,
            # 指导比例，用于调整生成图像与提示的相关性
            guidance_scale: float,
            # 可选的潜在表示，用于进一步控制生成过程
            latents: Optional[jnp.ndarray] = None,
            # 可选的负提示ID数组，用于增强生成效果
            neg_prompt_ids: Optional[jnp.ndarray] = None,
        # 使用装饰器替换示例文档字符串，提供函数的文档说明
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义调用函数，进行图像生成操作
        def __call__(
            # 输入的提示ID数组
            self,
            prompt_ids: jnp.ndarray,
            # 输入的掩码数组
            mask: jnp.ndarray,
            # 被掩码的图像数组
            masked_image: jnp.ndarray,
            # 模型参数
            params: Union[Dict, FrozenDict],
            # 随机数种子
            prng_seed: jax.Array,
            # 推理步骤的数量，默认为50
            num_inference_steps: int = 50,
            # 生成图像的高度，默认为None（可选）
            height: Optional[int] = None,
            # 生成图像的宽度，默认为None（可选）
            width: Optional[int] = None,
            # 指导比例，默认为7.5
            guidance_scale: Union[float, jnp.ndarray] = 7.5,
            # 可选的潜在表示，默认为None
            latents: jnp.ndarray = None,
            # 可选的负提示ID数组，默认为None
            neg_prompt_ids: jnp.ndarray = None,
            # 返回字典格式的结果，默认为True
            return_dict: bool = True,
            # 是否使用JIT编译，默认为False
            jit: bool = False,
# 静态参数为管道、推理步骤数、高度和宽度。更改会触发重新编译。
# 非静态参数为在其第一维度（因此为`0`）映射的（分片）输入张量。
@partial(
    jax.pmap,  # 使用 JAX 的并行映射功能
    in_axes=(None, 0, 0, 0, 0, 0, None, None, None, 0, 0, 0),  # 指定输入张量的维度映射
    static_broadcasted_argnums=(0, 6, 7, 8),  # 静态广播参数的索引
)
def _p_generate(
    pipe,  # 管道对象
    prompt_ids,  # 提示 ID
    mask,  # 掩码
    masked_image,  # 被掩码的图像
    params,  # 参数
    prng_seed,  # 随机种子
    num_inference_steps,  # 推理步骤数
    height,  # 图像高度
    width,  # 图像宽度
    guidance_scale,  # 引导比例
    latents,  # 潜在表示
    neg_prompt_ids,  # 负提示 ID
):
    return pipe._generate(  # 调用管道的生成方法
        prompt_ids,  # 提示 ID
        mask,  # 掩码
        masked_image,  # 被掩码的图像
        params,  # 参数
        prng_seed,  # 随机种子
        num_inference_steps,  # 推理步骤数
        height,  # 图像高度
        width,  # 图像宽度
        guidance_scale,  # 引导比例
        latents,  # 潜在表示
        neg_prompt_ids,  # 负提示 ID
    )


@partial(jax.pmap, static_broadcasted_argnums=(0,))  # 使用 JAX 的并行映射功能
def _p_get_has_nsfw_concepts(pipe, features, params):  # 检查特征是否包含 NSFW 概念
    return pipe._get_has_nsfw_concepts(features, params)  # 调用管道的方法


def unshard(x: jnp.ndarray):  # 定义 unshard 函数，接受一个 ndarray
    # einops.rearrange(x, 'd b ... -> (d b) ...')  # 用于调整张量的形状
    num_devices, batch_size = x.shape[:2]  # 获取设备数量和批次大小
    rest = x.shape[2:]  # 获取其余维度
    return x.reshape(num_devices * batch_size, *rest)  # 重新调整形状为 (d*b, ...)


def preprocess_image(image, dtype):  # 定义预处理图像的函数
    w, h = image.size  # 获取图像的宽度和高度
    w, h = (x - x % 32 for x in (w, h))  # 调整宽度和高度为 32 的整数倍
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])  # 按新大小调整图像
    image = jnp.array(image).astype(dtype) / 255.0  # 转换为 ndarray 并归一化
    image = image[None].transpose(0, 3, 1, 2)  # 调整维度顺序
    return 2.0 * image - 1.0  # 将图像值范围调整到 [-1, 1]


def preprocess_mask(mask, dtype):  # 定义预处理掩码的函数
    w, h = mask.size  # 获取掩码的宽度和高度
    w, h = (x - x % 32 for x in (w, h))  # 调整宽度和高度为 32 的整数倍
    mask = mask.resize((w, h))  # 按新大小调整掩码
    mask = jnp.array(mask.convert("L")).astype(dtype) / 255.0  # 转换为灰度并归一化
    mask = jnp.expand_dims(mask, axis=(0, 1))  # 扩展维度以适应模型输入

    return mask  # 返回处理后的掩码
```