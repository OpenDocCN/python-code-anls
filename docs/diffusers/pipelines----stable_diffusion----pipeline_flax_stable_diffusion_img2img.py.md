# `.\diffusers\pipelines\stable_diffusion\pipeline_flax_stable_diffusion_img2img.py`

```py
# 版权声明，表明此文件的版权归 HuggingFace 团队所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证进行授权，用户必须遵守该许可证使用此文件
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 用户可以在以下链接获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用的法律要求或书面同意，否则此文件按“原样”提供，没有任何明示或暗示的担保
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解特定语言管辖权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings  # 导入警告模块，用于发出警告消息
from functools import partial  # 从 functools 导入 partial 函数，用于部分应用
from typing import Dict, List, Optional, Union  # 导入类型提示相关的模块

import jax  # 导入 JAX 库，用于加速数值计算
import jax.numpy as jnp  # 导入 JAX 的 NumPy 模块，作为 jnp 使用
import numpy as np  # 导入 NumPy 库，作为 np 使用
from flax.core.frozen_dict import FrozenDict  # 从 flax 导入 FrozenDict，用于创建不可变字典
from flax.jax_utils import unreplicate  # 从 flax 导入 unreplicate 函数，用于去除 JAX 复制
from flax.training.common_utils import shard  # 从 flax 导入 shard 函数，用于数据切分
from PIL import Image  # 从 PIL 导入 Image 模块，用于图像处理
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel  # 导入 transformers 中的处理器和模型

from ...models import FlaxAutoencoderKL, FlaxUNet2DConditionModel  # 导入模型
from ...schedulers import (  # 从调度器模块导入各种调度器
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
from ...utils import PIL_INTERPOLATION, logging, replace_example_docstring  # 导入工具函数和日志模块
from ..pipeline_flax_utils import FlaxDiffusionPipeline  # 从 pipeline_flax_utils 导入 FlaxDiffusionPipeline
from .pipeline_output import FlaxStableDiffusionPipelineOutput  # 从 pipeline_output 导入 FlaxStableDiffusionPipelineOutput
from .safety_checker_flax import FlaxStableDiffusionSafetyChecker  # 从 safety_checker_flax 导入安全检查器

logger = logging.get_logger(__name__)  # 创建一个日志记录器，使用当前模块的名称

# 设置为 True 时使用 Python 的 for 循环，而非 jax.fori_loop，以便于调试
DEBUG = False

# 示例文档字符串的模板
EXAMPLE_DOC_STRING = """
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
    # 示例代码块，用于演示如何使用库和函数
    Examples:
        ```py
        # 导入 JAX 库
        >>> import jax
        # 导入 NumPy 库
        >>> import numpy as np
        # 导入 JAX 的 NumPy 实现
        >>> import jax.numpy as jnp
        # 从 flax.jax_utils 导入复制函数
        >>> from flax.jax_utils import replicate
        # 从 flax.training.common_utils 导入分片函数
        >>> from flax.training.common_utils import shard
        # 导入 requests 库用于发送 HTTP 请求
        >>> import requests
        # 从 io 模块导入 BytesIO 用于处理字节流
        >>> from io import BytesIO
        # 从 PIL 库导入 Image 类用于图像处理
        >>> from PIL import Image
        # 从 diffusers 导入 FlaxStableDiffusionImg2ImgPipeline 类
        >>> from diffusers import FlaxStableDiffusionImg2ImgPipeline


        # 定义一个创建随机数种子的函数
        >>> def create_key(seed=0):
        ...     # 返回一个基于给定种子的 JAX 随机数生成器密钥
        ...     return jax.random.PRNGKey(seed)


        # 使用种子 0 创建随机数生成器密钥
        >>> rng = create_key(0)

        # 定义要下载的图像 URL
        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        # 发送 GET 请求以获取图像
        >>> response = requests.get(url)
        # 从响应内容中读取图像，并转换为 RGB 模式
        >>> init_img = Image.open(BytesIO(response.content)).convert("RGB")
        # 调整图像大小为 768x512 像素
        >>> init_img = init_img.resize((768, 512))

        # 定义提示词
        >>> prompts = "A fantasy landscape, trending on artstation"

        # 从预训练模型中加载图像到图像生成管道
        >>> pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4",  # 模型名称
        ...     revision="flax",                  # 版本标识
        ...     dtype=jnp.bfloat16,               # 数据类型
        ... )

        # 获取设备数量以生成样本
        >>> num_samples = jax.device_count()
        # 根据设备数量拆分随机数生成器的密钥
        >>> rng = jax.random.split(rng, jax.device_count())
        # 准备输入的提示词和图像，复制 num_samples 次
        >>> prompt_ids, processed_image = pipeline.prepare_inputs(
        ...     prompt=[prompts] * num_samples,    # 创建提示词列表
        ...     image=[init_img] * num_samples     # 创建图像列表
        ... )
        # 复制参数以便在多个设备上使用
        >>> p_params = replicate(params)
        # 将提示词 ID 分片以适应设备
        >>> prompt_ids = shard(prompt_ids)
        # 将处理后的图像分片以适应设备
        >>> processed_image = shard(processed_image)

        # 调用管道生成图像
        >>> output = pipeline(
        ...     prompt_ids=prompt_ids,              # 提示词 ID
        ...     image=processed_image,              # 处理后的图像
        ...     params=p_params,                    # 复制的参数
        ...     prng_seed=rng,                      # 随机数种子
        ...     strength=0.75,                     # 强度参数
        ...     num_inference_steps=50,            # 推理步骤数
        ...     jit=True,                          # 启用 JIT 编译
        ...     height=512,                        # 输出图像高度
        ...     width=768,                         # 输出图像宽度
        ... ).images  # 获取生成的图像

        # 将输出的图像转换为 PIL 格式以便展示
        >>> output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
        ```py 
# 定义一个基于 Flax 的文本引导图像生成管道类，用于图像到图像的生成
class FlaxStableDiffusionImg2ImgPipeline(FlaxDiffusionPipeline):
    r"""
    基于 Flax 的管道，用于使用 Stable Diffusion 进行文本引导的图像到图像生成。

    该模型继承自 [`FlaxDiffusionPipeline`]。有关所有管道的通用方法的文档（下载、保存、在特定设备上运行等），请查看超类文档。

    参数：
        vae ([`FlaxAutoencoderKL`]):
            用于对图像进行编码和解码的变分自编码器（VAE）模型，将图像转换为潜在表示。
        text_encoder ([`~transformers.FlaxCLIPTextModel`]):
            冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于对文本进行标记化的 `CLIPTokenizer`。
        unet ([`FlaxUNet2DConditionModel`]):
            用于对编码图像潜在空间进行去噪的 `FlaxUNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用的调度器，用于去噪编码的图像潜在空间。可以是
            [`FlaxDDIMScheduler`], [`FlaxLMSDiscreteScheduler`], [`FlaxPNDMScheduler`] 或
            [`FlaxDPMSolverMultistepScheduler`] 之一。
        safety_checker ([`FlaxStableDiffusionSafetyChecker`]):
            分类模块，用于评估生成的图像是否可能被认为是冒犯性或有害的。
            有关模型潜在危害的更多详细信息，请参阅 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            用于从生成图像中提取特征的 `CLIPImageProcessor`；作为输入用于 `safety_checker`。
    """

    # 初始化方法，设置管道的各个组件
    def __init__(
        self,
        # 变分自编码器模型
        vae: FlaxAutoencoderKL,
        # 文本编码器模型
        text_encoder: FlaxCLIPTextModel,
        # 文本标记器
        tokenizer: CLIPTokenizer,
        # 去噪模型
        unet: FlaxUNet2DConditionModel,
        # 调度器，用于去噪处理
        scheduler: Union[
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],
        # 安全检查模块
        safety_checker: FlaxStableDiffusionSafetyChecker,
        # 特征提取器
        feature_extractor: CLIPImageProcessor,
        # 数据类型，默认为 float32
        dtype: jnp.dtype = jnp.float32,
    ):
        # 调用父类的构造函数
        super().__init__()
        # 设置数据类型属性
        self.dtype = dtype

        # 检查安全检查器是否为 None
        if safety_checker is None:
            # 记录警告，提醒用户禁用了安全检查器
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 注册模块，将各个组件进行初始化
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        # 计算 VAE 的缩放因子，基于其配置的输出通道数
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    # 准备输入，接受文本提示和图像
    def prepare_inputs(self, prompt: Union[str, List[str]], image: Union[Image.Image, List[Image.Image]]):
        # 检查 prompt 类型是否为字符串或列表
        if not isinstance(prompt, (str, list)):
            # 如果不符合类型，抛出错误
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查 image 类型是否为 PIL 图像或列表
        if not isinstance(image, (Image.Image, list)):
            # 如果不符合类型，抛出错误
            raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")

        # 如果 image 是单个图像，则转换为列表
        if isinstance(image, Image.Image):
            image = [image]

        # 预处理图像，并将它们拼接为一个数组
        processed_images = jnp.concatenate([preprocess(img, jnp.float32) for img in image])

        # 将文本提示编码为模型输入格式
        text_input = self.tokenizer(
            prompt,
            padding="max_length",  # 填充到最大长度
            max_length=self.tokenizer.model_max_length,  # 最大长度设置
            truncation=True,  # 超出最大长度时截断
            return_tensors="np",  # 返回 NumPy 格式的张量
        )
        # 返回文本输入 ID 和处理后的图像
        return text_input.input_ids, processed_images

    # 获取是否包含不适宜内容的概念
    def _get_has_nsfw_concepts(self, features, params):
        # 使用安全检查器检查特征是否包含不适宜内容
        has_nsfw_concepts = self.safety_checker(features, params)
        # 返回检查结果
        return has_nsfw_concepts
    # 定义一个安全检查器的运行方法，处理输入的图像
    def _run_safety_checker(self, images, safety_model_params, jit=False):
        # 当 jit 为 True 时，安全模型参数应已被复制
        pil_images = [Image.fromarray(image) for image in images]  # 将 NumPy 数组转换为 PIL 图像
        features = self.feature_extractor(pil_images, return_tensors="np").pixel_values  # 提取图像特征并返回像素值
    
        if jit:  # 如果启用 JIT 编译
            features = shard(features)  # 将特征分片以优化性能
            has_nsfw_concepts = _p_get_has_nsfw_concepts(self, features, safety_model_params)  # 检查是否存在 NSFW 概念
            has_nsfw_concepts = unshard(has_nsfw_concepts)  # 将结果反分片
            safety_model_params = unreplicate(safety_model_params)  # 反复制安全模型参数
        else:  # 如果没有启用 JIT 编译
            has_nsfw_concepts = self._get_has_nsfw_concepts(features, safety_model_params)  # 获取 NSFW 概念的存在性
    
        images_was_copied = False  # 标记图像是否已被复制
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):  # 遍历 NSFW 概念的列表
            if has_nsfw_concept:  # 如果检测到 NSFW 概念
                if not images_was_copied:  # 如果尚未复制图像
                    images_was_copied = True  # 标记为已复制
                    images = images.copy()  # 复制图像数组
    
                images[idx] = np.zeros(images[idx].shape, dtype=np.uint8)  # 用黑色图像替换原图像
    
            if any(has_nsfw_concepts):  # 如果任一图像有 NSFW 概念
                warnings.warn(  # 发出警告
                    "Potential NSFW content was detected in one or more images. A black image will be returned"
                    " instead. Try again with a different prompt and/or seed."
                )
    
        return images, has_nsfw_concepts  # 返回处理后的图像和 NSFW 概念的存在性
    
    # 定义获取开始时间步的方法
    def get_timestep_start(self, num_inference_steps, strength):
        # 使用初始时间步计算原始时间步
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)  # 计算初始时间步，确保不超出总步骤
    
        t_start = max(num_inference_steps - init_timestep, 0)  # 计算开始时间步，确保不为负
    
        return t_start  # 返回开始时间步
    
    # 定义生成方法
    def _generate(
        self,
        prompt_ids: jnp.ndarray,  # 输入提示的 ID
        image: jnp.ndarray,  # 输入图像
        params: Union[Dict, FrozenDict],  # 模型参数
        prng_seed: jax.Array,  # 随机种子
        start_timestep: int,  # 开始时间步
        num_inference_steps: int,  # 推理步骤数量
        height: int,  # 生成图像的高度
        width: int,  # 生成图像的宽度
        guidance_scale: float,  # 引导比例
        noise: Optional[jnp.ndarray] = None,  # 噪声选项
        neg_prompt_ids: Optional[jnp.ndarray] = None,  # 负提示 ID 选项
        @replace_example_docstring(EXAMPLE_DOC_STRING)  # 用示例文档字符串替换
        def __call__(  # 定义可调用方法
            self,
            prompt_ids: jnp.ndarray,  # 输入提示的 ID
            image: jnp.ndarray,  # 输入图像
            params: Union[Dict, FrozenDict],  # 模型参数
            prng_seed: jax.Array,  # 随机种子
            strength: float = 0.8,  # 强度参数，默认为 0.8
            num_inference_steps: int = 50,  # 推理步骤数量，默认为 50
            height: Optional[int] = None,  # 生成图像的高度，默认为 None
            width: Optional[int] = None,  # 生成图像的宽度，默认为 None
            guidance_scale: Union[float, jnp.ndarray] = 7.5,  # 引导比例，默认为 7.5
            noise: jnp.ndarray = None,  # 噪声，默认为 None
            neg_prompt_ids: jnp.ndarray = None,  # 负提示 ID，默认为 None
            return_dict: bool = True,  # 是否返回字典，默认为 True
            jit: bool = False,  # 是否启用 JIT 编译，默认为 False
# 静态参数为 pipe, start_timestep, num_inference_steps, height, width。任何更改都会触发重新编译。
# 非静态参数为 (sharded) 输入张量，按其第一维映射 (因此为 `0`)。
@partial(
    jax.pmap,  # 使用 JAX 的 pmap 函数进行并行映射
    in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0, 0, 0),  # 指定输入参数的维度
    static_broadcasted_argnums=(0, 5, 6, 7, 8),  # 静态广播的参数索引
)
def _p_generate(
    pipe,  # 生成管道对象
    prompt_ids,  # 输入的提示 ID
    image,  # 输入的图像数据
    params,  # 其他参数
    prng_seed,  # 随机数种子
    start_timestep,  # 开始的时间步
    num_inference_steps,  # 推理的步骤数
    height,  # 图像的高度
    width,  # 图像的宽度
    guidance_scale,  # 引导尺度
    noise,  # 噪声数据
    neg_prompt_ids,  # 负提示 ID
):
    # 调用管道的生成方法，传递所有必要的参数
    return pipe._generate(
        prompt_ids,  # 提示 ID
        image,  # 图像数据
        params,  # 其他参数
        prng_seed,  # 随机数种子
        start_timestep,  # 开始时间步
        num_inference_steps,  # 推理步骤数
        height,  # 图像高度
        width,  # 图像宽度
        guidance_scale,  # 引导尺度
        noise,  # 噪声数据
        neg_prompt_ids,  # 负提示 ID
    )


@partial(jax.pmap, static_broadcasted_argnums=(0,))  # 使用 JAX 的 pmap 函数进行并行映射
def _p_get_has_nsfw_concepts(pipe, features, params):
    # 调用管道的方法以获取是否包含 NSFW 概念的特征
    return pipe._get_has_nsfw_concepts(features, params)


def unshard(x: jnp.ndarray):
    # 将输入张量 x 重组为适合的形状，合并设备和批次维度
    num_devices, batch_size = x.shape[:2]  # 获取设备数量和批次大小
    rest = x.shape[2:]  # 获取剩余维度
    # 重新调整形状为 (num_devices * batch_size, 剩余维度)
    return x.reshape(num_devices * batch_size, *rest)


def preprocess(image, dtype):
    w, h = image.size  # 获取图像的宽度和高度
    # 调整宽度和高度为 32 的整数倍
    w, h = (x - x % 32 for x in (w, h))  
    # 重新调整图像大小，使用 Lanczos 插值法
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    # 将图像转换为 NumPy 数组并归一化到 [0, 1] 范围
    image = jnp.array(image).astype(dtype) / 255.0
    # 调整数组维度为 (1, 通道数, 高度, 宽度)
    image = image[None].transpose(0, 3, 1, 2)
    # 将图像值范围转换为 [-1, 1]
    return 2.0 * image - 1.0
```