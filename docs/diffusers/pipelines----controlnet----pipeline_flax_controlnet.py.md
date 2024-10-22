# `.\diffusers\pipelines\controlnet\pipeline_flax_controlnet.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）授权；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有规定，
# 否则根据许可证分发的软件是“按原样”提供的，
# 不提供任何形式的担保或条件，无论是明示或暗示。
# 有关许可证下的特定语言的权限和限制，请参见许可证。

import warnings  # 导入警告模块，用于处理警告信息
from functools import partial  # 从 functools 导入 partial，用于部分函数应用
from typing import Dict, List, Optional, Union  # 导入类型提示，方便函数参数和返回值的类型注释

import jax  # 导入 JAX，用于高性能数值计算
import jax.numpy as jnp  # 导入 JAX 的 NumPy 接口，提供数组操作功能
import numpy as np  # 导入 NumPy，提供数值计算功能
from flax.core.frozen_dict import FrozenDict  # 从 flax 导入 FrozenDict，用于不可变字典
from flax.jax_utils import unreplicate  # 从 flax 导入 unreplicate，用于在 JAX 中处理设备数据
from flax.training.common_utils import shard  # 从 flax 导入 shard，用于数据并行
from PIL import Image  # 从 PIL 导入 Image，用于图像处理
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel  # 导入 CLIP 相关模块，处理图像和文本

from ...models import FlaxAutoencoderKL, FlaxControlNetModel, FlaxUNet2DConditionModel  # 导入模型定义
from ...schedulers import (  # 导入调度器，用于训练过程中的控制
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
from ...utils import PIL_INTERPOLATION, logging, replace_example_docstring  # 导入工具函数和常量
from ..pipeline_flax_utils import FlaxDiffusionPipeline  # 导入扩散管道
from ..stable_diffusion import FlaxStableDiffusionPipelineOutput  # 导入稳定扩散管道输出
from ..stable_diffusion.safety_checker_flax import FlaxStableDiffusionSafetyChecker  # 导入安全检查器

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，方便调试和信息输出

# 设置为 True 以使用 Python 循环而不是 jax.fori_loop，以便于调试
DEBUG = False  # 调试模式标志，默认为关闭状态

EXAMPLE_DOC_STRING = """  # 示例文档字符串，可能用于文档生成或示例展示


```  # 示例结束标志
    Examples:
        ```py
        >>> import jax  # 导入 JAX 库，用于高性能数值计算
        >>> import numpy as np  # 导入 NumPy 库，支持数组操作
        >>> import jax.numpy as jnp  # 导入 JAX 的 NumPy，支持自动微分和GPU加速
        >>> from flax.jax_utils import replicate  # 从 Flax 导入 replicate 函数，用于参数复制
        >>> from flax.training.common_utils import shard  # 从 Flax 导入 shard 函数，用于数据分片
        >>> from diffusers.utils import load_image, make_image_grid  # 从 diffusers 导入图像加载和网格生成工具
        >>> from PIL import Image  # 导入 PIL 库，用于图像处理
        >>> from diffusers import FlaxStableDiffusionControlNetPipeline, FlaxControlNetModel  # 导入用于稳定扩散模型和控制网的类

        >>> def create_key(seed=0):  # 定义函数创建随机数生成器的密钥
        ...     return jax.random.PRNGKey(seed)  # 返回一个以 seed 为种子的 PRNG 密钥

        >>> rng = create_key(0)  # 创建随机数生成器的密钥，种子为 0

        >>> # get canny image  # 获取 Canny 边缘检测图像
        >>> canny_image = load_image(  # 使用 load_image 函数加载图像
        ...     "https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/blog_post_cell_10_output_0.jpeg"  # 指定图像的 URL
        ... )

        >>> prompts = "best quality, extremely detailed"  # 定义用于生成图像的正向提示
        >>> negative_prompts = "monochrome, lowres, bad anatomy, worst quality, low quality"  # 定义生成图像时要避免的负向提示

        >>> # load control net and stable diffusion v1-5  # 加载控制网络和稳定扩散模型 v1-5
        >>> controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(  # 从预训练模型加载控制网络及其参数
        ...     "lllyasviel/sd-controlnet-canny", from_pt=True, dtype=jnp.float32  # 指定模型名称、来源及数据类型
        ... )
        >>> pipe, params = FlaxStableDiffusionControlNetPipeline.from_pretrained(  # 从预训练模型加载稳定扩散管道及其参数
        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, revision="flax", dtype=jnp.float32  # 指定模型名称、控制网、版本和数据类型
        ... )
        >>> params["controlnet"] = controlnet_params  # 将控制网参数存入管道参数中

        >>> num_samples = jax.device_count()  # 获取当前设备的数量，设置样本数量
        >>> rng = jax.random.split(rng, jax.device_count())  # 将随机数生成器的密钥根据设备数量进行分割

        >>> prompt_ids = pipe.prepare_text_inputs([prompts] * num_samples)  # 准备正向提示的输入，针对每个样本复制
        >>> negative_prompt_ids = pipe.prepare_text_inputs([negative_prompts] * num_samples)  # 准备负向提示的输入，针对每个样本复制
        >>> processed_image = pipe.prepare_image_inputs([canny_image] * num_samples)  # 准备处理后的图像输入，针对每个样本复制

        >>> p_params = replicate(params)  # 复制参数以便在多个设备上使用
        >>> prompt_ids = shard(prompt_ids)  # 将正向提示的输入数据进行分片
        >>> negative_prompt_ids = shard(negative_prompt_ids)  # 将负向提示的输入数据进行分片
        >>> processed_image = shard(processed_image)  # 将处理后的图像输入数据进行分片

        >>> output = pipe(  # 调用管道生成输出
        ...     prompt_ids=prompt_ids,  # 传入正向提示 ID
        ...     image=processed_image,  # 传入处理后的图像
        ...     params=p_params,  # 传入复制的参数
        ...     prng_seed=rng,  # 传入随机数生成器的密钥
        ...     num_inference_steps=50,  # 设置推理的步骤数
        ...     neg_prompt_ids=negative_prompt_ids,  # 传入负向提示 ID
        ...     jit=True,  # 启用 JIT 编译
        ... ).images  # 获取生成的图像

        >>> output_images = pipe.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))  # 将输出图像转换为 PIL 格式
        >>> output_images = make_image_grid(output_images, num_samples // 4, 4)  # 将图像生成网格格式，指定每行显示的图像数量
        >>> output_images.save("generated_image.png")  # 保存生成的图像为 PNG 文件
        ``` 
# 定义一个类，基于 Flax 实现 Stable Diffusion 的控制网文本到图像生成管道
class FlaxStableDiffusionControlNetPipeline(FlaxDiffusionPipeline):
    r"""
    基于 Flax 的管道，用于使用 Stable Diffusion 和 ControlNet 指导进行文本到图像生成。

    此模型继承自 [`FlaxDiffusionPipeline`]。有关所有管道实现的通用方法（下载、保存、在特定设备上运行等），请查看超类文档。

    参数：
        vae ([`FlaxAutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`~transformers.FlaxCLIPTextModel`]):
            冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于对文本进行分词的 `CLIPTokenizer`。
        unet ([`FlaxUNet2DConditionModel`]):
            一个 `FlaxUNet2DConditionModel`，用于去噪编码后的图像潜在表示。
        controlnet ([`FlaxControlNetModel`]):
            在去噪过程中为 `unet` 提供额外的条件信息。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 结合使用的调度器，以去噪编码的图像潜在表示。可以是
            [`FlaxDDIMScheduler`], [`FlaxLMSDiscreteScheduler`], [`FlaxPNDMScheduler`] 或
            [`FlaxDPMSolverMultistepScheduler`] 中的一个。
        safety_checker ([`FlaxStableDiffusionSafetyChecker`]):
            分类模块，评估生成的图像是否可能被视为冒犯或有害。
            有关模型潜在危害的更多细节，请参阅 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            一个 `CLIPImageProcessor`，用于提取生成图像的特征；用于 `safety_checker` 的输入。
    """

    # 初始化方法，定义所需参数及其类型
    def __init__(
        # 变分自编码器（VAE）模型，用于图像编码和解码
        vae: FlaxAutoencoderKL,
        # 冻结的文本编码器模型
        text_encoder: FlaxCLIPTextModel,
        # 文本分词器
        tokenizer: CLIPTokenizer,
        # 去噪模型
        unet: FlaxUNet2DConditionModel,
        # 控制网模型
        controlnet: FlaxControlNetModel,
        # 图像去噪的调度器
        scheduler: Union[
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],
        # 安全检查模块
        safety_checker: FlaxStableDiffusionSafetyChecker,
        # 特征提取器
        feature_extractor: CLIPImageProcessor,
        # 数据类型，默认为 32 位浮点数
        dtype: jnp.dtype = jnp.float32,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置数据类型属性
        self.dtype = dtype

        # 检查安全检查器是否为 None
        if safety_checker is None:
            # 记录警告，告知用户已禁用安全检查器
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 注册各个模块，方便后续使用
        self.register_modules(
            vae=vae,  # 变分自编码器
            text_encoder=text_encoder,  # 文本编码器
            tokenizer=tokenizer,  # 分词器
            unet=unet,  # UNet 模型
            controlnet=controlnet,  # 控制网络
            scheduler=scheduler,  # 调度器
            safety_checker=safety_checker,  # 安全检查器
            feature_extractor=feature_extractor,  # 特征提取器
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def prepare_text_inputs(self, prompt: Union[str, List[str]]):
        # 检查 prompt 类型是否为字符串或列表
        if not isinstance(prompt, (str, list)):
            # 如果类型不符，抛出值错误
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 使用分词器处理输入文本
        text_input = self.tokenizer(
            prompt,  # 输入的提示文本
            padding="max_length",  # 填充到最大长度
            max_length=self.tokenizer.model_max_length,  # 设置最大长度为分词器的最大模型长度
            truncation=True,  # 如果超过最大长度，则截断
            return_tensors="np",  # 返回 NumPy 格式的张量
        )

        # 返回处理后的输入 ID
        return text_input.input_ids

    def prepare_image_inputs(self, image: Union[Image.Image, List[Image.Image]]):
        # 检查图像类型是否为 PIL.Image 或列表
        if not isinstance(image, (Image.Image, list)):
            # 如果类型不符，抛出值错误
            raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")

        # 如果输入是单个图像，将其转换为列表
        if isinstance(image, Image.Image):
            image = [image]

        # 对所有图像进行预处理，并合并为一个数组
        processed_images = jnp.concatenate([preprocess(img, jnp.float32) for img in image])

        # 返回处理后的图像数组
        return processed_images

    def _get_has_nsfw_concepts(self, features, params):
        # 使用安全检查器检查是否存在不适当内容概念
        has_nsfw_concepts = self.safety_checker(features, params)
        # 返回检查结果
        return has_nsfw_concepts
    # 定义一个安全检查的私有方法，接收图像、模型参数和是否使用 JIT 编译的标志
    def _run_safety_checker(self, images, safety_model_params, jit=False):
        # 当 jit 为 True 时，safety_model_params 应该已经被复制
        # 将输入的图像数组转换为 PIL 图像格式
        pil_images = [Image.fromarray(image) for image in images]
        # 使用特征提取器处理 PIL 图像，返回其像素值
        features = self.feature_extractor(pil_images, return_tensors="np").pixel_values

        # 如果启用 JIT 编译
        if jit:
            # 对特征进行分片处理
            features = shard(features)
            # 检查特征中是否存在 NSFW 概念
            has_nsfw_concepts = _p_get_has_nsfw_concepts(self, features, safety_model_params)
            # 取消特征的分片
            has_nsfw_concepts = unshard(has_nsfw_concepts)
            # 取消模型参数的复制
            safety_model_params = unreplicate(safety_model_params)
        else:
            # 否则，直接获取 NSFW 概念的检查结果
            has_nsfw_concepts = self._get_has_nsfw_concepts(features, safety_model_params)

        # 初始化一个标志，指示图像是否已经被复制
        images_was_copied = False
        # 遍历每个 NSFW 概念的检查结果
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            # 如果检测到 NSFW 概念
            if has_nsfw_concept:
                # 如果还没有复制图像
                if not images_was_copied:
                    # 标记为已复制，并进行图像复制
                    images_was_copied = True
                    images = images.copy()

                # 将对应的图像替换为全黑图像
                images[idx] = np.zeros(images[idx].shape, dtype=np.uint8)  # black image

            # 如果存在任何 NSFW 概念
            if any(has_nsfw_concepts):
                # 发出警告，提示可能检测到不适宜内容
                warnings.warn(
                    "Potential NSFW content was detected in one or more images. A black image will be returned"
                    " instead. Try again with a different prompt and/or seed."
                )

        # 返回处理后的图像和 NSFW 概念的检查结果
        return images, has_nsfw_concepts

    # 定义一个生成图像的私有方法，接收多个参数以控制生成过程
    def _generate(
        self,
        prompt_ids: jnp.ndarray,  # 输入的提示 ID 数组
        image: jnp.ndarray,  # 输入的图像数据
        params: Union[Dict, FrozenDict],  # 模型参数，可能是字典或不可变字典
        prng_seed: jax.Array,  # 随机种子，用于随机数生成
        num_inference_steps: int,  # 推理步骤的数量
        guidance_scale: float,  # 指导比例，用于控制生成质量
        latents: Optional[jnp.ndarray] = None,  # 潜在变量，默认值为 None
        neg_prompt_ids: Optional[jnp.ndarray] = None,  # 负提示 ID，默认值为 None
        controlnet_conditioning_scale: float = 1.0,  # 控制网络的条件缩放比例
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用的方法，接收多个参数以控制生成过程
    def __call__(
        self,
        prompt_ids: jnp.ndarray,  # 输入的提示 ID 数组
        image: jnp.ndarray,  # 输入的图像数据
        params: Union[Dict, FrozenDict],  # 模型参数，可能是字典或不可变字典
        prng_seed: jax.Array,  # 随机种子，用于随机数生成
        num_inference_steps: int = 50,  # 默认推理步骤的数量为 50
        guidance_scale: Union[float, jnp.ndarray] = 7.5,  # 默认指导比例为 7.5
        latents: jnp.ndarray = None,  # 潜在变量，默认值为 None
        neg_prompt_ids: jnp.ndarray = None,  # 负提示 ID，默认值为 None
        controlnet_conditioning_scale: Union[float, jnp.ndarray] = 1.0,  # 默认控制网络的条件缩放比例为 1.0
        return_dict: bool = True,  # 默认返回字典格式
        jit: bool = False,  # 默认不启用 JIT 编译
# 静态参数为 pipe 和 num_inference_steps，任何更改都会触发重新编译。
# 非静态参数是（分片）输入张量，这些张量在它们的第一维上被映射（因此为 `0`）。
@partial(
    jax.pmap,  # 使用 JAX 的 pmap 并行映射功能
    in_axes=(None, 0, 0, 0, 0, None, 0, 0, 0, 0),  # 指定输入张量的轴
    static_broadcasted_argnums=(0, 5),  # 指定静态广播参数的索引
)
def _p_generate(  # 定义生成函数
    pipe,  # 生成管道对象
    prompt_ids,  # 提示 ID
    image,  # 输入图像
    params,  # 生成参数
    prng_seed,  # 随机数生成种子
    num_inference_steps,  # 推理步骤数
    guidance_scale,  # 指导尺度
    latents,  # 潜在变量
    neg_prompt_ids,  # 负提示 ID
    controlnet_conditioning_scale,  # 控制网条件尺度
):
    return pipe._generate(  # 调用生成管道的生成方法
        prompt_ids,  # 提示 ID
        image,  # 输入图像
        params,  # 生成参数
        prng_seed,  # 随机数生成种子
        num_inference_steps,  # 推理步骤数
        guidance_scale,  # 指导尺度
        latents,  # 潜在变量
        neg_prompt_ids,  # 负提示 ID
        controlnet_conditioning_scale,  # 控制网条件尺度
    )


@partial(jax.pmap, static_broadcasted_argnums=(0,))  # 使用 JAX 的 pmap，并指定静态广播参数
def _p_get_has_nsfw_concepts(pipe, features, params):  # 定义检查是否有 NSFW 概念的函数
    return pipe._get_has_nsfw_concepts(features, params)  # 调用管道的相关方法


def unshard(x: jnp.ndarray):  # 定义反分片函数，接受一个张量
    # einops.rearrange(x, 'd b ... -> (d b) ...')  # 注释掉的排列操作
    num_devices, batch_size = x.shape[:2]  # 获取设备数量和批量大小
    rest = x.shape[2:]  # 获取其余维度
    return x.reshape(num_devices * batch_size, *rest)  # 重新调整形状以合并设备和批量维度


def preprocess(image, dtype):  # 定义图像预处理函数
    image = image.convert("RGB")  # 将图像转换为 RGB 模式
    w, h = image.size  # 获取图像的宽和高
    w, h = (x - x % 64 for x in (w, h))  # 将宽高调整为64的整数倍
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])  # 调整图像大小，使用 Lanczos 插值法
    image = jnp.array(image).astype(dtype) / 255.0  # 转换为 NumPy 数组并归一化到 [0, 1]
    image = image[None].transpose(0, 3, 1, 2)  # 添加新维度并调整通道顺序
    return image  # 返回处理后的图像
```