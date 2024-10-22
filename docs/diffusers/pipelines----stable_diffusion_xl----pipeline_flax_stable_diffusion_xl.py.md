# `.\diffusers\pipelines\stable_diffusion_xl\pipeline_flax_stable_diffusion_xl.py`

```py
# 版权所有 2024 The HuggingFace Team. 保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下地址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件在“按现状”基础上分发，
# 不提供任何形式的明示或暗示的保证或条件。
# 请参阅许可证以获取有关权限和限制的具体语言。

# 从 functools 模块导入 partial 函数，用于部分应用函数
from functools import partial
# 从 typing 模块导入类型提示，便于类型检查
from typing import Dict, List, Optional, Union

# 导入 jax 库，用于高性能数值计算
import jax
# 导入 jax.numpy，提供类似于 NumPy 的数组操作
import jax.numpy as jnp
# 从 flax.core 导入 FrozenDict，提供不可变字典的实现
from flax.core.frozen_dict import FrozenDict
# 从 transformers 导入 CLIPTokenizer 和 FlaxCLIPTextModel，用于文本编码
from transformers import CLIPTokenizer, FlaxCLIPTextModel

# 从 diffusers.utils 导入 logging 模块，用于日志记录
from diffusers.utils import logging

# 从相对路径导入 FlaxAutoencoderKL 和 FlaxUNet2DConditionModel 模型
from ...models import FlaxAutoencoderKL, FlaxUNet2DConditionModel
# 从相对路径导入各种调度器
from ...schedulers import (
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
# 从相对路径导入 FlaxDiffusionPipeline 基类
from ..pipeline_flax_utils import FlaxDiffusionPipeline
# 从相对路径导入 FlaxStableDiffusionXLPipelineOutput 类
from .pipeline_output import FlaxStableDiffusionXLPipelineOutput

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 设置为 True 将使用 Python 循环而不是 jax.fori_loop，方便调试
DEBUG = False


# 定义 FlaxStableDiffusionXLPipeline 类，继承自 FlaxDiffusionPipeline
class FlaxStableDiffusionXLPipeline(FlaxDiffusionPipeline):
    # 初始化方法，设置模型及其相关参数
    def __init__(
        self,
        text_encoder: FlaxCLIPTextModel,  # 文本编码器模型 1
        text_encoder_2: FlaxCLIPTextModel,  # 文本编码器模型 2
        vae: FlaxAutoencoderKL,  # 变分自编码器模型
        tokenizer: CLIPTokenizer,  # 文本标记器 1
        tokenizer_2: CLIPTokenizer,  # 文本标记器 2
        unet: FlaxUNet2DConditionModel,  # UNet 模型
        scheduler: Union[
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],  # 调度器，可选多种类型
        dtype: jnp.dtype = jnp.float32,  # 数据类型，默认为 float32
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置数据类型属性
        self.dtype = dtype

        # 注册模型模块到管道
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        # 计算变分自编码器的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    # 准备输入的方法，处理文本提示
    def prepare_inputs(self, prompt: Union[str, List[str]]):
        # 检查 prompt 的类型是否为字符串或列表
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 假设有两个编码器
        inputs = []  # 初始化输入列表
        # 遍历两个标记器
        for tokenizer in [self.tokenizer, self.tokenizer_2]:
            # 使用标记器处理输入提示，返回填充后的输入 ID
            text_inputs = tokenizer(
                prompt,
                padding="max_length",  # 填充到最大长度
                max_length=self.tokenizer.model_max_length,  # 最大长度设置为标记器的最大值
                truncation=True,  # 截断超出最大长度的部分
                return_tensors="np",  # 返回 NumPy 格式的张量
            )
            # 将输入 ID 添加到输入列表
            inputs.append(text_inputs.input_ids)
        # 将输入 ID 堆叠为一个数组，按轴 1 组合
        inputs = jnp.stack(inputs, axis=1)
        # 返回准备好的输入
        return inputs
    # 定义调用方法，接受多个参数用于生成图像
    def __call__(
        self,
        prompt_ids: jax.Array,  # 输入的提示 ID 数组
        params: Union[Dict, FrozenDict],  # 模型参数字典
        prng_seed: jax.Array,  # 随机数种子，用于生成随机数
        num_inference_steps: int = 50,  # 推理步骤数量，默认值为 50
        guidance_scale: Union[float, jax.Array] = 7.5,  # 引导比例，默认值为 7.5
        height: Optional[int] = None,  # 生成图像的高度，默认为 None
        width: Optional[int] = None,  # 生成图像的宽度，默认为 None
        latents: jnp.array = None,  # 潜在变量，默认为 None
        neg_prompt_ids: jnp.array = None,  # 负提示 ID 数组，默认为 None
        return_dict: bool = True,  # 是否返回字典格式，默认为 True
        output_type: str = None,  # 输出类型，默认为 None
        jit: bool = False,  # 是否启用 JIT 编译，默认为 False
    ):
        # 0. 默认高度和宽度设置为 unet 配置的样本大小乘以 VAE 缩放因子
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 如果引导比例为浮点数且启用 JIT 编译
        if isinstance(guidance_scale, float) and jit:
            # 将引导比例转换为张量，确保每个设备都有副本
            guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
            guidance_scale = guidance_scale[:, None]  # 增加维度以便于后续计算

        # 检查是否返回潜在变量
        return_latents = output_type == "latent"

        # 根据是否启用 JIT 调用不同的生成函数
        if jit:
            images = _p_generate(
                self,
                prompt_ids,
                params,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                latents,
                neg_prompt_ids,
                return_latents,
            )
        else:
            images = self._generate(
                prompt_ids,
                params,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                latents,
                neg_prompt_ids,
                return_latents,
            )

        # 如果不返回字典格式，直接返回图像
        if not return_dict:
            return (images,)

        # 返回包含生成图像的输出对象
        return FlaxStableDiffusionXLPipelineOutput(images=images)

    # 定义获取嵌入的方法
    def get_embeddings(self, prompt_ids: jnp.array, params):
        # 假设我们有两个编码器

        # bs, encoder_input, seq_length
        te_1_inputs = prompt_ids[:, 0, :]  # 获取第一个编码器的输入
        te_2_inputs = prompt_ids[:, 1, :]  # 获取第二个编码器的输入

        # 使用第一个文本编码器生成嵌入，输出隐藏状态
        prompt_embeds = self.text_encoder(te_1_inputs, params=params["text_encoder"], output_hidden_states=True)
        prompt_embeds = prompt_embeds["hidden_states"][-2]  # 获取倒数第二个隐藏状态
        prompt_embeds_2_out = self.text_encoder_2(
            te_2_inputs, params=params["text_encoder_2"], output_hidden_states=True
        )  # 使用第二个文本编码器生成嵌入
        prompt_embeds_2 = prompt_embeds_2_out["hidden_states"][-2]  # 获取倒数第二个隐藏状态
        text_embeds = prompt_embeds_2_out["text_embeds"]  # 获取文本嵌入
        # 将两个嵌入沿最后一个维度拼接
        prompt_embeds = jnp.concatenate([prompt_embeds, prompt_embeds_2], axis=-1)
        return prompt_embeds, text_embeds  # 返回拼接后的嵌入和文本嵌入

    # 定义获取添加时间 ID 的方法
    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, bs, dtype):
        # 将原始大小、裁剪坐标和目标大小组合成一个列表
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        # 将列表转换为指定数据类型的数组，并复制 bs 次
        add_time_ids = jnp.array([add_time_ids] * bs, dtype=dtype)
        return add_time_ids  # 返回添加时间 ID 的数组
    # 定义生成函数，接收多个参数
        def _generate(
            # 输入提示的 ID 数组
            self,
            prompt_ids: jnp.array,
            # 模型参数，可以是字典或冷冻字典
            params: Union[Dict, FrozenDict],
            # 随机种子，用于生成随机数
            prng_seed: jax.Array,
            # 推理步骤的数量
            num_inference_steps: int,
            # 生成图像的高度
            height: int,
            # 生成图像的宽度
            width: int,
            # 引导比例，用于控制生成质量
            guidance_scale: float,
            # 可选的潜在变量数组，默认为 None
            latents: Optional[jnp.array] = None,
            # 可选的负提示 ID 数组，默认为 None
            neg_prompt_ids: Optional[jnp.array] = None,
            # 返回潜在变量的标志，默认为 False
            return_latents=False,
# 静态参数是管道、推理步数、高度、宽度和返回潜在变量。任何更改都会触发重新编译。
# 非静态参数是输入张量，映射到其第一个维度（因此为 `0`）。
@partial(
    jax.pmap,  # 使用 JAX 的并行映射函数
    in_axes=(None, 0, 0, 0, None, None, None, 0, 0, 0, None),  # 指定输入参数在并行处理时的轴
    static_broadcasted_argnums=(0, 4, 5, 6, 10),  # 指定静态广播参数的索引
)
def _p_generate(  # 定义生成函数，处理图像生成任务
    pipe,  # 管道对象，用于生成图像
    prompt_ids,  # 提示词 ID 的张量输入
    params,  # 生成所需的参数
    prng_seed,  # 随机数种子，用于生成随机性
    num_inference_steps,  # 推理步数，决定生成过程的细节
    height,  # 输出图像的高度
    width,  # 输出图像的宽度
    guidance_scale,  # 引导比例，控制生成的质量
    latents,  # 潜在变量，用于生成过程中的信息
    neg_prompt_ids,  # 负提示词 ID 的张量输入
    return_latents,  # 是否返回潜在变量的标志
):
    return pipe._generate(  # 调用管道的生成方法
        prompt_ids,  # 传入提示词 ID
        params,  # 传入生成参数
        prng_seed,  # 传入随机数种子
        num_inference_steps,  # 传入推理步数
        height,  # 传入输出图像高度
        width,  # 传入输出图像宽度
        guidance_scale,  # 传入引导比例
        latents,  # 传入潜在变量
        neg_prompt_ids,  # 传入负提示词 ID
        return_latents,  # 传入是否返回潜在变量的标志
    )
```