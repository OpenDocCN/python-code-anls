# `.\diffusers\pipelines\stable_diffusion\safety_checker_flax.py`

```py
# 版权声明，表示此代码归 HuggingFace 团队所有，保留所有权利。
# 
# 根据 Apache 2.0 许可证（“许可证”）授权；
# 除非遵循该许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件是以“原样”基础分发的，
# 不提供任何明示或暗示的保证或条件。
# 有关许可证下特定语言的权限和限制，请参阅许可证。

# 导入可选类型和元组类型
from typing import Optional, Tuple

# 导入 jax 库及其 numpy 模块
import jax
import jax.numpy as jnp
# 导入 flax 的模块和相关类
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
# 导入 CLIP 配置和模型基类
from transformers import CLIPConfig, FlaxPreTrainedModel
# 导入 CLIP 视觉模块
from transformers.models.clip.modeling_flax_clip import FlaxCLIPVisionModule


# 定义计算两个嵌入之间的余弦距离的函数
def jax_cosine_distance(emb_1, emb_2, eps=1e-12):
    # 归一化第一个嵌入，防止除以零
    norm_emb_1 = jnp.divide(emb_1.T, jnp.clip(jnp.linalg.norm(emb_1, axis=1), a_min=eps)).T
    # 归一化第二个嵌入，防止除以零
    norm_emb_2 = jnp.divide(emb_2.T, jnp.clip(jnp.linalg.norm(emb_2, axis=1), a_min=eps)).T
    # 返回两个归一化嵌入的点积，作为余弦相似度
    return jnp.matmul(norm_emb_1, norm_emb_2.T)


# 定义 Flax 稳定扩散安全检查器模块的类
class FlaxStableDiffusionSafetyCheckerModule(nn.Module):
    # 定义配置和数据类型属性
    config: CLIPConfig
    dtype: jnp.dtype = jnp.float32

    # 设置模块的组件
    def setup(self):
        # 初始化视觉模型
        self.vision_model = FlaxCLIPVisionModule(self.config.vision_config)
        # 定义视觉投影层，使用无偏置和指定数据类型
        self.visual_projection = nn.Dense(self.config.projection_dim, use_bias=False, dtype=self.dtype)

        # 定义概念嵌入的参数，初始化为全1的矩阵
        self.concept_embeds = self.param("concept_embeds", jax.nn.initializers.ones, (17, self.config.projection_dim))
        # 定义特殊关怀嵌入的参数，初始化为全1的矩阵
        self.special_care_embeds = self.param(
            "special_care_embeds", jax.nn.initializers.ones, (3, self.config.projection_dim)
        )

        # 定义概念嵌入权重的参数，初始化为全1的向量
        self.concept_embeds_weights = self.param("concept_embeds_weights", jax.nn.initializers.ones, (17,))
        # 定义特殊关怀嵌入权重的参数，初始化为全1的向量
        self.special_care_embeds_weights = self.param("special_care_embeds_weights", jax.nn.initializers.ones, (3,))
    # 定义调用方法，接受输入片段
        def __call__(self, clip_input):
            # 通过视觉模型处理输入片段，获取池化输出
            pooled_output = self.vision_model(clip_input)[1]
            # 将池化输出映射到图像嵌入
            image_embeds = self.visual_projection(pooled_output)
    
            # 计算图像嵌入与特殊关怀嵌入之间的余弦距离
            special_cos_dist = jax_cosine_distance(image_embeds, self.special_care_embeds)
            # 计算图像嵌入与概念嵌入之间的余弦距离
            cos_dist = jax_cosine_distance(image_embeds, self.concept_embeds)
    
            # 增加该值可创建更强的 `nfsw` 过滤器
            # 但可能会增加过滤良性图像输入的可能性
            adjustment = 0.0
    
            # 计算特殊关怀分数，考虑权重和调整值
            special_scores = special_cos_dist - self.special_care_embeds_weights[None, :] + adjustment
            # 将特殊关怀分数四舍五入到小数点后三位
            special_scores = jnp.round(special_scores, 3)
            # 判断是否有特殊关怀分数大于0
            is_special_care = jnp.any(special_scores > 0, axis=1, keepdims=True)
            # 如果图像有任何特殊关怀概念，使用较低的阈值
            special_adjustment = is_special_care * 0.01
    
            # 计算概念分数，考虑权重和特殊调整值
            concept_scores = cos_dist - self.concept_embeds_weights[None, :] + special_adjustment
            # 将概念分数四舍五入到小数点后三位
            concept_scores = jnp.round(concept_scores, 3)
            # 判断是否有 nfsw 概念分数大于0
            has_nsfw_concepts = jnp.any(concept_scores > 0, axis=1)
    
            # 返回是否包含 nfsw 概念的布尔值
            return has_nsfw_concepts
# 定义一个 Flax 稳定扩散安全检查器类，继承自 FlaxPreTrainedModel
class FlaxStableDiffusionSafetyChecker(FlaxPreTrainedModel):
    # 指定配置类为 CLIPConfig
    config_class = CLIPConfig
    # 指定主输入名称为 "clip_input"
    main_input_name = "clip_input"
    # 指定模块类为 FlaxStableDiffusionSafetyCheckerModule
    module_class = FlaxStableDiffusionSafetyCheckerModule

    # 初始化方法，接受配置和其他参数
    def __init__(
        self,
        config: CLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 如果输入形状未提供，使用默认值 (1, 224, 224, 3)
        if input_shape is None:
            input_shape = (1, 224, 224, 3)
        # 创建模块实例，传入配置和数据类型
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化权重的方法
    def init_weights(self, rng: jax.Array, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 生成输入张量，使用随机正态分布
        clip_input = jax.random.normal(rng, input_shape)

        # 分割随机数生成器以获得不同的随机种子
        params_rng, dropout_rng = jax.random.split(rng)
        # 定义随机种子字典
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模块参数
        random_params = self.module.init(rngs, clip_input)["params"]

        # 返回初始化的参数
        return random_params

    # 定义调用方法
    def __call__(
        self,
        clip_input,
        params: dict = None,
    ):
        # 转置输入张量的维度
        clip_input = jnp.transpose(clip_input, (0, 2, 3, 1))

        # 应用模块并返回结果
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(clip_input, dtype=jnp.float32),
            rngs={},
        )
```