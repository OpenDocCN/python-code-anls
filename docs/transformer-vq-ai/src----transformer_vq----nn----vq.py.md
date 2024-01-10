# `transformer_vq\src\transformer_vq\nn\vq.py`

```
"""
Helper class for VQ Attention.

Contains mostly static methods (for ease of unit testing).
"""
# 导入必要的库
import dataclasses
import chex
import flax.linen as nn
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
import jax.scipy as jsp
from flax import struct

# 导入自定义模块
from transformer_vq.nn.grad import sg
from transformer_vq.nn.grad import st
from transformer_vq.nn.norm import LayerNorm
from transformer_vq.nn.pe import get_sinusoid_embs
from transformer_vq.nn.types import TransformerConfig

# 定义一个数据类 VQSpec
@struct.dataclass
class VQSpec:
    n_device: jax.Array
    n_block_per_update: jax.Array
    loss_mask: jax.Array

    # 创建 VQSpec 实例的类方法
    @classmethod
    def create(cls, **kwargs):
        # 获取 VQSpec 类的字段名和类型
        signature = {field.name: field.type for field in dataclasses.fields(VQSpec)}
        # 过滤传入参数，保留符合 VQSpec 类字段的参数
        filtered = {k: v for k, v in kwargs.items() if k in signature}
        # 创建并返回 VQSpec 实例
        return cls(**filtered)

# 定义一个函数，用于计算输入向量与码本的最近邻索引和误差
def get_shortcodes(vecs, codebook):
    # 定义输入向量和码本的维度
    dims = chex.Dimensions(
        B=vecs.shape[0],
        H=vecs.shape[1],
        L=vecs.shape[2],
        S=codebook.shape[1],
        K=codebook.shape[2],
        i=1,
    )
    # 检查输入向量和码本的形状是否符合定义的维度
    chex.assert_shape(vecs, dims["BHLK"])
    chex.assert_shape(codebook, dims["HSK"])
    # 计算输入向量与码本的欧氏距离平方
    diffs2 = (
        jnp.expand_dims(jnp.sum(jnp.square(vecs), axis=-1), -1)
        - 2.0 * jnp.einsum("bhlk,hsk->bhls", vecs, codebook)
        + jnp.expand_dims(jnp.sum(jnp.square(codebook), axis=-1), (0, 2))
    )  # B, H, L, S
    # 获取最近邻索引
    z = jnp.argmin(diffs2, axis=-1)
    chex.assert_shape(z, dims["BHL"])
    # 计算最小误差
    errs2 = jnp.min(diffs2, axis=-1)
    # 对误差取 relu，如果使用无限精度，则不会改变
    errs2 = jax.nn.relu(errs2)
    chex.assert_shape(errs2, dims["BHL"])
    # 返回最近邻索引和误差
    return z.astype(jnp.int32), errs2

# 定义一个函数，用于获取最近邻索引对应的码本向量
def get_codewords(shortcodes, codebook):
    # 定义最近邻索引和码本的维度
    dims = chex.Dimensions(
        B=shortcodes.shape[0],
        H=shortcodes.shape[1],
        L=shortcodes.shape[2],
        S=codebook.shape[1],
        d=codebook.shape[2],
        i=1,
    )
    # 在最近邻索引上增加一个维度
    shortcodes = shortcodes[..., None]
    # 在码本上增加一个维度
    codebook = codebook[None, ...]
    # 使用 chex.assert_shape 函数检查 shortcodes 的形状是否符合预期的维度
    chex.assert_shape(shortcodes, dims["BHLi"])
    # 使用 chex.assert_shape 函数检查 codebook 的形状是否符合预期的维度
    chex.assert_shape(codebook, dims["iHSd"])
    # 从 codebook 中按照 shortcodes 中的索引值取出对应的数据，组成新的数组 cz
    cz = jnp.take_along_axis(codebook, indices=shortcodes, axis=2)
    # 返回新的数组 cz
    return cz
# 定义一个名为LearnableVQ的类，继承自nn.Module
class LearnableVQ(nn.Module):
    # 定义一个名为config的属性，类型为TransformerConfig
    config: TransformerConfig

    # 定义一个名为setup的方法
    def setup(self):
        # 调用apply_config方法
        self.apply_config()
        # 定义cs_args列表，包含self.w_init, [self.n_head, self.n_code, self.d_k], self.param_dtype
        cs_args = [self.w_init, [self.n_head, self.n_code, self.d_k], self.param_dtype]
        # 定义cc_args列表，包含init.ones, [self.n_head, self.n_code], self.param_dtype
        cc_args = [init.ones, [self.n_head, self.n_code], self.param_dtype]
        # 创建名为c_sum的参数，调用param方法，传入"c_sum", *cs_args
        self.c_sum = self.param("c_sum", *cs_args)
        # 创建名为c_count的参数，调用param方法，传入"c_count", *cc_args
        self.c_count = self.param("c_count", *cc_args)

    # 定义一个名为apply_config的方法
    def apply_config(self):
        # 遍历self.config的属性和值，使用setattr方法将值设置为对应的属性
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    # 定义一个静态方法_get_codebook，接受c_sum和c_count两个参数
    @staticmethod
    def _get_codebook(c_sum, c_count):
        # 计算c，c_sum除以c_count的每个元素，同时对c_count的值进行截断，最小值为0.01
        c = c_sum / jnp.clip(c_count[..., None], a_min=0.01)
        # 返回sg(c)的结果
        return sg(c)

    # 定义一个名为get_codebook的方法
    def get_codebook(self):
        # 调用静态方法_get_codebook，传入self.c_sum和self.c_count作为参数
        return LearnableVQ._get_codebook(self.c_sum, self.c_count)

    # 定义一个静态方法
    @staticmethod
    # 计算码本的 EMA 目标值
    def get_codebook_ema_targets(vecs, shortcodes, c_sum, c_count, c_gamma, vq_spec):
        # 获取码本的大小
        n_code = c_sum.shape[1]
        # 定义数据维度
        dims = chex.Dimensions(
            B=vecs.shape[0],
            L=vecs.shape[2],
            H=vecs.shape[1],
            d=vecs.shape[-1],
            S=c_sum.shape[1],
            i=1,
        )
        # 检查数据维度是否符合要求
        chex.assert_shape(vecs, dims["BHLd"])
        chex.assert_shape(shortcodes, dims["BHL"])
        chex.assert_shape(c_sum, dims["HSd"])
        chex.assert_shape(c_count, dims["HS"])
        chex.assert_shape(vq_spec.loss_mask, dims["BL"])
        g = c_gamma
        d = vq_spec.n_device
        p = vq_spec.n_block_per_update
        chex.assert_shape(d, dims["i"])
        chex.assert_shape(p, dims["i"])
        # 使用 r 计算按 shortcode 分组的运行统计数据
        # 由于我们还想排除使用右填充标记生成的向量，因此使用 loss_mask 作为启发式方法从运行统计数据中删除所有这样的向量
        r = jax.nn.one_hot(shortcodes, num_classes=n_code, dtype=vecs.dtype)
        r *= jnp.expand_dims(jnp.expand_dims(vq_spec.loss_mask, 1), -1)
        c_sum_hat = d * p * jnp.einsum("bhts,bhtd->hsd", r, vecs)
        c_count_hat = d * p * jnp.sum(r, axis=(0, 2))
        c_sum_tgt = (1 - g) * c_sum_hat + g * c_sum
        c_count_tgt = (1 - g) * c_count_hat + g * c_count
        chex.assert_shape(c_sum_tgt, dims["HSd"])
        chex.assert_shape(c_count_tgt, dims["HS"])
        return c_sum_tgt, c_count_tgt

    # 计算码本的损失
    @staticmethod
    def get_codebook_loss(
        vecs,
        shortcodes,
        c_sum,
        c_count,
        c_gamma,
        vq_spec,
        ):
        # the returned l_codebook gives correct updates
        # if gradients are averaged over devices and blocks
        # and codebook optimizer is sgd with lr = 1.0.
        # 计算输入向量的形状信息
        batch_size = vecs.shape[0]
        n_head = vecs.shape[1]
        block_len = vecs.shape[2]
        d_k = vecs.shape[3]
        n_code = c_count.shape[1]
        # 创建包含输入向量形状信息的 Dimensions 对象
        dims = chex.Dimensions(B=batch_size, H=n_head, L=block_len, d=d_k, S=n_code)
        # 获取用于指数移动平均的目标码书和计数
        c_sum_tgt, c_count_tgt = LearnableVQ.get_codebook_ema_targets(
            vecs=vecs,
            shortcodes=shortcodes,
            c_sum=c_sum,
            c_count=c_count,
            c_gamma=c_gamma,
            vq_spec=vq_spec,
        )
        # 断言目标码书和计数的形状
        chex.assert_shape(c_sum_tgt, dims["HSd"])
        chex.assert_shape(c_count_tgt, dims["HS"])
        # 计算码书损失
        l_codebook_sum = jnp.sum(sg(c_sum - c_sum_tgt) * st(c_sum))
        l_codebook_count = jnp.sum(sg(c_count - c_count_tgt) * st(c_count))
        l_codebook = l_codebook_count + l_codebook_sum
        # 返回码书损失
        return l_codebook

    @staticmethod
    # 定义一个方法，接受vecs和vq_spec作为参数
    def __call__(self, vecs, vq_spec):
        # 保存vecs的原始数据类型
        orig_dtype = vecs.dtype
        # 将vecs转换为指定的数据类型
        vecs_hp = vecs.astype(self.param_dtype)
        # 获取代码本
        c = LearnableVQ._get_codebook(self.c_sum, self.c_count)
        # 获取vecs的短码和误差平方
        z, errs2 = get_shortcodes(vecs=vecs_hp, codebook=c)
        # 将errs2转换为指定的数据类型
        errs2 = errs2.astype(self.dtype)
        # 获取短码对应的码字
        cz = get_codewords(shortcodes=z, codebook=c)
        # 将cz转换为vecs的原始数据类型
        cz = cz.astype(orig_dtype)
        # 重构量化后的vecs
        vecs_hat = sg(cz) + st(vecs)
        # 如果是训练模式
        if self.is_train:
            # 获取损失掩码
            loss_mask = vq_spec.loss_mask
            # 计算损失函数中的commitment loss
            l_commit = jnp.mean(jnp.sum(jnp.expand_dims(loss_mask, 1) * errs2, axis=1))
            # 计算损失函数中的codebook loss
            l_codebook = LearnableVQ.get_codebook_loss(
                vecs=vecs_hp,
                shortcodes=z,
                c_sum=self.c_sum,
                c_count=self.c_count,
                c_gamma=self.c_gamma,
                vq_spec=vq_spec,
            ).astype(self.dtype)
        # 如果不是训练模式
        else:
            # 设置commitment loss为0
            l_commit = jnp.zeros(dtype=self.dtype, shape=[])
            # 设置codebook loss为0
            l_codebook = jnp.zeros(dtype=self.dtype, shape=[])
        # 如果是训练模式
        if self.is_train:
            # 获取量化指标
            metrics = LearnableVQ.get_quantization_metrics(
                vecs=sg(vecs),
                vecs_hat=sg(vecs_hat),
                errs2=sg(errs2),
                c_sum=sg(self.c_sum),
                c_count=sg(self.c_count),
                dtype=self.dtype,
            )
        # 如果不是训练模式
        else:
            # 设置metrics为空字典
            metrics = dict()
        # 返回包含量化结果、短码、commitment loss、codebook loss、量化指标和errs2的字典
        return dict(
            quantized=vecs_hat,
            shortcodes=z,
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metrics,
            errs2=errs2,
        )
class SimpleVQ(nn.Module):
    # 定义一个简单的向量量化模型，继承自 nn.Module

    config: TransformerConfig
    # 声明一个名为 config 的属性，类型为 TransformerConfig

    def setup(self):
        # 定义一个初始化方法
        self.apply_config()
        # 调用 apply_config 方法
        self.tau = self.d_k**0.5
        # 计算 tau 值
        self.norm = LayerNorm(
            input_dim=self.d_k,
            param_dtype=self.param_dtype,
            center=False,
            norm=True,
            gain=False,
            bias=False,
        )
        # 初始化一个 LayerNorm 层

    def apply_config(self):
        # 定义一个应用配置的方法
        for k, v in dataclasses.asdict(self.config).items():
            # 遍历配置对象的属性和值
            setattr(self, k, v)
            # 设置当前对象的属性值为配置对象的属性值

    def get_codebook(self):
        # 定义一个获取码书的方法
        c = get_sinusoid_embs(
            length=self.n_code, width=self.d_k, start=0, lam=self.pe_lam, flip=False
        )
        # 调用获取正弦嵌入的方法，得到码书 c
        return (self.tau**-0.5) * sg(self.norm(c))[None, ...]
        # 返回经过处理的码书 c

    def __call__(self, vecs, vq_spec):
        # 定义一个调用方法，接受两个参数 vecs 和 vq_spec
        orig_dtype = vecs.dtype
        # 获取 vecs 的数据类型
        vecs_hp = vecs.astype(self.param_dtype)
        # 将 vecs 转换为指定的数据类型
        c = self.get_codebook()
        # 调用获取码书的方法，得到码书 c
        z, errs2 = get_shortcodes(vecs=vecs_hp, codebook=c)
        # 调用获取短码字的方法，得到短码字 z 和误差 errs2
        errs2 = errs2.astype(self.dtype)
        # 将 errs2 转换为指定的数据类型
        cz = get_codewords(shortcodes=z, codebook=c)
        # 调用获取码字的方法，得到码字 cz
        cz = cz.astype(orig_dtype)
        # 将码字 cz 转换为原始数据类型
        vecs_hat = sg(cz) + st(vecs)
        # 计算重构后的向量 vecs_hat
        if self.is_train:
            # 如果处于训练状态
            loss_mask = vq_spec.loss_mask
            # 获取损失掩码
            l_commit = jnp.mean(jnp.sum(jnp.expand_dims(loss_mask, 1) * errs2, axis=1))
            # 计算损失 l_commit
            l_codebook = jnp.zeros(dtype=self.dtype, shape=[])
            # 初始化码书损失 l_codebook
        else:
            # 如果不处于训练状态
            l_commit = jnp.zeros(dtype=self.dtype, shape=[])
            # 初始化损失 l_commit
            l_codebook = jnp.zeros(dtype=self.dtype, shape=[])
            # 初始化码书损失 l_codebook
        metrics = dict()
        # 初始化一个空的字典 metrics
        return dict(
            quantized=vecs_hat,
            shortcodes=z,
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metrics,
            errs2=errs2,
        )
        # 返回包含各种结果的字典
```