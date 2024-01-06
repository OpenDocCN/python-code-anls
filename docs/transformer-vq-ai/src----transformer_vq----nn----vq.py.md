# `transformer_vq\src\transformer_vq\nn\vq.py`

```
"""
VQ 注意力的辅助类。

主要包含静态方法（便于单元测试）。
"""
# 导入必要的库
import dataclasses  # 用于创建不可变的类
import chex  # 用于对 JAX 数组进行类型检查
import flax.linen as nn  # Flax 模块
import jax  # JAX 库
import jax.nn.initializers as init  # JAX 中的初始化器
import jax.numpy as jnp  # JAX 中的 NumPy 替代品
import jax.scipy as jsp  # JAX 中的 SciPy 替代品
from flax import struct  # Flax 中的结构

# 导入自定义模块
from transformer_vq.nn.grad import sg  # 梯度相关的自定义模块
from transformer_vq.nn.grad import st  # 梯度相关的自定义模块
from transformer_vq.nn.norm import LayerNorm  # 自定义的层归一化模块
from transformer_vq.nn.pe import get_sinusoid_embs  # 获取正弦位置编码的方法
from transformer_vq.nn.types import TransformerConfig  # 自定义的 Transformer 配置类型
# 定义一个名为VQSpec的数据类，包含n_device、n_block_per_update和loss_mask三个属性
@struct.dataclass
class VQSpec:
    n_device: jax.Array  # 表示设备数量的数组
    n_block_per_update: jax.Array  # 表示每次更新的块数的数组
    loss_mask: jax.Array  # 表示损失掩码的数组

    # 创建VQSpec对象的类方法，接受任意关键字参数
    @classmethod
    def create(cls, **kwargs):
        # 获取VQSpec类的属性签名
        signature = {field.name: field.type for field in dataclasses.fields(VQSpec)}
        # 过滤出符合属性签名的关键字参数
        filtered = {k: v for k, v in kwargs.items() if k in signature}
        # 使用过滤后的参数创建VQSpec对象
        return cls(**filtered)


# 定义一个名为get_shortcodes的函数，接受vecs和codebook两个参数
def get_shortcodes(vecs, codebook):
    # 创建一个包含维度信息的Dimensions对象，包括B、H和L三个维度
    dims = chex.Dimensions(
        B=vecs.shape[0],  # B维度表示批量大小
        H=vecs.shape[1],  # H维度表示向量的高度
        L=vecs.shape[2],  # L维度表示向量的长度
# 定义一个函数，用于计算向量和码本之间的差异，并返回最小差异的索引和最小差异值
def get_codewords(shortcodes, codebook):
# 创建一个包含指定维度的 Dimensions 对象，用于描述数据的形状
dims = chex.Dimensions(
    B=shortcodes.shape[0],  # B 维度为 shortcodes 的第一个维度
    H=shortcodes.shape[1],  # H 维度为 shortcodes 的第二个维度
    L=shortcodes.shape[2],  # L 维度为 shortcodes 的第三个维度
    S=codebook.shape[1],    # S 维度为 codebook 的第二个维度
    d=codebook.shape[2],    # d 维度为 codebook 的第三个维度
    i=1,                    # i 维度为 1
)

# 在 shortcodes 的最后一个维度添加一个新的维度
shortcodes = shortcodes[..., None]

# 在 codebook 的第一个维度添加一个新的维度
codebook = codebook[None, ...]

# 检查 shortcodes 的形状是否符合指定的 dims
chex.assert_shape(shortcodes, dims["BHLi"])

# 检查 codebook 的形状是否符合指定的 dims
chex.assert_shape(codebook, dims["iHSd"])

# 从 codebook 中取出指定索引的数据，形成新的数组
cz = jnp.take_along_axis(codebook, indices=shortcodes, axis=2)

# 返回结果数组
return cz
# 应用配置参数
self.apply_config()
# 初始化 cs_args 列表
cs_args = [self.w_init, [self.n_head, self.n_code, self.d_k], self.param_dtype]
# 初始化 cc_args 列表
cc_args = [init.ones, [self.n_head, self.n_code], self.param_dtype]
# 创建 c_sum 参数
self.c_sum = self.param("c_sum", *cs_args)
# 创建 c_count 参数
self.c_count = self.param("c_count", *cc_args)

# 应用配置参数的方法
def apply_config(self):
    # 遍历配置参数字典，将其键值对设置为对象的属性
    for k, v in dataclasses.asdict(self.config).items():
        setattr(self, k, v)

# 获取 codebook 的静态方法
@staticmethod
def _get_codebook(c_sum, c_count):
    # 计算 codebook
    c = c_sum / jnp.clip(c_count[..., None], a_min=0.01)
    return sg(c)

# 获取 codebook 的方法
def get_codebook(self):
    return LearnableVQ._get_codebook(self.c_sum, self.c_count)

# 获取 codebook 的 EMA 目标的静态方法
@staticmethod
def get_codebook_ema_targets(vecs, shortcodes, c_sum, c_count, c_gamma, vq_spec):
# 获取编码的数量
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
# 检查向量的形状是否符合定义的维度
chex.assert_shape(vecs, dims["BHLd"])
chex.assert_shape(shortcodes, dims["BHL"])
chex.assert_shape(c_sum, dims["HSd"])
chex.assert_shape(c_count, dims["HS"])
chex.assert_shape(vq_spec.loss_mask, dims["BL"])
# 将 c_gamma 赋值给 g
g = c_gamma
# 将 vq_spec.n_device 赋值给 d
d = vq_spec.n_device
# 将 vq_spec.n_block_per_update 赋值给 p
p = vq_spec.n_block_per_update
# 检查 d 和 p 的形状是否符合定义的维度
chex.assert_shape(d, dims["i"])
chex.assert_shape(p, dims["i"])
# 可以使用 r 计算按 shortcode 分组的运行统计信息，见下文。
        # 由于我们还想排除使用右填充标记生成的向量，我们使用损失掩码作为启发式方法从运行统计数据中删除所有这样的向量。
        # 使用短码生成独热编码，num_classes为编码数量，dtype为向量的数据类型
        r = jax.nn.one_hot(shortcodes, num_classes=n_code, dtype=vecs.dtype)
        # 将独热编码乘以损失掩码，以排除使用右填充标记生成的向量
        r *= jnp.expand_dims(jnp.expand_dims(vq_spec.loss_mask, 1), -1)
        # 计算加权和，使用矩阵乘法计算
        c_sum_hat = d * p * jnp.einsum("bhts,bhtd->hsd", r, vecs)
        # 计算向量数量的加权和
        c_count_hat = d * p * jnp.sum(r, axis=(0, 2))
        # 计算目标加权和，使用混合参数g
        c_sum_tgt = (1 - g) * c_sum_hat + g * c_sum
        # 计算目标向量数量的加权和，使用混合参数g
        c_count_tgt = (1 - g) * c_count_hat + g * c_count
        # 断言目标加权和的形状符合预期
        chex.assert_shape(c_sum_tgt, dims["HSd"])
        # 断言目标向量数量的加权和的形状符合预期
        chex.assert_shape(c_count_tgt, dims["HS"])
        # 返回目标加权和和目标向量数量的加权和
        return c_sum_tgt, c_count_tgt

    @staticmethod
    def get_codebook_loss(
        vecs,
        shortcodes,
        c_sum,
        c_count,
        c_gamma,
        vq_spec,
        # 计算批量大小
        batch_size = vecs.shape[0]
        # 计算头数
        n_head = vecs.shape[1]
        # 计算块长度
        block_len = vecs.shape[2]
        # 计算向量维度
        d_k = vecs.shape[3]
        # 计算编码数量
        n_code = c_count.shape[1]
        # 创建维度对象
        dims = chex.Dimensions(B=batch_size, H=n_head, L=block_len, d=d_k, S=n_code)
        # 获取用于更新的目标码书和计数
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
        # 开启格式化，对下面的代码进行格式化处理
        l_codebook_sum = jnp.sum(sg(c_sum - c_sum_tgt) * st(c_sum))
        # 计算码书和目标码书之间的差异，然后对结果进行处理
        l_codebook_count = jnp.sum(sg(c_count - c_count_tgt) * st(c_count))
        # 计算码书数量和目标码书数量之间的差异，然后对结果进行处理
        l_codebook = l_codebook_count + l_codebook_sum
        # 返回码书的总体损失
        return l_codebook

    @staticmethod
    def get_quantization_metrics(vecs, vecs_hat, errs2, c_sum, c_count, dtype):
        # 在返回语句中调用了 stop gradients，因此此处不需要调用
        n_head, n_code = c_count.shape[0], c_count.shape[1]
        eps, errmin, errmax, maskval = 1e-2, 0e1, 1e1, 1e30
        # 对码书数量进行截断处理
        c_count = jnp.clip(c_count, a_min=eps)
        # 计算码书的均值
        c = c_sum / c_count[..., None]  # HSd
        # 对码书的范数进行截断处理
        c_norms = jnp.clip(jnp.linalg.norm(c, axis=-1), a_min=eps)  # HS
        # 对码书进行归一化处理
        c_normed = c / c_norms[..., None]  # HSd
        # 计算码书之间的相似度
        c_sims = jnp.einsum("hsd,hzd->hsz", c_normed, c_normed)  # HSS
        # 计算码书之间的距离
        c_dists = jnp.linalg.norm(
            jnp.expand_dims(c, 2) - jnp.expand_dims(c, 1), axis=-1
        )  # HSS
        # 对向量的范数进行截断处理
        vec_norms = jnp.clip(jnp.linalg.norm(vecs, axis=-1), a_min=eps)  # BHL
        # 计算向量的范数，并对结果进行截断
        vec_hat_norms = jnp.clip(jnp.linalg.norm(vecs_hat, axis=-1), a_min=eps)  # BHL
        # 计算 errs2 的平方根
        errs = jnp.sqrt(errs2)  # BHL
        # 计算相对误差，并对结果进行截断
        relative_errs = jnp.clip(errs / vec_norms, errmin, errmax)  # BHL
        # 计算概率
        probs = c_count / jnp.sum(c_count, axis=-1)[..., None]  # HS
        # 判断 c_count 是否超出阈值，并将结果转换为 float32 类型
        c_thresh_oob = jnp.logical_or(c_count < 1.0, 1_000_000 < c_count)
        c_thresh_oob = c_thresh_oob.astype(jnp.float32)

        # 创建全为1的矩阵，并生成上三角和下三角的掩码
        ones = jnp.ones([1, n_code, n_code], dtype=jnp.float32)
        up = jnp.triu(ones)  # upper triangular ones mask
        low = jnp.tril(ones, k=-1)  # strict lower triangular ones mask
        # 计算各种指标的最小值、平均值和最大值
        metrics = dict(
            c_sim_min=jnp.min(low * c_sims + maskval * up, axis=(1, 2)),  # [H]
            c_sim_mean=jnp.sum(low * c_sims, axis=(1, 2)) / jnp.sum(low, axis=(1, 2)),
            c_sim_max=jnp.max(low * c_sims - maskval * up, axis=(1, 2)),  # [H]
            c_dist_min=jnp.min(low * c_dists + maskval * up, axis=(1, 2)),  # [H]
            c_dist_mean=jnp.sum(low * c_dists, axis=(1, 2)) / jnp.sum(low, axis=(1, 2)),
            c_dist_max=jnp.max(low * c_dists - maskval * up, axis=(1, 2)),  # [H]
            c_norm_min=jnp.min(c_norms, axis=1),  # [H]
        c_norm_mean=jnp.mean(c_norms, axis=1),  # 计算 c_norms 按行的均值，返回形状为 [H] 的数组
        c_norm_max=jnp.max(c_norms, axis=1),  # 计算 c_norms 按行的最大值，返回形状为 [H] 的数组
        c_usage_min=jnp.min(c_count, axis=1),  # 计算 c_count 按行的最小值，返回形状为 [H] 的数组
        c_usage_mean=jnp.mean(c_count, axis=1),  # 计算 c_count 按行的均值，返回形状为 [H] 的数组
        c_usage_max=jnp.max(c_count, axis=1),  # 计算 c_count 按行的最大值，返回形状为 [H] 的数组
        c_thresh_oob=jnp.sum(c_thresh_oob, axis=1),  # 计算 c_thresh_oob 按行的和，返回形状为 [H] 的数组
        c_entropy=jnp.sum(jsp.special.entr(probs), axis=-1),  # 计算 probs 的熵，返回形状为 [H] 的数组
        vec_norm_mean=jnp.mean(vec_norms, axis=2),  # 计算 vec_norms 按第二个维度的均值，返回形状为 [B, H] 的数组
        vec_hat_norm_mean=jnp.mean(vec_hat_norms, axis=2),  # 计算 vec_hat_norms 按第二个维度的均值，返回形状为 [B, H] 的数组
        relative_err_min=jnp.min(relative_errs, axis=2),  # 计算 relative_errs 按第二个维度的最小值，返回形状为 [B, H] 的数组
        relative_err_mean=jnp.mean(relative_errs, axis=2),  # 计算 relative_errs 按第二个维度的均值，返回形状为 [B, H] 的数组
        relative_err_max=jnp.max(relative_errs, axis=2),  # 计算 relative_errs 按第二个维度的最大值，返回形状为 [B, H] 的数组
    )
    return jax.tree_util.tree_map(lambda x: jnp.mean(sg(x)).astype(dtype), metrics)  # 对 metrics 中的每个元素应用 jnp.mean 和 sg 函数，然后转换为指定的数据类型

def __call__(self, vecs, vq_spec):
    orig_dtype = vecs.dtype  # 获取 vecs 的数据类型
    vecs_hp = vecs.astype(self.param_dtype)  # 将 vecs 转换为指定的数据类型
    c = LearnableVQ._get_codebook(self.c_sum, self.c_count)  # 获取 codebook
    z, errs2 = get_shortcodes(vecs=vecs_hp, codebook=c)  # 调用 get_shortcodes 函数，传入参数 vecs_hp 和 codebook，并返回结果 z 和 errs2
# 将errs2转换为指定的数据类型
errs2 = errs2.astype(self.dtype)
# 根据给定的shortcodes和codebook获取codewords
cz = get_codewords(shortcodes=z, codebook=c)
# 将cz转换为原始数据类型
cz = cz.astype(orig_dtype)
# 计算vecs_hat，其中sg(cz)表示cz的sg函数值，st(vecs)表示vecs的st函数值
vecs_hat = sg(cz) + st(vecs)
# 如果是训练阶段
if self.is_train:
    # 获取loss_mask
    loss_mask = vq_spec.loss_mask
    # 计算l_commit
    l_commit = jnp.mean(jnp.sum(jnp.expand_dims(loss_mask, 1) * errs2, axis=1))
    # 计算l_codebook
    l_codebook = LearnableVQ.get_codebook_loss(
        vecs=vecs_hp,
        shortcodes=z,
        c_sum=self.c_sum,
        c_count=self.c_count,
        c_gamma=self.c_gamma,
        vq_spec=vq_spec,
    ).astype(self.dtype)
# 如果不是训练阶段
else:
    # l_commit置为0
    l_commit = jnp.zeros(dtype=self.dtype, shape=[])
    # l_codebook置为0
    l_codebook = jnp.zeros(dtype=self.dtype, shape=[])
# 如果是训练阶段
if self.is_train:
    # 获取quantization_metrics
    metrics = LearnableVQ.get_quantization_metrics(
# 设置vecs的sg值
vecs=sg(vecs),
# 设置vecs_hat的sg值
vecs_hat=sg(vecs_hat),
# 设置errs2的sg值
errs2=sg(errs2),
# 设置c_sum的sg值
c_sum=sg(self.c_sum),
# 设置c_count的sg值
c_count=sg(self.c_count),
# 设置dtype的值
dtype=self.dtype,
# 如果条件不成立，创建一个空的metrics字典
metrics = dict()
# 返回一个包含quantized、shortcodes、l_commit、l_codebook、metrics和errs2的字典
return dict(
    quantized=vecs_hat,
    shortcodes=z,
    l_commit=l_commit,
    l_codebook=l_codebook,
    metrics=metrics,
    errs2=errs2,
)
    # 定义一个变量config，类型为TransformerConfig
    config: TransformerConfig

    # 初始化方法，调用apply_config()方法，设置tau和norm变量，创建LayerNorm对象
    def setup(self):
        self.apply_config()
        self.tau = self.d_k**0.5
        self.norm = LayerNorm(
            input_dim=self.d_k,
            param_dtype=self.param_dtype,
            center=False,
            norm=True,
            gain=False,
            bias=False,
        )

    # 应用配置的方法，遍历config对象的属性，并将其设置为当前对象的属性
    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    # 获取codebook的方法
    def get_codebook(self):
        c = get_sinusoid_embs(
    # 定义一个函数，接受vecs和vq_spec两个参数
    def __call__(self, vecs, vq_spec):
        # 保存vecs的原始数据类型
        orig_dtype = vecs.dtype
        # 将vecs转换为self.param_dtype类型
        vecs_hp = vecs.astype(self.param_dtype)
        # 获取编码簿
        c = self.get_codebook()
        # 获取短码和错误平方
        z, errs2 = get_shortcodes(vecs=vecs_hp, codebook=c)
        # 将errs2转换为self.dtype类型
        errs2 = errs2.astype(self.dtype)
        # 获取编码词
        cz = get_codewords(shortcodes=z, codebook=c)
        # 将cz转换为原始数据类型
        cz = cz.astype(orig_dtype)
        # 重构vecs_hat
        vecs_hat = sg(cz) + st(vecs)
        # 如果是训练状态
        if self.is_train:
            # 获取损失掩码
            loss_mask = vq_spec.loss_mask
            # 计算l_commit
            l_commit = jnp.mean(jnp.sum(jnp.expand_dims(loss_mask, 1) * errs2, axis=1))
            # 初始化l_codebook
            l_codebook = jnp.zeros(dtype=self.dtype, shape=[])
        # 如果不是训练状态
        else:
            # 初始化l_commit
            l_commit = jnp.zeros(dtype=self.dtype, shape=[])
            # 初始化l_codebook
            l_codebook = jnp.zeros(dtype=self.dtype, shape=[])
# 创建一个空的字典用于存储指标数据
metrics = dict()
# 返回一个包含 quantized、shortcodes、l_commit、l_codebook、metrics 和 errs2 的字典
return dict(
    quantized=vecs_hat,  # 将vecs_hat存储在字典中的quantized键下
    shortcodes=z,  # 将z存储在字典中的shortcodes键下
    l_commit=l_commit,  # 将l_commit存储在字典中的l_commit键下
    l_codebook=l_codebook,  # 将l_codebook存储在字典中的l_codebook键下
    metrics=metrics,  # 将metrics存储在字典中的metrics键下
    errs2=errs2,  # 将errs2存储在字典中的errs2键下
)
```