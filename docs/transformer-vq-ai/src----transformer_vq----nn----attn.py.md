# `transformer_vq\src\transformer_vq\nn\attn.py`

```
# 导入必要的模块
import dataclasses  # 用于创建不可变的类
import chex  # 用于 JAX 的检查工具
import flax.linen as nn  # Flax 深度学习库的模块
import jax  # 用于自动微分和并行计算的库
import jax.numpy as jnp  # JAX 的 NumPy 替代品
import numpy as np  # NumPy 数学库

# 从自定义模块中导入必要的函数和类
from transformer_vq.nn.grad import sg  # 导入自定义的梯度函数
from transformer_vq.nn.norm import LayerNorm  # 导入自定义的层归一化函数
from transformer_vq.nn.pe import get_sinusoid_embs  # 导入自定义的位置编码函数
from transformer_vq.nn.types import TransformerConfig  # 导入自定义的 Transformer 配置类
from transformer_vq.nn.vq import LearnableVQ  # 导入自定义的可学习的向量量化类

# 定义一个常量，用于近似表示无穷大的掩码值
MASK_INFTY_APPROX = 1e30

# 定义一个类，表示 VQAttention 模块，继承自 nn.Module
class VQAttention(nn.Module):
    # 传入 TransformerConfig 类型的参数
    config: TransformerConfig
# 设置函数，用于初始化模型参数
def setup(self):
    # 应用配置
    self.apply_config()
    # 计算 tau 值
    self.tau = self.d_k**0.5
    # 初始化输入层归一化
    self.input_ln = LayerNorm(self.d_model, self.param_dtype)
    # 计算 q、k、v 的通道数
    q_ch = self.n_head * self.d_k
    k_ch = self.n_head * self.d_k
    v_ch = self.n_head * self.d_v
    # 初始化 q、k 层归一化
    self.q_ln = LayerNorm(self.d_k, self.param_dtype, gain=False, bias=False)
    self.k_ln = LayerNorm(self.d_k, self.param_dtype, gain=False, bias=False)
    # 设置投影层参数
    proj_kwargs = dict(
        kernel_init=self.w_init,
        use_bias=False,
        param_dtype=self.param_dtype,
        dtype=self.dtype,
    )
    # 初始化 q、k、v 投影层
    self.q_proj = nn.Dense(q_ch, **proj_kwargs)
    self.kvg_proj = nn.Dense(k_ch + v_ch + v_ch, **proj_kwargs)
    self.r_proj = nn.Dense(k_ch, **proj_kwargs)
    # 更新投影层参数
    proj_kwargs.update(dict(kernel_init=self.r_init))
    # 初始化残差投影层
    self.res_proj = nn.Dense(self.d_model, **proj_kwargs)
# 初始化参数 xl_u，xl_v，分别表示u和v的参数，使用self.b_init作为初始值，维度为[q_ch]，数据类型为self.param_dtype
self.xl_u = self.param("u", self.b_init, [q_ch], self.param_dtype)
self.xl_v = self.param("v", self.b_init, [q_ch], self.param_dtype)

# 初始化可学习的向量量化器，使用self.config作为参数
self.quantizer = LearnableVQ(self.config)

# 初始化输入信号的dropout层，使用self.p_dropsin作为dropout概率，根据self.is_train确定是否为训练模式
self.dropsin = nn.Dropout(
    self.p_dropsin, rng_collection="timeless", deterministic=not self.is_train
)

# 初始化残差信号的dropout层，使用self.p_dropres作为dropout概率，根据self.is_train确定是否为训练模式
self.dropres = nn.Dropout(
    self.p_dropres, rng_collection="ephemeral", deterministic=not self.is_train
)

# 应用配置参数到当前对象
def apply_config(self):
    for k, v in dataclasses.asdict(self.config).items():
        setattr(self, k, v)

# 初始化模型的初始状态，根据配置参数和批量大小计算初始状态的相关参数
@staticmethod
def initial_state(config, batch_size):
    prefix = [batch_size, config.n_head]
    s = config.n_code
    m = config.mem_len
    d_k = config.d_k
        # 从配置中获取 d_v 的值
        d_v = config.d_v
        # 返回一个包含各种数据结构的字典
        return dict(
            # 创建一个名为 pos_offset 的数组，元素为 0，数据类型为 jnp.int32
            pos_offset=jnp.array(0, dtype=jnp.int32),
            # 创建一个名为 xlcache 的字典，包含多个数组
            xlcache=dict(
                # 创建一个形状为 [*prefix, m]，元素类型为 jnp.int32，填充值为 s 的数组 z
                z=jnp.full(shape=[*prefix, m], dtype=jnp.int32, fill_value=s),
                # 创建一个形状为 [*prefix, m, d_k]，元素类型为 config.param_dtype 的数组 k_hat
                k_hat=jnp.zeros([*prefix, m, d_k], dtype=config.param_dtype),
                # 创建一个形状为 [*prefix, m, d_v]，元素类型为 config.dtype 的数组 v
                v=jnp.zeros([*prefix, m, d_v], dtype=config.dtype),
                # 创建一个形状为 [batch_size, m]，元素类型为 jnp.int32 的数组 doc_ids
                doc_ids=jnp.zeros([batch_size, m], jnp.int32),
            ),
            # 创建一个名为 aggcache 的字典，包含多个数组
            aggcache=dict(
                # 创建一个形状为 [*prefix, s, d_v]，元素类型为 config.dtype 的数组 upper_div_lower
                upper_div_lower=jnp.zeros([*prefix, s, d_v], dtype=config.dtype),
                # 创建一个形状为 [*prefix, s]，元素类型为 config.dtype 的数组 lower
                lower=jnp.zeros([*prefix, s], dtype=config.dtype),
                # 创建一个形状为 [batch_size]，元素类型为 jnp.int32 的数组 latest_doc_id
                latest_doc_id=jnp.zeros([batch_size], jnp.int32),
            ),
        )

    @staticmethod
    def rel_shift(x):
        # 获取 x 的形状信息
        *leading_shape, present_len, past_len = x.shape
        # 定义填充规范
        pad_spec = [(0, 0)] * len(leading_shape) + [(0, 0), (1, 0)]
        # 对输入数据进行填充，使其符合指定的填充规范
        x = jnp.pad(x, pad_spec)
        # 对输入数据进行重塑，将其变换为指定形状
        x = jnp.reshape(x, [*leading_shape, past_len + 1, present_len])
        # 对输入数据进行切片，去除第一列数据
        x = x[..., 1:, :]
        # 对输入数据进行重塑，将其变换为指定形状
        x = jnp.reshape(x, [*leading_shape, present_len, past_len])
        # 返回处理后的数据
        return x

    @staticmethod
    def get_causal_mask(block_len, mem_len, invalid_len, with_locality):
        # 确保无效长度是一个 JAX 数组，以便与 jit 正常工作
        chex.assert_shape(invalid_len, [])
        # 确保块长度大于0，记忆长度大于等于0
        assert block_len > 0 and mem_len >= 0
        # 生成一个长度为 block_len 的数组 i
        i = jnp.arange(block_len)[..., None]
        # 生成一个长度为 mem_len + block_len 的数组 j
        j = jnp.arange(mem_len + block_len)[None, ...]
        # 生成一个分配掩码，标记哪些位置可以分配
        alloc_mask = jnp.greater_equal(j, jnp.array([invalid_len])[None, ...])
        # 生成一个因果掩码，标记哪些位置是因果的
        causal_mask = jnp.less_equal(j - mem_len, i)
        # 生成一个窗口掩码，标记哪些位置在窗口内
        window_mask = jnp.greater_equal(j, i)
        # 生成一个保留掩码，标记哪些位置需要保留
        keep_mask = jnp.logical_and(alloc_mask, causal_mask)
        # 如果需要考虑局部性，再生成一个局部性掩码
        if with_locality:
            keep_mask = jnp.logical_and(keep_mask, window_mask)
        # 返回最终的掩码
        return keep_mask
    # 静态方法，用于获取聚合偏差
    @staticmethod
    def get_agg_biases(lower):
        # 使用 jnp.where 函数根据条件返回相应值
        result = jnp.where(
            jnp.equal(lower, jnp.zeros_like(lower)),  # 如果 lower 等于零，则返回 -MASK_INFTY_APPROX
            -MASK_INFTY_APPROX,
            jnp.log(jnp.maximum(lower, jnp.ones_like(lower))),  # 否则返回 lower 的对数，确保不会出现 nan
        )
        return result

    # 获取查询向量 q
    def get_q(self, x_tilde):
        # 获取输入 x_tilde 的形状信息
        bsz, present_len, *_ = x_tilde.shape
        # 使用 self.q_proj 对 x_tilde 进行投影
        q = self.q_proj(x_tilde)
        # 将投影后的结果进行形状重塑
        q = jnp.reshape(q, [bsz, present_len, self.n_head, self.d_k])
        # 对重塑后的结果进行 Layer Normalization，并乘以缩放因子
        q = self.q_ln(q) * (self.tau**-0.5)
        # 对结果进行维度转置
        q = jnp.transpose(q, (0, 2, 1, 3))
        # 将结果转换为指定的参数数据类型
        return q.astype(self.param_dtype)

    # 获取键值对 g
    def get_kvg(self, x_tilde):
        # 获取输入 x_tilde 的形状信息
        bsz, present_len, *_ = x_tilde.shape
        # 使用 self.kvg_proj 方法处理输入 x_tilde，得到 k, v, g 三个向量
        kvg = self.kvg_proj(x_tilde)
        # 计算 k, v, g 向量的累积和
        inds = np.cumsum(np.array([self.d_k, self.d_v]))
        # 将 kvg 向量按照 d_k 和 d_v 的大小进行分割
        k, v, g = jnp.split(kvg, self.n_head * inds, axis=-1)
        # 检查 k, v, g 的形状是否符合要求
        chex.assert_shape(k, [bsz, present_len, self.n_head * self.d_k])
        chex.assert_shape(v, [bsz, present_len, self.n_head * self.d_v])
        chex.assert_shape(g, [bsz, present_len, self.n_head * self.d_v])
        # 将 k, v 向量重新调整形状，以便后续处理
        k = jnp.reshape(k, [bsz, present_len, self.n_head, self.d_k])
        v = jnp.reshape(v, [bsz, present_len, self.n_head, self.d_v])
        # 对 k 向量进行处理，并乘以 self.tau 的负半次方
        k = self.k_ln(k) * (self.tau**-0.5)
        # 对 v 向量进行处理，使用 jax.nn.silu 函数
        v = jax.nn.silu(v)
        # 对 g 向量进行处理，使用 jax.nn.silu 函数
        g = jax.nn.silu(g)
        # 对 k, v 向量进行转置操作
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        # 返回处理后的 k, v, g 向量
        return k.astype(self.param_dtype), v, g

    def get_xl_helpers(self):
        # 计算用于 xl 偏置的辅助值
        xl_r = get_sinusoid_embs(
            length=self.mem_len + self.block_len,
            width=self.d_model,
# 定义一个函数attn，用于计算注意力机制的输出
def attn(self, present_q, present_k, present_v, present_doc_ids, state, vq_spec):
    # 获取输入张量的批量大小
    bsz = present_q.shape[0]
    # 定义张量的维度
    dims = chex.Dimensions(
        B=bsz,
# 设置参数L、M、W、H、S、K、V、i的值
L=self.block_len,
M=self.mem_len,
W=self.mem_len + self.block_len,
H=self.n_head,
S=self.n_code,
K=self.d_k,
V=self.d_v,
i=1,

# 检查present_v、state["xlcache"]["v"]、state["aggcache"]["upper_div_lower"]、state["aggcache"]["lower"]的数据类型是否相等
chex.assert_trees_all_equal_dtypes(
    present_v,
    state["xlcache"]["v"],
    state["aggcache"]["upper_div_lower"],
    state["aggcache"]["lower"],
)

# 检查present_q的形状是否为BHLK
chex.assert_shape(present_q, dims["BHLK"])

# 检查present_k的形状是否为BHLK
chex.assert_shape(present_k, dims["BHLK"])

# 检查present_v的形状是否为BHLV
chex.assert_shape(present_v, dims["BHLV"])

# 量化键，计算指标，提交损失，并计算码书替代损失
# 使用量化器对当前的 k 进行量化，得到量化后的结果字典
vq_output_dict = self.quantizer(present_k, vq_spec=vq_spec)
# 从量化结果字典中获取 shortcodes、quantized、l_commit、l_codebook 和 metrics
present_z = vq_output_dict["shortcodes"]
present_k_hat = vq_output_dict["quantized"]
l_commit = vq_output_dict["l_commit"]
l_codebook = vq_output_dict["l_codebook"]
metrics = vq_output_dict["metrics"]
# 检查 present_z 和 present_k_hat 的形状是否符合预期
chex.assert_shape(present_z, dims["BHL"])
chex.assert_shape(present_k_hat, dims["BHLK"])
chex.assert_trees_all_equal_dtypes(present_k, present_k_hat)

# 将滑动窗口缓存 xlcache 和聚合缓存 aggcache 连接到当前块上
xlcache = state["xlcache"]
aggcache = state["aggcache"]
# 检查 xlcache 中 z、k_hat 和 v 的形状是否符合预期
chex.assert_shape(xlcache["z"], dims["BHM"])
chex.assert_shape(xlcache["k_hat"], dims["BHMK"])
chex.assert_shape(xlcache["v"], dims["BHMV"])
# 将 present_z、present_k_hat、present_v 和 present_doc_ids 连接到最近的缓存中
recent_z = jnp.concatenate([xlcache["z"], present_z], axis=-1)
recent_k_hat = jnp.concatenate([xlcache["k_hat"], present_k_hat], axis=-2)
recent_v = jnp.concatenate([xlcache["v"], present_v], axis=-2)
recent_doc_ids = jnp.concatenate([xlcache["doc_ids"], present_doc_ids], axis=-1)
# 检查 recent_z 的形状是否符合预期
chex.assert_shape(recent_z, dims["BHW"])
# 检查 recent_k_hat 的形状是否符合预期
chex.assert_shape(recent_k_hat, dims["BHWK"])
# 检查 recent_v 的形状是否符合预期
chex.assert_shape(recent_v, dims["BHWV"])

# 计算 xl 偏置的辅助变量
xl_r, xl_u, xl_v = self.get_xl_helpers()

# 计算 aggcache 分数
c = self.quantizer.get_codebook()
cache_scores = jnp.einsum("bhlk,hsk->bhls", present_q + xl_u, c)
# 获取 aggcache 的偏置
cache_biases = VQAttention.get_agg_biases(aggcache["lower"])
cache_biases = jnp.expand_dims(cache_biases, axis=-2)
cache_scores += cache_biases

# 计算最近分数（present 和 xlcache）
recent_scores_ac = jnp.einsum("bhlk,bhwk->bhlw", present_q + xl_u, recent_k_hat)
recent_scores_bd = jnp.einsum("bhlk,hwk->bhlw", present_q + xl_v, xl_r)
recent_scores_bd = VQAttention.rel_shift(recent_scores_bd)
recent_scores_bd *= VQAttention.get_causal_mask(
    block_len=self.block_len,
# 继续添加更多注释...
# 设置内存长度
mem_len=self.mem_len
# 计算无效长度，使用ReLU函数确保结果非负
invalid_len=jax.nn.relu(self.mem_len - state["pos_offset"])
# 设置是否使用本地性，如果为True，则在bubble中将动态偏置清零
with_locality=True
# 将结果转换为jnp.int32类型
[None, None, ...].astype(jnp.int32)

# 计算最近得分
recent_scores = recent_scores_ac + recent_scores_bd
# 获取因果掩码，用于限制注意力只能关注过去的位置
keep_mask = VQAttention.get_causal_mask(
    block_len=self.block_len,
    mem_len=self.mem_len,
    invalid_len=jax.nn.relu(self.mem_len - state["pos_offset"]),
    with_locality=not self.agg_cache  # 当agg cache为False时，关注bubble
)[None, None, ...].astype(jnp.int32)
# 通过掩码保留最近得分，同时将不需要的部分置为负无穷近似值
recent_scores = recent_scores * keep_mask - MASK_INFTY_APPROX * (1 - keep_mask)

# 为了稳定性，减去最大得分（由于softmax的平移不变性，这是可以的）
# 计算缓存最大得分和最近最大得分
cache_max_scores = jnp.max(cache_scores, axis=-1)  # BHL
recent_max_scores = jnp.max(recent_scores, axis=-1)  # BHL
# 计算最大得分
max_scores = sg(jnp.maximum(cache_max_scores, recent_max_scores))  # BHL
# 确保max_scores的形状符合预期
chex.assert_shape(max_scores, dims["BHL"])
# 减去最大得分，以提高稳定性
cache_scores -= max_scores[..., None]
recent_scores -= max_scores[..., None]
        # 计算缓存得分的指数值，并转换为指定的数据类型
        cache_a = jnp.exp(cache_scores).astype(self.dtype)
        # 计算最近得分的指数值，并转换为指定的数据类型
        recent_a = jnp.exp(recent_scores).astype(self.dtype)
        # 检查缓存得分的形状是否符合预期
        chex.assert_shape(cache_a, dims["BHLS"])
        # 检查最近得分的形状是否符合预期
        chex.assert_shape(recent_a, dims["BHLW"])

        # 计算每个查询的归一化因子 d，并首先将未归一化的权重 a 除以它，
        # 以确保不会出现数值不稳定的表达式 av。
        d = jnp.sum(recent_a, axis=-1)
        # 如果需要聚合缓存，将缓存得分也加入到归一化因子 d 中
        if self.agg_cache:
            d += jnp.sum(cache_a, axis=-1)
        # 计算加权值 wv，使用 Einstein 求和约定
        wv = jnp.einsum("bhlw,bhwv->bhlv", recent_a / d[..., None], recent_v)
        # 如果需要聚合缓存，将缓存得分也加入到加权值 wv 中
        if self.agg_cache:
            wv += jnp.einsum(
                "bhls,bhsv->bhlv", cache_a / d[..., None], aggcache["upper_div_lower"]
            )

        # 调整加权值 wv 的维度顺序
        wv = jnp.transpose(wv, (0, 2, 1, 3))
        # 重塑加权值 wv 的形状
        wv = jnp.reshape(wv, [bsz, self.block_len, self.n_head * self.d_v])
        # 返回包含注意力输出的字典
        return dict(
            attn_out=wv,
# 更新模型状态，传入最近的隐藏状态、注意力权重、值、文档ID、以及当前状态
def update_state(self, recent_z, recent_k_hat, recent_v, recent_doc_ids, state):
    # 获取最近隐藏状态的批量大小
    bsz, *_ = recent_z.shape
    # 定义维度信息，包括批量大小、块长度、记忆长度、编码数量、头数量、键的维度、值的维度
    dims = chex.Dimensions(
        B=bsz,
        L=self.block_len,
        M=self.mem_len,
        S=self.n_code,
        H=self.n_head,
        K=self.d_k,
        V=self.d_v,
    )
# 从状态中获取聚合缓存
aggcache = state["aggcache"]
# 检查聚合缓存中"upper_div_lower"的形状是否符合预期
chex.assert_shape(aggcache["upper_div_lower"], dims["BHSV"])
# 检查聚合缓存中"lower"的形状是否符合预期
chex.assert_shape(aggcache["lower"], dims["BHS"])
# 检查recent_z的形状是否符合预期
chex.assert_shape(recent_z[..., : -self.mem_len], dims["BHL"])
# 检查recent_v的形状是否符合预期
chex.assert_shape(recent_v[..., : -self.mem_len, :], dims["BHLV"])
# 检查recent_k_hat的形状是否符合预期
chex.assert_shape(recent_k_hat[..., : -self.mem_len, :], dims["BHLK"])
# 检查recent_z的形状是否符合预期
chex.assert_shape(recent_z[..., -self.mem_len :], dims["BHM"])
# 检查recent_v的形状是否符合预期
chex.assert_shape(recent_v[..., -self.mem_len :, :], dims["BHMV"])
# 检查recent_k_hat的形状是否符合预期
chex.assert_shape(recent_k_hat[..., -self.mem_len :, :], dims["BHMK"])
# 计算Kronecker delta；从xlcache初始化编码的无效z转换为零向量
delta = jax.nn.one_hot(
    recent_z[..., : -self.mem_len],
    num_classes=self.n_code,
    dtype=self.dtype,
    axis=-1,
)  # BHLS
# 计算新的位置偏移量
new_pos_offset = state["pos_offset"] + self.block_len
new_lower = jnp.add(aggcache["lower"], jnp.sum(delta, axis=-2))
# 计算更新的上限缓存变量（以相对格式存储以确保稳定性）
# 计算 new_upper_div_lower，即通过将轴 S 除以 new_lower 中的计数来计算
f1 = aggcache["lower"] / jnp.clip(new_lower, a_min=1.0)
# 计算 f2，即 delta 除以 jnp.clip(new_lower, a_min=1.0) 的计数
f2 = delta / jnp.expand_dims(jnp.clip(new_lower, a_min=1.0), -2)
# 计算 new_upper_div_lower，通过使用 f1 乘以 aggcache["upper_div_lower"] 和 f2 与 recent_v 的乘积的和
new_upper_div_lower = jnp.add(
    f1[..., None] * aggcache["upper_div_lower"],
    jnp.einsum("bhls,bhlv->bhsv", f2, recent_v[..., : -self.mem_len, :]),
)
# 创建新的状态字典
new_state = dict(
    pos_offset=new_pos_offset,
    xlcache=dict(
        z=recent_z[..., -self.mem_len :],
        k_hat=recent_k_hat[..., -self.mem_len :, :],
        v=recent_v[..., -self.mem_len :, :],
        doc_ids=recent_doc_ids[..., -self.mem_len :],
    ),
    aggcache=dict(
        lower=new_lower,
        upper_div_lower=new_upper_div_lower,
        latest_doc_id=recent_doc_ids[..., -self.mem_len - 1],
    ),
    # 如果不通过缓存，则对新状态进行变换
    if not self.grad_thru_cache:
        new_state = jax.tree_util.tree_map(sg, new_state)
    # 调用函数时，从输入字典中弹出"doc_ids"、"vq_spec"和"input_features"，分别赋值给对应变量
    doc_ids = input_dict.pop("doc_ids")
    vq_spec = input_dict.pop("vq_spec")
    x = input_dict.pop("input_features")
    # 对输入特征进行 Layer Normalization
    x_tilde = self.input_ln(x)
    # 获取 q 值
    q = self.get_q(x_tilde=x_tilde)
    # 获取 k、v、g 值
    k, v, g = self.get_kvg(x_tilde=x_tilde)
    # 进行注意力计算，得到注意力输出字典
    attn_output_dict = self.attn(q, k, v, doc_ids, state, vq_spec)
    # 获取注意力输出
    wv = attn_output_dict.get("attn_out")
    # 对注意力输出进行加权
    o = wv * g
    # 通过残差连接进行投影
    res = self.res_proj(o)
    # 对投影结果进行 dropout
    res = self.dropres(res)
    # 检查输入特征和投影结果的数据类型是否一致
    chex.assert_trees_all_equal_dtypes(x, res)
    # 更新状态，将最近的 z 值加入新状态中
    new_state = self.update_state(
        recent_z=attn_output_dict.get("recent_z"),
# 从 attn_output_dict 中获取 recent_k_hat 的数值，并赋给 recent_k_hat
recent_k_hat=attn_output_dict.get("recent_k_hat"),
# 从 attn_output_dict 中获取 recent_v 的数值，并赋给 recent_v
recent_v=attn_output_dict.get("recent_v"),
# 从 attn_output_dict 中获取 recent_doc_ids 的数值，并赋给 recent_doc_ids
recent_doc_ids=attn_output_dict.get("recent_doc_ids"),
# 将 state 传递给函数，并获取返回的 new_state
state=state,
)
# 创建一个包含 res、metrics、l_commit 和 l_codebook 的字典，并赋给 output_dict
output_dict = dict(
res=res,
metrics=attn_output_dict.get("metrics"),
l_commit=attn_output_dict.get("l_commit"),
l_codebook=attn_output_dict.get("l_codebook"),
)
# 返回 new_state 和 output_dict
return new_state, output_dict
```