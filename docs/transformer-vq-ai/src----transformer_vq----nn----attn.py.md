# `transformer_vq\src\transformer_vq\nn\attn.py`

```py
# 导入必要的模块
import dataclasses
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
# 从自定义模块中导入必要的类和函数
from transformer_vq.nn.grad import sg
from transformer_vq.nn.norm import LayerNorm
from transformer_vq.nn.pe import get_sinusoid_embs
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.vq import LearnableVQ

# 定义一个常量，用于近似表示无穷大的掩码值
MASK_INFTY_APPROX = 1e30  

# 定义一个类，表示 VQAttention 模块
class VQAttention(nn.Module):
    # 初始化方法，接受一个 TransformerConfig 对象作为参数
    config: TransformerConfig

    # 设置方法，用于初始化模块的各个属性
    def setup(self):
        # 应用配置，将配置中的属性赋值给当前对象
        self.apply_config()
        # 计算 tau 值，即 d_k 的平方根
        self.tau = self.d_k**0.5
        # 初始化输入层归一化对象
        self.input_ln = LayerNorm(self.d_model, self.param_dtype)
        # 计算 q、k、v 的通道数
        q_ch = self.n_head * self.d_k
        k_ch = self.n_head * self.d_k
        v_ch = self.n_head * self.d_v
        # 初始化 q、k 层归一化对象
        self.q_ln = LayerNorm(self.d_k, self.param_dtype, gain=False, bias=False)
        self.k_ln = LayerNorm(self.d_k, self.param_dtype, gain=False, bias=False)
        # 定义线性变换的参数
        proj_kwargs = dict(
            kernel_init=self.w_init,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        # 初始化 q、k、v 的线性变换对象
        self.q_proj = nn.Dense(q_ch, **proj_kwargs)
        self.kvg_proj = nn.Dense(k_ch + v_ch + v_ch, **proj_kwargs)
        self.r_proj = nn.Dense(k_ch, **proj_kwargs)
        proj_kwargs.update(dict(kernel_init=self.r_init))
        # 初始化残差连接的线性变换对象
        self.res_proj = nn.Dense(self.d_model, **proj_kwargs)
        # 初始化 u、v 参数
        self.xl_u = self.param("u", self.b_init, [q_ch], self.param_dtype)
        self.xl_v = self.param("v", self.b_init, [q_ch], self.param_dtype)
        # 初始化量化器对象
        self.quantizer = LearnableVQ(self.config)
        # 初始化输入的随机失活层和残差的随机失活层
        self.dropsin = nn.Dropout(
            self.p_dropsin, rng_collection="timeless", deterministic=not self.is_train
        )
        self.dropres = nn.Dropout(
            self.p_dropres, rng_collection="ephemeral", deterministic=not self.is_train
        )

    # 应用配置的静态方法
    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    # 静态方法
    @staticmethod
    # 初始化模型的状态
    def initial_state(config, batch_size):
        # 定义前缀
        prefix = [batch_size, config.n_head]
        # 获取配置中的参数
        s = config.n_code
        m = config.mem_len
        d_k = config.d_k
        d_v = config.d_v
        # 返回状态字典
        return dict(
            pos_offset=jnp.array(0, dtype=jnp.int32),  # 设置位置偏移量
            xlcache=dict(
                z=jnp.full(shape=[*prefix, m], dtype=jnp.int32, fill_value=s),  # 初始化 z
                k_hat=jnp.zeros([*prefix, m, d_k], dtype=config.param_dtype),  # 初始化 k_hat
                v=jnp.zeros([*prefix, m, d_v], dtype=config.dtype),  # 初始化 v
                doc_ids=jnp.zeros([batch_size, m], jnp.int32),  # 初始化 doc_ids
            ),
            aggcache=dict(
                upper_div_lower=jnp.zeros([*prefix, s, d_v], dtype=config.dtype),  # 初始化 upper_div_lower
                lower=jnp.zeros([*prefix, s], dtype=config.dtype),  # 初始化 lower
                latest_doc_id=jnp.zeros([batch_size], jnp.int32),  # 初始化 latest_doc_id
            ),
        )

    # 实现相对位移操作
    @staticmethod
    def rel_shift(x):
        *leading_shape, present_len, past_len = x.shape
        pad_spec = [(0, 0)] * len(leading_shape) + [(0, 0), (1, 0)]
        x = jnp.pad(x, pad_spec)  # 对输入进行填充
        x = jnp.reshape(x, [*leading_shape, past_len + 1, present_len])  # 重新塑形
        x = x[..., 1:, :]  # 对 x 进行切片操作
        x = jnp.reshape(x, [*leading_shape, present_len, past_len])  # 重新塑形
        return x  # 返回结果

    # 获取因果掩码
    @staticmethod
    def get_causal_mask(block_len, mem_len, invalid_len, with_locality):
        # invalid len 必须是 jax 数组才能与 jit 正常工作
        chex.assert_shape(invalid_len, [])
        assert block_len > 0 and mem_len >= 0  # 断言 block_len 大于 0，mem_len 大于等于 0
        i = jnp.arange(block_len)[..., None]  # 生成长度为 block_len 的数组 i
        j = jnp.arange(mem_len + block_len)[None, ...]  # 生成长度为 mem_len + block_len 的数组 j
        alloc_mask = jnp.greater_equal(j, jnp.array([invalid_len])[None, ...])  # 生成分配掩码
        causal_mask = jnp.less_equal(j - mem_len, i)  # 生成因果掩码
        window_mask = jnp.greater_equal(j, i)  # 生成窗口掩码
        keep_mask = jnp.logical_and(alloc_mask, causal_mask)  # 生成保留掩码
        if with_locality:
            keep_mask = jnp.logical_and(keep_mask, window_mask)  # 如果包含局部性，再次生成保留掩码
        return keep_mask  # 返回最终的保留掩码

    @staticmethod
    # 计算聚合偏差，根据 lower 的值进行条件判断
    def get_agg_biases(lower):
        result = jnp.where(
            jnp.equal(lower, jnp.zeros_like(lower)),
            -MASK_INFTY_APPROX,
            jnp.log(jnp.maximum(lower, jnp.ones_like(lower))),  # this is never nan
        )
        return result

    # 计算查询向量 q
    def get_q(self, x_tilde):
        bsz, present_len, *_ = x_tilde.shape
        q = self.q_proj(x_tilde)  # 使用 q_proj 对输入进行投影
        q = jnp.reshape(q, [bsz, present_len, self.n_head, self.d_k])  # 重新调整形状
        q = self.q_ln(q) * (self.tau**-0.5)  # 对 q 进行 LayerNorm 处理，并乘以缩放因子
        q = jnp.transpose(q, (0, 2, 1, 3))  # 调整维度顺序
        return q.astype(self.param_dtype)  # 转换数据类型为指定的参数数据类型

    # 计算键、值、门控向量
    def get_kvg(self, x_tilde):
        bsz, present_len, *_ = x_tilde.shape
        kvg = self.kvg_proj(x_tilde)  # 使用 kvg_proj 对输入进行投影
        inds = np.cumsum(np.array([self.d_k, self.d_v]))  # 计算索引增量
        k, v, g = jnp.split(kvg, self.n_head * inds, axis=-1)  # 按照指定索引增量进行分割
        chex.assert_shape(k, [bsz, present_len, self.n_head * self.d_k])  # 检查 k 的形状
        chex.assert_shape(v, [bsz, present_len, self.n_head * self.d_v])  # 检查 v 的形状
        chex.assert_shape(g, [bsz, present_len, self.n_head * self.d_v])  # 检查 g 的形状
        k = jnp.reshape(k, [bsz, present_len, self.n_head, self.d_k])  # 重新调整 k 的形状
        v = jnp.reshape(v, [bsz, present_len, self.n_head, self.d_v])  # 重新调整 v 的形状
        k = self.k_ln(k) * (self.tau**-0.5)  # 对 k 进行 LayerNorm 处理，并乘以缩放因子
        v = jax.nn.silu(v)  # 对 v 进行 SiLU 激活函数处理
        g = jax.nn.silu(g)  # 对 g 进行 SiLU 激活函数处理
        k = jnp.transpose(k, (0, 2, 1, 3))  # 调整维度顺序
        v = jnp.transpose(v, (0, 2, 1, 3))  # 调整维度顺序
        return k.astype(self.param_dtype), v, g  # 返回转换数据类型后的 k、v、g
    def get_xl_helpers(self):
        # 计算 XL 偏置的辅助函数（z dai et al., 2019）
        # 获取正弦嵌入
        xl_r = get_sinusoid_embs(
            length=self.mem_len + self.block_len,  # 长度为记忆长度加块长度
            width=self.d_model,  # 宽度为模型维度
            lam=self.pe_lam,  # 正弦嵌入的参数
            flip=True,  # 是否翻转
        )
        xl_r = self.dropsin(xl_r)  # 对正弦嵌入进行丢弃操作
        xl_r = self.r_proj(xl_r)  # 对正弦嵌入进行投影操作
        xl_r = jnp.reshape(xl_r, [self.mem_len + self.block_len, self.n_head, self.d_k])  # 重新塑形
        xl_r = jnp.transpose(xl_r, (1, 0, 2))  # 转置
        xl_r = xl_r.astype(self.param_dtype) * (self.tau**-0.5)  # 类型转换并乘以缩放因子
        xl_u = jnp.reshape(self.xl_u, [1, self.n_head, 1, self.d_k]) * (self.tau**-0.5)  # 重新塑形并乘以缩放因子
        xl_v = jnp.reshape(self.xl_v, [1, self.n_head, 1, self.d_k]) * (self.tau**-0.5)  # 重新塑形并乘以缩放因子
        return xl_r, xl_u, xl_v  # 返回计算得到的结果

    def __call__(self, state, input_dict):
        doc_ids = input_dict.pop("doc_ids")  # 弹出键为"doc_ids"的值
        vq_spec = input_dict.pop("vq_spec")  # 弹出键为"vq_spec"的值
        x = input_dict.pop("input_features")  # 弹出键为"input_features"的值
        x_tilde = self.input_ln(x)  # 对输入进行 Layer Normalization
        q = self.get_q(x_tilde=x_tilde)  # 获取查询向量
        k, v, g = self.get_kvg(x_tilde=x_tilde)  # 获取键、值和门控向量
        attn_output_dict = self.attn(q, k, v, doc_ids, state, vq_spec)  # 进行注意力计算
        wv = attn_output_dict.get("attn_out")  # 获取注意力输出
        o = wv * g  # 注意力输出乘以门控向量
        res = self.res_proj(o)  # 对输出进行投影
        res = self.dropres(res)  # 对投影后的输出进行丢弃操作
        chex.assert_trees_all_equal_dtypes(x, res)  # 检查输入和输出的数据类型是否一致
        new_state = self.update_state(
            recent_z=attn_output_dict.get("recent_z"),  # 更新状态中的最近 z 值
            recent_k_hat=attn_output_dict.get("recent_k_hat"),  # 更新状态中的最近 k_hat 值
            recent_v=attn_output_dict.get("recent_v"),  # 更新状态中的最近 v 值
            recent_doc_ids=attn_output_dict.get("recent_doc_ids"),  # 更新状态中的最近 doc_ids 值
            state=state,  # 当前状态
        )
        output_dict = dict(
            res=res,  # 输出结果
            metrics=attn_output_dict.get("metrics"),  # 输出的指标
            l_commit=attn_output_dict.get("l_commit"),  # 输出的 l_commit
            l_codebook=attn_output_dict.get("l_codebook"),  # 输出的 l_codebook
        )
        return new_state, output_dict  # 返回更新后的状态和输出字典
```