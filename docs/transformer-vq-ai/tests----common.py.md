# `transformer_vq\tests\common.py`

```py
"""
Pytest fixtures. Due to the heterogeneity between tests, most fixtures are factories.
"""
# 导入操作系统模块
import os

# 设置本地设备数量
N_LOCAL_DEVICE = 8
# 设置环境变量 XLA_FLAGS，指定 XLA 强制使用的本地平台设备数量
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_LOCAL_DEVICE}"

# 导入 JAX 库
import jax
import jax.numpy as jnp
# 导入 pytest 库
import pytest
# 导入 dataclasses 模块
import dataclasses

# 导入自定义模块
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.vq import VQSpec
from transformer_vq.nn.attn import VQAttention
from transformer_vq.nn.model import Transformer

# 更新 JAX 配置，启用双精度浮点数
jax.config.update("jax_enable_x64", True)

# 定义宽化列表
WIDENINGS = [1]
# 定义数据类型列表
DTYPES = [jnp.float32]
# 定义容差字典
TOLERANCES = dict(atol=1e-5, rtol=1e-4)

# 生成长度元组
def gen_len_tuples(order, t=12):
    # order 是一个包含以下字符的字符串: {t, l, m}，分别对应序列长度、块长度和记忆长度
    tuples = []
    for m in range(1, t):
        if t % m == 0:
            for ell in range(1, t):
                if t % ell == 0:
                    dict_ = dict(t=t, l=ell, m=m)
                    list_ = []
                    for letter in order:
                        list_.append(dict_[letter])
                    tuple_ = tuple(list_)
                    tuples.append(tuple_)
    tuples = list(set(tuples))
    return tuples

# 定义 pytest fixture，用于生成随机数生成器
@pytest.fixture
def rng_fixture():
    def _rng(seed):
        return jax.random.PRNGKey(seed)

    return _rng

# 定义 pytest fixture，用于生成 transformer 配置
@pytest.fixture
def transformer_config_fixture():
    def _transformer_config(
        *,
        block_len,
        mem_len,
        agg_cache,
        widening,
        dtypes,
        is_train,
        n_vocab=10,
        c_gamma=0.99,
        no_emb=False,
        global_batch_size=None,  # 如果未使用，设置为 None
        sequence_len=None,  # 如果未使用，设置为 None
        update_len=None,  # 如果未使用，设置为 None
    # 创建 TransformerConfig 对象
    ):
        # 创建 TransformerConfig 对象并设置参数
        return TransformerConfig.create(
            param_dtype=dtypes,  # 设置参数数据类型
            dtype=dtypes,  # 设置数据类型
            global_batch_size=global_batch_size,  # 设置全局批量大小
            sequence_len=sequence_len,  # 设置序列长度
            update_len=update_len,  # 设置更新长度
            block_len=block_len,  # 设置块长度
            mem_len=mem_len,  # 设置记忆长度
            grad_thru_cache=True,  # 设置梯度通过缓存
            agg_cache=agg_cache,  # 设置聚合缓存
            d_model=4 * widening,  # 设置模型维度
            d_k=4 * widening,  # 设置键维度
            d_v=8 * widening,  # 设置值维度
            d_ff=0,  # 设置前馈网络维度
            n_head=2,  # 设置头数
            n_code=8,  # 设置编码数
            n_layer=2,  # 设置层数
            n_vocab=n_vocab,  # 设置词汇表大小
            pe_abs=True,  # 设置绝对位置编码
            pe_lam=10_000,  # 设置位置编码参数
            p_dropemb=0.0,  # 设置嵌入层丢弃率
            p_dropsin=0.0,  # 设置正弦位置编码丢弃率
            p_dropres=0.0,  # 设置残差连接丢弃率
            p_droplyr=0.0,  # 设置层丢弃率
            p_nucleus=0.8,  # 设置核心采样概率，仅用于采样
            c_beta=0.02,  # 设置 beta 参数
            c_gamma=c_gamma,  # 设置 gamma 参数
            e_tie=True,  # 设置嵌入层权重共享
            e_preln=True,  # 设置嵌入层前层归一化
            e_scale=1.0,  # 设置嵌入层缩放
            is_train=is_train,  # 设置是否训练
            no_emb=no_emb,  # 设置是否有嵌入层
        )

    return _transformer_config  # 返回 TransformerConfig 对象
# 定义一个多设备 VQ 规范的测试装置
@pytest.fixture
def multidev_vq_spec_fixture():
    # 定义一个内部函数，用于生成多设备 VQ 规范
    def _multidev_vq_spec(
        batch_size,
        block_len,
        n_device=N_LOCAL_DEVICE,
        n_local_device=N_LOCAL_DEVICE,
        n_update=1,
        n_block_per_update=1,
        update_id=0,
        block_id=0,
    ):
        # 断言批量大小能够被本地设备数整除
        assert batch_size % N_LOCAL_DEVICE == 0
        # 定义一个关键字参数字典
        kwargs = dict(shape=[n_device, 1], dtype=jnp.int32)
        # 创建一个多设备 VQ 规范对象并返回
        return VQSpec.create(
            n_device=jnp.full(fill_value=n_device, **kwargs),
            n_block_per_update=jnp.full(fill_value=n_block_per_update, **kwargs),
            loss_mask=jnp.ones(
                [N_LOCAL_DEVICE, batch_size // N_LOCAL_DEVICE, block_len], jnp.int32
            ),
        )

    return _multidev_vq_spec

# 定义一个基本 VQ 规范的测试装置
@pytest.fixture
def basic_vq_spec_fixture():
    # 定义一个内部函数，用于生成基本 VQ 规范
    def _basic_vq_spec(
        batch_size,
        block_len,
        n_update=1,
        n_block_per_update=1,
        update_id=0,
        block_id=0,
    ):
        # 创建一个基本 VQ 规范对象并返回
        return VQSpec.create(
            n_device=jnp.array([1]),
            n_block_per_update=jnp.array([n_block_per_update]),
            loss_mask=jnp.ones([batch_size, block_len], jnp.int32),
        )

    return _basic_vq_spec

# 定义一个注意力机制的测试装置，依赖于随机数生成器和基本 VQ 规范的测试装置
@pytest.fixture
def attn_fixture(rng_fixture, basic_vq_spec_fixture):
    # 定义一个函数_attn，接受一个TransformerConfig类型的参数config
    def _attn(config: TransformerConfig):
        # 对于初始化，需要使用不依赖于参数的初始状态变体，
        # 因此我们在这个装置内部覆盖了config以进行初始化。
        config = dict(dataclasses.asdict(config).items())  # 创建一个副本
        config.update({"is_train": True})  # 更新config中的"is_train"字段为True
        config = TransformerConfig.create(**config)  # 使用更新后的config创建TransformerConfig对象
        cls = VQAttention  # 将VQAttention类赋值给变量cls
        if hasattr(config, "attn_type"):  # 如果config中有"attn_type"属性
            atp = config.attn_type  # 将config中的"attn_type"属性赋值给变量atp
            if atp != "vq":  # 如果atp不等于"vq"
                raise ValueError(f"attention type {atp} not supported by fixture")  # 抛出数值错误，提示不支持的注意力类型
        present_len = config.block_len  # 将config中的block_len属性赋值给变量present_len
        # 使用cls创建一个实例，并初始化参数、状态和输入字典，得到参数params
        params = cls(config).init(
            dict(
                params=rng_fixture(1),  # 参数为rng_fixture(1)
                ephemeral=rng_fixture(2),  # ephemeral为rng_fixture(2)
                timeless=rng_fixture(3),  # timeless为rng_fixture(3)
            ),
            state=cls.initial_state(config=config, batch_size=1),  # 状态为使用config和batch_size=1创建的初始状态
            input_dict=dict(
                input_features=jax.random.normal(  # 输入特征为服从正态分布的随机数
                    rng_fixture(0), [1, present_len, config.d_model], dtype=config.dtype
                ),
                doc_ids=jnp.zeros([1, present_len], dtype=jnp.int32),  # doc_ids为全零数组
                vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=present_len),  # vq_spec为使用batch_size=1和block_len=present_len创建的基本vq_spec_fixture
            ),
        )["params"]  # 获取初始化后的参数
        return cls, params  # 返回cls和params

    return _attn  # 返回_attn函数
# 使用 pytest 的 fixture 装饰器定义一个名为 transformer_fixture 的测试装置
@pytest.fixture
def transformer_fixture(rng_fixture, basic_vq_spec_fixture):
    # 定义一个内部函数 _transformer，接受一个 config 参数
    def _transformer(config):
        # 为了初始化，需要使用不依赖于参数的初始状态变体，
        # 因此我们在这个 fixture 内部覆盖 config
        config = dict(dataclasses.asdict(config).items())  # 创建一个副本
        config.update({"is_train": True})  # 更新 config 中的 is_train 字段为 True
        config = TransformerConfig.create(**config)  # 使用更新后的 config 创建 TransformerConfig 对象
        present_len = config.block_len  # 获取 config 中的 block_len 字段值
        rngs = dict(
            params=rng_fixture(1),  # 使用 rng_fixture 生成一个名为 params 的随机数
            ephemeral=rng_fixture(2),  # 使用 rng_fixture 生成一个名为 ephemeral 的随机数
            timeless=rng_fixture(3),  # 使用 rng_fixture 生成一个名为 timeless 的随机数
        )
        if config.no_emb:  # 如果 config 中的 no_emb 字段为 True
            inputs = jax.random.normal(
                key=rng_fixture(0),  # 使用 rng_fixture 生成一个名为 key 的随机数
                shape=[1, present_len, config.d_model],  # 定义一个形状为 [1, present_len, config.d_model] 的数组
                dtype=config.dtype,  # 使用 config 中的 dtype 字段指定数据类型
            )
        else:
            inputs = jax.random.randint(
                key=rng_fixture(0),  # 使用 rng_fixture 生成一个名为 key 的随机数
                minval=0,  # 指定随机数的最小值为 0
                maxval=config.n_vocab,  # 指定随机数的最大值为 config.n_vocab
                shape=[1, present_len],  # 定义一个形状为 [1, present_len] 的数组
            )
        doc_ids = jnp.zeros([1, present_len], dtype=jnp.int32)  # 创建一个形状为 [1, present_len]，数据类型为 jnp.int32 的全零数组
        state = Transformer.initial_state(config=config, batch_size=1)  # 使用 config 和 batch_size=1 创建 Transformer 的初始状态
        params = Transformer(config).init(
            rngs,  # 传入随机数字典 rngs
            inputs=inputs,  # 传入输入数组 inputs
            doc_ids=doc_ids,  # 传入文档 ID 数组 doc_ids
            state=state,  # 传入初始状态 state
            vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=present_len),  # 传入基本的 vq_spec
        )["params"]  # 获取初始化后的参数
        return Transformer, params  # 返回 Transformer 类和参数

    return _transformer  # 返回内部函数 _transformer
```