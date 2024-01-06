# `transformer_vq\tests\common.py`

```
"""
Pytest fixtures. Due to the heterogeneity between tests, most fixtures are factories.
"""
# 导入操作系统模块
import os

# 设置本地设备数量为8
N_LOCAL_DEVICE = 8
# 设置环境变量 XLA_FLAGS 为指定的本地设备数量
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_LOCAL_DEVICE}"

# 导入 jax 库
import jax
# 导入 jax 的 numpy 模块，并重命名为 jnp
import jax.numpy as jnp
# 导入 pytest 库
import pytest
# 导入 dataclasses 模块
import dataclasses

# 从 transformer_vq.nn.types 模块中导入 TransformerConfig 类
from transformer_vq.nn.types import TransformerConfig
# 从 transformer_vq.nn.vq 模块中导入 VQSpec 类
from transformer_vq.nn.vq import VQSpec
# 从 transformer_vq.nn.attn 模块中导入 VQAttention 类
from transformer_vq.nn.attn import VQAttention
# 从 transformer_vq.nn.model 模块中导入 Transformer 类
from transformer_vq.nn.model import Transformer

# 更新 jax 的配置，启用 x64 模式
jax.config.update("jax_enable_x64", True)
# 定义一组扩展因子，这里只有一个值为1的扩展因子
WIDENINGS = [1]
# 定义一组数据类型，这里只有一个数据类型为 jnp.float32
DTYPES = [jnp.float32]
# 定义一个容差字典，包括绝对容差和相对容差
TOLERANCES = dict(atol=1e-5, rtol=1e-4)

# 定义一个生成长度元组的函数，参数 order 是一个包含字符 {t, l, m} 的字符串，分别代表序列长度、块长度和内存长度
def gen_len_tuples(order, t=12):
    # 初始化一个空列表用于存储生成的元组
    tuples = []
    # 遍历内存长度的可能取值
    for m in range(1, t):
        # 如果总长度能被内存长度整除
        if t % m == 0:
            # 遍历块长度的可能取值
            for ell in range(1, t):
                # 如果总长度能被块长度整除
                if t % ell == 0:
                    # 构建一个包含 t、l、m 三个键值对的字典
                    dict_ = dict(t=t, l=ell, m=m)
                    # 初始化一个空列表用于存储当前顺序对应的值
                    list_ = []
                    # 遍历 order 中的字符
                    for letter in order:
                        # 将对应键的值添加到列表中
                        list_.append(dict_[letter])
                    # 将列表转换为元组并添加到结果列表中
                    tuple_ = tuple(list_)
                    tuples.append(tuple_)
# 将列表转换为集合，然后再转换为列表，去除重复元素
tuples = list(set(tuples))
# 返回去重后的列表
return tuples

# 定义一个测试夹具，用于生成随机数生成器
@pytest.fixture
def rng_fixture():
    # 定义内部函数，接受种子参数，返回随机数生成器
    def _rng(seed):
        return jax.random.PRNGKey(seed)
    # 返回内部函数
    return _rng

# 定义一个测试夹具，用于生成变压器配置
@pytest.fixture
def transformer_config_fixture():
    # 定义内部函数，接受关键字参数，返回变压器配置
    def _transformer_config(
        *,
        block_len,
        mem_len,
        agg_cache,
        widening,
# 定义函数参数，包括数据类型、是否为训练集、词汇量、c_gamma值、是否使用嵌入层、全局批量大小、序列长度、更新长度
def function_name(
        dtypes,
        is_train,
        n_vocab=10,
        c_gamma=0.99,
        no_emb=False,
        global_batch_size=None,  # 如果未使用，设置为None
        sequence_len=None,  # 如果未使用，设置为None
        update_len=None,  # 如果未使用，设置为None
    ):
        # 创建TransformerConfig对象，设置参数数据类型、全局批量大小、序列长度、更新长度、块长度、记忆长度、梯度通过缓存、聚合缓存、d_model值
        return TransformerConfig.create(
            param_dtype=dtypes,
            dtype=dtypes,
            global_batch_size=global_batch_size,
            sequence_len=sequence_len,
            update_len=update_len,
            block_len=block_len,
            mem_len=mem_len,
            grad_thru_cache=True,
            agg_cache=agg_cache,
            d_model=4 * widening,
# 定义注意力机制中的查询向量的维度
d_k=4 * widening,
# 定义注意力机制中的数值向量的维度
d_v=8 * widening,
# 定义前馈神经网络中隐藏层的维度
d_ff=0,
# 定义注意力头的数量
n_head=2,
# 定义编码器和解码器中的编码器数量
n_code=8,
# 定义编码器和解码器中的层数
n_layer=2,
# 定义词汇表的大小
n_vocab=n_vocab,
# 定义位置编码是否使用绝对位置编码
pe_abs=True,
# 定义位置编码的参数
pe_lam=10_000,
# 定义词嵌入层的dropout概率
p_dropemb=0.0,
# 定义sinusoidal位置编码的dropout概率
p_dropsin=0.0,
# 定义残差连接的dropout概率
p_dropres=0.0,
# 定义每个层的dropout概率
p_droplyr=0.0,
# 定义用于采样的nucleus采样的参数
p_nucleus=0.8,  # 仅用于采样
# 定义beta参数，用于控制残差连接的权重
c_beta=0.02,
# 定义gamma参数，用于控制残差连接的权重
c_gamma=c_gamma,
# 定义词嵌入层和输出层是否共享权重
e_tie=True,
# 定义是否在层归一化之前应用残差连接
e_preln=True,
# 定义缩放因子，用于缩放输出
e_scale=1.0,
# 定义是否处于训练状态
is_train=is_train,
# 定义一个名为multidev_vq_spec_fixture的测试夹具
@pytest.fixture
# 定义一个名为_multidev_vq_spec的函数，用于生成多设备VQ规范
def multidev_vq_spec_fixture():
    # 定义_multidev_vq_spec函数，接受多个参数
    def _multidev_vq_spec(
        batch_size,  # 批处理大小
        block_len,   # 块长度
        n_device=N_LOCAL_DEVICE,  # 设备数量，默认为N_LOCAL_DEVICE
        n_local_device=N_LOCAL_DEVICE,  # 本地设备数量，默认为N_LOCAL_DEVICE
        n_update=1,  # 更新次数，默认为1
        n_block_per_update=1,  # 每次更新的块数，默认为1
        update_id=0,  # 更新ID，默认为0
        block_id=0,   # 块ID，默认为0
    ):
        # 断言批处理大小能够被N_LOCAL_DEVICE整除
        assert batch_size % N_LOCAL_DEVICE == 0
        # 创建一个包含shape和dtype的字典
        kwargs = dict(shape=[n_device, 1], dtype=jnp.int32)
# 创建一个 VQSpec 对象，使用给定的参数
return VQSpec.create(
    # 创建一个由 n_device 值填充的数组
    n_device=jnp.full(fill_value=n_device, **kwargs),
    # 创建一个由 n_block_per_update 值填充的数组
    n_block_per_update=jnp.full(fill_value=n_block_per_update, **kwargs),
    # 创建一个全为 1 的数组，用作损失掩码
    loss_mask=jnp.ones(
        [N_LOCAL_DEVICE, batch_size // N_LOCAL_DEVICE, block_len], jnp.int32
    ),
)

# 返回一个函数，用于生成基本的 VQSpec 对象
return _multidev_vq_spec

# 定义一个基本的 VQSpec fixture
@pytest.fixture
def basic_vq_spec_fixture():
    # 定义一个函数，用于生成基本的 VQSpec 对象
    def _basic_vq_spec(
        batch_size,
        block_len,
        n_update=1,
        n_block_per_update=1,
        update_id=0,
        block_id=0,
# 定义一个名为attn_fixture的测试夹具，依赖于rng_fixture和basic_vq_spec_fixture
@pytest.fixture
def attn_fixture(rng_fixture, basic_vq_spec_fixture):
    # 定义一个名为_attn的函数，接受一个TransformerConfig类型的参数config
    def _attn(config: TransformerConfig):
        # 对于初始化，需要使用不依赖于参数的初始状态变体，因此我们在这个夹具内部覆盖config
        config = dict(dataclasses.asdict(config).items())  # 创建一个副本
        config.update({"is_train": True})  # 更新config中的"is_train"字段为True
        config = TransformerConfig.create(**config)  # 使用更新后的config创建一个TransformerConfig对象
        cls = VQAttention  # 定义cls为VQAttention类
        # 如果config具有"attn_type"属性
        if hasattr(config, "attn_type"):
            # 返回一个VQSpec对象，使用VQSpec.create方法创建
            return VQSpec.create(
                n_device=jnp.array([1]),
                n_block_per_update=jnp.array([n_block_per_update]),
                loss_mask=jnp.ones([batch_size, block_len], jnp.int32),
            )

    return _attn
# 获取配置中的注意力类型
atp = config.attn_type
# 如果注意力类型不是 "vq"，则抛出数值错误
if atp != "vq":
    raise ValueError(f"attention type {atp} not supported by fixture")
# 获取配置中的块长度
present_len = config.block_len
# 使用类方法初始化参数
params = cls(config).init(
    dict(
        params=rng_fixture(1),
        ephemeral=rng_fixture(2),
        timeless=rng_fixture(3),
    ),
    state=cls.initial_state(config=config, batch_size=1),
    input_dict=dict(
        input_features=jax.random.normal(
            rng_fixture(0), [1, present_len, config.d_model], dtype=config.dtype
        ),
        doc_ids=jnp.zeros([1, present_len], dtype=jnp.int32),
        vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=present_len),
    ),
)["params"]
# 返回类和参数
return cls, params
# 返回 _attn 变量
    return _attn

# 定义一个名为 transformer_fixture 的测试装置，依赖于 rng_fixture 和 basic_vq_spec_fixture
@pytest.fixture
def transformer_fixture(rng_fixture, basic_vq_spec_fixture):
    # 定义一个名为 _transformer 的函数，接受一个 config 参数
    def _transformer(config):
        # 对于初始化，需要使用不依赖于参数的初始状态变体，
        # 因此我们在这个装置内部覆盖 config 以进行初始化。
        # 复制一份 config 对象
        config = dict(dataclasses.asdict(config).items())  # make a copy
        # 更新 config 内部的 "is_train" 属性为 True
        config.update({"is_train": True})
        # 创建一个新的 TransformerConfig 对象
        config = TransformerConfig.create(**config)
        # 获取 present_len 属性值
        present_len = config.block_len
        # 创建一个随机数生成器字典
        rngs = dict(
            params=rng_fixture(1),
            ephemeral=rng_fixture(2),
            timeless=rng_fixture(3),
        )
        # 如果 config.no_emb 为真
        if config.no_emb:
            # 生成一个服从正态分布的随机数作为输入
            inputs = jax.random.normal(
# 如果条件成立，使用给定的参数创建一个形状为[1, present_len, config.d_model]的张量
# 否则，使用给定的参数创建一个形状为[1, present_len]的张量
if condition:
    inputs = jax.random.normal(
        key=rng_fixture(0),
        shape=[1, present_len, config.d_model],
        dtype=config.dtype,
    )
else:
    inputs = jax.random.randint(
        key=rng_fixture(0),
        minval=0,
        maxval=config.n_vocab,
        shape=[1, present_len],
    )

# 创建一个形状为[1, present_len]的整型零张量
doc_ids = jnp.zeros([1, present_len], dtype=jnp.int32)

# 使用给定的配置和批量大小为1创建一个Transformer的初始状态
state = Transformer.initial_state(config=config, batch_size=1)

# 使用给定的参数初始化Transformer模型，返回模型参数
params = Transformer(config).init(
    rngs,
    inputs=inputs,
    doc_ids=doc_ids,
    state=state,
    vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=present_len),
)["params"]
# 返回 Transformer 和 params 变量
return Transformer, params

# 返回 _transformer 函数
return _transformer
```