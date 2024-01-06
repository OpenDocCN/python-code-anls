# `transformer_vq\tests\nn\test_attn.py`

```
# 导入需要的库
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# 导入自定义模块
from transformer_vq.nn.attn import VQAttention

# 导入测试所需的公共模块和数据
from tests.common import DTYPES
from tests.common import WIDENINGS
from tests.common import gen_len_tuples
from tests.common import attn_fixture
from tests.common import transformer_config_fixture
from tests.common import basic_vq_spec_fixture
from tests.common import rng_fixture

# 定义测试函数
def test_jax_nn_one_hot():
    # 断言测试结果是否与预期一致
    np.testing.assert_allclose(
        actual=jax.nn.one_hot(jnp.array(0), num_classes=3, axis=-1, dtype=jnp.float32),
# 测试 jax.nn.one_hot 函数，验证当输入为 0 时，生成的独热编码是否为 [1, 0, 0]
np.testing.assert_allclose(
    actual=jax.nn.one_hot(jnp.array(0), num_classes=3, axis=-1, dtype=jnp.float32),
    desired=jnp.array([1, 0, 0], dtype=jnp.float32),
)

# 测试 jax.nn.one_hot 函数，验证当输入为 1 时，生成的独热编码是否为 [0, 1, 0]
np.testing.assert_allclose(
    actual=jax.nn.one_hot(jnp.array(1), num_classes=3, axis=-1, dtype=jnp.float32),
    desired=jnp.array([0, 1, 0], dtype=jnp.float32),
)

# 测试 jax.nn.one_hot 函数，验证当输入为 2 时，生成的独热编码是否为 [0, 0, 1]
np.testing.assert_allclose(
    actual=jax.nn.one_hot(jnp.array(2), num_classes=3, axis=-1, dtype=jnp.float32),
    desired=jnp.array([0, 0, 1], dtype=jnp.float32),
)

# 测试 jax.nn.one_hot 函数，验证当输入为 3 时，生成的独热编码是否为 [0, 0, 0]
np.testing.assert_allclose(
    actual=jax.nn.one_hot(jnp.array(3), num_classes=3, axis=-1, dtype=jnp.float32),
    desired=jnp.array([0, 0, 0], dtype=jnp.float32),
)

# 测试 jnp.take_along_axis 函数，将数组 z 沿着 axis=2 的方向，根据数组 c 的索引取值
c = jnp.reshape(jnp.arange(30), [2, 3, 5])
z = jnp.reshape(jnp.arange(0, 5), (1, 1, 5))
cz_actual = jnp.take_along_axis(c, z, axis=2)
# 将变量 cz_expected 设置为变量 c 的值
cz_expected = c
# 使用 NumPy 测试库检查变量 cz_actual 是否与变量 cz_expected 全部接近
np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)

# 重新定义变量 c 为一个 2x3x5 的数组
c = jnp.reshape(jnp.arange(30), [2, 3, 5])
# 重新定义变量 z 为一个 1x1x4 的数组
z = jnp.reshape(jnp.arange(0, 5), (1, 1, 5))[:, :, 0:4]
# 使用 take_along_axis 函数获取 c 中 z 对应位置的元素，存储在变量 cz_actual 中
cz_actual = jnp.take_along_axis(c, z, axis=2)
# 重新定义变量 cz_expected 为 c 的第三个维度切片
cz_expected = c[:, :, 0:4]
# 使用 NumPy 测试库检查变量 cz_actual 是否与变量 cz_expected 全部接近
np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)

# 重新定义变量 c 为一个 2x3x5x7 的数组
c = jnp.reshape(jnp.arange(210), [2, 3, 5, 7])
# 重新定义变量 z 为一个 1x1x4 的数组
z = jnp.reshape(jnp.arange(0, 5), (1, 1, 5))[:, :, 0:4]
# 使用 take_along_axis 函数获取 c 中 z 对应位置的元素，存储在变量 cz_actual 中
cz_actual = jnp.take_along_axis(c, jnp.expand_dims(z, -1), axis=2)
# 重新定义变量 cz_expected 为 c 的第三个维度切片
cz_expected = c[:, :, 0:4, :]
# 使用 NumPy 测试库检查变量 cz_actual 是否与变量 cz_expected 全部接近
np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)

# 重新定义变量 c 为一个 2x3x5x7 的数组
c = jnp.reshape(jnp.arange(210), [2, 3, 5, 7])
# 重新定义变量 z 为一个 1x1x10 的数组
z = jnp.reshape(jnp.remainder(jnp.arange(0, 10), jnp.array(5)), (1, 1, 10))
# 使用 take_along_axis 函数获取 c 中 z 对应位置的元素，存储在变量 cz_actual 中
cz_actual = jnp.take_along_axis(c, jnp.expand_dims(z, -1), axis=2)
# 重新定义变量 cz_expected 为 c 沿第三个维度拼接自身
cz_expected = jnp.concatenate([c, c], axis=2)
# 使用 NumPy 测试库检查变量 cz_actual 是否与变量 cz_expected 全部接近
np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)
# 定义测试函数，测试获取聚合偏置的函数
def test_get_agg_biases(rng_fixture):
    # 设置时间步长为1024
    timesteps = 1024

    # 生成时间步长范围的数组
    counts = jnp.arange(timesteps)
    # 调用VQAttention类的get_agg_biases方法获取聚合偏置
    biases = VQAttention.get_agg_biases(counts)
    # 使用np.testing.assert_allclose进行数值比较
    np.testing.assert_allclose(jnp.exp(biases), counts)

    # 生成服从正态分布的注意力分数
    attn_scores = jax.random.normal(rng_fixture(0), shape=[timesteps])
    # 将注意力分数与偏置相加
    biased_attn_scores = attn_scores + biases
    # 使用np.testing.assert_allclose进行数值比较
    np.testing.assert_allclose(
        jnp.exp(biased_attn_scores), counts * jnp.exp(attn_scores)
    )

# 使用pytest的@parametrize装饰器，对widening和agg_cache参数进行组合测试
@pytest.mark.parametrize("widening", WIDENINGS)
@pytest.mark.parametrize("agg_cache", [True, False])
def test_attn_jacobian(
    rng_fixture,
```
```python
# 继续上面的代码

    # ...（省略部分代码）

    # 定义测试函数，测试注意力雅可比矩阵
    def test_attn_jacobian(
        rng_fixture,
        # ...（省略部分参数）
    ):
        # ...（省略部分测试代码）
```

注释：
- 定义了一个测试函数test_attn_jacobian，用于测试注意力雅可比矩阵的计算
- 使用pytest的@parametrize装饰器，对widening和agg_cache参数进行组合测试
- 定义了测试函数的参数rng_fixture，用于生成随机数种子
- 其他部分代码省略，未提供足够信息，无法添加注释
# 定义函数参数
def some_function(
    attn_fixture,  # 注意力机制的配置
    transformer_config_fixture,  # 转换器配置
    basic_vq_spec_fixture,  # 基本的VQ规范
    widening,  # 是否扩大
    agg_cache,  # 聚合缓存
    is_train=True,  # 是否训练
    sequence_len=10,  # 序列长度，默认为10
    dtype=jnp.float32,  # 数据类型，默认为32位浮点数
):
    # 使用给定的参数配置创建transformer配置
    config = transformer_config_fixture(
        block_len=sequence_len,  # 块长度为序列长度
        mem_len=sequence_len,  # 记忆长度为序列长度
        agg_cache=agg_cache,  # 聚合缓存
        widening=widening,  # 是否扩大
        dtypes=dtype,  # 数据类型
        is_train=is_train,  # 是否训练
    )
    # 使用给定的配置创建注意力机制实例和参数
    cls, params = attn_fixture(config)
    # 使用配置创建初始状态
    initial_state = cls.initial_state(config, batch_size=1)
# 定义一个函数，接受一个参数 x
def call_fn(x):
    # 对输入的注意力输入进行处理，并返回输出的注意力输出
    # 调用 cls 类的构造函数，传入配置参数 config，并调用 apply 方法
    # 传入参数 params，初始状态 initial_state，输入字典 input_dict 和随机数 rngs
    _, attn_output_dict = cls(config).apply(
        {"params": params},
        state=initial_state,
        input_dict=dict(
            # 对输入特征进行填充
            input_features=jnp.pad(
                x[None, ..., None], ((0, 0), (0, 0), (0, config.d_model - 1))
            ),
            # 设置文档 ID
            doc_ids=jnp.ones([1, sequence_len], dtype=jnp.int32),
            # 设置 vq_spec
            vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=sequence_len),
        ),
        rngs=dict(
            # 设置随机数种子
            ephemeral=rng_fixture(1),
            timeless=rng_fixture(2),
        ),
    )
    # 返回注意力输出字典中的特定结果
    return attn_output_dict["res"][0, :, 0]

# 生成一个服从正态分布的随机输入
inputs = jax.random.normal(rng_fixture(0), [sequence_len], dtype=config.dtype)
    # 使用自动微分计算函数 call_fn 对输入 inputs 的雅可比矩阵
    jac = jax.jacfwd(call_fn)(inputs)
    # 打印雅可比矩阵
    print(jac)
    # 检查输出对未来输入的梯度是否为零
    np.testing.assert_allclose(actual=jac, desired=jnp.tril(jac), atol=1e-9, rtol=1e-9)
    # 使用 pytest 检查是否抛出 AssertionError
    with pytest.raises(AssertionError):
        # 检查输出对过去/现在输入的梯度是否为零
        # 如果它们只有零梯度，断言将通过，测试将失败
        np.testing.assert_allclose(
            actual=jac, desired=jnp.zeros_like(jac), atol=1e-9, rtol=1e-9
        )

# 使用参数化测试，测试不同的参数组合
@pytest.mark.parametrize("widening", WIDENINGS)
@pytest.mark.parametrize("agg_cache", [True, False])
@pytest.mark.parametrize("sequence_len,block_len,mem_len", gen_len_tuples("tlm", 12))
def test_attn_forward_consistency(
    rng_fixture,
    attn_fixture,
    transformer_config_fixture,
    basic_vq_spec_fixture,
    widening,  # 宽度参数
    agg_cache,  # 聚合缓存参数
    sequence_len,  # 序列长度参数
    block_len,  # 块长度参数
    mem_len,  # 记忆长度参数
    dtype=jnp.float32,  # 数据类型，默认为 jnp.float32
    is_train=True,  # 是否为训练模式，默认为 True
):
    # 使用给定的参数创建 transformer_config_fixture 对象
    config_monoblock = transformer_config_fixture(
        block_len=sequence_len,  # 使用序列长度参数设置块长度
        mem_len=mem_len,  # 使用记忆长度参数设置记忆长度
        agg_cache=agg_cache,  # 使用聚合缓存参数设置聚合缓存
        widening=widening,  # 使用宽度参数设置宽度
        dtypes=dtype,  # 使用数据类型参数设置数据类型
        is_train=is_train,  # 使用训练模式参数设置训练模式
    )
    bsz = 3  # 设置批量大小为 3

    # 使用给定的配置创建 attn_fixture 对象
    cls, params = attn_fixture(config_monoblock)
    # 使用配置和批量大小创建初始状态
    initial_state = cls.initial_state(config=config_monoblock, batch_size=bsz)
# 定义输入数据的形状，包括批量大小、序列长度和模型维度
inputs_shape = [bsz, sequence_len, config_monoblock.d_model]
# 生成符合正态分布的随机输入数据
inputs = jax.random.normal(rng_fixture(0), inputs_shape, dtype=dtype)
# 调用 cls 函数，传入参数、初始状态和输入数据，获取输出字典
_, output_dict = cls(config_monoblock).apply(
    {"params": params},
    state=initial_state,
    input_dict=dict(
        input_features=inputs,
        doc_ids=jnp.ones([bsz, sequence_len], dtype=jnp.int32),
        vq_spec=basic_vq_spec_fixture(
            batch_size=bsz,
            block_len=config_monoblock.block_len,
        ),
    ),
    rngs=dict(ephemeral=rng_fixture(1), timeless=rng_fixture(2)),
)
# 获取输出字典中的预期结果
o_expected = output_dict["res"]

# 使用给定的块长度创建 transformer 的配置
config_multiblock = transformer_config_fixture(
    block_len=block_len,
# 设置内存长度、聚合缓存、扩展、数据类型和训练标志
mem_len=mem_len,
agg_cache=agg_cache,
widening=widening,
dtypes=dtype,
is_train=is_train,
)
# 初始化实际输出列表
o_actual = []
# 初始化多块配置状态
state_p = cls.initial_state(
    config=config_multiblock,
    batch_size=bsz,
)
# 循环处理每个块的序列
for p in range(sequence_len // block_len):
    # 应用多块配置，获取新状态和输出字典
    state_p, output_dict_p = cls(config_multiblock).apply(
        {"params": params},
        state=state_p,
        input_dict=dict(
            # 设置输入特征、文档 ID 和 VQ 规范
            input_features=inputs[:, p * block_len : (p + 1) * block_len, :],
            doc_ids=jnp.ones([bsz, block_len], dtype=jnp.int32),
            vq_spec=basic_vq_spec_fixture(
                batch_size=bsz,
# 设置参数 block_len，并传递给 config_multiblock.block_len
block_len=config_multiblock.block_len,
# 创建一个包含参数的字典，其中包括 block_len 和 rngs
),
rngs=dict(ephemeral=rng_fixture(1), timeless=rng_fixture(2)),
# 将 output_dict_p["res"] 添加到 o_actual 列表中
o_actual.append(output_dict_p["res"])
# 沿着 axis=1 连接 o_actual 列表中的数组
o_actual = jnp.concatenate(o_actual, axis=1)
# 断言 o_expected 的第二个维度等于 sequence_len
assert o_expected.shape[1] == sequence_len
# 断言 o_actual 的形状与 o_expected 的形状相等
assert o_actual.shape == o_expected.shape
# 如果 dtype 是 jnp.float64，则使用 np.testing.assert_allclose 检查 o_actual 和 o_expected 是否接近
if dtype is jnp.float64:
    np.testing.assert_allclose(
        actual=o_actual,
        desired=o_expected,
        atol=1e-5,
        rtol=1e-4,
    )
# 如果 dtype 是 jnp.float32，则使用 np.testing.assert_allclose 检查 o_actual 和 o_expected 是否接近
if dtype is jnp.float32:
    np.testing.assert_allclose(
        actual=o_actual,
        desired=o_expected,
    """
    # 设置默认的数值比较容忍度
    atol=5e-4,
    rtol=5e-3,
)

"""
# 如果需要检查哪些时间步骤出现问题，请取消注释以下代码：
for t in range(sequence_len):
    # 打印当前时间步骤
    print(t)
    # 如果数据类型是 jnp.float64，则进行数值比较
    if dtype is jnp.float64:
        np.testing.assert_allclose(
            actual=o_actual[:, t, :],
            desired=o_expected[:, t, :],
            atol=1e-5,  # 设置容忍度
            rtol=1e-4,  # 设置相对容忍度
        )
    # 如果数据类型是 jnp.float32，则进行数值比较
    if dtype is jnp.float32:
        np.testing.assert_allclose(
            actual=o_actual[:, t, :],
            desired=o_expected[:, t, :],
            atol=5e-4,  # 设置容忍度
# 使用pytest.mark.parametrize装饰器为测试函数添加参数化测试，参数为"WIDENINGS"列表中的值
@pytest.mark.parametrize("widening", WIDENINGS)
# 使用pytest.mark.parametrize装饰器为测试函数添加参数化测试，参数为布尔值True和False
@pytest.mark.parametrize("agg_cache", [True, False])
# 使用pytest.mark.parametrize装饰器为测试函数添加参数化测试，参数为gen_len_tuples("tm", 12)生成的元组
@pytest.mark.parametrize("sequence_len,mem_len", gen_len_tuples("tm", 12))
# 定义测试函数，参数包括rng_fixture, attn_fixture, transformer_config_fixture, basic_vq_spec_fixture, widening, agg_cache, sequence_len, mem_len, is_train, dtype
def test_attn_backward_consistency(
    rng_fixture,
    attn_fixture,
    transformer_config_fixture,
    basic_vq_spec_fixture,
    widening,
    agg_cache,
    sequence_len,
    mem_len,
    is_train=True,
    dtype=jnp.float32,
):
# 设置块大小为1
bsz = 1
# 使用给定的参数创建transformer配置
config_monoblock = transformer_config_fixture(
    block_len=sequence_len,  # 块长度
    mem_len=mem_len,  # 记忆长度
    agg_cache=agg_cache,  # 聚合缓存
    widening=widening,  # 扩展
    dtypes=dtype,  # 数据类型
    is_train=is_train,  # 是否训练
)
# 使用transformer配置创建注意力机制
cls, params = attn_fixture(config_monoblock)
# 使用初始状态创建单块初始状态
monoblock_initial_state = cls.initial_state(
    config=config_monoblock,  # 配置
    batch_size=bsz,  # 批处理大小
)

# 创建前缀列表
prefix = [bsz, config_monoblock.n_head, sequence_len]
# 获取d_k和d_v的值
d_k = config_monoblock.d_k
d_v = config_monoblock.d_v
# 生成正态分布的随机数作为查询向量
q = jax.random.normal(rng_fixture(1), [*prefix, d_k], dtype=config_monoblock.dtype)
# 获取查询向量的切片
q_slice = q[0, -1, :, -1]
    # 从随机数生成器中生成服从正态分布的随机数，用于创建键（k）向量
    k = jax.random.normal(rng_fixture(2), [*prefix, d_k], dtype=config_monoblock.dtype)
    # 从随机数生成器中生成服从正态分布的随机数，用于创建值（v）向量
    v = jax.random.normal(rng_fixture(3), [*prefix, d_v], dtype=config_monoblock.dtype)
    # 定义用于填充的规范
    pad_spec = (
        (0, 0),
        (config_monoblock.n_head - 1, 0),
        (0, 0),
        (config_monoblock.d_k - 1, 0),
    )
    # 创建用于全序列的基本 VQ 规范
    vq_spec_full_seq = basic_vq_spec_fixture(batch_size=bsz, block_len=sequence_len)
    # 创建随机数字典
    rngs = dict(ephemeral=rng_fixture(1), timeless=rng_fixture(2))

    def _get_expected_second_block_jaco_wrt_q():
        # 期望的第二个块雅可比矩阵，从完整的序列注意力计算得出
        jac_fn = jax.jacobian(
            lambda x: cls(config_monoblock).apply(
                {"params": params},
                present_q=jnp.pad(x[None, None, ..., None], pad_spec),
                present_k=k,
                present_v=v,
                present_doc_ids=jnp.ones([bsz, sequence_len], dtype=jnp.int32),
# 设置状态为monoblock_initial_state
state=monoblock_initial_state,
# 设置vq_spec为vq_spec_full_seq
vq_spec=vq_spec_full_seq,
# 设置rngs为rngs
rngs=rngs,
# 设置方法为cls.attn
method=cls.attn,
# 调用transformer_config_fixture函数，设置block_len为sequence_len的一半
block_len=sequence_len // 2,
# 设置mem_len为mem_len
mem_len=mem_len,
# 设置agg_cache为agg_cache
agg_cache=agg_cache,
# 设置widening为widening
widening=widening,
# 设置dtypes为dtype
dtypes=dtype,
# 设置is_train为is_train
is_train=is_train,
# 调用cls.initial_state函数，设置config为config_multiblock，batch_size为bsz
multiblock_initial_state = cls.initial_state(
    config=config_multiblock,
    batch_size=bsz,
)
# 创建一个基本的 VQ 规范的一半序列
vq_spec_half_seq = basic_vq_spec_fixture(
    batch_size=bsz, block_len=sequence_len // 2
)

# 应用多块配置的注意力机制
attn_outputs_midway = cls(config_multiblock).apply(
    {"params": params},
    present_q=jnp.pad(
        q_slice[0 : (sequence_len // 2)][None, None, ..., None],
        pad_spec,
    ),
    present_k=k[..., 0 : (sequence_len // 2), :],
    present_v=v[..., 0 : (sequence_len // 2), :],
    present_doc_ids=jnp.ones([bsz, sequence_len // 2], dtype=jnp.int32),
    state=multiblock_initial_state,
    vq_spec=vq_spec_half_seq,
    rngs=rngs,
    method=cls.attn,
)

# 更新参数的关键字参数
update_kwargs = dict(
    recent_z=attn_outputs_midway["recent_z"],
# 从 attn_outputs_midway 字典中获取 recent_k_hat、recent_v 和 recent_doc_ids，并传递给函数
recent_k_hat=attn_outputs_midway["recent_k_hat"],
recent_v=attn_outputs_midway["recent_v"],
recent_doc_ids=attn_outputs_midway["recent_doc_ids"],
)
# 使用 cls 类的 apply 方法，传递参数和更新参数的关键字参数，以及初始状态和更新状态的方法
midway_state = cls(config_multiblock).apply(
    {"params": params},
    **update_kwargs,
    state=multiblock_initial_state,
    method=cls.update_state,
)

# 定义一个函数 _get_actual_second_block_jaco_wrt_q
def _get_actual_second_block_jaco_wrt_q():
    # 定义一个函数 jac_fn，使用 jax.jacobian 计算 cls 类的 apply 方法的雅可比矩阵
    jac_fn = jax.jacobian(
        lambda x: cls(config_multiblock).apply(
            {"params": params},
            present_q=jnp.pad(x[None, None, ..., None], pad_spec),
            present_k=k[..., sequence_len // 2 :, :],
            present_v=v[..., sequence_len // 2 :, :],
            present_doc_ids=jnp.ones([bsz, sequence_len // 2], dtype=jnp.int32),
# 设置状态为中间状态
state=midway_state,
# 使用一半序列的 VQ 规范
vq_spec=vq_spec_half_seq,
# 使用给定的随机数生成器
rngs=rngs,
# 使用给定的注意力方法
method=cls.attn,
)["attn_out"][0, :, -1]
# 返回注意力输出的最后一个时间步的结果
)
# 返回对 Q 切片的实际雅可比矩阵
return jac_fn(q_slice[sequence_len // 2 :])

# 获取实际的第二个块相对于 Q 的雅可比矩阵
jac_actual = _get_actual_second_block_jaco_wrt_q()
# 获取期望的第二个块相对于 Q 的雅可比矩阵
jac_expected = _get_expected_second_block_jaco_wrt_q()
# 检查实际雅可比矩阵和期望雅可比矩阵是否在给定的容差范围内接近
np.testing.assert_allclose(
    actual=jac_actual, desired=jac_expected, atol=1e-5, rtol=1e-4
)
```