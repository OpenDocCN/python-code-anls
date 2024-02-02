# `transformer_vq\tests\nn\test_attn.py`

```py
# 导入需要的库
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# 导入自定义的模块和函数
from transformer_vq.nn.attn import VQAttention
from tests.common import DTYPES
from tests.common import WIDENINGS
from tests.common import gen_len_tuples
from tests.common import attn_fixture
from tests.common import transformer_config_fixture
from tests.common import basic_vq_spec_fixture
from tests.common import rng_fixture

# 定义测试函数 test_jax_nn_one_hot
def test_jax_nn_one_hot():
    # 测试 jax.nn.one_hot 函数，验证输出是否符合预期
    np.testing.assert_allclose(
        actual=jax.nn.one_hot(jnp.array(0), num_classes=3, axis=-1, dtype=jnp.float32),
        desired=jnp.array([1, 0, 0], dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        actual=jax.nn.one_hot(jnp.array(1), num_classes=3, axis=-1, dtype=jnp.float32),
        desired=jnp.array([0, 1, 0], dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        actual=jax.nn.one_hot(jnp.array(2), num_classes=3, axis=-1, dtype=jnp.float32),
        desired=jnp.array([0, 0, 1], dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        actual=jax.nn.one_hot(jnp.array(3), num_classes=3, axis=-1, dtype=jnp.float32),
        desired=jnp.array([0, 0, 0], dtype=jnp.float32),
    )

# 定义测试函数 test_jnp_take_along_axis
def test_jnp_take_along_axis():
    # 创建测试数据
    c = jnp.reshape(jnp.arange(30), [2, 3, 5])
    z = jnp.reshape(jnp.arange(0, 5), (1, 1, 5))
    # 测试 jnp.take_along_axis 函数，验证输出是否符合预期
    cz_actual = jnp.take_along_axis(c, z, axis=2)
    cz_expected = c
    np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)

    # 创建测试数据
    c = jnp.reshape(jnp.arange(30), [2, 3, 5])
    z = jnp.reshape(jnp.arange(0, 5), (1, 1, 5))[:, :, 0:4]
    # 测试 jnp.take_along_axis 函数，验证输出是否符合预期
    cz_actual = jnp.take_along_axis(c, z, axis=2)
    cz_expected = c[:, :, 0:4]
    np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)

    # 创建测试数据
    c = jnp.reshape(jnp.arange(210), [2, 3, 5, 7])
    z = jnp.reshape(jnp.arange(0, 5), (1, 1, 5))[:, :, 0:4]
    # 测试 jnp.take_along_axis 函数，验证输出是否符合预期
    cz_actual = jnp.take_along_axis(c, jnp.expand_dims(z, -1), axis=2)
    cz_expected = c[:, :, 0:4, :]
    np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)
    # 创建一个形状为[2, 3, 5, 7]的数组，其中包含0到209的整数
    c = jnp.reshape(jnp.arange(210), [2, 3, 5, 7])
    # 创建一个形状为[1, 1, 10]的数组，其中包含0到9的整数对5取余的结果
    z = jnp.reshape(jnp.remainder(jnp.arange(0, 10), jnp.array(5)), (1, 1, 10))
    # 从c数组中按照z数组的索引值取值，axis=2表示沿着第三个维度取值
    cz_actual = jnp.take_along_axis(c, jnp.expand_dims(z, -1), axis=2)
    # 沿着第三个维度将c数组和自身拼接起来
    cz_expected = jnp.concatenate([c, c], axis=2)
    # 检查cz_actual和cz_expected是否非常接近，如果不是则会引发AssertionError
    np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)
# 定义测试函数，用于测试获取聚合偏置
def test_get_agg_biases(rng_fixture):
    # 设置时间步长为1024
    timesteps = 1024

    # 生成长度为timesteps的数组
    counts = jnp.arange(timesteps)
    # 调用VQAttention类的get_agg_biases方法获取聚合偏置
    biases = VQAttention.get_agg_biases(counts)
    # 使用np.testing.assert_allclose方法检查biases的指数是否与counts相等
    np.testing.assert_allclose(jnp.exp(biases), counts)

    # 生成长度为timesteps的服从正态分布的随机数
    attn_scores = jax.random.normal(rng_fixture(0), shape=[timesteps])
    # 将偏置加到注意力分数上
    biased_attn_scores = attn_scores + biases
    # 使用np.testing.assert_allclose方法检查偏置后的注意力分数的指数是否与counts乘以原始注意力分数的指数相等
    np.testing.assert_allclose(
        jnp.exp(biased_attn_scores), counts * jnp.exp(attn_scores)
    )


# 使用pytest.mark.parametrize装饰器，参数化测试函数
@pytest.mark.parametrize("widening", WIDENINGS)
@pytest.mark.parametrize("agg_cache", [True, False])
def test_attn_jacobian(
    rng_fixture,
    attn_fixture,
    transformer_config_fixture,
    basic_vq_spec_fixture,
    widening,
    agg_cache,
    is_train=True,
    sequence_len=10,
    dtype=jnp.float32,
):
    # 根据给定的参数生成transformer的配置
    config = transformer_config_fixture(
        block_len=sequence_len,
        mem_len=sequence_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
    )
    # 调用attn_fixture方法获取注意力机制的类和参数
    cls, params = attn_fixture(config)
    # 根据配置生成初始状态
    initial_state = cls.initial_state(config, batch_size=1)

    # 定义call_fn函数，用于获取注意力输入的切片并返回输出的切片
    def call_fn(x):
        # 调用apply方法获取注意力输出字典
        _, attn_output_dict = cls(config).apply(
            {"params": params},
            state=initial_state,
            input_dict=dict(
                input_features=jnp.pad(
                    x[None, ..., None], ((0, 0), (0, 0), (0, config.d_model - 1))
                ),
                doc_ids=jnp.ones([1, sequence_len], dtype=jnp.int32),
                vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=sequence_len),
            ),
            rngs=dict(
                ephemeral=rng_fixture(1),
                timeless=rng_fixture(2),
            ),
        )
        # 返回注意力输出的切片
        return attn_output_dict["res"][0, :, 0]

    # 生成长度为sequence_len的服从正态分布的随机数作为输入
    inputs = jax.random.normal(rng_fixture(0), [sequence_len], dtype=config.dtype)
    # 使用jax.jacfwd方法计算call_fn的雅可比矩阵
    jac = jax.jacfwd(call_fn)(inputs)
    # 打印雅可比矩阵
    print(jac)
    # 检查输出是否对未来输入的梯度为零
    # check that outputs do not have nonzero grad wrt future inputs
    # 使用 NumPy 测试库检查两个数组是否在给定的容差范围内全部相等
    np.testing.assert_allclose(actual=jac, desired=jnp.tril(jac), atol=1e-9, rtol=1e-9)
    # 使用 pytest 库检查是否引发指定类型的异常
    with pytest.raises(AssertionError):
        # 检查输出相对于过去/现在的输入是否具有非零梯度：
        # 如果它们只有零梯度，断言将通过，测试将失败
        # 使用 NumPy 测试库检查两个数组是否在给定的容差范围内全部相等
        np.testing.assert_allclose(
            actual=jac, desired=jnp.zeros_like(jac), atol=1e-9, rtol=1e-9
        )
# 使用参数化测试，对不同的参数组合进行测试
@pytest.mark.parametrize("widening", WIDENINGS)
@pytest.mark.parametrize("agg_cache", [True, False])
@pytest.mark.parametrize("sequence_len,block_len,mem_len", gen_len_tuples("tlm", 12))
def test_attn_forward_consistency(
    rng_fixture,  # 随机数生成器的夹具
    attn_fixture,  # 注意力机制的夹具
    transformer_config_fixture,  # 变压器配置的夹具
    basic_vq_spec_fixture,  # 基本 VQ 规范的夹具
    widening,  # 是否进行扩展
    agg_cache,  # 是否进行聚合缓存
    sequence_len,  # 序列长度
    block_len,  # 块长度
    mem_len,  # 记忆长度
    dtype=jnp.float32,  # 数据类型，默认为 jnp.float32
    is_train=True,  # 是否为训练模式，默认为 True
):
    # 根据变压器配置生成单块配置
    config_monoblock = transformer_config_fixture(
        block_len=sequence_len,
        mem_len=mem_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
    )
    # 批量大小为 3
    bsz = 3

    # 生成注意力机制的参数
    cls, params = attn_fixture(config_monoblock)
    # 根据单块配置生成初始状态
    initial_state = cls.initial_state(config=config_monoblock, batch_size=bsz)

    # 生成输入数据的形状
    inputs_shape = [bsz, sequence_len, config_monoblock.d_model]
    # 生成输入数据
    inputs = jax.random.normal(rng_fixture(0), inputs_shape, dtype=dtype)
    # 应用注意力机制，得到输出字典
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
    # 期望输出结果
    o_expected = output_dict["res"]

    # 根据变压器配置生成多块配置
    config_multiblock = transformer_config_fixture(
        block_len=block_len,
        mem_len=mem_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
    )
    # 实际输出结果列表
    o_actual = []
    # 根据多块配置生成初始状态
    state_p = cls.initial_state(
        config=config_multiblock,
        batch_size=bsz,
    )
    # 根据序列长度和块长度计算循环次数
    for p in range(sequence_len // block_len):
        # 调用 cls 对象的 apply 方法，传入参数、状态、输入字典和随机数字典，获取状态和输出字典
        state_p, output_dict_p = cls(config_multiblock).apply(
            {"params": params},
            state=state_p,
            input_dict=dict(
                input_features=inputs[:, p * block_len : (p + 1) * block_len, :],
                doc_ids=jnp.ones([bsz, block_len], dtype=jnp.int32),
                vq_spec=basic_vq_spec_fixture(
                    batch_size=bsz,
                    block_len=config_multiblock.block_len,
                ),
            ),
            rngs=dict(ephemeral=rng_fixture(1), timeless=rng_fixture(2)),
        )
        # 将输出字典中的 "res" 对应的值添加到 o_actual 列表中
        o_actual.append(output_dict_p["res"])
    # 将 o_actual 列表中的数组按照 axis=1 进行拼接
    o_actual = jnp.concatenate(o_actual, axis=1)
    # 断言 o_expected 的第二维度与 sequence_len 相等
    assert o_expected.shape[1] == sequence_len
    # 断言 o_actual 的形状与 o_expected 相等
    assert o_actual.shape == o_expected.shape
    # 如果 dtype 是 jnp.float64，则使用 np.testing.assert_allclose 进行数值比较
    if dtype is jnp.float64:
        np.testing.assert_allclose(
            actual=o_actual,
            desired=o_expected,
            atol=1e-5,
            rtol=1e-4,
        )
    # 如果 dtype 是 jnp.float32，则使用 np.testing.assert_allclose 进行数值比较
    if dtype is jnp.float32:
        np.testing.assert_allclose(
            actual=o_actual,
            desired=o_expected,
            atol=5e-4,
            rtol=5e-3,
        )

    """
    # 取消注释以检查哪些时间步是错误的：
    for t in range(sequence_len):
        # 打印当前时间步
        print(t)
        # 如果 dtype 是 jnp.float64，则使用 np.testing.assert_allclose 进行数值比较
        if dtype is jnp.float64:
            np.testing.assert_allclose(
                actual=o_actual[:, t, :],
                desired=o_expected[:, t, :],
                atol=1e-5,
                rtol=1e-4,
            )
        # 如果 dtype 是 jnp.float32，则使用 np.testing.assert_allclose 进行数值比较
        if dtype is jnp.float32:
            np.testing.assert_allclose(
                actual=o_actual[:, t, :],
                desired=o_expected[:, t, :],
                atol=5e-4,
                rtol=5e-3,
            )
        """
# 使用参数化测试，测试不同的widening值
# 使用参数化测试，测试不同的agg_cache值
# 使用参数化测试，生成不同的sequence_len和mem_len组合
def test_attn_backward_consistency(
    rng_fixture,  # 随机数生成器
    attn_fixture,  # 注意力机制的fixture
    transformer_config_fixture,  # 变压器配置的fixture
    basic_vq_spec_fixture,  # 基本VQ规范的fixture
    widening,  # 是否进行扩展
    agg_cache,  # 是否进行缓存聚合
    sequence_len,  # 序列长度
    mem_len,  # 记忆长度
    is_train=True,  # 是否为训练模式
    dtype=jnp.float32,  # 数据类型
):
    bsz = 1  # 批量大小为1
    # 生成单块配置
    config_monoblock = transformer_config_fixture(
        block_len=sequence_len,  # 块长度
        mem_len=mem_len,  # 记忆长度
        agg_cache=agg_cache,  # 缓存聚合
        widening=widening,  # 扩展
        dtypes=dtype,  # 数据类型
        is_train=is_train,  # 是否为训练模式
    )
    # 生成注意力机制和参数
    cls, params = attn_fixture(config_monoblock)
    # 生成单块的初始状态
    monoblock_initial_state = cls.initial_state(
        config=config_monoblock,  # 单块配置
        batch_size=bsz,  # 批量大小
    )

    prefix = [bsz, config_monoblock.n_head, sequence_len]  # 前缀
    d_k = config_monoblock.d_k  # 查询键的维度
    d_v = config_monoblock.d_v  # 值的维度
    q = jax.random.normal(rng_fixture(1), [*prefix, d_k], dtype=config_monoblock.dtype)  # 生成查询
    q_slice = q[0, -1, :, -1]  # 查询的切片
    k = jax.random.normal(rng_fixture(2), [*prefix, d_k], dtype=config_monoblock.dtype)  # 生成键
    v = jax.random.normal(rng_fixture(3), [*prefix, d_v], dtype=config_monoblock.dtype)  # 生成值
    pad_spec = (  # 填充规范
        (0, 0),
        (config_monoblock.n_head - 1, 0),
        (0, 0),
        (config_monoblock.d_k - 1, 0),
    )
    vq_spec_full_seq = basic_vq_spec_fixture(batch_size=bsz, block_len=sequence_len)  # 生成完整序列的VQ规范
    rngs = dict(ephemeral=rng_fixture(1), timeless=rng_fixture(2))  # 随机数生成器字典
    # 定义一个函数，用于获取关于第二个块的期望雅可比矩阵
    def _get_expected_second_block_jaco_wrt_q():
        # 定义一个函数，计算从完整注意力机制计算得到的期望第二块雅可比矩阵
        jac_fn = jax.jacobian(
            # 定义一个 lambda 函数，调用 cls 类的 apply 方法，传入一系列参数
            lambda x: cls(config_monoblock).apply(
                {"params": params},
                present_q=jnp.pad(x[None, None, ..., None], pad_spec),
                present_k=k,
                present_v=v,
                present_doc_ids=jnp.ones([bsz, sequence_len], dtype=jnp.int32),
                state=monoblock_initial_state,
                vq_spec=vq_spec_full_seq,
                rngs=rngs,
                method=cls.attn,
            )["attn_out"][0, :, -1]
        )
        # 返回计算得到的雅可比矩阵
        return jac_fn(q_slice)[sequence_len // 2 :, sequence_len // 2 :]

    # 生成多块配置
    config_multiblock = transformer_config_fixture(
        block_len=sequence_len // 2,
        mem_len=mem_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
    )
    # 生成多块初始状态
    multiblock_initial_state = cls.initial_state(
        config=config_multiblock,
        batch_size=bsz,
    )

    # 生成一半序列的基本 VQ 规范
    vq_spec_half_seq = basic_vq_spec_fixture(
        batch_size=bsz, block_len=sequence_len // 2
    )
    # 调用 cls 类的 apply 方法，传入一系列参数，计算注意力输出
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
    # 定义一个字典，包含更新参数
    update_kwargs = dict(
        recent_z=attn_outputs_midway["recent_z"],
        recent_k_hat=attn_outputs_midway["recent_k_hat"],
        recent_v=attn_outputs_midway["recent_v"],
        recent_doc_ids=attn_outputs_midway["recent_doc_ids"],
    )
    # 应用给定的配置和参数更新到中间状态
    midway_state = cls(config_multiblock).apply(
        {"params": params},
        **update_kwargs,
        state=multiblock_initial_state,
        method=cls.update_state,
    )

    # 定义获取实际第二个块相对于 q 的雅可比矩阵的函数
    def _get_actual_second_block_jaco_wrt_q():
        # 实际第二个块的雅可比矩阵
        jac_fn = jax.jacobian(
            # 定义一个函数，应用给定的配置和参数，计算第二个块相对于 q 的雅可比矩阵
            lambda x: cls(config_multiblock).apply(
                {"params": params},
                present_q=jnp.pad(x[None, None, ..., None], pad_spec),
                present_k=k[..., sequence_len // 2 :, :],
                present_v=v[..., sequence_len // 2 :, :],
                present_doc_ids=jnp.ones([bsz, sequence_len // 2], dtype=jnp.int32),
                state=midway_state,
                vq_spec=vq_spec_half_seq,
                rngs=rngs,
                method=cls.attn,
            )["attn_out"][0, :, -1]
        )
        # 返回相对于 q 的雅可比矩阵
        return jac_fn(q_slice[sequence_len // 2 :])

    # 获取实际的第二个块相对于 q 的雅可比矩阵
    jac_actual = _get_actual_second_block_jaco_wrt_q()
    # 获取期望的第二个块相对于 q 的雅可比矩阵
    jac_expected = _get_expected_second_block_jaco_wrt_q()
    # 使用 numpy 测试实际的雅可比矩阵和期望的雅可比矩阵是否非常接近
    np.testing.assert_allclose(
        actual=jac_actual, desired=jac_expected, atol=1e-5, rtol=1e-4
    )
```