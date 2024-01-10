# `transformer_vq\tests\nn\test_model.py`

```
# 导入需要的库
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# 导入自定义的常量和函数
from tests.common import WIDENINGS
from tests.common import gen_len_tuples
from tests.common import transformer_fixture
from tests.common import transformer_config_fixture
from tests.common import basic_vq_spec_fixture
from tests.common import rng_fixture

# 参数化测试用例，测试不同的参数组合
@pytest.mark.parametrize("widening", WIDENINGS)
@pytest.mark.parametrize("agg_cache", [True, False])
@pytest.mark.parametrize("sequence_len,block_len,mem_len", gen_len_tuples("tlm", 12))
def test_model_jacobian(
    rng_fixture,
    transformer_fixture,
    transformer_config_fixture,
    basic_vq_spec_fixture,
    widening,
    agg_cache,
    sequence_len,
    block_len,
    mem_len,
    dtype=jnp.float32,
    is_train=True,
):
    # 根据参数生成 transformer 的配置
    config = transformer_config_fixture(
        block_len=block_len,
        mem_len=mem_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
        no_emb=True,
    )
    # 根据配置生成 transformer 的类和参数
    cls, params = transformer_fixture(config)
    # 生成 transformer 的初始状态
    initial_state = cls.initial_state(config, batch_size=1)

    # 定义一个函数，用于计算输入的 Jacobian 矩阵
    def call_fn(x):
        # 对输入进行处理，然后传入 transformer 模型中得到输出
        transformer_output_dict = cls(config).apply(
            {"params": params},
            state=initial_state,
            inputs=jnp.pad(
                x[None, ..., None], ((0, 0), (0, 0), (0, config.d_model - 1))
            ),
            doc_ids=jnp.ones([1, sequence_len], dtype=jnp.int32),
            vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=sequence_len),
            rngs=dict(
                ephemeral=rng_fixture(1),
                timeless=rng_fixture(2),
            ),
        )
        # 返回 transformer 输出中的 logprobs
        return transformer_output_dict["logprobs"][0, :, 0]

    # 生成随机输入
    inputs = jax.random.normal(rng_fixture(0), [sequence_len], dtype=config.dtype)
    # 计算输入的 Jacobian 矩阵
    jac = jax.jacfwd(call_fn)(inputs)
    # 打印 Jacobian 矩阵
    print(jac)
    # 检查输出对未来输入的梯度是否为零
    # check that outputs do not have nonzero grad wrt future inputs
    # 使用 NumPy 测试库检查两个数组的所有元素是否在指定的容差范围内接近
    np.testing.assert_allclose(actual=jac, desired=jnp.tril(jac), atol=1e-9, rtol=1e-9)
    # 使用 pytest 库检查是否引发指定类型的异常
    with pytest.raises(AssertionError):
        # 检查输出相对于过去/现在的输入是否具有非零梯度：
        # 如果它们只有零梯度，断言将通过，测试将失败
        # 使用 NumPy 测试库检查两个数组的所有元素是否在指定的容差范围内接近零
        np.testing.assert_allclose(
            actual=jac, desired=jnp.zeros_like(jac), atol=1e-9, rtol=1e-9
        )
# 使用参数化测试，测试不同的 widening 值
@pytest.mark.parametrize("widening", WIDENINGS)
# 使用参数化测试，测试不同的 agg_cache 值
@pytest.mark.parametrize("agg_cache", [True, False])
# 使用参数化测试，测试不同的 sequence_len, block_len, mem_len 组合
@pytest.mark.parametrize("sequence_len,block_len,mem_len", gen_len_tuples("tlm", 12))
def test_model_forward_consistency(
    rng_fixture,  # 随机数生成器的 fixture
    transformer_fixture,  # transformer 的 fixture
    transformer_config_fixture,  # transformer 配置的 fixture
    basic_vq_spec_fixture,  # 基本 vq_spec 的 fixture
    widening,  # 参数化测试中的 widening 值
    agg_cache,  # 参数化测试中的 agg_cache 值
    sequence_len,  # 参数化测试中的 sequence_len 值
    block_len,  # 参数化测试中的 block_len 值
    mem_len,  # 参数化测试中的 mem_len 值
    is_train=True,  # 是否为训练模式，默认为 True
    dtype=jnp.float32,  # 数据类型，默认为 jnp.float32
):
    bsz = 1  # batch size 为 1
    # 根据参数化测试的值生成 transformer 配置
    config = transformer_config_fixture(
        block_len=block_len,
        mem_len=mem_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
    )
    # 根据配置生成 transformer 类和参数
    cls, params = transformer_fixture(config)
    # 生成随机输入数据
    inputs = jax.random.randint(
        rng_fixture(0), minval=0, maxval=config.n_vocab, shape=[bsz, sequence_len]
    )
    # 生成初始状态
    initial_state = cls.initial_state(config=config, batch_size=bsz)
    # 生成随机数生成器字典
    rngs = dict(ephemeral=rng_fixture(1), timeless=rng_fixture(2))

    # 期望的输出结果
    o_expected = (
        cls(config)
        .apply(
            {"params": params},
            inputs=inputs,
            doc_ids=jnp.ones([bsz, sequence_len], dtype=jnp.int32),
            state=initial_state,
            vq_spec=basic_vq_spec_fixture(batch_size=bsz, block_len=sequence_len),
            rngs=rngs,
        )
        .get("logprobs")
    )
    o_actual = []  # 实际输出结果的列表
    # 生成初始状态
    state_p = cls.initial_state(config=config, batch_size=bsz)
    # 循环遍历序列长度除以块长度的次数
    for p in range(sequence_len // block_len):
        # 切片操作，获取当前块的索引范围
        slice_ = slice(p * block_len, (p + 1) * block_len)
        # 从输入数据中获取当前块的数据
        inputs_p = inputs[:, slice_]
        # 使用配置创建一个新的对象，并对其应用操作
        results = cls(config).apply(
            {"params": params},
            inputs=inputs_p,
            doc_ids=jnp.ones([bsz, block_len], dtype=jnp.int32),
            state=state_p,
            vq_spec=basic_vq_spec_fixture(batch_size=bsz, block_len=block_len),
            rngs=rngs,
        )
        # 获取结果中的对数概率
        o_p = results.get("logprobs")
        # 获取结果中的注意力状态
        state_p = results.get("attn_state")
        # 打印当前块的索引
        print(p)
        # 打印当前块的注意力状态
        print(state_p)
        # 将当前块的对数概率添加到实际结果列表中
        o_actual.append(o_p)
    # 将实际结果列表按列拼接成最终结果
    o_actual = jnp.concatenate(o_actual, axis=1)
    # 断言实际结果的列数与期望结果的列数相等
    assert o_expected.shape[1] == sequence_len
    # 断言实际结果的形状与期望结果的形状相等
    assert o_actual.shape == o_expected.shape
    # 使用 numpy 测试库检查实际结果与期望结果的接近程度
    np.testing.assert_allclose(
        actual=o_actual, desired=o_expected, atol=1e-5, rtol=1e-3
    )
```