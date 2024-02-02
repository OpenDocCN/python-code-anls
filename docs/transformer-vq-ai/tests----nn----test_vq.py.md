# `transformer_vq\tests\nn\test_vq.py`

```py
# 导入 functools 模块
import functools
# 导入 os 模块
import os

# 设置本地设备数量为 8
N_LOCAL_DEVICE = 8
# 设置环境变量 XLA_FLAGS 为指定的值
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_LOCAL_DEVICE}"

# 导入 jax 模块
import jax
# 导入 jax 中的 numpy 模块，并重命名为 jnp
import jax.numpy as jnp
# 导入 numpy 模块，并重命名为 np
import numpy as np
# 导入 pytest 模块
import pytest
# 从 flax 模块中导入 jax_utils
from flax import jax_utils
# 从 flax.training 模块中导入 common_utils
from flax.training import common_utils

# 从 transformer_vq.nn.vq 模块中导入 LearnableVQ, get_shortcodes, get_codewords
from transformer_vq.nn.vq import LearnableVQ, get_shortcodes, get_codewords
# 从 transformer_vq.nn.types 模块中导入 TransformerConfig
from transformer_vq.nn.types import TransformerConfig

# noreorder
# 从 tests.common 模块中导入 transformer_config_fixture
from tests.common import transformer_config_fixture
# 从 tests.common 模块中导入 rng_fixture
from tests.common import rng_fixture
# 从 tests.common 模块中导入 basic_vq_spec_fixture
from tests.common import basic_vq_spec_fixture
# 从 tests.common 模块中导入 multidev_vq_spec_fixture
from tests.common import multidev_vq_spec_fixture
# 从 tests.common 模块中导入 TOLERANCES
from tests.common import TOLERANCES

# 更新 jax 的配置，启用 x64 模式
jax.config.update("jax_enable_x64", True)

# 定义测试函数 test_num_devices
def test_num_devices():
    # 断言当前设备数量等于预设的本地设备数量
    assert jax.device_count() == N_LOCAL_DEVICE
    # 断言当前本地设备数量等于预设的本地设备数量
    assert jax.local_device_count() == N_LOCAL_DEVICE

# 定义 pytest 的 fixture quantizer_fixture
@pytest.fixture
def quantizer_fixture(rng_fixture, basic_vq_spec_fixture):
    # 定义内部函数 _make_quantizer
    def _make_quantizer(cls, config: TransformerConfig):
        # 设置 present_len 为 config.n_code
        present_len = config.n_code
        # 设置 inputs_shape 为指定的列表
        inputs_shape = [1, config.n_head, present_len, config.d_k]
        # 生成符合正态分布的随机数作为输入
        inputs = jax.random.normal(rng_fixture(0), inputs_shape, dtype=jnp.float32)
        # 设置随机数种子
        rngs = dict(
            params=rng_fixture(1),
            ephemeral=rng_fixture(2),
            timeless=rng_fixture(3),
        )
        # 生成 basic_vq_spec_fixture
        vq_spec = basic_vq_spec_fixture(batch_size=1, block_len=present_len)
        # 打印 present_len 的值
        print(f"present_len: {present_len}")
        # 打印 vq_spec.loss_mask 的形状
        print(f"vq_spec.loss_mask.shape: {vq_spec.loss_mask.shape}")
        # 初始化 quantizer 对象并返回参数
        params = cls(config=config).init(
            rngs,
            vecs=inputs.astype(config.dtype),
            vq_spec=vq_spec,
        )["params"]
        return cls, params

    return _make_quantizer

# 定义测试函数 test_get_shortcodes
def test_get_shortcodes():
    # 设置 h, l, s, d 的值
    h, l, s, d = 3, 5, 7, 11
    # 生成 codebook
    codebook = jax.nn.one_hot(jnp.arange(s), num_classes=d)
    codebook = jnp.arange(1, h + 1).reshape([h, 1, 1]) * codebook.reshape([1, s, d])
    vecs = codebook[:, -l:, :].reshape(1, h, l, d)
    # 调用 get_shortcodes 函数
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=codebook)
    # 使用 NumPy 测试模块中的 assert_allclose 函数，比较 shortcodes 和 jnp.tile(jnp.arange(s - l, s).reshape([1, 1, l]), reps=[1, h, 1]) 的值是否在允许的误差范围内相等
    np.testing.assert_allclose(
        # 实际值为 shortcodes
        actual=shortcodes,
        # 期望值为 jnp.tile(jnp.arange(s - l, s).reshape([1, 1, l]), reps=[1, h, 1])
        desired=jnp.tile(jnp.arange(s - l, s).reshape([1, 1, l]), reps=[1, h, 1]),
    )
# 测试获取码字
def test_get_codewords():
    # 定义参数 h, l, s, d
    h, l, s, d = 3, 5, 7, 11
    # 生成码本
    codebook = jax.nn.one_hot(jnp.arange(s), num_classes=d)
    # 生成码本
    codebook = jnp.arange(1, h + 1).reshape([h, 1, 1]) * codebook.reshape([1, s, d])
    # 生成向量
    vecs = codebook[:, -l:, :].reshape(1, h, l, d)
    # 获取短码
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=codebook)
    # 获取码字
    codewords = get_codewords(shortcodes=shortcodes, codebook=codebook)
    # 断言码字与向量相等
    np.testing.assert_allclose(actual=codewords, desired=vecs)


# 测试获取码本 EMA 目标
def test_get_codebook_ema_targets(basic_vq_spec_fixture):
    # 定义参数 h, l, s, d
    h, l, s, d = 3, 5, 7, 11
    # 定义参数 c_gamma
    c_gamma = 0.99
    # 生成全为1的数组
    ones = jnp.ones([h, s], dtype=jnp.float64)
    # 初始化计数
    c_count = ones
    # 生成码本
    c_sum = jax.nn.one_hot(jnp.arange(s), num_classes=d)
    # 生成码本
    c_sum = jnp.arange(1, h + 1).reshape([h, 1, 1]) * c_sum.reshape([1, s, d])
    # 生成向量
    vecs = c_sum[:, -l:, :].reshape(1, h, l, d)
    # 向向量添加0.1
    vecs = vecs + 0.1 * jnp.ones_like(vecs)
    # 获取短码
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=c_sum)
    # 断言短码与预期相等
    np.testing.assert_allclose(
        actual=shortcodes,
        desired=jnp.tile(jnp.arange(s - l, s).reshape([1, 1, l]), reps=[1, h, 1]),
    )
    # 计算码本 EMA 目标
    c_sum_tgt_expected = c_gamma * c_sum + (1 - c_gamma) * jnp.pad(
        vecs[0, ...], ((0, 0), (s - l, 0), (0, 0))
    )
    c_count_tgt_expected = c_gamma * c_count + (1 - c_gamma) * jnp.pad(
        ones[:, -l:], ((0, 0), (s - l, 0))
    )
    # 获取码本 EMA 目标
    c_sum_tgt_actual, c_count_tgt_actual = LearnableVQ.get_codebook_ema_targets(
        vecs=vecs,
        shortcodes=shortcodes,
        c_sum=c_sum,
        c_count=c_count,
        c_gamma=c_gamma,
        vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=l),
    )
    # 断言码本 EMA 目标与预期相等
    np.testing.assert_allclose(actual=c_sum_tgt_actual, desired=c_sum_tgt_expected)
    np.testing.assert_allclose(actual=c_count_tgt_actual, desired=c_count_tgt_expected)


# 测试获取码本损失 EMA
def test_get_codebook_loss_ema(rng_fixture, basic_vq_spec_fixture):
    # 定义参数 h, l, s, d
    h, l, s, d = 3, 22, 11, 41
    # 定义参数 c_gamma
    c_gamma = 0.99
    # 生成码本
    c_sum = jax.nn.one_hot(jnp.arange(s), num_classes=d)
    # 计算 c_sum，使用 jnp.arange 生成 1 到 h 的数组，reshape 成 [h, 1, 1]，与 c_sum 的 reshape 成 [1, s, d] 进行逐元素相乘
    c_sum = jnp.arange(1, h + 1).reshape([h, 1, 1]) * c_sum.reshape([1, s, d])
    # 创建一个形状为 [h, s] 的数组，元素值都为 2，数据类型为 jnp.float64
    c_count = 2 * jnp.ones([h, s], dtype=jnp.float64)
    # 创建一个字典，包含 c_sum 和 c_count
    params = dict(c_sum=c_sum, c_count=c_count)
    # 生成一个形状为 [1, h, l, d] 的正态分布随机数组，数据类型为 jnp.float64
    vecs = jax.random.normal(rng_fixture(0), shape=[1, h, l, d], dtype=jnp.float64)
    # 调用 get_shortcodes 函数，传入参数 vecs 和 c_sum，返回结果存储在 shortcodes 中，忽略第二个返回值
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=c_sum)

    # 计算预期的更新
    r = jax.nn.one_hot(shortcodes, num_classes=s, dtype=jnp.float64)
    c_sum_hat = jnp.einsum("bhts,bhtd->hsd", r, vecs)
    c_count_hat = jnp.sum(r, axis=(0, 2))
    c_sum_new_expected = c_gamma * c_sum + (1 - c_gamma) * c_sum_hat
    c_count_new_expected = c_gamma * c_count + (1 - c_gamma) * c_count_hat

    # 定义损失函数 loss_fn，接受参数 params_, vecs_, shortcodes_, vq_spec_，返回 LearnableVQ.get_codebook_loss 的结果
    def loss_fn(params_, vecs_, shortcodes_, vq_spec_):
        return LearnableVQ.get_codebook_loss(
            vecs=vecs_,
            shortcodes=shortcodes_,
            c_sum=params_["c_sum"],
            c_count=params_["c_count"],
            c_gamma=c_gamma,
            vq_spec=vq_spec_,
        )

    # 对每个参数计算梯度，存储在 grads 中
    grads = jax.grad(loss_fn)(
        params,
        vecs_=vecs,
        shortcodes_=shortcodes,
        vq_spec_=basic_vq_spec_fixture(batch_size=1, block_len=l),
    )
    # 计算实际的 c_sum 和 c_count 的更新值
    c_sum_new_actual = c_sum - grads["c_sum"]
    c_count_new_actual = c_count - grads["c_count"]
    # 使用 np.testing.assert_allclose 检查实际更新值和预期更新值是否接近
    np.testing.assert_allclose(actual=c_sum_new_actual, desired=c_sum_new_expected)
    np.testing.assert_allclose(actual=c_count_new_actual, desired=c_count_new_expected)

    # 对每个参数计算梯度，存储在 grads_block0 中，用于多个块的更新
    vq_spec = basic_vq_spec_fixture(
        batch_size=1, block_len=l // 2, n_block_per_update=2, block_id=0
    )
    grads_block0 = jax.grad(loss_fn)(
        params,
        vecs_=vecs[:, :, 0 : l // 2, :],
        shortcodes_=shortcodes[:, :, 0 : l // 2],
        vq_spec_=vq_spec,
    )
    # 更新 vq_spec
    vq_spec = basic_vq_spec_fixture(
        batch_size=1, block_len=l // 2, n_block_per_update=2, block_id=1
    )
    # 计算损失函数对参数的梯度
    grads_block1 = jax.grad(loss_fn)(
        params,
        vecs_=vecs[:, :, l // 2 :, :],
        shortcodes_=shortcodes[:, :, l // 2 :],
        vq_spec_=vq_spec,
    )
    # 将两个梯度对象的对应元素求和并乘以0.5
    grads = jax.tree_util.tree_map(
        lambda a, b: 0.5 * (a + b), grads_block0, grads_block1
    )
    # 计算新的 c_sum 的值
    c_sum_new_actual = c_sum - grads["c_sum"]
    # 计算新的 c_count 的值
    c_count_new_actual = c_count - grads["c_count"]
    # 检查新的 c_sum_new_actual 是否接近期望值 c_sum_new_expected
    np.testing.assert_allclose(actual=c_sum_new_actual, desired=c_sum_new_expected)
    # 检查新的 c_count_new_actual 是否接近期望值 c_count_new_expected
    np.testing.assert_allclose(actual=c_count_new_actual, desired=c_count_new_expected)
# 测试获取代码本书丢失EMA多设备
def test_get_codebook_loss_ema_multidev(rng_fixture, multidev_vq_spec_fixture):
    # 断言本地设备数量等于N_LOCAL_DEVICE
    assert jax.device_count() == N_LOCAL_DEVICE
    # 断言本地设备数量等于N_LOCAL_DEVICE
    assert jax.local_device_count() == N_LOCAL_DEVICE
    # 计算b的值
    b = 3 * jax.local_device_count()
    # 初始化h, l, s, d的值
    h, l, s, d = 3, 22, 11, 41
    # 初始化c_gamma的值
    c_gamma = 0.99
    # 生成服从正态分布的随机数c_sum
    c_sum = jax.random.normal(rng_fixture(0), shape=[h, s, d], dtype=jnp.float64)
    # 初始化c_count的值
    c_count = 2.0 * jnp.ones([h, s], dtype=jnp.float64)
    # 将c_sum和c_count封装成字典
    params = dict(c_sum=c_sum, c_count=c_count)
    # 生成服从正态分布的随机数vecs
    vecs = jax.random.normal(rng_fixture(1), shape=[b, h, l, d], dtype=jnp.float64)
    # 调用get_shortcodes函数获取shortcodes
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=c_sum)

    # 生成预期更新
    r = jax.nn.one_hot(shortcodes, num_classes=s, dtype=jnp.float64)
    c_sum_hat = jnp.einsum("bhts,bhtd->hsd", r, vecs)
    c_count_hat = jnp.sum(r, axis=(0, 2))
    c_sum_new_expected = c_gamma * c_sum + (1 - c_gamma) * c_sum_hat
    c_count_new_expected = c_gamma * c_count + (1 - c_gamma) * c_count_hat

    # 使用jax.pmap创建一个新的函数grad_fn
    @functools.partial(jax.pmap, axis_name="devices")
    def grad_fn(params_, vecs_, shortcodes_, vq_spec_):
        # 定义损失函数loss_fn
        def loss_fn(params__, vecs__, shortcodes__, vq_spec__):
            return LearnableVQ.get_codebook_loss(
                vecs=vecs__,
                shortcodes=shortcodes__,
                c_sum=params__["c_sum"],
                c_count=params__["c_count"],
                c_gamma=c_gamma,
                vq_spec=vq_spec__,
            )
        # 计算梯度
        grads_ = jax.grad(loss_fn)(params_, vecs_, shortcodes_, vq_spec_)
        return jax.lax.pmean(grads_, axis_name="devices")

    # 对每个更新进行单个块处理
    grads = grad_fn(
        jax_utils.replicate(params),
        vecs_=common_utils.shard(vecs),
        shortcodes_=common_utils.shard(shortcodes),
        vq_spec_=multidev_vq_spec_fixture(batch_size=b, block_len=l),
    )
    # 取消复制梯度
    grads = jax_utils.unreplicate(grads)
    # 计算实际的c_sum_new和c_count_new
    c_sum_new_actual = c_sum - grads["c_sum"]
    c_count_new_actual = c_count - grads["c_count"]
    # 使用 np.testing.assert_allclose 函数比较 c_sum_new_actual 和 c_sum_new_expected 的值，使用 TOLERANCES 中定义的容差
    np.testing.assert_allclose(
        actual=c_sum_new_actual, desired=c_sum_new_expected, **TOLERANCES
    )
    # 使用 np.testing.assert_allclose 函数比较 c_count_new_actual 和 c_count_new_expected 的值，使用 TOLERANCES 中定义的容差
    np.testing.assert_allclose(
        actual=c_count_new_actual, desired=c_count_new_expected, **TOLERANCES
    )

    # 对每个更新进行多个块的处理
    # 计算第一个块的梯度
    grads_block0 = grad_fn(
        jax_utils.replicate(params),
        vecs_=common_utils.shard(vecs[:, :, 0 : l // 2, :]),
        shortcodes_=common_utils.shard(shortcodes[:, :, 0 : l // 2]),
        vq_spec_=multidev_vq_spec_fixture(
            batch_size=b, block_len=l // 2, n_block_per_update=2, block_id=0
        ),
    )
    # 取消梯度的复制
    grads_block0 = jax_utils.unreplicate(grads_block0)
    # 计算第二个块的梯度
    grads_block1 = grad_fn(
        jax_utils.replicate(params),
        vecs_=common_utils.shard(vecs[:, :, l // 2 :, :]),
        shortcodes_=common_utils.shard(shortcodes[:, :, l // 2 :]),
        vq_spec_=multidev_vq_spec_fixture(
            batch_size=b, block_len=l // 2, n_block_per_update=2, block_id=0
        ),
    )
    # 取消梯度的复制
    grads_block1 = jax_utils.unreplicate(grads_block1)
    # 将两个块的梯度合并
    grads = jax.tree_util.tree_map(
        lambda a, b: 0.5 * (a + b), grads_block0, grads_block1
    )
    # 计算新的 c_sum_actual 值
    c_sum_new_actual = c_sum - grads["c_sum"]
    # 计算新的 c_count_actual 值
    c_count_new_actual = c_count - grads["c_count"]
    # 使用 np.testing.assert_allclose 函数比较 c_sum_new_actual 和 c_sum_new_expected 的值，使用 TOLERANCES 中定义的容差
    np.testing.assert_allclose(
        actual=c_sum_new_actual, desired=c_sum_new_expected, **TOLERANCES
    )
    # 使用 np.testing.assert_allclose 函数比较 c_count_new_actual 和 c_count_new_expected 的值，使用 TOLERANCES 中定义的容差
    np.testing.assert_allclose(
        actual=c_count_new_actual, desired=c_count_new_expected, **TOLERANCES
    )
# 定义测试向量量化器调用的函数，接受多个参数
def test_vector_quantizer_call(
    rng_fixture, quantizer_fixture, transformer_config_fixture, basic_vq_spec_fixture
):
    # 使用给定的配置参数创建配置对象
    config = transformer_config_fixture(
        block_len=100,
        mem_len=100,  # not used by VectorQuantizer
        agg_cache=True,
        widening=1,
        dtypes=jnp.float32,
        is_train=True,
    )
    # 设置向量量化器的类
    cls = LearnableVQ
    # 使用给定的类和配置参数创建量化器对象和参数
    _, params = quantizer_fixture(cls, config)

    # 设置向量的批量大小、块长度、头数和维度
    b = 3
    l = config.block_len
    h = config.n_head
    d = config.d_k
    # 生成指定形状和数据类型的随机向量
    vecs = jax.random.normal(rng_fixture(1), shape=[b, h, l, d], dtype=jnp.float64)
    # 创建随机数生成器字典
    rngs = dict(
        ephemeral=rng_fixture(2),
        timeless=rng_fixture(3),
    )

    # 定义获取提交损失的函数，接受向量和参数作为输入
    def get_commit_loss(vecs_, params_):
        # 调用向量量化器的应用方法，返回提交损失
        return cls(config).apply(
            {"params": params_},
            vecs=vecs_,
            vq_spec=basic_vq_spec_fixture(batch_size=b, block_len=l),
            rngs=rngs,
        )["l_commit"]

    # 定义获取码本损失的函数，接受参数和向量作为输入
    def get_codebook_loss(params_, vecs_):
        # 调用向量量化器的应用方法，返回码本损失
        return cls(config).apply(
            {"params": params_},
            vecs=vecs_,
            vq_spec=basic_vq_spec_fixture(batch_size=b, block_len=l),
            rngs=rngs,
        )["l_codebook"]

    # 计算提交损失的梯度
    grad = jax.grad(get_commit_loss)(vecs, params_=params)
    # 使用断言检查梯度的所有叶子节点是否都接近零
    with pytest.raises(AssertionError):
        for leaf in jax.tree_util.tree_leaves(grad):
            np.testing.assert_allclose(actual=leaf, desired=jnp.zeros_like(leaf))

    # 计算码本损失的梯度
    grad = jax.grad(get_codebook_loss)(params, vecs_=vecs)
    # 使用断言检查梯度的所有叶子节点是否都接近零
    with pytest.raises(AssertionError):
        for leaf in jax.tree_util.tree_leaves(grad):
            np.testing.assert_allclose(actual=leaf, desired=jnp.zeros_like(leaf))
```