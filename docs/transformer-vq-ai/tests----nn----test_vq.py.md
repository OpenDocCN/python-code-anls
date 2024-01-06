# `transformer_vq\tests\nn\test_vq.py`

```
# 导入 functools 模块，用于高阶函数和操作
import functools
# 导入 os 模块，用于与操作系统交互
import os

# 设置本地设备数量为 8
N_LOCAL_DEVICE = 8
# 设置环境变量 XLA_FLAGS，指定 XLA 强制使用的本地设备数量
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_LOCAL_DEVICE}"

# 导入 jax 模块，用于自动微分和加速数值计算
import jax
# 导入 jax 中的 numpy 模块，用于支持自动微分的数值计算
import jax.numpy as jnp
# 导入 numpy 模块，用于数值计算
import numpy as np
# 导入 pytest 模块，用于编写测试
import pytest
# 导入 flax 中的 jax_utils 模块，用于支持 JAX 的工具函数
from flax import jax_utils
# 导入 flax 中的 common_utils 模块，用于支持训练的常用工具函数
from flax.training import common_utils

# 从 transformer_vq.nn.vq 模块中导入 LearnableVQ、get_shortcodes、get_codewords 函数
from transformer_vq.nn.vq import LearnableVQ, get_shortcodes, get_codewords
# 从 transformer_vq.nn.types 模块中导入 TransformerConfig 类型
from transformer_vq.nn.types import TransformerConfig

# 导入测试用的 transformer_config_fixture、rng_fixture、basic_vq_spec_fixture
# noreorder 表示不重新排序导入的模块
from tests.common import transformer_config_fixture
from tests.common import rng_fixture
from tests.common import basic_vq_spec_fixture
# 从 tests.common 模块中导入 multidev_vq_spec_fixture 和 TOLERANCES
from tests.common import multidev_vq_spec_fixture
from tests.common import TOLERANCES

# 更新 JAX 配置，启用 64 位浮点数支持
jax.config.update("jax_enable_x64", True)

# 定义测试函数 test_num_devices
def test_num_devices():
    # 断言 JAX 设备数量等于 N_LOCAL_DEVICE
    assert jax.device_count() == N_LOCAL_DEVICE
    # 断言 JAX 本地设备数量等于 N_LOCAL_DEVICE
    assert jax.local_device_count() == N_LOCAL_DEVICE

# 定义夹具函数 quantizer_fixture，接受 rng_fixture 和 basic_vq_spec_fixture 作为参数
@pytest.fixture
def quantizer_fixture(rng_fixture, basic_vq_spec_fixture):
    # 定义内部函数 _make_quantizer，接受 cls 和 config 作为参数
    def _make_quantizer(cls, config: TransformerConfig):
        # 获取配置中的编码长度
        present_len = config.n_code
        # 定义输入形状
        inputs_shape = [1, config.n_head, present_len, config.d_k]
        # 生成符合正态分布的随机输入数据
        inputs = jax.random.normal(rng_fixture(0), inputs_shape, dtype=jnp.float32)
        # 定义随机数种子字典
        rngs = dict(
            params=rng_fixture(1),
# 定义测试函数 test_get_shortcodes
def test_get_shortcodes():
    # 定义变量 h, l, s, d 分别赋值为 3, 5, 7, 11
    h, l, s, d = 3, 5, 7, 11
    # 生成一个 one-hot 编码的 codebook，其中 num_classes 为 d
    codebook = jax.nn.one_hot(jnp.arange(s), num_classes=d)
    # 重新定义 codebook，将其转换为 h x s x d 的形状
    codebook = jnp.arange(1, h + 1).reshape([h, 1, 1]) * codebook.reshape([1, s, d])
# 从codebook中取出最后l个元素，然后reshape成1个h*l*d的数组vecs
vecs = codebook[:, -l:, :].reshape(1, h, l, d)

# 调用get_shortcodes函数，传入vecs和codebook参数，获取shortcodes和_
shortcodes, _ = get_shortcodes(vecs=vecs, codebook=codebook)

# 使用np.testing.assert_allclose函数对shortcodes和jnp.tile(jnp.arange(s - l, s).reshape([1, 1, l]), reps=[1, h, 1])进行比较
np.testing.assert_allclose(
    actual=shortcodes,
    desired=jnp.tile(jnp.arange(s - l, s).reshape([1, 1, l]), reps=[1, h, 1]),
)

# 定义test_get_codewords函数
def test_get_codewords():
    # 定义变量h, l, s, d
    h, l, s, d = 3, 5, 7, 11

    # 使用jax.nn.one_hot函数生成codebook
    codebook = jax.nn.one_hot(jnp.arange(s), num_classes=d)
    codebook = jnp.arange(1, h + 1).reshape([h, 1, 1]) * codebook.reshape([1, s, d])

    # 从codebook中取出最后l个元素，然后reshape成1个h*l*d的数组vecs
    vecs = codebook[:, -l:, :].reshape(1, h, l, d)

    # 调用get_shortcodes函数，传入vecs和codebook参数，获取shortcodes和_
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=codebook)

    # 调用get_codewords函数，传入shortcodes和codebook参数，获取codewords
    codewords = get_codewords(shortcodes=shortcodes, codebook=codebook)

    # 使用np.testing.assert_allclose函数对codewords和vecs进行比较
    np.testing.assert_allclose(actual=codewords, desired=vecs)

# 定义test_get_codebook_ema_targets函数
def test_get_codebook_ema_targets(basic_vq_spec_fixture):
    # 定义变量h, l, s, d
    h, l, s, d = 3, 5, 7, 11
    # 设置参数 c_gamma 为 0.99
    c_gamma = 0.99
    # 创建一个 h x s 大小的全为 1 的浮点数数组
    ones = jnp.ones([h, s], dtype=jnp.float64)
    # 将 c_count 初始化为 ones
    c_count = ones
    # 使用 jax.nn.one_hot 生成一个 one-hot 编码的数组
    c_sum = jax.nn.one_hot(jnp.arange(s), num_classes=d)
    # 生成 c_sum 数组的另一种形式
    c_sum = jnp.arange(1, h + 1).reshape([h, 1, 1]) * c_sum.reshape([1, s, d])
    # 从 c_sum 中提取特定部分，然后重新组织成 1 x h x l x d 的数组
    vecs = c_sum[:, -l:, :].reshape(1, h, l, d)
    # 将 vecs 中的每个元素都加上 0.1
    vecs = vecs + 0.1 * jnp.ones_like(vecs)
    # 调用 get_shortcodes 函数，获取 shortcodes
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=c_sum)
    # 使用 np.testing.assert_allclose 检查 shortcodes 是否等于指定的数组
    np.testing.assert_allclose(
        actual=shortcodes,
        desired=jnp.tile(jnp.arange(s - l, s).reshape([1, 1, l]), reps=[1, h, 1]),
    )
    # 计算 c_sum_tgt_expected
    c_sum_tgt_expected = c_gamma * c_sum + (1 - c_gamma) * jnp.pad(
        vecs[0, ...], ((0, 0), (s - l, 0), (0, 0))
    )
    # 计算 c_count_tgt_expected
    c_count_tgt_expected = c_gamma * c_count + (1 - c_gamma) * jnp.pad(
        ones[:, -l:], ((0, 0), (s - l, 0))
    )
    # 调用 LearnableVQ.get_codebook_ema_targets 函数，获取 c_sum_tgt_actual 和 c_count_tgt_actual
    c_sum_tgt_actual, c_count_tgt_actual = LearnableVQ.get_codebook_ema_targets(
        vecs=vecs,
# 设置测试所需的参数和数据
h, l, s, d = 3, 22, 11, 41
c_gamma = 0.99
c_sum = jax.nn.one_hot(jnp.arange(s), num_classes=d)
c_sum = jnp.arange(1, h + 1).reshape([h, 1, 1]) * c_sum.reshape([1, s, d])
c_count = 2 * jnp.ones([h, s], dtype=jnp.float64)
params = dict(c_sum=c_sum, c_count=c_count)
vecs = jax.random.normal(rng_fixture(0), shape=[1, h, l, d], dtype=jnp.float64)

# 调用函数获取短码和其他结果
shortcodes, _ = get_shortcodes(vecs=vecs, codebook=c_sum)

# 断言实际值和期望值的近似程度
np.testing.assert_allclose(actual=c_sum_tgt_actual, desired=c_sum_tgt_expected)
np.testing.assert_allclose(actual=c_count_tgt_actual, desired=c_count_tgt_expected)
    # 创建一个独热编码的张量，用于表示shortcodes的类别
    r = jax.nn.one_hot(shortcodes, num_classes=s, dtype=jnp.float64)
    # 使用爱因斯坦求和约定计算c_sum_hat
    c_sum_hat = jnp.einsum("bhts,bhtd->hsd", r, vecs)
    # 计算c_count_hat，对r进行求和操作
    c_count_hat = jnp.sum(r, axis=(0, 2))
    # 计算c_sum_new_expected，根据公式进行更新
    c_sum_new_expected = c_gamma * c_sum + (1 - c_gamma) * c_sum_hat
    # 计算c_count_new_expected，根据公式进行更新
    c_count_new_expected = c_gamma * c_count + (1 - c_gamma) * c_count_hat

    # 定义损失函数，用于计算VQ模型的损失
    def loss_fn(params_, vecs_, shortcodes_, vq_spec_):
        return LearnableVQ.get_codebook_loss(
            vecs=vecs_,
            shortcodes=shortcodes_,
            c_sum=params_["c_sum"],
            c_count=params_["c_count"],
            c_gamma=c_gamma,
            vq_spec=vq_spec_,
        )

    # 计算损失函数关于参数的梯度
    grads = jax.grad(loss_fn)(
        params,
    # 设置参数 vecs_, shortcodes_, vq_spec_，并调用 basic_vq_spec_fixture 函数生成 vq_spec_
    vecs_=vecs,
    shortcodes_=shortcodes,
    vq_spec_=basic_vq_spec_fixture(batch_size=1, block_len=l),
)

    # 计算新的 c_sum 和 c_count
    c_sum_new_actual = c_sum - grads["c_sum"]
    c_count_new_actual = c_count - grads["c_count"]
    # 使用 np.testing.assert_allclose 函数检查 c_sum_new_actual 和 c_count_new_actual 是否等于期望值
    np.testing.assert_allclose(actual=c_sum_new_actual, desired=c_sum_new_expected)
    np.testing.assert_allclose(actual=c_count_new_actual, desired=c_count_new_expected)

    # 设置参数 vq_spec，并调用 basic_vq_spec_fixture 函数生成 vq_spec
    # 计算梯度 grads_block0
    vq_spec = basic_vq_spec_fixture(
        batch_size=1, block_len=l // 2, n_block_per_update=2, block_id=0
    )
    grads_block0 = jax.grad(loss_fn)(
        params,
        vecs_=vecs[:, :, 0 : l // 2, :],
        shortcodes_=shortcodes[:, :, 0 : l // 2],
        vq_spec_=vq_spec,
    )
    # 重新设置参数 vq_spec
    vq_spec = basic_vq_spec_fixture(
# 设置批处理大小、块长度、每次更新的块数和块ID
batch_size=1, block_len=l // 2, n_block_per_update=2, block_id=1
)
# 计算第一个块的梯度
grads_block0 = jax.grad(loss_fn)(
    params,
    vecs_=vecs[:, :, :l // 2, :],
    shortcodes_=shortcodes[:, :, :l // 2],
    vq_spec_=vq_spec,
)
# 计算第二个块的梯度
grads_block1 = jax.grad(loss_fn)(
    params,
    vecs_=vecs[:, :, l // 2 :, :],
    shortcodes_=shortcodes[:, :, l // 2 :],
    vq_spec_=vq_spec,
)
# 将两个块的梯度取平均
grads = jax.tree_util.tree_map(
    lambda a, b: 0.5 * (a + b), grads_block0, grads_block1
)
# 更新c_sum和c_count
c_sum_new_actual = c_sum - grads["c_sum"]
c_count_new_actual = c_count - grads["c_count"]
# 断言新的c_sum和c_count与期望值相等
np.testing.assert_allclose(actual=c_sum_new_actual, desired=c_sum_new_expected)
np.testing.assert_allclose(actual=c_count_new_actual, desired=c_count_new_expected)

# 测试获取代码本损失的EMA多设备
def test_get_codebook_loss_ema_multidev(rng_fixture, multidev_vq_spec_fixture):
    # 断言本地设备数量与预期值相等
    assert jax.device_count() == N_LOCAL_DEVICE
    # 断言本地设备数量与预期值相等
    assert jax.local_device_count() == N_LOCAL_DEVICE
    # 计算本地设备数量的3倍
    b = 3 * jax.local_device_count()
    # 初始化变量h, l, s, d
    h, l, s, d = 3, 22, 11, 41
    # 设置参数c_gamma为0.99
    c_gamma = 0.99
    # 从随机数生成器中生成符合正态分布的随机数，形状为[h, s, d]，数据类型为64位浮点数
    c_sum = jax.random.normal(rng_fixture(0), shape=[h, s, d], dtype=jnp.float64)
    # 创建一个形状为[h, s]，数据类型为64位浮点数的数组，每个元素为2.0
    c_count = 2.0 * jnp.ones([h, s], dtype=jnp.float64)
    # 创建一个包含c_sum和c_count的字典
    params = dict(c_sum=c_sum, c_count=c_count)
    # 从随机数生成器中生成符合正态分布的随机数，形状为[b, h, l, d]，数据类型为64位浮点数
    vecs = jax.random.normal(rng_fixture(1), shape=[b, h, l, d], dtype=jnp.float64)
    # 调用get_shortcodes函数，传入vecs和c_sum作为参数，返回shortcodes和一个占位符
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=c_sum)

    # 计算预期的更新值
    # 将shortcodes转换为one-hot编码，类别数为s，数据类型为64位浮点数
    r = jax.nn.one_hot(shortcodes, num_classes=s, dtype=jnp.float64)
    # 使用矩阵乘法计算c_sum_hat
    c_sum_hat = jnp.einsum("bhts,bhtd->hsd", r, vecs)
    # 计算c_count_hat
    c_count_hat = jnp.sum(r, axis=(0, 2))
    # 计算新的c_sum的预期值
    c_sum_new_expected = c_gamma * c_sum + (1 - c_gamma) * c_sum_hat
    # 计算新的c_count的预期值
    c_count_new_expected = c_gamma * c_count + (1 - c_gamma) * c_count_hat

    # 定义一个使用pmap的函数grad_fn，指定axis_name为"devices"
    @functools.partial(jax.pmap, axis_name="devices")
    def grad_fn(params_, vecs_, shortcodes_, vq_spec_):
        # 定义损失函数loss_fn，接受params__, vecs__, shortcodes__, vq_spec__作为参数
        def loss_fn(params__, vecs__, shortcodes__, vq_spec__):
            # 返回LearnableVQ类的get_codebook_loss方法的结果
            return LearnableVQ.get_codebook_loss(
# 将参数传递给函数，包括vecs、shortcodes、c_sum、c_count、c_gamma和vq_spec
params_ = dict(
    vecs=vecs__,
    shortcodes=shortcodes__,
    c_sum=params__["c_sum"],
    c_count=params__["c_count"],
    c_gamma=c_gamma,
    vq_spec=vq_spec__,
)

# 计算损失函数的梯度
grads_ = jax.grad(loss_fn)(params_, vecs_, shortcodes_, vq_spec_)
# 对梯度进行平均处理
return jax.lax.pmean(grads_, axis_name="devices")

# 每次更新只处理一个块
# 计算梯度
grads = grad_fn(
    jax_utils.replicate(params),
    vecs_=common_utils.shard(vecs),
    shortcodes_=common_utils.shard(shortcodes),
    vq_spec_=multidev_vq_spec_fixture(batch_size=b, block_len=l),
)
# 取消梯度的复制
grads = jax_utils.unreplicate(grads)
# 计算新的c_sum的实际值
c_sum_new_actual = c_sum - grads["c_sum"]
    # 计算新的 c_count 值
    c_count_new_actual = c_count - grads["c_count"]
    # 检查 c_sum_new_actual 是否接近 c_sum_new_expected，使用 TOLERANCES 中定义的容差
    np.testing.assert_allclose(
        actual=c_sum_new_actual, desired=c_sum_new_expected, **TOLERANCES
    )
    # 检查 c_count_new_actual 是否接近 c_count_new_expected，使用 TOLERANCES 中定义的容差
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
        # ...
    )
# 将vecs按照指定规则进行分片
vecs_ = common_utils.shard(vecs[:, :, l // 2 :, :])
# 将shortcodes按照指定规则进行分片
shortcodes_ = common_utils.shard(shortcodes[:, :, l // 2 :])
# 创建一个多设备VQ规范的fixture
vq_spec_ = multidev_vq_spec_fixture(
    batch_size=b, block_len=l // 2, n_block_per_update=2, block_id=0
)
# 取消梯度的复制
grads_block1 = jax_utils.unreplicate(grads_block1)
# 对梯度进行操作，计算梯度的平均值
grads = jax.tree_util.tree_map(
    lambda a, b: 0.5 * (a + b), grads_block0, grads_block1
)
# 计算新的c_sum的值
c_sum_new_actual = c_sum - grads["c_sum"]
# 计算新的c_count的值
c_count_new_actual = c_count - grads["c_count"]
# 对比新的c_sum的值和期望的c_sum的值，检查它们是否在指定的误差范围内
np.testing.assert_allclose(
    actual=c_sum_new_actual, desired=c_sum_new_expected, **TOLERANCES
)
# 对比新的c_count的值和期望的c_count的值，检查它们是否在指定的误差范围内
np.testing.assert_allclose(
    actual=c_count_new_actual, desired=c_count_new_expected, **TOLERANCES
)
# 定义一个测试向量量化器调用的函数，接受多个参数
def test_vector_quantizer_call(
    rng_fixture, quantizer_fixture, transformer_config_fixture, basic_vq_spec_fixture
):
    # 使用给定的参数创建一个转换器配置
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
    # 使用给定的类和配置参数创建量化器实例和参数
    _, params = quantizer_fixture(cls, config)

    # 设置向量的维度
    b = 3
    l = config.block_len
    h = config.n_head
    d = config.d_k
    # 生成指定形状和数据类型的随机向量
    vecs = jax.random.normal(rng_fixture(1), shape=[b, h, l, d], dtype=jnp.float64)
    # 创建一个随机数生成器的字典
    rngs = dict(
# 定义了两个参数，一个是ephemeral，一个是timeless，分别使用rng_fixture函数生成随机数
ephemeral=rng_fixture(2),
timeless=rng_fixture(3),

# 定义了一个函数get_commit_loss，接受vecs_和params_两个参数，返回cls(config).apply()的结果中的["l_commit"]部分
def get_commit_loss(vecs_, params_):
    return cls(config).apply(
        {"params": params_},
        vecs=vecs_,
        vq_spec=basic_vq_spec_fixture(batch_size=b, block_len=l),
        rngs=rngs,
    )["l_commit"]

# 定义了一个函数get_codebook_loss，接受params_和vecs_两个参数，返回cls(config).apply()的结果中的["l_codebook"]部分
def get_codebook_loss(params_, vecs_):
    return cls(config).apply(
        {"params": params_},
        vecs=vecs_,
        vq_spec=basic_vq_spec_fixture(batch_size=b, block_len=l),
        rngs=rngs,
    )["l_codebook"]
# 使用自动微分计算 get_commit_loss 函数对 vecs 和 params 进行微分
grad = jax.grad(get_commit_loss)(vecs, params_=params)
# 使用 pytest 检查是否会抛出 AssertionError 异常
with pytest.raises(AssertionError):
    # 遍历 grad 中的每个叶子节点
    for leaf in jax.tree_util.tree_leaves(grad):
        # 使用 numpy.testing.assert_allclose 检查 leaf 是否接近于零向量
        np.testing.assert_allclose(actual=leaf, desired=jnp.zeros_like(leaf))

# 使用自动微分计算 get_codebook_loss 函数对 params 和 vecs 进行微分
grad = jax.grad(get_codebook_loss)(params, vecs_=vecs)
# 使用 pytest 检查是否会抛出 AssertionError 异常
with pytest.raises(AssertionError):
    # 遍历 grad 中的每个叶子节点
    for leaf in jax.tree_util.tree_leaves(grad):
        # 使用 numpy.testing.assert_allclose 检查 leaf 是否接近于零向量
        np.testing.assert_allclose(actual=leaf, desired=jnp.zeros_like(leaf))
```