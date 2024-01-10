# `transformer_vq\tests\utils\test_io.py`

```
# 导入所需的库
import os
import tempfile

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow as tf
from flax.training.train_state import TrainState

# 从自定义模块中导入所需的函数
from transformer_vq.utils.io import check_not_none
from transformer_vq.utils.io import load_checkpoint
from transformer_vq.utils.io import save_checkpoint
from transformer_vq.utils.io import save_pixels

# 从测试模块中导入所需的函数
from tests.common import rng_fixture

# 定义全局变量
STEPS = 7357
N_VOCAB = 123
BATCH_SIZE = 1
SEQUENCE_LEN = 100

# 定义测试函数，检查参数是否为 None
def test_check_not_none():
    check_not_none("something")
    with pytest.raises(ValueError):
        check_not_none(None)

# 定义神经网络模型类
class Model(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        x = nn.Embed(N_VOCAB, 1000)(inputs)
        x = nn.relu(x)
        x = nn.Dense(1000)(x)
        x = nn.relu(x)
        x = nn.Dense(N_VOCAB)(x)
        return x

# 定义测试函数，返回训练状态
@pytest.fixture
def train_state():
    def _train_state(init_rng):
        model = Model()
        sk1, sk2 = jax.random.split(init_rng)
        params = model.init(
            {"params": sk1},
            inputs=jnp.ones(dtype=jnp.int32, shape=[BATCH_SIZE, SEQUENCE_LEN]),
        )["params"].unfreeze()
        tx = optax.sgd(learning_rate=0.01)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        state = state.replace(step=STEPS)
        return state

    return _train_state

# 定义测试函数，保存训练状态的检查点
def test_save_checkpoint(rng_fixture, train_state, tmp_path):
    train_state = train_state(rng_fixture(0))
    save_checkpoint(
        save_dir=str(tmp_path),
        target=train_state,
        step=STEPS,
        prefix="checkpoint",
        keep=5,
    )
    assert os.path.exists(os.path.join(tmp_path, f"checkpoint_{STEPS}"))

# 定义测试函数，加载训练状态的检查点
def test_load_checkpoint(rng_fixture, train_state, tmp_path):
    state = train_state(rng_fixture(0))
    state_bytes = flax.serialization.to_bytes(state)
    # 保存检查点，将状态保存到指定目录下，设置保存步数、前缀和保留的检查点数量
    save_checkpoint(
        save_dir=str(tmp_path),
        target=state,
        step=STEPS,
        prefix="checkpoint",
        keep=5,
    )
    # 断言检查点文件是否存在
    assert os.path.exists(os.path.join(tmp_path, f"checkpoint_{STEPS}"))
    # 从指定目录加载检查点，加载训练状态，设置前缀
    state_loaded = load_checkpoint(
        load_dir=str(tmp_path),
        train_state=state,
        prefix="checkpoint",
    )
    # 将加载的状态转换为字节流
    state_bytes_loaded = flax.serialization.to_bytes(state_loaded)
    # 断言原始状态和加载的状态是否相等
    assert state_bytes == state_bytes_loaded
# 定义一个测试函数，用于测试保存像素数据的函数
def test_save_pixels():
    # 获取当前文件所在目录的绝对路径
    dir1 = os.path.dirname(os.path.realpath(__file__))
    # 设置要读取的图片文件名
    fname1 = "ref_img.png"  # example image taken from reddit's /r/programmerhumor
    # 使用 TensorFlow 的文件操作工具打开并读取图片文件
    with tf.io.gfile.GFile(tf.io.gfile.join(dir1, fname1), "rb") as f1:
        data1 = f1.read()
    # 对读取的图片数据进行 PNG 解码
    data1 = tf.io.decode_png(data1)

    # 创建一个临时目录
    tempdir = tempfile.TemporaryDirectory()
    # 获取临时目录的路径
    dir2 = tempdir.name
    # 设置要保存的图片文件名
    fname2 = "saved_pixels.png"
    # 调用保存像素数据的函数，将 data1 保存到 dir2 目录下的 fname2 文件中
    save_pixels(data1, dir2, fname2)

    # 使用 TensorFlow 的文件操作工具打开并读取保存的图片文件
    with tf.io.gfile.GFile(tf.io.gfile.join(dir2, fname2), "rb") as f2:
        data2 = f2.read()
    # 对读取的图片数据进行 PNG 解码
    data2 = tf.io.decode_png(data2)
    # 使用 NumPy 的测试工具，断言两个图片数据数组的所有元素是否全部接近
    np.testing.assert_allclose(data2, data1)
    # 清理临时目录
    tempdir.cleanup()
```