# `transformer_vq\tests\utils\test_io.py`

```
# 导入所需的模块
import os  # 操作系统相关的功能
import tempfile  # 临时文件和目录的功能

import flax  # 用于构建神经网络的库
import flax.linen as nn  # Flax 的神经网络模块
import jax  # 用于自动微分和并行计算的库
import jax.numpy as jnp  # JAX 的 NumPy 替代品
import numpy as np  # 数学计算库 NumPy
import optax  # 优化库
import pytest  # Python 的测试框架
import tensorflow as tf  # 用于构建和训练深度学习模型的库
from flax.training.train_state import TrainState  # Flax 的训练状态模块

from transformer_vq.utils.io import check_not_none  # 自定义的输入输出工具函数
from transformer_vq.utils.io import load_checkpoint  # 自定义的输入输出工具函数
from transformer_vq.utils.io import save_checkpoint  # 自定义的输入输出工具函数
from transformer_vq.utils.io import save_pixels  # 自定义的输入输出工具函数

# noreorder
from tests.common import rng_fixture  # 导入测试用的随机数生成器
# 定义常量，表示步数、词汇量、批处理大小和序列长度
STEPS = 7357
N_VOCAB = 123
BATCH_SIZE = 1
SEQUENCE_LEN = 100

# 定义一个测试函数，用于检查参数是否为 None
def test_check_not_none():
    # 检查参数是否为 None
    check_not_none("something")
    # 使用 pytest 检查是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        check_not_none(None)

# 定义一个神经网络模型类
class Model(nn.Module):
    # 使用 nn.compact 装饰器定义神经网络的前向传播过程
    @nn.compact
    def __call__(self, inputs):
        # 使用 Embed 层将输入数据映射到 1000 维的向量空间
        x = nn.Embed(N_VOCAB, 1000)(inputs)
        # 使用 relu 激活函数
        x = nn.relu(x)
        # 使用全连接层将输入数据映射到 1000 维的向量空间
        x = nn.Dense(1000)(x)
# 对输入数据进行 ReLU 激活函数处理
x = nn.relu(x)
# 使用全连接层对数据进行处理，输出词汇表大小的结果
x = nn.Dense(N_VOCAB)(x)
# 返回处理后的结果
return x

# 定义训练状态的 fixture
@pytest.fixture
def train_state():
    # 定义内部函数 _train_state，接受初始化随机数生成器作为参数
    def _train_state(init_rng):
        # 创建模型对象
        model = Model()
        # 将初始化随机数生成器分割成两部分
        sk1, sk2 = jax.random.split(init_rng)
        # 使用模型的初始化方法初始化参数，并解冻参数
        params = model.init(
            {"params": sk1},
            inputs=jnp.ones(dtype=jnp.int32, shape=[BATCH_SIZE, SEQUENCE_LEN]),
        )["params"].unfreeze()
        # 使用随机梯度下降优化器
        tx = optax.sgd(learning_rate=0.01)
        # 创建训练状态对象，包括模型应用函数、参数、优化器
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        # 替换训练状态的步数
        state = state.replace(step=STEPS)
        # 返回训练状态对象
        return state

    # 返回内部函数
    return _train_state
# 测试保存检查点的函数
def test_save_checkpoint(rng_fixture, train_state, tmp_path):
    # 生成训练状态
    train_state = train_state(rng_fixture(0))
    # 保存检查点
    save_checkpoint(
        save_dir=str(tmp_path),  # 保存路径
        target=train_state,  # 目标训练状态
        step=STEPS,  # 步数
        prefix="checkpoint",  # 文件名前缀
        keep=5,  # 保留的检查点数量
    )
    # 断言检查点文件是否存在
    assert os.path.exists(os.path.join(tmp_path, f"checkpoint_{STEPS}"))


# 测试加载检查点的函数
def test_load_checkpoint(rng_fixture, train_state, tmp_path):
    # 生成训练状态
    state = train_state(rng_fixture(0))
    # 将状态转换为字节流
    state_bytes = flax.serialization.to_bytes(state)
    # 保存检查点
    save_checkpoint(
        save_dir=str(tmp_path),  # 保存路径
        target=state,  # 目标状态
# 设置步数、前缀和保留的检查点数量
step=STEPS,
prefix="checkpoint",
keep=5,
)
# 检查是否存在指定步数的检查点文件
assert os.path.exists(os.path.join(tmp_path, f"checkpoint_{STEPS}"))
# 加载检查点
state_loaded = load_checkpoint(
    load_dir=str(tmp_path),
    train_state=state,
    prefix="checkpoint",
)
# 将加载的状态转换为字节流
state_bytes_loaded = flax.serialization.to_bytes(state_loaded)
# 检查加载的状态字节流与原状态字节流是否相等
assert state_bytes == state_bytes_loaded

# 测试保存像素数据
# 获取当前文件所在目录
dir1 = os.path.dirname(os.path.realpath(__file__))
# 设置示例图片文件名
fname1 = "ref_img.png"  # example image taken from reddit's /r/programmerhumor
# 以二进制读取示例图片数据
with tf.io.gfile.GFile(tf.io.gfile.join(dir1, fname1), "rb") as f1:
    data1 = f1.read()
# 解码 PNG 格式的图片数据
data1 = tf.io.decode_png(data1)
# 创建临时目录对象
tempdir = tempfile.TemporaryDirectory()
# 获取临时目录的路径
dir2 = tempdir.name
# 设置文件名
fname2 = "saved_pixels.png"
# 调用 save_pixels 函数保存数据到指定目录和文件名
save_pixels(data1, dir2, fname2)

# 使用 tf.io.gfile.GFile 打开文件，以二进制模式读取数据
with tf.io.gfile.GFile(tf.io.gfile.join(dir2, fname2), "rb") as f2:
    # 读取文件数据
    data2 = f2.read()
# 解码 PNG 格式的数据
data2 = tf.io.decode_png(data2)
# 使用 np.testing.assert_allclose 检查两个数据是否近似相等
np.testing.assert_allclose(data2, data1)
# 清理临时目录
tempdir.cleanup()
```