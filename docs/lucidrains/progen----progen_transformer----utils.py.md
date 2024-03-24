# `.\lucidrains\progen\progen_transformer\utils.py`

```
# 从 math 模块中导入 ceil 函数
from math import ceil
# 导入 os 和 errno 模块
import os, errno
# 从 shutil 模块中导入 rmtree 函数
from shutil import rmtree

# 导入 jax 库
import jax
# 从 jax 库中导入 random, nn, value_and_grad, vmap, pmap, jit, lax 模块
from jax import random, nn, value_and_grad, vmap, pmap, jit, lax
# 从 jax.numpy 模块中导入 np 别名
import jax.numpy as np

# 从 einops 模块中导入 rearrange 函数

from einops import rearrange

# 辅助函数

# 定义一个空操作函数
def noop(x):
    return x

# 判断值是否存在的函数
def exists(val):
    return val is not None

# 计算对数的函数
def log(t, eps = 1e-20):
    return np.log(t + eps)

# 确认函数
def confirm(question):
    while True:
        resp = input(f'{question} (y/n) ')
        lower_resp = resp.lower()
        if lower_resp in ('y', 'n'):
            return lower_resp == 'y'

# 清空目录的函数
def clear_directory_(path):
    rmtree(str(path), ignore_errors = True)
    path.mkdir(exist_ok = True, parents = True)

# 安静删除文件的函数
def silentremove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

# 训练函数

# 计算带掩码的均值的函数
def masked_mean(t, mask, axis = None):
    return (t * mask).sum(axis = axis) / mask.sum(axis = axis)

# 交叉熵损失函数
def cross_entropy(logits, targets, axis = -1, ignore_index = 0):
    logprobs = nn.log_softmax(logits, axis = axis)

    nll = np.take_along_axis(logprobs, np.expand_dims(targets, axis = axis), axis = axis)
    nll = nll.squeeze(-1)

    # 为损失创建掩码，以便从第一个填充标记中学习
    # 填充标记被重用作字符串结束标记，以简化
    mask = (targets != ignore_index)
    eos_mask = (~mask).cumsum(axis = -1) == 1
    mask = mask | eos_mask

    ce = -masked_mean(nll, mask, axis = -1)
    return ce

# 获取损失函数
def get_loss_fn(model, data_parallel = False):
    def loss_fn(params, key, data):
        ids, labels = data[:-1], data[1:]
        logits = model.apply(params, key, ids)
        return cross_entropy(logits, labels, axis = -1)

    loss_fn = jit(vmap(loss_fn, in_axes = (None, None, 0), out_axes = 0))

    if data_parallel:
        loss_fn = pmap(loss_fn, in_axes = (None, None, 0), out_axes = 0)

    @value_and_grad
    def batched_loss_fn(params, key, data):
        if not data_parallel:
            values = loss_fn(params, key, data)
            return np.mean(values)

        mask = np.ones((data.shape[0],))

        device_count = jax.local_device_count()
        batch_size = data.shape[0]

        remainder = (batch_size % device_count)
        if remainder != 0:
            padding = device_count - remainder
            data = np.pad(data, ((0, padding), (0, 0)))
            mask = np.pad(mask, ((0, padding)))

        data, mask = map(lambda t: rearrange(t, '(p b) ... -> p b ...', p = device_count), (data, mask))
        values = loss_fn(params, key, data)
        return masked_mean(values, mask)

    return batched_loss_fn

# 采样函数

# 选择前 k 个值的函数
def select_top_k(tensor, k):
    values, _ = top_k(tensor, k)
    mask = tensor > values.min()
    return mask, np.where(mask, tensor, 0.)

# 生成 Gumbel 噪声的函数
def gumbel_noise(rng, shape):
    noise = random.uniform(rng, shape = shape, minval = 0., maxval = 1.)
    return -log(-log(noise))

# 采样函数
def sample(rng, fn, params, prime, length, top_k = None, add_bos = False):
    start_pos = prime.shape[-1]
    pad_right = length - prime.shape[-1]

    padding = (0, pad_right) if not add_bos else (1, pad_right - 1)
    seq = np.pad(prime, padding)

    one_hots = np.eye(length, dtype = int)

    for curr_pos in range(start_pos, length):
        logits = fn(params, next(rng), seq)
        logits = logits[curr_pos - 1]

        noise = gumbel_noise(next(rng), logits.shape)

        if exists(top_k):
            mask, logits = select_top_k(logits, top_k)
            noise *= mask

        logits += noise
        sampled_ind = np.argmax(logits, axis = -1)

        one_hot = one_hots[curr_pos]
        seq += one_hot * sampled_ind

    # 目前，将第二个填充标记（eos）后的所有内容设置为填充
    remove_after_eos_mask = (seq == 0).cumsum(axis = -1) > 1
    seq *= ~remove_after_eos_mask

    return seq

# RNG 修复

# 硬件均匀分布函数
def hardware_uniform(
    rng_key,
    shape,
    dtype = np.float32,
    minval = np.float32(0),
    maxval = np.float32(1)
):
    del rng_key
    # 将最小值转换为指定数据类型
    minval = lax.convert_element_type(minval, dtype)
    # 将最大值转换为指定数据类型
    maxval = lax.convert_element_type(maxval, dtype)
    # 返回一个形状为 shape 的在 [minval, maxval) 范围内均匀分布的随机数
    return lax.rng_uniform(minval, maxval, shape)
# 定义一个硬件实现的伯努利分布函数，接受随机数生成器密钥、概率和形状参数
def hardware_bernoulli(rng_key, p = np.float32(0.5), shape = None):
    # 删除随机数生成器密钥参数
    del rng_key
    # 返回一个布尔数组，表示是否小于给定概率 p
    return lax.rng_uniform(0.0, 1.0, shape) < p

# 设置 JAX 库中的随机数生成器函数为硬件实现的伯努利分布函数
def set_hardware_rng_(jax):
    # 将 JAX 库中的伯努利分布函数替换为硬件实现的伯努利分布函数
    jax.random.bernoulli = hardware_bernoulli
    # 将 JAX 库中的均匀分布函数替换为硬件实现的均匀分布函数
    jax.random.uniform = hardware_uniform
    # 将 JAX 库中的源码中的均匀分布函数替换为硬件实现的均匀分布函数
    jax._src.random.uniform = hardware_uniform
```