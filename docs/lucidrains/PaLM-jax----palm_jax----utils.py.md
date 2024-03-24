# `.\lucidrains\PaLM-jax\palm_jax\utils.py`

```
# 导入所需的库
from jax import random
from jax.lax import top_k
import jax.numpy as np

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 计算对数，加上一个很小的值以避免出现对数零的情况
def log(t, eps = 1e-20):
    return np.log(t + eps)

# 采样函数

# 选择概率最高的前 k 个元素
def select_top_k(tensor, k):
    values, _ = top_k(tensor, k)
    mask = tensor > values.min()
    return mask, np.where(mask, tensor, 0.)

# 生成 Gumbel 噪声
def gumbel_noise(key, shape):
    noise = random.uniform(key, shape = shape, minval = 0., maxval = 1.)
    return -log(-log(noise))

# 生成样本序列
def sample(key, model, prime, length, top_k = None):
    start_pos = prime.shape[-1]
    seq = np.pad(prime, (0, length - prime.shape[-1]))
    one_hots = np.eye(length, dtype = int)

    for curr_pos in range(start_pos, length):
        logits = model(seq)
        logits = logits[curr_pos - 1]

        _, key = random.split(key)
        noise = gumbel_noise(key, logits.shape)

        if exists(top_k):
            mask, logits = select_top_k(logits, top_k)
            noise *= mask

        logits += noise
        sampled_ind = np.argmax(logits, axis = -1)

        one_hot = one_hots[curr_pos]
        seq += one_hot * sampled_ind

    return seq
```