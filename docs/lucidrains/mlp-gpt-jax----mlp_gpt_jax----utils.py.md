# `.\lucidrains\mlp-gpt-jax\mlp_gpt_jax\utils.py`

```
# 导入所需的库
from jax import random, nn, value_and_grad, vmap, jit
from jax.lax import top_k
import jax.numpy as np

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 计算对数，加上一个很小的值以避免出现对数零的情况
def log(t, eps = 1e-20):
    return np.log(t + eps)

# 训练函数

# 计算交叉熵损失
def cross_entropy(logits, targets, axis = -1):
    logprobs = nn.log_softmax(logits, axis = axis)
    nll = np.take_along_axis(logprobs, np.expand_dims(targets, axis = axis), axis = axis)
    ce = -np.mean(nll)
    return ce

# 获取训练损失函数
def get_train_loss_fn(model):
    batch_model_apply = jit(vmap(model.apply, in_axes = (None, None, 0), out_axes = 0))

    @value_and_grad
    def loss_fn(params, key, data):
        inp, labels = data[:, :-1], data[:, 1:]
        logits = batch_model_apply(params, key, inp)
        return cross_entropy(logits, labels, axis = -1)

    return loss_fn

# 采样函数

# 选择前 k 个最大值
def select_top_k(tensor, k):
    values, _ = top_k(tensor, k)
    mask = tensor > values.min()
    return mask, np.where(mask, tensor, 0.)

# 生成 Gumbel 噪声
def gumbel_noise(rng, shape):
    noise = random.uniform(rng, shape = shape, minval = 0., maxval = 1.)
    return -log(-log(noise))

# 生成样本序列
def sample(rng, fn, params, prime, length, top_k = None):
    start_pos = prime.shape[-1]
    seq = np.pad(prime, (0, length - prime.shape[-1]))
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

    return seq
```