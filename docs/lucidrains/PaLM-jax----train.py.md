# `.\lucidrains\PaLM-jax\train.py`

```py
# 导入必要的库
import os
from random import randrange
from functools import partial
import tqdm
import gzip
import numpy as np

import jax
import jax.numpy as jnp
from jax import nn

# 导入自定义库
import equinox as eqx
from optax import adam, clip_by_global_norm, chain, apply_every

# 导入自定义模块
from palm_jax.palm_lite import PaLM
from palm_jax.utils import sample

# 设置环境变量
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# 定义常量
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 0.5
VALIDATE_EVERY  = 100
SAMPLE_EVERY  = 500
SEQ_LEN = 1024

# 定义循环生成器函数
def cycle(loader):
    while True:
        for data in loader:
            yield data

# 解码单个 token 函数
def decode_token(token):
    return str(chr(max(32, token)))

# 解码一组 tokens 函数
def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# 读取 enwik8 数据集
with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    data_train, data_val = np.split(X, [int(90e6)])

# 从数据集中采样序列函数
def sample_seq_from_data(data, *, seq_len, batch_size):
    total_seq_len = data.shape[0]
    base_arange = np.arange(seq_len)
    start_indices = np.random.randint(0, total_seq_len - seq_len, (batch_size,))
    token_indices = start_indices[:, None] + base_arange
    return data[token_indices]

# 部分应用采样序列函数
sample_seq_fn = partial(sample_seq_from_data, seq_len = SEQ_LEN, batch_size = BATCH_SIZE)

# 初始化 PRNGKey
key = jax.random.PRNGKey(0)

# 初始化 PaLM 模型
model = PaLM(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    heads = 8,
    dim_head = 64,
    key = key
)

# 交叉熵损失函数
def cross_entropy(logits, targets, axis = -1):
    logprobs = nn.log_softmax(logits, axis = axis)
    nll = jnp.take_along_axis(logprobs, jnp.expand_dims(targets, axis = axis), axis = axis)
    cross_entropy = -jnp.mean(nll)
    return cross_entropy

# 定义损失函数
@eqx.filter_value_and_grad
def loss_fn(model, data):
    inp, labels = data[:, :-1], data[:, 1:]
    logits = model(inp)
    return cross_entropy(logits, labels, axis = -1)

# 初始化优化器
optim = chain(
    clip_by_global_norm(MAX_GRAD_NORM),
    adam(LEARNING_RATE),
    apply_every(GRADIENT_ACCUMULATE_EVERY)
)

optim_state = optim.init(model)

# 训练步骤
@eqx.filter_jit(kwargs=dict(data=True))
def train_step(model, data, optim_state):
    loss, grads = loss_fn(model, data)
    updates, optim_state = optim.update(grads, optim_state)
    model = eqx.apply_updates(model, updates)
    return model, optim_state, loss

# 训练过程
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        data = sample_seq_fn(data_train)
        model, optim_state, loss = train_step(model, data, optim_state)

    print(f'loss: {loss.item()}')

    if i % SAMPLE_EVERY == 0:
        valid_data = sample_seq_fn(data_val)
        prime = valid_data[0][:100]
        prime_str = decode_tokens(prime)
        print(prime_str, "\n", "*" * 40)

        sampled = sample(key, model, prime, SEQ_LEN, top_k = 25)
        sampled_str = decode_tokens(sampled[100:])
        print(sampled_str)
```