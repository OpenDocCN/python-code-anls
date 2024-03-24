# `.\lucidrains\mlp-gpt-jax\train.py`

```
# 从 random 模块中导入 randrange 函数
# 从 tqdm 模块中导入 tqdm 函数
# 从 gzip 模块中导入 gzip 模块
# 从 numpy 模块中导入 np 别名
from random import randrange
import tqdm
import gzip
import numpy as np

# 从 torch.utils.data 模块中导入 DataLoader, Dataset 类
# 从 jax 模块中导入 nn, random, jit 模块
# 从 optax 模块中导入 adam, clip_by_global_norm, chain, apply_updates, apply_every 模块
# 从 haiku 模块中导入 PRNGSequence 类
# 从 mlp_gpt_jax 模块中导入 TransformedMLPGpt 类
# 从 mlp_gpt_jax.utils 模块中导入 sample, get_train_loss_fn 函数
from torch.utils.data import DataLoader, Dataset
import jax
from jax import nn, random, jit
from optax import adam, clip_by_global_norm, chain, apply_updates, apply_every
from haiku import PRNGSequence
from mlp_gpt_jax import TransformedMLPGpt
from mlp_gpt_jax.utils import sample, get_train_loss_fn

# 常量定义
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 0.5
VALIDATE_EVERY  = 100
SAMPLE_EVERY  = 500
SEQ_LEN = 768

# 辅助函数定义

# 生成 DataLoader 的循环迭代器
def cycle(loader):
    while True:
        for data in loader:
            yield data

# 解码单个 token
def decode_token(token):
    return str(chr(max(32, token)))

# 解码一组 tokens
def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# 准备 enwik8 数据

# 使用 gzip 模块打开 enwik8.gz 文件
with gzip.open('./data/enwik8.gz') as file:
    # 从文件中读取前 95e6 个字节，转换为 numpy 数组 X
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    # 将数据 X 分割成训练集和验证集
    data_train, data_val = np.split(X, [int(90e6)])

# 定义 TextSamplerDataset 类，继承自 Dataset 类
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        # 随机生成起始位置，返回该位置开始的 seq_len + 1 长度的数据
        rand_start = randrange(0, self.data.shape[0] - self.seq_len - 1)
        return self.data[rand_start: rand_start + self.seq_len + 1]

    def __len__(self):
        # 返回数据长度除以 seq_len
        return self.data.shape[0] // self.seq_len

# 创建训练集和验证集的 DataLoader
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# 设置模型和参数

model_kwargs = dict(
    num_tokens = 256,
    dim = 512,
    seq_len = SEQ_LEN,
    depth = 8,
    attn_dim = 32,
)

# 初始化训练模型和评估模型
train_model = TransformedMLPGpt(**model_kwargs, layer_survival_prob = 0.95)
eval_model = TransformedMLPGpt(**model_kwargs)

# 创建 PRNGSequence 对象 rng
rng = PRNGSequence(42)
# 初始化模型参数 params
params = train_model.init(next(rng), train_dataset[0][:-1])

# 获取训练损失函数
loss_fn = get_train_loss_fn(train_model)

# 优化器

# 定义优化器链
optim = chain(
    clip_by_global_norm(MAX_GRAD_NORM),
    adam(LEARNING_RATE),
    apply_every(GRADIENT_ACCUMULATE_EVERY)
)

# 初始化优化器状态
optim_state = optim.init(params)

# 训练

# 循环训练 NUM_BATCHES 次
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    # 获取下一个训练数据
    data = next(train_loader).numpy()
    # 计算损失和梯度
    loss, grads = loss_fn(params, next(rng), data)
    # 更新参数
    updates, optim_state = optim.update(grads, optim_state, params)
    params = apply_updates(params, updates)

    # 每隔 GRADIENT_ACCUMULATE_EVERY 次输出损失
    if i % GRADIENT_ACCUMULATE_EVERY == 0:
        print(f'loss: {loss.item()}')

    # 每隔 SAMPLE_EVERY 次生成样本
    if i % SAMPLE_EVERY == 0:
        # 获取下一个验证数据
        valid_data = next(val_loader).numpy()
        prime = valid_data[0][:100]
        prime_str = decode_tokens(prime)
        print(prime_str, "\n", "*" * 40)

        # 生成样本并解码
        sampled = sample(rng, jit(eval_model.apply), params, prime, SEQ_LEN, top_k = 25)
        sampled_str = decode_tokens(sampled[100:])
        print(sampled_str)
```