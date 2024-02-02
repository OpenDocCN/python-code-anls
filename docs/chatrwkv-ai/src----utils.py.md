# `ChatRWKV\src\utils.py`

```py
# 导入所需的库和模块
import json, time, random, os
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

# 初始化时间槽字典和时间参考值
time_slot = {}
time_ref = time.time_ns()

# 记录时间的函数
def record_time(name):
    # 如果名称不在时间槽字典中，则初始化为一个较大的值
    if name not in time_slot:
        time_slot[name] = 1e20
    # 计算时间差并更新时间槽字典中的值
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

# TOKENIZER 类
class TOKENIZER():
    # 初始化函数，从文件中加载词汇表
    def __init__(self, WORD_NAME):
        self.tokenizer = Tokenizer.from_file(WORD_NAME)

    # 优化上下文的函数，去除空格和换行符
    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    # 编码函数，将文本转换为词汇表中的 id 列表
    def encode(self, x):
        return self.tokenizer.encode(x).ids
    
    # 解码函数，将 id 列表转换为文本
    def decode(self, x):
        return self.tokenizer.decode(x)
    # 从logits中采样，返回采样结果的索引
    def sample_logits(self, logits, x, ctx_len, temperature=1.0, top_p=1.0):
        # 对logits进行softmax操作，得到概率分布
        probs = F.softmax(logits.float(), dim=-1)

        # 如果运行设备是CPU
        if os.environ["RWKV_RUN_DEVICE"] == "cpu":
            # 将概率分布转换为numpy数组
            probs = probs.numpy()
            # 对概率进行降序排序
            sorted_probs = np.sort(probs)[::-1]
            # 计算累积概率
            cumulative_probs = np.cumsum(sorted_probs)
            # 找到累积概率大于等于top_p的位置
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            # 将低于cutoff的概率置为0
            probs[probs < cutoff] = 0
            # 如果温度不为1.0，进行温度调节
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            # 归一化概率分布
            probs = probs / np.sum(probs)
            # 从概率分布中进行采样
            out = np.random.choice(a=len(probs), p=probs)
            # 返回采样结果的索引
            return int(out)
        else:
            # 对概率进行降序排序
            sorted_probs = torch.sort(probs, descending=True)[0]
            # 计算累积概率
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            # 找到累积概率大于等于top_p的位置
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            # 将低于cutoff的概率置为0
            probs[probs < cutoff] = 0
            # 如果温度不为1.0，进行温度调节
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            # 从概率分布中进行多项式采样
            out = torch.multinomial(probs, num_samples=1)[0]
            # 返回采样结果的索引
            return int(out)
```