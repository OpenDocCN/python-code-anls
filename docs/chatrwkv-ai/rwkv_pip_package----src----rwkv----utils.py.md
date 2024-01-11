# `ChatRWKV\rwkv_pip_package\src\rwkv\utils.py`

```
# 导入所需的库
import os, sys
import numpy as np
import torch
from torch.nn import functional as F

# 定义管道参数类
class PIPELINE_ARGS():
    def __init__(self, temperature=1.0, top_p=0.85, top_k=0, alpha_frequency=0.2, alpha_presence=0.2, alpha_decay=0.996, token_ban=[], token_stop=[], chunk_len=256):
        self.temperature = temperature  # 温度参数
        self.top_p = top_p  # Top-p采样参数
        self.top_k = top_k  # Top-k采样参数
        self.alpha_frequency = alpha_frequency  # 频率惩罚（如GPT-3中所示）
        self.alpha_presence = alpha_presence  # 存在惩罚（如GPT-3中所示）
        self.alpha_decay = alpha_decay  # 逐渐衰减惩罚
        self.token_ban = token_ban  # 禁止生成某些标记
        self.token_stop = token_stop  # 在此处看到任何标记时停止生成
        self.chunk_len = chunk_len  # 将输入分割成块以节省VRAM（长度较短->速度较慢）

# 定义管道类
class PIPELINE():
    def __init__(self, model, WORD_NAME):
        self.model = model  # 模型
        if WORD_NAME == 'cl100k_base':
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(WORD_NAME)  # 获取编码器
        elif WORD_NAME == 'rwkv_vocab_v20230424':
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from rwkv_tokenizer import TRIE_TOKENIZER
            self.tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt')  # 使用特定的标记器
        else:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(WORD_NAME)  # 从文件中创建标记器
    # 优化文本内容，去除首尾空白字符并按行分割
    def refine_context(self, context):
        context = context.strip().split('\n')
        # 去除每行首尾空白字符和特殊字符
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        # 过滤空行并重新组合成字符串
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        # 如果内容为空，则设置为换行符
        if context == '':
            context = '\n'
        return context
    
    # 对输入进行编码处理
    def encode(self, x):
        # 如果使用的是 Tokenizer，则返回编码后的 ids
        if 'Tokenizer' in str(type(self.tokenizer)):
            return self.tokenizer.encode(x).ids
        # 否则返回普通的编码结果
        else:
            return self.tokenizer.encode(x)
    
    # 对输入进行解码处理
    def decode(self, x):
        return self.tokenizer.decode(x)
    # 从给定的logits中抽样，根据温度、top_p和top_k参数进行抽样
    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        # 对logits进行softmax处理，得到概率分布
        probs = F.softmax(logits.float(), dim=-1)
        # 将top_k参数转换为整数
        top_k = int(top_k)
        # 如果设备类型是CPU或者私有设备类型'privateuseone'
        if probs.device.type in ['cpu', 'privateuseone']:
            # 将概率分布转移到CPU上，并转换为numpy数组
            probs = probs.cpu().numpy()
            # 对概率进行排序，得到排序后的索引和概率值
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            # 计算累积概率
            cumulative_probs = np.cumsum(sorted_probs)
            # 找到概率累积超过top_p的位置，并得到对应的概率值作为截断点
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            # 将低于截断点的概率置为0
            probs[probs < cutoff] = 0
            # 如果top_k小于概率分布长度且大于0，则将较小的概率置为0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            # 如果温度不等于1.0，则对概率进行温度调节
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            # 对概率进行归一化
            probs = probs / np.sum(probs)
            # 根据概率分布进行抽样，返回抽样结果的索引
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            # 对概率进行排序，得到排序后的索引和概率值
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            # 对概率进行翻转，得到降序排列
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            # 计算累积概率
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            # 找到概率累积超过top_p的位置，并得到对应的概率值作为截断点
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            # 将低于截断点的概率置为0
            probs[probs < cutoff] = 0
            # 如果top_k小于概率分布长度且大于0，则将较小的概率置为0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            # 如果温度不等于1.0，则对概率进行温度调节
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            # 根据概率分布进行抽样，返回抽样结果的索引
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)
    # 生成文本内容
    def generate(self, ctx, token_count=100, args=PIPELINE_ARGS(), callback=None, state=None):
        # 存储所有生成的 token
        all_tokens = []
        # 上一次输出的位置
        out_last = 0
        # 输出的字符串
        out_str = ''
        # 记录 token 出现的次数
        occurrence = {}
        # 生成指定数量的 token
        for i in range(token_count):

            # 前向传播 & 调整概率
            tokens = self.encode(ctx) if i == 0 else [token]
            while len(tokens) > 0:
                out, state = self.model.forward(tokens[:args.chunk_len], state)
                tokens = tokens[args.chunk_len:]
                
            # 根据禁止的 token，将其概率设为负无穷
            for n in args.token_ban:
                out[n] = -float('inf')
            # 根据 token 出现的次数，调整其概率
            for n in occurrence:
                out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
            
            # 采样器
            token = self.sample_logits(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            # 如果 token 在停止列表中，则结束生成
            if token in args.token_stop:
                break
            # 将 token 添加到所有 token 中
            all_tokens += [token]
            # 对所有 token 出现的次数进行衰减
            for xxx in occurrence:
                occurrence[xxx] *= args.alpha_decay
            # 如果 token 不在出现次数记录中，则添加到记录中，否则出现次数加一
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1
            # 输出
            tmp = self.decode(all_tokens[out_last:])
            # 如果不包含非法字符，则输出
            if '\ufffd' not in tmp: # is valid utf-8 string?
                if callback:
                    callback(tmp)
                out_str += tmp
                out_last = i + 1
        # 返回生成的字符串
        return out_str
```