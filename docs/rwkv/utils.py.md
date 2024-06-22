# `.\rwkv\utils.py`

```py
# 导入所需的库
import os, sys
import numpy as np
import torch
from torch.nn import functional as F

# 定义管道参数类
class PIPELINE_ARGS():
    def __init__(self, temperature=1.0, top_p=0.85, top_k=0, alpha_frequency=0.2, alpha_presence=0.2, alpha_decay=0.996, token_ban=[], token_stop=[], chunk_len=256):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alpha_frequency = alpha_frequency # 频率惩罚（如 GPT-3 中所示）
        self.alpha_presence = alpha_presence # 存在惩罚（如 GPT-3 中所示）
        self.alpha_decay = alpha_decay # 逐渐减少惩罚
        self.token_ban = token_ban # 禁止生成某些标记
        self.token_stop = token_stop # 在此处看到任何标记时停止生成
        self.chunk_len = chunk_len # 将输入分成块以节省 VRAM（较短 -> 较慢）

# 定义管道类
class PIPELINE():
    def __init__(self, model, WORD_NAME):
        self.model = model
        # 根据不同的 WORD_NAME 加载不同的分词器
        if WORD_NAME == 'cl100k_base':
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(WORD_NAME)
        elif WORD_NAME == 'rwkv_vocab_v20230424':
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from rwkv_tokenizer import TRIE_TOKENIZER
            self.tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt')        
        else:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(WORD_NAME)

    # 优化上下文格式
    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    # 编码函数
    def encode(self, x):
        if 'Tokenizer' in str(type(self.tokenizer)):
            return self.tokenizer.encode(x).ids
        else:
            return self.tokenizer.encode(x)
    
    # 解码函数
    def decode(self, x):
        return self.tokenizer.decode(x)
    # 从给定的logits中采样输出一个值
    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        # 如果温度为0，则将温度设为1.0，同时将top_p设为0
        if temperature == 0:
            temperature = 1.0
            top_p = 0
        # 对logits进行softmax操作，得到概率分布
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        # 'privateuseone'是自定义设备类型，例如`torch_directml.device()`
        if probs.device.type in ['cpu', 'privateuseone']:
            # 将概率分布转移到CPU上，并转换为numpy数组
            probs = probs.cpu().numpy()
            # 对概率进行排序
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            # 找到概率累积大于等于top_p的位置，得到截断值
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            # 将低于截断值的概率设为0
            probs[probs < cutoff] = 0
            # 如果top_k小于概率分布长度且大于0，则将低于top_k的概率设为0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            # 如果温度不为1.0，则对概率进行温度调节
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            # 归一化概率分布
            probs = probs / np.sum(probs)
            # 从概率分布中随机选择一个值作为输出
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            # 对概率进行排序
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            # 从概率分布中多项式采样一个值作为输出
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)
    # 生成文本内容，根据给定的上下文和参数
    def generate(self, ctx, token_count=100, args=PIPELINE_ARGS(), callback=None, state=None):
        # 存储所有生成的 token
        all_tokens = []
        # 上一个输出的位置
        out_last = 0
        # 输出的字符串
        out_str = ''
        # 记录 token 出现次数的字典
        occurrence = {}
        # 循环生成指定数量的 token
        for i in range(token_count):

            # forward & adjust prob.
            # 如果是第一个 token，则根据上下文编码，否则使用前一个 token
            tokens = self.encode(ctx) if i == 0 else [token]
            # 循环进行模型前向传播，调整概率
            while len(tokens) > 0:
                out, state = self.model.forward(tokens[:args.chunk_len], state)
                tokens = tokens[args.chunk_len:]
                
            # 根据禁止的 token，将其概率设为负无穷
            for n in args.token_ban:
                out[n] = -float('inf')
            # 根据 token 出现次数和频率调整概率
            for n in occurrence:
                out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
            
            # 采样得到下一个 token
            token = self.sample_logits(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            # 如果生成的 token 在停止 token 中，则结束生成
            if token in args.token_stop:
                break
            # 将生成的 token 添加到列表中
            all_tokens += [token]
            # 根据 token 出现次数进行衰减
            for xxx in occurrence:
                occurrence[xxx] *= args.alpha_decay
            
            # 根据生成的 token 判断权重
            ttt = self.decode([token])
            www = 1
            if ttt in ' \t0123456789':
                www = 0
            # 如果 token 不在出现次数字典中，则添加，否则增加权重
            if token not in occurrence:
                occurrence[token] = www
            else:
                occurrence[token] += www
            # 打印出现次数字典，用于调试
            # print(occurrence) # debug
            
            # 输出生成的文本
            tmp = self.decode(all_tokens[out_last:])
            # 如果生成的文本是有效的 utf-8 字符串
            if '\ufffd' not in tmp: # is valid utf-8 string?
                # 如果有回调函数，则调用回调函数
                if callback:
                    callback(tmp)
                # 将生成的文本添加到输出字符串中
                out_str += tmp
                out_last = i + 1
        # 返回生成的文本字符串
        return out_str
```