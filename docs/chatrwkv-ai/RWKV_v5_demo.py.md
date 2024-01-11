# `ChatRWKV\RWKV_v5_demo.py`

```
# 导入所需的库
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
import torch.nn as nn
from torch.nn import functional as F

# 定义自定义模块和函数
MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method

# 定义 RWKV_TOKENIZER 类
class RWKV_TOKENIZER():
    # 定义类属性
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    
    # 初始化方法
    def __init__(self, file_name):
        # 初始化实例属性
        self.idx2token = {}
        sorted = [] # must be already sorted
        # 读取文件内容
        lines = open(file_name, "r", encoding="utf-8").readlines()
        # 遍历文件内容
        for l in lines:
            # 获取索引和对应的值
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            # 将字符串编码为 utf-8 格式的字节流
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        # 构建 token 到索引的映射
        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # 预先计算一些用于快速匹配的表
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        # 遍历已排序的 token
        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                # 将 token 添加到对应的表中
                self.table[s0][s1] += [s]
                # 更新最大 token 长度
                self.wlen[s0] = max(self.wlen[s0], len(s))
                # 将第二个字节添加到 good 集合中
                self.good[s0].add(s1)
    # 将字节流编码成整数列表
    def encodeBytes(self, src: bytes) -> list[int]:
        # 获取字节流的长度
        src_len: int = len(src)
        # 初始化整数列表
        tokens: list[int] = []
        # 初始化索引
        i: int = 0
        # 遍历字节流
        while i < src_len:
            # 获取当前字节
            s: bytes = src[i : i + 1]

            # 如果不是最后一个字节
            if i < src_len - 1:
                # 获取下一个字节和当前字节的整数值
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                # 如果下一个字节在当前字节对应的好前缀列表中
                if s1 in self.good[s0]:
                    # 获取匹配的子串
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        # 尝试从编码表中找到匹配的编码
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            # 将编码后的整数添加到列表中
            tokens.append(self.token2idx[s])
            # 更新索引
            i += len(s)

        # 返回整数列表
        return tokens

    # 将整数列表解码成字节流
    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    # 对字符串进行编码
    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    # 对整数列表进行解码
    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    # 打印编码后的整数列表
    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                # 尝试将字节流解码成字符串
                s = s.decode('utf-8')
            except:
                pass
            # 打印编码后的整数及其对应的字符
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()
# 定义一个函数，用于根据给定的输出计算采样的logits
def sample_logits(out, temperature=1.0, top_p=0.8):
    # 使用softmax函数对输出进行处理，并转换为numpy数组
    probs = F.softmax(out, dim=-1).numpy()
    # 对概率进行排序
    sorted_probs = np.sort(probs)[::-1]
    # 计算累积概率
    cumulative_probs = np.cumsum(sorted_probs)
    # 根据top_p找到截断点
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    # 将低于截断点的概率置为0
    probs[probs < cutoff] = 0
    # 如果温度不为1.0，则对概率进行调整
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    # 对概率进行归一化
    probs = probs / np.sum(probs)
    # 根据概率进行随机采样
    out = np.random.choice(a=len(probs), p=probs)
    return out

# 创建一个RWKV_TOKENIZER对象，指定词汇表文件路径
tokenizer = RWKV_TOKENIZER("/fsx/BlinkDL/CODE/_PUBLIC_/ChatRWKV/tokenizer/rwkv_vocab_v20230424.txt")

# 定义一个简单的命名空间对象，用于存储模型相关参数
args = types.SimpleNamespace()
args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/temp/RWKV-5-World-0B4-v2-OnlyForTest_71%_trained-20231104-ctx4096'
args.n_layer = 24
args.n_embd = 1024
args.vocab_size = 65536

# 定义对话的上下文
context = "\nElon Musk has"
# context = "\n我们发现"
# 定义进行采样的次数
NUM_TRIALS = 3
# 每次采样的长度
LENGTH_PER_TRIAL = 100
# 采样时的温度
TEMPERATURE = 1.0
# 采样时的top_p值
TOP_P = 0.7

# 创建一个RWKV_RNN类的实例，继承自MyModule类
class RWKV_RNN(MyModule):
    # 初始化函数，接受参数 args
    def __init__(self, args):
        # 调用父类的初始化函数
        super().__init__()
        # 将参数 args 存储在对象的属性中
        self.args = args
        # 设置 torch 为推断模式
        self.eval()

        # 从文件中加载模型参数，使用 CPU 运行
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        # 将所有参数转换为 float32 类型
        for k in w.keys():
            w[k] = w[k].float()
            # 如果参数名中包含 '.time_'，则将其压缩为一维
            if '.time_' in k: w[k] = w[k].squeeze()
            # 如果参数名中包含 '.time_decay'，则将其进行指数运算，并在最后添加一维
            if '.time_decay' in k: w[k] = torch.exp(-torch.exp(w[k])).unsqueeze(-1)
            # 如果参数名中包含 '.time_faaaa'，则在最后添加一维
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)

        # 设置对象的属性 n_head 为 w 中指定参数的形状的第一个维度大小
        self.n_head = w['blocks.0.att.time_decay'].shape[0]
        # 设置对象的属性 head_size 为 w 中指定参数的形状的第一个维度大小除以 n_head 的结果
        self.head_size = w['blocks.0.ln1.weight'].shape[0] // self.n_head

        # 创建一个简单的命名空间对象，并将其赋值给对象的属性 w
        self.w = types.SimpleNamespace()
        # 设置对象的属性 blocks 为一个空字典
        self.w.blocks = {}
        # 遍历 w 中的所有参数
        for k in w.keys():
            # 将参数名按 '.' 分割成多个部分
            parts = k.split('.')
            # 取出最后一个部分
            last = parts.pop()
            # 初始化一个变量 here 为 self.w
            here = self.w
            # 遍历除了最后一个部分以外的所有部分
            for p in parts:
                # 如果 p 是数字，则将其转换为整数，并检查是否在 here 中，如果不在则添加到 here 中
                if p.isdigit():
                    p = int(p)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                # 如果 p 不是数字，则检查 here 中是否有属性 p，如果没有则添加一个命名空间属性
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            # 将 w 中的参数值赋值给相应的属性
            setattr(here, last, w[k])

    # 定义一个层归一化函数，接受输入 x 和权重 w
    def layer_norm(self, x, w):
        # 使用 F.layer_norm 对输入 x 进行层归一化，指定归一化的维度和权重和偏置
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # 定义一个装饰器为 MyFunction 的通道混合函数，接受输入 x、状态 state、索引 i、时间混合参数和权重参数
    @MyFunction
    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        # 计算索引 i0
        i0 = (2+self.head_size)*i+0
        # 计算 xk 和 xr
        xk = x * time_mix_k + state[i0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[i0] * (1 - time_mix_r)
        # 更新状态中的值
        state[i0] = x
        # 计算 r 和 k
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))  # 使用平方激活函数，参考论文
        # 返回 r 乘以 vw 和 k 的结果
        return r * (vw @ k)

    # 定义一个装饰器为 MyFunction 的函数
    @MyFunction
    # 对输入进行时间混合操作，更新状态和计算输出
    def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_mix_g, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b):
        # 获取头数和头大小
        H = self.n_head
        S = self.head_size

        # 计算状态索引
        i1 = (2+S)*i+1
        # 计算加权输入
        xk = x * time_mix_k + state[i1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[i1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[i1] * (1 - time_mix_r)
        xg = x * time_mix_g + state[i1] * (1 - time_mix_g)
        # 更新状态
        state[i1] = x

        # 计算注意力权重
        r = (rw @ xr).view(H, 1, S)
        k = (kw @ xk).view(H, S, 1)
        v = (vw @ xv).view(H, 1, S)
        g = F.silu(gw @ xg)

        # 获取上一步的状态
        s = state[(2+S)*i+2:(2+S)*(i+1), :].reshape(H, S, S)

        # 初始化输出和注意力权重
        x = torch.zeros(H, S)
        a = k @ v
        x = r @ (time_first * a + s)
        s = a + time_decay * s
    
        # 更新状态和输出
        state[(2+S)*i+2:(2+S)*(i+1), :] = s.reshape(S, -1)
        x = x.flatten()

        # 进行分组归一化和权重计算
        x = F.group_norm(x.unsqueeze(0), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).squeeze(0) * g # same as gn(x/8, eps=1e-5)
        return ow @ x
    # 定义一个前向传播函数，接受 token 和 state 作为输入参数
    def forward(self, token, state):
        # 使用 torch.no_grad() 上下文管理器，确保在前向传播过程中不会进行梯度计算
        with torch.no_grad():
            # 如果 state 为 None，则初始化为全零张量
            if state == None:
                state = torch.zeros(self.args.n_layer * (2+self.head_size), self.args.n_embd)
            
            # 从词嵌入矩阵中获取 token 对应的词向量
            x = self.w.emb.weight[token]
            # 对词向量进行 Layer Normalization
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            # 遍历每个 Transformer 层
            for i in range(self.args.n_layer):
                # 获取当前层的注意力机制
                att = self.w.blocks[i].att
                # 使用时间混合函数对词向量进行时间混合
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i, 
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_mix_g, att.time_faaaa, att.time_decay, 
                    att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight,
                    att.ln_x.weight, att.ln_x.bias)
                # 获取当前层的前馈神经网络
                ffn = self.w.blocks[i].ffn
                # 使用通道混合函数对词向量进行通道混合
                x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                    ffn.time_mix_k, ffn.time_mix_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
            
            # 将词向量与输出权重相乘，得到最终输出
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            # 将输出转换为浮点数类型，并返回输出和更新后的 state
            return x.float(), state
# 打印提示信息，使用 CPU 加载模型的名称
print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
# 创建 RWKV_RNN 模型对象
model = RWKV_RNN(args)

# 打印提示信息，对上下文进行预处理（慢速版本，查看 v2/rwkv/model.py 获取快速版本）
init_state = None
# 遍历上下文中的每个标记，获取初始输出和状态
for token in tokenizer.encode(context):
    init_out, init_state = model.forward(token, init_state)

# 对于每次试验，循环执行以下操作
for TRIAL in range(NUM_TRIALS):
    # 打印试验信息和上下文
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), init_state.clone()
    # 对于每个试验长度，执行以下操作
    for i in range(LENGTH_PER_TRIAL):
        # 从输出概率中采样一个标记
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        try:
            # 尝试解码标记序列，如果是有效的 utf-8 字符串则打印
            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # 只有在有有效的 utf-8 字符串时才打印
                print(tmp, end="", flush=True)
                out_last = i + 1
        except:
            pass
        # 获取下一个输出和状态
        out, state = model.forward(token, state)       
print('\n')
```