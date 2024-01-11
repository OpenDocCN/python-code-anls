# `ChatRWKV\RWKV_in_150_lines.py`

```
# 导入所需的库
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
from torch.nn import functional as F
from tokenizers import Tokenizer

# 从文件中加载 tokenizer
tokenizer = Tokenizer.from_file("20B_tokenizer.json")

# 创建一个简单的命名空间对象，用于存储模型相关的参数
args = types.SimpleNamespace()
args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066'
args.n_layer = 24
args.n_embd = 1024

# 设置生成文本的上下文
context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
NUM_TRIALS = 3  # 设置生成文本的尝试次数
LENGTH_PER_TRIAL = 100  # 设置每次尝试生成文本的长度
TEMPERATURE = 1.0  # 设置生成文本的温度
TOP_P = 0.85  # 设置生成文本的 top-p 参数
    # 初始化函数，接受参数 args
    def __init__(self, args):
        # 调用父类的初始化函数
        super().__init__()
        # 将参数 args 存储在对象的属性中
        self.args = args
        # 设置 torch 为推断模式
        self.eval() # set torch to inference mode
        
        # 从文件中加载模型参数，使用 CPU 进行映射
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        # 遍历模型参数的键
        for k in w.keys():
            # 如果键中包含 '.time_'，则将对应的值压缩
            if '.time_' in k: w[k] = w[k].squeeze()
            # 如果键中包含 '.time_decay'，则将对应的值进行指数运算
            if '.time_decay' in k: w[k] = -torch.exp(w[k].float()) # the real time decay is like e^{-e^x}
            # 否则，将对应的值转换为 float32 类型
            else: w[k] = w[k].float() # convert to f32 type
        
        # 设置对象的属性 self.w 为 w
        self.w = types.SimpleNamespace() # set self.w from w
        # 初始化 self.w.blocks 为空字典
        self.w.blocks = {}
        # 遍历 w 的键
        for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            # 将键按 '.' 分割成部分
            parts = k.split('.')
            # 弹出最后一个部分作为键
            last = parts.pop()
            # 初始化变量 here 为 self.w
            here = self.w
            # 遍历部分
            for p in parts:
                # 如果部分是数字，则转换为整数
                if p.isdigit():
                    p = int(p)
                    # 如果 here 中不包含 p，则添加 p 为 SimpleNamespace 类型
                    if p not in here: here[p] = types.SimpleNamespace()
                    # 更新 here 为 here[p]
                    here = here[p]
                else:
                    # 如果 here 中没有属性 p，则添加 p 为 SimpleNamespace 类型
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    # 更新 here 为 getattr(here, p)
                    here = getattr(here, p)
            # 设置 here[last] 为 w[k]
            setattr(here, last, w[k])

    # 定义 layer_norm 函数，接受参数 x 和 w
    def layer_norm(self, x, w):
        # 使用 F.layer_norm 对 x 进行层归一化，指定维度和权重、偏置
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # 定义 channel_mixing 函数，接受参数 x, state, i, time_mix_k, time_mix_r, kw, vw, rw
    @torch.jit.script_method
    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        # 计算 xk 和 xr
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        # 更新 state[5*i+0] 为 x
        state[5*i+0] = x
        # 计算 r 和 k
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        # 返回 r * (vw @ k)
        return r * (vw @ k)

    # 定义 channel_mixing 函数
    @torch.jit.script_method
    # 对输入数据进行时间混合处理
    def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        # 计算加权后的输入数据
        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
        # 更新状态中的数据
        state[5*i+1] = x
        # 计算激活函数
        r = torch.sigmoid(rw @ xr)
        # 计算权重
        k = kw @ xk
        v = vw @ xv
        # 获取状态中的参数
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        # 计算时间加权值
        ww = time_first + k
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b
        # 更新状态中的参数
        ww = pp + time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = qq
        # 返回加权后的结果
        return ow @ (r * wkv)
    # 定义一个前向传播函数，接受 token 和 state 作为输入参数
    def forward(self, token, state):
        # 使用 torch.no_grad() 上下文管理器，确保前向传播过程中不会进行梯度计算
        with torch.no_grad():
            # 如果 state 为 None，则初始化为全零张量
            if state == None:
                state = torch.zeros(self.args.n_layer * 5, self.args.n_embd)
                # 对 state 的特定位置赋值为负无穷，用于后续计算
                for i in range(self.args.n_layer): state[5*i+4] = -1e30 # -infinity
            
            # 从词嵌入矩阵中获取 token 对应的词向量
            x = self.w.emb.weight[token]
            # 对词向量进行 Layer Normalization
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            # 遍历每个 Transformer 层
            for i in range(self.args.n_layer):
                # 获取当前层的注意力机制
                att = self.w.blocks[i].att
                # 对词向量进行时间混合和通道混合操作
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i, 
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                    att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)
                # 获取当前层的前馈神经网络
                ffn = self.w.blocks[i].ffn
                # 对词向量进行时间混合和通道混合操作
                x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                    ffn.time_mix_k, ffn.time_mix_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
            
            # 将处理后的词向量与输出权重相乘，得到最终的输出结果
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            # 将输出结果转换为浮点数类型，并返回结果和更新后的 state
            return x.float(), state
# 定义一个函数，用于根据模型输出的 logits 进行采样
def sample_logits(out, temperature=1.0, top_p=0.8):
    # 对输出进行 softmax 处理，并转换为 numpy 数组
    probs = F.softmax(out, dim=-1).numpy()
    # 对概率进行排序
    sorted_probs = np.sort(probs)[::-1]
    # 计算累积概率
    cumulative_probs = np.cumsum(sorted_probs)
    # 根据 top_p 获取截断值
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    # 将低于截断值的概率置为 0
    probs[probs < cutoff] = 0
    # 如果温度不为 1.0，则进行温度调节
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    # 对概率进行归一化
    probs = probs / np.sum(probs)
    # 根据概率进行采样，得到输出
    out = np.random.choice(a=len(probs), p=probs)
    return out

# 打印提示信息，加载模型
print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
model = RWKV_RNN(args)

# 打印提示信息，预处理上下文
print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
init_state = None
# 遍历上下文中的每个 token，并进行模型前向传播
for token in tokenizer.encode(context).ids:
    init_out, init_state = model.forward(token, init_state)

# 循环进行多次试验
for TRIAL in range(NUM_TRIALS):
    # 打印试验提示信息和上下文
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), init_state.clone()
    # 对每个试验进行多次迭代
    for i in range(LENGTH_PER_TRIAL):
        # 根据模型输出进行采样
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        tmp = tokenizer.decode(all_tokens[out_last:])
        # 如果采样得到的字符串是有效的 utf-8 字符串，则打印
        if '\ufffd' not in tmp:
            print(tmp, end="", flush=True)
            out_last = i + 1
        out, state = model.forward(token, state)       
print('\n')
```