# `ChatRWKV\src\model_run.py`

```py
########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# 导入所需的库
import types, math, os, gc
import torch
from torch.nn import functional as F
# 设置一些优化参数
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# 定义自定义模块和函数
MyModule = torch.nn.Module
def __nop(ob):
    return ob
MyFunction = __nop

# 如果环境变量 RWKV_JIT_ON 大于 0，则使用 torch.jit 提供的脚本模块和脚本方法
if int(os.environ["RWKV_JIT_ON"]) > 0:
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

# 打印环境变量 RWKV_JIT_ON 的值
print(f'\nRWKV_JIT_ON {os.environ["RWKV_JIT_ON"]}\n')

# 设置全局变量 RWKV_RESCALE_LAYER 的值为 6
RWKV_RESCALE_LAYER = 6 # set x = x/2 every X layer (to avoid FP16 overflow)

############################################################################################################

# 定义 RWKV_RNN 类，继承自 MyModule
class RWKV_RNN(MyModule):
    # 定义 LN 方法，用于对输入进行 Layer Normalization
    def LN(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # 定义 FF_one 方法，用于执行单个前馈神经网络层的计算
    @MyFunction
    def FF_one(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        # 计算加权平均值
        xx = state[5*i+0].to(dtype=self.FLOAT_MODE)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+0] = x.float()

        # 计算激活函数
        r = torch.sigmoid(xr @ rw)
        k = torch.square(torch.relu(xk @ kw))
        kv = k @ vw
        return r * kv

    # 定义 FF_seq 方法，用于执行序列前馈神经网络层的计算
    @MyFunction
    def FF_seq(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        # 计算加权平均值
        xx = torch.cat((state[5*i+0].to(dtype=self.FLOAT_MODE).unsqueeze(0), x[:-1,:]))
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+0] = x[-1,:].float()

        # 计算激活函数
        r = torch.sigmoid(xr @ rw)
        k = torch.square(torch.relu(xk @ kw))
        kv = k @ vw
        return r * kv

    @MyFunction
    # 定义一个方法，接受输入 x, state, i, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow
    def SA_one(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        # 从 state 中获取第 i 个元素的值，并转换为指定的数据类型
        xx = state[5*i+1].to(dtype=self.FLOAT_MODE)
        # 计算 x 和 xx 的加权平均值
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        # 将 x 转换为浮点数，并赋值给 state 中第 i 个元素
        state[5*i+1] = x.float()

        # 计算 sigmoid 函数的值
        r = torch.sigmoid(xr @ rw)
        # 计算 xk 和 kw 的点积，并转换为浮点数
        k = (xk @ kw).float()
        # 计算 xv 和 vw 的点积，并转换为浮点数
        v = (xv @ vw).float()

        # 从 state 中获取第 5*i+2 个元素的值
        aa = state[5*i+2]
        # 从 state 中获取第 5*i+3 个元素的值
        bb = state[5*i+3]
        # 从 state 中获取第 5*i+4 个元素的值
        pp = state[5*i+4]
        # 计算 ww 的值
        ww = time_first + k
        # 计算 p 的值，取 pp 和 ww 中的较大值
        p = torch.maximum(pp, ww)
        # 计算 e1 和 e2 的值
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        # 计算 a 和 b 的值
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        # 重新计算 ww 的值
        ww = pp + time_decay
        # 计算 p 的值，取 ww 和 k 中的较大值
        p = torch.maximum(ww, k)
        # 计算 e1 和 e2 的值，并更新 state 中的第 5*i+2 和第 5*i+3 个元素的值
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        # 计算 wkv 的值
        wkv = (a / b).to(dtype=self.FLOAT_MODE)
        # 返回 (r * wkv) 与 ow 的点积
        return (r * wkv) @ ow

    # 装饰器，用于修饰下面的函数
    @MyFunction
    # 定义一个方法，用于执行SA_seq操作，接受多个参数
    def SA_seq(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        # 将state中的第5*i+1个元素与x的前T-1行拼接成一个新的张量
        xx = torch.cat((state[5*i+1].to(dtype=self.FLOAT_MODE).unsqueeze(0), x[:-1,:]))
        # 计算x和xx的加权平均值，得到xk、xv、xr
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        # 更新state中的第5*i+1个元素为x的最后一行
        state[5*i+1] = x[-1,:].float()

        # 计算r、k、v
        r = torch.sigmoid(xr @ rw)
        k = (xk @ kw).float()
        v = (xv @ vw).float()

        # 获取state中的一些元素和x的行数T
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        T = x.shape[0]
        # 遍历T个时间步
        for t in range(T):
            # 计算ww和p
            ww = time_first + k[t]
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            a = e1 * aa + e2 * v[t]
            b = e1 * bb + e2
            ww = pp + time_decay
            p = torch.maximum(ww, k[t])
            e1 = torch.exp(ww - p)
            e2 = torch.exp(k[t] - p)
            # 根据t是否为T-1来更新aa、bb、pp或者更新state中的一些元素
            if t != T - 1:
                aa = e1 * aa + e2 * v[t]
                bb = e1 * bb + e2
                pp = p
            else:
                state[5*i+2] = e1 * aa + e2 * v[t]
                state[5*i+3] = e1 * bb + e2
                state[5*i+4] = p
            xx[t] = (a / b).to(dtype=self.FLOAT_MODE)
        # 返回r * xx与ow的矩阵乘积
        return (r * xx) @ ow
    # 定义一个前向传播函数，接受 tokens（输入的标记序列）、state（模型状态）、preprocess_only（是否只进行预处理）三个参数
    def forward(self, tokens, state, preprocess_only = False):
        # 禁用梯度计算
        with torch.no_grad():
            # 获取模型的权重和参数
            w = self.w
            args = self.args

            # 判断是否为序列模式
            seq_mode = len(tokens) > 1

            # 根据序列模式与否选择不同的输入处理方式
            x = w.emb.weight[tokens] if seq_mode else w.emb.weight[tokens[0]]
            # 如果运行设备为 cuda，则将输入数据转移到 cuda 设备上
            if 'cuda' in self.RUN_DEVICE:
                x = x.to(self.RUN_DEVICE)

            # 如果状态为空，则初始化为全零状态
            if state == None:
                state = torch.zeros(args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                for i in range(args.n_layer):
                    state[5*i+4] -= 1e30

            # 根据序列模式选择不同的自注意力层和前馈神经网络层
            SA = self.SA_seq if seq_mode else self.SA_one
            FF = self.FF_seq if seq_mode else self.FF_one

            # 遍历每一层的处理过程
            for i in range(args.n_layer):
                # 获取当前层的自注意力层和前馈神经网络层
                ww = w.blocks[i].att
                # 使用自注意力层处理输入数据
                x = x + SA(self.LN(x, w.blocks[i].ln1), state, i, 
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                
                # 获取当前层的前馈神经网络层
                ww = w.blocks[i].ffn
                # 使用前馈神经网络层处理输入数据
                x = x + FF(self.LN(x, w.blocks[i].ln2), state, i, 
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)
                
                # 如果浮点数模式为 fp16，则每隔一定层次对输入数据进行缩放
                if args.FLOAT_MODE == 'fp16':
                    if (i+1) % RWKV_RESCALE_LAYER == 0:
                        x = x / 2

            # 如果只进行预处理，则返回状态
            if preprocess_only:
                return state

            # 对最终的输出数据进行处理，返回输出结果和状态
            x = self.LN(x[-1,:], w.ln_out) if seq_mode else self.LN(x, w.ln_out)
            x = w.head.weight @ x

            return x.float(), state
```