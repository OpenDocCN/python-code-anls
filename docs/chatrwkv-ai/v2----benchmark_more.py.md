# `ChatRWKV\v2\benchmark_more.py`

```
# 导入所需的库
import os, sys, types, json, math, time
# 获取当前文件所在路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 将上级目录的 rwkv_pip_package/src 添加到系统路径中
sys.path.append(f'{current_path}/../rwkv_pip_package/src')
# 尝试设置环境变量 CUDA_VISIBLE_DEVICES 为命令行参数中的第一个参数
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
# 导入 numpy 库并设置打印选项
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# 设置环境变量 RWKV_JIT_ON 为 '1'
os.environ["RWKV_JIT_ON"] = '1'
# 设置环境变量 RWKV_CUDA_ON 为 '0'，用于控制是否使用 CUDA
os.environ["RWKV_CUDA_ON"] = '0' # set to '1' for faster processing

# 设置模型名称
MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v11x-Eng99%-Other1%-20230429-ctx8192'

# 打印加载信息
print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')
# 导入 torch 库并设置相关参数
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# 导入 RWKV 模型和相关工具
from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

# 打印加载模型信息
print(f'Loading model - {MODEL_NAME}')
# 创建 RWKV 模型对象
model = RWKV(model=MODEL_NAME, strategy='cuda fp16')
# 创建管道对象
pipeline = PIPELINE(model, "20B_tokenizer.json")

# 定义问题列表
QUESTIONS = '''
What is the tallest mountain in Argentina?
What country is mount Aconcagua in?
What is the tallest mountain in Australia?
What country is Mawson Peak (also known as Mount Kosciuszko) in?
What date was the first iphone announced?
What animal has a long neck and spots on its body?
# 定义一个空列表，可能会在后续的代码中被用到
PAD_TOKENS = [] # [] or [0] or [187] -> probably useful

# 打印模型名称
print(MODEL_NAME)

# 遍历问题列表
for q in QUESTIONS:
    # 打印问题
    print(f'Q: {q.strip()}\nA:', end = '')

    # 初始化输出标记列表、上一个标记位置、输出字符串、出现次数字典、状态和上下文
    out_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    ctx = f'Bob: {q.strip()}\n\nAlice:' # special prompt for Raven Q & A
    # 循环200次，生成文本
    for i in range(200):
        # 如果是第一次循环，将PAD_TOKENS和pipeline.encode(ctx)合并为tokens；否则将[token]赋值给tokens
        tokens = PAD_TOKENS + pipeline.encode(ctx) if i == 0 else [token]
        
        # 使用pipeline.model进行前向传播，得到输出和状态
        out, state = pipeline.model.forward(tokens, state)
        
        # 对输出中的特定标记进行重复惩罚
        for n in occurrence:
            out[n] -= (0.2 + occurrence[n] * 0.2) # repetition penalty
        
        # 从输出中采样下一个token，使用greedy decoding（top_p=0）
        token = pipeline.sample_logits(out, temperature=1.0, top_p=0)
        
        # 如果采样到的token是0，即'endoftext'，则跳出循环
        if token == 0: break # exit when 'endoftext'            
        
        # 将采样到的token添加到输出tokens中
        out_tokens += [token]
        
        # 更新token出现次数的记录
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        
        # 将out_tokens中的部分内容解码为文本
        tmp = pipeline.decode(out_tokens[out_last:])
        
        # 当解码后的文本是有效的utf-8编码且不以\n结尾时，打印文本并将其添加到out_str中
        if ('\ufffd' not in tmp) and (not tmp.endswith('\n')): # only print when the string is valid utf-8 and not end with \n
            print(tmp, end = '', flush = True)
            out_str += tmp
            out_last = i + 1
        
        # 如果解码后的文本中包含'\n\n'，则跳出循环
        if '\n\n' in tmp: # exit when '\n\n'
            out_str += tmp
            out_str = out_str.strip()
            break

    # 打印分隔线
    print('\n' + '=' * 50)
```