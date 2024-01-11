# `ChatRWKV\v2\benchmark_world.py`

```
# 导入所需的库
import os, sys, types, json, math, time
# 获取当前文件所在目录的路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 将上级目录的路径添加到系统路径中
sys.path.append(f'{current_path}/../rwkv_pip_package/src')
# 尝试设置环境变量 CUDA_VISIBLE_DEVICES 为命令行参数中的第一个参数
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
# 导入 numpy 库，并设置打印选项
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# 设置环境变量 RWKV_JIT_ON 为 1
os.environ["RWKV_JIT_ON"] = '1'
# 设置环境变量 RWKV_CUDA_ON 为 0，用于控制是否使用 CUDA
os.environ["RWKV_CUDA_ON"] = '0' # set to 1 for much faster generation

# 设置模型名称
MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-5-world/RWKV-5-World-0.4B-v2-20231113-ctx4096'

# 打印加载信息
print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')
# 导入 torch 库，并设置相关参数
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
# 创建 pipeline 对象
pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # !!! update rwkv pip package to 0.8+ !!!

# 预定义的问题列表
QUESTIONS = '''
Make a grocery list for a healthy meal.
You need to write a creative opening scene for a horror movie.
Write a python function that sorts a list from large to small.
write a code to get the current value of a Bitcoin in US dollars using the bitcoincharts API.
Write a program to compute the sum of integers from k to n.
Write a script in JavaScript for printing the current date and time.
请介绍北京的旅游景点
Énumérez les attractions touristiques à Paris
東京ではどこを訪れるべきですか？
Can coughing effectively stop a heart attack?
What happens if you crack your knuckles a lot?
'''
# 以上是一系列问题，不是代码，需要删除
# 定义一个空列表，可能用于填充
PAD_TOKENS = [] # [] or [0] or [187] -> probably useful

# 打印模型名称
print(MODEL_NAME)
# 遍历问题列表
for q in QUESTIONS:
    # 初始化变量
    out_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    ctx = f'User: {q.strip()}\n\nAssistant:'
    # 打印用户问题和助手回复的上下文
    print(ctx, end = '')
    # 循环生成回复
    for i in range(200):
        # 如果是第一次循环，使用PAD_TOKENS填充tokens，否则使用上一次的token
        tokens = PAD_TOKENS + pipeline.encode(ctx) if i == 0 else [token]
        
        # 使用模型进行前向推理
        out, state = pipeline.model.forward(tokens, state)
        # 对出现过的token进行重复惩罚
        for n in occurrence:
            out[n] -= (0 + occurrence[n] * 1.0) # repetition penalty
        
        # 从logits中采样出下一个token
        token = pipeline.sample_logits(out, temperature=1.0, top_p=0.1)
        # 当token为0时退出循环，即遇到'endoftext'
        if token == 0: break
        
        # 将token加入输出token列表，并更新token出现次数
        out_tokens += [token]
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        
        # 解码生成的token序列
        tmp = pipeline.decode(out_tokens[out_last:])
        # 当字符串是有效的utf-8且不以\n结尾时打印
        if ('\ufffd' not in tmp) and (not tmp.endswith('\n')):
            print(tmp, end = '', flush = True)
            out_str += tmp
            out_last = i + 1
        
        # 当遇到'\n\n'时退出循环
        if '\n\n' in tmp:
            out_str += tmp
            out_str = out_str.strip()
            break

    # 打印分隔线
    print('\n' + '=' * 50)
```