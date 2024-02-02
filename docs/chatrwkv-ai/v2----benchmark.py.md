# `ChatRWKV\v2\benchmark.py`

```py
# 导入所需的库
import os, sys, types, json, math, time
# 获取当前文件所在路径
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
# 以只读方式打开 lambada_test.jsonl 文件，并将每行的 JSON 数据加载为列表
with open(f"{current_path}/../misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    # 对每个文档的文本进行处理，将文本按最后一个空格分割成两部分
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

# 设置环境变量 RWKV_JIT_ON 为 '1'
os.environ["RWKV_JIT_ON"] = '1'
# 设置环境变量 RWKV_CUDA_ON 为 '1'
os.environ["RWKV_CUDA_ON"] = '1'

# 设置模型名称
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230213-8019'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20230109-ctx4096'
MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'

# 设置 PAD_SEQ 列表
PAD_SEQ = [187]

# 打印加载信息
print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')
# 导入 torch 库，并设置加速选项
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# 调整以下设置（对所有设置为 True/False 进行测试），以找到最快的设置：
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# 设置可以在 GPU 上融合操作
# 禁用张量表达式融合器
# 禁用 NV 融合器

# 导入 torch.nn 的 functional 模块，并重命名为 F
from torch.nn import functional as F
# 从 rwkv.model 模块中导入 RWKV 类
from rwkv.model import RWKV
# 从 rwkv.utils 模块中导入 PIPELINE 和 PIPELINE_ARGS
from rwkv.utils import PIPELINE, PIPELINE_ARGS

# 打印加载模型的信息
print(f'Loading model - {MODEL_NAME}')
# 创建 RWKV 模型对象，使用给定的模型名称和策略
model = RWKV(model=MODEL_NAME, strategy='cuda fp16')
# 创建 PIPELINE 对象，使用 model 和 "20B_tokenizer.json" 作为参数
pipeline = PIPELINE(model, "20B_tokenizer.json")

# 打印"Warmup..."信息
print('Warmup...')
# 调用模型的 forward 方法，传入输入数据和状态，获取输出和新状态
out, state = model.forward([187, 510, 1563, 310, 247], None, full_output=True)
# 打印最后一个输出的数据
print(out[-1,:].detach().cpu().numpy())
# 调用模型的 forward 方法，传入输入数据和状态，获取输出和新状态
out, state = model.forward([187], None)
# 打印输出数据
print(out.detach().cpu().numpy())
# 调用模型的 forward 方法，传入输入数据和状态，获取输出和新状态
out, state = model.forward([510, 1563], state)
out, state = model.forward([310, 247], state)
# 打印输出数据
print(out.detach().cpu().numpy())
# 调用模型的 forward 方法，传入输入数据和状态，获取输出和新状态
out, state = model.forward([187], None)
out, state = model.forward([510, 1563, 310, 247], state)
# 打印输出数据
print(out.detach().cpu().numpy())
# 调用模型的 forward 方法，传入输入数据和状态，获取输出和新状态
out, state = model.forward([187, 510, 1563, 310], None)
out, state = model.forward([247], state)
# 打印输出数据
print(out.detach().cpu().numpy())
# 调用模型的 forward 方法，传入输入数据和状态，获取输出和新状态
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)
out, state = model.forward([310, 247], state)
# 打印输出数据
print(out.detach().cpu().numpy())
# 初始化标记，使用pipeline对文本进行编码
init_token = pipeline.encode("In the event that the Purchaser defaults in the payment of any instalment of purchase price")

# 输出基准速度
print('Benchmark speed...')
# 初始化时间槽字典
time_slot = {}

# 记录时间的函数
def record_time(name):
    # 如果名称不在时间槽字典中，则初始化为一个很大的数
    if name not in time_slot:
        time_slot[name] = 1e20
    # 计算时间差，并将结果存入时间槽字典
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

# 循环10次
for i in range(10):
    # 记录当前时间
    time_ref = time.time_ns()
    # 模型前向传播，获取输出和状态
    out, state = model.forward(init_token, None)
    # 将输出转换为numpy数组，并记录时间
    aa = out.detach().cpu().numpy()
    record_time('fast')
    # 打印快速时间槽的时间和输出
    print(f"fast {round(time_slot['fast'], 4)}s {aa}")

    # 记录当前时间
    time_ref = time.time_ns()
    # 循环遍历init_token的每个元素
    for j in range(len(init_token)):
        out, state = model.forward([init_token[j]], None if j == 0 else state)
    # 将输出转换为numpy数组，并记录时间
    aa = out.detach().cpu().numpy()
    record_time('slow')
    # 打印慢速时间槽的时间和输出
    print(f"slow {round(time_slot['slow'], 4)}s {aa}")

# 退出程序
# exit(0)

########################################################################################################

# 输出检查LAMBADA
print('Check LAMBADA...')
# 初始化变量
xsum = 0
xcnt = 0
xacc = 0
# 遍历todo列表
for d in todo:
    # 对输入文本进行编码
    src = PAD_SEQ + pipeline.encode(d[0])
    dst = pipeline.encode(d[1])

    # 初始化logits和correct
    logits = 0
    correct = True
    # 模型前向传播，获取输出和模型状态
    out, model_state = model.forward(src+dst, None, full_output=True)
    # 遍历dst的每个元素
    for i in range(len(dst)):
        # 计算概率并累加logits，检查是否预测正确
        probs = F.softmax(out[len(src)-1+i,:], dim=-1)
        logits += math.log(probs[dst[i]])
        if torch.argmax(probs).item() != dst[i]:
            correct = False

    # 更新计数和累加值
    xcnt += 1
    xsum += logits
    xacc += 1 if correct else 0
    # 每100次或者最后一次打印结果
    if xcnt % 100 == 0 or xcnt == len(todo):
        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))
```