# `ChatRWKV\API_DEMO.py`

```
# 打印 RWKV 语言模型的 GitHub 链接
print('\nChatRWKV https://github.com/BlinkDL/ChatRWKV\n')

# 导入所需的库
import os, sys, torch
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# 设置当前路径和导入路径
# current_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(f'{current_path}/rwkv_pip_package/src')

# 调整下面的设置（对所有设置测试 True/False）以找到最快的设置：
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

# 在模型路径中使用 '/' 而不是 '\'，如果需要长上下文，请使用 ctx4096 模型。
# fp16 = 适用于 GPU（不支持 CPU）
# fp32 = 适用于 CPU
# bf16 = 精度较差，支持 CPU
# xxxi8（例如：fp16i8）= xxx 与 int8 量化，以节省 50% 的 VRAM/RAM，速度较慢，精度略低
# 阅读 https://pypi.org/project/rwkv/ 了解策略指南
# 在导入 RWKV 之前设置这些
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0'  # '1' 编译 CUDA 内核（速度提高 10 倍），需要 c++ 编译器和 cuda 库

# 导入 RWKV 模型并设置策略
from rwkv.model import RWKV  # pip install rwkv
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16i8')
# 创建 RWKV 对象，指定模型路径和计算策略
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16i8 *6 -> cuda fp16 *0+ -> cpu fp32 *1')
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cpu fp32')
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cpu fp32 *3 -> cuda fp16 *6+')
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cpu fp32')
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16')
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16 *8 -> cpu fp32')
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda:0 fp16 -> cuda:1 fp16 -> cpu fp32 *1')
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16 *6+')
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230213-8019', strategy='cuda fp16 *0+ -> cpu fp32 *1')
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096', strategy='cuda:0 fp16 *25 -> cuda:1 fp16')

# 调用模型的 forward 方法，传入输入数据和状态
out, state = model.forward([187, 510, 1563, 310, 247], None)
# 打印输出数据的 CPU 数组
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
out, state = model.forward([310, 247], state)
# 打印输出数据的 CPU 数组
print(out.detach().cpu().numpy())                   # same result as above

# 导入 PIPELINE 和 PIPELINE_ARGS
from rwkv.utils import PIPELINE, PIPELINE_ARGS
# 创建 PIPELINE 对象，传入模型和 tokenizer 文件路径
pipeline = PIPELINE(model, "20B_tokenizer.json")
# 定义一个包含龙的故事的字符串
ctx = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
# 打印字符串，不换行
print(ctx, end='')

# 定义一个自定义的打印函数，打印字符串，不换行，并刷新缓冲区
def my_print(s):
    print(s, end='', flush=True)

# 设置生成文本的参数，包括温度、top_p、top_k、alpha_frequency、alpha_presence、token_ban、token_stop、chunk_len
# 详细参数说明可参考链接：https://platform.openai.com/docs/api-reference/parameter-details
args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k=0, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

########################################################################################################
# 1. 如果可能，设置环境变量 os.environ["RWKV_CUDA_ON"] = '1'，以加快对长上下文的预处理。
# 2. 当多次运行相同的上下文时，重用状态（使用深拷贝进行克隆）。
# 生成文本，指定生成的标记数量为200，使用上述参数和自定义的回调函数
pipeline.generate(ctx, token_count=200, args=args, callback=my_print)

# 打印换行符
print('\n')
```