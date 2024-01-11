# `ChatRWKV\run_lm_eval.py`

```
# 导入所需的库
import os, sys, types, json, math, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

# 设置环境变量
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

# 导入 RWKV 模型和 PIPELINE
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

# 导入 lm_eval 中的任务和评估器
from lm_eval import tasks, evaluator
from lm_eval.models.gpt2 import GPT2LM

# 指定模型名称
MODEL_NAME = "/fsx/BlinkDL/HF-MODEL/rwkv-5-world/RWKV-5-World-1.5B-v2-OnlyForTest_14%_trained-20231001-ctx4096"

# 打印加载的模型名称
print(f'Loading model - {MODEL_NAME}')

# 加载 RWKV 模型
model = RWKV(model=MODEL_NAME, strategy='cuda fp16', verbose=False)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

# 初始化评估任务列表
eval_tasks = []
eval_tasks += ['lambada_openai']

# 初始化 RWKV_PAD，使用 '\n' 作为 PAD
RWKV_PAD = pipeline.tokenizer.encode('\n') 
print('RWKV_PAD', RWKV_PAD)

# 初始化日志缓冲区和正确结果缓冲区
logitBuf = {}
correctBuf = {}

# 定义 TokenizerWrapper 类
class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = 0
    # 定义一个方法，用于将输入的字符串编码成tokens
    def encode(self, string: str, add_special_tokens=False):
        # 调用tokenizer的encode方法，将输入的字符串编码成tokens
        return self.tokenizer.encode(string)
    
    # 定义一个方法，用于将tokens解码成字符串
    def decode(self, tokens):
        # 调用tokenizer的decode方法，将tokens解码成字符串
        return self.tokenizer.decode(tokens)
class EvalHarnessAdapter(GPT2LM):
    # 初始化方法
    def __init__(self):
        # 初始化分词器包装器，使用pipeline.tokenizer
        self.tokenizer = TokenizerWrapper(pipeline.tokenizer)

    # 贪婪搜索方法，用于coqa
    # def greedy_until(self, requests): # designed for coqa
    #     res = []
    #     遍历请求列表
    #     for i in range(len(requests)):
    #         if i % 50 == 0:
    #             打印当前处理的请求索引
    #             print(i)
    #         初始化输出token列表
    #         otoken = []
    #         循环直到条件满足
    #         while True:
    #             使用分词器对请求进行编码并加上otoken
    #             src = self.tokenizer.encode(requests[i][0]) + otoken

    #             限制src的长度不超过4096
    #             src = src[-4096:]
    #             调用模型的forward方法进行推理
    #             outputs, _ = model.forward(src, None)
                
    #             将输出的token加入otoken列表
    #             otoken += [int(torch.argmax(outputs))]
    #             将otoken转换为文本
    #             ss = self.tokenizer.decode(otoken)
    #             如果文本中包含换行符或长度超过200
    #             if '\n' in ss or len(ss) > 200:
    #                 如果文本不以换行符结尾，则加上换行符
    #                 if not ss.endswith('\n'):
    #                     ss = ss + '\n'
    #                 打印文本
    #                 print(ss)
    #                 将文本加入结果列表
    #                 res += [(ss)]
    #                 跳出循环
    #                 break
    #     打印结果列表
    #     print(res)
    #     返回结果列表
    #     return res
    # 计算给定请求的对数似然，返回结果列表
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # 声明全局变量
        global logitBuf, correctBuf

        # 初始化结果列表
        res = []

        # 遍历请求列表
        for COUNTER in range(len(requests)):
            n = COUNTER
            # 获取原始源文本
            raw_src = requests[n][0][0] + requests[n][0][1]

            # 获取处理后的源文本
            src = requests[n][1] + requests[n][2]

            # 对原始源文本和处理后的源文本进行处理
            raw_src = '\n' + raw_src
            src = RWKV_PAD + src

            # 将处理后的源文本转换为字符串
            sss = str(src)
            correct = True

            # 检查处理后的源文本是否在缓存中
            if sss in logitBuf:
                logit = logitBuf[sss]
                correct = correctBuf[sss]
            else:
                # 如果不在缓存中，则进行计算
                q_len = len(requests[n][1])
                q_len += len(RWKV_PAD)
                logit = 0
                
                # 使用 torch.no_grad() 上下文管理器，禁用梯度计算
                with torch.no_grad():
                    # 调用模型的前向传播方法，获取输出
                    outputs, _ = model.forward(src, None, full_output=True)
                    # 遍历计算对数似然
                    for i in range(q_len-1, len(src)-1):
                        oo = outputs[i].detach().float()
                        dst = src[i+1]
                        logit += math.log(F.softmax(oo, dim=-1)[dst])
                        _, s_index = torch.sort(oo, descending=True)
                        pred = s_index[0].item()
                        if pred != dst:
                            correct = False
                    outputs = None
                    pred = None
                # 将计算结果存入缓存
                logitBuf[sss] = logit
                correctBuf[sss] = correct
            
            # 将计算结果加入结果列表
            res += [(logit, correct)]
            # 每处理1000个请求打印一次进度
            if n % 1000 == 0:
                print(f'{n//1000}/{len(requests)//1000}', end = ' ', flush=True)
        # 返回结果列表
        return res

    # 评估方法，使用 torch.no_grad() 上下文管理器，禁用梯度计算
    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=0, bootstrap_iters=2):
        # 调用评估器的 evaluate 方法进行评估
        results = evaluator.evaluate(
            lm=self,
            task_dict=tasks.get_task_dict(eval_tasks),
            provide_description=False,
            num_fewshot=num_fewshot,
            limit=None,
            bootstrap_iters=bootstrap_iters,
        )
        # 返回评估结果
        return results
# 创建一个评估适配器对象
adapter = EvalHarnessAdapter()
# 运行评估适配器的评估任务，并设置 bootstrap_iters 参数为 10000
results = adapter.run_eval(
    eval_tasks=eval_tasks,
    bootstrap_iters=10000,
)
# 打印评估结果字典中的 'results' 键对应的数值
print(results['results'])
```