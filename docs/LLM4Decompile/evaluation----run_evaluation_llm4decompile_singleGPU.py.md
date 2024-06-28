# `.\LLM4Decompile\evaluation\run_evaluation_llm4decompile_singleGPU.py`

```
# 导入 subprocess 模块，用于执行外部命令
# 导入 transformers 库中的 AutoTokenizer 和 AutoModelForCausalLM 类
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 os 模块，提供了访问操作系统服务的功能
import os
# 导入 torch 库，用于构建和训练神经网络
import torch
# 导入 re 模块，用于处理正则表达式
import re
# 导入 json 模块，用于读取和写入 JSON 数据
import json
# 导入 tqdm 模块，用于显示进度条
from tqdm import tqdm, trange

# 禁用 Tokenizers 的并行处理，避免与 transformers 库的冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加 model_path 参数，指定模型路径，默认为 'LLM4Binary/llm4decompile-6.7b-v1.5'
parser.add_argument('--model_path', type=str, default='LLM4Binary/llm4decompile-6.7b-v1.5', required=False)
# 添加 data_path 参数，指定数据路径，默认为 '../decompile-eval/decompile-eval-executable-gcc-obj.json'
parser.add_argument('--data_path', type=str, default='../decompile-eval/decompile-eval-executable-gcc-obj.json', required=False)

# 解析命令行参数
args = parser.parse_args()

# 定义 evaluate_func 函数，用于评估编译和运行 C 函数的结果
def evaluate_func(c_func, c_test, c_func_decompile):
    # 初始化编译和运行的标志
    flag_compile = 0
    flag_run = 0
    c_include = ''

    # 处理 c_func 中的 #include 行，并将其从 c_func 中移除，添加到 c_include 中
    for line in c_func.split('\n'):
        if '#include' in line:
            c_include += line + '\n'
            c_func = c_func.replace(line, '')

    # 处理 c_test 中的 #include 行，并将其从 c_test 中移除，添加到 c_include 中
    for line in c_test.split('\n'):
        if '#include' in line:
            c_include += line + '\n'
            c_test = c_test.replace(line, '')

    # 将 c_func_decompile、c_include 和 c_test 组合成一个完整的 C 代码
    c_combine = c_include + '\n' + c_func_decompile + '\n' + c_test
    # 仅包含函数定义的 C 代码
    c_onlyfunc = c_include + '\n' + c_func_decompile

    # 定义 C 文件名和可执行文件名
    c_file = 'combine.c'
    executable = 'combine'
    # 如果已存在同名的可执行文件，则删除
    if os.path.exists(executable):
        os.remove(executable)

    c_file_onlyfunc = 'onlyfunc.c'
    executable_onlyfunc = 'onlyfunc'
    # 如果已存在同名的可执行文件，则删除
    if os.path.exists(executable_onlyfunc):
        os.remove(executable_onlyfunc)

    # 将 c_combine 写入 combine.c 文件
    with open(c_file, 'w') as f:
        f.write(c_combine)
    # 将 c_onlyfunc 写入 onlyfunc.c 文件
    with open(c_file_onlyfunc, 'w') as f:
        f.write(c_onlyfunc)

    # 编译 C 程序为汇编代码
    compile_command = f'gcc -S {c_file_onlyfunc} -o {executable_onlyfunc} -lm'
    try:
        subprocess.run(compile_command, shell=True, check=True)
        flag_compile = 1
    except:
        return flag_compile, flag_run

    # 编译 C 程序为可执行文件
    compile_command = f'gcc {c_file} -o {executable} -lm'
    try:
        subprocess.run(compile_command, shell=True, check=True)
        flag_compile = 1
    except:
        return flag_compile, flag_run

    # 运行编译后的可执行文件
    run_command = f'./{executable}'
    try:
        process = subprocess.run(run_command, shell=True, check=True, capture_output=True, timeout=5)
        flag_run = 1
    except subprocess.CalledProcessError as e:
        pass
    except Exception as e:
        pass

    # 返回编译和运行的标志
    return flag_compile, flag_run

# 加载预训练的 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
# 加载预训练的 model，并将其移至 GPU 上
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda()
print('Model Loaded!')

# 设置 tokenizer 的 pad_token 为 eos_token，并更新 pad_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# 更新 model 的配置中的 pad_token_id
model.config.pad_token_id = tokenizer.eos_token_id

# 定义优化选项
OPT = ["O0", "O1", "O2", "O3"]
# 从指定路径中读取 JSON 数据
with open(args.data_path, 'r') as f:
    data_all = json.load(f)
# 计算数据的分割数量
NUM = int(len(data_all) / 4)
# 初始化编译成功和运行成功的计数器
num_compile = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}
num_run = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}

# 遍历数据列表，并显示进度条
for idx in trange(len(data_all)):
    c_func = data_all[idx]['c_func']
    c_test = data_all[idx]['c_test']
    # 从 data_all 列表中获取索引为 idx 的字典，提取键为 'input_asm_prompt' 的值
    input_asm_prompt = data_all[idx]['input_asm_prompt']
    # 从 data_all 列表中获取索引为 idx 的字典，提取键为 'type' 的值，赋给 opt_state
    opt_state = data_all[idx]['type']
    # 创建一个字符串，包含优化状态 opt_state 的注释信息，并赋给 before 变量
    before = f"# This is the assembly code with {opt_state} optimization:\n"
    # 创建一个字符串，包含提示信息并赋给 after 变量
    after = "\n# What is the source code?\n"
    # 将 input_asm_prompt 字符串的前后部分加上 before 和 after 的内容，并去除首尾空白字符，重新赋给 input_asm_prompt
    input_asm_prompt = before + input_asm_prompt.strip() + after
    # 使用 tokenizer 对 input_asm_prompt 进行处理，返回 PyTorch 张量格式的输入数据，存储在 inputs 变量中
    inputs = tokenizer(input_asm_prompt, return_tensors="pt").to(model.device)
    # 禁用梯度计算，生成模型的输出，限制最大新增标记为 512，并将结果存储在 outputs 变量中
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    # 使用 tokenizer 解码模型生成的输出，从 inputs[0] 到倒数第二个位置，存储在 c_func_decompile 变量中
    c_func_decompile = tokenizer.decode(outputs[0][len(inputs[0]):-1])
    # 调用 evaluate_func 函数，评估编译后的函数 c_func、测试用例 c_test 和反编译得到的函数 c_func_decompile
    flag_compile, flag_run = evaluate_func(c_func, c_test, c_func_decompile)
    # 更新 num_compile 和 num_run 字典中 opt_state 对应的值，分别加上 flag_compile 和 flag_run
    num_compile[opt_state] += flag_compile
    num_run[opt_state] += flag_run
# 以追加模式打开文件'results.txt'，文件对象赋值给变量f，该文件在此代码块范围内有效
with open('results.txt','a') as f:
    # 迭代num_compile字典中的键，将每个键存储在变量opt_state中
    for opt_state in num_compile.keys():
        # 将格式化的字符串写入文件f，包括模型路径args.model_path、优化状态opt_state，
        # 以及num_compile[opt_state]除以NUM的值作为编译率和num_run[opt_state]除以NUM的值作为运行率
        f.write('model:{},opt:{},compile rate:{:.4f},run_rate:{:.4f}\n'.format(args.model_path,opt_state,num_compile[opt_state]/NUM,num_run[opt_state]/NUM))
```