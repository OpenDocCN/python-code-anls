# `qlora\examples\guanaco_generate.py`

```py
# 导入必要的库
import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer

# 获取最新的检查点目录
def get_last_checkpoint(checkpoint_dir):
    # 检查检查点目录是否存在
    if isdir(checkpoint_dir):
        # 检查是否已经完成训练
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # 已经完成
        max_step = 0
        # 遍历检查点目录下的文件
        for filename in os.listdir(checkpoint_dir):
            # 如果是目录并且以'checkpoint'开头
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                # 获取最大步数
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        # 如果没有检查点
        if max_step == 0: return None, is_completed # 训练已开始，但没有检查点
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # 找到检查点
    return None, False # 第一次训练

# TODO: 更新变量
max_new_tokens = 64
top_p = 0.9
temperature=0.7
user_question = "What is Einstein's theory of relativity?"

# 基础模型
model_name_or_path = 'huggyllama/llama-7b'
# 适配器名称在HF hub或本地检查点路径上
# adapter_path, _ = get_last_checkpoint('qlora/output/guanaco-7b')
adapter_path = 'timdettmers/guanaco-7b'

# 从预训练模型中加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# 修复一些早期LLaMA HF转换问题
tokenizer.bos_token_id = 1

# 加载模型（使用bf16进行更快的推断）
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
)

# 从预训练模型和适配器路径中加载PEFT模型
model = PeftModel.from_pretrained(model, adapter_path)
# 设置为评估模式
model.eval()

prompt = (
    # 人类和人工智能助手之间的对话
    # 助手给出有帮助、详细和礼貌的回答
    # 用户提出的问题
    # 助手的回答
# 定义一个生成函数，用于生成文本
def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    # 使用tokenizer将用户问题格式化为模型输入，并转换为PyTorch张量，然后移动到GPU上
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')

    # 使用模型生成文本
    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,  # 开启采样
            max_new_tokens=max_new_tokens,  # 生成的最大标记数
            top_p=top_p,  # 顶部p值，用于控制多样性
            temperature=temperature,  # 温度参数，用于控制生成的多样性
        )
    )

    # 将生成的文本转换为可读的字符串
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 打印生成的文本
    print(text)
    # 返回生成的文本
    return text

# 调用生成函数，生成文本
generate(model, user_question)
# 导入pdb模块，设置断点
import pdb; pdb.set_trace()
```