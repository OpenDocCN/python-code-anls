# `qlora\examples\guanaco_generate.py`

```
# 导入所需的模块
import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer

# 获取最新的检查点目录
def get_last_checkpoint(checkpoint_dir):
    # 检查目录是否存在
    if isdir(checkpoint_dir):
        # 检查是否已经完成训练
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # 已经完成训练
        max_step = 0
        # 遍历目录下的文件
        for filename in os.listdir(checkpoint_dir):
            # 如果是目录且以'checkpoint'开头
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                # 获取最大的步数
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        # 如果没有检查点
        if max_step == 0: return None, is_completed # 训练已经开始，但没有检查点
        # 获取最新的检查点目录
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # 找到检查点
    return None, False # 第一次训练
# TODO: 更新变量
# 设置最大新标记数
max_new_tokens = 64
# 设置 top-p 参数
top_p = 0.9
# 设置温度参数
temperature=0.7
# 用户问题
user_question = "What is Einstein's theory of relativity?"

# 基础模型
model_name_or_path = 'huggyllama/llama-7b'
# 适配器名称在 HF hub 或本地检查点路径
# 适配器路径
adapter_path = 'timdettmers/guanaco-7b'

# 从预训练模型中加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# 修复一些早期 LLaMA HF 转换问题
tokenizer.bos_token_id = 1

# 加载模型（使用 bf16 以加快推理速度）
model = AutoModelForCausalLM.from_pretrained(
# 定义模型的名称或路径
model_name_or_path,
# 定义 torch 数据类型为 bfloat16
torch_dtype=torch.bfloat16,
# 定义设备映射，空字符串对应设备 0
device_map={"": 0},
# 加载 4 位数据
load_in_4bit=True,
# 定义量化配置，包括加载 4 位数据、计算数据类型为 bfloat16、使用双量化、量化类型为 nf4
quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

# 从预训练模型或路径创建 PeftModel 模型
model = PeftModel.from_pretrained(model, adapter_path)
# 设置模型为评估模式
model.eval()

# 设置对话提示
prompt = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "### Human: {user_question}"
    "### Assistant: "
)
# 定义一个生成函数，用于生成模型的输出文本
def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    # 使用tokenizer将用户问题转换为模型输入的张量，并将其移动到GPU上
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')

    # 使用模型生成文本
    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,  # 设置为True表示使用采样方式生成文本
            max_new_tokens=max_new_tokens,  # 生成文本的最大token数
            top_p=top_p,  # 用于动态调整采样概率的参数
            temperature=temperature,  # 用于控制生成文本的多样性的参数
        )
    )

    # 将生成的文本张量解码为可读的文本
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 打印生成的文本
    print(text)
    # 返回生成的文本
    return text

# 调用生成函数，传入模型和用户问题作为参数
generate(model, user_question)
# 导入pdb模块并设置断点，用于在程序执行过程中进行调试。
```