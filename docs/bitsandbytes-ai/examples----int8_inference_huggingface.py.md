# `bitsandbytes\examples\int8_inference_huggingface.py`

```py
# 导入 torch 库
import torch
# 从 transformers 库中导入 AutoModelForCausalLM 和 AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 定义最大新标记数量
MAX_NEW_TOKENS = 128
# 模型名称
model_name = 'decapoda-research/llama-7b-hf'

# 初始文本
text = 'Hamburg is in which country?\n'
# 使用给定模型名称创建分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 使用分词器将文本转换为模型输入的张量
input_ids = tokenizer(text, return_tensors="pt").input_ids

# 获取当前 GPU 的可用内存大小
free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
# 计算模型最大可用内存
max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

# 获取当前系统中 GPU 的数量
n_gpus = torch.cuda.device_count()
# 为每个 GPU 设置最大可用内存
max_memory = {i: max_memory for i in range(n_gpus)}

# 使用给定参数从预训练模型中创建自动回归语言模型
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  device_map='auto',
  load_in_8bit=True,
  max_memory=max_memory
)
# 生成新的标记序列
generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
# 解码生成的标记序列并打印结果
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```