# `.\HippoRAG\src\processing.py`

```py
# 导入 JSON 模块，用于处理 JSON 数据
import json
# 导入正则表达式模块，用于模式匹配和替换
import re

# 导入 NumPy 模块，用于数值计算
import numpy as np
# 导入 PyTorch 模块，用于深度学习模型和张量操作
import torch


# 提取路径中的文件名，去掉 JSON 扩展名
def get_file_name(path):
    return path.split('/')[-1].replace('.jsonl', '').replace('.json', '')


# 对 token 嵌入进行均值池化操作，计算句子嵌入
def mean_pooling(token_embeddings, mask):
    # 将 mask 为 False 的位置在 token_embeddings 中填充为 0
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    # 按句子维度求和并除以有效 token 数量得到句子嵌入
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


# 生成输入字符串的均值池化嵌入，支持模型和设备选择
def mean_pooling_embedding(input_str: str, tokenizer, model, device='cuda'):
    # 使用 tokenizer 对输入字符串进行编码，转换为模型输入
    inputs = tokenizer(input_str, padding=True, truncation=True, return_tensors='pt').to(device)
    # 通过模型获取输出嵌入
    outputs = model(**inputs)

    # 对输出嵌入进行均值池化，并转移到 CPU，转换为 NumPy 数组
    embedding = mean_pooling(outputs[0], inputs['attention_mask']).to('cpu').detach().numpy()
    return embedding


# 生成归一化的均值池化嵌入
def mean_pooling_embedding_with_normalization(input_str, tokenizer, model, device='cuda'):
    # 对输入字符串进行编码，准备模型输入
    encoding = tokenizer(input_str, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    # 通过模型获取输出嵌入
    outputs = model(input_ids, attention_mask=attention_mask)
    # 对嵌入进行均值池化
    embeddings = mean_pooling(outputs[0], attention_mask)
    # 对嵌入进行归一化处理
    embeddings = embeddings.T.divide(torch.linalg.norm(embeddings, dim=1)).T

    return embeddings


# 处理短语，将非字母数字字符替换为空格，并将短语转为小写
def processing_phrases(phrase):
    return re.sub('[^A-Za-z0-9 ]', ' ', phrase.lower()).strip()


# 从文本中提取 JSON 字典
def extract_json_dict(text):
    # 定义用于匹配 JSON 对象的正则表达式模式
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}'
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            # 尝试将 JSON 字符串解析为字典
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            # 如果 JSON 解析失败，返回空字符串
            return ''
    else:
        # 如果没有找到 JSON 字符串，返回空字符串
        return ''


# 对输入数据进行最小-最大归一化
def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
```