# `.\Langchain-Chatchat\embeddings\add_embedding_keywords.py`

```
'''
该功能是为了将关键词加入到embedding模型中，以便于在embedding模型中进行关键词的embedding
该功能的实现是通过修改embedding模型的tokenizer来实现的
该功能仅仅对EMBEDDING_MODEL参数对应的的模型有效，输出后的模型保存在原本模型
感谢@CharlesJu1和@charlesyju的贡献提出了想法和最基础的PR

保存的模型的位置位于原本嵌入模型的目录下，模型的名称为原模型名称+Merge_Keywords_时间戳
'''
import sys

sys.path.append("..")
import os
import torch

from datetime import datetime
from configs import (
    MODEL_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_KEYWORD_FILE,
)

from safetensors.torch import save_model
from sentence_transformers import SentenceTransformer
from langchain_core._api import deprecated

# 标记函数已弃用
@deprecated(
        since="0.3.0",
        message="自定义关键词 Langchain-Chatchat 0.3.x 重写, 0.2.x中相关功能将废弃",
        removal="0.3.0"
    )
def get_keyword_embedding(bert_model, tokenizer, key_words):
    # 使用tokenizer处理关键词，返回张量
    tokenizer_output = tokenizer(key_words, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenizer_output['input_ids']
    input_ids = input_ids[:, 1:-1]

    # 获取关键词的embedding
    keyword_embedding = bert_model.embeddings.word_embeddings(input_ids)
    keyword_embedding = torch.mean(keyword_embedding, 1)
    return keyword_embedding

# 将关键词添加到模型中
def add_keyword_to_model(model_name=EMBEDDING_MODEL, keyword_file: str = "", output_model_path: str = None):
    key_words = []
    with open(keyword_file, "r") as f:
        for line in f:
            key_words.append(line.strip())

    # 加载SentenceTransformer模型
    st_model = SentenceTransformer(model_name)
    key_words_len = len(key_words)
    word_embedding_model = st_model._first_module()
    bert_model = word_embedding_model.auto_model
    tokenizer = word_embedding_model.tokenizer
    key_words_embedding = get_keyword_embedding(bert_model, tokenizer, key_words)

    embedding_weight = bert_model.embeddings.word_embeddings.weight
    embedding_weight_len = len(embedding_weight)
    tokenizer.add_tokens(key_words)
    bert_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
    embedding_weight = bert_model.embeddings.word_embeddings.weight
    # 使用 torch.no_grad() 上下文管理器，确保在此范围内的操作不会计算梯度
    with torch.no_grad():
        # 将关键词的嵌入权重赋值给指定范围的嵌入权重
        embedding_weight[embedding_weight_len:embedding_weight_len + key_words_len, :] = key_words_embedding

    # 如果指定了输出模型路径
    if output_model_path:
        # 创建输出模型路径，如果路径不存在则创建
        os.makedirs(output_model_path, exist_ok=True)
        # 保存词嵌入模型到输出路径
        word_embedding_model.save(output_model_path)
        # 安全地将模型保存到指定路径
        safetensors_file = os.path.join(output_model_path, "model.safetensors")
        metadata = {'format': 'pt'}
        save_model(bert_model, safetensors_file, metadata)
        # 打印保存模型的信息
        print("save model to {}".format(output_model_path))
# 将关键词添加到嵌入模型中
def add_keyword_to_embedding_model(path: str = EMBEDDING_KEYWORD_FILE):
    # 拼接关键词文件路径
    keyword_file = os.path.join(path)
    # 获取嵌入模型的名称
    model_name = MODEL_PATH["embed_model"][EMBEDDING_MODEL]
    # 获取嵌入模型的父目录
    model_parent_directory = os.path.dirname(model_name)
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 构建输出模型名称
    output_model_name = "{}_Merge_Keywords_{}".format(EMBEDDING_MODEL, current_time)
    # 拼接输出模型路径
    output_model_path = os.path.join(model_parent_directory, output_model_name)
    # 调用函数将关键词添加到模型中
    add_keyword_to_model(model_name, keyword_file, output_model_path)
```