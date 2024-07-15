# `.\Chat-Haruhi-Suzumiya\kyon_generator\utils.py`

```py
# 导入模块Namespace，用于创建命名空间
from argparse import Namespace

# 导入OpenAI模块
import openai
# 导入transformers库中的AutoModel和AutoTokenizer类
from transformers import AutoModel, AutoTokenizer
# 导入PyTorch库
import torch
# 导入随机模块
import random

# 检测CUDA是否可用，选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 下载模型函数
def download_models():
    print("正在下载Luotuo-Bert")
    # 创建模型参数命名空间对象
    model_args = Namespace(do_mlm=None, pooler_type="cls", temp=0.05, mlp_only_train=False,
                           init_embeddings_model=None)
    # 从预训练模型中加载Luotuo-Bert模型，并移至指定设备
    model = AutoModel.from_pretrained("silk-road/luotuo-bert-medium", trust_remote_code=True, model_args=model_args).to(
        device)
    print("Luotuo-Bert下载完毕")
    return model

# Luotuo-Bert模型文本嵌入函数
def luotuo_embedding(model, texts):
    # 根据预训练模型加载相应的分词器
    tokenizer = AutoTokenizer.from_pretrained("silk-road/luotuo-bert-medium")
    # 对输入文本进行分词、填充和转换为PyTorch张量
    inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    # 将输入数据移至指定设备
    inputs = inputs.to(device)
    # 使用模型生成文本嵌入
    with torch.no_grad():
        # 获取模型的隐藏状态并返回池化后的输出
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
    return embeddings

# 获取中文文本的嵌入向量函数
def get_embedding_for_chinese(model, texts):
    # 将模型移至指定设备
    model = model.to(device)
    # 如果输入是字符串，则转换为列表
    texts = texts if isinstance(texts, list) else [texts]
    # 截断文本，以确保长度不超过510
    for i in range(len(texts)):
        if len(texts[i]) > 510:
            texts[i] = texts[i][:510]
    # 如果文本数量大于等于64，则分块生成嵌入向量
    if len(texts) >= 64:
        embeddings = []
        chunk_size = 64
        for i in range(0, len(texts), chunk_size):
            embeddings.append(luotuo_embedding(model, texts[i: i + chunk_size]))
        return torch.cat(embeddings, dim=0)
    else:
        return luotuo_embedding(model, texts)

# 判断文本是中文还是英文函数
def is_chinese_or_english(text):
    # 将文本转换为字符列表
    text = list(text)
    is_chinese, is_english = 0, 0
    # 遍历文本中的每个字符，并判断其Unicode范围
    for char in text:
        if '\u4e00' <= char <= '\u9fa5':  # 中文字符的Unicode范围
            is_chinese += 1
        elif ('\u0041' <= char <= '\u005a') or ('\u0061' <= char <= '\u007a') or ('\u0021' <= char <= '\u007e'):
            # 英文字符的Unicode范围（包括大小写字母和常见标点符号）
            is_english += 1
    # 如果中文字符数量大于等于英文字符数量，则返回"chinese"，否则返回"english"
    if is_chinese >= is_english:
        return "chinese"
    else:
        return "english"

# 获取英文文本的嵌入向量函数
def get_embedding_for_english(text, model="text-embedding-ada-002"):
    # 将换行符替换为空格
    text = text.replace("\n", " ")
    print("over here")
    # 使用OpenAI API获取文本的嵌入向量
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# 获取文本的嵌入向量函数
def get_embedding(model, texts):
    """
        return type: list
    """
    # 如果输入为列表，则随机选择一个索引
    if isinstance(texts, list):
        index = random.randint(0, len(texts) - 1)
        # 如果随机选择的文本为中文，则返回中文文本的嵌入向量列表
        if is_chinese_or_english(texts[index]) == "chinese":
            return [embed.cpu().tolist() for embed in get_embedding_for_chinese(model, texts)]
        # 否则返回英文文本的嵌入向量列表
        else:
            return [get_embedding_for_english(text) for text in texts]
    # 如果前面的条件不满足，执行以下代码块
    else:
        # 如果文本是中文，调用 get_embedding_for_chinese 函数获取中文文本的嵌入向量，并转换为列表格式
        if is_chinese_or_english(texts) == "chinese":
            return get_embedding_for_chinese(model, texts)[0].cpu().tolist()
        # 如果文本是英文，调用 get_embedding_for_english 函数获取英文文本的嵌入向量
        else:
            return get_embedding_for_english(texts)
```