# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\utils.py`

```py
# 导入命名空间类 Namespace，用于解析命令行参数
from argparse import Namespace

# 导入 OpenAI 类，用于访问 OpenAI 的 API
from openai import OpenAI

# 导入 AutoModel 和 AutoTokenizer 类，用于加载预训练模型和分词器
from transformers import AutoModel, AutoTokenizer

# 导入 torch，用于深度学习相关操作
import torch

# 导入 random，用于生成随机数
import random

# 导入 tiktoken，具体功能未知
import tiktoken

# 导入 re，用于正则表达式操作
import re

# 导入 numpy，用于数值计算
import numpy as np

# 导入 base64，用于处理 base64 编解码
import base64

# 导入 struct，用于处理字节流与数据类型转换
import struct

# 导入 os，用于操作系统相关功能
import os

# 导入 tqdm，用于显示进度条
import tqdm

# 导入 requests，用于发送 HTTP 请求
import requests


# 从环境变量中获取 API_KEY 和 SECRET_KEY，并使用其生成访问令牌（Access Token）
def get_access_token():
    API_KEY = os.getenv("StoryAudit_API_AK")
    SECRET_KEY = os.getenv("StoryAudit_API_SK")

    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


'''
文本审核接口
'''
# 使用百度 AI 平台的文本审核接口，判断文本是否合规
def text_censor(text):
    request_url = "https://aip.baidubce.com/rest/2.0/solution/v1/text_censor/v2/user_defined"

    params = {"text":text}
    access_token = get_access_token()
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    # 返回文本审核结果是否合规的布尔值
    return response.json()["conclusion"] == "合规"

# 打包角色相关数据，包括系统提示、配置信息和文本文件内容的嵌入向量
def package_role(system_prompt, texts_path, embedding):
    datas = []

    # 暂时只有一种embedding 'luotuo_openai'
    embed_name = 'luotuo_openai'

    datas.append({'text': system_prompt, embed_name: 'system_prompt'})
    datas.append({'text': 'Reserve Config Setting Here', embed_name: 'config'})

    # 获取指定路径下的所有文件列表
    files = os.listdir(texts_path)

    # 遍历每个文件，处理以 .txt 结尾的文本文件
    for i in tqdm.tqdm(range(len(files))):
        file = files[i]
        if file.endswith(".txt"):
            file_path = os.path.join(texts_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                current_str = f.read()
                # 使用指定的嵌入方法将文本转换为向量
                current_vec = embedding(current_str)
                # 将向量转换为 base64 编码，并添加到数据列表中
                encode_vec = float_array_to_base64(current_vec)
                datas.append({'text': current_str, embed_name: encode_vec})

    return datas


# 将字符串转换为 base64 编码
def string_to_base64(text):
    byte_array = b''
    for char in text:
        num_bytes = char.encode('utf-8')
        byte_array += num_bytes

    base64_data = base64.b64encode(byte_array)
    return base64_data.decode('utf-8')

# 将 base64 编码转换为字符串
def base64_to_string(base64_data):
    byte_array = base64.b64decode(base64_data)
    text = byte_array.decode('utf-8')
    return text

# 将浮点数数组转换为 base64 编码
def float_array_to_base64(float_arr):
    byte_array = b''

    for f in float_arr:
        # 将每个浮点数打包为4字节的字节流
        num_bytes = struct.pack('!f', f)
        byte_array += num_bytes

    # 将字节数组进行base64编码
    base64_data = base64.b64encode(byte_array)

    return base64_data.decode('utf-8')

# 将 base64 编码转换为浮点数数组
def base64_to_float_array(base64_data):

    byte_array = base64.b64decode(base64_data)
    float_array = []

    # 每4个字节解析为一个浮点数，直到所有字节都被处理完毕
    for i in range(0, len(byte_array), 4):
        float_value = struct.unpack('!f', byte_array[i:i+4])[0]
        float_array.append(float_value)

    return float_array
    # 将 base64 编码的数据解码成字节数组
    byte_array = base64.b64decode(base64_data)
    
    # 初始化空的浮点数数组
    float_array = []
    
    # 每 4 个字节解析为一个浮点数
    for i in range(0, len(byte_array), 4):
        # 使用 struct 模块解析 4 字节数据为一个浮点数，并添加到浮点数数组中
        num = struct.unpack('!f', byte_array[i:i+4])[0] 
        float_array.append(num)

    # 返回解析后的浮点数数组
    return float_array
# 创建一个torch设备对象，如果CUDA可用则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化变量，用于存储Luotuo模型和Luotuo英文模型以及它们的tokenizer
_luotuo_model = None
_luotuo_model_en = None
_luotuo_en_tokenizer = None

# 初始化变量，用于存储编码模型
_enc_model = None

# ======== add bge_zh mmodel
# by Cheng Li
# 这一次我们试图一次性去适配更多的模型

# 初始化空的模型池和tokenizer池，用于存储多个不同模型的tokenizer和模型对象
_model_pool = {}
_tokenizer_pool = {}

# BAAI/bge-small-zh-v1.5

def get_general_embeddings(sentences, model_name="BAAI/bge-small-zh-v1.5"):
    global _model_pool
    global _tokenizer_pool

    # 如果模型名称不在模型池中，加载模型和tokenizer
    if model_name not in _model_pool:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer_pool[model_name] = AutoTokenizer.from_pretrained(model_name)
        _model_pool[model_name] = AutoModel.from_pretrained(model_name)

    _model_pool[model_name].eval()

    # Tokenize sentences
    encoded_input = _tokenizer_pool[model_name](sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

    # 计算token embeddings
    with torch.no_grad():
        model_output = _model_pool[model_name](**encoded_input)
        # 执行池化操作，这里使用cls池化
        sentence_embeddings = model_output[0][:, 0]

    # 对embeddings进行归一化处理
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().tolist()

def get_general_embedding(text_or_texts, model_name="BAAI/bge-small-zh-v1.5"):
    if isinstance(text_or_texts, str):
        return get_general_embeddings([text_or_texts], model_name)[0]
    else:
        return get_general_embeddings_safe(text_or_texts, model_name)

general_batch_size = 16

import math

def get_general_embeddings_safe(sentences, model_name="BAAI/bge-small-zh-v1.5"):
    embeddings = []
    
    # 计算批次数量
    num_batches = math.ceil(len(sentences) / general_batch_size)
    
    # 对每个批次的句子进行处理
    for i in tqdm.tqdm(range(num_batches)):
        start_index = i * general_batch_size
        end_index = min(len(sentences), start_index + general_batch_size)
        batch = sentences[start_index:end_index]
        embs = get_general_embeddings(batch, model_name)
        embeddings.extend(embs)
        
    return embeddings

def get_bge_zh_embedding(text_or_texts):
    return get_general_embedding(text_or_texts, "BAAI/bge-small-zh-v1.5")

## TODO: 重构bge_en部分的代码，复用general的函数

# ======== add bge model
# by Cheng Li
# for English only right now

# 初始化变量，用于存储BGE英文模型和其tokenizer
_bge_model = None
_bge_tokenizer = None

def get_bge_embeddings(sentences):
    # unsafe ensure batch size by yourself

    global _bge_model
    global _bge_tokenizer

    # 如果BGE模型尚未加载，则加载模型和tokenizer
    if _bge_model is None:
        from transformers import AutoTokenizer, AutoModel
        _bge_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
        _bge_model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')

    _bge_model.eval()

    # Tokenize sentences
    encoded_input = _bge_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

    # 计算token embeddings
    # 使用 torch.no_grad() 上下文管理器，确保在此代码块中不会计算梯度
    with torch.no_grad():
        # 调用 _bge_model 函数，并传入 encoded_input 中的所有参数，获取模型输出
        model_output = _bge_model(**encoded_input)
        # 执行池化操作，此处是使用 cls 池化（取第一个特征的第一个位置）
        sentence_embeddings = model_output[0][:, 0]
    
    # 对 embeddings 进行归一化处理，使用 L2 范数归一化，沿着第一维度（样本维度）
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    # 将归一化后的 embeddings 转移到 CPU 并转换为 Python 列表格式，然后返回
    return sentence_embeddings.cpu().tolist()
def get_bge_embedding(text_or_texts):
    # 如果输入是单个字符串，则将其转换为列表后调用get_bge_embeddings，返回第一个结果
    if isinstance(text_or_texts, str):
        return get_bge_embeddings([text_or_texts])[0]
    else:
        # 否则调用get_bge_embeddings_safe处理输入
        return get_bge_embeddings_safe(text_or_texts)

bge_batch_size = 32

import math
# from tqdm import tqdm

def get_bge_embeddings_safe(sentences):
    # 初始化一个空列表用于存储嵌入向量
    embeddings = []
    
    # 计算批次数，向上取整以确保涵盖所有句子
    num_batches = math.ceil(len(sentences) / bge_batch_size)
    
    # 遍历批次数的范围并显示进度条
    for i in tqdm.tqdm(range(num_batches)):
        # 计算当前批次的起始和结束索引
        start_index = i * bge_batch_size
        end_index = min(len(sentences), start_index + bge_batch_size)
        # 从句子列表中获取当前批次的数据
        batch = sentences[start_index:end_index]
        # 调用get_bge_embeddings获取当前批次的嵌入向量列表
        embs = get_bge_embeddings(batch)
        # 将获取的嵌入向量扩展到embeddings列表中
        embeddings.extend(embs)
        
    return embeddings

# === add bge model

def tiktokenizer(text):
    global _enc_model

    # 如果_enc_model尚未初始化，则使用默认模型"cl100k_base"进行初始化
    if _enc_model is None:
        _enc_model = tiktoken.get_encoding("cl100k_base")

    # 返回输入文本的编码长度
    return len(_enc_model.encode(text))
    
def response_postprocess(text, dialogue_bra_token='「', dialogue_ket_token='」'):
    # 按行分割文本
    lines = text.split('\n')
    new_lines = ""

    first_name = None

    # 遍历每一行文本
    for line in lines:
        line = line.strip(" ")
        # 使用正则表达式匹配形如"姓名:「对话内容」"的模式
        match = re.match(r'^(.*?)[:：]' + dialogue_bra_token + r"(.*?)" + dialogue_ket_token + r"$", line)

        if match:
            curr_name = match.group(1)
            # 如果是第一次匹配到姓名，则初始化first_name，并将对话内容加入new_lines
            if first_name is None:
                first_name = curr_name
                new_lines += (match.group(2))
            else:
                # 如果当前姓名与first_name不同，则返回first_name及其对应的对话内容
                if curr_name != first_name:
                    return first_name + ":" + dialogue_bra_token + new_lines + dialogue_ket_token
                else:
                    # 否则将当前对话内容加入new_lines
                    new_lines += (match.group(2))
            
        else:
            # 如果未匹配到有效对话模式，则根据是否有first_name返回相应结果
            if first_name is None:
                return text
            else:
                return first_name + ":" + dialogue_bra_token + new_lines + dialogue_ket_token
    # 返回最终的first_name及其对应的所有对话内容
    return first_name + ":" + dialogue_bra_token + new_lines + dialogue_ket_token

def download_models():
    # 打印正在下载Luotuo-Bert的信息
    print("正在下载Luotuo-Bert")
    # 定义模型参数
    model_args = Namespace(do_mlm=None, pooler_type="cls", temp=0.05, mlp_only_train=False,
                           init_embeddings_model=None)
    # 从预训练模型"silk-road/luotuo-bert-medium"下载模型，确保远程代码的可信性
    model = AutoModel.from_pretrained("silk-road/luotuo-bert-medium", trust_remote_code=True, model_args=model_args).to(
        device)
    # 下载完成后打印提示信息
    print("Luotuo-Bert下载完毕")
    # 返回下载的模型
    return model

def get_luotuo_model():
    global _luotuo_model
    # 如果_luotuo_model为空，则下载模型并赋值给_luotuo_model
    if _luotuo_model is None:
        _luotuo_model = download_models()
    # 返回_luotuo_model
    return _luotuo_model

def luotuo_embedding(model, texts):
    # 使用AutoTokenizer从"silk-road/luotuo-bert-medium"预训练模型对文本进行标记化处理
    tokenizer = AutoTokenizer.from_pretrained("silk-road/luotuo-bert-medium")
    # 将文本转换为模型可接受的输入格式，并在GPU上执行
    inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    inputs = inputs.to(device)
    # 提取文本的嵌入向量
    # 获取嵌入向量
    # 使用上下文管理器，确保在此代码块中不计算梯度
    with torch.no_grad():
        # 调用模型生成嵌入向量，关闭梯度计算
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
    # 返回生成的嵌入向量作为结果
    return embeddings
def luotuo_openai_embedding(texts, is_chinese=None):
    """
    when input is chinese, use luotuo_embedding
    when input is english, use openai_embedding
    texts can be a list or a string
    when texts is a list, return a list of embeddings, using batch inference
    when texts is a string, return a single embedding
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    # 如果输入的 texts 是一个列表
    if isinstance(texts, list):
        # 随机选择一个索引
        index = random.randint(0, len(texts) - 1)
        # 如果没有提供 OpenAI 的 API 密钥，或者选中的文本是中文
        if openai_key is None or is_chinese_or_english(texts[index]) == "chinese":
            # 对列表中的每个文本获取其对应的中文嵌入向量，返回一个嵌入向量列表
            return [embed.cpu().tolist() for embed in get_embedding_for_chinese(get_luotuo_model(), texts)]
        else:
            # 对列表中的每个英文文本获取其对应的嵌入向量，返回一个嵌入向量列表
            return [get_embedding_for_english(text) for text in texts]
    else:
        # 如果 texts 不是列表
        # 如果没有提供 OpenAI 的 API 密钥，或者输入的文本是中文
        if openai_key is None or is_chinese_or_english(texts) == "chinese":
            # 获取输入文本的中文嵌入向量，返回一个嵌入向量的列表中的第一个向量
            return get_embedding_for_chinese(get_luotuo_model(), texts)[0].cpu().tolist()
        else:
            # 获取输入文本的英文嵌入向量，返回一个嵌入向量
            return get_embedding_for_english(texts)
# 计算两个向量之间的余弦相似度
def get_cosine_similarity(v1, v2):
    # 将输入向量转换为PyTorch张量，并发送到适当的设备（如GPU）
    v1 = torch.tensor(v1).to(device)
    v2 = torch.tensor(v2).to(device)
    # 使用PyTorch计算两个向量之间的余弦相似度，取得标量结果
    return torch.cosine_similarity(v1, v2, dim=0).item()
```