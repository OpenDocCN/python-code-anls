# `d:/src/tocomm/Bert-VITS2\onnx_modules\V200\text\japanese_bert.py`

```
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数

import torch  # 导入torch模块，用于构建深度学习模型
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config变量
from .japanese import text2sep_kata  # 从当前目录下的japanese模块中导入text2sep_kata函数

LOCAL_PATH = "./bert/deberta-v2-large-japanese"  # 设置本地路径变量

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用预训练模型路径初始化tokenizer对象

models = dict()  # 创建一个空字典用于存储模型

def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):  # 定义一个名为get_bert_feature的函数，接受text、word2ph和device三个参数
    sep_text, _, _ = text2sep_kata(text)  # 调用text2sep_kata函数，将text转换为分隔的片假名文本
    sep_tokens = [tokenizer.tokenize(t) for t in sep_text]  # 使用tokenizer对分隔的片假名文本进行分词
    sep_ids = [tokenizer.convert_tokens_to_ids(t) for t in sep_tokens]  # 将分词后的文本转换为对应的token id
    sep_ids = [2] + [item for sublist in sep_ids for item in sublist] + [3]  # 对token id进行处理
    return get_bert_feature_with_token(sep_ids, word2ph, device)  # 调用函数get_bert_feature_with_token，传入参数sep_ids, word2ph, device，并返回结果

def get_bert_feature_with_token(tokens, word2ph, device=config.bert_gen_config.device):  # 定义函数get_bert_feature_with_token，接受参数tokens, word2ph, device，并设置默认值为config.bert_gen_config.device
    if (  # 如果条件判断语句开始
        sys.platform == "darwin"  # 判断当前操作系统是否为darwin
        and torch.backends.mps.is_available()  # 判断torch是否支持MPS
        and device == "cpu"  # 判断device是否为cpu
    ):  # 条件判断语句结束
        device = "mps"  # 如果条件成立，将device设置为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 将device设置为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)  # 将device对应的模型加载到models中，并设置其运行的device
    with torch.no_grad():  # 使用torch的no_grad上下文管理器
        inputs = torch.tensor(tokens).to(device).unsqueeze(0)  # 将tokens转换为tensor，并移动到指定的device上，然后增加一个维度
        token_type_ids = torch.zeros_like(inputs).to(device)  # 创建与inputs相同形状的全零tensor，并移动到指定的device上
        attention_mask = torch.ones_like(inputs).to(device)  # 创建与inputs相同形状的全一tensor，并移动到指定的device上
        inputs = {  # 创建一个字典inputs
            "input_ids": inputs,  # 将inputs中的"input_ids"键对应的值设置为inputs
        "token_type_ids": token_type_ids,  # 将token_type_ids添加到inputs字典中
        "attention_mask": attention_mask,  # 将attention_mask添加到inputs字典中
    }

    # for i in inputs:
    #     inputs[i] = inputs[i].to(device)  # 将inputs中的每个值转移到指定的设备上
    res = models[device](**inputs, output_hidden_states=True)  # 使用指定的模型和输入进行推理，输出隐藏状态
    res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态的最后三层拼接起来，并转移到CPU上

    assert inputs["input_ids"].shape[-1] == len(word2ph)  # 断言输入的input_ids的最后一个维度长度等于word2ph的长度
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 初始化一个空列表用于存储特征

    for i in range(len(word2phone)):  # 遍历word2phone的长度
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 将res中的每个元素重复word2phone[i]次，并按照第二个维度进行重复
        phone_level_feature.append(repeat_feature)  # 将重复后的特征添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 将phone_level_feature中的特征按照第一个维度拼接起来

    return phone_level_feature.T  # 返回phone_level_feature的转置
```