# `d:/src/tocomm/Bert-VITS2\oldVersion\V111\text\fix\japanese_bert.py`

```
import torch  # 导入 PyTorch 库
from transformers import AutoTokenizer, AutoModelForMaskedLM  # 从 transformers 库中导入 AutoTokenizer 和 AutoModelForMaskedLM 类
import sys  # 导入 sys 模块
from .japanese import text2sep_kata  # 从当前目录下的 japanese 模块中导入 text2sep_kata 函数
from config import config  # 从 config 模块中导入 config 对象

tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")  # 使用预训练的日语 BERT 模型初始化 tokenizer 对象

models = dict()  # 创建一个空的字典对象用于存储模型

def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):  # 定义一个名为 get_bert_feature 的函数，接受文本、word2ph 和设备参数
    sep_text, _ = text2sep_kata(text)  # 调用 text2sep_kata 函数将文本转换为片假名
    sep_tokens = [tokenizer.tokenize(t) for t in sep_text]  # 使用 tokenizer 对象将片假名文本分词
    sep_ids = [tokenizer.convert_tokens_to_ids(t) for t in sep_tokens]  # 将分词后的片假名转换为对应的 id
    sep_ids = [2] + [item for sublist in sep_ids for item in sublist] + [3]  # 将片假名 id 组合成一个列表
    return get_bert_feature_with_token(sep_ids, word2ph, device)  # 调用 get_bert_feature_with_token 函数并返回结果

def get_bert_feature_with_token(tokens, word2ph, device=config.bert_gen_config.device):  # 定义一个名为 get_bert_feature_with_token 的函数，接受 tokens、word2ph 和设备参数
    # 检查操作系统是否为 macOS，并且是否支持 MPS（Metal Performance Shaders），以及设备是否为 CPU，如果是，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备未被设置，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则使用预训练模型创建新的模型，并将其移动到设备上
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/bert-base-japanese-v3"
        ).to(device)
    # 使用 torch.no_grad() 上下文管理器，以确保在推断过程中不进行梯度计算
    with torch.no_grad():
        # 将 tokens 转换为 tensor，并移动到设备上，然后在第一维度上增加一个维度
        inputs = torch.tensor(tokens).to(device).unsqueeze(0)
        # 创建 token_type_ids 和 attention_mask，并移动到设备上
        token_type_ids = torch.zeros_like(inputs).to(device)
        attention_mask = torch.ones_like(inputs).to(device)
        # 将输入数据组合成字典
        inputs = {
            "input_ids": inputs,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

        # for i in inputs:
        #     inputs[i] = inputs[i].to(device)  # 将输入数据转移到指定的设备上
        res = models[device](**inputs, output_hidden_states=True)  # 使用指定设备上的模型进行推理，同时输出隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并转移到 CPU 上
    assert inputs["input_ids"].shape[-1] == len(word2ph)  # 断言输入的 input_ids 的最后一个维度长度与 word2ph 的长度相等
    word2phone = word2ph  # 将 word2ph 赋值给 word2phone
    phone_level_feature = []  # 初始化一个空列表用于存储特征
    for i in range(len(word2phone)):  # 遍历 word2phone 的长度
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 将 res[i] 重复 word2phone[i] 次，并沿着第二个维度进行拼接
        phone_level_feature.append(repeat_feature)  # 将重复的特征添加到列表中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 将列表中的特征拼接成一个张量
    return phone_level_feature.T  # 返回转置后的 phone_level_feature
```