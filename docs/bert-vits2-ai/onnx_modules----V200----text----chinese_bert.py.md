# `Bert-VITS2\onnx_modules\V200\text\chinese_bert.py`

```
# 导入sys模块
import sys

# 导入torch模块
import torch
# 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类
from transformers import AutoModelForMaskedLM, AutoTokenizer
# 从config模块中导入config变量
from config import config

# 设置本地路径
LOCAL_PATH = "./bert/chinese-roberta-wwm-ext-large"
# 使用AutoTokenizer类从预训练模型中加载分词器
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

# 创建空字典models
models = dict()

# 定义函数get_bert_feature，接受text、word2ph和device参数
def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):
    # 如果操作系统是macOS，并且支持多进程并行计算，并且设备是CPU，则将设备设置为"mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备为空，则将设备设置为"cuda"
    if not device:
        device = "cuda"
    # 如果设备不在models字典的键中
    if device not in models.keys():
        # 将models字典中对应设备的值设置为从本地路径加载的预训练模型，并移动到指定设备
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用分词器对文本进行编码，返回PyTorch张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将inputs中的每个值移动到指定设备
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 将输入传入对应设备的模型，输出隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 将隐藏状态的倒数第三层到倒数第二层的输出拼接起来，并移动到CPU
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    # 断言word2ph的长度等于text的长度加2
    assert len(word2ph) == len(text) + 2
    # 将word2ph赋值给word2phone
    word2phone = word2ph
    # 创建空列表phone_level_feature
    phone_level_feature = []
    # 遍历word2phone
    for i in range(len(word2phone)):
        # 将res[i]重复word2phone[i]次，并添加到phone_level_feature列表中
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 将phone_level_feature列表中的张量拼接起来
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回phone_level_feature的转置
    return phone_level_feature.T


# 如果���前脚本被执行
if __name__ == "__main__":
    # 创建一个随机张量word_level_feature，形状为(38, 1024)
    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
    # 创建一个列表word2phone
    word2phone = [
        1,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
    ]

    # 计算word2phone列表中的总和，即总帧数
    total_frames = sum(word2phone)
    # 打印word_level_feature的形状
    print(word_level_feature.shape)
    # 打印word2phone列表
    print(word2phone)
    # 创建空列表phone_level_feature
    phone_level_feature = []
    # 遍历 word2phone 列表的长度范围
    for i in range(len(word2phone)):
        # 打印 word_level_feature[i] 的形状
        print(word_level_feature[i].shape)

        # 对每个词重复 word2phone[i] 次，沿着第二维度进行重复
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        # 将重复后的特征添加到 phone_level_feature 列表中
        phone_level_feature.append(repeat_feature)

    # 将 phone_level_feature 列表中的 tensor 沿着第一维度拼接起来
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # 打印拼接后的 phone_level_feature 的形状
    print(phone_level_feature.shape)  # torch.Size([36, 1024])
```