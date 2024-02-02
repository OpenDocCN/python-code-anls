# `Bert-VITS2\oldVersion\V101\text\chinese_bert.py`

```py
import torch  # 导入 PyTorch 库
import sys  # 导入系统相关的库
from transformers import AutoTokenizer, AutoModelForMaskedLM  # 从 transformers 库中导入 AutoTokenizer 和 AutoModelForMaskedLM 类

device = torch.device(  # 设置设备，如果有 CUDA 则使用 CUDA，否则使用 MPS 或 CPU
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if sys.platform == "darwin" and torch.backends.mps.is_available()
        else "cpu"
    )
)

tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")  # 从预训练模型中加载分词器
model = AutoModelForMaskedLM.from_pretrained("./bert/chinese-roberta-wwm-ext-large").to(
    device
)  # 从预训练模型中加载模型，并将其移动到指定设备上


def get_bert_feature(text, word2ph):
    with torch.no_grad():  # 禁用梯度计算
        inputs = tokenizer(text, return_tensors="pt")  # 使用分词器对文本进行分词，并返回 PyTorch 张量
        for i in inputs:  # 将输入张量移动到指定设备上
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)  # 使用模型进行推理，获取隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并移动到 CPU 上

    assert len(word2ph) == len(text) + 2  # 断言确保 word2ph 的长度符合预期
    word2phone = word2ph  # 将 word2ph 赋值给 word2phone
    phone_level_feature = []  # 初始化 phone_level_feature 列表
    for i in range(len(word2phone)):  # 遍历 word2phone
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 将 res[i] 重复 word2phone[i] 次
        phone_level_feature.append(repeat_feature)  # 将重复的特征添加到 phone_level_feature 中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 沿着指定维度拼接 phone_level_feature

    return phone_level_feature.T  # 返回 phone_level_feature 的转置


if __name__ == "__main__":  # 如果当前脚本被直接执行
    # feature = get_bert_feature('你好,我是说的道理。')
    import torch  # 导入 PyTorch 库

    word_level_feature = torch.rand(38, 1024)  # 创建一个随机张量，表示每个词的特征
    word2phone = [  # 定义每个词对应的音素数
        1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1
    ]

    # 计算总帧数
    total_frames = sum(word2phone)  # 计算音素总数
    print(word_level_feature.shape)  # 打印 word_level_feature 的形状
    print(word2phone)  # 打印 word2phone 列表
    phone_level_feature = []  # 初始化 phone_level_feature 列表
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