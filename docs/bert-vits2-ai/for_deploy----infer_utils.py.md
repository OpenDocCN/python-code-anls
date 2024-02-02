# `Bert-VITS2\for_deploy\infer_utils.py`

```py
# 导入 sys 模块
import sys

# 导入 torch 模块
import torch
# 从 transformers 模块中导入所需的类
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DebertaV2Model,
    DebertaV2Tokenizer,
    ClapModel,
    ClapProcessor,
)

# 从 config 模块中导入 config 变量
from config import config
# 从 text.japanese 模块中导入 text2sep_kata 函数
from text.japanese import text2sep_kata

# 定义 BertFeature 类
class BertFeature:
    # 初始化方法
    def __init__(self, model_path, language="ZH"):
        # 设置模型路径和语言
        self.model_path = model_path
        self.language = language
        # 初始化 tokenizer、model 和 device
        self.tokenizer = None
        self.model = None
        self.device = None

        # 调用 _prepare 方法
        self._prepare()

    # 获取设备方法
    def _get_device(self, device=config.bert_gen_config.device):
        # 如果是 macOS 并且支持 MPS 并且设备是 CPU，则设备为 MPS
        if (
            sys.platform == "darwin"
            and torch.backends.mps.is_available()
            and device == "cpu"
        ):
            device = "mps"
        # 如果设备未指定，则设备为 CUDA
        if not device:
            device = "cuda"
        return device

    # 准备方法
    def _prepare(self):
        # 获取设备
        self.device = self._get_device()

        # 如果语言是英语
        if self.language == "EN":
            # 使用 DebertaV2Tokenizer 和 DebertaV2Model 初始化 tokenizer 和 model
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)
            self.model = DebertaV2Model.from_pretrained(self.model_path).to(self.device)
        else:
            # 使用 AutoTokenizer 和 AutoModelForMaskedLM 初始化 tokenizer 和 model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path).to(
                self.device
            )
        # 设置 model 为评估模式
        self.model.eval()
    # 获取 BERT 特征的方法，接受文本和单词到音素的映射作为参数
    def get_bert_feature(self, text, word2ph):
        # 如果语言是日语，将文本转换成片假名
        if self.language == "JP":
            text = "".join(text2sep_kata(text)[0])
        # 禁用梯度计算
        with torch.no_grad():
            # 使用 BERT 分词器对文本进行处理，返回 PyTorch 张量
            inputs = self.tokenizer(text, return_tensors="pt")
            # 将输入张量移动到指定设备上
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            # 使用 BERT 模型获取隐藏状态
            res = self.model(**inputs, output_hidden_states=True)
            # 从隐藏状态中取出倒数第三层的输出，并转移到 CPU 上
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

        # 获取单词到音素的映射
        word2phone = word2ph
        # 存储每个音素对应的特征
        phone_level_feature = []
        # 遍历单词到音素的映射
        for i in range(len(word2phone)):
            # 将 BERT 特征重复 word2phone[i] 次，并添加到列表中
            repeat_feature = res[i].repeat(word2phone[i], 1)
            phone_level_feature.append(repeat_feature)

        # 将所有音素对应的特征拼接成一个张量
        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        # 返回转置后的音素级特征
        return phone_level_feature.T
# 定义一个名为 ClapFeature 的类
class ClapFeature:
    # 初始化方法，接受一个 model_path 参数
    def __init__(self, model_path):
        # 将 model_path 参数赋值给实例变量 self.model_path
        self.model_path = model_path
        # 初始化实例变量 processor, model, device
        self.processor = None
        self.model = None
        self.device = None

        # 调用内部方法 _prepare()
        self._prepare()

    # 内部方法，用于获取设备信息
    def _get_device(self, device=config.bert_gen_config.device):
        # 如果是 macOS 平台，并且支持多进程并行计算，并且设备是 CPU，则将设备设置为 "mps"
        if (
            sys.platform == "darwin"
            and torch.backends.mps.is_available()
            and device == "cpu"
        ):
            device = "mps"
        # 如果设备为空，则将设备设置为 "cuda"
        if not device:
            device = "cuda"
        # 返回设备信息
        return device

    # 内部方法，用于准备模型和处理器
    def _prepare(self):
        # 获取设备信息
        self.device = self._get_device()
        # 使用 model_path 加载预训练的 ClapProcessor
        self.processor = ClapProcessor.from_pretrained(self.model_path)
        # 使用 model_path 加载预训练的 ClapModel，并将其移动到指定设备
        self.model = ClapModel.from_pretrained(self.model_path).to(self.device)
        # 设置模型为评估模式
        self.model.eval()

    # 获取音频特征的方法
    def get_clap_audio_feature(self, audio_data):
        # 禁用梯度计算
        with torch.no_grad():
            # 使用处理器处理音频数据，返回张量格式的输入数据，并移动到指定设备
            inputs = self.processor(
                audios=audio_data, return_tensors="pt", sampling_rate=48000
            ).to(self.device)
            # 获取音频特征
            emb = self.model.get_audio_features(**inputs)
        # 返回音频特征的转置
        return emb.T

    # 获取文本特征的方法
    def get_clap_text_feature(self, text):
        # 禁用梯度计算
        with torch.no_grad():
            # 使用处理器处理文本数据，返回张量格式的输入数据，并移动到指定设备
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            # 获取文本特征
            emb = self.model.get_text_features(**inputs)
        # 返回文本特征的转置
        return emb.T
```