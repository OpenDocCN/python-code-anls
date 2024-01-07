# `Bert-VITS2\for_deploy\infer_utils.py`

```

# 导入必要的模块
import sys
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DebertaV2Model,
    DebertaV2Tokenizer,
    ClapModel,
    ClapProcessor,
)
from config import config  # 从config模块中导入config对象
from text.japanese import text2sep_kata  # 从text.japanese模块中导入text2sep_kata函数

# 定义BertFeature类
class BertFeature:
    def __init__(self, model_path, language="ZH"):
        self.model_path = model_path
        self.language = language
        self.tokenizer = None
        self.model = None
        self.device = None
        self._prepare()  # 调用_prepare方法进行初始化

    # 获取设备信息
    def _get_device(self, device=config.bert_gen_config.device):
        # 根据系统平台和设备可用性调整设备信息
        if (
            sys.platform == "darwin"
            and torch.backends.mps.is_available()
            and device == "cpu"
        ):
            device = "mps"
        if not device:
            device = "cuda"
        return device

    # 准备模型和tokenizer
    def _prepare(self):
        self.device = self._get_device()  # 获取设备信息

        # 根据语言选择不同的tokenizer和model
        if self.language == "EN":
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)
            self.model = DebertaV2Model.from_pretrained(self.model_path).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path).to(
                self.device
            )
        self.model.eval()  # 设置模型为评估模式

    # 获取Bert特征
    def get_bert_feature(self, text, word2ph):
        if self.language == "JP":
            text = "".join(text2sep_kata(text)[0])  # 如果是日语，将文本转换为片假名
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")  # 使用tokenizer处理文本
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)  # 将输入数据移动到指定设备
            res = self.model(**inputs, output_hidden_states=True)  # 获取模型输出
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 处理模型输出

        word2phone = word2ph
        phone_level_feature = []
        for i in range(len(word2phone)):
            repeat_feature = res[i].repeat(word2phone[i], 1)  # 重复特征
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 拼接特征

        return phone_level_feature.T  # 返回特征的转置

# 定义ClapFeature类
class ClapFeature:
    def __init__(self, model_path):
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.device = None
        self._prepare()  # 调用_prepare方法进行初始化

    # 获取设备信息
    def _get_device(self, device=config.bert_gen_config.device):
        # 根据系统平台和设备可用性调整设备信息
        if (
            sys.platform == "darwin"
            and torch.backends.mps.is_available()
            and device == "cpu"
        ):
            device = "mps"
        if not device:
            device = "cuda"
        return device

    # 准备模型和processor
    def _prepare(self):
        self.device = self._get_device()  # 获取设备信息

        self.processor = ClapProcessor.from_pretrained(self.model_path)  # 使用processor处理模型
        self.model = ClapModel.from_pretrained(self.model_path).to(self.device)  # 加载模型到指定设备
        self.model.eval()  # 设置模型为评估模式

    # 获取Clap音频特征
    def get_clap_audio_feature(self, audio_data):
        with torch.no_grad():
            inputs = self.processor(
                audios=audio_data, return_tensors="pt", sampling_rate=48000
            ).to(self.device)  # 使用processor处理音频数据
            emb = self.model.get_audio_features(**inputs)  # 获取音频特征
        return emb.T  # 返回特征的转置

    # 获取Clap文本特征
    def get_clap_text_feature(self, text):
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)  # 使用processor处理文本数据
            emb = self.model.get_text_features(**inputs)  # 获取文本特征
        return emb.T  # 返回特征的转置

```