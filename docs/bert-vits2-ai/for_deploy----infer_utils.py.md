# `d:/src/tocomm/Bert-VITS2\for_deploy\infer_utils.py`

```
import sys  # 导入sys模块，用于访问系统相关的功能

import torch  # 导入torch模块，用于构建和训练神经网络
from transformers import (  # 导入transformers模块，用于自然语言处理任务
    AutoModelForMaskedLM,  # 导入AutoModelForMaskedLM类，用于掩码语言模型任务
    AutoTokenizer,  # 导入AutoTokenizer类，用于自动标记化文本
    DebertaV2Model,  # 导入DebertaV2Model类，用于DeBERTa V2模型
    DebertaV2Tokenizer,  # 导入DebertaV2Tokenizer类，用于DeBERTa V2模型的标记化
    ClapModel,  # 导入ClapModel类，用于CLAP模型
    ClapProcessor,  # 导入ClapProcessor类，用于CLAP模型的处理
)

from config import config  # 导入config模块中的config变量，用于配置信息
from text.japanese import text2sep_kata  # 导入text.japanese模块中的text2sep_kata函数，用于将日语文本转换为分隔的片假名


class BertFeature:  # 定义BertFeature类
    def __init__(self, model_path, language="ZH"):  # 定义初始化方法，接收模型路径和语言参数
        self.model_path = model_path  # 将模型路径赋值给实例变量model_path
        self.language = language  # 将语言参数赋值给实例变量language
        self.tokenizer = None
        self.model = None
        self.device = None
```
这段代码定义了三个变量`tokenizer`、`model`和`device`，并将它们初始化为`None`。

```
        self._prepare()
```
调用了`_prepare()`方法。

```
    def _get_device(self, device=config.bert_gen_config.device):
```
定义了一个名为`_get_device`的方法，该方法有一个参数`device`，默认值为`config.bert_gen_config.device`。

```
        if (
            sys.platform == "darwin"
            and torch.backends.mps.is_available()
            and device == "cpu"
        ):
            device = "mps"
```
如果操作系统是`darwin`（即MacOS），并且`torch.backends.mps.is_available()`返回`True`，并且`device`的值是`cpu`，则将`device`的值设置为`mps`。

```
        if not device:
            device = "cuda"
```
如果`device`的值为空，则将`device`的值设置为`cuda`。

```
        return device
```
返回`device`的值。

```
    def _prepare(self):
        self.device = self._get_device()
```
定义了一个名为`_prepare`的方法，该方法调用了`_get_device()`方法，并将返回的值赋给`self.device`。
        if self.language == "EN":
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)
            self.model = DebertaV2Model.from_pretrained(self.model_path).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path).to(
                self.device
            )
        self.model.eval()
```

这段代码根据`self.language`的值选择不同的模型和分词器。如果`self.language`为"EN"，则使用`DebertaV2Tokenizer`和`DebertaV2Model`，否则使用`AutoTokenizer`和`AutoModelForMaskedLM`。然后将模型移动到指定的设备上，并将模型设置为评估模式。

```
    def get_bert_feature(self, text, word2ph):
        if self.language == "JP":
            text = "".join(text2sep_kata(text)[0])
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
```

这段代码用于获取文本的BERT特征。如果`self.language`为"JP"，则将文本转换为特定格式。然后使用分词器对文本进行编码，并将编码后的输入移动到指定的设备上。接下来，使用模型对输入进行推理，并获取隐藏状态。最后，将隐藏状态的特定部分进行拼接，并将结果移动到CPU上。
        word2phone = word2ph
```
将`word2ph`赋值给`word2phone`变量。

```
        phone_level_feature = []
```
创建一个空列表`phone_level_feature`。

```
        for i in range(len(word2phone)):
```
对于`word2phone`的每个索引`i`，执行以下操作：

```
            repeat_feature = res[i].repeat(word2phone[i], 1)
```
将`res[i]`重复`word2phone[i]`次，并在维度1上重复。将结果赋值给`repeat_feature`变量。

```
            phone_level_feature.append(repeat_feature)
```
将`repeat_feature`添加到`phone_level_feature`列表中。

```
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
```
将`phone_level_feature`列表中的所有张量在维度0上进行拼接，得到一个新的张量，并将结果赋值给`phone_level_feature`变量。

```
        return phone_level_feature.T
```
返回`phone_level_feature`的转置。

```
class ClapFeature:
```
定义一个名为`ClapFeature`的类。

```
    def __init__(self, model_path):
```
定义`ClapFeature`类的初始化方法，接受一个`model_path`参数。

```
        self.model_path = model_path
```
将`model_path`赋值给`self.model_path`变量。

```
        self.processor = None
        self.model = None
        self.device = None
```
将`None`赋值给`self.processor`、`self.model`和`self.device`变量。

```
        self._prepare()
```
调用`_prepare()`方法。
    def _get_device(self, device=config.bert_gen_config.device):
        # 判断操作系统是否为 macOS，是否支持 MPS（Mac Pro Server）加速，以及设备是否为 CPU
        if (
            sys.platform == "darwin"
            and torch.backends.mps.is_available()
            and device == "cpu"
        ):
            # 如果满足条件，则将设备设置为 "mps"
            device = "mps"
        # 如果设备为空，则将设备设置为 "cuda"
        if not device:
            device = "cuda"
        # 返回设备
        return device

    def _prepare(self):
        # 获取设备
        self.device = self._get_device()

        # 从预训练模型路径加载 ClapProcessor，并将其移动到设备上
        self.processor = ClapProcessor.from_pretrained(self.model_path)
        # 从预训练模型路径加载 ClapModel，并将其移动到设备上
        self.model = ClapModel.from_pretrained(self.model_path).to(self.device)
        # 设置模型为评估模式
        self.model.eval()

    def get_clap_audio_feature(self, audio_data):
        # 禁用梯度计算
        with torch.no_grad():
# 根据输入的音频数据，使用预处理器将其转换为PyTorch张量，并设置采样率为48000
inputs = self.processor(
    audios=audio_data, return_tensors="pt", sampling_rate=48000
).to(self.device)
# 使用模型提取音频特征
emb = self.model.get_audio_features(**inputs)
# 返回音频特征的转置
return emb.T
```

```
# 根据输入的文本数据，使用预处理器将其转换为PyTorch张量
inputs = self.processor(text=text, return_tensors="pt").to(self.device)
# 使用模型提取文本特征
emb = self.model.get_text_features(**inputs)
# 返回文本特征的转置
return emb.T
```

这段代码是一个类的方法，其中包含两个函数。第一个函数`get_audio_feature`用于从音频数据中提取特征，第二个函数`get_clap_text_feature`用于从文本数据中提取特征。这两个函数的作用是相似的，都是将输入数据转换为PyTorch张量，并使用模型提取特征。最后，返回特征的转置。
```