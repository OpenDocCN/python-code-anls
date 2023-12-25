# Bert-VITS2 源码解析 5

# `D:\src\Bert-VITS2\oldVersion\V220\text\japanese_bert.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config类
from text.japanese import text2sep_kata  # 从text.japanese模块中导入text2sep_kata函数

LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"  # 设置LOCAL_PATH变量为"./bert/deberta-v2-large-japanese-char-wwm"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用AutoTokenizer类从预训练模型路径LOCAL_PATH中加载tokenizer

models = dict()  # 创建一个空字典models

def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    text = "".join(text2sep_kata(text)[0])  # 将text转换为片假名并连接成字符串
    if (
        sys.platform == "darwin"  # 如果系统平台是darwin
        and torch.backends.mps.is_available()  # 并且torch后端支持MPS
        and device == "cpu"  # 并且设备是CPU
    ):
        device = "mps"  # 将设备设置为"mps"
    if not device:  # 如果设备为空
        device = "cuda"  # 将设备设置为"cuda"
    if device not in models.keys():  # 如果设备不在models的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)  # 从预训练模型路径LOCAL_PATH中加载AutoModelForMaskedLM类，并将其移动到设备上
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对text进行编码并返回张量
        for i in inputs:  # 遍历inputs
            inputs[i] = inputs[i].to(device)  # 将inputs的每个值移动到设备上
        res = models[device](**inputs, output_hidden_states=True)  # 使用models对inputs进行预测，并返回隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并移动到CPU上
        if style_text:  # 如果style_text存在
            style_inputs = tokenizer(style_text, return_tensors="pt")  # 使用tokenizer对style_text进行编码并返回张量
            for i in style_inputs:  # 遍历style_inputs
                style_inputs[i] = style_inputs[i].to(device)  # 将style_inputs的每个值移动到设备上
            style_res = models[device](**style_inputs, output_hidden_states=True)  # 使用models对style_inputs进行预测，并返回隐藏状态
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并移动到CPU上
            style_res_mean = style_res.mean(0)  # 计算style_res的均值

    assert len(word2ph) == len(text) + 2  # 断言word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        if style_text:  # 如果style_text存在
            repeat_feature = (  # 计算repeat_feature
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)  # 重复res[i]并乘以(1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight  # 重复style_res_mean并乘以style_weight
            )
        else:  # 如果style_text不存在
            repeat_feature = res[i].repeat(word2phone[i], 1)  # 重复res[i]
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 沿着0维度拼接phone_level_feature

    return phone_level_feature.T  # 返回phone_level_feature的转置
```

# `D:\src\Bert-VITS2\oldVersion\V220\text\symbols.py`

```python
punctuation = ["!", "?", "…", ",", ".", "'", "-"]  # 创建一个包含标点符号的列表
pu_symbols = punctuation + ["SP", "UNK"]  # 创建一个包含标点符号和特殊标记的列表
pad = "_"  # 创建一个填充标记

# chinese
zh_symbols = [  # 创建一个包含中文音节的列表
    "E",
    "En",
    ...
    "OO",
]
num_zh_tones = 6  # 设置中文音节的音调数量为6

# japanese
ja_symbols = [  # 创建一个包含日文音节的列表
    "N",
    "a",
    ...
    "zy",
]
num_ja_tones = 2  # 设置日文音节的音调数量为2

# English
en_symbols = [  # 创建一个包含英文音素的列表
    "aa",
    "ae",
    ...
    "zh",
]
num_en_tones = 4  # 设置英文音素的音调数量为4

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))  # 合并所有音节并去重排序
symbols = [pad] + normal_symbols + pu_symbols  # 创建一个包含填充标记、所有音节和特殊标记的列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]  # 获取特殊标记在列表中的索引

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones  # 计算所有音节的总音调数量

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}  # 创建一个语言到ID的映射字典
num_languages = len(language_id_map.keys())  # 获取语言数量

language_tone_start_map = {  # 创建一个语言到音调起始位置的映射字典
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)  # 创建一个包含中文音节的集合
    b = set(en_symbols)  # 创建一个包含英文音素的集合
    print(sorted(a & b))  # 打印中英文共有的音节
```

# `D:\src\Bert-VITS2\oldVersion\V220\text\tone_sandhi.py`

```python
# the meaning of jieba pos tag: https://blog.csdn.net/weixin_44174352/article/details/113731041
# e.g.
# word: "家里"
# pos: "s"
# finals: ['ia1', 'i3']
def _neural_sandhi(self, word: str, pos: str, finals: List[str]) -> List[str]:
    # reduplication words for n. and v. e.g. 奶奶, 试试, 旺旺
    for j, item in enumerate(word):
        if (
            j - 1 >= 0
            and item == word[j - 1]
            and pos[0] in {"n", "v", "a"}
            and word not in self.must_not_neural_tone_words
        ):
            finals[j] = finals[j][:-1] + "5"
    ge_idx = word.find("个")
    if len(word) >= 1 and word[-1] in "吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶":
        finals[-1] = finals[-1][:-1] + "5"
    elif len(word) >= 1 and word[-1] in "的地得":
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 走了, 看着, 去过
    # elif len(word) == 1 and word in "了着过" and pos in {"ul", "uz", "ug"}:
    #     finals[-1] = finals[-1][:-1] + "5"
    elif (
        len(word) > 1
        and word[-1] in "们子"
        and pos in {"r", "n"}
        and word not in self.must_not_neural_tone_words
    ):
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 桌上, 地下, 家里
    elif len(word) > 1 and word[-1] in "上下里" and pos in {"s", "l", "f"}:
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 上来, 下去
    elif len(word) > 1 and word[-1] in "来去" and word[-2] in "上下进出回过起开":
        finals[-1] = finals[-1][:-1] + "5"
    # 个做量词
    elif (
        ge_idx >= 1
        and (word[ge_idx - 1].isnumeric() or word[ge_idx - 1] in "几有两半多各整每做是")
    ) or word == "个":
        finals[ge_idx] = finals[ge_idx][:-1] + "5"
    else:
        if (
            word in self.must_neural_tone_words
            or word[-2:] in self.must_neural_tone_words
        ):
            finals[-1] = finals[-1][:-1] + "5"

    word_list = self._split_word(word)
    finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]]
    for i, word in enumerate(word_list):
        # conventional neural in Chinese
        if (
            word in self.must_neural_tone_words
            or word[-2:] in self.must_neural_tone_words
        ):
            finals_list[i][-1] = finals_list[i][-1][:-1] + "5"
    finals = sum(finals_list, [])
    return finals
```

# `D:\src\Bert-VITS2\oldVersion\V220\text\__init__.py`

```python
from .symbols import *  # Import all symbols from the symbols module

_symbol_to_id = {s: i for i, s in enumerate(symbols)}  # Create a dictionary mapping symbols to their corresponding IDs

def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]  # Convert cleaned text symbols to their corresponding IDs
    tone_start = language_tone_start_map[language]  # Get the tone start value based on the language
    tones = [i + tone_start for i in tones]  # Add the tone start value to each tone
    lang_id = language_id_map[language]  # Get the language ID based on the language
    lang_ids = [lang_id for i in phones]  # Create a list of language IDs corresponding to the phones
    return phones, tones, lang_ids  # Return the converted phones, tones, and language IDs

def get_bert(norm_text, word2ph, language, device, style_text=None, style_weight=0.7):
    from .chinese_bert import get_bert_feature as zh_bert  # Import the get_bert_feature function from chinese_bert module
    from .english_bert_mock import get_bert_feature as en_bert  # Import the get_bert_feature function from english_bert_mock module
    from .japanese_bert import get_bert_feature as jp_bert  # Import the get_bert_feature function from japanese_bert module

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}  # Create a map of language to bert feature function
    bert = lang_bert_func_map[language](  # Get the bert feature using the language-specific bert feature function
        norm_text, word2ph, device, style_text, style_weight
    )
    return bert  # Return the bert feature

def check_bert_models():
    import json  # Import the json module
    from pathlib import Path  # Import the Path class from the pathlib module
    from config import config  # Import the config object from the config module
    from .bert_utils import _check_bert  # Import the _check_bert function from the bert_utils module

    if config.mirror.lower() == "openi":  # Check if the mirror in the config is set to "openi"
        import openi  # Import the openi module
        kwargs = {"token": config.openi_token} if config.openi_token else {}  # Create kwargs with token if openi_token is set in config
        openi.login(**kwargs)  # Login to openi using the token from config

    with open("./bert/bert_models.json", "r") as fp:  # Open the bert_models.json file for reading
        models = json.load(fp)  # Load the contents of the file as JSON
        for k, v in models.items():  # Iterate through the items in the loaded JSON
            local_path = Path("./bert").joinpath(k)  # Create a local path for the model file
            _check_bert(v["repo_id"], v["files"], local_path)  # Check the bert model using the _check_bert function

def init_openjtalk():
    import platform  # Import the platform module

    if platform.platform() == "Linux":  # Check if the platform is Linux
        import pyopenjtalk  # Import the pyopenjtalk module
        pyopenjtalk.g2p("こんにちは，世界。")  # Perform g2p conversion for the given text

init_openjtalk()  # Initialize openjtalk
check_bert_models()  # Check the bert models
```

# `D:\src\Bert-VITS2\onnx_modules\__init__.py`

```python
from utils import get_hparams_from_file, load_checkpoint  # 导入自定义模块中的函数
import json  # 导入json模块


def export_onnx(export_path, model_path, config_path, novq, dev):  # 定义函数export_onnx，接受5个参数
    hps = get_hparams_from_file(config_path)  # 调用get_hparams_from_file函数，将返回值赋给hps
    version = hps.version[0:3]  # 从hps中取出version的前3个字符，赋给version
    if version == "2.0" or (version == "2.1" and novq):  # 如果version等于"2.0"或者version等于"2.1"并且novq为真
        from .V200 import SynthesizerTrn, symbols  # 从.V200模块中导入SynthesizerTrn和symbols
    elif version == "2.1" and (not novq):  # 否则如果version等于"2.1"并且novq为假
        from .V210 import SynthesizerTrn, symbols  # 从.V210模块中导入SynthesizerTrn和symbols
    elif version == "2.2":  # 否则如果version等于"2.2"
        if novq and dev:  # 如果novq和dev都为真
            from .V220_novq_dev import SynthesizerTrn, symbols  # 从.V220_novq_dev模块中导入SynthesizerTrn和symbols
        else:  # 否则
            from .V220 import SynthesizerTrn, symbols  # 从.V220模块中导入SynthesizerTrn和symbols
    elif version == "2.3":  # 否则如果version等于"2.3"
        from .V230 import SynthesizerTrn, symbols  # 从.V230模块中导入SynthesizerTrn和symbols
    net_g = SynthesizerTrn(  # 创建SynthesizerTrn对象net_g
        len(symbols),  # 传入参数为symbols的长度
        hps.data.filter_length // 2 + 1,  # 传入参数为hps.data.filter_length除以2再加1
        hps.train.segment_size // hps.data.hop_length,  # 传入参数为hps.train.segment_size除以hps.data.hop_length
        n_speakers=hps.data.n_speakers,  # 传入参数n_speakers为hps.data.n_speakers
        **hps.model,  # 传入hps.model中的所有参数
    )
    _ = net_g.eval()  # 对net_g进行评估
    _ = load_checkpoint(model_path, net_g, None, skip_optimizer=True)  # 调用load_checkpoint函数
    net_g.cpu()  # 将net_g转移到CPU
    net_g.export_onnx(export_path)  # 调用net_g的export_onnx方法，传入参数export_path

    spklist = []  # 创建空列表spklist
    for key in hps.data.spk2id.keys():  # 遍历hps.data.spk2id的键
        spklist.append(key)  # 将键添加到spklist中

    MoeVSConf = {  # 创建字典MoeVSConf
        "Folder": f"{export_path}",  # 设置"Folder"键的值为export_path
        "Name": f"{export_path}",  # 设置"Name"键的值为export_path
        "Type": "BertVits",  # 设置"Type"键的值为"BertVits"
        "Symbol": symbols,  # 设置"Symbol"键的值为symbols
        "Cleaner": "",  # 设置"Cleaner"键的值为空字符串
        "Rate": hps.data.sampling_rate,  # 设置"Rate"键的值为hps.data.sampling_rate
        "CharaMix": True,  # 设置"CharaMix"键的值为True
        "Characters": spklist,  # 设置"Characters"键的值为spklist
        "LanguageMap": {"ZH": [0, 0], "JP": [1, 6], "EN": [2, 8]},  # 设置"LanguageMap"键的值为指定的字典
        "Dict": "BasicDict",  # 设置"Dict"键的值为"BasicDict"
        "BertPath": [  # 设置"BertPath"键的值为指定的列表
            "chinese-roberta-wwm-ext-large",
            "deberta-v2-large-japanese",
            "bert-base-japanese-v3",
        ],
        "Clap": "clap-htsat-fused",  # 设置"Clap"键的值为"clap-htsat-fused"
    }

    with open(f"onnx/{export_path}.json", "w") as MoeVsConfFile:  # 打开文件，写入模式，文件句柄为MoeVsConfFile
        json.dump(MoeVSConf, MoeVsConfFile, indent=4)  # 将MoeVSConf转换为JSON格式并写入文件，缩进为4
```

# `D:\src\Bert-VITS2\onnx_modules\V200\attentions_onnx.py`

```python
import math  # 导入数学库
import torch  # 导入PyTorch
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.nn import functional as F  # 从PyTorch中导入函数模块

import commons  # 导入自定义的commons模块
import logging  # 导入日志模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class LayerNorm(nn.Module):  # 定义LayerNorm类，继承自nn.Module
    def __init__(self, channels, eps=1e-5):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.channels = channels  # 设置channels属性
        self.eps = eps  # 设置eps属性

        self.gamma = nn.Parameter(torch.ones(channels))  # 初始化gamma参数
        self.beta = nn.Parameter(torch.zeros(channels))  # 初始化beta参数

    def forward(self, x):  # 前向传播函数
        x = x.transpose(1, -1)  # 调整x的维度
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)  # 对x进行Layer Norm
        return x.transpose(1, -1)  # 调整x的维度


@torch.jit.script  # 使用Torch Script装饰器
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):  # 定义fused_add_tanh_sigmoid_multiply函数
    n_channels_int = n_channels[0]  # 获取n_channels的第一个元素
    in_act = input_a + input_b  # 计算input_a和input_b的和
    t_act = torch.tanh(in_act[:, :n_channels_int, :])  # 计算tanh激活函数
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])  # 计算sigmoid激活函数
    acts = t_act * s_act  # 计算t_act和s_act的乘积
    return acts  # 返回结果


class Encoder(nn.Module):  # 定义Encoder类，继承自nn.Module
    def __init__(  # 初始化函数
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=4,
        isflow=True,
        **kwargs
    ):
        super().__init__()  # 调用父类的初始化函数
        self.hidden_channels = hidden_channels  # 设置hidden_channels属性
        self.filter_channels = filter_channels  # 设置filter_channels属性
        self.n_heads = n_heads  # 设置n_heads属性
        self.n_layers = n_layers  # 设置n_layers属性
        self.kernel_size = kernel_size  # 设置kernel_size属性
        self.p_dropout = p_dropout  # 设置p_dropout属性
        self.window_size = window_size  # 设置window_size属性
        self.cond_layer_idx = self.n_layers  # 设置cond_layer_idx属性
        if "gin_channels" in kwargs:  # 如果gin_channels在kwargs中
            self.gin_channels = kwargs["gin_channels"]  # 设置gin_channels属性
            if self.gin_channels != 0:  # 如果gin_channels不为0
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)  # 初始化spk_emb_linear
                self.cond_layer_idx = (  # 设置cond_layer_idx属性
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                logging.debug(self.gin_channels, self.cond_layer_idx)  # 记录日志
                assert (  # 断言
                    self.cond_layer_idx < self.n_layers
                ), "cond_layer_idx should be less than n_layers"  # 如果不满足条件则抛出异常
        self.drop = nn.Dropout(p_dropout)  # 初始化Dropout层
        self.attn_layers = nn.ModuleList()  # 初始化注意力层列表
        self.norm_layers_1 = nn.ModuleList()  # 初始化LayerNorm层列表
        self.ffn_layers = nn.ModuleList()  # 初始化FeedForward层列表
        self.norm_layers_2 = nn.ModuleList()  # 初始化LayerNorm层列表
        for i in range(self.n_layers):  # 遍历n_layers
            self.attn_layers.append(  # 向attn_layers列表中添加元素
                MultiHeadAttention(  # 创建MultiHeadAttention实例
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))  # 向norm_layers_1列表中添加元素
            self.ffn_layers.append(  # 向ffn_layers列表中添加元素
                FFN(  # 创建FFN实例
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))  # 向norm_layers_2列表中添加元素

    def forward(self, x, x_mask, g=None):  # 前向传播函数
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)  # 计算注意力掩码
        x = x * x_mask  # 对x进行掩码
        for i in range(self.n_layers):  # 遍历n_layers
            if i == self.cond_layer_idx and g is not None:  # 如果i等于cond_layer_idx并且g不为None
                g = self.spk_emb_linear(g.transpose(1, 2))  # 计算g
                g = g.transpose(1, 2)  # 调整g的维度
                x = x + g  # 更新x
                x = x * x_mask  # 对x进行掩码
            y = self.attn_layers[i](x, x, attn_mask)  # 计算注意力层输出
            y = self.drop(y)  # Dropout
            x = self.norm_layers_1[i](x + y)  # LayerNorm
            y = self.ffn_layers[i](x, x_mask)  # 计算FeedForward层输出
            y = self.drop(y)  # Dropout
            x = self.norm_layers_2[i](x + y)  # LayerNorm
        x = x * x_mask  # 对x进行掩码
        return x  # 返回结果


class MultiHeadAttention(nn.Module):  # 定义MultiHeadAttention类，继承自nn.Module
    def __init__(  # 初始化函数
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        block_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()  # 调用父类的初始化函数
        assert channels % n_heads == 0  # 断言

        self.channels = channels  # 设置channels属性
        self.out_channels = out_channels  # 设置out_channels属性
        self.n_heads = n_heads  # 设置n_heads属性
        self.p_dropout = p_dropout  # 设置p_dropout属性
        self.window_size = window_size  # 设置window_size属性
        self.heads_share = heads_share  # 设置heads_share属性
        self.block_length = block_length  # 设置block_length属性
        self.proximal_bias = proximal_bias  # 设置proximal_bias属性
        self.proximal_init = proximal_init  # 设置proximal_init属性
        self.attn = None  # 初始化attn属性

        self.k_channels = channels // n_heads  # 计算k_channels
        self.conv_q = nn.Conv1d(channels, channels, 1)  # 初始化conv_q
        self.conv_k = nn.Conv1d(channels, channels, 1)  # 初始化conv_k
        self.conv_v = nn.Conv1d(channels, channels, 1)  # 初始化conv_v
        self.conv_o = nn.Conv1d(channels, out_channels, 1)  # 初始化conv_o
        self.drop = nn.Dropout(p_dropout)  # 初始化Dropout层

        if window_size is not None:  # 如果window_size不为None
            n_heads_rel = 1 if heads_share else n_heads  # 计算n_heads_rel
            rel_stddev = self.k_channels**-0.5  # 计算rel_stddev
            self.emb_rel_k = nn.Parameter(  # 初始化emb_rel_k
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(  # 初始化emb_rel_v
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)  # 初始化conv_q的权重
        nn.init.xavier_uniform_(self.conv_k.weight)  # 初始化conv_k的权重
        nn.init.xavier_uniform_(self.conv_v.weight)  # 初始化conv_v的权重
        if proximal_init:  # 如果proximal_init为True
            with torch.no_grad():  # 关闭梯度计算
                self.conv_k.weight.copy_(self.conv_q.weight)  # 复制conv_q的权重到conv_k
                self.conv_k.bias.copy_(self.conv_q.bias)  # 复制conv_q的偏置到conv_k

    def forward(self, x, c, attn_mask=None):  # 前向传播函数
        q = self.conv_q(x)  # 计算q
        k = self.conv_k(c)  # 计算k
        v = self.conv_v(c)  # 计算v

        x, self.attn = self.attention(q, k, v, mask=attn_mask)  # 计算注意力

        x = self.conv_o(x)  # 计算输出
        return x  # 返回结果

    def attention(self, query, key, value, mask=None):  # 定义attention函数
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))  # 获取维度信息
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)  # 调整query的维度
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 调整key的维度
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 调整value的维度

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))  # 计算得分
        if self.window_size is not None:  # 如果window_size不为None
            assert (  # 断言
                t_s == t_t
            ), "Relative attention is only available for self-attention."  # 如果不满足条件则抛出异常
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)  # 获取相对位置编码
            rel_logits = self._matmul_with_relative_keys(  # 计算相对位置编码
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)  # 计算相对位置编码
            scores = scores + scores_local  # 更新得分
        if self.proximal_bias:  # 如果proximal_bias为True
            assert (  # 断言
                t_s == t_t
            ), "Proximal bias is only available for self-attention."  # 如果不满足条件则抛出异常
            scores = scores + self._attention_bias_proximal(t_s).to(  # 添加近似偏置
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:  # 如果mask不为None
            scores = scores.masked_fill(mask == 0, -1e4)  # 对得分进行掩码
            if self.block_length is not None:  # 如果block_length不为None
                assert (  # 断言
                    t_s == t_t
                ), "Local attention is only available for self-attention."  # 如果不满足条件则抛出异常
                block_mask = (  # 创建块掩码
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)  # 对得分进行掩码
        p_attn = F.softmax(scores, dim=-1)  # 计算注意力权重
        p_attn = self.drop(p_attn)  # Dropout
        output = torch.matmul(p_attn, value)  # 计算输出
        if self.window_size is not None:  # 如果window_size不为None
            relative_weights = self._absolute_position_to_relative_position(p_attn)  # 计算相对权重
            value_relative_embeddings = self._get_relative_embeddings(  # 获取相对位置编码
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(  # 计算相对位置编码
                relative_weights, value_relative_embeddings
            )
        output = (  # 调整输出��维度
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn  # 返回结果

    def _matmul_with_relative_values(self, x, y):  # 定义_matmul_with_relative_values函数
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))  # 矩阵相乘
        return ret  # 返回结果

    def _matmul_with_relative_keys(self, x, y):  # 定义_matmul_with_relative_keys函数
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))  # 矩阵相乘
        return ret  # 返回结果

    def _get_relative_embeddings(self, relative_embeddings, length):  # 定义_get_relative_embeddings函数
        max_relative_position = 2 * self.window_size + 1  # 计算最大相对位置
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)  # 计算填充长度
        slice_start_position = max((self.window_size + 1) - length, 0)  # 计算切片起始位置
        slice_end_position = slice_start_position + 2 * length - 1  # 计算切片结束位置
        if pad_length > 0:  # 如果pad_length大于0
            padded_relative_embeddings = F.pad(  # 对relative_embeddings进行填充
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:  # 否则
            padded_relative_embeddings = relative_embeddings  # 不进行填充
        used_relative_embeddings = padded_relative_embeddings[  # 获取使用的相对位置编码
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings  # 返回结果

    def _relative_position_to_absolute_position(self, x):  # 定义_relative_position_to_absolute_position函数
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()  # 获取维度信息
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))  # 对x进行填充

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])  # 调整x的维度
        x_flat = F.pad(  # 对x_flat进行填充
            x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]  # 调整x_final的维度
        return x_final  # 返回结果

    def _absolute_position_to_relative_position(self, x):  # 定义_absolute_position_to_relative_position函数
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()  # 获取维度信息
        # padd along column
        x = F.pad
```

# `D:\src\Bert-VITS2\onnx_modules\V200\models_onnx.py`

```python
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
from . import attentions_onnx

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from .text import symbols, num_tones, num_languages


class DurationDiscriminator(nn.Module):  # vits2
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels  # input channels
        self.filter_channels = filter_channels  # filter channels
        self.kernel_size = kernel_size  # kernel size
        self.p_dropout = p_dropout  # dropout probability
        self.gin_channels = gin_channels  # gin channels

        self.drop = nn.Dropout(p_dropout)  # dropout layer
        self.conv_1 = nn.Conv1d(  # 1D convolutional layer
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)  # layer normalization
        self.conv_2 = nn.Conv1d(  # 1D convolutional layer
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)  # layer normalization
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)  # 1D convolutional layer

        self.pre_out_conv_1 = nn.Conv1d(  # 1D convolutional layer
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)  # layer normalization
        self.pre_out_conv_2 = nn.Conv1d(  # 1D convolutional layer
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)  # layer normalization

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)  # 1D convolutional layer

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())  # linear layer followed by sigmoid activation function

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)  # projection of duration
        x = torch.cat([x, dur], dim=1)  # concatenation of tensors
        x = self.pre_out_conv_1(x * x_mask)  # convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.pre_out_norm_1(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = self.pre_out_conv_2(x * x_mask)  # convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.pre_out_norm_2(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = x * x_mask  # element-wise multiplication
        x = x.transpose(1, 2)  # transpose
        output_prob = self.output_layer(x)  # output layer
        return output_prob  # return output probability

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)  # detach tensor
        if g is not None:
            g = torch.detach(g)  # detach tensor
            x = x + self.cond(g)  # addition
        x = self.conv_1(x * x_mask)  # convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.norm_1(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = self.conv_2(x * x_mask)  # convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.norm_2(x)  # layer normalization
        x = self.drop(x)  # dropout

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)  # forward probability
            output_probs.append(output_prob)  # append output probability

        return output_probs  # return output probabilities
```

# `D:\src\Bert-VITS2\onnx_modules\V200\__init__.py`

```python
from .text.symbols import symbols  # 从text模块中的symbols文件中导入symbols变量
from .models_onnx import SynthesizerTrn  # 从models_onnx模块中导入SynthesizerTrn类

__all__ = ["symbols", "SynthesizerTrn"]  # 定义__all__变量，包含symbols和SynthesizerTrn，用于模块导入时指定可导入的内容
```

# `D:\src\Bert-VITS2\onnx_modules\V200\text\bert_utils.py`

```python
from pathlib import Path  # 导入Path类

from huggingface_hub import hf_hub_download  # 从huggingface_hub模块中导入hf_hub_download函数

from config import config  # 从config模块中导入config对象


MIRROR: str = config.mirror  # 从config对象中获取mirror属性，并赋值给MIRROR变量


def _check_bert(repo_id, files, local_path):  # 定义_check_bert函数，接受repo_id、files、local_path三个参数
    for file in files:  # 遍历files列表
        if not Path(local_path).joinpath(file).exists():  # 如果local_path下的file文件不存在
            if MIRROR.lower() == "openi":  # 如果MIRROR的值转换为小写后等于"openi"
                import openi  # 导入openi模块

                openi.model.download_model(  # 调用openi.model.download_model函数
                    "Stardust_minus/Bert-VITS2", repo_id.split("/")[-1], "./bert"  # 传入三个参数
                )
            else:  # 如果MIRROR的值转换为小写后不等于"openi"
                hf_hub_download(  # 调用hf_hub_download函数
                    repo_id, file, local_dir=local_path, local_dir_use_symlinks=False  # 传入四个参数
                )
```

# `D:\src\Bert-VITS2\onnx_modules\V200\text\chinese.py`

```python
import os  # 导入os模块
import re  # 导入re模块
import cn2an  # 导入cn2an模块
from pypinyin import lazy_pinyin, Style  # 从pypinyin模块中导入lazy_pinyin和Style
from .symbols import punctuation  # 从symbols模块中导入punctuation
from .tone_sandhi import ToneSandhi  # 从tone_sandhi模块中导入ToneSandhi

current_file_path = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
pinyin_to_symbol_map = {  # 创建pinyin_to_symbol_map字典
    line.split("\t")[0]: line.strip().split("\t")[1]  # 以"\t"分割行并创建字典键值对
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()  # 读取文件内容并遍历行
}

import jieba.posseg as psg  # 导入jieba.posseg模块并重命名为psg

rep_map = {  # 创建rep_map字典
    "：": ",",  # 键值对
    "；": ",",  # 键值对
    # ... 其他键值对
}

tone_modifier = ToneSandhi()  # 创建ToneSandhi类的实例对象


def replace_punctuation(text):  # 定义函数replace_punctuation，参数为text
    # 函数实现内容

def g2p(text):  # 定义函数g2p，参数为text
    # 函数实现内容

def _get_initials_finals(word):  # 定义函数_get_initials_finals，参数为word
    # 函数实现内容

def _g2p(segments):  # 定义函数_g2p，参数为segments
    # 函数实现内容

def text_normalize(text):  # 定义函数text_normalize，参数为text
    # 函数实现内容

def get_bert_feature(text, word2ph):  # 定义函数get_bert_feature，参数为text和word2ph
    # 函数实现内容

if __name__ == "__main__":  # 如果当前模块是主模块
    from text.chinese_bert import get_bert_feature  # 从text.chinese_bert模块中导入get_bert_feature函数

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"  # 定义文本
    text = text_normalize(text)  # 对文本进行规范化处理
    print(text)  # 打印处理后的文本
    phones, tones, word2ph = g2p(text)  # 调用g2p函数处理文本
    bert = get_bert_feature(text, word2ph)  # 调用get_bert_feature函数获取bert特征
    print(phones, tones, word2ph, bert.shape)  # 打印phones、tones、word2ph和bert的形状
```

# `D:\src\Bert-VITS2\onnx_modules\V200\text\chinese_bert.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config类

LOCAL_PATH = "./bert/chinese-roberta-wwm-ext-large"  # 设置LOCAL_PATH变量为"./bert/chinese-roberta-wwm-ext-large"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用AutoTokenizer类从预训练模型路径LOCAL_PATH中实例化tokenizer对象

models = dict()  # 创建空字典models

def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):  # 定义函数get_bert_feature，接受text、word2ph和device三个参数
    if (  # 如果条件判断
        sys.platform == "darwin"  # 当前操作系统为darwin
        and torch.backends.mps.is_available()  # torch后端支持mps
        and device == "cpu"  # device为cpu
    ):  # 条件判断结束
        device = "mps"  # 将device设置为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 将device设置为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)  # 从预训练模型路径LOCAL_PATH中实例化AutoModelForMaskedLM类，并将其赋值给models[device]
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对text进行编码，返回张量
        for i in inputs:  # 遍历inputs
            inputs[i] = inputs[i].to(device)  # 将inputs[i]转移到device
        res = models[device](**inputs, output_hidden_states=True)  # 使用models[device]对inputs进行预测，返回结果
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将res["hidden_states"]的倒数第3到倒数第2个元素进行拼接，然后取第一个元素，最后转移到cpu

    assert len(word2ph) == len(text) + 2  # 断言word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 将res[i]重复word2phone[i]次，沿着第1维度
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 对phone_level_feature进行拼接，沿着第0维度

    return phone_level_feature.T  # 返回phone_level_feature的转置

if __name__ == "__main__":  # 如果模块是直接运行的
    word_level_feature = torch.rand(38, 1024)  # 创建一个38x1024的随机张量，赋值给word_level_feature
    word2phone = [  # 创建列表word2phone
        1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1
    ]

    total_frames = sum(word2phone)  # 计算word2phone列表中所有元素的和，赋值给total_frames
    print(word_level_feature.shape)  # 打印word_level_feature的形状
    print(word2phone)  # 打印word2phone列表
    phone_level_feature = []  # 创建空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        print(word_level_feature[i].shape)  # 打印word_level_feature[i]的形状

        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)  # 将word_level_feature[i]重复word2phone[i]次，沿着第1维度
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 对phone_level_feature进行拼接，沿着第0维度
    print(phone_level_feature.shape)  # 打印phone_level_feature的形状
```

# `D:\src\Bert-VITS2\onnx_modules\V200\text\cleaner.py`

```python
# 导入模块 chinese, japanese, english, cleaned_text_to_sequence
from . import chinese, japanese, english, cleaned_text_to_sequence

# 创建语言模块映射
language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}

# 定义函数 clean_text，用于清洗文本
def clean_text(text, language):
    # 获取对应语言的模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

# 定义函数 clean_text_bert，用于对文本进行 BERT 处理
def clean_text_bert(text, language):
    # 获取对应语言的模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 获取 BERT 特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert

# 定义函数 text_to_sequence，用于将文本转换为序列
def text_to_sequence(text, language):
    # 对文本进行清洗
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 将清洗后的文本转换为序列
    return cleaned_text_to_sequence(phones, tones, language)

# 主程序入口
if __name__ == "__main__":
    pass
```

# `D:\src\Bert-VITS2\onnx_modules\V200\text\english.py`

```python
import pickle  # 导入pickle模块
import os  # 导入os模块
import re  # 导入re模块
from g2p_en import G2p  # 从g2p_en模块导入G2p类

from . import symbols  # 从当前目录导入symbols模块

current_file_path = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")  # 拼接路径
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")  # 拼接路径
_g2p = G2p()  # 创建G2p对象

# arpa是一个包含音素的集合
arpa = {
    "AH0",
    "S",
    ...
    "ˈ",
}

# post_replace_ph函数用于替换音素
def post_replace_ph(ph):
    ...

# read_dict函数用于读取字典
def read_dict():
    ...

# cache_dict函数用于缓存字典
def cache_dict(g2p_dict, file_path):
    ...

# get_dict函数用于获取字典
def get_dict():
    ...

eng_dict = get_dict()  # 获取字典

# refine_ph函数用于处理音素
def refine_ph(phn):
    ...

# refine_syllables函数用于处理音节
def refine_syllables(syllables):
    ...

# normalize_numbers函数用于规范化数字
def normalize_numbers(text):
    ...

# text_normalize函数用于规范化文本
def text_normalize(text):
    ...

# g2p函数用于将文本转换为音素
def g2p(text):
    ...

# get_bert_feature函数用于获取bert特征
def get_bert_feature(text, word2ph):
    ...

if __name__ == "__main__":
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))  # 打印文本的音素表示
```

# `D:\src\Bert-VITS2\onnx_modules\V200\text\english_bert_mock.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import DebertaV2Model, DebertaV2Tokenizer  # 从transformers模块中导入DebertaV2Model和DebertaV2Tokenizer类

from config import config  # 从config模块中导入config类

LOCAL_PATH = "./bert/deberta-v3-large"  # 设置LOCAL_PATH变量为"./bert/deberta-v3-large"

tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)  # 使用DebertaV2Tokenizer类从预训练模型路径LOCAL_PATH中加载tokenizer

models = dict()  # 创建一个空字典models

def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):  # 定义一个名为get_bert_feature的函数，接受text、word2ph和device三个参数
    if (  # 如果条件判断
        sys.platform == "darwin"  # 当前操作系统为darwin
        and torch.backends.mps.is_available()  # 并且torch后端支持MPS
        and device == "cpu"  # 并且device为"cpu"
    ):  # 则执行下面的语句
        device = "mps"  # 将device设置为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 则将device设置为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = DebertaV2Model.from_pretrained(LOCAL_PATH).to(device)  # 则将models[device]设置为从预训练模型路径LOCAL_PATH中加载的DebertaV2Model，并将其移动到device上
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对text进行编码，返回PyTorch张量
        for i in inputs:  # 遍历inputs
            inputs[i] = inputs[i].to(device)  # 将inputs[i]移动到device上
        res = models[device](**inputs, output_hidden_states=True)  # 使用models[device]对inputs进行推理，输出隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将res的倒数第三个到倒数第二个隐藏状态拼接起来，然后取第一个张量，并将其移动到CPU上
    # assert len(word2ph) == len(text)+2  # 断言word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 将res[i]重复word2phone[i]次，沿着第一个维度重复1次
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 将phone_level_feature沿着第一个维度拼接起来

    return phone_level_feature.T  # 返回phone_level_feature的转置
```

# `D:\src\Bert-VITS2\onnx_modules\V200\text\japanese.py`

```python
# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re  # Import the regular expression module
import unicodedata  # Import the unicodedata module

from transformers import AutoTokenizer  # Import the AutoTokenizer class from the transformers module

from . import punctuation, symbols  # Import punctuation and symbols from the current package

from num2words import num2words  # Import the num2words function from the num2words module

import pyopenjtalk  # Import the pyopenjtalk module
import jaconv  # Import the jaconv module

# Define the function kata2phoneme with parameters text and return type str
def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    text = text.strip()  # Remove leading and trailing whitespaces from the text
    if text == "ー":  # Check if the text is equal to "ー"
        return ["ー"]  # Return a list containing "ー"
    elif text.startswith("ー"):  # Check if the text starts with "ー"
        return ["ー"] + kata2phoneme(text[1:])  # Return a list containing "ー" concatenated with the result of kata2phoneme function called with the remaining text
    res = []  # Initialize an empty list
    prev = None  # Initialize prev variable to None
    while text:  # Start a while loop with text as the condition
        if re.match(_MARKS, text):  # Check if the text matches the regular expression _MARKS
            res.append(text)  # Append the text to the res list
            text = text[1:]  # Update the text by removing the first character
            continue  # Continue to the next iteration of the loop
        if text.startswith("ー"):  # Check if the text starts with "ー"
            if prev:  # Check if prev is not None
                res.append(prev[-1])  # Append the last character of prev to the res list
            text = text[1:]  # Update the text by removing the first character
            continue  # Continue to the next iteration of the loop
        res += pyopenjtalk.g2p(text).lower().replace("cl", "q").split(" ")  # Convert the text to phonemes and append the result to the res list
        break  # Exit the loop
    return res  # Return the res list

# Define the function hira2kata with parameters text and return type str
def hira2kata(text: str) -> str:
    return jaconv.hira2kata(text)  # Return the result of converting hiragana to katakana using the jaconv module

# Define the regular expression pattern for marks
_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Define the function text2kata with parameters text and return type str
def text2kata(text: str) -> str:
    # ... (code continues)
```

# `D:\src\Bert-VITS2\onnx_modules\V200\text\japanese_bert.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config变量
from .japanese import text2sep_kata  # 从当前目录下的japanese模块中导入text2sep_kata函数

LOCAL_PATH = "./bert/deberta-v2-large-japanese"  # 定义LOCAL_PATH变量并赋值为"./bert/deberta-v2-large-japanese"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用AutoTokenizer类从预训练模型路径LOCAL_PATH中加载tokenizer

models = dict()  # 创建一个空的字典models

def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):  # 定义get_bert_feature函数，接受text、word2ph和device三个参数
    sep_text, _, _ = text2sep_kata(text)  # 调用text2sep_kata函数，将返回的结果分别赋值给sep_text、_和_
    sep_tokens = [tokenizer.tokenize(t) for t in sep_text]  # 使用tokenizer对sep_text中的每个文本进行分词
    sep_ids = [tokenizer.convert_tokens_to_ids(t) for t in sep_tokens]  # 将分词后的文本转换为对应的id
    sep_ids = [2] + [item for sublist in sep_ids for item in sublist] + [3]  # 对sep_ids进行处理
    return get_bert_feature_with_token(sep_ids, word2ph, device)  # 调用get_bert_feature_with_token函数，传入sep_ids、word2ph和device参数

def get_bert_feature_with_token(tokens, word2ph, device=config.bert_gen_config.device):  # 定义get_bert_feature_with_token函数，接受tokens、word2ph和device三个参数
    if (  # 如果条件判断
        sys.platform == "darwin"  # 当前操作系统为darwin
        and torch.backends.mps.is_available()  # torch后端支持mps
        and device == "cpu"  # device为cpu
    ):  # 条件判断结束
        device = "mps"  # 将device赋值为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 将device赋值为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)  # 使用AutoModelForMaskedLM类从预训练模型路径LOCAL_PATH中加载模型，并将其移动到device上
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = torch.tensor(tokens).to(device).unsqueeze(0)  # 将tokens转换为tensor，并移动到device上，然后在第0维度上增加一个维度
        token_type_ids = torch.zeros_like(inputs).to(device)  # 创建与inputs相同形状的全零tensor，并移动到device上
        attention_mask = torch.ones_like(inputs).to(device)  # 创建与inputs相同形状的全一tensor，并移动到device上
        inputs = {  # 创建inputs字典
            "input_ids": inputs,  # "input_ids"对应inputs
            "token_type_ids": token_type_ids,  # "token_type_ids"对应token_type_ids
            "attention_mask": attention_mask,  # "attention_mask"对应attention_mask
        }  # 字典创建结束

        # for i in inputs:
        #     inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)  # 使用models中对应device的模型对inputs进行预测，并获取隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 对隐藏状态进行处理
    assert inputs["input_ids"].shape[-1] == len(word2ph)  # 断言inputs["input_ids"]的最后一个维度长度等于word2ph的长度
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 对res[i]进行重复操作
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 对phone_level_feature进行拼接操作

    return phone_level_feature.T  # 返回phone_level_feature的转置
```

# `D:\src\Bert-VITS2\onnx_modules\V200\text\symbols.py`

```python
punctuation = ["!", "?", "…", ",", ".", "'", "-"]  # 创建一个包含标点符号的列表
pu_symbols = punctuation + ["SP", "UNK"]  # 创建一个包含标点符号和特殊标记的列表
pad = "_"  # 创建一个填充标记

# chinese
zh_symbols = [  # 创建一个包含中文音节的列表
    "E",
    "En",
    ...
    "OO",
]
num_zh_tones = 6  # 设置中文音节的音调数量为6

# japanese
ja_symbols = [  # 创建一个包含日文音节的列表
    "N",
    "a",
    ...
    "zy",
]
num_ja_tones = 2  # 设置日文音节的音调数量为2

# English
en_symbols = [  # 创建一个包含英文音素的列表
    "aa",
    "ae",
    ...
    "zh",
]
num_en_tones = 4  # 设置英文音素的音调数量为4

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))  # 合并所有音节并去重排序
symbols = [pad] + normal_symbols + pu_symbols  # 创建一个包含填充标记、所有音节和特殊标记的列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]  # 获取特殊标记在列表中的索引

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones  # 计算所有音节的总音调数量

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}  # 创建一个语言到ID的映射字典
num_languages = len(language_id_map.keys())  # 获取语言数量

language_tone_start_map = {  # 创建一个语言到音调起始位置的映射字典
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)  # 创建一个包含中文音节的集合
    b = set(en_symbols)  # 创建一个包含英文音素的集合
    print(sorted(a & b))  # 打印中英文共有的音节
```

# `D:\src\Bert-VITS2\onnx_modules\V200\text\tone_sandhi.py`

```python
# the meaning of jieba pos tag: https://blog.csdn.net/weixin_44174352/article/details/113731041
# e.g.
# word: "家里"
# pos: "s"
# finals: ['ia1', 'i3']
def _neural_sandhi(self, word: str, pos: str, finals: List[str]) -> List[str]:
    # reduplication words for n. and v. e.g. 奶奶, 试试, 旺旺
    for j, item in enumerate(word):
        if (
            j - 1 >= 0
            and item == word[j - 1]
            and pos[0] in {"n", "v", "a"}
            and word not in self.must_not_neural_tone_words
        ):
            finals[j] = finals[j][:-1] + "5"
    ge_idx = word.find("个")
    if len(word) >= 1 and word[-1] in "吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶":
        finals[-1] = finals[-1][:-1] + "5"
    elif len(word) >= 1 and word[-1] in "的地得":
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 走了, 看着, 去过
    # elif len(word) == 1 and word in "了着过" and pos in {"ul", "uz", "ug"}:
    #     finals[-1] = finals[-1][:-1] + "5"
    elif (
        len(word) > 1
        and word[-1] in "们子"
        and pos in {"r", "n"}
        and word not in self.must_not_neural_tone_words
    ):
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 桌上, 地下, 家里
    elif len(word) > 1 and word[-1] in "上下里" and pos in {"s", "l", "f"}:
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 上来, 下去
    elif len(word) > 1 and word[-1] in "来去" and word[-2] in "上下进出回过起开":
        finals[-1] = finals[-1][:-1] + "5"
    # 个做量词
    elif (
        ge_idx >= 1
        and (word[ge_idx - 1].isnumeric() or word[ge_idx - 1] in "几有两半多各整每做是")
    ) or word == "个":
        finals[ge_idx] = finals[ge_idx][:-1] + "5"
    else:
        if (
            word in self.must_neural_tone_words
            or word[-2:] in self.must_neural_tone_words
        ):
            finals[-1] = finals[-1][:-1] + "5"

    word_list = self._split_word(word)
    finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]]
    for i, word in enumerate(word_list):
        # conventional neural in Chinese
        if (
            word in self.must_neural_tone_words
            or word[-2:] in self.must_neural_tone_words
        ):
            finals_list[i][-1] = finals_list[i][-1][:-1] + "5"
    finals = sum(finals_list, [])
    return finals
```

# `D:\src\Bert-VITS2\onnx_modules\V200\text\__init__.py`

```python
from .symbols import *  # 从symbols模块中导入所有的变量和函数
```

# `D:\src\Bert-VITS2\onnx_modules\V200_OnnxInference\__init__.py`

```python
import numpy as np  # 导入numpy库
import onnxruntime as ort  # 导入onnxruntime库


def convert_pad_shape(pad_shape):  # 定义函数convert_pad_shape
    layer = pad_shape[::-1]  # 反转pad_shape
    pad_shape = [item for sublist in layer for item in sublist]  # 将layer展开成一维数组
    return pad_shape  # 返回pad_shape


def sequence_mask(length, max_length=None):  # 定义函数sequence_mask
    if max_length is None:  # 如果max_length为空
        max_length = length.max()  # max_length取length的最大值
    x = np.arange(max_length, dtype=length.dtype)  # 生成一个长度为max_length的数组x
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)  # 返回x和length的比较结果


def generate_path(duration, mask):  # 定义函数generate_path
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape  # 获取mask的形状
    cum_duration = np.cumsum(duration, -1)  # 对duration进行累加

    cum_duration_flat = cum_duration.reshape(b * t_x)  # 将cum_duration展开成一维数组
    path = sequence_mask(cum_duration_flat, t_y)  # 生成path
    path = path.reshape(b, t_x, t_y)  # 调整path的形状
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]  # 对path进行异或操作
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)  # 调整path的形状
    return path  # 返回path


class OnnxInferenceSession:  # 定义类OnnxInferenceSession
    def __init__(self, path, Providers=["CPUExecutionProvider"]):  # 定义初始化函数
        self.enc = ort.InferenceSession(path["enc"], providers=Providers)  # 初始化self.enc
        self.emb_g = ort.InferenceSession(path["emb_g"], providers=Providers)  # 初始化self.emb_g
        self.dp = ort.InferenceSession(path["dp"], providers=Providers)  # 初始化self.dp
        self.sdp = ort.InferenceSession(path["sdp"], providers=Providers)  # 初始化self.sdp
        self.flow = ort.InferenceSession(path["flow"], providers=Providers)  # 初始化self.flow
        self.dec = ort.InferenceSession(path["dec"], providers=Providers)  # 初始化self.dec

    def __call__(  # 定义__call__函数
        self,
        seq,
        tone,
        language,
        bert_zh,
        bert_jp,
        bert_en,
        sid,
        seed=114514,
        seq_noise_scale=0.8,
        sdp_noise_scale=0.6,
        length_scale=1.0,
        sdp_ratio=0.0,
    ):
        if seq.ndim == 1:  # 如果seq的维度为1
            seq = np.expand_dims(seq, 0)  # 将seq扩展为二维数组
        if tone.ndim == 1:  # 如果tone的维度为1
            tone = np.expand_dims(tone, 0)  # 将tone扩展为二维数组
        if language.ndim == 1:  # 如果language的维度为1
            language = np.expand_dims(language, 0)  # 将language扩展为二维数组
        assert (seq.ndim == 2, tone.ndim == 2, language.ndim == 2)  # 断言seq、tone、language的维度为2
        g = self.emb_g.run(  # 运行self.emb_g
            None,
            {
                "sid": sid.astype(np.int64),  # 传入参数sid
            },
        )[0]  # 获取返回值的第一个元素
        g = np.expand_dims(g, -1)  # 将g扩展为三维数组
        enc_rtn = self.enc.run(  # 运行self.enc
            None,
            {
                "x": seq.astype(np.int64),  # 传入参数seq
                "t": tone.astype(np.int64),  # 传入参数tone
                "language": language.astype(np.int64),  # 传入参数language
                "bert_0": bert_zh.astype(np.float32),  # 传入参数bert_zh
                "bert_1": bert_jp.astype(np.float32),  # 传入参数bert_jp
                "bert_2": bert_en.astype(np.float32),  # 传入参数bert_en
                "g": g.astype(np.float32),  # 传入参数g
            },
        )
        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]  # 获取enc_rtn的四个元素
        np.random.seed(seed)  # 设置随机数种子
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale  # 生成zinput
        logw = self.sdp.run(  # 运行self.sdp
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}
        )[0] * (sdp_ratio) + self.dp.run(None, {"x": x, "x_mask": x_mask, "g": g})[
            0
        ] * (
            1 - sdp_ratio
        )  # 计算logw
        w = np.exp(logw) * x_mask * length_scale  # 计算w
        w_ceil = np.ceil(w)  # 对w进行向上取整
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(
            np.int64
        )  # 计算y_lengths
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)  # 生成y_mask
        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)  # 生成attn_mask
        attn = generate_path(w_ceil, attn_mask)  # 生成attn
        m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # 计算m_p
        logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # 计算logs_p

        z_p = (
            m_p
            + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
            * np.exp(logs_p)
            * seq_noise_scale
        )  # 计算z_p

        z = self.flow.run(  # 运行self.flow
            None,
            {
                "z_p": z_p.astype(np.float32),  # 传入参数z_p
                "y_mask": y_mask.astype(np.float32),  # 传入参数y_mask
                "g": g,  # 传入参数g
            },
        )[0]  # 获取返回值的第一个元素

        return self.dec.run(None, {"z_in": z.astype(np.float32), "g": g})[0]  # 运行self.dec并返回结果
```

# `D:\src\Bert-VITS2\onnx_modules\V210\attentions_onnx.py`

```python
import math  # 导入数学库
import torch  # 导入PyTorch
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.nn import functional as F  # 从PyTorch中导入函数模块

import commons  # 导入自定义的commons模块
import logging  # 导入日志模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class LayerNorm(nn.Module):  # 定义LayerNorm类，继承自nn.Module
    def __init__(self, channels, eps=1e-5):  # 初始化函数，channels为通道数，eps为epsilon值
        super().__init__()  # 调用父类的初始化函数
        self.channels = channels  # 设置通道数
        self.eps = eps  # 设置epsilon值

        self.gamma = nn.Parameter(torch.ones(channels))  # 初始化gamma参数
        self.beta = nn.Parameter(torch.zeros(channels))  # 初始化beta参数

    def forward(self, x):  # 前向传播函数，x为输入
        x = x.transpose(1, -1)  # 转置x
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)  # 对x进行Layer Norm
        return x.transpose(1, -1)  # 返回转置后的x


@torch.jit.script  # 使用Torch Script装饰器
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):  # 定义函数fused_add_tanh_sigmoid_multiply
    n_channels_int = n_channels[0]  # 获取n_channels的第一个元素
    in_act = input_a + input_b  # 计算input_a和input_b的和
    t_act = torch.tanh(in_act[:, :n_channels_int, :])  # 计算tanh
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])  # 计算sigmoid
    acts = t_act * s_act  # 计算t_act和s_act的乘积
    return acts  # 返回结果


class Encoder(nn.Module):  # 定义Encoder类，继承自nn.Module
    def __init__(  # 初始化函数
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=4,
        isflow=True,
        **kwargs
    ):
        super().__init__()  # 调用父类的初始化函数
        self.hidden_channels = hidden_channels  # 设置隐藏通道数
        self.filter_channels = filter_channels  # 设置过滤通道数
        self.n_heads = n_heads  # 设置头数
        self.n_layers = n_layers  # 设置层数
        self.kernel_size = kernel_size  # 设置卷积核大小
        self.p_dropout = p_dropout  # 设置dropout概率
        self.window_size = window_size  # 设置窗口大小
        self.cond_layer_idx = self.n_layers  # 设置条件层索引为n_layers
        if "gin_channels" in kwargs:  # 如果gin_channels在kwargs中
            self.gin_channels = kwargs["gin_channels"]  # 设置gin_channels
            if self.gin_channels != 0:  # 如果gin_channels不为0
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)  # 初始化线性层
                self.cond_layer_idx = (  # 设置条件层索引
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                logging.debug(self.gin_channels, self.cond_layer_idx)  # 记录日志
                assert (  # 断言条件
                    self.cond_layer_idx < self.n_layers
                ), "cond_layer_idx should be less than n_layers"  # 如果条件不满足，抛出异常
        self.drop = nn.Dropout(p_dropout)  # 初始化Dropout层
        self.attn_layers = nn.ModuleList()  # 初始化注意力层列表
        self.norm_layers_1 = nn.ModuleList()  # 初始化规范化层1列表
        self.ffn_layers = nn.ModuleList()  # 初始化前馈神经网络层列表
        self.norm_layers_2 = nn.ModuleList()  # 初始化规范化层2列表
        for i in range(self.n_layers):  # 遍历层数
            self.attn_layers.append(  # 向注意力层列表中添加元素
                MultiHeadAttention(  # 创建多头注意力层
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))  # 向规范化层1列表中添加元素
            self.ffn_layers.append(  # 向前馈神经网络层列表中添加元素
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))  # 向规范化层2列表中添加元素

    def forward(self, x, x_mask, g=None):  # 前向传播函数，x为输入，x_mask为掩码，g为条件
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)  # 计算注意力掩码
        x = x * x_mask  # 对x进行掩码
        for i in range(self.n_layers):  # 遍历层数
            if i == self.cond_layer_idx and g is not None:  # 如果i等于条件层索引且g不为空
                g = self.spk_emb_linear(g.transpose(1, 2))  # 计算说话人嵌入
                g = g.transpose(1, 2)  # 转置g
                x = x + g  # 更新x
                x = x * x_mask  # 对x进行掩码
            y = self.attn_layers[i](x, x, attn_mask)  # 计算注意力层输出
            y = self.drop(y)  # 使用Dropout层
            x = self.norm_layers_1[i](x + y)  # 规范化
            y = self.ffn_layers[i](x, x_mask)  # 计算前馈神经网络层输出
            y = self.drop(y)  # 使用Dropout层
            x = self.norm_layers_2[i](x + y)  # 规范化
        x = x * x_mask  # 对x进行掩码
        return x  # 返回结果


class MultiHeadAttention(nn.Module):  # 定义MultiHeadAttention类，继承自nn.Module
    def __init__(  # 初始化函数
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        block_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()  # 调用父类的初始化函数
        assert channels % n_heads == 0  # 断言条件

        self.channels = channels  # 设置通道数
        self.out_channels = out_channels  # 设置输出通道数
        self.n_heads = n_heads  # 设置头数
        self.p_dropout = p_dropout  # 设置dropout概率
        self.window_size = window_size  # 设置窗口大小
        self.heads_share = heads_share  # 设置头共享
        self.block_length = block_length  # 设置块长度
        self.proximal_bias = proximal_bias  # 设置近端偏置
        self.proximal_init = proximal_init  # 设置近端初始化
        self.attn = None  # 初始化注意力

        self.k_channels = channels // n_heads  # 计算k通道数
        self.conv_q = nn.Conv1d(channels, channels, 1)  # 初始化卷积层q
        self.conv_k = nn.Conv1d(channels, channels, 1)  # 初始化卷积层k
        self.conv_v = nn.Conv1d(channels, channels, 1)  # 初始化卷积层v
        self.conv_o = nn.Conv1d(channels, out_channels, 1)  # 初始化卷积层o
        self.drop = nn.Dropout(p_dropout)  # 初始化Dropout层

        if window_size is not None:  # 如果窗口大小不为空
            n_heads_rel = 1 if heads_share else n_heads  # 计算相对头数
            rel_stddev = self.k_channels**-0.5  # 计算相对标准差
            self.emb_rel_k = nn.Parameter(  # 初始化相对嵌入k
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(  # 初始化相对嵌入v
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)  # 初始化卷积层q的权重
        nn.init.xavier_uniform_(self.conv_k.weight)  # 初始化卷积层k的权重
        nn.init.xavier_uniform_(self.conv_v.weight)  # 初始化卷积层v的权重
        if proximal_init:  # 如果近端初始化为真
            with torch.no_grad():  # 不计算梯度
                self.conv_k.weight.copy_(self.conv_q.weight)  # 复制卷积层q的权重到卷积层k
                self.conv_k.bias.copy_(self.conv_q.bias)  # 复制卷积层q的偏置到卷积层k

    def forward(self, x, c, attn_mask=None):  # 前向传播函数，x为输入，c为条件，attn_mask为注意力掩码
        q = self.conv_q(x)  # 计算卷积层q的输出
        k = self.conv_k(c)  # 计算卷积层k的输出
        v = self.conv_v(c)  # 计算卷积层v的输出

        x, self.attn = self.attention(q, k, v, mask=attn_mask)  # 计算注意力

        x = self.conv_o(x)  # 计算卷积层o的输出
        return x  # 返回结果

    def attention(self, query, key, value, mask=None):  # 定义注意力函数，query为查询，key为键，value为值，mask为掩码
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))  # 获取维度
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)  # 重塑query
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 重塑key
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 重塑value

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))  # 计算得分
        if self.window_size is not None:  # 如果窗口大小不为空
            assert (  # 断言条件
                t_s == t_t
            ), "Relative attention is only available for self-attention."  # 如果条件不满足，抛出异常
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)  # 获取相对嵌入
            rel_logits = self._matmul_with_relative_keys(  # 计算相对logits
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)  # 计算相对位置到绝对位置
            scores = scores + scores_local  # 更新得分
        if self.proximal_bias:  # 如果近端偏置为真
            assert t_s == t_t, "Proximal bias is only available for self-attention."  # 如果条件不满足，抛出异常
            scores = scores + self._attention_bias_proximal(t_s).to(  # 更新得分
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:  # 如果掩码不为空
            scores = scores.masked_fill(mask == 0, -1e4)  # 对得分进行掩码
            if self.block_length is not None:  # 如果块长度不为空
                assert (  # 断言条件
                    t_s == t_t
                ), "Local attention is only available for self-attention."  # 如果条件不满足，抛出异常
                block_mask = (  # 计算块掩码
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)  # 对得分进行块掩码
        p_attn = F.softmax(scores, dim=-1)  # 计算注意力权重
        p_attn = self.drop(p_attn)  # 使用Dropout层
        output = torch.matmul(p_attn, value)  # 计算输出
        if self.window_size is not None:  # 如果窗口大小不为空
            relative_weights = self._absolute_position_to_relative_position(p_attn)  # 计算相对权重
            value_relative_embeddings = self._get_relative_embeddings(  # 获取相对嵌入
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(  # 计算相对值
                relative_weights, value_relative_embeddings
            )
        output = (  # 重塑输出
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn  # 返回结果

    def _matmul_with_relative_values(self, x, y):  # 定义_matmul_with_relative_values函数
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))  # 矩阵相乘
        return ret  # 返回结果

    def _matmul_with_relative_keys(self, x, y):  # 定义_matmul_with_relative_keys函数
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))  # 矩阵相乘
        return ret  # 返回结果

    def _get_relative_embeddings(self, relative_embeddings, length):  # 定义_get_relative_embeddings函数
        max_relative_position = 2 * self.window_size + 1  # 计算最大相对位置
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)  # 计算填充长度
        slice_start_position = max((self.window_size + 1) - length, 0)  # 计算切片起始位置
        slice_end_position = slice_start_position + 2 * length - 1  # 计算切片结束位置
        if pad_length > 0:  # 如果填充长度大于0
            padded_relative_embeddings = F.pad(  # 对相对嵌入进行填充
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:  # 否则
            padded_relative_embeddings = relative_embeddings  # 不填充
        used_relative_embeddings = padded_relative_embeddings[  # 获取使用的相对嵌入
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings  # 返回结果

    def _relative_position_to_absolute_position(self, x):  # 定义_relative_position_to_absolute_position函数
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()  # 获取维度
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))  # 对x进行填充

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])  # 重塑x
        x_flat = F.pad(  # 对x进行填
```