# `Bert-VITS2\infer.py`

```

# 版本管理、兼容推理及模型加载实现
# 版本说明
# 1. 版本号与github的release版本号对应，使用哪个release版本训练的模型即对应其版本号
# 2. 请在模型的config.json中显示声明版本号，添加一个字段"version" : "你的版本号"
# 特殊版本说明
# 1.1.1-fix： 1.1.1版本训练的模型，但是在推理时使用dev的日语修复
# 2.3：当前版本

# 导入所需的库
import torch
import commons
from text import cleaned_text_to_sequence, get_bert
from typing import Union
from text.cleaner import clean_text
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from oldVersion.V220.models import SynthesizerTrn as V220SynthesizerTrn
from oldVersion.V220.text import symbols as V220symbols
from oldVersion.V210.models import SynthesizerTrn as V210SynthesizerTrn
from oldVersion.V210.text import symbols as V210symbols
from oldVersion.V200.models import SynthesizerTrn as V200SynthesizerTrn
from oldVersion.V200.text import symbols as V200symbols
from oldVersion.V111.models import SynthesizerTrn as V111SynthesizerTrn
from oldVersion.V111.text import symbols as V111symbols
from oldVersion.V110.models import SynthesizerTrn as V110SynthesizerTrn
from oldVersion.V110.text import symbols as V110symbols
from oldVersion.V101.models import SynthesizerTrn as V101SynthesizerTrn
from oldVersion.V101.text import symbols as V101symbols
from oldVersion import V111, V110, V101, V200, V210, V220

# 当前版本信息
latest_version = "2.3"

# 版本兼容
SynthesizerTrnMap = {
    "2.2": V220SynthesizerTrn,
    "2.1": V210SynthesizerTrn,
    "2.0.2-fix": V200SynthesizerTrn,
    "2.0.1": V200SynthesizerTrn,
    "2.0": V200SynthesizerTrn,
    "1.1.1-fix": V111SynthesizerTrn,
    "1.1.1": V111SynthesizerTrn,
    "1.1": V110SynthesizerTrn,
    "1.1.0": V110SynthesizerTrn,
    "1.0.1": V101SynthesizerTrn,
    "1.0": V101SynthesizerTrn,
    "1.0.0": V101SynthesizerTrn,
}

symbolsMap = {
    "2.2": V220symbols,
    "2.1": V210symbols,
    "2.0.2-fix": V200symbols,
    "2.0.1": V200symbols,
    "2.0": V200symbols,
    "1.1.1-fix": V111symbols,
    "1.1.1": V111symbols,
    "1.1": V110symbols,
    "1.1.0": V110symbols,
    "1.0.1": V101symbols,
    "1.0": V101symbols,
    "1.0.0": V101symbols,
}

# 定义函数，根据模型路径、版本号、设备和超参数获取模型
def get_net_g(model_path: str, version: str, device: str, hps):
    if version != latest_version:
        net_g = SynthesizerTrnMap[version](
            len(symbolsMap[version]),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    else:
        # 当前版本模型 net_g
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    return net_g

# 定义函数，根据文本、语言、超参数、设备和样式文本获取文本特征
def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 省略部分代码...

# 定义函数，根据文本、情感、音频等参数进行推理
def infer(
    text,
    emotion: Union[int, str],
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    language,
    hps,
    net_g,
    device,
    reference_audio=None,
    skip_start=False,
    skip_end=False,
    style_text=None,
    style_weight=0.7,
def infer_multilang(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    language,
    hps,
    net_g,
    device,
    reference_audio=None,
    emotion=None,
    skip_start=False,
    skip_end=False,
):
    # 省略部分代码...

```