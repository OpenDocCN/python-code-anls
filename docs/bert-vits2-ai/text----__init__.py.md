# `d:/src/tocomm/Bert-VITS2\text\__init__.py`

```
# 导入 symbols 模块中的所有内容
from text.symbols import *

# 创建一个字典，将 symbols 列表中的每个符号与其索引对应起来
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

# 定义函数 cleaned_text_to_sequence，将文本转换为对应符号的 ID 序列
def cleaned_text_to_sequence(cleaned_text, tones, language):
    """将文本字符串转换为与文本中符号对应的 ID 序列。
    参数:
      cleaned_text: 要转换为序列的字符串
      tones: 文本的音调
      language: 文本的语言
    返回:
      与文本中符号对应的整数列表
    """
    # 将 cleaned_text 中的每个符号转换为对应的 ID，并存储在 phones 列表中
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    # 根据语言的音调起始位置，将 tones 列表中的每个音调值进行偏移
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    # 根据语言的 ID，创建与 phones 列表长度相同的 lang_ids 列表
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    # 返回 phones、tones 和 lang_ids 列表
    return phones, tones, lang_ids
def get_bert(norm_text, word2ph, language, device, style_text=None, style_weight=0.7):
    # 导入中文、英文和日文的BERT特征提取函数
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    # 创建语言到BERT特征提取函数的映射
    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # 根据语言选择对应的BERT特征提取函数，并调用该函数
    bert = lang_bert_func_map[language](
        norm_text, word2ph, device, style_text, style_weight
    )
    # 返回BERT特征
    return bert


def check_bert_models():
    # 导入json和Path模块
    import json
    from pathlib import Path

    # 导入config模块中的配置信息和bert_utils模块中的_check_bert函数
    from config import config
    from .bert_utils import _check_bert

    # 检查配置中的镜像是否为openi
    if config.mirror.lower() == "openi":
        import openi  # 导入 openi 模块

        kwargs = {"token": config.openi_token} if config.openi_token else {}  # 如果存在 openi_token，则创建包含 token 的字典，否则创建空字典
        openi.login(**kwargs)  # 使用关键字参数调用 openi 模块的 login 函数

    with open("./bert/bert_models.json", "r") as fp:  # 打开 bert_models.json 文件进行读取
        models = json.load(fp)  # 从文件中加载 JSON 数据并存储在 models 变量中
        for k, v in models.items():  # 遍历 models 字典的键值对
            local_path = Path("./bert").joinpath(k)  # 创建本地路径
            _check_bert(v["repo_id"], v["files"], local_path)  # 调用 _check_bert 函数并传入参数

def init_openjtalk():  # 定义 init_openjtalk 函数
    import platform  # 导入 platform 模块

    if platform.platform() == "Linux":  # 如果操作系统是 Linux
        import pyopenjtalk  # 导入 pyopenjtalk 模块

        pyopenjtalk.g2p("こんにちは，世界。")  # 调用 pyopenjtalk 模块的 g2p 函数并传入参数
# 初始化 OpenJTalk
init_openjtalk()
# 检查 BERT 模型
check_bert_models()
```