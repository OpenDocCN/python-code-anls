# `d:/src/tocomm/Bert-VITS2\oldVersion\V110\text\__init__.py`

```
from .symbols import *  # 导入 symbols 模块中的所有内容


_symbol_to_id = {s: i for i, s in enumerate(symbols)}  # 创建一个字典，将 symbols 列表中的每个元素作为键，对应的索引作为值


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """将一个文本字符串转换为与文本中的符号相对应的 ID 序列。
    参数:
      cleaned_text: 要转换为序列的字符串
      tones: 音调列表
      language: 语言
    返回:
      与文本中的符号相对应的整数列表
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]  # 将 cleaned_text 中的每个符号转换为对应的 ID，并存储在 phones 列表中
    tone_start = language_tone_start_map[language]  # 获取指定语言的音调起始索引
    tones = [i + tone_start for i in tones]  # 将 tones 列表中的每个元素加上音调起始索引，得到新的 tones 列表
    lang_id = language_id_map[language]  # 获取指定语言的 ID
    lang_ids = [lang_id for i in phones]  # 将 phones 列表中的每个元素都设置为指定语言的 ID，得到新的 lang_ids 列表
    return phones, tones, lang_ids  # 返回 phones、tones 和 lang_ids 列表作为结果
# 根据输入的文本、音素映射、语言和设备获取 BERT 特征
def get_bert(norm_text, word2ph, language, device):
    # 导入中文 BERT 特征提取函数
    from .chinese_bert import get_bert_feature as zh_bert
    # 导入英文 BERT 特征提取函数
    from .english_bert_mock import get_bert_feature as en_bert
    # 导入日文 BERT 特征提取函数
    from .japanese_bert import get_bert_feature as jp_bert

    # 创建语言到 BERT 特征提取函数的映射
    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # 根据语言选择对应的 BERT 特征提取函数，并传入文本、音素映射和设备进行特征提取
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    # 返回 BERT 特征
    return bert
```