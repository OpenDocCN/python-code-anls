# `d:/src/tocomm/Bert-VITS2\oldVersion\V111\text\__init__.py`

```
# 导入 symbols 模块中的所有内容
from .symbols import *

# 创建一个字典，将 symbols 列表中的元素作为键，对应的索引作为值
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

# 将清理后的文本转换为与文本中的符号对应的 ID 序列
def cleaned_text_to_sequence(cleaned_text, tones, language):
    """将文本字符串转换为与文本中符号对应的 ID 序列。
    参数:
      cleaned_text: 要转换为序列的字符串
      tones: 文本中的音调
      language: 文本所属的语言
    返回:
      与文本中符号对应的整数列表
    """
    # 将清理后的文本中的每个符号转换为对应的 ID，并存储在 phones 列表中
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    # 根据语言和音调的映射关系，将音调转换为对应的 ID，并存储在 tones 列表中
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    # 根据语言的映射关系，将语言转换为对应的 ID，并存储在 lang_ids 列表中
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    # 返回符号对应的 ID 序列、音调的 ID 序列和语言的 ID 序列
    return phones, tones, lang_ids
# 定义一个函数，用于获取 BERT 特征
def get_bert(norm_text, word2ph, language, device):
    # 从 chinese_bert 模块中导入 get_bert_feature 函数并重命名为 zh_bert
    from .chinese_bert import get_bert_feature as zh_bert
    # 从 english_bert_mock 模块中导入 get_bert_feature 函数并重命名为 en_bert
    from .english_bert_mock import get_bert_feature as en_bert
    # 从 japanese_bert 模块中导入 get_bert_feature 函数并重命名为 jp_bert
    from .japanese_bert import get_bert_feature as jp_bert

    # 创建一个字典，将语言和对应的 BERT 特征提取函数进行映射
    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # 根据给定的语言从映射中获取对应的 BERT 特征提取函数，并调用该函数获取特征
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    # 返回获取的 BERT 特征
    return bert


# 定义一个函数，用于获取修正后的 BERT 特征
def get_bert_fix(norm_text, word2ph, language, device):
    # 从 chinese_bert 模块中导入 get_bert_feature 函数并重命名为 zh_bert
    from .chinese_bert import get_bert_feature as zh_bert
    # 从 english_bert_mock 模块中导入 get_bert_feature 函数并重命名为 en_bert
    from .english_bert_mock import get_bert_feature as en_bert
    # 从 fix.japanese_bert 模块中导入 get_bert_feature 函数并重命名为 jp_bert
    from .fix.japanese_bert import get_bert_feature as jp_bert

    # 创建一个字典，将语言和对应的 BERT 特征提取函数进行映射
    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # 根据给定的语言从映射中获取对应的 BERT 特征提取函数，并调用该函数获取特征
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    # 返回获取的 BERT 特征
    return bert
```