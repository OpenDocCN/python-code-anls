# `Bert-VITS2\oldVersion\V110\text\__init__.py`

```
# 导入 symbols 模块中的所有内容
from .symbols import *

# 创建一个字典，将 symbols 列表中的元素作为键，对应的索引作为值
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

# 将清理后的文本转换为与文本中的符号对应的 ID 序列
def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    # 将清理后的文本中的每个符号转换为对应的 ID
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    # 根据语言和音调的映射关系，将音调转换为对应的值
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    # 根据语言映射关系，将语言转换为对应的 ID
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    # 返回符号 ID 序列、音调序列和语言 ID 序列
    return phones, tones, lang_ids

# 根据文本、word2ph、语言和设备获取 BERT 特征
def get_bert(norm_text, word2ph, language, device):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    # 创建一个语言到对应 BERT 特征获取函数的映射关系
    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # 根据语言获取对应的 BERT 特征
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    # 返回获取的 BERT 特征
    return bert
```