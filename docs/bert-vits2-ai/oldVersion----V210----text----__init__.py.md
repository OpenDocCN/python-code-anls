# `Bert-VITS2\oldVersion\V210\text\__init__.py`

```

# 导入 symbols 模块中的所有内容
from .symbols import *

# 创建一个字典，将 symbols 列表中的元素作为键，对应的索引作为值
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

# 将清理后的文本转换为符号序列的函数
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
    return phones, tones, lang_ids

# 获取 BERT 模型的函数
def get_bert(norm_text, word2ph, language, device, style_text, style_weight):
    # 根据语言选择对应的 BERT 模型
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # 调用对应语言的 BERT 模型函数
    bert = lang_bert_func_map[language](
        norm_text, word2ph, device, style_text, style_weight
    )
    return bert

# 检查 BERT 模型是否存在并下载
def check_bert_models():
    import json
    from pathlib import Path
    from config import config
    from .bert_utils import _check_bert

    # 如果配置为 openi 镜像，则导入 openi 模块并登录
    if config.mirror.lower() == "openi":
        import openi
        kwargs = {"token": config.openi_token} if config.openi_token else {}
        openi.login(**kwargs)

    # 从 bert_models.json 文件中读取模型信息，并检查模型是否存在并下载
    with open("./bert/bert_models.json", "r") as fp:
        models = json.load(fp)
        for k, v in models.items():
            local_path = Path("./bert").joinpath(k)
            _check_bert(v["repo_id"], v["files"], local_path)

# 调用检查 BERT 模型的函数
check_bert_models()

```