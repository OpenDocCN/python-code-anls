# `Bert-VITS2\text\__init__.py`

```

# 导入符号列表
from text.symbols import *

# 创建符号到ID的映射字典
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

# 将清理后的文本转换为符号ID序列的函数
def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    # 将符号转换为对应的ID
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    # 根据语言和音调映射表将音调转换为对应的值
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    # 获取语言ID
    lang_id = language_id_map[language]
    # 为每个符号添加对应的语言ID
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids

# 获取BERT特征的函数
def get_bert(norm_text, word2ph, language, device, style_text=None, style_weight=0.7):
    # 根据语言选择对应的BERT特征获取函数
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # 调用对应语言的BERT特征获取函数
    bert = lang_bert_func_map[language](
        norm_text, word2ph, device, style_text, style_weight
    )
    return bert

# 检查BERT模型是否存在并下载
def check_bert_models():
    import json
    from pathlib import Path

    from config import config
    from .bert_utils import _check_bert

    # 如果使用openi镜像，则登录
    if config.mirror.lower() == "openi":
        import openi

        kwargs = {"token": config.openi_token} if config.openi_token else {}
        openi.login(**kwargs)

    # 读取BERT模型信息并检查是否存在并下载
    with open("./bert/bert_models.json", "r") as fp:
        models = json.load(fp)
        for k, v in models.items():
            local_path = Path("./bert").joinpath(k)
            _check_bert(v["repo_id"], v["files"], local_path)

# 初始化openjtalk
def init_openjtalk():
    import platform

    # 如果是Linux系统，则导入pyopenjtalk并进行文本转换
    if platform.platform() == "Linux":
        import pyopenjtalk

        pyopenjtalk.g2p("こんにちは，世界。")

# 初始化openjtalk
init_openjtalk()
# 检查BERT模型是否存在并下载
check_bert_models()

```