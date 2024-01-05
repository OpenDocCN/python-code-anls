# `d:/src/tocomm/Bert-VITS2\oldVersion\V200\text\__init__.py`

```
# 从 symbols 模块中导入 symbols 列表
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
    # 根据语言和音调的映射关系，将音调转换为对应的值
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    # 根据语言的映射关系，将语言转换为对应的 ID
    lang_id = language_id_map[language]
    # 创建与 phones 列表长度相同的 lang_ids 列表，存储语言对应的 ID
    lang_ids = [lang_id for i in phones]
    # 返回 phones、tones 和 lang_ids 列表
    return phones, tones, lang_ids
def get_bert(norm_text, word2ph, language, device):
    # 导入中文BERT模型的获取函数
    from .chinese_bert import get_bert_feature as zh_bert
    # 导入英文BERT模型的获取函数
    from .english_bert_mock import get_bert_feature as en_bert
    # 导入日文BERT模型的获取函数
    from .japanese_bert import get_bert_feature as jp_bert

    # 创建语言到BERT模型获取函数的映射
    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # 使用给定语言对应的BERT模型获取函数获取BERT特征
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    # 返回获取的BERT特征
    return bert


def check_bert_models():
    # 导入json模块
    import json
    # 导入Path类
    from pathlib import Path
    # 从config模块中导入config对象
    from config import config
    # 导入检查BERT模型的工具函数
    from .bert_utils import _check_bert

    # 如果使用的镜像为openi
    if config.mirror.lower() == "openi":
        # 导入openi模块
        import openi
        kwargs = {"token": config.openi_token} if config.openi_token else {}  # 如果存在 openi_token，则创建包含 token 的字典，否则创建空字典
        openi.login(**kwargs)  # 调用 openi.login 方法，传入 kwargs 字典作为关键字参数

    with open("./bert/bert_models.json", "r") as fp:  # 打开文件 "./bert/bert_models.json" 以只读模式，并使用 fp 作为文件对象
        models = json.load(fp)  # 从文件对象 fp 中加载 JSON 数据并存储在 models 变量中
        for k, v in models.items():  # 遍历 models 字典的键值对
            local_path = Path("./bert").joinpath(k)  # 创建本地路径，使用 "./bert" 作为基础路径，并添加键 k
            _check_bert(v["repo_id"], v["files"], local_path)  # 调用 _check_bert 方法，传入 v["repo_id"]、v["files"] 和 local_path 作为参数
```