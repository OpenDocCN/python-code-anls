# `.\lucidrains\naturalspeech2-pytorch\naturalspeech2_pytorch\utils\cleaner.py`

```py
import re
from pathlib import Path
from naturalspeech2_pytorch.utils.expand.abbreviations import AbbreviationExpander
from naturalspeech2_pytorch.utils.expand.number_norm import NumberNormalizer
from naturalspeech2_pytorch.utils.expand.time_norm import TimeExpander

CURRENT_DIR = Path(__file__).resolve().parent

class TextProcessor:
    def __init__(self, lang="en"):
        self.lang = lang
        self._whitespace_re = re.compile(r"\s+")
        # 实例化缩写展开器对象
        self.ab_expander = AbbreviationExpander(str(CURRENT_DIR / 'expand/abbreviations.csv'))
        # 实例化时间展开器对象
        self.time_expander = TimeExpander()
        # 实例化数字归一化器对象
        self.num_normalizer = NumberNormalizer()
        # 添加货币转换率
        symbol = '$'
        conversion_rates ={0.01: "cent", 0.02: "cents", 1: "dollar", 2: "dollars" }
        self.num_normalizer.add_currency(symbol, conversion_rates)
    def lowercase(self, text):
        return text.lower()

    def collapse_whitespace(self, text):
        return re.sub(self._whitespace_re, " ", text).strip()

    def remove_aux_symbols(self, text):
        # 移除文本中的辅助符号
        text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
        return text

    def phoneme_cleaners(self, text, language = 'en'):
        # 展开时间表达式
        text = self.time_expander.expand_time(text, language=language)
        # 归一化数字
        text = self.num_normalizer.normalize_numbers(text, language=language)
        # 替换文本中的缩写
        text = self.ab_expander.replace_text_abbreviations(text, language=language)
        # 移除辅助符号
        text = self.remove_aux_symbols(text)
        # 合并多余空格
        text = self.collapse_whitespace(text)
        return text

if __name__ == "__main__":
    # 创建英语实例
    text_processor_en = TextProcessor(lang="en")

    # 处理英语文本
    english_text = "Hello, Mr. Example, this is 9:30 am and  my number is 30."
    processed_english_text = text_processor_en.phoneme_cleaners(english_text, language='en')
    print(processed_english_text)

    # 创建西班牙语实例
    text_processor_es = TextProcessor(lang="es")

    # 处理西班牙语文本
    spanish_text = "Hola, Sr. Ejemplo, son las 9:30 am y mi número es el 30."
    processed_spanish_text = text_processor_es.phoneme_cleaners(spanish_text, language='es')
    print(processed_spanish_text)
```