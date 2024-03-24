# `.\lucidrains\naturalspeech2-pytorch\naturalspeech2_pytorch\utils\expand\number_norm.py`

```py
import re
import inflect
from num2words import num2words
from num_to_words import num_to_word

# 创建一个数字标准化类
class NumberNormalizer:
    def __init__(self):
        # 初始化 inflect 引擎
        self._inflect = inflect.engine()
        # 编译正则表达式，用于匹配数字
        self._number_re = re.compile(r"-?[0-9]+")
        # 编译正则表达式，用于匹配货币
        self._currency_re = re.compile(r"([$€£¥₹])([0-9\,\.]*[0-9]+)")
        # 存储货币转换率的字典
        self._currencies = {}

    # 添加货币转换率
    def add_currency(self, symbol, conversion_rates):
        self._currencies[symbol] = conversion_rates

    # 标准化文本中的数字
    def normalize_numbers(self, text, language='en'):
        self._inflect = inflect.engine()
        self._set_language(language)
        # 替换文本中的货币
        text = re.sub(self._currency_re, self._expand_currency, text)
        # 替换文本中的数字
        text = re.sub(self._number_re, lambda match: self._expand_number(match, language), text)
        return text

    # 设置语言
    def _set_language(self, language):
        if language == 'en':
            self._inflect = inflect.engine()
        else:
            self._inflect = inflect.engine()
            # 在这里添加对其他语言的支持

    # 扩展货币
    def _expand_currency(self, match):
        unit = match.group(1)
        currency = self._currencies.get(unit)
        if currency:
            value = match.group(2)
            return self._expand_currency_value(value, currency)
        return match.group(0)

    # 扩展货币值
    def _expand_currency_value(self, value, inflection):
        parts = value.replace(",", "").split(".")
        if len(parts) > 2:
            return f"{value} {inflection[2]}"  # 意外的格式
        text = []
        integer = int(parts[0]) if parts[0] else 0
        if integer > 0:
            integer_unit = inflection.get(integer, inflection[2])
            text.append(f"{integer} {integer_unit}")
        fraction = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if fraction > 0:
            fraction_unit = inflection.get(fraction / 100, inflection[0.02])
            text.append(f"{fraction} {fraction_unit}")
        if not text:
            return f"zero {inflection[2]}"
        return " ".join(text)

    # 扩展数字
    def _expand_number(self, match, language: str) -> str:
        num = int(match.group(0))
        if 1000 < num < 3000:
            if num == 2000:
                return self._number_to_words(num, language)
            if 2000 < num < 2010:
                return f"{self._number_to_words(2000, language)} {self._number_to_words(num % 100, language)}"
            if num % 100 == 0:
                return f"{self._number_to_words(num // 100, language)} {self._get_word('hundred')}"
            return self._number_to_words(num, language)
        return self._number_to_words(num, language)

    # 将数字转换为单词
    def _number_to_words(self, n: int, language: str) -> str:
        try:
            if language == 'en':
                return self._inflect.number_to_words(n)
            else:
                return num2words(n, lang=language)
        except:
            try:
                return num_to_word(n, lang=language)
            except:
                raise NotImplementedError("language not implemented")

    # 获取单词
    def _get_word(self, word):
        return word

# 如果作为主程序运行
if __name__ == "__main__":
    # 创建 NumberNormalizer 的实例
    normalizer = NumberNormalizer()
    # 添加货币转换率
    symbol = '$'
    conversion_rates ={
            0.01: "cent",
            0.02: "cents",
            1: "dollar",
            2: "dollars",
        }
    normalizer.add_currency(symbol, conversion_rates)
    # 示例 1：英语（en）语言
    text_en = "I have $1,000 and 5 apples."
    normalized_text_en = normalizer.normalize_numbers(text_en, language='en')
    print(normalized_text_en)
    # 输出: "I have one thousand dollars and five apples."

    # 示例 2：西班牙语（es）语言
    text_es = "Tengo $1.000 y 5 manzanas."
    normalized_text_es = normalizer.normalize_numbers(text_es, language='es')
    print(normalized_text_es)
    # 输出: "Tengo mil dólares y cinco manzanas."
```