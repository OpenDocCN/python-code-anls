# `.\transformers\models\clvp\number_normalizer.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，禁止未经许可使用该文件
# 可以在以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

"""CLVP 的英语规范化类"""

# 导入正则表达式模块
import re

# 定义英语规范化类
class EnglishNormalizer:
    def __init__(self):
        # 缩写的 (正则表达式, 替换) 对列表
        self._abbreviations = [
            # 针对不同缩写的正则表达式和替换
            (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
            for x in [
                ("mrs", "misess"),
                ("mr", "mister"),
                ("dr", "doctor"),
                ("st", "saint"),
                ("co", "company"),
                ("jr", "junior"),
                ("maj", "major"),
                ("gen", "general"),
                ("drs", "doctors"),
                ("rev", "reverend"),
                ("lt", "lieutenant"),
                ("hon", "honorable"),
                ("sgt", "sergeant"),
                ("capt", "captain"),
                ("esq", "esquire"),
                ("ltd", "limited"),
                ("col", "colonel"),
                ("ft", "fort"),
            ]
        ]

        # 数字的英文表示
        self.ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.teens = [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        self.tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    def number_to_words(self, num: int) -> str:
        """
        Converts numbers(`int`) to words(`str`).

        Please note that it only supports upto - "'nine hundred ninety-nine quadrillion, nine hundred ninety-nine
        trillion, nine hundred ninety-nine billion, nine hundred ninety-nine million, nine hundred ninety-nine
        thousand, nine hundred ninety-nine'" or `number_to_words(999_999_999_999_999_999)`.
        """
        # 如果数字为0，返回"zero"
        if num == 0:
            return "zero"
        # 如果数字小于0，返回"minus " + 绝对值的数字的英文表示
        elif num < 0:
            return "minus " + self.number_to_words(abs(num))
        # 如果数字小于10，返回对应的数字英文表示
        elif num < 10:
            return self.ones[num]
        # 如果数字小于20，返回对应的数字英文表示
        elif num < 20:
            return self.teens[num - 10]
        # 如果数字小于100，返回对应的数字英文表示
        elif num < 100:
            return self.tens[num // 10] + ("-" + self.number_to_words(num % 10) if num % 10 != 0 else "")
        # 如果数字小于1000，返回对应的数字英文表示
        elif num < 1000:
            return (
                self.ones[num // 100] + " hundred" + (" " + self.number_to_words(num % 100) if num % 100 != 0 else "")
            )
        # 如果数字小于1,000,000，返回对应的数字英文表示
        elif num < 1_000_000:
            return (
                self.number_to_words(num // 1000)
                + " thousand"
                + (", " + self.number_to_words(num % 1000) if num % 1000 != 0 else "")
            )
        # 如果数字小于1,000,000,000，返回对应的数字英文表示
        elif num < 1_000_000_000:
            return (
                self.number_to_words(num // 1_000_000)
                + " million"
                + (", " + self.number_to_words(num % 1_000_000) if num % 1_000_000 != 0 else "")
            )
        # 如果数字小于1,000,000,000,000，返回对应的数字英文表示
        elif num < 1_000_000_000_000:
            return (
                self.number_to_words(num // 1_000_000_000)
                + " billion"
                + (", " + self.number_to_words(num % 1_000_000_000) if num % 1_000_000_000 != 0 else "")
            )
        # 如果数字小于1,000,000,000,000,000，返回对应的数字英文表示
        elif num < 1_000_000_000_000_000:
            return (
                self.number_to_words(num // 1_000_000_000_000)
                + " trillion"
                + (", " + self.number_to_words(num % 1_000_000_000_000) if num % 1_000_000_000_000 != 0 else "")
            )
        # 如果数字小于1,000,000,000,000,000,000，返回对应的数字英文表示
        elif num < 1_000_000_000_000_000_000:
            return (
                self.number_to_words(num // 1_000_000_000_000_000)
                + " quadrillion"
                + (
                    ", " + self.number_to_words(num % 1_000_000_000_000_000)
                    if num % 1_000_000_000_000_000 != 0
                    else ""
                )
            )
        # 如果数字超出范围，返回"number out of range"
        else:
            return "number out of range"

    def convert_to_ascii(self, text: str) -> str:
        """
        Converts unicode to ascii
        """
        # 将文本转换为 ASCII 编码，忽略非 ASCII 字符，然后再解码为 UTF-8 格式
        return text.encode("ascii", "ignore").decode("utf-8")
    def _expand_dollars(self, m: str) -> str:
        """
        This method is used to expand numerical dollar values into spoken words.
        """
        # 匹配到的数字字符串
        match = m.group(1)
        # 将匹配到的字符串以小数点为分隔符分成两部分
        parts = match.split(".")
        # 如果小数点分割后的部分数大于2，则格式不符合预期，直接返回原字符串加上" dollars"
        if len(parts) > 2:
            return match + " dollars"  # Unexpected format

        # 提取美元部分和美分部分的整数值
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        # 如果美元和美分都存在
        if dollars and cents:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            cent_unit = "cent" if cents == 1 else "cents"
            # 返回美元和美分的文字表示
            return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
        # 如果只有美元
        elif dollars:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            # 返回美元的文字表示
            return "%s %s" % (dollars, dollar_unit)
        # 如果只有美分
        elif cents:
            cent_unit = "cent" if cents == 1 else "cents"
            # 返回美分的文字表示
            return "%s %s" % (cents, cent_unit)
        # 如果美元和美分都为零
        else:
            return "zero dollars"

    def _remove_commas(self, m: str) -> str:
        """
        This method is used to remove commas from sentences.
        """
        # 移除句子中的逗号
        return m.group(1).replace(",", "")

    def _expand_decimal_point(self, m: str) -> str:
        """
        This method is used to expand '.' into spoken word ' point '.
        """
        # 将句子中的小数点替换为单词" point "
        return m.group(1).replace(".", " point ")

    def _expand_ordinal(self, num: str) -> str:
        """
        This method is used to expand ordinals such as '1st', '2nd' into spoken words.
        """
        # 定义序数后缀字典
        ordinal_suffixes = {1: "st", 2: "nd", 3: "rd"}

        # 将匹配到的序数字符串转换为整数
        num = int(num.group(0)[:-2])
        # 如果数字在 10 到 20 之间，或者个位数字为 1、2 或 3，则使用 "th" 后缀
        if 10 <= num % 100 and num % 100 <= 20:
            suffix = "th"
        else:
            suffix = ordinal_suffixes.get(num % 10, "th")
        # 返回数字加上对应的序数后缀
        return self.number_to_words(num) + suffix

    def _expand_number(self, m: str) -> str:
        """
        This method acts as a preprocessing step for numbers between 1000 and 3000 (same as the original repository,
        link :
        https://github.com/neonbjb/tortoise-tts/blob/4003544b6ff4b68c09856e04d3eff9da26d023c2/tortoise/utils/tokenizer.py#L86)
        """
        # 提取匹配到的数字字符串并转换为整数
        num = int(m.group(0))

        # 如果数字在 1000 到 3000 之间
        if num > 1000 and num < 3000:
            # 如果数字为 2000，则直接返回 "two thousand"
            if num == 2000:
                return "two thousand"
            # 如果数字在 2001 到 2009 之间，返回 "two thousand " 加上后两位数字的文字表示
            elif num > 2000 and num < 2010:
                return "two thousand " + self.number_to_words(num % 100)
            # 如果数字能整除 100，则返回百位的文字表示加上"hundred"
            elif num % 100 == 0:
                return self.number_to_words(num // 100) + " hundred"
            # 其他情况返回数字的文字表示
            else:
                return self.number_to_words(num)
        # 如果数字不在 1000 到 3000 之间，则直接返回数字的文字表示
        else:
            return self.number_to_words(num)
    # 此方法用于将文本中的数字标准化，例如将数字转换为单词，移除逗号等操作
    def normalize_numbers(self, text: str) -> str:
        # 使用正则表达式替换匹配到的数字及逗号，调用 _remove_commas 方法
        text = re.sub(re.compile(r"([0-9][0-9\,]+[0-9])"), self._remove_commas, text)
        # 使用正则表达式替换匹配到的以 £ 开头的数字及逗号，转换为英文形式
        text = re.sub(re.compile(r"£([0-9\,]*[0-9]+)"), r"\1 pounds", text)
        # 使用正则表达式替换匹配到的以 $ 开头的数字及逗号，调用 _expand_dollars 方法
        text = re.sub(re.compile(r"\$([0-9\.\,]*[0-9]+)"), self._expand_dollars, text)
        # 使用正则表达式替换匹配到的浮点数，调用 _expand_decimal_point 方法
        text = re.sub(re.compile(r"([0-9]+\.[0-9]+)"), self._expand_decimal_point, text)
        # 使用正则表达式替换匹配到的序数词，调用 _expand_ordinal 方法
        text = re.sub(re.compile(r"[0-9]+(st|nd|rd|th)"), self._expand_ordinal, text)
        # 使用正则表达式替换匹配到的数字，调用 _expand_number 方法
        text = re.sub(re.compile(r"[0-9]+"), self._expand_number, text)
        # 返回处理后的文本
        return text

    # 扩展缩写词
    def expand_abbreviations(self, text: str) -> str:
        # 遍历缩写词和其对应的替换规则
        for regex, replacement in self._abbreviations:
            # 使用正则表达式替换匹配到的缩写词
            text = re.sub(regex, replacement, text)
        # 返回处理后的文本
        return text

    # 移除多余的空白字符
    def collapse_whitespace(self, text: str) -> str:
        # 使用正则表达式替换多个连续的空白字符为单个空格
        return re.sub(re.compile(r"\s+"), " ", text)

    # 对象可调用时执行的方法，将文本转换为 ASCII 格式，将数字转换为英文形式，扩展缩写词，并移除多余的空白字符
    def __call__(self, text):
        # 将文本转换为 ASCII 格式
        text = self.convert_to_ascii(text)
        # 将文本转换为小写
        text = text.lower()
        # 标准化文本中的数字
        text = self.normalize_numbers(text)
        # 扩展缩写词
        text = self.expand_abbreviations(text)
        # 移除多余的空白字符
        text = self.collapse_whitespace(text)
        # 移除双引号
        text = text.replace('"', "")
        # 返回处理后的文本
        return text
```