# `.\transformers\models\speecht5\number_normalizer.py`

```py
# 导入正则表达式模块
import re

# 定义一个用于规范英文数字的类
class EnglishNumberNormalizer:
    # 类初始化方法
    def __init__(self):
        # 单位数字列表，索引对应相应的数字
        self.ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        # 十几的数字列表，索引对应相应的数字
        self.teens = [
            "",
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
        # 十位数字列表，索引对应相应的数字
        self.tens = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        # 千位以上的数字列表，索引对应相应的数字
        self.thousands = [
            "",
            "thousand",
            "million",
            "billion",
            "trillion",
            "quadrillion",
            "quintillion",
            "sextillion",
            "septillion",
            "octillion",
            "nonillion",
            "decillion",
        ]

        # 定义一个字典，用于将货币符号映射到其名称
        # 根据 https://en.wikipedia.org/wiki/Template:Most_traded_currencies
        self.currency_symbols = {
            "$": " dollars",
            "€": " euros",
            "£": " pounds",
            "¢": " cents",
            "¥": " japanese yen",
            "﷼": " saudi riyal",
            "₹": " indian rupees",
            "₽": " russian rubles",
            "฿": " thai baht",
            "₺": " turkish liras",
            "₴": " ukrainian hryvnia",
            "₣": " swiss francs",
            "₡": " costa rican colon",
            "₱": " philippine peso",
            "₪": " israeli shekels",
            "₮": " mongolian tögrög",
            "₩": " south korean won",
            "₦": " nigerian naira",
            "₫": " vietnamese Đồng",
        }
    # 将数字转换为英文表达
    def spell_number(self, num):
        # 如果数字为0，则直接返回"zero"
        if num == 0:
            return "zero"

        # 定义一个空列表，用于存放各个部分的英文表达
        parts = []
        # 遍历千位、百位、十位和个位
        for i in range(0, len(self.thousands)):
            # 如果该部分不为0
            if num % 1000 != 0:
                part = ""
                # 计算百位的数值
                hundreds = num % 1000 // 100
                # 计算十位和个位的数值
                tens_units = num % 100

                # 如果百位不为0
                if hundreds > 0:
                    part += self.ones[hundreds] + " hundred"
                    # 如果十位和个位不为0
                    if tens_units > 0:
                        part += " and "

                # 处理10到19的特殊情况
                if tens_units > 10 and tens_units < 20:
                    part += self.teens[tens_units - 10]
                else:
                    # 获取十位和个位的英文表达
                    tens_digit = self.tens[tens_units // 10]
                    ones_digit = self.ones[tens_units % 10]
                    if tens_digit:
                        part += tens_digit
                    if ones_digit:
                        if tens_digit:
                            part += " "
                        part += ones_digit

                # 将该部分的英文表达添加到列表中
                parts.append(part)

            # 将数字右移3位，相当于去除千位的数值
            num //= 1000

        # 返回所有部分的英文表达，反转后以空格分隔
        return " ".join(reversed(parts))
    def convert(self, number):
        """
        Converts an individual number passed in string form to spelt-out form
        """
        # 检查数字中是否包含小数点，将整数部分和小数部分分开
        if "." in number:
            integer_part, decimal_part = number.split(".")
        else:
            integer_part, decimal_part = number, "00"

        # 提取货币符号（如果有）
        currency_symbol = ""
        for symbol, name in self.currency_symbols.items():
            if integer_part.startswith(symbol):
                currency_symbol = name
                integer_part = integer_part[len(symbol) :]
                break

            if integer_part.startswith("-"):
                if integer_part[1:].startswith(symbol):
                    currency_symbol = name
                    integer_part = "-" + integer_part[len(symbol) + 1 :]
                    break

        # 提取负数的“minus”前缀
        minus_prefix = ""
        if integer_part.startswith("-"):
            minus_prefix = "minus "
            integer_part = integer_part[1:]
        elif integer_part.startswith("minus"):
            minus_prefix = "minus "
            integer_part = integer_part[len("minus") :]

        # 提取百分比后缀
        percent_suffix = ""
        if "%" in integer_part or "%" in decimal_part:
            percent_suffix = " percent"
            integer_part = integer_part.replace("%", "")
            decimal_part = decimal_part.replace("%", "")

        # 将整数部分填充至 3 的倍数长度
        integer_part = integer_part.zfill(3 * ((len(integer_part) - 1) // 3 + 1))

        # 将整数部分拆分成三位一组，并转换为对应的英文表示
        parts = []
        for i in range(0, len(integer_part), 3):
            chunk = int(integer_part[i : i + 3])
            if chunk > 0:
                part = self.spell_number(chunk)
                unit = self.thousands[len(integer_part[i:]) // 3 - 1]
                if unit:
                    part += " " + unit
                parts.append(part)

        spelled_integer = " ".join(parts)

        # 根据条件（是否有小数部分、货币符号、负数前缀等）格式化转换后的数字
        if decimal_part == "00":
            return (
                f"{minus_prefix}{spelled_integer}{percent_suffix}{currency_symbol}"
                if minus_prefix or currency_symbol
                else f"{spelled_integer}{percent_suffix}"
            )
        else:
            spelled_decimal = " ".join([self.spell_number(int(digit)) for digit in decimal_part])
            return (
                f"{minus_prefix}{spelled_integer} point {spelled_decimal}{percent_suffix}{currency_symbol}"
                if minus_prefix or currency_symbol
                else f"{minus_prefix}{spelled_integer} point {spelled_decimal}{percent_suffix}"
            )
    # 定义一个函数，将字符串中的数字或类似数字的部分替换成其对应的英文单词
    def __call__(self, text):
        """
        Convert numbers / number-like quantities in a string to their spelt-out counterparts
        """
        # 匹配所有货币符号的正则表达式
        pattern = r"(?<!\w)(-?\$?\€?\£?\¢?\¥?\₹?\₽?\฿?\₺?\₴?\₣?\₡?\₱?\₪?\₮?\₩?\₦?\₫?\﷼?\d+(?:\.\d{1,2})?%?)(?!\w)"

        # 找到并替换数字中的逗号（例如 15,000 -> 15000）
        text = re.sub(r"(\d+,\d+)", lambda match: match.group(1).replace(",", ""), text)

        # 使用正则表达式查找并替换文本中的数字
        converted_text = re.sub(pattern, lambda match: self.convert(match.group(1)), text)
        # 将多余的空格替换成一个空格
        converted_text = re.sub(" +", " ", converted_text)

        # 返回转换后的文本
        return converted_text
```