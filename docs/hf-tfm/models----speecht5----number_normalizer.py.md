# `.\models\speecht5\number_normalizer.py`

```py
# coding=utf-8
# Copyright 2023 The Fairseq Authors, Microsoft Research, and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Number Normalizer class for SpeechT5."""

import re

class EnglishNumberNormalizer:
    def __init__(self):
        # 单位数字（0-9）
        self.ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        # 十位数（11-19）
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
        # 十位数（10, 20, 30, ..., 90）
        self.tens = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        # 千位数（如thousand, million, billion, ...）
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

        # 定义一个字典，将货币符号映射到它们的名称
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
    # 将给定的数字转换成英文单词表示，例如将数字 1234 转换成 "one thousand two hundred thirty four"
    def spell_number(self, num):
        # 如果数字是 0，直接返回 "zero"
        if num == 0:
            return "zero"

        parts = []
        # 遍历 self.thousands 列表中的每个元素（"thousand", "million", "billion" 等）
        for i in range(0, len(self.thousands)):
            # 如果当前 num 不是以 000 结尾的，则继续处理
            if num % 1000 != 0:
                part = ""
                # 获取当前三位数中的百位数
                hundreds = num % 1000 // 100
                # 获取当前三位数中的十位和个位数的组合
                tens_units = num % 100

                # 如果百位数大于 0，则添加 "百" 位数的英文表达
                if hundreds > 0:
                    part += self.ones[hundreds] + " hundred"
                    # 如果十位和个位数的组合大于 0，则在百位数的后面添加 "and"
                    if tens_units > 0:
                        part += " and "

                # 判断十位和个位数的组合是否在 11 到 19 之间，若是，则添加对应的英文表达
                if tens_units >= 11 and tens_units <= 19:
                    part += self.teens[tens_units - 10]
                else:
                    # 否则，分别添加十位数和个位数的英文表达
                    tens_digit = self.tens[tens_units // 10]
                    ones_digit = self.ones[tens_units % 10]
                    if tens_digit:
                        part += tens_digit
                    if ones_digit:
                        if tens_digit:
                            part += " "
                        part += ones_digit

                # 将当前三位数的英文表达添加到 parts 列表中
                parts.append(part)

            # 将 num 变为其除以 1000 的整数部分，用于处理下一个 "thousand" 的处理
            num //= 1000

        # 将 parts 列表中的内容逆序拼接成最终的英文数字表达式并返回
        return " ".join(reversed(parts))
    def convert(self, number):
        """
        Converts an individual number passed in string form to spelt-out form
        将传入的字符串形式的数字转换为拼写形式
        """
        # Split number into integer and decimal parts if present
        if "." in number:
            integer_part, decimal_part = number.split(".")
        else:
            integer_part, decimal_part = number, "00"

        # Extract currency symbol if present
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

        # Extract 'minus' prefix for negative numbers
        minus_prefix = ""
        if integer_part.startswith("-"):
            minus_prefix = "minus "
            integer_part = integer_part[1:]
        elif integer_part.startswith("minus"):
            minus_prefix = "minus "
            integer_part = integer_part[len("minus") :]

        # Handle percentage suffix
        percent_suffix = ""
        if "%" in integer_part or "%" in decimal_part:
            percent_suffix = " percent"
            integer_part = integer_part.replace("%", "")
            decimal_part = decimal_part.replace("%", "")

        # Pad integer part to ensure proper grouping
        integer_part = integer_part.zfill(3 * ((len(integer_part) - 1) // 3 + 1))

        # Split integer part into groups of three for conversion
        parts = []
        for i in range(0, len(integer_part), 3):
            chunk = int(integer_part[i : i + 3])
            if chunk > 0:
                part = self.spell_number(chunk)
                unit = self.thousands[len(integer_part[i:]) // 3 - 1]
                if unit:
                    part += " " + unit
                parts.append(part)

        # Join parts to form the spelled-out integer
        spelled_integer = " ".join(parts)

        # Format the final output based on conditions
        if decimal_part == "00":
            return (
                f"{minus_prefix}{spelled_integer}{percent_suffix}{currency_symbol}"
                if minus_prefix or currency_symbol
                else f"{spelled_integer}{percent_suffix}"
            )
        else:
            # Convert decimal part to spelled-out form
            spelled_decimal = " ".join([self.spell_number(int(digit)) for digit in decimal_part])
            return (
                f"{minus_prefix}{spelled_integer} point {spelled_decimal}{percent_suffix}{currency_symbol}"
                if minus_prefix or currency_symbol
                else f"{minus_prefix}{spelled_integer} point {spelled_decimal}{percent_suffix}"
            )
    def __call__(self, text):
        """
        Convert numbers / number-like quantities in a string to their spelt-out counterparts
        """
        # 定义匹配各种货币符号的正则表达式模式
        pattern = r"(?<!\w)(-?\$?\€?\£?\¢?\¥?\₹?\₽?\฿?\₺?\₴?\₣?\₡?\₱?\₪?\₮?\₩?\₦?\₫?\﷼?\d+(?:\.\d{1,2})?%?)(?!\w)"

        # 查找并替换数字中的逗号（例如 15,000 -> 15000 等）
        text = re.sub(r"(\d+,\d+)", lambda match: match.group(1).replace(",", ""), text)

        # 使用正则表达式查找并替换文本中的数字为其对应的拼写形式
        converted_text = re.sub(pattern, lambda match: self.convert(match.group(1)), text)
        
        # 将连续多个空格替换为单个空格
        converted_text = re.sub(" +", " ", converted_text)

        # 返回转换后的文本
        return converted_text
```