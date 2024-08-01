# `.\DB-GPT-src\dbgpt\util\string_utils.py`

```py
import re                         # 导入正则表达式模块

from typing import Dict           # 导入类型提示中的字典类型


def is_all_chinese(text):
    ### 判断字符串是否全为中文
    pattern = re.compile(r"^[一-龥]+$")    # 定义匹配全中文的正则表达式模式
    match = re.match(pattern, text)        # 尝试匹配字符串与正则表达式模式
    return match is not None               # 返回匹配结果是否成功的布尔值


def contains_chinese(text):
    """检查文本是否包含中文字符。"""
    return re.search(r"[\u4e00-\u9fa5]", text) is not None   # 使用正则表达式搜索文本中是否包含中文字符


def is_number_chinese(text):
    ### 判断字符串是否由数字和中文组成
    pattern = re.compile(r"^[\d一-龥]+$")   # 定义匹配数字和中文的正则表达式模式
    match = re.match(pattern, text)         # 尝试匹配字符串与正则表达式模式
    return match is not None                # 返回匹配结果是否成功的布尔值


def is_chinese_include_number(text):
    ### 判断字符串是否只包含中文或中文和数字的组合
    pattern = re.compile(r"^[一-龥]+[\d一-龥]*$")   # 定义匹配只包含中文或中文和数字的正则表达式模式
    match = re.match(pattern, text)                # 尝试匹配字符串与正则表达式模式
    return match is not None                       # 返回匹配结果是否成功的布尔值


def is_scientific_notation(string):
    # 判断字符串是否为科学计数法表示的数字
    pattern = r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"   # 定义匹配科学计数法的正则表达式模式
    match = re.match(pattern, str(string))                  # 尝试匹配字符串与正则表达式模式
    if match is not None:
        return True   # 如果匹配成功，返回True
    else:
        return False  # 如果匹配失败，返回False


def extract_content(long_string, s1, s2, is_include: bool = False) -> Dict[int, str]:
    # 提取文本内容
    match_map = {}                        # 初始化匹配结果的字典
    start_index = long_string.find(s1)     # 在长字符串中查找第一个分隔符s1的位置
    while start_index != -1:               # 循环直到找不到s1为止
        if is_include:
            end_index = long_string.find(s2, start_index + len(s1) + 1)
            extracted_content = long_string[start_index : end_index + len(s2)]
        else:
            end_index = long_string.find(s2, start_index + len(s1))
            extracted_content = long_string[start_index + len(s1) : end_index]
        if extracted_content:
            match_map[start_index] = extracted_content   # 将提取的内容加入到匹配结果字典中
        start_index = long_string.find(s1, start_index + 1)   # 继续查找下一个s1的位置
    return match_map   # 返回匹配结果字典


def extract_content_open_ending(long_string, s1, s2, is_include: bool = False):
    # 提取文本内容，开放结尾
    match_map = {}                        # 初始化匹配结果的字典
    start_index = long_string.find(s1)     # 在长字符串中查找第一个分隔符s1的位置
    while start_index != -1:               # 循环直到找不到s1为止
        if long_string.find(s2, start_index) <= 0:
            end_index = len(long_string)
        else:
            if is_include:
                end_index = long_string.find(s2, start_index + len(s1) + 1)
            else:
                end_index = long_string.find(s2, start_index + len(s1))
        if is_include:
            extracted_content = long_string[start_index : end_index + len(s2)]
        else:
            extracted_content = long_string[start_index + len(s1) : end_index]
        if extracted_content:
            match_map[start_index] = extracted_content   # 将提取的内容加入到匹配结果字典中
        start_index = long_string.find(s1, start_index + 1)   # 继续查找下一个s1的位置
    return match_map   # 返回匹配结果字典


def str_to_bool(s):
    if s.lower() in ("true", "t", "1", "yes", "y"):
        return True    # 如果字符串表示True，则返回True
    elif s.lower().startswith("true"):
        return True    # 如果字符串以True开头，则返回True
    elif s.lower() in ("false", "f", "0", "no", "n"):
        return False   # 如果字符串表示False，则返回False
    else:
        return False   # 默认返回False


def _to_str(x, charset="utf8", errors="strict"):
    if x is None or isinstance(x, str):
        return x   # 如果x为空或者已经是字符串类型，则直接返回x
    # 检查变量 x 是否属于 bytes 类型
    if isinstance(x, bytes):
        # 如果是 bytes 类型，则使用指定的字符集 charset 和错误处理方式 errors 进行解码
        return x.decode(charset, errors)

    # 如果 x 不是 bytes 类型，则将其转换为字符串类型并返回
    return str(x)
```