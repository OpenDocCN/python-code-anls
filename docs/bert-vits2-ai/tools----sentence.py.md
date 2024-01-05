# `d:/src/tocomm/Bert-VITS2\tools\sentence.py`

```
import logging  # 导入日志记录模块

import regex as re  # 导入正则表达式模块

from tools.classify_language import classify_language, split_alpha_nonalpha  # 从工具包中导入分类语言和分割字母和非字母的函数


def check_is_none(item) -> bool:
    """none -> True, not none -> False"""
    return (
        item is None  # 如果 item 为 None，则返回 True
        or (isinstance(item, str) and str(item).isspace())  # 如果 item 是字符串且只包含空格，则返回 True
        or str(item) == ""  # 如果 item 为空字符串，则返回 True
    )


def markup_language(text: str, target_languages: list = None) -> str:
    pattern = (
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"  # 定义一个正则表达式模式
        r"\！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
    r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    # 定义一个正则表达式模式，用于匹配中文和其他特殊字符

    sentences = re.split(pattern, text)
    # 使用正则表达式模式对文本进行分割，得到句子列表

    pre_lang = ""
    p = 0
    # 初始化变量 pre_lang 和 p

    if target_languages is not None:
        # 如果目标语言不为空
        sorted_target_languages = sorted(target_languages)
        # 对目标语言列表进行排序
        if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:
            # 如果排序后的目标语言列表符合特定的语言组合
            new_sentences = []
            # 初始化一个新的句子列表
            for sentence in sentences:
                # 遍历原句子列表
                new_sentences.extend(split_alpha_nonalpha(sentence))
                # 将每个句子按照字母和非字母分割后加入新的句子列表
            sentences = new_sentences
            # 将原句子列表替换为新的句子列表

    for sentence in sentences:
        # 遍历句子列表
        if check_is_none(sentence):
            # 如果句子为空
            continue
            # 跳过当前循环

        lang = classify_language(sentence, target_languages)
        # 对句子进行语言分类，得到语言类型
        if pre_lang == "":  # 如果前一个语言为空
            text = text[:p] + text[p:].replace(  # 替换文本中的句子
                sentence, f"[{lang.upper()}]{sentence}", 1  # 在句子前添加大写语言标记
            )
            p += len(f"[{lang.upper()}]")  # 更新指针位置
        elif pre_lang != lang:  # 如果前一个语言不等于当前语言
            text = text[:p] + text[p:].replace(  # 替换文本中的句子
                sentence, f"[{pre_lang.upper()}][{lang.upper()}]{sentence}", 1  # 在句子前添加前一个语言和当前语言的大写标记
            )
            p += len(f"[{pre_lang.upper()}][{lang.upper()}]")  # 更新指针位置
        pre_lang = lang  # 更新前一个语言为当前语言
        p += text[p:].index(sentence) + len(sentence)  # 更新指针位置到下一个句子
    text += f"[{pre_lang.upper()}]"  # 在文本末尾添加前一个语言的大写标记

    return text  # 返回处理后的文本


def split_by_language(text: str, target_languages: list = None) -> list:  # 定义函数，根据语言拆分文本
    pattern = (  # 定义正则表达式模式
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"
        r"\！？\。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    )
    # 使用正则表达式模式来定义句子分隔符的模式

    sentences = re.split(pattern, text)
    # 使用正则表达式模式来分割文本，得到句子列表

    pre_lang = ""
    start = 0
    end = 0
    sentences_list = []
    # 初始化变量

    if target_languages is not None:
        sorted_target_languages = sorted(target_languages)
        # 对目标语言列表进行排序
        if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:
            # 如果排序后的目标语言列表符合特定的语言组合
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(split_alpha_nonalpha(sentence))
            sentences = new_sentences
            # 对每个句子进行处理，将字母和非字母字符分开，重新组成句子列表

    for sentence in sentences:
        # 遍历每个句子
        if check_is_none(sentence):  # 检查句子是否为空，如果是空句子则跳过
            continue

        lang = classify_language(sentence, target_languages)  # 使用给定的目标语言列表对句子进行语言分类

        end += text[end:].index(sentence)  # 找到当前句子在文本中的位置并更新结束索引
        if pre_lang != "" and pre_lang != lang:  # 如果前一个句子的语言不为空且与当前句子的语言不同
            sentences_list.append((text[start:end], pre_lang))  # 将前一个句子及其语言添加到句子列表中
            start = end  # 更新起始索引为当前句子的结束索引
        end += len(sentence)  # 更新结束索引为当前句子的结束索引
        pre_lang = lang  # 更新前一个句子的语言为当前句子的语言
    sentences_list.append((text[start:], pre_lang))  # 将最后一个句子及其语言添加到句子列表中

    return sentences_list  # 返回句子列表


def sentence_split(text: str, max: int) -> list:  # 定义一个函数，接受一个字符串和一个整数，返回一个列表
    pattern = r"[!(),—+\-.:;?？。，、；：]+"  # 定义一个正则表达式模式，用于匹配句子分隔符
    sentences = re.split(pattern, text)  # 使用正则表达式模式对文本进行分割，得到句子列表
    discarded_chars = re.findall(pattern, text)  # 使用正则表达式模式找到文本中被丢弃的字符，并存储在丢弃字符列表中
sentences_list, count, p = [], 0, 0  # 初始化一个空列表用于存储句子，以及计数器和指针变量

# 按被分割的符号遍历
for i, discarded_chars in enumerate(discarded_chars):  # 遍历被丢弃的字符列表
    count += len(sentences[i]) + len(discarded_chars)  # 计算句子长度和被丢弃字符长度的总和
    if count >= max:  # 如果总长度超过了最大长度
        sentences_list.append(text[p : p + count].strip())  # 将指定范围内的文本添加到句子列表中
        p += count  # 更新指针位置
        count = 0  # 重置计数器

# 加入最后剩余的文本
if p < len(text):  # 如果指针位置小于文本长度
    sentences_list.append(text[p:])  # 将剩余的文本添加到句子列表中

return sentences_list  # 返回句子列表


def sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None):
    # 如果该speaker只支持一种语言
    # 检查说话者语言是否不为空且长度为1
    if speaker_lang is not None and len(speaker_lang) == 1:
        # 如果语言不是"自动"或"混合"，且不等于说话者语言的第一个元素
        if lang.upper() not in ["AUTO", "MIX"] and lang.lower() != speaker_lang[0]:
            # 记录调试信息
            logging.debug(
                f'lang "{lang}" is not in speaker_lang {speaker_lang},automatically set lang={speaker_lang[0]}'
            )
        # 将语言设置为说话者语言的第一个元素
        lang = speaker_lang[0]

    # 初始化句子列表
    sentences_list = []
    # 如果语言不是"混合"
    if lang.upper() != "MIX":
        # 如果最大长度小于等于0
        if max <= 0:
            # 将文本进行标记化处理，如果语言是"自动"，否则使用指定语言标记文本
            sentences_list.append(
                markup_language(text, speaker_lang)
                if lang.upper() == "AUTO"
                else f"[{lang.upper()}]{text}[{lang.upper()}]"
            )
        else:
            # 将文本按最大长度分割成句子
            for i in sentence_split(text, max):
                # 如果句子为空
                if check_is_none(i):
                    # 继续下一次循环
                    continue
                # 将句子添加到句子列表中
                sentences_list.append(
                    markup_language(i, speaker_lang)  # 对每个句子进行标记语言处理
                    if lang.upper() == "AUTO"  # 如果语言参数为"auto"，则执行下面的操作
                    else f"[{lang.upper()}]{i}[{lang.upper()}]"  # 否则，在句子前后加上语言标记
                )
    else:  # 如果没有指定目标语言
        sentences_list.append(text)  # 直接将整个文本作为一个句子添加到句子列表中

    for i in sentences_list:  # 遍历句子列表
        logging.debug(i)  # 使用调试级别的日志记录每个句子

    return sentences_list  # 返回处理后的句子列表


if __name__ == "__main__":  # 如果当前脚本被直接执行
    text = "这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。我悄悄地披了大衫，带上门出去。"
    print(markup_language(text, target_languages=None))  # 打印标记语言处理后的文本
    print(sentence_split(text, max=50))  # 打印按照指定长度分割后的句子列表
    print(sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None))  # 打印按照指定长度分割并标记语言处理后的句子列表
    print(split_by_language(text, ["zh", "ja", "en"]))  # 打印按照指定语言分割后的句子列表
    text = "vits和Bert-VITS2是tts模型。花费3days.花费3天。Take 3 days"
    # 定义一个字符串变量

    print(split_by_language(text, ["zh", "ja", "en"]))
    # 调用split_by_language函数，传入text字符串和语言列表，打印输出结果

    print(split_by_language(text, ["zh", "en"]))
    # 调用split_by_language函数，传入text字符串和语言列表，打印输出结果

    text = "vits 和 Bert-VITS2 是 tts 模型。花费 3 days. 花费 3天。Take 3 days"
    # 重新定义字符串变量

    print(split_by_language(text, ["zh", "en"]))
    # 调用split_by_language函数，传入text字符串和语言列表，打印输出结果
```