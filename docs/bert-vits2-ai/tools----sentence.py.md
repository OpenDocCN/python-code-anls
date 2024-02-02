# `Bert-VITS2\tools\sentence.py`

```py
# 导入 logging 模块
import logging

# 导入 regex 模块，并重命名为 re
import regex as re

# 从 tools.classify_language 模块中导入 classify_language 和 split_alpha_nonalpha 函数
from tools.classify_language import classify_language, split_alpha_nonalpha

# 定义一个函数，用于检查输入是否为 None
def check_is_none(item) -> bool:
    """none -> True, not none -> False"""
    return (
        item is None
        or (isinstance(item, str) and str(item).isspace())
        or str(item) == ""
    )

# 定义一个函数，用于标记语言
def markup_language(text: str, target_languages: list = None) -> str:
    # 定义匹配标点符号的正则表达式模式
    pattern = (
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"
        r"\！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    )
    # 使用正则表达式模式分割文本，得到句子列表
    sentences = re.split(pattern, text)

    # 初始化前一个句子的语言和位置
    pre_lang = ""
    p = 0

    # 如果目标语言列表不为空
    if target_languages is not None:
        # 对目标语言列表进行排序
        sorted_target_languages = sorted(target_languages)
        # 如果排序后的目标语言列表在指定的三种组合中
        if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:
            # 初始化一个新的句子列表
            new_sentences = []
            # 遍历原句子列表
            for sentence in sentences:
                # 将每个句子按字母和非字母分割，并加入到新的句子列表中
                new_sentences.extend(split_alpha_nonalpha(sentence))
            # 将新的句子列表赋值给原句子列表
            sentences = new_sentences

    # 遍历句子列表
    for sentence in sentences:
        # 如果句子为空，则跳过
        if check_is_none(sentence):
            continue

        # 判断句子的语言
        lang = classify_language(sentence, target_languages)

        # 如果前一个句子的语言为空
        if pre_lang == "":
            # 在文本中标记当前句子的语言，并更新位置
            text = text[:p] + text[p:].replace(
                sentence, f"[{lang.upper()}]{sentence}", 1
            )
            p += len(f"[{lang.upper()}]")
        # 如果前一个句子的语言不为空且与当前句子的语言不同
        elif pre_lang != lang:
            # 在文本中标记前一个句子和当前句子的语言，并更新位置
            text = text[:p] + text[p:].replace(
                sentence, f"[{pre_lang.upper()}][{lang.upper()}]{sentence}", 1
            )
            p += len(f"[{pre_lang.upper()}][{lang.upper()}]")
        # 更新前一个句子的语言和位置
        pre_lang = lang
        p += text[p:].index(sentence) + len(sentence)
    # 在文本末尾标记最后一个句子的语言
    text += f"[{pre_lang.upper()}]"

    # 返回标记后的文本
    return text

# 定义一个函数，用于按语言拆分文本
def split_by_language(text: str, target_languages: list = None) -> list:
    # 定义匹配标点符号的正则表达式模式
    pattern = (
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"
        r"\！？\。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    )
    # 使用正则表达式模式分割文本，得到句子列表
    sentences = re.split(pattern, text)
    
    # 初始化变量
    pre_lang = ""
    start = 0
    end = 0
    sentences_list = []
    
    # 如果目标语言不为空
    if target_languages is not None:
        # 对目标语言进行排序
        sorted_target_languages = sorted(target_languages)
        # 如果排序后的目标语言在指定的列表中
        if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:
            # 对句子进行拆分
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(split_alpha_nonalpha(sentence))
            sentences = new_sentences
    
    # 遍历句子列表
    for sentence in sentences:
        # 如果句子为空，则跳过
        if check_is_none(sentence):
            continue
    
        # 判断句子的语言
        lang = classify_language(sentence, target_languages)
    
        # 更新结束位置
        end += text[end:].index(sentence)
        # 如果前一个句子的语言不为空且与当前句子的语言不同
        if pre_lang != "" and pre_lang != lang:
            # 将前一个句子及其语言添加到句子列表中
            sentences_list.append((text[start:end], pre_lang))
            # 更新起始位置
            start = end
        # 更新结束位置
        end += len(sentence)
        # 更新前一个句子的语言
        pre_lang = lang
    # 将最后一个句子及其语言添加到句子列表中
    sentences_list.append((text[start:], pre_lang)
    
    # 返回句子列表
    return sentences_list
def sentence_split(text: str, max: int) -> list:
    # 定义用于分割句子的正则表达式模式
    pattern = r"[!(),—+\-.:;?？。，、；：]+"
    # 使用正则表达式模式分割文本，得到句子列表
    sentences = re.split(pattern, text)
    # 使用正则表达式模式查找被丢弃的字符
    discarded_chars = re.findall(pattern, text)

    sentences_list, count, p = [], 0, 0

    # 按被分割的符号遍历
    for i, discarded_chars in enumerate(discarded_chars):
        # 计算句子和被丢弃字符的总长度
        count += len(sentences[i]) + len(discarded_chars)
        # 如果总长度超过最大长度
        if count >= max:
            # 将超出最大长度的部分加入句子列表
            sentences_list.append(text[p : p + count].strip())
            p += count
            count = 0

    # 加入最后剩余的文本
    if p < len(text):
        sentences_list.append(text[p:])

    return sentences_list


def sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None):
    # 如果该speaker只支持一种语言
    if speaker_lang is not None and len(speaker_lang) == 1:
        # 如果指定的语言不在支持的语言列表中
        if lang.upper() not in ["AUTO", "MIX"] and lang.lower() != speaker_lang[0]:
            # 记录日志
            logging.debug(
                f'lang "{lang}" is not in speaker_lang {speaker_lang},automatically set lang={speaker_lang[0]}'
            )
        lang = speaker_lang[0]

    sentences_list = []
    # 如果语言不是混合语言
    if lang.upper() != "MIX":
        # 如果最大长度小于等于0
        if max <= 0:
            # 将文本进行标记语言处理
            sentences_list.append(
                markup_language(text, speaker_lang)
                if lang.upper() == "AUTO"
                else f"[{lang.upper()}]{text}[{lang.upper()}]"
            )
        else:
            # 对文本进行句子分割
            for i in sentence_split(text, max):
                # 检查句子是否为空
                if check_is_none(i):
                    continue
                # 将句子进行标记语言处理
                sentences_list.append(
                    markup_language(i, speaker_lang)
                    if lang.upper() == "AUTO"
                    else f"[{lang.upper()}]{i}[{lang.upper()}]"
                )
    else:
        sentences_list.append(text)

    # 记录句子列表中的每个句子
    for i in sentences_list:
        logging.debug(i)

    return sentences_list


if __name__ == "__main__":
    text = "这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。我悄悄地披了大衫，带上门出去。"
    print(markup_language(text, target_languages=None))
    # 调用 sentence_split 函数，将文本按照最大长度 50 进行分割并打印结果
    print(sentence_split(text, max=50))
    # 调用 sentence_split_and_markup 函数，将文本按照最大长度 50 进行分割并标记，并打印结果
    print(sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None))
    
    # 重新定义文本内容
    text = "你好，这是一段用来测试自动标注的文本。こんにちは,これは自動ラベリングのテスト用テキストです.Hello, this is a piece of text to test autotagging.你好！今天我们要介绍VITS项目，其重点是使用了GAN Duration predictor和transformer flow,并且接入了Bert模型来提升韵律。Bert embedding会在稍后介绍。"
    # 调用 split_by_language 函数，将文本按照指定语言列表进行分割，并打印结果
    print(split_by_language(text, ["zh", "ja", "en"]))
    
    # 重新定义文本内容
    text = "vits和Bert-VITS2是tts模型。花费3days.花费3天。Take 3 days"
    # 调用 split_by_language 函数，将文本按照指定语言列表进行分割，并打印结果
    print(split_by_language(text, ["zh", "ja", "en"]))
    # 输出: [('vits', 'en'), ('和', 'ja'), ('Bert-VITS', 'en'), ('2是', 'zh'), ('tts', 'en'), ('模型。花费3', 'zh'), ('days.', 'en'), ('花费3天。', 'zh'), ('Take 3 days', 'en')]
    
    # 调用 split_by_language 函数，将文本按照指定语言列表进行分割，并打印结果
    print(split_by_language(text, ["zh", "en"]))
    # 输出: [('vits', 'en'), ('和', 'zh'), ('Bert-VITS', 'en'), ('2是', 'zh'), ('tts', 'en'), ('模型。花费3', 'zh'), ('days.', 'en'), ('花费3天。', 'zh'), ('Take 3 days', 'en')]
    
    # 重新定义文本内容
    text = "vits 和 Bert-VITS2 是 tts 模型。花费 3 days. 花费 3天。Take 3 days"
    # 调用 split_by_language 函数，将文本按照指定语言列表进行分割，并打印结果
    print(split_by_language(text, ["zh", "en"]))
    # 输出: [('vits ', 'en'), ('和 ', 'zh'), ('Bert-VITS2 ', 'en'), ('是 ', 'zh'), ('tts ', 'en'), ('模型。花费 ', 'zh'), ('3 days. ', 'en'), ('花费 3天。', 'zh'), ('Take 3 days', 'en')]
```