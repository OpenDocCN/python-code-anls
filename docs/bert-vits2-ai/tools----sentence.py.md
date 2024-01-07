# `Bert-VITS2\tools\sentence.py`

```

# 导入日志模块
import logging

# 导入正则表达式模块
import regex as re

# 从自定义工具包中导入classify_language和split_alpha_nonalpha函数
from tools.classify_language import classify_language, split_alpha_nonalpha

# 检查输入项是否为None
def check_is_none(item) -> bool:
    """none -> True, not none -> False"""
    return (
        item is None
        or (isinstance(item, str) and str(item).isspace())
        or str(item) == ""
    )

# 对文本进行标记化处理
def markup_language(text: str, target_languages: list = None) -> str:
    # 定义标点符号的正则表达式模式
    pattern = (
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"
        r"\！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    )
    # 使用正则表达式模式对文本进行分割
    sentences = re.split(pattern, text)

    pre_lang = ""
    p = 0

    # 如果指定了目标语言
    if target_languages is not None:
        sorted_target_languages = sorted(target_languages)
        # 如果目标语言是英文和中文、英文和日文、英文和日文和中文的组合
        if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:
            new_sentences = []
            # 对每个句子进行分割
            for sentence in sentences:
                new_sentences.extend(split_alpha_nonalpha(sentence))
            sentences = new_sentences

    # 遍历每个句子
    for sentence in sentences:
        if check_is_none(sentence):
            continue

        # 对句子进行语言分类
        lang = classify_language(sentence, target_languages)

        # 如果前一个句子的语言为空
        if pre_lang == "":
            # 在文本中标记当前句子的语言
            text = text[:p] + text[p:].replace(
                sentence, f"[{lang.upper()}]{sentence}", 1
            )
            p += len(f"[{lang.upper()}]")
        # 如果前一个句子的语言不等于当前句子的语言
        elif pre_lang != lang:
            # 在文本中标记当前句子的语言和前一个句子的语言
            text = text[:p] + text[p:].replace(
                sentence, f"[{pre_lang.upper()}][{lang.upper()}]{sentence}", 1
            )
            p += len(f"[{pre_lang.upper()}][{lang.upper()}]")
        pre_lang = lang
        p += text[p:].index(sentence) + len(sentence)
    # 在文本末尾标记最后一个句子的语言
    text += f"[{pre_lang.upper()}]"

    return text

# 根据语言对文本进行分割
def split_by_language(text: str, target_languages: list = None) -> list:
    # 定义标点符号的正则表达式模式
    pattern = (
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"
        r"\！？\。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    )
    # 使用正则表达式模式对文本进行分割
    sentences = re.split(pattern, text)

    pre_lang = ""
    start = 0
    end = 0
    sentences_list = []

    # 如果指定了目标语言
    if target_languages is not None:
        sorted_target_languages = sorted(target_languages)
        # 如果目标语言是英文和中文、英文和日文、英文和日文和中文的组合
        if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:
            new_sentences = []
            # 对每个句子进行分割
            for sentence in sentences:
                new_sentences.extend(split_alpha_nonalpha(sentence))
            sentences = new_sentences

    # 遍历每个句子
    for sentence in sentences:
        if check_is_none(sentence):
            continue

        # 对句子进行语言分类
        lang = classify_language(sentence, target_languages)

        end += text[end:].index(sentence)
        # 如果前一个句子的语言不为空且不等于当前句子的语言
        if pre_lang != "" and pre_lang != lang:
            # 将前一个句子及其语言加入到列表中
            sentences_list.append((text[start:end], pre_lang))
            start = end
        end += len(sentence)
        pre_lang = lang
    # 将最后一个句子及其语言加入到列表中
    sentences_list.append((text[start:], pre_lang))

    return sentences_list

# 将文本按句子分割
def sentence_split(text: str, max: int) -> list:
    # 定义标点符号的正则表达式模式
    pattern = r"[!(),—+\-.:;?？。，、；：]+"
    # 使用正则表达式模式对文本进行分割
    sentences = re.split(pattern, text)
    # 找出文本中的标点符号
    discarded_chars = re.findall(pattern, text)

    sentences_list, count, p = [], 0, 0

    # 按被分割的符号遍历
    for i, discarded_chars in enumerate(discarded_chars):
        count += len(sentences[i]) + len(discarded_chars)
        if count >= max:
            sentences_list.append(text[p : p + count].strip())
            p += count
            count = 0

    # 加入最后剩余的文本
    if p < len(text):
        sentences_list.append(text[p:])

    return sentences_list

# 将文本按句子分割并进行标记化处理
def sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None):
    # 如果该speaker只支持一种语言
    if speaker_lang is not None and len(speaker_lang) == 1:
        if lang.upper() not in ["AUTO", "MIX"] and lang.lower() != speaker_lang[0]:
            logging.debug(
                f'lang "{lang}" is not in speaker_lang {speaker_lang},automatically set lang={speaker_lang[0]}'
            )
        lang = speaker_lang[0]

    sentences_list = []
    # 如果指定的语言不是混合语言
    if lang.upper() != "MIX":
        # 如果最大长度小于等于0
        if max <= 0:
            sentences_list.append(
                markup_language(text, speaker_lang)
                if lang.upper() == "AUTO"
                else f"[{lang.upper()}]{text}[{lang.upper()}]"
            )
        else:
            for i in sentence_split(text, max):
                if check_is_none(i):
                    continue
                sentences_list.append(
                    markup_language(i, speaker_lang)
                    if lang.upper() == "AUTO"
                    else f"[{lang.upper()}]{i}[{lang.upper()}]"
                )
    else:
        sentences_list.append(text)

    for i in sentences_list:
        logging.debug(i)

    return sentences_list

# 测试代码
if __name__ == "__main__":
    text = "这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。我悄悄地披了大衫，带上门出去。"
    print(markup_language(text, target_languages=None))
    print(sentence_split(text, max=50))
    print(sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None))

    text = "你好，这是一段用来测试自动标注的文本。こんにちは,これは自動ラベリングのテスト用テキストです.Hello, this is a piece of text to test autotagging.你好！今天我们要介绍VITS项目，其重点是使用了GAN Duration predictor和transformer flow,并且接入了Bert模型来提升韵律。Bert embedding会在稍后介绍。"
    print(split_by_language(text, ["zh", "ja", "en"]))

    text = "vits和Bert-VITS2是tts模型。花费3days.花费3天。Take 3 days"

    print(split_by_language(text, ["zh", "ja", "en"]))
    # output: [('vits', 'en'), ('和', 'ja'), ('Bert-VITS', 'en'), ('2是', 'zh'), ('tts', 'en'), ('模型。花费3', 'zh'), ('days.', 'en'), ('花费3天。', 'zh'), ('Take 3 days', 'en')]

    print(split_by_language(text, ["zh", "en"]))
    # output: [('vits', 'en'), ('和', 'zh'), ('Bert-VITS', 'en'), ('2是', 'zh'), ('tts', 'en'), ('模型。花费3', 'zh'), ('days.', 'en'), ('花费3天。', 'zh'), ('Take 3 days', 'en')]

    text = "vits 和 Bert-VITS2 是 tts 模型。花费 3 days. 花费 3天。Take 3 days"
    print(split_by_language(text, ["zh", "en"]))
    # output: [('vits ', 'en'), ('和 ', 'zh'), ('Bert-VITS2 ', 'en'), ('是 ', 'zh'), ('tts ', 'en'), ('模型。花费 ', 'zh'), ('3 days. ', 'en'), ('花费 3天。', 'zh'), ('Take 3 days', 'en')]

```