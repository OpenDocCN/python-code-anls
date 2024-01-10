# `Bert-VITS2\text\japanese.py`

```
# 导入正则表达式模块
import re
# 导入unicodedata模块，用于处理Unicode字符
import unicodedata

# 从transformers模块中导入AutoTokenizer类
from transformers import AutoTokenizer

# 从text模块中导入标点符号和符号
from text import punctuation, symbols

# 从num2words模块中导入num2words函数，用于将数字转换为单词
from num2words import num2words

# 导入pyopenjtalk模块，用于处理日语文本
import pyopenjtalk
# 导入jaconv模块，用于处理日语文本的转换
import jaconv

# hiragana_map是一个映射表，将平假名映射到其音标的表示
hiragana_map = {
    # 以下是一些平假名到音标的映射关系
}
    "ぶぅ": " b u:",  # 将“ぶぅ”替换为“b u:”
    "ぶゅ": " by u",  # 将“ぶゅ”替换为“by u”
    "べぇ": " b e:",  # 将“べぇ”替换为“b e:”
    "ぼぉ": " b o:",  # 将“ぼぉ”替换为“b o:”
    "ぱぁ": " p a:",  # 将“ぱぁ”替换为“p a:”
    "ぴぃ": " p i:",  # 将“ぴぃ”替换为“p i:”
    "ぷぅ": " p u:",  # 将“ぷぅ”替换为“p u:”
    "ぷゃ": " py a",  # 将“ぷゃ”替换为“py a”
    "ぷゅ": " py u",  # 将“ぷゅ”替换为“py u”
    "ぷょ": " py o",  # 将“ぷょ”替换为“py o”
    "ぺぇ": " p e:",  # 将“ぺぇ”替换为“p e:”
    "ぽぉ": " p o:",  # 将“ぽぉ”替换为“p o:”
    "まぁ": " m a:",  # 将“まぁ”替换为“m a:”
    "みぃ": " m i:",  # 将“みぃ”替换为“m i:”
    "むぅ": " m u:",  # 将“むぅ”替换为“m u:”
    "むゃ": " my a",  # 将“むゃ”替换为“my a”
    "むゅ": " my u",  # 将“むゅ”替换为“my u”
    "むょ": " my o",  # 将“むょ”替换为“my o”
    "めぇ": " m e:",  # 将“めぇ”替换为“m e:”
    "もぉ": " m o:",  # 将“もぉ”替换为“m o:”
    "やぁ": " y a:",  # 将“やぁ”替换为“y a:”
    "ゆぅ": " y u:",  # 将“ゆぅ”替换为“y u:”
    "ゆゃ": " y a:",  # 将“ゆゃ”替换为“y a:”
    "ゆゅ": " y u:",  # 将“ゆゅ”替换为“y u:”
    "ゆょ": " y o:",  # 将“ゆょ”替换为“y o:”
    "よぉ": " y o:",  # 将“よぉ”替换为“y o:”
    "らぁ": " r a:",  # 将“らぁ”替换为“r a:”
    "りぃ": " r i:",  # 将“りぃ”替换为“r i:”
    "るぅ": " r u:",  # 将“るぅ”替换为“r u:”
    "るゃ": " ry a",  # 将“るゃ”替换为“ry a”
    "るゅ": " ry u",  # 将“るゅ”替换为“ry u”
    "るょ": " ry o",  # 将“るょ”替换为“ry o”
    "れぇ": " r e:",  # 将“れぇ”替换为“r e:”
    "ろぉ": " r o:",  # 将“ろぉ”替换为“r o:”
    "わぁ": " w a:",  # 将“わぁ”替换为“w a:”
    "をぉ": " o:",     # 将“をぉ”替换为“o:”
    "う゛": " b u",     # 将“う゛”替换为“b u”
    "でぃ": " d i",    # 将“でぃ”替换为“d i”
    "でゃ": " dy a",   # 将“でゃ”替换为“dy a”
    "でゅ": " dy u",   # 将“でゅ”替换为“dy u”
    "でょ": " dy o",   # 将“でょ”替换为“dy o”
    "てぃ": " t i",    # 将“てぃ”替换为“t i”
    "てゃ": " ty a",   # 将“てゃ”替换为“ty a”
    "てゅ": " ty u",   # 将“てゅ”替换为“ty u”
    "てょ": " ty o",   # 将“てょ”替换为“ty o”
    "すぃ": " s i",    # 将“すぃ”替换为“s i”
    "ずぁ": " z u",    # 将“ずぁ”替换为“z u”
    "ずぃ": " z i",    # 将“ずぃ”替换为“z i”
    "ずぇ": " z e",    # 将“ずぇ”替换为“z e”
    "ずぉ": " z o",    # 将“ずぉ”替换为“z o”
    "きゃ": " ky a",   # 将“きゃ”替换为“ky a”
    "きゅ": " ky u",   # 将“きゅ”替换为“ky u”
    "きょ": " ky o",   # 将“きょ”替换为“ky o”
    "しゃ": " sh a",   # 将“しゃ”替换为“sh a”
    "しゅ": " sh u",   # 将“しゅ”替换为“sh u”
    "しぇ": " sh e",   # 将“しぇ”替换为“sh e”
    "しょ": " sh o",   # 将“しょ”替换为“sh o”
    "ちゃ": " ch a",   # 将“ちゃ”替换为“ch a”
    "ちゅ": " ch u",   # 将“ちゅ”替换为“ch u”
    "ちぇ": " ch e",   # 将“ちぇ”替换为“ch e”
    "ちょ": " ch o",   # 将“ちょ”替换为“ch o”
    "とぅ": " t u",    # 将“とぅ”替换为“t u”
    "とゃ": " ty a",   # 将“とゃ”替换为“ty a”
    "とゅ": " ty u",   # 将“とゅ”替换为“ty u”
    "とょ": " ty o",   # 将“とょ”替换为“ty o”
    "どぁ": " d o ",   # 将“どぁ”替换为“d o ”
    "どぅ": " d u",    # 将“どぅ”替换为“d u”
    "どゃ": " dy a",   # 将“どゃ”替换为“dy a”
    "どゅ": " dy u",   # 将“どゅ”替换为“dy u”
    "どょ": " dy o",   # 将“どょ”替换为“dy o”
    "どぉ": " d o:",   # 将“どぉ”替换为“d o:”
    "にゃ": " ny a",   # 将“にゃ”替换为“ny a”
    "にゅ": " ny u",   # 将“にゅ”替换为“ny u”
    "にょ": " ny o",   # 将“にょ”替换为“ny o”
    "ひゃ": " hy a",   # 将“ひゃ”替换为“hy a”
    "ひゅ": " hy u",   # 将“ひゅ”替换为“hy u”
    "ひょ": " hy o",   # 将“ひょ”替换为“hy o”
    "みゃ": " my a",   # 将“みゃ”替换为“my a”
    "みゅ": " my u",   # 将“みゅ”替换为“my u”
    "みょ": " my o",   # 将“みょ”替换为“my o”
    "りゃ": " ry a",   # 将“りゃ”替换为“ry a”
    "りゅ": " ry u",   # 将“りゅ”替换为“ry u”
    "りょ": " ry o",   # 将“りょ”替换为“ry o”
    "ぎゃ": " gy a",   # 将“ぎゃ”替换为“gy a”
    "ぎゅ": " gy u",   # 将“ぎゅ”替换为“gy u”
    "ぎょ": " gy o",   # 将“ぎょ”替换为“gy o”
    "ぢぇ": " j e",    # 将“ぢぇ”替换为“j e”
    "ぢゃ": " j a",    # 将“ぢゃ”替换为“j a”
    "ぢゅ": " j u",    # 将“ぢゅ”替换为“j u”
    "ぢょ": " j o",    # 将“ぢょ”替换为“j o”
    "じぇ": " j e",    # 将“じぇ”替换为“j e”
    "じゃ": " j a",    # 将“じゃ”替换为“j a”
    "じゅ": " j u",    # 将“じゅ”替换为“j u”
    "じょ": " j o",    # 将“じょ”替换为“j o”
    "びゃ": " by a",   # 将“びゃ”替换为“by a”
    "びゅ": " by u",   # 将“びゅ”替换为“by u”
    "びょ": " by o",   # 将“びょ”替换为“by o”
    "ぴゃ": " py a",   # 将“ぴゃ”替换为“py a”
    "ぴゅ": " py u",   # 将“ぴゅ”替换为“py u”
    "ぴょ": " py o",   # 将“ぴょ”替换为“py o”
    "うぁ": " u a",    # 将“うぁ”替换为“u a”
    "うぃ": " w i",    # 将“うぃ”替换为“w i”
    "うぇ": " w e",    # 将“うぇ”替换为“w e”
    "うぉ": " w o",    # 将“うぉ”替换为“w o”
    "ふぁ": " f a",    # 将“ふぁ”替换为“f a”
    "ふぃ": " f i",    # 将“ふぃ”替换为“f i”
    # ふ行の仮名をローマ字に変換する
    "ふゅ": " hy u",
    "ふょ": " hy o",
    "ふぇ": " f e",
    "ふぉ": " f o",
    # 1音からなる変換規則
    "あ": " a",
    "い": " i",
    "う": " u",
    "ゔ": " v u",  # ゔの処理を追加
    "え": " e",
    "お": " o",
    "か": " k a",
    "き": " k i",
    "く": " k u",
    "け": " k e",
    "こ": " k o",
    "さ": " s a",
    "し": " sh i",
    "す": " s u",
    "せ": " s e",
    "そ": " s o",
    "た": " t a",
    "ち": " ch i",
    "つ": " ts u",
    "て": " t e",
    "と": " t o",
    "な": " n a",
    "に": " n i",
    "ぬ": " n u",
    "ね": " n e",
    "の": " n o",
    "は": " h a",
    "ひ": " h i",
    "ふ": " f u",
    "へ": " h e",
    "ほ": " h o",
    "ま": " m a",
    "み": " m i",
    "む": " m u",
    "め": " m e",
    "も": " m o",
    "ら": " r a",
    "り": " r i",
    "る": " r u",
    "れ": " r e",
    "ろ": " r o",
    "が": " g a",
    "ぎ": " g i",
    "ぐ": " g u",
    "げ": " g e",
    "ご": " g o",
    "ざ": " z a",
    "じ": " j i",
    "ず": " z u",
    "ぜ": " z e",
    "ぞ": " z o",
    "だ": " d a",
    "ぢ": " j i",
    "づ": " z u",
    "で": " d e",
    "ど": " d o",
    "ば": " b a",
    "び": " b i",
    "ぶ": " b u",
    "べ": " b e",
    "ぼ": " b o",
    "ぱ": " p a",
    "ぴ": " p i",
    "ぷ": " p u",
    "ぺ": " p e",
    "ぽ": " p o",
    "や": " y a",
    "ゆ": " y u",
    "よ": " y o",
    "わ": " w a",
    "ゐ": " i",
    "ゑ": " e",
    "ん": " N",
    "っ": " q",
    # ここまでに処理されてない ぁぃぅぇぉ はそのまま大文字扱い
    "ぁ": " a",
    "ぃ": " i",
    "ぅ": " u",
    "ぇ": " e",
    "ぉ": " o",
    "ゎ": " w a",
    # 長音の処理
    # for (pattern, replace_str) in JULIUS_LONG_VOWEL:
    #     text = pattern.sub(replace_str, text)
    # text = text.replace("o u", "o:")  # おう -> おーの音便
    "ー": ":",
    "〜": ":",
    "−": ":",
    "-": ":",
    # その他特別な処理
    "を": " o",
    # ここまでに処理されていないゅ等もそのまま大文字扱い（追加）
    "ゃ": " y a",
    "ゅ": " y u",
    "ょ": " y o",
# 定义一个将平假名转换为罗马音的函数
def hiragana2p(txt: str) -> str:
    """
    Modification of `jaconv.hiragana2julius`.
    - avoid using `:`, instead, `あーーー` -> `a a a a`.
    - avoid converting `o u` to `o o` (because the input is already actual `yomi`).
    - avoid using `N` for `ん` (for compatibility)
    - use `v` for `ゔ` related text.
    - add bare `ゃ` `ゅ` `ょ` to `y a` `y u` `y o` (for compatibility).
    """

    result = []  # 创建一个空列表用于存储转换后的结果
    skip = 0  # 初始化跳过的字符数为0
    for i in range(len(txt)):  # 遍历输入字符串的每个字符
        if skip:  # 如果有需要跳过的字符
            skip -= 1  # 减去需要跳过的字符数
            continue  # 继续下一次循环

        for length in range(3, 0, -1):  # 从3个字符到1个字符逐个尝试匹配
            if txt[i : i + length] in hiragana_map:  # 如果找到匹配的平假名
                result.append(hiragana_map[txt[i : i + length]])  # 将匹配的结果添加到列表中
                skip = length - 1  # 设置需要跳过的字符数
                break  # 跳出循环

    txt = "".join(result)  # 将列表中的结果拼接成字符串
    txt = txt.strip()  # 去除字符串两端的空格
    txt = txt.replace(":+", ":")  # 将特定字符串替换为另一个字符串

    # ここまで`jaconv.hiragana2julius`と音便処理と長音処理をのぞいて同じ
    # ここから`k a:: k i:`→`k a a a k i i`のように`:`の数だけ繰り返す処理
    pattern = r"(\w)(:*)"  # 定义一个正则表达式模式
    replacement = lambda m: m.group(1) + (" " + m.group(1)) * len(m.group(2))  # 定义一个替换函数

    txt = re.sub(pattern, replacement, txt)  # 使用正则表达式进行替换
    txt = txt.replace("N", "n")  # 将特定字符串替换为另一个字符串
    return txt  # 返回转换后的字符串


# 定义一个将片假名转换为音素的函数
def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    text = text.strip()  # 去除字符串两端的空格
    if text == "ー":  # 如果字符串为特定值
        return ["ー"]  # 返回一个列表
    elif text.startswith("ー"):  # 如果字符串以特定值开头
        return ["ー"] + kata2phoneme(text[1:])  # 返回一个列表
    res = []  # 创建一个空列表用于存储结果
    prev = None  # 初始化前一个字符为None
    while text:  # 当字符串不为空时
        if re.match(_MARKS, text):  # 如果字符串匹配特定模式
            res.append(text)  # 将字符串添加到列表中
            text = text[1:]  # 去除字符串的第一个字符
            continue  # 继续下一次循环
        if text.startswith("ー"):  # 如果字符串以特定值开头
            if prev:  # 如果前一个字符存在
                res.append(prev[-1])  # 将前一个字符的最后一个字符添加到列表中
            text = text[1:]  # 去除字符串的第一个字符
            continue  # 继续下一次循环
        res += hiragana2p(jaconv.kata2hira(text)).split(" ")  # 将转换后的结果拆分成列表并添加到结果列表中
        break  # 跳出循环
    # res = _COLON_RX.sub(":", res)
    return res  # 返回结果列表


_SYMBOL_TOKENS = set(list("・、。？！"))  # 创建一个包含特定字符的集合
_NO_YOMI_TOKENS = set(list("「」『』―（）［］[]"))  # 创建一个包含特定字符的集合
_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)  # 创建一个正则表达式模式
# 将文本转换为分隔片假名
def text2sep_kata(text: str):
    # 使用 pyopenjtalk 运行前端处理文本
    parsed = pyopenjtalk.run_frontend(text)
    res = []  # 存储处理后的结果
    sep = []  # 存储分隔片假名
    for parts in parsed:
        # 替换标点符号，并去除字符串中的特殊字符
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )
        if yomi:
            # 如果发音符合正则表达式定义的标记
            if re.match(_MARKS, yomi):
                # 如果单词长度大于1，则将单词拆分为字符列表
                if len(word) > 1:
                    word = [replace_punctuation(i) for i in list(word)]
                    yomi = word
                    res += yomi
                    sep += word
                    continue
                # 如果单词不在替换映射中，则将其替换为逗号
                elif word not in rep_map.keys() and word not in rep_map.values():
                    word = ","
                yomi = word
            res.append(yomi)
        else:
            # 如果单词是符号标记之一
            if word in _SYMBOL_TOKENS:
                res.append(word)
            # 如果单词是特殊字符之一
            elif word in ("っ", "ッ"):
                res.append("ッ")
            # 如果单词是无发音符号之一
            elif word in _NO_YOMI_TOKENS:
                pass
            else:
                res.append(word)
        sep.append(word)
    return sep, res, get_accent(parsed)


# 获取音节的重音信息
def get_accent(parsed):
    # 使用 pyopenjtalk 生成标签
    labels = pyopenjtalk.make_label(parsed)

    phonemes = []  # 存储音素
    accents = []  # 存储重音信息
    for n, label in enumerate(labels):
        # 从标签中提取音素
        phoneme = re.search(r"\-([^\+]*)\+", label).group(1)
        if phoneme not in ["sil", "pau"]:
            phonemes.append(phoneme.replace("cl", "q").lower())
        else:
            continue
        # 提取重音信息
        a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))
        a2 = int(re.search(r"\+(\d+)\+", label).group(1))
        if re.search(r"\-([^\+]*)\+", labels[n + 1]).group(1) in ["sil", "pau"]:
            a2_next = -1
        else:
            a2_next = int(re.search(r"\+(\d+)\+", labels[n + 1]).group(1)
        # 根据重音信息判断音节的重音类型
        if a1 == 0 and a2_next == a2 + 1:
            accents.append(-1)  # Falling
        elif a2 == 1 and a2_next == 2:
            accents.append(1)  # Rising
        else:
            accents.append(0)
    return list(zip(phonemes, accents))


# 特殊字符对应的发音
_ALPHASYMBOL_YOMI = {
    "#": "シャープ",
    "%": "パーセント",  # 将 "%" 替换为 "パーセント"
    "&": "アンド",  # 将 "&" 替换为 "アンド"
    "+": "プラス",  # 将 "+" 替换为 "プラス"
    "-": "マイナス",  # 将 "-" 替换为 "マイナス"
    ":": "コロン",  # 将 ":" 替换为 "コロン"
    ";": "セミコロン",  # 将 ";" 替换为 "セミコロン"
    "<": "小なり",  # 将 "<" 替换为 "小なり"
    "=": "イコール",  # 将 "=" 替换为 "イコール"
    ">": "大なり",  # 将 ">" 替换为 "大なり"
    "@": "アット",  # 将 "@" 替换为 "アット"
    "a": "エー",  # 将 "a" 替换为 "エー"
    "b": "ビー",  # 将 "b" 替换为 "ビー"
    "c": "シー",  # 将 "c" 替换为 "シー"
    "d": "ディー",  # 将 "d" 替换为 "ディー"
    "e": "イー",  # 将 "e" 替换为 "イー"
    "f": "エフ",  # 将 "f" 替换为 "エフ"
    "g": "ジー",  # 将 "g" 替换为 "ジー"
    "h": "エイチ",  # 将 "h" 替换为 "エイチ"
    "i": "アイ",  # 将 "i" 替换为 "アイ"
    "j": "ジェー",  # 将 "j" 替换为 "ジェー"
    "k": "ケー",  # 将 "k" 替换为 "ケー"
    "l": "エル",  # 将 "l" 替换为 "エル"
    "m": "エム",  # 将 "m" 替换为 "エム"
    "n": "エヌ",  # 将 "n" 替换为 "エヌ"
    "o": "オー",  # 将 "o" 替换为 "オー"
    "p": "ピー",  # 将 "p" 替换为 "ピー"
    "q": "キュー",  # 将 "q" 替换为 "キュー"
    "r": "アール",  # 将 "r" 替换为 "アール"
    "s": "エス",  # 将 "s" 替换为 "エス"
    "t": "ティー",  # 将 "t" 替换为 "ティー"
    "u": "ユー",  # 将 "u" 替换为 "ユー"
    "v": "ブイ",  # 将 "v" 替换为 "ブイ"
    "w": "ダブリュー",  # 将 "w" 替换为 "ダブリュー"
    "x": "エックス",  # 将 "x" 替换为 "エックス"
    "y": "ワイ",  # 将 "y" 替换为 "ワイ"
    "z": "ゼット",  # 将 "z" 替换为 "ゼット"
    "α": "アルファ",  # 将 "α" 替换为 "アルファ"
    "β": "ベータ",  # 将 "β" 替换为 "ベータ"
    "γ": "ガンマ",  # 将 "γ" 替换为 "ガンマ"
    "δ": "デルタ",  # 将 "δ" 替换为 "デルタ"
    "ε": "イプシロン",  # 将 "ε" 替换为 "イプシロン"
    "ζ": "ゼータ",  # 将 "ζ" 替换为 "ゼータ"
    "η": "イータ",  # 将 "η" 替换为 "イータ"
    "θ": "シータ",  # 将 "θ" 替换为 "シータ"
    "ι": "イオタ",  # 将 "ι" 替换为 "イオタ"
    "κ": "カッパ",  # 将 "κ" 替换为 "カッパ"
    "λ": "ラムダ",  # 将 "λ" 替换为 "ラムダ"
    "μ": "ミュー",  # 将 "μ" 替换为 "ミュー"
    "ν": "ニュー",  # 将 "ν" 替换为 "ニュー"
    "ξ": "クサイ",  # 将 "ξ" 替换为 "クサイ"
    "ο": "オミクロン",  # 将 "ο" 替换为 "オミクロン"
    "π": "パイ",  # 将 "π" 替换为 "パイ"
    "ρ": "ロー",  # 将 "ρ" 替换为 "ロー"
    "σ": "シグマ",  # 将 "σ" 替换为 "シグマ"
    "τ": "タウ",  # 将 "τ" 替换为 "タウ"
    "υ": "ウプシロン",  # 将 "υ" 替换为 "ウプシロン"
    "φ": "ファイ",  # 将 "φ" 替换为 "ファイ"
    "χ": "カイ",  # 将 "χ" 替换为 "カイ"
    "ψ": "プサイ",  # 将 "ψ" 替换为 "プサイ"
    "ω": "オメガ",  # 将 "ω" 替换为 "オメガ"
# 定义一个正则表达式，用于匹配带有逗号分隔符的数字
_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
# 定义一个货币符号到日语货币名称的映射字典
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
# 定义一个正则表达式，用于匹配货币符号和金额
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
# 定义一个正则表达式，用于匹配数字
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")

# 将数字转换为对应的日语文字
def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res

# 将文本中的英文字母和符号转换为对应的日语读音
def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])

# 检查字符是否为日语字符
def is_japanese_character(char):
    # 定义日语文字系统的 Unicode 范围
    japanese_ranges = [
        (0x3040, 0x309F),  # 平假名
        (0x30A0, 0x30FF),  # 片假名
        (0x4E00, 0x9FFF),  # 汉字 (CJK Unified Ideographs)
        (0x3400, 0x4DBF),  # 汉字扩展 A
        (0x20000, 0x2A6DF),  # 汉字扩展 B
        # 可以根据需要添加其他汉字扩展范围
    ]

    # 将字符的 Unicode 编码转换为整数
    char_code = ord(char)

    # 检查字符是否在任何一个日语范围内
    for start, end in japanese_ranges:
        if start <= char_code <= end:
            return True

    return False

# 定义一个替换映射字典，用于替换文本中的标点符号
rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "−": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

# 替换文本中的标点符号
def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    # 使用正则表达式替换文本中的非日语、中文和标点符号以外的字符
    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"  # 匹配日语字符范围
        + "".join(punctuation)  # 匹配标点符号
        + r"]+",  # 匹配除日语、中文和标点符号以外的字符
        "",  # 替换为空字符串
        replaced_text,  # 在replaced_text中进行替换
    )
    
    # 返回替换后的文本
    return replaced_text
# 对文本进行规范化处理，包括Unicode规范化和将数字转换为对应的日文单词
def text_normalize(text):
    res = unicodedata.normalize("NFKC", text)  # 使用NFKC规范化文本
    res = japanese_convert_numbers_to_words(res)  # 将文本中的数字转换为对应的日文单词
    # res = "".join([i for i in res if is_japanese_character(i)])  # 注释掉的代码，暂时不使用
    res = replace_punctuation(res)  # 替换文本中的标点符号
    res = res.replace("゙", "")  # 替换文本中的特定字符
    return res  # 返回处理后的文本


# 将电话号码分配给单词
def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word  # 初始化每个单词分配的电话号码数量
    for task in range(n_phone):  # 遍历每个电话号码
        min_tasks = min(phones_per_word)  # 找到分配的电话号码最少的单词数量
        min_index = phones_per_word.index(min_tasks)  # 找到分配的电话号码最少的单词的索引
        phones_per_word[min_index] += 1  # 将电话号码分配给数量最少的单词
    return phones_per_word  # 返回每个单词分配的电话号码数量列表


# 处理长音
def handle_long(sep_phonemes):
    for i in range(len(sep_phonemes)):  # 遍历音节列表
        if sep_phonemes[i][0] == "ー":  # 如果音节以"ー"开头
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]  # 将"ー"替换为前一个音节的最后一个字符
        if "ー" in sep_phonemes[i]:  # 如果音节中包含"ー"
            for j in range(len(sep_phonemes[i])):  # 遍历音节中的每个字符
                if sep_phonemes[i][j] == "ー":  # 如果字符为"ー"
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]  # 将"ー"替换为前一个字符的最后一个字符
    return sep_phonemes  # 返回处理后的音节列表


tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese-char-wwm")  # 使用预训练的分词器


# 对韵律和音节进行对齐
def align_tones(phones, tones):
    res = []  # 初始化结果列表
    for pho in phones:  # 遍历音节列表
        temp = [0] * len(pho)  # 初始化临时列表，用于存储每个音节的音调
        for idx, p in enumerate(pho):  # 遍历每个音节
            if len(tones) == 0:  # 如果音调列表为空
                break
            if p == tones[0][0]:  # 如果音节与音调匹配
                temp[idx] = tones[0][1]  # 将音调添加到临时列表中
                if idx > 0:  # 如果不是第一个音节
                    temp[idx] += temp[idx - 1]  # 将音调累加到前一个音节的音调上
                tones.pop(0)  # 移除已经匹配的音调
        temp = [0] + temp  # 在临时列表前添加一个0
        temp = temp[:-1]  # 在临时列表末尾移除一个元素
        if -1 in temp:  # 如果临时列表中存在-1
            temp = [i + 1 for i in temp]  # 将临时列表中的-1替换为1
        res.append(temp)  # 将临时列表添加到结果列表中
    res = [i for j in res for i in j]  # 将结果列表展开为一维列表
    assert not any([i < 0 for i in res]) and not any([i > 1 for i in res])  # 断言结果列表中不包含小于0或大于1的元素
    return res  # 返回处理后的音调列表


# 重新排列音调
def rearrange_tones(tones, phones):
    res = [0] * len(tones)  # 初始化结果列表为0
    # 遍历音调列表的索引范围
    for i in range(len(tones)):
        # 如果是第一个音调
        if i == 0:
            # 如果当前音调不是标点符号
            if tones[i] not in punctuation:
                # 将结果列表对应位置设为1
                res[i] = 1
        # 如果不是第一个音调
        elif tones[i] == prev:
            # 如果当前音素是标点符号
            if phones[i] in punctuation:
                # 将结果列表对应位置设为0
                res[i] = 0
            else:
                # 将结果列表对应位置设为1
                res[i] = 1
        # 如果当前音调大于前一个音调
        elif tones[i] > prev:
            # 将结果列表对应位置设为2
            res[i] = 2
        # 如果当前音调小于前一个音调
        elif tones[i] < prev:
            # 将前一个音调对应的结果列表位置设为3
            res[i - 1] = 3
            # 将当前音调对应的结果列表位置设为1
            res[i] = 1
        # 更新前一个音调
        prev = tones[i]
    # 返回结果列表
    return res
# 将规范化的文本转换为音素、音调和单词到音素的映射关系
def g2p(norm_text):
    # 将文本分割成单词和假名，并获取重音位置
    sep_text, sep_kata, acc = text2sep_kata(norm_text)
    sep_tokenized = []
    # 对分割后的单词进行处理，将非标点符号的单词进行分词
    for i in sep_text:
        if i not in punctuation:
            sep_tokenized.append(tokenizer.tokenize(i))
        else:
            sep_tokenized.append([i])

    # 将假名转换为音素，并处理长音
    sep_phonemes = handle_long([kata2phoneme(i) for i in sep_kata])
    # 异常处理，检查是否有未知的音素
    for i in sep_phonemes:
        for j in i:
            assert j in symbols, (sep_text, sep_kata, sep_phonemes)
    # 对音素进行音调对齐
    tones = align_tones(sep_phonemes, acc)

    word2ph = []
    # 将单词和音素进行对应
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        phone_len = len(phoneme)
        word_len = len(token)

        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
    phones = ["_"] + [j for i in sep_phonemes for j in i] + ["_"]
    # tones = [0] + rearrange_tones(tones, phones[1:-1]) + [0]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    assert len(phones) == len(tones)
    # 返回音素、音调和单词到音素的映射关系
    return phones, tones, word2ph


if __name__ == "__main__":
    # 从预训练模型中加载分词器
    tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")
    text = "hello,こんにちは、世界ー！……"
    from text.japanese_bert import get_bert_feature

    # 规范化文本
    text = text_normalize(text)
    print(text)

    # 获取音素、音调和单词到音素的映射关系
    phones, tones, word2ph = g2p(text)
    # 获取BERT特征
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)
```