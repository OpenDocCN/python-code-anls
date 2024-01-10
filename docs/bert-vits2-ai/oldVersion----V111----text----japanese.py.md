# `Bert-VITS2\oldVersion\V111\text\japanese.py`

```
# 导入正则表达式和Unicode数据处理模块
import re
import unicodedata

# 从transformers模块中导入AutoTokenizer类
from transformers import AutoTokenizer

# 从当前目录下的punctuation和symbols模块中导入相关内容
from . import punctuation, symbols

# 尝试导入MeCab模块，如果导入失败则抛出ImportError异常
try:
    import MeCab
except ImportError as e:
    raise ImportError("Japanese requires mecab-python3 and unidic-lite.") from e

# 从num2words模块中导入num2words函数
from num2words import num2words

# 定义日语文本到音素的转换规则列表
_CONVRULES = [
    # 两个字母的转换规则
    "アァ/ a a",
    "イィ/ i i",
    "イェ/ i e",
    "イャ/ y a",
    "ウゥ/ u:",
    "エェ/ e e",
    "オォ/ o:",
    "カァ/ k a:",
    "キィ/ k i:",
    "クゥ/ k u:",
    "クャ/ ky a",
    "クュ/ ky u",
    "クョ/ ky o",
    "ケェ/ k e:",
    "コォ/ k o:",
    # ... 省略部分转换规则 ...
    "ムゥ/ m u:",
    "ムャ/ my a",
    "ムュ/ my u",
    "ムョ/ my o",
    "メェ/ m e:",
    # 日文假名对应的罗马音
    "モォ/ m o:",
    "ヤァ/ y a:",
    "ユゥ/ y u:",
    "ユャ/ y a:",
    "ユュ/ y u:",
    "ユョ/ y o:",
    "ヨォ/ y o:",
    "ラァ/ r a:",
    "リィ/ r i:",
    "ルゥ/ r u:",
    "ルャ/ ry a",
    "ルュ/ ry u",
    "ルョ/ ry o",
    "レェ/ r e:",
    "ロォ/ r o:",
    "ワァ/ w a:",
    "ヲォ/ o:",
    "ディ/ d i",
    "デェ/ d e:",
    "デャ/ dy a",
    "デュ/ dy u",
    "デョ/ dy o",
    "ティ/ t i",
    "テェ/ t e:",
    "テャ/ ty a",
    "テュ/ ty u",
    "テョ/ ty o",
    "スィ/ s i",
    "ズァ/ z u a",
    "ズィ/ z i",
    "ズゥ/ z u",
    "ズャ/ zy a",
    "ズュ/ zy u",
    "ズョ/ zy o",
    "ズェ/ z e",
    "ズォ/ z o",
    "キャ/ ky a",
    "キュ/ ky u",
    "キョ/ ky o",
    "シャ/ sh a",
    "シュ/ sh u",
    "シェ/ sh e",
    "ショ/ sh o",
    "チャ/ ch a",
    "チュ/ ch u",
    "チェ/ ch e",
    "チョ/ ch o",
    "トゥ/ t u",
    "トャ/ ty a",
    "トュ/ ty u",
    "トョ/ ty o",
    "ドァ/ d o a",
    "ドゥ/ d u",
    "ドャ/ dy a",
    "ドュ/ dy u",
    "ドョ/ dy o",
    "ドォ/ d o:",
    "ニャ/ ny a",
    "ニュ/ ny u",
    "ニョ/ ny o",
    "ヒャ/ hy a",
    "ヒュ/ hy u",
    "ヒョ/ hy o",
    "ミャ/ my a",
    "ミュ/ my u",
    "ミョ/ my o",
    "リャ/ ry a",
    "リュ/ ry u",
    "リョ/ ry o",
    "ギャ/ gy a",
    "ギュ/ gy u",
    "ギョ/ gy o",
    "ヂェ/ j e",
    "ヂャ/ j a",
    "ヂュ/ j u",
    "ヂョ/ j o",
    "ジェ/ j e",
    "ジャ/ j a",
    "ジュ/ j u",
    "ジョ/ j o",
    "ビャ/ by a",
    "ビュ/ by u",
    "ビョ/ by o",
    "ピャ/ py a",
    "ピュ/ py u",
    "ピョ/ py o",
    "ウァ/ u a",
    "ウィ/ w i",
    "ウェ/ w e",
    "ウォ/ w o",
    "ファ/ f a",
    "フィ/ f i",
    "フゥ/ f u",
    "フャ/ hy a",
    "フュ/ hy u",
    "フョ/ hy o",
    "フェ/ f e",
    "フォ/ f o",
    "ヴァ/ b a",
    "ヴィ/ b i",
    "ヴェ/ b e",
    "ヴォ/ b o",
    "ヴュ/ by u",
    # 单个日文假名对应的罗马音
    "ア/ a",
    "イ/ i",
    "ウ/ u",
    "エ/ e",
    "オ/ o",
    "カ/ k a",
    "キ/ k i",
    "ク/ k u",
    "ケ/ k e",
    "コ/ k o",
    "サ/ s a",
    "シ/ sh i",
    "ス/ s u",
    "セ/ s e",
    "ソ/ s o",
    "タ/ t a",
    "チ/ ch i",
    "ツ/ ts u",
    "テ/ t e",
    "ト/ t o",
    "ナ/ n a",
    "ニ/ n i",
    "ヌ/ n u",
    "ネ/ n e",
    "ノ/ n o",
    "ハ/ h a",  # 日文字符对应的罗马音
    "ヒ/ h i",  # 日文字符对应的罗马音
    "フ/ f u",  # 日文字符对应的罗马音
    "ヘ/ h e",  # 日文字符对应的罗马音
    "ホ/ h o",  # 日文字符对应的罗马音
    "マ/ m a",  # 日文字符对应的罗马音
    "ミ/ m i",  # 日文字符对应的罗马音
    "ム/ m u",  # 日文字符对应的罗马音
    "メ/ m e",  # 日文字符对应的罗马音
    "モ/ m o",  # 日文字符对应的罗马音
    "ラ/ r a",  # 日文字符对应的罗马音
    "リ/ r i",  # 日文字符对应的罗马音
    "ル/ r u",  # 日文字符对应的罗马音
    "レ/ r e",  # 日文字符对应的罗马音
    "ロ/ r o",  # 日文字符对应的罗马音
    "ガ/ g a",  # 日文字符对应的罗马音
    "ギ/ g i",  # 日文字符对应的罗马音
    "グ/ g u",  # 日文字符对应的罗马音
    "ゲ/ g e",  # 日文字符对应的罗马音
    "ゴ/ g o",  # 日文字符对应的罗马音
    "ザ/ z a",  # 日文字符对应的罗马音
    "ジ/ j i",  # 日文字符对应的罗马音
    "ズ/ z u",  # 日文字符对应的罗马音
    "ゼ/ z e",  # 日文字符对应的罗马音
    "ゾ/ z o",  # 日文字符对应的罗马音
    "ダ/ d a",  # 日文字符对应的罗马音
    "ヂ/ j i",  # 日文字符对应的罗马音
    "ヅ/ z u",  # 日文字符对应的罗马音
    "デ/ d e",  # 日文字符对应的罗马音
    "ド/ d o",  # 日文字符对应的罗马音
    "バ/ b a",  # 日文字符对应的罗马音
    "ビ/ b i",  # 日文字符对应的罗马音
    "ブ/ b u",  # 日文字符对应的罗马音
    "ベ/ b e",  # 日文字符对应的罗马音
    "ボ/ b o",  # 日文字符对应的罗马音
    "パ/ p a",  # 日文字符对应的罗马音
    "ピ/ p i",  # 日文字符对应的罗马音
    "プ/ p u",  # 日文字符对应的罗马音
    "ペ/ p e",  # 日文字符对应的罗马音
    "ポ/ p o",  # 日文字符对应的罗马音
    "ヤ/ y a",  # 日文字符对应的罗马音
    "ユ/ y u",  # 日文字符对应的罗马音
    "ヨ/ y o",  # 日文字符对应的罗马音
    "ワ/ w a",  # 日文字符对应的罗马音
    "ヰ/ i",    # 日文字符对应的罗马音
    "ヱ/ e",    # 日文字符对应的罗马音
    "ヲ/ o",    # 日文字符对应的罗马音
    "ン/ N",    # 日文字符对应的罗马音
    "ッ/ q",    # 日文字符对应的罗马音
    "ヴ/ b u",  # 日文字符对应的罗马音
    "ー/:",      # 日文字符对应的罗马音
    # 尝试转换损坏的文本
    "ァ/ a",    # 日文字符对应的罗马音
    "ィ/ i",    # 日文字符对应的罗马音
    "ゥ/ u",    # 日文字符对应的罗马音
    "ェ/ e",    # 日文字符对应的罗马音
    "ォ/ o",    # 日文字符对应的罗马音
    "ヮ/ w a",  # 日文字符对应的罗马音
    "ォ/ o",    # 日文字符对应的罗马音
    # 符号
    "、/ ,",    # 日文字符对应的英文符号
    "。/ .",    # 日文字符对应的英文符号
    "！/ !",    # 日文字符对应的英文符号
    "？/ ?",    # 日文字符对应的英文符号
    "・/ ,",    # 日文字符对应的英文符号
# 导入正则表达式模块
import re

# 定义正则表达式，用于匹配冒号
_COLON_RX = re.compile(":+")
# 定义正则表达式，用于匹配非日语字符
_REJECT_RX = re.compile("[^ a-zA-Z:,.?]")

# 创建规则映射
def _makerulemap():
    # 将转换规则按照斜杠分割成元组，存储在列表中
    l = [tuple(x.split("/")) for x in _CONVRULES]
    # 将列表中的元组按照元组长度分组，存储在字典中
    return tuple({k: v for k, v in l if len(k) == i} for i in (1, 2))

# 获取规则映射
_RULEMAP1, _RULEMAP2 = _makerulemap()

# 将片假名文本转换为音素
def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    # 去除文本两端的空白字符
    text = text.strip()
    # 初始化结果列表
    res = []
    # 循环处理文本
    while text:
        if len(text) >= 2:
            # 获取两个字符的转换规则
            x = _RULEMAP2.get(text[:2])
            if x is not None:
                # 截取已处理的字符，将转换结果添加到结果列表中
                text = text[2:]
                res += x.split(" ")[1:]
                continue
        # 获取单个字符的转换规则
        x = _RULEMAP1.get(text[0])
        if x is not None:
            # 截取已处理的字符，将转换结果添加到结果列表中
            text = text[1:]
            res += x.split(" ")[1:]
            continue
        # 将未匹配到的字符添加到结果列表中
        res.append(text[0])
        text = text[1:]
    # 返回转换结果
    return res

# 定义片假名和平假名的 Unicode 范围
_KATAKANA = "".join(chr(ch) for ch in range(ord("ァ"), ord("ン") + 1))
_HIRAGANA = "".join(chr(ch) for ch in range(ord("ぁ"), ord("ん") + 1))
# 创建平假名到片假名的转换映射
_HIRA2KATATRANS = str.maketrans(_HIRAGANA, _KATAKANA)

# 将平假名文本转换为片假名
def hira2kata(text: str) -> str:
    # 使用转换映射进行文本转换
    text = text.translate(_HIRA2KATATRANS)
    # 替换特定的片假名字符
    return text.replace("う゛", "ヴ")

# 定义一些特殊符号的集合
_SYMBOL_TOKENS = set(list("・、。？！"))
_NO_YOMI_TOKENS = set(list("「」『』―（）［］[]"))
# 创建 MeCab 分词器对象
_TAGGER = MeCab.Tagger()

# 将文本转换为片假名
def text2kata(text: str) -> str:
    # 对文本进行分词
    parsed = _TAGGER.parse(text)
    # 初始化结果列表
    res = []
    # 遍历分词结果
    for line in parsed.split("\n"):
        if line == "EOS":
            break
        parts = line.split("\t")
        # 获取单词和对应的读音
        word, yomi = parts[0], parts[1]
        if yomi:
            # 如果有读音，将读音添加到结果列表中
            res.append(yomi)
        else:
            if word in _SYMBOL_TOKENS:
                # 如果是特殊符号，将符号添加到结果列表中
                res.append(word)
            elif word in ("っ", "ッ"):
                # 如果是小つ，将小つ添加到结果列表中
                res.append("ッ")
            elif word in _NO_YOMI_TOKENS:
                # 如果是不需要读音的特殊符号，跳过
                pass
            else:
                # 其他情况，将单词添加到结果列表中
                res.append(word)
    # 将结果列表中的平假名文本转换为片假名，返回结果
    return hira2kata("".join(res))

# 定义一些特殊符号的读音映射
_ALPHASYMBOL_YOMI = {
    "#": "シャープ",
    "%": "パーセント",
    "&": "アンド",
    "+": "プラス",
    "-": "マイナス",
    ":": "コロン",
    ";": "セミコロン",
    "<": "小なり",
    "=": "イコール",  # 将 "=" 符号翻译为日语，意为“イコール”
    ">": "大なり",  # 将 ">" 符号翻译为日语，意为“大なり”
    "@": "アット",  # 将 "@" 符号翻译为日语，意为“アット”
    "a": "エー",  # 将小写字母 "a" 翻译为日语，意为“エー”
    "b": "ビー",  # 将小写字母 "b" 翻译为日语，意为“ビー”
    "c": "シー",  # 将小写字母 "c" 翻译为日语，意为“シー”
    "d": "ディー",  # 将小写字母 "d" 翻译为日语，意为“ディー”
    "e": "イー",  # 将小写字母 "e" 翻译为日语，意为“イー”
    "f": "エフ",  # 将小写字母 "f" 翻译为日语，意为“エフ”
    "g": "ジー",  # 将小写字母 "g" 翻译为日语，意为“ジー”
    "h": "エイチ",  # 将小写字母 "h" 翻译为日语，意为“エイチ”
    "i": "アイ",  # 将小写字母 "i" 翻译为日语，意为“アイ”
    "j": "ジェー",  # 将小写字母 "j" 翻译为日语，意为“ジェー”
    "k": "ケー",  # 将小写字母 "k" 翻译为日语，意为“ケー”
    "l": "エル",  # 将小写字母 "l" 翻译为日语，意为“エル”
    "m": "エム",  # 将小写字母 "m" 翻译为日语，意为“エム”
    "n": "エヌ",  # 将小写字母 "n" 翻译为日语，意为“エヌ”
    "o": "オー",  # 将小写字母 "o" 翻译为日语，意为“オー”
    "p": "ピー",  # 将小写字母 "p" 翻译为日语，意为“ピー”
    "q": "キュー",  # 将小写字母 "q" 翻译为日语，意为“キュー”
    "r": "アール",  # 将小写字母 "r" 翻译为日语，意为“アール”
    "s": "エス",  # 将小写字母 "s" 翻译为日语，意为“エス”
    "t": "ティー",  # 将小写字母 "t" 翻译为日语，意为“ティー”
    "u": "ユー",  # 将小写字母 "u" 翻译为日语，意为“ユー”
    "v": "ブイ",  # 将小写字母 "v" 翻译为日语，意为“ブイ”
    "w": "ダブリュー",  # 将小写字母 "w" 翻译为日语，意为“ダブリュー”
    "x": "エックス",  # 将小写字母 "x" 翻译为日语，意为“エックス”
    "y": "ワイ",  # 将小写字母 "y" 翻译为日语，意为“ワイ”
    "z": "ゼット",  # 将小写字母 "z" 翻译为日语，意为“ゼット”
    "α": "アルファ",  # 将希腊字母 "α" 翻译为日语，意为“アルファ”
    "β": "ベータ",  # 将希腊字母 "β" 翻译为日语，意为“ベータ”
    "γ": "ガンマ",  # 将希腊字母 "γ" 翻译为日语，意为“ガンマ”
    "δ": "デルタ",  # 将希腊字母 "δ" 翻译为日语，意为“デルタ”
    "ε": "イプシロン",  # 将希腊字母 "ε" 翻译为日语，意为“イプシロン”
    "ζ": "ゼータ",  # 将希腊字母 "ζ" 翻译为日语，意为“ゼータ”
    "η": "イータ",  # 将希腊字母 "η" 翻译为日语，意为“イータ”
    "θ": "シータ",  # 将希腊字母 "θ" 翻译为日语，意为“シータ”
    "ι": "イオタ",  # 将希腊字母 "ι" 翻译为日语，意为“イオタ”
    "κ": "カッパ",  # 将希腊字母 "κ" 翻译为日语，意为“カッパ”
    "λ": "ラムダ",  # 将希腊字母 "λ" 翻译为日语，意为“ラムダ”
    "μ": "ミュー",  # 将希腊字母 "μ" 翻译为日语，意为“ミュー”
    "ν": "ニュー",  # 将希腊字母 "ν" 翻译为日语，意为“ニュー”
    "ξ": "クサイ",  # 将希腊字母 "ξ" 翻译为日语，意为“クサイ”
    "ο": "オミクロン",  # 将希腊字母 "ο" 翻译为日语，意为“オミクロン”
    "π": "パイ",  # 将希腊字母 "π" 翻译为日语，意为“パイ”
    "ρ": "ロー",  # 将希腊字母 "ρ" 翻译为日语，意为“ロー”
    "σ": "シグマ",  # 将希腊字母 "σ" 翻译为日语，意为“シグマ”
    "τ": "タウ",  # 将希腊字母 "τ" 翻译为日语，意为“タウ”
    "υ": "ウプシロン",  # 将希腊字母 "υ" 翻译为日语，意为“ウプシロン”
    "φ": "ファイ",  # 将希腊字母 "φ" 翻译为日语，意为“ファイ”
    "χ": "カイ",  # 将希腊字母 "χ" 翻译为日语，意为“カイ”
    "ψ": "プサイ",  # 将希腊字母 "ψ" 翻译为日语，意为“プサイ”
    "ω": "オメガ",  # 将希腊字母 "ω" 翻译为日语，意为“オメガ”
# 定义一个正则表达式，用于匹配带有千位分隔符的数字
_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
# 定义一个货币符号到日语货币名称的映射字典
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
# 定义一个正则表达式，用于匹配货币符号和金额
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
# 定义一个正则表达式，用于匹配数字
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")

# 将输入的文本中的数字转换为日语读法
def japanese_convert_numbers_to_words(text: str) -> str:
    # 使用正则表达式替换函数，去除数字中的千位分隔符
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    # 使用正则表达式替换函数，将货币符号和金额转换为日语读法
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    # 使用正则表达式替换函数，将数字转换为日语读法
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res

# 将输入的文本中的英文字母和符号转换为日语读法
def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])

# 将输入的日语文本转换为音素
def japanese_text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phonemes."""
    # 使用 unicodedata.normalize 函数将文本规范化为 NFKC 格式
    res = unicodedata.normalize("NFKC", text)
    # 将文本中的数字转换为日语读法
    res = japanese_convert_numbers_to_words(res)
    # 将文本中的英文字母和符号转换为日语读法
    # res = japanese_convert_alpha_symbols_to_words(res)
    # 将文本转换为片假名
    res = text2kata(res)
    # 将片假名转换为音素
    res = kata2phoneme(res)
    return res

# 判断输入的字符是否为日语字符
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

# 定义一个用于替换标点符号的映射字典
rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
}

# 将输入文本中的标点符号替换为日语标点符号
def replace_punctuation(text):
    # 使用 re.escape 函数对映射字典中的键进行转义，构建正则表达式模式
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    # 使用正则表达式替换函数，将文本中的标点符号替换为日语标点符号
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    # 使用正则表达式替换文本中除了日文、片假名、中文和标点符号之外的所有字符
    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF"
        + "".join(punctuation)
        + r"]+",
        "",
        replaced_text,
    )
    # 返回替换后的文本
    return replaced_text
# 对文本进行规范化处理，包括Unicode标准化、将数字转换为对应的日文单词、替换标点符号
def text_normalize(text):
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = "".join([i for i in res if is_japanese_character(i)])
    res = replace_punctuation(res)
    return res


# 分配电话号码给单词
def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


# 从预训练模型中加载分词器
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")


# 将文本转换为音素
def g2p(norm_text):
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    word2ph = []
    for group in ph_groups:
        phonemes = kata2phoneme(text2kata("".join(group)))
        # phonemes = [i for i in phonemes if i in symbols]
        for i in phonemes:
            assert i in symbols, (group, norm_text, tokenized)
        phone_len = len(phonemes)
        word_len = len(group)

        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa

        phs += phonemes
    phones = ["_"] + phs + ["_"]
    tones = [0 for i in phones]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


# 主函数
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")
    text = "hello,こんにちは、世界！……"
    from text.japanese_bert import get_bert_feature

    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)
```