# `Bert-VITS2\oldVersion\V110\text\japanese.py`

```py
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

# 定义正则表达式，用于匹配非英文字母、空格、逗号、句号、问号的字符
_REJECT_RX = re.compile("[^ a-zA-Z:,.?]")

# 创建规则映射的函数
def _makerulemap():
    # 将转换规则按照斜杠分割成元组，存储在列表中
    l = [tuple(x.split("/")) for x in _CONVRULES]
    # 返回一个元组，包含长度为1和长度为2的键值对字典
    return tuple({k: v for k, v in l if len(k) == i} for i in (1, 2))

# 调用_makerulemap函数，将返回的元组分别赋值给_RULEMAP1和_RULEMAP2
_RULEMAP1, _RULEMAP2 = _makerulemap()

# 定义kata2phoneme函数，接受一个字符串参数，返回一个字符串
def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    # 去除字符串两端的空格
    text = text.strip()
    # 初始化结果列表
    res = []
    # 循环处理文本
    while text:
        # 如果文本长度大于等于2
        if len(text) >= 2:
            # 获取_RULEMAP2中以text[:2]为键的值
            x = _RULEMAP2.get(text[:2])
            # 如果x不为空
            if x is not None:
                # 截取文本前两个字符
                text = text[2:]
                # 将x按空格分割后的列表添加到结果列表中
                res += x.split(" ")[1:]
                # 继续下一次循环
                continue
        # 获取_RULEMAP1中以text[0]为键的值
        x = _RULEMAP1.get(text[0])
        # 如果x不为空
        if x is not None:
            # 截取文本第一个字符
            text = text[1:]
            # 将x按空格分割后的列表添加到结果列表中
            res += x.split(" ")[1:]
            # 继续下一次循环
            continue
        # 将文本第一个字符添加到结果列表中
        res.append(text[0])
        # 截取文本第一个字符
        text = text[1:]
    # 返回结果列表
    return res

# 定义片假名字符范围
_KATAKANA = "".join(chr(ch) for ch in range(ord("ァ"), ord("ン") + 1))
# 定义平假名字符范围
_HIRAGANA = "".join(chr(ch) for ch in range(ord("ぁ"), ord("ん") + 1))
# 创建平假名到片假名的转换映射
_HIRA2KATATRANS = str.maketrans(_HIRAGANA, _KATAKANA)

# 定义平假名到片假名的转换函数，接受一个字符串参数，返回一个字符串
def hira2kata(text: str) -> str:
    # 使用_HIRA2KATATRANS映射将文本中的平假名转换为片假名
    text = text.translate(_HIRA2KATATRANS)
    # 将文本中的"う゛"替换为"ヴ"
    return text.replace("う゛", "ヴ")

# 定义符号标记集合
_SYMBOL_TOKENS = set(list("・、。？！"))
# 定义无读音标记集合
_NO_YOMI_TOKENS = set(list("「」『』―（）［］[]"))
# 创建MeCab的标注器对象
_TAGGER = MeCab.Tagger()

# 定义文本到片假名的转换函数，接受一个字符串参数，返回一个字符串
def text2kata(text: str) -> str:
    # 对文本进行分词和标注
    parsed = _TAGGER.parse(text)
    # 初始化结果列表
    res = []
    # 遍历分词和标注结果
    for line in parsed.split("\n"):
        # 如果遍历到EOS，跳出循环
        if line == "EOS":
            break
        # 将分词和标注结果按制表符分割
        parts = line.split("\t")
        # 获取单词和读音
        word, yomi = parts[0], parts[1]
        # 如果有读音
        if yomi:
            # 将读音添加到结果列表中
            res.append(yomi)
        else:
            # 如果单词是符号标记
            if word in _SYMBOL_TOKENS:
                # 将单词添加到结果列表中
                res.append(word)
            # 如果单词是"っ"或"ッ"
            elif word in ("っ", "ッ"):
                # 将"ッ"添加到结果列表中
                res.append("ッ")
            # 如果单词是无读音标记
            elif word in _NO_YOMI_TOKENS:
                # 跳过
                pass
            else:
                # 将单词添加到结果列表中
                res.append(word)
    # 将结果列表中的平假名转换为片假名，返回结果字符串
    return hira2kata("".join(res))

# 定义特殊符号的读音映射
_ALPHASYMBOL_YOMI = {
    "#": "シャープ",
    "%": "パーセント",
    "&": "アンド",
    "+": "プラス",
    "-": "マイナス",
    ":": "コロン",
    ";": "セミコロン",
    "<": "小なり",
    "=": "イコール",  # 将 "=" 符号翻译为日语中的 "イコール"
    ">": "大なり",    # 将 ">" 符号翻译为日语中的 "大なり"
    "@": "アット",    # 将 "@" 符号翻译为日语中的 "アット"
    "a": "エー",      # 将 "a" 字母翻译为日语中的 "エー"
    "b": "ビー",      # 将 "b" 字母翻译为日语中的 "ビー"
    "c": "シー",      # 将 "c" 字母翻译为日语中的 "シー"
    "d": "ディー",    # 将 "d" 字母翻译为日语中的 "ディー"
    "e": "イー",      # 将 "e" 字母翻译为日语中的 "イー"
    "f": "エフ",      # 将 "f" 字母翻译为日语中的 "エフ"
    "g": "ジー",      # 将 "g" 字母翻译为日语中的 "ジー"
    "h": "エイチ",    # 将 "h" 字母翻译为日语中的 "エイチ"
    "i": "アイ",      # 将 "i" 字母翻译为日语中的 "アイ"
    "j": "ジェー",    # 将 "j" 字母翻译为日语中的 "ジェー"
    "k": "ケー",      # 将 "k" 字母翻译为日语中的 "ケー"
    "l": "エル",      # 将 "l" 字母翻译为日语中的 "エル"
    "m": "エム",      # 将 "m" 字母翻译为日语中的 "エム"
    "n": "エヌ",      # 将 "n" 字母翻译为日语中的 "エヌ"
    "o": "オー",      # 将 "o" 字母翻译为日语中的 "オー"
    "p": "ピー",      # 将 "p" 字母翻译为日语中的 "ピー"
    "q": "キュー",    # 将 "q" 字母翻译为日语中的 "キュー"
    "r": "アール",    # 将 "r" 字母翻译为日语中的 "アール"
    "s": "エス",      # 将 "s" 字母翻译为日语中的 "エス"
    "t": "ティー",    # 将 "t" 字母翻译为日语中的 "ティー"
    "u": "ユー",      # 将 "u" 字母翻译为日语中的 "ユー"
    "v": "ブイ",      # 将 "v" 字母翻译为日语中的 "ブイ"
    "w": "ダブリュー",  # 将 "w" 字母翻译为日语中的 "ダブリュー"
    "x": "エックス",    # 将 "x" 字母翻译为日语中的 "エックス"
    "y": "ワイ",        # 将 "y" 字母翻译为日语中的 "ワイ"
    "z": "ゼット",      # 将 "z" 字母翻译为日语中的 "ゼット"
    "α": "アルファ",    # 将 "α" 符号翻译为日语中的 "アルファ"
    "β": "ベータ",      # 将 "β" 符号翻译为日语中的 "ベータ"
    "γ": "ガンマ",      # 将 "γ" 符号翻译为日语中的 "ガンマ"
    "δ": "デルタ",      # 将 "δ" 符号翻译为日语中的 "デルタ"
    "ε": "イプシロン",  # 将 "ε" 符号翻译为日语中的 "イプシロン"
    "ζ": "ゼータ",      # 将 "ζ" 符号翻译为日语中的 "ゼータ"
    "η": "イータ",      # 将 "η" 符号翻译为日语中的 "イータ"
    "θ": "シータ",      # 将 "θ" 符号翻译为日语中的 "シータ"
    "ι": "イオタ",      # 将 "ι" 符号翻译为日语中的 "イオタ"
    "κ": "カッパ",      # 将 "κ" 符号翻译为日语中的 "カッパ"
    "λ": "ラムダ",      # 将 "λ" 符号翻译为日语中的 "ラムダ"
    "μ": "ミュー",      # 将 "μ" 符号翻译为日语中的 "ミュー"
    "ν": "ニュー",      # 将 "ν" 符号翻译为日语中的 "ニュー"
    "ξ": "クサイ",      # 将 "ξ" 符号翻译为日语中的 "クサイ"
    "ο": "オミクロン",  # 将 "ο" 符号翻译为日语中的 "オミクロン"
    "π": "パイ",        # 将 "π" 符号翻译为日语中的 "パイ"
    "ρ": "ロー",        # 将 "ρ" 符号翻译为日语中的 "ロー"
    "σ": "シグマ",      # 将 "σ" 符号翻译为日语中的 "シグマ"
    "τ": "タウ",        # 将 "τ" 符号翻译为日语中的 "タウ"
    "υ": "ウプシロン",  # 将 "υ" 符号翻译为日语中的 "ウプシロン"
    "φ": "ファイ",      # 将 "φ" 符号翻译为日语中的 "ファイ"
    "χ": "カイ",        # 将 "χ" 符号翻译为日语中的 "カイ"
    "ψ": "プサイ",      # 将 "ψ" 符号翻译为日语中的 "プサイ"
    "ω": "オメガ",      # 将 "ω" 符号翻译为日语中的 "オメガ"
# 定义一个正则表达式，用于匹配带有逗号分隔符的数字
_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
# 定义一个货币符号到日语货币名称的映射字典
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
# 定义一个正则表达式，用于匹配货币符号和金额
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
# 定义一个正则表达式，用于匹配数字
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")

# 将文本中的数字转换为对应的日语单词
def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res

# 将文本中的字母和符号转换为对应的日语单词
def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])

# 将日语文本转换为音素
def japanese_text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phonemes."""
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = japanese_convert_alpha_symbols_to_words(res)
    res = text2kata(res)
    res = kata2phoneme(res)
    return res

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
    "·": ",",
    "、": ",",
    "...": "…",
}

# 替换文本中的标点符号
def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
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
# 对文本进行规范化处理，包括Unicode标准化和将数字转换为日文单词
def text_normalize(text):
    # 使用Unicode标准化方法"NFKC"对文本进行规范化处理
    res = unicodedata.normalize("NFKC", text)
    # 将文本中的数字转换为日文单词
    res = japanese_convert_numbers_to_words(res)
    # 替换文本中的标点符号
    res = replace_punctuation(res)
    return res


# 根据电话数量和单词数量分配电话
def distribute_phone(n_phone, n_word):
    # 初始化每个单词的电话数量列表
    phones_per_word = [0] * n_word
    # 遍历每个电话任务
    for task in range(n_phone):
        # 找到电话数量最少的单词
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        # 将电话分配给数量最少的单词
        phones_per_word[min_index] += 1
    return phones_per_word


# 从预训练模型中加载分词器
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")


# 将文本转换为音素
def g2p(norm_text):
    # 使用分词器对文本进行分词
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    ph_groups = []
    # 遍历分词结果
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    word2ph = []
    # 遍历分词后的组合
    for group in ph_groups:
        # 将日文转换为假名，再将假名转换为音素
        phonemes = kata2phoneme(text2kata("".join(group)))
        # 检查音素是否在符号列表中
        for i in phonemes:
            assert i in symbols, (group, norm_text, tokenized)
        phone_len = len(phonemes)
        word_len = len(group)
        # 分配电话给单词
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
        phs += phonemes
    phones = ["_"] + phs + ["_"]
    tones = [0 for i in phones]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


# 主函数
if __name__ == "__main__":
    # 从预训练模型中加载分词器
    tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")
    # 待处理的文本
    text = "hello,こんにちは、世界！……"
    # 导入获取BERT特征的函数
    from text.japanese_bert import get_bert_feature
    # 对文本进行规范化处理
    text = text_normalize(text)
    print(text)
    # 将文本转换为音素
    phones, tones, word2ph = g2p(text)
    # 获取BERT特征
    bert = get_bert_feature(text, word2ph)
    # 打印结果
    print(phones, tones, word2ph, bert.shape)
```