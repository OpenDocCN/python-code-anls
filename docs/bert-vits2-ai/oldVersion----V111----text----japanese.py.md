# `d:/src/tocomm/Bert-VITS2\oldVersion\V111\text\japanese.py`

```
# 导入必要的模块和库
import re  # 用于正则表达式操作
import unicodedata  # 用于Unicode字符的操作
from transformers import AutoTokenizer  # 用于自然语言处理任务的模型
from . import punctuation, symbols  # 导入自定义的标点符号和符号

# 尝试导入MeCab模块，如果导入失败则抛出ImportError异常
try:
    import MeCab  # 用于日语分词
except ImportError as e:
    raise ImportError("Japanese requires mecab-python3 and unidic-lite.") from e

# 导入num2words模块，用于将数字转换为单词
from num2words import num2words

# 定义一系列的转换规则，将日语文本转换为与Julius兼容的音素
_CONVRULES = [
    # 2个字母的转换
    "アァ/ a a",
    "イィ/ i i",
    "イェ/ i e",
    ...
]
    "イャ/ y a",  # 将字符串"イャ"映射为"y a"
    "ウゥ/ u:",  # 将字符串"ウゥ"映射为"u:"
    "エェ/ e e",  # 将字符串"エェ"映射为"e e"
    "オォ/ o:",  # 将字符串"オォ"映射为"o:"
    "カァ/ k a:",  # 将字符串"カァ"映射为"k a:"
    "キィ/ k i:",  # 将字符串"キィ"映射为"k i:"
    "クゥ/ k u:",  # 将字符串"クゥ"映射为"k u:"
    "クャ/ ky a",  # 将字符串"クャ"映射为"ky a"
    "クュ/ ky u",  # 将字符串"クュ"映射为"ky u"
    "クョ/ ky o",  # 将字符串"クョ"映射为"ky o"
    "ケェ/ k e:",  # 将字符串"ケェ"映射为"k e:"
    "コォ/ k o:",  # 将字符串"コォ"映射为"k o:"
    "ガァ/ g a:",  # 将字符串"ガァ"映射为"g a:"
    "ギィ/ g i:",  # 将字符串"ギィ"映射为"g i:"
    "グゥ/ g u:",  # 将字符串"グゥ"映射为"g u:"
    "グャ/ gy a",  # 将字符串"グャ"映射为"gy a"
    "グュ/ gy u",  # 将字符串"グュ"映射为"gy u"
    "グョ/ gy o",  # 将字符串"グョ"映射为"gy o"
    "ゲェ/ g e:",  # 将字符串"ゲェ"映射为"g e:"
    "ゴォ/ g o:",  # 将字符串"ゴォ"映射为"g o:"
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

需要注释的代码：

```
    "サァ/ s a:",
    "シィ/ sh i:",
    "スゥ/ s u:",
    "スャ/ sh a",
    "スュ/ sh u",
    "スョ/ sh o",
    "セェ/ s e:",
    "ソォ/ s o:",
    "ザァ/ z a:",
    "ジィ/ j i:",
    "ズゥ/ z u:",
    "ズャ/ zy a",
    "ズュ/ zy u",
    "ズョ/ zy o",
    "ゼェ/ z e:",
    "ゾォ/ z o:",
    "タァ/ t a:",
    "チィ/ ch i:",
    "ツァ/ ts a",
    "ツィ/ ts i",
```

注释：

这部分代码是一个字符串列表，包含了一些日语假名和对应的发音。这些字符串可能是用于某种文本处理或转换的数据。
    "ツゥ/ ts u:",  # 表示字符"ツゥ"对应的发音是"ts u:"
    "ツャ/ ch a",  # 表示字符"ツャ"对应的发音是"ch a"
    "ツュ/ ch u",  # 表示字符"ツュ"对应的发音是"ch u"
    "ツョ/ ch o",  # 表示字符"ツョ"对应的发音是"ch o"
    "ツェ/ ts e",  # 表示字符"ツェ"对应的发音是"ts e"
    "ツォ/ ts o",  # 表示字符"ツォ"对应的发音是"ts o"
    "テェ/ t e:",  # 表示字符"テェ"对应的发音是"t e:"
    "トォ/ t o:",  # 表示字符"トォ"对应的发音是"t o:"
    "ダァ/ d a:",  # 表示字符"ダァ"对应的发音是"d a:"
    "ヂィ/ j i:",  # 表示字符"ヂィ"对应的发音是"j i:"
    "ヅゥ/ d u:",  # 表示字符"ヅゥ"对应的发音是"d u:"
    "ヅャ/ zy a",  # 表示字符"ヅャ"对应的发音是"zy a"
    "ヅュ/ zy u",  # 表示字符"ヅュ"对应的发音是"zy u"
    "ヅョ/ zy o",  # 表示字符"ヅョ"对应的发音是"zy o"
    "デェ/ d e:",  # 表示字符"デェ"对应的发音是"d e:"
    "ドォ/ d o:",  # 表示字符"ドォ"对应的发音是"d o:"
    "ナァ/ n a:",  # 表示字符"ナァ"对应的发音是"n a:"
    "ニィ/ n i:",  # 表示字符"ニィ"对应的发音是"n i:"
    "ヌゥ/ n u:",  # 表示字符"ヌゥ"对应的发音是"n u:"
    "ヌャ/ ny a",  # 表示字符"ヌャ"对应的发音是"ny a"
    "ヌュ/ ny u",  # 表示"nyu"的发音
    "ヌョ/ ny o",  # 表示"nyo"的发音
    "ネェ/ n e:",  # 表示"ne"的发音
    "ノォ/ n o:",  # 表示"no"的发音
    "ハァ/ h a:",  # 表示"ha"的发音
    "ヒィ/ h i:",  # 表示"hi"的发音
    "フゥ/ f u:",  # 表示"fu"的发音
    "フャ/ hy a",  # 表示"hya"的发音
    "フュ/ hy u",  # 表示"hyu"的发音
    "フョ/ hy o",  # 表示"hyo"的发音
    "ヘェ/ h e:",  # 表示"he"的发音
    "ホォ/ h o:",  # 表示"ho"的发音
    "バァ/ b a:",  # 表示"ba"的发音
    "ビィ/ b i:",  # 表示"bi"的发音
    "ブゥ/ b u:",  # 表示"bu"的发音
    "フャ/ hy a",  # 表示"hya"的发音
    "ブュ/ by u",  # 表示"byu"的发音
    "フョ/ hy o",  # 表示"hyo"的发音
    "ベェ/ b e:",  # 表示"be"的发音
    "ボォ/ b o:",  # 表示"bo"的发音
```

这段代码是一系列字符串，每个字符串表示一个特定的发音。这些字符串可能是用于某种文本转语音的程序中的输入参数。注释解释了每个字符串表示的发音。
    "パァ/ p a:",  # 创建一个字符串键值对，键为"パァ"，值为"p a:"
    "ピィ/ p i:",  # 创建一个字符串键值对，键为"ピィ"，值为"p i:"
    "プゥ/ p u:",  # 创建一个字符串键值对，键为"プゥ"，值为"p u:"
    "プャ/ py a",  # 创建一个字符串键值对，键为"プャ"，值为"py a"
    "プュ/ py u",  # 创建一个字符串键值对，键为"プュ"，值为"py u"
    "プョ/ py o",  # 创建一个字符串键值对，键为"プョ"，值为"py o"
    "ペェ/ p e:",  # 创建一个字符串键值对，键为"ペェ"，值为"p e:"
    "ポォ/ p o:",  # 创建一个字符串键值对，键为"ポォ"，值为"p o:"
    "マァ/ m a:",  # 创建一个字符串键值对，键为"マァ"，值为"m a:"
    "ミィ/ m i:",  # 创建一个字符串键值对，键为"ミィ"，值为"m i:"
    "ムゥ/ m u:",  # 创建一个字符串键值对，键为"ムゥ"，值为"m u:"
    "ムャ/ my a",  # 创建一个字符串键值对，键为"ムャ"，值为"my a"
    "ムュ/ my u",  # 创建一个字符串键值对，键为"ムュ"，值为"my u"
    "ムョ/ my o",  # 创建一个字符串键值对，键为"ムョ"，值为"my o"
    "メェ/ m e:",  # 创建一个字符串键值对，键为"メェ"，值为"m e:"
    "モォ/ m o:",  # 创建一个字符串键值对，键为"モォ"，值为"m o:"
    "ヤァ/ y a:",  # 创建一个字符串键值对，键为"ヤァ"，值为"y a:"
    "ユゥ/ y u:",  # 创建一个字符串键值对，键为"ユゥ"，值为"y u:"
    "ユャ/ y a:",  # 创建一个字符串键值对，键为"ユャ"，值为"y a:"
    "ユュ/ y u:",  # 创建一个字符串键值对，键为"ユュ"，值为"y u:"
    "ユョ/ y o:",  # 创建一个键为"ユョ"，值为"y o:"的字符串
    "ヨォ/ y o:",  # 创建一个键为"ヨォ"，值为"y o:"的字符串
    "ラァ/ r a:",  # 创建一个键为"ラァ"，值为"r a:"的字符串
    "リィ/ r i:",  # 创建一个键为"リィ"，值为"r i:"的字符串
    "ルゥ/ r u:",  # 创建一个键为"ルゥ"，值为"r u:"的字符串
    "ルャ/ ry a",  # 创建一个键为"ルャ"，值为"ry a"的字符串
    "ルュ/ ry u",  # 创建一个键为"ルュ"，值为"ry u"的字符串
    "ルョ/ ry o",  # 创建一个键为"ルョ"，值为"ry o"的字符串
    "レェ/ r e:",  # 创建一个键为"レェ"，值为"r e:"的字符串
    "ロォ/ r o:",  # 创建一个键为"ロォ"，值为"r o:"的字符串
    "ワァ/ w a:",  # 创建一个键为"ワァ"，值为"w a:"的字符串
    "ヲォ/ o:",    # 创建一个键为"ヲォ"，值为"o:"的字符串
    "ディ/ d i",   # 创建一个键为"ディ"，值为"d i"的字符串
    "デェ/ d e:",  # 创建一个键为"デェ"，值为"d e:"的字符串
    "デャ/ dy a",  # 创建一个键为"デャ"，值为"dy a"的字符串
    "デュ/ dy u",  # 创建一个键为"デュ"，值为"dy u"的字符串
    "デョ/ dy o",  # 创建一个键为"デョ"，值为"dy o"的字符串
    "ティ/ t i",   # 创建一个键为"ティ"，值为"t i"的字符串
    "テェ/ t e:",  # 创建一个键为"テェ"，值为"t e:"的字符串
    "テャ/ ty a",  # 创建一个键为"テャ"，值为"ty a"的字符串
    "テュ/ ty u",  # 定义一个字符串 "テュ/ ty u"
    "テョ/ ty o",  # 定义一个字符串 "テョ/ ty o"
    "スィ/ s i",   # 定义一个字符串 "スィ/ s i"
    "ズァ/ z u a",  # 定义一个字符串 "ズァ/ z u a"
    "ズィ/ z i",   # 定义一个字符串 "ズィ/ z i"
    "ズゥ/ z u",   # 定义一个字符串 "ズゥ/ z u"
    "ズャ/ zy a",  # 定义一个字符串 "ズャ/ zy a"
    "ズュ/ zy u",  # 定义一个字符串 "ズュ/ zy u"
    "ズョ/ zy o",  # 定义一个字符串 "ズョ/ zy o"
    "ズェ/ z e",   # 定义一个字符串 "ズェ/ z e"
    "ズォ/ z o",   # 定义一个字符串 "ズォ/ z o"
    "キャ/ ky a",  # 定义一个字符串 "キャ/ ky a"
    "キュ/ ky u",  # 定义一个字符串 "キュ/ ky u"
    "キョ/ ky o",  # 定义一个字符串 "キョ/ ky o"
    "シャ/ sh a",  # 定义一个字符串 "シャ/ sh a"
    "シュ/ sh u",  # 定义一个字符串 "シュ/ sh u"
    "シェ/ sh e",  # 定义一个字符串 "シェ/ sh e"
    "ショ/ sh o",  # 定义一个字符串 "ショ/ sh o"
    "チャ/ ch a",  # 定义一个字符串 "チャ/ ch a"
    "チュ/ ch u",  # 定义一个字符串 "チュ/ ch u"
    "チェ/ ch e",  # 定义一个键值对，键为"チェ"，值为"ch e"
    "チョ/ ch o",  # 定义一个键值对，键为"チョ"，值为"ch o"
    "トゥ/ t u",  # 定义一个键值对，键为"トゥ"，值为"t u"
    "トャ/ ty a",  # 定义一个键值对，键为"トャ"，值为"ty a"
    "トュ/ ty u",  # 定义一个键值对，键为"トュ"，值为"ty u"
    "トョ/ ty o",  # 定义一个键值对，键为"トョ"，值为"ty o"
    "ドァ/ d o a",  # 定义一个键值对，键为"ドァ"，值为"d o a"
    "ドゥ/ d u",  # 定义一个键值对，键为"ドゥ"，值为"d u"
    "ドャ/ dy a",  # 定义一个键值对，键为"ドャ"，值为"dy a"
    "ドュ/ dy u",  # 定义一个键值对，键为"ドュ"，值为"dy u"
    "ドョ/ dy o",  # 定义一个键值对，键为"ドョ"，值为"dy o"
    "ドォ/ d o:",  # 定义一个键值对，键为"ドォ"，值为"d o:"
    "ニャ/ ny a",  # 定义一个键值对，键为"ニャ"，值为"ny a"
    "ニュ/ ny u",  # 定义一个键值对，键为"ニュ"，值为"ny u"
    "ニョ/ ny o",  # 定义一个键值对，键为"ニョ"，值为"ny o"
    "ヒャ/ hy a",  # 定义一个键值对，键为"ヒャ"，值为"hy a"
    "ヒュ/ hy u",  # 定义一个键值对，键为"ヒュ"，值为"hy u"
    "ヒョ/ hy o",  # 定义一个键值对，键为"ヒョ"，值为"hy o"
    "ミャ/ my a",  # 定义一个键值对，键为"ミャ"，值为"my a"
    "ミュ/ my u",  # 定义一个键值对，键为"ミュ"，值为"my u"
    "ミョ/ my o",  # 注释：定义了一个字符串 "ミョ/ my o"
    "リャ/ ry a",  # 注释：定义了一个字符串 "リャ/ ry a"
    "リュ/ ry u",  # 注释：定义了一个字符串 "リュ/ ry u"
    "リョ/ ry o",  # 注释：定义了一个字符串 "リョ/ ry o"
    "ギャ/ gy a",  # 注释：定义了一个字符串 "ギャ/ gy a"
    "ギュ/ gy u",  # 注释：定义了一个字符串 "ギュ/ gy u"
    "ギョ/ gy o",  # 注释：定义了一个字符串 "ギョ/ gy o"
    "ヂェ/ j e",   # 注释：定义了一个字符串 "ヂェ/ j e"
    "ヂャ/ j a",   # 注释：定义了一个字符串 "ヂャ/ j a"
    "ヂュ/ j u",   # 注释：定义了一个字符串 "ヂュ/ j u"
    "ヂョ/ j o",   # 注释：定义了一个字符串 "ヂョ/ j o"
    "ジェ/ j e",   # 注释：定义了一个字符串 "ジェ/ j e"
    "ジャ/ j a",   # 注释：定义了一个字符串 "ジャ/ j a"
    "ジュ/ j u",   # 注释：定义了一个字符串 "ジュ/ j u"
    "ジョ/ j o",   # 注释：定义了一个字符串 "ジョ/ j o"
    "ビャ/ by a",  # 注释：定义了一个字符串 "ビャ/ by a"
    "ビュ/ by u",  # 注释：定义了一个字符串 "ビュ/ by u"
    "ビョ/ by o",  # 注释：定义了一个字符串 "ビョ/ by o"
    "ピャ/ py a",  # 注释：定义了一个字符串 "ピャ/ py a"
    "ピュ/ py u",  # 注释：定义了一个字符串 "ピュ/ py u"
    "ピョ/ py o",  # 将 "ピョ" 映射为 "py o"
    "ウァ/ u a",  # 将 "ウァ" 映射为 "u a"
    "ウィ/ w i",  # 将 "ウィ" 映射为 "w i"
    "ウェ/ w e",  # 将 "ウェ" 映射为 "w e"
    "ウォ/ w o",  # 将 "ウォ" 映射为 "w o"
    "ファ/ f a",  # 将 "ファ" 映射为 "f a"
    "フィ/ f i",  # 将 "フィ" 映射为 "f i"
    "フゥ/ f u",  # 将 "フゥ" 映射为 "f u"
    "フャ/ hy a",  # 将 "フャ" 映射为 "hy a"
    "フュ/ hy u",  # 将 "フュ" 映射为 "hy u"
    "フョ/ hy o",  # 将 "フョ" 映射为 "hy o"
    "フェ/ f e",  # 将 "フェ" 映射为 "f e"
    "フォ/ f o",  # 将 "フォ" 映射为 "f o"
    "ヴァ/ b a",  # 将 "ヴァ" 映射为 "b a"
    "ヴィ/ b i",  # 将 "ヴィ" 映射为 "b i"
    "ヴェ/ b e",  # 将 "ヴェ" 映射为 "b e"
    "ヴォ/ b o",  # 将 "ヴォ" 映射为 "b o"
    "ヴュ/ by u",  # 将 "ヴュ" 映射为 "by u"
    # Conversion of 1 letter
    "ア/ a",  # 将 "ア" 映射为 "a"
    "イ/ i",  # 将字符"イ"映射为音节"i"
    "ウ/ u",  # 将字符"ウ"映射为音节"u"
    "エ/ e",  # 将字符"エ"映射为音节"e"
    "オ/ o",  # 将字符"オ"映射为音节"o"
    "カ/ k a",  # 将字符"カ"映射为音节"ka"
    "キ/ k i",  # 将字符"キ"映射为音节"ki"
    "ク/ k u",  # 将字符"ク"映射为音节"ku"
    "ケ/ k e",  # 将字符"ケ"映射为音节"ke"
    "コ/ k o",  # 将字符"コ"映射为音节"ko"
    "サ/ s a",  # 将字符"サ"映射为音节"sa"
    "シ/ sh i",  # 将字符"シ"映射为音节"shi"
    "ス/ s u",  # 将字符"ス"映射为音节"su"
    "セ/ s e",  # 将字符"セ"映射为音节"se"
    "ソ/ s o",  # 将字符"ソ"映射为音节"so"
    "タ/ t a",  # 将字符"タ"映射为音节"ta"
    "チ/ ch i",  # 将字符"チ"映射为音节"chi"
    "ツ/ ts u",  # 将字符"ツ"映射为音节"tsu"
    "テ/ t e",  # 将字符"テ"映射为音节"te"
    "ト/ t o",  # 将字符"ト"映射为音节"to"
    "ナ/ n a",  # 将字符"ナ"映射为音节"na"
    "ニ/ n i",  # 创建一个键为"ニ"，值为"n i"的字典项
    "ヌ/ n u",  # 创建一个键为"ヌ"，值为"n u"的字典项
    "ネ/ n e",  # 创建一个键为"ネ"，值为"n e"的字典项
    "ノ/ n o",  # 创建一个键为"ノ"，值为"n o"的字典项
    "ハ/ h a",  # 创建一个键为"ハ"，值为"h a"的字典项
    "ヒ/ h i",  # 创建一个键为"ヒ"，值为"h i"的字典项
    "フ/ f u",  # 创建一个键为"フ"，值为"f u"的字典项
    "ヘ/ h e",  # 创建一个键为"ヘ"，值为"h e"的字典项
    "ホ/ h o",  # 创建一个键为"ホ"，值为"h o"的字典项
    "マ/ m a",  # 创建一个键为"マ"，值为"m a"的字典项
    "ミ/ m i",  # 创建一个键为"ミ"，值为"m i"的字典项
    "ム/ m u",  # 创建一个键为"ム"，值为"m u"的字典项
    "メ/ m e",  # 创建一个键为"メ"，值为"m e"的字典项
    "モ/ m o",  # 创建一个键为"モ"，值为"m o"的字典项
    "ラ/ r a",  # 创建一个键为"ラ"，值为"r a"的字典项
    "リ/ r i",  # 创建一个键为"リ"，值为"r i"的字典项
    "ル/ r u",  # 创建一个键为"ル"，值为"r u"的字典项
    "レ/ r e",  # 创建一个键为"レ"，值为"r e"的字典项
    "ロ/ r o",  # 创建一个键为"ロ"，值为"r o"的字典项
    "ガ/ g a",  # 创建一个键为"ガ"，值为"g a"的字典项
    "ギ/ g i",  # 将字符串"ギ/ g i"添加到字典中
    "グ/ g u",  # 将字符串"グ/ g u"添加到字典中
    "ゲ/ g e",  # 将字符串"ゲ/ g e"添加到字典中
    "ゴ/ g o",  # 将字符串"ゴ/ g o"添加到字典中
    "ザ/ z a",  # 将字符串"ザ/ z a"添加到字典中
    "ジ/ j i",  # 将字符串"ジ/ j i"添加到字典中
    "ズ/ z u",  # 将字符串"ズ/ z u"添加到字典中
    "ゼ/ z e",  # 将字符串"ゼ/ z e"添加到字典中
    "ゾ/ z o",  # 将字符串"ゾ/ z o"添加到字典中
    "ダ/ d a",  # 将字符串"ダ/ d a"添加到字典中
    "ヂ/ j i",  # 将字符串"ヂ/ j i"添加到字典中
    "ヅ/ z u",  # 将字符串"ヅ/ z u"添加到字典中
    "デ/ d e",  # 将字符串"デ/ d e"添加到字典中
    "ド/ d o",  # 将字符串"ド/ d o"添加到字典中
    "バ/ b a",  # 将字符串"バ/ b a"添加到字典中
    "ビ/ b i",  # 将字符串"ビ/ b i"添加到字典中
    "ブ/ b u",  # 将字符串"ブ/ b u"添加到字典中
    "ベ/ b e",  # 将字符串"ベ/ b e"添加到字典中
    "ボ/ b o",  # 将字符串"ボ/ b o"添加到字典中
    "パ/ p a",  # 将字符串"パ/ p a"添加到字典中
    "ピ/ p i",  # 将"ピ"映射为"p i"
    "プ/ p u",  # 将"プ"映射为"p u"
    "ペ/ p e",  # 将"ペ"映射为"p e"
    "ポ/ p o",  # 将"ポ"映射为"p o"
    "ヤ/ y a",  # 将"ヤ"映射为"y a"
    "ユ/ y u",  # 将"ユ"映射为"y u"
    "ヨ/ y o",  # 将"ヨ"映射为"y o"
    "ワ/ w a",  # 将"ワ"映射为"w a"
    "ヰ/ i",    # 将"ヰ"映射为"i"
    "ヱ/ e",    # 将"ヱ"映射为"e"
    "ヲ/ o",    # 将"ヲ"映射为"o"
    "ン/ N",    # 将"ン"映射为"N"
    "ッ/ q",    # 将"ッ"映射为"q"
    "ヴ/ b u",  # 将"ヴ"映射为"b u"
    "ー/:",      # 将"ー"映射为":"
    # 尝试转换损坏的文本
    "ァ/ a",    # 将"ァ"映射为"a"
    "ィ/ i",    # 将"ィ"映射为"i"
    "ゥ/ u",    # 将"ゥ"映射为"u"
    "ェ/ e",    # 将"ェ"映射为"e"
    "ォ/ o",
    "ヮ/ w a",
    "ォ/ o",
    # Symbols
    "、/ ,",
    "。/ .",
    "！/ !",
    "？/ ?",
    "・/ ,",
]
```
这部分代码是一个包含替换规则的列表，每个元素都是一个字符串，包含了原始字符和替换字符，用"/"分隔。注释部分标明了这些规则是用来替换符号的。

```
_COLON_RX = re.compile(":+")
_REJECT_RX = re.compile("[^ a-zA-Z:,.?]")
```
这部分代码定义了两个正则表达式模式，一个用来匹配冒号的模式，另一个用来匹配除了空格、字母、冒号、逗号和问号之外的字符的模式。

```
def _makerulemap():
    l = [tuple(x.split("/")) for x in _CONVRULES]
    return tuple({k: v for k, v in l if len(k) == i} for i in (1, 2))
```
这部分代码定义了一个函数_makerulemap()，它将_CONVRULES中的替换规则转换为字典，并根据原始字符的长度将字典分组。
_RULEMAP1, _RULEMAP2 = _makerulemap()  # 调用_makerulemap函数，将返回的两个值分别赋给_RULEMAP1和_RULEMAP2变量


def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""  # 函数说明文档，说明该函数将片假名文本转换为音素
    text = text.strip()  # 去除文本两端的空白字符
    res = []  # 初始化一个空列表，用于存储转换后的音素
    while text:  # 进入循环，直到文本为空
        if len(text) >= 2:  # 如果文本长度大于等于2
            x = _RULEMAP2.get(text[:2])  # 获取_RULEMAP2中以text的前两个字符为键的值
            if x is not None:  # 如果获取到了值
                text = text[2:]  # 将文本前两个字符去除
                res += x.split(" ")[1:]  # 将获取到的值按空格分割后的第二个元素及之后的元素添加到res列表中
                continue  # 继续下一次循环
        x = _RULEMAP1.get(text[0])  # 获取_RULEMAP1中以text的第一个字符为键的值
        if x is not None:  # 如果获取到了值
            text = text[1:]  # 将文本的第一个字符去除
            res += x.split(" ")[1:]  # 将获取到的值按空格分割后的第二个元素及之后的元素添加到res列表中
            continue  # 继续下一次循环
        res.append(text[0])  # 如果以上条件都不满足，将文本的第一个字符添加到res列表中
        text = text[1:]  # 从第二个字符开始截取字符串，去掉第一个字符
    # res = _COLON_RX.sub(":", res)  # 使用正则表达式将 res 中的特定模式替换为冒号
    return res  # 返回处理后的字符串


_KATAKANA = "".join(chr(ch) for ch in range(ord("ァ"), ord("ン") + 1))  # 生成片假名字符集
_HIRAGANA = "".join(chr(ch) for ch in range(ord("ぁ"), ord("ん") + 1))  # 生成平假名字符集
_HIRA2KATATRANS = str.maketrans(_HIRAGANA, _KATAKANA)  # 生成平假名到片假名的转换表


def hira2kata(text: str) -> str:
    text = text.translate(_HIRA2KATATRANS)  # 使用转换表将平假名转换为片假名
    return text.replace("う゛", "ヴ")  # 将特定字符串替换为另一个字符串


_SYMBOL_TOKENS = set(list("・、。？！"))  # 生成包含特定符号的集合
_NO_YOMI_TOKENS = set(list("「」『』―（）［］[]"))  # 生成包含特定标点符号的集合
_TAGGER = MeCab.Tagger()  # 创建 MeCab 分词器对象
def text2kata(text: str) -> str:  # 定义一个函数，接受一个字符串参数并返回一个字符串
    parsed = _TAGGER.parse(text)  # 使用_TAGGER对文本进行解析
    res = []  # 创建一个空列表用于存储结果
    for line in parsed.split("\n"):  # 遍历解析后的文本的每一行
        if line == "EOS":  # 如果当前行是"EOS"，则跳出循环
            break
        parts = line.split("\t")  # 将当前行按制表符分割成多个部分

        word, yomi = parts[0], parts[1]  # 将分割后的部分分别赋值给word和yomi
        if yomi:  # 如果yomi不为空
            res.append(yomi)  # 将yomi添加到结果列表中
        else:  # 如果yomi为空
            if word in _SYMBOL_TOKENS:  # 如果word在_SYMBOL_TOKENS中
                res.append(word)  # 将word添加到结果列表中
            elif word in ("っ", "ッ"):  # 如果word是"っ"或"ッ"
                res.append("ッ")  # 将"ッ"添加到结果列表中
            elif word in _NO_YOMI_TOKENS:  # 如果word在_NO_YOMI_TOKENS中
                pass  # 不做任何操作
            else:  # 如果以上条件都不满足
                res.append(word)  # 将word添加到结果列表中
    return hira2kata("".join(res))
    # 调用名为hira2kata的函数，将res列表中的元素连接成字符串，并将其作为参数传递给hira2kata函数，返回函数的执行结果


_ALPHASYMBOL_YOMI = {
    "#": "シャープ",
    "%": "パーセント",
    "&": "アンド",
    "+": "プラス",
    "-": "マイナス",
    ":": "コロン",
    ";": "セミコロン",
    "<": "小なり",
    "=": "イコール",
    ">": "大なり",
    "@": "アット",
    "a": "エー",
    "b": "ビー",
    "c": "シー",
    "d": "ディー",
    "e": "イー",
    # 创建一个名为_ALPHASYMBOL_YOMI的字典，将特定的符号和对应的日语读音作为键值对存储
    "f": "エフ",  # 将字母"f"映射为日语发音"エフ"
    "g": "ジー",  # 将字母"g"映射为日语发音"ジー"
    "h": "エイチ",  # 将字母"h"映射为日语发音"エイチ"
    "i": "アイ",  # 将字母"i"映射为日语发音"アイ"
    "j": "ジェー",  # 将字母"j"映射为日语发音"ジェー"
    "k": "ケー",  # 将字母"k"映射为日语发音"ケー"
    "l": "エル",  # 将字母"l"映射为日语发音"エル"
    "m": "エム",  # 将字母"m"映射为日语发音"エム"
    "n": "エヌ",  # 将字母"n"映射为日语发音"エヌ"
    "o": "オー",  # 将字母"o"映射为日语发音"オー"
    "p": "ピー",  # 将字母"p"映射为日语发音"ピー"
    "q": "キュー",  # 将字母"q"映射为日语发音"キュー"
    "r": "アール",  # 将字母"r"映射为日语发音"アール"
    "s": "エス",  # 将字母"s"映射为日语发音"エス"
    "t": "ティー",  # 将字母"t"映射为日语发音"ティー"
    "u": "ユー",  # 将字母"u"映射为日语发音"ユー"
    "v": "ブイ",  # 将字母"v"映射为日语发音"ブイ"
    "w": "ダブリュー",  # 将字母"w"映射为日语发音"ダブリュー"
    "x": "エックス",  # 将字母"x"映射为日语发音"エックス"
    "y": "ワイ",  # 将字母"y"映射为日语发音"ワイ"
    "z": "ゼット",  # 将字母 z 对应的日文发音添加到字典中
    "α": "アルファ",  # 将希腊字母 α 对应的日文发音添加到字典中
    "β": "ベータ",  # 将希腊字母 β 对应的日文发音添加到字典中
    "γ": "ガンマ",  # 将希腊字母 γ 对应的日文发音添加到字典中
    "δ": "デルタ",  # 将希腊字母 δ 对应的日文发音添加到字典中
    "ε": "イプシロン",  # 将希腊字母 ε 对应的日文发音添加到字典中
    "ζ": "ゼータ",  # 将希腊字母 ζ 对应的日文发音添加到字典中
    "η": "イータ",  # 将希腊字母 η 对应的日文发音添加到字典中
    "θ": "シータ",  # 将希腊字母 θ 对应的日文发音添加到字典中
    "ι": "イオタ",  # 将希腊字母 ι 对应的日文发音添加到字典中
    "κ": "カッパ",  # 将希腊字母 κ 对应的日文发音添加到字典中
    "λ": "ラムダ",  # 将希腊字母 λ 对应的日文发音添加到字典中
    "μ": "ミュー",  # 将希腊字母 μ 对应的日文发音添加到字典中
    "ν": "ニュー",  # 将希腊字母 ν 对应的日文发音添加到字典中
    "ξ": "クサイ",  # 将希腊字母 ξ 对应的日文发音添加到字典中
    "ο": "オミクロン",  # 将希腊字母 ο 对应的日文发音添加到字典中
    "π": "パイ",  # 将希腊字母 π 对应的日文发音添加到字典中
    "ρ": "ロー",  # 将希腊字母 ρ 对应的日文发音添加到字典中
    "σ": "シグマ",  # 将希腊字母 σ 对应的日文发音添加到字典中
    "τ": "タウ",  # 将希腊字母 τ 对应的日文发音添加到字典中
    "υ": "ウプシロン",  # 将希腊字母"υ"映射为日语中的"ウプシロン"
    "φ": "ファイ",  # 将希腊字母"φ"映射为日语中的"ファイ"
    "χ": "カイ",  # 将希腊字母"χ"映射为日语中的"カイ"
    "ψ": "プサイ",  # 将希腊字母"ψ"映射为日语中的"プサイ"
    "ω": "オメガ",  # 将希腊字母"ω"映射为日语中的"オメガ"
}

_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")  # 匹配带有逗号分隔的数字的正则表达式
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}  # 将货币符号映射为日语中的货币名称
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")  # 匹配带有货币符号的数字的正则表达式
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")  # 匹配数字的正则表达式

def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)  # 使用正则表达式替换函数去除数字中的逗号分隔符
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)  # 使用正则表达式替换函数将货币符号替换为日语中的货币名称
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)  # 使用正则表达式替换函数将数字转换为对应的日语文本
    return res  # 返回转换后的文本
def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    # 将文本中的英文字母和符号转换为对应的日语单词
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])


def japanese_text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phonemes."""
    # 对文本进行 Unicode 规范化
    res = unicodedata.normalize("NFKC", text)
    # 将文本中的数字转换为对应的日语单词
    res = japanese_convert_numbers_to_words(res)
    # 将文本中的英文字母和符号转换为对应的日语单词
    # res = japanese_convert_alpha_symbols_to_words(res)
    # 将文本转换为片假名
    res = text2kata(res)
    # 将片假名转换为音素
    res = kata2phoneme(res)
    return res


def is_japanese_character(char):
    # 定义日语文字系统的 Unicode 范围
    japanese_ranges = [
        (0x3040, 0x309F),  # 平假名
        (0x30A0, 0x30FF),  # 片假名
        (0x4E00, 0x9FFF),  # 汉字 (CJK Unified Ideographs) - 定义了汉字的 Unicode 范围
        (0x3400, 0x4DBF),  # 汉字扩展 A - 定义了汉字扩展 A 的 Unicode 范围
        (0x20000, 0x2A6DF),  # 汉字扩展 B - 定义了汉字扩展 B 的 Unicode 范围
        # 可以根据需要添加其他汉字扩展范围

    ]

    # 将字符的 Unicode 编码转换为整数
    char_code = ord(char)  # 将字符转换为其 Unicode 编码的整数值

    # 检查字符是否在任何一个日语范围内
    for start, end in japanese_ranges:  # 遍历日语范围列表
        if start <= char_code <= end:  # 检查字符的 Unicode 编码是否在日语范围内
            return True  # 如果在日语范围内，则返回 True

    return False  # 如果不在任何一个日语范围内，则返回 False


rep_map = {
    "：": ",",  # 将中文冒号替换为英文逗号
    "；": ",",  # 将中文分号替换为英文逗号
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
}

# 定义一个函数，用于替换文本中的标点符号
def replace_punctuation(text):
    # 创建一个正则表达式模式，用于匹配需要替换的标点符号
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    # 使用正则表达式模式替换文本中的标点符号
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    # 使用正则表达式去除除了日文、片假名、中文和标点符号之外的所有字符
    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF"
        + "".join(punctuation)
        + r"]+",
        "",
        replaced_text,
    )

    return replaced_text


def text_normalize(text):
    # 使用 unicodedata 模块对文本进行规范化处理
    res = unicodedata.normalize("NFKC", text)
    # 将文本中的数字转换为对应的日文单词
    res = japanese_convert_numbers_to_words(res)
    # 去除文本中的标点符号
    res = replace_punctuation(res)
    return res


def distribute_phone(n_phone, n_word):
    # 初始化每个单词分配到的电话数量列表
    phones_per_word = [0] * n_word
    # 遍历每个电话号码
    for task in range(n_phone):
        # 找到分配到电话最少的单词
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")
# 从预训练模型中加载分词器

def g2p(norm_text):
    tokenized = tokenizer.tokenize(norm_text)
    # 使用分词器对输入文本进行分词
    phs = []
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    # 将分词结果按照 "#" 符号进行分组

    word2ph = []
    for group in ph_groups:
        phonemes = kata2phoneme(text2kata("".join(group)))
        # 将分组后的文本转换为假名，再将假名转换为音素
        # phonemes = [i for i in phonemes if i in symbols]
    # 这行代码被注释掉，可能是用于过滤音素的功能
        for i in phonemes:  # 遍历phonemes列表中的每个元素
            assert i in symbols, (group, norm_text, tokenized)  # 断言phonemes中的元素在symbols中，否则抛出异常并显示group, norm_text, tokenized的值
        phone_len = len(phonemes)  # 计算phonemes列表的长度并赋值给phone_len
        word_len = len(group)  # 计算group的长度并赋值给word_len

        aaa = distribute_phone(phone_len, word_len)  # 调用distribute_phone函数，传入phone_len和word_len作为参数，并将返回值赋给aaa
        word2ph += aaa  # 将aaa的值添加到word2ph列表中

        phs += phonemes  # 将phonemes列表中的元素添加到phs列表中
    phones = ["_"] + phs + ["_"]  # 创建一个新列表phones，包含"_"、phs列表的元素和"_"，并赋值给phones
    tones = [0 for i in phones]  # 创建一个与phones列表长度相同的列表，每个元素为0，并赋值给tones
    word2ph = [1] + word2ph + [1]  # 创建一个新列表word2ph，包含1、word2ph列表的元素和1，并赋值给word2ph
    return phones, tones, word2ph  # 返回phones, tones, word2ph这三个列表


if __name__ == "__main__":  # 如果当前脚本被直接执行
    tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")  # 从预训练模型中加载tokenizer，并赋值给tokenizer
    text = "hello,こんにちは、世界！……"  # 定义一个字符串变量text
    from text.japanese_bert import get_bert_feature  # 从text.japanese_bert模块中导入get_bert_feature函数
    # 对文本进行规范化处理
    text = text_normalize(text)
    # 打印处理后的文本
    print(text)
    # 使用文本进行音素转换，得到音素、音调和单词到音素的映射关系
    phones, tones, word2ph = g2p(text)
    # 获取文本的 BERT 特征
    bert = get_bert_feature(text, word2ph)
    # 打印音素、音调、单词到音素的映射关系以及 BERT 特征的形状
    print(phones, tones, word2ph, bert.shape)
```