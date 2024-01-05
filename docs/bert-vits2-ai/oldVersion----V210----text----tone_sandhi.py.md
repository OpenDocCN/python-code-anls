# `d:/src/tocomm/Bert-VITS2\oldVersion\V210\text\tone_sandhi.py`

```
# 引入必要的库
from typing import List  # 从 typing 库中引入 List 类型
from typing import Tuple  # 从 typing 库中引入 Tuple 类型
import jieba  # 引入 jieba 库，用于中文分词
from pypinyin import lazy_pinyin  # 从 pypinyin 库中引入 lazy_pinyin 函数，用于获取拼音
from pypinyin import Style  # 从 pypinyin 库中引入 Style 类型，用于指定拼音风格
class ToneSandhi:  # 定义一个名为ToneSandhi的类
    def __init__(self):  # 定义初始化方法，self代表类的实例
        self.must_neutral_tone_words = {  # 创建一个名为must_neutral_tone_words的实例变量，包含一组中性音调的词语
            "麻烦",
            "麻利",
            "鸳鸯",
            "高粱",
            "骨头",
            "骆驼",
            "马虎",
            "首饰",
            "馒头",
            "馄饨",
            "风筝",
            "难为",
            "队伍",
            "阔气",
            "闺女",
            "门道",
# 创建一个包含字符串的列表
words = [
    "锄头",
    "铺盖",
    "铃铛",
    "铁匠",
    "钥匙",
    "里脊",
    "里头",
    "部分",
    "那么",
    "道士",
    "造化",
    "迷糊",
    "连累",
    "这么",
    "这个",
    "运气",
    "过去",
    "软和",
    "转悠",
    "踏实",
]
# 创建一个包含字符串的列表
word_list = [
    "跳蚤",
    "跟头",
    "趔趄",
    "财主",
    "豆腐",
    "讲究",
    "记性",
    "记号",
    "认识",
    "规矩",
    "见识",
    "裁缝",
    "补丁",
    "衣裳",
    "衣服",
    "衙门",
    "街坊",
    "行李",
    "行当",
    "蛤蟆",
]
# 创建一个包含字符串的列表
words = [
    "蘑菇",
    "薄荷",
    "葫芦",
    "葡萄",
    "萝卜",
    "荸荠",
    "苗条",
    "苗头",
    "苍蝇",
    "芝麻",
    "舒服",
    "舒坦",
    "舌头",
    "自在",
    "膏药",
    "脾气",
    "脑袋",
    "脊梁",
    "能耐",
    "胳膊",
]
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，封装成字节流
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    zip.close()  # 关闭 ZIP 对象
    return fdict  # 返回结果字典
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
抱歉，给定的代码片段是一段中文词语列表，而不是程序代码。请提供正确的程序代码，我将很乐意为您添加注释。
抱歉，给定的代码片段是一段中文词语列表，而不是程序代码。请提供正确的程序代码，我将很乐意为您添加注释。
# 创建一个包含字符串的列表
word_list = [
    "烧饼",
    "烟筒",
    "烂糊",
    "点心",
    "炊帚",
    "灯笼",
    "火候",
    "漂亮",
    "滑溜",
    "溜达",
    "温和",
    "清楚",
    "消息",
    "浪头",
    "活泼",
    "比方",
    "正经",
    "欺负",
    "模糊",
    "槟榔",
]
# 创建一个包含字符串的列表
items = [
    "棺材",
    "棒槌",
    "棉花",
    "核桃",
    "栅栏",
    "柴火",
    "架势",
    "枕头",
    "枇杷",
    "机灵",
    "本事",
    "木头",
    "木匠",
    "朋友",
    "月饼",
    "月亮",
    "暖和",
    "明白",
    "时候",
    "新鲜",
]
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
# 创建一个包含字符串的列表
words = [
    "打扮",
    "打听",
    "打发",
    "扎实",
    "扁担",
    "戒指",
    "懒得",
    "意识",
    "意思",
    "情形",
    "悟性",
    "怪物",
    "思量",
    "怎么",
    "念头",
    "念叨",
    "快活",
    "忙活",
    "志气",
    "心思"
]
# 创建一个空的集合
word_set = set()

# 遍历给定的词语列表
for word in word_list:
    # 将词语添加到集合中
    word_set.add(word)

# 返回集合
return word_set
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，封装成字节流
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    zip.close()  # 关闭 ZIP 对象
    return fdict  # 返回结果字典
抱歉，这段代码看起来像是一些词语的列表，而不是需要注释的程序代码。如果你有其他需要解释的代码，请随时告诉我，我会很乐意帮助你添加注释。
# 创建一个包含汉字的列表
words = [
    "名堂",
    "合同",
    "吆喝",
    "叫唤",
    "口袋",
    "厚道",
    "厉害",
    "千斤",
    "包袱",
    "包涵",
    "匀称",
    "勤快",
    "动静",
    "动弹",
    "功夫",
    "力气",
    "前头",
    "刺猬",
    "刺激",
    "别扭",
]
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 创建一个字节流对象，将文件内容封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建一个 ZIP 对象
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
# 创建一个空的字典
word_dict = {}
# 遍历给定的字符串列表
for word in word_list:
    # 将每个字符串作为字典的键，值设为1
    word_dict[word] = 1
# 返回结果字典
return word_dict
# 创建一个包含字符串的列表
word_list = [
    "下水",
    "下巴",
    "上头",
    "上司",
    "丈夫",
    "丈人",
    "一辈",
    "那个",
    "菩萨",
    "父亲",
    "母亲",
    "咕噜",
    "邋遢",
    "费用",
    "冤家",
    "甜头",
    "介绍",
    "荒唐",
    "大人",
    "泥鳅",
]
# 创建一个包含字符串的列表
words = [
    "幸福",
    "熟悉",
    "计划",
    "扑腾",
    "蜡烛",
    "姥爷",
    "照顾",
    "喉咙",
    "吉他",
    "弄堂",
    "蚂蚱",
    "凤凰",
    "拖沓",
    "寒碜",
    "糟蹋",
    "倒腾",
    "报复",
    "逻辑",
    "盘缠",
    "喽啰",
]
        "牢骚",  # 添加“牢骚”到情感词集合中
        "咖喱",  # 添加“咖喱”到情感词集合中
        "扫把",  # 添加“扫把”到情感词集合中
        "惦记",  # 添加“惦记”到情感词集合中
    }
    self.must_not_neural_tone_words = {  # 创建不带有中性语气的词集合
        "男子",  # 添加“男子”到不带有中性语气的词集合中
        "女子",  # 添加“女子”到不带有中性语气的词集合中
        "分子",  # 添加“分子”到不带有中性语气的词集合中
        "原子",  # 添加“原子”到不带有中性语气的词集合中
        "量子",  # 添加“量子”到不带有中性语气的词集合中
        "莲子",  # 添加“莲子”到不带有中性语气的词集合中
        "石子",  # 添加“石子”到不带有中性语气的词集合中
        "瓜子",  # 添加“瓜子”到不带有中性语气的词集合中
        "电子",  # 添加“电子”到不带有中性语气的词集合中
        "人人",  # 添加“人人”到不带有中性语气的词集合中
        "虎虎",  # 添加“虎虎”到不带有中性语气的词集合中
    }
    self.punc = "：，；。？！“”‘’':,;.?!"  # 定义标点符号
    # the meaning of jieba pos tag: https://blog.csdn.net/weixin_44174352/article/details/113731041
    # e.g.
    # word: "家里"
    # pos: "s"
    # finals: ['ia1', 'i3']
    def _neural_sandhi(self, word: str, pos: str, finals: List[str]) -> List[str]:
        # reduplication words for n. and v. e.g. 奶奶, 试试, 旺旺
        # 对名词和动词进行重复词的处理，例如：奶奶，试试，旺旺
        for j, item in enumerate(word):
            if (
                j - 1 >= 0
                and item == word[j - 1]
                and pos[0] in {"n", "v", "a"}
                and word not in self.must_not_neural_tone_words
            ):
                finals[j] = finals[j][:-1] + "5"
        # find the index of "个" in the word
        # 寻找词中"个"的索引
        ge_idx = word.find("个")
        # if the last character of the word is in the specified characters, change the final tone to "5"
        # 如果词的最后一个字符在指定的字符中，则将最后一个音调改为"5"
        if len(word) >= 1 and word[-1] in "吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶":
            finals[-1] = finals[-1][:-1] + "5"
        # if the last character of the word is in "的地得", change the final tone to "5"
        # 如果词的最后一个字符在"的地得"中，则将最后一个音调改为"5"
        elif len(word) >= 1 and word[-1] in "的地得":
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 走了, 看着, 去过
        # 如果单词长度为1，且在"了着过"中，并且词性在{"ul", "uz", "ug"}中
        # 则将最后一个韵母替换为5
        # elif len(word) == 1 and word in "了着过" and pos in {"ul", "uz", "ug"}:
        #     finals[-1] = finals[-1][:-1] + "5"
        elif (
            len(word) > 1
            and word[-1] in "们子"
            and pos in {"r", "n"}
            and word not in self.must_not_neutral_tone_words
        ):
            # 如果单词长度大于1，且最后一个字在"们子"中，并且词性在{"r", "n"}中，并且单词不在self.must_not_neutral_tone_words中
            # 则将最后一个韵母替换为5
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 桌上, 地下, 家里
        elif len(word) > 1 and word[-1] in "上下里" and pos in {"s", "l", "f"}:
            # 如果单词长度大于1，且最后一个字在"上下里"中，并且词性在{"s", "l", "f"}中
            # 则将最后一个韵母替换为5
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 上来, 下去
        elif len(word) > 1 and word[-1] in "来去" and word[-2] in "上下进出回过起开":
            # 如果单词长度大于1，且最后一个字在"来去"中，并且倒数第二个字在"上下进出回过起开"中
            # 则将最后一个韵母替换为5
            finals[-1] = finals[-1][:-1] + "5"
        # 个做量词
        elif (
            ge_idx >= 1
            and (word[ge_idx - 1].isnumeric() or word[ge_idx - 1] in "几有两半多各整每做是")
        ) or word == "个":  # 如果当前词是"的"或者"个"，则将对应位置的韵母替换为"5"
            finals[ge_idx] = finals[ge_idx][:-1] + "5"
        else:  # 否则
            if (
                word in self.must_neural_tone_words  # 如果当前词在必须使用轻声的词列表中
                or word[-2:] in self.must_neural_tone_words  # 或者当前词的后两个字在必须使用轻声的词列表中
            ):
                finals[-1] = finals[-1][:-1] + "5"  # 将最后一个韵母替换为"5"

        word_list = self._split_word(word)  # 将当前词拆分成单个字的列表
        finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]  # 根据拆分后的单个字列表，将韵母列表也进行拆分
        for i, word in enumerate(word_list):  # 遍历拆分后的单个字列表
            # conventional neural in Chinese
            if (
                word in self.must_neural_tone_words  # 如果当前单个字在必须使用轻声的词列表中
                or word[-2:] in self.must_neural_tone_words  # 或者当前单个字的后两个字在必须使用轻声的词列表中
            ):
                finals_list[i][-1] = finals_list[i][-1][:-1] + "5"  # 将对应位置的韵母替换为"5"
        finals = sum(finals_list, [])  # 将拆分后的韵母列表合并成一个列表
        return finals  # 返回处理后的韵母列表
    def _bu_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # 对于词语长度为3且第二个字是"不"的情况，将对应的韵母列表中的第二个韵母的最后一个字符替换为5
        if len(word) == 3 and word[1] == "不":
            finals[1] = finals[1][:-1] + "5"
        else:
            for i, char in enumerate(word):
                # 如果"不"后面是第四声的情况，将对应的韵母列表中的当前韵母的最后一个字符替换为2
                if char == "不" and i + 1 < len(word) and finals[i + 1][-1] == "4":
                    finals[i] = finals[i][:-1] + "2"
        return finals

    def _yi_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # 如果词语中包含"一"且除了"一"以外的所有字符都是数字，则直接返回韵母列表
        if word.find("一") != -1 and all(
            [item.isnumeric() for item in word if item != "一"]
        ):
            return finals
        # 如果词语长度为3且第二个字是"一"且第一个字和最后一个字相同的情况，将对应的韵母列表返回
        elif len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
        # 如果"一"后面是"5"，则将"一"替换为"5"，例如："一年"变为"五年"
        finals[1] = finals[1][:-1] + "5"
        # 当"一"是序数词时，应该读作"yi1"，例如："第一"
        elif word.startswith("第一"):
            finals[1] = finals[1][:-1] + "1"
        else:
            for i, char in enumerate(word):
                if char == "一" and i + 1 < len(word):
                    # 如果"一"后面是声调4，则将"一"替换为"yi2"，例如："一段"
                    if finals[i + 1][-1] == "4":
                        finals[i] = finals[i][:-1] + "2"
                    # 如果"一"后面不是声调4，则将"一"替换为"yi4"，例如："一天"
                    else:
                        # 如果"一"后面是标点，则仍然读作一声
                        if word[i + 1] not in self.punc:
                            finals[i] = finals[i][:-1] + "4"
        # 返回处理后的音节列表
        return finals

    # 将词语分割成列表并按长度排序
    def _split_word(self, word: str) -> List[str]:
        # 使用结巴分词对词语进行分词
        word_list = jieba.cut_for_search(word)
        # 按照词语长度进行排序
        word_list = sorted(word_list, key=lambda i: len(i), reverse=False)
        first_subword = word_list[0]  # 获取单词列表中的第一个子词
        first_begin_idx = word.find(first_subword)  # 获取第一个子词在单词中的起始索引
        if first_begin_idx == 0:  # 如果第一个子词在单词中的起始索引为0
            second_subword = word[len(first_subword) :]  # 获取第二个子词
            new_word_list = [first_subword, second_subword]  # 创建新的单词列表，将第一个子词和第二个子词添加进去
        else:  # 如果第一个子词不在单词中的起始索引为0
            second_subword = word[: -len(first_subword)]  # 获取第二个子词
            new_word_list = [second_subword, first_subword]  # 创建新的单词列表，将第二个子词和第一个子词添加进去
        return new_word_list  # 返回新的单词列表

    def _three_sandhi(self, word: str, finals: List[str]) -> List[str]:
        if len(word) == 2 and self._all_tone_three(finals):  # 如果单词长度为2且所有韵母都是三声
            finals[0] = finals[0][:-1] + "2"  # 将第一个韵母的声调修改为二声
        elif len(word) == 3:  # 如果单词长度为3
            word_list = self._split_word(word)  # 将单词拆分成子词列表
            if self._all_tone_three(finals):  # 如果所有韵母都是三声
                #  disyllabic + monosyllabic, e.g. 蒙古/包
                if len(word_list[0]) == 2:  # 如果第一个子词的长度为2
                    finals[0] = finals[0][:-1] + "2"  # 将第一个韵母的声调修改为二声
                    finals[1] = finals[1][:-1] + "2"  # 将第二个韵母的声调修改为二声
                # 如果第一个词的长度为1
                elif len(word_list[0]) == 1:
                    # 将第二个音节的韵母替换为2
                    finals[1] = finals[1][:-1] + "2"
            else:
                # 将韵母列表分为两部分
                finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]]
                if len(finals_list) == 2:
                    for i, sub in enumerate(finals_list):
                        # 如果是三声调的词且长度为2
                        if self._all_tone_three(sub) and len(sub) == 2:
                            # 将第一个音节的韵母替换为2
                            finals_list[i][0] = finals_list[i][0][:-1] + "2"
                        # 如果是第二个词且不是三声调，且第一个词和第二个词的韵母最后一个字母都是3
                        elif (
                            i == 1
                            and not self._all_tone_three(sub)
                            and finals_list[i][0][-1] == "3"
                            and finals_list[0][-1][-1] == "3"
                        ):
                            # 将第一个词的韵母最后一个字母替换为2
                            finals_list[0][-1] = finals_list[0][-1][:-1] + "2"
                        finals = sum(finals_list, [])
        # 将成语拆分为长度为2的两个词
        elif len(word) == 4:  # 如果单词长度为4
            finals_list = [finals[:2], finals[2:]]  # 将韵母列表分为两部分
            finals = []  # 重置韵母列表
            for sub in finals_list:  # 遍历分割后的韵母列表
                if self._all_tone_three(sub):  # 如果所有韵母都是三声
                    sub[0] = sub[0][:-1] + "2"  # 将第一个韵母的声调改为二声
                finals += sub  # 将处理后的韵母列表合并
        return finals  # 返回处理后的韵母列表

    def _all_tone_three(self, finals: List[str]) -> bool:  # 定义一个方法来判断所有韵母是否都是三声
        return all(x[-1] == "3" for x in finals)  # 返回所有韵母是否都是三声的布尔值

    # merge "不" and the word behind it
    # if don't merge, "不" sometimes appears alone according to jieba, which may occur sandhi error
    def _merge_bu(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:  # 定义一个方法来合并"不"和它后面的词
        new_seg = []  # 创建一个新的分词列表
        last_word = ""  # 初始化上一个词为空字符串
        for word, pos in seg:  # 遍历分词列表中的词和词性
            if last_word == "不":  # 如果上一个词是"不"
                word = last_word + word  # 将上一个词和当前词合并
            if word != "不":  # 如果合并后的词不是"不"
                new_seg.append((word, pos))  # 将合并后的词和词性添加到新的分词结果中
            last_word = word[:]  # 更新上一个词为当前词
        if last_word == "不":  # 如果最后一个词是"不"
            new_seg.append((last_word, "d"))  # 将"不"和词性"d"添加到新的分词结果中
            last_word = ""  # 更新上一个词为空
        return new_seg  # 返回新的分词结果

    # function 1: merge "一" and reduplication words in it's left and right, e.g. "听","一","听" ->"听一听"
    # function 2: merge single  "一" and the word behind it
    # if don't merge, "一" sometimes appears alone according to jieba, which may occur sandhi error
    # e.g.
    # input seg: [('听', 'v'), ('一', 'm'), ('听', 'v')]
    # output seg: [['听一听', 'v']]
    def _merge_yi(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []  # 创建一个新的分词结果列表
        # function 1
        for i, (word, pos) in enumerate(seg):  # 遍历分词结果
            if (
# 检查当前位置的前一个位置是否存在
# 检查当前位置的词语是否为"一"
# 检查当前位置的后一个位置是否存在
# 检查当前位置的前一个词语和后一个词语是否相同
# 检查当前位置的前一个词性是否为"v"
if (
    i - 1 >= 0
    and word == "一"
    and i + 1 < len(seg)
    and seg[i - 1][0] == seg[i + 1][0]
    and seg[i - 1][1] == "v"
):
    # 如果满足条件，将前一个词语的内容重复一次并与原内容合并
    new_seg[i - 1][0] = new_seg[i - 1][0] + "一" + new_seg[i - 1][0]
else:
    # 如果不满足条件，继续执行下面的判断
    if (
        i - 2 >= 0
        and seg[i - 1][0] == "一"
        and seg[i - 2][0] == word
        and pos == "v"
    ):
        # 如果满足条件，跳过当前循环，继续下一次循环
        continue
    else:
        # 如果不满足条件，将当前词语和词性添加到新的列表中
        new_seg.append([word, pos])
# 将新的列表赋值给原列表，完成替换
seg = new_seg
new_seg = []
# 调用第二个函数
# function 2
        for i, (word, pos) in enumerate(seg):  # 遍历分词结果的每个词和词性
            if new_seg and new_seg[-1][0] == "一":  # 如果新的分词结果列表不为空，并且最后一个词是"一"
                new_seg[-1][0] = new_seg[-1][0] + word  # 将当前词与最后一个词合并
            else:
                new_seg.append([word, pos])  # 否则将当前词和词性添加到新的分词结果列表中
        return new_seg  # 返回合并后的分词结果列表

    # the first and the second words are all_tone_three
    def _merge_continuous_three_tones(
        self, seg: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:  # 定义一个方法，参数为分词结果列表，返回值为分词结果列表
        new_seg = []  # 初始化一个新的分词结果列表
        sub_finals_list = [  # 生成每个词的韵母带声调的列表
            lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        assert len(sub_finals_list) == len(seg)  # 断言韵母列表的长度与分词结果列表的长度相同
        merge_last = [False] * len(seg)  # 初始化一个与分词结果列表长度相同的合并标志列表
        for i, (word, pos) in enumerate(seg):  # 遍历分词结果的每个词和词性
            if (
                i - 1 >= 0  # 检查索引 i - 1 是否大于等于 0
                and self._all_tone_three(sub_finals_list[i - 1])  # 检查 sub_finals_list[i - 1] 是否满足 _all_tone_three 方法的条件
                and self._all_tone_three(sub_finals_list[i])  # 检查 sub_finals_list[i] 是否满足 _all_tone_three 方法的条件
                and not merge_last[i - 1]  # 检查 merge_last[i - 1] 是否为 False
            ):
                # 如果最后一个词是重复的，不合并，因为重复需要进行 _neural_sandhi 处理
                if (
                    not self._is_reduplication(seg[i - 1][0])  # 检查 seg[i - 1][0] 是否为重复
                    and len(seg[i - 1][0]) + len(seg[i][0]) <= 3  # 检查 seg[i - 1][0] 和 seg[i][0] 的长度是否小于等于 3
                ):
                    new_seg[-1][0] = new_seg[-1][0] + seg[i][0]  # 将新词组的最后一个词与当前词合并
                    merge_last[i] = True  # 将 merge_last[i] 设置为 True
                else:
                    new_seg.append([word, pos])  # 将当前词和词性添加到新词组中
            else:
                new_seg.append([word, pos])  # 将当前词和词性添加到新词组中

        return new_seg  # 返回新的词组

    def _is_reduplication(self, word: str) -> bool:  # 定义一个方法来判断一个词是否为重复的
        return len(word) == 2 and word[0] == word[1]  # 检查单词长度是否为2且第一个字符是否等于第二个字符

    # the last char of first word and the first char of second word is tone_three
    # 检查第一个词的最后一个字符和第二个词的第一个字符是否为 tone_three
    def _merge_continuous_three_tones_2(
        self, seg: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        new_seg = []  # 创建一个新的列表
        sub_finals_list = [  # 生成包含每个词拼音的列表
            lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        assert len(sub_finals_list) == len(seg)  # 断言拼音列表的长度与原词列表长度相同
        merge_last = [False] * len(seg)  # 创建一个与原词列表长度相同的布尔值列表
        for i, (word, pos) in enumerate(seg):  # 遍历原词列表
            if (
                i - 1 >= 0
                and sub_finals_list[i - 1][-1][-1] == "3"
                and sub_finals_list[i][0][-1] == "3"
                and not merge_last[i - 1]
            ):  # 检查条件是否满足
                # 如果上一个词不是重复的，并且上一个词的长度加上当前词的长度小于等于3，则不合并，因为重复需要进行_neural_sandhi
                if (
                    not self._is_reduplication(seg[i - 1][0])
                    and len(seg[i - 1][0]) + len(seg[i][0]) <= 3
                ):
                    new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
                    merge_last[i] = True
                else:
                    new_seg.append([word, pos])
            else:
                new_seg.append([word, pos])
        return new_seg

    def _merge_er(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        for i, (word, pos) in enumerate(seg):
            # 如果前一个词不是"#", 当前词是"儿"，则合并
            if i - 1 >= 0 and word == "儿" and seg[i - 1][0] != "#":
                new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
            else:
                new_seg.append([word, pos])
        return new_seg  # 返回合并后的分词结果

    def _merge_reduplication(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []  # 创建一个空列表用于存储合并后的分词结果
        for i, (word, pos) in enumerate(seg):  # 遍历输入的分词结果
            if new_seg and word == new_seg[-1][0]:  # 如果新列表不为空且当前词与前一个词相同
                new_seg[-1][0] = new_seg[-1][0] + seg[i][0]  # 将当前词与前一个词合并
            else:
                new_seg.append([word, pos])  # 否则将当前词添加到新列表中
        return new_seg  # 返回合并后的分词结果

    def pre_merge_for_modify(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        seg = self._merge_bu(seg)  # 调用_merge_bu方法对分词结果进行处理
        try:
            seg = self._merge_yi(seg)  # 尝试调用_merge_yi方法对分词结果进行处理
        except:
            print("_merge_yi failed")  # 如果调用_merge_yi方法失败，则打印错误信息
        seg = self._merge_reduplication(seg)  # 调用_merge_reduplication方法对分词结果进行处理
        seg = self._merge_continuous_three_tones(seg)  # 调用_merge_continuous_three_tones方法对分词结果进行处理
        seg = self._merge_continuous_three_tones_2(seg)  # 调用_merge_continuous_three_tones_2方法对分词结果进行处理
        seg = self._merge_er(seg)  # 调用 _merge_er 方法，将 seg 参数传入并返回结果赋值给 seg 变量
        return seg  # 返回经过处理后的 seg 变量

    def modified_tone(self, word: str, pos: str, finals: List[str]) -> List[str]:
        finals = self._bu_sandhi(word, finals)  # 调用 _bu_sandhi 方法，将 word 和 finals 参数传入并返回结果赋值给 finals 变量
        finals = self._yi_sandhi(word, finals)  # 调用 _yi_sandhi 方法，将 word 和 finals 参数传入并返回结果赋值给 finals 变量
        finals = self._neural_sandhi(word, pos, finals)  # 调用 _neural_sandhi 方法，将 word、pos 和 finals 参数传入并返回结果赋值给 finals 变量
        finals = self._three_sandhi(word, finals)  # 调用 _three_sandhi 方法，将 word 和 finals 参数传入并返回结果赋值给 finals 变量
        return finals  # 返回经过处理后的 finals 变量
```