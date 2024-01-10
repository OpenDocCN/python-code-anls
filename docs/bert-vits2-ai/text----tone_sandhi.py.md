# `Bert-VITS2\text\tone_sandhi.py`

```
# 版权声明
# 2021年PaddlePaddle作者保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证以“原样”分发，
# 没有任何明示或暗示的保证或条件，
# 请参阅许可证以获取特定语言的权限和
# 许可证下的限制
from typing import List
from typing import Tuple

import jieba
from pypinyin import lazy_pinyin
from pypinyin import Style

# 定义一个ToneSandhi类
class ToneSandhi:
    # jieba词性标记的含义：https://blog.csdn.net/weixin_44174352/article/details/113731041
    # 例如：
    # word: "家里"
    # pos: "s"
    # finals: ['ia1', 'i3']
    # 定义一个内部方法_bu_sandhi，接受一个字符串word和一个列表finals，返回一个列表
    def _bu_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # 如果词长为3且第二个字是"不"，则将第二个韵母的声调改为5，例如"看不懂"
        if len(word) == 3 and word[1] == "不":
            finals[1] = finals[1][:-1] + "5"
        else:
            for i, char in enumerate(word):
                # 如果字符是"不"且后面一个字的声调是4，则将"不"的声调改为2，例如"不怕"
                if char == "不" and i + 1 < len(word) and finals[i + 1][-1] == "4":
                    finals[i] = finals[i][:-1] + "2"
        return finals
    # 对于给定的单词和韵母列表，处理包含"一"的情况
    def _yi_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # 如果单词中包含"一"，并且除了"一"以外的字符都是数字，则返回韵母列表
        if word.find("一") != -1 and all(
            [item.isnumeric() for item in word if item != "一"]
        ):
            return finals
        # 如果单词中间有"一"，并且前后两个字相同，则将韵母列表中对应位置的韵母修改为yi5
        elif len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
            finals[1] = finals[1][:-1] + "5"
        # 如果"一"是序数词的一部分，则将韵母列表中对应位置的韵母修改为yi1
        elif word.startswith("第一"):
            finals[1] = finals[1][:-1] + "1"
        else:
            for i, char in enumerate(word):
                if char == "一" and i + 1 < len(word):
                    # 如果"一"后面是声调4，则将韵母列表中对应位置的韵母修改为yi2
                    if finals[i + 1][-1] == "4":
                        finals[i] = finals[i][:-1] + "2"
                    # 如果"一"后面不是声调4，则将韵母列表中对应位置的韵母修改为yi4
                    else:
                        # 如果"一"后面是标点，则将韵母列表中对应位置的韵母修改为yi4
                        if word[i + 1] not in self.punc:
                            finals[i] = finals[i][:-1] + "4"
        return finals

    # 分割单词，返回分割后的单词列表
    def _split_word(self, word: str) -> List[str]:
        # 使用结巴分词对单词进行分词，并按照长度升序排序
        word_list = jieba.cut_for_search(word)
        word_list = sorted(word_list, key=lambda i: len(i), reverse=False)
        # 获取分词后的第一个子词和其在原单词中的起始位置
        first_subword = word_list[0]
        first_begin_idx = word.find(first_subword)
        # 如果第一个子词在单词的开头，则将单词分割成两部分
        if first_begin_idx == 0:
            second_subword = word[len(first_subword) :]
            new_word_list = [first_subword, second_subword]
        # 如果第一个子词不在单词的开头，则将单词分割成两部分
        else:
            second_subword = word[: -len(first_subword)]
            new_word_list = [second_subword, first_subword]
        return new_word_list
    # 对于给定的词和韵母列表，处理三个音调的情况
    def _three_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # 如果词的长度为2且所有韵母都是三声调
        if len(word) == 2 and self._all_tone_three(finals):
            # 将第一个韵母的声调改为二声调
            finals[0] = finals[0][:-1] + "2"
        # 如果词的长度为3
        elif len(word) == 3:
            # 将词拆分成单个字
            word_list = self._split_word(word)
            # 如果所有韵母都是三声调
            if self._all_tone_three(finals):
                # 如果第一个字的长度为2
                if len(word_list[0]) == 2:
                    # 将第一个韵母的声调改为二声调
                    finals[0] = finals[0][:-1] + "2"
                    # 将第二个韵母的声调改为二声调
                    finals[1] = finals[1][:-1] + "2"
                # 如果第一个字的长度为1
                elif len(word_list[0]) == 1:
                    # 将第二个韵母的声调改为二声调
                    finals[1] = finals[1][:-1] + "2"
            else:
                # 将韵母列表拆分成两部分
                finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]]
                # 如果拆分后有两部分
                if len(finals_list) == 2:
                    for i, sub in enumerate(finals_list):
                        # 如果所有韵母都是三声调且长度为2
                        if self._all_tone_three(sub) and len(sub) == 2:
                            # 将韵母的声调改为二声调
                            finals_list[i][0] = finals_list[i][0][:-1] + "2"
                        # 如果是第二部分且不是所有韵母都是三声调，且最后一个韵母是三声调，且第一部分的最后一个韵母也是三声调
                        elif (
                            i == 1
                            and not self._all_tone_three(sub)
                            and finals_list[i][0][-1] == "3"
                            and finals_list[0][-1][-1] == "3"
                        ):
                            # 将第一部分的最后一个韵母的声调改为二声调
                            finals_list[0][-1] = finals_list[0][-1][:-1] + "2"
                        # 合并两部分韵母列表
                        finals = sum(finals_list, [])
        # 如果词的长度为4
        elif len(word) == 4:
            # 将韵母列表拆分成两部分
            finals_list = [finals[:2], finals[2:]]
            finals = []
            for sub in finals_list:
                # 如果所有韵母都是三声调
                if self._all_tone_three(sub):
                    # 将韵母的声调改为二声调
                    sub[0] = sub[0][:-1] + "2"
                # 合并韵母列表
                finals += sub
        # 返回处理后的韵母列表
        return finals

    # 判断韵母列表中是否所有韵母都是三声调
    def _all_tone_three(self, finals: List[str]) -> bool:
        return all(x[-1] == "3" for x in finals)

    # 合并"不"和其后的词
    # 如果不合并，根据结巴分词，"不" 有时会单独出现，这可能会导致连读错误
    def _merge_bu(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # 创建一个新的分词列表
        new_seg = []
        # 初始化上一个词为空字符串
        last_word = ""
        # 遍历分词列表中的每个词和词性
        for word, pos in seg:
            # 如果上一个词是"不"
            if last_word == "不":
                # 将当前词与上一个词合并
                word = last_word + word
            # 如果当前词不是"不"
            if word != "不":
                # 将当前词和词性添加到新的分词列表中
                new_seg.append((word, pos))
            # 更新上一个词为当前词
            last_word = word[:]
        # 如果上一个词是"不"
        if last_word == "不":
            # 将"不"和词性添加到新的分词列表中
            new_seg.append((last_word, "d"))
            # 重置上一个词为空字符串
            last_word = ""
        # 返回新的分词列表
        return new_seg
    
    # 函数1：合并"一"和它左右的重复词，例如"听","一","听" ->"听一听"
    # 函数2：合并单独的"一"和它后面的词
    # 如果不合并，根据结巴分词，"一" 有时会单独出现，这可能会导致连读错误
    # 例如：
    # 输入分词列表: [('听', 'v'), ('一', 'm'), ('听', 'v')]
    # 输出分词列表: [['听一听', 'v']]
    # 合并具有特定条件的词语，返回合并后的分词结果
    def _merge_yi(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # 创建一个新的空列表
        new_seg = [] * len(seg)
        # function 1
        # 初始化索引 i
        i = 0
        # 循环直到 i 等于分词列表的长度
        while i < len(seg):
            # 获取当前词语和词性
            word, pos = seg[i]
            # 检查特定条件是否满足，如果满足则合并词语
            if (
                i - 1 >= 0
                and word == "一"
                and i + 1 < len(seg)
                and seg[i - 1][0] == seg[i + 1][0]
                and seg[i - 1][1] == "v"
            ):
                new_seg[i - 1][0] = new_seg[i - 1][0] + "一" + new_seg[i - 1][0]
                i += 2
            else:
                # 检查另一种特定条件，如果满足则跳过当前词语
                if (
                    i - 2 >= 0
                    and seg[i - 1][0] == "一"
                    and seg[i - 2][0] == word
                    and pos == "v"
                ):
                    continue
                else:
                    new_seg.append([word, pos])
                i += 1
        # 重新赋值分词列表，去除空列表项
        seg = [i for i in new_seg if len(i) > 0]
        new_seg = []
        # function 2
        # 遍历分词列表，合并具有特定条件的词语
        for i, (word, pos) in enumerate(seg):
            if new_seg and new_seg[-1][0] == "一":
                new_seg[-1][0] = new_seg[-1][0] + word
            else:
                new_seg.append([word, pos])
        # 返回合并后的分词结果
        return new_seg

    # the first and the second words are all_tone_three
    # 合并具有特定条件的词语，返回合并后的分词结果
    def _merge_continuous_three_tones(
        self, seg: List[Tuple[str, str]]
    # 定义函数，接受一个参数并返回一个元组列表
    ) -> List[Tuple[str, str]]:
        # 初始化一个空列表
        new_seg = []
        # 使用lazy_pinyin函数将seg中的每个词转换成带声调的拼音，并存储在sub_finals_list中
        sub_finals_list = [
            lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        # 断言sub_finals_list和seg的长度相等
        assert len(sub_finals_list) == len(seg)
        # 初始化一个布尔值列表，长度与seg相等
        merge_last = [False] * len(seg)
        # 遍历seg中的每个词和词性
        for i, (word, pos) in enumerate(seg):
            # 检查是否可以合并连续的带声调的拼音
            if (
                i - 1 >= 0
                and self._all_tone_three(sub_finals_list[i - 1])
                and self._all_tone_three(sub_finals_list[i])
                and not merge_last[i - 1]
            ):
                # 如果上一个词是重复的，不合并，因为重复需要进行_neural_sandhi
                if (
                    not self._is_reduplication(seg[i - 1][0])
                    and len(seg[i - 1][0]) + len(seg[i][0]) <= 3
                ):
                    # 合并相邻的词
                    new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
                    merge_last[i] = True
                else:
                    # 将当前词添加到新的列表中
                    new_seg.append([word, pos])
            else:
                # 将当前词添加到新的列表中
                new_seg.append([word, pos])

        # 返回新的列表
        return new_seg

    # 判断一个词是否是重复的
    def _is_reduplication(self, word: str) -> bool:
        return len(word) == 2 and word[0] == word[1]

    # 合并连续的带声调的拼音
    # 第一个词的最后一个字符和第二个词的第一个字符都是带声调的
    def _merge_continuous_three_tones_2(
        self, seg: List[Tuple[str, str]]
    # 定义一个函数，接受一个名为seg的列表参数，返回一个元组列表
    def _merge_finals_tone3(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # 创建一个空列表用于存储处理后的分词结果
        new_seg = []
        # 使用lazy_pinyin函数将seg中的每个词转换为带声调的韵母列表
        sub_finals_list = [
            lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        # 断言sub_finals_list和seg的长度相等
        assert len(sub_finals_list) == len(seg)
        # 创建一个与seg长度相等的列表，用于标记是否需要合并
        merge_last = [False] * len(seg)
        # 遍历seg中的每个词和词性
        for i, (word, pos) in enumerate(seg):
            # 判断是否需要合并
            if (
                i - 1 >= 0
                and sub_finals_list[i - 1][-1][-1] == "3"
                and sub_finals_list[i][0][-1] == "3"
                and not merge_last[i - 1]
            ):
                # 如果上一个词是重复的，且不需要合并，则将当前词与上一个词合并
                if (
                    not self._is_reduplication(seg[i - 1][0])
                    and len(seg[i - 1][0]) + len(seg[i][0]) <= 3
                ):
                    new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
                    merge_last[i] = True
                # 否则将当前词和词性添加到new_seg中
                else:
                    new_seg.append([word, pos])
            # 如果不需要合并，则将当前词和词性添加到new_seg中
            else:
                new_seg.append([word, pos])
        # 返回处理后的分词结果
        return new_seg

    # 定义一个函数，接受一个名为seg的列表参数，返回一个元组列表
    def _merge_er(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # 创建一个空列表用于存储处理后的分词结果
        new_seg = []
        # 遍历seg中的每个词和词性
        for i, (word, pos) in enumerate(seg):
            # 如果当前词是"儿"且上一个词不是"#"
            if i - 1 >= 0 and word == "儿" and seg[i - 1][0] != "#":
                # 将当前词与上一个词合并
                new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
            # 否则将当前词和词性添加到new_seg中
            else:
                new_seg.append([word, pos])
        # 返回处理后的分词结果
        return new_seg

    # 定义一个函数，接受一个名为seg的列表参数，返回一个元组列表
    def _merge_reduplication(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # 创建一个空列表用于存储处理后的分词结果
        new_seg = []
        # 遍历seg中的每个词和词性
        for i, (word, pos) in enumerate(seg):
            # 如果new_seg不为空且当前词与上一个词相同
            if new_seg and word == new_seg[-1][0]:
                # 将当前词与上一个词合并
                new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
            # 否则将当前词和词性添加到new_seg中
            else:
                new_seg.append([word, pos])
        # 返回处理后的分词结果
        return new_seg
    # 对输入的分词列表进行预处理，用于修改
    def pre_merge_for_modify(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # 合并不属于的分词
        seg = self._merge_bu(seg)
        # 尝试合并属于的分词，如果失败则打印错误信息
        try:
            seg = self._merge_yi(seg)
        except:
            print("_merge_yi failed")
        # 合并重复的分词
        seg = self._merge_reduplication(seg)
        # 合并连续的三个音调
        seg = self._merge_continuous_three_tones(seg)
        # 再次合并连续的三个音调
        seg = self._merge_continuous_three_tones_2(seg)
        # 合并属于的分词
        seg = self._merge_er(seg)
        # 返回处理后的分词列表
        return seg

    # 修改音调
    def modified_tone(self, word: str, pos: str, finals: List[str]) -> List[str]:
        # 使用不合的音变规则
        finals = self._bu_sandhi(word, finals)
        # 使用属于的音变规则
        finals = self._yi_sandhi(word, finals)
        # 使用轻音变规则
        finals = self._neural_sandhi(word, pos, finals)
        # 使用三个音调的音变规则
        finals = self._three_sandhi(word, finals)
        # 返回处理后的音调列表
        return finals
```