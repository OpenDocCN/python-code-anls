# `Bert-VITS2\oldVersion\V210\text\tone_sandhi.py`

```
# 导入必要的模块
from typing import List
from typing import Tuple
import jieba
from pypinyin import lazy_pinyin
from pypinyin import Style

# 定义一个名为 ToneSandhi 的类
class ToneSandhi:
    # 定义一个名为 _bu_sandhi 的私有方法，接受一个字符串和一个列表作为参数，并返回一个列表
    # 该方法用于处理不的连读变调规则
    # 参数 word 表示输入的词语，finals 表示输入词语的韵母列表
    def _bu_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # 如果词语长度为3且第二个字是"不"，则将第二个韵母的声调修改为5
        if len(word) == 3 and word[1] == "不":
            finals[1] = finals[1][:-1] + "5"
        else:
            # 遍历词语中的每个字符和对应的韵母
            for i, char in enumerate(word):
                # 如果遇到"不"且后面一个字的声调为4，则将"不"的声调修改为2
                if char == "不" and i + 1 < len(word) and finals[i + 1][-1] == "4":
                    finals[i] = finals[i][:-1] + "2"
        # 返回处理后的韵母列表
        return finals
    # 对于给定的单词和韵母列表，处理包含"一"的情况
    def _yi_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # 如果单词中包含"一"，并且除了"一"以外的字符都是数字，则返回韵母列表
        if word.find("一") != -1 and all(
            [item.isnumeric() for item in word if item != "一"]
        ):
            return finals
        # 如果单词长度为3，且第二个字符是"一"，且第一个字符等于最后一个字符，则将韵母列表的第二个元音替换为yi5
        elif len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
            finals[1] = finals[1][:-1] + "5"
        # 如果"一"是序数词的一部分，则将韵母列表的第二个元音替换为yi1
        elif word.startswith("第一"):
            finals[1] = finals[1][:-1] + "1"
        else:
            for i, char in enumerate(word):
                if char == "一" and i + 1 < len(word):
                    # 如果"一"后面是声调4，则将韵母列表的第i个元音替换为yi2
                    if finals[i + 1][-1] == "4":
                        finals[i] = finals[i][:-1] + "2"
                    # 如果"一"后面不是声调4，则将韵母列表的第i个元音替换为yi4
                    else:
                        # 如果"一"后面是标点，则将韵母列表的第i个元音替换为yi4
                        if word[i + 1] not in self.punc:
                            finals[i] = finals[i][:-1] + "4"
        return finals

    # 分割单词，返回分割后的单词列表
    def _split_word(self, word: str) -> List[str]:
        # 使用结巴分词对单词进行分词
        word_list = jieba.cut_for_search(word)
        # 根据单词长度对分词结果进行排序
        word_list = sorted(word_list, key=lambda i: len(i), reverse=False)
        # 获取第一个子词和其在原单词中的起始位置
        first_subword = word_list[0]
        first_begin_idx = word.find(first_subword)
        # 如果第一个子词的起始位置为0，则第二个子词为原单词去除第一个子词后的部分
        if first_begin_idx == 0:
            second_subword = word[len(first_subword) :]
            new_word_list = [first_subword, second_subword]
        # 否则，第二个子词为原单词去除第一个子词前的部分
        else:
            second_subword = word[: -len(first_subword)]
            new_word_list = [second_subword, first_subword]
        return new_word_list
    # 对于一个词和它的韵母列表，处理三个连续的声调的情况
    def _three_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # 如果词的长度为2且所有韵母都是三声调
        if len(word) == 2 and self._all_tone_three(finals):
            # 将第一个韵母的声调改为二声调
            finals[0] = finals[0][:-1] + "2"
        # 如果词的长度为3
        elif len(word) == 3:
            word_list = self._split_word(word)
            if self._all_tone_three(finals):
                # 两个字的词 + 一个字的词，例如：蒙古/包
                if len(word_list[0]) == 2:
                    finals[0] = finals[0][:-1] + "2"
                    finals[1] = finals[1][:-1] + "2"
                # 一个字的词 + 两个字的词，例如：纸/老虎
                elif len(word_list[0]) == 1:
                    finals[1] = finals[1][:-1] + "2"
            else:
                finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]]
                if len(finals_list) == 2:
                    for i, sub in enumerate(finals_list):
                        # 例如：所有/人
                        if self._all_tone_three(sub) and len(sub) == 2:
                            finals_list[i][0] = finals_list[i][0][:-1] + "2"
                        # 例如：好/喜欢
                        elif (
                            i == 1
                            and not self._all_tone_three(sub)
                            and finals_list[i][0][-1] == "3"
                            and finals_list[0][-1][-1] == "3"
                        ):
                            finals_list[0][-1] = finals_list[0][-1][:-1] + "2"
                        finals = sum(finals_list, [])
        # 将成语拆分成两个长度为2的词
        elif len(word) == 4:
            finals_list = [finals[:2], finals[2:]]
            finals = []
            for sub in finals_list:
                if self._all_tone_three(sub):
                    sub[0] = sub[0][:-1] + "2"
                finals += sub
        return finals

    # 检查韵母列表中是否都是三声调
    def _all_tone_three(self, finals: List[str]) -> bool:
        return all(x[-1] == "3" for x in finals)

    # 合并"不"和它后面的词
    # 定义一个方法，用于合并分词结果中的"不"，避免出现单独的"不"导致的连读错误
    def _merge_bu(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []  # 创建一个新的分词结果列表
        last_word = ""  # 初始化上一个词为空字符串
        for word, pos in seg:  # 遍历分词结果中的每个词和词性
            if last_word == "不":  # 如果上一个词是"不"
                word = last_word + word  # 将当前词与上一个词合并
            if word != "不":  # 如果当前词不是"不"
                new_seg.append((word, pos))  # 将当前词和词性添加到新的分词结果列表中
            last_word = word[:]  # 更新上一个词为当前词
        if last_word == "不":  # 如果最后一个词是"不"
            new_seg.append((last_word, "d"))  # 将"不"和词性"d"添加到新的分词结果列表中
            last_word = ""  # 重置上一个词为空字符串
        return new_seg  # 返回合并后的分词结果列表
    
    # 定义一个方法，用于合并分词结果中的"一"和其左右相同的词，或者单独的"一"和其后的词
    # 避免出现单独的"一"导致的连读错误
    def _merge_yi(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []  # 创建一个新的分词结果列表
        # function 1
        for i, (word, pos) in enumerate(seg):  # 遍历分词结果中的每个词和词性
            if (
                i - 1 >= 0
                and word == "一"
                and i + 1 < len(seg)
                and seg[i - 1][0] == seg[i + 1][0]
                and seg[i - 1][1] == "v"
            ):  # 判断是否满足合并条件
                new_seg[i - 1][0] = new_seg[i - 1][0] + "一" + new_seg[i - 1][0]  # 合并符合条件的词
            else:
                if (
                    i - 2 >= 0
                    and seg[i - 1][0] == "一"
                    and seg[i - 2][0] == word
                    and pos == "v"
                ):  # 判断是否满足合并条件
                    continue  # 如果满足条件，则继续下一次循环
                else:
                    new_seg.append([word, pos])  # 将当前词和词性添加到新的分词结果列表中
        seg = new_seg  # 更新分词结果列表为新的分词结果列表
        new_seg = []  # 重置新的分词结果列表为空列表
        # function 2
        for i, (word, pos) in enumerate(seg):  # 遍历更新后的分词结果中的每个词和词性
            if new_seg and new_seg[-1][0] == "一":  # 如果新的分词结果列表不为空且最后一个词是"一"
                new_seg[-1][0] = new_seg[-1][0] + word  # 将当前词与"一"合并
            else:
                new_seg.append([word, pos])  # 将当前词和词性添加到新的分词结果列表中
        return new_seg  # 返回合并后的分词结果列表
    # 定义一个方法，用于合并连续的三声调的词语
    def _merge_continuous_three_tones(
        self, seg: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        # 创建一个新的词语列表
        new_seg = []
        # 获取每个词语的韵母和声调信息
        sub_finals_list = [
            lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        # 断言韵母列表和词语列表长度相同
        assert len(sub_finals_list) == len(seg)
        # 初始化一个标记列表，用于标记是否合并了上一个词语
        merge_last = [False] * len(seg)
        # 遍历词语列表
        for i, (word, pos) in enumerate(seg):
            # 判断是否需要合并当前词语和上一个词语
            if (
                i - 1 >= 0
                and self._all_tone_three(sub_finals_list[i - 1])
                and self._all_tone_three(sub_finals_list[i])
                and not merge_last[i - 1]
            ):
                # 如果上一个词语是重复的，不进行合并，因为重复需要进行中性音变
                if (
                    not self._is_reduplication(seg[i - 1][0])
                    and len(seg[i - 1][0]) + len(seg[i][0]) <= 3
                ):
                    # 合并当前词语和上一个词语
                    new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
                    merge_last[i] = True
                else:
                    new_seg.append([word, pos])
            else:
                new_seg.append([word, pos])

        return new_seg

    # 判断一个词语是否是重复的
    def _is_reduplication(self, word: str) -> bool:
        return len(word) == 2 and word[0] == word[1]

    # 判断第一个词语的最后一个字和第二个词语的第一个字是否都是三声调
    def _merge_continuous_three_tones_2(
        self, seg: List[Tuple[str, str]]
    # 定义一个函数，接受一个名为seg的列表参数，返回一个元组列表
    def _merge_finals_tone3(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # 创建一个空列表用于存储处理后的结果
        new_seg = []
        # 使用lazy_pinyin函数将seg中的每个单词转换为带声调的拼音，并存储在sub_finals_list中
        sub_finals_list = [
            lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        # 断言sub_finals_list和seg的长度相等
        assert len(sub_finals_list) == len(seg)
        # 创建一个布尔值列表，用于标记是否需要合并相邻的单词
        merge_last = [False] * len(seg)
        # 遍历seg中的每个单词和词性
        for i, (word, pos) in enumerate(seg):
            # 检查是否需要合并相邻的单词
            if (
                i - 1 >= 0
                and sub_finals_list[i - 1][-1][-1] == "3"
                and sub_finals_list[i][0][-1] == "3"
                and not merge_last[i - 1]
            ):
                # 如果上一个单词是重复的，且不需要合并，则将当前单词与上一个单词合并
                if (
                    not self._is_reduplication(seg[i - 1][0])
                    and len(seg[i - 1][0]) + len(seg[i][0]) <= 3
                ):
                    new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
                    merge_last[i] = True
                # 否则将当前单词添加到结果列表中
                else:
                    new_seg.append([word, pos])
            # 如果不需要合并，则将当前单词添加到结果列表中
            else:
                new_seg.append([word, pos])
        # 返回处理后的结果列表
        return new_seg

    # 定义一个函数，接受一个名为seg的列表参数，返回一个元组列表
    def _merge_er(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # 创建一个空列表用于存储处理后的结果
        new_seg = []
        # 遍历seg中的每个单词和词性
        for i, (word, pos) in enumerate(seg):
            # 如果当前单词是"儿"且前一个单词不是"#"
            if i - 1 >= 0 and word == "儿" and seg[i - 1][0] != "#":
                # 将当前单词与前一个单词合并
                new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
            # 否则将当前单词添加到结果列表中
            else:
                new_seg.append([word, pos])
        # 返回处理后的结果列表
        return new_seg

    # 定义一个函数，接受一个名为seg的列表参数���返回一个元组列表
    def _merge_reduplication(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # 创建一个空列表用于存储处理后的结果
        new_seg = []
        # 遍历seg中的每个单词和词性
        for i, (word, pos) in enumerate(seg):
            # 如果结果列表不为空且当前单词与上一个单词相同
            if new_seg and word == new_seg[-1][0]:
                # 将当前单词与上一个单词合并
                new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
            # 否则将当前单词添加到结果列表中
            else:
                new_seg.append([word, pos])
        # 返回处理后的结果列表
        return new_seg
    # 对输入的分词列表进行预处理，用于修改
    def pre_merge_for_modify(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # 合并不屈分词
        seg = self._merge_bu(seg)
        try:
            # 合并以分词
            seg = self._merge_yi(seg)
        except:
            # 如果合并以分词失败，则打印错误信息
            print("_merge_yi failed")
        # 合并重复分词
        seg = self._merge_reduplication(seg)
        # 合并连续三个音调的分词
        seg = self._merge_continuous_three_tones(seg)
        # 合并连续三个音调的分词（第二种方法）
        seg = self._merge_continuous_three_tones_2(seg)
        # 合并儿化分词
        seg = self._merge_er(seg)
        # 返回处理后的分词列表
        return seg
    
    # 修改音调
    def modified_tone(self, word: str, pos: str, finals: List[str]) -> List[str]:
        # 使用不变分词进行音变
        finals = self._bu_sandhi(word, finals)
        # 使用以变分词进行音变
        finals = self._yi_sandhi(word, finals)
        # 使用能变分词进行音变
        finals = self._neural_sandhi(word, pos, finals)
        # 使用三变分词进行音变
        finals = self._three_sandhi(word, finals)
        # 返回处理后的音变列表
        return finals
```