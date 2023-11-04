# BertVITS2源码解析 4

# `text/japanese_bert.py`

这段代码的作用是使用预训练的 BERT 模型来对文本进行编码，并将编码后的输出结果存储在字典中。具体来说，它实现了以下几个功能：

1. 从 `transformers` 库中加载了预训练的 BERT-base-japanese-v3 模型，并初始化了一个字典 `models`，用于存储编码后的结果。
2. 定义了一个函数 `get_bert_feature`，该函数接收一个文本表示和一个 word2ph  mapping，返回一个 BERT 模型的 features。该函数使用 BERT 模型的预训练权重在计算输入文本的编码，并将编码结果存储在 `device` 变量中，如果 `device` 环境变量没有被指定，则使用 CPU 设备进行计算。
3. 在 `get_bert_feature` 函数中，加载了预训练的 BERT 模型，并将其应用到 `tokenizer` 对象的 `from_pretrained` 方法中，以便从模型中获取预训练的 BERT 模型。然后，定义了一个 `device` 变量，用于在 `models` 字典中存储编码设备的标识符。
4. 在 `get_bert_feature` 函数中，定义了一个函数 `from_bert`，该函数接收一个 `tokenizer` 对象的 `from_pretrained` 方法，加载了预训练的 BERT 模型，并将其应用到 `device` 变量中。然后，使用 `tokenizer` 对象的 `encode` 方法对传入的文本进行编码，并将编码结果存储在 `device` 变量中。
5. 在 `tokenizer` 对象中，从 `bert-base-japanese-v3` 标识符的预训练权重在 `./bert/bert-base-japanese-v3` 目录中加载了 BERT 模型的预训练权重。
6. 在 `get_bert_feature` 函数中，定义了一个 `tokenizer` 对象，该对象使用 `AutoTokenizer` 类，从 `tokenizer` 对象中加载了预训练的 BERT 模型，并将其存储在 `AutoTokenizer.from_pretrained` 方法中。
7. 在 `get_bert_feature` 函数中，定义了一个 `models` 字典，用于存储编码后的结果。
8. 在 `tokenizer` 对象的 `from_pretrained` 方法中，传递了以下参数：
	- `model`:BERT模型的模型。
	- `save_pretrained_pretrained`:True，保存的模型为同一文件夹下的 model_name。
	- `num_labels`:None，不支持标签。
	- `is_linear`:False,False: 是线性。
	- `normalization_class_token_connections`:True，是。
	- `token_type_ids`:True，是。
	- `max_output_length`:256，是。



```py
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys

tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")

models = dict()


def get_bert_feature(text, word2ph, device=None):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/bert-base-japanese-v3"
        ).to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T

```

# `text/symbols.py`

这段代码定义了一个字符串变量 `punctuation`，其中包括常见的标点符号和一些特殊符号，如问号、感叹号、波浪号、减号等。

接着定义了一个字符串变量 `pu_symbols`，将 `punctuation` 和一些特定的符号(如下划线、单引号、双引号)合并起来。

然后定义了一个字符串变量 `pad`，是一个空字符串，用于在输入的数据中填充不足的字符。

接下来是定义一个包含常见中文汉字的字符串变量 `zh_symbols`，包括汉字拼音中的所有字母和一些特殊符号，如问号、叹号、引号等。

接着是定义一个字符串变量 `chinese_remaining_symbols`，包含除了 `zh_symbols` 中以外的所有字符和符号，如数轴上的符号、特殊符号等。

然后定义一个循环，将 `zh_symbols` 和一些特定的符号(如下划线、单引号、双引号)合并成一个字符串，并将结果保存到 `zh_chinese_remaining_symbols` 变量中。

接着是定义一个循环，将 `punctuation` 和一些特定的符号(如问号、感叹号、波浪号、减号等)合并成一个字符串，并将结果保存到 `punctuation_chinese_remaining_symbols` 变量中。

最后是定义一个循环，将 `pad` 和一些特定的符号(如空格、制表符、回车等)合并成一个字符串，并将结果保存到 `padded_chinese_remaining_symbols` 变量中。


```py
punctuation = ["!", "?", "…", ",", ".", "'", "-"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# chinese
zh_symbols = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
```

这段代码定义了一个名为 `num_zh_tones` 的变量，并将其初始化为 6。这个变量可能是用于计数某种东西的数量，比如中文谐音的数量。

接下来定义了一个包含 21 个 Japanese 字符的列表 `ja_symbols`。这些字符包括了日语中的平假名、片假名以及一些特殊符号。可以用于进行文本处理或者机器翻译等任务。


```py
num_zh_tones = 6

# japanese
ja_symbols = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "w",
    "y",
    "z",
    "zy",
]
```

这段代码定义了一个名为 `num_ja_tones` 的变量，并将其初始化为1。

具体来说，`num_ja_tones` 是一个整数类型（整型），它的值是1。这个变量的作用是用来表示日语假名（平假名、片假名）中元音数量的一个整数。在日语中，元音是指发音时声带振动且气流不受阻碍地通过口腔的音。


```py
num_ja_tones = 1

# English
en_symbols = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
```

这段代码的作用是计算日语（"ZH"）、朝鲜语（"JP"）和英语（"EN"）语素单位的数量，以及它们在拼音表中的编号。

首先，将三种语言的符号列表合并，并去重，得到一个包含所有符号的列表。

然后，根据给定的编号，将每个语素单位对准一个数字，分别加入到已知的音节数组中。

接下来，计算将实现的语素单位的总数，包括跨越多种语言的语素单位。

最后，根据所给的编号，将每种语言的音节数量存储在一个字典中，其中键是语言名称，值是音素单位的编号。


```py
num_en_tones = 4

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))
symbols = [pad] + normal_symbols + pu_symbols
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}
num_languages = len(language_id_map.keys())

language_tone_start_map = {
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

```

这段代码的作用是检查当前脚本是否作为主程序运行，如果不是，则执行以下操作：

1. 从 `zh_symbols` 集合中创建一个空集合 `a`，从 `en_symbols` 集合中创建一个空集合 `b`。
2. 对两个集合 `a` 和 `b` 进行交集操作，并将结果通过 `sorted` 函数进行排序。
3. 最后，输出排好序的元组数组 `a & b`。


```py
if __name__ == "__main__":
    a = set(zh_symbols)
    b = set(en_symbols)
    print(sorted(a & b))

```

# `text/tone_sandhi.py`

这段代码定义了一个名为`Tuple`的类，该类用于表示元组（Tuple）类型。这个类允许我们将一个或多个类型组合成一个元组，这个元组可以用类型名称、数字名称或字符串名称来引用。

具体来说，这段代码定义了一个名为`Tuple`的类，它有以下属性和方法：

* `__init__`方法接收两个参数，第一个参数是一个包含多个类型的列表，第二个参数是一个元组类型。这个方法创建一个新的元组对象，初始化它的值为`Tuple`类中定义的类型之一。
* `__repr__`方法返回一个表示元组对象的简洁字符串。这个字符串由两个空格和两个圆括号组成，第一个空格用于显示第一个元素，第二个空格用于显示第二个元素。
* `__len__`方法返回元组对象中元素的数量。

下面是一个示例，展示了如何使用这个类来创建和访问元组对象：
```pypython
from typing import Tuple

# 创建一个包含两个整数的元组
tuple_a = Tuple([1, 2])
tuple_b = Tuple([3, 4])

# 访问元组对象的第一个元素
print(tuple_a[0])  # 输出：1

# 访问元组对象的第二个元素
print(tuple_b[1])  # 输出：2

# 打印元组对象的长度
print(len(tuple_a))  # 输出：2

# 创建一个新的元组对象
tuple_c = Tuple([5, 6])

# 打印元组对象的第一个元素
print(tuple_c[0])  # 输出：5

# 打印元组对象的第二个元素
print(tuple_c[1])  # 输出：6
```
这段代码定义了一个`Tuple`类，用于将多个类型组合成一个元组。通过创建新的`Tuple`对象，我们可以引用元组中的类型，就像通常在Python中使用列表来引用元素一样。


```py
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
from typing import Tuple

```

This is a class that manages Chinese text segmentation. It has a method `pre_merge_for_modify` that takes a list of segments and returns a modified list of segments.

The `pre_merge_for_modify` method first calls the private method `_merge_bu` on the segment list, then it tries to call the private method `_merge_yi` on the segment list. If either method fails, it will print an error message.

If either `_merge_bu` or `_merge_yi` is successful, it will call the private method `_merge_reduplication` on the segment list.

Finally, it calls the `_merge_continuous_three_tones` and `_merge_continuous_three_tones_2` methods on the segment list, which are used to convert segments with continuous tone particles to simplified characters.

The class also has a method `merge`, which takes a list of segments, and returns a list of modified segments. This method is not guaranteed to work correctly, as it has not been fully tested.


```py
import jieba
from pypinyin import lazy_pinyin
from pypinyin import Style


class ToneSandhi:
    def __init__(self):
        self.must_neural_tone_words = {
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
            "胭脂",
            "胡萝",
            "胡琴",
            "胡同",
            "聪明",
            "耽误",
            "耽搁",
            "耷拉",
            "耳朵",
            "老爷",
            "老实",
            "老婆",
            "老头",
            "老太",
            "翻腾",
            "罗嗦",
            "罐头",
            "编辑",
            "结实",
            "红火",
            "累赘",
            "糨糊",
            "糊涂",
            "精神",
            "粮食",
            "簸箕",
            "篱笆",
            "算计",
            "算盘",
            "答应",
            "笤帚",
            "笑语",
            "笑话",
            "窟窿",
            "窝囊",
            "窗户",
            "稳当",
            "稀罕",
            "称呼",
            "秧歌",
            "秀气",
            "秀才",
            "福气",
            "祖宗",
            "砚台",
            "码头",
            "石榴",
            "石头",
            "石匠",
            "知识",
            "眼睛",
            "眯缝",
            "眨巴",
            "眉毛",
            "相声",
            "盘算",
            "白净",
            "痢疾",
            "痛快",
            "疟疾",
            "疙瘩",
            "疏忽",
            "畜生",
            "生意",
            "甘蔗",
            "琵琶",
            "琢磨",
            "琉璃",
            "玻璃",
            "玫瑰",
            "玄乎",
            "狐狸",
            "状元",
            "特务",
            "牲口",
            "牙碜",
            "牌楼",
            "爽快",
            "爱人",
            "热闹",
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
            "故事",
            "收拾",
            "收成",
            "提防",
            "挖苦",
            "挑剔",
            "指甲",
            "指头",
            "拾掇",
            "拳头",
            "拨弄",
            "招牌",
            "招呼",
            "抬举",
            "护士",
            "折腾",
            "扫帚",
            "打量",
            "打算",
            "打点",
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
            "心思",
            "得罪",
            "张罗",
            "弟兄",
            "开通",
            "应酬",
            "庄稼",
            "干事",
            "帮手",
            "帐篷",
            "希罕",
            "师父",
            "师傅",
            "巴结",
            "巴掌",
            "差事",
            "工夫",
            "岁数",
            "屁股",
            "尾巴",
            "少爷",
            "小气",
            "小伙",
            "将就",
            "对头",
            "对付",
            "寡妇",
            "家伙",
            "客气",
            "实在",
            "官司",
            "学问",
            "学生",
            "字号",
            "嫁妆",
            "媳妇",
            "媒人",
            "婆家",
            "娘家",
            "委屈",
            "姑娘",
            "姐夫",
            "妯娌",
            "妥当",
            "妖精",
            "奴才",
            "女婿",
            "头发",
            "太阳",
            "大爷",
            "大方",
            "大意",
            "大夫",
            "多少",
            "多么",
            "外甥",
            "壮实",
            "地道",
            "地方",
            "在乎",
            "困难",
            "嘴巴",
            "嘱咐",
            "嘟囔",
            "嘀咕",
            "喜欢",
            "喇嘛",
            "喇叭",
            "商量",
            "唾沫",
            "哑巴",
            "哈欠",
            "哆嗦",
            "咳嗽",
            "和尚",
            "告诉",
            "告示",
            "含糊",
            "吓唬",
            "后头",
            "名字",
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
            "利落",
            "利索",
            "利害",
            "分析",
            "出息",
            "凑合",
            "凉快",
            "冷战",
            "冤枉",
            "冒失",
            "养活",
            "关系",
            "先生",
            "兄弟",
            "便宜",
            "使唤",
            "佩服",
            "作坊",
            "体面",
            "位置",
            "似的",
            "伙计",
            "休息",
            "什么",
            "人家",
            "亲戚",
            "亲家",
            "交情",
            "云彩",
            "事情",
            "买卖",
            "主意",
            "丫头",
            "丧气",
            "两口",
            "东西",
            "东家",
            "世故",
            "不由",
            "不在",
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
            "牢骚",
            "咖喱",
            "扫把",
            "惦记",
        }
        self.must_not_neural_tone_words = {
            "男子",
            "女子",
            "分子",
            "原子",
            "量子",
            "莲子",
            "石子",
            "瓜子",
            "电子",
            "人人",
            "虎虎",
        }
        self.punc = "：，；。？！“”‘’':,;.?!"

    # the meaning of jieba pos tag: https://blog.csdn.net/weixin_44174352/article/details/113731041
    # e.g.
    # word: "家里"
    # pos: "s"
    # finals: ['ia1', 'i3']
    def _neural_sandhi(self, word: str, pos: str, finals: List[str]) -> List[str]:
        # reduplication words for n. and v. e.g. 奶奶, 试试, 旺旺
        for j, item in enumerate(word):
            if (
                j - 1 >= 0
                and item == word[j - 1]
                and pos[0] in {"n", "v", "a"}
                and word not in self.must_not_neural_tone_words
            ):
                finals[j] = finals[j][:-1] + "5"
        ge_idx = word.find("个")
        if len(word) >= 1 and word[-1] in "吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶":
            finals[-1] = finals[-1][:-1] + "5"
        elif len(word) >= 1 and word[-1] in "的地得":
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 走了, 看着, 去过
        # elif len(word) == 1 and word in "了着过" and pos in {"ul", "uz", "ug"}:
        #     finals[-1] = finals[-1][:-1] + "5"
        elif (
            len(word) > 1
            and word[-1] in "们子"
            and pos in {"r", "n"}
            and word not in self.must_not_neural_tone_words
        ):
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 桌上, 地下, 家里
        elif len(word) > 1 and word[-1] in "上下里" and pos in {"s", "l", "f"}:
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 上来, 下去
        elif len(word) > 1 and word[-1] in "来去" and word[-2] in "上下进出回过起开":
            finals[-1] = finals[-1][:-1] + "5"
        # 个做量词
        elif (
            ge_idx >= 1
            and (word[ge_idx - 1].isnumeric() or word[ge_idx - 1] in "几有两半多各整每做是")
        ) or word == "个":
            finals[ge_idx] = finals[ge_idx][:-1] + "5"
        else:
            if (
                word in self.must_neural_tone_words
                or word[-2:] in self.must_neural_tone_words
            ):
                finals[-1] = finals[-1][:-1] + "5"

        word_list = self._split_word(word)
        finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]]
        for i, word in enumerate(word_list):
            # conventional neural in Chinese
            if (
                word in self.must_neural_tone_words
                or word[-2:] in self.must_neural_tone_words
            ):
                finals_list[i][-1] = finals_list[i][-1][:-1] + "5"
        finals = sum(finals_list, [])
        return finals

    def _bu_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # e.g. 看不懂
        if len(word) == 3 and word[1] == "不":
            finals[1] = finals[1][:-1] + "5"
        else:
            for i, char in enumerate(word):
                # "不" before tone4 should be bu2, e.g. 不怕
                if char == "不" and i + 1 < len(word) and finals[i + 1][-1] == "4":
                    finals[i] = finals[i][:-1] + "2"
        return finals

    def _yi_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # "一" in number sequences, e.g. 一零零, 二一零
        if word.find("一") != -1 and all(
            [item.isnumeric() for item in word if item != "一"]
        ):
            return finals
        # "一" between reduplication words should be yi5, e.g. 看一看
        elif len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
            finals[1] = finals[1][:-1] + "5"
        # when "一" is ordinal word, it should be yi1
        elif word.startswith("第一"):
            finals[1] = finals[1][:-1] + "1"
        else:
            for i, char in enumerate(word):
                if char == "一" and i + 1 < len(word):
                    # "一" before tone4 should be yi2, e.g. 一段
                    if finals[i + 1][-1] == "4":
                        finals[i] = finals[i][:-1] + "2"
                    # "一" before non-tone4 should be yi4, e.g. 一天
                    else:
                        # "一" 后面如果是标点，还读一声
                        if word[i + 1] not in self.punc:
                            finals[i] = finals[i][:-1] + "4"
        return finals

    def _split_word(self, word: str) -> List[str]:
        word_list = jieba.cut_for_search(word)
        word_list = sorted(word_list, key=lambda i: len(i), reverse=False)
        first_subword = word_list[0]
        first_begin_idx = word.find(first_subword)
        if first_begin_idx == 0:
            second_subword = word[len(first_subword) :]
            new_word_list = [first_subword, second_subword]
        else:
            second_subword = word[: -len(first_subword)]
            new_word_list = [second_subword, first_subword]
        return new_word_list

    def _three_sandhi(self, word: str, finals: List[str]) -> List[str]:
        if len(word) == 2 and self._all_tone_three(finals):
            finals[0] = finals[0][:-1] + "2"
        elif len(word) == 3:
            word_list = self._split_word(word)
            if self._all_tone_three(finals):
                #  disyllabic + monosyllabic, e.g. 蒙古/包
                if len(word_list[0]) == 2:
                    finals[0] = finals[0][:-1] + "2"
                    finals[1] = finals[1][:-1] + "2"
                #  monosyllabic + disyllabic, e.g. 纸/老虎
                elif len(word_list[0]) == 1:
                    finals[1] = finals[1][:-1] + "2"
            else:
                finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]]
                if len(finals_list) == 2:
                    for i, sub in enumerate(finals_list):
                        # e.g. 所有/人
                        if self._all_tone_three(sub) and len(sub) == 2:
                            finals_list[i][0] = finals_list[i][0][:-1] + "2"
                        # e.g. 好/喜欢
                        elif (
                            i == 1
                            and not self._all_tone_three(sub)
                            and finals_list[i][0][-1] == "3"
                            and finals_list[0][-1][-1] == "3"
                        ):
                            finals_list[0][-1] = finals_list[0][-1][:-1] + "2"
                        finals = sum(finals_list, [])
        # split idiom into two words who's length is 2
        elif len(word) == 4:
            finals_list = [finals[:2], finals[2:]]
            finals = []
            for sub in finals_list:
                if self._all_tone_three(sub):
                    sub[0] = sub[0][:-1] + "2"
                finals += sub

        return finals

    def _all_tone_three(self, finals: List[str]) -> bool:
        return all(x[-1] == "3" for x in finals)

    # merge "不" and the word behind it
    # if don't merge, "不" sometimes appears alone according to jieba, which may occur sandhi error
    def _merge_bu(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        last_word = ""
        for word, pos in seg:
            if last_word == "不":
                word = last_word + word
            if word != "不":
                new_seg.append((word, pos))
            last_word = word[:]
        if last_word == "不":
            new_seg.append((last_word, "d"))
            last_word = ""
        return new_seg

    # function 1: merge "一" and reduplication words in it's left and right, e.g. "听","一","听" ->"听一听"
    # function 2: merge single  "一" and the word behind it
    # if don't merge, "一" sometimes appears alone according to jieba, which may occur sandhi error
    # e.g.
    # input seg: [('听', 'v'), ('一', 'm'), ('听', 'v')]
    # output seg: [['听一听', 'v']]
    def _merge_yi(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        # function 1
        for i, (word, pos) in enumerate(seg):
            if (
                i - 1 >= 0
                and word == "一"
                and i + 1 < len(seg)
                and seg[i - 1][0] == seg[i + 1][0]
                and seg[i - 1][1] == "v"
            ):
                new_seg[i - 1][0] = new_seg[i - 1][0] + "一" + new_seg[i - 1][0]
            else:
                if (
                    i - 2 >= 0
                    and seg[i - 1][0] == "一"
                    and seg[i - 2][0] == word
                    and pos == "v"
                ):
                    continue
                else:
                    new_seg.append([word, pos])
        seg = new_seg
        new_seg = []
        # function 2
        for i, (word, pos) in enumerate(seg):
            if new_seg and new_seg[-1][0] == "一":
                new_seg[-1][0] = new_seg[-1][0] + word
            else:
                new_seg.append([word, pos])
        return new_seg

    # the first and the second words are all_tone_three
    def _merge_continuous_three_tones(
        self, seg: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        new_seg = []
        sub_finals_list = [
            lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        assert len(sub_finals_list) == len(seg)
        merge_last = [False] * len(seg)
        for i, (word, pos) in enumerate(seg):
            if (
                i - 1 >= 0
                and self._all_tone_three(sub_finals_list[i - 1])
                and self._all_tone_three(sub_finals_list[i])
                and not merge_last[i - 1]
            ):
                # if the last word is reduplication, not merge, because reduplication need to be _neural_sandhi
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

    def _is_reduplication(self, word: str) -> bool:
        return len(word) == 2 and word[0] == word[1]

    # the last char of first word and the first char of second word is tone_three
    def _merge_continuous_three_tones_2(
        self, seg: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        new_seg = []
        sub_finals_list = [
            lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        assert len(sub_finals_list) == len(seg)
        merge_last = [False] * len(seg)
        for i, (word, pos) in enumerate(seg):
            if (
                i - 1 >= 0
                and sub_finals_list[i - 1][-1][-1] == "3"
                and sub_finals_list[i][0][-1] == "3"
                and not merge_last[i - 1]
            ):
                # if the last word is reduplication, not merge, because reduplication need to be _neural_sandhi
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
            if i - 1 >= 0 and word == "儿" and seg[i - 1][0] != "#":
                new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
            else:
                new_seg.append([word, pos])
        return new_seg

    def _merge_reduplication(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        for i, (word, pos) in enumerate(seg):
            if new_seg and word == new_seg[-1][0]:
                new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
            else:
                new_seg.append([word, pos])
        return new_seg

    def pre_merge_for_modify(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        seg = self._merge_bu(seg)
        try:
            seg = self._merge_yi(seg)
        except:
            print("_merge_yi failed")
        seg = self._merge_reduplication(seg)
        seg = self._merge_continuous_three_tones(seg)
        seg = self._merge_continuous_three_tones_2(seg)
        seg = self._merge_er(seg)
        return seg

    def modified_tone(self, word: str, pos: str, finals: List[str]) -> List[str]:
        finals = self._bu_sandhi(word, finals)
        finals = self._yi_sandhi(word, finals)
        finals = self._neural_sandhi(word, pos, finals)
        finals = self._three_sandhi(word, finals)
        return finals

```

# `text/__init__.py`

这段代码的作用是定义了一个名为“cleaned_text_to_sequence”的函数，它接受三个参数：清洁后的文本、音调和目标语言。这个函数返回的是符号序列编号，即文本中的符号映射到相应的数字编号。

具体来说，代码首先定义了一个名为“_symbol_to_id”的字典，将文本中的符号映射到对应的数字。接着，定义了一个名为“cleaned_text_to_sequence”的函数，它接受三个参数：清洁后的文本、音调和目标语言。函数内部先将清洁后的文本转换成相应的符号序列，然后根据目标语言将符号序列映射到对应的语言编号，最终返回符号序列、音调和语言编号。


```py
from text.symbols import *

_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


```

这段代码定义了一个名为 `get_bert` 的函数，它接受四个参数：`norm_text`、`word2ph`、`language` 和 `device`。函数的作用是返回一个使用 BERT 模型进行自然语言处理的 BERT 实例。

具体来说，函数通过调用一个名为 `lang_bert_func_map` 的字典，其中包含两个键：`ZH` 和 `EN`，分别对应中文和英文 BERT 模型的接口函数。此外，函数还引入了一个 `device` 参数，它指定了要使用的设备（如 CPU 或 GPU）。

在使用这些接口函数之后，函数返回 bert 实例，即 BERT 模型的实例，可以用于进行自然语言处理任务。


```py
def get_bert(norm_text, word2ph, language, device):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    return bert

```