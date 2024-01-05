# `d:/src/tocomm/Bert-VITS2\oldVersion\V110\text\tone_sandhi.py`

```
# 引入必要的模块和类型声明
from typing import List  # 导入List类型声明
from typing import Tuple  # 导入Tuple类型声明

import jieba  # 导入jieba模块，用于中文分词
from pypinyin import lazy_pinyin  # 导入lazy_pinyin函数，用于获取拼音
from pypinyin import Style  # 导入Style枚举类，用于指定拼音风格
# 定义一个名为ToneSandhi的类
class ToneSandhi:
    # 初始化方法，用于创建ToneSandhi对象
    def __init__(self):
        # 创建一个名为must_neural_tone_words的属性，值为一个包含多个字符串的集合
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
```

这段代码定义了一个名为ToneSandhi的类，其中包含一个初始化方法`__init__`和一个属性`must_neural_tone_words`。`must_neural_tone_words`是一个集合，包含了多个字符串元素。这些字符串元素代表了一些特定的词语。
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
            "锄头",  # 表示一个字符串，可能是一个词汇或者一个物品的名称
            "铺盖",  # 表示一个字符串，可能是一个词汇或者一个物品的名称
            "铃铛",  # 表示一个字符串，可能是一个词汇或者一个物品的名称
            "铁匠",  # 表示一个字符串，可能是一个词汇或者一个人的职业
            "钥匙",  # 表示一个字符串，可能是一个词汇或者一个物品的名称
            "里脊",  # 表示一个字符串，可能是一个词汇或者一个食物的名称
            "里头",  # 表示一个字符串，可能是一个词汇或者一个地方的名称
            "部分",  # 表示一个字符串，可能是一个词汇或者一个物品的名称
            "那么",  # 表示一个字符串，可能是一个词汇或者一个词组的一部分
            "道士",  # 表示一个字符串，可能是一个词汇或者一个人的职业
            "造化",  # 表示一个字符串，可能是一个词汇或者一个概念的名称
            "迷糊",  # 表示一个字符串，可能是一个词汇或者一个人的状态
            "连累",  # 表示一个字符串，可能是一个词汇或者一个动作的结果
            "这么",  # 表示一个字符串，可能是一个词汇或者一个词组的一部分
            "这个",  # 表示一个字符串，可能是一个词汇或者一个物品的名称
            "运气",  # 表示一个字符串，可能是一个词汇或者一个概念的名称
            "过去",  # 表示一个字符串，可能是一个词汇或者一个时间的概念
            "软和",  # 表示一个字符串，可能是一个词汇或者一个物品的状态
            "转悠",  # 表示一个字符串，可能是一个词汇或者一个动作的结果
            "踏实",  # 表示一个字符串，可能是一个词汇或者一个人的状态
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
            "跳蚤",  # 跳蚤的意思是指一种小型寄生虫，这里可能是一个字符串列表的元素
            "跟头",  # 跟头的意思是指摔倒或者翻跟斗，这里可能是一个字符串列表的元素
            "趔趄",  # 趔趄的意思是指行走不稳定或者蹒跚，这里可能是一个字符串列表的元素
            "财主",  # 财主的意思是指富有的人，这里可能是一个字符串列表的元素
            "豆腐",  # 豆腐的意思是指一种食物，这里可能是一个字符串列表的元素
            "讲究",  # 讲究的意思是指注重细节或者追求精致，这里可能是一个字符串列表的元素
            "记性",  # 记性的意思是指记忆力，这里可能是一个字符串列表的元素
            "记号",  # 记号的意思是指标记或者符号，这里可能是一个字符串列表的元素
            "认识",  # 认识的意思是指了解或者熟悉，这里可能是一个字符串列表的元素
            "规矩",  # 规矩的意思是指行为准则或者规则，这里可能是一个字符串列表的元素
            "见识",  # 见识的意思是指经历或者见闻，这里可能是一个字符串列表的元素
            "裁缝",  # 裁缝的意思是指制衣师傅，这里可能是一个字符串列表的元素
            "补丁",  # 补丁的意思是指修补或者补充，这里可能是一个字符串列表的元素
            "衣裳",  # 衣裳的意思是指衣服，这里可能是一个字符串列表的元素
            "衣服",  # 衣服的意思是指穿在身上的衣物，这里可能是一个字符串列表的元素
            "衙门",  # 衙门的意思是指官府或者衙役，这里可能是一个字符串列表的元素
            "街坊",  # 街坊的意思是指邻居或者街坊之间的关系，这里可能是一个字符串列表的元素
            "行李",  # 行李的意思是指旅行时携带的物品，这里可能是一个字符串列表的元素
            "行当",  # 行当的意思是指职业或者行业，这里可能是一个字符串列表的元素
            "蛤蟆",  # 蛤蟆的意思是指一种两栖动物，这里可能是一个字符串列表的元素
# 创建一个列表，包含了一些字符串元素
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
            "胭脂",  # 表示一个字符串，可能是一个词语或者一个名字
            "胡萝",  # 表示一个字符串，可能是一个词语或者一个名字
            "胡琴",  # 表示一个字符串，可能是一个词语或者一个名字
            "胡同",  # 表示一个字符串，可能是一个词语或者一个名字
            "聪明",  # 表示一个字符串，可能是一个词语或者一个名字
            "耽误",  # 表示一个字符串，可能是一个词语或者一个名字
            "耽搁",  # 表示一个字符串，可能是一个词语或者一个名字
            "耷拉",  # 表示一个字符串，可能是一个词语或者一个名字
            "耳朵",  # 表示一个字符串，可能是一个词语或者一个名字
            "老爷",  # 表示一个字符串，可能是一个词语或者一个名字
            "老实",  # 表示一个字符串，可能是一个词语或者一个名字
            "老婆",  # 表示一个字符串，可能是一个词语或者一个名字
            "老头",  # 表示一个字符串，可能是一个词语或者一个名字
            "老太",  # 表示一个字符串，可能是一个词语或者一个名字
            "翻腾",  # 表示一个字符串，可能是一个词语或者一个名字
            "罗嗦",  # 表示一个字符串，可能是一个词语或者一个名字
            "罐头",  # 表示一个字符串，可能是一个词语或者一个名字
            "编辑",  # 表示一个字符串，可能是一个词语或者一个名字
            "结实",  # 表示一个字符串，可能是一个词语或者一个名字
            "红火",  # 表示一个字符串，可能是一个词语或者一个名字
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
            "累赘",  # 表示某个词语或事物是累赘的，即负担或麻烦
            "糨糊",  # 指一种粘稠的食物或液体
            "糊涂",  # 形容思维混乱或不清楚
            "精神",  # 指精力充沛、思维敏捷的状态
            "粮食",  # 指用于人类或动物食用的谷物、豆类等食物
            "簸箕",  # 一种用来扬去杂物的工具
            "篱笆",  # 用竹子、木材等制成的围墙或隔离物
            "算计",  # 指计算、估算或策划
            "算盘",  # 一种用来计算的工具
            "答应",  # 表示同意或应允
            "笤帚",  # 用来扫地的工具
            "笑语",  # 指带有笑声的语言或言辞
            "笑话",  # 指用来引起笑声的故事或言辞
            "窟窿",  # 指洞穴或孔洞
            "窝囊",  # 形容软弱无能或无精打采
            "窗户",  # 用来通风和采光的开口
            "稳当",  # 形容稳定、可靠或牢固
            "稀罕",  # 表示珍惜或重视
            "称呼",  # 指对人或事物的称谓或称呼方式
            "秧歌",  # 一种民间舞蹈形式
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
            "秀气",  # 表示一个词语 "秀气"
            "秀才",  # 表示一个词语 "秀才"
            "福气",  # 表示一个词语 "福气"
            "祖宗",  # 表示一个词语 "祖宗"
            "砚台",  # 表示一个词语 "砚台"
            "码头",  # 表示一个词语 "码头"
            "石榴",  # 表示一个词语 "石榴"
            "石头",  # 表示一个词语 "石头"
            "石匠",  # 表示一个词语 "石匠"
            "知识",  # 表示一个词语 "知识"
            "眼睛",  # 表示一个词语 "眼睛"
            "眯缝",  # 表示一个词语 "眯缝"
            "眨巴",  # 表示一个词语 "眨巴"
            "眉毛",  # 表示一个词语 "眉毛"
            "相声",  # 表示一个词语 "相声"
            "盘算",  # 表示一个词语 "盘算"
            "白净",  # 表示一个词语 "白净"
            "痢疾",  # 表示一个词语 "痢疾"
            "痛快",  # 表示一个词语 "痛快"
            "疟疾",  # 表示一个词语 "疟疾"
```

这段代码是一个列表，包含了一系列的字符串，每个字符串表示一个词语。这些字符串将被用于后续的操作。
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
            "疙瘩",  # 表示一个词语或短语，具体含义不明确
            "疏忽",  # 表示一个词语或短语，具体含义不明确
            "畜生",  # 表示一个词语或短语，具体含义不明确
            "生意",  # 表示一个词语或短语，具体含义不明确
            "甘蔗",  # 表示一个词语或短语，具体含义不明确
            "琵琶",  # 表示一个词语或短语，具体含义不明确
            "琢磨",  # 表示一个词语或短语，具体含义不明确
            "琉璃",  # 表示一个词语或短语，具体含义不明确
            "玻璃",  # 表示一个词语或短语，具体含义不明确
            "玫瑰",  # 表示一个词语或短语，具体含义不明确
            "玄乎",  # 表示一个词语或短语，具体含义不明确
            "狐狸",  # 表示一个词语或短语，具体含义不明确
            "状元",  # 表示一个词语或短语，具体含义不明确
            "特务",  # 表示一个词语或短语，具体含义不明确
            "牲口",  # 表示一个词语或短语，具体含义不明确
            "牙碜",  # 表示一个词语或短语，具体含义不明确
            "牌楼",  # 表示一个词语或短语，具体含义不明确
            "爽快",  # 表示一个词语或短语，具体含义不明确
            "爱人",  # 表示一个词语或短语，具体含义不明确
            "热闹",  # 表示一个词语或短语，具体含义不明确
# 创建一个包含字符串的列表
words = [
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
            "故事",  # 表示一个字符串，可能是一个文件名或者数据
            "收拾",  # 表示一个字符串，可能是一个文件名或者数据
            "收成",  # 表示一个字符串，可能是一个文件名或者数据
            "提防",  # 表示一个字符串，可能是一个文件名或者数据
            "挖苦",  # 表示一个字符串，可能是一个文件名或者数据
            "挑剔",  # 表示一个字符串，可能是一个文件名或者数据
            "指甲",  # 表示一个字符串，可能是一个文件名或者数据
            "指头",  # 表示一个字符串，可能是一个文件名或者数据
            "拾掇",  # 表示一个字符串，可能是一个文件名或者数据
            "拳头",  # 表示一个字符串，可能是一个文件名或者数据
            "拨弄",  # 表示一个字符串，可能是一个文件名或者数据
            "招牌",  # 表示一个字符串，可能是一个文件名或者数据
            "招呼",  # 表示一个字符串，可能是一个文件名或者数据
            "抬举",  # 表示一个字符串，可能是一个文件名或者数据
            "护士",  # 表示一个字符串，可能是一个文件名或者数据
            "折腾",  # 表示一个字符串，可能是一个文件名或者数据
            "扫帚",  # 表示一个字符串，可能是一个文件名或者数据
            "打量",  # 表示一个字符串，可能是一个文件名或者数据
            "打算",  # 表示一个字符串，可能是一个文件名或者数据
            "打点",  # 表示一个字符串，可能是一个文件名或者数据
```

这些代码表示一系列字符串，可能是文件名或者数据的标识符。
# 创建一个函数，用于读取 ZIP 文件的内容，并返回一个字典，其中包含文件名和对应的数据。
def read_zip(fname):
    # 使用 BytesIO 类创建一个字节流对象，将 ZIP 文件的内容封装进去。
    bio = BytesIO(open(fname, 'rb').read())
    # 使用 zipfile 模块的 ZipFile 类创建一个 ZIP 对象，用于操作 ZIP 文件。
    zip = zipfile.ZipFile(bio, 'r')
    # 使用 zip.namelist() 方法获取 ZIP 文件中所有文件的文件名，并遍历这些文件名。
    # 使用 zip.read(n) 方法读取每个文件的数据，并将文件名和数据组成一个字典。
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象，释放资源。
    zip.close()
    # 返回包含文件名和数据的字典。
    return fdict
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
            "得罪",  # 表示冒犯或得罪
            "张罗",  # 表示安排或筹划
            "弟兄",  # 表示兄弟
            "开通",  # 表示开辟或通畅
            "应酬",  # 表示应酬或应付
            "庄稼",  # 表示农作物或农田
            "干事",  # 表示从事工作或担任职务
            "帮手",  # 表示帮助或助手
            "帐篷",  # 表示帐篷或蓬子
            "希罕",  # 表示稀奇或珍贵
            "师父",  # 表示师傅或老师
            "师傅",  # 表示师父或导师
            "巴结",  # 表示讨好或奉承
            "巴掌",  # 表示用手掌打
            "差事",  # 表示任务或工作
            "工夫",  # 表示时间或功夫
            "岁数",  # 表示年龄或岁数
            "屁股",  # 表示臀部或股沟
            "尾巴",  # 表示动物的尾巴
            "少爷",  # 表示年轻的男子或贵族的儿子
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
            "小气",  # 表示小气的意思
            "小伙",  # 表示年轻人的意思
            "将就",  # 表示勉强接受的意思
            "对头",  # 表示对手的意思
            "对付",  # 表示应付的意思
            "寡妇",  # 表示丈夫去世的女人的意思
            "家伙",  # 表示人的意思
            "客气",  # 表示礼貌的意思
            "实在",  # 表示真实的意思
            "官司",  # 表示法律纠纷的意思
            "学问",  # 表示知识的意思
            "学生",  # 表示在校学习的人的意思
            "字号",  # 表示字体大小的意思
            "嫁妆",  # 表示女子出嫁时带入婚姻的财产的意思
            "媳妇",  # 表示儿子的妻子的意思
            "媒人",  # 表示介绍婚姻的人的意思
            "婆家",  # 表示丈夫家的意思
            "娘家",  # 表示女儿家的意思
            "委屈",  # 表示受到不公平待遇的意思
            "姑娘",  # 表示年轻女子的意思
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
            "姐夫",  # 姐夫
            "妯娌",  # 妯娌
            "妥当",  # 妥当
            "妖精",  # 妖精
            "奴才",  # 奴才
            "女婿",  # 女婿
            "头发",  # 头发
            "太阳",  # 太阳
            "大爷",  # 大爷
            "大方",  # 大方
            "大意",  # 大意
            "大夫",  # 大夫
            "多少",  # 多少
            "多么",  # 多么
            "外甥",  # 外甥
            "壮实",  # 壮实
            "地道",  # 地道
            "地方",  # 地方
            "在乎",  # 在乎
            "困难",  # 困难
```

这段代码是一个函数 `read_zip`，用于读取 ZIP 文件中的内容并返回一个字典，其中键是文件名，值是文件的数据。

在函数中，首先根据给定的 ZIP 文件名，将其二进制内容封装成字节流对象 `bio`。然后使用字节流对象创建 ZIP 对象 `zip`，用于操作 ZIP 文件。接下来，通过遍历 ZIP 对象中包含的文件名，使用 `zip.read(n)` 读取每个文件的数据，并将文件名和数据组成字典 `fdict`。最后，关闭 ZIP 对象，并返回结果字典。

需要注释的代码是一个字符串列表，包含了一些中文词语。这些词语可能是作为文件名在 ZIP 文件中存在的文件。
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
            "嘴巴",  # 表示一个字符串，可能是一个文件名或者数据
            "嘱咐",  # 表示一个字符串，可能是一个文件名或者数据
            "嘟囔",  # 表示一个字符串，可能是一个文件名或者数据
            "嘀咕",  # 表示一个字符串，可能是一个文件名或者数据
            "喜欢",  # 表示一个字符串，可能是一个文件名或者数据
            "喇嘛",  # 表示一个字符串，可能是一个文件名或者数据
            "喇叭",  # 表示一个字符串，可能是一个文件名或者数据
            "商量",  # 表示一个字符串，可能是一个文件名或者数据
            "唾沫",  # 表示一个字符串，可能是一个文件名或者数据
            "哑巴",  # 表示一个字符串，可能是一个文件名或者数据
            "哈欠",  # 表示一个字符串，可能是一个文件名或者数据
            "哆嗦",  # 表示一个字符串，可能是一个文件名或者数据
            "咳嗽",  # 表示一个字符串，可能是一个文件名或者数据
            "和尚",  # 表示一个字符串，可能是一个文件名或者数据
            "告诉",  # 表示一个字符串，可能是一个文件名或者数据
            "告示",  # 表示一个字符串，可能是一个文件名或者数据
            "含糊",  # 表示一个字符串，可能是一个文件名或者数据
            "吓唬",  # 表示一个字符串，可能是一个文件名或者数据
            "后头",  # 表示一个字符串，可能是一个文件名或者数据
            "名字",  # 表示一个字符串，可能是一个文件名或者数据
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
            "名堂",  # 名堂
            "合同",  # 合同
            "吆喝",  # 叫喊
            "叫唤",  # 叫喊
            "口袋",  # 裤袋
            "厚道",  # 诚实宽厚
            "厉害",  # 强大
            "千斤",  # 重量单位
            "包袱",  # 负担
            "包涵",  # 包容
            "匀称",  # 均匀
            "勤快",  # 努力
            "动静",  # 活动和安静
            "动弹",  # 移动
            "功夫",  # 功夫
            "力气",  # 力量
            "前头",  # 前面
            "刺猬",  # 动物名
            "刺激",  # 激励
            "别扭",  # 不舒服
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

```python
            "利落",  # 表示利落的意思
            "利索",  # 表示利索的意思
            "利害",  # 表示利害的意思
            "分析",  # 表示分析的意思
            "出息",  # 表示出息的意思
            "凑合",  # 表示凑合的意思
            "凉快",  # 表示凉快的意思
            "冷战",  # 表示冷战的意思
            "冤枉",  # 表示冤枉的意思
            "冒失",  # 表示冒失的意思
            "养活",  # 表示养活的意思
            "关系",  # 表示关系的意思
            "先生",  # 表示先生的意思
            "兄弟",  # 表示兄弟的意思
            "便宜",  # 表示便宜的意思
            "使唤",  # 表示使唤的意思
            "佩服",  # 表示佩服的意思
            "作坊",  # 表示作坊的意思
            "体面",  # 表示体面的意思
            "位置",  # 表示位置的意思
```

这些代码是一系列字符串，每个字符串表示一个词语的意思。这些注释提供了这些词语的含义。
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
            "似的",  # 表示类似的意思
            "伙计",  # 表示朋友、同伴的意思
            "休息",  # 表示休息、放松的意思
            "什么",  # 表示疑问、不确定的意思
            "人家",  # 表示别人、他人的意思
            "亲戚",  # 表示亲属、亲人的意思
            "亲家",  # 表示婆家、娘家的意思
            "交情",  # 表示友情、关系的意思
            "云彩",  # 表示天空中的云的意思
            "事情",  # 表示事情、事件的意思
            "买卖",  # 表示买和卖的意思
            "主意",  # 表示想法、意见的意思
            "丫头",  # 表示女孩子的意思
            "丧气",  # 表示情绪低落、沮丧的意思
            "两口",  # 表示夫妻的意思
            "东西",  # 表示物品、东西的意思
            "东家",  # 表示卖方、东家的意思
            "世故",  # 表示圆滑、老练的意思
            "不由",  # 表示不由自主、不由得的意思
            "不在",  # 表示不在、不存在的意思
# 创建一个包含字符串的列表
words = [
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
```

这部分代码是一个字符串列表，包含了一些词语。这些词语可能是用来测试代码的输入数据，或者是其他用途。在这个函数中，这部分代码没有被使用到，可以忽略。
# 定义一个类，用于存储一些词汇和标点符号
class Vocabulary:
    # 初始化函数，用于设置类的属性
    def __init__(self):
        # 定义一个集合，存储一些情感词汇
        self.emotion_words = {
            "开心",
            "快乐",
            "悲伤",
            "愤怒",
            "惊讶",
            "害怕",
            "失望",
            "烦恼",
        }
        # 定义一个集合，存储一些中性词汇
        self.neutral_words = {
            "苹果",
            "香蕉",
            "橙子",
            "西瓜",
            "菠萝",
            "葡萄",
            "草莓",
            "樱桃",
        }
        # 定义一个集合，存储一些负面情感词汇
        self.negative_emotion_words = {
            "伤心",
            "痛苦",
            "愤怒",
            "恐惧",
            "失望",
            "焦虑",
            "厌烦",
            "绝望",
        }
        # 定义一个集合，存储一些积极情感词汇
        self.positive_emotion_words = {
            "快乐",
            "幸福",
            "喜悦",
            "兴奋",
            "满足",
            "舒适",
            "愉快",
            "欣喜",
        }
        # 定义一个集合，存储一些表示肯定的词汇
        self.positive_words = {
            "好",
            "棒",
            "赞",
            "优秀",
            "喜欢",
            "满意",
            "支持",
            "感谢",
        }
        # 定义一个集合，存储一些表示否定的词汇
        self.negative_words = {
            "不好",
            "差",
            "糟糕",
            "失望",
            "讨厌",
            "不满意",
            "反对",
            "抱怨",
        }
        # 定义一个集合，存储一些表示中性的词汇
        self.neutral_tone_words = {
            "苹果",
            "香蕉",
            "橙子",
            "西瓜",
            "菠萝",
            "葡萄",
            "草莓",
            "樱桃",
            "牢骚",
            "咖喱",
            "扫把",
            "惦记",
        }
        # 定义一个集合，存储一些表示非中性的词汇
        self.must_not_neutral_tone_words = {
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
        # 定义一个字符串，存储一些标点符号
        self.punc = "：，；。？！“”‘’':,;.?!"
# the meaning of jieba pos tag: https://blog.csdn.net/weixin_44174352/article/details/113731041
# e.g.
# word: "家里"
# pos: "s"
# finals: ['ia1', 'i3']
def _neural_sandhi(self, word: str, pos: str, finals: List[str]) -> List[str]:
    # 对于名词、动词和形容词的重复词进行处理，例如"奶奶"、"试试"、"旺旺"
    for j, item in enumerate(word):
        if (
            j - 1 >= 0
            and item == word[j - 1]
            and pos[0] in {"n", "v", "a"}
            and word not in self.must_not_neural_tone_words
        ):
            # 将重复词的韵母最后一个音调数字替换为5
            finals[j] = finals[j][:-1] + "5"
    # 查找是否包含"个"字
    ge_idx = word.find("个")
    # 如果词的最后一个字是"吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶"中的一个
    if len(word) >= 1 and word[-1] in "吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶":
        # 将最后一个韵母的音调数字替换为5
        finals[-1] = finals[-1][:-1] + "5"
    # 如果词的最后一个字是"的地得"中的一个
    elif len(word) >= 1 and word[-1] in "的地得":
        # 将最后一个韵母的音调数字替换为5
        finals[-1] = finals[-1][:-1] + "5"
        # e.g. 走了, 看着, 去过
        # 如果单词长度为1，并且在"了着过"中，并且词性在{"ul", "uz", "ug"}中
        # 则将最后一个韵母替换为"5"
        # elif len(word) == 1 and word in "了着过" and pos in {"ul", "uz", "ug"}:
        #     finals[-1] = finals[-1][:-1] + "5"
        elif (
            len(word) > 1
            and word[-1] in "们子"
            and pos in {"r", "n"}
            and word not in self.must_not_neural_tone_words
        ):
            # 如果单词长度大于1，并且最后一个字在"们子"中，并且词性在{"r", "n"}中，并且单词不在self.must_not_neural_tone_words中
            # 则将最后一个韵母替换为"5"
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 桌上, 地下, 家里
        elif len(word) > 1 and word[-1] in "上下里" and pos in {"s", "l", "f"}:
            # 如果单词长度大于1，并且最后一个字在"上下里"中，并且词性在{"s", "l", "f"}中
            # 则将最后一个韵母替换为"5"
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 上来, 下去
        elif len(word) > 1 and word[-1] in "来去" and word[-2] in "上下进出回过起开":
            # 如果单词长度大于1，并且最后一个字在"来去"中，并且倒数第二个字在"上下进出回过起开"中
            # 则将最后一个韵母替换为"5"
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
        finals = sum(finals_list, [])  # 将拆分后的韵母列表合并为一个列表
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
        elif len(word) == 4:  # 如果词的长度为4
            finals_list = [finals[:2], finals[2:]]  # 将韵母列表分成两部分
            finals = []  # 重置韵母列表
            for sub in finals_list:  # 遍历分割后的韵母列表
                if self._all_tone_three(sub):  # 如果所有韵母都是三声
                    sub[0] = sub[0][:-1] + "2"  # 将第一个韵母的声调改为二声
                finals += sub  # 将处理后的韵母列表合并
        return finals  # 返回处理后的韵母列表

    def _all_tone_three(self, finals: List[str]) -> bool:  # 定义一个方法，用于判断所有韵母是否为三声
        return all(x[-1] == "3" for x in finals)  # 返回所有韵母是否为三声的判断结果

    # merge "不" and the word behind it
    # if don't merge, "不" sometimes appears alone according to jieba, which may occur sandhi error
    def _merge_bu(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:  # 定义一个方法，用于合并"不"和它后面的词
        new_seg = []  # 创建一个新的分词列表
        last_word = ""  # 初始化上一个词为空
        for word, pos in seg:  # 遍历分词列表中的词和词性
            if last_word == "不":  # 如果上一个词是"不"
                word = last_word + word  # 将上一个词和当前词合并
            if word != "不":  # 如果合并后的词不是"不"
                new_seg.append((word, pos))  # 将合并后的词和词性添加到新的分词结果中
            last_word = word[:]  # 更新上一个词为当前词
        if last_word == "不":  # 如果最后一个词是"不"
            new_seg.append((last_word, "d"))  # 将"不"和词性"d"添加到新的分词结果中
            last_word = ""  # 清空上一个词
        return new_seg  # 返回合并后的分词结果

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
# 检查条件：i - 1 大于等于 0
# 检查条件：word 等于 "一"
# 检查条件：i + 1 小于 seg 的长度
# 检查条件：seg[i - 1][0] 等于 seg[i + 1][0]
# 检查条件：seg[i - 1][1] 等于 "v"
if (
    i - 1 >= 0
    and word == "一"
    and i + 1 < len(seg)
    and seg[i - 1][0] == seg[i + 1][0]
    and seg[i - 1][1] == "v"
):
    # 如果满足条件，将前一个词的内容重复一次并赋值给新的分词列表
    new_seg[i - 1][0] = new_seg[i - 1][0] + "一" + new_seg[i - 1][0]
else:
    # 如果不满足条件，继续检查其他条件
    if (
        i - 2 >= 0
        and seg[i - 1][0] == "一"
        and seg[i - 2][0] == word
        and pos == "v"
    ):
        # 如果满足条件，继续下一次循环
        continue
    else:
        # 如果不满足条件，将当前词和词性添加到新的分词列表中
        new_seg.append([word, pos])
# 将新的分词列表赋值给原始分词列表
seg = new_seg
# 清空新的分词列表
new_seg = []
# 调用第二个函数
# function 2
        for i, (word, pos) in enumerate(seg):
            # 遍历分词结果中的每个词和词性
            if new_seg and new_seg[-1][0] == "一":
                # 如果新的分词结果列表不为空，并且最后一个词是“一”
                new_seg[-1][0] = new_seg[-1][0] + word
                # 将当前词与最后一个词合并
            else:
                new_seg.append([word, pos])
                # 将当前词添加到新的分词结果列表中
        return new_seg
        # 返回合并后的分词结果列表

    # the first and the second words are all_tone_three
    def _merge_continuous_three_tones(
        self, seg: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        # 定义一个新的分词结果列表
        new_seg = []
        # 对每个词进行处理，获取其带声调的韵母列表
        sub_finals_list = [
            lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        # 断言带声调的韵母列表的长度与分词结果列表的长度相同
        assert len(sub_finals_list) == len(seg)
        # 定义一个列表，用于记录是否需要合并最后一个词
        merge_last = [False] * len(seg)
        for i, (word, pos) in enumerate(seg):
            # 遍历分词结果中的每个词和词性
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