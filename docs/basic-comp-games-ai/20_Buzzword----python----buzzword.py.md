# `20_Buzzword\python\buzzword.py`

```
# 这是一个 Buzzword 生成器的程序，用于生成教育技术演讲和简报的流行词汇
# 该程序提供三个高度可接受的词组，用于植入您的材料中
# 您的听众永远不会知道这些短语实际上并没有太多意义，因为它们听起来很棒！
# Buzzword 的完整运行说明在程序中给出
# 这个版本的 Buzzword 是由 David Ahl 编写的
# Python 版本由 Jeff Jetton 在 2019 年进行了移植
import random  # 导入 random 模块，用于生成随机数


def main() -> None:  # 定义一个名为 main 的函数，返回类型为 None
    words = [  # 创建一个包含多个列表的列表
        [
            "Ability",  # 单词1
            "Basal",  # 单词2
            "Behavioral",  # 单词3
            "Child-centered",  # 单词4
            "Differentiated",  # 单词5
            "Discovery",  # 单词6
            "Flexible",  # 单词7
            "Heterogeneous",  # 单词8
            "Homogenous",  # 单词9
            "Manipulative",  # 单词10
            "Modular",  # 单词11
            "Tavistock",  # 单词12
            "Individualized",  # 单词13
        ],
# 创建一个包含学习方法的列表
learning_methods = [
    "evaluative",  # 评估性的学习方法
    "objective",   # 目标导向的学习方法
    "cognitive",   # 认知性的学习方法
    "enrichment",  # 丰富性的学习方法
    "scheduling",  # 安排性的学习方法
    "humanistic",  # 人文主义的学习方法
    "integrated",  # 整合性的学习方法
    "non-graded",  # 非等级的学习方法
    "training",    # 训练性的学习方法
    "vertical age",# 垂直年龄的学习方法
    "motivational",# 激励性的学习方法
    "creative",    # 创造性的学习方法
]

# 创建一个包含教学方法的列表
teaching_methods = [
    "grouping",       # 分组教学方法
    "modification",   # 修改教学方法
    "accountability", # 责任制教学方法
    "process",        # 过程性教学方法
]
# 创建一个包含教育术语的列表
buzzwords = [
    [
        "collaborative",
        "holistic",
        "innovative",
        "interactive",
        "strategic",
        "synergistic",
        "critical thinking",
        "problem-solving",
        "project-based",
        "experiential",
    ],
    [
        "core curriculum",
        "algorithm",
        "performance",
        "reinforcement",
        "open classroom",
        "resource",
        "structure",
        "facility",
        "environment",
    ],
]

# 显示介绍文本
print("\n           Buzzword Generator")
print("Creative Computing  Morristown, New Jersey")
print("\n\n")
print("This program prints highly acceptable phrases in")
print("'educator-speak' that you can work into reports")
print("and speeches.  Whenever a question mark is printed,")
print("type a 'Y' for another phrase or 'N' to quit.")
    print("\n\nHere's the first phrase:")  # 打印提示信息

    still_running = True  # 初始化一个布尔变量，用于控制循环是否继续执行
    while still_running:  # 进入循环，条件为 still_running 为 True
        phrase = ""  # 初始化一个空字符串，用于存储生成的短语
        for section in words:  # 遍历列表 words 中的每个元素
            if len(phrase) > 0:  # 如果短语不为空
                phrase += " "  # 在短语末尾添加一个空格
            phrase += section[random.randint(0, len(section) - 1)]  # 在当前 section 中随机选择一个单词添加到短语中

        print(phrase)  # 打印生成的短语
        print()  # 打印空行

        response = input("? ")  # 获取用户输入的回答
        try:  # 尝试执行以下代码块
            if response.upper()[0] != "Y":  # 如果用户输入的回答的大写形式的第一个字符不是 "Y"
                still_running = False  # 将 still_running 设置为 False，结束循环
        except Exception:  # 捕获可能发生的异常
            still_running = False  # 将 still_running 设置为 False，结束循环
    print("Come back when you need help with another report!\n")
```
这行代码用于打印一条消息，提示用户何时需要帮助时可以再次返回。

```python
if __name__ == "__main__":
    main()
```
这行代码用于检查当前模块是否是主程序，如果是，则调用main函数。

```python
# Porting Notes
#
#   The original program stored all 39 words in one array, then
#   built the buzzword phrases by randomly sampling from each of the
#   three regions of the array (1-13, 14-26, and 27-39).
#
#   Here, we're storing the words for each section in separate
#   tuples.  That makes it easy to just loop through the sections
#   to stitch the phrase together, and it easily accomodates adding
#   (or removing) elements from any section.  They don't all need to
#   be the same length.
#
```
这段注释是关于代码移植的说明。原始程序将所有39个单词存储在一个数组中，然后通过从数组的三个区域（1-13，14-26和27-39）中随机抽样来构建流行语短语。在这里，我们将每个部分的单词存储在单独的元组中。这样做可以轻松地循环遍历各个部分，将短语拼接在一起，并且可以轻松地添加（或删除）任何部分的元素。它们不需要全部具有相同的长度。
# 该程序的作者（也是 Creative Computing 杂志的创始人）最初在 DEC（Digital Equipment Corporation）担任顾问，帮助公司将其计算机作为教育产品进行市场推广。后来，他担任了一份名为“EDU”的 DEC 通讯的编辑，该通讯专注于在教育环境中使用计算机。因此，这个程序中的流行词是针对教育工作者的！

# 修改想法
#   尝试添加更多/不同的单词。最好是向我们的 WORDS 元组添加第三个维度，以添加可能与不同领域相关的新单词集。商业流行词会是什么？工程流行词？艺术/音乐流行词？让用户选择一个领域，然后相应地选择流行词。
```