# `d:/src/tocomm/basic-computer-games\44_Hangman\python\hangman.py`

```
#!/usr/bin/env python3  # 指定脚本的解释器为 Python 3

"""
HANGMAN

Converted from BASIC to Python by Trevor Hobson and Daniel Piron
"""

import random  # 导入 random 模块
from typing import List  # 从 typing 模块导入 List 类型

class Canvas:  # 定义 Canvas 类，用于绘制基于文本的图形
    """For drawing text-based figures"""

    def __init__(self, width: int = 12, height: int = 12, fill: str = " ") -> None:  # 初始化方法，设置默认宽度、高度和填充字符
        self._buffer = []  # 创建一个空列表，用于存储绘制的图形
        for _ in range(height):  # 循环遍历高度
            line = []  # 创建一个空列表，用于存储每一行的字符
            for _ in range(width):  # 循环遍历宽度
                line.append("")  # 在每一行末尾添加一个空字符
            self._buffer.append(line)  # 将处理过的行添加到缓冲区

        self.clear()  # 调用clear方法清空缓冲区

    def clear(self, fill: str = " ") -> None:
        for row in self._buffer:  # 遍历缓冲区中的每一行
            for x in range(len(row)):  # 遍历每一行中的每个字符
                row[x] = fill  # 用指定的字符填充每个位置

    def render(self) -> str:
        lines = []
        for line in self._buffer:  # 遍历缓冲区中的每一行
            # Joining by the empty string ("") smooshes all of the
            # individual characters together as one line.
            lines.append("".join(line))  # 将每一行的字符连接成一个字符串并添加到lines列表中
        return "\n".join(lines)  # 将lines列表中的字符串用换行符连接成一个大字符串并返回

    def put(self, s: str, x: int, y: int) -> None:
        # In an effort to avoid distorting the drawn image, only write the
```
以上是对给定代码的每个语句添加注释，解释其作用。
# first character of the given string to the buffer.
# 将给定字符串的第一个字符放入缓冲区。
self._buffer[y][x] = s[0]

def init_gallows(canvas: Canvas) -> None:
    # Draw the vertical lines of the gallows
    # 绘制绞刑架的垂直线
    for i in range(12):
        canvas.put("X", 0, i)
    # Draw the horizontal lines of the gallows
    # 绘制绞刑架的水平线
    for i in range(7):
        canvas.put("X", i, 0)
    # Draw the top horizontal line of the gallows
    # 绘制绞刑架的顶部水平线
    canvas.put("X", 6, 1)

def draw_head(canvas: Canvas) -> None:
    # Draw the head of the hangman
    # 绘制绞刑架上的头部
    canvas.put("-", 5, 2)
    canvas.put("-", 6, 2)
    canvas.put("-", 7, 2)
    canvas.put("(", 4, 3)
    canvas.put(".", 5, 3)
    canvas.put(".", 7, 3)
    canvas.put(")", 8, 3)
    canvas.put("-", 5, 4)  # 在画布上的坐标(5, 4)处放置一个横线
    canvas.put("-", 6, 4)  # 在画布上的坐标(6, 4)处放置一个横线
    canvas.put("-", 7, 4)  # 在画布上的坐标(7, 4)处放置一个横线


def draw_body(canvas: Canvas) -> None:
    for i in range(5, 9, 1):  # 循环遍历5到8的数字
        canvas.put("X", 6, i)  # 在画布上的坐标(6, i)处放置一个X


def draw_right_arm(canvas: Canvas) -> None:
    for i in range(3, 7):  # 循环遍历3到6的数字
        canvas.put("\\", i - 1, i)  # 在画布上的坐标(i-1, i)处放置一个反斜杠


def draw_left_arm(canvas: Canvas) -> None:
    canvas.put("/", 10, 3)  # 在画布上的坐标(10, 3)处放置一个斜杠
    canvas.put("/", 9, 4)   # 在画布上的坐标(9, 4)处放置一个斜杠
    canvas.put("/", 8, 5)   # 在画布上的坐标(8, 5)处放置一个斜杠
    canvas.put("/", 7, 6)   # 在画布上的坐标(7, 6)处放置一个斜杠
# 定义一个函数，用于在画布上绘制右腿
def draw_right_leg(canvas: Canvas) -> None:
    # 在画布上放置斜杠，表示右腿
    canvas.put("/", 5, 9)
    # 在画布上放置斜杠，表示右脚
    canvas.put("/", 4, 10)


# 定义一个函数，用于在画布上绘制左腿
def draw_left_leg(canvas: Canvas) -> None:
    # 在画布上放置反斜杠，表示左腿
    canvas.put("\\", 7, 9)
    # 在画布上放置反斜杠，表示左脚
    canvas.put("\\", 8, 10)


# 定义一个函数，用于在画布上绘制左手
def draw_left_hand(canvas: Canvas) -> None:
    # 在画布上放置反斜杠，表示左手
    canvas.put("\\", 10, 2)


# 定义一个函数，用于在画布上绘制右手
def draw_right_hand(canvas: Canvas) -> None:
    # 在画布上放置斜杠，表示右手
    canvas.put("/", 2, 2)
def draw_left_foot(canvas: Canvas) -> None:
    # 在画布上放置左脚的斜杠
    canvas.put("\\", 9, 11)
    # 在画布上放置左脚的横线
    canvas.put("-", 10, 11)


def draw_right_foot(canvas: Canvas) -> None:
    # 在画布上放置右脚的横线
    canvas.put("-", 2, 11)
    # 在画布上放置右脚的斜杠
    canvas.put("/", 3, 11)


PHASES = (
    ("First, we draw a head", draw_head),  # 第一阶段，绘制头部
    ("Now we draw a body.", draw_body),  # 第二阶段，绘制身体
    ("Next we draw an arm.", draw_right_arm),  # 第三阶段，绘制右臂
    ("this time it's the other arm.", draw_left_arm),  # 第四阶段，绘制左臂
    ("Now, let's draw the right leg.", draw_right_leg),  # 第五阶段，绘制右腿
    ("This time we draw the left leg.", draw_left_leg),  # 第六阶段，绘制左腿
    ("Now we put up a hand.", draw_left_hand),  # 第七阶段，举起左手
    ("Next the other hand.", draw_right_hand),  # 第八阶段，举起右手
    ("Now we draw one foot", draw_left_foot),  # 第九阶段，绘制左脚
# 创建一个包含字符串和对应函数的元组列表
phrases = (
    ("Here's the other foot -- you're hung!!", draw_right_foot),
)

# 创建一个包含单词的列表
words = [
    "GUM",
    "SIN",
    "FOR",
    "CRY",
    "LUG",
    "BYE",
    "FLY",
    "UGLY",
    "EACH",
    "FROM",
    "WORK",
    "TALK",
    "WITH",
    "SELF",
    "PIZZA",
]
# 创建一个包含字符串的列表
words = [
    "THING",
    "FEIGN",
    "FIEND",
    "ELBOW",
    "FAULT",
    "DIRTY",
    "BUDGET",
    "SPIRIT",
    "QUAINT",
    "MAIDEN",
    "ESCORT",
    "PICKAX",
    "EXAMPLE",
    "TENSION",
    "QUININE",
    "KIDNEY",
    "REPLICA",
    "SLEEPER",
    "TRIANGLE",
    "KANGAROO",
]
    "MAHOGANY",  # 定义字符串常量"MAHOGANY"
    "SERGEANT",  # 定义字符串常量"SERGEANT"
    "SEQUENCE",  # 定义字符串常量"SEQUENCE"
    "MOUSTACHE",  # 定义字符串常量"MOUSTACHE"
    "DANGEROUS",  # 定义字符串常量"DANGEROUS"
    "SCIENTIST",  # 定义字符串常量"SCIENTIST"
    "DIFFERENT",  # 定义字符串常量"DIFFERENT"
    "QUIESCENT",  # 定义字符串常量"QUIESCENT"
    "MAGISTRATE",  # 定义字符串常量"MAGISTRATE"
    "ERRONEOUSLY",  # 定义字符串常量"ERRONEOUSLY"
    "LOUDSPEAKER",  # 定义字符串常量"LOUDSPEAKER"
    "PHYTOTOXIC",  # 定义字符串常量"PHYTOTOXIC"
    "MATRIMONIAL",  # 定义字符串常量"MATRIMONIAL"
    "PARASYMPATHOMIMETIC",  # 定义字符串常量"PARASYMPATHOMIMETIC"
    "THIGMOTROPISM",  # 定义字符串常量"THIGMOTROPISM"
]


def play_game(guess_target: str) -> None:
    """Play one round of the game"""
    # 定义一个名为play_game的函数，参数为guess_target，返回类型为None
    wrong_guesses = 0  # 初始化错误猜测次数为0

    guess_progress = ["-"] * len(guess_target)  # 创建一个包含与猜测目标相同长度的横线列表，用于表示猜测进度

    guess_list: List[str] = []  # 创建一个空列表，用于存储用户猜测的字母

    gallows = Canvas()  # 创建一个画布对象，用于绘制“绞刑架”图案
    init_gallows(gallows)  # 初始化“绞刑架”图案

    guess_count = 0  # 初始化猜测次数为0
    while True:  # 进入无限循环，直到游戏结束
        print("Here are the letters you used:")  # 打印已经猜过的字母列表
        print(",".join(guess_list) + "\n")

        print("".join(guess_progress) + "\n")  # 打印当前猜测进度

        guess_letter = ""  # 初始化猜测的字母为空字符串
        guess_word = ""  # 初始化猜测的单词为空字符串
        while guess_letter == "":  # 进入循环，直到用户输入有效的猜测字母

            guess_letter = input("What is your guess? ").upper()[0]  # 获取用户输入的猜测字母并转换为大写
            if not guess_letter.isalpha():  # 如果用户输入的不是字母
                guess_letter = ""  # 将猜测字母重置为空字符串
                print("Only letters are allowed!")  # 提示用户只能输入字母
            # 如果猜测的字母已经在猜测列表中，则重置猜测字母并打印提示信息
            elif guess_letter in guess_list:
                guess_letter = ""
                print("You guessed that letter before!")

        # 将猜测的字母添加到猜测列表中
        guess_list.append(guess_letter)
        # 猜测次数加一
        guess_count += 1
        # 如果猜测的字母在目标单词中
        if guess_letter in guess_target:
            # 找到目标单词中猜测字母的索引
            indices = [
                i for i, letter in enumerate(guess_target) if letter == guess_letter
            ]
            # 将猜测字母添加到猜测进度中的相应位置
            for i in indices:
                guess_progress[i] = guess_letter
            # 如果猜测进度与目标单词相同，则打印提示信息并结束游戏
            if "".join(guess_progress) == guess_target:
                print("You found the word!")
                break
            # 否则打印猜测进度并要求玩家猜测整个单词
            else:
                print("\n" + "".join(guess_progress) + "\n")
                while guess_word == "":
                    guess_word = input("What is your guess for the word? ").upper()
                    # 如果猜测的单词包含非字母字符，则继续要求玩家输入
                    if not guess_word.isalpha():
                        guess_word = ""  # 初始化猜测的单词为空字符串
                        print("Only words are allowed!")  # 打印提示信息，只允许输入单词
                if guess_word == guess_target:  # 如果猜测的单词与目标单词相同
                    print("Right!! It took you", guess_count, "guesses!")  # 打印猜对的提示信息和猜测次数
                    break  # 结束循环
        else:  # 如果上述条件不满足
            comment, draw_bodypart = PHASES[wrong_guesses]  # 从PHASES列表中获取错误猜测次数对应的提示和绘制人体部位的函数

            print(comment)  # 打印提示信息
            draw_bodypart(gallows)  # 调用绘制人体部位的函数
            print(gallows.render())  # 打印绘制的人体部位图案

            wrong_guesses += 1  # 错误猜测次数加1
            print("Sorry, that letter isn't in the word.")  # 打印提示信息，猜测的字母不在单词中

            if wrong_guesses == 10:  # 如果错误猜测次数达到10次
                print("Sorry, you lose. The word was " + guess_target)  # 打印游戏失败的提示信息和目标单词
                break  # 结束循环
def main() -> None:
    # 打印游戏标题
    print(" " * 32 + "HANGMAN")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")

    # 随机打乱单词列表中的单词顺序
    random.shuffle(words)
    # 初始化当前单词的索引和单词总数
    current_word = 0
    word_count = len(words)

    # 初始化游戏继续标志
    keep_playing = True
    # 当游戏继续标志为真时，循环进行游戏
    while keep_playing:
        # 进行游戏，传入当前单词
        play_game(words[current_word])
        # 更新当前单词索引
        current_word += 1

        # 如果已经玩过所有单词，则打印提示信息并将游戏继续标志设为假
        if current_word == word_count:
            print("You did all the words!!")
            keep_playing = False
        else:
            # 否则询问玩家是否想要继续玩下一个单词，根据输入判断是否继续
            keep_playing = (
                input("Want another word? (yes or no) ").lower().startswith("y")
    )

    print("It's been fun! Bye for now.")


if __name__ == "__main__":
    main()
```

这部分代码是程序的结尾部分，包括了程序的入口点和程序的结束。`if __name__ == "__main__":` 表示如果当前脚本被直接执行，而不是被导入到其他模块中，那么执行 `main()` 函数。`main()` 函数是程序的入口点，包含了整个程序的执行逻辑。最后的 `print("It's been fun! Bye for now.")` 语句用于输出一条结束语。
```