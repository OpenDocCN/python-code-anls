# `basic-computer-games\44_Hangman\python\hangman.py`

```
#!/usr/bin/env python3

"""
HANGMAN

Converted from BASIC to Python by Trevor Hobson and Daniel Piron
"""

import random
from typing import List


class Canvas:
    """For drawing text-based figures"""

    def __init__(self, width: int = 12, height: int = 12, fill: str = " ") -> None:
        # 初始化画布，创建一个二维数组作为缓冲区
        self._buffer = []
        for _ in range(height):
            line = []
            for _ in range(width):
                line.append("")
            self._buffer.append(line)

        # 清空画布，填充指定字符
        self.clear()

    def clear(self, fill: str = " ") -> None:
        # 清空画布，填充指定字符
        for row in self._buffer:
            for x in range(len(row)):
                row[x] = fill

    def render(self) -> str:
        # 渲染画布，将缓冲区内容转换为字符串
        lines = []
        for line in self._buffer:
            # Joining by the empty string ("") smooshes all of the
            # individual characters together as one line.
            lines.append("".join(line))
        return "\n".join(lines)

    def put(self, s: str, x: int, y: int) -> None:
        # 在指定位置放置字符，为了避免扭曲绘制的图像，只写入给定字符串的第一个字符到缓冲区
        self._buffer[y][x] = s[0]


def init_gallows(canvas: Canvas) -> None:
    # 初始化绞刑架
    for i in range(12):
        canvas.put("X", 0, i)
    for i in range(7):
        canvas.put("X", i, 0)
    canvas.put("X", 6, 1)


def draw_head(canvas: Canvas) -> None:
    # 绘制头部
    canvas.put("-", 5, 2)
    canvas.put("-", 6, 2)
    canvas.put("-", 7, 2)
    canvas.put("(", 4, 3)
    canvas.put(".", 5, 3)
    canvas.put(".", 7, 3)
    canvas.put(")", 8, 3)
    canvas.put("-", 5, 4)
    canvas.put("-", 6, 4)
    canvas.put("-", 7, 4)


def draw_body(canvas: Canvas) -> None:
    # 绘制身体
    for i in range(5, 9, 1):
        canvas.put("X", 6, i)


def draw_right_arm(canvas: Canvas) -> None:
    # 绘制右手臂
    for i in range(3, 7):
        canvas.put("\\", i - 1, i)


def draw_left_arm(canvas: Canvas) -> None:
    # 绘制左手臂
    canvas.put("/", 10, 3)
    canvas.put("/", 9, 4)
    canvas.put("/", 8, 5)
    canvas.put("/", 7, 6)
# 绘制右腿
def draw_right_leg(canvas: Canvas) -> None:
    # 在画布上放置斜杠
    canvas.put("/", 5, 9)
    # 在画布上放置斜杠
    canvas.put("/", 4, 10)


# 绘制左腿
def draw_left_leg(canvas: Canvas) -> None:
    # 在画布上放置反斜杠
    canvas.put("\\", 7, 9)
    # 在画布上放置反斜杠
    canvas.put("\\", 8, 10)


# 绘制左手
def draw_left_hand(canvas: Canvas) -> None:
    # 在画布上放置反斜杠
    canvas.put("\\", 10, 2)


# 绘制右手
def draw_right_hand(canvas: Canvas) -> None:
    # 在画布上放置斜杠
    canvas.put("/", 2, 2)


# 绘制左脚
def draw_left_foot(canvas: Canvas) -> None:
    # 在画布上放置反斜杠
    canvas.put("\\", 9, 11)
    # 在画布上放置横线
    canvas.put("-", 10, 11)


# 绘制右脚
def draw_right_foot(canvas: Canvas) -> None:
    # 在画布上放置横线
    canvas.put("-", 2, 11)
    # 在画布上放置斜杠
    canvas.put("/", 3, 11)


# 游戏阶段
PHASES = (
    ("First, we draw a head", draw_head),
    ("Now we draw a body.", draw_body),
    ("Next we draw an arm.", draw_right_arm),
    ("this time it's the other arm.", draw_left_arm),
    ("Now, let's draw the right leg.", draw_right_leg),
    ("This time we draw the left leg.", draw_left_leg),
    ("Now we put up a hand.", draw_left_hand),
    ("Next the other hand.", draw_right_hand),
    ("Now we draw one foot", draw_left_foot),
    ("Here's the other foot -- you're hung!!", draw_right_foot),
)

# 单词列表
words = [
    "GUM",
    "SIN",
    "FOR",
    # ... 其他单词
]

# 玩游戏
def play_game(guess_target: str) -> None:
    """Play one round of the game"""
    # 错误猜测次数
    wrong_guesses = 0
    # 猜测进度
    guess_progress = ["-"] * len(guess_target)
    # 猜测列表
    guess_list: List[str] = []
    # 创建一个画布对象
    gallows = Canvas()
    # 初始化绞刑架
    init_gallows(gallows)
    
    # 猜测次数计数器
    guess_count = 0
    # 无限循环，直到猜测正确或者猜错次数达到上限
    while True:
        # 打印已经猜过的字母
        print("Here are the letters you used:")
        print(",".join(guess_list) + "\n")
        # 打印当前猜测进度
        print("".join(guess_progress) + "\n")
        # 初始化猜测的字母和单词
        guess_letter = ""
        guess_word = ""
        # 循环直到猜测的字母有效
        while guess_letter == "":
            # 获取用户输入的猜测字母并转换为大写
            guess_letter = input("What is your guess? ").upper()[0]
            # 检查猜测字母是否为字母
            if not guess_letter.isalpha():
                guess_letter = ""
                print("Only letters are allowed!")
            # 检查猜测字母是否已经猜过
            elif guess_letter in guess_list:
                guess_letter = ""
                print("You guessed that letter before!")
    
        # 将猜测的字母添加到已猜过的列表中
        guess_list.append(guess_letter)
        # 猜测次数加一
        guess_count += 1
        # 如果猜测的字母在目标单词中
        if guess_letter in guess_target:
            # 找到猜测字母在目标单词中的位置
            indices = [
                i for i, letter in enumerate(guess_target) if letter == guess_letter
            ]
            # 更新猜测进度
            for i in indices:
                guess_progress[i] = guess_letter
            # 如果猜测进度与目标单词相同，表示猜测成功
            if "".join(guess_progress) == guess_target:
                print("You found the word!")
                break
            else:
                # 打印当前猜测进度
                print("\n" + "".join(guess_progress) + "\n")
                # 循环直到猜测的单词有效
                while guess_word == "":
                    guess_word = input("What is your guess for the word? ").upper()
                    # 检查猜测的单词是否为字母
                    if not guess_word.isalpha():
                        guess_word = ""
                        print("Only words are allowed!")
                    # 如果猜测的单词与目标单词相同，表示猜测成功
                    if guess_word == guess_target:
                        print("Right!! It took you", guess_count, "guesses!")
                        break
        else:
            # 获取当前错误猜测的提示和绘制身体部位的函数
            comment, draw_bodypart = PHASES[wrong_guesses]
    
            # 打印错误猜测的提示
            print(comment)
            # 绘制绞刑架的身体部位
            draw_bodypart(gallows)
            # 打印绞刑架的状态
            print(gallows.render())
    
            # 错误猜测次数加一
            wrong_guesses += 1
            print("Sorry, that letter isn't in the word.")
    
            # 如果错误猜测次数达到上限，游戏结束
            if wrong_guesses == 10:
                print("Sorry, you lose. The word was " + guess_target)
                break
# 定义主函数，不返回任何结果
def main() -> None:
    # 打印游戏标题
    print(" " * 32 + "HANGMAN")
    # 打印游戏信息
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")

    # 随机打乱单词列表中的单词顺序
    random.shuffle(words)
    # 初始化当前单词索引和单词总数
    current_word = 0
    word_count = len(words)

    # 初始化游戏继续标志
    keep_playing = True
    # 当游戏继续时循环执行
    while keep_playing:

        # 进行游戏，传入当前单词
        play_game(words[current_word])
        # 当前单词索引加一
        current_word += 1

        # 如果已经玩过所有单词
        if current_word == word_count:
            # 打印提示信息
            print("You did all the words!!")
            # 设置游戏继续标志为 False
            keep_playing = False
        else:
            # 否则询问是否继续游戏
            keep_playing = (
                input("Want another word? (yes or no) ").lower().startswith("y")
            )

    # 打印结束语
    print("It's been fun! Bye for now.")


# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()
```