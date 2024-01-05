# `96_Word\python\word.py`

```
#!/usr/bin/env python3  # 指定脚本的解释器为 Python 3

"""
WORD

Converted from BASIC to Python by Trevor Hobson
"""

import random  # 导入 random 模块

words = [  # 创建包含单词的列表
    "DINKY",
    "SMOKE",
    "WATER",
    "GRASS",
    "TRAIN",
    "MIGHT",
    "FIRST",
    "CANDY",
    "CHAMP",
    "WOULD",
    "CLUMP",
    "DOPEY",
]
# 定义一个包含三个五个字母单词的列表

def play_game() -> None:
    """Play one round of the game"""
    # 定义一个函数用于玩游戏

    random.shuffle(words)
    # 随机打乱单词列表中的顺序
    target_word = words[0]
    # 选择打乱后的列表中的第一个单词作为目标单词
    guess_count = 0
    # 初始化猜测次数为0
    guess_progress = ["-"] * 5
    # 初始化猜测进度为包含5个"-"的列表

    print("You are starting a new game...")
    # 打印提示信息，表示开始新游戏
    while True:
        # 进入循环
        guess_word = ""
        # 初始化猜测单词为空字符串
        while guess_word == "":
            # 进入循环，直到猜测单词不为空
            guess_word = input("\nGuess a five letter word. ").upper()
            # 获取用户输入的猜测单词并转换为大写
            if guess_word == "?":
                # 如果用户输入的是"?"，则执行以下操作
                break  # 结束当前循环，跳出循环体
            elif not guess_word.isalpha() or len(guess_word) != 5:  # 如果猜测的单词不全是字母或长度不是5
                guess_word = ""  # 重置猜测的单词为空字符串
                print("You must guess a five letter word. Start again.")  # 打印提示信息
        guess_count += 1  # 猜测次数加一
        if guess_word == "?":  # 如果猜测的单词是问号
            print("The secret word is", target_word)  # 打印出正确的单词
            break  # 结束当前循环，跳出循环体
        else:  # 否则
            common_letters = ""  # 初始化公共字母为空字符串
            matches = 0  # 匹配数初始化为0
            for i in range(5):  # 遍历5次
                for j in range(5):  # 再次遍历5次
                    if guess_word[i] == target_word[j]:  # 如果猜测的单词的第i个字母等于目标单词的第j个字母
                        matches += 1  # 匹配数加一
                        common_letters = common_letters + guess_word[i]  # 将匹配的字母添加到公共字母中
                        if i == j:  # 如果位置也相同
                            guess_progress[j] = guess_word[i]  # 更新猜测进度
            print(
                f"There were {matches}",  # 打印匹配数
def main() -> None:
    # 打印游戏标题
    print(" " * 33 + "WORD")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")

    # 打印游戏提示
    print("I am thinking of a word -- you guess it. I will give you")
    print("clues to help you get it. Good luck!!\n")
    keep_playing = True  # 设置一个变量，用于控制是否继续玩游戏
    while keep_playing:  # 当 keep_playing 为 True 时，循环执行下面的代码
        play_game()  # 调用 play_game() 函数，开始游戏
        keep_playing = input("\nWant to play again? ").lower().startswith("y")  # 获取用户输入，如果以 "y" 开头，则继续玩游戏，否则结束循环

if __name__ == "__main__":  # 如果当前脚本被直接执行，而不是被导入其他模块
    main()  # 调用 main() 函数，开始执行程序的主要逻辑
```