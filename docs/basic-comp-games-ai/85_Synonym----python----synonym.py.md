# `basic-computer-games\85_Synonym\python\synonym.py`

```
# 定义了一个多义词词汇测试程序
"""
SYNONYM

Vocabulary quiz

Ported by Dave LeCompte
"""

# 导入 random 模块
import random

# 定义页面宽度常量
PAGE_WIDTH = 64


# 定义打印居中文本的函数
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


# 定义打印标题的函数
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print()
    print()
    print()


# 定义打印说明的函数
def print_instructions() -> None:
    print("A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH")
    print("LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME MEANING.")
    print("I CHOOSE A WORD -- YOU TYPE A SYNONYM.")
    print("IF YOU CAN'T THINK OF A SYNONYM, TYPE THE WORD 'HELP'")
    print("AND I WILL TELL YOU A SYNONYM.")
    print()


# 定义正确答案的列表
right_words = ["RIGHT", "CORRECT", "FINE", "GOOD!", "CHECK"]

# 定义多义词列表
synonym_words = [
    ["FIRST", "START", "BEGINNING", "ONSET", "INITIAL"],
    ["SIMILAR", "ALIKE", "SAME", "LIKE", "RESEMBLING"],
    ["MODEL", "PATTERN", "PROTOTYPE", "STANDARD", "CRITERION"],
    ["SMALL", "INSIGNIFICANT", "LITTLE", "TINY", "MINUTE"],
    ["STOP", "HALT", "STAY", "ARREST", "CHECK", "STANDSTILL"],
    ["HOUSE", "DWELLING", "RESIDENCE", "DOMICILE", "LODGING", "HABITATION"],
    ["PIT", "HOLE", "HOLLOW", "WELL", "GULF", "CHASM", "ABYSS"],
    ["PUSH", "SHOVE", "THRUST", "PROD", "POKE", "BUTT", "PRESS"],
    ["RED", "ROUGE", "SCARLET", "CRIMSON", "FLAME", "RUBY"],
    ["PAIN", "SUFFERING", "HURT", "MISERY", "DISTRESS", "ACHE", "DISCOMFORT"],
]


# 定义打印正确答案的函数
def print_right() -> None:
    print(random.choice(right_words))


# 定义提问的函数
def ask_question(question_number: int) -> None:
    words = synonym_words[question_number]
    clues = words[:]
    base_word = clues.pop(0)
    # 无限循环，直到条件不满足
    while True:
        # 构造问题字符串，要求用户输入同义词
        question = f"     WHAT IS A SYNONYM OF {base_word}? "
        # 获取用户输入并转换为大写
        response = input(question).upper()

        # 如果用户输入为"HELP"，随机选择一个提示词并打印
        if response == "HELP":
            clue = random.choice(clues)
            print(f"**** A SYNONYM OF {base_word} IS {clue}.")
            print()

            # 从可用提示中移除已经使用的提示
            clues.remove(clue)
            # 继续下一轮循环
            continue

        # 如果用户输入不是基础单词，并且在单词列表中，则打印正确信息并返回
        if (response != base_word) and (response in words):
            print_right()
            return
# 定义一个没有返回值的函数，用于输出空行和完成提示信息
def finish() -> None:
    # 输出空行
    print()
    # 输出“SYNONYM DRILL COMPLETED.”提示信息
    print("SYNONYM DRILL COMPLETED.")


# 定义一个没有返回值的函数，用于执行主程序
def main() -> None:
    # 输出“SYNONYM”标题
    print_header("SYNONYM")
    # 输出游戏说明
    print_instructions()

    # 获取同义词列表的长度
    num_questions = len(synonym_words)
    # 创建一个包含所有同义词索引的列表
    word_indices = list(range(num_questions))
    # 打乱同义词索引的顺序
    random.shuffle(word_indices)

    # 遍历打乱后的同义词索引列表
    for word_number in word_indices:
        # 提问玩家问题
        ask_question(word_number)

    # 完成游戏
    finish()


# 如果当前脚本被直接执行，则执行主程序
if __name__ == "__main__":
    main()
```