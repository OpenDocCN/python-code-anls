# `basic-computer-games\85_Synonym\python\synonym.py`

```

"""
SYNONYM

Vocabulary quiz

Ported by Dave LeCompte
"""

import random  # 导入 random 模块

PAGE_WIDTH = 64  # 设置页面宽度为 64


def print_centered(msg: str) -> None:  # 定义一个打印居中文本的函数，参数为字符串，返回类型为 None
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算居中需要的空格数
    print(spaces + msg)  # 打印居中文本


def print_header(title: str) -> None:  # 定义一个打印标题的函数，参数为字符串，返回类型为 None
    print_centered(title)  # 调用打印居中文本的函数打印标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印固定文本
    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行


def print_instructions() -> None:  # 定义一个打印游戏说明的函数，返回类型为 None
    print("A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH")  # 打印游戏说明
    print("LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME MEANING.")
    print("I CHOOSE A WORD -- YOU TYPE A SYNONYM.")
    print("IF YOU CAN'T THINK OF A SYNONYM, TYPE THE WORD 'HELP'")
    print("AND I WILL TELL YOU A SYNONYM.")
    print()


right_words = ["RIGHT", "CORRECT", "FINE", "GOOD!", "CHECK"]  # 定义正确答案的单词列表

synonym_words = [  # 定义需要测试的单词及其同义词列表
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


def print_right() -> None:  # 定义一个打印正确提示的函数，返回类型为 None
    print(random.choice(right_words))  # 随机打印正确提示中的一个单词


def ask_question(question_number: int) -> None:  # 定义一个提问的函数，参数为整数，返回类型为 None
    words = synonym_words[question_number]  # 获取需要测试的单词及其同义词列表
    clues = words[:]  # 复制同义词列表
    base_word = clues.pop(0)  # 弹出同义词列表的第一个单词作为基准单词

    while True:  # 进入循环
        question = f"     WHAT IS A SYNONYM OF {base_word}? "  # 构造问题
        response = input(question).upper()  # 获取用户输入并转换为大写

        if response == "HELP":  # 如果用户输入为 "HELP"
            clue = random.choice(clues)  # 随机选择一个提示
            print(f"**** A SYNONYM OF {base_word} IS {clue}.")  # 打印提示
            print()

            # remove the clue from available clues
            clues.remove(clue)  # 从可用提示中移除已经使用的提示
            continue  # 继续下一轮循环

        if (response != base_word) and (response in words):  # 如果用户输入不是基准单词且在同义词列表中
            print_right()  # 调用打印正确提示的函数
            return  # 结束函数


def finish() -> None:  # 定义一个结束游戏的函数，返回类型为 None
    print()  # 打印空行
    print("SYNONYM DRILL COMPLETED.")  # 打印游戏结束提示


def main() -> None:  # 定义主函数，返回类型为 None
    print_header("SYNONYM")  # 调用打印标题的函数打印游戏标题
    print_instructions()  # 调用打印游戏说明的函数打印游戏说明

    num_questions = len(synonym_words)  # 获取需要测试的单词数量
    word_indices = list(range(num_questions))  # 生成单词索引列表
    random.shuffle(word_indices)  # 随机打乱单词索引列表

    for word_number in word_indices:  # 遍历打乱后的单词索引列表
        ask_question(word_number)  # 调用提问的函数进行测试

    finish()  # 调用结束游戏的函数


if __name__ == "__main__":  # 如果当前脚本为主程序
    main()  # 调用主函数开始游戏

```