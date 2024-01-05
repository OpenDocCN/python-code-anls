# `85_Synonym\python\synonym.py`

```
"""
SYNONYM

Vocabulary quiz

Ported by Dave LeCompte
"""

import random  # 导入 random 模块，用于生成随机数

PAGE_WIDTH = 64  # 设置页面宽度为 64

# 定义一个函数，用于打印居中的文本
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算需要添加的空格数，使得文本居中
    print(spaces + msg)  # 打印居中的文本

# 定义一个函数，用于打印标题
def print_header(title: str) -> None:
    print_centered(title)  # 调用 print_centered 函数打印标题
    # 打印居中对齐的文本
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印空行
    print()
    print()
    print()


def print_instructions() -> None:
    # 打印游戏说明
    print("A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH")
    print("LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME MEANING.")
    print("I CHOOSE A WORD -- YOU TYPE A SYNONYM.")
    print("IF YOU CAN'T THINK OF A SYNONYM, TYPE THE WORD 'HELP'")
    print("AND I WILL TELL YOU A SYNONYM.")
    print()


right_words = ["RIGHT", "CORRECT", "FINE", "GOOD!", "CHECK"]

synonym_words = [
    # 同义词列表
    ["FIRST", "START", "BEGINNING", "ONSET", "INITIAL"],
    ["SIMILAR", "ALIKE", "SAME", "LIKE", "RESEMBLING"],
    ["MODEL", "PATTERN", "PROTOTYPE", "STANDARD", "CRITERION"],  # 创建一个包含同义词的列表
    ["SMALL", "INSIGNIFICANT", "LITTLE", "TINY", "MINUTE"],  # 创建一个包含同义词的列表
    ["STOP", "HALT", "STAY", "ARREST", "CHECK", "STANDSTILL"],  # 创建一个包含同义词的列表
    ["HOUSE", "DWELLING", "RESIDENCE", "DOMICILE", "LODGING", "HABITATION"],  # 创建一个包含同义词的列表
    ["PIT", "HOLE", "HOLLOW", "WELL", "GULF", "CHASM", "ABYSS"],  # 创建一个包含同义词的列表
    ["PUSH", "SHOVE", "THRUST", "PROD", "POKE", "BUTT", "PRESS"],  # 创建一个包含同义词的列表
    ["RED", "ROUGE", "SCARLET", "CRIMSON", "FLAME", "RUBY"],  # 创建一个包含同义词的列表
    ["PAIN", "SUFFERING", "HURT", "MISERY", "DISTRESS", "ACHE", "DISCOMFORT"],  # 创建一个包含同义词的列表
]


def print_right() -> None:
    print(random.choice(right_words))  # 从同义词列表中随机选择一个词并打印出来


def ask_question(question_number: int) -> None:
    words = synonym_words[question_number]  # 从同义词列表中获取指定索引的同义词列表
    clues = words[:]  # 复制同义词列表
    base_word = clues.pop(0)  # 从复制的同义词列表中移除第一个词，并将其赋值给base_word
    while True:  # 创建一个无限循环，直到条件被满足才会退出循环
        question = f"     WHAT IS A SYNONYM OF {base_word}? "  # 创建一个包含base_word的问题字符串
        response = input(question).upper()  # 获取用户输入的答案并转换为大写

        if response == "HELP":  # 如果用户输入的是"HELP"
            clue = random.choice(clues)  # 从clues列表中随机选择一个线索
            print(f"**** A SYNONYM OF {base_word} IS {clue}.")  # 打印出base_word的一个同义词
            print()

            # remove the clue from available clues
            clues.remove(clue)  # 从可用的线索中移除选中的线索
            continue  # 继续下一次循环

        if (response != base_word) and (response in words):  # 如果用户输入的不是base_word且在words列表中
            print_right()  # 调用print_right函数
            return  # 退出当前函数

def finish() -> None:  # 定义一个名为finish的函数，不返回任何值
    print()  # 打印空行
    print("SYNONYM DRILL COMPLETED.")  # 打印提示信息，表示同义词练习已完成


def main() -> None:
    print_header("SYNONYM")  # 调用打印标题的函数，打印“SYNONYM”
    print_instructions()  # 调用打印指令的函数，打印练习指令

    num_questions = len(synonym_words)  # 获取同义词列表的长度，即问题的数量
    word_indices = list(range(num_questions))  # 创建一个包含问题数量的索引列表
    random.shuffle(word_indices)  # 随机打乱索引列表顺序

    for word_number in word_indices:  # 遍历打乱后的索引列表
        ask_question(word_number)  # 调用提问函数，提问对应索引的问题

    finish()  # 调用结束函数，表示练习结束


if __name__ == "__main__":
    main()  # 如果当前脚本作为主程序运行，则调用主函数
```