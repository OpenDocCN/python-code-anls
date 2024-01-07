# `basic-computer-games\70_Poetry\python\poetry.py`

```

"""
POETRY

A poetry generator

Ported by Dave LeCompte
"""

# 导入所需的模块
import random
from dataclasses import dataclass

# 设置页面宽度
PAGE_WIDTH = 64

# 定义数据类 State
@dataclass
class State:
    u: int = 0
    i: int = 0
    j: int = 0
    k: int = 0
    phrase: int = 1
    line: str = ""

# 定义函数，将文本居中打印
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)

# 定义处理第一句诗的函数
def process_phrase_1(state: State) -> str:
    line_1_options = [
        "MIDNIGHT DREARY",
        "FIERY EYES",
        "BIRD OR FIEND",
        "THING OF EVIL",
        "PROPHET",
    ]
    state.line = state.line + line_1_options[state.i]
    return state.line

# 定义处理第二句诗的函数
def process_phrase_2(state: State) -> None:
    line_2_options = [
        ("BEGUILING ME", 2),
        ("THRILLED ME", None),
        ("STILL SITTING....", None),
        ("NEVER FLITTING", 2),
        ("BURNED", None),
    ]
    words, u_modifier = line_2_options[state.i]
    state.line += words
    if not (u_modifier is None):
        state.u = u_modifier

# 定义处理第三句诗的函数
def process_phrase_3(state: State) -> None:
    phrases = [
        (False, "AND MY SOUL"),
        (False, "DARKNESS THERE"),
        (False, "SHALL BE LIFTED"),
        (False, "QUOTH THE RAVEN"),
        (True, "SIGN OF PARTING"),
    ]

    only_if_u, words = phrases[state.i]
    if (not only_if_u) or (state.u > 0):
        state.line = state.line + words

# 定义处理第四句诗的函数
def process_phrase_4(state: State) -> None:
    phrases = [
        ("NOTHING MORE"),
        ("YET AGAIN"),
        ("SLOWLY CREEPING"),
        ("...EVERMORE"),
        ("NEVERMORE"),
    ]

    state.line += phrases[state.i]

# 定义可能添加逗号的函数
def maybe_comma(state: State) -> None:
    if len(state.line) > 0 and state.line[-1] == ".":
        # 不要在句号后面添加逗号
        return

    if state.u != 0 and random.random() <= 0.19:
        state.line += ", "
        state.u = 2
    if random.random() <= 0.65:
        state.line += " "
        state.u += 1
    else:
        print(state.line)
        state.line = ""
        state.u = 0

# 定义选择诗句的函数
def pick_phrase(state: State) -> None:
    state.i = random.randint(0, 4)
    state.j += 1
    state.k += 1

    if state.u <= 0 and (state.j % 2) != 0:
        # 随机缩进是有趣的！
        state.line += " " * 5
    state.phrase = state.j + 1

# 主函数
def main() -> None:
    print_centered("POETRY")
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    state = State()

    phrase_processors = {
        1: process_phrase_1,
        2: process_phrase_2,
        3: process_phrase_3,
        4: process_phrase_4,
    }

    while True:
        if state.phrase >= 1 and state.phrase <= 4:
            phrase_processors[state.phrase](state)
            maybe_comma(state)
        elif state.phrase == 5:
            state.j = 0
            print(state.line)
            state.line = ""
            if state.k > 20:
                print()
                state.u = 0
                state.k = 0
            else:
                state.phrase = 2
                continue
        pick_phrase(state)

# 如果作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()

```