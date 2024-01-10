# `basic-computer-games\70_Poetry\python\poetry.py`

```
"""
POETRY

A poetry generator

Ported by Dave LeCompte
"""

# 导入随机模块和数据类模块
import random
from dataclasses import dataclass

# 设置页面宽度
PAGE_WIDTH = 64

# 定义状态类
@dataclass
class State:
    u: int = 0
    i: int = 0
    j: int = 0
    k: int = 0
    phrase: int = 1
    line: str = ""

# 定义打印居中函数
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)

# 处理第一句诗句
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

# 处理第二句诗句
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

# 处理第三句诗句
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

# 处理第四句诗句
def process_phrase_4(state: State) -> None:
    phrases = [
        ("NOTHING MORE"),
        ("YET AGAIN"),
        ("SLOWLY CREEPING"),
        ("...EVERMORE"),
        ("NEVERMORE"),
    ]

    state.line += phrases[state.i]

# 可能添加逗号
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
    # 如果条件不满足，则打印状态行
    else:
        print(state.line)
        # 清空状态行
        state.line = ""
        # 重置状态变量 u 为 0
        state.u = 0
# 定义函数，从状态对象中选择短语
def pick_phrase(state: State) -> None:
    # 从0到4之间随机选择一个整数，赋值给state.i
    state.i = random.randint(0, 4)
    # state.j增加1
    state.j += 1
    # state.k增加1
    state.k += 1

    # 如果state.u小于等于0并且state.j除以2的余数不等于0
    if state.u <= 0 and (state.j % 2) != 0:
        # 随机缩进是有趣的！
        state.line += " " * 5
    # 将state.j加1后的值赋给state.phrase
    state.phrase = state.j + 1


# 定义主函数
def main() -> None:
    # 打印居中显示的"POETRY"
    print_centered("POETRY")
    # 打印居中显示的"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n"
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    # 创建状态对象
    state = State()

    # 定义短语处理器字典，键为1到4的整数，值为对应的处理函数
    phrase_processors = {
        1: process_phrase_1,
        2: process_phrase_2,
        3: process_phrase_3,
        4: process_phrase_4,
    }

    # 无限循环
    while True:
        # 如果state.phrase大于等于1并且小于等于4
        if state.phrase >= 1 and state.phrase <= 4:
            # 调用对应的短语处理函数，并传入状态对象
            phrase_processors[state.phrase](state)
            # 可能添加逗号到state.line
            maybe_comma(state)
        # 如果state.phrase等于5
        elif state.phrase == 5:
            # state.j赋值为0
            state.j = 0
            # 打印state.line
            print(state.line)
            # 清空state.line
            state.line = ""
            # 如果state.k大于20
            if state.k > 20:
                # 打印空行
                print()
                # state.u赋值为0
                state.u = 0
                # state.k赋值为0
                state.k = 0
            else:
                # state.phrase赋值为2
                state.phrase = 2
                # 继续下一次循环
                continue
        # 从状态对象中选择短语
        pick_phrase(state)


# 如果当前脚本为主程序
if __name__ == "__main__":
    # 调用主函数
    main()
```