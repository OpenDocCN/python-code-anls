# `d:/src/tocomm/basic-computer-games\70_Poetry\python\poetry.py`

```
"""
POETRY

A poetry generator

Ported by Dave LeCompte
"""

import random  # 导入 random 模块，用于生成随机数
from dataclasses import dataclass  # 导入 dataclass 模块，用于创建数据类

PAGE_WIDTH = 64  # 设置页面宽度为 64

@dataclass  # 创建数据类装饰器
class State:  # 定义 State 类
    u: int = 0  # 初始化 u 属性为 0
    i: int = 0  # 初始化 i 属性为 0
    j: int = 0  # 初始化 j 属性为 0
    k: int = 0  # 初始化 k 属性为 0
    phrase: int = 1  # 定义一个整型变量 phrase，并赋值为 1
    line: str = ""   # 定义一个字符串变量 line，并赋值为空字符串


def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算需要添加的空格数，使得消息居中显示
    print(spaces + msg)  # 在居中位置打印消息


def process_phrase_1(state: State) -> str:
    line_1_options = [  # 定义一个包含多个字符串的列表
        "MIDNIGHT DREARY",
        "FIERY EYES",
        "BIRD OR FIEND",
        "THING OF EVIL",
        "PROPHET",
    ]
    state.line = state.line + line_1_options[state.i]  # 将 state.line 和 line_1_options[state.i] 拼接起来，并赋值给 state.line
    return state.line  # 返回拼接后的字符串
# 定义一个函数，接受一个State对象作为参数，返回空值
def process_phrase_2(state: State) -> None:
    # 定义一个包含不同选项的列表，每个选项是一个元组，包含一个字符串和一个可能为None的整数
    line_2_options = [
        ("BEGUILING ME", 2),
        ("THRILLED ME", None),
        ("STILL SITTING....", None),
        ("NEVER FLITTING", 2),
        ("BURNED", None),
    ]
    # 从选项列表中获取对应索引的单词和修饰符
    words, u_modifier = line_2_options[state.i]
    # 将获取的单词添加到State对象的line属性中
    state.line += words
    # 如果修饰符不为None，则将State对象的u属性设置为修饰符
    if not (u_modifier is None):
        state.u = u_modifier


# 定义一个函数，接受一个State对象作为参数，返回空值
def process_phrase_3(state: State) -> None:
    # 定义一个包含不同短语的列表，每个短语是一个元组，包含一个布尔值和一个字符串
    phrases = [
        (False, "AND MY SOUL"),
        (False, "DARKNESS THERE"),
        (False, "SHALL BE LIFTED"),
        (False, "QUOTH THE RAVEN"),  # 创建一个包含布尔值和字符串的元组
        (True, "SIGN OF PARTING"),   # 创建一个包含布尔值和字符串的元组
    ]

    only_if_u, words = phrases[state.i]  # 从phrases列表中获取指定索引位置的元组
    if (not only_if_u) or (state.u > 0):  # 如果only_if_u为假或者state.u大于0
        state.line = state.line + words  # 将words字符串添加到state.line中


def process_phrase_4(state: State) -> None:
    phrases = [
        ("NOTHING MORE"),       # 创建一个包含字符串的元组
        ("YET AGAIN"),          # 创建一个包含字符串的元组
        ("SLOWLY CREEPING"),    # 创建一个包含字符串的元组
        ("...EVERMORE"),        # 创建一个包含字符串的元组
        ("NEVERMORE"),          # 创建一个包含字符串的元组
    ]

    state.line += phrases[state.i]  # 将phrases列表中指定索引位置的字符串添加到state.line中
# 定义函数 maybe_comma，接受一个 State 对象作为参数，不返回任何结果
def maybe_comma(state: State) -> None:
    # 如果当前行的长度大于0且最后一个字符是句号，则不添加逗号
    if len(state.line) > 0 and state.line[-1] == ".":
        # 不管什么情况，都不在句号后面添加逗号
        return

    # 如果状态变量 u 不为0且随机数小于等于0.19
    if state.u != 0 and random.random() <= 0.19:
        # 在当前行末尾添加逗号和空格
        state.line += ", "
        state.u = 2
    # 如果随机数小于等于0.65
    if random.random() <= 0.65:
        # 在当前行末尾添加空格
        state.line += " "
        state.u += 1
    else:
        # 打印当前行内容
        print(state.line)
        # 重置当前行内容为空字符串
        state.line = ""
        # 重置状态变量 u 为0


# 定义函数 pick_phrase，接受一个 State 对象作为参数，不返回任何结果
def pick_phrase(state: State) -> None:
    # 生成一个0到4之间的随机整数，赋值给状态变量 i
    state.i = random.randint(0, 4)
    state.j += 1  # 增加 state 对象的 j 属性值
    state.k += 1  # 增加 state 对象的 k 属性值

    if state.u <= 0 and (state.j % 2) != 0:  # 如果 state 对象的 u 属性小于等于 0 并且 j 的值除以 2 的余数不等于 0
        # 随机缩进很有趣！
        state.line += " " * 5  # 在 state 对象的 line 属性后面添加 5 个空格
    state.phrase = state.j + 1  # 将 state 对象的 phrase 属性设置为 j 的值加 1


def main() -> None:
    print_centered("POETRY")  # 调用 print_centered 函数并传入字符串 "POETRY"
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 调用 print_centered 函数并传入字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n"

    state = State()  # 创建 State 类的实例对象并赋值给 state 变量

    phrase_processors = {  # 创建一个字典 phrase_processors
        1: process_phrase_1,  # 键为 1，值为 process_phrase_1 函数
        2: process_phrase_2,  # 键为 2，值为 process_phrase_2 函数
        3: process_phrase_3,  # 键为 3，值为 process_phrase_3 函数
        4: process_phrase_4,  # 键为 4，值为 process_phrase_4 函数
    }  # 结束 while 循环

    while True:  # 进入无限循环
        if state.phrase >= 1 and state.phrase <= 4:  # 如果 state.phrase 的值在 1 到 4 之间
            phrase_processors[state.phrase](state)  # 调用对应 state.phrase 值的 phrase_processors 函数，并传入 state 参数
            maybe_comma(state)  # 调用 maybe_comma 函数，传入 state 参数
        elif state.phrase == 5:  # 如果 state.phrase 的值为 5
            state.j = 0  # 将 state.j 的值设为 0
            print(state.line)  # 打印 state.line 的值
            state.line = ""  # 将 state.line 的值设为空字符串
            if state.k > 20:  # 如果 state.k 的值大于 20
                print()  # 打印空行
                state.u = 0  # 将 state.u 的值设为 0
                state.k = 0  # 将 state.k 的值设为 0
            else:  # 如果 state.k 的值不大于 20
                state.phrase = 2  # 将 state.phrase 的值设为 2
                continue  # 继续循环
        pick_phrase(state)  # 调用 pick_phrase 函数，传入 state 参数
# 如果当前脚本被直接执行，则执行 main() 函数
if __name__ == "__main__":
    main()
``` 

这段代码用于判断当前脚本是否被直接执行，如果是，则调用 main() 函数。这是一种常见的编程习惯，可以使代码更具可重用性和模块化。
```