# `basic-computer-games\55_Life\python\life.py`

```py
"""
LIFE

An implementation of John Conway's popular cellular automaton

Ported by Dave LeCompte
"""

from typing import Dict

# 设置页面宽度
PAGE_WIDTH = 64

# 设置最大宽度和高度
MAX_WIDTH = 70
MAX_HEIGHT = 24


# 打印居中文本
def print_centered(msg) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


# 打印标题
def print_header(title) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print()
    print()
    print()


# 获取模式
def get_pattern() -> Dict[int, str]:
    print("ENTER YOUR PATTERN:")
    c = 0

    pattern: Dict[int, str] = {}
    while True:
        line = input()
        if line == "DONE":
            return pattern

        # BASIC input would strip of leading whitespace.
        # Python input does not. The following allows you to start a
        # line with a dot to disable the whitespace stripping. This is
        # unnecessary for Python, but for historical accuracy, it's
        # staying in.

        if line[0] == ".":
            line = " " + line[1:]
        pattern[c] = line
        c += 1


# 主函数
def main() -> None:
    print_header("LIFE")

    pattern = get_pattern()

    pattern_height = len(pattern)
    pattern_width = 0
    for _line_num, line in pattern.items():
        pattern_width = max(pattern_width, len(line))

    min_x = 11 - pattern_height // 2
    min_y = 33 - pattern_width // 2
    max_x = MAX_HEIGHT - 1
    max_y = MAX_WIDTH - 1

    a = [[0 for y in range(MAX_WIDTH)] for x in range(MAX_HEIGHT)]
    p = 0
    g = 0
    invalid = False

    # line 140
    # 将输入模式转录到活动数组中
    for x in range(0, pattern_height):
        for y in range(0, len(pattern[x])):
            if pattern[x][y] != " ":
                a[min_x + x][min_y + y] = 1
                p += 1

    print()
    print()
    print()
if __name__ == "__main__":
    main()
```