# `73_Reverse\python\reverse.py`

```
#!/usr/bin/env python3  # 指定脚本的解释器为 Python3

import random  # 导入 random 模块，用于生成随机数
import textwrap  # 导入 textwrap 模块，用于格式化文本

NUMCNT = 9  # 定义常量 NUMCNT，表示游戏中使用的数字个数

def main() -> None:  # 定义主函数 main，返回类型为 None
    print("REVERSE".center(72))  # 在 72 个字符宽度内居中打印字符串 "REVERSE"
    print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".center(72))  # 在 72 个字符宽度内居中打印字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print()  # 打印空行
    print()  # 再次打印空行
    print("REVERSE -- A GAME OF SKILL")  # 打印游戏标题 "REVERSE -- A GAME OF SKILL"
    print()

    if not input("DO YOU WANT THE RULES? (yes/no) ").lower().startswith("n"):  # 如果用户输入的不是以字母 "n" 开头的字符串
        print_rules()  # 调用函数打印游戏规则

    while True:  # 进入无限循环
        game_loop()  # 调用游戏循环函数
        if not input("TRY AGAIN? (yes/no) ").lower().startswith("y"):
            return
```

这行代码是一个条件语句，如果用户输入的不是以字母y开头的字符串，则返回。这是一个用于询问用户是否要再次尝试的功能。

```
def game_loop() -> None:
    """Play the main game."""
    # Make a random list from 1 to NUMCNT
    numbers = list(range(1, NUMCNT + 1))
    random.shuffle(numbers)

    # Print original list and start the game
    print()
    print("HERE WE GO ... THE LIST IS:")
    print_list(numbers)

    turns = 0
    while True:
        try:
            howmany = int(input("HOW MANY SHALL I REVERSE? "))
```

这段代码是一个函数定义，用于定义一个游戏循环。在游戏循环中，首先生成一个从1到NUMCNT的随机列表，然后打印原始列表并开始游戏。接着进入一个无限循环，用户需要输入一个整数来确定要反转列表的数量。
# 确保 howmany 的值大于等于 0
assert howmany >= 0
# 如果 howmany 的值小于 0，则抛出 ValueError 或 AssertionError 异常，然后继续执行下一次循环
except (ValueError, AssertionError):
    continue
# 如果 howmany 的值等于 0，则返回
if howmany == 0:
    return
# 如果 howmany 的值大于 NUMCNT，则打印错误信息并继续执行下一次循环
if howmany > NUMCNT:
    print("OOPS! WRONG! I CAN REVERSE AT MOST", NUMCNT)
    continue
# 增加 turns 计数
turns += 1
# 将前 howmany 个元素进行反转，并将反转后的列表与剩余的元素组合成新的列表
newnums = numbers[0:howmany]
newnums.reverse()
newnums.extend(numbers[howmany:])
numbers = newnums
# 打印反转后的列表
print_list(numbers)
        # Check for a win
        # 检查是否获胜
        if all(numbers[i] == i + 1 for i in range(NUMCNT)):
            # 如果所有数字都按顺序排列，打印获胜信息
            print(f"YOU WON IT IN {turns} MOVES!")
            print()
            return
            # 返回

def print_list(numbers) -> None:
    # 打印列表中的数字
    print(" ".join(map(str, numbers)))

def print_rules() -> None:
    # 打印游戏规则
    help = textwrap.dedent(
        """
        THIS IS THE GAME OF "REVERSE".  TO WIN, ALL YOU HAVE
        TO DO IS ARRANGE A LIST OF NUMBERS (1 THROUGH {})
        IN NUMERICAL ORDER FROM LEFT TO RIGHT.  TO MOVE, YOU
        TELL ME HOW MANY NUMBERS (COUNTING FROM THE LEFT) TO
        REVERSE.  FOR EXAMPLE, IF THE CURRENT LIST IS:
2. AND YOU REVERSE 4, THE RESULT WILL BE: - 输出提示信息
3. 5 4 3 2 1 6 7 8 9 - 显示示例中的数字序列
4. NOW IF YOU REVERSE 5, YOU WIN! - 输出提示信息
5. 1 2 3 4 5 6 7 8 9 - 显示示例中的数字序列
6. NO DOUBT YOU WILL LIKE THIS GAME, BUT - 输出提示信息
7. IF YOU WANT TO QUIT, REVERSE 0 (ZERO). - 输出提示信息，提示如何退出游戏
if __name__ == "__main__":
    # 如果当前脚本被直接执行，而不是被导入到其他模块中，则执行以下代码
    try:
        # 调用主函数
        main()
    except KeyboardInterrupt:
        # 捕获键盘中断异常，不做任何处理
        pass
```