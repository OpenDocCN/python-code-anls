# `basic-computer-games\84_Super_Star_Trek\python\superstartrekins.py`

```

"""
SUPER STARTREK INSTRUCTIONS
MAR 5, 1978

Just the instructions for SUPERSTARTREK

Ported by Dave LeCompte
"""

# 定义一个函数，用于获取用户输入的是或否，返回布尔值
def get_yes_no(prompt: str) -> bool:
    # 获取用户输入并转换为大写
    response = input(prompt).upper()
    # 返回用户输入的第一个字符是否不是"N"
    return response[0] != "N"

# 定义一个函数，用于打印游戏的标题
def print_header() -> None:
    # 打印12行空行
    for _ in range(12):
        print()
    t10 = " " * 10
    # 打印游戏标题
    print(t10 + "*************************************")
    print(t10 + "*                                   *")
    print(t10 + "*                                   *")
    print(t10 + "*      * * SUPER STAR TREK * *      *")
    print(t10 + "*                                   *")
    print(t10 + "*                                   *")
    print(t10 + "*************************************")
    # 打印8行空行
    for _ in range(8)

# 定义主函数
def main() -> None:
    # 打印游戏标题
    print_header()
    # 如果用户不需要游戏说明，则返回
    if not get_yes_no("DO YOU NEED INSTRUCTIONS (Y/N)? "):
        return
    # 打印游戏说明
    print_instructions()

# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()

```