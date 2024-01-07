# `basic-computer-games\43_Hammurabi\python\hamurabi.py`

```

# 从 random 模块中导入 random 和 seed 函数
from random import random, seed

# 生成一个随机整数
def gen_random() -> int:
    return int(random() * 5) + 1

# 打印错误信息
def bad_input_850() -> None:
    print("\nHAMURABI:  I CANNOT DO WHAT YOU WISH.")
    print("GET YOURSELF ANOTHER STEWARD!!!!!")

# 打印错误信息和谷物数量
def bad_input_710(grain_bushels: int) -> None:
    print("HAMURABI:  THINK AGAIN.  YOU HAVE ONLY")
    print(f"{grain_bushels} BUSHELS OF GRAIN.  NOW THEN,")

# 打印错误信息和土地数量
def bad_input_720(acres: float) -> None:
    print(f"HAMURABI:  THINK AGAIN.  YOU OWN ONLY {acres} ACRES.  NOW THEN,")

# 打印国家管理不善的信息
def national_fink() -> None:
    print("DUE TO THIS EXTREME MISMANAGEMENT YOU HAVE NOT ONLY")
    print("BEEN IMPEACHED AND THROWN OUT OF OFFICE BUT YOU HAVE")
    print("ALSO BEEN DECLARED NATIONAL FINK!!!!")

# 模拟 BASIC 输入，拒绝非数字值
def b_input(promptstring: str) -> int:
    """emulate BASIC input. It rejects non-numeric values"""
    x = input(promptstring)
    while x.isalpha():
        x = input("?REDO FROM START\n? ")
    return int(x)

# 如果作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()

```