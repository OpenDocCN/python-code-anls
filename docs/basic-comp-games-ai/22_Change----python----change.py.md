# `basic-computer-games\22_Change\python\change.py`

```
"""
CHANGE

Change calculator

Port by Dave LeCompte
"""

# 页面宽度
PAGE_WIDTH = 64


# 打印居中文本
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


# 打印标题
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


# 打印介绍
def print_introduction() -> None:
    print("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE")
    print("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.\n\n")


# 将分转换为美元字符串
def pennies_to_dollar_string(p: float) -> str:
    d = p / 100
    return f"${d:0.2f}"


# 计算找零
def compute_change() -> None:
    print("COST OF ITEM?")
    cost = float(input())
    print("AMOUNT OF PAYMENT?")
    payment = float(input())

    change_in_pennies = round((payment - cost) * 100)
    if change_in_pennies == 0:
        print("CORRECT AMOUNT, THANK YOU.")
        return

    if change_in_pennies < 0:
        short = -change_in_pennies / 100

        print(f"SORRY, YOU HAVE SHORT-CHANGED ME ${short:0.2f}")
        print()
        return

    print(f"YOUR CHANGE, {pennies_to_dollar_string(change_in_pennies)}")

    d = change_in_pennies // 1000
    if d > 0:
        print(f"{d} TEN DOLLAR BILL(S)")
    change_in_pennies -= d * 1000

    e = change_in_pennies // 500
    if e > 0:
        print(f"{e} FIVE DOLLAR BILL(S)")
    change_in_pennies -= e * 500

    f = change_in_pennies // 100
    if f > 0:
        print(f"{f} ONE DOLLAR BILL(S)")
    change_in_pennies -= f * 100

    g = change_in_pennies // 50
    if g > 0:
        print("ONE HALF DOLLAR")
    change_in_pennies -= g * 50

    h = change_in_pennies // 25
    if h > 0:
        print(f"{h} QUARTER(S)")
    change_in_pennies -= h * 25

    i = change_in_pennies // 10
    if i > 0:
        print(f"{i} DIME(S)")
    change_in_pennies -= i * 10

    j = change_in_pennies // 5
    if j > 0:
        print(f"{j} NICKEL(S)")
    change_in_pennies -= j * 5
    # 如果找零大于0，打印找零的数量和单位
    if change_in_pennies > 0:
        print(f"{change_in_pennies} PENNY(S)")
# 定义主函数，没有返回值
def main() -> None:
    # 打印标题为"CHANGE"
    print_header("CHANGE")
    # 打印介绍信息
    print_introduction()

    # 进入循环，持续执行以下操作
    while True:
        # 计算找零
        compute_change()
        # 打印感谢信息
        print("THANK YOU, COME AGAIN.\n\n")

# 如果当前脚本作为主程序执行，则调用主函数
if __name__ == "__main__":
    main()
```