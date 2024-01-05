# `22_Change\python\change.py`

```
"""
CHANGE

Change calculator

Port by Dave LeCompte
"""

PAGE_WIDTH = 64  # 设置页面宽度为64


def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算需要添加的空格数，使得消息居中显示
    print(spaces + msg)  # 打印居中显示的消息


def print_header(title: str) -> None:
    print_centered(title)  # 打印居中显示的标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印居中显示的副标题
def print_introduction() -> None:
    # 打印程序介绍信息
    print("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE")
    print("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.\n\n")


def pennies_to_dollar_string(p: float) -> str:
    # 将以分为单位的金额转换为以美元为单位的字符串
    d = p / 100
    return f"${d:0.2f}"


def compute_change() -> None:
    # 打印提示信息，获取商品价格和付款金额
    print("COST OF ITEM?")
    cost = float(input())
    print("AMOUNT OF PAYMENT?")
    payment = float(input())

    # 计算找零金额（以分为单位），四舍五入到整数
    change_in_pennies = round((payment - cost) * 100)
    # 如果找零金额为0，则打印正确金额的提示信息
    if change_in_pennies == 0:
        print("CORRECT AMOUNT, THANK YOU.")
        return  # 返回空值，结束函数执行

    if change_in_pennies < 0:  # 如果零钱小于0
        short = -change_in_pennies / 100  # 计算出少给的金额

        print(f"SORRY, YOU HAVE SHORT-CHANGED ME ${short:0.2f}")  # 打印出少给的金额
        print()  # 打印空行
        return  # 结束函数执行

    print(f"YOUR CHANGE, {pennies_to_dollar_string(change_in_pennies)}")  # 打印出找零的金额

    d = change_in_pennies // 1000  # 计算出需要的十美元纸币数量
    if d > 0:  # 如果需要的十美元纸币数量大于0
        print(f"{d} TEN DOLLAR BILL(S)")  # 打印出需要的十美元纸币数量
    change_in_pennies -= d * 1000  # 更新零钱数量

    e = change_in_pennies // 500  # 计算出需要的五美元纸币数量
    if e > 0:  # 如果需要的五美元纸币数量大于0
        print(f"{e} FIVE DOLLAR BILL(S)")  # 打印出需要的五美元纸币数量
    change_in_pennies -= e * 500  # 更新零钱数量
    # 计算找零金额中包含的一美元纸币数量
    f = change_in_pennies // 100
    if f > 0:
        # 打印一美元纸币的数量
        print(f"{f} ONE DOLLAR BILL(S)")
    # 更新找零金额
    change_in_pennies -= f * 100

    # 计算找零金额中包含的半美元硬币数量
    g = change_in_pennies // 50
    if g > 0:
        # 打印半美元硬币
        print("ONE HALF DOLLAR")
    # 更新找零金额
    change_in_pennies -= g * 50

    # 计算找零金额中包含的25美分硬币数量
    h = change_in_pennies // 25
    if h > 0:
        # 打印25美分硬币的数量
        print(f"{h} QUARTER(S)")
    # 更新找零金额
    change_in_pennies -= h * 25

    # 计算找零金额中包含的10美分硬币数量
    i = change_in_pennies // 10
    if i > 0:
        # 打印10美分硬币的数量
        print(f"{i} DIME(S)")
    # 更新找零金额
    change_in_pennies -= i * 10
    j = change_in_pennies // 5  # 计算找零中有多少个五分硬币
    if j > 0:  # 如果有五分硬币
        print(f"{j} NICKEL(S)")  # 打印出五分硬币的数量
    change_in_pennies -= j * 5  # 更新找零金额，减去五分硬币的价值

    if change_in_pennies > 0:  # 如果还有找零金额
        print(f"{change_in_pennies} PENNY(S)")  # 打印出剩余的找零金额是多少个一分硬币


def main() -> None:
    print_header("CHANGE")  # 打印标题
    print_introduction()  # 打印介绍

    while True:  # 无限循环
        compute_change()  # 计算找零
        print("THANK YOU, COME AGAIN.\n\n")  # 打印感谢信息


if __name__ == "__main__":
# 调用名为main的函数，但是在给定的代码中并没有定义这个函数，所以这行代码会导致错误。
```