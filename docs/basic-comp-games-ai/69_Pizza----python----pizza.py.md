# `basic-computer-games\69_Pizza\python\pizza.py`

```
"""
PIZZA

A pizza delivery simulation

Ported by Dave LeCompte
"""

# 导入 random 模块
import random

# 页面宽度
PAGE_WIDTH = 64

# 顾客姓名列表
customer_names = [chr(65 + x) for x in range(16)]
# 街道名称列表
street_names = [str(n) for n in range(1, 5)]


# 打印居中文本
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


# 打印标题
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print()
    print()
    print()


# 打印分隔线
def print_ticks() -> None:
    for _ in range(4):
        print("-")


# 打印标尺
def print_ruler() -> None:
    print(" -----1-----2-----3-----4-----")


# 打印街道信息
def print_street(i: int) -> None:
    street_number = 3 - i

    street_name = street_names[street_number]
    line = street_name

    space = " " * 5
    for customer_index in range(4):
        line += space
        customer_name = customer_names[4 * street_number + customer_index]
        line += customer_name
    line += space
    line += street_name
    print(line)


# 打印地图
def print_map() -> None:
    print("MAP OF THE CITY OF HYATTSVILLE")
    print()
    print_ruler()
    for i in range(4):
        print_ticks()
        print_street(i)
    print_ticks()
    print_ruler()
    print()


# 打印游戏说明
def print_instructions() -> str:
    print("PIZZA DELIVERY GAME")
    print()
    print("WHAT IS YOUR FIRST NAME?")
    player_name = input()
    print()
    print(f"HI, {player_name}.  IN THIS GAME YOU ARE TO TAKE ORDERS")
    print("FOR PIZZAS.  THEN YOU ARE TO TELL A DELIVERY BOY")
    print("WHERE TO DELIVER THE ORDERED PIZZAS.")
    print()
    print()

    print_map()

    print("THE OUTPUT IS A MAP OF THE HOMES WHERE")
    print("YOU ARE TO SEND PIZZAS.")
    print()
    print("YOUR JOB IS TO GIVE A TRUCK DRIVER")
    print("THE LOCATION OR COORDINATES OF THE")
    print("HOME ORDERING THE PIZZA.")
    print()

    return player_name


# 是/否提示
def yes_no_prompt(msg: str) -> bool:
    # 无限循环，直到条件被满足才会退出
    while True:
        # 打印消息
        print(msg)
        # 获取用户输入并转换为大写
        response = input().upper()

        # 如果用户输入为"YES"，则返回True
        if response == "YES":
            return True
        # 如果用户输入为"NO"，则返回False
        elif response == "NO":
            return False
        # 如果用户输入既不是"YES"也不是"NO"，则打印提示信息
        print("'YES' OR 'NO' PLEASE, NOW THEN,")
# 定义一个函数，用于打印更多的指示信息，参数为玩家的名字，返回空值
def print_more_directions(player_name: str) -> None:
    # 打印空行
    print()
    # 打印提示信息
    print("SOMEBODY WILL ASK FOR A PIZZA TO BE")
    print("DELIVERED.  THEN A DELIVERY BOY WILL")
    print("ASK YOU FOR THE LOCATION.")
    print("     EXAMPLE:")
    print("THIS IS J.  PLEASE SEND A PIZZA.")
    print(f"DRIVER TO {player_name}.  WHERE DOES J LIVE?")
    print("YOUR ANSWER WOULD BE 2,3")
    print()


# 定义一个函数，用于计算顾客的索引，参数为 x 和 y 坐标，返回顾客的索引
def calculate_customer_index(x: int, y: int) -> int:
    return 4 * (y - 1) + x - 1


# 定义一个函数，用于将披萨送到顾客那里，参数为顾客的索引、顾客的名字和玩家的名字，返回布尔值
def deliver_to(customer_index, customer_name, player_name) -> bool:
    # 打印送货员询问的信息
    print(f"  DRIVER TO {player_name}:  WHERE DOES {customer_name} LIVE?")

    # 获取玩家输入的坐标
    coords = input()
    xc, yc = (int(c) for c in coords.split(","))
    # 计算送货的索引
    delivery_index = calculate_customer_index(xc, yc)
    # 判断送货的索引是否与顾客的索引相同
    if delivery_index == customer_index:
        print(f"HELLO {player_name}.  THIS IS {customer_name}, THANKS FOR THE PIZZA.")
        return True
    else:
        delivery_name = customer_names[delivery_index]
        print(f"THIS IS {delivery_name}.  I DID NOT ORDER A PIZZA.")
        print(f"I LIVE AT {xc},{yc}")
        return False


# 定义一个函数，用于玩游戏，参数为回合数和玩家的名字，返回空值
def play_game(num_turns, player_name) -> None:
    # 循环进行指定回合数的游戏
    for _turn in range(num_turns):
        # 随机生成顾客的坐标
        x = random.randint(1, 4)
        y = random.randint(1, 4)
        # 计算顾客的索引和名字
        customer_index = calculate_customer_index(x, y)
        customer_name = customer_names[customer_index]

        # 打印顾客的信息
        print()
        print(
            f"HELLO {player_name}'S PIZZA.  THIS IS {customer_name}.  PLEASE SEND A PIZZA."
        )
        # 循环进行送货，直到成功为止
        while True:
            success = deliver_to(customer_index, customer_name, player_name)
            if success:
                break


# 定义一个函数，用于主程序逻辑，无参数，返回空值
def main() -> None:
    # 打印游戏标题
    print_header("PIZZA")

    # 打印游戏说明，获取玩家名字
    player_name = print_instructions()

    # 询问玩家是否需要更多指示信息
    more_directions = yes_no_prompt("DO YOU NEED MORE DIRECTIONS?")
    # 如果有更多的指示，打印更多的指示给玩家
    if more_directions:
        print_more_directions(player_name)

        # 询问玩家是否理解指示
        understand = yes_no_prompt("UNDERSTAND?")

        # 如果玩家不理解，打印提示信息并结束游戏
        if not understand:
            print("THIS JOB IS DEFINITELY TOO DIFFICULT FOR YOU. THANKS ANYWAY")
            return

    # 打印提示信息，玩家准备好开始接受订单
    print("GOOD.  YOU ARE NOW READY TO START TAKING ORDERS.")
    print()
    print("GOOD LUCK!!")
    print()

    # 循环进行游戏
    while True:
        # 设置游戏回合数为5
        num_turns = 5
        # 进行游戏
        play_game(num_turns, player_name)

        print()
        # 询问玩家是否想要继续送比萨
        more = yes_no_prompt("DO YOU WANT TO DELIVER MORE PIZZAS?")
        # 如果不想继续，打印提示信息并结束游戏
        if not more:
            print(f"O.K. {player_name}, SEE YOU LATER!")
            print()
            return
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```