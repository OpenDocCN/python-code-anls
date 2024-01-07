# `basic-computer-games\69_Pizza\python\pizza.py`

```

"""
PIZZA

A pizza delivery simulation

Ported by Dave LeCompte
"""

import random  # 导入 random 模块，用于生成随机数

PAGE_WIDTH = 64  # 定义页面宽度常量

customer_names = [chr(65 + x) for x in range(16)]  # 生成包含 16 个顾客名字的列表
street_names = [str(n) for n in range(1, 5)]  # 生成包含 1 到 4 的街道名字的列表


def print_centered(msg: str) -> None:  # 定义一个打印居中文本的函数
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算居中需要的空格数
    print(spaces + msg)  # 打印居中文本


def print_header(title: str) -> None:  # 定义一个打印标题的函数
    print_centered(title)  # 打印居中标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印居中副标题
    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行


def print_ticks() -> None:  # 定义一个打印分隔线的函数
    for _ in range(4):  # 循环打印 4 条分隔线
        print("-")  # 打印分隔线


def print_ruler() -> None:  # 定义一个打印标尺的函数
    print(" -----1-----2-----3-----4-----")  # 打印标尺


def print_street(i: int) -> None:  # 定义一个打印街道信息的函数
    street_number = 3 - i  # 计算街道编号
    street_name = street_names[street_number]  # 获取街道名字
    line = street_name  # 初始化打印行

    space = " " * 5  # 初始化空格
    for customer_index in range(4):  # 遍历每个街道的 4 个顾客
        line += space  # 添加空格
        customer_name = customer_names[4 * street_number + customer_index]  # 获取顾客名字
        line += customer_name  # 添加顾客名字
    line += space  # 添加空格
    line += street_name  # 添加街道名字
    print(line)  # 打印街道信息


def print_map() -> None:  # 定义一个打印地图的函数
    print("MAP OF THE CITY OF HYATTSVILLE")  # 打印城市地图标题
    print()  # 打印空行
    print_ruler()  # 打印标尺
    for i in range(4):  # 遍历 4 条街道
        print_ticks()  # 打印分隔线
        print_street(i)  # 打印每条街道信息
    print_ticks()  # 打印分隔线
    print_ruler()  # 打印标尺
    print()  # 打印空行


def print_instructions() -> str:  # 定义一个打印游戏说明的函数
    print("PIZZA DELIVERY GAME")  # 打印游戏标题
    print()  # 打印空行
    print("WHAT IS YOUR FIRST NAME?")  # 提示输入玩家名字
    player_name = input()  # 获取玩家名字
    print()  # 打印空行
    print(f"HI, {player_name}.  IN THIS GAME YOU ARE TO TAKE ORDERS")  # 打印欢迎信息
    print("FOR PIZZAS.  THEN YOU ARE TO TELL A DELIVERY BOY")  # 打印游戏说明
    print("WHERE TO DELIVER THE ORDERED PIZZAS.")
    print()  # 打印空行
    print_map()  # 打印地图
    print("THE OUTPUT IS A MAP OF THE HOMES WHERE")  # 打印游戏说明
    print("YOU ARE TO SEND PIZZAS.")
    print()  # 打印空行
    print("YOUR JOB IS TO GIVE A TRUCK DRIVER")  # 打印游戏说明
    print("THE LOCATION OR COORDINATES OF THE")
    print("HOME ORDERING THE PIZZA.")
    print()  # 打印空行
    return player_name  # 返回玩家名字


def yes_no_prompt(msg: str) -> bool:  # 定义一个提示输入 yes 或 no 的函数
    while True:  # 循环直到输入合法的值
        print(msg)  # 打印提示信息
        response = input().upper()  # 获取用户输入并转换为大写

        if response == "YES":  # 如果输入是 YES
            return True  # 返回 True
        elif response == "NO":  # 如果输入是 NO
            return False  # 返回 False
        print("'YES' OR 'NO' PLEASE, NOW THEN,")  # 提示重新输入


def print_more_directions(player_name: str) -> None:  # 定义一个打印更多游戏说明的函数
    print()  # 打印空行
    print("SOMEBODY WILL ASK FOR A PIZZA TO BE")  # 打印游戏说明
    print("DELIVERED.  THEN A DELIVERY BOY WILL")
    print("ASK YOU FOR THE LOCATION.")
    print("     EXAMPLE:")
    print("THIS IS J.  PLEASE SEND A PIZZA.")
    print(f"DRIVER TO {player_name}.  WHERE DOES J LIVE?")
    print("YOUR ANSWER WOULD BE 2,3")
    print()


def calculate_customer_index(x: int, y: int) -> int:  # 定义一个计算顾客索引的函数
    return 4 * (y - 1) + x - 1  # 返回顾客索引


def deliver_to(customer_index, customer_name, player_name) -> bool:  # 定义一个派送披萨的函数
    print(f"  DRIVER TO {player_name}:  WHERE DOES {customer_name} LIVE?")  # 提示输入顾客位置

    coords = input()  # 获取输入的坐标
    xc, yc = (int(c) for c in coords.split(","))  # 解析坐标
    delivery_index = calculate_customer_index(xc, yc)  # 计算派送位置的顾客索引
    if delivery_index == customer_index:  # 如果派送位置和顾客索引一致
        print(f"HELLO {player_name}.  THIS IS {customer_name}, THANKS FOR THE PIZZA.")  # 打印派送成功信息
        return True  # 返回 True
    else:  # 如果派送位置和顾客索引不一致
        delivery_name = customer_names[delivery_index]  # 获取派送位置的顾客名字
        print(f"THIS IS {delivery_name}.  I DID NOT ORDER A PIZZA.")  # 打印派送失败信息
        print(f"I LIVE AT {xc},{yc}")  # 打印派送位置
        return False  # 返回 False


def play_game(num_turns, player_name) -> None:  # 定义一个游戏进行函数
    for _turn in range(num_turns):  # 循环进行游戏
        x = random.randint(1, 4)  # 随机生成 x 坐标
        y = random.randint(1, 4)  # 随机生成 y 坐标
        customer_index = calculate_customer_index(x, y)  # 计算顾客索引
        customer_name = customer_names[customer_index]  # 获取顾客名字

        print()  # 打印空行
        print(
            f"HELLO {player_name}'S PIZZA.  THIS IS {customer_name}.  PLEASE SEND A PIZZA."
        )  # 打印顾客点单信息
        while True:  # 循环直到派送成功
            success = deliver_to(customer_index, customer_name, player_name)  # 进行派送
            if success:  # 如果派送成功
                break  # 结束循环


def main() -> None:  # 定义主函数
    print_header("PIZZA")  # 打印游戏标题

    player_name = print_instructions()  # 打印游戏说明并获取玩家名字

    more_directions = yes_no_prompt("DO YOU NEED MORE DIRECTIONS?")  # 提示是否需要更多游戏说明

    if more_directions:  # 如果需要更多游戏说明
        print_more_directions(player_name)  # 打印更多游戏说明

        understand = yes_no_prompt("UNDERSTAND?")  # 提示是否理解了游戏说明

        if not understand:  # 如果没有理解
            print("THIS JOB IS DEFINITELY TOO DIFFICULT FOR YOU. THANKS ANYWAY")  # 提示工作太难了
            return  # 结束游戏

    print("GOOD.  YOU ARE NOW READY TO START TAKING ORDERS.")  # 提示准备好开始接单
    print()  # 打印空行
    print("GOOD LUCK!!")  # 祝好运
    print()  # 打印空行

    while True:  # 循环进行游戏
        num_turns = 5  # 设置游戏轮数
        play_game(num_turns, player_name)  # 进行游戏

        print()  # 打印空行
        more = yes_no_prompt("DO YOU WANT TO DELIVER MORE PIZZAS?")  # 提示是否继续派送披萨
        if not more:  # 如果不继续
            print(f"O.K. {player_name}, SEE YOU LATER!")  # 打印再见信息
            print()  # 打印空行
            return  # 结束游戏


if __name__ == "__main__":  # 如果是主程序入口
    main()  # 调用主函数

```