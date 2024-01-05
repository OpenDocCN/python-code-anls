# `d:/src/tocomm/basic-computer-games\69_Pizza\python\pizza.py`

```
"""
PIZZA

A pizza delivery simulation

Ported by Dave LeCompte
"""

import random  # 导入 random 模块

PAGE_WIDTH = 64  # 设置页面宽度为 64

customer_names = [chr(65 + x) for x in range(16)]  # 生成包含 16 个字母的列表

street_names = [str(n) for n in range(1, 5)]  # 生成包含 1 到 4 的字符串列表


def print_centered(msg: str) -> None:  # 定义一个函数，参数为字符串类型，返回值为 None
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算需要添加的空格数
    print(spaces + msg)  # 打印居中的消息
def print_header(title: str) -> None:
    # 打印标题并居中显示
    print_centered(title)
    # 打印固定的信息并居中显示
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印空行
    print()
    print()
    print()


def print_ticks() -> None:
    # 打印4个短横线
    for _ in range(4):
        print("-")


def print_ruler() -> None:
    # 打印标尺
    print(" -----1-----2-----3-----4-----")


def print_street(i: int) -> None:
    # 计算街道号并打印
    street_number = 3 - i
    street_name = street_names[street_number]  # 从街道名称列表中获取特定街道编号对应的街道名称
    line = street_name  # 将街道名称赋值给变量line

    space = " " * 5  # 创建一个包含5个空格的字符串
    for customer_index in range(4):  # 遍历4次，表示每条街道有4个客户
        line += space  # 在line后面添加5个空格
        customer_name = customer_names[4 * street_number + customer_index]  # 从客户名称列表中获取特定街道编号对应的客户名称
        line += customer_name  # 将客户名称添加到line后面
    line += space  # 在line后面再添加5个空格
    line += street_name  # 将街道名称再次添加到line后面
    print(line)  # 打印出line的内容


def print_map() -> None:
    print("MAP OF THE CITY OF HYATTSVILLE")  # 打印城市名称
    print()  # 打印空行
    print_ruler()  # 调用print_ruler函数，打印地图标尺
    for i in range(4):  # 遍历4次，表示有4条街道
        print_ticks()  # 调用print_ticks函数，打印地图刻度
        print_street(i)  # 调用print_street函数，传入参数i
    print_ticks()  # 调用print_ticks函数
    print_ruler()  # 调用print_ruler函数
    print()  # 打印空行


def print_instructions() -> str:  # 定义print_instructions函数，返回类型为字符串
    print("PIZZA DELIVERY GAME")  # 打印游戏标题
    print()  # 打印空行
    print("WHAT IS YOUR FIRST NAME?")  # 打印提示信息
    player_name = input()  # 获取玩家输入的名字
    print()  # 打印空行
    print(f"HI, {player_name}.  IN THIS GAME YOU ARE TO TAKE ORDERS")  # 打印欢迎信息，包括玩家名字
    print("FOR PIZZAS.  THEN YOU ARE TO TELL A DELIVERY BOY")  # 打印游戏说明
    print("WHERE TO DELIVER THE ORDERED PIZZAS.")  # 打印游戏说明
    print()  # 打印空行
    print()  # 打印空行

    print_map()  # 调用print_map函数
    print("THE OUTPUT IS A MAP OF THE HOMES WHERE")  # 打印输出信息，指示输出是一个家庭地图
    print("YOU ARE TO SEND PIZZAS.")  # 打印输出信息，指示要发送比萨饼
    print()  # 打印空行

    print("YOUR JOB IS TO GIVE A TRUCK DRIVER")  # 打印输出信息，指示你的工作是给卡车司机
    print("THE LOCATION OR COORDINATES OF THE")  # 打印输出信息，指示位置或坐标
    print("HOME ORDERING THE PIZZA.")  # 打印输出信息，指示订购比萨饼的家庭
    print()  # 打印空行

    return player_name  # 返回玩家姓名


def yes_no_prompt(msg: str) -> bool:  # 定义一个函数，接受一个字符串参数并返回布尔值
    while True:  # 进入无限循环
        print(msg)  # 打印传入的消息
        response = input().upper()  # 获取用户输入并转换为大写

        if response == "YES":  # 如果用户输入是"YES"
            return True  # 返回True
        elif response == "NO":  # 如果用户输入是"NO"
            return False  # 返回False
        print("'YES' OR 'NO' PLEASE, NOW THEN,")
# 打印提示信息，要求输入'YES'或'NO'

def print_more_directions(player_name: str) -> None:
# 定义一个函数，打印更多的指示信息，参数为玩家的名字，返回类型为None
    print()
    print("SOMEBODY WILL ASK FOR A PIZZA TO BE")
    print("DELIVERED.  THEN A DELIVERY BOY WILL")
    print("ASK YOU FOR THE LOCATION.")
    print("     EXAMPLE:")
    print("THIS IS J.  PLEASE SEND A PIZZA.")
    print(f"DRIVER TO {player_name}.  WHERE DOES J LIVE?")
    print("YOUR ANSWER WOULD BE 2,3")
    print()

def calculate_customer_index(x: int, y: int) -> int:
# 定义一个函数，计算顾客的索引，参数为x和y坐标，返回类型为int
    return 4 * (y - 1) + x - 1

def deliver_to(customer_index, customer_name, player_name) -> bool:
# 定义一个函数，将披萨送到顾客那里，参数为顾客的索引、顾客的名字和玩家的名字，返回类型为bool
    print(f"  DRIVER TO {player_name}:  WHERE DOES {customer_name} LIVE?")
    # 打印出司机询问顾客住址的消息

    coords = input()
    # 获取用户输入的坐标
    xc, yc = (int(c) for c in coords.split(","))
    # 将输入的坐标字符串按逗号分隔，并转换为整数类型的坐标
    delivery_index = calculate_customer_index(xc, yc)
    # 根据输入的坐标计算出对应的顾客索引

    if delivery_index == customer_index:
        # 如果计算出的顾客索引与当前顾客索引相同
        print(f"HELLO {player_name}.  THIS IS {customer_name}, THANKS FOR THE PIZZA.")
        # 打印出顾客确认收到披萨的消息
        return True
        # 返回True，表示成功送达披萨
    else:
        # 如果计算出的顾客索引与当前顾客索引不同
        delivery_name = customer_names[delivery_index]
        # 获取对应索引的顾客姓名
        print(f"THIS IS {delivery_name}.  I DID NOT ORDER A PIZZA.")
        # 打印出顾客未订购披萨的消息
        print(f"I LIVE AT {xc},{yc}")
        # 打印出顾客的实际住址
        return False
        # 返回False，表示未成功送达披萨

def play_game(num_turns, player_name) -> None:
    # 定义一个玩游戏的函数，参数为游戏轮数和玩家姓名
    for _turn in range(num_turns):
        # 循环进行游戏轮数次数
        x = random.randint(1, 4)
        # 生成1到4之间的随机整数作为x坐标
        y = random.randint(1, 4)
        # 生成1到4之间的随机整数作为y坐标
        customer_index = calculate_customer_index(x, y)
        # 根据生成的坐标计算出对应的顾客索引
        customer_name = customer_names[customer_index]  # 从顾客名单中根据索引获取顾客姓名

        print()  # 打印空行
        print(
            f"HELLO {player_name}'S PIZZA.  THIS IS {customer_name}.  PLEASE SEND A PIZZA."
        )  # 打印欢迎信息，包括玩家姓名和顾客姓名
        while True:  # 进入无限循环
            success = deliver_to(customer_index, customer_name, player_name)  # 调用deliver_to函数，将披萨送到指定顾客处
            if success:  # 如果送货成功
                break  # 退出循环


def main() -> None:  # 定义主函数，不返回任何结果
    print_header("PIZZA")  # 打印标题为"PIZZA"

    player_name = print_instructions()  # 调用print_instructions函数，获取玩家姓名

    more_directions = yes_no_prompt("DO YOU NEED MORE DIRECTIONS?")  # 调用yes_no_prompt函数，询问是否需要更多指引

    if more_directions:  # 如果需要更多指引
# 打印更多指示给玩家的信息
print_more_directions(player_name)

# 提示玩家是否理解了指示
understand = yes_no_prompt("UNDERSTAND?")

# 如果玩家不理解，则打印提示信息并结束程序
if not understand:
    print("THIS JOB IS DEFINITELY TOO DIFFICULT FOR YOU. THANKS ANYWAY")
    return

# 打印提示信息，玩家准备好接受订单
print("GOOD.  YOU ARE NOW READY TO START TAKING ORDERS.")
print()
print("GOOD LUCK!!")
print()

# 进入游戏循环
while True:
    # 设置游戏回合数为5
    num_turns = 5
    # 开始游戏
    play_game(num_turns, player_name)

    # 提示玩家是否想要继续送更多的比萨
    more = yes_no_prompt("DO YOU WANT TO DELIVER MORE PIZZAS?")
    # 如果玩家不想继续，则结束游戏循环
    if not more:
# 打印欢送消息，使用了 f-string 格式化输出
print(f"O.K. {player_name}, SEE YOU LATER!")
# 打印空行
print()
# 返回，结束程序
return

# 如果当前脚本被直接执行，则调用 main 函数
if __name__ == "__main__":
    main()
```