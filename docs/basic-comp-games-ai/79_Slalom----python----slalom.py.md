# `79_Slalom\python\slalom.py`

```
from random import random  # 从 random 模块中导入 random 函数

medals = {  # 创建一个字典 medals，包含金牌、银牌和铜牌的初始值
    "gold": 0,
    "silver": 0,
    "bronze": 0,
}

def ask(question: str) -> str:  # 定义一个函数 ask，接受一个字符串参数并返回一个字符串
    print(question, end="? ")  # 打印问题并以问号结尾
    return input().upper()  # 获取用户输入并转换为大写形式返回

def ask_int(question: str) -> int:  # 定义一个函数 ask_int，接受一个字符串参数并返回一个整数
    reply = ask(question)  # 调用 ask 函数获取用户输入
    return int(reply) if reply.isnumeric() else -1  # 如果用户输入是数字则转换为整数返回，否则返回-1

def pre_run(gates, max_speeds) -> None:  # 定义一个函数 pre_run，接受 gates 和 max_speeds 两个参数，不返回任何值
    # 打印提示信息
    print('\nType "INS" for instructions')
    # 打印提示信息
    print('Type "MAX" for approximate maximum speeds')
    # 打印提示信息
    print('Type "RUN" for the beginning of the race')
    # 询问用户输入指令
    cmd = ask("Command--")
    # 当用户输入的指令不是"RUN"时，进入循环
    while cmd != "RUN":
        # 如果用户输入的指令是"INS"，则打印游戏说明
        if cmd == "INS":
            print("\n*** Slalom: This is the 1976 Winter Olypic Giant Slalom.  You are")
            print("            the American team's only hope for a gold medal.\n")
            print("     0 -- Type this if you want to see how long you've taken.")
            print("     1 -- Type this if you want to speed up a lot.")
            print("     2 -- Type this if you want to speed up a little.")
            print("     3 -- Type this if you want to speed up a teensy.")
            print("     4 -- Type this if you want to keep going the same speed.")
            print("     5 -- Type this if you want to check a teensy.")
            print("     6 -- Type this if you want to check a little.")
            print("     7 -- Type this if you want to check a lot.")
            print("     8 -- Type this if you want to cheat and try to skip a gate.\n")
            print(" The place to use these options is when the Computer asks:\n")
            print("Option?\n")
            print("                Good Luck!\n")
            cmd = ask("Command--")  # 询问用户输入命令
        elif cmd == "MAX":  # 如果命令是"MAX"
            print("Gate Max")  # 打印"Gate Max"
            print(" # M.P.H.")  # 打印" # M.P.H."
            print("----------")  # 打印"----------"
            for i in range(0, gates):  # 遍历门的数量
                print(f" {i + 1}  {max_speeds[i]}")  # 打印门的编号和对应的最大速度
            cmd = ask("Command--")  # 询问用户输入命令
        else:  # 如果命令不是"MAX"
            cmd = ask(f'"{cmd}" is an illegal command--Retry')  # 询问用户重新输入命令


def run(gates, lvl, max_speeds) -> None:  # 定义一个名为run的函数，接受门的数量、级别和最大速度的列表作为参数
    global medals  # 声明使用全局变量medals
    print("The starter counts down...5...4...3...2...1...Go!")  # 打印倒计时
    time: float = 0  # 初始化时间为0
    speed = int(random() * (18 - 9) + 9)  # 生成一个随机速度
    print("You're off")  # 打印"You're off"
    for i in range(0, gates):  # 遍历门的数量
        while True:  # 进入无限循环
            # 打印关卡信息
            print(f"\nHere comes gate #{i + 1}:")
            # 打印速度信息
            print(f" {int(speed)} M.P.H.")
            # 保存当前速度
            old_speed = speed
            # 获取用户输入的选项
            opt = ask_int("Option")
            # 当选项不在1到8之间时，进行循环
            while opt < 1 or opt > 8:
                # 如果选项为0，打印已经花费的时间
                if opt == 0:
                    print(f"You've taken {int(time)} seconds.")
                else:
                    # 如果选项不在1到8之间，打印"What?"
                    print("What?")
                # 重新获取用户输入的选项
                opt = ask_int("Option")

            # 如果选项为8，进行作弊检测
            if opt == 8:
                # 打印作弊信息
                print("***Cheat")
                # 如果随机数小于0.7，表示被官员抓住
                if random() < 0.7:
                    print("An official caught you!")
                    # 打印实际花费的时间
                    print(f"You took {int(time + random())} seconds.")
                    # 结束程序
                    return
                else:
                    # 如果随机数大于等于0.7，表示成功通过作弊检测
                    print("You made it!")
                    # 增加1.5秒的时间
                    time += 1.5
            else:  # 如果没有匹配的选项
                match opt:  # 使用匹配表达式
                    case 1:  # 如果 opt 的值为 1
                        speed += int(random() * (10 - 5) + 5)  # 将速度增加一个随机值，范围在 5 到 10 之间

                    case 2:  # 如果 opt 的值为 2
                        speed += int(random() * (5 - 3) + 3)  # 将速度增加一个随机值，范围在 3 到 5 之间

                    case 3:  # 如果 opt 的值为 3
                        speed += int(random() * (4 - 1) + 1)  # 将速度增加一个随机值，范围在 1 到 4 之间

                    case 5:  # 如果 opt 的值为 5
                        speed -= int(random() * (4 - 1) + 1)  # 将速度减少一个随机值，范围在 1 到 4 之间

                    case 6:  # 如果 opt 的值为 6
                        speed -= int(random() * (5 - 3) + 3)  # 将速度减少一个随机值，范围在 3 到 5 之间

                    case 7:  # 如果 opt 的值为 7
                        speed -= int(random() * (10 - 5) + 5)  # 将速度减少一个随机值，范围在 5 到 10 之间
                print(f" {int(speed)} M.P.H.")  # 打印最终的速度值
                if speed > max_speeds[i]:  # 如果速度超过最大速度
                    if random() < ((speed - max_speeds[i]) * 0.1) + 0.2:  # 如果随机数小于超速的概率
                        print(
                            f"You went over the maximum speed and {'snagged a flag' if random() < .5 else 'wiped out'}!"
                        )  # 打印超速并根据随机数打印相应信息
                        print(f"You took {int(time + random())} seconds")  # 打印花费的时间
                        return  # 结束函数
                    else:  # 如果随机数大于超速的概率
                        print("You went over the maximum speed and made it!")  # 打印超速但成功通过
                if speed > max_speeds[i] - 1:  # 如果速度接近最大速度
                    print("Close one!")  # 打印接近最大速度
            if speed < 7:  # 如果速度小于7
                print("Let's be realistic, ok? Let's go back and try again...")  # 打印速度过慢的提示
                speed = old_speed  # 将速度恢复到之前的值
            else:  # 如果速度在合理范围内
                time += max_speeds[i] - speed + 1  # 增加花费的时间
                if speed > max_speeds[i]:  # 如果速度超过最大速度
                    time += 0.5  # 增加额外的时间
                break  # 结束循环
    print(f"\nYou took {int(time + random())} seconds.")  # 打印最终花费的时间
    avg = time / gates  # 计算平均速度
    if avg < 1.5 - (lvl * 0.1):  # 如果平均速度小于特定值，获得金牌
        print("Yout won a gold medal!")
        medals["gold"] += 1
    elif avg < 2.9 - (lvl * 0.1):  # 如果平均速度小于另一个特定值，获得银牌
        print("You won a silver medal!")
        medals["silver"] += 1
    elif avg < 4.4 - (lvl * 0.01):  # 如果平均速度小于另一个特定值，获得铜牌
        print("You won a bronze medal!")
        medals["bronze"] += 1


def main() -> None:
    print("Slalom".rjust(39))  # 在屏幕上右对齐打印"Slalom"
    print("Creative Computing Morristown, New Jersey\n\n\n".rjust(57))  # 在屏幕上右对齐打印"Creative Computing Morristown, New Jersey\n\n\n"

    max_speeds = [  # 创建一个包含最大速度的列表
        14,
        18,
        26,
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，封装成字节流
    使用字节流里面内容创建 ZIP 对象  # 使用字节流内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 创建 ZIP 对象
    遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 以文件名为键，文件数据为值，创建文件名到数据的字典
    # 关闭 ZIP 对象  # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典  # 返回文件名到数据的字典
    return fdict
        31,  # 定义一个整数列表，包含31和22两个整数
        22,
    ]

    while True:  # 创建一个无限循环
        gates = ask_int("How many gates does this course have (1 to 25)")  # 询问用户此赛道有多少个门，将用户输入的整数赋值给变量gates
        if gates < 1:  # 如果gates小于1
            print("Try again,")  # 打印提示信息
        else:  # 否则
            if gates > 25:  # 如果gates大于25
                print("25 is the limit.")  # 打印提示信息
            break  # 结束循环

    pre_run(gates, max_speeds)  # 调用pre_run函数，传入gates和max_speeds作为参数

    while True:  # 创建一个无限循环
        lvl = ask_int("Rate yourself as a skier, (1=Worst, 3=Best)")  # 询问用户对自己滑雪水平的评价，将用户输入的整数赋值给变量lvl
        if lvl < 1 or lvl > 3:  # 如果lvl小于1或者大于3
            print("The bounds are 1-3.")  # 打印提示信息
        else:  # 否则
            break  # 结束当前循环，跳出循环体

    while True:  # 进入无限循环
        run(gates, lvl, max_speeds)  # 调用run函数，传入参数gates, lvl, max_speeds
        while True:  # 进入内部无限循环
            answer = ask("Do you want to play again?")  # 调用ask函数，询问用户是否想再玩一次，将结果赋给answer变量
            if answer == "YES" or answer == "NO":  # 判断answer是否为"YES"或"NO"
                break  # 如果是，则跳出内部循环
            else:  # 如果answer不是"YES"或"NO"
                print('Please type "YES" or "NO"')  # 打印提示信息
        if answer == "NO":  # 判断answer是否为"NO"
            break  # 如果是，则跳出外部循环

    print("Thanks for the race")  # 打印感谢信息
    if medals["gold"] > 0:  # 判断gold奖牌数量是否大于0
        print(f"Gold medals: {medals['gold']}")  # 如果是，则打印金牌数量
    if medals["silver"] > 0:  # 判断silver奖牌数量是否大于0
        print(f"Silver medals: {medals['silver']}")  # 如果是，则打印银牌数量
    if medals["bronze"] > 0:  # 判断bronze奖牌数量是否大于0
        print(f"Bronze medals: {medals['bronze']}")  # 如果是，则打印铜牌数量
# 如果当前脚本被直接执行，则执行 main() 函数
if __name__ == "__main__":
    main()
```

这段代码用于判断当前脚本是否被直接执行，如果是，则调用 main() 函数。这是一种常见的编程习惯，可以让脚本既可以作为独立的程序运行，也可以作为模块被其他程序引用。
```