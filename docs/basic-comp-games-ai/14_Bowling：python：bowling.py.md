# `d:/src/tocomm/basic-computer-games\14_Bowling\python\bowling.py`

```
import random  # 导入 random 模块，用于生成随机数
from typing import List  # 从 typing 模块中导入 List 类型，用于声明参数和返回值的类型


def simulate_roll(pins: List[int]) -> None:
    for _ in range(20):  # 循环20次
        x = random.randint(0, 14)  # 生成一个0到14之间的随机整数
        if x < len(pins):  # 如果随机数小于 pins 列表的长度
            pins[x] = 1  # 将 pins 列表中对应位置的值设为1


def calculate_score(rolls: List[int]) -> int:
    score = 0  # 初始化得分为0
    frame = 1  # 初始化帧数为1
    b = 1  # 初始化 b 为1
    for index, pins in enumerate(rolls):  # 遍历 rolls 列表，同时获取索引和值
        score += pins  # 将当前击倒的瓶数加到总得分上
        if b == 1:  # 如果 b 等于1
            if pins == 10:  # 如果当前击倒的瓶数为10（全中）
                score += sum(rolls[index + 1 : index + 3])  # 在总得分上加上下两次投球的得分
                frame += 1  # 帧数加一
            else:
                b = 2  # 如果不是strike也不是spare，则b等于2
        else:
            if sum(rolls[index - 1 : index + 1]) == 10:  # 如果前一次和当前投掷的球的点数之和等于10，即为spare
                score += rolls[index + 1]  # 分数加上下一次投掷的球的点数
            b = 1  # 如果不是strike，则b等于1
            frame += 1  # 帧数加一
        if frame > 10:  # 如果帧数大于10，则跳出循环
            break

    return score  # 返回分数


class Player:
    def __init__(self, name: str) -> None:
        self.name = name  # 初始化玩家的名字
        self.rolls: List[int] = []  # 初始化玩家的投掷球的点数列表

    def play_frame(self, frame: int) -> None:  # 玩家进行一轮投掷
        extra = 0  # 初始化额外得分为0
        prev_score = 0  # 初始化上一次得分为0
        pins = [0] * 10  # 重置保龄球瓶的状态
        for ball in range(2):  # 循环两次，模拟两次投球
            simulate_roll(pins)  # 模拟投球，更新瓶的状态
            score = sum(pins)  # 计算本次得分
            self.show(pins)  # 展示瓶的状态
            pin_count = score - prev_score  # 计算本次击倒的瓶数
            self.rolls.append(pin_count)  # 记录本次击倒的瓶数
            print(f"{pin_count} for {self.name}")  # 打印本次得分
            if score - prev_score == 0:  # 如果本次得分和上次得分相同
                print("GUTTER!!!")  # 打印“零分！”
            if ball == 0:  # 如果是第一次投球
                if score == 10:  # 如果本次得分为10
                    print("STRIKE!!!")  # 打印“全中！”
                    extra = 2  # 额外得分为2
                    break  # 退出循环，因为全中后不能再投球
                else:  # 如果本次得分不为10
                    print(f"next roll {self.name}")  # 打印“下一次投球”
            else:  # 如果是第二次投球
                if score == 10:  # 如果得分为10
                    print("SPARE!")  # 打印“SPARE！”
                    extra = 1  # 将额外投球次数设为1

            prev_score = score  # 记住前一次击倒的瓶数，以区分...

        if frame == 9 and extra > 0:  # 如果是第9轮且有额外投球次数
            print(f"Extra rolls for {self.name}")  # 打印额外投球次数的信息
            pins = [0] * 10  # 重置瓶子的状态
            score = 0  # 重置得分
            for _ball in range(extra):  # 对于每次额外投球
                if score == 10:  # 如果得分为10
                    pins = [0] * 10  # 重置瓶子的状态
                simulate_roll(pins)  # 模拟投球
                score = sum(pins)  # 计算得分
                self.rolls.append(score)  # 将得分添加到投球记录中

    def __str__(self) -> str:  # 定义对象的字符串表示形式
        return f"{self.name}: {self.rolls}, total:{calculate_score(self.rolls)}"  # 返回对象的名称、投球记录和总得分

    def show(self, pins: List[int]) -> None:  # 定义展示瓶子状态的方法
        pins_iter = iter(pins)  # 创建一个迭代器对象，用于遍历列表 pins
        print()  # 打印一个空行
        for row in range(4):  # 循环4次，每次代表一行
            print(" " * row, end="")  # 打印空格，数量为当前行数
            for _ in range(4 - row):  # 循环4-row次，每次代表当前行的列数
                p = next(pins_iter)  # 从迭代器中获取下一个元素
                print("O " if p else "+ ", end="")  # 如果 p 为真，则打印 "O "，否则打印 "+ "
            print()  # 打印一个换行符


def centre_text(text: str, width: int) -> str:
    t = len(text)  # 获取文本的长度
    return (" " * ((width - t) // 2)) + text  # 返回一个居中的文本，两边填充空格


def main() -> None:
    print(centre_text("Bowl", 80))  # 打印居中的文本 "Bowl"
    print(centre_text("CREATIVE COMPUTING MORRISTOWN, NEW JERSEY", 80))  # 打印居中的文本 "CREATIVE COMPUTING MORRISTOWN, NEW JERSEY"
    print()  # 打印一个空行
    print("WELCOME TO THE ALLEY.")  # 打印欢迎信息
    print("BRING YOUR FRIENDS.")  # 打印“BRING YOUR FRIENDS.”
    print("OKAY LET'S FIRST GET ACQUAINTED.")  # 打印“OKAY LET'S FIRST GET ACQUAINTED.”

    while True:  # 进入无限循环
        print()  # 打印空行
        if input("THE INSTRUCTIONS (Y/N)? ") in "yY":  # 如果用户输入的是'y'或'Y'
            print("THE GAME OF BOWLING TAKES MIND AND SKILL. DURING THE GAME")  # 打印游戏规则
            print("THE COMPUTER WILL KEEP SCORE. YOU MAY COMPETE WITH")  # 打印提示
            print("OTHER PLAYERS[UP TO FOUR]. YOU WILL BE PLAYING TEN FRAMES.")  # 打印提示
            print("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE")  # 打印提示
            print("PIN IS STANDING. AFTER THE GAME THE COMPUTER WILL SHOW YOUR")  # 打印提示
            print("SCORES.")  # 打印提示

        total_players = int(input("FIRST OF ALL...HOW MANY ARE PLAYING? "))  # 获取玩家总数
        player_names = []  # 创建空列表用于存储玩家姓名
        print()  # 打印空行
        print("VERY GOOD...")  # 打印“VERY GOOD...”
        for index in range(total_players):  # 遍历玩家总数
            player_names.append(Player(input(f"Enter name for player {index + 1}: ")))  # 获取每位玩家的姓名并添加到列表中
        for frame in range(10):
            # 遍历 10 个球局
            for player in player_names:
                # 遍历玩家列表，让每个玩家进行当前球局
                player.play_frame(frame)

        for player in player_names:
            # 打印每个玩家的得分
            print(player)

        # 询问用户是否想再玩一局
        if input("DO YOU WANT ANOTHER GAME? ") not in "yY":
            # 如果用户输入不是 y 或 Y，则跳出循环，结束游戏
            break


if __name__ == "__main__":
    # 调用主函数
    main()


############################################################################################
#
# This is a fairly straight conversion to python with some exceptions.
# I have kept most of the upper case text that the program prints.
# I have added the feature of giving names to players.
# 我添加了一个Player类来存储玩家数据。
# 最后一次更改解决了原始存储数据在矩阵中的问题。
# 原始代码在计算索引时存在错误，导致程序会覆盖矩阵中的数据，因此打印出的结果包含错误。
# 最后的更改涉及严格的规则，允许玩家在最后一轮得到额外的投球机会，如果玩家在最后一轮得到了补中或全中。
# 该程序允许这些额外的投球，并计算正确的得分。
```