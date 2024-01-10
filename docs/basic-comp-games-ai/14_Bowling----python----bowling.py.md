# `basic-computer-games\14_Bowling\python\bowling.py`

```
# 导入 random 模块
import random
# 导入 List 类型
from typing import List

# 模拟投球
def simulate_roll(pins: List[int]) -> None:
    # 进行 20 次模拟投球
    for _ in range(20):
        # 随机生成 0 到 14 之间的整数
        x = random.randint(0, 14)
        # 如果生成的随机数小于针数的长度
        if x < len(pins):
            # 将对应位置的针数设为 1
            pins[x] = 1

# 计算得分
def calculate_score(rolls: List[int]) -> int:
    # 初始化得分和帧数
    score = 0
    frame = 1
    b = 1
    # 遍历投球结果
    for index, pins in enumerate(rolls):
        # 累加得分
        score += pins
        # 如果是本帧的第一次投球
        if b == 1:
            # 如果击倒全部 10 根球，即全中
            if pins == 10:  # strike
                # 累加额外得分，下两次投球的得分
                score += sum(rolls[index + 1 : index + 3])
                # 进入下一帧
                frame += 1
            else:
                # 否则，进入本帧的第二次投球
                b = 2
        else:
            # 如果本帧的两次投球共击倒 10 根球，即补中
            if sum(rolls[index - 1 : index + 1]) == 10:  # spare
                # 累加额外得分，下一次投球的得分
                score += rolls[index + 1]
            # 重置为本帧的第一次投球
            b = 1
            # 进入下一帧
            frame += 1
        # 如果已经进行了 10 帧
        if frame > 10:
            # 结束计算
            break
    # 返回总得分
    return score

# 定义玩家类
class Player:
    # 初始化方法
    def __init__(self, name: str) -> None:
        # 设置玩家姓名
        self.name = name
        # 初始化投球结果列表
        self.rolls: List[int] = []
    # 播放当前帧的保龄球比赛，接受一个整数参数表示帧数，没有返回值
    def play_frame(self, frame: int) -> None:
        extra = 0  # 初始化额外投球次数为0
        prev_score = 0  # 初始化上一次得分为0
        pins = [0] * 10  # 重置保龄球瓶的状态为未击倒
        for ball in range(2):  # 对于每一轮投球
            simulate_roll(pins)  # 模拟投球，更新瓶的状态
            score = sum(pins)  # 计算当前得分
            self.show(pins)  # 显示当前瓶的状态
            pin_count = score - prev_score  # 计算本次击倒的瓶数
            self.rolls.append(pin_count)  # 记录本次击倒的瓶数
            print(f"{pin_count} for {self.name}")  # 打印本次得分
            if score - prev_score == 0:  # 如果本次得分为0
                print("GUTTER!!!")  # 打印“沟道球”
            if ball == 0:  # 如果是第一次投球
                if score == 10:  # 如果本次得分为10
                    print("STRIKE!!!")  # 打印“全中”
                    extra = 2  # 额外投球次数为2
                    break  # 不能在一帧内投球超过一次
                else:  # 如果不是全中
                    print(f"next roll {self.name}")  # 打印“下一次投球”
            else:  # 如果是第二次投球
                if score == 10:  # 如果本次得分为10
                    print("SPARE!")  # 打印“补中”
                    extra = 1  # 额外投球次数为1
            prev_score = score  # 更新上一次得分为当前得分，用于区分...

        if frame == 9 and extra > 0:  # 如果是第9帧且有额外投球次数
            print(f"Extra rolls for {self.name}")  # 打印“额外投球次数”
            pins = [0] * 10  # 重置保龄球瓶的状态为未击倒
            score = 0  # 初始化得分为0
            for _ball in range(extra):  # 对于每一次额外投球
                if score == 10:  # 如果得分为10
                    pins = [0] * 10  # 重置保龄球瓶的状态为未击倒
                simulate_roll(pins)  # 模拟投球，更新瓶的状态
                score = sum(pins)  # 计算当前得分
                self.rolls.append(score)  # 记录当前得分

    # 返回对象的字符串表示形式
    def __str__(self) -> str:
        return f"{self.name}: {self.rolls}, total:{calculate_score(self.rolls)}"

    # 显示保龄球瓶的状态
    def show(self, pins: List[int]) -> None:
        pins_iter = iter(pins)  # 创建瓶状态的迭代器
        print()  # 打印空行
        for row in range(4):  # 对于每一行
            print(" " * row, end="")  # 打印空格
            for _ in range(4 - row):  # 对于每一个瓶
                p = next(pins_iter)  # 获取下一个瓶的状态
                print("O " if p else "+ ", end="")  # 如果瓶倒了打印“O”，否则打印“+”
            print()  # 换行
# 定义一个函数，用于将文本居中显示在指定宽度的字符串中
def centre_text(text: str, width: int) -> str:
    # 计算文本的长度
    t = len(text)
    # 返回一个空格填充的字符串，使得文本居中显示在指定宽度的字符串中
    return (" " * ((width - t) // 2)) + text


# 定义主函数
def main() -> None:
    # 打印居中显示的文本
    print(centre_text("Bowl", 80))
    print(centre_text("CREATIVE COMPUTING MORRISTOWN, NEW JERSEY", 80))
    print()
    print("WELCOME TO THE ALLEY.")
    print("BRING YOUR FRIENDS.")
    print("OKAY LET'S FIRST GET ACQUAINTED.")

    # 进入循环，直到用户选择退出
    while True:
        print()
        # 如果用户输入"Y"或"y"，则打印游戏说明
        if input("THE INSTRUCTIONS (Y/N)? ") in "yY":
            print("THE GAME OF BOWLING TAKES MIND AND SKILL. DURING THE GAME")
            print("THE COMPUTER WILL KEEP SCORE. YOU MAY COMPETE WITH")
            print("OTHER PLAYERS[UP TO FOUR]. YOU WILL BE PLAYING TEN FRAMES.")
            print("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE")
            print("PIN IS STANDING. AFTER THE GAME THE COMPUTER WILL SHOW YOUR")
            print("SCORES.")

        # 获取玩家数量
        total_players = int(input("FIRST OF ALL...HOW MANY ARE PLAYING? "))
        player_names = []
        print()
        print("VERY GOOD...")
        # 为每个玩家输入姓名，并存储在列表中
        for index in range(total_players):
            player_names.append(Player(input(f"Enter name for player {index + 1}: ")))

        # 为每个玩家进行十轮比赛
        for frame in range(10):
            for player in player_names:
                player.play_frame(frame)

        # 打印每个玩家的得分
        for player in player_names:
            print(player)

        # 如果用户不想再玩一局，则退出循环
        if input("DO YOU WANT ANOTHER GAME? ") not in "yY":
            break


# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()


############################################################################################
#
# This is a fairly straight conversion to python with some exceptions.
# I have kept most of the upper case text that the program prints.
# I have added the feature of giving names to players.
# I have added a Player class to store player data in.
# This last change works around the problems in the original storing data in a matrix.
# The original had bugs in calculating indexes which meant that the program
# 会覆盖矩阵中的数据，因此打印出的结果包含错误。
# 最后的更改涉及到严格的规则，允许玩家在最后一格得到额外的投掷机会，如果玩家在最后一格得到了补中或全中。
# 该程序允许这些额外的投掷，并计算正确的得分。
#
############################################################################################
```