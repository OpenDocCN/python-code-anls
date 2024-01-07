# `basic-computer-games\14_Bowling\python\bowling.py`

```

# 导入random模块和List类型
import random
from typing import List

# 模拟投球，将击倒的瓶子数记录在列表中
def simulate_roll(pins: List[int]) -> None:
    for _ in range(20):
        x = random.randint(0, 14)
        if x < len(pins):
            pins[x] = 1

# 计算得分
def calculate_score(rolls: List[int]) -> int:
    score = 0
    frame = 1
    b = 1
    for index, pins in enumerate(rolls):
        score += pins
        if b == 1:
            if pins == 10:  # 如果第一次就全中（strike）
                score += sum(rolls[index + 1 : index + 3])  # 加上接下来两次的得分
                frame += 1
            else:
                b = 2
        else:
            if sum(rolls[index - 1 : index + 1]) == 10:  # 如果两次投球的总分为10（spare）
                score += rolls[index + 1]  # 加上下一次的得分
            b = 1
            frame += 1
        if frame > 10:  # 如果已经进行了10个frame，结束循环
            break

    return score

# 球员类
class Player:
    def __init__(self, name: str) -> None:
        self.name = name
        self.rolls: List[int] = []

    # 进行一轮投球
    def play_frame(self, frame: int) -> None:
        extra = 0
        prev_score = 0
        pins = [0] * 10  # 重置瓶子
        for ball in range(2):
            simulate_roll(pins)  # 模拟投球
            score = sum(pins)
            self.show(pins)  # 展示击倒的瓶子情况
            pin_count = score - prev_score
            self.rolls.append(pin_count)  # 记录击倒的瓶子数
            print(f"{pin_count} for {self.name}")  # 打印本次得分
            if score - prev_score == 0:
                print("GUTTER!!!")  # 如果本次得分为0，打印“GUTTER!!!”
            if ball == 0:
                if score == 10:
                    print("STRIKE!!!")  # 如果第一次就全中，打印“STRIKE!!!”
                    extra = 2
                    break  # 不能在一轮中投两次
                else:
                    print(f"next roll {self.name}")  # 否则，提示下一次投球
            else:
                if score == 10:
                    print("SPARE!")  # 如果两次投球总分为10，打印“SPARE!”
                    extra = 1

            prev_score = score  # 记录上一次的得分

        if frame == 9 and extra > 0:  # 如果是第10轮且有额外投球
            print(f"Extra rolls for {self.name}")
            pins = [0] * 10  # 重置瓶子
            score = 0
            for _ball in range(extra):
                if score == 10:
                    pins = [0] * 10
                simulate_roll(pins)  # 模拟投球
                score = sum(pins)
                self.rolls.append(score)  # 记录得分

    def __str__(self) -> str:
        return f"{self.name}: {self.rolls}, total:{calculate_score(self.rolls)}"  # 打印球员名字、投球记录和总得分

    # 展示击倒的瓶子情况
    def show(self, pins: List[int]) -> None:
        pins_iter = iter(pins)
        print()
        for row in range(4):
            print(" " * row, end="")
            for _ in range(4 - row):
                p = next(pins_iter)
                print("O " if p else "+ ", end="")
            print()

# 居中文本
def centre_text(text: str, width: int) -> str:
    t = len(text)
    return (" " * ((width - t) // 2)) + text

# 主函数
def main() -> None:
    print(centre_text("Bowl", 80))  # 居中打印标题
    print(centre_text("CREATIVE COMPUTING MORRISTOWN, NEW JERSEY", 80))  # 居中打印副标题
    print()
    print("WELCOME TO THE ALLEY.")  # 打印欢迎词
    print("BRING YOUR FRIENDS.")  # 打印邀请词
    print("OKAY LET'S FIRST GET ACQUAINTED.")  # 打印提示

    while True:
        print()
        if input("THE INSTRUCTIONS (Y/N)? ") in "yY":  # 是否需要打印游戏说明
            print("THE GAME OF BOWLING TAKES MIND AND SKILL. DURING THE GAME")
            print("THE COMPUTER WILL KEEP SCORE. YOU MAY COMPETE WITH")
            print("OTHER PLAYERS[UP TO FOUR]. YOU WILL BE PLAYING TEN FRAMES.")
            print("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE")
            print("PIN IS STANDING. AFTER THE GAME THE COMPUTER WILL SHOW YOUR")
            print("SCORES.")

        total_players = int(input("FIRST OF ALL...HOW MANY ARE PLAYING? "))  # 输入玩家数量
        player_names = []
        print()
        print("VERY GOOD...")
        for index in range(total_players):
            player_names.append(Player(input(f"Enter name for player {index + 1}: ")))  # 输入玩家名字

        for frame in range(10):
            for player in player_names:
                player.play_frame(frame)  # 每个玩家进行一轮投球

        for player in player_names:
            print(player)  # 打印每个玩家的得分

        if input("DO YOU WANT ANOTHER GAME? ") not in "yY":  # 是否再玩一局
            break

# 如果是主程序，执行main函数
if __name__ == "__main__":
    main()

```