# `basic-computer-games\07_Basketball\python\basketball.py`

```py
"""
The basketball class is a computer game that allows you to play as
Dartmouth College's captain and playmaker
The game uses set probabilites to simulate outcomes of each posession
You are able to choose your shot types as well as defensive formations
"""

import random
from typing import List, Literal, Optional


def print_intro() -> None:
    # 打印游戏介绍
    print("\t\t\t Basketball")
    print("\t Creative Computing  Morristown, New Jersey\n\n\n")
    print("This is Dartmouth College basketball. ")
    print("Υou will be Dartmouth captain and playmaker.")
    print("Call shots as follows:")
    print(
        "1. Long (30ft.) Jump Shot; "
        "2. Short (15 ft.) Jump Shot; "
        "3. Lay up; 4. Set Shot"
    )
    print("Both teams will use the same defense. Call Defense as follows:")
    print("6. Press; 6.5 Man-to-Man; 7. Zone; 7.5 None.")
    print("To change defense, just type 0 as your next shot.")
    print("Your starting defense will be? ", end="")


class Basketball:
    def __init__(self) -> None:
        # 初始化游戏状态
        self.time = 0
        self.score = [0, 0]  # first value is opponents score, second is home
        self.defense_choices: List[float] = [6, 6.5, 7, 7.5]
        self.shot: Optional[int] = None
        self.shot_choices: List[Literal[0, 1, 2, 3, 4]] = [0, 1, 2, 3, 4]
        self.z1: Optional[float] = None

        # 打印游戏介绍
        print_intro()

        # 获取初始防守选择
        self.defense = get_defense_choice(self.defense_choices)

        # 获取对手名称并开始新的比赛周期
        self.opponent = get_opponents_name()
        self.start_of_period()

    def add_points(self, team: Literal[0, 1], points: Literal[0, 1, 2]) -> None:
        """
        Add points to the score.

        Team can take 0 or 1, for opponent or Dartmouth, respectively
        """
        # 给指定队伍加分
        self.score[team] += points
        self.print_score()

    def ball_passed_back(self) -> None:
        # 打印信息，球传回给你
        print("Ball passed back to you. ", end="")
        self.dartmouth_ball()
    # 改变防守方式，当用户输入0时调用
    def change_defense(self) -> None:
        """change defense, called when the user enters 0 for their shot"""
        # 初始化防守方式为None
        defense = None

        # 循环直到用户输入有效的防守方式
        while defense not in self.defense_choices:
            print("Your new defensive allignment is? ")
            try:
                # 尝试获取用户输入的防守方式，如果输入不是浮点数则继续循环
                defense = float(input())
            except ValueError:
                continue
        # 确保defense是浮点数类型
        assert isinstance(defense, float)
        # 将防守方式赋值给self.defense
        self.defense = defense
        # 调用dartmouth_ball方法
        self.dartmouth_ball()

    # 模拟球员两次罚球并加分
    def foul_shots(self, team: Literal[0, 1]) -> None:
        """Simulate two foul shots for a player and adds the points."""
        print("Shooter fouled.  Two shots.")
        if random.random() > 0.49:
            if random.random() > 0.75:
                print("Both shots missed.")
            else:
                print("Shooter makes one shot and misses one.")
                self.score[team] += 1
        else:
            print("Shooter makes both shots.")
            self.score[team] += 2

        # 打印当前比分
        self.print_score()

    # 中场休息，t = 50时调用，开始新的比赛阶段
    def halftime(self) -> None:
        """called when t = 50, starts a new period"""
        print("\n   ***** End of first half *****\n")
        # 打印当前比分
        self.print_score()
        # 调用start_of_period方法
        self.start_of_period()

    # 打印当前比分
    def print_score(self) -> None:
        """Print the current score"""
        print(f"Score:  {self.score[1]} to {self.score[0]}\n")

    # 开始新的比赛阶段，模拟中场跳球
    def start_of_period(self) -> None:
        """Simulate a center jump for posession at the beginning of a period"""
        print("Center jump")
        if random.random() > 0.6:
            print("Dartmouth controls the tap.\n")
            # 调用dartmouth_ball方法
            self.dartmouth_ball()
        else:
            print(self.opponent + " controls the tap.\n")
            # 调用opponent_ball方法
            self.opponent_ball()

    # 两分钟警告，t = 92时调用
    def two_minute_warning(self) -> None:
        """called when t = 92"""
        print("   *** Two minutes left in the game ***")
    # 定义达特茅斯非跳投方法，没有返回值
    def dartmouth_non_jump_shot(self) -> None:
        """
        Lay up, set shot, or defense change

        called when the user enters 0, 3, or 4
        """
        # 时间加一
        self.time += 1
        # 如果时间等于50，执行中场休息方法
        if self.time == 50:
            self.halftime()
        # 如果时间等于92，执行两分钟警告方法
        elif self.time == 92:
            self.two_minute_warning()

        # 如果投篮方式是4，打印"Set shot."
        if self.shot == 4:
            print("Set shot.")
        # 如果投篮方式是3，打印"Lay up."
        elif self.shot == 3:
            print("Lay up.")
        # 如果投篮方式是0，执行改变防守方法
        elif self.shot == 0:
            self.change_defense()

        # 模拟上篮或投篮后的不同结果
        if 7 / self.defense * random.random() > 0.4:
            if 7 / self.defense * random.random() > 0.7:
                if 7 / self.defense * random.random() > 0.875:
                    if 7 / self.defense * random.random() > 0.925:
                        # 打印"Charging foul. Dartmouth loses the ball."
                        print("Charging foul. Dartmouth loses the ball.\n")
                        # 对手控球
                        self.opponent_ball()
                    else:
                        # 打印"Shot blocked. 对手's ball."
                        print("Shot blocked. " + self.opponent + "'s ball.\n")
                        # 对手控球
                        self.opponent_ball()
                else:
                    # 罚球
                    self.foul_shots(1)
                    # 对手控球
                    self.opponent_ball()
            else:
                # 打印"Shot is off the rim."
                print("Shot is off the rim.")
                if random.random() > 2 / 3:
                    # 打印"Dartmouth controls the rebound."
                    print("Dartmouth controls the rebound.")
                    if random.random() > 0.4:
                        # 打印"Ball passed back to you."
                        print("Ball passed back to you.\n")
                        # 达特茅斯控球
                        self.dartmouth_ball()
                    else:
                        # 递归调用达特茅斯非跳投方法
                        self.dartmouth_non_jump_shot()
                else:
                    # 打印对手控球
                    print(self.opponent + " controls the rebound.\n")
                    # 对手控球
                    self.opponent_ball()
        else:
            # 打印"Shot is good. Two points."
            print("Shot is good. Two points.")
            # 加分
            self.add_points(1, 2)
            # 对手控球
            self.opponent_ball()
    def dartmouth_ball(self) -> None:
        """plays out a Dartmouth posession, starting with your choice of shot"""
        # 从给定的射击选择中获取一个射击动作
        shot = get_dartmouth_ball_choice(self.shot_choices)
        # 将获取的射击动作赋给当前对象的射击属性
        self.shot = shot

        # 如果比赛时间小于100或者随机数小于0.5
        if self.time < 100 or random.random() < 0.5:
            # 如果射击动作为1或2，则进行 Dartmouth 的跳投
            if self.shot == 1 or self.shot == 2:
                self.dartmouth_jump_shot()
            # 否则进行 Dartmouth 的非跳投
            else:
                self.dartmouth_non_jump_shot()
        # 如果比赛时间大于等于100且随机数大于等于0.5
        else:
            # 如果 Dartmouth 的得分不等于对手的得分
            if self.score[0] != self.score[1]:
                # 打印比赛结束信息和最终比分
                print("\n   ***** End Of Game *****")
                print(
                    "Final Score: Dartmouth: "
                    + str(self.score[1])
                    + "  "
                    + self.opponent
                    + ": "
                    + str(self.score[0])
                )
            # 如果 Dartmouth 的得分等于对手的得分
            else:
                # 打印比赛结束信息和比赛结束时的比分
                print("\n   ***** End Of Second Half *****")
                print("Score at end of regulation time:")
                print(
                    "     Dartmouth: "
                    + str(self.score[1])
                    + " "
                    + self.opponent
                    + ": "
                    + str(self.score[0])
                )
                print("Begin two minute overtime period")
                # 设置比赛时间为93，并开始新的比赛周期
                self.time = 93
                self.start_of_period()
    def opponent_jumpshot(self) -> None:
        """模拟对手的跳投"""
        # 打印"Jump Shot."
        print("Jump Shot.")
        # 如果 8 除以自身防守值乘以一个随机数大于 0.35
        if 8 / self.defense * random.random() > 0.35:
            # 如果 8 除以自身防守值乘以一个随机数大于 0.75
            if 8 / self.defense * random.random() > 0.75:
                # 如果 8 除以自身防守值乘以一个随机数大于 0.9
                if 8 / self.defense * random.random() > 0.9:
                    # 打印"Offensive foul. Dartmouth's ball."
                    print("Offensive foul. Dartmouth's ball.\n")
                    # 调用 self.dartmouth_ball() 方法
                    self.dartmouth_ball()
                else:
                    # 调用 self.foul_shots(0) 方法
                    self.foul_shots(0)
                    # 调用 self.dartmouth_ball() 方法
                    self.dartmouth_ball()
            else:
                # 打印"Shot is off the rim."
                print("Shot is off the rim.")
                # 如果自身防守值除以 6 乘以一个随机数大于 0.5
                if self.defense / 6 * random.random() > 0.5:
                    # 打印对手名称 + " controls the rebound."
                    print(self.opponent + " controls the rebound.")
                    # 如果自身防守值为 6
                    if self.defense == 6:
                        # 如果随机数大于 0.75
                        if random.random() > 0.75:
                            # 打印"Ball stolen. Easy lay up for Dartmouth."
                            print("Ball stolen. Easy lay up for Dartmouth.")
                            # 调用 self.add_points(1, 2) 方法
                            self.add_points(1, 2)
                            # 调用 self.opponent_ball() 方法
                            self.opponent_ball()
                        else:
                            # 如果随机数大于 0.5
                            if random.random() > 0.5:
                                # 调用 self.opponent_non_jumpshot() 方法
                                self.opponent_non_jumpshot()
                            else:
                                # 打印"Pass back to " + self.opponent + " guard.\n"
                                print("Pass back to " + self.opponent + " guard.\n")
                                # 调用 self.opponent_ball() 方法
                                self.opponent_ball()
                    else:
                        # 如果随机数大于 0.5
                        if random.random() > 0.5:
                            # 调用 self.opponent_non_jumpshot() 方法
                            self.opponent_non_jumpshot()
                        else:
                            # 打印"Pass back to " + self.opponent + " guard.\n"
                            print("Pass back to " + self.opponent + " guard.\n")
                            # 调用 self.opponent_ball() 方法
                            self.opponent_ball()
                else:
                    # 打印"Dartmouth controls the rebound.\n"
                    print("Dartmouth controls the rebound.\n")
                    # 调用 self.dartmouth_ball() 方法
                    self.dartmouth_ball()
        else:
            # 打印"Shot is good."
            print("Shot is good.")
            # 调用 self.add_points(0, 2) 方法
            self.add_points(0, 2)
            # 调用 self.dartmouth_ball() 方法
            self.dartmouth_ball()
    def opponent_non_jumpshot(self) -> None:
        """模拟对手的上篮或定点投篮。"""
        if self.z1 > 3:  # type: ignore
            print("定点投篮。")
        else:
            print("上篮")
        if 7 / self.defense * random.random() > 0.413:
            print("投篮不中。")
            if self.defense / 6 * random.random() > 0.5:
                print(self.opponent + " 控制篮板。")
                if self.defense == 6:
                    if random.random() > 0.75:
                        print("球被抢断。达特茅斯轻松上篮得分。")
                        self.add_points(1, 2)
                        self.opponent_ball()
                    else:
                        if random.random() > 0.5:
                            print()
                            self.opponent_non_jumpshot()
                        else:
                            print("传球回给 " + self.opponent + " 后卫。\n")
                            self.opponent_ball()
                else:
                    if random.random() > 0.5:
                        print()
                        self.opponent_non_jumpshot()
                    else:
                        print("传球回给 " + self.opponent + " 后卫。\n")
                        self.opponent_ball()
            else:
                print("达特茅斯控制篮板。\n")
                self.dartmouth_ball()
        else:
            print("投篮命中。")
            self.add_points(0, 2)
            self.dartmouth_ball()

    def opponent_ball(self) -> None:
        """
        模拟对手的进攻

        随机选择跳投或上篮/定点投篮。
        """
        self.time += 1
        if self.time == 50:
            self.halftime()
        self.z1 = 10 / 4 * random.random() + 1
        if self.z1 > 2:
            self.opponent_non_jumpshot()
        else:
            self.opponent_jumpshot()
# 从给定的防守选择列表中获取用户输入的防守选择
def get_defense_choice(defense_choices: List[float]) -> float:
    """获取防守选择的输入"""
    try:
        defense = float(input())  # 尝试将输入转换为浮点数
    except ValueError:
        defense = None  # 如果输入不是有效的浮点数，则将防守选择设为 None

    # 如果输入不是有效的防守选择，则重新获取输入
    while defense not in defense_choices:
        print("Your new defensive allignment is? ", end="")
        try:
            defense = float(input())  # 尝试将输入转换为浮点数
        except ValueError:
            continue  # 如果输入不是有效的浮点数，则继续循环
    assert isinstance(defense, float)  # 断言防守选择是浮点数类型
    return defense  # 返回防守选择


# 从给定的投篮选择列表中获取用户输入的投篮选择
def get_dartmouth_ball_choice(shot_choices: List[Literal[0, 1, 2, 3, 4]]) -> int:
    print("Your shot? ", end="")
    shot = None
    try:
        shot = int(input())  # 尝试将输入转换为整数
    except ValueError:
        shot = None  # 如果输入不是有效的整数，则将投篮选择设为 None

    while shot not in shot_choices:
        print("Incorrect answer. Retype it. Your shot? ", end="")
        try:
            shot = int(input())  # 尝试将输入转换为整数
        except Exception:
            continue  # 如果输入不是有效的整数，则继续循环
    assert isinstance(shot, int)  # 断言投篮选择是整数类型
    return shot  # 返回投篮选择


# 获取用户输入的对手名称
def get_opponents_name() -> str:
    """获取对手的名称输入"""
    print("\nChoose your opponent? ", end="")
    return input()  # 返回用户输入的对手名称


if __name__ == "__main__":
    Basketball()  # 调用 Basketball() 函数
```