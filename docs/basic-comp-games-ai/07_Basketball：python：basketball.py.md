# `d:/src/tocomm/basic-computer-games\07_Basketball\python\basketball.py`

```
# 导入 random 模块，用于生成随机数
import random
# 导入 List、Literal、Optional 类型提示
from typing import List, Literal, Optional

# 定义 print_intro 函数，无返回值
def print_intro() -> None:
    # 打印游戏标题
    print("\t\t\t Basketball")
    # 打印游戏信息
    print("\t Creative Computing  Morristown, New Jersey\n\n\n")
    print("This is Dartmouth College basketball. ")
    print("Υou will be Dartmouth captain and playmaker.")
    print("Call shots as follows:")
    # 打印可选的投篮方式
    print(
        "1. Long (30ft.) Jump Shot; "
        "2. Short (15 ft.) Jump Shot; "
        "3. Lay up; 4. Set Shot"  # 定义了投篮选项，3代表上篮，4代表定点投篮
    )
    print("Both teams will use the same defense. Call Defense as follows:")  # 打印消息，表示双方将使用相同的防守策略
    print("6. Press; 6.5 Man-to-Man; 7. Zone; 7.5 None.")  # 打印消息，列出防守选项
    print("To change defense, just type 0 as your next shot.")  # 打印消息，说明如何改变防守策略
    print("Your starting defense will be? ", end="")  # 打印消息，询问初始防守策略

class Basketball:
    def __init__(self) -> None:
        self.time = 0  # 初始化比赛时间
        self.score = [0, 0]  # 初始化比分，第一个值是对手的得分，第二个值是主队的得分
        self.defense_choices: List[float] = [6, 6.5, 7, 7.5]  # 初始化防守选项
        self.shot: Optional[int] = None  # 初始化投篮选择
        self.shot_choices: List[Literal[0, 1, 2, 3, 4]] = [0, 1, 2, 3, 4]  # 初始化投篮选项
        self.z1: Optional[float] = None  # 初始化变量z1

        print_intro()  # 调用打印介绍的函数

        self.defense = get_defense_choice(self.defense_choices)  # 获取防守选择
        self.opponent = get_opponents_name()  # 获取对手的名字
        self.start_of_period()  # 开始比赛周期

    def add_points(self, team: Literal[0, 1], points: Literal[0, 1, 2]) -> None:
        """
        Add points to the score.

        Team can take 0 or 1, for opponent or Dartmouth, respectively
        """
        self.score[team] += points  # 将得分加到对应队伍的分数上
        self.print_score()  # 打印当前比分

    def ball_passed_back(self) -> None:
        print("Ball passed back to you. ", end="")  # 打印信息，球传回给你
        self.dartmouth_ball()  # 调用处理达特茅斯队球权的方法

    def change_defense(self) -> None:
        """change defense, called when the user enters 0 for their shot"""
        defense = None  # 初始化防守变量为None
        while defense not in self.defense_choices:  # 当防守选择不在可选范围内时
            print("Your new defensive allignment is? ")  # 打印提示信息，要求输入新的防守选择
            try:  # 尝试执行以下代码
                defense = float(input())  # 将输入的值转换为浮点数并赋值给defense变量
            except ValueError:  # 如果出现数值错误
                continue  # 继续循环
        assert isinstance(defense, float)  # 断言defense是浮点数
        self.defense = defense  # 将defense赋值给对象的defense属性
        self.dartmouth_ball()  # 调用对象的dartmouth_ball方法

    def foul_shots(self, team: Literal[0, 1]) -> None:  # 定义foul_shots方法，参数为team，返回类型为None
        """Simulate two foul shots for a player and adds the points."""  # 方法的文档字符串
        print("Shooter fouled.  Two shots.")  # 打印提示信息
        if random.random() > 0.49:  # 如果随机数大于0.49
            if random.random() > 0.75:  # 如果随机数大于0.75
                print("Both shots missed.")  # 打印提示信息
            else:  # 否则
                print("Shooter makes one shot and misses one.")  # 打印提示信息
                self.score[team] += 1  # 将球队得分加1
        else:
            # 如果球员两次投篮都命中，则得分加2
            print("Shooter makes both shots.")
            self.score[team] += 2

        # 打印当前比分
        self.print_score()

    def halftime(self) -> None:
        """当 t = 50 时调用，开始新的半场"""
        print("\n   ***** End of first half *****\n")
        self.print_score()
        self.start_of_period()

    def print_score(self) -> None:
        """打印当前比分"""
        print(f"Score:  {self.score[1]} to {self.score[0]}\n")

    def start_of_period(self) -> None:
        """模拟每节开始时的跳球争夺"""
        print("Center jump")
        if random.random() > 0.6:
            print("Dartmouth controls the tap.\n")  # 打印输出“达特茅斯控制跳球。”
            self.dartmouth_ball()  # 调用类中的dartmouth_ball方法
        else:
            print(self.opponent + " controls the tap.\n")  # 打印输出对手名称 + “控制跳球。”
            self.opponent_ball()  # 调用类中的opponent_ball方法

    def two_minute_warning(self) -> None:
        """called when t = 92"""  # 当t = 92时调用此方法
        print("   *** Two minutes left in the game ***")  # 打印输出“***比赛还剩两分钟***”

    def dartmouth_jump_shot(self) -> None:
        """called when the user enters 1 or 2 for their shot"""  # 当用户输入1或2进行投篮时调用此方法
        self.time += 1  # 时间加1
        if self.time == 50:  # 如果时间等于50
            self.halftime()  # 调用类中的halftime方法
        elif self.time == 92:  # 如果时间等于92
            self.two_minute_warning()  # 调用类中的two_minute_warning方法
        print("Jump Shot.")  # 打印输出“跳投。”
        # 模拟不同可能结果的机会
        if random.random() > 0.341 * self.defense / 8:  # 如果随机数大于0.341 * self.defense / 8
            # 如果随机数大于0.682乘以自身防守能力的八分之一
            if random.random() > 0.682 * self.defense / 8:
                # 如果随机数大于0.782乘以自身防守能力的八分之一
                if random.random() > 0.782 * self.defense / 8:
                    # 如果随机数大于0.843乘以自身防守能力的八分之一
                    if random.random() > 0.843 * self.defense / 8:
                        # 打印信息并让对手控球
                        print("Charging foul. Dartmouth loses ball.\n")
                        self.opponent_ball()
                    else:
                        # 球员被犯规
                        self.foul_shots(1)
                        self.opponent_ball()
                else:
                    # 如果随机数大于0.5
                    if random.random() > 0.5:
                        # 打印信息并让对手控球
                        print(
                            "Shot is blocked. Ball controlled by "
                            + self.opponent
                            + ".\n"
                        )
                        self.opponent_ball()
                    else:
                        # 打印信息并让达特茅斯控球
                        print("Shot is blocked. Ball controlled by Dartmouth.")
                        self.dartmouth_ball()
            else:
                # 如果投篮偏离目标
                print("Shot is off target.")
                # 如果防守能力除以6乘以一个随机数大于0.45
                if self.defense / 6 * random.random() > 0.45:
                    # 球被反弹给对手
                    print("Rebound to " + self.opponent + "\n")
                    self.opponent_ball()
                else:
                    # 达特茅斯控制篮板
                    print("Dartmouth controls the rebound.")
                    # 如果随机数大于0.4
                    if random.random() > 0.4:
                        # 如果防守等级为6且随机数大于0.6
                        if self.defense == 6 and random.random() > 0.6:
                            # 球被对手抢断，容易上篮得分
                            print("Pass stolen by " + self.opponent + ", easy lay up")
                            self.add_points(0, 2)
                            self.dartmouth_ball()
                        else:
                            # 球被传回给你
                            self.ball_passed_back()
                    else:
                        # 达特茅斯非跳投
                        print()
                        self.dartmouth_non_jump_shot()
        else:
            # 投篮命中
            print("Shot is good.")
            self.add_points(1, 2)  # 调用 add_points 方法，传入参数 1 和 2
            self.opponent_ball()  # 调用 opponent_ball 方法

    def dartmouth_non_jump_shot(self) -> None:
        """
        Lay up, set shot, or defense change

        called when the user enters 0, 3, or 4
        """
        self.time += 1  # 时间加一
        if self.time == 50:  # 如果时间等于 50
            self.halftime()  # 调用 halftime 方法
        elif self.time == 92:  # 如果时间等于 92
            self.two_minute_warning()  # 调用 two_minute_warning 方法

        if self.shot == 4:  # 如果投篮方式为 4
            print("Set shot.")  # 打印 "Set shot."
        elif self.shot == 3:  # 如果投篮方式为 3
            print("Lay up.")  # 打印 "Lay up."
        elif self.shot == 0:  # 如果投篮方式为 0
            self.change_defense()  # 调用change_defense方法，改变防守状态

        # 模拟投篮或定点投篮后的不同结果
        if 7 / self.defense * random.random() > 0.4:  # 如果7除以防守值乘以一个随机数大于0.4
            if 7 / self.defense * random.random() > 0.7:  # 如果7除以防守值乘以一个随机数大于0.7
                if 7 / self.defense * random.random() > 0.875:  # 如果7除以防守值乘以一个随机数大于0.875
                    if 7 / self.defense * random.random() > 0.925:  # 如果7除以防守值乘以一个随机数大于0.925
                        print("Charging foul. Dartmouth loses the ball.\n")  # 打印信息，达特茅斯失去球权
                        self.opponent_ball()  # 调用opponent_ball方法，对手获得球权
                    else:
                        print("Shot blocked. " + self.opponent + "'s ball.\n")  # 打印信息，投篮被封堵，对手获得球权
                        self.opponent_ball()  # 调用opponent_ball方法，对手获得球权
                else:
                    self.foul_shots(1)  # 调用foul_shots方法，进行罚球
                    self.opponent_ball()  # 调用opponent_ball方法，对手获得球权
            else:
                print("Shot is off the rim.")  # 打印信息，投篮偏出篮筐
                if random.random() > 2 / 3:  # 如果随机数大于2/3
                    print("Dartmouth controls the rebound.")  # 打印信息，达特茅斯控制篮板
                    if random.random() > 0.4:  # 如果随机数大于0.4
                        print("Ball passed back to you.\n")  # 打印信息，将球传回给你
                        self.dartmouth_ball()  # 调用类中的方法，进行达特茅斯队的进攻
                    else:
                        self.dartmouth_non_jump_shot()  # 调用类中的方法，进行达特茅斯队的非跳投进攻
                else:
                    print(self.opponent + " controls the rebound.\n")  # 打印信息，对手控制篮板
                    self.opponent_ball()  # 调用类中的方法，对手发起进攻
        else:
            print("Shot is good. Two points.")  # 打印信息，投篮命中，得两分
            self.add_points(1, 2)  # 调用类中的方法，给自己队伍加分
            self.opponent_ball()  # 调用类中的方法，对手发起进攻

    def dartmouth_ball(self) -> None:
        """plays out a Dartmouth posession, starting with your choice of shot"""
        shot = get_dartmouth_ball_choice(self.shot_choices)  # 调用函数，获取达特茅斯队的投篮选择
        self.shot = shot  # 将投篮选择存储到类的属性中

        if self.time < 100 or random.random() < 0.5:  # 如果比赛时间小于100秒或者随机数小于0.5
            if self.shot == 1 or self.shot == 2:  # 如果投篮选择是1或2
                self.dartmouth_jump_shot()  # 调用类中的方法，进行达特茅斯队的跳投进攻
            else:  # 如果条件不满足，执行以下代码
                self.dartmouth_non_jump_shot()  # 调用类中的dartmouth_non_jump_shot方法
        else:  # 如果外层条件不满足，执行以下代码
            if self.score[0] != self.score[1]:  # 如果score列表中的第一个元素不等于第二个元素
                print("\n   ***** End Of Game *****")  # 打印结束比赛的提示
                print(
                    "Final Score: Dartmouth: "
                    + str(self.score[1])  # 打印Dartmouth队的最终得分
                    + "  "
                    + self.opponent  # 打印对手队伍的名称
                    + ": "
                    + str(self.score[0])  # 打印对手队伍的最终得分
                )
            else:  # 如果条件不满足，执行以下代码
                print("\n   ***** End Of Second Half *****")  # 打印上半场结束的提示
                print("Score at end of regulation time:")  # 打印常规时间结束时的比分
                print(
                    "     Dartmouth: "
                    + str(self.score[1])  # 打印Dartmouth队的得分
                    + " "  # 打印空格
                    + self.opponent  # 添加对手的得分
                    + ": "  # 添加冒号
                    + str(self.score[0])  # 将对手的得分转换为字符串
                )
                print("Begin two minute overtime period")  # 打印开始两分钟的加时赛
                self.time = 93  # 设置时间为93
                self.start_of_period()  # 调用开始新周期的方法

    def opponent_jumpshot(self) -> None:
        """Simulate the opponents jumpshot"""  # 模拟对手的跳投
        print("Jump Shot.")  # 打印跳投
        if 8 / self.defense * random.random() > 0.35:  # 如果对手的防守值除以8乘以一个随机数大于0.35
            if 8 / self.defense * random.random() > 0.75:  # 如果对手的防守值除以8乘以一个随机数大于0.75
                if 8 / self.defense * random.random() > 0.9:  # 如果对手的防守值除以8乘以一个随机数大于0.9
                    print("Offensive foul. Dartmouth's ball.\n")  # 打印进攻犯规，达特茅斯队的球
                    self.dartmouth_ball()  # 调用达特茅斯队的球方法
                else:
                    self.foul_shots(0)  # 调用罚球方法
                    self.dartmouth_ball()  # 调用达特茅斯队的球方法
            else:
                # 打印“投篮偏出”信息
                print("Shot is off the rim.")
                # 如果防守值除以6再乘以一个随机数大于0.5
                if self.defense / 6 * random.random() > 0.5:
                    # 打印对手控制篮板的信息
                    print(self.opponent + " controls the rebound.")
                    # 如果防守值为6
                    if self.defense == 6:
                        # 如果随机数大于0.75
                        if random.random() > 0.75:
                            # 打印“球被抢断，达特茅斯轻松上篮得分”的信息
                            print("Ball stolen. Easy lay up for Dartmouth.")
                            # 调用add_points函数，给自己加1分，对手加2分
                            self.add_points(1, 2)
                            # 对手控球
                            self.opponent_ball()
                        else:
                            # 如果随机数大于0.5
                            if random.random() > 0.5:
                                # 调用对手非跳投函数
                                self.opponent_non_jumpshot()
                            else:
                                # 打印“传球回给对手的后卫”的信息
                                print("Pass back to " + self.opponent + " guard.\n")
                                # 对手控球
                                self.opponent_ball()
                    else:
                        # 如果随机数大于0.5
                        if random.random() > 0.5:
                            # 调用对手非跳投函数
                            self.opponent_non_jumpshot()
                        else:
                            # 打印“传球回给对手的后卫”的信息
                            print("Pass back to " + self.opponent + " guard.\n")
                            self.opponent_ball()  # 调用opponent_ball方法，表示对手控球
                else:
                    print("Dartmouth controls the rebound.\n")  # 打印信息，表示达特茅斯控制篮板
                    self.dartmouth_ball()  # 调用dartmouth_ball方法，表示达特茅斯控球
        else:
            print("Shot is good.")  # 打印信息，表示投篮命中
            self.add_points(0, 2)  # 调用add_points方法，为达特茅斯队加2分
            self.dartmouth_ball()  # 调用dartmouth_ball方法，表示达特茅斯控球

    def opponent_non_jumpshot(self) -> None:
        """Simulate opponents lay up or set shot."""  # 方法注释，模拟对手的上篮或定点投篮
        if self.z1 > 3:  # type: ignore  # 如果z1大于3
            print("Set shot.")  # 打印信息，表示定点投篮
        else:
            print("Lay up")  # 打印信息，表示上篮
        if 7 / self.defense * random.random() > 0.413:  # 如果7除以防守值乘以随机数大于0.413
            print("Shot is missed.")  # 打印信息，表示投篮未中
            if self.defense / 6 * random.random() > 0.5:  # 如果防守值除以6乘以随机数大于0.5
                print(self.opponent + " controls the rebound.")  # 打印信息，表示对手控制篮板
                if self.defense == 6:  # 如果防守值等于6
                    # 如果随机数大于0.75，表示球被抢断，对手易得分
                    if random.random() > 0.75:
                        # 打印信息，球被抢断，对手易得分
                        print("Ball stolen. Easy lay up for Dartmouth.")
                        # 调用 add_points 方法，给对手加分
                        self.add_points(1, 2)
                        # 调用 opponent_ball 方法，对手控球
                        self.opponent_ball()
                    else:
                        # 如果随机数大于0.5，表示对手进行非跳投
                        if random.random() > 0.5:
                            # 打印信息
                            print()
                            # 调用 opponent_non_jumpshot 方法，对手进行非跳投
                            self.opponent_non_jumpshot()
                        else:
                            # 打印信息，传球回给对手的后卫
                            print("Pass back to " + self.opponent + " guard.\n")
                            # 调用 opponent_ball 方法，对手控球
                            self.opponent_ball()
                else:
                    # 如果随机数大于0.5，表示对手进行非跳投
                    if random.random() > 0.5:
                        # 打印信息
                        print()
                        # 调用 opponent_non_jumpshot 方法，对手进行非跳投
                        self.opponent_non_jumpshot()
                    else:
                        # 打印信息，传球回给对手的后卫
                        print("Pass back to " + self.opponent + " guard\n")
                        # 调用 opponent_ball 方法，对手控球
                        self.opponent_ball()
            else:
                # 打印信息，达特茅斯控制篮板
                print("Dartmouth controls the rebound.\n")
        self.dartmouth_ball()
```
调用类中的dartmouth_ball()方法，表示达特茅斯队获得球权。

```
        else:
            print("Shot is good.")
            self.add_points(0, 2)
            self.dartmouth_ball()
```
否则，打印"Shot is good."，调用add_points()方法为达特茅斯队增加2分，并调用dartmouth_ball()方法表示达特茅斯队获得球权。

```
    def opponent_ball(self) -> None:
```
定义opponent_ball()方法，表示对手球权。

```
        """
        Simulate an opponents possesion

        Randomly picks jump shot or lay up / set shot.
        """
```
方法的文档字符串，解释了该方法的作用，即模拟对手的进攻，随机选择跳投或上篮/定点投篮。

```
        self.time += 1
        if self.time == 50:
            self.halftime()
```
时间加1，如果时间等于50，则调用halftime()方法表示中场休息。

```
        self.z1 = 10 / 4 * random.random() + 1
```
生成一个随机数z1，范围在1到3.5之间。

```
        if self.z1 > 2:
            self.opponent_non_jumpshot()
```
如果z1大于2，则调用opponent_non_jumpshot()方法表示对手进行非跳投。

```
        else:
            self.opponent_jumpshot()
```
否则，调用opponent_jumpshot()方法表示对手进行跳投。
def get_defense_choice(defense_choices: List[float]) -> float:
    """获取防守选择的输入"""
    try:
        defense = float(input())  # 尝试将输入转换为浮点数
    except ValueError:
        defense = None  # 如果输入不是有效的浮点数，则将防守选择设为None

    # 如果输入不是有效的防守选择，则重新获取输入
    while defense not in defense_choices:
        print("Your new defensive allignment is? ", end="")
        try:
            defense = float(input())  # 尝试将输入转换为浮点数
        except ValueError:
            continue  # 如果输入不是有效的浮点数，则继续循环
    assert isinstance(defense, float)  # 确保防守选择是浮点数类型
    return defense  # 返回防守选择
def get_dartmouth_ball_choice(shot_choices: List[Literal[0, 1, 2, 3, 4]]) -> int:
    # 打印提示信息，等待用户输入
    print("Your shot? ", end="")
    # 初始化变量 shot
    shot = None
    # 尝试将用户输入转换为整数，如果出现 ValueError 则将 shot 设为 None
    try:
        shot = int(input())
    except ValueError:
        shot = None

    # 当用户输入不在 shot_choices 中时，提示用户重新输入
    while shot not in shot_choices:
        print("Incorrect answer. Retype it. Your shot? ", end="")
        # 尝试将用户输入转换为整数，如果出现异常则继续循环
        try:
            shot = int(input())
        except Exception:
            continue
    # 断言 shot 是整数类型
    assert isinstance(shot, int)
    # 返回用户输入的 shot
    return shot


def get_opponents_name() -> str:
    """Take input for opponent's name"""
```
在这个示例中，我们为两个函数添加了注释。第一个函数是 get_dartmouth_ball_choice，它接受一个名为 shot_choices 的列表参数，并返回一个整数。函数中的注释解释了函数的作用以及每个语句的作用。第二个函数是 get_opponents_name，它没有具体的实现代码，只有一个简单的注释说明函数的作用。
    print("\nChoose your opponent? ", end="")  # 打印提示信息，要求用户选择对手
    return input()  # 接收用户输入的对手选择并返回

if __name__ == "__main__":
    Basketball()  # 如果作为主程序运行，则调用Basketball函数
```