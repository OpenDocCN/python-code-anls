# `12_Bombs_Away\python\bombs_away.py`

```
"""
Bombs away

Ported from BASIC to Python3 by Bernard Cooke (bernardcooke53)
Tested with Python 3.8.10, formatted with Black and type checked with mypy.
"""
# 导入随机模块
import random
# 导入类型提示模块
from typing import Iterable


# 定义函数 _stdin_choice，接受一个字符串类型的参数 prompt 和一个可迭代对象类型的参数 choices，返回一个字符串类型的值
def _stdin_choice(prompt: str, *, choices: Iterable[str]) -> str:
    # 从标准输入中获取用户输入的值
    ret = input(prompt)
    # 当用户输入的值不在 choices 中时，提示用户重新输入
    while ret not in choices:
        print("TRY AGAIN...")
        ret = input(prompt)
    return ret


# 定义函数 player_survived，不接受任何参数，没有返回值
def player_survived() -> None:
    # 打印消息
    print("YOU MADE IT THROUGH TREMENDOUS FLAK!!")
def player_death() -> None:
    # 打印玩家死亡信息
    print("* * * * BOOM * * * *")
    print("YOU HAVE BEEN SHOT DOWN.....")
    print("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR")
    print("LAST TRIBUTE...")


def mission_success() -> None:
    # 打印任务成功信息和击杀数量
    print(f"DIRECT HIT!!!! {int(100 * random.random())} KILLED.")
    print("MISSION SUCCESSFUL.")


def death_with_chance(p_death: float) -> bool:
    """
    Takes a float between 0 and 1 and returns a boolean
    if the player has survived (based on random chance)

    Returns True if death, False if survived
    """
    """
    return p_death > random.random()  # 返回一个布尔值，表示是否发生死亡事件

def commence_non_kamikazi_attack() -> None:  # 定义一个没有返回值的函数
    while True:  # 进入无限循环
        try:  # 尝试执行以下代码
            nmissions = int(input("HOW MANY MISSIONS HAVE YOU FLOWN? "))  # 获取用户输入的飞行任务次数并转换为整数

            while nmissions >= 160:  # 当飞行任务次数大于等于160时
                print("MISSIONS, NOT MILES...")  # 打印提示信息
                print("150 MISSIONS IS HIGH EVEN FOR OLD-TIMERS")  # 打印提示信息
                nmissions = int(input("NOW THEN, HOW MANY MISSIONS HAVE YOU FLOWN? "))  # 获取用户输入的飞行任务次数并转换为整数
            break  # 跳出循环
        except ValueError:  # 如果出现值错误
            # In the BASIC implementation this
            # wasn't accounted for
            print("TRY AGAIN...")  # 打印提示信息
            continue  # 继续下一次循环
    if nmissions >= 100:  # 如果任务数量大于等于100
        print("THAT'S PUSHING THE ODDS!")  # 打印“THAT'S PUSHING THE ODDS!”

    if nmissions < 25:  # 如果任务数量小于25
        print("FRESH OUT OF TRAINING, EH?")  # 打印“FRESH OUT OF TRAINING, EH?”

    print()  # 打印空行
    return (  # 返回一个表达式的结果
        mission_success() if nmissions >= 160 * random.random() else mission_failure()  # 如果nmissions大于等于160乘以一个随机数的结果，则返回mission_success()，否则返回mission_failure()
    )


def mission_failure() -> None:  # 定义一个返回类型为None的函数mission_failure
    weapons_choices = {  # 创建一个名为weapons_choices的字典
        "1": "GUNS",  # 键为"1"，值为"GUNS"
        "2": "MISSILES",  # 键为"2"，值为"MISSILES"
        "3": "BOTH",  # 键为"3"，值为"BOTH"
    }
    print(f"MISSED TARGET BY {int(2 + 30 * random.random())} MILES!")  # 打印“MISSED TARGET BY x MILES!”，其中x为一个2到32之间的随机整数
    print("NOW YOU'RE REALLY IN FOR IT !!")  # 打印“NOW YOU'RE REALLY IN FOR IT !!”
    # 打印空行
    print()
    # 从标准输入中获取敌人的武器选择
    enemy_weapons = _stdin_choice(
        prompt="DOES THE ENEMY HAVE GUNS(1), MISSILES(2), OR BOTH(3)? ",
        choices=weapons_choices,
    )

    # 如果没有枪手（即武器选择为2），则我们认为枪手的准确度为0，用于计算玩家死亡的概率
    enemy_gunner_accuracy = 0.0
    if enemy_weapons != "2":
        # 如果敌人有枪支，枪手的准确度是多少？
        while True:
            try:
                # 获取敌人枪手的命中率
                enemy_gunner_accuracy = float(
                    input("WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS (10 TO 50)? ")
                )
                break
            except ValueError:
                # 在基本实现中没有考虑到这一点
                print("再试一次...")
                continue

        if enemy_gunner_accuracy < 10:
            print("你撒谎了，但你会付出代价...")
            return player_death()

    missile_threat_weighting = 0 if enemy_weapons == "1" else 35

    death = death_with_chance(
        p_death=(enemy_gunner_accuracy + missile_threat_weighting) / 100
    )

    return player_survived() if not death else player_death()


def play_italy() -> None:
    targets_to_messages = {
        # 1 - ALBANIA, 2 - GREECE, 3 - NORTH AFRICA
        # 定义一个字典，将数字对应的目标地点与相应的消息关联起来
        "1": "SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE.",
        "2": "BE CAREFUL!!!",
        "3": "YOU'RE GOING FOR THE OIL, EH?",
    }
    # 从用户输入中选择目标地点
    target = _stdin_choice(
        prompt="YOUR TARGET -- ALBANIA(1), GREECE(2), NORTH AFRICA(3)",
        choices=targets_to_messages,
    )

    # 打印选择的目标地点对应的消息
    print(targets_to_messages[target])
    # 开始非神风式的攻击
    return commence_non_kamikazi_attack()


def play_allies() -> None:
    # 定义一个字典，将飞机编号与相应的消息关联起来
    aircraft_to_message = {
        "1": "YOU'VE GOT 2 TONS OF BOMBS FLYING FOR PLOESTI.",
        "2": "YOU'RE DUMPING THE A-BOMB ON HIROSHIMA.",
        "3": "YOU'RE CHASING THE BISMARK IN THE NORTH SEA.",
        "4": "YOU'RE BUSTING A GERMAN HEAVY WATER PLANT IN THE RUHR.",
    }
    # 从标准输入中选择飞机类型
    aircraft = _stdin_choice(
        prompt="AIRCRAFT -- LIBERATOR(1), B-29(2), B-17(3), LANCASTER(4): ",
        choices=aircraft_to_message,
    )

    # 打印所选飞机的信息
    print(aircraft_to_message[aircraft])
    # 返回执行非神风特攻的结果
    return commence_non_kamikazi_attack()


def play_japan() -> None:
    # 打印飞行神风特攻任务的信息
    print("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.")
    # 询问是否是第一次执行神风特攻任务
    first_mission = input("YOUR FIRST KAMIKAZE MISSION? (Y OR N): ")
    # 如果不是第一次执行神风特攻任务，则返回玩家死亡
    if first_mission.lower() == "n":
        return player_death()
    # 如果随机数大于0.65，则返回任务成功，否则返回玩家死亡
    return mission_success() if random.random() > 0.65 else player_death()


def play_germany() -> None:
    # 创建目标到信息的字典
    targets_to_messages = {
        # 1 - RUSSIA, 2 - ENGLAND, 3 - FRANCE
        "1": "YOU'RE NEARING STALINGRAD.",
        "2": "NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR.",
        "3": "NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS.",
    }
    # 定义一个字典，键为1、2、3，值为对应国家的提示信息

    target = _stdin_choice(
        prompt="A NAZI, EH?  OH WELL.  ARE YOU GOING FOR RUSSIA(1),\nENGLAND(2), OR FRANCE(3)? ",
        choices=targets_to_messages,
    )
    # 从用户输入中选择目标国家，根据用户输入在targets_to_messages字典中查找对应的提示信息

    print(targets_to_messages[target])
    # 打印选择的目标国家对应的提示信息

    return commence_non_kamikazi_attack()
    # 返回执行非神风攻击的结果


def play_game() -> None:
    print("YOU ARE A PILOT IN A WORLD WAR II BOMBER.")
    # 打印游戏开始的提示信息
    sides = {"1": play_italy, "2": play_allies, "3": play_japan, "4": play_germany}
    # 定义一个字典，键为1、2、3、4，值为对应国家的游戏函数

    side = _stdin_choice(
        prompt="WHAT SIDE -- ITALY(1), ALLIES(2), JAPAN(3), GERMANY(4): ", choices=sides
    )
    # 从用户输入中选择游戏国家，根据用户输入在sides字典中查找对应的游戏函数
    )  # 结束 if 语句的条件判断
    return sides[side]()  # 返回 sides 字典中对应 side 键的值，并调用该值（假设它是一个函数）

if __name__ == "__main__":
    again = True  # 初始化变量 again 为 True
    while again:  # 当 again 为 True 时执行循环
        play_game()  # 调用 play_game 函数
        again = input("ANOTHER MISSION? (Y OR N): ").upper() == "Y"  # 将用户输入的值转换为大写后与 "Y" 比较，将结果赋给 again
```