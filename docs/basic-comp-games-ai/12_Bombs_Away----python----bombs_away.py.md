# `basic-computer-games\12_Bombs_Away\python\bombs_away.py`

```
"""
Bombs away

Ported from BASIC to Python3 by Bernard Cooke (bernardcooke53)
Tested with Python 3.8.10, formatted with Black and type checked with mypy.
"""
# 导入 random 模块
import random
# 导入 Iterable 类型
from typing import Iterable

# 定义函数 _stdin_choice，接收提示字符串和选择列表，返回用户输入的选择
def _stdin_choice(prompt: str, *, choices: Iterable[str]) -> str:
    ret = input(prompt)
    while ret not in choices:
        print("TRY AGAIN...")
        ret = input(prompt)
    return ret

# 定义函数 player_survived，打印玩家幸存消息
def player_survived() -> None:
    print("YOU MADE IT THROUGH TREMENDOUS FLAK!!")

# 定义函数 player_death，打印玩家死亡消息
def player_death() -> None:
    print("* * * * BOOM * * * *")
    print("YOU HAVE BEEN SHOT DOWN.....")
    print("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR")
    print("LAST TRIBUTE...")

# 定义函数 mission_success，打印任务成功消息
def mission_success() -> None:
    print(f"DIRECT HIT!!!! {int(100 * random.random())} KILLED.")
    print("MISSION SUCCESSFUL.")

# 定义函数 death_with_chance，接收死亡概率，返回玩家是否幸存
def death_with_chance(p_death: float) -> bool:
    """
    Takes a float between 0 and 1 and returns a boolean
    if the player has survived (based on random chance)

    Returns True if death, False if survived
    """
    return p_death > random.random()

# 定义函数 commence_non_kamikazi_attack，处理非神风攻击
def commence_non_kamikazi_attack() -> None:
    while True:
        try:
            nmissions = int(input("HOW MANY MISSIONS HAVE YOU FLOWN? "))

            while nmissions >= 160:
                print("MISSIONS, NOT MILES...")
                print("150 MISSIONS IS HIGH EVEN FOR OLD-TIMERS")
                nmissions = int(input("NOW THEN, HOW MANY MISSIONS HAVE YOU FLOWN? "))
            break
        except ValueError:
            # In the BASIC implementation this
            # wasn't accounted for
            print("TRY AGAIN...")
            continue

    if nmissions >= 100:
        print("THAT'S PUSHING THE ODDS!")

    if nmissions < 25:
        print("FRESH OUT OF TRAINING, EH?")

    print()
    return (
        mission_success() if nmissions >= 160 * random.random() else mission_failure()
    )

# 定义函数 mission_failure，处理任务失败
def mission_failure() -> None:
    # 定义敌人的武器选择字典
    weapons_choices = {
        "1": "GUNS",
        "2": "MISSILES",
        "3": "BOTH",
    }
    # 打印未命中目标的距离
    print(f"MISSED TARGET BY {int(2 + 30 * random.random())} MILES!")
    # 打印警告信息
    print("NOW YOU'RE REALLY IN FOR IT !!")
    print()
    # 获取用户输入的敌人武器选择
    enemy_weapons = _stdin_choice(
        prompt="DOES THE ENEMY HAVE GUNS(1), MISSILES(2), OR BOTH(3)? ",
        choices=weapons_choices,
    )
    
    # 如果没有炮手（即武器选择为2），则假设炮手的准确度为0，用于计算玩家死亡的概率
    enemy_gunner_accuracy = 0.0
    if enemy_weapons != "2":
        # 如果敌人有枪支，询问炮手的准确度
        while True:
            try:
                enemy_gunner_accuracy = float(
                    input("WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS (10 TO 50)? ")
                )
                break
            except ValueError:
                # 在基本实现中没有考虑到这一点
                print("TRY AGAIN...")
                continue
    
        if enemy_gunner_accuracy < 10:
            print("YOU LIE, BUT YOU'LL PAY...")
            return player_death()
    
    # 如果敌人只有枪支，则导弹威胁权重为0，否则为35
    missile_threat_weighting = 0 if enemy_weapons == "1" else 35
    
    # 根据死亡概率计算玩家是否死亡
    death = death_with_chance(
        p_death=(enemy_gunner_accuracy + missile_threat_weighting) / 100
    )
    
    # 如果未死亡，则返回玩家存活，否则返回玩家死亡
    return player_survived() if not death else player_death()
# 定义玩意大利任务的函数
def play_italy() -> None:
    # 创建目标到消息的字典
    targets_to_messages = {
        # 1 - ALBANIA, 2 - GREECE, 3 - NORTH AFRICA
        "1": "SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE.",
        "2": "BE CAREFUL!!!",
        "3": "YOU'RE GOING FOR THE OIL, EH?",
    }
    # 从用户输入中选择目标
    target = _stdin_choice(
        prompt="YOUR TARGET -- ALBANIA(1), GREECE(2), NORTH AFRICA(3)",
        choices=targets_to_messages,
    )

    # 打印选择的目标对应的消息
    print(targets_to_messages[target])
    # 返回执行非神风式攻击的结果
    return commence_non_kamikazi_attack()


# 定义玩盟军任务的函数
def play_allies() -> None:
    # 创建飞机到消息的字典
    aircraft_to_message = {
        "1": "YOU'VE GOT 2 TONS OF BOMBS FLYING FOR PLOESTI.",
        "2": "YOU'RE DUMPING THE A-BOMB ON HIROSHIMA.",
        "3": "YOU'RE CHASING THE BISMARK IN THE NORTH SEA.",
        "4": "YOU'RE BUSTING A GERMAN HEAVY WATER PLANT IN THE RUHR.",
    }
    # 从用户输入中选择飞机
    aircraft = _stdin_choice(
        prompt="AIRCRAFT -- LIBERATOR(1), B-29(2), B-17(3), LANCASTER(4): ",
        choices=aircraft_to_message,
    )

    # 打印选择的飞机对应的消息
    print(aircraft_to_message[aircraft])
    # 返回执行非神风式攻击的结果
    return commence_non_kamikazi_attack()


# 定义玩日本任务的函数
def play_japan() -> None:
    # 打印玩家正在执行的任务
    print("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.")
    # 获取玩家是否执行第一个神风式任务的输入
    first_mission = input("YOUR FIRST KAMIKAZE MISSION? (Y OR N): ")
    # 如果玩家选择不执行第一个神风式任务，则返回玩家死亡
    if first_mission.lower() == "n":
        return player_death()
    # 如果随机数大于0.65，则返回任务成功，否则返回玩家死亡
    return mission_success() if random.random() > 0.65 else player_death()


# 定义玩德国任务的函数
def play_germany() -> None:
    # 创建目标到消息的字典
    targets_to_messages = {
        # 1 - RUSSIA, 2 - ENGLAND, 3 - FRANCE
        "1": "YOU'RE NEARING STALINGRAD.",
        "2": "NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR.",
        "3": "NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS.",
    }
    # 从用户输入中选择目标
    target = _stdin_choice(
        prompt="A NAZI, EH?  OH WELL.  ARE YOU GOING FOR RUSSIA(1),\nENGLAND(2), OR FRANCE(3)? ",
        choices=targets_to_messages,
    )

    # 打印选择的目标对应的消息
    print(targets_to_messages[target])

    # 返回执行非神风式攻击的结果
    return commence_non_kamikazi_attack()


# 定义玩游戏的函数
def play_game() -> None:
    # 打印游戏背景
    print("YOU ARE A PILOT IN A WORLD WAR II BOMBER.")
    # 创建一个包含不同国家选项的字典，键为数字，值为对应的函数
    sides = {"1": play_italy, "2": play_allies, "3": play_japan, "4": play_germany}
    # 从标准输入中获取用户选择的国家，限定选择范围为字典中的键，然后调用对应的函数
    side = _stdin_choice(
        prompt="WHAT SIDE -- ITALY(1), ALLIES(2), JAPAN(3), GERMANY(4): ", choices=sides
    )
    # 返回用户选择的国家对应的函数的执行结果
    return sides[side]()
# 如果当前模块被直接执行，则执行以下代码
if __name__ == "__main__":
    # 设置一个标志变量，用于控制循环
    again = True
    # 当标志变量为真时，执行循环
    while again:
        # 调用 play_game() 函数来进行游戏
        play_game()
        # 询问用户是否要再次进行任务，并将用户输入转换为大写字母
        again = input("ANOTHER MISSION? (Y OR N): ").upper() == "Y"
```