# `basic-computer-games\42_Gunner\python\gunner.py`

```

#!/usr/bin/env python3
#
# Ported to Python by @iamtraction

# 从 math 模块中导入 sin 函数
from math import sin
# 从 random 模块中导入 random 函数
from random import random

# 定义 gunner 函数，无返回值
def gunner() -> None:
    # 随机生成炮的射程范围
    gun_range = int(40000 * random() + 20000)

    # 打印炮的最大射程
    print("\nMAXIMUM RANGE OF YOUR GUN IS", gun_range, "YARDS.")

    # 初始化击毁敌人的数量和 S1
    killed_enemies = 0
    S1 = 0

    # 无限循环，直到手动退出
    while True:
        # 随机生成目标距离
        target_distance = int(gun_range * (0.1 + 0.8 * random()))
        shots = 0

        # 打印目标距离
        print("\nDISTANCE TO THE TARGET IS", target_distance, "YARDS.")

        # 无限循环，直到手动退出
        while True:
            # 输入炮的仰角
            elevation = float(input("\n\nELEVATION? "))

            # 如果仰角大于89度，提示最大仰角为89度
            if elevation > 89:
                print("MAXIMUM ELEVATION IS 89 DEGREES.")
                continue

            # 如果仰角小于1度，提示最小仰角为1度
            if elevation < 1:
                print("MINIMUM ELEVATION IS ONE DEGREE.")
                continue

            # 射击次数加一
            shots += 1

            # 如果射击次数小于6
            if shots < 6:
                # 计算弹道落点
                B2 = 2 * elevation / 57.3
                shot_impact = gun_range * sin(B2)
                shot_proximity = target_distance - shot_impact
                shot_proximity_int = int(shot_proximity)

                # 如果落点与目标距离的绝对值小于100码，打印目标被摧毁的信息
                if abs(shot_proximity_int) < 100:
                    print(
                        "*** TARGET DESTROYED *** ",
                        shots,
                        "ROUNDS OF AMMUNITION EXPENDED.",
                    )
                    S1 += shots
                    # 如果击毁敌人数量为4，打印总共消耗的弹药数量，并根据情况给出评价
                    if killed_enemies == 4:
                        print("\n\nTOTAL ROUNDS EXPENDED WERE: ", S1)
                        if S1 > 18:
                            print("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
                            return
                        else:
                            print("NICE SHOOTING !!")
                            return
                    else:
                        # 击毁敌人数量加一，提示发现更多敌人的活动
                        killed_enemies += 1
                        print(
                            "\nTHE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY..."
                        )
                        break
                else:
                    # 如果落点与目标距离的绝对值大于100码，提示落点偏离目标的距离
                    if shot_proximity_int > 100:
                        print("SHORT OF TARGET BY", abs(shot_proximity_int), "YARDS.")
                    else:
                        print("OVER TARGET BY", abs(shot_proximity_int), "YARDS.")
            else:
                # 如果射击次数大于等于6，提示被敌人摧毁
                print("\nBOOM !!!!   YOU HAVE JUST BEEN DESTROYED BY THE ENEMY.\n\n\n")
                print("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
                return

# 定义主函数 main，无返回值
def main() -> None:
    # 打印游戏标题和介绍
    print(" " * 33 + "GUNNER")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n\n")
    print("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN")
    print("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE")
    print("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS")
    print("OF THE TARGET WILL DESTROY IT.")

    # 无限循环，直到手动退出
    while True:
        # 调用 gunner 函数
        gunner()

        # 询问是否再次尝试，如果不是则退出循环
        not_again = input("TRY AGAIN (Y OR N)? ").upper() != "Y"
        if not_again:
            print("\nOK.  RETURN TO BASE CAMP.")
            break

# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()

```