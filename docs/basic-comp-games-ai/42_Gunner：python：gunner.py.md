# `d:/src/tocomm/basic-computer-games\42_Gunner\python\gunner.py`

```
#!/usr/bin/env python3
#
# Ported to Python by @iamtraction

from math import sin  # 导入 sin 函数
from random import random  # 导入 random 函数


def gunner() -> None:  # 定义 gunner 函数，返回空值
    gun_range = int(40000 * random() + 20000)  # 生成枪的射程范围

    print("\nMAXIMUM RANGE OF YOUR GUN IS", gun_range, "YARDS.")  # 打印枪的最大射程

    killed_enemies = 0  # 初始化击杀敌人数
    S1 = 0  # 初始化 S1 变量

    while True:  # 进入无限循环
        target_distance = int(gun_range * (0.1 + 0.8 * random()))  # 生成目标距离
        shots = 0  # 初始化射击次数
        print("\nDISTANCE TO THE TARGET IS", target_distance, "YARDS.")  # 打印目标距离的信息

        while True:  # 进入无限循环
            elevation = float(input("\n\nELEVATION? "))  # 获取用户输入的仰角

            if elevation > 89:  # 如果仰角大于89度
                print("MAXIMUM ELEVATION IS 89 DEGREES.")  # 打印最大仰角限制信息
                continue  # 继续下一次循环

            if elevation < 1:  # 如果仰角小于1度
                print("MINIMUM ELEVATION IS ONE DEGREE.")  # 打印最小仰角限制信息
                continue  # 继续下一次循环

            shots += 1  # 射击次数加一

            if shots < 6:  # 如果射击次数小于6
                B2 = 2 * elevation / 57.3  # 计算B2的值
                shot_impact = gun_range * sin(B2)  # 计算射击点的影响
                shot_proximity = target_distance - shot_impact  # 计算射击点与目标的距离
                shot_proximity_int = int(shot_proximity)  # 将射击点与目标的距离转换为整数
# 如果射击的距离小于100，打印目标被摧毁的信息以及消耗的弹药数量
if abs(shot_proximity_int) < 100:
    print(
        "*** TARGET DESTROYED *** ",
        shots,
        "ROUNDS OF AMMUNITION EXPENDED.",
    )
    # 增加总共消耗的弹药数量
    S1 += shots
    # 如果已经杀死了4个敌人，打印总共消耗的弹药数量并根据条件打印不同的信息
    if killed_enemies == 4:
        print("\n\nTOTAL ROUNDS EXPENDED WERE: ", S1)
        # 如果消耗的弹药数量大于18，打印需要返回训练的信息并结束程序
        if S1 > 18:
            print("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
            return
        # 如果消耗的弹药数量小于等于18，打印射击很好的信息并结束程序
        else:
            print("NICE SHOOTING !!")
            return
    # 如果还没有杀死4个敌人，增加已杀敌人数量并打印发现更多敌人的信息
    else:
        killed_enemies += 1
        print(
            "\nTHE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY..."
                        )
                        break
                else:
                    if shot_proximity_int > 100:
                        print("SHORT OF TARGET BY", abs(shot_proximity_int), "YARDS.")
                    else:
                        print("OVER TARGET BY", abs(shot_proximity_int), "YARDS.")
            else:
                print("\nBOOM !!!!   YOU HAVE JUST BEEN DESTROYED BY THE ENEMY.\n\n\n")
                print("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
                return


def main() -> None:
    print(" " * 33 + "GUNNER")  # 打印标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印创意计算的位置
    print("\n\n\n")  # 打印空行
    print("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN")  # 打印提示信息
    print("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE")  # 打印提示信息
    print("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS")  # 打印提示信息
    print("OF THE TARGET WILL DESTROY IT.")  # 打印字符串 "OF THE TARGET WILL DESTROY IT."

    while True:  # 进入无限循环
        gunner()  # 调用 gunner() 函数

        not_again = input("TRY AGAIN (Y OR N)? ").upper() != "Y"  # 获取用户输入并将其转换为大写，然后判断是否不等于 "Y"，将结果赋值给变量 not_again
        if not_again:  # 如果 not_again 为 True
            print("\nOK.  RETURN TO BASE CAMP.")  # 打印字符串 "\nOK.  RETURN TO BASE CAMP."
            break  # 退出循环

if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用 main() 函数
```