# `91_Train\python\train.py`

```
    # 设置汽车速度为一个40到65之间的随机整数
    car_speed = random.randint(40, 65)
    # 设置时间差为一个5到20之间的随机整数
    time_difference = random.randint(5, 20)
    # 设置火车速度为一个20到39之间的随机整数
    train_speed = random.randint(20, 39)
    # 打印汽车以car_speed速度行驶可以在time_difference小时内完成一次旅行
    print("\nA car travelling", car_speed, "MPH can make a certain trip in")
    # 打印时间差为time_difference小时，火车以train_speed速度行驶
    print(time_difference, "hours less than a train travelling at", train_speed, "MPH")
    # 设置时间答案为0
    time_answer: float = 0
    # 当时间答案为0时，循环直到输入正确的时间
    while time_answer == 0:
        try:
            # 尝试获取用户输入的时间答案
            time_answer = float(input("How long does the trip take by car "))
        except ValueError:
            print("Please enter a number.")  # 打印提示信息，要求用户输入一个数字
    car_time = time_difference * train_speed / (car_speed - train_speed)  # 计算汽车行驶时间
    error_percent = int(abs((car_time - time_answer) * 100 / time_answer) + 0.5)  # 计算误差百分比
    if error_percent > 5:  # 如果误差百分比大于5
        print("Sorry. You were off by", error_percent, "percent.")  # 打印提示信息，显示误差百分比
        print("Correct answer is", round(car_time, 6), "hours")  # 打印正确答案
    else:  # 如果误差百分比小于等于5
        print("Good! Answer within", error_percent, "percent.")  # 打印提示信息，显示误差百分比


def main() -> None:
    print(" " * 33 + "TRAIN")  # 打印标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")  # 打印创意计算的地点信息
    print("Time - speed distance exercise")  # 打印练习题目

    keep_playing = True  # 初始化变量，用于控制是否继续游戏
    while keep_playing:  # 当继续游戏时
        play_game()  # 调用play_game函数进行游戏
        keep_playing = input("\nAnother problem (yes or no) ").lower().startswith("y")  # 获取用户输入，判断是否继续游戏
# 如果当前脚本被直接执行而不是被导入，则执行main函数
if __name__ == "__main__":
    main()
```